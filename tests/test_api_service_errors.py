"""HTTP service-error boundary tests (Task 1C).

Covers the shared server-error envelope for explicit ``HTTPException`` 5xx
responses (status/CORS/``Retry-After`` preserved, spoofed references ignored),
unchanged non-5xx responses, and bounded internal diagnostics that cannot leak
request bodies, authorization headers, captured locals, or hostile exception
strings.

The production adapter (``pyscnomics.api.router.get_grosssplit``) is monkeypatched
so every probe exercises the real FastAPI app, exception handlers, middleware,
and CORS stack.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Protocol

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from typing_extensions import override

from pyscnomics.api.main import app
from pyscnomics.contracts.gross_split import GrossSplitException
from pyscnomics.input_validation import InputValidationError, ValidationIssue
from tests._validation_api_support import response_body

_ORIGIN = "http://127.0.0.1:5173"
_ERROR_LOGGER = "pyscnomics.api.errors"
_LOG_BOUND = 100_000


class _RawResponse(Protocol):
    status_code: int
    headers: Mapping[str, str]
    text: str
    content: bytes

    def json(self) -> object: ...


class _RawClient(Protocol):
    def post(
        self,
        url: str,
        *,
        json: object,
        headers: Mapping[str, str] | None = ...,
    ) -> _RawResponse: ...


_raw_client: _RawClient = TestClient(app, raise_server_exceptions=False)


def _clean_payload() -> dict[str, object]:
    return {
        "setup": {
            "start_date": "01/01/2019",
            "end_date": "31/12/2050",
            "oil_onstream_date": "01/01/2029",
            "gas_onstream_date": "01/01/2029",
            "approval_year": 2024,
            "is_pod_1": False,
            "is_strict": True,
        },
        "summary_arguments": {},
        "contract_arguments": {},
        "grosssplit": {},
    }


class _HostileError(Exception):
    """Exception whose ``str``/``repr`` raise, to exercise hostile formatting."""

    @override
    def __str__(self) -> str:
        raise RuntimeError("hostile __str__")

    @override
    def __repr__(self) -> str:
        raise RuntimeError("hostile __repr__")


def _install_raiser(monkeypatch: pytest.MonkeyPatch, exc: BaseException) -> None:
    """Make the production adapter raise ``exc`` for the next requests."""
    import pyscnomics.api.router as router

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise exc

    monkeypatch.setattr(router, "get_grosssplit", _boom)


# ── Explicit HTTPException 5xx uses the safe shared envelope ─────────


def test_http_exception_500_uses_safe_server_error_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_raiser(
        monkeypatch, HTTPException(status_code=500, detail="SECRET-INTERNAL-DETAIL")
    )
    resp = _raw_client.post(
        "/api/grosssplit",
        json=_clean_payload(),
        headers={"Origin": _ORIGIN, "X-Request-ID": "spoofed-value"},
    )
    assert resp.status_code == 500
    body = response_body(resp)
    assert body["kind"] == "server_error"
    assert body["validation_complete"] is False
    assert body["detail"] == "Internal Server Error"
    assert body["request_id"] != "spoofed-value"
    assert resp.headers["X-Request-ID"] == body["request_id"]
    assert resp.headers["access-control-allow-origin"] == _ORIGIN
    assert "SECRET-INTERNAL-DETAIL" not in resp.text
    assert "Traceback" not in resp.text
    assert "can_calculate" not in body


def test_http_exception_503_preserves_status_cors_and_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_raiser(
        monkeypatch,
        HTTPException(
            status_code=503,
            detail="UPSTREAM-BOOM",
            headers={
                "x-request-id": "upstream-lower",
                "Retry-After": "30",
                "Content-Type": "text/plain",
                "Access-Control-Allow-Origin": "*",
            },
        ),
    )
    resp = _raw_client.post(
        "/api/grosssplit", json=_clean_payload(), headers={"Origin": _ORIGIN}
    )
    assert resp.status_code == 503
    body = response_body(resp)
    # The reference header is exactly the server-generated body reference; the
    # upstream value must be discarded entirely (no "a, b" comma joining).
    assert resp.headers["x-request-id"] == body["request_id"]
    assert "," not in resp.headers["x-request-id"]
    assert "upstream-lower" not in resp.headers["x-request-id"]
    # Regenerated JSON owns its content type; the upstream text/plain is dropped.
    assert resp.headers["content-type"].startswith("application/json")
    # Retry-After is a useful standard header and is retained.
    assert resp.headers["Retry-After"] == "30"
    # The allowed-origin policy is owned by CORSMiddleware, not reflected.
    assert resp.headers["access-control-allow-origin"] == _ORIGIN
    assert body["kind"] == "server_error"
    assert body["detail"] == "Service Unavailable"
    assert body["request_id"] != "upstream-lower"
    assert "currently unavailable" in str(body["message"])
    assert "UPSTREAM-BOOM" not in resp.text
    assert "Traceback" not in resp.text


def test_http_exception_5xx_does_not_reflect_disallowed_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_raiser(
        monkeypatch,
        HTTPException(
            status_code=503,
            detail="UPSTREAM-BOOM",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": "true",
                "X-REQUEST-ID": "upstream-mixed",
            },
        ),
    )
    resp = _raw_client.post(
        "/api/grosssplit",
        json=_clean_payload(),
        headers={"Origin": "http://evil.example"},
    )
    assert resp.status_code == 503
    body = response_body(resp)
    # A disallowed origin must not receive an allow-origin header at all; the
    # upstream "*" must not be reflected by our header copy.
    assert "access-control-allow-origin" not in resp.headers
    assert resp.headers["x-request-id"] == body["request_id"]
    assert "upstream-mixed" not in resp.headers["x-request-id"]


def test_http_exception_5xx_drops_body_and_hop_by_hop_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_raiser(
        monkeypatch,
        HTTPException(
            status_code=500,
            detail="boom",
            headers={
                "Content-Length": "999999",
                "Content-Type": "text/plain",
                "Transfer-Encoding": "chunked",
                "Connection": "close",
                "Retry-After": "12",
            },
        ),
    )
    resp = _raw_client.post("/api/grosssplit", json=_clean_payload())
    assert resp.status_code == 500
    assert resp.headers["content-type"].startswith("application/json")
    # The regenerated JSON length wins over the upstream Content-Length.
    assert resp.headers["content-length"] == str(len(resp.content))
    assert "999999" not in resp.headers["content-length"]
    assert resp.headers["Retry-After"] == "12"


def test_http_exception_5xx_drops_stale_content_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An upstream Content-Encoding would make HTTP clients try to decompress the
    # regenerated plain JSON body and fail before the reference can be read.
    _install_raiser(
        monkeypatch,
        HTTPException(
            status_code=503,
            detail="UPSTREAM-BOOM",
            headers={"Content-Encoding": "gzip", "Retry-After": "30"},
        ),
    )
    resp = _raw_client.post(
        "/api/grosssplit", json=_clean_payload(), headers={"Origin": _ORIGIN}
    )
    assert resp.status_code == 503
    assert "content-encoding" not in resp.headers
    assert resp.headers["Retry-After"] == "30"
    # Reading the body must succeed: httpx would raise DecodingError if the
    # stale gzip header survived on the uncompressed JSON.
    body = response_body(resp)
    assert body["kind"] == "server_error"
    assert resp.headers["x-request-id"] == body["request_id"]


@pytest.mark.parametrize("encoding", ["br", "deflate", "zstd", "identity"])
def test_http_exception_5xx_drops_any_content_encoding(
    monkeypatch: pytest.MonkeyPatch, encoding: str
) -> None:
    _install_raiser(
        monkeypatch,
        HTTPException(
            status_code=500,
            detail="boom",
            headers={"content-encoding": encoding},
        ),
    )
    resp = _raw_client.post("/api/grosssplit", json=_clean_payload())
    assert resp.status_code == 500
    assert "content-encoding" not in resp.headers
    assert response_body(resp)["kind"] == "server_error"


def test_http_exception_5xx_drops_stale_entity_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_raiser(
        monkeypatch,
        HTTPException(
            status_code=503,
            detail="boom",
            headers={
                "Content-Encoding": "gzip",
                "Content-Range": "bytes 0-10/11",
                "Content-MD5": "Q2hlY2sgSW50ZWdyaXR5IQ==",
                "Digest": "sha-256=abc",
                "Content-Language": "en",
                "Content-Location": "/old",
                "Content-Disposition": "attachment",
                "ETag": '"old-body"',
                "Last-Modified": "Wed, 21 Oct 2015 07:28:00 GMT",
                "Retry-After": "30",
            },
        ),
    )
    resp = _raw_client.post("/api/grosssplit", json=_clean_payload())
    assert resp.status_code == 503
    for header in (
        "content-encoding",
        "content-range",
        "content-md5",
        "digest",
        "content-language",
        "content-location",
        "content-disposition",
        "etag",
        "last-modified",
    ):
        assert header not in resp.headers, header
    assert resp.headers["Retry-After"] == "30"
    body = response_body(resp)
    assert body["kind"] == "server_error"
    assert resp.headers["x-request-id"] == body["request_id"]


def test_http_exception_5xx_does_not_reflect_credential_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_raiser(
        monkeypatch,
        HTTPException(
            status_code=503,
            detail="boom",
            headers={
                "Authorization": "Bearer upstream-secret",
                "Proxy-Authorization": "Basic dXNlcjpwYXNz",
                "X-Api-Key": "upstream-key",
            },
        ),
    )
    resp = _raw_client.post("/api/grosssplit", json=_clean_payload())
    assert resp.status_code == 503
    assert "authorization" not in resp.headers
    assert "proxy-authorization" not in resp.headers
    assert "x-api-key" not in resp.headers
    assert "upstream-secret" not in resp.text
    assert "upstream-key" not in resp.text
    assert response_body(resp)["kind"] == "server_error"


def test_http_exception_500_ignores_spoofed_reference_without_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_raiser(monkeypatch, HTTPException(status_code=500, detail="boom"))
    resp = _raw_client.post(
        "/api/grosssplit",
        json=_clean_payload(),
        headers={"X-Request-ID": "attacker-supplied"},
    )
    body = response_body(resp)
    assert body["request_id"] != "attacker-supplied"
    assert resp.headers["X-Request-ID"] == body["request_id"]


# ── Non-5xx responses stay standard and unchanged ────────────────────


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409])
def test_non_5xx_http_exception_is_unchanged(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    _install_raiser(
        monkeypatch, HTTPException(status_code=status, detail="standard detail")
    )
    resp = _raw_client.post("/api/grosssplit", json=_clean_payload())
    assert resp.status_code == status
    body = response_body(resp)
    assert body == {"detail": "standard detail"}
    assert "schema_version" not in body


def test_missing_route_stays_plain_404() -> None:
    resp = _raw_client.post("/api/no-such-endpoint", json={})
    assert resp.status_code == 404
    assert "schema_version" not in str(resp.json())


@pytest.mark.parametrize("status", [204, 304])
def test_bodyless_http_exception_keeps_empty_body(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    _install_raiser(
        monkeypatch,
        HTTPException(status_code=status, headers={"ETag": '"abc"'}),
    )
    resp = _raw_client.post("/api/grosssplit", json=_clean_payload())
    assert resp.status_code == status
    assert resp.content == b""
    assert resp.headers.get("ETag") == '"abc"'


def test_http_exception_401_preserves_www_authenticate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_raiser(
        monkeypatch,
        HTTPException(
            status_code=401,
            detail="nope",
            headers={"WWW-Authenticate": "Bearer"},
        ),
    )
    resp = _raw_client.post("/api/grosssplit", json=_clean_payload())
    assert resp.status_code == 401
    assert resp.headers["WWW-Authenticate"] == "Bearer"
    assert response_body(resp) == {"detail": "nope"}


# ── Domain exception classification is not broadened ─────────────────


def test_custom_grosssplit_exception_stays_internal_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_raiser(monkeypatch, GrossSplitException("engine-internal-secret"))
    resp = _raw_client.post("/api/grosssplit", json=_clean_payload())
    assert resp.status_code == 500
    body = response_body(resp)
    assert body["kind"] == "server_error"
    assert "engine-internal-secret" not in resp.text


def test_input_validation_error_stays_structured_422(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    issue = ValidationIssue(
        code="X",
        severity="error",
        message="Bad input.",
        guidance="Fix it.",
    )
    _install_raiser(monkeypatch, InputValidationError((issue,)))
    resp = _raw_client.post("/api/grosssplit", json=_clean_payload())
    assert resp.status_code == 422
    body = response_body(resp)
    assert body["kind"] == "input_validation"
    assert str(body["detail"]).startswith("PyscErr:")


# ── Bounded, leak-safe diagnostics ───────────────────────────────────


def test_huge_exception_value_log_is_bounded(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "AUTHORIZATION-SECRET-VALUE"
    _install_raiser(monkeypatch, RuntimeError("H" * 5_000_000))
    with caplog.at_level(logging.ERROR, logger=_ERROR_LOGGER):
        resp = _raw_client.post(
            "/api/grosssplit",
            json=_clean_payload(),
            headers={"Authorization": f"Bearer {secret}"},
        )
    assert resp.status_code == 500
    assert response_body(resp)["kind"] == "server_error"
    assert any(record.name == _ERROR_LOGGER for record in caplog.records)
    assert len(caplog.text) < _LOG_BOUND
    assert "H" * 10_000 not in caplog.text
    assert secret not in caplog.text


def test_diagnostics_do_not_capture_request_locals(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import pyscnomics.api.router as router

    secret = "LOCAL-PAYLOAD-SECRET"

    def _boom(*_args: object, **_kwargs: object) -> object:
        local_payload = {"token": secret}
        raise RuntimeError(f"boom {type(local_payload).__name__}")

    monkeypatch.setattr(router, "get_grosssplit", _boom)
    with caplog.at_level(logging.ERROR, logger=_ERROR_LOGGER):
        resp = _raw_client.post("/api/grosssplit", json=_clean_payload())
    assert resp.status_code == 500
    assert secret not in caplog.text


def test_hostile_str_exception_returns_json_500_reference(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _install_raiser(monkeypatch, _HostileError())
    with caplog.at_level(logging.ERROR, logger=_ERROR_LOGGER):
        resp = _raw_client.post(
            "/api/grosssplit", json=_clean_payload(), headers={"Origin": _ORIGIN}
        )
    assert resp.status_code == 500
    body = response_body(resp)
    assert body["kind"] == "server_error"
    assert body["request_id"]
    assert resp.headers["access-control-allow-origin"] == _ORIGIN
    assert resp.headers["X-Request-ID"] == body["request_id"]
    assert json.dumps(body)
