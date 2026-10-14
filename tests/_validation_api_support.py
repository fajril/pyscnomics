"""Typed TestClient helpers shared by the input-validation API tests.

Mirrors the Protocol style already used in ``tests/test_api.py`` so the new
tests stay free of unknown-type warnings under BasedPyright.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, TypeGuard

from fastapi.testclient import TestClient
from pydantic import TypeAdapter, ValidationError

from pyscnomics.api.main import app


class _Response(Protocol):
    status_code: int
    headers: Mapping[str, str]

    def json(self) -> object: ...


class _ApiClient(Protocol):
    def post(
        self,
        url: str,
        *,
        json: object,
        headers: Mapping[str, str] | None = ...,
    ) -> _Response: ...


client: _ApiClient = TestClient(app, raise_server_exceptions=False)

_STRING_RECORD = TypeAdapter(dict[str, object])
_RECORD_LIST = TypeAdapter(list[dict[str, object]])
_PATH_LIST = TypeAdapter(list[str | int])


def is_record(value: object) -> TypeGuard[dict[str, object]]:
    try:
        _ = _STRING_RECORD.validate_python(value, strict=True)
    except ValidationError:
        return False
    return True


def as_records(value: object) -> list[dict[str, object]]:
    """Validate a JSON array of objects, raising on malformed input."""
    return _RECORD_LIST.validate_python(value)


def as_path(value: object) -> list[str | int]:
    """Validate an issue-location path array."""
    return _PATH_LIST.validate_python(value)


def response_body(response: _Response) -> dict[str, object]:
    body = response.json()
    assert is_record(body)
    return body
