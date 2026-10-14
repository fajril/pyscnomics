"""Validation-only endpoint and direct-protection tests (Task 5).

Every assertion here goes through the real FastAPI ``TestClient`` and the real
request models. No fabricated request shapes or monkeypatched success paths are
used to prove protection: the expensive worker entry points are patched only to
prove they are *not* called when input is unsafe.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Protocol, cast

import pytest
from fastapi.testclient import TestClient

from pyscnomics.api.main import app
from tests._preflight_fixtures import (
    combine_request,
    corrected_payload,
    strict_transition_two_block_payload,
    transition_combinations,
    transition_payload_for,
)
from tests._validation_api_support import as_path, as_records, client, response_body

_WORKER_NAMES = (
    "get_baseproject",
    "get_costrecovery",
    "get_grosssplit",
    "get_transition",
    "get_contract_table",
    "get_detailed_summary",
    "get_contract_optimization",
    "get_grosssplit_split",
    "get_transition_split",
    "get_sensitivity",
    "get_uncertainty",
    "get_combine_summary",
    "get_increment_summary",
    "get_ltp_dict",
    "get_rpd_dict",
    "get_economic_limit",
    "get_asr_expenditures",
    "get_lbt_expenditures",
)


class _RawResponse(Protocol):
    status_code: int
    headers: Mapping[str, str]

    def json(self) -> object: ...


class _RawClient(Protocol):
    def post(
        self,
        url: str,
        *,
        content: str,
        headers: Mapping[str, str] | None = ...,
    ) -> _RawResponse: ...


_raw_client: _RawClient = TestClient(app, raise_server_exceptions=False)


def _gs_payload(
    *,
    strict: bool = True,
    pod1: bool = True,
    gas_onstream: str | None = "01/01/2029",
) -> dict[str, object]:
    return {
        "setup": {
            "start_date": "01/01/2019",
            "end_date": "31/12/2050",
            "oil_onstream_date": "01/01/2029",
            "gas_onstream_date": gas_onstream,
            "approval_year": 2024,
            "is_pod_1": pod1,
            "is_strict": strict,
        },
        "summary_arguments": {},
        "contract_arguments": {},
        "grosssplit": {},
        "intangible": {
            "base": {
                "start_year": 2019,
                "end_year": 2050,
                "expense_year": [2026, 2027, 2028],
                "cost": [1.0, 2.0, 3.0],
                "cost_allocation": ["Gas", "Gas", "Gas"],
                "cost_type": ["preonstream_cost"] * 3,
            }
        },
    }


def _cr_payload() -> dict[str, object]:
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
        "costrecovery": {},
        "lifting": {
            "Oil": {
                "start_year": 2019,
                "end_year": 2050,
                "prod_year": [2029, 2030],
                "lifting_rate": [1000.0, 1000.0],
                "price": [75.0, 75.0],
                "fluid_type": "Oil",
            }
        },
        "intangible": {
            "base": {
                "start_year": 2019,
                "end_year": 2050,
                "expense_year": [2030],
                "cost": [1000.0],
                "cost_allocation": ["Oil"],
                "cost_type": ["preonstream_cost"],
            }
        },
    }


def _bp_payload() -> dict[str, object]:
    return {
        "setup": {
            "start_date": "01/01/2020",
            "end_date": "31/12/2010",
            "oil_onstream_date": "01/01/2029",
            "gas_onstream_date": "01/01/2029",
            "approval_year": 2024,
            "is_pod_1": False,
            "is_strict": True,
        },
        "summary_arguments": {},
        "contract_arguments": {},
    }


def _paths(body: dict[str, object]) -> list[list[str | int]]:
    errors = as_records(body["errors"])
    return [
        as_path(loc["path"]) for error in errors for loc in as_records(error["locations"])
    ]


def _spy(monkeypatch: pytest.MonkeyPatch, *, forbidden: bool) -> list[str]:
    """Patch every expensive worker entry and record (or forbid) calls."""
    import pyscnomics.api.router as router

    calls: list[str] = []

    def make(name: str):
        def _worker(*_args: object, **_kwargs: object) -> object:
            calls.append(name)
            if forbidden:
                raise AssertionError(f"{name} must not run for unsafe input")
            return {}

        return _worker

    for name in _WORKER_NAMES:
        monkeypatch.setattr(router, name, make(name))
    return calls


# ── Validation-only endpoints ────────────────────────────────────────────────


def test_validate_returns_200_with_can_calculate_false() -> None:
    resp = client.post("/api/grosssplit/validate", json=_gs_payload())
    assert resp.status_code == 200
    body = response_body(resp)
    assert body["kind"] == "input_validation"
    assert body["can_calculate"] is False
    assert body["errors"]
    assert body["request_id"] == resp.headers["X-Request-ID"]


def test_validate_returns_can_calculate_true_for_clean_input() -> None:
    payload = _gs_payload(pod1=False)
    _ = payload.pop("intangible")
    resp = client.post("/api/grosssplit/validate", json=payload)
    assert resp.status_code == 200
    assert response_body(resp)["can_calculate"] is True


def test_malformed_preflight_returns_structured_422() -> None:
    resp = client.post("/api/grosssplit/validate", json={"setup": "bad"})
    assert resp.status_code == 422
    assert isinstance(response_body(resp).get("detail"), list)


@pytest.mark.parametrize(
    "route",
    [
        "/api/baseproject/validate",
        "/api/costrecovery/validate",
        "/api/grosssplit/validate",
        "/api/transition/validate",
        "/api/combine/validate",
        "/api/increment/validate",
    ],
)
def test_preflight_does_not_run_financial_or_analysis_work(
    monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    calls = _spy(monkeypatch, forbidden=True)
    if route.endswith("combine/validate") or route.endswith("increment/validate"):
        payload: object = combine_request(
            [{"cId": 1, "ctr": 2, "data": _gs_payload()}]
        )
    elif route.endswith("transition/validate"):
        payload = strict_transition_two_block_payload()
    elif route.endswith("costrecovery/validate"):
        payload = _cr_payload()
    elif route.endswith("baseproject/validate"):
        payload = _bp_payload()
    else:
        payload = _gs_payload()
    resp = client.post(route, json=payload)
    assert resp.status_code in (200, 422), route
    assert calls == [], route


def test_incomplete_checks_never_certify_validity() -> None:
    # A genuinely incomplete report (invalid project dates skip dependent
    # checks) must not certify validity.
    payload = _gs_payload()
    _setup(payload)["end_date"] = "01/01/2018"  # before the start date
    resp = client.post("/api/grosssplit/validate", json=payload)
    assert resp.status_code == 200
    body = response_body(resp)
    assert body["can_calculate"] is False
    assert body["skipped_checks"]


# ── Direct protection ────────────────────────────────────────────────────────


_GS_ROUTES = (
    "/api/grosssplit",
    "/api/grosssplit/detailed_summary",
    "/api/grosssplit/table",
    "/api/grosssplit/split",
    "/api/grosssplit/warnings",
    "/api/grosssplit/optimization",
    "/api/grosssplit/sensitivity",
    "/api/grosssplit/uncertainty",
)
_CR_ROUTES = (
    "/api/costrecovery",
    "/api/costrecovery/detailed_summary",
    "/api/costrecovery/table",
    "/api/costrecovery/warnings",
    "/api/costrecovery/optimization",
    "/api/costrecovery/sensitivity",
    "/api/costrecovery/uncertainty",
)
_BP_ROUTES = (
    "/api/baseproject",
    "/api/baseproject/table",
    "/api/baseproject/detailed_summary",
    "/api/baseproject/warnings",
    "/api/baseproject/sensitivity",
    "/api/baseproject/uncertainty",
)
_TRANSITION_ROUTES = (
    "/api/transition",
    "/api/transition/detailed_summary",
    "/api/transition/table",
    "/api/transition/split",
    "/api/transition/warnings",
    "/api/transition/optimization",
    "/api/transition/sensitivity",
    "/api/transition/uncertainty",
)


@pytest.mark.parametrize("route", _GS_ROUTES)
def test_direct_gs_calculation_cannot_bypass_checks(route: str) -> None:
    resp = client.post(route, json=_gs_payload())
    assert resp.status_code == 422, route
    assert response_body(resp)["kind"] == "input_validation"


@pytest.mark.parametrize("route", _CR_ROUTES)
def test_direct_cr_calculation_cannot_bypass_checks(route: str) -> None:
    resp = client.post(route, json=_cr_payload())
    assert resp.status_code == 422, route
    assert response_body(resp)["kind"] == "input_validation"


@pytest.mark.parametrize("route", _BP_ROUTES)
def test_direct_bp_calculation_cannot_bypass_checks(route: str) -> None:
    resp = client.post(route, json=_bp_payload())
    assert resp.status_code == 422, route
    assert response_body(resp)["kind"] == "input_validation"


@pytest.mark.parametrize("route", _TRANSITION_ROUTES)
def test_direct_transition_calculation_cannot_bypass_checks(route: str) -> None:
    resp = client.post(route, json=strict_transition_two_block_payload())
    assert resp.status_code == 422, route
    assert response_body(resp)["kind"] == "input_validation"


@pytest.mark.parametrize(
    "route",
    ["/api/grosssplit", "/api/grosssplit/table", "/api/grosssplit/sensitivity"],
)
def test_direct_unsafe_input_never_starts_expensive_worker(
    monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    calls = _spy(monkeypatch, forbidden=True)
    resp = client.post(route, json=_gs_payload())
    assert resp.status_code == 422, route
    assert calls == [], route


# ── Nested case validation ───────────────────────────────────────────────────


def test_outer_malformed_nested_case_is_not_certified() -> None:
    payload = combine_request(
        [{"cId": 2, "ctr": 2, "data": {"setup": {"is_pod_1": False}}}]
    )
    resp = client.post("/api/combine/validate", json=payload)
    assert resp.status_code == 422
    body = response_body(resp)
    assert body["kind"] == "input_validation"
    assert body.get("can_calculate") is not True
    assert body["errors"]
    assert isinstance(body.get("detail"), list)


def test_nested_missing_child_fields_keep_detail_array_and_relative_paths() -> None:
    payload = combine_request(
        [{"cId": 2, "ctr": 2, "data": {"setup": {"is_pod_1": False}}}]
    )
    resp = client.post("/api/combine/validate", json=payload)
    assert resp.status_code == 422
    body = response_body(resp)
    detail = as_records(body["detail"])
    assert detail
    # The standard detail keeps the FastAPI ``body`` prefix and diagnostics.
    assert all(as_path(entry["loc"])[0] == "body" for entry in detail)
    assert all("msg" in entry and "type" in entry and "input" in entry for entry in detail)
    # Envelope locations are relative and start at the nested case.
    paths = _paths(body)
    assert paths
    assert all(path[:3] == ["cases", 0, "data"] for path in paths)


def test_outer_valid_cid_options_cannot_certify_arbitrary_data() -> None:
    # A structurally valid outer case with a well-formed ``cId``/``ctr`` but a
    # data dict that is not a real contract payload must never be certified.
    payload = combine_request(
        [{"cId": 99, "ctr": 2, "data": {"arbitrary": {"nested": True}}}]
    )
    resp = client.post("/api/combine/validate", json=payload)
    assert resp.status_code == 422
    assert response_body(resp).get("can_calculate") is not True


def test_non_mapping_case_is_not_silently_continued() -> None:
    payload = combine_request([{"cId": 1, "ctr": 2, "data": []}])
    resp = client.post("/api/combine/validate", json=payload)
    assert resp.status_code == 422
    assert isinstance(response_body(resp).get("detail"), list)


def test_missing_ctr_is_not_silently_continued() -> None:
    payload = combine_request([{"cId": 1, "data": _gs_payload()}])
    resp = client.post("/api/combine/validate", json=payload)
    assert resp.status_code == 422


def test_nested_transition_recurses_both_contract_blocks() -> None:
    payload = combine_request(
        [{"cId": 1, "ctr": 3, "data": strict_transition_two_block_payload()}]
    )
    resp = client.post("/api/combine/validate", json=payload)
    body = response_body(resp)
    assert body["errors"]
    paths = _paths(body)
    prefixes = {tuple(path[:4]) for path in paths}
    assert ("cases", 0, "data", "contract_1") in prefixes
    assert ("cases", 0, "data", "contract_2") in prefixes


def test_nested_transition_missing_scheme_is_not_certified() -> None:
    data = transition_payload_for("cr", "cr")
    child = cast("dict[str, object]", data["contract_1"])
    child["costrecovery"] = None
    child["grosssplit"] = None
    payload = combine_request([{"cId": 1, "ctr": 3, "data": data}])
    resp = client.post("/api/combine/validate", json=payload)
    assert resp.status_code == 200
    body = response_body(resp)
    assert body["can_calculate"] is False
    # Neither scheme is a located error (F3), never a silent Base Project check.
    errors = as_records(body["errors"])
    assert any(str(issue["code"]) == "TRANSITION_SCHEME_UNRECOGNIZED" for issue in errors)
    scopes = {str(check["scope"]) for check in as_records(body["skipped_checks"])}
    assert "contract_1:scheme" in scopes
    # The direct route must not fall through to an opaque 500.
    direct = client.post("/api/combine", json=payload)
    assert direct.status_code == 422
    assert response_body(direct)["kind"] != "server_error"


def test_direct_combine_nested_malformed_is_structured_not_opaque() -> None:
    payload = combine_request(
        [{"cId": 2, "ctr": 2, "data": {"setup": {"is_pod_1": False}}}]
    )
    resp = client.post("/api/combine", json=payload)
    assert resp.status_code == 422
    body = response_body(resp)
    assert body["kind"] == "input_validation"
    assert body["kind"] != "server_error"


# ── Transition direct vs combine/increment parity ────────────────────────────


def test_strict_transition_two_blocks_direct_four_errors() -> None:
    resp = client.post("/api/transition", json=strict_transition_two_block_payload())
    assert resp.status_code == 422
    body = response_body(resp)
    errors = as_records(body["errors"])
    assert len(errors) == 4, [e["code"] for e in errors]
    prefixes = {tuple(path[:1]) for path in _paths(body)}
    assert prefixes == {("contract_1",), ("contract_2",)}


@pytest.mark.parametrize("route", ["/api/combine", "/api/increment"])
def test_strict_transition_errors_survive_combine_increment_prefix(route: str) -> None:
    direct = client.post(
        "/api/transition", json=strict_transition_two_block_payload()
    )
    direct_body = response_body(direct)
    direct_codes = sorted(
        str(e["code"]) for e in as_records(direct_body["errors"])
    )

    payload = combine_request(
        [{"cId": 1, "ctr": 3, "data": strict_transition_two_block_payload()}]
    )
    resp = client.post(route, json=payload)
    assert resp.status_code == 422, route
    body = response_body(resp)
    nested_codes = sorted(str(e["code"]) for e in as_records(body["errors"]))
    assert nested_codes == direct_codes
    paths = _paths(body)
    assert paths
    assert all(path[:3] == ["cases", 0, "data"] for path in paths)
    assert {tuple(path[3:4]) for path in paths} == {("contract_1",), ("contract_2",)}


# ── Real baseline mutations ──────────────────────────────────────────────────


_ROW = "xxx-pod-corrected - base"


def _setup(payload: dict[str, object]) -> dict[str, object]:
    return cast("dict[str, object]", payload["setup"])


def _row(payload: dict[str, object], section: str) -> dict[str, object]:
    tables = cast("dict[str, object]", payload[section])
    return cast("dict[str, object]", tables[_ROW])


def _series(payload: dict[str, object], section: str, field: str) -> list[object]:
    return cast("list[object]", _row(payload, section)[field])


def test_baseline_corrected_is_preflight_clean_and_direct() -> None:
    resp = client.post("/api/grosssplit/validate", json=corrected_payload())
    assert resp.status_code == 200
    assert response_body(resp)["can_calculate"] is True
    assert client.post("/api/grosssplit", json=corrected_payload()).status_code == 200


def test_reversed_project_date_preflight_not_false_valid_direct_not_opaque() -> None:
    payload = corrected_payload()
    _setup(payload)["end_date"] = "01/01/2018"
    preflight = client.post("/api/grosssplit/validate", json=payload)
    assert preflight.status_code == 200
    body = response_body(preflight)
    assert body["can_calculate"] is False
    assert body["errors"]
    direct = client.post("/api/grosssplit", json=payload)
    assert direct.status_code == 422
    assert response_body(direct)["kind"] == "input_validation"


def test_array_mismatch_preflight_not_false_valid_direct_not_opaque() -> None:
    payload = corrected_payload()
    _row(payload, "opex")["fixed_cost"] = [1.0, 2.0]
    preflight = client.post("/api/grosssplit/validate", json=payload)
    assert preflight.status_code == 200
    assert response_body(preflight)["can_calculate"] is False
    direct = client.post("/api/grosssplit", json=payload)
    assert direct.status_code == 422
    assert response_body(direct)["kind"] == "input_validation"


def test_year_2090_preflight_not_false_valid_direct_not_opaque() -> None:
    payload = corrected_payload()
    _series(payload, "opex", "expense_year")[0] = 2090
    preflight = client.post("/api/grosssplit/validate", json=payload)
    assert preflight.status_code == 200
    assert response_body(preflight)["can_calculate"] is False
    direct = client.post("/api/grosssplit", json=payload)
    assert direct.status_code == 422


def test_bad_number_preflight_not_false_valid_direct_not_opaque() -> None:
    payload = corrected_payload()
    _series(payload, "opex", "fixed_cost")[0] = "abc"
    preflight = client.post("/api/grosssplit/validate", json=payload)
    assert preflight.status_code == 422
    assert isinstance(response_body(preflight).get("detail"), list)
    direct = client.post("/api/grosssplit", json=payload)
    assert direct.status_code == 422
    assert isinstance(response_body(direct).get("detail"), list)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_raw_nonfinite_preflight_not_false_valid_direct_not_opaque(literal: str) -> None:
    payload = corrected_payload()
    _series(payload, "opex", "fixed_cost")[0] = 1.0
    raw = json.dumps(payload, allow_nan=True).replace("[1.0,", f"[{literal},", 1)
    preflight = _raw_client.post(
        "/api/grosssplit/validate",
        content=raw,
        headers={"content-type": "application/json"},
    )
    assert preflight.status_code == 422
    assert response_body(preflight)["kind"] == "input_validation"
    direct = _raw_client.post(
        "/api/grosssplit",
        content=raw,
        headers={"content-type": "application/json"},
    )
    assert direct.status_code == 422
    assert response_body(direct)["kind"] == "input_validation"


def test_null_onstream_valid_is_engine_accepted() -> None:
    payload = corrected_payload()
    _setup(payload)["gas_onstream_date"] = None
    direct = client.post("/api/grosssplit", json=payload)
    assert direct.status_code == 200
    preflight = client.post("/api/grosssplit/validate", json=payload)
    assert preflight.status_code == 200
    body = response_body(preflight)
    # The nullable default is engine-accepted: the effective onstream year is
    # resolved from revenue, so the report is complete and calculable (F7).
    assert body["errors"] == []
    assert body["can_calculate"] is True
    assert body["skipped_checks"] == []


@pytest.mark.parametrize(("label", "scheme_1", "scheme_2"), transition_combinations())
def test_valid_transition_combination_is_engine_accepted(
    label: str, scheme_1: str, scheme_2: str
) -> None:
    payload = transition_payload_for(scheme_1, scheme_2)
    direct = client.post("/api/transition", json=payload)
    assert direct.status_code == 200, label
    preflight = client.post("/api/transition/validate", json=payload)
    assert preflight.status_code == 200, label
    assert response_body(preflight)["errors"] == []


@pytest.mark.parametrize(("label", "scheme_1", "scheme_2"), transition_combinations())
def test_valid_transition_combination_injected_error_is_blocked(
    label: str, scheme_1: str, scheme_2: str
) -> None:
    payload = transition_payload_for(scheme_1, scheme_2)
    child = cast("dict[str, object]", payload["contract_2"])
    setup = cast("dict[str, object]", child["setup"])
    setup["start_date"] = "01/01/2030"
    setup["end_date"] = "01/01/2020"
    resp = client.post("/api/transition", json=payload)
    assert resp.status_code == 422, label
    body = response_body(resp)
    paths = _paths(body)
    assert paths
    assert all(path[0] in {"contract_1", "contract_2"} for path in paths)


# ── Internal failures stay distinct from user input ──────────────────────────


class _CustomInternalError(Exception):
    """A genuine internal failure, not a documented user-input error."""


def test_runtime_error_is_500_with_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    import pyscnomics.api.router as router

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected genuine runtime failure")

    monkeypatch.setattr(router, "get_grosssplit", _boom)
    resp = client.post("/api/grosssplit", json=corrected_payload())
    assert resp.status_code == 500
    body = response_body(resp)
    assert body["kind"] == "server_error"
    assert body["validation_complete"] is False
    assert "Traceback" not in json.dumps(body)
    assert resp.headers["X-Request-ID"] == body["request_id"]


def test_custom_internal_error_is_500_not_user_422(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyscnomics.api.router as router

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise _CustomInternalError("internal only")

    monkeypatch.setattr(router, "get_grosssplit", _boom)
    resp = client.post("/api/grosssplit", json=corrected_payload())
    assert resp.status_code == 500
    assert response_body(resp)["kind"] == "server_error"


# ── Successful shapes and old-client compatibility ───────────────────────────


def test_valid_direct_success_shape_unchanged() -> None:
    resp = client.post("/api/grosssplit", json=corrected_payload())
    assert resp.status_code == 200
    body = response_body(resp)
    assert "execution_info" in body


def test_valid_warnings_retain_message_field() -> None:
    resp = client.post("/api/grosssplit/warnings", json=corrected_payload())
    assert resp.status_code == 200
    warnings = resp.json()
    assert isinstance(warnings, list)
    for entry in cast("list[object]", warnings):
        assert isinstance(entry, dict)
        assert "Message" in cast("dict[str, object]", entry)


def test_single_summary_ignores_unused_analysis_blocks() -> None:
    """The summary worker never reads the submitted analysis blocks.

    Only the dedicated ``*/sensitivity``, ``*/uncertainty`` and
    ``*/optimization`` workers consume them; a summary calculation must reach
    the worker and return its exact result instead of being blocked.
    """
    import copy
    import json

    from pyscnomics.api.adapter import get_grosssplit
    from pyscnomics.api.converter import Data

    payload = copy.deepcopy(corrected_payload())
    payload["sensitivity_arguments"] = {
        "min_deviation": 0,
        "max_deviation": 0,
        "step": 0,
        "base_value": 1,
    }
    payload["uncertainty_arguments"] = {
        "run_number": 0,
        "oil_price_distribution": "Normal",
        "gas_price_distribution": "Normal",
        "opex_distribution": "Normal",
        "capex_distribution": "Normal",
        "lifting_distribution": "Normal",
    }

    direct = client.post("/api/grosssplit", json=payload)
    assert direct.status_code == 200, response_body(direct)
    dump = Data.model_validate(payload).model_dump()
    assert response_body(direct) == json.loads(json.dumps(get_grosssplit(data=dump)[0]))

    preflight = response_body(client.post("/api/grosssplit/validate", json=payload))
    assert preflight["can_calculate"] is False
    paths = _paths(preflight)
    assert ["sensitivity_arguments", "step"] in paths
    assert ["uncertainty_arguments", "run_number"] in paths

    consumed = client.post("/api/grosssplit/sensitivity", json=payload)
    assert consumed.status_code == 422
    body = response_body(consumed)
    issue = next(
        entry
        for entry in as_records(body["errors"])
        if entry["code"] == "SENSITIVITY_STEP_NOT_POSITIVE"
    )
    assert "Step is 0" in str(issue["message"])
    assert ["sensitivity_arguments", "step"] in _paths(body)

    consumed_uncertainty = client.post("/api/grosssplit/uncertainty", json=payload)
    assert consumed_uncertainty.status_code == 422
    assert any(
        path[0] == "uncertainty_arguments"
        for path in _paths(response_body(consumed_uncertainty))
    )


def test_valid_calculation_is_not_blocked_by_skipped_preflight() -> None:
    # The valid transition fixtures disclose skipped nullable checks but the
    # engine still accepts them; the direct route must not be blocked.
    payload = transition_payload_for("cr", "cr")
    assert client.post("/api/transition", json=payload).status_code == 200


# ── Real frontend Case enum for the four transition combinations ─────────────


_FRONTEND_CTR = {"cr-cr": 3, "cr-gs": 4, "gs-cr": 5, "gs-gs": 6}


@pytest.mark.parametrize(("label", "scheme_1", "scheme_2"), transition_combinations())
def test_nested_transition_uses_frontend_case_enum(
    label: str, scheme_1: str, scheme_2: str
) -> None:
    ctr = _FRONTEND_CTR[label]
    payload = combine_request(
        [{"cId": 1, "ctr": ctr, "data": transition_payload_for(scheme_1, scheme_2)}]
    )
    resp = client.post("/api/combine/validate", json=payload)
    assert resp.status_code == 200, label
    assert response_body(resp)["errors"] == []


@pytest.mark.parametrize(("label", "scheme_1", "scheme_2"), transition_combinations())
def test_nested_transition_injected_error_keeps_case_and_block_prefix(
    label: str, scheme_1: str, scheme_2: str
) -> None:
    ctr = _FRONTEND_CTR[label]
    data = transition_payload_for(scheme_1, scheme_2)
    child = cast("dict[str, object]", data["contract_2"])
    setup = cast("dict[str, object]", child["setup"])
    setup["start_date"] = "01/01/2030"
    setup["end_date"] = "01/01/2020"
    payload = combine_request([{"cId": 1, "ctr": ctr, "data": data}])
    resp = client.post("/api/increment/validate", json=payload)
    assert resp.status_code == 200, label
    body = response_body(resp)
    assert body["can_calculate"] is False
    paths = _paths(body)
    assert paths
    assert all(path[:3] == ["cases", 0, "data"] for path in paths)
    assert all(path[3] in {"contract_1", "contract_2"} for path in paths)


def test_nested_case_index_attribution_is_retained() -> None:
    valid = transition_payload_for("cr", "cr")
    bad = transition_payload_for("cr", "cr")
    bad_child = cast("dict[str, object]", bad["contract_1"])
    setup = cast("dict[str, object]", bad_child["setup"])
    setup["start_date"] = "01/01/2030"
    setup["end_date"] = "01/01/2020"
    payload = combine_request(
        [
            {"cId": 1, "ctr": 3, "data": valid},
            {"cId": 2, "ctr": 3, "data": bad},
        ]
    )
    resp = client.post("/api/combine/validate", json=payload)
    assert resp.status_code == 200
    body = response_body(resp)
    paths = _paths(body)
    assert paths
    assert all(path[:3] == ["cases", 1, "data"] for path in paths)
    assert not any(path[:3] == ["cases", 0, "data"] for path in paths)
