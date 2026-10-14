"""API translation tests for structured input errors (Task 1)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import cast

import pytest

from pyscnomics.api.adapter import (
    get_contract_optimization,
    get_sensitivity,
    get_transition,
    get_uncertainty,
)
from pyscnomics.api.errors import (
    validation_error_response,
    validation_success_response,
)
from pyscnomics.input_validation import (
    InputValidationError,
    IssueLocation,
    SkippedCheck,
    ValidationIssue,
    ValidationReport,
)
from tests._validation_api_support import as_path, as_records, client, response_body


def _gs_payload(*, strict: bool = True, pod1: bool = True) -> dict[str, object]:
    return {
        "setup": {
            "start_date": "01/01/2019",
            "end_date": "31/12/2050",
            "oil_onstream_date": "01/01/2029",
            "gas_onstream_date": "01/01/2029",
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
                "cost": [76951.27737, 183682.6991, 268439.1445],
                "cost_allocation": ["Gas", "Gas", "Gas"],
                "cost_type": ["preonstream_cost", "preonstream_cost", "preonstream_cost"],
            }
        },
        "opex": {
            "base": {
                "start_year": 2019,
                "end_year": 2050,
                "expense_year": [2025, 2026, 2027, 2028],
                "fixed_cost": [25521.58125, 60382.44525, 19108.18638, 18900.21773],
                "cost_allocation": ["Gas", "Gas", "Gas", "Gas"],
                "cost_type": [
                    "preonstream_cost",
                    "preonstream_cost",
                    "preonstream_cost",
                    "preonstream_cost",
                ],
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


def test_strict_gs_error_returns_structured_422() -> None:
    resp = client.post("/api/grosssplit", json=_gs_payload(strict=True))
    assert resp.status_code == 422
    body = response_body(resp)
    assert body["schema_version"] == 1
    assert body["kind"] == "input_validation"
    assert body["validation_complete"] is True
    assert str(body["detail"]).startswith("PyscErr:")
    assert resp.headers["X-Request-ID"] == body["request_id"]
    errors = as_records(body["errors"])
    by_code = {e["code"]: e for e in errors}
    assert set(by_code) == {
        "GS_POD1_NONCAPITAL_INTANGIBLE",
        "GS_POD1_NONCAPITAL_OPEX",
    }
    intangible_locations = as_records(by_code["GS_POD1_NONCAPITAL_INTANGIBLE"]["locations"])
    opex_locations = as_records(by_code["GS_POD1_NONCAPITAL_OPEX"]["locations"])
    assert len(intangible_locations) == 3
    assert len(opex_locations) == 4
    # Paths are relative to the payload and never include the FastAPI body prefix.
    assert intangible_locations[0]["path"] == [
        "intangible",
        "base",
        "cost_type",
        0,
    ]
    assert json.dumps(body, allow_nan=False)


def test_loose_gs_is_not_blocked() -> None:
    resp = client.post("/api/grosssplit", json=_gs_payload(strict=False))
    assert resp.status_code != 422


def test_cr_input_error_returns_structured_422() -> None:
    resp = client.post("/api/costrecovery", json=_cr_payload())
    assert resp.status_code == 422
    body = response_body(resp)
    assert body["kind"] == "input_validation"
    errors = as_records(body["errors"])
    assert any(e["code"] == "PREONSTREAM_YEAR_OUTSIDE_PROJECT" for e in errors)


def test_transition_error_retains_contract_prefix() -> None:
    child: dict[str, object] = {
        "setup": _gs_payload()["setup"],
        "contract_arguments": {},
        "lifting": {},
        "capital": {},
        "intangible": _gs_payload()["intangible"],
        "opex": _gs_payload()["opex"],
        "asr": {},
        "lbt": {},
        "grosssplit": {},
    }
    payload: dict[str, object] = {
        "contract_1": child,
        "contract_2": {**child, "grosssplit": {}},
        "contract_arguments": {"unrec_portion": 0.0},
        "summary_arguments": {},
    }
    resp = client.post("/api/transition", json=payload)
    assert resp.status_code == 422
    body = response_body(resp)
    errors = as_records(body["errors"])
    paths = [as_path(loc["path"]) for e in errors for loc in as_records(e["locations"])]
    assert paths
    assert all(path[0] in {"contract_1", "contract_2"} for path in paths)


def test_shape_errors_keep_detail_array() -> None:
    resp = client.post("/api/grosssplit", json={"setup": "not-an-object"})
    assert resp.status_code == 422
    body = response_body(resp)
    assert isinstance(body.get("detail"), list)


def test_invalid_econlimit_method_is_user_input_422() -> None:
    resp = client.post(
        "/api/econlimit",
        json={"years": [2020, 2021], "cash_flow": [1, 2], "method": "zzz"},
    )
    assert resp.status_code == 422
    body = response_body(resp)
    assert body["kind"] == "input_validation"
    errors = as_records(body["errors"])
    assert any(e["code"] == "ECONLIMIT_METHOD_UNRECOGNIZED" for e in errors)


def _raise_runtime_error(*_args: object, **_kwargs: object) -> object:
    raise RuntimeError("injected genuine internal failure")


def test_unexpected_runtime_error_remains_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "pyscnomics.api.router.get_economic_limit", _raise_runtime_error
    )
    resp = client.post(
        "/api/econlimit",
        json={"years": [2020, 2021], "cash_flow": [1, 2], "method": "Maximum NPV"},
    )
    assert resp.status_code == 500


def test_unexpected_error_has_reference_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "pyscnomics.api.router.get_economic_limit", _raise_runtime_error
    )
    resp = client.post(
        "/api/econlimit",
        json={"years": [2020, 2021], "cash_flow": [1, 2], "method": "Maximum NPV"},
    )
    body = response_body(resp)
    assert body["kind"] == "server_error"
    assert body["validation_complete"] is False
    assert str(body["request_id"]) in str(body["message"])
    assert "Traceback" not in json.dumps(body)
    assert resp.headers["X-Request-ID"] == body["request_id"]


def test_422_and_500_are_readable_through_allowed_cors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    origin = "http://127.0.0.1:5173"
    r422 = client.post(
        "/api/grosssplit", json=_gs_payload(strict=True), headers={"Origin": origin}
    )
    assert r422.status_code == 422
    assert r422.headers["access-control-allow-origin"] == origin
    assert "X-Request-ID" in r422.headers["access-control-expose-headers"]

    monkeypatch.setattr(
        "pyscnomics.api.router.get_economic_limit", _raise_runtime_error
    )
    r500 = client.post(
        "/api/econlimit",
        json={"years": [2020, 2021], "cash_flow": [1, 2], "method": "Maximum NPV"},
        headers={"Origin": origin},
    )
    assert r500.status_code == 500
    assert r500.headers["access-control-allow-origin"] == origin
    assert response_body(r500)["kind"] == "server_error"


def test_legacy_detail_contains_first_message_and_guidance() -> None:
    issue = ValidationIssue(
        code="X",
        severity="error",
        message="Cost Type is not allowed.",
        guidance="Open Intangible Costs and select Check again.",
        locations=(IssueLocation(("intangible",)),),
    )
    response = validation_error_response(InputValidationError((issue,)), "rid")
    body = cast("dict[str, object]", json.loads(bytes(response.body)))
    detail = str(body["detail"])
    assert detail.startswith("PyscErr:")
    assert "Cost Type is not allowed." in detail
    assert "Open Intangible Costs" in detail


def test_incomplete_success_does_not_claim_completion() -> None:
    report = ValidationReport(
        validation_complete=False,
        skipped_checks=(SkippedCheck("scope", "reason"),),
    )
    response = validation_success_response(report, "rid")
    body = cast("dict[str, object]", json.loads(bytes(response.body)))
    assert body["can_calculate"] is False
    assert "Input checks completed" not in str(body["message"])


def test_complete_clean_success_still_reports_completion() -> None:
    response = validation_success_response(ValidationReport(), "rid")
    body = cast("dict[str, object]", json.loads(bytes(response.body)))
    assert body["can_calculate"] is True
    assert str(body["message"]).startswith("Input checks completed")


# ── Adapter analysis-argument user-input errors (A14/A21/A22, Task 2C) ──

_FIXTURES = Path(__file__).parent / "fixtures" / "input_validation"


def _corrected_payload() -> dict[str, object]:
    """A preflight-clean payload that reaches the adapter analysis paths."""
    raw = cast(
        "dict[str, object]",
        json.loads((_FIXTURES / "corrected.json").read_text()),
    )
    payload = cast("dict[str, object]", raw["payload"])
    return copy.deepcopy(payload)


@pytest.mark.parametrize(
    ("route", "code", "field"),
    [
        ("/api/grosssplit/sensitivity", "SENSITIVITY_ARGUMENTS_MISSING", "sensitivity_arguments"),
        (
            "/api/grosssplit/optimization",
            "OPTIMIZATION_ARGUMENTS_MISSING",
            "optimization_arguments",
        ),
        (
            "/api/grosssplit/uncertainty",
            "UNCERTAINTY_ARGUMENTS_MISSING",
            "uncertainty_arguments",
        ),
    ],
)
def test_missing_analysis_arguments_are_structured_422(
    route: str, code: str, field: str
) -> None:
    resp = client.post(route, json=_corrected_payload())
    assert resp.status_code == 422
    body = response_body(resp)
    assert body["kind"] == "input_validation"
    errors = as_records(body["errors"])
    matches = [e for e in errors if e["code"] == code]
    assert matches, [e["code"] for e in errors]
    locations = as_records(matches[0]["locations"])
    assert locations
    assert as_path(locations[0]["path"])[0] == field
    assert str(body["detail"]).startswith("PyscErr:")
    assert field in str(body["detail"])


def test_empty_analysis_arguments_are_user_input_direct() -> None:
    for field, adapter_call, code in (
        ("sensitivity_arguments", get_sensitivity, "SENSITIVITY_ARGUMENTS_EMPTY"),
        (
            "optimization_arguments",
            get_contract_optimization,
            "OPTIMIZATION_ARGUMENTS_EMPTY",
        ),
        ("uncertainty_arguments", get_uncertainty, "UNCERTAINTY_ARGUMENTS_EMPTY"),
    ):
        payload = _corrected_payload()
        payload[field] = {}
        with pytest.raises(InputValidationError) as excinfo:
            _ = adapter_call(payload, "Gross Split")
        assert excinfo.value.issues[0].code == code
        assert excinfo.value.issues[0].locations[0].path == (field,)


def test_transition_scheme_selection_is_user_input() -> None:
    from pyscnomics.api.adapter import ContractException, ContractInputError

    with pytest.raises(ContractInputError) as excinfo:
        _ = get_transition(
            {
                "contract_1": {"costrecovery": None, "grosssplit": None},
                "contract_2": {"costrecovery": None, "grosssplit": None},
            }
        )
    assert isinstance(excinfo.value, ContractException)
    assert excinfo.value.issues[0].code == "TRANSITION_SCHEME_UNRECOGNIZED"
    assert excinfo.value.issues[0].locations[0].path == ("contract_1",)
