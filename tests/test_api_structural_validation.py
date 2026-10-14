"""Structural request-validation boundary tests (Task 1B).

Covers the FastAPI ``RequestValidationError`` handler, normalized issue
locations, numeric finite configuration on request models, server-generated
request references, and the unchanged lifecycle of non-validation responses.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Protocol, cast

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from pyscnomics.api.converter import (
    CapitalCostBM,
    ContractArgumentsBM,
    EconLimit,
    LiftingBM,
    LtpBM,
    RpdBM,
    SensitivityBM,
    SetupBM,
    SummaryArgumentsBM,
    UncertaintyBM,
)
from pyscnomics.api.main import app
from tests._validation_api_support import (
    as_path,
    as_records,
    client,
    response_body,
)

_ORIGIN = "http://127.0.0.1:5173"


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

    def get(self, url: str) -> _RawResponse: ...


_raw_client: _RawClient = TestClient(app, raise_server_exceptions=False)

_NONFINITE: tuple[object, ...] = (
    float("nan"),
    float("inf"),
    float("-inf"),
    "nan",
    "inf",
    "-inf",
    "Infinity",
)


def _setup() -> dict[str, object]:
    return {
        "start_date": "01/01/2019",
        "end_date": "31/12/2050",
        "oil_onstream_date": "01/01/2029",
        "gas_onstream_date": "01/01/2029",
        "approval_year": 2024,
        "is_pod_1": False,
        "is_strict": True,
    }


def _intangible_payload(
    *, cost_item: object = 1.0, table_name: str = "base"
) -> dict[str, object]:
    return {
        "setup": _setup(),
        "summary_arguments": {},
        "contract_arguments": {},
        "intangible": {
            table_name: {
                "start_year": 2019,
                "end_year": 2050,
                "expense_year": [2026],
                "cost": [cost_item],
                "cost_allocation": ["Gas"],
                "cost_type": ["preonstream_cost"],
            }
        },
    }


def _intangible_payload_with(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "start_year": 2019,
        "end_year": 2050,
        "expense_year": [2026],
        "cost": [1.0],
        "cost_allocation": ["Gas"],
        "cost_type": ["preonstream_cost"],
    }
    row.update(overrides)
    return {
        "setup": _setup(),
        "summary_arguments": {},
        "contract_arguments": {},
        "intangible": {"base": row},
    }


def _capital_payload_with(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "start_year": 2019,
        "end_year": 2050,
        "expense_year": [2026],
        "cost": [1.0],
        "cost_allocation": ["Gas"],
        "cost_type": ["preonstream_cost"],
    }
    row.update(overrides)
    return {
        "setup": _setup(),
        "summary_arguments": {},
        "contract_arguments": {},
        "capital": {"base": row},
    }


def _lifting_payload_with(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "start_year": 2020,
        "end_year": 2030,
        "lifting_rate": [1.0],
        "price": [1.0],
        "prod_year": [2025],
        "fluid_type": "Oil",
    }
    row.update(overrides)
    return {
        "setup": _setup(),
        "summary_arguments": {},
        "contract_arguments": {},
        "lifting": {"base": row},
    }


def _gs_union_payload(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "start_year": 2019,
        "end_year": 2050,
        "expense_year": [2026, 2027, 2028],
        "cost": [76951.27737, 183682.6991, 268439.1445],
        "cost_allocation": ["Gas", "Gas", "Gas"],
        "cost_type": ["preonstream_cost", "preonstream_cost", "preonstream_cost"],
    }
    row.update(overrides)
    setup = {**_setup(), "is_pod_1": True, "is_strict": False}
    return {
        "setup": setup,
        "summary_arguments": {},
        "contract_arguments": {},
        "grosssplit": {},
        "intangible": {"base": row},
    }


def _issues(body: dict[str, object]) -> list[dict[str, object]]:
    return as_records(body["errors"])


def _issue_field(error: dict[str, object]) -> object:
    context = cast("dict[str, object]", error["context"])
    return context.get("field")


def _first_error(body: dict[str, object], code: str) -> dict[str, object]:
    matching = [error for error in _issues(body) if error["code"] == code]
    assert matching
    return matching[0]


def _gs_pod1_payload() -> dict[str, object]:
    return {
        "setup": {**_setup(), "is_pod_1": True, "is_strict": True},
        "summary_arguments": {},
        "contract_arguments": {},
        "grosssplit": {},
        "intangible": {
            "base": {
                "start_year": 2019,
                "end_year": 2050,
                "expense_year": [2026, 2027],
                "cost": [10.0, 20.0],
                "cost_allocation": ["Gas", "Gas"],
                "cost_type": ["preonstream_cost", "preonstream_cost"],
            }
        },
    }


def _clean_gs_payload() -> dict[str, object]:
    return {
        "setup": _setup(),
        "summary_arguments": {},
        "contract_arguments": {},
        "grosssplit": {},
    }


# ── Structural handler ──────────────────────────────────────────────


def test_structural_detail_is_array_with_required_keys() -> None:
    resp = client.post("/api/grosssplit", json={"setup": "not-an-object"})
    assert resp.status_code == 422
    body = response_body(resp)
    assert body["kind"] == "input_validation"
    assert body["validation_complete"] is False
    assert body["message"] == "This input could not be read in the required format."
    assert as_records(body["skipped_checks"])
    detail = as_records(body["detail"])
    assert detail
    first = detail[0]
    assert {"type", "loc", "msg", "input"} <= set(first)
    # Standard detail keeps the raw pydantic diagnostics, including body prefix.
    loc = cast("list[object]", first["loc"])
    assert loc[0] == "body"
    assert json.dumps(body, allow_nan=False)


def test_normalized_location_strips_body_and_union_labels() -> None:
    resp = client.post(
        "/api/grosssplit", json=_intangible_payload(cost_item="bad-number")
    )
    assert resp.status_code == 422
    body = response_body(resp)
    errors = as_records(body["errors"])
    numeric = [e for e in errors if e["code"] == "NUMERIC_FIELD_NOT_A_NUMBER"]
    assert numeric
    paths = [
        as_path(loc["path"]) for e in numeric for loc in as_records(e["locations"])
    ]
    assert ["intangible", "base", "cost", 0] in paths
    for path in paths:
        assert "body" not in path
        assert all("[" not in str(segment) for segment in path)
    assert numeric[0]["message"] == "Cost needs a number."
    assert numeric[0]["guidance"] == (
        "Open the indicated input, correct its format, then select Check again."
    )
    # Diagnostic union branches remain available in the standard detail array.
    detail = as_records(body["detail"])
    locations = [cast(list[object], entry["loc"]) for entry in detail]
    assert any("list[float]" in str(segment) for loc in locations for segment in loc)


@pytest.mark.parametrize(
    "table_name", ["float", "int", "body", "SomeBM", "list[float]"]
)
def test_real_table_keys_survive_normalization(table_name: str) -> None:
    resp = client.post(
        "/api/grosssplit",
        json=_intangible_payload(cost_item="bad-number", table_name=table_name),
    )
    assert resp.status_code == 422
    body = response_body(resp)
    numeric = [e for e in _issues(body) if e["code"] == "NUMERIC_FIELD_NOT_A_NUMBER"]
    assert numeric
    paths = [
        as_path(loc["path"]) for e in numeric for loc in as_records(e["locations"])
    ]
    assert ["intangible", table_name, "cost", 0] in paths


def test_reproduced_strict_false_float_table_key() -> None:
    payload = _intangible_payload(cost_item="bad-number", table_name="float")
    cast("dict[str, object]", payload["setup"])["is_strict"] = False
    resp = client.post("/api/grosssplit", json=payload)
    assert resp.status_code == 422
    body = response_body(resp)
    numeric = [e for e in _issues(body) if e["code"] == "NUMERIC_FIELD_NOT_A_NUMBER"]
    assert numeric
    paths = [
        as_path(loc["path"]) for e in numeric for loc in as_records(e["locations"])
    ]
    assert paths
    assert all(path[:3] == ["intangible", "float", "cost"] for path in paths)


def test_standard_detail_keeps_body_and_union_branches() -> None:
    resp = client.post(
        "/api/grosssplit",
        json=_intangible_payload(cost_item="bad-number", table_name="body"),
    )
    body = response_body(resp)
    detail = as_records(body["detail"])
    locs = [cast("list[object]", entry["loc"]) for entry in detail]
    assert all(loc[0] == "body" for loc in locs)
    assert any("list[float]" in str(segment) for loc in locs for segment in loc)


def test_visible_label_year() -> None:
    resp = client.post(
        "/api/grosssplit", json=_intangible_payload_with(expense_year=["bad"])
    )
    body = response_body(resp)
    error = _first_error(body, "NUMERIC_FIELD_NOT_A_NUMBER")
    assert error["message"] == "Year needs a number."
    assert _issue_field(error) == "Year"


def test_visible_label_pis_year() -> None:
    resp = client.post(
        "/api/grosssplit", json=_capital_payload_with(pis_year=["bad"])
    )
    body = response_body(resp)
    error = _first_error(body, "NUMERIC_FIELD_NOT_A_NUMBER")
    assert error["message"] == (
        "PIS Year (placed-in-service year) needs a number."
    )
    assert _issue_field(error) == "PIS Year (placed-in-service year)"


def test_visible_label_cost_type_and_associated_with() -> None:
    cost_type_body = response_body(
        client.post("/api/grosssplit", json=_intangible_payload_with(cost_type=[123]))
    )
    assert _issue_field(_first_error(cost_type_body, "INPUT_FORMAT_INVALID")) == (
        "Cost Type"
    )
    allocation_body = response_body(
        client.post(
            "/api/grosssplit", json=_intangible_payload_with(cost_allocation=[123])
        )
    )
    assert _issue_field(_first_error(allocation_body, "INPUT_FORMAT_INVALID")) == (
        "Associated With"
    )


def test_visible_label_price() -> None:
    resp = client.post(
        "/api/grosssplit", json=_lifting_payload_with(price=["bad"])
    )
    body = response_body(resp)
    error = _first_error(body, "NUMERIC_FIELD_NOT_A_NUMBER")
    assert error["message"] == "Price needs a number."
    assert _issue_field(error) == "Price"


def test_alternate_union_branch_rows_are_not_reported() -> None:
    payload = _gs_union_payload(
        cost=["bad-number", 183682.6991, 268439.1445]
    )
    resp = client.post("/api/grosssplit", json=payload)
    assert resp.status_code == 422
    body = response_body(resp)
    detail = as_records(body["detail"])
    # All four pydantic diagnostics stay in the standard detail array.
    assert len(detail) == 4
    detail_types = [str(entry["type"]) for entry in detail]
    assert detail_types.count("int_from_float") == 2
    assert "float_parsing" in detail_types
    assert "int_parsing" in detail_types
    # Normalized issues address only the genuinely bad row 0.
    paths = [
        as_path(loc["path"])
        for error in _issues(body)
        for loc in as_records(error["locations"])
    ]
    assert paths
    assert all(path == ["intangible", "base", "cost", 0] for path in paths)
    assert all(
        error["code"] == "NUMERIC_FIELD_NOT_A_NUMBER" for error in _issues(body)
    )
    assert json.dumps(body, allow_nan=False)


def test_independent_bad_rows_and_whole_number_errors_survive() -> None:
    payload = _gs_union_payload(
        cost=["bad-number", "worse", 268439.1445],
        expense_year=[2026, 2027.5, 2028],
    )
    resp = client.post("/api/grosssplit", json=payload)
    assert resp.status_code == 422
    body = response_body(resp)
    by_path: dict[tuple[str | int, ...], list[str]] = {}
    for error in _issues(body):
        for loc in as_records(error["locations"]):
            by_path.setdefault(tuple(as_path(loc["path"])), []).append(
                str(error["code"])
            )
    assert by_path.get(("intangible", "base", "cost", 0))
    assert by_path.get(("intangible", "base", "cost", 1))
    assert ("intangible", "base", "cost", 2) not in by_path
    integer_codes = by_path.get(("intangible", "base", "expense_year", 1), [])
    assert "NUMERIC_FIELD_NOT_INTEGER" in integer_codes
    integer_issue = [
        error
        for error in _issues(body)
        if error["code"] == "NUMERIC_FIELD_NOT_INTEGER"
    ]
    assert integer_issue
    assert integer_issue[0]["message"] == "Year must be a whole number."


def test_valid_decimal_union_value_yields_no_cost_issue() -> None:
    payload = _gs_union_payload(cost=[183682.6991], expense_year="not-a-list")
    resp = client.post("/api/grosssplit", json=payload)
    assert resp.status_code == 422
    body = response_body(resp)
    paths = [
        as_path(loc["path"])
        for error in _issues(body)
        for loc in as_records(error["locations"])
    ]
    assert not any("cost" in path for path in paths)


def test_malformed_array_still_blocks() -> None:
    payload = _gs_union_payload(cost="not-a-list")
    resp = client.post("/api/grosssplit", json=payload)
    assert resp.status_code == 422
    body = response_body(resp)
    paths = [
        as_path(loc["path"])
        for error in _issues(body)
        for loc in as_records(error["locations"])
    ]
    assert ["intangible", "base", "cost"] in paths


def test_string_nan_reports_finite_error_at_real_field() -> None:
    resp = client.post("/api/grosssplit", json=_intangible_payload(cost_item="nan"))
    assert resp.status_code == 422
    body = response_body(resp)
    errors = as_records(body["errors"])
    finite = [e for e in errors if e["code"] == "NUMERIC_FIELD_NOT_FINITE"]
    assert finite
    assert finite[0]["message"] == "Cost must be a finite number."
    paths = [
        as_path(loc["path"]) for e in finite for loc in as_records(e["locations"])
    ]
    assert ["intangible", "base", "cost", 0] in paths


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_raw_nonfinite_body_is_422_and_json_safe(literal: str) -> None:
    raw = json.dumps(_intangible_payload(cost_item=1.0), allow_nan=True)
    raw = raw.replace("[1.0]", f"[{literal}]")
    resp = _raw_client.post(
        "/api/grosssplit",
        content=raw,
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422
    body = response_body(resp)
    assert body["kind"] == "input_validation"
    errors = as_records(body["errors"])
    assert any(e["code"] == "NUMERIC_FIELD_NOT_FINITE" for e in errors)
    detail = as_records(body["detail"])
    finite_inputs = [
        entry["input"] for entry in detail if entry.get("type") == "finite_number"
    ]
    assert finite_inputs
    assert all(isinstance(value, str) for value in finite_inputs)
    # The envelope must be strict-JSON serializable despite the nonfinite input.
    assert json.dumps(body, allow_nan=False)


def test_issue_context_excludes_raw_objects_and_keeps_entered_string() -> None:
    resp = client.post(
        "/api/grosssplit", json=_intangible_payload(cost_item="bad-number")
    )
    body = response_body(resp)
    errors = as_records(body["errors"])
    for error in errors:
        context = cast("dict[str, object]", error["context"])
        assert json.dumps(context, allow_nan=False)
    numeric = [e for e in errors if e["code"] == "NUMERIC_FIELD_NOT_A_NUMBER"]
    context = cast("dict[str, object]", numeric[0]["context"])
    technical = cast("dict[str, object]", context["technical_detail"])
    assert technical["input"] == "bad-number"
    assert technical["type"] == "float_parsing"


def test_context_does_not_embed_raw_payload_objects() -> None:
    resp = client.post("/api/grosssplit", json={"setup": "not-an-object"})
    body = response_body(resp)
    errors = as_records(body["errors"])
    for error in errors:
        context = cast("dict[str, object]", error["context"])
        technical = context.get("technical_detail")
        if isinstance(technical, dict):
            embedded = cast("dict[str, object]", technical)["input"]
            assert isinstance(embedded, (str, int, float, bool)) or embedded is None


# ── Request reference ───────────────────────────────────────────────


def test_inbound_request_id_is_not_trusted() -> None:
    resp = client.post(
        "/api/grosssplit",
        json={"setup": "bad"},
        headers={"X-Request-ID": "spoofed-value"},
    )
    body = response_body(resp)
    assert body["request_id"] != "spoofed-value"
    assert resp.headers["X-Request-ID"] == body["request_id"]


def test_spoofed_request_id_ignored_on_validation_success() -> None:
    resp = client.post(
        "/api/grosssplit/validate",
        json=_clean_gs_payload(),
        headers={"X-Request-ID": "spoofed-value"},
    )
    assert resp.status_code == 200
    body = response_body(resp)
    assert body["request_id"] != "spoofed-value"
    assert resp.headers["X-Request-ID"] == body["request_id"]


def test_internal_failure_reference_is_server_generated(monkeypatch: pytest.MonkeyPatch) -> None:
    import pyscnomics.api.router as router

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected internal failure")

    monkeypatch.setattr(router, "get_grosssplit", _boom)
    resp = client.post(
        "/api/grosssplit",
        json=_clean_gs_payload(),
        headers={"Origin": _ORIGIN, "X-Request-ID": "spoofed-value"},
    )
    assert resp.status_code == 500
    body = response_body(resp)
    assert body["kind"] == "server_error"
    assert body["validation_complete"] is False
    assert body["request_id"] != "spoofed-value"
    assert resp.headers["X-Request-ID"] == body["request_id"]
    assert resp.headers["access-control-allow-origin"] == _ORIGIN
    assert "Traceback" not in json.dumps(body)


def test_missing_route_returns_plain_404() -> None:
    resp = _raw_client.get("/api/no-such-endpoint")
    assert resp.status_code == 404
    assert "schema_version" not in str(resp.json())


def test_structural_422_exposes_request_id_to_allowed_origin() -> None:
    resp = client.post(
        "/api/grosssplit", json={"setup": "bad"}, headers={"Origin": _ORIGIN}
    )
    assert resp.status_code == 422
    assert resp.headers["access-control-allow-origin"] == _ORIGIN
    assert "X-Request-ID" in resp.headers["access-control-expose-headers"]


def test_domain_422_detail_carries_first_message_and_guidance() -> None:
    resp = client.post("/api/grosssplit", json=_gs_pod1_payload())
    assert resp.status_code == 422
    body = response_body(resp)
    detail = str(body["detail"])
    assert detail.startswith("PyscErr:")
    errors = as_records(body["errors"])
    assert str(errors[0]["message"]) in detail
    assert "select Check again" in detail


# ── Numeric request-model finite configuration ──────────────────────


def _lifting_payload(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "start_year": 2020,
        "end_year": 2030,
        "lifting_rate": [1.0],
        "price": [1.0],
        "prod_year": [2025],
        "fluid_type": "Oil",
    }
    base.update(overrides)
    return base


def _capital_payload(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "start_year": 2019,
        "end_year": 2050,
        "expense_year": [2026],
        "cost": [1.0],
    }
    base.update(overrides)
    return base


def _rpd_payload(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "year_rampup": 2,
        "drate": 0.1,
        "q_plateau_ratio": 0.2,
        "q_min_ratio": 0.05,
        "volume": 1.0,
        "start_year": 2020,
        "end_year": 2030,
    }
    base.update(overrides)
    return base


def _uncertainty_payload(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "run_number": 100,
        "oil_price_distribution": "Normal",
        "gas_price_distribution": "Normal",
        "opex_distribution": "Normal",
        "capex_distribution": "Normal",
        "lifting_distribution": "Normal",
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize("value", _NONFINITE)
def test_lifting_rejects_nonfinite(value: object) -> None:
    with pytest.raises(ValidationError):
        _ = LiftingBM.model_validate(_lifting_payload(price=[value]))


@pytest.mark.parametrize("value", _NONFINITE)
def test_cost_model_rejects_nonfinite(value: object) -> None:
    with pytest.raises(ValidationError):
        _ = CapitalCostBM.model_validate(_capital_payload(cost=[value]))


@pytest.mark.parametrize("value", _NONFINITE)
def test_summary_rejects_nonfinite_discount_rate(value: object) -> None:
    with pytest.raises(ValidationError):
        _ = SummaryArgumentsBM.model_validate({"discount_rate": value})


@pytest.mark.parametrize("value", _NONFINITE)
def test_fiscal_rejects_nonfinite_decline_factor(value: object) -> None:
    with pytest.raises(ValidationError):
        _ = ContractArgumentsBM.model_validate({"decline_factor": value})


@pytest.mark.parametrize("value", _NONFINITE)
def test_analysis_rejects_nonfinite(value: object) -> None:
    with pytest.raises(ValidationError):
        _ = SensitivityBM.model_validate({"min_deviation": value, "max_deviation": 2.0})
    with pytest.raises(ValidationError):
        _ = RpdBM.model_validate(_rpd_payload(drate=value))
    with pytest.raises(ValidationError):
        _ = UncertaintyBM.model_validate(_uncertainty_payload(oil_price_stddev=value))


def test_ltp_and_econlimit_reject_nonfinite() -> None:
    with pytest.raises(ValidationError):
        _ = LtpBM.model_validate(
            {"volume": float("inf"), "start_year": 2020, "end_year": 2030, "fluid_type": "Oil"}
        )
    with pytest.raises(ValidationError):
        _ = EconLimit.model_validate(
            {"years": [2020], "cash_flow": [float("nan")], "method": "x"}
        )


def test_setup_rejects_nonfinite_approval_year() -> None:
    with pytest.raises(ValidationError):
        _ = SetupBM.model_validate({"approval_year": float("nan")})


def test_ordinary_numeric_strings_and_zero_are_preserved() -> None:
    lifting = LiftingBM.model_validate(_lifting_payload(price=["1.5", 2, "0"]))
    assert lifting.price == [1.5, 2.0, 0.0]
    cost = CapitalCostBM.model_validate(_capital_payload(cost=["0", 0, "2.5"]))
    assert cost.cost == [0.0, 0.0, 2.5]
    summary = SummaryArgumentsBM.model_validate({"discount_rate": "0.12"})
    assert summary.discount_rate == 0.12


def test_nullable_defaults_remain_valid() -> None:
    summary = SummaryArgumentsBM.model_validate({})
    assert summary.discount_rate == 0.1
    cost = CapitalCostBM.model_validate(_capital_payload(cost=None))
    assert cost.cost is None
    rpd = RpdBM.model_validate(_rpd_payload())
    assert rpd.drate == 0.1
