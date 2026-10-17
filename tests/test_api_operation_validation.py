"""T7 regressions: standalone LTP/RPD/econlimit input mistakes are located 422
input errors before numerical work, not generic 500s (F4)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, cast

import numpy as np
import pytest
from tests._validation_api_support import (
    as_path,
    as_records,
    client,
    is_record,
    response_body,
)


def _valid_ltp() -> dict[str, object]:
    return {"volume": 100.0, "start_year": 2025, "end_year": 2050, "fluid_type": "Oil"}


def _valid_rpd() -> dict[str, object]:
    return {
        "year_rampup": 1,
        "drate": 0.1,
        "q_plateau_ratio": 0.5,
        "q_min_ratio": 0.1,
        "volume": 100.0,
        "start_year": 2025,
        "end_year": 2060,
    }


def _errors(body: Mapping[str, object]) -> list[dict[str, object]]:
    return as_records(body["errors"])


def _located_codes(body: Mapping[str, object]) -> list[str]:
    return [str(issue["code"]) for issue in _errors(body)]


def _located_paths(body: Mapping[str, object]) -> list[list[str | int]]:
    paths: list[list[str | int]] = []
    for issue in _errors(body):
        locations = issue["locations"]
        assert isinstance(locations, list)
        for loc in cast("list[object]", locations):
            if is_record(loc):
                paths.append(as_path(loc["path"]))
    return paths


def test_reversed_ltp_rpd_period_returns_located_422() -> None:
    cases: list[tuple[str, dict[str, object], str]] = [
        (
            "/api/ltp",
            {**_valid_ltp(), "start_year": 2025, "end_year": 2024},
            "LTP_PERIOD_REVERSED",
        ),
        (
            "/api/rpd",
            {**_valid_rpd(), "start_year": 2025, "end_year": 2024},
            "RPD_PERIOD_REVERSED",
        ),
    ]
    for route, payload, code in cases:
        response = client.post(route, json=payload)
        assert response.status_code == 422, route
        body = response_body(response)
        assert body["kind"] == "input_validation"
        assert code in _located_codes(body)
        paths = _located_paths(body)
        assert ["start_year"] in paths
        assert ["end_year"] in paths

        issue = next(issue for issue in _errors(body) if issue["code"] == code)
        assert "End Year must be the same as or later than Start Year" in str(
            issue["message"]
        )


def test_valid_ltp_rpd_periods_remain_accepted() -> None:
    assert client.post("/api/ltp", json=_valid_ltp()).status_code == 200
    assert client.post("/api/rpd", json=_valid_rpd()).status_code == 200


def test_econlimit_empty_and_mismatched_series_return_422() -> None:
    empty: dict[str, object] = {
        "years": [],
        "cash_flow": [],
        "method": "Maximum Cumulative Cashflow",
    }
    response = client.post("/api/econlimit", json=empty)
    assert response.status_code == 422
    body = response_body(response)
    assert "ECONLIMIT_SERIES_EMPTY" in _located_codes(body)
    paths = _located_paths(body)
    assert ["years"] in paths
    assert ["cash_flow"] in paths

    mismatched: dict[str, object] = {
        "years": [2025],
        "cash_flow": [1.0, 2.0],
        "method": "Maximum Cumulative Cashflow",
    }
    response = client.post("/api/econlimit", json=mismatched)
    assert response.status_code == 422
    body = response_body(response)
    assert "ECONLIMIT_SERIES_LENGTH_MISMATCH" in _located_codes(body)
    issue = next(
        issue
        for issue in _errors(body)
        if issue["code"] == "ECONLIMIT_SERIES_LENGTH_MISMATCH"
    )
    assert "1 year entries" in str(issue["message"])
    assert "2 cash-flow entries" in str(issue["message"])


def test_econlimit_accepted_boundaries_remain_accepted() -> None:
    # A one-entry series with a zero and a negative cash flow is valid input.
    for cash in ([0.0], [-1.0], [1.0]):
        response = client.post(
            "/api/econlimit",
            json={
                "years": [2025],
                "cash_flow": cash,
                "method": "Maximum Cumulative Cashflow",
            },
        )
        assert response.status_code == 200, cash


def test_asr_lbt_input_guard_preserves_accepted_defaults() -> None:
    # Model-valid flat public requests reach the normalized workers and return
    # their exact successful zero response.
    asr_flat: dict[str, object] = {
        "asr": {
            "start_year": 2010,
            "end_year": 2010,
            "expense_year": [2010],
            "cost": [0.0],
            "cost_allocation": ["Oil"],
            "description": ["Zero ASR"],
        }
    }
    asr = client.post("/api/asr_expenditures", json=asr_flat)
    assert asr.status_code == 200
    assert response_body(asr) == {
        "oil_asr_expenditures": {"2010": 0.0},
        "gas_asr_expenditures": {"2010": 0.0},
    }

    lbt_flat: dict[str, object] = {
        "lbt": {
            "start_year": 2010,
            "end_year": 2010,
            "expense_year": [2010],
            "cost": [0.0],
            "cost_allocation": ["Oil"],
            "description": ["Zero LBT"],
        }
    }
    lbt = client.post("/api/lbt_expenditures", json=lbt_flat)
    assert lbt.status_code == 200
    assert response_body(lbt) == {
        "oil_lbt_expenditures": {"2010": 0.0},
        "gas_lbt_expenditures": {"2010": 0.0},
    }


_ExpenditureOperation = Literal["asr_expenditures", "lbt_expenditures"]

_EXPENDITURE_ROUTES: dict[str, str] = {
    "asr_expenditures": "/api/asr_expenditures",
    "lbt_expenditures": "/api/lbt_expenditures",
}

_ACCEPTED_EXPENDITURE_BLOCK: dict[str, object] = {
    "start_year": 2010,
    "end_year": 2012,
    "expense_year": [2011, 2012],
    "cost": [5.0, 7.0],
    "cost_allocation": ["Oil", "Gas"],
    "description": ["A", "B"],
}


def _expenditure_payload(
    operation: _ExpenditureOperation,
    block: dict[str, object],
    *,
    start_date: object = "01/01/2010",
    end_date: object = "31/12/2012",
) -> dict[str, object]:
    key = "asr" if operation == "asr_expenditures" else "lbt"
    return {"start_date": start_date, "end_date": end_date, key: block}


@pytest.mark.parametrize("operation", ("asr_expenditures", "lbt_expenditures"))
def test_expenditure_nonzero_accepted_exact_worker_parity(
    operation: _ExpenditureOperation,
) -> None:
    """A model-valid public nonzero request is accepted with exact parity.

    The public route must return HTTP 200 (never 422) together with the complete
    JSON-normalized worker result, so this fixture cannot silently pass as a
    rejected request.
    """
    import json

    from pyscnomics.api.adapter import get_asr_expenditures, get_lbt_expenditures
    from pyscnomics.api.converter import ASRExpendituresBM, LBTExpendituresBM
    from pyscnomics.api.validation_inputs import normalize_expenditure_request

    payload = _expenditure_payload(operation, dict(_ACCEPTED_EXPENDITURE_BLOCK))
    response = client.post(_EXPENDITURE_ROUTES[operation], json=payload)
    assert response.status_code == 200

    if operation == "asr_expenditures":
        request = ASRExpendituresBM.model_validate(payload).model_dump()
        expected = get_asr_expenditures(
            data=normalize_expenditure_request(request, operation)
        )
    else:
        request = LBTExpendituresBM.model_validate(payload).model_dump()
        expected = get_lbt_expenditures(
            data=normalize_expenditure_request(request, operation)
        )
    assert response_body(response) == json.loads(json.dumps(expected))


@pytest.mark.parametrize("operation", ("asr_expenditures", "lbt_expenditures"))
def test_expenditure_malformed_public_date_is_located(
    operation: _ExpenditureOperation,
) -> None:
    """A malformed public date is a located 422, not a generic 500."""
    payload = _expenditure_payload(
        operation, dict(_ACCEPTED_EXPENDITURE_BLOCK), start_date="abc"
    )
    response = client.post(_EXPENDITURE_ROUTES[operation], json=payload)
    assert response.status_code == 422
    body = response_body(response)
    assert body["kind"] == "input_validation"
    assert body["validation_complete"] is False
    issue = next(item for item in _errors(body) if item["code"] == "DATE_FORMAT_INVALID")
    assert ["start_date"] in _located_paths(body)
    assert "'abc'" in str(issue["message"])
    assert "01/01/2010" in str(issue["message"])
    assert str(issue["guidance"]).strip()
    skipped = as_records(body["skipped_checks"])
    assert skipped
    blocking: list[list[str | int]] = []
    for record in skipped:
        paths = record["blocking_paths"]
        assert isinstance(paths, list)
        for entry in cast("list[object]", paths):
            if is_record(entry):
                blocking.append(as_path(entry["path"]))
    assert ["start_date"] in blocking


@pytest.mark.parametrize("operation", ("asr_expenditures", "lbt_expenditures"))
def test_expenditure_known_invalid_public_paths(
    operation: _ExpenditureOperation,
) -> None:
    """Known public inconsistencies carry exact public paths and values."""
    section = "asr" if operation == "asr_expenditures" else "lbt"
    route = _EXPENDITURE_ROUTES[operation]

    outside = client.post(
        route,
        json=_expenditure_payload(
            operation,
            {**_ACCEPTED_EXPENDITURE_BLOCK, "expense_year": [2009, 2012]},
        ),
    )
    assert outside.status_code == 422
    body = response_body(outside)
    issue = next(
        item for item in _errors(body) if item["code"] == "EXPENSE_YEAR_OUTSIDE_PROJECT"
    )
    paths = _located_paths(body)
    assert [section, "expense_year", 0] in paths
    assert "2009" in str(issue["message"])
    assert "2010" in str(issue["message"]) and "2012" in str(issue["message"])
    assert str(issue["guidance"]).strip()

    mismatch = client.post(
        route,
        json=_expenditure_payload(
            operation, {**_ACCEPTED_EXPENDITURE_BLOCK, "cost": [5.0]}
        ),
    )
    assert mismatch.status_code == 422
    body = response_body(mismatch)
    issue = next(
        item for item in _errors(body) if item["code"] == "ARRAY_LENGTH_MISMATCH"
    )
    message = str(issue["message"])
    assert "Cost has 1 entry" in message
    assert "Year has 2 entries" in message
    assert "future_rate" not in message and "tax_portion" not in message
    paths = _located_paths(body)
    assert [section, "cost"] in paths
    assert [section, "expense_year"] in paths
    assert as_records(body["skipped_checks"])

    reversed_dates = client.post(
        route,
        json=_expenditure_payload(
            operation,
            dict(_ACCEPTED_EXPENDITURE_BLOCK),
            start_date="31/12/2012",
            end_date="01/01/2010",
        ),
    )
    assert reversed_dates.status_code == 422
    body = response_body(reversed_dates)
    assert "PROJECT_PERIOD_REVERSED" in _located_codes(body)
    assert ["start_date"] in _located_paths(body)


@pytest.mark.parametrize("operation", ("asr_expenditures", "lbt_expenditures"))
def test_expenditure_normalizer_preserves_input_and_nested_shape(
    operation: _ExpenditureOperation,
) -> None:
    """The guard reads a view; it never mutates the request or reshapes nested input."""
    import copy

    from pyscnomics.operation_validation import (
        collect_operation_issues,
        normalize_expenditure_input,
    )

    payload = _expenditure_payload(operation, dict(_ACCEPTED_EXPENDITURE_BLOCK))
    snapshot = copy.deepcopy(payload)
    normalized = normalize_expenditure_input(payload, operation)
    assert isinstance(normalized.get("setup"), dict)
    _ = collect_operation_issues(payload, operation)
    assert payload == snapshot

    section = "asr" if operation == "asr_expenditures" else "lbt"
    table = "ASR" if operation == "asr_expenditures" else "LBT"
    internal: dict[str, object] = {
        "setup": {
            "start_date": "01/01/2010",
            "end_date": "31/12/2012",
            "oil_onstream_date": None,
            "gas_onstream_date": None,
            "approval_year": None,
            "is_pod_1": False,
        },
        "lifting": None,
        "capital": None,
        "intangible": None,
        "opex": None,
        "asr": None,
        "lbt": None,
        "cost_of_sales": None,
    }
    internal[section] = {table: dict(_ACCEPTED_EXPENDITURE_BLOCK)}
    result = normalize_expenditure_input(internal, operation)
    assert result == internal
    issues = collect_operation_issues(internal, operation)
    assert issues == ()


@pytest.mark.parametrize("operation", ("asr_expenditures", "lbt_expenditures"))
def test_expenditure_accepted_defaults_and_int_dates(
    operation: _ExpenditureOperation,
) -> None:
    """Default zero requests and accepted integer date stamps stay accepted."""
    from datetime import datetime

    zero_block: dict[str, object] = {
        "start_year": 2010,
        "end_year": 2010,
        "expense_year": [2010],
        "cost": [0.0],
        "cost_allocation": ["Oil"],
        "description": ["Zero"],
    }
    response = client.post(
        _EXPENDITURE_ROUTES[operation],
        json=_expenditure_payload(
            operation, zero_block, start_date="01/01/2010", end_date="01/01/2010"
        ),
    )
    assert response.status_code == 200

    start_stamp = int(datetime(2010, 1, 1, 12, 0).timestamp())
    end_stamp = int(datetime(2012, 12, 30, 12, 0).timestamp())
    stamped = client.post(
        _EXPENDITURE_ROUTES[operation],
        json=_expenditure_payload(
            operation,
            dict(_ACCEPTED_EXPENDITURE_BLOCK),
            start_date=start_stamp,
            end_date=end_stamp,
        ),
    )
    assert stamped.status_code == 200, response_body(stamped)


def test_asr_wrong_nested_shape_is_structural_detail() -> None:
    # The old invented nested ``asr.base`` shape is not the public model; it must
    # stay a FastAPI structural 422 with a ``detail`` array, not be accepted.
    response = client.post(
        "/api/asr_expenditures",
        json={"asr": {"base": {"expense_year": [2010], "cost": [0.0]}}},
    )
    assert response.status_code == 422
    body = response_body(response)
    assert body["kind"] == "input_validation"
    assert body["detail"]


def test_asr_genuine_worker_failure_is_safe_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyscnomics.api.router as router

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("unexpected ASR fault")

    monkeypatch.setattr(router, "get_asr_expenditures", _boom)
    response = client.post(
        "/api/asr_expenditures",
        json={
            "asr": {
                "start_year": 2010,
                "end_year": 2010,
                "expense_year": [2010],
                "cost": [0.0],
                "cost_allocation": ["Oil"],
                "description": ["Zero ASR"],
            }
        },
    )
    assert response.status_code == 500
    assert response_body(response)["kind"] == "server_error"


def test_bad_ltp_fluid_is_located_before_the_worker() -> None:
    payload = {
        "volume": 100.0,
        "start_year": 2025,
        "end_year": 2050,
        "fluid_type": "bad",
    }
    response = client.post("/api/ltp", json=payload)
    assert response.status_code == 422
    body = response_body(response)
    assert body["kind"] == "input_validation"
    assert "LTP_FLUID_UNRECOGNIZED" in _located_codes(body)
    assert ["fluid_type"] in _located_paths(body)
    issue = next(
        item for item in _errors(body) if item["code"] == "LTP_FLUID_UNRECOGNIZED"
    )
    assert "bad" in str(issue["message"])
    assert "Oil" in str(issue["message"]) and "Gas" in str(issue["message"])

    for fluid in ("Oil", "Gas"):
        accepted = client.post(
            "/api/ltp", json={**payload, "fluid_type": fluid}
        )
        assert accepted.status_code == 200, fluid


def test_ltp_gas_mapped_lpg_fluids_are_accepted() -> None:
    """LPG propane/butane map to the gas LTP model, so the preflight must accept
    them exactly as the worker does instead of reporting an unrecognized fluid."""
    base = _valid_ltp()
    responses: dict[str, object] = {}
    for fluid in ("Gas", "LPG propane", "LPG butane"):
        accepted = client.post("/api/ltp", json={**base, "fluid_type": fluid})
        assert accepted.status_code == 200, fluid
        responses[fluid] = response_body(accepted)["ltp"]
    assert responses["LPG propane"] == responses["Gas"]
    assert responses["LPG butane"] == responses["Gas"]

    # A genuinely unsupported lifting fluid stays a located input failure.
    unsupported = client.post("/api/ltp", json={**base, "fluid_type": "Sulfur"})
    assert unsupported.status_code == 422
    assert "LTP_FLUID_UNRECOGNIZED" in _located_codes(response_body(unsupported))


def test_forecast_zero_prerequisite_is_located() -> None:
    """A zero forecast prerequisite is a located 422 with the actual relationship."""
    base: dict[str, object] = {
        "year_rampup": 2,
        "drate": 0.08,
        "q_plateau_ratio": 0.1,
        "q_min_ratio": 0.05,
        "volume": 100.0,
        "start_year": 2025,
        "end_year": 2060,
    }
    cases: list[tuple[str, float, str, list[str]]] = [
        ("drate", 0.0, "RPD_DECLINE_RATE_ZERO_PREREQUISITE", ["drate"]),
        (
            "q_plateau_ratio",
            0.0,
            "RPD_PLATEAU_RATIO_ZERO_PREREQUISITE",
            ["q_plateau_ratio"],
        ),
        ("q_min_ratio", 0.0, "RPD_MIN_RATIO_ZERO_PREREQUISITE", ["q_min_ratio"]),
        ("volume", 0.0, "RPD_VOLUME_ZERO_PREREQUISITE", ["volume"]),
    ]
    for field, zero, code, path in cases:
        response = client.post("/api/rpd", json={**base, field: zero})
        assert response.status_code == 422, field
        body = response_body(response)
        assert body["kind"] == "input_validation"
        issue = next(item for item in _errors(body) if item["code"] == code)
        assert path in _located_paths(body)
        message = str(issue["message"])
        assert "0" in message
        assert str(issue["guidance"]).strip()
        context = issue["context"]
        assert is_record(context)
        assert context["actual"] == 0.0

    ltp = {"volume": 0.0, "start_year": 2025, "end_year": 2060, "fluid_type": "Oil"}
    response = client.post("/api/ltp", json=ltp)
    assert response.status_code == 422
    body = response_body(response)
    issue = next(
        item for item in _errors(body) if item["code"] == "LTP_VOLUME_ZERO_PREREQUISITE"
    )
    assert ["volume"] in _located_paths(body)
    assert "0" in str(issue["message"])
    assert str(issue["guidance"]).strip()


def test_forecast_zero_prerequisite_stays_a_forecast_exception() -> None:
    """Direct library callers still catch the existing forecast superclass."""
    from pyscnomics.api.adapter import (
        LTPModelException,
        RDPModelException,
        get_ltp,
        get_rdp,
    )
    from pyscnomics.econ.selection import FluidType
    from pyscnomics.input_validation import InputValidationError

    try:
        _ = get_rdp(
            start_year=2025,
            end_year=2060,
            year_rampup=2,
            drate=0.0,
            q_plateau_ratio=0.1,
            q_min_ratio=0.05,
            volume=100.0,
        )
    except InputValidationError as exc:
        assert isinstance(exc, RDPModelException)
        assert exc.errors
        assert exc.errors[0].code == "RPD_DECLINE_RATE_ZERO_PREREQUISITE"
    else:  # pragma: no cover - the zero prerequisite must fail for this input
        raise AssertionError("get_rdp did not raise for a zero decline rate")

    try:
        _ = get_ltp(
            volume=0.0,
            start_year=2025,
            end_year=2060,
            fluid_type=FluidType.OIL,
        )
    except InputValidationError as exc:
        assert isinstance(exc, LTPModelException)
        assert exc.errors
        assert exc.errors[0].code == "LTP_VOLUME_ZERO_PREREQUISITE"
    else:  # pragma: no cover - the zero prerequisite must fail for this input
        raise AssertionError("get_ltp did not raise for a zero volume")


def test_forecast_negative_and_zero_boundaries_remain_accepted() -> None:
    """Only the demonstrated zero prerequisites change; negatives keep working."""
    base: dict[str, object] = {
        "year_rampup": 2,
        "drate": 0.08,
        "q_plateau_ratio": 0.1,
        "q_min_ratio": 0.05,
        "volume": 100.0,
        "start_year": 2025,
        "end_year": 2060,
    }
    for probe in (
        {"drate": -0.08},
        {"volume": -100.0},
        {"year_rampup": -1},
        {"year_rampup": 0},
    ):
        response = client.post("/api/rpd", json={**base, **probe})
        assert response.status_code == 200, probe

    accepted_ltp = client.post(
        "/api/ltp",
        json={
            "volume": 100.0,
            "start_year": 2025,
            "end_year": 2060,
            "fluid_type": "Oil",
        },
    )
    assert accepted_ltp.status_code == 200


def test_rpd_forecast_period_too_short_is_located() -> None:
    payload: dict[str, object] = {
        "year_rampup": 2,
        "drate": 0.08,
        "q_plateau_ratio": 0.1,
        "q_min_ratio": 0.05,
        "volume": 100.0,
        "start_year": 2025,
        "end_year": 2025,
    }
    response = client.post("/api/rpd", json=payload)
    assert response.status_code == 422
    body = response_body(response)
    assert body["kind"] == "input_validation"
    issue = next(
        item
        for item in _errors(body)
        if item["code"] == "RPD_FORECAST_PERIOD_TOO_SHORT"
    )
    assert ["end_year"] in _located_paths(body)
    context = issue["context"]
    assert is_record(context)
    required = context["required_end_year"]
    assert isinstance(required, int)
    assert "2025" in str(issue["message"])
    assert str(required) in str(issue["message"])

    # Ending one year before the required horizon still rejects.
    before = {**payload, "end_year": required - 1}
    assert client.post("/api/rpd", json=before).status_code == 422

    # Ending at the required horizon is accepted.
    at_required = {**payload, "end_year": required}
    assert client.post("/api/rpd", json=at_required).status_code == 200


def test_rpd_forecast_error_stays_a_rdp_modelexception() -> None:
    from pyscnomics.api.adapter import RDPModelException, get_rdp
    from pyscnomics.input_validation import InputValidationError

    try:
        _ = get_rdp(
            start_year=2025,
            end_year=2025,
            year_rampup=2,
            drate=0.08,
            q_plateau_ratio=0.1,
            q_min_ratio=0.05,
            volume=100.0,
        )
    except InputValidationError as exc:
        assert isinstance(exc, RDPModelException)
        assert exc.errors
        assert exc.errors[0].code == "RPD_FORECAST_PERIOD_TOO_SHORT"
    else:  # pragma: no cover - the horizon rule must fail for this input
        raise AssertionError("get_rdp did not raise for an infeasible period")


def test_genuine_worker_failure_is_a_safe_500(monkeypatch: pytest.MonkeyPatch) -> None:
    import pyscnomics.api.router as router

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("unexpected numerical fault")

    monkeypatch.setattr(router, "get_rpd_dict", _boom)
    response = client.post("/api/rpd", json=_valid_rpd())
    assert response.status_code == 500
    body = response_body(response)
    assert body["kind"] == "server_error"
    assert body["request_id"]
    assert "Traceback" not in str(body.get("detail", ""))


def test_forecast_numpy_zero_volume_is_located_and_compatible() -> None:
    """Supported NumPy zeros get the located prerequisite (A3/F4)."""
    from pyscnomics.api.adapter import (
        LTPModelException,
        RDPModelException,
        get_ltp,
        get_rdp,
    )
    from pyscnomics.econ.selection import FluidType
    from pyscnomics.input_validation import InputValidationError

    for zero in (np.int64(0), np.float32(0), np.array(0.0)):
        label = type(zero).__name__
        with pytest.raises(InputValidationError) as ltp_exc:
            _ = get_ltp(cast("float", cast("object", zero)), 2025, 2100, FluidType.OIL)
        assert isinstance(ltp_exc.value, LTPModelException), label
        assert ltp_exc.value.errors[0].code == "LTP_VOLUME_ZERO_PREREQUISITE", label
        assert ltp_exc.value.errors[0].locations[0].path == ("volume",), label

        with pytest.raises(InputValidationError) as rpd_exc:
            _ = get_rdp(
                2025, 2100, 2, 0.08, 0.1, 0.05, cast("float", cast("object", zero))
            )
        assert isinstance(rpd_exc.value, RDPModelException), label
        assert rpd_exc.value.errors[0].code == "RPD_VOLUME_ZERO_PREREQUISITE", label
        assert rpd_exc.value.errors[0].locations[0].path == ("volume",), label


def test_forecast_numpy_zero_variants_for_other_rpd_prerequisites() -> None:
    """Other existing RPD zero prerequisites also accept NumPy scalar zeros."""
    from pyscnomics.api.adapter import RDPModelException, get_rdp
    from pyscnomics.input_validation import InputValidationError

    base: dict[str, object] = {
        "year_rampup": 2,
        "drate": 0.08,
        "q_plateau_ratio": 0.1,
        "q_min_ratio": 0.05,
        "volume": 100.0,
    }
    cases = [
        ("drate", "RPD_DECLINE_RATE_ZERO_PREREQUISITE"),
        ("q_plateau_ratio", "RPD_PLATEAU_RATIO_ZERO_PREREQUISITE"),
        ("q_min_ratio", "RPD_MIN_RATIO_ZERO_PREREQUISITE"),
    ]
    for zero in (np.int64(0), np.float32(0), np.array(0.0)):
        for field, code in cases:
            payload = {**base, field: zero}
            with pytest.raises(InputValidationError) as exc:
                _ = get_rdp(
                    start_year=2025,
                    end_year=2100,
                    year_rampup=2,
                    drate=cast("float", payload["drate"]),
                    q_plateau_ratio=cast("float", payload["q_plateau_ratio"]),
                    q_min_ratio=cast("float", payload["q_min_ratio"]),
                    volume=cast("float", payload["volume"]),
                )
            assert isinstance(exc.value, RDPModelException), field
            assert exc.value.errors[0].code == code, field
            assert exc.value.errors[0].locations[0].path == (field,), field


def test_forecast_numpy_nonzero_volume_keeps_exact_parity() -> None:
    """Accepted NumPy nonzero volume keeps exact numerical output (A3)."""
    from pyscnomics.api.adapter import get_ltp, get_rdp
    from pyscnomics.econ.selection import FluidType

    base_ltp = get_ltp(100.0, 2025, 2100, FluidType.OIL)
    base_rdp = get_rdp(2025, 2100, 2, 0.08, 0.1, 0.05, 100.0)
    for value in (np.int64(100), np.float32(100), np.array(100.0)):
        label = type(value).__name__
        numeric = cast("float", cast("object", value))
        ltp = get_ltp(numeric, 2025, 2100, FluidType.OIL)
        rdp = get_rdp(2025, 2100, 2, 0.08, 0.1, 0.05, numeric)
        assert np.array_equal(ltp, base_ltp), label
        assert np.array_equal(rdp, base_rdp), label


def test_forecast_numpy_longdouble_is_located_and_accepted() -> None:
    """np.longdouble scalar/0-d zeros locate and 100s keep exact parity (A3)."""
    from pyscnomics.api.adapter import (
        LTPModelException,
        RDPModelException,
        get_ltp,
        get_rdp,
    )
    from pyscnomics.econ.selection import FluidType
    from pyscnomics.input_validation import InputValidationError

    for zero in (np.longdouble(0), np.array(np.longdouble(0))):
        label = f"{type(zero).__name__}:{getattr(zero, 'ndim', 'scalar')}"
        with pytest.raises(InputValidationError) as ltp_exc:
            _ = get_ltp(cast("float", cast("object", zero)), 2025, 2100, FluidType.OIL)
        assert isinstance(ltp_exc.value, LTPModelException), label
        assert ltp_exc.value.errors[0].code == "LTP_VOLUME_ZERO_PREREQUISITE", label
        assert ltp_exc.value.errors[0].locations[0].path == ("volume",), label

        with pytest.raises(InputValidationError) as rpd_exc:
            _ = get_rdp(
                2025, 2100, 2, 0.08, 0.1, 0.05, cast("float", cast("object", zero))
            )
        assert isinstance(rpd_exc.value, RDPModelException), label
        assert rpd_exc.value.errors[0].code == "RPD_VOLUME_ZERO_PREREQUISITE", label

    base_ltp = get_ltp(100.0, 2025, 2100, FluidType.OIL)
    base_rdp = get_rdp(2025, 2100, 2, 0.08, 0.1, 0.05, 100.0)
    for value in (np.longdouble(100), np.array(np.longdouble(100))):
        label = f"{type(value).__name__}"
        numeric = cast("float", cast("object", value))
        ltp = get_ltp(numeric, 2025, 2100, FluidType.OIL)
        rdp = get_rdp(2025, 2100, 2, 0.08, 0.1, 0.05, numeric)
        assert rdp.dtype == np.dtype(np.float64), label
        assert np.array_equal(ltp, base_ltp), label
        assert np.array_equal(rdp, base_rdp), label
