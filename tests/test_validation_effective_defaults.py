"""T4 regressions: accepted nullable onstream defaults use the engine's shared
resolver, so preflight and execution agree and economics are unchanged (F7)."""

from __future__ import annotations

import copy
from typing import cast

import pytest

from pyscnomics.validation_rules import (
    OnstreamInputs,
    OnstreamResolutionError,
    resolve_onstream_year,
)
from tests._preflight_fixtures import (
    baseproject_payload,
    corrected_payload,
    transition_combinations,
    transition_payload_for,
)
from tests._validation_api_support import (
    as_path,
    as_records,
    client,
    is_record,
    response_body,
)


def _setup(payload: dict[str, object]) -> dict[str, object]:
    setup = payload["setup"]
    assert is_record(setup)
    return setup


def _strip_datetime(value: object) -> object:
    if isinstance(value, dict):
        mapping = cast("dict[str, object]", value)
        return {
            key: _strip_datetime(item)
            for key, item in mapping.items()
            if key != "execution_datetime"
        }
    if isinstance(value, list):
        items = cast("list[object]", value)
        return [_strip_datetime(item) for item in items]
    return value


@pytest.mark.parametrize(
    ("oil", "gas"),
    [
        (None, None),
        (None, "01/01/2029"),
        ("01/01/2029", None),
    ],
)
def test_nullable_onstream_preflight_execution_parity(oil: str | None, gas: str | None) -> None:
    baseline = corrected_payload()
    baseline_direct = client.post("/api/grosssplit", json=baseline)
    assert baseline_direct.status_code == 200

    payload = copy.deepcopy(corrected_payload())
    setup = _setup(payload)
    setup["oil_onstream_date"] = oil
    setup["gas_onstream_date"] = gas

    preflight = client.post("/api/grosssplit/validate", json=payload)
    assert preflight.status_code == 200
    report = response_body(preflight)
    assert report["validation_complete"] is True
    assert report["can_calculate"] is True
    assert report["skipped_checks"] == []
    assert report["errors"] == []

    direct = client.post("/api/grosssplit", json=payload)
    assert direct.status_code == 200
    assert _strip_datetime(response_body(direct)) == _strip_datetime(response_body(baseline_direct))


@pytest.mark.parametrize(("label", "scheme_1", "scheme_2"), transition_combinations())
def test_transition_nullable_onstream_is_complete(label: str, scheme_1: str, scheme_2: str) -> None:
    payload = transition_payload_for(scheme_1, scheme_2)
    preflight = client.post("/api/transition/validate", json=payload)
    assert preflight.status_code == 200, label
    report = response_body(preflight)
    assert report["validation_complete"] is True, label
    assert report["can_calculate"] is True, label
    assert report["skipped_checks"] == [], label


def test_effective_onstream_uses_engine_revenue() -> None:
    # Positive revenue, no provided date -> earliest positive-revenue year.
    resolved = resolve_onstream_year(
        fluid="gas",
        provided_year=None,
        start_year=2020,
        end_year=2030,
        revenue=(0.0, 0.0, 5.0, 1.0),
        lifting_prod_years=(2022,),
    )
    assert resolved.year == 2022
    assert resolved.source == "revenue"

    # A supplied date equal to the earliest lifting/revenue year is accepted.
    provided = resolve_onstream_year(
        fluid="gas",
        provided_year=2022,
        start_year=2020,
        end_year=2030,
        revenue=(0.0, 0.0, 5.0, 1.0),
        lifting_prod_years=(2022,),
    )
    assert provided.year == 2022
    assert provided.source == "provided"

    # No positive revenue -> the project-end fallback (never the first lifting).
    fallback = resolve_onstream_year(
        fluid="oil",
        provided_year=None,
        start_year=2020,
        end_year=2030,
        revenue=(0.0, 0.0, 0.0),
        lifting_prod_years=(2022,),
    )
    assert fallback.year == 2030
    assert fallback.source == "project_end"

    # A provided year outside the project is a located error.
    with pytest.raises(OnstreamResolutionError) as excinfo:
        _ = resolve_onstream_year(
            fluid="oil",
            provided_year=2099,
            start_year=2020,
            end_year=2030,
            revenue=(1.0,),
            lifting_prod_years=(2020,),
        )
    assert excinfo.value.code == "ONSTREAM_YEAR_OUTSIDE_PROJECT"


def test_onstream_inputs_is_immutable() -> None:
    inputs = OnstreamInputs(revenue=(1.0, 2.0), lifting_prod_years=(2020, 2021))
    assert inputs.revenue == (1.0, 2.0)
    assert inputs.lifting_prod_years == (2020, 2021)


def _error_paths(body: dict[str, object]) -> list[list[str | int]]:
    paths: list[list[str | int]] = []
    for issue in as_records(body["errors"]):
        locations = issue["locations"]
        assert isinstance(locations, list)
        for loc in cast("list[object]", locations):
            if is_record(loc):
                paths.append(as_path(loc["path"]))
    return paths


def _find_error(body: dict[str, object], code: str) -> dict[str, object]:
    for issue in as_records(body["errors"]):
        if issue["code"] == code:
            return issue
    raise AssertionError(f"{code} not found in {body.get('errors')!r}")


@pytest.mark.parametrize("strict", [False, True])
def test_malformed_nonnull_onstream_is_not_nullable_default(strict: bool) -> None:
    payload = corrected_payload()
    setup = _setup(payload)
    setup["gas_onstream_date"] = "abc"
    if strict:
        setup["is_strict"] = True

    preflight = client.post("/api/grosssplit/validate", json=payload)
    assert preflight.status_code == 200
    report = response_body(preflight)
    assert report["can_calculate"] is False
    assert report["validation_complete"] is False
    issue = _find_error(report, "ONSTREAM_DATE_INVALID")
    assert "abc" in str(issue["message"])
    assert "01/01/2029" in str(issue["message"])
    assert ["setup", "gas_onstream_date"] in _error_paths(report)
    # Dependent checks are disclosed, not silently certified.
    assert report["skipped_checks"]

    direct = client.post("/api/grosssplit", json=payload)
    assert direct.status_code == 422
    body = response_body(direct)
    assert body["kind"] == "input_validation"
    assert _find_error(body, "ONSTREAM_DATE_INVALID")
    # The submitted payload is never mutated.
    assert _setup(payload)["gas_onstream_date"] == "abc"


@pytest.mark.parametrize("oil", ["01/01/2029", None])
def test_supplied_onstream_uses_effective_engine_rule(oil: str | None) -> None:
    payload = corrected_payload()
    setup = _setup(payload)
    setup["oil_onstream_date"] = oil
    setup["gas_onstream_date"] = "01/01/2028"

    preflight = client.post("/api/grosssplit/validate", json=payload)
    assert preflight.status_code == 200
    report = response_body(preflight)
    assert report["can_calculate"] is False
    issue = _find_error(report, "ONSTREAM_YEAR_NOT_ALLOWED")
    assert ["setup", "gas_onstream_date"] in _error_paths(report)
    context = issue["context"]
    assert is_record(context)
    assert context["provided_year"] == 2028
    assert 2029 in cast("list[object]", context["allowed_years"])

    direct = client.post("/api/grosssplit", json=payload)
    assert direct.status_code == 422
    assert _find_error(response_body(direct), "ONSTREAM_YEAR_NOT_ALLOWED")


def _zero_revenue_postonstream_payload(strict: bool) -> dict[str, object]:
    payload = baseproject_payload()
    setup = _setup(payload)
    setup.update(
        {
            "start_date": "01/01/2022",
            "end_date": "31/12/2059",
            "oil_onstream_date": "01/01/2024",
            "gas_onstream_date": "01/01/2024",
            "approval_year": 2022,
            "is_pod_1": False,
            "is_strict": strict,
        }
    )
    # Keep the real accepted lifting shape; zero every price so the engine's
    # revenue is zero and the effective onstream falls back to the project end.
    lifting = payload["lifting"]
    assert is_record(lifting)
    for entry in lifting.values():
        if is_record(entry):
            price = entry.get("price")
            if isinstance(price, list):
                entry["price"] = [0.0 for _ in cast("list[object]", price)]
    payload["capital"] = {}
    payload["intangible"] = {}
    payload["opex"] = {
        "base": {
            "start_year": 2022,
            "end_year": 2059,
            "expense_year": [2025],
            "fixed_cost": [1000.0],
            "cost_allocation": ["Gas"],
            "cost_type": ["postonstream_cost"],
        }
    }
    payload["asr"] = {}
    payload["lbt"] = {}
    payload["cost_of_sales"] = {}
    return payload


def test_zero_revenue_supplied_dates_use_existing_end_year_fallback() -> None:
    strict_payload = _zero_revenue_postonstream_payload(strict=True)
    preflight = client.post("/api/baseproject/validate", json=strict_payload)
    assert preflight.status_code == 200
    report = response_body(preflight)
    assert report["can_calculate"] is False
    issue = _find_error(report, "COST_TYPE_PERIOD_MISMATCH")
    assert ["opex", "base", "cost_type", 0] in _error_paths(report)
    context = issue["context"]
    assert is_record(context)
    assert context["onstream_year"] == 2059

    direct = client.post("/api/baseproject", json=strict_payload)
    assert direct.status_code == 422
    body = response_body(direct)
    assert _find_error(body, "COST_TYPE_PERIOD_MISMATCH")
    assert ["opex", "base", "cost_type", 0] in _error_paths(body)

    # In loose mode the engine still rejects a post-onstream classification used
    # before the effective onstream year, so preflight and direct agree.
    loose_payload = _zero_revenue_postonstream_payload(strict=False)
    loose_preflight = client.post("/api/baseproject/validate", json=loose_payload)
    assert loose_preflight.status_code == 200
    loose_report = response_body(loose_preflight)
    assert loose_report["can_calculate"] is False
    assert ["opex", "base", "expense_year", 0] in _error_paths(loose_report)
    loose_direct = client.post("/api/baseproject", json=loose_payload)
    assert loose_direct.status_code == 422
    assert _error_paths(response_body(loose_direct))
