"""Structural, date and array parity tests (Task 2A).

Every shared predicate here is exercised both through the real engine
constructor/check site and through the API preflight, so the two cannot drift.
The real serialized fixtures are used (copies of the fresh baselines); the
originals are never edited.
"""

from __future__ import annotations

import copy
import json
from datetime import date
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from pyscnomics.api.validation import validate_contract_input
from pyscnomics.contracts.project import (
    BaseProject,
    BaseProjectException,
    BaseProjectInputError,
)
from pyscnomics.econ.costs import (
    ASR,
    CapitalCost,
    CapitalException,
    CapitalInputError,
    CostOfSales,
    Intangible,
    LBT,
    OPEX,
)
from pyscnomics.econ.revenue import Lifting, LiftingException, LiftingInputError
from pyscnomics.econ.selection import FluidType
from pyscnomics.input_validation import InputValidationError, ValidationReport
from pyscnomics.validation_rules import (
    OnstreamResolutionError,
    date_year,
    lifting_exceeds_production,
    parse_engine_date,
    period_is_reversed,
    resolve_onstream_year,
    unequal_lengths,
    year_above,
    year_below,
)

_FIXTURES = Path(__file__).parent / "fixtures" / "input_validation"


def _dict(value: object) -> dict[str, object]:
    return cast("dict[str, object]", value)


def _list(value: object) -> list[object]:
    return cast("list[object]", value)


def _corrected_payload() -> dict[str, object]:
    raw = cast(
        "dict[str, object]", json.loads((_FIXTURES / "corrected.json").read_text())
    )
    return copy.deepcopy(_dict(raw["payload"]))


def _report(payload: dict[str, object]) -> ValidationReport:
    return validate_contract_input(payload, "Gross Split", operation="validate")


def _codes(report: ValidationReport) -> set[str]:
    return {issue.code for issue in report.errors}


def _first_table(payload: dict[str, object], section: str) -> dict[str, object]:
    tables = _dict(payload[section])
    return _dict(tables[next(iter(tables))])


# ── Baseline parity ─────────────────────────────────────────────────────


def test_corrected_fixture_has_no_structural_errors() -> None:
    report = _report(_corrected_payload())
    assert _codes(report) == set()
    # Strict-off classification issues remain warnings and stay calculable.
    assert report.errors == ()
    assert len(report.warnings) == 2
    assert report.can_calculate is True


# ── Project period ──────────────────────────────────────────────────────


def test_bad_project_date_blocks_preflight() -> None:
    payload = _corrected_payload()
    _dict(payload["setup"])["start_date"] = "not-a-date"
    report = _report(payload)
    assert "DATE_FORMAT_INVALID" in _codes(report)
    assert report.can_calculate is False
    assert any(check.scope == "project_period_checks" for check in report.skipped_checks)


def test_reversed_project_dates_block_preflight() -> None:
    payload = _corrected_payload()
    setup = _dict(payload["setup"])
    setup["start_date"] = "31/12/2050"
    setup["end_date"] = "01/01/2019"
    report = _report(payload)
    assert "PROJECT_PERIOD_REVERSED" in _codes(report)
    assert report.can_calculate is False


def test_project_date_parser_uses_exact_format() -> None:
    # A malformed string must not yield a year scraped from arbitrary text.
    assert parse_engine_date("2090") is None
    assert parse_engine_date("Project 2090") is None
    assert parse_engine_date("01/01/2029") is not None
    assert date_year("01/01/2029") == 2029


def test_base_project_constructor_rejects_reversed_dates() -> None:
    with pytest.raises(BaseProjectInputError) as excinfo:
        _ = BaseProject(
            start_date=date(2032, 1, 1),
            end_date=date(2030, 12, 31),
            approval_year=2032,
        )
    assert isinstance(excinfo.value, BaseProjectException)
    assert isinstance(excinfo.value, InputValidationError)


# ── Cost component period and arrays ────────────────────────────────────


def test_array_length_mismatch_blocks_preflight_and_skips_dependent_checks() -> None:
    payload = _corrected_payload()
    row = _first_table(payload, "intangible")
    row["cost"] = [*_list(row["cost"]), 1.0]
    report = _report(payload)
    assert "ARRAY_LENGTH_MISMATCH" in _codes(report)
    assert report.can_calculate is False
    assert any("row_checks" in check.scope for check in report.skipped_checks)


def test_expense_year_outside_project_blocks_preflight() -> None:
    payload = _corrected_payload()
    _first_table(payload, "intangible")["expense_year"] = [2027, 2028, 2090]
    report = _report(payload)
    assert "EXPENSE_YEAR_OUTSIDE_PROJECT" in _codes(report)
    assert report.can_calculate is False


def test_nonfinite_numeric_blocks_preflight() -> None:
    payload = _corrected_payload()
    row = _first_table(payload, "intangible")
    costs = _list(row["cost"])
    costs[0] = float("nan")
    row["cost"] = costs
    report = _report(payload)
    assert "NUMERIC_FIELD_NOT_FINITE" in _codes(report)
    assert report.can_calculate is False


def test_capital_constructor_rejects_reversed_period() -> None:
    with pytest.raises(CapitalInputError) as excinfo:
        _ = CapitalCost(
            start_year=2030,
            end_year=2020,
            expense_year=np.array([2023]),
            cost=np.array([1.0]),
            cost_allocation=[FluidType.OIL],
        )
    assert isinstance(excinfo.value, CapitalException)
    assert isinstance(excinfo.value, InputValidationError)


def test_capital_constructor_rejects_unequal_lengths() -> None:
    with pytest.raises(CapitalInputError) as excinfo:
        _ = CapitalCost(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2023]),
            cost=np.array([1.0, 2.0]),
            cost_allocation=[FluidType.OIL],
        )
    assert isinstance(excinfo.value, CapitalException)


def test_capital_constructor_rejects_expense_year_outside_project() -> None:
    with pytest.raises(CapitalInputError):
        _ = CapitalCost(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2031]),
            cost=np.array([1.0]),
            cost_allocation=[FluidType.OIL],
        )


def test_asr_constructor_rejects_final_year_after_project() -> None:
    with pytest.raises(InputValidationError):
        _ = ASR(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2023]),
            cost=np.array([1.0]),
            cost_allocation=[FluidType.OIL],
            final_year=np.array([2035]),
        )


# ── Lifting: production years and lifting vs production ─────────────────


def test_production_year_outside_project_blocks_preflight() -> None:
    payload = _corrected_payload()
    entry = _dict(_dict(payload["lifting"])["GSA Gas Export #0"])
    prod_year = _list(entry["prod_year"])
    prod_year[-1] = 2090
    entry["prod_year"] = prod_year
    report = _report(payload)
    assert "PRODUCTION_YEAR_OUTSIDE_PROJECT" in _codes(report)
    assert report.can_calculate is False


def test_lifting_rate_exceeds_production_blocks_preflight() -> None:
    payload = _corrected_payload()
    entry = _dict(_dict(payload["lifting"])["GSA Gas Export #0"])
    lifting_rate = _list(entry["lifting_rate"])
    prod_rate = _list(entry["prod_rate"])
    prod_rate[0] = float(cast(float, lifting_rate[0])) / 2.0
    entry["prod_rate"] = prod_rate
    report = _report(payload)
    assert "LIFTING_RATE_EXCEEDS_PRODUCTION" in _codes(report)
    assert report.can_calculate is False


def test_lifting_constructor_rejects_lifting_exceeds_production() -> None:
    with pytest.raises(LiftingInputError) as excinfo:
        _ = Lifting(
            start_year=2023,
            end_year=2030,
            lifting_rate=np.array([10.0]),
            price=np.array([1.0]),
            prod_year=np.array([2023]),
            prod_rate=np.array([5.0]),
        )
    assert isinstance(excinfo.value, LiftingException)
    assert isinstance(excinfo.value, InputValidationError)


def test_lifting_constructor_rejects_production_year_outside_project() -> None:
    with pytest.raises(LiftingInputError):
        _ = Lifting(
            start_year=2023,
            end_year=2030,
            lifting_rate=np.array([10.0]),
            price=np.array([1.0]),
            prod_year=np.array([2031]),
        )


# ── Shared effective-onstream resolver ──────────────────────────────────


def test_resolver_infers_earliest_positive_revenue_year() -> None:
    resolved = resolve_onstream_year(
        fluid="oil",
        provided_year=None,
        start_year=2023,
        end_year=2030,
        revenue=[0.0, 0.0, 5.0, 0.0],
        lifting_prod_years=[2025, 2026],
    )
    assert resolved.year == 2025
    assert resolved.inferred is True
    assert resolved.source == "revenue"


def test_resolver_no_revenue_falls_back_to_project_end() -> None:
    resolved = resolve_onstream_year(
        fluid="gas",
        provided_year=2025,
        start_year=2023,
        end_year=2030,
        revenue=[0.0, 0.0, 0.0],
        lifting_prod_years=[2025],
    )
    assert resolved.year == 2030
    assert resolved.source == "project_end"


def test_resolver_rejects_disallowed_provided_year() -> None:
    with pytest.raises(OnstreamResolutionError) as excinfo:
        _ = resolve_onstream_year(
            fluid="oil",
            provided_year=2024,
            start_year=2023,
            end_year=2030,
            revenue=[0.0, 0.0, 5.0, 0.0],
            lifting_prod_years=[2025],
        )
    assert excinfo.value.code == "ONSTREAM_YEAR_NOT_ALLOWED"


def test_resolver_rejects_provided_year_outside_project() -> None:
    with pytest.raises(OnstreamResolutionError) as excinfo:
        _ = resolve_onstream_year(
            fluid="oil",
            provided_year=2035,
            start_year=2023,
            end_year=2030,
            revenue=[0.0, 0.0, 5.0, 0.0],
            lifting_prod_years=[2025],
        )
    assert excinfo.value.code == "ONSTREAM_YEAR_OUTSIDE_PROJECT"


def test_base_project_engine_uses_shared_resolver_for_nullable_onstream() -> None:
    oil = Lifting(
        start_year=2023,
        end_year=2030,
        lifting_rate=np.array([100.0, 100.0, 100.0]),
        price=np.array([10.0, 10.0, 10.0]),
        prod_year=np.array([2025, 2026, 2027]),
        fluid_type=FluidType.OIL,
    )
    project = BaseProject(
        start_date=date(2023, 1, 1),
        end_date=date(2030, 12, 31),
        approval_year=2023,
        oil_onstream_date=None,
        lifting=(oil,),
    )
    assert project.oil_onstream_date is not None
    assert project.oil_onstream_date.year == 2025


def test_base_project_no_revenue_falls_back_to_project_end() -> None:
    project = BaseProject(
        start_date=date(2023, 1, 1),
        end_date=date(2030, 12, 31),
        approval_year=2023,
    )
    assert project.oil_onstream_date == project.end_date
    assert project.gas_onstream_date == project.end_date


# ── Predicate primitives: zero/null/NumPy acceptance ────────────────────


def test_predicates_accept_numpy_scalars_and_arrays() -> None:
    assert period_is_reversed(np.int64(2023), np.int64(2030)) is False
    assert period_is_reversed(np.int64(2030), np.int64(2023)) is True
    assert year_below(np.array([2025, 2026]), 2023) is None
    assert year_above(np.array([2025, 2026]), 2030) is None
    assert unequal_lengths({"a": 3, "b": 3}) is None
    assert unequal_lengths({"a": 3, "b": 2}) == {"a": 3, "b": 2}


def test_lifting_exceeds_production_ignores_zero_and_missing() -> None:
    assert lifting_exceeds_production([0.0, 5.0], [0.0, 5.0]) is False
    assert lifting_exceeds_production([1.0], [0.5]) is True


def test_capital_accepts_zero_and_null_placeholders() -> None:
    cost = CapitalCost(
        start_year=2023,
        end_year=2030,
        expense_year=np.array([2023]),
        cost=np.array([0.0]),
        cost_allocation=[FluidType.OIL],
        salvage_value=np.array([None], dtype=object),
    )
    assert cost.cost is not None
    assert cost.cost.tolist() == [0.0]
    assert cost.salvage_value is not None
    assert cost.salvage_value.shape == (1,)


def test_intangible_opex_cost_of_sales_accept_zero_cost() -> None:
    for factory in (Intangible, CostOfSales):
        obj = factory(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2023]),
            cost=np.array([0.0]),
            cost_allocation=[FluidType.OIL],
        )
        assert obj.cost is not None
        assert obj.cost.tolist() == [0.0]
    opex = OPEX(
        start_year=2023,
        end_year=2030,
        expense_year=np.array([2023]),
        fixed_cost=np.array([0.0]),
        prod_rate=np.array([0.0]),
        cost_per_volume=np.array([0.0]),
        cost_allocation=[FluidType.OIL],
    )
    # OPEX cost is the engine's fixed + production * unit behavior.
    assert opex.cost is not None
    assert opex.cost.tolist() == [0.0]


def test_lbt_accepts_zero_area() -> None:
    lbt = LBT(
        start_year=2023,
        end_year=2030,
        expense_year=np.array([2023]),
        utilized_land_area=np.array([0.0]),
        utilized_building_area=np.array([0.0]),
        cost_allocation=[FluidType.OIL],
    )
    assert lbt.utilized_land_area is not None
    assert lbt.utilized_land_area.tolist() == [0.0]
