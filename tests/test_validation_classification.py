"""Classification, approval, PIS and CR/GS attribute parity tests (Task 2B).

Each rule here is exercised through the shared predicate, the real engine check
site (which must raise a specialized ``InputValidationError`` retaining its
domain superclass) and the API preflight. The real serialized fixtures are used
(copies of the fresh baselines); the originals are never edited.
"""

# The engine check sites are deliberately private; exercising them directly is
# the parity evidence this module exists to provide.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import copy
import json
import math
from datetime import date
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from pyscnomics.api.validation import validate_contract_input
from pyscnomics.contracts.costrecovery import (
    CostRecovery,
    CostRecoveryException,
    CostRecoveryInputError,
)
from pyscnomics.contracts.gross_split import (
    CumulativeProductionSplitException,
    CumulativeProductionSplitInputError,
    GrossSplit,
    GrossSplitException,
    GrossSplitInputError,
)
from pyscnomics.contracts.project import (
    BaseProject,
    BaseProjectException,
    BaseProjectInputError,
    PostOnstreamException,
    PostOnstreamInputError,
    PreOnstreamException,
    PreOnstreamInputError,
    SunkCostException,
    SunkCostInputError,
)
from pyscnomics.econ.costs import CapitalCost
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.selection import (
    CostType,
    FluidType,
    GrossSplitRegime,
    VariableSplit132024,
)
from pyscnomics.input_validation import InputValidationError, ValidationReport
from pyscnomics.validation_rules import (
    CLASSIFICATION_POSTONSTREAM,
    CLASSIFICATION_PREONSTREAM,
    CLASSIFICATION_SUNK,
    GS_VARIABLE_SPLIT_COMPONENTS_2017,
    GS_VARIABLE_SPLIT_COMPONENTS_2024,
    ContractType,
    approval_year_violation,
    boundary_year_classification_violation,
    classification_allowed_in_period,
    classification_period,
    collect_approval_year_issues,
    collect_capital_pis_issues,
    collect_cost_classification_issues,
    collect_costrecovery_attribute_issues,
    collect_costrecovery_capital_sunk_issues,
    collect_grosssplit_attribute_issues,
    collect_grosssplit_nonpod1_sunk_issues,
    collect_grosssplit_variable_split_issues,
    collect_postonstream_window_issues,
    collect_sunk_cutoff_issues,
    condition_dict_is_mixed_form,
    contains_negative,
    cum_split_offset_length_mismatch,
    default_cost_type_for_year,
    effective_classification,
    fraction_value_out_of_range,
    length_exceeds_project,
    length_mismatch,
    missing_variable_split_components,
    nonzero_indices,
    pis_after_onstream_indices,
    pis_before_onstream_indices,
    sunk_cost_cutoff_year,
    sunk_cost_year_exceeds_cutoff,
    variable_split_required_components,
)
from pyscnomics.validation_rules import ProjectBounds
from tests._validation_api_support import as_path, as_records, client, response_body

_FIXTURES = Path(__file__).parent / "fixtures" / "input_validation"


def _dict(value: object) -> dict[str, object]:
    return cast("dict[str, object]", value)


def _list(value: object) -> list[object]:
    return cast("list[object]", value)


def _set(table: dict[str, object], field: str, index: int, value: object) -> None:
    _list(table[field])[index] = value


def _fixture_payload(name: str) -> dict[str, object]:
    raw = cast(
        "dict[str, object]",
        json.loads((_FIXTURES / f"{name}.json").read_text()),
    )
    return copy.deepcopy(_dict(raw["payload"]))


def _report(
    payload: dict[str, object], contract_type: ContractType = "Gross Split"
) -> ValidationReport:
    return validate_contract_input(payload, contract_type, operation="validate")


def _codes(report: ValidationReport) -> set[str]:
    return {issue.code for issue in report.errors}


def _strict(payload: dict[str, object]) -> dict[str, object]:
    _dict(payload["setup"])["is_strict"] = True
    return payload


def _first_table(payload: dict[str, object], section: str) -> dict[str, object]:
    tables = _dict(payload[section])
    return _dict(tables[next(iter(tables))])


def _capital_table(payload: dict[str, object]) -> dict[str, object]:
    return _first_table(payload, "capital")


def _onstream(_fluid: str, year: int) -> date:
    return date(year, 1, 1)


def _lifting(fluid: FluidType, prod_year: int = 2028) -> Lifting:
    return Lifting(
        start_year=2023,
        end_year=2030,
        lifting_rate=np.array([1.0]),
        price=np.array([1.0]),
        prod_year=np.array([prod_year]),
        fluid_type=fluid,
    )


# ── Shared predicate primitives ─────────────────────────────────────────


def test_default_and_effective_classification_match_engine() -> None:
    assert default_cost_type_for_year(
        2020, approval_year=2024, onstream_year=2029, is_pod_1=True
    ) == CLASSIFICATION_SUNK
    assert default_cost_type_for_year(
        2020, approval_year=2024, onstream_year=2029, is_pod_1=False
    ) == CLASSIFICATION_PREONSTREAM
    assert default_cost_type_for_year(
        2026, approval_year=2024, onstream_year=2029, is_pod_1=True
    ) == CLASSIFICATION_PREONSTREAM
    assert default_cost_type_for_year(
        2030, approval_year=2024, onstream_year=2029, is_pod_1=True
    ) == CLASSIFICATION_POSTONSTREAM
    # An explicit classification is never overridden by the year default.
    assert (
        effective_classification(
            "postonstream_cost",
            expense_year=2020,
            approval_year=2024,
            onstream_year=2029,
            is_pod_1=True,
        )
        == CLASSIFICATION_POSTONSTREAM
    )
    # An unset classification receives the engine default.
    assert (
        effective_classification(
            None,
            expense_year=2020,
            approval_year=2024,
            onstream_year=2029,
            is_pod_1=True,
        )
        == CLASSIFICATION_SUNK
    )
    # A blank/None classification with no year stays unknown.
    assert (
        effective_classification(
            None,
            expense_year=None,
            approval_year=2024,
            onstream_year=2029,
            is_pod_1=True,
        )
        is None
    )


def test_classification_period_and_boundary_match_engine() -> None:
    assert classification_period(2020, approval_year=2024, onstream_year=2029) == CLASSIFICATION_SUNK
    assert classification_period(2026, approval_year=2024, onstream_year=2029) == CLASSIFICATION_PREONSTREAM
    assert classification_period(2030, approval_year=2024, onstream_year=2029) == CLASSIFICATION_POSTONSTREAM
    # The boundary years belong to no consistency period.
    assert classification_period(2024, approval_year=2024, onstream_year=2029) is None
    assert classification_period(2029, approval_year=2024, onstream_year=2029) is None
    assert classification_allowed_in_period(CLASSIFICATION_POSTONSTREAM, CLASSIFICATION_SUNK) is False
    assert classification_allowed_in_period(CLASSIFICATION_POSTONSTREAM, CLASSIFICATION_POSTONSTREAM) is True
    assert (
        boundary_year_classification_violation(
            2024,
            CLASSIFICATION_POSTONSTREAM,
            approval_year=2024,
            onstream_year=2029,
        )
        == "approval_year"
    )
    assert (
        boundary_year_classification_violation(
            2029,
            CLASSIFICATION_SUNK,
            approval_year=2024,
            onstream_year=2029,
        )
        == "onstream_year"
    )
    assert (
        boundary_year_classification_violation(
            2025,
            CLASSIFICATION_SUNK,
            approval_year=2024,
            onstream_year=2029,
        )
        is None
    )


def test_approval_sunk_and_pis_predicates_accept_numpy_and_zero() -> None:
    assert approval_year_violation(2020, start_year=2023, end_year=2030, onstream_year=2028) == "before_project"
    assert approval_year_violation(2031, start_year=2023, end_year=2030, onstream_year=2028) == "after_project"
    assert approval_year_violation(2029, start_year=2023, end_year=2030, onstream_year=2028) == "after_onstream"
    assert approval_year_violation(2025, start_year=2023, end_year=2030, onstream_year=2028) is None

    assert sunk_cost_cutoff_year(approval_year=np.int64(2024), onstream_year=np.int64(2029)) == 2024
    assert sunk_cost_year_exceeds_cutoff(np.int64(2025), 2024) is True
    assert sunk_cost_year_exceeds_cutoff(np.int64(2024), 2024) is False

    # PIS before onstream ignores zero-cost entries (a zero cost is never
    # amortized) and accepts NumPy scalars.
    before = pis_before_onstream_indices(
        np.array([2020, 2021, 2030]), np.array([0.0, 100.0, 5.0]), 2029
    )
    assert before == [1]
    assert pis_after_onstream_indices(np.array([2029, 2030, 2040]), 2029) == [1, 2]
    assert nonzero_indices(np.array([0.0, 1.0, -2.0])) == [1, 2]


def test_attribute_predicates_accept_zero_null_and_numpy() -> None:
    assert fraction_value_out_of_range(np.float64(0.5), 0.0, 1.0) is False
    assert fraction_value_out_of_range(np.float64(1.5), 0.0, 1.0) is True
    assert fraction_value_out_of_range(0.0, 0.0, 1.0) is False
    assert fraction_value_out_of_range(1.0, 0.0, 1.0) is False
    assert fraction_value_out_of_range(None, 0.0, 1.0) is False
    assert length_mismatch(2, 2) is False
    assert length_mismatch(3, 2) is True
    assert length_exceeds_project(2, 2) is False
    assert length_exceeds_project(3, 2) is True
    assert contains_negative([0.0, np.float64(0.0)]) is False
    assert contains_negative([0.0, np.float64(-1.0)]) is True
    assert condition_dict_is_mixed_form({"a": {"x": 1}, "b": [1]}) is True
    assert condition_dict_is_mixed_form({"a": {"x": 1}}) is False
    assert cum_split_offset_length_mismatch(2, 2) is False
    assert cum_split_offset_length_mismatch(3, 2) is True


# ── Engine: specialized errors retaining the domain superclass ──────────


def test_engine_approval_year_raises_specialized_error() -> None:
    with pytest.raises(BaseProjectInputError) as excinfo:
        _ = BaseProject(
            start_date=date(2023, 1, 1),
            end_date=date(2030, 12, 31),
            approval_year=2020,
        )
    assert isinstance(excinfo.value, BaseProjectException)
    assert isinstance(excinfo.value, InputValidationError)
    assert excinfo.value.errors[0].code == "APPROVAL_YEAR_OUTSIDE_PROJECT"


def test_engine_cost_type_consistency_raises_specialized_error() -> None:
    capital = CapitalCost(
        start_year=2023,
        end_year=2030,
        expense_year=np.array([2023]),
        cost=np.array([1.0]),
        cost_allocation=[FluidType.OIL],
        cost_type=[CostType.POST_ONSTREAM_COST],
    )
    with pytest.raises(BaseProjectInputError) as excinfo:
        _ = BaseProject(
            start_date=date(2023, 1, 1),
            end_date=date(2030, 12, 31),
            approval_year=2025,
            oil_onstream_date=_onstream("oil", 2028),
            gas_onstream_date=_onstream("gas", 2028),
            lifting=(_lifting(FluidType.OIL),),
            capital_cost=(capital,),
        )
    assert isinstance(excinfo.value, BaseProjectException)
    assert excinfo.value.errors[0].code == "COST_TYPE_PERIOD_MISMATCH"


def test_engine_boundary_year_raises_specialized_error() -> None:
    capital = CapitalCost(
        start_year=2023,
        end_year=2030,
        expense_year=np.array([2025]),
        cost=np.array([1.0]),
        cost_allocation=[FluidType.OIL],
        cost_type=[CostType.POST_ONSTREAM_COST],
    )
    with pytest.raises(BaseProjectInputError) as excinfo:
        _ = BaseProject(
            start_date=date(2023, 1, 1),
            end_date=date(2030, 12, 31),
            approval_year=2025,
            oil_onstream_date=_onstream("oil", 2028),
            gas_onstream_date=_onstream("gas", 2028),
            lifting=(_lifting(FluidType.OIL),),
            capital_cost=(capital,),
        )
    assert excinfo.value.errors[0].code == "COST_TYPE_POSTONSTREAM_AT_APPROVAL"


def test_engine_sunk_cutoff_strict_raises_specialized_error() -> None:
    capital = CapitalCost(
        start_year=2023,
        end_year=2030,
        expense_year=np.array([2026]),
        cost=np.array([1.0]),
        cost_allocation=[FluidType.OIL],
        cost_type=[CostType.SUNK_COST],
    )
    project = BaseProject(
        start_date=date(2023, 1, 1),
        end_date=date(2030, 12, 31),
        approval_year=2024,
        oil_onstream_date=_onstream("oil", 2029),
        gas_onstream_date=_onstream("gas", 2029),
        is_strict=True,
        lifting=(_lifting(FluidType.OIL, prod_year=2029),),
        capital_cost=(capital,),
    )
    with pytest.raises(SunkCostInputError) as excinfo:
        project._validate_sunkcost([project._oil_capital_sunk_cost])
    assert isinstance(excinfo.value, SunkCostException)
    assert isinstance(excinfo.value, InputValidationError)
    assert excinfo.value.errors[0].code == "SUNK_COST_AFTER_CUTOFF"


def test_engine_preonstream_strict_raises_specialized_error() -> None:
    project = BaseProject(
        start_date=date(2023, 1, 1),
        end_date=date(2030, 12, 31),
        approval_year=2024,
        oil_onstream_date=_onstream("oil", 2029),
        gas_onstream_date=_onstream("gas", 2029),
        is_strict=True,
        lifting=(_lifting(FluidType.OIL, prod_year=2029),),
    )
    capital = CapitalCost(
        start_year=2023,
        end_year=2030,
        expense_year=np.array([2030]),
        cost=np.array([1.0]),
        cost_allocation=[FluidType.OIL],
        cost_type=[CostType.PRE_ONSTREAM_COST],
    )
    with pytest.raises(PreOnstreamInputError) as excinfo:
        project._validate_preonstream([capital])
    assert isinstance(excinfo.value, PreOnstreamException)
    assert isinstance(excinfo.value, InputValidationError)
    assert excinfo.value.errors[0].code == "PREONSTREAM_YEAR_OUTSIDE_PROJECT"


def test_engine_sunk_cutoff_loose_records_warning() -> None:
    capital = CapitalCost(
        start_year=2023,
        end_year=2030,
        expense_year=np.array([2026]),
        cost=np.array([1.0]),
        cost_allocation=[FluidType.OIL],
        cost_type=[CostType.SUNK_COST],
    )
    project = BaseProject(
        start_date=date(2023, 1, 1),
        end_date=date(2030, 12, 31),
        approval_year=2024,
        oil_onstream_date=_onstream("oil", 2029),
        gas_onstream_date=_onstream("gas", 2029),
        is_strict=False,
        lifting=(_lifting(FluidType.OIL, prod_year=2029),),
        capital_cost=(capital,),
    )
    before = len(project.warning_messages)
    project._validate_sunkcost([project._oil_capital_sunk_cost])
    assert len(project.warning_messages) == before + 1


def test_engine_postonstream_raises_specialized_error() -> None:
    # Construct the project without this cost so the constructor-time
    # consistency check does not pre-empt the post-onstream rule; then drive the
    # real engine site with a post-onstream object whose year precedes onstream.
    project = BaseProject(
        start_date=date(2023, 1, 1),
        end_date=date(2030, 12, 31),
        approval_year=2024,
        oil_onstream_date=_onstream("oil", 2029),
        gas_onstream_date=_onstream("gas", 2029),
        lifting=(_lifting(FluidType.OIL, prod_year=2029),),
    )
    capital = CapitalCost(
        start_year=2023,
        end_year=2030,
        expense_year=np.array([2025]),
        cost=np.array([1.0]),
        cost_allocation=[FluidType.OIL],
        cost_type=[CostType.POST_ONSTREAM_COST],
    )
    with pytest.raises(PostOnstreamInputError) as excinfo:
        project._validate_postonstream([capital])
    assert isinstance(excinfo.value, PostOnstreamException)
    assert isinstance(excinfo.value, InputValidationError)
    assert excinfo.value.errors[0].code == "POSTONSTREAM_YEAR_BEFORE_ONSTREAM"


def test_engine_pis_before_and_after_raise_specialized_errors() -> None:
    contract = CostRecovery(
        start_date=date(2020, 1, 1),
        end_date=date(2025, 12, 31),
        approval_year=2020,
        is_strict=True,
    )
    capital = CapitalCost(
        start_year=2020,
        end_year=2025,
        expense_year=np.array([2020]),
        cost=np.array([100.0]),
        cost_allocation=[FluidType.OIL],
        cost_type=[CostType.PRE_ONSTREAM_COST],
        pis_year=np.array([2020]),
    )
    with pytest.raises(BaseProjectInputError) as excinfo:
        contract._check_capital_pis_years_before_onstream(
            obj_capital=capital,
            obj_name="oil_capital_preonstream",
            is_amortization=False,
            is_strict=True,
            onstream_year=2023,
        )
    assert excinfo.value.errors[0].code == "PIS_YEAR_BEFORE_ONSTREAM"

    capital_after = CapitalCost(
        start_year=2020,
        end_year=2025,
        expense_year=np.array([2020]),
        cost=np.array([100.0]),
        cost_allocation=[FluidType.OIL],
        cost_type=[CostType.PRE_ONSTREAM_COST],
        pis_year=np.array([2025]),
    )
    with pytest.raises(BaseProjectInputError) as excinfo2:
        contract._check_capital_pis_years_after_onstream(
            obj_capital=capital_after,
            obj_name="oil_capital_preonstream",
            is_strict=True,
            onstream_year=2023,
        )
    assert excinfo2.value.errors[0].code == "PIS_YEAR_AFTER_ONSTREAM"


def test_engine_cr_capital_sunk_and_attributes_raise_specialized_errors() -> None:
    contract = CostRecovery(
        start_date=date(2020, 1, 1),
        end_date=date(2025, 12, 31),
        approval_year=2020,
        is_strict=True,
    )
    capital = CapitalCost(
        start_year=2020,
        end_year=2025,
        expense_year=np.array([2020]),
        cost=np.array([100.0]),
        cost_allocation=[FluidType.OIL],
        cost_type=[CostType.SUNK_COST],
    )
    with pytest.raises(CostRecoveryInputError) as excinfo:
        contract._check_invalid_capital_sunk_costs(
            capital, "oil_capital_sunk_cost", is_strict=True, onstream_year=2023
        )
    assert isinstance(excinfo.value, CostRecoveryException)
    assert isinstance(excinfo.value, InputValidationError)
    assert excinfo.value.errors[0].code == "CR_CAPITAL_SUNK_COST"

    with pytest.raises(CostRecoveryInputError) as attr_exc:
        _ = CostRecovery(
            start_date=date(2020, 1, 1),
            end_date=date(2021, 12, 31),
            approval_year=2020,
            oil_ftp_portion=1.5,
        )
    assert attr_exc.value.errors[0].code == "CR_ATTRIBUTE_OUT_OF_RANGE"

    with pytest.raises(CostRecoveryInputError) as mixed_exc:
        _ = CostRecovery(
            start_date=date(2020, 1, 1),
            end_date=date(2021, 12, 31),
            approval_year=2020,
            condition_dict={  # pyright: ignore[reportArgumentType]
                "condition_1": {
                    "bot_limit": 0.0,
                    "top_limit": 1.0,
                    "ctr_oil": 0.1,
                    "ctr_gas": 0.2,
                },
                "condition_2": [1.0],
            },
        )
    assert mixed_exc.value.errors[0].code == "CR_CONDITION_DICT_MIXED_FORM"


def test_engine_gs_attributes_and_nonpod1_sunk_raise_specialized_errors() -> None:
    gross_split = GrossSplit(
        start_date=date(2020, 1, 1),
        end_date=date(2025, 12, 31),
        approval_year=2020,
        split_ministry_disc=1.5,
    )
    with pytest.raises(GrossSplitInputError) as excinfo:
        gross_split._check_attributes()
    assert isinstance(excinfo.value, GrossSplitException)
    assert isinstance(excinfo.value, InputValidationError)
    assert excinfo.value.errors[0].code == "GS_ATTRIBUTE_OUT_OF_RANGE"

    with pytest.raises(CumulativeProductionSplitInputError) as cum_exc:
        gross_split._check_cum_production_split_offset(np.array([1.0, 2.0]))
    assert isinstance(cum_exc.value, CumulativeProductionSplitException)
    assert cum_exc.value.errors[0].code == "GS_CUM_PRODUCTION_SPLIT_OFFSET_LENGTH"

    nonpod = GrossSplit(
        start_date=date(2020, 1, 1),
        end_date=date(2025, 12, 31),
        approval_year=2020,
        is_pod_1=False,
    )
    capital = CapitalCost(
        start_year=2020,
        end_year=2025,
        expense_year=np.array([2020]),
        cost=np.array([100.0]),
        cost_allocation=[FluidType.OIL],
        cost_type=[CostType.SUNK_COST],
    )
    with pytest.raises(GrossSplitInputError) as sunk_exc:
        nonpod._check_no_sunk_costs(
            sc_object=capital,
            sc_name="oil_capital_sunk_cost",
            is_capital=True,
            is_strict=True,
            onstream_year=2023,
        )
    assert sunk_exc.value.errors[0].code == "GS_NONPOD1_SUNK_COST"


# ── Preflight collectors ────────────────────────────────────────────────


def test_preflight_approval_year_after_onstream() -> None:
    payload = _strict(_fixture_payload("corrected"))
    _dict(payload["setup"])["approval_year"] = 2030
    report = _report(payload)
    assert "APPROVAL_YEAR_AFTER_ONSTREAM" in _codes(report)
    assert report.can_calculate is False


def test_preflight_cost_classification_boundary_and_period() -> None:
    payload = _strict(_fixture_payload("corrected"))
    row = _capital_table(payload)
    _set(row, "cost_type", 5, "postonstream_cost")  # year 2024 == approval year
    report = _report(payload)
    assert "COST_TYPE_POSTONSTREAM_AT_APPROVAL" in _codes(report)

    payload2 = _strict(_fixture_payload("corrected"))
    row2 = _first_table(payload2, "opex")
    _set(row2, "cost_type", 4, "sunk_cost")  # year 2029 == onstream year
    report2 = _report(payload2)
    assert "COST_TYPE_SUNK_AT_ONSTREAM" in _codes(report2)


def test_preflight_sunk_cutoff_and_postonstream_window() -> None:
    payload = _strict(_fixture_payload("corrected"))
    row = _capital_table(payload)
    _set(row, "cost_type", 6, "sunk_cost")  # year 2025 > cutoff 2024
    report = _report(payload)
    assert "SUNK_COST_AFTER_CUTOFF" in _codes(report)

    payload2 = _strict(_fixture_payload("corrected"))
    row2 = _first_table(payload2, "asr")
    _set(row2, "expense_year", 0, 2020)
    report2 = _report(payload2)
    assert "POSTONSTREAM_YEAR_BEFORE_ONSTREAM" in _codes(report2)


def test_preflight_pis_before_and_after_by_contract_class() -> None:
    # GS POD I applies both directions to capital rows.
    payload = _strict(_fixture_payload("corrected"))
    row = _capital_table(payload)
    _set(row, "pis_year", 0, 2020)
    assert "PIS_YEAR_BEFORE_ONSTREAM" in _codes(_report(payload))

    payload_after = _strict(_fixture_payload("corrected"))
    _set(_capital_table(payload_after), "pis_year", 0, 2040)
    assert "PIS_YEAR_AFTER_ONSTREAM" in _codes(_report(payload_after))

    # Cost Recovery only checks the before direction for capital rows.
    cr = _strict(_fixture_payload("corrected"))
    _dict(cr["setup"])["is_pod_1"] = False
    cr["costrecovery"] = {}
    _set(_capital_table(cr), "pis_year", 0, 2040)
    assert "PIS_YEAR_AFTER_ONSTREAM" not in _codes(_report(cr, "Cost Recovery"))


def test_preflight_cr_capital_sunk_and_gs_nonpod1_sunk() -> None:
    cr = _strict(_fixture_payload("corrected"))
    _dict(cr["setup"])["is_pod_1"] = False
    cr["costrecovery"] = {}
    _set(_capital_table(cr), "cost_type", 0, "sunk_cost")
    report = _report(cr, "Cost Recovery")
    assert "CR_CAPITAL_SUNK_COST" in _codes(report)

    gs = _strict(_fixture_payload("corrected"))
    _dict(gs["setup"])["is_pod_1"] = False
    _set(_capital_table(gs), "cost_type", 0, "sunk_cost")
    report2 = _report(gs, "Gross Split")
    assert "GS_NONPOD1_SUNK_COST" in _codes(report2)


def test_preflight_gs_pod1_noncapital_covers_sunk_and_preonstream() -> None:
    payload = _strict(_fixture_payload("corrected"))
    row = _first_table(payload, "intangible")
    row["cost_type"] = ["sunk_cost", "preonstream_cost", "preonstream_cost"]
    report = _report(payload)
    issue = next(e for e in report.errors if e.code == "GS_POD1_NONCAPITAL_INTANGIBLE")
    categories = {str(issue.context["cost_category"])}
    # Both non-capital classifications are permitted by the same rule.
    assert categories <= {"sunk_cost", "preonstream_cost"}


def test_preflight_aggregates_both_fluids_independently() -> None:
    payload = _strict(_fixture_payload("corrected"))
    row = _first_table(payload, "intangible")
    row["cost_allocation"] = ["Oil", "Oil", "Oil"]
    report = _report(payload)
    fluids = {
        str(e.context["fluid"])
        for e in report.errors
        if e.code == "GS_POD1_NONCAPITAL_INTANGIBLE"
    }
    assert fluids == {"Oil"}
    # The OPEX gas group is still reported independently.
    assert any(e.code == "GS_POD1_NONCAPITAL_OPEX" for e in report.errors)


def test_preflight_default_classification_is_effective() -> None:
    # A null capital classification before approval defaults to sunk for POD I;
    # a mismatch is only reported when the effective classification is invalid.
    payload = _strict(_fixture_payload("corrected"))
    row = _capital_table(payload)
    row["cost_type"] = [None] * len(_list(row["cost_type"]))
    report = _report(payload)
    # Default sunk/preonstream/postonstream assignments are consistent.
    assert "COST_TYPE_PERIOD_MISMATCH" not in _codes(report)


def test_preflight_explicit_classification_is_not_fabricated() -> None:
    # An explicit preonstream classification on a post-onstream year must be
    # reported, not silently replaced by the year default.
    payload = _strict(_fixture_payload("corrected"))
    row = _capital_table(payload)
    _set(row, "cost_type", 0, "postonstream_cost")  # year 2019 is the sunk period
    report = _report(payload)
    assert "COST_TYPE_PERIOD_MISMATCH" in _codes(report)


def test_preflight_uses_per_row_nonzero_cost() -> None:
    # The classification rules use each row's own cost, not a table sum: a
    # zero-cost row is excluded while a non-zero sibling is still reported.
    payload = _strict(_fixture_payload("corrected"))
    row = _first_table(payload, "intangible")
    row["cost"] = [0.0, 10.0, 20.0]
    report = _report(payload)
    issue = next(e for e in report.errors if e.code == "GS_POD1_NONCAPITAL_INTANGIBLE")
    indices = [loc.path[-1] for loc in issue.locations]
    assert 0 not in indices
    assert {1, 2} <= set(indices)


def test_preflight_opex_cost_is_normalized_fixed_plus_volume() -> None:
    payload = _strict(_fixture_payload("corrected"))
    row = _first_table(payload, "opex")
    # Zero fixed cost but non-zero production volume still counts as a cost.
    _set(row, "fixed_cost", 0, 0.0)
    _set(row, "prod_rate", 0, 10.0)
    _set(row, "cost_per_volume", 0, 5.0)
    report = _report(payload)
    issue = next(e for e in report.errors if e.code == "GS_POD1_NONCAPITAL_OPEX")
    assert 0 in [loc.path[-1] for loc in issue.locations]

    payload_zero = _strict(_fixture_payload("corrected"))
    row_zero = _first_table(payload_zero, "opex")
    _set(row_zero, "fixed_cost", 0, 0.0)
    _set(row_zero, "prod_rate", 0, 0.0)
    _set(row_zero, "cost_per_volume", 0, 0.0)
    report_zero = _report(payload_zero)
    issue_zero = next(
        e for e in report_zero.errors if e.code == "GS_POD1_NONCAPITAL_OPEX"
    )
    assert 0 not in [loc.path[-1] for loc in issue_zero.locations]


def test_preflight_invalid_prerequisite_skips_only_dependent_checks() -> None:
    payload = _strict(_fixture_payload("corrected"))
    # A genuinely invalid prerequisite (project end before project start) skips
    # the dependent project-period checks; a nullable date is engine-resolved (F7).
    _dict(payload["setup"])["end_date"] = "01/01/2018"
    report = _report(payload)
    # Checks that need valid project dates are disclosed as skipped.
    assert any(
        check.scope == "project_period_checks" for check in report.skipped_checks
    )
    assert report.validation_complete is False
    # The independent GS POD I classification rule still runs.
    assert any(
        issue.code == "GS_POD1_NONCAPITAL_INTANGIBLE" for issue in report.errors
    )


# ── Engine/preflight parity for the same rule ───────────────────────────


def test_approval_and_boundary_engine_preflight_parity() -> None:
    bounds = ProjectBounds(2023, 2030)
    payload = {
        "setup": {
            "start_date": "01/01/2023",
            "end_date": "31/12/2030",
            "oil_onstream_date": "01/01/2028",
            "gas_onstream_date": "01/01/2028",
            "approval_year": 2029,
            "is_pod_1": True,
            "is_strict": True,
        }
    }
    issues = collect_approval_year_issues(payload, bounds)
    assert [issue.code for issue in issues] == ["APPROVAL_YEAR_AFTER_ONSTREAM"]

    boundary_payload = {
        "setup": {
            "start_date": "01/01/2023",
            "end_date": "31/12/2030",
            "oil_onstream_date": "01/01/2028",
            "gas_onstream_date": "01/01/2028",
            "approval_year": 2025,
            "is_pod_1": True,
            "is_strict": True,
        },
        "capital": {
            "base": {
                "start_year": 2023,
                "end_year": 2030,
                "expense_year": [2025],
                "cost": [1.0],
                "cost_allocation": ["Oil"],
                "cost_type": ["postonstream_cost"],
            }
        },
    }
    boundary_issues = collect_cost_classification_issues(boundary_payload, bounds)
    assert "COST_TYPE_POSTONSTREAM_AT_APPROVAL" in {
        issue.code for issue in boundary_issues
    }


def test_sunk_and_postonstream_engine_preflight_parity() -> None:
    bounds = ProjectBounds(2023, 2030)
    setup = {
        "start_date": "01/01/2023",
        "end_date": "31/12/2030",
        "oil_onstream_date": "01/01/2029",
        "gas_onstream_date": "01/01/2029",
        "approval_year": 2024,
        "is_pod_1": True,
        "is_strict": True,
    }
    sunk_payload = {
        "setup": setup,
        "capital": {
            "base": {
                "start_year": 2023,
                "end_year": 2030,
                "expense_year": [2026],
                "cost": [1.0],
                "cost_allocation": ["Oil"],
                "cost_type": ["sunk_cost"],
            }
        },
    }
    assert {i.code for i in collect_sunk_cutoff_issues(sunk_payload, bounds)} == {
        "SUNK_COST_AFTER_CUTOFF"
    }
    post_payload = {
        "setup": setup,
        "capital": {
            "base": {
                "start_year": 2023,
                "end_year": 2030,
                "expense_year": [2025],
                "cost": [1.0],
                "cost_allocation": ["Oil"],
                "cost_type": ["postonstream_cost"],
            }
        },
    }
    assert {
        i.code for i in collect_postonstream_window_issues(post_payload, bounds)
    } == {"POSTONSTREAM_YEAR_BEFORE_ONSTREAM"}


def test_capital_pis_engine_preflight_parity() -> None:
    bounds = ProjectBounds(2023, 2030)
    payload = {
        "setup": {
            "start_date": "01/01/2023",
            "end_date": "31/12/2030",
            "oil_onstream_date": "01/01/2029",
            "gas_onstream_date": "01/01/2029",
            "approval_year": 2024,
            "is_pod_1": True,
            "is_strict": True,
        },
        "capital": {
            "base": {
                "start_year": 2023,
                "end_year": 2030,
                "expense_year": [2025],
                "cost": [1.0],
                "cost_allocation": ["Oil"],
                "cost_type": ["preonstream_cost"],
                "pis_year": [2020],
            }
        },
    }
    assert {i.code for i in collect_capital_pis_issues(payload, bounds, "Gross Split")} == {
        "PIS_YEAR_BEFORE_ONSTREAM"
    }


def test_cr_and_gs_attribute_preflight_parity() -> None:
    bounds = ProjectBounds(2020, 2021)
    cr_payload = {"costrecovery": {"oil_ftp_portion": 1.5}}
    assert {i.code for i in collect_costrecovery_attribute_issues(cr_payload, bounds)} == {
        "CR_ATTRIBUTE_OUT_OF_RANGE"
    }
    gs_payload = {
        "grosssplit": {"split_ministry_disc": 1.5},
        "contract_arguments": {"cum_production_split_offset": [0.0, 1.0, 2.0]},
    }
    codes = {i.code for i in collect_grosssplit_attribute_issues(gs_payload, bounds)}
    assert codes == {
        "GS_ATTRIBUTE_OUT_OF_RANGE",
        "GS_CUM_PRODUCTION_SPLIT_OFFSET_LENGTH",
    }

    # Condition-dict mixed form and CR carry-forward are preflight-covered too.
    mixed = {
        "costrecovery": {
            "condition_dict": {
                "condition_1": {
                    "bot_limit": 0.0,
                    "top_limit": 1.0,
                    "ctr_oil": 0.1,
                    "ctr_gas": 0.2,
                },
                "condition_2": [1.0],
            }
        }
    }
    assert {
        i.code for i in collect_costrecovery_attribute_issues(mixed, bounds)
    } == {"CR_CONDITION_DICT_MIXED_FORM"}
    cr_carry = {"costrecovery": {"oil_carry_forward_depreciation": [-1.0]}}
    assert {
        i.code for i in collect_costrecovery_attribute_issues(cr_carry, bounds)
    } == {"CR_CARRY_FORWARD_NEGATIVE"}

    # GS carry-forward length uses the project duration.
    gs_carry = {"grosssplit": {"oil_carry_forward_depreciation": [0.0, 0.0, 0.0]}}
    assert "GS_CARRY_FORWARD_TOO_LONG" in {
        i.code for i in collect_grosssplit_attribute_issues(gs_carry, bounds)
    }


def test_cr_and_gs_sunk_preflight_parity() -> None:
    bounds = ProjectBounds(2023, 2030)
    setup = {
        "start_date": "01/01/2023",
        "end_date": "31/12/2030",
        "oil_onstream_date": "01/01/2029",
        "gas_onstream_date": "01/01/2029",
        "approval_year": 2024,
        "is_pod_1": False,
        "is_strict": True,
    }
    payload = {
        "setup": setup,
        "capital": {
            "base": {
                "start_year": 2023,
                "end_year": 2030,
                "expense_year": [2020],
                "cost": [1.0],
                "cost_allocation": ["Oil"],
                "cost_type": ["sunk_cost"],
            }
        },
    }
    assert {
        i.code for i in collect_costrecovery_capital_sunk_issues(payload, bounds)
    } == {"CR_CAPITAL_SUNK_COST"}
    assert {
        i.code for i in collect_grosssplit_nonpod1_sunk_issues(payload, bounds)
    } == {"GS_NONPOD1_SUNK_COST"}


# ── Regressions: fresh serialized examples ──────────────────────────────


def test_corrected_fixture_strict_off_unchanged() -> None:
    report = _report(_fixture_payload("corrected"))
    assert report.errors == ()
    assert len(report.warnings) == 2
    assert report.can_calculate is True
    assert {w.code for w in report.warnings} == {
        "GS_POD1_NONCAPITAL_INTANGIBLE",
        "GS_POD1_NONCAPITAL_OPEX",
    }


def test_pod_fixture_strict_on_exactly_three_plus_four_rows() -> None:
    report = _report(_strict(_fixture_payload("pod")))
    by_code = {issue.code: issue for issue in report.errors}
    assert set(by_code) == {
        "GS_POD1_NONCAPITAL_INTANGIBLE",
        "GS_POD1_NONCAPITAL_OPEX",
    }
    assert len(by_code["GS_POD1_NONCAPITAL_INTANGIBLE"].locations) == 3
    assert len(by_code["GS_POD1_NONCAPITAL_OPEX"].locations) == 4
    # Every location is a distinct submitted row.
    paths = [
        loc.path for loc in by_code["GS_POD1_NONCAPITAL_OPEX"].locations
    ]
    assert len(set(paths)) == len(paths)


def test_sample_fixture_strict_on_three_plus_five_including_2029() -> None:
    report = _report(_strict(_fixture_payload("sample")))
    by_code = {issue.code: issue for issue in report.errors}
    assert set(by_code) == {
        "GS_POD1_NONCAPITAL_INTANGIBLE",
        "GS_POD1_NONCAPITAL_OPEX",
    }
    assert len(by_code["GS_POD1_NONCAPITAL_INTANGIBLE"].locations) == 3
    assert len(by_code["GS_POD1_NONCAPITAL_OPEX"].locations) == 5
    opex_paths = [
        loc.path for loc in by_code["GS_POD1_NONCAPITAL_OPEX"].locations
    ]
    assert any(path[-1] == 4 for path in opex_paths)


def test_corrected_strict_off_calculation_payload_is_not_blocked() -> None:
    payload = _fixture_payload("corrected")
    resp = client.post("/api/grosssplit", json=payload)
    assert resp.status_code == 200
    body = response_body(resp)
    # The financial summary is still returned unchanged for strict-off input.
    assert "gross_revenue" in body
    assert "ctr_net_share" in body


def test_corrected_strict_off_financial_fields_match_saved_response() -> None:
    raw = cast(
        "dict[str, object]",
        json.loads((_FIXTURES / "corrected.json").read_text()),
    )
    payload = _dict(raw["payload"])
    expected = _dict(raw["response"])
    resp = client.post("/api/grosssplit", json=payload)
    assert resp.status_code == 200
    body = response_body(resp)
    for key, expected_value in expected.items():
        if not isinstance(expected_value, (int, float)) or isinstance(expected_value, bool):
            continue
        actual = body.get(key)
        assert isinstance(actual, (int, float)), key
        assert math.isclose(float(actual), float(expected_value), rel_tol=1e-9, abs_tol=1e-9), key


def test_pod_strict_on_calculation_returns_structured_422() -> None:
    payload = _strict(_fixture_payload("pod"))
    resp = client.post("/api/grosssplit", json=payload)
    assert resp.status_code == 422
    body = response_body(resp)
    errors = as_records(body["errors"])
    codes = {str(error["code"]) for error in errors}
    assert codes == {
        "GS_POD1_NONCAPITAL_INTANGIBLE",
        "GS_POD1_NONCAPITAL_OPEX",
    }


# ── Gross Split variable-split completeness (G2 follow-up) ──────────────


def test_variable_split_required_components_per_regime() -> None:
    assert (
        variable_split_required_components("PERMEN_ESDM_13_2024", "conventional")
        == GS_VARIABLE_SPLIT_COMPONENTS_2024
    )
    assert (
        variable_split_required_components("PERMEN_ESDM_13_2024", "unconventional")
        == ()
    )
    assert (
        variable_split_required_components("PERMEN_ESDM_12_2020", "conventional")
        == GS_VARIABLE_SPLIT_COMPONENTS_2017
    )
    assert (
        variable_split_required_components("PERMEN_ESDM_8_2017", None)
        == GS_VARIABLE_SPLIT_COMPONENTS_2017
    )
    # An unrecognized regime has no shared requirement.
    assert variable_split_required_components(None, None) == ()
    assert variable_split_required_components("unknown", "conventional") == ()


def test_missing_variable_split_components_ignores_present_values() -> None:
    assert missing_variable_split_components(
        {"field_loc_2024": None, "infra_avail_2024": "available", "field_reserves_2024": "  "}
    ) == ["field_loc_2024", "field_reserves_2024"]
    assert missing_variable_split_components(
        {"field_loc_2024": "Onshore", "field_reserves_2024": "high"}
    ) == []


def test_collect_grosssplit_variable_split_issues_current_regulation() -> None:
    payload: dict[str, object] = {
        "contract_arguments": {
            "regime": "PERMEN_ESDM_13_2024",
            "reservoir_type_permen_2024": "conventional",
        },
        "grosssplit": {},
    }
    issues = collect_grosssplit_variable_split_issues(payload)
    assert {issue.code for issue in issues} == {"GS_VARIABLE_SPLIT_COMPONENT_MISSING"}
    assert {str(issue.context["field"]) for issue in issues} == set(
        GS_VARIABLE_SPLIT_COMPONENTS_2024
    )
    assert all(
        loc.path[:1] == ("grosssplit",)
        for issue in issues
        for loc in issue.locations
    )
    # A full block is clean.
    payload["grosssplit"] = {
        "field_loc_2024": "Onshore",
        "infra_avail_2024": "available",
        "field_reserves_2024": "high",
    }
    assert collect_grosssplit_variable_split_issues(payload) == ()
    # An unconventional reservoir needs no components.
    payload["grosssplit"] = {}
    payload["contract_arguments"] = {  # type: ignore[assignment]
        "regime": "PERMEN_ESDM_13_2024",
        "reservoir_type_permen_2024": "unconventional",
    }
    assert collect_grosssplit_variable_split_issues(payload) == ()


def test_engine_variable_split_completeness_raises_specialized_error() -> None:
    # ``field_reserves_2024`` is the component the adapter does not default; a
    # missing value must raise the specialized user-input error, not 500.
    gross_split = GrossSplit(
        start_date=date(2020, 1, 1),
        end_date=date(2025, 12, 31),
        approval_year=2020,
        field_reserves_2024=None,
    )
    with pytest.raises(GrossSplitInputError) as excinfo:
        gross_split._require_variable_split_components(
            regime=GrossSplitRegime.PERMEN_ESDM_13_2024,
            reservoir_type=VariableSplit132024.ReservoirType.MK,
        )
    assert isinstance(excinfo.value, GrossSplitException)
    assert isinstance(excinfo.value, InputValidationError)
    assert excinfo.value.errors[0].code == "GS_VARIABLE_SPLIT_COMPONENT_MISSING"
    assert "field_reserves_2024" in str(excinfo.value)


def test_http_empty_grosssplit_is_blocking_422() -> None:
    payload = _fixture_payload("corrected")
    payload["grosssplit"] = {}
    validate = client.post("/api/grosssplit/validate", json=payload)
    assert validate.status_code == 200
    body = response_body(validate)
    assert body["can_calculate"] is False
    assert body["validation_complete"] is False
    assert body["skipped_checks"]
    errors = as_records(body["errors"])
    assert {str(error["code"]) for error in errors} == {
        "GS_VARIABLE_SPLIT_COMPONENT_MISSING"
    }
    paths = [
        as_path(loc["path"])
        for error in errors
        for loc in as_records(error["locations"])
    ]
    assert paths
    assert all(path[:1] == ["grosssplit"] for path in paths)

    calc = client.post("/api/grosssplit", json=payload)
    assert calc.status_code == 422
    assert str(response_body(calc)["detail"]).startswith("PyscErr:")


def test_http_valid_gs_fixture_stays_200_and_financially_unchanged() -> None:
    raw = cast(
        "dict[str, object]",
        json.loads((_FIXTURES / "corrected.json").read_text()),
    )
    payload = _dict(raw["payload"])
    expected = _dict(raw["response"])
    validate = client.post("/api/grosssplit/validate", json=payload)
    assert validate.status_code == 200
    assert response_body(validate)["can_calculate"] is True
    calc = client.post("/api/grosssplit", json=payload)
    assert calc.status_code == 200
    body = response_body(calc)
    for key, expected_value in expected.items():
        if not isinstance(expected_value, (int, float)) or isinstance(
            expected_value, bool
        ):
            continue
        actual = body.get(key)
        assert isinstance(actual, (int, float)), key
        assert math.isclose(
            float(actual), float(expected_value), rel_tol=1e-9, abs_tol=1e-9
        ), key


def test_http_injected_internal_runtime_error_stays_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyscnomics.api.router as router

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected internal failure")

    monkeypatch.setattr(router, "get_grosssplit", _boom)
    # The valid fixture passes the preflight, so the injected internal failure
    # is what the calculation route reaches.
    resp = client.post("/api/grosssplit", json=_fixture_payload("corrected"))
    assert resp.status_code == 500
    body = response_body(resp)
    assert body["kind"] == "server_error"
    assert "Traceback" not in json.dumps(body)
