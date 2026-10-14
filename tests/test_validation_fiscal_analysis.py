"""Fiscal, summary and analysis input-validation parity tests (Task 2C).

Each owned rule is exercised through the shared predicate, the real engine check
site (which raises a specialized ``InputValidationError`` retaining its domain
superclass) and the API preflight. Engine check sites are deliberately private;
exercising them directly is the parity evidence this module exists to provide.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from pyscnomics.api.validation import validate_contract_input
from pyscnomics.econ.costs_tools import (
    TaxInflationException,
    TaxInflationInputError,
    get_cost_adjustment_by_inflation,
)
from pyscnomics.econ.depreciation import (
    DepreciationException,
    DepreciationInputError,
    UnitOfProductionException,
    UnitOfProductionInputError,
    psc_declining_balance_depreciation_rate,
    unit_of_production_rate,
)
from pyscnomics.econ.results import (
    CashFlow,
    CashFlowException,
    CashFlowInputError,
)
from pyscnomics.input_validation import InputValidationError, ValidationReport
from pyscnomics.optimize.optimization import (
    OptimizationException,
    OptimizationInputError,
    adjust_useful_life_years,
)
from pyscnomics.optimize.sensitivity import (
    SensitivityException,
    SensitivityInputError,
    get_sensitivity_multipliers,
)
from pyscnomics.optimize.uncertainty import (
    MonteCarloException,
    MonteCarloInputError,
    get_multipliers_montecarlo,
)
from pyscnomics.validation_rules import (
    ProjectBounds,
    adjustment_out_of_unit_range,
    collect_optimization_argument_issues,
    collect_sensitivity_argument_issues,
    collect_summary_argument_issues,
    collect_uncertainty_argument_issues,
    depreciation_factor_out_of_range,
    deviation_is_negative,
    discounting_mode_is_valid,
    increment_not_positive,
    npv_mode_is_valid,
    pis_year_appearance_count,
    reference_year_violation,
    run_number_not_positive,
    salvage_exceeds_cost,
    step_not_positive,
    uncertainty_distribution_is_valid,
    uncertainty_parameters_violation,
    useful_life_too_short,
    year_array_outside_project,
)

_BOUNDS = ProjectBounds(2019, 2030)


def _setup() -> dict[str, object]:
    return {
        "start_date": "01/01/2019",
        "end_date": "31/12/2030",
        "oil_onstream_date": "01/01/2025",
        "gas_onstream_date": "01/01/2025",
        "approval_year": 2020,
        "is_pod_1": True,
        "is_strict": True,
    }


def _report(payload: Mapping[str, object]) -> ValidationReport:
    return validate_contract_input(payload, "Base Project", operation="validate")


def _codes(report: ValidationReport) -> set[str]:
    return {issue.code for issue in report.errors}


# ── Shared predicate primitives ─────────────────────────────────────────


def test_npv_and_discounting_mode_predicates() -> None:
    for mode in (
        "SKK Full Cycle Real Terms",
        "SKK Full Cycle Nominal Terms",
        "Full Cycle Real Terms",
        "Full Cycle Nominal Terms",
        "Point Forward",
    ):
        assert npv_mode_is_valid(mode)
    assert not npv_mode_is_valid("zzz")
    assert not npv_mode_is_valid(None)
    assert discounting_mode_is_valid("End Year")
    assert discounting_mode_is_valid("Mid Year")
    assert not discounting_mode_is_valid("End Year Discounting")


@pytest.mark.parametrize(
    ("year", "expected"),
    [(2018, "before"), (2031, "after"), (2019, None), (2030, None), (2025, None)],
)
def test_reference_year_violation(year: int, expected: str | None) -> None:
    assert reference_year_violation(year, 2019, 2030) == expected


def test_year_array_outside_project() -> None:
    assert year_array_outside_project([2020, 2021], 2019, 2030) is None
    assert year_array_outside_project([2010, 2020], 2019, 2030) == "before"
    assert year_array_outside_project([2020, 2040], 2019, 2030) == "after"


def test_depreciation_and_pis_predicates() -> None:
    assert depreciation_factor_out_of_range(1.5)
    assert depreciation_factor_out_of_range(-0.1)
    assert not depreciation_factor_out_of_range(0.5)
    assert not depreciation_factor_out_of_range(None)
    assert salvage_exceeds_cost(200, 100)
    assert not salvage_exceeds_cost(50, 100)
    assert pis_year_appearance_count([2020, 2020, 2021], 2020) == 2
    assert pis_year_appearance_count([2020, 2021], 2022) == 0


def test_sensitivity_predicates() -> None:
    assert deviation_is_negative(-0.1)
    assert not deviation_is_negative(0.0)
    assert increment_not_positive(0.0)
    assert increment_not_positive(-1.0)
    assert not increment_not_positive(0.5)
    assert step_not_positive(0)
    assert not step_not_positive(10)


def test_uncertainty_predicates() -> None:
    assert run_number_not_positive(0)
    assert run_number_not_positive(-3)
    assert not run_number_not_positive(10)
    assert uncertainty_distribution_is_valid("Uniform")
    assert uncertainty_distribution_is_valid("lognormal")
    assert not uncertainty_distribution_is_valid("weibull")
    assert (
        uncertainty_parameters_violation(
            minimum=1.0, mean=1.0, maximum=1.0, std_dev=0.0, distribution="uniform"
        )
        is None
    )
    assert (
        uncertainty_parameters_violation(
            minimum=1.0, mean=0.0, maximum=2.0, std_dev=1.0, distribution="uniform"
        )
        == "zero_mean"
    )
    assert (
        uncertainty_parameters_violation(
            minimum=2.0, mean=1.0, maximum=1.0, std_dev=1.0, distribution="uniform"
        )
        == "range"
    )
    assert (
        uncertainty_parameters_violation(
            minimum=1.0, mean=2.0, maximum=3.0, std_dev=0.0, distribution="normal"
        )
        == "zero_std"
    )
    assert (
        uncertainty_parameters_violation(
            minimum=-1.0, mean=2.0, maximum=3.0, std_dev=1.0, distribution="lognormal"
        )
        == "lognormal_nonpositive"
    )
    assert (
        uncertainty_parameters_violation(
            minimum=1.0, mean=2.0, maximum=3.0, std_dev=1.0, distribution="weibull"
        )
        == "distribution"
    )


def test_optimization_predicates() -> None:
    assert adjustment_out_of_unit_range(1.5)
    assert adjustment_out_of_unit_range(-0.1)
    assert not adjustment_out_of_unit_range(0.5)
    assert useful_life_too_short(1)
    assert not useful_life_too_short(2)


# ── Engine conversions retain their domain superclass ───────────────────


def test_engine_uncertainty_zero_mean_is_user_input() -> None:
    with pytest.raises(MonteCarloInputError) as excinfo:
        _ = get_multipliers_montecarlo(
            run_number=10,
            distribution="uniform",
            min_value=1.0,
            mean_value=0.0,
            max_value=2.0,
            std_dev=1.0,
        )
    assert isinstance(excinfo.value, ValueError)
    assert isinstance(excinfo.value, MonteCarloException)
    assert isinstance(excinfo.value, InputValidationError)
    assert excinfo.value.issues[0].code == "UNCERTAINTY_ZERO_MEAN"


@pytest.mark.parametrize(
    ("distribution", "minimum", "mean", "maximum", "std_dev", "code"),
    [
        ("uniform", 2.0, 1.0, 1.0, 1.0, "UNCERTAINTY_RANGE_INVALID"),
        ("normal", 1.0, 2.0, 3.0, 0.0, "UNCERTAINTY_ZERO_STD"),
        (
            "lognormal",
            -1.0,
            2.0,
            3.0,
            1.0,
            "UNCERTAINTY_LOGNORMAL_NONPOSITIVE",
        ),
        ("weibull", 1.0, 2.0, 3.0, 1.0, "UNCERTAINTY_DISTRIBUTION_UNRECOGNIZED"),
    ],
)
def test_engine_uncertainty_violations(
    distribution: str,
    minimum: float,
    mean: float,
    maximum: float,
    std_dev: float,
    code: str,
) -> None:
    with pytest.raises(MonteCarloInputError) as excinfo:
        _ = get_multipliers_montecarlo(
            run_number=10,
            distribution=distribution,
            min_value=minimum,
            mean_value=mean,
            max_value=maximum,
            std_dev=std_dev,
        )
    assert excinfo.value.issues[0].code == code


def test_engine_sensitivity_deviation_is_user_input() -> None:
    with pytest.raises(SensitivityInputError) as excinfo:
        _ = get_sensitivity_multipliers(min_deviation=-0.1, max_deviation=0.2)
    assert isinstance(excinfo.value, SensitivityException)
    assert excinfo.value.issues[0].code == "SENSITIVITY_DEVIATION_NEGATIVE"


def test_engine_sensitivity_increment_and_step() -> None:
    with pytest.raises(SensitivityInputError) as excinfo:
        _ = get_sensitivity_multipliers(
            min_deviation=0.1, max_deviation=0.1, increment=0.0
        )
    assert excinfo.value.issues[0].code == "SENSITIVITY_INCREMENT_NOT_POSITIVE"

    with pytest.raises(SensitivityInputError) as excinfo:
        _ = get_sensitivity_multipliers(min_deviation=0.1, max_deviation=0.1, step=0)
    assert excinfo.value.issues[0].code == "SENSITIVITY_STEP_NOT_POSITIVE"


def test_engine_depreciation_factor_is_user_input() -> None:
    with pytest.raises(DepreciationInputError) as excinfo:
        _ = psc_declining_balance_depreciation_rate(
            cost=100.0, useful_life=5.0, depreciation_factor=2.0
        )
    assert isinstance(excinfo.value, DepreciationException)
    assert excinfo.value.issues[0].code == "DEPRECIATION_FACTOR_OUT_OF_RANGE"


def test_engine_unit_of_production_pis_not_unique() -> None:
    with pytest.raises(UnitOfProductionInputError) as excinfo:
        _ = unit_of_production_rate(
            project_years=np.array([2020, 2020, 2021]),
            prod=np.array([1.0, 1.0, 1.0]),
            cost=100.0,
            salvage_value=0.0,
            pis_year=2020,
        )
    assert isinstance(excinfo.value, UnitOfProductionException)
    assert excinfo.value.issues[0].code == "UOP_PIS_YEAR_NOT_UNIQUE"


def test_engine_unit_of_production_salvage_exceeds_cost() -> None:
    with pytest.raises(UnitOfProductionInputError) as excinfo:
        _ = unit_of_production_rate(
            project_years=np.array([2020, 2021]),
            prod=np.array([1.0, 1.0]),
            cost=100.0,
            salvage_value=200.0,
            pis_year=2020,
        )
    assert excinfo.value.issues[0].code == "UOP_SALVAGE_EXCEEDS_COST"


def test_engine_year_inflation_outside_project() -> None:
    with pytest.raises(TaxInflationInputError) as excinfo:
        _ = get_cost_adjustment_by_inflation(
            start_year=2020,
            end_year=2024,
            cost=np.array([100.0]),
            expense_year=np.array([2022]),
            project_years=np.array([2020, 2021, 2022, 2023, 2024]),
            year_inflation=np.array([2010]),
            inflation_rate=0.1,
        )
    assert isinstance(excinfo.value, TaxInflationException)
    assert excinfo.value.issues[0].code == "TAX_INFLATION_YEAR_OUTSIDE_PROJECT"


def test_engine_cashflow_reversed_period_is_user_input() -> None:
    from datetime import date

    with pytest.raises(CashFlowInputError) as excinfo:
        _ = CashFlow(
            start_date=date(2030, 1, 1),
            end_date=date(2020, 1, 1),
            cash=np.array([1.0]),
            cashed_year=np.array([2025]),
        )
    assert isinstance(excinfo.value, CashFlowException)
    assert excinfo.value.issues[0].code == "CASHFLOW_PERIOD_REVERSED"


def test_engine_cashflow_array_length_mismatch() -> None:
    from datetime import date

    with pytest.raises(CashFlowInputError) as excinfo:
        _ = CashFlow(
            start_date=date(2020, 1, 1),
            end_date=date(2030, 1, 1),
            cash=np.array([1.0, 2.0]),
            cashed_year=np.array([2025]),
        )
    assert excinfo.value.issues[0].code == "CASHFLOW_ARRAY_LENGTH_MISMATCH"


def test_engine_optimization_useful_life_is_user_input() -> None:
    with pytest.raises(OptimizationInputError) as excinfo:
        _ = adjust_useful_life_years(
            adjustment_value=0.5, useful_life_array=np.array([1, 1])
        )
    assert isinstance(excinfo.value, OptimizationException)
    assert excinfo.value.issues[0].code == "OPTIMIZATION_USEFUL_LIFE_TOO_SHORT"


def test_engine_optimization_adjustment_range() -> None:
    with pytest.raises(OptimizationInputError) as excinfo:
        _ = adjust_useful_life_years(
            adjustment_value=1.5, useful_life_array=np.array([3, 4])
        )
    assert excinfo.value.issues[0].code == "OPTIMIZATION_ADJUSTMENT_OUT_OF_RANGE"


# ── Preflight collectors and engine/preflight parity ────────────────────


def test_preflight_summary_argument_issues() -> None:
    payload = {
        "setup": _setup(),
        "summary_arguments": {
            "npv_mode": "zzz",
            "discounting_mode": "End Year Discounting",
            "discount_rate_start_year": 2010,
        },
    }
    codes = _codes(_report(payload))
    assert "SUMMARY_NPV_MODE_UNRECOGNIZED" in codes
    assert "SUMMARY_DISCOUNTING_MODE_UNRECOGNIZED" in codes
    assert "SUMMARY_REFERENCE_YEAR_OUTSIDE_PROJECT" in codes


def test_preflight_summary_reference_year_precedence() -> None:
    # A non-transition summary uses ``discount_rate_start_year``; the unused
    # ``reference_year`` field must not create a false issue.
    payload = {
        "setup": _setup(),
        "summary_arguments": {"reference_year": 2022, "discount_rate_start_year": 2010},
    }
    codes = _codes(_report(payload))
    assert "SUMMARY_REFERENCE_YEAR_OUTSIDE_PROJECT" in codes

    no_reference = {
        "setup": _setup(),
        "summary_arguments": {"reference_year": 2010},
    }
    assert "SUMMARY_REFERENCE_YEAR_OUTSIDE_PROJECT" not in _codes(_report(no_reference))


def test_preflight_sensitivity_argument_issues() -> None:
    payload = {
        "setup": _setup(),
        "sensitivity_arguments": {
            "min_deviation": -0.1,
            "max_deviation": 0.2,
            "increment": 0.0,
            "step": 0,
        },
    }
    codes = _codes(_report(payload))
    assert "SENSITIVITY_DEVIATION_NEGATIVE" in codes
    assert "SENSITIVITY_INCREMENT_NOT_POSITIVE" in codes
    assert "SENSITIVITY_STEP_NOT_POSITIVE" in codes


def test_preflight_uncertainty_argument_issues() -> None:
    payload = {
        "setup": _setup(),
        "uncertainty_arguments": {
            "run_number": 0,
            "oil_price_distribution": "weibull",
            "gas_price_distribution": "Uniform",
            "gas_price_stddev": 0.0,
            "mean_gas_price": 2.0,
            "min_gas_price": 2.0,
            "max_gas_price": 2.0,
            "opex_distribution": "LogNormal",
            "min_opex": -1.0,
            "mean_opex": 2.0,
            "max_opex": 3.0,
            "opex_stddev": 1.0,
        },
    }
    codes = _codes(_report(payload))
    assert "UNCERTAINTY_RUN_NUMBER_INVALID" in codes
    assert "UNCERTAINTY_DISTRIBUTION_UNRECOGNIZED" in codes
    assert "UNCERTAINTY_LOGNORMAL_NONPOSITIVE" in codes


def test_preflight_optimization_argument_issues() -> None:
    payload = {
        "setup": _setup(),
        "optimization_arguments": {
            "target_parameter": "Payback",
            "dict_optimization": {
                "parameter": ["Nope"],
                "min": [0.0],
                "max": [1.0],
            },
        },
    }
    codes = _codes(_report(payload))
    assert "OPTIMIZATION_TARGET_UNRECOGNIZED" in codes
    assert "OPTIMIZATION_PARAMETER_UNRECOGNIZED" in codes


def test_preflight_engine_parity_for_uncertainty() -> None:
    # The preflight and the engine report the same rule for the same parameters.
    payload = {
        "setup": _setup(),
        "uncertainty_arguments": {
            "oil_price_distribution": "Normal",
            "min_oil_price": 1.0,
            "mean_oil_price": 2.0,
            "max_oil_price": 3.0,
            "oil_price_stddev": 0.0,
        },
    }
    report = _report(payload)
    assert "UNCERTAINTY_ZERO_STD" in _codes(report)
    with pytest.raises(MonteCarloInputError) as excinfo:
        _ = get_multipliers_montecarlo(
            run_number=5,
            distribution="normal",
            min_value=1.0,
            mean_value=2.0,
            max_value=3.0,
            std_dev=0.0,
        )
    assert excinfo.value.issues[0].code == "UNCERTAINTY_ZERO_STD"


def test_independent_multiple_errors_preserve_unaffected_checks() -> None:
    payload = {
        "setup": _setup(),
        "summary_arguments": {"npv_mode": "zzz"},
        "sensitivity_arguments": {"min_deviation": -0.1, "max_deviation": 0.1},
        "uncertainty_arguments": {
            "run_number": 0,
            "oil_price_distribution": "weibull",
        },
        "optimization_arguments": {
            "target_parameter": "Payback",
            "dict_optimization": {"parameter": ["Nope"], "min": [0.0], "max": [1.0]},
        },
    }
    codes = _codes(_report(payload))
    assert {
        "SUMMARY_NPV_MODE_UNRECOGNIZED",
        "SENSITIVITY_DEVIATION_NEGATIVE",
        "UNCERTAINTY_RUN_NUMBER_INVALID",
        "UNCERTAINTY_DISTRIBUTION_UNRECOGNIZED",
        "OPTIMIZATION_TARGET_UNRECOGNIZED",
        "OPTIMIZATION_PARAMETER_UNRECOGNIZED",
    } <= codes


def test_collectors_are_pure_and_prefix_aware() -> None:
    summary_payload = {"summary_arguments": {"npv_mode": "zzz"}}
    issues = collect_summary_argument_issues(summary_payload, _BOUNDS)
    assert issues and issues[0].locations[0].path == ("summary_arguments", "npv_mode")

    prefixed = collect_sensitivity_argument_issues(
        {"sensitivity_arguments": {"increment": 0.0}},
        path_prefix=("contract_1",),
    )
    assert prefixed
    assert prefixed[0].locations[0].path == (
        "contract_1",
        "sensitivity_arguments",
        "increment",
    )

    assert collect_uncertainty_argument_issues({}) == ()
    assert collect_optimization_argument_issues({}) == ()


def test_summary_collector_accepts_valid_arguments() -> None:
    payload = {
        "summary_arguments": {
            "npv_mode": "SKK Full Cycle Nominal Terms",
            "discounting_mode": "End Year",
            "reference_year": 2022,
        }
    }
    assert collect_summary_argument_issues(payload, _BOUNDS) == ()


# ── Converter enum / numeric-text conversions (V5–V10) ──────────────────


def test_converter_enum_errors_are_user_input() -> None:
    from pyscnomics.api.converter import (
        ConverterInputError,
        ConverterTypeError,
        convert_str_to_int,
        convert_to_float,
        convert_to_uncertainty_distribution,
        converter_initial_amortization_year,
        converter_reservoir_type_permen_2024,
        converter_sunk_cost_method,
    )

    for call in (
        lambda: converter_sunk_cost_method("zzz"),
        lambda: converter_reservoir_type_permen_2024("zzz"),
        lambda: converter_initial_amortization_year("zzz"),
        lambda: convert_to_uncertainty_distribution("zzz"),
    ):
        with pytest.raises(ConverterInputError) as excinfo:
            _ = call()
        assert isinstance(excinfo.value, ValueError)
        assert isinstance(excinfo.value, InputValidationError)

    for call in (
        lambda: convert_str_to_int(object()),
        lambda: convert_to_float(object()),
    ):
        with pytest.raises(ConverterTypeError) as excinfo:
            _ = call()
        assert isinstance(excinfo.value, TypeError)
        assert isinstance(excinfo.value, InputValidationError)


def test_preflight_flags_nonfinite_curve_field() -> None:
    payload = {
        "setup": _setup(),
        "capital": {
            "well": {
                "start_year": 2019,
                "end_year": 2030,
                "expense_year": [2020],
                "cost": [100.0],
                "cost_allocation": ["Oil"],
                "depreciation_factor": [float("inf")],
            }
        },
    }
    assert "NUMERIC_FIELD_NOT_FINITE" in _codes(_report(payload))
