import math
from numbers import Real

import pytest
import numpy as np

from pyscnomics.contracts.costrecovery import CostRecovery, CostRecoverySummaryException
from pyscnomics.contracts.gross_split import GrossSplit
from pyscnomics.contracts.project import BaseProject
from pyscnomics.dataset.sample import load_data
from pyscnomics.econ.selection import (
    DiscountingMode,
    FTPTaxRegime,
    GrossSplitRegime,
    NPVSelection,
)
from pyscnomics.tools import get_summary
from pyscnomics.tools.summary import Summary, SummaryException


def _assert_approx(actual: object, expected: object) -> None:
    assert isinstance(actual, Real)
    assert isinstance(expected, Real)
    assert math.isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-12)


@pytest.fixture
def cost_recovery() -> CostRecovery:
    contract = load_data("case1", "cost_recovery")
    assert isinstance(contract, CostRecovery)
    contract.run()
    return contract


def test_cost_recovery_summary_requires_successful_run() -> None:
    contract = load_data("case1", "cost_recovery")
    assert isinstance(contract, CostRecovery)

    with pytest.raises(CostRecoverySummaryException, match=r"run\(\)"):
        _ = contract.get_summary()


@pytest.mark.parametrize(
    ("year_offset", "message"),
    [(-1, "is before the project years"), (1, "is after the project years")],
)
def test_get_summary_rejects_out_of_bounds_reference_year(
    year_offset: int, message: str
) -> None:
    """Catch missing wrapper reference-year validation."""

    contract = load_data("case1", "cost_recovery")
    assert isinstance(contract, CostRecovery)
    boundary_year = (
        contract.start_date.year if year_offset < 0 else contract.end_date.year
    )
    with pytest.raises(SummaryException, match=message):
        _ = get_summary(contract, reference_year=boundary_year + year_offset)


def test_get_summary_matches_cost_recovery_summary(
    cost_recovery: CostRecovery,
) -> None:
    """Catch non-Transition wrapper use of stale Cost Recovery internals."""

    actual = get_summary(cost_recovery)
    expected = cost_recovery.get_summary(
        discount_rate_start_year=cost_recovery.start_date.year
    )

    assert actual.keys() == expected.keys()
    for key in ("ctr_net_cashflow", "ctr_irr", "ctr_npv"):
        _assert_approx(actual[key], expected[key])


def test_get_summary_forwards_non_default_arguments(
    cost_recovery: CostRecovery,
) -> None:
    """Catch dropped public wrapper arguments during delegation."""

    actual = get_summary(
        cost_recovery,
        reference_year=2026,
        inflation_rate=0.05,
        discount_rate=0.08,
        npv_mode=NPVSelection.NPV_REAL_TERMS,
        discounting_mode=DiscountingMode.MID_YEAR,
        profitability_discounted=True,
    )
    expected = cost_recovery.get_summary(
        discount_rate=0.08,
        npv_mode=NPVSelection.NPV_REAL_TERMS,
        discounting_mode=DiscountingMode.MID_YEAR,
        discount_rate_start_year=2026,
        inflation_rate=0.05,
        profitability_discounted=True,
    )

    for key in ("ctr_npv", "ctr_pv_ratio", "ctr_pi"):
        _assert_approx(actual[key], expected[key])


def test_get_summary_maps_point_forward_reference_year(
    cost_recovery: CostRecovery,
) -> None:
    """Catch mapping wrapper reference year to point-forward valuation year."""

    actual = get_summary(
        cost_recovery,
        reference_year=2026,
        npv_mode=NPVSelection.NPV_POINT_FORWARD,
    )
    expected = cost_recovery.get_summary(
        npv_mode=NPVSelection.NPV_POINT_FORWARD,
        discount_rate_start_year=2026,
    )

    for key in ("ctr_net_cashflow", "ctr_irr", "ctr_npv"):
        _assert_approx(actual[key], expected[key])


def test_summary_point_forward_discounted_profitability_returns_summary_dict() -> None:
    contract = load_data("case1", "cost_recovery")
    assert isinstance(contract, CostRecovery)
    contract.run()

    summary = Summary(
        contract=contract,
        reference_year=contract.start_date.year,
        npv_mode=NPVSelection.NPV_POINT_FORWARD,
        profitability_discounted=True,
    )
    summary.run()

    assert summary.investment_npv != 0
    assert "ctr_pi" in summary.get_summary_dict()


@pytest.mark.parametrize(
    ("contract_type", "numpy_key", "python_key"),
    [
        ("project", "ctr_net_cashflow", "ctr_npv"),
        ("cost_recovery", "ctr_net_cashflow", "ctr_npv"),
        ("gross_split", "ctr_net_cashflow", "ctr_npv"),
    ],
)
def test_summary_preserves_feature_scalar_runtime_types(
    contract_type: str, numpy_key: str, python_key: str
) -> None:
    contract = load_data(
        "case1" if contract_type != "gross_split" else "case3", contract_type
    )
    assert isinstance(contract, (BaseProject, CostRecovery, GrossSplit))
    contract.run()
    result = contract.get_summary(discount_rate_start_year=contract.start_date.year)
    assert type(result[numpy_key]) is np.float64
    assert type(result[python_key]) is float


@pytest.mark.parametrize(
    ("contract_type", "expected_numpy_keys"),
    [
        (
            "project",
            ("ctr_pi", "ctr_pv_ratio"),
        ),
        (
            "cost_recovery",
            ("ctr_pi", "ctr_pv_ratio", "unrec_cost"),
        ),
        (
            "gross_split",
            (
                "ctr_gross_share",
                "ctr_net_share",
                "ctr_net_share_over_gross_share",
                "ctr_pi",
                "ctr_pv_ratio",
                "gas_capex",
                "gas_carry_forward_depreciation",
                "gas_indirect_taxes",
                "gov_gross_share",
                "gov_take",
                "gov_take_over_gross_rev",
                "gov_tax_income",
                "gross_revenue",
                "gross_revenue_gas",
                "gross_revenue_oil",
                "intangible",
                "lifting_gas",
                "lifting_oil",
                "oil_capex",
                "oil_carry_forward_depreciation",
                "oil_indirect_taxes",
                "sunk_cost",
                "total_carry_forward_depreciation",
            ),
        ),
    ],
)
def test_summary_residual_scalars_are_numpy_float64(
    contract_type: str,
    expected_numpy_keys: tuple[str, ...],
) -> None:
    contract = load_data(
        "case3" if contract_type == "gross_split" else "case1", contract_type
    )
    assert isinstance(contract, (BaseProject, CostRecovery, GrossSplit))
    if contract_type == "cost_recovery":
        assert isinstance(contract, CostRecovery)
        contract.run(
            effective_tax_rate=0.405,
            ftp_tax_regime=FTPTaxRegime.DIRECT_MODE,
            post_uu_22_year2001=True,
        )
    elif contract_type == "gross_split":
        assert isinstance(contract, GrossSplit)
        contract.run(
            effective_tax_rate=0.22,
            vat_rate=0.11,
            inflation_rate=0.0,
            inflation_rate_applied_to=None,
            regime=GrossSplitRegime.PERMEN_ESDM_12_2020,
        )
    else:
        assert isinstance(contract, BaseProject)
        contract.run()
    expected = contract.get_summary(discount_rate_start_year=contract.start_date.year)
    actual = get_summary(contract, reference_year=contract.start_date.year)
    assert actual.keys() == expected.keys()
    for summary in (expected, actual):
        for key in expected_numpy_keys:
            assert type(summary[key]) is np.float64, key


@pytest.mark.parametrize("contract_type", ["project", "gross_split"])
def test_get_summary_matches_non_cost_recovery_contract_summary(
    contract_type: str,
) -> None:
    """Catch stale wrapper internals for every supported non-Transition contract."""

    contract = load_data("case1", contract_type)
    assert isinstance(contract, (BaseProject, CostRecovery, GrossSplit))
    contract.run()

    actual = get_summary(contract)
    expected = contract.get_summary(discount_rate_start_year=contract.start_date.year)

    assert actual.keys() == expected.keys()
    for key in ("gross_revenue", "ctr_net_cashflow", "ctr_irr", "ctr_npv"):
        _assert_approx(actual[key], expected[key])

    if contract_type == "project":
        assert actual["gov_take"] == 0
        assert actual["cost_recovery/deductible_cost"] == 0
    else:
        assert actual["gov_take"] != 0
        assert actual["cost_recovery/deductible_cost"] != 0
