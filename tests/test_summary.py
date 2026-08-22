import pytest

from pyscnomics.contracts.costrecovery import CostRecovery, CostRecoverySummaryException
from pyscnomics.dataset.sample import load_data
from pyscnomics.econ.selection import DiscountingMode, NPVSelection
from pyscnomics.tools import get_summary  # pyright: ignore[reportPrivateLocalImportUsage, reportUnknownVariableType]
from pyscnomics.tools.summary import SummaryException


@pytest.fixture
def cost_recovery() -> CostRecovery:
    contract = load_data("case1", "cost_recovery")
    assert isinstance(contract, CostRecovery)
    contract.run()  # pyright: ignore[reportUnknownMemberType]
    return contract


def test_cost_recovery_summary_requires_successful_run() -> None:
    contract = load_data("case1", "cost_recovery")
    assert isinstance(contract, CostRecovery)

    with pytest.raises(CostRecoverySummaryException, match=r"run\(\)"):
        contract.get_summary()


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
        get_summary(  # pyright: ignore[reportUnusedCallResult]
            contract, reference_year=boundary_year + year_offset
        )


def test_get_summary_matches_cost_recovery_summary(
    cost_recovery: CostRecovery,
) -> None:
    """Catch non-Transition wrapper use of stale Cost Recovery internals."""

    actual = get_summary(cost_recovery)  # pyright: ignore[reportUnknownVariableType]
    expected = cost_recovery.get_summary(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        discount_rate_start_year=cost_recovery.start_date.year
    )

    assert actual.keys() == expected.keys()
    for key in ("ctr_net_cashflow", "ctr_irr", "ctr_npv"):
        assert actual[key] == pytest.approx(expected[key])  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]


def test_get_summary_forwards_non_default_arguments(
    cost_recovery: CostRecovery,
) -> None:
    """Catch dropped public wrapper arguments during delegation."""

    actual = get_summary(  # pyright: ignore[reportUnknownVariableType]
        cost_recovery,
        reference_year=2026,
        inflation_rate=0.05,
        discount_rate=0.08,
        npv_mode=NPVSelection.NPV_REAL_TERMS,
        discounting_mode=DiscountingMode.MID_YEAR,
        profitability_discounted=True,
    )
    expected = cost_recovery.get_summary(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        discount_rate=0.08,
        npv_mode=NPVSelection.NPV_REAL_TERMS,
        discounting_mode=DiscountingMode.MID_YEAR,
        discount_rate_start_year=2026,
        inflation_rate=0.05,
        profitability_discounted=True,
    )

    for key in ("ctr_npv", "ctr_pv_ratio", "ctr_pi"):
        assert actual[key] == pytest.approx(expected[key])  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]


def test_get_summary_maps_point_forward_reference_year(
    cost_recovery: CostRecovery,
) -> None:
    """Catch mapping wrapper reference year to point-forward valuation year."""

    actual = get_summary(  # pyright: ignore[reportUnknownVariableType]
        cost_recovery,
        reference_year=2026,
        npv_mode=NPVSelection.NPV_POINT_FORWARD,
    )
    expected = cost_recovery.get_summary(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        npv_mode=NPVSelection.NPV_POINT_FORWARD,
        discount_rate_start_year=2026,
    )

    for key in ("ctr_net_cashflow", "ctr_irr", "ctr_npv"):
        assert actual[key] == pytest.approx(expected[key])  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]


@pytest.mark.parametrize("contract_type", ["project", "gross_split"])
def test_get_summary_matches_non_cost_recovery_contract_summary(
    contract_type: str,
) -> None:
    """Catch stale wrapper internals for every supported non-Transition contract."""

    contract = load_data("case1", contract_type)
    contract.run()  # pyright: ignore[reportUnknownMemberType]

    actual = get_summary(contract)  # pyright: ignore[reportUnknownVariableType]
    expected = contract.get_summary(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        discount_rate_start_year=contract.start_date.year
    )

    assert actual.keys() == expected.keys()
    for key in ("gross_revenue", "ctr_net_cashflow", "ctr_irr", "ctr_npv"):
        assert actual[key] == pytest.approx(expected[key])  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

    if contract_type == "project":
        assert actual["gov_take"] == 0
        assert actual["cost_recovery/deductible_cost"] == 0
    else:
        assert actual["gov_take"] != 0
        assert actual["cost_recovery/deductible_cost"] != 0
