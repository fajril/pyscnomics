"""Point-forward valuation must not alter reported full-lifecycle cash flow."""

from pyscnomics.dataset.sample import load_data
from pyscnomics.econ.indicator import irr, npv_point_forward  # pyright: ignore[reportUnknownVariableType]
from pyscnomics.econ.selection import DiscountingMode, NPVSelection


def test_costrecovery_point_forward_net_cashflow_remains_full_lifecycle():
    """Cost Recovery reports project cash flow, not its point-forward subset."""

    contract = load_data("case1", "cost_recovery")
    contract.run()  # pyright: ignore[reportUnknownMemberType]
    reference_year = 2026

    summary = contract.get_summary(  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        npv_mode=NPVSelection.NPV_POINT_FORWARD,
        discount_rate_start_year=reference_year,
    )

    point_forward = contract.project_years >= reference_year  # pyright: ignore[reportUnknownMemberType]
    assert summary["ctr_net_cashflow"] == contract._consolidated_cashflow.sum(dtype=float)  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]
    assert summary["ctr_irr"] == irr(contract._consolidated_cashflow[point_forward])  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]
    assert summary["ctr_npv"] == npv_point_forward(
        cashflow=contract._consolidated_cashflow,  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]
        cashflow_years=contract.project_years,  # pyright: ignore[reportUnknownMemberType]
        discount_rate=0.1,
        reference_year=reference_year,
        discounting_mode=DiscountingMode.END_YEAR,
    )
