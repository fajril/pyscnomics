"""Point-forward valuation must not alter reported full-lifecycle cash flow."""

import numpy as np
import numpy.typing as npt

from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.dataset.sample import load_data
from pyscnomics.econ.indicator import irr, npv_point_forward
from pyscnomics.econ.selection import DiscountingMode, NPVSelection

FloatArray = npt.NDArray[np.float64]


def _require_cashflow(contract: CostRecovery) -> FloatArray:
    value = vars(contract).get("_consolidated_cashflow")
    if (
        not isinstance(value, np.ndarray)
        or value.ndim != 1
        or value.dtype != np.dtype(np.float64)
    ):
        raise AssertionError("Expected a float64 consolidated cashflow")
    return np.asarray(value, dtype=np.float64)


def test_costrecovery_point_forward_net_cashflow_remains_full_lifecycle():
    """Cost Recovery reports project cash flow, not its point-forward subset."""

    contract = load_data("case1", "cost_recovery")
    contract.run()
    reference_year = 2026

    summary = contract.get_summary(
        npv_mode=NPVSelection.NPV_POINT_FORWARD,
        discount_rate_start_year=reference_year,
    )

    cashflow = _require_cashflow(contract)
    project_years = contract.project_years
    assert cashflow is not None
    point_forward = project_years >= reference_year
    assert summary["ctr_net_cashflow"] == cashflow.sum(dtype=float)
    assert summary["ctr_irr"] == irr(cashflow[point_forward])
    assert summary["ctr_npv"] == npv_point_forward(
        cashflow=cashflow,
        cashflow_years=project_years,
        discount_rate=0.1,
        reference_year=reference_year,
        discounting_mode=DiscountingMode.END_YEAR,
    )
