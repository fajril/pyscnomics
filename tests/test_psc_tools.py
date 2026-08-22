"""Targeted tests for psc_tools transfer-cost and pre-UU DMO repairs."""

import numpy as np
import pytest
from datetime import date

from pyscnomics.contracts.psc_tools import get_dmo, get_unrec_cost_after_tf
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.selection import FluidType


def test_get_unrec_cost_after_tf_transfer_zero() -> None:
    """No transfers, deprecation-only first year yields [1.0, 0.0]."""
    depreciation = np.array([1.0, 0.0])
    non_capital = np.array([0.0, 0.0])
    revenue = np.array([0.0, 1.0])
    ftp_ctr = np.array([0.0, 0.0])
    ftp_gov = np.array([0.0, 0.0])
    ic = np.array([0.0, 0.0])
    transferred_cost_in = np.array([0.0, 0.0])
    transferred_cost_out = np.array([0.0, 0.0])

    result = get_unrec_cost_after_tf(
        depreciation=depreciation,
        non_capital=non_capital,
        revenue=revenue,
        ftp_ctr=ftp_ctr,
        ftp_gov=ftp_gov,
        ic=ic,
        transferred_cost_in=transferred_cost_in,
        transferred_cost_out=transferred_cost_out,
    )

    expected = np.array([1.0, 0.0])
    np.testing.assert_array_equal(result, expected)


def _make_lifting() -> Lifting:
    return Lifting(
        start_year=2020,
        end_year=2021,
        lifting_rate=np.array([1.0, 1.0]),
        price=np.array([50.0, 50.0]),
        prod_year=np.array([2020, 2021]),
        fluid_type=FluidType.OIL,
        ghv=np.array([1.0, 1.0]),
    )


def test_get_dmo_pre_uu_requires_inputs() -> None:
    """Pre-UU DMO must raise a descriptive ValueError when ets/ctr_ets/ctr_ftp are omitted."""
    lifting = _make_lifting()
    with pytest.raises(ValueError, match="Pre-UU"):
        _ = get_dmo(
            onstream_date=date(2020, 1, 1),
            start_date=date(2020, 1, 1),
            project_years=np.array([2020, 2021], dtype=np.int64),
            dmo_holiday_duration=0,
            dmo_volume_portion=0.25,
            dmo_fee_portion=0.25,
            lifting=lifting,
            price=np.array([50.0, 50.0]),
            ctr_pretax_share=0.5,
            unrecovered_cost=np.array([0.0, 0.0]),
            is_dmo_end_weighted=False,
            post_uu_22_year2001=False,
            # ets, ctr_ets, ctr_ftp intentionally omitted
        )
