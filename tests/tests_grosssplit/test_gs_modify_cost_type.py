"""
A collection of unit testings to validate the procedure of modifying
cost_type attribute in each cost Classes.
"""

import numpy as np
from datetime import date

from pyscnomics.econ.selection import OtherRevenue, FluidType, CostType
from pyscnomics.econ.costs import CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales
from pyscnomics.contracts.grossplit import GrossSplit


kwargs = {
    "start_date": date(year=2023, month=1, day=1),
    "end_date": date(year=2034, month=12, day=31),
    "oil_onstream_date": date(year=2030, month=1, day=1),
    "gas_onstream_date": date(year=2029, month=1, day=1),
    "approval_year": None,
    "base_split_ctr_oil": 0.43,
    "base_split_ctr_gas": 0.48,
    "split_ministry_disc": 0.08,
    "oil_dmo_volume_portion": 0.25,
    "oil_dmo_fee_portion": 1.0,
    "gas_dmo_volume_portion": 1.0,
    "gas_dmo_fee_portion": 1.0,
    "oil_dmo_holiday_duration": 60,
    "gas_dmo_holiday_duration": 60,
    "oil_carry_forward_depreciation": 10,
    "gas_carry_forward_depreciation": 10,
}


def test_modify_cost_type_capital_cost():
    pass


def test_modify_cost_type_intangible():
    pass


def test_modify_cost_type_lbt():
    pass

