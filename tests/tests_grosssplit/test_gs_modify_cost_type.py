"""
A collection of unit testings to validate the procedure of modifying
cost_type attribute in each cost Classes.
"""

import numpy as np
from datetime import date

from pyscnomics.econ.selection import OtherRevenue, FluidType, CostType
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.costs import CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales
from pyscnomics.contracts.grossplit import GrossSplit


kwargs = {
    "start_date": date(year=2023, month=1, day=1),
    "end_date": date(year=2034, month=12, day=31),
    "oil_onstream_date": date(year=2030, month=1, day=1),
    "gas_onstream_date": date(year=2029, month=1, day=1),
    "approval_year": 2027,
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


def test_modify_cost_type_capital():

    # Instance Lifting #1
    lifting_mangga = Lifting(
        start_year=2023,
        end_year=2034,
        prod_year=np.array([2030, 2030, 2031, 2031, 2032, 2033]),
        lifting_rate=np.array([200, 200, 100, 100, 50, 50]),
        price=np.array([1, 1, 1, 1, 1, 1]),
        fluid_type=FluidType.OIL,
    )

    # Instance Lifting #2
    lifting_apel = Lifting(
        start_year=2023,
        end_year=2034,
        prod_year=np.array([2029, 2029, 2030, 2030, 2031, 2032]),
        lifting_rate=np.array([20, 20, 10, 10, 5, 5]),
        price=np.array([1, 1, 1, 1, 1, 1]),
        fluid_type=FluidType.GAS,
    )

    # Instance CapitalCost #1
    capital_mangga = CapitalCost(
        start_year=2023,
        end_year=2034,
        expense_year=np.array(
            [
                2023, 2024, 2025, 2026, 2027, 2028,
                2029, 2030, 2031, 2032, 2033, 2034,
            ]
        ),
        cost=np.array(
            [
                200, 200, 200, 200, 150, 100,
                100, 75, 50, 50, 50, 50,
            ]
        ),
        cost_allocation=(
            [
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
            ]
        ),
        cost_type=(
            [
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.SUNK_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
            ]
        ),
        tax_portion=np.array(
            [
                1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1,
            ]
        ),
    )

    # Instance CapitalCost #2
    capital_apel = CapitalCost(
        start_year=2023,
        end_year=2034,
        expense_year=np.array(
            [
                2023, 2024, 2025, 2026, 2027, 2028,
                2029, 2030, 2031, 2032, 2033, 2034,
            ]
        ),
        cost=np.array(
            [
                20, 20, 20, 20, 15, 10,
                10, 7.5, 5, 5, 5, 5,
            ]
        ),
        cost_allocation=(
            [
                FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
            ]
        ),
        cost_type=(
            [
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
                CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
            ]
        ),
        tax_portion=np.array(
            [
                1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1,
            ]
        ),
    )

    lifting_total = tuple([lifting_mangga, lifting_apel])
    capital_total = tuple([capital_mangga, capital_apel])

    gs = GrossSplit(**kwargs, lifting=lifting_total, capital_cost=capital_total)




def test_modify_cost_type_lbt():
    pass

