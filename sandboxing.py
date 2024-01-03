"""
A module to execute preliminary evaluations
"""

import time as tm
import numpy as np
from datetime import date
from dataclasses import asdict

from pyscnomics.io.spreadsheet import Spreadsheet
from pyscnomics.io.aggregator import Aggregate
from pyscnomics.econ.selection import DeprMethod, FluidType, TaxType
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.costs import Tangible, Intangible, OPEX, ASR
# from pyscnomics.econ.depreciation import psc_declining_balance_depreciation_rate as psc_db
from pyscnomics.contracts.project import BaseProject
from pyscnomics.econ.results import CashFlow
# from pyscnomics.tools.helper import summarizer
from pyscnomics.io.parse import InitiateContract

from pyscnomics.optimize.sensitivity import (
    get_multipliers,
    get_oil_price_adjustment,
    get_gas_price_adjustment,
    get_opex_adjustment,
)


'----------------------------------------------- LIFTING DATA ----------------------------------------------'

oil_lifting_mangga = Lifting(
    start_year=2023,
    end_year=2030,
    lifting_rate=np.array([100, 100, 100]),
    price=np.array([10, 10, 10]),
    prod_year=np.array([2025, 2026, 2027]),
    fluid_type=FluidType.OIL,
)

oil_lifting_apel = Lifting(
    start_year=2023,
    end_year=2030,
    lifting_rate=np.array([100, 100, 100]),
    price=np.array([10, 10, 10]),
    prod_year=np.array([2028, 2029, 2030]),
    fluid_type=FluidType.OIL,
)

gas_lifting_mangga = Lifting(
    start_year=2023,
    end_year=2030,
    lifting_rate=np.array([10, 10, 10]),
    price=np.array([1, 1, 1]),
    prod_year=np.array([2025, 2026, 2027]),
    ghv=np.array([1, 1, 1]),
    fluid_type=FluidType.GAS,
)

gas_lifting_apel = Lifting(
    start_year=2023,
    end_year=2030,
    lifting_rate=np.array([10, 10, 10]),
    price=np.array([1, 1, 1]),
    prod_year=np.array([2028, 2029, 2030]),
    ghv=np.array([1, 1, 1]),
    fluid_type=FluidType.GAS,
)

sulfur_lifting_mangga = Lifting(
    start_year=2023,
    end_year=2030,
    lifting_rate=np.array([5, 5, 5]),
    price=np.array([1, 1, 1]),
    prod_year=np.array([2025, 2026, 2027]),
    ghv=np.array([1, 1, 1]),
    fluid_type=FluidType.SULFUR,
)

sulfur_lifting_apel = Lifting(
    start_year=2023,
    end_year=2030,
    lifting_rate=np.array([5, 5, 5]),
    price=np.array([1, 1, 1]),
    prod_year=np.array([2028, 2029, 2030]),
    ghv=np.array([1, 1, 1]),
    fluid_type=FluidType.SULFUR,
)

'------------------------------------------- TANGIBLE COST DATA --------------------------------------------'

tangible_mangga = Tangible(
    start_year=2023,
    end_year=2030,
    cost=np.array([100, 100, 100, 100]),
    expense_year=np.array([2023, 2024, 2025, 2026]),
    vat_portion=np.array([1, 1, 1, 1]),
    vat_discount=0.0,
    lbt_portion=np.array([1, 1, 1, 1]),
    lbt_discount=0.0,
    cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL],
)

tangible_apel = Tangible(
    start_year=2023,
    end_year=2030,
    cost=np.array([50, 50, 50]),
    expense_year=np.array([2028, 2029, 2030]),
    vat_portion=np.array([1, 1, 1]),
    vat_discount=0.5,
    lbt_portion=np.array([0.5, 0.5, 0.5]),
    lbt_discount=0.5,
    cost_allocation=[FluidType.GAS, FluidType.GAS, FluidType.GAS],
)

'------------------------------------------- INTANGIBLE COST DATA ------------------------------------------'

intangible_mangga = Intangible(
    start_year=2023,
    end_year=2030,
    cost=np.array([100, 100, 100, 100]),
    expense_year=np.array([2024, 2025, 2026, 2027]),
    cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL],
)

intangible_apel = Intangible(
    start_year=2023,
    end_year=2030,
    cost=np.array([50, 50, 50]),
    expense_year=np.array([2027, 2028, 2029]),
    cost_allocation=[FluidType.GAS, FluidType.GAS, FluidType.GAS]
)

'------------------------------------------------ OPEX DATA ------------------------------------------------'

opex_mangga = OPEX(
    start_year=2023,
    end_year=2030,
    fixed_cost=np.array([100, 100, 100, 100]),
    expense_year=np.array([2023, 2024, 2025, 2026]),
    cost_allocation=[FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS],
)

opex_apel = OPEX(
    start_year=2023,
    end_year=2030,
    fixed_cost=np.array([50, 50, 50]),
    expense_year=np.array([2023, 2024, 2025]),
    cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL],
)

'---------------------------------------------- ASR COST DATA ----------------------------------------------'

asr_mangga = ASR(
    start_year=2023,
    end_year=2030,
    cost=np.array([100, 100]),
    expense_year=np.array([2026, 2027]),
    cost_allocation=[FluidType.OIL, FluidType.GAS],
)

asr_apel = ASR(
    start_year=2023,
    end_year=2030,
    cost=np.array([50, 50]),
    expense_year=np.array([2026, 2027]),
    cost_allocation=[FluidType.OIL, FluidType.GAS],
)

'------------------------------------------------- CASHFLOW ------------------------------------------------'

oil_mangga_cashflow = CashFlow(
    start_date=date(2023, 1, 1),
    end_date=date(2030, 12, 31),
    cash=np.array([10, 10, 10, 10]),
    cashed_year=np.array([2026, 2027, 2028, 2029]),
    cash_allocation=FluidType.OIL
)

oil_apel_cashflow = CashFlow(
    start_date=date(2024, 1, 1),
    end_date=date(2029, 12, 31),
    cash=np.array([5, 5, 5, 5, 5]),
    cashed_year=np.array([2024, 2025, 2026, 2027, 2028]),
    cash_allocation=FluidType.OIL
)

oil_nanas_cashflow = CashFlow(
    start_date=date(2026, 1, 1),
    end_date=date(2027, 12, 31),
    cash=np.array([5, 5]),
    cashed_year=np.array([2026, 2027]),
    cash_allocation=FluidType.OIL
)

'---------------------------------------------- BASE PROJECT -----------------------------------------------'

lifting_data = (
    oil_lifting_mangga,
    oil_lifting_apel,
    gas_lifting_mangga,
    gas_lifting_apel,
    # sulfur_lifting_mangga,
    # sulfur_lifting_apel,
)

tangible_cost_data = (
    tangible_mangga,
    tangible_apel,
)

intangible_cost_data = (
    intangible_mangga,
    intangible_apel,
)

opex_data = (
    opex_mangga,
    opex_apel,
)

asr_cost_data = (
    asr_mangga,
    asr_apel,
)

'---------------------------------------------- COST RECOVERY ----------------------------------------------'

# lifting_sawo = Lifting(
#     start_year=2023,
#     end_year=2030,
#     prod_year=np.array([2026, 2023, 2029, 2023]),
#     lifting_rate=np.array([37, 80, 59, 20]),
#     price=np.array([10, 10, 10, 10]),
#     fluid_type=FluidType.OIL,
# )
#
# pr = BaseProject(
#     start_date=date(year=2023, month=1, day=1),
#     end_date=date(year=2030, month=12, day=31),
#     lifting=tuple([lifting_sawo]),
# )

'------------------------------------------------- EXECUTE -------------------------------------------------'

# timer_start = tm.time()
#
# data = Spreadsheet()
# data.prepare_data()
#
# timer_end = tm.time() - timer_start
#
# print('\t')
# print('timer end = ', timer_end)

# timer_start = tm.time()
#
# case = InitiateContract().activate()
#
# data = Aggregate()
# data.fit()
#
# timer_end = tm.time() - timer_start

# print('\t')
# print('timer end = ', timer_end)

case = InitiateContract()

# Sensitivity data
oil_lifting_aggregate_total = case.data.oil_lifting_aggregate_total
gas_lifting_aggregate_total = case.data.gas_lifting_aggregate_total
opex_aggregate = case.data.opex_aggregate

# print('\t')
# print(f'Filetype: {type(oil_lifting_aggregate_total)}')
# print(f'Length: {len(oil_lifting_aggregate_total)}')
# print('oil_lifting_aggregate_total = \n', oil_lifting_aggregate_total)

# Multipliers
multipliers = get_multipliers(
    min_deviation=1.,
    max_deviation=1.,
)

print('\t')
print(f'Filetype: {type(multipliers)}')
print(f'Shape: {multipliers.shape}')
print('multipliers = \n', multipliers)

print('\t')
print('========================================================================================')

oil_lifting_aggregate_total = get_oil_price_adjustment(
    contract_type=case.contract_type,
    oil_lifting_aggregate_total=oil_lifting_aggregate_total,
    oil_price_multiplier=multipliers[0, 0, 0],
)

gas_lifting_aggregate_total = get_gas_price_adjustment(
    contract_type=case.contract_type,
    gas_lifting_aggregate_total=gas_lifting_aggregate_total,
    gas_price_multiplier=multipliers[0, 0, 1],
)

opex_aggregate = get_opex_adjustment(
    contract_type=case.contract_type,
    opex_aggregate=opex_aggregate,
    opex_multiplier=multipliers[0, 0, 2],
)

# print('\t')
# print(f'Filetype: {type(opex_aggregate["PSC 2"][0].cost)}')
# print(f'Length: {len(opex_aggregate["PSC 2"][0].cost)}')
# print('opex_aggregate = \n', opex_aggregate["PSC 2"][0].cost)

# print('\t')
# print(f'Oil price after adjustment:')
# for i, val in enumerate(t1):
#     print(t1[i].price)

# print('\t')
# print(f'Filetype: {type(t1)}')
# print(f'Length: {len(t1)}')
# print('t1 = \n', t1)

# print('\t')
# print(f'Filetype: {type(t2)}')
# print(f'Length: {len(t2)}')
# print('t2 = ', t2)
