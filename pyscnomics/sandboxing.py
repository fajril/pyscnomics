"""
A module to execute preliminary evaluations
"""

import time as tm
import numpy as np
from datetime import date
from dataclasses import asdict

from pyscnomics.econ.selection import DeprMethod, FluidType, TaxType
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.costs import Tangible, Intangible, OPEX, ASR
from pyscnomics.econ.depreciation import psc_declining_balance_depreciation_rate as psc_db
from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.econ.results import CashFlow
from pyscnomics.tools.helper import summarizer


'----------------------------------------------- LIFTING DATA ----------------------------------------------'

oil_mangga_lifting = Lifting(
    start_year=2023,
    end_year=2030,
    lifting_rate=np.array([100, 100, 100]),
    price=np.array([10, 10, 10]),
    prod_year=np.array([2025, 2026, 2027]),
    fluid_type=FluidType.OIL,
)

oil_apel_lifting = Lifting(
    start_year=2023,
    end_year=2030,
    lifting_rate=np.array([100, 100, 100]),
    price=np.array([10, 10, 10]),
    prod_year=np.array([2028, 2029, 2030]),
    fluid_type=FluidType.OIL,
)

gas_mangga_lifting = Lifting(
    start_year=2023,
    end_year=2030,
    lifting_rate=np.array([10, 10, 10]),
    price=np.array([1, 1, 1]),
    prod_year=np.array([2025, 2026, 2027]),
    ghv=np.array([1, 1, 1]),
    fluid_type=FluidType.GAS,
)

gas_apel_lifting = Lifting(
    start_year=2023,
    end_year=2030,
    lifting_rate=np.array([10, 10, 10]),
    price=np.array([1, 1, 1]),
    prod_year=np.array([2028, 2029, 2030]),
    ghv=np.array([1, 1, 1]),
    fluid_type=FluidType.GAS,
)

sulfur_mangga_lifting = Lifting(
    start_year=2023,
    end_year=2030,
    lifting_rate=np.array([5, 5, 5]),
    price=np.array([1, 1, 1]),
    prod_year=np.array([2025, 2026, 2027]),
    ghv=np.array([1, 1, 1]),
    fluid_type=FluidType.SULFUR,
)

sulfur_apel_lifting = Lifting(
    start_year=2023,
    end_year=2030,
    lifting_rate=np.array([5, 5, 5]),
    price=np.array([1, 1, 1]),
    prod_year=np.array([2028, 2029, 2030]),
    ghv=np.array([1, 1, 1]),
    fluid_type=FluidType.SULFUR,
)

'------------------------------------------- TANGIBLE COST DATA --------------------------------------------'

# tangible_mangga = Tangible(
#     start_year=2023,
#     end_year=2030,
#     cost=np.array([100, 100]),
#     expense_year=np.array([2024, 2025]),
#     depreciation_factor=np.array([0.5, 0.5]),
#     cost_allocation=[FluidType.OIL, FluidType.OIL],
# )

# tangible_apel = Tangible(
#     start_year=2023,
#     end_year=2030,
#     cost=np.array([50, 50]),
#     expense_year=np.array([2023, 2028]),
#     depreciation_factor=np.array([0.5, 0.5]),
#     cost_allocation=[FluidType.GAS, FluidType.GAS],
# )

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
    cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL],
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
    cost_allocation=[FluidType.OIL, FluidType.OIL],
)

asr_apel = ASR(
    start_year=2023,
    end_year=2030,
    cost=np.array([50, 50]),
    expense_year=np.array([2026, 2027]),
    cost_allocation=[FluidType.GAS, FluidType.GAS],
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

'------------------------------------------------- EXECUTE -------------------------------------------------'

# lifting = (
#     oil_mangga_lifting,
#     oil_apel_lifting,
#     gas_mangga_lifting,
#     gas_apel_lifting,
#     sulfur_mangga_lifting,
#     sulfur_apel_lifting,
# )
#
# tangible_cost = (
#     oil_mangga1_tangible,
#     oil_apel1_tangible,
#     oil_apel2_tangible,
#     gas_mangga1_tangible,
#     gas_mangga2_tangible,
#     oil_mangga2_tangible,
#     oil_nanas1_tangible,
#     sulfur_mangga1_tangible,
#     sulfur_mangga2_tangible
# )

tangible_mangga = Tangible(
    start_year=2023,
    end_year=2030,
    cost=np.array([100, 100, 100, 100]),
    expense_year=np.array([2026, 2027, 2028, 2029]),
    depreciation_factor=np.array([0.5, 0.5, 0.5, 0.5]),
    vat_portion=np.array([1, 1, 1, 1]),
    vat_discount=0.0,
    lbt_portion=np.array([1, 1, 1, 1]),
    lbt_discount=0.0,
    useful_life=np.array([5, 3, 3, 2]),
    cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL],
)

# print('\t')
# print(f'Filetype: {type(tangible_mangga)}')
# print(f'Length: {len(tangible_mangga)}')
# print('tangible_mangga = \n', tangible_mangga)

t1 = tangible_mangga.total_depreciation_rate(
    tax_type=TaxType.VAT,
    vat_rate=0.0,
    # vat_rate=np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]),
    # lbt_rate=0.0,
    # inflation_rate=0.05,
    # inflation_rate=np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]),
    # year_ref=2029,
)

# t1 = tangible_mangga.total_depreciation_rate()
# t2 = tangible_mangga.total_depreciation_book_value()

print('\t')
print(f'Filetype: {type(t1)}')
print(f'Length: {len(t1)}')
print('t1 = \n', t1)

# print('\t')
# print(f'Filetype: {type(t2)}')
# print(f'Length: {len(t2)}')
# print('t2 = ', t2)

# get_gas_tangible = case._get_gas_tangible()
#
# print('\t')
# print(f'Filetype: {type(get_gas_tangible)}')
# print(f'Length: {len(get_gas_tangible)}')
# print('get_gas_tangible = \n', get_gas_tangible)
#
# get_sulfur_tangible = case._get_sulfur_tangible()
#
# print('\t')
# print(f'Filetype: {type(get_sulfur_tangible)}')
# print(f'Length: {len(get_sulfur_tangible)}')
# print('get_sulfur_tangible = \n', get_sulfur_tangible)

# add = np.zeros(8)
# for i in range(len(get_oil_tangible)):
#     add = add + get_oil_tangible[i].expenditures(
#         lbt_discount=0.5
#     )
#
# print('\t')
# print(f'Length: {len(add)}')
# print('add = ', add)


