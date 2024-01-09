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

from pyscnomics.optimize.sensitivity import SensitivityData
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts.transition import Transition
from pyscnomics.tools.summary import get_summary

from pyscnomics.optimize.sensitivity import (
    get_multipliers,
    get_price_and_rate_adjustment,
    get_rate_adjustment,
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

# Multipliers
multipliers, total_run = get_multipliers(
    min_deviation=0.5,
    max_deviation=0.5,
)

print('\t')
print(f'Filetype: {type(multipliers)}')
print(f'Shape: {multipliers.shape}')
print('multipliers = \n', multipliers)

print('\t')
print(f'Filetype: {type(total_run)}')
print('total_run = ', total_run)

print('\t')
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

def run_sensitivity(
    contract: CostRecovery | GrossSplit | Transition,
    contract_arguments: dict,
    summary_argument: dict,
):
    # Running contract
    contract.run(**contract_arguments)
    contract_summary = get_summary(**summary_argument)

    # print('\t')
    # print(f'Filetype: {type(contract_summary)}')
    # print('contract_summary = \n', contract_summary)

    sensitivity_attrs = [
        "oil_price",
        "gas_price",
        "opex",
        "capex",
        "cum_prod",
    ]

    sensitivity_target = [
        "npv",
        "irr",
        "pi",
        "pot",
        "gov_take",
        "ctr_net_share",
    ]

    results = {
        "npv": contract_summary["ctr_npv"],
        "irr": contract_summary["ctr_irr"],
        "pi": contract_summary["ctr_pi"],
        "pot": contract_summary["ctr_pot"],
        "gov_take": contract_summary["gov_take"],
        "ctr_net_share": contract_summary["ctr_net_share"],
    }

    print('\t')
    print(f'Filetype: {type(results)}')
    print('results = \n', results)


params = [
    "psc",
    "psc_arguments",
    "summary_arguments",
    "data",
]

sens = {
    key: [0 for _ in range(multipliers.shape[1])]
    for key in params
}

for i in range(multipliers.shape[1]):
    (
        sens["psc"][i],
        sens["psc_arguments"][i],
        sens["summary_arguments"][i],
        sens["data"][i],
    ) = SensitivityData(multipliers=multipliers[0, i, :]).activate()

print('\t')
print(f'Filetype: {type(sens)}')
print('sens = \n', sens)


# psc = [0 for _ in range(multipliers.shape[1])]
# psc_arguments = [0 for _ in range(multipliers.shape[1])]
# summary_arguments = [0 for _ in range(multipliers.shape[1])]
# data = [0 for _ in range(multipliers.shape[1])]
#
# for i in range(multipliers.shape[1]):
#     (
#         psc[i],
#         psc_arguments[i],
#         summary_arguments[i],
#         data[i]
#     ) = SensitivityData(multipliers=multipliers[0, i, :]).activate()
#
#     run_sensitivity(
#         contract=psc[i],
#         contract_arguments=psc_arguments[i],
#         summary_argument=summary_arguments[i],
#     )

# (
#     psc,
#     psc_arguments,
#     summary_arguments,
#     data
# ) = SensitivityData(
#     multipliers=multipliers[0, 0, :],
# ).activate()


# run_sensitivity(
#     contract=psc,
#     contract_arguments=psc_arguments,
#     summary_argument=summary_arguments,
# )

# print('\t')
# print(f'Filetype: {type(t1)}')
# print(f'Length: {len(t1)}')
# print('t1 = \n', t1)

# print('\t')
# print(f'Filetype: {type(t2)}')
# print(f'Length: {len(t2)}')
# print('t2 = ', t2)
