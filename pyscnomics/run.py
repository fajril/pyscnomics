
import numpy as np
from pyscnomics.econ.costs import SunkCost
from pyscnomics.econ.selection import FluidType, SunkCostInvestmentType, DeprMethod
from datetime import date
from pyscnomics.contracts.project import BaseProject
from pyscnomics.econ.selection import DeprMethod
from pyscnomics.example import ExampleCase


case = ExampleCase()

pr = BaseProject(
    start_date=date(year=2023, month=1, day=1),
    end_date=date(year=2030, month=12, day=31),
    oil_onstream_date=date(year=2027, month=1, day=1),
    gas_onstream_date=date(year=2027, month=1, day=1),
    lifting=tuple([case.lifting_mangga, case.lifting_apel]),
    capital_cost=tuple([case.capital_mangga, case.capital_apel]),
    intangible_cost=tuple([case.intangible_mangga, case.intangible_apel]),
    opex=tuple([case.opex_mangga, case.opex_apel]),
    asr_cost=tuple([case.asr_mangga, case.asr_apel]),
    lbt_cost=tuple([case.lbt_mangga, case.lbt_apel]),
    cost_of_sales=tuple([case.cos_mangga, case.cos_apel]),
    sunk_cost=tuple([case.sunk_cost_mangga, case.sunk_cost_apel]),
)

print('\t')
print('================================================================')

pr.fit_sunk_preonstream_cost(
    tax_rate=0.0,
    prod_year=np.array([2027, 2028]),
    prod=np.array([50, 1_000]),
    salvage_value=0.0,
    amortization_len=8,
)

# t1 = posc.sunk_cost_amortization_charge(
#     tax_rate=0.1,
#     fluid_type=FluidType.GAS,
#     investment_config=SunkCostInvestmentType.INTANGIBLE,
#     prod_year=np.array([2027, 2028]),
#     prod=np.array([50, 1_000]),
#     salvage_value=0.0,
#     amortization_len=8,
# )
#
# t2 = posc.preonstream_cost_amortization_charge(
#     tax_rate=0.1,
#     fluid_type=FluidType.GAS,
#     investment_config=SunkCostInvestmentType.INTANGIBLE,
#     prod_year=np.array([2027, 2028]),
#     prod=np.array([50, 1_000]),
#     salvage_value=0.0,
#     amortization_len=8,
# )

# print('\t')
# print(f'Filetype: {type()}')
# print(f'Length: {len()}')
# print()

# print('\t')
# print(f'Filetype: {type(t1)}')
# print(f'Length: {len(t1)}')
# print('t1 = \n', t1)

# print('\t')
# print(f'Filetype: {type(t2)}')
# print(f'Length: {len(t2)}')
# print('t2 = \n', t2)
