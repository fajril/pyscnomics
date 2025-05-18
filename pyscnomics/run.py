
import numpy as np
from pyscnomics.econ.costs import SunkCost
from pyscnomics.econ.selection import FluidType, SunkCostInvestmentType, DeprMethod
from datetime import date
from pyscnomics.contracts.project import BaseProject
from pyscnomics.econ.selection import DeprMethod
from pyscnomics.example import ExampleCase


case = ExampleCase()
sc_mangga = case.sunk_cost_mangga
sc_apel = case.sunk_cost_apel

# Calculate sunk cost and preonstream cost array
sc_oil_tangible_array = sc_mangga.get_sunk_cost_investment_array(
    tax_rate=0.0,
    fluid_type=FluidType.OIL,
    investment_config=SunkCostInvestmentType.TANGIBLE,
)

poc_gas_intangible_array = sc_mangga.get_preonstream_cost_investment_array(
    tax_rate=0.0,
    fluid_type=FluidType.GAS,
    investment_config=SunkCostInvestmentType.INTANGIBLE,
)

# Calculate sunk cost and preonstream cost bulk value
sc_oil_tangible_bulk = sc_mangga.get_investment_bulk(
    cost_investment_array=sc_oil_tangible_array
)

poc_gas_intangible_bulk = sc_mangga.get_investment_bulk(
    cost_investment_array=poc_gas_intangible_array
)

# Calculate amortization charge
sc_oil_tangible_amortization_charge = sc_mangga.get_amortization_charge(
    cost_bulk=sc_oil_tangible_bulk,
    prod=np.array([50, 1_000]),
    prod_year=np.array([2027, 2028]),
    salvage_value=0.0,
    amortization_len=8,
)

# Calculate amortization book value
sc_oil_tangible_amortization_bv = sc_mangga.get_amortization_book_value(
    cost_investment_array=sc_oil_tangible_array,
    cost_bulk=sc_oil_tangible_bulk,
    prod=np.array([50, 1_000]),
    prod_year=np.array([2027, 2028]),
    salvage_value=0.0,
    amortization_len=8,
)

# Calculate depreciation rate
sc_oil_tangible_depreciation_charge = sc_mangga.get_sunk_cost_tangible_depreciation_rate(
    fluid_type=FluidType.OIL,
    tax_rate=0.1,
)

poc_gas_tangible_depreciation_charge = (
    sc_apel.get_preonstream_cost_tangible_depreciation_rate(
        fluid_type=FluidType.GAS,
        depr_method=DeprMethod.PSC_DB,
        decline_factor=2,
        tax_rate=0.0,
    )
)

# Calculate depreciation book value
sc_oil_tangible_depreciation_bv = (
    sc_mangga.get_sunk_cost_tangible_depreciation_book_value(
        fluid_type=FluidType.OIL,
        depr_method=DeprMethod.PSC_DB,
        decline_factor=2,
        tax_rate=0.0,
    )
)

print('\t')
print('================================================================')

print('\t')
print(f'Filetype: {type(sc_oil_tangible_depreciation_bv)}')
print(f'Length: {len(sc_oil_tangible_depreciation_bv)}')
print('sc_oil_tangible_depreciation_bv = \n', sc_oil_tangible_depreciation_bv)

# project = BaseProject(
#     start_date=date(year=2023, month=1, day=1),
#     end_date=date(year=2030, month=12, day=31),
#     oil_onstream_date=date(year=2027, month=1, day=1),
#     gas_onstream_date=date(year=2027, month=1, day=1),
#     lifting=tuple([case.lifting_mangga, case.lifting_apel]),
#     capital_cost=tuple([case.capital_mangga, case.capital_apel]),
#     intangible_cost=tuple([case.intangible_mangga, case.intangible_apel]),
#     opex=tuple([case.opex_mangga, case.opex_apel]),
#     asr_cost=tuple([case.asr_mangga, case.asr_apel]),
#     lbt_cost=tuple([case.lbt_mangga, case.lbt_apel]),
#     cost_of_sales=tuple([case.cos_mangga, case.cos_apel]),
#     sunk_cost=tuple([case.sunk_cost_mangga, case.sunk_cost_apel]),
# )

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
