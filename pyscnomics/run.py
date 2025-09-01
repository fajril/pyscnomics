
import numpy as np
from pyscnomics.econ.selection import (
    FluidType,
    DeprMethod,
    InflationAppliedTo,
    OtherRevenue,
    CashflowType,
    TaxSplitTypeCR,
    CostType
)
from datetime import date
from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.econ.selection import DeprMethod
from pyscnomics.example import ExampleCase
from pyscnomics.econ.costs import CapitalCost

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

case = ExampleCase()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# params_base = {
#     "sulfur_revenue": OtherRevenue.REDUCTION_TO_OIL_OPEX,
#     "electricity_revenue": OtherRevenue.ADDITION_TO_OIL_REVENUE,
#     "co2_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
#     "tax_rate": 0.0,
#     "year_inflation": None,
#     "inflation_rate": 0.0,
#     "inflation_rate_applied_to": None,
# }
#
# pr = BaseProject(
#     start_date=date(year=2023, month=1, day=1),
#     end_date=date(year=2030, month=12, day=31),
#     oil_onstream_date=date(year=2026, month=1, day=1),
#     gas_onstream_date=date(year=2028, month=1, day=1),
#     approval_year=2024,
#     lifting=tuple([case.lifting_mangga, case.lifting_apel, case.lifting_nanas]),
#     capital_cost=tuple([case.capital_mangga, case.capital_apel]),
#     intangible_cost=tuple([case.intangible_mangga, case.intangible_apel]),
#     opex=tuple([case.opex_mangga, case.opex_apel]),
#     asr_cost=tuple([case.asr_mangga, case.asr_apel]),
#     lbt_cost=tuple([case.lbt_mangga, case.lbt_apel]),
#     cost_of_sales=tuple([case.cos_mangga, case.cos_apel]),
# )
#
# pr.run(**params_base)

# print('\t')
# print(f'Filetype: {type()}')
# print(f'Length: {len()}')
# print()

# print('\t')
# print(f'Filetype: {type(t1)}')
# print(f'Length: {len(t1)}')
# print('t1 = \n', t1)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

kwargs_gs = {
    "base_split_ctr_oil": 0.43,
    "base_split_ctr_gas": 0.48,
    "split_ministry_disc": 0.08,
    "oil_dmo_volume_portion": 0.25,
    "oil_dmo_fee_portion": 1.0,
    "gas_dmo_volume_portion": 1.0,
    "gas_dmo_fee_portion": 1.0,
    "oil_dmo_holiday_duration": 60,
    "gas_dmo_holiday_duration": 60,
    "oil_carry_forward_depreciation": np.array([100, 200, 300, 400, 500, 600]),
    "gas_carry_forward_depreciation": 10,
}

params_gs = {
    "sulfur_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
    "electricity_revenue": OtherRevenue.ADDITION_TO_OIL_REVENUE,
    "co2_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
    "vat_rate": 0.1,
    "year_inflation": None,
    "inflation_rate": 0.0,
    "inflation_rate_applied_to": None,
    "is_pod_1": True,
}

gs = GrossSplit(
    start_date=date(year=2023, month=1, day=1),
    end_date=date(year=2034, month=12, day=31),
    oil_onstream_date=date(year=2030, month=1, day=1),
    gas_onstream_date=date(year=2029, month=1, day=1),
    approval_year=2027,
    lifting=tuple([case.lifting_mangga, case.lifting_apel, case.lifting_nanas]),
    capital_cost=tuple([case.capital_mangga, case.capital_apel]),
    intangible_cost=tuple([case.intangible_mangga, case.intangible_apel]),
    opex=tuple([case.opex_mangga, case.opex_apel]),
    asr_cost=tuple([case.asr_mangga, case.asr_apel]),
    lbt_cost=tuple([case.lbt_mangga, case.lbt_apel]),
    cost_of_sales=tuple([case.cos_mangga, case.cos_apel]),
    **kwargs_gs,
)

# gs.run(**params_gs)

# print('\t')
# print(f'Filetype: {type()}')
# print(f'Length: {len()}')
# print()

# print('\t')
# print(f'Filetype: {type(t1)}')
# print(f'Length: {len(t1)}')
# print('t1 = \n', t1)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# kwargs_cr = {
#     "oil_ftp_is_available": True,
#     "oil_ftp_is_shared": True,
#     "oil_ftp_portion": 0.2,
#     "gas_ftp_is_available": True,
#     "gas_ftp_is_shared": True,
#     "gas_ftp_portion": 0.2,
#     "tax_split_type": TaxSplitTypeCR.CONVENTIONAL,
#     "condition_dict": dict,
#     "indicator_rc_icp_sliding": None,
#     "oil_ctr_pretax_share": 0.25,
#     "gas_ctr_pretax_share": 0.5,
#     "oil_ic_rate": 0.0,
#     "gas_ic_rate": 0.0,
#     "ic_is_available": False,
#     "oil_cr_cap_rate": 1.0,
#     "gas_cr_cap_rate": 1.0,
#     "oil_dmo_volume_portion": 0.25,
#     "oil_dmo_fee_portion": 0.25,
#     "oil_dmo_holiday_duration": 60,
#     "gas_dmo_volume_portion": 1.0,
#     "gas_dmo_fee_portion": 1.0,
#     "gas_dmo_holiday_duration": 60,
#     "oil_carry_forward_depreciation": np.array([100, 100, 100, 100, 100, 100, 100, 100]),
#     "gas_carry_forward_depreciation": 50,
# }
#
# params_cr = {
#     "sulfur_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
#     "electricity_revenue": OtherRevenue.ADDITION_TO_OIL_REVENUE,
#     "co2_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
#     "vat_rate": 0.0,
#     "year_inflation": None,
#     "inflation_rate": 0.0,
#     "inflation_rate_applied_to": None,
# }
#
# cr = CostRecovery(
#     start_date=date(year=2023, month=1, day=1),
#     end_date=date(year=2030, month=12, day=31),
#     oil_onstream_date=date(year=2027, month=1, day=1),
#     gas_onstream_date=date(year=2028, month=1, day=1),
#     lifting=tuple([case.lifting_mangga, case.lifting_apel, case.lifting_nanas]),
#     capital_cost=tuple([case.capital_mangga, case.capital_apel]),
#     intangible_cost=tuple([case.intangible_mangga, case.intangible_apel]),
#     opex=tuple([case.opex_mangga, case.opex_apel]),
#     asr_cost=tuple([case.asr_mangga, case.asr_apel]),
#     lbt_cost=tuple([case.lbt_mangga, case.lbt_apel]),
#     cost_of_sales=tuple([case.cos_mangga, case.cos_apel]),
#     **kwargs_cr,
# )
#
# cr.run(**params_cr)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
