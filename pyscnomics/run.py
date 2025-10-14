
import numpy as np
from pyscnomics.econ.selection import (
    FluidType,
    DeprMethod,
    InflationAppliedTo,
    OtherRevenue,
    CashflowType,
    TaxSplitTypeCR,
    CostType,
    InitialYearAmortizationIncurred,
    InitialYearDepreciationIncurred,
    VariableSplit082017,
    VariableSplit522017,
    VariableSplit132024,
    GrossSplitRegime,
    TaxRegime,
    FTPTaxRegime,
    SunkCostMethod,
)
from datetime import date
from dataclasses import asdict
from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.econ.selection import DeprMethod
from pyscnomics.example import ExampleCase
from pyscnomics.econ.costs import CapitalCost

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

case = ExampleCase()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# params = {
#     "depr_method": DeprMethod.PSC_DB,
#     "decline_factor": 2,
#     "year_inflation": None,
#     "inflation_rate": 0.01,
#     "tax_rate": 0.0,
#     "inflation_rate_applied_to": InflationAppliedTo.OPEX,
# }
#
# capital_mangga = case.capital_mangga
# t1 = capital_mangga.total_depreciation_rate(**params)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# kwargs_pr = {
#     "start_date": date(year=2023, month=1, day=1),
#     "end_date": date(year=2032, month=12, day=31),
#     "oil_onstream_date": date(year=2030, month=1, day=1),
#     "gas_onstream_date": date(year=2029, month=1, day=1),
#     "approval_year": 2026,
#     "is_pod_1": False,
# }
#
# params_base = {
#     "sulfur_revenue": OtherRevenue.REDUCTION_TO_GAS_OPEX,
#     "electricity_revenue": OtherRevenue.ADDITION_TO_OIL_REVENUE,
#     "co2_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
#     "tax_rate": 0.0,
#     "year_inflation": None,
#     "inflation_rate": 0.0,
#     "inflation_rate_applied_to": None,
# }
#
# pr = BaseProject(
#     **kwargs_pr,
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
# t1 = pr.get_summary()

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

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# kwargs_gs = {
#     "start_date": date(year=2023, month=1, day=1),
#     "end_date": date(year=2032, month=12, day=31),
#     "oil_onstream_date": date(year=2030, month=1, day=1),
#     "gas_onstream_date": date(year=2029, month=1, day=1),
#     "approval_year": 2026,
#     "is_pod_1": True,
#     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     "split_ministry_disc": 0.08,
#     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     "oil_dmo_volume_portion": 0.25,
#     "oil_dmo_fee_portion": 1.0,
#     "gas_dmo_volume_portion": 1.0,
#     "gas_dmo_fee_portion": 1.0,
#     "oil_dmo_holiday_duration": 60,
#     "gas_dmo_holiday_duration": 60,
#     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     "oil_carry_forward_depreciation": 0.0,
#     "gas_carry_forward_depreciation": 0.0,
# }
#
# params_gs = {
#     "sulfur_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
#     "electricity_revenue": OtherRevenue.ADDITION_TO_OIL_REVENUE,
#     "co2_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
#     "vat_rate": 0.0,
#     "year_inflation": None,
#     "inflation_rate": 0.0,
#     "inflation_rate_applied_to": InflationAppliedTo.CAPEX,
#     "cum_production_split_offset": 0.0,
#     # "cum_production_split_offset": np.array([0.2 for _ in range(10)]),
#     "depr_method": DeprMethod.PSC_DB,
#     "decline_factor": 2,
#     "sum_undepreciated_cost": False,
#     "is_dmo_end_weighted": False,
#     "tax_regime": TaxRegime.NAILED_DOWN,
#     "effective_tax_rate": 0.22,
#     "amortization": False,
#     "sunk_cost_method": SunkCostMethod.DEPRECIATED_TANGIBLE,
#     "regime": GrossSplitRegime.PERMEN_ESDM_13_2024,
#     "reservoir_type_permen_2024": VariableSplit132024.ReservoirType.MK,
#     "initial_amortization_year": InitialYearAmortizationIncurred.ONSTREAM_YEAR,
# }
#
# gs = GrossSplit(
#     **kwargs_gs,
#     # lifting=tuple([case.lifting_mangga, case.lifting_apel, case.lifting_nanas]),
#     lifting=tuple([case.lifting_mangga, case.lifting_apel]),
#     capital_cost=tuple([case.capital_mangga, case.capital_apel]),
#     intangible_cost=tuple([case.intangible_mangga, case.intangible_apel]),
#     opex=tuple([case.opex_mangga, case.opex_apel]),
#     asr_cost=tuple([case.asr_mangga, case.asr_apel]),
#     lbt_cost=tuple([case.lbt_mangga, case.lbt_apel]),
#     cost_of_sales=tuple([case.cos_mangga, case.cos_apel]),
# )
#
# gs.run(**params_gs)
# t1 = gs.get_summary()

# print('\t')
# print(f'Filetype: {type()}')
# print(f'Length: {len()}')
# print()
#
# print('\t')
# print(f'Filetype: {type(t1)}')
# print(f'Length: {len(t1)}')
# print('t1 = \n', t1)

# print('\t')
# print(f'Filetype: {type(t2)}')
# print(f'Length: {len(t2)}')
# print('t2 = \n', t2)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# kwargs_cr = {
#     "start_date": date(year=2023, month=1, day=1),
#     "end_date": date(year=2032, month=12, day=31),
#     "oil_onstream_date": date(year=2030, month=1, day=1),
#     "gas_onstream_date": date(year=2029, month=1, day=1),
#     "approval_year": 2026,
#     "is_pod_1": False,
#     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     "oil_ftp_is_available": True,
#     "oil_ftp_is_shared": True,
#     "oil_ftp_portion": 0.2,
#     "gas_ftp_is_available": True,
#     "gas_ftp_is_shared": True,
#     "gas_ftp_portion": 0.2,
#     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     "tax_split_type": TaxSplitTypeCR.CONVENTIONAL,
#     "condition_dict": dict,
#     "indicator_rc_icp_sliding": None,
#     "oil_ctr_pretax_share": 0.25,
#     "gas_ctr_pretax_share": 0.5,
#     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     "oil_ic_rate": 0.0,
#     "gas_ic_rate": 0.0,
#     "ic_is_available": False,
#     "oil_cr_cap_rate": 1.0,
#     "gas_cr_cap_rate": 1.0,
#     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     "oil_dmo_volume_portion": 0.25,
#     "oil_dmo_fee_portion": 0.25,
#     "oil_dmo_holiday_duration": 60,
#     "gas_dmo_volume_portion": 1.0,
#     "gas_dmo_fee_portion": 1.0,
#     "gas_dmo_holiday_duration": 60,
#     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     # "oil_carry_forward_depreciation": 100,
#     # "gas_carry_forward_depreciation": 50,
# }
#
# params_cr = {
#     "sulfur_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
#     "electricity_revenue": OtherRevenue.ADDITION_TO_OIL_REVENUE,
#     "co2_revenue": OtherRevenue.ADDITION_TO_OIL_REVENUE,
#     "vat_rate": 0.0,
#     "year_inflation": None,
#     "inflation_rate": 0.0,
#     "inflation_rate_applied_to": None,
#     "is_dmo_end_weighted": False,
#     "tax_regime": TaxRegime.NAILED_DOWN,
#     "effective_tax_rate": None,
#     "ftp_tax_regime": FTPTaxRegime.PDJP_20_2017,
#     "depr_method": DeprMethod.PSC_DB,
#     "decline_factor": 2,
#     "post_uu_22_year2001": True,
#     "oil_cost_of_sales_applied": False,
#     "gas_cost_of_sales_applied": False,
#     "sum_undepreciated_cost": False,
#     "sunk_cost_method": SunkCostMethod.DEPRECIATED_TANGIBLE,
# }
#
# cr = CostRecovery(
#     **kwargs_cr,
#     lifting=tuple([case.lifting_mangga, case.lifting_apel, case.lifting_nanas]),
#     capital_cost=tuple([case.capital_mangga, case.capital_apel]),
#     intangible_cost=tuple([case.intangible_mangga, case.intangible_apel]),
#     opex=tuple([case.opex_mangga, case.opex_apel]),
#     asr_cost=tuple([case.asr_mangga, case.asr_apel]),
#     lbt_cost=tuple([case.lbt_mangga, case.lbt_apel]),
#     cost_of_sales=tuple([case.cos_mangga, case.cos_apel]),
# )
#
# cr.run(**params_cr)
# cr.get_summary()

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

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from pyscnomics.api.adapter import get_setup_dict

data = {
    "setup": {
        "start_date": "01/01/2022",
        "end_date": "31/12/2035",
        "oil_onstream_date": "01/01/2024",
        "gas_onstream_date": None,
        "approval_year": None,
        "is_pod_1": True,
    },
    "summary_arguments": None,
    "grosssplit": None,
    "contract_arguments": None,
    "lifting": {
        "Oil Oil sources #1": {
            "start_year": 2022,
            "end_year": 2035,
            "prod_rate_baseline": [
                351000,
                351000,
                351000,
                351000,
                351000,
                351000,
                351000,
                351000,
                351000,
                351000,
                351000,
                351000,
            ],
            "prod_rate": None,
            "lifting_rate": [
                1003.01994465298,
                1752.43214595637,
                1660.63017314197,
                1646.06277634477,
                1303.30419844135,
                1206.56398735216,
                715.464597247899,
                575.351449953491,
                385.408188212388,
                177.896153424466,
                95.5608132946032,
                53.0868811432885
            ],
            "price": [
                75,
                75,
                75,
                75,
                75,
                75,
                75,
                75,
                75,
                75,
                75,
                75
            ],
            "prod_year": [
                2024,
                2025,
                2026,
                2027,
                2028,
                2029,
                2030,
                2031,
                2032,
                2033,
                2034,
                2035
            ],
            "fluid_type": "Oil"
        }
    },
    "capital": None,
    "intangible": None,
    "opex": None,
    "asr": None,
    "lbt": None,
    "cost_of_sales": None,
}

get_setup_dict(data=data)

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
