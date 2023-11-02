import numpy as np
import pandas as pd
import time
from datetime import datetime

from pyscnomics.econ.selection import FTPTaxRegime
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.selection import FluidType
from pyscnomics.econ.costs import Tangible, Intangible, OPEX, ASR
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit

# pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

"""
Cost Recovery
"""

# Defining Start Date and End Date
psc_1_start_date = datetime.strptime("01/01/2019", '%d/%m/%Y').date()
psc_1_end_date = datetime.strptime("22/4/2020", '%d/%m/%Y').date()

# Defining the Gas lifting data
psc1_gas_lifting = Lifting(
    start_year=2019,
    end_year=2020,
    lifting_rate=np.array([0.00158446786291200, 0.00098961346458056]),
    price=np.array([6.260, 6.260]),
    prod_year=np.array([2019, 2020]),
    ghv=np.array([1047.0, 1047.0]),
    fluid_type=FluidType.GAS)

# Defining the Gas Tangible Data - Drilling Tangible
psc1_gas_tangible = Tangible(
    start_year=2019,
    end_year=2020,
    cost=np.array([3.36367743703704000, 0.80273224043715800]),
    expense_year=np.array([2019, 2020]),
    pis_year=np.array([2019, 2020]),
    useful_life=np.array([5, 5]),
    depreciation_factor=np.array([0.25, 0.25]),
    cost_allocation=FluidType.GAS)

# Defining the Gas Intangible Data
psc1_gas_intang = Intangible(
    start_year=2019,
    end_year=2020,
    cost=np.array([9.532633600000]),
    expense_year=np.array([2019]),
    cost_allocation=FluidType.GAS)

# Defining the Gas OPEX Data
psc1_gas_opex_cost = OPEX(
    start_year=2019,
    end_year=2020,
    fixed_cost=np.array([2.076908222642980, 1.297582047244550]),
    expense_year=np.array([2019, 2020]),
    cost_allocation=FluidType.GAS)

# Defining the Gas ASR Data
psc1_gas_asr_cost_opx = OPEX(
    start_year=2019,
    end_year=2020,
    fixed_cost=np.array([0.035515809523809900, 0.010965263596148900]),
    expense_year=np.array([2019, 2020]),
    cost_allocation=FluidType.GAS)

psc1_gas_asr_cost = ASR(
    start_year=2019,
    end_year=2020,
    cost=np.array([0]),
    expense_year=np.array([2019]),
    cost_allocation=FluidType.GAS)

# Parsing the fiscal terms into Cost Recovery
psc1 = CostRecovery(
    start_date=psc_1_start_date,
    end_date=psc_1_end_date,
    lifting=tuple([psc1_gas_lifting]),
    tangible_cost=tuple([psc1_gas_tangible]),
    intangible_cost=tuple([psc1_gas_intang]),
    opex=tuple([psc1_gas_opex_cost, psc1_gas_asr_cost_opx]),
    asr_cost=tuple([psc1_gas_asr_cost]),
    oil_ftp_is_available=True,
    oil_ftp_is_shared=True,
    oil_ftp_portion=0.2,
    gas_ftp_is_available=True,
    gas_ftp_is_shared=True,
    gas_ftp_portion=0.2,
    oil_ctr_pretax_share=0.3361,
    gas_ctr_pretax_share=0.57692307692307700,
    oil_dmo_volume_portion=0.25,
    oil_dmo_fee_portion=0.15,
    oil_dmo_holiday_duration=0,
    gas_dmo_volume_portion=0.25,
    gas_dmo_fee_portion=0.15,
    gas_dmo_holiday_duration=0)

start_time = time.time()
ftp_tax_regime = FTPTaxRegime.PRE_2017
eff_tax_rate = 0.48
psc1.run(ftp_tax_regime=ftp_tax_regime, tax_rate=eff_tax_rate)
end_time = time.time()
print('Execution Time: ', end_time - start_time, '\n')

psc1_table_oil = pd.DataFrame()
psc1_table_oil['Year'] = psc1.project_years
psc1_table_oil['Lifting'] = psc1._oil_lifting.lifting_rate_arr()
psc1_table_oil['Price'] = psc1._oil_lifting.lifting_price_arr()
psc1_table_oil['Revenue'] = psc1._oil_revenue
psc1_table_oil['Depreciable'] = psc1._oil_tangible.expenditures()
psc1_table_oil['Intangible'] = psc1._oil_intangible.expenditures()
psc1_table_oil['Opex'] = psc1._oil_opex.expenditures()
psc1_table_oil['ASR'] = psc1._oil_asr.expenditures()
psc1_table_oil['Depreciation'] = psc1._oil_depreciation
psc1_table_oil['Non Capital'] = psc1._oil_non_capital
psc1_table_oil['FTP'] = psc1._oil_ftp
psc1_table_oil['FTP - CTR'] = psc1._oil_ftp_ctr
psc1_table_oil['FTP - GOV'] = psc1._oil_ftp_gov
psc1_table_oil['Investment Credit'] = psc1._oil_ic_paid
psc1_table_oil['Unrecovered Cost'] = psc1._oil_unrecovered_before_transfer
psc1_table_oil['Cost to Be Recovered'] = psc1._oil_cost_to_be_recovered
psc1_table_oil['Cost Recovery'] = psc1._oil_cost_recovery
psc1_table_oil['ETS Before Transfer'] = psc1._oil_ets_before_transfer
psc1_table_oil['Transfer to Oil'] = psc1._transfer_to_oil
psc1_table_oil['Transfer to Gas'] = psc1._transfer_to_gas
psc1_table_oil['Unrec after Transfer'] = psc1._oil_unrecovered_after_transfer
psc1_table_oil['Cost To Be Recovered After TF'] = psc1._oil_cost_to_be_recovered_after_tf
psc1_table_oil['Cost Recovery After TF'] = psc1._oil_cost_recovery_after_tf
psc1_table_oil['ETS After Transfer'] = psc1._oil_ets_after_transfer
psc1_table_oil['Contractor Share'] = psc1._oil_contractor_share
psc1_table_oil['Government Share'] = psc1._oil_government_share
psc1_table_oil['DMO Volume'] = psc1._oil_dmo_volume
psc1_table_oil['DMO Fee'] = psc1._oil_dmo_fee
psc1_table_oil['DDMO'] = psc1._oil_ddmo
psc1_table_oil['Taxable Income'] = psc1._oil_taxable_income
psc1_table_oil['Tax'] = psc1._oil_tax_payment
psc1_table_oil['Net Contractor Share'] = psc1._oil_ctr_net_share
psc1_table_oil['Contractor Take'] = psc1._gas_ctr_net_share
psc1_table_oil['Cashflow'] = psc1._oil_cashflow
psc1_table_oil['Government Take'] = psc1._oil_government_take
psc1_table_oil.loc['Column_Total'] = psc1_table_oil.sum(numeric_only=True, axis=0)
print(psc1_table_oil, '\n')

psc1_table_gas = pd.DataFrame()
psc1_table_gas['Year'] = psc1.project_years
psc1_table_gas['Lifting'] = psc1._gas_lifting.lifting_rate_arr()
psc1_table_gas['Price'] = psc1._gas_lifting.lifting_price_arr()
psc1_table_gas['Revenue'] = psc1._gas_revenue
psc1_table_gas['Depreciable'] = psc1._gas_tangible.expenditures()
psc1_table_gas['Intangible'] = psc1._gas_intangible.expenditures()
psc1_table_gas['Opex'] = psc1._gas_opex.expenditures()
psc1_table_gas['ASR'] = psc1._gas_asr.expenditures()
psc1_table_gas['Depreciation'] = psc1._gas_depreciation
psc1_table_gas['Non Capital'] = psc1._gas_non_capital
psc1_table_gas['FTP'] = psc1._gas_ftp
psc1_table_gas['FTP - CTR'] = psc1._gas_ftp_ctr
psc1_table_gas['FTP - GOV'] = psc1._gas_ftp_gov
psc1_table_gas['Investment Credit'] = psc1._gas_ic_paid
psc1_table_gas['Unrecovered Cost'] = psc1._gas_unrecovered_before_transfer
psc1_table_gas['Cost to Be Recovered'] = psc1._gas_cost_to_be_recovered
psc1_table_gas['Cost Recovery'] = psc1._gas_cost_recovery
psc1_table_gas['ETS Before Transfer'] = psc1._gas_ets_before_transfer
psc1_table_gas['Transfer to Oil'] = psc1._transfer_to_oil
psc1_table_gas['Transfer to Gas'] = psc1._transfer_to_gas
psc1_table_gas['Unrec after Transfer'] = psc1._gas_unrecovered_after_transfer
psc1_table_gas['Cost To Be Recovered After TF'] = psc1._gas_cost_to_be_recovered_after_tf
psc1_table_gas['Cost Recovery After TF'] = psc1._gas_cost_recovery_after_tf
psc1_table_gas['ETS After Transfer'] = psc1._gas_ets_after_transfer
psc1_table_gas['Contractor Share'] = psc1._gas_contractor_share
psc1_table_gas['Government Share'] = psc1._gas_government_share
psc1_table_gas['DMO Volume'] = psc1._gas_dmo_volume
psc1_table_gas['DMO Fee'] = psc1._gas_dmo_fee
psc1_table_gas['DDMO'] = psc1._gas_ddmo
psc1_table_gas['Taxable Income'] = psc1._gas_taxable_income
psc1_table_gas['Tax'] = psc1._gas_tax_payment
psc1_table_gas['Net Contractor Share'] = psc1._gas_ctr_net_share
psc1_table_gas['Contractor Take'] = psc1._gas_ctr_net_share
psc1_table_gas['Cashflow'] = psc1._gas_cashflow
psc1_table_gas['Government Take'] = psc1._gas_government_take
psc1_table_gas.loc['Column_Total'] = psc1_table_gas.sum(numeric_only=True, axis=0)
print(psc1_table_gas, '\n')

"""
Gross Split
"""
# Defining Start Date and End Date
psc_2_start_date = datetime.strptime("23/4/2020", '%d/%m/%Y').date()
psc_2_end_date = datetime.strptime("22/4/2027", '%d/%m/%Y').date()

# Defining the Gas lifting data
psc2_gas_lifting = Lifting(
    start_year=2020,
    end_year=2027,
    lifting_rate=np.array(
        [0.00221568324370692000, 0.00320769606628721000, 0.00329284116326400000, 0.00337370744832000000,
         0.00344718555313867000, 0.00340400062841705000, 0.00332543155814400000, 0.00200043667046400000]),
    price=np.array([6.2600, 6.2600, 6.2600, 6.2600, 6.2600, 6.2600, 6.2600, 6.2600]),
    prod_year=np.array([2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027]),
    ghv=np.array([1047.0, 1047.0, 1047.0, 1047.0, 1047.0, 1047.0, 1047.0, 1047.0, ]),
    fluid_type=FluidType.GAS)

# Defining the Gas Tangible Data - Drilling Tangible
psc2_gas_tangible = Tangible(
    start_year=2020,
    end_year=2027,
    cost=np.array([1.959561038251370, 2.834780000000000]),
    expense_year=np.array([2020, 2025]),
    pis_year=np.array([2020, 2025]),
    useful_life=np.array([5, 5]),
    depreciation_factor=np.array([0.25, 0.25]),
    cost_allocation=FluidType.GAS)

# Defining the Gas Intangible Data
psc2_gas_intang = Intangible(
    start_year=2020,
    end_year=2027,
    cost=np.array([0]),
    expense_year=np.array([2020]),
    cost_allocation=FluidType.GAS)

# Defining the Gas OPEX Data
psc2_gas_opex_cost = OPEX(
    start_year=2020,
    end_year=2027,
    fixed_cost=np.array(
        [3.1374561521272200, 3.8348277754882300, 3.9438595100171400, 4.0408953543109500, 4.1146574450911200,
         4.2724109557000400, 5.0576546904997800, 4.4627942473087100, ]),
    expense_year=np.array([2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027]),
    cost_allocation=FluidType.GAS)

psc2_gas_lbt_cost = OPEX(
    start_year=2020,
    end_year=2027,
    fixed_cost=np.array([0.63640338233533300, 1.06286476325927000, 1.09105051487026000, 1.11733210409363000, 1.14029666892703000, 1.29608012023605000, 1.12478850166335000, 0.78684662704513200]),
    expense_year=np.array([2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027]),
    cost_allocation=FluidType.GAS)

# Defining VAT and Import Duty
psc2_gas_vat = OPEX(
    start_year=2020,
    end_year=2027,
    fixed_cost=np.array([0.3965062479576550, 0.2867125283653920, 0.2947836073052500, 0.3019666899432990, 0.3074269319547460, 0.5538846425099840, 0.3772323444857470, 0.3331977740376320,]),
    expense_year=np.array([2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027,]),
    cost_allocation=FluidType.GAS)

psc2_gas_import_duty = OPEX(
    start_year=2020,
    end_year=2027,
    fixed_cost=np.array([0.159965353, 0.055922218, 0.057496452, 0.058897486, 0.059962486, 0.227566547, 0.073577773, 0.064988993]),
    expense_year=np.array([2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027]),
    cost_allocation=FluidType.GAS)

# Defining the Gas ASR Data
psc2_gas_asr_cost_opx = OPEX(
    start_year=2020,
    end_year=2027,
    fixed_cost=np.array(
        [0.026513186, 0.038355043, 0.038355043, 0.038355043, 0.038355043, 0.038355043, 0.038355043, 0.038355043, ]),
    expense_year=np.array([2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, ]),
    cost_allocation=FluidType.GAS)

psc2_gas_asr_cost = ASR(
    start_year=2020,
    end_year=2027,
    cost=np.array([0]),
    expense_year=np.array([2020]),
    cost_allocation=FluidType.GAS)

# Parsing the fiscal terms into Gross Split
psc2 = GrossSplit(
    start_date=psc_2_start_date,
    end_date=psc_2_end_date,
    lifting=tuple([psc2_gas_lifting]),
    tangible_cost=tuple([psc2_gas_tangible]),
    intangible_cost=tuple([psc2_gas_intang]),
    opex=tuple([psc2_gas_opex_cost, psc2_gas_asr_cost_opx, psc2_gas_lbt_cost]),
    asr_cost=tuple([psc2_gas_asr_cost]),
    field_status='No POD',
    field_loc='Onshore',
    res_depth='<=2500',
    infra_avail='Well Developed',
    res_type='Conventional',
    api_oil='>=25',
    domestic_use='70<=x<100',
    prod_stage='Primer',
    co2_content='<5',
    h2s_content='<100',
    ctr_effective_tax_rate=0.40,
    base_split_ctr_oil=0.43,
    base_split_ctr_gas=0.48,
    split_ministry_disc=0.0,
    oil_dmo_volume_portion=0.25,
    oil_dmo_fee_portion=1.0,
    oil_dmo_holiday_duration=0,
    gas_dmo_volume_portion=0.25,
    gas_dmo_fee_portion=1.0,
    gas_dmo_holiday_duration=0)

start_time = time.time()
psc2.run()
end_time = time.time()
print('Execution Time: ', end_time - start_time, '\n')

psc2_table_oil = pd.DataFrame()
psc2_table_oil['Years'] = psc2.project_years
psc2_table_oil['Lifting'] = psc2._oil_lifting.lifting_rate
psc2_table_oil['Price'] = psc2._oil_lifting.lifting_price_arr()
psc2_table_oil['Revenue'] = psc2._oil_revenue
psc2_table_oil['Base Split'] = psc2._oil_base_split
psc2_table_oil['Variable Split'] = psc2._var_split_array
psc2_table_oil['Progressive Split'] = psc2._oil_prog_split
psc2_table_oil['Contractor Split'] = psc2._oil_ctr_split
psc2_table_oil['Government Share'] = psc2._oil_gov_share
psc2_table_oil['Contractor Share'] = psc2._oil_ctr_share_before_transfer
psc2_table_oil['Depreciation'] = psc2._oil_depreciation
psc2_table_oil['Opex'] = psc2._oil_opex.expenditures()
psc2_table_oil['ASR'] = psc2._oil_asr.expenditures()
psc2_table_oil['Non Capital'] = psc2._oil_non_capital
psc2_table_oil['Total Expenses'] = psc2._oil_total_expenses
psc2_table_oil['Cost To Be Deducted'] = psc2._oil_cost_tobe_deducted
psc2_table_oil['Carry Forward Cost'] = psc2._oil_carward_deduct_cost
psc2_table_oil['Deductible Cost'] = psc2._oil_deductible_cost
psc2_table_oil['Transfer To Oil'] = psc2._transfer_to_oil
psc2_table_oil['Transfer To Gas'] = psc2._transfer_to_gas
psc2_table_oil['Carry Forward Cost after TF'] = psc2._oil_carward_cost_aftertf
psc2_table_oil['CTR Share After TF'] = psc2._oil_ctr_share_after_transfer
psc2_table_oil['CTR Net Operating Profit'] = psc2._oil_net_operating_profit
psc2_table_oil['DMO Volume'] = psc2._oil_dmo_volume
psc2_table_oil['DMO Fee'] = psc2._oil_dmo_fee
psc2_table_oil['DDMO'] = psc2._oil_ddmo
psc2_table_oil['Taxable Income'] = psc2._oil_taxable_income
psc2_table_oil['Tax'] = psc2._oil_tax
psc2_table_oil['Net CTR Share'] = psc2._oil_ctr_net_share
psc2_table_oil['CTR CashFlow'] = psc2._oil_ctr_cashflow
psc2_table_oil['Government Take'] = psc2._oil_gov_take
psc2_table_oil.loc['Column_Total'] = psc2_table_oil.sum(numeric_only=True, axis=0)
print(psc2_table_oil, '\n')

psc2_table_gas = pd.DataFrame()
psc2_table_gas['Years'] = psc2.project_years
psc2_table_gas['Lifting'] = psc2._gas_lifting.lifting_rate
psc2_table_gas['Price'] = psc2._gas_lifting.lifting_price_arr()
psc2_table_gas['Revenue'] = psc2._gas_revenue
psc2_table_gas['Base Split'] = psc2._gas_base_split
psc2_table_gas['Variable Split'] = psc2._var_split_array
psc2_table_gas['Progressive Split'] = psc2._gas_prog_split
psc2_table_gas['Contractor Split'] = psc2._gas_ctr_split
psc2_table_gas['Government Share'] = psc2._gas_gov_share
psc2_table_gas['Contractor Share'] = psc2._gas_ctr_share_before_transfer
psc2_table_gas['Depreciation'] = psc2._gas_depreciation
psc2_table_gas['Opex'] = psc2._gas_opex.expenditures()
psc2_table_gas['ASR'] = psc2._gas_asr.expenditures()
psc2_table_gas['Non Capital'] = psc2._gas_non_capital
psc2_table_gas['Total Expenses'] = psc2._gas_total_expenses
psc2_table_gas['Cost To Be Deducted'] = psc2._gas_cost_tobe_deducted
psc2_table_gas['Carry Forward Cost'] = psc2._gas_carward_deduct_cost
psc2_table_gas['Deductible Cost'] = psc2._gas_deductible_cost
psc2_table_gas['Transfer To Oil'] = psc2._transfer_to_oil
psc2_table_gas['Transfer To Gas'] = psc2._transfer_to_gas
psc2_table_gas['Carry Forward Cost after TF'] = psc2._gas_carward_cost_aftertf
psc2_table_gas['CTR Share After TF'] = psc2._gas_ctr_share_after_transfer
psc2_table_gas['CTR Net Operating Profit'] = psc2._gas_net_operating_profit
psc2_table_gas['DMO Volume'] = psc2._gas_dmo_volume
psc2_table_gas['DMO Fee'] = psc2._gas_dmo_fee
psc2_table_gas['DDMO'] = psc2._gas_ddmo
psc2_table_gas['Taxable Income'] = psc2._gas_taxable_income
psc2_table_gas['Tax'] = psc2._gas_tax
psc2_table_gas['Net CTR Share'] = psc2._gas_ctr_net_share
psc2_table_gas['CTR CashFlow'] = psc2._gas_ctr_cashflow
psc2_table_gas['Government Take'] = psc2._gas_gov_take
psc2_table_gas.loc['Column_Total'] = psc2_table_gas.sum(numeric_only=True, axis=0)
print(psc2_table_gas, '\n')
