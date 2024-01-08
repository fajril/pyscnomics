import numpy as np
import pandas as pd
import time
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.Optimize.sensitivity import sensitivity_psc
from pyscnomics.dataset.sample import load_data
from pyscnomics.econ.selection import (NPVSelection, DiscountingMode, OptimizationTarget, FTPTaxRegime,
                                       OptimizationParameter)
from pyscnomics.tools.summary import get_summary

# pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# Initiating the psc object
psc = load_data(dataset_type='case1', contract_type='cost_recovery')

# Editing the CostRecovery attribute as the corresponding case 1
# FTP
psc.oil_ftp_is_available = True
psc.oil_ftp_is_shared = True
psc.oil_ftp_portion = 0.20
psc.gas_ftp_is_available = True
psc.gas_ftp_is_shared = True
psc.gas_ftp_portion = 0.20

# Split Pre Tax
psc.oil_ctr_pretax_share = 0.34722220
psc.gas_ctr_pretax_share = 0.5208330

# DMO
psc.oil_dmo_volume_portion = 0.25
psc.oil_dmo_fee_portion = 0.25
psc.oil_dmo_holiday_duration = 60
psc.gas_dmo_volume_portion = 0.25
psc.gas_dmo_fee_portion = 1
psc.gas_dmo_holiday_duration = 60

# PSC Arguments
tax_rate = 0.424
discount_rate_year = 2021
ftp_tax_regime = FTPTaxRegime.PRE_PDJP_20_2017
sunk_cost_reference_year = 2021
argument_psc = {'tax_rate': tax_rate,
                'ftp_tax_regime': ftp_tax_regime,
                'sunk_cost_reference_year': sunk_cost_reference_year
                }

# Summary Arguments
reference_year = 2022
inflation_rate = 0.1
discount_rate = 0.1
npv_mode = NPVSelection.NPV_SKK_NOMINAL_TERMS
discounting_mode = DiscountingMode.END_YEAR

argument_summary = {'reference_year': reference_year,
                    'inflation_rate': inflation_rate,
                    'discount_rate': discount_rate,
                    'npv_mode': npv_mode,
                    'discounting_mode': discounting_mode}

# Trialing the sensitivity
start_time = time.time()
result, list_of_psc = sensitivity_psc(steps=1,
                                      diff=0.1,
                                      contract=psc,
                                      contract_arguments=argument_psc,
                                      summary_arguments=argument_summary)
end_time = time.time()
print('Execution Time:', end_time - start_time)
#
# print(result['OPEX']['ctr_npv'])
# print(result['CAPEX']['ctr_npv'])
# print(result['Oil Price']['ctr_npv'])

# # Printing the result
# for key in result.keys():
#     print(key)
#     df = pd.DataFrame()
#     df['steps'] = result[key]['steps']
#     df['ctr_npv'] = result[key]['ctr_npv']
#     df['ctr_irr'] = result[key]['ctr_irr']
#     df['ctr_pi'] = result[key]['ctr_pi']
#     df['gov_take'] = result[key]['gov_take']
#     print(df)
#     print('\n')

# Output as an Excel file to plot
list_of_df = []
for key in result.keys():
    df = pd.DataFrame()
    df['steps'] = result[key]['steps']
    df['ctr_npv'] = result[key]['ctr_npv']
    df['ctr_irr'] = result[key]['ctr_irr']
    df['ctr_pi'] = result[key]['ctr_pi']
    df['gov_take'] = result[key]['gov_take']
    df['Label'] = key
    list_of_df.append(df)
df_all = pd.concat(list_of_df, ignore_index=True)
print(df_all)


# Checking the psc result
from pyscnomics.tools.table import get_table

# psc = list_of_psc[37]
#
# # Filling the contract in summary arguments
# argument_summary['contract'] = psc
# summary_test = get_summary(**argument_summary)
#
#
# psc_table_oil = pd.DataFrame()
# psc_table_oil['Year'] = psc.project_years
# psc_table_oil['Lifting'] = psc._oil_lifting.get_lifting_rate_arr()
# psc_table_oil['Price'] = psc._oil_lifting.get_price_arr()
# psc_table_oil['Revenue'] = psc._oil_revenue
# psc_table_oil['Depreciable'] = psc._oil_tangible_expenditures
# psc_table_oil['Opex'] = psc._oil_opex_expenditures
# psc_table_oil['ASR'] = psc._oil_asr_expenditures
# psc_table_oil['Depreciation'] = psc._oil_depreciation
# psc_table_oil['Non Capital'] = psc._oil_non_capital
# psc_table_oil['FTP'] = psc._oil_ftp
# psc_table_oil['FTP - CTR'] = psc._oil_ftp_ctr
# psc_table_oil['FTP - GOV'] = psc._oil_ftp_gov
# psc_table_oil['Investment Credit'] = psc._oil_ic_paid
# psc_table_oil['Unrecovered Cost'] = psc._oil_unrecovered_before_transfer
# psc_table_oil['Cost to Be Recovered'] = psc._oil_cost_to_be_recovered
# psc_table_oil['Cost Recovery'] = psc._oil_cost_recovery
# psc_table_oil['ETS Before Transfer'] = psc._oil_ets_before_transfer
# psc_table_oil['Transfer to Oil'] = psc._transfer_to_oil
# psc_table_oil['Transfer to Gas'] = psc._transfer_to_gas
# psc_table_oil['Unrec after Transfer'] = psc._oil_unrecovered_after_transfer
# psc_table_oil['Cost To Be Recovered After TF'] = psc._oil_cost_to_be_recovered_after_tf
# psc_table_oil['Cost Recovery After TF'] = psc._oil_cost_recovery_after_tf
# psc_table_oil['ETS After Transfer'] = psc._oil_ets_after_transfer
# psc_table_oil['Contractor Share Prior Tax'] = psc._oil_contractor_share
# psc_table_oil['Government Share'] = psc._oil_government_share
# psc_table_oil['DMO Volume'] = psc._oil_dmo_volume
# psc_table_oil['DMO Fee'] = psc._oil_dmo_fee
# psc_table_oil['DDMO'] = psc._oil_ddmo
# psc_table_oil['Taxable Income'] = psc._oil_taxable_income
# psc_table_oil['Tax'] = psc._oil_tax_payment
# psc_table_oil['Contractor Share'] = psc._oil_ctr_net_share
# psc_table_oil['Contractor Take'] = psc._gas_ctr_net_share
# psc_table_oil['Cashflow'] = psc._oil_cashflow
# psc_table_oil['Government Take'] = psc._oil_government_take
# psc_table_oil.loc['Column_Total'] = psc_table_oil.sum(numeric_only=True, axis=0)
# print(psc_table_oil, '\n')
#
# psc_table_gas = pd.DataFrame()
# psc_table_gas['Year'] = psc.project_years
# psc_table_gas['Lifting'] = psc._gas_lifting.get_lifting_rate_arr()
# psc_table_gas['Price'] = psc._gas_wap_price
# psc_table_gas['Revenue'] = psc._gas_revenue
# psc_table_gas['Depreciable'] = psc._gas_tangible_expenditures
# psc_table_gas['Opex'] = psc._gas_opex_expenditures
# psc_table_gas['ASR'] = psc._gas_asr_expenditures
# psc_table_gas['Depreciation'] = psc._gas_depreciation
# psc_table_gas['Non Capital'] = psc._gas_non_capital
# psc_table_gas['FTP'] = psc._gas_ftp
# psc_table_gas['FTP - CTR'] = psc._gas_ftp_ctr
# psc_table_gas['FTP - GOV'] = psc._gas_ftp_gov
# psc_table_gas['Investment Credit'] = psc._gas_ic_paid
# psc_table_gas['Unrecovered Cost'] = psc._gas_unrecovered_before_transfer
# psc_table_gas['Cost to Be Recovered'] = psc._gas_cost_to_be_recovered
# psc_table_gas['Cost Recovery'] = psc._gas_cost_recovery
# psc_table_gas['ETS Before Transfer'] = psc._gas_ets_before_transfer
# psc_table_gas['Transfer to Oil'] = psc._transfer_to_oil
# psc_table_gas['Transfer to Gas'] = psc._transfer_to_gas
# psc_table_gas['Unrec after Transfer'] = psc._gas_unrecovered_after_transfer
# psc_table_gas['Cost To Be Recovered After TF'] = psc._gas_cost_to_be_recovered_after_tf
# psc_table_gas['Cost Recovery After TF'] = psc._gas_cost_recovery_after_tf
# psc_table_gas['ETS After Transfer'] = psc._gas_ets_after_transfer
# psc_table_gas['Contractor Share Prior Tax'] = psc._gas_contractor_share
# psc_table_gas['Government Share'] = psc._gas_government_share
# psc_table_gas['DMO Volume'] = psc._gas_dmo_volume
# psc_table_gas['DMO Fee'] = psc._gas_dmo_fee
# psc_table_gas['DDMO'] = psc._gas_ddmo
# psc_table_gas['Taxable Income'] = psc._gas_taxable_income
# psc_table_gas['Tax'] = psc._gas_tax_payment
# psc_table_gas['Contractor Share'] = psc._gas_ctr_net_share
# psc_table_gas['Contractor Take'] = psc._gas_ctr_net_share
# psc_table_gas['Cashflow'] = psc._gas_cashflow
# psc_table_gas['Government Take'] = psc._gas_government_take
# psc_table_gas.loc['Column_Total'] = psc_table_gas.sum(numeric_only=True, axis=0)
# print(psc_table_gas, '\n')
#
# psc_table_consolidated = pd.DataFrame()
# psc_table_consolidated['Year'] = psc.project_years
# psc_table_consolidated['cnsltd_revenue'] = psc._consolidated_revenue
# psc_table_consolidated['cnsltd_non_capital'] = psc._consolidated_non_capital
# psc_table_consolidated['cnsltd_ic'] = psc._consolidated_ic
# psc_table_consolidated['cnsltd_ic_unrecovered'] = psc._consolidated_ic_unrecovered
# psc_table_consolidated['cnsltd_ic_paid'] = psc._consolidated_ic_paid
# psc_table_consolidated['cnsltd_unrecovered_before_transfer'] = psc._consolidated_unrecovered_before_transfer
# psc_table_consolidated['cnsltd_cost_recovery'] = psc._consolidated_cost_recovery_before_transfer
# psc_table_consolidated['cnsltd_ets_before_transfer'] = psc._consolidated_ets_before_transfer
# psc_table_consolidated['cnsltd_unrecovered_after_transfer'] = psc._consolidated_unrecovered_after_transfer
# psc_table_consolidated['cnsltd_cost_to_be_recovered_after_tf'] = psc._consolidated_cost_to_be_recovered_after_tf
# psc_table_consolidated['cnsltd_cost_recovery_after_tf'] = psc._consolidated_cost_recovery_after_tf
# psc_table_consolidated['cnsltd_ets_after_transfer'] = psc._consolidated_ets_after_transfer
# psc_table_consolidated['cnsltd_contractor_share'] = psc._consolidated_contractor_share
# psc_table_consolidated['cnsltd_government_share'] = psc._consolidated_government_share
# psc_table_consolidated['cnsltd_dmo_volume'] = psc._consolidated_dmo_volume
# psc_table_consolidated['cnsltd_dmo_fee'] = psc._consolidated_dmo_fee
# psc_table_consolidated['cnsltd_ddmo'] = psc._consolidated_ddmo
# psc_table_consolidated['cnsltd_taxable_income'] = psc._consolidated_taxable_income
# psc_table_consolidated['cnsltd_tax_due'] = psc._consolidated_tax_due
# psc_table_consolidated['cnsltd_unpaid_tax_balance'] = psc._consolidated_unpaid_tax_balance
# psc_table_consolidated['cnsltd_tax_payment'] = psc._consolidated_tax_payment
# psc_table_consolidated['cnsltd_ctr_net_share'] = psc._consolidated_ctr_net_share
# psc_table_consolidated['cnsltd_contractor_take'] = psc._consolidated_contractor_take
# psc_table_consolidated['cnsltd_government_take'] = psc._consolidated_government_take
# psc_table_consolidated['cnsltd_cashflow'] = psc._consolidated_cashflow
# psc_table_consolidated['cum. cnsltd_cashflow'] = np.cumsum(psc._consolidated_cashflow)
# psc_table_consolidated.loc['Column_Total'] = psc_table_consolidated.sum(numeric_only=True, axis=0)
# print(psc_table_consolidated, '\n')
#
# for key, value in summary_test.items():
#     print(key, ':', value)
#
#
# psc_table_oil['Label'] = 'Oil Table'
# psc_table_gas['Label'] = 'Gas Table'
# psc_table_consolidated['Label'] = 'Consolidated Table'
#
# list_of_df = [psc_table_oil, psc_table_gas, psc_table_consolidated]
# df_combined = pd.concat(list_of_df)
# df_combined.to_excel('df_09 Gas Price.xlsx')