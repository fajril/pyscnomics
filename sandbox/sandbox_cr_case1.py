import numpy as np
import pandas as pd

from pyscnomics.dataset.sample import load_data, load_testing

'''
CR MS DN 1
'''

# pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# Initiating the psc object
psc = load_data(dataset_type='case1', contract_type='cost_recovery')

# Editing the CostRecovery attribute as the corresponding case 1
# FTP
psc.oil_ftp_is_available = True
psc.oil_ftp_is_shared = True
psc.oil_ftp_portion = 0.2
psc.gas_ftp_is_available = True
psc.gas_ftp_is_shared = True
psc.gas_ftp_portion = 0.2

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

tax_rate = 0.424

psc.run(tax_rate=tax_rate)

# base = np.asarray(load_testing(dataset_type='case1', key='Unrec Cost'))
# engine = psc._oil_unrecovered_before_transfer
# #
# print(base)
# print(engine)
# print(base-engine)
# input()

psc_table_oil = pd.DataFrame()
psc_table_oil['Year'] = psc.project_years
psc_table_oil['Lifting'] = psc._oil_lifting.lifting_rate_arr()
psc_table_oil['Price'] = psc._oil_lifting.lifting_price_arr()
psc_table_oil['Revenue'] = psc._oil_revenue
psc_table_oil['Depreciable'] = psc._oil_tangible.expenditures()
psc_table_oil['Opex'] = psc._oil_opex.expenditures()
psc_table_oil['ASR'] = psc._oil_asr.expenditures()
psc_table_oil['Depreciation'] = psc._oil_depreciation
psc_table_oil['Non Capital'] = psc._oil_non_capital
psc_table_oil['FTP'] = psc._oil_ftp
psc_table_oil['FTP - CTR'] = psc._oil_ftp_ctr
psc_table_oil['FTP - GOV'] = psc._oil_ftp_gov
psc_table_oil['Investment Credit'] = psc._oil_ic_paid
psc_table_oil['Unrecovered Cost'] = psc._oil_unrecovered_before_transfer
psc_table_oil['Cost to Be Recovered'] = psc._oil_cost_to_be_recovered
psc_table_oil['Cost Recovery'] = psc._oil_cost_recovery
psc_table_oil['ETS Before Transfer'] = psc._oil_ets_before_transfer
psc_table_oil['Transfer to Oil'] = psc._transfer_to_oil
psc_table_oil['Transfer to Gas'] = psc._transfer_to_gas
psc_table_oil['Unrec after Transfer'] = psc._oil_unrecovered_after_transfer
psc_table_oil['Cost To Be Recovered After TF'] = psc._oil_cost_to_be_recovered_after_tf
psc_table_oil['Cost Recovery After TF'] = psc._oil_cost_recovery_after_tf
psc_table_oil['ETS After Transfer'] = psc._oil_ets_after_transfer
psc_table_oil['Contractor Share'] = psc._oil_contractor_share
psc_table_oil['Government Share'] = psc._oil_government_share
psc_table_oil['DMO Volume'] = psc._oil_dmo_volume
psc_table_oil['DMO Fee'] = psc._oil_dmo_fee
psc_table_oil['DDMO'] = psc._oil_ddmo
psc_table_oil['Taxable Income'] = psc._oil_taxable_income
psc_table_oil['Tax'] = psc._oil_tax_payment
psc_table_oil['Contractor Share'] = psc._oil_ctr_net_share
psc_table_oil['Contractor Take'] = psc._gas_ctr_net_share
psc_table_oil['Cashflow'] = psc._oil_cashflow
psc_table_oil['Government Take'] = psc._oil_government_take
print(psc_table_oil, '\n')

print('Tax Payment: ', np.sum(psc._oil_tax_payment), '\n')


# psc_table.to_excel('Cost Recovery Case 1 Table from Pyscnomics.xlsx')

psc_table_gas = pd.DataFrame()
psc_table_gas['Year'] = psc.project_years
psc_table_gas['Lifting'] = psc._gas_lifting.lifting_rate_arr()
psc_table_gas['Price'] = psc._gas_lifting.lifting_price_arr()
psc_table_gas['Revenue'] = psc._gas_revenue
psc_table_gas['Depreciable'] = psc._gas_tangible.expenditures()
psc_table_gas['Opex'] = psc._gas_opex.expenditures()
psc_table_gas['ASR'] = psc._gas_asr.expenditures()
psc_table_gas['Depreciation'] = psc._gas_depreciation
psc_table_gas['Non Capital'] = psc._gas_non_capital
psc_table_gas['FTP'] = psc._gas_ftp
psc_table_gas['FTP - CTR'] = psc._gas_ftp_ctr
psc_table_gas['FTP - GOV'] = psc._gas_ftp_gov
psc_table_gas['Investment Credit'] = psc._gas_ic_paid
psc_table_gas['Unrecovered Cost'] = psc._gas_unrecovered_before_transfer
psc_table_gas['Cost to Be Recovered'] = psc._gas_cost_to_be_recovered
psc_table_gas['Cost Recovery'] = psc._gas_cost_recovery
psc_table_gas['ETS Before Transfer'] = psc._gas_ets_before_transfer
psc_table_gas['Transfer to Oil'] = psc._transfer_to_oil
psc_table_gas['Transfer to Gas'] = psc._transfer_to_gas
psc_table_gas['Unrec after Transfer'] = psc._gas_unrecovered_after_transfer
psc_table_gas['Cost To Be Recovered After TF'] = psc._gas_cost_to_be_recovered_after_tf
psc_table_gas['Cost Recovery After TF'] = psc._gas_cost_recovery_after_tf
psc_table_gas['ETS After Transfer'] = psc._gas_ets_after_transfer
psc_table_gas['Contractor Share'] = psc._gas_contractor_share
psc_table_gas['Government Share'] = psc._gas_government_share
psc_table_gas['DMO Volume'] = psc._gas_dmo_volume
psc_table_gas['DMO Fee'] = psc._gas_dmo_fee
psc_table_gas['DDMO'] = psc._gas_ddmo
psc_table_gas['Taxable Income'] = psc._gas_taxable_income
psc_table_gas['Tax'] = psc._gas_tax_payment
psc_table_gas['Contractor Share'] = psc._gas_ctr_net_share
psc_table_gas['Contractor Take'] = psc._gas_ctr_net_share
psc_table_gas['Cashflow'] = psc._gas_cashflow
psc_table_gas['Government Take'] = psc._gas_government_take
print(psc_table_gas, '\n')
print('Tax Payment: ', np.sum(psc._gas_tax_payment), '\n')
#

psc_table_consolidated = pd.DataFrame()
psc_table_consolidated['Year'] = psc.project_years
psc_table_consolidated['consolidated_non_capital'] = psc._consolidated_non_capital
psc_table_consolidated['consolidated_ic'] = psc._consolidated_ic
psc_table_consolidated['consolidated_ic_unrecovered'] = psc._consolidated_ic_unrecovered
psc_table_consolidated['consolidated_ic_paid'] = psc._consolidated_ic_paid
psc_table_consolidated['consolidated_unrecovered_before_transfer'] = psc._consolidated_unrecovered_before_transfer
psc_table_consolidated['consolidated_cost_to_be_recovered'] = psc._consolidated_cost_to_be_recovered
psc_table_consolidated['consolidated_cost_recovery'] = psc._consolidated_cost_recovery
psc_table_consolidated['consolidated_ets_before_transfer'] = psc._consolidated_ets_before_transfer
psc_table_consolidated['consolidated_unrecovered_after_transfer'] = psc._consolidated_unrecovered_after_transfer
psc_table_consolidated['consolidated_cost_to_be_recovered_after_tf'] = psc._consolidated_cost_to_be_recovered_after_tf
psc_table_consolidated['consolidated_cost_recovery_after_tf'] = psc._consolidated_cost_recovery_after_tf
psc_table_consolidated['consolidated_ets_after_transfer'] = psc._consolidated_ets_after_transfer
psc_table_consolidated['consolidated_contractor_share'] = psc._consolidated_contractor_share
psc_table_consolidated['consolidated_government_share'] = psc._consolidated_government_share
psc_table_consolidated['consolidated_dmo_volume'] = psc._consolidated_dmo_volume
psc_table_consolidated['consolidated_dmo_fee'] = psc._consolidated_dmo_fee
psc_table_consolidated['consolidated_ddmo'] = psc._consolidated_ddmo
psc_table_consolidated['consolidated_taxable_income'] = psc._consolidated_taxable_income
psc_table_consolidated['consolidated_tax_payment'] = psc._consolidated_tax_payment
psc_table_consolidated['consolidated_ctr_net_share'] = psc._consolidated_ctr_net_share
psc_table_consolidated['consolidated_contractor_take'] = psc._consolidated_contractor_take
psc_table_consolidated['consolidated_government_take'] = psc._consolidated_government_take
psc_table_consolidated['consolidated_cashflow'] = psc._consolidated_cashflow
print(psc_table_consolidated, '\n')
print('Tax Payment: ', np.sum(psc._consolidated_taxable_income), '\n')

# df_comparison = pd.DataFrame()
# # Calculated result
# engine = (psc._oil_ftp +
#           psc._oil_ic +
#           psc._oil_cost_recovery +
#           psc._oil_contractor_share +
#           psc._oil_government_share)
#
# # Expected result
# base = psc._oil_revenue
#
# df_comparison['Year'] = psc.project_years
# df_comparison['Base'] = base
# df_comparison['Engine'] = engine
# df_comparison['Diff'] = base - engine
# print(df_comparison)

# gross_revenue_calc = psc._oil_contractor_take + psc._oil_government_take
# gross_revenue_engine = psc._oil_revenue + psc._gas_revenue
#
# print(gross_revenue_calc)
# print(gross_revenue_engine)
