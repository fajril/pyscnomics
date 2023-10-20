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

psc_table = pd.DataFrame()
psc_table['Year'] = psc.project_years
# psc_table['Lifting'] = psc._oil_lifting.lifting_rate_arr()
# psc_table['Price'] = psc._oil_lifting.lifting_price_arr()
# psc_table['Revenue'] = psc._oil_revenue
# psc_table['Depreciable'] = psc._oil_tangible.expenditures()
# psc_table['Opex'] = psc._oil_opex.expenditures()
# psc_table['ASR'] = psc._oil_asr.expenditures()
# psc_table['Depreciation'] = psc._oil_depreciation
# psc_table['Non Capital'] = psc._oil_non_capital
# psc_table['FTP'] = psc._oil_ftp
# psc_table['FTP - CTR'] = psc._oil_ftp_ctr
# psc_table['FTP - GOV'] = psc._oil_ftp_gov
# psc_table['Investment Credit'] = psc._oil_ic_paid
# psc_table['Unrecovered Cost'] = psc._oil_unrecovered_before_transfer
# psc_table['Cost to Be Recovered'] = psc._oil_cost_to_be_recovered
# psc_table['Cost Recovery'] = psc._oil_cost_recovery
# psc_table['Transfer to Oil'] = psc._transfer_to_oil
# psc_table['Transfer to Gas'] = psc._transfer_to_gas
# psc_table['Unrec after Transfer'] = psc._oil_unrecovered_after_transfer
# psc_table['ETS Before Transfer'] = psc._oil_ets_before_transfer
# psc_table['ETS After Transfer'] = psc._oil_ets_after_transfer
# psc_table['Contractor Share'] = psc._oil_contractor_share
# psc_table['Government Share'] = psc._oil_government_share
# psc_table['DMO Volume'] = psc._oil_dmo_volume
# psc_table['DMO Fee'] = psc._oil_dmo_fee
# psc_table['DDMO'] = psc._oil_ddmo
psc_table['Taxable Income'] = psc._oil_taxable_income
psc_table['Tax'] = psc._oil_ftp_tax_payment
# psc_table['Contractor Take'] = psc._gas_contractor_take
# psc_table['Government Take'] = psc._gas_government_take
# psc_table['Cashflow'] = psc._gas_cashflow
print(psc_table, '\n')


# psc_table.to_excel('Cost Recovery Case 1 Table from Pyscnomics.xlsx')

psc_table = pd.DataFrame()
psc_table['Year'] = psc.project_years
# psc_table['Lifting'] = psc._gas_lifting.lifting_rate_arr()
# psc_table['Price'] = psc._gas_lifting.lifting_price_arr()
# psc_table['Revenue'] = psc._gas_revenue
# psc_table['Depreciable'] = psc._gas_tangible.expenditures()
# psc_table['Opex'] = psc._gas_opex.expenditures()
# psc_table['ASR'] = psc._gas_asr.expenditures()
# psc_table['Depreciation'] = psc._gas_depreciation
# psc_table['Non Capital'] = psc._gas_non_capital
# psc_table['FTP'] = psc._gas_ftp
# psc_table['FTP - CTR'] = psc._gas_ftp_ctr
# psc_table['FTP - GOV'] = psc._gas_ftp_gov
# psc_table['Investment Credit'] = psc._gas_ic_paid
# psc_table['Unrecovered Cost'] = psc._gas_unrecovered_before_transfer
# psc_table['Cost to Be Recovered'] = psc._gas_cost_to_be_recovered
# psc_table['Cost Recovery'] = psc._gas_cost_recovery
# psc_table['Transfer to Oil'] = psc._transfer_to_oil
# psc_table['Transfer to Gas'] = psc._transfer_to_gas
# psc_table['Unrec after Transfer'] = psc._gas_unrecovered_after_transfer
# psc_table['ETS Before Transfer'] = psc._gas_ets_before_transfer
# psc_table['ETS After Transfer'] = psc._gas_ets_after_transfer
# psc_table['Contractor Share'] = psc._gas_contractor_share
# psc_table['Government Share'] = psc._gas_government_share
# psc_table['DMO Volume'] = psc._gas_dmo_volume
# psc_table['DMO Fee'] = psc._gas_dmo_fee
# psc_table['DDMO'] = psc._gas_ddmo
psc_table['Taxable Income'] = psc._gas_taxable_income
psc_table['Tax'] = psc._gas_ftp_tax_payment
# psc_table['Contractor Take'] = psc._gas_contractor_take
# psc_table['Government Take'] = psc._gas_government_take
# psc_table['Cashflow'] = psc._gas_cashflow
print(psc_table, '\n')
#



df_comparison = pd.DataFrame()
base = np.array(load_testing(dataset_type='case1', key='gas_dmo_fee'))
engine = psc._gas_dmo_fee
df_comparison['Year'] = psc.project_years
df_comparison['Base'] = base
df_comparison['Engine'] = engine
df_comparison['Diff'] = base - engine
print(df_comparison)



