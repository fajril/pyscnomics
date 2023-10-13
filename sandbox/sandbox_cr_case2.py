import pandas as pd


from pyscnomics.dataset.sample import load_data
# pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# Initiating the psc object
psc = load_data(dataset_type='case2', contract_type='cost_recovery')

# Editing the CostRecovery attribute as the corresponding case 1
# FTP
psc.oil_ftp_is_available = True
psc.oil_ftp_is_shared = True
psc.oil_ftp_portion = 0.2
psc.gas_ftp_is_available = True
psc.gas_ftp_is_shared = True
psc.gas_ftp_portion = 0.05

# Split Pre Tax
psc.oil_ctr_pretax_share = 0.34722220
psc.gas_ctr_pretax_share = 0.72083300

# DMO
psc.oil_dmo_volume_portion = 0.25
psc.oil_dmo_fee_portion = 0.25
psc.oil_dmo_holiday_duration = 60
psc.gas_dmo_volume_portion = 0.25
psc.gas_dmo_fee_portion = 0.25
psc.gas_dmo_holiday_duration = 60


psc.run()


psc_table = pd.DataFrame()
psc_table['Revenue'] = psc._gas_revenue
psc_table['Depreciation'] = psc._gas_depreciation
psc_table['Non Capital'] = psc._gas_non_capital
psc_table['FTP - CTR'] = psc._gas_ftp_ctr
psc_table['FTP - GOV'] = psc._gas_ftp_gov
psc_table['Investment Credit'] = psc._gas_ic_paid
psc_table['Unrecovered Cost'] = psc._gas_unrecovered_before_transfer
psc_table['Cost to Be Recovered'] = psc._gas_cost_to_be_recovered
psc_table['Cost Recovery'] = psc._gas_cost_recovery
psc_table['ETS Before Transfer'] = psc._gas_ets_before_transfer
psc_table['Transfer'] = psc._transfer_to_gas
psc_table['ETS After Transfer'] = psc._gas_ets_before_transfer
psc_table['Contractor Share'] = psc._gas_contractor_share
psc_table['Government Share'] = psc._gas_government_share
psc_table['DMO Volume'] = psc._gas_dmo_volume
psc_table['DMO Fee'] = psc._gas_dmo_fee
psc_table['DDMO'] = psc._gas_ddmo
psc_table['Taxable Income'] = psc._gas_taxable_income
psc_table['Contractor Take'] = psc._gas_contractor_take
psc_table['Government Take'] = psc._gas_government_take
psc_table['Cashflow'] = psc._gas_cashflow
#
psc_table.to_excel('Cost Recovery Case 2 Table from Pyscnomics.xlsx')


