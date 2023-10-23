import numpy as np
import pandas as pd


from pyscnomics.dataset.sample import load_data
# pd.options.display.float_format = '{:,.2f}'.format

'''
B
'''

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

psc = load_data(dataset_type='small_gas', contract_type='cost_recovery')
psc.run()

psc_table = pd.DataFrame()
psc_table['Years'] = psc.project_years
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
psc_table['Tax'] = psc._gas_tax_payment
psc_table['Contractor Take'] = psc._gas_contractor_take
psc_table['Government Take'] = psc._gas_government_take
psc_table['Cashflow'] = psc._gas_cashflow
psc_table['Cum Cashflow'] = np.cumsum(psc._gas_cashflow)

psc_table.drop([i for i in range(12)], inplace=True)
print(psc_table, '\n')

psc_table.to_excel('Cost Recovery B Table from Pyscnomics.xlsx')






