import pandas as pd


from pyscnomics.dataset.sample import load_dataset
# pd.options.display.float_format = '{:,.2f}'.format

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

psc = load_dataset(dataset_type='medium_oil_2', contract_type='gross_split')
psc.run()

psc_table = pd.DataFrame()
psc_table['Years'] = psc.project_years
psc_table['Base Split'] = psc._oil_base_split
psc_table['Variable Split'] = psc._var_split_array
psc_table['Progressive Split'] = psc._oil_prog_split
psc_table['Contractor Split Split'] = psc._oil_ctr_split
psc_table['Government Share'] = psc._oil_gov_share
psc_table['Contractor Share'] = psc._oil_ctr_share
psc_table['Total Expenses'] = psc._oil_total_expenses
psc_table['Cost To Be Deducted'] = psc._oil_cost_tobe_deducted
psc_table['Carry Forward Cost'] = psc._oil_carward_deduct_cost
psc_table['Deductible Cost'] = psc._oil_deductible_cost
psc_table['CTR Net Operating Profit'] = psc._oil_net_operating_profit
psc_table['DMO Volume'] = psc._oil_dmo_volume
psc_table['DMO Fee'] = psc._oil_dmo_fee
psc_table['DDMO'] = psc._oil_ddmo
psc_table['Taxable Income'] = psc._oil_taxable_income
psc_table['Net CTR Share'] = psc._oil_ctr_net_share
psc_table['CTR CashFlow'] = psc._oil_ctr_cashflow


print(psc_table, '\n')




