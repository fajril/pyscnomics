import pandas as pd


from pyscnomics.dataset.sample_refactored import load_dataset
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




print(psc_table, '\n')




