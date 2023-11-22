import numpy as np
import pandas as pd
import time


from pyscnomics.dataset.sample import load_data, load_testing
from pyscnomics.tools.summary import get_summary
from pyscnomics.econ.selection import NPVSelection, DiscountingMode
# pd.options.display.float_format = '{:,.2f}'.format

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

psc = load_data(dataset_type='small_oil', contract_type='gross_split')
tax_rate = np.array([0.11, 0.11, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
inflation_rate = np.array([0, 0.0, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02])
start_time = time.time()
psc.run(tax_rate=0.22, vat_rate=tax_rate, inflation_rate=inflation_rate)
end_time = time.time()
print('Execution Time: ', end_time - start_time, '\n')


psc_table = pd.DataFrame()
psc_table['Years'] = psc.project_years
psc_table['Lifting'] = psc._oil_lifting.lifting_rate
psc_table['Revenue'] = psc._oil_revenue
psc_table['Base Split'] = psc._oil_base_split
psc_table['Variable Split'] = psc._var_split_array
psc_table['Progressive Split'] = psc._oil_prog_split
psc_table['Contractor Split Split'] = psc._oil_ctr_split
psc_table['Government Share'] = psc._oil_gov_share
psc_table['Contractor Share'] = psc._oil_ctr_share_before_transfer
psc_table['Tangible'] = psc._oil_tangible_expenditures
psc_table['OPEX'] = psc._oil_opex_expenditures
psc_table['Depreciation'] = psc._oil_depreciation
psc_table['Non Capital'] = psc._oil_non_capital
psc_table['Total Expenses'] = psc._oil_total_expenses
psc_table['Cost To Be Deducted'] = psc._oil_cost_tobe_deducted
psc_table['Carry Forward Cost'] = psc._oil_carward_deduct_cost
psc_table['Deductible Cost'] = psc._oil_deductible_cost
psc_table['Transfer To Oil'] = psc._transfer_to_oil
psc_table['Transfer To Gas'] = psc._transfer_to_gas
psc_table['Carry Forward Cost after TF'] = psc._oil_carward_cost_aftertf
psc_table['CTR Share After TF'] = psc._oil_ctr_share_after_transfer
psc_table['CTR Net Operating Profit'] = psc._oil_net_operating_profit
psc_table['DMO Volume'] = psc._oil_dmo_volume
psc_table['DMO Fee'] = psc._oil_dmo_fee
psc_table['DDMO'] = psc._oil_ddmo
psc_table['Taxable Income'] = psc._oil_taxable_income
psc_table['Tax'] = psc._oil_tax
psc_table['Net CTR Share'] = psc._oil_ctr_net_share
psc_table['CTR CashFlow'] = psc._oil_ctr_cashflow
psc_table['Government Take'] = psc._oil_government_take
psc_table['Cumulative CTR CashFlow'] = np.cumsum(psc._oil_ctr_cashflow)
psc_table.loc['Column_Total'] = psc_table.sum(numeric_only=True, axis=0)


print('Calculation Table: Gross Split \n', psc_table, '\n')


psc_summary = get_summary(contract=psc,
                          reference_year=2023,
                          inflation_rate=0.1,
                          discount_rate=0.1,
                          npv_mode=NPVSelection.NPV_SKK_NOMINAL_TERMS,
                          discounting_mode=DiscountingMode.END_YEAR)

for key, value in psc_summary.items():
    print(key, ":", value)


