"""
CR MS DN 1
"""
import numpy as np
import pandas as pd
import time

from pyscnomics.dataset.sample import load_data, load_testing

from pyscnomics.tools.summary import get_summary
from pyscnomics.econ.selection import NPVSelection, DiscountingMode, FTPTaxRegime
from pyscnomics.econ.costs import Tangible, Intangible

from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.econ.costs import Tangible, Intangible, OPEX, ASR
from pyscnomics.econ.revenue import Lifting

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
psc.oil_ctr_pretax_share = 0.3472220
psc.gas_ctr_pretax_share = 0.5208330

# DMO
psc.oil_dmo_volume_portion = 0.25
psc.oil_dmo_fee_portion = 0.25
psc.oil_dmo_holiday_duration = 60
psc.gas_dmo_volume_portion = 0.25
psc.gas_dmo_fee_portion = 1
psc.gas_dmo_holiday_duration = 60

# Tax Rate
tax_rate = 0.424

# Defining Factor
mul_factor_capex = 1
mul_factor_lifting = 0.44

# Adjusting Tangible
new_tangible_list = []
for tan in psc.tangible_cost:
    new_tan = Tangible(start_year=tan.start_year,
                       end_year=tan.end_year,
                       cost=tan.cost * mul_factor_capex,
                       expense_year=tan.expense_year,
                       cost_allocation=tan.cost_allocation,
                       description=tan.description,
                       vat_portion=tan.vat_portion,
                       vat_discount=tan.vat_discount,
                       lbt_portion=tan.lbt_portion,
                       lbt_discount=tan.lbt_discount,
                       pis_year=tan.pis_year,
                       salvage_value=tan.salvage_value,
                       useful_life=tan.useful_life,
                       depreciation_factor=tan.depreciation_factor,
                       is_ic_applied=tan.is_ic_applied, )

    new_tangible_list.append(new_tan)

# Adjusting Intangible
new_intangible_list = []
for intan in psc.intangible_cost:
    new_intan = Intangible(start_year=intan.start_year,
                           end_year=intan.end_year,
                           cost=intan.cost * mul_factor_capex,
                           expense_year=intan.expense_year,
                           cost_allocation=intan.cost_allocation,
                           description=intan.description,
                           vat_portion=intan.vat_portion,
                           vat_discount=intan.vat_discount,
                           lbt_portion=intan.lbt_portion,
                           lbt_discount=intan.lbt_discount, )

    new_intangible_list.append(new_intan)

# Get the OPEX
new_opex_list = []
for opx in psc.opex:
    opex = OPEX(start_year=opx.start_year,
                end_year=opx.end_year,
                expense_year=opx.expense_year,
                cost_allocation=opx.cost_allocation,
                description=opx.description,
                vat_portion=opx.vat_portion,
                vat_discount=opx.vat_discount,
                lbt_portion=opx.lbt_portion,
                lbt_discount=opx.lbt_discount,
                fixed_cost=opx.fixed_cost,
                prod_rate=opx.prod_rate,
                cost_per_volume=opx.cost_per_volume, )

    new_opex_list.append(opex)

# Get the asr
new_asr_list = []
for i in psc.asr_cost:
    comp = ASR(start_year=i.start_year,
               end_year=i.end_year,
               cost=i.cost,
               expense_year=i.expense_year,
               cost_allocation=i.cost_allocation,
               description=i.description,
               vat_portion=i.vat_portion,
               vat_discount=i.vat_discount,
               lbt_portion=i.lbt_portion,
               lbt_discount=i.lbt_discount, )

    new_asr_list.append(comp)

# Get the lifting
new_lifting_list = []
for lft in psc.lifting:
    lift = Lifting(start_year=lft.start_year,
                   end_year=lft.end_year,
                   lifting_rate=lft.lifting_rate *  mul_factor_lifting,
                   price=lft.price,
                   prod_year=lft.prod_year,
                   fluid_type=lft.fluid_type,
                   ghv=lft.ghv,
                   prod_rate=lft.prod_rate)

    new_lifting_list.append(lift)

# Retrieving the original data
psc_new = CostRecovery(start_date=psc.start_date,
                       end_date=psc.end_date,
                       lifting=tuple(new_lifting_list),
                       tangible_cost=tuple(new_tangible_list),
                       intangible_cost=tuple(new_intangible_list),
                       opex=tuple(new_opex_list),
                       asr_cost=tuple(new_asr_list),
                       oil_ftp_is_available=psc.oil_ftp_is_available,
                       oil_ftp_is_shared=psc.oil_ftp_is_shared,
                       oil_ftp_portion=psc.oil_ftp_portion,
                       gas_ftp_is_available=psc.gas_ftp_is_available,
                       gas_ftp_is_shared=psc.gas_ftp_is_shared,
                       gas_ftp_portion=psc.gas_ftp_portion,
                       oil_ctr_pretax_share=psc.oil_ctr_pretax_share,
                       gas_ctr_pretax_share=psc.gas_ctr_pretax_share,
                       oil_dmo_volume_portion=psc.oil_dmo_volume_portion,
                       oil_dmo_fee_portion=psc.oil_dmo_fee_portion,
                       oil_dmo_holiday_duration=psc.oil_dmo_holiday_duration,
                       gas_dmo_volume_portion=psc.gas_dmo_volume_portion,
                       gas_dmo_fee_portion=psc.gas_dmo_fee_portion,
                       gas_dmo_holiday_duration=psc.gas_dmo_holiday_duration, )

start_time = time.time()
psc_new.run(tax_rate=tax_rate,
            ftp_tax_regime=FTPTaxRegime.PRE_PDJP_20_2017,
            sunk_cost_reference_year=2021)
end_time = time.time()
print('Execution Time: ', end_time - start_time, '\n')

# base = np.asarray(load_testing(dataset_type='case1', key='Unrec Cost'))
# engine = psc._oil_unrecovered_before_transfer
# #
# print(base)
# print(engine)
# print(base-engine)
# input()

psc_table_oil = pd.DataFrame()
psc_table_oil['Year'] = psc_new.project_years
psc_table_oil['Lifting'] = psc_new._oil_lifting.get_lifting_rate_arr()
psc_table_oil['Price'] = psc_new._oil_lifting.get_price_arr()
psc_table_oil['Revenue'] = psc_new._oil_revenue
psc_table_oil['Depreciable'] = psc_new._oil_tangible_expenditures
psc_table_oil['Opex'] = psc_new._oil_opex_expenditures
psc_table_oil['ASR'] = psc_new._oil_asr_expenditures
psc_table_oil['Depreciation'] = psc_new._oil_depreciation
psc_table_oil['Non Capital'] = psc_new._oil_non_capital
psc_table_oil['FTP'] = psc_new._oil_ftp
psc_table_oil['FTP - CTR'] = psc_new._oil_ftp_ctr
psc_table_oil['FTP - GOV'] = psc_new._oil_ftp_gov
psc_table_oil['Investment Credit'] = psc_new._oil_ic_paid
psc_table_oil['Unrecovered Cost'] = psc_new._oil_unrecovered_before_transfer
psc_table_oil['Cost to Be Recovered'] = psc_new._oil_cost_to_be_recovered
psc_table_oil['Cost Recovery'] = psc_new._oil_cost_recovery
psc_table_oil['ETS Before Transfer'] = psc_new._oil_ets_before_transfer
psc_table_oil['Transfer to Oil'] = psc_new._transfer_to_oil
psc_table_oil['Transfer to Gas'] = psc_new._transfer_to_gas
psc_table_oil['Unrec after Transfer'] = psc_new._oil_unrecovered_after_transfer
psc_table_oil['Cost To Be Recovered After TF'] = psc_new._oil_cost_to_be_recovered_after_tf
psc_table_oil['Cost Recovery After TF'] = psc_new._oil_cost_recovery_after_tf
psc_table_oil['ETS After Transfer'] = psc_new._oil_ets_after_transfer
psc_table_oil['Contractor Share Prior Tax'] = psc_new._oil_contractor_share
psc_table_oil['Government Share'] = psc_new._oil_government_share
psc_table_oil['DMO Volume'] = psc_new._oil_dmo_volume
psc_table_oil['DMO Fee'] = psc_new._oil_dmo_fee
psc_table_oil['DDMO'] = psc_new._oil_ddmo
psc_table_oil['Taxable Income'] = psc_new._oil_taxable_income
psc_table_oil['Tax'] = psc_new._oil_tax_payment
psc_table_oil['Contractor Share'] = psc_new._oil_ctr_net_share
psc_table_oil['Contractor Take'] = psc_new._gas_ctr_net_share
psc_table_oil['Cashflow'] = psc_new._oil_cashflow
psc_table_oil['Government Take'] = psc_new._oil_government_take
psc_table_oil.loc['Column_Total'] = psc_table_oil.sum(numeric_only=True, axis=0)
print(psc_table_oil, '\n')
# psc_table_oil.to_excel("df_oil.xlsx")

psc_table_gas = pd.DataFrame()
psc_table_gas['Year'] = psc_new.project_years
psc_table_gas['Lifting'] = psc_new._gas_lifting.get_lifting_rate_arr()
psc_table_gas['Price'] = psc_new._gas_wap_price
psc_table_gas['Revenue'] = psc_new._gas_revenue
psc_table_gas['Depreciable'] = psc_new._gas_tangible_expenditures
psc_table_gas['Opex'] = psc_new._gas_opex_expenditures
psc_table_gas['ASR'] = psc_new._gas_asr_expenditures
psc_table_gas['Depreciation'] = psc_new._gas_depreciation
psc_table_gas['Non Capital'] = psc_new._gas_non_capital
psc_table_gas['FTP'] = psc_new._gas_ftp
psc_table_gas['FTP - CTR'] = psc_new._gas_ftp_ctr
psc_table_gas['FTP - GOV'] = psc_new._gas_ftp_gov
psc_table_gas['Investment Credit'] = psc_new._gas_ic_paid
psc_table_gas['Unrecovered Cost'] = psc_new._gas_unrecovered_before_transfer
psc_table_gas['Cost to Be Recovered'] = psc_new._gas_cost_to_be_recovered
psc_table_gas['Cost Recovery'] = psc_new._gas_cost_recovery
psc_table_gas['ETS Before Transfer'] = psc_new._gas_ets_before_transfer
psc_table_gas['Transfer to Oil'] = psc_new._transfer_to_oil
psc_table_gas['Transfer to Gas'] = psc_new._transfer_to_gas
psc_table_gas['Unrec after Transfer'] = psc_new._gas_unrecovered_after_transfer
psc_table_gas['Cost To Be Recovered After TF'] = psc_new._gas_cost_to_be_recovered_after_tf
psc_table_gas['Cost Recovery After TF'] = psc_new._gas_cost_recovery_after_tf
psc_table_gas['ETS After Transfer'] = psc_new._gas_ets_after_transfer
psc_table_gas['Contractor Share Prior Tax'] = psc_new._gas_contractor_share
psc_table_gas['Government Share'] = psc_new._gas_government_share
psc_table_gas['DMO Volume'] = psc_new._gas_dmo_volume
psc_table_gas['DMO Fee'] = psc_new._gas_dmo_fee
psc_table_gas['DDMO'] = psc_new._gas_ddmo
psc_table_gas['Taxable Income'] = psc_new._gas_taxable_income
psc_table_gas['Tax'] = psc_new._gas_tax_payment
psc_table_gas['Contractor Share'] = psc_new._gas_ctr_net_share
psc_table_gas['Contractor Take'] = psc_new._gas_ctr_net_share
psc_table_gas['Cashflow'] = psc_new._gas_cashflow
psc_table_gas['Government Take'] = psc_new._gas_government_take
psc_table_gas.loc['Column_Total'] = psc_table_gas.sum(numeric_only=True, axis=0)
print(psc_table_gas, '\n')
# psc_table_gas.to_excel("df_gas.xlsx")

psc_table_consolidated = pd.DataFrame()
psc_table_consolidated['Year'] = psc_new.project_years
psc_table_consolidated['cnsltd_revenue'] = psc_new._consolidated_revenue
psc_table_consolidated['cnsltd_non_capital'] = psc_new._consolidated_non_capital
psc_table_consolidated['cnsltd_ic'] = psc_new._consolidated_ic
psc_table_consolidated['cnsltd_ic_unrecovered'] = psc_new._consolidated_ic_unrecovered
psc_table_consolidated['cnsltd_ic_paid'] = psc_new._consolidated_ic_paid
psc_table_consolidated['cnsltd_unrecovered_before_transfer'] = psc_new._consolidated_unrecovered_before_transfer
psc_table_consolidated['cnsltd_cost_recovery'] = psc_new._consolidated_cost_recovery_before_transfer
psc_table_consolidated['cnsltd_ets_before_transfer'] = psc_new._consolidated_ets_before_transfer
psc_table_consolidated['cnsltd_unrecovered_after_transfer'] = psc_new._consolidated_unrecovered_after_transfer
psc_table_consolidated['cnsltd_cost_to_be_recovered_after_tf'] = psc_new._consolidated_cost_to_be_recovered_after_tf
psc_table_consolidated['cnsltd_cost_recovery_after_tf'] = psc_new._consolidated_cost_recovery_after_tf
psc_table_consolidated['cnsltd_ets_after_transfer'] = psc_new._consolidated_ets_after_transfer
psc_table_consolidated['cnsltd_contractor_share'] = psc_new._consolidated_contractor_share
psc_table_consolidated['cnsltd_government_share'] = psc_new._consolidated_government_share
psc_table_consolidated['cnsltd_dmo_volume'] = psc_new._consolidated_dmo_volume
psc_table_consolidated['cnsltd_dmo_fee'] = psc_new._consolidated_dmo_fee
psc_table_consolidated['cnsltd_ddmo'] = psc_new._consolidated_ddmo
psc_table_consolidated['cnsltd_taxable_income'] = psc_new._consolidated_taxable_income
psc_table_consolidated['cnsltd_tax_due'] = psc_new._consolidated_tax_due
psc_table_consolidated['cnsltd_unpaid_tax_balance'] = psc_new._consolidated_unpaid_tax_balance
psc_table_consolidated['cnsltd_tax_payment'] = psc_new._consolidated_tax_payment
psc_table_consolidated['cnsltd_ctr_net_share'] = psc_new._consolidated_ctr_net_share
psc_table_consolidated['cnsltd_contractor_take'] = psc_new._consolidated_contractor_take
psc_table_consolidated['cnsltd_government_take'] = psc_new._consolidated_government_take
psc_table_consolidated['cnsltd_cashflow'] = psc_new._consolidated_cashflow
psc_table_consolidated['cum. cnsltd_cashflow'] = np.cumsum(psc_new._consolidated_cashflow)
psc_table_consolidated.loc['Column_Total'] = psc_table_consolidated.sum(numeric_only=True, axis=0)
print(psc_table_consolidated, '\n')

# df_comparison = pd.DataFrame()
# # Calculated result
# engine = psc._consolidated_cashflow
#
# # Expected result
# base = load_testing(dataset_type='case1', key='cnsltd_cashflow')
#
# df_comparison['Year'] = psc.project_years
# df_comparison['Base'] = base
# df_comparison['Engine'] = engine
# df_comparison['Diff'] = base - engine
# print(df_comparison, '\n')
#
# gross_revenue_calc = psc._oil_contractor_take + psc._oil_government_take
# gross_revenue_engine = psc._oil_revenue + psc._gas_revenue
#
# print(gross_revenue_calc)
# print(gross_revenue_engine)

# psc.get_summary()
# print('Lifting Oil/Condensate', psc._summary.lifting_oil)
# print('Lifting Gas', psc._summary.lifting_gas)
# print('Oil WAP', psc._summary.oil_wap)
# print('Gas WAP', psc._summary.gas_wap)
# print('Gross Revenue', psc._summary.gross_revenue)
# print('CTR Gross Share', psc._summary.ctr_gross_share)
# print('GOV Gross Share', psc._summary.gov_gross_share)
# print('Sunk Cost', psc._summary.sunk_cost)
# print('Investment', psc._summary.investment)
# print('Tangible', psc._summary.tangible)
# print('Intangible', psc._summary.intangible)
# print('OPEX and ASR', psc._summary.opex_and_asr)
# print('OPEX', psc._summary.opex)
# print('ASR', psc._summary.asr)
# print('Cost Recovery', psc._summary.cost_recovery_or_deductible_cost)
# print('(%) Gross Rev', psc._summary.gross_rev_cost_recovery_or_deductible_cost)
# print('Unrec. Cost', psc._summary.unrec_or_carry_forward_deductible_cost)
# print('Unrec. Over Cost Recovery', psc._summary.unrec_over_cost_recovery)
# print('CTR Net Share)', psc._summary.ctr_net_share)
# print('(%) CTR Net Share over Gross Rev.))', psc._summary.ctr_net_share_over_gross_share)
# print('POT', psc._summary.ctr_pot)

# psc._get_sunk_cost(discount_rate_year=2021)
# print(psc._oil_sunk_cost)
# print(psc._gas_sunk_cost)
# print(np.sum(psc._oil_sunk_cost))
# print(np.sum(psc._gas_sunk_cost))

psc_summary = get_summary(contract=psc_new,
                          reference_year=2022,
                          inflation_rate=0.1,
                          discount_rate=0.1,
                          npv_mode=NPVSelection.NPV_SKK_NOMINAL_TERMS,
                          discounting_mode=DiscountingMode.END_YEAR)

for key, value in psc_summary.items():
    print(key, ":", value)

# print('')
# print('Oil')
# print(psc_new.lifting[0].fluid_type)
# print(psc_new.lifting[0].lifting_rate, '\n')
#
# print('GSA 1')
# print(psc_new.lifting[1].fluid_type)
# print(psc_new.lifting[1].lifting_rate, '\n')
#
# print('GSA 2')
# print(psc_new.lifting[2].fluid_type)
# print(psc_new.lifting[2].lifting_rate, '\n')
