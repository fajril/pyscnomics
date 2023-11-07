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

from pyscnomics.contracts import transition

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

print('------------------------------------------ Cost Recovery ------------------------------------------')
start_time = time.time()
ftp_tax_regime = FTPTaxRegime.PRE_2017
eff_tax_rate = 0.48
argument_contract1 = {'ftp_tax_regime': FTPTaxRegime.PRE_2017,
                      'tax_rate': eff_tax_rate}
psc1.run(**argument_contract1)
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
psc1_table_oil['FTP-CTR'] = psc1._oil_ftp_ctr
psc1_table_oil['FTP-GOV'] = psc1._oil_ftp_gov
psc1_table_oil['InvestmentCredit'] = psc1._oil_ic_paid
psc1_table_oil['UnrecoveredCost'] = psc1._oil_unrecovered_before_transfer
psc1_table_oil['CosttoBeRecovered'] = psc1._oil_cost_to_be_recovered
psc1_table_oil['CostRecovery'] = psc1._oil_cost_recovery
psc1_table_oil['ETSBeforeTransfer'] = psc1._oil_ets_before_transfer
psc1_table_oil['TransfertoOil'] = psc1._transfer_to_oil
psc1_table_oil['TransfertoGas'] = psc1._transfer_to_gas
psc1_table_oil['UnrecafterTransfer'] = psc1._oil_unrecovered_after_transfer
psc1_table_oil['CostToBeRecoveredAfterTF'] = psc1._oil_cost_to_be_recovered_after_tf
psc1_table_oil['CostRecoveryAfterTF'] = psc1._oil_cost_recovery_after_tf
psc1_table_oil['ETSAfterTransfer'] = psc1._oil_ets_after_transfer
psc1_table_oil['ContractorShare'] = psc1._oil_contractor_share
psc1_table_oil['GovernmentShare'] = psc1._oil_government_share
psc1_table_oil['DMOVolume'] = psc1._oil_dmo_volume
psc1_table_oil['DMOFee'] = psc1._oil_dmo_fee
psc1_table_oil['DDMO'] = psc1._oil_ddmo
psc1_table_oil['TaxableIncome'] = psc1._oil_taxable_income
psc1_table_oil['Tax'] = psc1._oil_tax_payment
psc1_table_oil['NetContractorShare'] = psc1._oil_ctr_net_share
psc1_table_oil['ContractorTake'] = psc1._gas_ctr_net_share
psc1_table_oil['Cashflow'] = psc1._oil_cashflow
psc1_table_oil['GovernmentTake'] = psc1._oil_government_take
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
psc1_table_gas['Non'] = psc1._gas_non_capital
psc1_table_gas['FTP'] = psc1._gas_ftp
psc1_table_gas['FTP'] = psc1._gas_ftp_ctr
psc1_table_gas['FTP'] = psc1._gas_ftp_gov
psc1_table_gas['InvestmentCredit'] = psc1._gas_ic_paid
psc1_table_gas['UnrecoveredCost'] = psc1._gas_unrecovered_before_transfer
psc1_table_gas['CosttoBeRecovered'] = psc1._gas_cost_to_be_recovered
psc1_table_gas['CostRecovery'] = psc1._gas_cost_recovery
psc1_table_gas['ETSBeforeTransfer'] = psc1._gas_ets_before_transfer
psc1_table_gas['TransfertoOil'] = psc1._transfer_to_oil
psc1_table_gas['TransfertoGas'] = psc1._transfer_to_gas
psc1_table_gas['UnrecafterTransfer'] = psc1._gas_unrecovered_after_transfer
psc1_table_gas['CostToBeRecoveredAfterTF'] = psc1._gas_cost_to_be_recovered_after_tf
psc1_table_gas['CostRecoveryAfterTF'] = psc1._gas_cost_recovery_after_tf
psc1_table_gas['ETSAfterTransfer'] = psc1._gas_ets_after_transfer
psc1_table_gas['ContractorShare'] = psc1._gas_contractor_share
psc1_table_gas['GovernmentShare'] = psc1._gas_government_share
psc1_table_gas['DMOVolume'] = psc1._gas_dmo_volume
psc1_table_gas['DMOFee'] = psc1._gas_dmo_fee
psc1_table_gas['DDMO'] = psc1._gas_ddmo
psc1_table_gas['TaxableIncome'] = psc1._gas_taxable_income
psc1_table_gas['Tax'] = psc1._gas_tax_payment
psc1_table_gas['NetContractorShare'] = psc1._gas_ctr_net_share
psc1_table_gas['ContractorTake'] = psc1._gas_ctr_net_share
psc1_table_gas['Cashflow'] = psc1._gas_cashflow
psc1_table_gas['GovernmentTake'] = psc1._gas_government_take
psc1_table_gas.loc['Column_Total'] = psc1_table_gas.sum(numeric_only=True, axis=0)
print(psc1_table_gas, '\n')

psc_table_consolidated = pd.DataFrame()
psc_table_consolidated['Year'] = psc1.project_years
psc_table_consolidated['cnsltd_revenue'] = psc1._consolidated_revenue
psc_table_consolidated['cnsltd_non_capital'] = psc1._consolidated_non_capital
psc_table_consolidated['cnsltd_ic'] = psc1._consolidated_ic
psc_table_consolidated['cnsltd_ic_unrecovered'] = psc1._consolidated_ic_unrecovered
psc_table_consolidated['cnsltd_ic_paid'] = psc1._consolidated_ic_paid
psc_table_consolidated['cnsltd_unrecovered_before_transfer'] = psc1._consolidated_unrecovered_before_transfer
psc_table_consolidated['cnsltd_cost_recovery'] = psc1._consolidated_cost_recovery_before_transfer
psc_table_consolidated['cnsltd_ets_before_transfer'] = psc1._consolidated_ets_before_transfer
psc_table_consolidated['cnsltd_unrecovered_after_transfer'] = psc1._consolidated_unrecovered_after_transfer
psc_table_consolidated['cnsltd_cost_to_be_recovered_after_tf'] = psc1._consolidated_cost_to_be_recovered_after_tf
psc_table_consolidated['cnsltd_cost_recovery_after_tf'] = psc1._consolidated_cost_recovery_after_tf
psc_table_consolidated['cnsltd_ets_after_transfer'] = psc1._consolidated_ets_after_transfer
psc_table_consolidated['cnsltd_contractor_share'] = psc1._consolidated_contractor_share
psc_table_consolidated['cnsltd_government_share'] = psc1._consolidated_government_share
psc_table_consolidated['cnsltd_dmo_volume'] = psc1._consolidated_dmo_volume
psc_table_consolidated['cnsltd_dmo_fee'] = psc1._consolidated_dmo_fee
psc_table_consolidated['cnsltd_ddmo'] = psc1._consolidated_ddmo
psc_table_consolidated['cnsltd_taxable_income'] = psc1._consolidated_taxable_income
psc_table_consolidated['cnsltd_tax_due'] = psc1._consolidated_tax_due
psc_table_consolidated['cnsltd_unpaid_tax_balance'] = psc1._consolidated_unpaid_tax_balance
psc_table_consolidated['cnsltd_tax_payment'] = psc1._consolidated_tax_payment
psc_table_consolidated['cnsltd_ctr_net_share'] = psc1._consolidated_ctr_net_share
psc_table_consolidated['cnsltd_contractor_take'] = psc1._consolidated_contractor_take
psc_table_consolidated['cnsltd_government_take'] = psc1._consolidated_government_take
psc_table_consolidated['cnsltd_cashflow'] = psc1._consolidated_cashflow
psc_table_consolidated['cum. cnsltd_cashflow'] = np.cumsum(psc1._consolidated_cashflow)
psc_table_consolidated.loc['Column_Total'] = psc_table_consolidated.sum(numeric_only=True, axis=0)
print(psc_table_consolidated, '\n')

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
    fixed_cost=np.array(
        [0.63640338233533300, 1.06286476325927000, 1.09105051487026000, 1.11733210409363000, 1.14029666892703000,
         1.29608012023605000, 1.12478850166335000, 0.78684662704513200]),
    expense_year=np.array([2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027]),
    cost_allocation=FluidType.GAS)

# Defining VAT and Import Duty
psc2_gas_vat = OPEX(
    start_year=2020,
    end_year=2027,
    fixed_cost=np.array(
        [0.3965062479576550, 0.2867125283653920, 0.2947836073052500, 0.3019666899432990, 0.3074269319547460,
         0.5538846425099840, 0.3772323444857470, 0.3331977740376320, ]),
    expense_year=np.array([2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, ]),
    cost_allocation=FluidType.GAS)

psc2_gas_import_duty = OPEX(
    start_year=2020,
    end_year=2027,
    fixed_cost=np.array(
        [0.159965353, 0.055922218, 0.057496452, 0.058897486, 0.059962486, 0.227566547, 0.073577773, 0.064988993]),
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
    base_split_ctr_oil=0.43,
    base_split_ctr_gas=0.48,
    split_ministry_disc=0.0,
    oil_dmo_volume_portion=0.25,
    oil_dmo_fee_portion=1.0,
    oil_dmo_holiday_duration=0,
    gas_dmo_volume_portion=0.25,
    gas_dmo_fee_portion=1.0,
    gas_dmo_holiday_duration=0)

print('------------------------------------------ Gross Split ------------------------------------------')
start_time = time.time()
argument_contract2 = {'tax_rate':0.40}
psc2.run(**argument_contract2)
end_time = time.time()
print('Execution Time: ', end_time - start_time, '\n')

psc2_table_oil = pd.DataFrame()
psc2_table_oil['Years'] = psc2.project_years
psc2_table_oil['Lifting'] = psc2._oil_lifting.lifting_rate
psc2_table_oil['Price'] = psc2._oil_lifting.lifting_price_arr()
psc2_table_oil['Revenue'] = psc2._oil_revenue
psc2_table_oil['BaseSplit'] = psc2._oil_base_split
psc2_table_oil['VariableSplit'] = psc2._var_split_array
psc2_table_oil['ProgressiveSplit'] = psc2._oil_prog_split
psc2_table_oil['ContractorSplit'] = psc2._oil_ctr_split
psc2_table_oil['GovernmentShare'] = psc2._oil_gov_share
psc2_table_oil['ContractorShare'] = psc2._oil_ctr_share_before_transfer
psc2_table_oil['Depreciation'] = psc2._oil_depreciation
psc2_table_oil['Opex'] = psc2._oil_opex.expenditures()
psc2_table_oil['ASR'] = psc2._oil_asr.expenditures()
psc2_table_oil['NonCapital'] = psc2._oil_non_capital
psc2_table_oil['TotalExpenses'] = psc2._oil_total_expenses
psc2_table_oil['CostToBeDeducted'] = psc2._oil_cost_tobe_deducted
psc2_table_oil['CarryForwardCost'] = psc2._oil_carward_deduct_cost
psc2_table_oil['DeductibleCost'] = psc2._oil_deductible_cost
psc2_table_oil['TransferToOil'] = psc2._transfer_to_oil
psc2_table_oil['TransferToGas'] = psc2._transfer_to_gas
psc2_table_oil['CarryForwardCostafterTF'] = psc2._oil_carward_cost_aftertf
psc2_table_oil['CTRShareAfterTF'] = psc2._oil_ctr_share_after_transfer
psc2_table_oil['CTRNetOperatingProfit'] = psc2._oil_net_operating_profit
psc2_table_oil['DMOVolume'] = psc2._oil_dmo_volume
psc2_table_oil['DMOFee'] = psc2._oil_dmo_fee
psc2_table_oil['DDMO'] = psc2._oil_ddmo
psc2_table_oil['TaxableIncome'] = psc2._oil_taxable_income
psc2_table_oil['Tax'] = psc2._oil_tax
psc2_table_oil['NetCTRShare'] = psc2._oil_ctr_net_share
psc2_table_oil['CTRCashFlow'] = psc2._oil_ctr_cashflow
psc2_table_oil['GovernmentTake'] = psc2._oil_government_take
psc2_table_oil.loc['Column_Total'] = psc2_table_oil.sum(numeric_only=True, axis=0)
print(psc2_table_oil, '\n')

psc2_table_gas = pd.DataFrame()
psc2_table_gas['Years'] = psc2.project_years
psc2_table_gas['Lifting'] = psc2._gas_lifting.lifting_rate
psc2_table_gas['Price'] = psc2._gas_lifting.lifting_price_arr()
psc2_table_gas['Revenue'] = psc2._gas_revenue
psc2_table_gas['BaseSplit'] = psc2._gas_base_split
psc2_table_gas['VariableSplit'] = psc2._var_split_array
psc2_table_gas['ProgressiveSplit'] = psc2._gas_prog_split
psc2_table_gas['ContractorSplit'] = psc2._gas_ctr_split
psc2_table_gas['GovernmentShare'] = psc2._gas_gov_share
psc2_table_gas['ContractorShare'] = psc2._gas_ctr_share_before_transfer
psc2_table_gas['Depreciation'] = psc2._gas_depreciation
psc2_table_gas['Opex'] = psc2._gas_opex.expenditures()
psc2_table_gas['ASR'] = psc2._gas_asr.expenditures()
psc2_table_gas['Non Capital'] = psc2._gas_non_capital
psc2_table_gas['TotalExpenses'] = psc2._gas_total_expenses
psc2_table_gas['CostToBeDeducted'] = psc2._gas_cost_tobe_deducted
psc2_table_gas['CarryForwardCost'] = psc2._gas_carward_deduct_cost
psc2_table_gas['DeductibleCost'] = psc2._gas_deductible_cost
psc2_table_gas['TransferToOil'] = psc2._transfer_to_oil
psc2_table_gas['TransferToGas'] = psc2._transfer_to_gas
psc2_table_gas['CarryForwardCostafterTF'] = psc2._gas_carward_cost_aftertf
psc2_table_gas['CTRShareAfterTF'] = psc2._gas_ctr_share_after_transfer
psc2_table_gas['CTRNetOperatingProfit'] = psc2._gas_net_operating_profit
psc2_table_gas['DMOVolume'] = psc2._gas_dmo_volume
psc2_table_gas['DMOFee'] = psc2._gas_dmo_fee
psc2_table_gas['DDMO'] = psc2._gas_ddmo
psc2_table_gas['TaxableIncome'] = psc2._gas_taxable_income
psc2_table_gas['Tax'] = psc2._gas_tax
psc2_table_gas['NetCTRShare'] = psc2._gas_ctr_net_share
psc2_table_gas['CTRCashFlow'] = psc2._gas_ctr_cashflow
psc2_table_gas['GovernmentTake'] = psc2._gas_government_take
psc2_table_gas.loc['Column_Total'] = psc2_table_gas.sum(numeric_only=True, axis=0)
print(psc2_table_gas, '\n')

print('------------------------------------------ Transition ------------------------------------------')
start_time = time.time()
trans1, trans2 = transition.transition(contract1=psc1, contract2=psc2, argument_contract1=argument_contract1, argument_contract2=argument_contract2)
end_time = time.time()
print('Execution Time: ', end_time - start_time, '\n')

print('Cost Recovery Transition')
trans1_table_oil = pd.DataFrame()
trans1_table_oil['Year'] = trans1.project_years
trans1_table_oil['Lifting'] = trans1._oil_lifting.lifting_rate_arr()
trans1_table_oil['Price'] = trans1._oil_lifting.lifting_price_arr()
trans1_table_oil['Revenue'] = trans1._oil_revenue
trans1_table_oil['Depreciable'] = trans1._oil_tangible.expenditures()
trans1_table_oil['Intangible'] = trans1._oil_intangible.expenditures()
trans1_table_oil['Opex'] = trans1._oil_opex.expenditures()
trans1_table_oil['ASR'] = trans1._oil_asr.expenditures()
trans1_table_oil['Depreciation'] = trans1._oil_depreciation
trans1_table_oil['NonCapital'] = trans1._oil_non_capital
trans1_table_oil['FTP'] = trans1._oil_ftp
trans1_table_oil['FTP_CTR'] = trans1._oil_ftp_ctr
trans1_table_oil['FTP_GOV'] = trans1._oil_ftp_gov
trans1_table_oil['InvestmentCredit'] = trans1._oil_ic_paid
trans1_table_oil['UnrecoveredCost'] = trans1._oil_unrecovered_before_transfer
trans1_table_oil['CosttoBeRecovered'] = trans1._oil_cost_to_be_recovered
trans1_table_oil['CostRecovery'] = trans1._oil_cost_recovery
trans1_table_oil['ETSBeforeTransfer'] = trans1._oil_ets_before_transfer
trans1_table_oil['TransfertoOil'] = trans1._transfer_to_oil
trans1_table_oil['TransfertoGas'] = trans1._transfer_to_gas
trans1_table_oil['UnrecafterTransfer'] = trans1._oil_unrecovered_after_transfer
trans1_table_oil['CostToBeRecoveredAfterTF'] = trans1._oil_cost_to_be_recovered_after_tf
trans1_table_oil['CostRecoveryAfterTF'] = trans1._oil_cost_recovery_after_tf
trans1_table_oil['ETSAfterTransfer'] = trans1._oil_ets_after_transfer
trans1_table_oil['ContractorShare'] = trans1._oil_contractor_share
trans1_table_oil['GovernmentShare'] = trans1._oil_government_share
trans1_table_oil['DMOVolume'] = trans1._oil_dmo_volume
trans1_table_oil['DMOFee'] = trans1._oil_dmo_fee
trans1_table_oil['DDMO'] = trans1._oil_ddmo
trans1_table_oil['TaxableIncome'] = trans1._oil_taxable_income
trans1_table_oil['Tax'] = trans1._oil_tax_payment
trans1_table_oil['NetContractorShare'] = trans1._oil_ctr_net_share
trans1_table_oil['ContractorTake'] = trans1._gas_ctr_net_share
trans1_table_oil['Cashflow'] = trans1._oil_cashflow
trans1_table_oil['GovernmentTake'] = trans1._oil_government_take
trans1_table_oil.loc['Column_Total'] = trans1_table_oil.sum(numeric_only=True, axis=0)
print(trans1_table_oil, '\n')

trans1_table_gas = pd.DataFrame()
trans1_table_gas['Year'] = trans1.project_years
trans1_table_gas['Lifting'] = trans1._gas_lifting.lifting_rate_arr()
trans1_table_gas['Price'] = trans1._gas_lifting.lifting_price_arr()
trans1_table_gas['Revenue'] = trans1._gas_revenue
trans1_table_gas['Depreciable'] = trans1._gas_tangible.expenditures()
trans1_table_gas['Intangible'] = trans1._gas_intangible.expenditures()
trans1_table_gas['Opex'] = trans1._gas_opex.expenditures()
trans1_table_gas['ASR'] = trans1._gas_asr.expenditures()
trans1_table_gas['Depreciation'] = trans1._gas_depreciation
trans1_table_gas['NonCapital'] = trans1._gas_non_capital
trans1_table_gas['FTP'] = trans1._gas_ftp
trans1_table_gas['FTP_CTR'] = trans1._gas_ftp_ctr
trans1_table_gas['FTP_GOV'] = trans1._gas_ftp_gov
trans1_table_gas['InvestmentCredit'] = trans1._gas_ic_paid
trans1_table_gas['UnrecoveredCost'] = trans1._gas_unrecovered_before_transfer
trans1_table_gas['CosttoBeRecovered'] = trans1._gas_cost_to_be_recovered
trans1_table_gas['CostRecovery'] = trans1._gas_cost_recovery
trans1_table_gas['ETSBeforeTransfer'] = trans1._gas_ets_before_transfer
trans1_table_gas['TransfertoOil'] = trans1._transfer_to_oil
trans1_table_gas['TransfertoGas'] = trans1._transfer_to_gas
trans1_table_gas['UnrecafterTransfer'] = trans1._gas_unrecovered_after_transfer
trans1_table_gas['CostToBeRecoveredAfterTF'] = trans1._gas_cost_to_be_recovered_after_tf
trans1_table_gas['CostRecoveryAfterTF'] = trans1._gas_cost_recovery_after_tf
trans1_table_gas['ETSAfterTransfer'] = trans1._gas_ets_after_transfer
trans1_table_gas['ContractorShare'] = trans1._gas_contractor_share
trans1_table_gas['GovernmentShare'] = trans1._gas_government_share
trans1_table_gas['DMOVolume'] = trans1._gas_dmo_volume
trans1_table_gas['DMOFee'] = trans1._gas_dmo_fee
trans1_table_gas['DDMO'] = trans1._gas_ddmo
trans1_table_gas['TaxableIncome'] = trans1._gas_taxable_income
trans1_table_gas['Tax'] = trans1._gas_tax_payment
trans1_table_gas['NetContractorShare'] = trans1._gas_ctr_net_share
trans1_table_gas['ContractorTake'] = trans1._gas_ctr_net_share
trans1_table_gas['Cashflow'] = trans1._gas_cashflow
trans1_table_gas['GovernmentTake'] = trans1._gas_government_take
trans1_table_gas.loc['Column_Total'] = trans1_table_gas.sum(numeric_only=True, axis=0)
print(trans1_table_gas, '\n')

print('Gross Split Transition')
trans2_table_oil = pd.DataFrame()
trans2_table_oil['Years'] = trans2.project_years
trans2_table_oil['Lifting'] = trans2._oil_lifting.lifting_rate
trans2_table_oil['Revenue'] = trans2._oil_revenue
trans2_table_oil['BaseSplit'] = trans2._oil_base_split
trans2_table_oil['VariableSplit'] = trans2._var_split_array
trans2_table_oil['ProgressiveSplit'] = trans2._oil_prog_split
trans2_table_oil['ContractorSplitSplit'] = trans2._oil_ctr_split
trans2_table_oil['GovernmentShare'] = trans2._oil_gov_share
trans2_table_oil['ContractorShare'] = trans2._oil_ctr_share_before_transfer
trans2_table_oil['Depreciation'] = trans2._oil_depreciation
trans2_table_oil['NonCapital'] = trans2._oil_non_capital
trans2_table_oil['TotalExpenses'] = trans2._oil_total_expenses
trans2_table_oil['CostToBeDeducted'] = trans2._oil_cost_tobe_deducted
trans2_table_oil['CarryForwardCost'] = trans2._oil_carward_deduct_cost
trans2_table_oil['DeductibleCost'] = trans2._oil_deductible_cost
trans2_table_oil['TransferToOil'] = trans2._transfer_to_oil
trans2_table_oil['TransferToGas'] = trans2._transfer_to_gas
trans2_table_oil['CarryForwardCostafterTF'] = trans2._oil_carward_cost_aftertf
trans2_table_oil['CTRShareAfterTF'] = trans2._oil_ctr_share_after_transfer
trans2_table_oil['CTRNetOperatingProfit'] = trans2._oil_net_operating_profit
trans2_table_oil['DMOVolume'] = trans2._oil_dmo_volume
trans2_table_oil['DMOFee'] = trans2._oil_dmo_fee
trans2_table_oil['DDMO'] = trans2._oil_ddmo
trans2_table_oil['TaxableIncome'] = trans2._oil_taxable_income
trans2_table_oil['Tax'] = trans2._oil_tax
trans2_table_oil['NetCTRShare'] = trans2._oil_ctr_net_share
trans2_table_oil['CTRCashFlow'] = trans2._oil_ctr_cashflow
trans2_table_oil['GovernmentTake'] = trans2._oil_government_take
trans2_table_oil.loc['Column_Total'] = trans2_table_oil.sum(numeric_only=True, axis=0)
print(trans2_table_oil, '\n')

trans2_table_gas = pd.DataFrame()
trans2_table_gas['Years'] = trans2.project_years
trans2_table_gas['Lifting'] = trans2._gas_lifting.lifting_rate
trans2_table_gas['Revenue'] = trans2._gas_revenue
trans2_table_gas['BaseSplit'] = trans2._gas_base_split
trans2_table_gas['VariableSplit'] = trans2._var_split_array
trans2_table_gas['ProgressiveSplit'] = trans2._gas_prog_split
trans2_table_gas['ContractorSplitSplit'] = trans2._gas_ctr_split
trans2_table_gas['GovernmentShare'] = trans2._gas_gov_share
trans2_table_gas['ContractorShare'] = trans2._gas_ctr_share_before_transfer
trans2_table_gas['Depreciation'] = trans2._gas_depreciation
trans2_table_gas['NonCapital'] = trans2._gas_non_capital
trans2_table_gas['TotalExpenses'] = trans2._gas_total_expenses
trans2_table_gas['CostToBeDeducted'] = trans2._gas_cost_tobe_deducted
trans2_table_gas['CarryForwardCost'] = trans2._gas_carward_deduct_cost
trans2_table_gas['DeductibleCost'] = trans2._gas_deductible_cost
trans2_table_gas['TransferToOil'] = trans2._transfer_to_oil
trans2_table_gas['TransferToGas'] = trans2._transfer_to_gas
trans2_table_gas['CarryForwardCostafterTF'] = trans2._gas_carward_cost_aftertf
trans2_table_gas['CTRShareAfterTF'] = trans2._gas_ctr_share_after_transfer
trans2_table_gas['CTRNetOperatingProfit'] = trans2._gas_net_operating_profit
trans2_table_gas['DMOVolume'] = trans2._gas_dmo_volume
trans2_table_gas['DMOFee'] = trans2._gas_dmo_fee
trans2_table_gas['DDMO'] = trans2._gas_ddmo
trans2_table_gas['TaxableIncome'] = trans2._gas_taxable_income
trans2_table_gas['Tax'] = trans2._gas_tax
trans2_table_gas['NetCTRShare'] = trans2._gas_ctr_net_share
trans2_table_gas['CTRCashFlow'] = trans2._gas_ctr_cashflow
trans2_table_gas['GovernmentTake'] = trans2._gas_government_take
trans2_table_gas.loc['Column_Total'] = trans2_table_gas.sum(numeric_only=True, axis=0)
print(trans2_table_gas, '\n')
