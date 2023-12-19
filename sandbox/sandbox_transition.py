import numpy as np
import pandas as pd
import time
from datetime import datetime

# from pyscnomics.econ.selection import FTPTaxRegime, FluidType, TaxPaymentMode
from pyscnomics.econ.selection import FTPTaxRegime, FluidType
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.costs import Tangible, Intangible, OPEX, ASR
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit

from pyscnomics.tools.summary import get_summary
from pyscnomics.econ.selection import NPVSelection, DiscountingMode

from pyscnomics.contracts import transition

# pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

"""
Cost Recovery
"""

# Defining Start Date and End Date
psc_1_start_date = datetime.strptime("01/01/2018", '%d/%m/%Y').date()
psc_1_end_date = datetime.strptime("22/4/2020", '%d/%m/%Y').date()

# Defining the Gas lifting data
psc1_gas_lifting = Lifting(
    start_year=2018,
    end_year=2020,
    lifting_rate=np.array([1.58446786291200, 0.98961346458056]),
    price=np.array([6.260, 6.260]),
    prod_year=np.array([2019, 2020]),
    ghv=np.array([1047.0, 1047.0]),
    fluid_type=FluidType.GAS)

# Defining the Gas Tangible Data - Drilling Tangible
psc1_gas_tangible = Tangible(
    start_year=2018,
    end_year=2020,
    cost=np.array([3363.67743703704000, 802.73224043715800]),
    expense_year=np.array([2019, 2020]),
    cost_allocation=[FluidType.GAS] * 2,
    pis_year=np.array([2019, 2020]),
    useful_life=np.array([5, 5]),
    depreciation_factor=np.array([0.25, 0.25]),
    )

# Defining the Gas Intangible Data
psc1_gas_intang = Intangible(
    start_year=2018,
    end_year=2020,
    cost=np.array([9532.633600000]),
    expense_year=np.array([2019]),
    cost_allocation=[FluidType.GAS])

# Defining the Gas OPEX Data
psc1_gas_opex_cost = OPEX(
    start_year=2018,
    end_year=2020,
    fixed_cost=np.array([2076.908222642980, 1297.582047244550]),
    expense_year=np.array([2019, 2020]),
    cost_allocation=[FluidType.GAS] * 2)

# Defining the Gas ASR Data
psc1_gas_asr_cost_opx = OPEX(
    start_year=2018,
    end_year=2020,
    fixed_cost=np.array([35.515809523809900, 10.965263596148900]),
    expense_year=np.array([2019, 2020]),
    cost_allocation=[FluidType.GAS] * 2)

psc1_gas_asr_cost = ASR(
    start_year=2018,
    end_year=2020,
    cost=np.array([0]),
    expense_year=np.array([2019]),
    cost_allocation=[FluidType.GAS])

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


"""
Gross Split
"""
# Defining Start Date and End Date
psc_2_start_date = datetime.strptime("23/4/2020", '%d/%m/%Y').date()
psc_2_end_date = datetime.strptime("22/4/2030", '%d/%m/%Y').date()

# Defining the Gas lifting data
psc2_gas_lifting = Lifting(
    start_year=2020,
    end_year=2030,
    lifting_rate=np.array(
        [2.21568324370692000, 3.20769606628721000, 3.29284116326400000, 3.37370744832000000,
         3.44718555313867000, 3.40400062841705000, 3.32543155814400000, 2.00043667046400000]),
    price=np.array([6.2600, 6.2600, 6.2600, 6.2600, 6.2600, 6.2600, 6.2600, 6.2600]),
    prod_year=np.array([2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027]),
    ghv=np.array([1047.0, 1047.0, 1047.0, 1047.0, 1047.0, 1047.0, 1047.0, 1047.0, ]),
    fluid_type=FluidType.GAS)

# Defining the Gas Tangible Data - Drilling Tangible
psc2_gas_tangible = Tangible(
    start_year=2020,
    end_year=2030,
    cost=np.array([1959.561038251370, 2834.780000000000]),
    expense_year=np.array([2020, 2025]),
    pis_year=np.array([2020, 2025]),
    useful_life=np.array([5, 5]),
    depreciation_factor=np.array([0.25, 0.25]),
    cost_allocation=[FluidType.GAS] * 2)

# Defining the Gas Intangible Data
psc2_gas_intang = Intangible(
    start_year=2020,
    end_year=2030,
    cost=np.array([0]),
    expense_year=np.array([2020]),
    cost_allocation=[FluidType.GAS])

# Defining the Gas OPEX Data
psc2_gas_opex_cost = OPEX(
    start_year=2020,
    end_year=2030,
    fixed_cost=np.array(
        [3137.4561521272200, 3834.8277754882300, 3943.8595100171400, 4040.8953543109500, 4114.6574450911200,
         4272.4109557000400, 5057.6546904997800, 4462.7942473087100, ]),
    expense_year=np.array([2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027]),
    cost_allocation=[FluidType.GAS] * 8)

psc2_gas_lbt_cost = OPEX(
    start_year=2020,
    end_year=2030,
    fixed_cost=np.array(
        [636.40338233533300, 1062.86476325927000, 1091.05051487026000, 1117.33210409363000, 1140.29666892703000,
         1296.08012023605000, 1124.78850166335000, 786.84662704513200]),
    expense_year=np.array([2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027]),
    cost_allocation=[FluidType.GAS] * 8)

# Defining VAT and Import Duty
psc2_gas_vat = OPEX(
    start_year=2020,
    end_year=2030,
    fixed_cost=np.array(
        [396.5062479576550, 286.7125283653920, 294.7836073052500, 301.9666899432990, 307.4269319547460,
         553.8846425099840, 377.2323444857470, 333.1977740376320, ]),
    expense_year=np.array([2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027]),
    cost_allocation=[FluidType.GAS] * 8)

psc2_gas_import_duty = OPEX(
    start_year=2020,
    end_year=2030,
    fixed_cost=np.array(
        [159.965353, 55.922218, 57.496452, 58.897486, 59.962486, 227.566547, 73.577773, 64.988993]),
    expense_year=np.array([2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027]),
    cost_allocation=[FluidType.GAS] * 8)

# Defining the Gas ASR Data
psc2_gas_asr_cost_opx = OPEX(
    start_year=2020,
    end_year=2030,
    fixed_cost=np.array(
        [26.513186, 38.355043, 38.355043, 38.355043, 38.355043, 38.355043, 38.355043, 38.355043, ]),
    expense_year=np.array([2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027]),
    cost_allocation=[FluidType.GAS] * 8)

psc2_gas_asr_cost = ASR(
    start_year=2020,
    end_year=2030,
    cost=np.array([0]),
    expense_year=np.array([2020]),
    cost_allocation=[FluidType.GAS])

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
    prod_stage='Primary',
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

ftp_tax_regime = FTPTaxRegime.DIRECT_MODE
eff_tax_rate = 0.48
# tax_payment_method = TaxPaymentMode.TAX_DIRECT_MODE
argument_contract1 = {'ftp_tax_regime': ftp_tax_regime,
                      'tax_rate': eff_tax_rate,
                      # 'tax_payment_method': tax_payment_method
                      }

argument_contract2 = {'tax_rate': 0.40}

psc_trans = transition.Transition(contract1=psc1,
                                  contract2=psc2,
                                  argument_contract1=argument_contract1,
                                  argument_contract2=argument_contract2)
start_time = time.time()
psc_trans.run(unrec_portion=0)
end_time = time.time()
print('Execution Time: ', end_time - start_time, '\n')

print('Cost Recovery Transition')
trans1_table_oil = pd.DataFrame()
trans1_table_oil['Year'] = psc_trans._contract1_transitioned.project_years
trans1_table_oil['Lifting'] = psc_trans._contract1_transitioned._oil_lifting.get_lifting_rate_arr()
trans1_table_oil['Price'] = psc_trans._contract1_transitioned._oil_lifting.get_price_arr()
trans1_table_oil['Revenue'] = psc_trans._contract1_transitioned._oil_revenue
trans1_table_oil['Depreciable'] = psc_trans._contract1_transitioned._oil_tangible_expenditures
trans1_table_oil['Intangible'] = psc_trans._contract1_transitioned._oil_intangible_expenditures
trans1_table_oil['Opex'] = psc_trans._contract1_transitioned._oil_opex_expenditures
trans1_table_oil['ASR'] = psc_trans._contract1_transitioned._oil_asr_expenditures
trans1_table_oil['Depreciation'] = psc_trans._contract1_transitioned._oil_depreciation
trans1_table_oil['NonCapital'] = psc_trans._contract1_transitioned._oil_non_capital
trans1_table_oil['FTP'] = psc_trans._contract1_transitioned._oil_ftp
trans1_table_oil['FTP_CTR'] = psc_trans._contract1_transitioned._oil_ftp_ctr
trans1_table_oil['FTP_GOV'] = psc_trans._contract1_transitioned._oil_ftp_gov
trans1_table_oil['InvestmentCredit'] = psc_trans._contract1_transitioned._oil_ic_paid
trans1_table_oil['UnrecoveredCost'] = psc_trans._contract1_transitioned._oil_unrecovered_before_transfer
trans1_table_oil['CosttoBeRecovered'] = psc_trans._contract1_transitioned._oil_cost_to_be_recovered
trans1_table_oil['CostRecovery'] = psc_trans._contract1_transitioned._oil_cost_recovery
trans1_table_oil['ETSBeforeTransfer'] = psc_trans._contract1_transitioned._oil_ets_before_transfer
trans1_table_oil['TransfertoOil'] = psc_trans._contract1_transitioned._transfer_to_oil
trans1_table_oil['TransfertoGas'] = psc_trans._contract1_transitioned._transfer_to_gas
trans1_table_oil['UnrecafterTransfer'] = psc_trans._contract1_transitioned._oil_unrecovered_after_transfer
trans1_table_oil['CostToBeRecoveredAfterTF'] = psc_trans._contract1_transitioned._oil_cost_to_be_recovered_after_tf
trans1_table_oil['CostRecoveryAfterTF'] = psc_trans._contract1_transitioned._oil_cost_recovery_after_tf
trans1_table_oil['ETSAfterTransfer'] = psc_trans._contract1_transitioned._oil_ets_after_transfer
trans1_table_oil['ContractorShare'] = psc_trans._contract1_transitioned._oil_contractor_share
trans1_table_oil['GovernmentShare'] = psc_trans._contract1_transitioned._oil_government_share
trans1_table_oil['DMOVolume'] = psc_trans._contract1_transitioned._oil_dmo_volume
trans1_table_oil['DMOFee'] = psc_trans._contract1_transitioned._oil_dmo_fee
trans1_table_oil['DDMO'] = psc_trans._contract1_transitioned._oil_ddmo
trans1_table_oil['TaxableIncome'] = psc_trans._contract1_transitioned._oil_taxable_income
trans1_table_oil['Tax'] = psc_trans._contract1_transitioned._oil_tax_payment
trans1_table_oil['NetContractorShare'] = psc_trans._contract1_transitioned._oil_ctr_net_share
trans1_table_oil['ContractorTake'] = psc_trans._contract1_transitioned._oil_ctr_net_share
trans1_table_oil['Cashflow'] = psc_trans._contract1_transitioned._oil_cashflow
trans1_table_oil['GovernmentTake'] = psc_trans._contract1_transitioned._oil_government_take
trans1_table_oil.loc['Column_Total'] = trans1_table_oil.sum(numeric_only=True, axis=0)
print(trans1_table_oil, '\n')

trans1_table_gas = pd.DataFrame()
trans1_table_gas['Year'] = psc_trans._contract1_transitioned.project_years
trans1_table_gas['Lifting'] = psc_trans._contract1_transitioned._gas_lifting.get_lifting_rate_arr()
trans1_table_gas['Price'] = psc_trans._contract1_transitioned._gas_lifting.get_price_arr()
trans1_table_gas['Revenue'] = psc_trans._contract1_transitioned._gas_revenue
trans1_table_gas['Depreciable'] = psc_trans._contract1_transitioned._gas_tangible_expenditures
trans1_table_gas['Intangible'] = psc_trans._contract1_transitioned._gas_intangible_expenditures
trans1_table_gas['Opex'] = psc_trans._contract1_transitioned._gas_opex_expenditures
trans1_table_gas['ASR'] = psc_trans._contract1_transitioned._gas_asr_expenditures
trans1_table_gas['Depreciation'] = psc_trans._contract1_transitioned._gas_depreciation
trans1_table_gas['NonCapital'] = psc_trans._contract1_transitioned._gas_non_capital
trans1_table_gas['FTP'] = psc_trans._contract1_transitioned._gas_ftp
trans1_table_gas['FTP_CTR'] = psc_trans._contract1_transitioned._gas_ftp_ctr
trans1_table_gas['FTP_GOV'] = psc_trans._contract1_transitioned._gas_ftp_gov
trans1_table_gas['InvestmentCredit'] = psc_trans._contract1_transitioned._gas_ic_paid
trans1_table_gas['UnrecoveredCost'] = psc_trans._contract1_transitioned._gas_unrecovered_before_transfer
trans1_table_gas['CosttoBeRecovered'] = psc_trans._contract1_transitioned._gas_cost_to_be_recovered
trans1_table_gas['CostRecovery'] = psc_trans._contract1_transitioned._gas_cost_recovery
trans1_table_gas['ETSBeforeTransfer'] = psc_trans._contract1_transitioned._gas_ets_before_transfer
trans1_table_gas['TransfertoOil'] = psc_trans._contract1_transitioned._transfer_to_oil
trans1_table_gas['TransfertoGas'] = psc_trans._contract1_transitioned._transfer_to_gas
trans1_table_gas['UnrecafterTransfer'] = psc_trans._contract1_transitioned._gas_unrecovered_after_transfer
trans1_table_gas['CostToBeRecoveredAfterTF'] = psc_trans._contract1_transitioned._gas_cost_to_be_recovered_after_tf
trans1_table_gas['CostRecoveryAfterTF'] = psc_trans._contract1_transitioned._gas_cost_recovery_after_tf
trans1_table_gas['ETSAfterTransfer'] = psc_trans._contract1_transitioned._gas_ets_after_transfer
trans1_table_gas['ContractorShare'] = psc_trans._contract1_transitioned._gas_contractor_share
trans1_table_gas['GovernmentShare'] = psc_trans._contract1_transitioned._gas_government_share
trans1_table_gas['DMOVolume'] = psc_trans._contract1_transitioned._gas_dmo_volume
trans1_table_gas['DMOFee'] = psc_trans._contract1_transitioned._gas_dmo_fee
trans1_table_gas['DDMO'] = psc_trans._contract1_transitioned._gas_ddmo
trans1_table_gas['TaxableIncome'] = psc_trans._contract1_transitioned._gas_taxable_income
trans1_table_gas['Tax'] = psc_trans._contract1_transitioned._gas_tax_payment
trans1_table_gas['NetContractorShare'] = psc_trans._contract1_transitioned._gas_ctr_net_share
trans1_table_gas['ContractorTake'] = psc_trans._contract1_transitioned._gas_ctr_net_share
trans1_table_gas['Cashflow'] = psc_trans._contract1_transitioned._gas_cashflow
trans1_table_gas['GovernmentTake'] = psc_trans._contract1_transitioned._gas_government_take
trans1_table_gas.loc['Column_Total'] = trans1_table_gas.sum(numeric_only=True, axis=0)
print(trans1_table_gas, '\n')

trans1_table_consolidated = pd.DataFrame()
trans1_table_consolidated['Year'] = psc_trans._contract1_transitioned.project_years
trans1_table_consolidated['cnsltd_revenue'] = psc_trans._contract1_transitioned._consolidated_revenue
trans1_table_consolidated['cnsltd_non_capital'] = psc_trans._contract1_transitioned._consolidated_non_capital
trans1_table_consolidated['cnsltd_ic'] = psc_trans._contract1_transitioned._consolidated_ic
trans1_table_consolidated['cnsltd_ic_unrecovered'] = psc_trans._contract1_transitioned._consolidated_ic_unrecovered
trans1_table_consolidated['cnsltd_ic_paid'] = psc_trans._contract1_transitioned._consolidated_ic_paid
trans1_table_consolidated['cnsltd_unrecovered_before_transfer'] = psc_trans._contract1_transitioned._consolidated_unrecovered_before_transfer
trans1_table_consolidated['cnsltd_cost_recovery'] = psc_trans._contract1_transitioned._consolidated_cost_recovery_before_transfer
trans1_table_consolidated['cnsltd_ets_before_transfer'] = psc_trans._contract1_transitioned._consolidated_ets_before_transfer
trans1_table_consolidated['cnsltd_unrecovered_after_transfer'] = psc_trans._contract1_transitioned._consolidated_unrecovered_after_transfer
trans1_table_consolidated['cnsltd_cost_to_be_recovered_after_tf'] = psc_trans._contract1_transitioned._consolidated_cost_to_be_recovered_after_tf
trans1_table_consolidated['cnsltd_cost_recovery_after_tf'] = psc_trans._contract1_transitioned._consolidated_cost_recovery_after_tf
trans1_table_consolidated['cnsltd_ets_after_transfer'] = psc_trans._contract1_transitioned._consolidated_ets_after_transfer
trans1_table_consolidated['cnsltd_contractor_share'] = psc_trans._contract1_transitioned._consolidated_contractor_share
trans1_table_consolidated['cnsltd_government_share'] = psc_trans._contract1_transitioned._consolidated_government_share
trans1_table_consolidated['cnsltd_dmo_volume'] = psc_trans._contract1_transitioned._consolidated_dmo_volume
trans1_table_consolidated['cnsltd_dmo_fee'] = psc_trans._contract1_transitioned._consolidated_dmo_fee
trans1_table_consolidated['cnsltd_ddmo'] = psc_trans._contract1_transitioned._consolidated_ddmo
trans1_table_consolidated['cnsltd_taxable_income'] = psc_trans._contract1_transitioned._consolidated_taxable_income
trans1_table_consolidated['cnsltd_tax_due'] = psc_trans._contract1_transitioned._consolidated_tax_due
trans1_table_consolidated['cnsltd_unpaid_tax_balance'] = psc_trans._contract1_transitioned._consolidated_unpaid_tax_balance
trans1_table_consolidated['cnsltd_tax_payment'] = psc_trans._contract1_transitioned._consolidated_tax_payment
trans1_table_consolidated['cnsltd_ctr_net_share'] = psc_trans._contract1_transitioned._consolidated_ctr_net_share
trans1_table_consolidated['cnsltd_contractor_take'] = psc_trans._contract1_transitioned._consolidated_contractor_take
trans1_table_consolidated['cnsltd_government_take'] = psc_trans._contract1_transitioned._consolidated_government_take
trans1_table_consolidated['cnsltd_cashflow'] = psc_trans._contract1_transitioned._consolidated_cashflow
trans1_table_consolidated['cum. cnsltd_cashflow'] = np.cumsum(psc_trans._contract1_transitioned._consolidated_cashflow)
trans1_table_consolidated.loc['Column_Total'] = trans1_table_consolidated.sum(numeric_only=True, axis=0)
print(trans1_table_consolidated, '\n')

print('Gross Split Transition')
trans2_table_oil = pd.DataFrame()
trans2_table_oil['Years'] = psc_trans._contract2_transitioned.project_years
trans2_table_oil['Lifting'] = psc_trans._contract2_transitioned._oil_lifting.get_lifting_rate_arr()
trans2_table_oil['Revenue'] = psc_trans._contract2_transitioned._oil_revenue
trans2_table_oil['BaseSplit'] = psc_trans._contract2_transitioned._oil_base_split
trans2_table_oil['VariableSplit'] = psc_trans._contract2_transitioned._var_split_array
trans2_table_oil['ProgressiveSplit'] = psc_trans._contract2_transitioned._oil_prog_split
trans2_table_oil['ContractorSplitSplit'] = psc_trans._contract2_transitioned._oil_ctr_split
trans2_table_oil['GovernmentShare'] = psc_trans._contract2_transitioned._oil_gov_share
trans2_table_oil['ContractorShare'] = psc_trans._contract2_transitioned._oil_ctr_share_before_transfer
trans2_table_oil['Depreciation'] = psc_trans._contract2_transitioned._oil_depreciation
trans2_table_oil['NonCapital'] = psc_trans._contract2_transitioned._oil_non_capital
trans2_table_oil['TotalExpenses'] = psc_trans._contract2_transitioned._oil_total_expenses
trans2_table_oil['CostToBeDeducted'] = psc_trans._contract2_transitioned._oil_cost_tobe_deducted
trans2_table_oil['CarryForwardCost'] = psc_trans._contract2_transitioned._oil_carward_deduct_cost
trans2_table_oil['DeductibleCost'] = psc_trans._contract2_transitioned._oil_deductible_cost
trans2_table_oil['TransferToOil'] = psc_trans._contract2_transitioned._transfer_to_oil
trans2_table_oil['TransferToGas'] = psc_trans._contract2_transitioned._transfer_to_gas
trans2_table_oil['CarryForwardCostafterTF'] = psc_trans._contract2_transitioned._oil_carward_cost_aftertf
trans2_table_oil['CTRShareAfterTF'] = psc_trans._contract2_transitioned._oil_ctr_share_after_transfer
trans2_table_oil['CTRNetOperatingProfit'] = psc_trans._contract2_transitioned._oil_net_operating_profit
trans2_table_oil['DMOVolume'] = psc_trans._contract2_transitioned._oil_dmo_volume
trans2_table_oil['DMOFee'] = psc_trans._contract2_transitioned._oil_dmo_fee
trans2_table_oil['DDMO'] = psc_trans._contract2_transitioned._oil_ddmo
trans2_table_oil['TaxableIncome'] = psc_trans._contract2_transitioned._oil_taxable_income
trans2_table_oil['Tax'] = psc_trans._contract2_transitioned._oil_tax
trans2_table_oil['NetCTRShare'] = psc_trans._contract2_transitioned._oil_ctr_net_share
trans2_table_oil['CTRCashFlow'] = psc_trans._contract2_transitioned._oil_ctr_cashflow
trans2_table_oil['GovernmentTake'] = psc_trans._contract2_transitioned._oil_government_take
trans2_table_oil.loc['Column_Total'] = trans2_table_oil.sum(numeric_only=True, axis=0)
print(trans2_table_oil, '\n')

trans2_table_gas = pd.DataFrame()
trans2_table_gas['Years'] = psc_trans._contract2_transitioned.project_years
trans2_table_gas['Lifting'] = psc_trans._contract2_transitioned._gas_lifting.get_lifting_rate_arr()
trans2_table_gas['Revenue'] = psc_trans._contract2_transitioned._gas_revenue
trans2_table_gas['BaseSplit'] = psc_trans._contract2_transitioned._gas_base_split
trans2_table_gas['VariableSplit'] = psc_trans._contract2_transitioned._var_split_array
trans2_table_gas['ProgressiveSplit'] = psc_trans._contract2_transitioned._gas_prog_split
trans2_table_gas['ContractorSplitSplit'] = psc_trans._contract2_transitioned._gas_ctr_split
trans2_table_gas['GovernmentShare'] = psc_trans._contract2_transitioned._gas_gov_share
trans2_table_gas['ContractorShare'] = psc_trans._contract2_transitioned._gas_ctr_share_before_transfer
trans2_table_gas['Depreciation'] = psc_trans._contract2_transitioned._gas_depreciation
trans2_table_gas['NonCapital'] = psc_trans._contract2_transitioned._gas_non_capital
trans2_table_gas['TotalExpenses'] = psc_trans._contract2_transitioned._gas_total_expenses
trans2_table_gas['CostToBeDeducted'] = psc_trans._contract2_transitioned._gas_cost_tobe_deducted
trans2_table_gas['CarryForwardCost'] = psc_trans._contract2_transitioned._gas_carward_deduct_cost
trans2_table_gas['DeductibleCost'] = psc_trans._contract2_transitioned._gas_deductible_cost
trans2_table_gas['TransferToOil'] = psc_trans._contract2_transitioned._transfer_to_oil
trans2_table_gas['TransferToGas'] = psc_trans._contract2_transitioned._transfer_to_gas
trans2_table_gas['CarryForwardCostafterTF'] = psc_trans._contract2_transitioned._gas_carward_cost_aftertf
trans2_table_gas['CTRShareAfterTF'] = psc_trans._contract2_transitioned._gas_ctr_share_after_transfer
trans2_table_gas['CTRNetOperatingProfit'] = psc_trans._contract2_transitioned._gas_net_operating_profit
trans2_table_gas['DMOVolume'] = psc_trans._contract2_transitioned._gas_dmo_volume
trans2_table_gas['DMOFee'] = psc_trans._contract2_transitioned._gas_dmo_fee
trans2_table_gas['DDMO'] = psc_trans._contract2_transitioned._gas_ddmo
trans2_table_gas['TaxableIncome'] = psc_trans._contract2_transitioned._gas_taxable_income
trans2_table_gas['Tax'] = psc_trans._contract2_transitioned._gas_tax
trans2_table_gas['NetCTRShare'] = psc_trans._contract2_transitioned._gas_ctr_net_share
trans2_table_gas['CTRCashFlow'] = psc_trans._contract2_transitioned._gas_ctr_cashflow
trans2_table_gas['GovernmentTake'] = psc_trans._contract2_transitioned._gas_government_take
trans2_table_gas.loc['Column_Total'] = trans2_table_gas.sum(numeric_only=True, axis=0)
print(trans2_table_gas, '\n')

trans2_table_consolidated = pd.DataFrame()
trans2_table_consolidated['Years'] = psc_trans._contract2_transitioned.project_years
trans2_table_consolidated['cnsltd_Revenue'] = psc_trans._contract2_transitioned._consolidated_revenue
trans2_table_consolidated['cnsltd_GovernmentShare'] = psc_trans._contract2_transitioned._consolidated_ctr_share_before_tf
trans2_table_consolidated['cnsltd_ContractorShare'] = psc_trans._contract2_transitioned._consolidated_gov_share_before_tf
trans2_table_consolidated['cnsltd_Depreciation'] = psc_trans._contract2_transitioned._consolidated_depreciation
trans2_table_consolidated['cnsltd_Opex'] = psc_trans._contract2_transitioned._consolidated_opex
trans2_table_consolidated['cnsltd_ASR'] = psc_trans._contract2_transitioned._consolidated_asr
trans2_table_consolidated['cnsltd_NonCapital'] = psc_trans._contract2_transitioned._consolidated_non_capital
trans2_table_consolidated['cnsltd_TotalExpenses'] = psc_trans._contract2_transitioned._consolidated_total_expenses
trans2_table_consolidated['cnsltd_CostToBeDeducted'] = psc_trans._contract2_transitioned._consolidated_cost_tobe_deducted
trans2_table_consolidated['cnsltd_CarryForwardCost'] = psc_trans._contract2_transitioned._consolidated_carward_deduct_cost
trans2_table_consolidated['cnsltd_DeductibleCost'] = psc_trans._contract2_transitioned._consolidated_deductible_cost
trans2_table_consolidated['cnsltd_CarryForwardCostafterTF'] = psc_trans._contract2_transitioned._consolidated_carward_cost_aftertf
trans2_table_consolidated['cnsltd_CTRShareAfterTF'] = psc_trans._contract2_transitioned._consolidated_ctr_share_after_transfer
trans2_table_consolidated['cnsltd_CTRNetOperatingProfit'] = psc_trans._contract2_transitioned._consolidated_net_operating_profit
trans2_table_consolidated['cnsltd_DMOVolume'] = psc_trans._contract2_transitioned._consolidated_dmo_volume
trans2_table_consolidated['cnsltd_DMOFee'] = psc_trans._contract2_transitioned._consolidated_dmo_fee
trans2_table_consolidated['cnsltd_DDMO'] = psc_trans._contract2_transitioned._consolidated_ddmo
trans2_table_consolidated['cnsltd_TaxableIncome'] = psc_trans._contract2_transitioned._consolidated_taxable_income
trans2_table_consolidated['cnsltd_Tax'] = psc_trans._contract2_transitioned._consolidated_tax_payment
trans2_table_consolidated['cnsltd_NetCTRShare'] = psc_trans._contract2_transitioned._consolidated_ctr_net_share
trans2_table_consolidated['cnsltd_CTRCashFlow'] = psc_trans._contract2_transitioned._consolidated_cashflow
trans2_table_consolidated['cnsltd_GovernmentTake'] = psc_trans._contract2_transitioned._consolidated_government_take
trans2_table_consolidated.loc['Column_Total'] = trans2_table_consolidated.sum(numeric_only=True, axis=0)
print(trans2_table_consolidated, '\n')

psc_summary = get_summary(contract=psc_trans,
                          reference_year=2018,
                          inflation_rate=0.1,
                          discount_rate=0.1,
                          npv_mode=NPVSelection.NPV_POINT_FORWARD,
                          discounting_mode=DiscountingMode.END_YEAR)

for key, value in psc_summary.items():
    print(key, ":", value)

# from pyscnomics.tools.table import get_table
#
# psc_table_oil, psc_table_gas, psc_table_consolidated = get_table(contract=psc_trans)
# print(psc_table_oil, '\n')
# print(psc_table_gas, '\n')
# print(psc_table_consolidated, '\n')




