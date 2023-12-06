import numpy as np
import pandas as pd
import time
from datetime import datetime

from pyscnomics.econ.selection import FluidType, TaxPaymentMode, FTPTaxRegime
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.costs import Tangible, Intangible, OPEX, ASR
from pyscnomics.contracts.costrecovery import CostRecovery

from pyscnomics.tools.summary import get_summary
from pyscnomics.econ.selection import NPVSelection, DiscountingMode

from pyscnomics.contracts import transition

# pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

"""
Cost Recovery BT BP
"""

# Defining Project Start Date and End Date
psc_start_date = datetime.strptime("01/01/2012", '%d/%m/%Y').date()
psc_end_date = datetime.strptime("22/4/2042", '%d/%m/%Y').date()


# Defining the Oil and Gas lifting data (MBBL)
lifting_oil = Lifting(
    start_year=2012,
    end_year=2042,
    lifting_rate=np.array(
        [0.00000, 0.00000, 0.00000, 1853.5229250000000, 6341.45752000014842171, 6056.3878166669100, 5476.1695208333400,
         4078.1255333199500, 3055.5392879890200, 2542.8591874987800, 2709.8625041691000, 3507.7193500000000,
         2868.0748200000000, 2110.7183500000000, 1501.9713500000000, 1130.9488500000000, 810.0751200000000,
         588.7121500000000, 456.3814000000000, 366.5476000000000, 300.6616800000000, 239.7995400000000,
         168.6300000000000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]) / 1000,
    price=np.array(
        [0.000, 0.000, 0.000, 37.334466514577400, 39.660, 53.832672127298400, 70.624729244952900, 66.248032437047300,
         47.660, 73.620, 70.00, 79.00, 65.00, 65.00, 65.00, 65.00, 65.00, 65.00, 65.00, 65.00, 65.00, 65.00, 65.00,
         65.00, 65.00, 65.00, 65.00, 65.00, 65.00, 65.00, 65.00]),
    prod_year=np.array(
        [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029,
         2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042]),
    fluid_type=FluidType.OIL)

lifting_gas = Lifting(
    start_year=2012,
    end_year=2042,
    lifting_rate=np.array(
        [0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 3.579756330000000,
         10.073967454168000, 10.893728558319900, 10.646670095894000, 7.475474054987810, 11.960630554154500,
         13.153253454166900, 11.271200000000000, 11.763240000000000, 10.033850000000000, 9.464450000000000,
         7.960650000000000, 7.030860000000000, 6.051700000000000, 5.274250000000000, 4.723100000000000,
         4.212660000000000, 3.755160000000000, 3.336100000000000, 0.000000000000000, 0.000000000000000,
         0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000, 0.000000000000000,
         0.000000000000000]),
    price=np.array(
        [0.0000000, 0.0000000, 0.0000000, 0.0000000, 5.5000000, 5.5800000, 5.7300000, 5.8900000, 6.1900000, 7.1667200,
         7.1667200, 7.1667200, 7.1000000, 7.1000000, 7.1000000, 7.1000000, 7.1000000, 7.1000000, 7.1000000, 7.1000000,
         7.1000000, 7.1000000, 7.1000000, 7.1000000, 7.1000000, 7.1000000, 7.1000000, 7.1000000, 7.1000000, 7.1000000,
         7.1000000, ]),
    prod_year=np.array(
        [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029,
         2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042]),
    ghv=np.array(
        [0.000, 0.000, 0.000, 0.000, 1.200, 1.200, 1.200, 1.200, 1.200, 1.200, 1.200, 1.200, 1.200, 1.200, 1.200, 1.200,
         1.200, 1.200, 1.200, 1.200, 1.200, 1.200, 1.200, 1.200, 1.200, 1.200, 1.200, 1.200, 1.200, 1.200, 1.200]),
    fluid_type=FluidType.GAS)

# Defining the Oil and Gas Tangible Data
tangible_drilling_oil = Tangible(
    start_year=2012,
    end_year=2042,
    cost=np.array(
        [0.65477000400, 6.08143738350, 19.17302483700, 10.01043930900, 0.00000000000, 4.84637437350, 2.53791382650,
         0.29445687450, 5.56380000000, 3.17775000000, 17.98500000000]),
    expense_year=np.array([2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]),
    cost_allocation=[FluidType.OIL] * 11,
    pis_year=np.array([2015, 2015, 2015, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]),
    useful_life=np.array([5.0] * 11),
    depreciation_factor=np.array([0.25] * 11))

tangible_facility_oil = Tangible(
    start_year=2012,
    end_year=2042,
    cost=np.array(
        [10.293000000, 47.393000000, 63.662000000, 19.140000000, 2.043000000, 3.582676690, 4.381758420, 22.363896000,
         36.538772000, 48.149000000, 10.638000000]),
    expense_year=np.array([2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]),
    cost_allocation=[FluidType.OIL] * 11,
    pis_year=np.array([2015, 2015, 2015, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]),
    useful_life=np.array([5.0] * 11),
    depreciation_factor=np.array([0.25] * 11))

tangible_drilling_gas = Tangible(
    start_year=2012,
    end_year=2042,
    cost=np.array([2.496906325500000, 6.419670424500000, 3.258858403070920]),
    expense_year=np.array([2019, 2020, 2021]),
    cost_allocation=[FluidType.GAS] * 3,
    pis_year=np.array([2019, 2020, 2021]),
    useful_life=np.array([4.0] * 3),
    depreciation_factor=np.array([0.25] * 3))

tangible_facility_gas = Tangible(
    start_year=2012,
    end_year=2042,
    cost=np.array(
        [4.1710000000, 91.0560000000, 84.6490000000, 20.4400000000, 0.0000000000, 4.8520000000, 0.0000000000, 3.6060148700, 6.4061937200, 1.4634783300, 3.6077800000]),
    expense_year=np.array([2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]),
    cost_allocation=[FluidType.GAS] * 11,
    pis_year=np.array([2016, 2016, 2016, 2016, 2016, 2017, 2018, 2019, 2020, 2021, 2022]),
    useful_life=np.array([4] * 11),
    depreciation_factor=np.array([0.1] * 11))

# Defining the Oil and Gas Intangible Data
intangible_oil = Intangible(
    start_year=2012,
    end_year=2042,
    cost=np.array([43.210363356000, 65.761478506500, 203.547140743000, 72.725822751000, 0.000000000000, 27.462788116500, 14.381511683500, 1.668588955500, 31.528200000000, 18.007250000000, 101.915000000000]),
    expense_year=np.arange(2012, 2022 + 1),
    cost_allocation=[FluidType.OIL] * 11)

intangible_sunk_cost_oil = Intangible(
    start_year=2012,
    end_year=2042,
    cost=np.array([114.900, 81.000]),
    expense_year=np.array([2012, 2024]),
    cost_allocation=[FluidType.OIL] * 2)

intangible_gas = Intangible(
    start_year=2012,
    end_year=2042,
    cost=np.array([14.149135844500000, 36.378132405500000, 18.466864284068600]),
    expense_year=np.array([2019, 2020, 2021]),
    cost_allocation=[FluidType.GAS] * 3)

# Defining the Oil and Gas OPEX Data
opex_oil = OPEX(
    start_year=2012,
    end_year=2042,
    fixed_cost=np.array(
        [6.3700000000000000, 8.4910000000000000, 68.5180000000000000, 72.4100000000000000, 90.8063124957636000,
         107.0790860961150000, 107.9727482409310000, 98.8126382172948000, 81.3214073651039000, 72.6089834151833000,
         74.5812976703827000, 68.1384476327563000, 68.5753368531589000, 60.2618905893554000, 60.4908195095819000,
         32.7801288413813000, 18.5539709776234000, 15.2870164429902000, 11.8880632710794000, 10.1688689828592000,
         7.9263021287872000, 6.5063411589870800, 5.8274867767473300]),
    expense_year=np.arange(2012, 2034 + 1),
    cost_allocation=[FluidType.OIL] * len(np.arange(2012, 2034 + 1)))

opex_gas = OPEX(
    start_year=2012,
    end_year=2042,
    fixed_cost=np.array(
        [8.885879320893040, 23.077555094094500, 21.783199084961600, 28.669361782705200, 32.299989141724000,
         41.558202770200900, 46.328504227410800, 38.751552367243700, 49.780125015919600, 50.702762632900000,
         67.464414095758200, 60.606023587269500, 71.459672052588600, 79.501814438043800, 79.942801784999900,
         85.756376234745600, 81.644792652618000, 82.826083783641600, 89.475283405160800]),
    expense_year=np.arange(2016, 2034 + 1),
    cost_allocation=[FluidType.GAS] * len(np.arange(2016, 2034 + 1)))

# Defining the Oil and Gas ASR Data
asr_opex_oil = OPEX(
    start_year=2012,
    end_year=2042,
    fixed_cost=np.array([0.990000000000000, 8.047343220000000, 13.000000000000000, 1.505941450000000, 1.213275460000000,
                         1.506583330000000, 1.506583330000000, 12.203166670000000, 12.205249702453600,
                         12.205249702453600, 12.205249702453600, 12.205249702453600, 9.189999732971190]),
    expense_year=np.arange(2015, 2027 + 1),
    cost_allocation=[FluidType.OIL] * len(np.arange(2015, 2027 + 1)))

asr_opex_gas = OPEX(
    start_year=2012,
    end_year=2042,
    fixed_cost=np.array([0]),
    expense_year=np.array([2012]),
    cost_allocation=[FluidType.GAS])

asr_oil = ASR(
    start_year=2012,
    end_year=2042,
    cost=np.array([0]),
    expense_year=np.array([2019]),
    cost_allocation=[FluidType.OIL])

asr_gas = ASR(
    start_year=2012,
    end_year=2042,
    cost=np.array([0]),
    expense_year=np.array([2019]),
    cost_allocation=[FluidType.OIL])

# Parsing the fiscal terms into Cost Recovery
psc = CostRecovery(
    start_date=psc_start_date,
    end_date=psc_end_date,
    lifting=tuple([lifting_oil, lifting_gas]),
    tangible_cost=tuple([tangible_drilling_oil, tangible_facility_oil,
                         tangible_drilling_gas, tangible_facility_gas]),
    intangible_cost=tuple([intangible_oil, intangible_sunk_cost_oil, intangible_gas]),
    opex=tuple([opex_oil, opex_gas, asr_opex_oil, asr_opex_gas]),
    asr_cost=tuple([asr_oil, asr_gas]),
    oil_ftp_is_available=True,
    oil_ftp_is_shared=True,
    oil_ftp_portion=0.2,
    gas_ftp_is_available=True,
    gas_ftp_is_shared=True,
    gas_ftp_portion=0.2,
    oil_ctr_pretax_share=0.2679,
    gas_ctr_pretax_share=0.625,
    oil_dmo_volume_portion=0.25,
    oil_dmo_fee_portion=0.25,
    oil_dmo_holiday_duration=60,
    gas_dmo_volume_portion=0.25,
    gas_dmo_fee_portion=1,
    gas_dmo_holiday_duration=60)

psc.run(tax_rate=0.44, ftp_tax_regime=FTPTaxRegime.PDJP_20_2017)
# tax_regime=,
# tax_rate=,
# ftp_tax_regime=,
# discount_rate_year=,
# depr_method=,

psc_table_oil = pd.DataFrame()
psc_table_oil['Year'] = psc.project_years
psc_table_oil['Lifting'] = psc._oil_lifting.get_lifting_rate_arr()
psc_table_oil['Price'] = psc._oil_lifting.get_price_arr()
psc_table_oil['Revenue'] = psc._oil_revenue
psc_table_oil['Revenue After FTP'] = psc._oil_revenue - psc._oil_ftp
psc_table_oil['Depreciable'] = psc._oil_tangible_expenditures
psc_table_oil['Opex'] = psc._oil_opex_expenditures
psc_table_oil['ASR'] = psc._oil_asr_expenditures
psc_table_oil['Intangible'] = psc._oil_intangible_expenditures
psc_table_oil['Depreciation'] = psc._oil_depreciation
psc_table_oil['Non Capital'] = psc._oil_non_capital
psc_table_oil['FTP'] = psc._oil_ftp
psc_table_oil['FTP -CTR'] = psc._oil_ftp_ctr
psc_table_oil['FTP - GOV'] = psc._oil_ftp_gov
psc_table_oil['Investment Credit'] = psc._oil_ic_paid
psc_table_oil['Unrecovered Cost'] = psc._oil_unrecovered_before_transfer
psc_table_oil['Cost to Be Recovered'] = psc._oil_cost_to_be_recovered
psc_table_oil['Cost Recovery'] = psc._oil_cost_recovery
psc_table_oil['ETS Before Transfer'] = psc._oil_ets_before_transfer
psc_table_oil['Transfer to Oil'] = psc._transfer_to_oil
# psc_table_oil['Transfer to Gas'] = psc._transfer_to_gas
psc_table_oil['Unrec after Transfer'] = psc._oil_unrecovered_after_transfer
psc_table_oil['Cost To Be Recovered After TF'] = psc._oil_cost_to_be_recovered_after_tf
psc_table_oil['Cost Recovery After TF'] = psc._oil_cost_recovery_after_tf
psc_table_oil['ETS After Transfer'] = psc._oil_ets_after_transfer
psc_table_oil['Contractor Share Prior Tax'] = psc._oil_contractor_share
psc_table_oil['Government Share'] = psc._oil_government_share
psc_table_oil['DMO Volume'] = psc._oil_dmo_volume
psc_table_oil['DMO Fee'] = psc._oil_dmo_fee
psc_table_oil['DDMO'] = psc._oil_ddmo
psc_table_oil['Taxable Income'] = psc._oil_taxable_income
psc_table_oil['Tax'] = psc._oil_tax_payment
psc_table_oil['Contractor Share'] = psc._oil_ctr_net_share
psc_table_oil['Contractor Take'] = psc._oil_contractor_take
psc_table_oil['Cashflow'] = psc._oil_cashflow
psc_table_oil['Government Take'] = psc._oil_government_take
psc_table_oil.loc['Column_Total'] = psc_table_oil.sum(numeric_only=True, axis=0)
print(psc_table_oil, '\n')

psc_table_gas = pd.DataFrame()
psc_table_gas['Year'] = psc.project_years
psc_table_gas['Lifting'] = psc._gas_lifting.get_lifting_rate_arr()
psc_table_gas['Price'] = psc._gas_wap_price
psc_table_gas['Revenue'] = psc._gas_revenue
psc_table_gas['Revenue After FTP'] = psc._gas_revenue - psc._gas_ftp
psc_table_gas['Depreciable'] = psc._gas_tangible_expenditures
psc_table_gas['Opex'] = psc._gas_opex_expenditures
psc_table_gas['ASR'] = psc._gas_asr_expenditures
psc_table_gas['Intangible'] = psc._gas_intangible_expenditures
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
psc_table_gas['Contractor Share Prior Tax'] = psc._gas_contractor_share
psc_table_gas['Government Share'] = psc._gas_government_share
psc_table_gas['DMO Volume'] = psc._gas_dmo_volume
psc_table_gas['DMO Fee'] = psc._gas_dmo_fee
psc_table_gas['DDMO'] = psc._gas_ddmo
psc_table_gas['Taxable Income'] = psc._gas_taxable_income
psc_table_gas['Tax'] = psc._gas_tax_payment
psc_table_gas['Contractor Share'] = psc._gas_ctr_net_share
psc_table_gas['Contractor Take'] = psc._gas_contractor_take
psc_table_gas['Cashflow'] = psc._gas_cashflow
psc_table_gas['Government Take'] = psc._gas_government_take
psc_table_gas.loc['Column_Total'] = psc_table_gas.sum(numeric_only=True, axis=0)
print(psc_table_gas, '\n')

psc_table_consolidated = pd.DataFrame()
psc_table_consolidated['Year'] = psc.project_years
psc_table_consolidated['cnsltd_revenue'] = psc._consolidated_revenue
psc_table_consolidated['cnsltd_non_capital'] = psc._consolidated_non_capital
psc_table_consolidated['cnsltd_ic'] = psc._consolidated_ic
psc_table_consolidated['cnsltd_ic_unrecovered'] = psc._consolidated_ic_unrecovered
psc_table_consolidated['cnsltd_ic_paid'] = psc._consolidated_ic_paid
psc_table_consolidated['cnsltd_unrecovered_before_transfer'] = psc._consolidated_unrecovered_before_transfer
psc_table_consolidated['cnsltd_cost_recovery'] = psc._consolidated_cost_recovery_before_transfer
psc_table_consolidated['cnsltd_ets_before_transfer'] = psc._consolidated_ets_before_transfer
psc_table_consolidated['cnsltd_unrecovered_after_transfer'] = psc._consolidated_unrecovered_after_transfer
psc_table_consolidated['cnsltd_cost_to_be_recovered_after_tf'] = psc._consolidated_cost_to_be_recovered_after_tf
psc_table_consolidated['cnsltd_cost_recovery_after_tf'] = psc._consolidated_cost_recovery_after_tf
psc_table_consolidated['cnsltd_ets_after_transfer'] = psc._consolidated_ets_after_transfer
psc_table_consolidated['cnsltd_contractor_share'] = psc._consolidated_contractor_share
psc_table_consolidated['cnsltd_government_share'] = psc._consolidated_government_share
psc_table_consolidated['cnsltd_dmo_volume'] = psc._consolidated_dmo_volume
psc_table_consolidated['cnsltd_dmo_fee'] = psc._consolidated_dmo_fee
psc_table_consolidated['cnsltd_ddmo'] = psc._consolidated_ddmo
psc_table_consolidated['cnsltd_taxable_income'] = psc._consolidated_taxable_income
psc_table_consolidated['cnsltd_tax_due'] = psc._consolidated_tax_due
psc_table_consolidated['cnsltd_unpaid_tax_balance'] = psc._consolidated_unpaid_tax_balance
psc_table_consolidated['cnsltd_tax_payment'] = psc._consolidated_tax_payment
psc_table_consolidated['cnsltd_ctr_net_share'] = psc._consolidated_ctr_net_share
psc_table_consolidated['cnsltd_contractor_take'] = psc._consolidated_contractor_take
psc_table_consolidated['cnsltd_government_take'] = psc._consolidated_government_take
psc_table_consolidated['cnsltd_cashflow'] = psc._consolidated_cashflow
psc_table_consolidated['cum. cnsltd_cashflow'] = np.cumsum(psc._consolidated_cashflow)
psc_table_consolidated.loc['Column_Total'] = psc_table_consolidated.sum(numeric_only=True, axis=0)
print(psc_table_consolidated, '\n')


