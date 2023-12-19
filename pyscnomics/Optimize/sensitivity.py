import numpy as np
import pandas as pd

from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.tools.summary import get_summary
from pyscnomics.tools import table
from pyscnomics.econ.selection import FluidType
from pyscnomics.econ.revenue import Lifting


def run_contract(contract: CostRecovery | GrossSplit,
                 contract_arguments: dict,
                 summary_arguments: dict):
    contract.run(**contract_arguments)

    summary_arguments['contract'] = contract
    summary = get_summary(**summary_arguments)

    # # Printing for Debugging
    # psc_table_oil = pd.DataFrame()
    # psc_table_oil['Year'] = contract.project_years
    # psc_table_oil['Lifting'] = contract._oil_lifting.get_lifting_rate_arr()
    # psc_table_oil['Price'] = contract._oil_lifting.get_price_arr()
    # psc_table_oil['Revenue'] = contract._oil_revenue
    # psc_table_oil['Depreciable'] = contract._oil_tangible_expenditures
    # psc_table_oil['Opex'] = contract._oil_opex_expenditures
    # psc_table_oil['ASR'] = contract._oil_asr_expenditures
    # psc_table_oil['Depreciation'] = contract._oil_depreciation
    # psc_table_oil['Non Capital'] = contract._oil_non_capital
    # psc_table_oil['FTP'] = contract._oil_ftp
    # psc_table_oil['FTP - CTR'] = contract._oil_ftp_ctr
    # psc_table_oil['FTP - GOV'] = contract._oil_ftp_gov
    # psc_table_oil['Investment Credit'] = contract._oil_ic_paid
    # psc_table_oil['Unrecovered Cost'] = contract._oil_unrecovered_before_transfer
    # psc_table_oil['Cost to Be Recovered'] = contract._oil_cost_to_be_recovered
    # psc_table_oil['Cost Recovery'] = contract._oil_cost_recovery
    # psc_table_oil['ETS Before Transfer'] = contract._oil_ets_before_transfer
    # psc_table_oil['Transfer to Oil'] = contract._transfer_to_oil
    # psc_table_oil['Transfer to Gas'] = contract._transfer_to_gas
    # psc_table_oil['Unrec after Transfer'] = contract._oil_unrecovered_after_transfer
    # psc_table_oil['Cost To Be Recovered After TF'] = contract._oil_cost_to_be_recovered_after_tf
    # psc_table_oil['Cost Recovery After TF'] = contract._oil_cost_recovery_after_tf
    # psc_table_oil['ETS After Transfer'] = contract._oil_ets_after_transfer
    # psc_table_oil['Contractor Share Prior Tax'] = contract._oil_contractor_share
    # psc_table_oil['Government Share'] = contract._oil_government_share
    # psc_table_oil['DMO Volume'] = contract._oil_dmo_volume
    # psc_table_oil['DMO Fee'] = contract._oil_dmo_fee
    # psc_table_oil['DDMO'] = contract._oil_ddmo
    # psc_table_oil['Taxable Income'] = contract._oil_taxable_income
    # psc_table_oil['Tax'] = contract._oil_tax_payment
    # psc_table_oil['Contractor Share'] = contract._oil_ctr_net_share
    # psc_table_oil['Contractor Take'] = contract._gas_ctr_net_share
    # psc_table_oil['Cashflow'] = contract._oil_cashflow
    # psc_table_oil['Government Take'] = contract._oil_government_take
    # psc_table_oil.loc['Column_Total'] = psc_table_oil.sum(numeric_only=True, axis=0)
    # print(psc_table_oil, '\n')
    # input()
    return summary


def make_new_contract(contract):
    # Getting the attributes of the contract
    attr_contract = vars(contract)

    # Condition where only one fluid is produced
    produced_fluid = [i.fluid_type for i in attr_contract['lifting']]

    if FluidType.OIL not in produced_fluid:
        attr_contract['oil_onstream_date'] = None
    if FluidType.GAS not in produced_fluid:
        attr_contract['gas_onstream_date'] = None

    if isinstance(contract, CostRecovery):
        new_contract = CostRecovery(
            start_date=attr_contract['start_date'],
            end_date=attr_contract['end_date'],
            oil_onstream_date=attr_contract['oil_onstream_date'],
            gas_onstream_date=attr_contract['gas_onstream_date'],
            lifting=attr_contract['lifting'],
            tangible_cost=attr_contract['tangible_cost'],
            intangible_cost=attr_contract['intangible_cost'],
            opex=attr_contract['opex'],
            asr_cost=attr_contract['asr_cost'],
            oil_ftp_is_available=attr_contract['oil_ftp_is_available'],
            oil_ftp_is_shared=attr_contract['oil_ftp_is_shared'],
            oil_ftp_portion=attr_contract['oil_ftp_portion'],
            gas_ftp_is_available=attr_contract['gas_ftp_is_available'],
            gas_ftp_is_shared=attr_contract['gas_ftp_is_shared'],
            gas_ftp_portion=attr_contract['gas_ftp_portion'],
            tax_split_type=attr_contract['tax_split_type'],
            condition_dict=attr_contract['condition_dict'],
            indicator_rc_icp_sliding=attr_contract['indicator_rc_icp_sliding'],
            oil_ctr_pretax_share=attr_contract['oil_ctr_pretax_share'],
            gas_ctr_pretax_share=attr_contract['gas_ctr_pretax_share'],
            oil_ic_rate=attr_contract['oil_ic_rate'],
            gas_ic_rate=attr_contract['gas_ic_rate'],
            ic_is_available=attr_contract['ic_is_available'],
            oil_cr_cap_rate=attr_contract['oil_cr_cap_rate'],
            gas_cr_cap_rate=attr_contract['gas_cr_cap_rate'],
            oil_dmo_volume_portion=attr_contract['oil_dmo_volume_portion'],
            oil_dmo_fee_portion=attr_contract['oil_dmo_fee_portion'],
            oil_dmo_holiday_duration=attr_contract['oil_dmo_holiday_duration'],
            gas_dmo_volume_portion=attr_contract['gas_dmo_volume_portion'],
            gas_dmo_fee_portion=attr_contract['gas_dmo_fee_portion'],
            gas_dmo_holiday_duration=attr_contract['gas_dmo_holiday_duration'])
    else:
        new_contract = GrossSplit(
            start_date=attr_contract['start_date'],
            end_date=attr_contract['end_date'],
            oil_onstream_date=attr_contract['oil_onstream_date'],
            gas_onstream_date=attr_contract['gas_onstream_date'],
            lifting=attr_contract['lifting'],
            tangible_cost=attr_contract['tangible_cost'],
            intangible_cost=attr_contract['intangible_cost'],
            opex=attr_contract['opex'],
            asr_cost=attr_contract['asr_cost'],
            field_status=attr_contract['field_status'],
            field_loc=attr_contract['field_loc'],
            res_depth=attr_contract['res_depth'],
            infra_avail=attr_contract['infra_avail'],
            res_type=attr_contract['res_type'],
            api_oil=attr_contract['api_oil'],
            domestic_use=attr_contract['domestic_use'],
            prod_stage=attr_contract['prod_stage'],
            co2_content=attr_contract['co2_content'],
            h2s_content=attr_contract['h2s_content'],
            base_split_ctr_oil=attr_contract['base_split_ctr_oil'],
            base_split_ctr_gas=attr_contract['base_split_ctr_gas'],
            split_ministry_disc=attr_contract['split_ministry_disc'],
            oil_dmo_volume_portion=attr_contract['oil_dmo_volume_portion'],
            oil_dmo_fee_portion=attr_contract['oil_dmo_fee_portion'],
            oil_dmo_holiday_duration=attr_contract['oil_dmo_holiday_duration'],
            gas_dmo_volume_portion=attr_contract['gas_dmo_volume_portion'],
            gas_dmo_fee_portion=attr_contract['gas_dmo_fee_portion'],
            gas_dmo_holiday_duration=attr_contract['gas_dmo_holiday_duration'],
            conversion_bboe2bscf=attr_contract['conversion_bboe2bscf'])

    return new_contract


def adjust_contract(contract: CostRecovery | GrossSplit,
                    adjusted_variable: str,
                    multiply_factor: float,
                    contract_arguments: dict,
                    summary_arguments: dict):
    # Some attributes that have to be adjusted, they are OPEX, CAPEX, Oil Price, Gas Price, Prod Factor

    # Modifying the OPEX if the chosen adjusted variable is OPEX
    new_opex = contract.opex
    if adjusted_variable == 'OPEX':
        list_opex = []
        for opx in contract.opex:
            opex = opx * multiply_factor
            list_opex.append(opex)
        new_opex = tuple(list_opex)

    # Modifying the CAPEX if the chosen adjusted variable is CAPEX
    new_capex = contract.tangible_cost
    if adjusted_variable == 'CAPEX':
        list_capex = []
        for cpx in contract.tangible_cost:
            capex = cpx * multiply_factor
            list_capex.append(capex)
        new_capex = tuple(list_capex)

    # Modifying the Price if the chosen adjusted variable is Oil Price or Gas Price
    # if adjusted_variable == 'Oil Price' or adjusted_variable == 'Gas Price':
    new_lifting = contract.lifting

    if adjusted_variable == 'Oil Price':
        list_lifting = []
        for lft in contract.lifting:
            if lft.fluid_type == FluidType.OIL:
                lifting_price = lft.price * multiply_factor
                lift = Lifting(start_year=lft.start_year,
                               end_year=lft.end_year,
                               lifting_rate=lft.lifting_rate,
                               price=lifting_price,
                               prod_year=lft.prod_year,
                               fluid_type=lft.fluid_type,
                               ghv=lft.ghv,
                               prod_rate=lft.prod_rate)
                list_lifting.append(lift)
            elif lft.fluid_type == FluidType.GAS:
                lift = Lifting(start_year=lft.start_year,
                               end_year=lft.end_year,
                               lifting_rate=lft.lifting_rate,
                               price=lft.price,
                               prod_year=lft.prod_year,
                               fluid_type=lft.fluid_type,
                               ghv=lft.ghv,
                               prod_rate=lft.prod_rate)
                list_lifting.append(lift)
        new_lifting = tuple(list_lifting)

    # Modifying the Price if the chosen adjusted variable is Oil Price or Gas Price
    # if adjusted_variable == 'Oil Price' or adjusted_variable == 'Gas Price':
    if adjusted_variable == 'Gas Price':
        list_lifting = []
        for lft in contract.lifting:
            if lft.fluid_type == FluidType.GAS:
                lifting_price = lft.price * multiply_factor
                lift = Lifting(start_year=lft.start_year,
                               end_year=lft.end_year,
                               lifting_rate=lft.lifting_rate,
                               price=lifting_price,
                               prod_year=lft.prod_year,
                               fluid_type=lft.fluid_type,
                               ghv=lft.ghv,
                               prod_rate=lft.prod_rate)
                list_lifting.append(lift)
            elif lft.fluid_type == FluidType.OIL:
                lift = Lifting(start_year=lft.start_year,
                               end_year=lft.end_year,
                               lifting_rate=lft.lifting_rate,
                               price=lft.price,
                               prod_year=lft.prod_year,
                               fluid_type=lft.fluid_type,
                               ghv=lft.ghv,
                               prod_rate=lft.prod_rate)
                list_lifting.append(lift)
        new_lifting = tuple(list_lifting)

    # Modifying the Lifting Rate if the chosen adjusted variable is Prod Factor
    if adjusted_variable == 'Prod Factor':
        list_lifting = []
        for lft in contract.lifting:
            lifting_rate_adjusted = lft.lifting_rate * multiply_factor
            lift = Lifting(start_year=lft.start_year,
                           end_year=lft.end_year,
                           lifting_rate=lifting_rate_adjusted,
                           price=lft.price,
                           prod_year=lft.prod_year,
                           fluid_type=lft.fluid_type,
                           ghv=lft.ghv,
                           prod_rate=lft.prod_rate)
            list_lifting.append(lift)

        new_lifting = tuple(list_lifting)

    # Making new contract object that filled with adjusted attributes
    # Getting the attributes of the contract
    attr_contract = vars(contract)

    # Condition where only one fluid is produced
    produced_fluid = [i.fluid_type for i in attr_contract['lifting']]

    if FluidType.OIL not in produced_fluid:
        attr_contract['oil_onstream_date'] = None
    if FluidType.GAS not in produced_fluid:
        attr_contract['gas_onstream_date'] = None

    new_contract = None
    if isinstance(contract, CostRecovery):
        new_contract = CostRecovery(
            start_date=attr_contract['start_date'],
            end_date=attr_contract['end_date'],
            oil_onstream_date=attr_contract['oil_onstream_date'],
            gas_onstream_date=attr_contract['gas_onstream_date'],
            lifting=new_lifting,
            tangible_cost=new_capex,
            intangible_cost=attr_contract['intangible_cost'],
            opex=new_opex,
            asr_cost=attr_contract['asr_cost'],
            oil_ftp_is_available=attr_contract['oil_ftp_is_available'],
            oil_ftp_is_shared=attr_contract['oil_ftp_is_shared'],
            oil_ftp_portion=attr_contract['oil_ftp_portion'],
            gas_ftp_is_available=attr_contract['gas_ftp_is_available'],
            gas_ftp_is_shared=attr_contract['gas_ftp_is_shared'],
            gas_ftp_portion=attr_contract['gas_ftp_portion'],
            tax_split_type=attr_contract['tax_split_type'],
            condition_dict=attr_contract['condition_dict'],
            indicator_rc_icp_sliding=attr_contract['indicator_rc_icp_sliding'],
            oil_ctr_pretax_share=attr_contract['oil_ctr_pretax_share'],
            gas_ctr_pretax_share=attr_contract['gas_ctr_pretax_share'],
            oil_ic_rate=attr_contract['oil_ic_rate'],
            gas_ic_rate=attr_contract['gas_ic_rate'],
            ic_is_available=attr_contract['ic_is_available'],
            oil_cr_cap_rate=attr_contract['oil_cr_cap_rate'],
            gas_cr_cap_rate=attr_contract['gas_cr_cap_rate'],
            oil_dmo_volume_portion=attr_contract['oil_dmo_volume_portion'],
            oil_dmo_fee_portion=attr_contract['oil_dmo_fee_portion'],
            oil_dmo_holiday_duration=attr_contract['oil_dmo_holiday_duration'],
            gas_dmo_volume_portion=attr_contract['gas_dmo_volume_portion'],
            gas_dmo_fee_portion=attr_contract['gas_dmo_fee_portion'],
            gas_dmo_holiday_duration=attr_contract['gas_dmo_holiday_duration'])

    # Running the new contract
    new_contract.run(**contract_arguments)
    summary_arguments['contract'] = new_contract
    contract_summary = get_summary(**summary_arguments)

    return contract_summary, new_contract


def sensitivity_psc(steps: int,
                    diff: float,
                    contract: CostRecovery | GrossSplit,
                    contract_arguments: dict,
                    summary_arguments: dict
                    ):
    # Defining the array named steps_arr that containing swings factor of the sens
    increment = diff / steps
    right_steps = np.arange(1 + increment, 1 + diff + increment, increment)
    left_steps = np.arange(1 - diff, 1, increment)
    steps_arr = np.concatenate((left_steps, np.array([1]), right_steps))

    # Summary attributes that need to be recorded are NPV, IRR, P/I, Government Take, Contractor Take

    # Defining the list to contain the result of the contract variance and running the contract along the steps_arr
    result_npv = np.empty(len(steps_arr))
    result_irr = np.empty(len(steps_arr))
    result_pi = np.empty(len(steps_arr))
    result_gov_take = np.empty(len(steps_arr))
    result_ctr_take = np.empty(len(steps_arr))

    for index, mul_factor in enumerate(steps_arr):
        # Adjusting the contract
        result_psc, contract_new = adjust_contract(contract=contract,
                                                   adjusted_variable='OPEX',
                                                   multiply_factor=mul_factor,
                                                   contract_arguments=contract_arguments,
                                                   summary_arguments=summary_arguments)

        # Filling the summary result to result list
        result_npv[index] = result_psc['ctr_npv']
        result_irr[index] = result_psc['ctr_irr']
        result_pi[index] = result_psc['ctr_pi']
        result_gov_take[index] = result_psc['gov_take']
        result_ctr_take[index] = result_psc['ctr_net_share']

        dict_var_result = {'ctr_npv': result_npv,
                           'ctr_irr': result_irr,
                           'ctr_pi': result_pi,
                           'gov_take': result_gov_take,
                           'ctr_net_share': result_ctr_take,
                           }

    df = pd.DataFrame()
    df['Steps'] = steps_arr
    df['result_npv'] = result_npv
    df['result_irr'] = result_irr
    df['result_pi'] = result_pi
    df['result_gov_take'] = result_gov_take
    df['result_ctr_take'] = result_ctr_take
    print(df)

    return NotImplemented
