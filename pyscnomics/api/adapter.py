"""
This file is utilized to be the adapter of the router into the core codes.
"""
import pandas as pd
from datetime import datetime

from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts.transition import Transition
from pyscnomics.tools.summary import get_summary
from pyscnomics.tools.table import get_table
from pyscnomics.optimize.optimization import optimize_psc
from pyscnomics.api.converter import (convert_str_to_date,
                                      convert_list_to_array_float_or_array,
                                      convert_dict_to_lifting,
                                      convert_dict_to_tangible,
                                      convert_dict_to_intangible,
                                      convert_dict_to_opex,
                                      convert_dict_to_asr,
                                      convert_list_to_array_float,
                                      convert_list_to_array_float_or_array_or_none,
                                      convert_str_to_taxsplit,
                                      convert_str_to_npvmode,
                                      convert_str_to_discountingmode,
                                      convert_str_to_otherrevenue,
                                      convert_str_to_taxregime,
                                      convert_str_to_ftptaxregime,
                                      convert_str_to_depremethod,
                                      convert_str_to_inflationappliedto,
                                      convert_summary_to_dict,
                                      convert_str_to_optimization_parameters,
                                      convert_str_to_optimization_targetparameter)


class ContractException(Exception):
    """Exception to be raised if contract type is misused"""

    pass


def get_setup_dict(data: dict) -> tuple:
    """
    Function to get conversion of the setup input from dictionary into acceptable core engine data format.

    Parameters
    ----------
    data: dict
        The dictionary of the data input

    Returns
    -------
    start_date: date
        The start date of the project.
    end_date: date
        The end date of the project.
    oil_onstream_date: date
        The oil onstream date.
    gas_onstream_date: date
        The gas onstream date.
    lifting: Lifting
        The lifting of the project, in Lifting Dataclass format.
    tangible: Tangible
        The tangible cost of the project, in Tangible Dataclass format.
    intangible: Intangible
        The intangible cost of the project, in Intangible Dataclass format.
    opex: OPEX
        The opex cost of the project, in OPEX Dataclass format.
    asr: ASR
        The asr cost of the project, in ASR Dataclass format.

    """
    # Parsing the contract setup into each corresponding variables
    start_date = convert_str_to_date(str_object=data['setup']['start_date'])
    end_date = convert_str_to_date(str_object=data['setup']['end_date'])
    oil_onstream_date = convert_str_to_date(str_object=data['setup']['oil_onstream_date'])
    gas_onstream_date = convert_str_to_date(str_object=data['setup']['gas_onstream_date'])
    lifting = convert_dict_to_lifting(data_raw=data)
    tangible = convert_dict_to_tangible(data_raw=data['tangible'])
    intangible = convert_dict_to_intangible(data_raw=data['intangible'])
    opex = convert_dict_to_opex(data_raw=data['opex'])
    asr = convert_dict_to_asr(data_raw=data['asr'])
    return start_date, end_date, oil_onstream_date, gas_onstream_date, lifting, tangible, intangible, opex, asr


def get_summary_dict(data: dict) -> dict:
    """
    Function to get the summary arguments from the dictionary data input.
    Parameters
    ----------
    data: dict
        The dictionary of the data input

    Returns
    -------
    summary_arguments_dict: dict
        The summary argument in the core engine acceptable format.



    """
    # Filling the argument with the input data
    reference_year = data['summary_arguments']['reference_year']
    inflation_rate = data['summary_arguments']['inflation_rate']
    discount_rate = data['summary_arguments']['discount_rate']
    npv_mode = convert_str_to_npvmode(str_object=data['summary_arguments']['npv_mode'])
    discounting_mode = convert_str_to_discountingmode(str_object=data['summary_arguments']['discounting_mode'])

    summary_arguments_dict = {
        'reference_year': reference_year,
        'inflation_rate': inflation_rate,
        'discount_rate': discount_rate,
        'npv_mode': npv_mode,
        'discounting_mode': discounting_mode,
    }

    return summary_arguments_dict


def get_summary_object(data: dict, contract: CostRecovery | GrossSplit | Transition) -> (dict, dict):
    """
    The function to get the summary dictionary object from the data and contract input.

    Parameters
    ----------
    data: dict
        The dictionary of the data input
    contract: CostRecovery | GrossSplit | Transition
        The contract object.

    Returns
    -------
    summary: dict
        The summary of the contract ini dictionary format.

    summary_arguments_dict: dict
        The summary arguments used in retrieving the summary of the contract.

    """
    if contract is Transition:
        summary_arguments_dict = get_summary_dict(data=data)
        summary_arguments_dict['contract'] = contract
        summary = get_summary(**summary_arguments_dict)

    else:
        summary_arguments_dict = get_summary_dict(data=data)
        summary_arguments_dict['contract'] = contract
        summary = get_summary(**summary_arguments_dict)

    return summary, summary_arguments_dict


def get_costrecovery(data: dict, summary_result: bool = True):
    """
    The function to get the Summary, Cost Recovery object, contract arguments, and summary arguments used.

    Parameters
    ----------
    data: dict
        The dictionary of the data input.
    summary_result: bool
        The condition if the summary result will be generated or not.

    Returns
    -------
    summary_skk: dict
        The executive summary of the contract.
    contract:
        The Cost Recovery contract object.
    contract_arguments_dict: dict
        The contract arguments used in running the contract calculation.
    summary_arguments_dic: dict
        The summary arguments used in retrieving the executive summary of the contract.

    """
    start_date, end_date, oil_onstream_date, gas_onstream_date, lifting, tangible, intangible, opex, asr = (
        get_setup_dict(data=data))

    contract = CostRecovery(
        start_date=start_date,
        end_date=end_date,
        oil_onstream_date=oil_onstream_date,
        gas_onstream_date=gas_onstream_date,
        lifting=lifting,
        tangible_cost=tangible,
        intangible_cost=intangible,
        opex=opex,
        asr_cost=asr,
        oil_ftp_is_available=data['costrecovery']['oil_ftp_is_available'],
        oil_ftp_is_shared=data['costrecovery']['oil_ftp_is_shared'],
        oil_ftp_portion=data['costrecovery']['oil_ftp_portion'],
        gas_ftp_is_available=data['costrecovery']['gas_ftp_is_available'],
        gas_ftp_is_shared=data['costrecovery']['gas_ftp_is_shared'],
        gas_ftp_portion=data['costrecovery']['gas_ftp_portion'],
        tax_split_type=convert_str_to_taxsplit(str_object=data['costrecovery']['tax_split_type']),
        condition_dict=data['costrecovery']['condition_dict'],
        indicator_rc_icp_sliding=convert_list_to_array_float(
            data_list=data['costrecovery']['indicator_rc_icp_sliding']),
        oil_ctr_pretax_share=data['costrecovery']['oil_ctr_pretax_share'],
        gas_ctr_pretax_share=data['costrecovery']['gas_ctr_pretax_share'],
        oil_ic_rate=data['costrecovery']['oil_ic_rate'],
        gas_ic_rate=data['costrecovery']['gas_ic_rate'],
        ic_is_available=data['costrecovery']['ic_is_available'],
        oil_cr_cap_rate=data['costrecovery']['oil_cr_cap_rate'],
        gas_cr_cap_rate=data['costrecovery']['gas_cr_cap_rate'],
        oil_dmo_volume_portion=data['costrecovery']['oil_dmo_volume_portion'],
        oil_dmo_fee_portion=data['costrecovery']['oil_dmo_fee_portion'],
        oil_dmo_holiday_duration=data['costrecovery']['oil_dmo_holiday_duration'],
        gas_dmo_volume_portion=data['costrecovery']['gas_dmo_volume_portion'],
        gas_dmo_fee_portion=data['costrecovery']['gas_dmo_fee_portion'],
        gas_dmo_holiday_duration=data['costrecovery']['gas_dmo_holiday_duration'],
    )

    # Filling the arguments of the contract with the data input
    contract_arguments_dict = {
        "sulfur_revenue": convert_str_to_otherrevenue(str_object=data['contract_arguments']['sulfur_revenue']),
        "electricity_revenue": convert_str_to_otherrevenue(
            str_object=data['contract_arguments']['electricity_revenue']),
        "co2_revenue": convert_str_to_otherrevenue(str_object=data['contract_arguments']['co2_revenue']),
        "is_dmo_end_weighted": data['contract_arguments']['is_dmo_end_weighted'],
        "tax_regime": convert_str_to_taxregime(str_object=data['contract_arguments']['tax_regime']),
        "tax_rate": convert_list_to_array_float_or_array_or_none(data_list=data['contract_arguments']['tax_rate']),
        "ftp_tax_regime": convert_str_to_ftptaxregime(str_object=data['contract_arguments']['ftp_tax_regime']),
        "sunk_cost_reference_year": data['contract_arguments']['sunk_cost_reference_year'],
        "depr_method": convert_str_to_depremethod(str_object=data['contract_arguments']['depr_method']),
        "decline_factor": data['contract_arguments']['decline_factor'],
        "vat_rate": convert_list_to_array_float_or_array(data_input=data['contract_arguments']['vat_rate']),
        "lbt_rate": convert_list_to_array_float_or_array(data_input=data['contract_arguments']['lbt_rate']),
        "inflation_rate": convert_list_to_array_float_or_array(data_input=data['contract_arguments']['inflation_rate']),
        "future_rate": data['contract_arguments']['future_rate'],
        "inflation_rate_applied_to": convert_str_to_inflationappliedto(str_object=data['contract_arguments']['inflation_rate_applied_to']),
        "post_uu_22_year2001": data['contract_arguments']['post_uu_22_year2001']
    }

    # Running the contract
    contract.run(**contract_arguments_dict)

    # Condition when summary is needed
    if summary_result is True:
        # Filling the summary arguments
        summary_arguments_dict = get_summary_dict(data=data)
        summary_arguments_dict['contract'] = contract
        summary = get_summary(**summary_arguments_dict)

        # Converting the summary format to skk summary format
        summary_skk = convert_summary_to_dict(dict_object=summary)

        # Adding the execution info
        summary_skk = add_execution_info(data=summary_skk)

    # Since the required object is only the contract, it will return None for the summary
    else:
        summary_skk = None
        summary_arguments_dict = None

    return summary_skk, contract, contract_arguments_dict, summary_arguments_dict


def get_contract_table(data: dict, contract_type: str = 'Cost Recovery') -> dict:
    """
    Function to get the cash flow table of the contract that has been run.

    Parameters
    ----------
    data: dict
        The dictionary of the data input.
    contract_type: str
        The option for the contract type. The available option are: ['Cost Recovery', 'Gross Split']

    Returns
    -------
    table_all_dict: dict
        The dictionary containing the oil, gas and consolidated cash flow table.
    """
    # Adjusting the variable to the corresponding contract type
    if contract_type == 'Cost Recovery':
        contract = get_costrecovery(data=data)[1]
        year_column = 'Year'

    elif contract_type == 'Gross Split':
        contract = get_grosssplit(data=data)[1]
        year_column = 'Years'

    else:
        contract = get_transition(data=data)[1]
        year_column = 'Year'

    # Condition when the contract is Transition
    if contract_type == 'Transition':
        # Retrieving the table
        table_oil, table_gas, table_consolidated = get_table(contract=contract)

        # Forming the table dictionary as the output
        table_all_dict = {
            'contract_1': {
                'oil': table_oil[0].set_index(table_oil[0].columns[0]).to_dict(),
                'gas': table_gas[0].set_index(table_gas[0].columns[0]).to_dict(),
                'consolidated': table_consolidated[0].set_index(table_consolidated[0].columns[0]).to_dict()
            },

            'contract_2': {
                'oil': table_oil[1].set_index(table_oil[1].columns[0]).to_dict(),
                'gas': table_gas[1].set_index(table_gas[1].columns[0]).to_dict(),
                'consolidated': table_consolidated[1].set_index(table_consolidated[1].columns[0]).to_dict()
            }
        }

    else:
        # Retrieving the table
        table_oil, table_gas, table_consolidated = get_table(contract=contract)

        # Forming the table dictionary as the output
        table_all_dict = {'oil': table_oil.set_index(year_column).to_dict(),
                          'gas': table_gas.set_index(year_column).to_dict(),
                          'consolidated': table_consolidated.set_index(year_column).to_dict()}

    # Adding the execution info
    table_all_dict = add_execution_info(data=table_all_dict)

    return table_all_dict


def get_contract_optimization(data: dict, contract_type: str = 'Cost Recovery') -> dict:
    """
    The function to run contract optimization. Resulting optimization result in dictionary format.

    Parameters
    ----------
    data: dict
        The dictionary of the data input.
    contract_type: str
        The option for the contract type. The available option are: ['Cost Recovery', 'Gross Split']

    Returns
    -------
    result_parameters: dict
        The result of the optimization in dictionary format

    """
    # Converting the parameters in dict_optimization to the corresponding enum
    optimization_parameters = [
        convert_str_to_optimization_parameters(str_object=i)
        for i in data['optimization_arguments']['dict_optimization']['parameter']
    ]

    # Generating the dictionary of the optimization arguments
    dict_optimization = {
        'parameter': optimization_parameters,
        'min': convert_list_to_array_float(data_list=data['optimization_arguments']['dict_optimization']['min']),
        'max': convert_list_to_array_float(data_list=data['optimization_arguments']['dict_optimization']['max']),
    }

    # Filling the optimization arguments with target_optimization and target_parameter,
    target_optimization_value = data['optimization_arguments']['target_optimization']
    target_parameter = convert_str_to_optimization_targetparameter(
        str_object=data['optimization_arguments']['target_parameter'])

    # Retrieving the contract, contract_arguments_dict, summary_arguments_dict based on the contract type
    if contract_type == 'Cost Recovery':
        contract = get_costrecovery(data=data)[1]
        contract_arguments = get_costrecovery(data=data)[2]
        summary_argument = get_costrecovery(data=data)[3]

    elif contract_type == 'Gross Split':
        contract = get_grosssplit(data=data)[1]
        contract_arguments = get_grosssplit(data=data)[2]
        summary_argument = get_grosssplit(data=data)[3]

    else:
        contract = NotImplemented
        contract_arguments = NotImplemented
        summary_argument = NotImplemented

    list_str, list_params_value, result_optimization, list_executed_contract = optimize_psc(
        dict_optimization=dict_optimization,
        contract=contract,
        contract_arguments=contract_arguments,
        target_optimization_value=target_optimization_value,
        summary_argument=summary_argument,
        target_parameter=target_parameter,
    )

    # Forming the optimization result into a dictionary object
    optimization_result = {
        'list_str': list_str,
        'list_params_value': list_params_value,
    }

    # Converting the result into dictionary format
    result_parameters = pd.DataFrame(optimization_result).set_index('list_str').to_dict()
    result_parameters['optimization_result'] = result_optimization

    # Adding the execution info
    result_parameters = add_execution_info(data=result_parameters)

    return result_parameters


def get_grosssplit(data: dict, summary_result: bool = True):
    """
    The function to get the Summary, Gross Split object, contract arguments, and summary arguments used.

    Parameters
    ----------
    data: dict
        The dictionary of the data input.
    summary_result: bool
        The condition if the summary result will be generated or not.

    Returns
    -------
    summary_skk: dict
        The executive summary of the contract.
    contract:
        The Gross Split contract object.
    contract_arguments_dict: dict
        The contract arguments used in running the contract calculation.
    summary_arguments_dic: dict
        The summary arguments used in retrieving the executive summary of the contract.

    """
    start_date, end_date, oil_onstream_date, gas_onstream_date, lifting, tangible, intangible, opex, asr = (
        get_setup_dict(data=data))

    contract = GrossSplit(
        start_date=start_date,
        end_date=end_date,
        oil_onstream_date=oil_onstream_date,
        gas_onstream_date=gas_onstream_date,
        lifting=lifting,
        tangible_cost=tangible,
        intangible_cost=intangible,
        opex=opex,
        asr_cost=asr,
        field_status=data['grosssplit']['field_status'],
        field_loc=data['grosssplit']['field_loc'],
        res_depth=data['grosssplit']['res_depth'],
        infra_avail=data['grosssplit']['infra_avail'],
        res_type=data['grosssplit']['res_type'],
        api_oil=data['grosssplit']['api_oil'],
        domestic_use=data['grosssplit']['domestic_use'],
        prod_stage=data['grosssplit']['prod_stage'],
        co2_content=data['grosssplit']['co2_content'],
        h2s_content=data['grosssplit']['h2s_content'],
        base_split_ctr_oil=data['grosssplit']['base_split_ctr_oil'],
        base_split_ctr_gas=data['grosssplit']['base_split_ctr_gas'],
        split_ministry_disc=data['grosssplit']['split_ministry_disc'],
        oil_dmo_volume_portion=data['grosssplit']['oil_dmo_volume_portion'],
        oil_dmo_fee_portion=data['grosssplit']['oil_dmo_fee_portion'],
        oil_dmo_holiday_duration=data['grosssplit']['oil_dmo_holiday_duration'],
        gas_dmo_volume_portion=data['grosssplit']['gas_dmo_volume_portion'],
        gas_dmo_fee_portion=data['grosssplit']['gas_dmo_fee_portion'],
        gas_dmo_holiday_duration=data['grosssplit']['gas_dmo_holiday_duration'],

    )

    # Filling the arguments of the contract with the data input
    contract_arguments_dict = {
        "sulfur_revenue": convert_str_to_otherrevenue(str_object=data['contract_arguments']['sulfur_revenue']),
        "electricity_revenue": convert_str_to_otherrevenue(
            str_object=data['contract_arguments']['electricity_revenue']),
        "co2_revenue": convert_str_to_otherrevenue(str_object=data['contract_arguments']['co2_revenue']),
        "is_dmo_end_weighted": data['contract_arguments']['is_dmo_end_weighted'],
        "tax_regime": convert_str_to_taxregime(str_object=data['contract_arguments']['tax_regime']),
        "tax_rate": convert_list_to_array_float_or_array_or_none(data_list=data['contract_arguments']['tax_rate']),
        # "regime": convert_str_to_ftptaxregime(str_object=data['contract_arguments']['ftp_tax_regime']),
        "sunk_cost_reference_year": data['contract_arguments']['sunk_cost_reference_year'],
        "depr_method": convert_str_to_depremethod(str_object=data['contract_arguments']['depr_method']),
        "decline_factor": data['contract_arguments']['decline_factor'],
        "vat_rate": convert_list_to_array_float_or_array(data_input=data['contract_arguments']['vat_rate']),
        "lbt_rate": convert_list_to_array_float_or_array(data_input=data['contract_arguments']['lbt_rate']),
        "inflation_rate": convert_list_to_array_float_or_array(data_input=data['contract_arguments']['inflation_rate']),
        "future_rate": data['contract_arguments']['future_rate'],
        "inflation_rate_applied_to": convert_str_to_inflationappliedto(
            str_object=data['contract_arguments']['inflation_rate_applied_to']),
        "cum_production_split_offset": convert_list_to_array_float_or_array(data_input=data["contract_arguments"]["cum_production_split_offset"]),
        "amortization": data["contract_arguments"]["amortization"]
    }

    # Running the contract
    contract.run(**contract_arguments_dict)

    # Condition when summary is needed
    if summary_result is True:
        # Filling the summary arguments
        summary_arguments_dict = get_summary_dict(data=data)
        summary_arguments_dict['contract'] = contract
        summary = get_summary(**summary_arguments_dict)

        # Converting the summary format to skk summary format
        summary_skk = convert_summary_to_dict(dict_object=summary)

        # Adding the execution info
        summary_skk = add_execution_info(data=summary_skk)

    else:
        summary_skk = None
        summary_arguments_dict = None

    return summary_skk, contract, contract_arguments_dict, summary_arguments_dict


def get_transition(data: dict):
    """
    The function to get the Summary, Transition object, contract arguments, and summary arguments used.

    Parameters
    ----------
    data: dict
        The dictionary of the data input.

    Returns
    -------
    summary_skk: dict
        The executive summary of the contract.
    contract:
        The Transition contract object.
    contract_arguments_dict: dict
        The contract arguments used in running the contract calculation.
    summary_arguments_dic: dict
        The summary arguments used in retrieving the executive summary of the contract.

    """
    # Defining contract_1
    if data['contract_1']['costrecovery'] is not None and data['contract_1']['grosssplit'] is None:
        _, contract_1, contract_arguments_1, _ = get_costrecovery(data=data['contract_1'], summary_result=False)

    elif data['contract_1']['grosssplit'] is not None and data['contract_1']['costrecovery'] is None:
        _, contract_1, contract_arguments_1, _ = get_grosssplit(data=data['contract_1'], summary_result=False)

    else:
        raise ContractException("Contract type is not recognized")

    # Defining contract_2
    if data['contract_2']['costrecovery'] is not None and data['contract_2']['grosssplit'] is None:
        _, contract_2, contract_arguments_2, _ = get_costrecovery(data=data['contract_2'], summary_result=False)

    elif data['contract_2']['grosssplit'] is not None and data['contract_2']['costrecovery'] is None:
        _, contract_2, contract_arguments_2, _ = get_grosssplit(data=data['contract_2'], summary_result=False)

    else:
        raise ContractException("Contract type is not recognized")

    # generating the transition contract object
    contract = Transition(contract1=contract_1,
                          contract2=contract_2,
                          argument_contract1=contract_arguments_1,
                          argument_contract2=contract_arguments_2, )

    # Generating the transition contract arguments
    contract_arguments_dict = data['contract_arguments']

    # Running the transition contract
    contract.run(**contract_arguments_dict)

    # Filling the summary arguments
    summary_arguments_dict = get_summary_dict(data=data)
    summary_arguments_dict['contract'] = contract

    summary = get_summary(**summary_arguments_dict)

    # Converting the summary format to skk summary format
    summary_skk = convert_summary_to_dict(dict_object=summary)

    # Adding the execution info
    summary_skk = add_execution_info(data=summary_skk)

    return summary_skk, contract, contract_arguments_dict, summary_arguments_dict


def add_execution_info(data: dict) -> dict:
    """
    Function to adding the execution info into a dictionary.

    Parameters
    ----------
    data: dict
        The dictionary which will added with execution info

    Returns
    -------
    out: dict
        Dictionary containing the execution info
    """
    # Defining the execution date
    execution_date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    # Defining the PySCnomics Version
    package_version = " "
    try:
        from importlib.metadata import version
        package_version = version('pyscnomics')

    except:
        pass

    # Parsing the data into the data output
    execution_info = {'execution_datetime': execution_date,
                      'package_version': package_version, }

    data['execution_info'] = execution_info
    return data


def get_detailed_summary(data: dict, contract_type: str):

    if contract_type == 'Cost Recovery':
        summary_args = get_costrecovery(data=data, summary_result=True)[3]

    elif contract_type == 'Gross Split':
        summary_args = get_grosssplit(data=data, summary_result=True)[3]

    elif contract_type == 'Transition':
        summary_args = get_transition(data=data)[3]

    else:
        summary_args = None

    return get_summary(**summary_args)








