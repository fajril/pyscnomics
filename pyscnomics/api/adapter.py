"""
This file is utilized to be the adapter of the router into the core codes.
"""
import numpy as np
import pandas as pd
from datetime import datetime

from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts.transition import Transition
from pyscnomics.optimize import sensitivity_psc, uncertainty_psc
from pyscnomics.tools.summary import get_summary
from pyscnomics.tools.table import get_table
from pyscnomics.optimize.optimization import optimize_psc
from pyscnomics.optimize.optimization_transition import optimize_psc_core as optimize_psc_trans
from pyscnomics.econ.selection import OptimizationParameter, FluidType
from pyscnomics.tools.ltp import oil_ltp_predict, gas_ltp_predict
from pyscnomics.tools.rpd import RPDModel
from pyscnomics.api.converter import (convert_str_to_date,
                                      convert_list_to_array_float_or_array,
                                      convert_dict_to_lifting,
                                      convert_dict_to_capital,
                                      convert_dict_to_intangible,
                                      convert_dict_to_opex,
                                      convert_dict_to_asr,
                                      convert_dict_to_lbt,
                                      convert_dict_to_cost_of_sales,
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
                                      convert_str_to_optimization_targetparameter,
                                      convert_grosssplitregime_to_enum,
                                      convert_to_float,
                                      read_fluid_type,
                                      convert_to_method_limit,
                                      convert_to_uncertainty_distribution)
from pyscnomics.econ.limit import econ_limit


class ContractException(Exception):
    """Exception to be raised if contract type is misused"""

    pass


class LTPModelException(Exception):
    """ Exception to be raised for an incorrect LTP configurations """

    pass


class RDPModelException(Exception):
    """ Exception to be raised for an incorrect RDP configurations """

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
    capital: Tangible
        The capital cost of the project, in Tangible Dataclass format.
    intangible: Intangible
        The intangible cost of the project, in Intangible Dataclass format.
    opex: OPEX
        The opex cost of the project, in OPEX Dataclass format.
    lbt: LBT
        The land and building tax of the project, in LBT Dataclass format.
    cost_of_sales: CostOfSales
        The opex cost of the project, in CostOfSales Dataclass format.
    asr: ASR
        The asr cost of the project, in ASR Dataclass format.

    """
    # Parsing the contract setup into each corresponding variables
    start_date = convert_str_to_date(str_object=data['setup']['start_date'])
    end_date = convert_str_to_date(str_object=data['setup']['end_date'])
    oil_onstream_date = convert_str_to_date(str_object=data['setup']['oil_onstream_date'])
    gas_onstream_date = convert_str_to_date(str_object=data['setup']['gas_onstream_date'])
    lifting = convert_dict_to_lifting(data_raw=data)
    capital = convert_dict_to_capital(data_raw=data['capital'])
    intangible = convert_dict_to_intangible(data_raw=data['intangible'])
    opex = convert_dict_to_opex(data_raw=data['opex'])
    asr = convert_dict_to_asr(data_raw=data['asr'])
    lbt = convert_dict_to_lbt(data_raw=data['lbt'])
    cost_of_sales = convert_dict_to_cost_of_sales(data_raw=data['cost_of_sales'])
    return start_date, end_date, oil_onstream_date, gas_onstream_date, lifting, capital, intangible, opex, asr, lbt, cost_of_sales


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
    reference_year = data['summary_arguments'].get('reference_year', None)
    inflation_rate = data['summary_arguments'].get('inflation_rate', None)
    discount_rate = data['summary_arguments'].get('discount_rate', 0.1)
    npv_mode = convert_str_to_npvmode(str_object=data['summary_arguments'].get('npv_mode', "Full Cycle Nominal Terms"))
    discounting_mode = convert_str_to_discountingmode(str_object=data['summary_arguments'].get('discounting_mode', 'discounting_mode'))
    profitability_discounted = data['summary_arguments'].get('profitability_discounted', False)

    summary_arguments_dict = {
        'reference_year': reference_year,
        'inflation_rate': inflation_rate,
        'discount_rate': discount_rate,
        'npv_mode': npv_mode,
        'discounting_mode': discounting_mode,
        'profitability_discounted': profitability_discounted
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
    start_date, end_date, oil_onstream_date, gas_onstream_date, lifting, tangible, intangible, opex, asr, lbt, cost_of_sales = get_setup_dict(data=data)

    contract = CostRecovery(
        start_date=start_date,
        end_date=end_date,
        oil_onstream_date=oil_onstream_date,
        gas_onstream_date=gas_onstream_date,
        lifting=lifting,
        capital_cost=tangible,
        intangible_cost=intangible,
        opex=opex,
        asr_cost=asr,
        lbt_cost=lbt,
        cost_of_sales=cost_of_sales,
        oil_ftp_is_available=data['costrecovery']['oil_ftp_is_available'],
        oil_ftp_is_shared=data['costrecovery']['oil_ftp_is_shared'],
        oil_ftp_portion=convert_to_float(target=data['costrecovery']['oil_ftp_portion']),
        gas_ftp_is_available=data['costrecovery']['gas_ftp_is_available'],
        gas_ftp_is_shared=data['costrecovery']['gas_ftp_is_shared'],
        gas_ftp_portion=convert_to_float(target=data['costrecovery']['gas_ftp_portion']),
        tax_split_type=convert_str_to_taxsplit(str_object=data['costrecovery']['tax_split_type']),
        condition_dict=data['costrecovery']['condition_dict'],
        indicator_rc_icp_sliding=convert_list_to_array_float(
            data_list=data['costrecovery']['indicator_rc_icp_sliding']),
        oil_ctr_pretax_share=convert_to_float(target=data['costrecovery']['oil_ctr_pretax_share']),
        gas_ctr_pretax_share=convert_to_float(target=data['costrecovery']['gas_ctr_pretax_share']),
        oil_ic_rate=convert_to_float(target=data['costrecovery']['oil_ic_rate']),
        gas_ic_rate=convert_to_float(target=data['costrecovery']['gas_ic_rate']),
        ic_is_available=data['costrecovery']['ic_is_available'],
        oil_cr_cap_rate=convert_to_float(target=data['costrecovery']['oil_cr_cap_rate']),
        gas_cr_cap_rate=convert_to_float(target=data['costrecovery']['gas_cr_cap_rate']),
        oil_dmo_volume_portion=convert_to_float(target=data['costrecovery']['oil_dmo_volume_portion']),
        oil_dmo_fee_portion=convert_to_float(target=data['costrecovery']['oil_dmo_fee_portion']),
        oil_dmo_holiday_duration=data['costrecovery']['oil_dmo_holiday_duration'],
        gas_dmo_volume_portion=convert_to_float(target=data['costrecovery']['gas_dmo_volume_portion']),
        gas_dmo_fee_portion=convert_to_float(target=data['costrecovery']['gas_dmo_fee_portion']),
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
        "effective_tax_rate": convert_list_to_array_float_or_array_or_none(data_list=data['contract_arguments']['effective_tax_rate']),
        "ftp_tax_regime": convert_str_to_ftptaxregime(str_object=data['contract_arguments']['ftp_tax_regime']),
        "sunk_cost_reference_year": data['contract_arguments']['sunk_cost_reference_year'],
        "depr_method": convert_str_to_depremethod(str_object=data['contract_arguments']['depr_method']),
        "decline_factor": data['contract_arguments']['decline_factor'],
        "vat_rate": convert_list_to_array_float_or_array(data_input=data['contract_arguments']['vat_rate']),
        "inflation_rate": convert_list_to_array_float_or_array(data_input=data['contract_arguments']['inflation_rate']),
        "inflation_rate_applied_to": convert_str_to_inflationappliedto(str_object=data['contract_arguments']['inflation_rate_applied_to']),
        "post_uu_22_year2001": True if 'post_uu_22_year2001' not in data['contract_arguments'] else
        data['contract_arguments']['post_uu_22_year2001'],
        "oil_cost_of_sales_applied": False if "oil_cost_of_sales_applied" not in data["contract_arguments"] else
        data["contract_arguments"]["oil_cost_of_sales_applied"],
        "gas_cost_of_sales_applied": False if "gas_cost_of_sales_applied" not in data["contract_arguments"] else
        data["contract_arguments"]["gas_cost_of_sales_applied"],
        "sum_undepreciated_cost": False if 'sum_undepreciated_cost' not in data['contract_arguments'] else
        data['contract_arguments']['sum_undepreciated_cost'],
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

    elif contract_type == 'Base Project':
        contract = get_baseproject(data=data)[1]
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
    if 'optimization_arguments' not in data:
        raise ContractException("The payload does not have the optimization_arguments key")

    if data['optimization_arguments'] is None:
        raise ContractException("The payload optimization_arguments does not have any values")

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

    elif contract_type == 'Transition':
        contract = get_transition(data=data)[1]
        contract_arguments = get_transition(data=data)[2]
        summary_argument = get_transition(data=data)[3]

    else:
        contract = NotImplemented
        contract_arguments = NotImplemented
        summary_argument = NotImplemented

    if contract_type == 'Transition':
        # Retrieve the original useful life of the capital cost
        useful_life_original = contract.contract2.capital_cost_total.useful_life.tolist()

        list_str, list_params_value, result_optimization, list_executed_contract = optimize_psc_trans(
            dict_optimization=dict_optimization,
            contract=contract,
            contract_arguments=contract_arguments,
            target_optimization_value=target_optimization_value,
            summary_argument=summary_argument,
            target_parameter=target_parameter,
        )

    else:
        # Retrieve the original useful life of the capital cost
        useful_life_original = contract.capital_cost_total.useful_life.tolist()
        list_str, list_params_value, result_optimization, list_executed_contract = optimize_psc(
            dict_optimization=dict_optimization,
            contract=contract,
            contract_arguments=contract_arguments,
            target_optimization_value=target_optimization_value,
            summary_argument=summary_argument,
            target_parameter=target_parameter,
        )

    # Treatment to add the useful life of optimization into the result
    def get_enum_index(enum_list: list, element: any) -> int | None:
        """
        Function to get the index of the OptimizationParameter.DEPRECIATION_ACCELERATION.

        Parameters
        ----------
        enum_list: list
            The source of list.
        element: any
            The corresponding element
        Returns
        -------
        out : int | None
        """
        try:
            return enum_list.index(element)
        except ValueError:
            return None

    # Get the index of the depreciation optimization parameter
    index_depreciation = get_enum_index(
        enum_list=optimization_parameters,
        element=OptimizationParameter.DEPRECIATION_ACCELERATION)

    # Adding condition of the contract type for retrieving the optimized contract
    if contract_type == 'Transition':
        contract_optimized = list_executed_contract[-1].contract2
    else:
        contract_optimized = list_executed_contract[-1]

    # Adding the information of optimized useful life into the list_params_value
    if index_depreciation is not None:
        optimized_capital_cost = {
            "year": contract_optimized.capital_cost_total.expense_year.tolist(),
            "cost_allocation": contract_optimized.capital_cost_total.cost_allocation,
            "cost": contract_optimized.capital_cost_total.cost.tolist(),
            "pis_year": contract_optimized.capital_cost_total.pis_year.tolist(),
            "useful_life_original": useful_life_original,
            "useful_life_optimized": contract_optimized.capital_cost_total.useful_life.tolist(),
            "description": contract_optimized.capital_cost_total.description
        }

        # Adding optimized_capital_cost into the result of the optimization
        list_params_value[index_depreciation] = {
            "depreciation acceleration": list_params_value[index_depreciation],
            "optimized_useful_life": optimized_capital_cost
        }

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
    start_date, end_date, oil_onstream_date, gas_onstream_date, lifting, tangible, intangible, opex, asr, lbt, cost_of_sales = (
        get_setup_dict(data=data))

    contract = GrossSplit(
        start_date=start_date,
        end_date=end_date,
        oil_onstream_date=oil_onstream_date,
        gas_onstream_date=gas_onstream_date,
        lifting=lifting,
        capital_cost=tangible,
        intangible_cost=intangible,
        opex=opex,
        asr_cost=asr,
        lbt_cost=lbt,
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
        base_split_ctr_oil=convert_to_float(target=data['grosssplit']['base_split_ctr_oil']),
        base_split_ctr_gas=convert_to_float(target=data['grosssplit']['base_split_ctr_gas']),
        split_ministry_disc=convert_to_float(target=data['grosssplit']['split_ministry_disc']),
        oil_dmo_volume_portion=convert_to_float(target=data['grosssplit']['oil_dmo_volume_portion']),
        oil_dmo_fee_portion=convert_to_float(target=data['grosssplit']['oil_dmo_fee_portion']),
        oil_dmo_holiday_duration=data['grosssplit']['oil_dmo_holiday_duration'],
        gas_dmo_volume_portion=convert_to_float(target=data['grosssplit']['gas_dmo_volume_portion']),
        gas_dmo_fee_portion=convert_to_float(target=data['grosssplit']['gas_dmo_fee_portion']),
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
        "effective_tax_rate": convert_list_to_array_float_or_array_or_none(data_list=data['contract_arguments']['effective_tax_rate']),
        "sunk_cost_reference_year": data['contract_arguments']['sunk_cost_reference_year'],
        "depr_method": convert_str_to_depremethod(str_object=data['contract_arguments']['depr_method']),
        "decline_factor": data['contract_arguments']['decline_factor'],
        "vat_rate": convert_list_to_array_float_or_array(data_input=data['contract_arguments']['vat_rate']),
        "inflation_rate": convert_list_to_array_float_or_array(data_input=data['contract_arguments']['inflation_rate']),
        "inflation_rate_applied_to": convert_str_to_inflationappliedto(
            str_object=data['contract_arguments']['inflation_rate_applied_to']),
        "cum_production_split_offset": convert_list_to_array_float_or_array(data_input=data["contract_arguments"]["cum_production_split_offset"]),
        "amortization": data["contract_arguments"]["amortization"],
        "regime": convert_grosssplitregime_to_enum(target=data["contract_arguments"]["regime"]),
        "sum_undepreciated_cost": False if 'sum_undepreciated_cost' not in data['contract_arguments'] else
        data['contract_arguments']['sum_undepreciated_cost'],
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

    elif contract_type == 'Base Project':
        summary_args = get_baseproject(data=data, summary_result=True)[3]

    else:
        summary_args = None

    return get_summary(**summary_args)


def get_baseproject(data: dict, summary_result: bool = True):
    """
    The function to get the Summary, Base Project object, contract arguments, and summary arguments used.

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
        The Base Project contract object.
    contract_arguments_dict: dict
        The contract arguments used in running the contract calculation.
    summary_arguments_dic: dict
        The summary arguments used in retrieving the executive summary of the contract.

    """
    start_date, end_date, oil_onstream_date, gas_onstream_date, lifting, tangible, intangible, opex, asr, lbt, cost_of_sales = (
        get_setup_dict(data=data))

    contract = BaseProject(start_date=start_date,
                           end_date=end_date,
                           oil_onstream_date=oil_onstream_date,
                           gas_onstream_date=gas_onstream_date,
                           lifting=lifting,
                           capital_cost=tangible,
                           intangible_cost=intangible,
                           opex=opex,
                           asr_cost=asr,
                           lbt_cost=lbt)

    contract_arguments_dict = {
        "sulfur_revenue": convert_str_to_otherrevenue(str_object=data['contract_arguments']['sulfur_revenue']),
        "electricity_revenue": convert_str_to_otherrevenue(
            str_object=data['contract_arguments']['electricity_revenue']),
        "co2_revenue": convert_str_to_otherrevenue(str_object=data['contract_arguments']['co2_revenue']),
        "sunk_cost_reference_year": data['contract_arguments']['sunk_cost_reference_year'],
        "tax_rate": convert_list_to_array_float_or_array(data_input=data['contract_arguments']['vat_rate']),
        "inflation_rate": convert_list_to_array_float_or_array(data_input=data['contract_arguments']['inflation_rate']),
        "inflation_rate_applied_to": convert_str_to_inflationappliedto(str_object=data['contract_arguments']['inflation_rate_applied_to']),
    }

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


def get_ltp(
        volume: float,
        start_year: int,
        end_year: int,
        fluid_type: FluidType
) -> np.ndarray:
    """
    Function to get the ltp array.

    Parameters
    ----------
    volume:float
        The volume of the reserves.
    start_year:int
        The start year.
    end_year:int
        The end year.
    fluid_type: FluidType
        The FluidType of the corresponding volume.

    Returns
    -------
    out: np.ndarray
        The array of ltp

    """
    # Condition checking for the fluid type for initiating the array of ltp
    if fluid_type == FluidType.OIL:
        rate_ltp = oil_ltp_predict(volume=volume)
    elif fluid_type == FluidType.GAS:
        rate_ltp = gas_ltp_predict(volume=volume)
    else:
        raise LTPModelException(
            f"Unsupported Fluid Type {fluid_type} "
        )

    # Initiating the array of years
    year_arr = np.arange(start_year, end_year + 1)
    gap = len(year_arr) - len(rate_ltp)

    # Checking condition for the given years gap
    if gap < 0:
        raise LTPModelException(
            f"Forecast years from {start_year} to {end_year} is too short. "
            f"Please set the end_year at least until {end_year + (-1 * gap)}"
        )

    rate_gap = np.zeros(gap)
    rate_adjusted = np.concatenate([rate_gap, rate_ltp])

    return rate_adjusted


def get_ltp_dict(data: dict):
    """
    The function to get the list of LTP from the given reserves volume.

    Parameters
    ----------
    data: dict
        The dictionary of the data input.

    Returns
    -------
    """
    volume = data['volume']
    start_year = data['start_year']
    end_year = data['end_year']
    fluid_type = read_fluid_type(fluid=data['fluid_type'])

    ltp_array = get_ltp(
        volume=volume,
        start_year=start_year,
        end_year=end_year,
        fluid_type=fluid_type,)

    return pd.DataFrame(
        {
            'year': np.arange(start_year, end_year + 1).tolist(),
            'ltp': ltp_array.tolist()
        }
    ).set_index('year').to_dict()


def get_rdp(
        start_year: int,
        end_year: int,
        year_rampup: int,
        drate: float,
        q_plateau_ratio: float,
        q_min_ratio: float,
        volume: float
) -> np.ndarray:
    """
    The function to get the RDP array.

    Parameters
    ----------
    start_year
    end_year
    year_rampup
    drate
    q_plateau_ratio
    q_min_ratio
    volume

    Returns
    -------
    out: np.ndarray
        The array of rdp
    """

    # Initiating Model
    model = RPDModel(
        year_rampup=year_rampup,
        drate=drate,
        q_plateau_ratio=q_plateau_ratio,
        q_min_ratio=q_min_ratio,
    )

    rate_rdp = model.predict(volume)

    # Initiating the array of years
    year_arr = np.arange(start_year, end_year + 1)
    gap = len(year_arr) - len(rate_rdp)

    # Checking condition for the given years gap
    if gap < 0:
        raise RDPModelException(
            f"Forecast years from {start_year} to {end_year} is too short. "
            f"Please set the end_year at least until {end_year + (-1 * gap)}"
        )

    rate_gap = np.zeros(gap)
    rate_adjusted = np.concatenate([rate_gap, rate_rdp])

    return rate_adjusted


def get_rpd_dict(data: dict):
    """
    The function to get the list of RPD from the given reserves volume.

    Parameters
    ----------
    data: dict
        The dictionary of the data input.

    Returns
    -------
    """
    # Initiating the data input
    year_rampup = data['year_rampup']
    drate = data['drate']
    q_plateau_ratio = data['q_plateau_ratio']
    q_min_ratio = data['q_min_ratio']
    volume = data['volume']
    start_year = data['start_year']
    end_year = data['end_year']

    # Get the array of rdp
    rdp_array = get_rdp(
        start_year=start_year,
        end_year=end_year,
        year_rampup=year_rampup,
        drate=drate,
        q_plateau_ratio=q_plateau_ratio,
        q_min_ratio=q_min_ratio,
        volume=volume,
    )

    return pd.DataFrame(
        {
            'year': np.arange(start_year, end_year + 1).tolist(),
            'rpd': rdp_array.tolist()
        }
    ).set_index('year').to_dict()

def get_grosssplit_split(data: dict):
    """
    The function to get the contractor split information from Gross Split Contract Scheme.

    Parameters
    ----------
    data: dict
        The dictionary of the data input.

    Returns
    -------
    dict
        The dictionary containing the information of the contractor split.


    """
    start_date, end_date, oil_onstream_date, gas_onstream_date, lifting, tangible, intangible, opex, asr, lbt, cost_of_sales = (
        get_setup_dict(data=data))

    contract = GrossSplit(
        start_date=start_date,
        end_date=end_date,
        oil_onstream_date=oil_onstream_date,
        gas_onstream_date=gas_onstream_date,
        lifting=lifting,
        capital_cost=tangible,
        intangible_cost=intangible,
        opex=opex,
        asr_cost=asr,
        lbt_cost=lbt,
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
        base_split_ctr_oil=convert_to_float(target=data['grosssplit']['base_split_ctr_oil']),
        base_split_ctr_gas=convert_to_float(target=data['grosssplit']['base_split_ctr_gas']),
        split_ministry_disc=convert_to_float(target=data['grosssplit']['split_ministry_disc']),
        oil_dmo_volume_portion=convert_to_float(target=data['grosssplit']['oil_dmo_volume_portion']),
        oil_dmo_fee_portion=convert_to_float(target=data['grosssplit']['oil_dmo_fee_portion']),
        oil_dmo_holiday_duration=data['grosssplit']['oil_dmo_holiday_duration'],
        gas_dmo_volume_portion=convert_to_float(target=data['grosssplit']['gas_dmo_volume_portion']),
        gas_dmo_fee_portion=convert_to_float(target=data['grosssplit']['gas_dmo_fee_portion']),
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
        "effective_tax_rate": convert_list_to_array_float_or_array_or_none(data_list=data['contract_arguments']['effective_tax_rate']),
        "sunk_cost_reference_year": data['contract_arguments']['sunk_cost_reference_year'],
        "depr_method": convert_str_to_depremethod(str_object=data['contract_arguments']['depr_method']),
        "decline_factor": data['contract_arguments']['decline_factor'],
        "vat_rate": convert_list_to_array_float_or_array(data_input=data['contract_arguments']['vat_rate']),
        "inflation_rate": convert_list_to_array_float_or_array(data_input=data['contract_arguments']['inflation_rate']),
        "inflation_rate_applied_to": convert_str_to_inflationappliedto(
            str_object=data['contract_arguments']['inflation_rate_applied_to']),
        "cum_production_split_offset": convert_list_to_array_float_or_array(data_input=data["contract_arguments"]["cum_production_split_offset"]),
        "amortization": data["contract_arguments"]["amortization"],
        "regime": convert_grosssplitregime_to_enum(target=data["contract_arguments"]["regime"])
    }

    # Running the contract
    contract.run(**contract_arguments_dict)

    # Retrieving the split information
    contractor_split = pd.DataFrame({
        'project_years': contract.project_years.tolist(),
        'oil_base_split': contract._oil_base_split.tolist(),
        'gas_base_split': contract._gas_base_split.tolist(),
        'var_split_array': contract._var_split_array.tolist(),
        'oil_prog_price_split': contract._oil_prog_price_split.tolist(),
        'gas_prog_price_split': contract._gas_prog_price_split.tolist(),
        'oil_prog_cumulative_production_split': contract._oil_prog_cum_split.tolist(),
        'gas_prog_cumulative_production_split': contract._gas_prog_cum_split.tolist(),
        'oil_prog_total_split': contract._oil_prog_split.tolist(),
        'gas_prog_total_split': contract._gas_prog_split.tolist(),
        'oil_ctr_split': contract._oil_ctr_split_prior_bracket.tolist(),
        'gas_ctr_split': contract._gas_ctr_split_prior_bracket.tolist(),
    }).set_index('project_years').to_dict()

    years_of_maximum_split = {
        'oil': contract._oil_year_maximum_ctr_split.tolist(),
        'gas': contract._gas_year_maximum_ctr_split.tolist(),
    }

    return {
        'contractor_split': contractor_split,
        'years_of_maximum_split': years_of_maximum_split
    }


def get_transition_split(data: dict):
    """
    The function to get the contractor split information from Transition Contract Scheme.

    Parameters
    ----------
    data: dict
        The dictionary of the data input.

    Returns
    -------
    dict
        The dictionary containing the information of the contractor split.

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

    # Making the base for the loops and container of the result
    result = {}

    # Retrieving the split information from first contract
    for index, contract in enumerate([contract_1, contract_2]):
        if isinstance(contract, GrossSplit):
            # Retrieving the split information
            contractor_split = pd.DataFrame({
                'project_years': contract.project_years.tolist(),
                'oil_base_split': contract._oil_base_split.tolist(),
                'gas_base_split': contract._gas_base_split.tolist(),
                'var_split_array': contract._var_split_array.tolist(),
                'oil_prog_split': contract._oil_prog_split.tolist(),
                'gas_prog_split': contract._gas_prog_split.tolist(),
                'oil_ctr_split': contract._oil_ctr_split_prior_bracket.tolist(),
                'gas_ctr_split': contract._gas_ctr_split_prior_bracket.tolist(),
            }).set_index('project_years').to_dict()

            years_of_maximum_split = {
                'oil': contract._oil_year_maximum_ctr_split.tolist(),
                'gas': contract._gas_year_maximum_ctr_split.tolist(),
            }

            result['contract_' + str(index+1)] = {
                'contractor_split': contractor_split,
                'years_of_maximum_split': years_of_maximum_split
            }
        else:
            result['contract_' + str(index + 1)] = {}
            pass

    return result

def get_economic_limit(
        data: dict,
):
    """
    The function to get the information of economic limit years from a cashflow using selected method.

    Parameters
    ----------
    data: dict

    Returns
    -------
    int
        The index
    """
    years = np.array(data['years'], dtype=int)
    cash_flow = np.array(data['cash_flow'], dtype=float)
    method = convert_to_method_limit(target=data['method'])
    index_limit = econ_limit(cashflow=cash_flow, method=method)
    return years[index_limit]


def get_asr_expenditures(data:dict) -> dict:
    """
    The Function to get the expenditures of an ASR cost.

    Parameters
    ----------
    data: dict

    Returns
    -------
    dict
        The dictionary of ASR expenditures.

    """
    # Initiating the asr data
    asr_pseudo = {'asr':data['asr']}

    # Mimics the baseproject data
    data_pseudo = {
        "setup": {
            "start_date": data['start_date'],
            "end_date": data['end_date'],
            "oil_onstream_date": None,
            "gas_onstream_date": None,
        },
        "summary_arguments":{
            "reference_year":None,
            "inflation_rate":0.0,
            "discount_rate": 0.1,
            "npv_mode": "Full Cycle Nominal Terms",
            "discounting_mode": "Mid Year",
            "profitability_discounted": False,
        },
        "contract_arguments":{
            "sulfur_revenue": "Addition to Gas Revenue",
            "electricity_revenue": "Addition to Oil Revenue",
            "co2_revenue": "Addition to Gas Revenue",
            "sunk_cost_reference_year": None,
            "year_inflation": 0,
            "inflation_rate": 0,
            "vat_rate": 0,
            "inflation_rate_applied_to": "CAPEX",
        },
        "lifting": None,
        "capital": None,
        "intangible": None,
        "opex": None,
        "asr": asr_pseudo,
        "lbt": None,
        "cost_of_sales": None,
    }

    # Parsing the data into base project dataclass
    contract = get_baseproject(data=data_pseudo, summary_result=False)[1]

    # Returning the ASR Expenditures
    df = pd.DataFrame(
        {
            'project_years': contract.project_years,
            'oil_asr_expenditures': contract._oil_asr_expenditures_post_tax,
            'gas_asr_expenditures': contract._gas_asr_expenditures_post_tax,
        }
    )
    df = df.set_index('project_years').to_dict()
    return df


def get_lbt_expenditures(data:dict) -> dict:
    """
    The Function to get the expenditures of an LBT cost.

    Parameters
    ----------
    data: dict

    Returns
    -------
    dict
        The dictionary of LBT expenditures.

    """
    # Initiating the LBT data
    lbt_pseudo = {'lbt':data['lbt']}

    # Mimics the baseproject data
    data_pseudo = {
        "setup": {
            "start_date": data['start_date'],
            "end_date": data['end_date'],
            "oil_onstream_date": None,
            "gas_onstream_date": None,
        },
        "summary_arguments":{
            "reference_year":None,
            "inflation_rate":0.0,
            "discount_rate": 0.1,
            "npv_mode": "Full Cycle Nominal Terms",
            "discounting_mode": "Mid Year",
            "profitability_discounted": False,
        },
        "contract_arguments":{
            "sulfur_revenue": "Addition to Gas Revenue",
            "electricity_revenue": "Addition to Oil Revenue",
            "co2_revenue": "Addition to Gas Revenue",
            "sunk_cost_reference_year": None,
            "year_inflation": 0,
            "inflation_rate": 0,
            "vat_rate": 0,
            "inflation_rate_applied_to": "CAPEX",
        },
        "lifting": None,
        "capital": None,
        "intangible": None,
        "opex": None,
        "asr": None,
        "lbt": lbt_pseudo,
        "cost_of_sales": None,
    }

    # Parsing the data into base project dataclass
    contract = get_baseproject(data=data_pseudo, summary_result=False)[1]

    # Returning the LBT Expenditures
    df = pd.DataFrame(
        {
            'project_years': contract.project_years,
            'oil_lbt_expenditures': contract._oil_lbt_expenditures_post_tax,
            'gas_lbt_expenditures': contract._gas_lbt_expenditures_post_tax,
        }
    )
    df = df.set_index('project_years').to_dict()
    return df

def get_sensitivity(data:dict, contract_type:str):
    if 'sensitivity_arguments' not in data:
        raise ContractException("The payload does not have the sensitivity_arguments key")

    if data['sensitivity_arguments'] is None:
        raise ContractException("The payload sensitivity_arguments does not have any values")

    # Retrieving the contract, contract_arguments_dict, summary_arguments_dict based on the contract type
    if contract_type == 'Cost Recovery':
        contract = get_costrecovery(data=data)[1]
        contract_arguments = get_costrecovery(data=data)[2]
        summary_argument = get_costrecovery(data=data)[3]

    elif contract_type == 'Gross Split':
        contract = get_grosssplit(data=data)[1]
        contract_arguments = get_grosssplit(data=data)[2]
        summary_argument = get_grosssplit(data=data)[3]

    elif contract_type == 'Transition':
        contract = get_transition(data=data)[1]
        contract_arguments = get_transition(data=data)[2]
        summary_argument = get_transition(data=data)[3]

    else:
        contract = get_baseproject(data=data)[1]
        contract_arguments = get_baseproject(data=data)[2]
        summary_argument = get_baseproject(data=data)[3]

    # Constructing the sensitivity arguments
    sensitivity_result = sensitivity_psc(
        contract=contract,
        contract_arguments=contract_arguments,
        summary_arguments=summary_argument,
        min_deviation=data['sensitivity_arguments']['min_deviation'],
        max_deviation=data['sensitivity_arguments']['max_deviation'],
        base_value=data['sensitivity_arguments']['base_value'],
        step=data['sensitivity_arguments']['step'],
        dataframe_output=False,
    )

    return sensitivity_result

def get_uncertainty(data: dict, contract_type: str):
    if 'uncertainty_arguments' not in data:
        raise ContractException("The payload does not have the uncertainty_arguments key")

    if data['uncertainty_arguments'] is None:
        raise ContractException("The payload uncertainty_arguments does not have any values")

    # Retrieving the contract, contract_arguments_dict, summary_arguments_dict based on the contract type
    if contract_type == 'Cost Recovery':
        contract = get_costrecovery(data=data)[1]
        contract_arguments = get_costrecovery(data=data)[2]
        summary_argument = get_costrecovery(data=data)[3]

    elif contract_type == 'Gross Split':
        contract = get_grosssplit(data=data)[1]
        contract_arguments = get_grosssplit(data=data)[2]
        summary_argument = get_grosssplit(data=data)[3]

    elif contract_type == 'Transition':
        contract = get_transition(data=data)[1]
        contract_arguments = get_transition(data=data)[2]
        summary_argument = get_transition(data=data)[3]

    else:
        contract = get_baseproject(data=data)[1]
        contract_arguments = get_baseproject(data=data)[2]
        summary_argument = get_baseproject(data=data)[3]

    # Constructing the sensitivity arguments
    uncertainty_args = {
        'contract': contract,
        'contract_arguments': contract_arguments,
        'summary_arguments': summary_argument,
        'run_number': data['uncertainty_arguments']['run_number'],
        'min_oil_price': data['uncertainty_arguments']['min_oil_price'],
        'mean_oil_price': data['uncertainty_arguments']['mean_oil_price'],
        'max_oil_price': data['uncertainty_arguments']['max_oil_price'],
        'min_gas_price': data['uncertainty_arguments']['min_gas_price'],
        'mean_gas_price': data['uncertainty_arguments']['mean_gas_price'],
        'max_gas_price': data['uncertainty_arguments']['max_gas_price'],
        'min_opex': data['uncertainty_arguments']['min_opex'],
        'mean_opex': data['uncertainty_arguments']['mean_opex'],
        'max_opex': data['uncertainty_arguments']['max_opex'],
        'min_capex': data['uncertainty_arguments']['min_capex'],
        'mean_capex': data['uncertainty_arguments']['mean_capex'],
        'max_capex': data['uncertainty_arguments']['max_capex'],
        'min_lifting': data['uncertainty_arguments']['min_lifting'],
        'mean_lifting': data['uncertainty_arguments']['mean_lifting'],
        'max_lifting': data['uncertainty_arguments']['max_lifting'],
        'oil_price_stddev': data['uncertainty_arguments']['oil_price_stddev'],
        'gas_price_stddev': data['uncertainty_arguments']['gas_price_stddev'],
        'opex_stddev': data['uncertainty_arguments']['opex_stddev'],
        'capex_stddev': data['uncertainty_arguments']['capex_stddev'],
        'lifting_stddev': data['uncertainty_arguments']['lifting_stddev'],
        'oil_price_distribution': convert_to_uncertainty_distribution(target=data['uncertainty_arguments']['oil_price_distribution']),
        'gas_price_distribution': convert_to_uncertainty_distribution(target=data['uncertainty_arguments']['gas_price_distribution']),
        'opex_distribution': convert_to_uncertainty_distribution(target=data['uncertainty_arguments']['opex_distribution']),
        'capex_distribution': convert_to_uncertainty_distribution(target=data['uncertainty_arguments']['capex_distribution']),
        'lifting_distribution': convert_to_uncertainty_distribution(target=data['uncertainty_arguments']['lifting_distribution']),
    }

    return uncertainty_psc(**uncertainty_args)

