"""
This file is utilized to be the adapter of the router into the core codes.
"""
import pandas as pd

from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
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


def get_setup_dict(data: dict):
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


def get_summary_dict(data: dict):
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


def get_costrecovery(data: dict):
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
        indicator_rc_icp_sliding=convert_list_to_array_float(data_list=data['costrecovery']['indicator_rc_icp_sliding']),
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
        "electricity_revenue": convert_str_to_otherrevenue(str_object=data['contract_arguments']['electricity_revenue']),
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
    }

    # Running the contract
    contract.run(**contract_arguments_dict)

    # Filling the summary arguments
    summary_arguments_dict = get_summary_dict(data=data)
    summary_arguments_dict['contract'] = contract
    summary = get_summary(**summary_arguments_dict)

    # Converting the summary format to skk summary format
    summary_skk = convert_summary_to_dict(dict_object=summary)

    return summary_skk, contract, contract_arguments_dict, summary_arguments_dict


def get_contract_table(data: dict, contract_type: str = 'Cost Recovery'):
    # Adjusting the variable to the corresponding contract type
    if contract_type == 'Cost Recovery':
        contract = get_costrecovery(data=data)[1]
        year_column = 'Year'

    elif contract_type == 'Gross Split':
        contract = get_grosssplit(data=data)[1]
        year_column = 'Years'

    else:
        contract = NotImplemented
        year_column = 'Year'

    # Retrieving the table
    table_oil, table_gas, table_consolidated = get_table(contract=contract)

    # Forming the table dictionary as the output
    table_all_dict = {'oil': table_oil.set_index(year_column).to_dict(),
                      'gas': table_gas.set_index(year_column).to_dict(),
                      'consolidated': table_consolidated.set_index(year_column).to_dict()}

    return table_all_dict


def get_contract_optimization(data: dict, contract_type: str = 'Cost Recovery'):
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

    list_str, list_params_value, result_optimization = optimize_psc(
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

    return result_parameters


def get_grosssplit(data: dict):
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
    )

    # Filling the arguments of the contract with the data input
    contract_arguments_dict = {
        "sulfur_revenue": convert_str_to_otherrevenue(str_object=data['contract_arguments']['sulfur_revenue']),
        "electricity_revenue": convert_str_to_otherrevenue(str_object=data['contract_arguments']['electricity_revenue']),
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
        "inflation_rate_applied_to": convert_str_to_inflationappliedto(str_object=data['contract_arguments']['inflation_rate_applied_to']),
    }

    # Running the contract
    contract.run(**contract_arguments_dict)

    # Filling the summary arguments
    summary_arguments_dict = get_summary_dict(data=data)
    summary_arguments_dict['contract'] = contract
    summary = get_summary(**summary_arguments_dict)

    # Converting the summary format to skk summary format
    summary_skk = convert_summary_to_dict(dict_object=summary)

    return summary_skk, contract, contract_arguments_dict, summary_arguments_dict
