"""
[EXPERIMENTAL]
Configuration to undertake optimization study.
"""
import numpy as np
from scipy.optimize import minimize_scalar

from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.econ.selection import OptimizationParameter, OptimizationTarget
from pyscnomics.tools.summary import get_summary

from pyscnomics.econ.costs import Tangible, Intangible, OPEX, ASR


class OptimizationException(Exception):
    """Exception to raise for a misuse of BaseProject class"""

    pass


def adjust_contract(
    contract: CostRecovery | GrossSplit,
    contract_arguments: dict,
    variable: OptimizationParameter,
    value: float,
    summary_argument: dict,
    target_parameter: str
) -> (CostRecovery | GrossSplit, dict):
    """
    The function used to adjust the variable within a psc contract object.
    This function will be used by optimize_psc().

    Parameters
    ----------
    contract: CostRecovery | GrossSplit
        The contract object.
    contract_arguments: dict
        The contract arguments that passed on the fly .run() of the contract dataclass.
    variable: OptimizationParameter
        The enum selection of variable that will be changed.
    value: float
        The new value of the variable that will be replaced.
    summary_argument: dict
        The dictionary containing the arguments that passed summary function.
    target_parameter: str
        The string of the targeted parameter.

    Returns
    -------
    out: tuple
        The result of the target parameter and contract object that has been modified and run.

    """
    # When the optimization is VAT
    if variable is OptimizationParameter.VAT_DISCOUNT:
        contract = adjust_cost_element(contract=contract,
                                       adjustment_value=value,
                                       adjustment_variable='VAT', )

    # When the optimization is LBT
    elif variable is OptimizationParameter.LBT_DISCOUNT:
        contract = adjust_cost_element(contract=contract,
                                       adjustment_value=value,
                                       adjustment_variable='LBT', )

    # The condition when contract is Cost Recovery
    if isinstance(contract, CostRecovery):
        # Changing the attributes of the contract based on the chosen variable
        if variable is OptimizationParameter.OIL_CTR_PRETAX:
            contract.oil_ctr_pretax_share = value

        if variable is OptimizationParameter.GAS_CTR_PRETAX:
            contract.gas_ctr_pretax_share = value

        if variable is OptimizationParameter.OIL_FTP_PORTION:
            contract.oil_ftp_portion = value

        if variable is OptimizationParameter.GAS_FTP_PORTION:
            contract.gas_ftp_portion = value

        if variable is OptimizationParameter.OIL_IC:
            contract.oil_ic_rate = value

        if variable is OptimizationParameter.GAS_IC:
            contract.gas_ic_rate = value

        if variable is OptimizationParameter.OIL_DMO_FEE:
            contract.oil_dmo_fee_portion = value

        if variable is OptimizationParameter.GAS_DMO_FEE:
            contract.gas_dmo_fee_portion = value

        if variable is OptimizationParameter.VAT_RATE:
            contract_arguments['vat_rate'] = value

        if variable is OptimizationParameter.EFFECTIVE_TAX_RATE:
            contract_arguments['tax_rate'] = value

    # The condition when contract is Gross Split
    if isinstance(contract, GrossSplit):
        # Changing the attributes of the contract based on the chosen variable
        if variable is OptimizationParameter.MINISTERIAL_DISCRETION:
            contract.split_ministry_disc = value

        if variable is OptimizationParameter.VAT_RATE:
            contract_arguments['vat_rate'] = value

        if variable is OptimizationParameter.EFFECTIVE_TAX_RATE:
            contract_arguments['tax_rate'] = value

    # Running the contract
    contract.run(**contract_arguments)

    # Get the summary of the new contract and get its value of the targeted optimization
    summary_argument['contract'] = contract
    result_psc = get_summary(**summary_argument)[target_parameter]

    return result_psc, contract


def optimize_psc(
    dict_optimization: dict,
    contract: CostRecovery | GrossSplit,
    contract_arguments: dict,
    target_optimization_value: float,
    summary_argument: dict,
    target_parameter: OptimizationTarget = OptimizationTarget.IRR,
 ) -> (list, list, float, list):
    """
    The function to get contract variable(s) that resulting the desired target or contract's economic target.

    Parameters
    ----------
    dict_optimization: dict
        The optimization dictionary that containing the information about minimum boundary and upper boundary
        of the optimized parameters.
    contract: CostRecovery | GrossSplit
        The contract object.
    contract_arguments: dict
        The contract arguments that passed on the fly .run() of the contract dataclass.
    target_optimization_value: float
        The desired target value.
    summary_argument: dict
        The dictionary containing the arguments that passed summary function.
    target_parameter: OptimizationTarget
        The enum selection for economic indicator  that will be optimized.

    Notes
    -------
    The dictionary of dict_optimization should be at least having the structure as the following:
        dict_optimization = {'parameters': list[OptimizationParameter],
                             'min': np.ndarray,
                             'max': np.ndarray}

        'parameters' keys containing the enum list of the variable that will be optimized to achieve the target.
        'min' keys containing the minimum value of each parameter.
        'max' keys containing the minimum value of each parameter.


    Returns
    -------
    out : tuple

    list_str: list
        The list of parameter that passed and has been optimized.
    list_params_value: list
        The list of parameter's value that passed and has been optimized.
    result_optimization: float
        The value of the targeted parameter which the result of the optimization.
    list_executed_contract: list
        The list of executed contracts.
    """
    # Changing the Optimization selection from Enum to string in order to retrieve
    # the result from summary dictionary
    if target_parameter is OptimizationTarget.IRR:
        target_parameter = 'ctr_irr'

    elif target_parameter is OptimizationTarget.NPV:
        target_parameter = 'ctr_npv'

    elif target_parameter is OptimizationTarget.PI:
        target_parameter = 'ctr_pi'

    else:
        raise OptimizationException(
            f"target {target_parameter} "
            f"is should be one of: {[OptimizationTarget.value]}"
        )

    # Changing the parameters list[str] into list[OptimizationParameters(Enum)]
    list_params = dict_optimization['parameter']

    # Defining Base Value list to contain value of optimized parameters and status of the optimization
    list_params_value = ['Base Value'] * len(list_params)

    # Defining the empty result of optimization target, will be defined later
    result_optimization = None

    # Defining the executed contract list
    list_executed_contract = []

    psc = contract
    for index, param in enumerate(list_params):

        # Get the maximum value of each params
        if param is OptimizationParameter.OIL_CTR_PRETAX:
            max_value = dict_optimization['max'][index]
        elif param is OptimizationParameter.GAS_CTR_PRETAX:
            max_value = dict_optimization['max'][index]
        elif param is OptimizationParameter.OIL_FTP_PORTION:
            max_value = dict_optimization['min'][index]
        elif param is OptimizationParameter.GAS_FTP_PORTION:
            max_value = dict_optimization['min'][index]
        elif param is OptimizationParameter.OIL_IC:
            max_value = dict_optimization['max'][index]
        elif param is OptimizationParameter.GAS_IC:
            max_value = dict_optimization['max'][index]
        elif param is OptimizationParameter.OIL_DMO_FEE:
            max_value = dict_optimization['max'][index]
        elif param is OptimizationParameter.GAS_DMO_FEE:
            max_value = dict_optimization['max'][index]
        elif param is OptimizationParameter.VAT_RATE:
            max_value = dict_optimization['min'][index]
        elif param is OptimizationParameter.EFFECTIVE_TAX_RATE:
            max_value = dict_optimization['min'][index]
        elif param is OptimizationParameter.VAT_DISCOUNT:
            max_value = dict_optimization['max'][index]
        elif param is OptimizationParameter.LBT_DISCOUNT:
            max_value = dict_optimization['max'][index]
        else:
            # Ministerial Discretion
            max_value = dict_optimization['max'][index]

        # Changing the Contract parameter value based on the given input
        result_psc, psc = adjust_contract(contract=psc,
                                          contract_arguments=contract_arguments,
                                          variable=param,
                                          value=max_value,
                                          summary_argument=summary_argument,
                                          target_parameter=target_parameter, )

        # Able to conduct optimization since the result is greater than the target
        if result_psc > target_optimization_value:
            # Defining the upper and lower limit of the optimized variable
            bounds = (dict_optimization['min'][index],
                      dict_optimization['max'][index])

            def objective_run(new_value):
                result_psc_obj, executed_contract = adjust_contract(contract=psc,
                                                                    contract_arguments=contract_arguments,
                                                                    variable=param,
                                                                    value=new_value,
                                                                    summary_argument=summary_argument,
                                                                    target_parameter=target_parameter)

                result_obj = abs(result_psc_obj - target_optimization_value)
                return result_obj

            # Optimization of the objective function
            optim_result = minimize_scalar(objective_run, bounds=bounds, method='bounded')

            # Difference value from target and optimization result
            optimized_diff = optim_result.fun

            # Optimized Parameter Value
            optimized_parameter = optim_result.x

            # Result of the objective function
            function_result = optimized_diff + target_optimization_value

            # Writing the result of optimization to the list_params_value
            list_params_value[index] = optimized_parameter

            # Defining the result_optimization
            result_optimization = function_result

            # Defining the executed contract
            executed_contract = adjust_contract(contract=psc,
                                                contract_arguments=contract_arguments,
                                                variable=param,
                                                value=optimized_parameter,
                                                summary_argument=summary_argument,
                                                target_parameter=target_parameter)[1]

            # Filling the list with executed contract
            list_executed_contract.append(executed_contract)

            # # Printing for debugging
            # print('Parameter:', param)
            # print('Optimized Parameter Value:', optimized_parameter)
            # print('Targeted Value:', target_optimization_value)
            # print('Optimization Value:', function_result)
            # print('Difference of Target and Optimization Result:', optimized_diff)

            # Exiting the loop since the target has been achieved
            break

        elif result_psc <= target_optimization_value:
            # Writing the maximum value to the list_params_value
            list_params_value[index] = max_value

            # Defining the result_optimization
            result_optimization = result_psc

            list_executed_contract.append(psc)

    # Converting the list of enum into list of str enum value
    list_str = [enum_value.value for enum_value in list_params]

    # # Printing for debugging
    # print(list_str)
    # print(list_params_value)
    # print(result_optimization)

    return list_str, list_params_value, result_optimization, list_executed_contract


def adjust_cost_element(contract: CostRecovery | GrossSplit,
                        adjustment_value: float = 1,
                        adjustment_variable: str = 'VAT') -> CostRecovery | GrossSplit:
    """
    Function to adjust the vat discount or lbt discount of the cost element
    by multiplying them with the adjustment factor.

    Parameters
    ----------
    contract: CostRecovery | GrossSplit
        The contract which will be adjusted
    adjustment_value: float
        The factor which will be applied for the adjustment
    adjustment_variable: str
        The cost element that will be adjusted. The options are: "VAT" or "LBT"

    Returns
    -------
    contract_adjusted: CostRecovery | GrossSplit
        The contract that the cost element has been adjusted.

    """

    # Condition when the VAT of each cost element will be adjusted
    if adjustment_variable == 'VAT':
        # Adjusting the Tangible cost of the contract
        tangible_adjusted = tuple([
            Tangible(start_year=tan.start_year,
                     end_year=tan.end_year,
                     cost=tan.cost,
                     expense_year=tan.expense_year,
                     cost_allocation=tan.cost_allocation,
                     description=tan.description,
                     vat_portion=tan.vat_portion,
                     vat_discount=adjustment_value,
                     lbt_portion=tan.lbt_portion,
                     lbt_discount=tan.lbt_discount,
                     pis_year=tan.pis_year,
                     salvage_value=tan.salvage_value,
                     useful_life=tan.useful_life,
                     depreciation_factor=tan.depreciation_factor,
                     is_ic_applied=tan.is_ic_applied,
                     ) for tan in contract.tangible_cost
        ])

        # Adjusting the Intangible cost of the contract
        intangible_adjusted = tuple([
            Intangible(start_year=intang.start_year,
                       end_year=intang.end_year,
                       cost=intang.cost,
                       expense_year=intang.expense_year,
                       cost_allocation=intang.cost_allocation,
                       description=intang.description,
                       vat_portion=intang.vat_portion,
                       vat_discount=adjustment_value,
                       lbt_portion=intang.lbt_portion,
                       lbt_discount=intang.lbt_discount, ) for intang in contract.intangible_cost
        ])

        # Adjusting the OPEX cost of the contract
        opex_adjusted = tuple([
            OPEX(start_year=opx.start_year,
                 end_year=opx.end_year,
                 expense_year=opx.expense_year,
                 cost_allocation=opx.cost_allocation,
                 description=opx.description,
                 vat_portion=opx.vat_portion,
                 vat_discount=adjustment_value,
                 lbt_portion=opx.lbt_portion,
                 lbt_discount=opx.lbt_discount,
                 fixed_cost=opx.fixed_cost,
                 prod_rate=opx.prod_rate,
                 cost_per_volume=opx.cost_per_volume, ) for opx in contract.opex
        ])

        # Adjusting the ASR cost of the contract
        asr_adjusted = tuple([
            ASR(start_year=asr.start_year,
                end_year=asr.end_year,
                cost=asr.cost,
                expense_year=asr.expense_year,
                cost_allocation=asr.cost_allocation,
                description=asr.description,
                vat_portion=asr.vat_portion,
                vat_discount=adjustment_value,
                lbt_portion=asr.lbt_portion,
                lbt_discount=asr.lbt_discount, ) for asr in contract.asr_cost
        ])

    elif adjustment_variable == 'LBT':
        # Adjusting the Tangible cost of the contract
        tangible_adjusted = tuple([
            Tangible(start_year=tan.start_year,
                     end_year=tan.end_year,
                     cost=tan.cost,
                     expense_year=tan.expense_year,
                     cost_allocation=tan.cost_allocation,
                     description=tan.description,
                     vat_portion=tan.vat_portion,
                     vat_discount=tan.vat_discount,
                     lbt_portion=tan.lbt_portion,
                     lbt_discount=adjustment_value,
                     pis_year=tan.pis_year,
                     salvage_value=tan.salvage_value,
                     useful_life=tan.useful_life,
                     depreciation_factor=tan.depreciation_factor,
                     is_ic_applied=tan.is_ic_applied,
                     ) for tan in contract.tangible_cost
        ])

        # Adjusting the Intangible cost of the contract
        intangible_adjusted = tuple([
            Intangible(start_year=intang.start_year,
                       end_year=intang.end_year,
                       cost=intang.cost,
                       expense_year=intang.expense_year,
                       cost_allocation=intang.cost_allocation,
                       description=intang.description,
                       vat_portion=intang.vat_portion,
                       vat_discount=intang.vat_discount,
                       lbt_portion=intang.lbt_portion,
                       lbt_discount=adjustment_value, ) for intang in contract.intangible_cost
        ])

        # Adjusting the OPEX cost of the contract
        opex_adjusted = tuple([
            OPEX(start_year=opx.start_year,
                 end_year=opx.end_year,
                 expense_year=opx.expense_year,
                 cost_allocation=opx.cost_allocation,
                 description=opx.description,
                 vat_portion=opx.vat_portion,
                 vat_discount=opx.vat_discount,
                 lbt_portion=opx.lbt_portion,
                 lbt_discount=adjustment_value,
                 fixed_cost=opx.fixed_cost,
                 prod_rate=opx.prod_rate,
                 cost_per_volume=opx.cost_per_volume, ) for opx in contract.opex
        ])

        # Adjusting the ASR cost of the contract
        asr_adjusted = tuple([
            ASR(start_year=asr.start_year,
                end_year=asr.end_year,
                cost=asr.cost,
                expense_year=asr.expense_year,
                cost_allocation=asr.cost_allocation,
                description=asr.description,
                vat_portion=asr.vat_portion,
                vat_discount=asr.vat_discount,
                lbt_portion=asr.lbt_portion,
                lbt_discount=adjustment_value, ) for asr in contract.asr_cost
        ])

        # Condition when the chosen option is not recognized
    else:
        raise OptimizationException(f"Adjustment Variable {adjustment_variable} "
                                    f"do not exist. It should be VAT or LBT in string data type")

    # When the contract is CostRecovery, parsing back the adjusted cost elements to the cost recovery contract
    if isinstance(contract, CostRecovery):
        contract_adjusted = CostRecovery(start_date=contract.start_date,
                                         end_date=contract.end_date,
                                         oil_onstream_date=contract.oil_onstream_date,
                                         gas_onstream_date=contract.gas_onstream_date,
                                         lifting=contract.lifting,
                                         tangible_cost=tangible_adjusted,
                                         intangible_cost=intangible_adjusted,
                                         opex=opex_adjusted,
                                         asr_cost=asr_adjusted,
                                         oil_ftp_is_available=contract.oil_ftp_is_available,
                                         oil_ftp_is_shared=contract.oil_ftp_is_shared,
                                         oil_ftp_portion=contract.oil_ftp_portion,
                                         gas_ftp_is_available=contract.gas_ftp_is_available,
                                         gas_ftp_is_shared=contract.gas_ftp_is_shared,
                                         gas_ftp_portion=contract.gas_ftp_portion,
                                         tax_split_type=contract.tax_split_type,
                                         condition_dict=contract.condition_dict,
                                         indicator_rc_icp_sliding=contract.indicator_rc_icp_sliding,
                                         oil_ctr_pretax_share=contract.oil_ctr_pretax_share,
                                         gas_ctr_pretax_share=contract.gas_ctr_pretax_share,
                                         oil_ic_rate=contract.oil_ic_rate,
                                         gas_ic_rate=contract.gas_ic_rate,
                                         ic_is_available=contract.ic_is_available,
                                         oil_cr_cap_rate=contract.oil_cr_cap_rate,
                                         gas_cr_cap_rate=contract.gas_cr_cap_rate,
                                         oil_dmo_volume_portion=contract.oil_dmo_volume_portion,
                                         oil_dmo_fee_portion=contract.oil_dmo_fee_portion,
                                         oil_dmo_holiday_duration=contract.oil_dmo_holiday_duration,
                                         gas_dmo_volume_portion=contract.gas_dmo_volume_portion,
                                         gas_dmo_fee_portion=contract.gas_dmo_fee_portion,
                                         gas_dmo_holiday_duration=contract.gas_dmo_holiday_duration, )

    # When the contract is GrossSplit, parsing back the adjusted cost elements to the gross split contract
    elif isinstance(contract, GrossSplit):
        contract_adjusted = GrossSplit(start_date=contract.start_date,
                                       end_date=contract.end_date,
                                       oil_onstream_date=contract.oil_onstream_date,
                                       gas_onstream_date=contract.gas_onstream_date,
                                       lifting=contract.lifting,
                                       tangible_cost=tangible_adjusted,
                                       intangible_cost=intangible_adjusted,
                                       opex=opex_adjusted,
                                       asr_cost=asr_adjusted,
                                       field_status=contract.field_status,
                                       field_loc=contract.field_loc,
                                       res_depth=contract.res_depth,
                                       infra_avail=contract.infra_avail,
                                       res_type=contract.res_type,
                                       api_oil=contract.api_oil,
                                       domestic_use=contract.domestic_use,
                                       prod_stage=contract.prod_stage,
                                       co2_content=contract.co2_content,
                                       h2s_content=contract.h2s_content,
                                       base_split_ctr_oil=contract.base_split_ctr_oil,
                                       base_split_ctr_gas=contract.base_split_ctr_gas,
                                       split_ministry_disc=contract.split_ministry_disc,
                                       oil_dmo_volume_portion=contract.oil_dmo_volume_portion,
                                       oil_dmo_fee_portion=contract.oil_dmo_fee_portion,
                                       oil_dmo_holiday_duration=contract.oil_dmo_holiday_duration,
                                       gas_dmo_volume_portion=contract.gas_dmo_volume_portion,
                                       gas_dmo_fee_portion=contract.gas_dmo_fee_portion,
                                       gas_dmo_holiday_duration=contract.gas_dmo_holiday_duration,)

    # When the contract is not recognized, raise an exception
    else:
        raise OptimizationException(f"Contract Type {type(contract)} , is not recognized for optimization module")

    return contract_adjusted
