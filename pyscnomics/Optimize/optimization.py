from scipy.optimize import minimize_scalar

from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.econ.selection import OptimizationParameter, OptimizationTarget
from pyscnomics.tools.summary import get_summary


class OptimizationException(Exception):
    """Exception to raise for a misuse of BaseProject class"""

    pass


def adjust_contract(contract: CostRecovery | GrossSplit,
                    contract_arguments: dict,
                    variable: OptimizationParameter,
                    value: float,
                    summary_argument: dict,
                    target_parameter: str) -> (CostRecovery | GrossSplit, dict):
    """
    The function used to adjust the variable within a psc contract object. This function will be used by optimize_psc().

    Parameters
    ----------
    contract: CostRecovery | GrossSplit
        The contract object.
    contract_arguments: dict
        The contract arguments that passed on the fly (.run() of the contract dataclass.
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


def optimize_psc(dict_optimization: dict,
                 contract: CostRecovery | GrossSplit,
                 contract_arguments: dict,
                 target_optimization_value: float,
                 summary_argument: dict,
                 target_parameter: OptimizationTarget = OptimizationTarget.IRR,
                 ) -> (list, list, float):
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
        The contract arguments that passed on the fly (.run() of the contract dataclass.
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
        The list of parameter that passed to has been optimized.
    list_params_value: list
        The list of parameter's value that passed has been optimized.
    result_optimization: float
        The value of the targeted parameter which the result of the optimization.
    """
    # Changing the Optimization selection from Enum to string in order to retrieve the result from summary dictionary
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
    list_params = dict_optimization['parameters']

    # Defining empty list to contain value of optimized parameters and status of the optimization
    list_params_value = ['Base Value'] * len(list_params)

    # Defining the empty result of optimization target, will be defined later
    result_optimization = None

    psc = contract
    for index, param in enumerate(list_params):

        # Get the maximum value of each params
        if param is OptimizationParameter.OIL_CTR_PRETAX:
            max_value = dict_optimization['max'][index]
        elif param is OptimizationParameter.GAS_CTR_PRETAX:
            max_value = dict_optimization['max'][index]
        elif param is OptimizationParameter.OIL_FTP_PORTION:
            max_value = dict_optimization['max'][index]
        elif param is OptimizationParameter.GAS_FTP_PORTION:
            max_value = dict_optimization['max'][index]
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
                result_psc_obj, _ = adjust_contract(contract=psc,
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

    # Converting the list of enum into list of str enum value
    list_str = [enum_value.value for enum_value in list_params]

    # # Printing for debugging
    # print(list_str)
    # print(list_params_value)
    # print(result_optimization)

    return list_str, list_params_value, result_optimization
