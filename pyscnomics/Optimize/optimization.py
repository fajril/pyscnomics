import numpy as np
# from scipy.optimize import minimize
from scipy.optimize import minimize

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
                 ):
    # Changing the Optimization selection from Enum to string in order to retrieve the result from summary dictionary
    if target_parameter is OptimizationTarget.IRR:
        target_parameter = 'ctr_irr'

    elif target_parameter is OptimizationTarget.NPV:
        target_parameter = 'ctr_npv'

    elif target_parameter is OptimizationTarget.PV_RATIO:
        target_parameter = 'ctr_pv_ratio'

    else:
        raise OptimizationException(
            f"target {target_parameter} "
            f"is should be one of: {[OptimizationTarget.value]}"
        )

    # Changing the parameters list[str] into list[OptimizationParameters(Enum)]
    list_params = [OptimizationParameter[param] for param in dict_optimization['parameters']]

    # Defining empty list to contain value of optimized parameters
    list_params_value = [None] * len(list_params)

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
            bounds = [(dict_optimization['min'][index],
                       dict_optimization['max'][index])]

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
            optim_result = minimize(objective_run, dict_optimization['min'][index], bounds=bounds)

            # Difference value from target and optimization result
            optimized_diff = optim_result.fun

            # Optimized Parameter Value
            optimized_parameter = optim_result.x.item()

            # Appending the result of optimization to the list_params_value
            list_params_value[index] = optimized_parameter

            # # Printing for debugging
            # print('Parameter:', param)
            # print('Optimized Parameter Value:', optimized_parameter)
            # print('Targeted Value:', target_optimization_value)
            # print('Difference of Target and Optimization Result:', optimized_diff)
            # input()

            # Exiting the loop
            break

        else:
            list_params_value[index] = max_value

    print(list_params)
    print(list_params_value)
