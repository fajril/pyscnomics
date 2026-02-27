"""
Configuration to undertake optimization study.
"""

import numpy as np
from scipy.optimize import minimize_scalar

from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.econ.selection import OptimizationParameter, OptimizationTarget
# from pyscnomics.tools.summary import get_summary

from pyscnomics.econ.costs import CapitalCost, Intangible, OPEX, ASR, LBT


class OptimizationException(Exception):
    """Exception to raise for a misuse of BaseProject class"""

    pass


def adjust_contract(
    contract: CostRecovery | GrossSplit,
    contract_arguments: dict,
    summary_argument: dict,
    variable: OptimizationParameter,
    value: float,
    target_parameter: str,
) -> (CostRecovery | GrossSplit, dict):
    """
    Adjust one optimization variable, re-run the contract, and return the updated target metric.

    The selected variable is applied either by modifying contract attributes directly or via
    `adjust_cost_element()` for cost-related adjustments, followed by contract execution.

    Parameters
    ----------
    contract : CostRecovery | GrossSplit
        PSC contract to adjust and execute.
    contract_arguments : dict
        Arguments passed to `contract.run()`.
    summary_argument : dict
        Arguments passed to `contract.get_summary()`.
    variable : OptimizationParameter
        Optimization variable to modify.
    value : float
        New value assigned to the variable.
    target_parameter : str
        Summary key representing the optimization target.

    Returns
    -------
    tuple[float, CostRecovery | GrossSplit]
        (updated target value, executed contract).

    Raises
    ------
    OptimizationException
        If contract type or optimization variable is unsupported.

    TL;DR: Change one parameter → run contract → return updated KPI and contract.
    *KPI = Key Performance Indicator.
    """

    # Optimization parameter is VAT
    if variable is OptimizationParameter.VAT_DISCOUNT:
        contract = adjust_cost_element(
            contract=contract,
            adjustment_value=value,
            adjustment_variable=OptimizationParameter.VAT_DISCOUNT,
        )

    # Optimization parameter is LBT
    elif variable is OptimizationParameter.LBT_DISCOUNT:
        contract = adjust_cost_element(
            contract=contract,
            adjustment_value=value,
            adjustment_variable=OptimizationParameter.LBT_DISCOUNT,
        )

    # Optimization parameter is accelerated depreciation
    elif variable is OptimizationParameter.DEPRECIATION_ACCELERATION:
        contract = adjust_cost_element(
            contract=contract,
            adjustment_value=value,
            adjustment_variable=OptimizationParameter.DEPRECIATION_ACCELERATION,
        )

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
            contract_arguments["vat_rate"] = value

        if variable is OptimizationParameter.EFFECTIVE_TAX_RATE:
            contract_arguments["effective_tax_rate"] = value

    # The condition when contract is Gross Split
    elif isinstance(contract, GrossSplit):
        # Changing the attributes of the contract based on the chosen variable
        if variable is OptimizationParameter.MINISTERIAL_DISCRETION:
            contract.split_ministry_disc = value

        if variable is OptimizationParameter.OIL_DMO_FEE:
            contract.oil_dmo_fee_portion = value

        if variable is OptimizationParameter.GAS_DMO_FEE:
            contract.gas_dmo_fee_portion = value

        if variable is OptimizationParameter.VAT_RATE:
            contract_arguments["vat_rate"] = value

        if variable is OptimizationParameter.EFFECTIVE_TAX_RATE:
            contract_arguments["effective_tax_rate"] = value

    # Running the contract
    contract.run(**contract_arguments)

    # Get the summary of the new contract and get its value of the targeted optimization
    result_psc = contract.get_summary(**summary_argument)[target_parameter]

    return result_psc, contract


def optimize_psc_core(
    contract: CostRecovery | GrossSplit,
    contract_arguments: dict,
    summary_argument: dict,
    target_optimization_value: float,
    dict_optimization: dict,
    target_parameter: OptimizationTarget,
) -> (list, list, float, list):
    """
    Sequential PSC parameter optimization toward a target Key Performance
    Indicator (KPI) value.

    The function iteratively modifies PSC fiscal parameters, executes the
    contract, and checks whether the selected optimization target
    (IRR / NPV / PI) exceeds the desired value. Once the target is reached,
    bounded scalar optimization is performed to find the parameter value
    that minimizes the deviation from the target KPI.

    Optimization stops immediately after the first successful parameter
    achieving the target.

    Parameters
    ----------
    contract : CostRecovery | GrossSplit
        Base PSC contract object.
    contract_arguments : dict
        Arguments required to execute the contract model.
    summary_argument : dict
        Arguments used to compute summary/KPI results.
    target_optimization_value : float
        Desired KPI target value.
    dict_optimization : dict
        Optimization configuration containing:
        - parameter : list[OptimizationParameter]
        - min : lower bounds
        - max : upper bounds
    target_parameter : OptimizationTarget
        KPI used as optimization objective (IRR, NPV, PI).

    Returns
    -------
    list[str]
        Names of optimized parameters.
    list
        Final parameter values ("Base Value" or optimized value).
    float
        Resulting KPI after optimization.
    list
        Executed contract objects for each evaluated step.

    Notes
    -----
    • Parameters are optimized sequentially (not jointly).
    • Each parameter first tested at its bound (min/max rule).
    • `scipy.optimize.minimize_scalar(method="bounded")`
      is used once the target is achievable.
    • Loop exits after first successful optimization.

    TL;DR:
    Try parameters one-by-one → run PSC → if target reachable,
    scalar-optimize that parameter and return updated KPI + contracts.
    """

    # Changing the Optimization selection from Enum to string in order to retrieve
    # the result from summary dictionary
    target_param_mapping = {
        OptimizationTarget.IRR: "ctr_irr",
        OptimizationTarget.NPV: "ctr_npv",
        OptimizationTarget.PI: "ctr_pi",
    }

    try:
        target_param = target_param_mapping[target_parameter]
    except KeyError:
        raise OptimizationException(f"Unrecognized target parameter: {target_parameter}.")

    # Changing the parameters list[str] into list[OptimizationParameters(Enum)]
    list_params = dict_optimization["parameter"]

    # Defining Base Value list to contain value of optimized parameters and
    # status of the optimization: repeat "Base Value" N times
    list_params_value = ["Base Value"] * len(list_params)

    # Defining the empty result of optimization target, which will be defined later
    result_optimization = None

    # Defining the executed contract list
    list_executed_contract = []

    psc = contract

    # Mapping for optimum value
    param_bound_source = {
        OptimizationParameter.OIL_CTR_PRETAX: "max",
        OptimizationParameter.GAS_CTR_PRETAX: "max",
        OptimizationParameter.OIL_FTP_PORTION: "min",
        OptimizationParameter.GAS_FTP_PORTION: "min",
        OptimizationParameter.OIL_IC: "max",
        OptimizationParameter.GAS_IC: "max",
        OptimizationParameter.OIL_DMO_FEE: "max",
        OptimizationParameter.GAS_DMO_FEE: "max",
        OptimizationParameter.VAT_RATE: "min",
        OptimizationParameter.EFFECTIVE_TAX_RATE: "min",
        OptimizationParameter.MINISTERIAL_DISCRETION: "max",
        OptimizationParameter.VAT_DISCOUNT: "max",
        OptimizationParameter.LBT_DISCOUNT: "max",
        OptimizationParameter.DEPRECIATION_ACCELERATION: "max",
    }

    for index, param in enumerate(list_params):

        # Specify the "optimum" value of a target parameter
        # Optimum: choose between maximum or minimum value for a target parameter.
        try:
            bound_key = param_bound_source[param]
        except KeyError:
            raise OptimizationException(
                f"Invalid optimization parameter: ({param}). "
                f"Optimization parameter should be chosen from OptimizationParameter enum."
            )

        max_value = dict_optimization[bound_key][index]

        # Modify contract parameter value based on the given input
        result_psc, psc = adjust_contract(
            contract=psc,
            contract_arguments=contract_arguments,
            summary_argument=summary_argument,
            variable=param,
            value=max_value,
            target_parameter=target_param,
        )

        # Able to conduct optimization since the result is greater than the target
        if result_psc > target_optimization_value:
            # Defining the upper and lower limit of the optimized variable
            bounds = (dict_optimization["min"][index], dict_optimization["max"][index])

            def objective_run(new_value):
                """
                A helper function to execute an updated contract
                """
                result_psc_obj, executed_contract = adjust_contract(
                    contract=psc,
                    contract_arguments=contract_arguments,
                    summary_argument=summary_argument,
                    variable=param,
                    value=new_value,
                    target_parameter=target_param,
                )

                result_obj = abs(result_psc_obj - target_optimization_value)

                return result_obj

            # Optimization of the objective function
            optim_result = minimize_scalar(
                objective_run, bounds=bounds, method="bounded"
            )

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
            executed_contract = adjust_contract(
                contract=psc,
                contract_arguments=contract_arguments,
                summary_argument=summary_argument,
                variable=param,
                value=optimized_parameter,
                target_parameter=target_param,
            )[1]

            # Filling the list with executed contract
            list_executed_contract.append(executed_contract)

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

    return list_str, list_params_value, result_optimization, list_executed_contract


def adjust_cost_element(
    contract: CostRecovery | GrossSplit,
    adjustment_value: float = 1.0,
    adjustment_variable: OptimizationParameter = OptimizationParameter.VAT_DISCOUNT,
) -> CostRecovery | GrossSplit:
    """
    Adjust selected cost elements of a PSC contract.

    Applies an optimization factor to a chosen cost component and returns
    a reconstructed contract with updated costs.

    Supported adjustments:
    - VAT_DISCOUNT → scales VAT discount across cost elements
    - LBT_DISCOUNT → scales LBT discount
    - DEPRECIATION_ACCELERATION → shortens capital useful life

    Parameters
    ----------
    contract : CostRecovery | GrossSplit
        PSC contract to modify.
    adjustment_value : float, default=1.0
        Adjustment factor ∈ [0, 1].
    adjustment_variable : OptimizationParameter
        Target cost component to adjust.

    Returns
    -------
    CostRecovery | GrossSplit
        New contract instance with adjusted cost elements.

    Raises
    ------
    OptimizationException
        - invalid adjustment_value
        - unsupported adjustment_variable
        - unsupported contract type

    TL;DR: Modify one cost component (VAT, LBT, or depreciation) and rebuild the contract.
    """

    # Check for unsuitable "adjustment_value" parameter
    if not 0.0 <= adjustment_value <= 1.0:
        raise OptimizationException(
            f"Adjustment value {adjustment_value} is out of valid range [0, 1]"
        )

    capital_adjusted = contract.capital_cost
    intangible_adjusted = contract.intangible_cost
    opex_adjusted = contract.opex
    asr_adjusted = contract.asr_cost
    lbt_adjusted = contract.lbt_cost
    cos_adjusted = contract.cost_of_sales

    # Condition when the VAT of each cost element will be adjusted
    if adjustment_variable == OptimizationParameter.VAT_DISCOUNT:
        # Adjusting the Capital Cost of the contract
        capital_adjusted = tuple(
            [
                CapitalCost(
                    start_year=cap.start_year,
                    end_year=cap.end_year,
                    cost=cap.cost,
                    expense_year=cap.expense_year,
                    cost_allocation=cap.cost_allocation,
                    cost_type=cap.cost_type,
                    description=cap.description,
                    tax_portion=cap.tax_portion,
                    tax_discount=adjustment_value,
                    pis_year=cap.pis_year,
                    salvage_value=cap.salvage_value,
                    useful_life=cap.useful_life,
                    depreciation_factor=cap.depreciation_factor,
                    is_ic_applied=cap.is_ic_applied,
                )
                for cap in contract.capital_cost
            ]
        )

        # Adjusting the Intangible cost of the contract
        intangible_adjusted = tuple(
            [
                Intangible(
                    start_year=intang.start_year,
                    end_year=intang.end_year,
                    cost=intang.cost,
                    expense_year=intang.expense_year,
                    cost_allocation=intang.cost_allocation,
                    cost_type=intang.cost_type,
                    description=intang.description,
                    tax_portion=intang.tax_portion,
                    tax_discount=adjustment_value,
                )
                for intang in contract.intangible_cost
            ]
        )

        # Adjusting the OPEX cost of the contract
        opex_adjusted = tuple(
            [
                OPEX(
                    start_year=opx.start_year,
                    end_year=opx.end_year,
                    expense_year=opx.expense_year,
                    cost_allocation=opx.cost_allocation,
                    cost_type=opx.cost_type,
                    description=opx.description,
                    tax_portion=opx.tax_portion,
                    tax_discount=adjustment_value,
                    fixed_cost=opx.fixed_cost,
                    prod_rate=opx.prod_rate,
                    cost_per_volume=opx.cost_per_volume,
                )
                for opx in contract.opex
            ]
        )

        # Adjusting the ASR cost of the contract
        asr_adjusted = tuple(
            [
                ASR(
                    start_year=asr.start_year,
                    end_year=asr.end_year,
                    cost=asr.cost,
                    expense_year=asr.expense_year,
                    cost_allocation=asr.cost_allocation,
                    cost_type=asr.cost_type,
                    description=asr.description,
                    tax_portion=asr.tax_portion,
                    tax_discount=adjustment_value,
                    final_year=asr.final_year,
                    future_rate=asr.future_rate,
                )
                for asr in contract.asr_cost
            ]
        )

    elif adjustment_variable == OptimizationParameter.LBT_DISCOUNT:
        # Adjusting the LBT of the contract
        lbt_adjusted = tuple(
            [
                LBT(
                    start_year=bt.start_year,
                    end_year=bt.end_year,
                    expense_year=bt.expense_year,
                    cost=bt.cost,
                    cost_allocation=bt.cost_allocation,
                    cost_type=bt.cost_type,
                    description=bt.description,
                    tax_portion=bt.tax_portion,
                    tax_discount=adjustment_value,
                    final_year=bt.final_year,
                    utilized_land_area=bt.utilized_land_area,
                    utilized_building_area=bt.utilized_building_area,
                    njop_land=bt.njop_land,
                    njop_building=bt.njop_building,
                    gross_revenue=bt.gross_revenue,
                )
                for bt in contract.lbt_cost
            ]
        )

    elif adjustment_variable == OptimizationParameter.DEPRECIATION_ACCELERATION:
        # Adjusting the useful life of the capital cost of the contract
        capital_adjusted = tuple(
            [
                CapitalCost(
                    start_year=tan.start_year,
                    end_year=tan.end_year,
                    cost=tan.cost,
                    expense_year=tan.expense_year,
                    cost_allocation=tan.cost_allocation,
                    cost_type=tan.cost_type,
                    description=tan.description,
                    tax_portion=tan.tax_portion,
                    tax_discount=tan.tax_discount,
                    pis_year=tan.pis_year,
                    salvage_value=tan.salvage_value,
                    useful_life=adjust_useful_life_years(
                        adjustment_value=adjustment_value,
                        useful_life_array=tan.useful_life,
                    ),
                    depreciation_factor=tan.depreciation_factor,
                    is_ic_applied=tan.is_ic_applied,
                )
                for tan in contract.capital_cost
            ]
        )

    # Condition when the chosen option is not recognized
    else:
        raise OptimizationException(
            f"Adjustment Variable {adjustment_variable} "
            f"do not exist. It should be VAT or LBT in string data type"
        )

    # Onstream date treatment for OIL and/or GAS
    def _resolve_onstream_date(contract, revenue_attr: str, onstream_attr: str):
        revenue = getattr(contract, revenue_attr)
        return (
            None
            if np.allclose(revenue, 0)
            else getattr(contract, onstream_attr)
        )

    oil_onstream_date = _resolve_onstream_date(
        contract=contract, revenue_attr="_oil_revenue", onstream_attr="oil_onstream_date"
    )

    gas_onstream_date = _resolve_onstream_date(
        contract=contract, revenue_attr="_gas_revenue", onstream_attr="gas_onstream_date"
    )

    # Specify base arguments
    kwargs_base = {
        "start_date": contract.start_date,
        "end_date": contract.end_date,
        "oil_onstream_date": oil_onstream_date,
        "gas_onstream_date": gas_onstream_date,
        "approval_year": contract.approval_year,
        "is_pod_1": contract.is_pod_1,
        "is_strict": contract.is_strict,
    }

    # Specify arguments associated with lifting and costs
    kwargs_lifting_costs = {
        "lifting": contract.lifting,
        "capital_cost": capital_adjusted,
        "intangible_cost": intangible_adjusted,
        "opex": opex_adjusted,
        "asr_cost": asr_adjusted,
        "lbt_cost": lbt_adjusted,
        "cost_of_sales": cos_adjusted,
    }

    # When the contract is CostRecovery, parsing back the adjusted cost elements
    # to the cost recovery contract
    if isinstance(contract, CostRecovery):
        kwargs_costrec = {
            # Base parameters
            **kwargs_base,

            # Lifting and costs
            **kwargs_lifting_costs,

            # FTP
            "oil_ftp_is_available": contract.oil_ftp_is_available,
            "oil_ftp_is_shared": contract.oil_ftp_is_shared,
            "oil_ftp_portion": contract.oil_ftp_portion,
            "gas_ftp_is_available": contract.gas_ftp_is_available,
            "gas_ftp_is_shared": contract.gas_ftp_is_shared,
            "gas_ftp_portion": contract.gas_ftp_portion,

            # Tax split
            "tax_split_type": contract.tax_split_type,
            "condition_dict": contract.condition_dict,
            "indicator_rc_icp_sliding": contract.indicator_rc_icp_sliding,
            "oil_ctr_pretax_share": contract.oil_ctr_pretax_share,
            "gas_ctr_pretax_share": contract.gas_ctr_pretax_share,

            # Investment credit
            "oil_ic_rate": contract.oil_ic_rate,
            "gas_ic_rate": contract.gas_ic_rate,
            "ic_is_available": contract.ic_is_available,
            "oil_cr_cap_rate": contract.oil_cr_cap_rate,
            "gas_cr_cap_rate": contract.gas_cr_cap_rate,

            # DMO
            "oil_dmo_volume_portion": contract.oil_dmo_volume_portion,
            "oil_dmo_fee_portion": contract.oil_dmo_fee_portion,
            "oil_dmo_holiday_duration": contract.oil_dmo_holiday_duration,
            "gas_dmo_volume_portion": contract.gas_dmo_volume_portion,
            "gas_dmo_fee_portion": contract.gas_dmo_fee_portion,
            "gas_dmo_holiday_duration": contract.gas_dmo_holiday_duration,

            # Carry forward depreciation
            "oil_carry_forward_depreciation": contract.oil_carry_forward_depreciation,
            "gas_carry_forward_depreciation": contract.gas_carry_forward_depreciation,
        }

        return CostRecovery(**kwargs_costrec)

    # When the contract is GrossSplit, parsing back the adjusted cost elements
    # to the gross split contract
    elif isinstance(contract, GrossSplit):
        kwargs_gs = {
            # Base parameters
            **kwargs_base,

            # Lifting and costs
            **kwargs_lifting_costs,

            # Field and reservoir properties
            "field_status": contract.field_status,
            "field_loc": contract.field_loc,
            "res_depth": contract.res_depth,
            "infra_avail": contract.infra_avail,
            "res_type": contract.res_type,
            "api_oil": contract.api_oil,
            "domestic_use": contract.domestic_use,
            "prod_stage": contract.prod_stage,
            "co2_content": contract.co2_content,
            "h2s_content": contract.h2s_content,
            "field_reserves_2024": contract.field_reserves_2024,
            "infra_avail_2024": contract.infra_avail_2024,
            "field_loc_2024": contract.field_loc_2024,

            # Ministry discretion
            "split_ministry_disc": contract.split_ministry_disc,

            # DMO
            "oil_dmo_volume_portion": contract.oil_dmo_volume_portion,
            "oil_dmo_fee_portion": contract.oil_dmo_fee_portion,
            "oil_dmo_holiday_duration": contract.oil_dmo_holiday_duration,
            "gas_dmo_volume_portion": contract.gas_dmo_volume_portion,
            "gas_dmo_fee_portion": contract.gas_dmo_fee_portion,
            "gas_dmo_holiday_duration": contract.gas_dmo_holiday_duration,

            # Carry forward depreciation
            "oil_carry_forward_depreciation": contract.oil_carry_forward_depreciation,
            "gas_carry_forward_depreciation": contract.gas_carry_forward_depreciation,
        }

        return GrossSplit(**kwargs_gs)

    # When the contract is not recognized, raise an exception
    else:
        raise OptimizationException(
            f"Contract type {type(contract)}, is not a valid contract "
            f"for optimization module"
        )


def adjust_useful_life_years(
    adjustment_value: float,
    useful_life_array: np.ndarray
) -> np.ndarray:
    """
    Adjust useful life using a depreciation acceleration factor.

    Each useful life value is scaled by (1 - adjustment_value) → rounded up
    to the nearest year → enforced to be ≥ 2 years.

    Parameters
    ----------
    adjustment_value : float
        Depreciation acceleration factor ∈ [0, 1].
    useful_life_array : np.ndarray
        Original useful life values (years).

    Returns
    -------
    np.ndarray
        Adjusted useful life (int years), ceil-rounded and floored at 2.

    Raises
    ------
    OptimizationException
        - adjustment_value outside [0, 1]
        - any useful life < 2 years

    TL;DR: Scale useful life by (1 − factor), ceil it, and never allow < 2 years.
    """

    # Check for unsuitable "adjustment_value" input
    if not 0.0 <= adjustment_value <= 1.0:
        raise OptimizationException(
            f"Adjustment value {adjustment_value} is out of valid range [0, 1]"
        )

    # Does not allow optimization if useful life is below 2 years
    below_mask = useful_life_array < 2

    if np.all(below_mask):
        raise OptimizationException(
            f"Cannot optimize useful life for project duration < 2 years. "
            f"Please remove 'depreciation_acceleration' as target optimization parameter."
        )

    if np.any(below_mask):
        values_below = useful_life_array[below_mask]
        raise OptimizationException(
            f"Useful life contains values below 2 years: {values_below}"
        )

    # Adjust "useful_life" with "acceleration_rate", ensuring at least 2 years
    min_useful_life = 2
    accel_rate = 1 - adjustment_value

    adjusted_useful_life = np.ceil(useful_life_array * accel_rate)
    adjusted_useful_life = np.maximum(adjusted_useful_life, min_useful_life)

    return adjusted_useful_life.astype(int)


def optimize_psc(
    dict_optimization: dict,
    contract: CostRecovery | GrossSplit,
    contract_arguments: dict,
    target_optimization_value: float,
    summary_arguments: dict,
    target_parameter: OptimizationTarget = OptimizationTarget.IRR,
) -> (list, list, float, list):
    """
    Run PSC optimization workflow including multi-value parameter handling.

    This function performs end-to-end optimization of a PSC contract by:

    • evaluating the base-case economics
    • optionally collapsing multi-value inputs (e.g., array VAT) into a
      single representative optimization variable
    • running ``optimize_psc_core`` to reach the target economic indicator
    • restoring optimized values back to their original array structure
    • re-running the contract when transformed arguments are required

    Supports both Cost Recovery and Gross Split contracts.

    Parameters
    ----------
    dict_optimization : dict
        Optimization configuration containing:
        • parameter → list[OptimizationParameter]
        • min → lower bounds
        • max → upper bounds
    contract : CostRecovery | GrossSplit
        PSC contract instance to be optimized.
    contract_arguments : dict
        Arguments passed to ``contract.run()``.
    target_optimization_value : float
        Desired value of the selected economic indicator.
    summary_arguments : dict
        Arguments passed to ``contract.get_summary()``.
    target_parameter : OptimizationTarget, default=OptimizationTarget.IRR
        Economic indicator used as optimization target
        (IRR, NPV, or PI).

    Returns
    -------
    list[str]
        Names of optimized parameters.
    list
        Optimized parameter values (or "Base Value" if unchanged).
    float
        Final optimized value of the selected economic indicator.
    list
        Sequence of executed contract states during optimization.

    Raises
    ------
    OptimizationException
        If the optimization target or configuration is invalid.

    Notes
    -----
    • Multi-value inputs (e.g., VAT arrays) are temporarily reduced to a
      scalar optimization problem and later reconstructed.
    • Final contract execution reflects transformed arguments when
      proportional scaling is required.

    TL;DR:
    Full PSC optimization pipeline: baseline → core optimization →
    reconstruct multi-values → return optimized economics + contracts.
    """

    # Get the summary of the base case
    contract.run(**contract_arguments)
    summary_base = contract.get_summary(**summary_arguments)

    # Retrieve the economic indicator of the base case corresponding to the chosen indicator
    _target_key_mapping = {
        OptimizationTarget.IRR: "ctr_irr",
        OptimizationTarget.NPV: "ctr_npv",
        OptimizationTarget.PI: "ctr_pi",
    }

    try:
        target_value_base = summary_base[_target_key_mapping[target_parameter]]
    except KeyError:
        raise OptimizationException(f"Optimization target ({target_parameter}) is not recognized.")

    # Defining the dictionary that could be in multi-values
    pseudo_dict = {
        "key": ["vat_rate"],
        "selection": [OptimizationParameter.VAT_RATE],
        "index_in_parameter": [],
        "value": [],
    }

    # Initiating new contract argument which will be adjusted in the following loop
    contract_arguments_new = contract_arguments.copy()

    # Checking the existence of the multi-values optimization parameter
    for key, selection in zip(pseudo_dict["key"], pseudo_dict["selection"]):
        if not (
            key in contract_arguments
            and isinstance(contract_arguments[key], np.ndarray)
            and selection in dict_optimization["parameter"]
        ):
            continue

        # Get the index of the min and max of the selection values and
        # store it in the pseudo_dict
        index_pseudo = dict_optimization["parameter"].index(selection)
        pseudo_dict["index_in_parameter"].append(index_pseudo)

        # Specify base optimization scenario
        dict_opt_pseudo = {
            "parameter": [selection],
            "min": np.array([dict_optimization["min"][index_pseudo]], dtype=float),
            "max": np.array([dict_optimization["max"][index_pseudo]], dtype=float),
        }

        # Copy the contract and summary argument to avoid argument overwriting
        contract_arguments_pseudo = contract_arguments.copy()
        summary_argument_pseudo = summary_arguments.copy()

        # Retrieving the result_optim_pseudo which will be used
        # as the base for resul_optim_new
        optim_base_result = optimize_psc_core(
            dict_optimization=dict_opt_pseudo,
            contract=contract,
            contract_arguments=contract_arguments_pseudo,
            target_optimization_value=target_value_base,
            summary_argument=summary_argument_pseudo,
            target_parameter=target_parameter,
        )

        variable_value_pseudo = optim_base_result[1][0]

        # Replacing the original multiple variable value into single value of
        # variable_value_pseudo
        contract_arguments_new[key] = variable_value_pseudo

        # Storing the variable_value_pseudo into the pseudo_dict
        pseudo_dict["value"].append(variable_value_pseudo)

    # Get the optimization of the contract
    result = optimize_psc_core(
        dict_optimization=dict_optimization,
        contract=contract,
        contract_arguments=contract_arguments_new,
        target_optimization_value=target_optimization_value,
        summary_argument=summary_arguments,
        target_parameter=target_parameter,
    )

    list_str = result[0]
    list_params_value = result[1]
    result_optimization = result[2]
    list_executed_contract = result[3]

    contract = list_executed_contract[-1]

    # Initiating the adjusted contract arguments
    contract_arguments_adjusted = contract_arguments.copy()

    for key, pseudo_value, index in zip(
        pseudo_dict["key"],
        pseudo_dict["value"],
        pseudo_dict["index_in_parameter"],
    ):
        # The condition when the optimization value is "Base Value":
        # do not need to change the form
        if isinstance(list_params_value[index], str):
            continue

        # Skip if not eligible for transformation
        if not (
            isinstance(list_params_value[index], (float, int, np.ndarray))
            and key in contract_arguments
        ):
            continue

        # Defining the factor of transformation: VAT_New / VAT_Old
        factor = np.divide(
            list_params_value[index],
            pseudo_value,
            where=list_params_value[index] != 0,
        )

        # Defining the proportioned argument based on the obtained factor:
        # factor * VATi
        transformed_value = contract_arguments[key] * np.full_like(
            contract_arguments_new[key], fill_value=factor, dtype=float
        )

        # Deforming the array into a list due to the consistency of
        # the optimization result
        list_params_value[index] = transformed_value.tolist()
        contract_arguments_adjusted[key] = transformed_value

    # Running the contract using the adjusted contract argument when
    # the VAT is multi values
    if len(pseudo_dict["index_in_parameter"]) > 0:

        contract.run(**contract_arguments_adjusted)

        #  Replacing the executed contract from the optimization function
        #  into the adjusted contract
        list_executed_contract[-1] = contract

        # Retrieving the summary of the contract
        summary_optimized = contract.get_summary(**summary_arguments)

        #  Retrieving the corresponding target value
        if target_parameter == OptimizationTarget.IRR:
            result_optimization = summary_optimized["ctr_irr"]
        elif target_parameter == OptimizationTarget.NPV:
            result_optimization = summary_optimized["ctr_npv"]
        elif target_parameter == OptimizationTarget.PI:
            result_optimization = summary_optimized["ctr_pi"]
        else:
            raise OptimizationException(
                f"Optimization Target {target_parameter} is not recognized"
            )

    return list_str, list_params_value, result_optimization, list_executed_contract


"""
FORMER APPROACHES
+++++++++++++++++

def optimize_psc(
    dict_optimization: dict,
    contract: CostRecovery | GrossSplit,
    contract_arguments: dict,
    target_optimization_value: float,
    summary_argument: dict,
    target_parameter: OptimizationTarget = OptimizationTarget.IRR,
) -> (list, list, float, list):

    # Get the summary of the base case
    contract.run(**contract_arguments)
    summary_base = contract.get_summary(**summary_argument)
    
    # Former approach
    # ---------------
    # summary_argument["contract"] = contract
    # summary_base = get_summary(**summary_argument)
    
    # Retrieve the economic indicator of the base case corresponding to the chosen indicator
    if target_parameter == OptimizationTarget.IRR:
        target_value_base = summary_base["ctr_irr"]
    elif target_parameter == OptimizationTarget.NPV:
        target_value_base = summary_base["ctr_npv"]
    elif target_parameter == OptimizationTarget.PI:
        target_value_base = summary_base["ctr_pi"]
    else:
        raise OptimizationException(
            f"Optimization Target {target_parameter} is not recognized"
        )

    # Defining the dictionary that could be in multi-values
    pseudo_dict = {
        "key": ["vat_rate"],
        "selection": [OptimizationParameter.VAT_RATE],
        "index_in_parameter": [],
        "value": [],
    }

    # Initiating new contract argument which will be adjusted in the following loop
    contract_arguments_new = contract_arguments.copy()

    # Checking the existence of the multi-values optimization parameter
    for key, selection in zip(pseudo_dict["key"], pseudo_dict["selection"]):
        if (
            key in contract_arguments.keys()
            and isinstance(contract_arguments[key], np.ndarray)
            and selection in dict_optimization["parameter"]
        ):

            # Get the index of the min and max of the selection values and
            # store it in the pseudo_dict
            index_pseudo = dict_optimization["parameter"].index(selection)
            pseudo_dict["index_in_parameter"].append(index_pseudo)

            # Defining the dictionary of base optimization
            dict_opt_pseudo = {
                "parameter": [selection],
                "min": np.array([dict_optimization["min"][index_pseudo]], dtype=float),
                "max": np.array([dict_optimization["max"][index_pseudo]], dtype=float),
            }

            # Copying the contract and summary argument to avoid argument overwriting
            contract_arguments_pseudo = contract_arguments.copy()
            summary_argument_pseudo = summary_argument.copy()

            # Retrieving the result_optim_pseudo which it will be used
            # as base for resul_optim_new
            optim_base_result = optimize_psc_core(
                dict_optimization=dict_opt_pseudo,
                contract=contract,
                contract_arguments=contract_arguments_pseudo,
                target_optimization_value=target_value_base,
                summary_argument=summary_argument_pseudo,
                target_parameter=target_parameter,
            )

            variable_value_pseudo = optim_base_result[1][0]

            # Replacing the original multiple variable value into single value of
            # variable_value_pseudo
            contract_arguments_new[key] = variable_value_pseudo

            # Storing the variable_value_pseudo into the pseudo_dict
            pseudo_dict["value"].append(variable_value_pseudo)

        else:
            pass

    # Get the optimization of the contract
    result = optimize_psc_core(
        dict_optimization=dict_optimization,
        contract=contract,
        contract_arguments=contract_arguments_new,
        target_optimization_value=target_optimization_value,
        summary_argument=summary_argument,
        target_parameter=target_parameter,
    )

    list_str = result[0]
    list_params_value = result[1]
    result_optimization = result[2]
    list_executed_contract = result[3]

    contract = list_executed_contract[-1]

    # Initiating the adjusted contract arguments
    contract_arguments_adjusted = contract_arguments.copy()

    # Iterating in order to the single argument back into the original form (array)
    for key, pseudo_value, index in zip(
        pseudo_dict["key"], pseudo_dict["value"], pseudo_dict["index_in_parameter"]
    ):
        # The condition when the optimization value is "Base Value",
        # do not need to change the form
        if isinstance(list_params_value[index], str):
            pass

        # The condition when the optimization value is not "Base Value"
        elif (
            isinstance(list_params_value[index], (float, int, np.ndarray))
            and key in contract_arguments.keys()
        ):
            #  Defining the factor of transformation : VAT_New / VAT_Old
            factor = np.divide(
                list_params_value[index],
                pseudo_value,
                where=list_params_value[index] != 0,
            )

            # Defining the proportioned argument based on the obtained factor:
            # factor * VATi
            transformed_value = contract_arguments[key] * np.full_like(
                contract_arguments_new[key], fill_value=factor, dtype=float
            )

            # Deforming the array into a list due to the consistency of
            # the optimization result
            list_params_value[index] = transformed_value.tolist()

            contract_arguments_adjusted[key] = transformed_value

        # Pass the other condition
        else:
            pass

    # Running the contract using the adjusted contract argument when
    # the VAT is multi values
    if len(pseudo_dict["index_in_parameter"]) > 0:

        contract.run(**contract_arguments_adjusted)

        #  Replacing the executed contract from the optimization function
        #  into the adjusted contract
        list_executed_contract[-1] = contract

        # Retrieving the summary of the contract
        summary_optimized = contract.get_summary(**summary_argument)

        # Former approach
        # ---------------
        # summary_argument["contract"] = contract
        # summary_optimized = get_summary(**summary_argument)
        
        #  Retrieving the corresponding target value
        if target_parameter == OptimizationTarget.IRR:
            result_optimization = summary_optimized["ctr_irr"]
        elif target_parameter == OptimizationTarget.NPV:
            result_optimization = summary_optimized["ctr_npv"]
        elif target_parameter == OptimizationTarget.PI:
            result_optimization = summary_optimized["ctr_pi"]
        else:
            raise OptimizationException(
                f"Optimization Target {target_parameter} is not recognized"
            )

    return list_str, list_params_value, result_optimization, list_executed_contract
    
================================================================================================

def adjust_useful_life_years(
    adjustment_value: float,
    useful_life_array: np.ndarray
) -> np.ndarray:

    # Check for unsuitable "adjustment_value" input
    if not 0.0 <= adjustment_value <= 1.0:
        raise OptimizationException(
            f"Adjustment value {adjustment_value} is out of valid range [0, 1]"
        )

    # Catch the useful life below 2 years
    below_mask = useful_life_array < 2
    if np.any(below_mask):
        values_below = useful_life_array[below_mask]
        raise OptimizationException(
            f"Useful life contains values below 2 years: {values_below}"
        )
        
    # Former approach
    # ---------------
    # index_below = np.argwhere(useful_life_array < 2).ravel()
    # if len(index_below) > 0:
    #     raise OptimizationException(
    #         f"Useful life at index {index_below}, is/are below 2 years"
    #     )
    
    # Adjust "useful_life" with "acceleration_rate", ensuring at least 2 years
    min_useful_life = 2
    accel_rate = 1 - adjustment_value

    adjusted_useful_life = np.ceil(useful_life_array * accel_rate)
    adjusted_useful_life = np.maximum(adjusted_useful_life, min_useful_life)

    # Former approach
    # ---------------
    # # Defining the acceleration rate
    # accel_rate = 1 - adjustment_value
    # 
    # # Defining the maximum accelerated useful life years
    # maximum_useful_life = np.full_like(useful_life_array, fill_value=2, dtype=int)
    # 
    # # Multiplying the useful life with the acceleration rate
    # adjusted_useful_life = np.where(
    #     useful_life_array == maximum_useful_life,
    #     maximum_useful_life,
    #     useful_life_array * accel_rate,
    # )
    # 
    # # Rounding up the adjusted use full life
    # adjusted_useful_life = np.ceil(adjusted_useful_life)
    # 
    # # Catch the useful life below 2 years
    # adjusted_useful_life = np.where(
    #     adjusted_useful_life < maximum_useful_life,
    #     maximum_useful_life,
    #     adjusted_useful_life,
    # )
    
    return adjusted_useful_life.astype(int)
    
================================================================================================

def adjust_cost_element(
    contract: CostRecovery | GrossSplit,
    adjustment_value: float = 1.0,
    adjustment_variable: OptimizationParameter = OptimizationParameter.VAT_DISCOUNT,
) -> CostRecovery | GrossSplit:

    # Check for unsuitable "adjustment_value" parameter
    if not 0.0 <= adjustment_value <= 1.0:
        raise OptimizationException(
            f"Adjustment value {adjustment_value} is out of valid range [0, 1]"
        )

    capital_adjusted = contract.capital_cost
    intangible_adjusted = contract.intangible_cost
    opex_adjusted = contract.opex
    asr_adjusted = contract.asr_cost
    lbt_adjusted = contract.lbt_cost
    cos_adjusted = contract.cost_of_sales

    # Condition when the VAT of each cost element will be adjusted
    if adjustment_variable == OptimizationParameter.VAT_DISCOUNT:
        # Adjusting the Capital Cost of the contract
        capital_adjusted = tuple(
            [
                CapitalCost(
                    start_year=tan.start_year,
                    end_year=tan.end_year,
                    cost=tan.cost,
                    expense_year=tan.expense_year,
                    cost_allocation=tan.cost_allocation,
                    cost_type=tan.cost_type,
                    description=tan.description,
                    tax_portion=tan.tax_portion,
                    tax_discount=adjustment_value,
                    pis_year=tan.pis_year,
                    salvage_value=tan.salvage_value,
                    useful_life=tan.useful_life,
                    depreciation_factor=tan.depreciation_factor,
                    is_ic_applied=tan.is_ic_applied,
                )
                for tan in contract.capital_cost
            ]
        )

        # Adjusting the Intangible cost of the contract
        intangible_adjusted = tuple(
            [
                Intangible(
                    start_year=intang.start_year,
                    end_year=intang.end_year,
                    cost=intang.cost,
                    expense_year=intang.expense_year,
                    cost_allocation=intang.cost_allocation,
                    cost_type=intang.cost_type,
                    description=intang.description,
                    tax_portion=intang.tax_portion,
                    tax_discount=adjustment_value,
                )
                for intang in contract.intangible_cost
            ]
        )

        # Adjusting the OPEX cost of the contract
        opex_adjusted = tuple(
            [
                OPEX(
                    start_year=opx.start_year,
                    end_year=opx.end_year,
                    expense_year=opx.expense_year,
                    cost_allocation=opx.cost_allocation,
                    cost_type=opx.cost_type,
                    description=opx.description,
                    tax_portion=opx.tax_portion,
                    tax_discount=adjustment_value,
                    fixed_cost=opx.fixed_cost,
                    prod_rate=opx.prod_rate,
                    cost_per_volume=opx.cost_per_volume,
                )
                for opx in contract.opex
            ]
        )

        # Adjusting the ASR cost of the contract
        asr_adjusted = tuple(
            [
                ASR(
                    start_year=asr.start_year,
                    end_year=asr.end_year,
                    cost=asr.cost,
                    expense_year=asr.expense_year,
                    cost_allocation=asr.cost_allocation,
                    cost_type=asr.cost_type,
                    description=asr.description,
                    tax_portion=asr.tax_portion,
                    tax_discount=adjustment_value,
                    final_year=asr.final_year,
                    future_rate=asr.future_rate,
                )
                for asr in contract.asr_cost
            ]
        )

    elif adjustment_variable == OptimizationParameter.LBT_DISCOUNT:
        # Adjusting the LBT of the contract
        lbt_adjusted = tuple(
            [
                LBT(
                    start_year=bt.start_year,
                    end_year=bt.end_year,
                    expense_year=bt.expense_year,
                    cost=bt.cost,
                    cost_allocation=bt.cost_allocation,
                    cost_type=bt.cost_type,
                    description=bt.description,
                    tax_portion=bt.tax_portion,
                    tax_discount=adjustment_value,
                    final_year=bt.final_year,
                    utilized_land_area=bt.utilized_land_area,
                    utilized_building_area=bt.utilized_building_area,
                    njop_land=bt.njop_land,
                    njop_building=bt.njop_building,
                    gross_revenue=bt.gross_revenue,
                )
                for bt in contract.lbt_cost
            ]
        )

    elif adjustment_variable == OptimizationParameter.DEPRECIATION_ACCELERATION:
        # Adjusting the useful life of the capital cost of the contract
        capital_adjusted = tuple(
            [
                CapitalCost(
                    start_year=tan.start_year,
                    end_year=tan.end_year,
                    cost=tan.cost,
                    expense_year=tan.expense_year,
                    cost_allocation=tan.cost_allocation,
                    cost_type=tan.cost_type,
                    description=tan.description,
                    tax_portion=tan.tax_portion,
                    tax_discount=tan.tax_discount,
                    pis_year=tan.pis_year,
                    salvage_value=tan.salvage_value,
                    useful_life=adjust_useful_life_years(
                        adjustment_value=adjustment_value,
                        useful_life_array=tan.useful_life,
                    ),
                    depreciation_factor=tan.depreciation_factor,
                    is_ic_applied=tan.is_ic_applied,
                )
                for tan in contract.capital_cost
            ]
        )

    # Condition when the chosen option is not recognized
    else:
        raise OptimizationException(
            f"Adjustment Variable {adjustment_variable} "
            f"do not exist. It should be VAT or LBT in string data type"
        )

    # Onstream date treatment (OIL)
    if np.sum(getattr(contract, f"_oil_revenue")) == 0:
        oil_onstream_date = None
    else:
        oil_onstream_date = contract.oil_onstream_date

    # Onstream date treatment (GAS)
    if np.sum(getattr(contract, f"_gas_revenue")) == 0:
        gas_onstream_date = None
    else:
        gas_onstream_date = contract.gas_onstream_date

    # Former approach
    # ---------------
    # if np.sum(contract._oil_revenue) == 0:
    #     oil_onstream_date = None
    # 
    # else:
    #     oil_onstream_date = contract.oil_onstream_date
    # 
    # if np.sum(contract._gas_revenue) == 0:
    #     gas_onstream_date = None
    # else:
    #     gas_onstream_date = contract.gas_onstream_date
    
    # Specify base arguments
    kwargs_base = {
        "start_date": contract.start_date,
        "end_date": contract.end_date,
        "oil_onstream_date": oil_onstream_date,
        "gas_onstream_date": gas_onstream_date,
        "approval_year": contract.approval_year,
        "is_pod_1": contract.is_pod_1,
    }

    # Specify arguments associated with lifting and costs
    kwargs_lifting_costs = {
        "lifting": contract.lifting,
        "capital_cost": capital_adjusted,
        "intangible_cost": intangible_adjusted,
        "opex": opex_adjusted,
        "asr_cost": asr_adjusted,
        "lbt_cost": lbt_adjusted,
        "cost_of_sales": cos_adjusted,
    }

    # When the contract is CostRecovery, parsing back the adjusted cost elements
    # to the cost recovery contract
    if isinstance(contract, CostRecovery):
        kwargs_costrec = {
            # Base parameters
            **kwargs_base,

            # Lifting and costs
            **kwargs_lifting_costs,

            # FTP
            "oil_ftp_is_available": contract.oil_ftp_is_available,
            "oil_ftp_is_shared": contract.oil_ftp_is_shared,
            "oil_ftp_portion": contract.oil_ftp_portion,
            "gas_ftp_is_available": contract.gas_ftp_is_available,
            "gas_ftp_is_shared": contract.gas_ftp_is_shared,
            "gas_ftp_portion": contract.gas_ftp_portion,

            # Tax split
            "tax_split_type": contract.tax_split_type,
            "condition_dict": contract.condition_dict,
            "indicator_rc_icp_sliding": contract.indicator_rc_icp_sliding,
            "oil_ctr_pretax_share": contract.oil_ctr_pretax_share,
            "gas_ctr_pretax_share": contract.gas_ctr_pretax_share,

            # Investment credit
            "oil_ic_rate": contract.oil_ic_rate,
            "gas_ic_rate": contract.gas_ic_rate,
            "ic_is_available": contract.ic_is_available,
            "oil_cr_cap_rate": contract.oil_cr_cap_rate,
            "gas_cr_cap_rate": contract.gas_cr_cap_rate,

            # DMO
            "oil_dmo_volume_portion": 0.25,
            "oil_dmo_fee_portion": 0.25,
            "oil_dmo_holiday_duration": 60,
            "gas_dmo_volume_portion": 1.0,
            "gas_dmo_fee_portion": 1.0,
            "gas_dmo_holiday_duration": 60,

            # Carry forward depreciation
            "oil_carry_forward_depreciation": 0.0,
            "gas_carry_forward_depreciation": 0.0,
        }

        return CostRecovery(**kwargs_costrec)

    # # When the contract is GrossSplit, parsing back the adjusted cost elements
    # # to the gross split contract
    elif isinstance(contract, GrossSplit):
        kwargs_gs = {
            # Base parameters
            **kwargs_base,

            # Lifting and costs
            **kwargs_lifting_costs,

            # Field and reservoir properties
            "field_status": contract.field_status,
            "field_loc": contract.field_loc,
            "res_depth": contract.res_depth,
            "infra_avail": contract.infra_avail,
            "res_type": contract.res_type,
            "api_oil": contract.api_oil,
            "domestic_use": contract.domestic_use,
            "prod_stage": contract.prod_stage,
            "co2_content": contract.co2_content,
            "h2s_content": contract.h2s_content,
            "field_reserves_2024": contract.field_reserves_2024,
            "infra_avail_2024": contract.infra_avail_2024,
            "field_loc_2024": contract.field_loc_2024,

            # Ministry discretion
            "split_ministry_disc": contract.split_ministry_disc,

            # DMO
            "oil_dmo_volume_portion": contract.oil_dmo_volume_portion,
            "oil_dmo_fee_portion": contract.oil_dmo_fee_portion,
            "oil_dmo_holiday_duration": contract.oil_dmo_holiday_duration,
            "gas_dmo_volume_portion": contract.gas_dmo_volume_portion,
            "gas_dmo_fee_portion": contract.gas_dmo_fee_portion,
            "gas_dmo_holiday_duration": contract.gas_dmo_holiday_duration,

            # Carry forward depreciation
            "oil_carry_forward_depreciation": contract.oil_carry_forward_depreciation,
            "gas_carry_forward_depreciation": contract.gas_carry_forward_depreciation,
        }

        return GrossSplit(**kwargs_gs)

    # When the contract is not recognized, raise an exception
    else:
        raise OptimizationException(
            f"Contract type {type(contract)}, is not a valid contract "
            f"for optimization module"
        )
        
================================================================================================

def optimize_psc_core(
    contract: CostRecovery | GrossSplit,
    contract_arguments: dict,
    summary_argument: dict,
    target_optimization_value: float,
    dict_optimization: dict,
    target_parameter: OptimizationTarget = OptimizationTarget.IRR,
) -> (list, list, float, list):
    
    # Changing the Optimization selection from Enum to string in order to retrieve
    # the result from summary dictionary
    if target_parameter is OptimizationTarget.IRR:
        target_parameter = "ctr_irr"

    elif target_parameter is OptimizationTarget.NPV:
        target_parameter = "ctr_npv"

    elif target_parameter is OptimizationTarget.PI:
        target_parameter = "ctr_pi"

    else:
        raise OptimizationException(
            f"target {target_parameter} "
            f"is should be one of: {[OptimizationTarget.value]}"
        )

    # Changing the parameters list[str] into list[OptimizationParameters(Enum)]
    list_params = dict_optimization["parameter"]

    # Defining Base Value list to contain value of optimized parameters and
    # status of the optimization
    list_params_value = ["Base Value"] * len(list_params)

    # Defining the empty result of optimization target, will be defined later
    result_optimization = None

    # Defining the executed contract list
    list_executed_contract = []

    psc = contract
    for index, param in enumerate(list_params):

        # Get the maximum value of each params
        if param is OptimizationParameter.OIL_CTR_PRETAX:
            max_value = dict_optimization["max"][index]

        elif param is OptimizationParameter.GAS_CTR_PRETAX:
            max_value = dict_optimization["max"][index]

        elif param is OptimizationParameter.OIL_FTP_PORTION:
            max_value = dict_optimization["min"][index]

        elif param is OptimizationParameter.GAS_FTP_PORTION:
            max_value = dict_optimization["min"][index]

        elif param is OptimizationParameter.OIL_IC:
            max_value = dict_optimization["max"][index]

        elif param is OptimizationParameter.GAS_IC:
            max_value = dict_optimization["max"][index]

        elif param is OptimizationParameter.OIL_DMO_FEE:
            max_value = dict_optimization["max"][index]

        elif param is OptimizationParameter.GAS_DMO_FEE:
            max_value = dict_optimization["max"][index]

        elif param is OptimizationParameter.VAT_RATE:
            max_value = dict_optimization["min"][index]

        elif param is OptimizationParameter.EFFECTIVE_TAX_RATE:
            max_value = dict_optimization["min"][index]

        elif param is OptimizationParameter.VAT_DISCOUNT:
            max_value = dict_optimization["max"][index]

        elif param is OptimizationParameter.LBT_DISCOUNT:
            max_value = dict_optimization["max"][index]

        elif param is OptimizationParameter.DEPRECIATION_ACCELERATION:
            max_value = dict_optimization["max"][index]

        elif param is OptimizationParameter.MINISTERIAL_DISCRETION:
            max_value = dict_optimization["max"][index]

        else:
            raise OptimizationException(
                f" Optimization parameter, {param}, is not recognized."
                f" Optimization parameter should be chosen from OptimizationParameter enum."
            )

        # Changing the Contract parameter value based on the given input
        result_psc, psc = adjust_contract(
            contract=psc,
            contract_arguments=contract_arguments,
            variable=param,
            value=max_value,
            summary_argument=summary_argument,
            target_parameter=target_parameter,
        )

        # Able to conduct optimization since the result is greater than the target
        if result_psc > target_optimization_value:
            # Defining the upper and lower limit of the optimized variable
            bounds = (dict_optimization["min"][index], dict_optimization["max"][index])

            def objective_run(new_value):
                result_psc_obj, executed_contract = adjust_contract(
                    contract=psc,
                    contract_arguments=contract_arguments,
                    variable=param,
                    value=new_value,
                    summary_argument=summary_argument,
                    target_parameter=target_parameter,
                )

                result_obj = abs(result_psc_obj - target_optimization_value)

                return result_obj

            # Optimization of the objective function
            optim_result = minimize_scalar(
                objective_run, bounds=bounds, method="bounded"
            )

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
            executed_contract = adjust_contract(
                contract=psc,
                contract_arguments=contract_arguments,
                variable=param,
                value=optimized_parameter,
                summary_argument=summary_argument,
                target_parameter=target_parameter,
            )[1]

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
    
================================================================================================
    
def adjust_contract(
    contract: CostRecovery | GrossSplit,
    contract_arguments: dict,
    summary_argument: dict,
    variable: OptimizationParameter,
    value: float,
    target_parameter: str,
) -> (CostRecovery | GrossSplit, dict):

    # Optimization parameter is VAT
    if variable is OptimizationParameter.VAT_DISCOUNT:
        contract = adjust_cost_element(
            contract=contract,
            adjustment_value=value,
            adjustment_variable=OptimizationParameter.VAT_DISCOUNT,
        )

    # Optimization parameter is LBT
    elif variable is OptimizationParameter.LBT_DISCOUNT:
        contract = adjust_cost_element(
            contract=contract,
            adjustment_value=value,
            adjustment_variable=OptimizationParameter.LBT_DISCOUNT,
        )

    # Optimization parameter is accelerated depreciation
    elif variable is OptimizationParameter.DEPRECIATION_ACCELERATION:
        contract = adjust_cost_element(
            contract=contract,
            adjustment_value=value,
            adjustment_variable=OptimizationParameter.DEPRECIATION_ACCELERATION,
        )

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
            contract_arguments["vat_rate"] = value

        if variable is OptimizationParameter.EFFECTIVE_TAX_RATE:
            contract_arguments["effective_tax_rate"] = value

    # The condition when contract is Gross Split
    elif isinstance(contract, GrossSplit):
        # Changing the attributes of the contract based on the chosen variable
        if variable is OptimizationParameter.MINISTERIAL_DISCRETION:
            contract.split_ministry_disc = value

        if variable is OptimizationParameter.OIL_DMO_FEE:
            contract.oil_dmo_fee_portion = value

        if variable is OptimizationParameter.GAS_DMO_FEE:
            contract.gas_dmo_fee_portion = value

        if variable is OptimizationParameter.VAT_RATE:
            contract_arguments["vat_rate"] = value

        if variable is OptimizationParameter.EFFECTIVE_TAX_RATE:
            contract_arguments["effective_tax_rate"] = value

    # Running the contract
    contract.run(**contract_arguments)

    # Get the summary of the new contract and get its value of the targeted optimization
    result_psc = contract.get_summary(**summary_argument)[target_parameter]

    # Former approach
    # ---------------
    # summary_argument["contract"] = contract
    # result_psc = get_summary(**summary_argument)[target_parameter]
    # 
    # return result_psc, contract

    return result_psc, contract
    
"""