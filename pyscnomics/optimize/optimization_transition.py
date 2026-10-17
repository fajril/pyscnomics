from typing import Literal, TypeAlias, TypeVar, TypedDict

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar

from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.gross_split import GrossSplit
from pyscnomics.contracts.project import require_float_array
from pyscnomics.contracts.transition import Transition
from pyscnomics.econ.selection import (
    OptimizationParameter,
    FluidType,
    OptimizationTarget,
    DiscountingMode,
    NPVSelection,
)
from pyscnomics.tools.summary import get_summary

from pyscnomics.econ.costs import CapitalCost, Intangible, OPEX, ASR, LBT
from pyscnomics.optimize.optimization import adjust_useful_life_years


class OptimizationExceptionTransition(Exception):
    """Exception to raise for a misuse of BaseProject class"""

    pass


_SummaryValue: TypeAlias = float | np.float64
_OptimizationScalar: TypeAlias = float | np.float64
_OptimizationValue: TypeAlias = str | _OptimizationScalar


class _TransitionRunArguments(TypedDict, total=False):
    unrec_portion: float


class _SummaryArguments(TypedDict, total=False):
    contract: Transition
    reference_year: int | None
    discount_rate: float
    npv_mode: NPVSelection
    discounting_mode: DiscountingMode
    inflation_rate: float
    profitability_discounted: bool


_DynamicArguments: TypeAlias = _TransitionRunArguments | _SummaryArguments
_CostItem = TypeVar("_CostItem")


class _OptimizationConfiguration(TypedDict):
    parameter: list[OptimizationParameter]
    min: npt.NDArray[np.float64]
    max: npt.NDArray[np.float64]


def _required_items(items: tuple[_CostItem, ...] | None) -> tuple[_CostItem, ...]:
    """Retain the original iterable failure for uninitialized schedules."""

    if items is None:
        raise TypeError("'NoneType' object is not iterable")
    return items


def _read_optimization_result(
    result: object,
) -> tuple[_OptimizationScalar, _OptimizationScalar]:
    """Validate the mapping interface returned by SciPy's scalar optimizer."""

    if not isinstance(result, dict):
        raise OptimizationExceptionTransition(
            "scalar optimization returned an invalid result"
        )
    if not isinstance(result["fun"], float) or not isinstance(result["x"], float):
        raise OptimizationExceptionTransition(
            "scalar optimization returned an invalid result"
        )
    return result["fun"], result["x"]


def _summary_scalar(value: object, target_parameter: str) -> _SummaryValue:
    """Validate that an optimization target resolves to a scalar summary value."""

    if isinstance(value, float):
        return value
    if type(value) is np.float64:
        return value
    raise TypeError(f"Summary target {target_parameter} must be a scalar value")


def _run_float(value: _OptimizationScalar) -> float:
    """Expose a scalar as the float type required by contract run arguments."""

    return value if isinstance(value, float) else float(value)


def _optimization_bound_value(
    bounds: npt.NDArray[np.float64], index: int
) -> _OptimizationScalar:
    """Read and validate a scalar bound from the optimization configuration."""

    if bounds.ndim != 1 or bounds.dtype != np.dtype(np.float64):
        raise TypeError("optimization bounds must be a one-dimensional float64 array")
    return bounds.flat[index]


def adjust_contract(
    contract: Transition,
    contract_arguments: _TransitionRunArguments,
    variable: OptimizationParameter,
    value: _OptimizationScalar,
    summary_argument: _SummaryArguments,
    target_parameter: str,
) -> tuple[_SummaryValue, Transition]:
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
    contract_value: float = value if isinstance(value, float) else float(value)

    # When the optimization is VAT
    if variable is OptimizationParameter.VAT_DISCOUNT:
        contract = adjust_cost_element(
            contract=contract,
            adjustment_value=value,
            adjustment_variable=OptimizationParameter.VAT_DISCOUNT,
        )

    # When the optimization is LBT
    elif variable is OptimizationParameter.LBT_DISCOUNT:
        contract = adjust_cost_element(
            contract=contract,
            adjustment_value=value,
            adjustment_variable=OptimizationParameter.LBT_DISCOUNT,
        )

    elif variable is OptimizationParameter.DEPRECIATION_ACCELERATION:
        contract = adjust_cost_element(
            contract=contract,
            adjustment_value=value,
            adjustment_variable=OptimizationParameter.DEPRECIATION_ACCELERATION,
        )

    # The condition when contract is Cost Recovery
    if isinstance(contract.contract2, CostRecovery):
        # Changing the attributes of the contract based on the chosen variable
        if variable is OptimizationParameter.OIL_CTR_PRETAX:
            contract.contract2.oil_ctr_pretax_share = contract_value

        if variable is OptimizationParameter.GAS_CTR_PRETAX:
            contract.contract2.gas_ctr_pretax_share = contract_value

        if variable is OptimizationParameter.OIL_FTP_PORTION:
            contract.contract2.oil_ftp_portion = contract_value

        if variable is OptimizationParameter.GAS_FTP_PORTION:
            contract.contract2.gas_ftp_portion = contract_value

        if variable is OptimizationParameter.OIL_IC:
            contract.contract2.oil_ic_rate = contract_value

        if variable is OptimizationParameter.GAS_IC:
            contract.contract2.gas_ic_rate = contract_value

        if variable is OptimizationParameter.OIL_DMO_FEE:
            contract.contract2.oil_dmo_fee_portion = contract_value

        if variable is OptimizationParameter.GAS_DMO_FEE:
            contract.contract2.gas_dmo_fee_portion = contract_value

        if variable is OptimizationParameter.VAT_RATE:
            contract.argument_contract2["vat_rate"] = _run_float(value)

        if variable is OptimizationParameter.EFFECTIVE_TAX_RATE:
            contract.argument_contract2["effective_tax_rate"] = _run_float(value)

    # The condition when contract is Gross Split
    if isinstance(contract.contract2, GrossSplit):
        # Changing the attributes of the contract based on the chosen variable
        if variable is OptimizationParameter.MINISTERIAL_DISCRETION:
            contract.contract2.split_ministry_disc = contract_value

        if variable is OptimizationParameter.OIL_DMO_FEE:
            contract.contract2.oil_dmo_fee_portion = contract_value

        if variable is OptimizationParameter.GAS_DMO_FEE:
            contract.contract2.gas_dmo_fee_portion = contract_value

        if variable is OptimizationParameter.VAT_RATE:
            contract.argument_contract2["vat_rate"] = _run_float(value)

        if variable is OptimizationParameter.EFFECTIVE_TAX_RATE:
            contract.argument_contract2["effective_tax_rate"] = _run_float(value)

    # Running the contract
    contract.run(**contract_arguments)

    # Get the summary of the new contract and get its value of the targeted optimization
    summary_argument["contract"] = contract
    result_psc = _summary_scalar(
        get_summary(**summary_argument)[target_parameter], target_parameter
    )

    return result_psc, contract


def adjust_cost_element(
    contract: Transition,
    adjustment_value: _OptimizationScalar = 1,
    adjustment_variable: OptimizationParameter = OptimizationParameter.VAT_DISCOUNT,
) -> Transition:
    """
    Function to adjust the vat discount or lbt discount of the cost element
    by multiplying them with the adjustment factor.

    Parameters
    ----------
    contract: Transition
        The contract which will be adjusted
    adjustment_value: float
        The factor which will be applied for the adjustment
    adjustment_variable: str
        The cost element that will be adjusted. The options are: "VAT" or "LBT"

    Returns
    -------
    contract_adjusted: Transition
        The contract that the cost element has been adjusted.

    """

    adjustment_value = (
        adjustment_value
        if isinstance(adjustment_value, float)
        else float(adjustment_value)
    )
    contract2 = contract.contract2

    capital_adjusted = contract2.capital_cost
    intangible_adjusted = contract2.intangible_cost
    opex_adjusted = contract2.opex
    asr_adjusted = contract2.asr_cost
    lbt_adjusted = contract2.lbt_cost
    cos_adjusted = contract2.cost_of_sales

    # Condition when the VAT of each cost element will be adjusted
    if adjustment_variable == OptimizationParameter.VAT_DISCOUNT:
        # Adjusting the Tangible cost of the contract
        capital_adjusted = tuple(
            [
                CapitalCost(
                    start_year=tan.start_year,
                    end_year=tan.end_year,
                    cost=tan.cost,
                    expense_year=tan.expense_year,
                    cost_allocation=tan.cost_allocation,
                    description=tan.description,
                    tax_portion=tan.tax_portion,
                    tax_discount=adjustment_value,
                    pis_year=tan.pis_year,
                    salvage_value=tan.salvage_value,
                    useful_life=tan.useful_life,
                    depreciation_factor=tan.depreciation_factor,
                    is_ic_applied=tan.is_ic_applied,
                )
                for tan in _required_items(contract2.capital_cost)
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
                    description=intang.description,
                    tax_portion=intang.tax_portion,
                    tax_discount=adjustment_value,
                )
                for intang in _required_items(contract2.intangible_cost)
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
                    description=opx.description,
                    tax_portion=opx.tax_portion,
                    tax_discount=adjustment_value,
                    fixed_cost=opx.fixed_cost,
                    prod_rate=opx.prod_rate,
                    cost_per_volume=opx.cost_per_volume,
                )
                for opx in _required_items(contract2.opex)
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
                    description=asr.description,
                    tax_portion=asr.tax_portion,
                    tax_discount=adjustment_value,
                    final_year=asr.final_year,
                    future_rate=asr.future_rate,
                )
                for asr in _required_items(contract2.asr_cost)
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
                for bt in _required_items(contract2.lbt_cost)
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
                    description=tan.description,
                    tax_portion=tan.tax_portion,
                    tax_discount=tan.tax_discount,
                    pis_year=tan.pis_year,
                    salvage_value=tan.salvage_value,
                    useful_life=np.asarray(
                        adjust_useful_life_years(
                            adjustment_value=adjustment_value,
                            useful_life_array=require_float_array(
                                tan.useful_life, "useful_life"
                            ),
                        ),
                        dtype=np.float64,
                    ),
                    depreciation_factor=tan.depreciation_factor,
                    is_ic_applied=tan.is_ic_applied,
                )
                for tan in _required_items(contract2.capital_cost)
            ]
        )

    # Condition when the chosen option is not recognized
    else:
        raise OptimizationExceptionTransition(
            f"Adjustment Variable {adjustment_variable} do not exist. It should be VAT or LBT in string data type"
        )

    # Condition where only one fluid is produced
    produced_fluid = [item.fluid_type for item in _required_items(contract2.lifting)]

    if FluidType.OIL not in produced_fluid:
        oil_onstream_date = None
    else:
        oil_onstream_date = contract2.oil_onstream_date

    if FluidType.GAS not in produced_fluid:
        gas_onstream_date = None
    else:
        gas_onstream_date = contract2.gas_onstream_date

    # When the contract is CostRecovery, parsing back the adjusted cost elements
    # to the cost recovery contract
    if isinstance(contract2, CostRecovery):
        contract2_adjusted = CostRecovery(
            start_date=contract2.start_date,
            end_date=contract2.end_date,
            approval_year=contract2.approval_year,
            oil_onstream_date=oil_onstream_date,
            gas_onstream_date=gas_onstream_date,
            is_pod_1=contract2.is_pod_1,
            is_strict=contract2.is_strict,
            lifting=contract2.lifting,
            capital_cost=capital_adjusted,
            intangible_cost=intangible_adjusted,
            opex=opex_adjusted,
            asr_cost=asr_adjusted,
            lbt_cost=lbt_adjusted,
            cost_of_sales=cos_adjusted,
            oil_ftp_is_available=contract2.oil_ftp_is_available,
            oil_ftp_is_shared=contract2.oil_ftp_is_shared,
            oil_ftp_portion=contract2.oil_ftp_portion,
            gas_ftp_is_available=contract2.gas_ftp_is_available,
            gas_ftp_is_shared=contract2.gas_ftp_is_shared,
            gas_ftp_portion=contract2.gas_ftp_portion,
            tax_split_type=contract2.tax_split_type,
            condition_dict=contract2.condition_dict,
            indicator_rc_icp_sliding=contract2.indicator_rc_icp_sliding,
            oil_ctr_pretax_share=contract2.oil_ctr_pretax_share,
            gas_ctr_pretax_share=contract2.gas_ctr_pretax_share,
            oil_ic_rate=contract2.oil_ic_rate,
            gas_ic_rate=contract2.gas_ic_rate,
            ic_is_available=contract2.ic_is_available,
            oil_cr_cap_rate=contract2.oil_cr_cap_rate,
            gas_cr_cap_rate=contract2.gas_cr_cap_rate,
            oil_dmo_volume_portion=contract2.oil_dmo_volume_portion,
            oil_dmo_fee_portion=contract2.oil_dmo_fee_portion,
            oil_dmo_holiday_duration=contract2.oil_dmo_holiday_duration,
            gas_dmo_volume_portion=contract2.gas_dmo_volume_portion,
            gas_dmo_fee_portion=contract2.gas_dmo_fee_portion,
            gas_dmo_holiday_duration=contract2.gas_dmo_holiday_duration,
            oil_carry_forward_depreciation=contract2.oil_carry_forward_depreciation,
            gas_carry_forward_depreciation=contract2.gas_carry_forward_depreciation,
        )

    # When the contract is GrossSplit, parsing back the adjusted cost elements
    # to the gross split contract
    else:
        contract2_adjusted = GrossSplit(
            start_date=contract2.start_date,
            end_date=contract2.end_date,
            approval_year=contract2.approval_year,
            oil_onstream_date=oil_onstream_date,
            gas_onstream_date=gas_onstream_date,
            is_pod_1=contract2.is_pod_1,
            is_strict=contract2.is_strict,
            lifting=contract2.lifting,
            capital_cost=capital_adjusted,
            intangible_cost=intangible_adjusted,
            opex=opex_adjusted,
            asr_cost=asr_adjusted,
            lbt_cost=lbt_adjusted,
            cost_of_sales=cos_adjusted,
            field_status=contract2.field_status,
            field_loc=contract2.field_loc,
            res_depth=contract2.res_depth,
            infra_avail=contract2.infra_avail,
            res_type=contract2.res_type,
            api_oil=contract2.api_oil,
            domestic_use=contract2.domestic_use,
            prod_stage=contract2.prod_stage,
            co2_content=contract2.co2_content,
            h2s_content=contract2.h2s_content,
            field_reserves_2024=contract2.field_reserves_2024,
            infra_avail_2024=contract2.infra_avail_2024,
            field_loc_2024=contract2.field_loc_2024,
            split_ministry_disc=contract2.split_ministry_disc,
            oil_dmo_volume_portion=contract2.oil_dmo_volume_portion,
            oil_dmo_fee_portion=contract2.oil_dmo_fee_portion,
            oil_dmo_holiday_duration=contract2.oil_dmo_holiday_duration,
            gas_dmo_volume_portion=contract2.gas_dmo_volume_portion,
            gas_dmo_fee_portion=contract2.gas_dmo_fee_portion,
            gas_dmo_holiday_duration=contract2.gas_dmo_holiday_duration,
            oil_carry_forward_depreciation=contract2.oil_carry_forward_depreciation,
            gas_carry_forward_depreciation=contract2.gas_carry_forward_depreciation,
        )

    # Replacing the second contract with the contract_adjusted
    contract_adjusted = contract
    contract_adjusted.contract2 = contract2_adjusted

    return contract_adjusted


def optimize_psc_core(
    dict_optimization: _OptimizationConfiguration,
    contract: Transition,
    contract_arguments: _TransitionRunArguments,
    target_optimization_value: _SummaryValue,
    summary_argument: _SummaryArguments,
    target_parameter: OptimizationTarget = OptimizationTarget.IRR,
) -> tuple[
    list[str],
    list[_OptimizationValue],
    _SummaryValue | None,
    list[Transition],
]:
    """
    The function to get contract variable(s) that resulting the desired target or
    contract's economic target.

    Parameters
    ----------
    dict_optimization: dict
        The optimization dictionary that containing the information about minimum
        boundary and upper boundary of the optimized parameters.
    contract: Transition
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
    The dictionary of dict_optimization should be at least having the structure as
    the following:
        dict_optimization = {'parameters': list[OptimizationParameter],
                             'min': np.ndarray,
                             'max': np.ndarray}

        'parameters' keys containing the enum list of the variable that will be
        optimized to achieve the target.
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
    target_param_mapping: dict[OptimizationTarget, str] = {
        OptimizationTarget.IRR: "ctr_irr",
        OptimizationTarget.NPV: "ctr_npv",
        OptimizationTarget.PI: "ctr_pi",
    }
    try:
        target_param = target_param_mapping[target_parameter]
    except KeyError as error:
        raise OptimizationExceptionTransition(
            f"target {target_parameter} is should be one of: {[item.value for item in OptimizationTarget]}"
        ) from error

    # Changing the parameters list[str] into list[OptimizationParameters(Enum)]
    list_params = dict_optimization["parameter"]

    # Defining Base Value list to contain value of optimized parameters
    # and status of the optimization
    list_params_value: list[_OptimizationValue] = ["Base Value"] * len(list_params)

    # Defining the empty result of optimization target, will be defined later
    result_optimization: _SummaryValue | None = None

    # Defining the executed contract list
    list_executed_contract: list[Transition] = []

    psc = contract
    param_bound_source: dict[OptimizationParameter, Literal["min", "max"]] = {
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
        try:
            bound_key = param_bound_source[param]
        except KeyError as error:
            raise OptimizationExceptionTransition(
                f"Optimization parameter, {param}, is not recognized. Optimization parameter should be chosen from OptimizationParameter enum."
            ) from error

        max_value = _optimization_bound_value(dict_optimization[bound_key], index)

        # Changing the Contract parameter value based on the given input
        result_psc, psc = adjust_contract(
            contract=psc,
            contract_arguments=contract_arguments,
            variable=param,
            value=max_value,
            summary_argument=summary_argument,
            target_parameter=target_param,
        )

        # Able to conduct optimization since the result is greater than the target
        if result_psc > target_optimization_value:
            # Defining the upper and lower limit of the optimized variable
            bounds: tuple[_OptimizationScalar, _OptimizationScalar] = (
                _optimization_bound_value(dict_optimization["min"], index),
                _optimization_bound_value(dict_optimization["max"], index),
            )

            def objective_run(new_value: float) -> _SummaryValue:
                result_psc_obj, _ = adjust_contract(
                    contract=psc,
                    contract_arguments=contract_arguments,
                    variable=param,
                    value=new_value,
                    summary_argument=summary_argument,
                    target_parameter=target_param,
                )

                result_obj = abs(result_psc_obj - target_optimization_value)
                return result_obj

            # Optimization of the objective function
            optim_result: object = minimize_scalar(
                objective_run, bounds=bounds, method="bounded"
            )

            # Difference value from target and optimization result
            optimized_diff, optimized_parameter = _read_optimization_result(
                optim_result
            )

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
                target_parameter=target_param,
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
