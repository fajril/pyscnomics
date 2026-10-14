"""
Collection of functions to administer sensitivity analysis.

This Code is made to detached the sensitivity module in the previous version
which depended with the excel into fully able to be run in python.
"""

from collections.abc import Mapping
from datetime import date
from enum import Enum
from typing import Literal, TypeAlias, TypedDict, TypeGuard, TypeVar, overload

import numpy as np
import numpy.typing as npt
import pandas as pd

from pyscnomics.econ import Lifting, FluidType
# from pyscnomics.io.aggregator import Aggregate
# from pyscnomics.optimize.adjuster import AdjustData

from pyscnomics.contracts.project import BaseProject, require_float_array
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.gross_split import GrossSplit
from pyscnomics.contracts.transition import Transition
from pyscnomics.econ.costs import CapitalCost, Intangible, OPEX
from pyscnomics.econ.selection import (
    DiscountingMode,
    InflationAppliedTo,
    NPVSelection,
    OtherRevenue,
)
from pyscnomics.contracts.run_arguments import (
    validate_cost_recovery_run_arguments,
    validate_gross_split_run_arguments,
)
from pyscnomics.input_validation import InputValidationError, engine_input_issue
from pyscnomics.validation_rules import (
    deviation_is_negative,
    increment_not_positive,
    step_not_positive,
)


FloatArray: TypeAlias = npt.NDArray[np.float64]
_Contract: TypeAlias = BaseProject | CostRecovery | GrossSplit
_ContractArguments: TypeAlias = dict[str, object]
_SummaryArguments: TypeAlias = dict[str, object]
_SummaryValue: TypeAlias = float | int | np.float64 | None
_Summary: TypeAlias = Mapping[str, _SummaryValue]
_Multiplier: TypeAlias = float | np.float64
_SensitivityFrames: TypeAlias = dict[str, pd.DataFrame]
_SensitivityValues: TypeAlias = dict[str, dict[np.float64, dict[str, _SummaryValue]]]
_SensitivityResult: TypeAlias = _SensitivityFrames | _SensitivityValues
_Item = TypeVar("_Item")
_EnumValue = TypeVar("_EnumValue", bound=Enum)
_SENSITIVITY_INDICATORS = (
    "ctr_npv",
    "ctr_irr",
    "ctr_pi",
    "ctr_pot",
    "gov_take",
    "ctr_net_share",
)


class _BaseProjectRunArguments(TypedDict, total=False):
    sulfur_revenue: OtherRevenue
    electricity_revenue: OtherRevenue
    co2_revenue: OtherRevenue
    vat_rate: FloatArray | float
    year_inflation: npt.NDArray[np.int64] | None
    inflation_rate: FloatArray | float
    inflation_rate_applied_to: InflationAppliedTo | None


class _ValidatedSummaryArguments(TypedDict, total=False):
    discount_rate: float
    npv_mode: NPVSelection
    discounting_mode: DiscountingMode
    discount_rate_start_year: int | None
    inflation_rate: float
    profitability_discounted: bool


class _AdjustedParameters(TypedDict):
    capital: tuple[CapitalCost, ...]
    intangible: tuple[Intangible, ...]
    opex: tuple[OPEX, ...]
    lifting: tuple[Lifting, ...]


class _OnstreamDates(TypedDict):
    oil: date | None
    gas: date | None


def _is_float_array(value: object) -> TypeGuard[FloatArray]:
    return isinstance(value, np.ndarray) and value.dtype == np.dtype(np.float64)


def _is_year_array(value: object) -> TypeGuard[npt.NDArray[np.int64]]:
    return isinstance(value, np.ndarray) and value.dtype == np.dtype(np.int64)


def _require_enum(value: object, expected: type[_EnumValue], name: str) -> _EnumValue:
    if not isinstance(value, expected):
        raise SensitivityException(f"{name} has an invalid value")
    return value


def _require_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SensitivityException(f"{name} must be a float")
    return float(value)


def _is_summary_value(value: object) -> TypeGuard[float | int | np.float64]:
    return isinstance(value, (float, int)) or type(value) is np.float64


def _validate_base_project_run_arguments(
    arguments: Mapping[str, object],
) -> _BaseProjectRunArguments:
    allowed = frozenset(_BaseProjectRunArguments.__annotations__)
    if any(key not in allowed for key in arguments):
        raise SensitivityInputError(
            "contract arguments contain an unsupported keyword",
            code="SENSITIVITY_UNSUPPORTED_KEYWORD",
        )

    result: _BaseProjectRunArguments = {}
    for key in ("sulfur_revenue", "electricity_revenue", "co2_revenue"):
        if key in arguments:
            revenue = _require_enum(arguments[key], OtherRevenue, key)
            if key == "sulfur_revenue":
                result["sulfur_revenue"] = revenue
            elif key == "electricity_revenue":
                result["electricity_revenue"] = revenue
            else:
                result["co2_revenue"] = revenue

    for key in ("vat_rate", "inflation_rate"):
        if key in arguments:
            value = arguments[key]
            if not isinstance(value, float) and not _is_float_array(value):
                raise SensitivityInputError(
                    f"{key} must be a float or float64 NumPy array",
                    code="SENSITIVITY_RUN_ARGUMENT_TYPE",
                )
            if key == "vat_rate":
                result["vat_rate"] = value
            else:
                result["inflation_rate"] = value

    if "year_inflation" in arguments:
        value = arguments["year_inflation"]
        if value is not None and not _is_year_array(value):
            raise SensitivityInputError(
                "year_inflation must be an int64 NumPy array or None",
                code="SENSITIVITY_RUN_ARGUMENT_TYPE",
            )
        result["year_inflation"] = value

    if "inflation_rate_applied_to" in arguments:
        value = arguments["inflation_rate_applied_to"]
        result["inflation_rate_applied_to"] = (
            None
            if value is None
            else _require_enum(value, InflationAppliedTo, "inflation_rate_applied_to")
        )
    return result


def _validate_summary_arguments(
    arguments: Mapping[str, object],
) -> _ValidatedSummaryArguments:
    allowed = frozenset(_ValidatedSummaryArguments.__annotations__)
    if any(key not in allowed for key in arguments):
        raise SensitivityInputError(
            "summary arguments contain an unsupported keyword",
            code="SENSITIVITY_UNSUPPORTED_KEYWORD",
        )

    result: _ValidatedSummaryArguments = {}
    if "discount_rate" in arguments:
        result["discount_rate"] = _require_float(
            arguments["discount_rate"], "discount_rate"
        )
    if "npv_mode" in arguments:
        result["npv_mode"] = _require_enum(
            arguments["npv_mode"], NPVSelection, "npv_mode"
        )
    if "discounting_mode" in arguments:
        result["discounting_mode"] = _require_enum(
            arguments["discounting_mode"], DiscountingMode, "discounting_mode"
        )
    if "discount_rate_start_year" in arguments:
        value = arguments["discount_rate_start_year"]
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            raise SensitivityInputError(
                "discount_rate_start_year must be an int or None",
                code="SENSITIVITY_SUMMARY_ARGUMENT_TYPE",
            )
        result["discount_rate_start_year"] = value
    if "inflation_rate" in arguments:
        result["inflation_rate"] = _require_float(
            arguments["inflation_rate"], "inflation_rate"
        )
    if "profitability_discounted" in arguments:
        value = arguments["profitability_discounted"]
        if not isinstance(value, bool):
            raise SensitivityInputError(
                "profitability_discounted must be a bool",
                code="SENSITIVITY_SUMMARY_ARGUMENT_TYPE",
            )
        result["profitability_discounted"] = value
    return result


def _get_summary(
    contract: _Contract | Transition, summary_arguments: _SummaryArguments
) -> _Summary:
    """Summarize a contract after validating the public keyword payload."""

    arguments = _validate_summary_arguments(summary_arguments)
    if isinstance(contract, Transition):
        raise SensitivityException(
            "Transition sensitivity summary is unavailable because Transition "
            + "does not expose the contract summary API"
        )
    summary: Mapping[str, object] = contract.get_summary(**arguments)

    result: dict[str, _SummaryValue] = {}
    for indicator in _SENSITIVITY_INDICATORS:
        value = summary.get(indicator)
        if value is not None and not _is_summary_value(value):
            raise SensitivityException(
                f"summary indicator {indicator} must be a numeric scalar or None"
            )
        result[indicator] = value
    return result


def _run_contract(contract: _Contract, arguments: _ContractArguments) -> None:
    """Run a contract after validating its concrete keyword payload."""

    if isinstance(contract, CostRecovery):
        contract.run(**validate_cost_recovery_run_arguments(arguments))
    elif isinstance(contract, GrossSplit):
        contract.run(**validate_gross_split_run_arguments(arguments))
    else:
        contract.run(**_validate_base_project_run_arguments(arguments))


def _run_adjusted_contract(
    contract: _Contract | Transition, arguments: _ContractArguments
) -> None:
    """Run either a single contract or a transition with its own arguments."""

    if isinstance(contract, Transition):
        value = arguments.get("unrec_portion", 0.0)
        if not isinstance(value, (float, int)):
            raise SensitivityInputError(
                "unrec_portion must be numeric",
                code="SENSITIVITY_RUN_ARGUMENT_TYPE",
            )
        contract.run(unrec_portion=value)
        return
    _run_contract(contract, arguments)


def _multiplier_values(multipliers: FloatArray) -> tuple[np.float64, ...]:
    """Keep NumPy scalar multiplier keys while avoiding an iterator stub gap."""

    return tuple(multipliers[index] for index in range(multipliers.size))


def _sensitivity_dataframe(
    data: dict[np.float64, dict[str, _SummaryValue]],
) -> pd.DataFrame:
    """Create a result frame from the typed sensitivity mapping."""

    return pd.DataFrame(data).T


class SensitivityException(Exception):
    """Exception to be raised for a misuse of Sensitivity Method"""

    pass


class SensitivityInputError(  # pyright: ignore[reportUnsafeMultipleInheritance]
    SensitivityException, InputValidationError
):
    """Verified user-input sensitivity error retaining ``SensitivityException``."""

    _input_code: str = "SENSITIVITY_INPUT_INVALID"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        InputValidationError.__init__(
            self,
            (engine_input_issue(code=code or self._input_code, message=message),),
        )


def get_sensitivity_multipliers(
    min_deviation: float,
    max_deviation: float,
    base_value: float = 1.0,
    step: int = 10,
    increment: float | None = None,
) -> FloatArray:
    """
    Generate an array of multipliers symmetrically distributed around a base value.

    The multipliers are created using linear spacing between
    ``base_value - min_deviation`` and ``base_value + max_deviation``.

    The lower and upper ranges are combined into a single one-dimensional
    array, excluding the duplicated base value at the center.

    Parameters
    ----------
    min_deviation : float
        The deviation below the base value.
        Must be non-negative.
    max_deviation : float
        The deviation above the base value.
        Must be non-negative.
    base_value : float, optional
        The central reference value around which the multipliers are generated.
        Default is ``1.0``.
    step : int, optional
        The number of intervals used to generate linearly spaced values
        on each side of the base value. Must be greater than zero.
        Default is ``10``.
    increment : float or None, optional
        Fixed spacing between consecutive multipliers. When provided, it
        overrides ``step`` and produces a uniform increment on both sides of
        the base value while always including ``base_value``. Must be greater
        than zero. Default is ``None``.

    Returns
    -------
    np.ndarray
        A one-dimensional NumPy array containing the combined multipliers.
        The array starts from ``base_value - min_deviation`` and ends at
        ``base_value + max_deviation``.

    Raises
    ------
    SensitivityException
        If ``min_deviation`` or ``max_deviation`` is negative, or if ``step``
        is less than or equal to zero, or if ``increment`` is not greater
        than zero.
    """

    # Throw exceptions for unsuitable input data
    if deviation_is_negative(min_deviation) or deviation_is_negative(max_deviation):
        raise SensitivityInputError(
            "Deviations must be non negative.",
            code="SENSITIVITY_DEVIATION_NEGATIVE",
        )

    if increment is not None:
        if increment_not_positive(increment):
            raise SensitivityInputError(
                "Increment must be greater than zero.",
                code="SENSITIVITY_INCREMENT_NOT_POSITIVE",
            )

        n_low = round(min_deviation / increment)
        n_high = round(max_deviation / increment)
        lower_mults = base_value - np.arange(n_low, 0, -1) * increment
        upper_mults = base_value + np.arange(1, n_high + 1) * increment

        return np.asarray(
            np.concatenate((lower_mults, [base_value], upper_mults)), dtype=np.float64
        )

    if step_not_positive(step):
        raise SensitivityInputError(
            "Step must be greater than zero.",
            code="SENSITIVITY_STEP_NOT_POSITIVE",
        )

    # Specify the minimum and maximum values
    min_val = base_value - min_deviation
    max_val = base_value + max_deviation

    # Create an array of minimum and maximum values
    lower_mults = np.linspace(min_val, base_value, step + 1)
    upper_mults = np.linspace(base_value, max_val, step + 1)

    # Previous approach
    # -----------------
    # lower_mults = np.linspace(min_val, base_value, step + 1).round(decimals=2)
    # upper_mults = np.linspace(base_value, max_val, step + 1).round(decimals=2)

    # Combine `min_multipliers` with `max_multipliers into a single array`
    return np.asarray(np.concatenate((lower_mults, upper_mults[1:])), dtype=np.float64)


def _prepare_adjusted_parameters_single_contract(
    contract: _Contract,
    adjustment_value: _Multiplier,
    element: str,
) -> _AdjustedParameters:
    """
    Prepare adjusted economic parameters for a single contract instance.

    This function applies a proportional adjustment to one category of cost or
    production-related data within a given contract object. The specific parameter
    to adjust (CAPEX, OPEX, oil/gas price, or oil/gas lifting rate) is controlled
    by the ``element`` argument. All other parameters remain unchanged.

    The function supports contract types based on the Indonesian Production
    Sharing Contract (PSC) schemes: ``BaseProject``, ``CostRecovery``, and ``GrossSplit``.

    Parameters
    ----------
    contract : BaseProject or CostRecovery or GrossSplit
        The contract instance containing economic data to be adjusted. The object
        must define attributes such as ``capital_cost``, ``intangible_cost``,
        ``opex``, and ``lifting``.
    adjustment_value : float
        The multiplier applied to the selected parameter. For instance, ``1.1``
        represents a +10% increase, while ``0.9`` represents a −10% decrease.
    element : {"CAPEX", "OPEX", "OILPRICE", "GASPRICE", "LIFTING", "OILLIFTING", "GASLIFTING"}
        The target economic component to adjust:

        - ``"CAPEX"`` — Scales both capital and intangible costs.
        - ``"OPEX"`` — Scales operating costs (fixed and per-volume).
        - ``"OILPRICE"`` — Scales the oil price component in the lifting data.
        - ``"GASPRICE"`` — Scales the gas price component in the lifting data.
        - ``"LIFTING"`` — Scales lifting rates for every producer type.
        - ``"OILLIFTING"`` — Scales the oil lifting rate component.
        - ``"GASLIFTING"`` — Scales the gas lifting rate component.

    Returns
    -------
    dict
        A dictionary of adjusted contract components containing the same structure
        as the original contract attributes:
        ```python
        {
            "capital": tuple[CapitalCost],
            "intangible": tuple[Intangible],
            "opex": tuple[OPEX],
            "lifting": tuple[Lifting],
        }
        ```
        Each tuple contains new instances of the corresponding data class, updated
        according to the specified adjustment factor.

    Notes
    -----
    - The adjustment is **multiplicative**, not additive.
    - Unselected components are returned unchanged.
    - This function is designed for **economic sensitivity analysis** in upstream
      oil and gas modeling, where parameter variation is required to assess project
      robustness.
    - Internally, a helper function ``_adjust_lifting()`` is used to handle lifting-
      related modifications efficiently.
    """

    def _required_tuple(
        value: tuple[_Item, ...] | None, name: str
    ) -> tuple[_Item, ...]:
        if value is None:
            raise SensitivityException(f"{name} must be initialized before adjustment")
        return value

    capital_adjusted = _required_tuple(contract.capital_cost, "capital_cost")
    intangible_adjusted = _required_tuple(contract.intangible_cost, "intangible_cost")
    opex_adjusted = _required_tuple(contract.opex, "opex")
    lifting_adjusted = _required_tuple(contract.lifting, "lifting")

    # Helper function to assist lifting adjustment
    def _adjust_lifting(
        fluid_type: FluidType | None, target_field: Literal["lifting_rate", "price"]
    ) -> tuple[Lifting, ...]:

        if target_field not in ["lifting_rate", "price"]:
            raise SensitivityInputError(
                f"Unrecognized target_field: {target_field}. "
                + "Parameter target_field must be 'lifting_rate' or 'price'.",
                code="SENSITIVITY_TARGET_FIELD_UNRECOGNIZED",
            )

        return tuple(
            Lifting(
                start_year=lft.start_year,
                end_year=lft.end_year,
                lifting_rate=(
                    lft.lifting_rate * adjustment_value
                    if target_field == "lifting_rate"
                    and (fluid_type is None or lft.fluid_type is fluid_type)
                    else lft.lifting_rate
                ),
                price=(
                    lft.price * adjustment_value
                    if target_field == "price"
                    and fluid_type is not None
                    and lft.fluid_type is fluid_type
                    else lft.price
                ),
                prod_year=lft.prod_year,
                fluid_type=lft.fluid_type,
                ghv=lft.ghv,
                # prod_rate=lft.prod_rate,
                prod_rate_baseline=lft.prod_rate_baseline,
            )
            for lft in lifting_adjusted
        )

    # Adjustment to CAPEX cost (capital + intangible costs),
    # modify the `cost` attribute of class CapitalCost and Intangible
    if element == "CAPEX":
        capital_adjusted = tuple(
            [
                CapitalCost(
                    start_year=cap.start_year,
                    end_year=cap.end_year,
                    expense_year=cap.expense_year,
                    cost=require_float_array(cap.cost, "cost") * adjustment_value,
                    cost_allocation=cap.cost_allocation,
                    cost_type=cap.cost_type,
                    description=cap.description,
                    tax_portion=cap.tax_portion,
                    tax_discount=cap.tax_discount,
                    pis_year=cap.pis_year,
                    salvage_value=cap.salvage_value,
                    useful_life=cap.useful_life,
                    depreciation_factor=cap.depreciation_factor,
                    is_ic_applied=cap.is_ic_applied,
                )
                for cap in capital_adjusted
            ]
        )

        intangible_adjusted = tuple(
            [
                Intangible(
                    start_year=intang.start_year,
                    end_year=intang.end_year,
                    cost=require_float_array(intang.cost, "cost") * adjustment_value,
                    cost_type=intang.cost_type,
                    expense_year=intang.expense_year,
                    cost_allocation=intang.cost_allocation,
                    description=intang.description,
                    tax_portion=intang.tax_portion,
                    tax_discount=intang.tax_discount,
                )
                for intang in intangible_adjusted
            ]
        )

    # Adjustment to OPEX cost, modify the `fixed cost` attribute of class OPEX
    elif element == "OPEX":
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
                    tax_discount=opx.tax_discount,
                    fixed_cost=require_float_array(opx.fixed_cost, "fixed_cost")
                    * adjustment_value,
                    prod_rate=opx.prod_rate,
                    cost_per_volume=require_float_array(
                        opx.cost_per_volume, "cost_per_volume"
                    )
                    * adjustment_value,
                )
                for opx in opex_adjusted
            ]
        )

    # Adjustment to oil price; modify the `price` attribute of class Lifting
    elif element == "OILPRICE":
        lifting_adjusted = _adjust_lifting(
            fluid_type=FluidType.OIL, target_field="price"
        )

    # Adjustment to gas price; modify the `price` attribute of class Lifting
    elif element == "GASPRICE":
        lifting_adjusted = _adjust_lifting(
            fluid_type=FluidType.GAS, target_field="price"
        )

    # Adjustment to all lifting rates; modify every producer type.
    elif element == "LIFTING":
        lifting_adjusted = _adjust_lifting(fluid_type=None, target_field="lifting_rate")

    # Adjustment to oil lifting; modify the `lifting_rate` attribute of class Lifting
    elif element == "OILLIFTING":
        lifting_adjusted = _adjust_lifting(
            fluid_type=FluidType.OIL, target_field="lifting_rate"
        )

    # Adjustment to gas lifting; modify the `lifting_rate` attribute of class Lifting
    elif element == "GASLIFTING":
        lifting_adjusted = _adjust_lifting(
            fluid_type=FluidType.GAS, target_field="lifting_rate"
        )

    else:
        raise SensitivityException(f"Invalid element: {element!r}")

    return {
        "capital": capital_adjusted,
        "intangible": intangible_adjusted,
        "opex": opex_adjusted,
        "lifting": lifting_adjusted,
    }


def _prepare_onstream_dates_single_contract(
    contract: _Contract,
) -> _OnstreamDates:
    """
    Prepare oil and gas onstream dates for a single contract instance.

    This function determines the onstream dates for oil and gas by checking
    whether the corresponding revenue attributes (`_oil_revenue` and `_gas_revenue`)
    contain nonzero values. If the total revenue for a fluid type is zero,
    the onstream date is set to ``None``. Otherwise, the onstream date is retrieved
    from the contract's respective onstream date attribute.

    Parameters
    ----------
    contract : BaseProject or CostRecovery or GrossSplit
        The contract instance containing oil and gas revenue information,
        as well as the corresponding onstream date attributes.

    Returns
    -------
    dict
        A dictionary containing the oil and gas onstream dates, with the following keys:
        - ``"oil"`` : datetime or None
          The oil onstream date, or ``None`` if `_oil_revenue` equals zero.
        - ``"gas"`` : datetime or None
          The gas onstream date, or ``None`` if `_gas_revenue` equals zero.

    Notes
    -----
    The onstream decision uses the contract's public lifting records, so it remains
    valid before and after a contract run without reaching into private results.
    """

    lifting = contract.lifting
    if lifting is None:
        raise SensitivityException("lifting must be initialized before adjustment")

    def _has_revenue(fluid_type: FluidType) -> bool:
        return any(
            np.count_nonzero(lft.revenue()) != 0
            for lft in lifting
            if lft.fluid_type is fluid_type
        )

    return {
        "oil": contract.oil_onstream_date if _has_revenue(FluidType.OIL) else None,
        "gas": contract.gas_onstream_date if _has_revenue(FluidType.GAS) else None,
    }


@overload
def _adjust_element_single_contract(
    contract: CostRecovery,
    contract_arguments: _ContractArguments,
    element: str,
    adjustment_value: _Multiplier,
    run_contract: bool = False,
) -> CostRecovery: ...


@overload
def _adjust_element_single_contract(
    contract: GrossSplit,
    contract_arguments: _ContractArguments,
    element: str,
    adjustment_value: _Multiplier,
    run_contract: bool = False,
) -> GrossSplit: ...


@overload
def _adjust_element_single_contract(
    contract: BaseProject,
    contract_arguments: _ContractArguments,
    element: str,
    adjustment_value: _Multiplier,
    run_contract: bool = False,
) -> BaseProject: ...


def _adjust_element_single_contract(
    contract: object,
    contract_arguments: _ContractArguments,
    element: str,
    adjustment_value: _Multiplier,
    run_contract: bool = False,
) -> _Contract:
    """
    Adjusts a single parameter of a PSC contract and creates a new adjusted contract instance.

    This function modifies a specified element (e.g., lifting, cost, or onstream parameter)
    for a given contract instance (`BaseProject`, `CostRecovery`, or `GrossSplit`).
    It reconstructs a new contract with updated parameters and optionally executes
    the economic run with the provided contract arguments.

    Parameters
    ----------
    contract : BaseProject or CostRecovery or GrossSplit
        The contract instance to be adjusted.
    contract_arguments : dict
        Arguments required for running the contract model.
        Passed to the contract's ``run()`` method if ``run_contract`` is ``True``.
    element : str
        The name of the parameter element to be adjusted.
        Typically refers to a key in the contract parameter dictionary.
    adjustment_value : float
        The adjustment factor or value applied to the specified element.
    run_contract : bool, default=False
        Whether to execute the contract immediately after adjustment.

    Returns
    -------
    BaseProject or CostRecovery or GrossSplit
        A new contract instance with adjusted parameters. If ``run_contract=True``,
        the returned instance will contain post-run results.
    """

    if not isinstance(contract, BaseProject):
        raise SensitivityException(
            f"Invalid contract type: {contract.__class__.__qualname__}"
        )

    # Prepare adjusted parameters
    params_adjusted = _prepare_adjusted_parameters_single_contract(
        contract=contract,
        adjustment_value=adjustment_value,
        element=element,
    )

    # Prepare onstream dates
    onstream_date = _prepare_onstream_dates_single_contract(contract=contract)

    # Create a new instance of CostRecovery
    if isinstance(contract, CostRecovery):
        contract_adjusted = CostRecovery(
            start_date=contract.start_date,
            end_date=contract.end_date,
            oil_onstream_date=onstream_date["oil"],
            gas_onstream_date=onstream_date["gas"],
            approval_year=contract.approval_year,
            is_pod_1=contract.is_pod_1,
            is_strict=contract.is_strict,
            lifting=params_adjusted["lifting"],
            capital_cost=params_adjusted["capital"],
            intangible_cost=params_adjusted["intangible"],
            opex=params_adjusted["opex"],
            asr_cost=contract.asr_cost,
            lbt_cost=contract.lbt_cost,
            cost_of_sales=contract.cost_of_sales,
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
            gas_dmo_holiday_duration=contract.gas_dmo_holiday_duration,
            oil_carry_forward_depreciation=contract.oil_carry_forward_depreciation,
            gas_carry_forward_depreciation=contract.gas_carry_forward_depreciation,
        )

    # Create a new instance of GrossSplit
    elif isinstance(contract, GrossSplit):
        contract_adjusted = GrossSplit(
            start_date=contract.start_date,
            end_date=contract.end_date,
            oil_onstream_date=onstream_date["oil"],
            gas_onstream_date=onstream_date["gas"],
            approval_year=contract.approval_year,
            is_pod_1=contract.is_pod_1,
            is_strict=contract.is_strict,
            lifting=params_adjusted["lifting"],
            capital_cost=params_adjusted["capital"],
            intangible_cost=params_adjusted["intangible"],
            opex=params_adjusted["opex"],
            asr_cost=contract.asr_cost,
            lbt_cost=contract.lbt_cost,
            cost_of_sales=contract.cost_of_sales,
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
            field_reserves_2024=contract.field_reserves_2024,
            infra_avail_2024=contract.infra_avail_2024,
            field_loc_2024=contract.field_loc_2024,
            split_ministry_disc=contract.split_ministry_disc,
            oil_dmo_volume_portion=contract.oil_dmo_volume_portion,
            oil_dmo_fee_portion=contract.oil_dmo_fee_portion,
            oil_dmo_holiday_duration=contract.oil_dmo_holiday_duration,
            gas_dmo_volume_portion=contract.gas_dmo_volume_portion,
            gas_dmo_fee_portion=contract.gas_dmo_fee_portion,
            gas_dmo_holiday_duration=contract.gas_dmo_holiday_duration,
            oil_carry_forward_depreciation=contract.oil_carry_forward_depreciation,
            gas_carry_forward_depreciation=contract.gas_carry_forward_depreciation,
        )

    # Create a new instance of BaseProject
    else:
        contract_adjusted = BaseProject(
            start_date=contract.start_date,
            end_date=contract.end_date,
            oil_onstream_date=onstream_date["oil"],
            gas_onstream_date=onstream_date["gas"],
            approval_year=contract.approval_year,
            is_pod_1=contract.is_pod_1,
            is_strict=contract.is_strict,
            lifting=params_adjusted["lifting"],
            capital_cost=params_adjusted["capital"],
            intangible_cost=params_adjusted["intangible"],
            opex=params_adjusted["opex"],
            asr_cost=contract.asr_cost,
            lbt_cost=contract.lbt_cost,
            cost_of_sales=contract.cost_of_sales,
        )

    if run_contract:
        _run_adjusted_contract(contract_adjusted, contract_arguments)

    return contract_adjusted


def _adjust_element_transition_contract(
    contract: Transition,
    contract_arguments: _ContractArguments,
    element: str,
    adjustment_value: _Multiplier,
    run_contract: bool = False,
) -> Transition:
    """
    The function to adjust the element of the transition contract based on the adjustment_value.

    Parameters
    ----------
    contract: Transition
        The contract which will be adjusted.
    contract_arguments: dict
        The contract arguments.
    element: str
        The element of the contract that will be adjusted.
    adjustment_value: float
        The adjustment value which will be multiplied by the corresponding element.
    run_contract:bool
        The option to run the contract or not.

    Returns
    -------
    contract_adjusted: Transition
        The contract that has been adjusted.
    """
    # Retrieving the first contract
    argument_contract1 = contract.argument_contract1
    if isinstance(contract.contract1, CostRecovery):
        contract_1_adjusted = _adjust_element_single_contract(
            contract=contract.contract1,
            contract_arguments=argument_contract1,
            element=element,
            adjustment_value=adjustment_value,
            run_contract=run_contract,
        )
    else:
        contract_1_adjusted = _adjust_element_single_contract(
            contract=contract.contract1,
            contract_arguments=argument_contract1,
            element=element,
            adjustment_value=adjustment_value,
            run_contract=run_contract,
        )

    # Retrieving the second contract
    argument_contract2 = contract.argument_contract2
    if isinstance(contract.contract2, CostRecovery):
        contract_2_adjusted = _adjust_element_single_contract(
            contract=contract.contract2,
            contract_arguments=argument_contract2,
            element=element,
            adjustment_value=adjustment_value,
            run_contract=run_contract,
        )
    else:
        contract_2_adjusted = _adjust_element_single_contract(
            contract=contract.contract2,
            contract_arguments=argument_contract2,
            element=element,
            adjustment_value=adjustment_value,
            run_contract=run_contract,
        )

    # Reconstructing the transition contract
    contract_adjusted = Transition(
        contract1=contract_1_adjusted,
        contract2=contract_2_adjusted,
        argument_contract1=argument_contract1,
        argument_contract2=argument_contract2,
    )

    # Condition when the run_contract is true
    if run_contract is True:
        _run_adjusted_contract(contract_adjusted, contract_arguments)

    return contract_adjusted


@overload
def adjust_sensitivity_contract(
    contract: Transition,
    contract_arguments: _ContractArguments,
    element: str,
    adjustment_value: _Multiplier,
    run_contract: bool = False,
) -> Transition: ...


@overload
def adjust_sensitivity_contract(
    contract: _Contract,
    contract_arguments: _ContractArguments,
    element: str,
    adjustment_value: _Multiplier,
    run_contract: bool = False,
) -> _Contract: ...


@overload
def adjust_sensitivity_contract(
    contract: _Contract | Transition,
    contract_arguments: _ContractArguments,
    element: str,
    adjustment_value: _Multiplier,
    run_contract: bool = False,
) -> _Contract | Transition: ...


def adjust_sensitivity_contract(
    contract: _Contract | Transition,
    contract_arguments: _ContractArguments,
    element: str,
    adjustment_value: _Multiplier,
    run_contract: bool = False,
) -> _Contract | Transition:
    """
    Function act as the wrapper to get sensitivity of a contract.

    Parameters
    ----------
    contract: BaseProject | CostRecovery | GrossSplit | Transition
        The contract which will be adjusted.
    contract_arguments: dict
        The contract arguments.
    element: str
        The element of the contract that will be adjusted.
    adjustment_value: float
        The adjustment value which will be multiplied by the corresponding element.
    run_contract:bool
        The option to run the contract or not.
    """

    if isinstance(contract, Transition):
        return _adjust_element_transition_contract(
            contract=contract,
            contract_arguments=contract_arguments,
            element=element,
            adjustment_value=adjustment_value,
            run_contract=run_contract,
        )
    else:
        return _adjust_element_single_contract(
            contract=contract,
            contract_arguments=contract_arguments,
            element=element,
            adjustment_value=adjustment_value,
            run_contract=run_contract,
        )


@overload
def sensitivity_psc(
    contract: BaseProject | CostRecovery | GrossSplit | Transition,
    contract_arguments: dict[str, object],
    summary_arguments: dict[str, object],
    min_deviation: float,
    max_deviation: float,
    base_value: float = 1.0,
    step: int = 10,
    increment: float | None = None,
    dataframe_output: Literal[True] = True,
) -> _SensitivityFrames: ...


@overload
def sensitivity_psc(
    contract: BaseProject | CostRecovery | GrossSplit | Transition,
    contract_arguments: dict[str, object],
    summary_arguments: dict[str, object],
    min_deviation: float,
    max_deviation: float,
    base_value: float,
    step: int,
    increment: float | None,
    dataframe_output: Literal[False],
) -> _SensitivityValues: ...


@overload
def sensitivity_psc(
    contract: BaseProject | CostRecovery | GrossSplit | Transition,
    contract_arguments: dict[str, object],
    summary_arguments: dict[str, object],
    min_deviation: float,
    max_deviation: float,
    base_value: float = 1.0,
    step: int = 10,
    increment: float | None = None,
    *,
    dataframe_output: Literal[False],
) -> _SensitivityValues: ...


@overload
def sensitivity_psc(
    contract: BaseProject | CostRecovery | GrossSplit | Transition,
    contract_arguments: dict[str, object],
    summary_arguments: dict[str, object],
    min_deviation: float,
    max_deviation: float,
    base_value: float,
    step: int,
    increment: float | None,
    dataframe_output: bool,
) -> _SensitivityResult: ...


@overload
def sensitivity_psc(
    contract: BaseProject | CostRecovery | GrossSplit | Transition,
    contract_arguments: dict[str, object],
    summary_arguments: dict[str, object],
    min_deviation: float,
    max_deviation: float,
    base_value: float = 1.0,
    step: int = 10,
    increment: float | None = None,
    *,
    dataframe_output: bool,
) -> _SensitivityResult: ...


def sensitivity_psc(
    contract: BaseProject | CostRecovery | GrossSplit | Transition,
    contract_arguments: _ContractArguments,
    summary_arguments: _SummaryArguments,
    min_deviation: float,
    max_deviation: float,
    base_value: float = 1.0,
    step: int = 10,
    increment: float | None = None,
    dataframe_output: bool = True,
) -> _SensitivityResult:
    """
    Run one-factor-at-a-time sensitivity analysis on a PSC contract.

    Selected economic components (CAPEX, OPEX, prices, and lifting rates) are
    perturbed over a range of multipliers. For each perturbation, the contract
    is re-evaluated and key economic indicators are collected.

    Parameters
    ----------
    contract : BaseProject or CostRecovery or GrossSplit or Transition
        Contract instance to be analyzed.
    contract_arguments : dict
        Arguments passed to `_adjust_contract`.
    summary_arguments : dict
        Arguments passed to `get_summary()`.
    min_deviation, max_deviation : float
        Lower and upper bounds of sensitivity multipliers.
    base_value : float, optional
        Base multiplier (default 1.0).
    step : int, optional
        Number of multiplier steps (default 10).
    increment : float or None, optional
        Fixed spacing between consecutive multipliers. When provided, it
        overrides ``step`` and produces a uniform increment on both sides of
        the base value. Default is ``None``.
    dataframe_output : bool, optional
        If True, return results as pandas DataFrames.

    Returns
    -------
    dict or pandas.DataFrame
        Sensitivity results structured as
        `indicator -> multiplier -> element -> value`, or as DataFrames if
        `dataframe_output=True`.
    """

    # Get the multipliers
    multipliers = get_sensitivity_multipliers(
        min_deviation=min_deviation,
        max_deviation=max_deviation,
        base_value=base_value,
        step=step,
        increment=increment,
    )
    multiplier_values = _multiplier_values(multipliers)

    elements = [
        "CAPEX",
        "OPEX",
        "OILPRICE",
        "GASPRICE",
        "LIFTING",
        "OILLIFTING",
        "GASLIFTING",
    ]
    indicators = _SENSITIVITY_INDICATORS

    # Adjust the element of the contract and contain it in a list
    psc_adjusted_dict = {
        element: {
            mul: adjust_sensitivity_contract(
                contract=contract,
                contract_arguments=contract_arguments,
                element=element,
                adjustment_value=mul,
                run_contract=True,
            )
            for mul in multiplier_values
        }
        for element in elements
    }

    # Get the summary of each contract in psc_adjusted_dict and contain it in a dictionary.
    # dict: element -> multiplier = value
    summary_adjusted_dict: dict[str, dict[np.float64, _Summary]] = {}
    for element in elements:
        summary_adjusted_dict[element] = {
            mult: _get_summary(psc_adjusted_dict[element][mult], summary_arguments)
            for mult in multiplier_values
        }

    # Retrieve NPV, IRR, PI, POT, Gov Take, and Contractor Share.
    # Transform summary_adjusted_dict into the following structure:
    # dict: indicator -> multiplier -> element = value
    sensitivity_result: _SensitivityValues = {}
    for indicator in indicators:
        sensitivity_result[indicator] = {}
        for mult in multiplier_values:
            sensitivity_result[indicator][mult] = {
                element: summary_adjusted_dict[element][mult].get(indicator, None)
                for element in elements
            }

    """
    Previous Approach
    -----------------
    sensitivity_result = {
        indicator: {
            multiplier: {
                element: summary_adjusted_dict[element][multiplier].get(indicator, None)
                for element in summary_adjusted_dict
            }
            for multiplier in summary_adjusted_dict[next(iter(summary_adjusted_dict))]
        }
        for indicator in indicators
    }
    """

    # Transform result dictionary into DataFrames
    if dataframe_output is True:
        sensitivity_result_df: _SensitivityFrames = {}
        for indicator, data in sensitivity_result.items():
            dataframe = _sensitivity_dataframe(data)
            sensitivity_result_df[indicator] = (
                dataframe.reset_index()
                .rename(columns={"index": "Factor"})
                .set_index("Factor")
            )
        return sensitivity_result_df

    else:
        return sensitivity_result


"""
Previous approach
-----------------

def _adjust_element_single_contract(
        contract: BaseProject | CostRecovery | GrossSplit,
        contract_arguments: dict,
        element: str,
        adjustment_value: float,
        run_contract: bool = False,
) -> CostRecovery | GrossSplit:

    capital_adjusted = contract.capital_cost
    intangible_adjusted = contract.intangible_cost
    opex_adjusted = contract.opex
    asr_adjusted = contract.asr_cost
    lbt_adjusted = contract.lbt_cost
    cos_adjusted = contract.cost_of_sales
    lifting_adjusted = contract.lifting

    if element == "CAPEX":
        # Adjusting the Capital Cost of the contract
        capital_adjusted = tuple(
            [
                CapitalCost(
                    start_year=tan.start_year,
                    end_year=tan.end_year,
                    cost=tan.cost * adjustment_value,
                    expense_year=tan.expense_year,
                    cost_allocation=tan.cost_allocation,
                    description=tan.description,
                    tax_portion=tan.tax_portion,
                    tax_discount=tan.tax_discount,
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
                    cost=intang.cost * adjustment_value,
                    expense_year=intang.expense_year,
                    cost_allocation=intang.cost_allocation,
                    description=intang.description,
                    tax_portion=intang.tax_portion,
                    tax_discount=intang.tax_discount,
                )
                for intang in contract.intangible_cost
            ]
        )

    elif element == "OPEX":
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
                    tax_discount=opx.tax_discount,
                    fixed_cost=opx.fixed_cost * adjustment_value,
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
                    cost=asr.cost * adjustment_value,
                    expense_year=asr.expense_year,
                    cost_allocation=asr.cost_allocation,
                    description=asr.description,
                    tax_portion=asr.tax_portion,
                    tax_discount=asr.tax_discount,
                    final_year=asr.final_year,
                    future_rate=asr.future_rate,
                )
                for asr in contract.asr_cost
            ]
        )

        # Adjusting the LBT of the contract
        lbt_adjusted = tuple(
            [
                LBT(
                    start_year=bt.start_year,
                    end_year=bt.end_year,
                    expense_year=bt.expense_year,
                    cost=bt.cost * adjustment_value,
                    cost_allocation=bt.cost_allocation,
                    description=bt.description,
                    tax_portion=bt.tax_portion,
                    tax_discount=bt.tax_discount,
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

        # Adjusting the Intangible cost of the contract
        cos_adjusted = tuple(
            [
                CostOfSales(
                    start_year=cos.start_year,
                    end_year=cos.end_year,
                    cost=cos.cost * adjustment_value,
                    expense_year=cos.expense_year,
                    cost_allocation=cos.cost_allocation,
                    description=cos.description,
                    tax_portion=cos.tax_portion,
                    tax_discount=cos.tax_discount,
                )
                for cos in contract.cost_of_sales
            ]
        )

    elif element == "OILPRICE":
        # Adjusting the oil price
        lifting_adjusted = tuple(
            [
                Lifting(
                    start_year=lift.start_year,
                    end_year=lift.end_year,
                    lifting_rate=lift.lifting_rate,
                    price=(
                        lift.price * adjustment_value
                        if lift.fluid_type is FluidType.OIL
                        else lift.price
                    ),
                    prod_year=lift.prod_year,
                    fluid_type=lift.fluid_type,
                    ghv=lift.ghv,
                    prod_rate=lift.prod_rate,
                    prod_rate_baseline=lift.prod_rate_baseline,
                )
                for lift in contract.lifting
            ]
        )

    elif element == "GASPRICE":
        # Adjusting the oil price
        lifting_adjusted = tuple(
            [
                Lifting(
                    start_year=lift.start_year,
                    end_year=lift.end_year,
                    lifting_rate=lift.lifting_rate,
                    price=(
                        lift.price * adjustment_value
                        if lift.fluid_type is FluidType.GAS
                        else lift.price
                    ),
                    prod_year=lift.prod_year,
                    fluid_type=lift.fluid_type,
                    ghv=lift.ghv,
                    prod_rate=lift.prod_rate,
                    prod_rate_baseline=lift.prod_rate_baseline,
                )
                for lift in contract.lifting
            ]
        )

    elif element == "LIFTING":
        # Adjusting the lifting
        lifting_adjusted = tuple(
            [
                Lifting(
                    start_year=lift.start_year,
                    end_year=lift.end_year,
                    lifting_rate=lift.lifting_rate * adjustment_value,
                    price=lift.price,
                    prod_year=lift.prod_year,
                    fluid_type=lift.fluid_type,
                    ghv=lift.ghv,
                    # prod_rate=lift.prod_rate,
                    # prod_rate is being by passed in the routine of sensitivity,
                    # thus it will be filled with the lifting value
                    prod_rate_baseline=lift.prod_rate_baseline,
                )
                for lift in contract.lifting
            ]
        )

    elif element == "OILLIFTING":
        # Adjusting the lifting
        lifting_adjusted = tuple(
            [
                Lifting(
                    start_year=lift.start_year,
                    end_year=lift.end_year,
                    lifting_rate=(
                        lift.lifting_rate * adjustment_value
                        if lift.fluid_type is FluidType.OIL
                        else lift.lifting_rate
                    ),
                    price=lift.price,
                    prod_year=lift.prod_year,
                    fluid_type=lift.fluid_type,
                    ghv=lift.ghv,
                    # prod_rate=lift.prod_rate,
                    # prod_rate is being by passed in the routine of sensitivity,
                    # thus it will be filled with the lifting value
                    prod_rate_baseline=lift.prod_rate_baseline,
                )
                for lift in contract.lifting
            ]
        )

    elif element == "GASLIFTING":
        # Adjusting the lifting
        lifting_adjusted = tuple(
            [
                Lifting(
                    start_year=lift.start_year,
                    end_year=lift.end_year,
                    lifting_rate=(
                        lift.lifting_rate * adjustment_value
                        if lift.fluid_type is FluidType.GAS
                        else lift.lifting_rate
                    ),
                    price=lift.price,
                    prod_year=lift.prod_year,
                    fluid_type=lift.fluid_type,
                    ghv=lift.ghv,
                    # prod_rate=lift.prod_rate,
                    # prod_rate is being by passed in the routine of sensitivity,
                    # thus it will be filled with the lifting value
                    prod_rate_baseline=lift.prod_rate_baseline,
                )
                for lift in contract.lifting
            ]
        )

    else:
        raise SensitivityException(f"The element value, {element}, is not recognized.")

    # Adjust the contract with the adjusted element
    # On stream date treatment
    if np.sum(contract._oil_revenue) == 0:
        oil_onstream_date = None
    else:
        oil_onstream_date = contract.oil_onstream_date

    if np.sum(contract._gas_revenue) == 0:
        gas_onstream_date = None
    else:
        gas_onstream_date = contract.gas_onstream_date

    # When the contract is CostRecovery, parsing back the adjusted cost elements
    # to the cost recovery contract
    if isinstance(contract, CostRecovery):
        contract_adjusted = CostRecovery(
            start_date=contract.start_date,
            end_date=contract.end_date,
            oil_onstream_date=oil_onstream_date,
            gas_onstream_date=gas_onstream_date,
            lifting=lifting_adjusted,
            capital_cost=capital_adjusted,
            intangible_cost=intangible_adjusted,
            opex=opex_adjusted,
            asr_cost=asr_adjusted,
            lbt_cost=lbt_adjusted,
            cost_of_sales=cos_adjusted,
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
            gas_dmo_holiday_duration=contract.gas_dmo_holiday_duration,

            oil_carry_forward_depreciation=contract.oil_carry_forward_depreciation,
            gas_carry_forward_depreciation=contract.gas_carry_forward_depreciation,
        )

    # When the contract is GrossSplit, parsing back the adjusted cost elements
    # to the gross split contract
    elif isinstance(contract, GrossSplit):
        contract_adjusted = GrossSplit(
            start_date=contract.start_date,
            end_date=contract.end_date,
            oil_onstream_date=oil_onstream_date,
            gas_onstream_date=gas_onstream_date,
            lifting=contract.lifting,
            capital_cost=capital_adjusted,
            intangible_cost=intangible_adjusted,
            opex=opex_adjusted,
            asr_cost=asr_adjusted,
            lbt_cost=lbt_adjusted,
            cost_of_sales=cos_adjusted,
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
            gas_dmo_holiday_duration=contract.gas_dmo_holiday_duration,

            oil_carry_forward_depreciation=contract.oil_carry_forward_depreciation,
            gas_carry_forward_depreciation=contract.gas_carry_forward_depreciation,
        )

    elif isinstance(contract, BaseProject):
        contract_adjusted = BaseProject(
            start_date=contract.start_date,
            end_date=contract.end_date,
            oil_onstream_date=oil_onstream_date,
            gas_onstream_date=gas_onstream_date,
            lifting=contract.lifting,
            capital_cost=capital_adjusted,
            intangible_cost=intangible_adjusted,
            opex=opex_adjusted,
            asr_cost=asr_adjusted,
            lbt_cost=lbt_adjusted,
            cost_of_sales=cos_adjusted,
        )

    # When the contract is not recognized, raise an exception
    else:
        raise SensitivityException(
            f"The contract type, {type(contract)}, is not recognized."
        )
    if run_contract is True:
        contract_adjusted.run(**contract_arguments)
    else:
        pass

    return contract_adjusted
"""
