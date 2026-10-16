"""
Configure base project as the foundation (or parent class) for PSC contract.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from dataclasses import dataclass, field
from datetime import date
from functools import reduce
from typing import TypeVar, cast

from pyscnomics.econ._array import sum_float64 as _sum_float
from pyscnomics.econ.revenue import Lifting, aggregate_lifting_for_fluid
from pyscnomics.contracts.summary_profile import PostRunSummaryProfile
from pyscnomics.econ.selection import (
    FluidType,
    CostType,
    OtherRevenue,
    InflationAppliedTo,
    NPVSelection,
    DiscountingMode,
    ContractType,
)
from pyscnomics.econ.costs import (
    CapitalCost,
    Intangible,
    OPEX,
    ASR,
    LBT,
    CostOfSales,
    GeneralCost,
)
from pyscnomics.econ.indicator import (
    irr,
    npv_nominal_terms,
    npv_real_terms,
    npv_skk_nominal_terms,
    npv_skk_real_terms,
    npv_point_forward,
    pot_psc,
)
from pyscnomics.validation_rules import (
    CLASSIFICATION_ORDER,
    OnstreamResolutionError,
    approval_year_violation,
    boundary_year_classification_violation,
    classification_allowed_in_period,
    classification_period,
    dates_reversed,
    default_cost_type_for_year,
    inconsistent_component_years,
    pis_after_onstream_indices,
    pis_before_onstream_indices,
    resolve_onstream_year,
    sunk_cost_cutoff_year,
    sunk_cost_year_exceeds_cutoff,
)
from pyscnomics.input_validation import InputValidationError, engine_input_issue

# Set display for pandas dataframe
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_colwidth", 100)

_ArrayDType = TypeVar("_ArrayDType", bound=np.generic)
_ListValue = TypeVar("_ListValue")
WarningMessage = tuple[str, str]
TableArray = npt.NDArray[np.float64] | npt.NDArray[np.int64]


def _equals_mask(values: list[_ListValue], target: _ListValue) -> npt.NDArray[np.bool_]:
    matches: list[bool] = []
    for value in values:
        left: object = value
        right: object = target
        matches.append(left == right)
    return np.asarray(matches, dtype=np.bool_)


def _select_list(
    values: list[_ListValue], mask: npt.NDArray[np.bool_]
) -> list[_ListValue]:
    return [value for index, value in enumerate(values) if bool(mask.item(index))]


def _require_float_array(value: object, field_name: str) -> npt.NDArray[np.float64]:
    if not isinstance(value, np.ndarray):
        raise BaseProjectException(f"Attribute {field_name} must be a NumPy array")
    return np.asarray(value, dtype=np.float64)


def require_float_array(value: object, field_name: str) -> npt.NDArray[np.float64]:
    """Validate and return a concrete float array for cross-module callers."""
    return _require_float_array(value, field_name)


def _require_int_array(value: object, field_name: str) -> npt.NDArray[np.int64]:
    if not isinstance(value, np.ndarray):
        raise BaseProjectException(f"Attribute {field_name} must be a NumPy array")
    return np.asarray(value, dtype=np.int64)


def _require_fluid_list(
    value: list[_ListValue] | None, field_name: str
) -> list[FluidType]:
    if not isinstance(value, list):
        raise BaseProjectException(f"Attribute {field_name} must be a FluidType list")
    items: list[object] = list(value)
    fluid_types: list[FluidType] = []
    for item in items:
        if not isinstance(item, FluidType):
            raise BaseProjectException(
                f"Attribute {field_name} must be a FluidType list"
            )
        fluid_types.append(item)
    return fluid_types


def _require_cost_type_list(
    value: list[_ListValue] | None, field_name: str
) -> list[CostType | None]:
    if not isinstance(value, list):
        raise BaseProjectException(f"Attribute {field_name} must be a CostType list")
    items: list[object] = list(value)
    cost_types: list[CostType | None] = []
    for item in items:
        if item is not None and not isinstance(item, CostType):
            raise BaseProjectException(
                f"Attribute {field_name} must be a CostType list"
            )
        cost_types.append(item)
    return cost_types


def _require_str_list(value: list[_ListValue] | None, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise BaseProjectException(f"Attribute {field_name} must be a string list")
    items: list[object] = list(value)
    strings: list[str] = []
    for item in items:
        if not isinstance(item, str):
            raise BaseProjectException(f"Attribute {field_name} must be a string list")
        strings.append(item)
    return strings


def _require_bool_list(value: list[_ListValue] | None, field_name: str) -> list[bool]:
    if not isinstance(value, list):
        raise BaseProjectException(f"Attribute {field_name} must be a boolean list")
    items: list[object] = list(value)
    booleans: list[bool] = []
    for item in items:
        if not isinstance(item, bool):
            raise BaseProjectException(f"Attribute {field_name} must be a boolean list")
        booleans.append(item)
    return booleans


def _is_tuple(value: object) -> bool:
    return isinstance(value, tuple)


def _is_bool(value: object) -> bool:
    return isinstance(value, bool)


@dataclass(frozen=True)
class _NormalizedCostData:
    expense_year: npt.NDArray[np.int64]
    cost: npt.NDArray[np.float64]
    cost_allocation: list[FluidType]
    cost_type: list[CostType | None]
    description: list[str]
    tax_portion: npt.NDArray[np.float64]
    tax_discount: npt.NDArray[np.float64]


@dataclass(frozen=True)
class _NormalizedCapitalCostData:
    cost: _NormalizedCostData
    pis_year: npt.NDArray[np.int64]
    salvage_value: npt.NDArray[np.float64]
    useful_life: npt.NDArray[np.float64]
    depreciation_factor: npt.NDArray[np.float64]
    is_ic_applied: list[bool]


@dataclass(frozen=True)
class _NormalizedOpexData:
    cost: _NormalizedCostData
    fixed_cost: npt.NDArray[np.float64]
    prod_rate: npt.NDArray[np.float64]
    cost_per_volume: npt.NDArray[np.float64]


@dataclass(frozen=True)
class _NormalizedAsrData:
    cost: _NormalizedCostData
    final_year: npt.NDArray[np.int64]
    future_rate: npt.NDArray[np.float64]


@dataclass(frozen=True)
class _NormalizedLbtData:
    cost: _NormalizedCostData
    final_year: npt.NDArray[np.int64]
    utilized_land_area: npt.NDArray[np.float64]
    utilized_building_area: npt.NDArray[np.float64]
    njop_land: npt.NDArray[np.float64]
    njop_building: npt.NDArray[np.float64]
    gross_revenue: npt.NDArray[np.float64]


def _normalize_cost_data(cost: GeneralCost) -> _NormalizedCostData:
    return _NormalizedCostData(
        expense_year=_require_int_array(cost.expense_year, "expense_year"),
        cost=_require_float_array(cost.cost, "cost"),
        cost_allocation=_require_fluid_list(cost.cost_allocation, "cost_allocation"),
        cost_type=_require_cost_type_list(cost.cost_type, "cost_type"),
        description=_require_str_list(cost.description, "description"),
        tax_portion=_require_float_array(cost.tax_portion, "tax_portion"),
        tax_discount=_require_float_array(cost.tax_discount, "tax_discount"),
    )


def _normalize_capital_cost_data(cost: CapitalCost) -> _NormalizedCapitalCostData:
    return _NormalizedCapitalCostData(
        cost=_normalize_cost_data(cost),
        pis_year=_require_int_array(cost.pis_year, "pis_year"),
        salvage_value=_require_float_array(cost.salvage_value, "salvage_value"),
        useful_life=_require_float_array(cost.useful_life, "useful_life"),
        depreciation_factor=_require_float_array(
            cost.depreciation_factor, "depreciation_factor"
        ),
        is_ic_applied=_require_bool_list(cost.is_ic_applied, "is_ic_applied"),
    )


def _normalize_opex_data(cost: OPEX) -> _NormalizedOpexData:
    return _NormalizedOpexData(
        cost=_normalize_cost_data(cost),
        fixed_cost=_require_float_array(cost.fixed_cost, "fixed_cost"),
        prod_rate=_require_float_array(cost.prod_rate, "prod_rate"),
        cost_per_volume=_require_float_array(cost.cost_per_volume, "cost_per_volume"),
    )


def _normalize_asr_data(cost: ASR) -> _NormalizedAsrData:
    return _NormalizedAsrData(
        cost=_normalize_cost_data(cost),
        final_year=_require_int_array(cost.final_year, "final_year"),
        future_rate=_require_float_array(cost.future_rate, "future_rate"),
    )


def _normalize_lbt_data(cost: LBT) -> _NormalizedLbtData:
    return _NormalizedLbtData(
        cost=_normalize_cost_data(cost),
        final_year=_require_int_array(cost.final_year, "final_year"),
        utilized_land_area=_require_float_array(
            cost.utilized_land_area, "utilized_land_area"
        ),
        utilized_building_area=_require_float_array(
            cost.utilized_building_area, "utilized_building_area"
        ),
        njop_land=_require_float_array(cost.njop_land, "njop_land"),
        njop_building=_require_float_array(cost.njop_building, "njop_building"),
        gross_revenue=_require_float_array(cost.gross_revenue, "gross_revenue"),
    )


def _assemble_baseproject_table_columns(
    *,
    years: TableArray,
    lifting: TableArray,
    price: TableArray,
    revenue: TableArray,
    sulfur_lifting: TableArray,
    sulfur_price: TableArray,
    sulfur_revenue: TableArray,
    electricity_lifting: TableArray,
    electricity_price: TableArray,
    electricity_revenue: TableArray,
    co2_lifting: TableArray,
    co2_price: TableArray,
    co2_revenue: TableArray,
    sunk_cost_depreciable: TableArray,
    sunk_cost_non_depreciable: TableArray,
    sunk_cost: TableArray,
    preonstream_depreciable: TableArray,
    preonstream_non_depreciable: TableArray,
    preonstream: TableArray,
    postonstream_depreciable: TableArray,
    postonstream_non_depreciable: TableArray,
    postonstream: TableArray,
    capital_expenditures_pre_tax: TableArray,
    intangible_expenditures_pre_tax: TableArray,
    opex_expenditures_pre_tax: TableArray,
    asr_expenditures_pre_tax: TableArray,
    lbt_expenditures_pre_tax: TableArray,
    cost_of_sales_expenditures_pre_tax: TableArray,
    capital_indirect_tax: TableArray,
    intangible_indirect_tax: TableArray,
    opex_indirect_tax: TableArray,
    asr_indirect_tax: TableArray,
    lbt_indirect_tax: TableArray,
    cost_of_sales_indirect_tax: TableArray,
    capital_postonstream: TableArray,
    intangible_postonstream: TableArray,
    opex_postonstream: TableArray,
    asr_postonstream: TableArray,
    lbt_postonstream: TableArray,
    cost_of_sales_postonstream: TableArray,
    expenses_capital: TableArray,
    expenses_non_capital: TableArray,
    expenses_total: TableArray,
    cashflow: TableArray,
) -> dict[str, TableArray]:
    """Assemble one ordered BaseProject table from explicitly supplied arrays."""
    return {
        "years": years,
        "lifting": lifting,
        "price": price,
        "revenue": revenue,
        "lifting_sulfur": sulfur_lifting,
        "price_sulfur": sulfur_price,
        "revenue_sulfur": sulfur_revenue,
        "lifting_electricity": electricity_lifting,
        "price_electricity": electricity_price,
        "revenue_electricity": electricity_revenue,
        "lifting_co2": co2_lifting,
        "price_co2": co2_price,
        "revenue_co2": co2_revenue,
        "sunk_cost_depreciable": sunk_cost_depreciable,
        "sunk_cost_non_depreciable": sunk_cost_non_depreciable,
        "sunk_cost": sunk_cost,
        "preonstream_depreciable": preonstream_depreciable,
        "preonstream_non_depreciable": preonstream_non_depreciable,
        "preonstream": preonstream,
        "postonstream_depreciable": postonstream_depreciable,
        "postonstream_non_depreciable": postonstream_non_depreciable,
        "postonstream": postonstream,
        "capital_expenditures_pre_tax": capital_expenditures_pre_tax,
        "intangible_expenditures_pre_tax": intangible_expenditures_pre_tax,
        "opex_expenditures_pre_tax": opex_expenditures_pre_tax,
        "asr_expenditures_pre_tax": asr_expenditures_pre_tax,
        "lbt_expenditures_pre_tax": lbt_expenditures_pre_tax,
        "cost_of_sales_expenditures_pre_tax": cost_of_sales_expenditures_pre_tax,
        "capital_indirect_tax": capital_indirect_tax,
        "intangible_indirect_tax": intangible_indirect_tax,
        "opex_indirect_tax": opex_indirect_tax,
        "asr_indirect_tax": asr_indirect_tax,
        "lbt_indirect_tax": lbt_indirect_tax,
        "cost_of_sales_indirect_tax": cost_of_sales_indirect_tax,
        "capital_postonstream": capital_postonstream,
        "intangible_postonstream": intangible_postonstream,
        "opex_postonstream": opex_postonstream,
        "asr_postonstream": asr_postonstream,
        "lbt_postonstream": lbt_postonstream,
        "cost_of_sales_postonstream": cost_of_sales_postonstream,
        "expenses_capital": expenses_capital,
        "expenses_non_capital": expenses_non_capital,
        "expenses_total": expenses_total,
        "cashflow": cashflow,
    }


class BaseProjectException(Exception):
    """Exception to be raised for a misuse of BaseProject class"""

    pass


class BaseProjectInputError(  # pyright: ignore[reportUnsafeMultipleInheritance]
    BaseProjectException, InputValidationError
):
    """Verified user-input project error retaining ``BaseProjectException``."""

    _input_code: str = "BASE_PROJECT_INPUT_INVALID"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        InputValidationError.__init__(
            self,
            (engine_input_issue(code=code or self._input_code, message=message),),
        )


class OtherRevenueException(Exception):
    """Exception to be raised for a misuse of Other Revenue"""

    pass


class SunkCostException(Exception):
    """Exception to be raised for a misuse of _get_sunk_cost() method"""

    pass


class SunkCostInputError(  # pyright: ignore[reportUnsafeMultipleInheritance]
    SunkCostException, InputValidationError
):
    """Verified user-input sunk-cost error retaining ``SunkCostException``."""

    _input_code: str = "SUNK_COST_AFTER_CUTOFF"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        InputValidationError.__init__(
            self,
            (engine_input_issue(code=code or self._input_code, message=message),),
        )


class PreOnstreamException(Exception):
    """Exception to be raised for an incorrect preonstream cost configuration"""

    pass


class PreOnstreamInputError(  # pyright: ignore[reportUnsafeMultipleInheritance]
    PreOnstreamException, InputValidationError
):
    """Verified user-input pre-onstream error retaining ``PreOnstreamException``."""

    _input_code: str = "PREONSTREAM_YEAR_OUTSIDE_PROJECT"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        InputValidationError.__init__(
            self,
            (engine_input_issue(code=code or self._input_code, message=message),),
        )


class PostOnstreamException(Exception):
    """Exception to be raised for incorrect postonstream cost configurations"""

    pass


class PostOnstreamInputError(  # pyright: ignore[reportUnsafeMultipleInheritance]
    PostOnstreamException, InputValidationError
):
    """Verified user-input post-onstream error retaining ``PostOnstreamException``."""

    _input_code: str = "POSTONSTREAM_YEAR_BEFORE_ONSTREAM"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        InputValidationError.__init__(
            self,
            (engine_input_issue(code=code or self._input_code, message=message),),
        )


class BaseProjectSummaryException(Exception):
    """Exception to be raised for a misuse of get_summary() method"""

    pass


@dataclass
class BaseProject:
    """
    Represents a base project with start and end dates, lifting information,
    capital and intangible costs, operational expenses (OPEX), ASR costs,
    LBT costs, and Cost of Sales.

    Parameters
    ----------
    start_date : date
        The start date of the project.
    end_date : date
        The end date of the project.
    oil_onstream_date: date
        The start date of oil production.
    gas_onstream_date: date
        The start date of gas production.
    lifting : tuple[Lifting]
        A tuple of lifting information objects.
    capital_cost : tuple[CapitalCost]
        A tuple of capital cost objects.
    intangible_cost : tuple[Intangible], optional
        A tuple of intangible cost objects. Defaults to None.
    opex : tuple[OPEX], optional
        A tuple of operational expense objects. Defaults to None.
    asr_cost : tuple[ASR], optional
        A tuple of ASR (Abandonment and Site Restoration) cost objects.
        Defaults to None.
    lbt_cost: tuple[LBT], optional
        A tuple of LBT (Land and Building Tax) cost objects.
        Defaults to None.
    cost_of_sales : tuple[CostOfSales]
        A tuple of CostOfSales objects. Defaults to None.
    """

    # List of required arguments
    start_date: date
    end_date: date
    approval_year: int
    oil_onstream_date: date | None = field(default=None)
    gas_onstream_date: date | None = field(default=None)
    is_pod_1: bool | None = field(default=False)
    is_strict: bool | None = field(default=True)
    lifting: tuple[Lifting, ...] | None = field(default=None)
    capital_cost: tuple[CapitalCost, ...] | None = field(default=None)
    intangible_cost: tuple[Intangible, ...] | None = field(default=None)
    opex: tuple[OPEX, ...] | None = field(default=None)
    asr_cost: tuple[ASR, ...] | None = field(default=None)
    lbt_cost: tuple[LBT, ...] | None = field(default=None)
    cost_of_sales: tuple[CostOfSales, ...] | None = field(default=None)

    # Attributes associated with project duration
    project_duration: int = field(init=False)
    project_years: npt.NDArray[np.int64] = field(init=False)
    _has_run: bool = field(default=False, init=False, repr=False)

    # Attributes associated with lifting for each fluid types
    _oil_lifting: Lifting = field(init=False, repr=False)
    _gas_lifting: Lifting = field(init=False, repr=False)
    _sulfur_lifting: Lifting = field(init=False, repr=False)
    _electricity_lifting: Lifting = field(init=False, repr=False)
    _co2_lifting: Lifting = field(init=False, repr=False)

    # Attributes associated with revenue for each fluid types
    _oil_revenue: npt.NDArray[np.float64] = field(init=False, repr=False)
    _gas_revenue: npt.NDArray[np.float64] = field(init=False, repr=False)
    _sulfur_revenue: npt.NDArray[np.float64] = field(init=False, repr=False)
    _electricity_revenue: npt.NDArray[np.float64] = field(init=False, repr=False)
    _co2_revenue: npt.NDArray[np.float64] = field(init=False, repr=False)

    # Atributes associated with WAP price
    _oil_wap_price: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_wap_price: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _sulfur_wap_price: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _electricity_wap_price: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _co2_wap_price: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )

    # Attributes associated with total cost per component
    capital_cost_total: CapitalCost = field(init=False, repr=False)
    intangible_cost_total: Intangible = field(init=False, repr=False)
    opex_total: OPEX = field(init=False, repr=False)
    asr_cost_total: ASR = field(init=False, repr=False)
    lbt_cost_total: LBT = field(init=False, repr=False)
    cost_of_sales_total: CostOfSales = field(init=False, repr=False)

    # Attributes associated with post onstream cost in each cost categories
    _oil_capital_postonstream: CapitalCost = field(init=False, repr=False)
    _gas_capital_postonstream: CapitalCost = field(init=False, repr=False)
    _oil_intangible_postonstream: Intangible = field(init=False, repr=False)
    _gas_intangible_postonstream: Intangible = field(init=False, repr=False)
    _oil_opex_postonstream: OPEX = field(init=False, repr=False)
    _gas_opex_postonstream: OPEX = field(init=False, repr=False)
    _oil_asr_postonstream: ASR = field(init=False, repr=False)
    _gas_asr_postonstream: ASR = field(init=False, repr=False)
    _oil_lbt_postonstream: LBT = field(init=False, repr=False)
    _gas_lbt_postonstream: LBT = field(init=False, repr=False)
    _oil_cost_of_sales_postonstream: CostOfSales = field(init=False, repr=False)
    _gas_cost_of_sales_postonstream: CostOfSales = field(init=False, repr=False)

    # Attributes associated with preonstream cost in each cost categories
    _oil_capital_preonstream: CapitalCost = field(init=False, repr=False)
    _gas_capital_preonstream: CapitalCost = field(init=False, repr=False)
    _oil_intangible_preonstream: Intangible = field(init=False, repr=False)
    _gas_intangible_preonstream: Intangible = field(init=False, repr=False)
    _oil_opex_preonstream: OPEX = field(init=False, repr=False)
    _gas_opex_preonstream: OPEX = field(init=False, repr=False)
    _oil_asr_preonstream: ASR = field(init=False, repr=False)
    _gas_asr_preonstream: ASR = field(init=False, repr=False)
    _oil_lbt_preonstream: LBT = field(init=False, repr=False)
    _gas_lbt_preonstream: LBT = field(init=False, repr=False)
    _oil_cost_of_sales_preonstream: CostOfSales = field(init=False, repr=False)
    _gas_cost_of_sales_preonstream: CostOfSales = field(init=False, repr=False)

    # Attributes associated with sunkcost in each cost categories
    _oil_capital_sunk_cost: CapitalCost = field(init=False, repr=False)
    _gas_capital_sunk_cost: CapitalCost = field(init=False, repr=False)
    _oil_intangible_sunk_cost: Intangible = field(init=False, repr=False)
    _gas_intangible_sunk_cost: Intangible = field(init=False, repr=False)
    _oil_opex_sunk_cost: OPEX = field(init=False, repr=False)
    _gas_opex_sunk_cost: OPEX = field(init=False, repr=False)
    _oil_asr_sunk_cost: ASR = field(init=False, repr=False)
    _gas_asr_sunk_cost: ASR = field(init=False, repr=False)
    _oil_lbt_sunk_cost: LBT = field(init=False, repr=False)
    _gas_lbt_sunk_cost: LBT = field(init=False, repr=False)
    _oil_cost_of_sales_sunk_cost: CostOfSales = field(init=False, repr=False)
    _gas_cost_of_sales_sunk_cost: CostOfSales = field(init=False, repr=False)

    # Attributes associated with sunk costs
    _oil_depreciable_sunk_cost: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_depreciable_sunk_cost: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _oil_non_depreciable_sunk_cost: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_non_depreciable_sunk_cost: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _oil_sunk_cost: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_sunk_cost: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )

    # Attributes associated with preonstream costs
    _oil_depreciable_preonstream: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_depreciable_preonstream: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _oil_non_depreciable_preonstream: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_non_depreciable_preonstream: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _oil_preonstream: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_preonstream: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )

    # Attributes associated with postonstream costs
    _oil_depreciable_postonstream: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_depreciable_postonstream: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _oil_non_depreciable_postonstream: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_non_depreciable_postonstream: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _oil_postonstream: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_postonstream: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )

    # Attributes associated with pre tax expenditures for each cost categories
    _oil_capital_expenditures_pre_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_capital_expenditures_pre_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _oil_intangible_expenditures_pre_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_intangible_expenditures_pre_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _oil_opex_expenditures_pre_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_opex_expenditures_pre_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _oil_asr_expenditures_pre_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_asr_expenditures_pre_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _oil_lbt_expenditures_pre_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_lbt_expenditures_pre_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _oil_cost_of_sales_expenditures_pre_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_cost_of_sales_expenditures_pre_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )

    # Attributes associated with indirect taxes for each cost categories
    _oil_capital_indirect_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_capital_indirect_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _oil_intangible_indirect_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_intangible_indirect_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _oil_opex_indirect_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_opex_indirect_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _oil_asr_indirect_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_asr_indirect_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _oil_lbt_indirect_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_lbt_indirect_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _oil_cost_of_sales_indirect_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_cost_of_sales_indirect_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )

    # Attributes associated with post tax expenditures for each cost categories
    _oil_capital_expenditures_post_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_capital_expenditures_post_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _oil_intangible_expenditures_post_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_intangible_expenditures_post_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _oil_opex_expenditures_post_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_opex_expenditures_post_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _oil_asr_expenditures_post_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_asr_expenditures_post_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _oil_lbt_expenditures_post_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_lbt_expenditures_post_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _oil_cost_of_sales_expenditures_post_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_cost_of_sales_expenditures_post_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )

    # Attributes associated with total expenditures and indirect taxes for each fluid
    _oil_total_expenditures_pre_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_total_expenditures_pre_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _oil_total_indirect_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_total_indirect_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _oil_total_expenditures_post_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_total_expenditures_post_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )

    # Attributes associated with capital costs, non capital costs. and total expenses
    _oil_capital: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_capital: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _oil_non_capital: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_non_capital: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _oil_total_expenses: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_total_expenses: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )

    # Attributes associated with cashflow
    _oil_cashflow: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _gas_cashflow: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )

    # Attribute to store warning messages
    warning_messages: list[WarningMessage] = field(
        default_factory=list, init=False, repr=False
    )

    # Attributes associated with consolidated profiles
    _consolidated_lifting: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_wap_price: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_revenue: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )

    _consolidated_depreciable_sunk_cost: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_non_depreciable_sunk_cost: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_sunk_cost: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )

    _consolidated_depreciable_preonstream: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_non_depreciable_preonstream: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_preonstream: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )

    _consolidated_depreciable_postonstream: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_non_depreciable_postonstream: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_postonstream: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )

    _consolidated_capital_expenditures_pre_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_intangible_expenditures_pre_tax: npt.NDArray[np.float64] | None = (
        field(default=None, init=False, repr=False)
    )
    _consolidated_opex_expenditures_pre_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_asr_expenditures_pre_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_lbt_expenditures_pre_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_cost_of_sales_expenditures_pre_tax: npt.NDArray[np.float64] | None = (
        field(default=None, init=False, repr=False)
    )
    _consolidated_total_expenditures_pre_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )

    _consolidated_capital_indirect_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_intangible_indirect_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_opex_indirect_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_asr_indirect_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_lbt_indirect_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_cost_of_sales_indirect_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_total_indirect_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )

    _consolidated_capital_expenditures_post_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_intangible_expenditures_post_tax: npt.NDArray[np.float64] | None = (
        field(default=None, init=False, repr=False)
    )
    _consolidated_opex_expenditures_post_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_asr_expenditures_post_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_lbt_expenditures_post_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_cost_of_sales_expenditures_post_tax: (
        npt.NDArray[np.float64] | None
    ) = field(default=None, init=False, repr=False)
    _consolidated_total_expenditures_post_tax: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )

    _consolidated_capital: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_non_capital: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_total_expenses: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_cashflow: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self):
        """
        Handles the following operations/procedures:
        -   Prepare attributes project_duration and project_years;
        -   Prepare attribute lifting;
        -   Prepare attributes associated with lifting for each fluid types.
            The attributes are: _oil_lifting, _gas_lifting, _sulfur_lifting,
            _electricity_lifting, and _co2_lifting;
        -   Prepare attributes associated with revenue for each fluid types.
            The attributes are: _oil_revenue, _gas_revenue, _sulfur_revenue,
            _electricity_revenue, and _co2_revenue;
        -   Prepare attribute oil_onstream_date;
        -   Prepare attribute gas_onstream_date;
        -   Prepare attribute capital_cost;
        -   Prepare attribute intangible_cost;
        -   Prepare attribute opex;
        -   Prepare attribute asr_cost;
        -   Prepare attribute lbt_cost;
        -   Prepare attribute cost_of_sales;
        -   Prepare attributes associated with total cost per component
            (leveraging dunder method __add__). The attributes are:
            capital_cost_total, intangible_cost_total, opex_total, asr_cost_total,
            lbt_cost_total, and cost_of_sales_total;
        -   Classify cost categories by fluid
        -   Modify cost_type in each cost categories, accounting for engineering sense
        -   Define postonstream, sunk cost, and preonstream attributes
        -   Raise an exception if the start year of the project is inconsistent;
        -   Raise an exception if the end year of the project is inconsistent.
        """

        # Prepare attributes project_duration and project_years
        if not dates_reversed(self.start_date, self.end_date):
            self.project_duration = self.end_date.year - self.start_date.year + 1
            self.project_years = np.arange(
                self.start_date.year, self.end_date.year + 1, 1, dtype=np.int64
            )

        else:
            raise BaseProjectInputError(
                f"start date {self.start_date} "
                + f"is after the end date: {self.end_date}"
            )

        # Prepare attribute "is_pod_1"
        if self.is_pod_1 is None:
            self.is_pod_1 = False
        elif not _is_bool(self.is_pod_1):
            raise BaseProjectException(
                f"Attribute is_pod_1 must be a boolean, not a/an {self.is_pod_1.__class__.__qualname__}"
            )
        # Prepare attribute "is_strict"
        if self.is_strict is None:
            self.is_strict = True
        elif not _is_bool(self.is_strict):
            raise BaseProjectException(
                f"Attribute is_strict must be a boolean, not a/an {self.is_strict.__class__.__qualname__}"
            )
        # Prepare attribute lifting (for OIL, GAS, SULFUR, ELECTRICITY, and CO2)
        if self.lifting is None:
            self.lifting = (
                Lifting(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    prod_year=self.project_years.copy(),
                    lifting_rate=np.zeros(self.project_duration),
                    price=np.zeros(self.project_duration),
                    fluid_type=FluidType.OIL,
                ),
                Lifting(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    prod_year=self.project_years.copy(),
                    lifting_rate=np.zeros(self.project_duration),
                    price=np.zeros(self.project_duration),
                    fluid_type=FluidType.GAS,
                ),
                Lifting(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    prod_year=self.project_years.copy(),
                    lifting_rate=np.zeros(self.project_duration),
                    price=np.zeros(self.project_duration),
                    fluid_type=FluidType.SULFUR,
                ),
                Lifting(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    prod_year=self.project_years.copy(),
                    lifting_rate=np.zeros(self.project_duration),
                    price=np.zeros(self.project_duration),
                    fluid_type=FluidType.ELECTRICITY,
                ),
                Lifting(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    prod_year=self.project_years.copy(),
                    lifting_rate=np.zeros(self.project_duration),
                    price=np.zeros(self.project_duration),
                    fluid_type=FluidType.CO2,
                ),
            )
        elif not _is_tuple(self.lifting):
            raise BaseProjectException(
                "Attribute lifting must be provided as a tuple of Lifting instances, "
                + f"not as an/a {self.lifting.__class__.__qualname__}"
            )
        # Prepare attributes associated with lifting for each fluid types
        self._oil_lifting = self._get_oil_lifting()
        self._gas_lifting = self._get_gas_lifting()
        self._sulfur_lifting = self._get_sulfur_lifting()
        self._electricity_lifting = self._get_electricity_lifting()
        self._co2_lifting = self._get_co2_lifting()

        # Prepare attributes associated with revenues for each fluid types
        self._oil_revenue = self._oil_lifting.revenue()
        self._gas_revenue = self._gas_lifting.revenue()
        self._sulfur_revenue = self._sulfur_lifting.revenue()
        self._electricity_revenue = self._electricity_lifting.revenue()
        self._co2_revenue = self._co2_lifting.revenue()

        # Prepare onstream dates for OIL and GAS
        self.oil_onstream_date = self._get_onstream_date(fluid="oil")
        self.gas_onstream_date = self._get_onstream_date(fluid="gas")

        # Prepare attribute capital_cost (for both OIL and GAS)
        if self.capital_cost is None:
            self.capital_cost = (
                CapitalCost(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    expense_year=np.array([self.start_date.year]),
                    cost=np.array([0]),
                    cost_allocation=[FluidType.OIL],
                ),
                CapitalCost(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    expense_year=np.array([self.start_date.year]),
                    cost=np.array([0]),
                    cost_allocation=[FluidType.GAS],
                ),
            )
        elif not _is_tuple(self.capital_cost):
            raise BaseProjectException(
                "Attribute capital_cost must be provided as a tuple of CapitalCost "
                + f"instances, not as an/a {self.capital_cost.__class__.__qualname__}"
            )
        # Prepare attribute intangible_cost (for both OIL and GAS)
        if self.intangible_cost is None:
            self.intangible_cost = (
                Intangible(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    expense_year=np.array([self.start_date.year]),
                    cost=np.array([0]),
                    cost_allocation=[FluidType.OIL],
                ),
                Intangible(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    expense_year=np.array([self.start_date.year]),
                    cost=np.array([0]),
                    cost_allocation=[FluidType.GAS],
                ),
            )
        elif not _is_tuple(self.intangible_cost):
            raise BaseProjectException(
                "Attribute intangible_cost must be provided as a tuple of Intangible "
                + f"instances, not as an/a {self.intangible_cost.__class__.__qualname__}"
            )
        # Prepare attribute opex (for both OIL and GAS)
        if self.opex is None:
            self.opex = (
                OPEX(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    expense_year=np.array([self.start_date.year]),
                    fixed_cost=np.array([0]),
                    cost_allocation=[FluidType.OIL],
                ),
                OPEX(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    expense_year=np.array([self.start_date.year]),
                    fixed_cost=np.array([0]),
                    cost_allocation=[FluidType.GAS],
                ),
            )
        elif not _is_tuple(self.opex):
            raise BaseProjectException(
                "Attribute opex must be provided as a tuple of OPEX instances, "
                + f"not as an/a {self.opex.__class__.__qualname__}"
            )
        # Prepare attribute asr_cost (for both OIL and GAS)
        if self.asr_cost is None:
            self.asr_cost = (
                ASR(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    expense_year=np.array([self.start_date.year]),
                    cost=np.array([0]),
                    cost_allocation=[FluidType.OIL],
                ),
                ASR(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    expense_year=np.array([self.start_date.year]),
                    cost=np.array([0]),
                    cost_allocation=[FluidType.GAS],
                ),
            )
        elif not _is_tuple(self.asr_cost):
            raise BaseProjectException(
                "Attribute asr_cost must be provided as a tuple of ASR instances, "
                + f"not as an/a {self.asr_cost.__class__.__qualname__}"
            )
        # Prepare attribute lbt_cost (for both OIL and GAS)
        if self.lbt_cost is None:
            self.lbt_cost = (
                LBT(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    expense_year=np.array([self.start_date.year]),
                    cost=np.array([0]),
                    cost_allocation=[FluidType.OIL],
                ),
                LBT(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    expense_year=np.array([self.start_date.year]),
                    cost=np.array([0]),
                    cost_allocation=[FluidType.GAS],
                ),
            )
        elif not _is_tuple(self.lbt_cost):
            raise BaseProjectException(
                "Attribute lbt_cost must be provided as a tuple of LBT instances, "
                + f"not as an/a {self.lbt_cost.__class__.__qualname__}"
            )
        # Prepare attribute cost_of_sales (for both OIL and GAS)
        if self.cost_of_sales is None:
            self.cost_of_sales = (
                CostOfSales(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    expense_year=np.array([self.start_date.year]),
                    cost=np.array([0]),
                    cost_allocation=[FluidType.OIL],
                ),
                CostOfSales(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    expense_year=np.array([self.start_date.year]),
                    cost=np.array([0]),
                    cost_allocation=[FluidType.GAS],
                ),
            )
        elif not _is_tuple(self.cost_of_sales):
            raise BaseProjectException(
                "Attribute cost_of_sales must be provided as a tuple of CostOfSales "
                + f"instances, not as an/a {self.cost_of_sales.__class__.__qualname__}"
            )
        # Prepare attributes associated with total cost per component
        self.capital_cost_total = reduce(lambda x, y: x + y, self.capital_cost)
        self.intangible_cost_total = reduce(lambda x, y: x + y, self.intangible_cost)
        self.opex_total = reduce(lambda x, y: x + y, self.opex)
        self.asr_cost_total = reduce(lambda x, y: x + y, self.asr_cost)
        self.lbt_cost_total = reduce(lambda x, y: x + y, self.lbt_cost)
        self.cost_of_sales_total = reduce(lambda x, y: x + y, self.cost_of_sales)

        # Classify cost categories by fluid
        capital_cost = {
            "oil": self.classify_capital_cost_by_fluid(FluidType.OIL),
            "gas": self.classify_capital_cost_by_fluid(FluidType.GAS),
        }
        intangible = {
            "oil": self._classify_intangible_cost_by_fluid(FluidType.OIL),
            "gas": self._classify_intangible_cost_by_fluid(FluidType.GAS),
        }
        opex = {
            "oil": self._classify_opex_by_fluid(FluidType.OIL),
            "gas": self._classify_opex_by_fluid(FluidType.GAS),
        }
        asr = {
            "oil": self._classify_asr_cost_by_fluid(FluidType.OIL),
            "gas": self._classify_asr_cost_by_fluid(FluidType.GAS),
        }
        lbt = {
            "oil": self._classify_lbt_cost_by_fluid(FluidType.OIL),
            "gas": self._classify_lbt_cost_by_fluid(FluidType.GAS),
        }
        cost_of_sales = {
            "oil": self.classify_cost_of_sales_by_fluid(FluidType.OIL),
            "gas": self.classify_cost_of_sales_by_fluid(FluidType.GAS),
        }

        # Modify cost_type in each cost categories, accounting for engineering sense
        fluid_types = [FluidType.OIL.name.lower(), FluidType.GAS.name.lower()]

        costs_list = [
            capital_cost,
            intangible,
            opex,
            asr,
            lbt,
            cost_of_sales,
        ]

        for cost in costs_list:
            for ftype in fluid_types:
                self._prepare_cost_types(cost_obj=cost[ftype])

        # Define lifecycle costs explicitly so every stored field remains visible to
        # both the type checker and future accounting changes.
        self._oil_capital_postonstream = self._filter_capital_cost(
            capital_cost["oil"], CostType.POST_ONSTREAM_COST
        )
        self._gas_capital_postonstream = self._filter_capital_cost(
            capital_cost["gas"], CostType.POST_ONSTREAM_COST
        )
        self._oil_capital_preonstream = self._filter_capital_cost(
            capital_cost["oil"], CostType.PRE_ONSTREAM_COST
        )
        self._gas_capital_preonstream = self._filter_capital_cost(
            capital_cost["gas"], CostType.PRE_ONSTREAM_COST
        )
        self._oil_capital_sunk_cost = self._filter_capital_cost(
            capital_cost["oil"], CostType.SUNK_COST
        )
        self._gas_capital_sunk_cost = self._filter_capital_cost(
            capital_cost["gas"], CostType.SUNK_COST
        )
        self._oil_intangible_postonstream = self._filter_intangible(
            intangible["oil"], CostType.POST_ONSTREAM_COST
        )
        self._gas_intangible_postonstream = self._filter_intangible(
            intangible["gas"], CostType.POST_ONSTREAM_COST
        )
        self._oil_intangible_preonstream = self._filter_intangible(
            intangible["oil"], CostType.PRE_ONSTREAM_COST
        )
        self._gas_intangible_preonstream = self._filter_intangible(
            intangible["gas"], CostType.PRE_ONSTREAM_COST
        )
        self._oil_intangible_sunk_cost = self._filter_intangible(
            intangible["oil"], CostType.SUNK_COST
        )
        self._gas_intangible_sunk_cost = self._filter_intangible(
            intangible["gas"], CostType.SUNK_COST
        )
        self._oil_opex_postonstream = self._filter_opex(
            opex["oil"], CostType.POST_ONSTREAM_COST
        )
        self._gas_opex_postonstream = self._filter_opex(
            opex["gas"], CostType.POST_ONSTREAM_COST
        )
        self._oil_opex_preonstream = self._filter_opex(
            opex["oil"], CostType.PRE_ONSTREAM_COST
        )
        self._gas_opex_preonstream = self._filter_opex(
            opex["gas"], CostType.PRE_ONSTREAM_COST
        )
        self._oil_opex_sunk_cost = self._filter_opex(opex["oil"], CostType.SUNK_COST)
        self._gas_opex_sunk_cost = self._filter_opex(opex["gas"], CostType.SUNK_COST)
        self._oil_asr_postonstream = self._filter_asr(
            asr["oil"], CostType.POST_ONSTREAM_COST
        )
        self._gas_asr_postonstream = self._filter_asr(
            asr["gas"], CostType.POST_ONSTREAM_COST
        )
        self._oil_asr_preonstream = self._filter_asr(
            asr["oil"], CostType.PRE_ONSTREAM_COST
        )
        self._gas_asr_preonstream = self._filter_asr(
            asr["gas"], CostType.PRE_ONSTREAM_COST
        )
        self._oil_asr_sunk_cost = self._filter_asr(asr["oil"], CostType.SUNK_COST)
        self._gas_asr_sunk_cost = self._filter_asr(asr["gas"], CostType.SUNK_COST)
        self._oil_lbt_postonstream = self._filter_lbt(
            lbt["oil"], CostType.POST_ONSTREAM_COST
        )
        self._gas_lbt_postonstream = self._filter_lbt(
            lbt["gas"], CostType.POST_ONSTREAM_COST
        )
        self._oil_lbt_preonstream = self._filter_lbt(
            lbt["oil"], CostType.PRE_ONSTREAM_COST
        )
        self._gas_lbt_preonstream = self._filter_lbt(
            lbt["gas"], CostType.PRE_ONSTREAM_COST
        )
        self._oil_lbt_sunk_cost = self._filter_lbt(lbt["oil"], CostType.SUNK_COST)
        self._gas_lbt_sunk_cost = self._filter_lbt(lbt["gas"], CostType.SUNK_COST)
        self._oil_cost_of_sales_postonstream = self._filter_cost_of_sales(
            cost_of_sales["oil"], CostType.POST_ONSTREAM_COST
        )
        self._gas_cost_of_sales_postonstream = self._filter_cost_of_sales(
            cost_of_sales["gas"], CostType.POST_ONSTREAM_COST
        )
        self._oil_cost_of_sales_preonstream = self._filter_cost_of_sales(
            cost_of_sales["oil"], CostType.PRE_ONSTREAM_COST
        )
        self._gas_cost_of_sales_preonstream = self._filter_cost_of_sales(
            cost_of_sales["gas"], CostType.PRE_ONSTREAM_COST
        )
        self._oil_cost_of_sales_sunk_cost = self._filter_cost_of_sales(
            cost_of_sales["oil"], CostType.SUNK_COST
        )
        self._gas_cost_of_sales_sunk_cost = self._filter_cost_of_sales(
            cost_of_sales["gas"], CostType.SUNK_COST
        )

        # Raise an exception error if the start year of the project is inconsistent
        self._check_inconsistent_start_year()

        # Raise an exception error if the end year of the project is inconsistent
        self._check_inconsistent_end_year()

    def _get_onstream_date(self, fluid: str) -> date:
        """
        Resolve and validate onstream date for a given fluid (oil | gas).

        Logic:
        - If revenue exists:
          • onstream_date = first revenue year (if not provided)
          • otherwise → validate against project period & allowed years
        - If no revenue → fallback to project end date

        Parameters
        ----------
        fluid : str
            Fluid type → {"oil", "gas"}.

        Returns
        -------
        datetime.date
            Validated (or inferred) onstream date.

        Raises
        ------
        BaseProjectException
            If fluid is invalid or onstream year violates project / lifting rules.

        TL;DR:
        Safely infer or validate oil/gas onstream date based on revenue, lifting, and project bounds.
        """

        # Specify allowed fluid types
        if fluid not in {"oil", "gas"}:
            raise BaseProjectException(
                f"Invalid fluid type: {fluid}. Must choose between 'oil' or 'gas'."
            )

        if fluid == "oil":
            onstream_date = self.oil_onstream_date
            lifting = self._oil_lifting
            revenue = self._oil_revenue
        else:
            onstream_date = self.gas_onstream_date
            lifting = self._gas_lifting
            revenue = self._gas_revenue

        provided_year = onstream_date.year if onstream_date is not None else None

        # Delegate the effective-onstream rule to the shared pure resolver so
        # preflight and the engine cannot drift. The resolver implements: infer
        # the earliest positive-revenue year when no date is provided; validate
        # a provided date against the project period and the allowed lifting /
        # revenue years; and fall back to the project end date when there is no
        # positive revenue at all.
        try:
            resolved = resolve_onstream_year(
                fluid=fluid,
                provided_year=provided_year,
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                revenue=revenue,
                lifting_prod_years=list(lifting.prod_year),
            )
        except OnstreamResolutionError as exc:
            raise BaseProjectInputError(str(exc)) from exc

        # No positive revenue always falls back to the project end date,
        # regardless of any supplied date.
        if resolved.source == "project_end":
            return self.end_date

        if onstream_date is not None:
            return onstream_date

        return date(year=resolved.year, month=1, day=1)

    def _require_onstream_dates(self) -> tuple[date, date]:
        oil_onstream_date = self.oil_onstream_date
        gas_onstream_date = self.gas_onstream_date
        if oil_onstream_date is None or gas_onstream_date is None:
            raise BaseProjectException("Oil and gas onstream dates must be available")
        return oil_onstream_date, gas_onstream_date

    def _require_array(
        self,
        value: npt.NDArray[_ArrayDType] | None,
        field_name: str,
    ) -> npt.NDArray[_ArrayDType]:
        if value is None:
            raise BaseProjectException(f"Attribute {field_name} is missing")
        return value

    def _get_lifting_by_commodity(self, commodity: FluidType) -> Lifting:
        """
        Get the aggregated lifting data for a specific commodity type.

        If the commodity doesn't exist in the project's lifting data, returns a zero-filled
        Lifting object with the project's time parameters. Otherwise, returns the sum of all
        lifting data for the specified commodity.

        Parameters
        ----------
        commodity : FluidType
            The fluid commodity type to retrieve the associated lifting data.
            Options:    - FluidType.OIL
                        - FluidType.GAS
                        - FluidType.SULFUR
                        - FluidType.ELECTRICITY
                        - FluidType.CO2

        Returns
        -------
        Lifting
            A Lifting object containing either:
            - Zero-filled arrays if the commodity doesn't exist in lifting data.
            - The sum of all lifting data for the specified commodity.

        Notes
        -----
        The sum of all lifting instances of a particular commodity is carried out
        using dunder method __add__() of class Lifting.
        """

        lifting = self.lifting
        if lifting is None:
            raise BaseProjectException("Attribute lifting is missing")

        return aggregate_lifting_for_fluid(
            lifting,
            fluid_type=commodity,
            start_year=self.start_date.year,
            end_year=self.end_date.year,
        )

    def _get_oil_lifting(self) -> Lifting:
        """
        Get the aggregated lifting data for oil commodity.
        Equivalent to calling `_get_lifting_by_commodity(FluidType.OIL)`

        Returns
        -------
        Lifting
            A Lifting object containing either:
            - Zero-filled arrays if no oil lifting data exists
            - The sum of all oil lifting data if available

        See Also
        --------
        _get_commodity_lifting : The general method for any fluid type
        """
        return self._get_lifting_by_commodity(commodity=FluidType.OIL)

    def _get_gas_lifting(self) -> Lifting:
        """
        Get the aggregated lifting data for gas commodity.
        Equivalent to calling `_get_lifting_by_commodity(FluidType.GAS)`

        Returns
        -------
        Lifting
            A Lifting object containing either:
            - Zero-filled arrays if no gas lifting data exists
            - The sum of all gas lifting data if available

        See Also
        --------
        _get_commodity_lifting : The general method for any fluid type
        """
        return self._get_lifting_by_commodity(commodity=FluidType.GAS)

    def _get_sulfur_lifting(self) -> Lifting:
        """
        Get the aggregated lifting data for sulfur commodity.
        Equivalent to calling `_get_lifting_by_commodity(FluidType.SULFUR)`

        Returns
        -------
        Lifting
            A Lifting object containing either:
            - Zero-filled arrays if no sulfur lifting data exists
            - The sum of all sulfur lifting data if available

        See Also
        --------
        _get_commodity_lifting : The general method for any fluid type
        """
        return self._get_lifting_by_commodity(commodity=FluidType.SULFUR)

    def _get_electricity_lifting(self) -> Lifting:
        """
        Get the aggregated lifting data for electricity commodity.
        Equivalent to calling `_get_lifting_by_commodity(FluidType.ELECTRICITY)`.

        Returns
        -------
        Lifting
            A Lifting object containing either:
            - Zero-filled arrays if no electricity production data exists
            - The sum of all electricity production data if available

        See Also
        --------
        _get_commodity_lifting : The general method for any fluid type
        """
        return self._get_lifting_by_commodity(commodity=FluidType.ELECTRICITY)

    def _get_co2_lifting(self) -> Lifting:
        """
        Get the aggregated lifting data for CO₂ commodity.
        Equivalent to calling `_get_lifting_by_commodity(FluidType.CO2)`.

        Returns
        -------
        Lifting
            A Lifting object containing either:
            - Zero-filled arrays if no CO₂ data exists
            - The sum of all CO₂ lifting data if available

        See Also
        --------
        _get_commodity_lifting : The general method for any fluid type
        """
        return self._get_lifting_by_commodity(commodity=FluidType.CO2)

    def get_lifting_by_fluid(self, fluid_type: FluidType) -> Lifting:
        """Return the project-wide lifting aggregate for one fluid."""

        return self._get_lifting_by_commodity(commodity=fluid_type)

    def _calc_wap_price(self, fluidtype: FluidType) -> npt.NDArray[np.float64]:
        """
        Compute the Weighted Average Price (WAP) for a given fluid type.

        This method calculates the WAP price by weighting the price of the fluid type
        by its lifting volume over the project duration. If the total volume is zero
        for any time step, the corresponding WAP price is set to zero.

        Parameters
        ----------
        fluidtype : FluidType
            The fluid type (e.g., OIL, GAS, SULFUR) for which to compute the WAP price.

        Returns
        -------
        np.ndarray
            A NumPy array representing the WAP price for each project year.

        Notes
        -----
        - The calculation considers only lifting instances that match the specified
          fluid type.
        - If `total_vol` is zero for any time step, the result is explicitly set to
          zero to avoid division errors.
        """
        lifting = self.lifting
        if lifting is None:
            raise BaseProjectException("Attribute lifting is missing")

        # Initialize attributes with zero values with length = project_duration
        vol_x_price = np.zeros_like(self.project_years, dtype=np.float64)
        total_vol = np.zeros_like(self.project_years, dtype=np.float64)

        # Add contributions from every FLUID TYPE lifting instances
        for lft in lifting:
            if lft.fluid_type == fluidtype:
                lft_rate = lft.get_lifting_rate_ghv_arr()
                vol_x_price += lft_rate * lft.get_price_arr()
                total_vol += lft_rate

        # Calculate FLUID WAP price
        return np.divide(
            vol_x_price, total_vol, where=(total_vol > 0), out=np.zeros_like(total_vol)
        )

    def _get_oil_wap_price(self) -> None:
        """
        Compute and store the Weighted Average Price (WAP) for oil.

        This method calculates the WAP price for oil by calling `_calc_wap_price`
        with `FluidType.OIL` and stores the result in `self._oil_wap_price`.

        Returns
        -------
        None
            The computed oil WAP price is stored in `self._oil_wap_price`.

        Notes
        -----
        - The WAP price is computed based on the lifting rate and price for oil.
        - The stored `_oil_wap_price` is a NumPy array representing the WAP price
          for each project year.
        """

        self._oil_wap_price = self._calc_wap_price(fluidtype=FluidType.OIL)

    def _get_gas_wap_price(self) -> None:
        """
        Compute and store the Weighted Average Price (WAP) for gas.

        This method calculates the WAP price for gas by calling `_calc_wap_price`
        with `FluidType.GAS` and stores the result in `self._gas_wap_price`.

        Returns
        -------
        None
            The computed gas WAP price is stored in `self._gas_wap_price`.

        Notes
        -----
        - The WAP price is computed based on the lifting rate and price for gas.
        - The stored `_gas_wap_price` is a NumPy array representing the WAP price
          for each project year.
        """

        self._gas_wap_price = self._calc_wap_price(fluidtype=FluidType.GAS)

    def _get_sulfur_wap_price(self) -> None:
        """
        Compute and store the Weighted Average Price (WAP) for sulfur.

        This method calculates the WAP price for sulfur by calling `_calc_wap_price`
        with `FluidType.SULFUR` and stores the result in `self._sulfur_wap_price`.

        Returns
        -------
        None
            The computed sulfur WAP price is stored in `self._sulfur_wap_price`.

        Notes
        -----
        - The WAP price is computed based on the lifting rate and price for sulfur.
        - The stored `_sulfur_wap_price` is a NumPy array representing the WAP price
          for each project year.
        """

        self._sulfur_wap_price = self._calc_wap_price(fluidtype=FluidType.SULFUR)

    def _get_electricity_wap_price(self) -> None:
        """
        Compute and store the Weighted Average Price (WAP) for electricity.

        This method calculates the WAP price for electricity by calling `_calc_wap_price`
        with `FluidType.ELECTRICITY` and stores the result in `self._electricity_wap_price`.

        Returns
        -------
        None
            The computed electricity WAP price is stored in `self._electricity_wap_price`.

        Notes
        -----
        - The WAP price is computed based on the lifting rate and price for electricity.
        - The stored `_electricity_wap_price` is a NumPy array representing the WAP price
          for each project year.
        """

        self._electricity_wap_price = self._calc_wap_price(
            fluidtype=FluidType.ELECTRICITY
        )

    def _get_co2_wap_price(self) -> None:
        """
        Compute and store the Weighted Average Price (WAP) for CO₂.

        This method calculates the WAP price for CO₂ by calling `_calc_wap_price`
        with `FluidType.CO2` and stores the result in `self._co2_wap_price`.

        Returns
        -------
        None
            The computed CO₂ WAP price is stored in `self._co2_wap_price`.

        Notes
        -----
        - The WAP price is computed based on the lifting rate and price for CO₂.
        - The stored `_co2_wap_price` is a NumPy array representing the WAP price
          for each project year.
        """

        self._co2_wap_price = self._calc_wap_price(fluidtype=FluidType.CO2)

    def _get_wap_price(self) -> None:
        """
        Compute and store the Weighted Average Price (WAP) for all fluid types.

        This method sequentially calculates the WAP prices for oil, gas, sulfur,
        electricity, and CO₂ by calling their respective WAP calculation methods.

        The following methods are invoked:
        -   `_get_oil_wap_price`: Computes the WAP for oil.
        -   `_get_gas_wap_price`: Computes the WAP for gas.
        -   `_get_sulfur_wap_price`: Computes the WAP for sulfur.
        -   `_get_electricity_wap_price`: Computes the WAP for electricity.
        -   `_get_co2_wap_price`: Computes the WAP for CO2.

        Returns
        -------
        None
            The computed WAP prices for all fluid types are stored in their respective
            attributes.

            The method updates the following attributes with their corresponding WAP values:
            -   `self._oil_wap_price`
            -   `self._gas_wap_price`
            -   `self._sulfur_wap_price`
            -   `self._electricity_wap_price`
            -   `self._co2_wap_price`

        Notes
        -----
        - This method calls `_get_oil_wap_price`, `_get_gas_wap_price`, `_get_sulfur_wap_price`,
          `_get_electricity_wap_price`, and `_get_co2_wap_price` to compute WAP prices.
        - Each fluid type's WAP price is stored as a NumPy array in its corresponding attribute.
        """

        self._get_oil_wap_price()
        self._get_gas_wap_price()
        self._get_sulfur_wap_price()
        self._get_electricity_wap_price()
        self._get_co2_wap_price()

    def classify_capital_cost_by_fluid(self, fluid_type: FluidType) -> CapitalCost:
        """
        Classify and filter capital costs by fluid type (oil or gas).

        This method extracts the portion of the total capital cost
        (`capital_cost_total`) that is allocated to a given fluid type.
        If the specified fluid type does not exist in the cost allocation,
        a zero-cost `CapitalCost` object is returned as a placeholder.

        Parameters
        ----------
        fluid_type : FluidType
            The fluid type to filter capital costs by.
            Typically one of ``FluidType.OIL`` or ``FluidType.GAS``.

        Returns
        -------
        CapitalCost
            A `CapitalCost` object containing only the costs allocated
            to the specified fluid type. If no allocation exists for
            the fluid, a zero-cost object with default values is returned.

        Notes
        -----
        - The filtering is performed by masking the `cost_allocation` array
          of the total capital cost object.
        - If the fluid type is not found, a placeholder is returned with:
          ``expense_year = [start_year]`` and ``cost = [0]``.
        """

        cct = self.capital_cost_total
        cost_data = _normalize_cost_data(cct)

        if fluid_type not in cost_data.cost_allocation:
            oil_onstream_date, gas_onstream_date = self._require_onstream_dates()

            onstream_yr = {
                FluidType.OIL: oil_onstream_date.year,
                FluidType.GAS: gas_onstream_date.year,
            }[fluid_type]

            return CapitalCost(
                start_year=cct.start_year,
                end_year=cct.end_year,
                expense_year=np.array([cct.start_year]),
                cost=np.array([0]),
                cost_allocation=[fluid_type],
                pis_year=np.array([onstream_yr]),
            )

        else:
            mask = _equals_mask(cost_data.cost_allocation, fluid_type)
            pis_year = _require_int_array(cct.pis_year, "pis_year")
            salvage_value = _require_float_array(cct.salvage_value, "salvage_value")
            useful_life = _require_float_array(cct.useful_life, "useful_life")
            depreciation_factor = _require_float_array(
                cct.depreciation_factor, "depreciation_factor"
            )
            is_ic_applied = _require_bool_list(cct.is_ic_applied, "is_ic_applied")

            return CapitalCost(
                start_year=cct.start_year,
                end_year=cct.end_year,
                expense_year=cost_data.expense_year[mask],
                cost=cost_data.cost[mask],
                cost_allocation=_select_list(cost_data.cost_allocation, mask),
                cost_type=_select_list(cost_data.cost_type, mask),
                description=_select_list(cost_data.description, mask),
                tax_portion=cost_data.tax_portion[mask],
                tax_discount=cost_data.tax_discount[mask],
                pis_year=pis_year[mask],
                salvage_value=salvage_value[mask],
                useful_life=useful_life[mask],
                depreciation_factor=depreciation_factor[mask],
                is_ic_applied=_select_list(is_ic_applied, mask),
            )

    def _classify_intangible_cost_by_fluid(self, fluid_type: FluidType) -> Intangible:
        """
        Classify and filter intangible costs by fluid type (oil or gas).

        This method extracts the portion of the total intangible cost
        (`intangible_cost_total`) that is allocated to a given fluid type.
        If the specified fluid type does not exist in the cost allocation,
        a zero-cost `Intangible` object is returned as a placeholder.

        Parameters
        ----------
        fluid_type : FluidType
            The fluid type to filter intangible costs by.
            Typically one of ``FluidType.OIL`` or ``FluidType.GAS``.

        Returns
        -------
        Intangible
            An `Intangible` object containing only the costs allocated
            to the specified fluid type. If no allocation exists for
            the fluid, a zero-cost object with default values is returned.

        Notes
        -----
        - The filtering is performed by masking the `cost_allocation` array
          of the total intangible cost object.
        - If the fluid type is not found, a placeholder is returned with:
          ``expense_year = [start_year]`` and ``cost = [0]``.
        """

        ict = self.intangible_cost_total
        cost_data = _normalize_cost_data(ict)

        if fluid_type not in cost_data.cost_allocation:
            return Intangible(
                start_year=ict.start_year,
                end_year=ict.end_year,
                expense_year=np.array([ict.start_year]),
                cost=np.array([0]),
                cost_allocation=[fluid_type],
            )

        else:
            mask = _equals_mask(cost_data.cost_allocation, fluid_type)

            return Intangible(
                start_year=ict.start_year,
                end_year=ict.end_year,
                expense_year=cost_data.expense_year[mask],
                cost=cost_data.cost[mask],
                cost_allocation=_select_list(cost_data.cost_allocation, mask),
                description=_select_list(cost_data.description, mask),
                cost_type=_select_list(cost_data.cost_type, mask),
                tax_portion=cost_data.tax_portion[mask],
                tax_discount=cost_data.tax_discount[mask],
            )

    def _classify_opex_by_fluid(self, fluid_type: FluidType) -> OPEX:
        """
        Classify and filter operating expenditure (OPEX) by fluid type (oil or gas).

        This method extracts the portion of the total OPEX (`opex_total`)
        that is allocated to a given fluid type. If the specified fluid type
        does not exist in the cost allocation, a zero-cost `OPEX` object is
        returned as a placeholder.

        Parameters
        ----------
        fluid_type : FluidType
            The fluid type to filter OPEX costs by.
            Typically one of ``FluidType.OIL`` or ``FluidType.GAS``.

        Returns
        -------
        OPEX
            An `OPEX` object containing only the costs allocated to the specified
            fluid type. If no allocation exists for the fluid, a zero-cost object
            with default values is returned.

        Notes
        -----
        - The filtering is performed by masking the `cost_allocation` array
          of the total OPEX object.
        - If the fluid type is not found, a placeholder is returned with:
          ``expense_year = [start_year]``, ``fixed_cost = [0]``, and
          ``cost_allocation = [fluid_type]``.
        """

        ot = self.opex_total
        cost_data = _normalize_cost_data(ot)
        if fluid_type not in cost_data.cost_allocation:
            return OPEX(
                start_year=ot.start_year,
                end_year=ot.end_year,
                expense_year=np.array([ot.start_year]),
                fixed_cost=np.array([0]),
                cost_allocation=[fluid_type],
            )

        else:
            mask = _equals_mask(cost_data.cost_allocation, fluid_type)
            fixed_cost = _require_float_array(ot.fixed_cost, "fixed_cost")
            prod_rate = _require_float_array(ot.prod_rate, "prod_rate")
            cost_per_volume = _require_float_array(
                ot.cost_per_volume, "cost_per_volume"
            )

            return OPEX(
                start_year=ot.start_year,
                end_year=ot.end_year,
                expense_year=cost_data.expense_year[mask],
                cost_allocation=_select_list(cost_data.cost_allocation, mask),
                cost_type=_select_list(cost_data.cost_type, mask),
                description=_select_list(cost_data.description, mask),
                tax_portion=cost_data.tax_portion[mask],
                tax_discount=cost_data.tax_discount[mask],
                fixed_cost=fixed_cost[mask],
                prod_rate=prod_rate[mask],
                cost_per_volume=cost_per_volume[mask],
            )

    def _classify_asr_cost_by_fluid(self, fluid_type: FluidType) -> ASR:
        """
        Classify and filter Abandonment and Site Restoration (ASR) costs
        by fluid type (oil or gas).

        This method extracts the portion of the total ASR costs
        (`asr_cost_total`) that is allocated to a given fluid type.
        If the specified fluid type does not exist in the cost allocation,
        a zero-cost `ASR` object is returned as a placeholder.

        Parameters
        ----------
        fluid_type : FluidType
            The fluid type to filter ASR costs by.
            Typically one of ``FluidType.OIL`` or ``FluidType.GAS``.

        Returns
        -------
        ASR
            An `ASR` object containing only the costs allocated to the
            specified fluid type. If no allocation exists for the fluid,
            a zero-cost object with default values is returned.

        Notes
        -----
        - Filtering is done by applying a boolean mask to the
          `cost_allocation` array of the total ASR object.
        - If the fluid type is not found, a placeholder is returned with:
          ``expense_year = [start_year]``, ``cost = [0]``, and
          ``cost_allocation = [fluid_type]``.
        """

        act = self.asr_cost_total
        cost_data = _normalize_cost_data(act)
        if fluid_type not in cost_data.cost_allocation:
            return ASR(
                start_year=act.start_year,
                end_year=act.end_year,
                expense_year=np.array([act.start_year]),
                cost=np.array([0]),
                cost_allocation=[fluid_type],
            )

        else:
            mask = _equals_mask(cost_data.cost_allocation, fluid_type)
            final_year = _require_int_array(act.final_year, "final_year")
            future_rate = _require_float_array(act.future_rate, "future_rate")

            return ASR(
                start_year=act.start_year,
                end_year=act.end_year,
                expense_year=cost_data.expense_year[mask],
                cost=cost_data.cost[mask],
                cost_allocation=_select_list(cost_data.cost_allocation, mask),
                cost_type=_select_list(cost_data.cost_type, mask),
                description=_select_list(cost_data.description, mask),
                tax_portion=cost_data.tax_portion[mask],
                tax_discount=cost_data.tax_discount[mask],
                final_year=final_year[mask],
                future_rate=future_rate[mask],
            )

    def _classify_lbt_cost_by_fluid(self, fluid_type: FluidType) -> LBT:
        """
        Classify and filter Land and Building Tax (LBT) costs
        by fluid type (oil or gas).

        This method extracts the portion of the total LBT costs
        (`lbt_cost_total`) that is allocated to a given fluid type.
        If the specified fluid type does not exist in the cost allocation,
        a zero-cost `LBT` object is returned as a placeholder.

        Parameters
        ----------
        fluid_type : FluidType
            The fluid type to filter LBT costs by.
            Typically one of ``FluidType.OIL`` or ``FluidType.GAS``.

        Returns
        -------
        LBT
            An `LBT` object containing only the costs allocated to the
            specified fluid type. If no allocation exists for the fluid,
            a zero-cost object with default values is returned.

        Notes
        -----
        - Filtering is performed using a boolean mask applied to the
          `cost_allocation` array of the total LBT object.
        - If the fluid type is not found, a placeholder is returned with:
          ``expense_year = [start_year]``, ``cost = [0]``, and
          ``cost_allocation = [fluid_type]``.
        """

        lct = self.lbt_cost_total
        cost_data = _normalize_cost_data(lct)
        if fluid_type not in cost_data.cost_allocation:
            return LBT(
                start_year=lct.start_year,
                end_year=lct.end_year,
                expense_year=np.array([lct.start_year]),
                cost=np.array([0]),
                cost_allocation=[fluid_type],
            )

        else:
            mask = _equals_mask(cost_data.cost_allocation, fluid_type)
            final_year = _require_int_array(lct.final_year, "final_year")
            utilized_land_area = _require_float_array(
                lct.utilized_land_area, "utilized_land_area"
            )
            utilized_building_area = _require_float_array(
                lct.utilized_building_area, "utilized_building_area"
            )
            njop_land = _require_float_array(lct.njop_land, "njop_land")
            njop_building = _require_float_array(lct.njop_building, "njop_building")
            gross_revenue = _require_float_array(lct.gross_revenue, "gross_revenue")

            return LBT(
                start_year=lct.start_year,
                end_year=lct.end_year,
                expense_year=cost_data.expense_year[mask],
                cost_allocation=_select_list(cost_data.cost_allocation, mask),
                cost_type=_select_list(cost_data.cost_type, mask),
                description=_select_list(cost_data.description, mask),
                tax_portion=cost_data.tax_portion[mask],
                tax_discount=cost_data.tax_discount[mask],
                final_year=final_year[mask],
                utilized_land_area=utilized_land_area[mask],
                utilized_building_area=utilized_building_area[mask],
                njop_land=njop_land[mask],
                njop_building=njop_building[mask],
                gross_revenue=gross_revenue[mask],
                cost=cost_data.cost[mask],
            )

    def classify_cost_of_sales_by_fluid(self, fluid_type: FluidType) -> CostOfSales:
        """
        Classify and filter Cost of Sales (CoS) by fluid type (oil or gas).

        This method extracts the portion of the total Cost of Sales
        (`cost_of_sales_total`) that is allocated to a given fluid type.
        If the specified fluid type does not exist in the cost allocation,
        a zero-cost `CostOfSales` object is returned as a placeholder.

        Parameters
        ----------
        fluid_type : FluidType
            The fluid type to filter Cost of Sales by.
            Typically one of ``FluidType.OIL`` or ``FluidType.GAS``.

        Returns
        -------
        CostOfSales
            A `CostOfSales` object containing only the costs allocated
            to the specified fluid type. If no allocation exists for
            the fluid, a zero-cost object with default values is returned.

        Notes
        -----
        - Filtering is performed using a boolean mask applied to the
          `cost_allocation` array of the total Cost of Sales object.
        - If the fluid type is not found, a placeholder is returned with:
          ``expense_year = [start_year]``, ``cost = [0]``, and
          ``cost_allocation = [fluid_type]``.
        """

        cst = self.cost_of_sales_total
        cost_data = _normalize_cost_data(cst)
        if fluid_type not in cost_data.cost_allocation:
            return CostOfSales(
                start_year=cst.start_year,
                end_year=cst.end_year,
                expense_year=np.array([cst.start_year]),
                cost=np.array([0]),
                cost_allocation=[fluid_type],
            )

        else:
            mask = _equals_mask(cost_data.cost_allocation, fluid_type)

            return CostOfSales(
                start_year=cst.start_year,
                end_year=cst.end_year,
                expense_year=cost_data.expense_year[mask],
                cost=cost_data.cost[mask],
                cost_allocation=_select_list(cost_data.cost_allocation, mask),
                cost_type=_select_list(cost_data.cost_type, mask),
                description=_select_list(cost_data.description, mask),
                tax_portion=cost_data.tax_portion[mask],
                tax_discount=cost_data.tax_discount[mask],
            )

    def _validate_approval_year(self) -> None:
        """
        Validate the project approval year against project timeline constraints.

        The method checks that ``approval_year`` is provided, is an integer,
        and falls within the valid project window. Specifically, the approval
        year must not precede the project start year, exceed the project end
        year, or occur after the earliest onstream year (oil or gas).

        Raises
        ------
        BaseProjectException
            If ``approval_year`` is missing.
        BaseProjectException
            If ``approval_year`` is not an integer.
        BaseProjectException
            If ``approval_year`` is earlier than the project start year.
        BaseProjectException
            If ``approval_year`` is later than the project end year.
        BaseProjectException
            If ``approval_year`` is later than the earliest onstream year.

        Notes
        -----
        The earliest onstream year is derived from both oil and gas
        onstream dates.
        """

        oil_onstream_date, gas_onstream_date = self._require_onstream_dates()
        onstream_yr = min([oil_onstream_date.year, gas_onstream_date.year])

        # Delegate the approval-year rule to the shared predicate so the
        # preflight and the engine cannot drift.
        violation = approval_year_violation(
            self.approval_year,
            start_year=self.start_date.year,
            end_year=self.end_date.year,
            onstream_year=onstream_yr,
        )

        # Raise an error: approval year is before the start year of the project
        if violation == "before_project":
            raise BaseProjectInputError(
                f"Approval year ({self.approval_year}) is before the project "
                + f"start year ({self.start_date.year})",
                code="APPROVAL_YEAR_OUTSIDE_PROJECT",
            )

        # Raise an error: approval year is after the end year of the project
        if violation == "after_project":
            raise BaseProjectInputError(
                f"Approval year ({self.approval_year}) is after the project "
                + f"end year ({self.end_date.year})",
                code="APPROVAL_YEAR_OUTSIDE_PROJECT",
            )

        # Raise an error: approval year is after onstream year
        if violation == "after_onstream":
            raise BaseProjectInputError(
                f"Approval year ({self.approval_year}) as after "
                + f"onstream year ({onstream_yr})",
                code="APPROVAL_YEAR_AFTER_ONSTREAM",
            )

    def _assign_default_cost_types(
        self,
        onstream_year: int,
        cost_obj: CapitalCost | Intangible | OPEX | ASR | LBT | CostOfSales,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.object_]]:
        """
        Assign default cost types based on reference years and project milestones.

        This method assigns default ``CostType`` values to cost entries whose cost
        type is not explicitly provided. Defaults are determined from the expense
        year relative to the project approval year and onstream year. For non-POD I
        projects, sunk costs are not assigned as default values.

        Parameters
        ----------
        onstream_year : int
            Project onstream year used to distinguish pre- and post-onstream periods.
        cost_obj : CapitalCost | Intangible | OPEX | ASR | LBT | CostOfSales
            Cost object containing expense years and (optional) cost type
            classifications.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple ``(ry, ct)`` where ``ry`` is the array of expense years and ``ct``
            is the array of assigned cost types.

        Notes
        -----
        - Default assignment applies only to entries with ``None`` as cost type.
        - For non-POD I projects, sunk costs default to pre-onstream costs.
        """

        # Extract cost types and reference (expense) years
        ct = np.array(cost_obj.cost_type)
        ry = _require_int_array(cost_obj.expense_year, "expense_year")

        # Assign the engine default to every unset classification through the
        # shared predicate; an explicit classification is never overwritten.
        # For non-POD I projects, sunk rows default to pre-onstream costs.
        years = cast("list[int]", ry.tolist())
        existing = cast("list[object]", ct.tolist())
        for index in range(len(existing)):
            if existing[index] is not None:
                continue
            ct[index] = CostType(
                default_cost_type_for_year(
                    years[index],
                    approval_year=self.approval_year,
                    onstream_year=onstream_year,
                    is_pod_1=bool(self.is_pod_1),
                )
            )

        return ry, ct

    def _validate_cost_type_consistency(
        self,
        ct: npt.NDArray[np.object_],
        ry: npt.NDArray[np.int64],
        onstream_year: int,
    ) -> None:
        """
        Validate consistency between cost types and their applicable project periods.

        This method verifies that cost types assigned to each record are compatible
        with the project period inferred from the reference year. Project periods
        (sunk, pre-onstream, post-onstream) impose constraints on which cost types
        are allowed. Any mismatch results in an exception.

        Parameters
        ----------
        ct : np.ndarray
            Array of cost types, represented as ``CostType`` enum members.
        ry : np.ndarray
            Array of reference years corresponding to each cost entry.
        onstream_year : int
            Project onstream year used to distinguish pre- and post-onstream periods.

        Raises
        ------
        BaseProjectException
            If a cost type is not allowed for the inferred project period.

        Notes
        -----
        - Period classification is based on comparison with the project approval
          year and the onstream year.
        - Validation stops at the first detected inconsistency.
        """

        # Group the classifications that violate the shared period rule, in the
        # engine's original period order (sunk, pre-onstream, post-onstream).
        found_by_period: dict[str, list[str]] = {}
        years = cast("list[int]", ry.tolist())
        classifications = cast("list[object]", ct.tolist())
        for index in range(len(classifications)):
            classification = classifications[index]
            if classification is None:
                continue
            period = classification_period(
                years[index],
                approval_year=self.approval_year,
                onstream_year=onstream_year,
            )
            if period is None:
                continue
            value = (
                classification.value
                if isinstance(classification, CostType)
                else str(classification)
            )
            if not classification_allowed_in_period(period, value):
                found_by_period.setdefault(period, []).append(value)

        # Apply validation rules sequentially; stop at the first period with a
        # violation, matching the original behavior.
        for period in CLASSIFICATION_ORDER:
            found = found_by_period.get(period)
            if found:
                msg = (
                    "Cost type classification mismatch. "
                    + f"Found {found} in {period} period."
                )
                raise BaseProjectInputError(
                    msg, code="COST_TYPE_PERIOD_MISMATCH"
                )

    def _validate_boundary_year_cost_types(
        self,
        ct: npt.NDArray[np.object_],
        ry: npt.NDArray[np.int64],
        onstream_year: int,
    ) -> None:
        """
        Validate cost type assignments at boundary project years.

        This method enforces consistency rules for cost types occurring exactly
        at the project approval year and the onstream year. Certain cost types
        are not permitted at these boundary years, and any violation results
        in an exception.

        Parameters
        ----------
        ct : np.ndarray
            Array of cost types represented as ``CostType`` enum members.
        ry : np.ndarray
            Array of reference years corresponding to each cost entry.
        onstream_year : int
            Project onstream year used as the boundary between pre- and
            post-onstream periods.

        Raises
        ------
        BaseProjectException
            If a disallowed cost type is detected at a boundary year.

        Notes
        -----
        - At the approval year, ``POST_ONSTREAM_COST`` is not permitted.
        - At the onstream year, ``SUNK_COST`` is not permitted.
        - This method performs validation only and does not modify cost types.
        """

        # Delegate the boundary-year rule to the shared predicate. An explicit
        # classification is what is checked; defaults are already assigned. The
        # approval-year rule is evaluated before the onstream-year rule, matching
        # the original order so the reported message is unchanged.
        years = cast("list[int]", ry.tolist())
        classifications = cast("list[object]", ct.tolist())

        def _values() -> list[str]:
            result: list[str] = []
            for classification in classifications:
                result.append(
                    classification.value
                    if isinstance(classification, CostType)
                    else str(classification)
                )
            return result

        boundaries = [
            boundary_year_classification_violation(
                years[index],
                value,
                approval_year=self.approval_year,
                onstream_year=onstream_year,
            )
            for index, value in enumerate(_values())
        ]
        if "approval_year" in boundaries:
            raise BaseProjectInputError(
                "Cannot accept POST ONSTREAM COST as cost type at approval year "
                + f"({self.approval_year}).",
                code="COST_TYPE_POSTONSTREAM_AT_APPROVAL",
            )
        if "onstream_year" in boundaries:
            raise BaseProjectInputError(
                f"Cannot accept SUNK COST as cost type at onstream year ({onstream_year}).",
                code="COST_TYPE_SUNK_AT_ONSTREAM",
            )

    def _prepare_cost_types(
        self,
        cost_obj: CapitalCost | Intangible | OPEX | ASR | LBT | CostOfSales,
    ) -> None:
        """
        Prepare, assign, and validate cost type classifications for a cost object.

        This method determines the effective project onstream year, assigns default
        cost types where missing, and validates that all cost type classifications
        are consistent with project timing rules. Additional validation is applied
        at boundary years (approval and onstream) to enforce strict classification
        constraints.

        Parameters
        ----------
        cost_obj : CapitalCost | Intangible | OPEX | ASR | LBT | CostOfSales
            Cost object whose ``cost_type`` attribute will be prepared and validated.

        Raises
        ------
        BaseProjectException
            If an invalid cost type classification is detected.

        Notes
        -----
        - The effective onstream year is defined as the earliest of oil and gas
          onstream dates.
        - Default cost types are assigned only where the input value is ``None``.
        - Boundary-year validation is applied only when the approval year precedes
          the onstream year.
        - This method updates ``cost_obj.cost_type`` in place.
        """

        # Validate approval year
        self._validate_approval_year()

        # Specify relevant onstream year
        oil_onstream_date, gas_onstream_date = self._require_onstream_dates()
        onstream_year = min([oil_onstream_date.year, gas_onstream_date.year])

        # Assign default values for "None" input
        ry, ct = self._assign_default_cost_types(
            onstream_year=onstream_year, cost_obj=cost_obj
        )

        # Validate cost type
        self._validate_cost_type_consistency(
            ct=ct,
            ry=ry,
            onstream_year=onstream_year,
        )

        # Validate cost type at boundary years
        if self.approval_year < onstream_year:
            self._validate_boundary_year_cost_types(
                ct=ct,
                ry=ry,
                onstream_year=onstream_year,
            )

        # Modify cost_type attribute of cost_obj
        cost_obj.cost_type = ct.tolist()

    def _filter_capital_cost(
        self, cost_obj_fluid: CapitalCost, include_cost_type: CostType
    ) -> CapitalCost:
        """
        Filter a CapitalCost object (pre-filtered by fluid type) by a specific
        cost type.

        This method extracts only the entries from ``cost_obj_fluid`` whose
        ``cost_type`` matches the specified ``include_cost_type``. If no
        matching entries exist, it returns a placeholder ``CapitalCost``
        object with zero cost at a default reference year.

        Unlike the generic version, ``cost_obj_fluid`` is expected to
        already represent a single fluid type (``FluidType.OIL`` or
        ``FluidType.GAS``). Its ``cost_allocation`` attribute therefore
        contains only one fluid type.

        Parameters
        ----------
        cost_obj_fluid : CapitalCost
            The input CapitalCost object that has already been filtered to
            represent a single fluid type. It contains attributes such as
            ``expense_year``, ``cost``, ``cost_allocation``, and others.

        include_cost_type : CostType
            The cost type to filter for. Supported values:
            ``CostType.SUNK_COST``, ``CostType.PRE_ONSTREAM_COST``,
            or ``CostType.POST_ONSTREAM_COST``.

        Returns
        -------
        CapitalCost
            A new CapitalCost object containing only the entries that
            match ``include_cost_type``. If no entries match, a placeholder
            object with zero cost and a default reference year is returned.

        Notes
        -----
        - Default reference year for placeholder objects:
            * ``CostType.SUNK_COST`` → project start year.
            * ``CostType.PRE_ONSTREAM_COST`` → project approval year.
            * ``CostType.POST_ONSTREAM_COST`` → earliest onstream year
              between oil and gas.
        - Because ``cost_obj_fluid`` already contains a single fluid type
          in its ``cost_allocation``, that allocation is preserved in both
          filtered and placeholder outputs.
        - All relevant attributes (e.g., ``description``, ``tax_portion``,
          ``useful_life``, ``pis_year``, etc.) are filtered consistently
          using the same boolean mask.
        - Returned attributes are converted to lists where applicable.
        """

        capital_data = _normalize_capital_cost_data(cost_obj_fluid)
        cost_data = capital_data.cost

        oil_onstream_date, gas_onstream_date = self._require_onstream_dates()
        onstream_yr = min([oil_onstream_date.year, gas_onstream_date.year])

        allocation = (
            FluidType.OIL
            if FluidType.OIL in cost_data.cost_allocation
            else FluidType.GAS
        )

        ey = {
            CostType.SUNK_COST: cost_obj_fluid.start_year,
            CostType.PRE_ONSTREAM_COST: self.approval_year,
            CostType.POST_ONSTREAM_COST: onstream_yr,
        }[include_cost_type]

        # Conduct masking process
        mask = _equals_mask(cost_data.cost_type, include_cost_type)

        if not np.any(mask):
            return CapitalCost(
                start_year=cost_obj_fluid.start_year,
                end_year=cost_obj_fluid.end_year,
                expense_year=np.array([ey]),
                cost=np.array([0]),
                cost_allocation=[allocation],
                cost_type=[include_cost_type],
                pis_year=np.array([onstream_yr]),
            )

        return CapitalCost(
            start_year=cost_obj_fluid.start_year,
            end_year=cost_obj_fluid.end_year,
            expense_year=cost_data.expense_year[mask],
            cost=cost_data.cost[mask],
            cost_allocation=_select_list(cost_data.cost_allocation, mask),
            cost_type=_select_list(cost_data.cost_type, mask),
            description=_select_list(cost_data.description, mask),
            tax_portion=cost_data.tax_portion[mask],
            tax_discount=cost_data.tax_discount[mask],
            pis_year=capital_data.pis_year[mask],
            salvage_value=capital_data.salvage_value[mask],
            useful_life=capital_data.useful_life[mask],
            depreciation_factor=capital_data.depreciation_factor[mask],
            is_ic_applied=_select_list(capital_data.is_ic_applied, mask),
        )

    def _filter_intangible(
        self, cost_obj_fluid: Intangible, include_cost_type: CostType
    ) -> Intangible:
        """
        Filter an Intangible cost object (pre-filtered by fluid type)
        by a specific cost type.

        This method extracts only the entries from ``cost_obj_fluid`` whose
        ``cost_type`` matches the specified ``include_cost_type``. If no
        matching entries exist, it returns a placeholder ``Intangible``
        object with zero cost at a default reference year.

        Unlike the generic version, ``cost_obj_fluid`` is expected to
        already represent a single fluid type (``FluidType.OIL`` or
        ``FluidType.GAS``). Its ``cost_allocation`` attribute therefore
        contains only one fluid type.

        Parameters
        ----------
        cost_obj_fluid : Intangible
            The input Intangible object that has already been filtered to
            represent a single fluid type. It contains attributes such as
            ``expense_year``, ``cost``, ``cost_allocation``, and others.

        include_cost_type : CostType
            The cost type to filter for. Supported values:
            ``CostType.SUNK_COST``, ``CostType.PRE_ONSTREAM_COST``,
            or ``CostType.POST_ONSTREAM_COST``.

        Returns
        -------
        Intangible
            A new Intangible object containing only the entries that
            match ``include_cost_type``. If no entries match, a placeholder
            object with zero cost and a default reference year is returned.

        Notes
        -----
        - Default reference year for placeholder objects:
            * ``CostType.SUNK_COST`` → project start year.
            * ``CostType.PRE_ONSTREAM_COST`` → project approval year.
            * ``CostType.POST_ONSTREAM_COST`` → earliest onstream year
              between oil and gas.
        - Because ``cost_obj_fluid`` already contains a single fluid type
          in its ``cost_allocation``, that allocation is preserved in both
          filtered and placeholder outputs.
        - All relevant attributes (e.g., ``description``, ``tax_portion``,
          ``tax_discount``) are filtered consistently using the same
          boolean mask.
        - Returned attributes are converted to lists where applicable.
        """

        cost_data = _normalize_cost_data(cost_obj_fluid)

        oil_onstream_date, gas_onstream_date = self._require_onstream_dates()
        onstream_yr = min([oil_onstream_date.year, gas_onstream_date.year])

        allocation = (
            FluidType.OIL
            if FluidType.OIL in cost_data.cost_allocation
            else FluidType.GAS
        )

        ey = {
            CostType.SUNK_COST: cost_obj_fluid.start_year,
            CostType.PRE_ONSTREAM_COST: self.approval_year,
            CostType.POST_ONSTREAM_COST: onstream_yr,
        }[include_cost_type]

        # Conduct masking process
        mask = _equals_mask(cost_data.cost_type, include_cost_type)

        if not np.any(mask):
            return Intangible(
                start_year=cost_obj_fluid.start_year,
                end_year=cost_obj_fluid.end_year,
                expense_year=np.array([ey]),
                cost=np.array([0]),
                cost_allocation=[allocation],
                cost_type=[include_cost_type],
            )

        return Intangible(
            start_year=cost_obj_fluid.start_year,
            end_year=cost_obj_fluid.end_year,
            expense_year=cost_data.expense_year[mask],
            cost=cost_data.cost[mask],
            cost_allocation=_select_list(cost_data.cost_allocation, mask),
            cost_type=_select_list(cost_data.cost_type, mask),
            description=_select_list(cost_data.description, mask),
            tax_portion=cost_data.tax_portion[mask],
            tax_discount=cost_data.tax_discount[mask],
        )

    def _filter_opex(self, cost_obj_fluid: OPEX, include_cost_type: CostType) -> OPEX:
        """
        Filter an OPEX cost object (pre-filtered by fluid type) by a
        specific cost type.

        This method extracts only the entries from ``cost_obj_fluid`` whose
        ``cost_type`` matches the specified ``include_cost_type``. If no
        matching entries exist, it returns a placeholder ``OPEX`` object
        with zero fixed cost at a default reference year.

        Unlike the generic version, ``cost_obj_fluid`` is expected to
        already represent a single fluid type (``FluidType.OIL`` or
        ``FluidType.GAS``). Its ``cost_allocation`` attribute therefore
        contains only one fluid type.

        Parameters
        ----------
        cost_obj_fluid : OPEX
            The input OPEX object that has already been filtered to
            represent a single fluid type. It contains attributes such as
            ``expense_year``, ``fixed_cost``, ``prod_rate``,
            ``cost_per_volume``, and others.

        include_cost_type : CostType
            The cost type to filter for. Supported values:
            ``CostType.SUNK_COST``, ``CostType.PRE_ONSTREAM_COST``,
            or ``CostType.POST_ONSTREAM_COST``.

        Returns
        -------
        OPEX
            A new OPEX object containing only the entries that match
            ``include_cost_type``. If no entries match, a placeholder
            object with zero fixed cost and a default reference year
            is returned.

        Notes
        -----
        - Default reference year for placeholder objects:
            * ``CostType.SUNK_COST`` → project start year.
            * ``CostType.PRE_ONSTREAM_COST`` → project approval year.
            * ``CostType.POST_ONSTREAM_COST`` → earliest onstream year
              between oil and gas.
        - Because ``cost_obj_fluid`` already contains a single fluid type
          in its ``cost_allocation``, that allocation is preserved in both
          filtered and placeholder outputs.
        - All relevant attributes (e.g., ``description``, ``tax_portion``,
          ``tax_discount``, ``prod_rate``, ``cost_per_volume``) are
          filtered consistently using the same boolean mask.
        - Returned attributes are converted to lists where applicable.
        """

        opex_data = _normalize_opex_data(cost_obj_fluid)
        cost_data = opex_data.cost

        oil_onstream_date, gas_onstream_date = self._require_onstream_dates()
        onstream_yr = min([oil_onstream_date.year, gas_onstream_date.year])

        allocation = (
            FluidType.OIL
            if FluidType.OIL in cost_data.cost_allocation
            else FluidType.GAS
        )

        ey = {
            CostType.SUNK_COST: cost_obj_fluid.start_year,
            CostType.PRE_ONSTREAM_COST: self.approval_year,
            CostType.POST_ONSTREAM_COST: onstream_yr,
        }[include_cost_type]

        # Conduct masking process
        mask = _equals_mask(cost_data.cost_type, include_cost_type)

        if not np.any(mask):
            return OPEX(
                start_year=cost_obj_fluid.start_year,
                end_year=cost_obj_fluid.end_year,
                expense_year=np.array([ey]),
                fixed_cost=np.array([0]),
                cost_allocation=[allocation],
                cost_type=[include_cost_type],
            )

        return OPEX(
            start_year=cost_obj_fluid.start_year,
            end_year=cost_obj_fluid.end_year,
            expense_year=cost_data.expense_year[mask],
            cost_allocation=_select_list(cost_data.cost_allocation, mask),
            cost_type=_select_list(cost_data.cost_type, mask),
            description=_select_list(cost_data.description, mask),
            tax_portion=cost_data.tax_portion[mask],
            tax_discount=cost_data.tax_discount[mask],
            fixed_cost=opex_data.fixed_cost[mask],
            prod_rate=opex_data.prod_rate[mask],
            cost_per_volume=opex_data.cost_per_volume[mask],
        )

    def _filter_asr(self, cost_obj_fluid: ASR, include_cost_type: CostType) -> ASR:
        """
        Filter an ASR (Abandonment and Site Restoration) cost object by a
        specific cost type.

        This method extracts only the entries from `cost_obj_fluid` whose
        `cost_type` matches the specified `include_cost_type`. The input
        `cost_obj_fluid` must already be filtered by fluid type, meaning its
        `cost_allocation` contains only one fluid type (either
        ``FluidType.OIL`` or ``FluidType.GAS``).

        If no matching entries exist,
        it returns a placeholder `ASR` object with zero cost at a default
        reference year.

        Parameters
        ----------
        cost_obj_fluid : ASR
            The input ASR object (already filtered by fluid type) containing
            attributes such as `expense_year`, `cost`, `final_year`,
            and `future_rate`. If the input is not an ASR instance, a
            ``BaseProjectException`` is raised.

        include_cost_type : CostType
            The cost type to filter for (e.g.,
            ``CostType.SUNK_COST``, ``CostType.PRE_ONSTREAM_COST``,
            or ``CostType.POST_ONSTREAM_COST``).

        Returns
        -------
        ASR
            A new ASR object containing only the entries that match
            `include_cost_type`. If no entries match, a placeholder
            object with zero cost is returned.

        Notes
        -----
        -   Default reference year for placeholder objects:
              * ``CostType.SUNK_COST`` → project start year.
              * ``CostType.PRE_ONSTREAM_COST`` → project approval year.
              * ``CostType.POST_ONSTREAM_COST`` → the earliest onstream year
                between oil and gas (i.e., min(oil_onstream_year, gas_onstream_year)).
        -   A default `cost_allocation` is inferred: if ``FluidType.OIL`` exists
            in the original allocation, it is used; otherwise, ``FluidType.GAS``
            is assigned.
        -   All relevant attributes (e.g., `description`, `tax_portion`,
            `tax_discount`, `final_year`, `future_rate`) are filtered consistently
            using the same boolean mask.
        -   Returned attributes are converted to lists where applicable.
        """

        asr_data = _normalize_asr_data(cost_obj_fluid)
        cost_data = asr_data.cost

        oil_onstream_date, gas_onstream_date = self._require_onstream_dates()
        onstream_yr = min([oil_onstream_date.year, gas_onstream_date.year])

        allocation = (
            FluidType.OIL
            if FluidType.OIL in cost_data.cost_allocation
            else FluidType.GAS
        )

        ey = {
            CostType.SUNK_COST: cost_obj_fluid.start_year,
            CostType.PRE_ONSTREAM_COST: self.approval_year,
            CostType.POST_ONSTREAM_COST: onstream_yr,
        }[include_cost_type]

        # Conduct masking process
        mask = _equals_mask(cost_data.cost_type, include_cost_type)

        if not np.any(mask):
            return ASR(
                start_year=cost_obj_fluid.start_year,
                end_year=cost_obj_fluid.end_year,
                expense_year=np.array([ey]),
                cost=np.array([0]),
                cost_allocation=[allocation],
                cost_type=[include_cost_type],
            )

        return ASR(
            start_year=cost_obj_fluid.start_year,
            end_year=cost_obj_fluid.end_year,
            expense_year=cost_data.expense_year[mask],
            cost=cost_data.cost[mask],
            cost_allocation=_select_list(cost_data.cost_allocation, mask),
            cost_type=_select_list(cost_data.cost_type, mask),
            description=_select_list(cost_data.description, mask),
            tax_portion=cost_data.tax_portion[mask],
            tax_discount=cost_data.tax_discount[mask],
            final_year=asr_data.final_year[mask],
            future_rate=asr_data.future_rate[mask],
        )

    def _filter_lbt(self, cost_obj_fluid: LBT, include_cost_type: CostType) -> LBT:
        """
        Filter an LBT (Land and Building Tax) cost object by a specific
        cost type.

        This method extracts only the entries from `cost_obj_fluid` whose
        `cost_type` matches the specified `include_cost_type`. The input
        `cost_obj_fluid` must already be filtered by fluid type, meaning its
        `cost_allocation` contains only one fluid type (either
        ``FluidType.OIL`` or ``FluidType.GAS``).

        If no matching entries exist, it returns a placeholder `LBT` object
        with zero cost at a default reference year.

        Parameters
        ----------
        cost_obj_fluid : LBT
            The input LBT object (already filtered by fluid type) containing
            attributes such as `expense_year`, `cost`, `njop_land`,
            `njop_building`, `utilized_land_area`, and `gross_revenue`.
            If the input is not an LBT instance, a ``BaseProjectException``
            is raised.

        include_cost_type : CostType
            The cost type to filter for (e.g.,
            ``CostType.SUNK_COST``, ``CostType.PRE_ONSTREAM_COST``,
            or ``CostType.POST_ONSTREAM_COST``).

        Returns
        -------
        LBT
            A new LBT object containing only the entries that match
            `include_cost_type`. If no entries match, a placeholder
            object with zero cost is returned.

        Notes
        -----
        -   Default reference year for placeholder objects:
              * ``CostType.SUNK_COST`` → project start year.
              * ``CostType.PRE_ONSTREAM_COST`` → project approval year.
              * ``CostType.POST_ONSTREAM_COST`` → the earliest onstream year
                between oil and gas (i.e., min(oil_onstream_year, gas_onstream_year)).
        -   A default `cost_allocation` is inferred: if ``FluidType.OIL`` exists
            in the original allocation, it is used; otherwise, ``FluidType.GAS``
            is assigned.
        -   All relevant attributes (e.g., `description`, `tax_portion`,
            `tax_discount`, `final_year`, `utilized_land_area`,
            `utilized_building_area`, `njop_land`, `njop_building`,
            `gross_revenue`, and `cost`) are filtered consistently using the
            same boolean mask.
        -   Returned attributes are converted to lists where applicable.
        """

        lbt_data = _normalize_lbt_data(cost_obj_fluid)
        cost_data = lbt_data.cost

        oil_onstream_date, gas_onstream_date = self._require_onstream_dates()
        onstream_yr = min([oil_onstream_date.year, gas_onstream_date.year])

        allocation = (
            FluidType.OIL
            if FluidType.OIL in cost_data.cost_allocation
            else FluidType.GAS
        )

        ey = {
            CostType.SUNK_COST: cost_obj_fluid.start_year,
            CostType.PRE_ONSTREAM_COST: self.approval_year,
            CostType.POST_ONSTREAM_COST: onstream_yr,
        }[include_cost_type]

        # Conduct masking process
        mask = _equals_mask(cost_data.cost_type, include_cost_type)

        if not np.any(mask):
            return LBT(
                start_year=cost_obj_fluid.start_year,
                end_year=cost_obj_fluid.end_year,
                expense_year=np.array([ey]),
                cost=np.array([0]),
                cost_allocation=[allocation],
                cost_type=[include_cost_type],
            )

        return LBT(
            start_year=cost_obj_fluid.start_year,
            end_year=cost_obj_fluid.end_year,
            expense_year=cost_data.expense_year[mask],
            cost_allocation=_select_list(cost_data.cost_allocation, mask),
            cost_type=_select_list(cost_data.cost_type, mask),
            description=_select_list(cost_data.description, mask),
            tax_portion=cost_data.tax_portion[mask],
            tax_discount=cost_data.tax_discount[mask],
            final_year=lbt_data.final_year[mask],
            utilized_land_area=lbt_data.utilized_land_area[mask],
            utilized_building_area=lbt_data.utilized_building_area[mask],
            njop_land=lbt_data.njop_land[mask],
            njop_building=lbt_data.njop_building[mask],
            gross_revenue=lbt_data.gross_revenue[mask],
            cost=cost_data.cost[mask],
        )

    def _filter_cost_of_sales(
        self, cost_obj_fluid: CostOfSales, include_cost_type: CostType
    ) -> CostOfSales:
        """
        Filter a CostOfSales cost object by a specific cost type.

        This method extracts only the entries from `cost_obj_fluid` whose
        `cost_type` matches the specified `include_cost_type`. The input
        `cost_obj_fluid` must already be filtered by fluid type, meaning its
        `cost_allocation` contains only one fluid type (either
        ``FluidType.OIL`` or ``FluidType.GAS``).

        If no matching entries exist, it returns a placeholder `CostOfSales`
        object with zero cost at a default reference year.

        Parameters
        ----------
        cost_obj_fluid : CostOfSales
            The input CostOfSales object (already filtered by fluid type) containing
            attributes such as `expense_year`, `cost`, `description`,
            `tax_portion`, and `tax_discount`. If the input is not a
            CostOfSales instance, a ``BaseProjectException`` is raised.

        include_cost_type : CostType
            The cost type to filter for (e.g.,
            ``CostType.SUNK_COST``, ``CostType.PRE_ONSTREAM_COST``,
            or ``CostType.POST_ONSTREAM_COST``).

        Returns
        -------
        CostOfSales
            A new CostOfSales object containing only the entries that match
            `include_cost_type`. If no entries match, a placeholder
            object with zero cost is returned.

        Notes
        -----
        -   Default reference year for placeholder objects:
              * ``CostType.SUNK_COST`` → project start year.
              * ``CostType.PRE_ONSTREAM_COST`` → project approval year.
              * ``CostType.POST_ONSTREAM_COST`` → the earliest onstream year
                between oil and gas (i.e., min(oil_onstream_year, gas_onstream_year)).
        -   A default `cost_allocation` is inferred: if ``FluidType.OIL`` exists
            in the original allocation, it is used; otherwise, ``FluidType.GAS``
            is assigned.
        -   All relevant attributes (e.g., `description`, `tax_portion`,
            `tax_discount`, `cost`) are filtered consistently using the
            same boolean mask.
        -   Returned attributes are converted to lists where applicable.
        """

        cost_data = _normalize_cost_data(cost_obj_fluid)

        oil_onstream_date, gas_onstream_date = self._require_onstream_dates()
        onstream_yr = min([oil_onstream_date.year, gas_onstream_date.year])

        allocation = (
            FluidType.OIL
            if FluidType.OIL in cost_data.cost_allocation
            else FluidType.GAS
        )

        ey = {
            CostType.SUNK_COST: cost_obj_fluid.start_year,
            CostType.PRE_ONSTREAM_COST: self.approval_year,
            CostType.POST_ONSTREAM_COST: onstream_yr,
        }[include_cost_type]

        # Conduct masking process
        mask = _equals_mask(cost_data.cost_type, include_cost_type)

        if not np.any(mask):
            return CostOfSales(
                start_year=cost_obj_fluid.start_year,
                end_year=cost_obj_fluid.end_year,
                expense_year=np.array([ey]),
                cost=np.array([0]),
                cost_allocation=[allocation],
                cost_type=[include_cost_type],
            )

        return CostOfSales(
            start_year=cost_obj_fluid.start_year,
            end_year=cost_obj_fluid.end_year,
            expense_year=cost_data.expense_year[mask],
            cost=cost_data.cost[mask],
            cost_allocation=_select_list(cost_data.cost_allocation, mask),
            cost_type=_select_list(cost_data.cost_type, mask),
            description=_select_list(cost_data.description, mask),
            tax_portion=cost_data.tax_portion[mask],
            tax_discount=cost_data.tax_discount[mask],
        )

    def _check_inconsistent_start_year(self) -> None:
        """
        Validate consistency of start years across all project components.

        This method ensures that the start year of every associated project
        component (lifting, capital costs, intangibles, OPEX, ASR, LBT, and
        cost of sales for oil, gas, sulfur, electricity, and CO₂) matches the
        base project's `start_date.year`.

        If any component's `start_year` is inconsistent with the base project,
        a `BaseProjectException` is raised with a detailed error message
        identifying the mismatched component(s).

        Raises
        ------
        BaseProjectException
            If any component's start year does not match the base project
            `start_date.year`. The exception message lists the base project
            year alongside all component start years for debugging.

        Notes
        -----
        - Components checked include:
          * Hydrocarbon lifting: oil, gas, sulfur, electricity, CO₂.
          * Capital costs: sunk cost, pre-onstream, and post-onstream
            (oil and gas).
          * Intangible costs: sunk cost, pre-onstream, and post-onstream
            (oil and gas).
          * OPEX: sunk cost, pre-onstream, and post-onstream (oil and gas).
          * ASR: sunk cost, pre-onstream, and post-onstream (oil and gas).
          * LBT: sunk cost, pre-onstream, and post-onstream (oil and gas).
          * Cost of sales: sunk cost, pre-onstream, and post-onstream
            (oil and gas).
        """

        component_years = [
                self._oil_lifting.start_year,
                self._gas_lifting.start_year,
                self._sulfur_lifting.start_year,
                self._electricity_lifting.start_year,
                self._co2_lifting.start_year,
                self._oil_capital_postonstream.start_year,
                self._gas_capital_postonstream.start_year,
                self._oil_intangible_postonstream.start_year,
                self._gas_intangible_postonstream.start_year,
                self._oil_opex_postonstream.start_year,
                self._gas_opex_postonstream.start_year,
                self._oil_asr_postonstream.start_year,
                self._gas_asr_postonstream.start_year,
                self._oil_lbt_postonstream.start_year,
                self._gas_lbt_postonstream.start_year,
                self._oil_cost_of_sales_postonstream.start_year,
                self._gas_cost_of_sales_postonstream.start_year,
                self._oil_capital_sunk_cost.start_year,
                self._gas_capital_sunk_cost.start_year,
                self._oil_intangible_sunk_cost.start_year,
                self._gas_intangible_sunk_cost.start_year,
                self._oil_opex_sunk_cost.start_year,
                self._gas_opex_sunk_cost.start_year,
                self._oil_asr_sunk_cost.start_year,
                self._gas_asr_sunk_cost.start_year,
                self._oil_lbt_sunk_cost.start_year,
                self._gas_lbt_sunk_cost.start_year,
                self._oil_cost_of_sales_sunk_cost.start_year,
                self._gas_cost_of_sales_sunk_cost.start_year,
                self._oil_capital_preonstream.start_year,
                self._gas_capital_preonstream.start_year,
                self._oil_intangible_preonstream.start_year,
                self._gas_intangible_preonstream.start_year,
                self._oil_opex_preonstream.start_year,
                self._gas_opex_preonstream.start_year,
                self._oil_asr_preonstream.start_year,
                self._gas_asr_preonstream.start_year,
                self._oil_lbt_preonstream.start_year,
                self._gas_lbt_preonstream.start_year,
                self._oil_cost_of_sales_preonstream.start_year,
                self._gas_cost_of_sales_preonstream.start_year,
            ]
        if inconsistent_component_years(self.start_date.year, component_years):
            raise BaseProjectInputError(
                "Inconsistent start project year: "
                + f"Base project ({self.start_date.year}), "
                + f"Oil lifting ({self._oil_lifting.start_year}), "
                + f"Gas lifting ({self._gas_lifting.start_year}), "
                + f"Sulfur lifting ({self._sulfur_lifting.start_year}), "
                + f"Electricity lifting ({self._electricity_lifting.start_year}), "
                + f"CO2 lifting ({self._co2_lifting.start_year}), "
                + f"Oil capital postonstream ({self._oil_capital_postonstream.start_year}), "
                + f"Gas capital postonstream ({self._gas_capital_postonstream.start_year}), "
                + "Oil intangible postonstream "
                + f"({self._oil_intangible_postonstream.start_year}), "
                + "Gas intangible postonstream "
                + f"({self._gas_intangible_postonstream.start_year}), "
                + f"Oil opex postonstream ({self._oil_opex_postonstream.start_year}), "
                + f"Gas opex postonstream ({self._gas_opex_postonstream.start_year}), "
                + f"Oil asr postonstream ({self._oil_asr_postonstream.start_year}), "
                + f"Gas asr postonstream ({self._gas_asr_postonstream.start_year}), "
                + f"Oil lbt postonstream ({self._oil_lbt_postonstream.start_year}), "
                + f"Gas lbt postonstream ({self._gas_lbt_postonstream.start_year}), "
                + "Oil cost of sales postonstream "
                + f"({self._oil_cost_of_sales_postonstream.start_year}), "
                + "Gas cost of sales postonstream "
                + f"({self._gas_cost_of_sales_postonstream.start_year}), "
                + f"Oil capital sunkcost ({self._oil_capital_sunk_cost.start_year}), "
                + f"Gas capital sunkcost ({self._gas_capital_sunk_cost.start_year}), "
                + f"Oil intangible sunkcost ({self._oil_intangible_sunk_cost.start_year}), "
                + f"Gas intangible sunkcost ({self._gas_intangible_sunk_cost.start_year}), "
                + f"Oil opex sunkcost ({self._oil_opex_sunk_cost.start_year}), "
                + f"Gas opex sunkcost ({self._gas_opex_sunk_cost.start_year}), "
                + f"Oil asr sunkcost ({self._oil_asr_sunk_cost.start_year}), "
                + f"Gas asr sunkcost ({self._gas_asr_sunk_cost.start_year}), "
                + f"Oil lbt sunkcost ({self._oil_lbt_sunk_cost.start_year}), "
                + f"Gas lbt sunkcost ({self._gas_lbt_sunk_cost.start_year}), "
                + "Oil cost of sales sunkcost "
                + f"({self._oil_cost_of_sales_sunk_cost.start_year}), "
                + "Gas cost of sales sunkcost "
                + f"({self._gas_cost_of_sales_sunk_cost.start_year}), "
                + f"Oil capital preonstream ({self._oil_capital_preonstream.start_year}), "
                + f"Gas capital preonstream ({self._gas_capital_preonstream.start_year}), "
                + f"Oil intangible preonstream ({self._oil_intangible_preonstream.start_year}), "
                + f"Gas intangible preonstream ({self._gas_intangible_preonstream.start_year}), "
                + f"Oil opex preonstream ({self._oil_opex_preonstream.start_year}), "
                + f"Gas opex preonstream ({self._gas_opex_preonstream.start_year}), "
                + f"Oil asr preonstream ({self._oil_asr_preonstream.start_year}), "
                + f"Gas asr preonstream ({self._gas_asr_preonstream.start_year}), "
                + f"Oil lbt preonstream ({self._oil_lbt_preonstream.start_year}), "
                + f"Gas lbt preonstream ({self._gas_lbt_preonstream.start_year}), "
                + "Oil cost of sales preonstream "
                + f"({self._oil_cost_of_sales_preonstream.start_year}), "
                + "Gas cost of sales preonstream "
                + f"({self._gas_cost_of_sales_preonstream.start_year})."
            )

    def _check_inconsistent_end_year(self) -> None:
        """
        Validate consistency of end years across all project components.

        This method ensures that the `end_year` of every associated project
        component (lifting, capital costs, intangibles, OPEX, ASR, LBT, and
        cost of sales for oil, gas, sulfur, electricity, and CO₂) matches the
        base project's `end_date.year`.

        If any component's `end_year` is inconsistent with the base project,
        a `BaseProjectException` is raised with a detailed error message
        identifying the mismatched component(s).

        Raises
        ------
        BaseProjectException
            If any component's end year does not match the base project
            `end_date.year`. The exception message lists the base project
            year alongside all component end years for debugging.

        Notes
        -----
        - Components checked include:
          * Hydrocarbon lifting: oil, gas, sulfur, electricity, CO₂.
          * Capital costs: sunk cost, pre-onstream, and post-onstream
            (oil and gas).
          * Intangible costs: sunk cost, pre-onstream, and post-onstream
            (oil and gas).
          * OPEX: sunk cost, pre-onstream, and post-onstream (oil and gas).
          * ASR: sunk cost, pre-onstream, and post-onstream (oil and gas).
          * LBT: sunk cost, pre-onstream, and post-onstream (oil and gas).
          * Cost of sales: sunk cost, pre-onstream, and post-onstream
            (oil and gas).
        """

        component_years = [
                self._oil_lifting.end_year,
                self._gas_lifting.end_year,
                self._sulfur_lifting.end_year,
                self._electricity_lifting.end_year,
                self._co2_lifting.end_year,
                self._oil_capital_postonstream.end_year,
                self._gas_capital_postonstream.end_year,
                self._oil_intangible_postonstream.end_year,
                self._gas_intangible_postonstream.end_year,
                self._oil_opex_postonstream.end_year,
                self._gas_opex_postonstream.end_year,
                self._oil_asr_postonstream.end_year,
                self._gas_asr_postonstream.end_year,
                self._oil_lbt_postonstream.end_year,
                self._gas_lbt_postonstream.end_year,
                self._oil_cost_of_sales_postonstream.end_year,
                self._gas_cost_of_sales_postonstream.end_year,
                self._oil_capital_sunk_cost.end_year,
                self._gas_capital_sunk_cost.end_year,
                self._oil_intangible_sunk_cost.end_year,
                self._gas_intangible_sunk_cost.end_year,
                self._oil_opex_sunk_cost.end_year,
                self._gas_opex_sunk_cost.end_year,
                self._oil_asr_sunk_cost.end_year,
                self._gas_asr_sunk_cost.end_year,
                self._oil_lbt_sunk_cost.end_year,
                self._gas_lbt_sunk_cost.end_year,
                self._oil_cost_of_sales_sunk_cost.end_year,
                self._gas_cost_of_sales_sunk_cost.end_year,
                self._oil_capital_preonstream.end_year,
                self._gas_capital_preonstream.end_year,
                self._oil_intangible_preonstream.end_year,
                self._gas_intangible_preonstream.end_year,
                self._oil_opex_preonstream.end_year,
                self._gas_opex_preonstream.end_year,
                self._oil_asr_preonstream.end_year,
                self._gas_asr_preonstream.end_year,
                self._oil_lbt_preonstream.end_year,
                self._gas_lbt_preonstream.end_year,
                self._oil_cost_of_sales_preonstream.end_year,
                self._gas_cost_of_sales_preonstream.end_year,
            ]
        if inconsistent_component_years(self.end_date.year, component_years):
            raise BaseProjectInputError(
                "Inconsistent end project year: "
                + f"Base project ({self.end_date.year}), "
                + f"Oil lifting ({self._oil_lifting.end_year}), "
                + f"Gas lifting ({self._gas_lifting.end_year}), "
                + f"Sulfur lifting ({self._sulfur_lifting.end_year}), "
                + f"Electricity lifting ({self._electricity_lifting.end_year}), "
                + f"CO2 lifting ({self._co2_lifting.end_year}), "
                + f"Oil capital postonstream ({self._oil_capital_postonstream.end_year}), "
                + f"Gas capital postonstream ({self._gas_capital_postonstream.end_year}), "
                + "Oil intangible postonstream "
                + f"({self._oil_intangible_postonstream.end_year}), "
                + "Gas intangible postonstream "
                + f"({self._gas_intangible_postonstream.end_year}), "
                + f"Oil opex postonstream ({self._oil_opex_postonstream.end_year}), "
                + f"Gas opex postonstream ({self._gas_opex_postonstream.end_year}), "
                + f"Oil asr postonstream ({self._oil_asr_postonstream.end_year}), "
                + f"Gas asr postonstream ({self._gas_asr_postonstream.end_year}), "
                + f"Oil lbt postonstream ({self._oil_lbt_postonstream.end_year}), "
                + f"Gas lbt postonstream ({self._gas_lbt_postonstream.end_year}), "
                + "Oil cost of sales postonstream "
                + f"({self._oil_cost_of_sales_postonstream.end_year}), "
                + "Gas cost of sales postonstream "
                + f"({self._gas_cost_of_sales_postonstream.end_year}), "
                + f"Oil capital sunkcost ({self._oil_capital_sunk_cost.end_year}), "
                + f"Gas capital sunkcost ({self._gas_capital_sunk_cost.end_year}), "
                + f"Oil intangible sunkcost ({self._oil_intangible_sunk_cost.end_year}), "
                + f"Gas intangible sunkcost ({self._gas_intangible_sunk_cost.end_year}), "
                + f"Oil opex sunkcost ({self._oil_opex_sunk_cost.end_year}), "
                + f"Gas opex sunkcost ({self._gas_opex_sunk_cost.end_year}), "
                + f"Oil asr sunkcost ({self._oil_asr_sunk_cost.end_year}), "
                + f"Gas asr sunkcost ({self._gas_asr_sunk_cost.end_year}), "
                + f"Oil lbt sunkcost ({self._oil_lbt_sunk_cost.end_year}), "
                + f"Gas lbt sunkcost ({self._gas_lbt_sunk_cost.end_year}), "
                + "Oil cost of sales sunkcost "
                + f"({self._oil_cost_of_sales_sunk_cost.end_year}), "
                + "Gas cost of sales sunkcost "
                + f"({self._gas_cost_of_sales_sunk_cost.end_year}), "
                + f"Oil capital preonstream ({self._oil_capital_preonstream.end_year}), "
                + f"Gas capital preonstream ({self._gas_capital_preonstream.end_year}), "
                + f"Oil intangible preonstream ({self._oil_intangible_preonstream.end_year}), "
                + f"Gas intangible preonstream ({self._gas_intangible_preonstream.end_year}), "
                + f"Oil opex preonstream ({self._oil_opex_preonstream.end_year}), "
                + f"Gas opex preonstream ({self._gas_opex_preonstream.end_year}), "
                + f"Oil asr preonstream ({self._oil_asr_preonstream.end_year}), "
                + f"Gas asr preonstream ({self._gas_asr_preonstream.end_year}), "
                + f"Oil lbt preonstream ({self._oil_lbt_preonstream.end_year}), "
                + f"Gas lbt preonstream ({self._gas_lbt_preonstream.end_year}), "
                + "Oil cost of sales preonstream "
                + f"({self._oil_cost_of_sales_preonstream.end_year}), "
                + "Gas cost of sales preonstream "
                + f"({self._gas_cost_of_sales_preonstream.end_year})."
            )

    def _validate_sunkcost(
        self,
        sunkcost_objects: list[
            CapitalCost | Intangible | OPEX | ASR | LBT | CostOfSales
        ],
    ) -> None:
        """
        Validate that sunk cost objects do not extend beyond the cutoff year.

        This method checks the maximum ``expense_year`` of each sunk cost object and
        verifies that none exceed the cutoff year, defined as the earlier of the
        project approval year and the earliest onstream year (oil or gas).

        Behavior depends on ``self.is_strict``:
        - ``True``  → raise ``SunkCostException`` on violation.
        - ``False`` → record a warning message.

        Parameters
        ----------
        sunkcost_objects : list
            List of sunk cost objects, each exposing an ``expense_year`` array. The
            list is assumed to contain only sunk cost items.

        Raises
        ------
        SunkCostException
            Raised in strict mode if any sunk cost year exceeds the cutoff year.
        """

        # Determine onstream year: the earlier between oil and gas onstream year
        oil_onstream_date, gas_onstream_date = self._require_onstream_dates()
        onstream_yr = min([oil_onstream_date.year, gas_onstream_date.year])

        # Extract maximum expense year for each sunkcost objects
        sunkcost_max_years = np.array(
            [
                int(np.max(_require_int_array(sc.expense_year, "expense_year")))
                for sc in sunkcost_objects
            ],
            dtype=np.int64,
        )
        cutoff_year = sunk_cost_cutoff_year(
            approval_year=self.approval_year, onstream_year=onstream_yr
        )
        max_years = cast("list[int]", sunkcost_max_years.tolist())
        mask = np.array(
            [
                sunk_cost_year_exceeds_cutoff(year, cutoff_year)
                for year in max_years
            ],
            dtype=np.bool_,
        )

        if np.any(mask):
            # Collect offending years for reporting
            incorrect_years = sunkcost_max_years[mask]
            message = f"Sunk cost years ({incorrect_years}) exceed cutoff year ({cutoff_year})."

            # Strict mode: raise an error
            if self.is_strict:
                raise SunkCostInputError(message, code="SUNK_COST_AFTER_CUTOFF")

            # Loose mode: display a warning message
            else:
                self.warning_messages.append((ContractType.BASE_PROJECT.value, message))

    def _validate_preonstream(
        self,
        preonstream_objects: list[
            CapitalCost | Intangible | OPEX | ASR | LBT | CostOfSales
        ],
    ) -> None:
        """
        Validate year bounds of pre-onstream cost objects.

        This method checks each object's ``expense_year`` against the allowable
        pre-onstream window using the earliest onstream year (oil or gas) as the
        upper bound. Validation rules depend on the contract type:

        - **POD I**     → ``approval_year <= min(expense_year)``
                          and ``max(expense_year) <= onstream_year``
        - **Non-POD I** → ``max(expense_year) <= onstream_year``

        Behavior is controlled by ``self.is_strict``:
        - ``True``  → raise ``PreOnstreamException`` on violation.
        - ``False`` → record a warning message.

        Parameters
        ----------
        preonstream_objects : list
            List of pre-onstream cost objects exposing an ``expense_year`` array.

        Raises
        ------
        PreOnstreamException
            Raised in strict mode if one or more objects violate the year constraints.
        """

        # Determine onstream year: the earlier between oil and gas onstream year
        oil_onstream_date, gas_onstream_date = self._require_onstream_dates()
        onstream_yr = min([oil_onstream_date.year, gas_onstream_date.year])

        preonstream_max_years = np.array(
            [
                int(np.max(_require_int_array(pr.expense_year, "expense_year")))
                for pr in preonstream_objects
            ],
            dtype=np.int64,
        )
        preonstream_min_years = np.array(
            [
                int(np.min(_require_int_array(pr.expense_year, "expense_year")))
                for pr in preonstream_objects
            ],
            dtype=np.int64,
        )
        message = None

        # POD I contract: requires both lower and upper bound checks
        if self.is_pod_1:
            # Identify lower- and upper-bound violations
            min_years = preonstream_min_years < self.approval_year
            max_years = preonstream_max_years > onstream_yr
            mask = min_years | max_years

            has_min_violation = np.any(min_years)
            has_max_violation = np.any(max_years)

            incorrect_years = None

            if np.any(mask):
                # Reporting offending years based on violation type
                if has_min_violation:
                    incorrect_years = preonstream_min_years[mask]

                if has_max_violation:
                    incorrect_years = preonstream_max_years[mask]

                if has_min_violation and has_max_violation:
                    incorrect_years = list(
                        zip(preonstream_min_years[mask], preonstream_max_years[mask])
                    )

                message = (
                    f"Preonstream years ({incorrect_years}) fall outside the allowable "
                    + f"range: {self.approval_year} <= preonstream_years <= {onstream_yr}."
                )

        # Non-POD I contract: only check the upper bound
        else:
            mask = preonstream_max_years > onstream_yr
            if np.any(mask):
                incorrect_years = preonstream_max_years[mask]
                message = (
                    "Preonstream years fall outside the allowable range: "
                    + f"preonstream_years ({incorrect_years} > {onstream_yr})."
                )

        # Handle violation according to strictness policy
        if message is not None:
            # Strict mode: raise an error
            if self.is_strict:
                raise PreOnstreamInputError(
                    message, code="PREONSTREAM_YEAR_OUTSIDE_PROJECT"
                )

            # Loose mode: emit a warning and continue
            else:
                self.warning_messages.append((ContractType.BASE_PROJECT.value, message))

    def _validate_postonstream(
        self,
        postonstream_objects: list[
            CapitalCost | Intangible | OPEX | ASR | LBT | CostOfSales
        ],
    ) -> None:
        """
        Validate year bounds of post-onstream cost objects.

        This method checks the minimum ``expense_year`` of each post-onstream cost
        object and ensures that all values are greater than or equal to the project's
        earliest onstream year (oil or gas).

        Parameters
        ----------
        postonstream_objects : list
            List of post-onstream cost objects exposing an ``expense_year`` array.

        Returns
        -------
        None

        Raises
        ------
        PostOnstreamException
            Raised if one or more post-onstream cost objects contain years earlier than
            the onstream year.
        """

        # Determine onstream year: the earlier between oil and gas onstream year
        oil_onstream_date, gas_onstream_date = self._require_onstream_dates()
        onstream_yr = min([oil_onstream_date.year, gas_onstream_date.year])

        # Extract minimum expense year for each postonstream objects
        postonstream_min_years = np.array(
            [
                int(np.min(_require_int_array(po.expense_year, "expense_year")))
                for po in postonstream_objects
            ],
            dtype=np.int64,
        )
        mask = postonstream_min_years < onstream_yr

        if np.any(mask):
            # Collect offending years for reporting
            incorrect_years = postonstream_min_years[mask]
            message = (
                f"Postonstream cost years ({incorrect_years}) occur before "
                + f"onstream year ({onstream_yr})"
            )

            raise PostOnstreamInputError(
                message, code="POSTONSTREAM_YEAR_BEFORE_ONSTREAM"
            )

    def _get_cost_objects_validation(self) -> None:
        """
        Validate all cost objects (sunk, preonstream, and postonstream) across
        categories and fluids.

        This method collects cost objects from all categories (CapitalCost,
        Intangible, OPEX, ASR, LBT, and CostOfSales) for both oil and gas,
        grouped by the timing of the cost: sunk, preonstream, and postonstream.

        Each group of objects is then validated using the corresponding validator
        method to ensure their `expense_year` values are consistent with project
        milestones (approval year and onstream dates).

        Validation rules:
        -   Sunk costs must not exceed the minimum of the approval year and earliest
            onstream year.
        -   Preonstream costs must be between the approval year and the earliest
            onstream year.
        -   Postonstream costs must occur no earlier than the earliest onstream year.

        Notes
        -----
        -   Each input cost object is assumed to be already filtered by fluid type.
        -   The method automates validation across multiple cost categories and
            timing groups, ensuring consistency of project cost modeling.
        """

        # Collect a list of sunkcost objects
        sunkcost_objects = [
            self._oil_capital_sunk_cost,
            self._oil_intangible_sunk_cost,
            self._oil_opex_sunk_cost,
            self._oil_asr_sunk_cost,
            self._oil_lbt_sunk_cost,
            self._oil_cost_of_sales_sunk_cost,
            self._gas_capital_sunk_cost,
            self._gas_intangible_sunk_cost,
            self._gas_opex_sunk_cost,
            self._gas_asr_sunk_cost,
            self._gas_lbt_sunk_cost,
            self._gas_cost_of_sales_sunk_cost,
        ]

        # Collect a list of preonstream objects
        preonstream_objects = [
            self._oil_capital_preonstream,
            self._oil_intangible_preonstream,
            self._oil_opex_preonstream,
            self._oil_asr_preonstream,
            self._oil_lbt_preonstream,
            self._oil_cost_of_sales_preonstream,
            self._gas_capital_preonstream,
            self._gas_intangible_preonstream,
            self._gas_opex_preonstream,
            self._gas_asr_preonstream,
            self._gas_lbt_preonstream,
            self._gas_cost_of_sales_preonstream,
        ]

        # Collect a list of postonstream objects
        postonstream_objects = [
            self._oil_capital_postonstream,
            self._oil_intangible_postonstream,
            self._oil_opex_postonstream,
            self._oil_asr_postonstream,
            self._oil_lbt_postonstream,
            self._oil_cost_of_sales_postonstream,
            self._gas_capital_postonstream,
            self._gas_intangible_postonstream,
            self._gas_opex_postonstream,
            self._gas_asr_postonstream,
            self._gas_lbt_postonstream,
            self._gas_cost_of_sales_postonstream,
        ]

        # Run validator for each cost objects
        validation_mapping = [
            (sunkcost_objects, self._validate_sunkcost),
            (preonstream_objects, self._validate_preonstream),
            (postonstream_objects, self._validate_postonstream),
        ]

        for objects, validator in validation_mapping:
            validator(objects)

    def _get_sunkcost_array(self) -> None:
        """
        Construct and organize sunk cost arrays for OIL and GAS.

        This method gathers pre-tax expenditures for each sunk cost category
        (Capital, Intangible, OPEX, ASR, LBT, and Cost of Sales), then groups
        them into depreciable and non-depreciable components. Finally, total
        sunk costs are computed as the sum of these two components.

        Notes
        -----
        - Depreciable sunk costs include only capital expenditures.
        - Non-depreciable sunk costs are formed by summing intangible, OPEX,
          ASR, LBT, and cost of sales components.
        - Total sunk costs are the sum of depreciable and non-depreciable costs.
        - The following internal attributes are set:
            * ``_oil_depreciable_sunk_cost``
            * ``_gas_depreciable_sunk_cost``
            * ``_oil_non_depreciable_sunk_cost``
            * ``_gas_non_depreciable_sunk_cost``
            * ``_oil_sunk_cost``
            * ``_gas_sunk_cost``
        - Internally relies on each cost object's
          ``expenditures_pre_tax()`` method to compute expenditures.
        """

        # Prepare sunkcost arrays for each sunkcost objects
        sunkcost_map = {
            "oil_capital": self._oil_capital_sunk_cost,
            "gas_capital": self._gas_capital_sunk_cost,
            "oil_intangible": self._oil_intangible_sunk_cost,
            "gas_intangible": self._gas_intangible_sunk_cost,
            "oil_opex": self._oil_opex_sunk_cost,
            "gas_opex": self._gas_opex_sunk_cost,
            "oil_asr": self._oil_asr_sunk_cost,
            "gas_asr": self._gas_asr_sunk_cost,
            "oil_lbt": self._oil_lbt_sunk_cost,
            "gas_lbt": self._gas_lbt_sunk_cost,
            "oil_cost_of_sales": self._oil_cost_of_sales_sunk_cost,
            "gas_cost_of_sales": self._gas_cost_of_sales_sunk_cost,
        }

        sunkcost_arr = {
            key: val.expenditures_pre_tax() for key, val in sunkcost_map.items()
        }

        # Define depreciable sunk costs
        oil_depreciable_sunk_cost = sunkcost_arr["oil_capital"]
        gas_depreciable_sunk_cost = sunkcost_arr["gas_capital"]
        self._oil_depreciable_sunk_cost = oil_depreciable_sunk_cost
        self._gas_depreciable_sunk_cost = gas_depreciable_sunk_cost

        # Define non-depreciable sunk costs
        oil_non_depreciable_sunk_cost = (
            sunkcost_arr["oil_intangible"]
            + sunkcost_arr["oil_opex"]
            + sunkcost_arr["oil_asr"]
            + sunkcost_arr["oil_lbt"]
            + sunkcost_arr["oil_cost_of_sales"]
        )

        gas_non_depreciable_sunk_cost = (
            sunkcost_arr["gas_intangible"]
            + sunkcost_arr["gas_opex"]
            + sunkcost_arr["gas_asr"]
            + sunkcost_arr["gas_lbt"]
            + sunkcost_arr["gas_cost_of_sales"]
        )
        self._oil_non_depreciable_sunk_cost = oil_non_depreciable_sunk_cost
        self._gas_non_depreciable_sunk_cost = gas_non_depreciable_sunk_cost

        # Calculate (total = depreciable + non_depreciable costs)
        self._oil_sunk_cost = oil_depreciable_sunk_cost + oil_non_depreciable_sunk_cost
        self._gas_sunk_cost = gas_depreciable_sunk_cost + gas_non_depreciable_sunk_cost

    def _get_preonstream_array(self) -> None:
        """
        Construct and organize preonstream cost arrays for OIL and GAS.

        This method gathers pre-tax expenditures for each preonstream cost category
        (Capital, Intangible, OPEX, ASR, LBT, and Cost of Sales), then groups them
        into depreciable and non-depreciable components. Finally, total preonstream
        costs are computed as the sum of these two components.

        Notes
        -----
        - Depreciable preonstream costs include only capital expenditures.
        - Non-depreciable preonstream costs are formed by summing intangible, OPEX,
          ASR, LBT, and cost of sales components.
        - Total preonstream costs are the sum of depreciable and non-depreciable costs.
        - The following internal attributes are set:
            * ``_oil_depreciable_preonstream``
            * ``_gas_depreciable_preonstream``
            * ``_oil_non_depreciable_preonstream``
            * ``_gas_non_depreciable_preonstream``
            * ``_oil_preonstream``
            * ``_gas_preonstream``
        - Internally relies on each cost object's
          ``expenditures_pre_tax()`` method to compute expenditures.
        """

        # Prepare preonstream arrays for each preonstream objects
        preonstream_map = {
            "oil_capital": self._oil_capital_preonstream,
            "gas_capital": self._gas_capital_preonstream,
            "oil_intangible": self._oil_intangible_preonstream,
            "gas_intangible": self._gas_intangible_preonstream,
            "oil_opex": self._oil_opex_preonstream,
            "gas_opex": self._gas_opex_preonstream,
            "oil_asr": self._oil_asr_preonstream,
            "gas_asr": self._gas_asr_preonstream,
            "oil_lbt": self._oil_lbt_preonstream,
            "gas_lbt": self._gas_lbt_preonstream,
            "oil_cost_of_sales": self._oil_cost_of_sales_preonstream,
            "gas_cost_of_sales": self._gas_cost_of_sales_preonstream,
        }

        preonstream_arr = {
            key: val.expenditures_pre_tax() for key, val in preonstream_map.items()
        }

        # Define depreciable preonstream costs
        oil_depreciable_preonstream = preonstream_arr["oil_capital"]
        gas_depreciable_preonstream = preonstream_arr["gas_capital"]
        self._oil_depreciable_preonstream = oil_depreciable_preonstream
        self._gas_depreciable_preonstream = gas_depreciable_preonstream

        # Define non-depreciable preonstream costs
        oil_non_depreciable_preonstream = (
            preonstream_arr["oil_intangible"]
            + preonstream_arr["oil_opex"]
            + preonstream_arr["oil_asr"]
            + preonstream_arr["oil_lbt"]
            + preonstream_arr["oil_cost_of_sales"]
        )
        gas_non_depreciable_preonstream = (
            preonstream_arr["gas_intangible"]
            + preonstream_arr["gas_opex"]
            + preonstream_arr["gas_asr"]
            + preonstream_arr["gas_lbt"]
            + preonstream_arr["gas_cost_of_sales"]
        )
        self._oil_non_depreciable_preonstream = oil_non_depreciable_preonstream
        self._gas_non_depreciable_preonstream = gas_non_depreciable_preonstream

        # Calculate (total = depreciable + non_depreciable costs)
        self._oil_preonstream = (
            oil_depreciable_preonstream + oil_non_depreciable_preonstream
        )
        self._gas_preonstream = (
            gas_depreciable_preonstream + gas_non_depreciable_preonstream
        )

    def _get_postonstream_array(self) -> None:
        """
        Construct post-onstream cost arrays for oil and gas.

        This method separates depreciable and non-depreciable post-onstream
        costs from post-tax expenditures, then aggregates them to obtain total
        post-onstream costs for oil and gas. All results are stored as internal
        attributes.

        Sets
        ----
        _oil_depreciable_postonstream : np.ndarray
            Depreciable oil post-onstream costs.
        _gas_depreciable_postonstream : np.ndarray
            Depreciable gas post-onstream costs.
        _oil_non_depreciable_postonstream : np.ndarray
            Non-depreciable oil post-onstream costs.
        _gas_non_depreciable_postonstream : np.ndarray
            Non-depreciable gas post-onstream costs.
        _oil_postonstream : np.ndarray
            Total oil post-onstream costs.
        _gas_postonstream : np.ndarray
            Total gas post-onstream costs.
        """

        oil_depreciable_postonstream = self._require_array(
            self._oil_capital_expenditures_post_tax,
            "_oil_capital_expenditures_post_tax",
        )
        gas_depreciable_postonstream = self._require_array(
            self._gas_capital_expenditures_post_tax,
            "_gas_capital_expenditures_post_tax",
        )
        oil_total_expenditures_post_tax = self._require_array(
            self._oil_total_expenditures_post_tax,
            "_oil_total_expenditures_post_tax",
        )
        gas_total_expenditures_post_tax = self._require_array(
            self._gas_total_expenditures_post_tax,
            "_gas_total_expenditures_post_tax",
        )

        # Define depreciable postonstream costs
        self._oil_depreciable_postonstream = oil_depreciable_postonstream
        self._gas_depreciable_postonstream = gas_depreciable_postonstream

        # Define non-depreciable preonstream costs
        oil_non_depreciable_postonstream = (
            oil_total_expenditures_post_tax - oil_depreciable_postonstream
        )
        gas_non_depreciable_postonstream = (
            gas_total_expenditures_post_tax - gas_depreciable_postonstream
        )
        self._oil_non_depreciable_postonstream = oil_non_depreciable_postonstream
        self._gas_non_depreciable_postonstream = gas_non_depreciable_postonstream

        # Calculate (total = depreciable + non_depreciable costs)
        self._oil_postonstream = (
            oil_depreciable_postonstream + oil_non_depreciable_postonstream
        )
        self._gas_postonstream = (
            gas_depreciable_postonstream + gas_non_depreciable_postonstream
        )

    def _get_investments(self) -> None:
        """
        Calculate and categorize total investments for oil and gas.

        This method computes both capital and non-capital investments for oil and
        gas separately, and then derives the total investment (expenses) as the
        sum of these components.

        Capital investments are derived from:
            - Depreciable pre-onstream costs
            - Capital expenditures (post-tax)

        Non-capital investments are derived from:
            - Non-depreciable pre-onstream costs
            - Intangible expenditures (post-tax)
            - Operating expenditures (post-tax)
            - Abandonment site restoration (ASR) costs (post-tax)
            - Land and building tax (LBT) (post-tax)
            - Cost-of-sales expenditures (post-tax)

        Returns
        -------
        None
            This method modifies the following attributes in place:

            - ``self._oil_capital`` : float
                Capital investment for oil.
            - ``self._gas_capital`` : float
                Capital investment for gas.
            - ``self._oil_non_capital`` : float
                Non-capital investment for oil.
            - ``self._gas_non_capital`` : float
                Non-capital investment for gas.
            - ``self._oil_total_expenses`` : float
                Total investment for oil (capital + non-capital).
            - ``self._gas_total_expenses`` : float
                Total investment for gas (capital + non-capital).

        Notes
        -----
        - All input components are assumed to be pre-computed and stored as
          attributes of the class instance.
        - Values are post-tax unless specified otherwise.
        - The calculation follows PSC economic evaluation conventions by
          distinguishing capital and non-capital investment categories.
        """

        oil_depreciable_preonstream = self._require_array(
            self._oil_depreciable_preonstream, "_oil_depreciable_preonstream"
        )
        gas_depreciable_preonstream = self._require_array(
            self._gas_depreciable_preonstream, "_gas_depreciable_preonstream"
        )
        oil_non_depreciable_preonstream = self._require_array(
            self._oil_non_depreciable_preonstream,
            "_oil_non_depreciable_preonstream",
        )
        gas_non_depreciable_preonstream = self._require_array(
            self._gas_non_depreciable_preonstream,
            "_gas_non_depreciable_preonstream",
        )
        oil_capital_expenditures_post_tax = self._require_array(
            self._oil_capital_expenditures_post_tax,
            "_oil_capital_expenditures_post_tax",
        )
        gas_capital_expenditures_post_tax = self._require_array(
            self._gas_capital_expenditures_post_tax,
            "_gas_capital_expenditures_post_tax",
        )
        oil_intangible_expenditures_post_tax = self._require_array(
            self._oil_intangible_expenditures_post_tax,
            "_oil_intangible_expenditures_post_tax",
        )
        gas_intangible_expenditures_post_tax = self._require_array(
            self._gas_intangible_expenditures_post_tax,
            "_gas_intangible_expenditures_post_tax",
        )
        oil_opex_expenditures_post_tax = self._require_array(
            self._oil_opex_expenditures_post_tax,
            "_oil_opex_expenditures_post_tax",
        )
        gas_opex_expenditures_post_tax = self._require_array(
            self._gas_opex_expenditures_post_tax,
            "_gas_opex_expenditures_post_tax",
        )
        oil_asr_expenditures_post_tax = self._require_array(
            self._oil_asr_expenditures_post_tax,
            "_oil_asr_expenditures_post_tax",
        )
        gas_asr_expenditures_post_tax = self._require_array(
            self._gas_asr_expenditures_post_tax,
            "_gas_asr_expenditures_post_tax",
        )
        oil_lbt_expenditures_post_tax = self._require_array(
            self._oil_lbt_expenditures_post_tax,
            "_oil_lbt_expenditures_post_tax",
        )
        gas_lbt_expenditures_post_tax = self._require_array(
            self._gas_lbt_expenditures_post_tax,
            "_gas_lbt_expenditures_post_tax",
        )
        oil_cost_of_sales_expenditures_post_tax = self._require_array(
            self._oil_cost_of_sales_expenditures_post_tax,
            "_oil_cost_of_sales_expenditures_post_tax",
        )
        gas_cost_of_sales_expenditures_post_tax = self._require_array(
            self._gas_cost_of_sales_expenditures_post_tax,
            "_gas_cost_of_sales_expenditures_post_tax",
        )

        # Capital investments
        oil_capital = oil_depreciable_preonstream + oil_capital_expenditures_post_tax
        gas_capital = gas_depreciable_preonstream + gas_capital_expenditures_post_tax
        self._oil_capital = oil_capital
        self._gas_capital = gas_capital

        # Non-capital investments
        oil_non_capital = (
            oil_non_depreciable_preonstream
            + oil_intangible_expenditures_post_tax
            + oil_opex_expenditures_post_tax
            + oil_asr_expenditures_post_tax
            + oil_lbt_expenditures_post_tax
            + oil_cost_of_sales_expenditures_post_tax
        )

        gas_non_capital = (
            gas_non_depreciable_preonstream
            + gas_intangible_expenditures_post_tax
            + gas_opex_expenditures_post_tax
            + gas_asr_expenditures_post_tax
            + gas_lbt_expenditures_post_tax
            + gas_cost_of_sales_expenditures_post_tax
        )
        self._oil_non_capital = oil_non_capital
        self._gas_non_capital = gas_non_capital

        # Total investments
        self._oil_total_expenses = oil_capital + oil_non_capital
        self._gas_total_expenses = gas_capital + gas_non_capital

    def _calc_pre_tax_expenditures(
        self,
        target_attr: CapitalCost | Intangible | OPEX | ASR | LBT,
        year_inflation: npt.NDArray[np.int64] | None = None,
        inflation_rate: npt.NDArray[np.float64] | int | float = 0.0,
        inflation_rate_applied_to: InflationAppliedTo | None = None,
    ) -> npt.NDArray[np.float64]:
        """
        Calculate pre-tax expenditures with optional inflation adjustments.

        This method computes pre-tax expenditures for a given cost category
        (e.g., capital, intangible, OPEX, ASR, LBT), applying inflation
        rates conditionally based on the specified `inflation_rate_applied_to`
        setting.

        The inflation can be applied selectively to:
        - Capital expenditures (CAPEX)
        - Operating expenditures (OPEX)
        - Both CAPEX and OPEX
        - Or not applied at all

        Parameters
        ----------
        target_attr : CapitalCost or Intangible or OPEX or ASR or LBT
            The cost object representing a specific expenditure category.
        year_inflation : numpy.ndarray, optional
            Array of project years aligned with inflation application.
            Default is ``None`` (no inflation applied by year).
        inflation_rate : numpy.ndarray or int or float, default=0.0
            Inflation rate(s) to apply. Can be a scalar (constant rate),
            a 1D array matching `year_inflation`, or 0.0 (no inflation).
        inflation_rate_applied_to : InflationAppliedTo or None, optional
            Specifies where inflation is applied. Options:
            - ``None`` : No inflation applied (default).
            - ``InflationAppliedTo.CAPEX`` : Apply only to CAPEX-related attributes.
            - ``InflationAppliedTo.OPEX`` : Apply only to OPEX-related attributes.
            - ``InflationAppliedTo.CAPEX_AND_OPEX`` : Apply to both CAPEX and OPEX.

        Returns
        -------
        numpy.ndarray
            Array of pre-tax expenditures, adjusted for inflation according
            to the specified rules.

        Raises
        ------
        BaseProjectException
            If `inflation_rate_applied_to` is not recognized.
        """

        inflation_target: object = inflation_rate_applied_to
        # No inflation rate
        if inflation_target is None:
            return target_attr.expenditures_pre_tax(
                year_inflation=year_inflation,
                inflation_rate=0.0,
            )

        # Inflation rate applied to CAPEX only
        elif inflation_target == InflationAppliedTo.CAPEX:
            if (
                target_attr is self._oil_capital_postonstream
                or target_attr is self._gas_capital_postonstream
                or target_attr is self._oil_intangible_postonstream
                or target_attr is self._gas_intangible_postonstream
            ):
                return target_attr.expenditures_pre_tax(
                    year_inflation=year_inflation,
                    inflation_rate=inflation_rate,
                )

            else:
                return target_attr.expenditures_pre_tax(
                    year_inflation=year_inflation,
                    inflation_rate=0.0,
                )

        # Inflation rate applied to OPEX only
        elif inflation_target == InflationAppliedTo.OPEX:
            if (
                target_attr is self._oil_opex_postonstream
                or target_attr is self._gas_opex_postonstream
            ):
                return target_attr.expenditures_pre_tax(
                    year_inflation=year_inflation,
                    inflation_rate=inflation_rate,
                )

            else:
                return target_attr.expenditures_pre_tax(
                    year_inflation=year_inflation,
                    inflation_rate=0.0,
                )

        # Inflation rate applied to CAPEX and OPEX
        elif inflation_target == InflationAppliedTo.CAPEX_AND_OPEX:
            if (
                target_attr is self._oil_capital_postonstream
                or target_attr is self._gas_capital_postonstream
                or target_attr is self._oil_intangible_postonstream
                or target_attr is self._gas_intangible_postonstream
                or target_attr is self._oil_opex_postonstream
                or target_attr is self._gas_opex_postonstream
            ):
                return target_attr.expenditures_pre_tax(
                    year_inflation=year_inflation, inflation_rate=inflation_rate
                )

            else:
                return target_attr.expenditures_pre_tax(
                    year_inflation=year_inflation,
                    inflation_rate=0.0,
                )

    def _get_expenditures_pre_tax(
        self,
        year_inflation: npt.NDArray[np.int64] | None = None,
        inflation_rate: npt.NDArray[np.float64] | int | float = 0.0,
        inflation_rate_applied_to: InflationAppliedTo | None = None,
    ) -> None:
        """
        Calculate and assign pre-tax expenditures for various categories,
        adjusted for inflation.

        This method calculates the pre-tax expenditures for multiple cost
        categories related to oil and gas, adjusted by the provided inflation
        rate. The expenditures are then assigned to the corresponding attributes
        for both oil and gas.

        Parameters
        ----------
        year_inflation : np.ndarray, optional
            A NumPy array representing the years during which inflation is
            applied to the costs. If not provided, defaults to repeating the
            `start_year` for all costs.
        inflation_rate : np.ndarray or float, optional
            The inflation rate(s) to apply to the project costs. If provided as a float,
            a uniform inflation rate is applied. If provided as a NumPy array, different
            rates are applied based on the corresponding project years. Default is 0.0.
        inflation_rate_applied_to
            The selection of where the cost inflation will be applied to.

        Returns
        -------
        None
            This method does not return a value. It updates the following attributes
            with the calculated pre-tax expenditures:
            -   `_oil_capital_expenditures_pre_tax`
            -   `_gas_capital_expenditures_pre_tax`
            -   `_oil_intangible_expenditures_pre_tax`
            -   `_gas_intangible_expenditures_pre_tax`
            -   `_oil_opex_expenditures_pre_tax`
            -   `_gas_opex_expenditures_pre_tax`
            -   `_oil_asr_expenditures_pre_tax`
            -   `_gas_asr_expenditures_pre_tax`
            -   `_oil_lbt_expenditures_pre_tax`
            -   `_gas_lbt_expenditures_pre_tax`
            -   `_oil_cost_of_sales_expenditures_pre_tax`
            -   `_gas_cost_of_sales_expenditures_pre_tax`
        """

        # Prepare expenditures pre tax associated with capital, intangible,
        # opex, asr, and lbt costs
        (
            self._oil_capital_expenditures_pre_tax,
            self._gas_capital_expenditures_pre_tax,
            self._oil_intangible_expenditures_pre_tax,
            self._gas_intangible_expenditures_pre_tax,
            self._oil_opex_expenditures_pre_tax,
            self._gas_opex_expenditures_pre_tax,
            self._oil_asr_expenditures_pre_tax,
            self._gas_asr_expenditures_pre_tax,
            self._oil_lbt_expenditures_pre_tax,
            self._gas_lbt_expenditures_pre_tax,
        ) = [
            self._calc_pre_tax_expenditures(
                target_attr=attr,
                year_inflation=year_inflation,
                inflation_rate=inflation_rate,
                inflation_rate_applied_to=inflation_rate_applied_to,
            )
            for attr in [
                self._oil_capital_postonstream,
                self._gas_capital_postonstream,
                self._oil_intangible_postonstream,
                self._gas_intangible_postonstream,
                self._oil_opex_postonstream,
                self._gas_opex_postonstream,
                self._oil_asr_postonstream,
                self._gas_asr_postonstream,
                self._oil_lbt_postonstream,
                self._gas_lbt_postonstream,
            ]
        ]

        # Prepare expenditures pre tax associated with cost of sales
        (
            self._oil_cost_of_sales_expenditures_pre_tax,
            self._gas_cost_of_sales_expenditures_pre_tax,
        ) = [
            attr.expenditures_pre_tax()
            for attr in [
                self._oil_cost_of_sales_postonstream,
                self._gas_cost_of_sales_postonstream,
            ]
        ]

    def _get_indirect_taxes(
        self, tax_rate: npt.NDArray[np.float64] | float = 0.0
    ) -> None:
        """
        Calculate and assign indirect taxes for various oil and gas
        expenditure categories.

        This method computes the indirect taxes (such as VAT or other
        applicable indirect taxes) for multiple categories related to
        oil and gas. It adjusts these taxes based on the provided
        tax portion, tax rate, and tax discount.

        Parameters
        ----------
        tax_rate : np.ndarray or float, optional
            A NumPy array or float representing the tax rate applied to
            the costs. If not provided, a default rate of 0.0 will be used.
            When provided as an array, it should match the project years.

        Returns
        -------
        None
            This method does not return any values. It updates the following
            attributes with the calculated indirect taxes for oil and gas:
            -   `_oil_capital_indirect_tax`
            -   `_gas_capital_indirect_tax`
            -   `_oil_intangible_indirect_tax`
            -   `_gas_intangible_indirect_tax`
            -   `_oil_opex_indirect_tax`
            -   `_gas_opex_indirect_tax`
            -   `_oil_asr_indirect_tax`
            -   `_gas_asr_indirect_tax`
            -   `_oil_lbt_indirect_tax`
            -   `_gas_lbt_indirect_tax`
            -   `_oil_cost_of_sales_indirect_tax`
            -   `_gas_cost_of_sales_indirect_tax`
        """

        # Prepare indirect taxes associated with capital, intangible,
        # opex, asr, and lbt costs
        (
            self._oil_capital_indirect_tax,
            self._gas_capital_indirect_tax,
            self._oil_intangible_indirect_tax,
            self._gas_intangible_indirect_tax,
            self._oil_opex_indirect_tax,
            self._gas_opex_indirect_tax,
            self._oil_asr_indirect_tax,
            self._gas_asr_indirect_tax,
            self._oil_lbt_indirect_tax,
            self._gas_lbt_indirect_tax,
        ) = [
            attr.indirect_taxes(tax_rate=tax_rate)
            for attr in [
                self._oil_capital_postonstream,
                self._gas_capital_postonstream,
                self._oil_intangible_postonstream,
                self._gas_intangible_postonstream,
                self._oil_opex_postonstream,
                self._gas_opex_postonstream,
                self._oil_asr_postonstream,
                self._gas_asr_postonstream,
                self._oil_lbt_postonstream,
                self._gas_lbt_postonstream,
            ]
        ]

        # Prepare indirect taxes associated with cost of sales
        (
            self._oil_cost_of_sales_indirect_tax,
            self._gas_cost_of_sales_indirect_tax,
        ) = [
            attr.indirect_taxes()
            for attr in [
                self._oil_cost_of_sales_postonstream,
                self._gas_cost_of_sales_postonstream,
            ]
        ]

    def _get_expenditures_post_tax(self) -> None:
        """
        Compute and assign post-tax expenditures for all oil and gas cost categories.

        This method iterates over each defined cost category for both oil and gas.
        For every category, it retrieves the corresponding pre-tax expenditure and
        indirect tax attributes, computes their sum, and assigns the result to the
        appropriate post-tax expenditure attribute.

        Notes
        -----
        - Supported cost categories include:
            * ``capital``
            * ``intangible``
            * ``opex``
            * ``asr`` (abandonment & site restoration)
            * ``lbt`` (land and building tax)
            * ``cost_of_sales``
        - For each category and fluid type (``oil`` and ``gas``), the method expects
          the following attributes to exist:
            * ``_{fluid}_{category}_expenditures_pre_tax``
            * ``_{fluid}_{category}_indirect_tax``
        - The computed results are stored in:
            * ``_{fluid}_{category}_expenditures_post_tax``

        Examples
        --------
        For ``oil_capital``:
            - Reads ``_oil_capital_expenditures_pre_tax`` and ``_oil_capital_indirect_tax``.
            - Stores the sum in ``_oil_capital_expenditures_post_tax``.
        """

        self._oil_capital_expenditures_post_tax = self._require_array(
            self._oil_capital_expenditures_pre_tax, "_oil_capital_expenditures_pre_tax"
        ) + self._require_array(
            self._oil_capital_indirect_tax, "_oil_capital_indirect_tax"
        )
        self._gas_capital_expenditures_post_tax = self._require_array(
            self._gas_capital_expenditures_pre_tax, "_gas_capital_expenditures_pre_tax"
        ) + self._require_array(
            self._gas_capital_indirect_tax, "_gas_capital_indirect_tax"
        )
        self._oil_intangible_expenditures_post_tax = self._require_array(
            self._oil_intangible_expenditures_pre_tax,
            "_oil_intangible_expenditures_pre_tax",
        ) + self._require_array(
            self._oil_intangible_indirect_tax, "_oil_intangible_indirect_tax"
        )
        self._gas_intangible_expenditures_post_tax = self._require_array(
            self._gas_intangible_expenditures_pre_tax,
            "_gas_intangible_expenditures_pre_tax",
        ) + self._require_array(
            self._gas_intangible_indirect_tax, "_gas_intangible_indirect_tax"
        )
        self._oil_opex_expenditures_post_tax = self._require_array(
            self._oil_opex_expenditures_pre_tax, "_oil_opex_expenditures_pre_tax"
        ) + self._require_array(self._oil_opex_indirect_tax, "_oil_opex_indirect_tax")
        self._gas_opex_expenditures_post_tax = self._require_array(
            self._gas_opex_expenditures_pre_tax, "_gas_opex_expenditures_pre_tax"
        ) + self._require_array(self._gas_opex_indirect_tax, "_gas_opex_indirect_tax")
        self._oil_asr_expenditures_post_tax = self._require_array(
            self._oil_asr_expenditures_pre_tax, "_oil_asr_expenditures_pre_tax"
        ) + self._require_array(self._oil_asr_indirect_tax, "_oil_asr_indirect_tax")
        self._gas_asr_expenditures_post_tax = self._require_array(
            self._gas_asr_expenditures_pre_tax, "_gas_asr_expenditures_pre_tax"
        ) + self._require_array(self._gas_asr_indirect_tax, "_gas_asr_indirect_tax")
        self._oil_lbt_expenditures_post_tax = self._require_array(
            self._oil_lbt_expenditures_pre_tax, "_oil_lbt_expenditures_pre_tax"
        ) + self._require_array(self._oil_lbt_indirect_tax, "_oil_lbt_indirect_tax")
        self._gas_lbt_expenditures_post_tax = self._require_array(
            self._gas_lbt_expenditures_pre_tax, "_gas_lbt_expenditures_pre_tax"
        ) + self._require_array(self._gas_lbt_indirect_tax, "_gas_lbt_indirect_tax")
        self._oil_cost_of_sales_expenditures_post_tax = self._require_array(
            self._oil_cost_of_sales_expenditures_pre_tax,
            "_oil_cost_of_sales_expenditures_pre_tax",
        ) + self._require_array(
            self._oil_cost_of_sales_indirect_tax, "_oil_cost_of_sales_indirect_tax"
        )
        self._gas_cost_of_sales_expenditures_post_tax = self._require_array(
            self._gas_cost_of_sales_expenditures_pre_tax,
            "_gas_cost_of_sales_expenditures_pre_tax",
        ) + self._require_array(
            self._gas_cost_of_sales_indirect_tax, "_gas_cost_of_sales_indirect_tax"
        )

    def _get_other_revenue(
        self,
        sulfur_revenue: OtherRevenue,
        electricity_revenue: OtherRevenue,
        co2_revenue: OtherRevenue,
    ) -> None:
        """
        Allocate additional revenue streams (sulfur, electricity, and CO₂) to
        either oil revenue, gas revenue, or as a reduction to operating
        expenditures (OPEX).

        Parameters
        ----------
        sulfur_revenue : OtherRevenue, optional
            The allocation method for sulfur revenue.
            Possible options:
            - `OtherRevenue.ADDITION_TO_OIL_REVENUE`: Adds to oil revenue.
            - `OtherRevenue.ADDITION_TO_GAS_REVENUE`: Adds to gas revenue.
            - `OtherRevenue.REDUCTION_TO_OIL_OPEX`: Reduces oil OPEX.
            - `OtherRevenue.REDUCTION_TO_GAS_OPEX`: Reduces gas OPEX.

        electricity_revenue : OtherRevenue, optional
            The allocation method for electricity revenue.
            Possible options are the same as `sulfur_revenue`.

        co2_revenue : OtherRevenue, optional
            The allocation method for CO₂ revenue.
            Possible options are the same as `sulfur_revenue`.

        Raises
        ------
        OtherRevenueException
            If an invalid revenue allocation option is provided for sulfur,
            electricity, or CO₂.

        Notes
        -----
        - Depending on the selected revenue allocation, sulfur, electricity,
          and CO₂ revenue may be added to oil or gas revenue or used to offset
          post-tax OPEX expenditures.
        - If an invalid allocation option is provided, an `OtherRevenueException`
          is raised.
        """

        sulfur_selection: object = sulfur_revenue
        electricity_selection: object = electricity_revenue
        co2_selection: object = co2_revenue

        # Configure sulfur revenue
        if sulfur_selection is OtherRevenue.ADDITION_TO_OIL_REVENUE:
            self._oil_revenue += self._sulfur_revenue

        elif sulfur_selection is OtherRevenue.ADDITION_TO_GAS_REVENUE:
            self._gas_revenue += self._sulfur_revenue

        elif sulfur_selection is OtherRevenue.REDUCTION_TO_OIL_OPEX:
            oil_opex_expenditures_post_tax = self._require_array(
                self._oil_opex_expenditures_post_tax,
                "_oil_opex_expenditures_post_tax",
            )
            oil_opex_expenditures_post_tax -= self._sulfur_revenue

        elif sulfur_selection is OtherRevenue.REDUCTION_TO_GAS_OPEX:
            gas_opex_expenditures_post_tax = self._require_array(
                self._gas_opex_expenditures_post_tax,
                "_gas_opex_expenditures_post_tax",
            )
            gas_opex_expenditures_post_tax -= self._sulfur_revenue

        # Configure electricity revenue
        if electricity_selection is OtherRevenue.ADDITION_TO_OIL_REVENUE:
            self._oil_revenue += self._electricity_revenue

        elif electricity_selection is OtherRevenue.ADDITION_TO_GAS_REVENUE:
            self._gas_revenue += self._electricity_revenue

        elif electricity_selection is OtherRevenue.REDUCTION_TO_OIL_OPEX:
            oil_opex_expenditures_post_tax = self._require_array(
                self._oil_opex_expenditures_post_tax,
                "_oil_opex_expenditures_post_tax",
            )
            oil_opex_expenditures_post_tax -= self._electricity_revenue

        elif electricity_selection is OtherRevenue.REDUCTION_TO_GAS_OPEX:
            gas_opex_expenditures_post_tax = self._require_array(
                self._gas_opex_expenditures_post_tax,
                "_gas_opex_expenditures_post_tax",
            )
            gas_opex_expenditures_post_tax -= self._electricity_revenue

        # Configure CO2 revenue
        if co2_selection is OtherRevenue.ADDITION_TO_OIL_REVENUE:
            self._oil_revenue += self._co2_revenue

        elif co2_selection is OtherRevenue.ADDITION_TO_GAS_REVENUE:
            self._gas_revenue += self._co2_revenue

        elif co2_selection is OtherRevenue.REDUCTION_TO_OIL_OPEX:
            oil_opex_expenditures_post_tax = self._require_array(
                self._oil_opex_expenditures_post_tax,
                "_oil_opex_expenditures_post_tax",
            )
            oil_opex_expenditures_post_tax -= self._co2_revenue

        elif co2_selection is OtherRevenue.REDUCTION_TO_GAS_OPEX:
            gas_opex_expenditures_post_tax = self._require_array(
                self._gas_opex_expenditures_post_tax,
                "_gas_opex_expenditures_post_tax",
            )
            gas_opex_expenditures_post_tax -= self._co2_revenue

    def _get_consolidated_profiles(self) -> None:
        """
        Aggregate oil and gas profiles into consolidated project-wide profiles.

        This method combines oil and gas arrays across multiple categories
        (lifting, prices, revenues, sunk costs, pre-onstream costs,
        expenditures, indirect taxes, and cash flows) into consolidated
        project-level attributes.

        The aggregation is performed by element-wise addition of the corresponding
        oil and gas arrays.

        Returns
        -------
        None
            Updates instance attributes with consolidated values.

        Notes
        -----
        - Consolidation is applied to the following categories:
            * Lifting, weighted average price (WAP), and revenue
            * Sunk costs (depreciable, non-depreciable, and total)
            * Pre-onstream costs (depreciable, non-depreciable, and total)
            * Expenditures (pre-tax, post-tax, and by category)
            * Indirect taxes (total and by category)
            * Cash flow and non-capital components
        - Expenditure- and tax-related categories are consolidated dynamically
          using predefined category names (``capital``, ``intangible``, ``opex``,
          ``asr``, ``lbt``, ``cost_of_sales``).
        - All oil and gas attributes must be defined and have matching array shapes.
        """

        oil_wap_price = self._require_array(self._oil_wap_price, "_oil_wap_price")
        gas_wap_price = self._require_array(self._gas_wap_price, "_gas_wap_price")
        oil_depreciable_sunk_cost = self._require_array(
            self._oil_depreciable_sunk_cost, "_oil_depreciable_sunk_cost"
        )
        gas_depreciable_sunk_cost = self._require_array(
            self._gas_depreciable_sunk_cost, "_gas_depreciable_sunk_cost"
        )
        oil_non_depreciable_sunk_cost = self._require_array(
            self._oil_non_depreciable_sunk_cost,
            "_oil_non_depreciable_sunk_cost",
        )
        gas_non_depreciable_sunk_cost = self._require_array(
            self._gas_non_depreciable_sunk_cost,
            "_gas_non_depreciable_sunk_cost",
        )
        oil_sunk_cost = self._require_array(self._oil_sunk_cost, "_oil_sunk_cost")
        gas_sunk_cost = self._require_array(self._gas_sunk_cost, "_gas_sunk_cost")
        oil_depreciable_preonstream = self._require_array(
            self._oil_depreciable_preonstream, "_oil_depreciable_preonstream"
        )
        gas_depreciable_preonstream = self._require_array(
            self._gas_depreciable_preonstream, "_gas_depreciable_preonstream"
        )
        oil_non_depreciable_preonstream = self._require_array(
            self._oil_non_depreciable_preonstream,
            "_oil_non_depreciable_preonstream",
        )
        gas_non_depreciable_preonstream = self._require_array(
            self._gas_non_depreciable_preonstream,
            "_gas_non_depreciable_preonstream",
        )
        oil_preonstream = self._require_array(self._oil_preonstream, "_oil_preonstream")
        gas_preonstream = self._require_array(self._gas_preonstream, "_gas_preonstream")
        oil_depreciable_postonstream = self._require_array(
            self._oil_depreciable_postonstream, "_oil_depreciable_postonstream"
        )
        gas_depreciable_postonstream = self._require_array(
            self._gas_depreciable_postonstream, "_gas_depreciable_postonstream"
        )
        oil_non_depreciable_postonstream = self._require_array(
            self._oil_non_depreciable_postonstream,
            "_oil_non_depreciable_postonstream",
        )
        gas_non_depreciable_postonstream = self._require_array(
            self._gas_non_depreciable_postonstream,
            "_gas_non_depreciable_postonstream",
        )
        oil_postonstream = self._require_array(
            self._oil_postonstream, "_oil_postonstream"
        )
        gas_postonstream = self._require_array(
            self._gas_postonstream, "_gas_postonstream"
        )

        # Attributes associated with consolidated lifting
        self._consolidated_lifting = (
            self._oil_lifting.get_lifting_rate_ghv_arr()
            + self._gas_lifting.get_lifting_rate_ghv_arr()
        )
        self._consolidated_wap_price = oil_wap_price + gas_wap_price
        self._consolidated_revenue = self._oil_revenue + self._gas_revenue

        # Attributes associated with consolidated sunkcost
        self._consolidated_depreciable_sunk_cost = (
            oil_depreciable_sunk_cost + gas_depreciable_sunk_cost
        )
        self._consolidated_non_depreciable_sunk_cost = (
            oil_non_depreciable_sunk_cost + gas_non_depreciable_sunk_cost
        )
        self._consolidated_sunk_cost = oil_sunk_cost + gas_sunk_cost

        # Attributes associated with consolidated preonstream
        self._consolidated_depreciable_preonstream = (
            oil_depreciable_preonstream + gas_depreciable_preonstream
        )
        self._consolidated_non_depreciable_preonstream = (
            oil_non_depreciable_preonstream + gas_non_depreciable_preonstream
        )
        self._consolidated_preonstream = oil_preonstream + gas_preonstream

        # Attributes associated with consolidated postonstream
        self._consolidated_depreciable_postonstream = (
            oil_depreciable_postonstream + gas_depreciable_postonstream
        )
        self._consolidated_non_depreciable_postonstream = (
            oil_non_depreciable_postonstream + gas_non_depreciable_postonstream
        )
        self._consolidated_postonstream = oil_postonstream + gas_postonstream

        # Consolidate each typed expenditure category explicitly.  These arrays are
        # part of the public table/summary calculation order and must not be looked
        # up dynamically.
        self._consolidated_capital_expenditures_pre_tax = self._require_array(
            self._oil_capital_expenditures_pre_tax, "_oil_capital_expenditures_pre_tax"
        ) + self._require_array(
            self._gas_capital_expenditures_pre_tax, "_gas_capital_expenditures_pre_tax"
        )
        self._consolidated_capital_indirect_tax = self._require_array(
            self._oil_capital_indirect_tax, "_oil_capital_indirect_tax"
        ) + self._require_array(
            self._gas_capital_indirect_tax, "_gas_capital_indirect_tax"
        )
        self._consolidated_capital_expenditures_post_tax = self._require_array(
            self._oil_capital_expenditures_post_tax,
            "_oil_capital_expenditures_post_tax",
        ) + self._require_array(
            self._gas_capital_expenditures_post_tax,
            "_gas_capital_expenditures_post_tax",
        )
        self._consolidated_intangible_expenditures_pre_tax = self._require_array(
            self._oil_intangible_expenditures_pre_tax,
            "_oil_intangible_expenditures_pre_tax",
        ) + self._require_array(
            self._gas_intangible_expenditures_pre_tax,
            "_gas_intangible_expenditures_pre_tax",
        )
        self._consolidated_intangible_indirect_tax = self._require_array(
            self._oil_intangible_indirect_tax, "_oil_intangible_indirect_tax"
        ) + self._require_array(
            self._gas_intangible_indirect_tax, "_gas_intangible_indirect_tax"
        )
        self._consolidated_intangible_expenditures_post_tax = self._require_array(
            self._oil_intangible_expenditures_post_tax,
            "_oil_intangible_expenditures_post_tax",
        ) + self._require_array(
            self._gas_intangible_expenditures_post_tax,
            "_gas_intangible_expenditures_post_tax",
        )
        self._consolidated_opex_expenditures_pre_tax = self._require_array(
            self._oil_opex_expenditures_pre_tax, "_oil_opex_expenditures_pre_tax"
        ) + self._require_array(
            self._gas_opex_expenditures_pre_tax, "_gas_opex_expenditures_pre_tax"
        )
        self._consolidated_opex_indirect_tax = self._require_array(
            self._oil_opex_indirect_tax, "_oil_opex_indirect_tax"
        ) + self._require_array(self._gas_opex_indirect_tax, "_gas_opex_indirect_tax")
        self._consolidated_opex_expenditures_post_tax = self._require_array(
            self._oil_opex_expenditures_post_tax, "_oil_opex_expenditures_post_tax"
        ) + self._require_array(
            self._gas_opex_expenditures_post_tax, "_gas_opex_expenditures_post_tax"
        )
        self._consolidated_asr_expenditures_pre_tax = self._require_array(
            self._oil_asr_expenditures_pre_tax, "_oil_asr_expenditures_pre_tax"
        ) + self._require_array(
            self._gas_asr_expenditures_pre_tax, "_gas_asr_expenditures_pre_tax"
        )
        self._consolidated_asr_indirect_tax = self._require_array(
            self._oil_asr_indirect_tax, "_oil_asr_indirect_tax"
        ) + self._require_array(self._gas_asr_indirect_tax, "_gas_asr_indirect_tax")
        self._consolidated_asr_expenditures_post_tax = self._require_array(
            self._oil_asr_expenditures_post_tax, "_oil_asr_expenditures_post_tax"
        ) + self._require_array(
            self._gas_asr_expenditures_post_tax, "_gas_asr_expenditures_post_tax"
        )
        self._consolidated_lbt_expenditures_pre_tax = self._require_array(
            self._oil_lbt_expenditures_pre_tax, "_oil_lbt_expenditures_pre_tax"
        ) + self._require_array(
            self._gas_lbt_expenditures_pre_tax, "_gas_lbt_expenditures_pre_tax"
        )
        self._consolidated_lbt_indirect_tax = self._require_array(
            self._oil_lbt_indirect_tax, "_oil_lbt_indirect_tax"
        ) + self._require_array(self._gas_lbt_indirect_tax, "_gas_lbt_indirect_tax")
        self._consolidated_lbt_expenditures_post_tax = self._require_array(
            self._oil_lbt_expenditures_post_tax, "_oil_lbt_expenditures_post_tax"
        ) + self._require_array(
            self._gas_lbt_expenditures_post_tax, "_gas_lbt_expenditures_post_tax"
        )
        self._consolidated_cost_of_sales_expenditures_pre_tax = self._require_array(
            self._oil_cost_of_sales_expenditures_pre_tax,
            "_oil_cost_of_sales_expenditures_pre_tax",
        ) + self._require_array(
            self._gas_cost_of_sales_expenditures_pre_tax,
            "_gas_cost_of_sales_expenditures_pre_tax",
        )
        self._consolidated_cost_of_sales_indirect_tax = self._require_array(
            self._oil_cost_of_sales_indirect_tax, "_oil_cost_of_sales_indirect_tax"
        ) + self._require_array(
            self._gas_cost_of_sales_indirect_tax, "_gas_cost_of_sales_indirect_tax"
        )
        self._consolidated_cost_of_sales_expenditures_post_tax = self._require_array(
            self._oil_cost_of_sales_expenditures_post_tax,
            "_oil_cost_of_sales_expenditures_post_tax",
        ) + self._require_array(
            self._gas_cost_of_sales_expenditures_post_tax,
            "_gas_cost_of_sales_expenditures_post_tax",
        )

        oil_total_expenditures_pre_tax = self._require_array(
            self._oil_total_expenditures_pre_tax,
            "_oil_total_expenditures_pre_tax",
        )
        gas_total_expenditures_pre_tax = self._require_array(
            self._gas_total_expenditures_pre_tax,
            "_gas_total_expenditures_pre_tax",
        )
        oil_total_indirect_tax = self._require_array(
            self._oil_total_indirect_tax, "_oil_total_indirect_tax"
        )
        gas_total_indirect_tax = self._require_array(
            self._gas_total_indirect_tax, "_gas_total_indirect_tax"
        )
        oil_total_expenditures_post_tax = self._require_array(
            self._oil_total_expenditures_post_tax,
            "_oil_total_expenditures_post_tax",
        )
        gas_total_expenditures_post_tax = self._require_array(
            self._gas_total_expenditures_post_tax,
            "_gas_total_expenditures_post_tax",
        )
        oil_capital = self._require_array(self._oil_capital, "_oil_capital")
        gas_capital = self._require_array(self._gas_capital, "_gas_capital")
        oil_non_capital = self._require_array(self._oil_non_capital, "_oil_non_capital")
        gas_non_capital = self._require_array(self._gas_non_capital, "_gas_non_capital")
        oil_total_expenses = self._require_array(
            self._oil_total_expenses, "_oil_total_expenses"
        )
        gas_total_expenses = self._require_array(
            self._gas_total_expenses, "_gas_total_expenses"
        )
        oil_cashflow = self._require_array(self._oil_cashflow, "_oil_cashflow")
        gas_cashflow = self._require_array(self._gas_cashflow, "_gas_cashflow")

        self._consolidated_total_expenditures_pre_tax = (
            oil_total_expenditures_pre_tax + gas_total_expenditures_pre_tax
        )

        self._consolidated_total_indirect_tax = (
            oil_total_indirect_tax + gas_total_indirect_tax
        )

        self._consolidated_total_expenditures_post_tax = (
            oil_total_expenditures_post_tax + gas_total_expenditures_post_tax
        )

        # Attributes associated with consolidated investments
        self._consolidated_capital = oil_capital + gas_capital
        self._consolidated_non_capital = oil_non_capital + gas_non_capital
        self._consolidated_total_expenses = oil_total_expenses + gas_total_expenses

        # Attribute associated with consolidated cashflow
        self._consolidated_cashflow = oil_cashflow + gas_cashflow

    def run(
        self,
        sulfur_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_GAS_REVENUE,
        electricity_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_OIL_REVENUE,
        co2_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_GAS_REVENUE,
        vat_rate: npt.NDArray[np.float64] | float = 0.0,
        year_inflation: npt.NDArray[np.int64] | None = None,
        inflation_rate: npt.NDArray[np.float64] | float = 0.0,
        inflation_rate_applied_to: InflationAppliedTo | None = None,
    ) -> None:
        """
        Executes the full economic calculation workflow for a base project.

        This method performs the complete sequence of calculations required to produce
        oil, gas, and consolidated economic profiles. It validates sunk costs, computes
        expenditures (pre-tax and post-tax), indirect taxes, other revenues, and
        cashflows, then prepares the consolidated profiles for reporting or further analysis.

        Parameters
        ----------
        sulfur_revenue : OtherRevenue, default=OtherRevenue.ADDITION_TO_GAS_REVENUE
            How sulfur revenue is allocated (e.g., added to gas revenue or treated separately).
        electricity_revenue : OtherRevenue, default=OtherRevenue.ADDITION_TO_OIL_REVENUE
            How electricity revenue is allocated.
        co2_revenue : OtherRevenue, default=OtherRevenue.ADDITION_TO_GAS_REVENUE
            How CO₂ revenue is allocated.
        vat_rate : np.ndarray or float, default=0.0
            Indirect tax rate(s) applied to applicable expenditures.
        year_inflation : np.ndarray, optional
            Array specifying the inflation year for each time step.
        inflation_rate : np.ndarray or float, default=0.0
            Inflation rate(s) to be applied.
        inflation_rate_applied_to : InflationAppliedTo, optional
            Specifies which expenditures the inflation rate should be applied to.
        """

        self._has_run = False

        # WAP (Weighted Average Price) for each produced fluid
        self._get_wap_price()

        # Validate sunk cost, pre-onstream, and post-onstream objects
        self._get_cost_objects_validation()

        # Calculate pre tax expenditures
        self._get_expenditures_pre_tax(
            year_inflation=year_inflation,
            inflation_rate=inflation_rate,
            inflation_rate_applied_to=inflation_rate_applied_to,
        )

        # Calculate indirect taxes
        self._get_indirect_taxes(tax_rate=vat_rate)

        # Calculate post tax expenditures
        self._get_expenditures_post_tax()

        # Other revenue
        self._get_other_revenue(
            sulfur_revenue=sulfur_revenue,
            electricity_revenue=electricity_revenue,
            co2_revenue=co2_revenue,
        )

        oil_capital_expenditures_pre_tax = self._require_array(
            self._oil_capital_expenditures_pre_tax,
            "_oil_capital_expenditures_pre_tax",
        )
        oil_intangible_expenditures_pre_tax = self._require_array(
            self._oil_intangible_expenditures_pre_tax,
            "_oil_intangible_expenditures_pre_tax",
        )
        oil_opex_expenditures_pre_tax = self._require_array(
            self._oil_opex_expenditures_pre_tax,
            "_oil_opex_expenditures_pre_tax",
        )
        oil_asr_expenditures_pre_tax = self._require_array(
            self._oil_asr_expenditures_pre_tax,
            "_oil_asr_expenditures_pre_tax",
        )
        oil_lbt_expenditures_pre_tax = self._require_array(
            self._oil_lbt_expenditures_pre_tax,
            "_oil_lbt_expenditures_pre_tax",
        )
        oil_cost_of_sales_expenditures_pre_tax = self._require_array(
            self._oil_cost_of_sales_expenditures_pre_tax,
            "_oil_cost_of_sales_expenditures_pre_tax",
        )

        # Total OIL pre-tax expenditures
        self._oil_total_expenditures_pre_tax = (
            oil_capital_expenditures_pre_tax
            + oil_intangible_expenditures_pre_tax
            + oil_opex_expenditures_pre_tax
            + oil_asr_expenditures_pre_tax
            + oil_lbt_expenditures_pre_tax
            + oil_cost_of_sales_expenditures_pre_tax
        )

        gas_capital_expenditures_pre_tax = self._require_array(
            self._gas_capital_expenditures_pre_tax,
            "_gas_capital_expenditures_pre_tax",
        )
        gas_intangible_expenditures_pre_tax = self._require_array(
            self._gas_intangible_expenditures_pre_tax,
            "_gas_intangible_expenditures_pre_tax",
        )
        gas_opex_expenditures_pre_tax = self._require_array(
            self._gas_opex_expenditures_pre_tax,
            "_gas_opex_expenditures_pre_tax",
        )
        gas_asr_expenditures_pre_tax = self._require_array(
            self._gas_asr_expenditures_pre_tax,
            "_gas_asr_expenditures_pre_tax",
        )
        gas_lbt_expenditures_pre_tax = self._require_array(
            self._gas_lbt_expenditures_pre_tax,
            "_gas_lbt_expenditures_pre_tax",
        )
        gas_cost_of_sales_expenditures_pre_tax = self._require_array(
            self._gas_cost_of_sales_expenditures_pre_tax,
            "_gas_cost_of_sales_expenditures_pre_tax",
        )

        # Total GAS pre-tax expenditures
        self._gas_total_expenditures_pre_tax = (
            gas_capital_expenditures_pre_tax
            + gas_intangible_expenditures_pre_tax
            + gas_opex_expenditures_pre_tax
            + gas_asr_expenditures_pre_tax
            + gas_lbt_expenditures_pre_tax
            + gas_cost_of_sales_expenditures_pre_tax
        )

        oil_capital_indirect_tax = self._require_array(
            self._oil_capital_indirect_tax, "_oil_capital_indirect_tax"
        )
        oil_intangible_indirect_tax = self._require_array(
            self._oil_intangible_indirect_tax, "_oil_intangible_indirect_tax"
        )
        oil_opex_indirect_tax = self._require_array(
            self._oil_opex_indirect_tax, "_oil_opex_indirect_tax"
        )
        oil_asr_indirect_tax = self._require_array(
            self._oil_asr_indirect_tax, "_oil_asr_indirect_tax"
        )
        oil_lbt_indirect_tax = self._require_array(
            self._oil_lbt_indirect_tax, "_oil_lbt_indirect_tax"
        )
        oil_cost_of_sales_indirect_tax = self._require_array(
            self._oil_cost_of_sales_indirect_tax,
            "_oil_cost_of_sales_indirect_tax",
        )

        # Total OIL indirect taxes
        self._oil_total_indirect_tax = (
            oil_capital_indirect_tax
            + oil_intangible_indirect_tax
            + oil_opex_indirect_tax
            + oil_asr_indirect_tax
            + oil_lbt_indirect_tax
            + oil_cost_of_sales_indirect_tax
        )

        gas_capital_indirect_tax = self._require_array(
            self._gas_capital_indirect_tax, "_gas_capital_indirect_tax"
        )
        gas_intangible_indirect_tax = self._require_array(
            self._gas_intangible_indirect_tax, "_gas_intangible_indirect_tax"
        )
        gas_opex_indirect_tax = self._require_array(
            self._gas_opex_indirect_tax, "_gas_opex_indirect_tax"
        )
        gas_asr_indirect_tax = self._require_array(
            self._gas_asr_indirect_tax, "_gas_asr_indirect_tax"
        )
        gas_lbt_indirect_tax = self._require_array(
            self._gas_lbt_indirect_tax, "_gas_lbt_indirect_tax"
        )
        gas_cost_of_sales_indirect_tax = self._require_array(
            self._gas_cost_of_sales_indirect_tax,
            "_gas_cost_of_sales_indirect_tax",
        )

        # Total GAS indirect taxes
        self._gas_total_indirect_tax = (
            gas_capital_indirect_tax
            + gas_intangible_indirect_tax
            + gas_opex_indirect_tax
            + gas_asr_indirect_tax
            + gas_lbt_indirect_tax
            + gas_cost_of_sales_indirect_tax
        )

        oil_capital_expenditures_post_tax = self._require_array(
            self._oil_capital_expenditures_post_tax,
            "_oil_capital_expenditures_post_tax",
        )
        oil_intangible_expenditures_post_tax = self._require_array(
            self._oil_intangible_expenditures_post_tax,
            "_oil_intangible_expenditures_post_tax",
        )
        oil_opex_expenditures_post_tax = self._require_array(
            self._oil_opex_expenditures_post_tax,
            "_oil_opex_expenditures_post_tax",
        )
        oil_asr_expenditures_post_tax = self._require_array(
            self._oil_asr_expenditures_post_tax,
            "_oil_asr_expenditures_post_tax",
        )
        oil_lbt_expenditures_post_tax = self._require_array(
            self._oil_lbt_expenditures_post_tax,
            "_oil_lbt_expenditures_post_tax",
        )
        oil_cost_of_sales_expenditures_post_tax = self._require_array(
            self._oil_cost_of_sales_expenditures_post_tax,
            "_oil_cost_of_sales_expenditures_post_tax",
        )

        # Total OIL post-tax expenditures
        self._oil_total_expenditures_post_tax = (
            oil_capital_expenditures_post_tax
            + oil_intangible_expenditures_post_tax
            + oil_opex_expenditures_post_tax
            + oil_asr_expenditures_post_tax
            + oil_lbt_expenditures_post_tax
            + oil_cost_of_sales_expenditures_post_tax
        )

        gas_capital_expenditures_post_tax = self._require_array(
            self._gas_capital_expenditures_post_tax,
            "_gas_capital_expenditures_post_tax",
        )
        gas_intangible_expenditures_post_tax = self._require_array(
            self._gas_intangible_expenditures_post_tax,
            "_gas_intangible_expenditures_post_tax",
        )
        gas_opex_expenditures_post_tax = self._require_array(
            self._gas_opex_expenditures_post_tax,
            "_gas_opex_expenditures_post_tax",
        )
        gas_asr_expenditures_post_tax = self._require_array(
            self._gas_asr_expenditures_post_tax,
            "_gas_asr_expenditures_post_tax",
        )
        gas_lbt_expenditures_post_tax = self._require_array(
            self._gas_lbt_expenditures_post_tax,
            "_gas_lbt_expenditures_post_tax",
        )
        gas_cost_of_sales_expenditures_post_tax = self._require_array(
            self._gas_cost_of_sales_expenditures_post_tax,
            "_gas_cost_of_sales_expenditures_post_tax",
        )

        # Total GAS post-tax expenditures
        self._gas_total_expenditures_post_tax = (
            gas_capital_expenditures_post_tax
            + gas_intangible_expenditures_post_tax
            + gas_opex_expenditures_post_tax
            + gas_asr_expenditures_post_tax
            + gas_lbt_expenditures_post_tax
            + gas_cost_of_sales_expenditures_post_tax
        )

        # Prepare sunk costs and preonstream costs
        self._get_sunkcost_array()
        self._get_preonstream_array()
        self._get_postonstream_array()

        # Prepare capital, non-capital, and total investments
        self._get_investments()

        oil_sunk_cost = self._require_array(self._oil_sunk_cost, "_oil_sunk_cost")
        gas_sunk_cost = self._require_array(self._gas_sunk_cost, "_gas_sunk_cost")
        oil_preonstream = self._require_array(self._oil_preonstream, "_oil_preonstream")
        gas_preonstream = self._require_array(self._gas_preonstream, "_gas_preonstream")
        oil_total_expenditures_post_tax = self._require_array(
            self._oil_total_expenditures_post_tax,
            "_oil_total_expenditures_post_tax",
        )
        gas_total_expenditures_post_tax = self._require_array(
            self._gas_total_expenditures_post_tax,
            "_gas_total_expenditures_post_tax",
        )

        # Configure base cashflow for OIL and GAS
        self._oil_cashflow = self._oil_revenue - (
            oil_sunk_cost + oil_preonstream + oil_total_expenditures_post_tax
        )

        self._gas_cashflow = self._gas_revenue - (
            gas_sunk_cost + gas_preonstream + gas_total_expenditures_post_tax
        )

        # Prepare consolidated profiles
        self._get_consolidated_profiles()

        self._has_run = True

        # Display warning messages as pandas DataFrame
        # self.warnings_to_dataframe()

    def get_oil_table_columns(self) -> dict[str, TableArray]:
        """Construct oil table columns after ``run()``."""
        self._require_run()
        return _assemble_baseproject_table_columns(
            years=self.project_years,
            lifting=self._oil_lifting.get_lifting_rate_ghv_arr(),
            price=self._require_array(self._oil_wap_price, "_oil_wap_price"),
            revenue=self._oil_revenue,
            sulfur_lifting=self._sulfur_lifting.get_lifting_rate_ghv_arr(),
            sulfur_price=self._require_array(
                self._sulfur_wap_price, "_sulfur_wap_price"
            ),
            sulfur_revenue=self._sulfur_revenue,
            electricity_lifting=self._electricity_lifting.get_lifting_rate_ghv_arr(),
            electricity_price=self._require_array(
                self._electricity_wap_price, "_electricity_wap_price"
            ),
            electricity_revenue=self._electricity_revenue,
            co2_lifting=self._co2_lifting.get_lifting_rate_ghv_arr(),
            co2_price=self._require_array(self._co2_wap_price, "_co2_wap_price"),
            co2_revenue=self._co2_revenue,
            sunk_cost_depreciable=self._require_array(
                self._oil_depreciable_sunk_cost, "_oil_depreciable_sunk_cost"
            ),
            sunk_cost_non_depreciable=self._require_array(
                self._oil_non_depreciable_sunk_cost,
                "_oil_non_depreciable_sunk_cost",
            ),
            sunk_cost=self._require_array(self._oil_sunk_cost, "_oil_sunk_cost"),
            preonstream_depreciable=self._require_array(
                self._oil_depreciable_preonstream, "_oil_depreciable_preonstream"
            ),
            preonstream_non_depreciable=self._require_array(
                self._oil_non_depreciable_preonstream,
                "_oil_non_depreciable_preonstream",
            ),
            preonstream=self._require_array(self._oil_preonstream, "_oil_preonstream"),
            postonstream_depreciable=self._require_array(
                self._oil_depreciable_postonstream, "_oil_depreciable_postonstream"
            ),
            postonstream_non_depreciable=self._require_array(
                self._oil_non_depreciable_postonstream,
                "_oil_non_depreciable_postonstream",
            ),
            postonstream=self._require_array(
                self._oil_postonstream, "_oil_postonstream"
            ),
            capital_expenditures_pre_tax=self._require_array(
                self._oil_capital_expenditures_pre_tax,
                "_oil_capital_expenditures_pre_tax",
            ),
            intangible_expenditures_pre_tax=self._require_array(
                self._oil_intangible_expenditures_pre_tax,
                "_oil_intangible_expenditures_pre_tax",
            ),
            opex_expenditures_pre_tax=self._require_array(
                self._oil_opex_expenditures_pre_tax,
                "_oil_opex_expenditures_pre_tax",
            ),
            asr_expenditures_pre_tax=self._require_array(
                self._oil_asr_expenditures_pre_tax, "_oil_asr_expenditures_pre_tax"
            ),
            lbt_expenditures_pre_tax=self._require_array(
                self._oil_lbt_expenditures_pre_tax, "_oil_lbt_expenditures_pre_tax"
            ),
            cost_of_sales_expenditures_pre_tax=self._require_array(
                self._oil_cost_of_sales_expenditures_pre_tax,
                "_oil_cost_of_sales_expenditures_pre_tax",
            ),
            capital_indirect_tax=self._require_array(
                self._oil_capital_indirect_tax, "_oil_capital_indirect_tax"
            ),
            intangible_indirect_tax=self._require_array(
                self._oil_intangible_indirect_tax, "_oil_intangible_indirect_tax"
            ),
            opex_indirect_tax=self._require_array(
                self._oil_opex_indirect_tax, "_oil_opex_indirect_tax"
            ),
            asr_indirect_tax=self._require_array(
                self._oil_asr_indirect_tax, "_oil_asr_indirect_tax"
            ),
            lbt_indirect_tax=self._require_array(
                self._oil_lbt_indirect_tax, "_oil_lbt_indirect_tax"
            ),
            cost_of_sales_indirect_tax=self._require_array(
                self._oil_cost_of_sales_indirect_tax,
                "_oil_cost_of_sales_indirect_tax",
            ),
            capital_postonstream=self._require_array(
                self._oil_capital_expenditures_post_tax,
                "_oil_capital_expenditures_post_tax",
            ),
            intangible_postonstream=self._require_array(
                self._oil_intangible_expenditures_post_tax,
                "_oil_intangible_expenditures_post_tax",
            ),
            opex_postonstream=self._require_array(
                self._oil_opex_expenditures_post_tax,
                "_oil_opex_expenditures_post_tax",
            ),
            asr_postonstream=self._require_array(
                self._oil_asr_expenditures_post_tax, "_oil_asr_expenditures_post_tax"
            ),
            lbt_postonstream=self._require_array(
                self._oil_lbt_expenditures_post_tax, "_oil_lbt_expenditures_post_tax"
            ),
            cost_of_sales_postonstream=self._require_array(
                self._oil_cost_of_sales_expenditures_post_tax,
                "_oil_cost_of_sales_expenditures_post_tax",
            ),
            expenses_capital=self._require_array(self._oil_capital, "_oil_capital"),
            expenses_non_capital=self._require_array(
                self._oil_non_capital, "_oil_non_capital"
            ),
            expenses_total=self._require_array(
                self._oil_total_expenses, "_oil_total_expenses"
            ),
            cashflow=self._require_array(self._oil_cashflow, "_oil_cashflow"),
        )

    def get_gas_table_columns(self) -> dict[str, TableArray]:
        """Construct gas table columns after ``run()``."""
        self._require_run()
        return _assemble_baseproject_table_columns(
            years=self.project_years,
            lifting=self._gas_lifting.get_lifting_rate_ghv_arr(),
            price=self._require_array(self._gas_wap_price, "_gas_wap_price"),
            revenue=self._gas_revenue,
            sulfur_lifting=self._sulfur_lifting.get_lifting_rate_ghv_arr(),
            sulfur_price=self._require_array(
                self._sulfur_wap_price, "_sulfur_wap_price"
            ),
            sulfur_revenue=self._sulfur_revenue,
            electricity_lifting=self._electricity_lifting.get_lifting_rate_ghv_arr(),
            electricity_price=self._require_array(
                self._electricity_wap_price, "_electricity_wap_price"
            ),
            electricity_revenue=self._electricity_revenue,
            co2_lifting=self._co2_lifting.get_lifting_rate_ghv_arr(),
            co2_price=self._require_array(self._co2_wap_price, "_co2_wap_price"),
            co2_revenue=self._co2_revenue,
            sunk_cost_depreciable=self._require_array(
                self._gas_depreciable_sunk_cost, "_gas_depreciable_sunk_cost"
            ),
            sunk_cost_non_depreciable=self._require_array(
                self._gas_non_depreciable_sunk_cost,
                "_gas_non_depreciable_sunk_cost",
            ),
            sunk_cost=self._require_array(self._gas_sunk_cost, "_gas_sunk_cost"),
            preonstream_depreciable=self._require_array(
                self._gas_depreciable_preonstream, "_gas_depreciable_preonstream"
            ),
            preonstream_non_depreciable=self._require_array(
                self._gas_non_depreciable_preonstream,
                "_gas_non_depreciable_preonstream",
            ),
            preonstream=self._require_array(self._gas_preonstream, "_gas_preonstream"),
            postonstream_depreciable=self._require_array(
                self._gas_depreciable_postonstream, "_gas_depreciable_postonstream"
            ),
            postonstream_non_depreciable=self._require_array(
                self._gas_non_depreciable_postonstream,
                "_gas_non_depreciable_postonstream",
            ),
            postonstream=self._require_array(
                self._gas_postonstream, "_gas_postonstream"
            ),
            capital_expenditures_pre_tax=self._require_array(
                self._gas_capital_expenditures_pre_tax,
                "_gas_capital_expenditures_pre_tax",
            ),
            intangible_expenditures_pre_tax=self._require_array(
                self._gas_intangible_expenditures_pre_tax,
                "_gas_intangible_expenditures_pre_tax",
            ),
            opex_expenditures_pre_tax=self._require_array(
                self._gas_opex_expenditures_pre_tax,
                "_gas_opex_expenditures_pre_tax",
            ),
            asr_expenditures_pre_tax=self._require_array(
                self._gas_asr_expenditures_pre_tax, "_gas_asr_expenditures_pre_tax"
            ),
            lbt_expenditures_pre_tax=self._require_array(
                self._gas_lbt_expenditures_pre_tax, "_gas_lbt_expenditures_pre_tax"
            ),
            cost_of_sales_expenditures_pre_tax=self._require_array(
                self._gas_cost_of_sales_expenditures_pre_tax,
                "_gas_cost_of_sales_expenditures_pre_tax",
            ),
            capital_indirect_tax=self._require_array(
                self._gas_capital_indirect_tax, "_gas_capital_indirect_tax"
            ),
            intangible_indirect_tax=self._require_array(
                self._gas_intangible_indirect_tax, "_gas_intangible_indirect_tax"
            ),
            opex_indirect_tax=self._require_array(
                self._gas_opex_indirect_tax, "_gas_opex_indirect_tax"
            ),
            asr_indirect_tax=self._require_array(
                self._gas_asr_indirect_tax, "_gas_asr_indirect_tax"
            ),
            lbt_indirect_tax=self._require_array(
                self._gas_lbt_indirect_tax, "_gas_lbt_indirect_tax"
            ),
            cost_of_sales_indirect_tax=self._require_array(
                self._gas_cost_of_sales_indirect_tax,
                "_gas_cost_of_sales_indirect_tax",
            ),
            capital_postonstream=self._require_array(
                self._gas_capital_expenditures_post_tax,
                "_gas_capital_expenditures_post_tax",
            ),
            intangible_postonstream=self._require_array(
                self._gas_intangible_expenditures_post_tax,
                "_gas_intangible_expenditures_post_tax",
            ),
            opex_postonstream=self._require_array(
                self._gas_opex_expenditures_post_tax,
                "_gas_opex_expenditures_post_tax",
            ),
            asr_postonstream=self._require_array(
                self._gas_asr_expenditures_post_tax, "_gas_asr_expenditures_post_tax"
            ),
            lbt_postonstream=self._require_array(
                self._gas_lbt_expenditures_post_tax, "_gas_lbt_expenditures_post_tax"
            ),
            cost_of_sales_postonstream=self._require_array(
                self._gas_cost_of_sales_expenditures_post_tax,
                "_gas_cost_of_sales_expenditures_post_tax",
            ),
            expenses_capital=self._require_array(self._gas_capital, "_gas_capital"),
            expenses_non_capital=self._require_array(
                self._gas_non_capital, "_gas_non_capital"
            ),
            expenses_total=self._require_array(
                self._gas_total_expenses, "_gas_total_expenses"
            ),
            cashflow=self._require_array(self._gas_cashflow, "_gas_cashflow"),
        )

    def get_consolidated_table_columns(self) -> dict[str, TableArray]:
        """Construct consolidated table columns after ``run()``."""
        self._require_run()
        return _assemble_baseproject_table_columns(
            years=self.project_years,
            lifting=self._require_array(
                self._consolidated_lifting, "_consolidated_lifting"
            ),
            price=self._require_array(
                self._consolidated_wap_price, "_consolidated_wap_price"
            ),
            revenue=self._require_array(
                self._consolidated_revenue, "_consolidated_revenue"
            ),
            sulfur_lifting=self._sulfur_lifting.get_lifting_rate_ghv_arr(),
            sulfur_price=self._require_array(
                self._sulfur_wap_price, "_sulfur_wap_price"
            ),
            sulfur_revenue=self._sulfur_revenue,
            electricity_lifting=self._electricity_lifting.get_lifting_rate_ghv_arr(),
            electricity_price=self._require_array(
                self._electricity_wap_price, "_electricity_wap_price"
            ),
            electricity_revenue=self._electricity_revenue,
            co2_lifting=self._co2_lifting.get_lifting_rate_ghv_arr(),
            co2_price=self._require_array(self._co2_wap_price, "_co2_wap_price"),
            co2_revenue=self._co2_revenue,
            sunk_cost_depreciable=self._require_array(
                self._consolidated_depreciable_sunk_cost,
                "_consolidated_depreciable_sunk_cost",
            ),
            sunk_cost_non_depreciable=self._require_array(
                self._consolidated_non_depreciable_sunk_cost,
                "_consolidated_non_depreciable_sunk_cost",
            ),
            sunk_cost=self._require_array(
                self._consolidated_sunk_cost, "_consolidated_sunk_cost"
            ),
            preonstream_depreciable=self._require_array(
                self._consolidated_depreciable_preonstream,
                "_consolidated_depreciable_preonstream",
            ),
            preonstream_non_depreciable=self._require_array(
                self._consolidated_non_depreciable_preonstream,
                "_consolidated_non_depreciable_preonstream",
            ),
            preonstream=self._require_array(
                self._consolidated_preonstream, "_consolidated_preonstream"
            ),
            postonstream_depreciable=self._require_array(
                self._consolidated_depreciable_postonstream,
                "_consolidated_depreciable_postonstream",
            ),
            postonstream_non_depreciable=self._require_array(
                self._consolidated_non_depreciable_postonstream,
                "_consolidated_non_depreciable_postonstream",
            ),
            postonstream=self._require_array(
                self._consolidated_postonstream, "_consolidated_postonstream"
            ),
            capital_expenditures_pre_tax=self._require_array(
                self._consolidated_capital_expenditures_pre_tax,
                "_consolidated_capital_expenditures_pre_tax",
            ),
            intangible_expenditures_pre_tax=self._require_array(
                self._consolidated_intangible_expenditures_pre_tax,
                "_consolidated_intangible_expenditures_pre_tax",
            ),
            opex_expenditures_pre_tax=self._require_array(
                self._consolidated_opex_expenditures_pre_tax,
                "_consolidated_opex_expenditures_pre_tax",
            ),
            asr_expenditures_pre_tax=self._require_array(
                self._consolidated_asr_expenditures_pre_tax,
                "_consolidated_asr_expenditures_pre_tax",
            ),
            lbt_expenditures_pre_tax=self._require_array(
                self._consolidated_lbt_expenditures_pre_tax,
                "_consolidated_lbt_expenditures_pre_tax",
            ),
            cost_of_sales_expenditures_pre_tax=self._require_array(
                self._consolidated_cost_of_sales_expenditures_pre_tax,
                "_consolidated_cost_of_sales_expenditures_pre_tax",
            ),
            capital_indirect_tax=self._require_array(
                self._consolidated_capital_indirect_tax,
                "_consolidated_capital_indirect_tax",
            ),
            intangible_indirect_tax=self._require_array(
                self._consolidated_intangible_indirect_tax,
                "_consolidated_intangible_indirect_tax",
            ),
            opex_indirect_tax=self._require_array(
                self._consolidated_opex_indirect_tax,
                "_consolidated_opex_indirect_tax",
            ),
            asr_indirect_tax=self._require_array(
                self._consolidated_asr_indirect_tax,
                "_consolidated_asr_indirect_tax",
            ),
            lbt_indirect_tax=self._require_array(
                self._consolidated_lbt_indirect_tax,
                "_consolidated_lbt_indirect_tax",
            ),
            cost_of_sales_indirect_tax=self._require_array(
                self._consolidated_cost_of_sales_indirect_tax,
                "_consolidated_cost_of_sales_indirect_tax",
            ),
            capital_postonstream=self._require_array(
                self._consolidated_capital_expenditures_post_tax,
                "_consolidated_capital_expenditures_post_tax",
            ),
            intangible_postonstream=self._require_array(
                self._consolidated_intangible_expenditures_post_tax,
                "_consolidated_intangible_expenditures_post_tax",
            ),
            opex_postonstream=self._require_array(
                self._consolidated_opex_expenditures_post_tax,
                "_consolidated_opex_expenditures_post_tax",
            ),
            asr_postonstream=self._require_array(
                self._consolidated_asr_expenditures_post_tax,
                "_consolidated_asr_expenditures_post_tax",
            ),
            lbt_postonstream=self._require_array(
                self._consolidated_lbt_expenditures_post_tax,
                "_consolidated_lbt_expenditures_post_tax",
            ),
            cost_of_sales_postonstream=self._require_array(
                self._consolidated_cost_of_sales_expenditures_post_tax,
                "_consolidated_cost_of_sales_expenditures_post_tax",
            ),
            expenses_capital=self._require_array(
                self._consolidated_capital, "_consolidated_capital"
            ),
            expenses_non_capital=self._require_array(
                self._consolidated_non_capital, "_consolidated_non_capital"
            ),
            expenses_total=self._require_array(
                self._consolidated_total_expenses, "_consolidated_total_expenses"
            ),
            cashflow=self._require_array(
                self._consolidated_cashflow, "_consolidated_cashflow"
            ),
        )

    @staticmethod
    def _calc_division(
        numerator: float | np.float64,
        denominator: float | np.float64,
        default: float = 0.0,
    ) -> float | np.float64:
        """
        Safely perform division with a zero-denominator check.

        Returns the result of ``numerator / denominator`` if the denominator
        is nonzero. Otherwise, returns the specified default value.

        Parameters
        ----------
        numerator : float
            The numerator of the division.
        denominator : float
            The denominator of the division. If zero, the function returns `default`.
        default : float, default=0.0
            The value returned when `denominator` is zero.

        Returns
        -------
        float
            The division result if `denominator` is nonzero; otherwise, `default`.
        """
        return (numerator / denominator) if denominator != 0 else default

    def _check_capital_pis_years_before_onstream(
        self,
        obj_capital: CapitalCost,
        obj_name: str,
        is_amortization: bool,
        is_strict: bool,
        onstream_year: int,
    ) -> None:
        """
        Validate capital PIS years relative to the onstream year.

        Detects PIS years in a ``CapitalCost`` object that occur before the
        specified onstream year. Behavior depends on strictness and cost
        treatment (amortization vs depreciation).

        - Strict mode → raise exception
        - Non-strict mode → record warning with context-specific messaging

        Parameters
        ----------
        obj_capital : CapitalCost
            Capital cost object containing cost values and PIS years.
        obj_name : str
            Descriptive name used in error/warning messages.
        is_amortization : bool
            Controls warning message semantics (amortization vs depreciation).
        is_strict : bool
            If True, invalid PIS years raise an exception.
        onstream_year : int
            Minimum allowable PIS year.

        Raises
        ------
        BaseProjectException
            If invalid PIS years are found and ``is_strict`` is True.

        Notes
        -----
        - Only PIS years strictly earlier than ``onstream_year`` are flagged.
        - In non-strict mode, warnings are appended to ``self.warning_messages``.
        """

        # Extract costs of a CapitalCost object and their corresponding PIS years
        costs = _require_float_array(obj_capital.cost, "cost")
        pis_yrs = _require_int_array(obj_capital.pis_year, "pis_year")

        # Identify PIS years occurring before the onstream year through the
        # shared predicate. Only entries carrying cost are considered: a
        # zero-cost entry is never depreciated or amortized.
        indices = pis_before_onstream_indices(pis_yrs, costs, onstream_year)

        # Exit early if all PIS years are NOT before onstream year
        if not indices:
            return

        # Collect invalid costs and their associated PIS years
        invalid_costs = costs[indices]
        invalid_pis_yrs = pis_yrs[indices]

        # Pair invalid PIS years with their corresponding costs
        invalid = [
            f"{int(invalid_pis_yrs.item(index))}: {float(invalid_costs.item(index))}"
            for index in range(len(invalid_pis_yrs))
        ]

        # Specify messages
        msg_error = (
            f"Cannot have {obj_name!r} PIS years ({invalid}) before "
            + f"onstream year ({onstream_year})."
        )

        if is_amortization:
            msg_warning = (
                f"Found {obj_name!r} PIS years ({invalid}) before onstream year "
                + f"({onstream_year}). Please note that PSCnomics will ALWAYS apply amortization "
                + "charge AT ONSTREAM YEAR, namely after lifting occurs. Although PSCnomics will "
                + "produce correct calculation when PIS years occur before onstream year, "
                + "for the sake of being consistent with the underlying concept, we suggest "
                + f"the User to reset PIS years configuration ({invalid}) so that they "
                + f"prevail exactly at the onstream year ({onstream_year})."
            )

        else:
            msg_warning = (
                f"Found {obj_name!r} PIS years ({invalid}) before onstream year "
                + f"({onstream_year}). PSCnomics will ALWAYS charge {obj_name!r} depreciations "
                + "at their corresponding PIS years. You may want to reset the configuration "
                + f"of the following PIS years ({invalid}) so that they prevail at the onstream "
                + f"year ({onstream_year})."
            )

        # Strict mode: raise an error and stop execution
        if is_strict:
            raise BaseProjectInputError(msg_error, code="PIS_YEAR_BEFORE_ONSTREAM")

        # Loose mode: record a warning message and continue execution
        else:
            self.warning_messages.append((ContractType.BASE_PROJECT.value, msg_warning))

    def _check_capital_pis_years_after_onstream(
        self,
        obj_capital: CapitalCost,
        obj_name: str,
        is_strict: bool,
        onstream_year: int,
    ) -> None:
        """
        Validate capital PIS years occurring after the onstream year.

        Detects PIS years in a ``CapitalCost`` object that fall after the
        specified onstream year. In strict mode, violations raise an exception;
        otherwise, a warning is recorded describing the potential impact on
        amortization behavior.

        Parameters
        ----------
        obj_capital : CapitalCost
            Capital cost object containing cost values and PIS years.
        obj_name : str
            Descriptive name used in error/warning messages.
        is_strict : bool
            If True, invalid PIS years raise an exception.
        onstream_year : int
            Reference onstream year for validation.

        Raises
        ------
        BaseProjectException
            If invalid PIS years are found and ``is_strict`` is True.

        Notes
        -----
        - Only PIS years strictly greater than ``onstream_year`` are flagged.
        - In non-strict mode, warnings are appended to ``self.warning_messages``.
        """

        # Extract costs of a CapitalCost object and their corresponding PIS years
        costs = _require_float_array(obj_capital.cost, "cost")
        pis_yrs = _require_int_array(obj_capital.pis_year, "pis_year")

        # Identify PIS years occurring after the onstream year through the
        # shared predicate.
        indices = pis_after_onstream_indices(pis_yrs, onstream_year)

        # Exit early if all PIS years are NOT after onstream year
        if not indices:
            return

        # Collect invalid costs and their associated PIS years
        invalid_costs = costs[indices]
        invalid_pis_yrs = pis_yrs[indices]

        # Pair invalid PIS years with their corresponding costs
        invalid = [
            f"{int(invalid_pis_yrs.item(index))}: {float(invalid_costs.item(index))}"
            for index in range(len(invalid_pis_yrs))
        ]

        # Specify messages
        msg_error = (
            f"Cannot have {obj_name!r} PIS years ({invalid}) after "
            + f"onstream year ({onstream_year})."
        )

        msg_warning = (
            f"Found {obj_name!r} PIS years ({invalid}) after onstream year ({onstream_year}). "
            + "Setting PIS years after onstream year will result in different production to "
            + "reserve ratios, which will reflect on different amortization charge. You may want "
            + f"to consider shifting the PIS years ({invalid}) to onstream year ({onstream_year})."
        )

        # Strict mode: raise an error and stop execution
        if is_strict:
            raise BaseProjectInputError(msg_error, code="PIS_YEAR_AFTER_ONSTREAM")

        # Loose mode: record a warning message and continue execution
        else:
            self.warning_messages.append((ContractType.BASE_PROJECT.value, msg_warning))

    def warnings_to_dataframe(self) -> pd.DataFrame:
        """
        Convert warning messages from a list to a pandas DataFrame.
        """

        messages = pd.DataFrame(
            {
                "No.": np.arange(1, len(self.warning_messages) + 1),
                "Category": [c for c, _ in self.warning_messages],
                "Message": [m for _, m in self.warning_messages],
            }
        )

        return messages

    def _require_run(self) -> None:
        if not self._has_run:
            raise BaseProjectSummaryException(
                "Call run() successfully before requesting a project summary."
            )

    def _post_run_common_summary_profile(self) -> PostRunSummaryProfile:
        """Validate and snapshot reporting data shared by all contract schemes."""
        self._require_run()
        return PostRunSummaryProfile(
            contract_type="base_project",
            project_years=self.project_years,
            oil_lifting=self._oil_lifting,
            gas_lifting=self._gas_lifting,
            sulfur_lifting=self._sulfur_lifting,
            electricity_lifting=self._electricity_lifting,
            co2_lifting=self._co2_lifting,
            oil_revenue=self._require_array(self._oil_revenue, "_oil_revenue"),
            gas_revenue=self._require_array(self._gas_revenue, "_gas_revenue"),
            sulfur_revenue=self._require_array(self._sulfur_revenue, "_sulfur_revenue"),
            electricity_revenue=self._require_array(
                self._electricity_revenue, "_electricity_revenue"
            ),
            co2_revenue=self._require_array(self._co2_revenue, "_co2_revenue"),
            gas_wap_price=self._require_array(self._gas_wap_price, "_gas_wap_price"),
            oil_capital_expenditures_pre_tax=self._require_array(
                self._oil_capital_expenditures_pre_tax,
                "_oil_capital_expenditures_pre_tax",
            ),
            oil_intangible_expenditures_pre_tax=self._require_array(
                self._oil_intangible_expenditures_pre_tax,
                "_oil_intangible_expenditures_pre_tax",
            ),
            oil_opex_expenditures_pre_tax=self._require_array(
                self._oil_opex_expenditures_pre_tax, "_oil_opex_expenditures_pre_tax"
            ),
            oil_asr_expenditures_pre_tax=self._require_array(
                self._oil_asr_expenditures_pre_tax, "_oil_asr_expenditures_pre_tax"
            ),
            oil_lbt_expenditures_pre_tax=self._require_array(
                self._oil_lbt_expenditures_pre_tax, "_oil_lbt_expenditures_pre_tax"
            ),
            oil_cost_of_sales_expenditures_pre_tax=self._require_array(
                self._oil_cost_of_sales_expenditures_pre_tax,
                "_oil_cost_of_sales_expenditures_pre_tax",
            ),
            gas_capital_expenditures_pre_tax=self._require_array(
                self._gas_capital_expenditures_pre_tax,
                "_gas_capital_expenditures_pre_tax",
            ),
            gas_intangible_expenditures_pre_tax=self._require_array(
                self._gas_intangible_expenditures_pre_tax,
                "_gas_intangible_expenditures_pre_tax",
            ),
            gas_opex_expenditures_pre_tax=self._require_array(
                self._gas_opex_expenditures_pre_tax, "_gas_opex_expenditures_pre_tax"
            ),
            gas_asr_expenditures_pre_tax=self._require_array(
                self._gas_asr_expenditures_pre_tax, "_gas_asr_expenditures_pre_tax"
            ),
            gas_lbt_expenditures_pre_tax=self._require_array(
                self._gas_lbt_expenditures_pre_tax, "_gas_lbt_expenditures_pre_tax"
            ),
            gas_cost_of_sales_expenditures_pre_tax=self._require_array(
                self._gas_cost_of_sales_expenditures_pre_tax,
                "_gas_cost_of_sales_expenditures_pre_tax",
            ),
            oil_capital_expenditures_post_tax=self._require_array(
                self._oil_capital_expenditures_post_tax,
                "_oil_capital_expenditures_post_tax",
            ),
            oil_intangible_expenditures_post_tax=self._require_array(
                self._oil_intangible_expenditures_post_tax,
                "_oil_intangible_expenditures_post_tax",
            ),
            oil_opex_expenditures_post_tax=self._require_array(
                self._oil_opex_expenditures_post_tax, "_oil_opex_expenditures_post_tax"
            ),
            oil_asr_expenditures_post_tax=self._require_array(
                self._oil_asr_expenditures_post_tax, "_oil_asr_expenditures_post_tax"
            ),
            oil_lbt_expenditures_post_tax=self._require_array(
                self._oil_lbt_expenditures_post_tax, "_oil_lbt_expenditures_post_tax"
            ),
            oil_cost_of_sales_expenditures_post_tax=self._require_array(
                self._oil_cost_of_sales_expenditures_post_tax,
                "_oil_cost_of_sales_expenditures_post_tax",
            ),
            gas_capital_expenditures_post_tax=self._require_array(
                self._gas_capital_expenditures_post_tax,
                "_gas_capital_expenditures_post_tax",
            ),
            gas_intangible_expenditures_post_tax=self._require_array(
                self._gas_intangible_expenditures_post_tax,
                "_gas_intangible_expenditures_post_tax",
            ),
            gas_opex_expenditures_post_tax=self._require_array(
                self._gas_opex_expenditures_post_tax, "_gas_opex_expenditures_post_tax"
            ),
            gas_asr_expenditures_post_tax=self._require_array(
                self._gas_asr_expenditures_post_tax, "_gas_asr_expenditures_post_tax"
            ),
            gas_lbt_expenditures_post_tax=self._require_array(
                self._gas_lbt_expenditures_post_tax, "_gas_lbt_expenditures_post_tax"
            ),
            gas_cost_of_sales_expenditures_post_tax=self._require_array(
                self._gas_cost_of_sales_expenditures_post_tax,
                "_gas_cost_of_sales_expenditures_post_tax",
            ),
            oil_capital_indirect_tax=self._require_array(
                self._oil_capital_indirect_tax, "_oil_capital_indirect_tax"
            ),
            oil_intangible_indirect_tax=self._require_array(
                self._oil_intangible_indirect_tax, "_oil_intangible_indirect_tax"
            ),
            oil_opex_indirect_tax=self._require_array(
                self._oil_opex_indirect_tax, "_oil_opex_indirect_tax"
            ),
            oil_asr_indirect_tax=self._require_array(
                self._oil_asr_indirect_tax, "_oil_asr_indirect_tax"
            ),
            oil_lbt_indirect_tax=self._require_array(
                self._oil_lbt_indirect_tax, "_oil_lbt_indirect_tax"
            ),
            oil_cost_of_sales_indirect_tax=self._require_array(
                self._oil_cost_of_sales_indirect_tax, "_oil_cost_of_sales_indirect_tax"
            ),
            gas_capital_indirect_tax=self._require_array(
                self._gas_capital_indirect_tax, "_gas_capital_indirect_tax"
            ),
            gas_intangible_indirect_tax=self._require_array(
                self._gas_intangible_indirect_tax, "_gas_intangible_indirect_tax"
            ),
            gas_opex_indirect_tax=self._require_array(
                self._gas_opex_indirect_tax, "_gas_opex_indirect_tax"
            ),
            gas_asr_indirect_tax=self._require_array(
                self._gas_asr_indirect_tax, "_gas_asr_indirect_tax"
            ),
            gas_lbt_indirect_tax=self._require_array(
                self._gas_lbt_indirect_tax, "_gas_lbt_indirect_tax"
            ),
            gas_cost_of_sales_indirect_tax=self._require_array(
                self._gas_cost_of_sales_indirect_tax, "_gas_cost_of_sales_indirect_tax"
            ),
            oil_sunk_cost=self._require_array(self._oil_sunk_cost, "_oil_sunk_cost"),
            gas_sunk_cost=self._require_array(self._gas_sunk_cost, "_gas_sunk_cost"),
            oil_cashflow=(
                None
                if self._oil_cashflow is None
                else self._require_array(self._oil_cashflow, "_oil_cashflow")
            ),
            gas_cashflow=(
                None
                if self._gas_cashflow is None
                else self._require_array(self._gas_cashflow, "_gas_cashflow")
            ),
            oil_total_indirect_tax=self._require_array(
                self._oil_total_indirect_tax, "_oil_total_indirect_tax"
            ),
            gas_total_indirect_tax=self._require_array(
                self._gas_total_indirect_tax, "_gas_total_indirect_tax"
            ),
            consolidated_revenue=self._require_array(
                self._consolidated_revenue, "_consolidated_revenue"
            ),
            consolidated_sunk_cost=self._require_array(
                self._consolidated_sunk_cost, "_consolidated_sunk_cost"
            ),
            consolidated_cashflow=self._require_array(
                self._consolidated_cashflow, "_consolidated_cashflow"
            ),
        )

    def get_post_run_summary_profile(self) -> PostRunSummaryProfile:
        """Return named reporting data from a completed contract run."""
        return self._post_run_common_summary_profile()

    def get_summary(
        self,
        discount_rate: float = 0.1,
        npv_mode: NPVSelection = NPVSelection.NPV_SKK_REAL_TERMS,
        discounting_mode: DiscountingMode = DiscountingMode.END_YEAR,
        discount_rate_start_year: int | None = None,
        inflation_rate: float = 0.0,
        profitability_discounted: bool = False,
    ) -> dict[str, float | np.float64]:
        """
        Compute the economic summary of a petroleum project under various NPV modes.

        This method calculates key economic indicators such as Net Present Value (NPV),
        Internal Rate of Return (IRR), Pay-Out Time (POT), Profitability Index (PI),
        as well as gross revenues, costs, and contractor shares for both oil and gas.
        It supports multiple discounting approaches (real, nominal, SKK, or point-forward)
        and allows selection of whether profitability is calculated using discounted
        or undiscounted investment.

        Parameters
        ----------
        discount_rate : float, default=0.1
            Annual discount rate (as a decimal) used for NPV calculation.

        npv_mode : NPVSelection, default=NPVSelection.NPV_SKK_REAL_TERMS
            Method used for NPV calculation. Options include:
            - ``NPV_SKK_REAL_TERMS`` : SKK-based NPV in real terms.
            - ``NPV_SKK_NOMINAL_TERMS`` : SKK-based NPV in nominal terms.
            - ``NPV_NOMINAL_TERMS`` : Conventional NPV in nominal terms.
            - ``NPV_REAL_TERMS`` : Conventional NPV in real terms.
            - ``NPV_POINT_FORWARD`` : Point-forward NPV where cashflows before
              the reference year are neglected.

        discounting_mode : DiscountingMode, default=DiscountingMode.END_YEAR
            Defines when discounting occurs. Typically ``START_YEAR`` or ``END_YEAR``.

        discount_rate_start_year : int or None, optional
            Year from which the discounting reference is applied. If ``None``,
            it defaults to the project start year. Must fall between project
            start and end years.

        inflation_rate : float or np.ndarray, default=0.0
            Inflation rate(s) applied when using real-term NPV calculations.
            Can be a scalar or an array corresponding to project years.

        profitability_discounted : bool, default=False
            If ``True``, the Profitability Index (PI) is calculated using discounted
            investment. Otherwise, undiscounted investment is used.

        Raises
        ------
        BaseProjectSummaryException
            If the discounting reference year is outside the project duration
            (i.e., before start year or after end year).

        Returns
        -------
        summary : dict of {str: float}
            A dictionary containing summarized economic indicators and project
            financial metrics.

        Notes
        -----
        The NPV and other economic indicators depend on the selected ``npv_mode``
        and discounting assumptions. When using the *point-forward* mode, only cashflows
        from the reference year onward are considered in NPV and net cashflow calculations.
        """

        self._require_run()
        gas_wap_price = self._require_array(self._gas_wap_price, "_gas_wap_price")
        oil_sunk_cost = self._require_array(self._oil_sunk_cost, "_oil_sunk_cost")
        gas_sunk_cost = self._require_array(self._gas_sunk_cost, "_gas_sunk_cost")
        oil_capital_expenditures_post_tax = self._require_array(
            self._oil_capital_expenditures_post_tax,
            "_oil_capital_expenditures_post_tax",
        )
        gas_capital_expenditures_post_tax = self._require_array(
            self._gas_capital_expenditures_post_tax,
            "_gas_capital_expenditures_post_tax",
        )
        oil_depreciable_preonstream = self._require_array(
            self._oil_depreciable_preonstream, "_oil_depreciable_preonstream"
        )
        gas_depreciable_preonstream = self._require_array(
            self._gas_depreciable_preonstream, "_gas_depreciable_preonstream"
        )
        oil_intangible_expenditures_post_tax = self._require_array(
            self._oil_intangible_expenditures_post_tax,
            "_oil_intangible_expenditures_post_tax",
        )
        gas_intangible_expenditures_post_tax = self._require_array(
            self._gas_intangible_expenditures_post_tax,
            "_gas_intangible_expenditures_post_tax",
        )
        oil_opex_expenditures_post_tax = self._require_array(
            self._oil_opex_expenditures_post_tax,
            "_oil_opex_expenditures_post_tax",
        )
        gas_opex_expenditures_post_tax = self._require_array(
            self._gas_opex_expenditures_post_tax,
            "_gas_opex_expenditures_post_tax",
        )
        oil_asr_expenditures_post_tax = self._require_array(
            self._oil_asr_expenditures_post_tax,
            "_oil_asr_expenditures_post_tax",
        )
        gas_asr_expenditures_post_tax = self._require_array(
            self._gas_asr_expenditures_post_tax,
            "_gas_asr_expenditures_post_tax",
        )
        oil_lbt_expenditures_post_tax = self._require_array(
            self._oil_lbt_expenditures_post_tax,
            "_oil_lbt_expenditures_post_tax",
        )
        gas_lbt_expenditures_post_tax = self._require_array(
            self._gas_lbt_expenditures_post_tax,
            "_gas_lbt_expenditures_post_tax",
        )
        oil_total_indirect_tax = self._require_array(
            self._oil_total_indirect_tax, "_oil_total_indirect_tax"
        )
        gas_total_indirect_tax = self._require_array(
            self._gas_total_indirect_tax, "_gas_total_indirect_tax"
        )
        consolidated_cashflow = self._require_array(
            self._consolidated_cashflow, "_consolidated_cashflow"
        )
        consolidated_revenue = self._require_array(
            self._consolidated_revenue, "_consolidated_revenue"
        )
        oil_capital = self._require_array(self._oil_capital, "_oil_capital")
        gas_capital = self._require_array(self._gas_capital, "_gas_capital")

        # Prepare discount rate start year
        if discount_rate_start_year is None:
            discount_rate_start_year = self.start_date.year

        # Cannot have discount rate year before the start year of the project
        if discount_rate_start_year < self.start_date.year:
            raise BaseProjectSummaryException(
                f"The discounting reference year ({discount_rate_start_year}) "
                + f"is before start year of the project ({self.start_date.year})."
            )

        # Cannot have discount rate year after the end year of the project
        if discount_rate_start_year > self.end_date.year:
            raise BaseProjectSummaryException(
                f"The discounting reference year ({discount_rate_start_year}) "
                + f"is after the end year of the project ({self.end_date.year})."
            )

        # Prepare OIL lifting summary
        oil_lifting_ghv = self._oil_lifting.get_lifting_rate_ghv_arr()
        oil_lifting_ghv_sum = _sum_float(oil_lifting_ghv)
        oil_wap_sum = self._calc_division(
            numerator=_sum_float(self._oil_revenue),
            denominator=oil_lifting_ghv_sum,
        )

        # Prepare GAS lifting summary
        gas_lifting_ghv = self._gas_lifting.get_lifting_rate_ghv_arr()
        gas_lifting_ghv_sum = _sum_float(gas_lifting_ghv)
        gas_wap_sum = self._calc_division(
            numerator=_sum_float(gas_wap_price * gas_lifting_ghv),
            denominator=gas_lifting_ghv_sum,
        )

        # Prepare gross revenue summary
        oil_gross_revenue_sum = _sum_float(self._oil_revenue)
        gas_gross_revenue_sum = _sum_float(self._gas_revenue)
        total_gross_revenue_sum = oil_gross_revenue_sum + gas_gross_revenue_sum

        # Prepare sunk cost summary
        sunk_cost_sum = _sum_float(oil_sunk_cost + gas_sunk_cost)

        # Prepare preonstream costs
        preonstream_map = {
            "oil_intangible": self._oil_intangible_preonstream,
            "gas_intangible": self._gas_intangible_preonstream,
            "oil_opex": self._oil_opex_preonstream,
            "gas_opex": self._gas_opex_preonstream,
            "oil_asr": self._oil_asr_preonstream,
            "gas_asr": self._gas_asr_preonstream,
            "oil_lbt": self._oil_lbt_preonstream,
            "gas_lbt": self._gas_lbt_preonstream,
        }

        preonstream_costs = {
            key: val.expenditures_pre_tax() for key, val in preonstream_map.items()
        }

        # Prepare tangible and intangible cost summary
        tangible_cost = (
            oil_capital_expenditures_post_tax
            + gas_capital_expenditures_post_tax
            + oil_depreciable_preonstream
            + gas_depreciable_preonstream
        )

        intangible_cost = (
            oil_intangible_expenditures_post_tax
            + gas_intangible_expenditures_post_tax
            + preonstream_costs["oil_intangible"]
            + preonstream_costs["gas_intangible"]
        )
        investment_cashflow = (tangible_cost + intangible_cost).astype(
            np.float64, copy=False
        )

        tangible_sum = _sum_float(tangible_cost)
        intangible_sum = _sum_float(intangible_cost)
        investment_sum = tangible_sum + intangible_sum

        # Prepare OPEX summary
        opex_cost = (
            oil_opex_expenditures_post_tax
            + gas_opex_expenditures_post_tax
            + preonstream_costs["oil_opex"]
            + preonstream_costs["gas_opex"]
        )

        opex_sum = _sum_float(opex_cost)

        # Prepare ASR summary
        asr_cost = (
            oil_asr_expenditures_post_tax
            + gas_asr_expenditures_post_tax
            + preonstream_costs["oil_asr"]
            + preonstream_costs["gas_asr"]
        )

        asr_sum = _sum_float(asr_cost)

        # Prepare LBT summary
        lbt_cost = (
            oil_lbt_expenditures_post_tax
            + gas_lbt_expenditures_post_tax
            + preonstream_costs["oil_lbt"]
            + preonstream_costs["gas_lbt"]
        )

        lbt_sum = _sum_float(lbt_cost)

        # Prepare indirect taxes summary
        oil_indirect_tax_sum = _sum_float(oil_total_indirect_tax)
        gas_indirect_tax_sum = _sum_float(gas_total_indirect_tax)

        # Calculate IRR
        ctr_irr = irr(cashflow=consolidated_cashflow)

        # Calculate NPV
        # NPV method: SKK real terms
        if npv_mode == NPVSelection.NPV_SKK_REAL_TERMS:
            # Contractor NPV
            ctr_npv = npv_skk_real_terms(
                cashflow=consolidated_cashflow,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                discounting_mode=discounting_mode,
            )

            # Contractor investment NPV
            investment_npv = npv_skk_real_terms(
                cashflow=investment_cashflow,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                discounting_mode=discounting_mode,
            )

        # NPV method: SKK nominal terms
        elif npv_mode == NPVSelection.NPV_SKK_NOMINAL_TERMS:
            # Contractor NPV
            ctr_npv = npv_skk_nominal_terms(
                cashflow=consolidated_cashflow,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                discounting_mode=discounting_mode,
            )

            # Contractor investment NPV
            investment_npv = npv_skk_nominal_terms(
                cashflow=investment_cashflow,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                discounting_mode=discounting_mode,
            )

        # NPV method: nominal terms
        elif npv_mode == NPVSelection.NPV_NOMINAL_TERMS:
            # Contractor NPV
            ctr_npv = npv_nominal_terms(
                cashflow=consolidated_cashflow,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                discounting_mode=discounting_mode,
            )

            # Contractor investment NPV
            investment_npv = npv_nominal_terms(
                cashflow=investment_cashflow,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                discounting_mode=discounting_mode,
            )

        # NPV method: real terms
        elif npv_mode == NPVSelection.NPV_REAL_TERMS:
            # Contractor NPV
            ctr_npv = npv_real_terms(
                cashflow=consolidated_cashflow,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                inflation_rate=inflation_rate,
                discounting_mode=discounting_mode,
            )

            # Contractor investment NPV
            investment_npv = npv_real_terms(
                cashflow=investment_cashflow,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                inflation_rate=inflation_rate,
                discounting_mode=discounting_mode,
            )

        # NPV method: point forward
        else:
            # Contractor NPV
            ctr_npv = npv_point_forward(
                cashflow=consolidated_cashflow,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                discounting_mode=discounting_mode,
            )

            # Contractor investment NPV
            investment_npv = npv_point_forward(
                cashflow=investment_cashflow,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                discounting_mode=discounting_mode,
            )

        # Contractor present value ratio to the investment NPV
        # Profitability Index is calculated using discounted investment
        if profitability_discounted:
            ctr_pv_ratio = self._calc_division(
                numerator=ctr_npv, denominator=investment_npv
            )

        # Profitability Index is calculated using undiscounted investment
        else:
            investment_pi = _sum_float(tangible_cost + intangible_cost)
            ctr_pv_ratio = self._calc_division(
                numerator=ctr_npv, denominator=investment_pi
            )

        ctr_pi = 1 + ctr_pv_ratio

        # Contractor POT
        ctr_pot = pot_psc(
            cashflow=consolidated_cashflow,
            cashflow_years=self.project_years,
            reference_year=discount_rate_start_year,
        )

        # Prepare contractor net share summary
        ctr_net_share_sum = (
            oil_gross_revenue_sum
            + gas_gross_revenue_sum
            - investment_sum
            - opex_sum
            - asr_sum
            - lbt_sum
        )
        ctr_net_share_over_gross_share = self._calc_division(
            numerator=ctr_net_share_sum,
            denominator=total_gross_revenue_sum,
        )

        # Prepare contractor net cashflow summary
        if npv_mode == NPVSelection.NPV_POINT_FORWARD:
            # Modify contractor net cashflow since the cashflow before discount rate
            # start year is neglected
            ref_year_arr = np.full_like(
                consolidated_cashflow, fill_value=discount_rate_start_year
            )
            cashflow_point_forward = np.where(
                self.project_years >= ref_year_arr,
                consolidated_cashflow,
                0,
            )
            gross_revenue_point_forward = np.where(
                self.project_years >= ref_year_arr,
                consolidated_revenue,
                0,
            )

            # Contractor net cashflow
            ctr_net_cashflow_sum = _sum_float(cashflow_point_forward)
            ctr_net_cashflow_over_gross_rev = self._calc_division(
                numerator=ctr_net_cashflow_sum,
                denominator=_sum_float(gross_revenue_point_forward),
            )

        else:
            ctr_net_cashflow_sum = ctr_net_share_sum
            ctr_net_cashflow_over_gross_rev = ctr_net_share_over_gross_share

        return {
            "lifting_oil": oil_lifting_ghv_sum,
            "oil_wap": oil_wap_sum,
            "lifting_gas": gas_lifting_ghv_sum,
            "gas_wap": gas_wap_sum,
            "gross_revenue": total_gross_revenue_sum,
            "gross_revenue_oil": oil_gross_revenue_sum,
            "gross_revenue_gas": gas_gross_revenue_sum,
            "investment": investment_sum,
            "oil_capex": oil_capital.sum(),
            "gas_capex": gas_capital.sum(),
            "sunk_cost": sunk_cost_sum,
            "tangible": tangible_sum,
            "intangible": intangible_sum,
            "opex_and_asr": opex_sum + asr_sum,
            "opex_asr_lbt": opex_sum + asr_sum + lbt_sum,
            "opex": opex_sum,
            "asr": asr_sum,
            "lbt": lbt_sum,
            "ctr_npv": ctr_npv,
            "ctr_irr": ctr_irr,
            "ctr_pot": ctr_pot,
            "ctr_pv_ratio": ctr_pv_ratio,
            "ctr_pi": ctr_pi,
            "ctr_gross_share": total_gross_revenue_sum,
            "ctr_net_share": ctr_net_share_sum,
            "ctr_net_share_over_gross_share": ctr_net_share_over_gross_share,
            "ctr_net_cashflow": ctr_net_cashflow_sum,
            "ctr_net_cashflow_over_gross_rev": ctr_net_cashflow_over_gross_rev,
            "total_indirect_taxes": oil_indirect_tax_sum + gas_indirect_tax_sum,
            "oil_indirect_taxes": oil_indirect_tax_sum,
            "gas_indirect_taxes": gas_indirect_tax_sum,
            # Zero values for the PSC terms
            "gov_gross_share": 0,
            "cost_recovery/deductible_cost": 0,
            "cost_recovery_over_gross_rev": 0,
            "unrec_cost": 0,
            "unrec_over_gross_rev": 0,
            "gov_ftp_share": 0,
            "gov_ddmo": 0,
            "gov_tax_income": 0,
            "gov_take": 0,
            "gov_take_over_gross_rev": 0,
            "gov_take_npv": 0,
            "undepreciated_asset_oil": 0,
            "undepreciated_asset_gas": 0,
            "undepreciated_asset_total": 0,
        }

    def __len__(self):
        return self.project_duration

    """
    PREVIOUS CODES (DEPRECATED)
    ===========================
    
    # Prepare attribute gas_onstream_date: set default value and error message
        oil_revenue_index = np.argwhere(self._oil_revenue > 0).ravel()
        
        if len(oil_revenue_index) > 0:
            if self.oil_onstream_date is not None:
                if self.oil_onstream_date.year < self.start_date.year:
                    raise BaseProjectException(
                        f"Oil onstream year ({self.oil_onstream_date.year}) is before " +
                        f"the start project year ({self.start_date.year})"
                    )

                if self.oil_onstream_date.year > self.end_date.year:
                    raise BaseProjectException(
                        f"Oil onstream year ({self.oil_onstream_date.year}) is after " +
                        f"the end year of the project ({self.end_date.year})"
                    )

                # Ensure oil_onstream_date provided by the user is consistent
                # with the beginning of oil lifting, indicated by the first year
                # of positive oil revenue
                oil_onstream_index = int(
                    np.argwhere(self.oil_onstream_date.year == self.project_years).ravel()
                )

                if oil_onstream_index != oil_revenue_index[0]:
                    raise BaseProjectException(
                        f"Oil onstream year ({self.oil_onstream_date.year}) is different " +
                        f"from the starting year of oil production " +
                        f"({self.project_years[oil_revenue_index[0]]})"
                    )

            else:
                self.oil_onstream_date = date(
                    year=self.project_years[oil_revenue_index[0]], month=1, day=1
                )

        else:
            self.oil_onstream_date = self.end_date

        # Prepare attribute gas_onstream_date: set default value and error message
        gas_revenue_index = np.argwhere(self._gas_revenue > 0).ravel()

        if len(gas_revenue_index) > 0:
            if self.gas_onstream_date is not None:
                if self.gas_onstream_date.year < self.start_date.year:
                    raise BaseProjectException(
                        f"Gas onstream year ({self.gas_onstream_date.year}) is before " +
                        f"the start project year ({self.start_date.year})"
                    )

                if self.gas_onstream_date.year > self.end_date.year:
                    raise BaseProjectException(
                        f"Gas onstream year ({self.gas_onstream_date.year}) is after " +
                        f"the end year of the project ({self.end_date.year})"
                    )

                # Ensure gas_onstream_date provided by the user is consistent
                # with the beginning of gas lifting, indicated by the first year
                # of positive gas revenue
                gas_onstream_index = int(
                    np.argwhere(self.gas_onstream_date.year == self.project_years).ravel()
                )

                if gas_onstream_index != gas_revenue_index[0]:
                    raise BaseProjectException(
                        f"Gas onstream year ({self.gas_onstream_date.year}) is different " +
                        f"from the starting year of gas production " +
                        f"({self.project_years[gas_revenue_index[0]]})"
                    )

            else:
                self.gas_onstream_date = date(
                    year=int(self.project_years[gas_revenue_index[0]]), month=1, day=1
                )

        else:
            self.gas_onstream_date = self.end_date
    """
