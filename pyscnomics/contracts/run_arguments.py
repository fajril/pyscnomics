"""Validate dynamic keyword mappings before contract execution."""

from collections.abc import Mapping
from enum import Enum
from typing import TypeGuard, TypeVar, TypedDict

import numpy as np
import numpy.typing as npt

from pyscnomics.econ.selection import (
    DeprMethod,
    FTPTaxRegime,
    GrossSplitRegime,
    InflationAppliedTo,
    OtherRevenue,
    TaxRegime,
    VariableSplit132024,
)


class CostRecoveryRunArguments(TypedDict, total=False):
    sulfur_revenue: OtherRevenue
    electricity_revenue: OtherRevenue
    co2_revenue: OtherRevenue
    vat_rate: npt.NDArray[np.float64] | float
    year_inflation: npt.NDArray[np.int64] | None
    inflation_rate: npt.NDArray[np.float64] | float
    inflation_rate_applied_to: InflationAppliedTo | None
    is_dmo_end_weighted: bool
    tax_regime: TaxRegime
    effective_tax_rate: npt.NDArray[np.float64] | float | None
    ftp_tax_regime: FTPTaxRegime
    depr_method: DeprMethod
    decline_factor: float | int
    post_uu_22_year2001: bool
    oil_cost_of_sales_applied: bool
    gas_cost_of_sales_applied: bool
    sum_undepreciated_cost: bool


class GrossSplitRunArguments(TypedDict, total=False):
    sulfur_revenue: OtherRevenue
    electricity_revenue: OtherRevenue
    co2_revenue: OtherRevenue
    vat_rate: npt.NDArray[np.float64] | float
    year_inflation: npt.NDArray[np.int64] | None
    inflation_rate: npt.NDArray[np.float64] | float
    inflation_rate_applied_to: InflationAppliedTo | None
    cum_production_split_offset: npt.NDArray[np.float64] | float | None
    depr_method: DeprMethod
    decline_factor: float | int
    sum_undepreciated_cost: bool
    is_dmo_end_weighted: bool
    tax_regime: TaxRegime
    effective_tax_rate: npt.NDArray[np.float64] | float | int | None
    amortization: bool
    regime: GrossSplitRegime
    reservoir_type_permen_2024: VariableSplit132024.ReservoirType


def _is_float_array(value: object) -> TypeGuard[npt.NDArray[np.float64]]:
    return isinstance(value, np.ndarray) and value.dtype == np.dtype(np.float64)


_EnumValue = TypeVar("_EnumValue", bound=Enum)


def _require_enum(value: object, expected: type[_EnumValue], name: str) -> _EnumValue:
    if not isinstance(value, expected):
        raise TypeError(f"{name} has an invalid value")
    return value


def _require_float_array_or_float(
    value: object, name: str
) -> npt.NDArray[np.float64] | float:
    if isinstance(value, float):
        return value
    if _is_float_array(value):
        return value
    raise TypeError(f"{name} must be a float or float64 NumPy array")


def _require_float_array_or_float_or_none(
    value: object, name: str
) -> npt.NDArray[np.float64] | float | None:
    if value is None:
        return None
    return _require_float_array_or_float(value, name)


def _require_float_array_or_number_or_none(
    value: object, name: str
) -> npt.NDArray[np.float64] | float | int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    return _require_float_array_or_float(value, name)


def _require_year_array_or_none(
    value: object, name: str
) -> npt.NDArray[np.int64] | None:
    if value is None:
        return None
    if _is_year_array(value):
        return value
    raise TypeError(f"{name} must be an int64 NumPy array or None")


def _is_year_array(value: object) -> TypeGuard[npt.NDArray[np.int64]]:
    return isinstance(value, np.ndarray) and value.dtype == np.dtype(np.int64)


def _require_bool(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError(f"{name} must be a bool")


def _require_number(value: object, name: str) -> float | int:
    if isinstance(value, (float, int)):
        return value
    raise TypeError(f"{name} must be a float or int")


def _validate_keys(arguments: Mapping[str, object], allowed: frozenset[str]) -> None:
    if any(key not in allowed for key in arguments):
        raise TypeError("contract arguments contain an unsupported keyword")


def validate_cost_recovery_run_arguments(
    arguments: Mapping[str, object],
) -> CostRecoveryRunArguments:
    allowed = frozenset(CostRecoveryRunArguments.__annotations__)
    _validate_keys(arguments, allowed)
    result: CostRecoveryRunArguments = {}
    if "sulfur_revenue" in arguments:
        result["sulfur_revenue"] = _require_enum(
            arguments["sulfur_revenue"], OtherRevenue, "sulfur_revenue"
        )
    if "electricity_revenue" in arguments:
        result["electricity_revenue"] = _require_enum(
            arguments["electricity_revenue"], OtherRevenue, "electricity_revenue"
        )
    if "co2_revenue" in arguments:
        result["co2_revenue"] = _require_enum(
            arguments["co2_revenue"], OtherRevenue, "co2_revenue"
        )
    if "vat_rate" in arguments:
        result["vat_rate"] = _require_float_array_or_float(
            arguments["vat_rate"], "vat_rate"
        )
    if "year_inflation" in arguments:
        result["year_inflation"] = _require_year_array_or_none(
            arguments["year_inflation"], "year_inflation"
        )
    if "inflation_rate" in arguments:
        result["inflation_rate"] = _require_float_array_or_float(
            arguments["inflation_rate"], "inflation_rate"
        )
    if "inflation_rate_applied_to" in arguments:
        value = arguments["inflation_rate_applied_to"]
        result["inflation_rate_applied_to"] = (
            None
            if value is None
            else _require_enum(value, InflationAppliedTo, "inflation_rate_applied_to")
        )
    if "is_dmo_end_weighted" in arguments:
        result["is_dmo_end_weighted"] = _require_bool(
            arguments["is_dmo_end_weighted"], "is_dmo_end_weighted"
        )
    if "post_uu_22_year2001" in arguments:
        result["post_uu_22_year2001"] = _require_bool(
            arguments["post_uu_22_year2001"], "post_uu_22_year2001"
        )
    if "oil_cost_of_sales_applied" in arguments:
        result["oil_cost_of_sales_applied"] = _require_bool(
            arguments["oil_cost_of_sales_applied"], "oil_cost_of_sales_applied"
        )
    if "gas_cost_of_sales_applied" in arguments:
        result["gas_cost_of_sales_applied"] = _require_bool(
            arguments["gas_cost_of_sales_applied"], "gas_cost_of_sales_applied"
        )
    if "sum_undepreciated_cost" in arguments:
        result["sum_undepreciated_cost"] = _require_bool(
            arguments["sum_undepreciated_cost"], "sum_undepreciated_cost"
        )
    if "tax_regime" in arguments:
        result["tax_regime"] = _require_enum(
            arguments["tax_regime"], TaxRegime, "tax_regime"
        )
    if "effective_tax_rate" in arguments:
        result["effective_tax_rate"] = _require_float_array_or_float_or_none(
            arguments["effective_tax_rate"], "effective_tax_rate"
        )
    if "ftp_tax_regime" in arguments:
        result["ftp_tax_regime"] = _require_enum(
            arguments["ftp_tax_regime"], FTPTaxRegime, "ftp_tax_regime"
        )
    if "depr_method" in arguments:
        result["depr_method"] = _require_enum(
            arguments["depr_method"], DeprMethod, "depr_method"
        )
    if "decline_factor" in arguments:
        result["decline_factor"] = _require_number(
            arguments["decline_factor"], "decline_factor"
        )
    return result


def validate_gross_split_run_arguments(
    arguments: Mapping[str, object],
) -> GrossSplitRunArguments:
    allowed = frozenset(GrossSplitRunArguments.__annotations__)
    _validate_keys(arguments, allowed)
    result: GrossSplitRunArguments = {}
    if "sulfur_revenue" in arguments:
        result["sulfur_revenue"] = _require_enum(
            arguments["sulfur_revenue"], OtherRevenue, "sulfur_revenue"
        )
    if "electricity_revenue" in arguments:
        result["electricity_revenue"] = _require_enum(
            arguments["electricity_revenue"], OtherRevenue, "electricity_revenue"
        )
    if "co2_revenue" in arguments:
        result["co2_revenue"] = _require_enum(
            arguments["co2_revenue"], OtherRevenue, "co2_revenue"
        )
    if "vat_rate" in arguments:
        result["vat_rate"] = _require_float_array_or_float(
            arguments["vat_rate"], "vat_rate"
        )
    if "inflation_rate" in arguments:
        result["inflation_rate"] = _require_float_array_or_float(
            arguments["inflation_rate"], "inflation_rate"
        )
    if "cum_production_split_offset" in arguments:
        result["cum_production_split_offset"] = _require_float_array_or_float_or_none(
            arguments["cum_production_split_offset"], "cum_production_split_offset"
        )
    if "year_inflation" in arguments:
        result["year_inflation"] = _require_year_array_or_none(
            arguments["year_inflation"], "year_inflation"
        )
    if "sum_undepreciated_cost" in arguments:
        result["sum_undepreciated_cost"] = _require_bool(
            arguments["sum_undepreciated_cost"], "sum_undepreciated_cost"
        )
    if "is_dmo_end_weighted" in arguments:
        result["is_dmo_end_weighted"] = _require_bool(
            arguments["is_dmo_end_weighted"], "is_dmo_end_weighted"
        )
    if "amortization" in arguments:
        result["amortization"] = _require_bool(
            arguments["amortization"], "amortization"
        )
    if "tax_regime" in arguments:
        result["tax_regime"] = _require_enum(
            arguments["tax_regime"], TaxRegime, "tax_regime"
        )
    if "effective_tax_rate" in arguments:
        value = arguments["effective_tax_rate"]
        result["effective_tax_rate"] = (
            None
            if value is None
            else _require_float_array_or_number_or_none(value, "effective_tax_rate")
        )
    if "depr_method" in arguments:
        result["depr_method"] = _require_enum(
            arguments["depr_method"], DeprMethod, "depr_method"
        )
    if "decline_factor" in arguments:
        result["decline_factor"] = _require_number(
            arguments["decline_factor"], "decline_factor"
        )
    if "regime" in arguments:
        result["regime"] = _require_enum(
            arguments["regime"], GrossSplitRegime, "regime"
        )
    if "reservoir_type_permen_2024" in arguments:
        result["reservoir_type_permen_2024"] = _require_enum(
            arguments["reservoir_type_permen_2024"],
            VariableSplit132024.ReservoirType,
            "reservoir_type_permen_2024",
        )
    return result
