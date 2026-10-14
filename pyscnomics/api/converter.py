"""
This file containing the tools which utilized by API adapter.
"""

from datetime import date, datetime
from collections.abc import Mapping, Sequence
from typing import ClassVar, Protocol, Required, TypeGuard, runtime_checkable

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from typing_extensions import TypedDict

from pyscnomics.econ.costs import CapitalCost, Intangible, OPEX, ASR, CostOfSales, LBT
from pyscnomics.dataset.sample import assign_lifting, read_fluid_type, read_cost_type
from pyscnomics.validation_rules import parse_engine_date
from pyscnomics.econ.selection import (
    CostType,
    TaxRegime,
    TaxType,
    FTPTaxRegime,
    FluidType,
    GrossSplitRegime,
    LimitMethod,
    UncertaintyDistribution,
    SunkCostMethod,
    VariableSplit132024,
    InitialYearAmortizationIncurred,
    DeprMethod,
    DiscountingMode,
    InflationAppliedTo,
    NPVSelection,
    OptimizationParameter,
    OptimizationTarget,
    OtherRevenue,
    TaxSplitTypeCR,
)
from pyscnomics.econ.revenue import Lifting
from pyscnomics.tools.helper import (
    get_inflation_applied_converter,
    get_npv_mode_converter,
    get_discounting_mode_converter,
    get_depreciation_method_converter,
    get_other_revenue_converter,
    get_split_type_converter,
    get_optimization_target_converter,
    get_optimization_parameter_converter,
)
from pyscnomics.input_validation import InputValidationError, engine_input_issue


class ConverterInputError(  # pyright: ignore[reportUnsafeMultipleInheritance]
    ValueError, InputValidationError
):
    """Verified user-input converter error retaining ``ValueError``.

    The converter raised ``ValueError`` for an unrecognized enum/label, so this
    class keeps that documented superclass catchable for library callers.
    """

    _input_code: str = "CONVERTER_INPUT_INVALID"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        InputValidationError.__init__(
            self,
            (engine_input_issue(code=code or self._input_code, message=message),),
        )


class ConverterTypeError(  # pyright: ignore[reportUnsafeMultipleInheritance]
    TypeError, InputValidationError
):
    """Verified user-input converter type error retaining ``TypeError``."""

    _input_code: str = "CONVERTER_TYPE_INVALID"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        InputValidationError.__init__(
            self,
            (engine_input_issue(code=code or self._input_code, message=message),),
        )


class _CostInput(TypedDict, total=False):
    """Validated cost payload shape consumed by the converter boundaries."""

    start_year: Required[int]
    end_year: Required[int]
    expense_year: Required[list[int]]
    cost: list[float | int] | None
    cost_allocation: list[str] | None
    cost_type: list[str | None] | None
    description: list[str] | None
    tax_portion: list[float | int] | None
    tax_discount: list[float | int] | None
    pis_year: list[int | None] | None
    salvage_value: list[float | None] | None
    useful_life: list[float | None] | None
    depreciation_factor: list[float | None] | None
    is_ic_applied: list[bool | None] | None
    fixed_cost: list[float | int] | None
    prod_rate: list[float | int] | None
    cost_per_volume: list[float | int] | None
    final_year: list[float | int] | None
    future_rate: list[float | int] | float | int | None
    utilized_land_area: list[float | int] | None
    utilized_building_area: list[float | int] | None
    njop_land: list[float | int] | None
    njop_building: list[float | int] | None
    gross_revenue: list[float | int] | None


_STRING_OBJECT_ADAPTER: TypeAdapter[dict[str, object] | None] = TypeAdapter(
    dict[str, object] | None
)
_COST_INPUT_ADAPTER: TypeAdapter[dict[str, _CostInput] | None] = TypeAdapter(
    dict[str, _CostInput] | None
)


class _FiniteRequestModel(BaseModel):
    """Base for request models that must reject non-finite JSON numbers.

    ``allow_inf_nan=False`` rejects ``NaN``/``Infinity`` and their coercible
    string forms (``"nan"``, ``"inf"``, ``"Infinity"``) at the request boundary,
    before any engine work. Ordinary numeric strings, zero, and nullable
    defaults keep their existing coercion semantics.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(allow_inf_nan=False)



@runtime_checkable
class _SupportsFloat(Protocol):
    def __float__(self) -> float: ...


@runtime_checkable
class _SupportsIndex(Protocol):
    def __index__(self) -> int: ...


def _is_float_convertible(
    value: object,
) -> TypeGuard[_SupportsFloat | _SupportsIndex | str | bytes | bytearray | memoryview]:
    return isinstance(
        value,
        (_SupportsFloat, _SupportsIndex, str, bytes, bytearray, memoryview),
    )


def _validate_string_object_mapping(data_raw: object) -> dict[str, object] | None:
    return _STRING_OBJECT_ADAPTER.validate_python(data_raw, strict=True)


def _is_cost_input_mapping(
    value: object,
) -> TypeGuard[dict[str, _CostInput]]:
    if not isinstance(value, dict):
        return False
    try:
        validated = _COST_INPUT_ADAPTER.validate_python(value, strict=True)
    except (TypeError, ValueError):
        return False
    return validated is not None


def _validate_cost_input_mapping(data_raw: object) -> dict[str, _CostInput] | None:
    validated: dict[str, _CostInput] | None = _COST_INPUT_ADAPTER.validate_python(
        data_raw, strict=True
    )
    if validated is None:
        return None
    if not _is_cost_input_mapping(validated):
        raise TypeError("Cost input must be a mapping of string keys to records")
    return validated


def _optional_float_array(
    value: Sequence[float | int | None] | None,
) -> npt.NDArray[np.float64] | None:
    """Return a float array while preserving the established ``None`` sentinel."""
    return None if value is None else np.array(value, dtype=float)


def _optional_array(
    value: Sequence[float | int | None] | None,
) -> npt.NDArray[np.float64] | None:
    """Convert optional numeric cost fields to the engine's float array type."""
    if value is None:
        return None
    return np.array(
        [0.0 if item is None else item for item in value],
        dtype=np.float64,
    )


def _optional_cost_types(
    value: list[str | None] | None,
) -> list[CostType | None] | None:
    return None if value is None else read_cost_type(cost_type=value)


def _optional_allocations(value: list[str] | None) -> list[FluidType] | None:
    return None if value is None else read_fluid_type(fluid=value)


def _normalized_ic_flags(value: list[bool | None] | None) -> list[bool] | None:
    """Normalize nullable API flags to the concrete engine representation."""
    return (
        None if value is None else [False if flag is None else flag for flag in value]
    )


def _optional_future_rate(
    value: Sequence[float | int] | float | int | None,
) -> npt.NDArray[np.float64] | None:
    return None if value is None else np.array(value, dtype=float)


class _CostArgumentsBase(TypedDict, total=False):
    start_year: int
    end_year: int
    expense_year: npt.NDArray[np.int64]
    cost_type: list[CostType | None]
    cost_allocation: list[FluidType]
    description: list[str]
    tax_portion: npt.NDArray[np.float64]
    tax_discount: float | npt.NDArray[np.float64]


class _GeneralCostArguments(_CostArgumentsBase, total=False):
    cost: npt.NDArray[np.float64]


class _CapitalCostArguments(_GeneralCostArguments, total=False):
    pis_year: npt.NDArray[np.int64]
    salvage_value: npt.NDArray[np.float64]
    useful_life: npt.NDArray[np.float64]
    depreciation_factor: npt.NDArray[np.float64]
    is_ic_applied: list[bool]


class _OPEXArguments(_CostArgumentsBase, total=False):
    fixed_cost: npt.NDArray[np.float64]
    prod_rate: npt.NDArray[np.float64]
    cost_per_volume: npt.NDArray[np.float64]


class _ASRArguments(_GeneralCostArguments, total=False):
    final_year: npt.NDArray[np.int64]
    future_rate: npt.NDArray[np.float64]


class _LBTArguments(_GeneralCostArguments, total=False):
    final_year: npt.NDArray[np.int64]
    utilized_land_area: npt.NDArray[np.float64]
    utilized_building_area: npt.NDArray[np.float64]
    njop_land: npt.NDArray[np.float64]
    njop_building: npt.NDArray[np.float64]
    gross_revenue: npt.NDArray[np.float64]


def _capital_pis_year(
    value: Sequence[int | None] | None,
    expense_year: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64] | None:
    if value is None:
        return None
    # A PIS year of 0 means "unset" (like None) and defaults to the entry's
    # expense year, mirroring the existing None handling.
    missing = np.array([item is None or item == 0 for item in value], dtype=bool)
    pis_year = np.array([0 if item is None else item for item in value], dtype=np.int64)
    pis_year[missing] = expense_year[missing]
    return pis_year


def _normalized_final_year(
    value: Sequence[float | int] | None,
    expense_year: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64] | None:
    if value is None:
        return None
    final_year = np.array(value, dtype=np.float64)
    missing = np.isnan(final_year)
    normalized = np.zeros(final_year.shape, dtype=np.int64)
    normalized[~missing] = final_year[~missing]
    normalized[missing] = expense_year[missing]
    return normalized


def _tax_discount(item: _CostInput) -> float | npt.NDArray[np.float64]:
    value = item.get("tax_discount")
    return 0.0 if value is None else np.array(value, dtype=np.float64)


def _capital_cost_arguments(item: _CostInput) -> _CapitalCostArguments:
    if "cost" not in item:
        raise KeyError("cost")
    expense_year = np.array(item["expense_year"], dtype=np.int64)
    arguments: _CapitalCostArguments = {
        "start_year": item["start_year"],
        "end_year": item["end_year"],
        "cost": np.array(item.get("cost"), dtype=np.float64),
        "expense_year": expense_year,
        "tax_discount": _tax_discount(item),
    }
    cost_type = _optional_cost_types(item.get("cost_type"))
    if cost_type is not None:
        arguments["cost_type"] = cost_type
    cost_allocation = _optional_allocations(item.get("cost_allocation"))
    if cost_allocation is not None:
        arguments["cost_allocation"] = cost_allocation
    description = item.get("description")
    if description is not None:
        arguments["description"] = description
    tax_portion = _optional_float_array(item.get("tax_portion"))
    if tax_portion is not None:
        arguments["tax_portion"] = tax_portion
    pis_year = _capital_pis_year(item.get("pis_year"), expense_year)
    if pis_year is not None:
        arguments["pis_year"] = pis_year
    for key in ("salvage_value", "useful_life", "depreciation_factor"):
        value = _optional_array(item.get(key))
        if value is not None:
            arguments[key] = value
    is_ic_applied = _normalized_ic_flags(item.get("is_ic_applied"))
    if is_ic_applied is not None:
        arguments["is_ic_applied"] = is_ic_applied
    return arguments


def _intangible_arguments(item: _CostInput) -> _GeneralCostArguments:
    if "cost" not in item:
        raise KeyError("cost")
    arguments: _GeneralCostArguments = {
        "start_year": item["start_year"],
        "end_year": item["end_year"],
        "cost": np.array(item.get("cost"), dtype=np.float64),
        "expense_year": np.array(item["expense_year"], dtype=np.int64),
        "tax_discount": _tax_discount(item),
    }
    cost_type = _optional_cost_types(item.get("cost_type"))
    if cost_type is not None:
        arguments["cost_type"] = cost_type
    cost_allocation = _optional_allocations(item.get("cost_allocation"))
    if cost_allocation is not None:
        arguments["cost_allocation"] = cost_allocation
    description = item.get("description")
    if description is not None:
        arguments["description"] = description
    tax_portion = _optional_float_array(item.get("tax_portion"))
    if tax_portion is not None:
        arguments["tax_portion"] = tax_portion
    return arguments


def _opex_arguments(item: _CostInput) -> _OPEXArguments:
    arguments: _OPEXArguments = {
        "start_year": item["start_year"],
        "end_year": item["end_year"],
        "expense_year": np.array(item["expense_year"], dtype=np.int64),
        "tax_discount": _tax_discount(item),
    }
    cost_type = _optional_cost_types(item.get("cost_type"))
    if cost_type is not None:
        arguments["cost_type"] = cost_type
    cost_allocation = _optional_allocations(item.get("cost_allocation"))
    if cost_allocation is not None:
        arguments["cost_allocation"] = cost_allocation
    description = item.get("description")
    if description is not None:
        arguments["description"] = description
    tax_portion = _optional_float_array(item.get("tax_portion"))
    if tax_portion is not None:
        arguments["tax_portion"] = tax_portion
    for key in ("fixed_cost", "prod_rate", "cost_per_volume"):
        value = _optional_float_array(item.get(key))
        if value is not None:
            arguments[key] = value
    return arguments


def _asr_arguments(item: _CostInput) -> _ASRArguments:
    if "cost" not in item:
        raise KeyError("cost")
    expense_year = np.array(item["expense_year"], dtype=np.int64)
    arguments: _ASRArguments = {
        "start_year": item["start_year"],
        "end_year": item["end_year"],
        "cost": np.array(item.get("cost"), dtype=np.float64),
        "expense_year": expense_year,
        "tax_discount": _tax_discount(item),
    }
    cost_type = _optional_cost_types(item.get("cost_type"))
    if cost_type is not None:
        arguments["cost_type"] = cost_type
    cost_allocation = _optional_allocations(item.get("cost_allocation"))
    if cost_allocation is not None:
        arguments["cost_allocation"] = cost_allocation
    description = item.get("description")
    if description is not None:
        arguments["description"] = description
    tax_portion = _optional_float_array(item.get("tax_portion"))
    if tax_portion is not None:
        arguments["tax_portion"] = tax_portion
    if "final_year" not in item:
        raise KeyError("final_year")
    final_year = _normalized_final_year(item.get("final_year"), expense_year)
    if final_year is not None:
        arguments["final_year"] = final_year
    future_rate = _optional_future_rate(item.get("future_rate"))
    if future_rate is not None:
        arguments["future_rate"] = future_rate
    return arguments


def _lbt_arguments(item: _CostInput) -> _LBTArguments:
    expense_year = np.array(item["expense_year"], dtype=np.int64)
    arguments: _LBTArguments = {
        "start_year": item["start_year"],
        "end_year": item["end_year"],
        "expense_year": expense_year,
        "tax_discount": _tax_discount(item),
    }
    cost = _optional_float_array(item.get("cost"))
    if cost is not None:
        arguments["cost"] = cost
    cost_type = _optional_cost_types(item.get("cost_type"))
    if cost_type is not None:
        arguments["cost_type"] = cost_type
    cost_allocation = _optional_allocations(item.get("cost_allocation"))
    if cost_allocation is not None:
        arguments["cost_allocation"] = cost_allocation
    description = item.get("description")
    if description is not None:
        arguments["description"] = description
    tax_portion = _optional_float_array(item.get("tax_portion"))
    if tax_portion is not None:
        arguments["tax_portion"] = tax_portion
    if "final_year" not in item:
        raise KeyError("final_year")
    final_year = _normalized_final_year(item.get("final_year"), expense_year)
    if final_year is not None:
        arguments["final_year"] = final_year
    for key in (
        "utilized_land_area",
        "utilized_building_area",
        "njop_land",
        "njop_building",
        "gross_revenue",
    ):
        if key not in item:
            raise KeyError(key)
        value = _optional_float_array(item.get(key))
        if value is not None:
            arguments[key] = value
    return arguments


class SetupBM(_FiniteRequestModel):
    """
    The BaseModel to validate the API setup input data.

    Parameters
    ----------
    start_date : str | int
        The start date of the project.
    end_date : str | int
        The end date of the project.
    oil_onstream_date: str | int | None
        The start date of oil production.
    gas_onstream_date: str | int | None
        The start date of gas production.
    approval_year: int | None
        The project approval year.
    is_pod_1: bool | None
        Whether the project is POD I.
    is_strict: bool | None
        Whether strict cost validation is enabled.
    """

    start_date: str | int = "01/01/2010"
    end_date: str | int = "31/12/2045"
    oil_onstream_date: str | int | None = None
    gas_onstream_date: str | int | None = None
    approval_year: int | None = None
    is_pod_1: bool | None = None
    is_strict: bool | None = None


class SummaryArgumentsBM(_FiniteRequestModel):
    """
    BaseModel to validate the API summary arguments input data.

    Parameters
    ----------
    reference_year: int
        The reference year used to calculate the economic indicator.
    inflation_rate: float
        The inflation rate used to calculate the economic indicator and cost inflation.
    discount_rate: float
        The discount rate used to calculate the discounted economic indicator.
    npv_mode: str
        The mode of the NPV used in the NPV calculation. The available option are:
        [SKK Full Cycle Real Terms, SKK Full Cycle Nominal Terms,
        Full Cycle Real Terms, Full Cycle Nominal Terms, Point Forward]
    discounting_mode: str
        The discounting mode used in the NPV calculation. The available option are: [End Year, Mid Year]

    """

    reference_year: int = 2022
    discount_rate_start_year: int | None = None
    inflation_rate: float | int | list[object] = 0.1
    discount_rate: float | int = 0.1
    npv_mode: str = "SKK Full Cycle Nominal Terms"
    discounting_mode: str = "End Year"
    profitability_discounted: bool = False


class CostRecoveryBM(_FiniteRequestModel):
    """
    The BaseModel to validate the API Cost Recovery input data.

    Parameters
    ----------
    oil_ftp_is_available: bool
        The oil ftp availability of the cost recovery contract.
    oil_ftp_is_shared: bool
        The oil shared ftp condition of the cost recovery contract.
    oil_ftp_portion: float
        The oil ftp portion.
    gas_ftp_is_available: bool
        The gas ftp availability of the cost recovery contract.
    gas_ftp_is_shared: bool
        The gas shared ftp condition of the cost recovery contract.
    gas_ftp_portion: float
        The gas ftp portion.
    tax_split_type: str
        The tax split type used in the contract.
        The available option are: [Conventional, RC Split, ICP Sliding Scale].
    condition_dict: dict
        The condition input for the tax split type except for the conventional split.
    indicator_rc_icp_sliding: list
        The indicator input for the tax split type except for the conventional split.
    oil_ctr_pretax_share: float
        The oil contractor pre-tax share.
    gas_ctr_pretax_share: float
        The gas contractor pre-tax share.
    oil_ic_rate: float
        The oil Investment Credit (IC) rate.
    gas_ic_rate: float
        The gas Investment Credit (IC) rate.
    ic_is_available: bool
        The Investment Credit (IC) availability.
    oil_cr_cap_rate: float
        The oil cost recovery cap rate
    gas_cr_cap_rate: float
        The gas cost recovery cap rate
    oil_dmo_volume_portion: float
        The oil Domestic Market Obligation (DMO) volume portion.
    oil_dmo_fee_portion: float
        The oil Domestic Market Obligation (DMO) fee portion.
    oil_dmo_holiday_duration: float
        The oil Domestic Market Obligation (DMO) Holiday duration.
    gas_dmo_volume_portion: float
        The gas Domestic Market Obligation (DMO) volume portion.
    gas_dmo_fee_portion: float
        The gas Domestic Market Obligation (DMO) fee portion.
    gas_dmo_holiday_duration: float
        The gas Domestic Market Obligation (DMO) Holiday duration.
    oil_carry_forward_depreciation
        The oil carry forward depreciation.
    gas_carry_forward_depreciation
        The gas carry forward depreciation.
    """

    oil_ftp_is_available: bool = True
    oil_ftp_is_shared: bool = True
    oil_ftp_portion: float | int | list[float] = 0.2
    gas_ftp_is_available: bool = True
    gas_ftp_is_shared: bool = True
    gas_ftp_portion: float | int | list[float] = 0.2
    tax_split_type: str = "Conventional"
    condition_dict: dict[str, object] = {}
    indicator_rc_icp_sliding: list[float] = []
    oil_ctr_pretax_share: float | int | list[float] = 0.34722220
    gas_ctr_pretax_share: float | int | list[float] = 0.5208330
    oil_ic_rate: float | int = 0
    gas_ic_rate: float | int = 0
    ic_is_available: bool = False
    oil_cr_cap_rate: float | int = 1
    gas_cr_cap_rate: float | int = 1
    oil_dmo_volume_portion: float | int | list[float] = 0.25
    oil_dmo_fee_portion: float | int | list[float] = 0.25
    oil_dmo_holiday_duration: float | int = 60
    gas_dmo_volume_portion: float | int | list[float] = 0.25
    gas_dmo_fee_portion: float | int | list[float] = 1
    gas_dmo_holiday_duration: float | int = 60
    oil_carry_forward_depreciation: float | int | list[float] = 0.0
    gas_carry_forward_depreciation: float | int | list[float] = 0.0


class GrossSplitBM(_FiniteRequestModel):
    """
    The BaseModel to validate the API Gross Split input data.

    Parameters
    ----------
    field_status: str
        The field status. The available options are:
        [POD I, POD II, POFD, No POD].
    field_loc: str
        The field location. The available options are:
        [Onshore, Offshore (0<h<=20), Offshore (20<h<=50),
        Offshore (50<h<=150), Offshore (150<h<=1000), Offshore (h>1000)].
    res_depth: str
        The reservoir depth of the field. The available options are:
        [<=2500, >2500]
    infra_avail: str
        The infrastructure availability of the field. The available options are:
        [Well Developed, New Frontier].
    res_type: str
        The reservoir type of the field. The available options are:
        [Conventional, Non-Conventional]
    api_oil: str
        The oil API classification. The available options are:
        [<25, >=25].
    domestic_use: str
        The local content of the contract. The available options are:
        [<30, 30<=x<50, 50<=x<70, 70<=x<100]
    prod_stage: str
        The production stage of the contract. The available options are:
        [Primary, Secondary, Tertiary]
    co2_content: str
        The CO2 content of the field produced fluid. The available options are:
        [<5, 5<=x<10, 10<=x<20, 20<=x<40, 40<=x<60, x>=60]
    h2s_content: str
        The H2S content of the field produced fluid. The available options are:
        [<100, 100<=x<300, 300<=x<500, x>=500]
    base_split_ctr_oil: float
        The contractor base split for oil.
    base_split_ctr_gas: float
        The contractor base split for gas.
    split_ministry_disc: float
        The ministerial discretion split.
    oil_dmo_volume_portion: float
        The oil Domestic Market Obligation (DMO) volume portion.
    oil_dmo_fee_portion: float
        The oil Domestic Market Obligation (DMO) fee portion.
    oil_dmo_holiday_duration: float
        The oil Domestic Market Obligation (DMO) Holiday duration.
    gas_dmo_volume_portion: float
        The gas Domestic Market Obligation (DMO) volume portion.
    gas_dmo_fee_portion: float
        The gas Domestic Market Obligation (DMO) fee portion.
    gas_dmo_holiday_duration: float
        The gas Domestic Market Obligation (DMO) Holiday duration.
    oil_carry_forward_depreciation
        The oil carry forward depreciation.
    gas_carry_forward_depreciation
        The gas carry forward depreciation.

    """

    field_status: str | None = Field(default=None)
    field_loc: str | None = Field(default=None)
    field_loc_2024: str | None = Field(default=None)
    res_depth: str | None = Field(default=None)
    infra_avail: str | None = Field(default=None)
    infra_avail_2024: str | None = Field(default=None)
    res_type: str | None = Field(default=None)
    api_oil: str | None = Field(default=None)
    domestic_use: str | None = Field(default=None)
    prod_stage: str | None = Field(default=None)
    co2_content: str | None = Field(default=None)
    h2s_content: str | None = Field(default=None)
    field_reserves: str | None = Field(default=None)
    field_reserves_2024: str | None = Field(default=None)
    base_split_ctr_oil: float | int = 0.43
    base_split_ctr_gas: float | int = 0.48
    split_ministry_disc: float | int = 0.08
    oil_dmo_volume_portion: float | int | list[float] = 0.25
    oil_dmo_fee_portion: float | int | list[float] = 1.0
    oil_dmo_holiday_duration: float | int = 60
    gas_dmo_volume_portion: float | int | list[float] = 1.0
    gas_dmo_fee_portion: float | int | list[float] = 1.0
    gas_dmo_holiday_duration: float | int = 60
    oil_carry_forward_depreciation: float | int | list[float] = 0.0
    gas_carry_forward_depreciation: float | int | list[float] = 0.0


class ContractArgumentsBM(_FiniteRequestModel):
    """
    The BaseModel to validate the Contract Arguments input data.

    Parameters
    ----------
    sulfur_revenue: str
        The treatment of the sulfur revenue. The options are:
        [Addition to Oil Revenue, Addition to Gas Revenue, Reduction to Oil OPEX, Reduction to Gas OPEX]
    electricity_revenue: str
        The treatment of the electricity revenue. The options are:
        [Addition to Oil Revenue, Addition to Gas Revenue, Reduction to Oil OPEX, Reduction to Gas OPEX]
    co2_revenue: str
        The treatment of the CO2 revenue. The options are:
        [Addition to Oil Revenue, Addition to Gas Revenue, Reduction to Oil OPEX, Reduction to Gas OPEX]
    is_dmo_end_weighted: bool
        The option treatment of the DMO.
    tax_regime: str
        The tax regime used in tax calculation. The options are:
        [nailed, prevailing, UU No.36 Tahun 2008, UU No.02 Tahun 2020, UU No.07 Tahun 2021]
    effective_tax_rate: float
        The tax rate used for the tax payment.
    ftp_tax_regime: str
        The tax scheme used to calculate the tax payment. The options are:
        [PDJP No.20 Tahun 2017, Pre PDJP No.20 Tahun 2017, Direct]
    sunk_cost_reference_year: int
        The sunk cost reference year.
    depr_method: str
        The depreciation method used for calculating the depreciation. The options are:
        [PSC Declining Balance, Declining Balance, Unit Of Production, Straight Line]
    decline_factor: float
        The decline factor for the depreciation if the depreciation method is Declining Balance.
    vat_rate: list
        The Value Added Tax (VAT) rate used in the calculation.
    lbt_rate: list
        The Land and Building (LBT) rate used in the calculation.
    inflation_rate: list
        The inflation rate used in the calculation.
    inflation_rate_applied_to: str
        The option of where the inflation will be applied to. The options are:
        [CAPEX, OPEX, CAPEX AND OPEX]

    """

    sulfur_revenue: str = "Addition to Oil Revenue"
    electricity_revenue: str = "Addition to Oil Revenue"
    co2_revenue: str = "Addition to Oil Revenue"
    is_dmo_end_weighted: bool = False
    tax_regime: str = "nailed down"
    effective_tax_rate: float | list[object] | int | None = 0.424
    ftp_tax_regime: str = "Pre PDJP No.20 Tahun 2017"
    sunk_cost_reference_year: int | None = 2021
    depr_method: str = "PSC Declining Balance"
    decline_factor: float | int = 2
    vat_rate: list[object] | float | int = 0.0
    lbt_rate: list[object] | float | int = 0.0
    inflation_rate: list[object] | float | int = 0.0
    inflation_rate_applied_to: str = "CAPEX"
    post_uu_22_year2001: bool = True
    cum_production_split_offset: list[object] | float | int | None = None
    amortization: bool = False
    regime: str = "PERMEN_ESDM_12_2020"
    reservoir_type_permen_2024: str = "conventional"
    oil_cost_of_sales_applied: bool = False
    gas_cost_of_sales_applied: bool = False
    sum_undepreciated_cost: bool = False


class ContractArgumentsTransitionBM(_FiniteRequestModel):
    """
    The BaseModel to validate the Contract Arguments of Transition input data.

    Parameters
    ----------
    unrec_portion: float
        The unrec portion that will be transferred into second contract.
    """

    unrec_portion: float | int


class LiftingBM(_FiniteRequestModel):
    """
    The BaseModel to validate the Lifting input data.

    Parameters
    ----------
    start_year: int
        The start year of the lifting.
    end_year: int
        The end year of the lifting.
    lifting_rate: list[float]
        The list containing the lifting rate.
    price: list[float]
        The list containing the price.
    prod_year: list[int]
        The list containing the production year of the corresponding lifting.
    fluid_type: str
        The type of the corresponding fluid.
    ghv: list[float] | None
        The list containing the Gross Heating Value (GHV) of the corresponding fluid.
    prod_rate: list[float] | None
        The list containing the production rate of the corresponding lifting.
    prod_rate_baseline: list[float] | list[int] | None
        The list containing the production rate baseline of the corresponding lifting.
    """

    start_year: int
    end_year: int
    lifting_rate: list[float] | list[int]
    price: list[float] | list[int]
    prod_year: list[int]
    fluid_type: str
    ghv: list[float] | list[int] | None = None
    prod_rate: list[float] | list[int] | None = None
    prod_rate_baseline: list[float] | list[int] | None = None


class GeneralCostBM(_FiniteRequestModel):
    start_year: int
    end_year: int
    expense_year: list[int]
    cost: list[float] | list[int] | None
    cost_allocation: list[str] | None = Field(default=None)
    cost_type: list[str | None] | None = Field(default=None)
    description: list[str] | None = Field(default=None)
    tax_portion: list[float] | list[int] | None = Field(default=None)
    tax_discount: list[float] | list[int] | None = Field(default=None)


class CapitalCostBM(GeneralCostBM):
    """
    The BaseModel to validate the Capital Cost input data.

    Parameters
    ----------
    pis_year: list[int]
        The list representing the PIS year of a capital asset.
    salvage_value: list[float]
        The list representing the salvage value of a capital asset.
    useful_life: list[int]
        The list representing the useful life of a capital asset.
    depreciation_factor: list[float]
        The value of depreciation factor to be used in PSC DB depreciation method.
        Default value is 0.5 for the entire project duration.
    is_ic_applied: list[bool]
        The condition of will the IC applied or not.
    """

    pis_year: list[int | None] | None = Field(default=None)
    salvage_value: list[float | None] | None = Field(default=None)
    useful_life: list[float | None] | None = Field(default=None)
    depreciation_factor: list[float | None] | None = Field(default=None)
    is_ic_applied: list[bool | None] | None = Field(default=None)


class IntangibleBM(GeneralCostBM):
    """
    The BaseModel to validate the Intangible input data.
    """


class OpexBM(GeneralCostBM):
    """
    The BaseModel to validate the Opex input data.

    Parameters
    ----------
    fixed_cost : list
        An list representing the fixed cost of an opex asset.
    prod_rate: list
        The production rate of a particular fluid type.
    cost_per_volume: list
        Cost associated with production of a particular fluid type.
    """

    fixed_cost: list[float] | list[int]
    prod_rate: list[float] | list[int] | None = Field(default=None)
    cost_per_volume: list[float] | list[int] | None = Field(default=None)

    cost: list[float] | list[int] | None = Field(default=None, exclude=True)


class AsrBM(GeneralCostBM):
    """
    The BaseModel to validate the ASR input data.

    Parameters
    ----------
    final_year
        A list representing the final year of an ASR asset.
    future_rate
        A list representing the future rate of an ASR asset.
    """

    final_year: list[float] | list[int] | None = Field(default=None)
    future_rate: list[float] | list[int] | float | int | None = Field(default=None)


class CostOfSalesBM(GeneralCostBM):
    """
    The BaseModel to validate the cost of sales input data.
    """


class LbtBM(GeneralCostBM):
    """
    The BaseModel to validate the Land and Building Tax (LBT) input data.

    Parameters
    ----------
    final_year
        The year in which cost distribution ends for each cost element.
    utilized_land_area
        The utilized land area.
    utilized_building_area
        The utilized building area.
    njop_land
        NJOP related to land.
    njop_building
        NJOP related to building.
    gross_revenue
        The array of gross revenues.

    """

    final_year: list[float] | list[int] | None = Field(default=None)
    utilized_land_area: list[float] | list[int] | None = Field(default=None)
    utilized_building_area: list[float] | list[int] | None = Field(default=None)
    njop_land: list[float] | list[int] | None = Field(default=None)
    njop_building: list[float] | list[int] | None = Field(default=None)
    gross_revenue: list[float] | list[int] | None = Field(default=None)


class OptimizationDictBM(_FiniteRequestModel):
    """
    The BaseModel to validate the Optimization Dictionary input data.

    Parameters
    ----------
    parameter: list[str]
        The list of string containing the parameter which will be obtained.
    min: list[float]
        The list of float containing the minimum boundary for the optimization variable.
    max: list[float]
        The list of float containing the maximum boundary for the optimization variable.

    """

    parameter: list[str] = Field(min_length=1)
    min: list[float] | list[int] = Field(min_length=1)
    max: list[float] | list[int] = Field(min_length=1)


class OptimizationBM(_FiniteRequestModel):
    """
    The BaseModel to validate the Optimization input data.

    Parameters
    ----------
    dict_optimization: OptimizationDictBM
        The dictionary containing the optimization parameter
    target_optimization: float
        The value of the optimization target.
    target_parameter: str
        The targeted optimization parameter.
    """

    dict_optimization: OptimizationDictBM
    target_optimization: float | int
    target_parameter: str


class SensitivityBM(_FiniteRequestModel):
    """
    The BaseModel to validate the Sensitivity input data.

    Parameters
    ----------
    min_deviation: float | int
        The minimum deviation from the base value.
    max_deviation: float | int
        The maximum deviation from the base value.
    step: int
        The number of steps to create multipliers. Default is 10.
    base_value: float | int
        The base value for the multipliers. Default is 1.0.
    increment: float | int | None
        Fixed spacing between consecutive multipliers. When provided, it
        overrides ``step`` and produces a uniform increment on both sides of
        the base value. Default is None.
    """

    min_deviation: float | int
    max_deviation: float | int
    step: int = 10
    base_value: float | int = 1.0
    increment: float | int | None = None


class UncertaintyBM(_FiniteRequestModel):
    """
    The BaseModel to validate the uncertainty input data.

    Parameters
    ----------
    run_number: int
        The number of the simulation.
    """

    run_number: int
    oil_price_distribution: str
    gas_price_distribution: str
    opex_distribution: str
    capex_distribution: str
    lifting_distribution: str
    min_oil_price: float | int | None = Field(default=None)
    mean_oil_price: float | int | None = Field(default=None)
    max_oil_price: float | int | None = Field(default=None)
    min_gas_price: float | int | None = Field(default=None)
    mean_gas_price: float | int | None = Field(default=None)
    max_gas_price: float | int | None = Field(default=None)
    min_opex: float | int | None = Field(default=None)
    mean_opex: float | int | None = Field(default=None)
    max_opex: float | int | None = Field(default=None)
    min_capex: float | int | None = Field(default=None)
    mean_capex: float | int | None = Field(default=None)
    max_capex: float | int | None = Field(default=None)
    min_lifting: float | int | None = Field(default=None)
    mean_lifting: float | int | None = Field(default=None)
    max_lifting: float | int | None = Field(default=None)
    oil_price_stddev: float | int = Field(default=1.25)
    gas_price_stddev: float | int = Field(default=1.25)
    opex_stddev: float | int = Field(default=1.25)
    capex_stddev: float | int = Field(default=1.25)
    lifting_stddev: float | int = Field(default=1.25)


class LtpBM(_FiniteRequestModel):
    """
    The BaseModel to validate the LTP data input .

    Parameters
    ----------
    volume: float | int
        The volume of the reserves.
    start_year: int
        The start year.
    end_year: int
        The end year.
    fluid_type: str
        The fluid type of the corresponding volume. Should be "Oil" or "Gas".

    """

    volume: float | int
    start_year: int
    end_year: int
    fluid_type: str


class RpdBM(_FiniteRequestModel):
    """
    The BaseModel to validate the RPD data input .

    Parameters
    ----------
    year_rampup: int
        Number of year from onstream to peak/plateau rate (yr).
    drate: float
        Arps yearly decline rate (1/yr).
    q_plateau_ratio: float
        Ratio of plateau rate and volume (1/yr).
    q_min_ratio: float
        Ratio of minimum rate at abandoned year and volume (1/yr).
    volume: float | int
        The volume of the reserves.
    start_year: int
        The start year.
    end_year: int
        The end year.
    """

    year_rampup: int
    drate: float | int
    q_plateau_ratio: float | int
    q_min_ratio: float | int
    volume: float | int
    start_year: int
    end_year: int


class Data(_FiniteRequestModel):
    """
    The BaseModel to validate the Data input.

    Parameters
    ----------
    setup: SetupBM
        The setup input in form of SetupBM BaseModel.
    summary_arguments: SummaryArgumentsBM
        The summary arguments in form of SummaryArgumentsBM BaseModel.
    contract_arguments: ContractArgumentsBM
        The contract arguments in form of ContractArgumentsBM BaseModel.
    lifting: Dict[str, LiftingBM]
        The lifting input in form of LiftingBM BaseModel.
    capital: Dict[str, CapitalCostBM]
        The capital input in form of TangibleBM BaseModel.
    intangible: Dict[str, IntangibleBM]
        The intangible input in form of IntangibleBM BaseModel.
    opex: Dict[str, OpexBM]
        The opex input in form of OpexBM BaseModel.
    asr: Dict[str, AsrBM]
        The asr input in form of AsrBM BaseModel.
    lbt: Dict[str, AsrBM]
        The lbt input in form of LBT BaseModel.
    cost_of_sales: Dict[str, CosOfSalesBM]
        The asr input in form of Cost of Sales BaseModel.
    optimization_arguments: OptimizationBM = None
        The optimization arguments in form of OptimizationBM BaseModel.
    sensitivity_arguments: SensitivityBM = None
        The sensitivity arguments in form of SensitivityBM BaseModel.
    uncertainty_arguments: UncertaintyBM = None
        The uncertainty arguments in form of UncertaintyBM BaseModel.
    costrecovery: CostRecoveryBM = None
        The cost recovery object in form of CostRecoveryBM BaseModel.
    grosssplit: GrossSplitBM = None
        The gross split object in form of GrossSplit BaseModel.
    result: dict = None
        The result of the running contract.
    """

    setup: SetupBM
    summary_arguments: SummaryArgumentsBM
    contract_arguments: ContractArgumentsBM
    lifting: dict[str, LiftingBM] | None = Field(default=None)
    capital: dict[str, CapitalCostBM] | None = Field(default=None)
    intangible: dict[str, IntangibleBM] | None = Field(default=None)
    opex: dict[str, OpexBM] | None = Field(default=None)
    asr: dict[str, AsrBM] | None = Field(default=None)
    lbt: dict[str, LbtBM] | None = Field(default=None)
    cost_of_sales: dict[str, CostOfSalesBM] | None = Field(default=None)
    optimization_arguments: OptimizationBM | None = Field(default=None)
    sensitivity_arguments: SensitivityBM | None = Field(default=None)
    uncertainty_arguments: UncertaintyBM | None = Field(default=None)
    costrecovery: CostRecoveryBM | None = Field(default=None)
    grosssplit: GrossSplitBM | None = Field(default=None)
    result: dict[str, object] | None = Field(default=None)


class TransitionBM(_FiniteRequestModel):
    """
    The BaseModel to validate the contract in the Transition input data.

    Parameters
    ----------
    setup: SetupBM
        The setup input in form of SetupBM BaseModel.
    contract_arguments: ContractArgumentsBM
        The contract arguments in form of ContractArgumentsBM BaseModel.
    lifting: Dict[str, LiftingBM]
        The lifting input in form of LiftingBM BaseModel.
    capital: Dict[str, CapitalCostBM]
        The tangible input in form of TangibleBM BaseModel.
    intangible: Dict[str, IntangibleBM]
        The intangible input in form of IntangibleBM BaseModel.
    opex: Dict[str, OpexBM]
        The opex input in form of OpexBM BaseModel.
    asr: Dict[str, AsrBM]
        The asr input in form of AsrBM BaseModel.
    costrecovery: CostRecoveryBM = None
        The cost recovery object in form of CostRecoveryBM BaseModel.
    grosssplit: GrossSplitBM = None
        The gross split object in form of GrossSplit BaseModel.
    """

    setup: SetupBM
    contract_arguments: ContractArgumentsBM
    lifting: dict[str, LiftingBM]
    capital: dict[str, CapitalCostBM]
    intangible: dict[str, IntangibleBM]
    opex: dict[str, OpexBM]
    asr: dict[str, AsrBM]
    lbt: dict[str, LbtBM]
    cost_of_sales: dict[str, CostOfSalesBM] | None = None
    costrecovery: CostRecoveryBM | None = None
    grosssplit: GrossSplitBM | None = None


class DataTransition(_FiniteRequestModel):
    """
    The BaseModel to validate the Transition Data input.

    Parameters
    ----------
    contract_1: TransitionBM
        The first contract object in form of TransitionBM
    contract_2: TransitionBM
        The second contract object in form of TransitionBM
    contract_arguments: ContractArgumentsTransitionBM
        The contract arguments in form of ContractArgumentsBM BaseModel.
    summary_arguments: SummaryArgumentsBM
        The summary arguments in form of SummaryArgumentsBM BaseModel.
    uncertainty_arguments: UncertaintyBM = None
        The uncertainty arguments in form of UncertaintyBM BaseModel.
    result: dict = None
        The result of the running contract.
    """

    contract_1: TransitionBM
    contract_2: TransitionBM
    contract_arguments: ContractArgumentsTransitionBM
    summary_arguments: SummaryArgumentsBM
    result: dict[str, object] | None = None
    optimization_arguments: OptimizationBM | None = None
    sensitivity_arguments: SensitivityBM | None = None
    uncertainty_arguments: UncertaintyBM | None = None


class EconLimit(_FiniteRequestModel):
    """
    The BaseModel to validate the Economic Limit.

    Parameters
    ----------
    """

    years: list[int]
    cash_flow: list[int] | list[float]
    method: str


class ASRExpendituresBM(_FiniteRequestModel):
    """
    The BaseModel to validate the ASR Expenditures Data.

    Parameters
    ----------
    start_date : str | int
        The start date of the project.
    end_date : str | int
        The end date of the project.
    asr : AsrBM
        The asr input in form of AsrBM BaseModel.
    """

    start_date: str | int = "01/01/2010"
    end_date: str | int = "01/01/2010"
    asr: AsrBM


class LBTExpendituresBM(_FiniteRequestModel):
    """
    The BaseModel to validate the LBT Expenditures Data.

    Parameters
    ----------
    start_date : str | int
        The start date of the project.
    end_date : str | int
        The end date of the project.
    lbt : LbtBM
        The lbt input in form of LBT BaseModel.
    """

    start_date: str | int = "01/01/2010"
    end_date: str | int = "01/01/2010"
    lbt: LbtBM


def convert_str_to_date(str_object: object) -> date | None:
    """
    The function to convert string or integer unix timestamp format object into
    dateformat.

    Parameters
    ----------
    str_object: str | int
        The object which will be converted into date format

    Returns
    -------
    date
    """

    if str_object is None:
        return None
    if isinstance(str_object, str):
        # Reuse the shared engine date rule so preflight and the adapter parse
        # exactly the same format; a malformed date raises here as before.
        parsed = parse_engine_date(str_object)
        if parsed is None:
            raise ValueError(
                f"time data '{str_object}' does not match format '%d/%m/%Y'"
            )
        return parsed

    if isinstance(str_object, int):
        value = datetime.fromtimestamp(str_object)
        date_value = value.strftime("%d/%m/%Y")
        return datetime.strptime(date_value, "%d/%m/%Y").date()
    return None


def convert_str_to_int(str_object: object) -> int | None:
    """
    Convert a string representation of an integer into an integer type.

    If the input string is ``None``, the function returns ``None`` instead of
    raising an error. Otherwise, it converts the string to an integer using
    the built-in ``int()`` function.

    Parameters
    ----------
    str_object : str
        The string to be converted into an integer. If ``None``, no conversion
        is performed.

    Returns
    -------
    int or None
        The integer converted from the input string, or ``None`` if the input
        is ``None``.
    """

    if str_object is None:
        return None

    if isinstance(str_object, str | int):
        return int(str_object)
    raise ConverterTypeError(
        f"Expected str or int, got {type(str_object).__name__}",
        code="CONVERTER_INT_TYPE",
    )


def convert_list_to_array_float(
    data_list: object,
) -> npt.NDArray[np.float64]:
    """
    The function to convert list into array of float.

    Parameters
    ----------
    data_list: list
        The list which will be converted into numpy array.

    Returns
    -------
    array
        np.ndarray
    """
    return np.array(data_list, dtype=float)


def convert_list_to_array_float_or_array(
    data_input: object,
) -> npt.NDArray[np.float64] | float:
    """
    The function to convert list or float into float or array of float.

    Parameters
    ----------
    data_input: list | float
        The list or float which will be converted

    Returns
    -------
    output
        float | np.ndarray
    """
    if isinstance(data_input, float):
        return data_input

    if isinstance(data_input, int):
        return float(data_input)
    if isinstance(data_input, str):
        return float(data_input) if data_input.strip() != "" else 0
    return np.array(data_input, dtype=float)


def convert_list_to_array_float_or_array_or_none(
    data_list: object,
) -> npt.NDArray[np.float64] | float | None:
    """
    Function to convert list into array of float, None or array.

    Parameters
    ----------
    data_list
        The list which will be converted.

    Returns
    -------
    out
        np.ndarray | float | None

    """
    if isinstance(data_list, float):
        return data_list
    if isinstance(data_list, int):
        return float(data_list)
    if isinstance(data_list, str):
        return float(data_list) if data_list.strip() != "" else None
    if data_list is None:
        return None
    return np.array(data_list, dtype=float)


def convert_list_to_array_int(data_list: object) -> npt.NDArray[np.int_]:
    """
    The function to convert list into array of integer.

    Parameters
    ----------
    data_list: list
        The list which will be converted into numpy array of integer.

    Returns
    -------
    out
        np. array

    """
    return np.array(data_list, dtype=int)


def convert_dict_to_lifting(
    data_raw: object,
) -> tuple[Lifting, ...] | None:
    """
    The function to convert dictionary into tuple of Lifting dataclass.

    Parameters
    ----------
    data_raw: dict
        The dictionary which will be converted into tuple of Lifting


    Returns
    -------
    out:
        tuple[Lifting]
    """
    if data_raw is None:
        return None
    raw = _validate_string_object_mapping(data_raw)
    if raw is None:
        return None
    return assign_lifting(data_raw=raw)


def convert_dict_to_capital(
    data_raw: object,
) -> tuple[CapitalCost, ...] | None:
    """
    Convert a dictionary of capital cost data into a tuple of ``CapitalCost``
    dataclass instances.

    This function iterates through each key-value pair in the input dictionary and
    constructs a corresponding ``CapitalCost`` object using the available fields.
    Missing or ``None`` values in optional fields are handled gracefully by assigning
    default values (``None`` or ``0.0`` depending on context).

    Parameters
    ----------
    data_raw : dict
        A dictionary containing capital cost information for one or more cost items.
        Each key in the dictionary represents a cost component, and its value should
        itself be a dictionary with the following possible keys:

        - ``start_year`` : int
          The first year of the capital cost occurrence.
        - ``end_year`` : int
          The last year of the capital cost occurrence.
        - ``cost`` : array-like
          The cost values corresponding to each ``expense_year``.
        - ``expense_year`` : array-like of int
          The years during which the costs are incurred.
        - ``cost_type`` : str or None, optional
          The type of cost (converted via ``read_cost_type`` if provided).
        - ``cost_allocation`` : str or None, optional
          The cost allocation type (converted via ``read_fluid_type`` if provided).
        - ``description`` : str or None, optional
          A brief description of the cost item.
        - ``tax_portion`` : array-like of float, optional
          The tax portion applied to each cost element.
        - ``tax_discount`` : array-like of float, optional
          The tax discount factor; defaults to 0.0 if missing.
        - ``pis_year`` : array-like, optional
          The year(s) in which project investment spending (PIS) occurs.
        - ``salvage_value`` : array-like, optional
          The salvage value associated with the cost item.
        - ``useful_life`` : array-like, optional
          The economic life of the asset in years.
        - ``depreciation_factor`` : array-like, optional
          The factor used in calculating depreciation.
        - ``is_ic_applied`` : bool or None, optional
          Indicates whether investment credit (IC) is applied.

    Returns
    -------
    tuple of CapitalCost or None
        A tuple containing ``CapitalCost`` instances for each key in ``data_raw``.
        Returns ``None`` if ``data_raw`` is ``None``.

    Examples
    --------
    # >>> data = {
    # ...     "Platform A": {
    # ...         "start_year": 2020,
    # ...         "end_year": 2025,
    # ...         "cost": [1000, 1200, 800],
    # ...         "expense_year": [2020, 2021, 2022],
    # ...         "cost_type": "CAPEX",
    # ...         "useful_life": [10],
    # ...     }
    # ... }
    # >>> convert_dict_to_capital(data)
    # (<CapitalCost(start_year=2020, end_year=2025, ... )>,)
    """

    raw = _validate_cost_input_mapping(data_raw)
    capital = (
        tuple([CapitalCost(**_capital_cost_arguments(item)) for item in raw.values()])
        if raw is not None
        else None
    )

    return capital


def convert_dict_to_intangible(
    data_raw: object,
) -> tuple[Intangible, ...] | None:
    """
    Convert a dictionary of intangible cost data into a tuple of ``Intangible``
    dataclass instances.

    This function iterates through each key-value pair in the input dictionary and
    constructs a corresponding ``Intangible`` object using the available fields.
    Optional or missing fields are handled gracefully with default values
    (``None`` or ``0.0``).

    NumPy arrays are used for numerical and sequence data fields to ensure consistency
    in subsequent calculations.

    Parameters
    ----------
    data_raw : dict
        A dictionary containing intangible cost data for one or more cost components.
        Each key in the dictionary represents an intangible cost item, and its value
        should itself be a dictionary with the following possible keys:

        - ``start_year`` : int
          The first year of the intangible cost occurrence.
        - ``end_year`` : int
          The last year of the intangible cost occurrence.
        - ``cost`` : array-like of float
          The cost values corresponding to each ``expense_year``.
        - ``expense_year`` : array-like of int
          The years during which the costs are incurred.
        - ``cost_type`` : str or None, optional
          The type of cost (converted using ``read_cost_type`` if provided).
        - ``cost_allocation`` : str or None, optional
          The cost allocation type (converted using ``read_fluid_type`` if provided).
        - ``description`` : str or None, optional
          A brief description of the intangible cost item.
        - ``tax_portion`` : array-like of float, optional
          The tax portion applied to each cost element.
        - ``tax_discount`` : array-like of float, optional
          The tax discount factor; defaults to 0.0 if missing.

    Returns
    -------
    tuple of Intangible or None
        A tuple containing ``Intangible`` instances for each key in ``data_raw``.
        Returns ``None`` if ``data_raw`` is ``None``.

    Examples
    --------
    # >>> data = {
    # ...     "Seismic Study": {
    # ...         "start_year": 2020,
    # ...         "end_year": 2022,
    # ...         "cost": [500, 400, 300],
    # ...         "expense_year": [2020, 2021, 2022],
    # ...         "cost_type": "INTANGIBLE",
    # ...         "tax_portion": [0.8, 0.8, 0.8],
    # ...     }
    # ... }
    # >>> convert_dict_to_intangible(data)
    # (<Intangible(start_year=2020, end_year=2022, ... )>,)
    # >>> convert_dict_to_intangible(None)
    # None
    """

    raw = _validate_cost_input_mapping(data_raw)
    intangible = (
        tuple([Intangible(**_intangible_arguments(item)) for item in raw.values()])
        if raw is not None
        else None
    )

    return intangible


def convert_dict_to_opex(
    data_raw: object,
) -> tuple[OPEX, ...] | None:
    """
    Convert a dictionary of operating expense (OPEX) data into a tuple of
    ``OPEX`` dataclass instances.

    This function iterates through each key-value pair in the input dictionary and
    constructs a corresponding ``OPEX`` object using the available fields.
    Optional or missing fields are handled gracefully by assigning default values
    (``None`` or ``0.0``).

    Numerical and sequence data are converted to NumPy arrays for consistency in
    subsequent economic calculations.

    Parameters
    ----------
    data_raw : dict
        A dictionary containing OPEX data for one or more cost components.
        Each key in the dictionary represents an operating cost item, and its value
        should itself be a dictionary with the following possible keys:

        - ``start_year`` : int
          The first year of the OPEX occurrence.
        - ``end_year`` : int
          The last year of the OPEX occurrence.
        - ``expense_year`` : array-like of int
          The years during which the OPEX costs are incurred.
        - ``cost_type`` : str or None, optional
          The type of cost (converted using ``read_cost_type`` if provided).
        - ``cost_allocation`` : str or None, optional
          The cost allocation type (converted using ``read_fluid_type`` if provided).
        - ``description`` : str or None, optional
          A short description of the OPEX component.
        - ``tax_portion`` : array-like of float, optional
          The tax portion applied to each OPEX element.
        - ``tax_discount`` : array-like of float, optional
          The tax discount factor; defaults to 0.0 if missing.
        - ``fixed_cost`` : array-like of float, optional
          Fixed costs incurred each year, independent of production rate.
        - ``prod_rate`` : array-like of float, optional
          The production rate associated with the cost calculation.
        - ``cost_per_volume`` : array-like of float, optional
          The variable cost per unit of production (e.g., $/BOE).

    Returns
    -------
    tuple of OPEX or None
        A tuple containing ``OPEX`` instances for each key in ``data_raw``.
        Returns ``None`` if ``data_raw`` is ``None``.

    Examples
    --------
    # >>> data = {
    # ...     "Field Operation": {
    # ...         "start_year": 2023,
    # ...         "end_year": 2030,
    # ...         "expense_year": [2023, 2024, 2025],
    # ...         "fixed_cost": [1000, 1000, 1000],
    # ...         "cost_per_volume": [2.5, 2.4, 2.3],
    # ...         "prod_rate": [500, 600, 700],
    # ...         "cost_type": "OPEX",
    # ...     }
    # ... }
    # >>> convert_dict_to_opex(data)
    # (<OPEX(start_year=2023, end_year=2030, ... )>,)
    # >>> convert_dict_to_opex(None)
    # None
    """

    raw = _validate_cost_input_mapping(data_raw)
    opex = (
        tuple([OPEX(**_opex_arguments(item)) for item in raw.values()])
        if raw is not None
        else None
    )

    return opex


def convert_dict_to_asr(
    data_raw: object,
) -> tuple[ASR, ...] | None:
    """
    Convert a dictionary of Abandonment, Site Restoration (ASR) cost data into
    a tuple of ``ASR`` dataclass instances.

    This function iterates through each key-value pair in the input dictionary
    and constructs a corresponding ``ASR`` object using the provided fields.
    Missing or optional fields are handled gracefully with default values
    (``None`` or ``0.0``). All numerical and sequence data are converted to
    NumPy arrays to maintain consistency in subsequent economic evaluations.

    Parameters
    ----------
    data_raw : dict
        A dictionary containing ASR (Abandonment and Site Restoration) cost data
        for one or more cost components.
        Each key in the dictionary represents a specific ASR cost item, and its value
        should itself be a dictionary with the following possible keys:

        - ``start_year`` : int
          The first year of the ASR cost occurrence.
        - ``end_year`` : int
          The last year of the ASR cost occurrence.
        - ``cost`` : array-like of float
          The ASR cost values corresponding to each ``expense_year``.
        - ``expense_year`` : array-like of int
          The years during which ASR costs are incurred.
        - ``cost_type`` : str or None, optional
          The type of cost (converted using ``read_cost_type`` if provided).
        - ``cost_allocation`` : str or None, optional
          The cost allocation type (converted using ``read_fluid_type`` if provided).
        - ``description`` : str or None, optional
          A short description of the ASR item.
        - ``tax_portion`` : array-like of float, optional
          The tax portion applied to each ASR cost.
        - ``tax_discount`` : array-like of float, optional
          The tax discount factor; defaults to 0.0 if not provided.
        - ``final_year`` : array-like of float, optional
          The final year(s) associated with abandonment and site restoration activities.
        - ``future_rate`` : array-like of float, optional
          The escalation or discount rate applied to estimate future ASR costs.

    Returns
    -------
    tuple of ASR or None
        A tuple containing ``ASR`` instances for each key in ``data_raw``.
        Returns ``None`` if ``data_raw`` is ``None``.

     Examples
    --------
    # >>> data = {
    # ...     "Well Abandonment": {
    # ...         "start_year": 2030,
    # ...         "end_year": 2032,
    # ...         "cost": [5000, 7000],
    # ...         "expense_year": [2030, 2031],
    # ...         "final_year": [2032],
    # ...         "future_rate": [0.03],
    # ...         "cost_type": "ASR",
    # ...     }
    # ... }
    # >>> convert_dict_to_asr(data)
    # (<ASR(start_year=2030, end_year=2032, ... )>,)
    # >>> convert_dict_to_asr(None)
    # None
    """
    raw = _validate_cost_input_mapping(data_raw)
    asr = (
        tuple([ASR(**_asr_arguments(item)) for item in raw.values()])
        if raw is not None
        else None
    )

    return asr


def convert_dict_to_lbt(
    data_raw: object,
) -> tuple[LBT, ...] | None:
    """
    Convert a raw input dictionary into a tuple of `LBT` objects.

    This function parses a nested dictionary containing Land and Building Tax (LBT)
    data into a structured tuple of `LBT` class instances. Each dictionary entry
    must contain the relevant attributes for the `LBT` object, including cost data,
    years, tax parameters, and NJOP-related fields. Missing or `None` fields are
    handled gracefully, returning `None` or default values as appropriate.

    Parameters
    ----------
    data_raw : dict
        A dictionary containing the raw LBT data to be converted.
        Each key represents a record, and its value must be a dictionary with the
        following structure:

        - ``start_year`` : int
          The start year of the LBT cost period.
        - ``end_year`` : int
          The end year of the LBT cost period.
        - ``cost`` : array_like or None
          The cost values associated with LBT.
        - ``expense_year`` : array_like of int
          The corresponding expense years.
        - ``cost_type`` : str or None
          The type of cost (converted via ``read_cost_type``).
        - ``cost_allocation`` : str or None
          The cost allocation type (converted via ``read_fluid_type``).
        - ``description`` : str or None
          Optional description of the cost item.
        - ``tax_portion`` : array_like of float or None
          The tax portion applicable to each expense.
        - ``tax_discount`` : float or array_like of float
          The tax discount rate; defaults to ``0.0`` if missing.
        - ``final_year`` : array_like of float or None
          The final year values associated with the LBT computation.
        - ``utilized_land_area`` : array_like of float or None
          The area of land utilized, in square meters.
        - ``utilized_building_area`` : array_like of float or None
          The area of building utilized, in square meters.
        - ``njop_land`` : array_like of float or None
          The NJOP (taxable value) for land.
        - ``njop_building`` : array_like of float or None
          The NJOP (taxable value) for building.
        - ``gross_revenue`` : array_like of float or None
          The gross revenue associated with the LBT calculation.

    Returns
    -------
    tuple of LBT or None
        A tuple containing instances of the `LBT` class, each populated with the
        relevant attributes extracted from ``data_raw``. Returns ``None`` if
        ``data_raw`` is ``None``.

    Examples
    --------
    # >>> data_raw = {
    # ...     "item1": {
    # ...         "start_year": 2020,
    # ...         "end_year": 2025,
    # ...         "cost": [1000, 1200, 1400],
    # ...         "expense_year": [2020, 2021, 2022],
    # ...         "tax_discount": [0.05, 0.05, 0.05],
    # ...         "njop_land": [200, 220, 250],
    # ...         "njop_building": [400, 420, 450],
    # ...     }
    # ... }
    # >>> convert_dict_to_lbt(data_raw)
    # (<LBT object at 0x...>,)
    """
    raw = _validate_cost_input_mapping(data_raw)
    lbt = (
        tuple([LBT(**_lbt_arguments(item)) for item in raw.values()])
        if raw is not None
        else None
    )

    return lbt


def convert_dict_to_cost_of_sales(
    data_raw: object,
) -> tuple[CostOfSales, ...] | None:
    """
    Convert a raw dictionary into a tuple of `CostOfSales` dataclass instances.

    This function transforms a nested input dictionary containing cost-of-sales data
    into a structured tuple of `CostOfSales` objects. Each key in the input dictionary
    represents a cost record, while its value contains attributes such as cost,
    expense year, cost type, and tax-related information. Missing or ``None`` fields
    are handled gracefully with default or ``None`` values.

    Parameters
    ----------
    data_raw : dict
        A dictionary containing raw cost-of-sales data to be converted.
        Each key corresponds to a record name or identifier, and each value
        must be a dictionary with the following structure:

        - ``start_year`` : int
          The starting year of the cost-of-sales period.
        - ``end_year`` : int
          The ending year of the cost-of-sales period.
        - ``cost`` : array_like of float
          Cost values associated with each expense year.
        - ``expense_year`` : array_like of int
          The corresponding years in which the expenses occurred.
        - ``cost_type`` : str or None
          The type of cost, converted using :func:`read_cost_type`.
        - ``cost_allocation`` : str or None
          The cost allocation type, converted using :func:`read_fluid_type`.
        - ``description`` : str or None
          An optional description of the cost item.
        - ``tax_portion`` : array_like of float or None
          The portion of cost subject to tax.
        - ``tax_discount`` : float or array_like of float, default 0.0
          The applicable tax discount. Defaults to ``0.0`` if missing.

    Returns
    -------
    tuple of CostOfSales or None
        A tuple containing instances of the `CostOfSales` dataclass.
        Returns ``None`` if ``data_raw`` is ``None``.

    Examples
    --------
    # >>> data_raw = {
    # ...     "record1": {
    # ...         "start_year": 2020,
    # ...         "end_year": 2023,
    # ...         "cost": [500, 600, 700],
    # ...         "expense_year": [2020, 2021, 2022],
    # ...         "cost_type": "Operating",
    # ...         "tax_discount": [0.05, 0.05, 0.05]
    # ...     }
    # ... }
    # >>> convert_dict_to_cost_of_sales(data_raw)
    # (<CostOfSales object at 0x...>,)
    """

    raw = _validate_cost_input_mapping(data_raw)
    cos = (
        tuple(
            [
                CostOfSales(
                    start_year=item["start_year"],
                    end_year=item["end_year"],
                    cost=np.array(item.get("cost"), dtype=np.float64),
                    expense_year=np.array(item["expense_year"], dtype=np.int64),
                    cost_type=_optional_cost_types(item.get("cost_type")),
                    cost_allocation=_optional_allocations(item.get("cost_allocation")),
                    description=item.get("description"),
                    tax_portion=_optional_float_array(item.get("tax_portion")),
                    tax_discount=_tax_discount(item),
                )
                for item in raw.values()
            ]
        )
        if raw is not None
        else None
    )

    return cos


def convert_str_to_taxsplit(str_object: object) -> TaxSplitTypeCR | None:
    """
    The function to converts a string representing a tax split type to its corresponding
    enum value from the TaxSplitTypeCR class.

    Parameters
    ----------
    str_object: str
        The string representation of the tax split type.

    Returns
    -------
    Union: TaxSplitTypeCR, None
        The corresponding enum value if the target matches one of the predefined split types.
        Returns None if no match is found.

    Example
    -------
    >>> result = get_split_type_converter("RC Split")
    >>> print(result)
    TaxSplitTypeCR.R2C
    """
    return (
        get_split_type_converter(target=str_object)
        if isinstance(str_object, str)
        else None
    )


def convert_str_to_npvmode(str_object: object) -> NPVSelection | None:
    """
    The function to converts a string representing an NPV mode to its corresponding
    enum value from the NPVSelection class.

    Parameters
    ----------
    str_object: str
        The string representation of the NPV mode.

    Returns
    -------
    Union[NPVSelection, None]
        The corresponding enum value if the target matches one of the predefined NPV modes.
        Returns None if no match is found.

    Example
    -------
    >>> result = get_npv_mode_converter("SKK Full Cycle Real Terms")
    >>> print(result)
    NPVSelection.NPV_SKK_REAL_TERMS
    """
    return (
        get_npv_mode_converter(target=str_object)
        if isinstance(str_object, str)
        else None
    )


def convert_str_to_discountingmode(str_object: object) -> DiscountingMode | None:
    """
    The function to converts a string representing a discounting mode to its corresponding
    enum value from the DiscountingMode class.

    Parameters
    ----------
    str_object: str
        The string representation of the discounting mode.

    Returns
    -------
        Union[DiscountingMode, None]
        The corresponding enum value if the target matches one of the predefined discounting modes.
        Returns None if no match is found.

    Example
    -------
    >>> result = get_discounting_mode_converter("End Year")
    >>> print(result)
    DiscountingMode.END_YEAR

    """
    return (
        get_discounting_mode_converter(target=str_object)
        if isinstance(str_object, str)
        else None
    )


def convert_str_to_otherrevenue(str_object: object) -> OtherRevenue | None:
    """
    The function to convert a string representation of other revenue types
    to the corresponding enum value.

    Parameters
    ----------
    str_object: str
        The string representation of the other revenue type.

    Returns
    -------
    OtherRevenue or None
        The enum value corresponding to the provided string representation
        of other revenue.Returns None if the string does not match any known
        other revenue types.

    Notes
    -----
    -   The function uses a mapping of string representations to OtherRevenue
        enum values.
    -   If the target string matches a known other revenue type, the corresponding
        enum value is returned.
    - If the target string does not match any known types, the function returns None.
    """
    return (
        get_other_revenue_converter(target=str_object)
        if isinstance(str_object, str)
        else None
    )


def converter_sunk_cost_method(str_obj: object) -> SunkCostMethod:
    """
    Convert a string identifier to its corresponding SunkCostMethod enum member.

    Parameters
    ----------
    str_obj : str
        The string representation of the sunk cost method.
        Must be one of:
        - "depreciated_tangible"
        - "pooled_first_year"

    Returns
    -------
    SunkCostMethod
        The corresponding enum member from `SunkCostMethod`.

    Raises
    ------
    ValueError
        If `str_obj` does not match any known sunk cost method.
    """

    try:
        if not isinstance(str_obj, str):
            raise ConverterInputError(
                f"Invalid sunk cost method: {str_obj!r}",
                code="CONVERTER_SUNK_COST_METHOD",
            )
        return {
            "depreciated_tangible": SunkCostMethod.DEPRECIATED_TANGIBLE,
            "pooled_first_year": SunkCostMethod.POOLED_1ST_YEAR,
        }[str_obj]

    except KeyError:
        raise ConverterInputError(
            f"Invalid sunk cost method: {str_obj!r}",
            code="CONVERTER_SUNK_COST_METHOD",
        )


def converter_reservoir_type_permen_2024(
    target_str: object,
) -> VariableSplit132024.ReservoirType:
    """
    Convert a string identifier to its corresponding reservoir type defined in
    Permen ESDM No.13/2024.

    Parameters
    ----------
    target_str : str
        The string representation of the reservoir type.
        Must be one of:
        - ``"conventional"`` → MK (Migas Konvensional)
        - ``"unconventional"`` → MNK (Migas Non-Konvensional)

    Returns
    -------
    VariableSplit132024.ReservoirType
        The corresponding enum member from
        ``VariableSplit132024.ReservoirType``.

    Raises
    ------
    ValueError
        If the input string does not match any valid reservoir type
        defined in Permen ESDM No.13/2024.
    """

    try:
        if not isinstance(target_str, str):
            raise ConverterInputError(
                f"Invalid reservoir type permen 2024 method: {target_str!r}",
                code="CONVERTER_RESERVOIR_TYPE_2024",
            )
        return {
            "conventional": VariableSplit132024.ReservoirType.MK,
            "unconventional": VariableSplit132024.ReservoirType.MNK,
        }[target_str]

    except KeyError:
        raise ConverterInputError(
            f"Invalid reservoir type permen 2024 method: {target_str!r}",
            code="CONVERTER_RESERVOIR_TYPE_2024",
        )


def converter_initial_amortization_year(
    target_str: object,
) -> InitialYearAmortizationIncurred:
    """
    Convert a string identifier to its corresponding initial amortization
    year option.

    Parameters
    ----------
    target_str : str
        The string representation of the initial amortization year option.
        Must be one of:
        - ``"onstream_year"`` → amortization starts in the onstream year.
        - ``"approval_year"`` → amortization starts in the approval year.

    Returns
    -------
    InitialYearAmortizationIncurred
        The corresponding enum member from
        ``InitialYearAmortizationIncurred``.

    Raises
    ------
    ValueError
        If the input string does not match any valid initial amortization
        year option.
    """

    try:
        if not isinstance(target_str, str):
            raise ConverterInputError(
                f"Invalid initial amortization year option: {target_str!r}",
                code="CONVERTER_INITIAL_AMORTIZATION_YEAR",
            )
        return {
            "onstream_year": InitialYearAmortizationIncurred.ONSTREAM_YEAR,
            "approval_year": InitialYearAmortizationIncurred.APPROVAL_YEAR,
        }[target_str]

    except KeyError:
        raise ConverterInputError(
            f"Invalid initial amortization year option: {target_str!r}",
            code="CONVERTER_INITIAL_AMORTIZATION_YEAR",
        )


def convert_str_to_taxregime(str_object: object) -> TaxRegime | None:
    """
    The function to convert string into tax regime selection.

    Parameters
    ----------
    str_object: str
        The string object which will be converted

    Returns
    -------
    out:
        TaxRegime | None
    """
    dict_tax_regime = {i.value: i for i in TaxRegime}

    for key in dict_tax_regime.keys():
        if str_object == key:
            return dict_tax_regime[key]
        else:
            return None


def convert_str_to_ftptaxregime(str_object: object) -> FTPTaxRegime | None:
    """
    The functon to converts a string representing a tax regime to its corresponding
    enum value from the FTPTaxRegime class.

    Parameters
    ----------
    str_object: str
        The string representation of the tax regime.


    Returns
    -------
    Union[FTPTaxRegime, None]
        The corresponding enum value if the target matches one of the predefined tax regimes.
        Returns None if no match is found.

    """
    attrs = {
        "PDJP No.20 Tahun 2017": FTPTaxRegime.PDJP_20_2017,
        "Pre PDJP No.20 Tahun 2017": FTPTaxRegime.PRE_PDJP_20_2017,
        "Direct Mode": FTPTaxRegime.DIRECT_MODE,
    }

    for key in attrs.keys():
        if str_object == key:
            return attrs[key]


def convert_str_to_depremethod(str_object: object) -> DeprMethod | None:
    """
    The function to converts a string representing a depreciation method to its corresponding
    enum value from the DeprMethod class.

    Parameters
    ----------
    str_object: str
        The string representation of the depreciation method.

    Returns
    -------
    Union[DeprMethod, None]
        The corresponding enum value if the target matches one of the predefined depreciation methods.
        Returns None if no match is found.

    Example
    -------
    >>> result = get_depreciation_method_converter("PSC Declining Balance")
    >>> print(result)
    DeprMethod.PSC_DB

    """
    return (
        get_depreciation_method_converter(target=str_object)
        if isinstance(str_object, str)
        else None
    )


def convert_str_to_taxtype(str_object: object) -> TaxType | None:
    """
    The function to convert a string object into tax type selection.

    Parameters
    ----------
    str_object: str
        The string representation of tax type.

    Returns
    -------
    out:
        TaxType
    """
    dict_tax_regime = {i.value: i for i in TaxType}

    for key in dict_tax_regime.keys():
        if str_object == key:
            return dict_tax_regime[key]
        else:
            return None


def convert_str_to_inflationappliedto(
    str_object: object,
) -> InflationAppliedTo | None:
    """
    The function to converts a string representing the application of inflation to its
    corresponding enum value from the InflationAppliedTo class.

    Parameters
    ----------
    str_object: str
        The string representation of the inflation application.

    Returns
    -------
    Union[InflationAppliedTo, None]
        The corresponding enum value if the target matches one of the predefined options.
        Returns None if no match is found.

    Example:
    >>> result = get_inflation_applied_converter("OPEX")
    >>> print(result)
    InflationAppliedTo.OPEX

    """
    return (
        get_inflation_applied_converter(target=str_object)
        if isinstance(str_object, str)
        else None
    )


def convert_grosssplitregime_to_enum(target: object) -> GrossSplitRegime | None:
    """
    Converts a string representing the Gross Split Regime to its
    corresponding enum value from the GrossSplitRegime enum class.

    Parameters
    ----------
    target: str
        The string representation of the gross split regime.

    Returns
    -------

    """
    attrs = {
        "PERMEN_ESDM_8_2017": GrossSplitRegime.PERMEN_ESDM_8_2017,
        "PERMEN_ESDM_52_2017": GrossSplitRegime.PERMEN_ESDM_52_2017,
        "PERMEN_ESDM_20_2019": GrossSplitRegime.PERMEN_ESDM_20_2019,
        "PERMEN_ESDM_12_2020": GrossSplitRegime.PERMEN_ESDM_12_2020,
        "PERMEN_ESDM_13_2024": GrossSplitRegime.PERMEN_ESDM_13_2024,
    }

    for key in attrs.keys():
        if target == key:
            return attrs[key]


def convert_summary_to_dict(
    dict_object: Mapping[str, object],
) -> dict[str, object]:
    """
    The function to convert the summary into skk executive summary format.
    Parameters
    ----------
    dict_object: dict
        The dictionary representation of the summary.

    Returns
    -------
    out: dict
        The dictionary of skk executive summary.

    """
    summary_skk_format = {
        "lifting_oil": dict_object["lifting_oil"],
        "oil_wap": dict_object["oil_wap"],
        "lifting_gas": dict_object["lifting_gas"],
        "gas_wap": dict_object["gas_wap"],
        "gross_revenue": dict_object["gross_revenue"],
        "ctr_gross_share": dict_object["ctr_gross_share"],
        "sunk_cost": dict_object["sunk_cost"],
        "investment": dict_object["investment"],
        "tangible": dict_object["tangible"],
        "intangible": dict_object["intangible"],
        "opex_asr_lbt": dict_object["opex_asr_lbt"],
        "opex": dict_object["opex"],
        "asr": dict_object["asr"],
        "lbt": dict_object["lbt"],
        "cost_recovery/deductible_cost": dict_object["cost_recovery/deductible_cost"],
        "cost_recovery_over_gross_rev": dict_object["cost_recovery_over_gross_rev"],
        "unrec_cost": dict_object["unrec_cost"],
        "unrec_over_gross_rev": dict_object["unrec_over_gross_rev"],
        "ctr_net_share": dict_object["ctr_net_share"],
        "ctr_net_share_over_gross_share": dict_object["ctr_net_share_over_gross_share"],
        "ctr_net_cashflow": dict_object["ctr_net_cashflow"],
        "ctr_net_cashflow_over_gross_rev": dict_object[
            "ctr_net_cashflow_over_gross_rev"
        ],
        "ctr_npv": dict_object["ctr_npv"],
        "ctr_irr": dict_object["ctr_irr"],
        "ctr_pot": dict_object["ctr_pot"],
        "ctr_pv_ratio": dict_object["ctr_pv_ratio"],
        "ctr_pi": dict_object["ctr_pi"],
        "gov_gross_share": dict_object["gov_gross_share"],
        "gov_ftp_share": dict_object["gov_ftp_share"],
        "gov_ddmo": dict_object["gov_ddmo"],
        "gov_tax_income": dict_object["gov_tax_income"],
        "gov_take": dict_object["gov_take"],
        "gov_take_over_gross_rev": dict_object["gov_take_over_gross_rev"],
        "gov_take_npv": dict_object["gov_take_npv"],
        "undepreciated_asset_oil": dict_object["undepreciated_asset_oil"],
        "undepreciated_asset_gas": dict_object["undepreciated_asset_gas"],
        "undepreciated_asset_total": dict_object["undepreciated_asset_total"],
        "total_indirect_taxes": dict_object["total_indirect_taxes"],
        "oil_indirect_taxes": dict_object["oil_indirect_taxes"],
        "gas_indirect_taxes": dict_object["gas_indirect_taxes"],
        "total_carry_forward_depreciation": dict_object[
            "total_carry_forward_depreciation"
        ],
        "oil_carry_forward_depreciation": dict_object["oil_carry_forward_depreciation"],
        "gas_carry_forward_depreciation": dict_object["gas_carry_forward_depreciation"],
    }

    return summary_skk_format


def convert_to_skk_summary_baseproject(
    dict_object: Mapping[str, object],
) -> dict[str, object]:
    """
    Convert a BaseProject summary dictionary into SKK-compatible summary format.

    Parameters
    ----------
    dict_object : dict
        The summary dictionary generated from the BaseProject contract,
        containing various economic indicators such as lifting, cost recovery,
        government take, and contractor net share.

    Returns
    -------
    summary_skk : dict
        A dictionary containing the BaseProject summary formatted for SKK use.

    Notes
    -----
    This function standardizes the BaseProject output to ensure compatibility
    with SKK Migas reporting format and downstream processing within the
    economic evaluation framework.
    """

    return {
        "lifting_oil": dict_object["lifting_oil"],
        "oil_wap": dict_object["oil_wap"],
        "lifting_gas": dict_object["lifting_gas"],
        "gas_wap": dict_object["gas_wap"],
        "gross_revenue": dict_object["gross_revenue"],
        "ctr_gross_share": dict_object["ctr_gross_share"],
        "sunk_cost": dict_object["sunk_cost"],
        "investment": dict_object["investment"],
        "tangible": dict_object["tangible"],
        "intangible": dict_object["intangible"],
        "opex_asr_lbt": dict_object["opex_asr_lbt"],
        "opex": dict_object["opex"],
        "asr": dict_object["asr"],
        "lbt": dict_object["lbt"],
        "cost_recovery/deductible_cost": dict_object["cost_recovery/deductible_cost"],
        "cost_recovery_over_gross_rev": dict_object["cost_recovery_over_gross_rev"],
        "unrec_cost": dict_object["unrec_cost"],
        "unrec_over_gross_rev": dict_object["unrec_over_gross_rev"],
        "ctr_net_share": dict_object["ctr_net_share"],
        "ctr_net_share_over_gross_share": dict_object["ctr_net_share_over_gross_share"],
        "ctr_net_cashflow": dict_object["ctr_net_cashflow"],
        "ctr_net_cashflow_over_gross_rev": dict_object[
            "ctr_net_cashflow_over_gross_rev"
        ],
        "ctr_npv": dict_object["ctr_npv"],
        "ctr_irr": dict_object["ctr_irr"],
        "ctr_pot": dict_object["ctr_pot"],
        "ctr_pv_ratio": dict_object["ctr_pv_ratio"],
        "ctr_pi": dict_object["ctr_pi"],
        "gov_gross_share": dict_object["gov_gross_share"],
        "gov_ftp_share": dict_object["gov_ftp_share"],
        "gov_ddmo": dict_object["gov_ddmo"],
        "gov_tax_income": dict_object["gov_tax_income"],
        "gov_take": dict_object["gov_take"],
        "gov_take_over_gross_rev": dict_object["gov_take_over_gross_rev"],
        "gov_take_npv": dict_object["gov_take_npv"],
        "undepreciated_asset_oil": dict_object["undepreciated_asset_oil"],
        "undepreciated_asset_gas": dict_object["undepreciated_asset_gas"],
        "undepreciated_asset_total": dict_object["undepreciated_asset_total"],
        "total_indirect_taxes": dict_object["total_indirect_taxes"],
        "oil_indirect_taxes": dict_object["oil_indirect_taxes"],
        "gas_indirect_taxes": dict_object["gas_indirect_taxes"],
        "total_carry_forward_depreciation": 0.0,
        "oil_carry_forward_depreciation": 0.0,
        "gas_carry_forward_depreciation": 0.0,
    }


def convert_str_to_optimization_targetparameter(
    str_object: object,
) -> OptimizationTarget | None:
    """
    The function to converts a string representation of an optimization target to the corresponding
    OptimizationTarget enum value.

    Parameters
    ----------
    str_object: str
        The string representation of the optimization target.

    Returns
    -------
    Optional[OptimizationTarget]
        The OptimizationTarget enum value if the target is valid, or None if the target is not found.

    Example
    -------
    >>> get_optimization_target_converter("NPV")
    OptimizationTarget.NPV
    >>> get_optimization_target_converter("IRR")
    OptimizationTarget.IRR
    >>> get_optimization_target_converter("PI")
    OptimizationTarget.PI
    >>> get_optimization_target_converter("InvalidTarget")
    None
    """
    return (
        get_optimization_target_converter(target=str_object)
        if isinstance(str_object, str)
        else None
    )


def convert_str_to_optimization_parameters(
    str_object: object,
) -> OptimizationParameter | None:
    """
    The function to converts a string representation of an optimization parameter to the corresponding
    OptimizationParameter enum value.

    Parameters
    ----------
    str_object: str
        The string representation of the optimization parameter.

    Returns
    -------
    Union[OptimizationParameter, None]
        The OptimizationParameter enum value if the target is valid,
        or None if the target is not found.

    Example
    -------
    >>> get_optimization_parameter_converter("Oil Contractor Pre Tax")
    OptimizationParameter.OIL_CTR_PRETAX
    >>> get_optimization_parameter_converter("Gas FTP Portion")
    OptimizationParameter.GAS_FTP_PORTION
    >>> get_optimization_parameter_converter("VAT Rate")
    OptimizationParameter.VAT_RATE
    >>> get_optimization_parameter_converter("InvalidParameter")
    None

    """
    return (
        get_optimization_parameter_converter(target=str_object)
        if isinstance(str_object, str)
        else None
    )


def convert_to_float(
    target: object = int,
) -> float:
    """
    Function to convert integer into float.

    Parameters
    ----------
    target: int
        The target that will be converted.

    Returns
    -------
    float
    """
    if _is_float_convertible(target):
        return float(target)
    raise ConverterTypeError(
        f"Expected int, float, or str, got {type(target).__name__}",
        code="CONVERTER_FLOAT_TYPE",
    )


def convert_to_method_limit(target: object) -> LimitMethod | ValueError:
    """
    Function to convert string into LimitMethod selection.

    Parameters
    ----------
    target: dict
        The target that will be converted.

    Returns
    -------
    LimitMethod

    """
    if target == "Maximum Cumulative Cashflow":
        return LimitMethod.MAX_CUM_CASHFLOW
    elif target == "Maximum NPV":
        return LimitMethod.MAX_NPV
    elif target == "Negative Cashflow":
        return LimitMethod.NEGATIVE_CASHFLOW
    else:
        return ValueError("Invalid LimitMethod provided.")


def convert_to_uncertainty_distribution(target: object) -> UncertaintyDistribution:
    """
    Convert a string label into the corresponding distribution enum.

    Parameters
    ----------
    target : str
        The name of the uncertainty distribution.
        Accepted values are:
        - "Uniform"
        - "Triangular"
        - "Normal"
        - "LogNormal"

    Returns
    -------
    UncertaintyDistribution
        The corresponding `UncertaintyDistribution` enum value.

    Raises
    ------
    ValueError
        If the input string does not match any valid distribution type.
    """
    mapping = {
        "Uniform": UncertaintyDistribution.UNIFORM,
        "Triangular": UncertaintyDistribution.TRIANGULAR,
        "Normal": UncertaintyDistribution.NORMAL,
        "LogNormal": UncertaintyDistribution.LOGNORMAL,
    }

    if not isinstance(target, str) or target not in mapping:
        raise ConverterInputError(
            f"Invalid distribution: {target!r}. Expected one of {list(mapping.keys())}.",
            code="CONVERTER_UNCERTAINTY_DISTRIBUTION",
        )
    return mapping[target]


class CombineCaseBM(BaseModel):
    """
    BaseModel for one case in a combine/incremental request.

    ``data`` is the same direct JSON contract payload accepted by the
    canonical contract endpoints (setup, summary_arguments,
    contract_arguments, lifting, costs, and the scheme-specific block).
    """

    cId: int
    # ``ctr`` carries the case contract type. The backend convention is
    # 0=Base Project, 1=Cost Recovery, 2=Gross Split and any transition index
    # (3..6) as Transition; the real frontend ``Case`` enum uses 1..6, where
    # 3..6 are the four CR/GS transition combinations. All transition values
    # select the ``DataTransition`` contract model in nested validation.
    ctr: int = Field(ge=0, le=6)
    data: dict[str, object]


class CombineOptionsBM(_FiniteRequestModel):
    """
    Summary options shared by combine/incremental requests.

    Field names match the existing frontend combine/incremental option
    payload; values are the same strings used by the canonical summary
    endpoints (for example ``npv_mode`` and ``discounting_mode``).
    """

    reference_year: int
    inflation_rate: float | int = 0.0
    discount_rate: float | int = 0.1
    npv_mode: str = "SKK Full Cycle Nominal Terms"
    discounting_mode: str = "End Year"
    profitability_discounted: bool = True


class CasesRequestBM(BaseModel):
    """BaseModel for the combine/incremental request body."""

    cases: list[CombineCaseBM] = Field(min_length=1)
    options: CombineOptionsBM
