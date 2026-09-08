"""
This file is utilized to be the adapter of the router into the core codes.
"""

from collections.abc import Mapping
from datetime import date, datetime
import math
from typing import Literal, TypeGuard, TypeVar, TypedDict, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import TypeAdapter, ValidationError

from pyscnomics.api.converter import (
    convert_dict_to_asr,
    convert_dict_to_capital,
    convert_dict_to_cost_of_sales,
    convert_dict_to_intangible,
    convert_dict_to_lbt,
    convert_dict_to_lifting,
    convert_dict_to_opex,
    convert_grosssplitregime_to_enum,
    convert_list_to_array_float,
    convert_list_to_array_float_or_array,
    convert_list_to_array_float_or_array_or_none,
    convert_str_to_date,
    convert_str_to_depremethod,
    convert_str_to_discountingmode,
    convert_str_to_ftptaxregime,
    convert_str_to_inflationappliedto,
    convert_str_to_int,
    convert_str_to_npvmode,
    convert_str_to_optimization_parameters,
    convert_str_to_optimization_targetparameter,
    convert_str_to_otherrevenue,
    convert_str_to_taxregime,
    convert_str_to_taxsplit,
    convert_summary_to_dict,
    convert_to_float,
    convert_to_method_limit,
    convert_to_skk_summary_baseproject,
    convert_to_uncertainty_distribution,
    converter_reservoir_type_permen_2024,
)
from pyscnomics.contracts.project import require_float_array
from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.gross_split import GrossSplit
from pyscnomics.contracts.transition import Transition
from pyscnomics.dataset.sample import read_fluid_type
from pyscnomics.econ.costs import ASR, CapitalCost, CostOfSales, Intangible, LBT, OPEX
from pyscnomics.econ.revenue import Lifting
from pyscnomics.optimize import sensitivity_psc
from pyscnomics.optimize.optimization import optimize_psc
from pyscnomics.optimize.optimization_transition import optimize_psc_core
from pyscnomics.optimize.uncertainty import uncertainty_psc
from pyscnomics.tools.summary import get_summary
from pyscnomics.tools.table import (
    ContractTable,
    TransitionTable,
    get_table,
)
from pyscnomics.econ.selection import (
    DeprMethod,
    DiscountingMode,
    FTPTaxRegime,
    FluidType,
    GrossSplitRegime,
    InflationAppliedTo,
    LimitMethod,
    NPVSelection,
    OptimizationParameter,
    OtherRevenue,
    TaxRegime,
    TaxSplitTypeCR,
    VariableSplit132024,
)
from pyscnomics.tools.ltp import oil_ltp_predict, gas_ltp_predict
from pyscnomics.tools.rpd import RPDModel
from pyscnomics.tools.summarizer import Summary
from pyscnomics.econ.limit import econ_limit


_T = TypeVar("_T")
_FloatArray = npt.NDArray[np.float64]
_Number = float | int
_STRING_OBJECT_MAPPING = TypeAdapter(dict[str, object])


class _BaseProjectKwargs(TypedDict):
    start_date: date
    end_date: date
    approval_year: int
    oil_onstream_date: date | None
    gas_onstream_date: date | None
    is_pod_1: bool | None
    is_strict: bool | None
    lifting: tuple[Lifting, ...] | None
    capital_cost: tuple[CapitalCost, ...] | None
    intangible_cost: tuple[Intangible, ...] | None
    opex: tuple[OPEX, ...] | None
    asr_cost: tuple[ASR, ...] | None
    lbt_cost: tuple[LBT, ...] | None
    cost_of_sales: tuple[CostOfSales, ...] | None


class _BaseRunArguments(TypedDict, total=False):
    sulfur_revenue: OtherRevenue
    electricity_revenue: OtherRevenue
    co2_revenue: OtherRevenue
    vat_rate: _FloatArray | float
    inflation_rate: _FloatArray | float
    inflation_rate_applied_to: InflationAppliedTo | None


class _SummaryArguments(TypedDict):
    reference_year: int
    discount_rate_start_year: int | None
    inflation_rate: float | int | list[object]
    discount_rate: float | int
    npv_mode: NPVSelection
    discounting_mode: DiscountingMode
    profitability_discounted: bool


class _CostRecoveryKwargs(_BaseProjectKwargs):
    oil_ftp_is_available: bool
    oil_ftp_is_shared: bool
    oil_ftp_portion: _FloatArray | float
    gas_ftp_is_available: bool
    gas_ftp_is_shared: bool
    gas_ftp_portion: _FloatArray | float
    tax_split_type: TaxSplitTypeCR
    condition_dict: Mapping[str, npt.ArrayLike]
    indicator_rc_icp_sliding: _FloatArray | None
    oil_ctr_pretax_share: _FloatArray | float
    gas_ctr_pretax_share: _FloatArray | float
    oil_ic_rate: _FloatArray | float
    gas_ic_rate: _FloatArray | float
    ic_is_available: bool
    oil_cr_cap_rate: float
    gas_cr_cap_rate: float
    oil_dmo_volume_portion: _FloatArray | float
    oil_dmo_fee_portion: _FloatArray | float
    oil_dmo_holiday_duration: int
    gas_dmo_volume_portion: _FloatArray | float
    gas_dmo_fee_portion: _FloatArray | float
    gas_dmo_holiday_duration: int
    oil_carry_forward_depreciation: _FloatArray | float
    gas_carry_forward_depreciation: _FloatArray | float


class _CostRecoveryRunArguments(_BaseRunArguments, total=False):
    year_inflation: npt.NDArray[np.int64] | None
    ftp_tax_regime: FTPTaxRegime
    is_dmo_end_weighted: bool
    tax_regime: TaxRegime
    effective_tax_rate: _FloatArray | float | None
    post_uu_22_year2001: bool
    depr_method: DeprMethod
    decline_factor: float | int
    sum_undepreciated_cost: bool
    oil_cost_of_sales_applied: bool
    gas_cost_of_sales_applied: bool


class _GrossSplitKwargs(_BaseProjectKwargs):
    field_status: str | None
    field_loc: str
    res_depth: str | None
    infra_avail: str
    res_type: str | None
    api_oil: str | None
    domestic_use: str | None
    prod_stage: str | None
    co2_content: str | None
    h2s_content: str | None
    field_reserves_2024: str | None
    infra_avail_2024: str
    field_loc_2024: str
    split_ministry_disc: float
    oil_dmo_volume_portion: _FloatArray | float
    oil_dmo_fee_portion: _FloatArray | float
    oil_dmo_holiday_duration: int
    gas_dmo_volume_portion: _FloatArray | float
    gas_dmo_fee_portion: _FloatArray | float
    gas_dmo_holiday_duration: int
    oil_carry_forward_depreciation: _FloatArray | float
    gas_carry_forward_depreciation: _FloatArray | float


class _GrossSplitRunArguments(_BaseRunArguments, total=False):
    year_inflation: npt.NDArray[np.int64] | None
    cum_production_split_offset: _FloatArray | float | None
    depr_method: DeprMethod
    decline_factor: float | int
    sum_undepreciated_cost: bool
    is_dmo_end_weighted: bool
    tax_regime: TaxRegime
    effective_tax_rate: _FloatArray | float | int | None
    amortization: bool
    regime: GrossSplitRegime
    reservoir_type_permen_2024: VariableSplit132024.ReservoirType


class _TransitionRunArguments(TypedDict, total=False):
    unrec_portion: float


class _OptimizationConfiguration(TypedDict):
    parameter: list[OptimizationParameter]
    min: _FloatArray
    max: _FloatArray


class _OptimizationSummaryArguments(TypedDict, total=False):
    contract: Transition
    reference_year: int | None
    discount_rate: float
    npv_mode: NPVSelection
    discounting_mode: DiscountingMode
    discount_rate_start_year: int | None
    inflation_rate: float
    profitability_discounted: bool


class _CombineOptions(TypedDict):
    reference_year: int | None
    inflation_rate: float
    discount_rate: float
    npv_mode: str
    discounting_mode: str
    profitability_discounted: bool


class _CombineCase(TypedDict):
    cId: object
    ctr: int
    data: dict[str, object]


class ContractException(Exception):
    """Exception to be raised if contract type is misused"""

    pass


class LTPModelException(Exception):
    """Exception to be raised for an incorrect LTP configurations"""

    pass


class RDPModelException(Exception):
    """Exception to be raised for an incorrect RDP configurations"""

    pass


def _extract_from_dict(
    target_key: str,
    source: Mapping[str, object],
    default: object = None,
) -> object:
    """
    Safely extract a value from a dictionary.

    Returns `default` if the key is missing or its value is None;
    otherwise returns the stored value.

    Parameters
    ----------
    target_key : str
        Key to extract.
    source : dict
        Source dictionary.
    default : any, optional
        Fallback value if key is absent or None.

    Returns
    -------
    any
        Extracted value or `default`.
    """
    return (
        default
        if (target_key not in source) or (source[target_key] is None)
        else source[target_key]
    )


def _is_string_object_mapping(value: object) -> TypeGuard[dict[str, object]]:
    """Return whether a request mapping has string keys without copying it."""
    if not isinstance(value, dict):
        return False
    try:
        _ = _STRING_OBJECT_MAPPING.validate_python(value, strict=True)
    except ValidationError:
        return False
    return True


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _is_numeric_array(value: object) -> TypeGuard[npt.NDArray[np.generic]]:
    return isinstance(value, np.ndarray)


def _require_mapping(value: object, *, field: str) -> dict[str, object]:
    """Validate a nested request mapping while retaining its identity."""
    if not _is_string_object_mapping(value):
        raise TypeError(f"{field} must be a dictionary with string keys")
    return value


def _as_object_mapping(value: Mapping[str, object]) -> dict[object, object]:
    """Copy a string-keyed payload into a legacy object-keyed mapping."""
    result: dict[object, object] = {}
    for key, item in value.items():
        result[key] = item
    return result


def _require_array_mapping(value: object, *, field: str) -> dict[str, npt.ArrayLike]:
    """Validate a string-keyed mapping whose values are array-like."""
    mapping = _require_mapping(value, field=field)
    result: dict[str, npt.ArrayLike] = {}
    for key, item in mapping.items():
        if isinstance(item, (str, bytes)):
            raise TypeError(f"{field}.{key} must be array-like")
        result[key] = np.asarray(item)
    return result


def _optional_bool(value: object, *, field: str) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    raise TypeError(f"{field} must be a boolean or null")


def _optional_str(value: object, *, field: str) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"{field} must be a string or null")


def _require_value(value: _T | None, *, field: str) -> _T:
    if value is None:
        raise TypeError(f"{field} is required")
    return value


def _require_int(value: object, *, field: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise TypeError(f"{field} must be an integer")


def _require_float(value: object, *, field: str) -> _Number:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    raise TypeError(f"{field} must be a number")


def _optional_float(value: object, *, field: str) -> _Number | None:
    if value is None:
        return None
    return _require_float(value, field=field)


def _require_bool(value: object, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError(f"{field} must be a boolean")


def _require_str(value: object, *, field: str) -> str:
    if isinstance(value, str):
        return value
    raise TypeError(f"{field} must be a string")


def _require_list(value: object, *, field: str) -> list[object]:
    if _is_object_list(value):
        return value
    raise TypeError(f"{field} must be a list")


def _require_summary_inflation(
    value: object, *, field: str
) -> float | int | list[object]:
    """Narrow the historical scalar-or-list summary model without coercion."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if _is_object_list(value):
        return value
    raise TypeError(f"{field} must be a number or list")


def _to_list(value: object, *, field: str) -> list[object]:
    if not _is_numeric_array(value):
        raise TypeError(f"{field} must be a NumPy array")
    if value.ndim != 1:
        raise TypeError(f"{field} must be one-dimensional")
    return [value[index] for index in range(value.size)]


def _frame_to_payload(frame: pd.DataFrame) -> dict[str, object]:
    """Return a JSON-ready, column-keyed representation of a result frame."""
    return {str(column): frame[column].to_dict() for column in frame.columns}


def _require_combine_cases(value: object) -> list[_CombineCase]:
    cases = _require_list(value, field="cases")
    result: list[_CombineCase] = []
    for index, item in enumerate(cases):
        case = _require_mapping(item, field=f"cases[{index}]")
        result.append(
            {
                "cId": case["cId"],
                "ctr": _require_int(case["ctr"], field=f"cases[{index}].ctr"),
                "data": _require_mapping(case["data"], field=f"cases[{index}].data"),
            }
        )
    return result


def _require_combine_options(value: object) -> _CombineOptions:
    options = _require_mapping(value, field="options")
    reference_year_value = options["reference_year"]
    reference_year = (
        None
        if reference_year_value is None
        else _require_int(reference_year_value, field="options.reference_year")
    )
    return {
        "reference_year": reference_year,
        "inflation_rate": _require_float(
            options["inflation_rate"], field="options.inflation_rate"
        ),
        "discount_rate": _require_float(
            options["discount_rate"], field="options.discount_rate"
        ),
        "npv_mode": str(options["npv_mode"]),
        "discounting_mode": str(options["discounting_mode"]),
        "profitability_discounted": _require_bool(
            options["profitability_discounted"],
            field="options.profitability_discounted",
        ),
    }


def _transition_run_arguments(value: object) -> _TransitionRunArguments:
    """Parse the explicit public arguments accepted by ``Transition.run``."""
    arguments = _require_mapping(value, field="contract_arguments")
    if "unrec_portion" not in arguments:
        return {}
    return {
        "unrec_portion": _require_float(
            arguments["unrec_portion"], field="contract_arguments.unrec_portion"
        )
    }


def get_setup_dict(
    data: dict[str, object],
) -> tuple[
    date,
    date,
    int,
    date | None,
    date | None,
    bool | None,
    bool | None,
    tuple[Lifting, ...] | None,
    tuple[CapitalCost, ...] | None,
    tuple[Intangible, ...] | None,
    tuple[OPEX, ...] | None,
    tuple[ASR, ...] | None,
    tuple[LBT, ...] | None,
    tuple[CostOfSales, ...] | None,
]:
    """
    Extract core project setup and cost objects.
    - Parses dates, approval year, POD flag, strict mode
    - Converts cost-related sections into engine-ready objects
    - Missing cost sections → None

    Parameters
    ----------
    data : dict
        Project input dictionary (must contain ``setup``).

    Returns
    -------
    tuple
        (
            start_date, end_date, approval_year,
            oil_onstream_date, gas_onstream_date,
            is_pod_1, is_strict,
            lifting, capital, intangible,
            opex, asr, lbt, cost_of_sales
        )
    """

    # Check whether "setup" exist in "data" and prepare it accordingly
    se = _require_mapping(
        _extract_from_dict(target_key="setup", source=data), field="setup"
    )

    # Parsing the contract setup into each corresponding variables
    start_date = convert_str_to_date(str_object=se["start_date"])
    assert start_date is not None
    end_date = convert_str_to_date(str_object=se["end_date"])
    assert end_date is not None
    approval_year = convert_str_to_int(
        str_object=se.get("approval_year", start_date.year)
    )
    if approval_year is None:
        approval_year = start_date.year
    oil_onstream_date = convert_str_to_date(
        str_object=se.get("oil_onstream_date", None)
    )
    gas_onstream_date = convert_str_to_date(
        str_object=se.get("gas_onstream_date", None)
    )
    is_pod_1 = _optional_bool(
        _extract_from_dict(target_key="is_pod_1", source=se), field="is_pod_1"
    )
    is_strict = _optional_bool(
        _extract_from_dict(target_key="is_strict", source=se), field="is_strict"
    )
    lifting = convert_dict_to_lifting(data_raw=data) if "lifting" in data else None
    capital = (
        convert_dict_to_capital(data_raw=data["capital"]) if "capital" in data else None
    )
    intangible = (
        convert_dict_to_intangible(data_raw=data["intangible"])
        if "intangible" in data
        else None
    )
    opex = convert_dict_to_opex(data_raw=data["opex"]) if "opex" in data else None
    asr = convert_dict_to_asr(data_raw=data["asr"]) if "asr" in data else None
    lbt = convert_dict_to_lbt(data_raw=data["lbt"]) if "lbt" in data else None
    cost_of_sales = (
        convert_dict_to_cost_of_sales(data_raw=data["cost_of_sales"])
        if "cost_of_sales" in data
        else None
    )

    return (
        start_date,
        end_date,
        approval_year,
        oil_onstream_date,
        gas_onstream_date,
        is_pod_1,
        is_strict,
        lifting,
        capital,
        intangible,
        opex,
        asr,
        lbt,
        cost_of_sales,
    )


def get_summary_dict(data: dict[str, object]) -> _SummaryArguments:
    """
    Extract and normalize summary-related arguments.

    Parameters
    ----------
    data : dict
        Input configuration containing a required ``summary_arguments`` section.

    Returns
    -------
    dict
        Summary arguments with defaults applied and enums converted:
        - discount_rate_start_year
        - inflation_rate
        - discount_rate
        - npv_mode
        - discounting_mode
        - profitability_discounted
    """

    # Check whether "summary_arguments" exists in "data" and prepare it accordingly
    sa = _require_mapping(
        _extract_from_dict(target_key="summary_arguments", source=data),
        field="summary_arguments",
    )

    # Fill get_summary() argument with input data
    reference_year = _require_int(
        sa.get("reference_year", 2022), field="summary_arguments.reference_year"
    )
    discount_rate_start_year = sa.get("discount_rate_start_year")
    if discount_rate_start_year is not None:
        discount_rate_start_year = _require_int(
            discount_rate_start_year,
            field="summary_arguments.discount_rate_start_year",
        )
    inflation_rate = _require_summary_inflation(
        sa.get("inflation_rate", 0.1), field="summary_arguments.inflation_rate"
    )
    discount_rate = _require_float(
        sa.get("discount_rate", 0.1), field="summary_arguments.discount_rate"
    )
    npv_mode = convert_str_to_npvmode(
        str_object=sa.get("npv_mode", "Full Cycle Nominal Terms")
    )
    if npv_mode is None:
        raise TypeError("summary_arguments.npv_mode is not recognized")
    discounting_mode = convert_str_to_discountingmode(
        str_object=sa.get("discounting_mode", "End Year Discounting")
    )
    if discounting_mode is None:
        raise TypeError("summary_arguments.discounting_mode is not recognized")
    profitability_discounted = _require_bool(
        sa.get("profitability_discounted", False),
        field="summary_arguments.profitability_discounted",
    )

    return {
        "reference_year": reference_year,
        "discount_rate_start_year": discount_rate_start_year,
        "inflation_rate": inflation_rate,
        "discount_rate": discount_rate,
        "npv_mode": npv_mode,
        "discounting_mode": discounting_mode,
        "profitability_discounted": profitability_discounted,
    }


def _summary_reference_year(summary_arguments: _SummaryArguments) -> int | None:
    """Select the external helper's reference year without losing either input."""
    return (
        summary_arguments["discount_rate_start_year"]
        if summary_arguments["discount_rate_start_year"] is not None
        else summary_arguments["reference_year"]
    )


def _contract_summary(
    contract: BaseProject | CostRecovery | GrossSplit,
    summary_arguments: _SummaryArguments,
) -> Mapping[str, object]:
    """Call the summary method implemented by non-transition contracts."""
    return contract.get_summary(
        discount_rate_start_year=summary_arguments["discount_rate_start_year"],
        inflation_rate=_require_float(
            summary_arguments["inflation_rate"],
            field="summary_arguments.inflation_rate",
        ),
        discount_rate=summary_arguments["discount_rate"],
        npv_mode=summary_arguments["npv_mode"],
        discounting_mode=summary_arguments["discounting_mode"],
        profitability_discounted=summary_arguments["profitability_discounted"],
    )


def _contract_summary_payload(
    summary_arguments: _SummaryArguments,
) -> dict[str, object]:
    """Remove external-only reference_year for instance-summary consumers."""
    return {
        "discount_rate_start_year": summary_arguments["discount_rate_start_year"],
        "inflation_rate": summary_arguments["inflation_rate"],
        "discount_rate": summary_arguments["discount_rate"],
        "npv_mode": summary_arguments["npv_mode"],
        "discounting_mode": summary_arguments["discounting_mode"],
        "profitability_discounted": summary_arguments["profitability_discounted"],
    }


def _optimization_summary_payload(
    summary_arguments: _SummaryArguments,
) -> _OptimizationSummaryArguments:
    """Validate summary values accepted by the optimization engines."""
    return {
        "discount_rate_start_year": summary_arguments["discount_rate_start_year"],
        "inflation_rate": _require_float(
            summary_arguments["inflation_rate"],
            field="summary_arguments.inflation_rate",
        ),
        "discount_rate": _require_float(
            summary_arguments["discount_rate"], field="summary_arguments.discount_rate"
        ),
        "npv_mode": summary_arguments["npv_mode"],
        "discounting_mode": summary_arguments["discounting_mode"],
        "profitability_discounted": summary_arguments["profitability_discounted"],
    }


def _external_summary(
    contract: BaseProject | CostRecovery | GrossSplit | Transition,
    summary_arguments: _SummaryArguments,
) -> Mapping[str, object]:
    """Call the shared helper used for Transition summary calculations."""
    return get_summary(
        contract=contract,
        reference_year=_summary_reference_year(summary_arguments),
        inflation_rate=_require_float(
            summary_arguments["inflation_rate"],
            field="summary_arguments.inflation_rate",
        ),
        discount_rate=summary_arguments["discount_rate"],
        npv_mode=summary_arguments["npv_mode"],
        discounting_mode=summary_arguments["discounting_mode"],
        profitability_discounted=summary_arguments["profitability_discounted"],
    )


def build_baseproject_instance(data: dict[str, object]) -> BaseProject:
    """
    Build and initialize a Base Project contract instance.

    This function extracts fundamental parameters from the input data,
    prepares the required keyword arguments, and returns a fully
    constructed :class:`BaseProject` instance for economic evaluation.

    Parameters
    ----------
    data : dict
        Input dictionary containing all parameters necessary to
        configure a Base Project contract.

    Returns
    -------
    BaseProject
        Initialized :class:`BaseProject` object ready for evaluation.
    """

    # Specify base arguments
    (
        start_date,
        end_date,
        approval_year,
        oil_onstream_date,
        gas_onstream_date,
        is_pod_1,
        is_strict,
        lifting,
        capital,
        intangible,
        opex,
        asr,
        lbt,
        cost_of_sales,
    ) = get_setup_dict(data=data)

    # Prepare contract attributes for BaseProject
    contract_kwargs: _BaseProjectKwargs = {
        # Base parameters
        "start_date": start_date,
        "end_date": end_date,
        "approval_year": approval_year,
        "oil_onstream_date": oil_onstream_date,
        "gas_onstream_date": gas_onstream_date,
        "is_pod_1": is_pod_1,
        "is_strict": is_strict,
        # Lifting and costs
        "lifting": lifting,
        "capital_cost": capital,
        "intangible_cost": intangible,
        "opex": opex,
        "asr_cost": asr,
        "lbt_cost": lbt,
        "cost_of_sales": cost_of_sales,
    }

    return BaseProject(**contract_kwargs)


def build_baseproject_arguments(data: dict[str, object]) -> _BaseRunArguments:
    """
    Build the argument dictionary for a Base Project contract.

    This function extracts and converts key economic parameters such as
    revenues, tax rate, and inflation information from the input data,
    returning a dictionary suitable for initializing or executing a
    :class:`BaseProject` instance.

    Parameters
    ----------
    data : dict
        Input dictionary containing contract arguments and economic
        parameters for the Base Project.

    Returns
    -------
    dict
        Dictionary of processed Base Project arguments, ready for use
        in model evaluation.
    """

    # Check whether "contract_arguments" exist in "data" and prepare it accordingly
    ca = _require_mapping(
        _extract_from_dict(target_key="contract_arguments", source=data),
        field="contract_arguments",
    )

    return {
        # Other revenues
        "sulfur_revenue": _require_value(
            convert_str_to_otherrevenue(
                str_object=ca.get("sulfur_revenue", "Addition to Gas Revenue")
            ),
            field="contract_arguments.sulfur_revenue",
        ),
        "electricity_revenue": _require_value(
            convert_str_to_otherrevenue(
                str_object=ca.get("electricity_revenue", "Addition to Gas Revenue")
            ),
            field="contract_arguments.electricity_revenue",
        ),
        "co2_revenue": _require_value(
            convert_str_to_otherrevenue(
                str_object=ca.get("co2_revenue", "Addition to Gas Revenue")
            ),
            field="contract_arguments.co2_revenue",
        ),
        # VAT and inflation
        "vat_rate": convert_list_to_array_float_or_array(
            data_input=ca.get("vat_rate", 0.0)
        ),
        "inflation_rate": convert_list_to_array_float_or_array(
            data_input=ca.get("inflation_rate", 0.0)
        ),
        "inflation_rate_applied_to": convert_str_to_inflationappliedto(
            str_object=ca.get("inflation_rate_applied_to", None)
        ),
    }


@overload
def get_baseproject(
    data: dict[str, _T], summary_result: Literal[True] = True
) -> tuple[dict[str, object], BaseProject, _BaseRunArguments, _SummaryArguments]: ...


@overload
def get_baseproject(
    data: dict[str, _T], summary_result: Literal[False]
) -> tuple[None, BaseProject, _BaseRunArguments, None]: ...


@overload
def get_baseproject(
    data: dict[str, _T], summary_result: bool
) -> tuple[
    dict[str, object] | None,
    BaseProject,
    _BaseRunArguments,
    _SummaryArguments | None,
]: ...


def get_baseproject(
    data: dict[str, object], summary_result: bool = True
) -> tuple[
    dict[str, object] | None,
    BaseProject,
    _BaseRunArguments,
    _SummaryArguments | None,
]:
    """
    Build and run a BaseProject evaluation.

    - Builds BaseProject instance
    - Prepares and executes contract arguments
    - Optionally returns SKK Migas–formatted summary

    Parameters
    ----------
    data : dict
        Project input dictionary (setup, contract args, summary args).
    summary_result : bool, default=True
        Return SKK Migas summary if True.

    Returns
    -------
    tuple
        (
            summary_skk,              # dict | None
            contract,                 # BaseProject
            contract_arguments_dict,  # dict
            summary_arguments_dict    # dict | None
        )
    """

    # Specify contract and contract arguments
    contract = build_baseproject_instance(data=data)
    contract_arguments_dict = build_baseproject_arguments(data=data)

    # Execute BaseProject instance
    contract.run(**contract_arguments_dict)

    # Display (or undisplay) summary
    if summary_result is True:
        # Fill summary arguments
        summary_arguments_dict = get_summary_dict(data=data)
        summary = _contract_summary(contract, summary_arguments_dict)

        # Display summary using SKK Migas format
        summary_skk = convert_to_skk_summary_baseproject(dict_object=summary)

        # Add execution info
        summary_skk = add_execution_info(data=summary_skk)

    # Since the required object is only the contract, it will return None for the summary
    else:
        summary_skk = None
        summary_arguments_dict = None

    return summary_skk, contract, contract_arguments_dict, summary_arguments_dict


def build_costrecovery_instance(data: dict[str, object]) -> CostRecovery:
    """
    Build a fully configured CostRecovery contract instance.

    Parses setup data via ``get_setup_dict`` and converts cost-recovery
    parameters (FTP, split, IC, DMO, cap rate) into engine-ready inputs,
    then initializes and returns a ``CostRecovery`` object.

    Parameters
    ----------
    data : dict
        Project input dictionary containing ``setup`` and ``costrecovery``
        sections, plus optional lifting and cost components.

    Returns
    -------
    CostRecovery
        Initialized CostRecovery contract, ready to run.
    """

    # Specify base arguments
    (
        start_date,
        end_date,
        approval_year,
        oil_onstream_date,
        gas_onstream_date,
        is_pod_1,
        is_strict,
        lifting,
        capital,
        intangible,
        opex,
        asr,
        lbt,
        cost_of_sales,
    ) = get_setup_dict(data=data)

    # Check whether "costrecovery" exist in "data" and prepare it accordingly
    cr = _require_mapping(
        _extract_from_dict(target_key="costrecovery", source=data),
        field="costrecovery",
    )

    # Prepare contract attributes for CostRecovery
    contract_kwargs: _CostRecoveryKwargs = {
        # Base parameters
        "start_date": start_date,
        "end_date": end_date,
        "approval_year": approval_year,
        "oil_onstream_date": oil_onstream_date,
        "gas_onstream_date": gas_onstream_date,
        "is_pod_1": is_pod_1,
        "is_strict": is_strict,
        # Lifting and costs
        "lifting": lifting,
        "capital_cost": capital,
        "intangible_cost": intangible,
        "opex": opex,
        "asr_cost": asr,
        "lbt_cost": lbt,
        "cost_of_sales": cost_of_sales,
        # FTP
        "oil_ftp_is_available": _require_bool(
            cr["oil_ftp_is_available"], field="costrecovery.oil_ftp_is_available"
        ),
        "oil_ftp_is_shared": _require_bool(
            cr["oil_ftp_is_shared"], field="costrecovery.oil_ftp_is_shared"
        ),
        "oil_ftp_portion": convert_list_to_array_float_or_array(
            data_input=cr["oil_ftp_portion"]
        ),
        "gas_ftp_is_available": _require_bool(
            cr["gas_ftp_is_available"], field="costrecovery.gas_ftp_is_available"
        ),
        "gas_ftp_is_shared": _require_bool(
            cr["gas_ftp_is_shared"], field="costrecovery.gas_ftp_is_shared"
        ),
        "gas_ftp_portion": convert_list_to_array_float_or_array(
            data_input=cr["gas_ftp_portion"]
        ),
        # Split
        "tax_split_type": _require_value(
            convert_str_to_taxsplit(str_object=cr["tax_split_type"]),
            field="costrecovery.tax_split_type",
        ),
        "condition_dict": _require_array_mapping(
            cr["condition_dict"], field="costrecovery.condition_dict"
        ),
        "indicator_rc_icp_sliding": convert_list_to_array_float(
            data_list=cr["indicator_rc_icp_sliding"]
        ),
        "oil_ctr_pretax_share": convert_list_to_array_float_or_array(
            data_input=cr["oil_ctr_pretax_share"]
        ),
        "gas_ctr_pretax_share": convert_list_to_array_float_or_array(
            data_input=cr["gas_ctr_pretax_share"]
        ),
        # Investment credit and cap rate
        "oil_ic_rate": convert_to_float(target=cr["oil_ic_rate"]),
        "gas_ic_rate": convert_to_float(target=cr["gas_ic_rate"]),
        "ic_is_available": _require_bool(
            cr["ic_is_available"], field="costrecovery.ic_is_available"
        ),
        "oil_cr_cap_rate": _require_float(
            cr["oil_cr_cap_rate"], field="costrecovery.oil_cr_cap_rate"
        ),
        "gas_cr_cap_rate": _require_float(
            cr["gas_cr_cap_rate"], field="costrecovery.gas_cr_cap_rate"
        ),
        # DMO
        "oil_dmo_volume_portion": convert_list_to_array_float_or_array(
            data_input=cr["oil_dmo_volume_portion"]
        ),
        "oil_dmo_fee_portion": convert_list_to_array_float_or_array(
            data_input=cr["oil_dmo_fee_portion"]
        ),
        "oil_dmo_holiday_duration": _require_int(
            cr["oil_dmo_holiday_duration"],
            field="costrecovery.oil_dmo_holiday_duration",
        ),
        "gas_dmo_volume_portion": convert_list_to_array_float_or_array(
            data_input=cr["gas_dmo_volume_portion"]
        ),
        "gas_dmo_fee_portion": convert_list_to_array_float_or_array(
            data_input=cr["gas_dmo_fee_portion"]
        ),
        "gas_dmo_holiday_duration": _require_int(
            cr["gas_dmo_holiday_duration"],
            field="costrecovery.gas_dmo_holiday_duration",
        ),
        # Carry forward depreciation
        "oil_carry_forward_depreciation": 0.0,
        "gas_carry_forward_depreciation": 0.0,
    }

    return CostRecovery(**contract_kwargs)


def build_costrecovery_arguments(
    data: dict[str, object],
) -> _CostRecoveryRunArguments:
    """
    Build normalized cost-recovery arguments from raw input data.

    Extracts `contract_arguments` from `data`, applies defaults where needed,
    and converts string/list inputs into internal enum/array representations
    required by the cost-recovery model.

    Parameters
    ----------
    data : dict
        Raw input dictionary containing `contract_arguments`.

    Returns
    -------
    dict
        Sanitized and converted cost-recovery arguments ready for model use.
    """

    # Check whether "contract_arguments" exist in "data" and prepare it accordingly
    ca = _require_mapping(
        _extract_from_dict(target_key="contract_arguments", source=data),
        field="contract_arguments",
    )

    return {
        # Other revenues
        "sulfur_revenue": _require_value(
            convert_str_to_otherrevenue(str_object=ca["sulfur_revenue"]),
            field="contract_arguments.sulfur_revenue",
        ),
        "electricity_revenue": _require_value(
            convert_str_to_otherrevenue(str_object=ca["electricity_revenue"]),
            field="contract_arguments.electricity_revenue",
        ),
        "co2_revenue": _require_value(
            convert_str_to_otherrevenue(str_object=ca["co2_revenue"]),
            field="contract_arguments.co2_revenue",
        ),
        # FTP
        "ftp_tax_regime": _require_value(
            convert_str_to_ftptaxregime(str_object=ca["ftp_tax_regime"]),
            field="contract_arguments.ftp_tax_regime",
        ),
        # VAT and inflation
        "vat_rate": convert_list_to_array_float_or_array(data_input=ca["vat_rate"]),
        "inflation_rate": convert_list_to_array_float_or_array(
            data_input=ca["inflation_rate"]
        ),
        "inflation_rate_applied_to": convert_str_to_inflationappliedto(
            str_object=ca["inflation_rate_applied_to"]
        ),
        # DMO and tax
        "is_dmo_end_weighted": _require_bool(
            ca["is_dmo_end_weighted"], field="contract_arguments.is_dmo_end_weighted"
        ),
        "tax_regime": _require_value(
            convert_str_to_taxregime(str_object=ca["tax_regime"]),
            field="contract_arguments.tax_regime",
        ),
        "effective_tax_rate": convert_list_to_array_float_or_array_or_none(
            data_list=ca["effective_tax_rate"]
        ),
        "post_uu_22_year2001": _require_bool(
            _extract_from_dict(
                target_key="post_uu_22_year2001", source=ca, default=True
            ),
            field="contract_arguments.post_uu_22_year2001",
        ),
        # Depreciation
        "depr_method": _require_value(
            convert_str_to_depremethod(str_object=ca["depr_method"]),
            field="contract_arguments.depr_method",
        ),
        "decline_factor": _require_float(
            ca["decline_factor"], field="contract_arguments.decline_factor"
        ),
        "sum_undepreciated_cost": _require_bool(
            _extract_from_dict(
                target_key="sum_undepreciated_cost", source=ca, default=False
            ),
            field="contract_arguments.sum_undepreciated_cost",
        ),
        # Cost of sales
        "oil_cost_of_sales_applied": _require_bool(
            _extract_from_dict(
                target_key="oil_cost_of_sales_applied", source=ca, default=False
            ),
            field="contract_arguments.oil_cost_of_sales_applied",
        ),
        "gas_cost_of_sales_applied": _require_bool(
            _extract_from_dict(
                target_key="gas_cost_of_sales_applied", source=ca, default=False
            ),
            field="contract_arguments.gas_cost_of_sales_applied",
        ),
    }


@overload
def get_costrecovery(
    data: dict[str, _T], summary_result: Literal[True] = True
) -> tuple[
    dict[str, object],
    CostRecovery,
    _CostRecoveryRunArguments,
    _SummaryArguments,
]: ...


@overload
def get_costrecovery(
    data: dict[str, _T], summary_result: Literal[False]
) -> tuple[None, CostRecovery, _CostRecoveryRunArguments, None]: ...


@overload
def get_costrecovery(
    data: dict[str, _T], summary_result: bool
) -> tuple[
    dict[str, object] | None,
    CostRecovery,
    _CostRecoveryRunArguments,
    _SummaryArguments | None,
]: ...


def get_costrecovery(
    data: dict[str, object], summary_result: bool = True
) -> tuple[
    dict[str, object] | None,
    CostRecovery,
    _CostRecoveryRunArguments,
    _SummaryArguments | None,
]:
    """
    Run Cost Recovery PSC evaluation (+ optional summary).

    - Build CostRecovery contract
    - Prepare & execute contract arguments
    - Optionally return SKK Migas–formatted summary

    Parameters
    ----------
    data : dict
        Full input dictionary for Cost Recovery evaluation.
    summary_result : bool, default=True
        Return summary if True; skip if False.

    Returns
    -------
    tuple
        (
            summary_skk,              # dict | None
            contract,                 # CostRecovery
            contract_arguments_dict,  # dict
            summary_arguments_dict    # dict | None
        )

    TL;DR: Build → run Cost Recovery contract → optionally return SKK-style summary.
    """

    # Specify contract and contract arguments
    contract = build_costrecovery_instance(data=data)
    contract_arguments_dict = build_costrecovery_arguments(data=data)

    # Execute CostRecovery instance
    contract.run(**contract_arguments_dict)

    # Display (or undisplay) summary
    if summary_result is True:
        # Fill summary arguments
        summary_arguments_dict = get_summary_dict(data=data)
        # ==== IGNORED IN 1.4.0 =====
        # summary_arguments_dict["contract"] = contract
        # summary = get_summary(**summary_arguments_dict)
        summary = _contract_summary(contract, summary_arguments_dict)

        # Display summary using SKK Migas format
        summary_skk = convert_summary_to_dict(dict_object=summary)

        # Add execution info
        summary_skk = add_execution_info(data=summary_skk)

    # Since the required object is only the contract, it will return None for the summary
    else:
        summary_skk = None
        summary_arguments_dict = None

    return summary_skk, contract, contract_arguments_dict, summary_arguments_dict


def build_grosssplit_instance(data: dict[str, object]) -> GrossSplit:
    """
    Build and initialize a GrossSplit contract instance from input data.

    Parses common project setup (dates, POD/strict flags, lifting, costs) and
    Gross Split–specific parameters (field/reservoir attributes, ministry
    discretion, DMO settings), then instantiates a :class:`GrossSplit` object.

    Parameters
    ----------
    data : dict
        Input dictionary containing:
        - ``setup`` section (project dates, flags)
        - cost sections (lifting, capital, opex, etc.)
        - ``grosssplit`` section with field, reservoir, split, and DMO parameters.

    Returns
    -------
    GrossSplit
        Initialized GrossSplit contract instance (not yet executed).

    Notes
    -----
    - Missing or optional Gross Split attributes are handled via
      ``_extract_from_dict`` (→ default ``None``).
    - Carry-forward depreciation is initialized to zero for both oil and gas.
    - No economic calculation is performed here; call ``GrossSplit.run(...)``
      separately.

    TL;DR: Parse setup + Gross Split inputs and return a ready-to-run GrossSplit instance.
    """

    # Specify base arguments
    (
        start_date,
        end_date,
        approval_year,
        oil_onstream_date,
        gas_onstream_date,
        is_pod_1,
        is_strict,
        lifting,
        capital,
        intangible,
        opex,
        asr,
        lbt,
        cost_of_sales,
    ) = get_setup_dict(data=data)

    # Check whether "grosssplit" exist in "data" and prepare it accordingly
    gs = _require_mapping(
        _extract_from_dict(target_key="grosssplit", source=data), field="grosssplit"
    )

    # Prepare contract attributes for GrossSplit
    contract_kwargs: _GrossSplitKwargs = {
        # Base parameters
        "start_date": start_date,
        "end_date": end_date,
        "approval_year": approval_year,
        "oil_onstream_date": oil_onstream_date,
        "gas_onstream_date": gas_onstream_date,
        "is_pod_1": is_pod_1,
        "is_strict": is_strict,
        # Lifting and costs
        "lifting": lifting,
        "capital_cost": capital,
        "intangible_cost": intangible,
        "opex": opex,
        "asr_cost": asr,
        "lbt_cost": lbt,
        "cost_of_sales": cost_of_sales,
        # Field and reservoir properties
        "field_status": _optional_str(
            _extract_from_dict(target_key="field_status", source=gs),
            field="grosssplit.field_status",
        ),
        "field_loc": _require_str(
            _extract_from_dict(target_key="field_loc", source=gs, default="Onshore"),
            field="grosssplit.field_loc",
        ),
        "res_depth": _optional_str(
            _extract_from_dict(target_key="res_depth", source=gs),
            field="grosssplit.res_depth",
        ),
        "infra_avail": _require_str(
            _extract_from_dict(
                target_key="infra_avail", source=gs, default="Well Developed"
            ),
            field="grosssplit.infra_avail",
        ),
        "res_type": _optional_str(
            _extract_from_dict(target_key="res_type", source=gs),
            field="grosssplit.res_type",
        ),
        "api_oil": _optional_str(
            _extract_from_dict(target_key="api_oil", source=gs),
            field="grosssplit.api_oil",
        ),
        "domestic_use": _optional_str(
            _extract_from_dict(target_key="domestic_use", source=gs),
            field="grosssplit.domestic_use",
        ),
        "prod_stage": _optional_str(
            _extract_from_dict(target_key="prod_stage", source=gs),
            field="grosssplit.prod_stage",
        ),
        "co2_content": _optional_str(
            _extract_from_dict(target_key="co2_content", source=gs),
            field="grosssplit.co2_content",
        ),
        "h2s_content": _optional_str(
            _extract_from_dict(target_key="h2s_content", source=gs),
            field="grosssplit.h2s_content",
        ),
        "field_reserves_2024": _optional_str(
            _extract_from_dict(target_key="field_reserves_2024", source=gs),
            field="grosssplit.field_reserves_2024",
        ),
        "infra_avail_2024": _require_str(
            _extract_from_dict(
                target_key="infra_avail_2024", source=gs, default="available"
            ),
            field="grosssplit.infra_avail_2024",
        ),
        "field_loc_2024": _require_str(
            _extract_from_dict(
                target_key="field_loc_2024", source=gs, default="Onshore"
            ),
            field="grosssplit.field_loc_2024",
        ),
        "split_ministry_disc": convert_to_float(target=gs["split_ministry_disc"]),
        # DMO parameters
        "oil_dmo_volume_portion": convert_list_to_array_float_or_array(
            data_input=gs["oil_dmo_volume_portion"]
        ),
        "oil_dmo_fee_portion": convert_list_to_array_float_or_array(
            data_input=gs["oil_dmo_fee_portion"]
        ),
        "oil_dmo_holiday_duration": _require_int(
            _extract_from_dict(target_key="oil_dmo_holiday_duration", source=gs),
            field="grosssplit.oil_dmo_holiday_duration",
        ),
        "gas_dmo_volume_portion": convert_list_to_array_float_or_array(
            data_input=gs["gas_dmo_volume_portion"]
        ),
        "gas_dmo_fee_portion": convert_list_to_array_float_or_array(
            data_input=gs["gas_dmo_fee_portion"]
        ),
        "gas_dmo_holiday_duration": _require_int(
            _extract_from_dict(target_key="gas_dmo_holiday_duration", source=gs),
            field="grosssplit.gas_dmo_holiday_duration",
        ),
        # Carry forward depreciation
        "oil_carry_forward_depreciation": 0.0,
        "gas_carry_forward_depreciation": 0.0,
    }

    return GrossSplit(**contract_kwargs)


def build_grosssplit_arguments(
    data: dict[str, object],
) -> _GrossSplitRunArguments:
    """
    Build runtime arguments for executing a GrossSplit contract.

    Extracts the ``contract_arguments`` section and converts user-facing inputs
    (revenues, VAT/inflation, production offsets, depreciation, DMO/tax rules,
    sunk cost treatment, and fiscal regime flags) into engine-ready formats.

    Parameters
    ----------
    data : dict
        Input dictionary containing a ``contract_arguments`` section with
        Gross Split–specific runtime parameters.

    Returns
    -------
    dict
        Dictionary of keyword arguments to be passed to
        :meth:`GrossSplit.run`.

    TL;DR:  Convert ``contract_arguments`` into a ready-to-use kwargs dict for
            `GrossSplit.run()`.
    """

    # Check whether "contract_arguments" exist in "data" and prepare it accordingly
    ca = _require_mapping(
        _extract_from_dict(target_key="contract_arguments", source=data),
        field="contract_arguments",
    )

    return {
        # Other revenues
        "sulfur_revenue": _require_value(
            convert_str_to_otherrevenue(str_object=ca["sulfur_revenue"]),
            field="contract_arguments.sulfur_revenue",
        ),
        "electricity_revenue": _require_value(
            convert_str_to_otherrevenue(str_object=ca["electricity_revenue"]),
            field="contract_arguments.electricity_revenue",
        ),
        "co2_revenue": _require_value(
            convert_str_to_otherrevenue(str_object=ca["co2_revenue"]),
            field="contract_arguments.co2_revenue",
        ),
        # # VAT and inflation
        "vat_rate": convert_list_to_array_float_or_array(data_input=ca["vat_rate"]),
        "inflation_rate": convert_list_to_array_float_or_array(
            data_input=ca["inflation_rate"]
        ),
        "inflation_rate_applied_to": convert_str_to_inflationappliedto(
            str_object=ca["inflation_rate_applied_to"]
        ),
        # Production offset
        "cum_production_split_offset": convert_list_to_array_float_or_array(
            data_input=ca["cum_production_split_offset"]
        ),
        # Depreciation
        "depr_method": _require_value(
            convert_str_to_depremethod(str_object=ca["depr_method"]),
            field="contract_arguments.depr_method",
        ),
        "decline_factor": _require_float(
            ca["decline_factor"], field="contract_arguments.decline_factor"
        ),
        "sum_undepreciated_cost": _require_bool(
            _extract_from_dict(
                target_key="sum_undepreciated_cost", source=ca, default=False
            ),
            field="contract_arguments.sum_undepreciated_cost",
        ),
        # DMO and tax
        "is_dmo_end_weighted": _require_bool(
            _extract_from_dict(
                target_key="is_dmo_end_weighted", source=ca, default=False
            ),
            field="contract_arguments.is_dmo_end_weighted",
        ),
        "tax_regime": _require_value(
            convert_str_to_taxregime(str_object=ca["tax_regime"]),
            field="contract_arguments.tax_regime",
        ),
        "effective_tax_rate": convert_list_to_array_float_or_array_or_none(
            data_list=ca["effective_tax_rate"]
        ),
        # Sunk cost
        "amortization": _require_bool(
            _extract_from_dict(target_key="amortization", source=ca, default=False),
            field="contract_arguments.amortization",
        ),
        # Fiscal regime
        "regime": _require_value(
            convert_grosssplitregime_to_enum(target=ca["regime"]),
            field="contract_arguments.regime",
        ),
        "reservoir_type_permen_2024": converter_reservoir_type_permen_2024(
            target_str=ca["reservoir_type_permen_2024"]
        ),
    }


@overload
def get_grosssplit(
    data: dict[str, _T], summary_result: Literal[True] = True
) -> tuple[
    dict[str, object], GrossSplit, _GrossSplitRunArguments, _SummaryArguments
]: ...


@overload
def get_grosssplit(
    data: dict[str, _T], summary_result: Literal[False]
) -> tuple[None, GrossSplit, _GrossSplitRunArguments, None]: ...


@overload
def get_grosssplit(
    data: dict[str, _T], summary_result: bool
) -> tuple[
    dict[str, object] | None,
    GrossSplit,
    _GrossSplitRunArguments,
    _SummaryArguments | None,
]: ...


def get_grosssplit(
    data: dict[str, object], summary_result: bool = True
) -> tuple[
    dict[str, object] | None,
    GrossSplit,
    _GrossSplitRunArguments,
    _SummaryArguments | None,
]:
    """
    Execute a Gross Split PSC evaluation and optionally return the summary results.

    This function builds the contract instance, prepares input arguments, runs
    the Gross Split model, and optionally generates the SKK Migas–formatted
    summary.

    Parameters
    ----------
    data : dict
        Input data containing all parameters required for the Gross Split evaluation.
    summary_result : bool, default=True
        If True, return the SKK Migas–formatted summary; otherwise, omit it.

    Returns
    -------
    tuple
        (summary_skk, contract, contract_arguments_dict, summary_arguments_dict)

        - **summary_skk** : dict or None
          Summary of results in SKK Migas format, or None if not requested.
        - **contract** : GrossSplit
          Executed Gross Split contract instance.
        - **contract_arguments_dict** : dict
          Arguments used in :meth:`GrossSplit.run`.
        - **summary_arguments_dict** : dict or None
          Arguments used for summary generation, or None if skipped.
    """

    # Specify contract and contract arguments
    contract: GrossSplit = build_grosssplit_instance(data=data)
    contract_arguments_dict = build_grosssplit_arguments(data=data)

    # Execute GrossSplit instance
    contract.run(**contract_arguments_dict)

    # Display (or undisplay) summary
    if summary_result is True:
        # Fill summary arguments
        summary_arguments_dict = get_summary_dict(data=data)
        summary = _contract_summary(contract, summary_arguments_dict)

        # Display summary using SKK Migas format
        summary_skk = convert_summary_to_dict(dict_object=summary)

        # Add execution info
        summary_skk = add_execution_info(data=summary_skk)

    else:
        summary_skk = None
        summary_arguments_dict = None

    return summary_skk, contract, contract_arguments_dict, summary_arguments_dict


def _gross_split_reporting_payload(
    contract: GrossSplit, *, include_components: bool
) -> dict[str, object]:
    """Render the contract's public split profile in the established API shape."""
    profile = contract.get_reporting_split_profile()
    oil_progressive_key = (
        "oil_prog_total_split" if include_components else "oil_prog_split"
    )
    gas_progressive_key = (
        "gas_prog_total_split" if include_components else "gas_prog_split"
    )
    columns: dict[str, list[object]] = {
        "project_years": _to_list(contract.project_years, field="project_years"),
        "oil_base_split": _to_list(profile.oil_base_split, field="oil_base_split"),
        "gas_base_split": _to_list(profile.gas_base_split, field="gas_base_split"),
        "var_split_array": _to_list(profile.variable_split, field="variable_split"),
        oil_progressive_key: _to_list(
            profile.oil_progressive_split, field="oil_progressive_split"
        ),
        gas_progressive_key: _to_list(
            profile.gas_progressive_split, field="gas_progressive_split"
        ),
        "oil_ctr_split": _to_list(
            profile.oil_contractor_split_before_bracket,
            field="oil_contractor_split_before_bracket",
        ),
        "gas_ctr_split": _to_list(
            profile.gas_contractor_split_before_bracket,
            field="gas_contractor_split_before_bracket",
        ),
    }
    if include_components:
        columns.update(
            {
                "oil_prog_price_split": _to_list(
                    profile.oil_progressive_price_split,
                    field="oil_progressive_price_split",
                ),
                "gas_prog_price_split": _to_list(
                    profile.gas_progressive_price_split,
                    field="gas_progressive_price_split",
                ),
                "oil_prog_cumulative_production_split": _to_list(
                    profile.oil_progressive_cumulative_production_split,
                    field="oil_progressive_cumulative_production_split",
                ),
                "gas_prog_cumulative_production_split": _to_list(
                    profile.gas_progressive_cumulative_production_split,
                    field="gas_progressive_cumulative_production_split",
                ),
            }
        )
    contractor_split = _frame_to_payload(
        pd.DataFrame(columns).set_index("project_years")
    )
    return {
        "contractor_split": contractor_split,
        "years_of_maximum_split": {
            "oil": _to_list(
                profile.oil_years_of_maximum_contractor_split,
                field="oil_years_of_maximum_contractor_split",
            ),
            "gas": _to_list(
                profile.gas_years_of_maximum_contractor_split,
                field="gas_years_of_maximum_contractor_split",
            ),
        },
    }


def get_grosssplit_split(data: dict[str, object]) -> dict[str, object]:
    """
    Retrieve contractor split information from the Gross Split PSC scheme.

    This function builds and executes a :class:`GrossSplit` contract instance
    based on the provided input data, then extracts yearly oil and gas
    split components, including base, variable, progressive, and total splits,
    as well as the years with maximum contractor split.

    Parameters
    ----------
    data : dict
        Input data containing parameters required for the Gross Split
        contract evaluation.

    Returns
    -------
    dict
        Dictionary with two main elements:

        - **contractor_split** : dict
          Yearly contractor split information for oil and gas, including base,
          variable, and progressive components, indexed by project years.

        - **years_of_maximum_split** : dict
          Years when the contractor achieved the maximum split for oil and gas.
    """

    # Specify contract and contract arguments
    contract = build_grosssplit_instance(data=data)
    contract_arguments_dict = build_grosssplit_arguments(data=data)

    # Execute GrossSplit intance
    contract.run(**contract_arguments_dict)

    return _gross_split_reporting_payload(contract, include_components=True)


@overload
def get_transition(
    data: dict[str, _T], summary_result: Literal[True] = True
) -> tuple[
    dict[str, object],
    Transition,
    _TransitionRunArguments,
    _SummaryArguments,
]: ...


@overload
def get_transition(
    data: dict[str, _T], summary_result: Literal[False]
) -> tuple[
    dict[str, object],
    Transition,
    _TransitionRunArguments,
    _SummaryArguments | None,
]: ...


@overload
def get_transition(
    data: dict[str, _T], summary_result: bool
) -> tuple[
    dict[str, object],
    Transition,
    _TransitionRunArguments,
    _SummaryArguments | None,
]: ...


def get_transition(
    data: dict[str, object], summary_result: bool = True
) -> tuple[
    dict[str, object],
    Transition,
    _TransitionRunArguments,
    _SummaryArguments | None,
]:
    """
    The function to get the Summary, Transition object, contract arguments,
    and summary arguments used.

    Parameters
    ----------
    data: dict
        The dictionary of the data input.

    Returns
    -------
    summary_skk: dict
        The executive summary of the contract.
    contract:
        The Transition contract object.
    contract_arguments_dict: dict
        The contract arguments used in running the contract calculation.
    summary_arguments_dic: dict
        The summary arguments used in retrieving the executive summary of the contract.

    """
    contract_1_data = _require_mapping(data["contract_1"], field="contract_1")
    contract_2_data = _require_mapping(data["contract_2"], field="contract_2")

    # Defining contract_1
    if (
        contract_1_data["costrecovery"] is not None
        and contract_1_data["grosssplit"] is None
    ):
        _, contract_1, contract_arguments_1, _ = get_costrecovery(
            data=contract_1_data, summary_result=False
        )

    elif (
        contract_1_data["grosssplit"] is not None
        and contract_1_data["costrecovery"] is None
    ):
        _, contract_1, contract_arguments_1, _ = get_grosssplit(
            data=contract_1_data, summary_result=False
        )

    else:
        raise ContractException("Contract type is not recognized")

    # Defining contract_2
    if (
        contract_2_data["costrecovery"] is not None
        and contract_2_data["grosssplit"] is None
    ):
        _, contract_2, contract_arguments_2, _ = get_costrecovery(
            data=contract_2_data, summary_result=False
        )

    elif (
        contract_2_data["grosssplit"] is not None
        and contract_2_data["costrecovery"] is None
    ):
        _, contract_2, contract_arguments_2, _ = get_grosssplit(
            data=contract_2_data, summary_result=False
        )

    else:
        raise ContractException("Contract type is not recognized")

    # generating the transition contract object
    contract = Transition(
        contract1=contract_1,
        contract2=contract_2,
        argument_contract1=_require_mapping(
            contract_arguments_1, field="argument_contract1"
        ),
        argument_contract2=_require_mapping(
            contract_arguments_2, field="argument_contract2"
        ),
    )

    # Generating the transition contract arguments
    contract_arguments_dict = _transition_run_arguments(data["contract_arguments"])

    # Running the transition contract
    contract.run(**contract_arguments_dict)

    if summary_result is True:
        # Filling the summary arguments
        summary_arguments_dict = get_summary_dict(data=data)
        summary = _external_summary(contract, summary_arguments_dict)

        # Converting the summary format to skk summary format
        summary_skk = convert_summary_to_dict(dict_object=summary)

        # Adding the execution info
        summary_skk = add_execution_info(data=summary_skk)
        return summary_skk, contract, contract_arguments_dict, summary_arguments_dict

    return {}, contract, contract_arguments_dict, None


def add_execution_info(data: dict[str, object]) -> dict[str, object]:
    """
    Function to adding the execution info into a dictionary.

    Parameters
    ----------
    data: dict
        The dictionary which will added with execution info

    Returns
    -------
    out: dict
        Dictionary containing the execution info
    """
    # Defining the execution date
    execution_date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    # Defining the PySCnomics Version
    from importlib.metadata import PackageNotFoundError

    try:
        package_version = "1.5.0"
        # package_version = version("pyscnomics")

    except PackageNotFoundError:
        package_version = "unknown"

    # The following codes are replaced in version 1.4.0
    # package_version = " "
    # try:
    #     from importlib.metadata import version
    #     package_version = version("pyscnomics")
    # except:
    #     pass

    # Parsing the data into the data output
    execution_info = {
        "execution_datetime": execution_date,
        "package_version": package_version,
    }

    data["execution_info"] = execution_info

    return data


def get_contract_table(
    data: dict[str, object], contract_type: str = "Cost Recovery"
) -> tuple[dict[str, object], list[object]]:
    """
    Function to get the cash flow table of the contract that has been run.

    Parameters
    ----------
    data: dict
        The dictionary of the data input.
    contract_type: str
        The option for the contract type.
        The available option are: ['Cost Recovery', 'Gross Split']

    Returns
    -------
    table_all_dict: dict
        The dictionary containing the oil, gas and consolidated cash flow table.
    """

    # Adjusting the variable to the corresponding contract type
    if contract_type == "Cost Recovery":
        contract = get_costrecovery(data=data)[1]
        year_column = "years"
        # year_column = "Year"

    elif contract_type == "Gross Split":
        contract = get_grosssplit(data=data)[1]
        year_column = "years"
        # year_column = "Years"

    elif contract_type == "Base Project":
        contract = get_baseproject(data=data)[1]
        year_column = "years"
        # year_column = "Years"

    elif contract_type == "Transition":
        contract = get_transition(data=data, summary_result=False)[1]
        year_column = "years"
        # year_column = "Year"

    else:
        raise ValueError(f"Invalid contract type: {contract_type!r}")

    if isinstance(contract, Transition):
        transition_table: TransitionTable = get_table(contract=contract)
        table_oil_1, table_oil_2 = transition_table[0]
        table_gas_1, table_gas_2 = transition_table[1]
        table_consolidated_1, table_consolidated_2 = transition_table[2]
        table_all_dict: dict[str, object] = {
            "contract_1": {
                "oil": table_oil_1.set_index(table_oil_1.columns[0]).to_dict(),
                "gas": table_gas_1.set_index(table_gas_1.columns[0]).to_dict(),
                "consolidated": table_consolidated_1.set_index(
                    table_consolidated_1.columns[0]
                ).to_dict(),
            },
            "contract_2": {
                "oil": table_oil_2.set_index(table_oil_2.columns[0]).to_dict(),
                "gas": table_gas_2.set_index(table_gas_2.columns[0]).to_dict(),
                "consolidated": table_consolidated_2.set_index(
                    table_consolidated_2.columns[0]
                ).to_dict(),
            },
        }

    else:
        single_contract: BaseProject | CostRecovery | GrossSplit = contract
        contract_table: ContractTable = get_table(contract=single_contract)
        table_oil, table_gas, table_consolidated = contract_table
        table_all_dict = {
            "oil": table_oil.set_index(year_column).to_dict(),
            "gas": table_gas.set_index(year_column).to_dict(),
            "consolidated": table_consolidated.set_index(year_column).to_dict(),
        }
    # Adding the execution info
    table_all_dict = add_execution_info(data=table_all_dict)

    warnings: list[object] = []
    if not isinstance(contract, Transition):
        warnings.extend(contract.warning_messages)
    return table_all_dict, warnings


def _run_case_contract(
    data: dict[str, object], ctr: int
) -> BaseProject | CostRecovery | GrossSplit | Transition:
    """Build and run one combine/incremental case contract."""
    if ctr == 0:
        return get_baseproject(data=data, summary_result=False)[1]
    if ctr == 1:
        return get_costrecovery(data=data, summary_result=False)[1]
    if ctr == 2:
        return get_grosssplit(data=data, summary_result=False)[1]
    return get_transition(data=data, summary_result=False)[1]


def _merged_result(
    cases: list[_CombineCase], options: _CombineOptions, mode: str
) -> list[dict[str, object]]:
    """Run Summary merge and return per-case rows plus the derived row."""
    contracts: list[BaseProject | CostRecovery | GrossSplit | Transition] = []
    per_case: list[dict[str, object]] = []
    for case in cases:
        contract = _run_case_contract(data=case["data"], ctr=case["ctr"])
        contracts.append(contract)
        if isinstance(contract, Transition):
            transition_table = get_table(contract=contract)
            table_frame = transition_table[2][1]
        else:
            contract_table = get_table(contract=contract)
            table_frame = contract_table[2]
        table = table_frame.set_index(table_frame.columns[0]).to_dict()
        per_case.append(
            {
                "cId": case["cId"],
                "ctr": case["ctr"],
                "summary": _case_summary(data=case["data"], ctr=case["ctr"]),
                "table_consolidated": table,
            }
        )

    summary = Summary(
        contract=tuple(contracts),
        reference_year=options["reference_year"],
        inflation_rate=options["inflation_rate"],
        discount_rate=options["discount_rate"],
        npv_mode=_require_value(
            convert_str_to_npvmode(str_object=options["npv_mode"]),
            field="options.npv_mode",
        ),
        discounting_mode=_require_value(
            convert_str_to_discountingmode(str_object=options["discounting_mode"]),
            field="options.discounting_mode",
        ),
        profitability_discounted=options["profitability_discounted"],
    )
    merged = summary.case_combine() if mode == "combine" else summary.case_incremental()
    exec_summary = summary.get_executive_summary()
    merged_lists: dict[str, object] = {}
    for key, value in merged.items():
        if _is_numeric_array(value):
            merged_lists[key] = _to_list(value, field=key)
        else:
            merged_lists[key] = value
    merged_years = _require_list(merged_lists.pop("years"), field="years")
    merged_columns: dict[str, list[object]] = {
        key: value for key, value in merged_lists.items() if isinstance(value, list)
    }
    merged_columns["years"] = merged_years
    merged_table: dict[str, object] = {
        key: dict(zip(merged_years, value, strict=False))
        for key, value in merged_columns.items()
    }
    per_case.append(
        {
            "cId": 0,
            "ctr": -1,
            "summary": _clean_summary(exec_summary),
            "table_consolidated": merged_table,
        }
    )
    return per_case


def get_combine_summary(payload: dict[str, object]) -> dict[str, object]:
    """
    Combine case summaries plus a merged derived row.

    Reuses the core ``Summary`` combine merge for the economic results and
    the canonical table generator for each case's year-keyed consolidated
    table.
    """
    per_case = _merged_result(
        cases=_require_combine_cases(payload["cases"]),
        options=_require_combine_options(payload["options"]),
        mode="combine",
    )
    return {"combine": per_case}


def get_increment_summary(payload: dict[str, object]) -> dict[str, object]:
    """Incremental case summaries plus a merged derived row (base - others)."""
    per_case = _merged_result(
        cases=_require_combine_cases(payload["cases"]),
        options=_require_combine_options(payload["options"]),
        mode="incremental",
    )
    return {"increment": per_case}


def _clean_summary(summary: Mapping[str, object]) -> dict[str, object]:
    """Convert numpy scalars to JSON-native values and drop None."""
    cleaned: dict[str, object] = {}
    for key, value in summary.items():
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            continue
        cleaned[key] = value.item() if isinstance(value, np.generic) else value
    return cleaned


def _case_summary(data: dict[str, object], ctr: int) -> dict[str, object]:
    """Return the SKK summary for one case (used only for per-case rows)."""
    if ctr == 0:
        return get_baseproject(data=data, summary_result=True)[0]
    if ctr == 1:
        return get_costrecovery(data=data, summary_result=True)[0]
    if ctr == 2:
        return get_grosssplit(data=data, summary_result=True)[0]
    return get_transition(data=data)[0]


def get_detailed_summary(
    data: dict[str, object], contract_type: str
) -> dict[str, object]:
    """
    Generate a detailed economic summary for a given PSC (Production Sharing Contract)
    type.

    This method dynamically selects and executes the appropriate PSC computation
    function (e.g., Cost Recovery, Gross Split, Transition, or Base Project) based
    on the specified contract type.

    The function returns a detailed economic summary generated from the selected
    contract model.

    Parameters
    ----------
    data : dict
        Input data containing the economic parameters and production profile required
        for the PSC evaluation.
        The expected structure depends on the specific contract type.
    contract_type : str
        Type of PSC contract. Must be one of the following:
            - "Cost Recovery"
            - "Gross Split"
            - "Transition"
            - "Base Project"

    Returns
    -------
    dict
        A dictionary representing the detailed economic summary generated by the
        selected PSC model. The contents typically include key financial and fiscal
        indicators such as government take, contractor take, NPV, and other
        economic metrics derived from the contract's `get_summary` method.
    """

    if contract_type == "Cost Recovery":
        _, contract, _, summary_args = get_costrecovery(data=data)
        return dict(_contract_summary(contract, summary_args))
    if contract_type == "Gross Split":
        _, contract, _, summary_args = get_grosssplit(data=data)
        return dict(_contract_summary(contract, summary_args))
    if contract_type == "Transition":
        _, contract, _, summary_args = get_transition(data=data)
        return dict(_external_summary(contract, summary_args))
    if contract_type == "Base Project":
        _, contract, _, summary_args = get_baseproject(data=data)
        return dict(_contract_summary(contract, summary_args))
    raise ValueError(f"Invalid contract type: {contract_type!r}")


def get_ltp(
    volume: float, start_year: int, end_year: int, fluid_type: FluidType
) -> npt.NDArray[np.float64]:
    """
    Function to get the ltp array.

    Parameters
    ----------
    volume:float
        The volume of the reserves.
    start_year:int
        The start year.
    end_year:int
        The end year.
    fluid_type: FluidType
        The FluidType of the corresponding volume.

    Returns
    -------
    out: np.ndarray
        The array of ltp

    """
    # Condition checking for the fluid type for initiating the array of ltp
    if fluid_type == FluidType.OIL:
        rate_ltp = oil_ltp_predict(volume=volume)

    elif fluid_type == FluidType.GAS:
        rate_ltp = gas_ltp_predict(volume=volume)

    else:
        raise LTPModelException(f"Unsupported Fluid Type {fluid_type} ")

    # Initiating the array of years
    year_arr = np.arange(start_year, end_year + 1)
    gap = len(year_arr) - len(rate_ltp)

    # Checking condition for the given years gap
    if gap < 0:
        raise LTPModelException(
            f"Forecast years from {start_year} to {end_year} is too short. Please "
            + f"set the end_year at least until {end_year + (-1 * gap)}"
        )

    rate_gap = np.zeros(gap)
    rate_adjusted = np.concatenate([rate_gap, rate_ltp])

    return rate_adjusted


def get_ltp_dict(data: dict[str, object]) -> dict[str, object]:
    """
    The function to get the list of LTP from the given reserves volume.

    Parameters
    ----------
    data: dict
        The dictionary of the data input.

    Returns
    -------
    """

    volume = _require_float(data["volume"], field="volume")
    start_year = _require_int(data["start_year"], field="start_year")
    end_year = _require_int(data["end_year"], field="end_year")
    fluid_type = _require_value(
        read_fluid_type(fluid=_require_str(data["fluid_type"], field="fluid_type")),
        field="fluid_type",
    )

    ltp_array = get_ltp(
        volume=volume,
        start_year=start_year,
        end_year=end_year,
        fluid_type=fluid_type,
    )

    result = _frame_to_payload(
        pd.DataFrame(
            {
                "year": np.arange(start_year, end_year + 1).tolist(),
                "ltp": ltp_array.tolist(),
            }
        ).set_index("year")
    )
    return result


def get_rdp(
    start_year: int,
    end_year: int,
    year_rampup: int,
    drate: float,
    q_plateau_ratio: float,
    q_min_ratio: float,
    volume: float,
) -> npt.NDArray[np.float64]:
    """
    The function to get the RDP array.

    Parameters
    ----------
    start_year
    end_year
    year_rampup
    drate
    q_plateau_ratio
    q_min_ratio
    volume

    Returns
    -------
    out: np.ndarray
        The array of rdp
    """

    # Initiating Model
    model = RPDModel(
        year_rampup=year_rampup,
        drate=drate,
        q_plateau_ratio=q_plateau_ratio,
        q_min_ratio=q_min_ratio,
    )

    rate_rdp = model.predict(volume)

    # Initiating the array of years
    year_arr = np.arange(start_year, end_year + 1)
    gap = len(year_arr) - len(rate_rdp)

    # Checking condition for the given years gap
    if gap < 0:
        raise RDPModelException(
            f"Forecast years from {start_year} to {end_year} is too short. Please "
            + f"set the end_year at least until {end_year + (-1 * gap)}"
        )

    rate_gap = np.zeros(gap)
    rate_adjusted = np.concatenate([rate_gap, rate_rdp])

    return rate_adjusted


def get_rpd_dict(data: dict[str, object]) -> dict[str, object]:
    """
    The function to get the list of RPD from the given reserves volume.

    Parameters
    ----------
    data: dict
        The dictionary of the data input.

    Returns
    -------
    """

    # Initiating the data input
    year_rampup = _require_int(data["year_rampup"], field="year_rampup")
    drate = _require_float(data["drate"], field="drate")
    q_plateau_ratio = _require_float(data["q_plateau_ratio"], field="q_plateau_ratio")
    q_min_ratio = _require_float(data["q_min_ratio"], field="q_min_ratio")
    volume = _require_float(data["volume"], field="volume")
    start_year = _require_int(data["start_year"], field="start_year")
    end_year = _require_int(data["end_year"], field="end_year")

    # Get the array of rdp
    rdp_array = get_rdp(
        start_year=start_year,
        end_year=end_year,
        year_rampup=year_rampup,
        drate=drate,
        q_plateau_ratio=q_plateau_ratio,
        q_min_ratio=q_min_ratio,
        volume=volume,
    )

    result = _frame_to_payload(
        pd.DataFrame(
            {
                "year": np.arange(start_year, end_year + 1).tolist(),
                "rpd": rdp_array.tolist(),
            }
        ).set_index("year")
    )
    return result


def get_transition_split(data: dict[str, object]) -> dict[str, object]:
    """
    The function to get the contractor split information from Transition Contract Scheme.

    Parameters
    ----------
    data: dict
        The dictionary of the data input.

    Returns
    -------
    dict
        The dictionary containing the information of the contractor split.

    """
    contract_1_data = _require_mapping(data["contract_1"], field="contract_1")
    contract_2_data = _require_mapping(data["contract_2"], field="contract_2")

    # Defining contract_1
    if (
        contract_1_data["costrecovery"] is not None
        and contract_1_data["grosssplit"] is None
    ):
        _, contract_1, contract_arguments_1, _ = get_costrecovery(
            data=contract_1_data, summary_result=False
        )

    elif (
        contract_1_data["grosssplit"] is not None
        and contract_1_data["costrecovery"] is None
    ):
        _, contract_1, contract_arguments_1, _ = get_grosssplit(
            data=contract_1_data, summary_result=False
        )

    else:
        raise ContractException("Contract type is not recognized")

    # Defining contract_2
    if (
        contract_2_data["costrecovery"] is not None
        and contract_2_data["grosssplit"] is None
    ):
        _, contract_2, contract_arguments_2, _ = get_costrecovery(
            data=contract_2_data, summary_result=False
        )

    elif (
        contract_2_data["grosssplit"] is not None
        and contract_2_data["costrecovery"] is None
    ):
        _, contract_2, contract_arguments_2, _ = get_grosssplit(
            data=contract_2_data, summary_result=False
        )

    else:
        raise ContractException("Contract type is not recognized")

    # generating the transition contract object
    contract = Transition(
        contract1=contract_1,
        contract2=contract_2,
        argument_contract1=_require_mapping(
            contract_arguments_1, field="argument_contract1"
        ),
        argument_contract2=_require_mapping(
            contract_arguments_2, field="argument_contract2"
        ),
    )

    # Generating the transition contract arguments
    contract_arguments_dict = _transition_run_arguments(data["contract_arguments"])

    # Running the transition contract
    contract.run(**contract_arguments_dict)

    # Making the base for the loops and container of the result
    result: dict[str, object] = {}

    # Retrieving the split information from first contract
    for index, contract in enumerate([contract_1, contract_2]):
        if isinstance(contract, GrossSplit):
            result["contract_" + str(index + 1)] = _gross_split_reporting_payload(
                contract, include_components=False
            )
        else:
            result["contract_" + str(index + 1)] = {}

    return result


def get_economic_limit(data: dict[str, object]) -> np.int64:
    """
    The function to get the information of economic limit years from
    a cashflow using selected method.

    Parameters
    ----------
    data: dict

    Returns
    -------
    int
        The index
    """

    years: npt.NDArray[np.int64] = np.asarray(data["years"], dtype=np.int64)
    cash_flow: npt.NDArray[np.float64] = np.asarray(data["cash_flow"], dtype=np.float64)
    method = convert_to_method_limit(target=data["method"])
    if not isinstance(method, LimitMethod):
        raise method
    index_limit = econ_limit(cashflow=cash_flow, method=method)

    return np.take(years, index_limit)


def get_asr_expenditures(data: dict[str, object]) -> dict[str, object]:
    """
    Compute and return the post-tax ASR (Abandonment, Site Restoration)
    expenditures from the given input data.

    This function constructs a pseudo Base Project environment using
    the provided ASR and setup data, runs the project simulation,
    and extracts the ASR expenditures for oil and gas over project years.

    Parameters
    ----------
    data : dict
        Input data dictionary containing at least the following keys:
        - ``asr`` : ASR-related cost data.
        - ``setup`` : Dictionary with setup information
          (e.g., start/end date, onstream dates).
        - ``lifting`` : Lifting profile or data for oil and gas.

    Returns
    -------
    dict
        Dictionary of ASR expenditures with project years as keys and
        sub-dictionaries containing:
        - ``oil_asr_expenditures`` : Post-tax ASR expenditures for oil.
        - ``gas_asr_expenditures`` : Post-tax ASR expenditures for gas.

    Notes
    -----
    - The function internally mimics a Base Project execution by
      constructing a minimal pseudo-input structure compatible with
      :func:`get_baseproject`.
    - ASR expenditures are extracted from the executed contract object
      and converted into a dictionary using a Pandas DataFrame.
    - No profitability or inflation adjustments are applied.
    """

    # Initiating the asr data
    asr_pseudo = {"asr": data["asr"]}

    # Mimics the baseproject data
    _setup = _require_mapping(data["setup"], field="setup")

    data_pseudo = {
        "setup": {
            "start_date": _setup["start_date"],
            "end_date": _setup["end_date"],
            "oil_onstream_date": _setup["oil_onstream_date"],
            "gas_onstream_date": _setup["gas_onstream_date"],
            "approval_year": None,
            "is_pod_1": False,
        },
        "summary_arguments": {
            "discount_rate_start_year": None,
            "inflation_rate": 0.0,
            "discount_rate": 0.1,
            "npv_mode": "Full Cycle Nominal Terms",
            "discounting_mode": "Mid Year",
            "profitability_discounted": False,
        },
        "contract_arguments": {
            "sulfur_revenue": "Addition to Gas Revenue",
            "electricity_revenue": "Addition to Oil Revenue",
            "co2_revenue": "Addition to Gas Revenue",
            # "sunk_cost_reference_year": None,
            # "year_inflation": 0,
            "inflation_rate": 0,
            "vat_rate": 0,
            "inflation_rate_applied_to": "CAPEX",
        },
        "lifting": data["lifting"],
        "capital": None,
        "intangible": None,
        "opex": None,
        "asr": asr_pseudo["asr"],
        "lbt": None,
        "cost_of_sales": None,
    }

    # Parsing the data into base project dataclass
    # contract = get_baseproject(data=data_pseudo, summary_result=False)[1]
    _, contract, _contract_arguments, _ = get_baseproject(
        data=data_pseudo, summary_result=False
    )

    # Convert BaseProject instance into dictionary using method `vars()`
    contract_as_dict = vars(contract)

    # Returning the ASR Expenditures
    df = pd.DataFrame(
        {
            "project_years": contract.project_years,
            "oil_asr_expenditures": contract_as_dict["_oil_asr_expenditures_post_tax"],
            "gas_asr_expenditures": contract_as_dict["_gas_asr_expenditures_post_tax"],
            # "oil_asr_expenditures": contract._oil_asr_expenditures_post_tax,
            # "gas_asr_expenditures": contract._gas_asr_expenditures_post_tax,
        }
    )

    return _frame_to_payload(df.set_index("project_years"))


def get_lbt_expenditures(data: dict[str, object]) -> dict[str, object]:
    """
    Compute and return the post-tax ASR (Abandonment, Site Restoration)
    expenditures from the given input data.

    This function constructs a pseudo Base Project environment using
    the provided ASR and setup data, runs the project simulation,
    and extracts the ASR expenditures for oil and gas over project years.

    Parameters
    ----------
    data : dict
        Input data dictionary containing at least the following keys:
        - ``asr`` : ASR-related cost data.
        - ``setup`` : Dictionary with setup information
          (e.g., start/end date, onstream dates).
        - ``lifting`` : Lifting profile or data for oil and gas.

    Returns
    -------
    dict
        Dictionary of ASR expenditures with project years as keys and
        sub-dictionaries containing:
        - ``oil_asr_expenditures`` : Post-tax ASR expenditures for oil.
        - ``gas_asr_expenditures`` : Post-tax ASR expenditures for gas.

    Notes
    -----
    - The function internally mimics a Base Project execution by
      constructing a minimal pseudo-input structure compatible with
      :func:`get_baseproject`.
    - ASR expenditures are extracted from the executed contract object
      and converted into a dictionary using a Pandas DataFrame.
    - No profitability or inflation adjustments are applied.
    """

    # Initiating the LBT data
    lbt_pseudo = {"lbt": data["lbt"]}

    # Mimics the baseproject data
    _setup = _require_mapping(data["setup"], field="setup")

    # Mimics the baseproject data
    data_pseudo = {
        "setup": {
            "start_date": _setup["start_date"],
            "end_date": _setup["end_date"],
            "oil_onstream_date": _setup["oil_onstream_date"],
            "gas_onstream_date": _setup["gas_onstream_date"],
            "approval_year": None,
            "is_pod_1": False,
        },
        "summary_arguments": {
            "discount_rate_start_year": None,
            "inflation_rate": 0.0,
            "discount_rate": 0.1,
            "npv_mode": "Full Cycle Nominal Terms",
            "discounting_mode": "Mid Year",
            "profitability_discounted": False,
        },
        "contract_arguments": {
            "sulfur_revenue": "Addition to Gas Revenue",
            "electricity_revenue": "Addition to Oil Revenue",
            "co2_revenue": "Addition to Gas Revenue",
            # "sunk_cost_reference_year": None,
            # "year_inflation": 0,
            "inflation_rate": 0,
            "vat_rate": 0,
            "inflation_rate_applied_to": "CAPEX",
        },
        "lifting": data["lifting"],
        "capital": None,
        "intangible": None,
        "opex": None,
        "asr": None,
        "lbt": lbt_pseudo["lbt"],
        "cost_of_sales": None,
    }

    # Parsing the data into base project dataclass
    # contract = get_baseproject(data=data_pseudo, summary_result=False)[1]
    _, contract, _contract_arguments, _ = get_baseproject(
        data=data_pseudo, summary_result=False
    )

    # Convert BaseProject instance into dictionary using method `vars()`
    contract_as_dict = vars(contract)

    # Returning the LBT Expenditures
    df = pd.DataFrame(
        {
            "project_years": contract.project_years,
            "oil_lbt_expenditures": contract_as_dict["_oil_lbt_expenditures_post_tax"],
            "gas_lbt_expenditures": contract_as_dict["_gas_lbt_expenditures_post_tax"],
            # "oil_lbt_expenditures": contract._oil_lbt_expenditures_post_tax,
            # "gas_lbt_expenditures": contract._gas_lbt_expenditures_post_tax,
        }
    )

    return _frame_to_payload(df.set_index("project_years"))


def get_contract_optimization(
    data: dict[str, object], contract_type: str = "Cost Recovery"
) -> dict[str, object]:
    """
    The function to run contract optimization. Resulting optimization result
    in dictionary format.

    Parameters
    ----------
    data: dict
        The dictionary of the data input.
    contract_type: str
        The option for the contract type. The available option are:
        ['Cost Recovery', 'Gross Split']

    Returns
    -------
    result_parameters: dict
        The result of the optimization in dictionary format
    """

    if "optimization_arguments" not in data:
        raise ContractException(
            "The payload does not have the optimization_arguments key"
        )

    if data["optimization_arguments"] is None:
        raise ContractException(
            "The payload optimization_arguments does not have any values"
        )

    optimization_arguments = _require_mapping(
        data["optimization_arguments"], field="optimization_arguments"
    )
    optimization_bounds = _require_mapping(
        optimization_arguments["dict_optimization"],
        field="optimization_arguments.dict_optimization",
    )

    # Converting the parameters in dict_optimization to the corresponding enum
    optimization_parameters = [
        _require_value(
            convert_str_to_optimization_parameters(str_object=i),
            field="optimization_arguments.dict_optimization.parameter",
        )
        for i in _require_list(
            optimization_bounds["parameter"],
            field="optimization_arguments.dict_optimization.parameter",
        )
    ]

    # Generating the dictionary of the optimization arguments
    dict_optimization: _OptimizationConfiguration = {
        "parameter": optimization_parameters,
        "min": convert_list_to_array_float(data_list=optimization_bounds["min"]),
        "max": convert_list_to_array_float(data_list=optimization_bounds["max"]),
    }

    # Filling the optimization arguments with target_optimization and target_parameter,
    target_optimization_value = _require_float(
        optimization_arguments["target_optimization"],
        field="optimization_arguments.target_optimization",
    )
    target_parameter = _require_value(
        convert_str_to_optimization_targetparameter(
            str_object=optimization_arguments["target_parameter"]
        ),
        field="optimization_arguments.target_parameter",
    )

    # Retrieving the contract, contract_arguments_dict,
    # summary_arguments_dict based on the contract type
    if contract_type == "Transition":
        transition = get_transition(data=data)[1]
        transition_arguments = get_transition(data=data)[2]
        summary_argument = get_transition(data=data)[3]
        # Retrieve the original useful life of the capital cost
        useful_life_original = _to_list(
            transition.contract2.capital_cost_total.useful_life,
            field="useful_life",
        )

        transition_result = optimize_psc_core(
            dict_optimization=dict_optimization,
            contract=transition,
            contract_arguments=transition_arguments,
            target_optimization_value=target_optimization_value,
            summary_argument=_optimization_summary_payload(summary_argument),
            target_parameter=target_parameter,
        )
        list_str = transition_result[0]
        list_params_value: list[object] = [item for item in transition_result[1]]
        result_optimization = transition_result[2]
        transition_contracts = transition_result[3]
        contract_optimized: CostRecovery | GrossSplit = transition_contracts[
            -1
        ].contract2

    else:
        if contract_type == "Cost Recovery":
            psc_contract = get_costrecovery(data=data)[1]
            contract_arguments = get_costrecovery(data=data)[2]
            summary_argument = get_costrecovery(data=data)[3]
        elif contract_type == "Gross Split":
            psc_contract = get_grosssplit(data=data)[1]
            contract_arguments = get_grosssplit(data=data)[2]
            summary_argument = get_grosssplit(data=data)[3]
        else:
            raise ContractException(f"Contract type is not recognized: {contract_type}")

        # Retrieve the original useful life of the capital cost
        useful_life_original = _to_list(
            psc_contract.capital_cost_total.useful_life,
            field="useful_life",
        )
        optimization_output = optimize_psc(
            dict_optimization=dict_optimization,
            contract=psc_contract,
            contract_arguments=contract_arguments,
            target_optimization_value=target_optimization_value,
            summary_arguments=_optimization_summary_payload(summary_argument),
            target_parameter=target_parameter,
        )
        list_str = optimization_output[0]
        list_params_value = [item for item in optimization_output[1]]
        result_optimization = optimization_output[2]
        psc_contracts = optimization_output[3]
        contract_optimized = psc_contracts[-1]

    # Treatment to add the useful life of optimization into the result
    def get_enum_index(
        enum_list: list[OptimizationParameter],
        element: OptimizationParameter,
    ) -> int | None:
        """
        Function to get the index of the OptimizationParameter.DEPRECIATION_ACCELERATION.

        Parameters
        ----------
        enum_list: list
            The source of list.
        element: any
            The corresponding element
        Returns
        -------
        out : int | None
        """
        try:
            return enum_list.index(element)
        except ValueError:
            return None

    # Get the index of the depreciation optimization parameter
    index_depreciation = get_enum_index(
        enum_list=optimization_parameters,
        element=OptimizationParameter.DEPRECIATION_ACCELERATION,
    )

    # Adding the information of optimized useful life into the list_params_value
    if index_depreciation is not None:
        optimized_capital_cost = {
            "year": _to_list(
                contract_optimized.capital_cost_total.expense_year,
                field="expense_year",
            ),
            "cost_allocation": contract_optimized.capital_cost_total.cost_allocation,
            "cost": require_float_array(
                contract_optimized.capital_cost_total.cost, "cost"
            ).tolist(),
            "pis_year": _to_list(
                contract_optimized.capital_cost_total.pis_year,
                field="pis_year",
            ),
            "useful_life_original": useful_life_original,
            "useful_life_optimized": _to_list(
                contract_optimized.capital_cost_total.useful_life,
                field="useful_life",
            ),
            "description": contract_optimized.capital_cost_total.description,
        }

        # Adding optimized_capital_cost into the result of the optimization
        list_params_value[index_depreciation] = {
            "depreciation acceleration": list_params_value[index_depreciation],
            "optimized_useful_life": optimized_capital_cost,
        }

    # Forming the optimization result into a dictionary object
    optimization_result = {
        "list_str": list_str,
        "list_params_value": list_params_value,
    }

    # Converting the result into dictionary format
    result_parameters = _frame_to_payload(
        pd.DataFrame(optimization_result).set_index("list_str")
    )

    result_parameters["optimization_result"] = result_optimization

    # Adding the execution info
    result_parameters = add_execution_info(data=result_parameters)

    return result_parameters


def get_sensitivity(data: dict[str, object], contract_type: str) -> dict[str, object]:
    """
    Run sensitivity analysis for a given PSC contract type.

    This function retrieves the appropriate contract instance, contract arguments,
    and summary arguments based on the specified `contract_type`, and then performs
    a sensitivity analysis using the `sensitivity_psc` function.

    The analysis adjusts selected economic parameters (CAPEX, OPEX, OILPRICE, GASPRICE,
    ILLIFTING, GASLIFTING) according to the sensitivity configuration provided in the
    input dictionary.

    Parameters
    ----------
    data : dict
        A dictionary containing the full project input payload.
    contract_type : str
        The type of PSC contract to be evaluated. Accepted values include:
        - `"Base Project"`
        - `"Cost Recovery"`
        - `"Gross Split"`
        - `"Transition"`

    Returns
    -------
    dict
        A nested dictionary containing sensitivity results for several key indicators.

    Notes
    -----
    - Internally, this function calls one of the following constructors based on
      `contract_type`:
        * `get_baseproject(data)`
        * `get_costrecovery(data)`
        * `get_grosssplit(data)`
        * `get_transition(data)`
    - Each constructor is assumed to return a tuple whose elements include:
        `(status, contract_object, contract_arguments, summary_arguments)`.
    - The sensitivity analysis is performed by calling `sensitivity_psc(...)` with
      `dataframe_output=False`, so the result is returned as a dictionary rather than
      a collection of DataFrames.
    """

    # Check for invalid conditions
    if "sensitivity_arguments" not in data:
        raise ContractException(
            "The payload does not have the sensitivity_arguments key"
        )

    if data["sensitivity_arguments"] is None:
        raise ContractException(
            "The payload sensitivity_arguments does not have any values"
        )

    # Retrieving the contract, contract_arguments_dict,
    # summary_arguments_dict based on the contract type.
    # CostRecovery
    if contract_type == "Cost Recovery":
        _, contract, contract_arguments, summary_argument = get_costrecovery(data=data)

    # GrossSplit
    elif contract_type == "Gross Split":
        _, contract, contract_arguments, summary_argument = get_grosssplit(data=data)

    # Transition
    elif contract_type == "Transition":
        _, contract, contract_arguments, summary_argument = get_transition(data=data)

    # BaseProject
    else:
        _, contract, contract_arguments, summary_argument = get_baseproject(data=data)

    # Constructing the sensitivity arguments
    sensitivity_arguments = _require_mapping(
        data["sensitivity_arguments"], field="sensitivity_arguments"
    )
    increment = _optional_float(
        sensitivity_arguments.get("increment"), field="sensitivity_arguments.increment"
    )
    sensitivity_result = sensitivity_psc(
        contract=contract,
        contract_arguments=_require_mapping(
            contract_arguments, field="contract_arguments"
        ),
        summary_arguments=_contract_summary_payload(summary_argument),
        min_deviation=_require_float(
            sensitivity_arguments["min_deviation"],
            field="sensitivity_arguments.min_deviation",
        ),
        max_deviation=_require_float(
            sensitivity_arguments["max_deviation"],
            field="sensitivity_arguments.max_deviation",
        ),
        base_value=_require_float(
            sensitivity_arguments["base_value"],
            field="sensitivity_arguments.base_value",
        ),
        step=_require_int(
            sensitivity_arguments.get("step", 10), field="sensitivity_arguments.step"
        ),
        increment=increment,
        dataframe_output=False,
    )

    return _require_mapping(sensitivity_result, field="sensitivity_result")


def get_uncertainty(data: dict[str, object], contract_type: str) -> dict[str, object]:
    """
    Run Monte Carlo uncertainty analysis for a specified PSC contract type.

    This function extracts contract setup, statistical parameters, and
    uncertainty configurations from the input data, then executes
    probabilistic simulations through `uncertainty_psc()`.

    Returns
    -------
    dict
        Dictionary containing probabilistic results (P10, P50, P90) and
        simulation details.

    Notes
    -----
    - Supported contract types: "Base Project", "Cost Recovery",
      "Gross Split", and "Transition".
    - Each uncertainty parameter (e.g., oil price, gas price, CAPEX, OPEX,
      lifting) is simulated using its mean, range, standard deviation,
      and selected probability distribution.
    - Raises `ContractException` if `uncertainty_arguments` are missing or invalid.
    """

    # Prepare input "uncertainty_arguments"
    ua = _require_mapping(
        _extract_from_dict(target_key="uncertainty_arguments", source=data),
        field="uncertainty_arguments",
    )

    # Retrieving the contract, contract_arguments_dict,
    # summary_arguments_dict based on the contract type
    if contract_type == "Cost Recovery":
        contract = get_costrecovery(data=data)[1]
        contract_arguments = get_costrecovery(data=data)[2]
        summary_argument = get_costrecovery(data=data)[3]

    elif contract_type == "Gross Split":
        contract = get_grosssplit(data=data)[1]
        contract_arguments = get_grosssplit(data=data)[2]
        summary_argument = get_grosssplit(data=data)[3]

    elif contract_type == "Transition":
        contract = get_transition(data=data)[1]
        contract_arguments = get_transition(data=data)[2]
        summary_argument = get_transition(data=data)[3]

    else:
        contract = get_baseproject(data=data)[1]
        contract_arguments = get_baseproject(data=data)[2]
        summary_argument = get_baseproject(data=data)[3]

    # Helper function
    def _convert_distribution_enum_to_str(target: str):
        return convert_to_uncertainty_distribution(target=ua[target])

    # Abbreviations
    to_str = _convert_distribution_enum_to_str

    result = uncertainty_psc(
        contract=contract,
        contract_arguments=_as_object_mapping(contract_arguments),
        summary_arguments=_contract_summary_payload(summary_argument),
        run_number=_require_int(
            ua["run_number"], field="uncertainty_arguments.run_number"
        ),
        min_oil_price=_optional_float(
            ua["min_oil_price"], field="uncertainty_arguments.min_oil_price"
        ),
        mean_oil_price=_optional_float(
            ua["mean_oil_price"], field="uncertainty_arguments.mean_oil_price"
        ),
        max_oil_price=_optional_float(
            ua["max_oil_price"], field="uncertainty_arguments.max_oil_price"
        ),
        min_gas_price=_optional_float(
            ua["min_gas_price"], field="uncertainty_arguments.min_gas_price"
        ),
        mean_gas_price=_optional_float(
            ua["mean_gas_price"], field="uncertainty_arguments.mean_gas_price"
        ),
        max_gas_price=_optional_float(
            ua["max_gas_price"], field="uncertainty_arguments.max_gas_price"
        ),
        min_opex=_optional_float(
            ua["min_opex"], field="uncertainty_arguments.min_opex"
        ),
        mean_opex=_optional_float(
            ua["mean_opex"], field="uncertainty_arguments.mean_opex"
        ),
        max_opex=_optional_float(
            ua["max_opex"], field="uncertainty_arguments.max_opex"
        ),
        min_capex=_optional_float(
            ua["min_capex"], field="uncertainty_arguments.min_capex"
        ),
        mean_capex=_optional_float(
            ua["mean_capex"], field="uncertainty_arguments.mean_capex"
        ),
        max_capex=_optional_float(
            ua["max_capex"], field="uncertainty_arguments.max_capex"
        ),
        min_lifting=_optional_float(
            ua["min_lifting"], field="uncertainty_arguments.min_lifting"
        ),
        mean_lifting=_optional_float(
            ua["mean_lifting"], field="uncertainty_arguments.mean_lifting"
        ),
        max_lifting=_optional_float(
            ua["max_lifting"], field="uncertainty_arguments.max_lifting"
        ),
        oil_price_stddev=_require_float(
            ua["oil_price_stddev"], field="uncertainty_arguments.oil_price_stddev"
        ),
        gas_price_stddev=_require_float(
            ua["gas_price_stddev"], field="uncertainty_arguments.gas_price_stddev"
        ),
        opex_stddev=_require_float(
            ua["opex_stddev"], field="uncertainty_arguments.opex_stddev"
        ),
        capex_stddev=_require_float(
            ua["capex_stddev"], field="uncertainty_arguments.capex_stddev"
        ),
        lifting_stddev=_require_float(
            ua["lifting_stddev"], field="uncertainty_arguments.lifting_stddev"
        ),
        oil_price_distribution=to_str("oil_price_distribution"),
        gas_price_distribution=to_str("gas_price_distribution"),
        opex_distribution=to_str("opex_distribution"),
        capex_distribution=to_str("capex_distribution"),
        lifting_distribution=to_str("lifting_distribution"),
    )
    return result


"""
FORMER APPROACH
---------------

def get_summary_object(
    data: dict[str, object], contract: CostRecovery | GrossSplit | Transition
) -> (dict, dict):
    
    # The function to get the summary dictionary object from the data and contract input.
    # 
    # Parameters
    # ----------
    # data: dict
    #     The dictionary of the data input
    # contract: CostRecovery | GrossSplit | Transition
    #     The contract object.
    # 
    # Returns
    # -------
    # summary: dict
    #     The summary of the contract ini dictionary format.
    # 
    # summary_arguments_dict: dict
    #     The summary arguments used in retrieving the summary of the contract.

    if contract is Transition:
        summary_arguments_dict = get_summary_dict(data=data)
        summary_arguments_dict["contract"] = contract
        summary = get_summary(**summary_arguments_dict)

    else:
        summary_arguments_dict = get_summary_dict(data=data)
        summary_arguments_dict["contract"] = contract
        summary = get_summary(**summary_arguments_dict)

    return summary, summary_arguments_dict

=============================================================================================

def get_uncertainty(data: dict[str, object], contract_type: str):
    
    # Filter unwanted inputs
    if "uncertainty_arguments" not in data:
        raise ContractException(
            "The payload does not have the uncertainty_arguments key"
        )

    if data["uncertainty_arguments"] is None:
        raise ContractException(
            "The payload uncertainty_arguments does not have any values"
        )

    # Prepare input "uncertainty_arguments"
    ua = _extract_from_dict(target_key="uncertainty_arguments", source=data)

    # Retrieving the contract, contract_arguments_dict,
    # summary_arguments_dict based on the contract type
    if contract_type == "Cost Recovery":
        contract = get_costrecovery(data=data)[1]
        contract_arguments = get_costrecovery(data=data)[2]
        summary_argument = get_costrecovery(data=data)[3]

    elif contract_type == "Gross Split":
        contract = get_grosssplit(data=data)[1]
        contract_arguments = get_grosssplit(data=data)[2]
        summary_argument = get_grosssplit(data=data)[3]

    elif contract_type == "Transition":
        contract = get_transition(data=data)[1]
        contract_arguments = get_transition(data=data)[2]
        summary_argument = get_transition(data=data)[3]

    else:
        contract = get_baseproject(data=data)[1]
        contract_arguments = get_baseproject(data=data)[2]
        summary_argument = get_baseproject(data=data)[3]

    # Helper function
    def _convert_distribution_enum_to_str(target: str):
        return convert_to_uncertainty_distribution(target=ua[target])

        # return convert_to_uncertainty_distribution(
        #     target=data["uncertainty_arguments"][target]
        # )

    # Abbreviations
    # ua = data["uncertainty_arguments"]
    to_str = _convert_distribution_enum_to_str

    # Constructing the sensitivity arguments
    uncertainty_kwargs = {
        # Base parameters
        "contract": contract,
        "contract_arguments": contract_arguments,
        "summary_arguments": summary_argument,
        "run_number": ua["run_number"],

        # Statistics parameters for OIL PRICE
        "min_oil_price": ua["min_oil_price"],
        "mean_oil_price": ua["mean_oil_price"],
        "max_oil_price": ua["max_oil_price"],

        # Statistics parameters for GAS PRICE
        "min_gas_price": ua["min_gas_price"],
        "mean_gas_price": ua["mean_gas_price"],
        "max_gas_price": ua["max_gas_price"],

        # Statistics parameters for OPEX
        "min_opex": ua["min_opex"],
        "mean_opex": ua["mean_opex"],
        "max_opex": ua["max_opex"],

        # Statistics parameters for CAPEX
        "min_capex": ua["min_capex"],
        "mean_capex": ua["mean_capex"],
        "max_capex": ua["max_capex"],

        # Statistics parameters for Lifting
        "min_lifting": ua["min_lifting"],
        "mean_lifting": ua["mean_lifting"],
        "max_lifting": ua["max_lifting"],

        # Standard deviations
        "oil_price_stddev": ua["oil_price_stddev"],
        "gas_price_stddev": ua["gas_price_stddev"],
        "opex_stddev": ua["opex_stddev"],
        "capex_stddev": ua["capex_stddev"],
        "lifting_stddev": ua["lifting_stddev"],

        # Distribution
        "oil_price_distribution": to_str("oil_price_distribution"),
        "gas_price_distribution": to_str("gas_price_distribution"),
        "opex_distribution": to_str("opex_distribution"),
        "capex_distribution": to_str("capex_distribution"),
        "lifting_distribution": to_str("lifting_distribution"),
    }

    return uncertainty_psc(**uncertainty_kwargs)
    
=============================================================================================
"""
