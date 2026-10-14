"""
Collection of functions to administer Monte Carlo simulation.
The code below is the modification of the code from PSCnomics.

The routine and result of this module is maintaining the
requirements of the PSCnomics.
"""

import copy
import math
import os
from collections.abc import Iterable, Mapping
from datetime import date
from numbers import Real
from typing import TypeGuard, TypeVar, TypedDict

# import traceback
import numpy as np
import numpy.typing as npt
from scipy.stats import uniform, triang, truncnorm
from pathos.pools import ProcessPool as Pool
from pathos.helpers import mp

from pyscnomics.tools.summary import get_summary
from pyscnomics.contracts.project import BaseProject, require_float_array
from pyscnomics.contracts.costrecovery import CostRecovery, RCICPInput
from pyscnomics.contracts.gross_split import GrossSplit
from pyscnomics.contracts.transition import Transition
from pyscnomics.econ import FluidType
from pyscnomics.econ.costs import ASR, CapitalCost, CostOfSales, Intangible, LBT, OPEX
from pyscnomics.input_validation import InputValidationError, engine_input_issue
from pyscnomics.validation_rules import uncertainty_parameters_violation
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.selection import (
    DeprMethod,
    DiscountingMode,
    FTPTaxRegime,
    GrossSplitRegime,
    InflationAppliedTo,
    NPVSelection,
    OtherRevenue,
    TaxRegime,
    TaxSplitTypeCR,
    UncertaintyDistribution,
    VariableSplit132024,
)
from pyscnomics.io.getattr import get_contract_attributes
from pyscnomics.api.converter import (
    convert_str_to_date,
    convert_str_to_int,
    convert_list_to_array_float_or_array,
    convert_dict_to_lifting,
    convert_dict_to_capital,
    convert_dict_to_intangible,
    convert_dict_to_opex,
    convert_dict_to_asr,
    convert_dict_to_lbt,
    convert_dict_to_cost_of_sales,
    convert_list_to_array_float,
    convert_list_to_array_float_or_array_or_none,
    convert_str_to_taxsplit,
    convert_str_to_npvmode,
    convert_str_to_discountingmode,
    convert_str_to_otherrevenue,
    convert_str_to_taxregime,
    convert_str_to_ftptaxregime,
    convert_str_to_depremethod,
    convert_str_to_inflationappliedto,
    convert_grosssplitregime_to_enum,
    convert_to_float,
    converter_reservoir_type_permen_2024,
    # read_fluid_type,
    # convert_to_method_limit,
    # convert_to_uncertainty_distribution,
    # convert_to_skk_summary_baseproject,
    # convert_summary_to_dict,
    # convert_str_to_optimization_parameters,
    # convert_str_to_optimization_targetparameter,
)

np.set_printoptions(suppress=True, precision=4)


FloatArray = npt.NDArray[np.float64]
Summary = Mapping[str, object]
SerializedContract = dict[str, object]
ProjectContract = BaseProject | CostRecovery | GrossSplit
T = TypeVar("T")


class BaseProjectKwargs(TypedDict):
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


class BaseProjectArguments(TypedDict):
    sulfur_revenue: OtherRevenue
    electricity_revenue: OtherRevenue
    co2_revenue: OtherRevenue
    vat_rate: FloatArray | float
    inflation_rate: FloatArray | float
    inflation_rate_applied_to: InflationAppliedTo | None


class SummaryArguments(TypedDict):
    discount_rate_start_year: int | None
    inflation_rate: float
    discount_rate: float
    npv_mode: NPVSelection
    discounting_mode: DiscountingMode
    profitability_discounted: bool


class CostRecoveryKwargs(BaseProjectKwargs):
    oil_ftp_is_available: bool
    oil_ftp_is_shared: bool
    oil_ftp_portion: FloatArray | float
    gas_ftp_is_available: bool
    gas_ftp_is_shared: bool
    gas_ftp_portion: FloatArray | float
    tax_split_type: TaxSplitTypeCR
    condition_dict: RCICPInput
    indicator_rc_icp_sliding: FloatArray | None
    oil_ctr_pretax_share: FloatArray | float
    gas_ctr_pretax_share: FloatArray | float
    oil_ic_rate: FloatArray | float
    gas_ic_rate: FloatArray | float
    ic_is_available: bool
    oil_cr_cap_rate: float
    gas_cr_cap_rate: float
    oil_dmo_volume_portion: FloatArray | float
    oil_dmo_fee_portion: FloatArray | float
    oil_dmo_holiday_duration: int
    gas_dmo_volume_portion: FloatArray | float
    gas_dmo_fee_portion: FloatArray | float
    gas_dmo_holiday_duration: int
    oil_carry_forward_depreciation: FloatArray | float
    gas_carry_forward_depreciation: FloatArray | float


class CostRecoveryArguments(BaseProjectArguments):
    ftp_tax_regime: FTPTaxRegime
    is_dmo_end_weighted: bool
    tax_regime: TaxRegime
    effective_tax_rate: FloatArray | float | None
    post_uu_22_year2001: bool
    depr_method: DeprMethod
    decline_factor: float | int
    sum_undepreciated_cost: bool
    oil_cost_of_sales_applied: bool
    gas_cost_of_sales_applied: bool


class GrossSplitKwargs(BaseProjectKwargs):
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
    oil_dmo_volume_portion: FloatArray | float
    oil_dmo_fee_portion: FloatArray | float
    oil_dmo_holiday_duration: int
    gas_dmo_volume_portion: FloatArray | float
    gas_dmo_fee_portion: FloatArray | float
    gas_dmo_holiday_duration: int
    oil_carry_forward_depreciation: FloatArray | float
    gas_carry_forward_depreciation: FloatArray | float


class GrossSplitArguments(BaseProjectArguments):
    cum_production_split_offset: FloatArray | float | None
    depr_method: DeprMethod
    decline_factor: float | int
    sum_undepreciated_cost: bool
    is_dmo_end_weighted: bool
    tax_regime: TaxRegime
    effective_tax_rate: FloatArray | float | int | None
    amortization: bool
    regime: GrossSplitRegime
    reservoir_type_permen_2024: VariableSplit132024.ReservoirType


class Parameter(TypedDict):
    id: int
    fluid: FluidType | None
    dist: UncertaintyDistribution
    min: float
    base: float
    max: float
    stddev: float


class MonteResult(TypedDict):
    n: int
    output: tuple[float, float, float, float, float, float]


class MonteCarloException(Exception):
    """Exception to be raised for a misuse of MonteCarlo class"""

    pass


class MonteCarloInputError(  # pyright: ignore[reportUnsafeMultipleInheritance]
    MonteCarloException, ValueError, InputValidationError
):
    """Verified user-input Monte Carlo error retaining its documented bases.

    The engine raised ``ValueError`` for the invalid distribution parameters and
    ``MonteCarloException`` for an unrecognized distribution, so this class keeps
    both catchable for Python library callers.
    """

    _input_code: str = "UNCERTAINTY_INPUT_INVALID"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        InputValidationError.__init__(
            self,
            (engine_input_issue(code=code or self._input_code, message=message),),
        )


def _extract_from_dict(
    target_key: str, source: Mapping[str, object], default: object = None
) -> object:
    """
    Check whether a key exists in a dictionary and return a safe value.

    Parameters
    ----------
    target_key : str
        Key to look for in `data`.
    source : dict
        Input dictionary.
    default : any, optional
        Fallback value if key is missing or its value is None.

    Returns
    -------
    any
        `data[target_key]` if present and not None, otherwise `default`.
    """
    return (
        default
        if (target_key not in source) or (source[target_key] is None)
        else source[target_key]
    )


def _has_string_keys(
    value: dict[object, object],
) -> TypeGuard[SerializedContract]:
    return all(isinstance(key, str) for key in value)


def _is_object_mapping(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)


def _object_mapping(value: object) -> dict[object, object] | None:
    return value if _is_object_mapping(value) else None


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _require_mapping(value: object, *, field: str) -> SerializedContract:
    if not _is_object_mapping(value):
        raise TypeError(f"{field} must be a dictionary")
    if not _has_string_keys(value):
        raise TypeError(f"{field} must have string keys")
    return value


def _require_date(value: date | None, *, field: str) -> date:
    if value is None:
        raise TypeError(f"{field} must be a date")
    return value


def _require_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    return value


def _require_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be a boolean")
    return value


def _optional_bool(value: object, *, field: str) -> bool | None:
    if value is None:
        return None
    return _require_bool(value, field=field)


def _require_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    return float(value)


def _require_summary_inflation(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError("summary_arguments.inflation_rate must be a number")
    return float(value)


def _require_summary_discount(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError("summary_arguments.discount_rate must be a number")
    return float(value)


def _require_present(value: T | None, *, field: str) -> T:
    if value is None:
        raise TypeError(f"{field} must not be null")
    return value


def _optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string or null")
    return value


def _require_string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    return value


def _is_rcicp_input(value: object) -> TypeGuard[RCICPInput]:
    if not _is_object_mapping(value):
        return False
    if not _has_string_keys(value):
        return False
    if not value:
        return True

    items = tuple(value.values())
    if all(isinstance(item, dict) for item in items):
        required = {"bot_limit", "top_limit", "ctr_oil", "ctr_gas"}
        for raw_item in items:
            item = _object_mapping(raw_item)
            if item is None or not all(name in item for name in required):
                return False
            for name in required:
                item_value = item[name]
                if isinstance(item_value, bool) or not isinstance(item_value, Real):
                    return False
        return True
    if any(isinstance(item, dict) for item in items):
        return False

    required_columns = {
        "RC Bottom Limit",
        "RC Top Limit",
        "Pre Tax CTR Oil",
        "Pre Tax CTR Gas",
    }
    if not required_columns <= set(value):
        return False
    arrays: list[npt.NDArray[np.float64]] = []
    for name, raw_column in value.items():
        try:
            array = np.asarray(raw_column, dtype=np.float64)
        except (TypeError, ValueError):
            return False
        if name in required_columns and array.ndim != 1:
            return False
        if name in required_columns:
            arrays.append(array)
    return len({len(array) for array in arrays}) == 1


def _require_rcicp_input(value: object) -> RCICPInput:
    if not _is_rcicp_input(value):
        raise TypeError("costrecovery.condition_dict must be a supported mapping")
    return value


def _get_summary(contract: ProjectContract, arguments: SummaryArguments) -> Summary:
    return contract.get_summary(**arguments)


def _require_lifting_items(
    value: tuple[Lifting, ...] | None,
) -> tuple[Lifting, ...]:
    if value is None:
        raise TypeError("lifting must be initialized")
    return value


def _project_contracts(
    contract: BaseProject | CostRecovery | GrossSplit | Transition,
) -> tuple[ProjectContract, ...]:
    if isinstance(contract, Transition):
        return contract.contract1, contract.contract2
    return (contract,)


def _has_named_records(
    value: dict[object, object],
) -> TypeGuard[dict[str, dict[str, object]]]:
    for key, item in value.items():
        if not isinstance(key, str) or not _is_object_mapping(item):
            return False
        if not _has_string_keys(item):
            return False
    return True


def _require_records(value: object, *, field: str) -> dict[str, dict[str, object]]:
    if not _is_object_mapping(value):
        raise TypeError(f"{field} must be a mapping")
    if not _has_named_records(value):
        raise TypeError(f"{field} must contain named records with string keys")
    return value


def _is_parameters(value: object) -> TypeGuard[list[Parameter]]:
    if not _is_object_list(value):
        return False
    required = {"id", "fluid", "dist", "min", "base", "max", "stddev"}
    return all(
        _is_object_mapping(item)
        and _has_string_keys(item)
        and required <= set(item)
        and isinstance(item.get("id"), int)
        and not isinstance(item.get("id"), bool)
        and item.get("fluid") in (FluidType.OIL, FluidType.GAS, None)
        and isinstance(item.get("dist"), UncertaintyDistribution)
        and all(
            isinstance(item.get(name), (int, float))
            and not isinstance(item.get(name), bool)
            for name in ("min", "base", "max", "stddev")
        )
        for item in value
    )


def _require_parameters(value: object) -> list[Parameter]:
    if not _is_parameters(value):
        raise TypeError("uncertainty parameters must have numeric bounds")
    return [
        {
            **item,
            "min": float(item["min"]),
            "base": float(item["base"]),
            "max": float(item["max"]),
            "stddev": float(item["stddev"]),
        }
        for item in value
    ]


def get_setup_dict(
    data: SerializedContract,
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
    Parse project setup and optional cost components into internal objects.

    Parameters
    ----------
    data : dict
        Project configuration dictionary containing a required ``setup`` section
        and optional cost-related sections.

    Returns
    -------
    tuple
        Parsed setup values and converted cost objects, returned in a fixed order
        expected by the economic engine.
    """

    # Check whether "setup" exist in "data" and prepare it accordingly
    se = _require_mapping(
        _extract_from_dict(target_key="setup", source=data), field="setup"
    )

    # Parsing the contract setup into each corresponding variables
    start_date = _require_date(
        convert_str_to_date(str_object=se["start_date"]), field="setup.start_date"
    )
    end_date = _require_date(
        convert_str_to_date(str_object=se["end_date"]), field="setup.end_date"
    )
    approval_year = _require_int(
        convert_str_to_int(str_object=se["approval_year"]), field="setup.approval_year"
    )
    oil_onstream_date = convert_str_to_date(
        str_object=se.get("oil_onstream_date", None)
    )
    gas_onstream_date = convert_str_to_date(
        str_object=se.get("gas_onstream_date", None)
    )
    is_pod_1 = _optional_bool(se.get("is_pod_1", None), field="setup.is_pod_1")
    is_strict = _optional_bool(se.get("is_strict", None), field="setup.is_strict")
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


def get_summary_dict(data: SerializedContract) -> SummaryArguments:
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
    discount_rate_start_year_raw = sa.get("discount_rate_start_year", None)
    discount_rate_start_year = (
        None
        if discount_rate_start_year_raw is None
        else _require_int(
            discount_rate_start_year_raw,
            field="summary_arguments.discount_rate_start_year",
        )
    )
    inflation_rate = _require_summary_inflation(sa.get("inflation_rate", 0.0))
    discount_rate = _require_summary_discount(sa.get("discount_rate", 0.1))
    npv_mode = _require_present(
        convert_str_to_npvmode(
            str_object=sa.get("npv_mode", "Full Cycle Nominal Terms")
        ),
        field="summary_arguments.npv_mode",
    )
    discounting_mode = _require_present(
        convert_str_to_discountingmode(
            str_object=sa.get("discounting_mode", "End Year")
        ),
        field="summary_arguments.discounting_mode",
    )
    profitability_discounted = _require_bool(
        sa.get("profitability_discounted", False),
        field="summary_arguments.profitability_discounted",
    )

    summary_arguments_dict: SummaryArguments = {
        "discount_rate_start_year": discount_rate_start_year,
        "inflation_rate": inflation_rate,
        "discount_rate": discount_rate,
        "npv_mode": npv_mode,
        "discounting_mode": discounting_mode,
        "profitability_discounted": profitability_discounted,
    }

    return summary_arguments_dict


def build_baseproject_instance(data: SerializedContract) -> BaseProject:
    """
    Build a ``BaseProject`` instance from input configuration data.

    Parameters
    ----------
    data : dict
        Project configuration containing ``setup`` and optional cost/lifting sections.

    Returns
    -------
    BaseProject
        Initialized BaseProject with setup, lifting, and cost components attached.

    Notes
    -----
    - Internally relies on ``get_setup_dict()`` for parsing & conversion.
    - Missing optional sections → corresponding attributes set to ``None``.
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
    contract_kwargs: BaseProjectKwargs = {
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


def build_baseproject_arguments(data: SerializedContract) -> BaseProjectArguments:
    """
    Build BaseProject contract arguments from input data.

    Parameters
    ----------
    data : dict
        Project input dictionary containing ``contract_arguments``.

    Returns
    -------
    dict
        Parsed BaseProject arguments, including:
        - other revenues → sulfur | electricity | CO₂
        - VAT & inflation settings

    Notes
    -----
    - ``contract_arguments`` must exist in ``data``
    - Values are converted to engine-ready types
    """

    # Check whether "contract_arguments" exist in "data", and prepare it accordingly
    ca = _require_mapping(
        _extract_from_dict(target_key="contract_arguments", source=data),
        field="contract_arguments",
    )

    arguments: BaseProjectArguments = {
        # Other revenues
        "sulfur_revenue": _require_present(
            convert_str_to_otherrevenue(str_object=ca["sulfur_revenue"]),
            field="contract_arguments.sulfur_revenue",
        ),
        "electricity_revenue": _require_present(
            convert_str_to_otherrevenue(str_object=ca["electricity_revenue"]),
            field="contract_arguments.electricity_revenue",
        ),
        "co2_revenue": _require_present(
            convert_str_to_otherrevenue(str_object=ca["co2_revenue"]),
            field="contract_arguments.co2_revenue",
        ),
        # VAT and inflation
        "vat_rate": convert_list_to_array_float_or_array(data_input=ca["vat_rate"]),
        "inflation_rate": convert_list_to_array_float_or_array(
            data_input=ca["inflation_rate"]
        ),
        "inflation_rate_applied_to": convert_str_to_inflationappliedto(
            str_object=ca["inflation_rate_applied_to"]
        ),
    }
    return arguments


def get_baseproject(data: SerializedContract) -> Summary:
    """
    Build, execute, and summarize a Base Project contract evaluation.

    This function constructs a :class:`BaseProject` instance using the provided
    input data, executes its economic evaluation, and generates a summary of the
    results.

    Parameters
    ----------
    data : dict
        Dictionary containing setup information, contract parameters, and summary
        configuration.

    Returns
    -------
    dict
        Summary of the Base Project contract evaluation.
    """

    # Specify contract and contract arguments
    contract = build_baseproject_instance(data=data)
    contract_arguments_dict = build_baseproject_arguments(data=data)

    # Execute BaseProject instance
    contract.run(**contract_arguments_dict)

    # Fill summary arguments
    summary_arguments_dict = get_summary_dict(data=data)

    return _get_summary(contract, summary_arguments_dict)


def build_costrecovery_instance(data: SerializedContract) -> CostRecovery:
    """
    Build a CostRecovery contract instance from input data.

    Parameters
    ----------
    data : dict
        Project configuration dictionary containing:
        - ``setup`` (base project info)
        - ``costrecovery`` (CR-specific terms)

    Returns
    -------
    CostRecovery
        Initialized CostRecovery contract object.

    Notes
    -----
    - Base attributes are parsed via ``get_setup_dict()``
    - CR terms include FTP, split, IC, cap rate, and DMO settings
    - Inputs are converted to engine-ready types before instantiation
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

    # Check whether "costrecovery" exist in "data", and prepare it accordingly
    cr = _require_mapping(
        _extract_from_dict(target_key="costrecovery", source=data),
        field="costrecovery",
    )

    # Prepare contract attributes for CostRecovery
    contract_kwargs: CostRecoveryKwargs = {
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
        "tax_split_type": _require_present(
            convert_str_to_taxsplit(str_object=cr["tax_split_type"]),
            field="costrecovery.tax_split_type",
        ),
        "condition_dict": _require_rcicp_input(cr["condition_dict"]),
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
            convert_to_float(target=cr["oil_cr_cap_rate"]),
            field="costrecovery.oil_cr_cap_rate",
        ),
        "gas_cr_cap_rate": _require_float(
            convert_to_float(target=cr["gas_cr_cap_rate"]),
            field="costrecovery.gas_cr_cap_rate",
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


def build_costrecovery_arguments(data: SerializedContract) -> CostRecoveryArguments:
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

    # Check whether "contract_arguments" exist in "data", and prepare it accordingly
    ca = _require_mapping(
        _extract_from_dict(target_key="contract_arguments", source=data),
        field="contract_arguments",
    )

    arguments: CostRecoveryArguments = {
        # Other revenues
        "sulfur_revenue": _require_present(
            convert_str_to_otherrevenue(str_object=ca["sulfur_revenue"]),
            field="contract_arguments.sulfur_revenue",
        ),
        "electricity_revenue": _require_present(
            convert_str_to_otherrevenue(str_object=ca["electricity_revenue"]),
            field="contract_arguments.electricity_revenue",
        ),
        "co2_revenue": _require_present(
            convert_str_to_otherrevenue(str_object=ca["co2_revenue"]),
            field="contract_arguments.co2_revenue",
        ),
        # FTP
        "ftp_tax_regime": _require_present(
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
        "tax_regime": _require_present(
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
        "depr_method": _require_present(
            convert_str_to_depremethod(str_object=ca["depr_method"]),
            field="contract_arguments.depr_method",
        ),
        "decline_factor": _require_summary_discount(ca["decline_factor"]),
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
    return arguments


def get_costrecovery(data: SerializedContract) -> Summary:
    """
    Run cost-recovery calculation and return its summary.

    Builds the cost-recovery contract instance and arguments from `data`,
    executes the model, and extracts the final summary output.

    Parameters
    ----------
    data : dict
        Raw input dictionary containing contract setup, arguments,
        and summary configuration.

    Returns
    -------
    dict
        Cost-recovery summary results.
    """

    # Specify contract and contract arguments
    contract = build_costrecovery_instance(data=data)
    contract_arguments_dict = build_costrecovery_arguments(data=data)

    # Execute CostRecovery instance
    contract.run(**contract_arguments_dict)

    # Fill summary arguments
    summary_arguments_dict = get_summary_dict(data=data)

    return _get_summary(contract, summary_arguments_dict)


def build_grosssplit_instance(data: SerializedContract) -> GrossSplit:
    """
    Build and initialize a GrossSplit contract instance.

    Extracts setup, lifting, cost, and Gross Split–specific parameters
    from `data`, prepares contract arguments, and returns a ready-to-run
    `GrossSplit` object.

    Parameters
    ----------
    data : dict
        Raw input dictionary containing project setup and gross split
        contract configuration.

    Returns
    -------
    GrossSplit
        Initialized GrossSplit contract instance.
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

    # Check whether "grosssplit" exist in "data", and prepare it accordingly
    gs = _require_mapping(
        _extract_from_dict(target_key="grosssplit", source=data), field="grosssplit"
    )

    # Prepare contract attributes for GrossSplit
    contract_kwargs: GrossSplitKwargs = {
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
        "field_status": _optional_string(
            _extract_from_dict(target_key="field_status", source=gs),
            field="grosssplit.field_status",
        ),
        "field_loc": _require_string(
            _extract_from_dict(target_key="field_loc", source=gs),
            field="grosssplit.field_loc",
        ),
        "res_depth": _optional_string(
            _extract_from_dict(target_key="res_depth", source=gs),
            field="grosssplit.res_depth",
        ),
        "infra_avail": _require_string(
            _extract_from_dict(target_key="infra_avail", source=gs),
            field="grosssplit.infra_avail",
        ),
        "res_type": _optional_string(
            _extract_from_dict(target_key="res_type", source=gs),
            field="grosssplit.res_type",
        ),
        "api_oil": _optional_string(
            _extract_from_dict(target_key="api_oil", source=gs),
            field="grosssplit.api_oil",
        ),
        "domestic_use": _optional_string(
            _extract_from_dict(target_key="domestic_use", source=gs),
            field="grosssplit.domestic_use",
        ),
        "prod_stage": _optional_string(
            _extract_from_dict(target_key="prod_stage", source=gs),
            field="grosssplit.prod_stage",
        ),
        "co2_content": _optional_string(
            _extract_from_dict(target_key="co2_content", source=gs),
            field="grosssplit.co2_content",
        ),
        "h2s_content": _optional_string(
            _extract_from_dict(target_key="h2s_content", source=gs),
            field="grosssplit.h2s_content",
        ),
        "field_reserves_2024": _optional_string(
            _extract_from_dict(target_key="field_reserves_2024", source=gs),
            field="grosssplit.field_reserves_2024",
        ),
        "infra_avail_2024": _require_string(
            _extract_from_dict(target_key="infra_avail_2024", source=gs),
            field="grosssplit.infra_avail_2024",
        ),
        "field_loc_2024": _require_string(
            _extract_from_dict(target_key="field_loc_2024", source=gs),
            field="grosssplit.field_loc_2024",
        ),
        "split_ministry_disc": _require_float(
            convert_to_float(target=gs["split_ministry_disc"]),
            field="grosssplit.split_ministry_disc",
        ),
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


def build_grosssplit_arguments(data: SerializedContract) -> GrossSplitArguments:
    """
    Build Gross Split contract argument dictionary.

    Extracts `contract_arguments` from `data`, applies defaults where needed,
    performs type/enum conversions, and prepares keyword arguments for
    Gross Split contract execution.

    Parameters
    ----------
    data : dict
        Raw input dictionary containing contract arguments.

    Returns
    -------
    dict
        Parsed and converted Gross Split contract arguments.
    """

    # Check whether "contract_arguments" exist in "data", and prepare it accordingly
    ca = _require_mapping(
        _extract_from_dict(target_key="contract_arguments", source=data),
        field="contract_arguments",
    )

    arguments: GrossSplitArguments = {
        # Other revenues
        "sulfur_revenue": _require_present(
            convert_str_to_otherrevenue(str_object=ca["sulfur_revenue"]),
            field="contract_arguments.sulfur_revenue",
        ),
        "electricity_revenue": _require_present(
            convert_str_to_otherrevenue(str_object=ca["electricity_revenue"]),
            field="contract_arguments.electricity_revenue",
        ),
        "co2_revenue": _require_present(
            convert_str_to_otherrevenue(str_object=ca["co2_revenue"]),
            field="contract_arguments.co2_revenue",
        ),
        # VAT and inflation
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
        "depr_method": _require_present(
            convert_str_to_depremethod(str_object=ca["depr_method"]),
            field="contract_arguments.depr_method",
        ),
        "decline_factor": _require_summary_discount(
            _extract_from_dict(target_key="decline_factor", source=ca, default=2)
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
        "tax_regime": _require_present(
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
        "regime": _require_present(
            convert_grosssplitregime_to_enum(target=ca["regime"]),
            field="contract_arguments.regime",
        ),
        "reservoir_type_permen_2024": _require_present(
            converter_reservoir_type_permen_2024(
                target_str=ca["reservoir_type_permen_2024"]
            ),
            field="contract_arguments.reservoir_type_permen_2024",
        ),
    }
    return arguments


def get_grosssplit(data: SerializedContract) -> Summary:
    """
    Build, execute, and summarize a Gross Split contract evaluation.

    This function constructs a :class:`GrossSplit` instance using the provided
    input data, executes its economic evaluation, and generates a summary of the
    results.

    Parameters
    ----------
    data : dict
        Dictionary containing setup information, contract parameters, and summary
        configuration.

    Returns
    -------
    dict
        Summary of the Gross Split contract evaluation.
    """

    # Specify contract and contract arguments
    contract = build_grosssplit_instance(data=data)
    contract_arguments_dict = build_grosssplit_arguments(data=data)

    # Execute GrossSplit instance
    contract.run(**contract_arguments_dict)

    # Fill summary arguments
    summary_arguments_dict = get_summary_dict(data=data)

    return _get_summary(contract, summary_arguments_dict)


def get_transition(data: SerializedContract) -> Summary:
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
    """

    def _build_contract(
        data_part: SerializedContract,
    ) -> tuple[CostRecovery | GrossSplit, dict[str, object]]:
        if (
            data_part.get("costrecovery") is not None
            and data_part.get("grosssplit") is None
        ):
            contract_part = build_costrecovery_instance(data_part)
            costrecovery_arguments = build_costrecovery_arguments(data_part)
            contract_part.run(**costrecovery_arguments)
            arguments_part: dict[str, object] = {
                key: value for key, value in costrecovery_arguments.items()
            }
        elif (
            data_part.get("grosssplit") is not None
            and data_part.get("costrecovery") is None
        ):
            contract_part = build_grosssplit_instance(data_part)
            grosssplit_arguments = build_grosssplit_arguments(data_part)
            contract_part.run(**grosssplit_arguments)
            arguments_part = {key: value for key, value in grosssplit_arguments.items()}
        else:
            raise MonteCarloException("Contract type is not recognized")
        return contract_part, arguments_part

    contract_1, contract_arguments_1 = _build_contract(
        _require_mapping(data["contract_1"], field="contract_1")
    )
    contract_2, contract_arguments_2 = _build_contract(
        _require_mapping(data["contract_2"], field="contract_2")
    )

    # generating the transition contract object
    contract = Transition(
        contract1=contract_1,
        contract2=contract_2,
        argument_contract1=contract_arguments_1,
        argument_contract2=contract_arguments_2,
    )

    # Generating the transition contract arguments
    contract_arguments_dict = _require_mapping(
        data["contract_arguments"], field="contract_arguments"
    )

    # Running the transition contract
    contract.run(
        unrec_portion=_require_float(
            contract_arguments_dict.get("unrec_portion", 0.0),
            field="contract_arguments.unrec_portion",
        )
    )

    # Filling the summary arguments
    summary_arguments_dict = get_summary_dict(data=data)
    return get_summary(
        contract=contract,
        reference_year=None,
        inflation_rate=summary_arguments_dict["inflation_rate"],
        discount_rate=summary_arguments_dict["discount_rate"],
        npv_mode=summary_arguments_dict["npv_mode"],
        discounting_mode=summary_arguments_dict["discounting_mode"],
        profitability_discounted=summary_arguments_dict["profitability_discounted"],
    )


def get_multipliers_montecarlo(
    run_number: int,
    distribution: str,
    min_value: float,
    mean_value: float,
    max_value: float,
    std_dev: float,
) -> FloatArray:
    """
    Generate an array of multipliers for Monte Carlo simulation based on
    the specified probability distribution.

    All distributions are generated in *ratio space* (value / mean_value),
    so the resulting multipliers are centered around 1.0.

    Parameters
    ----------
    run_number : int
        Number of Monte Carlo runs.
    distribution : str
        Distribution type (case-insensitive):
        {"uniform", "triangular", "normal", "lognormal"}.
    min_value : float
        Minimum value of the distribution.
    mean_value : float
        Mean (central) value of the distribution.
    max_value : float
        Maximum value of the distribution.
    std_dev : float
        Standard deviation (used for Normal and LogNormal distributions).

    Returns
    -------
    np.ndarray
        Array of Monte Carlo multipliers.

    Raises
    ------
    ValueError
        If input values are invalid.
    MonteCarloException
        If the distribution type is unsupported.

    TL;DR:
    Generate mean-normalized Monte Carlo multipliers using Uniform, Triangular,
    Normal (truncated), or LogNormal (truncated) distributions.
    """

    # Return an array of ones if min = mean = max
    if min_value == mean_value == max_value:
        return np.ones(run_number, dtype=np.float64)

    violation = uncertainty_parameters_violation(
        minimum=min_value,
        mean=mean_value,
        maximum=max_value,
        std_dev=std_dev,
        distribution=distribution,
    )
    if violation == "zero_mean":
        raise MonteCarloInputError(
            "Cannot have zero mean value", code="UNCERTAINTY_ZERO_MEAN"
        )
    if violation == "range":
        raise MonteCarloInputError(
            "Paramater max_value must be greater than min_value",
            code="UNCERTAINTY_RANGE_INVALID",
        )
    if violation == "distribution":
        raise MonteCarloInputError(
            "The type of distribution is unrecognized. "
            + "Please select type of distribution between: "
            + "Uniform, Triangular, Normal, and LogNormal.",
            code="UNCERTAINTY_DISTRIBUTION_UNRECOGNIZED",
        )
    if violation == "zero_std":
        raise MonteCarloInputError(
            "Cannot have zero standard deviation", code="UNCERTAINTY_ZERO_STD"
        )
    if violation == "lognormal_nonpositive":
        raise MonteCarloInputError(
            "All input must be positive for a log-normal distribution",
            code="UNCERTAINTY_LOGNORMAL_NONPOSITIVE",
        )

    # Normalize distribution name
    distribution = distribution.lower()

    # Uniform distribution
    if distribution == "uniform":
        # Normalize parameters
        min_ratio, max_ratio = [float(v / mean_value) for v in (min_value, max_value)]

        # Draw samples, then assign them as `multipliers`
        multipliers = uniform.rvs(
            loc=min_ratio,
            scale=max_ratio - min_ratio,
            size=run_number,
        )

    # Triangular distribution
    elif distribution == "triangular":
        # Normalize parameters
        (min_ratio, mean_ratio, max_ratio) = [
            float(v / mean_value) for v in (min_value, mean_value, max_value)
        ]

        # Determine mode (central point)
        c = (mean_ratio - min_ratio) / (max_ratio - min_ratio)

        # Draw samples, then assign them as `multipliers`
        multipliers = triang.rvs(
            c=c,
            loc=min_ratio,
            scale=max_ratio - min_ratio,
            size=run_number,
        )

    # Normal distribution
    elif distribution == "normal":
        # Validate input: cannot have zero standard deviation
        if std_dev == 0:
            raise MonteCarloInputError(
                "Cannot have zero standard deviation", code="UNCERTAINTY_ZERO_STD"
            )

        # Normalize parameters
        (min_ratio, mean_ratio, max_ratio, std_ratio) = [
            float(v / mean_value) for v in (min_value, mean_value, max_value, std_dev)
        ]

        # Determine z-values
        z_min = (min_ratio - mean_ratio) / std_ratio
        z_max = (max_ratio - mean_ratio) / std_ratio

        # Draw samples, then assign them as `multipliers`
        multipliers_init = truncnorm.rvs(
            a=z_min,
            b=z_max,
            loc=0,
            scale=1,
            size=run_number,
        )

        multipliers = (multipliers_init * std_ratio) + mean_ratio

    # Log normal distribution
    elif distribution == "lognormal":
        # Validate input: cannot have zero standard deviation
        if std_dev == 0:
            raise MonteCarloInputError(
                "Cannot have zero standard deviation", code="UNCERTAINTY_ZERO_STD"
            )

        # Validate input
        if min_value <= 0 or mean_value <= 0 or max_value <= 0:
            raise MonteCarloInputError(
                "All input must be positive for a log-normal distribution",
                code="UNCERTAINTY_LOGNORMAL_NONPOSITIVE",
            )

        # Normalize parameters
        (min_ratio, mean_ratio, max_ratio, std_ratio) = [
            float(v / mean_value) for v in (min_value, mean_value, max_value, std_dev)
        ]

        # Convert to log-space
        log_min = math.log(min_ratio)
        log_mean = math.log(mean_ratio)
        log_max = math.log(max_ratio)
        log_std = std_ratio / mean_ratio

        # Determine z-values in log-space
        z_min = (log_min - log_mean) / log_std
        z_max = (log_max - log_mean) / log_std

        if z_min > z_max:
            raise MonteCarloInputError(
                f"Invalid truncation bounds: z_min = {z_min}, z_max = {z_max}",
                code="UNCERTAINTY_TRUNCATION_BOUNDS",
            )

        # Draw truncated normal samples in log-space
        log_samples = truncnorm.rvs(
            a=z_min,
            b=z_max,
            loc=0,
            scale=1,
            size=run_number,
        )

        # Transform back
        multipliers = np.exp((log_samples * log_std) + log_mean)

    else:
        raise MonteCarloInputError(
            "The type of distribution is unrecognized. "
            + "Please select type of distribution between: "
            + "Uniform, Triangular, Normal, and LogNormal.",
            code="UNCERTAINTY_DISTRIBUTION_UNRECOGNIZED",
        )

    return np.asarray(multipliers, dtype=np.float64)


def min_mean_max_retriever(
    contract: BaseProject | CostRecovery | GrossSplit | Transition,
    verbose: bool = False,
) -> dict[str, float]:
    """
    Retrieve minimum, mean, and maximum values for key economic parameters.

    This function computes summary statistics (min, mean, max) for CAPEX,
    OPEX, lifting rates, and oil/gas prices based on data stored in the
    contract object.

    Parameters
    ----------
    contract : BaseProject | CostRecovery | GrossSplit | Transition
        Contract instance containing cost, lifting, and price data.
    verbose : bool, default False
        If True, print computed statistics to stdout.

    Returns
    -------
    dict
        Dictionary containing min, mean, and max values for each parameter.
    """

    def _get_statistics(arrays_list: list[FloatArray]) -> tuple[float, float, float]:
        min_value = min([np.min(arr) for arr in arrays_list])
        mean_value = np.mean([np.mean(arr) for arr in arrays_list])
        max_value = max([np.max(arr) for arr in arrays_list])
        return float(min_value), float(mean_value), float(max_value)

    # Compute min/mean/max for a single array (e.g., prices)
    def _get_statistics_price(array: FloatArray) -> tuple[float, float, float]:
        return _get_statistics([array])

    projects = _project_contracts(contract)

    lifting_by_project = [
        _require_lifting_items(project.lifting) for project in projects
    ]

    def _price_profile(fluid_type: FluidType) -> FloatArray:
        project_profiles: list[FloatArray] = []
        for project, lifting in zip(projects, lifting_by_project):
            prices = [
                np.asarray(item.price, dtype=np.float64)
                for item in lifting
                if item.fluid_type == fluid_type
            ]
            project_profiles.append(
                np.concatenate(prices)
                if prices
                else np.zeros(project.project_duration, dtype=np.float64)
            )
        return np.concatenate(project_profiles)

    # Calculate statistics (min, mean, max) for CAPEX.
    # --- CAPEX: capital + intangible costs ---
    (min_capex, mean_capex, max_capex) = _get_statistics(
        arrays_list=[
            array
            for project in projects
            for array in (
                require_float_array(project.capital_cost_total.cost, "capital cost"),
                require_float_array(
                    project.intangible_cost_total.cost, "intangible cost"
                ),
            )
        ]
    )

    # Calculate statistics (min, mean, max) for OPEX.
    # --- OPEX: operating, ASR, LBT, and cost of sales ---
    (min_opex, mean_opex, max_opex) = _get_statistics(
        arrays_list=[
            array
            for project in projects
            for array in (
                require_float_array(project.opex_total.cost, "opex cost"),
                require_float_array(project.asr_cost_total.cost, "asr cost"),
                require_float_array(project.lbt_cost_total.cost, "lbt cost"),
                require_float_array(project.cost_of_sales_total.cost, "cost of sales"),
            )
        ]
    )

    # Calculate statistics (min, mean, max) for LIFTING.
    # --- LIFTING: every producer type ---
    (min_lifting, mean_lifting, max_lifting) = _get_statistics(
        arrays_list=[
            item.lifting_rate for lifting in lifting_by_project for item in lifting
        ]
    )

    # Calculate statistics (min, mean, max) for OIL PRICE
    (min_oil_price, mean_oil_price, max_oil_price) = _get_statistics_price(
        array=_price_profile(FluidType.OIL)
    )

    # Calculate statistics (min, mean, max) for GAS PRICE
    (min_gas_price, mean_gas_price, max_gas_price) = _get_statistics_price(
        array=_price_profile(FluidType.GAS)
    )

    # Collect all statistics into a dictionary
    results = {
        "min_capex": min_capex,
        "mean_capex": mean_capex,
        "max_capex": max_capex,
        "min_opex": min_opex,
        "mean_opex": mean_opex,
        "max_opex": max_opex,
        "min_oil_price": min_oil_price,
        "mean_oil_price": mean_oil_price,
        "max_oil_price": max_oil_price,
        "min_gas_price": min_gas_price,
        "mean_gas_price": mean_gas_price,
        "max_gas_price": max_gas_price,
        "min_lifting": min_lifting,
        "mean_lifting": mean_lifting,
        "max_lifting": max_lifting,
    }

    # Optional debug output
    if verbose is True:
        print("Parameter used:")
        for key, value in results.items():
            print(key, ": ", value)
        print("")

    return results


class ProcessMonte:
    """
    Prepare and execute Monte Carlo simulation for uncertainty assessment.
    """

    type: int
    numSim: int
    baseContract: SerializedContract
    parameter: list[Parameter]
    multipliers: FloatArray

    target: list[str] = ["npv", "irr", "pi", "pot", "gov_take", "ctr_net_share"]

    def __init__(
        self,
        contract_type: int,
        contract: SerializedContract,
        numSim: int,
        params: list[Parameter],
    ) -> None:

        self.type = contract_type
        self.numSim = numSim
        self.baseContract = contract
        self.parameter = params

        # Prepare multipliers
        self.multipliers = np.ones([self.numSim, len(self.parameter)], dtype=np.float64)

        for i, param in enumerate(self.parameter):
            self.multipliers[:, i] = get_multipliers_montecarlo(
                run_number=self.numSim,
                distribution=param["dist"].value,
                min_value=param["min"],
                mean_value=param["base"],
                max_value=param["max"],
                std_dev=param["stddev"],
            )

    def Adjust_Data(self, multipliers: Iterable[float]) -> SerializedContract:
        """
        Apply parameter-based multipliers to adjust contract economic data.

        Creates a deep copy of the base contract and scales relevant fields
        (price, cost, lifting rate) according to the provided multipliers.

        Parameters
        ----------
        multipliers : np.ndarray
            Array of scaling factors for target parameters
            (Oil Price, Gas Price, OPEX, CAPEX, Lifting).

        Returns
        -------
        dict
            Deep-copied contract with adjusted economic attributes.

        Notes
        -----
        The adjustment is performed using an internal helper function and
        does not modify the original base contract.
        """

        contract_adjusted: SerializedContract = copy.deepcopy(self.baseContract)

        def _adjust_partial_data(
            contract_: SerializedContract,
            target_param: str,
            key: str,
            multiplier: float,
            datakeys: list[str] | None = None,
        ) -> None:
            """
            Helper function to apply partial data adjustment to selected contract
            attributes.

            Scales numeric fields in the specified section (e.g., 'lifting', 'opex',
            'capex') based on the target parameter and multiplier.

            Parameters
            ----------
            contract_ : dict
                Contract data structure to modify.
            target_param : str
                Parameter to adjust (e.g., "Oil Price", "Gas Price", "Lifting").
            key : str
                Section name containing target data.
            multiplier : float
                Scaling factor applied to numeric fields.
            datakeys : list of str, optional
                Field names to adjust for non-lifting sections.

            Notes
            -----
            Lifting adjustments depend on `target_param` and `fluid_type`;
            other sections use `datakeys`.
            """

            if datakeys is None:
                datakeys = []

            for item in _require_records(contract_[key], field=key).values():
                # Specify target keys for lifting-related target
                if key == "lifting":
                    if target_param == "Oil Price" and item["fluid_type"] == "Oil":
                        target_keys = ["price"]
                    elif target_param == "Gas Price" and item["fluid_type"] == "Gas":
                        target_keys = ["price"]
                    elif target_param == "Lifting":
                        target_keys = ["lifting_rate", "prod_rate"]
                    else:
                        continue

                # Specify target keys for non-lifting targets
                else:
                    target_keys = datakeys

                # Adjust target attributes by multiplication with the prescribed multipliers
                for k in target_keys:
                    if k in item:
                        item[k] = (np.array(item[k]) * multiplier).tolist()

        # Specify attribute `contract_` based on `contract_type`
        contract_: SerializedContract = (
            contract_adjusted
            if self.type < 3
            else _require_mapping(
                contract_adjusted[f"contract_{2}"], field=f"contract_{2}"
            )
        )

        # Contract adjustments per single run simulation (i.e., per row)
        for val, sampled_multiplier in zip(self.parameter, multipliers, strict=True):
            multiplier = float(sampled_multiplier)
            # Target parameter: Oil Price
            if val["id"] == 0:
                _adjust_partial_data(
                    contract_=contract_,
                    target_param="Oil Price",
                    key="lifting",
                    multiplier=multiplier,
                )

            # Target parameter: Gas Price
            elif val["id"] == 1:
                _adjust_partial_data(
                    contract_=contract_,
                    target_param="Gas Price",
                    key="lifting",
                    multiplier=multiplier,
                )

            # Target parameter: OPEX
            elif val["id"] == 2:
                # Adjust class OPEX
                _adjust_partial_data(
                    contract_=contract_,
                    target_param="OPEX",
                    key="opex",
                    multiplier=multiplier,
                    datakeys=["fixed_cost", "cost_per_volume", "cost"],
                )

                # Adjust class ASR
                _adjust_partial_data(
                    contract_=contract_,
                    target_param="ASR",
                    key="asr",
                    multiplier=multiplier,
                    datakeys=["cost"],
                )

                # Adjust class LBT
                _adjust_partial_data(
                    contract_=contract_,
                    target_param="LBT",
                    key="lbt",
                    multiplier=multiplier,
                    datakeys=["cost"],
                )

                # Adjust class CostOfSales
                _adjust_partial_data(
                    contract_=contract_,
                    target_param="COS",
                    key="cost_of_sales",
                    multiplier=multiplier,
                    datakeys=["cost"],
                )

            # Target parameter: CAPEX
            elif val["id"] == 3:
                # Adjust class CapitalCost
                _adjust_partial_data(
                    contract_=contract_,
                    target_param="CAPEX",
                    key="capital",
                    multiplier=multiplier,
                    datakeys=["cost"],
                )

                # Adjust class Intangible
                _adjust_partial_data(
                    contract_=contract_,
                    target_param="CAPEX",
                    key="intangible",
                    multiplier=multiplier,
                    datakeys=["cost"],
                )

            # Target parameter: Lifting
            elif val["id"] == 4:
                _adjust_partial_data(
                    contract_=contract_,
                    target_param="Lifting",
                    key="lifting",
                    multiplier=multiplier,
                )

        return contract_

    def calcContract(self, n: int) -> MonteResult:
        """
        Execute a Monte Carlo simulation for the selected contract type and
        return key economic metrics.

        Parameters
        ----------
        n : int
            Simulation index used to select the corresponding multiplier set
            for data adjustment.

        Returns
        -------
        dict
            A dictionary with keys:
            - 'n': int, the simulation index.
            - 'output': tuple of float, containing
              (ctr_npv, ctr_irr, ctr_pi, ctr_pot, gov_take, ctr_net_share).

        Notes
        -----
        The function automatically selects and runs the appropriate contract evaluation
        (Cost Recovery, Gross Split, Transition, or Base Project) based on ``self.type``.
        In case of an error, zeros are returned for all output metrics.
        """
        try:
            # print(f"Monte Progress: {n}", flush=True)
            # time.sleep(100)

            # Specify adjusted data by calling the "Adjust_Data()" method
            dataAdj = self.Adjust_Data(self.multipliers[n, :])

            # Execute the corresponding contract and return the result in terms of summary
            mapping_summary = {
                1: get_costrecovery,
                2: get_grosssplit,
                3: get_transition,
            }

            csummary = mapping_summary.get(self.type, get_baseproject)(data=dataAdj)

            del dataAdj
            return {
                "n": n,
                "output": (
                    _require_float(csummary["ctr_npv"], field="ctr_npv"),
                    _require_float(csummary["ctr_irr"], field="ctr_irr"),
                    _require_float(csummary["ctr_pi"], field="ctr_pi"),
                    _require_float(csummary["ctr_pot"], field="ctr_pot"),
                    _require_float(csummary["gov_take"], field="gov_take"),
                    _require_float(csummary["ctr_net_share"], field="ctr_net_share"),
                ),
            }

        except Exception as err:
            print(f"Error: {err}")
            return {
                "n": n,
                "output": (
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ),
            }

    def get_outcomes(self, results: FloatArray) -> dict[str, object]:
        """
        Summarize Monte Carlo results into probabilistic outcomes.

        Sorts simulation outputs, assigns cumulative probabilities, and
        computes P10 / P50 / P90 per parameter.

        Parameters
        ----------
        results : np.ndarray
            Simulation results, shape (numSim, n_params).

        Returns
        -------
        dict
            Outcome summary with:
            - params : list[str]
            - results : list[list[float]]  # prob + values
            - P10, P50, P90 : list[float]

        Notes
        -----
        - Sorting is done per parameter.
        - Percentiles use `method="higher"`.

        TL;DR:
        Turn Monte Carlo outputs into probability tables and P10/P50/P90.
        """

        row_number = self.numSim

        # Sort results in ascending manner
        results_sorted = np.take_along_axis(
            arr=results, indices=np.argsort(results, axis=0), axis=0
        )

        # Specify probability
        prob = np.arange(1, row_number + 1, dtype=float) / row_number

        # Arrange the results
        results_arranged = np.concatenate((prob[:, np.newaxis], results_sorted), axis=1)

        # Calculate P10, P50, P90
        percentiles = np.asarray(
            np.percentile(
                a=results_arranged,
                q=[10, 50, 90],
                method="higher",
                axis=0,
            ),
            dtype=np.float64,
        )

        # Determine indices of data
        indices = np.linspace(0, row_number, 101)[0:-1].astype(int)
        # indices[0] = 1
        if indices[-1] != row_number - 1:
            indices = np.append(indices, int(row_number - 1))

        # Identify parameter names
        target_indices = np.array([val["id"] for _, val in enumerate(self.parameter)])
        param_names = np.array(["oil_price", "gas_price", "opex", "capex", "lifting"])[
            target_indices
        ]

        # Final outcomes
        outcomes = {
            "params": param_names.tolist(),
            "results": results_arranged[indices, :].tolist(),
            "P10": percentiles[0, :].tolist(),
            "P50": percentiles[1, :].tolist(),
            "P90": percentiles[2, :].tolist(),
        }

        return outcomes

    def calculate_single_core(self) -> dict[str, object]:
        """
        Run Monte Carlo simulation sequentially on a single CPU core.

        Executes multiple contract evaluations in series, storing key
        economic indicators and parameter multipliers for each iteration.

        Returns
        -------
        dict
            Dictionary of summarized outcomes from all simulations, including:
            - **params** : list of str
                Names of economic parameters.
            - **results** : list of list
                Simulation results with cumulative probabilities.
            - **P10**, **P50**, **P90** : list of float
                Key percentiles summarizing uncertainty in results.
        """

        row_number = self.numSim

        # Designate a container to store Monte Carlo simulation result
        results: FloatArray = np.zeros(
            [row_number, len(self.target) + len(self.parameter)], dtype=float
        )

        for n in range(row_number):
            # Execute Monte Carlo simulation by calling function calcContract()
            out = self.calcContract(n)
            rnum = out["n"]
            output = out["output"]

            # Columns 0 until 5:
            # Fill with NPV, IRR, PI, POT, gov_take, and ctr_net_share, respectively
            results[rnum, 0 : len(self.target)] = output

            # Column 6 until end:
            # Fill with the multipliers associated with:
            # oil price, gas price, opex, capex, and lifting, accordingly
            results[rnum, len(self.target) :] = [
                self.multipliers[rnum, idx] * items["base"]
                for idx, items in enumerate(self.parameter)
            ]

        # Get the outcomes of Monte Carlo simulation
        return self.get_outcomes(results=results)

    def calculate_multi_cores(self) -> dict[str, object]:
        """
        Run Monte Carlo simulation in parallel using multiple CPU cores.

        Distributes contract evaluations across available processors and
        aggregates results for percentile-based outcome analysis.

        Returns
        -------
        dict
            Dictionary of summarized outcomes including:
            - **params** : list of str
                Economic parameter names.
            - **results** : list of list
                Combined simulation results with cumulative probabilities.
            - **P10**, **P50**, **P90** : list of float
                Key percentiles representing uncertainty ranges.
        """

        mp.freeze_support()

        row_number = self.numSim

        # Specify the number of active CPU for parallel computation
        cpu_count = os.cpu_count()
        if cpu_count is None:
            raise RuntimeError("CPU count is unavailable")
        n_processes = max(1, int(cpu_count - 2))
        # n_processes = max(1, int(os.cpu_count() / 2))

        # Designate a container to store Monte Carlo simulation results
        results: FloatArray = np.zeros(
            [row_number, len(self.target) + len(self.parameter)], dtype=float
        )

        # Specify a helper function
        def worker(
            n: int,
        ) -> tuple[int, tuple[float, float, float, float, float, float], list[float]]:
            """
            Run a single Monte Carlo iteration.

            Parameters
            ----------
            n : int
                Simulation index.

            Returns
            -------
            tuple
                (rnum, output, multipliers) containing the run index,
                economic results, and parameter multipliers.
            """
            out = self.calcContract(n)
            rnum = out["n"]
            output = out["output"]
            multipliers = [
                self.multipliers[rnum, idx] * items["base"]
                for idx, items in enumerate(self.parameter)
            ]

            return rnum, output, multipliers

        # Instantiate pathos pool
        pl = Pool(nodes=n_processes)

        # Map each Monte Carlo run in parallel
        results_list = pl.map(worker, range(row_number))

        # Close the pool instance
        pl.close()
        pl.join()

        for rnum, output, mult in results_list:
            # Columns 0 until 5:
            # Fill with NPV, IRR, PI, POT, gov_take, and ctr_net_share, respectively
            results[rnum, 0 : len(self.target)] = output

            # Column 6 until end:
            # Fill with the multipliers associated with:
            # oil price, gas price, opex, capex, and lifting, accordingly
            results[rnum, len(self.target) :] = mult

        return self.get_outcomes(results=results)


def uncertainty_psc(
    contract: BaseProject | CostRecovery | GrossSplit | Transition,
    contract_arguments: dict[object, object],
    summary_arguments: dict[str, object],
    run_number: int = 10,
    min_oil_price: float | None = None,
    mean_oil_price: float | None = None,
    max_oil_price: float | None = None,
    min_gas_price: float | None = None,
    mean_gas_price: float | None = None,
    max_gas_price: float | None = None,
    min_opex: float | None = None,
    mean_opex: float | None = None,
    max_opex: float | None = None,
    min_capex: float | None = None,
    mean_capex: float | None = None,
    max_capex: float | None = None,
    min_lifting: float | None = None,
    mean_lifting: float | None = None,
    max_lifting: float | None = None,
    oil_price_stddev: float = 1.25,
    gas_price_stddev: float = 1.25,
    opex_stddev: float = 1.25,
    capex_stddev: float = 1.25,
    lifting_stddev: float = 1.25,
    oil_price_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL,
    gas_price_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL,
    opex_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL,
    capex_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL,
    lifting_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL,
    verbose: bool = False,
) -> dict[str, object]:
    """
    Perform Monte Carlo uncertainty analysis for a PSC economic model.

    Generates probabilistic outcomes (P10, P50, P90) by varying key
    parameters—oil price, gas price, OPEX, CAPEX, and lifting—according
    to defined distributions and standard deviations.

    Returns
    -------
    dict
        Summary of probabilistic outcomes including P10, P50, P90, and
        associated simulation results.

    Notes
    -----
    The function builds and executes a PSC contract (BaseProject,
    CostRecovery, GrossSplit, or Transition) across multiple simulation
    runs. Each run applies random multipliers to key parameters,
    evaluates the economic results, and aggregates them to estimate
    uncertainty ranges.

    Automatically switches to parallel mode when `run_number > 400`.
    """

    # Translating the contract type before parsing into ProcessMonte class
    if isinstance(contract, CostRecovery):
        contract_type = 1

    elif isinstance(contract, GrossSplit):
        contract_type = 2

    elif isinstance(contract, Transition):
        contract_type = 3

    else:
        contract_type = 0

    # Extract original statistics (min, mean, max) of several input parameters
    min_max_mean_original = min_mean_max_retriever(contract=contract, verbose=verbose)

    # Organize the statistics of several input parameters (from user's input)
    min_max_mean_std: dict[str, float | None] = {
        "min_oil_price": min_oil_price,
        "mean_oil_price": mean_oil_price,
        "max_oil_price": max_oil_price,
        "min_gas_price": min_gas_price,
        "mean_gas_price": mean_gas_price,
        "max_gas_price": max_gas_price,
        "min_opex": min_opex,
        "mean_opex": mean_opex,
        "max_opex": max_opex,
        "min_capex": min_capex,
        "mean_capex": mean_capex,
        "max_capex": max_capex,
        "min_lifting": min_lifting,
        "mean_lifting": mean_lifting,
        "max_lifting": max_lifting,
    }

    # When user does not provide statistics input for the required parameters, then
    # fill them with the original values taken from variable "min_max_mean_original"
    for element in min_max_mean_std.keys():
        if min_max_mean_std[element] is None:
            min_max_mean_std[element] = min_max_mean_original[element]

    # Create a list of input arguments' configuration for each uncertainty parameters.
    # --- Uncertainty parameter:
    # --- (1) OIL PRICE, (2) GAS PRICE, (3) OPEX, (4) CAPEX, (5) Lifting
    parameter_initial = [
        {
            "id": 0,
            "fluid": FluidType.OIL,
            "dist": oil_price_distribution,
            "min": min_max_mean_std["min_oil_price"],
            "base": min_max_mean_std["mean_oil_price"],
            "max": min_max_mean_std["max_oil_price"],
            "stddev": oil_price_stddev,
        },
        {
            "id": 1,
            "fluid": FluidType.GAS,
            "dist": gas_price_distribution,
            "min": min_max_mean_std["min_gas_price"],
            "base": min_max_mean_std["mean_gas_price"],
            "max": min_max_mean_std["max_gas_price"],
            "stddev": gas_price_stddev,
        },
        {
            "id": 2,
            "fluid": None,
            "dist": opex_distribution,
            "min": min_max_mean_std["min_opex"],
            "base": min_max_mean_std["mean_opex"],
            "max": min_max_mean_std["max_opex"],
            "stddev": opex_stddev,
        },
        {
            "id": 3,
            "fluid": None,
            "dist": capex_distribution,
            "min": min_max_mean_std["min_capex"],
            "base": min_max_mean_std["mean_capex"],
            "max": min_max_mean_std["max_capex"],
            "stddev": capex_stddev,
        },
        {
            "id": 4,
            "fluid": None,
            "dist": lifting_distribution,
            "min": min_max_mean_std["min_lifting"],
            "base": min_max_mean_std["mean_lifting"],
            "max": min_max_mean_std["max_lifting"],
            "stddev": lifting_stddev,
        },
    ]

    # Leave out configuration for OIL price or GAS price based on the available fluid
    fluid_produced = [
        lifting.fluid_type
        for project in _project_contracts(contract)
        for lifting in _require_lifting_items(project.lifting)
    ]
    parameter = _require_parameters(
        [
            p
            for p in parameter_initial
            if p["fluid"] is None or p["fluid"] in fluid_produced
        ]
    )

    # Constructing the contract key
    contract_dict: SerializedContract = get_contract_attributes(
        contract=contract,
        contract_arguments=_require_mapping(
            contract_arguments, field="contract_arguments"
        ),
        summary_arguments=summary_arguments,
    )

    # Executing the montecarlo
    monte = ProcessMonte(
        contract_type=contract_type,
        contract=contract_dict,
        params=parameter,
        numSim=run_number,
    )

    # Use multiprocessing if run_number is large (i.e. larger than 400)
    try:
        if run_number <= 400:
            return monte.calculate_single_core()
        else:
            return monte.calculate_multi_cores()

    # Should error occurs, return back to serial processing
    except Exception as e:
        print(f"[Warning] Parallel mode failed: {e}")
        # print(traceback.format_exc(limit=2))

        return monte.calculate_single_core()
