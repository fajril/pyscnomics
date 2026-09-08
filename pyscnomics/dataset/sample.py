"""
A collection of methods to convert string datatype from JSON inputs
to their corresponding datatypes in the core engine.
"""

from importlib.resources import as_file, files
import json
import os
from collections.abc import Mapping
from datetime import date, datetime
from typing import (
    Callable,
    Literal,
    NotRequired,
    TypeAlias,
    TypedDict,
    TypeGuard,
    TypeVar,
    overload,
)

import numpy as np
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.gross_split import GrossSplit
from pyscnomics.contracts.project import BaseProject
from pyscnomics.econ.costs import ASR, LBT, OPEX, CapitalCost, CostOfSales, Intangible
from pyscnomics.econ.revenue import Lifting, LiftingException
from pyscnomics.econ.selection import CostType, FluidType

_T = TypeVar("_T")
_FluidName: TypeAlias = Literal[
    "Oil",
    "Gas",
    "LPG propane",
    "LPG butane",
    "Sulfur",
    "Electricity",
    "CO2",
]
_CostName: TypeAlias = Literal["sunk_cost", "preonstream_cost", "postonstream_cost"]
_Number: TypeAlias = int | float
_NumberList: TypeAlias = list[_Number]


# TypedDicts project fields consumed from trusted bundled JSON; unused extra keys stay omitted.
class _LiftingRecord(TypedDict):
    start_year: int
    end_year: int
    lifting_rate: _NumberList
    price: _NumberList
    prod_year: list[int]
    fluid_type: str
    ghv: NotRequired[_NumberList | None]
    prod_rate: NotRequired[_NumberList | None]
    prod_rate_baseline: NotRequired[_NumberList | None]


class _CostRecord(TypedDict):
    start_year: int
    end_year: int
    cost_allocation: list[str]
    description: list[str]


class _TaxedCostRecord(_CostRecord):
    tax_portion: _NumberList
    tax_discount: _NumberList


class _CapitalCostRecord(_TaxedCostRecord):
    cost: _NumberList
    expense_year: list[int]
    pis_year: list[int]
    salvage_value: _NumberList
    useful_life: list[int]
    depreciation_factor: _NumberList
    is_ic_applied: list[bool]


class _IntangibleRecord(_TaxedCostRecord):
    cost: _NumberList
    expense_year: list[int]


class _OpexRecord(_TaxedCostRecord):
    expense_year: list[int]
    fixed_cost: _NumberList


class _AsrRecord(_CostRecord):
    cost: _NumberList
    expense_year: list[int]
    final_year: list[int]
    future_rate: _NumberList


class _LbtRecord(_TaxedCostRecord):
    cost: _NumberList
    expense_year: list[int]
    final_year: list[int]
    utilized_land_area: _NumberList
    utilized_building_area: _NumberList
    njop_land: _NumberList
    njop_building: _NumberList
    gross_revenue: _NumberList


class _CostOfSalesRecord(_TaxedCostRecord):
    cost: _NumberList
    expense_year: list[int]


_LiftingData: TypeAlias = dict[str, _LiftingRecord]
_CapitalCostData: TypeAlias = dict[str, _CapitalCostRecord]
_IntangibleData: TypeAlias = dict[str, _IntangibleRecord]
_OpexData: TypeAlias = dict[str, _OpexRecord]
_AsrData: TypeAlias = dict[str, _AsrRecord]
_LbtData: TypeAlias = dict[str, _LbtRecord]
_CostOfSalesData: TypeAlias = dict[str, _CostOfSalesRecord]
_TestingValue: TypeAlias = list[_Number] | None


_json_loads: Callable[[str], object] = json.loads


def _is_object_dict(data: object) -> TypeGuard[dict[object, object]]:
    return isinstance(data, dict)


def _is_object_list(data: object) -> TypeGuard[list[object]]:
    return isinstance(data, list)


def _has_string_keys(data: dict[object, object]) -> bool:
    return all(isinstance(key, str) for key in data)


class _ProjectData(TypedDict):
    start_date: str
    end_date: str
    oil_onstream_date: str | None
    gas_onstream_date: str | None
    lifting: _LiftingData | None
    capital_cost: _CapitalCostData | None
    intangible: _IntangibleData | None
    opex: _OpexData | None
    asr: _AsrData | None
    lbt: _LbtData | None
    cost_of_sales: _CostOfSalesData | None


class _CostRecoveryConfig(TypedDict):
    oil_ftp_is_available: bool
    oil_ftp_is_shared: bool
    oil_ftp_portion: float
    gas_ftp_is_available: bool
    gas_ftp_is_shared: bool
    gas_ftp_portion: float
    oil_ctr_pretax_share: float
    gas_ctr_pretax_share: float
    oil_ic_rate: float
    gas_ic_rate: float
    ic_is_available: bool
    oil_cr_cap_rate: float
    gas_cr_cap_rate: float
    oil_dmo_volume_portion: float
    oil_dmo_fee_portion: float
    oil_dmo_holiday_duration: int
    gas_dmo_volume_portion: float
    gas_dmo_fee_portion: float
    gas_dmo_holiday_duration: int


class _GrossSplitConfig(TypedDict):
    field_status: str
    field_loc: str
    res_depth: str
    infra_avail: str
    res_type: str
    api_oil: str
    domestic_use: str
    prod_stage: str
    co2_content: str
    h2s_content: str
    oil_dmo_volume_portion: float
    oil_dmo_fee_portion: float
    oil_dmo_holiday_duration: int
    gas_dmo_volume_portion: float
    gas_dmo_fee_portion: float
    gas_dmo_holiday_duration: int


def get_json_file_names() -> list[str]:
    """
    A function to get a list of json file in the chosen directory.

    Returns
    -------
    list_json_file
        The of available dataset.
    """

    list_json_file: list[str] = []
    if "pyscnomics\\pyscnomics" in os.path.dirname(os.getcwd()):
        directory = (os.path.dirname(os.getcwd())) + "/dataset"

    else:
        directory = (os.path.dirname(os.getcwd())) + "/pyscnomics/dataset"

    for filenames in os.listdir(os.path.abspath(directory)):
        if filenames.endswith(".json"):
            full_file_name = filenames
            file_name = os.path.splitext(full_file_name)[0]
            list_json_file.append(file_name)

    return list_json_file


def read_json_file(file_name: str) -> dict[str, object]:
    """
    A function to read json file.

    Parameters
    ----------
    file_name: str
        The file name of the json file.

    Returns
    -------
    json.loads(file_contents)
        The dictionary of the json file.

    """
    # if 'pyscnomics\\pyscnomics' in os.path.dirname(os.getcwd()):
    #     directory = (os.path.dirname(os.getcwd())) + '/dataset/' + file_name + '.json'
    #
    # else:
    #     directory = (os.path.dirname(os.getcwd())) + '/pyscnomics/dataset/' + file_name + '.json'
    #
    # final_dir = os.path.abspath(directory)
    # with open(final_dir) as user_file:
    #     file_contents = user_file.read()

    # return json.loads(file_contents)

    # Locate the JSON file within the package and read its contents.
    resource = files("pyscnomics.dataset").joinpath(file_name + ".json")
    with as_file(resource) as final_dir, final_dir.open() as user_file:
        file_contents = user_file.read()

    parsed = _json_loads(file_contents)
    result: dict[str, object] = {}
    if not _is_object_dict(parsed):
        raise ValueError(
            f"JSON document {file_name!r} must be an object with string keys"
        )
    for key, value in parsed.items():
        if not isinstance(key, str):
            raise ValueError(
                f"JSON document {file_name!r} must be an object with string keys"
            )
        result[key] = value
    return result


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> TypeGuard[_Number]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_number_list(value: object) -> TypeGuard[_NumberList]:
    return _is_object_list(value) and all(_is_number(item) for item in value)


def _is_int_list(value: object) -> TypeGuard[list[int]]:
    return _is_object_list(value) and all(_is_int(item) for item in value)


def _is_bool_list(value: object) -> TypeGuard[list[bool]]:
    return _is_object_list(value) and all(isinstance(item, bool) for item in value)


def _is_string_list(value: object) -> TypeGuard[list[str]]:
    return _is_object_list(value) and all(isinstance(item, str) for item in value)


def _is_optional_number_list(value: object) -> TypeGuard[_NumberList | None]:
    return value is None or _is_number_list(value)


def _is_lifting_record(data: object) -> TypeGuard[_LiftingRecord]:
    if not _is_object_dict(data):
        return False
    return (
        _has_string_keys(data)
        and _is_int(data.get("start_year"))
        and _is_int(data.get("end_year"))
        and _is_number_list(data.get("lifting_rate"))
        and _is_number_list(data.get("price"))
        and _is_int_list(data.get("prod_year"))
        and isinstance(data.get("fluid_type"), str)
        and ("ghv" not in data or _is_optional_number_list(data["ghv"]))
        and ("prod_rate" not in data or _is_optional_number_list(data["prod_rate"]))
        and (
            "prod_rate_baseline" not in data
            or _is_optional_number_list(data["prod_rate_baseline"])
        )
    )


def _is_lifting_data(data: object) -> TypeGuard[_LiftingData]:
    if not _is_object_dict(data):
        return False
    return _has_string_keys(data) and all(
        _is_lifting_record(value) for value in data.values()
    )


def _has_cost_record_fields(data: object) -> TypeGuard[dict[str, object]]:
    if not _is_object_dict(data):
        return False
    return (
        _has_string_keys(data)
        and _is_int(data.get("start_year"))
        and _is_int(data.get("end_year"))
        and _is_string_list(data.get("cost_allocation"))
        and _is_string_list(data.get("description"))
    )


def _is_taxed_cost_record(data: object) -> TypeGuard[dict[str, object]]:
    if not _has_cost_record_fields(data):
        return False
    return _is_number_list(data.get("tax_portion")) and _is_number_list(
        data.get("tax_discount")
    )


def _is_capital_cost_record(data: object) -> TypeGuard[_CapitalCostRecord]:
    if not _is_taxed_cost_record(data):
        return False
    return (
        _is_number_list(data.get("cost"))
        and _is_int_list(data.get("expense_year"))
        and _is_int_list(data.get("pis_year"))
        and _is_number_list(data.get("salvage_value"))
        and _is_int_list(data.get("useful_life"))
        and _is_number_list(data.get("depreciation_factor"))
        and _is_bool_list(data.get("is_ic_applied"))
    )


def _is_intangible_record(data: object) -> TypeGuard[_IntangibleRecord]:
    if not _is_taxed_cost_record(data):
        return False
    return _is_number_list(data.get("cost")) and _is_int_list(data.get("expense_year"))


def _is_opex_record(data: object) -> TypeGuard[_OpexRecord]:
    if not _is_taxed_cost_record(data):
        return False
    return _is_int_list(data.get("expense_year")) and _is_number_list(
        data.get("fixed_cost")
    )


def _is_asr_record(data: object) -> TypeGuard[_AsrRecord]:
    if not _has_cost_record_fields(data):
        return False
    return (
        _is_number_list(data.get("cost"))
        and _is_int_list(data.get("expense_year"))
        and _is_int_list(data.get("final_year"))
        and _is_number_list(data.get("future_rate"))
    )


def _is_lbt_record(data: object) -> TypeGuard[_LbtRecord]:
    if not _is_taxed_cost_record(data):
        return False
    return (
        _is_number_list(data.get("cost"))
        and _is_int_list(data.get("expense_year"))
        and _is_int_list(data.get("final_year"))
        and _is_number_list(data.get("utilized_land_area"))
        and _is_number_list(data.get("utilized_building_area"))
        and _is_number_list(data.get("njop_land"))
        and _is_number_list(data.get("njop_building"))
        and _is_number_list(data.get("gross_revenue"))
    )


def _is_cost_of_sales_record(data: object) -> TypeGuard[_CostOfSalesRecord]:
    if not _is_taxed_cost_record(data):
        return False
    return _is_number_list(data.get("cost")) and _is_int_list(data.get("expense_year"))


def _is_record_data(
    data: object,
    record_validator: Callable[[object], bool],
) -> bool:
    if not _is_object_dict(data):
        return False
    return _has_string_keys(data) and all(
        record_validator(value) for value in data.values()
    )


def _is_capital_cost_data(data: object) -> TypeGuard[_CapitalCostData]:
    return _is_record_data(data, _is_capital_cost_record)


def _is_intangible_data(data: object) -> TypeGuard[_IntangibleData]:
    return _is_record_data(data, _is_intangible_record)


def _is_opex_data(data: object) -> TypeGuard[_OpexData]:
    return _is_record_data(data, _is_opex_record)


def _is_asr_data(data: object) -> TypeGuard[_AsrData]:
    return _is_record_data(data, _is_asr_record)


def _is_lbt_data(data: object) -> TypeGuard[_LbtData]:
    return _is_record_data(data, _is_lbt_record)


def _is_cost_of_sales_data(data: object) -> TypeGuard[_CostOfSalesData]:
    return _is_record_data(data, _is_cost_of_sales_record)


def _is_optional_data(
    data: object,
    validator: Callable[[object], bool],
) -> bool:
    return data is None or validator(data)


def _is_project_data(data: object) -> TypeGuard[_ProjectData]:
    if not _is_object_dict(data):
        return False
    return (
        _has_string_keys(data)
        and all(
            key in data
            for key in (
                "start_date",
                "end_date",
                "oil_onstream_date",
                "gas_onstream_date",
                "lifting",
                "capital_cost",
                "intangible",
                "opex",
                "asr",
                "lbt",
                "cost_of_sales",
            )
        )
        and isinstance(data.get("start_date"), str)
        and isinstance(data.get("end_date"), str)
        and (
            data.get("oil_onstream_date") is None
            or isinstance(data.get("oil_onstream_date"), str)
        )
        and (
            data.get("gas_onstream_date") is None
            or isinstance(data.get("gas_onstream_date"), str)
        )
        and _is_optional_data(data.get("lifting"), _is_lifting_data)
        and _is_optional_data(data.get("capital_cost"), _is_capital_cost_data)
        and _is_optional_data(data.get("intangible"), _is_intangible_data)
        and _is_optional_data(data.get("opex"), _is_opex_data)
        and _is_optional_data(data.get("asr"), _is_asr_data)
        and _is_optional_data(data.get("lbt"), _is_lbt_data)
        and _is_optional_data(data.get("cost_of_sales"), _is_cost_of_sales_data)
    )


def _is_cost_recovery_config(data: object) -> TypeGuard[_CostRecoveryConfig]:
    if not _is_object_dict(data):
        return False
    return (
        _has_string_keys(data)
        and all(
            key in data
            for key in (
                "oil_ftp_is_available",
                "oil_ftp_is_shared",
                "oil_ftp_portion",
                "gas_ftp_is_available",
                "gas_ftp_is_shared",
                "gas_ftp_portion",
                "oil_ctr_pretax_share",
                "gas_ctr_pretax_share",
                "oil_ic_rate",
                "gas_ic_rate",
                "ic_is_available",
                "oil_cr_cap_rate",
                "gas_cr_cap_rate",
                "oil_dmo_volume_portion",
                "oil_dmo_fee_portion",
                "oil_dmo_holiday_duration",
                "gas_dmo_volume_portion",
                "gas_dmo_fee_portion",
                "gas_dmo_holiday_duration",
            )
        )
        and isinstance(data.get("oil_ftp_is_available"), bool)
        and isinstance(data.get("oil_ftp_is_shared"), bool)
        and _is_number(data.get("oil_ftp_portion"))
        and isinstance(data.get("gas_ftp_is_available"), bool)
        and isinstance(data.get("gas_ftp_is_shared"), bool)
        and _is_number(data.get("gas_ftp_portion"))
        and _is_number(data.get("oil_ctr_pretax_share"))
        and _is_number(data.get("gas_ctr_pretax_share"))
        and _is_number(data.get("oil_ic_rate"))
        and _is_number(data.get("gas_ic_rate"))
        and isinstance(data.get("ic_is_available"), bool)
        and _is_number(data.get("oil_cr_cap_rate"))
        and _is_number(data.get("gas_cr_cap_rate"))
        and _is_number(data.get("oil_dmo_volume_portion"))
        and _is_number(data.get("oil_dmo_fee_portion"))
        and _is_int(data.get("oil_dmo_holiday_duration"))
        and _is_number(data.get("gas_dmo_volume_portion"))
        and _is_number(data.get("gas_dmo_fee_portion"))
        and _is_int(data.get("gas_dmo_holiday_duration"))
    )


def _is_gross_split_config(data: object) -> TypeGuard[_GrossSplitConfig]:
    if not _is_object_dict(data):
        return False
    return (
        _has_string_keys(data)
        and all(
            key in data
            for key in (
                "field_status",
                "field_loc",
                "res_depth",
                "infra_avail",
                "res_type",
                "api_oil",
                "domestic_use",
                "prod_stage",
                "co2_content",
                "h2s_content",
                "oil_dmo_volume_portion",
                "oil_dmo_fee_portion",
                "oil_dmo_holiday_duration",
                "gas_dmo_volume_portion",
                "gas_dmo_fee_portion",
                "gas_dmo_holiday_duration",
            )
        )
        and all(
            isinstance(data.get(key), str)
            for key in (
                "field_status",
                "field_loc",
                "res_depth",
                "infra_avail",
                "res_type",
                "api_oil",
                "domestic_use",
                "prod_stage",
                "co2_content",
                "h2s_content",
            )
        )
        and _is_number(data.get("oil_dmo_volume_portion"))
        and _is_number(data.get("oil_dmo_fee_portion"))
        and _is_int(data.get("oil_dmo_holiday_duration"))
        and _is_number(data.get("gas_dmo_volume_portion"))
        and _is_number(data.get("gas_dmo_fee_portion"))
        and _is_int(data.get("gas_dmo_holiday_duration"))
    )


def _is_testing_data(data: object) -> TypeGuard[dict[str, _TestingValue]]:
    if not _is_object_dict(data):
        return False
    return _has_string_keys(data) and all(
        value is None or _is_number_list(value) for value in data.values()
    )


@overload
def read_fluid_type(fluid: _FluidName) -> FluidType: ...


@overload
def read_fluid_type(fluid: str) -> FluidType | None: ...


@overload
def read_fluid_type(fluid: list[_T]) -> list[FluidType]: ...


def read_fluid_type(
    fluid: str | list[object],
) -> list[FluidType] | FluidType | None:
    """
    A function to convert the str input into FluidType Enum class.

    Parameters
    ----------
    fluid: str | list
        The fluid type.

    Returns
    -------
    list[FluidType] | FluidType
        The enum class for FluidType.

    """
    if isinstance(fluid, str):
        if fluid == "Oil":
            return FluidType.OIL
        elif fluid == "Gas":
            return FluidType.GAS
        elif fluid in ("LPG propane", "LPG butane"):
            # LPG is included in backend gas lifting aggregate.
            return FluidType.GAS
        elif fluid == "Sulfur":
            return FluidType.SULFUR
        elif fluid == "Electricity":
            return FluidType.ELECTRICITY
        elif fluid == "CO2":
            return FluidType.CO2

    else:
        fluid_mapping: dict[object, FluidType] = {
            "Oil": FluidType.OIL,
            "Gas": FluidType.GAS,
            "LPG propane": FluidType.GAS,
            "LPG butane": FluidType.GAS,
            "Sulfur": FluidType.SULFUR,
            "Electricity": FluidType.ELECTRICITY,
            "CO2": FluidType.CO2,
        }

        # Replace elements in the list using the mapping
        result = [fluid_mapping[i] for i in fluid if i in fluid_mapping]

        return result


@overload
def read_cost_type(cost_type: _CostName) -> CostType: ...


@overload
def read_cost_type(cost_type: str) -> CostType: ...


@overload
def read_cost_type(cost_type: list[_T]) -> list[CostType | None]: ...


@overload
def read_cost_type(
    cost_type: object,
) -> CostType | list[CostType | None] | TypeError: ...


def read_cost_type(
    cost_type: object,
) -> CostType | list[CostType | None] | TypeError:
    """
    Convert cost type(s) from string representation to corresponding CostType enum.

    This function accepts either a single string or a list of strings representing
    cost types and returns the corresponding `CostType` enum instance(s).
    Valid cost type strings include:
    - "sunk_cost"
    - "preonstream_cost"
    - "postonstream_cost"

    Parameters
    ----------
    cost_type : str or list of str
        Cost type(s) to be converted. Can be a single string (e.g., "Sunk Cost")
        or a list of strings (e.g., ["Sunk Cost", "Postonstream Cost"]).

    Returns
    -------
    CostType or list of CostType
        The corresponding `CostType` enum instance(s) matching the input.
        Returns a single `CostType` if input is a string, otherwise returns
        a list of `CostType` objects.

    Raises
    ------
    TypeError
        If `cost_type` is not a string or list, or if any element within the list
        is not a valid cost type string.
    KeyError
        If a string input does not match any recognized cost type.

    Notes
    -----
    Valid cost type strings are case-sensitive and must exactly match one of the
    following:
    - "sunk_cost"
    - "preonstream_cost"
    - "postonstream_cost"
    """

    # Transformation string -> CostType(Enum) for string input
    if isinstance(cost_type, str):
        if cost_type == "sunk_cost":
            return CostType.SUNK_COST

        elif cost_type == "preonstream_cost":
            return CostType.PRE_ONSTREAM_COST

        elif cost_type == "postonstream_cost":
            return CostType.POST_ONSTREAM_COST

        else:
            raise KeyError(f"Parameter cost_type ({cost_type}) is unrecognized")

    # Transformation string -> CostType(Enum) for list input
    if not _is_object_list(cost_type):
        return TypeError(
            f"Parameter cost_type must be a string or a list, not {cost_type.__class__.__qualname__}"
        )
    result: list[CostType | None] = []
    for ct in cost_type:
        if ct == "sunk_cost":
            result.append(CostType.SUNK_COST)
        elif ct == "preonstream_cost":
            result.append(CostType.PRE_ONSTREAM_COST)
        elif ct == "postonstream_cost":
            result.append(CostType.POST_ONSTREAM_COST)
        elif ct is None:
            result.append(None)
        else:
            raise TypeError(f"Invalid cost type: {ct}")

    return result


def assign_onstream_date(date_data: None | str) -> date | None:
    """
    A funtion to assign onstream date into datetime format.
    Parameters
    ----------
    date_data: None | str
        date data read from the json file or Ms. Excel file.

    Returns
    -------
    result_date: datetime
        The date in the datetime format

    """
    if date_data is not None:
        result_date = datetime.strptime(date_data, "%d/%m/%Y").date()
    else:
        result_date = None

    return result_date


def _build_lifting(lifting_value: _LiftingData | None) -> tuple[Lifting, ...] | None:
    """
    A function to assign the lifting data to the list of Lifting dataclass.

    Parameters
    ----------
    data_raw: dict
        The dictionary containing the lifting data.

    Returns
    -------
    lifting_list: tuple | None
        The list containing the lifting dataclass.
    """

    # Defining data source and container for Lifting (container as list datatype).
    # Then, assign them based on their fluid type
    if lifting_value is None:
        return None

    lifting_list: list[Lifting] = []
    for key in lifting_value:
        record = lifting_value[key]
        if "prod_rate" not in record:
            raise KeyError("prod_rate")
        if "prod_rate_baseline" not in record:
            raise KeyError("prod_rate_baseline")
        prod_rate_data = record.get("prod_rate")
        prod_rate_baseline_data = record.get("prod_rate_baseline")
        fluid_value = record["fluid_type"]
        if "Gas" in key or "GSA" in key:
            if "ghv" not in record:
                raise KeyError("ghv")
            ghv_data = record.get("ghv")
            # Since the Lifting data for gas has different arguments input,
            # conditional formatting is applied
            lifting = Lifting(
                start_year=record["start_year"],
                end_year=record["end_year"],
                lifting_rate=np.array(record["lifting_rate"]),
                price=np.array(record["price"]),
                prod_year=np.array(record["prod_year"]),
                fluid_type=read_fluid_type(fluid_value),
                ghv=(None if ghv_data is None else np.array(ghv_data)),
                prod_rate=(
                    None if prod_rate_data is None else np.array(prod_rate_data)
                ),
                prod_rate_baseline=(
                    None
                    if prod_rate_baseline_data is None
                    else np.array(prod_rate_baseline_data)
                ),
            )
        else:
            lifting = Lifting(
                start_year=record["start_year"],
                end_year=record["end_year"],
                lifting_rate=np.array(record["lifting_rate"]),
                price=np.array(record["price"]),
                prod_year=np.array(record["prod_year"]),
                fluid_type=read_fluid_type(fluid_value),
                prod_rate=(
                    None if prod_rate_data is None else np.array(prod_rate_data)
                ),
                prod_rate_baseline=(
                    None
                    if prod_rate_baseline_data is None
                    else np.array(prod_rate_baseline_data)
                ),
            )
        lifting_list.append(lifting)

    return tuple(lifting_list)


def assign_lifting(data_raw: Mapping[str, object]) -> tuple[Lifting, ...] | None:
    """Convert raw lifting data while retaining the historical public errors."""
    lifting_value = data_raw["lifting"]
    if lifting_value is None:
        return None
    if not _is_lifting_data(lifting_value):
        if _is_object_dict(lifting_value):
            records = lifting_value
            for record_key, record_value in records.items():
                if _is_object_dict(record_value):
                    record = record_value
                    for field in (
                        "start_year",
                        "end_year",
                        "lifting_rate",
                        "price",
                        "prod_year",
                        "fluid_type",
                    ):
                        if field not in record:
                            raise KeyError(field)
                    if "prod_rate" not in record:
                        raise KeyError("prod_rate")
                    if "prod_rate_baseline" not in record:
                        raise KeyError("prod_rate_baseline")
                    fluid_value = record.get("fluid_type")
                    if not isinstance(fluid_value, str):
                        raise LiftingException(
                            f"Attribute fluid_type must be provided as an instance of FluidType, not as an/a {type(fluid_value).__qualname__}"
                        )
                    if (
                        "Gas" in str(record_key) or "GSA" in str(record_key)
                    ) and "ghv" not in record:
                        raise KeyError("ghv")
        raise ValueError("Invalid lifting data")
    return _build_lifting(lifting_value)


def _build_cost(
    capital_cost_value: _CapitalCostData | None,
    intangible_value: _IntangibleData | None,
    opex_value: _OpexData | None,
    asr_value: _AsrData | None,
    lbt_value: _LbtData | None,
    cos_value: _CostOfSalesData | None,
) -> tuple[
    list[CapitalCost] | None,
    list[Intangible] | None,
    list[OPEX] | None,
    list[ASR] | None,
    list[LBT] | None,
    list[CostOfSales] | None,
]:
    """
    A function to assign the cost data to the list of Tangible, Intangible, OPEX, ASR, LBT and CostOfSales dataclass.

    Parameters
    ----------
    data_raw: dict

    Returns
    -------
    tangible_list : tuple
        A tuple containing the Tangible dataclass.
    intangible_list : tuple
        A tuple containing the Intangible dataclass.
    opex_list
        A tuple containing the OPEX dataclass.
    asr_list
        A tuple containing the ASR dataclass.
    lbt_list
        A tuple containing the LBT dataclass.
    cos_list
        A tuple containing the CostOfSales dataclass.

    """
    # Defining the data source and the list container for Tangible. Then, assign them based on their fluid type
    if capital_cost_value is None:
        capital_cost_list = None
    else:
        capital_cost_list = [
            CapitalCost(
                start_year=capital_cost_value[key]["start_year"],
                end_year=capital_cost_value[key]["end_year"],
                cost=np.array(capital_cost_value[key]["cost"]),
                expense_year=np.array(capital_cost_value[key]["expense_year"]),
                cost_allocation=read_fluid_type(
                    fluid=capital_cost_value[key]["cost_allocation"]
                ),
                description=capital_cost_value[key]["description"],
                tax_portion=np.array(capital_cost_value[key]["tax_portion"]),
                tax_discount=np.array(capital_cost_value[key]["tax_discount"]),
                pis_year=np.array(capital_cost_value[key]["pis_year"]),
                salvage_value=np.array(capital_cost_value[key]["salvage_value"]),
                useful_life=np.array(capital_cost_value[key]["useful_life"]),
                depreciation_factor=np.array(
                    capital_cost_value[key]["depreciation_factor"]
                ),
                is_ic_applied=capital_cost_value[key]["is_ic_applied"],
            )
            for key in capital_cost_value
        ]

    # Defining the data source and the list container for Intangible. Then, assign them based on their fluid type
    if intangible_value is None:
        intangible_list = None
    else:
        intangible_list = [
            Intangible(
                start_year=intangible_value[key]["start_year"],
                end_year=intangible_value[key]["end_year"],
                cost=np.array(intangible_value[key]["cost"]),
                expense_year=np.array(intangible_value[key]["expense_year"]),
                cost_allocation=read_fluid_type(
                    fluid=intangible_value[key]["cost_allocation"]
                ),
                description=intangible_value[key]["description"],
                tax_portion=np.array(intangible_value[key]["tax_portion"]),
                tax_discount=np.array(intangible_value[key]["tax_discount"]),
            )
            for key in intangible_value
        ]

    # Defining the data source and the list container for Intangible. Then, them based on their fluid type
    if opex_value is None:
        opex_list = None
    else:
        opex_list = [
            OPEX(
                start_year=opex_value[key]["start_year"],
                end_year=opex_value[key]["end_year"],
                expense_year=np.array(opex_value[key]["expense_year"]),
                cost_allocation=read_fluid_type(
                    fluid=opex_value[key]["cost_allocation"]
                ),
                description=opex_value[key]["description"],
                tax_portion=np.array(opex_value[key]["tax_portion"]),
                tax_discount=np.array(opex_value[key]["tax_discount"]),
                fixed_cost=np.array(opex_value[key]["fixed_cost"]),
            )
            for key in opex_value
        ]

    # Defining the data source and the list container for ASR. Then, them based on their fluid type
    if asr_value is None:
        asr_list = None
    else:
        asr_list = [
            ASR(
                start_year=asr_value[key]["start_year"],
                end_year=asr_value[key]["end_year"],
                cost=np.array(asr_value[key]["cost"]),
                expense_year=np.array(asr_value[key]["expense_year"]),
                cost_allocation=read_fluid_type(
                    fluid=asr_value[key]["cost_allocation"]
                ),
                description=asr_value[key]["description"],
                final_year=np.array(asr_value[key]["final_year"]),
                future_rate=np.array(asr_value[key]["future_rate"]),
            )
            for key in asr_value
        ]

    # Defining the data source and the list container for ASR. Then, them based on their fluid type
    if lbt_value is None:
        lbt_list = None
    else:
        lbt_list = [
            LBT(
                start_year=lbt_value[key]["start_year"],
                end_year=lbt_value[key]["end_year"],
                cost=np.array(lbt_value[key]["cost"]),
                expense_year=np.array(lbt_value[key]["expense_year"]),
                cost_allocation=read_fluid_type(
                    fluid=lbt_value[key]["cost_allocation"]
                ),
                description=lbt_value[key]["description"],
                tax_portion=np.array(lbt_value[key]["tax_portion"]),
                tax_discount=np.array(lbt_value[key]["tax_discount"]),
                final_year=np.array(lbt_value[key]["final_year"]),
                utilized_land_area=np.array(lbt_value[key]["utilized_land_area"]),
                utilized_building_area=np.array(
                    lbt_value[key]["utilized_building_area"]
                ),
                njop_land=np.array(lbt_value[key]["njop_land"]),
                njop_building=np.array(lbt_value[key]["njop_building"]),
                gross_revenue=np.array(lbt_value[key]["gross_revenue"]),
            )
            for key in lbt_value
        ]

    # Defining the data source and the list container for Cost of Sales. Then, them based on their fluid type
    if cos_value is None:
        cos_list = None
    else:
        cos_list = [
            CostOfSales(
                start_year=cos_value[key]["start_year"],
                end_year=cos_value[key]["end_year"],
                cost=np.array(cos_value[key]["cost"]),
                expense_year=np.array(cos_value[key]["expense_year"]),
                cost_allocation=read_fluid_type(
                    fluid=cos_value[key]["cost_allocation"]
                ),
                description=cos_value[key]["description"],
                tax_portion=np.array(cos_value[key]["tax_portion"]),
                tax_discount=np.array(cos_value[key]["tax_discount"]),
            )
            for key in cos_value
        ]

    return capital_cost_list, intangible_list, opex_list, asr_list, lbt_list, cos_list


def _raise_missing_cost_key(data: object, fields: tuple[str, ...]) -> None:
    if not _is_object_dict(data):
        raise ValueError("Invalid cost data")
    for record_value in data.values():
        if not _is_object_dict(record_value):
            raise ValueError("Invalid cost data")
        for field in fields:
            if field not in record_value:
                raise KeyError(field)


def assign_cost(
    data_raw: Mapping[str, object],
) -> tuple[
    list[CapitalCost] | None,
    list[Intangible] | None,
    list[OPEX] | None,
    list[ASR] | None,
    list[LBT] | None,
    list[CostOfSales] | None,
]:
    """Convert raw cost data while retaining the historical public errors."""
    capital_raw = data_raw["capital_cost"]
    intangible_raw = data_raw["intangible"]
    opex_raw = data_raw["opex"]
    asr_raw = data_raw["asr"]
    lbt_raw = data_raw["lbt"]
    cos_raw = data_raw["cost_of_sales"]

    if capital_raw is None:
        capital_data = None
    elif _is_capital_cost_data(capital_raw):
        capital_data = capital_raw
    else:
        _raise_missing_cost_key(
            capital_raw,
            (
                "start_year",
                "end_year",
                "cost",
                "expense_year",
                "cost_allocation",
                "description",
                "tax_portion",
                "tax_discount",
                "pis_year",
                "salvage_value",
                "useful_life",
                "depreciation_factor",
                "is_ic_applied",
            ),
        )
        raise ValueError("Invalid capital cost data")
    if intangible_raw is None:
        intangible_data = None
    elif _is_intangible_data(intangible_raw):
        intangible_data = intangible_raw
    else:
        _raise_missing_cost_key(
            intangible_raw,
            (
                "start_year",
                "end_year",
                "cost",
                "expense_year",
                "cost_allocation",
                "description",
                "tax_portion",
                "tax_discount",
            ),
        )
        raise ValueError("Invalid intangible data")
    if opex_raw is None:
        opex_data = None
    elif _is_opex_data(opex_raw):
        opex_data = opex_raw
    else:
        _raise_missing_cost_key(
            opex_raw,
            (
                "start_year",
                "end_year",
                "expense_year",
                "cost_allocation",
                "description",
                "tax_portion",
                "tax_discount",
                "fixed_cost",
            ),
        )
        raise ValueError("Invalid opex data")
    if asr_raw is None:
        asr_data = None
    elif _is_asr_data(asr_raw):
        asr_data = asr_raw
    else:
        _raise_missing_cost_key(
            asr_raw,
            (
                "start_year",
                "end_year",
                "cost",
                "expense_year",
                "cost_allocation",
                "description",
                "final_year",
                "future_rate",
            ),
        )
        raise ValueError("Invalid asr data")
    if lbt_raw is None:
        lbt_data = None
    elif _is_lbt_data(lbt_raw):
        lbt_data = lbt_raw
    else:
        _raise_missing_cost_key(
            lbt_raw,
            (
                "start_year",
                "end_year",
                "cost",
                "expense_year",
                "cost_allocation",
                "description",
                "tax_portion",
                "tax_discount",
                "final_year",
                "utilized_land_area",
                "utilized_building_area",
                "njop_land",
                "njop_building",
                "gross_revenue",
            ),
        )
        raise ValueError("Invalid lbt data")
    if cos_raw is None:
        cos_data = None
    elif _is_cost_of_sales_data(cos_raw):
        cos_data = cos_raw
    else:
        _raise_missing_cost_key(
            cos_raw,
            (
                "start_year",
                "end_year",
                "cost",
                "expense_year",
                "cost_allocation",
                "description",
                "tax_portion",
                "tax_discount",
            ),
        )
        raise ValueError("Invalid cost of sales data")

    return _build_cost(
        capital_data, intangible_data, opex_data, asr_data, lbt_data, cos_data
    )


def load_testing(dataset_type: str, key: str) -> _TestingValue | ValueError:
    """
    A function to load the available test dataset.

    Parameters
    ----------
    dataset_type
    key

    Returns
    -------

    """
    data = read_json_file(file_name=dataset_type)
    # Condition where the testing data is not available within the chosen dataset
    testing = data.get("testing")
    if testing is None:
        return ValueError("The testing data for this dataset is still in development.")

    if not _is_testing_data(testing):
        raise ValueError("Testing data must be an object with string keys")
    return testing[key]


@overload
def load_data(
    dataset_type: str,
    contract_type: Literal["project"] = "project",
) -> BaseProject: ...


@overload
def load_data(
    dataset_type: str, contract_type: Literal["cost_recovery"]
) -> CostRecovery: ...


@overload
def load_data(
    dataset_type: str, contract_type: Literal["gross_split"]
) -> GrossSplit: ...


@overload
def load_data(
    dataset_type: str, contract_type: str
) -> BaseProject | CostRecovery | GrossSplit | None: ...


def load_data(
    dataset_type: str,
    contract_type: str = "project",
) -> BaseProject | CostRecovery | GrossSplit | None:
    """
    A function to load the available dataset.

    Parameters
    ----------
    dataset_type: str
        The type of the dataset.
    contract_type
        The type of the contract.
    Returns
    -------
    BaseProject | CostRecovery | GrossSplit

    """
    # Checking the availability of the dataset, if not available raise a Value Error
    # dataset_list = get_json_file_names()
    # contract_list = ['project', 'cost_recovery', 'gross_split']
    # if dataset_type not in dataset_list or contract_type not in contract_list:
    #     raise ValueError('Unknown dataset: "{0}", please check the Dataset and Contract Type that available.'
    #                      .format(dataset_type))

    # Reading the project json file
    data_raw = read_json_file(file_name=dataset_type)
    if not _is_project_data(data_raw):
        raise ValueError("Invalid project data")

    # Assigning the general config and converting them to datetime format.
    project_start_date = datetime.strptime(data_raw["start_date"], "%d/%m/%Y").date()
    project_end_date = datetime.strptime(data_raw["end_date"], "%d/%m/%Y").date()

    oil_onstream_date = assign_onstream_date(data_raw["oil_onstream_date"])
    gas_onstream_date = assign_onstream_date(data_raw["gas_onstream_date"])

    # Assigning the lifting and cost data.
    lifting_tuple = _build_lifting(data_raw["lifting"])
    capital_list, intangible_list, opex_list, asr_list, lbt_list, cos_list = (
        _build_cost(
            data_raw["capital_cost"],
            data_raw["intangible"],
            data_raw["opex"],
            data_raw["asr"],
            data_raw["lbt"],
            data_raw["cost_of_sales"],
        )
    )
    capital_tuple = tuple(capital_list) if capital_list is not None else None
    intangible_tuple = tuple(intangible_list) if intangible_list is not None else None
    opex_tuple = tuple(opex_list) if opex_list is not None else None
    asr_tuple = tuple(asr_list) if asr_list is not None else None
    lbt_tuple = tuple(lbt_list) if lbt_list is not None else None
    cos_tuple = tuple(cos_list) if cos_list is not None else None

    if contract_type == "project":
        return BaseProject(
            start_date=project_start_date,
            end_date=project_end_date,
            approval_year=project_start_date.year,
            oil_onstream_date=oil_onstream_date,
            gas_onstream_date=gas_onstream_date,
            lifting=lifting_tuple,
            capital_cost=capital_tuple,
            intangible_cost=intangible_tuple,
            opex=opex_tuple,
            asr_cost=asr_tuple,
            lbt_cost=lbt_tuple,
            cost_of_sales=cos_tuple,
        )

    elif contract_type == "cost_recovery":
        cr_config_raw = read_json_file(file_name=contract_type)
        if not _is_cost_recovery_config(cr_config_raw):
            raise ValueError("Invalid cost recovery configuration")
        cr_config = cr_config_raw
        return CostRecovery(
            start_date=project_start_date,
            end_date=project_end_date,
            approval_year=project_start_date.year,
            oil_onstream_date=oil_onstream_date,
            gas_onstream_date=gas_onstream_date,
            lifting=lifting_tuple,
            capital_cost=capital_tuple,
            intangible_cost=intangible_tuple,
            opex=opex_tuple,
            asr_cost=asr_tuple,
            lbt_cost=lbt_tuple,
            cost_of_sales=cos_tuple,
            oil_ftp_is_available=cr_config["oil_ftp_is_available"],
            oil_ftp_is_shared=cr_config["oil_ftp_is_shared"],
            oil_ftp_portion=cr_config["oil_ftp_portion"],
            gas_ftp_is_available=cr_config["gas_ftp_is_available"],
            gas_ftp_is_shared=cr_config["gas_ftp_is_shared"],
            gas_ftp_portion=cr_config["gas_ftp_portion"],
            oil_ctr_pretax_share=cr_config["oil_ctr_pretax_share"],
            gas_ctr_pretax_share=cr_config["gas_ctr_pretax_share"],
            oil_ic_rate=cr_config["oil_ic_rate"],
            gas_ic_rate=cr_config["gas_ic_rate"],
            ic_is_available=cr_config["ic_is_available"],
            oil_cr_cap_rate=cr_config["oil_cr_cap_rate"],
            gas_cr_cap_rate=cr_config["gas_cr_cap_rate"],
            oil_dmo_volume_portion=cr_config["oil_dmo_volume_portion"],
            oil_dmo_fee_portion=cr_config["oil_dmo_fee_portion"],
            oil_dmo_holiday_duration=cr_config["oil_dmo_holiday_duration"],
            gas_dmo_volume_portion=cr_config["gas_dmo_volume_portion"],
            gas_dmo_fee_portion=cr_config["gas_dmo_fee_portion"],
            gas_dmo_holiday_duration=cr_config["gas_dmo_holiday_duration"],
        )

    elif contract_type == "gross_split":
        gs_config_raw = read_json_file(file_name=contract_type)
        if not _is_gross_split_config(gs_config_raw):
            raise ValueError("Invalid gross split configuration")
        gs_config = gs_config_raw
        return GrossSplit(
            start_date=project_start_date,
            end_date=project_end_date,
            approval_year=project_start_date.year,
            oil_onstream_date=oil_onstream_date,
            gas_onstream_date=gas_onstream_date,
            lifting=lifting_tuple,
            capital_cost=capital_tuple,
            intangible_cost=intangible_tuple,
            opex=opex_tuple,
            asr_cost=asr_tuple,
            lbt_cost=lbt_tuple,
            cost_of_sales=cos_tuple,
            field_status=gs_config["field_status"],
            field_loc=gs_config["field_loc"],
            res_depth=gs_config["res_depth"],
            infra_avail=gs_config["infra_avail"],
            res_type=gs_config["res_type"],
            api_oil=gs_config["api_oil"],
            domestic_use=gs_config["domestic_use"],
            prod_stage=gs_config["prod_stage"],
            co2_content=gs_config["co2_content"],
            h2s_content=gs_config["h2s_content"],
            oil_dmo_volume_portion=gs_config["oil_dmo_volume_portion"],
            oil_dmo_fee_portion=gs_config["oil_dmo_fee_portion"],
            oil_dmo_holiday_duration=gs_config["oil_dmo_holiday_duration"],
            gas_dmo_volume_portion=gs_config["gas_dmo_volume_portion"],
            gas_dmo_fee_portion=gs_config["gas_dmo_fee_portion"],
            gas_dmo_holiday_duration=gs_config["gas_dmo_holiday_duration"],
        )
