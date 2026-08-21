"""
A collection of methods to convert string datatype from JSON inputs
to their corresponding datatypes in the core engine.
"""

import importlib.resources as pkg_resources
import json
import os
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Literal, NotRequired, TypeAlias, TypedDict, TypeVar, cast, overload

import numpy as np

from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.gross_split import GrossSplit
from pyscnomics.contracts.project import BaseProject
from pyscnomics.econ.costs import ASR, LBT, OPEX, CapitalCost, CostOfSales, Intangible
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.selection import CostType, FluidType

_T = TypeVar('_T')
_FluidName: TypeAlias = Literal['Oil', 'Gas', 'Sulfur', 'Electricity', 'CO2']
_CostName: TypeAlias = Literal['sunk_cost', 'preonstream_cost', 'postonstream_cost']
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
    if 'pyscnomics\\pyscnomics' in os.path.dirname(os.getcwd()):
        directory = (os.path.dirname(os.getcwd())) + '/dataset'

    else:
        directory = (os.path.dirname(os.getcwd())) + '/pyscnomics/dataset'

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

    # Use importlib.resources.path to locate the JSON file within the package
    with pkg_resources.path('pyscnomics.dataset', file_name + '.json') as file_path:
        final_dir = Path(file_path)

    # Open and read the JSON file, returning its contents as a dictionary
    with open(final_dir) as user_file:
        file_contents = user_file.read()

    parsed: object = cast(object, json.loads(file_contents))
    if not isinstance(parsed, dict):
        raise ValueError(
            f"JSON document {file_name!r} must be an object with string keys"
        )
    parsed_dict = cast(dict[object, object], parsed)
    if not all(isinstance(key, str) for key in parsed_dict):
        raise ValueError(
            f"JSON document {file_name!r} must be an object with string keys"
        )

    return cast(dict[str, object], parsed_dict)


# Bundled fixture schemas are trusted after read_json_file's top-level check.
def _as_project_data(data: object) -> _ProjectData:
    return cast(_ProjectData, data)


def _as_cost_recovery_config(data: object) -> _CostRecoveryConfig:
    return cast(_CostRecoveryConfig, data)


def _as_gross_split_config(data: object) -> _GrossSplitConfig:
    return cast(_GrossSplitConfig, data)


def _as_testing_data(data: object) -> dict[str, _TestingValue]:
    if not isinstance(data, dict):
        raise ValueError('Testing data must be an object with string keys')
    data_dict = cast(dict[object, object], data)
    if not all(isinstance(key, str) for key in data_dict):
        raise ValueError('Testing data must be an object with string keys')
    return cast(dict[str, _TestingValue], data_dict)


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
        if fluid == 'Oil':
            return FluidType.OIL
        elif fluid == 'Gas':
            return FluidType.GAS
        elif fluid == 'Sulfur':
            return FluidType.SULFUR
        elif fluid == 'Electricity':
            return FluidType.ELECTRICITY
        elif fluid == 'CO2':
            return FluidType.CO2

    else:
        fluid_mapping: dict[object, FluidType] = {'Oil': FluidType.OIL,
                         'Gas': FluidType.GAS,
                         'Sulfur': FluidType.SULFUR,
                         'Electricity': FluidType.ELECTRICITY,
                         'CO2': FluidType.CO2}

        # Replace elements in the list using the mapping
        result = [fluid_mapping[i] for i in fluid if i in fluid_mapping]

        return result


def _read_fluid(fluid: str) -> FluidType | None:
    return read_fluid_type(fluid)


@overload
def read_cost_type(cost_type: _CostName) -> CostType: ...


@overload
def read_cost_type(cost_type: str) -> CostType: ...


@overload
def read_cost_type(cost_type: list[_T]) -> list[CostType | None]: ...


@overload
def read_cost_type(cost_type: object) -> CostType | list[CostType | None] | TypeError: ...


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

    # Only allow string or list as the datatype of `cost_type`
    if not isinstance(cost_type, (str, list)):
        return TypeError(
            f"Parameter cost_type must be a string or a list, not {cost_type.__class__.__qualname__}"
        )

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
    else:
        cost_type_list = cast(list[object], cost_type)
        cost_type_mapping: dict[object, CostType | None] = {
            "sunk_cost": CostType.SUNK_COST,
            "preonstream_cost": CostType.PRE_ONSTREAM_COST,
            "postonstream_cost": CostType.POST_ONSTREAM_COST,
            None: None,
        }

        # Replace each string inside the cost_type list into their
        # corresponding CostType(Enum) counterparts
        result: list[CostType | None] = [None] * len(cost_type_list)
        for i, ct in enumerate(cost_type_list):
            if ct in cost_type_mapping.keys():
                result[i] = cost_type_mapping[ct]
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
        result_date = datetime.strptime(date_data, '%d/%m/%Y').date()
    else:
        result_date = None

    return result_date


def assign_lifting(data_raw: Mapping[str, object]) -> tuple[Lifting, ...] | None:
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
    lifting_data = cast(_LiftingData | None, data_raw['lifting'])

    if lifting_data is None:
        return None

    else:
        lifting_list: list[Lifting] = []
        for key in lifting_data.keys():
            record = lifting_data[key]
            # Since the Lifting data for gas has different arguments input,
            # conditional formatting is applied
            if 'Gas' in key or 'GSA' in key:
                lifting = Lifting(
                    start_year=record["start_year"],
                    end_year=record["end_year"],
                    lifting_rate=np.array(record["lifting_rate"]),
                    price=np.array(record["price"]),
                    prod_year=np.array(record["prod_year"]),
                    fluid_type=_read_fluid(record["fluid_type"]),  # pyright: ignore[reportArgumentType]  # legacy-nullability
                    # legacy-nullability: Lifting normalizes optional bundled arrays.
                    ghv=(  # pyright: ignore[reportArgumentType]  # legacy-nullability
                        None
                        if record["ghv"] is None  # pyright: ignore[reportTypedDictNotRequiredAccess]
                        else np.array(record["ghv"])  # pyright: ignore[reportTypedDictNotRequiredAccess]
                    ),
                    prod_rate=(  # pyright: ignore[reportArgumentType]  # legacy-nullability
                        None
                        if record["prod_rate"] is None  # pyright: ignore[reportTypedDictNotRequiredAccess]
                        or "prod_rate" not in record
                        else np.array(record["prod_rate"])
                    ),
                    prod_rate_baseline=(  # pyright: ignore[reportArgumentType]  # legacy-nullability
                        None
                        if record["prod_rate_baseline"] is None  # pyright: ignore[reportTypedDictNotRequiredAccess]
                        or "prod_rate_baseline" not in record
                        else np.array(record["prod_rate_baseline"])
                    ),
                )

                lifting_list.append(lifting)

            else:
                lifting = Lifting(
                    start_year=record["start_year"],
                    end_year=record["end_year"],
                    lifting_rate=np.array(record["lifting_rate"]),
                    price=np.array(record["price"]),
                    prod_year=np.array(record["prod_year"]),
                    fluid_type=_read_fluid(record["fluid_type"]),  # pyright: ignore[reportArgumentType]  # legacy-nullability
                    # legacy-nullability: Lifting normalizes optional bundled arrays.
                    prod_rate=(  # pyright: ignore[reportArgumentType]  # legacy-nullability
                        None
                        if record["prod_rate"] is None  # pyright: ignore[reportTypedDictNotRequiredAccess]
                        or "prod_rate" not in record
                        else np.array(record["prod_rate"])
                    ),
                    prod_rate_baseline=(  # pyright: ignore[reportArgumentType]  # legacy-nullability
                        None
                        if record["prod_rate_baseline"] is None  # pyright: ignore[reportTypedDictNotRequiredAccess]
                        or "prod_rate_baseline" not in record
                        else np.array(record["prod_rate_baseline"])
                    ),
                )

                lifting_list.append(lifting)

        return tuple(lifting_list)


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
    capital_cost_data = cast(_CapitalCostData | None, data_raw['capital_cost'])
    capital_cost_list = ([
        CapitalCost(
            start_year=capital_cost_data[key]['start_year'],
            end_year=capital_cost_data[key]['end_year'],
            cost=np.array(capital_cost_data[key]['cost']),
            expense_year=np.array(capital_cost_data[key]['expense_year']),
            cost_allocation=read_fluid_type(fluid=capital_cost_data[key]['cost_allocation']),
            description=capital_cost_data[key]['description'],
            tax_portion=np.array(capital_cost_data[key]['tax_portion']),
            tax_discount=np.array(capital_cost_data[key]['tax_discount']),
            pis_year=np.array(capital_cost_data[key]['pis_year']),
            salvage_value=np.array(capital_cost_data[key]['salvage_value']),
            useful_life=np.array(capital_cost_data[key]['useful_life']),
            depreciation_factor=np.array(capital_cost_data[key]['depreciation_factor']),
            is_ic_applied=capital_cost_data[key]['is_ic_applied'],
        ) for key in capital_cost_data.keys()
    ]) if capital_cost_data is not None else None

    # Defining the data source and the list container for Intangible. Then, assign them based on their fluid type
    intangible_data = cast(_IntangibleData | None, data_raw['intangible'])
    intangible_list = ([
        Intangible(
            start_year=intangible_data[key]['start_year'],
            end_year=intangible_data[key]['end_year'],
            cost=np.array(intangible_data[key]['cost']),
            expense_year=np.array(intangible_data[key]['expense_year']),
            cost_allocation=read_fluid_type(fluid=intangible_data[key]['cost_allocation']),
            description=intangible_data[key]['description'],
            tax_portion=np.array(intangible_data[key]['tax_portion']),
            tax_discount=np.array(intangible_data[key]['tax_discount']),
        ) for key in intangible_data.keys()
    ]) if intangible_data is not None else None

    # Defining the data source and the list container for Intangible. Then, them based on their fluid type
    opex_data = cast(_OpexData | None, data_raw['opex'])
    opex_list = ([
        OPEX(
            start_year=opex_data[key]['start_year'],
            end_year=opex_data[key]['end_year'],
            expense_year=np.array(opex_data[key]['expense_year']),
            cost_allocation=read_fluid_type(fluid=opex_data[key]['cost_allocation']),
            description=opex_data[key]['description'],
            tax_portion=np.array(opex_data[key]['tax_portion']),
            tax_discount=np.array(opex_data[key]['tax_discount']),
            fixed_cost=np.array(opex_data[key]['fixed_cost'])
        ) for key in opex_data.keys()
    ]) if opex_data is not None else None

    # Defining the data source and the list container for ASR. Then, them based on their fluid type
    asr_data = cast(_AsrData | None, data_raw['asr'])
    asr_list = ([
        ASR(
            start_year=asr_data[key]['start_year'],
            end_year=asr_data[key]['end_year'],
            cost=np.array(asr_data[key]['cost']),
            expense_year=np.array(asr_data[key]['expense_year']),
            cost_allocation=read_fluid_type(fluid=asr_data[key]['cost_allocation']),
            description=asr_data[key]['description'],
            final_year=np.array(asr_data[key]['final_year']),
            future_rate=np.array(asr_data[key]['future_rate']),
        ) for key in asr_data.keys()
    ]) if asr_data is not None else None

    # Defining the data source and the list container for ASR. Then, them based on their fluid type
    lbt_data = cast(_LbtData | None, data_raw['lbt'])
    lbt_list = ([
        LBT(
            start_year=lbt_data[key]['start_year'],
            end_year=lbt_data[key]['end_year'],
            cost=np.array(lbt_data[key]['cost']),
            expense_year=np.array(lbt_data[key]['expense_year']),
            cost_allocation=read_fluid_type(fluid=lbt_data[key]['cost_allocation']),
            description=lbt_data[key]['description'],
            tax_portion=np.array(lbt_data[key]['tax_portion']),
            tax_discount=np.array(lbt_data[key]['tax_discount']),
            final_year=np.array(lbt_data[key]['final_year']),
            utilized_land_area=np.array(lbt_data[key]['utilized_land_area']),
            utilized_building_area=np.array(lbt_data[key]['utilized_building_area']),
            njop_land=np.array(lbt_data[key]['njop_land']),
            njop_building=np.array(lbt_data[key]['njop_building']),
            gross_revenue=np.array(lbt_data[key]['gross_revenue']),
        ) for key in lbt_data.keys()
    ]) if lbt_data is not None else None

    # Defining the data source and the list container for Cost of Sales. Then, them based on their fluid type
    cos_data = cast(_CostOfSalesData | None, data_raw['cost_of_sales'])
    cos_list = ([
        CostOfSales(
            start_year=cos_data[key]['start_year'],
            end_year=cos_data[key]['end_year'],
            cost=np.array(cos_data[key]['cost']),
            expense_year=np.array(cos_data[key]['expense_year']),
            cost_allocation=read_fluid_type(fluid=cos_data[key]['cost_allocation']),
            description=cos_data[key]['description'],
            tax_portion=np.array(cos_data[key]['tax_portion']),
            tax_discount=np.array(cos_data[key]['tax_discount']),
        ) for key in cos_data.keys()
    ]) if cos_data is not None else None

    return capital_cost_list, intangible_list, opex_list, asr_list, lbt_list, cos_list


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
    testing = data.get('testing')
    if testing is None:
        return ValueError("The testing data for this dataset is still in development.")

    else:
        testing_data = _as_testing_data(testing)
        data_test = testing_data[key]
        return data_test


@overload
def load_data(
    dataset_type: str,
    contract_type: Literal['project'] = 'project',
) -> BaseProject: ...


@overload
def load_data(dataset_type: str, contract_type: Literal['cost_recovery']) -> CostRecovery: ...


@overload
def load_data(dataset_type: str, contract_type: Literal['gross_split']) -> GrossSplit: ...


@overload
def load_data(dataset_type: str, contract_type: str) -> BaseProject | CostRecovery | GrossSplit | None: ...


def load_data(
    dataset_type: str,
    contract_type: str = 'project',
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
    data_raw: _ProjectData = _as_project_data(read_json_file(file_name=dataset_type))

    # Assigning the general config and converting them to datetime format.
    project_start_date = datetime.strptime(data_raw['start_date'], '%d/%m/%Y').date()
    project_end_date = datetime.strptime(data_raw['end_date'], '%d/%m/%Y').date()

    oil_onstream_date = assign_onstream_date(data_raw['oil_onstream_date'])
    gas_onstream_date = assign_onstream_date(data_raw['gas_onstream_date'])

    # Assigning the lifting and cost data.
    lifting_tuple = assign_lifting(data_raw)
    capital_list, intangible_list, opex_list, asr_list, lbt_list, cos_list = assign_cost(data_raw)
    capital_tuple = tuple(capital_list) if capital_list is not None else None
    intangible_tuple = tuple(intangible_list) if intangible_list is not None else None
    opex_tuple = tuple(opex_list) if opex_list is not None else None
    asr_tuple = tuple(asr_list) if asr_list is not None else None
    lbt_tuple = tuple(lbt_list) if lbt_list is not None else None
    cos_tuple = tuple(cos_list) if cos_list is not None else None

    if contract_type == 'project':
        return BaseProject(
            start_date=project_start_date,
            end_date=project_end_date,
            approval_year=project_start_date.year,
            oil_onstream_date=oil_onstream_date,  # pyright: ignore[reportArgumentType]  # legacy-nullability
            gas_onstream_date=gas_onstream_date,  # pyright: ignore[reportArgumentType]  # legacy-nullability
            lifting=lifting_tuple,  # pyright: ignore[reportArgumentType]  # legacy-nullability
            capital_cost=capital_tuple,  # pyright: ignore[reportArgumentType]  # legacy-nullability
            intangible_cost=intangible_tuple,  # pyright: ignore[reportArgumentType]  # legacy-nullability
            opex=opex_tuple,  # pyright: ignore[reportArgumentType]  # legacy-nullability
            asr_cost=asr_tuple,  # pyright: ignore[reportArgumentType]  # legacy-nullability
            lbt_cost=lbt_tuple,  # pyright: ignore[reportArgumentType]  # legacy-nullability
            cost_of_sales=cos_tuple,  # pyright: ignore[reportArgumentType]  # legacy-nullability
        )

    elif contract_type == 'cost_recovery':
        cr_config: _CostRecoveryConfig = _as_cost_recovery_config(
            read_json_file(file_name=contract_type)
        )
        return CostRecovery(
            start_date=project_start_date,
            end_date=project_end_date,
            approval_year=project_start_date.year,
            oil_onstream_date=oil_onstream_date,  # pyright: ignore[reportArgumentType]  # legacy-nullability
            gas_onstream_date=gas_onstream_date,  # pyright: ignore[reportArgumentType]  # legacy-nullability
            lifting=lifting_tuple,  # pyright: ignore[reportArgumentType]  # legacy-nullability
            capital_cost=capital_tuple,  # pyright: ignore[reportArgumentType]  # legacy-nullability
            intangible_cost=intangible_tuple,  # pyright: ignore[reportArgumentType]  # legacy-nullability
            opex=opex_tuple,  # pyright: ignore[reportArgumentType]  # legacy-nullability
            asr_cost=asr_tuple,  # pyright: ignore[reportArgumentType]  # legacy-nullability
            lbt_cost=lbt_tuple,  # pyright: ignore[reportArgumentType]  # legacy-nullability
            cost_of_sales=cos_tuple,  # pyright: ignore[reportArgumentType]  # legacy-nullability
            oil_ftp_is_available=cr_config['oil_ftp_is_available'],
            oil_ftp_is_shared=cr_config['oil_ftp_is_shared'],
            oil_ftp_portion=cr_config['oil_ftp_portion'],
            gas_ftp_is_available=cr_config['gas_ftp_is_available'],
            gas_ftp_is_shared=cr_config['gas_ftp_is_shared'],
            gas_ftp_portion=cr_config['gas_ftp_portion'],
            oil_ctr_pretax_share=cr_config['oil_ctr_pretax_share'],
            gas_ctr_pretax_share=cr_config['gas_ctr_pretax_share'],
            oil_ic_rate=cr_config['oil_ic_rate'],
            gas_ic_rate=cr_config['gas_ic_rate'],
            ic_is_available=cr_config['ic_is_available'],
            oil_cr_cap_rate=cr_config['oil_cr_cap_rate'],
            gas_cr_cap_rate=cr_config['gas_cr_cap_rate'],
            oil_dmo_volume_portion=cr_config['oil_dmo_volume_portion'],
            oil_dmo_fee_portion=cr_config['oil_dmo_fee_portion'],
            oil_dmo_holiday_duration=cr_config['oil_dmo_holiday_duration'],
            gas_dmo_volume_portion=cr_config['gas_dmo_volume_portion'],
            gas_dmo_fee_portion=cr_config['gas_dmo_fee_portion'],
            gas_dmo_holiday_duration=cr_config['gas_dmo_holiday_duration']
        )

    elif contract_type == 'gross_split':
        gs_config: _GrossSplitConfig = _as_gross_split_config(
            read_json_file(file_name=contract_type)
        )
        return GrossSplit(
            start_date=project_start_date,
            end_date=project_end_date,
            approval_year=project_start_date.year,
            oil_onstream_date=oil_onstream_date,  # pyright: ignore[reportArgumentType]  # legacy-nullability
            gas_onstream_date=gas_onstream_date,  # pyright: ignore[reportArgumentType]  # legacy-nullability
            lifting=lifting_tuple,  # pyright: ignore[reportArgumentType]  # legacy-nullability
            capital_cost=capital_tuple,  # pyright: ignore[reportArgumentType]  # legacy-nullability
            intangible_cost=intangible_tuple,  # pyright: ignore[reportArgumentType]  # legacy-nullability
            opex=opex_tuple,  # pyright: ignore[reportArgumentType]  # legacy-nullability
            asr_cost=asr_tuple,  # pyright: ignore[reportArgumentType]  # legacy-nullability
            lbt_cost=lbt_tuple,  # pyright: ignore[reportArgumentType]  # legacy-nullability
            cost_of_sales=cos_tuple,  # pyright: ignore[reportArgumentType]  # legacy-nullability
            field_status=gs_config['field_status'],
            field_loc=gs_config['field_loc'],
            res_depth=gs_config['res_depth'],
            infra_avail=gs_config['infra_avail'],
            res_type=gs_config['res_type'],
            api_oil=gs_config['api_oil'],
            domestic_use=gs_config['domestic_use'],
            prod_stage=gs_config['prod_stage'],
            co2_content=gs_config['co2_content'],
            h2s_content=gs_config['h2s_content'],
            oil_dmo_volume_portion=gs_config['oil_dmo_volume_portion'],
            oil_dmo_fee_portion=gs_config['oil_dmo_fee_portion'],
            oil_dmo_holiday_duration=gs_config['oil_dmo_holiday_duration'],
            gas_dmo_volume_portion=gs_config['gas_dmo_volume_portion'],
            gas_dmo_fee_portion=gs_config['gas_dmo_fee_portion'],
            gas_dmo_holiday_duration=gs_config['gas_dmo_holiday_duration']
        )
