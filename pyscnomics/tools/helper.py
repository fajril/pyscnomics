"""
Handles summation operation on two arrays, accounting for different starting years.
"""

from array import array
from collections.abc import Sequence
from datetime import datetime
from functools import wraps
from typing import Callable, ParamSpec, Protocol, TypeAlias, TypeVar

from typing_extensions import Self

import numpy as np
import numpy.typing as npt

from pyscnomics.econ.selection import (
    FluidType,
    # TaxType,
    TaxSplitTypeCR,
    InflationAppliedTo,
    # TaxPaymentMode,
    FTPTaxRegime,
    NPVSelection,
    DiscountingMode,
    DeprMethod,
    OtherRevenue,
    OptimizationTarget,
    OptimizationParameter,
)


_NumericScalar: TypeAlias = (
    np.bool_
    | np.int8
    | np.int16
    | np.int32
    | np.int64
    | np.uint8
    | np.uint16
    | np.uint32
    | np.uint64
    | np.float16
    | np.float32
    | np.float64
)
_NumericArray: TypeAlias = npt.NDArray[_NumericScalar]
_IntArray: TypeAlias = npt.NDArray[np.int64]
_FloatArray: TypeAlias = npt.NDArray[np.float64]
_Array: TypeAlias = _IntArray | _FloatArray
_ArrayMap: TypeAlias = dict[str, _Array]
_ValueMap: TypeAlias = dict[str, _FloatArray]
_NestedValueMap: TypeAlias = dict[str, dict[str, _FloatArray]]
_AdvancedValueMap: TypeAlias = dict[str, dict[str, dict[str, _FloatArray]]]
_NestedIndexMap: TypeAlias = dict[str, dict[str, _IntArray]]
_AdvancedIndexMap: TypeAlias = dict[str, dict[str, dict[str, _IntArray]]]
_Identifier: TypeAlias = Sequence[Sequence[int] | _IntArray]
_P = ParamSpec("_P")
_NumericFunction: TypeAlias = Callable[_P, _NumericArray]
_NumericDecorator: TypeAlias = Callable[[_NumericFunction[_P]], _NumericFunction[_P]]


class _TargetInstance(Protocol):
    cost_allocation: object
    vat_portion: object
    pdri_portion: object


class _Combinable(Protocol):
    def __add__(self, other: object) -> Self: ...


_T = TypeVar("_T", bound=_Combinable)


def _argwhere(condition: object) -> _IntArray:
    """Return integer coordinates for a boolean-compatible condition."""
    boolean_condition = np.asarray(condition, dtype=np.bool_)
    return np.argwhere(boolean_condition).astype(np.int64, copy=False)


def _int_values(values: _IntArray) -> list[int]:
    """Copy a NumPy integer vector into precisely typed Python integers."""
    return [int(value) for value in array("q", values)]


def _equal_positions(values: _IntArray, target: int) -> _IntArray:
    """Return positions whose integer value equals ``target``."""
    return np.fromiter(
        (index for index, value in enumerate(_int_values(values)) if value == target),
        dtype=np.int64,
    )


class TaxInflationException(Exception):
    """Exception to be raised for an incorrect tax and inflation configurations"""

    pass


class CreateArrayException(Exception):
    """Exception to be raised for an incorrect use of function get_array_from_target"""

    pass


class SplitDataException(Exception):
    """Exception to be raised for an incorrect setting of splitting data in PSC transition"""

    pass


def check_input(target_func: _NumericArray, param: object) -> _FloatArray:
    """
    Check and prepare input parameters for subsequent analysis.

    Parameters
    ----------
    target_func: Function
        A target function.

    param : numpy.ndarray or float or int
        Parameter to be checked and prepared for analysis.

    Returns
    -------
    numpy.ndarray
        A properly formatted parameter array for analysis.

    Notes:
    ------
    This function checks the input parameter 'param' and ensures it is properly
    prepared for the subsequent analysis.
    If 'param' is a float or int, it creates an array of the same length as
    'target_func' with all elements set to 'param'.
    If 'param' is a numpy.ndarray, it checks that its length matches the length
    of 'target_func'. If not, it raises an exception.
    """

    if not isinstance(param, (float, int, np.ndarray)):
        param_type = param.__class__
        raise TaxInflationException(
            f"Input parameter must be given as a float, int, or numpy.ndarray, not as a/an {param_type.__qualname__}"
        )

    if isinstance(param, np.ndarray):
        param_arr = np.asarray(param, dtype=np.float64)
        if len(param_arr) != len(target_func):
            target_func_type = target_func.__class__
            raise TaxInflationException(
                f"Unequal length of arrays: {np.ndarray.__qualname__}({param}): ({len(param_arr)}), {target_func_type.__qualname__}({target_func}): ({len(target_func)})."
            )
        return param_arr

    return np.full(len(target_func), param, dtype=np.float64)


def apply_inflation(
    inflation_rate: _NumericArray | float,
) -> _NumericDecorator[_P]:
    """
    Decorator function to apply inflation/escalation to a target function's result.

    Parameters
    ----------
    inflation_rate: np.ndarray or float or int
        The inflation rate to apply. Can be a single value or an array

    Returns
    -------
    Callable
        A decorated function that applies inflation multiplier to a target function's result.

    Notes
    -----
    This decorator is used to apply inflation multiplier to the result of a target function.
    It takes inflation rate as an input argument and returns a decorated function.
    """

    def _decorated(f: _NumericFunction[_P]) -> _NumericFunction[_P]:
        @wraps(f)
        def _wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _NumericArray:
            exponents = np.arange(0, len(f(*args, **kwargs)), 1)
            inflation_rate_arr = check_input(
                target_func=f(*args, **kwargs), param=inflation_rate
            )
            inflation_mult = (1.0 + inflation_rate_arr) ** exponents
            modified_arr = f(*args, **kwargs) * inflation_mult
            return modified_arr

        return _wrapper

    return _decorated


def apply_vat_and_pdri(
    vat_portion: float | int,
    vat_rate: _NumericArray | float | int,
    vat_discount: _NumericArray | float | int,
    pdri_portion: float | int,
    pdri_rate: _NumericArray | float | int,
    pdri_discount: _NumericArray | float | int,
) -> _NumericDecorator[_P]:
    """
    Decorator for applying VAT and PDRI multipliers to a target function's result.

    Parameters
    ----------
    vat_portion : float or int
        The portion of VAT (Value Added Tax) to be applied.
    vat_rate : numpy.ndarray or float or int
        The VAT/PPN rate(s) to apply. Can be a single value or an array.
    vat_discount : numpy.ndarray or float or int
        The VAT discount(s) to apply. Can be a single value or an array.
    pdri_portion : float or int
        The portion of PDRI to be applied.
    pdri_rate : numpy.ndarray or float or int
        The PDRI rate(s) to apply. Can be a single value or an array.
    pdri_discount : numpy.ndarray or float or int
        The PDRI discount(s) to apply. Can be a single value or an array.

    Returns
    -------
    Callable
        A decorated function that applies VAT and PDRI multipliers to a target function's result.

    Notes
    -----
    This decorator is used to apply VAT and PDRI multipliers to the result of a
    target function. It takes several parameters related to VAT and PDRI calculations
    and returns a decorated function.

    The decorated function first calculates VAT and PDRI multipliers based on the
    input parameters. Afterwards, total multiplier is determined by adding the
    VAT multiplier with the PDRI multiplier. The total multiplier is then invoked
    into the result of the target function.
    """

    def _decorated(f: _NumericFunction[_P]) -> _NumericFunction[_P]:
        @wraps(f)
        def _wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _NumericArray:
            def _get_vat():
                vat_portion_arr = np.repeat(vat_portion, len(f(*args, **kwargs)))
                vat_rate_arr = check_input(
                    target_func=f(*args, **kwargs), param=vat_rate
                )
                vat_discount_arr = check_input(
                    target_func=f(*args, **kwargs), param=vat_discount
                )
                vat_multiplier = (
                    vat_portion_arr * vat_rate_arr * (1.0 - vat_discount_arr)
                )
                return vat_multiplier

            def _get_pdri():
                pdri_portion_arr = np.repeat(pdri_portion, len(f(*args, **kwargs)))
                pdri_rate_arr = check_input(
                    target_func=f(*args, **kwargs), param=pdri_rate
                )
                pdri_discount_arr = check_input(
                    target_func=f(*args, **kwargs), param=pdri_discount
                )
                pdri_multiplier = (
                    pdri_portion_arr * pdri_rate_arr * (1.0 - pdri_discount_arr)
                )
                return pdri_multiplier

            total_multiplier = _get_vat() + _get_pdri()
            return f(*args, **kwargs) * (1.0 + total_multiplier)

        return _wrapper

    return _decorated


def apply_lbt(
    lbt_portion: float | int,
    lbt_rate: _NumericArray | float | int,
    lbt_discount: _NumericArray | float | int,
) -> _NumericDecorator[_P]:
    """
    Decorator for applying LBT scheme to a target function's result.

    Parameters
    ----------
    lbt_portion : float or int
        The portion of LBT (Land and Building Tax) to be applied.
    lbt_rate : numpy.ndarray or float or int
        The LBT/PBB rate(s) to apply. Can be a single value or an array.
    lbt_discount : numpy.ndarray or float or int
        The LBT/PBB discount(s) to apply. Can be a single value or an array.

    Returns
    -------
    Callable
        A decorated function that applies LBT/PBB scheme to a target function's result.

    Notes
    -----
    This decorator is used to apply LBT/PBB scheme to the result of a target function.
    It takes several parameters related to LBT/PBB and returns a decorated function.
    """

    def _decorated(f: _NumericFunction[_P]) -> _NumericFunction[_P]:
        @wraps(f)
        def _wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _NumericArray:
            lbt_portion_arr = np.repeat(lbt_portion, len(f(*args, **kwargs)))
            lbt_rate_arr = check_input(target_func=f(*args, **kwargs), param=lbt_rate)
            lbt_discount_arr = check_input(
                target_func=f(*args, **kwargs), param=lbt_discount
            )
            lbt_multiplier = lbt_portion_arr * lbt_rate_arr * (1.0 - lbt_discount_arr)
            return f(*args, **kwargs) * (1.0 + lbt_multiplier)

        return _wrapper

    return _decorated


def apply_pdrd(
    pdrd_portion: float | int,
    pdrd_rate: _NumericArray | float | int,
    pdrd_discount: _NumericArray | float | int,
) -> _NumericDecorator[_P]:
    """
    Decorator for applying PDRD scheme to a target function's result.

    Parameters
    ----------
    pdrd_portion : float or int
        The portion of PDRD to be applied.
    pdrd_rate : numpy.ndarray or float or int
        The PDRD rate(s) to apply. Can be a single value or an array.
    pdrd_discount : numpy.ndarray or float or int
        The PDRD discount(s) to apply. Can be a single value or an array.

    Returns
    -------
    Callable
        A decorated function that applies PDRD scheme to a target function's result.

    Notes
    -----
    This decorator is used to apply PDRD scheme to the result of a target function.
    It takes several parameters related to PDRD and returns a decorated function.
    """

    def _decorated(f: _NumericFunction[_P]) -> _NumericFunction[_P]:
        @wraps(f)
        def _wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _NumericArray:
            pdrd_portion_arr = np.repeat(pdrd_portion, len(f(*args, **kwargs)))
            pdrd_rate_arr = check_input(target_func=f(*args, **kwargs), param=pdrd_rate)
            pdrd_discount_arr = check_input(
                target_func=f(*args, **kwargs), param=pdrd_discount
            )
            pdrd_multiplier = (
                pdrd_portion_arr * pdrd_rate_arr * (1.0 - pdrd_discount_arr)
            )
            return f(*args, **kwargs) * (1.0 + pdrd_multiplier)

        return _wrapper

    return _decorated


def get_identifier(
    target_instances: tuple[_TargetInstance, ...], cost_alloc: FluidType
) -> list[npt.NDArray[np.int64]]:
    """
    Identify the element of instances to add in target data.

    Parameter
    ---------
    target_instances: tuple
        A tuple of target data to be configured.
    cost_alloc: FluidType
        The cost allocation of the target instances.

    Return
    ------
    identifier: list
        An array of indices (as a list) depicting the elements of target array to be added.

    Notes
    -----
    The core calculations are as follows:
    (1) Check each elements of target array; whether it has any identical element(s).
        The determining characteristics were (i) vat_portion, and (ii) pdri_portion,
    (2) Identify the index location where two (or more) elements are similar.
        Each collection of elements constitute a group,
    (3) Carry out multiplication of elements in a group,
    (4) Configure the unique elements in (3),
    (5) Specify the elements of target data to be added.
    """
    # Operations associated with Notes #1 (see the above docstring)
    id_arr_values: list[list[int]] = [
        [0 for _ in range(len(target_instances))] for _ in range(len(target_instances))
    ]
    for i in range(len(target_instances)):
        if target_instances[i].cost_allocation == cost_alloc:
            for j in range(len(target_instances)):
                id_arr_values[i][j] = 1 * (
                    target_instances[i].vat_portion == target_instances[j].vat_portion
                    and target_instances[i].pdri_portion
                    == target_instances[j].pdri_portion
                )

    id_arr = np.asarray(id_arr_values, dtype=np.int64)

    # Operations associated with Notes #2 (see the above docstring)
    group_arr: list[_IntArray] = [
        _equal_positions(id_arr[i : i + 1, :].ravel(), 1)
        for i in range(len(target_instances))
    ]

    # Operations associated with Notes #3 (see the above docstring)
    group_id_values: list[np.int64] = [np.int64(0) for _ in range(len(group_arr))]
    for i in range(len(group_arr)):
        if len(group_arr[i]) == 0:
            group_id_values[i] = np.int64(-1)
        if len(group_arr[i]) > 0:
            group_id_values[i] = np.prod(group_arr[i])

    group_id = np.array(group_id_values)

    # Operations associated with Notes #4 (see the above docstring)
    loc_id = np.unique(group_id)

    # Operations associated with Notes #5 (see the above docstring)
    identifier = [
        _equal_positions(group_id, location)
        for location in _int_values(loc_id)
        if location >= 0
    ]

    return identifier


def get_instances(
    target_instances: tuple[_T, ...], identifier: _Identifier
) -> tuple[_T, ...]:
    """
    Retrieve instances from a target_instances tuple based on an identifier list.

    Parameters
    ----------
    target_instances: tuple
        A tuple containing instances.
    identifier: list
        A list of lists, where each inner list contains indices for target_instances.

    Return
    ------
    tuple
        A tuple of instances retrieved based on the identifier.

    The function processes the identifier list and returns a tuple of instances from
    target_instances. If an inner list in the identifier contains a single index, the
    corresponding instance is added directly. If an inner list contains multiple indices,
    the instances at those indices are summed, and the result is added to the output tuple.

    Example:
    >>> instances = ('A', 'B', 'C', 'D', 'E')
    >>> ids = [[0, 2], [3], [1, 4]]
    >>> get_instances(instances, ids)
    ('AC', 'D', 'BE')
    """

    result: list[_T] = []
    for i in range(len(identifier)):
        if len(identifier[i]) == 1:
            result.append(target_instances[identifier[i][0]])

        elif len(identifier[i]) > 1:
            add = target_instances[identifier[i][0]]
            for j in range(1, len(identifier[i])):
                add = add + target_instances[identifier[i][j]]
                result.append(add)

    return tuple(result)


def get_datetime(ordinal_date: int) -> datetime:
    """
    Convert an ordinal date to a datetime object.

    Parameters
    ----------
    ordinal_date: int
        Ordinal date to be converted.

    Returns
    -------
    datetime
        Datetime object corresponding to the given ordinal date.
    """
    date_time = datetime.fromordinal(
        datetime(1900, 1, 1).toordinal() + ordinal_date - 2
    )
    return date_time


def get_inflation_applied_converter(target: str | None) -> InflationAppliedTo | None:
    """
    Converts a string representing the application of inflation to its
    corresponding enum value from the InflationAppliedTo class.

    Parameters
    ----------
    target: str
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
    attrs = {
        "CAPEX": InflationAppliedTo.CAPEX,
        "OPEX": InflationAppliedTo.OPEX,
        "CAPEX AND OPEX": InflationAppliedTo.CAPEX_AND_OPEX,
    }

    for key in attrs.keys():
        if target == key:
            return attrs[key]

    return None


def get_tax_payment_converter(target: str) -> FTPTaxRegime | None:
    """
    Converts a string representing a tax payment mode to its corresponding
    enum value from the TaxPaymentMode class.

    Parameters
    ----------
    target: str
        The string representation of the tax payment mode.

    Returns
    -------
    Union[TaxPaymentMode, None]
        The corresponding enum value if the target matches one of the predefined payment modes.
        Returns None if no match is found.

    Example
    -------
    >>> result = get_tax_payment_converter("Direct Mode")
    >>> print(result)
    TaxPaymentMode.TAX_DIRECT_MODE
    """
    attrs = {
        "Direct Mode": FTPTaxRegime.DIRECT_MODE,
        "PDJP No.20 Tahun 2017": FTPTaxRegime.PDJP_20_2017,
        "Pre-PDJP No.20 Tahun 2017": FTPTaxRegime.PRE_PDJP_20_2017,
    }

    for key in attrs.keys():
        if target == key:
            return attrs[key]

    return None


def get_tax_regime_converter(target: str) -> FTPTaxRegime | None:
    """
    Converts a string representing a tax regime to its corresponding
    enum value from the FTPTaxRegime class.

    Parameters
    ----------
    target: str
        The string representation of the tax regime.

    Returns
    -------
    Union[FTPTaxRegime, None]
        The corresponding enum value if the target matches one of the predefined tax regimes.
        Returns None if no match is found.

    Example
    -------
    >>> result = get_tax_regime_converter("PDJP No.20 Tahun 2017")
    >>> print(result)
    FTPTaxRegime.PDJP_20_2017
    """
    attrs = {
        "PDJP No.20 Tahun 2017": FTPTaxRegime.PDJP_20_2017,
        "Pre PDJP No.20 Tahun 2017": FTPTaxRegime.PRE_PDJP_20_2017,
    }

    for key in attrs.keys():
        if target == key:
            return attrs[key]

    return None


def get_npv_mode_converter(target: str) -> NPVSelection | None:
    """
    Converts a string representing an NPV mode to its corresponding
    enum value from the NPVSelection class.

    Parameters
    ----------
    target: str
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
    attrs = {
        "SKK Full Cycle Real Terms": NPVSelection.NPV_SKK_REAL_TERMS,
        "SKK Full Cycle Nominal Terms": NPVSelection.NPV_SKK_NOMINAL_TERMS,
        "Full Cycle Real Terms": NPVSelection.NPV_REAL_TERMS,
        "Full Cycle Nominal Terms": NPVSelection.NPV_NOMINAL_TERMS,
        "Point Forward": NPVSelection.NPV_POINT_FORWARD,
    }

    for key in attrs.keys():
        if target == key:
            return attrs[key]

    return None


def get_discounting_mode_converter(target: str) -> DiscountingMode | None:
    """
    Converts a string representing a discounting mode to its corresponding
    enum value from the DiscountingMode class.

    Parameters
    ----------
    target: str
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
    attrs = {
        "End Year": DiscountingMode.END_YEAR,
        "Mid Year": DiscountingMode.MID_YEAR,
    }

    for key in attrs.keys():
        if target == key:
            return attrs[key]

    return None


def get_depreciation_method_converter(target: str) -> DeprMethod | None:
    """
    Converts a string representing a depreciation method to its corresponding
    enum value from the DeprMethod class.

    Parameters
    ----------
    target: str
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
    attrs = {
        "PSC Declining Balance": DeprMethod.PSC_DB,
        "Declining Balance": DeprMethod.DB,
        "Unit Of Production": DeprMethod.UOP,
        "Straight Line": DeprMethod.SL,
    }

    for key in attrs.keys():
        if target == key:
            return attrs[key]

    return None


def get_other_revenue_converter(target: str) -> OtherRevenue | None:
    """
    Convert a string representation of other revenue types to the corresponding enum value.

    Parameters
    ----------
    target: str
        The string representation of the other revenue type.

    Returns
    -------
    OtherRevenue or None
        The enum value corresponding to the provided string representation of other revenue.
        Returns None if the string does not match any known other revenue types.

    Notes
    -----
    - The function uses a mapping of string representations to OtherRevenue enum values.
    - If the target string matches a known other revenue type, the corresponding enum value is returned.
    - If the target string does not match any known types, the function returns None.
    """
    attrs = {
        "Addition to Oil Revenue": OtherRevenue.ADDITION_TO_OIL_REVENUE,
        "Addition to Gas Revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
        "Reduction to Oil OPEX": OtherRevenue.REDUCTION_TO_OIL_OPEX,
        "Reduction to Gas OPEX": OtherRevenue.REDUCTION_TO_GAS_OPEX,
    }

    for key in attrs.keys():
        if target == key:
            return attrs[key]

    return None


def get_array_from_target(target: _ArrayMap, project_years: _Array) -> _FloatArray:
    """
    Create a new NumPy array of tax rates aligned with project years based on input data.

    Parameters
    ----------
    target: dict
        A dictionary containing 'year' and 'rate' arrays.
    project_years: np.ndarray
        1D NumPy array representing the project years.

    Returns
    -------
    np.ndarray
        A new NumPy array of 'target' aligned with project_years.

    Raises
    ------
    CreateArrayException
        If the lengths of 'year' and 'rate' arrays in 'target' are not equal.
    """
    year_source = np.asarray(target["year"], dtype=np.float64)
    rate_source = np.asarray(target["rate"], dtype=np.float64)
    years = year_source[~np.isnan(year_source)].astype(np.int64, copy=False)
    rates = rate_source[~np.isnan(rate_source)]
    project_years_int = np.asarray(project_years, dtype=np.int64)

    # Raise error for unequal length of "year" and "rate" in "multi_adj"
    if len(years) != len(rates):
        raise CreateArrayException(
            f"Unequal number of arrays: year: {len(years)}, tax_rate: {len(rates)}."
        )

    year_values = _int_values(years)
    project_year_values = _int_values(project_years_int)
    min_year = min(min(year_values), min(project_year_values))
    max_year = max(max(year_values), max(project_year_values))
    all_years = np.arange(min_year, max_year + 1, dtype=np.int64)

    # Select the most recent configured rate for each year. Values before the
    # first configured year use the first rate; values after the last use the last.
    rate_indices = np.searchsorted(years, all_years, side="right") - 1
    rate_indices = np.clip(rate_indices, 0, len(rates) - 1)
    expanded_rates = rates[rate_indices]
    project_start = min(project_year_values) - min_year
    project_stop = max(project_year_values) - min_year + 1
    return np.asarray(expanded_rates[project_start:project_stop], dtype=np.float64)


def get_lifting_data_split_simple(
    target_attr: _ValueMap,
    is_target_attr_volume: bool,
    prod_year_init: dict[str, _IntArray],
    end_date_contract_1: datetime,
    start_date_contract_2: datetime,
) -> _NestedValueMap:
    """
    Split target attribute data into two corresponding Production Sharing Contracts (PSC).

    Parameters
    ----------
    target_attr: dict
        Dictionary containing the original target attribute data.
    is_target_attr_volume: bool
        Flag indicating whether the target attribute represents volume data.
    prod_year_init: dict
        Dictionary containing the production years for each contract.
    end_date_contract_1: datetime
        End date of the first contract.
    start_date_contract_2: datetime
        Start date of the second contract.

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        A modified dictionary with the original target attribute data split into two
        corresponding PSC contracts

    Raises
    ------
    SplitDataException
        If the provided production data is insufficient for splitting into two PSC contracts.

    Notes
    -----
    (1) The function checks the contract transition period and adjusts the data accordingly,
    (2) If the end year of the first contract is the same as the start year of the second
        contract, the target data is split based on the transition year, and adjustments
        are made for volume data.
    (3) If the end year of the first contract is different from the start year of the second
        contract, the target data is split based on the respective transition years.
    """
    psc_regimes = ["PSC 1", "PSC 2"]

    # End year of the first contract is the same as the start year of the second contract
    if end_date_contract_1.year == start_date_contract_2.year:
        # Specify the multiplier
        days_diff = end_date_contract_1 - datetime(
            day=1, month=1, year=end_date_contract_1.year
        )
        days_delta = datetime(
            day=31, month=12, year=end_date_contract_1.year
        ) - datetime(day=1, month=1, year=end_date_contract_1.year)
        multiplier = (days_diff.days + 1) / (days_delta.days + 2)

        # Identify the index location of data: (1) at the transition year,
        # (2) before the transition year, and (3) after the transition year
        id_transition_init: _NestedIndexMap = {
            ws: {
                "exact": _equal_positions(prod_year_init[ws], end_date_contract_1.year),
                "before": _argwhere(
                    prod_year_init[ws] < end_date_contract_1.year
                ).ravel(),
                "after": _argwhere(
                    prod_year_init[ws] > end_date_contract_1.year
                ).ravel(),
            }
            for ws in prod_year_init.keys()
        }

        # Identify the indices of the data that falls within PSC 1 and PSC 2, respectively
        id_transition: _NestedIndexMap = {
            ws: {
                psc_regimes[0]: np.concatenate(
                    (id_transition_init[ws]["before"], id_transition_init[ws]["exact"])
                ),
                psc_regimes[1]: np.concatenate(
                    (id_transition_init[ws]["exact"], id_transition_init[ws]["after"])
                ),
            }
            for ws in prod_year_init.keys()
        }

        # Raise exception if the data is insuffiecient to be splitted into
        # two corresponding psc regimes
        if any(
            [
                len(id_transition[ws][psc]) == 0
                for ws in prod_year_init.keys()
                for psc in psc_regimes
            ]
        ):
            raise SplitDataException(
                "Cannot split data for PSC transition. Please check the production data. The provided production data is not sufficient to perform splitting into two PSC contracts."
            )

        # Split the original target data into two corresponding PSC contracts
        target_attr_modified = {
            ws: {
                psc: target_attr[ws][id_transition[ws][psc]].copy()
                for psc in psc_regimes
            }
            for ws in prod_year_init.keys()
        }

        # Identify the location of transition year in "id_transition"
        if is_target_attr_volume is True:
            id_exact = {
                ws: {
                    psc: None
                    if len(id_transition_init[ws]["exact"]) == 0
                    else np.array(
                        [
                            _equal_positions(id_transition[ws][psc], transition_index)
                            for transition_index in _int_values(
                                id_transition_init[ws]["exact"]
                            )
                        ]
                    ).ravel()
                    for psc in psc_regimes
                }
                for ws in prod_year_init.keys()
            }

            # Adjust the value of target_data_modified at the transition year
            multipliers = [multiplier, (1.0 - multiplier)]
            for ws in prod_year_init.keys():
                for i, psc in enumerate(psc_regimes):
                    if id_exact[ws][psc] is None:
                        target_attr_modified[ws][psc] = target_attr_modified[ws][
                            psc
                        ].copy()
                    else:
                        target_attr_modified[ws][psc][id_exact[ws][psc]] *= multipliers[
                            i
                        ]

    # End year of the first contract is different from the start year of the second contract
    else:
        # Identify the index location of the transition year
        id_transition = {
            ws: {
                psc_regimes[0]: (
                    _argwhere(prod_year_init[ws] <= end_date_contract_1.year).ravel()
                ),
                psc_regimes[1]: (
                    _argwhere(prod_year_init[ws] >= start_date_contract_2.year).ravel()
                ),
            }
            for ws in prod_year_init.keys()
        }

        # Raise exception if the data is insuffiecient to be splitted into
        # two corresponding psc regimes
        if any(
            [
                len(id_transition[ws][psc]) == 0
                for ws in prod_year_init.keys()
                for psc in psc_regimes
            ]
        ):
            raise SplitDataException(
                "Cannot split data for PSC transition. Please check the production data. The provided production data is not sufficient to perform splitting into two PSC contracts."
            )

        # Split the original target data into two corresponding PSC contracts
        target_attr_modified = {
            ws: {
                psc: target_attr[ws][id_transition[ws][psc]].copy()
                for psc in psc_regimes
            }
            for ws in prod_year_init.keys()
        }

    return target_attr_modified


def get_lifting_data_split_advanced(
    target_attr: _NestedValueMap,
    is_target_attr_volume: bool,
    prod_year_init: dict[str, _IntArray],
    end_date_contract_1: datetime,
    start_date_contract_2: datetime,
    gsa_number: int,
) -> _AdvancedValueMap:
    """
    Split target attribute data into two corresponding Production Sharing Contracts (PSC).

    Parameters
    ----------
    target_attr: dict
        Nested dictionary containing the original target attribute data for each production year.
    is_target_attr_volume: bool
        Flag indicating whether the target attribute represents volume data.
    prod_year_init: dict
        Dictionary containing the production years for each contract.
    end_date_contract_1: datetime
        End date of the first contract.
    start_date_contract_2: datetime
        Start date of the second contract.
    gsa_number: int
        The number of active GSA.

    Returns
    -------
    Dict[str, Dict[str, Dict[str, np.ndarray]]]
        A modified nested dictionary with the original target attribute data split into two
        corresponding PSC contracts.

    Raises
    ------
    SplitDataException
        If the provided production data is insufficient for splitting into two PSC contracts.

    Notes
    -----
    (1) The function checks the contract transition period and adjusts the data accordingly,
    (2) If the end year of the first contract is the same as the start year of the second
        contract, the target data is split based on the transition year, and adjustments are
        made for volume data.
    (3) If the end year of the first contract is different from the start year of the second
        contract, the target data is split based on the respective transition years.
    """
    psc_regimes = ["PSC 1", "PSC 2"]
    gsa_variables = ["GSA {0}".format(gsa + 1) for gsa in range(gsa_number)]

    # End year of the first contract is the same as the start year of the second contract
    if end_date_contract_1.year == start_date_contract_2.year:
        # Specify the multiplier
        days_diff = end_date_contract_1 - datetime(
            day=1, month=1, year=end_date_contract_1.year
        )
        days_delta = datetime(
            day=31, month=12, year=end_date_contract_1.year
        ) - datetime(day=1, month=1, year=end_date_contract_1.year)
        multiplier = (days_diff.days + 1) / (days_delta.days + 2)

        # Identify the index location of data: (1) at the transition year,
        # (2) before the transition year, and (3) after the transition year
        id_transition_init: _NestedIndexMap = {
            ws: {
                "exact": _equal_positions(prod_year_init[ws], end_date_contract_1.year),
                "before": _argwhere(
                    prod_year_init[ws] < end_date_contract_1.year
                ).ravel(),
                "after": _argwhere(
                    prod_year_init[ws] > end_date_contract_1.year
                ).ravel(),
            }
            for ws in prod_year_init.keys()
        }

        # Identify the indices of the data that falls within PSC 1 and PSC 2, respectively
        id_transition: _AdvancedIndexMap = {
            ws: {
                gsa: {
                    psc_regimes[0]: np.concatenate(
                        (
                            id_transition_init[ws]["before"],
                            id_transition_init[ws]["exact"],
                        )
                    ),
                    psc_regimes[1]: np.concatenate(
                        (
                            id_transition_init[ws]["exact"],
                            id_transition_init[ws]["after"],
                        )
                    ),
                }
                for gsa in gsa_variables
            }
            for ws in prod_year_init.keys()
        }

        # Raise exception if the data is insuffiecient to be splitted into
        # two corresponding psc regimes
        if any(
            [
                len(id_transition[ws][gsa][psc]) == 0
                for ws in prod_year_init.keys()
                for gsa in gsa_variables
                for psc in psc_regimes
            ]
        ):
            raise SplitDataException(
                "Cannot split data for PSC transition. Please check the production data. The provided production data is not sufficient to perform splitting into two PSC contracts."
            )

        # Split the original target data into two corresponding PSC contracts
        target_attr_modified = {
            ws: {
                gsa: {
                    psc: target_attr[ws][gsa][id_transition[ws][gsa][psc]].copy()
                    for psc in psc_regimes
                }
                for gsa in gsa_variables
            }
            for ws in prod_year_init.keys()
        }

        # Identify the location of transition year in "id_transition"
        if is_target_attr_volume is True:
            id_exact = {
                ws: {
                    gsa: {
                        psc: None
                        if len(id_transition_init[ws]["exact"]) == 0
                        else np.array(
                            [
                                _equal_positions(
                                    id_transition[ws][gsa][psc], transition_index
                                )
                                for transition_index in _int_values(
                                    id_transition_init[ws]["exact"]
                                )
                            ]
                        ).ravel()
                        for psc in psc_regimes
                    }
                    for gsa in gsa_variables
                }
                for ws in prod_year_init.keys()
            }

            # Adjust the value of target_data_modified at the transition year
            multipliers = [multiplier, (1.0 - multiplier)]
            for ws in prod_year_init.keys():
                for gsa in gsa_variables:
                    for i, psc in enumerate(psc_regimes):
                        if id_exact[ws][gsa][psc] is None:
                            target_attr_modified[ws][gsa][psc] = target_attr_modified[
                                ws
                            ][gsa][psc].copy()
                        else:
                            target_attr_modified[ws][gsa][psc][
                                id_exact[ws][gsa][psc]
                            ] *= multipliers[i]

    # End year of the first contract is different from the start year of the second contract
    else:
        # Identify the index location of the transition year
        id_transition = {
            ws: {
                gsa: {
                    psc_regimes[0]: _argwhere(
                        prod_year_init[ws] <= end_date_contract_1.year
                    ).ravel(),
                    psc_regimes[1]: _argwhere(
                        prod_year_init[ws] >= start_date_contract_2.year
                    ).ravel(),
                }
                for gsa in gsa_variables
            }
            for ws in prod_year_init.keys()
        }

        # Raise exception if the data is insuffiecient to be splitted into
        # two corresponding psc regimes
        if any(
            [
                len(id_transition[ws][gsa][psc]) == 0
                for ws in prod_year_init.keys()
                for gsa in gsa_variables
                for psc in psc_regimes
            ]
        ):
            raise SplitDataException(
                "Cannot split data for PSC transition. Please check the production data. The provided production data is not sufficient to perform splitting into two PSC contracts."
            )

        # Split the original target data into two corresponding PSC contracts
        target_attr_modified = {
            ws: {
                gsa: {
                    psc: target_attr[ws][gsa][id_transition[ws][gsa][psc]].copy()
                    for psc in psc_regimes
                }
                for gsa in gsa_variables
            }
            for ws in prod_year_init.keys()
        }

    return target_attr_modified


def get_cost_data_split(
    target_attr: list[object] | _FloatArray,
    is_target_attr_adjusted: bool,
    expense_year_init: _IntArray,
    end_date_contract_1: datetime,
    start_date_contract_2: datetime,
) -> dict[str, _FloatArray]:
    """
    Split cost data based on the transition between two Production Sharing Contracts (PSCs).

    Parameters
    ----------
    target_attr: list, np.ndarray
        The target attribute data to be split.
    is_target_attr_adjusted: bool
        Indicates if the target attribute will be adjusted.
    expense_year_init: np.ndarray
        Array of years corresponding to the cost data.
    end_date_contract_1: datetime
        End date of the first contract.
    start_date_contract_2: datetime
        Start date of the second contract.

    Returns
    -------
    dict
        A dictionary containing split target attribute data for PSC 1 and PSC 2.

    Raises
    ------
    SplitDataException
        If the provided production data is insufficient for splitting into two PSC contracts.

    Notes
    -----
    (1) The function checks the contract transition period and adjusts the data accordingly,
    (2) If the end year of the first contract is the same as the start year of the second
        contract, the target data is split based on the transition year, and adjustments
        are made for volume data.
    (3) If the end year of the first contract is different from the start year of the second
        contract, the target data is split based on the respective transition years.
    """
    if isinstance(target_attr, list):
        target_attr = np.array(target_attr)

    psc_regimes = ["PSC 1", "PSC 2"]

    # End year of the first contract is the same as the start year of the second contract
    if end_date_contract_1.year == start_date_contract_2.year:
        # Specify the multiplier
        days_diff = end_date_contract_1 - datetime(
            day=1, month=1, year=end_date_contract_1.year
        )
        days_delta = datetime(
            day=31, month=12, year=end_date_contract_1.year
        ) - datetime(day=1, month=1, year=end_date_contract_1.year)
        multiplier = (days_diff.days + 1) / (days_delta.days + 2)

        # Identify the index location of data: (1) at the transition year,
        # (2) before the transition year, and (3) after the transition year
        id_transition_init = {
            "exact": _equal_positions(expense_year_init, end_date_contract_1.year),
            "before": _argwhere(expense_year_init < end_date_contract_1.year).ravel(),
            "after": _argwhere(expense_year_init > end_date_contract_1.year).ravel(),
        }

        # Identify the indices of the data that falls within PSC 1 and PSC 2, respectively
        id_transition = {
            psc_regimes[0]: np.concatenate(
                (id_transition_init["before"], id_transition_init["exact"])
            ),
            psc_regimes[1]: np.concatenate(
                (id_transition_init["exact"], id_transition_init["after"])
            ),
        }

        # Raise exception if the data is insuffiecient to be splitted into
        # two corresponding psc regimes
        if any([len(id_transition[psc]) == 0 for psc in psc_regimes]):
            raise SplitDataException(
                "Cannot split data for PSC transition. Please check cost data. The provided cost data is not sufficient to perform splitting into two PSC contracts."
            )

        # Split the original target data into two corresponding PSC contracts
        target_attr_modified = {
            psc: target_attr[id_transition[psc]].copy() for psc in psc_regimes
        }

        # Identify the location of transition year in "id_transition"
        if is_target_attr_adjusted is True:
            id_exact = {
                psc: None
                if len(id_transition_init["exact"]) == 0
                else np.array(
                    [
                        _equal_positions(id_transition[psc], transition_index)
                        for transition_index in _int_values(id_transition_init["exact"])
                    ]
                ).ravel()
                for psc in psc_regimes
            }

            # Adjust the value of target_data_modified at the transition year
            multipliers = [multiplier, (1.0 - multiplier)]
            for i, psc in enumerate(psc_regimes):
                if id_exact[psc] is None:
                    target_attr_modified[psc] = target_attr_modified[psc].copy()
                else:
                    target_attr_modified[psc][id_exact[psc]] *= multipliers[i]

    # End year of the first contract is different from the start year of the second contract
    else:
        # Identify the index location of the transition year
        id_transition = {
            psc_regimes[0]: _argwhere(
                expense_year_init <= end_date_contract_1.year
            ).ravel(),
            psc_regimes[1]: _argwhere(
                expense_year_init >= start_date_contract_2.year
            ).ravel(),
        }

        # Raise exception if the data is insuffiecient to be splitted into
        # two corresponding psc regimes
        if any([len(id_transition[psc]) == 0 for psc in psc_regimes]):
            raise SplitDataException(
                "Cannot split data for PSC transition. Please check cost data. The provided cost data is not sufficient to perform splitting into two PSC contracts."
            )

        # Split the original target data into two corresponding PSC contracts
        target_attr_modified = {
            psc: target_attr[id_transition[psc]].copy() for psc in psc_regimes
        }

    return target_attr_modified


def get_to_list_converter(target_param: _ArrayMap) -> dict[str, list[object]]:
    """
    Converts the values of a dictionary from NumPy arrays to Python lists.

    Parameters
    ----------
    target_param: dict
        A dictionary with NumPy arrays as values.

    Returns
    -------
    dict
        A new dictionary with the same keys as `target_param`, but the values are
        converted to Python lists using the `tolist()` method of NumPy arrays.

    Example:
    >>> input_dict = {'a': np.array([1, 2, 3]), 'b': np.array([4, 5, 6])}
    >>> result = get_to_list_converter(input_dict)
    >>> print(result)
    {'a': [1, 2, 3], 'b': [4, 5, 6]}
    """
    return {key: target_param[key].tolist() for key in target_param.keys()}


def get_fluidtype_converter(target: str) -> FluidType | None:
    """
    Get the corresponding FluidType enum for a given target fluid.

    Parameters
    ----------
    target: str
        The target fluid for which the FluidType enum is to be retrieved.

    Returns
    -------
    FluidType or None
        The FluidType enum corresponding to the provided target fluid.
        Returns None if the target fluid is not found in the predefined attributes.

    Examples
    --------
    >>> get_fluidtype_converter("Oil")
    <FluidType.OIL: 'Oil'>

    >>> get_fluidtype_converter("CO2")
    <FluidType.CO2: 'CO2'>

    >>> get_fluidtype_converter("Water")
    None
    """
    attrs = {
        "Oil": FluidType.OIL,
        "Gas": FluidType.GAS,
        "LPG propane": FluidType.GAS,
        "LPG butane": FluidType.GAS,
        "Sulfur": FluidType.SULFUR,
        "Electricity": FluidType.ELECTRICITY,
        "CO2": FluidType.CO2,
    }

    for key in attrs.keys():
        if target == key:
            return attrs[key]

    return None


def get_boolean_converter(target: str) -> bool | None:
    """
    Get the boolean value corresponding to the provided target.

    Parameters
    ----------
    target: str
        The target string representing a boolean value ("Yes" or "No").

    Returns
    -------
    bool or None
        The boolean value corresponding to the provided target.
        Returns True if target is "Yes", False if target is "No".
        Returns None if the target is neither "Yes" nor "No".

    Examples
    --------
    >>> get_boolean_converter("Yes")
    True

    >>> get_boolean_converter("No")
    False

    >>> get_boolean_converter("Maybe")
    None
    """
    attrs = {"Yes": True, "No": False}

    for key in attrs.keys():
        if target == key:
            return attrs[key]

    return None


def get_split_type_converter(target: str) -> TaxSplitTypeCR | None:
    """
    Converts a string representing a tax split type to its corresponding
    enum value from the TaxSplitTypeCR class.

    Parameters
    ----------
    target: str
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
    attrs = {
        "Conventional": TaxSplitTypeCR.CONVENTIONAL,
        "RC Split": TaxSplitTypeCR.R2C,
        "ICP Sliding Scale": TaxSplitTypeCR.SLIDING_SCALE,
    }

    for key in attrs.keys():
        if target == key:
            return attrs[key]

    return None


def get_optimization_target_converter(target: str) -> OptimizationTarget | None:
    """
    Converts a string representation of an optimization target to the corresponding
    OptimizationTarget enum value.

    Parameters
    ----------
    target: str
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
    attrs = {
        "NPV": OptimizationTarget.NPV,
        "IRR": OptimizationTarget.IRR,
        "PI": OptimizationTarget.PI,
    }

    for key in attrs.keys():
        if target == key:
            return attrs[key]

    return None


def get_optimization_parameter_converter(target: str) -> OptimizationParameter | None:
    """
    Converts a string representation of an optimization parameter to the corresponding
    OptimizationParameter enum value.

    Parameters
    ----------
    target: str
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
    attrs = {
        "Oil Contractor Pre Tax": OptimizationParameter.OIL_CTR_PRETAX,
        "Gas Contractor Pre Tax": OptimizationParameter.GAS_CTR_PRETAX,
        "Oil FTP Portion": OptimizationParameter.OIL_FTP_PORTION,
        "Gas FTP Portion": OptimizationParameter.GAS_FTP_PORTION,
        "Oil IC": OptimizationParameter.OIL_IC,
        "Gas IC": OptimizationParameter.GAS_IC,
        "Oil DMO Fee": OptimizationParameter.OIL_DMO_FEE,
        "Gas DMO Fee": OptimizationParameter.GAS_DMO_FEE,
        "VAT Rate": OptimizationParameter.VAT_RATE,
        "Effective Tax Rate": OptimizationParameter.EFFECTIVE_TAX_RATE,
        "Ministerial Discretion": OptimizationParameter.MINISTERIAL_DISCRETION,
        "Depreciation Acceleration": OptimizationParameter.DEPRECIATION_ACCELERATION,
        "VAT Discount": OptimizationParameter.VAT_DISCOUNT,
        "LBT Discount": OptimizationParameter.LBT_DISCOUNT,
    }

    for key in attrs.keys():
        if target == key:
            return attrs[key]

    return None


def summarizer(
    array1: _Array,
    array2: _Array,
    startYear1: int,
    startYear2: int,
    endYear1: int | None = None,
    endYear2: int | None = None,
) -> tuple[_Array, int, int]:
    """
    Perform summation operation on two arrays, accounting for different starting years.

    Parameters
    ----------
    array1 : np.ndarray
        The first input array for summation.
    array2 : np.ndarray
        The second input array for summation.
    startYear1 : int
        The starting year of data in array1.
    startYear2 : int
        The starting year of data in array2.
    endYear1: int (optional)
        The end year of data in array1.
    endYear2: int (optional)
        The end year of data in array1.

    Returns
    -------
    tuple
        An array containing the result of element-wise summation of array1 and array2,
        start_year and end_year.
        The years are aligned according to the starting year of each input array.

    Notes
    -----
    This function performs element-wise summation on array1 and array2, taking into account
    their respective starting years. The result array will span from the minimum starting
    year to the maximum ending year of the input arrays.

    """

    # Configure minimum start_year
    start_year = min(startYear1, startYear2)

    # Configure minimum end_year
    if endYear1 is None:
        endYear1 = startYear1 + len(array1) - 1

    else:
        array1.resize(endYear1 - start_year + 1, refcheck=False)

    if endYear2 is None:
        endYear2 = startYear2 + len(array2) - 1

    else:
        array2.resize(endYear2 - start_year + 1, refcheck=False)

    end_year = max(endYear1, endYear2)

    # Containers
    years = np.arange(start_year, end_year + 1, 1)
    res_arr1 = np.zeros_like(years)
    res_arr2 = np.zeros_like(years)

    # Indices where the elements of array1 are not aligned with array years, then update the original array
    indices1 = _argwhere(
        np.logical_and(startYear1 <= years, years <= endYear1)
    ).flatten()
    res_arr1[indices1] = array1

    # Indices where the elements of array2 are not aligned with array years, then update the original array
    indices2 = _argwhere(
        np.logical_and(startYear2 <= years, years <= endYear2)
    ).flatten()
    res_arr2[indices2] = array2

    return res_arr1 + res_arr2, start_year, end_year


def sum_remainder(start_year: int, end_year: int, arr: _Array) -> np.int64 | np.float64:
    """
    Calculate the sum of elements in the 'arr' numpy array that occur after a specified duration.

    Parameters
    ----------
    start_year : int
        The starting year for the duration; usually pis_year.
    end_year : int
        The ending year for the duration; usually end of contract.
    arr : np.ndarray
        The numpy array containing the elements to be summed.

    Returns
    -------
    float
        The sum of elements in 'arr' that occur after the specified duration from
        'start_year' to 'end_year' (inclusive).

    Notes
    -----
    This function calculates the duration between 'start_year' and 'end_year' (inclusive),
    and then returns the sum of elements in 'arr' that are positioned after this duration.
    """
    duration = end_year - start_year + 1

    return np.sum(arr[duration:])
