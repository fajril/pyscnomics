"""
Handles summation operation on two arrays, accounting for different starting years.
"""

import numpy as np
from datetime import datetime
from functools import wraps
from pyscnomics.econ.selection import FluidType, TaxType


class TaxInflationException(Exception):
    """ Exception to be raised for an incorrect tax and inflation configurations """

    pass


def check_input(target_func, param: np.ndarray | float | int) -> np.ndarray:
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
    if isinstance(param, np.ndarray):
        if len(param) != len(target_func):
            raise TaxInflationException(
                f"Unequal length of arrays: "
                f"{param.__class__.__qualname__}({param}): ({len(param)}), "
                f"{target_func.__class__.__qualname__}({target_func}): ({len(target_func)})."
            )
        param_arr = param

    if isinstance(param, (float, int)):
        param_arr = np.repeat(param, len(target_func))

    if not isinstance(param, (float, int, np.ndarray)):
        raise TaxInflationException(
            f"Input parameter must be of datatype np.ndarray, int, or float. "
            f"{param} is of datatype ({param.__class__.__qualname__})."
        )

    return param_arr


def get_cost_adjustment_by_tax(
    start_year: int,
    cost: np.ndarray,
    expense_year: np.ndarray,
    project_years: np.ndarray,
    tax_portion: np.ndarray,
    tax_rate: np.ndarray | float,
    tax_discount: float,
) -> np.ndarray:
    """
    Adjust cost based on tax scheme over the prescribed time period.

    Parameters
    ----------
    start_year : int
        The start year of the project.
    cost : np.ndarray
        An array of costs to be adjusted.
    expense_year : np.ndarray
        An array specifying the expense years for each cost element.
    project_years : np.ndarray
        An array of project years.
    tax_portion: np.ndarray
        The portion of 'cost' that is subject to tax.
        Must be an array of length equals to the length of 'cost' array.
    tax_rate: np.ndarray | float
        The tax rate to apply. Can be a single value or an array.
    tax_discount: float
        The tax discount to apply.

    Returns
    -------
    cost_adjusted: np.ndarray
        An array of adjusted costs after applying tax scheme.

    Notes
    -----
    The core operations are as follows:
    (1) Create 'tax_rate_arr' by calling 'check_input()' function. If 'tax_rate'
        is given as an array, the length of the array must be consistent with
        the duration of the project. If it is given as a single value, then create
        an array of length equal to the project duration with all elements set equal
        to the single value,
    (2) Identify the index location of 'tax_rate' based on the associated 'expense_year',
    (3) Calculate the multiplier:
        mult = 1.0 + tax_portion * tax_rate_arr[tax_id] * (1.0 - tax_discount),
    (4) Apply cost adjustment = cost * mult.
    """
    tax_rate_arr = check_input(target_func=project_years, param=tax_rate)
    tax_rate_id = (expense_year - start_year).astype("int")

    return cost * (
        1.0 + tax_portion * tax_rate_arr[tax_rate_id] * (1.0 - tax_discount)
    )


def apply_cost_adjustment(
    start_year: int,
    end_year: int,
    cost: np.ndarray,
    expense_year: np.ndarray,
    project_years: np.ndarray,
    year_ref: int,
    tax_type: TaxType,
    vat_portion: np.ndarray,
    vat_rate: np.ndarray | float,
    vat_discount: float,
    lbt_portion: np.ndarray,
    lbt_rate: np.ndarray | float,
    lbt_discount: float,
    inflation_rate: np.ndarray | float,
) -> np.ndarray:
    """
    Adjusts cost based on inflation and tax scheme over the specified time period.

    Parameters
    ----------
    start_year : int
        The start year of the project.
    end_year : int
        The end year of the project.
    cost : np.ndarray
        An array of costs to be adjusted.
    expense_year : np.ndarray
        An array specifying the expense years for each cost element.
    project_years : np.ndarray
        An array of project years.
    year_ref : int
        The reference year for inflation calculation.
    tax_type: TaxType
        The type of tax used for calculation.
        The options are TaxType.VAT and TaxType.LBT
    vat_portion: np.ndarray
        The portion of 'cost' that is subject to VAT.
        Must be an array of length equals to the length of 'cost' array.
    vat_rate: np.ndarray | float
        The VAT rate to apply. Can be a single value or an array.
    vat_discount: float
        The VAT discount to apply.
    lbt_portion: np.ndarray
        The portion of 'cost' that is subject to LBT.
        Must be an array of length equals to the length of 'cost' array.
    lbt_rate: np.ndarray | float
        The LBT rate to apply. Can be a single value or an array.
    lbt_discount: float
        The LBT discount to apply.
    inflation_rate : np.ndarray | float
        The inflation rate to apply. Can be a single value or an array.

    Returns
    -------
    cost_adjusted: np.ndarray
        An array of adjusted costs after applying inflation and tax scheme.

    Notes
    -----
    The core operations are as follows:
    (1) Apply cost adjustment due tax scheme by calling function get_cost_adjustment_by_tax().
    (2) Check 'inflation_rate' whether it is provided as an array or as a single value.
        If it is given as an array, the length of the array must be consistent with the
        duration of the project. If it is given as a single value, then create an array
        of length equal to project duration with all elements set equal to the single
        value.
    (3) Parameter 'id_year_ref' identify the index location of 'year_ref' in array
        'project_years'. The result is then added by unity. This parameter sets up
        the first index to slice the 'inflation_rate_arr',
    (4) Parameter 'id_rate_arr' configure the index location of 'expense_year' in array
        'project_years' based on the associated 'year_ref'. The result is then added by unity.
        This parameter sets up the second index to slice the 'inflation_rate_arr',
    (5) Slice 'inflation_rate_arr' according to 'id_year_ref' and 'id_rate_arr'. Add the results
        by unity, then multiple the associated elements.
    (5) Cost adjustment is undertaken by multiplication: 'cost_adjusted_by_tax' * 'mult'.
    """
    # Cost adjustment due to tax
    if tax_type == TaxType.VAT:
        cost_adjusted = get_cost_adjustment_by_tax(
            start_year=start_year,
            cost=cost,
            expense_year=expense_year,
            project_years=project_years,
            tax_portion=vat_portion,
            tax_rate=vat_rate,
            tax_discount=vat_discount,
        )

    if tax_type == TaxType.LBT:
        cost_adjusted = get_cost_adjustment_by_tax(
            start_year=start_year,
            cost=cost,
            expense_year=expense_year,
            project_years=project_years,
            tax_portion=lbt_portion,
            tax_rate=lbt_rate,
            tax_discount=lbt_discount,
        )

    # Cost adjustment due to inflation
    if year_ref < start_year:
        raise TaxInflationException(
            f"year_ref ({year_ref}) is before start_year of the project ({start_year})."
        )

    if year_ref > end_year:
        raise TaxInflationException(
            f"year_ref ({year_ref}) is after end_year of the project ({end_year})."
        )

    inflation_rate_arr = check_input(target_func=project_years, param=inflation_rate)
    id_year_ref = int(np.argwhere(year_ref == project_years).ravel()[0] + 1)
    id_rate_arr = ((expense_year - year_ref) + (year_ref - start_year) + 1).astype("int")
    mult = np.array([np.prod(1.0 + inflation_rate_arr[id_year_ref:i]) for i in id_rate_arr])

    return cost_adjusted * mult


def apply_inflation(inflation_rate: np.ndarray | float) -> callable:
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

    def _decorated(f):
        @wraps(f)
        def _wrapper(*args, **kwargs):
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
    vat_rate: np.ndarray | float | int,
    vat_discount: np.ndarray | float | int,
    pdri_portion: float | int,
    pdri_rate: np.ndarray | float | int,
    pdri_discount: np.ndarray | float | int,
) -> callable:
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

    def _decorated(f):
        @wraps(f)
        def _wrapper(*args, **kwargs):
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
    lbt_rate: np.ndarray | float | int,
    lbt_discount: np.ndarray | float | int,
) -> callable:
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

    def _decorated(f):
        @wraps(f)
        def _wrapper(*args, **kwargs):
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
    pdrd_rate: np.ndarray | float | int,
    pdrd_discount: np.ndarray | float | int,
) -> callable:
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

    def _decorated(f):
        @wraps(f)
        def _wrapper(*args, **kwargs):
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


def get_identifier(target_instances: tuple, cost_alloc: FluidType) -> list:
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
    id_arr = [
        [0 for _ in range(len(target_instances))] for _ in range(len(target_instances))
    ]
    for i in range(len(target_instances)):
        if target_instances[i].cost_allocation == cost_alloc:
            for j in range(len(target_instances)):
                id_arr[i][j] = 1 * (
                    target_instances[i].vat_portion == target_instances[j].vat_portion
                    and target_instances[i].pdri_portion
                    == target_instances[j].pdri_portion
                )

    id_arr = np.array(id_arr)

    # Operations associated with Notes #2 (see the above docstring)
    group_arr = [
        np.argwhere(id_arr[i, :] == 1).ravel() for i in range(len(target_instances))
    ]

    # Operations associated with Notes #3 (see the above docstring)
    group_id = [0 for _ in range(len(group_arr))]
    for i in range(len(group_arr)):
        if len(group_arr[i]) == 0:
            group_id[i] = -1
        if len(group_arr[i]) > 0:
            group_id[i] = np.prod(group_arr[i])

    group_id = np.array(group_id)

    # Operations associated with Notes #4 (see the above docstring)
    loc_id = np.unique(group_id)

    # Operations associated with Notes #5 (see the above docstring)
    identifier = [
        np.argwhere(group_id == loc_id[i]).ravel()
        for i in range(len(loc_id))
        if loc_id[i] >= 0
    ]

    return identifier


def get_instances(target_instances: tuple, identifier: list) -> tuple:
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

    result = []
    for i in range(len(identifier)):
        if len(identifier[i]) == 1:
            result.append(target_instances[identifier[i][0]])

        elif len(identifier[i]) > 1:
            add = target_instances[identifier[i][0]]
            for j in range(1, len(identifier[i])):
                add += target_instances[identifier[i][j]]
                result.append(add)

    return tuple(result)


def get_datetime(ordinal_date):
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


def get_lifting_data_split_non_gas(
    target_attr: dict,
    is_target_attr_volume: bool,
    project_years: np.ndarray,
    end_date_contract_1: datetime,
    start_date_contract_2: datetime,
) -> dict:
    """
    Split non-gas lifting data based on Production Sharing Contract (PSC) transitions.

    Parameters
    ----------
    target_attr: Dict[str, np.ndarray]
        Dictionary of target attribute data to be split.
    is_target_attr_volume: bool
        Indicates if the target attribute represents volume.
    project_years: np.ndarray
        Array of years corresponding to the project data.
    end_date_contract_1: datetime
        End date of the first contract.
    start_date_contract_2: datetime
        Start date of the second contract.

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        A nested dictionary containing split target attribute data for PSC 1 and PSC 2.
        The outer dictionary keys represent different target attributes, and the inner
        dictionary keys are 'PSC 1' and 'PSC 2', each with a numpy array of the split target_attr.

    Notes
    -----
    - If end year of the first contract is the same as the start year of the second contract,
      the target_attr for each attribute is split based on the transition year, adjusting
      the volume accordingly.
    - If end year of the first contract is different from the start year of the second contract,
      the target_attr for each attribute is split based on the transition years.
    """
    keys_transition = ["PSC 1", "PSC 2"]

    # End year of the first contract is the same as the start year of the second contract
    if end_date_contract_1.year == start_date_contract_2.year:
        id_transition = np.argwhere(project_years == end_date_contract_1.year).ravel().astype("int")
        days_diff = (
            end_date_contract_1
            - datetime(day=1, month=1, year=end_date_contract_1.year)
        )
        days_delta = (
            datetime(day=31, month=12, year=end_date_contract_1.year)
            - datetime(day=1, month=1, year=end_date_contract_1.year)
        )

        multiplier = (days_diff.days + 1) / (days_delta.days + 2)

        for key in target_attr.keys():
            target_attr[key] = {
                keys_transition[0]: target_attr[key][:id_transition[0] + 1].astype("float"),
                keys_transition[1]: target_attr[key][id_transition[0]:].astype("float"),
            }

        for key in target_attr.keys():
            if is_target_attr_volume is True:
                target_attr[key][keys_transition[0]][-1] *= multiplier
                target_attr[key][keys_transition[1]][0] *= (1.0 - multiplier)

    # End year of the first contract is different from the start year of the second contract
    else:
        id_transition = np.array(
            [
                np.argwhere(project_years == i).ravel().astype("int") for i in
                [end_date_contract_1.year, start_date_contract_2.year]
            ]
        ).ravel()

        for key in target_attr.keys():
            target_attr[key] = {
                keys_transition[0]: target_attr[key][:id_transition[0] + 1].astype("float"),
                keys_transition[1]: target_attr[key][id_transition[1]:].astype("float"),
            }

    return target_attr


def get_lifting_data_split_gas_no_nested(
    target_attr: dict,
    # is_target_attr_volume: bool,
    prod_year: dict,
    end_date_contract_1: datetime,
    start_date_contract_2: datetime,
):

    keys_transition = ["PSC 1", "PSC 2"]

    # End year of the first contract is the same as the start year of the second contract
    if end_date_contract_1.year == start_date_contract_2.year:
        id_transition = {
            key: np.argwhere(
                prod_year[key] == end_date_contract_1.year
            ).ravel().astype("int")
            for key in prod_year.keys()
        }

        for key in target_attr.keys():
            target_attr[key] = {
                keys_transition[0]: target_attr[key][:id_transition[key][0] + 1].astype("float"),
                keys_transition[1]: target_attr[key][id_transition[key][0]:].astype("float")
            }

    else:
        id_transition = {
            key: np.array(
                [
                    np.argwhere(prod_year[key] == i).ravel().astype("int")
                    for i in [end_date_contract_1.year, start_date_contract_2.year]
                ]
            ).ravel()
            for key in prod_year.keys()
        }

        print('\t')
        print(f'Filetype: {type(id_transition)}')
        print('id_transition = ', id_transition)

        for key in target_attr.keys():
            if target_attr[key] is None:
                target_attr[key] = {keys_transition[0]: None, keys_transition[1]: None}
            else:
                target_attr[key] = {
                    keys_transition[0]: target_attr[key][:id_transition[key][0] + 1].astype("float"),
                    keys_transition[1]: target_attr[key][id_transition[key][1]:].astype("float")
                }

    return target_attr


def get_cost_data_split(
    target_attr: list | np.ndarray,
    is_target_attr_volume: bool,
    expense_year: np.ndarray,
    end_date_contract_1: datetime,
    start_date_contract_2: datetime,
) -> dict:
    """
    Split cost data based on the transition between two Production Sharing Contracts (PSCs).

    Parameters
    ----------
    target_attr: list, np.ndarray
        The target attribute data to be split.
    is_target_attr_volume: bool
        Indicates if the target attribute represents volume.
    expense_year: np.ndarray
        Array of years corresponding to the cost data.
    end_date_contract_1: datetime
        End date of the first contract.
    start_date_contract_2: datetime
        Start date of the second contract.

    Returns
    -------
    dict
        A dictionary containing split target attribute data for PSC 1 and PSC 2.

    Notes
    -----
    - If end year of the first contract is the same as the start year of the second contract,
      the target_attr is split based on the transition year, adjusting the volume accordingly.
    - If end year of the first contract is different from the start year of the second contract,
      the target_attr is split based on the transition year.
    """
    keys_transition = ["PSC 1", "PSC 2"]

    id_transition = np.argwhere(expense_year == end_date_contract_1.year).ravel()

    if isinstance(target_attr, list):
        target_attr = np.array(target_attr)

    # End year of the first contract is the same as the start year of the second contract
    if end_date_contract_1.year == start_date_contract_2.year:
        days_diff = (
            end_date_contract_1
            - datetime(day=1, month=1, year=end_date_contract_1.year)
        )
        days_delta = (
            datetime(day=31, month=12, year=end_date_contract_1.year)
            - datetime(day=1, month=1, year=end_date_contract_1.year)
        )

        multiplier = (days_diff.days + 1) / (days_delta.days + 2)

        target_attr = {
            keys_transition[0]: target_attr[:int(max(id_transition) + 1)],
            keys_transition[1]: target_attr[int(min(id_transition)):]
        }

        if is_target_attr_volume is True:
            target_attr[keys_transition[0]][-1] = target_attr[keys_transition[0]][-1] * multiplier
            target_attr[keys_transition[1]][0] = target_attr[keys_transition[1]][0] * (1.0 - multiplier)

    # End year of the first contract is different from the start year of the second contract
    else:
        target_attr = {
            keys_transition[0]: target_attr[0:int(max(id_transition) + 1)],
            keys_transition[1]: target_attr[int(max(id_transition) + 1):]
        }

    return target_attr


def summarizer(
    array1: np.ndarray,
    array2: np.ndarray,
    startYear1: int,
    startYear2: int,
    endYear1: int = None,
    endYear2: int = None,
) -> tuple:
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
    indices1 = np.argwhere(
        np.logical_and(startYear1 <= years, years <= endYear1)
    ).flatten()
    res_arr1[indices1] = array1

    # Indices where the elements of array2 are not aligned with array years, then update the original array
    indices2 = np.argwhere(
        np.logical_and(startYear2 <= years, years <= endYear2)
    ).flatten()
    res_arr2[indices2] = array2

    return res_arr1 + res_arr2, start_year, end_year


def sum_remainder(start_year: int, end_year: int, arr: np.ndarray) -> float:
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
