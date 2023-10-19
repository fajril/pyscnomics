"""
Handles summation operation on two arrays, accounting for different starting years.
"""

import numpy as np
from functools import wraps, reduce
from pyscnomics.econ.selection import FluidType


class TaxesException(Exception):
    """Exception to be raised for a misuse of VAT and PDRI function"""

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
            raise TaxesException(
                f"Unequal length of arrays: "
                f"{param.__class__.__qualname__} ({len(param)}), "
                f"{target_func.__class__.__qualname__} ({len(target_func)})."
            )
        param_arr = param

    if isinstance(param, (float, int)):
        param_arr = np.repeat(param, len(target_func))

    return param_arr


def apply_inflation(inflation_rate: float | int) -> callable:
    """
    Decorator function to apply escalation/inflation to a target function's result.

    Parameters
    ----------
    inflation_rate: float or int
        A constant depicting the escalation/inflation rate.

    Returns
    -------
    Callable
        A decorated function that applies inflation multipliers to a target function's result.

    Notes
    -----
    This decorator is used to apply inflation multipliers to the result of a target function.
    It takes inflation rate as an input argument and returns a decorated function.
    """
    def _decorated(f):
        @wraps(f)
        def _wrapper(*args, **kwargs):
            exponents = np.arange(0, len(f(*args, **kwargs)), 1)
            inflation_arr = (1.0 + inflation_rate) ** exponents
            modified_arr = f(*args, **kwargs) * inflation_arr
            return modified_arr

        return _wrapper

    return _decorated


def apply_vat_pdri(
    vat_portion: float | int,
    vat_rate: np.ndarray | float | int,
    vat_discount: np.ndarray | float | int,
    pdri_portion: float | int,
    pdri_rate: np.ndarray | float | int,
    pdri_discount: np.ndarray | float | int,
) -> callable:
    """
    Decorator for applying VAT and PDRI multipliers to a target function's result.

    Parameters:
    -----------
    vat_portion : float or int
        The portion of VAT (Value Added Tax) to be applied.

    vat_rate : numpy.ndarray or float or int
        The VAT rate(s) to be applied as a multiplier.
        Can be a single value or an array.

    vat_discount : numpy.ndarray or float or int
        The VAT discount(s) to be applied as a multiplier.
        Can be a single value or an array.

    pdri_portion : float or int
        The portion of PDRI to be applied.

    pdri_rate : numpy.ndarray or float or int
        The PDRI rate(s) to be applied as a multiplier.
        Can be a single value or an array.

    pdri_discount : numpy.ndarray or float or int
        The PDRI discount(s) to be applied as a multiplier.
        Can be a single value or an array.

    Returns:
    --------
    Callable
        A decorated function that applies VAT and PDRI multipliers to a target function's result.

    Notes:
    ------
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
                vat_rate_arr = check_input(target_func=f(*args, **kwargs), param=vat_rate)
                vat_discount_arr = check_input(
                    target_func=f(*args, **kwargs), param=vat_discount
                )
                vat_multiplier = vat_portion_arr * vat_rate_arr * (1.0 - vat_discount_arr)
                return vat_multiplier

            def _get_pdri():
                pdri_portion_arr = np.repeat(pdri_portion, len(f(*args, **kwargs)))
                pdri_rate_arr = check_input(target_func=f(*args, **kwargs), param=pdri_rate)
                pdri_discount_arr = check_input(
                    target_func=f(*args, **kwargs), param=pdri_discount
                )
                pdri_multiplier = pdri_portion_arr * pdri_rate_arr * (1.0 - pdri_discount_arr)
                return pdri_multiplier

            total_multiplier = _get_vat() + _get_pdri()
            return f(*args, **kwargs) * (1.0 + total_multiplier)

        return _wrapper

    return _decorated


def apply_lbt(lbt_discount: np.ndarray | float | int) -> callable:
    """
    Decorator for applying Land and Building Tax (LBT/PBB) discount to a target function's result.

    Parameters:
    -----------
    lbt_discount : numpy.ndarray or float or int
        The LBT discount(s) to be applied as a multiplier.
        Can be a single value or an array.

    Returns:
    --------
    Callable
        A decorated function that applies LBT discount to a target function's result.

    Notes:
    ------
    This decorator is used to apply LBT/PBB discount to the result of a target function.
    It takes the LBT discount as a parameter and returns a decorated function.

    The decorated function first calculates the LBT discount multiplier based on the
    input parameter and then applies this multiplier to the result of the target
    function. The final result is adjusted by subtracting the discount.
    """
    def _decorated(f):
        @wraps(f)
        def _wrapper(*args, **kwargs):
            lbt_discount_arr = check_input(target_func=f(*args, **kwargs), param=lbt_discount)
            return f(*args, **kwargs) * (1.0 - lbt_discount_arr)
        return _wrapper
    return _decorated


def apply_pdrd(pdrd_discount: np.ndarray | float | int) -> callable:
    """
    Decorator for applying PDRD to a target function's result.

    Parameters:
    -----------
    pdrd_discount : numpy.ndarray or float or int
        The PDRD discount(s) to be applied as a multiplier. Can be a single value or an array.

    Returns:
    --------
    Callable
        A decorated function that applies PDRD discount to its result.

    Notes:
    ------
    This decorator is used to apply PDRD discount to the result of a target function.
    It takes the PDRD discount as a parameter and returns a decorated function.

    The decorated function first calculates the PDRD discount multiplier based on the
    input parameter and then applies this multiplier to the result of the target function.
    The final result is adjusted by subtracting the discount.
    """
    def _decorated(f):
        @wraps(f)
        def _wrapper(*args, **kwargs):
            pdrd_discount_arr = check_input(target_func=f(*args, **kwargs), param=pdrd_discount)
            return f(*args, **kwargs) * (1.0 - pdrd_discount_arr)
        return _wrapper
    return _decorated


def apply_cost_modification(
    start_year: int,
    cost: np.ndarray,
    expense_year: np.ndarray,
    inflation_rate: float | int,
    vat_portion: float | int,
    vat_rate: np.ndarray | float | int,
    vat_discount: np.ndarray | float | int,
    pdri_portion: float | int,
    pdri_rate: np.ndarray | float | int,
    pdri_discount: np.ndarray | float | int,
    lbt_discount: np.ndarray | float | int,
    pdrd_discount: np.ndarray | float | int,
) -> np.ndarray:
    """
    Apply cost modifications to the given cost array based on various parameters.

    Parameters:
    -----------
    start_year : int
        The start year of the project.
    cost : np.ndarray
        An array containing the original cost data.
    expense_year : np.ndarray
        An array specifying the expense years for each cost element.
    inflation_rate : float | int
        The annual inflation rate as a decimal or integer.
    vat_portion : float | int
        The portion of cost subject to Value Added Tax (VAT) as a decimal or integer.
    vat_rate : np.ndarray | float | int
        An array or single value representing the VAT rate(s) as a decimal or integer.
    vat_discount : np.ndarray | float | int
        An array or single value representing the VAT discount(s) as a decimal or integer.
    pdri_portion : float | int
        The portion of cost subject to PDRI as a decimal or integer.
    pdri_rate : np.ndarray | float | int
        An array or single value representing the PDRI rate(s) as a decimal or integer.
    pdri_discount : np.ndarray | float | int
        An array or single value representing the PDRI discount(s) as a decimal or integer.
    lbt_discount : np.ndarray | float | int
        An array or single value representing the Land and Building Tax discount(s)
        as a decimal or integer.
    pdrd_discount : np.ndarray | float | int
        An array or single value representing the PDRD discount(s) as a decimal or integer.

    Returns:
    --------
    cost_modified: np.ndarray
        An array containing the modified cost data after applying various adjustments.
    """

    # Apply inflation
    exponents = expense_year - start_year
    inflation_arr = (1.0 + inflation_rate) ** exponents
    cost_modified = cost * inflation_arr

    # Apply VAT/PPN and PDRI
    vat_portion_arr = np.repeat(vat_portion, len(cost))
    vat_rate_arr = check_input(target_func=cost, param=vat_rate)
    vat_discount_arr = check_input(target_func=cost, param=vat_discount)
    vat_multiplier = vat_portion_arr * vat_rate_arr * (1.0 - vat_discount_arr)

    pdri_portion_arr = np.repeat(pdri_portion, len(cost))
    pdri_rate_arr = check_input(target_func=cost, param=pdri_rate)
    pdri_discount_arr = check_input(target_func=cost, param=pdri_discount)
    pdri_multiplier = pdri_portion_arr * pdri_rate_arr * (1.0 - pdri_discount_arr)

    add_multiplier = vat_multiplier + pdri_multiplier
    cost_modified *= 1.0 + add_multiplier

    # Apply LBT/PPN
    lbt_discount_arr = check_input(target_func=cost, param=lbt_discount)
    cost_modified *= 1.0 - lbt_discount_arr

    # Apply PDRD
    pdrd_discount_arr = check_input(target_func=cost, param=pdrd_discount)
    cost_modified *= 1.0 - pdrd_discount_arr

    return cost_modified


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
    id_arr = [[0 for _ in range(len(target_instances))] for _ in range(len(target_instances))]
    for i in range(len(target_instances)):
        if target_instances[i].cost_allocation == cost_alloc:
            for j in range(len(target_instances)):
                id_arr[i][j] = 1 * (
                    target_instances[i].vat_portion == target_instances[j].vat_portion
                    and target_instances[i].pdri_portion == target_instances[j].pdri_portion
                )

    id_arr = np.array(id_arr)

    # Operations associated with Notes #2 (see the above docstring)
    group_arr = [np.argwhere(id_arr[i, :] == 1).ravel() for i in range(len(target_instances))]

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
    identifier = [np.argwhere(group_id == loc_id[i]).ravel() for i in range(len(loc_id)) if loc_id[i] >= 0]

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
