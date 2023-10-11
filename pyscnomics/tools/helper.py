"""
Handles summation operation on two arrays, accounting for different starting years.
"""

import numpy as np
from functools import wraps

from pyscnomics.econ.selection import TaxRegime


def apply_inflation(inflation_rate: float):
    """
    Apply escalation/inflation to a particular function.

    Parameters
    ----------
    inflation_rate: float
        A constant depicting the escalation/inflation rate.

    Returns
    -------
    decorated: function
        A decorated function that modifies the target function.
    """
    def _decorated(f):
        @wraps(f)
        def _wrapper(*args, **kwargs):
            exponents = np.arange(0, len(f(*args, **kwargs)), 1)
            inflation_arr = (1 + inflation_rate) ** exponents
            modified_arr = f(*args, **kwargs) * inflation_arr
            return modified_arr
        return _wrapper
    return _decorated


def apply_vat(tax_regime: str, vat_rate: float):
    """
    Apply Value Added Tax (VAT/PPN) to a particular function.

    Parameters
    ----------
    tax_regime: str
        The administered tax regime.
    vat_rate: float
        The rate of VAT/PPN.

    Returns
    -------
    decorated: function
        A decorated function that modifies the target function.
    """
    def _decorated(f):
        @wraps(f)
        def _wrapper(*args, **kwargs):
            if tax_regime == TaxRegime.NAILED_DOWN:
                modified_arr = f(*args, **kwargs) * (1 + vat_rate)
            return modified_arr
        return _wrapper
    return _decorated


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
