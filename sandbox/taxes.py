"""
Handles taxes calculations.
"""

import numpy as np
import datetime as dt

from pyscnomics.econ.selection import TaxRegime


# FIXME: Fix the value of BPT and CIT
regime = {
    TaxRegime.UU_36_2008: {
        dt.date(year=2009, month=1, day=1).year: {"cit": 0.28, "bpt": 0.2},
        dt.date(year=2011, month=1, day=1).year: {"cit": 0.25, "bpt": 0.2},
        dt.date(year=2013, month=1, day=1).year: {"cit": 0.25, "bpt": 0.2},
        dt.date(year=2014, month=1, day=1).year: {"cit": 0.25, "bpt": 0.2},
    },
    TaxRegime.UU_02_2020: {
        dt.date(year=2020, month=1, day=1).year: {"cit": 0.22, "bpt": 0.0},
        dt.date(year=2021, month=1, day=1).year: {"cit": 0.2, "bpt": 0.0},
    },
    TaxRegime.UU_07_2021: {
        dt.date(year=2021, month=1, day=1).year: {"cit": 0.22, "bpt": 0.0},
        dt.date(year=2022, month=1, day=1).year: {"cit": 0.2, "bpt": 0.0},
    },
}

regime2 = {
    TaxRegime.UU_36_2008: {
        "cit": {2009: 0.28,
                2011: 0.25,
                2013: 0.3,
                2014: 0.5,
                }

    }
}


def get_fiscal_nailed_down(start_year: int, end_year: int, target_dict: dict) -> np.ndarray:
    """
    Generate an array of fiscal regimes based on a target dictionary for specified years.

    This function generates an array of fiscal regimes corresponding to the years between
    the start year (inclusive) and the end year (exclusive), using a given target dictionary
    that maps years to fiscal regimes.

    If a year is not present in the target dictionary, the function fills it with the closest
    preceding available year's fiscal regime.

    Args:
        start_year (int): The starting year (inclusive) for generating the fiscal regimes array.
        end_year (int): The ending year (exclusive) for generating the fiscal regimes array.
        target_dict (dict): A dictionary mapping years to corresponding fiscal regimes.

    Returns:
        np.ndarray: An array containing the fiscal regimes corresponding to the years between
        start_year and end_year.
    """

    arr = np.array([i if i in target_dict.keys() else np.nan for i in range(start_year, end_year)])
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, out=idx)
    arr[mask] = arr[idx[mask]]

    return [target_dict[r] for r in arr]


def get_fiscal_is():

    start_year = 2010
    end_year = 2025

    years_arr = np.arange(start_year, end_year + 1)
    tax_arr = np.zeros_like(years_arr)

    tax_regime = TaxRegime.UU_36_2008

    A = list(regime[tax_regime].keys())
    B = []

    for year in A:
        B.append(regime[tax_regime][year]["cit"])

    for idx, year in enumerate(A):
        if idx + 1 < len(A):
            # print(idx, year, len(A))
            indices = np.argwhere(np.logical_and(years_arr >= A[idx],
                                                 years_arr < A[idx + 1])).flatten()
        else:
            # print(idx, year, len(A))
            indices = np.argwhere(years_arr >= max(A))

        tax_arr[indices] = year

        if idx + 1 >= len(A):
            break

    return tax_arr

yr = get_fiscal_is()
print(yr)

target_dict = [regime2[TaxRegime.UU_36_2008]["cit"][r] for r in yr]
print(target_dict)
