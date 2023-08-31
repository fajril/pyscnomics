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

# def get_fiscal_nailed_down(start_year, end_year, tax_regime: TaxRegime, fiscal="cit"):
#
#     max_year = max(regime[tax_regime].keys())
#     cit = [regime[tax_regime][yr][fiscal]
#            if yr >= start_year
#            else regime[tax_regime][max_year][fiscal]
#            for yr, val in regime[tax_regime].items()]
#
#     cit_length = len(regime[tax_regime].keys())
#
#     return np.concatenate((cit, np.repeat(cit[-1], end_year - start_year + 1 - cit_length)))
#
#
# def get_fiscal_nd(start_year, end_year, tax_regime: TaxRegime):
#
#     arr_years = np.arange(start_year, end_year + 1, 1)
#     arr_cit = np.zeros_like(arr_years)
#     arr_uu = [i for i in regime[tax_regime].keys()]
#
#     x = []
#     for i in range(len(arr_uu), 1, -1):
#         x.append(arr_uu[i - 1] - arr_uu[i - 2])
#
#     x.reverse()
#
#     y = np.concatenate(([0], np.cumsum(x)))
#
#     print('\t')
#     print('y = ', y)
#
#     arr_cit[y[0]:y[1]] = arr_uu[0]
#     arr_cit[y[1]:y[2]] = arr_uu[1]
#     arr_cit[y[2]:y[3]] = arr_uu[2]
#
#     print('\t')
#     print('arr_years = ', arr_years)
#     print('arr_uu = ', arr_uu)
#     print('x = ', x)
#     print('arr_cit = ', arr_cit)


def get_fiscal_fa(start_year, end_year, a):

    arr = np.array([i if i in a.keys() else np.nan for i in range(start_year, end_year)])
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, out=idx)
    arr[mask] = arr[idx[mask]]

    return [a[r] for r in arr]

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

a = [regime2[TaxRegime.UU_36_2008]["cit"][r] for r in yr]
print(a)

# arr = get_fiscal_fa(
#     dt.date(2009, 1, 1).year,
#     dt.date(2030, 1, 1).year,
#     regime2[TaxRegime.UU_36_2008]["cit"]
# )

# cit_arr = get_fiscal_nailed_down(
#     dt.date(2009, 1, 1).year,
#     dt.date(2015, 1, 1).year,
#     TaxRegime.UU_36_2008
# )

# print('\t')
# print(cit_arr)

# def get_cit(start_date: dt.date, end_date: dt.date, tax_regime: TaxRegime):
#
#     years_arr = np.arange(start_date.year, end_date.year + 1, 1)
#
#     # a = []
#     print('\t')
#     print('years_arr = ', years_arr)
#
#     print('\t')
#
#     arr_key = []
#
#     arr_cit = np.zeros_like(years_arr)
#
#     for key in regime[tax_regime].keys():
#         arr_key.append(key)
#
#         for i in arr_key:
#             indices = np.argwhere(i == years_arr).ravel()
#
#             print(regime[tax_regime][i]["cit"])
#
#             arr_cit[indices:] = regime[tax_regime][i]["cit"]

            # print('\t')
            # print(indices)

            # if start_date.year == i:
            #     arr_cit[]

            # np.where(years_arr < i)
            # idx = np.argwhere(i == years_arr).ravel()

    # print('\t')
    # # print('arr_key = ', arr_key)
    # print('arr_cit = ', arr_cit)


    # for i, val in regime[tax_regime].items():
    #     for j in val.values():
    #
    #         print(j)
    #     print(i, val)

    #     pass
    #
    # for i in range(len(list(regime[tax_regime].keys()))):
    #     a.append(np.argwhere(list(regime[tax_regime].keys())[i] == years_arr).ravel())
    #
    # print('\t')
    # print(years_arr)
    # print(f'Filetypetype: {type(list(regime[tax_regime].keys()))}')
    # print(regime[tax_regime].keys())
    # print('a = ', a)

# print('\t')
# print('cit_arr = ', cit_arr)


# def calc_contractor_tax(
#     start_year: int, end_year: int, tax_regime: regime, tax_treaty: float = 0.0
# ):
#     # CIT = Corporate Income Tax = PPH
#     # BPT = Branch Profit Tax = PDBR
#
#
#
#     return (1 - tax_regime["cit"]) * (tax_regime["bpt"] - tax_treaty) + tax_regime[
#         "cit"
#     ]
#
#
# def get_pdri():
#     raise NotImplementedError
#
#
# def get_pdrd():
#     raise NotImplementedError
