"""
Handles calculations associated with the economic indicators of the project.
"""
from datetime import date
import numpy as np
import pyxirr
import numpy_financial as npf

from pyscnomics.econ.selection import DiscountingMode


def pot(
    cashflow: np.ndarray,
) -> float:
    """
    Calculate the payout time (POT) for a series of cashflows.

    Parameters
    ----------
    cashflow : np.ndarray
        An array containing the cashflows over time.

    Returns
    -------
    float
        The payout time (POT), which represents the time at which
        the cumulative cashflows turn positive for the first time.
        It is calculated as the time between the last negative cumulative
        cashflow and the first positive cumulative cashflow divided by
        the difference between the first positive and last negative cumulative
        cashflows, plus the index of the first positive cumulative cashflow.

    Notes
    ------
    - This function assumes that `cashflow` is a numpy array.
    - If `cashflow` does not contain any positive cumulative values,
      the function will return an index of 0, indicating that no payout
      has occurred.
    """
    sum_cf = np.cumsum(cashflow)
    idx = np.argwhere(sum_cf > 0)[0]

    return sum_cf[idx - 1] / (sum_cf[idx] - sum_cf[idx - 1]) + idx


def npv(
    cashflow: np.ndarray,
    disc_rate: float = 0.1,
) -> float:
    """
    Calculate the Net Present Value (NPV) of a series of cashflows.

    Parameters:
    ----------
    cashflow : np.ndarray
        An array containing the cashflows over time.

    disc_rate : float, optional
        The discount rate used to calculate the NPV. Default is 0.1 (10%).

    Returns:
    -------
    float
        The Net Present Value (NPV) of the given cashflows based on the
        specified discount rate.

    Notes:
    ------
    - This function calculates the NPV using the provided discount rate
      and the cashflows.
    - The NPV represents the present value of future cashflows, taking
      into account the time value of money. It indicates whether an
      investment or project is financially viable (positive NPV) or not
      (negative NPV).
    """
    return pyxirr.npv(disc_rate, amounts=cashflow)


def xnpv(
    cashflow: np.ndarray,
    start_date: date,
    end_date: date = None,
    disc_rate: float = 0.1,
) -> float:
    """
    Calculate the Extended Net Present Value (XNPV) of a series of cashflows.

    Parameters:
    ----------
    cashflow : np.ndarray
        An array containing the cashflows over time.

    start_date : date
        The start date of the cashflows.

    end_date : date, optional
        The end date of the cashflows.

    disc_rate : float, optional
        The discount rate used to calculate the XNPV. Default is 0.1 (10%).

    Returns:
    -------
    float
        The Extended Net Present Value (XNPV) of the given cashflows based on
        the specified discount rate and date range.

    Notes:
    ------
    - This function calculates the XNPV using the provided discount rate,
      cashflows, and date range.
    - XNPV accounts for the exact timing of cashflows based on their respective
      dates, providing a more precise measure of present value than the regular
      NPV.
    """
    dates = _date_arange(
        start_date=start_date, end_date=end_date, arr_length=len(cashflow)
    )
    return pyxirr.xnpv(disc_rate, dates=dates, amounts=cashflow)


def irr(
    cashflow: np.ndarray,
) -> float:
    """
    Calculate the Internal Rate of Return (IRR) of a series of cashflows.

    Parameters:
    ----------
    cashflow : np.ndarray
        An array containing the cashflows over time.

    Returns:
    -------
    float
        The Internal Rate of Return (IRR) of the given cashflows.

    Notes:
    ------
    - This function calculates the IRR, which represents the discount rate at
      which the Net Present Value (NPV) of the cashflows becomes zero.
    - The IRR is a measure of the profitability of an investment or project.
      If the IRR is greater than the discount rate, the investment
      is considered financially attractive.
    """
    # Condition where all the cashflow is positive
    if np.all(cashflow >= 0) or np.all(cashflow <= 0):
        irr_result = 0
    else:
        irr_result = npf.irr(cashflow)

    # Condition where the irr resulting negative
    if irr_result is None:
        irr_result = 0
    elif irr_result < 0:
        irr_result = 0
    else:
        pass

    return irr_result


def xirr(
    cashflow: np.ndarray,
    start_date: date,
    end_date: date = None,
) -> float:
    """
    Calculate the Extended Internal Rate of Return (XIRR) of a series of cashflows.

    Parameters:
    ----------
    cashflow : np.ndarray
        An array containing the cashflows over time.

    start_date : date
        The start date of the cashflows.

    end_date : date, optional
        The end date of the cashflows.

    Returns:
    -------
    float
        The Extended Internal Rate of Return (XIRR) of the given cashflows,
        taking into account the timing of cashflows based on their respective dates.

    Notes:
    ------
    - This function calculates the XIRR, which is an extension of the Internal
      Rate of Return (IRR) that considers the exact timing of cashflows based on
      their respective dates.
    """
    dates = _date_arange(
        start_date=start_date, end_date=end_date, arr_length=len(cashflow)
    )
    return pyxirr.xirr(dates=dates, amounts=cashflow)


def _date_arange(
    start_date: date,
    end_date: date = None,
    arr_length: int = 0,
) -> list:
    """
    Generate a list of dates within a specified date range or of a specific length.

    Parameters:
    ----------
    start_date : date
        The starting date for generating the list of dates.

    end_date : date, optional
        The ending date for generating the list of dates. If not provided, the list
        will be generated with a specified length instead.

    arr_length : int, optional
        The desired length of the date list when `end_date` is not provided.
        Ignored if `end_date` is provided.

    Returns:
    -------
    list of date
        A list of date objects containing dates within the specified range or of
        the specified length.

    Notes:
    ------
    - This function can be used to generate a sequence of dates, either within a
      specified date range (from `start_date` to `end_date`) or of a specific length.
    - If `end_date` is not provided, the function generates a list of dates with
      a length of `arr_length` starting from `start_date`.
    - If `end_date` is provided, the list includes `start_date` and `end_date`.
    """
    if end_date is None:
        dates = [
            date(yr, 1, 1)
            for yr in range(start_date.year + 1, start_date.year + arr_length)
        ]
    else:
        dates = [date(yr, 1, 1) for yr in range(start_date.year + 1, end_date.year)]

    dates.insert(0, start_date)

    if end_date is not None:
        dates.append(end_date)

    return dates


def npv_nominal_terms(
    cashflow: np.ndarray,
    cashflow_years: np.ndarray,
    discount_rate: float,
    reference_year: int,
    discounting_mode: DiscountingMode.END_YEAR
) -> float:
    """
    A function to calculate the Net Present Value (NPV) of a series of cashflows in nominal terms.

    Parameters
    ----------
    cashflow : np.ndarray
        Array of cashflows for each period.

    cashflow_years : np.ndarray
        Array of corresponding years for each cashflow..

    discount_rate : float
        The discount rate to be applied.

    reference_year : int
        The reference year used to calculate NPV.

    discounting_mode : DiscountingMode
        Enum representing the discounting mode,
        either DiscountingMode.END_YEAR or DiscountingMode.MID_YEAR.

    Returns
    -------
    float
        The Net Present Value of the cashflows in nominal terms.

    Notes
    -----
    If discounting_mode is DiscountingMode.END_YEAR, the discounting is applied
    at the end of each year. If it is DiscountingMode.MID_YEAR, the discounting is
    applied at the mid-year of each period.
    """
    reference_year_arr = np.full_like(cashflow, fill_value=reference_year)
    t_arr = cashflow_years - reference_year_arr

    if discounting_mode is DiscountingMode.END_YEAR:
        year_factor = 0
    else:
        year_factor = 0.5

    dcf = np.where(t_arr > 0,
                   1 / (np.power(1 + discount_rate, t_arr + year_factor)),
                   1)

    discounted_cashflow = cashflow * dcf
    return float(np.sum(discounted_cashflow))


def npv_real_terms(
    cashflow: np.ndarray,
    cashflow_years: np.ndarray,
    discount_rate: float,
    reference_year: int,
    inflation_rate: float,
    discounting_mode: DiscountingMode.END_YEAR
) -> float:
    """
    A function to calculate the Net Present Value (NPV) of a series of cashflows in real terms.

    Parameters
    ----------
    cashflow : np.ndarray
        Array of cashflows for each period.
    cashflow_years : np.ndarray
        Array of corresponding years for each cashflow.
    discount_rate : float
        The discount rate to be applied.
    reference_year : int
        The reference year for discounting and inflation adjustments.
    inflation_rate : float
        The inflation rate for adjusting future cashflows.
    discounting_mode : DiscountingMode
        Enum representing the discounting mode,
        either DiscountingMode.END_YEAR or DiscountingMode.MID_YEAR.

    Returns
    -------
    float
        The Net Present Value of the cashflows in real terms.

    Notes
    -----
    If discounting_mode is DiscountingMode.END_YEAR, the discounting is applied
    at the end of each year. If it is DiscountingMode.MID_YEAR, the discounting is
    applied at the mid-year of each period.
    """
    reference_year_arr = np.full_like(cashflow, fill_value=reference_year)
    t_arr = cashflow_years - reference_year_arr

    if discounting_mode is DiscountingMode.END_YEAR:
        year_factor = 0
    else:
        year_factor = 0.5

    dcf = np.where(t_arr > 0,
                   1 / (np.power(1 + discount_rate, t_arr + year_factor)),
                   1)

    discounted_cashflow = np.where(
        t_arr > 0,
        dcf * cashflow,
        cashflow * np.power((1 + inflation_rate), np.max(cashflow_years) - reference_year)
    )
    return float(np.sum(discounted_cashflow))


def npv_skk_nominal_terms(
    cashflow: np.ndarray,
    cashflow_years: np.ndarray,
    discount_rate: float,
    discounting_mode: DiscountingMode.END_YEAR
) -> float:
    """
    A function to calculate the Net Present Value (NPV) of a series of cashflows
    in SKK Nominal terms method.

    Parameters
    ----------
    cashflow : np.ndarray
        Array of cashflows for each period.
    cashflow_years : np.ndarray
        Array of corresponding years for each cashflow.
    discount_rate : float
        The discount rate to be applied.
    discounting_mode : DiscountingMode
        Enum representing the discounting mode,
        either DiscountingMode.END_YEAR or DiscountingMode.MID_YEAR.

    Returns
    -------
    float
        The Net Present Value of the cashflows in SKK nominal terms.

    Notes
    -----
    [1] If discounting_mode is DiscountingMode.END_YEAR, the discounting is applied
    at the end of each year. If it is DiscountingMode.MID_YEAR, the discounting is
    applied at the mid-year of each period.

    [2] In SKK Nominal terms, the reference year is set to the starting year of the project.
    """

    if discounting_mode is DiscountingMode.END_YEAR:
        year_factor = 0
    else:
        year_factor = 0.5

    reference_year_arr = np.full_like(cashflow, fill_value=np.min(cashflow_years))
    dcf = 1/(np.power((1+discount_rate), cashflow_years - reference_year_arr + year_factor))
    discounted_cashflow = cashflow * dcf
    return float(np.sum(discounted_cashflow))


def npv_skk_real_terms(
    cashflow: np.ndarray,
    cashflow_years: np.ndarray,
    discount_rate: float,
    reference_year: int,
    discounting_mode: DiscountingMode.END_YEAR
) -> float:
    """
    A function to calculate the Net Present Value (NPV) of a series of cashflows
    in SKK Real terms method.

    Parameters
    ----------
    cashflow : np.ndarray
        Array of cashflows for each period.
    cashflow_years : np.ndarray
        Array of corresponding years for each cashflow.
    discount_rate : float
        The discount rate to be applied.
    reference_year : int
        The reference year for discounting and inflation adjustments.
    discounting_mode : DiscountingMode
        Enum representing the discounting mode,
        either DiscountingMode.END_YEAR or DiscountingMode.MID_YEAR.

    Returns
    -------
    float
        The Net Present Value of the cashflows in SKK real terms.

    Notes
    -----
    [1] If discounting_mode is DiscountingMode.END_YEAR, the discounting is applied
    at the end of each year. If it is DiscountingMode.MID_YEAR, the discounting is
    applied at the mid-year of each period.

    [2] In SKK Real terms, the inflation rate is set to as the same as discount rate.

    """

    reference_year_arr = np.full_like(cashflow, fill_value=reference_year)
    t_arr = cashflow_years - reference_year_arr

    if discounting_mode is DiscountingMode.END_YEAR:
        year_factor = 0
    else:
        year_factor = 0.5

    dcf = np.where(t_arr >= 0,
                   1/np.power((1+discount_rate), t_arr + year_factor),
                   1)

    discounted_cashflow = (
        np.where(cashflow_years > reference_year,
                 dcf * cashflow,
                 dcf * cashflow * np.power((1 + discount_rate),
                                           np.max(cashflow_years) - reference_year) + year_factor)
    )
    return float(np.sum(discounted_cashflow))


def npv_point_forward(
    cashflow: np.ndarray,
    cashflow_years: np.ndarray,
    discount_rate: float,
    reference_year: int,
    discounting_mode: DiscountingMode.END_YEAR
) -> float:
    """
    Calculate the Net Present Value (NPV) of a series of cashflows using a point-forward approach.

    Parameters
    ----------
    cashflow : np.ndarray
        Array of cashflows for each period.
    cashflow_years : np.ndarray
        Array of corresponding years for each cashflow.
    discount_rate : float
        The discount rate to be applied.
    reference_year : int
        The reference year for discounting.
    discounting_mode : DiscountingMode
        Enum representing the discounting mode,
        either DiscountingMode.END_YEAR or DiscountingMode.MID_YEAR.

    Returns
    -------
    float
        The Net Present Value of the cashflows using the point-forward approach.

    Notes
    -----
    [1] If discounting_mode is DiscountingMode.END_YEAR, the discounting is applied
    at the end of each year. If it is DiscountingMode.MID_YEAR, the discounting is
    applied at the mid-year of each period.

    [2] The point-forward approach is that any cashflow prior to the reference year will be neglected..
    """
    reference_year_arr = np.full_like(cashflow, fill_value=reference_year)
    t_arr = cashflow_years - reference_year_arr

    if discounting_mode is DiscountingMode.END_YEAR:
        year_factor = 0
    else:
        year_factor = 0.5

    dcf = np.where(t_arr >= 0,
                   1/np.power((1+discount_rate), t_arr + year_factor),
                   1)

    discounted_cashflow = np.where(t_arr < 0,
                                   0,
                                   dcf * cashflow)
    return float(np.sum(discounted_cashflow))


def pot_psc(
    cashflow: np.ndarray,
    cashflow_years: np.ndarray,
    reference_year: int
):
    """
    Calculate Pay Out Time (POT) for a Production Sharing Contract (PSC).

    Parameters
    ----------
    cashflow: np.ndarray
        Array of cash flows for the project.
    cashflow_years: np.ndarray
        Array of years corresponding to the cash flows.
    reference_year: int
        Reference year for the calculation.

    Returns
    -------
    float:
        The Pay Out Time (POT) for the given cash flows.
    """
    project_year = np.arange(1, (np.max(cashflow_years) - reference_year + 1) + 1)
    project_year = np.concatenate((np.zeros(reference_year - np.min(cashflow_years)), project_year))
    cum_cashflow = np.cumsum(cashflow)

    # Create an array of zeros with the same length as cum_cashflow
    pot = np.zeros_like(cum_cashflow, dtype=float)

    # Find indices where cum_cashflow changes sign from negative to positive
    positive_change_indices = np.where((cum_cashflow[:-1] < 0) & (cum_cashflow[1:] >= 0))[0]

    # Calculate the values using vectorized operations
    pot[positive_change_indices] = (
        (project_year[positive_change_indices] +
         (project_year[positive_change_indices + 1] - project_year[positive_change_indices])
         /
         (cum_cashflow[positive_change_indices + 1] - cum_cashflow[positive_change_indices])
         *
         (0 - cum_cashflow[positive_change_indices]))
    )

    return float(np.max(pot))

# def pot_psc(cashflow: np.ndarray,
#             cashflow_years: np.ndarray,
#             reference_year: int):
#
#     project_year = np.arange(1, (np.max(cashflow_years) - reference_year + 1) + 1)
#     project_year = np.concatenate((np.zeros(reference_year - np.min(cashflow_years)), project_year))
#     cum_cashflow = np.cumsum(cashflow)
#
#     pot = []
#     for index, cash in enumerate(cum_cashflow):
#         print(index, cash)
#         if index == 0 or index == len(cum_cashflow) - 1:
#             value = 0
#             pot.append(float(value))
#         else:
#             value = (
#                 np.where(np.logical_and(cum_cashflow[index] < 0, cum_cashflow[index + 1] > 0),
#                          project_year[index] + np.divide((project_year[index + 1] - project_year[index]),
#                                                          (cum_cashflow[index + 1] - cum_cashflow[index]),
#                                                          where=(cum_cashflow[index + 1] -
#                                                                 cum_cashflow[index]) != 0)
#                          * (0 - cum_cashflow[index]),
#                          0)
#             )
#             pot.append(float(value))
#
#     return float(np.max(pot))
