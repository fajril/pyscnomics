"""
Handles calculations associated with the economic indicators of the project.
"""
from datetime import date
import numpy as np
import pyxirr


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
    return pyxirr.irr(cashflow)


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


def npv_skk_real_terms(cashflow: np.ndarray,
                       cashflow_years: np.ndarray,
                       discount_rate: float,
                       reference_year: int):
    t = np.arange(1, len(cashflow_years) + 1)
    reference_year = np.full_like(a=cashflow, fill_value=reference_year, dtype=float)
    cashflow_disc = np.where(t > 1,
                             cashflow / np.power((1 + discount_rate), (cashflow_years - reference_year)),
                             cashflow * np.power((1 + discount_rate), len(cashflow_years))
                             )
    return np.sum(cashflow_disc)


def npv_skk_nominal_terms(cashflow: np.ndarray,
                          cashflow_years: np.ndarray,
                          discount_rate: float,
                          ):
    t = np.arange(1, len(cashflow_years) + 1)
    reference_year = np.full_like(a=cashflow, fill_value=np.min(cashflow_years), dtype=float)
    cashflow_disc = np.where(t > 1,
                             cashflow / np.power((1 + discount_rate), (cashflow_years - reference_year)),
                             cashflow)

    return np.sum(cashflow_disc)
