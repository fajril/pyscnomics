from datetime import date

import numpy as np
import pyxirr

def pot(
    cashflow: np.ndarray,
) -> float:
    sum_cf = np.cumsum(cashflow)
    idx = np.argwhere(sum_cf > 0)[0]
    return sum_cf[idx - 1] / (sum_cf[idx] - sum_cf[idx - 1]) + idx

def npv(
    cashflow: np.ndarray,
    disc_rate: float = .1,
) -> float:
    return pyxirr.npv(disc_rate, amounts=cashflow)

def xnpv(
    cashflow: np.ndarray,
    start_date: date,
    end_date: date = None,
    disc_rate: float = .1,
) -> float:
    dates = _date_arange(
        start_date=start_date, 
        end_date=end_date,
        arr_length=len(cashflow)
    )
    return pyxirr.xnpv(disc_rate, dates=dates, amounts=cashflow)

def irr(
    cashflow:np.ndarray,
) -> float:
    return pyxirr.irr(cashflow)

def xirr(
    cashflow: np.ndarray,
    start_date: date,
    end_date: date = None,
) -> float:
    dates = _date_arange(
        start_date=start_date, 
        end_date=end_date,
        arr_length=len(cashflow)
    )
    return pyxirr.xirr(dates=dates, amounts=cashflow)

def _date_arange(
    start_date:date,
    end_date:date= None,
    arr_length: int = 0,
) -> list:
    if end_date is None:
        dates = [
            date(yr, 1, 1)
            for yr in range(
                start_date.year + 1,
                start_date.year + arr_length)
        ]
    else:
        dates = [
            date(yr, 1, 1)
            for yr in range(start_date.year + 1, end_date.year)
        ]
    dates.insert(0, start_date)
    if end_date is not None:
        dates.append(end_date)
    return dates