import numpy as np
import dateutils
from datetime import date

from pyscnomics.econ.revenue import Lifting


def get_unrecovered_cost(depreciation: np.ndarray,
                         non_capital: np.ndarray,
                         revenue: np.ndarray,
                         ftp_ctr: np.ndarray,
                         ftp_gov: np.ndarray,
                         ic: np.ndarray) -> np.ndarray:
    """
    A function to get the array of unrecovered cost.

    Parameters
    ----------
    depreciation: np.ndarray
        The array containing the depreciated expenditures.
    non_capital: : np.ndarray
        The array containing the non-capital expenditures.
        non_capital expenditures is consisting of:
        Intangible, Operating Expenditures (OPEX) and Abandonment Site and Restoration (ASR) Expenditures.
    revenue: np.ndarray
        The array containing the revenue.
    ftp_ctr: np.ndarray
        The array containing the Contractor's First Tranche Petroleum (FTP).
    ftp_gov: np.ndarray
        The array containing the Government's FTP.
    ic: np.ndarray
        The array containing the Paid Investment Credit (IC).
    Returns
    -------
    out: np.ndarray
        The array of Unrecovered Cost.
    """

    unrecovered_cost = np.cumsum(depreciation + non_capital) - np.cumsum(
        revenue - (ftp_ctr + ftp_gov) - ic
    )

    unrecovered_cost = np.where(unrecovered_cost >= 0, unrecovered_cost, 0)

    # Condition where there is no revenue but still there is depreciation + non-capital
    left_cost = np.where(np.logical_and((revenue - ftp_ctr - ftp_gov - ic) < depreciation + non_capital,
                                        unrecovered_cost == 0),
                         (depreciation + non_capital) - (revenue - ftp_ctr - ftp_gov - ic), 0)
    unrecovered_cost_final = unrecovered_cost + np.cumsum(left_cost)
    return unrecovered_cost_final


def get_cost_to_be_recovered(unrecovered_cost: np.ndarray) -> np.ndarray:
    """
    A function to get the array of cost to be recovered.

    Parameters
    ----------
    unrecovered_cost: np.ndarray
        The array containing the unrecovered cost.

    Returns
    -------
    out: np.ndarray
        The array of cost to be recovered.
    """
    ctr = np.concatenate((np.zeros(1), -np.diff(unrecovered_cost)))
    return np.where(ctr > 0, ctr, 0)


def get_cost_to_be_recovered_after_tf(unrecovered_cost: np.ndarray,
                                      transferred_cost) -> np.ndarray:
    """
    A function to get the array of cost to be recovered.

    Parameters
    ----------
    unrecovered_cost: np.ndarray
        The array containing the unrecovered cost.
    transferred_cost

    Returns
    -------
    out: np.ndarray
        The array of cost to be recovered.
    """
    ctr = np.concatenate((np.zeros(1), -np.diff(unrecovered_cost - transferred_cost)))
    result = np.where(ctr > 0, ctr, 0)

    return result


def get_transfer(gas_unrecovered: np.ndarray,
                 oil_unrecovered: np.ndarray,
                 gas_ets_pretransfer: np.ndarray,
                 oil_ets_pretransfer: np.ndarray) -> tuple:
    """
    A function to get the transferred cost between oil and gas.

    Parameters
    ----------
    gas_unrecovered: np.ndarray
        The array containing the unrecovered cost from gas.
    oil_unrecovered: np.ndarray
        The array containing the unrecovered cost from oil.
    gas_ets_pretransfer: np.ndarray
        The array containing the gas's Equity To be Split (ETS) before transfer.
    oil_ets_pretransfer: np.ndarray
        The array containing the oil's Equity To be Split (ETS) before transfer.

    Returns
    -------
    out: tuple
        trf2oil: np.ndarray
            The transferred cost from gas to oil.
        trf2gas: np.ndarray
            The transferred cost from oil to gas.
    """

    trf2oil = np.zeros_like(oil_unrecovered)
    trf2gas = np.zeros_like(gas_unrecovered)

    # Transfer to oil
    combined_condition_oil = np.logical_and(
        np.greater(oil_unrecovered, 0), np.equal(gas_unrecovered, 0)
    )

    oil_indices = np.argwhere(combined_condition_oil)

    if np.size(oil_indices) > 0:
        trf2oil[oil_indices] = np.minimum(
            gas_ets_pretransfer[oil_indices], oil_unrecovered[oil_indices]
        )

    # Transfer to gas
    combined_condition_gas = np.logical_and(
        np.equal(oil_unrecovered, 0), np.greater(gas_unrecovered, 0)
    )

    indices_gas = np.argwhere(combined_condition_gas)

    if np.size(indices_gas) > 0:
        trf2gas[indices_gas] = np.minimum(
            oil_ets_pretransfer[indices_gas], gas_unrecovered[indices_gas]
        )

    return trf2oil, trf2gas


def get_unrec_cost_after_tf(depreciation,
                            non_capital,
                            revenue,
                            ftp_ctr,
                            ftp_gov,
                            ic,
                            transferred_cost):
    unrecovered_cost = np.cumsum(depreciation + non_capital) - np.cumsum(
        revenue - (ftp_ctr + ftp_gov) - ic
    )

    unrecovered_cost = np.where(unrecovered_cost >= 0, unrecovered_cost - transferred_cost, 0)

    # Condition where there is no revenue but still there is depreciation + non-capital
    left_cost = np.where(np.logical_and((revenue - ftp_ctr - ftp_gov - ic) < depreciation + non_capital,
                                        unrecovered_cost == 0),
                         (depreciation + non_capital) - (revenue - ftp_ctr - ftp_gov - ic) - transferred_cost, 0)
    unrecovered_cost_final = unrecovered_cost + np.cumsum(left_cost)
    return unrecovered_cost_final


def get_ets_after_transfer(ets_before_transfer: np.ndarray,
                           trfto: np.ndarray,
                           unrecovered_after_transfer: np.ndarray):
    """
    A function to get the Equity To be Split (ETS) after transfer.

    Parameters
    ----------
    ets_before_transfer: np.ndarray
        The array containing the ETS before transfer.
    trfto
        The array containing the transferred cost.
    unrecovered_after_transfer
        The array containing the unrecovered cost after transfer.

    Returns
    -------
    ets_after_transfer: np.ndarray
        The array of ETS after transfer.
    """

    ets_after_transfer = np.zeros_like(ets_before_transfer)

    indices = np.equal(unrecovered_after_transfer, 0)

    if np.size(indices) > 0:
        ets_after_transfer[indices] = ets_before_transfer[indices] + trfto[indices]

    return ets_after_transfer


# def get_ets_after_transfer(ets_before_transfer: np.ndarray,
#                            trfto: np.ndarray,
#                            trffrom: np.ndarray,
#                            unrecovered_after_transfer: np.ndarray):
#     """
#     A function to get the Equity To be Split (ETS) after transfer.
#
#     Parameters
#     ----------
#     ets_before_transfer: np.ndarray
#         The array containing the ETS before transfer.
#     trfto
#         The array containing the transferred cost.
#     unrecovered_after_transfer
#         The array containing the unrecovered cost after transfer.
#
#     Returns
#     -------
#     ets_after_transfer: np.ndarray
#         The array of ETS after transfer.
#     """
#
#     ets_after_transfer = np.zeros_like(ets_before_transfer)
#
#     indices = np.equal(unrecovered_after_transfer, 0)
#
#     if np.size(indices) > 0:
#         ets_after_transfer[indices] = ets_before_transfer[indices] - trfto[indices] + trffrom[indices]
#
#     return ets_after_transfer


def get_dmo(onstream_date: date,
            start_date: date,
            project_years: np.ndarray,
            dmo_holiday_duration: int,
            dmo_volume_portion: float,
            dmo_fee_portion: float,
            lifting: Lifting,
            price: np.ndarray,
            ctr_pretax_share: float,
            unrecovered_cost: np.ndarray,
            is_dmo_end_weighted) -> tuple:
    """
    A function to get the array of Domestic Market Obligation (DMO).

    Parameters
    ----------
    onstream_date: date
        The onstream date.
    start_date: date
        The start date of the contract.
    project_years: np.ndarray
        The array containing the contract years from the beginning until the end of the contract.
    dmo_holiday_duration: int
        The duration of the DMO holiday.
    dmo_volume_portion: float
        The DMO volume portion.
    dmo_fee_portion: float
        The DMO fee portion.
    lifting: Lifting
        The Lifting class of the produced fluid.
    price: np.ndarray
        The Weighted Average Price (WAP) price of the produced fluid.
    ctr_pretax_share: float
        The Contractor PreTax.
    unrecovered_cost: np.ndarray
        The array containing the Unrecovered Cost.
    is_dmo_end_weighted: bool
        The condition of whether DMO is weighted or not.

    Returns
    -------
    out: tuple
        dmo_volume: np.ndarray
            The array of DMO Volume.
        dmo_fee: np.ndarray
            The array of DMO Fee Volume.
        ddmo: np.ndarray
            The array of Difference DMO Fee.
    """

    # DMO end date
    dmo_end_date = onstream_date + dateutils.relativedelta(
        months=dmo_holiday_duration
    )

    # Identify position of dmo start year in project years array
    dmo_indices = onstream_date.year - start_date.year

    # Calculate DMO volume
    dmo_holiday = np.where(project_years >= dmo_end_date.year, False, True)
    dmo_volume = dmo_volume_portion * lifting.lifting_rate_arr() * ctr_pretax_share
    dmo_fee = np.where(np.logical_and(unrecovered_cost == 0, ~dmo_holiday),
                       dmo_fee_portion * price * dmo_volume,
                       dmo_volume * price)

    # Weighted dmo fee condition if the period of dmo is ended in the middle of the year
    if unrecovered_cost[dmo_indices] > 0 and is_dmo_end_weighted:
        dmo_fee[dmo_indices] = (
                    dmo_end_date.month / 12 * price[dmo_indices] * dmo_volume[dmo_indices] +
                    (1 - dmo_end_date.month / 12) * dmo_volume[dmo_indices] *
                    dmo_fee_portion * price[dmo_indices])

    # Calculate Net DMO
    ddmo = (dmo_volume * price) - dmo_fee

    return dmo_volume, dmo_fee, ddmo
