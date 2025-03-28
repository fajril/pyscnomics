"""
Calculate depreciation.
"""

import numpy as np
import pandas as pd


class DepreciationException(Exception):
    """Raise an exception error for a misuse of depreciation method"""

    pass


class UnitOfProductionException(Exception):
    """Exception to be raised for a misuse of unit of production method"""

    pass


def straight_line_depreciation_rate(
    cost: float, salvage_value: float, useful_life: int, depreciation_len: int = 0
) -> np.ndarray:
    """
    Calculate the straight-line depreciation charge for each period.

    Parameters
    ----------
    cost : float
        Cost of the asset.
    salvage_value : float
        Remaining value after depreciation.
    useful_life : int
        Duration for depreciation.
    depreciation_len : int, optional
        Length of the depreciation charge array beyond the useful life.
        The extended values will be set to zero. (default: 0)

    Returns
    -------
    depreciation_charge : numpy.ndarray
        The straight-line depreciation charge for each period.

    Notes
    -----
    The straight-line depreciation method allocates an equal amount of depreciation charge
    for each period over the useful life of an asset. The depreciation rate is calculated as:

    depreciation_rate = (cost - salvage_value) / useful_life

    The depreciation charge is then repeated for each period.

    If the `depreciation_len` is greater than the `useful_life`, the depreciation charge array
    is extended with zero values for the additional periods.
    """

    depreciation_rate = (cost - salvage_value) / useful_life
    depreciation_charge = np.repeat(depreciation_rate, useful_life)

    # Extend the depreciation charge array beyond useful life if needed
    if depreciation_len > useful_life:
        extension = np.zeros(int(depreciation_len - useful_life))
        depreciation_charge = np.concatenate((depreciation_charge, extension))

    return depreciation_charge


def straight_line_book_value(
    cost: float, salvage_value: float, useful_life: int, depreciation_len: int = 0
) -> np.ndarray:
    """
    Calculate the book value of an asset over time using the straight-line depreciation method.

    Parameters
    ----------
    cost : float
        Cost of the asset.
    salvage_value : float
        Remaining value after depreciation.
    useful_life : int
        Duration for depreciation.
    depreciation_len : int, optional
        Length of the book value array beyond the useful life.
        The extended values will be calculated based on the depreciation charge. (default: 0)

    Returns
    -------
    book_value : numpy.ndarray
        The book value of the asset for each period.

    Notes
    -----
    The straight-line depreciation method allocates an equal amount of depreciation charge
    for each period over the useful life of an asset. The depreciation charge is calculated
    using the `straight_line_depreciation_rate` function.

    The book value of the asset is calculated as the initial cost minus the cumulative sum of
    the depreciation charge for each period.

    If the `depreciation_len` is greater than the `useful_life`, the book value array is extended
    by calculating the remaining book values based on the depreciation charge.
    """

    # Calculate depreciation charge from function straight_line_depreciation_rate()
    depreciation_charge = straight_line_depreciation_rate(
        cost=cost,
        salvage_value=salvage_value,
        useful_life=useful_life,
        depreciation_len=depreciation_len,
    )

    # Calculate book value
    book_value = cost - np.cumsum(depreciation_charge)

    return book_value


def declining_balance_depreciation_rate(
    cost: float,
    salvage_value: float,
    useful_life: int,
    decline_factor: float = 2,
    depreciation_len: int = 0,
) -> np.ndarray:
    """
    Calculates the declining balance depreciation schedule for an asset.

    Parameters
    ----------
    cost : float
        The initial cost of the asset.
    salvage_value : float
        The residual value of the asset at the end of its useful life.
    useful_life : int
        The total useful life of the asset in years.
    decline_factor : float, optional
        The factor by which the depreciation is accelerated.
        Default is 2 (double-declining balance method).
    depreciation_len : int, optional
        The total duration for which depreciation should be calculated.
        If greater than `useful_life`, the depreciation schedule is extended
        with zero values. Default is 0, meaning it is set to `useful_life`.

    Returns
    -------
    np.ndarray
        An array representing the depreciation charge for each year of the asset's useful life.

    Notes
    -----
    - The function ensures that the sum of depreciation charges does not exceed
      `cost - salvage_value`.
    - If the depreciation exceeds this limit, the excess is adjusted in the final periods.
    - If `depreciation_len` is greater than `useful_life`, the schedule is extended with zeros.
    """

    # Exponent value
    periods = np.arange(1, int(useful_life), 1, dtype=np.int64)

    # Depreciation factor
    depreciation_factor = decline_factor / useful_life

    # Depreciation charge
    depreciation_charge = (
            depreciation_factor * cost * np.power(1 - depreciation_factor, periods - 1)
    )

    # Depreciation charge reaches the salvage value
    if depreciation_charge.sum() > (cost - salvage_value):
        remaining_depreciation = cost - salvage_value - np.cumsum(depreciation_charge)
        remaining_depreciation_modified = (
            np.where(remaining_depreciation < 0, 0, remaining_depreciation)
        )
        remaining_depreciation_unique_sum = np.sum(np.unique(remaining_depreciation_modified))

        # Depreciation charge is paid off since the first year
        if remaining_depreciation_unique_sum == 0:
            depreciation_charge_new = np.zeros_like(depreciation_charge)
            depreciation_charge_new[0] = depreciation_charge[0] + remaining_depreciation[0]
            depreciation_charge = depreciation_charge_new.copy()

        # Depreciation charge is paid off after the first year
        elif remaining_depreciation_unique_sum > 0:
            idx = np.argmin(remaining_depreciation_modified)
            depreciation_charge[idx] = remaining_depreciation[idx - 1]
            depreciation_charge[idx + 1:] = 0

        else:
            raise DepreciationException(
                f"Cannot have a negative value in variable remaining_depreciation_unique_sum: "
                f"{remaining_depreciation_unique_sum}"
            )

    # Add the remaining charge as the last element of depreciation_charge array
    depreciation_charge = np.concatenate(
        (
            depreciation_charge,
            cost - salvage_value - np.sum(depreciation_charge, keepdims=True)
        )
    )

    # Extend the depreciation charge array beyond useful life if
    # project duration is longer than useful life
    if depreciation_len > useful_life:
        extension = np.zeros(int(depreciation_len - useful_life))
        depreciation_charge = np.concatenate((depreciation_charge, extension))

    return depreciation_charge


def declining_balance_book_value(
    cost: float,
    salvage_value: float,
    useful_life: int,
    decline_factor: int = 1,
    depreciation_len: int = 0,
) -> np.ndarray:
    """
    Calculate the net book value of a depreciated asset using the Decline Balance Method.

    Parameters
    ----------
    cost : float
        Cost of the asset.
    salvage_value : float
        Remaining value after depreciation.
    useful_life : int
        Duration for depreciation.
    decline_factor : int, optional
        Depreciation factor. Set to 1 for Decline Balance or 2 for Double Decline Balance.
        (default: 1)
    depreciation_len : int, optional
        Length of the net book value array beyond its useful life.
        The extended values will be set using the salvage value. (default: 0)

    Returns
    -------
    book_value : numpy.ndarray
        The book value of the depreciated asset.

    Notes
    -----
    The book value (V_t) at time t is calculated using the formula:

    .. math::

        V_t = (1 - d)^t \times V_0

    where:
    - V_t is the net book value at time t,
    - d is the depreciation factor per year,
    - t is the time in years, and
    - V_0 is the initial cost of the asset.
    """
    depreciation_charge = declining_balance_depreciation_rate(
        cost=cost,
        salvage_value=salvage_value,
        useful_life=useful_life,
        decline_factor=decline_factor,
        depreciation_len=depreciation_len,
    )
    book_value = cost - np.cumsum(depreciation_charge)

    return book_value


def psc_declining_balance_depreciation_rate(
    cost: float,
    useful_life: float,
    depreciation_factor: float = 0.5,
    depreciation_len: int = 0,
) -> np.ndarray:
    """
    Calculate the declining balance depreciation charges following
    psc declining balance method.

    Parameters:
    -----------
    cost : float
        Initial cost of the asset.

    useful_life : float
        Total useful life of the asset in periods.

    depreciation_factor : float, optional
        Depreciation factor for declining balance calculation (default is 0.5).

    depreciation_len : int, optional
        Length of the resulting depreciation charge array (default is 0).
        If specified and greater than the calculated length, the array will be extended with zeros.

    Returns:
    --------
    depreciation_charge: np.ndarray
        An array containing the calculated depreciation charges for each period.

    Notes:
    ------
    The value of depreciation_factor must fall within the following interval:
    0 <= depreciation_factor <= 1.

    A value that falls outside the prescribed interval is in contrast with the logical
    concept of psc declining balance.
    """

    if depreciation_factor > 1 or depreciation_factor < 0:
        raise DepreciationException(
            f"The value of depreciation_factor must fall within the following interval: "
            f"0 <= depreciation_factor <= 1"
        )

    # Create an array of exponents
    periods = np.arange(1, int(useful_life), 1, dtype=np.int64)

    # Calculate depreciation charge
    depreciation_charge = (
        depreciation_factor * cost * np.power(1 - depreciation_factor, periods - 1)
    )

    # Modify depreciation charge; specify the last element as the remaining cost
    depreciation_charge = np.concatenate(
        (
            depreciation_charge,
            cost - np.sum(depreciation_charge, keepdims=True)
        )
    )

    # Modify depreciation charge; accounting for project duration
    if depreciation_len > len(depreciation_charge):
        extension = np.zeros(int(depreciation_len - len(depreciation_charge)))
        depreciation_charge = np.concatenate((depreciation_charge, extension))

    return depreciation_charge


def psc_declining_balance_book_value(
    cost: float,
    useful_life: float,
    depreciation_factor: float = 0.5,
    depreciation_len: int = 0,
) -> np.ndarray:
    """
    Calculate the book value of an asset over time using declining balance depreciation method.

    Parameters:
    -----------
    cost : float
        Initial cost of the asset.

    useful_life : float
        Total useful life of the asset in periods.

    depreciation_factor : float, optional
        Depreciation factor for declining balance calculation (default is 0.5).

    depreciation_len : int, optional
        Length of the resulting book value array (default is 0).
        If specified and greater than the calculated length, the array will be extended.

    Returns:
    --------
    book_value: np.ndarray
        An array containing the calculated book values of the asset over time.

    Notes:
    ------
    This function calculates the book value of an asset over time using the declining balance
    depreciation method. The book value at each period is calculated as the difference between
    the initial cost and the cumulative sum of depreciation charges.

    If `depreciation_len` is provided and is greater than the calculated length,
    the resulting array is extended.
    """

    depreciation_charge = psc_declining_balance_depreciation_rate(
        cost=cost,
        useful_life=useful_life,
        depreciation_factor=depreciation_factor,
        depreciation_len=depreciation_len,
    )

    book_value = cost - np.cumsum(depreciation_charge)

    return book_value


def unit_of_production_rate(
    start_year_project: int,
    cost: float,
    prod: np.ndarray,
    prod_year: np.ndarray,
    salvage_value: float = 0.0,
    amortization_len: int = 0,
) -> np.ndarray:
    """
    Calculates the unit of production amortization schedule based on production data.

    Parameters
    ----------
    start_year_project : int
        The start year of the project.
    cost : float
        The initial cost of the asset.
    prod : np.ndarray
        An array representing production values for each year.
    prod_year : np.ndarray
        An array representing corresponding years for the production values.
    salvage_value : float, optional
        The residual value of the asset at the end of its useful life. Default is 0.0.
    amortization_len : int, optional
        The total duration for which amortization should be calculated.
        If greater than the number of unique years, the amortization schedule is extended
        with zero values. Default is 0.

    Returns
    -------
    np.ndarray
        An array representing the amortization charge for each year.

    Notes
    -----
    - The function ensures that the amortization charge is allocated correctly based
      on production values.
    - It prevents depreciation exceeding `cost - salvage_value` by adjusting the
      final values.
    - The function calculates amortization charge using 2 * unit of production method.
    - If `amortization_len` is greater than the number of unique production years,
      the schedule is extended with zeros.
    """

    # Prepare parameter prod
    if not isinstance(prod, np.ndarray):
        raise UnitOfProductionException(
            f"Parameter prod must be given as a numpy.ndarray datatype, "
            f"not {prod.__class__.__qualname__}."
        )

    else:
        prod_nan_id = np.argwhere(pd.isna(prod)).ravel()
        if len(prod_nan_id) > 0:
            prod[prod_nan_id] = np.zeros(len(prod_nan_id))

    prod = prod.astype(np.float64)

    # Prepare parameter prod_year
    if not isinstance(prod_year, np.ndarray):
        raise UnitOfProductionException(
            f"Parameter prod_year must be given as a numpy.ndarray datatype, "
            f"not {prod_year.__class__.__qualname__}"
        )

    else:
        prod_year_nan_sum = np.sum(pd.isna(prod_year))
        if prod_year_nan_sum > 0:
            raise UnitOfProductionException(
                f"Incomplete data for parameter 'prod_year'"
            )

    prod_year = prod_year.astype(int)

    # Raise an exception for unequal length of arrays: prod and prod_year
    if len(prod) != len(prod_year):
        raise UnitOfProductionException(
            f"Unequal number of arrays: "
            f"prod: {len(prod)}, "
            f"prod_year: {len(prod_year)}."
        )

    # Raise an exception if prod_year is before the start year of the project
    if min(prod_year) < start_year_project:
        raise UnitOfProductionException(
            f"Production year ({min(prod_year)}) is before the start year "
            f"of the project ({start_year_project})."
        )

    # Raise an exception if prod_year is after the end year of the project
    if max(prod_year) > int(start_year_project + amortization_len - 1):
        raise UnitOfProductionException(
            f"Production year ({max(prod_year)}) is after the end year "
            f"of the project ({int(start_year_project + amortization_len - 1)})"
        )

    # Raise an exception if salvage_value is larger than the associated cost
    if salvage_value > cost:
        raise UnitOfProductionException(
            f"Salvage value ({salvage_value}) is larger than the associated cost ({cost})."
        )

    # Specify cumulative production
    cum_prod = np.sum(prod, dtype=np.float64)

    # Raise an exception for zero or negative value of cum_prod
    if cum_prod <= 0:
        raise UnitOfProductionException(
            f"Inappropriate value of production data. "
            f"The sum of yearly_prod ({prod}) is {cum_prod}."
        )

    # Calculate amortization charge (1 * UOP)
    amortization_charge = np.divide(prod, cum_prod, where=cum_prod != 0) * (cost - salvage_value)

    # Calculate amortization charge (2 * UOP)
    amortization_charge = 2.0 * amortization_charge

    # Calculate remaining amortization
    remaining_amortization = cost - salvage_value - np.cumsum(amortization_charge)
    remaining_amortization_modified = np.where(
        remaining_amortization < 0, 0, remaining_amortization
    )
    remaining_amortization_unique_sum = np.sum(np.unique(remaining_amortization_modified))

    # Amortization charge is paid off since the first year
    if remaining_amortization_unique_sum == 0:
        amortization_charge_new = np.zeros_like(amortization_charge)
        amortization_charge_new[0] = amortization_charge[0] + remaining_amortization[0]
        amortization_charge = amortization_charge_new.copy()

    # Amortization charge is paid off after the first year
    elif remaining_amortization_unique_sum > 0:
        idx = np.argmin(remaining_amortization_modified)
        amortization_charge[idx] = remaining_amortization[idx - 1]
        amortization_charge[idx + 1:] = 0

    else:
        raise UnitOfProductionException(
            f"Cannot have a negative value in variable remaining_amortization_unique_sum: "
            f"{remaining_amortization_unique_sum}"
        )

    # Allocate amortization_charge according to their associated year
    amortization_charge = np.bincount(prod_year - start_year_project, weights=amortization_charge)

    # Modify amortization charge, accounting for project duration
    if amortization_len > len(amortization_charge):
        extension = np.zeros(amortization_len - len(amortization_charge))
        amortization_charge = np.concatenate((amortization_charge, extension))

    return amortization_charge


def unit_of_production_book_value(
    start_year_project: int,
    cost: float,
    prod: np.ndarray,
    prod_year: np.ndarray,
    salvage_value: float = 0.0,
    amortization_len: int = 0,
) -> np.ndarray:
    """
    Calculate the book value of an asset over time using unit of production method.

    Parameters
    ----------
    start_year_project : int
        The starting year of the project.
    cost : float
        Initial cost of the asset.
    prod : np.ndarray
        Hydrocarbon production starting from onstream year.
    prod_year : np.ndarray
        An array of years corresponding to the hydrocarbon production.
    salvage_value : float, optional
        The salvage value of the project at the end of its life (default is 0.0).
    amortization_len : int, optional
        The length of the amortization charge array (default is 0).
        If specified and greater than the calculated length, the array will be extended with zeros.

    Returns
    --------
    book_value: np.ndarray
        An array containing the calculated book values of the asset over time.
    """

    amortization_charge = unit_of_production_rate(
        start_year_project=start_year_project,
        cost=cost,
        prod=prod,
        prod_year=prod_year,
        salvage_value=salvage_value,
        amortization_len=amortization_len,
    )

    book_value = cost - np.cumsum(amortization_charge)

    return book_value
