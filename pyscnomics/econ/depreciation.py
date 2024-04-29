"""
Calculate depreciation.
"""

import numpy as np


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
    decline_factor: float = 1,
    depreciation_len: int = 0,
) -> np.ndarray:
    """
    Calculate the declining balance depreciation charge for each period.

    Parameters
    ----------
    cost: float
        Cost of the asset.
    salvage_value: float
        Remaining value after depreciation.
    useful_life: int
        Duration for depreciation.
    decline_factor: float, optional
        Depreciation factor. Usually the value is between 1 and 2.
        When the value is set as 2, it is called DDB (default: 1)
    depreciation_len: int, optional
        Length of the net book value array beyond its useful life.
        The extended values will be set as zero. (default: 0)

    Returns
    -------
    depreciation_charge : numpy.ndarray
        The depreciation charge for each period.
    """

    # Exponent value
    periods = np.arange(1, int(useful_life), 1, dtype=np.int_)

    # Depreciation factor
    depreciation_factor = decline_factor / useful_life

    # Depreciation charge
    depreciation_charge = (
        depreciation_factor * cost * np.power(1 - depreciation_factor, periods - 1)
    )

    # When depreciation charge reaches the salvage value
    if depreciation_charge.sum() > (cost - salvage_value):
        remaining_depreciation = cost - salvage_value - np.cumsum(depreciation_charge)
        remaining_depreciation = np.where(remaining_depreciation > 0, remaining_depreciation, 0)
        idx = np.argmin(remaining_depreciation)

        # Adjust the depreciation charge to take only the remainder when it reaches the salvage value
        depreciation_charge[idx] = (cost - salvage_value - np.cumsum(depreciation_charge)[idx - 1])
        depreciation_charge[idx + 1:] = 0

    # Add an element to array depreciation_charge, accounting the remaining unpaid cost
    depreciation_charge = np.concatenate(
        (
            depreciation_charge,
            cost - salvage_value - np.sum(depreciation_charge, keepdims=True),
        )
    )

    # Extend the depreciation charge array beyond useful life if project duration
    # is longer than useful life
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
    cost: float
        Cost of the asset.
    salvage_value: float
        Remaining value after depreciation.
    useful_life: int
        Duration for depreciation.
    decline_factor: int, optional
        Depreciation factor. Set to 1 for Decline Balance or 2 for Double Decline Balance.
        (default: 1)
    depreciation_len: int, optional
        Length of the net book value array beyond its useful life.
        The extended values will be set using the salvage value. (default: 0)

    Returns
    -------
    book_value: numpy.ndarray
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

    if depreciation_factor > 1:
        raise DepreciationException(
            f"The value of depreciation_factor must fall within the following interval: "
            f"0 <= depreciation_factor <= 1 "
        )

    elif depreciation_factor < 0:
        raise DepreciationException(
            f"The value of depreciation_factor must fall within the following interval: "
            f"0 <= depreciation_factor <= 1 "
        )

    else:
        # Create an array of exponents
        periods = np.arange(1, int(useful_life), 1, dtype=np.int_)

        # Calculate depreciation charge
        depreciation_charge = (
            depreciation_factor * cost * np.power(1 - depreciation_factor, periods - 1)
        )

        # Modify depreciation charge; specify the last element as the remaining (unpaid) cost
        depreciation_charge = np.concatenate(
            (depreciation_charge, cost - np.sum(depreciation_charge, keepdims=True))
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
    cost: float,
    cum_prod: float,
    yearly_prod: np.ndarray,
    production_period: int = None,
    salvage_value: float = 0.0,
    amortization_len: int = 0,
) -> np.ndarray:
    """
    Calculate amortization charge based on the unit of production method.

    Parameters:
    -----------
    cost: float
        Total cost of the project.
    cum_prod: float
        Cumulative production of the project.
    yearly_production: np.ndarray
        Array containing yearly production data.
    production_period: int
        Total production period of the project.
    salvage_value: float, optional
        Salvage value of the project (default is 0.).
    amortization_len: int, optional
        Length of amortization period (default is 0).

    Returns:
    --------
    np.ndarray
        Array containing yearly amortization charge.

    Raises:
    -------
    UnitOfProductionException
        If 'yearly_production' is not given as a numpy.ndarray datatype,
        or if the number of production data listed in 'yearly_production'
        does not match the prescribed production period,
        or if the sum of production data in 'yearly_production'
        exceeds the prescribed reserve.
    """
    # Raise exception if 'yearly_prod' is not given as a numpy.array datatype
    if not isinstance(yearly_prod, np.ndarray):
        raise UnitOfProductionException(
            f"Parameter yearly_production must be given as a numpy.ndarray datatype."
        )

    # Specify default value for production_period
    if production_period is None:
        production_period = len(yearly_prod)

    else:
        # Raise exception if the number of production data listed in 'yearly_prod'
        # does not match the prescribed production_period
        if len(yearly_prod) != production_period:
            raise UnitOfProductionException(
                f"The number of production data in 'yearly_production' "
                f"does not match the (prescribed) production period."
            )

    # Raise an exception if the sum of production data in 'yearly_prod'
    # exceeds the (prescribed) value of reserve
    if (yearly_prod.sum() - cum_prod) > 1E-5:
        raise UnitOfProductionException(
            f"Production data in 'yearly_production' exceeds the (prescribed) reserve."
        )

    # Calculate amortization charge
    amortization_charge = (yearly_prod / cum_prod) * (cost - salvage_value)

    # When the sum of amortization charge is less than (cost - salvage_value)
    if amortization_charge.sum() < (cost - salvage_value):
        remaining_amortization = cost - salvage_value - amortization_charge.sum()
        amortization_charge[-1] = amortization_charge[-1] + remaining_amortization

    # Extend the amortization charge array beyond useful life if project duration
    # is longer than production_period
    if amortization_len > production_period:
        extension = np.zeros(int(amortization_len - production_period))
        amortization_charge = np.concatenate((amortization_charge, extension))

    return amortization_charge


def unit_of_production_book_value(
    cost: float,
    cum_prod: float,
    yearly_prod: np.ndarray,
    production_period: int = None,
    salvage_value: float = 0.0,
    amortization_len: int = 0,
) -> np.ndarray:
    """
    Calculates the net book value of a resource or asset based on the unit of production method.

    Parameters
    ----------
    cost: float
        Cost of the resource or asset.
    cum_prod: float
        Cumulative production of the project.
    yearly_prod : numpy.ndarray
        Array containing yearly production quantities.
    production_period: int
        Total production period of the project.
    salvage_value : float
        Estimated value of the resource or asset at the end of its useful life. Default is 0.
    amortization_len : int, optional
        Length of the amortization charge array. Default is 0.

    Returns
    -------
    book_value : numpy.ndarray
        Array of net book values corresponding to each unit of production.

    Notes
    -----
    The net book value for each unit of production is calculated using the following steps:
    1. Calculate the amortization charge for each unit of production
       using the unit_of_production_rate function.
    2. Subtract the cumulative sum of amortization charges from the cost to get the net book value.
    """
    amortization_charge = unit_of_production_rate(
        cost=cost,
        cum_prod=cum_prod,
        yearly_prod=yearly_prod,
        production_period=production_period,
        salvage_value=salvage_value,
        amortization_len=amortization_len,
    )

    book_value = cost - np.cumsum(amortization_charge)

    return book_value
