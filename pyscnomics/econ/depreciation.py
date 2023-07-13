import numpy as np


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

    Examples
    --------
    >>> book_value = straight_line_book_value(cost=1000, salvage_value=100, useful_life=5)
    >>> print(book_value)
    [ 820.  640.  460.  280.  100.]
    """
    depreciation_charge = straight_line_depreciation_rate(
        cost=cost,
        salvage_value=salvage_value,
        useful_life=useful_life,
        depreciation_len=depreciation_len,
    )
    book_value = cost - np.cumsum(depreciation_charge)
    return book_value


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

    Examples
    --------
    >>> depreciation_charge = straight_line_depreciation_rate(cost=1000, salvage_value=100, useful_life=5)
    >>> print(depreciation_charge)
    [180. 180. 180. 180. 180.]
    """
    depreciation_rate = (cost - salvage_value) / useful_life
    depreciation_charge = np.repeat(depreciation_rate, useful_life)

    # Extend the depreciation charge array beyond useful life if needed
    if depreciation_len > useful_life:
        extension = np.zeros(depreciation_len - useful_life)
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

    Examples
    --------
    >>> book_value = declining_balance_book_value(cost=200_000, salvage_value=25_000, useful_life=5, decline_factor=2, net_book_len=10)
    >>> print(book_value)
    [120000.  72000.  43200.  25920.  25000.  25000.  25000.  25000.  25000.
    25000.]
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
    cost : float
        Cost of the asset.
    salvage_value : float
        Remaining value after depreciation.
    useful_life : int
        Duration for depreciation.
    decline_factor : float, optional
        Depreciation factor. Usually the value is between 1 and 2.
        When the value is set as 2, it is called DDB (default: 1)
    depreciation_len : int, optional
        Length of the net book value array beyond its useful life.
        The extended values will be set as zero. (default: 0)

    Returns
    -------
    depreciation_charge : numpy.ndarray
        The depreciation charge for each period.
    """
    periods = np.arange(useful_life)
    depreciation_rate = decline_factor / useful_life
    depreciation_charge = (
        depreciation_rate * cost * np.power(1 - depreciation_rate, periods)
    )

    # Handle the condition when depreciation charge reaches the salvage value
    if depreciation_charge.sum() > (cost - salvage_value):
        remaining_depreciation = cost - salvage_value - np.cumsum(depreciation_charge)
        remaining_depreciation = np.where(
            remaining_depreciation > 0, remaining_depreciation, 0
        )
        idx = np.argmin(remaining_depreciation)

        # Adjust the depreciation charge to take only the remainder when it reaches the salvage value
        depreciation_charge[idx] = (
            cost - salvage_value - np.cumsum(depreciation_charge)[idx - 1]
        )
        depreciation_charge[idx + 1 :] = 0

    # Extend the depreciation charge array beyond useful life if needed
    if depreciation_len > useful_life:
        extension = np.zeros(depreciation_len - useful_life)
        depreciation_charge = np.concatenate((depreciation_charge, extension))
    return depreciation_charge


def unit_of_production_rate(
    cost: float,
    salvage_value: float,
    resources: float,
    yearly_production: np.ndarray,
    decline_factor: float = 2,
    amortization_len: int = 0,
) -> np.ndarray:
    """
    Calculates the amortization charge for each unit of production based on the unit of production method.

    Parameters
    ----------
    cost : float
        Cost of the resource or asset.
    salvage_value : float
        Estimated value of the resource or asset at the end of its useful life.
    resources : float
        Total amount of resources available.
    yearly_production : numpy.ndarray
        Array containing yearly production quantities.
    decline_factor : float, optional
        Factor determining the rate of decline in production. Default is 2.
    amortization_len : int, optional
        Length of the amortization charge array. Default is 0.

    Returns
    -------
    amortization_charge : numpy.ndarray
        Array of amortization charges corresponding to each unit of production.

    Notes
    -----
    The amortization charge for each unit of production is calculated using the following steps:
    1. Calculate the resources remaining over time
       based on the cumulative sum of yearly production quantities.
    2. Compute the amortization charge using the formula:
       amortization_charge = decline_factor * cost * yearly_production / resources_over_time.
    3. Adjust the amortization charges if the sum exceeds the cost minus salvage value
       to ensure the total amortization matches the cost minus salvage value.
    4. If the amortization charge array is shorter than the specified amortization length,
       extend the array with zeros.

    Examples
    --------
    >>> cost = 10000
    >>> salvage_value = 2000
    >>> resources = 5000
    >>> yearly_production = np.array([1000, 800, 700, 600, 500])
    >>> amortization_charge = unit_of_production_book_value(cost, salvage_value, resources, yearly_production)
    >>> print(amortization_charge)
    [40.  40.  80. 120. 200.]
    """
    # TODO: fix the doctest with real result

    resources_over_time = resources - np.cumsum(yearly_production)
    amortization_charge = (
        decline_factor * cost * yearly_production / resources_over_time
    )

    if amortization_charge.sum() > (cost - salvage_value):
        remaining_amortization = cost - salvage_value - np.cumsum(amortization_charge)
        remaining_amortization = np.where(
            remaining_amortization > 0, remaining_amortization, 0
        )
        idx = np.argmin(remaining_amortization)
        amortization_charge[idx] = (
            cost - salvage_value - np.cumsum(amortization_charge)[idx - 1]
        )
        amortization_charge[idx + 1 :] = 0

    if amortization_charge.size < amortization_len:
        extension = np.zeros(amortization_len - amortization_charge.size)
        amortization_charge = np.concatenate(amortization_charge, extension)
    return amortization_charge


def unit_of_production_book_value(
    cost: float,
    salvage_value: float,
    resources: float,
    yearly_production: np.ndarray,
    decline_factor: float = 2,
    amortization_len: int = 0,
) -> np.ndarray:
    """
    Calculates the net book value of a resource or asset based on the unit of production method.

    Parameters
    ----------
    cost : float
        Cost of the resource or asset.
    salvage_value : float
        Estimated value of the resource or asset at the end of its useful life.
    resources : float
        Total amount of resources available.
    yearly_production : numpy.ndarray
        Array containing yearly production quantities.
    decline_factor : float, optional
        Factor determining the rate of decline in production. Default is 2.
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

    Examples
    --------
    >>> cost = 10000
    >>> salvage_value = 2000
    >>> resources = 5000
    >>> yearly_production = np.array([1000, 800, 700, 600, 500])
    >>> book_value = unit_of_production_book_value(cost, salvage_value, resources, yearly_production)
    >>> print(book_value)
    [ 9600.  8760.  7680.  6480.   4980.]
    """
    # TODO: fix the doctest with real result
    amortization_charge = unit_of_production_rate(
        cost=cost,
        salvage_value=salvage_value,
        resources=resources,
        yearly_production=yearly_production,
        decline_factor=decline_factor,
        amortization_len=amortization_len,
    )
    book_value = cost - np.cumsum(amortization_charge)
    return book_value
