"""
Calculate depreciation.
"""

import numpy as np
import numpy.typing as npt

from pyscnomics.input_validation import InputValidationError, engine_input_issue
from pyscnomics.validation_rules import (
    depreciation_factor_out_of_range,
    salvage_exceeds_cost,
)


class DepreciationException(Exception):
    """Raise an exception error for a misuse of depreciation method"""

    pass


class DepreciationInputError(  # pyright: ignore[reportUnsafeMultipleInheritance]
    DepreciationException, InputValidationError
):
    """Verified user-input depreciation error retaining ``DepreciationException``."""

    _input_code: str = "DEPRECIATION_INPUT_INVALID"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        InputValidationError.__init__(
            self,
            (engine_input_issue(code=code or self._input_code, message=message),),
        )


class UnitOfProductionException(Exception):
    """Exception to be raised for a misuse of unit of production method"""

    pass


class UnitOfProductionInputError(  # pyright: ignore[reportUnsafeMultipleInheritance]
    UnitOfProductionException, InputValidationError
):
    """Verified user-input UOP error retaining ``UnitOfProductionException``."""

    _input_code: str = "UNIT_OF_PRODUCTION_INPUT_INVALID"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        InputValidationError.__init__(
            self,
            (engine_input_issue(code=code or self._input_code, message=message),),
        )


def _as_float_array(value: object, parameter_name: str) -> npt.NDArray[np.float64]:
    """Validate an ndarray input and return a float64 working array."""

    if not isinstance(value, np.ndarray):
        raise UnitOfProductionException(
            f"Parameter '{parameter_name}' must be given as a numpy.ndarray, not {value.__class__.__qualname__}."
        )

    return np.asarray(value, dtype=np.float64)


def straight_line_depreciation_rate(
    cost: float,
    salvage_value: float,
    useful_life: int,
    depreciation_len: int = 0,
) -> npt.NDArray[np.float64]:
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
    cost: float,
    salvage_value: float,
    useful_life: int,
    depreciation_len: int = 0,
) -> npt.NDArray[np.float64]:
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
    return cost - np.cumsum(depreciation_charge)


def declining_balance_depreciation_rate(
    cost: float,
    salvage_value: float,
    useful_life: int,
    decline_factor: float = 2,
    depreciation_len: int = 0,
) -> npt.NDArray[np.float64]:
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
    depreciation_charge = np.asarray(
        depreciation_factor * cost * np.power(1 - depreciation_factor, periods - 1),
        dtype=np.float64,
    )

    # Depreciation charge reaches the salvage value
    if depreciation_charge.sum() > (cost - salvage_value):
        remaining_depreciation = cost - salvage_value - np.cumsum(depreciation_charge)
        remaining_depreciation_modified = np.where(
            remaining_depreciation < 0, 0, remaining_depreciation
        )
        is_fully_paid_mask = np.equal(remaining_depreciation_modified, 0)
        is_fully_paid = np.all(is_fully_paid_mask)

        # Depreciation charge is paid off since the first year
        if is_fully_paid:
            depreciation_charge_new = np.zeros_like(depreciation_charge)
            depreciation_charge_new[0] = (
                depreciation_charge[0] + remaining_depreciation[0]
            )
            depreciation_charge = depreciation_charge_new.copy()

        # Depreciation charge is paid off after the first year
        elif np.any(remaining_depreciation_modified > 0):
            idx = np.argmin(remaining_depreciation_modified)
            depreciation_charge[idx] = remaining_depreciation[idx - 1]
            depreciation_charge[idx + 1 :] = 0

        else:
            raise DepreciationException(
                "Cannot have a negative value in variable "
                + f"remaining_depreciation_unique_sum: {remaining_depreciation_modified.sum()}"
            )

    # Add the remaining charge as the last element of depreciation_charge array
    depreciation_charge = np.concatenate(
        (
            depreciation_charge,
            cost - salvage_value - np.sum(depreciation_charge, keepdims=True),
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
) -> npt.NDArray[np.float64]:
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
) -> npt.NDArray[np.float64]:
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

    if depreciation_factor_out_of_range(depreciation_factor):
        raise DepreciationInputError(
            "The value of depreciation_factor must fall within the following interval: "
            + "0 <= depreciation_factor <= 1",
            code="DEPRECIATION_FACTOR_OUT_OF_RANGE",
        )

    # Create an array of exponents
    periods = np.arange(1, int(useful_life), 1, dtype=np.int64)

    # Calculate depreciation charge
    depreciation_charge = (
        depreciation_factor * cost * np.power(1 - depreciation_factor, periods - 1)
    )

    # Modify depreciation charge; specify the last element as the remaining cost
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
) -> npt.NDArray[np.float64]:
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


def prepare_amortization(
    project_years: npt.NDArray[np.generic],
    prod: npt.NDArray[np.generic],
    cost: float,
    salvage_value: float,
    pis_year: int,
) -> None:
    """
    Validate and prepare inputs for unit-of-production amortization.

    This method performs defensive checks and preprocessing on project
    timeline, production, and cost inputs prior to amortization
    calculation. Missing production values are treated as zero, and all
    inputs are validated for consistency and economic feasibility.

    Parameters
    ----------
    project_years : np.ndarray
        Project year timeline.
    prod : np.ndarray
        Production volumes aligned with ``project_years``.
    cost : float
        Capital cost subject to amortization.
    salvage_value : float
        Residual value deducted from amortizable cost.
    pis_year : int
        Year when the asset is placed in service.

    Raises
    ------
    UnitOfProductionException
        If inputs are invalid, inconsistent, or economically infeasible.
    """

    # Validate "project_years"
    project_years_array = _as_float_array(project_years, "project_years")
    project_years_nan = np.isnan(project_years_array)
    if np.any(project_years_nan):
        raise UnitOfProductionException(
            f"Incomplete data for 'project_years': {project_years}."
        )

    # Validate "prod"
    prod_array = _as_float_array(prod, "prod")
    prod_nan_id = np.argwhere(np.isnan(prod_array)).ravel()
    if len(prod_nan_id) > 0:
        prod[prod_nan_id] = np.zeros(len(prod_nan_id))

    # Raise an error: unequal length of arrays
    if len(prod_array) != len(project_years_array):
        raise UnitOfProductionInputError(
            f"Unequal number of arrays: project_years: {len(project_years_array)}, prod: {len(prod_array)}.",
            code="UOP_ARRAY_LENGTH_MISMATCH",
        )

    # Raise an error: salvage value is larger than cost
    if salvage_exceeds_cost(salvage_value, cost):
        raise UnitOfProductionInputError(
            f"Salvage value ({salvage_value}) is larger than the associated cost ({cost}).",
            code="UOP_SALVAGE_EXCEEDS_COST",
        )

    # Raise an error: PIS year is before the start year of the project
    if pis_year < project_years.min():
        raise UnitOfProductionInputError(
            f"Cannot have PIS year ({pis_year}) before the start year of the project ({project_years.min()}).",
            code="UOP_PIS_YEAR_OUTSIDE_PROJECT",
        )

    # Raise an error: PIS year is after the end year of the project
    if pis_year > project_years.max():
        raise UnitOfProductionInputError(
            f"Cannot have PIS year ({pis_year}) after the end year of the project ({project_years.max()}).",
            code="UOP_PIS_YEAR_OUTSIDE_PROJECT",
        )


def unit_of_production_rate(
    project_years: npt.NDArray[np.generic],
    prod: npt.NDArray[np.generic],
    cost: float,
    salvage_value: float,
    pis_year: int,
) -> npt.NDArray[np.float64]:
    """
    Compute unit-of-production (UOP) amortization charges.

    This method calculates an accelerated UOP amortization schedule based
    on production from the placed-in-service (PIS) year onward. The total
    amortizable cost (``cost - salvage_value``) is allocated proportionally
    to production and scaled by a factor of two. Any remaining balance is
    settled by adjusting the final applicable year.

    Parameters
    ----------
    project_years : np.ndarray
        Project year timeline.
    prod : np.ndarray
        Production volumes aligned with ``project_years``.
    cost : float
        Capital cost subject to amortization.
    salvage_value : float
        Residual value deducted from the amortizable base.
    pis_year : int
        Year when the asset is placed in service.

    Returns
    -------
    np.ndarray
        Amortization charge per project year, aligned with
        ``project_years``.

    Raises
    ------
    UnitOfProductionException
        If input validation fails or amortization termination is invalid.
    """

    # Validate input data
    prepare_amortization(
        project_years=project_years,
        prod=prod,
        cost=cost,
        salvage_value=salvage_value,
        pis_year=pis_year,
    )
    project_years_array = _as_float_array(project_years, "project_years")
    prod_array = _as_float_array(prod, "prod")

    # Identify the location of PIS year in array project years (in the form of index)
    match_mask = np.equal(project_years_array, pis_year)
    match = np.flatnonzero(match_mask)
    if match.size != 1:
        raise UnitOfProductionInputError(
            f"PIS year ({pis_year}) must appear exactly once in project_years.",
            code="UOP_PIS_YEAR_NOT_UNIQUE",
        )
    match_indices: list[int] = list(match)
    pis_idx = match_indices[0]

    # Calculate cumulative production starting from PIS year index
    prod_values: list[float] = list(prod_array)
    cum_prod = sum(prod_values[pis_idx:])

    if cum_prod == 0:
        amortization_charge = np.zeros_like(project_years_array, dtype=float)
        return amortization_charge

    # Calculate prod / cum_prod
    ratio = np.zeros_like(project_years_array, dtype=np.float64)
    ratio[pis_idx:] = prod_array[pis_idx:] / cum_prod

    # Calculate amortization charge = 2 * Unit of Production
    amortization_charge = 2.0 * (ratio * (cost - salvage_value))

    # Calculate remaining amortization
    remaining_amortization = cost - salvage_value - np.cumsum(amortization_charge)
    remaining_amortization_modified = np.where(
        remaining_amortization < 0, 0, remaining_amortization
    )
    is_fully_paid_mask = np.equal(remaining_amortization_modified, 0)
    is_fully_paid = np.all(is_fully_paid_mask)

    # Amortization charge is paid off since the first year
    if is_fully_paid:
        amortization_charge_new = np.zeros_like(project_years_array, dtype=float)
        amortization_charge_new[0] = amortization_charge[0] + remaining_amortization[0]
        amortization_charge = amortization_charge_new.copy()

    # Amortization charge is paid off after the first year
    else:
        idx = np.argmin(remaining_amortization_modified)

        if idx == 0:
            raise UnitOfProductionException("Invalid amortization termination index.")

        amortization_charge[idx] = remaining_amortization[idx - 1]
        amortization_charge[idx + 1 :] = 0

    return amortization_charge


def unit_of_production_book_value(
    project_years: npt.NDArray[np.generic],
    prod: npt.NDArray[np.generic],
    cost: float,
    salvage_value: float,
    pis_year: int,
) -> npt.NDArray[np.float64]:
    """
    Compute book value under the unit-of-production method.

    This method derives the asset book value over time by applying the
    unit-of-production amortization schedule and cumulatively deducting
    amortization charges from the original cost.

    Parameters
    ----------
    project_years : np.ndarray
        Project year timeline.
    prod : np.ndarray
        Production volumes aligned with ``project_years``.
    cost : float
        Capital cost subject to amortization.
    salvage_value : float
        Residual value (passed through to amortization calculation).
    pis_year : int
        Year when the asset is placed in service.

    Returns
    -------
    np.ndarray
        Book value per project year, aligned with ``project_years``.
    """

    amortization_charge = unit_of_production_rate(
        project_years=project_years,
        prod=prod,
        cost=cost,
        salvage_value=salvage_value,
        pis_year=pis_year,
    )

    book_value = cost - np.cumsum(amortization_charge)

    return book_value


"""
Prior functions (deprecated):


def prepare_amortization(
    prod: np.ndarray,
    prod_year: np.ndarray,
    project_years: np.ndarray,
    cost: float,
    salvage_value: float,
) -> None:

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
    if min(prod_year) < project_years.min():
        raise UnitOfProductionException(
            f"Production year ({min(prod_year)}) is before the start year "
            f"of the project ({project_years.min()})."
        )

    # Raise an exception if prod_year is after the end year of the project
    if max(prod_year) > project_years.max():
        # if max(prod_year) > int(start_year_project + amortization_len - 1):
        raise UnitOfProductionException(
            f"Production year ({max(prod_year)}) is after the end year "
            f"of the project ({int(project_years.max())})"
        )

    # Raise an exception if salvage_value is larger than the associated cost
    if salvage_value > cost:
        raise UnitOfProductionException(
            f"Salvage value ({salvage_value}) is larger than the associated cost ({cost})."
        )
        
        
def align_amortization(
    amortization_charge: np.ndarray,
    project_years: np.ndarray,
    target_year: int,
) -> np.ndarray:
    
    # First index where amortization_charge > 0
    idx_amort = np.flatnonzero(amortization_charge > 0).min()

    # Find the index of the target year
    idx_target = int(np.flatnonzero(project_years == target_year))

    # Place the amortization charges starting at the target index
    capture_amort = amortization_charge[idx_amort:]
    new_amort = np.zeros_like(project_years, dtype=np.float64)
    new_amort[idx_target:int(idx_target + len(capture_amort))] = capture_amort

    return new_amort
    
    
def unit_of_production_rate(
    project_years: np.ndarray,
    approval_year: int,
    cost: float,
    prod: np.ndarray,
    prod_year: np.ndarray,
    # initial_amortization_year: InitialYearAmortizationIncurred,
    salvage_value: float,
) -> np.ndarray:

    # Prepare the associated parameters
    prepare_amortization(
        prod=prod,
        prod_year=prod_year,
        project_years=project_years,
        cost=cost,
        salvage_value=salvage_value,
    )

    # Specify cumulative production
    cum_prod = np.sum(prod, dtype=np.float64)

    # If cum_prod is zero, return zero array of "amortization_charge"
    if cum_prod == 0:
        amortization_charge = np.zeros_like(project_years, dtype=float)
        return amortization_charge

    else:
        # Raise an exception for negative value of cum_prod
        if cum_prod < 0:
            raise UnitOfProductionException(
                f"Cannot have a negative value ({cum_prod}) as cumulative production"
            )

        # Calculate amortization charge (1 * UOP)
        amortization_charge = np.divide(prod, cum_prod) * (cost - salvage_value)

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

        # Check whether amortization charge array is all zero
        if np.all(amortization_charge == 0):
            amortization_charge = amortization_charge

        else:
            # Allocate amortization charge
            amortization_charge = np.bincount(
                prod_year - project_years.min(), weights=amortization_charge
            )

            # Allocate amortization_charge according to their associated year
            if initial_amortization_year == InitialYearAmortizationIncurred.ONSTREAM_YEAR:
                amortization_charge = np.bincount(
                    prod_year - project_years.min(), weights=amortization_charge
                )

            elif initial_amortization_year == InitialYearAmortizationIncurred.APPROVAL_YEAR:
                amortization_charge = align_amortization(
                    amortization_charge=amortization_charge,
                    project_years=project_years,
                    target_year=approval_year,
                )

            else:
                raise UnitOfProductionException(
                    f"Unrecognized initial amortization year: "
                    f"{initial_amortization_year.__class__.__qualname__}. "
                )

        # Modify amortization charge, accounting for project duration
        if len(amortization_charge) < len(project_years):
            extension = np.zeros(len(project_years) - len(amortization_charge))
            amortization_charge = np.concatenate((amortization_charge, extension))

        return amortization_charge
"""
