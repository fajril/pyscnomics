"""
Prepares lifting data and calculate the associated revenue.
"""

from dataclasses import dataclass, field
from functools import reduce
from typing import cast, overload

import numpy as np
import numpy.typing as npt
from typing_extensions import override

from pyscnomics.econ.selection import FluidType
from pyscnomics.econ.costs import (
    CapitalCost,
    Intangible,
    OPEX,
    ASR,
    LBT,
    CostOfSales,
)
from pyscnomics.validation_rules import (
    lifting_exceeds_production,
    missing_value_indices,
    period_is_reversed,
    unequal_lengths,
    year_above,
    year_below,
)
from pyscnomics.input_validation import InputValidationError, engine_input_issue


class LiftingException(Exception):
    """Exception to be raised when class Lifting is misused"""

    pass


class LiftingInputError(  # pyright: ignore[reportUnsafeMultipleInheritance]
    LiftingException, InputValidationError
):
    """Verified user-input lifting error retaining ``LiftingException``."""

    _input_code: str = "LIFTING_INPUT_INVALID"

    def __init__(self, message: str) -> None:
        InputValidationError.__init__(
            self, (engine_input_issue(code=self._input_code, message=message),)
        )


def _missing_indices(values: object) -> npt.NDArray[np.intp]:
    numeric_values = np.asarray(values, dtype=np.float64)
    return np.flatnonzero(np.isnan(numeric_values))


def _require_float_array(value: object, name: str) -> npt.NDArray[np.float64]:
    if not isinstance(value, np.ndarray):
        raise LiftingException(
            f"Attribute {name} must be provided as a numpy ndarray, "
            + f"not as a/an {value.__class__.__qualname__}."
        )
    return value.astype(np.float64, copy=False)


@dataclass(init=False)
class Lifting:
    """
    Create an object that represents lifting.

    Parameters
    ----------
    start_year: int
        The start year of the project.
    end_year: int
        The end year of the project.
    lifting_rate: np.ndarray
        The lifting data of a particular fluid type.
    price: np.ndarray
        The associated price of a particular fluid type.
    prod_year: np.ndarray
        The production year of a particular fluid type.
    fluid_type: FluidType
        The fluid type.
    ghv: np.ndarray
        The value of ghv of a particular fluid type (default value = 1).
    prod_rate: np.ndarray
        The production rate of a particular fluid type.
    prod_rate_baseline: np.ndarray
        Baseline production rate for a particular fluid type.

    Notes
    -----
    (1) The unit used in the lifting rate should be in M unit, where the M is stands for 1000.
        While the price should be in US Dollar (USD).

    (2) The corresponding lifting unit for each fluid type are as the following:
        - Oil: MBBL
        - Gas: BSCF
        - LPG Propane: M-Ton (MT)
        - LPG Butane: M-Ton (MT)
        - Sulfur: M-Ton (MT)
        - CO2: M-Ton (MT)
        - Electricity: M-Unit

    (3) The corresponding price unit for each fluid type are as the following:
        - Oil:  USD/BBL
        - Gas:  USD/MM-British Thermal Unit (MMBTU),
                while the GHV will be: M-BTU/Standard Cubic Feet (SCF)
        - LPG Propane: USD/MT
        - LPG Butane: USD/MT
        - Sulfur: USD/MT
        - CO2: USD/MT
        - Electricity: USD/MT
    """

    start_year: int
    end_year: int
    lifting_rate: npt.NDArray[np.float64] = field(init=False)
    price: npt.NDArray[np.float64] = field(init=False)
    prod_year: npt.NDArray[np.int64] = field(init=False)
    fluid_type: FluidType | None = field(init=False)
    ghv: npt.NDArray[np.float64] | None = field(init=False)
    prod_rate: npt.NDArray[np.float64] | None = field(init=False)
    prod_rate_baseline: npt.NDArray[np.float64] | None = field(init=False)

    _lifting_rate_input: object = field(init=False, repr=False, compare=False)
    _price_input: object = field(init=False, repr=False, compare=False)
    _prod_year_input: object = field(init=False, repr=False, compare=False)
    _fluid_type_input: object = field(init=False, repr=False, compare=False)
    _ghv_input: object = field(init=False, repr=False, compare=False)
    _prod_rate_input: object = field(init=False, repr=False, compare=False)
    _prod_rate_baseline_input: object = field(init=False, repr=False, compare=False)

    # Attributes to be defined later
    project_duration: int = field(init=False)
    project_years: npt.NDArray[np.int64] = field(init=False)
    prod_rate_total: npt.NDArray[np.float64] = field(init=False)

    def __init__(
        self,
        start_year: int,
        end_year: int,
        lifting_rate: object,
        price: object,
        prod_year: object,
        fluid_type: object = FluidType.OIL,
        ghv: object = None,
        prod_rate: object = None,
        prod_rate_baseline: object = None,
    ) -> None:
        self.start_year = start_year
        self.end_year = end_year
        self._lifting_rate_input = lifting_rate
        self._price_input = price
        self._prod_year_input = prod_year
        self._fluid_type_input = fluid_type
        self._ghv_input = ghv
        self._prod_rate_input = prod_rate
        self._prod_rate_baseline_input = prod_rate_baseline
        self.__post_init__()

    def __post_init__(self) -> None:
        """
        Handles the following operations/procedures:
        -   Prepare attribute project_years and project_duration,
        -   Prepare attribute prod_year,
        -   Prepare attribute lifting_rate,
        -   Prepare attribute price,
        -   Prepare attribute fluid_type,
        -   Prepare attribute ghv,
        -   Prepare attribute prod_rate,
        -   Prepare attribute prod_rate_baseline,
        -   Check for unequal length of arrays data,
        -   Prepare attribute prod_rate_total,
        -   Raise an exception: prod_year is before the start year of the project,
        -   Raise an exception: prod_year is after the end year of the project.
        """

        # Prepare attribute project_years and project_duration
        if not period_is_reversed(self.start_year, self.end_year):
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(
                self.start_year, self.end_year + 1, dtype=np.int64
            )

        else:
            raise LiftingInputError(
                f"start_year ({self.start_year}) is after the end year ({self.end_year})."
            )

        # Prepare attribute prod_year
        prod_year = self._prod_year_input
        if not isinstance(prod_year, np.ndarray):
            raise LiftingException(
                "Attribute prod_year must be provided as a numpy ndarray, "
                + f"not as a/an {prod_year.__class__.__qualname__}."
            )

        else:
            prod_year_nan_id = missing_value_indices(
                cast("npt.NDArray[np.generic]", prod_year)
            )
            if len(prod_year_nan_id) > 0:
                raise LiftingInputError(
                    f"Missing values in array prod_year: {prod_year}"
                )

        prod_year_array: npt.NDArray[np.int64] = prod_year.astype(np.int64)
        self.prod_year = prod_year_array

        # Prepare attribute lifting rate
        lifting_rate = self._lifting_rate_input
        if not isinstance(lifting_rate, np.ndarray):
            raise LiftingException(
                "Attribute lifting rate must be provided as a numpy ndarray, "
                + f"not as an/a {lifting_rate.__class__.__qualname__}."
            )

        else:
            lifting_rate_nan_id = _missing_indices(lifting_rate.astype(np.float64))
            if len(lifting_rate_nan_id) > 0:
                lifting_rate[lifting_rate_nan_id] = np.zeros(len(lifting_rate_nan_id))

        lifting_rate_array: npt.NDArray[np.float64] = lifting_rate.astype(np.float64)
        self.lifting_rate = lifting_rate_array

        # Prepare attribute price
        price = self._price_input
        if not isinstance(price, np.ndarray):
            raise LiftingException(
                "Attribute price must be provided as a numpy ndarray, "
                + f"not as a/an {price.__class__.__qualname__}."
            )

        else:
            price_nan_id = _missing_indices(price.astype(np.float64))
            if len(price_nan_id) > 0:
                price[price_nan_id] = np.zeros(len(price_nan_id))

        price_array: npt.NDArray[np.float64] = price.astype(np.float64)
        self.price = price_array

        # Prepare attribute fluid_type
        fluid_type = self._fluid_type_input
        if fluid_type is None:
            self.fluid_type = FluidType.OIL
        elif not isinstance(fluid_type, FluidType):
            raise LiftingException(
                "Attribute fluid_type must be provided as an instance of FluidType, "
                + f"not as an/a {fluid_type.__class__.__qualname__}"
            )
        else:
            self.fluid_type = fluid_type

        # Prepare attribute ghv
        ghv = self._ghv_input
        if ghv is None:
            ghv = np.ones_like(prod_year_array, dtype=np.float64)

        else:
            if not isinstance(ghv, np.ndarray):
                raise LiftingException(
                    "Attribute ghv must be provided as a numpy ndarray, "
                    + f"not as a/an {ghv.__class__.__qualname__}."
                )

            ghv_nan_id = _missing_indices(ghv.astype(np.float64))
            if len(ghv_nan_id) > 0:
                ghv[ghv_nan_id] = np.ones(len(ghv_nan_id))

        ghv_array: npt.NDArray[np.float64] = ghv.astype(np.float64)
        self.ghv = ghv_array

        # Prepare attribute prod_rate
        prod_rate = self._prod_rate_input
        if prod_rate is None:
            prod_rate = lifting_rate_array.copy()

        else:
            if not isinstance(prod_rate, np.ndarray):
                raise LiftingException(
                    "Attribute prod_rate must be provided as a numpy ndarray, "
                    + f"not as a/an {prod_rate.__class__.__qualname__}."
                )

            prod_rate_nan_id = _missing_indices(prod_rate.astype(np.float64))
            if len(prod_rate_nan_id) > 0:
                prod_rate[prod_rate_nan_id] = lifting_rate_array[
                    prod_rate_nan_id
                ].copy()

            # Raise an exception: prod_rate is smaller than lifting_rate
            if lifting_exceeds_production(
                lifting_rate_array, cast("npt.NDArray[np.float64]", prod_rate)
            ):
                raise LiftingInputError(
                    f"Lifting rate ({lifting_rate_array}) is larger than "
                    + f"the production rate ({prod_rate})"
                )

        prod_rate_array: npt.NDArray[np.float64] = prod_rate.astype(np.float64)
        self.prod_rate = prod_rate_array

        # Prepare attribute prod_rate_baseline
        prod_rate_baseline = self._prod_rate_baseline_input
        if prod_rate_baseline is None:
            prod_rate_baseline = np.zeros_like(prod_year_array, dtype=np.float64)

        else:
            if not isinstance(prod_rate_baseline, np.ndarray):
                raise LiftingException(
                    "Attribute prod_rate_baseline must be provided as a numpy ndarray, "
                    + f"not as a/an {prod_rate_baseline.__class__.__qualname__}."
                )

            prod_rate_baseline_nan_id = _missing_indices(
                prod_rate_baseline.astype(np.float64)
            )
            if len(prod_rate_baseline_nan_id) > 0:
                prod_rate_baseline[prod_rate_baseline_nan_id] = np.zeros(
                    len(prod_rate_baseline_nan_id)
                )

        prod_rate_baseline_array: npt.NDArray[np.float64] = prod_rate_baseline.astype(
            np.float64
        )
        self.prod_rate_baseline = prod_rate_baseline_array

        # Check for unequal length of arrays data
        length_map = {
            "prod_year": len(prod_year_array),
            "lifting_rate": len(lifting_rate_array),
            "price": len(price_array),
            "ghv": len(ghv_array),
            "prod_rate": len(prod_rate_array),
            "prod_rate_baseline": len(prod_rate_baseline_array),
        }
        if unequal_lengths(length_map) is not None:
            raise LiftingInputError(
                "Unequal length of arrays: "
                + f"prod_year: {len(prod_year_array)}, "
                + f"lifting_rate: {len(lifting_rate_array)}, "
                + f"price: {len(price_array)}, "
                + f"ghv: {len(ghv_array)}, "
                + f"prod_rate: {len(prod_rate_array)}, "
                + f"prod_rate_baseline: {len(prod_rate_baseline_array)}."
            )

        # Prepare attribute prod_rate_total
        prod_rate_total = np.asarray(
            prod_rate_array + prod_rate_baseline_array,
            dtype=np.float64,
        )
        self.prod_rate_total = prod_rate_total

        # Raise an exception: prod_year is before the start year of the project
        below = year_below(prod_year_array, self.start_year)
        if below is not None:
            raise LiftingInputError(
                f"The production year ({below}) "
                + f"is before the start year of the project ({self.start_year})."
            )

        # Raise an exception: prod_year is after the end year of the project
        above = year_above(prod_year_array, self.end_year)
        if above is not None:
            raise LiftingInputError(
                f"The production year ({above}) "
                + f"is after the end year of the project ({self.end_year})."
            )

    def _get_array(
        self, target_param: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Aggregates and aligns `target_param` values with the corresponding
        production years.

        This function uses `np.bincount` to sum the values in `target_param` based on
        their associated production year (relative to the project's start year).
        The resulting array is then padded with zeros if the project duration exceeds
        the number of aggregated entries.

        Parameters
        ----------
        target_param : np.ndarray
            An array of numeric values to be aggregated and aligned by production year.

        Returns
        -------
        np.ndarray
            A 1D array of the same length as the project duration, where each element
            represents the aggregated value for a production year. Missing years are filled with zeros.

        Notes
        -----
        - `np.bincount` is used to group and sum values based on the difference between
          `self.prod_year` and `self.start_year`.
        - If the resulting array is shorter than the project duration, it is extended
          with zeros to ensure alignment.
        """

        prod_year = self.prod_year
        param_arr = np.bincount(prod_year - self.start_year, weights=target_param)
        zeros = np.zeros(self.project_duration - len(param_arr), dtype=np.float64)

        return np.asarray(np.concatenate((param_arr, zeros)), dtype=np.float64)

    def get_lifting_rate_ghv_arr(self) -> npt.NDArray[np.float64]:
        """
        Computes and aligns the array of lifting rate multiplied by GHV with
        production years.

        This method calculates the product of `lifting_rate` and `ghv` for each
        time step, then uses `_get_array` to aggregate and align the values
        according to the corresponding production year.

        The final array is padded with zeros if necessary to match the project
        duration.

        Returns
        -------
        np.ndarray
            A 1D array of length equal to the project duration, where each element
            represents the aligned value of `lifting_rate * ghv` for that production year.

        Notes
        -----
        - Internally uses `_get_array()` with `target_param = lifting_rate * ghv`.
        - Ensures alignment with production years and fills missing years with zeros.
        """
        lifting_rate = _require_float_array(self.lifting_rate, "lifting_rate")
        ghv = _require_float_array(self.ghv, "ghv")
        return self._get_array(target_param=lifting_rate * ghv)

    def get_lifting_rate_arr(self) -> npt.NDArray[np.float64]:
        """
        Creates an array of lifting_rate and aligns it according to the
        corresponding production year.

        Returns
        -------
        np.ndarray
            The array of lifting_rate with length equals to project duration.

        Notes
        -----
        Array of lifting_rate is generated using method _get_array() with
        target_param = lifting_rate.
        """
        lifting_rate = _require_float_array(self.lifting_rate, "lifting_rate")
        return self._get_array(target_param=lifting_rate)

    def get_ghv_arr(self) -> npt.NDArray[np.float64]:
        """
        Creates an array of ghv and aligns it according to the corresponding production year.

        Returns
        -------
        np.ndarray
            The array of GHV with length equals to the project duration.

        Notes
        -----
        Array of GHV is generated using method _get_array() with target_param = ghv.
        """
        ghv = _require_float_array(self.ghv, "ghv")
        return self._get_array(target_param=ghv)

    def get_prod_rate_arr(self) -> npt.NDArray[np.float64]:
        """
        Creates an array of production rate and aligns it according to the
        corresponding production year.

        Returns
        -------
        np.ndarray
            The array of production rate with length equals to project duration.

        Notes
        -----
        Array of production rate is generated using method _get_array() with
        target_param = prod_rate.
        """
        prod_rate = _require_float_array(self.prod_rate, "prod_rate")
        return self._get_array(target_param=prod_rate)

    def get_prod_rate_baseline_arr(self) -> npt.NDArray[np.float64]:
        """
        Creates an array of production rate baseline and aligns it according to
        the corresponding production year.

        Returns
        -------
        np.ndarray
            Array of production rate baseline with length equals to project duration.

        Notes
        -----
        Array of production rate baseline is generated using method _get_array() with
        target_param = prod_rate_baseline.
        """
        prod_rate_baseline = _require_float_array(
            self.prod_rate_baseline, "prod_rate_baseline"
        )
        return self._get_array(target_param=prod_rate_baseline)

    def get_prod_rate_total_arr(self) -> npt.NDArray[np.float64]:
        """
        Creates an array of total production rate (production rate + production rate baseline)
        and aligns it to the corresponding production year.

        Returns
        -------
        np.ndarray
            The array of total production rate (production rate + production rate baseline)
            with length equals to the project duration.

        Notes
        -----
        Array of total production rate is generated using method _get_array() with
        target_param = prod_rate_total.
        """
        return self._get_array(target_param=self.prod_rate_total)

    def get_price_arr(self) -> npt.NDArray[np.float64]:
        """
        Calculate the weighted average price (WAP) for each production year and align it
        with the project duration.

        The method computes WAP based on the `price`, `lifting_rate`, and `ghv` attributes,
        weighted by the product of `lifting_rate` and `ghv`. The resulting array is sorted
        by production year and extended with zeros to match the project duration.

        Returns
        -------
        np.ndarray
            An array of weighted average prices aligned with the project duration. If no
            production data exists for certain years, zeros are appended to the result.

        Notes
        -----
        If `lifting_rate` or `ghv` is zero, a small value (1E-33) is used to avoid
        division by zero.
        """
        lifting_rate = _require_float_array(self.lifting_rate, "lifting_rate")
        ghv = _require_float_array(self.ghv, "ghv")
        price = _require_float_array(self.price, "price")
        prod_year = self.prod_year
        # Identify unique year in prod_year array
        unique_year, indices = np.unique(prod_year, return_inverse=True)

        # Calculate weight
        weight = (lifting_rate * ghv).copy()
        weight[weight == 0] = 1e-33

        # Calculate WAP (Weighted Average Price)
        wap = np.asarray(
            [
                float(np.average(price[indices == i], weights=weight[indices == i]))
                for i in range(len(unique_year))
            ],
            dtype=np.float64,
        )

        # Sorted WAP so as to align with the corresponding prod_year
        wap_sorted = np.bincount(unique_year - self.start_year, weights=wap)
        zeros = np.zeros(self.project_duration - len(wap_sorted), dtype=np.float64)

        return np.asarray(np.concatenate((wap_sorted, zeros)), dtype=np.float64)

    def get_aggregate_ghv(self) -> npt.NDArray[np.float64]:
        """
        Calculate the weighted average gross heating value (WAGHV) for each production year
        and align it with the project duration.

        The method computes the WAGHV based on the `ghv` attribute, weighted by the
        `lifting_rate`. The resulting array is sorted by production year and extended
        with zeros to match the project duration.

        Returns
        -------
        np.ndarray
            An array of weighted average gross heating values aligned with the project duration.
            If no production data exists for certain years, zeros are appended to the result.

        Notes
        -----
        If `lifting_rate` is zero, a small value (1E-33) is used to avoid division by zero
        in the weighting calculation.
        """
        lifting_rate = _require_float_array(self.lifting_rate, "lifting_rate")
        ghv = _require_float_array(self.ghv, "ghv")
        prod_year = self.prod_year
        # Identify unique year in prod_year array
        unique_year, indices = np.unique(prod_year, return_inverse=True)

        # Determine weight values
        weight = lifting_rate.copy()
        weight[weight == 0] = 1e-33

        # Calculate the Weighted Average Gross Heating Value (WAGHV)
        waghv = np.asarray(
            [
                float(np.average(ghv[indices == i], weights=weight[indices == i]))
                for i in range(len(unique_year))
            ],
            dtype=np.float64,
        )

        # Sorted WAGHV so as to align with the corresponding prod_year
        waghv_sorted = np.bincount(unique_year - self.start_year, weights=waghv)
        zeros = np.zeros(self.project_duration - len(waghv_sorted), dtype=np.float64)

        return np.asarray(np.concatenate((waghv_sorted, zeros)), dtype=np.float64)

    def revenue(self) -> npt.NDArray[np.float64]:
        """
        Calculate the revenue of a particular fluid type and aligns it with the
        corresponding production year.

        Returns
        -------
        np.ndarray
            The revenue of a particular fluid type.

        Notes
        -----
        The revenue is calculated as follows: revenue = lifting rate * price * ghv.
        The function np.bincount() is used to align the revenue elements with its
        correponding year.
        """

        lifting_rate = _require_float_array(self.lifting_rate, "lifting_rate")
        price = _require_float_array(self.price, "price")
        ghv = _require_float_array(self.ghv, "ghv")
        prod_year = self.prod_year
        rev = lifting_rate * price * ghv
        rev_update = np.bincount(prod_year - self.start_year, weights=rev)

        # Modify revenue, accounting for project duration
        zeros = np.zeros(self.project_duration - len(rev_update), dtype=np.float64)

        return np.asarray(np.concatenate((rev_update, zeros)), dtype=np.float64)

    def __len__(self) -> int:
        prod_year = self.prod_year
        return len(prod_year)

    @override
    def __eq__(self, other: object) -> bool:
        # Between two instances of lifting
        if isinstance(other, Lifting):
            self_lifting_rate = _require_float_array(self.lifting_rate, "lifting_rate")
            self_price = _require_float_array(self.price, "price")
            self_prod_year = self.prod_year
            self_ghv = _require_float_array(self.ghv, "ghv")
            self_prod_rate = _require_float_array(self.prod_rate, "prod_rate")
            self_prod_rate_baseline = _require_float_array(
                self.prod_rate_baseline, "prod_rate_baseline"
            )
            other_lifting_rate = _require_float_array(
                other.lifting_rate, "lifting_rate"
            )
            other_price = _require_float_array(other.price, "price")
            other_prod_year = other.prod_year
            other_ghv = _require_float_array(other.ghv, "ghv")
            other_prod_rate = _require_float_array(other.prod_rate, "prod_rate")
            other_prod_rate_baseline = _require_float_array(
                other.prod_rate_baseline, "prod_rate_baseline"
            )
            return all(
                (
                    self.start_year == other.start_year,
                    self.end_year == other.end_year,
                    self.fluid_type == other.fluid_type,
                    np.allclose(self_lifting_rate, other_lifting_rate),
                    np.allclose(self_price, other_price),
                    np.allclose(self_prod_year, other_prod_year),
                    np.allclose(self_ghv, other_ghv),
                    np.allclose(self_prod_rate, other_prod_rate),
                    np.allclose(self_prod_rate_baseline, other_prod_rate_baseline),
                )
            )

        # Between an instance of Lifting and an integer/float
        elif isinstance(other, (int, float)):
            return np.allclose(np.sum(self.revenue()), other)

        else:
            return False

    def __lt__(self, other: object) -> bool | np.bool_:
        # Between two instances of Lifting
        if isinstance(other, Lifting):
            return np.sum(self.revenue()) < np.sum(other.revenue())

        # Between an instance of Lifting and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.revenue()) < other

        else:
            raise LiftingException(
                "Must compare an instance of Lifting with another instance "
                + "of Lifting, an integer, or a float."
            )

    def __le__(self, other: object) -> bool | np.bool_:
        # Between two instances of Lifting
        if isinstance(other, Lifting):
            return np.sum(self.revenue()) <= np.sum(other.revenue())

        # Between an instance of Lifting and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.revenue()) <= other

        else:
            raise LiftingException(
                "Must compare an instance of Lifting with another instance "
                + "of Lifting, an integer, or a float."
            )

    def __gt__(self, other: object) -> bool | np.bool_:
        # Between two instances of Lifting
        if isinstance(other, Lifting):
            return np.sum(self.revenue()) > np.sum(other.revenue())

        # Between an instance of Lifting and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.revenue()) > other

        else:
            raise LiftingException(
                "Must compare an instance of Lifting with another "
                + "instance of Lifting, an integer, or a float."
            )

    def __ge__(self, other: object) -> bool | np.bool_:
        # Between two instances of Lifting
        if isinstance(other, Lifting):
            return np.sum(self.revenue()) >= np.sum(other.revenue())

        # Between an instance of Lifting and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.revenue()) >= other

        else:
            raise LiftingException(
                "Must compare an instance of Lifting with another "
                + "instance of Lifting, an integer, or a float."
            )

    @overload
    def __add__(self, other: "Lifting") -> "Lifting": ...

    @overload
    def __add__(self, other: int | float) -> npt.NDArray[np.float64]: ...

    def __add__(self, other: object) -> "Lifting | npt.NDArray[np.float64]":
        # Between an instance of Lifting with another instance of Lifting
        if isinstance(other, Lifting):
            self_lifting_rate = _require_float_array(self.lifting_rate, "lifting_rate")
            self_price = _require_float_array(self.price, "price")
            self_prod_year = self.prod_year
            self_ghv = _require_float_array(self.ghv, "ghv")
            self_prod_rate = _require_float_array(self.prod_rate, "prod_rate")
            self_prod_rate_baseline = _require_float_array(
                self.prod_rate_baseline, "prod_rate_baseline"
            )
            other_lifting_rate = _require_float_array(
                other.lifting_rate, "lifting_rate"
            )
            other_price = _require_float_array(other.price, "price")
            other_prod_year = other.prod_year
            other_ghv = _require_float_array(other.ghv, "ghv")
            other_prod_rate = _require_float_array(other.prod_rate, "prod_rate")
            other_prod_rate_baseline = _require_float_array(
                other.prod_rate_baseline, "prod_rate_baseline"
            )
            # Raise exception error if self.fluid_type is not equal to other.fluid_type
            if self.fluid_type != other.fluid_type:
                raise LiftingException(
                    "Fluid types of the two instances are different. "
                    + f"First instance is {self.fluid_type}, "
                    + f"Second instance is {other.fluid_type}."
                )

            else:
                start_year = min(self.start_year, other.start_year)
                end_year = max(self.end_year, other.end_year)
                lifting_rate = np.concatenate((self_lifting_rate, other_lifting_rate))
                price = np.concatenate((self_price, other_price))
                prod_year = np.concatenate((self_prod_year, other_prod_year))
                ghv = np.concatenate((self_ghv, other_ghv))
                prod_rate = np.concatenate((self_prod_rate, other_prod_rate))
                prod_rate_baseline = np.concatenate(
                    (self_prod_rate_baseline, other_prod_rate_baseline)
                )

                return Lifting(
                    start_year=start_year,
                    end_year=end_year,
                    lifting_rate=lifting_rate,
                    price=price,
                    prod_year=prod_year,
                    fluid_type=self.fluid_type,
                    ghv=ghv,
                    prod_rate=prod_rate,
                    prod_rate_baseline=prod_rate_baseline,
                )

        # Between an instance of Lifting with an integer/float
        elif isinstance(other, (int, float)):
            return self.revenue() + other

        else:
            raise LiftingException(
                "Must add an instance of Lifting with another instance "
                + "of Lifting or an int/a float. "
                + f"{other}({other.__class__.__qualname__}) is not an instance of "
                + "Lifting or an int/a float."
            )

    @overload
    def __iadd__(self, other: "Lifting") -> "Lifting": ...

    @overload
    def __iadd__(self, other: int | float) -> npt.NDArray[np.float64]: ...

    def __iadd__(self, other: object) -> "Lifting | npt.NDArray[np.float64]":
        if not isinstance(other, (Lifting, int, float)):
            raise LiftingException(
                "Must add an instance of Lifting with another instance "
                + "of Lifting or an int/a float. "
                + f"{other}({other.__class__.__qualname__}) is not an instance of "
                + "Lifting or an int/a float."
            )
        return self.__add__(other)

    def __sub__(self, other: object) -> npt.NDArray[np.float64]:
        # If "other" is an instance of Lifting object
        if isinstance(other, Lifting):
            # Configure the minimum and maximum values of the start and end year, respectively
            start_year = min(self.start_year, other.start_year)
            end_year = max(self.end_year, other.end_year)

            self_revenue = self.revenue().copy()
            other_revenue = other.revenue().copy()

            # If the length of self_revenue data is less than project duration
            if len(self_revenue) < end_year - start_year + 1:
                # Modify the size of self_revenue data
                self_revenue.resize(end_year - start_year + 1, refcheck=False)

                # If self.start_year > other.start_year, roll the array n steps to the right;
                # n = self.start_year - start_year
                if self.start_year > other.start_year:
                    self_revenue = np.roll(self_revenue, (self.start_year - start_year))

            # If the length of other_revenue data is less than project duration
            if len(other_revenue) < end_year - start_year + 1:
                # Modify the size of other_revenue data
                other_revenue.resize(end_year - start_year + 1, refcheck=False)

                # If other.start_year > self.start_year, roll the array n steps to the right;
                # n = other.start_year - start_year
                if other.start_year > self.start_year:
                    other_revenue = np.roll(
                        other_revenue, (other.start_year - start_year)
                    )

            return self_revenue - other_revenue

        # If "other" is int, float, or numpy array
        elif isinstance(other, (int, float, np.ndarray)):
            return self.revenue() - other

        else:
            raise LiftingException(
                "Must subtract an instance of Lifting with another instance "
                + "of Lifting or an int/a float. "
                + f"{other}({other.__class__.__qualname__}) is not an instance of "
                + "Lifting or an int/a float"
            )

    def __mul__(self, other: object) -> "Lifting":
        # Multiplication is allowed only with an integer or a float
        if isinstance(other, (int, float)):
            lifting_rate = _require_float_array(self.lifting_rate, "lifting_rate")
            return Lifting(
                start_year=self.start_year,
                end_year=self.end_year,
                lifting_rate=lifting_rate * other,
                price=_require_float_array(self.price, "price"),
                prod_year=self.prod_year,
                fluid_type=self.fluid_type,
                ghv=_require_float_array(self.ghv, "ghv"),
                prod_rate=_require_float_array(self.prod_rate, "prod_rate"),
                prod_rate_baseline=_require_float_array(
                    self.prod_rate_baseline, "prod_rate_baseline"
                ),
            )

        else:
            raise LiftingException(
                "Must multiply with an integer of a float, "
                + f"{other}({other.__class__.__qualname__}) is not an integer nor a float."
            )

    def __rmul__(self, other: object) -> "Lifting":
        return self.__mul__(other)

    @overload
    def __truediv__(self, other: "Lifting") -> np.float64: ...

    @overload
    def __truediv__(
        self, other: CapitalCost | Intangible | OPEX | ASR | LBT | CostOfSales
    ) -> np.float64: ...

    @overload
    def __truediv__(self, other: int | float) -> "Lifting": ...

    @overload
    def __truediv__(
        self, other: npt.NDArray[np.generic]
    ) -> np.float64 | np.complex128: ...

    def __truediv__(self, other: object) -> "Lifting | np.float64 | np.complex128":
        # Between two instances of Lifting
        if isinstance(other, Lifting):
            return np.sum(self.revenue()) / np.sum(other.revenue())

        # Between an instance of Lifting and an instance of
        # Capital/Intangible/OPEX/ASR/LBT/CostOfSales
        elif isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales)):
            return np.sum(self.revenue()) / np.sum(other.expenditures_post_tax())

        # Between an instance of Lifting and an integer/a float
        elif isinstance(other, (int, float)):
            # Cannot divide with zero
            if other == 0:
                raise LiftingException("Cannot divide with zero.")

            else:
                return Lifting(
                    start_year=self.start_year,
                    end_year=self.end_year,
                    lifting_rate=_require_float_array(self.lifting_rate, "lifting_rate")
                    / other,
                    price=_require_float_array(self.price, "price"),
                    prod_year=self.prod_year,
                    fluid_type=self.fluid_type,
                    ghv=_require_float_array(self.ghv, "ghv"),
                    prod_rate=_require_float_array(self.prod_rate, "prod_rate"),
                    prod_rate_baseline=_require_float_array(
                        self.prod_rate_baseline, "prod_rate_baseline"
                    ),
                )

        # Between an instance of Lifting with a NumPy ndarray
        elif isinstance(other, np.ndarray):
            if np.issubdtype(other.dtype, np.complexfloating):
                other_array = other.astype(np.complex128, copy=False)
            else:
                other_array = other.astype(np.float64, copy=False)
            return np.sum(self.revenue()) / np.sum(other_array)

        else:
            raise LiftingException(
                "Does not allow division operation between an instance of Lifting "
                + f"with a/an {other.__class__.__qualname__}."
            )


def aggregate_lifting_for_fluid(
    lifting: tuple[Lifting, ...] | list[Lifting],
    *,
    fluid_type: FluidType,
    start_year: int,
    end_year: int,
) -> Lifting:
    """Aggregate every lifting stream of one fluid, or a zero-filled stream.

    This is the shared aggregation used by ``BaseProject`` and the API input
    preparer so preflight and execution resolve onstream from identical revenue.
    A project without a stream of the requested fluid receives the existing
    zero-filled behavior (never an error and never an approximate value).
    """
    project_years = np.arange(start_year, end_year + 1, 1, dtype=np.int64)
    project_duration = end_year - start_year + 1
    streams = [lft for lft in lifting if lft.fluid_type == fluid_type]
    if not streams:
        return Lifting(
            start_year=start_year,
            end_year=end_year,
            prod_year=project_years,
            lifting_rate=np.zeros(project_duration),
            price=np.zeros(project_duration),
            fluid_type=fluid_type,
            prod_rate=np.zeros(project_duration),
            prod_rate_baseline=np.zeros(project_duration),
        )

    return reduce(lambda x, y: x + y, streams)
