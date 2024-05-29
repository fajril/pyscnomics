"""
Prepares lifting data and calculate the revenue.
"""

import numpy as np
from dataclasses import dataclass, field

from pyscnomics.econ.selection import FluidType
from pyscnomics.econ.costs import Tangible, Intangible, OPEX, ASR


class LiftingException(Exception):
    """Exception to be raised if class Lifting is misused"""

    pass


@dataclass
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
    fluid_type: FluidType
        The fluid type.
    ghv: np.ndarray
        The value of ghv of a particular fluid type (default value = 1).
    prod_rate: np.ndarray
        The production rate of a particular fluid type.
    """

    start_year: int
    end_year: int
    lifting_rate: np.ndarray
    price: np.ndarray
    prod_year: np.ndarray
    fluid_type: FluidType = field(default=FluidType.OIL)
    ghv: np.ndarray = field(default=None, repr=False)
    prod_rate: np.ndarray = field(default=None, repr=False)

    # Attribute to be defined later on
    project_duration: int = field(default=None, init=False)
    project_years: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # When user does not insert production rate data
        if self.prod_rate is None:
            self.prod_rate = self.lifting_rate.copy()

        # When user does not insert GHV data
        if self.ghv is None:
            self.ghv = np.ones(len(self.lifting_rate))

        # Check for inappropriate input data
        arr_length = self.lifting_rate.shape[0]

        if not all(
            len(arr) == arr_length
            for arr in [self.price, self.ghv, self.prod_rate, self.prod_year]
        ):
            raise LiftingException(
                f"Inequal length of array: lifting_rate: {len(self.lifting_rate)}, "
                f"price: {len(self.price)}, "
                f"ghv: {len(self.ghv)}, "
                f"production: {len(self.prod_rate)}, "
                f"prod_year: {len(self.prod_year)}, "
            )

        # Raise an error message: start_year is after end_year
        if self.end_year >= self.start_year:
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise LiftingException(
                f"start year {self.start_year} is after the end year: {self.end_year}"
            )

        # Raise an error message: prod_year is before the start year of the project
        if np.min(self.prod_year) < self.start_year:
            raise LiftingException(
                f"The production year ({np.min(self.prod_year)}) "
                f"is before the start year of the project ({self.start_year})"
            )

        # Raise an error message: prod_year is after the end year of the project
        if np.max(self.prod_year) > self.end_year:
            raise LiftingException(
                f"The production year ({np.max(self.prod_year)}) "
                f"is after the end year of the project ({self.end_year})"
            )

    def _get_array(self, target_param) -> np.ndarray:
        """
        Create an array of target_param.

        Parameters
        ----------
        target_param: np.ndarray
            An array containing the parameters to be weighted.

        Returns
        -------
        np.ndarray
            An array with values weighted by `target_param`, aligned with production years.

        Notes
        -----
        (1) Function np.bincount() is used to align the target_param according to
            its corresponding prod year,
        (2) If len(param_arr) < project_duration, then add the remaining elements
            with zeros.
        """
        param_arr = np.bincount(self.prod_year - self.start_year, weights=target_param)
        zeros = np.zeros(self.project_duration - len(param_arr))
        param_arr = np.concatenate((param_arr, zeros))

        return param_arr

    def revenue(self) -> np.ndarray:
        """
        Calculate the revenue of a particular fluid type.

        Returns
        -------
        rev: np.ndarray
            The revenue of a particular fluid type.

        Notes
        -----
        The revenue is calculated as follows: revenue = lifting rate * price * ghv.
        The function np.bincount() is used to align the revenue elements with its
        correponding year.
        """

        rev = self.lifting_rate * self.price * self.ghv
        rev_update = np.bincount(self.prod_year - self.start_year, weights=rev)

        # Modify revenues, acoounting for project duration
        zeros = np.zeros(self.project_duration - len(rev_update))
        rev_update = np.concatenate((rev_update, zeros))

        return rev_update

    def get_lifting_rate_arr(self) -> np.ndarray:
        """
        Create an array of lifting rate according to the corresponding production year.

        Returns
        -------
        np.ndarray
            The array of lifting rate with length equals to project duration.

        Notes
        -----
        Array of lifting rate is generated by calling the private method self._get_array().
        """
        return self._get_array(target_param=self.lifting_rate * self.ghv)

    def get_prod_rate_arr(self) -> np.ndarray:
        """
        Create an array of production rate according to the corresponding production year.

        Returns
        -------
        np.ndarray
            The array of lifting rate with length equals to project duration.

        Notes
        -----
        Array of lifting rate is generated by calling the private method self._get_array().
        """
        return self._get_array(target_param=self.prod_rate)

    def get_price_arr(self) -> np.ndarray:
        """
        Create an array of price according to the corresponding production year.

        Returns
        -------
        np.ndarray
            The array of price with length equals to project duration.

        Notes
        -----
        Array of price is generated by calling the private method self._get_array().
        """
        return self._get_array(target_param=self.price)

    def get_lifting_ghv_arr(self) -> np.ndarray:
        """
        Create an array of ghv according to the corresponding production year.

        Returns
        -------
        np.ndarray
            The array of lifting rate with length equals to project duration.

        Notes
        -----
        Array of ghv is generated by calling the private method self._get_array().
        """
        return self._get_array(target_param=self.ghv)

    def __len__(self):
        return self.project_duration

    def __eq__(self, other):
        # Between two instances of Lifting
        if isinstance(other, Lifting):
            return all(
                (
                    self.fluid_type == other.fluid_type,
                    self.start_year == other.start_year,
                    self.end_year == other.end_year,
                    np.allclose(self.lifting_rate, other.lifting_rate),
                    np.allclose(self.price, other.price),
                    np.allclose(self.ghv, other.ghv),
                    np.allclose(self.prod_rate, other.prod_rate),
                    np.allclose(self.prod_year, other.prod_year),
                )
            )

        # Between an instance of Lifting and an integer/float
        elif isinstance(other, (int, float)):
            return np.allclose(np.sum(self.revenue()), other)

        else:
            return False

    def __lt__(self, other):
        # Between two instances of Lifting
        if isinstance(other, Lifting):
            return np.sum(self.revenue()) < np.sum(other.revenue())

        # Between an instance of Lifting and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.revenue()) < other

        else:
            raise LiftingException(
                f"Must compare an instance of Lifting with another instance of Lifting, "
                f"an integer, or a float"
            )

    def __le__(self, other):
        # Between two instances of Lifting
        if isinstance(other, Lifting):
            return np.sum(self.revenue()) <= np.sum(other.revenue())

        # Between an instance of Lifting and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.revenue()) <= other

        else:
            raise LiftingException(
                f"Must compare an instance of Lifting with another instance of Lifting, "
                f"an integer, or a float"
            )

    def __gt__(self, other):
        # Between two instances of Lifting
        if isinstance(other, Lifting):
            return np.sum(self.revenue()) > np.sum(other.revenue())

        # Between an instance of Lifting and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.revenue()) > other

        else:
            raise LiftingException(
                f"Must compare an instance of Lifting with another instance of Lifting, "
                f"an integer, or a float"
            )

    def __ge__(self, other):
        # Between two instances of Lifting
        if isinstance(other, Lifting):
            return np.sum(self.revenue()) >= np.sum(other.revenue())

        # Between an instance of Lifting and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.revenue()) >= other

        else:
            raise LiftingException(
                f"Must compare an instance of Lifting with another instance of Lifting, "
                f"an integer, or a float"
            )

    def __add__(self, other):
        # Between an instance of Lifting with another instance of Lifting
        if isinstance(other, Lifting):
            # Raise exception error if self.cost_allocation is not equal to other.cost_allocation
            if self.fluid_type != other.fluid_type:
                raise LiftingException(
                    "Cost allocation is not equal. "
                    f"First instance is {self.fluid_type}, "
                    f"second instance is {other.fluid_type} "
                )

            else:
                start_year = min(self.start_year, other.start_year)
                end_year = max(self.end_year, other.end_year)
                lifting_rate = np.concatenate((self.lifting_rate, other.lifting_rate))
                price = np.concatenate((self.price, other.price))
                prod_year = np.concatenate((self.prod_year, other.prod_year))
                ghv = np.concatenate((self.ghv, other.ghv))
                prod_rate = np.concatenate((self.prod_rate, other.prod_rate))

                return Lifting(
                    start_year=start_year,
                    end_year=end_year,
                    lifting_rate=lifting_rate,
                    price=price,
                    prod_year=prod_year,
                    fluid_type=self.fluid_type,
                    ghv=ghv,
                    prod_rate=prod_rate,
                )

        # Between an instance of Lifting with an integer/float
        elif isinstance(other, (int, float)):
            return self.revenue() + other

        else:
            raise LiftingException(
                f"Must add an instance of Lifting with another instance "
                f"of Lifting or int/float "
                f"{other}({other.__class__.__qualname__}) is not an instance of "
                f"Lifting or int/float"
            )

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
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
                f"Must subtract an instance of Lifting with another instance "
                f"of Lifting or int/float "
                f"{other}({other.__class__.__qualname__}) is not an instance of "
                f"Lifting or int/float"
            )

    def __mul__(self, other):
        # Multiplication is allowed only with an integer/a float
        if isinstance(other, (int, float)):
            return Lifting(
                start_year=self.start_year,
                end_year=self.end_year,
                lifting_rate=self.lifting_rate * other,
                price=self.price,
                prod_year=self.prod_year,
                ghv=self.ghv,
                prod_rate=self.prod_rate,
            )

        else:
            raise LiftingException(
                f"Must multiply with an integer or a float; "
                f"{other} is not an integer nor a float"
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        # Between two instances of Lifting
        if isinstance(other, Lifting):
            return np.sum(self.revenue()) / np.sum(other.revenue())

        # Between an instance of Lifting and an instance of Tangible/Intangible/OPEX/ASR
        elif isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.revenue()) / np.sum(other.expenditures())

        # Between an instance of Lifting and an integer/float
        elif isinstance(other, (int, float)):
            # Cannot divide with zero
            if other == 0:
                raise LiftingException(f"Cannot divide with zero")

            else:
                return Lifting(
                    start_year=self.start_year,
                    end_year=self.end_year,
                    lifting_rate=self.lifting_rate / other,
                    price=self.price,
                    prod_year=self.prod_year,
                    ghv=self.ghv,
                    prod_rate=self.prod_rate,
                )

        elif isinstance(other, np.ndarray):
            return np.sum(self.revenue()) / np.sum(other)

        else:
            raise LiftingException(
                f"Does not allow division operation of an instance of Lifting "
                f"and {other.__class__.__qualname__}"
            )
