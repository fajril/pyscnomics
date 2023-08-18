"""
Prepare and classify cost data based on its components. The associated cost components are:
(1) Tangible,
(2) Intangible,
(3) OPEX,
(4) ASR
"""

from dataclasses import dataclass, field

import numpy as np
import pyscnomics.econ.depreciation as depr
from pyscnomics.econ.selection import FluidType, DeprMethod


class TangibleException(Exception):
    """Exception to be raised if class Tangible is misused"""

    pass


class IntangibleException(Exception):
    """Exception to be raised if class Intangible is misused"""

    pass


class OPEXException(Exception):
    """Exception to be raised if class OPEX is misused"""

    pass


class ASRException(Exception):
    """Exception to be raised if class ASR is misused"""

    pass


@dataclass
class Tangible:
    """
    Manages a tangible asset.

    Parameters
    ----------
    start_year : int
        The start year of the project.
    end_year : int
        The end year of the project.
    cost : numpy.ndarray
        An array representing the cost of the tangible asset.
    expense_year : numpy.ndarray
        An array representing the expense year of the tangible asset.
    pis_year : numpy.ndarray, optional
        An array representing the PIS (Placed-in-Service) year
        of the tangible asset.
    cost_allocation : list[FluidType]
        A list representing the cost allocation of the tangible asset.
    salvage_value : numpy.ndarray, optional
        An array representing the salvage value of the tangible asset.
    useful_life : numpy.ndarray, optional
        An array representing the useful life of the tangible asset.
    """

    start_year: int
    end_year: int
    cost: np.ndarray
    expense_year: np.ndarray
    cost_allocation: FluidType = field(default=None)
    pis_year: np.ndarray = field(default=None, repr=False)
    salvage_value: np.ndarray = field(default=None, repr=False)
    useful_life: np.ndarray = field(default=None, repr=False)

    # Attribute to be defined later on
    project_years: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self):

        # Configure attribute expense_year
        self.expense_year = np.asarray(self.expense_year)

        # Configure attribute pis_year
        if self.pis_year is None:
            self.pis_year = self.expense_year.copy()
        else:
            self.pis_year = np.asarray(self.pis_year)

        # Configure attribute salvage_value
        if self.salvage_value is None:
            self.salvage_value = np.zeros(len(self.cost))
        else:
            self.salvage_value = np.asarray(self.salvage_value)

        # Configure attribute useful_life
        if self.useful_life is None:
            self.useful_life = np.repeat(5, len(self.cost))
        else:
            self.useful_life = np.asarray(self.useful_life)

        # Check input data for unequal length
        arr_length = len(self.cost)

        if not all(
            len(arr) == arr_length
            for arr in [
                self.expense_year,
                self.pis_year,
                self.salvage_value,
                self.useful_life,
            ]
        ):
            raise TangibleException(
                f"Inequal length of array: "
                f"cost: {len(self.cost)}, "
                f"expense_year: {len(self.expense_year)}, "
                f"pis_year: {len(self.pis_year)}, "
                f"salvage_value: {len(self.salvage_value)}, "
                f"useful_life: {len(self.useful_life)}"
            )

        # Check for inappropriate start and end year project
        if self.end_year > self.start_year:
            self.project_length = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise TangibleException(
                f"start year {self.start_year} "
                f"is after the end year: {self.end_year}"
            )

        # Raise error message when expense year if after the end year of the project
        if np.max(self.expense_year) > self.end_year:
            raise TangibleException(
                f"Expense year ({np.max(self.expense_year)}) "
                f"is beyond the end project year: {self.end_year}"
            )

    def __len__(self):
        return self.project_length

    def __eq__(self, other):

        # Equality condition between two instances of Tangible class
        if isinstance(other, Tangible):
            return all(
                (
                    self.cost_allocation == other.cost_allocation,
                    np.allclose(self.expense_year, other.expense_year),
                    np.allclose(self.pis_year, other.pis_year),
                    np.allclose(self.cost, other.cost),
                )
            )

        # Equality condition between an instance of Tangible class and an integer/float
        if isinstance(other, (int, float)):
            return np.allclose(sum(self.cost), other)

        return False

    def __lt__(self, other):

        # Between two instances of class
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) < np.sum(other.cost)

        # Between an instance of a class and an integer (or a float)
        if isinstance(other, (int, float)):
            return np.sum(self.cost) < other

        return False

    def __le__(self, other):

        # Between two instances of class
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) <= np.sum(other.cost)

        # Between an instance of a class and an integer (or a float)
        if isinstance(other, (int, float)):
            return np.sum(self.cost) <= other

        return False

    def __gt__(self, other):

        # Between two instances of class
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) > np.sum(other.cost)

        # Between an instance of a class and an integer (or a float)
        if isinstance(other, (int, float)):
            return np.sum(self.cost) > other

        return False

    def __ge__(self, other):

        # Between two instances of class
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) >= np.sum(other.cost)

        # Between an instance of a class and an integer (or a float)
        if isinstance(other, (int, float)):
            return np.sum(self.cost) >= other

        return False

    def __add__(self, other):

        # Rule for addition between two class instances
        if not isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            raise TangibleException(f"Must add two instances of class.")

        else:

            # Raise exception error if self.cost_allocation is not equal to other.cost_allocation
            if self.cost_allocation != other.cost_allocation:
                raise TangibleException(
                    "Cost allocation is not equal. "
                    f"First instance is {self.cost_allocation}, "
                    f"second instance is {other.cost_allocation} "
                )

            else:

                start_year = min(self.start_year, other.start_year)
                end_year = max(self.end_year, other.end_year)
                cost = np.concatenate((self.cost, other.cost))
                expense_year = np.concatenate((self.expense_year, other.expense_year))
                pis_year = np.concatenate((self.pis_year, other.pis_year))
                salvage_value = np.concatenate(
                    (self.salvage_value, other.salvage_value)
                )
                useful_life = np.concatenate((self.useful_life, other.useful_life))

                new_tangible = Tangible(
                    start_year=start_year,
                    end_year=end_year,
                    cost=cost,
                    expense_year=expense_year,
                    pis_year=pis_year,
                    salvage_value=salvage_value,
                    useful_life=useful_life,
                    cost_allocation=self.cost_allocation,
                )

                return new_tangible

    def __mul__(self, other):

        # Multiplication is allowed only with an integer or a float
        if not isinstance(other, (int, float)):
            raise TangibleException(
                f"Must multiply with an integer or a float. "
                f"{other} is not an integer nor a float."
            )

        # Cannot multiply with a negative integer/float
        if other < 0:
            raise TangibleException(f"Cannot multiply with a negative integer/float.")

        else:

            new_tangible = Tangible(
                start_year=self.start_year,
                end_year=self.end_year,
                cost=self.cost * other,
                expense_year=self.expense_year.copy(),
                pis_year=self.pis_year.copy(),
                salvage_value=self.salvage_value.copy(),
                useful_life=self.useful_life.copy(),
                cost_allocation=self.cost_allocation,
            )
            return new_tangible

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):

        # Division operation between two instances of class
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) / np.sum(other.cost)

        # Division operation between an instance and an integer/float
        if isinstance(other, (int, float)):

            # Cannot divide with a zero or a negative value
            if other <= 0:
                raise TangibleException(
                    f"Cannot divide with zero or with a negative value."
                )

            else:

                new_tangible = Tangible(
                    start_year=self.start_year,
                    end_year=self.end_year,
                    cost=self.cost / other,
                    expense_year=self.expense_year.copy(),
                    pis_year=self.pis_year.copy(),
                    salvage_value=self.salvage_value.copy(),
                    useful_life=self.useful_life.copy(),
                    cost_allocation=self.cost_allocation,
                )

                return new_tangible

    def expenditures(self):
        """
        Calculate tangible expenditures per year.

        This method calculates the tangible expenditures per year
        based on the expense year and cost data provided.

        Returns
        -------
        numpy.ndarray
            An array representing the tangible expenditures for each year.
        """

        expenses = np.bincount(self.expense_year - self.start_year, weights=self.cost)

        zeros = np.zeros(self.project_length - len(expenses))
        expenses = np.concatenate((expenses, zeros))

        return expenses

    def total_depreciation_rate(
        self,
        depr_method: DeprMethod = DeprMethod.DB,
        decline_factor: float = 2,
    ):
        """
        Calculate the total depreciation rate.

        Parameters
        ----------
        depr_method : DeprMethod
            The depreciation method to use. Defaults to DeprMethod.DB.
        decline_factor : float
            The decline factor used in the declining balance method. Defaults to 2.

        Returns
        -------
        numpy.ndarray
            An array representing the total depreciation rate for each year.

        Notes
        -----
        - The `fluid_type` argument should be an instance
          of the `FluidType` enum.
        - The `depr_method` argument should be an instance
          of the `DeprMethod` enum.
        """

        # Declining balance
        if depr_method == DeprMethod.DB:
            depreciation_charge = np.asarray(
                [
                    depr.declining_balance_depreciation_rate(
                        cost=c,
                        salvage_value=sv,
                        useful_life=ul,
                        decline_factor=decline_factor,
                        depreciation_len=self.project_length,
                    )
                    for c, sv, ul in zip(
                        self.cost,
                        self.salvage_value,
                        self.useful_life,
                    )
                ]
            )

        # Straight line
        if depr_method == DeprMethod.SL:
            depreciation_charge = np.asarray(
                [
                    depr.straight_line_depreciation_rate(
                        cost=c,
                        salvage_value=sv,
                        useful_life=ul,
                        depreciation_len=self.project_length,
                    )
                    for c, sv, ul in zip(
                        self.cost,
                        self.salvage_value,
                        self.useful_life,
                    )
                ]
            )

        shift_indices = self.pis_year - self.start_year
        depreciation_charge = np.asarray(
            [
                np.concatenate((np.zeros(i), row[:-i])) if i > 0 else row
                for row, i in zip(depreciation_charge, shift_indices)
            ]
        )
        return depreciation_charge.sum(axis=0)

    def total_depreciation_book_value(
        self,
        depr_method: DeprMethod = DeprMethod.DB,
        decline_factor: float = 2,
    ):
        """
        Calculate the total book value of depreciation for the asset.

        Parameters
        ----------
        depr_method : DeprMethod
            The depreciation method to use. Defaults to DeprMethod.DB.
        decline_factor : float
            The decline factor used in the declining balance method.
            Defaults to 2.

        Returns
        -------
        numpy.ndarray
            An array representing the total depreciation book value
            for each year.

        Notes
        -----
        - The function uses the `total_depreciation_rate` method
          to calculate the depreciation charge.
        - The book value of depreciation is calculated
          by subtracting the cumulative depreciation charge
          from the cumulative tangible expenditures.

        Examples
        --------
        # Calculate the total book value of depreciation using the default parameters
        result = total_depreciation_book_value()
        """

        depreciation_charge = self.total_depreciation_rate(
            depr_method=depr_method,
            decline_factor=decline_factor,
        )

        return np.cumsum(self.expenditures()) - np.cumsum(depreciation_charge)


@dataclass
class Intangible:
    """
    Manages an intangible asset.
    """

    start_year: int
    end_year: int
    cost: np.ndarray = field(repr=False)
    expense_year: np.ndarray = field(repr=False)
    cost_allocation: FluidType = field(default=FluidType.OIL, repr=False)

    def __post_init__(self):
        if self.end_year > self.start_year:
            self.project_length = self.end_year - self.start_year + 1
        else:
            raise ValueError(
                f"start year {self.start_year} "
                f" is after the end year: {self.end_year}"
            )

    def __len__(self):
        return self.project_length

    def __eq__(self, other):
        if isinstance(other, Intangible):
            return all(
                (
                    self.cost_allocation == other.cost_allocation,
                    np.allclose(self.expense_year, other.expense_year),
                    np.allclose(self.cost, other.cost),
                )
            )
        if isinstance(other, (int, float)):
            return np.allclose(sum(self.cost), other)
        return False

    def __lt__(self, other):
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) < np.sum(other.cost)
        if isinstance(other, (int, float)):
            return np.sum(self.cost) < other
        return False

    def __le__(self, other):
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) <= np.sum(other.cost)
        if isinstance(other, (int, float)):
            return np.sum(self.cost) <= other
        return False

    def __gt__(self, other):
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) > np.sum(other.cost)
        if isinstance(other, (int, float)):
            return np.sum(self.cost) > other
        return False

    def __ge__(self, other):
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) >= np.sum(other.cost)
        if isinstance(other, (int, float)):
            return np.sum(self.cost) >= other
        return False

    def __add__(self, other):
        raise NotImplemented

    def __sub__(self, other):
        raise NotImplemented

    def __mul__(self, other):
        raise NotImplemented

    def __rmul__(self, other):
        raise NotImplemented

    def __truediv__(self, other):
        raise NotImplemented

    def expenditures(self):
        """
        Calculate intangible expenditures per year.

        This method calculates the intangible expenditures per year
        based on the expense year and cost data provided.

        Returns
        -------
        numpy.ndarray
            An array representing the intangible expenditures for each year.
        """
        expenditures = np.bincount(
            self.expense_year - self.start_year, weights=self.cost
        )
        zeros = np.zeros(self.project_length - len(expenditures))
        expenditures = np.concatenate((expenditures, zeros))
        return expenditures


@dataclass
class OPEX:
    """
    Manages an OPEX asset.
    """

    start_year: int
    end_year: int
    fixed_cost: np.ndarray
    variable_cost: np.ndarray = field(init=None, repr=False)
    cost_allocation: FluidType = field(init=FluidType.OIL)

    def __post_init__(self):
        if self.end_year > self.start_year:
            self.project_length = self.end_year - self.start_year + 1
        else:
            raise ValueError(
                f"start year {self.start_year} "
                f" is after the end year: {self.end_year}"
            )
        if self.variable_cost is None:
            self.variable_cost = np.zeros(self.project_length)

        self.cost = self.fixed_cost + self.variable_cost

    def __len__(self):
        return self.project_length

    def __eq__(self, other):
        if isinstance(other, OPEX):
            return all(
                (
                    self.cost_allocation == other.cost_allocation,
                    np.allclose(self.fixed_cost, other.fixed_cost),
                    np.allclose(self.variable_cost, other.variable_cost),
                )
            )
        if isinstance(other, (int, float)):
            return np.allclose(sum(self.fixed_cost, self.variable_cost), other)
        return False

    def __lt__(self, other):
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) < np.sum(other.cost)
        if isinstance(other, (int, float)):
            return np.sum(self.cost) < other

    def __le__(self, other):
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) <= np.sum(other.cost)
        if isinstance(other, (int, float)):
            return np.sum(self.cost) <= other
        return False

    def __gt__(self, other):
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) > np.sum(other.cost)
        if isinstance(other, (int, float)):
            return np.sum(self.cost) > other
        return False

    def __ge__(self, other):
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) >= np.sum(other.cost)
        if isinstance(other, (int, float)):
            return np.sum(self.cost) >= other
        return False

    def __add__(self, other):
        raise NotImplemented

    def __sub__(self, other):
        raise NotImplemented

    def __mul__(self, other):
        raise NotImplemented

    def __rmul__(self, other):
        raise NotImplemented

    def __truediv__(self, other):
        raise NotImplemented

    def expenditures(
        self, prod_rate: np.ndarray = None, cost_per_volume: np.ndarray = None
    ):
        """1234"""

        if None not in [prod_rate, cost_per_volume]:
            self.variable_cost = prod_rate * cost_per_volume
            self.cost = self.fixed_cost + self.variable_cost

        return self.cost


@dataclass
class ASR:
    """1234"""

    start_year: int
    end_year: int
    cost: np.ndarray = field(repr=False)
    expense_year: np.ndarray = field(repr=False)
    cost_allocation: FluidType = field(default=FluidType.OIL)
    rate: float = field(default=0.02)

    def __post_init__(self):
        if self.end_year > self.start_year:
            self.project_length = self.end_year - self.start_year + 1
        else:
            raise ValueError(
                f"start year {self.start_year} "
                f" is after the end year: {self.end_year}"
            )

    def __len__(self):
        return self.project_length

    def __eq__(self, other):
        if isinstance(other, ASR):
            return all(
                (
                    np.allclose(self.cost, other.cost),
                    np.allclose(self.expense_year, other.expense_year),
                    self.cost_allocation == other.cost_allocation,
                )
            )
        if isinstance(other, (int, float)):
            return np.allclose(sum(self.cost), other)
        return False

    def __lt__(self, other):
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) < np.sum(other.cost)
        if isinstance(other, (int, float)):
            return np.sum(self.cost) < other
        return False

    def __le__(self, other):
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) <= np.sum(other.cost)
        if isinstance(other, (int, float)):
            return np.sum(self.cost) <= other
        return False

    def __gt__(self, other):
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) > np.sum(other.cost)
        if isinstance(other, (int, float)):
            return np.sum(self.cost) > other
        return False

    def __ge__(self, other):
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) >= np.sum(other.cost)
        if isinstance(other, (int, float)):
            return np.sum(self.cost) >= other
        return False

    def __add__(self, other):
        raise NotImplemented

    def __mul__(self, other):
        raise NotImplemented

    def __rmul__(self, other):
        raise NotImplemented

    def future_cost(self):
        """1234"""

        return self.cost * np.power((1 + self.rate), self.end_year - self.expense_year)

    def expenditures(self):
        """1234"""

        cost_duration = self.end_year - self.expense_year
        cost_alloc = self.future_cost() / cost_duration
        shift_indices = self.expense_year - self.start_year

        asr_alloc = np.asarray(
            [
                np.concatenate((np.zeros(i), np.repeat(ca, cd)))
                for i, ca, cd in zip(shift_indices, cost_alloc, cost_duration)
            ]
        )

        return asr_alloc.sum(axis=0)
