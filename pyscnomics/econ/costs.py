from enum import Enum
from dataclasses import dataclass, field

import numpy as np
import pyscnomics.econ.depreciation as depr

class DeprMethod(Enum):
    """
    Enumeration of depreciation methods.

    Attributes
    ----------
    SL : str
        Represents the straight-line depreciation method.
    DB : str
        Represents the declining balance depreciation method.
    UOP : str
        Represents the units of production depreciation method.
    """
    SL = "sl"
    DB = "db"
    UOP = "uop"


class FluidType(Enum):
    """
    Enumeration of fluid types for depreciation calculation.

    Attributes
    ----------
    ALL : str
        Represents all fluid types.
    OIL : str
        Represents oil as the fluid type.
    GAS : str
        Represents gas as the fluid type.
    """
    ALL = "all"
    OIL = "oil"
    GAS = "gas"


@dataclass
class Tangible:
    """
    Represents a tangible asset.

    Parameters
    ----------
    start_year : int
        The start year of the project.
    end_year : int
        The end year of the project.
    cost : numpy.ndarray, optional
        An array representing the cost of the tangible asset.
    expense_year : numpy.ndarray, optional
        An array representing the expense year of the tangible asset.
    pis_year : numpy.ndarray, optional
        An array representing the PIS (Placed-in-Service) year 
        of the tangible asset.
    cost_allocation : list[FluidType], optional
        A list representing the cost allocation of the tangible asset.
    salvage_value : numpy.ndarray, optional
        An array representing the salvage value of the tangible asset.
    useful_life : numpy.ndarray, optional
        An array representing the useful life of the tangible asset.
    """
    start_year: int
    end_year: int
    cost: np.ndarray
    expense_year: np.ndarray = field(repr=False)
    pis_year: np.ndarray = field(default=None, repr=False)
    cost_allocation: list[FluidType] = field(default=None, repr=False)
    salvage_value: np.ndarray = field(default=None, repr=False)
    useful_life: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        if self.pis_year is None:
            self.pis_year = self.expense_year.copy()
        if self.cost_allocation is None:
            self.cost_allocation = [FluidType.ALL for _ in self.cost]
        if self.salvage_value is None:
            self.salvage_value = np.zeros(self.cost.shape)
        if self.useful_life is None:
            self.useful_life = np.repeat(5, self.cost.shape)

        arr_length = self.cost.shape[0]
        if not all(
            len(arr) == arr_length
            for arr in [
                self.expense_year,
                self.pis_year,
                self.cost_allocation,
                self.salvage_value,
                self.useful_life,
            ]
        ):
            raise ValueError(
                f"Inequal length of array: cost: {len(self.cost)}, salvage_value: {len(self.salvage_value)} \
                    expense_year: {len(self.expense_year)}, pis_year: {len(self.cost_allocation)}"
            )
        if self.end_year > self.start_year:
            self.project_length = self.end_year - self.start_year + 1
        else:
            raise ValueError(
                f"start year {self.start_year} is after the end year: {self.end_year}"
            )
        if np.max(self.expense_year) > self.end_year:
            raise ValueError(
                f"Expense year ({np.max(self.expense_year)}) is beyond the end project year: {self.end_year}"
            )

    def __eq__(self, other):
        return all((
            all(s == o for s, o in zip(self.cost_allocation, other.cost_allocation)),
            np.allclose(self.expense_year, other.expense_year),
            np.allclose(self.pis_year, other.pis_year),
            np.allclose(self.cost, other.cost),
        ))

    def __lt__(self, other):
        return np.sum(self.cost) < np.sum(other.cost)

    def __le__(self, other):
        return np.sum(self.cost) <= np.sum(other.cost)

    def __gt__(self, other):
        return np.sum(self.cost) > np.sum(other.cost)

    def __ge__(self, other):
        return np.sum(self.cost) >= np.sum(other.cost)

    def __add__(self, other):
        start_year = min(self.start_year, other.start_year)
        end_year = max(self.end_year, other.end_year)
        cost = np.concatenate((self.cost, other.cost))
        expense_year = np.concatenate((self.expense_year, other.expense_year))
        pis_year = np.concatenate((self.pis_year, other.pis_year))
        salvage_value = np.concatenate((self.salvage_value, other.salvage_value))
        useful_life = np.concatenate((self.useful_life, other.useful_life))
        cost_allocation = self.cost_allocation + other.cost_allocation
        new_tangible = Tangible(
            start_year=start_year,
            end_year=end_year,
            cost=cost,
            expense_year=expense_year,
            pis_year=pis_year,
            salvage_value=salvage_value,
            useful_life=useful_life,
            cost_allocation=cost_allocation,
        )
        return new_tangible

    def __mul__(self, other):
        new_tangible = Tangible(
            start_year=self.start_year,
            end_year=self.end_year,
            cost=self.cost * other,
            expense_year=self.expense_year.copy(),
            pis_year=self.pis_year.copy(),
            salvage_value=self.salvage_value.copy(),
            useful_life=self.useful_life.copy(),
            cost_allocation=self.cost_allocation.copy(),
        )
        return new_tangible

    def __truediv__(self, other):
        if isinstance(other, Tangible):
            return np.sum(self.cost) / np.sum(other.cost)
        else:
            new_tangible = Tangible(
            start_year=self.start_year,
            end_year=self.end_year,
            cost=self.cost / other,
            expense_year=self.expense_year.copy(),
            pis_year=self.pis_year.copy(),
            salvage_value=self.salvage_value.copy(),
            useful_life=self.useful_life.copy(),
            cost_allocation=self.cost_allocation.copy(),
        )
            return new_tangible

    def tangible_expenditures(self):
        """
        Calculate tangible expenditures per year.

        This method calculates the tangible expenditures per year 
        based on the expense year and cost data provided.

        Returns
        -------
        numpy.ndarray
            An array representing the tangible expenditures for each year.
        """
        expenditures = np.bincount(
            self.expense_year - self.start_year, weights=self.cost
        )
        zeros = np.zeros(self.project_length - len(expenditures))
        expenditures = np.concatenate((expenditures, zeros))
        return expenditures

    def total_depreciation_book_value(
        self,
        fluid_type: FluidType = FluidType.ALL,
        depr_method: DeprMethod = DeprMethod.DB,
        decline_factor: float = 2,
    ):
        """
        Calculate the total book value of depreciation for the asset.

        Parameters
        ----------
        fluid_type : FluidType, optional
            The type of fluid for which depreciation is calculated. 
            Defaults to FluidType.ALL.
        depr_method : DeprMethod, optional
            The depreciation method to use. Defaults to DeprMethod.DB.
        decline_factor : float, optional
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
            fluid_type=fluid_type,
            depr_method=depr_method,
            decline_factor=decline_factor
        )
        return np.sumcum(self.tangible_expenditures()) - np.sumcum(depreciation_charge)

    def total_depreciation_rate(
        self,
        fluid_type: FluidType = FluidType.ALL,
        depr_method: DeprMethod = DeprMethod.DB,
        decline_factor: float = 2,
    ):
        """
        Calculate the total depreciation rate.

        Parameters
        ----------
        fluid_type : FluidType, optional
            The type of fluid for which depreciation is calculated. 
            Defaults to FluidType.ALL.
        depr_method : DeprMethod, optional
            The depreciation method to use. Defaults to DeprMethod.DB.
        decline_factor : float, optional
            The decline factor used in the declining balance method. 
            Defaults to 2.

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
                    if ca is fluid_type or fluid_type is FluidType.ALL
                    else np.zeros(self.project_length)
                    for c, sv, ul, ca in zip(
                        self.cost,
                        self.salvage_value,
                        self.useful_life,
                        self.cost_allocation,
                    )
                ]
            )
        if depr_method == DeprMethod.SL:
            depreciation_charge = np.asarray(
                [
                    depr.straight_line_depreciation_rate(
                        cost=c,
                        salvage_value=sv,
                        useful_life=ul,
                        depreciation_len=self.project_length,
                    )
                    if ca is fluid_type or fluid_type is FluidType.ALL
                    else np.zeros(self.project_length)
                    for c, sv, ul, ca in zip(
                        self.cost,
                        self.salvage_value,
                        self.useful_life,
                        self.cost_allocation,
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

@dataclass
class Intangible:
    start_year: int
    end_year: int
    cost: np.ndarray = field(repr=False)
    expense_year: np.ndarray = field(repr=False)
    pis_year: np.ndarray = field(default=None, repr=False)
    cost_allocation: list[FluidType] = field(default=None, repr=False)

    def __post_init__(self):
        if self.pis_year is None:
            self.pis_year = self.expense_year.copy()
        if self.cost_allocation is None:
            self.cost_allocation = [FluidType.ALL for _ in self.cost]
        if self.end_year > self.start_year:
            self.project_length = self.end_year - self.start_year + 1
        else:
            raise ValueError(
                f"start year {self.start_year} is after the end year: {self.end_year}"
            )
            
    def intangible_expenditures(self):
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