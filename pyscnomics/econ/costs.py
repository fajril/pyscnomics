"""
Prepare and classify cost data based on its components. The associated cost components are:
(1) Tangible,
(2) Intangible,
(3) OPEX,
(4) ASR
"""

import numpy as np
from dataclasses import dataclass, field

import pyscnomics.econ.depreciation as depr
from pyscnomics.tools.helper import apply_inflation, apply_vat
from pyscnomics.econ.selection import (
        FluidType,
        DeprMethod,
        YearReference,
        TaxRegime,
)


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
        An array representing the cost of a tangible asset.
    expense_year : numpy.ndarray
        An array representing the expense year of a tangible asset.
    pis_year : numpy.ndarray, optional
        An array representing the PIS (Placed-in-Service) year of a tangible asset.
    cost_allocation : FluidType
        A list representing the cost allocation of a tangible asset.
    salvage_value : numpy.ndarray, optional
        An array representing the salvage value of a tangible asset.
    useful_life : numpy.ndarray, optional
        An array representing the useful life of a tangible asset.
    depreciation_factor: np.ndarray, optional
        The value of depreciation factor to be used in PSC DB depreciation method.
        Default value is 0.5 for the entire project duration.
    is_ic_applied: bool, optional
        Whether investment credit is applied or not. Default value is False.
    """

    start_year: int
    end_year: int
    cost: np.ndarray
    expense_year: np.ndarray
    cost_allocation: FluidType = field(default=FluidType.OIL)
    pis_year: np.ndarray = field(default=None, repr=False)
    salvage_value: np.ndarray = field(default=None, repr=False)
    useful_life: np.ndarray = field(default=None, repr=False)
    depreciation_factor: np.ndarray = field(default=None, repr=False)
    is_ic_applied: bool = field(default=False)

    # Attribute to be defined later on
    project_duration: int = field(default=None, init=False, repr=False)
    project_years: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self):

        # When user does not provide pis_year data
        if self.pis_year is None:
            self.pis_year = self.expense_year.copy()

        # When user does not provide salvage_value data
        if self.salvage_value is None:
            self.salvage_value = np.zeros(len(self.cost))

        # When user does not provide useful_life data
        if self.useful_life is None:
            self.useful_life = np.repeat(5, len(self.cost))

        # When user does not provide depreciation_factor data
        if self.depreciation_factor is None:
            self.depreciation_factor = np.repeat(0.5, len(self.cost))

        # Check input data for unequal length
        arr_length = len(self.cost)

        if not all(
            len(arr) == arr_length
            for arr in [
                self.expense_year,
                self.pis_year,
                self.salvage_value,
                self.useful_life,
                self.depreciation_factor,
            ]
        ):
            raise TangibleException(
                f"Unequal length of array: "
                f"cost: {len(self.cost)}, "
                f"expense_year: {len(self.expense_year)}, "
                f"pis_year: {len(self.pis_year)}, "
                f"salvage_value: {len(self.salvage_value)}, "
                f"useful_life: {len(self.useful_life)}, "
                f"depreciation_factor: {len(self.depreciation_factor)}"
            )

        # Check for inappropriate start and end year project
        if self.end_year >= self.start_year:
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise TangibleException(
                f"start year {self.start_year} "
                f"is after the end year: {self.end_year}"
            )

        # Raise an error message: expense year is after the end year of the project
        if np.max(self.expense_year) > self.end_year:
            raise TangibleException(
                f"Expense year ({np.max(self.expense_year)}) "
                f"is after the end year of the project ({self.end_year})"
            )

        # Raise an error message: expense year is before the start year of the project
        if np.min(self.expense_year) < self.start_year:
            raise TangibleException(
                f"Expense year ({np.min(self.expense_year)}) "
                f"is before the start year of the project ({self.start_year})"
            )

    def expenditures(
        self,
        year_ref: YearReference = YearReference.EXPENSE_YEAR,
        tax_regime: TaxRegime = TaxRegime.NAILED_DOWN,
        vat_rate: float | int = 0,
        inflation_rate: float | int = 0
    ):
        """
        Calculate tangible expenditures per year.

        This method calculates the tangible expenditures per year
        based on the expense year and cost data provided.

        Parameters
        ----------
        year_ref: YearReference
            A point of reference to calculate expenses.
        tax_regime: TaxRegime
            The adopted regime/fiscal regulation to determine the
            configuration of VAT calculations.
        vat_rate: float | int
            Rate of Value Added Tax (VAT).
        inflation_rate: float | int
            Rate of inflation.

        Returns
        -------
        expenses: np.ndarray
                An array representing the tangible expenditures each year, taking
                into account the inflation and Value Added Tax (VAT/PPN).

        Notes
        -----
        (1) Function np.bincount() is used to align the cost elements according
            to its corresponding expense year,
        (2) If len(expenses) < project_duration, then add the remaining elements
            with zeros,
        (3) The return value is modified by taking into account the inflation,
            using the custom decorator @apply_inflation,
        (4) The return value is modified further by taking into account the Value
            Added Tax (VAT/PPN), using the custom decorator @apply_vat.
        """
        @apply_vat(tax_regime=tax_regime, vat_rate=vat_rate)
        @apply_inflation(inflation_rate=inflation_rate)
        def _expenditures() -> np.ndarray:
            if year_ref == YearReference.EXPENSE_YEAR:
                expenses = np.bincount(self.expense_year - self.start_year, weights=self.cost)

            else:
                expenses = np.bincount(self.pis_year - self.start_year, weights=self.cost)

            zeros = np.zeros(self.project_duration - len(expenses))
            expenses = np.concatenate((expenses, zeros))
            return expenses
        return _expenditures()

    def total_depreciation_rate(
        self,
        depr_method: DeprMethod = DeprMethod.PSC_DB,
        decline_factor: float | int = 2,
        tax_regime: TaxRegime = TaxRegime.NAILED_DOWN,
        vat_rate: float | int = 0,
        inflation_rate: float | int = 0
    ) -> tuple:
        """
        Calculate total depreciation charge and undepreciated asset for a set of assets.

        Parameters
        ----------
        depr_method : DeprMethod, optional
            The depreciation method to use (default is DeprMethod.PSC_DB).
        decline_factor : float or int, optional
            The decline factor for declining balance depreciation (default is 2).
        tax_regime : TaxRegime, optional
            The tax regime to apply to cost calculations (default is TaxRegime.NAILED_DOWN).
        vat_rate : float or int, optional
            The Value Added Tax (VAT) rate as a decimal or integer (default is 0).
        inflation_rate : float or int, optional
            The annual inflation rate as a decimal or integer (default is 0).

        Returns
        -------
        tuple
            A tuple containing two numpy arrays:
            - total_depreciation_charge: An array of total depreciation charges for each year.
            - undepreciated_asset: The undepreciated asset value at the end of the analysis.

        Notes
        -----
        - If tax_regime is TaxRegime.NAILED_DOWN, VAT/PPN is applied to the cost.
        - Inflation is applied to the cost based on the pis_year and start_year.
        - Depreciation charges are calculated based on the selected depreciation method.
        - The relative difference of pis_year and start_year is used to align expenditures.
        - Total depreciation charges and undepreciated asset are returned as numpy arrays.

        """

        # Apply VAT/PPN to cost
        if tax_regime == TaxRegime.NAILED_DOWN:
            modified_cost = np.float_(self.cost * (1 + vat_rate))

        # Apply Inflation to cost
        exponents = self.pis_year - self.start_year
        inflation_arr = (1. + inflation_rate) ** exponents
        modified_cost *= inflation_arr

        # Straight line
        if depr_method == DeprMethod.SL:
            depreciation_charge = np.asarray(
                [
                    depr.straight_line_depreciation_rate(
                        cost=c,
                        salvage_value=sv,
                        useful_life=ul,
                        depreciation_len=self.project_duration,
                    )
                    for c, sv, ul in zip(
                        modified_cost,
                        self.salvage_value,
                        self.useful_life,
                    )
                ]
            )

        # Declining balance/double declining balance
        if depr_method == DeprMethod.DB:
            depreciation_charge = np.asarray(
                [
                    depr.declining_balance_depreciation_rate(
                        cost=c,
                        salvage_value=sv,
                        useful_life=ul,
                        decline_factor=decline_factor,
                        depreciation_len=self.project_duration,
                    )
                    for c, sv, ul in zip(
                        modified_cost,
                        self.salvage_value,
                        self.useful_life,
                    )
                ]
            )

        # PSC_DB
        if depr_method == DeprMethod.PSC_DB:
            depreciation_charge = np.asarray(
                [
                    depr.psc_declining_balance_depreciation_rate(
                        cost=c,
                        depreciation_factor=dr,
                        useful_life=ul,
                        depreciation_len=self.project_duration,
                    )
                    for c, dr, ul in zip(
                        modified_cost,
                        self.depreciation_factor,
                        self.useful_life,
                    )
                ]
            )

        # The relative difference of pis_year and start_year
        shift_indices = self.pis_year - self.start_year

        # Modify depreciation_charge so that expenditures are aligned with
        # the corresponding pis_year (or expense_year)
        depreciation_charge = np.asarray(
            [
                np.concatenate((np.zeros(i), row[:-i])) if i > 0 else row
                for row, i in zip(depreciation_charge, shift_indices)
            ]
        )

        total_depreciation_charge = depreciation_charge.sum(axis=0)
        undepreciated_asset = np.sum(modified_cost) - np.sum(total_depreciation_charge)

        return total_depreciation_charge, undepreciated_asset

    def total_depreciation_book_value(
        self,
        depr_method: DeprMethod = DeprMethod.PSC_DB,
        decline_factor: float = 2,
        tax_regime: TaxRegime = TaxRegime.NAILED_DOWN,
        vat_rate: float | int = 0,
        inflation_rate: float | int = 0
    ) -> np.ndarray:
        """
        Calculate the total book value of depreciation for the asset.

        Parameters
        ----------
        depr_method : DeprMethod
            The depreciation method to use. Defaults to DeprMethod.DB.
        decline_factor : float
            The decline factor used in the declining balance method.
            Defaults to 2.
        tax_regime
        vat_rate
        inflation_rate

        Returns
        -------
        total_depreciation_book_value
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

        # Calculate total depreciation charge from method total_depreciation_rate
        depreciation_charge = self.total_depreciation_rate(
            depr_method=depr_method,
            decline_factor=decline_factor,
            tax_regime=tax_regime,
            vat_rate=vat_rate,
            inflation_rate=inflation_rate
        )[0]

        return np.cumsum(self.expenditures()) - np.cumsum(depreciation_charge)

    def __len__(self):
        return self.project_duration

    def __eq__(self, other):

        # Between two instances of Tangible
        if isinstance(other, Tangible):
            return all(
                (
                    np.allclose(self.cost, other.cost),
                    np.allclose(self.expense_year, other.expense_year),
                    np.allclose(self.pis_year, other.pis_year),
                    np.allclose(self.salvage_value, other.salvage_value),
                    np.allclose(self.useful_life, other.useful_life),
                    np.allclose(self.depreciation_factor, other.depreciation_factor),
                    self.cost_allocation == other.cost_allocation,
                    self.is_ic_applied == other.is_ic_applied,
                )
            )

        # Between an instance of Tangible and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) == other

        else:
            return False

    def __lt__(self, other):

        # Between an instance of Tangible with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) < np.sum(other.cost)

        # Between an instance of Tangible and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) < other

        else:
            raise TangibleException(
                f"Must compare an instance of Tangible with another instance of "
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float"
            )

    def __le__(self, other):

        # Between an instance of Tangible with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) <= np.sum(other.cost)

        # Between an instance of Tangible and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) <= other

        else:
            raise TangibleException(
                f"Must compare an instance of Tangible with another instance of "
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float"
            )

    def __gt__(self, other):

        # Between an instance of Tangible with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) > np.sum(other.cost)

        # Between an instance of Tangible and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) > other

        else:
            raise TangibleException(
                f"Must compare an instance of Tangible with another instance of "
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float"
            )

    def __ge__(self, other):

        # Between an instance of Tangible with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) >= np.sum(other.cost)

        # Between an instance of Tangible and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) >= other

        else:
            raise TangibleException(
                f"Must compare an instance of Tangible with another instance of "
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float"
            )

    def __add__(self, other):

        # Between an instance of Tangible with another instance of Tangible
        if isinstance(other, Tangible):

            # Raise exception error if self.cost_allocation is not equal to other.cost_allocation
            if self.cost_allocation != other.cost_allocation:
                raise TangibleException(
                    "Cost allocation is not equal. "
                    f"First instance is {self.cost_allocation}, "
                    f"second instance is {other.cost_allocation} "
                )

            else:
                combined_start_year = min(self.start_year, other.start_year)
                combined_end_year = max(self.end_year, other.end_year)

                return Tangible(
                    start_year=combined_start_year,
                    end_year=combined_end_year,
                    cost=np.concatenate((self.cost, other.cost)),
                    expense_year=np.concatenate((self.expense_year, other.expense_year)),
                    pis_year=np.concatenate((self.pis_year, other.pis_year)),
                    salvage_value=np.concatenate((self.salvage_value, other.salvage_value)),
                    useful_life=np.concatenate((self.useful_life, other.useful_life)),
                    depreciation_factor=np.concatenate(
                        (self.depreciation_factor, other.depreciation_factor)
                    ),
                    cost_allocation=self.cost_allocation,
                )

        # Between an instance of Tangible with an instance of Intangible/OPEX/ASR
        elif isinstance(other, (Intangible, OPEX, ASR)):
            # Raise exception error if self.cost_allocation is not equal to other.cost_allocation
            if self.cost_allocation != other.cost_allocation:
                raise TangibleException(
                    "Cost allocation is not equal. "
                    f"First instance is {self.cost_allocation}, "
                    f"second instance is {other.cost_allocation} "
                )

            else:
                return self.expenditures() + other.expenditures()

        # Between an instance of Tangible and an integer/float
        elif isinstance(other, (int, float)):
            return self.expenditures() + other

        else:
            raise TangibleException(
                f"Must add an instance of Tangible with another instance "
                f"of Tangible/Intangible/OPEX/ASR/int/float. "
                f"{other}({other.__class__.__qualname__}) is not an instance of "
                f"Tangible/Intangible/OPEX/ASR/int/float."
            )

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):

        # Between an instance of Tangible and another instance of Tangible
        if isinstance(other, Tangible):

            # Raise exception error if self.cost_allocation is not equal to other.cost_allocation
            if self.cost_allocation != other.cost_allocation:
                raise TangibleException(
                    "Cost allocation is not equal. "
                    f"First instance is {self.cost_allocation}, "
                    f"second instance is {other.cost_allocation}"
                )

            else:
                start_year = min(self.start_year, other.start_year)
                end_year = max(self.end_year, other.end_year)

                return Tangible(
                    start_year=start_year,
                    end_year=end_year,
                    cost=np.concatenate((self.cost, -other.cost)),
                    expense_year=np.concatenate(
                        (self.expense_year, other.expense_year)
                    ),
                    pis_year=np.concatenate((self.pis_year, other.pis_year)),
                    salvage_value=np.concatenate(
                        (self.salvage_value, other.salvage_value)
                    ),
                    useful_life=np.concatenate((self.useful_life, other.useful_life)),
                    depreciation_factor=np.concatenate(
                        (self.depreciation_factor, other.depreciation_factor)
                    ),
                    cost_allocation=self.cost_allocation,
                )

        # Between an instance of Tangible and an instance of Intangible/OPEX/ASR
        elif isinstance(other, (Intangible, OPEX, ASR)):
            if self.cost_allocation != other.cost_allocation:
                raise TangibleException(
                    "Cost allocation is not equal. "
                    f"First instance is {self.cost_allocation}, "
                    f"second instance is {other.cost_allocation}"
                )

            else:
                return self.expenditures() - other.expenditures()

        # Between an instance of Tangible and an integer/float
        elif isinstance(other, (int, float)):
            return self.expenditures() - other

        else:
            raise TangibleException(
                f"Must subtract between an instance of Tangible "
                f"with another instance of Tangible/Intangible/OPEX/ASR/int/float. "
                f"{other}({other.__class__.__qualname__}) is not "
                f"an instance of Tangible/Intangible/OPEX/ASR/int/float."
            )

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):

        # Multiplication is allowed only with an integer/a float
        if isinstance(other, (int, float)):
            return Tangible(
                start_year=self.start_year,
                end_year=self.end_year,
                cost=self.cost * other,
                expense_year=self.expense_year.copy(),
                pis_year=self.pis_year.copy(),
                salvage_value=self.salvage_value.copy(),
                useful_life=self.useful_life.copy(),
                depreciation_factor=self.depreciation_factor.copy(),
                cost_allocation=self.cost_allocation,
                is_ic_applied=self.is_ic_applied,
            )

        else:
            raise TangibleException(
                f"Must multiply with an integer or a float; "
                f"{other} is of {other.__class__.__qualname__} datatype."
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):

        # Between an instance of Tangible with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) / np.sum(other.cost)

        # Between an instance of Tangible and an integer/float
        elif isinstance(other, (int, float)):

            # Cannot divide with zero
            if other == 0:
                raise TangibleException(f"Cannot divide with zero")

            else:
                return Tangible(
                    start_year=self.start_year,
                    end_year=self.end_year,
                    cost=self.cost / other,
                    expense_year=self.expense_year.copy(),
                    pis_year=self.pis_year.copy(),
                    salvage_value=self.salvage_value.copy(),
                    useful_life=self.useful_life.copy(),
                    cost_allocation=self.cost_allocation,
                )

        else:
            raise TangibleException(
                f"Must divide with an instance of Tangible/Intangible/OPEX/ASR, integer or a float; "
                f"{other}({other.__class__.__qualname__}) is not an instance "
                f"of Tangible/Intangible/OPEX/ASR nor an integer nor a float."
            )


@dataclass
class Intangible:
    """
    Manages an intangible asset.

    Parameters
    ----------
    start_year : int
        The start year of the project.
    end_year : int
        The end year of the project.
    cost : numpy.ndarray
        An array representing the cost of an intangible asset.
    expense_year : numpy.ndarray
        An array representing the expense year of an intangible asset.
    cost_allocation : FluidType
        A string depicting the cost allocation of an intangible asset.
    """

    start_year: int
    end_year: int
    cost: np.ndarray
    expense_year: np.ndarray
    cost_allocation: FluidType = field(default=FluidType.OIL)

    # Attribute to be defined later on
    project_duration: int = field(default=None, init=False, repr=False)
    project_years: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self):

        # Check input data for unequal length
        if len(self.expense_year) != len(self.cost):
            raise IntangibleException(
                f"Unequal length of array: "
                f"cost: {len(self.cost)}, "
                f"expense_year: {len(self.expense_year)}"
            )

        # Check for inappropriate start and end year project
        if self.end_year >= self.start_year:
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise IntangibleException(
                f"start year {self.start_year} "
                f"is after the end year {self.end_year}"
            )

        # Raise an error message: expense year is after the end year of the project
        if np.max(self.expense_year) > self.end_year:
            raise IntangibleException(
                f"Expense year ({np.max(self.expense_year)}) "
                f"is after the end year of the project ({self.end_year})"
            )

        # Raise an error message: expense year is before the start year of the project
        if np.min(self.expense_year) < self.start_year:
            raise IntangibleException(
                f"Expense year ({np.min(self.expense_year)}) "
                f"is before the start year of the project ({self.start_year})"
            )

    def expenditures(
        self,
        tax_regime: TaxRegime = TaxRegime.NAILED_DOWN,
        vat_rate: float | int = 0,
        inflation_rate: float | int = 0
    ) -> np.ndarray:
        """
        Calculate intangible expenditures per year.

        This method calculates the intangible expenditures per year
        based on the expense year and cost data provided.

        Parameters
        ----------
        tax_regime: TaxRegime
            The adopted regime/fiscal regulation to determine the
            configuration of VAT calculations.
        vat_rate: float | int
            Rate of Value Added Tax (VAT).
        inflation_rate: float | int
            Rate of inflation.

        Returns
        -------
        expenses: np.ndarray
            An array depicting the intangible expenditures each year, taking
            into account the inflation and Value Added Tax (VAT/PPN).

        Notes
        -----
        (1) Function np.bincount() is used to align the cost elements according
            to its corresponding expense year,
        (2) If len(expenses) < project_duration, then add the remaining elements
            with zeros,
        (3) The return value is modified by taking into account the inflation,
            using the custom decorator @apply_inflation,
        (4) The return value is modified further by taking into account the Value
            Added Tax (VAT/PPN), using the custom decorator @apply_vat.
        """
        @apply_vat(tax_regime=tax_regime, vat_rate=vat_rate)
        @apply_inflation(inflation_rate=inflation_rate)
        def _expenditures() -> np.ndarray:
            expenses = np.bincount(self.expense_year - self.start_year, weights=self.cost)
            zeros = np.zeros(self.project_duration - len(expenses))
            expenses = np.concatenate((expenses, zeros))
            return expenses
        return _expenditures()

    def __len__(self):
        return self.project_duration

    def __eq__(self, other):

        # Between two instances of Intangible
        if isinstance(other, Intangible):
            return all(
                (
                    self.cost_allocation == other.cost_allocation,
                    np.allclose(self.cost, other.cost),
                    np.allclose(self.expense_year, other.expense_year),
                )
            )

        # Between an instance of Intangible and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) == other

        else:
            return False

    def __lt__(self, other):

        # Between an instance of Intangible with another instance of Intangible/Tangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) < np.sum(other.cost)

        # Between an instance of Intangible and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) < other

        else:
            raise IntangibleException(
                f"Must compare an instance of Intangible with another instance of "
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float."
            )

    def __le__(self, other):

        # Between an instance of Intangible with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) <= np.sum(other.cost)

        # Between an instance of Intangible and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) <= other

        else:
            raise IntangibleException(
                f"Must compare an instance of Intangible with another instance of "
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float."
            )

    def __gt__(self, other):

        # Between an instance of Intangible with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) > np.sum(other.cost)

        # Between an instance of Intangible and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) > other

        else:
            raise IntangibleException(
                f"Must compare an instance of Intangible with another instance of "
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float."
            )

    def __ge__(self, other):

        # Between an instance of Intangible with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) >= np.sum(other.cost)

        # Between an instance of Intangible and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) >= other

        else:
            raise IntangibleException(
                f"Must compare an instance of Intangible with another instance of "
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float."
            )

    def __add__(self, other):

        # Between an instance of Intangible and another instance of Intangible
        if isinstance(other, Intangible):

            # Raise an expection error is self.cost_allocation is not equal to other.cost_allocation
            if self.cost_allocation != other.cost_allocation:
                raise IntangibleException(
                    "Cost allocation is not equal. "
                    f"First instance is {self.cost_allocation}, "
                    f"second instance is {other.cost_allocation} "
                )

            else:
                combined_start_year = min(self.start_year, other.start_year)
                combined_end_year = max(self.end_year, other.end_year)

                return Intangible(
                    start_year=combined_start_year,
                    end_year=combined_end_year,
                    cost=np.concatenate((self.cost, other.cost)),
                    expense_year=np.concatenate((self.expense_year, other.expense_year)),
                    cost_allocation=self.cost_allocation,
                )

        else:
            raise IntangibleException(
                f"Must add between an instance of Intangible "
                f"with another instance of Intangible. "
                f"{other}({other.__class__.__qualname__}) is not "
                f"an instance of Intangible."
            )

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):

        # Between an instance of Intangible and another instance of Intangible
        if isinstance(other, Intangible):

            # Raise exception error if self.cost_allocation is not equal to other.cost_allocation
            if self.cost_allocation != other.cost_allocation:
                raise IntangibleException(
                    "Cost allocation is not equal. "
                    f"First instance is {self.cost_allocation}, "
                    f"second instance is {other.cost_allocation}"
                )

            else:
                combined_start_year = min(self.start_year, other.start_year)
                combined_end_year = max(self.end_year, other.end_year)

                return Intangible(
                    start_year=combined_start_year,
                    end_year=combined_end_year,
                    cost=np.concatenate((self.cost, -other.cost)),
                    expense_year=np.concatenate((self.expense_year, other.expense_year)),
                    cost_allocation=self.cost_allocation,
                )

        else:
            raise IntangibleException(
                f"Must subtract between an instance of Intangible "
                f"with another instance of Intangible. "
                f"{other}({other.__class__.__qualname__}) is not "
                f"an instance of Intangible."
            )

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):

        # Multiplication is allowed only with an integer/a float
        if isinstance(other, (int, float)):
            return Intangible(
                start_year=self.start_year,
                end_year=self.end_year,
                cost=self.cost * other,
                expense_year=self.expense_year.copy(),
                cost_allocation=self.cost_allocation,
            )

        else:
            raise IntangibleException(
                f"Must multiply with an integer or a float; "
                f"{other}({other.__class__.__qualname__}) is not an integer nor a float."
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):

        # Between an instance of Intangible with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) / np.sum(other.cost)

        # Between an instance of Intangible and an integer/float
        elif isinstance(other, (int, float)):

            # Cannot divide with zero
            if other == 0:
                raise IntangibleException(f"Cannot divide with zero")

            else:
                return Intangible(
                    start_year=self.start_year,
                    end_year=self.end_year,
                    cost=self.cost / other,
                    expense_year=self.expense_year.copy(),
                    cost_allocation=self.cost_allocation,
                )

        else:
            raise IntangibleException(
                f"Must divide with an instance of Tangible/Intangible/OPEX/ASR, integer or a float; "
                f"{other}({other.__class__.__qualname__}) is not an instance "
                f"of Tangible/Intangible/OPEX/ASR nor an integer nor a float."
            )


@dataclass
class OPEX:
    """
    Manages an OPEX asset.

    Parameters
    ----------
    start_year : int
        The start year of the project.
    end_year : int
        The end year of the project.
    fixed_cost : np.ndarray
        An array representing the fixed cost of an OPEX asset.
    expense_year : np.ndarray
        An array representing the expense year of an intangible asset.
    cost_allocation : FluidType
        A string depicting the cost allocation of an OPEX asset.
    prod_rate: np.ndarray
        The production rate of a particular fluid type.
    cost_per_volume: np.ndarray
        Cost associated with production of a particular fluid type.
    """

    start_year: int
    end_year: int
    fixed_cost: np.ndarray
    expense_year: np.ndarray
    cost_allocation: FluidType = field(default=FluidType.OIL)
    prod_rate: np.ndarray = field(default=None, repr=False)
    cost_per_volume: np.ndarray = field(default=None, repr=False)

    # Attribute to be defined later on
    variable_cost: np.ndarray = field(default=None, init=False, repr=False)
    cost: np.ndarray = field(default=None, init=False, repr=False)
    project_duration: int = field(default=None, init=False, repr=False)
    project_years: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self):

        # Check for inappropriate start and end year project
        if self.end_year >= self.start_year:
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise OPEXException(
                f"start year {self.start_year} "
                f"is after the end year {self.end_year}"
            )

        # Raise an error message: expense year is after the end year of the project
        if np.max(self.expense_year) > self.end_year:
            raise OPEXException(
                f"Expense year ({np.max(self.expense_year)}) "
                f"is after the end year of the project ({self.end_year})"
            )

        # Raise an error message: expense year is before the start year of the project
        if np.min(self.expense_year) < self.start_year:
            raise OPEXException(
                f"Expense year ({np.min(self.expense_year)}) "
                f"is before the start year of the project ({self.start_year})"
            )

        # User provides both prod_rate and cost_per_volume data
        if self.prod_rate is not None and self.cost_per_volume is not None:

            # Check input data for unequal length
            if not all(
                len(i) == len(self.fixed_cost)
                for i in [self.expense_year, self.prod_rate, self.cost_per_volume]
            ):
                raise OPEXException(
                    f"Unequal length of array: "
                    f"fixed_cost: {len(self.fixed_cost)}, "
                    f"expense_year: {len(self.expense_year)}, "
                    f"prod_rate: {len(self.prod_rate)}, "
                    f"cost_per_volume: {len(self.cost_per_volume)}"
                )

            # Specify attribute variable_cost
            self.variable_cost = self.prod_rate * self.cost_per_volume

        # User only provides prod_rate data
        elif self.prod_rate is not None and self.cost_per_volume is None:
            raise OPEXException(f"cost_per_volume data is missing")

        # User only provides cost_per_volume data
        elif self.prod_rate is None and self.cost_per_volume is not None:
            raise OPEXException(f"prod_rate data is missing")

        # User does not provide both prod_rate and cost_per_volume data
        elif self.prod_rate is None and self.cost_per_volume is None:
            self.prod_rate = np.zeros_like(self.fixed_cost)
            self.cost_per_volume = np.zeros_like(self.fixed_cost)
            self.variable_cost = np.zeros_like(self.fixed_cost)

        # Define cost
        self.cost = self.fixed_cost + self.variable_cost

    def expenditures(
        self,
        tax_regime: TaxRegime = TaxRegime.NAILED_DOWN,
        vat_rate: float | int = 0,
        inflation_rate: float | int = 0
    ) -> np.ndarray:
        """
        Calculate OPEX expenditures per year.
        Allocate OPEX expenditures following the associated expense year.

        Parameters
        ----------
        tax_regime: TaxRegime
            The adopted regime/fiscal regulation to determine the
            configuration of VAT calculations.
        vat_rate: float | int
            Rate of Value Added Tax (VAT).
        inflation_rate: float | int
            Rate of inflation.

        Returns
        -------
        expenses: np.ndarray
            An array of OPEX expenses aligned with the expense year.

        Notes
        -----
        (1) Function np.bincount() is used to align the cost elements according
            to its corresponding expense year,
        (2) If len(expenses) < project_duration, then add the remaining elements
            with zeros,
        (3) The return value is modified by taking into account the inflation,
            using the custom decorator @apply_inflation,
        (4) The return value is modified further by taking into account the Value
            Added Tax (VAT/PPN), using the custom decorator @apply_vat.
        """
        @apply_vat(tax_regime=tax_regime, vat_rate=vat_rate)
        @apply_inflation(inflation_rate=inflation_rate)
        def _expenditures() -> np.ndarray:
            expenses = np.bincount(self.expense_year - self.start_year, weights=self.cost)
            zeros = np.zeros(self.project_duration - len(expenses))
            expenses = np.concatenate((expenses, zeros))
            return expenses
        return _expenditures()

    def __len__(self):
        return self.project_duration

    def __eq__(self, other):

        # Between two instances of OPEX
        if isinstance(other, OPEX):
            return all(
                (
                    self.cost_allocation == other.cost_allocation,
                    np.allclose(self.fixed_cost, other.fixed_cost),
                    np.allclose(self.variable_cost, other.variable_cost),
                    np.allclose(self.expense_year, other.expense_year),
                )
            )

        # Between an instance of OPEX and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.fixed_cost, self.variable_cost) == other

        else:
            return False

    def __lt__(self, other):

        # Between an instance of OPEX with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) < np.sum(other.cost)

        # Between an instance of OPEX and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) < other

        else:
            raise OPEXException(
                f"Must compare an instance of OPEX with another instance of "
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float"
            )

    def __le__(self, other):

        # Between an instance of OPEX with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) <= np.sum(other.cost)

        # Between an instance of OPEX and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) <= other

        else:
            raise OPEXException(
                f"Must compare an instance of OPEX with another instance of "
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float"
            )

    def __gt__(self, other):

        # Between an instance of OPEX with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) > np.sum(other.cost)

        # Between an instance of OPEX and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) > other

        else:
            raise OPEXException(
                f"Must compare an instance of OPEX with another instance of "
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float"
            )

    def __ge__(self, other):

        # Between an instance of OPEX with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) >= np.sum(other.cost)

        # Between an instance of OPEX and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) >= other

        else:
            raise OPEXException(
                f"Must compare an instance of OPEX with another instance of "
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float"
            )

    def __add__(self, other):

        # Between an instance of OPEX with another instance of OPEX
        if isinstance(other, OPEX):

            # Raise exception error if self.cost_allocation is not equal to other.cost_allocation
            if self.cost_allocation != other.cost_allocation:
                raise TangibleException(
                    "Cost allocation is not equal. "
                    f"First instance is {self.cost_allocation}, "
                    f"second instance is {other.cost_allocation} "
                )

            else:
                combined_start_year = min(self.start_year, other.start_year)
                combined_end_year = max(self.end_year, other.end_year)

                return OPEX(
                    start_year=combined_start_year,
                    end_year=combined_end_year,
                    fixed_cost=np.concatenate((self.fixed_cost, other.fixed_cost)),
                    expense_year=np.concatenate((self.expense_year, other.expense_year)),
                    cost_allocation=self.cost_allocation,
                    prod_rate=np.concatenate((self.prod_rate, other.prod_rate)),
                    cost_per_volume=np.concatenate((self.cost_per_volume, other.cost_per_volume)),
                )

        else:
            raise OPEXException(
                f"Must add an instance of OPEX with another "
                f"instance of OPEX. "
                f"{other}({other.__class__.__qualname__}) is not "
                f"an instance of OPEX."
            )

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):

        # Between an instance of OPEX with another instance of OPEX
        if isinstance(other, OPEX):

            # Raise exception error if self.cost_allocation is not equal to other.cost_allocation
            if self.cost_allocation != other.cost_allocation:
                raise OPEXException(
                    "Cost allocation is not equal. "
                    f"First instance is {self.cost_allocation}, "
                    f"second instance is {other.cost_allocation}"
                )

            else:
                combined_start_year = min(self.start_year, other.start_year)
                combined_end_year = max(self.end_year, other.end_year)

                return OPEX(
                    start_year=combined_start_year,
                    end_year=combined_end_year,
                    fixed_cost=np.concatenate((self.fixed_cost, -other.fixed_cost)),
                    expense_year=np.concatenate((self.expense_year, other.expense_year)),
                    cost_allocation=self.cost_allocation,
                    prod_rate=np.concatenate((self.prod_rate, -other.prod_rate)),
                    cost_per_volume=np.concatenate((self.cost_per_volume, other.cost_per_volume)),
                )

        else:
            raise OPEXException(
                f"Must add between an instance of OPEX "
                f"with another instance of OPEX. "
                f"{other}({other.__class__.__qualname__}) is "
                f"not an instance of OPEX"
            )

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):

        # Multiplication is allowed only with an integer/a float
        if isinstance(other, (int, float)):
            return OPEX(
                start_year=self.start_year,
                end_year=self.end_year,
                fixed_cost=self.fixed_cost * other,
                expense_year=self.expense_year,
                cost_allocation=self.cost_allocation,
                prod_rate=self.prod_rate * other,
                cost_per_volume=self.cost_per_volume,
            )

        else:
            raise OPEXException(
                f"Must multiply with an integer or a float; "
                f"{other}({other.__class__.__qualname__}) is not an integer nor a float."
            )

    def __rmul__(self, other):
        raise self.__mul__(other)

    def __truediv__(self, other):

        # Between an instance of OPEX with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) / np.sum(other.cost)

        # Between an instance of OPEX and an integer/float
        elif isinstance(other, (int, float)):

            # Cannot divide with zero
            if other == 0:
                raise OPEXException(f"Cannot divide with zero")

            else:
                return OPEX(
                    start_year=self.start_year,
                    end_year=self.end_year,
                    fixed_cost=self.fixed_cost / other,
                    expense_year=self.expense_year,
                    cost_allocation=self.cost_allocation,
                    prod_rate=self.prod_rate / other,
                    cost_per_volume=self.cost_per_volume,
                )

        else:
            raise OPEXException(
                f"Must divide with an instance of Tangible/Intangible/OPEX/ASR, integer or a float; "
                f"{other}({other.__class__.__qualname__}) is not an instance "
                f"of Tangible/Intangible/OPEX/ASR nor an integer nor a float."
            )


@dataclass
class ASR:
    """
    Manages an ASR asset.

    Parameters
    ----------
    start_year : int
        The start year of the project.
    end_year : int
        The end year of the project.
    cost : numpy.ndarray
        An array representing the cost of an intangible asset.
    expense_year : numpy.ndarray
        An array representing the expense year of an intangible asset.
    cost_allocation : FluidType
        A string depicting the cost allocation of an intangible asset.
    """

    start_year: int
    end_year: int
    cost: np.ndarray
    expense_year: np.ndarray
    cost_allocation: FluidType = field(default=FluidType.OIL)

    # Attribute to de defined later on
    project_duration: int = field(default=None, init=False, repr=False)
    project_years: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self):

        # Check input data for unequal length
        if len(self.expense_year) != len(self.cost):
            raise ASRException(
                f"Unequal length of array: "
                f"cost: {len(self.cost)}, "
                f"expense_year: {len(self.expense_year)}"
            )

        # Check for inappropriate start and end year project
        if self.end_year >= self.start_year:
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise ASRException(
                f"start year {self.start_year} "
                f"is after the end year {self.end_year}"
            )

        # Raise an error message: expense year is after the end year of the project
        if np.max(self.expense_year) > self.end_year:
            raise ASRException(
                f"Expense year ({np.max(self.expense_year)}) "
                f"is after the end year of the project ({self.end_year})"
            )

        # Raise an error message: expense year is before the start year of the project
        if np.min(self.expense_year) < self.start_year:
            raise ASRException(
                f"Expense year ({np.min(self.expense_year)}) "
                f"is before the start year of the project ({self.start_year})"
            )

    def future_cost(
        self,
        rate: float = 0.02,
        tax_regime: TaxRegime = TaxRegime.NAILED_DOWN,
        vat_rate: float | int = 0,
        inflation_rate: float | int = 0
    ) -> np.ndarray:
        """
        Calculate the future cost of an asset.

        Parameters
        ----------
        rate: float
            An exponent value in the equation to determine future value of an asset.
        tax_regime : TaxRegime, optional
            The adopted regime/fiscal regulation to determine the configuration
            of VAT calculations (default is TaxRegime.NAILED_DOWN).
        vat_rate : float or int, optional
            The Value Added Tax (VAT) rate as a decimal or integer (default is 0).
        inflation_rate : float or int, optional
            The annual inflation rate as a decimal or integer (default is 0).

        Returns
        -------
        future_cost: np.ndarray
            An array depicting the future cost of an asset.

        Notes
        -----
        (1) The value of cost is modified by taking into account VAT/PPN
            and inflation,
        (2) Future cost is determined based on the modified cost.
        """

        # Apply VAT/PPN to cost
        if tax_regime == TaxRegime.NAILED_DOWN:
            modified_cost = np.float_(self.cost * (1 + vat_rate))

        # Apply inflation to cost
        exponents = self.expense_year - self.start_year
        inflation_arr = (1. + inflation_rate) ** exponents
        modified_cost *= inflation_arr

        return modified_cost * np.power((1 + rate), self.end_year - self.expense_year + 1)

    def expenditures(
        self,
        rate: float = 0.02,
        tax_regime: TaxRegime = TaxRegime.NAILED_DOWN,
        vat_rate: float | int = 0,
        inflation_rate: float | int = 0
    ) -> np.ndarray:
        """
        Calculate ASR expenditures per year.

        This method calculates the ASR expenditures per year
        based on the expense year and cost data provided.

        Parameters
        ----------
        rate: float
            An exponent value in the equation to determine future value of an asset.
        tax_regime : TaxRegime, optional
            The adopted regime/fiscal regulation to determine the configuration
            of VAT calculations (default is TaxRegime.NAILED_DOWN).
        vat_rate : float or int, optional
            The Value Added Tax (VAT) rate as a decimal or integer (default is 0).
        inflation_rate : float or int, optional
            The annual inflation rate as a decimal or integer (default is 0).

        Returns
        -------
        expenses: np.ndarray
            An array of ASR expenses aligned with the expense year.

        Notes
        -----
        The value of cost is modified by taking into account VAT/PPN
        and inflation.
        """

        # Distance of expense year from the end year of the project
        cost_duration = self.end_year - self.expense_year + 1

        # Cost allocation: cost distribution per year
        cost_alloc = self.future_cost(
            rate=rate, tax_regime=tax_regime, vat_rate=vat_rate, inflation_rate=inflation_rate
        ) / cost_duration

        # Distance of expense year from the start year of the project
        shift_indices = self.expense_year - self.start_year

        # ASR allocation per element per distributed year
        asr_alloc = np.asarray(
            [
                np.concatenate((np.zeros(i), np.repeat(ca, cd)))
                for i, ca, cd in zip(shift_indices, cost_alloc, cost_duration)
            ]
        )

        return asr_alloc.sum(axis=0)

    def __len__(self):
        return self.project_duration

    def __eq__(self, other):

        # Between two instances of ASR
        if isinstance(other, ASR):
            return all(
                (
                    self.cost_allocation == other.cost_allocation,
                    np.allclose(self.cost, other.cost),
                    np.allclose(self.expense_year, other.expense_year),
                )
            )

        # Between an instance of ASR and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) == other

        else:
            return False

    def __lt__(self, other):

        # Between an instance of ASR with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) < np.sum(other.cost)

        # Between an instance of ASR with an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) < other

        else:
            raise ASRException(
                f"Must compare an instance of ASR with another instance of "
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float."
            )

    def __le__(self, other):

        # Between an instance of ASR with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) <= np.sum(other.cost)

        # Between an instance of ASR and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) <= other

        else:
            raise ASRException(
                f"Must compare an instance of ASR with another instance of "
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float."
            )

    def __gt__(self, other):

        # Between an instance of ASR with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) > np.sum(other.cost)

        # Between an instance of ASR with an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) > other

        else:
            raise ASRException(
                f"Must compare an instance of ASR with another instance of "
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float."
            )

    def __ge__(self, other):

        # Between an instance of ASR with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) >= np.sum(other.cost)

        # Between an instance of ASR and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) >= other

        else:
            raise ASRException(
                f"Must compare an instance of ASR with another instance of "
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float."
            )

    def __add__(self, other):

        # Between an instance of ASR with an instance of ASR
        if isinstance(other, ASR):
            if self.cost_allocation != other.cost_allocation:
                raise ASRException(
                    "Cost allocation is not equal. "
                    f"First instance is {self.cost_allocation}, "
                    f"second instance is {other.cost_allocation}"
                )

            else:
                combined_start_year = min(self.start_year, other.start_year)
                combined_end_year = max(self.end_year, other.end_year)

                return ASR(
                    start_year=combined_start_year,
                    end_year=combined_end_year,
                    cost=np.concatenate((self.cost, other.cost)),
                    expense_year=np.concatenate((self.expense_year, other.expense_year)),
                    cost_allocation=self.cost_allocation,
                )

        else:
            raise ASRException(
                f"Must add between an instance of ASR "
                f"with another instance of ASR. "
                f"{other}({other.__class__.__qualname__}) is "
                f"not an instance of ASR."
            )

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):

        # Between an instance of ASR with an intance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, ASR):

            # Raise exception error if self.cost_allocation is not equal to other.cost_allocation
            if self.cost_allocation != other.cost_allocation:
                raise ASRException(
                    "Cost allocation is not equal. "
                    f"First instance is {self.cost_allocation}, "
                    f"second instance is {other.cost_allocation}"
                )

            else:
                combined_start_year = min(self.start_year, other.start_year)
                combined_end_year = max(self.end_year, other.end_year)

                return ASR(
                    start_year=combined_start_year,
                    end_year=combined_end_year,
                    cost=np.concatenate((self.cost, -other.cost)),
                    expense_year=np.concatenate((self.expense_year, other.expense_year)),
                    cost_allocation=self.cost_allocation,
                )

        else:
            raise ASRException(
                f"Must subtract between an instance of ASR "
                f"with another instance of ASR. "
                f"{other}({other.__class__.__qualname__}) is not "
                f"an instance of ASR."
            )

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):

        # Multiplication is allowed only with an integer/a float
        if isinstance(other, (int, float)):
            return ASR(
                start_year=self.start_year,
                end_year=self.end_year,
                cost=self.cost * other,
                expense_year=self.expense_year,
                cost_allocation=self.cost_allocation,
            )

        else:
            raise ASRException(
                f"Must multiply with an integer or a float; "
                f"{other}({other.__class__.__qualname__}) is not an integer nor a float."
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):

        # Between an instance of ASR with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) / np.sum(other.cost)

        # Between an instance of ASR and an integer/float
        elif isinstance(other, (int, float)):

            # Cannot divide with zero
            if other == 0:
                raise ASRException(f"Cannot divide with zero")

            else:
                return ASR(
                    start_year=self.start_year,
                    end_year=self.end_year,
                    cost=self.cost / other,
                    expense_year=self.expense_year,
                    cost_allocation=self.cost_allocation,
                )

        else:
            raise ASRException(
                f"Must divide with an instance of Tangible/Intangible/OPEX/ASR, integer or a float; "
                f"{other}({other.__class__.__qualname__}) is not an instance "
                f"of Tangible/Intangible/OPEX/ASR nor an integer nor a float."
            )
