"""
Prepare and classify cost data based on its components. The associated cost components are:
(1) Tangible,
(2) Intangible,
(3) OPEX,
(4) ASR,
(5) LBT
"""

import numpy as np
from dataclasses import dataclass, field

import pyscnomics.econ.depreciation as depr
from pyscnomics.tools.helper import apply_cost_adjustment
from pyscnomics.econ.selection import FluidType, DeprMethod
# from pyscnomics.econ.selection import TaxType


class CapitalException(Exception):
    """Exception to be raised if class Capital Cost is misused"""

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


class LBTException(Exception):
    """Exception to be raised when LBT class is misused"""

    pass


class ASRCalculatorException(Exception):
    """Exception to be raised for an incorrect use of class ASRCalculator"""

    pass


class CostOfSalesException(Exception):
    """Exception to be raised if class CostOfSales is misused"""

    pass


@dataclass
class GeneralCost:
    """
    Manages a GeneralCost asset.

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
    cost_allocation : list[FluidType]
        A list representing the cost allocation of a tangible asset.
    description: list[str]
        A list of string description regarding the associated tangible cost.

    Notes
    -----
    The unit used in the cost should be in M unit of United States Dollar (USD),
    where the M is stands for 1000. Thus, the unit cost should be: M-USD.
    """
    start_year: int
    end_year: int
    cost: np.ndarray
    expense_year: np.ndarray
    cost_allocation: list[FluidType] = field(default=None)
    description: list[str] = field(default=None)

    # Attribute to be defined later on
    project_duration: int = field(default=None, init=False, repr=False)
    project_years: np.ndarray = field(default=None, init=False, repr=False)

    def __len__(self):
        return self.project_duration


@dataclass
class CapitalCost(GeneralCost):
    """
    Manages a capital asset.

    Parameters
    ----------
    The attributes are inherited from class GeneralCost. Local attributes associated
    with class CapitalCost are:

    pis_year : numpy.ndarray, optional
        An array representing the PIS year of a capital asset.
    salvage_value : numpy.ndarray, optional
        An array representing the salvage value of a capital asset.
    useful_life : numpy.ndarray, optional
        An array representing the useful life of a capital asset.
    depreciation_factor: np.ndarray, optional
        The value of depreciation factor to be used in PSC_DB depreciation method.
        Default value is 0.5 for the entire project duration.
    is_ic_applied: list
        Whether investment credit is applied for each corresponding capital assets.
    vat_portion: np.ndarray
        The portion of 'cost' that is subject to VAT.
        Must be an array of length equals to the length of 'cost' array.
    vat_discount: float
        The VAT discount to apply.
    """

    pis_year: np.ndarray = field(default=None)
    salvage_value: np.ndarray = field(default=None)
    useful_life: np.ndarray = field(default=None)
    depreciation_factor: np.ndarray = field(default=None)
    is_ic_applied: list[bool] = field(default=None)
    vat_portion: np.ndarray = field(default=None)
    vat_discount: float | np.ndarray = field(default=0.0)

    def __post_init__(self):
        # Prepare attribute project_duration and project_years
        if self.end_year >= self.start_year:
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise CapitalException(
                f"start year {self.start_year} "
                f"is after the end year: {self.end_year}"
            )

        # Prepare attribute VAT portion
        if self.vat_portion is None:
            self.vat_portion = np.zeros_like(self.cost)

        elif self.vat_portion is not None:
            if not isinstance(self.vat_portion, np.ndarray):
                raise CapitalException(
                    f"Attribute VAT portion must be a numpy ndarray; "
                    f"VAT portion ({self.vat_portion}) is of datatype "
                    f"{self.vat_portion.__class__.__qualname__}."
                )

        self.vat_portion = self.vat_portion.astype(np.float64)

        # Prepare attribute VAT discount
        if not isinstance(self.vat_discount, (float, np.ndarray)):
            raise CapitalException(
                f"Attribute VAT discount must be given as a float (for single value) "
                f"or a numpy array (for an array). The inserted VAT discount is given "
                f"as a {self.vat_discount.__class__.__qualname__}."
            )

        if isinstance(self.vat_discount, float):
            self.vat_discount = np.repeat(self.vat_discount, len(self.cost))

        self.vat_discount = self.vat_discount.astype(np.float64)

        # Prepare attribute description
        if self.description is None:
            self.description = [" " for _ in range(len(self.cost))]

        elif self.description is not None:
            if not isinstance(self.description, list):
                raise CapitalException(
                    f"Attribute description must be given as a list; "
                    f"description (datatype: {self.description.__class__.__qualname__}) "
                    f"is not a list."
                )

        # Prepare attribute cost_allocation
        if self.cost_allocation is None:
            self.cost_allocation = [FluidType.OIL for _ in range(len(self.cost))]

        elif self.cost_allocation is not None:
            if not isinstance(self.cost_allocation, list):
                raise CapitalException(
                    f"Attribute cost_allocation must be given as a list; "
                    f"cost_allocation (datatype: {self.cost_allocation.__class__.__qualname__}) "
                    f"is not a list."
                )

        # Prepare attribute pis_year
        if self.pis_year is None:
            self.pis_year = self.expense_year.copy()

        elif self.pis_year is not None:
            if not isinstance(self.pis_year, np.ndarray):
                raise CapitalException(
                    f"Attribute pis_year must be given as a numpy.ndarray; "
                    f"pis_year (datatype: {self.pis_year.__class__.__qualname__}) "
                    f"is not a numpy.ndarray."
                )

        # Prepare attribute salvage_value
        if self.salvage_value is None:
            self.salvage_value = np.zeros_like(self.cost)

        elif self.salvage_value is not None:
            if not isinstance(self.salvage_value, np.ndarray):
                raise CapitalException(
                    f"Attribute salvage_value must be given as a numpy.ndarray; "
                    f"salvage_value (datatype: {self.salvage_value.__class__.__qualname__}) "
                    f"is not a numpy.ndarray."
                )

        self.salvage_value = self.salvage_value.astype(np.float64)

        # Prepare attribute useful_life
        if self.useful_life is None:
            self.useful_life = np.repeat(5.0, len(self.cost))

        elif self.useful_life is not None:
            if not isinstance(self.useful_life, np.ndarray):
                raise CapitalException(
                    f"Attribute useful_life must be given as a numpy.ndarray; "
                    f"useful_life (datatype: {self.useful_life.__class__.__qualname__}) "
                    f"is not a numpy.ndarray."
                )

        self.useful_life = self.useful_life.astype(np.float64)

        # Prepare attribute depreciation_factor
        if self.depreciation_factor is None:
            self.depreciation_factor = np.repeat(0.5, len(self.cost))

        elif self.depreciation_factor is not None:
            if not isinstance(self.depreciation_factor, np.ndarray):
                raise CapitalException(
                    f"Attribute depreciation_factor must be given as a numpy.ndarray; "
                    f"depreciation_factor (datatype: {self.depreciation_factor.__class__.__qualname__}) "
                    f"is not a numpy.ndarray."
                )

        self.depreciation_factor = self.depreciation_factor.astype(np.float64)

        # Prepare attribute is_ic_applied
        if self.is_ic_applied is None:
            self.is_ic_applied = [False for _ in range(len(self.cost))]

        elif self.is_ic_applied is not None:
            if not isinstance(self.is_ic_applied, list):
                raise CapitalException(
                    f"Attribute is_ic_applied must be given as a list; "
                    f"is_ic_applied (datatype: {self.is_ic_applied.__class__.__qualname__}) "
                    f"is not a list."
                )

        # Check input data for unequal length
        arr_length = len(self.cost)

        if not all(
            len(arr) == arr_length
            for arr in [
                self.expense_year,
                self.cost_allocation,
                self.description,
                self.vat_portion,
                self.vat_discount,
                self.pis_year,
                self.salvage_value,
                self.useful_life,
                self.depreciation_factor,
                self.is_ic_applied,
            ]
        ):
            raise CapitalException(
                f"Unequal length of array: "
                f"cost: {len(self.cost)}, "
                f"expense_year: {len(self.expense_year)}, "
                f"cost_allocation: {len(self.cost_allocation)}, "
                f"description: {len(self.description)}, "
                f"vat_portion: {len(self.vat_portion)}, "
                f"vat_discount: {len(self.vat_discount)}, "
                f"pis_year: {len(self.pis_year)}, "
                f"salvage_value: {len(self.salvage_value)}, "
                f"useful_life: {len(self.useful_life)}, "
                f"depreciation_factor: {len(self.depreciation_factor)}, "
                f"is_ic_applied: {len(self.is_ic_applied)}."
            )

        # Raise an error message: expense year is after the end year of the project
        if np.max(self.expense_year) > self.end_year:
            raise CapitalException(
                f"Expense year ({np.max(self.expense_year)}) "
                f"is after the end year of the project ({self.end_year})"
            )

        # Raise an error message: expense year is before the start year of the project
        if np.min(self.expense_year) < self.start_year:
            raise CapitalException(
                f"Expense year ({np.min(self.expense_year)}) "
                f"is before the start year of the project ({self.start_year})"
            )

    def expenditures(
        self,
        year_ref: np.ndarray = None,
        vat_rate: np.ndarray | float = 0.0,
        inflation_rate: np.ndarray | float = 0.0,
    ) -> np.ndarray:
        """
        Calculate expenditures per year.

        This method calculates the expenditures per year based on the expense year
        and cost data provided.

        Parameters
        ----------
        year_ref : np.ndarray
            The reference year for inflation calculation.
        vat_rate: np.ndarray | float
            The VAT rate to apply. Can be a single value or an array (default is 0.0).
        inflation_rate: np.ndarray | float
            The inflation rate to apply. Can be a single value or an array (default is 0.0).

        Returns
        -------
        np.ndarray
            An array depicting the expenditures each year, adjusted by tax
            and inflation schemes (if any).

        Notes
        -----
        This method calculates expenditures while considering tax and inflation schemes.
        The core calculations are as follows:
        (1) Apply adjustment to cost due to tax and inflation (if any), by calling
            'apply_cost_adjustment()' function,
        (2) Function np.bincount() is used to align the 'cost_adjusted' elements
            according to its corresponding expense year,
        (3) If len(expenses) < project_duration, then add the remaining elements
            with zeros.
        """
        cost_adjusted = apply_cost_adjustment(
            start_year=self.start_year,
            end_year=self.end_year,
            cost=self.cost,
            expense_year=self.expense_year,
            project_years=self.project_years,
            year_ref=year_ref,
            tax_portion=self.vat_portion,
            tax_rate=vat_rate,
            tax_discount=self.vat_discount,
            inflation_rate=inflation_rate,
        )

        expenses = np.bincount(
            self.expense_year - self.start_year, weights=cost_adjusted
        )
        zeros = np.zeros(self.project_duration - len(expenses))

        return np.concatenate((expenses, zeros))

    def total_depreciation_rate(
        self,
        depr_method: DeprMethod = DeprMethod.PSC_DB,
        decline_factor: float | int = 2,
        year_ref: np.ndarray = None,
        vat_rate: np.ndarray | float = 0.0,
        inflation_rate: np.ndarray | float = 0.0,
    ) -> tuple:
        """
        Calculate total depreciation charge and undepreciated asset value based on various parameters.

        Parameters
        ----------
        depr_method : DeprMethod, optional
            The depreciation method to use (default is DeprMethod.PSC_DB).
        decline_factor : float | int, optional
            The decline factor used for declining balance depreciation (default is 2).
        year_ref : np.ndarray
            The reference year for inflation calculation.
        vat_rate: np.ndarray | float
            The VAT rate to apply. Can be a single value or an array (default is 0.0).
        inflation_rate: np.ndarray | float
            The inflation rate to apply. Can be a single value or an array (default is 0.0).

        Returns
        -------
        tuple
            A tuple containing:
            (1) Total depreciation charges for each period.
            (2) The undepreciated asset value at the end of the analysis.

        Notes
        ------
        (1) This method calculates depreciation charges based on the specified
            depreciation method.
        (2) Prior to the core calculations, attribute 'cost' is adjusted by tax
            and inflation schemes (if any),
        (3) The depreciation charges are aligned with the corresponding periods
            based on pis_year.
        """
        cost_adjusted = apply_cost_adjustment(
            start_year=self.start_year,
            end_year=self.end_year,
            cost=self.cost,
            expense_year=self.expense_year,
            project_years=self.project_years,
            year_ref=year_ref,
            tax_portion=self.vat_portion,
            tax_rate=vat_rate,
            tax_discount=self.vat_discount,
            inflation_rate=inflation_rate,
        )

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
                        cost_adjusted,
                        self.salvage_value,
                        self.useful_life,
                    )
                ]
            )

        # Declining balance/double declining balance
        elif depr_method == DeprMethod.DB:
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
                        cost_adjusted,
                        self.salvage_value,
                        self.useful_life,
                    )
                ]
            )

        # PSC_DB
        elif depr_method == DeprMethod.PSC_DB:
            depreciation_charge = np.asarray(
                [
                    depr.psc_declining_balance_depreciation_rate(
                        cost=c,
                        depreciation_factor=dr,
                        useful_life=ul,
                        depreciation_len=self.project_duration,
                    )
                    for c, dr, ul in zip(
                        cost_adjusted,
                        self.depreciation_factor,
                        self.useful_life,
                    )
                ]
            )

        else:
            raise CapitalException(
                f"Depreciation method ({depr_method}) is unrecognized."
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
        undepreciated_asset = np.sum(cost_adjusted) - np.sum(total_depreciation_charge)

        return total_depreciation_charge, undepreciated_asset

    def total_depreciation_book_value(
        self,
        depr_method: DeprMethod = DeprMethod.PSC_DB,
        decline_factor: float | int = 2,
        year_ref: np.ndarray = None,
        vat_rate: np.ndarray | float = 0.0,
        inflation_rate: np.ndarray | float = 0.0,
    ) -> np.ndarray:
        """
        Calculate the total book value of depreciation for the asset.

        Parameters
        ----------
        depr_method : DeprMethod, optional
            The depreciation method to use (default is DeprMethod.PSC_DB).
        decline_factor : float, optional
            The decline factor used for declining balance depreciation (default is 2).
        year_ref : np.ndarray
            The reference year for inflation calculation.
        vat_rate: np.ndarray | float
            The VAT rate to apply. Can be a single value or an array (default is 0.0).
        inflation_rate: np.ndarray | float
            The inflation rate to apply. Can be a single value or an array (default is 0.0).

        Returns
        -------
        np.ndarray
            An array containing the cumulative book value of depreciation for each period,
            taking into account tax and inflation schemes.

        Notes
        -----
        (1) This method calculates the cumulative book value of depreciation for the asset
            based on the specified depreciation method and other parameters.
        (2) The cumulative book value is obtained by subtracting the cumulative
            depreciation charges from the cumulative expenditures.
        """
        # Calculate total depreciation charge from method total_depreciation_rate
        total_depreciation_charge = self.total_depreciation_rate(
            depr_method=depr_method,
            decline_factor=decline_factor,
            year_ref=year_ref,
            vat_rate=vat_rate,
            inflation_rate=inflation_rate,
        )[0]

        return np.cumsum(
            self.expenditures(
                year_ref=year_ref,
                vat_rate=vat_rate,
                inflation_rate=inflation_rate,
            )
        ) - np.cumsum(total_depreciation_charge)

    def __eq__(self, other):
        # Between two instances of CapitalCost
        if isinstance(other, CapitalCost):
            return all(
                (
                    np.allclose(self.cost, other.cost),
                    np.allclose(self.expense_year, other.expense_year),
                    np.allclose(self.pis_year, other.pis_year),
                    np.allclose(self.salvage_value, other.salvage_value),
                    np.allclose(self.useful_life, other.useful_life),
                    np.allclose(self.depreciation_factor, other.depreciation_factor),
                    np.allclose(self.vat_portion, other.vat_portion),
                    np.allclose(self.vat_discount, other.vat_discount),
                    self.cost_allocation == other.cost_allocation,
                )
            )

        # Between an instance of CapitalCost and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) == other

        else:
            return False

    def __lt__(self, other):
        # Between an instance of CapitalCost with another instance of CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) < np.sum(other.cost)

        # Between an instance of CapitalCost and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) < other

        else:
            raise CapitalException(
                f"Must compare an instance of CapitalCost with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer, or a float."
            )

    def __le__(self, other):
        # Between an instance of CapitalCost with another instance of CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) <= np.sum(other.cost)

        # Between an instance of CapitalCost and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) <= other

        else:
            raise CapitalException(
                f"Must compare an instance of CapitalCost with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer, or a float."
            )

    def __gt__(self, other):
        # Between an instance of CapitalCost with another instance of CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) > np.sum(other.cost)

        # Between an instance of CapitalCost and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) > other

        else:
            raise CapitalException(
                f"Must compare an instance of CapitalCost with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer, or a float."
            )

    def __ge__(self, other):
        # Between an instance of CapitalCost with another instance of CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) >= np.sum(other.cost)

        # Between an instance of CapitalCost and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) >= other

        else:
            raise CapitalException(
                f"Must compare an instance of CapitalCost with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer, or a float."
            )

    def __add__(self, other):
        # Only allows addition between an instance of CapitalCost and another instance of CapitalCost
        if isinstance(other, CapitalCost):
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            cost_combined = np.concatenate((self.cost, other.cost))
            expense_year_combined = np.concatenate((self.expense_year, other.expense_year))
            cost_allocation_combined = self.cost_allocation + other.cost_allocation
            description_combined = self.description + other.description
            vat_portion_combined = np.concatenate((self.vat_portion, other.vat_portion))
            vat_discount_combined = np.concatenate((self.vat_discount, other.vat_discount))
            pis_year_combined = np.concatenate((self.pis_year, other.pis_year))
            salvage_value_combined = np.concatenate((self.salvage_value, other.salvage_value))
            useful_life_combined = np.concatenate((self.useful_life, other.useful_life))
            depreciation_factor_combined = np.concatenate(
                (self.depreciation_factor, other.depreciation_factor)
            )
            is_ic_applied_combined = self.is_ic_applied + other.is_ic_applied

            return CapitalCost(
                start_year=start_year_combined,
                end_year=end_year_combined,
                cost=cost_combined,
                expense_year=expense_year_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
                vat_portion=vat_portion_combined,
                vat_discount=vat_discount_combined,
                pis_year=pis_year_combined,
                salvage_value=salvage_value_combined,
                useful_life=useful_life_combined,
                depreciation_factor=depreciation_factor_combined,
                is_ic_applied=is_ic_applied_combined,
            )

        else:
            raise CapitalException(
                f"Must add an instance of CapitalCost with another instance of CapitalCost. "
                f"{other}({other.__class__.__qualname__}) is not an instance of CapitalCost."
            )

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        # Only allows subtraction between an instance of CapitalCost and another instance of CapitalCost
        if isinstance(other, CapitalCost):
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            cost_combined = np.concatenate((self.cost, -other.cost))
            expense_year_combined = np.concatenate((self.expense_year, other.expense_year))
            cost_allocation_combined = self.cost_allocation + other.cost_allocation
            description_combined = self.description + other.description
            vat_portion_combined = np.concatenate((self.vat_portion, other.vat_portion))
            vat_discount_combined = np.concatenate((self.vat_discount, other.vat_discount))
            pis_year_combined = np.concatenate((self.pis_year, other.pis_year))
            salvage_value_combined = np.concatenate((self.salvage_value, other.salvage_value))
            useful_life_combined = np.concatenate((self.useful_life, other.useful_life))
            depreciation_factor_combined = np.concatenate(
                (self.depreciation_factor, other.depreciation_factor)
            )
            is_ic_applied_combined = self.is_ic_applied + other.is_ic_applied

            return CapitalCost(
                start_year=start_year_combined,
                end_year=end_year_combined,
                cost=cost_combined,
                expense_year=expense_year_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
                vat_portion=vat_portion_combined,
                vat_discount=vat_discount_combined,
                pis_year=pis_year_combined,
                salvage_value=salvage_value_combined,
                useful_life=useful_life_combined,
                depreciation_factor=depreciation_factor_combined,
                is_ic_applied=is_ic_applied_combined,
            )

        else:
            raise CapitalException(
                f"Must subtract between an instance of CapitalCost "
                f"with another instance of CapitalCost. "
                f"{other}({other.__class__.__qualname__}) is not "
                f"an instance of CapitalCost."
            )

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        # Multiplication is allowed only with an integer/a float
        if isinstance(other, (int, float)):
            return CapitalCost(
                start_year=self.start_year,
                end_year=self.end_year,
                cost=self.cost * other,
                expense_year=self.expense_year,
                cost_allocation=self.cost_allocation,
                description=self.description,
                vat_portion=self.vat_portion,
                vat_discount=self.vat_discount,
                pis_year=self.pis_year,
                salvage_value=self.salvage_value,
                useful_life=self.useful_life,
                depreciation_factor=self.depreciation_factor,
                is_ic_applied=self.is_ic_applied,
            )

        else:
            raise CapitalException(
                f"Must multiply with an integer or a float; "
                f"{other}({other.__class__.__qualname__}) "
                f"is not an integer nor a float."
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        # Between an instance of CapitalCost with another instance of CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) / np.sum(other.cost)

        # Between an instance of CapitalCost and an integer/float
        elif isinstance(other, (int, float)):
            # Cannot divide with zero
            if other == 0:
                raise CapitalException(f"Cannot divide with zero")

            else:
                return CapitalCost(
                    start_year=self.start_year,
                    end_year=self.end_year,
                    cost=self.cost / other,
                    expense_year=self.expense_year,
                    cost_allocation=self.cost_allocation,
                    description=self.description,
                    vat_portion=self.vat_portion,
                    vat_discount=self.vat_discount,
                    pis_year=self.pis_year,
                    salvage_value=self.salvage_value,
                    useful_life=self.useful_life,
                    depreciation_factor=self.depreciation_factor,
                    is_ic_applied=self.is_ic_applied,
                )

        else:
            raise CapitalException(
                f"Must divide with an instance of CapitalCost/Intangible/OPEX/ASR/LBT, "
                f"integer or a float; {other}({other.__class__.__qualname__}) is not an instance "
                f"of CapitalCost/Intangible/OPEX/ASR/LBT nor an integer nor a float."
            )


@dataclass
class Intangible(GeneralCost):
    """
    Manages an intangible asset.

    Parameters
    ----------
    The attributes are inherited from class GeneralCost.
    Local attributes associated with class Intangible are:

    vat_portion: np.ndarray
        The portion of 'cost' that is subject to VAT.
        Must be an array of length equals to the length of 'cost' array.
    vat_discount: float
        The VAT discount to apply.
    """

    vat_portion: np.ndarray = field(default=None)
    vat_discount: float | np.ndarray = field(default=0.0)

    def __post_init__(self):
        # Prepare attribute project_duration and project_years
        if self.end_year >= self.start_year:
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise IntangibleException(
                f"start year {self.start_year} "
                f"is after the end year {self.end_year}"
            )

        # Prepare attribute VAT portion
        if self.vat_portion is None:
            self.vat_portion = np.zeros_like(self.cost)

        elif self.vat_portion is not None:
            if not isinstance(self.vat_portion, np.ndarray):
                raise IntangibleException(
                    f"Attribute vat_portion must be given as a numpy.ndarray; "
                    f"vat_portion (datatype: {self.vat_portion.__class__.__qualname__}) "
                    f"is not a numpy.ndarray."
                )

        self.vat_portion = self.vat_portion.astype(np.float64)

        # Prepare attribute VAT discount
        if not isinstance(self.vat_discount, (float, np.ndarray)):
            raise IntangibleException(
                f"Attribute VAT discount must be given as a float (for single value) "
                f"or a numpy array (for an array). The inserted VAT discount is given "
                f"as a {self.vat_discount.__class__.__qualname__}."
            )

        if isinstance(self.vat_discount, float):
            self.vat_discount = np.repeat(self.vat_discount, len(self.cost))

        self.vat_discount = self.vat_discount.astype(np.float64)

        # Prepare attribute description
        if self.description is None:
            self.description = [" " for _ in range(len(self.cost))]

        elif self.description is not None:
            if not isinstance(self.description, list):
                raise IntangibleException(
                    f"Attribute description must be given as a list; "
                    f"description (datatype: {self.description.__class__.__qualname__}) "
                    f"is not a list."
                )

        # Prepare attribute cost_allocation
        if self.cost_allocation is None:
            self.cost_allocation = [FluidType.OIL for _ in range(len(self.cost))]

        elif self.cost_allocation is not None:
            if not isinstance(self.cost_allocation, list):
                raise IntangibleException(
                    f"Attribute cost_allocation must be given as a list; "
                    f"cost_allocation (datatype: {self.cost_allocation.__class__.__qualname__}) "
                    f"is not a list."
                )

        # Check input data for unequal length
        arr_length = len(self.cost)

        if not all(
            len(arr) == arr_length
            for arr in [
                self.expense_year,
                self.cost_allocation,
                self.description,
                self.vat_portion,
                self.vat_discount,
            ]
        ):
            raise IntangibleException(
                f"Unequal length of array: "
                f"cost: {len(self.cost)}, "
                f"expense_year: {len(self.expense_year)}, "
                f"cost_allocation: {len(self.cost_allocation)}, "
                f"description: {len(self.description)}, "
                f"vat_portion: {len(self.vat_portion)}, "
                f"vat_discount: {len(self.vat_discount)}. "
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
        year_ref: np.ndarray = None,
        vat_rate: np.ndarray | float = 0.0,
        inflation_rate: np.ndarray | float = 0.0,
    ) -> np.ndarray:
        """
        Calculate expenditures per year.

        This method calculates the expenditures per year based on the expense year
        and cost data provided.

        Parameters
        ----------
        year_ref : np.ndarray
            The reference year for inflation calculation.
        vat_rate: np.ndarray | float
            The VAT rate to apply. Can be a single value or an array (default is 0.0).
        inflation_rate: np.ndarray | float
            The inflation rate to apply. Can be a single value or an array (default is 0.0).

        Returns
        -------
        np.ndarray
            An array depicting the expenditures each year, adjusted by tax
            and inflation schemes (if any).

        Notes
        -----
        This method calculates expenditures while considering tax and inflation schemes.
        The core calculations are as follows:
        (1) Apply adjustment to cost due to tax and inflation (if any), by calling
            'apply_cost_adjustment()' function,
        (2) Function np.bincount() is used to align the 'cost_adjusted' elements
            according to its corresponding expense year,
        (3) If len(expenses) < project_duration, then add the remaining elements
            with zeros.
        """
        cost_adjusted = apply_cost_adjustment(
            start_year=self.start_year,
            end_year=self.end_year,
            cost=self.cost,
            expense_year=self.expense_year,
            project_years=self.project_years,
            year_ref=year_ref,
            tax_portion=self.vat_portion,
            tax_rate=vat_rate,
            tax_discount=self.vat_discount,
            inflation_rate=inflation_rate,
        )

        expenses = np.bincount(
            self.expense_year - self.start_year, weights=cost_adjusted
        )
        zeros = np.zeros(self.project_duration - len(expenses))

        return np.concatenate((expenses, zeros))

    def __eq__(self, other):
        # Between two instances of Intangible
        if isinstance(other, Intangible):
            return all(
                (
                    np.allclose(self.cost, other.cost),
                    np.allclose(self.expense_year, other.expense_year),
                    np.allclose(self.vat_portion, other.vat_portion),
                    np.allclose(self.vat_discount, other.vat_discount),
                    self.cost_allocation == other.cost_allocation,
                )
            )

        # Between an instance of Intangible and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) == other

        else:
            return False

    def __lt__(self, other):
        # Between an instance of Intangible with another instance of CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) < np.sum(other.cost)

        # Between an instance of Intangible and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) < other

        else:
            raise IntangibleException(
                f"Must compare an instance of Intangible with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer, or a float."
            )

    def __le__(self, other):
        # Between an instance of Intangible with another instance of CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) <= np.sum(other.cost)

        # Between an instance of Intangible and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) <= other

        else:
            raise IntangibleException(
                f"Must compare an instance of Intangible with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer, or a float."
            )

    def __gt__(self, other):
        # Between an instance of Intangible with another instance of CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) > np.sum(other.cost)

        # Between an instance of Intangible and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) > other

        else:
            raise IntangibleException(
                f"Must compare an instance of Intangible with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer, or a float."
            )

    def __ge__(self, other):
        # Between an instance of Intangible with another instance of CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) >= np.sum(other.cost)

        # Between an instance of Intangible and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) >= other

        else:
            raise IntangibleException(
                f"Must compare an instance of Intangible with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer, or a float."
            )

    def __add__(self, other):
        # Only allows addition between an instance of Intangible and another instance of Intangible
        if isinstance(other, Intangible):
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            cost_combined = np.concatenate((self.cost, other.cost))
            expense_year_combined = np.concatenate((self.expense_year, other.expense_year))
            cost_allocation_combined = self.cost_allocation + other.cost_allocation
            description_combined = self.description + other.description
            vat_portion_combined = np.concatenate((self.vat_portion, other.vat_portion))
            vat_discount_combined = np.concatenate((self.vat_discount, other.vat_discount))

            return Intangible(
                start_year=start_year_combined,
                end_year=end_year_combined,
                cost=cost_combined,
                expense_year=expense_year_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
                vat_portion=vat_portion_combined,
                vat_discount=vat_discount_combined,
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
        # Only allows subtraction between an instance of Intangible and another instance of Intangible
        if isinstance(other, Intangible):
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            cost_combined = np.concatenate((self.cost, -other.cost))
            expense_year_combined = np.concatenate((self.expense_year, other.expense_year))
            cost_allocation_combined = self.cost_allocation + other.cost_allocation
            description_combined = self.description + other.description
            vat_portion_combined = np.concatenate((self.vat_portion, other.vat_portion))
            vat_discount_combined = np.concatenate((self.vat_discount, other.vat_discount))

            return Intangible(
                start_year=start_year_combined,
                end_year=end_year_combined,
                cost=cost_combined,
                expense_year=expense_year_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
                vat_portion=vat_portion_combined,
                vat_discount=vat_discount_combined,
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
                expense_year=self.expense_year,
                cost_allocation=self.cost_allocation,
                description=self.description,
                vat_portion=self.vat_portion,
                vat_discount=self.vat_discount,
            )

        else:
            raise IntangibleException(
                f"Must multiply with an integer or a float; "
                f"{other}({other.__class__.__qualname__}) is not an integer nor a float."
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        # Between an instance of Intangible with another instance of CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
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
                    expense_year=self.expense_year,
                    cost_allocation=self.cost_allocation,
                    description=self.description,
                    vat_portion=self.vat_portion,
                    vat_discount=self.vat_discount,
                )

        else:
            raise IntangibleException(
                f"Must divide with an instance of CapitalCost/Intangible/OPEX/ASR/LBT, "
                f"integer or a float; {other}({other.__class__.__qualname__}) is not an instance "
                f"of CapitalCost/Intangible/OPEX/ASR/LBT nor an integer nor a float."
            )


@dataclass
class LBT(GeneralCost):
    """
    Manages an LBT asset.

    Parameters
    ----------
    The attributes are inherited from class GeneralCost.
    Local attributes associated with class LBT are:

    lbt_portion: np.ndarray
        The portion of 'cost' that is subject to LBT.
        Must be an array of length equals to the length of 'cost' array.
    lbt_discount: float
        The LBT discount to apply.
    """

    lbt_portion: np.ndarray = field(default=None)
    lbt_discount: float | np.ndarray = field(default=0.0)

    def __post_init__(self):
        # Prepare attribute project_duration and project_years
        if self.end_year >= self.start_year:
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise LBTException(
                f"start year {self.start_year} "
                f"is after the end year {self.end_year}"
            )

        # Prepare attribute LBT portion
        if self.lbt_portion is None:
            self.lbt_portion = np.zeros_like(self.cost)

        elif self.lbt_portion is not None:
            if not isinstance(self.lbt_portion, np.ndarray):
                raise LBTException(
                    f"Attribute lbt_portion must be given as a numpy.ndarray; "
                    f"lbt_portion (datatype: {self.lbt_portion.__class__.__qualname__}) "
                    f"is not a numpy.ndarray."
                )

        self.lbt_portion = self.lbt_portion.astype(np.float64)

        # Prepare attribute LBT discount
        if not isinstance(self.lbt_discount, (float, np.ndarray)):
            raise LBTException(
                f"Attribute LBT discount must be given as a float (for a single value) "
                f"or a numpy array (for an array). The inserted LBT discount is given "
                f"as a {self.lbt_discount.__class__.__qualname__}."
            )

        if isinstance(self.lbt_discount, float):
            self.lbt_discount = np.repeat(self.lbt_discount, len(self.cost))

        self.lbt_discount = self.lbt_discount.astype(np.float64)

        # Prepare attribute description
        if self.description is None:
            self.description = [" " for _ in range(len(self.cost))]

        elif self.description is not None:
            if not isinstance(self.description, list):
                raise LBTException(
                    f"Attribute description must be given as a list; "
                    f"desription (datatype: {self.description.__class__.__qualname__}) "
                    f"is not a list."
                )

        # Prepare attribute cost_allocation
        if self.cost_allocation is None:
            self.cost_allocation = [FluidType.OIL for _ in range(len(self.cost))]

        elif self.cost_allocation is not None:
            if not isinstance(self.cost_allocation, list):
                raise LBTException(
                    f"Attribute cost_allocation must be given as a list; "
                    f"cost_allocation (datatype: {self.cost_allocation.__class__.__qualname__}) "
                    f"is not a list."
                )

        # Check input data for unequal length
        arr_length = len(self.cost)

        if not all(
            len(arr) == arr_length
            for arr in [
                self.expense_year,
                self.cost_allocation,
                self.description,
                self.lbt_portion,
                self.lbt_discount,
            ]
        ):
            raise LBTException(
                f"Unequal length of array: "
                f"cost: {len(self.cost)}, "
                f"expense_year: {len(self.expense_year)}, "
                f"cost_allocation: {len(self.cost_allocation)}, "
                f"description: {len(self.description)}, "
                f"lbt_portion: {len(self.lbt_portion)}, "
                f"lbt_discount: {len(self.lbt_discount)}."
            )

        # Raise an error message: expense year is after the end year of the project
        if np.max(self.expense_year) > self.end_year:
            raise LBTException(
                f"Expense year ({np.max(self.expense_year)}) "
                f"is after the end year of the project ({self.end_year})"
            )

        # Raise an error message: expense year is before the start year of the project
        if np.min(self.expense_year) < self.start_year:
            raise LBTException(
                f"Expense year ({np.min(self.expense_year)}) "
                f"is before the start year of the project ({self.start_year})"
            )

    def expenditures(
        self,
        year_ref: np.ndarray = None,
        lbt_rate: np.ndarray | float = 0.0,
        inflation_rate: np.ndarray | float = 0.0,
    ) -> np.ndarray:
        """
        Calculate expenditures per year.

        This method calculates the expenditures per year based on the expense year
        and cost data provided.

        Parameters
        ----------
        year_ref : np.ndarray
            The reference year for inflation calculation.
        lbt_rate: np.ndarray | float
            The LBT rate to apply. Can be a single value or an array (default is 0.0).
        inflation_rate: np.ndarray | float
            The inflation rate to apply. Can be a single value or an array (default is 0.0).

        Returns
        -------
        np.ndarray
            An array depicting the expenditures each year, adjusted by tax
            and inflation schemes (if any).

        Notes
        -----
        This method calculates expenditures while considering tax and inflation schemes.
        The core calculations are as follows:
        (1) Apply adjustment to cost due to tax and inflation (if any), by calling
            'apply_cost_adjustment()' function,
        (2) Function np.bincount() is used to align the 'cost_adjusted' elements
            according to its corresponding expense year,
        (3) If len(expenses) < project_duration, then add the remaining elements
            with zeros.
        """
        cost_adjusted = apply_cost_adjustment(
            start_year=self.start_year,
            end_year=self.end_year,
            cost=self.cost,
            expense_year=self.expense_year,
            project_years=self.project_years,
            year_ref=year_ref,
            tax_portion=self.lbt_portion,
            tax_rate=lbt_rate,
            tax_discount=self.lbt_discount,
            inflation_rate=inflation_rate,
        )

        expenses = np.bincount(
            self.expense_year - self.start_year, weights=cost_adjusted
        )
        zeros = np.zeros(self.project_duration - len(expenses))

        return np.concatenate((expenses, zeros))

    def __eq__(self, other):
        # Between two instances of LBT
        if isinstance(other, LBT):
            return all(
                (
                    np.allclose(self.cost, other.cost),
                    np.allclose(self.expense_year, other.expense_year),
                    np.allclose(self.lbt_portion, other.lbt_portion),
                    np.allclose(self.lbt_discount, other.lbt_discount),
                    self.cost_allocation == other.cost_allocation,
                )
            )

        # Between an instance of LBT and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) == other

        else:
            return False

    def __lt__(self, other):
        # Between an instance of LBT with another instance of LBT/CapitalCost/Intangible/OPEX/ASR
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) < np.sum(other.cost)

        # Between an instance of LBT and an integer/a float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) < other

        else:
            raise LBTException(
                f"Must compare an instance of LBT with another instance of "
                f"LBT/CapitalCost/Intangible/OPEX/ASR, an integer, or a float."
            )

    def __le__(self, other):
        # Between an instance of LBT with another instance of LBT/CapitalCost/Intangible/OPEX/ASR
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) <= np.sum(other.cost)

        # Between an instance of LBT and an integer/a float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) <= other

        else:
            raise LBTException(
                f"Must compare an instance of LBT with another instance of "
                f"LBT/CapitalCost/Intangible/OPEX/ASR, an integer, or a float."
            )

    def __gt__(self, other):
        # Between an instance of LBT with another instance of LBT/CapitalCost/Intangible/OPEX/ASR
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) > np.sum(other.cost)

        # Between an instance of LBT and an integer/a float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) > other

        else:
            raise LBTException(
                f"Must compare an instance of LBT with another instance of "
                f"LBT/CapitalCost/Intangible/OPEX/ASR, an integer, or a float."
            )

    def __ge__(self, other):
        # Between an instance of LBT with another instance of LBT/CapitalCost/Intangible/OPEX/ASR
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) >= np.sum(other.cost)

        # Between an instance of LBT and an integer/a float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) >= other

        else:
            raise LBTException(
                f"Must compare an instance of LBT with another instance of "
                f"LBT/CapitalCost/Intangible/OPEX/ASR, an integer, or a float."
            )

    def __add__(self, other):
        # Only allows addition between an instance of LBT and another instance of LBT
        if isinstance(other, LBT):
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            cost_combined = np.concatenate((self.cost, other.cost))
            expense_year_combined = np.concatenate((self.expense_year, other.expense_year))
            cost_allocation_combined = self.cost_allocation + other.cost_allocation
            description_combined = self.description + other.description
            lbt_portion_combined = np.concatenate((self.lbt_portion, other.lbt_portion))
            lbt_discount_combined = np.concatenate((self.lbt_discount, other.lbt_discount))

            return LBT(
                start_year=start_year_combined,
                end_year=end_year_combined,
                cost=cost_combined,
                expense_year=expense_year_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
                lbt_portion=lbt_portion_combined,
                lbt_discount=lbt_discount_combined,
            )

        else:
            raise LBTException(
                f"Must add between an instance of LBT "
                f"with another instance of LBT. "
                f"{other}({other.__class__.__qualname__}) is not "
                f"an instance of LBT."
            )

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        # Only allows subtraction between an instance of LBT and another instance of LBT
        if isinstance(other, LBT):
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            cost_combined = np.concatenate((self.cost, -other.cost))
            expense_year_combined = np.concatenate((self.expense_year, other.expense_year))
            cost_allocation_combined = self.cost_allocation + other.cost_allocation
            description_combined = self.description + other.description
            lbt_portion_combined = np.concatenate((self.lbt_portion, other.lbt_portion))
            lbt_discount_combined = np.concatenate((self.lbt_discount, other.lbt_discount))

            return LBT(
                start_year=start_year_combined,
                end_year=end_year_combined,
                cost=cost_combined,
                expense_year=expense_year_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
                lbt_portion=lbt_portion_combined,
                lbt_discount=lbt_discount_combined,
            )

        else:
            raise LBTException(
                f"Must subtract between an instance of LBT "
                f"with another instance of LBT. "
                f"{other}({other.__class__.__qualname__}) is not "
                f"an instance of LBT."
            )

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        # Multiplication is allowed only with an integer/a float
        if isinstance(other, (int, float)):
            return LBT(
                start_year=self.start_year,
                end_year=self.end_year,
                cost=self.cost * other,
                expense_year=self.expense_year,
                cost_allocation=self.cost_allocation,
                description=self.description,
                lbt_portion=self.lbt_portion,
                lbt_discount=self.lbt_discount,
            )

        else:
            raise LBTException(
                f"Must multiply with an integer of a float; "
                f"{other}({other.__class__.__qualname__}) is not an integer nor a float."
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        # Between an instance of LBT with another instance of LBT/CapitalCost/Intangible/OPEX/ASR
        if isinstance(other, (LBT, CapitalCost, Intangible, OPEX, ASR)):
            return np.sum(self.cost) / np.sum(other.cost)

        # Between an instance of LBT and an integer/float
        elif isinstance(other, (int, float)):
            # Cannot divide with zero
            if other == 0:
                raise LBTException(f"Cannot divide with zero")

            else:
                return LBT(
                    start_year=self.start_year,
                    end_year=self.end_year,
                    cost=self.cost / other,
                    expense_year=self.expense_year,
                    cost_allocation=self.cost_allocation,
                    description=self.description,
                    lbt_portion=self.lbt_portion,
                    lbt_discount=self.lbt_discount,
                )

        else:
            raise LBTException(
                f"Must divide with an instance of LBT/CapitalCost/Intangible/OPEX/ASR, "
                f"integer or a float; "
                f"{other}({other.__class__.__qualname__}) is not an instance "
                f"of LBT/CapitalCost/Intangible/OPEX/ASR nor an integer nor a float."
            )


@dataclass
class OPEX(GeneralCost):
    """
    Manages an OPEX asset.

    Parameters
    ----------
    The attributes are inherited from class GeneralCost. Local attributes associated
    with class OPEX are:

    fixed_cost : np.ndarray
        An array representing the fixed cost of an OPEX asset.
    prod_rate: np.ndarray
        The production rate of a particular fluid type.
    cost_per_volume: np.ndarray
        Cost associated with production of a particular fluid type.
    vat_portion: np.ndarray
        The portion of 'cost' that is subject to VAT.
        Must be an array of length equals to the length of 'cost' array.
    vat_discount: float
        The VAT discount to apply.

    Notes
    -----
    The unit used in the fixed_cost should be in M unit of United States Dollar (USD),
    where the M is stands for 1000. Thus, the unit cost should be: M-USD.
    """

    fixed_cost: np.ndarray = field(default=None)
    prod_rate: np.ndarray = field(default=None)
    cost_per_volume: np.ndarray = field(default=None)
    vat_portion: np.ndarray = field(default=None)
    vat_discount: float | np.ndarray = field(default=0.0)

    # Attribute to be defined later on
    variable_cost: np.ndarray = field(default=None, init=False)
    cost: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # Prepare attribute project_duration and project_years
        if self.end_year >= self.start_year:
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise OPEXException(
                f"start year {self.start_year} "
                f"is after the end year {self.end_year}"
            )

        # Prepare attribute VAT portion
        if self.vat_portion is None:
            self.vat_portion = np.zeros_like(self.fixed_cost)

        elif self.vat_portion is not None:
            if not isinstance(self.vat_portion, np.ndarray):
                raise OPEXException(
                    f"Attribute vat_portion must be given as a numpy.ndarray; "
                    f"vat_portion (datatype: {self.vat_portion.__class__.__qualname__}) "
                    f"is not a numpy.ndarray."
                )

        self.vat_portion = self.vat_portion.astype(np.float64)

        # Prepare attribute VAT discount
        if not isinstance(self.vat_discount, (float, np.ndarray)):
            raise OPEXException(
                f"Attribute VAT discount must be given as a float (for single value) "
                f"or a numpy.ndarray (for an array). The inserted VAT discount is given "
                f"as a {self.vat_discount.__class__.__qualname__}."
            )

        if isinstance(self.vat_discount, float):
            self.vat_discount = np.repeat(self.vat_discount, len(self.fixed_cost))

        self.vat_discount = self.vat_discount.astype(np.float64)

        # Prepare attribute description
        if self.description is None:
            self.description = [" " for _ in range(len(self.fixed_cost))]

        elif self.description is not None:
            if not isinstance(self.description, list):
                raise OPEXException(
                    f"Attribute description must be given as a list; "
                    f"description (datatype: {self.description.__class__.__qualname__}) "
                    f"is not a list."
                )

        # Prepare attribute cost_allocation
        if self.cost_allocation is None:
            self.cost_allocation = [FluidType.OIL for _ in range(len(self.fixed_cost))]

        elif self.cost_allocation is not None:
            if not isinstance(self.cost_allocation, list):
                raise OPEXException(
                    f"Attribute cost_allocation must be given as a list; "
                    f"cost_allocation (datatype: {self.cost_allocation.__class__.__qualname__}) "
                    f"is not a list."
                )

        # User provides both prod_rate and cost_per_volume data
        if self.prod_rate is not None and self.cost_per_volume is not None:

            # Check input data for unequal length
            arr_length = len(self.fixed_cost)

            if not all(
                len(i) == arr_length
                for i in [
                    self.expense_year,
                    self.cost_allocation,
                    self.description,
                    self.vat_portion,
                    self.vat_discount,
                    self.prod_rate,
                    self.cost_per_volume,
                ]
            ):
                raise OPEXException(
                    f"Unequal length of array: "
                    f"fixed_cost: {len(self.fixed_cost)}, "
                    f"expense_year: {len(self.expense_year)}, "
                    f"cost_allocation: {len(self.cost_allocation)}, "
                    f"description: {len(self.description)}, "
                    f"vat_portion: {len(self.vat_portion)}, "
                    f"vat_discount: {len(self.vat_discount)}, "
                    f"prod_rate: {len(self.prod_rate)}, "
                    f"cost_per_volume: {len(self.cost_per_volume)}."
                )

            # Specify attribute variable_cost
            self.variable_cost = self.prod_rate * self.cost_per_volume

        # User only provides prod_rate data
        if self.prod_rate is not None and self.cost_per_volume is None:
            raise OPEXException(f"cost_per_volume data is missing")

        # User only provides cost_per_volume data
        if self.prod_rate is None and self.cost_per_volume is not None:
            raise OPEXException(f"prod_rate data is missing")

        # User does not provide both prod_rate and cost_per_volume data
        if self.prod_rate is None and self.cost_per_volume is None:
            self.prod_rate = np.zeros_like(self.fixed_cost)
            self.cost_per_volume = np.zeros_like(self.fixed_cost)
            self.variable_cost = np.zeros_like(self.fixed_cost)

        self.prod_rate = self.prod_rate.astype(np.float64)
        self.cost_per_volume = self.cost_per_volume.astype(np.float64)
        self.variable_cost = self.variable_cost.astype(np.float64)

        # Define cost
        self.cost = self.fixed_cost + self.variable_cost

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

    def expenditures(
        self,
        year_ref: np.ndarray = None,
        vat_rate: np.ndarray | float = 0.0,
        inflation_rate: np.ndarray | float = 0.0,
    ) -> np.ndarray:
        """
        Calculate expenditures per year.

        This method calculates the expenditures per year based on the expense year
        and cost data provided.

        Parameters
        ----------
        year_ref : np.ndarray
            The reference year for inflation calculation.
        vat_rate: np.ndarray | float
            The VAT rate to apply. Can be a single value or an array (default is 0.0).
        inflation_rate: np.ndarray | float
            The inflation rate to apply. Can be a single value or an array (default is 0.0).

        Returns
        -------
        np.ndarray
            An array depicting the expenditures each year, adjusted by tax
            and inflation schemes (if any).

        Notes
        -----
        This method calculates expenditures while considering tax and inflation schemes.
        The core calculations are as follows:
        (1) Apply adjustment to cost due to tax and inflation (if any), by calling
            'apply_cost_adjustment()' function,
        (2) Function np.bincount() is used to align the 'cost_adjusted' elements
            according to its corresponding expense year,
        (3) If len(expenses) < project_duration, then add the remaining elements
            with zeros.
        """
        cost_adjusted = apply_cost_adjustment(
            start_year=self.start_year,
            end_year=self.end_year,
            cost=self.cost,
            expense_year=self.expense_year,
            project_years=self.project_years,
            year_ref=year_ref,
            tax_portion=self.vat_portion,
            tax_rate=vat_rate,
            tax_discount=self.vat_discount,
            inflation_rate=inflation_rate,
        )

        expenses = np.bincount(
            self.expense_year - self.start_year, weights=cost_adjusted
        )
        zeros = np.zeros(self.project_duration - len(expenses))

        return np.concatenate((expenses, zeros))

    def __eq__(self, other):
        # Between two instances of OPEX
        if isinstance(other, OPEX):
            return all(
                (
                    np.allclose(self.fixed_cost, other.fixed_cost),
                    np.allclose(self.variable_cost, other.variable_cost),
                    np.allclose(self.expense_year, other.expense_year),
                    np.allclose(self.vat_portion, other.vat_portion),
                    np.allclose(self.vat_discount, other.vat_discount),
                    self.cost_allocation == other.cost_allocation,
                )
            )

        # Between an instance of OPEX and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.fixed_cost, self.variable_cost) == other

        else:
            return False

    def __lt__(self, other):
        # Between an instance of OPEX with another instance of CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) < np.sum(other.cost)

        # Between an instance of OPEX and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) < other

        else:
            raise OPEXException(
                f"Must compare an instance of OPEX with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer, or a float."
            )

    def __le__(self, other):
        # Between an instance of OPEX with another instance of CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) <= np.sum(other.cost)

        # Between an instance of OPEX and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) <= other

        else:
            raise OPEXException(
                f"Must compare an instance of OPEX with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer, or a float."
            )

    def __gt__(self, other):
        # Between an instance of OPEX with another instance of CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) > np.sum(other.cost)

        # Between an instance of OPEX and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) > other

        else:
            raise OPEXException(
                f"Must compare an instance of OPEX with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer, or a float."
            )

    def __ge__(self, other):
        # Between an instance of OPEX with another instance of CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) >= np.sum(other.cost)

        # Between an instance of OPEX and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) >= other

        else:
            raise OPEXException(
                f"Must compare an instance of OPEX with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer, or a float."
            )

    def __add__(self, other):
        # Only allows addition between an instance of OPEX and another instance of OPEX
        if isinstance(other, OPEX):
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            expense_year_combined = np.concatenate((self.expense_year, other.expense_year))
            cost_allocation_combined = self.cost_allocation + other.cost_allocation
            description_combined = self.description + other.description
            vat_portion_combined = np.concatenate((self.vat_portion, other.vat_portion))
            vat_discount_combined = np.concatenate((self.vat_discount, other.vat_discount))
            fixed_cost_combined = np.concatenate((self.fixed_cost, other.fixed_cost))
            prod_rate_combined = np.concatenate((self.prod_rate, other.prod_rate))
            cost_per_volume_combined = np.concatenate((self.cost_per_volume, other.cost_per_volume))

            return OPEX(
                start_year=start_year_combined,
                end_year=end_year_combined,
                expense_year=expense_year_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
                vat_portion=vat_portion_combined,
                vat_discount=vat_discount_combined,
                fixed_cost=fixed_cost_combined,
                prod_rate=prod_rate_combined,
                cost_per_volume=cost_per_volume_combined,
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
        # Only allows subtraction between an instance of OPEX and another instance of OPEX
        if isinstance(other, OPEX):
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            expense_year_combined = np.concatenate((self.expense_year, other.expense_year))
            cost_allocation_combined = self.cost_allocation + other.cost_allocation
            description_combined = self.description + other.description
            vat_portion_combined = np.concatenate((self.vat_portion, other.vat_portion))
            vat_discount_combined = np.concatenate((self.vat_discount, other.vat_discount))
            fixed_cost_combined = np.concatenate((self.fixed_cost, -other.fixed_cost))
            prod_rate_combined = np.concatenate((self.prod_rate, -other.prod_rate))
            cost_per_volume_combined = np.concatenate((self.cost_per_volume, other.cost_per_volume))

            return OPEX(
                start_year=start_year_combined,
                end_year=end_year_combined,
                expense_year=expense_year_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
                vat_portion=vat_portion_combined,
                vat_discount=vat_discount_combined,
                fixed_cost=fixed_cost_combined,
                prod_rate=prod_rate_combined,
                cost_per_volume=cost_per_volume_combined,
            )

        else:
            raise OPEXException(
                f"Must subtract between an instance of OPEX "
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
                expense_year=self.expense_year,
                cost_allocation=self.cost_allocation,
                description=self.description,
                vat_portion=self.vat_portion,
                vat_discount=self.vat_discount,
                fixed_cost=self.fixed_cost * other,
                prod_rate=self.prod_rate * other,
                cost_per_volume=self.cost_per_volume,
            )

        else:
            raise OPEXException(
                f"Must multiply with an integer or a float; "
                f"{other}({other.__class__.__qualname__}) is not an integer nor a float."
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        # Between an instance of OPEX with another instance of CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
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
                    expense_year=self.expense_year,
                    cost_allocation=self.cost_allocation,
                    description=self.description,
                    vat_portion=self.vat_portion,
                    vat_discount=self.vat_discount,
                    fixed_cost=self.fixed_cost / other,
                    prod_rate=self.prod_rate / other,
                    cost_per_volume=self.cost_per_volume,
                )

        else:
            raise OPEXException(
                f"Must divide with an instance of CapitalCost/Intangible/OPEX/ASR/lBT, "
                f"integer or a float; {other}({other.__class__.__qualname__}) is not an instance "
                f"of CapitalCost/Intangible/OPEX/ASR/LBT nor an integer nor a float."
            )


@dataclass
class ASR(GeneralCost):
    """
    Manages an ASR asset.

    Parameters
    ----------
    The attributes are inherited from class GeneralCost.
    Local attributes associated with class ASR are:

    vat_portion: np.ndarray
        The portion of 'cost' that is subject to VAT.
        Must be an array of length equals to the length of 'cost' array.
    vat_discount: float
        The VAT discount to apply.
    """

    vat_portion: np.ndarray = field(default=None)
    vat_discount: float | np.ndarray = field(default=0.0)

    def __post_init__(self):
        # Prepare attributes project_duration and project_years
        if self.end_year >= self.start_year:
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise ASRException(
                f"start year {self.start_year} "
                f"is after the end year {self.end_year}"
            )

        # Prepare attribute VAT portion
        if self.vat_portion is None:
            self.vat_portion = np.zeros_like(self.cost)

        elif self.vat_portion is not None:
            if not isinstance(self.vat_portion, np.ndarray):
                raise ASRException(
                    f"Attribute vat_portion must be given as a numpy.ndarray; "
                    f"vat_portion (datatype: {self.vat_portion.__class__.__qualname__}) "
                    f"is not a numpy.ndarray."
                )

        self.vat_portion = self.vat_portion.astype(np.float64)

        # Prepare attribute VAT discount
        if not isinstance(self.vat_discount, (float, np.ndarray)):
            raise ASRException(
                f"Attribute VAT discount must be given as a float (for single value) "
                f"or a numpy array (for an array). The inserted VAT discount is given "
                f"as a {self.vat_discount.__class__.__qualname__}."
            )

        if isinstance(self.vat_discount, float):
            self.vat_discount = np.repeat(self.vat_discount, len(self.cost))

        self.vat_discount = self.vat_discount.astype(np.float64)

        # Prepare attribute description
        if self.description is None:
            self.description = [" " for _ in range(len(self.cost))]

        elif self.description is not None:
            if not isinstance(self.description, list):
                raise ASRException(
                    f"Attribute description must be given as a list; "
                    f"description (datatype: {self.description.__class__.__qualname__}) "
                    f"is not a list."
                )

        # Prepare attribute cost_allocation
        if self.cost_allocation is None:
            self.cost_allocation = [FluidType.OIL for _ in range(len(self.cost))]

        elif self.cost_allocation is not None:
            if not isinstance(self.cost_allocation, list):
                raise ASRException(
                    f"Attribute cost_allocation must be given as a list; "
                    f"cost_allocation (datatype: {self.cost_allocation.__class__.__qualname__}) "
                    f"is not a list."
                )

        # Check input data for unequal length
        arr_length = len(self.cost)

        if not all(
            len(arr) == arr_length
            for arr in [
                self.expense_year,
                self.cost_allocation,
                self.description,
                self.vat_portion,
                self.vat_discount,
            ]
        ):
            raise ASRException(
                f"Unequal length of array: "
                f"cost: {len(self.cost)}, "
                f"expense_year: {len(self.expense_year)}, "
                f"cost_allocation: {len(self.cost_allocation)}, "
                f"description: {len(self.description)}, "
                f"vat_portion: {len(self.vat_portion)}, "
                f"vat_discount: {len(self.vat_discount)}."
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

    def expenditures(
        self,
        year_ref: np.ndarray = None,
        vat_rate: np.ndarray | float = 0.0,
        inflation_rate: np.ndarray | float = 0.0,
    ) -> np.ndarray:
        """
        Calculate expenditures per year.

        This method calculates the expenditures per year based on the expense year
        and cost data provided.

        Parameters
        ----------
        year_ref : np.ndarray
            The reference year for inflation calculation.
        vat_rate: np.ndarray | float
            The VAT rate to apply. Can be a single value or an array (default is 0.0).
        inflation_rate: np.ndarray | float
            The inflation rate to apply. Can be a single value or an array (default is 0.0).

        Returns
        -------
        np.ndarray
            An array depicting the expenditures each year, adjusted by tax
            and inflation schemes (if any).

        Notes
        -----
        This method calculates expenditures while considering tax and inflation schemes.
        The core calculations are as follows:
        (1) Apply adjustment to cost due to tax and inflation (if any), by calling
            'apply_cost_adjustment()' function,
        (2) Function np.bincount() is used to align the 'cost_adjusted' elements
            according to its corresponding expense year,
        (3) If len(expenses) < project_duration, then add the remaining elements
            with zeros.
        """
        cost_adjusted = apply_cost_adjustment(
            start_year=self.start_year,
            end_year=self.end_year,
            cost=self.cost,
            expense_year=self.expense_year,
            project_years=self.project_years,
            year_ref=year_ref,
            tax_portion=self.vat_portion,
            tax_rate=vat_rate,
            tax_discount=self.vat_discount,
            inflation_rate=inflation_rate,
        )

        expenses = np.bincount(
            self.expense_year - self.start_year, weights=cost_adjusted
        )
        zeros = np.zeros(self.project_duration - len(expenses))

        return np.concatenate((expenses, zeros))

    def future_cost(
        self,
        year_ref: np.ndarray = None,
        vat_rate: np.ndarray | float = 0.0,
        inflation_rate: np.ndarray | float = 0.0,
        future_rate: float = 0.02,
    ) -> np.ndarray:
        """
        Calculate the future cost of an asset.

        Parameters
        ----------
        year_ref : np.ndarray
            The reference year for inflation calculation.
        vat_rate: np.ndarray | float
            The VAT rate to apply. Can be a single value or an array (default is 0.0).
        inflation_rate : np.ndarray or float or int, optional
            The inflation rate to apply. Can be a single value or an array (default is 0).
        future_rate : float, optional
            The future rate used in cost calculation (default is 0.02).

        Returns
        -------
        np.ndarray
            An array containing the future cost of the asset.

        Notes
        -----
        This function calculates the future cost of an asset, taking into
        account tax and inflation schemes.
        """
        # Only allows float datatype as an input for future_rate
        if not isinstance(future_rate, float):
            raise ASRException(
                f"Argument future_rate must be given as a float, not a "
                f"{future_rate.__class__.__qualname__}."
            )

        # Cost adjustment due to VAT and inflation schemes
        cost_adjusted = apply_cost_adjustment(
            start_year=self.start_year,
            end_year=self.end_year,
            cost=self.cost,
            expense_year=self.expense_year,
            project_years=self.project_years,
            year_ref=year_ref,
            tax_portion=self.vat_portion,
            tax_rate=vat_rate,
            tax_discount=self.vat_discount,
            inflation_rate=inflation_rate,
        )

        # Cost adjustment due to future_rate
        return cost_adjusted * np.power(
            (1.0 + future_rate), self.end_year - self.expense_year + 1
        )

    def proportion(
        self,
        year_ref: np.ndarray = None,
        vat_rate: np.ndarray | float = 0.0,
        inflation_rate: np.ndarray | float = 0.0,
        future_rate: float = 0.02,
    ) -> np.ndarray:
        """
        Allocate ASR expenditures per year.

        Parameters
        ----------
        year_ref : np.ndarray
            The reference year for inflation calculation.
        vat_rate: np.ndarray | float
            The VAT rate to apply. Can be a single value or an array (default is 0.0).
        inflation_rate : np.ndarray or float or int, optional
            The inflation rate to apply. Can be a single value or an array (default is 0).
        future_rate : float, optional
            The future rate used in cost calculation (default is 0.02).

        Returns
        -------
        np.ndarray
            An array depicting allocation of ASR expenditures each year, adjusted by
            tax and inflation (if any).
        """
        # Only allows float datatype as an input for future_rate
        if not isinstance(future_rate, float):
            raise ASRException(
                f"Argument future_rate must be given as a float, not a "
                f"{future_rate.__class__.__qualname__}"
            )

        # Distance of expense year from the end year of the project
        cost_duration = self.end_year - self.expense_year + 1

        # Configure future cost
        cost_future = self.future_cost(
            year_ref=year_ref,
            vat_rate=vat_rate,
            inflation_rate=inflation_rate,
            future_rate=future_rate,
        )

        # Allocation of future cost: cost distribution per year
        cost_alloc = cost_future / cost_duration

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

    def __eq__(self, other):
        # Between two instances of ASR
        if isinstance(other, ASR):
            return all(
                (
                    np.allclose(self.cost, other.cost),
                    np.allclose(self.expense_year, other.expense_year),
                    np.allclose(self.vat_portion, other.vat_portion),
                    np.allclose(self.vat_discount, other.vat_discount),
                    self.cost_allocation == other.cost_allocation,
                )
            )

        # Between an instance of ASR and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) == other

        else:
            return False

    def __lt__(self, other):
        # Between an instance of ASR with another instance of CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) < np.sum(other.cost)

        # Between an instance of ASR and an integer/float
        elif isinstance(other, (float, int)):
            return np.sum(self.cost) < other

        else:
            raise ASRException(
                f"Must compare an instance of ASR with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer, or a float."
            )

    def __le__(self, other):
        # Between an instance of ASR with another instance of CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) <= np.sum(other.cost)

        # Between an instance of ASR and an integer/float
        elif isinstance(other, (float, int)):
            return np.sum(self.cost) <= other

        else:
            raise ASRException(
                f"Must compare an instance of ASR with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer, or a float."
            )

    def __gt__(self, other):
        # Between an instance of ASR with another instance of CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) > np.sum(other.cost)

        # Between an instance of ASR and an integer/float
        elif isinstance(other, (float, int)):
            return np.sum(self.cost) > other

        else:
            raise ASRException(
                f"Must compare an instance of ASR with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer, or a float."
            )

    def __ge__(self, other):
        # Between an instance of ASR with another instance of CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) >= np.sum(other.cost)

        # Between an instance of ASR and an integer/float
        elif isinstance(other, (float, int)):
            return np.sum(self.cost) >= other

        else:
            raise ASRException(
                f"Must compare an instance of ASR with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer, or a float."
            )

    def __add__(self, other):
        # Only allows addition between an instance of ASR and another instance of ASR
        if isinstance(other, ASR):
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            cost_combined = np.concatenate((self.cost, other.cost))
            expense_year_combined = np.concatenate((self.expense_year, other.expense_year))
            cost_allocation_combined = self.cost_allocation + other.cost_allocation
            description_combined = self.description + other.description
            vat_portion_combined = np.concatenate((self.vat_portion, other.vat_portion))
            vat_discount_combined = np.concatenate((self.vat_discount, other.vat_discount))

            return ASR(
                start_year=start_year_combined,
                end_year=end_year_combined,
                cost=cost_combined,
                expense_year=expense_year_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
                vat_portion=vat_portion_combined,
                vat_discount=vat_discount_combined,
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
        # Only allows subtraction between an instance of ASR and another instance of ASR
        if isinstance(other, ASR):
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            cost_combined = np.concatenate((self.cost, -other.cost))
            expense_year_combined = np.concatenate((self.expense_year, other.expense_year))
            cost_allocation_combined = self.cost_allocation + other.cost_allocation
            description_combined = self.description + other.description
            vat_portion_combined = np.concatenate((self.vat_portion, other.vat_portion))
            vat_discount_combined = np.concatenate((self.vat_discount, other.vat_discount))

            return ASR(
                start_year=start_year_combined,
                end_year=end_year_combined,
                cost=cost_combined,
                expense_year=expense_year_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
                vat_portion=vat_portion_combined,
                vat_discount=vat_discount_combined,
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
                description=self.description,
                vat_portion=self.vat_portion,
                vat_discount=self.vat_discount,
            )

        else:
            raise ASRException(
                f"Must multiply with an integer or a float; "
                f"{other}({other.__class__.__qualname__}) is not an integer nor a float."
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        # Between an instance of ASR with another instance of CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
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
                    description=self.description,
                    vat_portion=self.vat_portion,
                    vat_discount=self.vat_discount,
                )

        else:
            raise ASRException(
                f"Must divide with an instance of CapitalCost/Intangible/OPEX/ASR/LBT, "
                f"integer or a float; "
                f"{other}({other.__class__.__qualname__}) is not an instance "
                f"of CapitalCost/Intangible/OPEX/ASR/LBT nor an integer nor a float."
            )


@dataclass
class CostOfSales(GeneralCost):
    """
    Manages a cost of sales.

    Parameters
    ----------
    The attributes are inherited from class GeneralCost.

    Notes
    -----
    The inherited attributes (from class GeneralCost) are overridden in this class,
    except for start_year and end_year.
    """
    # Inherited attributes with the modified initialization
    cost: np.ndarray = field(default=None)
    expense_year: np.ndarray = field(default=None)
    cost_allocation: list[FluidType] = field(default=None)

    # Inherited attributes which are being excluded
    description: list[str] = field(default=None, init=False, repr=False)
    vat_portion: np.ndarray = field(default=None, init=False, repr=False)
    vat_discount: float | np.ndarray = field(default=None, init=False, repr=False)
    lbt_portion: np.ndarray = field(default=None, init=False, repr=False)
    lbt_discount: float | np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # Prepare attributes project_duration and project_years
        if self.end_year >= self.start_year:
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise CostOfSalesException(
                f"Project start year {self.start_year} "
                f"is after project's end year {self.end_year}"
            )

        # Prepare attribute expense_year
        if self.expense_year is None:
            self.expense_year = self.project_years.copy()

        else:
            if not isinstance(self.expense_year, np.ndarray):
                raise CostOfSalesException(
                    f"Attribute expense_year must be given as a numpy.ndarray datatype, "
                    f"not a ({self.expense_year.__class__.__qualname__})"
                )

        # Prepare attribute cost
        if self.cost is None:
            self.cost = np.zeros_like(self.expense_year, dtype=np.float64)

        else:
            if not isinstance(self.cost, np.ndarray):
                raise CostOfSalesException(
                    f"Attribute cost must be given as a numpy.ndarray datatype, "
                    f"not a ({self.cost.__class__.__qualname__})"
                )

        # Prepare attribute cost_allocation
        if self.cost_allocation is None:
            self.cost_allocation = [FluidType.OIL for _ in range(len(self.expense_year))]

        else:
            if not isinstance(self.cost_allocation, list):
                raise CostOfSalesException(
                    f"Attribute cost_allocation must be given as a list, "
                    f"not a ({self.cost_allocation.__class__.__qualname__})"
                )

        # Check input data for unequal length
        arr_length = len(self.cost)

        if not all(
            len(arr) == arr_length
            for arr in [self.expense_year, self.cost_allocation]
        ):
            raise CostOfSalesException(
                f"Unequal length of array: "
                f"cost: {len(self.cost)}, "
                f"expense_year: {len(self.expense_year)}, "
                f"cost_allocation: {len(self.cost_allocation)}."
            )

        # Raise an error message: expense_year is after the end year of the project
        if np.max(self.expense_year) > self.end_year:
            raise CostOfSalesException(
                f"Expense year ({np.max(self.expense_year)}) "
                f"is after the end year of the project ({self.end_year})"
            )

        # Raise an error message: expense_year is before the start year of the project
        if np.min(self.expense_year) < self.start_year:
            raise CostOfSalesException(
                f"Expense year ({np.min(self.expense_year)}) "
                f"is before the start year of the project ({self.start_year})"
            )

    def _get_array(self, target_param: np.ndarray) -> np.ndarray:
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
        param_arr = np.bincount(self.expense_year - self.start_year, weights=target_param)
        zeros = np.zeros(self.project_duration - len(param_arr))

        return np.concatenate((param_arr, zeros))

    def get_cost_of_sales_arr(self) -> np.ndarray:
        """
        Create an array of cost of sales according to the corresponding expense year.

        Returns
        -------
        np.ndarray
            The array of cost of sales with length equals to the project duration.

        Notes
        -----
        Array of cost of sales is generated by calling the private method self._get_array().
        """
        return self._get_array(target_param=self.cost)

    def __add__(self, other):
        # Only allows addition between an instance of CostOfSales and another instance of CostOfSales
        if isinstance(other, CostOfSales):
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            expense_year_combined = np.concatenate((self.expense_year, other.expense_year))
            cost_combined = np.concatenate((self.cost, other.cost))
            cost_allocation_combined = self.cost_allocation + other.cost_allocation

            return CostOfSales(
                start_year=start_year_combined,
                end_year=end_year_combined,
                expense_year=expense_year_combined,
                cost=cost_combined,
                cost_allocation=cost_allocation_combined,
            )

        else:
            raise CostOfSalesException(
                f"Must add between an instance of CostOfSales "
                f"with another instance of CostOfSales. "
                f"{other}({other.__class__.__qualname__}) is not "
                f"an instance of CostOfSales."
            )

    def __iadd__(self, other):
        return self.__add__(other)

    def __eq__(self, other):
        # Between two instances of CostOfSales
        if isinstance(other, CostOfSales):
            return all(
                (
                    np.allclose(self.expense_year, other.expense_year),
                    np.allclose(self.cost, other.cost),
                    self.cost_allocation == other.cost_allocation,
                )
            )

        # Between an instance of CostOfSales and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) == other

        else:
            return False

    def __lt__(self, other):
        # Between an instance of CostOfSales with another instance of CostOfSales
        if isinstance(other, CostOfSales):
            return np.sum(self.cost) < np.sum(other.cost)

        # Between an instance of CostOfSales and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) < other

        else:
            raise CostOfSalesException(
                f"Must compare an instance of CostOfSales with another instance "
                f"of CostOfSales, an integer, or a float."
            )

    def __le__(self, other):
        # Between an instance of CostOfSales with another instance of CostOfSales
        if isinstance(other, CostOfSales):
            return np.sum(self.cost) <= np.sum(other.cost)

        # Between an instance of CostOfSales and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) <= other

        else:
            raise CostOfSalesException(
                f"Must compare an instance of CostOfSales with another instance "
                f"of CostOfSales, an integer, or a float."
            )

    def __gt__(self, other):
        # Between an instance of CostOfSales with another instance of CostOfSales
        if isinstance(other, CostOfSales):
            return np.sum(self.cost) > np.sum(other.cost)

        # Between an instance of CostOfSales and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) > other

        else:
            raise CostOfSalesException(
                f"Must compare an instance of CostOfSales with another instance "
                f"of CostOfSales, an integer, or a float."
            )

    def __ge__(self, other):
        # Between an instance of CostOfSales with another instance of CostOfSales
        if isinstance(other, CostOfSales):
            return np.sum(self.cost) >= np.sum(other.cost)

        # Between an instance of CostOfSales and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) >= other

        else:
            raise CostOfSalesException(
                f"Must compare an instance of CostOfSales with another instance "
                f"of CostOfSales, an integer, or a float."
            )


@dataclass
class ASRCalculator:
    """
    Carry out ASR calculator funcionality.

    Parameters
    ----------
    start_year_project : int
        The start year of the project.
    end_year_project : int
        The end year of the project.
    cost_total : np.ndarray
        An array representing the total costs for each cost element.
    begin_year_split : np.ndarray
        The year in which cost distribution begins for each cost element.
    final_year_split : np.ndarray
        The year in which cost distribution ends for each cost element.
    future_rate : np.ndarray, optional
        Future rates applied to costs for each year, defaulting to a rate of 2%.
    vat_portion : np.ndarray, optional
        VAT portion applied to each cost element, defaulting to zero.
    vat_discount : float or np.ndarray, optional
        VAT discount as a float (for a uniform discount) or an array (for element-wise discounts),
        defaulting to zero.
    """
    start_year_project: int
    end_year_project: int
    cost_total: np.ndarray
    begin_year_split: np.ndarray
    final_year_split: np.ndarray
    future_rate: np.ndarray = field(default=None)
    vat_portion: np.ndarray = field(default=None)
    vat_discount: float | np.ndarray = field(default=0.0)

    # Attribute to be defined later on
    project_duration: int = field(default=None, init=False)
    project_years: np.ndarray = field(default=None, init=False)

    def __post_init__(self):
        # Prepare attribute project_duration and project_years
        if self.end_year_project >= self.start_year_project:
            self.project_duration = self.end_year_project - self.start_year_project + 1
            self.project_years = np.arange(self.start_year_project, self.end_year_project + 1, 1)

        else:
            raise ASRCalculatorException(
                f"start year of the project {self.start_year_project} "
                f"is after the end year {self.end_year_project}"
            )

        # Prepare attribute cost_total
        if not isinstance(self.cost_total, np.ndarray):
            raise ASRCalculatorException(
                f"Attribute cost_total must be given as a numpy.ndarray, "
                f"not as {self.cost_total.__class__.__qualname__}"
            )

        self.cost_total = self.cost_total.astype(np.float64)

        # Prepare attribute begin_year_split
        if not isinstance(self.begin_year_split, np.ndarray):
            raise ASRCalculatorException(
                f"Attribute begin_year_split must be given as a numpy.ndarray, "
                f"not as {self.begin_year_split.__class__.__qualname__}"
            )

        self.begin_year_split = self.begin_year_split.astype(int)

        # Prepare attribute final_year_split
        if not isinstance(self.final_year_split, np.ndarray):
            raise ASRCalculatorException(
                f"Attribute final_year_split must be given as a numpy.ndarray, "
                f"not as {self.final_year_split.__class__.__qualname__}"
            )

        self.final_year_split = self.final_year_split.astype(int)

        # Prepare attribute future rate
        if self.future_rate is None:
            self.future_rate = np.repeat(0.02, len(self.cost_total))

        elif self.future_rate is not None:
            if not isinstance(self.future_rate, np.ndarray):
                raise ASRCalculatorException(
                    f"Attribute future_rate must be given as a numpy.ndarray, "
                    f"not as {self.future_rate.__class__.__qualname__}"
                )

        self.future_rate = self.future_rate.astype(np.float64)

        # Prepare attribute VAT portion
        if self.vat_portion is None:
            self.vat_portion = np.zeros_like(self.cost_total)

        elif self.vat_portion is not None:
            if not isinstance(self.vat_portion, np.ndarray):
                raise ASRCalculatorException(
                    f"Attribute vat_portion must be given as a numpy.ndarray,  "
                    f"not {self.vat_portion.__class__.__qualname__}"
                )

        self.vat_portion = self.vat_portion.astype(np.float64)

        # Prepare attribute VAT discount
        if not isinstance(self.vat_discount, (float, np.ndarray)):
            raise ASRCalculatorException(
                f"Attribute VAT discount must be given as a float (for single value) "
                f"or a numpy.ndarray (for an array), not a "
                f"{self.vat_discount.__class__.__qualname__}"
            )

        if isinstance(self.vat_discount, float):
            self.vat_discount = np.repeat(self.vat_discount, len(self.cost_total))

        self.vat_discount = self.vat_discount.astype(np.float64)

        # Check input data for unequal length
        arr_length = len(self.cost_total)

        if not all(
            len(arr) == arr_length
            for arr in [
                self.begin_year_split,
                self.final_year_split,
                self.future_rate,
                self.vat_portion,
                self.vat_discount,
            ]
        ):
            raise ASRCalculatorException(
                f"Unequal length of array: "
                f"cost_total: {len(self.cost_total)}, "
                f"begin_year_split: {len(self.begin_year_split)}, "
                f"final_year_split: {len(self.final_year_split)}, "
                f"future_rate: {len(self.future_rate)}, "
                f"vat_portion: {len(self.vat_portion)}, "
                f"vat_discount: {len(self.vat_discount)}, "
            )

        # Raise error: final_year_split is after the end year of the project
        if np.max(self.final_year_split) > self.end_year_project:
            raise ASRCalculatorException(
                f"Final year split ({np.max(self.final_year_split)}) "
                f"is after the end year of the project ({self.end_year_project})"
            )

        # Raise error: begin_year_split is before the start year of the project
        if np.min(self.begin_year_split) < self.start_year_project:
            raise ASRCalculatorException(
                f"Begin year split ({np.min(self.begin_year_split)}) "
                f"is before the start year of the project"
            )

        # Raise error: incorrect input for final_year_split
        for i, val in enumerate(self.begin_year_split):
            if self.final_year_split[i] < self.begin_year_split[i]:
                raise ASRCalculatorException(
                    f"Attribute final_year_split ({self.final_year_split[i]}) "
                    f"is before begin_year_split ({self.begin_year_split[i]}) "
                )

    def _get_future_values(
        self,
        year_ref: np.ndarray = None,
        vat_rate: np.ndarray = None,
        inflation_rate: np.ndarray = None,
    ) -> np.ndarray:
        """
        Calculate future adjusted cost values accounting for VAT and inflation schemes.

        This method adjusts the `cost_total` array based on value-added tax (VAT) and
        inflation rates. The costs are further adjusted using a future rate over
        the duration of the project.

        Parameters
        ----------
        year_ref : np.ndarray, optional
            A reference year for each cost element. If not provided, the `begin_year_split`
            will be used. Must be a numpy array of the same length as `cost_total`.
        vat_rate : np.ndarray, optional
            VAT rates applied to each cost element. If not provided, a zero array is used
            (i.e., no VAT). Must be a numpy array of the same length as `cost_total`.
        inflation_rate : np.ndarray, optional
            Inflation rates applied to each cost element. If not provided, a zero array is used
            (i.e., no inflation). Must be a numpy array of the same length as `cost_total`.

        Returns
        -------
        np.ndarray
            A 1D array representing the future-adjusted costs for each cost element over
            the project duration.

        Notes
        -----
        - The method first checks the validity of input arrays (`year_ref`, `vat_rate`,
          and `inflation_rate`), ensuring they are either `None` or valid numpy arrays
          of the same length as `cost_total`.
        - Costs are adjusted using a custom function `apply_cost_adjustment` for VAT and inflation.
        - The final cost is further adjusted using the future rate applied over the difference
          between the project end year and the `begin_year_split`.
        """
        # Prepare parameter year_ref
        if year_ref is None:
            year_ref = self.begin_year_split.copy()

        elif year_ref is not None:
            if not isinstance(year_ref, np.ndarray):
                raise ASRCalculatorException(
                    f"Parameter year_ref must be provided as a numpy.ndarray, "
                    f"not a/an {year_ref.__class__.__qualname__}"
                )

        # Prepare parameter vat_rate
        if vat_rate is None:
            vat_rate = np.zeros_like(self.cost_total, dtype=np.float64)

        elif vat_rate is not None:
            if not isinstance(vat_rate, np.ndarray):
                raise ASRCalculatorException(
                    f"Parameter vat_rate must be provided as a numpy.ndarray, "
                    f"not a/an {vat_rate.__class__.__qualname__}"
                )

        # Prepare parameter inflation_rate
        if inflation_rate is None:
            inflation_rate = np.zeros_like(self.cost_total, dtype=np.float64)

        elif inflation_rate is not None:
            if not isinstance(inflation_rate, np.ndarray):
                raise ASRCalculatorException(
                    f"Parameter inflation_rate must be provided as a numpy.ndarray, "
                    f"not a/an {inflation_rate.__class__.__qualname__}"
                )

        # Check for unequal length of arrays
        if not all(
            len(arr) == len(self.cost_total)
            for arr in [
                year_ref,
                vat_rate,
                inflation_rate,
            ]
        ):
            raise ASRCalculatorException(
                f"Unequal length of arrays. "
                f"cost_total: {len(self.cost_total)}, "
                f"year_ref: {len(year_ref)}, "
                f"vat_rate: {len(vat_rate)}, "
                f"inflation_rate: {len(inflation_rate)}"
            )

        # Cost adjustment due to VAT and inflation schemes
        cost_adjusted = np.array(
            [
                apply_cost_adjustment(
                    start_year=self.start_year_project,
                    end_year=self.end_year_project,
                    cost=self.cost_total[i],
                    expense_year=self.begin_year_split[i],
                    project_years=self.project_years,
                    year_ref=year_ref[i],
                    tax_portion=self.vat_portion[i],
                    tax_rate=vat_rate[i],
                    tax_discount=self.vat_discount[i],
                    inflation_rate=inflation_rate[i],
                )
                for i, val in enumerate(self.cost_total)
            ]
        ).ravel()

        # Distance between the end year of project with the begin year split
        year_diff = self.end_year_project - self.begin_year_split + 1

        return cost_adjusted * np.power((1.0 + self.future_rate), year_diff)

    def get_distributed_cost(
        self,
        year_ref: np.ndarray = None,
        vat_rate: np.ndarray = None,
        inflation_rate: np.ndarray = None,
    ) -> np.ndarray:
        """
        Calculate and distribute future costs over the project duration.

        This method splits future costs over a specified number of years for each cost element,
        based on the `begin_year_split` and `final_year_split` values.

        The costs are evenly distributed across the corresponding years
        and summed for each project year.

        Returns
        -------
        np.ndarray
            A 1D array representing the sum of distributed costs for each year of the project duration.

        Notes
        -----
        - The `begin_year_split` defines the start of cost distribution for each cost element.
        - The `final_year_split` defines the end of cost distribution for each cost element.
        - The future values (costs) are obtained from `self.get_future_values()` and divided evenly over
          the number of years in the split period.
        - The costs are distributed for each year between `begin_year_split` and `final_year_split`
          (inclusive) and summed for each year.
        """

        # Total years to decompose each cost elements
        years_to_split = (self.final_year_split - self.begin_year_split + 1).astype(int)

        # Decomposed future value for each cost elements
        future_cost_split = (
            self._get_future_values(
                year_ref=year_ref,
                vat_rate=vat_rate,
                inflation_rate=inflation_rate,
            ) / years_to_split
        )

        # Start and end indices to slice the array 'distributed_cost'
        ids_start = np.array(
            [
                np.argwhere(val == self.project_years).ravel()[0]
                for i, val in enumerate(self.begin_year_split)
            ]
        )

        ids_end = (ids_start + years_to_split).astype(int)

        # Distributed value for each cost elements
        distributed_cost = np.zeros(
            [self.project_duration, len(self.cost_total)], dtype=np.float64
        )

        for i, val in enumerate(self.cost_total):
            distributed_cost[ids_start[i]:ids_end[i], i] = future_cost_split[i]

        # Return the summation of distributed cost for each corresponding project years
        return np.sum(distributed_cost, axis=1, keepdims=False)


@dataclass
class LBTCalculator:
    pass
