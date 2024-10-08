"""
Prepare and classify cost data based on its components. The associated cost components are:
(1) Tangible,
(2) Intangible,
(3) OPEX,
(4) ASR,
(5) LBT,
(6) CostOfSales,
"""

import numpy as np
from dataclasses import dataclass, field

import pyscnomics.econ.depreciation as depr
from pyscnomics.econ.selection import FluidType, DeprMethod
from pyscnomics.econ.costs_tools import (
    get_cost_adjustment_by_inflation,
    calc_indirect_tax,
    calc_distributed_cost,
)


class GeneralCostException(Exception):
    """Exception to be raised for an incorrect use of class GeneralCost"""

    pass


class CapitalException(Exception):
    """Exception to be raised for an incorrect use of class CapitalCost"""

    pass


class IntangibleException(Exception):
    """Exception to be raised for an incorrect use of class Intangible"""

    pass


class OPEXException(Exception):
    """Exception to be raised for an incorrect use of class OPEX"""

    pass


class ASRException(Exception):
    """Exception to be raised for an incorrect use of class ASR"""

    pass


class LBTException(Exception):
    """Exception to be raised for an incorrect use of class LBT"""

    pass


class CostOfSalesException(Exception):
    """Exception to be raised for an incorrect use of class CostOfSales"""

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
    The unit for cost should be in M unit of United States Dollar (USD),
    where the M is stands for 1000. Thus, the unit cost should be: M-USD.
    """

    start_year: int
    end_year: int
    expense_year: np.ndarray
    cost: np.ndarray
    cost_allocation: list[FluidType] = field(default=None)
    description: list[str] = field(default=None)

    # Attribute to be defined later on
    project_duration: int = field(default=None, init=False)
    project_years: np.ndarray = field(default=None, init=False)

    def expenditures_pre_tax(
        self,
        year_inflation: np.ndarray = None,
        inflation_rate: np.ndarray | float = 0.0,
    ) -> np.ndarray:
        """
        Calculate pre-tax expenditures adjusted by inflation.

        This function adjusts the project costs based on inflation rates and allocates
        the costs to their respective expense years, returning the pre-tax expenditures
        for the entire project duration. The function performs cost adjustments using
        inflation rates and expense years, ensuring that costs are properly distributed
        across project years.

        Parameters
        ----------
        year_inflation : np.ndarray, optional
            A NumPy array representing the years during which inflation is applied to the costs.
            If not provided, defaults to repeating the `start_year` for all costs.
        inflation_rate : np.ndarray or float, optional
            The inflation rate(s) to apply to the project costs. If provided as a float,
            a uniform inflation rate is applied. If provided as a NumPy array, different
            rates are applied based on the corresponding project years. Default is 0.0.

        Returns
        -------
        np.ndarray
            A NumPy array representing the pre-tax expenditures adjusted by inflation,
            allocated to the expense years, and extended to the full project duration.

        Notes
        -----
        -   The inflation adjustment is performed using the `calc_inflation` function,
            which computes the inflation-adjusted costs.
        -   Costs are allocated by their associated expense years, and any missing years
            are filled with zeros to match the total project duration.
        -   The indirect tax is allocated across expense years using `np.bincount`.
        """

        # Cost adjustment by inflation scheme
        cost_adjusted_by_inflation = get_cost_adjustment_by_inflation(
            start_year=self.start_year,
            end_year=self.end_year,
            cost=self.cost,
            expense_year=self.expense_year,
            project_years=self.project_years,
            year_inflation=year_inflation,
            inflation_rate=inflation_rate,
        )

        # Allocate costs by their associated expense year
        expenses = np.bincount(
            self.expense_year - self.start_year, weights=cost_adjusted_by_inflation
        )
        zeros = np.zeros(self.project_duration - len(expenses))

        return np.concatenate((expenses, zeros))

    def indirect_taxes(
        self,
        tax_portion: np.ndarray = None,
        tax_rate: np.ndarray | float = 0.0,
        tax_discount: float = 0.0,
    ) -> np.ndarray:
        """
        Calculate and allocate indirect taxes on project costs.

        This function calculates the indirect tax by applying the specified tax portion,
        tax rates, and tax discount to the project costs. The calculated tax is then allocated
        by the associated expense year.

        Parameters
        ----------
        tax_portion : np.ndarray, optional
            A NumPy array representing the portion of the cost subject to tax. If not provided,
            an array of zeros with the same shape as the project cost will be used.
        tax_rate : np.ndarray or float, optional
            A NumPy array or float representing the tax rate applied to the costs. If not provided,
            a default rate of 0.0 will be used. When provided as an array, it should match
            the project years.
        tax_discount : float, optional
            A discount factor applied to the tax, reducing the overall tax impact. The default is 0.0.

        Returns
        -------
        np.ndarray
            A NumPy array containing the allocated indirect taxes per year, adjusted by
            the tax portion, tax rates, and tax discount. The array has the length equal to
            the project duration.

        Notes
        -----
        -   The function computes the tax rate based on the expense year and `start_year`,
            and then applies the tax rate, portion, and discount to the project costs.
        -   The indirect tax is allocated across expense years using `np.bincount`.
        """

        # Calculate indirect tax
        indirect_tax = calc_indirect_tax(
            start_year=self.start_year,
            cost=self.cost,
            expense_year=self.expense_year,
            project_years=self.project_years,
            tax_portion=tax_portion,
            tax_rate=tax_rate,
            tax_discount=tax_discount,
        )

        # Allocate indirect tax by their associated expense year
        expenses = np.bincount(
            self.expense_year - self.start_year, weights=indirect_tax
        )
        zeros = np.zeros(self.project_duration - len(expenses))

        return np.concatenate((expenses, zeros))

    def expenditures_post_tax(
        self,
        year_inflation: np.ndarray = None,
        inflation_rate: np.ndarray | float = 0.0,
        tax_portion: np.ndarray = None,
        tax_rate: np.ndarray | float = 0.0,
        tax_discount: float = 0.0,
    ) -> np.ndarray:
        """
        Calculate post-tax expenditures, adjusted for inflation and indirect taxes.

        This function computes the total project expenditures after adjusting for inflation and
        applying indirect taxes. It first calculates pre-tax expenditures adjusted for inflation,
        and then adds the indirect taxes based on the tax portion, rate, and discount.

        Parameters
        ----------
        year_inflation : np.ndarray, optional
            An array of years representing when inflation impacts each cost. If not provided,
            it defaults to the `start_year` of the project for all costs. The array must have
            the same length as `self.cost`.
        inflation_rate : np.ndarray or float, optional
            The inflation rate(s) to apply. If a single float is provided, it is applied uniformly
            across all years. If an array is provided, each inflation rate corresponds to a specific
            project year (default is 0.0).
        tax_portion : np.ndarray, optional
            A NumPy array representing the portion of each cost subject to taxation. If not provided,
            defaults to an array of zeros, implying no taxation.
        tax_rate : np.ndarray or float, optional
            The tax rate to apply to the costs. If a float is provided, it applies uniformly across all
            project years. If a NumPy array is provided, the rate can vary by year (default is 0.0).
        tax_discount : float, optional
            A discount applied to the tax rate, represented as a decimal fraction (e.g., 0.1 for 10%).
            Default is 0.0, meaning no discount is applied.

        Returns
        -------
        np.ndarray
            An array representing the post-tax expenditures for each project year, adjusted for
            inflation and indirect taxes.

        Notes
        -----
        This function combines two steps:
        1.  Calls `expenditures_pre_tax` to adjust the costs for inflation.
        2.  Calls `indirect_taxes` to compute indirect taxes on the costs based on the specified
            tax portion, rate, and discount.

        The formula used to calculate post-tax expenditures is:
            expenditures_post_tax = expenditures_pre_tax + indirect_tax
        """

        return (
            self.expenditures_pre_tax(
                year_inflation=year_inflation, inflation_rate=inflation_rate
            ) +
            self.indirect_taxes(
                tax_portion=tax_portion, tax_rate=tax_rate, tax_discount=tax_discount,
            )
        )

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
    """

    pis_year: np.ndarray = field(default=None)
    salvage_value: np.ndarray = field(default=None)
    useful_life: np.ndarray = field(default=None)
    depreciation_factor: np.ndarray = field(default=None)
    is_ic_applied: list[bool] = field(default=None)

    def __post_init__(self):
        """
        Handles the following operations/procedures:
        -   Prepare attributes project_duration and project_years,
        -   Prepare attribute description,
        -   Prepare attribute cost_allocation,
        -   Prepare attribute pis_year,
        -   Prepare attribute salvage_value,
        -   Prepare attribute useful_life,
        -   Prepare attribute depreciation_factor,
        -   Prepare attribute is_ic_applied,
        -   Initial check for unequal length of input arrays,
        -   Raise an error: expense_year is after the end year of the project,
        -   Raise an error: expense_year is before the start year of the project,
        """

        # Prepare attributes project_duration and project_years
        if self.end_year >= self.start_year:
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise CapitalException(
                f"start year {self.start_year} "
                f"is after the end year: {self.end_year}"
            )

        # Prepare attribute description
        if self.description is None:
            self.description = [" " for _ in range(len(self.cost))]

        else:
            if not isinstance(self.description, list):
                raise CapitalException(
                    f"Attribute description must be given as a list, "
                    f"not as a/an {self.description.__class__.__qualname__}"
                )

        # Prepare attribute cost_allocation
        if self.cost_allocation is None:
            self.cost_allocation = [FluidType.OIL for _ in range(len(self.cost))]

        else:
            if not isinstance(self.cost_allocation, list):
                raise CapitalException(
                    f"Attribute cost_allocation must be given as a list, "
                    f"not as a/an {self.cost_allocation.__class__.__qualname__}"
                )

        # Prepare attribute pis_year
        if self.pis_year is None:
            self.pis_year = self.expense_year.copy()

        else:
            if not isinstance(self.pis_year, np.ndarray):
                raise CapitalException(
                    f"Attribute pis_year must be given as a numpy.ndarray, "
                    f"not as a/an {self.pis_year.__class__.__qualname__}"
                )

        self.pis_year = self.pis_year.astype(int)

        # Prepare attribute salvage_value
        if self.salvage_value is None:
            self.salvage_value = np.zeros_like(self.cost)

        else:
            if not isinstance(self.salvage_value, np.ndarray):
                raise CapitalException(
                    f"Attribute salvage_value must be given as a numpy.ndarray, "
                    f"not as a/an {self.salvage_value.__class__.__qualname__}"
                )

        self.salvage_value = self.salvage_value.astype(np.float64)

        # Prepare attribute useful_life
        if self.useful_life is None:
            self.useful_life = np.repeat(5.0, len(self.cost))

        else:
            if not isinstance(self.useful_life, np.ndarray):
                raise CapitalException(
                    f"Attribute useful_life must be given as a numpy.ndarray, "
                    f"not as a/an {self.useful_life.__class__.__qualname__}"
                )

        self.useful_life = self.useful_life.astype(np.float64)

        # Prepare attribute depreciation_factor
        if self.depreciation_factor is None:
            self.depreciation_factor = np.repeat(0.5, len(self.cost))

        else:
            if not isinstance(self.depreciation_factor, np.ndarray):
                raise CapitalException(
                    f"Attribute depreciation_factor must be given as a numpy.ndarray, "
                    f"not as a/an {self.depreciation_factor.__class__.__qualname__}"
                )

        self.depreciation_factor = self.depreciation_factor.astype(np.float64)

        # Prepare attribute is_ic_applied
        if self.is_ic_applied is None:
            self.is_ic_applied = [False for _ in range(len(self.cost))]

        else:
            if not isinstance(self.is_ic_applied, list):
                raise CapitalException(
                    f"Attribute is_ic_applied must be given as a list, "
                    f"not as a/an {self.is_ic_applied.__class__.__qualname__}"
                )

        # Check input data for unequal length
        arr_length = len(self.cost)

        if not all(
            len(arr) == arr_length
            for arr in [
                self.expense_year,
                self.cost_allocation,
                self.description,
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
                f"pis_year: {len(self.pis_year)}, "
                f"salvage_value: {len(self.salvage_value)}, "
                f"useful_life: {len(self.useful_life)}, "
                f"depreciation_factor: {len(self.depreciation_factor)}, "
                f"is_ic_applied: {len(self.is_ic_applied)}."
            )

        # Raise an error: expense year is after the end year of the project
        if np.max(self.expense_year) > self.end_year:
            raise CapitalException(
                f"Expense year ({np.max(self.expense_year)}) "
                f"is after the end year of the project ({self.end_year})"
            )

        # Raise an error: expense year is before the start year of the project
        if np.min(self.expense_year) < self.start_year:
            raise CapitalException(
                f"Expense year ({np.min(self.expense_year)}) "
                f"is before the start year of the project ({self.start_year})"
            )

    def total_depreciation_rate(
        self,
        depr_method: DeprMethod = DeprMethod.PSC_DB,
        decline_factor: float | int = 2,
        year_inflation: np.ndarray = None,
        inflation_rate: np.ndarray | float = 0.0,
        tax_portion: np.ndarray = None,
        tax_rate: np.ndarray | float = 0.0,
        tax_discount: float = 0.0,
    ) -> tuple:
        """
        This function calculates the total depreciation for a project based on the specified
        depreciation method, inflation rates, tax portions, and tax discounts. It also returns
        the undepreciated asset value at the end of the depreciation period.

        Parameters
        ----------
        depr_method : DeprMethod, optional
            The depreciation method to be applied. Available methods include:
            - `DeprMethod.SL` for straight-line depreciation,
            - `DeprMethod.DB` for declining balance,
            - `DeprMethod.PSC_DB` for PSC declining balance.
            Default is `DeprMethod.PSC_DB`.
        decline_factor : float or int, optional
            The decline factor used in the declining balance depreciation method.
            Default is 2, which corresponds to double-declining balance.
        year_inflation : np.ndarray, optional
            A NumPy array representing the years during which inflation is applied to the costs.
            If not provided, defaults to repeating the `start_year` for all costs.
        inflation_rate : np.ndarray or float, optional
            The inflation rate(s) to apply to the project costs. If provided as a float,
            a uniform inflation rate is applied. If provided as a NumPy array, different
            rates are applied based on the corresponding project years. Default is 0.0.
        tax_portion : np.ndarray, optional
            A NumPy array representing the portion of the cost subject to tax. If not provided,
            an array of zeros with the same shape as the project cost will be used.
        tax_rate : np.ndarray or float, optional
            A NumPy array or float representing the tax rate applied to the costs. If not provided,
            a default rate of 0.0 will be used. When provided as an array, it should match
            the project years.
        tax_discount : float, optional
            A discount factor applied to the tax, reducing the overall tax impact. The default is 0.0.

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

        # Cost adjustment
        cost_adjusted = (
            get_cost_adjustment_by_inflation(
                start_year=self.start_year,
                end_year=self.end_year,
                cost=self.cost,
                expense_year=self.expense_year,
                project_years=self.project_years,
                year_inflation=year_inflation,
                inflation_rate=inflation_rate,
            ) +
            calc_indirect_tax(
                start_year=self.start_year,
                cost=self.cost,
                expense_year=self.expense_year,
                project_years=self.project_years,
                tax_portion=tax_portion,
                tax_rate=tax_rate,
                tax_discount=tax_discount,
            )
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
        year_inflation: np.ndarray = None,
        inflation_rate: np.ndarray | float = 0.0,
        tax_portion: np.ndarray = None,
        tax_rate: np.ndarray | float = 0.0,
        tax_discount: float = 0.0,
    ) -> np.ndarray:
        """
        Calculate the total book value of depreciation for the asset.

        Parameters
        ----------
        depr_method : DeprMethod, optional
            The depreciation method to be applied. Available methods include:
            - `DeprMethod.SL` for straight-line depreciation,
            - `DeprMethod.DB` for declining balance,
            - `DeprMethod.PSC_DB` for PSC declining balance.
            Default is `DeprMethod.PSC_DB`.
        decline_factor : float or int, optional
            The decline factor used in the declining balance depreciation method.
            Default is 2, which corresponds to double-declining balance.
        year_inflation : np.ndarray, optional
            A NumPy array representing the years during which inflation is applied to the costs.
            If not provided, defaults to repeating the `start_year` for all costs.
        inflation_rate : np.ndarray or float, optional
            The inflation rate(s) to apply to the project costs. If provided as a float,
            a uniform inflation rate is applied. If provided as a NumPy array, different
            rates are applied based on the corresponding project years. Default is 0.0.
        tax_portion : np.ndarray, optional
            A NumPy array representing the portion of the cost subject to tax. If not provided,
            an array of zeros with the same shape as the project cost will be used.
        tax_rate : np.ndarray or float, optional
            A NumPy array or float representing the tax rate applied to the costs. If not provided,
            a default rate of 0.0 will be used. When provided as an array, it should match
            the project years.
        tax_discount : float, optional
            A discount factor applied to the tax, reducing the overall tax impact. The default is 0.0.

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
            year_inflation=year_inflation,
            inflation_rate=inflation_rate,
            tax_portion=tax_portion,
            tax_rate=tax_rate,
            tax_discount=tax_discount,
        )[0]

        return np.cumsum(
            self.expenditures_post_tax(
                year_inflation=year_inflation,
                inflation_rate=inflation_rate,
                tax_portion=tax_portion,
                tax_rate=tax_rate,
                tax_discount=tax_discount,
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
    """

    def __post_init__(self):
        """
        Handles the following operations/procedures:
        -   Prepare attributes project_duration and project_years,
        -   Prepare attribute description,
        -   Prepare attribute cost_allocation,
        -   Initial check for unequal length of input arrays,
        -   Raise an error: expense_year is after the end year of the project,
        -   Raise an error: expense_year is before the start year of the project,
        """

        # Prepare attributes project_duration and project_years
        if self.end_year >= self.start_year:
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise IntangibleException(
                f"start year {self.start_year} "
                f"is after the end year {self.end_year}"
            )

        # Prepare attribute description
        if self.description is None:
            self.description = [" " for _ in range(len(self.cost))]

        else:
            if not isinstance(self.description, list):
                raise IntangibleException(
                    f"Attribute description must be given as a list, "
                    f"not as a/an {self.description.__class__.__qualname__}"
                )

        # Prepare attribute cost_allocation
        if self.cost_allocation is None:
            self.cost_allocation = [FluidType.OIL for _ in range(len(self.cost))]

        else:
            if not isinstance(self.cost_allocation, list):
                raise IntangibleException(
                    f"Attribute cost_allocation must be given as a list, "
                    f"not as a/an {self.cost_allocation.__class__.__qualname__}"
                )

        # Check input data for unequal length
        arr_length = len(self.cost)

        if not all(
            len(arr) == arr_length
            for arr in [
                self.expense_year,
                self.cost_allocation,
                self.description,
            ]
        ):
            raise IntangibleException(
                f"Unequal length of array: "
                f"cost: {len(self.cost)}, "
                f"expense_year: {len(self.expense_year)}, "
                f"cost_allocation: {len(self.cost_allocation)}, "
                f"description: {len(self.description)}, "
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

    def __eq__(self, other):
        # Between two instances of Intangible
        if isinstance(other, Intangible):
            return all(
                (
                    np.allclose(self.cost, other.cost),
                    np.allclose(self.expense_year, other.expense_year),
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

            return Intangible(
                start_year=start_year_combined,
                end_year=end_year_combined,
                cost=cost_combined,
                expense_year=expense_year_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
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

            return Intangible(
                start_year=start_year_combined,
                end_year=end_year_combined,
                cost=cost_combined,
                expense_year=expense_year_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
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
                )

        else:
            raise IntangibleException(
                f"Must divide with an instance of CapitalCost/Intangible/OPEX/ASR/LBT, "
                f"integer or a float; {other}({other.__class__.__qualname__}) is not an "
                f"instance of CapitalCost/Intangible/OPEX/ASR/LBT nor an integer nor a float."
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

    Notes
    -----
    The unit used in the fixed_cost should be in M unit of United States Dollar (USD),
    where the M is stands for 1000. Thus, the unit cost should be: M-USD.
    """

    fixed_cost: np.ndarray = field(default=None)
    prod_rate: np.ndarray = field(default=None)
    cost_per_volume: np.ndarray = field(default=None)

    # Attribute to be defined later on
    variable_cost: np.ndarray = field(default=None, init=False)
    cost: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """
        Handles the following operations/procedures:
        -   Prepare attributes project_duration and project_years,
        -   Prepare attribute description,
        -   Prepare attribute cost_allocation,
        -   Prepare attributes prod_rate and cost_per_volume,
        -   Prepare attribute variable_cost,
        -   Prepare attribute cost,
        -   Raise an error: expense_year is after the end year of the project,
        -   Raise an error: expense_year is before the start year of the project,
        """

        # Prepare attributes project_duration and project_years
        if self.end_year >= self.start_year:
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise OPEXException(
                f"start year {self.start_year} "
                f"is after the end year {self.end_year}"
            )

        # Prepare attribute description
        if self.description is None:
            self.description = [" " for _ in range(len(self.fixed_cost))]

        else:
            if not isinstance(self.description, list):
                raise OPEXException(
                    f"Attribute description must be given as a list, "
                    f"not as a/an {self.description.__class__.__qualname__}"
                )

        # Prepare attribute cost_allocation
        if self.cost_allocation is None:
            self.cost_allocation = [FluidType.OIL for _ in range(len(self.fixed_cost))]

        else:
            if not isinstance(self.cost_allocation, list):
                raise OPEXException(
                    f"Attribute cost_allocation must be given as a list, "
                    f"not as a/an {self.cost_allocation.__class__.__qualname__}"
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

        # Raise an error: expense year is after the end year of the project
        if np.max(self.expense_year) > self.end_year:
            raise OPEXException(
                f"Expense year ({np.max(self.expense_year)}) "
                f"is after the end year of the project ({self.end_year})"
            )

        # Raise an error: expense year is before the start year of the project
        if np.min(self.expense_year) < self.start_year:
            raise OPEXException(
                f"Expense year ({np.min(self.expense_year)}) "
                f"is before the start year of the project ({self.start_year})"
            )

    def __eq__(self, other):
        # Between two instances of OPEX
        if isinstance(other, OPEX):
            return all(
                (
                    np.allclose(self.fixed_cost, other.fixed_cost),
                    np.allclose(self.variable_cost, other.variable_cost),
                    np.allclose(self.expense_year, other.expense_year),
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
            fixed_cost_combined = np.concatenate((self.fixed_cost, other.fixed_cost))
            prod_rate_combined = np.concatenate((self.prod_rate, other.prod_rate))
            cost_per_volume_combined = np.concatenate((self.cost_per_volume, other.cost_per_volume))

            return OPEX(
                start_year=start_year_combined,
                end_year=end_year_combined,
                expense_year=expense_year_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
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
            fixed_cost_combined = np.concatenate((self.fixed_cost, -other.fixed_cost))
            prod_rate_combined = np.concatenate((self.prod_rate, -other.prod_rate))
            cost_per_volume_combined = np.concatenate((self.cost_per_volume, other.cost_per_volume))

            return OPEX(
                start_year=start_year_combined,
                end_year=end_year_combined,
                expense_year=expense_year_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
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
                    fixed_cost=self.fixed_cost / other,
                    prod_rate=self.prod_rate / other,
                    cost_per_volume=self.cost_per_volume,
                )

        else:
            raise OPEXException(
                f"Must divide with an instance of CapitalCost/Intangible/OPEX/ASR/LBT, "
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

    final_year : np.ndarray
        The year in which cost distribution ends for each cost element.
    future_rate : float or np.ndarray, optional
        Future rates applied to costs for each year, defaulting to a rate of 0%.
    """

    final_year: np.ndarray = field(default=None)
    future_rate: float | np.ndarray = field(default=None)

    def __post_init__(self):
        """
        Handles the following operations/procedures:
        -   Prepare attributes project_duration and project_years,
        -   Prepare attribute dscription,
        -   Prepare attribute cost_allocation,
        -   Prepare attribute final_year,
        -   Prepare attribute future_rate,
        -   Initial check for unequal length of input arrays,
        -   Raise an error: final_year is before expense_year,
        -   Raise an error: expense_year is after the end year of the project,
        -   Raise an error: expense_year is before the start year of the project,
        """

        # Prepare attributes project_duration and project_years
        if self.end_year >= self.start_year:
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise ASRException(
                f"The start_year of the project ({self.start_year}) is after "
                f"the end year of the project ({self.end_year})"
            )

        # Prepare attribute description
        if self.description is None:
            self.description = [" " for _ in range(len(self.cost))]

        else:
            if not isinstance(self.description, list):
                raise ASRException(
                    f"Attribute description must be given as a list, "
                    f"not as a/an {self.description.__class__.__qualname__}"
                )

        # Prepare attribute cost_allocation
        if self.cost_allocation is None:
            self.cost_allocation = [FluidType.OIL for _ in range(len(self.cost))]

        else:
            if not isinstance(self.cost_allocation, list):
                raise ASRException(
                    f"Attribute cost_allocation must be given as a list, "
                    f"not as a/an {self.cost_allocation.__class__.__qualname__}"
                )

        # Prepare attribute final_year
        if self.final_year is None:
            self.final_year = self.expense_year.copy()

        else:
            if not isinstance(self.final_year, np.ndarray):
                raise ASRException(
                    f"Attribute final_year must be given as a numpy.ndarray, "
                    f"not as a/an {self.final_year.__class__.__qualname__}"
                )

        self.final_year = self.final_year.astype(int)

        # Prepare attribute future_rate
        if self.future_rate is None:
            self.future_rate = np.repeat(0.0, len(self.cost)).astype(np.float64)

        else:
            if isinstance(self.future_rate, float):
                if self.future_rate < 0.0 or self.future_rate > 1.0:
                    raise ASRException(
                        f"Attribute future_rate must be between 0 and 1"
                    )

        self.future_rate = self.future_rate.astype(np.float64)

        # Check input data for unequal length
        arr_length = len(self.cost)

        if not all(
            len(arr) == arr_length
            for arr in [
                self.expense_year,
                self.cost_allocation,
                self.description,
                self.final_year,
                self.future_rate,
            ]
        ):
            raise ASRException(
                f"Unequal length of array: "
                f"cost: {len(self.cost)}, "
                f"expense_year: {len(self.expense_year)}, "
                f"cost_allocation: {len(self.cost_allocation)}, "
                f"description: {len(self.description)}, "
                f"final_year: {len(self.final_year)}, "
                f"future_rate: {len(self.future_rate)}"
            )

        # Raise an error: final_year is before expense_year
        for i, _ in enumerate(self.final_year):
            if self.final_year[i] < self.expense_year[i]:
                raise ASRException(
                    f"Attribute final year ({self.final_year[i]}) "
                    f"is before the start expense year ({self.expense_year[i]})"
                )

        # Raise an error: final_year is after the end year of the project
        if np.max(self.final_year) > self.end_year:
            raise ASRException(
                f"Final year ({np.max(self.final_year)}) "
                f"is after the end year of the project ({self.end_year})"
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

    def _calc_future_cost(self) -> np.ndarray:
        """
        Calculate the future cost of each element by applying a growth rate
        over the project's timeline.

        This method computes the future cost of each element based on the time difference
        between the project's end year and the expense year, adjusting for a specified future rate.

        Returns
        -------
        np.ndarray
            A 1D array of future costs for each element, adjusted by the future growth rate
            over the time difference between the expense year and the project end year.

        Notes
        -----
        -   The future cost is calculated by multiplying the current cost by
            `(1 + future_rate) ** year_diff`, where `year_diff` is the number
            of years between the `expense_year` and the `end_year`.
        """

        # Distance between the end year of the project with expense year
        year_diff = self.end_year - self.expense_year + 1

        return self.cost * np.power((1.0 + self.future_rate), year_diff)

    def expenditures_pre_tax(
        self,
        year_inflation: np.ndarray = None,
        inflation_rate: np.ndarray | float = 0.0,
    ) -> np.ndarray:
        """
        Calculate the pre-tax expenditures by adjusting future costs for inflation and
        distributing them over the project duration.

        This method first calculates the future cost of each element, adjusts these costs
        for inflation, and then distributes the adjusted costs over the project duration.
        The total pre-tax expenditures are summed for each project year.

        Parameters
        ----------
        year_inflation : np.ndarray, optional
            A NumPy array representing the years during which inflation is applied to the costs.
            If not provided, defaults to repeating the `start_year` for all costs.
        inflation_rate : np.ndarray or float, optional
            The inflation rate(s) to apply to the project costs. If provided as a float,
            a uniform inflation rate is applied. If provided as a NumPy array, different
            rates are applied based on the corresponding project years. Default is 0.0.

        Returns
        -------
        np.ndarray
            A 1D array representing the total pre-tax expenditures for each project year
            after adjusting for inflation and distributing the future costs.

        Notes
        -----
        -   The method first calculates the future cost using the `_calc_future_cost` method.
        -   It adjusts the future cost for inflation using `get_cost_adjustment_by_inflation`,
            which accounts for inflation rates across the project's duration.
        -   Finally, it distributes the adjusted cost using `calc_distributed_cost`, summing
            the distributed costs for each year to provide the total pre-tax expenditures.
        """

        # Calculate future cost
        cost_future = self._calc_future_cost()

        # Cost adjustment due to inflation
        cost_adjusted_by_inflation = get_cost_adjustment_by_inflation(
            start_year=self.start_year,
            end_year=self.end_year,
            cost=cost_future,
            expense_year=self.expense_year,
            project_years=self.project_years,
            year_inflation=year_inflation,
            inflation_rate=inflation_rate,
        )

        # Calculate distributed cost
        distributed_cost = calc_distributed_cost(
            cost=cost_adjusted_by_inflation,
            expense_year=self.expense_year,
            final_year=self.final_year,
            project_years=self.project_years,
            project_duration=self.project_duration,
        )

        return np.sum(distributed_cost, axis=1, keepdims=False)

    def indirect_taxes(
        self,
        tax_portion: np.ndarray = None,
        tax_rate: np.ndarray | float = 0.0,
        tax_discount: float = 0.0,
    ) -> np.ndarray:
        """
        Calculate and distribute indirect taxes over the project duration.

        This method calculates indirect taxes based on future costs, tax portions, tax rates,
        and any applicable tax discounts. The taxes are then distributed across the project's
        timeline, and the total indirect taxes are summed for each project year.

        Parameters
        ----------
        tax_portion : np.ndarray, optional
            A NumPy array representing the portion of the cost subject to tax. If not provided,
            an array of zeros with the same shape as the project cost will be used.
        tax_rate : np.ndarray or float, optional
            A NumPy array or float representing the tax rate applied to the costs. If not provided,
            a default rate of 0.0 will be used. When provided as an array, it should match
            the project years.
        tax_discount : float, optional
            A discount factor applied to the tax, reducing the overall tax impact.
            The default is 0.0.

        Returns
        -------
        np.ndarray
            A 1D array representing the total distributed indirect taxes for each project year.

        Notes
        -----
        -   The method first calculates the future cost using the `_calc_future_cost` method.
        -   It calculates the indirect tax using `calc_indirect_tax`, which applies the
            tax portion, tax rate, and tax discount.
        -   The method then distributes the calculated taxes across the project's timeline
            using `calc_distributed_cost`.
        -   The final result is the sum of the distributed taxes for each project year.
        """

        # Calculate future cost
        cost_future = self._calc_future_cost()

        # Calculate indirect tax
        indirect_tax = calc_indirect_tax(
            start_year=self.start_year,
            cost=cost_future,
            expense_year=self.expense_year,
            project_years=self.project_years,
            tax_portion=tax_portion,
            tax_rate=tax_rate,
            tax_discount=tax_discount,
        )

        # Calculate distributed indirect taxes
        distributed_taxes = calc_distributed_cost(
            cost=indirect_tax,
            expense_year=self.expense_year,
            final_year=self.final_year,
            project_years=self.project_years,
            project_duration=self.project_duration,
        )

        return np.sum(distributed_taxes, axis=1, keepdims=False)

    def __eq__(self, other):
        # Between two instances of ASR
        if isinstance(other, ASR):
            return all(
                (
                    np.allclose(self.cost, other.cost),
                    np.allclose(self.expense_year, other.expense_year),
                    np.allclose(self.final_year, other.final_year),
                    np.allclose(self.future_rate, other.future_rate),
                    self.cost_allocation == other.cost_allocation,
                )
            )

        # Between an instance of ASR and integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) == other

        else:
            return False

    def __lt__(self, other):
        # Between an instance of ASR with another instance of CapitalCost/Instangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) < np.sum(other.cost)

        # Between an instance of ASR and an integer/float
        elif isinstance(other, (float, int)):
            return np.sum(self.cost) < other

        else:
            raise ASRException(
                f"Must compare an instance of ASR with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer or a float. "
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
            final_year_combined = np.concatenate((self.final_year, other.final_year))
            future_rate_combined = np.concatenate((self.future_rate, other.future_rate))

            return ASR(
                start_year=start_year_combined,
                end_year=end_year_combined,
                cost=cost_combined,
                expense_year=expense_year_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
                final_year=final_year_combined,
                future_rate=future_rate_combined,
            )

        else:
            raise ASRException(
                f"Must add between an instance of ASR with another instance of ASR. "
                f"{other}({other.__class__.__qualname__}) is not an instance of ASR."
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
            final_year_combined = np.concatenate((self.final_year, other.final_year))
            future_rate_combined = np.concatenate((self.future_rate, other.future_rate))

            return ASR(
                start_year=start_year_combined,
                end_year=end_year_combined,
                cost=cost_combined,
                expense_year=expense_year_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
                final_year=final_year_combined,
                future_rate=future_rate_combined,
            )

        else:
            raise ASRException(
                f"Must subtract between an instance of ASR with another instance of ASR. "
                f"{other}{other.__class__.__qualname__} is not an instance of ASR. "
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
                final_year=self.final_year,
                future_rate=self.future_rate,
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
                    final_year=self.final_year,
                    future_rate=self.future_rate,
                )

        else:
            raise ASRException(
                f"Must divide with an instance of CapitalCost/Intangible/OPEX/ASR/LBT, "
                f"integer or a float; {other}({other.__class__.__qualname__}) is not an instance "
                f"of CapitalCost/Intangible/OPEX/ASR/LBT nor an integer nor a float."
            )


@dataclass
class LBT(GeneralCost):
    """
    Manages LBT asset.

    Parameters
    ----------
    final_year : np.ndarray
        The year in which cost distribution ends for each cost element.
    utilized_land_area : np.ndarray
        The utilized land area.
    utilized_building_area : np.ndarray
        The utilized building area.
    njop_land : np.ndarray
        NJOP related to land.
    njop_building : np.ndarray
        NJOP related to building.
    gross_revenue : np.ndarray
        The array of gross revenues.
    """

    final_year: np.ndarray = field(default=None)
    utilized_land_area: np.ndarray = field(default=None)
    utilized_building_area: np.ndarray = field(default=None)
    njop_land: np.ndarray = field(default=None)
    njop_building: np.ndarray = field(default=None)
    gross_revenue: np.ndarray = field(default=None)

    # Override attribute cost
    cost: np.ndarray = field(default=None)

    # Attributes to be defined later on (associated with surface and subsurface lbt components)
    _surface_lbt_cost: np.ndarray = field(default=None, init=False, repr=False)
    _subsurface_lbt_cost: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """
        Handles the following operations/procedures:
        -   Prepare attributes project_duration and project_years,
        -   Prepare attribute description,
        -   Prepare attribute cost_allocation,
        -   Prepare attribute final_year,
        -   Prepare attribute utilized_land_area,
        -   Prepare attribute utilized_building_area,
        -   Prepare attribute njop_land,
        -   Prepare attribute njop_building,
        -   Prepare attribute gross_revenue,
        -   Prepare attribute _surface_lbt_cost,
        -   Prepare attribute _subsurface_lbt_cost,
        -   Prepare attribute cost,
        -   Check for unequal length of input arrays,
        -   Raise an error: final_year is before expense_year,
        -   Raise an error: final_year is after the end year of the project,
        -   Raise an error: expense_year is after the end year of the project,
        -   Raise an error: expense_year is before the start year of the project,
        """

        # Prepare attributes project_duration and project_years
        if self.end_year >= self.start_year:
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise LBTException(
                f"The start year of the project ({self.start_year}) is after "
                f"the end year of the project ({self.end_year})"
            )

        # Prepare attribute description
        if self.description is None:
            self.description = [" " for _ in range(len(self.expense_year))]

        else:
            if not isinstance(self.description, list):
                raise LBTException(
                    f"Attribute description must be given as a list, "
                    f"not as a/an {self.description.__class__.__qualname__}"
                )

        # Prepare attribute cost_allocation
        if self.cost_allocation is None:
            self.cost_allocation = [FluidType.OIL for _ in range(len(self.expense_year))]

        else:
            if not isinstance(self.cost_allocation, list):
                raise LBTException(
                    f"Attribute cost_allocation must be given as a list, "
                    f"not as a/an {self.cost_allocation.__class__.__qualname__}"
                )

        # Prepare attribute final_year
        if self.final_year is None:
            self.final_year = self.expense_year.copy()

        else:
            if not isinstance(self.final_year, np.ndarray):
                raise LBTException(
                    f"Attribute final_year must be given as a numpy.ndarray, "
                    f"not as a/an {self.final_year.__class__.__qualname__}"
                )

        self.final_year = self.final_year.astype(int)

        # Prepare attribute utilized_land_area
        if self.utilized_land_area is None:
            self.utilized_land_area = np.zeros_like(self.expense_year)

        else:
            if not isinstance(self.utilized_land_area, np.ndarray):
                raise LBTException(
                    f"Attribute utilized_land_area must be given as a numpy.ndarray, "
                    f"not as a/an {self.utilized_land_area.__class__.__qualname__}"
                )

        self.utilized_land_area = self.utilized_land_area.astype(np.float64)

        # Prepare attribute utilized_building_area
        if self.utilized_building_area is None:
            self.utilized_building_area = np.zeros_like(self.expense_year)

        else:
            if not isinstance(self.utilized_building_area, np.ndarray):
                raise LBTException(
                    f"Attribute utilized_building_area must be given as a numpy.ndarray, "
                    f"not as a/an {self.utilized_building_area.__class__.__qualname__}"
                )

        self.utilized_building_area = self.utilized_building_area.astype(np.float64)

        # Prepare attribute njop_land
        if self.njop_land is None:
            self.njop_land = np.zeros_like(self.expense_year)

        else:
            if not isinstance(self.njop_land, np.ndarray):
                raise LBTException(
                    f"Attribute njop_land must be given as a numpy.ndarray, "
                    f"not as a/an {self.njop_land.__class__.__qualname__}"
                )

        self.njop_land = self.njop_land.astype(np.float64)

        # Prepare attribute njop_building
        if self.njop_building is None:
            self.njop_building = np.zeros_like(self.expense_year)

        else:
            if not isinstance(self.njop_building, np.ndarray):
                raise LBTException(
                    f"Attribute njop_building must be given as a numpy.ndarray, "
                    f"not as a/an {self.njop_building.__class__.__qualname__}"
                )

        self.njop_building = self.njop_building.astype(np.float64)

        # Prepare attribute gross_revenue
        if self.gross_revenue is None:
            self.gross_revenue = np.zeros_like(self.expense_year)

        else:
            if not isinstance(self.gross_revenue, np.ndarray):
                raise LBTException(
                    f"Attribute gross_revenue must be given as a numpy.ndarray, "
                    f"not as a/an {self.gross_revenue.__class__.__qualname__}"
                )

        self.gross_revenue = self.gross_revenue.astype(np.float64)

        # Prepare attribute _surface_lbt_cost
        surface_land = self.utilized_land_area * self.njop_land
        surface_building = self.utilized_building_area * self.njop_building
        self._surface_lbt_cost = (0.5 / 100) * (40 / 100) * 1.0 * (surface_land + surface_building)

        # Prepare attribute _subsurface_lbt_cost
        self._subsurface_lbt_cost = (0.5 / 100) * (40 / 100) * 10.04 * self.gross_revenue

        # Prepare attribute cost
        if self.cost is None:
            self.cost = self._surface_lbt_cost + self._subsurface_lbt_cost

        else:
            if not isinstance(self.cost, np.ndarray):
                raise LBTException(
                    f"Attribute cost must be given as a numpy.ndarray, "
                    f"not as a/an {self.cost.__class__.__qualname__}"
                )

        self.cost = self.cost.astype(np.float64)

        # Check for unequal length of arrays
        arr_length = len(self.expense_year)

        if not all(
            len(arr) == arr_length
            for arr in [
                self.description,
                self.cost_allocation,
                self.final_year,
                self.utilized_land_area,
                self.utilized_building_area,
                self.njop_land,
                self.njop_building,
                self.gross_revenue,
                self.cost,
            ]
        ):
            raise LBTException(
                f"Unequal length of arrays: "
                f"expense_year: {len(self.expense_year)}, "
                f"description: {len(self.description)}, "
                f"cost_allocation: {len(self.cost_allocation)}, "
                f"final_year: {len(self.final_year)}, "
                f"utilized_land_area: {len(self.utilized_land_area)}, "
                f"utilized_building_area: {len(self.utilized_building_area)}, "
                f"njop_land: {len(self.njop_land)}, "
                f"njop_building: {len(self.njop_building)}, "
                f"gross_revenue: {len(self.gross_revenue)}, "
                f"cost: {len(self.cost)}"
            )

        # Raise an error: final_year is before expense_year
        for i, _ in enumerate(self.expense_year):
            if self.final_year[i] < self.expense_year[i]:
                raise LBTException(
                    f"Attribute final_year ({self.final_year[i]}) "
                    f"is before the start expense year ({self.expense_year[i]})"
                )

        # Raise an error: final_year is after the end year of the project
        if np.max(self.final_year) > self.end_year:
            raise LBTException(
                f"Final year ({np.max(self.final_year)}) "
                f"is after the end year of the project ({self.end_year})"
            )

        # Raise an error: expense year is after the end year of the project
        if np.max(self.expense_year) > self.end_year:
            raise LBTException(
                f"Expense year ({np.max(self.expense_year)}) "
                f"is after the end year of the project ({self.end_year})"
            )

        # Raise an error: expense_year is before the start year of the project
        if np.min(self.expense_year) < self.start_year:
            raise LBTException(
                f"Expense year ({np.min(self.expense_year)}) "
                f"is before the start year of the project ({self.start_year})"
            )

    def expenditures_pre_tax(
        self,
        year_inflation: np.ndarray = None,
        inflation_rate: np.ndarray | float = 0.0,
    ) -> np.ndarray:
        """
        Calculate pre-tax expenditures adjusted for inflation.

        This method calculates the total pre-tax expenditures for each project year
        by adjusting the cost for inflation and distributing the adjusted costs
        over the project's timeline.

        Parameters
        ----------
        year_inflation : np.ndarray, optional
            A NumPy array representing the years during which inflation is applied to the costs.
            If not provided, defaults to repeating the `start_year` for all costs.
        inflation_rate : np.ndarray or float, optional
            The inflation rate(s) to apply to the project costs. If provided as a float,
            a uniform inflation rate is applied. If provided as a NumPy array, different
            rates are applied based on the corresponding project years. Default is 0.0.

        Returns
        -------
        np.ndarray
            A 1D array representing the total pre-tax expenditures for each project year,
            adjusted for inflation.

        Notes
        -----
        -   The method first adjusts future costs using the `get_cost_adjustment_by_inflation`
            function, which applies inflation rates based on the provided parameters.
        -   It then calculates the distributed costs across the project's timeline using
            `calc_distributed_cost`.
        -   The final result is the sum of the distributed costs for each project year.
        """

        # Cost adjustment due to inflation
        cost_adjusted_by_inflation = get_cost_adjustment_by_inflation(
            start_year=self.start_year,
            end_year=self.end_year,
            cost=self.cost,
            expense_year=self.expense_year,
            project_years=self.project_years,
            year_inflation=year_inflation,
            inflation_rate=inflation_rate,
        )

        # Calculate distributed cost
        distributed_cost = calc_distributed_cost(
            cost=cost_adjusted_by_inflation,
            expense_year=self.expense_year,
            final_year=self.final_year,
            project_years=self.project_years,
            project_duration=self.project_duration,
        )

        return np.sum(distributed_cost, axis=1, keepdims=False)

    def indirect_taxes(
        self,
        tax_portion: np.ndarray = None,
        tax_rate: np.ndarray | float = 0.0,
        tax_discount: float = 0.0,
    ) -> np.ndarray:
        """
        Calculate and distribute indirect taxes over the project duration.

        This method computes indirect taxes by applying the provided tax rates, portions, and
        discounts to the associated costs. The calculated taxes are then distributed over the
        project's timeline, and the total indirect taxes are summed for each project year.

        Parameters
        ----------
        tax_portion : np.ndarray, optional
            A NumPy array representing the portion of the cost subject to tax. If not provided,
            an array of zeros with the same shape as the project cost will be used.
        tax_rate : np.ndarray or float, optional
            A NumPy array or float representing the tax rate applied to the costs. If not provided,
            a default rate of 0.0 will be used. When provided as an array, it should match
            the project years.
        tax_discount : float, optional
            A discount factor applied to the tax, reducing the overall tax impact.
            The default is 0.0.

        Returns
        -------
        np.ndarray
            A 1D array representing the total distributed indirect taxes for each project year.

        Notes
        -----
        -   The indirect tax is first calculated using the `calc_indirect_tax` function, which
            applies the specified tax rate, portion, and discount to the project's costs.
        -   The calculated taxes are then distributed over the project timeline using the
            `calc_distributed_cost` function.
        -   The final result is the sum of distributed indirect taxes for each year in
            the project timeline.
        """

        # Calculate indirect tax
        indirect_tax = calc_indirect_tax(
            start_year=self.start_year,
            cost=self.cost,
            expense_year=self.expense_year,
            project_years=self.project_years,
            tax_portion=tax_portion,
            tax_rate=tax_rate,
            tax_discount=tax_discount,
        )

        # Calculate distributed indirect taxes
        distributed_taxes = calc_distributed_cost(
            cost=indirect_tax,
            expense_year=self.expense_year,
            final_year=self.final_year,
            project_years=self.project_years,
            project_duration=self.project_duration,
        )

        return np.sum(distributed_taxes, axis=1, keepdims=False)

    def __eq__(self, other):
        # Between two instances of LBT
        if isinstance(other, LBT):
            return all(
                (
                    np.allclose(self.cost, other.cost),
                    np.allclose(self.expense_year, other.expense_year),
                    np.allclose(self.final_year, other.final_year),
                    np.allclose(self.utilized_land_area, other.utilized_land_area),
                    np.allclose(self.utilized_building_area, other.utilized_building_area),
                    np.allclose(self.njop_land, other.njop_land),
                    np.allclose(self.njop_building, other.njop_building),
                    np.allclose(self.gross_revenue, other.gross_revenue),
                    self.cost_allocation == other.cost_allocation,
                )
            )

        # Between an instance of LBT and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) == other

        else:
            return False

    def __lt__(self, other):
        # Between an instance of LBT with another instance of CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) < np.sum(other.cost)

        # Between an instance of LBT and an integer/ a float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) < other

        else:
            raise LBTException(
                f"Must compare an instance of LBT with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer, or a float."
            )

    def __le__(self, other):
        # Between an instance of LBT with another instance of CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) <= np.sum(other.cost)

        # Between an instance of LBT and an integer/ a float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) <= other

        else:
            raise LBTException(
                f"Must compare an instance of LBT with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer, or a float."
            )

    def __gt__(self, other):
        # Between an instance of LBT with another instance of CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) > np.sum(other.cost)

        # Between an instance of LBT and an integer/ a float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) > other

        else:
            raise LBTException(
                f"Must compare an instance of LBT with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer, or a float."
            )

    def __ge__(self, other):
        # Between an instance of LBT with another instance of CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) >= np.sum(other.cost)

        # Between an instance of LBT and an integer/ a float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) >= other

        else:
            raise LBTException(
                f"Must compare an instance of LBT with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer, or a float."
            )

    def __add__(self, other):
        # Only allows addition between an instance of LBT and another instance of LBT
        if isinstance(other, LBT):
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            expense_year_combined = np.concatenate((self.expense_year, other.expense_year))
            final_year_combined = np.concatenate((self.final_year, other.final_year))
            utilized_land_area_combined = np.concatenate(
                (self.utilized_land_area, other.utilized_land_area)
            )
            utilized_building_area_combined = np.concatenate(
                (self.utilized_building_area, other.utilized_building_area)
            )
            njop_land_combined = np.concatenate((self.njop_land, other.njop_land))
            njop_building_combined = np.concatenate((self.njop_building, other.njop_building))
            gross_revenue_combined = np.concatenate((self.gross_revenue, other.gross_revenue))
            cost_combined = np.concatenate((self.cost, other.cost))
            cost_allocation_combined = self.cost_allocation + other.cost_allocation
            description_combined = self.description + other.description

            return LBT(
                start_year=start_year_combined,
                end_year=end_year_combined,
                expense_year=expense_year_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
                final_year=final_year_combined,
                utilized_land_area=utilized_land_area_combined,
                utilized_building_area=utilized_building_area_combined,
                njop_land=njop_land_combined,
                njop_building=njop_building_combined,
                gross_revenue=gross_revenue_combined,
                cost=cost_combined,
            )

        else:
            raise LBTException(
                f"Must add between an instance of LBT with another instance of LBT. "
                f"{other}({other.__class__.__qualname__}) is not an instance of LBT."
            )

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        # Only allows subtraction between an instance of LBT and another instance of LBT
        if isinstance(other, LBT):
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            expense_year_combined = np.concatenate((self.expense_year, other.expense_year))
            final_year_combined = np.concatenate((self.final_year, other.final_year))
            utilized_land_area_combined = np.concatenate(
                (self.utilized_land_area, other.utilized_land_area)
            )
            utilized_building_area_combined = np.concatenate(
                (self.utilized_building_area, other.utilized_building_area)
            )
            njop_land_combined = np.concatenate((self.njop_land, -other.njop_land))
            njop_building_combined = np.concatenate((self.njop_building, -other.njop_building))
            gross_revenue_combined = np.concatenate((self.gross_revenue, -other.gross_revenue))
            cost_combined = np.concatenate((self.cost, -other.cost))
            cost_allocation_combined = self.cost_allocation + other.cost_allocation
            description_combined = self.description + other.description

            return LBT(
                start_year=start_year_combined,
                end_year=end_year_combined,
                expense_year=expense_year_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
                final_year=final_year_combined,
                utilized_land_area=utilized_land_area_combined,
                utilized_building_area=utilized_building_area_combined,
                njop_land=njop_land_combined,
                njop_building=njop_building_combined,
                gross_revenue=gross_revenue_combined,
                cost=cost_combined,
            )

        else:
            raise LBTException(
                f"Must subtract between an instance of LBT with another instance of LBT. "
                f"{other}({other.__class__.__qualname__}) is not an instance of LBT."
            )

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        # Multiplication is allowed only with an integer/a float
        if isinstance(other, (int, float)):
            return LBT(
                start_year=self.start_year,
                end_year=self.end_year,
                expense_year=self.expense_year,
                cost_allocation=self.cost_allocation,
                description=self.description,
                final_year=self.final_year,
                utilized_land_area=self.utilized_land_area,
                utilized_building_area=self.utilized_building_area,
                njop_land=self.njop_land * other,
                njop_building=self.njop_building * other,
                gross_revenue=self.gross_revenue * other,
            )

        else:
            raise LBTException(
                f"Must multiply with an integer or a float; "
                f"{other}({other.__class__.__qualname__}) is not an integer nor a float."
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        # Between an instance of LBT with another instance of CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
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
                    expense_year=self.expense_year,
                    cost_allocation=self.cost_allocation,
                    description=self.description,
                    final_year=self.final_year,
                    utilized_land_area=self.utilized_land_area,
                    utilized_building_area=self.utilized_building_area,
                    njop_land=self.njop_land / other,
                    njop_building=self.njop_building / other,
                    gross_revenue=self.gross_revenue / other,
                )

        else:
            raise LBTException(
                f"Must divide with an instance of CapitalCost/Intangible/OPEX/ASR/LBT, "
                f"integer or a float; {other}({other.__class__.__qualname__}) is not an instance "
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
    The inherited attributes (expense_year and cost) are overridden in this class.
    """

    # Inherited attributes with the modified initialization
    expense_year: np.ndarray = field(default=None)
    cost: np.ndarray = field(default=None)

    def __post_init__(self):
        """
        Handles the following operations/procedures:
        -   Prepare attributes project_duration and project_years,
        -   Prepare attribute expense_year,
        -   Prepare attribute cost,
        -   Prepare attribute cost_allocation,
        -   Prepare attribute description,
        -   Raise an error: expense_year is after the end year of the project,
        -   Raise an error: expense_year is before the start year of the project,
        """

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
                    f"Attribute expense_year must be given as a numpy.ndarray, "
                    f"not as a/an ({self.expense_year.__class__.__qualname__})"
                )

        # Prepare attribute cost
        if self.cost is None:
            self.cost = np.zeros_like(self.expense_year)

        else:
            if not isinstance(self.cost, np.ndarray):
                raise CostOfSalesException(
                    f"Attribute cost must be given as a numpy.ndarray, "
                    f"not as a/an ({self.cost.__class__.__qualname__})"
                )

        self.cost = self.cost.astype(np.float64)

        # Prepare attribute cost_allocation
        if self.cost_allocation is None:
            self.cost_allocation = [FluidType.OIL for _ in range(len(self.expense_year))]

        else:
            if not isinstance(self.cost_allocation, list):
                raise CostOfSalesException(
                    f"Attribute cost_allocation must be given as a list, "
                    f"not as a/an ({self.cost_allocation.__class__.__qualname__})"
                )

        # Prepare attribute description
        if self.description is None:
            self.description = [" " for _ in range(len(self.expense_year))]

        else:
            if not isinstance(self.description, list):
                raise CostOfSalesException(
                    f"Attribute description must be given as a list, "
                    f"not as a/an {self.description.__class__.__qualname__}"
                )

        # Check input data for unequal length
        arr_length = len(self.expense_year)

        if not all(
            len(arr) == arr_length
            for arr in [
                self.cost,
                self.cost_allocation,
                self.description,
            ]
        ):
            raise CostOfSalesException(
                f"Unequal length of arrays: "
                f"expense_year: {len(self.expense_year)}, "
                f"cost: {len(self.cost)}, "
                f"cost_allocation: {len(self.cost_allocation)}, "
                f"description: {len(self.description)}"
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

    def __eq__(self, other):
        # Between two instances of CostOfSales
        if isinstance(other, CostOfSales):
            return all(
                (
                    np.allclose(self.expense_year, other.expense_year),
                    np.allclose(self.cost, other.cost),
                    self.cost_allocation == other.cost_allocation,
                    self.description == other.description,
                )
            )

        # Between an instance of CosrOfSales and an integer/float
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
                f"Must compare an instance of CostOfSales with another instance of "
                f"CostOfSales, an integer, or a float"
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
                f"Must compare an instance of CostOfSales with another instance of "
                f"CostOfSales, an integer, or a float"
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
                f"Must compare an instance of CostOfSales with another instance of "
                f"CostOfSales, an integer, or a float"
            )

    def __ge__(self, other):
        if isinstance(other, CostOfSales):
            return np.sum(self.cost) >= np.sum(other.cost)

        elif isinstance(other, (int, float)):
            return np.sum(self.cost) >= other

        else:
            raise CostOfSalesException(
                f"Must compare an instance of CostOfSales with another instance of "
                f"CostOfSales, an integer, or a float"
            )

    def __add__(self, other):
        # Only allows addition between an instance of CostOfSales and another instance of CostOfSales
        if isinstance(other, CostOfSales):
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            expense_year_combined = np.concatenate((self.expense_year, other.expense_year))
            cost_combined = np.concatenate((self.cost, other.cost))
            cost_allocation_combined = self.cost_allocation + other.cost_allocation
            description_combined = self.description + other.description

            return CostOfSales(
                start_year=start_year_combined,
                end_year=end_year_combined,
                expense_year=expense_year_combined,
                cost=cost_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
            )

        else:
            raise CostOfSalesException(
                f"Must add between an instance of CostOfSales with another instance of "
                f"CostOfSales. {other}({other.__class__.__qualname__}) is not an instance of "
                f"CostOfSales."
            )

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        # Only allows subtraction between an instance of CostOfSales and another instance of CostOfSales
        if isinstance(other, CostOfSales):
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            expense_year_combined = np.concatenate((self.expense_year, other.expense_year))
            cost_combined = np.concatenate((self.cost, -other.cost))
            cost_allocation_combined = self.cost_allocation + other.cost_allocation
            description_combined = self.description + other.description

            return CostOfSales(
                start_year=start_year_combined,
                end_year=end_year_combined,
                expense_year=expense_year_combined,
                cost=cost_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
            )

        else:
            raise CostOfSalesException(
                f"Must subtract between an instance of CostOfSales with another instance of "
                f"CostOfSales. {other}({other.__class__.__qualname__}) is not an instance of "
                f"CostOfSales. "
            )

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        # Multiplication is allowed only with an integer/a float
        if isinstance(other, (int, float)):
            return CostOfSales(
                start_year=self.start_year,
                end_year=self.end_year,
                expense_year=self.expense_year,
                cost=self.cost * other,
                cost_allocation=self.cost_allocation,
                description=self.description,
            )

        else:
            raise CostOfSalesException(
                f"Must multiply with an integer or a float. {other}({other.__class__.__qualname__}) "
                f"is not an integer nor a float."
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        # Division is allowed only with an integer or a float
        if isinstance(other, (int, float)):
            # Cannot divide with zero
            if other == 0:
                raise CostOfSalesException(f"Cannot divide with zero.")

            else:
                return CostOfSales(
                    start_year=self.start_year,
                    end_year=self.end_year,
                    expense_year=self.expense_year,
                    cost=self.cost / other,
                    cost_allocation=self.cost_allocation,
                    description=self.description,
                )

        else:
            raise CostOfSalesException(
                f"Must divide with an integer or a float, {other}({other.__class__.__qualname__}) "
                f"is not an integer nor a float."
            )
