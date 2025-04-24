"""
Prepare and classify cost data based on its components.
The associated cost components are:
(1) Capital (also known as Tangible),
(2) Intangible,
(3) OPEX,
(4) ASR,
(5) LBT,
(6) CostOfSales.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

import pyscnomics.econ.depreciation as depr
from pyscnomics.econ.selection import (
    FluidType,
    DeprMethod,
    SunkCostEndPoint,
)
from pyscnomics.econ.costs_tools import (
    get_cost_adjustment_by_inflation,
    calc_indirect_tax,
    calc_distributed_cost,
)


class GeneralCostException(Exception):
    """ Exception to be raised for an incorrect use of class GeneralCost """

    pass


class CapitalException(Exception):
    """ Exception to be raised for an incorrect use of class CapitalCost """

    pass


class IntangibleException(Exception):
    """ Exception to be raised for an incorrect use of class Intangible """

    pass


class OPEXException(Exception):
    """ Exception to be raised for an incorrect use of class OPEX """

    pass


class ASRException(Exception):
    """ Exception to be raised for an incorrect use of class ASR """

    pass


class LBTException(Exception):
    """ Exception to be raised for an incorrect use of class LBT """

    pass


class CostOfSalesException(Exception):
    """ Exception to be raised for an incorrect use of class CostOfSales """

    pass


class SunkCostException(Exception):
    """ Exception to be raised for an incorrect use of class SunkCost """

    pass


class PreOnstreamCostException(Exception):
    """ Exception to be raised for an incorrect use of class PreOnstreamCost """

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
    tax_portion : np.ndarray, optional
        A NumPy array representing the portion of the cost subject to tax. If not provided,
        an array of zeros with the same shape as the project cost will be used.
    tax_discount : float | np.ndarray, optional
        A discount factor applied to the tax, reducing the overall tax impact. The default is 0.0.
    """

    start_year: int
    end_year: int
    expense_year: np.ndarray
    cost: np.ndarray
    cost_allocation: list[FluidType] = field(default=None)
    description: list[str] = field(default=None)
    tax_portion: np.ndarray = field(default=None)
    tax_discount: np.ndarray | float | int = field(default=0.0)

    # Attributes to be defined in the __post_init__
    project_duration: int = field(default=None, init=False)
    project_years: np.ndarray = field(default=None, init=False)

    def expenditures_pre_tax(
        self,
        year_inflation: np.ndarray = None,
        inflation_rate: np.ndarray | int | float = 0.0,
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

        # Allocate adjusted costs by their associated expense year
        expenses = np.bincount(
            self.expense_year - self.start_year, weights=cost_adjusted_by_inflation
        )
        zeros = np.zeros(self.project_duration - len(expenses))

        return np.concatenate((expenses, zeros))

    def indirect_taxes(
        self,
        tax_rate: np.ndarray | float = 0.0
    ) -> np.ndarray:
        """
        Calculate and allocate indirect taxes on project costs.

        This function calculates the indirect tax by applying the specified tax portion,
        tax rates, and tax discount to the project costs. The calculated tax is then allocated
        by the associated expense year.

        Parameters
        ----------
        tax_rate : np.ndarray or float, optional
            A NumPy array or float representing the tax rate applied to the costs. If not provided,
            a default rate of 0.0 will be used. When provided as an array, it should match the
            length of the project years.

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
            tax_portion=self.tax_portion,
            tax_rate=tax_rate,
            tax_discount=self.tax_discount,
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
        inflation_rate: np.ndarray | int | float = 0.0,
        tax_rate: np.ndarray | float = 0.0,
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
            The inflation rate(s) to apply. If a single float is provided, it is applied
            uniformly across all years. If an array is provided, each inflation rate
            corresponds to a specific project year (default is 0.0).
        tax_rate : np.ndarray or float, optional
            The tax rate to apply to the costs. If a float is provided, it applies uniformly
            across all project years. If a NumPy array is provided, the rate can vary by year
            (default is 0.0).

        Returns
        -------
        np.ndarray
            An array representing the post-tax expenditures for each project year, adjusted
            for inflation and indirect taxes.

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
            self.expenditures_pre_tax(year_inflation=year_inflation, inflation_rate=inflation_rate)
            + self.indirect_taxes(tax_rate=tax_rate)
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
        -   Prepare attribute expense_year,
        -   Prepare attribute cost,
        -   Prepare attribute description,
        -   Prepare attribute cost_allocation,
        -   Prepare attribute pis_year,
        -   Prepare attribute salvage_value,
        -   Prepare attribute useful_life,
        -   Prepare attribute depreciation_factor,
        -   Prepare attribute is_ic_applied,
        -   Prepare attribute tax_portion,
        -   Prepare attribute tax_discount,
        -   Check for unequal length of arrays data,
        -   Raise an exception: salvage_value is larger than the associated cost,
        -   Raise an error: expense_year is after the end year of the project,
        -   Raise an error: expense_year is before the start year of the project,
        """

        # Prepare attributes project_duration and project_years
        if self.end_year >= self.start_year:
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise CapitalException(
                f"start year {self.start_year} is after the end year {self.end_year}"
            )

        # Prepare attribute expense_year
        if not isinstance(self.expense_year, np.ndarray):
            raise CapitalException(
                f"Attribute expense_year must be provided as a numpy.ndarray, "
                f"not as a/an {self.expense_year.__class__.__qualname__}"
            )

        else:
            expense_year_nan_sum = np.sum(pd.isna(self.expense_year), dtype=np.float64)
            if expense_year_nan_sum > 0:
                raise CapitalException(
                    f"Missing values in array expense_year: {self.expense_year}"
                )

        self.expense_year = self.expense_year.astype(int)

        # Prepare attribute cost
        if not isinstance(self.cost, np.ndarray):
            raise CapitalException(
                f"Attribute cost must be provided as a numpy.ndarray, "
                f"not as a/an {self.cost.__class__.__qualname__}"
            )

        else:
            cost_nan_id = np.argwhere(pd.isna(self.cost)).ravel()
            if len(cost_nan_id) > 0:
                self.cost[cost_nan_id] = np.zeros(len(cost_nan_id))

        self.cost = self.cost.astype(np.float64)

        # Prepare attribute description
        if self.description is None:
            self.description = [" " for _ in range(len(self.cost))]

        else:
            if not isinstance(self.description, list):
                raise CapitalException(
                    f"Attribute description must be given as a list, "
                    f"not as a/an {self.description.__class__.__qualname__}"
                )

            self.description = [
                " " if pd.isna(val) else val for _, val in enumerate(self.description)
            ]

        # Prepare attribute cost_allocation
        if self.cost_allocation is None:
            self.cost_allocation = [FluidType.OIL for _ in range(len(self.cost))]

        else:
            if not isinstance(self.cost_allocation, list):
                raise CapitalException(
                    f"Attribute cost_allocation must be given as a list, "
                    f"not as a/an {self.cost_allocation.__class__.__qualname__}"
                )

            self.cost_allocation = [
                FluidType.OIL if pd.isna(val) else val for _, val in enumerate(self.cost_allocation)
            ]

        # Prepare attribute pis_year
        if self.pis_year is None:
            self.pis_year = self.expense_year.copy()

        else:
            if not isinstance(self.pis_year, np.ndarray):
                raise CapitalException(
                    f"Attribute pis_year must be provided as a numpy.ndarray, "
                    f"not as a/an {self.pis_year.__class__.__qualname__}"
                )

            pis_year_nan_id = np.argwhere(pd.isna(self.pis_year)).ravel()
            if len(pis_year_nan_id) > 0:
                self.pis_year[pis_year_nan_id] = self.expense_year[pis_year_nan_id].copy()

        self.pis_year = self.pis_year.astype(int)

        # Prepare attribute salvage_value
        if self.salvage_value is None:
            self.salvage_value = np.zeros_like(self.cost)

        else:
            if not isinstance(self.salvage_value, np.ndarray):
                raise CapitalException(
                    f"Attribute salvage_value must be provided as a numpy.ndarray, "
                    f"not as a/an {self.salvage_value.__class__.__qualname__}"
                )

            salvage_value_nan_id = np.argwhere(pd.isna(self.salvage_value)).ravel()
            if len(salvage_value_nan_id) > 0:
                self.salvage_value[salvage_value_nan_id] = np.zeros(len(salvage_value_nan_id))

        self.salvage_value = self.salvage_value.astype(np.float64)

        # Prepare attribute useful_life
        if self.useful_life is None:
            self.useful_life = np.repeat(5.0, len(self.cost))

        else:
            if not isinstance(self.useful_life, np.ndarray):
                raise CapitalException(
                    f"Attribute useful_life must be provided as a numpy.ndarray, "
                    f"not as a/an {self.useful_life.__class__.__qualname__}"
                )

            useful_life_nan_id = np.argwhere(pd.isna(self.useful_life)).ravel()
            if len(useful_life_nan_id) > 0:
                self.useful_life[useful_life_nan_id] = np.repeat(5.0, len(useful_life_nan_id))

        self.useful_life = self.useful_life.astype(np.float64)

        # Prepare attribute depreciation_factor
        if self.depreciation_factor is None:
            self.depreciation_factor = np.repeat(0.5, len(self.cost))

        else:
            if not isinstance(self.depreciation_factor, np.ndarray):
                raise CapitalException(
                    f"Attribute depreciation_factor must be provided as a numpy.ndarray, "
                    f"not as a/an {self.depreciation_factor.__class__.__qualname__}"
                )

            depreciation_factor_nan_id = np.argwhere(pd.isna(self.depreciation_factor)).ravel()
            if len(depreciation_factor_nan_id) > 0:
                self.depreciation_factor[depreciation_factor_nan_id] = (
                    np.repeat(0.5, len(depreciation_factor_nan_id))
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

            self.is_ic_applied = [
                False if pd.isna(val) else val for _, val in enumerate(self.is_ic_applied)
            ]

        # Prepare attribute tax_portion
        if self.tax_portion is None:
            self.tax_portion = np.zeros_like(self.cost)

        else:
            if not isinstance(self.tax_portion, np.ndarray):
                raise CapitalException(
                    f"Attribute tax_portion must be given as a numpy.ndarray, "
                    f"not as a/an {self.tax_portion.__class__.__qualname__}"
                )

            tax_portion_nan_id = np.argwhere(pd.isna(self.tax_portion)).ravel()
            if len(tax_portion_nan_id) > 0:
                self.tax_portion[tax_portion_nan_id] = np.zeros(len(tax_portion_nan_id))

            tax_portion_large = np.sum(self.tax_portion > 1.0, dtype=np.float64)
            tax_portion_negative = np.sum(self.tax_portion < 0.0, dtype=np.float64)
            if tax_portion_large > 0 or tax_portion_negative > 0:
                raise CapitalException(
                    f"The value of tax_portion must be: 0 < tax_portion < 1, "
                    f"tax_portion: {self.tax_portion}"
                )

        self.tax_portion = self.tax_portion.astype(np.float64)

        # Prepare attribute tax_discount
        if not isinstance(self.tax_discount, (float, int, np.ndarray)):
            raise CapitalException(
                f"Attribute tax_discount must be provided as a float/int or as a numpy.ndarray, "
                f"not as a/an {self.tax_discount.__class__.__qualname__}"
            )

        if isinstance(self.tax_discount, (float, int)):
            if self.tax_discount < 0 or self.tax_discount > 1:
                raise CapitalException(f"Attribute tax_discount must be between 0 and 1")

            self.tax_discount = np.repeat(self.tax_discount, len(self.cost))

        elif isinstance(self.tax_discount, np.ndarray):
            tax_discount_nan_id = np.argwhere(pd.isna(self.tax_discount)).ravel()
            if len(tax_discount_nan_id) > 0:
                self.tax_discount[tax_discount_nan_id] = np.zeros(len(tax_discount_nan_id))

            tax_discount_large = np.sum(self.tax_discount > 1.0, dtype=np.float64)
            tax_discount_negative = np.sum(self.tax_discount < 0.0, dtype=np.float64)
            if tax_discount_large > 0 or tax_discount_negative > 0:
                raise CapitalException(
                    f"The value of tax_discount must be: 0 < tax_discount < 1, "
                    f"tax_discount: {self.tax_discount}"
                )

        self.tax_discount = self.tax_discount.astype(np.float64)

        # Check input data for unequal length
        arr_reference = len(self.cost)

        if not all(
            len(arr) == arr_reference
            for arr in [
                self.expense_year,
                self.cost_allocation,
                self.description,
                self.tax_portion,
                self.tax_discount,
                self.pis_year,
                self.salvage_value,
                self.useful_life,
                self.depreciation_factor,
                self.is_ic_applied,
            ]
        ):
            raise CapitalException(
                f"Unequal length of arrays: "
                f"cost: {len(self.cost)}, "
                f"expense_year: {len(self.expense_year)}, "
                f"cost_allocation: {len(self.cost_allocation)}, "
                f"description: {len(self.description)}, "
                f"tax_portion: {len(self.tax_portion)}, "
                f"tax_discount: {len(self.tax_discount)}, "
                f"pis_year: {len(self.pis_year)}, "
                f"salvage_value: {len(self.salvage_value)}, "
                f"useful_life: {len(self.useful_life)}, "
                f"depreciation_factor: {len(self.depreciation_factor)}, "
                f"is_ic_applied: {len(self.is_ic_applied)} "
            )

        # Raise an exception: salvage_value is larger than the associated cost
        if np.sum(self.salvage_value > self.cost) > 0:
            raise CapitalException(
                f"Attribute salvage_value ({self.salvage_value}) is larger "
                f"than the associated cost ({self.cost})."
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
        inflation_rate: np.ndarray | int | float = 0.0,
        tax_rate: np.ndarray | float = 0.0,
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
        tax_rate : np.ndarray or float, optional
            A NumPy array or float representing the tax rate applied to the costs. If not provided,
            a default rate of 0.0 will be used. When provided as an array, it should match
            the project years.

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

        # Cost adjustment due to inflation and tax
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
                tax_portion=self.tax_portion,
                tax_rate=tax_rate,
                tax_discount=self.tax_discount,
            )
        )

        # Calculate depreciation
        # Depreciation method is straight line
        if depr_method == DeprMethod.SL:
            depreciation_charge = np.array(
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

        # Depreciation method is declining balance
        elif depr_method == DeprMethod.DB:
            depreciation_charge = np.array(
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

        # Depreciation method is PSC declining balance
        elif depr_method == DeprMethod.PSC_DB:
            depreciation_charge = np.array(
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
                f"Depreciation method ({depr_method}) is not recognized"
            )

        # The relative difference of pis_year and start_year
        shift_indices = self.pis_year - self.start_year

        # Modify depreciation_charge so that expenditures are aligned with
        # the corresponding pis_year (or expense_year)
        depreciation_charge = np.array(
            [
                np.concatenate((np.zeros(i), row[:-i])) if i > 0 else row
                for row, i in zip(depreciation_charge, shift_indices)
            ]
        )

        # Calculate undepreciated asset
        total_depreciation_charge = depreciation_charge.sum(axis=0)
        undepreciated_asset = np.sum(cost_adjusted) - np.sum(total_depreciation_charge)

        return total_depreciation_charge, undepreciated_asset

    def total_depreciation_book_value(
        self,
        depr_method: DeprMethod = DeprMethod.PSC_DB,
        decline_factor: float | int = 2,
        year_inflation: np.ndarray = None,
        inflation_rate: np.ndarray | int | float = 0.0,
        tax_rate: np.ndarray | float = 0.0,
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
        tax_rate : np.ndarray or float, optional
            A NumPy array or float representing the tax rate applied to the costs. If not provided,
            a default rate of 0.0 will be used. When provided as an array, it should match
            the project years.

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

        # Calculate total depreciation charge from method total_depreciation_rate()
        total_depreciation_charge = self.total_depreciation_rate(
            depr_method=depr_method,
            decline_factor=decline_factor,
            year_inflation=year_inflation,
            inflation_rate=inflation_rate,
            tax_rate=tax_rate,
        )[0]

        # Calculate total depreciation book value
        return (
            np.cumsum(self.expenditures_post_tax(
                year_inflation=year_inflation,
                inflation_rate=inflation_rate,
                tax_rate=tax_rate,
            ))
            - np.cumsum(total_depreciation_charge)
        )

    def __eq__(self, other):
        # Between two instances of CapitalCost
        if isinstance(other, CapitalCost):
            return all(
                (
                    self.start_year == other.start_year,
                    self.end_year == other.end_year,
                    np.allclose(self.expense_year, other.expense_year),
                    np.allclose(self.cost, other.cost),
                    np.allclose(self.pis_year, other.pis_year),
                    np.allclose(self.salvage_value, other.salvage_value),
                    np.allclose(self.useful_life, other.useful_life),
                    np.allclose(self.depreciation_factor, other.depreciation_factor),
                    np.allclose(self.tax_portion, other.tax_portion),
                    np.allclose(self.tax_discount, other.tax_discount),
                    self.cost_allocation == other.cost_allocation,
                    self.is_ic_applied == other.is_ic_applied,
                )
            )

        # Between an instance of CapitalCost and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) == other

        else:
            return False

    def __lt__(self, other):
        # Between an instance of CapitalCost with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT
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
        # Between an instance of CapitalCost with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT
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
        # Between an instance of CapitalCost with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT
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
        # Between an instance of CapitalCost with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT
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
        # Only allows addition between an instance of CapitalCost
        # and another instance of CapitalCost
        if isinstance(other, CapitalCost):
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            expense_year_combined = np.concatenate((self.expense_year, other.expense_year))
            cost_combined = np.concatenate((self.cost, other.cost))
            pis_year_combined = np.concatenate((self.pis_year, other.pis_year))
            salvage_value_combined = np.concatenate((self.salvage_value, other.salvage_value))
            useful_life_combined = np.concatenate((self.useful_life, other.useful_life))
            depreciation_factor_combined = np.concatenate(
                (self.depreciation_factor, other.depreciation_factor)
            )
            tax_portion_combined = np.concatenate((self.tax_portion, other.tax_portion))
            tax_discount_combined = np.concatenate((self.tax_discount, other.tax_discount))
            cost_allocation_combined = self.cost_allocation + other.cost_allocation
            description_combined = self.description + other.description
            is_ic_applied_combined = self.is_ic_applied + other.is_ic_applied

            return CapitalCost(
                start_year=start_year_combined,
                end_year=end_year_combined,
                expense_year=expense_year_combined,
                cost=cost_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
                tax_portion=tax_portion_combined,
                tax_discount=tax_discount_combined,
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
        # Only allows subtraction between an instance of CapitalCost
        # and another instance of CapitalCost
        if isinstance(other, CapitalCost):
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            expense_year_combined = np.concatenate((self.expense_year, other.expense_year))
            cost_combined = np.concatenate((self.cost, -other.cost))
            pis_year_combined = np.concatenate((self.pis_year, other.pis_year))
            salvage_value_combined = np.concatenate((self.salvage_value, other.salvage_value))
            useful_life_combined = np.concatenate((self.useful_life, other.useful_life))
            depreciation_factor_combined = np.concatenate(
                (self.depreciation_factor, other.depreciation_factor)
            )
            tax_portion_combined = np.concatenate((self.tax_portion, other.tax_portion))
            tax_discount_combined = np.concatenate((self.tax_discount, other.tax_discount))
            cost_allocation_combined = self.cost_allocation + other.cost_allocation
            description_combined = self.description + other.description
            is_is_applied_combined = self.is_ic_applied + other.is_ic_applied

            return CapitalCost(
                start_year=start_year_combined,
                end_year=end_year_combined,
                expense_year=expense_year_combined,
                cost=cost_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
                tax_portion=tax_portion_combined,
                tax_discount=tax_discount_combined,
                pis_year=pis_year_combined,
                salvage_value=salvage_value_combined,
                useful_life=useful_life_combined,
                depreciation_factor=depreciation_factor_combined,
                is_ic_applied=is_is_applied_combined,
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
                expense_year=self.expense_year,
                cost=self.cost * other,
                cost_allocation=self.cost_allocation,
                description=self.description,
                tax_portion=self.tax_portion,
                tax_discount=self.tax_discount,
                pis_year=self.pis_year,
                salvage_value=self.salvage_value,
                useful_life=self.useful_life,
                depreciation_factor=self.depreciation_factor,
                is_ic_applied=self.is_ic_applied,
            )

        else:
            raise CapitalException(
                f"Must multiply with an integer or a float. "
                f"{other}({other.__class__.__qualname__}) "
                f"is not an integer nor a float."
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        # Between an instance of CapitalCost with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) / np.sum(other.cost)

        # Between an instance of CapitalCost and an integer/a float
        elif isinstance(other, (int, float)):
            # Cannot divide with zero
            if other == 0:
                raise CapitalException(f"Cannot divide with zero")

            else:
                return CapitalCost(
                    start_year=self.start_year,
                    end_year=self.end_year,
                    expense_year=self.expense_year,
                    cost=self.cost / other,
                    cost_allocation=self.cost_allocation,
                    description=self.description,
                    tax_portion=self.tax_portion,
                    tax_discount=self.tax_discount,
                    pis_year=self.pis_year,
                    salvage_value=self.salvage_value,
                    useful_life=self.useful_life,
                    depreciation_factor=self.depreciation_factor,
                    is_ic_applied=self.is_ic_applied,
                )

        else:
            raise CapitalException(
                f"Must divide with an instance of CapitalCost/Intangible/OPEX/ASR/LBT, "
                f"integer or a float. {other}({other.__class__.__qualname__}) is not "
                f"an instance of CapitalCost/Intangible/OPEX/ASR/LBT not an integer "
                f"nor a float."
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
        -   Prepare attribute expense_year,
        -   Prepare attribute cost,
        -   Prepare attribute description,
        -   Prepare attribute cost_allocation,
        -   Prepare attribute tax_portion,
        -   Prepare attribute tax_discount,
        -   Check for unequal length of input arrays,
        -   Raise an error: expense_year is after the end year of the project,
        -   Raise an error: expense_year is before the start year of the project,
        """
        # Prepare attributes project_duration and project_years
        if self.end_year >= self.start_year:
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise IntangibleException(
                f"start year {self.start_year} is after the end year {self.end_year}"
            )

        # Prepare attribute expense_year
        if not isinstance(self.expense_year, np.ndarray):
            raise IntangibleException(
                f"Attribute expense_year must be provided as a numpy.ndarray, "
                f"not as a/an {self.expense_year.__class__.__qualname__}"
            )

        else:
            expense_year_nan = np.sum(pd.isna(self.expense_year), dtype=np.float64)
            if expense_year_nan > 0:
                raise IntangibleException(
                    f"Missing values in array expense_year: {self.expense_year}"
                )

        self.expense_year = self.expense_year.astype(int)

        # Prepare attribute cost
        if not isinstance(self.cost, np.ndarray):
            raise IntangibleException(
                f"Attribute cost must be provided as a numpy.ndarray, "
                f"not as a/an {self.cost.__class__.__qualname__}"
            )

        else:
            cost_nan_id = np.argwhere(pd.isna(self.cost)).ravel()
            if len(cost_nan_id) > 0:
                self.cost[cost_nan_id] = np.zeros(len(cost_nan_id))

        self.cost = self.cost.astype(np.float64)

        # Prepare attribute cost_allocation
        if self.cost_allocation is None:
            self.cost_allocation = [FluidType.OIL for _ in range(len(self.cost))]

        else:
            if not isinstance(self.cost_allocation, list):
                raise IntangibleException(
                    f"Attribute cost_allocation must be given as a list, "
                    f"not as a/an {self.cost_allocation.__class__.__qualname__}"
                )

            self.cost_allocation = [
                FluidType.OIL if pd.isna(val) else val for _, val in enumerate(self.cost_allocation)
            ]

        # Prepare attribute description
        if self.description is None:
            self.description = [" " for _ in range(len(self.cost))]

        else:
            if not isinstance(self.description, list):
                raise IntangibleException(
                    f"Attribute description must be given as a list, "
                    f"not as a/an {self.description.__class__.__qualname__}"
                )

            self.description = [
                " " if pd.isna(val) else val for _, val in enumerate(self.description)
            ]

        # Prepare attribute tax_portion
        if self.tax_portion is None:
            self.tax_portion = np.zeros_like(self.cost)

        else:
            if not isinstance(self.tax_portion, np.ndarray):
                raise IntangibleException(
                    f"Attribute tax_portion must be given as a numpy.ndarray, "
                    f"not as a/an {self.tax_portion.__class__.__qualname__}"
                )

            tax_portion_nan_id = np.argwhere(pd.isna(self.tax_portion)).ravel()
            if len(tax_portion_nan_id) > 0:
                self.tax_portion[tax_portion_nan_id] = np.zeros(len(tax_portion_nan_id))

            tax_portion_large = np.sum(self.tax_portion > 1.0, dtype=np.float64)
            tax_portion_negative = np.sum(self.tax_portion < 0.0, dtype=np.float64)
            if tax_portion_large > 0 or tax_portion_negative > 0:
                raise IntangibleException(
                    f"The value of tax_portion must be: 0 < tax_portion < 1, "
                    f"tax_portion: {self.tax_portion}"
                )

        self.tax_portion = self.tax_portion.astype(np.float64)

        # Prepare attribute tax_discount
        if not isinstance(self.tax_discount, (float, int, np.ndarray)):
            raise IntangibleException(
                f"Attribute tax_discount must be provided as a float/int or as a numpy.ndarray, "
                f"not as a/an {self.tax_discount.__class__.__qualname__}"
            )

        if isinstance(self.tax_discount, (float, int)):
            if self.tax_discount < 0 or self.tax_discount > 1:
                raise IntangibleException(f"Attribute tax_discount must be between 0 and 1")

            self.tax_discount = np.repeat(self.tax_discount, len(self.cost))

        elif isinstance(self.tax_discount, np.ndarray):
            tax_discount_nan_id = np.argwhere(pd.isna(self.tax_discount)).ravel()
            if len(tax_discount_nan_id) > 0:
                self.tax_discount[tax_discount_nan_id] = np.zeros(len(tax_discount_nan_id))

            tax_discount_large = np.sum(self.tax_discount > 1.0, dtype=np.float64)
            tax_discount_negative = np.sum(self.tax_discount < 0.0, dtype=np.float64)
            if tax_discount_large > 0 or tax_discount_negative > 0:
                raise IntangibleException(
                    f"The value of tax_discount must be: 0 < tax_discount < 1, "
                    f"tax_discount: {self.tax_discount}"
                )

        self.tax_discount = self.tax_discount.astype(np.float64)

        # Check input data for unequal length
        arr_reference = len(self.cost)

        if not all(
            len(arr) == arr_reference
            for arr in [
                self.expense_year,
                self.cost_allocation,
                self.description,
                self.tax_portion,
                self.tax_discount,
            ]
        ):
            raise IntangibleException(
                f"Unequal length of arrays: "
                f"cost: {len(self.cost)}, "
                f"expense_year: {len(self.expense_year)}, "
                f"cost_allocation: {len(self.cost_allocation)}, "
                f"description: {len(self.description)}, "
                f"tax_portion: {len(self.tax_portion)}, "
                f"tax_discount: {len(self.tax_discount)}, "
            )

        # Raise an error: expense year is after the end year of the project
        if np.max(self.expense_year) > self.end_year:
            raise IntangibleException(
                f"Expense year ({np.max(self.expense_year)}) "
                f"is after the end year of the project ({self.end_year})"
            )

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
                    self.start_year == other.end_year,
                    self.end_year == other.end_year,
                    np.allclose(self.expense_year, other.expense_year),
                    np.allclose(self.cost, other.cost),
                    np.allclose(self.tax_portion, other.tax_portion),
                    np.allclose(self.tax_discount, other.tax_discount),
                    self.cost_allocation == other.cost_allocation
                )
            )

        # Between an instance of Intangible and an integer/a float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) == other

        else:
            return False

    def __lt__(self, other):
        # Between an instance of Intangible with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) < np.sum(other.cost)

        # Between an instance of Intangible and an integer/a float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) < other

        else:
            raise IntangibleException(
                f"Must compare an instance of Intangible with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer, or a float."
            )

    def __le__(self, other):
        # Between an instance of Intangible with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) <= np.sum(other.cost)

        # Between an instance of Intangible and an integer/a float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) <= other

        else:
            raise IntangibleException(
                f"Must compare an instance of Intangible with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer, or a float."
            )

    def __gt__(self, other):
        # Between an instance of Intangible with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) > np.sum(other.cost)

        # Between an instance of Intangible and an integer/a float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) > other

        else:
            raise IntangibleException(
                f"Must compare an instance of Intangible with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer, or a float."
            )

    def __ge__(self, other):
        # Between an instance of Intangible with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) >= np.sum(other.cost)

        # Between an instance of Intangible and an integer/a float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) >= other

        else:
            raise IntangibleException(
                f"Must compare an instance of Intangible with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer, or a float."
            )

    def __add__(self, other):
        # Only allows addition between an instance of Intangible
        # and another instance of Intangible
        if isinstance(other, Intangible):
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            expense_year_combined = np.concatenate((self.expense_year, other.expense_year))
            cost_combined = np.concatenate((self.cost, other.cost))
            tax_portion_combined = np.concatenate((self.tax_portion, other.tax_portion))
            tax_discount_combined = np.concatenate((self.tax_discount, other.tax_discount))
            cost_allocation_combined = self.cost_allocation + other.cost_allocation
            description_combined = self.description + other.description

            return Intangible(
                start_year=start_year_combined,
                end_year=end_year_combined,
                expense_year=expense_year_combined,
                cost=cost_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
                tax_portion=tax_portion_combined,
                tax_discount=tax_discount_combined,
            )

        else:
            raise IntangibleException(
                f"Must add an instance of Intangible with another instance of Intangible. "
                f"{other}({other.__class__.__qualname__}) is not an instance of Intangible."
            )

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        # Only allows subtraction between an instance of Intangible
        # and another instance of Intangible
        if isinstance(other, Intangible):
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            expense_year_combined = np.concatenate((self.expense_year, other.expense_year))
            cost_combined = np.concatenate((self.cost, -other.cost))
            tax_portion_combined = np.concatenate((self.tax_portion, other.tax_portion))
            tax_discount_combined = np.concatenate((self.tax_discount, other.tax_discount))
            cost_allocation_combined = self.cost_allocation + other.cost_allocation
            description_combined = self.description + other.description

            return Intangible(
                start_year=start_year_combined,
                end_year=end_year_combined,
                expense_year=expense_year_combined,
                cost=cost_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
                tax_portion=tax_portion_combined,
                tax_discount=tax_discount_combined,
            )

        else:
            raise IntangibleException(
                f"Must subtract an instance of Intangible with another instance of Intangible. "
                f"{other}({other.__class__.__qualname__}) is not an instance of Intangible."
            )

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        # Multiplication is allowed only with an integer/a float
        if isinstance(other, (int, float)):
            return Intangible(
                start_year=self.start_year,
                end_year=self.end_year,
                expense_year=self.expense_year,
                cost=self.cost * other,
                cost_allocation=self.cost_allocation,
                description=self.description,
                tax_portion=self.tax_portion,
                tax_discount=self.tax_discount,
            )

        else:
            raise IntangibleException(
                f"Must multiply with an integer or a float. "
                f"{other}({other.__class__.__qualname__}) "
                f"is not an integer nor a float."
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        # Between an instance of Intangible with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) / np.sum(other.cost)

        # Between an instance of Intangible and an integer/a float
        elif isinstance(other, (int, float)):
            # Cannot divide by zero
            if other == 0:
                raise IntangibleException(f"Cannot divide by zero")

            else:
                return Intangible(
                    start_year=self.start_year,
                    end_year=self.end_year,
                    expense_year=self.expense_year,
                    cost=self.cost / other,
                    cost_allocation=self.cost_allocation,
                    description=self.description,
                    tax_portion=self.tax_portion,
                    tax_discount=self.tax_discount,
                )

        else:
            raise IntangibleException(
                f"Must divide with an instance of CapitalCost/Intangible/OPEX/ASR/LBT, "
                f"integer or a float. {other}({other.__class__.__qualname__}) is not an "
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
    """

    fixed_cost: np.ndarray = field(default=None)
    prod_rate: np.ndarray = field(default=None)
    cost_per_volume: np.ndarray = field(default=None)

    # Attributes to be defined later
    variable_cost: np.ndarray = field(default=None, init=False)
    cost: np.ndarray = field(default=None, init=False)

    def __post_init__(self):
        """
        Handles the following operations/procedures:
        -   Prepare attributes project_duration and project_years,
        -   Prepare attribute expense_year,
        -   Prepare attribute fixed_cost,
        -   Prepare attribute cost_allocation,
        -   Prepare attribute description,
        -   Prepare attribute tax_portion,
        -   Prepare attribute tax_discount,
        -   Prepare attribute prod_rate,
        -   Prepare attribute cost_per_volume,
        -   Check input data for unequal length of arrays,
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
                f"start year {self.start_year} is after the end year {self.end_year}"
            )

        # Prepare attribute expense_year
        if not isinstance(self.expense_year, np.ndarray):
            raise OPEXException(
                f"Attribute expense_year must be provided as a numpy.ndarray, "
                f"not as a/an {self.expense_year.__class__.__qualname__}"
            )

        else:
            expense_year_nan = np.sum(pd.isna(self.expense_year), dtype=np.float64)
            if expense_year_nan > 0:
                raise OPEXException(
                    f"Missing values in array expense_year: {self.expense_year}"
                )

        self.expense_year = self.expense_year.astype(int)

        # Prepare attribute fixed_cost
        if self.fixed_cost is None:
            self.fixed_cost = np.zeros_like(self.expense_year)

        else:
            if not isinstance(self.fixed_cost, np.ndarray):
                raise OPEXException(
                    f"Attribute fixed_cost must be provided as a numpy.ndarray, "
                    f"not as a/an {self.fixed_cost.__class__.__qualname__}"
                )

            else:
                fixed_cost_nan_id = np.argwhere(pd.isna(self.fixed_cost)).ravel()
                if len(fixed_cost_nan_id) > 0:
                    self.fixed_cost[fixed_cost_nan_id] = np.zeros(len(fixed_cost_nan_id))

        self.fixed_cost = self.fixed_cost.astype(np.float64)

        # Prepare attribute cost_allocation
        if self.cost_allocation is None:
            self.cost_allocation = [FluidType.OIL for _ in range(len(self.expense_year))]

        else:
            if not isinstance(self.cost_allocation, list):
                raise OPEXException(
                    f"Attribute cost_allocation must be given as a list, "
                    f"not as a/an {self.cost_allocation.__class__.__qualname__}"
                )

            self.cost_allocation = [
                FluidType.OIL if pd.isna(val) else val for _, val in enumerate(self.cost_allocation)
            ]

        # Prepare attribute description
        if self.description is None:
            self.description = [" " for _ in range(len(self.expense_year))]

        else:
            if not isinstance(self.description, list):
                raise OPEXException(
                    f"Attribute description must be given as a list, "
                    f"not as a/an {self.description.__class__.__qualname__}"
                )

            self.description = [
                " " if pd.isna(val) else val for _, val in enumerate(self.description)
            ]

        # Prepare attribute tax_portion
        if self.tax_portion is None:
            self.tax_portion = np.zeros_like(self.expense_year)

        else:
            if not isinstance(self.tax_portion, np.ndarray):
                raise OPEXException(
                    f"Attribute tax_portion must be given as a numpy.ndarray, "
                    f"not as a/an {self.tax_portion.__class__.__qualname__}"
                )

            tax_portion_nan_id = np.argwhere(pd.isna(self.tax_portion)).ravel()
            if len(tax_portion_nan_id) > 0:
                self.tax_portion[tax_portion_nan_id] = np.zeros(len(tax_portion_nan_id))

            tax_portion_large = np.sum(self.tax_portion > 1.0, dtype=np.float64)
            tax_portion_negative = np.sum(self.tax_portion < 0.0, dtype=np.float64)
            if tax_portion_large > 0 or tax_portion_negative > 0:
                raise OPEXException(
                    f"The value of tax_portion must be: 0 < tax_portion < 1, "
                    f"tax_portion: {self.tax_portion}"
                )

        self.tax_portion = self.tax_portion.astype(np.float64)

        # Prepare attribute tax_discount
        if not isinstance(self.tax_discount, (float, int, np.ndarray)):
            raise OPEXException(
                f"Attribute tax_discount must be provided as a float/int or as a numpy.ndarray, "
                f"not as a/an {self.tax_discount.__class__.__qualname__}"
            )

        if isinstance(self.tax_discount, (float, int)):
            if self.tax_discount < 0 or self.tax_discount > 1:
                raise OPEXException(f"Attribute tax_discount must be between 0 and 1")

            self.tax_discount = np.repeat(self.tax_discount, len(self.expense_year))

        elif isinstance(self.tax_discount, np.ndarray):
            tax_discount_nan_id = np.argwhere(pd.isna(self.tax_discount)).ravel()
            if len(tax_discount_nan_id) > 0:
                self.tax_discount[tax_discount_nan_id] = np.zeros(len(tax_discount_nan_id))

            tax_discount_large = np.sum(self.tax_discount > 1.0, dtype=np.float64)
            tax_discount_negative = np.sum(self.tax_discount < 0.0, dtype=np.float64)
            if tax_discount_large > 0 or tax_discount_negative > 0:
                raise OPEXException(
                    f"The value of tax_discount must be: 0 < tax_discount < 1, "
                    f"tax_discount: {self.tax_discount}"
                )

        self.tax_discount = self.tax_discount.astype(np.float64)

        # Prepare attribute prod_rate
        if self.prod_rate is None:
            self.prod_rate = np.zeros_like(self.expense_year)

        else:
            if not isinstance(self.prod_rate, np.ndarray):
                raise OPEXException(
                    f"Attribute prod_rate must be provided as a numpy.ndarray, "
                    f"not as a/an {self.prod_rate.__class__.__qualname__}"
                )

            else:
                prod_rate_nan_id = np.argwhere(pd.isna(self.prod_rate)).ravel()
                if len(prod_rate_nan_id) > 0:
                    self.prod_rate[prod_rate_nan_id] = np.zeros(len(prod_rate_nan_id))

        self.prod_rate = self.prod_rate.astype(np.float64)

        # Prepare attribute cost_per_volume
        if self.cost_per_volume is None:
            self.cost_per_volume = np.zeros_like(self.expense_year)

        else:
            if not isinstance(self.cost_per_volume, np.ndarray):
                raise OPEXException(
                    "Attribute cost_per_volume must be provided as a numpy.ndarray, "
                    f"not as a/an {self.cost_per_volume.__class__.__qualname__}"
                )

            else:
                cost_per_volume_nan_id = np.argwhere(pd.isna(self.cost_per_volume)).ravel()
                if len(cost_per_volume_nan_id) > 0:
                    self.cost_per_volume[cost_per_volume_nan_id] = (
                        np.zeros(len(cost_per_volume_nan_id))
                    )

        self.cost_per_volume = self.cost_per_volume.astype(np.float64)

        # Check input data for unequal length
        arr_reference = len(self.expense_year)

        if not all(
                len(arr) == arr_reference
                for arr in [
                    self.fixed_cost,
                    self.cost_allocation,
                    self.description,
                    self.tax_portion,
                    self.tax_discount,
                    self.prod_rate,
                    self.cost_per_volume,
                ]
        ):
            raise OPEXException(
                f"Unequal length of arrays: "
                f"expense_year: {len(self.expense_year)}, "
                f"fixed_cost: {len(self.fixed_cost)}, "
                f"cost_allocation: {len(self.cost_allocation)}, "
                f"description: {len(self.description)}, "
                f"tax_portion: {len(self.tax_portion)}, "
                f"tax_discount: {len(self.tax_discount)}, "
                f"prod_rate: {len(self.prod_rate)}, "
                f"cost_per_volume: {len(self.cost_per_volume)}, "
            )

        # Prepare attribute variable_cost
        self.variable_cost = self.prod_rate * self.cost_per_volume

        # Prepare attribute cost
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
                    self.start_year == other.start_year,
                    self.end_year == other.end_year,
                    np.allclose(self.expense_year, other.expense_year),
                    np.allclose(self.fixed_cost, other.fixed_cost),
                    np.allclose(self.tax_portion, other.tax_portion),
                    np.allclose(self.tax_discount, other.tax_discount),
                    np.allclose(self.variable_cost, other.variable_cost),
                    np.allclose(self.cost, other.cost),
                    self.cost_allocation == other.cost_allocation,
                )
            )

        # Between an instance of OPEX and an integer/a float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) == other

        else:
            return False

    def __lt__(self, other):
        # Between an instance of OPEX with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT
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
        # Between an instance of OPEX with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT
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
        # Between an instance of OPEX with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) > other.cost

        # Between an instance of OPEX and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) > other

        else:
            raise OPEXException(
                f"Must compare an instance of OPEX with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer, or a float."
            )

    def __ge__(self, other):
        # Between an instance of OPEX with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) >= other.cost

        # Between an instance of OPEX and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) >= other

        else:
            raise OPEXException(
                f"Must compare an instance of OPEX with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer, or a float."
            )

    def __add__(self, other):
        # Only allows addition between an instance of OPEX
        # and another instance of OPEX
        if isinstance(other, OPEX):
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            expense_year_combined = np.concatenate((self.expense_year, other.expense_year))
            fixed_cost_combined = np.concatenate((self.fixed_cost, other.fixed_cost))
            tax_portion_combined = np.concatenate((self.tax_portion, other.tax_portion))
            tax_discount_combined = np.concatenate((self.tax_discount, other.tax_discount))
            prod_rate_combined = np.concatenate((self.prod_rate, other.prod_rate))
            cost_per_volume_combined = np.concatenate((self.cost_per_volume, other.cost_per_volume))
            cost_allocation_combined = self.cost_allocation + other.cost_allocation
            description_combined = self.description + other.description

            return OPEX(
                start_year=start_year_combined,
                end_year=end_year_combined,
                expense_year=expense_year_combined,
                fixed_cost=fixed_cost_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
                tax_portion=tax_portion_combined,
                tax_discount=tax_discount_combined,
                prod_rate=prod_rate_combined,
                cost_per_volume=cost_per_volume_combined,
            )

        else:
            raise OPEXException(
                f"Must add an instance of OPEX with another instance of OPEX. "
                f"{other}({other.__class__.__qualname__}) is not an instance of OPEX."
            )

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        # Only allows subtraction between an instance of OPEX
        # and another instance of OPEX
        if isinstance(other, OPEX):
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            expense_year_combined = np.concatenate((self.expense_year, other.expense_year))
            fixed_cost_combined = np.concatenate((self.fixed_cost, -other.fixed_cost))
            tax_portion_combined = np.concatenate((self.tax_portion, other.tax_portion))
            tax_discount_combined = np.concatenate((self.tax_discount, other.tax_discount))
            prod_rate_combined = np.concatenate((self.prod_rate, other.prod_rate))
            cost_per_volume_combined = np.concatenate((self.cost_per_volume, -other.cost_per_volume))
            cost_allocation_combined = self.cost_allocation + other.cost_allocation
            description_combined = self.description + other.description

            return OPEX(
                start_year=start_year_combined,
                end_year=end_year_combined,
                expense_year=expense_year_combined,
                fixed_cost=fixed_cost_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
                tax_portion=tax_portion_combined,
                tax_discount=tax_discount_combined,
                prod_rate=prod_rate_combined,
                cost_per_volume=cost_per_volume_combined,
            )

        else:
            raise OPEXException(
                f"Must subtract an instance of OPEX with another instance of OPEX. "
                f"{other}({other.__class__.__qualname__}) is not an instance of OPEX."
            )

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        # Multiplication is only allowed with an integer/a float
        if isinstance(other, (int, float)):
            return OPEX(
                start_year=self.start_year,
                end_year=self.end_year,
                expense_year=self.expense_year,
                fixed_cost=self.fixed_cost * other,
                cost_allocation=self.cost_allocation,
                description=self.description,
                tax_portion=self.tax_portion,
                tax_discount=self.tax_discount,
                prod_rate=self.prod_rate,
                cost_per_volume=self.cost_per_volume * other,
            )

        else:
            raise OPEXException(
                f"Must multiply with an integer or a float. "
                f"{other}({other.__class__.__qualname__}) "
                f"is not an integer nor a float."
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        # Between an instance of OPEX with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) / np.sum(other.cost)

        # Between an instance of OPEX and an integer/a float
        elif isinstance(other, (int, float)):
            # Cannot divide by zero
            if other == 0:
                raise OPEXException(f"Cannot divide with zero")

            else:
                return OPEX(
                    start_year=self.start_year,
                    end_year=self.end_year,
                    expense_year=self.expense_year,
                    fixed_cost=self.fixed_cost / other,
                    cost_allocation=self.cost_allocation,
                    description=self.description,
                    tax_portion=self.tax_portion,
                    tax_discount=self.tax_discount,
                    prod_rate=self.prod_rate,
                    cost_per_volume=self.cost_per_volume / other,
                )

        else:
            raise OPEXException(
                f"Must divide with an instance of CapitalCost/Intangible/OPEX/ASR/LBT, "
                f"integer or a float. {other}({other.__class__.__qualname__}) is not "
                f"an instance of CapitalCost/Intangible/OPEX/ASR/LBT not an integer "
                f"nor a float."
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
        -   Prepare attribute tax_portion,
        -   Prepare attribute tax_discount,
        -   Prepare attribute future_rate,
        -   Prepare attribute final_year,
        -   Raise an error: expense_year is after the end year of the project,
        -   Raise an error: expense_year is before the start year of the project,
        """

        # Prepare attributes project_duration and project_years
        if self.end_year >= self.start_year:
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise ASRException(
                f"start year {self.start_year} is after the end year {self.end_year}"
            )

        # Prepare attribute expense_year
        if not isinstance(self.expense_year, np.ndarray):
            raise ASRException(
                f"Attribute expense_year must be provided as a numpy.ndarray, "
                f"not as a/an {self.expense_year.__class__.__qualname__}"
            )

        else:
            expense_year_nan_sum = np.sum(pd.isna(self.expense_year), dtype=np.float64)
            if expense_year_nan_sum > 0:
                raise ASRException(f"Missing values in array expense_year: {self.expense_year}")

        self.expense_year = self.expense_year.astype(int)

        # Prepare attribute cost
        if not isinstance(self.cost, np.ndarray):
            raise ASRException(
                f"Attribute cost must be provided as a numpy.ndarray, "
                f"not as a/an {self.cost.__class__.__qualname__}"
            )

        else:
            cost_nan_id = np.argwhere(pd.isna(self.cost)).ravel()
            if len(cost_nan_id) > 0:
                self.cost[cost_nan_id] = np.zeros(len(cost_nan_id))

        self.cost = self.cost.astype(np.float64)

        # Prepare attribute cost_allocation
        if self.cost_allocation is None:
            self.cost_allocation = [FluidType.OIL for _ in range(len(self.cost))]

        else:
            if not isinstance(self.cost_allocation, list):
                raise ASRException(
                    f"Attribute cost_allocation must be given as a list, "
                    f"not as a/an {self.cost_allocation.__class__.__qualname__}"
                )

            self.cost_allocation = [
                FluidType.OIL if pd.isna(val) else val for _, val in enumerate(self.cost_allocation)
            ]

        # Prepare attribute description
        if self.description is None:
            self.description = [" " for _ in range(len(self.cost))]

        else:
            if not isinstance(self.description, list):
                raise ASRException(
                    f"Attribute description must be given as a list, "
                    f"not as a/an {self.description.__class__.__qualname__}"
                )

            self.description = [
                " " if pd.isna(val) else val for _, val in enumerate(self.description)
            ]

        # Prepare attribute tax_portion
        if self.tax_portion is None:
            self.tax_portion = np.zeros_like(self.cost)

        else:
            if not isinstance(self.tax_portion, np.ndarray):
                raise ASRException(
                    f"Attribute tax_portion must be given as a numpy.ndarray, "
                    f"not as a/an {self.tax_portion.__class__.__qualname__}"
                )

            tax_portion_nan_id = np.argwhere(pd.isna(self.tax_portion)).ravel()
            if len(tax_portion_nan_id) > 0:
                self.tax_portion[tax_portion_nan_id] = np.zeros(len(tax_portion_nan_id))

            tax_portion_large = np.sum(self.tax_portion > 1.0, dtype=np.float64)
            tax_portion_negative = np.sum(self.tax_portion < 0.0, dtype=np.float64)
            if tax_portion_large > 0 or tax_portion_negative > 0:
                raise ASRException(
                    f"The value of tax_portion must be: 0 < tax_portion < 1, "
                    f"tax_portion: {self.tax_portion}"
                )

        self.tax_portion = self.tax_portion.astype(np.float64)

        # Prepare attribute tax_discount
        if not isinstance(self.tax_discount, (float, int, np.ndarray)):
            raise ASRException(
                f"Attribute tax_discount must be provided as a float/int or as a numpy.ndarray, "
                f"not as a/an {self.tax_discount.__class__.__qualname__}"
            )

        if isinstance(self.tax_discount, (float, int)):
            if self.tax_discount < 0 or self.tax_discount > 1:
                raise ASRException(f"Attribute tax_discount must be between 0 and 1")

            self.tax_discount = np.repeat(self.tax_discount, len(self.cost))

        elif isinstance(self.tax_discount, np.ndarray):
            tax_discount_nan_id = np.argwhere(pd.isna(self.tax_discount)).ravel()
            if len(tax_discount_nan_id) > 0:
                self.tax_discount[tax_discount_nan_id] = np.zeros(len(tax_discount_nan_id))

            tax_discount_large = np.sum(self.tax_discount > 1.0, dtype=np.float64)
            tax_discount_negative = np.sum(self.tax_discount < 0.0, dtype=np.float64)
            if tax_discount_large > 0 or tax_discount_negative > 0:
                raise ASRException(
                    f"The value of tax_discount must be: 0 < tax_discount < 1, "
                    f"tax_discount: {self.tax_discount}"
                )

        self.tax_discount = self.tax_discount.astype(np.float64)

        # Prepare attribute future_rate
        if self.future_rate is None:
            self.future_rate = np.zeros_like(self.cost)

        else:
            if not isinstance(self.future_rate, (float, np.ndarray)):
                raise ASRException(
                    f"Attribute future_rate must be provided as a numpy ndarray, "
                    f"not as an/a {self.future_rate.__class__.__qualname__}"
                )

            if isinstance(self.future_rate, float):
                if self.future_rate < 0.0 or self.future_rate > 1.0:
                    raise ASRException(f"Attribute future_rate must be between 0 and 1")
                else:
                    self.future_rate = np.repeat(self.future_rate, len(self.cost))

            elif isinstance(self.future_rate, np.ndarray):
                future_rate_nan_id = np.argwhere(pd.isna(self.future_rate)).ravel()
                if len(future_rate_nan_id) > 0:
                    self.future_rate[future_rate_nan_id] = np.zeros(len(future_rate_nan_id))

                future_rate_negative = np.sum(self.future_rate < 0.0)
                future_rate_large = np.sum(self.future_rate > 1.0)
                if future_rate_negative > 0 or future_rate_large > 0:
                    raise ASRException(
                        f"The value of future_rate must fall within the following "
                        f"interval: 0 < future_rate < 1 (future_rate: {self.future_rate})"
                    )

        self.future_rate = self.future_rate.astype(np.float64)

        # Prepare attribute final_year
        if self.final_year is None:
            self.final_year = self.expense_year.copy()

        else:
            if not isinstance(self.final_year, np.ndarray):
                raise ASRException(
                    f"Attribute final_year must be given as a numpy.ndarray, "
                    f"not as a/an {self.final_year.__class__.__qualname__}"
                )

            final_year_nan = np.argwhere(pd.isna(self.final_year)).ravel()
            if len(final_year_nan) > 0:
                self.final_year[final_year_nan] = self.expense_year[final_year_nan].copy()

            # Exception: final_year is before expense_year
            final_year_small_sum = np.sum(self.final_year < self.expense_year)
            if final_year_small_sum > 0:
                raise ASRException(
                    f"Attribute final_year ({self.final_year}) is before "
                    f"the expense_year ({self.expense_year})"
                )

            # Exception: final_year is after the end year of the project
            if np.max(self.final_year) > self.end_year:
                raise ASRException(
                    f"Final year ({np.max(self.final_year)}) "
                    f"is after the end year of the project ({self.end_year})"
                )

        self.final_year = self.final_year.astype(int)

        # Check input data for unequal length
        arr_reference = len(self.cost)

        if not all(
            len(arr) == arr_reference
            for arr in [
                self.expense_year,
                self.cost_allocation,
                self.description,
                self.tax_portion,
                self.tax_discount,
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
                f"future_rate: {len(self.future_rate)}, "
                f"tax_portion: {len(self.tax_portion)}, "
                f"tax_discount: {len(self.tax_discount)}. "
            )

        # Raise an exception: expense year is after the end year of the project
        if np.max(self.expense_year) > self.end_year:
            raise ASRException(
                f"Expense year ({np.max(self.expense_year)}) "
                f"is after the end year of the project ({self.end_year})"
            )

        # Raise an exception: expense year is before the start year of the project
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
        between the project's end year and the expense year, adjusting for a specified
        future rate.

        Returns
        -------
        np.ndarray
            A 1D array of future costs for each element, adjusted by the future growth rate
            over the time difference between the expense year and the project end year.

        Notes
        -----
        The future cost is calculated by multiplying the current cost by
        `(1 + future_rate) ** year_diff`, where `year_diff` is the number
        of years between the `expense_year` and the `end_year`.
        """

        # Distance between the end year of the project with expense year
        year_diff = self.end_year - self.expense_year + 1

        # Calculate future value of cost
        return self.cost * np.power((1.0 + self.future_rate), year_diff)

    def expenditures_pre_tax(
        self,
        year_inflation: np.ndarray = None,
        inflation_rate: np.ndarray | int | float = 0.0,
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
        tax_rate: np.ndarray | float = 0.0
    ) -> np.ndarray:
        """
        Calculate and distribute indirect taxes over the project duration.

        This method calculates indirect taxes based on future costs, tax portions, tax rates,
        and any applicable tax discounts. The taxes are then distributed across the project's
        timeline, and the total indirect taxes are summed for each project year.

        Parameters
        ----------
        tax_rate : np.ndarray or float, optional
            A NumPy array or float representing the tax rate applied to the costs. If not provided,
            a default rate of 0.0 will be used. When provided as an array, it should match
            the project years.

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
            tax_portion=self.tax_portion,
            tax_rate=tax_rate,
            tax_discount=self.tax_discount,
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
                    self.start_year == other.start_year,
                    self.end_year == other.end_year,
                    np.allclose(self.expense_year, other.expense_year),
                    np.allclose(self.cost, other.cost),
                    np.allclose(self.tax_portion, other.tax_portion),
                    np.allclose(self.tax_discount, other.tax_discount),
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
        # Between an instance of ASR with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) < np.sum(other.cost)

        # Between an instance of ASR and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) < other

        else:
            raise ASRException(
                f"Must compare an instance of ASR with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer or a float. "
            )

    def __le__(self, other):
        # Between an instance of ASR with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) <= np.sum(other.cost)

        # Between an instance of ASR and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) <= other

        else:
            raise ASRException(
                f"Must compare an instance of ASR with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer or a float. "
            )

    def __gt__(self, other):
        # Between an instance of ASR with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) > np.sum(other.cost)

        # Between an instance of ASR and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) > other

        else:
            raise ASRException(
                f"Must compare an instance of ASR with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer, or a float."
            )

    def __ge__(self, other):
        # Between an instance of ASR with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) >= np.sum(other.cost)

        # Between an instance of ASR and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) >= other

        else:
            raise ASRException(
                f"Must compare an instance of ASR with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer, or a float."
            )

    def __add__(self, other):
        # Only allows addition between an instance of ASR
        # and another instance of ASR
        if isinstance(other, ASR):
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            expense_year_combined = np.concatenate((self.expense_year, other.expense_year))
            cost_combined = np.concatenate((self.cost, other.cost))
            tax_portion_combined = np.concatenate((self.tax_portion, other.tax_portion))
            tax_discount_combined = np.concatenate((self.tax_discount, other.tax_discount))
            final_year_combined = np.concatenate((self.final_year, other.final_year))
            future_rate_combined = np.concatenate((self.future_rate, other.future_rate))
            cost_allocation_combined = self.cost_allocation + other.cost_allocation
            description_combined = self.description + other.description

            return ASR(
                start_year=start_year_combined,
                end_year=end_year_combined,
                expense_year=expense_year_combined,
                cost=cost_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
                tax_portion=tax_portion_combined,
                tax_discount=tax_discount_combined,
                final_year=final_year_combined,
                future_rate=future_rate_combined,
            )

        else:
            raise ASRException(
                f"Must add an instance of ASR with another instance of ASR. "
                f"{other}({other.__class__.__qualname__}) is not an instance of ASR."
            )

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        # Only allows subtraction between an instance of ASR
        # and another instance of ASR
        if isinstance(other, ASR):
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            expense_year_combined = np.concatenate((self.expense_year, other.expense_year))
            cost_combined = np.concatenate((self.cost, -other.cost))
            tax_portion_combined = np.concatenate((self.tax_portion, other.tax_portion))
            tax_discount_combined = np.concatenate((self.tax_discount, other.tax_discount))
            final_year_combined = np.concatenate((self.final_year, other.final_year))
            future_rate_combined = np.concatenate((self.future_rate, other.future_rate))
            cost_allocation_combined = self.cost_allocation + other.cost_allocation
            description_combined = self.description + other.description

            return ASR(
                start_year=start_year_combined,
                end_year=end_year_combined,
                expense_year=expense_year_combined,
                cost=cost_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
                tax_portion=tax_portion_combined,
                tax_discount=tax_discount_combined,
                final_year=final_year_combined,
                future_rate=future_rate_combined,
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
                expense_year=self.expense_year,
                cost=self.cost * other,
                cost_allocation=self.cost_allocation,
                description=self.description,
                tax_portion=self.tax_portion,
                tax_discount=self.tax_discount,
                final_year=self.final_year,
                future_rate=self.future_rate,
            )

        else:
            raise ASRException(
                f"Must multiply with an integer or a float. "
                f"{other}({other.__class__.__qualname__}) "
                f"is not an integer nor a float."
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        # Between an instance of ASR with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) / np.sum(other.cost)

        # Between an instance of ASR and an integer/a float
        elif isinstance(other, (int, float)):
            if other == 0:
                raise ASRException(f"Cannot divide with zero")

            else:
                return ASR(
                    start_year=self.start_year,
                    end_year=self.end_year,
                    expense_year=self.expense_year,
                    cost=self.cost / other,
                    cost_allocation=self.cost_allocation,
                    description=self.description,
                    tax_portion=self.tax_portion,
                    tax_discount=self.tax_discount,
                    final_year=self.final_year,
                    future_rate=self.future_rate,
                )

        else:
            raise ASRException(
                f"Must divide with an instance of CapitalCost/Intangible/OPEX/ASR/LBT, "
                f"integer or a float. {other}({other.__class__.__qualname__}) is not "
                f"an instance of CapitalCost/Intangible/OPEX/ASR/LBT not an integer "
                f"nor a float."
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

    # Override attribute
    cost: np.ndarray = field(default=None)

    # Attributes to be defined later
    # (associated with surface and subsurface LBT components)
    _surface_lbt_cost: np.ndarray = field(default=None, init=False, repr=False)
    _subsurface_lbt_cost: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """
        Handles the following operations/procedures:
        -   Prepare attributes project_duration and project_years,
        -   Prepare attribute expense_year,
        -   Prepare attribute cost_allocation,
        -   Prepare attribute description,
        -   Prepare attribute tax_portion,
        -   Prepare attribute tax_discount,
        -   Prepare attribute final_year,
        -   Prepare attribute utilized_land_area,
        -   Prepare attribute utilized_building_area,
        -   Prepare attribute njop_land,
        -   Prepare attribute njop_building,
        -   Prepare attribute gross_revenue,
        -   Prepare attribute _surface_lbt_cost,
        -   Prepare attribute _subsurface_lbt_cost,
        -   Prepare attribute cost,
        -   Check for unequal length of array data,
        -   Raise an exception: expense_year is after the end_year of the project,
        -   Raise an exception: expense_year is before the start_year of the project,
        """

        # Prepare attributes project_duration and project_years
        if self.end_year >= self.start_year:
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise LBTException(
                f"start year {self.start_year} is after the end year {self.end_year}"
            )

        # Prepare attribute expense_year
        if not isinstance(self.expense_year, np.ndarray):
            raise LBTException(
                f"Attribute expense_year must be provided as a numpy.ndarray, "
                f"not as a/an {self.expense_year.__class__.__qualname__}"
            )

        else:
            expense_year_nan_sum = np.sum(pd.isna(self.expense_year))
            if expense_year_nan_sum > 0:
                raise LBTException(
                    f"Missing values in array expense_year: {self.expense_year}"
                )

        self.expense_year = self.expense_year.astype(int)

        # Prepare attribute cost_allocation
        if self.cost_allocation is None:
            self.cost_allocation = [FluidType.OIL for _ in range(len(self.expense_year))]

        else:
            if not isinstance(self.cost_allocation, list):
                raise LBTException(
                    f"Attribute cost_allocation must be given as a list, "
                    f"not as a/an {self.cost_allocation.__class__.__qualname__}"
                )

            self.cost_allocation = [
                FluidType.OIL if pd.isna(val) else val for _, val in enumerate(self.cost_allocation)
            ]

        # Prepare attribute description
        if self.description is None:
            self.description = [" " for _ in range(len(self.expense_year))]

        else:
            if not isinstance(self.description, list):
                raise LBTException(
                    f"Attribute description must be given as a list, "
                    f"not as a/an {self.description.__class__.__qualname__}"
                )

            self.description = [
                " " if pd.isna(val) else val for _, val in enumerate(self.description)
            ]

        # Prepare attribute tax_portion
        if self.tax_portion is None:
            self.tax_portion = np.zeros_like(self.expense_year)

        else:
            if not isinstance(self.tax_portion, np.ndarray):
                raise LBTException(
                    f"Attribute tax_portion must be given as a numpy.ndarray, "
                    f"not as a/an {self.tax_portion.__class__.__qualname__}"
                )

            tax_portion_nan_id = np.argwhere(pd.isna(self.tax_portion)).ravel()
            if len(tax_portion_nan_id) > 0:
                self.tax_portion[tax_portion_nan_id] = np.zeros(len(tax_portion_nan_id))

            tax_portion_large = np.sum(self.tax_portion > 1.0, dtype=np.float64)
            tax_portion_negative = np.sum(self.tax_portion < 0.0, dtype=np.float64)
            if tax_portion_large > 0 or tax_portion_negative > 0:
                raise LBTException(
                    f"The value of tax_portion must be: 0 < tax_portion < 1, "
                    f"tax_portion: {self.tax_portion}"
                )

        self.tax_portion = self.tax_portion.astype(np.float64)

        # Prepare attribute tax_discount
        if not isinstance(self.tax_discount, (float, int, np.ndarray)):
            raise LBTException(
                f"Attribute tax_discount must be provided as a float/int or as a numpy.ndarray, "
                f"not as a/an {self.tax_discount.__class__.__qualname__}"
            )

        if isinstance(self.tax_discount, (float, int)):
            if self.tax_discount < 0 or self.tax_discount > 1:
                raise LBTException(f"Attribute tax_discount must be between 0 and 1")

            self.tax_discount = np.repeat(self.tax_discount, len(self.expense_year))

        elif isinstance(self.tax_discount, np.ndarray):
            tax_discount_nan_id = np.argwhere(pd.isna(self.tax_discount)).ravel()
            if len(tax_discount_nan_id) > 0:
                self.tax_discount[tax_discount_nan_id] = np.zeros(len(tax_discount_nan_id))

            tax_discount_large = np.sum(self.tax_discount > 1.0, dtype=np.float64)
            tax_discount_negative = np.sum(self.tax_discount < 0.0, dtype=np.float64)
            if tax_discount_large > 0 or tax_discount_negative > 0:
                raise LBTException(
                    f"The value of tax_discount must be: 0 < tax_discount < 1, "
                    f"tax_discount: {self.tax_discount}"
                )

        self.tax_discount = self.tax_discount.astype(np.float64)

        # Prepare attribute final_year
        if self.final_year is None:
            self.final_year = self.expense_year.copy()

        else:
            if not isinstance(self.final_year, np.ndarray):
                raise LBTException(
                    f"Attribute final_year must be given as a numpy.ndarray, "
                    f"not as a/an {self.final_year.__class__.__qualname__}"
                )

            final_year_nan_id = np.argwhere(pd.isna(self.final_year)).ravel()
            if len(final_year_nan_id) > 0:
                self.final_year[final_year_nan_id] = self.expense_year[final_year_nan_id].copy()

            # Exception: final_year is before expense_year
            final_year_small_sum = np.sum(self.final_year < self.expense_year)
            if final_year_small_sum > 0:
                raise LBTException(
                    f"Attribute final_year ({self.final_year}) is before "
                    f"the expense_year ({self.expense_year})"
                )

            # Exception: final_year is after the end year of the project
            if np.max(self.final_year) > self.end_year:
                raise LBTException(
                    f"Final year ({np.max(self.final_year)}) "
                    f"is after the end year of the project ({self.end_year})"
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

            # Missing data in array utilized_land_area
            utilized_land_nan_id = np.argwhere(pd.isna(self.utilized_land_area)).ravel()
            if len(utilized_land_nan_id) > 0:
                self.utilized_land_area[utilized_land_nan_id] = np.zeros(len(utilized_land_nan_id))

            # Exception: negative values in array utilized_land_area
            utilized_land_area_negative_sum = np.sum(self.utilized_land_area < 0)
            if utilized_land_area_negative_sum > 0:
                raise LBTException(
                    f"Negative values in array utilized_land_area: {self.utilized_land_area}"
                )

        self.utilized_land_area = self.utilized_land_area.astype(np.float64)

        # Prepare attribute utilized_building_area
        if self.utilized_building_area is None:
            self.utilized_building_area = np.zeros_like(self.expense_year)

        else:
            if not isinstance(self.utilized_building_area, np.ndarray):
                raise LBTException(
                    f"Attribute utilized_building_area must be given as a numpy.ndarray, "
                    f"not as an/a {self.utilized_building_area.__class__.__qualname__}"
                )

            # Missing data in array utilized_building_area
            utilized_building_nan_id = np.argwhere(pd.isna(self.utilized_building_area)).ravel()
            if len(utilized_building_nan_id) > 0:
                self.utilized_building_area[utilized_building_nan_id] = (
                    np.zeros(len(utilized_building_nan_id))
                )

            # Exception: negative values in array utilized_building_area
            utilized_building_negative_sum = np.sum(self.utilized_building_area < 0)
            if utilized_building_negative_sum > 0:
                raise LBTException(
                    f"Negative values in array utilized_building_area: {self.utilized_building_area}"
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

            # Missing data of array njop_land
            njop_land_nan_id = np.argwhere(pd.isna(self.njop_land)).ravel()
            if len(njop_land_nan_id) > 0:
                self.njop_land[njop_land_nan_id] = np.zeros(len(njop_land_nan_id))

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

            # Missing data in array njop_building
            njop_building_nan_id = np.argwhere(pd.isna(self.njop_building)).ravel()
            if len(njop_building_nan_id) > 0:
                self.njop_building[njop_building_nan_id] = np.zeros(len(njop_building_nan_id))

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

            # Missing data in array gross_revenue
            gross_revenue_nan_id = np.argwhere(pd.isna(self.gross_revenue)).ravel()
            if len(gross_revenue_nan_id) > 0:
                self.gross_revenue[gross_revenue_nan_id] = np.zeros(len(gross_revenue_nan_id))

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

            else:
                cost_nan_id = np.argwhere(pd.isna(self.cost)).ravel()
                if len(cost_nan_id) > 0:
                    self.cost[cost_nan_id] = np.zeros(len(cost_nan_id))

        self.cost = self.cost.astype(np.float64)

        # Check for unequal length of array data
        arr_reference = len(self.expense_year)

        if not all(
            len(arr) == arr_reference
            for arr in [
                self.cost_allocation,
                self.description,
                self.tax_portion,
                self.tax_discount,
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
                f"cost_allocation: {len(self.cost_allocation)}, "
                f"description: {len(self.description)}, "
                f"tax_portion: {len(self.tax_portion)}, "
                f"tax_discount: {len(self.tax_discount)}, "
                f"final_year: {len(self.final_year)}, "
                f"utilized_land_area: {len(self.utilized_land_area)}, "
                f"utilized_building_area: {len(self.utilized_building_area)}, "
                f"njop_land: {len(self.njop_land)}, "
                f"njop_building: {len(self.njop_building)}, "
                f"gross_revenue: {len(self.gross_revenue)}, "
                f"cost: {len(self.cost)}"
            )

        # Exception: expense year is after the end year of the project
        if np.max(self.expense_year) > self.end_year:
            raise LBTException(
                f"Expense year ({np.max(self.expense_year)}) "
                f"is after the end year of the project ({self.end_year})"
            )

        # Exception: expense_year is before the start year of the project
        if np.min(self.expense_year) < self.start_year:
            raise LBTException(
                f"Expense year ({np.min(self.expense_year)}) "
                f"is before the start year of the project ({self.start_year})"
            )

    def expenditures_pre_tax(
        self,
        year_inflation: np.ndarray = None,
        inflation_rate: np.ndarray | int | float = 0.0,
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
        tax_rate: np.ndarray | float = 0.0
    ) -> np.ndarray:
        """
        Calculate and distribute indirect taxes over the project duration.

        This method computes indirect taxes by applying the provided tax rates, portions, and
        discounts to the associated costs. The calculated taxes are then distributed over the
        project's timeline, and the total indirect taxes are summed for each project year.

        Parameters
        ----------
        tax_rate : np.ndarray or float, optional
            A NumPy array or float representing the tax rate applied to the costs. If not provided,
            a default rate of 0.0 will be used. When provided as an array, it should match
            the project years.

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
            tax_portion=self.tax_portion,
            tax_rate=tax_rate,
            tax_discount=self.tax_discount,
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
                    self.start_year == other.start_year,
                    self.end_year == other.end_year,
                    np.allclose(self.expense_year, other.expense_year),
                    np.allclose(self.tax_portion, other.tax_portion),
                    np.allclose(self.tax_discount, other.tax_discount),
                    np.allclose(self.final_year, other.final_year),
                    np.allclose(self.utilized_land_area, other.utilized_land_area),
                    np.allclose(self.utilized_building_area, other.utilized_building_area),
                    np.allclose(self.njop_land, other.njop_land),
                    np.allclose(self.njop_building, other.njop_building),
                    np.allclose(self.gross_revenue, other.gross_revenue),
                    np.allclose(self.cost, other.cost),
                    self.cost_allocation == other.cost_allocation,
                )
            )

        # Between an instance of LBT and an integer/a float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) == other

        else:
            return False

    def __lt__(self, other):
        # Between an instance of LBT with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) < np.sum(other.cost)

        # Between an instance of LBT and an integer/a float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) < other

        else:
            raise LBTException(
                f"Must compare an instance of LBT with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer, or a float."
            )

    def __le__(self, other):
        # Between an instance of LBT with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) <= np.sum(other.cost)

        # Between an instance of LBT and an integer/a float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) <= other

        else:
            raise LBTException(
                f"Must compare an instance of LBT with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer, or a float."
            )

    def __gt__(self, other):
        # Between an instance of LBT with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) > np.sum(other.cost)

        # Between an instance of ASR and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) > other

        else:
            raise LBTException(
                f"Must compare an instance of ASR with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer, or a float."
            )

    def __ge__(self, other):
        # Between an instance of LBT with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) >= np.sum(other.cost)

        # Between an instance of LBT and an integer/a float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) >= other

        else:
            raise LBTException(
                f"Must compare an instance of LBT with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR/LBT, an integer, or a float."
            )

    def __add__(self, other):
        # Only allows addition between an instance of LBT
        # and another instance of LBT
        if isinstance(other, LBT):
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            expense_year_combined = np.concatenate((self.expense_year, other.expense_year))
            tax_portion_combined = np.concatenate((self.tax_portion, other.tax_portion))
            tax_discount_combined = np.concatenate((self.tax_discount, other.tax_discount))
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
                tax_portion=tax_portion_combined,
                tax_discount=tax_discount_combined,
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
        # Only allows subtraction between an instance of LBT
        # and another instance of LBT
        if isinstance(other, LBT):
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            expense_year_combined = np.concatenate((self.expense_year, other.expense_year))
            tax_portion_combined = np.concatenate((self.tax_portion, other.tax_portion))
            tax_discount_combined = np.concatenate((self.tax_discount, other.tax_discount))
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
                tax_portion=tax_portion_combined,
                tax_discount=tax_discount_combined,
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
                tax_portion=self.tax_portion,
                tax_discount=self.tax_discount,
                final_year=self.final_year,
                utilized_land_area=self.utilized_land_area,
                utilized_building_area=self.utilized_building_area,
                njop_land=self.njop_land * other,
                njop_building=self.njop_building * other,
                gross_revenue=self.gross_revenue * other,
            )

        else:
            raise LBTException(
                f"Must multiply with an integer or a float. "
                f"{other}({other.__class__.__qualname__}) "
                f"is not an integer nor a float."
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        # Between an instance of LBT with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT)):
            return np.sum(self.cost) / np.sum(other.cost)

        # Between an instance of LBT and an integer/a float
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
                    tax_portion=self.tax_portion,
                    tax_discount=self.tax_discount,
                )

        else:
            raise LBTException(
                f"Must divide with an instance of CapitalCost/Intangible/OPEX/ASR/LBT, "
                f"integer or a float. {other}({other.__class__.__qualname__}) is not "
                f"an instance  of CapitalCost/Intangible/OPEX/ASR/LBT nor an integer "
                f"nor a float."
            )


@dataclass
class CostOfSales(GeneralCost):
    """
    Manages a cost of sales.

    Parameters
    ----------
    Parameters are inherited from class GeneralCost

    Notes
    -----
    Overridden attributes are: expense_year and cost
    """

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
        -   Prepare attribute tax_portion,
        -   Prepare attribute tax_discount,
        -   Raise an exception: expense_year is after the end year of the project,
        -   Raise an exception: expense_year is before the start_year of the project
        """

        # Prepare attributes project_duration and project_years
        if self.end_year >= self.start_year:
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise CostOfSalesException(
                f"start year {self.start_year} is after the end year {self.end_year}"
            )

        # Prepare attribute expense_year
        if self.expense_year is None:
            self.expense_year = self.project_years.copy()

        else:
            if not isinstance(self.expense_year, np.ndarray):
                raise CostOfSalesException(
                    f"Attribute expense_year must be provided as a numpy.ndarray, "
                    f"not as a/an {self.expense_year.__class__.__qualname__}"
                )

            expense_year_nan_sum = np.sum(pd.isna(self.expense_year))
            if expense_year_nan_sum > 0:
                raise CostOfSalesException(
                    f"Missing values in array expense_year: {self.expense_year}"
                )

        self.expense_year = self.expense_year.astype(int)

        # Prepare attribute cost
        if self.cost is None:
            self.cost = np.zeros_like(self.expense_year)

        else:
            if not isinstance(self.cost, np.ndarray):
                raise CostOfSalesException(
                    f"Attribute cost must be provided as a numpy.ndarray, "
                    f"not as an/a {self.cost.__class__.__qualname__}"
                )

            cost_nan_id = np.argwhere(pd.isna(self.cost)).ravel()
            if len(cost_nan_id) > 0:
                self.cost[cost_nan_id] = np.zeros(len(cost_nan_id))

        self.cost = self.cost.astype(np.float64)

        # Prepare attribute cost_allocation
        if self.cost_allocation is None:
            self.cost_allocation = [FluidType.OIL for _ in range(len(self.expense_year))]

        else:
            if not isinstance(self.cost_allocation, list):
                raise CostOfSalesException(
                    f"Attribute cost allocation must be provided as a list, "
                    f"not as an/a {self.cost_allocation.__class__.__qualname__}"
                )

            self.cost_allocation = [
                FluidType.OIL if pd.isna(val) else val for _, val in enumerate(self.cost_allocation)
            ]

        # Prepare attribute description
        if self.description is None:
            self.description = [" " for _ in range(len(self.expense_year))]

        else:
            if not isinstance(self.description, list):
                raise CostOfSalesException(
                    f"Attribute description must be provided as a list, "
                    f"not as an/a {self.description.__class__.__qualname__}"
                )

            self.description = [
                " " if pd.isna(val) else val for _, val in enumerate(self.description)
            ]

        # Prepare attribute tax_portion
        if self.tax_portion is None:
            self.tax_portion = np.zeros_like(self.expense_year)

        else:
            if not isinstance(self.tax_portion, np.ndarray):
                raise CostOfSalesException(
                    f"Attribute tax_portion must be given as a numpy.ndarray, "
                    f"not as a/an {self.tax_portion.__class__.__qualname__}"
                )

            tax_portion_nan_id = np.argwhere(pd.isna(self.tax_portion)).ravel()
            if len(tax_portion_nan_id) > 0:
                self.tax_portion[tax_portion_nan_id] = np.zeros(len(tax_portion_nan_id))

            tax_portion_large = np.sum(self.tax_portion > 1.0, dtype=np.float64)
            tax_portion_negative = np.sum(self.tax_portion < 0.0, dtype=np.float64)
            if tax_portion_large > 0 or tax_portion_negative > 0:
                raise CostOfSalesException(
                    f"The value of tax_portion must be: 0 < tax_portion < 1, "
                    f"tax_portion: {self.tax_portion}"
                )

        self.tax_portion = self.tax_portion.astype(np.float64)

        # Prepare attribute tax_discount
        if not isinstance(self.tax_discount, (float, int, np.ndarray)):
            raise CostOfSalesException(
                f"Attribute tax_discount must be provided as a float/int or as a numpy.ndarray, "
                f"not as a/an {self.tax_discount.__class__.__qualname__}"
            )

        if isinstance(self.tax_discount, (float, int)):
            if self.tax_discount < 0 or self.tax_discount > 1:
                raise CostOfSalesException(f"Attribute tax_discount must be between 0 and 1")

            self.tax_discount = np.repeat(self.tax_discount, len(self.expense_year))

        elif isinstance(self.tax_discount, np.ndarray):
            tax_discount_nan_id = np.argwhere(pd.isna(self.tax_discount)).ravel()
            if len(tax_discount_nan_id) > 0:
                self.tax_discount[tax_discount_nan_id] = np.zeros(len(tax_discount_nan_id))

            tax_discount_large = np.sum(self.tax_discount > 1.0, dtype=np.float64)
            tax_discount_negative = np.sum(self.tax_discount < 0.0, dtype=np.float64)
            if tax_discount_large > 0 or tax_discount_negative > 0:
                raise CostOfSalesException(
                    f"The value of tax_discount must be: 0 < tax_discount < 1, "
                    f"tax_discount: {self.tax_discount}"
                )

        self.tax_discount = self.tax_discount.astype(np.float64)

        # Check input data for unequal length
        arr_length = len(self.expense_year)

        if not all(
            len(arr) == arr_length
            for arr in [
                self.cost,
                self.cost_allocation,
                self.description,
                self.tax_portion,
                self.tax_discount,
            ]
        ):
            raise CostOfSalesException(
                f"Unequal length of arrays: "
                f"expense_year: {len(self.expense_year)}, "
                f"cost: {len(self.cost)}, "
                f"cost_allocation: {len(self.cost_allocation)}, "
                f"description: {len(self.description)}, "
                f"tax_portion: {len(self.tax_portion)}, "
                f"tax_discount: {len(self.tax_discount)} "
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
                    self.start_year == other.start_year,
                    self.end_year == other.end_year,
                    np.allclose(self.expense_year, other.expense_year),
                    np.allclose(self.cost, other.cost),
                    np.allclose(self.tax_portion, other.tax_portion),
                    np.allclose(self.tax_discount, other.tax_discount),
                    self.cost_allocation == other.cost_allocation
                )
            )

        # Between an instance of CostOfSales and an integer/a float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) == other

        else:
            return False

    def __lt__(self, other):
        # Between an instance of CostOfSales with another instance of CostOfSales
        if isinstance(other, CostOfSales):
            return np.sum(self.cost) < np.sum(other.cost)

        # Between an instance of CostOfSales and an integer/a float
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

        # Between an instance of CostOfSales and an integer/a float
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

        # Between an instance of CostOfSales and an integer/a float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) > other

        else:
            raise CostOfSalesException(
                f"Must compare an instance of CostOfSales with another instance of "
                f"CostOfSales, an integer, or a float"
            )

    def __ge__(self, other):
        # Between an instance of CostOfSales with another instance of CostOfSales
        if isinstance(other, CostOfSales):
            return np.sum(self.cost) >= np.sum(other.cost)

        # Between an instance of CostOfSales and an integer/a float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) >= other

        else:
            raise CostOfSalesException(
                f"Must compare an instance of CostOfSales with another instance of "
                f"CostOfSales, an integer, or a float"
            )

    def __add__(self, other):
        # Only allows addition between an instance of CostOfSales
        # and another instance of CostOfSales
        if isinstance(other, CostOfSales):
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            cost_allocation_combined = self.cost_allocation + other.cost_allocation
            description_combined = self.description + other.description
            tax_portion_combined = np.concatenate((self.tax_portion, other.tax_portion))
            tax_discount_combined = np.concatenate((self.tax_discount, other.tax_discount))
            expense_year_combined = np.concatenate((self.expense_year, other.expense_year))
            cost_combined = np.concatenate((self.cost, other.cost))

            return CostOfSales(
                start_year=start_year_combined,
                end_year=end_year_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
                tax_portion=tax_portion_combined,
                tax_discount=tax_discount_combined,
                expense_year=expense_year_combined,
                cost=cost_combined,
            )

        else:
            raise CostOfSalesException(
                f"Must add between an instance of CostOfSales with another instance "
                f"of CostOfSales. {other}({other.__class__.__qualname__}) is not "
                f"an instance of CostOfSales."
            )

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        # Only allows subtraction between an instance of CostOfSales
        # and another instance of CostOfSales
        if isinstance(other, CostOfSales):
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            cost_allocation_combined = self.cost_allocation + other.cost_allocation
            description_combined = self.description + other.description
            tax_portion_combined = np.concatenate((self.tax_portion, other.tax_portion))
            tax_discount_combined = np.concatenate((self.tax_discount, other.tax_discount))
            expense_year_combined = np.concatenate((self.expense_year, other.expense_year))
            cost_combined = np.concatenate((self.cost, -other.cost))

            return CostOfSales(
                start_year=start_year_combined,
                end_year=end_year_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
                tax_portion=tax_portion_combined,
                tax_discount=tax_discount_combined,
                expense_year=expense_year_combined,
                cost=cost_combined,
            )

        else:
            raise CostOfSalesException(
                f"Must subtract between an instance of CostOfSales with another instance "
                f"of CostOfSales. {other}({other.__class__.__qualname__}) is not "
                f"an instance of CostOfSales."
            )

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        # Multiplication is allowed only with an integer/a float
        if isinstance(other, (int, float)):
            return CostOfSales(
                start_year=self.start_year,
                end_year=self.end_year,
                cost_allocation=self.cost_allocation,
                description=self.description,
                tax_portion=self.tax_portion,
                tax_discount=self.tax_discount,
                expense_year=self.expense_year,
                cost=self.cost * other,
            )

        else:
            raise CostOfSalesException(
                f"Must multiply with an integer or a float. "
                f"{other}({other.__class__.__qualname__}) "
                f"is not an integer nor a float."
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        # Between an instance of CostOfSales with another instance of CostOfSales
        if isinstance(other, CostOfSales):
            return np.sum(self.cost) / np.sum(other.cost)

        # Between an instance of CostOfSales and an integer/a float
        elif isinstance(other, (int, float)):
            # Cannot divide by zero
            if other == 0:
                raise CostOfSalesException(f"Cannot divide with zero.")

            else:
                return CostOfSales(
                    start_year=self.start_year,
                    end_year=self.end_year,
                    cost_allocation=self.cost_allocation,
                    description=self.description,
                    tax_portion=self.tax_portion,
                    tax_discount=self.tax_discount,
                    expense_year=self.expense_year,
                    cost=self.cost / other,
                )

        else:
            raise CostOfSalesException(
                f"Must divide with an instance of CostOfSales, an integer, or a float, "
                f"{other}({other.__class__.__qualname__}) is not an instance of "
                f"CostOfSales nor an integer nor a float."
            )


@dataclass
class SunkCost(GeneralCost):

    # Local arguments
    onstream_year: int = field(default=None)
    pod1_year: int = field(default=None)

    # Overridden argument
    cost: np.ndarray = field(default=None)

    # Attributes to be defined later
    sunk_cost_oil_total: float = field(default=0.0, init=False)
    sunk_cost_gas_total: float = field(default=0.0, init=False)
    pre_onstream_cost_oil_total: float = field(default=0.0, init=False)
    pre_onstream_cost_gas_total: float = field(default=0.0, init=False)

    def __post_init__(self):
        """
        Handles the following operations/procedures:
        -   Prepare attribute project_duration and project_years,
        -   Prepare attribute onstream_year,
        -   Prepare attribute pod1_year,
        -   Prepare attribute expense_year,
        -   Prepare attribute cost,
        -   Prepare attribute cost_allocation,
        -   Prepare attribute description,
        -   Prepare attribute tax_portion,
        -   Prepare attribute tax_discount,
        -   Check input data for unequal length of arrays,
        -   Raise an error: expense year is after the end year of the project,
        -   Raise an error: expense year is after the end year of the project
        """

        # Prepare attribute project_duration and project_years
        if self.end_year >= self.start_year:
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise SunkCostException(
                f"start year {self.start_year} is after the end year {self.end_year} "
                f"of the project"
            )

        # Prepare attribute onstream_year
        if self.onstream_year is None:
            raise SunkCostException(
                f"Missing data for onstream_year: {self.onstream_year}"
            )

        else:
            if not isinstance(self.onstream_year, int):
                raise SunkCostException(
                    f"Attribute onstream_year must be provided as an int, "
                    f"not as a/an {self.onstream_year.__class__.__qualname__}"
                )

            if self.onstream_year < self.start_year:
                raise SunkCostException(
                    f"Onstream year ({self.onstream_year}) is before the start "
                    f"year of the project ({self.start_year})"
                )

            if self.onstream_year > self.end_year:
                raise SunkCostException(
                    f"Onstream year ({self.onstream_year}) is after the end year "
                    f"of the project ({self.end_year})"
                )

        # Prepare attribute pod1_year
        if self.pod1_year is None:
            self.pod1_year = self.onstream_year

        else:
            if not isinstance(self.pod1_year, int):
                raise SunkCostException(
                    f"Attribute pod1_year must be provided as an int, "
                    f"not as a/an {self.pod1_year.__class__.__qualname__}"
                )

            if self.pod1_year > self.onstream_year:
                raise SunkCostException(
                    f"POD I year ({self.pod1_year}) is after the onstream "
                    f"year ({self.onstream_year})"
                )

            if self.pod1_year < self.start_year:
                raise SunkCostException(
                    f"POD I year ({self.pod1_year}) is before the start year "
                    f"of the project ({self.start_year})"
                )

            if self.pod1_year > self.end_year:
                raise SunkCostException(
                    f"POD I year ({self.pod1_year}) is after the end year "
                    f"of the project ({self.end_year})"
                )

        # Prepare attribute expense_year
        if not isinstance(self.expense_year, np.ndarray):
            raise SunkCostException(
                f"Attribute expense_year must be provided as a numpy.ndarray, "
                f"not as a/an {self.expense_year.__class__.__qualname__}"
            )

        else:
            expense_year_nan_sum = np.sum(pd.isna(self.expense_year), dtype=np.float64)
            if expense_year_nan_sum > 0:
                raise SunkCostException(
                    f"Missing values in array expense_year: {self.expense_year}"
                )

            expense_year_large_sum = np.sum(self.expense_year > self.onstream_year, dtype=int)
            if expense_year_large_sum > 0:
                raise SunkCostException(
                    f"Cannot accept expense_year > onstream_year, "
                    f"expense_year: ({self.expense_year}), "
                    f"onstream_year: ({self.onstream_year})"
                )

        self.expense_year = self.expense_year.astype(int)

        # Prepare attribute cost
        if self.cost is None:
            self.cost = np.zeros_like(self.project_years, dtype=np.float64)

        else:
            if not isinstance(self.cost, np.ndarray):
                raise SunkCostException(
                    f"Attribute cost must be provided as a numpy.ndarray, "
                    f"not as a/an {self.cost.__class__.__qualname__}"
                )

            else:
                cost_nan_id = np.argwhere(pd.isna(self.cost)).ravel()
                if len(cost_nan_id) > 0:
                    self.cost[cost_nan_id] = np.zeros(len(cost_nan_id))

        self.cost = self.cost.astype(np.float64)

        # Prepare attribute cost_allocation
        if self.cost_allocation is None:
            self.cost_allocation = [FluidType.OIL for _ in range(len(self.expense_year))]

        else:
            if not isinstance(self.cost_allocation, list):
                raise SunkCostException(
                    f"Attribute cost_allocation must be given as a list, "
                    f"not as a/an {self.cost_allocation.__class__.__qualname__}"
                )

            self.cost_allocation = [
                FluidType.OIL if pd.isna(val) else val for _, val in enumerate(self.cost_allocation)
            ]

        # Prepare attribute description
        if self.description is None:
            self.description = [" " for _ in range(len(self.expense_year))]

        else:
            if not isinstance(self.description, list):
                raise SunkCostException(
                    f"Attribute description must be provided as a list, "
                    f"not as a/an {self.description.__class__.__qualname__}"
                )

            self.description = [
                " " if pd.isna(val) else val for _, val in enumerate(self.description)
            ]

        # Prepare attribute tax_portion
        if self.tax_portion is None:
            self.tax_portion = np.zeros_like(self.expense_year)

        else:
            if not isinstance(self.tax_portion, np.ndarray):
                raise SunkCostException(
                    f"Attribute tax_portion must be given as a numpy.ndarray, "
                    f"not as a/an {self.tax_portion.__class__.__qualname__}"
                )

            tax_portion_nan_id = np.argwhere(pd.isna(self.tax_portion)).ravel()
            if len(tax_portion_nan_id) > 0:
                self.tax_portion[tax_portion_nan_id] = np.zeros(len(tax_portion_nan_id))

            tax_portion_large = np.sum(self.tax_portion > 1.0, dtype=np.float64)
            tax_portion_negative = np.sum(self.tax_portion < 0.0, dtype=np.float64)
            if tax_portion_large > 0 or tax_portion_negative > 0:
                raise SunkCostException(
                    f"The value of tax_portion must be: 0 < tax_portion < 1, "
                    f"tax_portion: {self.tax_portion}"
                )

        self.tax_portion = self.tax_portion.astype(np.float64)

        # Prepare attribute tax_discount
        if not isinstance(self.tax_discount, (float, int, np.ndarray)):
            raise SunkCostException(
                f"Attribute tax_discount must be provided as a float/int "
                f"or as a numpy.ndarray, not as a/an "
                f"{self.tax_discount.__class__.__qualname__}"
            )

        if isinstance(self.tax_discount, (float, int)):
            if self.tax_discount < 0 or self.tax_discount > 1:
                raise SunkCostException(f"Attribute tax_discount must be between 0 and 1")

            self.tax_discount = np.repeat(self.tax_discount, len(self.expense_year))

        elif isinstance(self.tax_discount, np.ndarray):
            tax_discount_nan_id = np.argwhere(pd.isna(self.tax_discount)).ravel()
            if len(tax_discount_nan_id) > 0:
                self.tax_discount[tax_discount_nan_id] = np.zeros(len(tax_discount_nan_id))

            tax_discount_large = np.sum(self.tax_discount > 1.0, dtype=np.float64)
            tax_discount_negative = np.sum(self.tax_discount < 0.0, dtype=np.float64)
            if tax_discount_large > 0 or tax_discount_negative > 0:
                raise SunkCostException(
                    f"The value of tax_discount must be: 0 < tax_discount < 1, "
                    f"tax_discount: {self.tax_discount}"
                )

        self.tax_discount = self.tax_discount.astype(np.float64)

        # Check input data for unequal length of arrays
        arr_reference = len(self.expense_year)

        if not all(
            len(arr) == arr_reference
            for arr in [
                self.cost,
                self.cost_allocation,
                self.description,
                self.tax_portion,
                self.tax_discount,
            ]
        ):
            raise SunkCostException(
                f"Unequal length of arrays: "
                f"expense_year: {len(self.expense_year)}, "
                f"cost: {len(self.cost)}, "
                f"cost_allocation: {len(self.cost_allocation)}, "
                f"description: {len(self.description)}, "
                f"tax_portion: {len(self.tax_portion)}, "
                f"tax_discount: {len(self.tax_discount)} "
            )

        # Raise an error: expense year is after the end year of the project
        if np.max(self.expense_year) > self.end_year:
            raise SunkCostException(
                f"Expense year ({np.max(self.expense_year)}) "
                f"is after the end year of the project ({self.end_year})"
            )

        # Raise an error: expense year is after the end year of the project
        if np.min(self.expense_year) < self.start_year:
            raise SunkCostException(
                f"Expense year ({np.min(self.expense_year)}) "
                f"is before the start year of the project ({self.start_year})"
            )

    def _get_oil_cost_classification(
        self,
        tax_rate: np.ndarray | float = 0.0,
    ):
        """
        Calculate oil-related cost classifications (sunk cost and pre-onstream cost)
        adjusted by VAT.

        The function categorizes costs into:
        1. Sunk costs: expenses incurred before or during POD I approval year
        2. Pre-onstream costs: expenses between POD I approval and onstream year

        Parameters
        ----------
        tax_rate : np.ndarray or float, optional
            Tax rate to be applied for VAT calculation. Default is 0.0 (no tax).

        Returns
        -------
        tuple
            A tuple containing:
            - sunk_cost_oil_total : float
                Total sunk costs allocated to oil
            - pre_onstream_cost_oil_total : float
                Total pre-onstream costs allocated to oil

        Raises
        ------
        SunkCostException
            If POD I approval year is later than onstream year

        Notes
        -----
        - Costs are first adjusted by adding VAT calculated using `calc_indirect_tax`
        - If FluidType.OIL is not in cost allocation, both returned values will be 0
        - When POD I year equals onstream year, all pre-onstream costs are 0
        - When POD I year is before onstream year, costs are split between:
            * Sunk costs (<= POD I year)
            * Pre-onstream costs (> POD I year and <= onstream year)
        """

        # Adjust cost by VAT
        cost_adjusted_by_vat = (
            self.cost +
            calc_indirect_tax(
                start_year=self.start_year,
                cost=self.cost,
                expense_year=self.expense_year,
                project_years=self.project_years,
                tax_portion=self.tax_portion,
                tax_rate=tax_rate,
                tax_discount=self.tax_discount,
            )
        )

        # Determine oil sunk cost and oil pre-onstream cost
        if FluidType.OIL not in self.cost_allocation:
            sunk_cost_oil_total = 0.0
            pre_onstream_cost_oil_total = 0.0

        else:
            # Year of POD I approval equals to onstream year
            if self.pod1_year == self.onstream_year:

                # Oil sunk cost (total)
                sunk_cost_id = np.argwhere(self.expense_year <= self.onstream_year).ravel()
                cost_allocation_id = np.array(
                    [self.cost_allocation[val] for _, val in enumerate(sunk_cost_id)]
                )
                sunk_cost_oil_id = np.array(
                    [i for i, val in enumerate(cost_allocation_id) if val == FluidType.OIL]
                )

                if len(sunk_cost_oil_id) > 0:
                    sunk_cost_oil_total = np.sum(
                        cost_adjusted_by_vat[sunk_cost_oil_id], dtype=np.float64
                    )
                else:
                    sunk_cost_oil_total = 0.0

                # Oil pre-onstream cost (total)
                pre_onstream_cost_oil_total = 0.0

            # Year of POD I approval is before the onstream year
            elif self.pod1_year < self.onstream_year:

                # Oil sunk cost (total)
                sunk_cost_id = np.argwhere(self.expense_year <= self.pod1_year).ravel()
                cost_allocation_id = np.array(
                    [self.cost_allocation[val] for _, val in enumerate(sunk_cost_id)]
                )
                sunk_cost_oil_id = np.array(
                    [i for i, val in enumerate(cost_allocation_id) if val == FluidType.OIL]
                )

                if len(sunk_cost_oil_id) > 0:
                    sunk_cost_oil_total = np.sum(
                        cost_adjusted_by_vat[sunk_cost_oil_id], dtype=np.float64
                    )
                else:
                    sunk_cost_oil_total = 0.0

                # Oil pre-onstream cost (total)
                pre_onstream_cost_id = np.argwhere(self.expense_year >= self.pod1_year).ravel()
                cost_allocation_id = np.array(
                    [self.cost_allocation[val] for _, val in enumerate(pre_onstream_cost_id)]
                )
                pre_onstream_cost_oil_id = np.array(
                    [i for i, val in enumerate(cost_allocation_id) if val == FluidType.OIL]
                )

                if len(pre_onstream_cost_oil_id) > 0:
                    pre_onstream_cost_oil_total = np.sum(
                        cost_adjusted_by_vat[pre_onstream_cost_oil_id], dtype=np.float64
                    )
                else:
                    pre_onstream_cost_oil_total = 0.0

            else:
                raise SunkCostException(
                    f"Cannot have POD I year larger than onstream year"
                )

        return sunk_cost_oil_total, pre_onstream_cost_oil_total

    def _get_gas_cost_classification(
        self,
        tax_rate: np.ndarray | float = 0.0,
    ):
        pass

    def get_cost_classification(
        self,
        tax_rate: np.ndarray | float = 0.0,
    ):
        
        self._get_oil_cost_classification(tax_rate=tax_rate)
        self._get_gas_cost_classification(tax_rate=tax_rate)

    def total_amortization_rate(self):
        pass

    def total_amortization_book_value(self):
        pass

    def __len__(self):
        pass

    def __eq__(self, other):
        pass

    def __lt__(self, other):
        pass

    def __le__(self, other):
        pass

    def __gt__(self, other):
        pass

    def __ge__(self, other):
        pass

    def __add__(self, other):
        pass

    def __iadd__(self, other):
        pass

    def __sub__(self, other):
        pass

    def __rsub__(self, other):
        pass

    def __mul__(self, other):
        pass

    def __rmul__(self, other):
        pass

    def __truediv__(self, other):
        pass
