"""
Prepare and classify cost data based on its components. The associated cost components are:
(1) Tangible,
(2) Intangible,
(3) OPEX,
(4) ASR,
(5) LBT,
"""

import numpy as np
from dataclasses import dataclass, field

import pyscnomics.econ.depreciation as depr
from pyscnomics.tools.helper import apply_cost_adjustment, check_input, get_cost_adjustment_by_inflation
from pyscnomics.econ.selection import FluidType, DeprMethod, TaxType


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
    cost: np.ndarray
    expense_year: np.ndarray
    cost_allocation: list[FluidType] = field(default=None)
    description: list[str] = field(default=None)

    # Attribute to be defined later on
    project_duration: int = field(default=None, init=False)
    project_years: np.ndarray = field(default=None, init=False)

    def _prepare_expenditures_pre_tax(
        self,
        year_inflation: np.ndarray = None,
        inflation_rate: np.ndarray | float = 0.0,
    ) -> tuple:
        """
        Calculate the pre-tax expenditures, adjusted for inflation, across the project duration.

        This function adjusts the project costs for inflation based on the specified inflation
        rates and years, then allocates the costs to the corresponding expense years.

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

        Returns
        -------
        tuple
            A tuple containing two elements:
            - `cost_adjusted_by_inflation` : np.ndarray
                The array of costs adjusted for inflation.
            - `expenses` : np.ndarray
                The allocated expenses by the corresponding expense year, padded with zeros if
                necessary to match the project duration.

        Notes
        -----
        The function performs the following steps:
        1. Checks the validity of `year_inflation`, ensuring it is a NumPy array and within the valid
           range of project years (`start_year` to `end_year`).
        2. Adjusts costs based on the inflation scheme using the `get_cost_adjustment_by_inflation`
           function, which applies the inflation rates for the given `year_inflation`.
        3. Allocates the adjusted costs to the associated expense years and returns the result as a
           NumPy array, padded with zeros if necessary to match the project duration.
        """

        # Prepare attribute year_inflation
        if year_inflation is None:
            year_inflation = np.repeat(self.start_year, len(self.cost))

        else:
            if not isinstance(year_inflation, np.ndarray):
                raise GeneralCostException(
                    f"Attribute year_inflation must be given as a numpy.ndarray, "
                    f"not as a/an {year_inflation.__class__.__qualname__}"
                )

            if len(year_inflation) != len(self.cost):
                raise GeneralCostException(
                    f"Unequal length of arrays: "
                    f"cost: {len(self.cost)}, "
                    f"year_inflation: {len(year_inflation)}"
                )

        year_inflation = year_inflation.astype(np.int64)

        # Raise an error: year_inflation is before the start year of the project
        if np.min(year_inflation) < self.start_year:
            raise GeneralCostException(
                f"year_inflation ({np.min(year_inflation)}) is before the start year "
                f"of the project ({self.start_year})"
            )

        # Raise an error: year_inflation is after the end year of the project
        if np.max(year_inflation) > self.end_year:
            raise GeneralCostException(
                f"year_inflation ({np.max(year_inflation)}) is after the end year "
                f"of the project ({self.end_year})"
            )

        # Cost adjustment by inflation scheme
        cost_adjusted_by_inflation = get_cost_adjustment_by_inflation(
            start_year=self.start_year,
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

        return (
            cost_adjusted_by_inflation,
            np.concatenate((expenses, zeros))
        )

    def _prepare_indirect_tax(
        self,
        tax_portion: np.ndarray = None,
        tax_rate: np.ndarray | float = 0.0,
        tax_discount: float = 0.0,
    ) -> tuple:
        """
        Calculate the indirect tax for the project costs based on tax portions, rates,
        and discounts.

        This function applies indirect tax calculations on the project costs by considering
        a specified portion of costs subject to tax, tax rates, and any applicable tax discounts.
        If no tax portion is provided, it defaults to zero (i.e., no tax).

        Parameters
        ----------
        tax_portion : np.ndarray, optional
            A NumPy array representing the portion of the cost subject to taxation for each cost entry.
            If not provided, it defaults to an array of zeros, implying no portion of the costs
            is subject to tax.
        tax_rate : np.ndarray or float, optional
            The tax rate to apply. If provided as a float, the same rate is applied uniformly
            across all years. If provided as an array, it should correspond to the project years
            (default is 0.0).
        tax_discount : float, optional
            A discount on the tax rate, applied uniformly across all costs. This should be a
            decimal fraction representing the percentage of discount, where 1.0 corresponds to
            a 100% discount (default is 0.0).

        Returns
        -------
        tuple
            A tuple containing:
            - indirect_tax (np.ndarray): The calculated indirect tax for each cost.
            - allocated_tax (np.ndarray): The indirect tax allocated by the corresponding
              expense year. The result is padded with zeros to match the total project duration.

        Notes
        -----
        The function calculates the indirect tax using the formula:

            indirect_tax = cost * (tax_portion * tax_rate * (1.0 - tax_discount))
        """

        # Prepare attribute tax portion
        if tax_portion is None:
            tax_portion = np.zeros_like(self.cost)

        else:
            if not isinstance(tax_portion, np.ndarray):
                raise GeneralCostException(
                    f"Attribute tax portion must be given as a numpy.ndarray, "
                    f"not as a/an {tax_portion.__class__.__qualname__}"
                )

        tax_portion = tax_portion.astype(np.float64)

        # Prepare attribute tax rate
        if not isinstance(tax_rate, (np.ndarray, float)):
            raise GeneralCostException(
                f"Attribute tax rate must be given as a float or as a numpy.ndarray, "
                f"not as a/an {tax_rate.__class__.__qualname__}"
            )

        else:
            tax_rate_arr = check_input(target_func=self.project_years, param=tax_rate)

        # Prepare attribute tax discount
        if not isinstance(tax_discount, float):
            raise GeneralCostException(
                f"Attribute tax discount must be given as a numpy.ndarray, "
                f"not as a/an {tax_discount.__class__.__qualname__}"
            )

        else:
            tax_discount = np.repeat(tax_discount, len(self.cost))

        tax_discount = tax_discount.astype(np.float64)

        # Calculate indirect tax
        tax_rate_ids = (self.expense_year - self.start_year).astype(np.int64)
        indirect_tax = self.cost * (tax_portion * tax_rate_arr[tax_rate_ids] * (1.0 - tax_discount))

        # Allocate indirect tax by their associated expense year
        expenses = np.bincount(
            self.expense_year - self.start_year, weights=indirect_tax
        )
        zeros = np.zeros(self.project_duration - len(expenses))

        return (
            indirect_tax,
            np.concatenate((expenses, zeros))
        )

    def get_cost_adjusted_by_inflation(
        self,
        year_inflation: np.ndarray = None,
        inflation_rate: np.ndarray | float = 0.0,
    ) -> np.ndarray:
        """
        Get project costs adjusted for inflation.

        This function calculates and returns the costs after adjusting for inflation
        based on the specified `year_inflation` and `inflation_rate`. It utilizes the
        `_prepare_expenditures_pre_tax` method to perform the inflation adjustment.

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

        Returns
        -------
        np.ndarray
            The array of costs adjusted for inflation.

        Notes
        -----
        This function extracts the inflation-adjusted costs from the first element of
        the tuple returned by the `_prepare_expenditures_pre_tax` method.
        """

        return self._prepare_expenditures_pre_tax(
            year_inflation=year_inflation,
            inflation_rate=inflation_rate,
        )[0]

    def expenditures_pre_tax(
        self,
        year_inflation: np.ndarray = None,
        inflation_rate: np.ndarray | float = 0.0,
    ) -> np.ndarray:
        """
        Get pre-tax expenditures after adjusting for inflation.

        This function calculates the pre-tax expenditures, which are adjusted based on
        the given inflation scheme. It uses the `_prepare_expenditures_pre_tax` method
        to adjust costs for inflation and allocate them to the appropriate expense years.

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

        Returns
        -------
        np.ndarray
            The array of pre-tax expenditures, allocated by the corresponding expense
            year and adjusted for inflation. The result is padded with zeros if necessary
            to match the total project duration.

        Notes
        -----
        This function returns the second element from the tuple produced by the
        `_prepare_expenditures_pre_tax` method, which represents the pre-tax expenditures
        allocated by expense year.
        """

        return self._prepare_expenditures_pre_tax(
            year_inflation=year_inflation,
            inflation_rate=inflation_rate,
        )[1]

    def get_indirect_tax(
        self,
        tax_portion: np.ndarray = None,
        tax_rate: np.ndarray | float = 0.0,
        tax_discount: float = 0.0,
    ) -> np.ndarray:
        """
        Calculate and return the indirect tax for each cost.

        This function computes the indirect tax for the project based on the tax portion,
        tax rate, and tax discount. The calculation is performed using the
        `_prepare_indirect_tax` method, and only the indirect tax (without allocation)
        is returned.

        Parameters
        ----------
        tax_portion : np.ndarray, optional
            A NumPy array representing the portion of the cost subject to taxation for each cost entry.
            If not provided, it defaults to an array of zeros, implying no portion of the costs
            is subject to tax.
        tax_rate : np.ndarray or float, optional
            The tax rate to apply. If provided as a float, the same rate is applied uniformly
            across all years. If provided as an array, it should correspond to the project years
            (default is 0.0).
        tax_discount : float, optional
            A discount on the tax rate, applied uniformly across all costs. This should be a
            decimal fraction representing the percentage of discount, where 1.0 corresponds to
            a 100% discount (default is 0.0).

        Returns
        -------
        np.ndarray
            The calculated indirect tax for each cost in the project.

        Notes
        -----
        This function retrieves the first element from the tuple returned by
        `_prepare_indirect_tax`, which represents the computed indirect tax for each cost
        without considering the allocation by expense year.
        """

        return self._prepare_indirect_tax(
            tax_portion=tax_portion,
            tax_rate=tax_rate,
            tax_discount=tax_discount,
        )[0]

    def indirect_tax(
        self,
        tax_portion: np.ndarray = None,
        tax_rate: np.ndarray | float = 0.0,
        tax_discount: float = 0.0,
    ) -> np.ndarray:
        """
        Calculate and allocate indirect tax by expense year.

        This function computes the indirect tax for the project, adjusts the costs
        based on tax portion, tax rate, and tax discount, and allocates the indirect tax
        to the corresponding expense year. The calculation is performed using
        the `_prepare_indirect_tax` method, and the allocated tax (including padding
        for project duration) is returned.

        Parameters
        ----------
        tax_portion : np.ndarray, optional
            A NumPy array representing the portion of the cost subject to taxation for each cost entry.
            If not provided, it defaults to an array of zeros, implying no portion of the costs
            is subject to tax.
        tax_rate : np.ndarray or float, optional
            The tax rate to apply. If provided as a float, the same rate is applied uniformly
            across all years. If provided as an array, it should correspond to the project years
            (default is 0.0).
        tax_discount : float, optional
            A discount on the tax rate, applied uniformly across all costs. This should be a
            decimal fraction representing the percentage of discount, where 1.0 corresponds to
            a 100% discount (default is 0.0).

        Returns
        -------
        np.ndarray
            The indirect tax allocated by the corresponding expense year, padded with zeros
            to match the total project duration.

        Notes
        -----
        - This function returns the second element of the tuple from `_prepare_indirect_tax`,
          which represents the indirect tax allocated by expense year.
        - The output includes padding with zeros to match the total project duration.
        """

        return self._prepare_indirect_tax(
            tax_portion=tax_portion,
            tax_rate=tax_rate,
            tax_discount=tax_discount,
        )[1]

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
        2.  Calls `indirect_tax` to compute indirect taxes on the costs based on the specified
            tax portion, rate, and discount.

        The formula used to calculate post-tax expenditures is:
            expenditures_post_tax = expenditures_pre_tax + indirect_tax
        """

        return (
            self.expenditures_pre_tax(year_inflation=year_inflation, inflation_rate=inflation_rate)
            + self.indirect_tax(tax_portion=tax_portion, tax_rate=tax_rate, tax_discount=tax_discount)
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

        self.pis_year = self.pis_year.astype(np.int64)

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
        Calculate total depreciation charge and undepreciated asset value based on various parameters.

        Parameters
        ----------


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
            self.get_cost_adjusted_by_inflation(
                year_inflation=year_inflation,
                inflation_rate=inflation_rate,
            ) +
            self.get_indirect_tax(
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
        year_ref: int = None,
        tax_type: TaxType = TaxType.VAT,
        vat_rate: np.ndarray | float = 0.0,
        lbt_rate: np.ndarray | float = 0.0,
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
        year_ref : int
            The reference year for inflation calculation.
        tax_type: TaxType
            The type of tax applied to the corresponding asset.
            Available options: TaxType.VAT or TaxType.LBT (default is TaxType.VAT).
        vat_rate: np.ndarray | float
            The VAT rate to apply. Can be a single value or an array (default is 0.0).
        lbt_rate: np.ndarray | float
            The LBT rate to apply. Can be a single value or an array (default is 0.0).
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
        if year_ref is None:
            year_ref = self.start_year

        # Calculate total depreciation charge from method total_depreciation_rate
        total_depreciation_charge = self.total_depreciation_rate(
            depr_method=depr_method,
            decline_factor=decline_factor,
            year_ref=year_ref,
            tax_type=tax_type,
            vat_rate=vat_rate,
            lbt_rate=lbt_rate,
            inflation_rate=inflation_rate,
        )[0]

        return np.cumsum(
            self.expenditures_post_tax(
                year_ref=year_ref,
                tax_type=tax_type,
                vat_rate=vat_rate,
                lbt_rate=lbt_rate,
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
                    np.allclose(self.lbt_portion, other.lbt_portion),
                    np.allclose(self.lbt_discount, other.lbt_discount),
                    self.cost_allocation == other.cost_allocation,
                )
            )

        # Between an instance of CapitalCost and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) == other

        else:
            return False

    def __lt__(self, other):
        # Between an instance of CapitalCost with another instance of CapitalCost/Intangible/OPEX/ASR
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR)):
            return np.sum(self.cost) < np.sum(other.cost)

        # Between an instance of CapitalCost and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) < other

        else:
            raise CapitalException(
                f"Must compare an instance of CapitalCost with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR, an integer, or a float."
            )

    def __le__(self, other):
        # Between an instance of CapitalCost with another instance of CapitalCost/Intangible/OPEX/ASR
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR)):
            return np.sum(self.cost) <= np.sum(other.cost)

        # Between an instance of CapitalCost and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) <= other

        else:
            raise CapitalException(
                f"Must compare an instance of CapitalCost with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR, an integer, or a float."
            )

    def __gt__(self, other):
        # Between an instance of CapitalCost with another instance of CapitalCost/Intangible/OPEX/ASR
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR)):
            return np.sum(self.cost) > np.sum(other.cost)

        # Between an instance of CapitalCost and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) > other

        else:
            raise CapitalException(
                f"Must compare an instance of CapitalCost with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR, an integer, or a float."
            )

    def __ge__(self, other):
        # Between an instance of CapitalCost with another instance of CapitalCost/Intangible/OPEX/ASR
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR)):
            return np.sum(self.cost) >= np.sum(other.cost)

        # Between an instance of CapitalCost and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) >= other

        else:
            raise CapitalException(
                f"Must compare an instance of CapitalCost with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR, an integer, or a float."
            )

    def __add__(self, other):
        # Only allows addition between an instance of CapitalCost and another instance of CapitalCost
        if isinstance(other, CapitalCost):
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            cost_combined = np.concatenate((self.cost, other.cost))
            expense_year_combined = np.concatenate(
                (self.expense_year, other.expense_year)
            )
            cost_allocation_combined = self.cost_allocation + other.cost_allocation
            description_combined = self.description + other.description
            vat_portion_combined = np.concatenate((self.vat_portion, other.vat_portion))
            vat_discount_combined = np.concatenate(
                (self.vat_discount, other.vat_discount)
            )
            lbt_portion_combined = np.concatenate((self.lbt_portion, other.lbt_portion))
            lbt_discount_combined = np.concatenate(
                (self.lbt_discount, other.lbt_discount)
            )
            pis_year_combined = np.concatenate((self.pis_year, other.pis_year))
            salvage_value_combined = np.concatenate(
                (self.salvage_value, other.salvage_value)
            )
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
                lbt_portion=lbt_portion_combined,
                lbt_discount=lbt_discount_combined,
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
            expense_year_combined = np.concatenate(
                (self.expense_year, other.expense_year)
            )
            cost_allocation_combined = self.cost_allocation + other.cost_allocation
            description_combined = self.description + other.description
            vat_portion_combined = np.concatenate((self.vat_portion, other.vat_portion))
            vat_discount_combined = np.concatenate(
                (self.vat_discount, other.vat_discount)
            )
            lbt_portion_combined = np.concatenate((self.lbt_portion, other.lbt_portion))
            lbt_discount_combined = np.concatenate(
                (self.lbt_discount, other.lbt_discount)
            )
            pis_year_combined = np.concatenate((self.pis_year, other.pis_year))
            salvage_value_combined = np.concatenate(
                (self.salvage_value, other.salvage_value)
            )
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
                lbt_portion=lbt_portion_combined,
                lbt_discount=lbt_discount_combined,
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
                lbt_portion=self.lbt_portion,
                lbt_discount=self.lbt_discount,
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
        # Between an instance of CapitalCost with another instance of CapitalCost/Intangible/OPEX/ASR
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR)):
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
                    lbt_portion=self.lbt_portion,
                    lbt_discount=self.lbt_discount,
                    pis_year=self.pis_year,
                    salvage_value=self.salvage_value,
                    useful_life=self.useful_life,
                    depreciation_factor=self.depreciation_factor,
                    is_ic_applied=self.is_ic_applied,
                )

        else:
            raise CapitalException(
                f"Must divide with an instance of CapitalCost/Intangible/OPEX/ASR, "
                f"integer or a float; "
                f"{other}({other.__class__.__qualname__}) is not an instance "
                f"of CapitalCost/Intangible/OPEX/ASR nor an integer nor a float."
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
                f"instance of CapitalCost/Intangible/OPEX/ASR nor an integer nor a float."
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
    (1) The unit used in the fixed_cost should be in M unit of United States Dollar (USD),
        where the M is stands for 1000. Thus, the unit cost should be: M-USD.
    """

    fixed_cost: np.ndarray = field(default=None)
    prod_rate: np.ndarray = field(default=None)
    cost_per_volume: np.ndarray = field(default=None)

    # Attribute to be defined later on
    variable_cost: np.ndarray = field(default=None, init=False)
    cost: np.ndarray = field(default=None, init=False, repr=False)

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

        # Configure VAT portion
        if self.vat_portion is None:
            self.vat_portion = np.ones_like(self.fixed_cost)

        if self.vat_portion is not None:
            if not isinstance(self.vat_portion, np.ndarray):
                raise OPEXException(
                    f"Attribute VAT portion must be a numpy ndarray; "
                    f"VAT portion ({self.vat_portion}) is of datatype "
                    f"{self.vat_portion.__class__.__qualname__}."
                )

        # Configure LBT portion
        if self.lbt_portion is None:
            self.lbt_portion = np.ones_like(self.fixed_cost)

        if self.lbt_portion is not None:
            if not isinstance(self.lbt_portion, np.ndarray):
                raise OPEXException(
                    f"Attribute LBT portion must be a numpy ndarray; "
                    f"LBT portion ({self.lbt_portion}) is of datatype "
                    f"{self.lbt_portion.__class__.__qualname__}."
                )

        # Configure VAT discount
        if isinstance(self.vat_discount, float):
            self.vat_discount = np.repeat(self.vat_discount, len(self.fixed_cost))

        # Configure LBT discount
        if isinstance(self.lbt_discount, float):
            self.lbt_discount = np.repeat(self.lbt_discount, len(self.fixed_cost))

        # Configure description data
        if self.description is None:
            self.description = [" " for _ in range(len(self.fixed_cost))]

        if self.description is not None:
            if not isinstance(self.description, list):
                raise OPEXException(
                    f"Attribute description must be a list; "
                    f"description ({self.description.__class__.__qualname__}) "
                    f"is not a list."
                )

        # Configure cost_allocation data
        if self.cost_allocation is None:
            self.cost_allocation = [FluidType.OIL for _ in range(len(self.fixed_cost))]

        if self.cost_allocation is not None:
            if not isinstance(self.cost_allocation, list):
                raise OPEXException(
                    f"Attribute cost_allocation must be a list. "
                    f"cost_allocation ({self.cost_allocation.__class__.__qualname__}) "
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
                    self.lbt_portion,
                    self.lbt_discount,
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
                    f"lbt_portion: {len(self.lbt_portion)}, "
                    f"lbt_discount: {len(self.lbt_discount)}, "
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
                    np.allclose(self.lbt_portion, other.lbt_portion),
                    np.allclose(self.lbt_discount, other.lbt_discount),
                    self.cost_allocation == other.cost_allocation,
                )
            )

        # Between an instance of OPEX and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.fixed_cost, self.variable_cost) == other

        else:
            return False

    def __lt__(self, other):
        # Between an instance of OPEX with another instance of CapitalCost/Intangible/OPEX/ASR
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR)):
            return np.sum(self.cost) < np.sum(other.cost)

        # Between an instance of OPEX and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) < other

        else:
            raise OPEXException(
                f"Must compare an instance of OPEX with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR, an integer, or a float."
            )

    def __le__(self, other):
        # Between an instance of OPEX with another instance of CapitalCost/Intangible/OPEX/ASR
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR)):
            return np.sum(self.cost) <= np.sum(other.cost)

        # Between an instance of OPEX and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) <= other

        else:
            raise OPEXException(
                f"Must compare an instance of OPEX with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR, an integer, or a float."
            )

    def __gt__(self, other):
        # Between an instance of OPEX with another instance of CapitalCost/Intangible/OPEX/ASR
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR)):
            return np.sum(self.cost) > np.sum(other.cost)

        # Between an instance of OPEX and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) > other

        else:
            raise OPEXException(
                f"Must compare an instance of OPEX with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR, an integer, or a float."
            )

    def __ge__(self, other):
        # Between an instance of OPEX with another instance of CapitalCost/Intangible/OPEX/ASR
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR)):
            return np.sum(self.cost) >= np.sum(other.cost)

        # Between an instance of OPEX and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) >= other

        else:
            raise OPEXException(
                f"Must compare an instance of OPEX with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR, an integer, or a float."
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
            lbt_portion_combined = np.concatenate((self.lbt_portion, other.lbt_portion))
            lbt_discount_combined = np.concatenate((self.lbt_discount, other.lbt_discount))
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
                lbt_portion=lbt_portion_combined,
                lbt_discount=lbt_discount_combined,
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
            lbt_portion_combined = np.concatenate((self.lbt_portion, other.lbt_portion))
            lbt_discount_combined = np.concatenate((self.lbt_discount, other.lbt_discount))
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
                lbt_portion=lbt_portion_combined,
                lbt_discount=lbt_discount_combined,
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
                lbt_portion=self.lbt_portion,
                lbt_discount=self.lbt_discount,
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
        # Between an instance of OPEX with another instance of CapitalCost/Intangible/OPEX/ASR
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR)):
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
                    lbt_portion=self.lbt_portion,
                    lbt_discount=self.lbt_discount,
                    fixed_cost=self.fixed_cost / other,
                    prod_rate=self.prod_rate / other,
                    cost_per_volume=self.cost_per_volume,
                )

        else:
            raise OPEXException(
                f"Must divide with an instance of CapitalCost/Intangible/OPEX/ASR, integer or a float; "
                f"{other}({other.__class__.__qualname__}) is not an instance "
                f"of CapitalCost/Intangible/OPEX/ASR nor an integer nor a float."
            )


@dataclass
class ASR(GeneralCost):
    """
    Manages an ASR asset.

    Parameters
    ----------
    The attributes are inherited from class GeneralCost.
    """
    def __post_init__(self):
        # Check for inappropriate start and end year project
        if self.end_year >= self.start_year:
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise ASRException(
                f"start year {self.start_year} "
                f"is after the end year {self.end_year}"
            )

        # Configure VAT portion
        if self.vat_portion is None:
            self.vat_portion = np.ones_like(self.cost)

        if self.vat_portion is not None:
            if not isinstance(self.vat_portion, np.ndarray):
                raise ASRException(
                    f"Attribute VAT portion must be a numpy ndarray; "
                    f"VAT portion ({self.vat_portion}) is of datatype "
                    f"{self.vat_portion.__class__.__qualname__}."
                )

        # Configure LBT portion
        if self.lbt_portion is None:
            self.lbt_portion = np.ones_like(self.cost)

        if self.lbt_portion is not None:
            if not isinstance(self.lbt_portion, np.ndarray):
                raise ASRException(
                    f"Attribute LBT portion must be a numpy ndarray; "
                    f"LBT portion ({self.lbt_portion}) is of datatype "
                    f"{self.lbt_portion.__class__.__qualname__}."
                )

        # Configure VAT discount
        if isinstance(self.vat_discount, float):
            self.vat_discount = np.repeat(self.vat_discount, len(self.cost))

        # Configure LBT discount
        if isinstance(self.lbt_discount, float):
            self.lbt_discount = np.repeat(self.lbt_discount, len(self.cost))

        # Configure description data
        if self.description is None:
            self.description = [" " for _ in range(len(self.cost))]

        if self.description is not None:
            if not isinstance(self.description, list):
                raise ASRException(
                    f"Attribute description must be a list; "
                    f"description (datatype: {self.description.__class__.__qualname__}) "
                    f"is not a list."
                )

        # Configure cost_allocation data
        if self.cost_allocation is None:
            self.cost_allocation = [FluidType.OIL for _ in range(len(self.cost))]

        if self.cost_allocation is not None:
            if not isinstance(self.cost_allocation, list):
                raise ASRException(
                    f"Attribute cost_allocation must be a list. "
                    f"cost_allocation ({self.cost_allocation.__class__.__qualname__}) "
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
                self.lbt_portion,
                self.lbt_discount,
            ]
        ):
            raise ASRException(
                f"Unequal length of array: "
                f"cost: {len(self.cost)}, "
                f"expense_year: {len(self.expense_year)}, "
                f"cost_allocation: {len(self.cost_allocation)}, "
                f"description: {len(self.description)}, "
                f"vat_portion: {len(self.vat_portion)}, "
                f"vat_discount: {len(self.vat_discount)}, "
                f"lbt_portion: {len(self.lbt_portion)}, "
                f"lbt_discount: {len(self.lbt_discount)}."
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
        year_ref: int = None,
        tax_type: TaxType = TaxType.VAT,
        vat_rate: np.ndarray | float = 0.0,
        lbt_rate: np.ndarray | float = 0.0,
        inflation_rate: np.ndarray | float = 0.0,
        future_rate: float = 0.02,
    ) -> np.ndarray:
        """
        Calculate the future cost of an asset.

        Parameters
        ----------
        year_ref : int
            The reference year for inflation calculation.
        tax_type: TaxType
            The type of tax applied to the corresponding asset.
            Available options: TaxType.VAT or TaxType.LBT
        vat_rate: np.ndarray | float
            The VAT rate to apply. Can be a single value or an array (default is 0.0).
        lbt_rate: np.ndarray | float
            The LBT rate to apply. Can be a single value or an array (default is 0.0).
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
        if year_ref is None:
            year_ref = self.start_year

        if not isinstance(future_rate, float):
            raise ASRException(
                f"Future rate must be a float, not a "
                f"{future_rate.__class__.__qualname__}"
            )

        cost_adjusted = apply_cost_adjustment(
            start_year=self.start_year,
            end_year=self.end_year,
            cost=self.cost,
            expense_year=self.expense_year,
            project_years=self.project_years,
            year_ref=year_ref,
            tax_type=tax_type,
            vat_portion=self.vat_portion,
            vat_rate=vat_rate,
            vat_discount=self.vat_discount,
            lbt_portion=self.lbt_portion,
            lbt_rate=lbt_rate,
            lbt_discount=self.lbt_discount,
            inflation_rate=inflation_rate,
        )

        return cost_adjusted * np.power(
            (1 + future_rate), self.end_year - self.expense_year + 1
        )

    def expenditures_post_tax(
        self,
        year_ref: int = None,
        tax_type: TaxType = TaxType.VAT,
        vat_rate: np.ndarray | float = 0.0,
        lbt_rate: np.ndarray | float = 0.0,
        inflation_rate: np.ndarray | float = 0.0,
        future_rate: float = 0.02,
    ):
        """
        Calculate ASR expenditures per year.

        This method calculates the expenditures per year based on the expense year
        and cost data provided.

        Parameters
        ----------
        year_ref : int
            The reference year for inflation calculation.
        tax_type: TaxType
            The type of tax applied to the corresponding asset.
            Available options: TaxType.VAT or TaxType.LBT
        vat_rate: np.ndarray | float
            The VAT rate to apply. Can be a single value or an array (default is 0.0).
        lbt_rate: np.ndarray | float
            The LBT rate to apply. Can be a single value or an array (default is 0.0).
        inflation_rate: np.ndarray | float
            The inflation rate to apply. Can be a single value or an array (default is 0.0).
        future_rate : float, optional
            The future rate used in cost calculation (default is 0.02).

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
        (2) Apply further adjustment to cost due to future_rate,
        (3) Function np.bincount() is used to align the 'cost_adjusted' elements
            according to its corresponding expense year,
        (4) If len(expenses) < project_duration, then add the remaining elements
            with zeros.
        """
        if year_ref is None:
            year_ref = self.start_year

        if not isinstance(future_rate, float):
            raise ASRException(
                f"Future rate must be a float, not a "
                f"{future_rate.__class__.__qualname__}"
            )

        cost_adjusted = apply_cost_adjustment(
            start_year=self.start_year,
            end_year=self.end_year,
            cost=self.cost,
            expense_year=self.expense_year,
            project_years=self.project_years,
            year_ref=year_ref,
            tax_type=tax_type,
            vat_portion=self.vat_portion,
            vat_rate=vat_rate,
            vat_discount=self.vat_discount,
            lbt_portion=self.lbt_portion,
            lbt_rate=lbt_rate,
            lbt_discount=self.lbt_discount,
            inflation_rate=inflation_rate,
        )

        cost_adjusted *= np.power((1 + future_rate), self.end_year - self.expense_year + 1)
        expenses = np.bincount(self.expense_year - self.start_year, weights=cost_adjusted)
        zeros = np.zeros(self.project_duration - len(expenses))

        return np.concatenate((expenses, zeros))

    def proportion(
        self,
        year_ref: int = None,
        tax_type: TaxType = TaxType.VAT,
        vat_rate: np.ndarray | float = 0.0,
        lbt_rate: np.ndarray | float = 0.0,
        inflation_rate: np.ndarray | float = 0.0,
        future_rate: float = 0.02,
    ) -> np.ndarray:
        """
        Allocate ASR expenditures per year.

        Parameters
        ----------
        year_ref : int
            The reference year for inflation calculation.
        tax_type: TaxType
            The type of tax applied to the corresponding asset.
            Available options: TaxType.VAT or TaxType.LBT
        vat_rate: np.ndarray | float
            The VAT rate to apply. Can be a single value or an array (default is 0.0).
        lbt_rate: np.ndarray | float
            The LBT rate to apply. Can be a single value or an array (default is 0.0).
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
        if year_ref is None:
            year_ref = self.start_year

        if not isinstance(future_rate, float):
            raise ASRException(
                f"Future rate must be a float, not a "
                f"{future_rate.__class__.__qualname__}"
            )

        # Distance of expense year from the end year of the project
        cost_duration = self.end_year - self.expense_year + 1

        # Configure future cost
        cost_future = self.future_cost(
            year_ref=year_ref,
            tax_type=tax_type,
            vat_rate=vat_rate,
            lbt_rate=lbt_rate,
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
                    np.allclose(self.lbt_portion, other.lbt_portion),
                    np.allclose(self.lbt_discount, other.lbt_discount),
                    self.cost_allocation == other.cost_allocation,
                )
            )

        # Between an instance of ASR and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) == other

        else:
            return False

    def __lt__(self, other):
        # Between an instance of ASR with another instance of CapitalCost/Intangible/OPEX/ASR
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR)):
            return np.sum(self.cost) < np.sum(other.cost)

        # Between an instance of ASR and an integer/float
        elif isinstance(other, (float, int)):
            return np.sum(self.cost) < other

        else:
            raise ASRException(
                f"Must compare an instance of ASR with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR, an integer, or a float."
            )

    def __le__(self, other):
        # Between an instance of ASR with another instance of CapitalCost/Intangible/OPEX/ASR
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR)):
            return np.sum(self.cost) <= np.sum(other.cost)

        # Between an instance of ASR and an integer/float
        elif isinstance(other, (float, int)):
            return np.sum(self.cost) <= other

        else:
            raise ASRException(
                f"Must compare an instance of ASR with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR, an integer, or a float."
            )

    def __gt__(self, other):
        # Between an instance of ASR with another instance of CapitalCost/Intangible/OPEX/ASR
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR)):
            return np.sum(self.cost) > np.sum(other.cost)

        # Between an instance of ASR and an integer/float
        elif isinstance(other, (float, int)):
            return np.sum(self.cost) > other

        else:
            raise ASRException(
                f"Must compare an instance of ASR with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR, an integer, or a float."
            )

    def __ge__(self, other):
        # Between an instance of ASR with another instance of CapitalCost/Intangible/OPEX/ASR
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR)):
            return np.sum(self.cost) >= np.sum(other.cost)

        # Between an instance of ASR and an integer/float
        elif isinstance(other, (float, int)):
            return np.sum(self.cost) >= other

        else:
            raise ASRException(
                f"Must compare an instance of ASR with another instance of "
                f"CapitalCost/Intangible/OPEX/ASR, an integer, or a float."
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
            lbt_portion_combined = np.concatenate((self.lbt_portion, other.lbt_portion))
            lbt_discount_combined = np.concatenate((self.lbt_discount, other.lbt_discount))

            return ASR(
                start_year=start_year_combined,
                end_year=end_year_combined,
                cost=cost_combined,
                expense_year=expense_year_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
                vat_portion=vat_portion_combined,
                vat_discount=vat_discount_combined,
                lbt_portion=lbt_portion_combined,
                lbt_discount=lbt_discount_combined,
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
            lbt_portion_combined = np.concatenate((self.lbt_portion, other.lbt_portion))
            lbt_discount_combined = np.concatenate((self.lbt_discount, other.lbt_discount))

            return ASR(
                start_year=start_year_combined,
                end_year=end_year_combined,
                cost=cost_combined,
                expense_year=expense_year_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
                vat_portion=vat_portion_combined,
                vat_discount=vat_discount_combined,
                lbt_portion=lbt_portion_combined,
                lbt_discount=lbt_discount_combined,
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
                lbt_portion=self.lbt_portion,
                lbt_discount=self.lbt_discount,
            )

        else:
            raise ASRException(
                f"Must multiply with an integer or a float; "
                f"{other}({other.__class__.__qualname__}) is not an integer nor a float."
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        # Between an instance of ASR with another instance of CapitalCost/Intangible/OPEX/ASR
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR)):
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
                    lbt_portion=self.lbt_portion,
                    lbt_discount=self.lbt_discount,
                )

        else:
            raise ASRException(
                f"Must divide with an instance of CapitalCost/Intangible/OPEX/ASR, "
                f"integer or a float; "
                f"{other}({other.__class__.__qualname__}) is not an instance "
                f"of CapitalCost/Intangible/OPEX/ASR nor an integer nor a float."
            )


@dataclass
class LBT(GeneralCost):
    pass


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
