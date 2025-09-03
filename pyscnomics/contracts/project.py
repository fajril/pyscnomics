"""
Configure base project as the foundation (or parent class) for PSC contract.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import date
from functools import reduce

from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.selection import (
    FluidType,
    CostType,
    OtherRevenue,
    InflationAppliedTo,
)
from pyscnomics.econ.costs import (
    CapitalCost,
    Intangible,
    OPEX,
    ASR,
    LBT,
    CostOfSales,
)
# from pyscnomics.econ.results import CashFlow


class BaseProjectException(Exception):
    """ Exception to be raised for a misuse of BaseProject class """

    pass


class OtherRevenueException(Exception):
    """ Exception to be raised for a misuse of Other Revenue """

    pass


class SunkCostException(Exception):
    """ Exception to be raised for a misuse of _get_sunk_cost() method """

    pass


class PreOnstreamException(Exception):
    """ Exception to be raised for an incorrect preonstream cost configuration """

    pass


class PostOnstreamException(Exception):
    """ Exception to be raised for incorrect postonstream cost configurations """

    pass


@dataclass
class BaseProject:
    """
    Represents a base project with start and end dates, lifting information,
    capital and intangible costs, operational expenses (OPEX), ASR costs,
    LBT costs, and Cost of Sales.

    Parameters
    ----------
    start_date : date
        The start date of the project.
    end_date : date
        The end date of the project.
    oil_onstream_date: date
        The start date of oil production.
    gas_onstream_date: date
        The start date of gas production.
    lifting : tuple[Lifting]
        A tuple of lifting information objects.
    capital_cost : tuple[CapitalCost]
        A tuple of capital cost objects.
    intangible_cost : tuple[Intangible], optional
        A tuple of intangible cost objects. Defaults to None.
    opex : tuple[OPEX], optional
        A tuple of operational expense objects. Defaults to None.
    asr_cost : tuple[ASR], optional
        A tuple of ASR (Abandonment and Site Restoration) cost objects.
        Defaults to None.
    lbt_cost: tuple[LBT], optional
        A tuple of LBT (Land and Building Tax) cost objects.
        Defaults to None.
    cost_of_sales : tuple[CostOfSales]
        A tuple of CostOfSales objects. Defaults to None.
    """

    # List of required arguments
    start_date: date
    end_date: date
    oil_onstream_date: date = field(default=None)
    gas_onstream_date: date = field(default=None)
    approval_year: int = field(default=None)
    lifting: tuple[Lifting, ...] = field(default=None)
    capital_cost: tuple[CapitalCost, ...] = field(default=None)
    intangible_cost: tuple[Intangible, ...] = field(default=None)
    opex: tuple[OPEX, ...] = field(default=None)
    asr_cost: tuple[ASR, ...] = field(default=None)
    lbt_cost: tuple[LBT, ...] = field(default=None)
    cost_of_sales: tuple[CostOfSales, ...] = field(default=None)

    # Attributes associated with project duration
    project_duration: int = field(default=None, init=False)
    project_years: np.ndarray = field(default=None, init=False)

    # Attributes associated with lifting for each fluid types
    _oil_lifting: Lifting = field(default=None, init=False, repr=False)
    _gas_lifting: Lifting = field(default=None, init=False, repr=False)
    _sulfur_lifting: Lifting = field(default=None, init=False, repr=False)
    _electricity_lifting: Lifting = field(default=None, init=False, repr=False)
    _co2_lifting: Lifting = field(default=None, init=False, repr=False)

    # Attributes associated with revenue for each fluid types
    _oil_revenue: np.ndarray = field(default=None, init=False, repr=False)
    _gas_revenue: np.ndarray = field(default=None, init=False, repr=False)
    _sulfur_revenue: np.ndarray = field(default=None, init=False, repr=False)
    _electricity_revenue: np.ndarray = field(default=None, init=False, repr=False)
    _co2_revenue: np.ndarray = field(default=None, init=False, repr=False)

    # Atributes associated with WAP price
    _oil_wap_price: np.ndarray = field(default=None, init=False, repr=False)
    _gas_wap_price: np.ndarray = field(default=None, init=False, repr=False)
    _sulfur_wap_price: np.ndarray = field(default=None, init=False, repr=False)
    _electricity_wap_price: np.ndarray = field(default=None, init=False, repr=False)
    _co2_wap_price: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes associated with total cost per component
    capital_cost_total: CapitalCost = field(default=None, init=False, repr=False)
    intangible_cost_total: Intangible = field(default=None, init=False, repr=False)
    opex_total: OPEX = field(default=None, init=False, repr=False)
    asr_cost_total: ASR = field(default=None, init=False, repr=False)
    lbt_cost_total: LBT = field(default=None, init=False, repr=False)
    cost_of_sales_total: CostOfSales = field(default=None, init=False, repr=False)

    # Attributes associated with post onstream cost in each cost categories
    _oil_capital_postonstream: CapitalCost = field(default=None, init=False, repr=False)
    _gas_capital_postonstream: CapitalCost = field(default=None, init=False, repr=False)
    _oil_intangible_postonstream: Intangible = field(default=None, init=False, repr=False)
    _gas_intangible_postonstream: Intangible = field(default=None, init=False, repr=False)
    _oil_opex_postonstream: OPEX = field(default=None, init=False, repr=False)
    _gas_opex_postonstream: OPEX = field(default=None, init=False, repr=False)
    _oil_asr_postonstream: ASR = field(default=None, init=False, repr=False)
    _gas_asr_postonstream: ASR = field(default=None, init=False, repr=False)
    _oil_lbt_postonstream: LBT = field(default=None, init=False, repr=False)
    _gas_lbt_postonstream: LBT = field(default=None, init=False, repr=False)
    _oil_cost_of_sales_postonstream: CostOfSales = field(default=None, init=False, repr=False)
    _gas_cost_of_sales_postonstream: CostOfSales = field(default=None, init=False, repr=False)

    # Attributes associated with preonstream cost in each cost categories
    _oil_capital_preonstream: CapitalCost = field(default=None, init=False, repr=False)
    _gas_capital_preonstream: CapitalCost = field(default=None, init=False, repr=False)
    _oil_intangible_preonstream: Intangible = field(default=None, init=False, repr=False)
    _gas_intangible_preonstream: Intangible = field(default=None, init=False, repr=False)
    _oil_opex_preonstream: OPEX = field(default=None, init=False, repr=False)
    _gas_opex_preonstream: OPEX = field(default=None, init=False, repr=False)
    _oil_asr_preonstream: ASR = field(default=None, init=False, repr=False)
    _gas_asr_preonstream: ASR = field(default=None, init=False, repr=False)
    _oil_lbt_preonstream: LBT = field(default=None, init=False, repr=False)
    _gas_lbt_preonstream: LBT = field(default=None, init=False, repr=False)
    _oil_cost_of_sales_preonstream: CostOfSales = field(default=None, init=False, repr=False)
    _gas_cost_of_sales_preonstream: CostOfSales = field(default=None, init=False, repr=False)

    # Attributes associated with sunkcost in each cost categories
    _oil_capital_sunk_cost: CapitalCost = field(default=None, init=False, repr=False)
    _gas_capital_sunk_cost: CapitalCost = field(default=None, init=False, repr=False)
    _oil_intangible_sunk_cost: Intangible = field(default=None, init=False, repr=False)
    _gas_intangible_sunk_cost: Intangible = field(default=None, init=False, repr=False)
    _oil_opex_sunk_cost: OPEX = field(default=None, init=False, repr=False)
    _gas_opex_sunk_cost: OPEX = field(default=None, init=False, repr=False)
    _oil_asr_sunk_cost: ASR = field(default=None, init=False, repr=False)
    _gas_asr_sunk_cost: ASR = field(default=None, init=False, repr=False)
    _oil_lbt_sunk_cost: LBT = field(default=None, init=False, repr=False)
    _gas_lbt_sunk_cost: LBT = field(default=None, init=False, repr=False)
    _oil_cost_of_sales_sunk_cost: CostOfSales = field(default=None, init=False, repr=False)
    _gas_cost_of_sales_sunk_cost: CostOfSales = field(default=None, init=False, repr=False)

    # Attributes associated with preonstream costs
    _oil_depreciable_preonstream: np.ndarray = field(default=None, init=False, repr=False)
    _gas_depreciable_preonstream: np.ndarray = field(default=None, init=False, repr=False)
    _oil_non_depreciable_preonstream: np.ndarray = field(default=None, init=False, repr=False)
    _gas_non_depreciable_preonstream: np.ndarray = field(default=None, init=False, repr=False)
    _oil_preonstream: np.ndarray = field(default=None, init=False, repr=False)
    _gas_preonstream: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes associated with sunk costs
    _oil_depreciable_sunk_cost: np.ndarray = field(default=None, init=False, repr=False)
    _gas_depreciable_sunk_cost: np.ndarray = field(default=None, init=False, repr=False)
    _oil_non_depreciable_sunk_cost: np.ndarray = field(default=None, init=False, repr=False)
    _gas_non_depreciable_sunk_cost: np.ndarray = field(default=None, init=False, repr=False)
    _oil_sunk_cost: np.ndarray = field(default=None, init=False, repr=False)
    _gas_sunk_cost: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes associated with pre tax expenditures for each cost categories
    _oil_capital_expenditures_pre_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _gas_capital_expenditures_pre_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _oil_intangible_expenditures_pre_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _gas_intangible_expenditures_pre_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _oil_opex_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_opex_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    _oil_asr_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_asr_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    _oil_lbt_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_lbt_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    _oil_cost_of_sales_expenditures_pre_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _gas_cost_of_sales_expenditures_pre_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )

    # Attributes associated with indirect taxes for each cost categories
    _oil_capital_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_capital_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    _oil_intangible_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_intangible_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    _oil_opex_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_opex_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    _oil_asr_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_asr_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    _oil_lbt_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_lbt_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    _oil_cost_of_sales_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_cost_of_sales_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes associated with post tax expenditures for each cost categories
    _oil_capital_expenditures_post_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _gas_capital_expenditures_post_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _oil_intangible_expenditures_post_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _gas_intangible_expenditures_post_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _oil_opex_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_opex_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    _oil_asr_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_asr_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    _oil_lbt_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_lbt_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    _oil_cost_of_sales_expenditures_post_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _gas_cost_of_sales_expenditures_post_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )

    # Attributes associated with total expenditures and indirect taxes for each fluid
    _oil_total_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_total_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    _oil_total_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_total_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    _oil_total_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_total_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes associated with non capital costs
    _oil_non_capital: np.ndarray = field(default=None, init=False, repr=False)
    _gas_non_capital: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes associated with cashflow
    _oil_cashflow: np.ndarray = field(default=None, init=False, repr=False)
    _gas_cashflow: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes associated with consolidated profiles
    _consolidated_lifting: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_wap_price: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_revenue: np.ndarray = field(default=None, init=False, repr=False)

    _consolidated_depreciable_sunk_cost: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_non_depreciable_sunk_cost: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_sunk_cost: np.ndarray = field(default=None, init=False, repr=False)

    _consolidated_depreciable_preonstream: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_non_depreciable_preonstream: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_preonstream: np.ndarray = field(default=None, init=False, repr=False)

    _consolidated_capital_expenditures_pre_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_intangible_expenditures_pre_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_opex_expenditures_pre_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_asr_expenditures_pre_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_lbt_expenditures_pre_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_cost_of_sales_expenditures_pre_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_total_expenditures_pre_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )

    _consolidated_capital_indirect_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_intangible_indirect_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_opex_indirect_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_asr_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_lbt_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_cost_of_sales_indirect_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_total_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)

    _consolidated_capital_expenditures_post_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_intangible_expenditures_post_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_opex_expenditures_post_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_asr_expenditures_post_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_lbt_expenditures_post_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_cost_of_sales_expenditures_post_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_total_expenditures_post_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )

    _consolidated_non_capital: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_cashflow: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """
        Handles the following operations/procedures:
        -   Prepare attributes project_duration and project_years;
        -   Prepare attribute lifting;
        -   Prepare attributes associated with lifting for each fluid types.
            The attributes are: _oil_lifting, _gas_lifting, _sulfur_lifting,
            _electricity_lifting, and _co2_lifting;
        -   Prepare attributes associated with revenue for each fluid types.
            The attributes are: _oil_revenue, _gas_revenue, _sulfur_revenue,
            _electricity_revenue, and _co2_revenue;
        -   Prepare attribute oil_onstream_date;
        -   Prepare attribute gas_onstream_date;
        -   Prepare attribute capital_cost;
        -   Prepare attribute intangible_cost;
        -   Prepare attribute opex;
        -   Prepare attribute asr_cost;
        -   Prepare attribute lbt_cost;
        -   Prepare attribute cost_of_sales;
        -   Prepare attributes associated with total cost per component
            (leveraging dunder method __add__). The attributes are:
            capital_cost_total, intangible_cost_total, opex_total, asr_cost_total,
            lbt_cost_total, and cost_of_sales_total;
        -   Classify cost categories by fluid
        -   Modify cost_type in each cost categories, accounting for engineering sense
        -   Define postonstream, sunk cost, and preonstream attributes
        -   Raise an exception if the start year of the project is inconsistent;
        -   Raise an exception if the end year of the project is inconsistent.
        """

        # Prepare attributes project_duration and project_years
        if self.start_date <= self.end_date:
            self.project_duration = self.end_date.year - self.start_date.year + 1
            self.project_years = np.arange(
                self.start_date.year, self.end_date.year + 1, 1
            )

        else:
            raise BaseProjectException(
                f"start date {self.start_date} "
                f"is after the end date: {self.end_date}"
            )

        # Prepare attribute lifting (for OIL, GAS, SULFUR, ELECTRICITY, and CO2)
        if self.lifting is None:
            self.lifting = (
                Lifting(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    prod_year=self.project_years.copy(),
                    lifting_rate=np.zeros(self.project_duration),
                    price=np.zeros(self.project_duration),
                    fluid_type=FluidType.OIL,
                ),
                Lifting(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    prod_year=self.project_years.copy(),
                    lifting_rate=np.zeros(self.project_duration),
                    price=np.zeros(self.project_duration),
                    fluid_type=FluidType.GAS,
                ),
                Lifting(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    prod_year=self.project_years.copy(),
                    lifting_rate=np.zeros(self.project_duration),
                    price=np.zeros(self.project_duration),
                    fluid_type=FluidType.SULFUR,
                ),
                Lifting(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    prod_year=self.project_years.copy(),
                    lifting_rate=np.zeros(self.project_duration),
                    price=np.zeros(self.project_duration),
                    fluid_type=FluidType.ELECTRICITY,
                ),
                Lifting(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    prod_year=self.project_years.copy(),
                    lifting_rate=np.zeros(self.project_duration),
                    price=np.zeros(self.project_duration),
                    fluid_type=FluidType.CO2,
                ),
            )

        else:
            if not isinstance(self.lifting, tuple):
                raise BaseProjectException(
                    f"Attribute lifting must be provided as a tuple of Lifting instances, "
                    f"not as an/a {self.lifting.__class__.__qualname__}"
                )

        # Prepare attributes associated with lifting for each fluid types
        self._oil_lifting = self._get_oil_lifting()
        self._gas_lifting = self._get_gas_lifting()
        self._sulfur_lifting = self._get_sulfur_lifting()
        self._electricity_lifting = self._get_electricity_lifting()
        self._co2_lifting = self._get_co2_lifting()

        # Prepare attributes associated with revenues for each fluid types
        self._oil_revenue = self._oil_lifting.revenue()
        self._gas_revenue = self._gas_lifting.revenue()
        self._sulfur_revenue = self._sulfur_lifting.revenue()
        self._electricity_revenue = self._electricity_lifting.revenue()
        self._co2_revenue = self._co2_lifting.revenue()

        # Prepare attribute oil_onstream_date: set default value and error message
        oil_revenue_index = np.argwhere(self._oil_revenue > 0).ravel()

        if len(oil_revenue_index) > 0:
            if self.oil_onstream_date is not None:
                if self.oil_onstream_date.year < self.start_date.year:
                    raise BaseProjectException(
                        f"Oil onstream year ({self.oil_onstream_date.year}) is before "
                        f"the start project year ({self.start_date.year})"
                    )

                if self.oil_onstream_date.year > self.end_date.year:
                    raise BaseProjectException(
                        f"Oil onstream year ({self.oil_onstream_date.year}) is after "
                        f"the end year of the project ({self.end_date.year})"
                    )

                # Ensure oil_onstream_date provided by the user is consistent
                # with the beginning of oil lifting, indicated by the first year
                # of positive oil revenue
                oil_onstream_index = int(
                    np.argwhere(self.oil_onstream_date.year == self.project_years).ravel()
                )

                if oil_onstream_index != oil_revenue_index[0]:
                    raise BaseProjectException(
                        f"Oil onstream year ({self.oil_onstream_date.year}) is different "
                        f"from the starting year of oil production "
                        f"({self.project_years[oil_revenue_index[0]]})"
                    )

            else:
                self.oil_onstream_date = date(
                    year=self.project_years[oil_revenue_index[0]], month=1, day=1
                )

        else:
            self.oil_onstream_date = self.end_date

        # Prepare attribute gas_onstream_date: set default value and error message
        gas_revenue_index = np.argwhere(self._gas_revenue > 0).ravel()

        if len(gas_revenue_index) > 0:
            if self.gas_onstream_date is not None:
                if self.gas_onstream_date.year < self.start_date.year:
                    raise BaseProjectException(
                        f"Gas onstream year ({self.gas_onstream_date.year}) is before "
                        f"the start project year ({self.start_date.year})"
                    )

                if self.gas_onstream_date.year > self.end_date.year:
                    raise BaseProjectException(
                        f"Gas onstream year ({self.gas_onstream_date.year}) is after "
                        f"the end year of the project ({self.end_date.year})"
                    )

                # Ensure gas_onstream_date provided by the user is consistent
                # with the beginning of gas lifting, indicated by the first year
                # of positive gas revenue
                gas_onstream_index = int(
                    np.argwhere(self.gas_onstream_date.year == self.project_years).ravel()
                )

                if gas_onstream_index != gas_revenue_index[0]:
                    raise BaseProjectException(
                        f"Gas onstream year ({self.gas_onstream_date.year}) is different "
                        f"from the starting year of gas production "
                        f"({self.project_years[gas_revenue_index[0]]})"
                    )

            else:
                self.gas_onstream_date = date(
                    year=int(self.project_years[gas_revenue_index[0]]), month=1, day=1
                )

        else:
            self.gas_onstream_date = self.end_date

        # Prepare attribute capital_cost (for both OIL and GAS)
        if self.capital_cost is None:
            self.capital_cost = (
                CapitalCost(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    expense_year=np.array([self.start_date.year]),
                    cost=np.array([0]),
                    cost_allocation=[FluidType.OIL],
                ),
                CapitalCost(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    expense_year=np.array([self.start_date.year]),
                    cost=np.array([0]),
                    cost_allocation=[FluidType.GAS],
                ),
            )

        else:
            if not isinstance(self.capital_cost, tuple):
                raise BaseProjectException(
                    f"Attribute capital_cost must be provided as a tuple of CapitalCost "
                    f"instances, not as an/a {self.capital_cost.__class__.__qualname__}"
                )

        # Prepare attribute intangible_cost (for both OIL and GAS)
        if self.intangible_cost is None:
            self.intangible_cost = (
                Intangible(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    expense_year=np.array([self.start_date.year]),
                    cost=np.array([0]),
                    cost_allocation=[FluidType.OIL],
                ),
                Intangible(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    expense_year=np.array([self.start_date.year]),
                    cost=np.array([0]),
                    cost_allocation=[FluidType.GAS],
                ),
            )

        else:
            if not isinstance(self.intangible_cost, tuple):
                raise BaseProjectException(
                    f"Attribute intangible_cost must be provided as a tuple of Intangible "
                    f"instances, not as an/a {self.intangible_cost.__class__.__qualname__}"
                )

        # Prepare attribute opex (for both OIL and GAS)
        if self.opex is None:
            self.opex = (
                OPEX(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    expense_year=np.array([self.start_date.year]),
                    fixed_cost=np.array([0]),
                    cost_allocation=[FluidType.OIL],
                ),
                OPEX(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    expense_year=np.array([self.start_date.year]),
                    fixed_cost=np.array([0]),
                    cost_allocation=[FluidType.GAS],
                ),
            )

        else:
            if not isinstance(self.opex, tuple):
                raise BaseProjectException(
                    f"Attribute opex must be provided as a tuple of OPEX instances, "
                    f"not as an/a {self.opex.__class__.__qualname__}"
                )

        # Prepare attribute asr_cost (for both OIL and GAS)
        if self.asr_cost is None:
            self.asr_cost = (
                ASR(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    expense_year=np.array([self.start_date.year]),
                    cost=np.array([0]),
                    cost_allocation=[FluidType.OIL],
                ),
                ASR(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    expense_year=np.array([self.start_date.year]),
                    cost=np.array([0]),
                    cost_allocation=[FluidType.GAS],
                ),
            )

        else:
            if not isinstance(self.asr_cost, tuple):
                raise BaseProjectException(
                    f"Attribute asr_cost must be provided as a tuple of ASR instances, "
                    f"not as an/a {self.asr_cost.__class__.__qualname__}"
                )

        # Prepare attribute lbt_cost (for both OIL and GAS)
        if self.lbt_cost is None:
            self.lbt_cost = (
                LBT(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    expense_year=np.array([self.start_date.year]),
                    cost=np.array([0]),
                    cost_allocation=[FluidType.OIL],
                ),
                LBT(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    expense_year=np.array([self.start_date.year]),
                    cost=np.array([0]),
                    cost_allocation=[FluidType.GAS],
                ),
            )

        else:
            if not isinstance(self.lbt_cost, tuple):
                raise BaseProjectException(
                    f"Attribute lbt_cost must be provided as a tuple of LBT instances, "
                    f"not as an/a {self.lbt_cost.__class__.__qualname__}"
                )

        # Prepare attribute cost_of_sales (for both OIL and GAS)
        if self.cost_of_sales is None:
            self.cost_of_sales = (
                CostOfSales(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    expense_year=np.array([self.start_date.year]),
                    cost=np.array([0]),
                    cost_allocation=[FluidType.OIL],
                ),
                CostOfSales(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    expense_year=np.array([self.start_date.year]),
                    cost=np.array([0]),
                    cost_allocation=[FluidType.GAS],
                ),
            )

        else:
            if not isinstance(self.cost_of_sales, tuple):
                raise BaseProjectException(
                    f"Attribute cost_of_sales must be provided as a tuple of CostOfSales "
                    f"instances, not as an/a {self.cost_of_sales.__class__.__qualname__}"
                )

        # Prepare attributes associated with total cost per component
        self.capital_cost_total = reduce(lambda x, y: x + y, self.capital_cost)
        self.intangible_cost_total = reduce(lambda x, y: x + y, self.intangible_cost)
        self.opex_total = reduce(lambda x, y: x + y, self.opex)
        self.asr_cost_total = reduce(lambda x, y: x + y, self.asr_cost)
        self.lbt_cost_total = reduce(lambda x, y: x + y, self.lbt_cost)
        self.cost_of_sales_total = reduce(lambda x, y: x + y, self.cost_of_sales)

        # Classify cost categories by fluid
        (
            capital_cost,
            intangible,
            opex,
            asr,
            lbt,
            cost_of_sales,
        ) = [
            self._classify_costs_by_fluid(classifier=func)
            for func in [
                self._classify_capital_cost_by_fluid,
                self._classify_intangible_cost_by_fluid,
                self._classify_opex_by_fluid,
                self._classify_asr_cost_by_fluid,
                self._classify_lbt_cost_by_fluid,
                self._classify_cost_of_sales_by_fluid,
            ]
        ]

        # Modify cost_type in each cost categories, accounting for engineering sense
        costs_list = [
            capital_cost,
            intangible,
            opex,
            asr,
            lbt,
            cost_of_sales,
        ]

        for cost in costs_list:
            for ftype in [FluidType.OIL, FluidType.GAS]:
                self._prepare_cost_types(cost_obj=cost[ftype.name.lower()])

        # Define post-onstream cost, pre-onstream cost, and sunk cost attributes
        costs_mapping = (
            ("capital", self._filter_capital_cost, capital_cost),
            ("intangible", self._filter_intangible, intangible),
            ("opex", self._filter_opex, opex),
            ("asr", self._filter_asr, asr),
            ("lbt", self._filter_lbt, lbt),
            ("cost_of_sales", self._filter_cost_of_sales, cost_of_sales),
        )

        fluid_types = (
            FluidType.OIL.name.lower(),
            FluidType.GAS.name.lower(),
        )

        categories = (
            ("postonstream", CostType.POST_ONSTREAM_COST),
            ("preonstream", CostType.PRE_ONSTREAM_COST),
            ("sunk_cost", CostType.SUNK_COST),
        )

        for prefix, filter_func, source in costs_mapping:
            for ftype in fluid_types:
                for categ_name, categ_type in categories:
                    setattr(
                        self,
                        f"_{ftype}_{prefix}_{categ_name}",
                        filter_func(cost_obj_fluid=source[ftype], include_cost_type=categ_type)
                    )

        # Raise an exception error if the start year of the project is inconsistent
        self._check_inconsistent_start_year()

        # Raise an exception error if the end year of the project is inconsistent
        self._check_inconsistent_end_year()

    def _get_lifting_by_commodity(self, commodity: FluidType) -> Lifting:
        """
        Get the aggregated lifting data for a specific commodity type.

        If the commodity doesn't exist in the project's lifting data, returns a zero-filled
        Lifting object with the project's time parameters. Otherwise, returns the sum of all
        lifting data for the specified commodity.

        Parameters
        ----------
        commodity : FluidType
            The fluid commodity type to retrieve the associated lifting data.
            Options:    - FluidType.OIL
                        - FluidType.GAS
                        - FluidType.SULFUR
                        - FluidType.ELECTRICITY
                        - FluidType.CO2

        Returns
        -------
        Lifting
            A Lifting object containing either:
            - Zero-filled arrays if the commodity doesn't exist in lifting data.
            - The sum of all lifting data for the specified commodity.

        Notes
        -----
        The sum of all lifting instances of a particular commodity is carried out
        using dunder method __add__() of class Lifting.
        """

        fluid_types = [lft.fluid_type for lft in self.lifting]

        if commodity not in fluid_types:
            return Lifting(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                prod_year=self.project_years,
                lifting_rate=np.zeros(self.project_duration),
                price=np.zeros(self.project_duration),
                fluid_type=commodity,
                prod_rate=np.zeros(self.project_duration),
                prod_rate_baseline=np.zeros(self.project_duration),
            )

        return reduce(
            lambda x, y: x + y, (lft for lft in self.lifting if lft.fluid_type == commodity)
        )

    def _get_oil_lifting(self) -> Lifting:
        """
        Get the aggregated lifting data for oil commodity.
        Equivalent to calling `_get_lifting_by_commodity(FluidType.OIL)`

        Returns
        -------
        Lifting
            A Lifting object containing either:
            - Zero-filled arrays if no oil lifting data exists
            - The sum of all oil lifting data if available

        See Also
        --------
        _get_commodity_lifting : The general method for any fluid type
        """
        return self._get_lifting_by_commodity(commodity=FluidType.OIL)

    def _get_gas_lifting(self) -> Lifting:
        """
        Get the aggregated lifting data for gas commodity.
        Equivalent to calling `_get_lifting_by_commodity(FluidType.GAS)`

        Returns
        -------
        Lifting
            A Lifting object containing either:
            - Zero-filled arrays if no gas lifting data exists
            - The sum of all gas lifting data if available

        See Also
        --------
        _get_commodity_lifting : The general method for any fluid type
        """
        return self._get_lifting_by_commodity(commodity=FluidType.GAS)

    def _get_sulfur_lifting(self) -> Lifting:
        """
        Get the aggregated lifting data for sulfur commodity.
        Equivalent to calling `_get_lifting_by_commodity(FluidType.SULFUR)`

        Returns
        -------
        Lifting
            A Lifting object containing either:
            - Zero-filled arrays if no sulfur lifting data exists
            - The sum of all sulfur lifting data if available

        See Also
        --------
        _get_commodity_lifting : The general method for any fluid type
        """
        return self._get_lifting_by_commodity(commodity=FluidType.SULFUR)

    def _get_electricity_lifting(self) -> Lifting:
        """
        Get the aggregated lifting data for electricity commodity.
        Equivalent to calling `_get_lifting_by_commodity(FluidType.ELECTRICITY)`.

        Returns
        -------
        Lifting
            A Lifting object containing either:
            - Zero-filled arrays if no electricity production data exists
            - The sum of all electricity production data if available

        See Also
        --------
        _get_commodity_lifting : The general method for any fluid type
        """
        return self._get_lifting_by_commodity(commodity=FluidType.ELECTRICITY)

    def _get_co2_lifting(self) -> Lifting:
        """
        Get the aggregated lifting data for CO₂ commodity.
        Equivalent to calling `_get_lifting_by_commodity(FluidType.CO2)`.

        Returns
        -------
        Lifting
            A Lifting object containing either:
            - Zero-filled arrays if no CO₂ data exists
            - The sum of all CO₂ lifting data if available

        See Also
        --------
        _get_commodity_lifting : The general method for any fluid type
        """
        return self._get_lifting_by_commodity(commodity=FluidType.CO2)

    def _calc_wap_price(self, fluidtype: FluidType) -> np.ndarray:
        """
        Compute the Weighted Average Price (WAP) for a given fluid type.

        This method calculates the WAP price by weighting the price of the fluid type
        by its lifting volume over the project duration. If the total volume is zero
        for any time step, the corresponding WAP price is set to zero.

        Parameters
        ----------
        fluidtype : FluidType
            The fluid type (e.g., OIL, GAS, SULFUR) for which to compute the WAP price.

        Returns
        -------
        np.ndarray
            A NumPy array representing the WAP price for each project year.

        Notes
        -----
        - The calculation considers only lifting instances that match the specified
          fluid type.
        - If `total_vol` is zero for any time step, the result is explicitly set to
          zero to avoid division errors.
        """
        # Initialize attributes with zero values with length = project_duration
        vol_x_price = np.zeros_like(self.project_years, dtype=np.float64)
        total_vol = np.zeros_like(self.project_years, dtype=np.float64)

        # Add contributions from every FLUID TYPE lifting instances
        for lft in self.lifting:
            if lft.fluid_type == fluidtype:
                lft_rate = lft.get_lifting_rate_ghv_arr()
                vol_x_price += lft_rate * lft.get_price_arr()
                total_vol += lft_rate

        # Calculate FLUID WAP price
        return np.divide(
            vol_x_price, total_vol, where=(total_vol > 0), out=np.zeros_like(total_vol)
        )

    def _get_oil_wap_price(self) -> None:
        """
        Compute and store the Weighted Average Price (WAP) for oil.

        This method calculates the WAP price for oil by calling `_calc_wap_price`
        with `FluidType.OIL` and stores the result in `self._oil_wap_price`.

        Returns
        -------
        None
            The computed oil WAP price is stored in `self._oil_wap_price`.

        Notes
        -----
        - The WAP price is computed based on the lifting rate and price for oil.
        - The stored `_oil_wap_price` is a NumPy array representing the WAP price
          for each project year.
        """

        self._oil_wap_price = self._calc_wap_price(fluidtype=FluidType.OIL)

    def _get_gas_wap_price(self) -> None:
        """
        Compute and store the Weighted Average Price (WAP) for gas.

        This method calculates the WAP price for gas by calling `_calc_wap_price`
        with `FluidType.GAS` and stores the result in `self._gas_wap_price`.

        Returns
        -------
        None
            The computed gas WAP price is stored in `self._gas_wap_price`.

        Notes
        -----
        - The WAP price is computed based on the lifting rate and price for gas.
        - The stored `_gas_wap_price` is a NumPy array representing the WAP price
          for each project year.
        """

        self._gas_wap_price = self._calc_wap_price(fluidtype=FluidType.GAS)

    def _get_sulfur_wap_price(self) -> None:
        """
        Compute and store the Weighted Average Price (WAP) for sulfur.

        This method calculates the WAP price for sulfur by calling `_calc_wap_price`
        with `FluidType.SULFUR` and stores the result in `self._sulfur_wap_price`.

        Returns
        -------
        None
            The computed sulfur WAP price is stored in `self._sulfur_wap_price`.

        Notes
        -----
        - The WAP price is computed based on the lifting rate and price for sulfur.
        - The stored `_sulfur_wap_price` is a NumPy array representing the WAP price
          for each project year.
        """

        self._sulfur_wap_price = self._calc_wap_price(fluidtype=FluidType.SULFUR)

    def _get_electricity_wap_price(self) -> None:
        """
        Compute and store the Weighted Average Price (WAP) for electricity.

        This method calculates the WAP price for electricity by calling `_calc_wap_price`
        with `FluidType.ELECTRICITY` and stores the result in `self._electricity_wap_price`.

        Returns
        -------
        None
            The computed electricity WAP price is stored in `self._electricity_wap_price`.

        Notes
        -----
        - The WAP price is computed based on the lifting rate and price for electricity.
        - The stored `_electricity_wap_price` is a NumPy array representing the WAP price
          for each project year.
        """

        self._electricity_wap_price = self._calc_wap_price(fluidtype=FluidType.ELECTRICITY)

    def _get_co2_wap_price(self) -> None:
        """
        Compute and store the Weighted Average Price (WAP) for CO₂.

        This method calculates the WAP price for CO₂ by calling `_calc_wap_price`
        with `FluidType.CO2` and stores the result in `self._co2_wap_price`.

        Returns
        -------
        None
            The computed CO₂ WAP price is stored in `self._co2_wap_price`.

        Notes
        -----
        - The WAP price is computed based on the lifting rate and price for CO₂.
        - The stored `_co2_wap_price` is a NumPy array representing the WAP price
          for each project year.
        """

        self._co2_wap_price = self._calc_wap_price(fluidtype=FluidType.CO2)

    def _get_wap_price(self) -> None:
        """
        Compute and store the Weighted Average Price (WAP) for all fluid types.

        This method sequentially calculates the WAP prices for oil, gas, sulfur,
        electricity, and CO₂ by calling their respective WAP calculation methods.

        The following methods are invoked:
        -   `_get_oil_wap_price`: Computes the WAP for oil.
        -   `_get_gas_wap_price`: Computes the WAP for gas.
        -   `_get_sulfur_wap_price`: Computes the WAP for sulfur.
        -   `_get_electricity_wap_price`: Computes the WAP for electricity.
        -   `_get_co2_wap_price`: Computes the WAP for CO2.

        Returns
        -------
        None
            The computed WAP prices for all fluid types are stored in their respective
            attributes.

            The method updates the following attributes with their corresponding WAP values:
            -   `self._oil_wap_price`
            -   `self._gas_wap_price`
            -   `self._sulfur_wap_price`
            -   `self._electricity_wap_price`
            -   `self._co2_wap_price`

        Notes
        -----
        - This method calls `_get_oil_wap_price`, `_get_gas_wap_price`, `_get_sulfur_wap_price`,
          `_get_electricity_wap_price`, and `_get_co2_wap_price` to compute WAP prices.
        - Each fluid type's WAP price is stored as a NumPy array in its corresponding attribute.
        """

        self._get_oil_wap_price()
        self._get_gas_wap_price()
        self._get_sulfur_wap_price()
        self._get_electricity_wap_price()
        self._get_co2_wap_price()

    def _classify_capital_cost_by_fluid(self, fluid_type: FluidType) -> CapitalCost:
        """
        Classify and filter capital costs by fluid type (oil or gas).

        This method extracts the portion of the total capital cost
        (`capital_cost_total`) that is allocated to a given fluid type.
        If the specified fluid type does not exist in the cost allocation,
        a zero-cost `CapitalCost` object is returned as a placeholder.

        Parameters
        ----------
        fluid_type : FluidType
            The fluid type to filter capital costs by.
            Typically one of ``FluidType.OIL`` or ``FluidType.GAS``.

        Returns
        -------
        CapitalCost
            A `CapitalCost` object containing only the costs allocated
            to the specified fluid type. If no allocation exists for
            the fluid, a zero-cost object with default values is returned.

        Notes
        -----
        - The filtering is performed by masking the `cost_allocation` array
          of the total capital cost object.
        - If the fluid type is not found, a placeholder is returned with:
          ``expense_year = [start_year]`` and ``cost = [0]``.
        """

        cct = self.capital_cost_total

        kwargs = {
            "start_year": cct.start_year,
            "end_year": cct.end_year,
        }

        if fluid_type not in cct.cost_allocation:
            return CapitalCost(
                **kwargs,
                expense_year=np.array([cct.start_year]),
                cost=np.array([0]),
                cost_allocation=[fluid_type],
            )

        else:
            allocation_array = np.array(cct.cost_allocation)
            mask = (allocation_array == fluid_type)

            return CapitalCost(
                **kwargs,
                expense_year=cct.expense_year[mask],
                cost=cct.cost[mask],
                cost_allocation=allocation_array[mask].tolist(),
                cost_type=np.array(cct.cost_type)[mask].tolist(),
                description=np.array(cct.description)[mask].tolist(),
                tax_portion=cct.tax_portion[mask],
                tax_discount=cct.tax_discount[mask],
                pis_year=cct.pis_year[mask],
                salvage_value=cct.salvage_value[mask],
                useful_life=cct.useful_life[mask],
                depreciation_factor=cct.depreciation_factor[mask],
                is_ic_applied=np.array(cct.is_ic_applied)[mask].tolist(),
            )

    def _classify_intangible_cost_by_fluid(self, fluid_type: FluidType) -> Intangible:
        """
        Classify and filter intangible costs by fluid type (oil or gas).

        This method extracts the portion of the total intangible cost
        (`intangible_cost_total`) that is allocated to a given fluid type.
        If the specified fluid type does not exist in the cost allocation,
        a zero-cost `Intangible` object is returned as a placeholder.

        Parameters
        ----------
        fluid_type : FluidType
            The fluid type to filter intangible costs by.
            Typically one of ``FluidType.OIL`` or ``FluidType.GAS``.

        Returns
        -------
        Intangible
            An `Intangible` object containing only the costs allocated
            to the specified fluid type. If no allocation exists for
            the fluid, a zero-cost object with default values is returned.

        Notes
        -----
        - The filtering is performed by masking the `cost_allocation` array
          of the total intangible cost object.
        - If the fluid type is not found, a placeholder is returned with:
          ``expense_year = [start_year]`` and ``cost = [0]``.
        """

        ict = self.intangible_cost_total

        kwargs = {
            "start_year": ict.start_year,
            "end_year": ict.end_year,
        }

        if fluid_type not in ict.cost_allocation:
            return Intangible(
                **kwargs,
                expense_year=np.array([ict.start_year]),
                cost=np.array([0]),
                cost_allocation=[fluid_type],
            )

        else:
            allocation_array = np.array(ict.cost_allocation)
            mask = (allocation_array == fluid_type)

            return Intangible(
                **kwargs,
                expense_year=ict.expense_year[mask],
                cost=ict.cost[mask],
                cost_allocation=allocation_array[mask].tolist(),
                description=np.array(ict.description)[mask].tolist(),
                cost_type=np.array(ict.cost_type)[mask].tolist(),
                tax_portion=ict.tax_portion[mask],
                tax_discount=ict.tax_discount[mask],
            )

    def _classify_opex_by_fluid(self, fluid_type: FluidType) -> OPEX:
        """
        Classify and filter operating expenditure (OPEX) by fluid type (oil or gas).

        This method extracts the portion of the total OPEX (`opex_total`)
        that is allocated to a given fluid type. If the specified fluid type
        does not exist in the cost allocation, a zero-cost `OPEX` object is
        returned as a placeholder.

        Parameters
        ----------
        fluid_type : FluidType
            The fluid type to filter OPEX costs by.
            Typically one of ``FluidType.OIL`` or ``FluidType.GAS``.

        Returns
        -------
        OPEX
            An `OPEX` object containing only the costs allocated to the specified
            fluid type. If no allocation exists for the fluid, a zero-cost object
            with default values is returned.

        Notes
        -----
        - The filtering is performed by masking the `cost_allocation` array
          of the total OPEX object.
        - If the fluid type is not found, a placeholder is returned with:
          ``expense_year = [start_year]``, ``fixed_cost = [0]``, and
          ``cost_allocation = [fluid_type]``.
        """

        ot = self.opex_total
        kwargs = {
            "start_year": ot.start_year,
            "end_year": ot.end_year,
        }

        if fluid_type not in ot.cost_allocation:
            return OPEX(
                **kwargs,
                expense_year=np.array([ot.start_year]),
                fixed_cost=np.array([0]),
                cost_allocation=[fluid_type],
            )

        else:
            allocation_array = np.array(ot.cost_allocation)
            mask = (allocation_array == fluid_type)

            return OPEX(
                **kwargs,
                expense_year=ot.expense_year[mask],
                cost_allocation=allocation_array[mask].tolist(),
                cost_type=np.array(ot.cost_type)[mask].tolist(),
                description=np.array(ot.description)[mask].tolist(),
                tax_portion=ot.tax_portion[mask],
                tax_discount=ot.tax_discount[mask],
                fixed_cost=ot.fixed_cost[mask],
                prod_rate=ot.prod_rate[mask],
                cost_per_volume=ot.cost_per_volume[mask],
            )

    def _classify_asr_cost_by_fluid(self, fluid_type: FluidType) -> ASR:
        """
        Classify and filter Abandonment and Site Restoration (ASR) costs
        by fluid type (oil or gas).

        This method extracts the portion of the total ASR costs
        (`asr_cost_total`) that is allocated to a given fluid type.
        If the specified fluid type does not exist in the cost allocation,
        a zero-cost `ASR` object is returned as a placeholder.

        Parameters
        ----------
        fluid_type : FluidType
            The fluid type to filter ASR costs by.
            Typically one of ``FluidType.OIL`` or ``FluidType.GAS``.

        Returns
        -------
        ASR
            An `ASR` object containing only the costs allocated to the
            specified fluid type. If no allocation exists for the fluid,
            a zero-cost object with default values is returned.

        Notes
        -----
        - Filtering is done by applying a boolean mask to the
          `cost_allocation` array of the total ASR object.
        - If the fluid type is not found, a placeholder is returned with:
          ``expense_year = [start_year]``, ``cost = [0]``, and
          ``cost_allocation = [fluid_type]``.
        """

        act = self.asr_cost_total
        kwargs = {
            "start_year": act.start_year,
            "end_year": act.end_year,
        }

        if fluid_type not in act.cost_allocation:
            return ASR(
                **kwargs,
                expense_year=np.array([act.start_year]),
                cost=np.array([0]),
                cost_allocation=[fluid_type],
            )

        else:
            allocation_array = np.array(act.cost_allocation)
            mask = (allocation_array == fluid_type)

            return ASR(
                **kwargs,
                expense_year=act.expense_year[mask],
                cost=act.cost[mask],
                cost_allocation=allocation_array[mask].tolist(),
                cost_type=np.array(act.cost_type)[mask].tolist(),
                description=np.array(act.description)[mask].tolist(),
                tax_portion=act.tax_portion[mask],
                tax_discount=act.tax_discount[mask],
                final_year=act.final_year[mask],
                future_rate=act.future_rate[mask],
            )

    def _classify_lbt_cost_by_fluid(self, fluid_type: FluidType) -> LBT:
        """
        Classify and filter Land and Building Tax (LBT) costs
        by fluid type (oil or gas).

        This method extracts the portion of the total LBT costs
        (`lbt_cost_total`) that is allocated to a given fluid type.
        If the specified fluid type does not exist in the cost allocation,
        a zero-cost `LBT` object is returned as a placeholder.

        Parameters
        ----------
        fluid_type : FluidType
            The fluid type to filter LBT costs by.
            Typically one of ``FluidType.OIL`` or ``FluidType.GAS``.

        Returns
        -------
        LBT
            An `LBT` object containing only the costs allocated to the
            specified fluid type. If no allocation exists for the fluid,
            a zero-cost object with default values is returned.

        Notes
        -----
        - Filtering is performed using a boolean mask applied to the
          `cost_allocation` array of the total LBT object.
        - If the fluid type is not found, a placeholder is returned with:
          ``expense_year = [start_year]``, ``cost = [0]``, and
          ``cost_allocation = [fluid_type]``.
        """

        lct = self.lbt_cost_total
        kwargs = {
            "start_year": lct.start_year,
            "end_year": lct.end_year,
        }

        if fluid_type not in lct.cost_allocation:
            return LBT(
                **kwargs,
                expense_year=np.array([lct.start_year]),
                cost=np.array([0]),
                cost_allocation=[fluid_type],
            )

        else:
            allocation_array = np.array(lct.cost_allocation)
            mask = (allocation_array == fluid_type)

            return LBT(
                **kwargs,
                expense_year=lct.expense_year[mask],
                cost_allocation=allocation_array[mask].tolist(),
                cost_type=np.array(lct.cost_type)[mask].tolist(),
                description=np.array(lct.description)[mask].tolist(),
                tax_portion=lct.tax_portion[mask],
                tax_discount=lct.tax_discount[mask],
                final_year=lct.final_year[mask],
                utilized_land_area=lct.utilized_land_area[mask],
                utilized_building_area=lct.utilized_building_area[mask],
                njop_land=lct.njop_land[mask],
                njop_building=lct.njop_building[mask],
                gross_revenue=lct.gross_revenue[mask],
                cost=lct.cost[mask],
            )

    def _classify_cost_of_sales_by_fluid(self, fluid_type: FluidType) -> CostOfSales:
        """
        Classify and filter Cost of Sales (CoS) by fluid type (oil or gas).

        This method extracts the portion of the total Cost of Sales
        (`cost_of_sales_total`) that is allocated to a given fluid type.
        If the specified fluid type does not exist in the cost allocation,
        a zero-cost `CostOfSales` object is returned as a placeholder.

        Parameters
        ----------
        fluid_type : FluidType
            The fluid type to filter Cost of Sales by.
            Typically one of ``FluidType.OIL`` or ``FluidType.GAS``.

        Returns
        -------
        CostOfSales
            A `CostOfSales` object containing only the costs allocated
            to the specified fluid type. If no allocation exists for
            the fluid, a zero-cost object with default values is returned.

        Notes
        -----
        - Filtering is performed using a boolean mask applied to the
          `cost_allocation` array of the total Cost of Sales object.
        - If the fluid type is not found, a placeholder is returned with:
          ``expense_year = [start_year]``, ``cost = [0]``, and
          ``cost_allocation = [fluid_type]``.
        """

        cst = self.cost_of_sales_total
        kwargs = {
            "start_year": cst.start_year,
            "end_year": cst.end_year,
        }

        if fluid_type not in cst.cost_allocation:
            return CostOfSales(
                **kwargs,
                expense_year=np.array([cst.start_year]),
                cost=np.array([0]),
                cost_allocation=[fluid_type],
            )

        else:
            allocation_array = np.array(cst.cost_allocation)
            mask = (allocation_array == fluid_type)

            return CostOfSales(
                **kwargs,
                expense_year=cst.expense_year[mask],
                cost=cst.cost[mask],
                cost_allocation=allocation_array[mask].tolist(),
                cost_type=np.array(cst.cost_type)[mask].tolist(),
                description=np.array(cst.description)[mask].tolist(),
                tax_portion=cst.tax_portion[mask],
                tax_discount=cst.tax_discount[mask],
            )

    @staticmethod
    def _classify_costs_by_fluid(classifier) -> dict:
        """
        Classify and filter costs by fluid type (oil and gas).

        This method applies a given classifier function to both fluid types
        (`FluidType.OIL` and `FluidType.GAS`) and returns a dictionary mapping
        each fluid string key ("oil", "gas") to the corresponding classified object.

        Parameters
        ----------
        classifier : callable
            A function that takes a `FluidType` (e.g., `FluidType.OIL`,
            `FluidType.GAS`) as input and returns a cost object (such as
            `CapitalCost`, `Intangible`, `OPEX`, etc.).

        Returns
        -------
        dict
            A dictionary with string keys ("oil", "gas") and values equal to
            the results of applying the classifier to the corresponding
            `FluidType`.

        Notes
        -----
        - This method serves as a generic utility to avoid repeating similar
          classification logic for each cost type.
        - The actual returned cost objects depend on the `classifier` function
          passed in.
        """

        fluid_map = {
            "oil": FluidType.OIL,
            "gas": FluidType.GAS,
        }

        return {fluid: classifier(ftype) for fluid, ftype in fluid_map.items()}

    def _validate_approval_year(self) -> None:
        """
        Validate and set the POD I approval year against project and fluid timelines.

        This method ensures that the approval year is valid relative to the project's
        start and end dates, as well as the earliest onstream year among the fluids
        (oil and gas). If `approval_year` is not set (None), it defaults to the
        earliest fluid onstream year. Otherwise, it performs consistency checks to
        prevent invalid approval years.

        Raises
        ------
        BaseProjectException
            If `approval_year` is not an integer.
            If `approval_year` is earlier than the project start year.
            If `approval_year` is later than the project end year.
            If `approval_year` is later than the earliest fluid onstream year.

        Notes
        -----
        - If `approval_year` is not provided, it is automatically set to the earliest
          onstream year between oil and gas.
        - The validation prevents approval years outside the project timeframe or
          inconsistent with the fluid onstream timeline.
        """

        onstream_yr = min([self.oil_onstream_date.year, self.gas_onstream_date.year])

        if self.approval_year is None:
            self.approval_year = onstream_yr

        else:
            if not isinstance(self.approval_year, int):
                raise BaseProjectException(
                    f"Attribute approval_year must be an integer, not "
                    f"{self.approval_year.__class__.__qualname__}"
                )

            if self.approval_year < self.start_date.year:
                raise BaseProjectException(
                    f"Approval year ({self.approval_year}) is before the project "
                    f"start year ({self.start_date.year})"
                )

            if self.approval_year > self.end_date.year:
                raise BaseProjectException(
                    f"Approval year ({self.approval_year}) is after the project "
                    f"end year ({self.end_date.year})"
                )

            if self.approval_year > onstream_yr:
                raise BaseProjectException(
                    f"Approval year ({self.approval_year}) is after "
                    f"onstream year ({onstream_yr})"
                )

    def _prepare_cost_types(
        self,
        cost_obj: CapitalCost | Intangible | OPEX | ASR | LBT | CostOfSales,
    ) -> None:
        """
        Assign and validate cost types for a given cost object based on
        project timelines.

        This method classifies each expense in `cost_obj` into one of three
        categories:
        -   ``PRE_ONSTREAM_COST``: Expenses occurring between the approval year
            and the onstream year.
        -   ``POST_ONSTREAM_COST``: Expenses occurring after the onstream year.
        -   ``SUNK_COST``: Expenses occurring before the approval year.

        The method validates that classifications are consistent with project
        rules, including explicit checks for boundary years (approval and
        onstream).

        Parameters
        ----------
        cost_obj : CapitalCost | Intangible | OPEX | ASR | LBT | CostOfSales
            The cost object whose expense years will be classified into cost types.

        Returns
        -------
        None
            The method modifies the ``cost_type`` attribute of `cost_obj` in place.

        Raises
        ------
        BaseProjectException
            - If any assignment of cost types is inconsistent with the expected
              classification.
            - If ``POST_ONSTREAM_COST`` is assigned at the approval year.
            - If ``SUNK_COST`` is assigned at the onstream year.

        Notes
        -----
        - The onstream year is automatically determined as the earliest
          onstream year between oil and gas.
        - The approval year is validated before classification.
        - Cost type assignment is performed using boolean masks on
          ``expense_year``.
        - Boundary years (approval and onstream) are checked explicitly to
          prevent invalid classifications.
        """

        # Validate approval_year
        self._validate_approval_year()

        # Specify relevant onstream year corresponds to OIL or GAS
        onstream_year = min([self.oil_onstream_date.year, self.gas_onstream_date.year])

        # Build masks
        ct = np.array(cost_obj.cost_type)
        ey = cost_obj.expense_year

        post_onstream = ey > onstream_year
        sunk_cost = ey < self.approval_year
        pre_onstream = (ey > self.approval_year) & (ey < onstream_year)

        # Assign cost types using the masks
        ct[post_onstream] = CostType.POST_ONSTREAM_COST
        ct[sunk_cost] = CostType.SUNK_COST
        ct[pre_onstream] = CostType.PRE_ONSTREAM_COST

        # Validate cost types assignments
        rules = [
            (post_onstream, CostType.POST_ONSTREAM_COST, "post-onstream"),
            (sunk_cost, CostType.SUNK_COST, "sunk cost"),
            (pre_onstream, CostType.PRE_ONSTREAM_COST, "pre-onstream")
        ]

        for mask, expected, label in rules:
            if np.any(mask) and not np.all(ct[mask] == expected):
                raise BaseProjectException(f"Mismatch in {label} classification")

        # Validate cost types at exact approval year boundary
        at_approval = (ey == self.approval_year)
        if (
            np.any(at_approval)
            and CostType.POST_ONSTREAM_COST in ct[at_approval]
        ):
            raise BaseProjectException(f"Cannot accept POST ONSTREAM at approval year")

        # Validate cost types at exact onstream year boundary
        at_onstream = (ey == onstream_year)
        if (
            self.approval_year < onstream_year
            and np.any(at_onstream)
            and CostType.SUNK_COST in ct[at_onstream]
        ):
            raise BaseProjectException(f"Cannot accept SUNK COST at onstream year")

        # Modify cost_type attribute of the cost_obj
        cost_obj.cost_type = ct.tolist()

    def _filter_capital_cost(
        self, cost_obj_fluid: CapitalCost, include_cost_type: CostType
    ) -> CapitalCost:
        """
        Filter a CapitalCost object (pre-filtered by fluid type) by a specific
        cost type.

        This method extracts only the entries from ``cost_obj_fluid`` whose
        ``cost_type`` matches the specified ``include_cost_type``. If no
        matching entries exist, it returns a placeholder ``CapitalCost``
        object with zero cost at a default reference year.

        Unlike the generic version, ``cost_obj_fluid`` is expected to
        already represent a single fluid type (``FluidType.OIL`` or
        ``FluidType.GAS``). Its ``cost_allocation`` attribute therefore
        contains only one fluid type.

        Parameters
        ----------
        cost_obj_fluid : CapitalCost
            The input CapitalCost object that has already been filtered to
            represent a single fluid type. It contains attributes such as
            ``expense_year``, ``cost``, ``cost_allocation``, and others.

        include_cost_type : CostType
            The cost type to filter for. Supported values:
            ``CostType.SUNK_COST``, ``CostType.PRE_ONSTREAM_COST``,
            or ``CostType.POST_ONSTREAM_COST``.

        Returns
        -------
        CapitalCost
            A new CapitalCost object containing only the entries that
            match ``include_cost_type``. If no entries match, a placeholder
            object with zero cost and a default reference year is returned.

        Notes
        -----
        - Default reference year for placeholder objects:
            * ``CostType.SUNK_COST`` → project start year.
            * ``CostType.PRE_ONSTREAM_COST`` → project approval year.
            * ``CostType.POST_ONSTREAM_COST`` → earliest onstream year
              between oil and gas.
        - Because ``cost_obj_fluid`` already contains a single fluid type
          in its ``cost_allocation``, that allocation is preserved in both
          filtered and placeholder outputs.
        - All relevant attributes (e.g., ``description``, ``tax_portion``,
          ``useful_life``, ``pis_year``, etc.) are filtered consistently
          using the same boolean mask.
        - Returned attributes are converted to lists where applicable.
        """

        # Raise an exception for an incorrect input
        if not isinstance(cost_obj_fluid, CapitalCost):
            raise BaseProjectException(
                f"Invalid input: cost_obj_fluid must be a CapitalCost object. "
                f"Received {cost_obj_fluid.__class__.__qualname__} instead."
            )

        # Define intermediate parameters
        kwargs = {
            "start_year": cost_obj_fluid.start_year,
            "end_year": cost_obj_fluid.end_year,
        }

        onstream_yr = min([self.oil_onstream_date.year, self.gas_onstream_date.year])

        allocation = (
            FluidType.OIL if FluidType.OIL in cost_obj_fluid.cost_allocation
            else FluidType.GAS
        )

        ey = {
            CostType.SUNK_COST: cost_obj_fluid.start_year,
            CostType.PRE_ONSTREAM_COST: self.approval_year,
            CostType.POST_ONSTREAM_COST: onstream_yr,
        }[include_cost_type]

        # Conduct masking process
        cost_type_array = np.array(cost_obj_fluid.cost_type)
        mask = (cost_type_array == include_cost_type)

        if not np.any(mask):
            return CapitalCost(
                **kwargs,
                expense_year=np.array([ey]),
                cost=np.array([0]),
                cost_allocation=[allocation],
                cost_type=[include_cost_type],
            )

        return CapitalCost(
            **kwargs,
            expense_year=cost_obj_fluid.expense_year[mask],
            cost=cost_obj_fluid.cost[mask],
            cost_allocation=np.array(cost_obj_fluid.cost_allocation)[mask].tolist(),
            cost_type=np.array(cost_obj_fluid.cost_type)[mask].tolist(),
            description=np.array(cost_obj_fluid.description)[mask].tolist(),
            tax_portion=cost_obj_fluid.tax_portion[mask],
            tax_discount=cost_obj_fluid.tax_discount[mask],
            pis_year=cost_obj_fluid.pis_year[mask],
            salvage_value=cost_obj_fluid.salvage_value[mask],
            useful_life=cost_obj_fluid.useful_life[mask],
            depreciation_factor=cost_obj_fluid.depreciation_factor[mask],
            is_ic_applied=np.array(cost_obj_fluid.is_ic_applied)[mask].tolist(),
        )

    def _filter_intangible(
        self, cost_obj_fluid: Intangible, include_cost_type: CostType
    ) -> Intangible:
        """
        Filter an Intangible cost object (pre-filtered by fluid type)
        by a specific cost type.

        This method extracts only the entries from ``cost_obj_fluid`` whose
        ``cost_type`` matches the specified ``include_cost_type``. If no
        matching entries exist, it returns a placeholder ``Intangible``
        object with zero cost at a default reference year.

        Unlike the generic version, ``cost_obj_fluid`` is expected to
        already represent a single fluid type (``FluidType.OIL`` or
        ``FluidType.GAS``). Its ``cost_allocation`` attribute therefore
        contains only one fluid type.

        Parameters
        ----------
        cost_obj_fluid : Intangible
            The input Intangible object that has already been filtered to
            represent a single fluid type. It contains attributes such as
            ``expense_year``, ``cost``, ``cost_allocation``, and others.

        include_cost_type : CostType
            The cost type to filter for. Supported values:
            ``CostType.SUNK_COST``, ``CostType.PRE_ONSTREAM_COST``,
            or ``CostType.POST_ONSTREAM_COST``.

        Returns
        -------
        Intangible
            A new Intangible object containing only the entries that
            match ``include_cost_type``. If no entries match, a placeholder
            object with zero cost and a default reference year is returned.

        Notes
        -----
        - Default reference year for placeholder objects:
            * ``CostType.SUNK_COST`` → project start year.
            * ``CostType.PRE_ONSTREAM_COST`` → project approval year.
            * ``CostType.POST_ONSTREAM_COST`` → earliest onstream year
              between oil and gas.
        - Because ``cost_obj_fluid`` already contains a single fluid type
          in its ``cost_allocation``, that allocation is preserved in both
          filtered and placeholder outputs.
        - All relevant attributes (e.g., ``description``, ``tax_portion``,
          ``tax_discount``) are filtered consistently using the same
          boolean mask.
        - Returned attributes are converted to lists where applicable.
        """

        # Raise an exception for an incorrect input
        if not isinstance(cost_obj_fluid, Intangible):
            raise BaseProjectException(
                f"Invalid input: cost_obj_fluid must be an Intangible object. "
                f"Received {cost_obj_fluid.__class__.__qualname__} instead."
            )

        # Define intermediate parameters
        kwargs = {
            "start_year": cost_obj_fluid.start_year,
            "end_year": cost_obj_fluid.end_year,
        }

        onstream_yr = min([self.oil_onstream_date.year, self.gas_onstream_date.year])

        allocation = (
            FluidType.OIL if FluidType.OIL in cost_obj_fluid.cost_allocation
            else FluidType.GAS
        )

        ey = {
            CostType.SUNK_COST: cost_obj_fluid.start_year,
            CostType.PRE_ONSTREAM_COST: self.approval_year,
            CostType.POST_ONSTREAM_COST: onstream_yr,
        }[include_cost_type]

        # Conduct masking process
        cost_type_array = np.array(cost_obj_fluid.cost_type)
        mask = (cost_type_array == include_cost_type)

        if not np.any(mask):
            return Intangible(
                **kwargs,
                expense_year=np.array([ey]),
                cost=np.array([0]),
                cost_allocation=[allocation],
                cost_type=[include_cost_type],
            )

        return Intangible(
            **kwargs,
            expense_year=cost_obj_fluid.expense_year[mask],
            cost=cost_obj_fluid.cost[mask],
            cost_allocation=np.array(cost_obj_fluid.cost_allocation)[mask].tolist(),
            cost_type=np.array(cost_obj_fluid.cost_type)[mask].tolist(),
            description=np.array(cost_obj_fluid.description)[mask].tolist(),
            tax_portion=cost_obj_fluid.tax_portion[mask],
            tax_discount=cost_obj_fluid.tax_discount[mask],
        )

    def _filter_opex(self, cost_obj_fluid: OPEX, include_cost_type: CostType) -> OPEX:
        """
        Filter an OPEX cost object (pre-filtered by fluid type) by a
        specific cost type.

        This method extracts only the entries from ``cost_obj_fluid`` whose
        ``cost_type`` matches the specified ``include_cost_type``. If no
        matching entries exist, it returns a placeholder ``OPEX`` object
        with zero fixed cost at a default reference year.

        Unlike the generic version, ``cost_obj_fluid`` is expected to
        already represent a single fluid type (``FluidType.OIL`` or
        ``FluidType.GAS``). Its ``cost_allocation`` attribute therefore
        contains only one fluid type.

        Parameters
        ----------
        cost_obj_fluid : OPEX
            The input OPEX object that has already been filtered to
            represent a single fluid type. It contains attributes such as
            ``expense_year``, ``fixed_cost``, ``prod_rate``,
            ``cost_per_volume``, and others.

        include_cost_type : CostType
            The cost type to filter for. Supported values:
            ``CostType.SUNK_COST``, ``CostType.PRE_ONSTREAM_COST``,
            or ``CostType.POST_ONSTREAM_COST``.

        Returns
        -------
        OPEX
            A new OPEX object containing only the entries that match
            ``include_cost_type``. If no entries match, a placeholder
            object with zero fixed cost and a default reference year
            is returned.

        Notes
        -----
        - Default reference year for placeholder objects:
            * ``CostType.SUNK_COST`` → project start year.
            * ``CostType.PRE_ONSTREAM_COST`` → project approval year.
            * ``CostType.POST_ONSTREAM_COST`` → earliest onstream year
              between oil and gas.
        - Because ``cost_obj_fluid`` already contains a single fluid type
          in its ``cost_allocation``, that allocation is preserved in both
          filtered and placeholder outputs.
        - All relevant attributes (e.g., ``description``, ``tax_portion``,
          ``tax_discount``, ``prod_rate``, ``cost_per_volume``) are
          filtered consistently using the same boolean mask.
        - Returned attributes are converted to lists where applicable.
        """

        # Raise an exception for an incorrect input
        if not isinstance(cost_obj_fluid, OPEX):
            raise BaseProjectException(
                f"Invalid input: cost_obj_fluid must be an OPEX object. "
                f"Received {cost_obj_fluid.__class__.__qualname__} instead."
            )

        # Define intermediate parameters
        kwargs = {
            "start_year": cost_obj_fluid.start_year,
            "end_year": cost_obj_fluid.end_year,
        }

        onstream_yr = min([self.oil_onstream_date.year, self.gas_onstream_date.year])

        allocation = (
            FluidType.OIL if FluidType.OIL in cost_obj_fluid.cost_allocation
            else FluidType.GAS
        )

        ey = {
            CostType.SUNK_COST: cost_obj_fluid.start_year,
            CostType.PRE_ONSTREAM_COST: self.approval_year,
            CostType.POST_ONSTREAM_COST: onstream_yr
        }[include_cost_type]

        # Conduct masking process
        cost_type_array = np.array(cost_obj_fluid.cost_type)
        mask = (cost_type_array == include_cost_type)

        if not np.any(mask):
            return OPEX(
                **kwargs,
                expense_year=np.array([ey]),
                fixed_cost=np.array([0]),
                cost_allocation=[allocation],
                cost_type=[include_cost_type],
            )

        return OPEX(
            **kwargs,
            expense_year=cost_obj_fluid.expense_year[mask],
            cost_allocation=np.array(cost_obj_fluid.cost_allocation)[mask].tolist(),
            cost_type=np.array(cost_obj_fluid.cost_type)[mask].tolist(),
            description=np.array(cost_obj_fluid.description)[mask].tolist(),
            tax_portion=cost_obj_fluid.tax_portion[mask],
            tax_discount=cost_obj_fluid.tax_discount[mask],
            fixed_cost=cost_obj_fluid.fixed_cost[mask],
            prod_rate=cost_obj_fluid.prod_rate[mask],
            cost_per_volume=cost_obj_fluid.cost_per_volume[mask],
        )

    def _filter_asr(self, cost_obj_fluid: ASR, include_cost_type: CostType) -> ASR:
        """
        Filter an ASR (Abandonment and Site Restoration) cost object by a
        specific cost type.

        This method extracts only the entries from `cost_obj_fluid` whose
        `cost_type` matches the specified `include_cost_type`. The input
        `cost_obj_fluid` must already be filtered by fluid type, meaning its
        `cost_allocation` contains only one fluid type (either
        ``FluidType.OIL`` or ``FluidType.GAS``).

        If no matching entries exist,
        it returns a placeholder `ASR` object with zero cost at a default
        reference year.

        Parameters
        ----------
        cost_obj_fluid : ASR
            The input ASR object (already filtered by fluid type) containing
            attributes such as `expense_year`, `cost`, `final_year`,
            and `future_rate`. If the input is not an ASR instance, a
            ``BaseProjectException`` is raised.

        include_cost_type : CostType
            The cost type to filter for (e.g.,
            ``CostType.SUNK_COST``, ``CostType.PRE_ONSTREAM_COST``,
            or ``CostType.POST_ONSTREAM_COST``).

        Returns
        -------
        ASR
            A new ASR object containing only the entries that match
            `include_cost_type`. If no entries match, a placeholder
            object with zero cost is returned.

        Notes
        -----
        -   Default reference year for placeholder objects:
              * ``CostType.SUNK_COST`` → project start year.
              * ``CostType.PRE_ONSTREAM_COST`` → project approval year.
              * ``CostType.POST_ONSTREAM_COST`` → the earliest onstream year
                between oil and gas (i.e., min(oil_onstream_year, gas_onstream_year)).
        -   A default `cost_allocation` is inferred: if ``FluidType.OIL`` exists
            in the original allocation, it is used; otherwise, ``FluidType.GAS``
            is assigned.
        -   All relevant attributes (e.g., `description`, `tax_portion`,
            `tax_discount`, `final_year`, `future_rate`) are filtered consistently
            using the same boolean mask.
        -   Returned attributes are converted to lists where applicable.
        """

        # Raise an exception for an incorrect input
        if not isinstance(cost_obj_fluid, ASR):
            raise BaseProjectException(
                f"Invalid input: cost_obj_fluid must be an ASR object. "
                f"Received {cost_obj_fluid.__class__.__qualname__} instead."
            )

        kwargs = {
            "start_year": cost_obj_fluid.start_year,
            "end_year": cost_obj_fluid.end_year,
        }

        onstream_yr = min([self.oil_onstream_date.year, self.gas_onstream_date.year])

        allocation = (
            FluidType.OIL if FluidType.OIL in cost_obj_fluid.cost_allocation
            else FluidType.GAS
        )

        ey = {
            CostType.SUNK_COST: cost_obj_fluid.start_year,
            CostType.PRE_ONSTREAM_COST: self.approval_year,
            CostType.POST_ONSTREAM_COST: onstream_yr
        }[include_cost_type]

        # Conduct masking process
        cost_type_array = np.array(cost_obj_fluid.cost_type)
        mask = (cost_type_array == include_cost_type)

        if not np.any(mask):
            return ASR(
                **kwargs,
                expense_year=np.array([ey]),
                cost=np.array([0]),
                cost_allocation=[allocation],
                cost_type=[include_cost_type],
            )

        return ASR(
            **kwargs,
            expense_year=cost_obj_fluid.expense_year[mask],
            cost=cost_obj_fluid.cost[mask],
            cost_allocation=np.array(cost_obj_fluid.cost_allocation)[mask].tolist(),
            cost_type=np.array(cost_obj_fluid.cost_type)[mask].tolist(),
            description=np.array(cost_obj_fluid.description)[mask].tolist(),
            tax_portion=cost_obj_fluid.tax_portion[mask],
            tax_discount=cost_obj_fluid.tax_discount[mask],
            final_year=cost_obj_fluid.final_year[mask],
            future_rate=cost_obj_fluid.future_rate[mask],
        )

    def _filter_lbt(self, cost_obj_fluid: LBT, include_cost_type: CostType) -> LBT:
        """
        Filter an LBT (Land and Building Tax) cost object by a specific
        cost type.

        This method extracts only the entries from `cost_obj_fluid` whose
        `cost_type` matches the specified `include_cost_type`. The input
        `cost_obj_fluid` must already be filtered by fluid type, meaning its
        `cost_allocation` contains only one fluid type (either
        ``FluidType.OIL`` or ``FluidType.GAS``).

        If no matching entries exist, it returns a placeholder `LBT` object
        with zero cost at a default reference year.

        Parameters
        ----------
        cost_obj_fluid : LBT
            The input LBT object (already filtered by fluid type) containing
            attributes such as `expense_year`, `cost`, `njop_land`,
            `njop_building`, `utilized_land_area`, and `gross_revenue`.
            If the input is not an LBT instance, a ``BaseProjectException``
            is raised.

        include_cost_type : CostType
            The cost type to filter for (e.g.,
            ``CostType.SUNK_COST``, ``CostType.PRE_ONSTREAM_COST``,
            or ``CostType.POST_ONSTREAM_COST``).

        Returns
        -------
        LBT
            A new LBT object containing only the entries that match
            `include_cost_type`. If no entries match, a placeholder
            object with zero cost is returned.

        Notes
        -----
        -   Default reference year for placeholder objects:
              * ``CostType.SUNK_COST`` → project start year.
              * ``CostType.PRE_ONSTREAM_COST`` → project approval year.
              * ``CostType.POST_ONSTREAM_COST`` → the earliest onstream year
                between oil and gas (i.e., min(oil_onstream_year, gas_onstream_year)).
        -   A default `cost_allocation` is inferred: if ``FluidType.OIL`` exists
            in the original allocation, it is used; otherwise, ``FluidType.GAS``
            is assigned.
        -   All relevant attributes (e.g., `description`, `tax_portion`,
            `tax_discount`, `final_year`, `utilized_land_area`,
            `utilized_building_area`, `njop_land`, `njop_building`,
            `gross_revenue`, and `cost`) are filtered consistently using the
            same boolean mask.
        -   Returned attributes are converted to lists where applicable.
        """

        # Raise an exception for an incorrect input
        if not isinstance(cost_obj_fluid, LBT):
            raise BaseProjectException(
                f"Invalid input: cost_obj_fluid must be an LBT object. "
                f"Received {cost_obj_fluid.__class__.__qualname__} instead."
            )

        # Define intermediate parameters
        kwargs = {
            "start_year": cost_obj_fluid.start_year,
            "end_year": cost_obj_fluid.end_year,
        }

        onstream_yr = min([self.oil_onstream_date.year, self.gas_onstream_date.year])

        allocation = (
            FluidType.OIL if FluidType.OIL in cost_obj_fluid.cost_allocation
            else FluidType.GAS
        )

        ey = {
            CostType.SUNK_COST: cost_obj_fluid.start_year,
            CostType.PRE_ONSTREAM_COST: self.approval_year,
            CostType.POST_ONSTREAM_COST: onstream_yr,
        }[include_cost_type]

        # Conduct masking process
        cost_type_array = np.array(cost_obj_fluid.cost_type)
        mask = (cost_type_array == include_cost_type)

        if not np.any(mask):
            return LBT(
                **kwargs,
                expense_year=np.array([ey]),
                cost=np.array([0]),
                cost_allocation=[allocation],
                cost_type=[include_cost_type],
            )

        return LBT(
            **kwargs,
            expense_year=cost_obj_fluid.expense_year[mask],
            cost_allocation=np.array(cost_obj_fluid.cost_allocation)[mask].tolist(),
            cost_type=np.array(cost_obj_fluid.cost_type)[mask].tolist(),
            description=np.array(cost_obj_fluid.description)[mask].tolist(),
            tax_portion=cost_obj_fluid.tax_portion[mask],
            tax_discount=cost_obj_fluid.tax_discount[mask],
            final_year=cost_obj_fluid.final_year[mask],
            utilized_land_area=cost_obj_fluid.utilized_land_area[mask],
            utilized_building_area=cost_obj_fluid.utilized_building_area[mask],
            njop_land=cost_obj_fluid.njop_land[mask],
            njop_building=cost_obj_fluid.njop_building[mask],
            gross_revenue=cost_obj_fluid.gross_revenue[mask],
            cost=cost_obj_fluid.cost[mask],
        )

    def _filter_cost_of_sales(
        self, cost_obj_fluid: CostOfSales, include_cost_type: CostType
    ) -> CostOfSales:
        """
        Filter a CostOfSales cost object by a specific cost type.

        This method extracts only the entries from `cost_obj_fluid` whose
        `cost_type` matches the specified `include_cost_type`. The input
        `cost_obj_fluid` must already be filtered by fluid type, meaning its
        `cost_allocation` contains only one fluid type (either
        ``FluidType.OIL`` or ``FluidType.GAS``).

        If no matching entries exist, it returns a placeholder `CostOfSales`
        object with zero cost at a default reference year.

        Parameters
        ----------
        cost_obj_fluid : CostOfSales
            The input CostOfSales object (already filtered by fluid type) containing
            attributes such as `expense_year`, `cost`, `description`,
            `tax_portion`, and `tax_discount`. If the input is not a
            CostOfSales instance, a ``BaseProjectException`` is raised.

        include_cost_type : CostType
            The cost type to filter for (e.g.,
            ``CostType.SUNK_COST``, ``CostType.PRE_ONSTREAM_COST``,
            or ``CostType.POST_ONSTREAM_COST``).

        Returns
        -------
        CostOfSales
            A new CostOfSales object containing only the entries that match
            `include_cost_type`. If no entries match, a placeholder
            object with zero cost is returned.

        Notes
        -----
        -   Default reference year for placeholder objects:
              * ``CostType.SUNK_COST`` → project start year.
              * ``CostType.PRE_ONSTREAM_COST`` → project approval year.
              * ``CostType.POST_ONSTREAM_COST`` → the earliest onstream year
                between oil and gas (i.e., min(oil_onstream_year, gas_onstream_year)).
        -   A default `cost_allocation` is inferred: if ``FluidType.OIL`` exists
            in the original allocation, it is used; otherwise, ``FluidType.GAS``
            is assigned.
        -   All relevant attributes (e.g., `description`, `tax_portion`,
            `tax_discount`, `cost`) are filtered consistently using the
            same boolean mask.
        -   Returned attributes are converted to lists where applicable.
        """

        # Raise an exception for an incorrect input
        if not isinstance(cost_obj_fluid, CostOfSales):
            raise BaseProjectException(
                f"Invalid input: cost_obj_fluid must be a CostOfSales object. "
                f"Received {cost_obj_fluid.__class__.__qualname__} instead."
            )

        # Define intermediate parameters
        kwargs = {
            "start_year": cost_obj_fluid.start_year,
            "end_year": cost_obj_fluid.end_year,
        }

        onstream_yr = min([self.oil_onstream_date.year, self.gas_onstream_date.year])

        allocation = (
            FluidType.OIL if FluidType.OIL in cost_obj_fluid.cost_allocation
            else FluidType.GAS
        )

        ey = {
            CostType.SUNK_COST: cost_obj_fluid.start_year,
            CostType.PRE_ONSTREAM_COST: self.approval_year,
            CostType.POST_ONSTREAM_COST: onstream_yr,
        }[include_cost_type]

        # Conduct masking process
        cost_type_array = np.array(cost_obj_fluid.cost_type)
        mask = (cost_type_array == include_cost_type)

        if not np.any(mask):
            return CostOfSales(
                **kwargs,
                expense_year=np.array([ey]),
                cost=np.array([0]),
                cost_allocation=[allocation],
                cost_type=[include_cost_type],
            )

        return CostOfSales(
            **kwargs,
            expense_year=cost_obj_fluid.expense_year[mask],
            cost=cost_obj_fluid.cost[mask],
            cost_allocation=np.array(cost_obj_fluid.cost_allocation)[mask].tolist(),
            cost_type=np.array(cost_obj_fluid.cost_type)[mask].tolist(),
            description=np.array(cost_obj_fluid.description)[mask].tolist(),
            tax_portion=cost_obj_fluid.tax_portion[mask],
            tax_discount=cost_obj_fluid.tax_discount[mask],
        )

    def _check_inconsistent_start_year(self) -> None:
        """
        Validate consistency of start years across all project components.

        This method ensures that the start year of every associated project
        component (lifting, capital costs, intangibles, OPEX, ASR, LBT, and
        cost of sales for oil, gas, sulfur, electricity, and CO₂) matches the
        base project's `start_date.year`.

        If any component's `start_year` is inconsistent with the base project,
        a `BaseProjectException` is raised with a detailed error message
        identifying the mismatched component(s).

        Raises
        ------
        BaseProjectException
            If any component's start year does not match the base project
            `start_date.year`. The exception message lists the base project
            year alongside all component start years for debugging.

        Notes
        -----
        - Components checked include:
          * Hydrocarbon lifting: oil, gas, sulfur, electricity, CO₂.
          * Capital costs: sunk cost, pre-onstream, and post-onstream
            (oil and gas).
          * Intangible costs: sunk cost, pre-onstream, and post-onstream
            (oil and gas).
          * OPEX: sunk cost, pre-onstream, and post-onstream (oil and gas).
          * ASR: sunk cost, pre-onstream, and post-onstream (oil and gas).
          * LBT: sunk cost, pre-onstream, and post-onstream (oil and gas).
          * Cost of sales: sunk cost, pre-onstream, and post-onstream
            (oil and gas).
        """

        if not all(
            i == self.start_date.year
            for i in [
                self._oil_lifting.start_year,
                self._gas_lifting.start_year,
                self._sulfur_lifting.start_year,
                self._electricity_lifting.start_year,
                self._co2_lifting.start_year,
                self._oil_capital_postonstream.start_year,
                self._gas_capital_postonstream.start_year,
                self._oil_intangible_postonstream.start_year,
                self._gas_intangible_postonstream.start_year,
                self._oil_opex_postonstream.start_year,
                self._gas_opex_postonstream.start_year,
                self._oil_asr_postonstream.start_year,
                self._gas_asr_postonstream.start_year,
                self._oil_lbt_postonstream.start_year,
                self._gas_lbt_postonstream.start_year,
                self._oil_cost_of_sales_postonstream.start_year,
                self._gas_cost_of_sales_postonstream.start_year,
                self._oil_capital_sunk_cost.start_year,
                self._gas_capital_sunk_cost.start_year,
                self._oil_intangible_sunk_cost.start_year,
                self._gas_intangible_sunk_cost.start_year,
                self._oil_opex_sunk_cost.start_year,
                self._gas_opex_sunk_cost.start_year,
                self._oil_asr_sunk_cost.start_year,
                self._gas_asr_sunk_cost.start_year,
                self._oil_lbt_sunk_cost.start_year,
                self._gas_lbt_sunk_cost.start_year,
                self._oil_cost_of_sales_sunk_cost.start_year,
                self._gas_cost_of_sales_sunk_cost.start_year,
                self._oil_capital_preonstream.start_year,
                self._gas_capital_preonstream.start_year,
                self._oil_intangible_preonstream.start_year,
                self._gas_intangible_preonstream.start_year,
                self._oil_opex_preonstream.start_year,
                self._gas_opex_preonstream.start_year,
                self._oil_asr_preonstream.start_year,
                self._gas_asr_preonstream.start_year,
                self._oil_lbt_preonstream.start_year,
                self._gas_lbt_preonstream.start_year,
                self._oil_cost_of_sales_preonstream.start_year,
                self._gas_cost_of_sales_preonstream.start_year,
            ]
        ):
            raise BaseProjectException(
                f"Inconsistent start project year: "
                f"Base project ({self.start_date.year}), "
                f"Oil lifting ({self._oil_lifting.start_year}), "
                f"Gas lifting ({self._gas_lifting.start_year}), "
                f"Sulfur lifting ({self._sulfur_lifting.start_year}), "
                f"Electricity lifting ({self._electricity_lifting.start_year}), "
                f"CO2 lifting ({self._co2_lifting.start_year}), "
                f"Oil capital postonstream ({self._oil_capital_postonstream.start_year}), "
                f"Gas capital postonstream ({self._gas_capital_postonstream.start_year}), "
                f"Oil intangible postonstream "
                f"({self._oil_intangible_postonstream.start_year}), "
                f"Gas intangible postonstream "
                f"({self._gas_intangible_postonstream.start_year}), "
                f"Oil opex postonstream ({self._oil_opex_postonstream.start_year}), "
                f"Gas opex postonstream ({self._gas_opex_postonstream.start_year}), "
                f"Oil asr postonstream ({self._oil_asr_postonstream.start_year}), "
                f"Gas asr postonstream ({self._gas_asr_postonstream.start_year}), "
                f"Oil lbt postonstream ({self._oil_lbt_postonstream.start_year}), "
                f"Gas lbt postonstream ({self._gas_lbt_postonstream.start_year}), "
                f"Oil cost of sales postonstream "
                f"({self._oil_cost_of_sales_postonstream.start_year}), "
                f"Gas cost of sales postonstream "
                f"({self._gas_cost_of_sales_postonstream.start_year}), "
                f"Oil capital sunkcost ({self._oil_capital_sunk_cost.start_year}), "
                f"Gas capital sunkcost ({self._gas_capital_sunk_cost.start_year}), "
                f"Oil intangible sunkcost ({self._oil_intangible_sunk_cost.start_year}), "
                f"Gas intangible sunkcost ({self._gas_intangible_sunk_cost.start_year}), "
                f"Oil opex sunkcost ({self._oil_opex_sunk_cost.start_year}), "
                f"Gas opex sunkcost ({self._gas_opex_sunk_cost.start_year}), "
                f"Oil asr sunkcost ({self._oil_asr_sunk_cost.start_year}), "
                f"Gas asr sunkcost ({self._gas_asr_sunk_cost.start_year}), "
                f"Oil lbt sunkcost ({self._oil_lbt_sunk_cost.start_year}), "
                f"Gas lbt sunkcost ({self._gas_lbt_sunk_cost.start_year}), "
                f"Oil cost of sales sunkcost "
                f"({self._oil_cost_of_sales_sunk_cost.start_year}), "
                f"Gas cost of sales sunkcost "
                f"({self._gas_cost_of_sales_sunk_cost.start_year}), "
                f"Oil capital preonstream ({self._oil_capital_preonstream.start_year}), "
                f"Gas capital preonstream ({self._gas_capital_preonstream.start_year}), "
                f"Oil intangible preonstream ({self._oil_intangible_preonstream.start_year}), "
                f"Gas intangible preonstream ({self._gas_intangible_preonstream.start_year}), "
                f"Oil opex preonstream ({self._oil_opex_preonstream.start_year}), "
                f"Gas opex preonstream ({self._gas_opex_preonstream.start_year}), "
                f"Oil asr preonstream ({self._oil_asr_preonstream.start_year}), "
                f"Gas asr preonstream ({self._gas_asr_preonstream.start_year}), "
                f"Oil lbt preonstream ({self._oil_lbt_preonstream.start_year}), "
                f"Gas lbt preonstream ({self._gas_lbt_preonstream.start_year}), "
                f"Oil cost of sales preonstream "
                f"({self._oil_cost_of_sales_preonstream.start_year}), "
                f"Gas cost of sales preonstream "
                f"({self._gas_cost_of_sales_preonstream.start_year})."
            )

    def _check_inconsistent_end_year(self) -> None:
        """
        Validate consistency of end years across all project components.

        This method ensures that the `end_year` of every associated project
        component (lifting, capital costs, intangibles, OPEX, ASR, LBT, and
        cost of sales for oil, gas, sulfur, electricity, and CO₂) matches the
        base project's `end_date.year`.

        If any component's `end_year` is inconsistent with the base project,
        a `BaseProjectException` is raised with a detailed error message
        identifying the mismatched component(s).

        Raises
        ------
        BaseProjectException
            If any component's end year does not match the base project
            `end_date.year`. The exception message lists the base project
            year alongside all component end years for debugging.

        Notes
        -----
        - Components checked include:
          * Hydrocarbon lifting: oil, gas, sulfur, electricity, CO₂.
          * Capital costs: sunk cost, pre-onstream, and post-onstream
            (oil and gas).
          * Intangible costs: sunk cost, pre-onstream, and post-onstream
            (oil and gas).
          * OPEX: sunk cost, pre-onstream, and post-onstream (oil and gas).
          * ASR: sunk cost, pre-onstream, and post-onstream (oil and gas).
          * LBT: sunk cost, pre-onstream, and post-onstream (oil and gas).
          * Cost of sales: sunk cost, pre-onstream, and post-onstream
            (oil and gas).
        """

        if not all(
            i == self.end_date.year
            for i in [
                self._oil_lifting.end_year,
                self._gas_lifting.end_year,
                self._sulfur_lifting.end_year,
                self._electricity_lifting.end_year,
                self._co2_lifting.end_year,
                self._oil_capital_postonstream.end_year,
                self._gas_capital_postonstream.end_year,
                self._oil_intangible_postonstream.end_year,
                self._gas_intangible_postonstream.end_year,
                self._oil_opex_postonstream.end_year,
                self._gas_opex_postonstream.end_year,
                self._oil_asr_postonstream.end_year,
                self._gas_asr_postonstream.end_year,
                self._oil_lbt_postonstream.end_year,
                self._gas_lbt_postonstream.end_year,
                self._oil_cost_of_sales_postonstream.end_year,
                self._gas_cost_of_sales_postonstream.end_year,
                self._oil_capital_sunk_cost.end_year,
                self._gas_capital_sunk_cost.end_year,
                self._oil_intangible_sunk_cost.end_year,
                self._gas_intangible_sunk_cost.end_year,
                self._oil_opex_sunk_cost.end_year,
                self._gas_opex_sunk_cost.end_year,
                self._oil_asr_sunk_cost.end_year,
                self._gas_asr_sunk_cost.end_year,
                self._oil_lbt_sunk_cost.end_year,
                self._gas_lbt_sunk_cost.end_year,
                self._oil_cost_of_sales_sunk_cost.end_year,
                self._gas_cost_of_sales_sunk_cost.end_year,
                self._oil_capital_preonstream.end_year,
                self._gas_capital_preonstream.end_year,
                self._oil_intangible_preonstream.end_year,
                self._gas_intangible_preonstream.end_year,
                self._oil_opex_preonstream.end_year,
                self._gas_opex_preonstream.end_year,
                self._oil_asr_preonstream.end_year,
                self._gas_asr_preonstream.end_year,
                self._oil_lbt_preonstream.end_year,
                self._gas_lbt_preonstream.end_year,
                self._oil_cost_of_sales_preonstream.end_year,
                self._gas_cost_of_sales_preonstream.end_year,
            ]
        ):
            raise BaseProjectException(
                f"Inconsistent end project year: "
                f"Base project ({self.end_date.year}), "
                f"Oil lifting ({self._oil_lifting.end_year}), "
                f"Gas lifting ({self._gas_lifting.end_year}), "
                f"Sulfur lifting ({self._sulfur_lifting.end_year}), "
                f"Electricity lifting ({self._electricity_lifting.end_year}), "
                f"CO2 lifting ({self._co2_lifting.end_year}), "
                f"Oil capital postonstream ({self._oil_capital_postonstream.end_year}), "
                f"Gas capital postonstream ({self._gas_capital_postonstream.end_year}), "
                f"Oil intangible postonstream "
                f"({self._oil_intangible_postonstream.end_year}), "
                f"Gas intangible postonstream "
                f"({self._gas_intangible_postonstream.end_year}), "
                f"Oil opex postonstream ({self._oil_opex_postonstream.end_year}), "
                f"Gas opex postonstream ({self._gas_opex_postonstream.end_year}), "
                f"Oil asr postonstream ({self._oil_asr_postonstream.end_year}), "
                f"Gas asr postonstream ({self._gas_asr_postonstream.end_year}), "
                f"Oil lbt postonstream ({self._oil_lbt_postonstream.end_year}), "
                f"Gas lbt postonstream ({self._gas_lbt_postonstream.end_year}), "
                f"Oil cost of sales postonstream "
                f"({self._oil_cost_of_sales_postonstream.end_year}), "
                f"Gas cost of sales postonstream "
                f"({self._gas_cost_of_sales_postonstream.end_year}), "
                f"Oil capital sunkcost ({self._oil_capital_sunk_cost.end_year}), "
                f"Gas capital sunkcost ({self._gas_capital_sunk_cost.end_year}), "
                f"Oil intangible sunkcost ({self._oil_intangible_sunk_cost.end_year}), "
                f"Gas intangible sunkcost ({self._gas_intangible_sunk_cost.end_year}), "
                f"Oil opex sunkcost ({self._oil_opex_sunk_cost.end_year}), "
                f"Gas opex sunkcost ({self._gas_opex_sunk_cost.end_year}), "
                f"Oil asr sunkcost ({self._oil_asr_sunk_cost.end_year}), "
                f"Gas asr sunkcost ({self._gas_asr_sunk_cost.end_year}), "
                f"Oil lbt sunkcost ({self._oil_lbt_sunk_cost.end_year}), "
                f"Gas lbt sunkcost ({self._gas_lbt_sunk_cost.end_year}), "
                f"Oil cost of sales sunkcost "
                f"({self._oil_cost_of_sales_sunk_cost.end_year}), "
                f"Gas cost of sales sunkcost "
                f"({self._gas_cost_of_sales_sunk_cost.end_year}), "
                f"Oil capital preonstream ({self._oil_capital_preonstream.end_year}), "
                f"Gas capital preonstream ({self._gas_capital_preonstream.end_year}), "
                f"Oil intangible preonstream ({self._oil_intangible_preonstream.end_year}), "
                f"Gas intangible preonstream ({self._gas_intangible_preonstream.end_year}), "
                f"Oil opex preonstream ({self._oil_opex_preonstream.end_year}), "
                f"Gas opex preonstream ({self._gas_opex_preonstream.end_year}), "
                f"Oil asr preonstream ({self._oil_asr_preonstream.end_year}), "
                f"Gas asr preonstream ({self._gas_asr_preonstream.end_year}), "
                f"Oil lbt preonstream ({self._oil_lbt_preonstream.end_year}), "
                f"Gas lbt preonstream ({self._gas_lbt_preonstream.end_year}), "
                f"Oil cost of sales preonstream "
                f"({self._oil_cost_of_sales_preonstream.end_year}), "
                f"Gas cost of sales preonstream "
                f"({self._gas_cost_of_sales_preonstream.end_year})."
            )

    def _validate_sunkcost(self, sunkcost_objects: list) -> None:
        """
        Validate that all sunk cost objects have expense years within the allowed range.

        This method checks each object in `sunkcost_objects` to ensure that the maximum
        expense year does not exceed the cutoff year. The cutoff year is defined as the
        earliest of the project approval year or the earliest onstream year (between oil
        and gas).

        Parameters
        ----------
        sunkcost_objects : list
            A list of sunk cost objects from each cost category (CapitalCost, Intangible,
            OPEX, ASR, LBT, or CostOfSales), already filtered by fluid type. Each object
            must have an `expense_year` attribute (array-like) representing the years in
            which costs are incurred.

        Raises
        ------
        SunkCostException
            If the maximum expense year of any sunk cost object exceeds the cutoff year.
            The exception message lists the offending years and the cutoff year.

        Notes
        -----
        - This validation ensures that no sunk costs are mistakenly recorded beyond the
          approval or onstream year, maintaining consistency in project cost modeling.
        """

        onstream_yr = min([self.oil_onstream_date.year, self.gas_onstream_date.year])
        sunkcost_max_years = np.array([np.max(sc.expense_year) for sc in sunkcost_objects])
        cutoff_year = min([self.approval_year, onstream_yr])
        mask = (sunkcost_max_years > cutoff_year)

        if np.any(mask):
            incorrect_years = sunkcost_max_years[mask]
            raise SunkCostException(
                f"Sunk cost years ({incorrect_years}) exceed cutoff year ({cutoff_year})"
            )

    def _validate_preonstream(self, preonstream_objects: list) -> None:
        """
        Validate that all preonstream cost objects have expense years within
        the allowed range.

        This method checks each object in `preonstream_objects` to ensure that
        the expense years fall within the allowable range: no earlier than the
        project approval year and no later than the earliest onstream year
        (between oil and gas).

        Parameters
        ----------
        preonstream_objects : list
            A list of preonstream cost objects from each cost category
            (CapitalCost, Intangible, OPEX, ASR, LBT, or CostOfSales), already
            filtered by fluid type. Each object must have an `expense_year`
            attribute (array-like) representing the years in which costs are
            incurred.

        Raises
        ------
        PreOnstreamException
            If the minimum or maximum expense year of any preonstream cost object
            falls outside the allowable range.

        Notes
        -----
        This validation ensures that no preonstream costs are recorded before
        project approval or after the earliest onstream year, maintaining
        consistency in project cost modeling.
        """

        onstream_yr = min([self.oil_onstream_date.year, self.gas_onstream_date.year])

        preonstream_max_years = np.array(
            [np.max(pr.expense_year) for pr in preonstream_objects]
        )

        preonstream_min_years = np.array(
            [np.min(pr.expense_year) for pr in preonstream_objects]
        )

        mask = (
            (preonstream_min_years < self.approval_year)
            | (preonstream_max_years > onstream_yr)
        )

        if np.any(mask):
            incorrect_years = list(
                zip(preonstream_min_years[mask], preonstream_max_years[mask])
            )
            raise PreOnstreamException(
                f"Preonstream years ({incorrect_years}) fall outside the allowable "
                f"range: {self.approval_year} <= preonstream_years <= {onstream_yr}"
            )

    def _validate_postonstream(self, postonstream_objects: list) -> None:
        """
        Validate that all postonstream cost objects have expense years not
        earlier than the onstream year.

        This method checks each object in `postonstream_objects` to ensure
        that the minimum expense year is no earlier than the earliest
        onstream year (between oil and gas).

        Parameters
        ----------
        postonstream_objects : list
            A list of postonstream cost objects from each cost category
            (CapitalCost, Intangible, OPEX, ASR, LBT, or CostOfSales),
            already filtered by fluid type. Each object must have an
            `expense_year` attribute (array-like) representing the years
            in which costs are incurred.

        Raises
        ------
        PostOnstreamException
            If the minimum expense year of any postonstream cost object
            occurs before the earliest onstream year. The exception message
            lists the offending years and the cutoff year.

        Notes
        -----
        This validation ensures that no postonstream costs are recorded
        before the project reaches its onstream date, maintaining consistency
        in project cost modeling.
        """

        onstream_yr = min([self.oil_onstream_date.year, self.gas_onstream_date.year])
        postonstream_min_years = np.array(
            [np.min(po.expense_year) for po in postonstream_objects]
        )
        mask = (postonstream_min_years < onstream_yr)

        if np.any(mask):
            incorrect_years = postonstream_min_years[mask]
            raise PostOnstreamException(
                f"Postonstream cost years ({incorrect_years}) occur before "
                f"onstream year ({onstream_yr}) "
            )

    def _get_cost_objects_validation(self) -> None:
        """
        Validate all cost objects (sunk, preonstream, and postonstream) across
        categories and fluids.

        This method collects cost objects from all categories (CapitalCost,
        Intangible, OPEX, ASR, LBT, and CostOfSales) for both oil and gas,
        grouped by the timing of the cost: sunk, preonstream, and postonstream.

        Each group of objects is then validated using the corresponding validator
        method to ensure their `expense_year` values are consistent with project
        milestones (approval year and onstream dates).

        Validation rules:
        -   Sunk costs must not exceed the minimum of the approval year and earliest
            onstream year.
        -   Preonstream costs must be between the approval year and the earliest
            onstream year.
        -   Postonstream costs must occur no earlier than the earliest onstream year.

        Raises
        ------
        SunkCostException
            If any sunk cost object has `expense_year` exceeding the cutoff year.
        PreOnstreamException
            If any preonstream cost object has `expense_year` outside the allowable
            range.
        PostOnstreamException
            If any postonstream cost object has `expense_year` earlier than the
            onstream year.

        Notes
        -----
        -   Each input cost object is assumed to be already filtered by fluid type.
        -   The method automates validation across multiple cost categories and
            timing groups, ensuring consistency of project cost modeling.
        """

        # Collect a list of sunkcost objects
        sunkcost_objects = [
            self._oil_capital_sunk_cost,
            self._oil_intangible_sunk_cost,
            self._oil_opex_sunk_cost,
            self._oil_asr_sunk_cost,
            self._oil_lbt_sunk_cost,
            self._oil_cost_of_sales_sunk_cost,
            self._gas_capital_sunk_cost,
            self._gas_intangible_sunk_cost,
            self._gas_opex_sunk_cost,
            self._gas_asr_sunk_cost,
            self._gas_lbt_sunk_cost,
            self._gas_cost_of_sales_sunk_cost,
        ]

        # Collect a list of preonstream objects
        preonstream_objects = [
            self._oil_capital_preonstream,
            self._oil_intangible_preonstream,
            self._oil_opex_preonstream,
            self._oil_asr_preonstream,
            self._oil_lbt_preonstream,
            self._oil_cost_of_sales_preonstream,
            self._gas_capital_preonstream,
            self._gas_intangible_preonstream,
            self._gas_opex_preonstream,
            self._gas_asr_preonstream,
            self._gas_lbt_preonstream,
            self._gas_cost_of_sales_preonstream,
        ]

        # Collect a list of postonstream objects
        postonstream_objects = [
            self._oil_capital_postonstream,
            self._oil_intangible_postonstream,
            self._oil_opex_postonstream,
            self._oil_asr_postonstream,
            self._oil_lbt_postonstream,
            self._oil_cost_of_sales_postonstream,
            self._gas_capital_postonstream,
            self._gas_intangible_postonstream,
            self._gas_opex_postonstream,
            self._gas_asr_postonstream,
            self._gas_lbt_postonstream,
            self._gas_cost_of_sales_postonstream,
        ]

        # Run validator for each cost objects
        validation_mapping = [
            (sunkcost_objects, self._validate_sunkcost),
            (preonstream_objects, self._validate_preonstream),
            (postonstream_objects, self._validate_postonstream),
        ]

        for objects, validator in validation_mapping:
            validator(objects)

    def _get_sunkcost_array(self) -> None:
        """
        Prepare sunk cost arrays for OIL and GAS.

        This method constructs arrays of pre-tax expenditures for each sunk
        cost category (Capital, Intangible, OPEX, ASR, LBT, and Cost of Sales),
        aggregates them into depreciable and non-depreciable components.

        Notes
        -----
        - Depreciable sunk costs include only capital costs.
        - Non-depreciable sunk costs are formed by summing intangible,
          OPEX, ASR, LBT, and cost of sales components.
        - The following internal attributes are set:
            * ``_oil_depreciable_sunk_cost``
            * ``_gas_depreciable_sunk_cost``
            * ``_oil_non_depreciable_sunk_cost``
            * ``_gas_non_depreciable_sunk_cost``
        - Internally relies on each cost object's
          ``expenditures_pre_tax()`` method to compute expenditures.
        """

        # Prepare sunkcost arrays for each sunkcost objects
        sunkcost_map = {
            "oil_capital": self._oil_capital_sunk_cost,
            "gas_capital": self._gas_capital_sunk_cost,
            "oil_intangible": self._oil_intangible_sunk_cost,
            "gas_intangible": self._gas_intangible_sunk_cost,
            "oil_opex": self._oil_opex_sunk_cost,
            "gas_opex": self._gas_opex_sunk_cost,
            "oil_asr": self._oil_asr_sunk_cost,
            "gas_asr": self._gas_asr_sunk_cost,
            "oil_lbt": self._oil_lbt_sunk_cost,
            "gas_lbt": self._gas_lbt_sunk_cost,
            "oil_cost_of_sales": self._oil_cost_of_sales_sunk_cost,
            "gas_cost_of_sales": self._gas_cost_of_sales_sunk_cost,
        }

        sunkcost_arr = {
            key: val.expenditures_pre_tax() for key, val in sunkcost_map.items()
        }

        # Define depreciable sunk costs
        self._oil_depreciable_sunk_cost = sunkcost_arr["oil_capital"]
        self._gas_depreciable_sunk_cost = sunkcost_arr["gas_capital"]

        # Define non-depreciable sunk costs
        self._oil_non_depreciable_sunk_cost = (
            sunkcost_arr["oil_intangible"]
            + sunkcost_arr["oil_opex"]
            + sunkcost_arr["oil_asr"]
            + sunkcost_arr["oil_lbt"]
            + sunkcost_arr["oil_cost_of_sales"]
        )

        self._gas_non_depreciable_sunk_cost = (
            sunkcost_arr["gas_intangible"]
            + sunkcost_arr["gas_opex"]
            + sunkcost_arr["gas_asr"]
            + sunkcost_arr["gas_lbt"]
            + sunkcost_arr["gas_cost_of_sales"]
        )

    def _get_preonstream_array(self) -> None:
        """
        Prepare preonstream cost arrays for OIL and GAS.

        This method constructs arrays of pre-tax expenditures for each preonstream
        cost category (Capital, Intangible, OPEX, ASR, LBT, and Cost of Sales),
        aggregates them into depreciable and non-depreciable components.

        Notes
        -----
        - Depreciable preonstream costs include only capital costs.
        - Non-depreciable preonstream costs are formed by summing intangible,
          OPEX, ASR, LBT, and cost of sales components.
        - The following internal attributes are set:
            * ``_oil_depreciable_preonstream``
            * ``_gas_depreciable_preonstream``
            * ``_oil_non_depreciable_preonstream``
            * ``_gas_non_depreciable_preonstream``
        - Internally relies on each cost object's
          ``expenditures_pre_tax()`` method to compute expenditures.
        """

        # Prepare preonstream arrays for each preonstream objects
        preonstream_map = {
            "oil_capital": self._oil_capital_preonstream,
            "gas_capital": self._gas_capital_preonstream,
            "oil_intangible": self._oil_intangible_preonstream,
            "gas_intangible": self._gas_intangible_preonstream,
            "oil_opex": self._oil_opex_preonstream,
            "gas_opex": self._gas_opex_preonstream,
            "oil_asr": self._oil_asr_preonstream,
            "gas_asr": self._gas_asr_preonstream,
            "oil_lbt": self._oil_lbt_preonstream,
            "gas_lbt": self._gas_lbt_preonstream,
            "oil_cost_of_sales": self._oil_cost_of_sales_preonstream,
            "gas_cost_of_sales": self._gas_cost_of_sales_preonstream,
        }

        preonstream_arr = {
            key: val.expenditures_pre_tax() for key, val in preonstream_map.items()
        }

        # Define depreciable preonstream costs
        self._oil_depreciable_preonstream = preonstream_arr["oil_capital"]
        self._gas_depreciable_preonstream = preonstream_arr["gas_capital"]

        # Define non-depreciable preonstream costs
        self._oil_non_depreciable_preonstream = (
            preonstream_arr["oil_intangible"]
            + preonstream_arr["oil_opex"]
            + preonstream_arr["oil_asr"]
            + preonstream_arr["oil_lbt"]
            + preonstream_arr["oil_cost_of_sales"]
        )
        self._gas_non_depreciable_preonstream = (
            preonstream_arr["gas_intangible"]
            + preonstream_arr["gas_opex"]
            + preonstream_arr["gas_asr"]
            + preonstream_arr["gas_lbt"]
            + preonstream_arr["gas_cost_of_sales"]
        )

    def _calc_pre_tax_expenditures(
        self,
        target_attr: CapitalCost | Intangible | OPEX | ASR | LBT,
        year_inflation: np.ndarray = None,
        inflation_rate: np.ndarray | int | float = 0.0,
        inflation_rate_applied_to: InflationAppliedTo | None = None,
    ) -> np.ndarray:
        """
        Calculate pre-tax expenditures with optional inflation adjustments.

        This method computes pre-tax expenditures for a given cost category
        (e.g., capital, intangible, OPEX, ASR, LBT), applying inflation
        rates conditionally based on the specified `inflation_rate_applied_to`
        setting.

        The inflation can be applied selectively to:
        - Capital expenditures (CAPEX)
        - Operating expenditures (OPEX)
        - Both CAPEX and OPEX
        - Or not applied at all

        Parameters
        ----------
        target_attr : CapitalCost or Intangible or OPEX or ASR or LBT
            The cost object representing a specific expenditure category.
        year_inflation : numpy.ndarray, optional
            Array of project years aligned with inflation application.
            Default is ``None`` (no inflation applied by year).
        inflation_rate : numpy.ndarray or int or float, default=0.0
            Inflation rate(s) to apply. Can be a scalar (constant rate),
            a 1D array matching `year_inflation`, or 0.0 (no inflation).
        inflation_rate_applied_to : InflationAppliedTo or None, optional
            Specifies where inflation is applied. Options:
            - ``None`` : No inflation applied (default).
            - ``InflationAppliedTo.CAPEX`` : Apply only to CAPEX-related attributes.
            - ``InflationAppliedTo.OPEX`` : Apply only to OPEX-related attributes.
            - ``InflationAppliedTo.CAPEX_AND_OPEX`` : Apply to both CAPEX and OPEX.

        Returns
        -------
        numpy.ndarray
            Array of pre-tax expenditures, adjusted for inflation according
            to the specified rules.

        Raises
        ------
        BaseProjectException
            If `inflation_rate_applied_to` is not recognized.
        """

        # No inflation rate
        if inflation_rate_applied_to is None:
            return target_attr.expenditures_pre_tax(
                year_inflation=year_inflation,
                inflation_rate=0.0,
            )

        # Inflation rate applied to CAPEX only
        elif inflation_rate_applied_to == InflationAppliedTo.CAPEX:
            if (
                target_attr is self._oil_capital_postonstream
                or target_attr is self._gas_capital_postonstream
                or target_attr is self._oil_intangible_postonstream
                or target_attr is self._gas_intangible_postonstream
            ):
                return target_attr.expenditures_pre_tax(
                    year_inflation=year_inflation,
                    inflation_rate=inflation_rate,
                )

            else:
                return target_attr.expenditures_pre_tax(
                    year_inflation=year_inflation,
                    inflation_rate=0.0,
                )

        # Inflation rate applied to OPEX only
        elif inflation_rate_applied_to == InflationAppliedTo.OPEX:
            if (
                target_attr is self._oil_opex_postonstream
                or target_attr is self._gas_opex_postonstream
            ):
                return target_attr.expenditures_pre_tax(
                    year_inflation=year_inflation,
                    inflation_rate=inflation_rate,
                )

            else:
                return target_attr.expenditures_pre_tax(
                    year_inflation=year_inflation,
                    inflation_rate=0.0,
                )

        # Inflation rate applied to CAPEX and OPEX
        elif inflation_rate_applied_to == InflationAppliedTo.CAPEX_AND_OPEX:
            if (
                target_attr is self._oil_capital_postonstream
                or target_attr is self._gas_capital_postonstream
                or target_attr is self._oil_intangible_postonstream
                or target_attr is self._gas_intangible_postonstream
                or target_attr is self._oil_opex_postonstream
                or target_attr is self._gas_opex_postonstream
            ):
                return target_attr.expenditures_pre_tax(
                    year_inflation=year_inflation,
                    inflation_rate=inflation_rate
                )

            else:
                return target_attr.expenditures_pre_tax(
                    year_inflation=year_inflation,
                    inflation_rate=0.0,
                )

        else:
            raise BaseProjectException(
                f"Parameter inflation_rate_applied_to is not recognized. "
                f"Available options are: CAPEX, OPEX, CAPEX and OPEX, or None. "
            )

    def _get_expenditures_pre_tax(
        self,
        year_inflation: np.ndarray = None,
        inflation_rate: np.ndarray | int | float = 0.0,
        inflation_rate_applied_to: InflationAppliedTo | None = None,
    ) -> None:
        """
        Calculate and assign pre-tax expenditures for various categories,
        adjusted for inflation.

        This method calculates the pre-tax expenditures for multiple cost
        categories related to oil and gas, adjusted by the provided inflation
        rate. The expenditures are then assigned to the corresponding attributes
        for both oil and gas.

        Parameters
        ----------
        year_inflation : np.ndarray, optional
            A NumPy array representing the years during which inflation is
            applied to the costs. If not provided, defaults to repeating the
            `start_year` for all costs.
        inflation_rate : np.ndarray or float, optional
            The inflation rate(s) to apply to the project costs. If provided as a float,
            a uniform inflation rate is applied. If provided as a NumPy array, different
            rates are applied based on the corresponding project years. Default is 0.0.
        inflation_rate_applied_to
            The selection of where the cost inflation will be applied to.

        Returns
        -------
        None
            This method does not return a value. It updates the following attributes
            with the calculated pre-tax expenditures:
            -   `_oil_capital_expenditures_pre_tax`
            -   `_gas_capital_expenditures_pre_tax`
            -   `_oil_intangible_expenditures_pre_tax`
            -   `_gas_intangible_expenditures_pre_tax`
            -   `_oil_opex_expenditures_pre_tax`
            -   `_gas_opex_expenditures_pre_tax`
            -   `_oil_asr_expenditures_pre_tax`
            -   `_gas_asr_expenditures_pre_tax`
            -   `_oil_lbt_expenditures_pre_tax`
            -   `_gas_lbt_expenditures_pre_tax`
            -   `_oil_cost_of_sales_expenditures_pre_tax`
            -   `_gas_cost_of_sales_expenditures_pre_tax`
        """

        # Prepare expenditures pre tax associated with capital, intangible,
        # opex, asr, and lbt costs
        (
            self._oil_capital_expenditures_pre_tax,
            self._gas_capital_expenditures_pre_tax,
            self._oil_intangible_expenditures_pre_tax,
            self._gas_intangible_expenditures_pre_tax,
            self._oil_opex_expenditures_pre_tax,
            self._gas_opex_expenditures_pre_tax,
            self._oil_asr_expenditures_pre_tax,
            self._gas_asr_expenditures_pre_tax,
            self._oil_lbt_expenditures_pre_tax,
            self._gas_lbt_expenditures_pre_tax,
        ) = [
            self._calc_pre_tax_expenditures(
                target_attr=attr,
                year_inflation=year_inflation,
                inflation_rate=inflation_rate,
                inflation_rate_applied_to=inflation_rate_applied_to,
            )
            for attr in [
                self._oil_capital_postonstream,
                self._gas_capital_postonstream,
                self._oil_intangible_postonstream,
                self._gas_intangible_postonstream,
                self._oil_opex_postonstream,
                self._gas_opex_postonstream,
                self._oil_asr_postonstream,
                self._gas_asr_postonstream,
                self._oil_lbt_postonstream,
                self._gas_lbt_postonstream,
            ]
        ]

        # Prepare expenditures pre tax associated with cost of sales
        (
            self._oil_cost_of_sales_expenditures_pre_tax,
            self._gas_cost_of_sales_expenditures_pre_tax,
        ) = [
            attr.expenditures_pre_tax()
            for attr in [
                self._oil_cost_of_sales_postonstream,
                self._gas_cost_of_sales_postonstream,
            ]
        ]

    def _get_indirect_taxes(self, tax_rate: np.ndarray | float = 0.0) -> None:
        """
        Calculate and assign indirect taxes for various oil and gas
        expenditure categories.

        This method computes the indirect taxes (such as VAT or other
        applicable indirect taxes) for multiple categories related to
        oil and gas. It adjusts these taxes based on the provided
        tax portion, tax rate, and tax discount.

        Parameters
        ----------
        tax_rate : np.ndarray or float, optional
            A NumPy array or float representing the tax rate applied to
            the costs. If not provided, a default rate of 0.0 will be used.
            When provided as an array, it should match the project years.

        Returns
        -------
        None
            This method does not return any values. It updates the following
            attributes with the calculated indirect taxes for oil and gas:
            -   `_oil_capital_indirect_tax`
            -   `_gas_capital_indirect_tax`
            -   `_oil_intangible_indirect_tax`
            -   `_gas_intangible_indirect_tax`
            -   `_oil_opex_indirect_tax`
            -   `_gas_opex_indirect_tax`
            -   `_oil_asr_indirect_tax`
            -   `_gas_asr_indirect_tax`
            -   `_oil_lbt_indirect_tax`
            -   `_gas_lbt_indirect_tax`
            -   `_oil_cost_of_sales_indirect_tax`
            -   `_gas_cost_of_sales_indirect_tax`
        """

        # Prepare indirect taxes associated with capital, intangible,
        # opex, asr, and lbt costs
        (
            self._oil_capital_indirect_tax,
            self._gas_capital_indirect_tax,
            self._oil_intangible_indirect_tax,
            self._gas_intangible_indirect_tax,
            self._oil_opex_indirect_tax,
            self._gas_opex_indirect_tax,
            self._oil_asr_indirect_tax,
            self._gas_asr_indirect_tax,
            self._oil_lbt_indirect_tax,
            self._gas_lbt_indirect_tax,
        ) = [
            attr.indirect_taxes(tax_rate=tax_rate)
            for attr in [
                self._oil_capital_postonstream,
                self._gas_capital_postonstream,
                self._oil_intangible_postonstream,
                self._gas_intangible_postonstream,
                self._oil_opex_postonstream,
                self._gas_opex_postonstream,
                self._oil_asr_postonstream,
                self._gas_asr_postonstream,
                self._oil_lbt_postonstream,
                self._gas_lbt_postonstream,
            ]
        ]

        # Prepare indirect taxes associated with cost of sales
        (
            self._oil_cost_of_sales_indirect_tax,
            self._gas_cost_of_sales_indirect_tax,
        ) = [
            attr.indirect_taxes()
            for attr in [
                self._oil_cost_of_sales_postonstream,
                self._gas_cost_of_sales_postonstream,
            ]
        ]

    def _get_expenditures_post_tax(self) -> None:
        """
        Compute and assign post-tax expenditures for all oil and gas cost categories.

        This method iterates over each defined cost category for both oil and gas.
        For every category, it retrieves the corresponding pre-tax expenditure and
        indirect tax attributes, computes their sum, and assigns the result to the
        appropriate post-tax expenditure attribute.

        Notes
        -----
        - Supported cost categories include:
            * ``capital``
            * ``intangible``
            * ``opex``
            * ``asr`` (abandonment & site restoration)
            * ``lbt`` (land and building tax)
            * ``cost_of_sales``
        - For each category and fluid type (``oil`` and ``gas``), the method expects
          the following attributes to exist:
            * ``_{fluid}_{category}_expenditures_pre_tax``
            * ``_{fluid}_{category}_indirect_tax``
        - The computed results are stored in:
            * ``_{fluid}_{category}_expenditures_post_tax``

        Examples
        --------
        For ``oil_capital``:
            - Reads ``_oil_capital_expenditures_pre_tax`` and ``_oil_capital_indirect_tax``.
            - Stores the sum in ``_oil_capital_expenditures_post_tax``.
        """

        categories = [
            "capital",
            "intangible",
            "opex",
            "asr",
            "lbt",
            "cost_of_sales",
        ]

        for fluid in ["oil", "gas"]:
            for categ in categories:
                pre_attr = getattr(self, f"_{fluid}_{categ}_expenditures_pre_tax")
                tax_attr = getattr(self, f"_{fluid}_{categ}_indirect_tax")
                setattr(self, f"_{fluid}_{categ}_expenditures_post_tax", pre_attr + tax_attr)

    def _get_other_revenue(
        self,
        sulfur_revenue: OtherRevenue,
        electricity_revenue: OtherRevenue,
        co2_revenue: OtherRevenue
    ) -> None:
        """
        Allocate additional revenue streams (sulfur, electricity, and CO₂) to
        either oil revenue, gas revenue, or as a reduction to operating
        expenditures (OPEX).

        Parameters
        ----------
        sulfur_revenue : OtherRevenue, optional
            The allocation method for sulfur revenue.
            Possible options:
            - `OtherRevenue.ADDITION_TO_OIL_REVENUE`: Adds to oil revenue.
            - `OtherRevenue.ADDITION_TO_GAS_REVENUE`: Adds to gas revenue.
            - `OtherRevenue.REDUCTION_TO_OIL_OPEX`: Reduces oil OPEX.
            - `OtherRevenue.REDUCTION_TO_GAS_OPEX`: Reduces gas OPEX.

        electricity_revenue : OtherRevenue, optional
            The allocation method for electricity revenue.
            Possible options are the same as `sulfur_revenue`.

        co2_revenue : OtherRevenue, optional
            The allocation method for CO₂ revenue.
            Possible options are the same as `sulfur_revenue`.

        Raises
        ------
        OtherRevenueException
            If an invalid revenue allocation option is provided for sulfur,
            electricity, or CO₂.

        Notes
        -----
        - Depending on the selected revenue allocation, sulfur, electricity,
          and CO₂ revenue may be added to oil or gas revenue or used to offset
          post-tax OPEX expenditures.
        - If an invalid allocation option is provided, an `OtherRevenueException`
          is raised.
        """

        # Configure sulfur revenue
        if sulfur_revenue is OtherRevenue.ADDITION_TO_OIL_REVENUE:
            self._oil_revenue += self._sulfur_revenue

        elif sulfur_revenue is OtherRevenue.ADDITION_TO_GAS_REVENUE:
            self._gas_revenue += self._sulfur_revenue

        elif sulfur_revenue is OtherRevenue.REDUCTION_TO_OIL_OPEX:
            self._oil_opex_expenditures_post_tax -= self._sulfur_revenue

        elif sulfur_revenue is OtherRevenue.REDUCTION_TO_GAS_OPEX:
            self._gas_opex_expenditures_post_tax -= self._sulfur_revenue

        else:
            raise OtherRevenueException(
                f"Other Revenue selection is not available: {sulfur_revenue} "
            )

        # Configure electricity revenue
        if electricity_revenue is OtherRevenue.ADDITION_TO_OIL_REVENUE:
            self._oil_revenue += self._electricity_revenue

        elif electricity_revenue is OtherRevenue.ADDITION_TO_GAS_REVENUE:
            self._gas_revenue += self._electricity_revenue

        elif electricity_revenue is OtherRevenue.REDUCTION_TO_OIL_OPEX:
            self._oil_opex_expenditures_post_tax -= self._electricity_revenue

        elif electricity_revenue is OtherRevenue.REDUCTION_TO_GAS_OPEX:
            self._gas_opex_expenditures_post_tax -= self._electricity_revenue

        else:
            raise OtherRevenueException(
                f"Other revenue selection is not available: {electricity_revenue}"
            )

        # Configure CO2 revenue
        if co2_revenue is OtherRevenue.ADDITION_TO_OIL_REVENUE:
            self._oil_revenue += self._co2_revenue

        elif co2_revenue is OtherRevenue.ADDITION_TO_GAS_REVENUE:
            self._gas_revenue += self._co2_revenue

        elif co2_revenue is OtherRevenue.REDUCTION_TO_OIL_OPEX:
            self._oil_opex_expenditures_post_tax -= self._co2_revenue

        elif co2_revenue is OtherRevenue.REDUCTION_TO_GAS_OPEX:
            self._gas_opex_expenditures_post_tax -= self._co2_revenue

        else:
            raise OtherRevenueException(
                f"Other revenue selection is not available: {co2_revenue}"
            )

    def _get_consolidated_profiles(self) -> None:
        """
        Aggregate oil and gas profiles into consolidated project-wide profiles.

        This method combines oil and gas arrays across multiple categories
        (lifting, prices, revenues, sunk costs, pre-onstream costs,
        expenditures, indirect taxes, and cash flows) into consolidated
        project-level attributes. The aggregation is performed by element-wise
        addition of the corresponding oil and gas arrays.

        Returns
        -------
        None
            Updates instance attributes with consolidated values.

        Notes
        -----
        - Consolidation is applied to the following categories:
            * Lifting, weighted average price (WAP), and revenue
            * Sunk costs (depreciable, non-depreciable, and total)
            * Pre-onstream costs (depreciable, non-depreciable, and total)
            * Expenditures (pre-tax, post-tax, and by category)
            * Indirect taxes (total and by category)
            * Cash flow and non-capital components
        - Expenditure- and tax-related categories are consolidated dynamically
          using predefined category names (``capital``, ``intangible``, ``opex``,
          ``asr``, ``lbt``, ``cost_of_sales``).
        - All oil and gas attributes must be defined and have matching array shapes.
        """

        # Attributes associated with consolidated lifting
        self._consolidated_lifting = (
            self._oil_lifting.get_lifting_rate_arr()
            + self._gas_lifting.get_lifting_rate_arr()
        )
        self._consolidated_wap_price = self._oil_wap_price + self._gas_wap_price
        self._consolidated_revenue = self._oil_revenue + self._gas_revenue

        # Attributes associated with consolidated sunkcost
        self._consolidated_depreciable_sunk_cost = (
            self._oil_depreciable_sunk_cost + self._gas_depreciable_sunk_cost
        )
        self._consolidated_non_depreciable_sunk_cost = (
            self._oil_non_depreciable_sunk_cost + self._gas_non_depreciable_sunk_cost
        )
        self._consolidated_sunk_cost = self._oil_sunk_cost + self._gas_sunk_cost

        # Attributes associated with consolidated preonstream
        self._consolidated_depreciable_preonstream = (
            self._oil_depreciable_preonstream + self._gas_depreciable_preonstream
        )
        self._consolidated_non_depreciable_preonstream = (
            self._oil_non_depreciable_preonstream + self._gas_non_depreciable_preonstream
        )
        self._consolidated_preonstream = self._oil_preonstream + self._gas_preonstream

        categories = [
            "capital",
            "intangible",
            "opex",
            "asr",
            "lbt",
            "cost_of_sales"
        ]

        # Attributes associated with consolidated expenditures pre tax
        for categ in categories:
            oil_pre_tax = getattr(self, f"_oil_{categ}_expenditures_pre_tax")
            gas_pre_tax = getattr(self, f"_gas_{categ}_expenditures_pre_tax")
            setattr(
                self,
                f"_consolidated_{categ}_expenditures_pre_tax",
                oil_pre_tax + gas_pre_tax
            )

        self._consolidated_total_expenditures_pre_tax = (
            self._oil_total_expenditures_pre_tax + self._gas_total_expenditures_pre_tax
        )

        # Attributes associated with consolidated indirect tax
        for categ in categories:
            oil_indirect_tax = getattr(self, f"_oil_{categ}_indirect_tax")
            gas_indirect_tax = getattr(self, f"_gas_{categ}_indirect_tax")
            setattr(
                self,
                f"_consolidated_{categ}_indirect_tax",
                oil_indirect_tax + gas_indirect_tax
            )

        self._consolidated_total_indirect_tax = (
            self._oil_total_indirect_tax + self._gas_total_indirect_tax
        )

        # Attributes associated with consolidated expenditures post tax
        for categ in categories:
            oil_post_tax = getattr(self, f"_oil_{categ}_expenditures_post_tax")
            gas_post_tax = getattr(self, f"_gas_{categ}_expenditures_post_tax")
            setattr(
                self,
                f"_consolidated_{categ}_expenditures_post_tax",
                oil_post_tax + gas_post_tax
            )

        self._consolidated_total_expenditures_post_tax = (
            self._oil_total_expenditures_post_tax + self._gas_total_expenditures_post_tax
        )

        # Attribute associated with consolidated cashflow
        self._consolidated_non_capital = self._oil_non_capital + self._gas_non_capital
        self._consolidated_cashflow = self._oil_cashflow + self._gas_cashflow

    def _get_attrs_for_results(self) -> dict:
        """
        Collect attributes and their corresponding names for oil, gas,
        and consolidated results.

        This method gathers time series attributes (e.g., revenues, expenditures,
        taxes, sunk costs, and cashflows) for each fluid type (oil, gas, and consolidated)
        and organizes them into a structured dictionary. It also provides a consistent
        set of attribute names that can be used for labeling results in further processing
        or DataFrame construction.

        Returns
        -------
        dict
            A dictionary containing two keys:

            - ``"attributes"`` : dict
                Maps each fluid type (``"oil"``, ``"gas"``, ``"consolidated"``)
                to a list of NumPy arrays or attributes representing project years,
                revenues, costs, taxes, and cashflows.

            - ``"names"`` : list of str
                A list of attribute names corresponding to the order of attributes
                in each fluid type. The naming convention uses a placeholder ``f``
                for the fluid type (e.g., ``f_revenue`` becomes ``oil_revenue``
                when applied to oil).

        Notes
        -----
        - The attributes cover financial elements such as:
          revenues, weighted average prices, sunk costs, pre-onstream costs,
          expenditures (capital, intangible, OPEX, ASR, LBT, cost of sales),
          indirect taxes, total expenditures, and cashflows.
        - The placeholder ``f`` in the names is dynamically replaced by the
          fluid type key (``oil``, ``gas``, or ``consolidated``).
        """

        # Specify oil attributes
        oil_attrs = [
            self.project_years,
            self._oil_revenue,
            self._sulfur_revenue,
            self._electricity_revenue,
            self._co2_revenue,
            self._oil_wap_price,
            self._oil_depreciable_sunk_cost,
            self._oil_non_depreciable_sunk_cost,
            self._oil_depreciable_preonstream,
            self._oil_non_depreciable_preonstream,
            self._oil_sunk_cost,
            self._oil_preonstream,
            self._oil_capital_expenditures_pre_tax,
            self._oil_intangible_expenditures_pre_tax,
            self._oil_opex_expenditures_pre_tax,
            self._oil_asr_expenditures_pre_tax,
            self._oil_lbt_expenditures_pre_tax,
            self._oil_cost_of_sales_expenditures_pre_tax,
            self._oil_capital_indirect_tax,
            self._oil_intangible_indirect_tax,
            self._oil_opex_indirect_tax,
            self._oil_asr_indirect_tax,
            self._oil_lbt_indirect_tax,
            self._oil_cost_of_sales_indirect_tax,
            self._oil_capital_expenditures_post_tax,
            self._oil_intangible_expenditures_post_tax,
            self._oil_opex_expenditures_post_tax,
            self._oil_asr_expenditures_post_tax,
            self._oil_lbt_expenditures_post_tax,
            self._oil_cost_of_sales_expenditures_post_tax,
            self._oil_total_expenditures_pre_tax,
            self._oil_total_indirect_tax,
            self._oil_total_expenditures_post_tax,
            self._oil_non_capital,
            self._oil_cashflow,
        ]

        # Specify gas attributes
        gas_attrs = [
            self.project_years,
            self._gas_revenue,
            self._sulfur_revenue,
            self._electricity_revenue,
            self._co2_revenue,
            self._gas_wap_price,
            self._gas_depreciable_sunk_cost,
            self._gas_non_depreciable_sunk_cost,
            self._gas_depreciable_preonstream,
            self._gas_non_depreciable_preonstream,
            self._gas_sunk_cost,
            self._gas_preonstream,
            self._gas_capital_expenditures_pre_tax,
            self._gas_intangible_expenditures_pre_tax,
            self._gas_opex_expenditures_pre_tax,
            self._gas_asr_expenditures_pre_tax,
            self._gas_lbt_expenditures_pre_tax,
            self._gas_cost_of_sales_expenditures_pre_tax,
            self._gas_capital_indirect_tax,
            self._gas_intangible_indirect_tax,
            self._gas_opex_indirect_tax,
            self._gas_asr_indirect_tax,
            self._gas_lbt_indirect_tax,
            self._gas_cost_of_sales_indirect_tax,
            self._gas_capital_expenditures_post_tax,
            self._gas_intangible_expenditures_post_tax,
            self._gas_opex_expenditures_post_tax,
            self._gas_asr_expenditures_post_tax,
            self._gas_lbt_expenditures_post_tax,
            self._gas_cost_of_sales_expenditures_post_tax,
            self._gas_total_expenditures_pre_tax,
            self._gas_total_indirect_tax,
            self._gas_total_expenditures_post_tax,
            self._gas_non_capital,
            self._gas_cashflow,
        ]

        # Specify consolidated attributes
        consolidated_attrs = [
            self.project_years,
            self._consolidated_revenue,
            self._sulfur_revenue,
            self._electricity_revenue,
            self._co2_revenue,
            self._consolidated_wap_price,
            self._consolidated_depreciable_sunk_cost,
            self._consolidated_non_depreciable_sunk_cost,
            self._consolidated_depreciable_preonstream,
            self._consolidated_non_depreciable_preonstream,
            self._consolidated_sunk_cost,
            self._consolidated_preonstream,
            self._consolidated_capital_expenditures_pre_tax,
            self._consolidated_intangible_expenditures_pre_tax,
            self._consolidated_opex_expenditures_pre_tax,
            self._consolidated_asr_expenditures_pre_tax,
            self._consolidated_lbt_expenditures_pre_tax,
            self._consolidated_cost_of_sales_expenditures_pre_tax,
            self._consolidated_capital_indirect_tax,
            self._consolidated_intangible_indirect_tax,
            self._consolidated_opex_indirect_tax,
            self._consolidated_asr_indirect_tax,
            self._consolidated_lbt_indirect_tax,
            self._consolidated_cost_of_sales_indirect_tax,
            self._consolidated_capital_expenditures_post_tax,
            self._consolidated_intangible_expenditures_post_tax,
            self._consolidated_opex_expenditures_post_tax,
            self._consolidated_asr_expenditures_post_tax,
            self._consolidated_lbt_expenditures_post_tax,
            self._consolidated_cost_of_sales_expenditures_post_tax,
            self._consolidated_total_expenditures_pre_tax,
            self._consolidated_total_indirect_tax,
            self._consolidated_total_expenditures_post_tax,
            self._consolidated_non_capital,
            self._consolidated_cashflow,
        ]

        # Store attributes as a dictionary
        attrs = {
            "oil": oil_attrs,
            "gas": gas_attrs,
            "consolidated": consolidated_attrs,
        }

        attrs_name = [
            "years",
            "f_revenue",
            "sulfur_revenue",
            "electricity_revenue",
            "co2_revenue",
            "f_wap_price",
            "f_depre_sunkcost",
            "f_non_depre_sunkcost",
            "f_depre_preonstream",
            "f_non_depre_preonstream",
            "f_sunkcost",
            "f_preonstream",
            "f_capital_exp_pretax",
            "f_intangible_exp_pretax",
            "f_opex_exp_pretax",
            "f_asr_exp_pretax",
            "f_lbt_exp_pretax",
            "f_cost_of_sales_exp_pretax",
            "f_capital_indirect_tax",
            "f_intangible_indirect_tax",
            "f_opex_indirect_tax",
            "f_asr_indirect_tax",
            "f_lbt_indirect_tax",
            "f_cost_of_sales_indirect_tax",
            "f_capital_exp_posttax",
            "f_intangible_exp_posttax",
            "f_opex_exp_posttax",
            "f_asr_exp_posttax",
            "f_lbt_exp_posttax",
            "f_cost_of_sales_exp_posttax",
            "f_total_exp_pretax",
            "f_total_indirect_tax",
            "f_total_exp_posttax",
            "f_non_capital",
            "f_cashflow",
        ]

        return {
            "attributes": attrs,
            "names": attrs_name,
        }

    def _prepare_results(self) -> np.ndarray:
        """
        Prepare and structure calculation results into DataFrames.

        This method aligns attributes for oil, gas, and consolidated results,
        validates their consistency, and organizes them into tabular format.
        Each fluid type is represented as a DataFrame with project years as
        the index and attribute names as columns.

        Returns
        -------
        dict of {str: pandas.DataFrame}
            A dictionary mapping fluid types to their corresponding results:

            - ``"oil"`` : DataFrame
                Tabular results for oil, with each column corresponding to a
                financial or project attribute (e.g., revenue, costs, cashflow).
            - ``"gas"`` : DataFrame
                Tabular results for gas, structured in the same way as oil.
            - ``"consolidated"`` : DataFrame
                Consolidated results across all fluids, structured similarly.

        Raises
        ------
        BaseProjectException
            If the number of attributes or names is inconsistent across fluid types.

        Notes
        -----
        - Internally, results are stored in a 3D NumPy array of shape
          ``(n_fluids, project_duration, n_attributes)`` before being
          converted to DataFrames.
        - Column names are derived from the standardized names returned by
          :meth:`_get_attrs_for_results`.
        """

        # Define attributes and names of the attributes
        attrs = self._get_attrs_for_results()
        attributes = attrs["attributes"]
        names = attrs["names"]

        fluids = ["oil", "gas", "consolidated"]

        # Ensure consistency
        attributes_names_length = [len(attributes[f]) for f in fluids]
        attributes_names_length.append(len(names))

        if np.unique(attributes_names_length).size != 1:
            raise BaseProjectException("Mismatch in attribute lengths across fluids")

        # Create a 3D NumPy array to store calculation results
        n_cols = attributes_names_length[0]
        results = np.full(
            (len(fluids), self.project_duration, n_cols),
            fill_value=np.nan, dtype=np.float64
        )

        for i, key in enumerate(["oil", "gas", "consolidated"]):
            for idx in range(n_cols):
                results[i, :, idx] = attributes[key][idx]

        return {
            key: pd.DataFrame(results[i, :, :], columns=names)
            for i, key in enumerate(fluids)
        }

    def get_results(self, ftype: str = "oil", chunk_size: int = 5) -> pd.DataFrame:
        """
        Print calculation results for a specified fluid type in chunks.

        This method retrieves the prepared results for the given fluid type and
        prints them in column-wise chunks to improve readability, especially when
        the number of columns is large.

        Parameters
        ----------
        chunk_size : int
            The number of columns to display in each printed chunk.
        ftype : str
            The fluid type to display results for. Must be one of:
            {"oil", "gas", "consolidated"}.

        Returns
        -------
        None
            The method prints the results to the console and does not return any value.

        Raises
        ------
        KeyError
            If `ftype` is not one of {"oil", "gas", "consolidated"}.
        """

        df_map: dict = self._prepare_results()

        def _prepare_print(chunk_size: int, df: pd.DataFrame):
            cols = df.columns.tolist()

            print('\t')
            print(f"Fluid: {ftype}")
            print(f"========================================================")

            for i in range(0, len(cols), chunk_size):
                print(f"\nColumns {i + 1} to {min(i + chunk_size, len(cols))}: ")
                print(df[cols[i:i + chunk_size]])

        _prepare_print(chunk_size=chunk_size, df=df_map[ftype])

    def run(
        self,
        sulfur_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_GAS_REVENUE,
        electricity_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_OIL_REVENUE,
        co2_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_GAS_REVENUE,
        tax_rate: np.ndarray | float = 0.0,
        year_inflation: np.ndarray = None,
        inflation_rate: np.ndarray | float = 0.0,
        inflation_rate_applied_to: InflationAppliedTo = None,
    ) -> None:
        """
        Executes the full economic calculation workflow for a base project.

        This method performs the complete sequence of calculations required to produce
        oil, gas, and consolidated economic profiles. It validates sunk costs, computes
        expenditures (pre-tax and post-tax), indirect taxes, other revenues, and
        cashflows, then prepares the consolidated profiles for reporting or further analysis.

        Parameters
        ----------
        sulfur_revenue : OtherRevenue, default=OtherRevenue.ADDITION_TO_GAS_REVENUE
            How sulfur revenue is allocated (e.g., added to gas revenue or treated separately).
        electricity_revenue : OtherRevenue, default=OtherRevenue.ADDITION_TO_OIL_REVENUE
            How electricity revenue is allocated.
        co2_revenue : OtherRevenue, default=OtherRevenue.ADDITION_TO_GAS_REVENUE
            How CO₂ revenue is allocated.
        tax_rate : np.ndarray or float, default=0.0
            Indirect tax rate(s) applied to applicable expenditures.
        year_inflation : np.ndarray, optional
            Array specifying the inflation year for each time step.
        inflation_rate : np.ndarray or float, default=0.0
            Inflation rate(s) to be applied.
        inflation_rate_applied_to : InflationAppliedTo, optional
            Specifies which expenditures the inflation rate should be applied to.
        """

        # WAP (Weighted Average Price) for each produced fluid
        self._get_wap_price()

        # Validate sunk cost, pre-onstream, and post-onstream objects
        self._get_cost_objects_validation()

        # Prepare sunk costs and preonstream costs
        self._get_sunkcost_array()
        self._get_preonstream_array()

        # Calculate (total = depreciable + non_depreciable costs)
        # for sunk cost and preonstream cost
        for ctype in ["sunk_cost", "preonstream"]:
            for ftype in ["oil", "gas"]:
                depreciable = getattr(self, f"_{ftype}_depreciable_{ctype}")
                non_depreciable = getattr(self, f"_{ftype}_non_depreciable_{ctype}")
                setattr(self, f"_{ftype}_{ctype}", depreciable + non_depreciable)

        # Calculate pre tax expenditures
        self._get_expenditures_pre_tax(
            year_inflation=year_inflation,
            inflation_rate=inflation_rate,
            inflation_rate_applied_to=inflation_rate_applied_to,
        )

        # Calculate indirect taxes
        self._get_indirect_taxes(tax_rate=tax_rate)

        # Calculate post tax expenditures
        self._get_expenditures_post_tax()

        # Other revenue
        self._get_other_revenue(
            sulfur_revenue=sulfur_revenue,
            electricity_revenue=electricity_revenue,
            co2_revenue=co2_revenue,
        )

        # Non-capital costs (intangible + opex + asr + lbt + cost of sales)
        self._oil_non_capital = (
            self._oil_intangible_expenditures_post_tax
            + self._oil_opex_expenditures_post_tax
            + self._oil_asr_expenditures_post_tax
            + self._oil_lbt_expenditures_post_tax
            + self._oil_cost_of_sales_expenditures_post_tax
        )

        self._gas_non_capital = (
            self._gas_intangible_expenditures_post_tax
            + self._gas_opex_expenditures_post_tax
            + self._gas_asr_expenditures_post_tax
            + self._gas_lbt_expenditures_post_tax
            + self._gas_cost_of_sales_expenditures_post_tax
        )

        # Total OIL pre-tax expenditures
        self._oil_total_expenditures_pre_tax = (
            self._oil_capital_expenditures_pre_tax
            + self._oil_intangible_expenditures_pre_tax
            + self._oil_opex_expenditures_pre_tax
            + self._oil_asr_expenditures_pre_tax
            + self._oil_lbt_expenditures_pre_tax
            + self._oil_cost_of_sales_expenditures_pre_tax
        )

        # Total GAS pre-tax expenditures
        self._gas_total_expenditures_pre_tax = (
            self._gas_capital_expenditures_pre_tax
            + self._gas_intangible_expenditures_pre_tax
            + self._gas_opex_expenditures_pre_tax
            + self._gas_asr_expenditures_pre_tax
            + self._gas_lbt_expenditures_pre_tax
            + self._gas_cost_of_sales_expenditures_pre_tax
        )

        # Total OIL indirect taxes
        self._oil_total_indirect_tax = (
            self._oil_capital_indirect_tax
            + self._oil_intangible_indirect_tax
            + self._oil_opex_indirect_tax
            + self._oil_asr_indirect_tax
            + self._oil_lbt_indirect_tax
            + self._oil_cost_of_sales_indirect_tax
        )

        # Total GAS indirect taxes
        self._gas_total_indirect_tax = (
            self._gas_capital_indirect_tax
            + self._gas_intangible_indirect_tax
            + self._gas_opex_indirect_tax
            + self._gas_asr_indirect_tax
            + self._gas_lbt_indirect_tax
            + self._gas_cost_of_sales_indirect_tax
        )

        # Total OIL post-tax expenditures
        self._oil_total_expenditures_post_tax = (
            self._oil_capital_expenditures_post_tax
            + self._oil_intangible_expenditures_post_tax
            + self._oil_opex_expenditures_post_tax
            + self._oil_asr_expenditures_post_tax
            + self._oil_lbt_expenditures_post_tax
            + self._oil_cost_of_sales_expenditures_post_tax
        )

        # Total GAS post-tax expenditures
        self._gas_total_expenditures_post_tax = (
            self._gas_capital_expenditures_post_tax
            + self._gas_intangible_expenditures_post_tax
            + self._gas_opex_expenditures_post_tax
            + self._gas_asr_expenditures_post_tax
            + self._gas_lbt_expenditures_post_tax
            + self._gas_cost_of_sales_expenditures_post_tax
        )

        # Configure base cashflow for OIL and GAS
        self._oil_cashflow = self._oil_revenue - (
            self._oil_sunk_cost
            + self._oil_preonstream
            + self._oil_total_expenditures_post_tax
        )

        self._gas_cashflow = self._gas_revenue - (
            self._gas_sunk_cost
            + self._gas_preonstream
            + self._gas_total_expenditures_post_tax
        )

        # Prepare consolidated profiles
        self._get_consolidated_profiles()

    def __len__(self):
        return self.project_duration
