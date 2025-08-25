"""
Configure base project as the foundation (or parent class) for PSC contract.
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import date
from functools import reduce

from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.selection import (
    FluidType,
    CostType,
    ExpendituresType,
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


class PreonstreamCostException(Exception):
    """ Exception to be raised for an incorrect preonstream cost configuration """

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

    # Attributes associated with sunk costs
    _oil_depreciable_sunk_cost: np.ndarray = field(default=None, init=False, repr=False)
    _gas_depreciable_sunk_cost: np.ndarray = field(default=None, init=False, repr=False)
    _oil_non_depreciable_sunk_cost: np.ndarray = field(default=None, init=False, repr=False)
    _gas_non_depreciable_sunk_cost: np.ndarray = field(default=None, init=False, repr=False)
    _oil_sunk_cost: np.ndarray = field(default=None, init=False, repr=False)
    _gas_sunk_cost: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes associated with preonstream costs
    _oil_depreciable_preonstream: np.ndarray = field(default=None, init=False, repr=False)
    _gas_depreciable_preonstream: np.ndarray = field(default=None, init=False, repr=False)
    _oil_non_depreciable_preonstream: np.ndarray = field(default=None, init=False, repr=False)
    _gas_non_depreciable_preonstream: np.ndarray = field(default=None, init=False, repr=False)
    _oil_preonstream: np.ndarray = field(default=None, init=False, repr=False)
    _gas_preonstream: np.ndarray = field(default=None, init=False, repr=False)

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
    _consolidated_expenditures_pre_tax: np.ndarray = field(
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
    _consolidated_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)

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
    _consolidated_expenditures_post_tax: np.ndarray = field(
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

        for costs in costs_list:
            for ftype in [FluidType.OIL, FluidType.GAS]:
                self._prepare_cost_types(
                    cost_obj=costs[ftype.name.lower()],
                    fluid_type=ftype,
                )

        # Define postonstream, sunk cost, and preonstream attributes
        cost_groups = (
            ("capital", self._filter_capital_cost, capital_cost),
            ("intangible", self._filter_intangible, intangible),
            ("opex", self._filter_opex, opex),
            ("asr", self._filter_asr, asr),
            ("lbt", self._filter_lbt, lbt),
            ("cost_of_sales", self._filter_cost_of_sales, cost_of_sales)
        )

        categories = (
            ("postonstream", CostType.POST_ONSTREAM_COST),
            ("sunk_cost", CostType.SUNK_COST),
            ("preonstream", CostType.PRE_ONSTREAM_COST),
        )

        for prefix, filter_func, source in cost_groups:
            for ftype in ("oil", "gas"):
                for categ, ctype in categories:
                    setattr(
                        self,
                        f"_{ftype}_{prefix}_{categ}",
                        filter_func(cost_obj=source[ftype], include_cost_type=ctype)
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

    def _validate_approval_year(self, fluid_type: FluidType) -> None:
        """
        Validate and set the POD I approval year against project and fluid timelines.

        This method ensures that the approval year is valid relative to the project's
        start and end dates, as well as the onstream year for the specified fluid type.
        If `approval_year` is not set (None), it defaults to the fluid's onstream year.
        Otherwise, it performs consistency checks to prevent invalid approval years.

        Parameters
        ----------
        fluid_type : FluidType
            The fluid type (`FluidType.OIL` or `FluidType.GAS`) used to determine the
            corresponding onstream year for validation.

        Raises
        ------
        BaseProjectException
            If `approval_year` is not an integer.
            If `approval_year` is earlier than the project start year.
            If `approval_year` is later than the project end year.
            If `approval_year` is later than the fluid onstream year.

        Notes
        -----
        - If `approval_year` is not provided, it is automatically set to the onstream year
          of the given fluid type.
        - The validation prevents approval years outside the project timeframe or inconsistent
          with the fluid's onstream year.
        """

        onstream_yr = {
            FluidType.OIL: self.oil_onstream_date.year,
            FluidType.GAS: self.gas_onstream_date.year,
        }[fluid_type]

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
                    f"{fluid_type.name.lower()} onstream year ({onstream_yr})"
                )

    def _prepare_cost_types(
        self,
        fluid_type: FluidType,
        cost_obj: CapitalCost | Intangible | OPEX | ASR | LBT | CostOfSales,
    ) -> None:
        """
        Assign and validate cost types for a given cost object based on
        project timelines and fluid type.

        This method classifies each expense in `cost_obj` into one of three
        categories:
        - PRE_ONSTREAM_COST: Expenses occurring between approval year and onstream year.
        - POST_ONSTREAM_COST: Expenses occurring after the fluid's onstream year.
        - SUNK_COST: Expenses occurring before the approval year.

        The method also validates that classifications are consistent with
        project rules, including exact boundary checks for approval and
        onstream years.

        Parameters
        ----------
        fluid_type : FluidType
            The fluid type associated with the costs. Typically ``FluidType.OIL``
            or ``FluidType.GAS``. Determines which onstream year is used.

        cost_obj : CapitalCost | Intangible | OPEX | ASR | LBT | CostOfSales
            The cost object whose expense years will be classified into cost types.

        Returns
        -------
        None
            The method modifies the `cost_type` attribute of `cost_obj` in place.

        Raises
        ------
        BaseProjectException
            -   If any assignment of cost types is inconsistent with the expected
                classification.
            -   If POST_ONSTREAM_COST is assigned at the approval year.
            -   If SUNK_COST is assigned at the fluid's onstream year.

        Notes
        -----
        -   The method first validates that `approval_year` is within
            project start and end dates.
        -   Cost type assignment is done using boolean masks on `expense_year`.
        -   Boundary years (approval and onstream) are checked explicitly to
            prevent invalid classifications.
        """

        # Validate approval_year
        self._validate_approval_year(fluid_type=fluid_type)

        # Specify relevant onstream year corresponds to OIL or GAS
        onstream_year = {
            FluidType.OIL: self.oil_onstream_date.year,
            FluidType.GAS: self.gas_onstream_date.year
        }[fluid_type]

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
            if mask.any() and not np.all(ct[mask] == expected):
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
        self, cost_obj: CapitalCost, include_cost_type: CostType
    ) -> CapitalCost:
        """
        Filter a CapitalCost object by a specific cost type.

        This method extracts only the entries from `cost_obj` whose
        `cost_type` matches the specified `include_cost_type`. If no
        matching entries exist, it returns a placeholder `CapitalCost`
        object with zero cost at a default reference year.

        Parameters
        ----------
        cost_obj : CapitalCost
            The input CapitalCost object containing attributes such as
            `expense_year`, `cost`, `cost_allocation`, and others.

        include_cost_type : CostType
            The cost type to filter for (e.g.,
            ``CostType.SUNK_COST``, ``CostType.PRE_ONSTREAM_COST``,
            or ``CostType.POST_ONSTREAM_COST``).

        Returns
        -------
        CapitalCost
            A new CapitalCost object containing only the entries that
            match `include_cost_type`. If no entries match, a placeholder
            object with zero cost is returned.

        Notes
        -----
        -   Default reference year for placeholder objects:
              * ``CostType.SUNK_COST`` → project start year.
              * ``CostType.PRE_ONSTREAM_COST`` → project approval year.
              * ``CostType.POST_ONSTREAM_COST`` → oil or gas onstream year
                (depending on the `cost_allocation`).
        -   A default `cost_allocation` is inferred: if
            ``FluidType.OIL`` exists in the original allocation, it is used;
            otherwise, ``FluidType.GAS`` is assigned.
        -   All relevant attributes (e.g., `description`, `tax_portion`,
            `useful_life`, etc.) are filtered consistently using the same
            boolean mask.
        -   Returned attributes are converted to lists where applicable.
        """

        kwargs = {
            "start_year": cost_obj.start_year,
            "end_year": cost_obj.end_year,
        }

        onstream_yr = (
            self.oil_onstream_date.year if FluidType.OIL in cost_obj.cost_allocation
            else self.gas_onstream_date.year
        )

        allocation = (
            FluidType.OIL if FluidType.OIL in cost_obj.cost_allocation else FluidType.GAS
        )

        ey = {
            CostType.SUNK_COST: cost_obj.start_year,
            CostType.PRE_ONSTREAM_COST: self.approval_year,
            CostType.POST_ONSTREAM_COST: onstream_yr,
        }[include_cost_type]

        cost_type_array = np.array(cost_obj.cost_type)
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
            expense_year=cost_obj.expense_year[mask],
            cost=cost_obj.cost[mask],
            cost_allocation=np.array(cost_obj.cost_allocation)[mask].tolist(),
            cost_type=np.array(cost_obj.cost_type)[mask].tolist(),
            description=np.array(cost_obj.description)[mask].tolist(),
            tax_portion=cost_obj.tax_portion[mask],
            tax_discount=cost_obj.tax_discount[mask],
            pis_year=cost_obj.pis_year[mask],
            salvage_value=cost_obj.salvage_value[mask],
            useful_life=cost_obj.useful_life[mask],
            depreciation_factor=cost_obj.depreciation_factor[mask],
            is_ic_applied=np.array(cost_obj.is_ic_applied)[mask].tolist(),
        )

    def _filter_intangible(
        self, cost_obj: Intangible, include_cost_type: CostType
    ) -> Intangible:
        """
        Filter an Intangible cost object by a specific cost type.

        This method extracts only the entries from `cost_obj` whose
        `cost_type` matches the specified `include_cost_type`. If no
        matching entries exist, it returns a placeholder `Intangible`
        object with zero cost at a default reference year.

        Parameters
        ----------
        cost_obj : Intangible
            The input Intangible object containing attributes such as
            `expense_year`, `cost`, `cost_allocation`, and others.

        include_cost_type : CostType
            The cost type to filter for (e.g.,
            ``CostType.SUNK_COST``, ``CostType.PRE_ONSTREAM_COST``,
            or ``CostType.POST_ONSTREAM_COST``).

        Returns
        -------
        Intangible
            A new Intangible object containing only the entries that
            match `include_cost_type`. If no entries match, a placeholder
            object with zero cost is returned.

        Notes
        -----
        -   Default reference year for placeholder objects:
              * ``CostType.SUNK_COST`` → project start year.
              * ``CostType.PRE_ONSTREAM_COST`` → project approval year.
              * ``CostType.POST_ONSTREAM_COST`` → oil or gas onstream year
                (depending on the `cost_allocation`).
        -   A default `cost_allocation` is inferred: if
            ``FluidType.OIL`` exists in the original allocation, it is used;
            otherwise, ``FluidType.GAS`` is assigned.
        -   All relevant attributes (e.g., `description`, `tax_portion`,
            `tax_discount`) are filtered consistently using the same
            boolean mask.
        -   Returned attributes are converted to lists where applicable.
        """

        kwargs = {
            "start_year": cost_obj.start_year,
            "end_year": cost_obj.end_year,
        }

        onstream_yr = (
            self.oil_onstream_date.year if FluidType.OIL in cost_obj.cost_allocation
            else self.gas_onstream_date.year
        )

        allocation = (
            FluidType.OIL if FluidType.OIL in cost_obj.cost_allocation else FluidType.GAS
        )

        ey = {
            CostType.SUNK_COST: cost_obj.start_year,
            CostType.PRE_ONSTREAM_COST: self.approval_year,
            CostType.POST_ONSTREAM_COST: onstream_yr,
        }[include_cost_type]

        cost_type_array = np.array(cost_obj.cost_type)
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
            expense_year=cost_obj.expense_year[mask],
            cost=cost_obj.cost[mask],
            cost_allocation=np.array(cost_obj.cost_allocation)[mask].tolist(),
            cost_type=np.array(cost_obj.cost_type)[mask].tolist(),
            description=np.array(cost_obj.description)[mask].tolist(),
            tax_portion=cost_obj.tax_portion[mask],
            tax_discount=cost_obj.tax_discount[mask],
        )

    def _filter_opex(self, cost_obj: OPEX, include_cost_type: CostType) -> OPEX:
        """
        Filter an OPEX cost object by a specific cost type.

        This method extracts only the entries from `cost_obj` whose
        `cost_type` matches the specified `include_cost_type`. If no
        matching entries exist, it returns a placeholder `OPEX` object
        with zero cost at a default reference year.

        Parameters
        ----------
        cost_obj : OPEX
            The input OPEX object containing attributes such as
            `expense_year`, `fixed_cost`, `prod_rate`, `cost_per_volume`,
            and others.

        include_cost_type : CostType
            The cost type to filter for (e.g.,
            ``CostType.SUNK_COST``, ``CostType.PRE_ONSTREAM_COST``,
            or ``CostType.POST_ONSTREAM_COST``).

        Returns
        -------
        OPEX
            A new OPEX object containing only the entries that match
            `include_cost_type`. If no entries match, a placeholder
            object with zero fixed cost is returned.

        Notes
        -----
        -   Default reference year for placeholder objects:
              * ``CostType.SUNK_COST`` → project start year.
              * ``CostType.PRE_ONSTREAM_COST`` → project approval year.
              * ``CostType.POST_ONSTREAM_COST`` → oil or gas onstream year
                (depending on the `cost_allocation`).
        -   A default `cost_allocation` is inferred: if
            ``FluidType.OIL`` exists in the original allocation, it is used;
            otherwise, ``FluidType.GAS`` is assigned.
        -   All relevant attributes (e.g., `description`, `tax_portion`,
            `tax_discount`, `prod_rate`, `cost_per_volume`) are filtered
            consistently using the same boolean mask.
        -   Returned attributes are converted to lists where applicable.
        """

        kwargs = {
            "start_year": cost_obj.start_year,
            "end_year": cost_obj.end_year,
        }

        onstream_yr = (
            self.oil_onstream_date.year if FluidType.OIL in cost_obj.cost_allocation
            else self.gas_onstream_date.year
        )

        allocation = (
            FluidType.OIL if FluidType.OIL in cost_obj.cost_allocation else FluidType.GAS
        )

        ey = {
            CostType.SUNK_COST: cost_obj.start_year,
            CostType.PRE_ONSTREAM_COST: self.approval_year,
            CostType.POST_ONSTREAM_COST: onstream_yr
        }[include_cost_type]

        cost_type_array = np.array(cost_obj.cost_type)
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
            expense_year=cost_obj.expense_year[mask],
            cost_allocation=np.array(cost_obj.cost_allocation)[mask].tolist(),
            cost_type=np.array(cost_obj.cost_type)[mask].tolist(),
            description=np.array(cost_obj.description)[mask].tolist(),
            tax_portion=cost_obj.tax_portion[mask],
            tax_discount=cost_obj.tax_discount[mask],
            fixed_cost=cost_obj.fixed_cost[mask],
            prod_rate=cost_obj.prod_rate[mask],
            cost_per_volume=cost_obj.cost_per_volume[mask],
        )

    def _filter_asr(self, cost_obj: ASR, include_cost_type: CostType) -> ASR:
        """
        Filter an ASR (Abandonment and Site Restoration) cost object by
        a specific cost type.

        This method extracts only the entries from `cost_obj` whose
        `cost_type` matches the specified `include_cost_type`. If no
        matching entries exist, it returns a placeholder `ASR` object
        with zero cost at a default reference year.

        Parameters
        ----------
        cost_obj : ASR
            The input ASR object containing attributes such as
            `expense_year`, `cost`, `final_year`, and `future_rate`.

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
              * ``CostType.POST_ONSTREAM_COST`` → oil or gas onstream year
                (depending on the `cost_allocation`).
        -   A default `cost_allocation` is inferred: if
            ``FluidType.OIL`` exists in the original allocation, it is used;
            otherwise, ``FluidType.GAS`` is assigned.
        -   All relevant attributes (e.g., `description`, `tax_portion`,
            `tax_discount`, `final_year`, `future_rate`) are filtered
            consistently using the same boolean mask.
        -   Returned attributes are converted to lists where applicable.
        """

        kwargs = {
            "start_year": cost_obj.start_year,
            "end_year": cost_obj.end_year,
        }

        onstream_yr = (
            self.oil_onstream_date.year if FluidType.OIL in cost_obj.cost_allocation
            else self.gas_onstream_date.year
        )

        allocation = (
            FluidType.OIL if FluidType.OIL in cost_obj.cost_allocation else FluidType.GAS
        )

        ey = {
            CostType.SUNK_COST: cost_obj.start_year,
            CostType.PRE_ONSTREAM_COST: self.approval_year,
            CostType.POST_ONSTREAM_COST: onstream_yr
        }[include_cost_type]

        cost_type_array = np.array(cost_obj.cost_type)
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
            expense_year=cost_obj.expense_year[mask],
            cost=cost_obj.cost[mask],
            cost_allocation=np.array(cost_obj.cost_allocation)[mask].tolist(),
            cost_type=np.array(cost_obj.cost_type)[mask].tolist(),
            description=np.array(cost_obj.description)[mask].tolist(),
            tax_portion=cost_obj.tax_portion[mask],
            tax_discount=cost_obj.tax_discount[mask],
            final_year=cost_obj.final_year[mask],
            future_rate=cost_obj.future_rate[mask],
        )

    def _filter_lbt(self, cost_obj: LBT, include_cost_type: CostType) -> LBT:
        """
        Filter an LBT (Land and Building Tax) cost object by a specific cost type.

        This method extracts only the entries from `cost_obj` whose
        `cost_type` matches the specified `include_cost_type`. If no
        matching entries exist, it returns a placeholder `LBT` object
        with zero cost at a default reference year.

        Parameters
        ----------
        cost_obj : LBT
            The input LBT object containing attributes such as
            `expense_year`, `cost`, `njop_land`, `njop_building`,
            `utilized_land_area`, and `gross_revenue`.

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
              * ``CostType.POST_ONSTREAM_COST`` → oil or gas onstream year
                (depending on the `cost_allocation`).
        -   A default `cost_allocation` is inferred: if
            ``FluidType.OIL`` exists in the original allocation, it is used;
            otherwise, ``FluidType.GAS`` is assigned.
        -   All relevant attributes (e.g., `description`, `tax_portion`,
            `tax_discount`, `final_year`, `utilized_land_area`,
            `utilized_building_area`, `njop_land`, `njop_building`,
            `gross_revenue`, and `cost`) are filtered consistently using
            the same boolean mask.
        -   Returned attributes are converted to lists where applicable.
        """

        kwargs = {
            "start_year": cost_obj.start_year,
            "end_year": cost_obj.end_year,
        }

        onstream_yr = (
            self.oil_onstream_date.year if FluidType.OIL in cost_obj.cost_allocation
            else self.gas_onstream_date.year
        )

        allocation = (
            FluidType.OIL if FluidType.OIL in cost_obj.cost_allocation else FluidType.GAS
        )

        ey = {
            CostType.SUNK_COST: cost_obj.start_year,
            CostType.PRE_ONSTREAM_COST: self.approval_year,
            CostType.POST_ONSTREAM_COST: onstream_yr,
        }[include_cost_type]

        cost_type_array = np.array(cost_obj.cost_type)
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
            expense_year=cost_obj.expense_year[mask],
            cost_allocation=np.array(cost_obj.cost_allocation)[mask].tolist(),
            cost_type=np.array(cost_obj.cost_type)[mask].tolist(),
            description=np.array(cost_obj.description)[mask].tolist(),
            tax_portion=cost_obj.tax_portion[mask],
            tax_discount=cost_obj.tax_discount[mask],
            final_year=cost_obj.final_year[mask],
            utilized_land_area=cost_obj.utilized_land_area[mask],
            utilized_building_area=cost_obj.utilized_building_area[mask],
            njop_land=cost_obj.njop_land[mask],
            njop_building=cost_obj.njop_building[mask],
            gross_revenue=cost_obj.gross_revenue[mask],
            cost=cost_obj.cost[mask],
        )

    def _filter_cost_of_sales(
        self, cost_obj: CostOfSales, include_cost_type: CostType
    ) -> CostOfSales:
        """
        Filter a CostOfSales object by a specific cost type.

        This method extracts only the entries from `cost_obj` whose
        `cost_type` matches the specified `include_cost_type`. If no
        matching entries exist, it returns a placeholder `CostOfSales`
        object with zero cost at a default reference year.

        Parameters
        ----------
        cost_obj : CostOfSales
            The input CostOfSales object containing attributes such as
            `expense_year`, `cost`, `description`, `tax_portion`, and
            `tax_discount`.

        include_cost_type : CostType
            The cost type to filter for (e.g.,
            ``CostType.SUNK_COST``, ``CostType.PRE_ONSTREAM_COST``,
            or ``CostType.POST_ONSTREAM_COST``).

        Returns
        -------
        CostOfSales
            A new CostOfSales object containing only the entries that
            match `include_cost_type`. If no entries match, a placeholder
            object with zero cost is returned.

        Notes
        -----
        -   Default reference year for placeholder objects:
              * ``CostType.SUNK_COST`` → project start year.
              * ``CostType.PRE_ONSTREAM_COST`` → project approval year.
              * ``CostType.POST_ONSTREAM_COST`` → oil or gas onstream year
                (depending on the `cost_allocation`).
        -   A default `cost_allocation` is inferred: if
            ``FluidType.OIL`` exists in the original allocation, it is used;
            otherwise, ``FluidType.GAS`` is assigned.
        -   All relevant attributes (`description`, `tax_portion`,
            `tax_discount`, etc.) are filtered consistently using the
            same boolean mask.
        -   Returned attributes are converted to lists where applicable.
        """

        kwargs = {
            "start_year": cost_obj.start_year,
            "end_year": cost_obj.end_year,
        }

        onstream_yr = (
            self.oil_onstream_date.year if FluidType.OIL in cost_obj.cost_allocation
            else self.gas_onstream_date.year
        )

        allocation = (
            FluidType.OIL if FluidType.OIL in cost_obj.cost_allocation else FluidType.GAS
        )

        ey = {
            CostType.SUNK_COST: cost_obj.start_year,
            CostType.PRE_ONSTREAM_COST: self.approval_year,
            CostType.POST_ONSTREAM_COST: onstream_yr,
        }[include_cost_type]

        cost_type_array = np.array(cost_obj.cost_type)
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
            expense_year=cost_obj.expense_year[mask],
            cost=cost_obj.cost[mask],
            cost_allocation=np.array(cost_obj.cost_allocation)[mask].tolist(),
            cost_type=np.array(cost_obj.cost_type)[mask].tolist(),
            description=np.array(cost_obj.description)[mask].tolist(),
            tax_portion=cost_obj.tax_portion[mask],
            tax_discount=cost_obj.tax_discount[mask],
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

    def _validate_sunkcost(self, fluid_type: FluidType, sunkcost_objects: list) -> None:
        """
        Validate that sunk cost years do not exceed the onstream year.

        This method checks whether the maximum expense year of any sunk cost
        object occurs after the onstream year for the specified `fluid_type`.
        If such cases are found, a `SunkCostException` is raised.

        Parameters
        ----------
        fluid_type : FluidType
            The fluid type for which sunk cost validation is performed.
            Must be either ``FluidType.OIL`` or ``FluidType.GAS``.

        sunkcost_objects : list of CapitalCost | Intangible | OPEX | ASR | LBT | CostOfSales
            A list of sunk cost objects, each containing an ``expense_year``
            attribute from which the maximum year is extracted.

        Raises
        ------
        SunkCostException
            If any sunk cost object's maximum expense year exceeds the
            onstream year of the given `fluid_type`.

        Notes
        -----
        - The onstream year is determined by ``oil_onstream_date.year`` or
          ``gas_onstream_date.year`` depending on the `fluid_type`.
        - This validation ensures sunk costs are incurred only before or up
          to the respective onstream year.
        """

        onstream_yr = {
            FluidType.OIL: self.oil_onstream_date.year,
            FluidType.GAS: self.gas_onstream_date.year,
        }[fluid_type]

        sunkcost_max_years = np.array([np.max(sc.expense_year) for sc in sunkcost_objects])
        mask = sunkcost_max_years > onstream_yr

        if np.any(mask):
            incorrect_years = sunkcost_max_years[mask]
            raise SunkCostException(
                f"Sunk cost years ({incorrect_years}) exceed {fluid_type.name.lower()} "
                f"onstream year ({onstream_yr})"
            )

    def _validate_preonstream(
        self, fluid_type: FluidType, preonstream_objects: list
    ) -> None:
        """
        Validate that all pre-onstream cost objects fall within the allowable project years.

        This method checks that the expense years of each object in `preonstream_objects`
        lie within the range defined by the project's `approval_year` and the fluid's
        onstream year. It considers both the minimum and maximum expense years of each
        object. If any object falls outside this range, an exception is raised.

        Parameters
        ----------
        fluid_type : FluidType
            The type of fluid being validated (e.g., `FluidType.OIL` or `FluidType.GAS`).
            Determines the relevant onstream year for the validation.

        preonstream_objects : list
            A list of pre-onstream cost objects (e.g., CapitalCost, Intangible, OPEX, ASR,
            LBT, or CostOfSales). Each object must have an `expense_year` attribute that
            is iterable (e.g., NumPy array or list) representing the years of expenses.

        Raises
        ------
        PreonstreamCostException
            If any expense year in any object is less than the project's `approval_year`
            or exceeds the onstream year for the given fluid. The exception message
            includes the `(min_year, max_year)` ranges that are invalid.

        Notes
        -----
        - The method uses both minimum and maximum expense years for each cost object
          to ensure the entire range is valid.
        - This is typically used to enforce correct sequencing of pre-onstream costs
          relative to project approvals and onstream dates.
        """

        onstream_yr = {
            FluidType.OIL: self.oil_onstream_date.year,
            FluidType.GAS: self.gas_onstream_date.year,
        }[fluid_type]

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
            raise PreonstreamCostException(
                f"Preonstream years ({incorrect_years}) fall outside the allowable "
                f"range: {self.approval_year} <= preonstream_years <= {onstream_yr}"
            )

    def _get_sunkcost_validation(self) -> None:
        """
        Validate all sunk cost objects for OIL and GAS fluids.

        This method iterates over all sunk cost objects associated with OIL and GAS and
        validates that their expense years do not exceed the corresponding fluid's
        onstream year. Validation is delegated to the `_validate_sunkcost` method.

        Raises
        ------
        SunkCostException
            If any expense year of a sunk cost object exceeds the onstream year for its
            respective fluid type.

        Notes
        -----
        - The method covers all relevant sunk cost categories: capital, intangible,
          OPEX, ASR, LBT, and cost of sales.
        - Uses `FluidType.OIL` and `FluidType.GAS` to determine the applicable onstream
          year for each group of cost objects.
        - Internally relies on `_validate_sunkcost` for the actual year range checks.
        """

        fluid_object_map = {
            FluidType.OIL: [
                self._oil_capital_sunk_cost,
                self._oil_intangible_sunk_cost,
                self._oil_opex_sunk_cost,
                self._oil_asr_sunk_cost,
                self._oil_lbt_sunk_cost,
                self._oil_cost_of_sales_sunk_cost,
            ],
            FluidType.GAS: [
                self._gas_capital_sunk_cost,
                self._gas_intangible_sunk_cost,
                self._gas_opex_sunk_cost,
                self._gas_asr_sunk_cost,
                self._gas_lbt_sunk_cost,
                self._gas_cost_of_sales_sunk_cost,
            ],
        }

        for ftype, obj in fluid_object_map.items():
            self._validate_sunkcost(fluid_type=ftype, sunkcost_objects=obj)

    def _get_preonstream_validation(self) -> None:
        """
        Validate all pre-onstream cost objects for OIL and GAS fluids.

        This method iterates over all pre-onstream cost objects associated with OIL and GAS
        and validates that their expense years fall within the allowable range, from the
        project approval year up to the corresponding fluid's onstream year. Validation is
        performed using the `_validate_preonstream` method.

        Raises
        ------
        PreonstreamCostException
            If any pre-onstream cost object's expense year is earlier than the approval year
            or later than the onstream year for its respective fluid type.

        Notes
        -----
        - The method covers all relevant pre-onstream categories: capital, intangible, OPEX,
          ASR, LBT, and cost of sales.
        - Uses `FluidType.OIL` and `FluidType.GAS` to determine the applicable onstream
          year for each group of cost objects.
        - Internally relies on `_validate_preonstream` for the actual year range checks.
        """

        fluid_object_map = {
            FluidType.OIL: [
                self._oil_capital_preonstream,
                self._oil_intangible_preonstream,
                self._oil_opex_preonstream,
                self._oil_asr_preonstream,
                self._oil_lbt_preonstream,
                self._oil_cost_of_sales_preonstream,
            ],
            FluidType.GAS: [
                self._gas_capital_preonstream,
                self._gas_intangible_preonstream,
                self._gas_opex_preonstream,
                self._gas_asr_preonstream,
                self._gas_lbt_preonstream,
                self._gas_cost_of_sales_preonstream,
            ]
        }

        for ftype, obj in fluid_object_map.items():
            self._validate_preonstream(fluid_type=ftype, preonstream_objects=obj)

    def _get_sunkcost_array(self) -> None:
        """
        Validate and prepare sunk cost arrays for OIL and GAS.

        This method validates all sunk cost objects for OIL and GAS
        using `_get_sunkcost_validation`. It then constructs arrays of
        pre-tax expenditures for each sunk cost category, aggregates
        them into depreciable and non-depreciable components, and finally
        computes the total sunk costs for both fluids.
    
        Raises
        ------
        SunkCostException
            If any sunk cost years exceed the respective fluid's onstream year,
            as enforced by `_get_sunkcost_validation`.
    
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
            * ``_oil_sunk_cost`` (total = depreciable + non-depreciable)
            * ``_gas_sunk_cost`` (total = depreciable + non-depreciable)
        - Internally relies on each cost object's
          ``expenditures_pre_tax()`` method to compute expenditures.
        """

        # Validate OIL and GAS sunkcost
        self._get_sunkcost_validation()

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

        # Define total sunk cost
        self._oil_sunk_cost = (
            self._oil_depreciable_sunk_cost + self._oil_non_depreciable_sunk_cost
        )

        self._gas_sunk_cost = (
            self._gas_depreciable_sunk_cost + self._gas_non_depreciable_sunk_cost
        )

    def _get_preonstream_array(self) -> None:
        """
        Validate and prepare pre-onstream cost arrays for OIL and GAS.

        This method validates all pre-onstream cost objects for OIL and GAS
        using `_get_preonstream_validation`. It then constructs arrays of
        pre-tax expenditures for each pre-onstream cost category, aggregates
        them into depreciable and non-depreciable components, and finally
        computes the total pre-onstream costs for both fluids.

        Raises
        ------
        PreonstreamCostException
            If any pre-onstream years fall outside the allowable range defined
            by the project approval year and the respective fluid's onstream year,
            as enforced by `_get_preonstream_validation`.

        Notes
        -----
        - Depreciable pre-onstream costs include only capital costs.
        - Non-depreciable pre-onstream costs are formed by summing intangible,
          OPEX, ASR, LBT, and cost of sales components.
        - The following internal attributes are set:
            * ``_oil_depreciable_preonstream``
            * ``_gas_depreciable_preonstream``
            * ``_oil_non_depreciable_preonstream``
            * ``_gas_non_depreciable_preonstream``
            * ``_oil_preonstream`` (total = depreciable + non-depreciable)
            * ``_gas_preonstream`` (total = depreciable + non-depreciable)
        - Internally relies on each cost object's
          ``expenditures_pre_tax()`` method to compute expenditures.
        """

        # Validate OIL and GAS preonstream
        self._get_preonstream_validation()

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

        # Define total preonstream costs
        self._oil_preonstream = (
            self._oil_depreciable_preonstream + self._oil_non_depreciable_preonstream
        )

        self._gas_preonstream = (
            self._gas_depreciable_preonstream + self._gas_non_depreciable_preonstream
        )

    def _calc_expenditures(
        self,
        target_attr: CapitalCost | Intangible | OPEX | ASR | LBT,
        expenditures_type: ExpendituresType,
        year_inflation: np.ndarray = None,
        inflation_rate: np.ndarray | int | float = 0.0,
        inflation_rate_applied_to: InflationAppliedTo | None = None,
        tax_rate: np.ndarray | float = 0.0,
    ) -> np.ndarray:
        """
        Calculate pre-tax or post-tax expenditures for the given target attribute, with
        optional inflation and tax adjustments.

        This method calculates expenditures (either pre-tax or post-tax) for a given
        target attribute, applying inflation rates and taxes based on the specified
        `expenditures_type` and `inflation_rate_applied_to` parameters.

        Parameters
        ----------
        target_attr : object
            The target attribute for which expenditures are being calculated
            (e.g., oil or gas costs).
        expenditures_type : ExpendituresType
            Specifies whether to calculate pre-tax or post-tax expenditures.
            Must be one of `ExpendituresType.PRE_TAX` or `ExpendituresType.POST_TAX`.
        year_inflation : np.ndarray, optional
            A NumPy array of inflation rates per year. Default is None.
        inflation_rate : np.ndarray or float, optional
            The inflation rate applied to the expenditures. Default is 0.0.
        inflation_rate_applied_to : InflationAppliedTo or None, optional
            Specifies whether inflation applies to CAPEX, OPEX, both, or none.
            Must be one of `InflationAppliedTo.CAPEX`, `InflationAppliedTo.OPEX`,
            `InflationAppliedTo.CAPEX_AND_OPEX`, or `None`. Default is None.
        tax_rate : np.ndarray or float, optional
            The tax rate applied for post-tax expenditures. Default is 0.0.

        Returns
        -------
        np.ndarray
            The calculated expenditures, either pre-tax or post-tax, depending on
            the `expenditures_type`.

        Notes
        -----
        -   The method supports applying inflation selectively to CAPEX, OPEX, or both for
            both pre-tax and post-tax expenditures.
        -   If inflation is applied only to CAPEX or OPEX, the other expenditure type remains
            unaffected by inflation.
        -   Post-tax expenditures include the impact of the specified tax rate.
        """

        # For pre-tax expenditures
        if expenditures_type == ExpendituresType.PRE_TAX:

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

        # For post-tax expenditures
        elif expenditures_type == ExpendituresType.POST_TAX:

            # No inflation rate
            if inflation_rate_applied_to is None:
                return target_attr.expenditures_post_tax(
                    year_inflation=year_inflation,
                    inflation_rate=0.0,
                    tax_rate=tax_rate,
                )

            # Inflation rate applied to CAPEX only
            elif inflation_rate_applied_to == InflationAppliedTo.CAPEX:
                if (
                    target_attr is self._oil_capital_postonstream
                    or target_attr is self._gas_capital_postonstream
                    or target_attr is self._oil_intangible_postonstream
                    or target_attr is self._gas_intangible_postonstream
                ):
                    return target_attr.expenditures_post_tax(
                        year_inflation=year_inflation,
                        inflation_rate=inflation_rate,
                        tax_rate=tax_rate,
                    )

                else:
                    return target_attr.expenditures_post_tax(
                        year_inflation=year_inflation,
                        inflation_rate=0.0,
                        tax_rate=tax_rate,
                    )

            # Inflation rate applied to OPEX only
            elif inflation_rate_applied_to == InflationAppliedTo.OPEX:
                if (
                    target_attr is self._oil_opex_postonstream
                    or target_attr is self._gas_opex_postonstream
                ):
                    return target_attr.expenditures_post_tax(
                        year_inflation=year_inflation,
                        inflation_rate=inflation_rate,
                        tax_rate=tax_rate,
                    )

                else:
                    return target_attr.expenditures_post_tax(
                        year_inflation=year_inflation,
                        inflation_rate=0.0,
                        tax_rate=tax_rate,
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
                    return target_attr.expenditures_post_tax(
                        year_inflation=year_inflation,
                        inflation_rate=inflation_rate,
                        tax_rate=tax_rate,
                    )

                else:
                    return target_attr.expenditures_post_tax(
                        year_inflation=year_inflation,
                        inflation_rate=0.0,
                        tax_rate=tax_rate,
                    )

            else:
                raise BaseProjectException(
                    f"Parameter inflation_rate_applied_to is not recognized. "
                    f"Available options are: CAPEX, OPEX, CAPEX and OPEX, or None. "
                )

        else:
            raise BaseProjectException(
                f"Parameter expenditures_type ({expenditures_type}) is not recognized. "
                f"Choose between ExpendituresType.PRE_TAX or ExpendituresType.POST_TAX. "
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
            self._calc_expenditures(
                target_attr=attr,
                expenditures_type=ExpendituresType.PRE_TAX,
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
        Calculate and assign post-tax expenditures for multiple oil and gas
        cost categories.

        This method updates each post-tax expenditure attribute by adding its
        corresponding pre-tax expenditure and indirect tax values for all cost
        categories (e.g., capital, intangible, operating, abandonment & site
        restoration (ASR), land and building tax (LBT), and cost of sales) for
        both oil and gas.

        Notes
        -----
        The method assumes that:
            -   Each cost category has corresponding ``*_pre_tax`` and
                ``*_indirect_tax`` attributes.
            -   Post-tax results will be stored in matching ``*_post_tax``
                attributes.
        """

        pre_tax = [
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
            self._oil_cost_of_sales_expenditures_pre_tax,
            self._gas_cost_of_sales_expenditures_pre_tax
        ]

        indirect_tax = [
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
            self._oil_cost_of_sales_indirect_tax,
            self._gas_cost_of_sales_indirect_tax,
        ]

        post_tax = [pt + it for pt, it in zip(pre_tax, indirect_tax)]

        (
            self._oil_capital_expenditures_post_tax,
            self._gas_capital_expenditures_post_tax,
            self._oil_intangible_expenditures_post_tax,
            self._gas_intangible_expenditures_post_tax,
            self._oil_opex_expenditures_post_tax,
            self._gas_opex_expenditures_post_tax,
            self._oil_asr_expenditures_post_tax,
            self._gas_asr_expenditures_post_tax,
            self._oil_lbt_expenditures_post_tax,
            self._gas_lbt_expenditures_post_tax,
            self._oil_cost_of_sales_expenditures_post_tax,
            self._gas_cost_of_sales_expenditures_post_tax,
        ) = post_tax

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
        Aggregates oil and gas profiles into consolidated project-wide profiles.

        This method sums corresponding oil and gas arrays to produce consolidated
        lifting, prices, revenues, sunk costs, expenditures (pre- and post-tax),
        indirect taxes, and cash flows. The results are stored in instance attributes
        for further use.

        Returns
        -------
        None
            Updates instance attributes with consolidated values.

        Notes
        -----
        - All oil and gas attributes must be defined and have matching shapes.
        - Consolidation is performed via element-wise addition.
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

        # Attributes associated with consolidated expenditures pre tax
        self._consolidated_capital_expenditures_pre_tax = (
            self._oil_capital_expenditures_pre_tax
            + self._gas_capital_expenditures_pre_tax
        )
        self._consolidated_intangible_expenditures_pre_tax = (
            self._oil_intangible_expenditures_pre_tax
            + self._gas_intangible_expenditures_pre_tax
        )
        self._consolidated_opex_expenditures_pre_tax = (
            self._oil_opex_expenditures_pre_tax + self._gas_opex_expenditures_pre_tax
        )
        self._consolidated_asr_expenditures_pre_tax = (
            self._oil_asr_expenditures_pre_tax + self._gas_asr_expenditures_pre_tax
        )
        self._consolidated_lbt_expenditures_pre_tax = (
            self._oil_lbt_expenditures_pre_tax + self._gas_lbt_expenditures_pre_tax
        )
        self._consolidated_cost_of_sales_expenditures_pre_tax = (
            self._oil_cost_of_sales_expenditures_pre_tax
            + self._gas_cost_of_sales_expenditures_pre_tax
        )
        self._consolidated_expenditures_pre_tax = (
            self._oil_total_expenditures_pre_tax + self._gas_total_expenditures_pre_tax
        )

        # Attributes associated with consolidated indirect tax
        self._consolidated_capital_indirect_tax = (
            self._oil_capital_indirect_tax + self._gas_capital_indirect_tax
        )
        self._consolidated_intangible_indirect_tax = (
            self._oil_intangible_indirect_tax + self._gas_intangible_indirect_tax
        )
        self._consolidated_opex_indirect_tax = (
            self._oil_opex_indirect_tax + self._gas_opex_indirect_tax
        )
        self._consolidated_asr_indirect_tax = (
            self._oil_asr_indirect_tax + self._gas_asr_indirect_tax
        )
        self._consolidated_lbt_indirect_tax = (
            self._oil_lbt_indirect_tax + self._gas_lbt_indirect_tax
        )
        self._consolidated_cost_of_sales_indirect_tax = (
            self._oil_cost_of_sales_indirect_tax + self._gas_cost_of_sales_indirect_tax
        )
        self._consolidated_indirect_tax = (
            self._oil_total_indirect_tax + self._gas_total_indirect_tax
        )

        # Attributes associated with consolidated expenditures post tax
        self._consolidated_capital_expenditures_post_tax = (
            self._oil_capital_expenditures_post_tax
            + self._gas_capital_expenditures_post_tax
        )
        self._consolidated_intangible_expenditures_post_tax = (
            self._oil_intangible_expenditures_post_tax
            + self._gas_intangible_expenditures_post_tax
        )
        self._consolidated_opex_expenditures_post_tax = (
            self._oil_opex_expenditures_post_tax + self._gas_opex_expenditures_post_tax
        )
        self._consolidated_asr_expenditures_post_tax = (
            self._oil_asr_expenditures_post_tax + self._gas_asr_expenditures_post_tax
        )
        self._consolidated_lbt_expenditures_post_tax = (
            self._oil_lbt_expenditures_post_tax + self._gas_lbt_expenditures_post_tax
        )
        self._consolidated_cost_of_sales_expenditures_post_tax = (
            self._oil_cost_of_sales_expenditures_post_tax
            + self._gas_cost_of_sales_expenditures_post_tax
        )
        self._consolidated_expenditures_post_tax = (
            self._oil_total_expenditures_post_tax + self._gas_total_expenditures_post_tax
        )

        # Attribute associated with consolidated cashflow
        self._consolidated_non_capital = self._oil_non_capital + self._gas_non_capital
        self._consolidated_cashflow = self._oil_cashflow + self._gas_cashflow

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

        # Prepare preonstream and sunk costs
        self._get_sunkcost_array()
        self._get_preonstream_array()

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

        # # Total OIL post-tax expenditures
        # self._oil_total_expenditures_post_tax = (
        #     self._oil_capital_expenditures_post_tax
        #     + self._oil_intangible_expenditures_post_tax
        #     + self._oil_opex_expenditures_post_tax
        #     + self._oil_asr_expenditures_post_tax
        #     + self._oil_lbt_expenditures_post_tax
        #     + self._oil_cost_of_sales_expenditures_post_tax
        # )
        #
        # # Total GAS post-tax expenditures
        # self._gas_total_expenditures_post_tax = (
        #     self._gas_capital_expenditures_post_tax
        #     + self._gas_intangible_expenditures_post_tax
        #     + self._gas_opex_expenditures_post_tax
        #     + self._gas_asr_expenditures_post_tax
        #     + self._gas_lbt_expenditures_post_tax
        #     + self._gas_cost_of_sales_expenditures_post_tax
        # )
        #
        # # Non-capital costs (intangible + opex + asr + lbt + cost of sales)
        # self._oil_non_capital = (
        #     self._oil_intangible_expenditures_post_tax
        #     + self._oil_opex_expenditures_post_tax
        #     + self._oil_asr_expenditures_post_tax
        #     + self._oil_lbt_expenditures_post_tax
        #     + self._oil_cost_of_sales_expenditures_post_tax
        # )
        #
        # self._gas_non_capital = (
        #     self._gas_intangible_expenditures_post_tax
        #     + self._gas_opex_expenditures_post_tax
        #     + self._gas_asr_expenditures_post_tax
        #     + self._gas_lbt_expenditures_post_tax
        #     + self._gas_cost_of_sales_expenditures_post_tax
        # )

        # # Configure base cashflow for OIL and GAS
        # self._oil_cashflow = (
        #     self._oil_revenue - (self._oil_sunk_cost + self._oil_total_expenditures_post_tax)
        # )
        #
        # self._gas_cashflow = (
        #     self._gas_revenue - (self._gas_sunk_cost + self._gas_total_expenditures_post_tax)
        # )
        #
        # # Prepare consolidated profiles
        # self._get_consolidated_profiles()

    def __len__(self):
        return self.project_duration
