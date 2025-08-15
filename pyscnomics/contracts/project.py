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

    start_date: date
    end_date: date
    oil_onstream_date: date = field(default=None)
    gas_onstream_date: date = field(default=None)
    lifting: tuple[Lifting, ...] = field(default=None)
    capital_cost: tuple[CapitalCost, ...] = field(default=None)
    intangible_cost: tuple[Intangible, ...] = field(default=None)
    opex: tuple[OPEX, ...] = field(default=None)
    asr_cost: tuple[ASR, ...] = field(default=None)
    lbt_cost: tuple[LBT, ...] = field(default=None)
    cost_of_sales: tuple[CostOfSales, ...] = field(default=None)

    # Attributes to be defined later (associated with project duration)
    project_duration: int = field(default=None, init=False)
    project_years: np.ndarray = field(default=None, init=False)

    # Attributes associated with cashflow table generation
    _last_run_params: dict = field(default=None, init=False, repr=False)
    _run_completed: bool = field(default=False, init=False, repr=False)

    # Attributes to be defined later (associated with lifting for each fluid types)
    _oil_lifting: Lifting = field(default=None, init=False, repr=False)
    _gas_lifting: Lifting = field(default=None, init=False, repr=False)
    _sulfur_lifting: Lifting = field(default=None, init=False, repr=False)
    _electricity_lifting: Lifting = field(default=None, init=False, repr=False)
    _co2_lifting: Lifting = field(default=None, init=False, repr=False)

    # Attributes to be defined later (associated with revenue for each fluid types)
    _oil_revenue: np.ndarray = field(default=None, init=False, repr=False)
    _gas_revenue: np.ndarray = field(default=None, init=False, repr=False)
    _sulfur_revenue: np.ndarray = field(default=None, init=False, repr=False)
    _electricity_revenue: np.ndarray = field(default=None, init=False, repr=False)
    _co2_revenue: np.ndarray = field(default=None, init=False, repr=False)

    # Atributes to be defined later (associated with WAP price)
    _oil_wap_price: np.ndarray = field(default=None, init=False, repr=False)
    _gas_wap_price: np.ndarray = field(default=None, init=False, repr=False)
    _sulfur_wap_price: np.ndarray = field(default=None, init=False, repr=False)
    _electricity_wap_price: np.ndarray = field(default=None, init=False, repr=False)
    _co2_wap_price: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes to be defined later (associated with total cost per component)
    capital_cost_total: CapitalCost = field(default=None, init=False, repr=False)
    intangible_cost_total: Intangible = field(default=None, init=False, repr=False)
    opex_total: OPEX = field(default=None, init=False, repr=False)
    asr_cost_total: ASR = field(default=None, init=False, repr=False)
    lbt_cost_total: LBT = field(default=None, init=False, repr=False)
    cost_of_sales_total: CostOfSales = field(default=None, init=False, repr=False)

    # Attributes to be defined later (associated with costs)
    _oil_capital_cost: CapitalCost = field(default=None, init=False, repr=False)
    _gas_capital_cost: CapitalCost = field(default=None, init=False, repr=False)
    _oil_intangible: Intangible = field(default=None, init=False, repr=False)
    _gas_intangible: Intangible = field(default=None, init=False, repr=False)
    _oil_opex: OPEX = field(default=None, init=False, repr=False)
    _gas_opex: OPEX = field(default=None, init=False, repr=False)
    _oil_asr: ASR = field(default=None, init=False, repr=False)
    _gas_asr: ASR = field(default=None, init=False, repr=False)
    _oil_lbt: LBT = field(default=None, init=False, repr=False)
    _gas_lbt: LBT = field(default=None, init=False, repr=False)
    _oil_cost_of_sales: CostOfSales = field(default=None, init=False, repr=False)
    _gas_cost_of_sales: CostOfSales = field(default=None, init=False, repr=False)

    # Attributes to be defined later (associated with sunkcost in each cost categories)
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

    # Attributes to be defined later (associated with sunk cost)
    _oil_depreciable_sunk_cost: np.ndarray = field(default=None, init=False, repr=False)
    _gas_depreciable_sunk_cost: np.ndarray = field(default=None, init=False, repr=False)
    _oil_non_depreciable_sunk_cost: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _gas_non_depreciable_sunk_cost: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _oil_sunk_cost: np.ndarray = field(default=None, init=False, repr=False)
    _gas_sunk_cost: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes to be defined later
    # (Associated with pre tax expenditures for each cost elements)
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

    # Attributes to be defined later
    # (Associated with indirect taxes for each cost element)
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

    # Attributes to be defined later
    # (Associated with post tax expenditures for each cost elements)
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

    # Attributes to be defined later
    # (Associated with total expenditures and indirect taxes for each fluid)
    _oil_total_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_total_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    _oil_total_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_total_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    _oil_total_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_total_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes to be defined later (associated with non capital costs)
    _oil_non_capital: np.ndarray = field(default=None, init=False, repr=False)
    _gas_non_capital: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes to be defined later (associated with cashflow)
    _oil_cashflow: np.ndarray = field(default=None, init=False, repr=False)
    _gas_cashflow: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes to be defined later (associated with consolidated lifting)
    _consolidated_lifting: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_wap_price: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_revenue: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes to be defined later (associated with consolidated sunk cost)
    _consolidated_depreciable_sunk_cost: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_non_depreciable_sunk_cost: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_sunk_cost: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes to be defined later (associated with consolidated expenditures pre tax)
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

    # Attributes to be defined later (associated with consolidated indirect tax)
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

    # Attributes to be defined later (associated with consolidated expenditures post tax)
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

    # Attributes to be defined later (associated with consolidated cashflow)
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
        -   Prepare attributes _oil_capital_cost and _gas_capital_cost;
        -   Prepare attributes _oil_intangible and _gas_intangible;
        -   Prepare attributes _oil_opex and _gas_opex;
        -   Prepare attributes _oil_asr and _gas_asr;
        -   Prepare attributes _oil_lbt and _gas_lbt;
        -   Prepare attributes _oil_cost_of_sales and _gas_cost_of_sales;
        -   Prepare attributes _oil_capital_sunk_cost and _gas_capital_sunk_cost;
        -   Prepare attributes _oil_intangible_sunk_cost and _gas_intangible_sunk_cost;
        -   Prepare attributes _oil_opex_sunk_cost and _gas_opex_sunk_cost;
        -   Prepare attributes _oil_asr_sunk_cost and _gas_asr_sunk_cost;
        -   Prepare attributes _oil_lbt_sunk_cost and _gas_lbt_sunk_cost;
        -   Prepare attribute _oil_cost_of_sales_sunk_cost;
        -   Prepare attribute _gas_cost_of_sales_sunk_cost;
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

        # Prepare attributes associated with costs
        self._oil_capital_cost = self._get_oil_capital()
        self._gas_capital_cost = self._get_gas_capital()
        self._oil_intangible = self._get_oil_intangible()
        self._gas_intangible = self._get_gas_intangible()
        self._oil_opex = self._get_oil_opex()
        self._gas_opex = self._get_gas_opex()
        self._oil_asr = self._get_oil_asr()
        self._gas_asr = self._get_gas_asr()
        self._oil_lbt = self._get_oil_lbt()
        self._gas_lbt = self._get_gas_lbt()
        self._oil_cost_of_sales = self._get_oil_cost_of_sales()
        self._gas_cost_of_sales = self._get_gas_cost_of_sales()

        # Prepare attributes associated with sunk costs
        self._oil_capital_sunk_cost = self._get_oil_capital_sunk_cost()
        self._gas_capital_sunk_cost = self._get_gas_capital_sunk_cost()
        self._oil_intangible_sunk_cost = self._get_oil_intangible_sunk_cost()
        self._gas_intangible_sunk_cost = self._get_gas_intangible_sunk_cost()
        self._oil_opex_sunk_cost = self._get_oil_opex_sunk_cost()
        self._gas_opex_sunk_cost = self._get_gas_opex_sunk_cost()
        self._oil_asr_sunk_cost = self._get_oil_asr_sunk_cost()
        self._gas_asr_sunk_cost = self._get_gas_asr_sunk_cost()
        self._oil_lbt_sunk_cost = self._get_oil_lbt_sunk_cost()
        self._gas_lbt_sunk_cost = self._get_gas_lbt_sunk_cost()
        self._oil_cost_of_sales_sunk_cost = self._get_oil_cost_of_sales_sunk_cost()
        self._gas_cost_of_sales_sunk_cost = self._get_gas_cost_of_sales_sunk_cost()

        # Raise an exception error if the start year of the project is inconsistent
        if not all(
            i == self.start_date.year
            for i in [
                self._oil_lifting.start_year,
                self._gas_lifting.start_year,
                self._sulfur_lifting.start_year,
                self._electricity_lifting.start_year,
                self._co2_lifting.start_year,
                self._oil_capital_cost.start_year,
                self._gas_capital_cost.start_year,
                self._oil_intangible.start_year,
                self._gas_intangible.start_year,
                self._oil_opex.start_year,
                self._gas_opex.start_year,
                self._oil_asr.start_year,
                self._gas_asr.start_year,
                self._oil_lbt.start_year,
                self._gas_lbt.start_year,
                self._oil_cost_of_sales.start_year,
                self._gas_cost_of_sales.start_year,
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
                f"Oil tangible ({self._oil_capital_cost.start_year}), "
                f"Gas tangible ({self._gas_capital_cost.start_year}), "
                f"Oil intangible ({self._oil_intangible.start_year}), "
                f"Gas intangible ({self._gas_intangible.start_year}), "
                f"Oil opex ({self._oil_opex.start_year}), "
                f"Gas opex ({self._gas_opex.start_year}), "
                f"Oil asr ({self._oil_asr.start_year}), "
                f"Gas asr ({self._gas_asr.start_year}), "
                f"Oil LBT ({self._oil_lbt.start_year}), "
                f"Gas LBT ({self._gas_lbt.start_year}), "
                f"Oil cost of sales ({self._oil_cost_of_sales.start_year}), "
                f"Gas cost of sales ({self._gas_cost_of_sales.start_year}), "
                f"Oil capital sunkcost ({self._oil_capital_sunk_cost.start_year}), "
                f"Gas capital sunkcost ({self._gas_capital_sunk_cost.start_year}), "
                f"Oil intangible sunkcost ({self._oil_intangible_sunk_cost.start_year}), "
                f"Gas intangible sunkcost ({self._gas_intangible_sunk_cost.start_year}), "
                f"Oil OPEX sunkcost ({self._oil_opex_sunk_cost.start_year}), "
                f"Gas OPEX sunkcost ({self._gas_opex_sunk_cost.start_year}), "
                f"Oil ASR sunkcost ({self._oil_asr_sunk_cost.start_year}), "
                f"Gas ASR sunkcost ({self._gas_asr_sunk_cost.start_year}), "
                f"Oil LBT sunkcost ({self._oil_lbt_sunk_cost.start_year}), "
                f"Gas LBT sunkcost ({self._gas_lbt_sunk_cost.start_year}), "
                f"Oil cost of sales sunkcost: ({self._oil_cost_of_sales_sunk_cost.start_year}), "
                f"Gas cost of sales sunkcost: ({self._gas_cost_of_sales_sunk_cost.start_year})."
            )

        # Raise an exception error if the end year of the project is inconsistent
        if not all(
            i == self.end_date.year
            for i in [
                self._oil_lifting.end_year,
                self._gas_lifting.end_year,
                self._sulfur_lifting.end_year,
                self._electricity_lifting.end_year,
                self._co2_lifting.end_year,
                self._oil_capital_cost.end_year,
                self._gas_capital_cost.end_year,
                self._oil_intangible.end_year,
                self._gas_intangible.end_year,
                self._oil_opex.end_year,
                self._gas_opex.end_year,
                self._oil_asr.end_year,
                self._gas_asr.end_year,
                self._oil_lbt.end_year,
                self._gas_lbt.end_year,
                self._oil_cost_of_sales.end_year,
                self._gas_cost_of_sales.end_year,
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
                f"Oil tangible ({self._oil_capital_cost.end_year}), "
                f"Gas tangible ({self._gas_capital_cost.end_year}), "
                f"Oil intangible ({self._oil_intangible.end_year}), "
                f"Gas intangible ({self._gas_intangible.end_year}), "
                f"Oil opex ({self._oil_opex.end_year}), "
                f"Gas opex ({self._gas_opex.end_year}), "
                f"Oil asr ({self._oil_asr.end_year}), "
                f"Gas asr ({self._gas_asr.end_year}), "
                f"Oil LBT ({self._oil_lbt.end_year}), "
                f"Gas LBT ({self._gas_lbt.end_year}), "
                f"Oil cost of sales ({self._oil_cost_of_sales.end_year}), "
                f"Gas cost of sales ({self._gas_cost_of_sales.end_year}), "
                f"Oil capital sunkcost ({self._oil_capital_sunk_cost.end_year}), "
                f"Gas capital sunkcost ({self._gas_capital_sunk_cost.end_year}), "
                f"Oil intangible sunkcost ({self._oil_intangible_sunk_cost.end_year}), "
                f"Gas intangible sunkcost ({self._gas_intangible_sunk_cost.end_year}), "
                f"Oil OPEX sunkcost ({self._oil_opex_sunk_cost.end_year}), "
                f"Gas OPEX sunkcost ({self._gas_opex_sunk_cost.end_year}), "
                f"Oil ASR sunkcost ({self._oil_asr_sunk_cost.end_year}), "
                f"Gas ASR sunkcost ({self._gas_asr_sunk_cost.end_year}), "
                f"Oil LBT sunkcost ({self._oil_lbt_sunk_cost.end_year}), "
                f"Gas LBT sunkcost ({self._gas_lbt_sunk_cost.end_year}), "
                f"Oil cost of sales sunkcost ({self._oil_cost_of_sales_sunk_cost.end_year}), "
                f"Gas cost of sales sunkcost ({self._gas_cost_of_sales_sunk_cost.end_year})."
            )

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

    def _get_filtered_capital_cost(
        self,
        fluid_type: FluidType,
        include_sunkcost: bool,
    ) -> CapitalCost:
        """
        Get capital costs filtered by fluid type and sunk cost inclusion.

        This method retrieves the portion of the project's total capital costs
        (`self.capital_cost_total`) that match both a specific fluid type
        (`fluid_type`) and sunk cost flag (`include_sunkcost`).

        Parameters
        ----------
        fluid_type : FluidType
            The fluid type to filter capital costs by (e.g., `FluidType.OIL`,
            `FluidType.GAS`).
        include_sunkcost : bool
            If True, include only sunk costs. If False, exclude sunk costs.

        Returns
        -------
        CapitalCost
            A new `CapitalCost` instance containing only the costs that match the
            given fluid type and sunk cost flag.

            If no matching costs are found, returns a `CapitalCost` instance with:
            - Zero cost
            - Single expense year (project start year)
            - Specified `fluid_type` in cost_allocation
            - Specified `include_sunkcost` in is_sunkcost

        Notes
        -----
        - Matching is performed using a logical AND on `cost_allocation` and
          `is_sunkcost` arrays.
        - When no matches exist, a dummy `CapitalCost` instance is returned
          with preserved metadata (start_year, end_year, fluid type, sunk cost flag).
        - All returned arrays (cost, expense_year, etc.) have consistent lengths.
        - Preserves all other original attributes (description, tax_portion,
          tax_discount, pis_year, salvage_value, useful_life, depreciation_factor,
          is_ic_applied) for the filtered indices.
        """

        cct = self.capital_cost_total

        allocation_array = np.array(cct.cost_allocation)
        sunkcost_array = np.array(cct.is_sunkcost)
        mask = np.logical_and(
            allocation_array == fluid_type,
            sunkcost_array == include_sunkcost,
        )
        indices = np.flatnonzero(mask)

        if len(indices) == 0:
            return CapitalCost(
                start_year=cct.start_year,
                end_year=cct.end_year,
                expense_year=np.array([cct.start_year]),
                cost=np.array([0]),
                cost_allocation=[fluid_type],
                is_sunkcost=[include_sunkcost],
            )

        return CapitalCost(
            start_year=cct.start_year,
            end_year=cct.end_year,
            expense_year=cct.expense_year[indices],
            cost=cct.cost[indices],
            cost_allocation=np.array(cct.cost_allocation)[indices].tolist(),
            description=np.array(cct.description)[indices].tolist(),
            is_sunkcost=np.array(cct.is_sunkcost)[indices].tolist(),
            tax_portion=cct.tax_portion[indices],
            tax_discount=cct.tax_discount[indices],
            pis_year=cct.pis_year[indices],
            salvage_value=cct.salvage_value[indices],
            useful_life=cct.useful_life[indices],
            depreciation_factor=cct.depreciation_factor[indices],
            is_ic_applied=np.array(cct.is_ic_applied)[indices].tolist(),
        )

    def _get_oil_capital(self) -> CapitalCost:
        """
         Get oil-related capital costs excluding sunk costs.

        Returns
        -------
        CapitalCost
            A CapitalCost instance containing only the capital costs that:
            1. Are allocated to oil (`FluidType.OIL`)
            2. Are not classified as sunk costs (`include_sunkcost=False`)

            If no matching costs are found, returns a CapitalCost instance with:
            - Zero cost
            - Single expense year (project start year)
            - `FluidType.OIL` in cost_allocation
            - `False` in is_sunkcost

        Notes
        -----
        - This method is a specialized call to `_get_filtered_capital_cost`
          with `fluid_type=FluidType.OIL` and `include_sunkcost=False`.
        """
        return self._get_filtered_capital_cost(
            fluid_type=FluidType.OIL, include_sunkcost=False
        )

    def _get_gas_capital(self) -> CapitalCost:
        """
        Get gas-related capital costs excluding sunk costs.

        Returns
        -------
        CapitalCost
            A CapitalCost instance containing only the capital costs that:
            1. Are allocated to gas (`FluidType.GAS`)
            2. Are not classified as sunk costs (`include_sunkcost=False`)

            If no matching costs are found, returns a CapitalCost instance with:
            - Zero cost
            - Single expense year (project start year)
            - `FluidType.GAS` in cost_allocation
            - `False` in is_sunkcost

        Notes
        -----
        - This method is a specialized call to `_get_filtered_capital_cost`
          with `fluid_type=FluidType.GAS` and `include_sunkcost=False`.
        """
        return self._get_filtered_capital_cost(
            fluid_type=FluidType.GAS, include_sunkcost=False
        )

    def _get_oil_capital_sunk_cost(self) -> CapitalCost:
        """
        Get capital costs for oil including sunk costs.

        This method retrieves the portion of the project's total capital costs
        (`self.capital_cost_total`) that are allocated to oil (`FluidType.OIL`)
        and are considered sunk costs (`include_sunkcost=True`).

        Returns
        -------
        CapitalCost
            A new `CapitalCost` instance containing only the costs that match:
            1. `FluidType.OIL` in cost_allocation
            2. `is_sunkcost=True`

            If no matching costs are found, returns a `CapitalCost` instance with:
            - Zero cost
            - Single expense year (project start year)
            - `FluidType.OIL` in cost_allocation
            - `is_sunkcost=True`

        Notes
        -----
        - Internally calls `_get_filtered_capital_cost()` with
          `fluid_type=FluidType.OIL` and `include_sunkcost=True`.
        """
        return self._get_filtered_capital_cost(
            fluid_type=FluidType.OIL, include_sunkcost=True
        )

    def _get_gas_capital_sunk_cost(self) -> CapitalCost:
        """
        Get capital costs for gas including sunk costs.

        This method retrieves the portion of the project's total capital costs
        (`self.capital_cost_total`) that are allocated to gas (`FluidType.GAS`)
        and are considered sunk costs (`include_sunkcost=True`).

        Returns
        -------
        CapitalCost
            A new `CapitalCost` instance containing only the costs that match:
            1. `FluidType.GAS` in cost_allocation
            2. `is_sunkcost=True`

            If no matching costs are found, returns a `CapitalCost` instance with:
            - Zero cost
            - Single expense year (project start year)
            - `FluidType.GAS` in cost_allocation
            - `is_sunkcost=True`

        Notes
        -----
        - Internally calls `_get_filtered_capital_cost()` with
          `fluid_type=FluidType.GAS` and `include_sunkcost=True`.
        """
        return self._get_filtered_capital_cost(
            fluid_type=FluidType.GAS, include_sunkcost=True
        )

    def _get_filtered_intangible(
        self,
        fluid_type: FluidType,
        include_sunkcost: bool,
    ) -> Intangible:
        """
        Get intangible costs filtered by fluid type and sunk cost inclusion.

        This method retrieves the portion of the project's total intangible costs
        (`self.intangible_cost_total`) that match both a specific fluid type
        (`fluid_type`) and sunk cost flag (`include_sunkcost`).

        Parameters
        ----------
        fluid_type : FluidType
            The fluid type to filter intangible costs by (e.g., `FluidType.OIL`,
            `FluidType.GAS`).
        include_sunkcost : bool
            If True, include only sunk costs. If False, exclude sunk costs.

        Returns
        -------
        Intangible
            A new `Intangible` instance containing only the costs that match the
            given fluid type and sunk cost flag.

            If no matching costs are found, returns an `Intangible` instance with:
            - Zero cost
            - Single expense year (project start year)
            - Specified `fluid_type` in cost_allocation
            - Specified `include_sunkcost` in is_sunkcost

        Notes
        -----
        - Matching is performed using a logical AND on `cost_allocation` and
          `is_sunkcost` arrays.
        - When no matches exist, a dummy `Intangible` instance is returned
          with preserved metadata (start_year, end_year, fluid type, sunk cost flag).
        - All returned arrays (cost, expense_year, etc.) have consistent lengths.
        """

        ict = self.intangible_cost_total

        allocation_array = np.array(ict.cost_allocation)
        sunkcost_array = np.array(ict.is_sunkcost)
        mask = np.logical_and(
            allocation_array == fluid_type,
            sunkcost_array == include_sunkcost,
        )
        indices = np.flatnonzero(mask)

        if len(indices) == 0:
            return Intangible(
                start_year=ict.start_year,
                end_year=ict.end_year,
                expense_year=np.array([ict.start_year]),
                cost=np.array([0]),
                cost_allocation=[fluid_type],
                is_sunkcost=[include_sunkcost],
            )

        return Intangible(
            start_year=ict.start_year,
            end_year=ict.end_year,
            expense_year=ict.expense_year[indices],
            cost=ict.cost[indices],
            cost_allocation=np.array(ict.cost_allocation)[indices].tolist(),
            description=np.array(ict.description)[indices].tolist(),
            is_sunkcost=np.array(ict.is_sunkcost)[indices].tolist(),
            tax_portion=ict.tax_portion[indices],
            tax_discount=ict.tax_discount[indices],
        )

    def _get_oil_intangible(self) -> Intangible:
        """
        Get intangible costs for oil excluding sunk costs.

        This method retrieves the portion of the project's total intangible costs
        (`self.intangible_cost_total`) that are allocated to oil (`FluidType.OIL`)
        and are not considered sunk costs (`include_sunkcost=False`).

        Returns
        -------
        Intangible
            A new `Intangible` instance containing only the costs that match:
            1. `FluidType.OIL` in cost_allocation
            2. `is_sunkcost=False`

            If no matching costs are found, returns an `Intangible` instance with:
            - Zero cost
            - Single expense year (project start year)
            - `FluidType.OIL` in cost_allocation
            - `is_sunkcost=False`

        Notes
        -----
        - Internally calls `_get_filtered_intangible()` with
          `fluid_type=FluidType.OIL` and `include_sunkcost=False`.
        - All returned arrays (cost, expense_year, etc.) have consistent lengths.
        - When no matching costs exist, a dummy instance is returned
          preserving the specified fluid type and sunk cost flag.
        """
        return self._get_filtered_intangible(fluid_type=FluidType.OIL, include_sunkcost=False)

    def _get_gas_intangible(self) -> Intangible:
        """
        Get intangible costs for gas excluding sunk costs.

        This method retrieves the portion of the project's total intangible costs
        (`self.intangible_cost_total`) that are allocated to gas (`FluidType.GAS`)
        and are not considered sunk costs (`include_sunkcost=False`).

        Returns
        -------
        Intangible
            A new `Intangible` instance containing only the costs that match:
            1. `FluidType.GAS` in cost_allocation
            2. `is_sunkcost=False`

            If no matching costs are found, returns an `Intangible` instance with:
            - Zero cost
            - Single expense year (project start year)
            - `FluidType.GAS` in cost_allocation
            - `is_sunkcost=False`

        Notes
        -----
        - Internally calls `_get_filtered_intangible()` with
          `fluid_type=FluidType.GAS` and `include_sunkcost=False`.
        - All returned arrays (cost, expense_year, etc.) have consistent lengths.
        - When no matching costs exist, a dummy instance is returned
          preserving the specified fluid type and sunk cost flag.
        """
        return self._get_filtered_intangible(fluid_type=FluidType.GAS, include_sunkcost=False)

    def _get_oil_intangible_sunk_cost(self) -> Intangible:
        """
        Get intangible costs for oil that are classified as sunk costs.

        This method retrieves the portion of the project's total intangible costs
        (`self.intangible_cost_total`) that are allocated to oil (`FluidType.OIL`)
        and are considered sunk costs (`include_sunkcost=True`).

        Returns
        -------
        Intangible
            A new `Intangible` instance containing only the costs that match:
            1. `FluidType.OIL` in cost_allocation
            2. `is_sunkcost=True`

            If no matching costs are found, returns an `Intangible` instance with:
            - Zero cost
            - Single expense year (project start year)
            - `FluidType.OIL` in cost_allocation
            - `is_sunkcost=True`

        Notes
        -----
        - Internally calls `_get_filtered_intangible()` with
          `fluid_type=FluidType.OIL` and `include_sunkcost=True`.
        - All returned arrays (cost, expense_year, etc.) have consistent lengths.
        - When no matching costs exist, a dummy instance is returned
          preserving the specified fluid type and sunk cost flag.
        """
        return self._get_filtered_intangible(fluid_type=FluidType.OIL, include_sunkcost=True)

    def _get_gas_intangible_sunk_cost(self) -> Intangible:
        """
        Get intangible costs for gas that are classified as sunk costs.

        This method retrieves the portion of the project's total intangible costs
        (`self.intangible_cost_total`) that are allocated to gas (`FluidType.GAS`)
        and are considered sunk costs (`include_sunkcost=True`).

        Returns
        -------
        Intangible
            A new `Intangible` instance containing only the costs that match:
            1. `FluidType.GAS` in cost_allocation
            2. `is_sunkcost=True`

            If no matching costs are found, returns an `Intangible` instance with:
            - Zero cost
            - Single expense year (project start year)
            - `FluidType.GAS` in cost_allocation
            - `is_sunkcost=True`

        Notes
        -----
        - Internally calls `_get_filtered_intangible()` with
          `fluid_type=FluidType.GAS` and `include_sunkcost=True`.
        - All returned arrays (cost, expense_year, etc.) have consistent lengths.
        - When no matching costs exist, a dummy instance is returned
          preserving the specified fluid type and sunk cost flag.
        """
        return self._get_filtered_intangible(fluid_type=FluidType.GAS, include_sunkcost=True)

    def _get_filtered_opex(
        self,
        fluid_type: FluidType,
        include_sunkcost: bool,
    ) -> OPEX:
        """
        Get operating expenses (OPEX) filtered by fluid type and sunk cost inclusion.

        This method retrieves the portion of the project's total operating expenses
        (`self.opex_total`) that match the specified fluid type (`fluid_type`)
        and sunk cost inclusion (`include_sunkcost`).

        Parameters
        ----------
        fluid_type : FluidType
            The fluid type to filter OPEX by (e.g., `FluidType.OIL` or `FluidType.GAS`).
        include_sunkcost : bool
            Whether to include sunk costs in the filtered results.
            If True, includes sunk costs; if False, excludes them.

        Returns
        -------
        OPEX
            A new `OPEX` instance containing only the costs that match:
            1. The specified fluid type in `cost_allocation`
            2. The specified sunk cost inclusion in `is_sunkcost`

            If no matching costs are found, returns an `OPEX` instance with:
            - Zero cost
            - Single expense year (project start year)
            - The requested fluid type in `cost_allocation`
            - The requested sunk cost flag in `is_sunkcost`

        Notes
        -----
        -   All returned arrays (fixed_cost, prod_rate, cost_per_volume, etc.) have
            consistent lengths.
        -   When no matching costs exist, a dummy instance is returned preserving
            the specified fluid type and sunk cost flag.
        -   This method is used as the base for fluid-specific OPEX retrieval
            (e.g., `_get_oil_opex()`, `_get_gas_opex()`, etc.).
        """

        ot = self.opex_total

        allocation_array = np.array(ot.cost_allocation)
        sunkcost_array = np.array(ot.is_sunkcost)
        mask = np.logical_and(
            allocation_array == fluid_type,
            sunkcost_array == include_sunkcost,
        )
        indices = np.flatnonzero(mask)

        if len(indices) == 0:
            return OPEX(
                start_year=ot.start_year,
                end_year=ot.end_year,
                expense_year=np.array([ot.start_year]),
                fixed_cost=np.array([0]),
                cost_allocation=[fluid_type],
                is_sunkcost=[include_sunkcost],
            )

        return OPEX(
            start_year=ot.start_year,
            end_year=ot.end_year,
            expense_year=ot.expense_year[indices],
            cost_allocation=np.array(ot.cost_allocation)[indices].tolist(),
            description=np.array(ot.description)[indices].tolist(),
            is_sunkcost=np.array(ot.is_sunkcost)[indices].tolist(),
            tax_portion=ot.tax_portion[indices],
            tax_discount=ot.tax_discount[indices],
            fixed_cost=ot.fixed_cost[indices],
            prod_rate=ot.prod_rate[indices],
            cost_per_volume=ot.cost_per_volume[indices],
        )

    def _get_oil_opex(self) -> OPEX:
        """
        Get operating expenses (OPEX) for oil excluding sunk costs.

        This method retrieves the portion of the project's total OPEX (`self.opex_total`)
        that is allocated to oil (`FluidType.OIL`) and is not classified as sunk cost
        (`include_sunkcost=False`).

        Returns
        -------
        OPEX
            A new `OPEX` instance containing only the oil OPEX excluding sunk costs.

            If no matching costs are found, returns a dummy `OPEX` instance with:
            - Zero cost
            - Single expense year (project start year)
            - `FluidType.OIL` in `cost_allocation`
            - `is_sunkcost=False`

        Notes
        -----
        -   Internally calls `_get_filtered_opex()` with `fluid_type=FluidType.OIL`
            and `include_sunkcost=False`.
        -   Returned arrays (fixed_cost, prod_rate, cost_per_volume, etc.) have
            consistent lengths.
        """
        return self._get_filtered_opex(fluid_type=FluidType.OIL, include_sunkcost=False)

    def _get_gas_opex(self) -> OPEX:
        """
        Get operating expenses (OPEX) for gas excluding sunk costs.

        This method retrieves the portion of the project's total OPEX (`self.opex_total`)
        that is allocated to gas (`FluidType.GAS`) and is not classified as sunk cost
        (`include_sunkcost=False`).

        Returns
        -------
        OPEX
            A new `OPEX` instance containing only the gas OPEX excluding sunk costs.

            If no matching costs are found, returns a dummy `OPEX` instance with:
            - Zero cost
            - Single expense year (project start year)
            - `FluidType.GAS` in `cost_allocation`
            - `is_sunkcost=False`

        Notes
        -----
        -   Internally calls `_get_filtered_opex()` with `fluid_type=FluidType.GAS`
            and `include_sunkcost=False`.
        -   Returned arrays (fixed_cost, prod_rate, cost_per_volume, etc.) have
            consistent lengths.
        """
        return self._get_filtered_opex(fluid_type=FluidType.GAS, include_sunkcost=False)

    def _get_oil_opex_sunk_cost(self) -> OPEX:
        """
        Get operating expenses (OPEX) for oil that are classified as sunk costs.

        This method retrieves the portion of the project's total OPEX (`self.opex_total`)
        that is allocated to oil (`FluidType.OIL`) and is considered sunk cost
        (`include_sunkcost=True`).

        Returns
        -------
        OPEX
            A new `OPEX` instance containing only the oil OPEX classified as sunk costs.

            If no matching costs are found, returns a dummy `OPEX` instance with:
            - Zero cost
            - Single expense year (project start year)
            - `FluidType.OIL` in `cost_allocation`
            - `is_sunkcost=True`

        Notes
        -----
        -   Internally calls `_get_filtered_opex()` with `fluid_type=FluidType.OIL`
            and `include_sunkcost=True`.
        -   Returned arrays (fixed_cost, prod_rate, cost_per_volume, etc.) have
            consistent lengths.
        """
        return self._get_filtered_opex(fluid_type=FluidType.OIL, include_sunkcost=True)

    def _get_gas_opex_sunk_cost(self) -> OPEX:
        """
        Get operating expenses (OPEX) for gas that are classified as sunk costs.

        This method retrieves the portion of the project's total OPEX (`self.opex_total`)
        that is allocated to gas (`FluidType.GAS`) and is considered sunk cost
        (`include_sunkcost=True`).

        Returns
        -------
        OPEX
            A new `OPEX` instance containing only the gas OPEX classified as sunk costs.

            If no matching costs are found, returns a dummy `OPEX` instance with:
            - Zero cost
            - Single expense year (project start year)
            - `FluidType.GAS` in `cost_allocation`
            - `is_sunkcost=True`

        Notes
        -----
        -   Internally calls `_get_filtered_opex()` with `fluid_type=FluidType.GAS`
            and `include_sunkcost=True`.
        -   Returned arrays (fixed_cost, prod_rate, cost_per_volume, etc.) have
            consistent lengths.
        """
        return self._get_filtered_opex(fluid_type=FluidType.GAS, include_sunkcost=True)

    def _get_filtered_asr(
        self,
        fluid_type: FluidType,
        include_sunkcost: bool,
    ) -> ASR:
        """
        Get Abandonment Site and Restoration (ASR) costs filtered by fluid type
        and sunk cost inclusion.

        This method retrieves the portion of the project's total ASR costs
        (`self.asr_cost_total`) that matches the specified fluid type (`fluid_type`)
        and sunk cost classification (`include_sunkcost`).

        Parameters
        ----------
        fluid_type : FluidType
            The fluid type to filter ASR costs by (e.g., OIL or GAS).
        include_sunkcost : bool
            Whether to include sunk costs in the filtered results.
            If True, includes sunk costs; if False, excludes them.

        Returns
        -------
        ASR
            A new `ASR` instance containing only the costs that match the criteria.

            If no matching costs are found, returns a dummy `ASR` instance with:
            - Zero cost
            - Single expense year (project start year)
            - The requested `fluid_type` in `cost_allocation`
            - The requested `include_sunkcost` in `is_sunkcost`

        Notes
        -----
        - The method filters the data from `self.asr_cost_total` based on the criteria.
        - All returned arrays (cost, expense_year, etc.) will have consistent lengths.
        - When no matching costs exist, the returned dummy instance will include:
          * The requested `fluid_type` in `cost_allocation`
          * The requested `include_sunkcost` value in `is_sunkcost`
        - Other ASR-specific attributes (`final_year`, `future_rate`) are preserved for
          the matching entries.
        """

        act = self.asr_cost_total

        allocation_array = np.array(act.cost_allocation)
        sunkcost_array = np.array(act.is_sunkcost)
        mask = np.logical_and(
            allocation_array == fluid_type,
            sunkcost_array == include_sunkcost,
        )
        indices = np.flatnonzero(mask)

        if len(indices) == 0:
            return ASR(
                start_year=act.start_year,
                end_year=act.end_year,
                expense_year=np.array([act.start_year]),
                cost=np.array([0]),
                cost_allocation=[fluid_type],
                is_sunkcost=[include_sunkcost],
            )

        return ASR(
            start_year=act.start_year,
            end_year=act.end_year,
            expense_year=act.expense_year[indices],
            cost=act.cost[indices],
            cost_allocation=np.array(act.cost_allocation)[indices].tolist(),
            description=np.array(act.description)[indices].tolist(),
            is_sunkcost=np.array(act.is_sunkcost)[indices].tolist(),
            tax_portion=act.tax_portion[indices],
            tax_discount=act.tax_discount[indices],
            final_year=act.final_year[indices],
            future_rate=act.future_rate[indices],
        )

    def _get_oil_asr(self) -> ASR:
        """
        Get ASR for oil that are not classified as sunk costs.

        This method retrieves the portion of the project's total ASR
        (`self.asr_cost_total`) that is allocated to oil (`FluidType.OIL`)
        and is not considered sunk cost (`include_sunkcost=False`).

        Returns
        -------
        ASR
            A new `ASR` instance containing only the oil ASR excluding sunk costs.

            If no matching costs are found, returns a dummy `ASR` instance with:
            - Zero cost
            - Single expense year (project start year)
            - `FluidType.OIL` in `cost_allocation`
            - `is_sunkcost=False`

        Notes
        -----
        -   Internally calls `_get_filtered_asr()` with `fluid_type=FluidType.OIL`
            and `include_sunkcost=False`.
        -   Returned arrays (cost, expense_year, final_year, future_rate, etc.) have
            consistent lengths.
        """
        return self._get_filtered_asr(fluid_type=FluidType.OIL, include_sunkcost=False)

    def _get_gas_asr(self) -> ASR:
        """
        Get ASR for gas that are not classified as sunk costs.

        This method retrieves the portion of the project's total ASR (`self.asr_cost_total`)
        that is allocated to gas (`FluidType.GAS`) and is not considered sunk cost
        (`include_sunkcost=False`).

        Returns
        -------
        ASR
            A new `ASR` instance containing only the gas ASR excluding sunk costs.

            If no matching costs are found, returns a dummy `ASR` instance with:
            - Zero cost
            - Single expense year (project start year)
            - `FluidType.GAS` in `cost_allocation`
            - `is_sunkcost=False`

        Notes
        -----
        -   Internally calls `_get_filtered_asr()` with `fluid_type=FluidType.GAS`
            and `include_sunkcost=False`.
        -   Returned arrays (cost, expense_year, final_year, future_rate, etc.) have
            consistent lengths.
        """
        return self._get_filtered_asr(fluid_type=FluidType.GAS, include_sunkcost=False)

    def _get_oil_asr_sunk_cost(self) -> ASR:
        """
        Get ASR for oil that are classified as sunk costs.

        This method retrieves the portion of the project's total ASR (`self.asr_cost_total`)
        that is allocated to oil (`FluidType.OIL`) and is considered sunk cost
        (`include_sunkcost=True`).

        Returns
        -------
        ASR
            A new `ASR` instance containing only the oil ASR classified as sunk costs.

            If no matching costs are found, returns a dummy `ASR` instance with:
            - Zero cost
            - Single expense year (project start year)
            - `FluidType.OIL` in `cost_allocation`
            - `is_sunkcost=True`

        Notes
        -----
        -   Internally calls `_get_filtered_asr()` with `fluid_type=FluidType.OIL`
            and `include_sunkcost=True`.
        -   Returned arrays (cost, expense_year, final_year, future_rate, etc.) have
            consistent lengths.
        """
        return self._get_filtered_asr(fluid_type=FluidType.OIL, include_sunkcost=True)

    def _get_gas_asr_sunk_cost(self) -> ASR:
        """
        Get ASR for gas that are classified as sunk costs.

        This method retrieves the portion of the project's total ASR (`self.asr_cost_total`)
        that is allocated to gas (`FluidType.GAS`) and is considered sunk cost
        (`include_sunkcost=True`).

        Returns
        -------
        ASR
            A new `ASR` instance containing only the gas ASR classified as sunk costs.

            If no matching costs are found, returns a dummy `ASR` instance with:
            - Zero cost
            - Single expense year (project start year)
            - `FluidType.GAS` in `cost_allocation`
            - `is_sunkcost=True`

        Notes
        -----
        -   Internally calls `_get_filtered_asr()` with `fluid_type=FluidType.GAS`
            and `include_sunkcost=True`.
        -   Returned arrays (cost, expense_year, final_year, future_rate, etc.) have
            consistent lengths.
        """
        return self._get_filtered_asr(fluid_type=FluidType.GAS, include_sunkcost=True)

    def _get_filtered_lbt(
        self,
        fluid_type: FluidType,
        include_sunkcost: bool,
    ) -> LBT:
        """
        Get Land and Building Tax (LBT) costs filtered by fluid type and sunk cost inclusion.

        This method retrieves the portion of the project's total LBT costs (`self.lbt_cost_total`)
        that is allocated to the specified fluid type (`fluid_type`) and matches the sunk cost
        inclusion flag (`include_sunkcost`).

        Parameters
        ----------
        fluid_type : FluidType
            The fluid type (`FluidType.OIL`, `FluidType.GAS`, etc.) to filter LBT costs by.
        include_sunkcost : bool
            Whether to include only sunk cost items (`True`) or exclude them (`False`).

        Returns
        -------
        LBT
            A new `LBT` instance containing only the filtered LBT costs.

            If no matching costs are found, returns a dummy `LBT` instance with:
            - Zero cost
            - Single expense year (project start year)
            - `fluid_type` in `cost_allocation`
            - `is_sunkcost` matching the provided flag

        Notes
        -----
        - Internally applies a boolean mask over `cost_allocation` and `is_sunkcost`
          fields of `self.lbt_cost_total`.
        - Returned arrays (e.g., `cost`, `tax_portion`, `utilized_land_area`, etc.)
          have consistent lengths and are subset versions of the original arrays.
        """

        lct = self.lbt_cost_total

        allocation_array = np.array(lct.cost_allocation)
        sunkcost_array = np.array(lct.is_sunkcost)
        mask = np.logical_and(
            allocation_array == fluid_type,
            sunkcost_array == include_sunkcost,
        )
        indices = np.flatnonzero(mask)

        if len(indices) == 0:
            return LBT(
                start_year=lct.start_year,
                end_year=lct.end_year,
                expense_year=np.array([lct.start_year]),
                cost=np.array([0]),
                cost_allocation=[fluid_type],
                is_sunkcost=[include_sunkcost],
            )

        return LBT(
            start_year=lct.start_year,
            end_year=lct.end_year,
            expense_year=lct.expense_year[indices],
            cost_allocation=np.array(lct.cost_allocation)[indices].tolist(),
            description=np.array(lct.description)[indices].tolist(),
            is_sunkcost=np.array(lct.is_sunkcost)[indices].tolist(),
            tax_portion=lct.tax_portion[indices],
            tax_discount=lct.tax_discount[indices],
            final_year=lct.final_year[indices],
            utilized_land_area=lct.utilized_land_area[indices],
            utilized_building_area=lct.utilized_building_area[indices],
            njop_land=lct.njop_land[indices],
            njop_building=lct.njop_building[indices],
            gross_revenue=lct.gross_revenue[indices],
            cost=lct.cost[indices],
        )

    def _get_oil_lbt(self) -> LBT:
        """
        Get Land and Building Tax (LBT) for oil, excluding sunk costs.

        This method retrieves the portion of the project's total LBT (`self.lbt_cost_total`)
        that is allocated to oil (`FluidType.OIL`) and excludes sunk costs
        (`include_sunkcost=False`).

        Returns
        -------
        LBT
            A new `LBT` instance containing only the oil LBT excluding sunk costs.

            If no matching costs are found, returns a dummy `LBT` instance with:
            - Zero cost
            - Single expense year (project start year)
            - `FluidType.OIL` in `cost_allocation`
            - `is_sunkcost=False`

        Notes
        -----
        -   Internally calls `_get_filtered_lbt()` with `fluid_type=FluidType.OIL`
            and `include_sunkcost=False`.
        -   Returned arrays (tax_portion, tax_discount, final_year, utilized_land_area, etc.)
            have consistent lengths.
        """
        return self._get_filtered_lbt(fluid_type=FluidType.OIL, include_sunkcost=False)

    def _get_gas_lbt(self) -> LBT:
        """
        Get Land and Building Tax (LBT) for gas, excluding sunk costs.

        This method retrieves the portion of the project's total LBT (`self.lbt_cost_total`)
        that is allocated to gas (`FluidType.GAS`) and excludes sunk costs
        (`include_sunkcost=False`).

        Returns
        -------
        LBT
            A new `LBT` instance containing only the gas LBT excluding sunk costs.

            If no matching costs are found, returns a dummy `LBT` instance with:
            - Zero cost
            - Single expense year (project start year)
            - `FluidType.GAS` in `cost_allocation`
            - `is_sunkcost=False`

        Notes
        -----
        -   Internally calls `_get_filtered_lbt()` with `fluid_type=FluidType.GAS`
            and `include_sunkcost=False`.
        -   Returned arrays (tax_portion, tax_discount, final_year, utilized_land_area, etc.)
            have consistent lengths.
        """
        return self._get_filtered_lbt(fluid_type=FluidType.GAS, include_sunkcost=False)

    def _get_oil_lbt_sunk_cost(self) -> LBT:
        """
        Get Land and Building Tax (LBT) for oil that is classified as sunk costs.

        This method retrieves the portion of the project's total LBT (`self.lbt_cost_total`)
        that is allocated to oil (`FluidType.OIL`) and is considered sunk cost
        (`include_sunkcost=True`).

        Returns
        -------
        LBT
            A new `LBT` instance containing only the oil LBT classified as sunk costs.

            If no matching costs are found, returns a dummy `LBT` instance with:
            - Zero cost
            - Single expense year (project start year)
            - `FluidType.OIL` in `cost_allocation`
            - `is_sunkcost=True`

        Notes
        -----
        -   Internally calls `_get_filtered_lbt()` with `fluid_type=FluidType.OIL`
            and `include_sunkcost=True`.
        -   Returned arrays (tax_portion, tax_discount, final_year, utilized_land_area, etc.)
            have consistent lengths.
        """
        return self._get_filtered_lbt(fluid_type=FluidType.OIL, include_sunkcost=True)

    def _get_gas_lbt_sunk_cost(self) -> LBT:
        """
        Get Land and Building Tax (LBT) for gas that is classified as sunk costs.

        This method retrieves the portion of the project's total LBT (`self.lbt_cost_total`)
        that is allocated to gas (`FluidType.GAS`) and is considered sunk cost
        (`include_sunkcost=True`).

        Returns
        -------
        LBT
            A new `LBT` instance containing only the gas LBT classified as sunk costs.

            If no matching costs are found, returns a dummy `LBT` instance with:
            - Zero cost
            - Single expense year (project start year)
            - `FluidType.GAS` in `cost_allocation`
            - `is_sunkcost=True`

        Notes
        -----
        -   Internally calls `_get_filtered_lbt()` with `fluid_type=FluidType.GAS`
            and `include_sunkcost=True`.
        -   Returned arrays (tax_portion, tax_discount, final_year, utilized_land_area, etc.)
            have consistent lengths.
        """
        return self._get_filtered_lbt(fluid_type=FluidType.GAS, include_sunkcost=True)

    def _get_filtered_cost_of_sales(
        self,
        fluid_type: FluidType,
        include_sunkcost: bool,
    ) -> CostOfSales:
        """
        Get cost of sales filtered by fluid type and sunk cost inclusion.

        This method retrieves the portion of the project's total cost of sales
        (`self.cost_of_sales_total`) that is allocated to the specified fluid type
        (`fluid_type`) and matches the sunk cost inclusion flag (`include_sunkcost`).

        Parameters
        ----------
        fluid_type : FluidType
            The fluid type (`FluidType.OIL`, `FluidType.GAS`, etc.) to filter cost of sales by.
        include_sunkcost : bool
            Whether to include only sunk cost items (`True`) or exclude them (`False`).

        Returns
        -------
        CostOfSales
            A new `CostOfSales` instance containing only the filtered cost of sales.

            If no matching costs are found, returns a dummy `CostOfSales` instance with:
            - Zero cost
            - Single expense year (project start year)
            - `fluid_type` in `cost_allocation`
            - `is_sunkcost` matching the provided flag

        Notes
        -----
        - Internally applies a boolean mask over `cost_allocation` and `is_sunkcost`
          fields of `self.cost_of_sales_total`.
        - Returned arrays (e.g., `cost`, `tax_portion`, `tax_discount`, etc.)
          have consistent lengths and are subset versions of the original arrays.
        """

        cst = self.cost_of_sales_total

        allocation_array = np.array(cst.cost_allocation)
        sunkcost_array = np.array(cst.is_sunkcost)
        mask = np.logical_and(
            allocation_array == fluid_type,
            sunkcost_array == include_sunkcost,
        )
        indices = np.flatnonzero(mask)

        if len(indices) == 0:
            return CostOfSales(
                start_year=cst.start_year,
                end_year=cst.end_year,
                expense_year=np.array([cst.start_year]),
                cost=np.array([0]),
                cost_allocation=[fluid_type],
                is_sunkcost=[include_sunkcost],
            )

        return CostOfSales(
            start_year=cst.start_year,
            end_year=cst.end_year,
            expense_year=cst.expense_year[indices],
            cost=cst.cost[indices],
            cost_allocation=np.array(cst.cost_allocation)[indices].tolist(),
            description=np.array(cst.description)[indices].tolist(),
            is_sunkcost=np.array(cst.is_sunkcost)[indices].tolist(),
            tax_portion=cst.tax_portion[indices],
            tax_discount=cst.tax_discount[indices],
        )

    def _get_oil_cost_of_sales(self) -> CostOfSales:
        """
        Get oil cost of sales excluding sunk costs.

        This method retrieves the portion of the project's total cost of sales
        (`self.cost_of_sales_total`) that is allocated to crude oil (`FluidType.OIL`)
        and excludes any sunk cost items (`include_sunkcost=False`).

        Returns
        -------
        CostOfSales
            A new `CostOfSales` instance containing only the filtered oil cost of sales.

            If no matching costs are found, returns a dummy `CostOfSales` instance with:
            - Zero cost
            - Single expense year (project start year)
            - `fluid_type` set to `FluidType.OIL` in `cost_allocation`
            - `is_sunkcost` set to `False`

        Notes
        -----
        - Internally applies a boolean mask over `cost_allocation` and `is_sunkcost`
          fields of `self.cost_of_sales_total`.
        - Returned arrays (e.g., `cost`, `tax_portion`, `tax_discount`, etc.)
          have consistent lengths and are subset versions of the original arrays.
        """
        return self._get_filtered_cost_of_sales(
            fluid_type=FluidType.OIL, include_sunkcost=False
        )

    def _get_gas_cost_of_sales(self) -> CostOfSales:
        """
        Get gas cost of sales excluding sunk costs.

        This method retrieves the portion of the project's total cost of sales
        (`self.cost_of_sales_total`) that is allocated to natural gas (`FluidType.GAS`)
        and excludes any sunk cost items (`include_sunkcost=False`).

        Returns
        -------
        CostOfSales
            A new `CostOfSales` instance containing only the filtered gas cost of sales.

            If no matching costs are found, returns a dummy `CostOfSales` instance with:
            - Zero cost
            - Single expense year (project start year)
            - `fluid_type` set to `FluidType.GAS` in `cost_allocation`
            - `is_sunkcost` set to `False`

        Notes
        -----
        - Internally applies a boolean mask over `cost_allocation` and `is_sunkcost`
          fields of `self.cost_of_sales_total`.
        - Returned arrays (e.g., `cost`, `tax_portion`, `tax_discount`, etc.)
          have consistent lengths and are subset versions of the original arrays.
        """
        return self._get_filtered_cost_of_sales(
            fluid_type=FluidType.GAS, include_sunkcost=False
        )

    def _get_oil_cost_of_sales_sunk_cost(self) -> CostOfSales:
        """
        Get oil cost of sales for sunk cost items.

        This method retrieves the portion of the project's total cost of sales
        (`self.cost_of_sales_total`) that is allocated to crude oil (`FluidType.OIL`)
        and includes only sunk cost items (`include_sunkcost=True`).

        Returns
        -------
        CostOfSales
            A new `CostOfSales` instance containing only the filtered oil cost of sales
            for sunk cost items.

            If no matching costs are found, returns a dummy `CostOfSales` instance with:
            - Zero cost
            - Single expense year (project start year)
            - `fluid_type` set to `FluidType.OIL` in `cost_allocation`
            - `is_sunkcost` set to `True`

        Notes
        -----
        - Internally applies a boolean mask over `cost_allocation` and `is_sunkcost`
          fields of `self.cost_of_sales_total`.
        - Returned arrays (e.g., `cost`, `tax_portion`, `tax_discount`, etc.)
          have consistent lengths and are subset versions of the original arrays.
        """
        return self._get_filtered_cost_of_sales(
            fluid_type=FluidType.OIL, include_sunkcost=True
        )

    def _get_gas_cost_of_sales_sunk_cost(self) -> CostOfSales:
        """
        Get gas cost of sales for sunk cost items.

        This method retrieves the portion of the project's total cost of sales
        (`self.cost_of_sales_total`) that is allocated to natural gas (`FluidType.GAS`)
        and includes only sunk cost items (`include_sunkcost=True`).

        Returns
        -------
        CostOfSales
            A new `CostOfSales` instance containing only the filtered gas cost of sales
            for sunk cost items.

            If no matching costs are found, returns a dummy `CostOfSales` instance with:
            - Zero cost
            - Single expense year (project start year)
            - `fluid_type` set to `FluidType.GAS` in `cost_allocation`
            - `is_sunkcost` set to `True`

        Notes
        -----
        - Internally applies a boolean mask over `cost_allocation` and `is_sunkcost`
          fields of `self.cost_of_sales_total`.
        - Returned arrays (e.g., `cost`, `tax_portion`, `tax_discount`, etc.)
          have consistent lengths and are subset versions of the original arrays.
        """
        return self._get_filtered_cost_of_sales(
            fluid_type=FluidType.GAS, include_sunkcost=True
        )

    @staticmethod
    def _validate_sunkcost(fluid_onstream_year: int, sunkcost_objects: list) -> None:
        """
        Validate that all sunk cost expense years do not exceed the fluid onstream year.

        This method calculates the maximum expense year for each sunk cost object and
        verifies that none of them occur after the specified fluid onstream year.
        If any expense year exceeds the onstream year, a ``SunkCostException`` is raised.

        Parameters
        ----------
        fluid_onstream_year : int
            The year in which the fluid is scheduled to go onstream.
        sunkcost_objects : list of object
            A list of sunk cost objects. Each object must have an
            ``expense_year`` attribute that is array-like and contains
            the years associated with sunk cost expenses.

        Raises
        ------
        SunkCostException
            If any sunk cost's maximum expense year is greater than the
            provided fluid onstream year.
        """
        sunkcost_max_years = np.array([np.max(sc.expense_year) for sc in sunkcost_objects])

        if np.any(sunkcost_max_years > fluid_onstream_year):
            incorrect_years = sunkcost_max_years[sunkcost_max_years > fluid_onstream_year]
            raise SunkCostException(
                f"Sunk cost years ({incorrect_years}) exceed onstream year "
                f"({fluid_onstream_year})"
            )

    def _get_sunkcost_validation(self):
        """
        Validate sunk cost data for both oil and gas streams.

        This method checks that all sunk cost items for oil and gas are consistent with
        their respective onstream years. It calls the internal `_validate_sunkcost`
        method for each fluid type, passing the appropriate list of sunk cost objects.

        Validation ensures:
        - Sunk cost expense years are not later than the fluid's onstream year.
        - Objects maintain correct data length and structure after filtering.

        Returns
        -------
        None
            This method performs validation in place and does not return a value.
        """

        # Validate OIL sunkcost
        self._validate_sunkcost(
            fluid_onstream_year=self.oil_onstream_date.year,
            sunkcost_objects=[
                self._oil_capital_sunk_cost,
                self._oil_intangible_sunk_cost,
                self._oil_opex_sunk_cost,
                self._oil_asr_sunk_cost,
                self._oil_lbt_sunk_cost,
                self._oil_cost_of_sales_sunk_cost,
            ]
        )

        # Validate GAS sunkcost
        self._validate_sunkcost(
            fluid_onstream_year=self.gas_onstream_date.year,
            sunkcost_objects=[
                self._gas_capital_sunk_cost,
                self._gas_intangible_sunk_cost,
                self._gas_opex_sunk_cost,
                self._gas_asr_sunk_cost,
                self._gas_lbt_sunk_cost,
                self._gas_cost_of_sales_sunk_cost,
            ]
        )

    def _get_sunkcost_array(self):
        """
        Compute and categorize sunk costs for oil and gas.

        This method validates the sunk costs for both oil and gas, then constructs
        arrays representing various sunk cost categories (capital, intangible,
        operational, abandonment, tax, and cost of sales). It classifies them into
        depreciable and non-depreciable categories and calculates total sunk costs
        for each fluid type.

        Notes
        -----
        - Depreciable sunk costs include only capital expenditures.
        - Non-depreciable sunk costs include intangible, operational, abandonment,
          tax, and cost of sales expenditures.
        - All values are computed pre-tax using the `expenditures_pre_tax` method
          from each sunk cost object.

        Returns
        -------
        None
            This method does not return a value. It sets the following internal
            attributes:
            - _oil_depreciable_sunk_cost
            - _gas_depreciable_sunk_cost
            - _oil_non_depreciable_sunk_cost
            - _gas_non_depreciable_sunk_cost
            - _oil_sunk_cost
            - _gas_sunk_cost

        See Also
        --------
        -   _get_sunkcost_validation : Validates sunk cost data before computation.
        -   expenditures_pre_tax : Returns the pre-tax expenditures for a cost object.
        """

        # Validate OIL and GAS sunkcost
        self._get_sunkcost_validation()

        # Prepare sunkcost arrays for each sunkcost objects
        keys = [
            "oil_capital",
            "gas_capital",
            "oil_intangible",
            "gas_intangible",
            "oil_opex",
            "gas_opex",
            "oil_asr",
            "gas_asr",
            "oil_lbt",
            "gas_lbt",
            "oil_cost_of_sales",
            "gas_cost_of_sales",
        ]

        vals = [
            self._oil_capital_sunk_cost,
            self._gas_capital_sunk_cost,
            self._oil_intangible_sunk_cost,
            self._gas_intangible_sunk_cost,
            self._oil_opex_sunk_cost,
            self._gas_opex_sunk_cost,
            self._oil_asr_sunk_cost,
            self._gas_asr_sunk_cost,
            self._oil_lbt_sunk_cost,
            self._gas_lbt_sunk_cost,
            self._oil_cost_of_sales_sunk_cost,
            self._gas_cost_of_sales_sunk_cost,
        ]

        sunkcost_arr = {key: val.expenditures_pre_tax() for key, val in zip(keys, vals)}

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

        # Defone total sunkcost for OIL and GAS
        self._oil_sunk_cost = (
            self._oil_depreciable_sunk_cost + self._oil_non_depreciable_sunk_cost
        )

        self._gas_sunk_cost = (
            self._gas_depreciable_sunk_cost + self._gas_non_depreciable_sunk_cost
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
                    target_attr is self._oil_capital_cost
                    or target_attr is self._gas_capital_cost
                    or target_attr is self._oil_intangible
                    or target_attr is self._gas_intangible
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
                if target_attr is self._oil_opex or target_attr is self._gas_opex:
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
                    target_attr is self._oil_capital_cost
                    or target_attr is self._gas_capital_cost
                    or target_attr is self._oil_intangible
                    or target_attr is self._gas_intangible
                    or target_attr is self._oil_opex
                    or target_attr is self._gas_opex
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
                    target_attr is self._oil_capital_cost
                    or target_attr is self._gas_capital_cost
                    or target_attr is self._oil_intangible
                    or target_attr is self._gas_intangible
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
                if target_attr is self._oil_opex or target_attr is self._gas_opex:
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
                    target_attr is self._oil_capital_cost
                    or target_attr is self._gas_capital_cost
                    or target_attr is self._oil_intangible
                    or target_attr is self._gas_intangible
                    or target_attr is self._oil_opex
                    or target_attr is self._gas_opex
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
        Calculate and assign pre-tax expenditures for various categories, adjusted for inflation.

        This method calculates the pre-tax expenditures for multiple cost categories related
        to oil and gas, adjusted by the provided inflation rate. The expenditures are then
        assigned to the corresponding attributes for both oil and gas.

        Parameters
        ----------
        year_inflation : np.ndarray, optional
            A NumPy array representing the years during which inflation is applied to the costs.
            If not provided, defaults to repeating the `start_year` for all costs.
        inflation_rate : np.ndarray or float, optional
            The inflation rate(s) to apply to the project costs. If provided as a float,
            a uniform inflation rate is applied. If provided as a NumPy array, different
            rates are applied based on the corresponding project years. Default is 0.0.
        inflation_rate_applied_to
            The selection of where the cost inflation will be applied to.

        Returns
        -------
        None
            This method does not return a value. It updates the following attributes with the
            calculated pre-tax expenditures:

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
                self._oil_capital_cost,
                self._gas_capital_cost,
                self._oil_intangible,
                self._gas_intangible,
                self._oil_opex,
                self._gas_opex,
                self._oil_asr,
                self._gas_asr,
                self._oil_lbt,
                self._gas_lbt,
            ]
        ]

        # Prepare expenditures pre tax associated with cost of sales
        (
            self._oil_cost_of_sales_expenditures_pre_tax,
            self._gas_cost_of_sales_expenditures_pre_tax,
        ) = [
            attr.expenditures_pre_tax()
            for attr in [
                self._oil_cost_of_sales,
                self._gas_cost_of_sales,
            ]
        ]

    def _get_indirect_taxes(self, tax_rate: np.ndarray | float = 0.0) -> None:
        """
        Calculate and assign indirect taxes for various oil and gas expenditure categories.

        This method computes the indirect taxes (such as VAT or other applicable indirect taxes)
        for multiple categories related to oil and gas. It adjusts these taxes based on the
        provided tax portion, tax rate, and tax discount.

        Parameters
        ----------
        tax_rate : np.ndarray or float, optional
            A NumPy array or float representing the tax rate applied to the costs. If not
            provided, a default rate of 0.0 will be used. When provided as an array, it
            should match the project years.

        Returns
        -------
        None
            This method does not return any values. It updates the following attributes with
            the calculated indirect taxes for oil and gas:

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
                self._oil_capital_cost,
                self._gas_capital_cost,
                self._oil_intangible,
                self._gas_intangible,
                self._oil_opex,
                self._gas_opex,
                self._oil_asr,
                self._gas_asr,
                self._oil_lbt,
                self._gas_lbt,
            ]
        ]

        # Prepare indirect taxes associated with cost of sales
        (
            self._oil_cost_of_sales_indirect_tax,
            self._gas_cost_of_sales_indirect_tax,
        ) = [
            attr.indirect_taxes()
            for attr in [
                self._oil_cost_of_sales,
                self._gas_cost_of_sales,
            ]
        ]

    def _get_expenditures_post_tax(self) -> None:
        """
        Calculate and assign post-tax expenditures for multiple oil and gas cost categories.

        This method updates each post-tax expenditure attribute by adding its corresponding
        pre-tax expenditure and indirect tax values for all cost categories
        (e.g., capital, intangible, operating, abandonment & site restoration (ASR),
        land and building tax (LBT), and cost of sales) for both oil and gas.

        Notes
        -----
        The method assumes that:
            -   Each cost category has corresponding ``*_pre_tax`` and ``*_indirect_tax``
                attributes.
            -   Post-tax results will be stored in matching ``*_post_tax`` attributes.
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

        # Prepare sunkcost
        self._get_sunkcost_array()

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

        # Configure base cashflow for OIL and GAS
        self._oil_cashflow = (
            self._oil_revenue - (self._oil_sunk_cost + self._oil_total_expenditures_post_tax)
        )

        self._gas_cashflow = (
            self._gas_revenue - (self._gas_sunk_cost + self._gas_total_expenditures_post_tax)
        )

        # Prepare consolidated profiles
        self._get_consolidated_profiles()

    def __len__(self):
        return self.project_duration
