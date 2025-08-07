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
    TaxRegime,
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

    # Attributes to be defined later (associated with sunk cost in each cost categories)
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

    # Attributes to be defined later (associated with sunk cost)
    _oil_sunk_cost: np.ndarray = field(default=None, init=False, repr=False)
    _gas_sunk_cost: np.ndarray = field(default=None, init=False, repr=False)
    _oil_sunk_cost_amortization_charge: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _gas_sunk_cost_amortization_charge: np.ndarray = field(
        default=None, init=False, repr=False
    )

    # Attributes to be defined later (associated with cashflow)
    _oil_cashflow: np.ndarray = field(default=None, init=False, repr=False)
    _gas_cashflow: np.ndarray = field(default=None, init=False, repr=False)

    # Atributes to be defined later (associated with WAP price)
    _oil_wap_price: np.ndarray = field(default=None, init=False, repr=False)
    _gas_wap_price: np.ndarray = field(default=None, init=False, repr=False)
    _sulfur_wap_price: np.ndarray = field(default=None, init=False, repr=False)
    _electricity_wap_price: np.ndarray = field(default=None, init=False, repr=False)
    _co2_wap_price: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes to be defined later (associated with non capital costs)
    _oil_non_capital: np.ndarray = field(default=None, init=False, repr=False)
    _gas_non_capital: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes to be defined later (associated with consolidated profiles)
    _consolidated_revenue: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_cashflow: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_government_take: np.ndarray = field(
        default=None, init=False, repr=False
    )

    # Attributes to be defined later (associated with consolidated sunk cost)
    _consolidated_sunk_cost_array: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_sunk_cost_amortization_charge: np.ndarray = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self):
        """
        Handles the following operations/procedures:
        -   Prepare attributes project_duration and project_years;
        -   Prepare attribute lifting;
        -   Prepare attribute capital_cost;
        -   Prepare attribute intangible_cost;
        -   Prepare attribute opex;
        -   Prepare attribute asr_cost;
        -   Prepare attribute lbt_cost;
        -   Prepare attribute cost_of_sales;
        -   Prepare attributes associated with total cost per component
            (leveraging dunder method __add__). The attributes are: capital_cost_total,
            intangible_cost_total, opex_total, asr_cost_total, lbt_cost_total, and cost_of_sales_total;
        -   Prepare attributes associated with lifting for each fluid types. The attributes are:
            _oil_lifting, _gas_lifting, _sulfur_lifting, _electricity_lifting, and _co2_lifting;
        -   Prepare attributes associated with revenue for each fluid types. The attributes are:
            _oil_revenue, _gas_revenue, _sulfur_revenue, _electricity_revenue, and _co2_revenue;
        -   Prepare attributes _oil_capital_cost and _gas_capital_cost;
        -   Prepare attributes _oil_intangible and _gas_intangible;
        -   Prepare attributes _oil_opex and _gas_opex;
        -   Prepare attributes _oil_asr and _gas_asr;
        -   Prepare attributes _oil_lbt and _gas_lbt;
        -   Prepare attributes _oil_cost_of_sales and _gas_cost_of_sales;
        -   Raise an error: the start year of the project is inconsistent;
        -   Raise an error: the end year of the project is inconsistent;
        -   Prepare attribute oil_onstream_date;
        -   Prepare attribute gas_onstream_date;
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

        print('\t')
        print(f'Filetype: {type(self.lbt_cost_total)}')
        print(f'Length: {len(self.lbt_cost_total)}')
        print('lbt_cost_total = ', self.lbt_cost_total)

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
        # self._oil_cost_of_sales = self._get_oil_cost_of_sales()
        # self._gas_cost_of_sales = self._get_gas_cost_of_sales()
        # # self._oil_sunk_cost = self._get_oil_sunk_cost()
        # # self._gas_sunk_cost = self._get_gas_sunk_cost()

        print('\t')
        print(f'Filetype: {type(self._gas_lbt)}')
        print(f'Length: {len(self._gas_lbt)}')
        print('_gas_lbt = ', self._gas_lbt)

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

        print('\t')
        print(f'Filetype: {type(self._gas_lbt_sunk_cost)}')
        print(f'Length: {len(self._gas_lbt_sunk_cost)}')
        print('_gas_lbt_sunk_cost = ', self._gas_lbt_sunk_cost)


        # # Raise an exception error if the start year of the project is inconsistent
        # if not all(
        #     i == self.start_date.year
        #     for i in [
        #         self._oil_lifting.start_year,
        #         self._gas_lifting.start_year,
        #         self._sulfur_lifting.start_year,
        #         self._electricity_lifting.start_year,
        #         self._co2_lifting.start_year,
        #         self._oil_capital_cost.start_year,
        #         self._gas_capital_cost.start_year,
        #         self._oil_intangible.start_year,
        #         self._gas_intangible.start_year,
        #         self._oil_opex.start_year,
        #         self._gas_opex.start_year,
        #         self._oil_asr.start_year,
        #         self._gas_asr.start_year,
        #         self._oil_lbt.start_year,
        #         self._gas_lbt.start_year,
        #         self._oil_cost_of_sales.start_year,
        #         self._gas_cost_of_sales.start_year,
        #         # self._oil_sunk_cost.start_year,
        #         # self._gas_sunk_cost.start_year,
        #     ]
        # ):
        #     raise BaseProjectException(
        #         f"Inconsistent start project year: "
        #         f"Base project ({self.start_date.year}), "
        #         f"Oil lifting ({self._oil_lifting.start_year}), "
        #         f"Gas lifting ({self._gas_lifting.start_year}), "
        #         f"Sulfur lifting ({self._sulfur_lifting.start_year}), "
        #         f"Electricity lifting ({self._electricity_lifting.start_year}), "
        #         f"CO2 lifting ({self._co2_lifting.start_year}), "
        #         f"Oil tangible ({self._oil_capital_cost.start_year}), "
        #         f"Gas tangible ({self._gas_capital_cost.start_year}), "
        #         f"Oil intangible ({self._oil_intangible.start_year}), "
        #         f"Gas intangible ({self._gas_intangible.start_year}), "
        #         f"Oil opex ({self._oil_opex.start_year}), "
        #         f"Gas opex ({self._gas_opex.start_year}), "
        #         f"Oil asr ({self._oil_asr.start_year}), "
        #         f"Gas asr ({self._gas_asr.start_year}), "
        #         f"Oil LBT ({self._oil_lbt.start_year}), "
        #         f"Gas LBT ({self._gas_lbt.start_year}), "
        #         f"Oil cost of sales ({self._oil_cost_of_sales.start_year}), "
        #         f"Gas cost of sales ({self._gas_cost_of_sales.start_year}), "
        #         # f"Oil sunk cost ({self._oil_sunk_cost.start_year}), "
        #         # f"Gas sunk cost ({self._gas_sunk_cost.start_year}). "
        #     )
        #
        # # Raise an exception error if the end year of the project is inconsistent
        # if not all(
        #     i == self.end_date.year
        #     for i in [
        #         self._oil_lifting.end_year,
        #         self._gas_lifting.end_year,
        #         self._sulfur_lifting.end_year,
        #         self._electricity_lifting.end_year,
        #         self._co2_lifting.end_year,
        #         self._oil_capital_cost.end_year,
        #         self._gas_capital_cost.end_year,
        #         self._oil_intangible.end_year,
        #         self._gas_intangible.end_year,
        #         self._oil_opex.end_year,
        #         self._gas_opex.end_year,
        #         self._oil_asr.end_year,
        #         self._gas_asr.end_year,
        #         self._oil_lbt.end_year,
        #         self._gas_lbt.end_year,
        #         self._oil_cost_of_sales.end_year,
        #         self._gas_cost_of_sales.end_year,
        #         # self._oil_sunk_cost.end_year,
        #         # self._gas_sunk_cost.end_year,
        #     ]
        # ):
        #     raise BaseProjectException(
        #         f"Inconsistent end project year: "
        #         f"Base project ({self.end_date.year}), "
        #         f"Oil lifting ({self._oil_lifting.end_year}), "
        #         f"Gas lifting ({self._gas_lifting.end_year}), "
        #         f"Sulfur lifting ({self._sulfur_lifting.end_year}), "
        #         f"Electricity lifting ({self._electricity_lifting.end_year}), "
        #         f"CO2 lifting ({self._co2_lifting.end_year}), "
        #         f"Oil tangible ({self._oil_capital_cost.end_year}), "
        #         f"Gas tangible ({self._gas_capital_cost.end_year}), "
        #         f"Oil intangible ({self._oil_intangible.end_year}), "
        #         f"Gas intangible ({self._gas_intangible.end_year}), "
        #         f"Oil opex ({self._oil_opex.end_year}), "
        #         f"Gas opex ({self._gas_opex.end_year}), "
        #         f"Oil asr ({self._oil_asr.end_year}), "
        #         f"Gas asr ({self._gas_asr.end_year}), "
        #         f"Oil LBT ({self._oil_lbt.end_year}), "
        #         f"Gas LBT ({self._gas_lbt.end_year}), "
        #         f"Oil cost of sales ({self._oil_cost_of_sales.end_year}), "
        #         f"Gas cost of sales ({self._gas_cost_of_sales.end_year}), "
        #         # f"Oil sunk cost ({self._oil_sunk_cost.end_year}), "
        #         # f"Gas sunk cost ({self._gas_sunk_cost.end_year}). "
        #     )

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

    def _get_filtered_capital_cost(
        self,
        fluid_type: FluidType,
        include_sunkcost: bool,
    ) -> CapitalCost:
        """
        Get capital costs filtered by fluid type and sunk cost inclusion.

        Parameters
        ----------
        fluid_type : FluidType
            The fluid type to filter capital costs by.
        include_sunkcost : bool
            Whether to include sunk costs in the filtered results.
            If True, includes sunk costs; if False, excludes them.

        Returns
        -------
        CapitalCost
            A new CapitalCost instance containing only the costs that match:
            1. The specified fluid type
            2. The sunk cost inclusion criteria

            If no matching costs are found, returns a CapitalCost instance with:
            - Zero cost
            - Single expense year (project start year)
            - The requested fluid type in cost_allocation
            - The requested sunk cost flag in is_sunkcost

        Notes
        -----
        - The method filters the data from self.capital_cost_total based on the criteria.
        - All returned arrays (cost, expense_year, etc.) will have the same length.
        - The filtering preserves all original CapitalCost attributes, but with only
          the matching elements in each array.
        - When no matching costs exist, the returned dummy instance will include:
          * The requested fluid_type in cost_allocation
          * The requested include_sunkcost value in is_sunkcost
        """

        # Handle case when requested fluid type is not found
        if fluid_type not in self.capital_cost_total.cost_allocation:
            return CapitalCost(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                expense_year=np.array([self.start_date.year]),
                cost=np.array([0]),
                cost_allocation=[fluid_type],
                is_sunkcost=[include_sunkcost],
            )

        else:
            # Configure indices to slice data according to fluid type and sunk cost
            allocation_array = np.array(self.capital_cost_total.cost_allocation)
            sunkcost_array = np.array(self.capital_cost_total.is_sunkcost)
            mask = np.logical_and(
                allocation_array == fluid_type,
                sunkcost_array == include_sunkcost,
            )
            indices = np.flatnonzero(mask)

            # Slice filtered data, return a new instance of CapitalCost with filtered data
            return CapitalCost(
                start_year=self.capital_cost_total.start_year,
                end_year=self.capital_cost_total.end_year,
                expense_year=self.capital_cost_total.expense_year[indices],
                cost=self.capital_cost_total.cost[indices],
                cost_allocation=(
                    np.array(self.capital_cost_total.cost_allocation)[indices].tolist()
                ),
                description=np.array(self.capital_cost_total.description)[indices].tolist(),
                is_sunkcost=np.array(self.capital_cost_total.is_sunkcost)[indices].tolist(),
                tax_portion=self.capital_cost_total.tax_portion[indices],
                tax_discount=self.capital_cost_total.tax_discount[indices],
                pis_year=self.capital_cost_total.pis_year[indices],
                salvage_value=self.capital_cost_total.salvage_value[indices],
                useful_life=self.capital_cost_total.useful_life[indices],
                depreciation_factor=self.capital_cost_total.depreciation_factor[indices],
                is_ic_applied=(
                    np.array(self.capital_cost_total.is_ic_applied)[indices].tolist()
                ),
            )

    def _get_oil_capital(self) -> CapitalCost:
        """
        Get capital costs specifically for OIL, excluding sunk costs.

        Returns
        -------
        CapitalCost
            A new CapitalCost instance containing only the costs that:
            1. Are allocated to OIL fluid type
            2. Are not sunk costs

            If no matching costs are found, returns a CapitalCost instance with:
            - Zero cost
            - Single expense year (project start year)
            - OIL as the cost_allocation
            - False in is_sunkcost (matching the include_sunkcost=False filter)

        Notes
        -----
        - This is a convenience wrapper around _get_filtered_capital_cost() with:
          fluid_type=FluidType.OIL and include_sunkcost=False
        - All returned arrays will have the same length.
        - The dummy instance (when no costs exist) will include:
          * OIL in cost_allocation
          * False in is_sunkcost
        - See _get_filtered_capital_cost() for more detailed filtering behavior.
        """
        return self._get_filtered_capital_cost(
            fluid_type=FluidType.OIL, include_sunkcost=False
        )

    def _get_gas_capital(self) -> CapitalCost:
        """
        Get capital costs specifically for GAS, excluding sunk costs.

        Returns
        -------
        CapitalCost
            A new CapitalCost instance containing only the costs that:
            1. Are allocated to GAS fluid type
            2. Are not sunk costs

            If no matching costs are found, returns a CapitalCost instance with:
            - Zero cost
            - Single expense year (project start year)
            - GAS as the cost_allocation
            - False in is_sunkcost (matching the include_sunkcost=False filter)

        Notes
        -----
        - This is a convenience wrapper around _get_filtered_capital_cost() with:
          fluid_type=FluidType.GAS and include_sunkcost=False
        - All returned arrays will have the same length.
        - The dummy instance (when no costs exist) will include:
          * GAS in cost_allocation
          * False in is_sunkcost
        - See _get_filtered_capital_cost() for more detailed filtering behavior.
        """
        return self._get_filtered_capital_cost(
            fluid_type=FluidType.GAS, include_sunkcost=False
        )

    def _get_oil_capital_sunk_cost(self) -> CapitalCost:
        """
        Get capital costs specifically for OIL, including only sunk costs.

        Returns
        -------
        CapitalCost
            A new CapitalCost instance containing only the costs that:
            1. Are allocated to OIL fluid type
            2. Are sunk costs (already incurred)

            If no matching costs are found, returns a CapitalCost instance with:
            - Zero cost
            - Single expense year (project start year)
            - OIL as the cost_allocation
            - True in is_sunkcost (matching the include_sunkcost=True filter)

        Notes
        -----
        - This is a convenience wrapper around _get_filtered_capital_cost() with:
          fluid_type=FluidType.OIL and include_sunkcost=True
        - All returned arrays will have the same length.
        - The dummy instance (when no costs exist) will include:
          * OIL in cost_allocation
          * True in is_sunkcost
        - See _get_filtered_capital_cost() for more detailed filtering behavior.
        """
        return self._get_filtered_capital_cost(
            fluid_type=FluidType.OIL, include_sunkcost=True
        )

    def _get_gas_capital_sunk_cost(self) -> CapitalCost:
        """
        Get capital costs specifically for GAS, including only sunk costs.

        Returns
        -------
        CapitalCost
            A new CapitalCost instance containing only the costs that:
            1. Are allocated to GAS fluid type
            2. Are sunk costs (already incurred)

            If no matching costs are found, returns a CapitalCost instance with:
            - Zero cost
            - Single expense year (project start year)
            - GAS as the cost_allocation
            - True in is_sunkcost (matching the include_sunkcost=True filter)

        Notes
        -----
        - This is a convenience wrapper around _get_filtered_capital_cost() with:
          fluid_type=FluidType.GAS and include_sunkcost=True
        - All returned arrays will have the same length.
        - The dummy instance (when no costs exist) will include:
          * GAS in cost_allocation
          * True in is_sunkcost
        - See _get_filtered_capital_cost() for more detailed filtering behavior.
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

        Parameters
        ----------
        fluid_type : FluidType
            The fluid type to filter intangible costs by.
        include_sunkcost : bool
            Whether to include sunk costs in the filtered results.
            If True, includes sunk costs; if False, excludes them.

        Returns
        -------
        Intangible
            A new Intangible instance containing only the costs that match:
            1. The specified fluid type
            2. The sunk cost inclusion criteria

            If no matching costs are found, returns an Intangible instance with:
            - Zero cost
            - Single expense year (project start year)
            - The requested fluid type in cost_allocation
            - The requested sunk cost flag in is_sunkcost

        Notes
        -----
        - The method filters the data from self.intangible_cost_total based on the criteria.
        - All returned arrays (cost, expense_year, etc.) will have the same length.
        - The filtering preserves all original Intangible attributes, but with only
          the matching elements in each array.
        - When no matching costs exist, the returned dummy instance will include:
          * The requested fluid_type in cost_allocation
          * The requested include_sunkcost value in is_sunkcost
        """

        # Handle case when requested fluid type is not found
        if fluid_type not in self.intangible_cost_total.cost_allocation:
            return Intangible(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                expense_year=np.array([self.start_date.year]),
                cost=np.array([0]),
                cost_allocation=[fluid_type],
                is_sunkcost=[include_sunkcost],
            )

        else:
            # Configure indices to slice data according to fluid type and sunk cost
            allocation_array = np.array(self.intangible_cost_total.cost_allocation)
            sunkcost_array = np.array(self.intangible_cost_total.is_sunkcost)
            mask = np.logical_and(
                allocation_array == fluid_type,
                sunkcost_array == include_sunkcost,
            )
            indices = np.flatnonzero(mask)

            # Slice filtered data, return a new instance of Intangible with filtered data
            return Intangible(
                start_year=self.intangible_cost_total.start_year,
                end_year=self.intangible_cost_total.end_year,
                expense_year=self.intangible_cost_total.expense_year[indices],
                cost=self.intangible_cost_total.cost[indices],
                cost_allocation=(
                    np.array(self.intangible_cost_total.cost_allocation)[indices].tolist()
                ),
                description=np.array(self.intangible_cost_total.description)[indices].tolist(),
                is_sunkcost=np.array(self.intangible_cost_total.is_sunkcost)[indices].tolist(),
                tax_portion=self.intangible_cost_total.tax_portion[indices],
                tax_discount=self.intangible_cost_total.tax_discount[indices],
            )

    def _get_oil_intangible(self) -> Intangible:
        """
        Get intangible costs specifically for OIL, excluding sunk costs.

        Returns
        -------
        Intangible
            A new Intangible instance containing only the costs that:
            1. Are allocated to OIL fluid type
            2. Are not sunk costs

            If no matching costs are found, returns an Intangible instance with:
            - Zero cost
            - Single expense year (project start year)
            - OIL as the cost_allocation
            - False in is_sunkcost (matching the include_sunkcost=False filter)

        Notes
        -----
        - This is a convenience wrapper around _get_filtered_intangible() with:
          fluid_type=FluidType.OIL and include_sunkcost=False
        - All returned arrays will have the same length.
        - The dummy instance (when no costs exist) will include:
          * OIL in cost_allocation
          * False in is_sunkcost
        - See _get_filtered_intangible() for more detailed filtering behavior.
        """
        return self._get_filtered_intangible(fluid_type=FluidType.OIL, include_sunkcost=False)

    def _get_gas_intangible(self) -> Intangible:
        """
        Get intangible costs specifically for GAS, excluding sunk costs.

        Returns
        -------
        Intangible
            A new Intangible instance containing only the costs that:
            1. Are allocated to GAS fluid type
            2. Are not sunk costs

            If no matching costs are found, returns an Intangible instance with:
            - Zero cost
            - Single expense year (project start year)
            - GAS as the cost_allocation
            - False in is_sunkcost (matching the include_sunkcost=False filter)

        Notes
        -----
        - This is a convenience wrapper around _get_filtered_intangible() with:
          fluid_type=FluidType.GAS and include_sunkcost=False
        - All returned arrays will have the same length.
        - The dummy instance (when no costs exist) will include:
          * GAS in cost_allocation
          * False in is_sunkcost
        - See _get_filtered_intangible() for more detailed filtering behavior.
        """
        return self._get_filtered_intangible(fluid_type=FluidType.GAS, include_sunkcost=False)

    def _get_oil_intangible_sunk_cost(self) -> Intangible:
        """
        Get intangible costs specifically for OIL, including only sunk costs.

        Returns
        -------
        Intangible
            A new Intangible instance containing only the costs that:
            1. Are allocated to OIL fluid type
            2. Are sunk costs (already incurred)

            If no matching costs are found, returns an Intangible instance with:
            - Zero cost
            - Single expense year (project start year)
            - OIL as the cost_allocation
            - True in is_sunkcost (matching the include_sunkcost=True filter)

        Notes
        -----
        - This is a convenience wrapper around _get_filtered_intangible() with:
          fluid_type=FluidType.OIL and include_sunkcost=True
        - All returned arrays will have the same length.
        - The dummy instance (when no costs exist) will include:
          * OIL in cost_allocation
          * True in is_sunkcost
        - See _get_filtered_intangible() for more detailed filtering behavior.
        """
        return self._get_filtered_intangible(fluid_type=FluidType.OIL, include_sunkcost=True)

    def _get_gas_intangible_sunk_cost(self) -> Intangible:
        """
        Get intangible costs specifically for GAS, including only sunk costs.

        Returns
        -------
        Intangible
            A new Intangible instance containing only the costs that:
            1. Are allocated to GAS fluid type
            2. Are sunk costs (already incurred)

            If no matching costs are found, returns an Intangible instance with:
            - Zero cost
            - Single expense year (project start year)
            - GAS as the cost_allocation
            - True in is_sunkcost (matching the include_sunkcost=True filter)

        Notes
        -----
        - This is a convenience wrapper around _get_filtered_intangible() with:
          fluid_type=FluidType.GAS and include_sunkcost=True
        - All returned arrays will have the same length.
        - The dummy instance (when no costs exist) will include:
          * GAS in cost_allocation
          * True in is_sunkcost
        - See _get_filtered_intangible() for more detailed filtering behavior.
        """
        return self._get_filtered_intangible(fluid_type=FluidType.GAS, include_sunkcost=True)

    def _get_filtered_opex(
        self,
        fluid_type: FluidType,
        include_sunkcost: bool,
    ) -> OPEX:
        """
        Get operating expenditures (OPEX) filtered by fluid type and sunk cost inclusion.

        Parameters
        ----------
        fluid_type : FluidType
            The fluid type to filter operating costs by.
        include_sunkcost : bool
            Whether to include sunk costs in the filtered results.
            If True, includes sunk costs; if False, excludes them.

        Returns
        -------
        OPEX
            A new OPEX instance containing only the costs that match:
            1. The specified fluid type
            2. The sunk cost inclusion criteria

            If no matching costs are found, returns an OPEX instance with:
            - Zero fixed cost
            - Single expense year (project start year)
            - The requested fluid type in cost_allocation
            - The requested sunk cost flag in is_sunkcost

        Notes
        -----
        - The method filters the data from self.opex_total based on the criteria.
        - All returned arrays (fixed_cost, expense_year, etc.) will have the same length.
        - The filtering preserves all original OPEX attributes, but with only
          the matching elements in each array.
        - When no matching costs exist, the returned dummy instance will include:
          * The requested fluid_type in cost_allocation
          * The requested include_sunkcost value in is_sunkcost
        """

        # Handle case when requested fluid type is not found
        if fluid_type not in self.opex_total.cost_allocation:
            return OPEX(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                expense_year=np.array([self.start_date.year]),
                fixed_cost=np.array([0]),
                cost_allocation=[fluid_type],
                is_sunkcost=[include_sunkcost],
            )

        else:
            # Configure indices to slice data according to fluid type and sunk cost
            allocation_array = np.array(self.opex_total.cost_allocation)
            sunkcost_array = np.array(self.opex_total.is_sunkcost)
            mask = np.logical_and(
                allocation_array == fluid_type,
                sunkcost_array == include_sunkcost,
            )
            indices = np.flatnonzero(mask)

            # Slice filtered data, return a new instance of OPEX with filtered data
            return OPEX(
                start_year=self.opex_total.start_year,
                end_year=self.opex_total.end_year,
                expense_year=self.opex_total.expense_year[indices],
                cost_allocation=np.array(self.opex_total.cost_allocation)[indices].tolist(),
                description=np.array(self.opex_total.description)[indices].tolist(),
                is_sunkcost=np.array(self.opex_total.is_sunkcost)[indices].tolist(),
                tax_portion=self.opex_total.tax_portion[indices],
                tax_discount=self.opex_total.tax_discount[indices],
                fixed_cost=self.opex_total.fixed_cost[indices],
                prod_rate=self.opex_total.prod_rate[indices],
                cost_per_volume=self.opex_total.cost_per_volume[indices],
            )

    def _get_oil_opex(self) -> OPEX:
        """
        Get operating expenditures (OPEX) specifically for OIL, excluding sunk costs.

        Returns
        -------
        OPEX
            A new OPEX instance containing only the costs that:
            1. Are allocated to OIL fluid type
            2. Are not sunk costs

            If no matching costs are found, returns an OPEX instance with:
            - Zero fixed cost
            - Single expense year (project start year)
            - OIL as the cost_allocation
            - False in is_sunkcost (matching the include_sunkcost=False filter)

        Notes
        -----
        - This is a convenience wrapper around _get_filtered_opex() with:
          fluid_type=FluidType.OIL and include_sunkcost=False
        - All returned arrays will have the same length.
        - The dummy instance (when no costs exist) will include:
          * OIL in cost_allocation
          * False in is_sunkcost
        - See _get_filtered_opex() for more detailed filtering behavior.
        """
        return self._get_filtered_opex(fluid_type=FluidType.OIL, include_sunkcost=False)

    def _get_gas_opex(self) -> OPEX:
        """
        Get operating expenditures (OPEX) specifically for GAS, excluding sunk costs.

        Returns
        -------
        OPEX
            A new OPEX instance containing only the costs that:
            1. Are allocated to GAS fluid type
            2. Are not sunk costs

            If no matching costs are found, returns an OPEX instance with:
            - Zero fixed cost
            - Single expense year (project start year)
            - GAS as the cost_allocation
            - False in is_sunkcost (matching the include_sunkcost=False filter)

        Notes
        -----
        - This is a convenience wrapper around _get_filtered_opex() with:
          fluid_type=FluidType.GAS and include_sunkcost=False
        - All returned arrays will have the same length.
        - The dummy instance (when no costs exist) will include:
          * GAS in cost_allocation
          * False in is_sunkcost
        - See _get_filtered_opex() for more detailed filtering behavior.
        """
        return self._get_filtered_opex(fluid_type=FluidType.GAS, include_sunkcost=False)

    def _get_oil_opex_sunk_cost(self) -> OPEX:
        """
        Get operating expenditures (OPEX) specifically for OIL, including only sunk costs.

        Returns
        -------
        OPEX
            A new OPEX instance containing only the costs that:
            1. Are allocated to OIL fluid type
            2. Are sunk costs (already incurred)

            If no matching costs are found, returns an OPEX instance with:
            - Zero fixed cost
            - Single expense year (project start year)
            - OIL as the cost_allocation
            - True in is_sunkcost (matching the include_sunkcost=True filter)

        Notes
        -----
        - This is a convenience wrapper around _get_filtered_opex() with:
          fluid_type=FluidType.OIL and include_sunkcost=True
        - All returned arrays will have the same length.
        - The dummy instance (when no costs exist) will include:
          * OIL in cost_allocation
          * True in is_sunkcost
        - See _get_filtered_opex() for more detailed filtering behavior.
        """
        return self._get_filtered_opex(fluid_type=FluidType.OIL, include_sunkcost=True)

    def _get_gas_opex_sunk_cost(self) -> OPEX:
        """
        Get operating expenditures (OPEX) specifically for GAS, including only sunk costs.

        Returns
        -------
        OPEX
            A new OPEX instance containing only the costs that:
            1. Are allocated to GAS fluid type
            2. Are sunk costs (already incurred)

            If no matching costs are found, returns an OPEX instance with:
            - Zero fixed cost
            - Single expense year (project start year)
            - GAS as the cost_allocation
            - True in is_sunkcost (matching the include_sunkcost=True filter)

        Notes
        -----
        - This is a convenience wrapper around _get_filtered_opex() with:
          fluid_type=FluidType.GAS and include_sunkcost=True
        - All returned arrays will have the same length.
        - The dummy instance (when no costs exist) will include:
          * GAS in cost_allocation
          * True in is_sunkcost
        - See _get_filtered_opex() for more detailed filtering behavior.
        """
        return self._get_filtered_opex(fluid_type=FluidType.GAS, include_sunkcost=True)

    def _get_filtered_asr(
        self,
        fluid_type: FluidType,
        include_sunkcost: bool,
    ) -> ASR:
        """
        Get abandonment, site restoration, and post-operation costs (ASR) filtered
        by fluid type and sunk cost inclusion.

        Parameters
        ----------
        fluid_type : FluidType
            The fluid type to filter ASR costs by.
        include_sunkcost : bool
            Whether to include sunk costs in the filtered results.
            If True, includes sunk costs; if False, excludes them.

        Returns
        -------
        ASR
            A new ASR instance containing only the costs that match:
            1. The specified fluid type
            2. The sunk cost inclusion criteria

            If no matching costs are found, returns an ASR instance with:
            - Zero cost
            - Single expense year (project start year)
            - The requested fluid type in cost_allocation
            - The requested sunk cost flag in is_sunkcost

        Notes
        -----
        - The method filters the data from self.asr_cost_total based on the criteria.
        - All returned arrays (cost, expense_year, etc.) will have the same length.
        - The filtering preserves all original ASR attributes, but with only
          the matching elements in each array.
        - When no matching costs exist, the returned dummy instance will include:
          * The requested fluid_type in cost_allocation
          * The requested include_sunkcost value in is_sunkcost
        """

        # Handle case when requested fluid type is not found
        if fluid_type not in self.asr_cost_total.cost_allocation:
            return ASR(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                expense_year=np.array([self.start_date.year]),
                cost=np.array([0]),
                cost_allocation=[fluid_type],
                is_sunkcost=[include_sunkcost],
            )

        else:
            # Configure indices to slice data according to fluid type and sunk cost
            allocation_array = np.array(self.asr_cost_total.cost_allocation)
            sunkcost_array = np.array(self.asr_cost_total.is_sunkcost)
            mask = np.logical_and(
                allocation_array == fluid_type,
                sunkcost_array == include_sunkcost,
            )
            indices = np.flatnonzero(mask)

            # Slice filtered data, return a new instance of ASR with filtered data
            return ASR(
                start_year=self.asr_cost_total.start_year,
                end_year=self.asr_cost_total.end_year,
                expense_year=self.asr_cost_total.expense_year[indices],
                cost=self.asr_cost_total.cost[indices],
                cost_allocation=np.array(self.asr_cost_total.cost_allocation)[indices].tolist(),
                description=np.array(self.asr_cost_total.description)[indices].tolist(),
                is_sunkcost=np.array(self.asr_cost_total.is_sunkcost)[indices].tolist(),
                tax_portion=self.asr_cost_total.tax_portion[indices],
                tax_discount=self.asr_cost_total.tax_discount[indices],
                final_year=self.asr_cost_total.final_year[indices],
                future_rate=self.asr_cost_total.future_rate[indices],
            )

    def _get_oil_asr(self) -> ASR:
        """
        Get ASR (abandonment, site restoration, and post-operation) costs
        specifically for OIL, excluding sunk costs.

        Returns
        -------
        ASR
            A new ASR instance containing only the costs that:
            1. Are allocated to OIL fluid type
            2. Are not sunk costs

            If no matching costs are found, returns an ASR instance with:
            - Zero cost
            - Single expense year (project start year)
            - OIL as the cost_allocation
            - False in is_sunkcost (matching the include_sunkcost=False filter)

        Notes
        -----
        - This is a convenience wrapper around _get_filtered_asr() with:
          fluid_type=FluidType.OIL and include_sunkcost=False
        - All returned arrays will have the same length.
        - The dummy instance (when no costs exist) will include:
          * OIL in cost_allocation
          * False in is_sunkcost
        - See _get_filtered_asr() for more detailed filtering behavior.
        """
        return self._get_filtered_asr(fluid_type=FluidType.OIL, include_sunkcost=False)

    def _get_gas_asr(self) -> ASR:
        """
        Get ASR (abandonment, site restoration, and post-operation) costs
        specifically for GAS, excluding sunk costs.

        Returns
        -------
        ASR
            A new ASR instance containing only the costs that:
            1. Are allocated to GAS fluid type
            2. Are not sunk costs

            If no matching costs are found, returns an ASR instance with:
            - Zero cost
            - Single expense year (project start year)
            - GAS as the cost_allocation
            - False in is_sunkcost (matching the include_sunkcost=False filter)

        Notes
        -----
        - This is a convenience wrapper around _get_filtered_asr() with:
          fluid_type=FluidType.GAS and include_sunkcost=False
        - All returned arrays will have the same length.
        - The dummy instance (when no costs exist) will include:
          * GAS in cost_allocation
          * False in is_sunkcost
        - See _get_filtered_asr() for more detailed filtering behavior.
        """
        return self._get_filtered_asr(fluid_type=FluidType.GAS, include_sunkcost=False)

    def _get_oil_asr_sunk_cost(self) -> ASR:
        """
        Get ASR (abandonment, site restoration, and post-operation) costs
        specifically for OIL, including only sunk costs.

        Returns
        -------
        ASR
            A new ASR instance containing only the costs that:
            1. Are allocated to OIL fluid type
            2. Are sunk costs (already incurred)

            If no matching costs are found, returns an ASR instance with:
            - Zero cost
            - Single expense year (project start year)
            - OIL as the cost_allocation
            - True in is_sunkcost (matching the include_sunkcost=True filter)

        Notes
        -----
        - This is a convenience wrapper around _get_filtered_asr() with:
          fluid_type=FluidType.OIL and include_sunkcost=True
        - All returned arrays will have the same length.
        - The dummy instance (when no costs exist) will include:
          * OIL in cost_allocation
          * True in is_sunkcost
        - See _get_filtered_asr() for more detailed filtering behavior.
        """
        return self._get_filtered_asr(fluid_type=FluidType.OIL, include_sunkcost=True)

    def _get_gas_asr_sunk_cost(self) -> ASR:
        """
        Get ASR (abandonment, site restoration, and post-operation) costs
        specifically for GAS, including only sunk costs.

        Returns
        -------
        ASR
            A new ASR instance containing only the costs that:
            1. Are allocated to GAS fluid type
            2. Are sunk costs (already incurred)

            If no matching costs are found, returns an ASR instance with:
            - Zero cost
            - Single expense year (project start year)
            - GAS as the cost_allocation
            - True in is_sunkcost (matching the include_sunkcost=True filter)

        Notes
        -----
        - This is a convenience wrapper around _get_filtered_asr() with:
          fluid_type=FluidType.GAS and include_sunkcost=True
        - All returned arrays will have the same length.
        - The dummy instance (when no costs exist) will include:
          * GAS in cost_allocation
          * True in is_sunkcost
        - See _get_filtered_asr() for more detailed filtering behavior.
        """
        return self._get_filtered_asr(fluid_type=FluidType.GAS, include_sunkcost=True)

    def _get_filtered_lbt(
        self,
        fluid_type: FluidType,
        include_sunkcost: bool,
    ) -> LBT:
        """
        Get land and building tax (LBT) costs filtered by fluid type and
        sunk cost inclusion.

        Parameters
        ----------
        fluid_type : FluidType
            The fluid type to filter LBT costs by.
        include_sunkcost : bool
            Whether to include sunk costs in the filtered results.
            If True, includes sunk costs; if False, excludes them.

        Returns
        -------
        LBT
            A new LBT instance containing only the costs that match:
            1. The specified fluid type
            2. The sunk cost inclusion criteria

            If no matching costs are found, returns an LBT instance with:
            - Zero cost
            - Single expense year (project start year)
            - The requested fluid type in cost_allocation
            - The requested sunk cost flag in is_sunkcost

        Notes
        -----
        - The method filters the data from self.lbt_cost_total based on the criteria.
        - All returned arrays (cost, expense_year, etc.) will have the same length.
        - The filtering preserves all original LBT attributes, but with only
          the matching elements in each array.
        - When no matching costs exist, the returned dummy instance will include:
          * The requested fluid_type in cost_allocation
          * The requested include_sunkcost value in is_sunkcost
        """

        # Handle case when requested fluid type is not found
        if fluid_type not in self.lbt_cost_total.cost_allocation:
            return LBT(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                expense_year=np.array([self.start_date.year]),
                cost=np.array([0]),
                cost_allocation=[fluid_type],
                is_sunkcost=[include_sunkcost],
            )

        else:
            # Configure indices to slice data according to fluid type and sunk cost
            allocation_array = np.array(self.lbt_cost_total.cost_allocation)
            sunkcost_array = np.array(self.lbt_cost_total.is_sunkcost)
            mask = np.logical_and(
                allocation_array == fluid_type,
                sunkcost_array == include_sunkcost,
            )
            indices = np.flatnonzero(mask)

            # Slice filtered data, return a new instance with filtered data
            return LBT(
                start_year=self.lbt_cost_total.start_year,
                end_year=self.lbt_cost_total.end_year,
                expense_year=self.lbt_cost_total.expense_year[indices],
                cost_allocation=np.array(self.lbt_cost_total.cost_allocation)[indices].tolist(),
                description=np.array(self.lbt_cost_total.description)[indices].tolist(),
                is_sunkcost=np.array(self.lbt_cost_total.is_sunkcost)[indices].tolist(),
                tax_portion=self.lbt_cost_total.tax_portion[indices],
                tax_discount=self.lbt_cost_total.tax_discount[indices],
                final_year=self.lbt_cost_total.final_year[indices],
                utilized_land_area=self.lbt_cost_total.utilized_land_area[indices],
                utilized_building_area=self.lbt_cost_total.utilized_building_area[indices],
                njop_land=self.lbt_cost_total.njop_land[indices],
                njop_building=self.lbt_cost_total.njop_building[indices],
                gross_revenue=self.lbt_cost_total.gross_revenue[indices],
                cost=self.lbt_cost_total.cost[indices],
            )

    def _get_oil_lbt(self) -> LBT:
        """
        Get LBT (land and building tax) costs specifically for OIL, excluding sunk costs.

        Returns
        -------
        LBT
            A new LBT instance containing only the costs that:
            1. Are allocated to OIL fluid type
            2. Are not sunk costs

            If no matching costs are found, returns an LBT instance with:
            - Zero cost
            - Single expense year (project start year)
            - OIL as the cost_allocation
            - False in is_sunkcost (matching the include_sunkcost=False filter)

        Notes
        -----
        - This is a convenience wrapper around _get_filtered_lbt() with:
          fluid_type=FluidType.OIL and include_sunkcost=False
        - All returned arrays will have the same length.
        - The dummy instance (when no costs exist) will include:
          * OIL in cost_allocation
          * False in is_sunkcost
        - See _get_filtered_lbt() for more detailed filtering behavior.
        """
        return self._get_filtered_lbt(fluid_type=FluidType.OIL, include_sunkcost=False)

    def _get_gas_lbt(self) -> LBT:
        """
        Get LBT (land and building tax) costs specifically for GAS, excluding sunk costs.

        Returns
        -------
        LBT
            A new LBT instance containing only the costs that:
            1. Are allocated to GAS fluid type
            2. Are not sunk costs

            If no matching costs are found, returns an LBT instance with:
            - Zero cost
            - Single expense year (project start year)
            - GAS as the cost_allocation
            - False in is_sunkcost (matching the include_sunkcost=False filter)

        Notes
        -----
        - This is a convenience wrapper around _get_filtered_lbt() with:
          fluid_type=FluidType.GAS and include_sunkcost=False
        - All returned arrays will have the same length.
        - The dummy instance (when no costs exist) will include:
          * GAS in cost_allocation
          * False in is_sunkcost
        - See _get_filtered_lbt() for more detailed filtering behavior.
        """
        return self._get_filtered_lbt(fluid_type=FluidType.GAS, include_sunkcost=False)

    def _get_oil_lbt_sunk_cost(self) -> LBT:
        """
        Get LBT (land and building tax) costs specifically for OIL, including sunk costs.

        Returns
        -------
        LBT
            A new LBT instance containing only the costs that:
            1. Are allocated to OIL fluid type
            2. Are sunk costs (include_sunkcost=True)

            If no matching costs are found, returns an LBT instance with:
            - Zero cost
            - Single expense year (project start year)
            - OIL as the cost_allocation
            - True in is_sunkcost (matching the include_sunkcost=True filter)

        Notes
        -----
        - This is a convenience wrapper around _get_filtered_lbt() with:
          fluid_type=FluidType.OIL and include_sunkcost=True
        - All returned arrays will have the same length.
        - The dummy instance (when no costs exist) will include:
          * OIL in cost_allocation
          * True in is_sunkcost
        - See _get_filtered_lbt() for more detailed filtering behavior.
        """
        return self._get_filtered_lbt(fluid_type=FluidType.OIL, include_sunkcost=True)

    def _get_gas_lbt_sunk_cost(self) -> LBT:
        """
        Get LBT (land and building tax) costs specifically for GAS, including sunk costs.

        Returns
        -------
        LBT
            A new LBT instance containing only the costs that:
            1. Are allocated to GAS fluid type
            2. Are sunk costs (include_sunkcost=True)

            If no matching costs are found, returns an LBT instance with:
            - Zero cost
            - Single expense year (project start year)
            - GAS as the cost_allocation
            - True in is_sunkcost (matching the include_sunkcost=True filter)

        Notes
        -----
        - This is a convenience wrapper around _get_filtered_lbt() with:
          fluid_type=FluidType.GAS and include_sunkcost=True
        - All returned arrays will have the same length.
        - The dummy instance (when no costs exist) will include:
          * GAS in cost_allocation
          * True in is_sunkcost
        - See _get_filtered_lbt() for more detailed filtering behavior.
        """
        return self._get_filtered_lbt(fluid_type=FluidType.GAS, include_sunkcost=True)

    def _get_filtered_cost_of_sales(
        self,
        fluid_type: FluidType,
        include_sunkcost: bool,
    ) -> CostOfSales:
        pass

    def _get_oil_cost_of_sales(self) -> CostOfSales:
        pass

    def _get_gas_cost_of_sales(self) -> CostOfSales:
        pass

    def _get_oil_cost_of_sales_sunk_cost(self) -> CostOfSales:
        pass

    def _get_gas_cost_of_sales_sunk_cost(self) -> CostOfSales:
        pass

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

    def _get_expenditures_post_tax(
        self,
        year_inflation: np.ndarray = None,
        inflation_rate: np.ndarray | int | float = 0.0,
        tax_rate: np.ndarray | float = 0.0,
        inflation_rate_applied_to: InflationAppliedTo | None = None,
    ) -> None:
        """
        Calculate and assign post-tax expenditures for various oil and gas expenditure
        categories.

        This method computes the post-tax expenditures for multiple categories related
        to oil and gas. It adjusts these expenditures by accounting for inflation,
        tax portions, tax rates, and tax discounts.

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
        inflation_rate_applied_to
            The selection of where the cost inflation will be applied to.

        Returns
        -------
        None
            This method does not return any values. It updates the following attributes with
            the calculated post-tax expenditures for oil and gas:

            -   `_oil_capital_expenditures_post_tax`
            -   `_gas_capital_expenditures_post_tax`
            -   `_oil_intangible_expenditures_post_tax`
            -   `_gas_intangible_expenditures_post_tax`
            -   `_oil_opex_expenditures_post_tax`
            -   `_gas_opex_expenditures_post_tax`
            -   `_oil_asr_expenditures_post_tax`
            -   `_gas_asr_expenditures_post_tax`
            -   `_oil_lbt_expenditures_post_tax`
            -   `_gas_lbt_expenditures_post_tax`
            -   `_oil_cost_of_sales_expenditures_post_tax`
            -   `_gas_cost_of_sales_expenditures_post_tax`
        """

        # Prepare expenditures post tax associated with capital, intangible,
        # opex, asr, and lbt costs
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
        ) = [
            self._calc_expenditures(
                target_attr=attr,
                expenditures_type=ExpendituresType.POST_TAX,
                year_inflation=year_inflation,
                inflation_rate=inflation_rate,
                inflation_rate_applied_to=inflation_rate_applied_to,
                tax_rate=tax_rate,
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

        # Prepare expenditures post tax associated with cost of sales
        (
            self._oil_cost_of_sales_expenditures_post_tax,
            self._gas_cost_of_sales_expenditures_post_tax,
        ) = [
            attr.expenditures_post_tax()
            for attr in [
                self._oil_cost_of_sales,
                self._gas_cost_of_sales,
            ]
        ]

    def _get_tax_by_regime(self, tax_regime) -> np.ndarray:
        """
        Determine the tax rate array based on the tax regime and project years.

        This method computes the tax rates for the project years depending on the specified
        `tax_regime`. It uses predefined tax configurations for certain years (2013, 2016, 2020)
        and allows for specific tax regimes that override these configurations. For tax regimes
        like `NAILED_DOWN`, the tax rate is fixed based on the start year of the project.

        Parameters
        ----------
        tax_regime : TaxRegime
            The tax regime to be applied. It determines how tax rates are selected and can be
            one of the following:
            -   `TaxRegime.UU_07_2021`
            -   `TaxRegime.UU_02_2020`
            -   `TaxRegime.UU_36_2008`
            -   `TaxRegime.NAILED_DOWN`

        Returns
        -------
        tax_rate_arr : np.ndarray
            A 1D array of tax rates for each project year. The tax rate is determined based on
            the tax regime and the project's starting year in relation to the predefined tax
            configurations.
        """
        tax_config = {
            2013: 0.44,
            2016: 0.42,
            2020: 0.40
        }

        tax_rate_arr = np.full_like(
            self.project_years, fill_value=tax_config[min(tax_config)], dtype=float
        )

        for year in tax_config:
            indices = np.array(np.where(self.project_years >= year)).ravel()
            tax_rate_arr[indices] = tax_config[year]

        if tax_regime == TaxRegime.UU_07_2021:
            tax_rate_arr = np.full_like(self.project_years, fill_value=0.40, dtype=float)

        if tax_regime == TaxRegime.UU_02_2020:
            tax_rate_arr = np.full_like(self.project_years, fill_value=0.42, dtype=float)

        if tax_regime == TaxRegime.UU_36_2008:
            tax_rate_arr = np.full_like(self.project_years, fill_value=0.44, dtype=float)

        if tax_regime == TaxRegime.NAILED_DOWN:
            if self.start_date.year >= max(tax_config):
                tax_rate_arr = np.full_like(
                    self.project_years, fill_value=tax_config[max(tax_config)], dtype=float
                )
            else:
                tax_rate_arr = np.full_like(
                    self.project_years, fill_value=tax_config[min(tax_config)], dtype=float
                )

        return tax_rate_arr

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

    def _get_consolidated_profile(self, is_dict: bool, oil_param, gas_param):
        """
        Combine oil and gas parameters into consolidated profiles.

        Creates consolidated results by either summing dictionary values or adding
        array-like objects, depending on the input type.

        Parameters
        ----------
        is_dict : bool
            Flag indicating whether inputs are dictionaries (True) or array-like (False).
        oil_param : dict or array-like
            Oil-related parameters. Either:
            - Dictionary (when is_dict=True) with keys matching self._investment_type_list
            - Array-like object supporting addition (when is_dict=False)
        gas_param : dict or array-like
            Gas-related parameters. Must be same type as oil_param.

        Returns
        -------
        dict or array-like
            Consolidated results matching input type:
            - If is_dict=True: Dictionary with summed values for each investment type
            - If is_dict=False: Sum of oil_param and gas_param

        Notes
        -----
        - For dictionary inputs: Only keys in self._investment_type_list are processed
        - For array inputs: Both parameters must support the + operator
        - Input types must be consistent (both dict or both array-like)
        - Typically used to combine oil and gas cost profiles or amortization charges
        """
        if is_dict is True:
            return {
                key: oil_param[key] + gas_param[key] for key in self._investment_type_list
            }

        else:
            return oil_param + gas_param

    def run(
        self,
        prod_year: np.ndarray,
        prod: np.ndarray,
        tax_rate: np.ndarray | float = 0.0,
        # depr_method: DeprMethod = DeprMethod.PSC_DB,
        decline_factor: float | int = 2,
        salvage_value: float = 0.0,
        year_inflation: np.ndarray = None,
        inflation_rate: np.ndarray | float = 0.0,
        inflation_rate_applied_to: InflationAppliedTo = None,
        sulfur_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_GAS_REVENUE,
        electricity_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_OIL_REVENUE,
        co2_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_GAS_REVENUE,
    ) -> None:
        """
        Execute complete economic model calculation pipeline.

        This orchestrates the full economic calculation sequence including:
        - Cost amortization and depreciation
        - Expenditure calculations (pre-tax and post-tax)
        - Revenue and cashflow computations
        - Consolidated oil and gas results

        Parameters
        ----------
        prod_year : np.ndarray
            Array of production years for economic calculations.
        prod : np.ndarray
            Array of production volumes corresponding to prod_year.
        tax_rate : np.ndarray or float, optional
            Tax rate(s) to apply. Can be single value or time-varying array.
            Default is 0.0 (no tax).
        depr_method : DeprMethod, optional
            Depreciation method for tangible assets.
            Default is DeprMethod.PSC_DB (PSC declining balance).
        decline_factor : float or int, optional
            Decline factor for depreciation calculations.
            Default is 2 (double declining balance).
        salvage_value : float, optional
            Residual value of assets at end of amortization period.
            Default is 0.0.
        year_inflation : np.ndarray, optional
            Array of inflation adjustment years. If None, no inflation adjustment.
            Default is None.
        inflation_rate : np.ndarray or float, optional
            Inflation rate(s) for cost adjustments.
            Default is 0.0 (no inflation).
        inflation_rate_applied_to : InflationAppliedTo, optional
            Specifies which costs to apply inflation to.
            Default is None (no inflation application).
        sulfur_revenue : OtherRevenue, optional
            Treatment of sulfur byproduct revenue.
            Default is OtherRevenue.ADDITION_TO_GAS_REVENUE.
        electricity_revenue : OtherRevenue, optional
            Treatment of electricity byproduct revenue.
            Default is OtherRevenue.ADDITION_TO_OIL_REVENUE.
        co2_revenue : OtherRevenue, optional
            Treatment of CO2 byproduct revenue.
            Default is OtherRevenue.ADDITION_TO_GAS_REVENUE.

        Returns
        -------
        None
            Results are stored in numerous instance variables including:
            - Cost arrays and bulk values (_oil/gas_sunk/preonstream_cost_*)
            - Amortization and depreciation charges
            - Expenditure components (capital, opex, ASR, LBT, etc.)
            - Revenue and cashflow calculations
            - Consolidated oil+gas results (_consolidated_*)
        """
        # # Prepare several attributes associated with sunk cost and preonstream cost
        # self.fit_sunk_preonstream_cost(
        #     prod_year=prod_year,
        #     prod=prod,
        #     tax_rate=tax_rate,
        #     depr_method=depr_method,
        #     decline_factor=decline_factor,
        #     salvage_value=salvage_value,
        # )

        # Calculate pre tax expenditures
        self._get_expenditures_pre_tax(
            year_inflation=year_inflation,
            inflation_rate=inflation_rate,
            inflation_rate_applied_to=inflation_rate_applied_to,
        )

        # Calculate indirect taxes
        self._get_indirect_taxes(tax_rate=tax_rate)

        # Calculate post tax expenditures
        self._get_expenditures_post_tax(
            year_inflation=year_inflation,
            inflation_rate=inflation_rate,
            tax_rate=tax_rate,
            inflation_rate_applied_to=inflation_rate_applied_to,
        )

        # Non-capital costs (intangible + opex + asr + lbt)
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

        # WAP (Weighted Average Price) for each produced fluid
        self._get_wap_price()

        # Other revenue
        self._get_other_revenue(
            sulfur_revenue=sulfur_revenue,
            electricity_revenue=electricity_revenue,
            co2_revenue=co2_revenue,
        )

        # Total pre-tax expenditures for OIL and GAS
        self._oil_total_expenditures_pre_tax = (
            self._oil_capital_expenditures_pre_tax
            + self._oil_intangible_expenditures_pre_tax
            + self._oil_opex_expenditures_pre_tax
            + self._oil_asr_expenditures_pre_tax
            + self._oil_lbt_expenditures_pre_tax
            + self._oil_cost_of_sales_expenditures_pre_tax
        )

        self._gas_total_expenditures_pre_tax = (
            self._gas_capital_expenditures_pre_tax
            + self._gas_intangible_expenditures_pre_tax
            + self._gas_opex_expenditures_pre_tax
            + self._gas_asr_expenditures_pre_tax
            + self._gas_lbt_expenditures_pre_tax
            + self._gas_cost_of_sales_expenditures_pre_tax
        )

        # Total indirect taxes for OIL and GAS
        self._oil_total_indirect_tax = (
            self._oil_capital_indirect_tax
            + self._oil_intangible_indirect_tax
            + self._oil_opex_indirect_tax
            + self._oil_asr_indirect_tax
            + self._oil_lbt_indirect_tax
            + self._oil_cost_of_sales_indirect_tax
        )

        self._gas_total_indirect_tax = (
            self._gas_capital_indirect_tax
            + self._gas_intangible_indirect_tax
            + self._gas_opex_indirect_tax
            + self._gas_asr_indirect_tax
            + self._gas_lbt_indirect_tax
            + self._gas_cost_of_sales_indirect_tax
        )

        # Total post-tax expenditures for OIL and GAS
        self._oil_total_expenditures_post_tax = (
            self._oil_total_expenditures_pre_tax + self._oil_total_indirect_tax
        )

        self._gas_total_expenditures_post_tax = (
            self._gas_total_expenditures_pre_tax + self._gas_total_indirect_tax
        )

        # Configure base cashflow for OIL and GAS
        self._oil_cashflow = (
            self._oil_revenue -
            (
                self._oil_capital_expenditures_post_tax
                + self._oil_intangible_expenditures_post_tax
                + self._oil_opex_expenditures_post_tax
                + self._oil_asr_expenditures_post_tax
                + self._oil_lbt_expenditures_post_tax
                + self._oil_cost_of_sales_expenditures_post_tax
            )
        )

        self._gas_cashflow = (
            self._gas_revenue -
            (
                self._gas_capital_expenditures_post_tax
                + self._gas_intangible_expenditures_post_tax
                + self._gas_opex_expenditures_post_tax
                + self._gas_asr_expenditures_post_tax
                + self._gas_lbt_expenditures_post_tax
                + self._gas_cost_of_sales_expenditures_post_tax
            )
        )

        # Prepare attributes associated with consolidated sunk cost and preonstream cost
        (
            self._consolidated_sunk_cost_array,
            self._consolidated_preonstream_cost_array,
            self._consolidated_sunk_cost_bulk,
            self._consolidated_preonstream_cost_bulk,
            self._consolidated_sunk_cost_amortization_charge,
            self._consolidated_preonstream_cost_amortization_charge
        ) = [
            self._get_consolidated_profile(is_dict=True, oil_param=i, gas_param=j)
            for i, j, in zip(
                [
                    self._oil_sunk_cost_array,
                    self._oil_preonstream_cost_array,
                    self._oil_sunk_cost_bulk,
                    self._oil_preonstream_cost_bulk,
                    self._oil_sunk_cost_amortization_charge,
                    self._oil_preonstream_cost_amortization_charge
                ],
                [
                    self._gas_sunk_cost_array,
                    self._gas_preonstream_cost_array,
                    self._gas_sunk_cost_bulk,
                    self._gas_preonstream_cost_bulk,
                    self._gas_sunk_cost_amortization_charge,
                    self._gas_preonstream_cost_amortization_charge
                ]
            )
        ]

        (
            self._consolidated_sunk_cost_tangible_depreciation_charge,
            self._consolidated_preonstream_cost_tangible_depreciation_charge,
            self._consolidated_sunk_cost_tangible_undepreciated_asset,
            self._consolidated_preonstream_cost_tangible_undepreciated_asset
        ) = [
            self._get_consolidated_profile(is_dict=False, oil_param=i, gas_param=j)
            for i, j, in zip(
                [
                    self._oil_sunk_cost_tangible_depreciation_charge,
                    self._oil_preonstream_cost_tangible_depreciation_charge,
                    self._oil_sunk_cost_tangible_undepreciated_asset,
                    self._oil_preonstream_cost_tangible_undepreciated_asset
                ],
                [
                    self._gas_sunk_cost_tangible_depreciation_charge,
                    self._gas_preonstream_cost_tangible_depreciation_charge,
                    self._gas_sunk_cost_tangible_undepreciated_asset,
                    self._gas_preonstream_cost_tangible_undepreciated_asset
                ]
            )
        ]

        # Prepare consolidated profiles
        self._consolidated_revenue = self._oil_revenue + self._gas_revenue
        self._consolidated_cashflow = self._oil_cashflow + self._gas_cashflow
        self._consolidated_government_take = np.zeros_like(self._consolidated_cashflow)

    def __len__(self):
        return self.project_duration

    def __eq__(self, other):
        # Between two instances of BaseProject
        if isinstance(other, BaseProject):
            revenue_self = np.sum(self._oil_revenue + self._gas_revenue)
            revenue_other = np.sum(other._oil_revenue + other._gas_revenue)
            capital_self = sum(self._oil_capital_cost.cost) + sum(self._gas_capital_cost.cost)
            capital_other = sum(other._oil_capital_cost.cost) + sum(other._gas_capital_cost.cost)
            intangible_self = sum(self._oil_intangible.cost) + sum(self._gas_intangible.cost)
            intangible_other = sum(other._oil_intangible.cost) + sum(other._gas_intangible.cost)
            opex_self = sum(self._oil_opex.cost) + sum(self._gas_opex.cost)
            opex_other = sum(other._oil_opex.cost) + sum(other._gas_opex.cost)
            asr_self = sum(self._oil_asr.cost) + sum(self._gas_asr.cost)
            asr_other = sum(other._oil_asr.cost) + sum(other._gas_asr.cost)
            lbt_self = sum(self._oil_lbt.cost) + sum(self._gas_lbt.cost)
            lbt_other = sum(other._oil_lbt.cost) + sum(other._gas_lbt.cost)
            cos_self = sum(self._oil_cost_of_sales.cost) + sum(self._gas_cost_of_sales.cost)
            cos_other = sum(other._oil_cost_of_sales.cost) + sum(other._gas_cost_of_sales.cost)
            sc_self = sum(self._oil_sunk_cost.cost) + sum(self._gas_sunk_cost.cost)
            sc_other = sum(other._oil_sunk_cost.cost) + sum(other._gas_sunk_cost.cost)

            return all(
                (
                    self.start_date.year == other.start_date.year,
                    self.start_date.month == other.start_date.month,
                    self.start_date.day == other.start_date.day,
                    self.end_date.year == other.end_date.year,
                    self.end_date.month == other.end_date.month,
                    self.end_date.day == other.end_date.day,
                    self.oil_onstream_date.year == other.oil_onstream_date.year,
                    self.oil_onstream_date.month == other.oil_onstream_date.month,
                    self.oil_onstream_date.day == other.oil_onstream_date.day,
                    self.gas_onstream_date.year == other.gas_onstream_date.year,
                    self.gas_onstream_date.month == other.gas_onstream_date.month,
                    self.gas_onstream_date.day == other.gas_onstream_date.day,
                    revenue_self == revenue_other,
                    capital_self == capital_other,
                    intangible_self == intangible_other,
                    opex_self == opex_other,
                    asr_self == asr_other,
                    lbt_self == lbt_other,
                    cos_self == cos_other,
                    sc_self == sc_other,
                )
            )

        else:
            return False

    def __lt__(self, other):
        # Between an instance of BaseProject with another instance of BaseProject
        if isinstance(other, BaseProject):
            revenue_self = np.sum(self._oil_revenue + self._gas_revenue)
            revenue_other = np.sum(other._oil_revenue + other._gas_revenue)
            tangible_self = sum(self._oil_capital_cost.cost) + sum(self._gas_capital_cost.cost)
            tangible_other = sum(other._oil_capital_cost.cost) + sum(other._gas_capital_cost.cost)
            intangible_self = sum(self._oil_intangible.cost) + sum(self._gas_intangible.cost)
            intangible_other = sum(other._oil_intangible.cost) + sum(other._gas_intangible.cost)
            opex_self = sum(self._oil_opex.cost) + sum(self._gas_opex.cost)
            opex_other = sum(other._oil_opex.cost) + sum(other._gas_opex.cost)
            asr_self = sum(self._oil_asr.cost) + sum(self._gas_asr.cost)
            asr_other = sum(other._oil_asr.cost) + sum(other._gas_asr.cost)
            lbt_self = sum(self._oil_lbt.cost) + sum(self._gas_lbt.cost)
            lbt_other = sum(other._oil_lbt.cost) + sum(other._gas_lbt.cost)
            cos_self = sum(self._oil_cost_of_sales.cost) + sum(self._gas_cost_of_sales.cost)
            cos_other = sum(other._oil_cost_of_sales.cost) + sum(other._gas_cost_of_sales.cost)
            sc_self = sum(self._oil_sunk_cost.cost) + sum(self._gas_sunk_cost.cost)
            sc_other = sum(other._oil_sunk_cost.cost) + sum(other._gas_sunk_cost.cost)

            expense_self = reduce(
                lambda x, y: x + y,
                [
                    tangible_self, intangible_self, opex_self, asr_self,
                    lbt_self, cos_self, sc_self
                ]
            )
            expense_other = reduce(
                lambda x, y: x + y,
                [
                    tangible_other, intangible_other, opex_other, asr_other,
                    lbt_other, cos_other, sc_other
                ]
            )

            return all(
                (
                    revenue_self < revenue_other,
                    expense_self < expense_other,
                )
            )

        else:
            raise BaseProjectException(
                f"Must compare an instance of BaseProject and another instance "
                f"of BaseProject. {other}({other.__class__.__qualname__}) is "
                f"not an instance of BaseProject."
            )

    def __le__(self, other):
        # Between an instance of BaseProject with another instance of BaseProject
        if isinstance(other, BaseProject):
            revenue_self = np.sum(self._oil_revenue + self._gas_revenue)
            revenue_other = np.sum(other._oil_revenue + other._gas_revenue)
            tangible_self = sum(self._oil_capital_cost.cost) + sum(self._gas_capital_cost.cost)
            tangible_other = sum(other._oil_capital_cost.cost) + sum(other._gas_capital_cost.cost)
            intangible_self = sum(self._oil_intangible.cost) + sum(self._gas_intangible.cost)
            intangible_other = sum(other._oil_intangible.cost) + sum(other._gas_intangible.cost)
            opex_self = sum(self._oil_opex.cost) + sum(self._gas_opex.cost)
            opex_other = sum(other._oil_opex.cost) + sum(other._gas_opex.cost)
            asr_self = sum(self._oil_asr.cost) + sum(self._gas_asr.cost)
            asr_other = sum(other._oil_asr.cost) + sum(other._gas_asr.cost)
            lbt_self = sum(self._oil_lbt.cost) + sum(self._gas_lbt.cost)
            lbt_other = sum(other._oil_lbt.cost) + sum(other._gas_lbt.cost)
            cos_self = sum(self._oil_cost_of_sales.cost) + sum(self._gas_cost_of_sales.cost)
            cos_other = sum(other._oil_cost_of_sales.cost) + sum(other._gas_cost_of_sales.cost)
            sc_self = sum(self._oil_sunk_cost.cost) + sum(self._gas_sunk_cost.cost)
            sc_other = sum(other._oil_sunk_cost.cost) + sum(other._gas_sunk_cost.cost)

            expense_self = reduce(
                lambda x, y: x + y,
                [
                    tangible_self, intangible_self, opex_self, asr_self,
                    lbt_self, cos_self, sc_self
                ]
            )
            expense_other = reduce(
                lambda x, y: x + y,
                [
                    tangible_other, intangible_other, opex_other, asr_other,
                    lbt_other, cos_other, sc_other
                ]
            )

            return all(
                (
                    revenue_self <= revenue_other,
                    expense_self <= expense_other,
                )
            )

        else:
            raise BaseProjectException(
                f"Must compare an instance of BaseProject and another instance "
                f"of BaseProject. {other}({other.__class__.__qualname__}) is "
                f"not an instance of BaseProject."
            )

    def __gt__(self, other):
        # Between an instance of BaseProject with another instance of BaseProject
        if isinstance(other, BaseProject):
            revenue_self = np.sum(self._oil_revenue + self._gas_revenue)
            revenue_other = np.sum(other._oil_revenue + other._gas_revenue)
            tangible_self = sum(self._oil_capital_cost.cost) + sum(self._gas_capital_cost.cost)
            tangible_other = sum(other._oil_capital_cost.cost) + sum(other._gas_capital_cost.cost)
            intangible_self = sum(self._oil_intangible.cost) + sum(self._gas_intangible.cost)
            intangible_other = sum(other._oil_intangible.cost) + sum(other._gas_intangible.cost)
            opex_self = sum(self._oil_opex.cost) + sum(self._gas_opex.cost)
            opex_other = sum(other._oil_opex.cost) + sum(other._gas_opex.cost)
            asr_self = sum(self._oil_asr.cost) + sum(self._gas_asr.cost)
            asr_other = sum(other._oil_asr.cost) + sum(other._gas_asr.cost)
            lbt_self = sum(self._oil_lbt.cost) + sum(self._gas_lbt.cost)
            lbt_other = sum(other._oil_lbt.cost) + sum(other._gas_lbt.cost)
            cos_self = sum(self._oil_cost_of_sales.cost) + sum(self._gas_cost_of_sales.cost)
            cos_other = sum(other._oil_cost_of_sales.cost) + sum(other._gas_cost_of_sales.cost)
            sc_self = sum(self._oil_sunk_cost.cost) + sum(self._gas_sunk_cost.cost)
            sc_other = sum(other._oil_sunk_cost.cost) + sum(other._gas_sunk_cost.cost)

            expense_self = reduce(
                lambda x, y: x + y,
                [
                    tangible_self, intangible_self, opex_self, asr_self,
                    lbt_self, cos_self, sc_self
                ]
            )
            expense_other = reduce(
                lambda x, y: x + y,
                [
                    tangible_other, intangible_other, opex_other, asr_other,
                    lbt_other, cos_other, sc_other
                ]
            )

            return all(
                (
                    revenue_self > revenue_other,
                    expense_self > expense_other,
                )
            )

        else:
            raise BaseProjectException(
                f"Must compare an instance of BaseProject and another instance "
                f"of BaseProject. {other}({other.__class__.__qualname__}) is "
                f"not an instance of BaseProject."
            )

    def __ge__(self, other):
        # Between an instance of BaseProject with another instance of BaseProject
        if isinstance(other, BaseProject):
            revenue_self = np.sum(self._oil_revenue + self._gas_revenue)
            revenue_other = np.sum(other._oil_revenue + other._gas_revenue)
            tangible_self = sum(self._oil_capital_cost.cost) + sum(self._gas_capital_cost.cost)
            tangible_other = sum(other._oil_capital_cost.cost) + sum(other._gas_capital_cost.cost)
            intangible_self = sum(self._oil_intangible.cost) + sum(self._gas_intangible.cost)
            intangible_other = sum(other._oil_intangible.cost) + sum(other._gas_intangible.cost)
            opex_self = sum(self._oil_opex.cost) + sum(self._gas_opex.cost)
            opex_other = sum(other._oil_opex.cost) + sum(other._gas_opex.cost)
            asr_self = sum(self._oil_asr.cost) + sum(self._gas_asr.cost)
            asr_other = sum(other._oil_asr.cost) + sum(other._gas_asr.cost)
            lbt_self = sum(self._oil_lbt.cost) + sum(self._gas_lbt.cost)
            lbt_other = sum(other._oil_lbt.cost) + sum(other._gas_lbt.cost)
            cos_self = sum(self._oil_cost_of_sales.cost) + sum(self._gas_cost_of_sales.cost)
            cos_other = sum(other._oil_cost_of_sales.cost) + sum(other._gas_cost_of_sales.cost)
            sc_self = sum(self._oil_sunk_cost.cost) + sum(self._gas_sunk_cost.cost)
            sc_other = sum(other._oil_sunk_cost.cost) + sum(other._gas_sunk_cost.cost)

            expense_self = reduce(
                lambda x, y: x + y,
                [
                    tangible_self, intangible_self, opex_self, asr_self,
                    lbt_self, cos_self, sc_self
                ]
            )
            expense_other = reduce(
                lambda x, y: x + y,
                [
                    tangible_other, intangible_other, opex_other, asr_other,
                    lbt_other, cos_other, sc_other
                ]
            )

            return all(
                (
                    revenue_self >= revenue_other,
                    expense_self >= expense_other,
                )
            )

        else:
            raise BaseProjectException(
                f"Must compare an instance of BaseProject and another instance "
                f"of BaseProject. {other}({other.__class__.__qualname__}) is "
                f"not an instance of BaseProject."
            )
