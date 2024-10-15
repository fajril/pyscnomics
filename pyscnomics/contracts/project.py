"""
Configure base project as a base framework for contract.
"""

from dataclasses import dataclass, field
from datetime import date
from functools import reduce
import numpy as np

from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.selection import (
    FluidType,
    TaxType,
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


class SunkCostException(Exception):
    """Exception to raise for a misuse of Sunk Cost Method"""

    pass


class BaseProjectException(Exception):
    """Exception to raise for a misuse of BaseProject class"""

    pass


class OtherRevenueException(Exception):
    """Exception to raise for a misuse of Other Revenue"""

    pass


@dataclass
class BaseProject:
    """
    Represents a base project with start and end dates, lifting information,
    capital and intangible costs, operational expenses (OPEX), and ASR costs.

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
    lifting: tuple[Lifting] = field(default=None)
    capital_cost: tuple[CapitalCost] = field(default=None)
    intangible_cost: tuple[Intangible] = field(default=None)
    opex: tuple[OPEX] = field(default=None)
    asr_cost: tuple[ASR] = field(default=None)
    lbt_cost: tuple[LBT] = field(default=None)
    cost_of_sales: tuple[CostOfSales] = field(default=None)

    # Attributes to be defined later (associated with project duration)
    project_duration: int = field(default=None, init=False)
    project_years: np.ndarray = field(default=None, init=False)

    # Attributes to be defined later (associated with total cost per component)
    capital_cost_total: CapitalCost = field(default=None, init=False, repr=False)
    intangible_cost_total: Intangible = field(default=None, init=False, repr=False)
    opex_total: OPEX = field(default=None, init=False, repr=False)
    asr_cost_total: ASR = field(default=None, init=False, repr=False)
    lbt_cost_total: LBT = field(default=None, init=False, repr=False)
    cost_of_sales_total: CostOfSales = field(default=None, init=False, repr=False)

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
    _consolidated_revenue: np.ndarray = field(default=None, init=False, repr=False)

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

    # Attributes to be defined later (associated with pre tax expenditures for each cost elements)
    _oil_capital_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_capital_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    _oil_intangible_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_intangible_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    _oil_opex_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_opex_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    _oil_asr_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_asr_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    _oil_lbt_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_lbt_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    _oil_cost_of_sales_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_cost_of_sales_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes to be defined later (associated with indirect taxes for each cost element)
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

    # Attributes to be defined later (associated with post tax expenditures for each cost elements)
    _oil_capital_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_capital_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    _oil_intangible_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_intangible_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    _oil_opex_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_opex_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    _oil_asr_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_asr_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    _oil_lbt_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_lbt_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    _oil_cost_of_sales_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_cost_of_sales_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes to be defined later (associated with sunk cost)
    _oil_sunk_cost: np.ndarray = field(default=None, init=False, repr=False)
    _gas_sunk_cost: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes to be defined later (associated with cashflow)
    _oil_cashflow: np.ndarray = field(default=None, init=False, repr=False)
    _gas_cashflow: np.ndarray = field(default=None, init=False, repr=False)

    # Atributes to be defined later (associated with WAP price)
    _oil_wap_price: np.ndarray = field(default=None, init=False, repr=False)
    _gas_wap_price: np.ndarray = field(default=None, init=False, repr=False)
    _sulfur_wap_price: np.ndarray = field(default=None, init=False, repr=False)
    _electricity_wap_price: np.ndarray = field(default=None, init=False, repr=False)
    _co2_wap_price: np.ndarray = field(default=None, init=False, repr=False)

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

        # Prepare attribute lifting (for all FluidType)
        if self.lifting is None:
            self.lifting = (
                Lifting(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    lifting_rate=np.zeros(self.project_duration),
                    price=np.zeros(self.project_duration),
                    prod_year=self.project_years,
                    fluid_type=FluidType.OIL,
                ),
                Lifting(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    lifting_rate=np.zeros(self.project_duration),
                    price=np.zeros(self.project_duration),
                    prod_year=self.project_years,
                    fluid_type=FluidType.GAS,
                ),
                Lifting(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    lifting_rate=np.zeros(self.project_duration),
                    price=np.zeros(self.project_duration),
                    prod_year=self.project_years,
                    fluid_type=FluidType.SULFUR,
                ),
                Lifting(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    lifting_rate=np.zeros(self.project_duration),
                    price=np.zeros(self.project_duration),
                    prod_year=self.project_years,
                    fluid_type=FluidType.ELECTRICITY,
                ),
                Lifting(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    lifting_rate=np.zeros(self.project_duration),
                    price=np.zeros(self.project_duration),
                    prod_year=self.project_years,
                    fluid_type=FluidType.CO2,
                )
            )

        # Prepare attribute capital_cost (for both OIL and GAS)
        if self.capital_cost is None:
            self.capital_cost = (
                CapitalCost(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    cost=np.array([0]),
                    expense_year=np.array([self.start_date.year]),
                    cost_allocation=[FluidType.OIL],
                ),
                CapitalCost(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    cost=np.array([0]),
                    expense_year=np.array([self.start_date.year]),
                    cost_allocation=[FluidType.GAS],
                )
            )

        # Prepare attribute intangible_cost (for both OIL and GAS)
        if self.intangible_cost is None:
            self.intangible_cost = (
                Intangible(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    cost=np.array([0]),
                    expense_year=np.array([self.start_date.year]),
                    cost_allocation=[FluidType.OIL],
                ),
                Intangible(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    cost=np.array([0]),
                    expense_year=np.array([self.start_date.year]),
                    cost_allocation=[FluidType.GAS],
                ),
            )

        # Prepare attribute opex (for both OIL and GAS)
        if self.opex is None:
            self.opex = (
                OPEX(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    fixed_cost=np.array([0]),
                    expense_year=np.array([self.start_date.year]),
                    cost_allocation=[FluidType.OIL],
                ),
                OPEX(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    fixed_cost=np.array([0]),
                    expense_year=np.array([self.start_date.year]),
                    cost_allocation=[FluidType.GAS],
                ),
            )

        # Prepare attribute asr_cost (for both OIL and GAS)
        if self.asr_cost is None:
            self.asr_cost = (
                ASR(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    cost=np.array([0]),
                    expense_year=np.array([self.start_date.year]),
                    cost_allocation=[FluidType.OIL],
                ),
                ASR(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    cost=np.array([0]),
                    expense_year=np.array([self.start_date.year]),
                    cost_allocation=[FluidType.GAS],
                ),
            )

        # Prepare attribute lbt_cost (for both OIL and GAS)
        if self.lbt_cost is None:
            self.lbt_cost = (
                LBT(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    cost=np.array([0]),
                    expense_year=np.array([self.start_date.year]),
                    cost_allocation=[FluidType.OIL],
                ),
                LBT(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    cost=np.array([0]),
                    expense_year=np.array([self.start_date.year]),
                    cost_allocation=[FluidType.GAS],
                ),
            )

        # Prepare attribute cost_of_sales (for both OIL and GAS)
        if self.cost_of_sales is None:
            self.cost_of_sales = (
                CostOfSales(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    cost=np.zeros(self.project_duration, dtype=np.float64),
                    expense_year=self.project_years.copy(),
                    cost_allocation=[FluidType.OIL for _ in range(self.project_duration)],
                ),
                CostOfSales(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    cost=np.zeros(self.project_duration, dtype=np.float64),
                    expense_year=self.project_years.copy(),
                    cost_allocation=[FluidType.GAS for _ in range(self.project_duration)],
                ),
            )

        # Prepare attributes associated with total cost per component
        self.capital_cost_total = reduce(lambda x, y: x + y, self.capital_cost)
        self.intangible_cost_total = reduce(lambda x, y: x + y, self.intangible_cost)
        self.opex_total = reduce(lambda x, y: x + y, self.opex)
        self.asr_cost_total = reduce(lambda x, y: x + y, self.asr_cost)
        self.lbt_cost_total = reduce(lambda x, y: x + y, self.lbt_cost)
        self.cost_of_sales_total = reduce(lambda x, y: x + y, self.cost_of_sales)

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
                f"Gas cost of sales ({self._gas_cost_of_sales.start_year})."
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
                f"Gas cost of sales ({self._gas_cost_of_sales.end_year})."
            )

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

                oil_onstream_index = int(
                    np.argwhere(
                        self.oil_onstream_date.year == self.project_years
                    ).ravel()
                )

                if oil_onstream_index != oil_revenue_index[0]:
                    raise BaseProjectException(
                        f"Oil onstream year ({self.oil_onstream_date.year}) is different from "
                        f"the starting year of oil production ({self.project_years[oil_revenue_index[0]]})"
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

                gas_onstream_index = int(
                    np.argwhere(
                        self.gas_onstream_date.year == self.project_years
                    ).ravel()
                )

                if gas_onstream_index != gas_revenue_index[0]:
                    raise BaseProjectException(
                        f"Gas onstream year ({self.gas_onstream_date.year}) is different from "
                        f"the starting year of gas production ({self.project_years[gas_revenue_index[0]]})"
                    )

            else:
                self.gas_onstream_date = date(
                    year=self.project_years[gas_revenue_index[0]], month=1, day=1
                )

        else:
            if self.gas_onstream_date is not None:
                raise BaseProjectException(
                    f"Gas onstream year is given ({self.gas_onstream_date.year}) "
                    f"but gas lifting rate is missing or zero for the entire project duration"
                )

            else:
                self.gas_onstream_date = self.end_date

    def _get_oil_lifting(self) -> Lifting:
        """
        Determines total oil Lifting from the number of oil Lifting instances in
        attribute self.lifting.

        Return
        ------
        total_oil_lifting: Lifting
            Total oil lifting as a new instance of Lifting class where the associated
            atrributes are determined based on the prescribed rule in the corresponding
            dunder method __add__() of the Lifting class.

        Notes
        -----
        The order of operations is as follows:
        (1) Check the fluid_type in attribute self.lifting,
        (2) If OIL is not available as an instance in attribute self.lifting, then
            establish a new instance of OIL Lifting with the following attributes set
            to zero: lifting_rate, price.
        (3) Add the instances with fluid_type OIL, return the result as a new instance of
            Lifting class following the rule prescribed in the dunder method __add__() of
            class Lifting.
        """

        fluid_types = [lft.fluid_type for lft in self.lifting]

        if FluidType.OIL not in fluid_types:
            return Lifting(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                lifting_rate=np.zeros(self.project_duration),
                price=np.zeros(self.project_duration),
                prod_year=self.project_years,
                fluid_type=FluidType.OIL,
                prod_rate=np.zeros(self.project_duration),
                prod_rate_baseline=np.zeros(self.project_duration),
            )

        return reduce(
            lambda x, y: x + y,
            (lft for lft in self.lifting if lft.fluid_type == FluidType.OIL),
        )

    def _get_gas_lifting(self) -> Lifting:
        """
        Determines total gas Lifting from the number of gas Lifting instances in
        attribute self.lifting.

        Returns
        -------
        total_gas_lifting: Lifting
            Total gas Lifting as a new instance of Lifting class where the associated
            attributes are determined based on the prescribed rule in the corresponding
            dunder method __add__() of the Lifting class.

        Notes
        -----
        The order of operations is as follows:
        (1) Check the fluid_type in attribute self.lifting,
        (2) If GAS is not available as an instance in attribute self.lifting, then
            establish a new instance of GAS Lifting with the following attributes set
            to zero: lifting_rate, price.
        (3) Add the instances with fluid_type GAS, return the result as a new instance of
            Lifting class following the rule prescribed in the dunder method __add__() of
            class Lifting.
        """

        fluid_types = [lft.fluid_type for lft in self.lifting]

        if FluidType.GAS not in fluid_types:
            return Lifting(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                lifting_rate=np.zeros(self.project_duration),
                price=np.zeros(self.project_duration),
                prod_year=self.project_years,
                fluid_type=FluidType.GAS,
                ghv=np.zeros(self.project_duration),
                prod_rate=np.zeros(self.project_duration),
                prod_rate_baseline=np.zeros(self.project_duration),
            )

        return reduce(
            lambda x, y: x + y,
            (lft for lft in self.lifting if lft.fluid_type == FluidType.GAS),
        )

    def _get_sulfur_lifting(self) -> Lifting:
        """
        Determines total sulfur Lifting from the number of sulfur Lifting instances in
        attribute self.lifting.

        Returns
        -------
        total_sulfur_lifting: Lifting
            Total sulfur Lifting as a new instance of sulfur Lifting where the associated
            attributes are determined based on the prescribed rule in the corresponding
            dunder method __add__() of the Lifting class.

        Notes
        -----
        The order of operations is as follows:
        (1) Check the fluid_type in attribute self.lifting,
        (2) If SULFUR is not available as an instance in attribute self.lifting, then
            establish a new instance of SULFUR Lifting with the following attributes set
            to zero: lifting_rate, price.
        (3) Add the instances with fluid_type SULFUR, return the result as a new instance of
            SULFUR Lifting following the rule prescribed in the dunder method __add__() of
            class Lifting.
        """

        fluid_types = [lft.fluid_type for lft in self.lifting]

        if FluidType.SULFUR not in fluid_types:
            return Lifting(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                lifting_rate=np.zeros(self.project_duration),
                price=np.zeros(self.project_duration),
                prod_year=self.project_years,
                fluid_type=FluidType.SULFUR,
            )

        return reduce(
            lambda x, y: x + y,
            (lft for lft in self.lifting if lft.fluid_type == FluidType.SULFUR),
        )

    def _get_electricity_lifting(self) -> Lifting:
        """
        Determines total ELECTRICITY Lifting from the number of ELECTRICITY Lifting instances in
        attribute self.lifting.

        Returns
        -------
        total_electricity_lifting: Lifting
            Total ELECTRICITY Lifting as a new instance of ELECTRICITY Lifting where the associated
            attributes are determined based on the prescribed rule in the corresponding
            dunder method __add__() of the Lifting class.

        Notes
        -----
        The order of operations is as follows:
        (1) Check the fluid_type in attribute self.lifting,
        (2) If ELECTRICITY is not available as an instance in attribute self.lifting, then
            establish a new instance of ELECTRICITY Lifting with the following attributes set
            to zero: lifting_rate, price.
        (3) Add the instances with fluid_type ELECTRICITY, return the result as a new instance of
            ELECTRICITY Lifting following the rule prescribed in the dunder method __add__() of
            class Lifting.
        """

        fluid_types = [lft.fluid_type for lft in self.lifting]

        if FluidType.ELECTRICITY not in fluid_types:
            return Lifting(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                lifting_rate=np.zeros(self.project_duration),
                price=np.zeros(self.project_duration),
                prod_year=self.project_years,
                fluid_type=FluidType.ELECTRICITY,
            )

        return reduce(
            lambda x, y: x + y,
            (lft for lft in self.lifting if lft.fluid_type == FluidType.ELECTRICITY),
        )

    def _get_co2_lifting(self) -> Lifting:
        """
        Determines total CO2 Lifting from the number of CO2 Lifting instances in
        attribute self.lifting.

        Returns
        -------
        total_CO2_lifting: Lifting
            Total CO2 Lifting as a new instance of CO2 Lifting where the associated
            attributes are determined based on the prescribed rule in the corresponding
            dunder method __add__() of the Lifting class.

        Notes
        -----
        The order of operations is as follows:
        (1) Check the fluid_type in attribute self.lifting,
        (2) If CO2 is not available as an instance in attribute self.lifting, then
            establish a new instance of CO2 Lifting with the following attributes set
            to zero: lifting_rate, price.
        (3) Add the instances with fluid_type CO2, return the result as a new instance of
            CO2 Lifting following the rule prescribed in the dunder method __add__() of
            class Lifting.
        """

        fluid_types = [lft.fluid_type for lft in self.lifting]

        if FluidType.CO2 not in fluid_types:
            return Lifting(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                lifting_rate=np.zeros(self.project_duration),
                price=np.zeros(self.project_duration),
                prod_year=self.project_years,
                fluid_type=FluidType.CO2,
            )

        return reduce(
            lambda x, y: x + y,
            (lft for lft in self.lifting if lft.fluid_type == FluidType.CO2),
        )

    def _get_oil_capital(self) -> CapitalCost:
        """
        Determines total oil CapitalCost from the number of oil CapitalCost instances in
        attribute self.capital_cost_total.

        Returns
        -------
        CapitalCost
            An instance of CapitalCost that only includes FluidType.OIL as the associated
            cost_allocation that has been combined altogether following the rules prescribed
            in the dunder method __add__() of CapitalCost class.

        Notes
        -----
        The core operations are as follows:
        (1) Check the attribute cost_allocation in attribute self.capital_cost_total,
        (2) If OIL is not available as an instance in attribute self.capital_cost_total,
            then establish a new instance of OIL CapitalCost with the following attribute set
            to zero: cost.
        (3) Identify index location where cost_allocation is FluidType.OIL in attribute
            self.capital_cost_total,
        (4) Create a new instance of CapitalCost with only FluidType.OIL as its cost_allocation.
        """

        if FluidType.OIL not in self.capital_cost_total.cost_allocation:
            return CapitalCost(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                cost=np.array([0]),
                expense_year=np.array([self.start_date.year]),
                cost_allocation=[FluidType.OIL],
            )

        else:
            oil_capital_id = np.argwhere(
                np.array(self.capital_cost_total.cost_allocation) == FluidType.OIL
            ).ravel()

            start_year = self.capital_cost_total.start_year
            end_year = self.capital_cost_total.end_year
            cost = self.capital_cost_total.cost[oil_capital_id]
            expense_year = self.capital_cost_total.expense_year[oil_capital_id]
            cost_allocation = np.array(self.capital_cost_total.cost_allocation)[oil_capital_id]
            description = np.array(self.capital_cost_total.description)[oil_capital_id]
            pis_year = self.capital_cost_total.pis_year[oil_capital_id]
            salvage_value = self.capital_cost_total.salvage_value[oil_capital_id]
            useful_life = self.capital_cost_total.useful_life[oil_capital_id]
            depreciation_factor = self.capital_cost_total.depreciation_factor[oil_capital_id]
            is_ic_applied = np.array(self.capital_cost_total.is_ic_applied)[oil_capital_id]
            tax_portion = self.capital_cost_total.tax_portion[oil_capital_id]
            tax_discount = self.capital_cost_total.tax_discount[oil_capital_id]

            return CapitalCost(
                start_year=start_year,
                end_year=end_year,
                expense_year=expense_year,
                cost=cost,
                cost_allocation=cost_allocation.tolist(),
                description=description.tolist(),
                pis_year=pis_year,
                salvage_value=salvage_value,
                useful_life=useful_life,
                depreciation_factor=depreciation_factor,
                is_ic_applied=is_ic_applied.tolist(),
                tax_portion=tax_portion,
                tax_discount=tax_discount,
            )

    def _get_gas_capital(self) -> CapitalCost:
        """
        Determines total gas CapitalCost from the number of gas CapitalCost instances in
        attribute self.capital_cost_total.

        Returns
        -------
        CapitalCost
            An instance of CapitalCost that only includes FluidType.GAS as the associated
            cost_allocation that has been combined altogether following the rules prescribed
            in the dunder method __add__() of CapitalCost class.

        Notes
        -----
        The core operations are as follows:
        (1) Check the attribute cost_allocation in attribute self.capital_cost_total,
        (2) If GAS is not available as an instance in attribute self.capital_cost_total,
            then establish a new instance of GAS CapitalCost with the following attribute set
            to zero: cost.
        (3) Identify index location where cost_allocation is FluidType.GAS in attribute
            self.capital_cost_total,
        (4) Create a new instance of CapitalCost with only FluidType.GAS as its cost_allocation.
        """

        if FluidType.GAS not in self.capital_cost_total.cost_allocation:
            return CapitalCost(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                cost=np.array([0]),
                expense_year=np.array([self.start_date.year]),
                cost_allocation=[FluidType.GAS],
            )

        else:
            gas_capital_id = np.argwhere(
                np.array(self.capital_cost_total.cost_allocation) == FluidType.GAS
            ).ravel()

            start_year = self.capital_cost_total.start_year
            end_year = self.capital_cost_total.end_year
            cost = self.capital_cost_total.cost[gas_capital_id]
            expense_year = self.capital_cost_total.expense_year[gas_capital_id]
            cost_allocation = np.array(self.capital_cost_total.cost_allocation)[gas_capital_id]
            description = np.array(self.capital_cost_total.description)[gas_capital_id]
            pis_year = self.capital_cost_total.pis_year[gas_capital_id]
            salvage_value = self.capital_cost_total.salvage_value[gas_capital_id]
            useful_life = self.capital_cost_total.useful_life[gas_capital_id]
            depreciation_factor = self.capital_cost_total.depreciation_factor[gas_capital_id]
            is_ic_applied = np.array(self.capital_cost_total.is_ic_applied)[gas_capital_id]
            tax_portion = self.capital_cost_total.tax_portion[gas_capital_id]
            tax_discount = self.capital_cost_total.tax_discount[gas_capital_id]

            return CapitalCost(
                start_year=start_year,
                end_year=end_year,
                expense_year=expense_year,
                cost=cost,
                cost_allocation=cost_allocation.tolist(),
                description=description.tolist(),
                pis_year=pis_year,
                salvage_value=salvage_value,
                useful_life=useful_life,
                depreciation_factor=depreciation_factor,
                is_ic_applied=is_ic_applied.tolist(),
                tax_portion=tax_portion,
                tax_discount=tax_discount,
            )

    def _get_oil_intangible(self) -> Intangible:
        """
        Determines total oil Intangible from the number of oil Intangible instances in
        attribute self.intangible_cost_total.

        Returns
        -------
        Intangible
            An instance of Intangible that only includes FluidType.OIL as the associated
            cost_allocation that has been combined altogether following the rules prescribed
            in the dunder method __add__() of Intangible class.

        Notes
        -----
        The core operations are as follows:
        (1) Check the attribute cost_allocation in attribute self.intangible_cost_total,
        (2) If OIL is not available as an instance in attribute self.intangible_cost_total,
            then establish a new instance of OIL Intangible with the following attribute set
            to zero: cost.
        (3) Identify index location where cost_allocation is FluidType.OIL in attribute
            self.intangible_cost_total,
        (4) Create a new instance of Intangible with only FluidType.OIL as its cost_allocation.
        """

        if FluidType.OIL not in self.intangible_cost_total.cost_allocation:
            return Intangible(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                cost=np.array([0]),
                expense_year=np.array([self.start_date.year]),
                cost_allocation=[FluidType.OIL],
            )

        else:
            oil_intangible_id = np.argwhere(
                np.array(self.intangible_cost_total.cost_allocation) == FluidType.OIL
            ).ravel()

            start_year = self.intangible_cost_total.start_year
            end_year = self.intangible_cost_total.end_year
            cost = self.intangible_cost_total.cost[oil_intangible_id]
            expense_year = self.intangible_cost_total.expense_year[oil_intangible_id]
            cost_allocation = np.array(self.intangible_cost_total.cost_allocation)[oil_intangible_id]
            description = np.array(self.intangible_cost_total.description)[oil_intangible_id]
            tax_portion = self.intangible_cost_total.tax_portion[oil_intangible_id]
            tax_discount = self.intangible_cost_total.tax_discount[oil_intangible_id]

            return Intangible(
                start_year=start_year,
                end_year=end_year,
                cost=cost,
                expense_year=expense_year,
                cost_allocation=cost_allocation.tolist(),
                description=description.tolist(),
                tax_portion=tax_portion,
                tax_discount=tax_discount,
            )

    def _get_gas_intangible(self) -> Intangible:
        """
        Determines total gas Intangible from the number of gas Intangible instances in
        attribute self.intangible_cost_total.

        Returns
        -------
        Intangible
            An instance of Intangible that only includes FluidType.GAS as the associated
            cost_allocation that has been combined altogether following the rules prescribed
            in the dunder method __add__() of Intangible class.

        Notes
        -----
        The core operations are as follows:
        (1) Check the attribute cost_allocation in attribute self.intangible_cost_total,
        (2) If GAS is not available as an instance in attribute self.intangible_cost_total,
            then establish a new instance of GAS Intangible with the following attribute set
            to zero: cost.
        (3) Identify index location where cost_allocation is FluidType.GAS in attribute
            self.intangible_cost_total,
        (4) Create a new instance of Intangible with only FluidType.GAS as its cost_allocation.
        """

        if FluidType.GAS not in self.intangible_cost_total.cost_allocation:
            return Intangible(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                cost=np.array([0]),
                expense_year=np.array([self.start_date.year]),
                cost_allocation=[FluidType.GAS],
            )

        else:
            gas_intangible_id = np.argwhere(
                np.array(self.intangible_cost_total.cost_allocation) == FluidType.GAS
            ).ravel()

            start_year = self.intangible_cost_total.start_year
            end_year = self.intangible_cost_total.end_year
            cost = self.intangible_cost_total.cost[gas_intangible_id]
            expense_year = self.intangible_cost_total.expense_year[gas_intangible_id]
            cost_allocation = np.array(self.intangible_cost_total.cost_allocation)[gas_intangible_id]
            description = np.array(self.intangible_cost_total.description)[gas_intangible_id]
            tax_portion = self.intangible_cost_total.tax_portion[gas_intangible_id]
            tax_discount = self.intangible_cost_total.tax_discount[gas_intangible_id]

            return Intangible(
                start_year=start_year,
                end_year=end_year,
                cost=cost,
                expense_year=expense_year,
                cost_allocation=cost_allocation.tolist(),
                description=description.tolist(),
                tax_portion=tax_portion,
                tax_discount=tax_discount,
            )

    def _get_oil_opex(self) -> OPEX:
        """
        Determines total oil OPEX from the number of oil OPEX instances in
        attribute self.opex_total.

        Returns
        -------
        OPEX
            An instance of OPEX that only includes FluidType.OIL as the associated
            cost_allocation that has been combined altogether following the rules prescribed
            in the dunder method __add__() of OPEX class.

        Notes
        -----
        The core operations are as follows:
        (1) Check the attribute cost_allocation in attribute self.opex_total,
        (2) If OIL is not available as an instance in attribute self.opex_total,
            then establish a new instance of OIL OPEX with the following attribute set
            to zero: cost.
        (3) Identify index location where cost_allocation is FluidType.OIL in attribute
            self.opex_total,
        (4) Create a new instance of OPEX with only FluidType.OIL as its cost_allocation.
        """

        if FluidType.OIL not in self.opex_total.cost_allocation:
            return OPEX(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                fixed_cost=np.array([0]),
                expense_year=np.array([self.start_date.year]),
                cost_allocation=[FluidType.OIL],
            )

        else:
            oil_opex_id = np.argwhere(
                np.array(self.opex_total.cost_allocation) == FluidType.OIL
            ).ravel()

            start_year = self.opex_total.start_year
            end_year = self.opex_total.end_year
            expense_year = self.opex_total.expense_year[oil_opex_id]
            cost_allocation = np.array(self.opex_total.cost_allocation)[oil_opex_id]
            description = np.array(self.opex_total.description)[oil_opex_id]
            fixed_cost = self.opex_total.fixed_cost[oil_opex_id]
            prod_rate = self.opex_total.prod_rate[oil_opex_id]
            cost_per_volume = self.opex_total.cost_per_volume[oil_opex_id]
            tax_portion = self.opex_total.tax_portion[oil_opex_id]
            tax_discount = self.opex_total.tax_discount[oil_opex_id]

            return OPEX(
                start_year=start_year,
                end_year=end_year,
                expense_year=expense_year,
                cost_allocation=cost_allocation.tolist(),
                description=description.tolist(),
                fixed_cost=fixed_cost,
                prod_rate=prod_rate,
                cost_per_volume=cost_per_volume,
                tax_portion=tax_portion,
                tax_discount=tax_discount,
            )

    def _get_gas_opex(self) -> OPEX:
        """
        Determines total gas OPEX from the number of gas OPEX instances in
        attribute self.opex_total.

        Returns
        -------
        OPEX
            An instance of OPEX that only includes FluidType.GAS as the associated
            cost_allocation that has been combined altogether following the rules prescribed
            in the dunder method __add__() of OPEX class.

        Notes
        -----
        The core operations are as follows:
        (1) Check the attribute cost_allocation in attribute self.opex_total,
        (2) If GAS is not available as an instance in attribute self.opex_total,
            then establish a new instance of GAS OPEX with the following attribute set
            to zero: cost.
        (3) Identify index location where cost_allocation is FluidType.GAS in attribute
            self.opex_total,
        (4) Create a new instance of OPEX with only FluidType.GAS as its cost_allocation.
        """

        if FluidType.GAS not in self.opex_total.cost_allocation:
            return OPEX(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                fixed_cost=np.array([0]),
                expense_year=np.array([self.start_date.year]),
                cost_allocation=[FluidType.GAS],
            )

        else:
            gas_opex_id = np.argwhere(
                np.array(self.opex_total.cost_allocation) == FluidType.GAS
            ).ravel()

            start_year = self.opex_total.start_year
            end_year = self.opex_total.end_year
            expense_year = self.opex_total.expense_year[gas_opex_id]
            cost_allocation = np.array(self.opex_total.cost_allocation)[gas_opex_id]
            description = np.array(self.opex_total.description)[gas_opex_id]
            fixed_cost = self.opex_total.fixed_cost[gas_opex_id]
            prod_rate = self.opex_total.prod_rate[gas_opex_id]
            cost_per_volume = self.opex_total.cost_per_volume[gas_opex_id]
            tax_portion = self.opex_total.tax_portion[gas_opex_id]
            tax_discount = self.opex_total.tax_discount[gas_opex_id]

            return OPEX(
                start_year=start_year,
                end_year=end_year,
                expense_year=expense_year,
                cost_allocation=cost_allocation.tolist(),
                description=description.tolist(),
                fixed_cost=fixed_cost,
                prod_rate=prod_rate,
                cost_per_volume=cost_per_volume,
                tax_portion=tax_portion,
                tax_discount=tax_discount,
            )

    def _get_oil_asr(self) -> ASR:
        """
        Determines total oil ASR from the number of oil ASR instances in
        attribute self.asr_cost_total.

        Returns
        -------
        ASR
            An instance of ASR that only includes FluidType.OIL as the associated
            cost_allocation that has been combined altogether following the rules
            prescribed in the dunder method __add__() of ASR class.

        Notes
        -----
        The core operations are as follows:
        (1) Check the attribute cost_allocation in attribute self.asr_cost_total,
        (2) If OIL is not available as an instance in attribute self.asr_cost_total,
            then establish a new instance of OIL ASR with the following attribute set
            to zero: cost.
        (3) Identify index location where cost_allocation is FluidType.OIL in attribute
            self.asr_cost_total,
        (4) Create a new instance of ASR with only FluidType.OIL as its cost_allocation.
        """

        if FluidType.OIL not in self.asr_cost_total.cost_allocation:
            return ASR(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                cost=np.array([0]),
                expense_year=np.array([self.start_date.year]),
                cost_allocation=[FluidType.OIL],
            )

        else:
            oil_asr_id = np.argwhere(
                np.array(self.asr_cost_total.cost_allocation) == FluidType.OIL
            ).ravel()

            start_year = self.asr_cost_total.start_year
            end_year = self.asr_cost_total.end_year
            cost = self.asr_cost_total.cost[oil_asr_id]
            expense_year = self.asr_cost_total.expense_year[oil_asr_id]
            cost_allocation = np.array(self.asr_cost_total.cost_allocation)[oil_asr_id]
            description = np.array(self.asr_cost_total.description)[oil_asr_id]
            final_year = self.asr_cost_total.final_year[oil_asr_id]
            future_rate = self.asr_cost_total.future_rate[oil_asr_id]
            tax_portion = self.asr_cost_total.tax_portion[oil_asr_id]
            tax_discount = self.asr_cost_total.tax_discount[oil_asr_id]

            return ASR(
                start_year=start_year,
                end_year=end_year,
                cost=cost,
                expense_year=expense_year,
                cost_allocation=cost_allocation.tolist(),
                description=description.tolist(),
                final_year=final_year,
                future_rate=future_rate,
                tax_portion=tax_portion,
                tax_discount=tax_discount,
            )

    def _get_gas_asr(self) -> ASR:
        """
        Determines total gas ASR from the number of gas ASR instances in
        attribute self.asr_cost_total.

        Returns
        -------
        ASR
            An instance of ASR that only includes FluidType.GAS as the associated
            cost_allocation that has been combined altogether following the rules
            prescribed in the dunder method __add__() of ASR class.

        Notes
        -----
        The core operations are as follows:
        (1) Check the attribute cost_allocation in attribute self.asr_cost_total,
        (2) If GAS is not available as an instance in attribute self.asr_cost_total,
            then establish a new instance of GAS ASR with the following attribute set
            to zero: cost.
        (3) Identify index location where cost_allocation is FluidType.GAS in attribute
            self.asr_cost_total,
        (4) Create a new instance of ASR with only FluidType.GAS as its cost_allocation.
        """

        if FluidType.GAS not in self.asr_cost_total.cost_allocation:
            return ASR(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                cost=np.array([0]),
                expense_year=np.array([self.start_date.year]),
                cost_allocation=[FluidType.GAS],
            )

        else:
            gas_asr_id = np.argwhere(
                np.array(self.asr_cost_total.cost_allocation) == FluidType.GAS
            ).ravel()

            start_year = self.asr_cost_total.start_year
            end_year = self.asr_cost_total.end_year
            cost = self.asr_cost_total.cost[gas_asr_id]
            expense_year = self.asr_cost_total.expense_year[gas_asr_id]
            cost_allocation = np.array(self.asr_cost_total.cost_allocation)[gas_asr_id]
            description = np.array(self.asr_cost_total.description)[gas_asr_id]
            final_year = self.asr_cost_total.final_year[gas_asr_id]
            future_rate = self.asr_cost_total.future_rate[gas_asr_id]
            tax_portion = self.asr_cost_total.tax_portion[gas_asr_id]
            tax_discount = self.asr_cost_total.tax_discount[gas_asr_id]

            return ASR(
                start_year=start_year,
                end_year=end_year,
                cost=cost,
                expense_year=expense_year,
                cost_allocation=cost_allocation.tolist(),
                description=description.tolist(),
                final_year=final_year,
                future_rate=future_rate,
                tax_portion=tax_portion,
                tax_discount=tax_discount,
            )

    def _get_oil_lbt(self) -> LBT:
        """
        Determines total oil LBT from the number of oil LBT instances in
        attribute self.lbt_cost_total.

        Returns
        -------
        LBT
            An instance of LBT that only includes FluidType.OIL as the associated
            cost_allocation that has been combined altogether following the rules
            prescribed in the dunder method __add__() of LBT class.

        Notes
        -----
        The core operations are as follows:
        (1) Check the attribute cost_allocation in attribute self.lbt_cost_total,
        (2) If OIL is not available as an instance in attribute self.lbt_cost_total,
            then establish a new instance of OIL LBT with the following attribute set
            to zero: cost.
        (3) Identify index location where cost_allocation is FluidType.OIL in attribute
            self.lbt_cost_total,
        (4) Create a new instance of LBT with only FluidType.OIL as its cost_allocation.
        """

        if FluidType.OIL not in self.lbt_cost_total.cost_allocation:
            return LBT(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                expense_year=np.array([self.start_date.year]),
                cost_allocation=[FluidType.OIL],
            )

        else:
            oil_lbt_id = np.argwhere(
                np.array(self.lbt_cost_total.cost_allocation) == FluidType.OIL
            ).ravel()

            start_year = self.lbt_cost_total.start_year
            end_year = self.lbt_cost_total.end_year
            expense_year = self.lbt_cost_total.expense_year[oil_lbt_id]
            cost_allocation = np.array(self.lbt_cost_total.cost_allocation)[oil_lbt_id]
            description = np.array(self.lbt_cost_total.description)[oil_lbt_id]
            final_year = self.lbt_cost_total.final_year[oil_lbt_id]
            utilized_land_area = self.lbt_cost_total.utilized_land_area[oil_lbt_id]
            utilized_building_area = self.lbt_cost_total.utilized_building_area[oil_lbt_id]
            njop_land = self.lbt_cost_total.njop_land[oil_lbt_id]
            njop_building = self.lbt_cost_total.njop_building[oil_lbt_id]
            gross_revenue = self.lbt_cost_total.gross_revenue[oil_lbt_id]
            cost = self.lbt_cost_total.cost[oil_lbt_id]
            tax_portion = self.lbt_cost_total.tax_portion[oil_lbt_id]
            tax_discount = self.lbt_cost_total.tax_discount[oil_lbt_id]

            return LBT(
                start_year=start_year,
                end_year=end_year,
                expense_year=expense_year,
                cost_allocation=cost_allocation.tolist(),
                description=description.tolist(),
                final_year=final_year,
                utilized_land_area=utilized_land_area,
                utilized_building_area=utilized_building_area,
                njop_land=njop_land,
                njop_building=njop_building,
                gross_revenue=gross_revenue,
                cost=cost,
                tax_portion=tax_portion,
                tax_discount=tax_discount,
            )

    def _get_gas_lbt(self) -> LBT:
        """
        Determines total gas LBT from the number of gas LBT instances in
        attribute self.lbt_cost_total.

        Returns
        -------
        LBT
            An instance of LBT that only includes FluidType.GAS as the associated
            cost_allocation that has been combined altogether following the rules
            prescribed in the dunder method __add__() of LBT class.

        Notes
        -----
        The core operations are as follows:
        (1) Check the attribute cost_allocation in attribute self.lbt_cost_total,
        (2) If GAS is not available as an instance in attribute self.lbt_cost_total,
            then establish a new instance of GAS LBT with the following attribute set
            to zero: cost.
        (3) Identify index location where cost_allocation is FluidType.GAS in attribute
            self.lbt_cost_total,
        (4) Create a new instance of LBT with only FluidType.GAS as its cost_allocation.
        """

        if FluidType.GAS not in self.lbt_cost_total.cost_allocation:
            return LBT(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                expense_year=np.array([self.start_date.year]),
                cost_allocation=[FluidType.OIL],
            )

        else:
            gas_lbt_id = np.argwhere(
                np.array(self.lbt_cost_total.cost_allocation) == FluidType.GAS
            ).ravel()

            start_year = self.lbt_cost_total.start_year
            end_year = self.lbt_cost_total.end_year
            expense_year = self.lbt_cost_total.expense_year[gas_lbt_id]
            cost_allocation = np.array(self.lbt_cost_total.cost_allocation)[gas_lbt_id]
            description = np.array(self.lbt_cost_total.description)[gas_lbt_id]
            final_year = self.lbt_cost_total.final_year[gas_lbt_id]
            utilized_land_area = self.lbt_cost_total.utilized_land_area[gas_lbt_id]
            utilized_building_area = self.lbt_cost_total.utilized_building_area[gas_lbt_id]
            njop_land = self.lbt_cost_total.njop_land[gas_lbt_id]
            njop_building = self.lbt_cost_total.njop_building[gas_lbt_id]
            gross_revenue = self.lbt_cost_total.gross_revenue[gas_lbt_id]
            cost = self.lbt_cost_total.cost[gas_lbt_id]
            tax_portion = self.lbt_cost_total.tax_portion[gas_lbt_id]
            tax_discount = self.lbt_cost_total.tax_discount[gas_lbt_id]

            return LBT(
                start_year=start_year,
                end_year=end_year,
                expense_year=expense_year,
                cost_allocation=cost_allocation.tolist(),
                description=description.tolist(),
                final_year=final_year,
                utilized_land_area=utilized_land_area,
                utilized_building_area=utilized_building_area,
                njop_land=njop_land,
                njop_building=njop_building,
                gross_revenue=gross_revenue,
                cost=cost,
                tax_portion=tax_portion,
                tax_discount=tax_discount,
            )

    def _get_oil_cost_of_sales(self) -> CostOfSales:
        """
        Retrieve or create the cost of sales for oil.

        This method checks if the `FluidType.OIL` is present in the `cost_allocation`
        of `cost_of_sales_total`. If not present, it creates a new `CostOfSales` instance
        with zero costs for the duration of the project. If present, it extracts the relevant
        cost information for `FluidType.OIL` and returns a `CostOfSales` instance populated
        with this data.

        Returns
        -------
        CostOfSales
            An instance of `CostOfSales` representing the cost of sales for oil.

        Notes
        -----
        - If `FluidType.OIL` is not in `cost_allocation`, a new `CostOfSales` instance is
          created with zero costs.
        - If `FluidType.OIL` is in `cost_allocation`, the corresponding cost data is extracted
          and used to create the `CostOfSales` instance.
        """

        if FluidType.OIL not in self.cost_of_sales_total.cost_allocation:
            return CostOfSales(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                expense_year=self.project_years,
                cost=np.zeros_like(self.project_years, dtype=np.float64),
                cost_allocation=[FluidType.OIL for _ in range(self.project_duration)],
            )

        else:
            oil_cost_of_sales_id = np.argwhere(
                np.array(self.cost_of_sales_total.cost_allocation) == FluidType.OIL
            ).ravel()

            start_year = self.cost_of_sales_total.start_year
            end_year = self.cost_of_sales_total.end_year
            expense_year = self.cost_of_sales_total.expense_year[oil_cost_of_sales_id]
            cost = self.cost_of_sales_total.cost[oil_cost_of_sales_id]
            cost_allocation = np.array(self.cost_of_sales_total.cost_allocation)[oil_cost_of_sales_id]
            description = np.array(self.cost_of_sales_total.description)[oil_cost_of_sales_id]
            tax_portion = self.cost_of_sales_total.tax_portion[oil_cost_of_sales_id]
            tax_discount = self.cost_of_sales_total.tax_discount[oil_cost_of_sales_id]

            return CostOfSales(
                start_year=start_year,
                end_year=end_year,
                expense_year=expense_year,
                cost=cost,
                cost_allocation=cost_allocation.tolist(),
                description=description.tolist(),
                tax_portion=tax_portion,
                tax_discount=tax_discount,
            )

    def _get_gas_cost_of_sales(self) -> CostOfSales:
        """
        Retrieve or create the cost of sales for gas.

        This method checks if the `FluidType.GAS` is present in the `cost_allocation`
        of `cost_of_sales_total`. If not present, it creates a new `CostOfSales` instance
        with zero costs for the duration of the project. If present, it extracts the relevant
        cost information for `FluidType.GAS` and returns a `CostOfSales` instance populated
        with this data.

        Returns
        -------
        CostOfSales
            An instance of `CostOfSales` representing the cost of sales for gas.

        Notes
        -----
        - If `FluidType.GAS` is not in `cost_allocation`, a new `CostOfSales` instance is
          created with zero costs.
        - If `FluidType.GAS` is in `cost_allocation`, the corresponding cost data is extracted
          and used to create the `CostOfSales` instance.
        """

        if FluidType.GAS not in self.cost_of_sales_total.cost_allocation:
            return CostOfSales(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                expense_year=self.project_years,
                cost=np.zeros_like(self.project_years, dtype=np.float64),
                cost_allocation=[FluidType.GAS for _ in range(self.project_duration)],
            )

        else:
            gas_cost_of_sales_id = np.argwhere(
                np.array(self.cost_of_sales_total.cost_allocation) == FluidType.GAS
            ).ravel()

            start_year = self.cost_of_sales_total.start_year
            end_year = self.cost_of_sales_total.end_year
            expense_year = self.cost_of_sales_total.expense_year[gas_cost_of_sales_id]
            cost = self.cost_of_sales_total.cost[gas_cost_of_sales_id]
            cost_allocation = np.array(self.cost_of_sales_total.cost_allocation)[gas_cost_of_sales_id]
            description = np.array(self.cost_of_sales_total.description)[gas_cost_of_sales_id]
            tax_portion = self.cost_of_sales_total.tax_portion[gas_cost_of_sales_id]
            tax_discount = self.cost_of_sales_total.tax_discount[gas_cost_of_sales_id]

            return CostOfSales(
                start_year=start_year,
                end_year=end_year,
                expense_year=expense_year,
                cost=cost,
                cost_allocation=cost_allocation.tolist(),
                description=description.tolist(),
                tax_portion=tax_portion,
                tax_discount=tax_discount,
            )

    def _calc_expenditures_pre_tax(
        self,
        target_attr,
        year_inflation: np.ndarray = None,
        inflation_rate: np.ndarray | float = 0.0,
        inflation_rate_applied_to: InflationAppliedTo | None = None,
    ):

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

    def _get_expenditures_pre_tax(
        self,
        year_inflation: np.ndarray = None,
        inflation_rate: np.ndarray | float = 0.0,
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

        # Prepare expenditures pre tax associated with capital, intangible, opex, asr, and lbt costs
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
            self._calc_expenditures_pre_tax(
                target_attr=attr,
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
            A NumPy array or float representing the tax rate applied to the costs. If not provided,
            a default rate of 0.0 will be used. When provided as an array, it should match
            the project years.

        Returns
        -------
        None
            This method does not return any values. It updates the following attributes with the
            calculated indirect taxes for oil and gas:

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

        # Prepare indirect taxes associated with capital, intangible, opex, asr, and lbt costs
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
        inflation_rate: np.ndarray | float = 0.0,
        tax_rate: np.ndarray | float = 0.0,
    ) -> None:
        """
        Calculate and assign post-tax expenditures for various oil and gas expenditure categories.

        This method computes the post-tax expenditures for multiple categories related to oil
        and gas. It adjusts these expenditures by accounting for inflation, tax portions, tax rates,
        and tax discounts.

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
        tax_rate : np.ndarray or float, optional
            The tax rate to apply to the costs. If a float is provided, it applies uniformly across all
            project years. If a NumPy array is provided, the rate can vary by year (default is 0.0).

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

        # Prepare expenditures post tax associated with capital, intangible, opex, asr, and lbt costs
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
            attr.expenditures_post_tax(
                year_inflation=year_inflation,
                inflation_rate=inflation_rate,
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

    def _get_oil_wap_price(self) -> None:
        """
        Calculate the weighted average price (WAP) of oil for the project years.

        This method computes the oil weighted average price (WAP) for each project year based on the
        lifting rates and prices of oil. The weighted average price is determined by dividing the
        volume-weighted prices by the total oil lifting volume for each year.

        The lifting data is iterated through, and the lifting rates and prices are considered only
        for `FluidType.OIL`.

        Returns
        -------
        None
            The method updates the attribute `self._oil_wap_price` as a numpy array of weighted average
            prices for each project year.

        Notes
        -----
        -   The method first initializes two arrays, `vol_x_price` and `total_vol`, both of which
            are the same size as `self.project_years`. These arrays are used to accumulate the
            volume-weighted price and total volume of oil for each project year.
        -   The method iterates over the `self.lifting` attribute, which contains lifting data for
            various fluid types. Only entries where `lift.fluid_type == FluidType.OIL` are considered.
        -   For oil liftings, the lifting rate array and price array are used to calculate the
            volume-weighted price (`vol_x_price`), and the total lifting volume (`total_vol`).
        -   The oil weighted average price is then calculated as the element-wise division of
            `vol_x_price` by `total_vol`, only for years where `total_vol` is not zero.
        -   The result is stored in `self._oil_wap_price`.
        """

        vol_x_price = np.zeros_like(self.project_years, dtype=float)
        total_vol = np.zeros_like(self.project_years, dtype=float)

        for lift in self.lifting:
            if lift.fluid_type == FluidType.OIL:
                vol_x_price = vol_x_price + lift.get_lifting_rate_arr() * lift.get_price_arr()
                total_vol = total_vol + lift.get_lifting_rate_arr()

        self._oil_wap_price = np.divide(vol_x_price, total_vol, where=total_vol != 0)

    def _get_gas_wap_price(self) -> None:
        """
        Calculate the weighted average price (WAP) of gas based on the lifting volumes and prices.

        This method computes the weighted average price of gas over the project years. The gas WAP is
        calculated by multiplying the lifting rates with the associated prices and dividing by the
        total lifting volumes. If there is no lifting data (both total volume and volume x price are
        zero), the method assigns zero to the gas WAP.

        Returns
        -------
        None
            The method sets the `self._gas_wap_price` attribute to a NumPy array containing the
            weighted average gas price for each project year. If no lifting data is available,
            it assigns zeros.

        Notes
        -----
        -   The `self.lifting` attribute is expected to contain objects that have the attributes
            `fluid_type`, `get_lifting_rate_arr()`, and `get_price_arr()`.
        -   The method calculates the weighted average price for gas only, skipping any lifting
            records that are not related to gas (`FluidType.GAS`).
        """

        vol_x_price = np.zeros_like(self.project_years, dtype=float)
        total_vol = np.zeros_like(self.project_years, dtype=float)

        for lift in self.lifting:
            if lift.fluid_type == FluidType.GAS:
                vol_x_price = vol_x_price + lift.get_lifting_rate_arr() * lift.get_price_arr()
                total_vol = total_vol + lift.get_lifting_rate_arr()

        if np.sum(vol_x_price) + np.sum(total_vol) == 0:
            self._gas_wap_price = np.zeros_like(vol_x_price)

        else:
            self._gas_wap_price = np.divide(vol_x_price, total_vol, where=total_vol != 0)

    def _get_sulfur_wap_price(self) -> None:
        """
        Calculate the weighted average price (WAP) of sulfur based on lifting volumes and prices.

        This method computes the weighted average price of sulfur for the project years. The sulfur
        WAP is calculated by multiplying the sulfur lifting rates with their respective prices and
        dividing by the total lifting volumes for sulfur. The method skips lifting data unrelated
        to sulfur.

        Returns
        -------
        None
            The method sets the `self._sulfur_wap_price` attribute to a NumPy array containing the
            weighted average sulfur price for each project year.

        Notes
        -----
        -   The `self.lifting` attribute is expected to contain objects that have the attributes
            `fluid_type`, `get_lifting_rate_arr()`, and `get_price_arr()`.
        -   The method calculates the weighted average price for sulfur only, skipping any lifting
            records that are not related to sulfur (`FluidType.SULFUR`).
        -   If there are no valid lifting records for sulfur, the division is avoided by using
            `np.divide` with the `where` condition to ensure that no division by zero occurs.
        """

        vol_x_price = np.zeros_like(self.project_years, dtype=float)
        total_vol = np.zeros_like(self.project_years, dtype=float)

        for lift in self.lifting:
            if lift.fluid_type == FluidType.SULFUR:
                vol_x_price = vol_x_price + lift.get_lifting_rate_arr() * lift.get_price_arr()
                total_vol = total_vol + lift.get_lifting_rate_arr()

        self._sulfur_wap_price = np.divide(vol_x_price, total_vol, where=total_vol != 0)

    def _get_electricity_wap_price(self) -> None:
        """
        Calculate and set the weighted average price (WAP) of electricity for the project years.

        This method computes the weighted average price (WAP) of electricity based on the lifting rates
        and prices associated with electricity. It multiplies the electricity lifting volumes by their
        respective prices and then divides the sum by the total lifting volumes to calculate the WAP.

        Returns
        -------
        None
            The method updates the attribute `self._electricity_wap_price`, which contains a NumPy
            array with the weighted average price of electricity for each project year.

        Notes
        -----
        -   The `self.lifting` attribute is expected to contain objects that provide methods
            `get_lifting_rate_arr()` and `get_price_arr()` to retrieve lifting rates and prices.
        -   Only entries with `FluidType.ELECTRICITY` are used in the WAP calculation.
        -   The `np.divide` function ensures safe division, avoiding division by zero when no
            volumes are present.
        """

        vol_x_price = np.zeros_like(self.project_years, dtype=float)
        total_vol = np.zeros_like(self.project_years, dtype=float)

        for lift in self.lifting:
            if lift.fluid_type == FluidType.ELECTRICITY:
                vol_x_price = vol_x_price + lift.get_lifting_rate_arr() * lift.get_price_arr()
                total_vol = total_vol + lift.get_lifting_rate_arr()

        self._electricity_wap_price = np.divide(vol_x_price, total_vol, where=total_vol != 0)

    def _get_co2_wap_price(self) -> None:
        """
        Calculate and set the weighted average price (WAP) of CO2 for the project years.

        This method computes the weighted average price (WAP) of CO2 based on the lifting rates
        and prices associated with CO2. It multiplies the CO2 lifting volumes by their respective
        prices and then divides the sum by the total lifting volumes to calculate the WAP.

        Returns
        -------
        None
            The method updates the attribute `self._co2_wap_price`, which contains a NumPy array
            with the weighted average price of CO2 for each project year.

        Notes
        -----
        -   The `self.lifting` attribute is expected to contain objects that provide methods
            `get_lifting_rate_arr()` and `get_price_arr()` to retrieve lifting rates and prices.
        -   Only entries with `FluidType.CO2` are used in the WAP calculation.
        -   The `np.divide` function ensures safe division, avoiding division by zero when no
            volumes are present.
        """

        vol_x_price = np.zeros_like(self.project_years, dtype=float)
        total_vol = np.zeros_like(self.project_years, dtype=float)

        for lift in self.lifting:
            if lift.fluid_type == FluidType.CO2:
                vol_x_price = vol_x_price + lift.get_lifting_rate_arr() * lift.get_price_arr()
                total_vol = total_vol + lift.get_lifting_rate_arr()

        self._co2_wap_price = np.divide(vol_x_price, total_vol, where=total_vol != 0)

    def _get_wap_price(self) -> None:
        """
        Calculate and set the weighted average prices (WAP) for oil, gas, sulfur, electricity,
        and CO2 across the project years.

        This method sequentially calls individual methods to compute the weighted average prices
        for different fluid types (oil, gas, sulfur, electricity, and CO2). It then stores the
        resulting values in corresponding attributes of the object.

        The following methods are invoked:
        -   `_get_oil_wap_price`: Computes the WAP for oil.
        -   `_get_gas_wap_price`: Computes the WAP for gas.
        -   `_get_sulfur_wap_price`: Computes the WAP for sulfur.
        -   `_get_electricity_wap_price`: Computes the WAP for electricity.
        -   `_get_co2_wap_price`: Computes the WAP for CO2.

        Returns
        -------
        None
            The method updates the following attributes with their corresponding WAP values:
            -   `self._oil_wap_price`
            -   `self._gas_wap_price`
            -   `self._sulfur_wap_price`
            -   `self._electricity_wap_price`
            -   `self._co2_wap_price`

        Notes
        -----
        -   Each WAP is computed by considering the lifting rates and prices for the corresponding
            fluid type.
        -   The WAP values are stored as attributes of the object for future use.
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
        Configure treatments for other revenue sources.

        This method adjusts the financial attributes (revenues and expenditures) based on the
        revenue sources for sulfur, electricity, and CO2. The adjustment is made according to
        the type of other revenue selected for each source (either addition to revenue or reduction
        to operating expenses).

        The method updates the following attributes depending on the input revenue selections:
        -   `self._oil_revenue`
        -   `self._gas_revenue`
        -   `self._oil_opex_expenditures_post_tax`
        -   `self._gas_opex_expenditures_post_tax`

        Notes
        -----
        -   This method modifies the internal attributes based on the selected revenue types for
            sulfur, electricity, and CO2.
        -   If an unsupported revenue type is provided, an `OtherRevenueException` is raised.
        """

        # Configure sulfur revenue
        if sulfur_revenue is OtherRevenue.ADDITION_TO_OIL_REVENUE:
            self._oil_revenue = self._oil_revenue + self._sulfur_revenue

        elif sulfur_revenue is OtherRevenue.ADDITION_TO_GAS_REVENUE:
            self._gas_revenue = self._gas_revenue + self._sulfur_revenue

        elif sulfur_revenue is OtherRevenue.REDUCTION_TO_OIL_OPEX:
            self._oil_opex_expenditures_post_tax = (
                self._oil_opex_expenditures_post_tax - self._sulfur_revenue
            )

        elif sulfur_revenue is OtherRevenue.REDUCTION_TO_GAS_OPEX:
            self._gas_opex_expenditures_post_tax = (
                self._gas_opex_expenditures_post_tax - self._sulfur_revenue
            )

        else:
            raise OtherRevenueException(
                f"Other Revenue Selection is not available {sulfur_revenue} "
            )

        # Configure electricity revenue
        if electricity_revenue is OtherRevenue.ADDITION_TO_OIL_REVENUE:
            self._oil_revenue = self._oil_revenue + self._electricity_revenue

        elif electricity_revenue is OtherRevenue.ADDITION_TO_GAS_REVENUE:
            self._gas_revenue = self._gas_revenue + self._sulfur_revenue

        elif electricity_revenue is OtherRevenue.REDUCTION_TO_OIL_OPEX:
            self._oil_opex_expenditures_post_tax = (
                self._oil_opex_expenditures_post_tax - self._sulfur_revenue
            )

        elif electricity_revenue is OtherRevenue.REDUCTION_TO_GAS_OPEX:
            self._gas_opex_expenditures_post_tax = (
                self._gas_opex_expenditures_post_tax - self._sulfur_revenue
            )

        else:
            raise OtherRevenueException(
                f"Other Revenue Selection is not available {electricity_revenue} "
            )

        # Configure CO2 revenue
        if co2_revenue is OtherRevenue.ADDITION_TO_OIL_REVENUE:
            self._oil_revenue = self._oil_revenue + self._co2_revenue

        elif co2_revenue is OtherRevenue.ADDITION_TO_GAS_REVENUE:
            self._gas_revenue = self._gas_revenue + self._co2_revenue

        elif co2_revenue is OtherRevenue.REDUCTION_TO_OIL_OPEX:
            self._oil_opex_expenditures_post_tax = (
                self._oil_opex_expenditures_post_tax - self._co2_revenue
            )

        elif co2_revenue is OtherRevenue.REDUCTION_TO_GAS_OPEX:
            self._gas_opex_expenditures_post_tax = (
                self._gas_opex_expenditures_post_tax - self._co2_revenue
            )

        else:
            raise OtherRevenueException(
                f"Other Revenue Selection is not available {co2_revenue} "
            )

    def _get_sunk_cost(self, sunk_cost_reference_year: int) -> None:
        """
        Calculate and update the sunk costs for oil and gas up to a specified reference year.

        This method computes the sunk costs for both oil and gas by adding capital and non-capital
        expenditures up to the given `sunk_cost_reference_year`. The method updates internal
        attributes for oil and gas sunk costs.

        The sunk costs are calculated as follows:
        -   For oil: `self._oil_sunk_cost` is updated with the sum of
            `self._oil_capital_expenditures_post_tax` and `self._oil_non_capital`
            up to the specified year.
        -   For gas: `self._gas_sunk_cost` is updated similarly with
            `self._gas_capital_expenditures_post_tax` and `self._gas_non_capital`.

        If the `sunk_cost_reference_year` is the same as the project's start year, both oil and gas
        sunk costs are set to zero.

        Parameters
        ----------
        sunk_cost_reference_year : int
            The year up to which the sunk costs for oil and gas should be calculated. It must be
            greater than or equal to the project's start year.

        Attributes Updated
        ------------------
        self._oil_sunk_cost : np.ndarray
            The calculated sunk costs for oil up to the specified reference year.
        self._gas_sunk_cost : np.ndarray
            The calculated sunk costs for gas up to the specified reference year.
        """

        # Oil sunk cost
        oil_cost_raw = (
            self._oil_capital_expenditures_post_tax
            + self._oil_non_capital
        )

        self._oil_sunk_cost = oil_cost_raw[
                              : (sunk_cost_reference_year - self.start_date.year + 1)
                              ]

        # Gas sunk cost
        gas_cost_raw = (
            self._gas_capital_expenditures_post_tax
            + self._gas_non_capital
        )

        self._gas_sunk_cost = gas_cost_raw[
                              : (sunk_cost_reference_year - self.start_date.year + 1)
                              ]

        if sunk_cost_reference_year == self.start_date.year:
            self._oil_sunk_cost = np.zeros(1)
            self._gas_sunk_cost = np.zeros(1)

    def run(
        self,
        sulfur_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_GAS_REVENUE,
        electricity_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_OIL_REVENUE,
        co2_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_GAS_REVENUE,
        sunk_cost_reference_year: int = None,
        year_inflation: np.ndarray = None,
        inflation_rate: np.ndarray | float = 0.0,
        # tax_portion: np.ndarray = None,
        tax_rate: np.ndarray | float = 0.0,
        # tax_discount: float = 0.0,
        inflation_rate_applied_to: InflationAppliedTo = None,
        # year_ref: int = None,
        # tax_type: TaxType = TaxType.VAT,
        # vat_rate: np.ndarray | float = 0.0,
        # lbt_rate: np.ndarray | float = 0.0,
        # inflation_rate: np.ndarray | float = 0.0,
        # future_rate: float = 0.02,
    ) -> None:
        """
        Run the economic analysis, calculating expenditures and configuring
        cashflows for OIL and GAS.

        Parameters
        ----------

        Returns
        -------
        None

        Notes
        -----
        The core procedures are as follows:
        (1) Calculate the expenditures by calling the private method _get_expenditures(),
        (2) Calculate total expenditures for OIL and GAS,
        (3) Configure the cashflow for OIL and GAS.
        """

        # Configure Sunk Cost Reference Year
        if sunk_cost_reference_year is None:
            sunk_cost_reference_year = self.start_date.year

        if sunk_cost_reference_year > self.oil_onstream_date.year:
            raise SunkCostException(
                f"Sunk Cost Reference Year {sunk_cost_reference_year} "
                f"is after the on stream date: {self.oil_onstream_date}"
            )

        if sunk_cost_reference_year > self.gas_onstream_date.year:
            raise SunkCostException(
                f"Sunk Cost Reference Year {sunk_cost_reference_year} "
                f"is after the on stream date: {self.gas_onstream_date}"
            )

        if sunk_cost_reference_year < self.start_date.year:
            raise SunkCostException(
                f"Sunk Cost Reference Year {sunk_cost_reference_year} "
                f"is before the project start date: {self.start_date}"
            )

        if sunk_cost_reference_year > self.end_date.year:
            raise SunkCostException(
                f"Sunk Cost Reference Year {sunk_cost_reference_year} "
                f"is after the project end date: {self.end_date}"
            )

        # Calculate pre tax expenditures
        self._get_expenditures_pre_tax(
            year_inflation=year_inflation,
            inflation_rate=inflation_rate,
            inflation_rate_applied_to=inflation_rate_applied_to,
        )

        # # Calculate indirect taxes
        # self._get_indirect_taxes(tax_rate=tax_rate)
        #
        # # Calculate post tax expenditures
        # self._get_expenditures_post_tax(
        #     year_inflation=year_inflation,
        #     inflation_rate=inflation_rate,
        #     tax_rate=tax_rate,
        # )

        print('\t')
        print(f'Filetype: {type(self._oil_asr_expenditures_pre_tax)}')
        print(f'Length: {len(self._oil_asr_expenditures_pre_tax)}')
        print('_oil_asr_expenditures_pre_tax = \n', self._oil_asr_expenditures_pre_tax)

        print('\t')
        print(f'Filetype: {type(self._oil_intangible_expenditures_pre_tax)}')
        print(f'Length: {len(self._oil_intangible_expenditures_pre_tax)}')
        print('_oil_intangible_expenditures_pre_tax = \n', self._oil_intangible_expenditures_pre_tax)

        # # Prepare the data
        # self._get_expenditures(
        #     year_ref=year_ref,
        #     tax_type=tax_type,
        #     vat_rate=vat_rate,
        #     lbt_rate=lbt_rate,
        #     inflation_rate=inflation_rate,
        #     future_rate=future_rate,
        # )
        #
        # # Non-capital costs (intangible + opex + asr)
        # self._oil_non_capital = (
        #         self._oil_intangible_expenditures
        #         + self._oil_opex_expenditures
        #         + self._oil_asr_expenditures
        # )
        #
        # self._gas_non_capital = (
        #         self._gas_intangible_expenditures
        #         + self._gas_opex_expenditures
        #         + self._gas_asr_expenditures
        # )
        #
        # # Get Sunk Cost
        # self._get_sunk_cost(sunk_cost_reference_year)
        #
        # # Get the wap of each produced fluid
        # self._get_wap_price()
        #
        # # Get the other revenue
        # self._get_other_revenue(sulfur_revenue=sulfur_revenue,
        #                         electricity_revenue=electricity_revenue,
        #                         co2_revenue=co2_revenue,)
        #
        # # Configure total expenditures for OIL and GAS
        # oil_total_expenditures = reduce(
        #     lambda x, y: x + y,
        #     [
        #         self._oil_capital_expenditures, self._oil_intangible_expenditures,
        #         self._oil_opex_expenditures, self._oil_asr_expenditures
        #     ]
        # )
        #
        # gas_total_expenditures = reduce(
        #     lambda x, y: x + y,
        #     [
        #         self._gas_capital_expenditures, self._gas_intangible_expenditures,
        #         self._gas_opex_expenditures, self._gas_asr_expenditures
        #     ]
        # )
        #
        # # Configure base cashflow
        # self._oil_cashflow = self._oil_revenue - (self._oil_capital_expenditures +
        #                                           self._oil_intangible_expenditures +
        #                                           self._oil_opex_expenditures +
        #                                           self._oil_asr_expenditures)
        #
        # self._gas_cashflow = self._gas_revenue - (self._gas_capital_expenditures +
        #                                           self._gas_intangible_expenditures +
        #                                           self._gas_opex_expenditures +
        #                                           self._gas_asr_expenditures)
        #
        # self._consolidated_cashflow = self._oil_cashflow + self._gas_cashflow
        # self._consolidated_sunk_cost = self._oil_sunk_cost + self._gas_sunk_cost
        # self._consolidated_government_take = np.zeros_like(self._consolidated_cashflow)
        # self._consolidated_revenue = self._oil_revenue + self._gas_revenue

    def __len__(self):
        return self.project_duration

    def __eq__(self, other):
        # Between two instances of BaseProject
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
                    tangible_self == tangible_other,
                    intangible_self == intangible_other,
                    opex_self == opex_other,
                    asr_self == asr_other,
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

            expense_self = reduce(
                lambda x, y: x + y,
                [tangible_self, intangible_self, opex_self, asr_self]
            )
            expense_other = reduce(
                lambda x, y: x + y,
                [tangible_other, intangible_other, opex_other, asr_other]
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

            expense_self = reduce(
                lambda x, y: x + y,
                [tangible_self, intangible_self, opex_self, asr_self]
            )
            expense_other = reduce(
                lambda x, y: x + y,
                [tangible_other, intangible_other, opex_other, asr_other]
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

            expense_self = reduce(
                lambda x, y: x + y,
                [tangible_self, intangible_self, opex_self, asr_self]
            )
            expense_other = reduce(
                lambda x, y: x + y,
                [tangible_other, intangible_other, opex_other, asr_other]
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

            expense_self = reduce(
                lambda x, y: x + y,
                [tangible_self, intangible_self, opex_self, asr_self]
            )
            expense_other = reduce(
                lambda x, y: x + y,
                [tangible_other, intangible_other, opex_other, asr_other]
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
