"""
Configure base project as a base framework for PSC contract.
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import date
from functools import reduce
from typing import Callable

from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.selection import (
    FluidType,
    ExpendituresType,
    TaxRegime,
    OtherRevenue,
    InflationAppliedTo,
    SunkCostInvestmentType,
    DeprMethod
)
from pyscnomics.econ.costs import (
    CapitalCost,
    Intangible,
    OPEX,
    ASR,
    LBT,
    CostOfSales,
    SunkCost
)
# from pyscnomics.econ.results import CashFlow


class SunkCostException(Exception):
    """ Exception to be raised for a misuse of class SunkCost"""

    pass


class BaseProjectException(Exception):
    """ Exception to be raised for a misuse of BaseProject class """

    pass


class OtherRevenueException(Exception):
    """ Exception to be raised for a misuse of Other Revenue """

    pass


@dataclass
class BaseProject:
    """
    Represents a base project with start and end dates, lifting information,
    capital and intangible costs, operational expenses (OPEX), ASR costs,
    LBT costs, Cost of Sales, and Sunk Costs.

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
    sunk_cost: tuple[SunkCost]
        A tuple of SunkCost objects. Defaults to None.
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
    cost_of_sales: tuple[CostOfSales] = field(default=None)
    sunk_cost: tuple[SunkCost, ...] = field(default=None)

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
    sunk_cost_total: SunkCost = field(default=None, init=False, repr=False)

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
    _oil_sunk_cost: SunkCost = field(default=None, init=False, repr=False)
    _gas_sunk_cost: SunkCost = field(default=None, init=False, repr=False)

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
    _oil_opex_expenditures_pre_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _gas_opex_expenditures_pre_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _oil_asr_expenditures_pre_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _gas_asr_expenditures_pre_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _oil_lbt_expenditures_pre_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _gas_lbt_expenditures_pre_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _oil_cost_of_sales_expenditures_pre_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _gas_cost_of_sales_expenditures_pre_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )

    # Attributes to be defined later
    # (Associated with indirect taxes for each cost element)
    _oil_capital_indirect_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _gas_capital_indirect_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _oil_intangible_indirect_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _gas_intangible_indirect_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _oil_opex_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_opex_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    _oil_asr_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_asr_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    _oil_lbt_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    _gas_lbt_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    _oil_cost_of_sales_indirect_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _gas_cost_of_sales_indirect_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )

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
    _oil_opex_expenditures_post_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _gas_opex_expenditures_post_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _oil_asr_expenditures_post_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _gas_asr_expenditures_post_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _oil_lbt_expenditures_post_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _gas_lbt_expenditures_post_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _oil_cost_of_sales_expenditures_post_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _gas_cost_of_sales_expenditures_post_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )

    # Attributes to be defined later
    # (Associated with total expenditures and indirect taxes for each fluid)
    _oil_total_expenditures_pre_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _gas_total_expenditures_pre_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _oil_total_indirect_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _gas_total_indirect_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _oil_total_expenditures_post_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _gas_total_expenditures_post_tax: np.ndarray = field(
        default=None, init=False, repr=False
    )

    # Attributes to be defined later
    # Associated with investment_type and investment config
    _investment_type_list: list[str] = field(default=None, init=False, repr=False)
    _investment_config_list: list[SunkCostInvestmentType] = field(
        default=None, init=False, repr=False
    )

    # Attributes to be defined later
    # Associated with sunk cost and preonstream cost
    _oil_sunk_cost_array: dict = field(default=None, init=False, repr=False)
    _gas_sunk_cost_array: dict = field(default=None, init=False, repr=False)
    _oil_preonstream_cost_array: dict = field(default=None, init=False, repr=False)
    _gas_preonstream_cost_array: dict = field(default=None, init=False, repr=False)
    _oil_sunk_cost_bulk: dict = field(default=None, init=False, repr=False)
    _gas_sunk_cost_bulk: dict = field(default=None, init=False, repr=False)
    _oil_preonstream_cost_bulk: dict = field(default=None, init=False, repr=False)
    _gas_preonstream_cost_bulk: dict = field(default=None, init=False, repr=False)
    _oil_sunk_cost_amortization_charge: dict = field(
        default=None, init=False, repr=False
    )
    _gas_sunk_cost_amortization_charge: dict = field(
        default=None, init=False, repr=False
    )
    _oil_preonstream_cost_amortization_charge: dict = field(
        default=None, init=False, repr=False
    )
    _gas_preonstream_cost_amortization_charge: dict = field(
        default=None, init=False, repr=False
    )
    _oil_sunk_cost_tangible_depreciation_charge: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _gas_sunk_cost_tangible_depreciation_charge: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _oil_sunk_cost_tangible_undepreciated_asset: float = field(
        default=None, init=False, repr=False
    )
    _gas_sunk_cost_tangible_undepreciated_asset: float = field(
        default=None, init=False, repr=False
    )
    _oil_preonstream_cost_tangible_depreciation_charge: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _gas_preonstream_cost_tangible_depreciation_charge: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _oil_preonstream_cost_tangible_undepreciated_asset: float = field(
        default=None, init=False, repr=False
    )
    _gas_preonstream_cost_tangible_undepreciated_asset: float = field(
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

    # Attributes to be defined later
    # (associated with consolidated sunk cost and preonstream cost)
    _consolidated_sunk_cost_array: dict = field(default=None, init=False, repr=False)
    _consolidated_preonstream_cost_array: dict = field(default=None, init=False, repr=False)
    _consolidated_sunk_cost_bulk: dict = field(default=None, init=False, repr=False)
    _consolidated_preonstream_cost_bulk: dict = field(default=None, init=False, repr=False)
    _consolidated_sunk_cost_amortization_charge: dict = field(
        default=None, init=False, repr=False
    )
    _consolidated_preonstream_cost_amortization_charge: dict = field(
        default=None, init=False, repr=False
    )
    _consolidated_sunk_cost_tangible_depreciation_charge: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_preonstream_cost_tangible_depreciation_charge: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_sunk_cost_tangible_undepreciated_asset: float = field(
        default=None, init=False, repr=False
    )
    _consolidated_preonstream_cost_tangible_undepreciated_asset: float = field(
        default=None, init=False, repr=False
    )

    # Attributes to be defined later (associated with consolidated profiles)
    _consolidated_revenue: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_cashflow: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_government_take: np.ndarray = field(
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

        # Prepare attributes _investment_type_list and _investment_config_list
        self._investment_type_list = ["Tangible", "Intangible"]
        self._investment_config_list = [
            SunkCostInvestmentType.TANGIBLE, SunkCostInvestmentType.INTANGIBLE
        ]

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

                oil_onstream_index = int(
                    np.argwhere(
                        self.oil_onstream_date.year == self.project_years
                    ).ravel()
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

                gas_onstream_index = int(
                    np.argwhere(
                        self.gas_onstream_date.year == self.project_years
                    ).ravel()
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

        # Prepare attribute sunk_cost (for both OIL and GAS)
        if self.sunk_cost is None:
            self.sunk_cost = (
                SunkCost(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    pod1_year=self.start_date.year,
                    expense_year=np.array([self.start_date.year]),
                    cost=np.array([0]),
                    cost_allocation=[FluidType.OIL],
                    investment_type=[SunkCostInvestmentType.TANGIBLE],
                ),
                SunkCost(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    pod1_year=self.start_date.year,
                    expense_year=np.array([self.start_date.year]),
                    cost=np.array([0]),
                    cost_allocation=[FluidType.OIL],
                    investment_type=[SunkCostInvestmentType.INTANGIBLE],
                ),
                SunkCost(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    pod1_year=self.start_date.year,
                    expense_year=np.array([self.start_date.year]),
                    cost=np.array([0]),
                    cost_allocation=[FluidType.GAS],
                    investment_type=[SunkCostInvestmentType.TANGIBLE],
                ),
                SunkCost(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    pod1_year=self.start_date.year,
                    expense_year=np.array([self.start_date.year]),
                    cost=np.array([0]),
                    cost_allocation=[FluidType.GAS],
                    investment_type=[SunkCostInvestmentType.INTANGIBLE],
                )
            )

        else:
            if not isinstance(self.sunk_cost, tuple):
                raise BaseProjectException(
                    f"Attribute sunk_cost must be provided as a tuple of SunkCost "
                    f"instances, not as an/a {self.sunk_cost.__class__.__qualname__}"
                )

        # Prepare attributes associated with total cost per component
        self.capital_cost_total = reduce(lambda x, y: x + y, self.capital_cost)
        self.intangible_cost_total = reduce(lambda x, y: x + y, self.intangible_cost)
        self.opex_total = reduce(lambda x, y: x + y, self.opex)
        self.asr_cost_total = reduce(lambda x, y: x + y, self.asr_cost)
        self.lbt_cost_total = reduce(lambda x, y: x + y, self.lbt_cost)
        self.cost_of_sales_total = reduce(lambda x, y: x + y, self.cost_of_sales)
        self.sunk_cost_total = reduce(lambda x, y: x + y, self.sunk_cost)

        # Prepare attribute onstream_year of `sunk_cost_total`
        self.sunk_cost_total.onstream_year = min(
            self.oil_onstream_date.year, self.gas_onstream_date.year
        )

        if not isinstance(self.sunk_cost_total.onstream_year, int):
            raise SunkCostException(
                f"Attribute onstream_year must be provided as an int, not as an/a "
                f"{self.sunk_cost_total.onstream_year.__class__.__qualname__}"
            )

        if self.sunk_cost_total.onstream_year < self.start_date.year:
            raise SunkCostException(
                f"Onstream year ({self.sunk_cost_total.onstream_year}) is before "
                f"the start year of the project ({self.start_date.year})"
            )

        if self.sunk_cost_total.onstream_year > self.end_date.year:
            raise SunkCostException(
                f"Onstream year ({self.sunk_cost_total.onstream_year}) is after "
                f"the end year of the project ({self.end_date.year})"
            )

        if self.sunk_cost_total.onstream_year < self.sunk_cost_total.pod1_year:
            raise SunkCostException(
                f"POD I year ({self.sunk_cost_total.pod1_year}) is after the "
                f"onstream year ({self.sunk_cost_total.onstream_year})"
            )

        # Prepare attribute expense_year of `sunk_cost_total`
        sc_expense_year_large_sum = np.sum(
            self.sunk_cost_total.expense_year > self.sunk_cost_total.onstream_year
        )

        if sc_expense_year_large_sum > 0:
            raise SunkCostException(
                f"Cannot accept attribute expense_year larger than onstream_year "
                f"in an instance of SunkCost: "
                f"onstream_year: ({self.sunk_cost_total.onstream_year}), "
                f"expense_year: ({self.sunk_cost_total.expense_year}) "
            )

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
        self._oil_sunk_cost = self._get_oil_sunk_cost()
        self._gas_sunk_cost = self._get_gas_sunk_cost()

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
                self._oil_sunk_cost.start_year,
                self._gas_sunk_cost.start_year,
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
                f"Oil sunk cost ({self._oil_sunk_cost.start_year}), "
                f"Gas sunk cost ({self._gas_sunk_cost.start_year}). "
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
                self._oil_sunk_cost.end_year,
                self._gas_sunk_cost.end_year,
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
                f"Oil sunk cost ({self._oil_sunk_cost.end_year}), "
                f"Gas sunk cost ({self._gas_sunk_cost.end_year}). "
            )

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
                prod_year=self.project_years,
                lifting_rate=np.zeros(self.project_duration),
                price=np.zeros(self.project_duration),
                fluid_type=FluidType.OIL,
                prod_rate=np.zeros(self.project_duration),
                prod_rate_baseline=np.zeros(self.project_duration),
            )

        return reduce(
            lambda x, y: x + y,
            (lft for lft in self.lifting if lft.fluid_type == FluidType.OIL)
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
                prod_year=self.project_years,
                lifting_rate=np.zeros(self.project_duration),
                price=np.zeros(self.project_duration),
                fluid_type=FluidType.GAS,
                prod_rate=np.zeros(self.project_duration),
                prod_rate_baseline=np.zeros(self.project_duration),
            )

        return reduce(
            lambda x, y: x + y,
            (lft for lft in self.lifting if lft.fluid_type == FluidType.GAS)
        )

    def _get_sulfur_lifting(self) -> Lifting:
        """
        Determines total sulfur Lifting from the number of sulfur Lifting instances
        in attribute self.lifting.

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
                prod_year=self.project_years,
                lifting_rate=np.zeros(self.project_duration),
                price=np.zeros(self.project_duration),
                fluid_type=FluidType.SULFUR,
                prod_rate=np.zeros(self.project_duration),
                prod_rate_baseline=np.zeros(self.project_duration),
            )

        return reduce(
            lambda x, y: x + y,
            (lft for lft in self.lifting if lft.fluid_type == FluidType.SULFUR)
        )

    def _get_electricity_lifting(self) -> Lifting:
        """
        Determines total ELECTRICITY Lifting from the number of ELECTRICITY Lifting
        instances in attribute self.lifting.

        Returns
        -------
        total_electricity_lifting: Lifting
            Total ELECTRICITY Lifting as a new instance of ELECTRICITY Lifting where
            the associated attributes are determined based on the prescribed rule in
            the corresponding dunder method __add__() of the Lifting class.

        Notes
        -----
        The order of operations is as follows:
        (1) Check the fluid_type in attribute self.lifting,
        (2) If ELECTRICITY is not available as an instance in attribute self.lifting,
            then establish a new instance of ELECTRICITY Lifting with the following
            attributes set to zero: lifting_rate, price.
        (3) Add the instances with fluid_type ELECTRICITY, return the result as a new
            instance of ELECTRICITY Lifting following the rule prescribed in the dunder
            method __add__() of class Lifting.
        """

        fluid_types = [lft.fluid_type for lft in self.lifting]

        if FluidType.ELECTRICITY not in fluid_types:
            return Lifting(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                prod_year=self.project_years,
                lifting_rate=np.zeros(self.project_duration),
                price=np.zeros(self.project_duration),
                fluid_type=FluidType.ELECTRICITY,
                prod_rate=np.zeros(self.project_duration),
                prod_rate_baseline=np.zeros(self.project_duration),
            )

        return reduce(
            lambda x, y: x + y,
            (lft for lft in self.lifting if lft.fluid_type == FluidType.ELECTRICITY)
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
            establish a new instance of CO2 Lifting with the following attributes
            set to zero: lifting_rate, price.
        (3) Add the instances with fluid_type CO2, return the result as a new instance
            of CO2 Lifting following the rule prescribed in the dunder method __add__()
            of class Lifting.
        """

        fluid_types = [lft.fluid_type for lft in self.lifting]

        if FluidType.CO2 not in fluid_types:
            return Lifting(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                prod_year=self.project_years,
                lifting_rate=np.zeros(self.project_duration),
                price=np.zeros(self.project_duration),
                fluid_type=FluidType.CO2,
                prod_rate=np.zeros(self.project_duration),
                prod_rate_baseline=np.zeros(self.project_duration),
            )

        return reduce(
            lambda x, y: x + y,
            (lft for lft in self.lifting if lft.fluid_type == FluidType.CO2)
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
                expense_year=np.array([self.start_date.year]),
                cost=np.array([0]),
                cost_allocation=[FluidType.OIL],
            )

        else:
            oil_capital_id = np.argwhere(
                np.array(self.capital_cost_total.cost_allocation) == FluidType.OIL
            ).ravel()

            start_year = self.capital_cost_total.start_year
            end_year = self.capital_cost_total.end_year
            expense_year = self.capital_cost_total.expense_year[oil_capital_id]
            cost = self.capital_cost_total.cost[oil_capital_id]
            cost_allocation = np.array(self.capital_cost_total.cost_allocation)[oil_capital_id]
            description = np.array(self.capital_cost_total.description)[oil_capital_id]
            tax_portion = self.capital_cost_total.tax_portion[oil_capital_id]
            tax_discount = self.capital_cost_total.tax_discount[oil_capital_id]
            pis_year = self.capital_cost_total.pis_year[oil_capital_id]
            salvage_value = self.capital_cost_total.salvage_value[oil_capital_id]
            useful_life = self.capital_cost_total.useful_life[oil_capital_id]
            depreciation_factor = self.capital_cost_total.depreciation_factor[oil_capital_id]
            is_ic_applied = np.array(self.capital_cost_total.is_ic_applied)[oil_capital_id]

            return CapitalCost(
                start_year=start_year,
                end_year=end_year,
                expense_year=expense_year,
                cost=cost,
                cost_allocation=cost_allocation.tolist(),
                description=description.tolist(),
                tax_portion=tax_portion,
                tax_discount=tax_discount,
                pis_year=pis_year,
                salvage_value=salvage_value,
                useful_life=useful_life,
                depreciation_factor=depreciation_factor,
                is_ic_applied=is_ic_applied.tolist(),
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
                expense_year=np.array([self.start_date.year]),
                cost=np.array([0]),
                cost_allocation=[FluidType.GAS],
            )

        else:
            gas_capital_id = np.argwhere(
                np.array(self.capital_cost_total.cost_allocation) == FluidType.GAS
            ).ravel()

            start_year = self.capital_cost_total.start_year
            end_year = self.capital_cost_total.end_year
            expense_year = self.capital_cost_total.expense_year[gas_capital_id]
            cost = self.capital_cost_total.cost[gas_capital_id]
            cost_allocation = np.array(self.capital_cost_total.cost_allocation)[gas_capital_id]
            description = np.array(self.capital_cost_total.description)[gas_capital_id]
            tax_portion = self.capital_cost_total.tax_portion[gas_capital_id]
            tax_discount = self.capital_cost_total.tax_discount[gas_capital_id]
            pis_year = self.capital_cost_total.pis_year[gas_capital_id]
            salvage_value = self.capital_cost_total.salvage_value[gas_capital_id]
            useful_life = self.capital_cost_total.useful_life[gas_capital_id]
            depreciation_factor = self.capital_cost_total.depreciation_factor[gas_capital_id]
            is_ic_applied = np.array(self.capital_cost_total.is_ic_applied)[gas_capital_id]

            return CapitalCost(
                start_year=start_year,
                end_year=end_year,
                expense_year=expense_year,
                cost=cost,
                cost_allocation=cost_allocation.tolist(),
                description=description.tolist(),
                tax_portion=tax_portion,
                tax_discount=tax_discount,
                pis_year=pis_year,
                salvage_value=salvage_value,
                useful_life=useful_life,
                depreciation_factor=depreciation_factor,
                is_ic_applied=is_ic_applied.tolist(),
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
                expense_year=np.array([self.start_date.year]),
                cost=np.array([0]),
                cost_allocation=[FluidType.OIL],
            )

        else:
            oil_intangible_id = np.argwhere(
                np.array(self.intangible_cost_total.cost_allocation) == FluidType.OIL
            ).ravel()

            start_year = self.intangible_cost_total.start_year
            end_year = self.intangible_cost_total.end_year
            expense_year = self.intangible_cost_total.expense_year[oil_intangible_id]
            cost = self.intangible_cost_total.cost[oil_intangible_id]
            cost_allocation = np.array(
                self.intangible_cost_total.cost_allocation
            )[oil_intangible_id]
            description = np.array(self.intangible_cost_total.description)[oil_intangible_id]
            tax_portion = self.intangible_cost_total.tax_portion[oil_intangible_id]
            tax_discount = self.intangible_cost_total.tax_discount[oil_intangible_id]

            return Intangible(
                start_year=start_year,
                end_year=end_year,
                expense_year=expense_year,
                cost=cost,
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
                expense_year=np.array([self.start_date.year]),
                cost=np.array([0]),
                cost_allocation=[FluidType.GAS],
            )

        else:
            gas_intangible_id = np.argwhere(
                np.array(self.intangible_cost_total.cost_allocation) == FluidType.GAS
            ).ravel()

            start_year = self.intangible_cost_total.start_year
            end_year = self.intangible_cost_total.end_year
            expense_year = self.intangible_cost_total.expense_year[gas_intangible_id]
            cost = self.intangible_cost_total.cost[gas_intangible_id]
            cost_allocation = np.array(self.intangible_cost_total.cost_allocation)[gas_intangible_id]
            description = np.array(self.intangible_cost_total.description)[gas_intangible_id]
            tax_portion = self.intangible_cost_total.tax_portion[gas_intangible_id]
            tax_discount = self.intangible_cost_total.tax_discount[gas_intangible_id]

            return Intangible(
                start_year=start_year,
                end_year=end_year,
                expense_year=expense_year,
                cost=cost,
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
                expense_year=np.array([self.start_date.year]),
                fixed_cost=np.array([0]),
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
            tax_portion = self.opex_total.tax_portion[oil_opex_id]
            tax_discount = self.opex_total.tax_discount[oil_opex_id]
            fixed_cost = self.opex_total.fixed_cost[oil_opex_id]
            prod_rate = self.opex_total.prod_rate[oil_opex_id]
            cost_per_volume = self.opex_total.cost_per_volume[oil_opex_id]

            return OPEX(
                start_year=start_year,
                end_year=end_year,
                expense_year=expense_year,
                cost_allocation=cost_allocation.tolist(),
                description=description.tolist(),
                tax_portion=tax_portion,
                tax_discount=tax_discount,
                fixed_cost=fixed_cost,
                prod_rate=prod_rate,
                cost_per_volume=cost_per_volume,
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
                expense_year=np.array([self.start_date.year]),
                fixed_cost=np.array([0]),
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
            tax_portion = self.opex_total.tax_portion[gas_opex_id]
            tax_discount = self.opex_total.tax_discount[gas_opex_id]
            fixed_cost = self.opex_total.fixed_cost[gas_opex_id]
            prod_rate = self.opex_total.prod_rate[gas_opex_id]
            cost_per_volume = self.opex_total.cost_per_volume[gas_opex_id]

            return OPEX(
                start_year=start_year,
                end_year=end_year,
                expense_year=expense_year,
                cost_allocation=cost_allocation.tolist(),
                description=description.tolist(),
                tax_portion=tax_portion,
                tax_discount=tax_discount,
                fixed_cost=fixed_cost,
                prod_rate=prod_rate,
                cost_per_volume=cost_per_volume,
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
                expense_year=np.array([self.start_date.year]),
                cost=np.array([0]),
                cost_allocation=[FluidType.OIL]
            )

        else:
            oil_asr_id = np.argwhere(
                np.array(self.asr_cost_total.cost_allocation) == FluidType.OIL
            ).ravel()

            start_year = self.asr_cost_total.start_year
            end_year = self.asr_cost_total.end_year
            expense_year = self.asr_cost_total.expense_year[oil_asr_id]
            cost = self.asr_cost_total.cost[oil_asr_id]
            cost_allocation = np.array(self.asr_cost_total.cost_allocation)[oil_asr_id]
            description = np.array(self.asr_cost_total.description)[oil_asr_id]
            tax_portion = self.asr_cost_total.tax_portion[oil_asr_id]
            tax_discount = self.asr_cost_total.tax_discount[oil_asr_id]
            final_year = self.asr_cost_total.final_year[oil_asr_id]
            future_rate = self.asr_cost_total.future_rate[oil_asr_id]

            return ASR(
                start_year=start_year,
                end_year=end_year,
                expense_year=expense_year,
                cost=cost,
                cost_allocation=cost_allocation.tolist(),
                description=description.tolist(),
                tax_portion=tax_portion,
                tax_discount=tax_discount,
                final_year=final_year,
                future_rate=future_rate,
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
                expense_year=np.array([self.start_date.year]),
                cost=np.array([0]),
                cost_allocation=[FluidType.GAS],
            )

        else:
            gas_asr_id = np.argwhere(
                np.array(self.asr_cost_total.cost_allocation) == FluidType.GAS
            ).ravel()

            start_year = self.asr_cost_total.start_year
            end_year = self.asr_cost_total.end_year
            expense_year = self.asr_cost_total.expense_year[gas_asr_id]
            cost = self.asr_cost_total.cost[gas_asr_id]
            cost_allocation = np.array(self.asr_cost_total.cost_allocation)[gas_asr_id]
            description = np.array(self.asr_cost_total.description)[gas_asr_id]
            tax_portion = self.asr_cost_total.tax_portion[gas_asr_id]
            tax_discount = self.asr_cost_total.tax_discount[gas_asr_id]
            final_year = self.asr_cost_total.final_year[gas_asr_id]
            future_rate = self.asr_cost_total.future_rate[gas_asr_id]

            return ASR(
                start_year=start_year,
                end_year=end_year,
                expense_year=expense_year,
                cost=cost,
                cost_allocation=cost_allocation.tolist(),
                description=description.tolist(),
                tax_portion=tax_portion,
                tax_discount=tax_discount,
                final_year=final_year,
                future_rate=future_rate,
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
            tax_portion = self.lbt_cost_total.tax_portion[oil_lbt_id]
            tax_discount = self.lbt_cost_total.tax_discount[oil_lbt_id]
            final_year = self.lbt_cost_total.final_year[oil_lbt_id]
            utilized_land_area = self.lbt_cost_total.utilized_land_area[oil_lbt_id]
            utilized_building_area = self.lbt_cost_total.utilized_building_area[oil_lbt_id]
            njop_land = self.lbt_cost_total.njop_land[oil_lbt_id]
            njop_building = self.lbt_cost_total.njop_building[oil_lbt_id]
            gross_revenue = self.lbt_cost_total.gross_revenue[oil_lbt_id]
            cost = self.lbt_cost_total.cost[oil_lbt_id]

            return LBT(
                start_year=start_year,
                end_year=end_year,
                expense_year=expense_year,
                cost_allocation=cost_allocation.tolist(),
                description=description.tolist(),
                tax_portion=tax_portion,
                tax_discount=tax_discount,
                final_year=final_year,
                utilized_land_area=utilized_land_area,
                utilized_building_area=utilized_building_area,
                njop_land=njop_land,
                njop_building=njop_building,
                gross_revenue=gross_revenue,
                cost=cost,
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
                expense_year=[self.start_date.year],
                cost_allocation=[FluidType.GAS]
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
            tax_portion = self.lbt_cost_total.tax_portion[gas_lbt_id]
            tax_discount = self.lbt_cost_total.tax_discount[gas_lbt_id]
            final_year = self.lbt_cost_total.final_year[gas_lbt_id]
            utilized_land_area = self.lbt_cost_total.utilized_land_area[gas_lbt_id]
            utilized_building_area = self.lbt_cost_total.utilized_building_area[gas_lbt_id]
            njop_land = self.lbt_cost_total.njop_land[gas_lbt_id]
            njop_building = self.lbt_cost_total.njop_building[gas_lbt_id]
            gross_revenue = self.lbt_cost_total.gross_revenue[gas_lbt_id]
            cost = self.lbt_cost_total.cost[gas_lbt_id]

            return LBT(
                start_year=start_year,
                end_year=end_year,
                expense_year=expense_year,
                cost_allocation=cost_allocation.tolist(),
                description=description.tolist(),
                tax_portion=tax_portion,
                tax_discount=tax_discount,
                final_year=final_year,
                utilized_land_area=utilized_land_area,
                utilized_building_area=utilized_building_area,
                njop_land=njop_land,
                njop_building=njop_building,
                gross_revenue=gross_revenue,
                cost=cost,
            )

    def _get_oil_cost_of_sales(self) -> CostOfSales:
        """
        Retrieve the oil cost of sales from the total cost of sales data.

        If oil-related costs are not found in the cost allocation, returns a default
        `CostOfSales` instance with zero cost. Otherwise, extracts and returns the
        relevant cost details for oil from `cost_of_sales_total`.

        Returns
        -------
        CostOfSales
            An instance of `CostOfSales` containing the cost of sales data for oil,
            including start and end years, expense year, cost, cost allocation,
            description, tax portion, and tax discount.

        Notes
        -----
        - The method checks if `FluidType.OIL` exists in `cost_of_sales_total.cost_allocation`.
        - If it does not exist, a default `CostOfSales` with zero cost is returned.
        - If it exists, relevant attributes are extracted using `np.argwhere` and returned.
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
            cost_allocation = np.array(
                self.cost_of_sales_total.cost_allocation
            )[oil_cost_of_sales_id]
            description = np.array(
                self.cost_of_sales_total.description
            )[oil_cost_of_sales_id]
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
        Retrieve the gas cost of sales from the total cost of sales data.

        If gas-related costs are not found in the cost allocation, returns a default
        `CostOfSales` instance with zero cost. Otherwise, extracts and returns the
        relevant cost details for gas from `cost_of_sales_total`.

        Returns
        -------
        CostOfSales
            An instance of `CostOfSales` containing the cost of sales data for gas,
            including start and end years, expense year, cost, cost allocation,
            description, tax portion, and tax discount.

        Notes
        -----
        - The method checks if `FluidType.GAS` exists in `cost_of_sales_total.cost_allocation`.
        - If it does not exist, a default `CostOfSales` with zero cost is returned.
        - If it exists, relevant attributes are extracted using `np.argwhere` and returned.
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
            cost_allocation = np.array(
                self.cost_of_sales_total.cost_allocation
            )[gas_cost_of_sales_id]
            description = np.array(
                self.cost_of_sales_total.description
            )[gas_cost_of_sales_id]
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

    def _get_oil_sunk_cost(self) -> SunkCost:
        """
        Retrieve or construct the sunk costs associated with oil production.

        This method returns a SunkCost object containing either:
        - A zero-cost allocation for oil if no oil costs exist in the total sunk costs
        - The subset of sunk costs specifically allocated to oil production

        Returns
        -------
        SunkCost
            A SunkCost object containing:
            - Basic timeline information (start_year, end_year, etc.)
            - Cost and expense year arrays
            - Allocation and type information
            - Tax and depreciation parameters

            If no oil allocation exists, returns a minimal SunkCost with zero cost
            for the project timeline.

        Notes
        -----
        - The returned SunkCost will only contain oil-related costs, even if the
          total sunk costs include other fluid types.
        - All array fields in the returned object will be filtered to only include
          oil-related entries when oil costs exist in the total sunk costs.
        - When no oil costs exist, a single zero-cost entry is returned with the
          project's start year as the expense year.
        """
        if FluidType.OIL not in self.sunk_cost_total.cost_allocation:
            return SunkCost(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                onstream_year=self.start_date.year,
                pod1_year=self.start_date.year,
                expense_year=np.array([self.start_date.year]),
                cost=np.array([0]),
                cost_allocation=[FluidType.OIL],
            )

        else:
            oil_sunk_cost_id = np.argwhere(
                np.array(self.sunk_cost_total.cost_allocation) == FluidType.OIL
            ).ravel()

            start_year = self.sunk_cost_total.start_year
            end_year = self.sunk_cost_total.end_year
            onstream_year = self.sunk_cost_total.onstream_year
            pod1_year = self.sunk_cost_total.pod1_year
            expense_year = self.sunk_cost_total.expense_year[oil_sunk_cost_id]
            cost = self.sunk_cost_total.cost[oil_sunk_cost_id]
            salvage_value = self.sunk_cost_total.salvage_value[oil_sunk_cost_id]
            depreciation_period = self.sunk_cost_total.depreciation_period[oil_sunk_cost_id]
            depreciation_factor = self.sunk_cost_total.depreciation_factor[oil_sunk_cost_id]
            cost_allocation = np.array(self.sunk_cost_total.cost_allocation)[oil_sunk_cost_id]
            investment_type = np.array(self.sunk_cost_total.investment_type)[oil_sunk_cost_id]
            description = np.array(self.sunk_cost_total.description)[oil_sunk_cost_id]
            tax_portion = self.sunk_cost_total.tax_portion[oil_sunk_cost_id]
            tax_discount = self.sunk_cost_total.tax_discount[oil_sunk_cost_id]

            return SunkCost(
                start_year=start_year,
                end_year=end_year,
                onstream_year=onstream_year,
                pod1_year=pod1_year,
                expense_year=expense_year,
                cost=cost,
                salvage_value=salvage_value,
                depreciation_period=depreciation_period,
                depreciation_factor=depreciation_factor,
                cost_allocation=cost_allocation.tolist(),
                investment_type=investment_type.tolist(),
                description=description.tolist(),
                tax_portion=tax_portion,
                tax_discount=tax_discount,
            )

    def _get_gas_sunk_cost(self) -> SunkCost:
        """
        Retrieve or construct the sunk costs associated with gas production.

        This method returns a `SunkCost` object containing either:
        - A zero-cost allocation for gas if no gas costs exist in the total sunk costs.
        - The subset of sunk costs specifically allocated to gas production.

        Returns
        -------
        SunkCost
            A `SunkCost` object containing:
            - Timeline attributes (`start_year`, `end_year`, `onstream_year`, `pod1_year`).
            - Cost-related arrays (`expense_year`, `cost`, `salvage_value`).
            - Depreciation parameters (`depreciation_period`, `depreciation_factor`).
            - Allocation metadata (`cost_allocation`, `investment_type`, `description`).
            - Tax-related fields (`tax_portion`, `tax_discount`).

            If no gas allocation exists, returns a minimal `SunkCost` with zero cost
            for the project timeline.

        Notes
        -----
        - The returned `SunkCost` will only contain gas-related costs, even if the
          total sunk costs include other fluid types (e.g., oil).
        - All array fields (e.g., `cost`, `expense_year`) are filtered to include only
          entries where `cost_allocation == FluidType.GAS`.
        - If no gas costs are present, the returned object will have:
            - A single `expense_year` set to the project's `start_date.year`.
            - A zero `cost` array.
            - `cost_allocation = [FluidType.GAS]`.
        """
        if FluidType.GAS not in self.sunk_cost_total.cost_allocation:
            return SunkCost(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                onstream_year=self.start_date.year,
                pod1_year=self.start_date.year,
                expense_year=np.array([self.start_date.year]),
                cost=np.array([0]),
                cost_allocation=[FluidType.GAS],
            )

        else:
            gas_sunk_cost_id = np.argwhere(
                np.array(self.sunk_cost_total.cost_allocation) == FluidType.GAS
            ).ravel()

            start_year = self.sunk_cost_total.start_year
            end_year = self.sunk_cost_total.end_year
            onstream_year = self.sunk_cost_total.onstream_year
            pod1_year = self.sunk_cost_total.pod1_year
            expense_year = self.sunk_cost_total.expense_year[gas_sunk_cost_id]
            cost = self.sunk_cost_total.cost[gas_sunk_cost_id]
            salvage_value = self.sunk_cost_total.salvage_value[gas_sunk_cost_id]
            depreciation_period = self.sunk_cost_total.depreciation_period[gas_sunk_cost_id]
            depreciation_factor = self.sunk_cost_total.depreciation_factor[gas_sunk_cost_id]
            cost_allocation = np.array(self.sunk_cost_total.cost_allocation)[gas_sunk_cost_id]
            investment_type = np.array(self.sunk_cost_total.investment_type)[gas_sunk_cost_id]
            description = np.array(self.sunk_cost_total.description)[gas_sunk_cost_id]
            tax_portion = self.sunk_cost_total.tax_portion[gas_sunk_cost_id]
            tax_discount = self.sunk_cost_total.tax_discount[gas_sunk_cost_id]

            return SunkCost(
                start_year=start_year,
                end_year=end_year,
                onstream_year=onstream_year,
                pod1_year=pod1_year,
                expense_year=expense_year,
                cost=cost,
                salvage_value=salvage_value,
                depreciation_period=depreciation_period,
                depreciation_factor=depreciation_factor,
                cost_allocation=cost_allocation.tolist(),
                investment_type=investment_type.tolist(),
                description=description.tolist(),
                tax_portion=tax_portion,
                tax_discount=tax_discount,
            )

    def _calc_sunk_cost_array(
        self,
        fluid_type: FluidType,
        cost_obj: SunkCost,
        tax_rate: np.ndarray | float = 0.0,
    ) -> dict:
        """
        Computes the sunk cost arrays for each investment type for a given fluid.

        This method iterates through investment configurations and applies the
        `get_sunk_cost_investment_array` method from the provided `SunkCost` object
        to compute the per-type sunk cost array, considering the specified fluid
        and tax rate.

        Parameters
        ----------
        fluid_type : FluidType
            The fluid type (e.g., `FluidType.OIL` or `FluidType.GAS`) for which the
            sunk costs are being calculated.

        cost_obj : SunkCost
            The SunkCost object containing cost data and methods for retrieving
            investment-specific sunk cost arrays.

        tax_rate : float or np.ndarray, optional
            The applicable tax rate or array of tax rates used in the sunk cost
            computation. Default is 0.0.

        Returns
        -------
        dict
            A dictionary mapping investment type names (as strings) to their
            corresponding sunk cost arrays (typically `np.ndarray`).
        """
        return {
            key: cost_obj.get_sunk_cost_investment_array(
                fluid_type=fluid_type,
                investment_config=config,
                tax_rate=tax_rate,
            )
            for key, config in zip(self._investment_type_list, self._investment_config_list)
        }

    def _calc_preonstream_cost_array(
        self,
        fluid_type: FluidType,
        cost_obj: SunkCost,
        tax_rate: np.ndarray | float = 0.0,
    ) -> dict:
        """
        Computes the pre-onstream cost arrays for each investment type for a given fluid.

        This method applies the `get_preonstream_cost_investment_array` function of the
        given `SunkCost` object to each investment configuration. It returns a dictionary
        that maps each investment type to its corresponding pre-onstream cost array,
        computed for the specified fluid type and tax rate.

        Parameters
        ----------
        fluid_type : FluidType
            The fluid type (e.g., `FluidType.OIL` or `FluidType.GAS`) for which the
            pre-onstream costs are calculated.

        cost_obj : SunkCost
            An instance of the `SunkCost` class containing cost data and methods for
            retrieving investment-specific pre-onstream cost arrays.

        tax_rate : float or np.ndarray, optional
            The applicable tax rate or array of tax rates used in the pre-onstream
            cost calculations. Default is 0.0.

        Returns
        -------
        dict
            A dictionary mapping investment type names (as strings) to their
            pre-onstream cost arrays (`np.ndarray`).
        """
        return {
            key: cost_obj.get_preonstream_cost_investment_array(
                fluid_type=fluid_type,
                investment_config=config,
                tax_rate=tax_rate,
            )
            for key, config in zip(self._investment_type_list, self._investment_config_list)
        }

    def _calc_cost_bulk(
        self,
        cost_obj: SunkCost,
        cost_array: dict,
    ) -> dict:
        """
        Computes bulk investment costs for each investment type using a SunkCost object.

        This method applies the `get_investment_bulk` method of the provided `cost_obj`
        to each investment type's cost array, producing a dictionary of bulk cost values.

        Parameters
        ----------
        cost_obj : SunkCost
            An instance of the `SunkCost` class that provides the `get_investment_bulk`
            method used to compute the bulk cost for each investment type.

        cost_array : np.ndarray or dict-like
            A dictionary or array-like structure that maps each investment type (from
            `self._investment_type_list`) to its associated cost array.

        Returns
        -------
        dict
            A dictionary mapping each investment type to its computed bulk investment cost.
        """
        return {
            key: cost_obj.get_investment_bulk(cost_investment_array=cost_array[key])
            for key in self._investment_type_list
        }

    def _calc_amortization_charge(
        self,
        cost_obj: SunkCost,
        cost_bulk: dict,
        prod_year: np.ndarray,
        prod: np.ndarray,
        salvage_value: float = 0.0,
    ) -> dict:
        """
        Calculate amortization charges for different investment types.

        Parameters
        ----------
        cost_obj : SunkCost
            The cost object containing the amortization calculation method.
        cost_bulk : dict
            Dictionary containing bulk cost data for each investment type.
            Keys should match those in `self._investment_type_list`.
        prod_year : np.ndarray
            Array of production years for which to calculate amortization.
        prod : np.ndarray
            Array of production values corresponding to `prod_year`.
        salvage_value : float, optional
            The salvage value of the asset at end of amortization period.
            Default is 0.0.

        Returns
        -------
        dict
            Dictionary containing amortization charges for each investment type.
            Keys are the same as in `cost_bulk` and `self._investment_type_list`.

        Notes
        -----
        This method delegates the actual amortization calculation to the
        `get_amortization_charge` method of the `cost_obj` for each investment type.
        """
        return {
            key: cost_obj.get_amortization_charge(
                cost_bulk=cost_bulk[key],
                prod_year=prod_year,
                prod=prod,
                salvage_value=salvage_value,
                amortization_len=self.project_duration,
            )
            for key in self._investment_type_list
        }

    @staticmethod
    def _calc_tangible_depreciation_charge(
        cost_mode: Callable,
        fluid_type: FluidType,
        depr_method: DeprMethod = DeprMethod.PSC_DB,
        decline_factor: float | int = 2,
        tax_rate: np.ndarray | float = 0.0,
    ) -> tuple:
        """
        Calculate tangible depreciation charges using a specified cost calculation mode.

        This method serves as a generic wrapper to execute different depreciation calculation
        methods, returning the results as a tuple. The actual computation is delegated to
        the provided cost_mode function.

        Parameters
        ----------
        cost_mode : Callable
            The depreciation calculation function to execute. Expected signature:
            func(fluid_type: FluidType, depr_method: DeprMethod,
                 decline_factor: float | int, tax_rate: np.ndarray | float) -> tuple
        fluid_type : FluidType
            The type of fluid (OIL or GAS) for which to calculate depreciation.
        depr_method : DeprMethod, optional
            The depreciation method to use (e.g., PSC declining balance).
            Default is DeprMethod.PSC_DB.
        decline_factor : float or int, optional
            The decline factor for depreciation calculations.
            Default is 2.
        tax_rate : np.ndarray or float, optional
            Tax rate(s) to apply. Can be a single value or array matching production periods.
            Default is 0.0 (no tax).

        Returns
        -------
        tuple
            A tuple containing depreciation calculation results, typically:
            - np.ndarray: Depreciation charge amounts
            - float: Undepreciated asset values
            Exact contents depend on the cost_mode implementation.

        Notes
        -----
        This method provides a consistent interface for different depreciation calculators
        """
        return cost_mode(
            fluid_type=fluid_type,
            depr_method=depr_method,
            decline_factor=decline_factor,
            tax_rate=tax_rate,
        )

    def _get_sunk_cost_array(self, tax_rate: np.ndarray | float = 0.0) -> None:
        """
        Initializes the sunk cost investment arrays for oil and gas based on
        the specified tax rate.

        This method calculates and assigns the sunk cost arrays for both oil
        and gas by calling `_calc_sunk_cost_array` with the respective `fluid_type`
        and `SunkCost` object. The results are stored in `self._oil_sunk_cost_array`
        and `self._gas_sunk_cost_array`.

        Parameters
        ----------
        tax_rate : float or np.ndarray, optional
            The tax rate(s) applied in the sunk cost calculation.
            Can be a scalar or array of values. Default is 0.0.

        Returns
        -------
        None
            This method does not return anything. It updates internal attributes in-place.

        Attributes Updated
        ------------------
        self._oil_sunk_cost_array : dict
            Dictionary containing sunk cost arrays for oil, keyed by investment type.
        self._gas_sunk_cost_array : dict
            Dictionary containing sunk cost arrays for gas, keyed by investment type.
        """
        self._oil_sunk_cost_array = self._calc_sunk_cost_array(
            fluid_type=FluidType.OIL,
            cost_obj=self._oil_sunk_cost,
            tax_rate=tax_rate,
        )

        self._gas_sunk_cost_array = self._calc_sunk_cost_array(
            fluid_type=FluidType.GAS,
            cost_obj=self._gas_sunk_cost,
            tax_rate=tax_rate,
        )

    def _get_preonstream_cost_array(self, tax_rate: np.ndarray | float = 0.0) -> None:
        """
        Initializes the pre-onstream cost arrays for oil and gas based on
        the provided tax rate.

        This method computes and stores the pre-onstream cost investment arrays
        for both oil and gas by invoking `_calc_preonstream_cost_array` with the
        appropriate `fluid_type` and `SunkCost` object.

        The computed arrays are assigned to `self._oil_preonstream_cost_array`
        and `self._gas_preonstream_cost_array`.

        Parameters
        ----------
        tax_rate : float or np.ndarray, optional
            The tax rate(s) used in the pre-onstream cost calculation.
            Can be a scalar or an array. Default is 0.0.

        Returns
        -------
        None
            This method updates internal attributes and does not return a value.

        Attributes Updated
        ------------------
        self._oil_preonstream_cost_array : dict
            Dictionary of pre-onstream cost arrays for oil, keyed by investment type.
        self._gas_preonstream_cost_array : dict
            Dictionary of pre-onstream cost arrays for gas, keyed by investment type.
        """
        self._oil_preonstream_cost_array = self._calc_preonstream_cost_array(
            fluid_type=FluidType.OIL,
            cost_obj=self._oil_sunk_cost,
            tax_rate=tax_rate,
        )

        self._gas_preonstream_cost_array = self._calc_preonstream_cost_array(
            fluid_type=FluidType.GAS,
            cost_obj=self._gas_sunk_cost,
            tax_rate=tax_rate,
        )

    def _get_sunk_cost_bulk(self) -> None:
        """
        Computes and stores bulk sunk costs for oil and gas.

        This method calculates the total (bulk) sunk investment costs for each
        investment type related to `FluidType.OIL` and `FluidType.GAS`. It uses
        the respective `SunkCost` objects and their associated cost arrays.
        The resulting bulk costs are stored in internal attributes.

        Returns
        -------
        None
            This method updates internal attributes in-place and does not return a value.

        Sets the following instance attributes:
            - `self._oil_sunk_cost_bulk` : dict
            - `self._gas_sunk_cost_bulk` : dict

        Notes
        -----
        This method internally calls `_calc_cost_bulk`, which applies the
        `get_investment_bulk` method of the provided `SunkCost` object to
        each investment type's cost array.
        """
        self._oil_sunk_cost_bulk = self._calc_cost_bulk(
            cost_obj=self._oil_sunk_cost,
            cost_array=self._oil_sunk_cost_array,
        )

        self._gas_sunk_cost_bulk = self._calc_cost_bulk(
            cost_obj=self._gas_sunk_cost,
            cost_array=self._gas_sunk_cost_array,
        )

    def _get_preonstream_cost_bulk(self) -> None:
        """
        Computes and stores bulk pre-onstream costs for oil and gas.

        This method calculates the total (bulk) pre-onstream investment costs for
        each investment type associated with `FluidType.OIL` and `FluidType.GAS`.
        It uses the respective `SunkCost` objects and their corresponding
        pre-onstream cost arrays. The computed bulk costs are stored as instance
        attributes for later use in economic evaluations.

        Returns
        -------
        None
            This method performs internal updates and does not return a value.

        Sets the following instance attributes:
            - `self._oil_preonstream_cost_bulk` : dict
            - `self._gas_preonstream_cost_bulk` : dict

        Notes
        -----
        Internally, this method calls `_calc_cost_bulk`, which applies the
        `get_investment_bulk` method of each `SunkCost` object to the corresponding
        pre-onstream cost array.
        """
        self._oil_preonstream_cost_bulk = self._calc_cost_bulk(
            cost_obj=self._oil_sunk_cost,
            cost_array=self._oil_preonstream_cost_array,
        )

        self._gas_preonstream_cost_bulk = self._calc_cost_bulk(
            cost_obj=self._gas_sunk_cost,
            cost_array=self._gas_preonstream_cost_array,
        )

    def _get_sunk_cost_amortization_charge(
        self,
        prod_year: np.ndarray,
        prod: np.ndarray,
        salvage_value: float = 0.0,
    ) -> None:
        """
        Calculate and store amortization charges for oil and gas sunk costs.

        This method computes the amortization charges for both oil and gas sunk costs
        using production data and stores the results in instance variables.

        Parameters
        ----------
        prod_year : np.ndarray
            Array of production years for which to calculate amortization.
            Should match the length of `prod`.
        prod : np.ndarray
            Array of production values corresponding to `prod_year`.
        salvage_value : float, optional
            The salvage value of the assets at end of amortization period.
            Default is 0.0.

        Returns
        -------
        None
            This method doesn't return anything but updates the following instance variables:
            - self._oil_sunk_cost_amortization_charge
            - self._gas_sunk_cost_amortization_charge

        Notes
        -----
        - The calculation is performed separately for oil and gas sunk costs using
          the respective cost objects and bulk cost data.
        - This method delegates the actual calculation to `_calc_amortization_charge`.
        - The results are stored in instance variables rather than returned.
        """
        self._oil_sunk_cost_amortization_charge = self._calc_amortization_charge(
            cost_obj=self._oil_sunk_cost,
            cost_bulk=self._oil_sunk_cost_bulk,
            prod_year=prod_year,
            prod=prod,
            salvage_value=salvage_value,
        )

        self._gas_sunk_cost_amortization_charge = self._calc_amortization_charge(
            cost_obj=self._gas_sunk_cost,
            cost_bulk=self._gas_sunk_cost_bulk,
            prod_year=prod_year,
            prod=prod,
            salvage_value=salvage_value,
        )

    def _get_preonstream_cost_amortization_charge(
        self,
        prod_year: np.ndarray,
        prod: np.ndarray,
        salvage_value: float = 0.0,
    ) -> None:
        """
        Calculate and store amortization charges for pre-onstream oil and gas costs.

        Computes amortization charges for pre-production (pre-onstream) costs for both
        oil and gas assets using production data and stores results in instance variables.

        Parameters
        ----------
        prod_year : np.ndarray
            Array of production years for amortization calculation.
            Should be the same length as `prod`.
        prod : np.ndarray
            Array of production volumes corresponding to `prod_year`.
        salvage_value : float, optional
            Residual value of assets at end of amortization period.
            Default is 0.0 (no salvage value).

        Returns
        -------
        None
            Results are stored in instance variables:
            - self._oil_preonstream_cost_amortization_charge
            - self._gas_preonstream_cost_amortization_charge

        Notes
        -----
        - Uses sunk cost objects but applies them to pre-onstream cost bulk data
        - Delegates actual calculation to _calc_amortization_charge method
        - Processes oil and gas costs separately using their respective bulk data
        - Results are stored rather than returned to enable access throughout class
        """
        self._oil_preonstream_cost_amortization_charge = self._calc_amortization_charge(
            cost_obj=self._oil_sunk_cost,
            cost_bulk=self._oil_preonstream_cost_bulk,
            prod_year=prod_year,
            prod=prod,
            salvage_value=salvage_value,
        )

        self._gas_preonstream_cost_amortization_charge = self._calc_amortization_charge(
            cost_obj=self._gas_sunk_cost,
            cost_bulk=self._gas_preonstream_cost_bulk,
            prod_year=prod_year,
            prod=prod,
            salvage_value=salvage_value,
        )

    def _get_sunk_cost_tangible_depreciation_charge(
        self,
        depr_method: DeprMethod = DeprMethod.PSC_DB,
        decline_factor: float | int = 2,
        tax_rate: np.ndarray | float = 0.0,
    ) -> None:
        """
        Calculate and store tangible depreciation charges for oil and gas sunk costs.

        Computes depreciation charges for both oil and gas sunk costs using the specified
        depreciation method and stores the results in instance variables. The calculation
        is performed for both fluid types in a single operation using list comprehension.

        Parameters
        ----------
        depr_method : DeprMethod, optional
            The depreciation method to use for calculations.
            Default is DeprMethod.PSC_DB (PSC declining balance).
        decline_factor : float or int, optional
            The decline factor to apply in depreciation calculations.
            Default is 2.
        tax_rate : np.ndarray or float, optional
            Tax rate(s) to apply to depreciation calculations. Can be either:
            - A single float value (constant rate)
            - A numpy array of rates (time-varying rates)
            Default is 0.0 (no tax effect).

        Returns
        -------
        None
            Results are stored in the following instance variables:
            - self._oil_sunk_cost_tangible_depreciation_charge : np.ndarray
            - self._oil_sunk_cost_tangible_undepreciated_asset : float
            - self._gas_sunk_cost_tangible_depreciation_charge : np.ndarray
            - self._gas_sunk_cost_tangible_undepreciated_asset : float

        Notes
        -----
        - Uses list comprehension to efficiently process both oil and gas calculations
        - Delegates actual computation to _calc_tangible_depreciation_charge
        - Each fluid type's calculation uses its respective sunk cost object's method
        - Maintains consistent interface with other depreciation charge methods
        - Results are stored rather than returned for class-wide accessibility
        """
        [
            (
                self._oil_sunk_cost_tangible_depreciation_charge,
                self._oil_sunk_cost_tangible_undepreciated_asset
            ),
            (
                self._gas_sunk_cost_tangible_depreciation_charge,
                self._gas_sunk_cost_tangible_undepreciated_asset
            )
        ] = [
            self._calc_tangible_depreciation_charge(
                cost_mode=cm,
                fluid_type=ft,
                depr_method=depr_method,
                decline_factor=decline_factor,
                tax_rate=tax_rate,
            )
            for cm, ft in zip(
                [
                    self._oil_sunk_cost.get_sunk_cost_tangible_depreciation_charge,
                    self._gas_sunk_cost.get_sunk_cost_tangible_depreciation_charge,
                ],
                [FluidType.OIL, FluidType.GAS]
            )
        ]

    def _get_preonstream_cost_tangible_depreciation_charge(
        self,
        depr_method: DeprMethod = DeprMethod.PSC_DB,
        decline_factor: float | int = 2,
        tax_rate: np.ndarray | float = 0.0,
    ) -> None:
        """
        Calculate and store tangible depreciation charges for pre-onstream costs.

        Computes depreciation charges for pre-production (pre-onstream) costs for both
        oil and gas assets using the specified depreciation method. Results are stored
        in instance variables for class-wide access.

        Parameters
        ----------
        depr_method : DeprMethod, optional
            The depreciation accounting method to apply.
            Default is DeprMethod.PSC_DB (PSC declining balance).
        decline_factor : float or int, optional
            The acceleration factor for declining balance depreciation.
            Default is 2 (double declining balance).
        tax_rate : np.ndarray or float, optional
            Tax rate(s) to apply to depreciation calculations. Can be either:
            - Single float for constant rate
            - np.ndarray for time-varying rates
            Default is 0.0 (tax-exempt).

        Returns
        -------
        None
            Results are stored in these instance variables:
            - self._oil_preonstream_cost_tangible_depreciation_charge : np.ndarray
            - self._oil_preonstream_cost_tangible_undepreciated_asset : float
            - self._gas_preonstream_cost_tangible_depreciation_charge : np.ndarray
            - self._gas_preonstream_cost_tangible_undepreciated_asset : float

        Notes
        -----
        - Processes both oil and gas pre-onstream costs in a single operation
        - Uses sunk cost objects but applies to pre-production cost basis
        - Delegates calculation to _calc_tangible_depreciation_charge
        - Follows same pattern as sunk cost depreciation but for pre-production phase
        - Results stored rather than returned for consistency with other charge methods
        """
        [
            (
                self._oil_preonstream_cost_tangible_depreciation_charge,
                self._oil_preonstream_cost_tangible_undepreciated_asset
            ),
            (
                self._gas_preonstream_cost_tangible_depreciation_charge,
                self._gas_preonstream_cost_tangible_undepreciated_asset
            )
        ] = [
            self._calc_tangible_depreciation_charge(
                cost_mode=cm,
                fluid_type=ft,
                depr_method=depr_method,
                decline_factor=decline_factor,
                tax_rate=tax_rate,
            )
            for cm, ft in zip(
                [
                    self._oil_sunk_cost.get_preonstream_cost_tangible_depreciation_charge,
                    self._gas_sunk_cost.get_preonstream_cost_tangible_depreciation_charge
                ],
                [FluidType.OIL, FluidType.GAS]
            )
        ]

    def fit_sunk_preonstream_cost(
        self,
        prod_year: np.ndarray,
        prod: np.ndarray,
        tax_rate: np.ndarray | float = 0.0,
        depr_method: DeprMethod = DeprMethod.PSC_DB,
        decline_factor: float | int = 2,
        salvage_value: float = 0.0,
    ) -> None:
        """
        Calculate and fit all sunk and pre-onstream cost components.

        This comprehensive method computes and stores:
        - Cost arrays (both sunk and pre-onstream)
        - Cost bulk values
        - Amortization charges
        - Tangible depreciation charges
        for both sunk costs and pre-production (pre-onstream) costs.

        Parameters
        ----------
        prod_year : np.ndarray
            Array of production years for cost calculations.
        prod : np.ndarray
            Array of production volumes corresponding to prod_year.
        salvage_value : float, optional
            Residual value of assets at end of amortization period.
            Default is 0.0.
        depr_method : DeprMethod, optional
            Depreciation method to use for tangible assets.
            Default is DeprMethod.PSC_DB (PSC declining balance).
        decline_factor : float or int, optional
            Decline factor for depreciation calculations.
            Default is 2 (double declining balance).
        tax_rate : np.ndarray or float, optional
            Tax rate(s) to apply. Can be single value or time-varying array.
            Default is 0.0 (no tax effect).

        Returns
        -------
        None
            Results are stored in various instance variables including:
            - Cost arrays:
                self._sunk_cost_array, self._preonstream_cost_array
            - Cost bulk:
                self._sunk_cost_bulk, self._preonstream_cost_bulk
            - Amortization charges:
                self._[oil/gas]_[sunk/preonstream]_cost_amortization_charge
            - Depreciation charges:
                self._[oil/gas]_[sunk/preonstream]_cost_tangible_depreciation_charge

        Notes
        -----
        - Orchestrates complete cost calculation pipeline in proper sequence
        - Handles both sunk costs and pre-production costs
        - Processes both amortization and depreciation components
        - Uses consistent parameters across all calculations
        - All results are stored in instance variables for class-wide access
        - See individual component methods for implementation details
        """
        # Determine sunk cost and preonstream cost array
        self._get_sunk_cost_array(tax_rate=tax_rate)
        self._get_preonstream_cost_array(tax_rate=tax_rate)

        # Determine sunk cost and preonstream cost bulk
        self._get_sunk_cost_bulk()
        self._get_preonstream_cost_bulk()

        # Determine sunk cost and preonstrem cost amortization charge
        self._get_sunk_cost_amortization_charge(
            prod_year=prod_year,
            prod=prod,
            salvage_value=salvage_value,
        )
        self._get_preonstream_cost_amortization_charge(
            prod_year=prod_year,
            prod=prod,
            salvage_value=salvage_value,
        )

        # Determine sunk cost and preonstream cost tangible depreciation charge
        self._get_sunk_cost_tangible_depreciation_charge(
            depr_method=depr_method,
            decline_factor=decline_factor,
            tax_rate=tax_rate,
        )
        self._get_preonstream_cost_tangible_depreciation_charge(
            depr_method=depr_method,
            decline_factor=decline_factor,
            tax_rate=tax_rate,
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
                lft_rate = lft.get_lifting_rate_arr()
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
        depr_method: DeprMethod = DeprMethod.PSC_DB,
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
        # Prepare several attributes associated with sunk cost and preonstream cost
        self.fit_sunk_preonstream_cost(
            prod_year=prod_year,
            prod=prod,
            tax_rate=tax_rate,
            depr_method=depr_method,
            decline_factor=decline_factor,
            salvage_value=salvage_value,
        )

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
