"""
Configure base project as a base framework for contract.
"""

from dataclasses import dataclass, field
from datetime import date
from functools import reduce
import numpy as np

from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.selection import FluidType, TaxType
from pyscnomics.econ.costs import Tangible, Intangible, OPEX, ASR
from pyscnomics.econ.results import CashFlow


class BaseProjectException(Exception):
    """Exception to raise for a misuse of BaseProject class"""

    pass


@dataclass
class BaseProject:
    """
    Represents a base project with start and end dates, lifting information,
    tangible and intangible costs, operational expenses (OPEX), and ASR costs.

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
    tangible_cost : tuple[Tangible]
        A tuple of tangible cost objects.
    intangible_cost : tuple[Intangible], optional
        A tuple of intangible cost objects. Defaults to None.
    opex : tuple[OPEX], optional
        A tuple of operational expense objects. Defaults to None.
    asr_cost : tuple[ASR], optional
        A tuple of ASR (Abandonment and Site Restoration) cost objects.
        Defaults to None.
    """

    start_date: date
    end_date: date
    oil_onstream_date: date = field(default=None)
    gas_onstream_date: date = field(default=None)
    lifting: tuple[Lifting] = field(default=None)
    tangible_cost: tuple[Tangible] = field(default=None)
    intangible_cost: tuple[Intangible] = field(default=None)
    opex: tuple[OPEX] = field(default=None)
    asr_cost: tuple[ASR] = field(default=None)

    # Attributes associated with project duration
    project_duration: int = field(default=None, init=False, repr=False)
    project_years: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes associated with total cost per component
    tangible_cost_total: Tangible = field(default=None, init=False, repr=False)
    intangible_cost_total: Intangible = field(default=None, init=False, repr=False)
    opex_total: OPEX = field(default=None, init=False, repr=False)
    asr_cost_total: ASR = field(default=None, init=False, repr=False)

    # Private attributes (associated with lifting)
    _oil_lifting: Lifting = field(default=None, init=False, repr=False)
    _gas_lifting: Lifting = field(default=None, init=False, repr=False)
    _sulfur_lifting: Lifting = field(default=None, init=False, repr=False)
    _electricity_lifting: Lifting = field(default=None, init=False, repr=False)
    _co2_lifting: Lifting = field(default=None, init=False, repr=False)

    # Private attributes (associated with revenue)
    _oil_revenue: np.ndarray = field(default=None, init=False, repr=False)
    _gas_revenue: np.ndarray = field(default=None, init=False, repr=False)
    _sulfur_revenue: np.ndarray = field(default=None, init=False, repr=False)
    _electricity_revenue: np.ndarray = field(default=None, init=False, repr=False)
    _co2_revenue: np.ndarray = field(default=None, init=False, repr=False)

    # Private attributes (associated with cost)
    _oil_tangible: Tangible = field(default=None, init=False, repr=False)
    _gas_tangible: Tangible = field(default=None, init=False, repr=False)
    _oil_intangible: Intangible = field(default=None, init=False, repr=False)
    _gas_intangible: Intangible = field(default=None, init=False, repr=False)
    _oil_opex: OPEX = field(default=None, init=False, repr=False)
    _gas_opex: OPEX = field(default=None, init=False, repr=False)
    _oil_asr: ASR = field(default=None, init=False, repr=False)
    _gas_asr: ASR = field(default=None, init=False, repr=False)

    # Private attributes associated with expenditures
    _oil_tangible_expenditures: np.ndarray = field(default=None, init=False, repr=False)
    _gas_tangible_expenditures: np.ndarray = field(default=None, init=False, repr=False)
    _oil_intangible_expenditures: np.ndarray = field(default=None, init=False, repr=False)
    _gas_intangible_expenditures: np.ndarray = field(default=None, init=False, repr=False)
    _oil_opex_expenditures: np.ndarray = field(default=None, init=False, repr=False)
    _gas_opex_expenditures: np.ndarray = field(default=None, init=False, repr=False)
    _oil_asr_expenditures: np.ndarray = field(default=None, init=False, repr=False)
    _gas_asr_expenditures: np.ndarray = field(default=None, init=False, repr=False)

    # Private attributes associated with cashflow
    _oil_cashflow: np.ndarray = field(default=None, init=False, repr=False)
    _gas_cashflow: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self):

        # Specify project duration and project years, raise error for inappropriate start date
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

        # User does not provide lifting data (both OIL and GAS)
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

        # User does not provide tangible_cost data (both OIL and GAS)
        if self.tangible_cost is None:
            self.tangible_cost = (
                Tangible(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    cost=np.array([0]),
                    expense_year=np.array([self.start_date.year]),
                    cost_allocation=[FluidType.OIL],
                ),
                Tangible(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    cost=np.array([0]),
                    expense_year=np.array([self.start_date.year]),
                    cost_allocation=[FluidType.GAS],
                )
            )

        # User does not provide intangible_cost data (both OIL and GAS)
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

        # User does not provide opex data (both OIL and GAS)
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

        # User does not provide asr_cost data (both OIL and GAS)
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

        # Fill in the total cost per component
        self.tangible_cost_total = reduce(lambda x, y: x + y, self.tangible_cost)
        self.intangible_cost_total = reduce(lambda x, y: x + y, self.intangible_cost)
        self.opex_total = reduce(lambda x, y: x + y, self.opex)
        self.asr_cost_total = reduce(lambda x, y: x + y, self.asr_cost)

        # Specify lifting data
        self._oil_lifting = self._get_oil_lifting()
        self._gas_lifting = self._get_gas_lifting()
        self._sulfur_lifting = self._get_sulfur_lifting()
        self._electricity_lifting = self._get_electricity_lifting()
        self._co2_lifting = self._get_co2_lifting()

        # Specify revenue data
        self._oil_revenue = self._oil_lifting.revenue()
        self._gas_revenue = self._gas_lifting.revenue()
        self._sulfur_revenue = self._sulfur_lifting.revenue()
        self._electricity_revenue = self._electricity_lifting.revenue()
        self._co2_revenue = self._co2_lifting.revenue()

        # Specify cost data
        self._oil_tangible = self._get_oil_tangible()
        self._gas_tangible = self._get_gas_tangible()
        self._oil_intangible = self._get_oil_intangible()
        self._gas_intangible = self._get_gas_intangible()
        self._oil_opex = self._get_oil_opex()
        self._gas_opex = self._get_gas_opex()
        self._oil_asr = self._get_oil_asr()
        self._gas_asr = self._get_gas_asr()

        # Raise an exception error if the start year of the project is inconsistent
        if not all(
            i == self.start_date.year
            for i in [
                self._oil_lifting.start_year,
                self._gas_lifting.start_year,
                self._sulfur_lifting.start_year,
                self._electricity_lifting.start_year,
                self._co2_lifting.start_year,
                self._oil_tangible.start_year,
                self._gas_tangible.start_year,
                self._oil_intangible.start_year,
                self._gas_intangible.start_year,
                self._oil_opex.start_year,
                self._gas_opex.start_year,
                self._oil_asr.start_year,
                self._gas_asr.start_year,
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
                f"Oil tangible ({self._oil_tangible.start_year}), "
                f"Gas tangible ({self._gas_tangible.start_year}), "
                f"Oil intangible ({self._oil_intangible.start_year}), "
                f"Gas intangible ({self._gas_intangible.start_year}), "
                f"Oil opex ({self._oil_opex.start_year}), "
                f"Gas opex ({self._gas_opex.start_year}), "
                f"Oil asr ({self._oil_asr.start_year}), "
                f"Gas asr ({self._gas_asr.start_year})."
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
                self._oil_tangible.end_year,
                self._gas_tangible.end_year,
                self._oil_intangible.end_year,
                self._gas_intangible.end_year,
                self._oil_opex.end_year,
                self._gas_opex.end_year,
                self._oil_asr.end_year,
                self._gas_asr.end_year,
            ]
        ):
            raise BaseProjectException(
                f"Inconsistent end project year: "

            )

        # Configure oil_onstream_date: set default value and error message
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
            if self.oil_onstream_date is not None:
                raise BaseProjectException(
                    f"Oil onstream year is given ({self.oil_onstream_date.year}) "
                    f"but oil lifting rate is missing or zero for the entire project duration"
                )

            else:
                self.oil_onstream_date = self.end_date

        # Configure gas_onstream_date: set default value and error message
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

    def _get_oil_tangible(self) -> Tangible:
        """
        Determines total oil Tangible from the number of oil Tangible instances in
        attribute self.tangible_cost_total.

        Returns
        -------
        Tangible
            An instance of Tangible that only includes FluidType.OIL as the associated
            cost_allocation that has been combined altogether following the rules prescribed
            in the dunder method __add__() of Tangible class.

        Notes
        -----
        The core operations are as follows:
        (1) Check the attribute cost_allocation in attribute self.tangible_cost_total,
        (2) If OIL is not available as an instance in attribute self.tangible_cost_total,
            then establish a new instance of OIL Tangible with the following attribute set
            to zero: cost.
        (3) Identify index location where cost_allocation is FluidType.OIL in attribute
            self.tangible_cost_total,
        (4) Create a new instance of Tangible with only FluidType.OIL as its cost_allocation.
        """
        if FluidType.OIL not in self.tangible_cost_total.cost_allocation:
            return Tangible(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                cost=np.array([0]),
                expense_year=np.array([self.start_date.year]),
                cost_allocation=[FluidType.OIL],
            )

        else:
            oil_tangible_id = np.argwhere(
                np.array(self.tangible_cost_total.cost_allocation) == FluidType.OIL
            ).ravel()

            start_year = self.tangible_cost_total.start_year
            end_year = self.tangible_cost_total.end_year
            cost = self.tangible_cost_total.cost[oil_tangible_id]
            expense_year = self.tangible_cost_total.expense_year[oil_tangible_id]
            cost_allocation = np.array(self.tangible_cost_total.cost_allocation)[oil_tangible_id]
            vat_portion = self.tangible_cost_total.vat_portion[oil_tangible_id]
            vat_discount = self.tangible_cost_total.vat_discount[oil_tangible_id]
            lbt_portion = self.tangible_cost_total.lbt_portion[oil_tangible_id]
            lbt_discount = self.tangible_cost_total.lbt_discount[oil_tangible_id]
            pis_year = self.tangible_cost_total.pis_year[oil_tangible_id]
            salvage_value = self.tangible_cost_total.salvage_value[oil_tangible_id]
            useful_life = self.tangible_cost_total.useful_life[oil_tangible_id]
            depreciation_factor = self.tangible_cost_total.depreciation_factor[oil_tangible_id]
            is_ic_applied = np.array(self.tangible_cost_total.is_ic_applied)[oil_tangible_id]

            return Tangible(
                start_year=start_year,
                end_year=end_year,
                cost=cost,
                expense_year=expense_year,
                cost_allocation=cost_allocation.tolist(),
                vat_portion=vat_portion,
                vat_discount=vat_discount,
                lbt_portion=lbt_portion,
                lbt_discount=lbt_discount,
                pis_year=pis_year,
                salvage_value=salvage_value,
                useful_life=useful_life,
                depreciation_factor=depreciation_factor,
                is_ic_applied=is_ic_applied.tolist(),
            )

    def _get_gas_tangible(self) -> Tangible:
        """
        Determines total gas Tangible from the number of gas Tangible instances in
        attribute self.tangible_cost_total.

        Returns
        -------
        Tangible
            An instance of Tangible that only includes FluidType.GAS as the associated
            cost_allocation that has been combined altogether following the rules prescribed
            in the dunder method __add__() of Tangible class.

        Notes
        -----
        The core operations are as follows:
        (1) Check the attribute cost_allocation in attribute self.tangible_cost_total,
        (2) If GAS is not available as an instance in attribute self.tangible_cost_total,
            then establish a new instance of GAS Tangible with the following attribute set
            to zero: cost.
        (3) Identify index location where cost_allocation is FluidType.GAS in attribute
            self.tangible_cost_total,
        (4) Create a new instance of Tangible with only FluidType.GAS as its cost_allocation.
        """
        if FluidType.GAS not in self.tangible_cost_total.cost_allocation:
            return Tangible(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                cost=np.array([0]),
                expense_year=np.array([self.start_date.year]),
                cost_allocation=[FluidType.GAS],
            )

        else:
            gas_tangible_id = np.argwhere(
                np.array(self.tangible_cost_total.cost_allocation) == FluidType.GAS
            ).ravel()

            start_year = self.tangible_cost_total.start_year
            end_year = self.tangible_cost_total.end_year
            cost = self.tangible_cost_total.cost[gas_tangible_id]
            expense_year = self.tangible_cost_total.expense_year[gas_tangible_id]
            cost_allocation = np.array(self.tangible_cost_total.cost_allocation)[gas_tangible_id]
            vat_portion = self.tangible_cost_total.vat_portion[gas_tangible_id]
            vat_discount = self.tangible_cost_total.vat_discount[gas_tangible_id]
            lbt_portion = self.tangible_cost_total.lbt_portion[gas_tangible_id]
            lbt_discount = self.tangible_cost_total.lbt_discount[gas_tangible_id]
            pis_year = self.tangible_cost_total.pis_year[gas_tangible_id]
            salvage_value = self.tangible_cost_total.salvage_value[gas_tangible_id]
            useful_life = self.tangible_cost_total.useful_life[gas_tangible_id]
            depreciation_factor = self.tangible_cost_total.depreciation_factor[gas_tangible_id]
            is_ic_applied = np.array(self.tangible_cost_total.is_ic_applied)[gas_tangible_id]

            return Tangible(
                start_year=start_year,
                end_year=end_year,
                cost=cost,
                expense_year=expense_year,
                cost_allocation=cost_allocation.tolist(),
                vat_portion=vat_portion,
                vat_discount=vat_discount,
                lbt_portion=lbt_portion,
                lbt_discount=lbt_discount,
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
            vat_portion = self.intangible_cost_total.vat_portion[oil_intangible_id]
            vat_discount = self.intangible_cost_total.vat_discount[oil_intangible_id]
            lbt_portion = self.intangible_cost_total.lbt_portion[oil_intangible_id]
            lbt_discount = self.intangible_cost_total.lbt_discount[oil_intangible_id]

            return Intangible(
                start_year=start_year,
                end_year=end_year,
                cost=cost,
                expense_year=expense_year,
                cost_allocation=cost_allocation.tolist(),
                vat_portion=vat_portion,
                vat_discount=vat_discount,
                lbt_portion=lbt_portion,
                lbt_discount=lbt_discount,
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
            vat_portion = self.intangible_cost_total.vat_portion[gas_intangible_id]
            vat_discount = self.intangible_cost_total.vat_discount[gas_intangible_id]
            lbt_portion = self.intangible_cost_total.lbt_portion[gas_intangible_id]
            lbt_discount = self.intangible_cost_total.lbt_discount[gas_intangible_id]

            return Intangible(
                start_year=start_year,
                end_year=end_year,
                cost=cost,
                expense_year=expense_year,
                cost_allocation=cost_allocation.tolist(),
                vat_portion=vat_portion,
                vat_discount=vat_discount,
                lbt_portion=lbt_portion,
                lbt_discount=lbt_discount,
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
            vat_portion = self.opex_total.vat_portion[oil_opex_id]
            vat_discount = self.opex_total.vat_discount[oil_opex_id]
            lbt_portion = self.opex_total.lbt_portion[oil_opex_id]
            lbt_discount = self.opex_total.lbt_discount[oil_opex_id]
            fixed_cost = self.opex_total.fixed_cost[oil_opex_id]
            prod_rate = self.opex_total.prod_rate[oil_opex_id]
            cost_per_volume = self.opex_total.cost_per_volume[oil_opex_id]

            return OPEX(
                start_year=start_year,
                end_year=end_year,
                expense_year=expense_year,
                cost_allocation=cost_allocation.tolist(),
                vat_portion=vat_portion,
                vat_discount=vat_discount,
                lbt_portion=lbt_portion,
                lbt_discount=lbt_discount,
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
            vat_portion = self.opex_total.vat_portion[gas_opex_id]
            vat_discount = self.opex_total.vat_discount[gas_opex_id]
            lbt_portion = self.opex_total.lbt_portion[gas_opex_id]
            lbt_discount = self.opex_total.lbt_discount[gas_opex_id]
            fixed_cost = self.opex_total.fixed_cost[gas_opex_id]
            prod_rate = self.opex_total.prod_rate[gas_opex_id]
            cost_per_volume = self.opex_total.cost_per_volume[gas_opex_id]

            return OPEX(
                start_year=start_year,
                end_year=end_year,
                expense_year=expense_year,
                cost_allocation=cost_allocation.tolist(),
                vat_portion=vat_portion,
                vat_discount=vat_discount,
                lbt_portion=lbt_portion,
                lbt_discount=lbt_discount,
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
            vat_portion = self.asr_cost_total.vat_portion[oil_asr_id]
            vat_discount = self.asr_cost_total.vat_discount[oil_asr_id]
            lbt_portion = self.asr_cost_total.lbt_portion[oil_asr_id]
            lbt_discount = self.asr_cost_total.lbt_discount[oil_asr_id]

            return ASR(
                start_year=start_year,
                end_year=end_year,
                cost=cost,
                expense_year=expense_year,
                cost_allocation=cost_allocation.tolist(),
                vat_portion=vat_portion,
                vat_discount=vat_discount,
                lbt_portion=lbt_portion,
                lbt_discount=lbt_discount,
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
            vat_portion = self.asr_cost_total.vat_portion[gas_asr_id]
            vat_discount = self.asr_cost_total.vat_discount[gas_asr_id]
            lbt_portion = self.asr_cost_total.lbt_portion[gas_asr_id]
            lbt_discount = self.asr_cost_total.lbt_discount[gas_asr_id]

            return ASR(
                start_year=start_year,
                end_year=end_year,
                cost=cost,
                expense_year=expense_year,
                cost_allocation=cost_allocation.tolist(),
                vat_portion=vat_portion,
                vat_discount=vat_discount,
                lbt_portion=lbt_portion,
                lbt_discount=lbt_discount,
            )

    def _get_expenditures(
        self,
        year_ref: int = None,
        tax_type: TaxType = TaxType.VAT,
        vat_rate: np.ndarray | float = 0.0,
        lbt_rate: np.ndarray | float = 0.0,
        inflation_rate: np.ndarray | float = 0.0,
        future_rate: float = 0.02,
    ) -> None:
        """
        Calculate and assign expenditures for various cost components.

        Parameters
        ----------
        year_ref: int, optional
            Reference year for inflation calculation. Defaults to None.
        tax_type: TaxType, optional
            The type of tax applied to the corresponding asset.
            Available options: TaxType.VAT or TaxType.LBT.
            Defaults to TaxType.VAT.
        vat_rate: np.ndarray | float, optional
            The VAT rate to apply. Can be a single value or an array (defaults to 0.0).
        lbt_rate: np.ndarray | float, optional
            The LBT rate to apply. Can be a single value or an array (defaults to 0.0).
        inflation_rate: np.ndarray | float, optional
            The inflation rate to apply. Can be a single value or an array (defaults to 0.0).
        future_rate : float, optional
            The future rate used in cost calculation (defaults to 0.02).

        Returns
        -------
        None

        Notes
        -----
        The core procedures are as follows:
        (1) Create an inner function named calc_expenses() to calculate the expenditures
            of a particular target attribute,
        (2) Calculate the expenditures of all cost components by performing iteration
            over all available cost components using Python's build-in map() method,
        (3) Extract individual expenditures and assign them to the commensurate attribute.
        """

        if year_ref is None:
            year_ref = self.start_date.year

        def calc_expenses(target_attr):
            """ Calculate expenditures for a target attribute """

            if target_attr is self._oil_asr or target_attr is self._gas_asr:
                return target_attr.expenditures(
                    year_ref=year_ref,
                    tax_type=tax_type,
                    vat_rate=vat_rate,
                    lbt_rate=lbt_rate,
                    inflation_rate=inflation_rate,
                    future_rate=future_rate,
                )

            else:
                return target_attr.expenditures(
                    year_ref=year_ref,
                    tax_type=tax_type,
                    vat_rate=vat_rate,
                    lbt_rate=lbt_rate,
                    inflation_rate=inflation_rate,
                )

        (
            self._oil_tangible_expenditures,
            self._gas_tangible_expenditures,
            self._oil_intangible_expenditures,
            self._gas_intangible_expenditures,
            self._oil_opex_expenditures,
            self._gas_opex_expenditures,
            self._oil_asr_expenditures,
            self._gas_asr_expenditures,
        ) = list(
            map(
                calc_expenses,
                [self._oil_tangible, self._gas_tangible, self._oil_intangible,
                 self._gas_intangible, self._oil_opex, self._gas_opex,
                 self._oil_asr, self._gas_asr]
            )
        )

    def run(
        self,
        year_ref: int = None,
        tax_type: TaxType = TaxType.VAT,
        vat_rate: np.ndarray | float = 0.0,
        lbt_rate: np.ndarray | float = 0.0,
        inflation_rate: np.ndarray | float = 0.0,
        future_rate: float = 0.02,
    ) -> None:
        """
        Run the economic analysis, calculating expenditures and configuring
        cashflows for OIL and GAS.

        Parameters
        ----------
        year_ref: int, optional
            Reference year for inflation calculation. Defaults to None.
        tax_type: TaxType, optional
            The type of tax applied to the corresponding asset.
            Available options: TaxType.VAT or TaxType.LBT.
            Defaults to TaxType.VAT.
        vat_rate: np.ndarray | float, optional
            The VAT rate to apply. Can be a single value or an array (defaults to 0.0).
        lbt_rate: np.ndarray | float, optional
            The LBT rate to apply. Can be a single value or an array (defaults to 0.0).
        inflation_rate: np.ndarray | float, optional
            The inflation rate to apply. Can be a single value or an array (defaults to 0.0).
        future_rate : float, optional
            The future rate used in cost calculation (defaults to 0.02).

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
        # Prepare the data
        self._get_expenditures(
            year_ref=year_ref,
            tax_type=tax_type,
            vat_rate=vat_rate,
            lbt_rate=lbt_rate,
            inflation_rate=inflation_rate,
            future_rate=future_rate,
        )

        # Configure total expenditures for OIL and GAS
        oil_total_expenditures = reduce(
            lambda x, y: x + y,
            [
                self._oil_tangible_expenditures, self._oil_intangible_expenditures,
                self._oil_opex_expenditures, self._oil_asr_expenditures
            ]
        )

        gas_total_expenditures = reduce(
            lambda x, y: x + y,
            [
                self._gas_tangible_expenditures, self._gas_intangible_expenditures,
                self._gas_opex_expenditures, self._gas_asr_expenditures
            ]
        )

        # Configure base cashflow
        self._oil_cashflow = CashFlow(
            start_date=self.start_date,
            end_date=self.end_date,
            cash=self._oil_revenue - oil_total_expenditures,
            cashed_year=self.project_years,
            cash_allocation=FluidType.OIL,
        )

        self._gas_cashflow = CashFlow(
            start_date=self.start_date,
            end_date=self.end_date,
            cash=self._gas_revenue - gas_total_expenditures,
            cashed_year=self.project_years,
            cash_allocation=FluidType.GAS,
        )

    def __len__(self):
        return self.project_duration

    def __eq__(self, other):
        # Between two instances of BaseProject
        if isinstance(other, BaseProject):
            revenue_self = np.sum(self._oil_revenue + self._gas_revenue)
            revenue_other = np.sum(other._oil_revenue + other._gas_revenue)
            tangible_self = sum(self._oil_tangible.cost) + sum(self._gas_tangible.cost)
            tangible_other = sum(other._oil_tangible.cost) + sum(other._gas_tangible.cost)
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
            tangible_self = sum(self._oil_tangible.cost) + sum(self._gas_tangible.cost)
            tangible_other = sum(other._oil_tangible.cost) + sum(other._gas_tangible.cost)
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
            tangible_self = sum(self._oil_tangible.cost) + sum(self._gas_tangible.cost)
            tangible_other = sum(other._oil_tangible.cost) + sum(other._gas_tangible.cost)
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
            tangible_self = sum(self._oil_tangible.cost) + sum(self._gas_tangible.cost)
            tangible_other = sum(other._oil_tangible.cost) + sum(other._gas_tangible.cost)
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
            tangible_self = sum(self._oil_tangible.cost) + sum(self._gas_tangible.cost)
            tangible_other = sum(other._oil_tangible.cost) + sum(other._gas_tangible.cost)
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

    def __add__(self, other):
        # Between an instance of BaseProject and another instance of BaseProject
        if isinstance(other, BaseProject):
            # Specify the start_date and end_date of the combined instances
            start_date_combined = date(
                year=min(self.start_date.year, other.start_date.year),
                month=min(self.start_date.month, other.start_date.month),
                day=min(self.start_date.day, other.start_date.day),
            )

            end_date_combined = date(
                year=max(self.end_date.year, other.end_date.year),
                month=max(self.end_date.month, other.end_date.month),
                day=max(self.end_date.day, other.end_date.day),
            )

            return BaseProject(
                start_date=start_date_combined,
                end_date=end_date_combined,
                lifting=self.lifting + other.lifting,
                tangible_cost=self.tangible_cost + other.tangible_cost,
                intangible_cost=self.intangible_cost + other.intangible_cost,
                opex=self.opex + other.opex,
                asr_cost=self.asr_cost + other.asr_cost,
            )

        else:
            raise BaseProjectException(
                f"Must add between an instance of BaseProject with "
                f"another instance of BaseProject. "
                f"{other}({other.__class__.__qualname__}) is not an instance "
                f"of BaseProject."
            )
