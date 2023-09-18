"""
Configure base project as a foundational framework for contract.
"""

from dataclasses import dataclass, field
from datetime import date
from functools import reduce
import numpy as np

from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.selection import FluidType
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
    tangible_cost: tuple[Tangible] = field(default=None, repr=False)
    intangible_cost: tuple[Intangible] = field(default=None, repr=False)
    opex: tuple[OPEX] = field(default=None, repr=False)
    asr_cost: tuple[ASR] = field(default=None, repr=False)

    # Attribute to be defined later
    project_duration: int = field(default=None, init=False, repr=False)
    project_years: np.ndarray = field(default=None, init=False, repr=False)
    summary_base_project: dict = field(default=None, init=False, repr=False)

    _oil_lifting: Lifting = field(default=None, init=False, repr=False)
    _gas_lifting: Lifting = field(default=None, init=False, repr=False)
    _oil_revenue: np.ndarray = field(default=None, init=False, repr=False)
    _gas_revenue: np.ndarray = field(default=None, init=False, repr=False)
    _oil_tangible: Tangible = field(default=None, init=False, repr=False)
    _gas_tangible: Tangible = field(default=None, init=False, repr=False)
    _oil_intangible: Intangible = field(default=None, init=False, repr=False)
    _gas_intangible: Intangible = field(default=None, init=False, repr=False)
    _oil_opex: OPEX = field(default=None, init=False, repr=False)
    _gas_opex: OPEX = field(default=None, init=False, repr=False)
    _oil_asr: ASR = field(default=None, init=False, repr=False)
    _gas_asr: ASR = field(default=None, init=False, repr=False)

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
            )

        # User does not provide tangible_cost data (both OIL and GAS)
        if self.tangible_cost is None:
            self.tangible_cost = (
                Tangible(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    cost=np.array([0]),
                    expense_year=np.array([self.start_date.year]),
                    cost_allocation=FluidType.OIL,
                ),
                Tangible(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    cost=np.array([0]),
                    expense_year=np.array([self.start_date.year]),
                    cost_allocation=FluidType.GAS,
                ),
            )

        # User does not provide intangible_cost data (both OIL and GAS)
        if self.intangible_cost is None:
            self.intangible_cost = (
                Intangible(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    cost=np.array([0]),
                    expense_year=np.array([self.start_date.year]),
                    cost_allocation=FluidType.OIL,
                ),
                Intangible(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    cost=np.array([0]),
                    expense_year=np.array([self.start_date.year]),
                    cost_allocation=FluidType.GAS,
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
                    cost_allocation=FluidType.OIL,
                ),
                OPEX(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    fixed_cost=np.array([0]),
                    expense_year=np.array([self.start_date.year]),
                    cost_allocation=FluidType.GAS,
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
                    cost_allocation=FluidType.OIL,
                ),
                ASR(
                    start_year=self.start_date.year,
                    end_year=self.end_date.year,
                    cost=np.array([0]),
                    expense_year=np.array([self.start_date.year]),
                    cost_allocation=FluidType.GAS,
                ),
            )

        # Configure oil_onstream_date: set default value and error message
        oil_revenue_index = np.argwhere(self._get_oil_lifting().revenue() > 0).ravel()

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
                    np.argwhere(self.oil_onstream_date.year == self.project_years).ravel()
                )

                if oil_onstream_index != oil_revenue_index[0]:
                    raise BaseProjectException(
                        f"Oil onstream year ({self.oil_onstream_date.year}) is different from "
                        f"the starting year of oil production ({self.project_years[oil_revenue_index[0]]})"
                    )

            else:
                self.oil_onstream_date = date(
                    year=self.project_years[oil_revenue_index[0]],
                    month=1,
                    day=1
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
        gas_revenue_index = np.argwhere(self._get_gas_lifting().revenue() > 0).ravel()

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
                    np.argwhere(self.gas_onstream_date.year == self.project_years).ravel()
                )

                if gas_onstream_index != gas_revenue_index[0]:
                    raise BaseProjectException(
                        f"Gas onstream year ({self.gas_onstream_date.year}) is different from "
                        f"the starting year of gas production ({self.project_years[gas_revenue_index[0]]})"
                    )

            else:
                self.gas_onstream_date = date(
                    year=self.project_years[gas_revenue_index[0]],
                    month=1,
                    day=1
                )

        else:
            if self.gas_onstream_date is not None:
                raise BaseProjectException(
                    f"Gas onstream year is given ({self.gas_onstream_date.year}) "
                    f"but gas lifting rate is missing or zero for the entire project duration"
                )

            else:
                self.gas_onstream_date = self.end_date

        # Initiate attribute summary base project
        self.summary_base_project = {}

        # Fill the values of private attributes
        self._oil_lifting = self._get_oil_lifting()
        self._gas_lifting = self._get_gas_lifting()

    def _get_oil_lifting(self):
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

    def _get_gas_lifting(self):
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

    def _get_oil_tangible(self):
        """
        Determines total oil Tangible from the number of oil Tangible instances in
        attribute self.tangible_cost.

        Returns
        -------
        total_oil_tangible: Tangible
            Total oil Tangible as a new instance of Tangible class where the associated
            atrributes are determined based on the prescribed rule in the corresponding
            dunder method __add__() of the Tangible class.

        Notes
        -----
        The order of operations is as follows:
        (1) Check the attribute cost_allocation in attribute self.tangible_cost,
        (2) If OIL is not available as an instance in attribute self.tangible_cost, then
            establish a new instance of OIL Tangible with the following attribute set
            to zero: cost.
        (3) Add the Tangible instances with fluid_type OIL, return the result as a new instance
            of Tangible class following the rule prescribed in the dunder method __add__() of
            class Tangible.
        """

        fluid_types = [cst.cost_allocation for cst in self.tangible_cost]

        if FluidType.OIL not in fluid_types:
            return Tangible(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                cost=np.array([0]),
                expense_year=np.array([self.start_date.year]),
                cost_allocation=FluidType.OIL,
            )

        return reduce(
            lambda x, y: x + y,
            (
                tang
                for tang in self.tangible_cost
                if tang.cost_allocation == FluidType.OIL
            ),
        )

    def _get_gas_tangible(self):
        """
        Determines total gas Tangible from the number of gas Tangible instances in
        attribute self.tangible_cost.

        Returns
        -------
        total_gas_tangible: Tangible
            Total gas Tangible as a new instance of Tangible class where the associated
            atrributes are determined based on the prescribed rule in the corresponding
            dunder method __add__() of the Tangible class.

        Notes
        -----
        The order of operations is as follows:
        (1) Check the attribute cost_allocation in attribute self.tangible_cost,
        (2) If GAS is not available as an instance in attribute self.tangible_cost, then
            establish a new instance of GAS Tangible with the following attribute set
            to zero: cost.
        (3) Add the Tangible instances with fluid_type GAS, return the result as a new instance
            of Tangible class following the rule prescribed in the dunder method __add__() of
            class Tangible.
        """

        fluid_types = [cst.cost_allocation for cst in self.tangible_cost]

        if FluidType.GAS not in fluid_types:
            return Tangible(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                cost=np.array([0]),
                expense_year=np.array([self.start_date.year]),
                cost_allocation=FluidType.GAS,
            )

        return reduce(
            lambda x, y: x + y,
            (
                tang
                for tang in self.tangible_cost
                if tang.cost_allocation == FluidType.GAS
            ),
        )

    def _get_oil_intangible(self):
        """
        Determines total oil Intangible from the number of oil Intangible instances in
        attribute self.intangible_cost.

        Returns
        -------
        total_oil_intangible: Intangible
            Total oil Intangible as a new instance of Intangible class where the associated
            atrributes are determined based on the prescribed rule in the corresponding
            dunder method __add__() of the Intangible class.

        Notes
        -----
        The order of operations is as follows:
        (1) Check the attribute cost_allocation in attribute self.intangible_cost,
        (2) If OIL is not available as an instance in attribute self.intangible_cost, then
            establish a new instance of OIL Intangible with the following attribute set
            to zero: cost.
        (3) Add the Intangible instances with fluid_type OIL, return the result as a new instance
            of Intangible class following the rule prescribed in the dunder method __add__() of
            class Intangible.
        """

        fluid_types = [cst.cost_allocation for cst in self.intangible_cost]

        if FluidType.OIL not in fluid_types:
            return Intangible(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                cost=np.array([0]),
                expense_year=np.array([self.start_date.year]),
                cost_allocation=FluidType.OIL,
            )

        return reduce(
            lambda x, y: x + y,
            (i for i in self.intangible_cost if i.cost_allocation == FluidType.OIL),
        )

    def _get_gas_intangible(self):
        """
        Determines total gas Intangible from the number of gas Intangible instances in
        attribute self.intangible_cost.

        Returns
        -------
        total_gas_intangible: Intangible
            Total gas Intangible as a new instance of Intangible class where the associated
            atrributes are determined based on the prescribed rule in the corresponding
            dunder method __add__() of the Intangible class.

        Notes
        -----
        The order of operations is as follows:
        (1) Check the attribute cost_allocation in attribute self.intangible_cost,
        (2) If GAS is not available as an instance in attribute self.intangible_cost, then
            establish a new instance of GAS Intangible with the following attribute set
            to zero: cost.
        (3) Add the Intangible instances with fluid_type GAS, return the result as a new instance
            of Intangible class following the rule prescribed in the dunder method __add__() of
            class Intangible.
        """

        fluid_types = [cst.cost_allocation for cst in self.intangible_cost]

        if FluidType.GAS not in fluid_types:
            return Intangible(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                cost=np.array([0]),
                expense_year=np.array([self.start_date.year]),
                cost_allocation=FluidType.GAS,
            )

        return reduce(
            lambda x, y: x + y,
            (i for i in self.intangible_cost if i.cost_allocation == FluidType.GAS),
        )

    def _get_oil_opex(self):
        """
        Determines total oil OPEX from the number of oil OPEX instances in
        attribute self.opex.

        Returns
        -------
        total_oil_opex: OPEX
            Total oil OPEX as a new instance of OPEX class where the associated
            atrributes are determined based on the prescribed rule in the corresponding
            dunder method __add__() of the OPEX class.

        Notes
        -----
        The order of operations is as follows:
        (1) Check the attribute cost_allocation in attribute self.opex,
        (2) If OIL is not available as an instance in attribute self.opex, then
            establish a new instance of OIL OPEX with the following attribute set
            to zero: cost.
        (3) Add the OPEX instances with fluid_type OIL, return the result as a new instance
            of OPEX class following the rule prescribed in the dunder method __add__() of
            class OPEX.
        """

        fluid_types = [cst.cost_allocation for cst in self.opex]

        if FluidType.OIL not in fluid_types:
            return OPEX(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                fixed_cost=np.array([0]),
                expense_year=np.array([self.start_date.year]),
                cost_allocation=FluidType.OIL,
            )

        return reduce(
            lambda x, y: x + y,
            (i for i in self.opex if i.cost_allocation == FluidType.OIL),
        )

    def _get_gas_opex(self):
        """
        Determines total gas OPEX from the number of gas OPEX instances in
        attribute self.opex.

        Returns
        -------
        total_gas_opex: OPEX
            Total gas OPEX as a new instance of OPEX class where the associated
            atrributes are determined based on the prescribed rule in the corresponding
            dunder method __add__() of the OPEX class.

        Notes
        -----
        The order of operations is as follows:
        (1) Check the attribute cost_allocation in attribute self.opex,
        (2) If GAS is not available as an instance in attribute self.opex, then
            establish a new instance of GAS OPEX with the following attribute set
            to zero: cost.
        (3) Add the OPEX instances with fluid_type GAS, return the result as a new instance
            of OPEX class following the rule prescribed in the dunder method __add__() of
            class OPEX.
        """

        fluid_types = [cst.cost_allocation for cst in self.opex]

        if FluidType.GAS not in fluid_types:
            return OPEX(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                fixed_cost=np.array([0]),
                expense_year=np.array([self.start_date.year]),
                cost_allocation=FluidType.GAS,
            )

        return reduce(
            lambda x, y: x + y,
            (i for i in self.opex if i.cost_allocation == FluidType.GAS),
        )

    def _get_oil_asr(self):
        """
        Determines total oil ASR from the number of oil ASR instances in
        attribute self.asr_cost.

        Returns
        -------
        total_oil_asr: ASR
            Total oil ASR as a new instance of ASR class where the associated
            atrributes are determined based on the prescribed rule in the corresponding
            dunder method __add__() of the ASR class.

        Notes
        -----
        The order of operations is as follows:
        (1) Check the attribute cost_allocation in attribute self.asr_cost,
        (2) If OIL is not available as an instance in attribute self.asr_cost, then
            establish a new instance of OIL ASR with the following attribute set
            to zero: cost.
        (3) Add the ASR instances with fluid_type OIL, return the result as a new instance
            of ASR class following the rule prescribed in the dunder method __add__() of
            class ASR.
        """

        fluid_types = [cst.cost_allocation for cst in self.asr_cost]

        if FluidType.OIL not in fluid_types:
            return ASR(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                cost=np.array([0]),
                expense_year=np.array([self.start_date.year]),
                cost_allocation=FluidType.OIL,
            )

        return reduce(
            lambda x, y: x + y,
            (i for i in self.asr_cost if i.cost_allocation == FluidType.OIL),
        )

    def _get_gas_asr(self):
        """
        Determines total gas ASR from the number of gas ASR instances in
        attribute self.asr_cost.

        Returns
        -------
        total_gas_asr: ASR
            Total gas ASR as a new instance of ASR class where the associated
            atrributes are determined based on the prescribed rule in the corresponding
            dunder method __add__() of the ASR class.

        Notes
        -----
        The order of operations is as follows:
        (1) Check the attribute cost_allocation in attribute self.asr_cost,
        (2) If GAS is not available as an instance in attribute self.asr_cost, then
            establish a new instance of GAS ASR with the following attribute set
            to zero: cost.
        (3) Add the ASR instances with fluid_type GAS, return the result as a new instance
            of ASR class following the rule prescribed in the dunder method __add__() of
            class ASR.
        """

        fluid_types = [cst.cost_allocation for cst in self.asr_cost]

        if FluidType.GAS not in fluid_types:
            return ASR(
                start_year=self.start_date.year,
                end_year=self.end_date.year,
                cost=np.array([0]),
                expense_year=np.array([self.start_date.year]),
                cost_allocation=FluidType.GAS,
            )

        return reduce(
            lambda x, y: x + y,
            (i for i in self.asr_cost if i.cost_allocation == FluidType.GAS),
        )

    def run(self):
        """
        Calculate the cash flow based on revenues and expenditures over the project duration.

        Returns
        -------
        CashFlow
            A CashFlow object representing the calculated cash flow.

        Notes
        -----
        - The method calculates the cash flow by iterating through various sources of
          revenues (lift.revenue()) and expenditures (cost.expenditures()).
        - The resulting cash flow is the difference between total revenues and total
          expenditures for each time step within the project duration.
        - The calculated cash flow is encapsulated within a CashFlow object, which
          includes the start and end dates of the project and the calculated cash values.
        """

        self._oil_lifting = self._get_oil_lifting()
        self._gas_lifting = self._get_gas_lifting()
        self._oil_revenue = self._get_oil_lifting().revenue()
        self._gas_revenue = self._get_gas_lifting().revenue()
        self._oil_tangible = self._get_oil_tangible()
        self._gas_tangible = self._get_gas_tangible()

        # Calculate Base CashFlow
        # revenues = np.zeros(self.project_duration)
        #
        # for lft in self.lifting:
        #     revenue = revenue + lft.revenue()
        #
        # expenditures = np.zeros(self.project_duration)
        #
        # for cost in self.tangible_cost:
        #     expenditures = expenditures + cost.expenditures()
        #
        # for cost in self.intangible_cost:
        #     expenditures = expenditures + cost.expenditures()
        #
        # for cost in self.opex:
        #     expenditures = expenditures + cost.expenditures()
        #
        # for cost in self.asr_cost:
        #     expenditures = expenditures + cost.expenditures()

        # return CashFlow(
        #     start_date=self.start_date,
        #     end_date=self.end_date,
        #     cash=revenues - expenditures,
        # )
