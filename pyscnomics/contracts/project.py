"""
Configure base project as a base framework for contract.
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
    tangible_cost: tuple[Tangible] = field(default=None)
    intangible_cost: tuple[Intangible] = field(default=None)
    opex: tuple[OPEX] = field(default=None)
    asr_cost: tuple[ASR] = field(default=None)

    # Attributes to be defined later
    project_duration: int = field(default=None, init=False, repr=False)
    project_years: np.ndarray = field(default=None, init=False, repr=False)

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
    _oil_total_expenses: np.ndarray = field(default=None, init=False, repr=False)
    _gas_total_expenses: np.ndarray = field(default=None, init=False, repr=False)
    _oil_base_cashflow: CashFlow = field(default=None, init=False, repr=False)
    _gas_base_cashflow: CashFlow = field(default=None, init=False, repr=False)

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

        # Fill the values of private attributes
        self._oil_lifting = self._get_oil_lifting()
        self._gas_lifting = self._get_gas_lifting()
        self._oil_revenue = self._get_oil_lifting().revenue()
        self._gas_revenue = self._get_gas_lifting().revenue()
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
                for i in [self._oil_lifting.start_year, self._gas_lifting.start_year]
        ):
            raise BaseProjectException(
                f"Inconsistent start project data: "
                f"Base project ({self.start_date.year}), "
                f"Oil lifting ({self._oil_lifting.start_year}), "
                f"Gas lifting ({self._gas_lifting.start_year}), "
            )

        # Raise an exception error if the end year of the project is inconsistent
        if not all(
                i == self.end_date.year
                for i in [self._oil_lifting.end_year, self._gas_lifting.end_year]
        ):
            raise BaseProjectException(
                f"Inconsistent end project data: "
                f"Base project ({self.end_date.year}), "
                f"Oil lifting ({self._oil_lifting.end_year}), "
                f"Gas lifting ({self._gas_lifting.end_year}), "
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

        # Calculate total expenses for OIL;
        # Here, for Tangible cost, we only take the regular expenditures, not the depreciated value
        self._oil_total_expenses = (
            self._oil_tangible.expenditures()
            + self._oil_intangible.expenditures()
            + self._oil_opex.expenditures()
            + self._oil_asr.expenditures()
        )

        # Calculate total expenses for GAS;
        # Here, for Tangible cost, we only take the regular expenditures, not the depreciated value
        self._gas_total_expenses = (
            self._gas_tangible.expenditures()
            + self._gas_intangible.expenditures()
            + self._gas_opex.expenditures()
            + self._gas_asr.expenditures()
        )

        self._oil_base_cashflow = CashFlow(
            start_date=self.start_date,
            end_date=self.end_date,
            cash=self._oil_revenue - self._oil_total_expenses,
            cashed_year=self.project_years,
            cash_allocation=FluidType.OIL
        )

        self._gas_base_cashflow = CashFlow(
            start_date=self.start_date,
            end_date=self.end_date,
            cash=self._gas_revenue - self._gas_total_expenses,
            cashed_year=self.project_years,
            cash_allocation=FluidType.GAS
        )

    def __len__(self):
        return self.project_duration

    def __eq__(self, other):

        # Between two instances of BaseProject
        if isinstance(other, BaseProject):
            return all(
                (
                    np.allclose(self._oil_lifting.lifting_rate, other._oil_lifting.lifting_rate),
                    np.allclose(self._gas_lifting.lifting_rate, other._gas_lifting.lifting_rate),
                    np.allclose(self._oil_revenue, other._oil_revenue),
                    np.allclose(self._gas_revenue, other._gas_revenue),
                    np.allclose(self._oil_tangible.expenditures(), other._oil_tangible.expenditures()),
                    np.allclose(self._gas_tangible.expenditures(), other._gas_tangible.expenditures()),
                    np.allclose(self._oil_intangible.expenditures(), other._oil_intangible.expenditures()),
                    np.allclose(self._gas_intangible.expenditures(), other._gas_intangible.expenditures()),
                    np.allclose(self._oil_opex.expenditures(), other._oil_opex.expenditures()),
                    np.allclose(self._gas_opex.expenditures(), other._gas_opex.expenditures()),
                    np.allclose(self._oil_asr.expenditures(), other._oil_asr.expenditures()),
                    np.allclose(self._gas_asr.expenditures(), other._gas_asr.expenditures()),
                )
            )

        else:
            return False

    def __lt__(self, other):

        if isinstance(other, BaseProject):

            self.run()
            other.run()

            self_total_base_cashflow = self._oil_base_cashflow + self._gas_base_cashflow
            other_total_base_cashflow = other._oil_base_cashflow + other._gas_base_cashflow

            return self_total_base_cashflow.irr() < other_total_base_cashflow.irr()

        else:
            raise BaseProjectException

    def __le__(self, other):

        if isinstance(other, BaseProject):

            self.run()
            other.run()

            self_total_base_cashflow = self._oil_base_cashflow + self._gas_base_cashflow
            other_total_base_cashflow = other._oil_base_cashflow + other._gas_base_cashflow

            return self_total_base_cashflow.irr() <= other_total_base_cashflow.irr()

        else:
            raise BaseProjectException

    def __gt__(self, other):

        if isinstance(other, BaseProject):

            self.run()
            other.run()

            self_total_base_cashflow = self._oil_base_cashflow + self._gas_base_cashflow
            other_total_base_cashflow = other._oil_base_cashflow + other._gas_base_cashflow

            return self_total_base_cashflow.irr() > other_total_base_cashflow.irr()

        else:
            raise BaseProjectException

    def __ge__(self, other):

        if isinstance(other, BaseProject):

            self.run()
            other.run()

            self_total_base_cashflow = self._oil_base_cashflow + self._gas_base_cashflow
            other_total_base_cashflow = other._oil_base_cashflow + other._gas_base_cashflow

            return self_total_base_cashflow.irr() >= other_total_base_cashflow.irr()

        else:
            raise BaseProjectException

    def __add__(self, other):
        # Between an instance of BaseProject and another instance of BaseProject
        if isinstance(other, BaseProject):

            start_year = min(self.start_date.year, other.start_date.year)
            start_month = min(self.start_date.month, other.start_date.month)
            start_day = min(self.start_date.day, other.start_date.day)

            end_year = max(self.end_date.year, other.end_date.year)
            end_month = max(self.end_date.month, other.end_date.month)
            end_day = max(self.end_date.day, other.end_date.day)

            # Specify the start_date and end_date of the combined instances
            start_date_combined = date(year=start_year, month=start_month, day=start_day)
            end_date_combined = date(year=end_year, month=end_month, day=end_day)

            return BaseProject(
                start_date=start_date_combined,
                end_date=end_date_combined,
                lifting=self.lifting + other.lifting,
                tangible_cost=self.tangible_cost + other.tangible_cost,
                intangible_cost=self.intangible_cost + other.intangible_cost,
                opex=self.opex + other.opex,
                asr_cost=self.asr_cost + other.asr_cost
            )

        else:
            raise BaseProjectException(
                f"Must add between an instance of BaseProject with "
                f"another instance of BaseProject. "
                f"{other}({other.__class__.__qualname__}) is not an instance "
                f"of BaseProject."
            )

    def __sub__(self, other):

        # Between an instance of BaseProject and another instance of BaseProject
        if isinstance(other, BaseProject):

            start_year = min(self.start_date.year, other.start_date.year)
            start_month = min(self.start_date.month, other.start_date.month)
            start_day = min(self.start_date.day, other.start_date.day)

            end_year = max(self.end_date.year, other.end_date.year)
            end_month = max(self.end_date.month, other.end_date.month)
            end_day = max(self.end_date.day, other.end_date.day)

            # Specify the start_date and end_date of the combined instances
            start_date_combined = date(year=start_year, month=start_month, day=start_day)
            end_date_combined = date(year=end_year, month=end_month, day=end_day)

            # Invoke a negative value to lifting_rate attribute of Lifting class (subtraction)
            for lft in other.lifting:
                lft.lifting_rate = (-1 * lft.lifting_rate).copy()

            # Invoke a negative value to cost attribute of Tangible class (subtraction)
            for tan in other.tangible_cost:
                tan.cost = (-1 * tan.cost).copy()

            # Invoke a negative value to cost attribute of Intangible class (subtraction)
            for intan in other.intangible_cost:
                intan.cost = (-1 * intan.cost).copy()

            # Invoke a negative value to fixed_cost and prod_rate attributes of OPEX class (subtraction)
            for op in other.opex:
                op.fixed_cost = (-1 * op.fixed_cost).copy()
                op.prod_rate = (-1 * op.prod_rate).copy()

            # Invoke a negative value to cost attribute of ASR class (subtraction)
            for asr in other.asr_cost:
                asr.cost = (-1 * asr.cost).copy()

            return BaseProject(
                start_date=start_date_combined,
                end_date=end_date_combined,
                lifting=self.lifting + other.lifting,
                tangible_cost=self.tangible_cost + other.tangible_cost,
                intangible_cost=self.intangible_cost + other.intangible_cost,
                opex=self.opex + other.opex,
                asr_cost=self.asr_cost + other.asr_cost
            )

        else:
            raise BaseProjectException(
                f"Must subtract between an instance of BaseProject with "
                f"another instance of BaseProject. "
                f"{other}({other.__class__.__qualname__}) is not an instance "
                f"of BaseProject."
            )

    def __rsub__(self, other):
        return self.__sub__(other)
