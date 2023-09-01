"""
Configure base project as a foundational framework for contract.
"""

from dataclasses import dataclass, field
from datetime import date
from functools import reduce

import numpy as np

from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.selection import FluidType
from pyscnomics.econ.costs import (
    Tangible,
    Intangible,
    OPEX,
    ASR,
)
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
    onstream_date: date
    lifting: tuple
    tangible_cost: tuple = field(default=None, repr=False)
    intangible_cost: tuple = field(default=None, repr=False)
    opex: tuple = field(default=None, repr=False)
    asr_cost: tuple = field(default=None, repr=False)

    def __post_init__(self):

        # Specify project duration and project years, raise error for inappropriate start date
        if self.start_date < self.end_date:
            self.project_duration = self.end_date.year - self.start_date.year + 1
            self.project_years = np.arange(
                self.start_date.year, self.end_date.year + 1, 1
            )

        else:
            raise BaseProjectException(
                f"start date {self.start_date} "
                f"is after the end date: {self.end_date}"
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

        # TODO: write validation for costs date

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

        revenues = np.zeros(self.project_duration)

        for lft in self.lifting:
            revenue = revenue + lft.revenue()

        expenditures = np.zeros(self.project_duration)

        for cost in self.tangible_cost:
            expenditures = expenditures + cost.expenditures()

        for cost in self.intangible_cost:
            expenditures = expenditures + cost.expenditures()

        for cost in self.opex:
            expenditures = expenditures + cost.expenditures()

        for cost in self.asr_cost:
            expenditures = expenditures + cost.expenditures()

        return CashFlow(
            start_date=self.start_date,
            end_date=self.end_date,
            cash=revenues - expenditures,
        )

    def _get_oil_lifting(self):

        # Check the fluid type of Lifting instances
        fluid_types = [lft.fluid_type for lft in self.lifting]

        # There is no oil in fluid_types
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
