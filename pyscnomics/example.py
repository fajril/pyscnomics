"""
Example case to be executed.
"""

import numpy as np
from dataclasses import dataclass, field

from pyscnomics.econ.selection import FluidType, CostType
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.costs import (
    CapitalCost,
    Intangible,
    OPEX,
    ASR,
    LBT,
    CostOfSales,
)


# Synthetic costs data
expense_year_cases = {
    "oil": np.array([2023, 2024, 2025, 2026, 2027]),
    "gas": np.array([2023, 2024, 2025, 2026, 2027]),
    "all_sunkcost": np.array([2023, 2024, 2023, 2024, 2023]),
    "all_postonstream": np.array([2029, 2030, 2029, 2029, 2030]),
    "all_preonstream": np.array([2026, 2026, 2026, 2026, 2026]),
}

cost_cases = {
    "oil": np.array([200, 150, 100, 75, 50]),
    "gas": np.array([20, 15, 10, 7.5, 5]),
}

cost_allocation_cases = {
    "oil": [FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL],
    "gas": [FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS],
}

cost_type_cases = {
    "oil": [
        CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
        CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST
    ],
    "gas": [
        CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
        CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST
    ],
    "all_sunkcost": [
        CostType.SUNK_COST, CostType.SUNK_COST, CostType.SUNK_COST,
        CostType.SUNK_COST, CostType.SUNK_COST
    ],
    "all_postonstream": [
        CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
        CostType.PRE_ONSTREAM_COST, CostType.POST_ONSTREAM_COST
    ]
}

tax_portion_cases = {
    "oil": np.array([1, 1, 1, 1, 1]),
    "gas": np.array([1, 1, 1, 1, 1]),
}


@dataclass
class ExampleCase:
    """
    A class consisting a collection of prescribed synthetic data designed as
    example to run class BaseProject, CostRecovery, and GrossSplit.
    """

    lifting_mangga: Lifting = field(default=None, init=False, repr=False)
    lifting_apel: Lifting = field(default=None, init=False, repr=False)
    lifting_nanas: Lifting = field(default=None, init=False, repr=False)
    capital_mangga: CapitalCost = field(default=None, init=False, repr=False)
    capital_apel: CapitalCost = field(default=None, init=False, repr=False)
    intangible_mangga: Intangible = field(default=None, init=False, repr=False)
    intangible_apel: Intangible = field(default=None, init=False, repr=False)
    opex_mangga: OPEX = field(default=None, init=False, repr=False)
    opex_apel: OPEX = field(default=None, init=False, repr=False)
    asr_mangga: ASR = field(default=None, init=False, repr=False)
    asr_apel: ASR = field(default=None, init=False, repr=False)
    lbt_mangga: LBT = field(default=None, init=False, repr=False)
    lbt_apel: LBT = field(default=None, init=False, repr=False)
    cos_mangga: CostOfSales = field(default=None, init=False, repr=False)
    cos_apel: CostOfSales = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._get_lifting_data()
        self._get_capital_cost_data()
        self._get_intangible_cost_data()
        self._get_opex_data()
        self._get_asr_cost_data()
        self._get_lbt_cost_data()
        self._get_cost_of_sales_data()

    def _get_lifting_data(self):
        """
        Generate synthetic lifting data.
        """
        self.lifting_mangga = Lifting(
            start_year=2023,
            end_year=2030,
            prod_year=np.array([2026, 2028, 2027, 2028, 2030, 2029]),
            lifting_rate=np.array([100, 100, 100, 200, 200, 200]),
            price=np.array([10, 10, 10, 5, 5, 5]),
            fluid_type=FluidType.OIL,
        )

        self.lifting_apel = Lifting(
            start_year=2023,
            end_year=2030,
            prod_year=np.array([2028, 2028, 2029, 2029, 2030, 2030]),
            lifting_rate=np.array([10, 10, 10, 20, 20, 20]),
            price=np.array([2, 2, 2, 1, 1, 1]),
            fluid_type=FluidType.GAS,
        )

        self.lifting_nanas = Lifting(
            start_year=2023,
            end_year=2030,
            prod_year=np.array([2027, 2028, 2027, 2028, 2030, 2029]),
            lifting_rate=np.array([10, 10, 10, 20, 20, 20]),
            price=np.array([2, 2, 2, 1, 1, 1]),
            fluid_type=FluidType.SULFUR,
        )

    def _get_capital_cost_data(self):
        """
        Generate synthetic data for capital cost.
        """
        self.capital_mangga = CapitalCost(
            start_year=2023,
            end_year=2030,
            expense_year=expense_year_cases["oil"],
            cost=cost_cases["oil"],
            cost_allocation=cost_allocation_cases["oil"],
            cost_type=cost_type_cases["oil"],
            tax_portion=tax_portion_cases["oil"],
        )

        self.capital_apel = CapitalCost(
            start_year=2023,
            end_year=2030,
            expense_year=expense_year_cases["gas"],
            cost=cost_cases["gas"],
            cost_allocation=cost_allocation_cases["gas"],
            cost_type=cost_type_cases["gas"],
            tax_portion=tax_portion_cases["gas"],
        )

    def _get_intangible_cost_data(self):
        """
        Generate synthetic data for intangible cost.
        """
        self.intangible_mangga = Intangible(
            start_year=2023,
            end_year=2030,
            expense_year=expense_year_cases["oil"],
            cost=cost_cases["oil"],
            cost_allocation=cost_allocation_cases["oil"],
            cost_type=cost_type_cases["oil"],
            tax_portion=tax_portion_cases["oil"],
        )

        self.intangible_apel = Intangible(
            start_year=2023,
            end_year=2030,
            expense_year=expense_year_cases["gas"],
            cost=cost_cases["gas"],
            cost_allocation=cost_allocation_cases["gas"],
            cost_type=cost_type_cases["gas"],
            tax_portion=tax_portion_cases["gas"],
        )

    def _get_opex_data(self):
        """
        Generate synthetic data for OPEX.
        """
        self.opex_mangga = OPEX(
            start_year=2023,
            end_year=2030,
            expense_year=expense_year_cases["oil"],
            fixed_cost=cost_cases["oil"],
            cost_allocation=cost_allocation_cases["oil"],
            cost_type=cost_type_cases["oil"],
            tax_portion=tax_portion_cases["oil"],
        )

        self.opex_apel = OPEX(
            start_year=2023,
            end_year=2030,
            expense_year=expense_year_cases["gas"],
            fixed_cost=cost_cases["gas"],
            cost_allocation=cost_allocation_cases["gas"],
            cost_type=cost_type_cases["gas"],
            tax_portion=tax_portion_cases["gas"],
        )

    def _get_asr_cost_data(self):
        """
        Generate synthetic data for ASR cost.
        """
        self.asr_mangga = ASR(
            start_year=2023,
            end_year=2030,
            expense_year=expense_year_cases["oil"],
            cost=cost_cases["oil"],
            cost_allocation=cost_allocation_cases["oil"],
            cost_type=cost_type_cases["oil"],
            tax_portion=tax_portion_cases["oil"],
        )

        self.asr_apel = ASR(
            start_year=2023,
            end_year=2030,
            expense_year=expense_year_cases["gas"],
            cost=cost_cases["gas"],
            cost_allocation=cost_allocation_cases["gas"],
            cost_type=cost_type_cases["gas"],
            tax_portion=tax_portion_cases["gas"],
        )

    def _get_lbt_cost_data(self):
        """
        Generate synthetic data for LBT cost.
        """
        self.lbt_mangga = LBT(
            start_year=2023,
            end_year=2030,
            expense_year=expense_year_cases["oil"],
            cost=cost_cases["oil"],
            cost_allocation=cost_allocation_cases["oil"],
            cost_type=cost_type_cases["oil"],
            tax_portion=tax_portion_cases["oil"],
        )

        self.lbt_apel = LBT(
            start_year=2023,
            end_year=2030,
            expense_year=expense_year_cases["gas"],
            cost=cost_cases["gas"],
            cost_allocation=cost_allocation_cases["gas"],
            cost_type=cost_type_cases["gas"],
            tax_portion=tax_portion_cases["gas"],
        )

    def _get_cost_of_sales_data(self):
        """
        Generate synthetic data for cost of sales.
        """
        self.cos_mangga = CostOfSales(
            start_year=2023,
            end_year=2030,
            expense_year=expense_year_cases["oil"],
            cost=cost_cases["oil"],
            cost_allocation=cost_allocation_cases["oil"],
            cost_type=cost_type_cases["oil"],
        )

        self.cos_apel = CostOfSales(
            start_year=2023,
            end_year=2030,
            expense_year=expense_year_cases["gas"],
            cost=cost_cases["gas"],
            cost_allocation=cost_allocation_cases["gas"],
            cost_type=cost_type_cases["gas"],
        )
