"""
Example case to be executed.
"""

import numpy as np
from dataclasses import dataclass, field

from pyscnomics.econ.selection import FluidType, SunkCostInvestmentType
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.costs import (
    CapitalCost,
    Intangible,
    OPEX,
    ASR,
    LBT,
    CostOfSales,
)


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
            prod_year=np.array([2025, 2026, 2027, 2028, 2027, 2025]),
            lifting_rate=np.array([10, 10, 10, 10, 10, 10]),
            price=np.array([10, 10, 10, 10, 10, 10]),
            fluid_type=FluidType.OIL,
        )

        self.lifting_apel = Lifting(
            start_year=2023,
            end_year=2030,
            prod_year=np.array([2024, 2024, 2024]),
            lifting_rate=np.array([np.nan, 10, None]),
            price=np.array([2, 2, 2]),
            fluid_type=FluidType.GAS,
        )

    def _get_capital_cost_data(self):
        """
        Generate synthetic data for capital cost.
        """
        self.capital_mangga = CapitalCost(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2023, 2024, 2025, 2024]),
            cost=np.array([25, 100, 100, 25]),
            cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL],
            is_sunkcost=[True, False, False, True],
        )

        self.capital_apel = CapitalCost(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2023, 2024, 2025, 2024]),
            cost=np.array([12.5, 50, 50, 12.5]),
            cost_allocation=[FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS],
            is_sunkcost=[True, False, False, True],
        )

    def _get_intangible_cost_data(self):
        """
        Generate synthetic data for intangible cost.
        """
        # Synthetic data: Intangible
        self.intangible_mangga = Intangible(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2023, 2024, 2025, 2024]),
            cost=np.array([100, 100, 50, 50]),
            cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL],
            is_sunkcost=[False, False, True, True],
        )

        self.intangible_apel = Intangible(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2023, 2024, 2025, 2024]),
            cost=np.array([10, 10, 5, 5]),
            cost_allocation=[FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS],
            is_sunkcost=[False, False, True, True],
        )

    def _get_opex_data(self):
        """
        Generate synthetic data for OPEX.
        """
        self.opex_mangga = OPEX(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2023, 2024, 2025, 2024]),
            fixed_cost=np.array([100, 100, 50, 50]),
            cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL],
            is_sunkcost=[False, False, True, True],
        )

        self.opex_apel = OPEX(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2023, 2024, 2025, 2024]),
            fixed_cost=np.array([10, 10, 5, 5]),
            cost_allocation=[FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS],
            is_sunkcost=[False, False, True, True],
        )

    def _get_asr_cost_data(self):
        """
        Generate synthetic data for ASR cost.
        """
        self.asr_mangga = ASR(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2023, 2024, 2025, 2024]),
            cost=np.array([100, 100, 50, 50]),
            cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL],
            is_sunkcost=[False, False, True, True],
        )

        self.asr_apel = ASR(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2023, 2024, 2025, 2024]),
            cost=np.array([10, 10, 5, 5]),
            cost_allocation=[FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS],
            is_sunkcost=[False, False, True, True],
        )

    def _get_lbt_cost_data(self):
        """
        Generate synthetic data for LBT cost.
        """
        # Synthetic data: LBT
        self.lbt_mangga = LBT(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2023, 2024, 2025, 2024]),
            cost=np.array([100, 100, 50, 50]),
            cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL],
            is_sunkcost=[False, False, True, True],
        )

        self.lbt_apel = LBT(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2023, 2024, 2025, 2024]),
            cost=np.array([10, 10, 5, 5]),
            cost_allocation=[FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS],
            is_sunkcost=[False, False, True, True],
        )

    def _get_cost_of_sales_data(self):
        """
        Generate synthetic data for cost of sales.
        """
        self.cos_mangga = CostOfSales(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2023, 2024, 2025, 2024]),
            cost=np.array([np.nan, 100, 100, None]),
            cost_allocation=[np.nan, FluidType.OIL, FluidType.OIL, None],
            is_sunkcost=[np.nan, False, False, None],
        )

        self.cos_apel = CostOfSales(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2023, 2024, 2025, 2024]),
            cost=np.array([np.nan, 10, 10, None]),
            cost_allocation=[FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS],
            is_sunkcost=[True, True, True, True],
        )
