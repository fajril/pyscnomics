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
    SunkCost,
)


@dataclass
class ExampleCase:
    """
    A class consisting a collection of prescribed synthetic data designed as
    example to run class BaseProject, CostRecovery, and GrossSplit.
    """

    lifting_mangga: Lifting = field(default=None, init=False, repr=False)
    lifting_apel: Lifting = field(default=None, init=False, repr=False)
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
    sunk_cost_mangga: SunkCost = field(default=None, init=False, repr=False)
    sunk_cost_apel: SunkCost = field(default=None, init=False, repr=False)

    def __post_init__(self):

        # Synthetic data: Lifting
        self.lifting_mangga = Lifting(
            start_year=2023,
            end_year=2030,
            prod_year=np.array([2026, 2027, 2028, 2027]),
            lifting_rate=np.array([np.nan, None, 100, 100]),
            price=np.array([10, 10, 10, 10]),
            fluid_type=FluidType.OIL,
        )

        self.lifting_apel = Lifting(
            start_year=2023,
            end_year=2030,
            prod_year=np.array([2026, 2027, 2028, 2027]),
            lifting_rate=np.array([np.nan, None, 5, 5]),
            price=np.array([2, 2, 2, 2]),
            fluid_type=FluidType.GAS,
        )

        # Synthetic data: CapitalCost
        self.capital_mangga = CapitalCost(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2023, 2024, 2025, 2024]),
            cost=np.array([np.nan, 100, 100, None]),
            cost_allocation=[np.nan, FluidType.OIL, FluidType.OIL, None],
        )

        self.capital_apel = CapitalCost(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2023, 2024, 2025, 2024]),
            cost=np.array([np.nan, 10, 10, None]),
            cost_allocation=[np.nan, FluidType.GAS, FluidType.GAS, None],
        )

        # Synthetic data: Intangible
        self.intangible_mangga = Intangible(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2023, 2024, 2025, 2024]),
            cost=np.array([np.nan, 100, 100, None]),
            cost_allocation=[np.nan, FluidType.OIL, FluidType.OIL, None],
        )

        self.intangible_apel = Intangible(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2023, 2024, 2025, 2024]),
            cost=np.array([np.nan, 10, 10, None]),
            cost_allocation=[np.nan, FluidType.GAS, FluidType.GAS, None],
        )

        # Synthetic data: OPEX
        self.opex_mangga = OPEX(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2023, 2024, 2025, 2024]),
            fixed_cost=np.array([np.nan, 100, 100, None]),
            cost_allocation=[np.nan, FluidType.OIL, FluidType.OIL, None],
        )

        self.opex_apel = OPEX(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2023, 2024, 2025, 2024]),
            fixed_cost=np.array([np.nan, 10, 10, None]),
            cost_allocation=[FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS],
        )

        # Synthetic data: ASR
        self.asr_mangga = ASR(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2023, 2024, 2025, 2024]),
            cost=np.array([np.nan, 100, 100, None]),
            cost_allocation=[np.nan, FluidType.OIL, FluidType.OIL, None],
        )

        self.asr_apel = ASR(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2023, 2024, 2025, 2024]),
            cost=np.array([np.nan, 10, 10, None]),
            cost_allocation=[FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS],
        )

        # Synthetic data: LBT
        self.lbt_mangga = LBT(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2023, 2024, 2025, 2024]),
            cost=np.array([np.nan, 100, 100, None]),
            cost_allocation=[np.nan, FluidType.OIL, FluidType.OIL, None],
        )

        self.lbt_apel = LBT(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2023, 2024, 2025, 2024]),
            cost=np.array([np.nan, 10, 10, None]),
            cost_allocation=[FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS],
        )

        # Synthetic data: CostOfSales
        self.cos_mangga = CostOfSales(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2023, 2024, 2025, 2024]),
            cost=np.array([np.nan, 100, 100, None]),
            cost_allocation=[np.nan, FluidType.OIL, FluidType.OIL, None],
        )

        self.cos_apel = CostOfSales(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2023, 2024, 2025, 2024]),
            cost=np.array([np.nan, 10, 10, None]),
            cost_allocation=[FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS],
        )

        # Synthetic data: SunkCost
        self.sunk_cost_mangga = SunkCost(
            start_year=2023,
            end_year=2030,
            onstream_year=2027,
            pod1_year=2025,
            expense_year=np.array([2023, 2024, 2024, 2026, 2025, 2027]),
            cost=np.array([50, 50, 50, 200, 50, 200]),
            cost_allocation=[
                FluidType.OIL,
                FluidType.OIL,
                FluidType.OIL,
                FluidType.OIL,
                FluidType.OIL,
                FluidType.OIL,
            ],
            investment_type=[
                SunkCostInvestmentType.TANGIBLE,
                SunkCostInvestmentType.TANGIBLE,
                SunkCostInvestmentType.TANGIBLE,
                SunkCostInvestmentType.INTANGIBLE,
                SunkCostInvestmentType.INTANGIBLE,
                SunkCostInvestmentType.INTANGIBLE,
            ],
        )

        self.sunk_cost_apel = SunkCost(
            start_year=2023,
            end_year=2030,
            onstream_year=2027,
            pod1_year=2025,
            expense_year=np.array([2023, 2024, 2024, 2026, 2025, 2027]),
            cost=np.array([5, 5, 5, 20, 5, 20]),
            cost_allocation=[
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
                FluidType.GAS,
            ],
            investment_type=[
                SunkCostInvestmentType.TANGIBLE,
                SunkCostInvestmentType.TANGIBLE,
                SunkCostInvestmentType.TANGIBLE,
                SunkCostInvestmentType.INTANGIBLE,
                SunkCostInvestmentType.INTANGIBLE,
                SunkCostInvestmentType.INTANGIBLE,
            ],
        )
