"""
A collection of unit testing for class CostRecovery
"""

import pytest
import numpy as np
from datetime import date
from dataclasses import dataclass, field
from pyscnomics.econ.selection import FluidType
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.costs import CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales
from pyscnomics.contracts.costrecovery import CostRecovery


@dataclass
class CostRecoveryExample:
    oil_lifting_mangga: Lifting = field(default=None, init=False, repr=False)
    oil_lifting_apel: Lifting = field(default=None, init=False, repr=False)
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

    def _get_lifting_data(self):
        """ Generate synthetic lifting data """

        self.oil_lifting_mangga = Lifting(
            start_year=2023,
            end_year=2030,
            prod_year=np.array([2027, 2028, 2027, 2028, 2029, 2030]),
            lifting_rate=np.array([100, 100, 100, 200, 200, 200]),
            price=np.array([10, 10, 10, 5, 5, 5]),
            fluid_type=FluidType.OIL,
        )

        self.oil_lifting_apel = Lifting(
            start_year=2023,
            end_year=2030,
            prod_year=np.array([2027, 2028, 2027, 2028, 2029, 2030]),
            lifting_rate=np.array([10, 10, 10, 20, 20, 20]),
            price=np.array([2, 2, 2, 1, 1, 1]),
            fluid_type=FluidType.GAS,
        )

        self.lifting_nanas = Lifting(
            start_year=2023,
            end_year=2030,
            prod_year=np.array([2027, 2028, 2027, 2028, 2029, 2030]),
            lifting_rate=np.array([10, 10, 10, 20, 20, 20]),
            price=np.array([2, 2, 2, 1, 1, 1]),
            fluid_type=FluidType.SULFUR,
        )

    def _get_capital_cost_data(self):
        self.capital_mangga = CapitalCost(
            start_year=2023,
            end_year=2030,
            expense_year=np.array([2023, 2024, 2025, 2026, 2027, 2028]),
            cost=np.array([100, 100, 50, 50, 25, 25]),
            cost_allocation=[
                FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL,
            ],
            is_sunkcost=[True, True, True, True, True, True],
            tax_portion=np.array([1, 1, 1, 1, 1, 1]),
        )

        pass

    def _get_intangible_cost_data(self):
        pass

    def _get_opex_data(self):
        pass

    def _get_asr_cost_data(self):
        pass

    def _get_lbt_cost_data(self):
        pass

    def _get_cost_of_sales_data(self):
        pass


