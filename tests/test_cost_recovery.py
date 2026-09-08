"""
A collection of unit testing for class CostRecovery
"""

from dataclasses import dataclass, field
from datetime import date
import numpy as np

from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.dataset.sample import load_data
from pyscnomics.econ.costs import CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.selection import CostType, FluidType


def test_cost_recovery_reuses_normalized_condition_dict():
    raw_conditions = {
        "RC Bottom Limit": [0.0, 1.0],
        "RC Top Limit": [1.0, 2.0],
        "Pre Tax CTR Oil": [0.25, 0.3],
        "Pre Tax CTR Gas": [0.5, 0.55],
    }
    contract = CostRecovery(
        start_date=date(2020, 1, 1),
        end_date=date(2021, 12, 31),
        approval_year=2020,
        condition_dict=raw_conditions,
    )

    copied = CostRecovery(
        start_date=date(2020, 1, 1),
        end_date=date(2021, 12, 31),
        approval_year=2020,
        condition_dict=contract.condition_dict,
    )

    assert copied.condition_dict == {
        "condition_1": {
            "bot_limit": 0.0,
            "top_limit": 1.0,
            "ctr_oil": 0.25,
            "ctr_gas": 0.5,
        },
        "condition_2": {
            "bot_limit": 1.0,
            "top_limit": 2.0,
            "ctr_oil": 0.3,
            "ctr_gas": 0.55,
        },
    }


def test_cost_recovery_unsupported_effective_tax_rate_defaults_to_zero():
    contract = load_data("case1", "cost_recovery")

    contract.run(effective_tax_rate="unsupported")

    assert contract.get_summary()["gov_tax_income"] == 0.0


@dataclass
class CostRecoveryExample:
    oil_lifting_mangga: Lifting | None = field(default=None, init=False, repr=False)
    oil_lifting_apel: Lifting | None = field(default=None, init=False, repr=False)
    lifting_nanas: Lifting | None = field(default=None, init=False, repr=False)
    capital_mangga: CapitalCost | None = field(default=None, init=False, repr=False)
    capital_apel: CapitalCost | None = field(default=None, init=False, repr=False)
    intangible_mangga: Intangible | None = field(default=None, init=False, repr=False)
    intangible_apel: Intangible | None = field(default=None, init=False, repr=False)
    opex_mangga: OPEX | None = field(default=None, init=False, repr=False)
    opex_apel: OPEX | None = field(default=None, init=False, repr=False)
    asr_mangga: ASR | None = field(default=None, init=False, repr=False)
    asr_apel: ASR | None = field(default=None, init=False, repr=False)
    lbt_mangga: LBT | None = field(default=None, init=False, repr=False)
    lbt_apel: LBT | None = field(default=None, init=False, repr=False)
    cos_mangga: CostOfSales | None = field(default=None, init=False, repr=False)
    cos_apel: CostOfSales | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._get_lifting_data()

    def _get_lifting_data(self):
        """Generate synthetic lifting data"""

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
                FluidType.OIL,
                FluidType.OIL,
                FluidType.OIL,
                FluidType.OIL,
                FluidType.OIL,
                FluidType.OIL,
            ],
            cost_type=[CostType.SUNK_COST for _ in range(6)],
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
