"""
A collection of function to generate a synthetic case
"""

import numpy as np
from datetime import date
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
from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.econ.selection import (
    TaxSplitTypeCR,
    OtherRevenue,
    InflationAppliedTo,
    TaxRegime,
    FTPTaxRegime,
    DeprMethod,
    SunkCostMethod,
    InitialYearAmortizationIncurred,
    GrossSplitRegime,
    VariableSplit082017,
    VariableSplit132024,
    NPVSelection,
    DiscountingMode,
)


# General cost kwargs for OIL
cost_oil_kwargs = {
    "start_year": 2023,
    "end_year": 2032,
    "expense_year": np.array(
        [
            2023, 2024, 2025, 2026, 2027,
            2028, 2029, 2030, 2031, 2032,
        ]
    ),
    "cost_allocation": (
        [
            FluidType.OIL, FluidType.OIL,
            FluidType.OIL, FluidType.OIL,
            FluidType.OIL, FluidType.OIL,
            FluidType.OIL, FluidType.OIL,
            FluidType.OIL, FluidType.OIL,
        ]
    ),
    "cost_type": (
        [
            CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
            CostType.PRE_ONSTREAM_COST, CostType.SUNK_COST,
            CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
            CostType.POST_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
            CostType.PRE_ONSTREAM_COST, CostType.PRE_ONSTREAM_COST,
        ]
    ),
    "tax_portion": np.array(
        [
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
        ]
    ),
}


@dataclass
class PrepareLiftingCosts:
    lifting: dict = field(default_factory=lambda: {}, init=False, repr=False)
    capital_cost: dict = field(default_factory=lambda : {}, init=False, repr=False)
    intangible_cost: dict = field(default_factory=lambda : {}, init=False, repr=False)
    opex: dict = field(default_factory=lambda : {}, init=False, repr=False)
    asr_cost: dict = field(default_factory=lambda : {}, init=False, repr=False)
    lbt_cost: dict = field(default_factory=lambda : {}, init=False, repr=False)
    cost_of_sales: dict = field(default_factory=lambda : {}, init=False, repr=False)

    def __post_init__(self):
        self._get_lifting_data()
        self._get_capital_data()

    def _get_lifting_data(self):

        # Lifting data for OIL
        lifting_oil = {
            "dummy": Lifting(
                start_year=2023,
                end_year=2032,
                prod_year=np.array([2030, 2031, 2032]),
                lifting_rate=np.array([100, 100, 100]),
                price=np.array([120, 120, 120]),
                fluid_type=FluidType.OIL,
            ),
            "benuang": Lifting(
                start_year=2023,
                end_year=2037,
                prod_year=np.array(
                    [
                        2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032,
                        2033, 2034, 2035
                    ]
                ),
                lifting_rate=np.array(
                    [
                        460.240475451, 873.845020492, 802.981698938, 932.529841824,
                        928.962143147, 962.198861205, 592.127288123, 688.191243627,
                        609.136555531, 459.317272429, 422.929806095, 324.406696975
                    ]
                ),
                price=np.array(
                    [65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65]
                ),
                fluid_type=FluidType.OIL,
            ),
        }

        # Lifting data for GAS
        lifting_gas = {
            "dummy": Lifting(
                start_year=2023,
                end_year=2032,
                prod_year=np.array([2029, 2030, 2031]),
                lifting_rate=np.array([10, 10, 10]),
                price=np.array([1, 1, 1]),
                fluid_type=FluidType.GAS,
            ),
            "benuang": Lifting(
                start_year=2023,
                end_year=2037,
                prod_year=np.array(
                    [
                        2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032,
                        2033, 2034, 2035
                    ]
                ),
                lifting_rate=np.array(
                    [
                        1.00495513, 1.96626503, 2.5581599, 4.76693091, 6.0547015,
                        6.25418605, 4.04543102, 4.58236067, 4.26727342, 2.46265306,
                        2.34562227, 2.10730217
                    ]
                ),
                price=np.array(
                    [
                        5.28767105, 5.23662554, 5.20574913, 5.50581852, 5.50165717,
                        5.50819591, 5.66529057, 5.61780989, 5.70251656, 5.68642526,
                        5.69206885, 5.66643698
                    ]
                ),
                ghv=np.array(
                    [
                        1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010,
                        1010, 1010
                    ]
                ),
                fluid_type=FluidType.GAS,
            ),
        }

        # Lifting data for SULFUR
        lifting_sulfur = {
            "dummy": Lifting(
                start_year=2023,
                end_year=2032,
                prod_year=np.array([2030, 2031, 2032]),
                lifting_rate=np.array([10, 10, 10]),
                price=np.array([1, 1, 1]),
                fluid_type=FluidType.SULFUR,
            ),
            "benuang": None,
        }

        # Lifting data for electricity
        lifting_electricity = {
            "dummy": None,
            "benuang": None,
        }

        # Lifting data for CO2
        lifting_co2 = {
            "dummy": None,
            "benuang": None,
        }

        # Specify attribute lifting
        self.lifting = {
            "oil": lifting_oil,
            "gas": lifting_gas,
            "sulfur": lifting_sulfur,
            "electricity": lifting_electricity,
            "co2": lifting_co2,
        }

    def _get_capital_data(self):

        # Capital cost data for OIL
        capital_cost_oil = {
            "dummy": {
                **cost_oil_kwargs,
                "cost": np.array(
                    [
                        200, 200, 200, 200, 50, 50,
                        100, 100, 100, 100,
                    ]
                ),
            },
            "benuang": {
                "start_year": 2023,
                "end_year": 2027,
                "expense_year": np.array(
                    [
                        2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033,
                        2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033
                    ]
                ),
                "cost_allocation": (
                    [
                        FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                        FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                        FluidType.OIL, FluidType.OIL, FluidType.GAS, FluidType.GAS,
                        FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                        FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                    ]
                ),
                "cost_type": (
                    [
                        CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                        CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                        CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                        CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                        CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                        CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                        CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                        CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                        CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                        CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    ]
                ),
                "cost": np.array(
                    [
                        25714.707321, 21578.906603, 18978.850357, 15273.457471,
                        7640.015402, 194.226795, 419.762154, 262.961878, 556.462722,
                        406.618528, 4613.344573, 3950.905221, 4890.837055, 6679.491296,
                        4256.871068, 108.051922, 252.454555, 152.843852, 345.419185,
                        192.630464,
                    ]
                ),
                "pis_year": np.array(
                    [
                        2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033,
                        2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033,
                    ]
                ),
                "useful_life": None,
                "depreciation_factor": None,
                "tax_portion": None,
            },
        }

        # Capital cost data for GAS
        capital_cost_gas = {}

        self.capital_cost = {
            "oil": None,
            "gas": None,
        }

    def _get_intangible_data(self):
        pass

    def _get_opex_data(self):
        pass

    def _get_asr_data(self):
        pass

    def _get_lbt_data(self):
        pass

    def _get_cost_of_sales_data(self):
        pass



