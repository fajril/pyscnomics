"""
CASE 0
"""

import numpy as np
from datetime import date
from dataclasses import dataclass, field

from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.econ.selection import (
    FluidType,
    CostType,
    VariableSplit082017,
    VariableSplit522017,
    VariableSplit132024,
    TaxSplitTypeCR,
    ContractType,
    OtherRevenue,
    TaxRegime,
    FTPTaxRegime,
    DeprMethod,
    SunkCostMethod,
    InflationAppliedTo,
    GrossSplitRegime,
    InitialYearAmortizationIncurred,
    NPVSelection,
    DiscountingMode,
)
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.costs import (
    CapitalCost,
    Intangible,
    OPEX,
    ASR,
    LBT,
    CostOfSales,
)
from pyscnomics.io.getattr import (
    convert_object,
    construct_lifting_attr,
    construct_cost_attr,
)


@dataclass
class Case0:

    contract_type: ContractType

    # Attributes associated with lifting
    lifting: dict = field(default_factory=lambda: {}, init=False, repr=False)

    # Attributes associated with costs
    capital: dict = field(default_factory=lambda: {}, init=False, repr=False)
    intangible: dict = field(default_factory=lambda: {}, init=False, repr=False)
    opex: dict = field(default_factory=lambda: {}, init=False, repr=False)
    asr: dict = field(default_factory=lambda: {}, init=False, repr=False)

    setup_arguments: dict = field(default_factory=lambda: {}, init=False, repr=False)
    class_arguments: dict = field(default_factory=lambda: {}, init=False, repr=False)
    contract_arguments: dict = field(default_factory=lambda: {}, init=False, repr=False)
    summary_arguments: dict = field(default_factory=lambda: {}, init=False, repr=False)

    def __post_init__(self):
        self.get_lifting()
        self.get_capital()
        self.get_intangible()
        self.get_opex()
        self.get_asr()
        self.get_lbt()
        self.get_cos()
        self.get_setup_arguments()
        self.get_class_arguments()
        self.get_contract_arguments()
        self.get_summary_arguments()

    def get_lifting(self):

        # Prepare lifting data: OIL
        lifting_oil = {
            "start_year": 2023,
            "end_year": 2032,
            "prod_year": np.array([2030, 2031, 2032]),
            "lifting_rate": np.array([100, 100, 100]),
            "price": np.array([120, 120, 120]),
            "fluid_type": FluidType.OIL,
        }

        # Prepare lifting data: GAS
        lifting_gas = {
            "start_year": 2023,
            "end_year": 2032,
            "prod_year": np.array([2030, 2031, 2032]),
            "lifting_rate": np.array([10, 10, 10]),
            "price": np.array([1, 1, 1]),
            "fluid_type": FluidType.GAS,
        }

        # Prepare lifting data: SULFUR
        lifting_sulfur = {
            "start_year": 2023,
            "end_year": 2032,
            "prod_year": np.array([2030, 2031, 2032]),
            "lifting_rate": np.array([10, 10, 10]),
            "price": np.array([1, 1, 1]),
            "fluid_type": FluidType.SULFUR,
        }

        # Store lifting data as the class's attribute: "self.lifting"
        self.lifting = {
            "oil": lifting_oil,
            "gas": lifting_gas,
            "sulfur": lifting_sulfur
        }

    def get_capital(self):

        # Prepare capital data: OIL
        capital_oil = {
            "start_year": 2023,
            "end_year": 2032,
            "expense_year": np.array(
                [
                    2023, 2024, 2025, 2026, 2027,
                    2028, 2029, 2030, 2031, 2032,
                ]
            ),
            "cost": np.array(
                [
                    200, 200, 200, 200,
                    50, 50,
                    100, 100, 100, 100,
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

        # Prepare capital data: GAS
        capital_gas = {
            "start_year": 2023,
            "end_year": 2032,
            "expense_year": np.array(
                [
                    2023, 2024, 2025, 2026, 2027,
                    2028, 2029, 2030, 2031, 2032,
                ]
            ),
            "cost": np.array(
                [
                    20, 20, 20, 20,
                    5, 5,
                    10, 10, 10, 10,
                ]
            ),
            "cost_allocation": (
                [
                    FluidType.GAS, FluidType.GAS,
                    FluidType.GAS, FluidType.GAS,
                    FluidType.GAS, FluidType.GAS,
                    FluidType.GAS, FluidType.GAS,
                    FluidType.GAS, FluidType.GAS,
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

        # Store capital costs as the class's attribute: "self.capital"
        self.capital = {
            "oil": capital_oil,
            "gas": capital_gas
        }

    def get_intangible(self):

        # Prepare intangible cost: OIL
        intangible_oil = {
            "start_year": 2023,
            "end_year": 2032,
            "expense_year": np.array(
                [
                    2023, 2024, 2025, 2026, 2027,
                    2028, 2029, 2030, 2031, 2032,
                ]
            ),
            "cost": np.array(
                [
                    200, 200, 200, 200,
                    50, 50,
                    100, 100, 100, 100,
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

        # Prepare intangible cost: GAS
        intangible_gas = {
            "start_year": 2023,
            "end_year": 2032,
            "expense_year": np.array(
                [
                    2023, 2024, 2025, 2026, 2027,
                    2028, 2029, 2030, 2031, 2032,
                ]
            ),
            "cost": np.array(
                [
                    20, 20, 20, 20,
                    5, 5,
                    10, 10, 10, 10,
                ]
            ),
            "cost_allocation": (
                [
                    FluidType.GAS, FluidType.GAS,
                    FluidType.GAS, FluidType.GAS,
                    FluidType.GAS, FluidType.GAS,
                    FluidType.GAS, FluidType.GAS,
                    FluidType.GAS, FluidType.GAS,
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

        # Store intangible cost as the class's attribute: "self.intangible"
        self.intangible = {
            "oil": intangible_oil,
            "gas": intangible_gas,
        }

    def get_opex(self):
        pass

    def get_asr(self):
        pass

    def get_lbt(self):
        pass

    def get_cos(self):
        pass

    def get_setup_arguments(self):
        pass

    def get_class_arguments(self):
        pass

    def get_contract_arguments(self):
        pass

    def get_summary_arguments(self):
        pass

    def as_dict(self):
        pass

    def as_class(self):
        pass
