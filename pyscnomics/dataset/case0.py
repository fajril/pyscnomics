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
    """
    A container class for building and managing base case contract data.

    This class initializes predefined datasets for lifting, capital, intangible,
    and other economic parameters based on the selected contract type. It provides
    helper methods to generate cost, lifting, and contract argument structures
    required for PSC-type economic simulations.

    Parameters
    ----------
    contract_type : ContractType
        The type of contract to initialize (e.g., Base Project, Cost Recovery,
        or Gross Split).

    Notes
    -----
    - The class auto-initializes its attributes during instantiation.
    - The design emphasizes clarity and traceability over computational efficiency.
    - Methods like `as_dict()` and `as_class()` are placeholders for data transformation.
    """

    contract_type: ContractType

    # Attributes associated with lifting
    lifting: dict = field(default_factory=lambda: {}, init=False, repr=False)

    # Attributes associated with costs
    capital: dict = field(default_factory=lambda: {}, init=False, repr=False)
    intangible: dict = field(default_factory=lambda: {}, init=False, repr=False)
    opex: dict = field(default_factory=lambda: {}, init=False, repr=False)
    asr: dict = field(default_factory=lambda: {}, init=False, repr=False)
    lbt: dict = field(default_factory=lambda: {}, init=False, repr=False)
    cos: dict = field(default_factory=lambda: {}, init=False, repr=False)

    # Attributes associated with arguments
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

    def get_lifting(self) -> None:
        """
        Initialize and store lifting data for each fluid type.

        This method prepares sample lifting datasets for oil, gas, and sulfur,
        including production years, lifting rates, and prices.

        The resulting data are stored in the class attribute `self.lifting`.

        Returns
        -------
        None

        Notes
        -----
        The generated lifting data are hardcoded examples for demonstration or testing
        purposes and should be replaced with actual input data in real evaluations.
        """

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

    def get_capital(self) -> None:
        """
        Initialize and store capital cost data for oil and gas.

        This method defines yearly capital expenditures for oil and gas,
        including expense years, cost values, allocation, cost types, and
        tax portions, then stores them in the `self.capital` attribute.

        Returns
        -------
        None

        Notes
        -----
        The capital data are predefined for demonstration purposes and
        should be replaced with actual input data in production analyses.
        """

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

    def get_intangible(self) -> None:
        """
        Initialize and store intangible cost data for oil and gas.

        This method defines yearly intangible expenditures for oil and gas,
        including expense years, cost values, allocation, cost types, and
        tax portions, then stores them in the `self.intangible` attribute.

        Returns
        -------
        None

        Notes
        -----
        The intangible cost data are predefined for illustrative purposes
        and should be replaced with actual project data in practical use.
        """

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

    def get_opex(self) -> None:
        """
        Initialize and store operating expenditure (OPEX) data for oil and gas.

        Defines yearly OPEX information such as expense years, fixed costs,
        allocation, cost type, and tax portion for both oil and gas, then
        stores them in the `self.opex` attribute.

        Notes
        -----
        The OPEX data are predefined for demonstration purposes and should
        be replaced with actual project-specific data in real applications.
        """

        # Prepare opex: OIL
        opex_oil = {
            "start_year": 2023,
            "end_year": 2032,
            "expense_year": np.array(
                [
                    2023, 2024, 2025, 2026, 2027,
                    2028, 2029, 2030, 2031, 2032,
                ]
            ),
            "fixed_cost": np.array(
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
            "tax_portion": (
                [
                    1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1,
                ]
            ),
        }

        # Prepare opex: GAS
        opex_gas = {
            "start_year": 2023,
            "end_year": 2032,
            "expense_year": np.array(
                [
                    2023, 2024, 2025, 2026, 2027,
                    2028, 2029, 2030, 2031, 2032,
                ]
            ),
            "fixed_cost": np.array(
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

        # Store opex as class's attribute: "self.opex"
        self.opex = {
            "oil": opex_oil,
            "gas": opex_gas
        }

    def get_asr(self) -> None:
        """
        Initialize and store Abandonment and Site Restoration (ASR) cost data
        for oil and gas.

        Defines yearly ASR information such as expense years, cost values,
        cost allocation, cost type, and tax portion for both oil and gas,
        then stores them in the `self.asr` attribute.

        Notes
        -----
        The ASR data are predefined for demonstration purposes and should
        be updated with actual project-specific values in real applications.
        """

        # Prepare ASR: OIL
        oil_asr = {
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

        # Prepare ASR: GAS
        gas_asr = {
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

        # Store ASR cost as class's attribute: "self.asr"
        self.asr = {
            "oil": oil_asr,
            "gas": gas_asr,
        }

    def get_lbt(self) -> None:
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
