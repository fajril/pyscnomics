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
    # VariableSplit522017,
    VariableSplit132024,
    TaxSplitTypeCR,
    ContractType,
    OtherRevenue,
    TaxRegime,
    FTPTaxRegime,
    DeprMethod,
    SunkCostMethod,
    # InflationAppliedTo,
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
    A container class for building and managing contract data for CASE 0.

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
        """
        Initialize all contract components after object creation.

        This method automatically populates key contract attributes such as
        lifting, capital, operating costs, fiscal terms, and summary parameters
        immediately after class instantiation.

        Notes
        -----
        - Ensures all essential arguments and cost components are initialized.
        - Designed to maintain object consistency without requiring manual setup calls.
        - Typically executed automatically after dataclass initialization.
        """

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
            "prod_year": np.array([2029, 2031, 2032]),
            "lifting_rate": np.array([10, 10, 10]),
            "price": np.array([1, 1, 1]),
            "fluid_type": FluidType.GAS,
        }

        # Prepare lifting data: SULFUR
        lifting_sulfur = {
            "start_year": 2023,
            "end_year": 2032,
            "prod_year": np.array([2029, 2031, 2032]),
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
            "tax_portion": np.array(
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
        """
        Initialize and store Land and Building Tax (LBT) cost data for oil and gas.

        Defines yearly LBT information including expense years, cost values,
        cost allocation, cost type, and tax portion for both oil and gas,
        then stores them in the `self.lbt` attribute.

        Notes
        -----
        The LBT data are predefined for illustrative purposes and should be
        updated with actual fiscal or project-specific inputs in real use cases.
        """

        # Prepare LBT: OIL
        lbt_oil = {
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

        # Prepare LBT: GAS
        lbt_gas = {
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

        # Store LBT cost data in class's attribute "self.lbt"
        self.lbt = {
            "oil": lbt_oil,
            "gas": lbt_gas,
        }

    def get_cos(self) -> None:
        """
        Initialize and store Cost of Sales (COS) data for oil and gas.

        Defines yearly COS information including expense years, cost values,
        cost allocation, cost type, and tax portion for both oil and gas,
        then stores them in the `self.cos` attribute.

        Notes
        -----
        The COS data are predefined placeholders intended for demonstration.
        Replace these values with actual project-specific cost data in practice.
        """

        # Prepare COS: OIL
        cos_oil = {
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

        # Prepare COS: GAS
        cos_gas = {
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

        # Store Cost Of Sales data in class's attribute "self.cos"
        self.cos = {
            "oil": cos_oil,
            "gas": cos_gas
        }

    def get_setup_arguments(self) -> None:
        """
        Define and store general setup arguments for the project.

        Initializes key project timeline parameters such as start and end dates,
        onstream dates for oil and gas, approval year, and POD-1 status, then
        stores them in the `self.setup_arguments` attribute.

        Notes
        -----
        The setup arguments provide a foundational timeline reference for
        subsequent economic and technical evaluations.
        """
        self.setup_arguments = {
            "start_date": date(year=2023, month=1, day=1),
            "end_date": date(year=2032, month=12, day=31),
            "oil_onstream_date": date(year=2030, month=1, day=1),
            "gas_onstream_date": date(year=2029, month=1, day=1),
            "approval_year": 2026,
            "is_pod_1": True,
        }

    def get_class_arguments(self) -> None:
        """
        Define and store contract-specific class arguments.

        Prepares configuration dictionaries for different PSC contract types
        (Cost Recovery, Gross Split, and Base Project) and assigns the
        appropriate one to `self.class_arguments` based on the selected
        `contract_type`.

        Notes
        -----
        The method ensures that each contract type uses consistent and
        predefined parameters for fiscal terms, DMO, and cost recovery rules.
        Raises a ValueError if the contract type is not recognized.
        """

        # Cost recovery
        kwargs_cost_recovery = {
            # FTP
            "oil_ftp_is_available": True,
            "oil_ftp_is_shared": True,
            "oil_ftp_portion": 0.2,
            "gas_ftp_is_available": True,
            "gas_ftp_is_shared": True,
            "gas_ftp_portion": 0.2,

            # Tax split
            "tax_split_type": TaxSplitTypeCR.CONVENTIONAL,
            "condition_dict": dict,
            "indicator_rc_icp_sliding": None,
            "oil_ctr_pretax_share": 0.25,
            "gas_ctr_pretax_share": 0.5,

            # Investment credit
            "oil_ic_rate": 0.0,
            "gas_ic_rate": 0.0,
            "ic_is_available": False,
            "oil_cr_cap_rate": 1.0,
            "gas_cr_cap_rate": 1.0,

            # DMO
            "oil_dmo_volume_portion": 0.25,
            "oil_dmo_fee_portion": 0.25,
            "oil_dmo_holiday_duration": 60,
            "gas_dmo_volume_portion": 1.0,
            "gas_dmo_fee_portion": 1.0,
            "gas_dmo_holiday_duration": 60,

            # Carry forward depreciation
            "oil_carry_forward_depreciation": 0.0,
            "gas_carry_forward_depreciation": 0.0,
        }

        # Gross split
        VS_08 = VariableSplit082017
        VS_13 = VariableSplit132024

        kwargs_gross_split = {
            # Field and reservoir properties
            "field_status": VS_08.FieldStatus.NO_POD,
            "field_loc": VS_08.FieldLocation.ONSHORE,
            "res_depth": VS_08.ReservoirDepth.LESSEQUAL_2500,
            "infra_avail": VS_08.InfrastructureAvailability.WELL_DEVELOPED,
            "res_type": VS_08.ReservoirType.CONVENTIONAL,
            "api_oil": VS_08.APIOil.LESSTHAN_25,
            "domestic_use": VS_08.DomesticUse.EQUAL_50_UNTIL_LESSTHAN_70,
            "prod_stage": VS_08.ProductionStage.SECONDARY,
            "co2_content": VS_08.CO2Content.LESSTHAN_5,
            "h2s_content": VS_08.H2SContent.LESSTHAN_100,
            "field_reserves_2024": VS_13.FieldReservesAmount.MEDIUM,
            "infra_avail_2024": VS_13.InfrastructureAvailability.PARTIALLY_AVAILABLE,
            "field_loc_2024": VS_13.FieldLocation.ONSHORE,

            # Ministry discretion
            "split_ministry_disc": 0.08,

            # DMO
            "oil_dmo_volume_portion": 0.25,
            "oil_dmo_fee_portion": 1.0,
            "gas_dmo_volume_portion": 1.0,
            "gas_dmo_fee_portion": 1.0,
            "oil_dmo_holiday_duration": 60,
            "gas_dmo_holiday_duration": 60,

            # Carry forward depreciation
            "oil_carry_forward_depreciation": 0.0,
            "gas_carry_forward_depreciation": 0.0,
        }

        class_args_map = {
            ContractType.COST_RECOVERY: kwargs_cost_recovery,
            ContractType.GROSS_SPLIT: kwargs_gross_split,
            ContractType.BASE_PROJECT: {},
        }

        try:
            self.class_arguments = class_args_map[self.contract_type]

        except KeyError:
            raise ValueError(f"Unrecognized contract type: {self.contract_type!r}")

    def get_contract_arguments(self) -> None:
        """
        Define and store contract-level configuration arguments.

        Prepares parameter dictionaries for each PSC contract type
        (Cost Recovery, Gross Split, and Base Project) and assigns the
        appropriate one to `self.contract_arguments` based on the current
        `contract_type`.

        Notes
        -----
        This method centralizes fiscal and technical configuration settings
        such as tax regime, depreciation method, and revenue treatment.
        Raises a ValueError if the contract type is not recognized.
        """

        # Base project
        args_base_project = {
            "sulfur_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
            "electricity_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
            "co2_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
            "vat_rate": 0.0,
            "year_inflation": None,
            "inflation_rate": 0.0,
            "inflation_rate_applied_to": None,
        }

        # Cost recovery
        args_cost_recovery = {
            **args_base_project,
            "is_dmo_end_weighted": False,
            "tax_regime": TaxRegime.NAILED_DOWN,
            "effective_tax_rate": None,
            "ftp_tax_regime": FTPTaxRegime.PDJP_20_2017,
            "depr_method": DeprMethod.PSC_DB,
            "decline_factor": 2,
            "post_uu_22_year2001": True,
            "oil_cost_of_sales_applied": False,
            "gas_cost_of_sales_applied": False,
            "sum_undepreciated_cost": False,
            "sunk_cost_method": SunkCostMethod.DEPRECIATED_TANGIBLE,
        }

        # Base project
        args_gross_split = {
            **args_base_project,
            "cum_production_split_offset": 0.0,
            "depr_method": DeprMethod.PSC_DB,
            "decline_factor": 2,
            "sum_undepreciated_cost": False,
            "is_dmo_end_weighted": False,
            "tax_regime": TaxRegime.NAILED_DOWN,
            "effective_tax_rate": 0.22,
            "amortization": False,
            "sunk_cost_method": SunkCostMethod.DEPRECIATED_TANGIBLE,
            "regime": GrossSplitRegime.PERMEN_ESDM_13_2024,
            "reservoir_type_permen_2024": VariableSplit132024.ReservoirType.MK,
            "initial_amortization_year": InitialYearAmortizationIncurred.ONSTREAM_YEAR,
        }

        # Pooled args
        ctr_args = {
            ContractType.COST_RECOVERY: args_cost_recovery,
            ContractType.GROSS_SPLIT: args_gross_split,
            ContractType.BASE_PROJECT: args_base_project,
        }

        try:
            self.contract_arguments = ctr_args[self.contract_type]

        except KeyError:
            raise ValueError(f"Unrecognized contract type: {self.contract_type!r}")

    def get_summary_arguments(self) -> None:
        """
        Define and store economic summary configuration parameters.

        Sets up key parameters for project evaluation such as discount rate,
        NPV mode, and discounting convention, and assigns them to
        `self.summary_arguments`.

        Notes
        -----
        These parameters control how economic indicators like NPV and
        profitability are calculated in the financial evaluation.
        """
        self.summary_arguments = {
            "discount_rate": 0.1,
            "npv_mode": NPVSelection.NPV_SKK_NOMINAL_TERMS,
            "discounting_mode": DiscountingMode.END_YEAR,
            "discount_rate_start_year": 2023,
            "inflation_rate": 0.0,
            "profitability_discounted": False,
        }

    def as_dict(self) -> dict:
        """
        Convert all contract-related attributes into a unified dictionary format.

        This method consolidates and converts setup, summary, contract, class,
        lifting, and cost attributes into serializable dictionary objects for
        export or reconstruction.

        Notes
        -----
        - Handles both Cost Recovery and Gross Split contract types.
        - Applies object conversion to ensure compatibility with JSON serialization.
        - Returns a mapping of all major project components (e.g., capital, OPEX, ASR).
        """

        # Helper function to convert data stored in an argument dictionary
        def _converter(source: dict):
            return {key: convert_object(objects=val) for key, val in source.items()}

        # Convert data in "setup_arguments", "summary_arguments", and "contract_arguments"
        setup = _converter(source=self.setup_arguments)
        summary_arguments = _converter(source=self.summary_arguments)
        contract_arguments = _converter(source=self.contract_arguments)

        # Convert data in "class_arguments"
        gs = (
            _converter(source=self.class_arguments)
            if self.contract_type == ContractType.GROSS_SPLIT
            else None
        )

        cr = (
            _converter(source=self.class_arguments)
            if self.contract_type == ContractType.COST_RECOVERY
            else None
        )

        # Convert data in "lifting"
        lifting = construct_lifting_attr(
            lifting=tuple([Lifting(**lft) for lft in self.lifting.values()])
        )

        # Convert data in "capital", "intangible", "opex", "asr", "lbt", "cost_of_sales"
        # Helper method to convert data associated with costs
        def _construct_cost_attributes(source: dict, Cls):
            items = tuple([Cls(**val) for val in source.values()])
            return construct_cost_attr(cost=items)

        cap = _construct_cost_attributes(source=self.capital, Cls=CapitalCost)
        itg = _construct_cost_attributes(source=self.intangible, Cls=Intangible)
        op = _construct_cost_attributes(source=self.opex, Cls=OPEX)
        asr = _construct_cost_attributes(source=self.asr, Cls=ASR)
        lbt = _construct_cost_attributes(source=self.lbt, Cls=LBT)
        cos = _construct_cost_attributes(source=self.cos, Cls=CostOfSales)

        # Mapping converted data
        mapping_converted_data = (
            ("setup", setup),
            ("summary_arguments", summary_arguments),
            ("contract_arguments", contract_arguments),
            ("grosssplit", gs),
            ("costrecovery", cr),
            ("lifting", lifting),
            ("capital", cap),
            ("intangible", itg),
            ("opex", op),
            ("asr", asr),
            ("lbt", lbt),
            ("cost_of_sales", cos),
        )

        return {key: val for key, val in mapping_converted_data}

    def as_class(self) -> CostRecovery | GrossSplit | BaseProject:
        """
        Construct and return a contract instance based on the specified contract type.

        Combines setup, class, and cost-related arguments to instantiate the
        appropriate contract object (CostRecovery, GrossSplit, or BaseProject).

        Notes
        -----
        - The method dynamically builds input arguments from previously defined
          attributes such as lifting, capital, OPEX, and ASR.
        - The selected class is determined by `self.contract_type`.
        - Raises a ValueError if the contract type is unrecognized.
        """

        fluids_lifting = ["oil", "gas", "sulfur"]
        fluids_costs = ["oil", "gas"]

        fl = fluids_lifting
        fc = fluids_costs

        # Create per fluid instances for lifting and each cost category
        instances = {
            "lifting": {f: Lifting(**self.lifting[f]) for f in fl},
            "capital": {f: CapitalCost(**self.capital[f]) for f in fc},
            "intangible": {f: Intangible(**self.intangible[f]) for f in fc},
            "opex": {f: OPEX(**self.opex[f]) for f in fc},
            "asr": {f: ASR(**self.asr[f]) for f in fc},
            "lbt": {f: LBT(**self.lbt[f]) for f in fc},
            "cos": {f: CostOfSales(**self.cos[f]) for f in fc},
        }

        # Construct tuples from the created instances
        instances_as_tuple = {
            "lifting": tuple(instances["lifting"].values()),
            "capital_cost": tuple(instances["capital"].values()),
            "intangible_cost": tuple(instances["intangible"].values()),
            "opex": tuple(instances["opex"].values()),
            "asr_cost": tuple(instances["asr"].values()),
            "lbt_cost": tuple(instances["lbt"].values()),
            "cost_of_sales": tuple(instances["cos"].values()),
        }

        # Merge keyword arguments to create an instance of contract
        kwargs_merged = {
            **self.setup_arguments,     # setup arguments
            **instances_as_tuple,       # lifting and costs arguments
            **self.class_arguments,     # class's arguments
        }

        # Create and return the appropriate contract instance
        if self.contract_type == ContractType.COST_RECOVERY:
            return CostRecovery(**kwargs_merged)
        elif self.contract_type == ContractType.GROSS_SPLIT:
            return GrossSplit(**kwargs_merged)
        elif self.contract_type == ContractType.BASE_PROJECT:
            return BaseProject(**kwargs_merged)
        else:
            raise ValueError(f"Unrecognized contract type: {self.contract_type!r}")
