"""
A collection of required data for CASE 1: DUMMY
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
    PoolData,
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


@dataclass
class Case1:

    contract_type: ContractType

    # Attributes associated with lifting
    lifting: dict = field(default_factory=lambda: {}, init=False, repr=False)

    # Attributes associated with costs
    capital: dict = field(default_factory=lambda: {}, init=False, repr=False)
    intangible: dict = field(default_factory=lambda: {}, init=False, repr=False)
    opex: dict = field(default_factory=lambda: {}, init=False, repr=False)
    asr: dict = field(default_factory=lambda: {}, init=False, repr=False)
    lbt: dict = field(default_factory=lambda: {}, init=False, repr=False)
    cost_of_sales: dict = field(default_factory=lambda: {}, init=False, repr=False)

    kwargs_contract: dict = field(default_factory=lambda: {}, init=False, repr=False)
    contract_arguments: ContractType = field(default_factory=lambda: {}, init=False, repr=False)
    summary_arguments: dict = field(default_factory=lambda: {}, init=False, repr=False)

    def __post_init__(self):
        """
        Handles the following operations:
            -   Prepare lifting data
            -   Prepare capital data
            -   Prepare intangible data
            -   Prepare opex data
            -   Prepare asr data
            -   Prepare lbt data
            -   Prepare cost of sales data
        """
        self.get_lifting()
        self.get_capital()
        self.get_intangible()
        self.get_opex()
        self.get_asr()
        self.get_lbt()
        self.get_cost_of_sales()
        self.get_kwargs()
        self.get_contract_arguments()
        self.get_summary_arguments()

    def get_lifting(self) -> None:
        """
        Create and assign lifting data for oil and gas.

        Constructs lifting data for oil and gas as dictionaries, then
        instantiates corresponding `Lifting` objects. Both raw data and
        object instances are stored in `self.lifting`.

        Returns
        -------
        None
            Updates `self.lifting` with lifting data and instances.

        Attributes
        ----------
        self.lifting : dict
            Contains two sections:

            - **"instance"** : dict of {str: Lifting}
                Lifting object instances for "oil" and "gas".
            - **"dict"** : dict of {str: dict}
                Raw lifting data used to initialize each instance.
        """

        # Prepare lifting data: OIL
        lifting_oil = {
            "start_year": 2023,
            "end_year": 2037,
            "prod_year": np.array(
                [
                    2024, 2025, 2026, 2027, 2028, 2029,
                    2030, 2031, 2032, 2033, 2034, 2035
                ]
            ),
            "lifting_rate": np.array(
                [
                    460.240475451, 873.845020492, 802.981698938, 932.529841824,
                    928.962143147, 962.198861205, 592.127288123, 688.191243627,
                    609.136555531, 459.317272429, 422.929806095, 324.406696975
                ]
            ),
            "price": np.array(
                [
                    65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65
                ]
            ),
            "fluid_type": FluidType.OIL,
        }

        # Prepare lifting data: GAS
        lifting_gas = {
            "start_year": 2023,
            "end_year": 2037,
            "prod_year": np.array(
                [
                    2024, 2025, 2026, 2027, 2028, 2029,
                    2030, 2031, 2032, 2033, 2034, 2035
                ]
            ),
            "lifting_rate": np.array(
                [
                    1.00495513, 1.96626503, 2.5581599, 4.76693091,
                    6.0547015, 6.25418605, 4.04543102, 4.58236067,
                    4.26727342, 2.46265306, 2.34562227, 2.10730217
                ]
            ),
            "price": np.array(
                [
                    5.28767105, 5.23662554, 5.20574913, 5.50581852,
                    5.50165717, 5.50819591, 5.66529057, 5.61780989,
                    5.70251656, 5.68642526, 5.69206885, 5.66643698
                ]
            ),
            "ghv": np.array(
                [
                    1010, 1010, 1010, 1010, 1010, 1010,
                    1010, 1010, 1010, 1010, 1010, 1010,
                ]
            ),
            "fluid_type": FluidType.GAS,
        }

        # Store lifting as the class's attribute: "lifting"
        self.lifting = {
            "instance": {
                "oil": Lifting(**lifting_oil),
                "gas": Lifting(**lifting_gas),
            },
            "dict": {
                "oil": lifting_oil,
                "gas": lifting_gas,
            }
        }

    def get_capital(self) -> None:
        """
        Create and assign capital cost data for oil and gas.

        Builds capital cost data as dictionaries and initializes
        corresponding `CapitalCost` objects. Both raw data and instances
        are stored in `self.capital`.

        Returns
        -------
        None
            Updates `self.capital` with capital cost data and instances.

        Attributes
        ----------
        self.capital : dict
            Contains two sections:
            - **"instance"** : dict of {str: CapitalCost}
                Capital cost object instances for "oil" and "gas".
            - **"dict"** : dict of {str: dict}
                Raw capital cost data used to initialize each instance.
        """

        # Prepare capital data: OIL
        capital_oil = {
            "start_year": 2023,
            "end_year": 2037,
            "expense_year": np.array(
                [
                    2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033
                ]
            ),
            "cost": np.array(
                [
                    25714.707321, 21578.906603, 18978.850357, 15273.457471,
                    7640.015402, 194.226795, 419.762154, 262.961878, 556.462722,
                    406.618528
                ]
            ),
            "cost_allocation": (
                [
                    FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                    FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                    FluidType.OIL, FluidType.OIL
                ]
            ),
            "cost_type": (
                [
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST
                ]
            ),
            "pis_year": np.array(
                [
                    2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033
                ]
            ),
            "useful_life": np.array(
                [
                    5, 5, 5, 5, 5, 5, 5, 5, 5, 5
                ]
            ),
            "depreciation_factor": np.array(
                [
                    0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25
                ]
            ),
            "tax_portion": np.array(
                [
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1
                ]
            ),
        }

        # Prepare capital data: GAS
        capital_gas = {
            "start_year": 2023,
            "end_year": 2037,
            "expense_year": np.array(
                [
                    2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033
                ]
            ),
            "cost": np.array(
                [
                    4613.344573, 3950.905221, 4890.837055, 6679.491296, 4256.871068,
                    108.051922, 252.454555, 152.843852, 345.419185, 192.630464
                ]
            ),
            "cost_allocation": (
                [
                    FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                    FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                    FluidType.GAS, FluidType.GAS
                ]
            ),
            "cost_type": (
                [
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                ]
            ),
            "pis_year": np.array(
                [
                    2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033
                ]
            ),
            "useful_life": np.array(
                [
                    5, 5, 5, 5, 5, 5, 5, 5, 5, 5
                ]
            ),
            "depreciation_factor": np.array(
                [
                    0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25
                ]
            ),
            "tax_portion": np.array(
                [
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1
                ]
            ),
        }

        # Store capital costs as the class's attribute: "capital"
        self.capital = {
            "instance": {
                "oil": CapitalCost(**capital_oil),
                "gas": CapitalCost(**capital_gas),
            },
            "dict": {
                "oil": capital_oil,
                "gas": capital_gas,
            },
        }

    def get_intangible(self) -> None:
        """
        Create and assign intangible cost data for oil and gas.

        Builds intangible cost data as dictionaries and initializes
        corresponding `Intangible` objects. Both raw data and instances
        are stored in `self.intangible`.

        Returns
        -------
        None
            Updates `self.intangible` with intangible cost data and instances.

        Attributes
        ----------
        self.intangible : dict
            Contains two sections:
            - **"instance"** : dict of {str: Intangible}
                Intangible cost object instances for "oil" and "gas".
            - **"dict"** : dict of {str: dict}
                Raw intangible cost data used to initialize each instance.
        """

        # Prepare intangible cost: OIL
        intangible_oil = {
            "start_year": 2023,
            "end_year": 2037,
            "expense_year": np.array(
                [
                    2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033
                ]
            ),
            "cost": np.array(
                [
                    46849.00562, 41944.72775, 22099.70561, 23079.23951, 21928.15442,
                    718.934075, 978.88238, 830.661181, 1863.113433, 1594.436516
                ]
            ),
            "cost_allocation": (
                [
                    FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                    FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                    FluidType.OIL, FluidType.OIL
                ]
            ),
            "cost_type": (
                [
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST
                ]
            ),
            "tax_portion": np.array(
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
        }

        # Prepare instangible cost: GAS
        intangible_gas = {
            "start_year": 2023,
            "end_year": 2037,
            "expense_year": np.array(
                [
                    2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033
                ]
            ),
            "cost": np.array(
                [
                    8404.941309, 7679.705323, 5695.079368, 10093.16847, 12217.94999,
                    399.956189, 588.722239, 482.813158, 1156.510758, 755.344445
                ]
            ),
            "cost_allocation": (
                [
                    FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                    FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                    FluidType.GAS, FluidType.GAS
                ]
            ),
            "cost_type": (
                [
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST
                ]
            ),
            "tax_portion": np.array(
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
        }

        # Store intangible costs as the class's attribute: "intangible"
        self.intangible = {
            "instance": {
                "oil": Intangible(**intangible_oil),
                "gas": Intangible(**intangible_gas),
            },
            "dict": {
                "oil": intangible_oil,
                "gas": intangible_gas,
            },
        }

    def get_opex(self) -> None:
        """
        Create and assign operating expenditure (OPEX) data for oil and gas.

        Builds OPEX data as dictionaries and initializes corresponding
        `OPEX` objects. Both raw data and instances are stored in `self.opex`.

        Returns
        -------
        None
            Updates `self.opex` with OPEX data and instances.

        Attributes
        ----------
        self.opex : dict
            Contains two sections:
            - **"instance"** : dict of {str: OPEX}
                OPEX object instances for "oil" and "gas".
            - **"dict"** : dict of {str: dict}
                Raw OPEX data used to initialize each instance.
        """

        # Prepare opex: OIL
        opex_oil = {
            "start_year": 2023,
            "end_year": 2037,
            "expense_year": np.array(
                [
                    2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035
                ]
            ),
            "fixed_cost": np.array(
                [
                    785.7212242, 4351.968932, 4703.91775, 6325.825984, 6390.700982,
                    6575.827111, 4434.656934, 4938.484606, 8351.939437, 3282.075895,
                    3226.479026, 3468.367892
                ]
            ),
            "cost_allocation": (
                [
                    FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                    FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                    FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL
                ]
            ),
            "cost_type": (
                [
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST
                ]
            ),
            "tax_portion": np.array(
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
        }

        # Prepare opex: GAS
        opex_gas = {
            "start_year": 2023,
            "end_year": 2037,
            "expense_year": np.array(
                [
                    2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035
                ]
            ),
            "fixed_cost": np.array(
                [
                    140.96224, 796.806673, 1212.196461, 2766.452827, 3560.776868,
                    3658.253024, 2667.104051, 2870.44273, 5184.39062, 1554.842585,
                    1582.693399, 1983.719401
                ]
            ),
            "cost_allocation": (
                [
                    FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                    FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                    FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS
                ]
            ),
            "cost_type": (
                [
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST
                ]
            ),
            "tax_portion": np.array(
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
        }

        # Store opex as the class's attributes: "opex"
        self.opex = {
            "instance": {
                "oil": OPEX(**opex_oil),
                "gas": OPEX(**opex_gas),
            },
            "dict": {
                "oil": opex_oil,
                "gas": opex_gas,
            },
        }

    def get_asr(self) -> None:
        """
        Create and assign Abandonment and Site Restoration (ASR) cost data for oil and gas.

        Builds ASR cost data as dictionaries and initializes corresponding
        `ASR` objects. Both raw data and instances are stored in `self.asr`.

        Returns
        -------
        None
            Updates `self.asr` with ASR cost data and instances.

        Attributes
        ----------
        self.asr : dict
            Contains two sections:
            - **"instance"** : dict of {str: ASR}
                ASR object instances for "oil" and "gas".
            - **"dict"** : dict of {str: dict}
                Raw ASR cost data used to initialize each instance.
        """

        # Prepare ASR: OIL
        oil_asr = {
            "start_year": 2023,
            "end_year": 2037,
            "expense_year": np.array(
                [
                    2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035
                ]
            ),
            "cost": np.array(
                [
                    173.28832, 316.4429143, 387.4743104, 688.6020255, 717.0204906,
                    726.1726918, 708.4741216, 779.1434817, 770.3641929, 730.4658999
                ]
            ),
            "cost_allocation": (
                [
                    FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                    FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                    FluidType.OIL, FluidType.OIL
                ]
            ),
            "cost_type": (
                [
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                ]
            ),
            "final_year": np.array(
                [
                    2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035
                ]
            ),
            "tax_portion": np.array(
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
        }

        # Prepare ASR: GAS
        gas_asr = {
            "start_year": 2023,
            "end_year": 2037,
            "expense_year": np.array(
                [
                    2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035
                ]
            ),
            "cost": np.array(
                [
                    44.65628428, 138.3889467, 215.8933058, 383.081915,
                    431.2325133, 422.0803121, 439.7788823, 369.1095222,
                    377.8888109, 417.7871039,
                ]
            ),
            "cost_allocation": (
                [
                    FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                    FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS,
                    FluidType.GAS, FluidType.GAS
                ]
            ),
            "cost_type": (
                [
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                    CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST
                ]
            ),
            "final_year": np.array(
                [
                    2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035
                ]
            ),
            "tax_portion": np.array(
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
        }

        # Store ASR costs as the class's attribute: "asr"
        self.asr = {
            "instance": {
                "oil": ASR(**oil_asr),
                "gas": ASR(**gas_asr),
            },
            "dict": {
                "oil": oil_asr,
                "gas": gas_asr,
            },
        }

    def get_lbt(self) -> None:
        """
        Create and assign Land and Building Tax (LBT) cost data for oil and gas.
        """

        # Prepare LBT cost: OIL
        lbt_oil = None

        # Prepare LBT cost: GAS
        lbt_gas = None

        # Store LBT costs as the class's attribute: "lbt"
        self.lbt = {
            "instance": {
                "oil": None,
                "gas": None,
            },
            "dict": {
                "oil": lbt_oil,
                "gas": lbt_gas,
            },
        }

    def get_cost_of_sales(self) -> None:
        """
        Create and assign cost of sales for oil and gas.
        """

        # Prepare COS: OIL
        cos_oil = None

        # Prepare COS: GAS
        cos_gas = None

        # Store COS as the class's attribute: "cost_of_sales"
        self.cost_of_sales = {
            "instance": {
                "oil": None,
                "gas": None,
            },
            "dict": {
                "oil": cos_oil,
                "gas": cos_gas,
            },
        }

    def get_kwargs(self) -> dict:
        """
        Build and return keyword argument dictionaries for each contract type.

        Constructs predefined keyword arguments for Base Project, Cost Recovery,
        and Gross Split contracts based on stored cost dictionaries and lifting data.

        Returns
        -------
        dict
            Keyword argument dictionary matching the current contract type.

        Raises
        ------
        ValueError
            If the specified contract type is not recognized.

        Notes
        -----
        The returned dictionary includes base parameters (dates, lifting, and costs)
        and contract-specific parameters (e.g., FTP, DMO, investment credit, tax split).
        """

        lft = self.lifting["dict"]
        cap = self.capital["dict"]
        intang = self.intangible["dict"]
        opx = self.opex["dict"]
        asr = self.asr["dict"]
        lbt = self.lbt["dict"]
        cos = self.cost_of_sales["dict"]

        # Base project
        kwargs_base_project = {
            # Base parameters
            "start_date": date(year=2023, month=1, day=1),
            "end_date": date(year=2032, month=12, day=31),
            "oil_onstream_date": date(year=2030, month=1, day=1),
            "gas_onstream_date": date(year=2029, month=1, day=1),
            "approval_year": 2026,
            "is_pod_1": False,

            # Lifting and costs
            "lifting": tuple([lft["oil"], lft["gas"]]),
            "capital_cost": tuple([cap["oil"], cap["gas"]]),
            "intangible_cost": tuple([intang["oil"], intang["gas"]]),
            "opex": tuple([opx["oil"], opx["gas"]]),
            "asr_cost": tuple([asr["oil"], asr["gas"]]),
            "lbt_cost": tuple([lbt["oil"], lbt["gas"]]),
            "cost_of_sales": tuple([cos["oil"], cos["gas"]]),
        }

        # Cost recovery
        kwargs_cost_recovery = {
            # Base parameters
            **kwargs_base_project,

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
            # Base parameters
            **kwargs_base_project,

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

        self.kwargs_contract = {
            ContractType.BASE_PROJECT: kwargs_base_project,
            ContractType.COST_RECOVERY: kwargs_cost_recovery,
            ContractType.GROSS_SPLIT: kwargs_gross_split,
        }

        try:
            self.kwargs_contract[self.contract_type]

        except KeyError:
            raise ValueError(f"Unrecognized contract type: {self.contract_type!r}")

    def get_contract_arguments(self) -> dict:
        """
        Build and return contract argument dictionaries for each contract type.

        Constructs predefined argument sets for Base Project, Cost Recovery,
        and Gross Split contracts, including fiscal, inflation, depreciation,
        and revenue configurations.

        Returns
        -------
        dict
            Argument dictionary corresponding to the current contract type.

        Raises
        ------
        ValueError
            If the specified contract type is not recognized.

        Notes
        -----
        The returned dictionary defines parameters such as tax regime,
        depreciation method, inflation settings, DMO handling, and other revenues.
        """

        # Base project
        args_base_project = {
            "sulfur_revenue": OtherRevenue.REDUCTION_TO_GAS_OPEX,
            "electricity_revenue": OtherRevenue.ADDITION_TO_OIL_REVENUE,
            "co2_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
            "tax_rate": 0.0,
            "year_inflation": None,
            "inflation_rate": 0.0,
            "inflation_rate_applied_to": None,
        }

        # Cost recovery
        args_cost_recovery = {
            "sulfur_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
            "electricity_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
            "co2_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
            "vat_rate": 0.0,
            "year_inflation": None,
            "inflation_rate": 0.0,
            "inflation_rate_applied_to": None,
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
            "sulfur_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
            "electricity_revenue": OtherRevenue.ADDITION_TO_OIL_REVENUE,
            "co2_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
            "vat_rate": 0.0,
            "inflation_rate": 0.0,
            "inflation_rate_applied_to": InflationAppliedTo.CAPEX,
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
            ContractType.BASE_PROJECT: args_base_project,
            ContractType.COST_RECOVERY: args_cost_recovery,
            ContractType.GROSS_SPLIT: args_gross_split,
        }

        try:
            self.contract_arguments = ctr_args[self.contract_type]

        except KeyError:
            raise ValueError(f"Unrecognized contract type: {self.contract_type!r}")

    def get_summary_arguments(self) -> dict:
        """
        Define and assign project summary arguments.

        Sets key economic parameters for NPV and profitability evaluation,
        including discount rate, inflation, and discounting mode.

        Returns
        -------
        dict
            Dictionary containing summary argument values.
        """

        self.summary_arguments = {
            "discount_rate": 0.1,
            "npv_mode": NPVSelection.NPV_SKK_NOMINAL_TERMS,
            "discounting_mode": DiscountingMode.END_YEAR,
            "discount_rate_start_year": 2023,
            "inflation_rate": 0.0,
            "profitability_discounted": False,
        }

    def fit_dict(self):

        t1 = self.kwargs_contract

        print('\t')
        print(f'Filetype: {type(t1)}')
        print(f'Length: {len(t1)}')
        print('t1 = \n', t1)


    def fit_instance(self):
        pass
