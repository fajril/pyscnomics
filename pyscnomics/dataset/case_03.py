"""
CASE 03
"""

import numpy as np
from datetime import date
from dataclasses import dataclass, field
from functools import reduce
# from itertools import chain

# from pyscnomics.contracts.project import BaseProject
# from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.econ.selection import (
    FluidType,
    CostType,
    VariableSplit082017,
    # VariableSplit522017,
    # VariableSplit132024,
    # TaxSplitTypeCR,
    ContractType,
    OtherRevenue,
    TaxRegime,
    # FTPTaxRegime,
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
    # CostOfSales,
)
from pyscnomics.io.getattr import (
    convert_object,
    construct_lifting_attr,
    construct_cost_attr,
)


@dataclass
class Case03:
    """
    A container class for building and managing contract data for CASE 03.

    This class initializes predefined datasets for lifting, capital, intangible,
    and other economic parameters for CASE 03.

    Parameters
    ----------
    contract_type : ContractType
        The type of contract to initialize, default to `ContractType.GROSS_SPLIT`
    """

    contract_type: ContractType = field(default=ContractType.GROSS_SPLIT)

    # Attributes associated with lifting
    lifting: dict = field(default_factory=lambda: {}, init=False, repr=False)

    # Attributes associated with costs
    capital: dict = field(default_factory=lambda: {}, init=False, repr=False)
    intangible: dict = field(default_factory=lambda: {}, init=False, repr=False)
    opex: dict = field(default_factory=lambda: {}, init=False, repr=False)
    asr: dict = field(default_factory=lambda: {}, init=False, repr=False)
    lbt: dict = field(default_factory=lambda: {}, init=False, repr=False)

    # Attributes associated with arguments
    setup_arguments: dict = field(default_factory=lambda: {}, init=False, repr=False)
    class_arguments: dict = field(default_factory=lambda: {}, init=False, repr=False)
    contract_arguments: dict = field(default_factory=lambda: {}, init=False, repr=False)
    summary_arguments: dict = field(default_factory=lambda: {}, init=False, repr=False)

    def __post_init__(self):
        """
        Finalize contract initialization after dataclass construction.

        This method validates that the selected contract type is appropriate for
        CASE 03 (i.e., ``ContractType.GROSS_SPLIT``), and initializes all core
        contract components immediately after object creation.

        It automatically computes lifting, expenditures, fiscal terms, and contract
        metadata so no additional setup calls are required.

        Notes
        -----
        - Raises a ``ValueError`` if ``contract_type`` is not ``ContractType.GROSS_SPLIT``.
        - Ensures consistent initialization of all cost components and parameters.
        - Executed automatically after dataclass instantiation.
        """

        # Only allows GrossSplit as the corresponding contract for CASE 03
        if self.contract_type is not ContractType.GROSS_SPLIT:
            raise ValueError(
                f"Contract type for CASE 03 must be ContractType.GROSS_SPLIT, "
                f"not {self.contract_type}"
            )

        # Initializes attributes
        self.get_lifting()
        self.get_capital()
        self.get_intangible()
        self.get_opex()
        self.get_asr()
        self.get_lbt()
        self.get_setup_arguments()
        self.get_class_arguments()
        self.get_contract_arguments()
        self.get_summary_arguments()

    def get_lifting(self) -> None:
        """
        Initialize the lifting profile for oil production.

        Defines the annual oil lifting schedule—including production years, lifting
        rates, prices, and fluid type—and assigns the resulting structure to
        ``self.lifting``. The data span 2022–2041 and represent the base lifting
        assumptions used in contract and economic calculations.

        Returns
        -------
        None
            Updates ``self.lifting`` in place.
        """

        # Prepare lifting data: OIL
        lifting_oil = {
            "start_year": 2022,
            "end_year": 2041,
            "prod_year": np.array(
                [
                    2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031,
                    2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041
                ]
            ),
            "lifting_rate": np.array(
                [
                    0.0, 0.0, 1109.76, 2899.99, 3936.48, 3565.12, 2466.51, 1917.88,
                    1583.69, 1357.06, 1167.95, 1019.10, 901.50, 815.53, 735.83,
                    657.22, 615.44, 585.86, 557.64, 328.02
                ]
            ),
            "price": np.array(
                [
                    69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69,
                    69, 69, 69, 69, 69
                ]
            ),
            "fluid_type": FluidType.OIL,
        }

        # Store lifting data as the class's attribute: "self.lifting"
        self.lifting = {"oil": lifting_oil}

    def get_capital(self) -> None:
        """
        Initialize capital expenditure data for oil.

        Constructs the capital cost profile—including expense years, cost amounts,
        cost allocation, PIS years, useful life, depreciation factors, tax portions,
        and descriptive labels—and assigns it to ``self.capital``. The data span
        2022–2041 and represent post-onstream capital expenditures used in fiscal
        and depreciation calculations.

        Returns
        -------
        None
            Updates ``self.capital`` in place.
        """

        # Prepare capital data: OIL
        capital_oil = {
            "start_year": 2022,
            "end_year": 2041,
            "expense_year": np.array([2023, 2024, 2025, 2026, 2023, 2024, 2025, 2026]),
            "cost": np.array(
                [
                    2804.42, 22850, 27109.48, 15063.18, 2935.44, 41706.1, 51200.25, 22869.6
                ]
            ),
            "cost_allocation": [
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL
            ],
            "pis_year": np.array([2024, 2024, 2025, 2026, 2024, 2024, 2025, 2026]),
            "useful_life": np.array([5, 5, 5, 5, 5, 5, 5, 5]),
            "depreciation_factor": np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]),
            "cost_type": [
                CostType.SUNK_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.SUNK_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST
            ],
            "tax_portion": np.array([1, 1, 1, 1, 1, 1, 1, 1]),
            "description": [
                "Tang DWO", "Tang DWO", "Tang DWO", "Tang DWO", "PF", "PF", "PF", "PF"
            ],
        }

        # Store capital costs as the class's attribute: "self.capital"
        self.capital = {"oil": capital_oil}

    def get_intangible(self) -> None:
        """
        Initialize intangible expenditure data for oil.

        Defines the intangible cost profile—including expense years, cost amounts,
        cost allocation, cost type, tax portions, and descriptions—and assigns it to
        ``self.intangible``. These values represent sunk and post-onstream
        intangible expenditures used in fiscal and cost-recovery calculations.

        Returns
        -------
        None
            Updates ``self.intangible`` in place.
        """

        # Prepare intangible cost: OIL
        intangible_oil = {
            "start_year": 2022,
            "end_year": 2041,
            "expense_year": np.array([2023, 2024, 2025, 2026]),
            "cost": np.array([12827.5575, 116290.7349, 186723.1887, 102764.0144]),
            "cost_allocation": [
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL
            ],
            "cost_type": [
                CostType.SUNK_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST
            ],
            "tax_portion": np.array([1, 1, 1, 1]),
            "description": ["Intang DWO", "Intang DWO", "Intang DWO", "Intang DWO"],
        }

        # Store intangible cost as the class's attribute: "self.intangible"
        self.intangible = {"oil": intangible_oil}

    def get_opex(self) -> None:
        """
        Construct and store the operating expenditure (OPEX) dataset for oil.

        Builds individual OPEX components—WIWS, operation and maintenance,
        electricity, and carbon tax—each defined by yearly cost attributes.

        These components are then combined into a single consolidated OPEX
        dictionary using `_combine`, which concatenates NumPy array fields
        and flattens list fields.

        The final merged OPEX structure is stored under ``self.opex["oil"]``.

        Notes
        -----
        - Numerical fields (``expense_year``, ``fixed_cost``, ``tax_portion``)
          are concatenated as NumPy arrays.
        - Categorical fields (``cost_allocation``, ``cost_type``, ``description``)
          are flattened into Python lists.
        """

        # OPEX WIWS
        wiws = {
            "expense_year": np.array(
                [
                    2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031,
                    2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041,
                ]
            ),
            "fixed_cost": np.array(
                [
                    0, 0, 310.7241078, 1024.549761, 1654.395925, 1797.161056,
                    1797.161056, 1797.161056, 1797.161056, 1797.161056, 1780.365158,
                    1713.181567, 1654.395925, 1612.406181, 1562.018488, 1503.232846,
                    1436.049255, 1427.651306, 1427.651306, 1410.855408
                ]
            ),
            "cost_allocation": [
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL
            ],
            "cost_type": [
                CostType.SUNK_COST, CostType.SUNK_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST
            ],
            "tax_portion": np.array(
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ),
            "description": [
                "OPEX WIWS", "OPEX WIWS", "OPEX WIWS", "OPEX WIWS", "OPEX WIWS",
                "OPEX WIWS", "OPEX WIWS", "OPEX WIWS", "OPEX WIWS", "OPEX WIWS",
                "OPEX WIWS", "OPEX WIWS", "OPEX WIWS", "OPEX WIWS", "OPEX WIWS",
                "OPEX WIWS", "OPEX WIWS", "OPEX WIWS", "OPEX WIWS", "OPEX WIWS"
            ],
        }

        # OPEX operation
        operation = {
            "expense_year": np.array(
                [
                    2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031,
                    2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041
                ]
            ),
            "fixed_cost": np.array(
                [
                    0, 0, 93.99623328, 295.6253904, 467.722378, 508.4673573, 509.3369801,
                    508.402422, 508.4347049, 508.2551227, 500.4444166, 478.1836374,
                    458.8633128, 446.2216302, 430.2434006, 410.0402278, 390.9620195,
                    388.6757859, 389.3266767, 380.6260563
                ]
            ),
            "cost_allocation": [
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL
            ],
            "cost_type": [
                CostType.SUNK_COST, CostType.SUNK_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST
            ],
            "tax_portion": np.array(
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ),
            "description": [
                "OPEX Operation and Maintenance", "OPEX Operation and Maintenance",
                "OPEX Operation and Maintenance", "OPEX Operation and Maintenance",
                "OPEX Operation and Maintenance", "OPEX Operation and Maintenance",
                "OPEX Operation and Maintenance", "OPEX Operation and Maintenance",
                "OPEX Operation and Maintenance", "OPEX Operation and Maintenance",
                "OPEX Operation and Maintenance", "OPEX Operation and Maintenance",
                "OPEX Operation and Maintenance", "OPEX Operation and Maintenance",
                "OPEX Operation and Maintenance", "OPEX Operation and Maintenance",
                "OPEX Operation and Maintenance", "OPEX Operation and Maintenance",
                "OPEX Operation and Maintenance", "OPEX Operation and Maintenance"
            ],
        }

        # OPEX electricity
        electricity = {
            "expense_year": np.array(
                [
                    2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031,
                    2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041,
                ]
            ),
            "fixed_cost": np.array(
                [
                    0, 0, 1203.698502, 2520.801891, 3063.054019, 3367.41954,
                    3375.643238, 3360.633473, 3364.007193, 3345.239959, 2912.238201,
                    2528.318526, 2137.237408, 1978.967933, 1632.74114, 1221.239368,
                    1088.030519, 1081.677592, 1084.037437, 705.5804279
                ]
            ),
            "cost_allocation": [
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL
            ],
            "cost_type": [
                CostType.SUNK_COST, CostType.SUNK_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST
            ],
            "tax_portion": np.array(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ),
            "description": [
                "OPEX Electricity", "OPEX Electricity", "OPEX Electricity", "OPEX Electricity",
                "OPEX Electricity", "OPEX Electricity", "OPEX Electricity", "OPEX Electricity",
                "OPEX Electricity", "OPEX Electricity", "OPEX Electricity", "OPEX Electricity",
                "OPEX Electricity", "OPEX Electricity", "OPEX Electricity", "OPEX Electricity",
                "OPEX Electricity", "OPEX Electricity", "OPEX Electricity", "OPEX Electricity"
            ],
        }

        # OPEX carbon tax
        cartax = {
            "expense_year": np.array(
                [
                    2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031,
                    2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041,
                ]
            ),
            "fixed_cost": np.array(
                [
                    0, 0, 0, 86.34313551, 117.203278, 106.1464782, 73.4369768,
                    57.10211745, 47.1521686, 40.40472871, 34.77400465, 30.34241466,
                    26.84090537, 24.28143326, 21.90835802, 19.56788226, 18.3238454,
                    17.44323657, 16.603004, 9.766410286
                ]
            ),
            "cost_allocation": [
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL
            ],
            "cost_type": [
                CostType.SUNK_COST, CostType.SUNK_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST
            ],
            "tax_portion": np.array(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ),
            "description": [
                "OPEX Carbon Tax", "OPEX Carbon Tax", "OPEX Carbon Tax", "OPEX Carbon Tax",
                "OPEX Carbon Tax", "OPEX Carbon Tax", "OPEX Carbon Tax", "OPEX Carbon Tax",
                "OPEX Carbon Tax", "OPEX Carbon Tax", "OPEX Carbon Tax", "OPEX Carbon Tax",
                "OPEX Carbon Tax", "OPEX Carbon Tax", "OPEX Carbon Tax", "OPEX Carbon Tax",
                "OPEX Carbon Tax", "OPEX Carbon Tax", "OPEX Carbon Tax", "OPEX Carbon Tax"
            ],
        }

        # Prepare opex OIL: Combine WIWS + Operation + Electricity + Carbon Tax
        attrs_numpy: set = {"expense_year", "fixed_cost", "tax_portion"}
        sources = [wiws, operation, electricity, cartax]

        def _combine(target: str):
            """
            Merge values for a specified key across multiple source dictionaries.

            Parameters
            ----------
            target : str
                Key to extract and merge from each source.

            Returns
            -------
            numpy.ndarray or list
                Concatenated NumPy array if ``target`` is in ``attrs_numpy``;
                otherwise a single list formed by sequentially adding list items.

            Notes
            -----
            - All sources are assumed to contain ``target``.
            - List merging uses ``reduce`` with element-wise concatenation.
            """
            items = [src[target] for src in sources]
            if target in attrs_numpy:
                return np.concatenate(items)
            return reduce(lambda a, b: a + b, items)

        opex_oil = {
            "start_year": 2022,
            "end_year": 2041,
            "expense_year": _combine(target="expense_year"),
            "fixed_cost": _combine(target="fixed_cost"),
            "cost_allocation": _combine(target="cost_allocation"),
            "cost_type": _combine(target="cost_type"),
            "tax_portion": _combine(target="tax_portion"),
            "description": _combine(target="description"),
        }

        # Store opex as class's attribute: "self.opex"
        self.opex = {"oil": opex_oil}

    def get_asr(self) -> None:
        """
        Construct and store the abandonment, site restoration (ASR) cost
        dataset for oil.

        Defines yearly ASR cost attributes—including expense year, cost,
        allocation, cost type, and tax portion—and stores the resulting
        structure under ``self.asr["oil"]``.

        Notes
        -----
        - Numerical fields are stored as NumPy arrays.
        - Categorical fields are stored as Python lists.
        - Covers the full project period from 2022 to 2041.
        """

        # Prepare ASR: OIL
        asr_oil = {
            "start_year": 2022,
            "end_year": 2041,
            "expense_year": np.array(
                [
                    2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031,
                    2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041
                ]
            ),
            "cost": np.array(
                [
                    0, 0, 385.937354, 1324.703891, 2204.797519, 2417.584601, 2417.584601,
                    2417.584601, 2417.584601, 2417.584601, 2417.584601, 2417.584601,
                    2417.584601, 2417.584601, 2417.584601, 2417.584601, 2417.584601,
                    2417.584601, 2417.584601, 2417.584601
                ]
            ),
            "cost_allocation": [
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL
            ],
            "cost_type": [
                CostType.SUNK_COST, CostType.SUNK_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST
            ],
            "tax_portion": np.array(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ),
        }

        # Store ASR cost as class's attribute: "self.asr"
        self.asr = {"oil": asr_oil}

    def get_lbt(self) -> None:
        """
        Construct and store the land and building tax (LBT) dataset for oil.

        Defines yearly LBT attributes—including expense year, cost,
        allocation, cost type, and tax portion—for the 2022–2041 period
        and stores the resulting structure under ``self.lbt["oil"]``.

        Notes
        -----
        - Numerical fields are stored as NumPy arrays.
        - Categorical fields are stored as Python lists.
        - All entries are assigned to the oil stream.
        """

        # Prepare LBT: OIL
        lbt_oil = {
            "start_year": 2022,
            "end_year": 2041,
            "expense_year": np.array(
                [
                    2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031,
                    2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041
                ]
            ),
            "cost": np.array(
                [
                    0, 0, 0, 1537.599994, 4017.990147, 5454.071285, 4939.541527,
                    3417.400207, 2657.255194, 2194.232903, 1880.239823, 1618.213274,
                    1411.988601, 1249.045366, 1129.939966, 1019.508571, 910.5941967,
                    852.7027638, 811.7235058, 772.6231632
                ]
            ),
            "cost_allocation": [
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL
            ],
            "cost_type": [
                CostType.SUNK_COST, CostType.SUNK_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST
            ],
            "tax_portion": np.array(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ),
        }

        # Store LBT cost data as class's attribute "self.lbt"
        self.lbt = {"oil": lbt_oil}

    def get_setup_arguments(self) -> None:
        """
        Define and store general setup arguments for the project.

        Initializes key project timeline parameters such as start and end dates,
        onstream dates for oil and gas, approval year, and POD-1 status, then
        stores them in the `self.setup_arguments` attribute.
        """

        self.setup_arguments = {
            "start_date": date(year=2022, month=1, day=1),
            "end_date": date(year=2041, month=12, day=31),
            "oil_onstream_date": date(year=2024, month=1, day=1),
            "gas_onstream_date": None,
            "approval_year": None,
            "is_pod_1": False,
        }

    def get_class_arguments(self) -> None:
        """
        Set class-level parameters for the Gross Split PSC.

        Builds the dictionary of VS_08 (Variable Split 08/2017) inputs—including
        reservoir descriptors, operational conditions, ministry discretion, DMO
        terms, and depreciation settings—and assigns it to
        ``self.class_arguments``.

        Returns
        -------
        None
            Updates ``self.class_arguments`` in place.
        """

        # Gross split regime
        VS_08 = VariableSplit082017

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

            # Ministry discretion
            "split_ministry_disc": 0.08,

            # DMO
            "oil_dmo_volume_portion": 0.25,
            "oil_dmo_fee_portion": 1.0,
            "oil_dmo_holiday_duration": 60,

            # Carry forward depreciation
            "oil_carry_forward_depreciation": 0.0,
            "gas_carry_forward_depreciation": 0.0,
        }

        # Assign kwargs_gross_split as attribute "self.class_arguments"
        self.class_arguments = kwargs_gross_split

    def get_contract_arguments(self) -> None:
        """
        Build and assign contract-level arguments for Gross Split PSC evaluation.

        This method constructs the full set of fiscal and economic parameters used
        during contract execution. It starts with base-project settings (e.g.,
        revenue classification, VAT, inflation) and extends them with Gross
        Split–specific terms such as tax regime, depreciation method,
        amortization rules, and cumulative production split adjustments. The final
        dictionary is stored in ``self.contract_arguments``.

        Returns
        -------
        None
            Updates ``self.contract_arguments`` in place.
        """

        # Base project
        args_base_project = {
            "sulfur_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
            "electricity_revenue": OtherRevenue.ADDITION_TO_OIL_REVENUE,
            "co2_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
            "vat_rate": np.array(
                [
                    0.11, 0.11, 0.11, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12,
                    0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12
                ]
            ),
            "year_inflation": None,
            "inflation_rate": np.array(
                [
                    0.0, 0., 0.0, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
                    0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02
                ]
            ),
            "inflation_rate_applied_to": InflationAppliedTo.CAPEX,
        }

        # Gross split
        args_gross_split = {
            **args_base_project,
            "cum_production_split_offset": 0.0,
            "depr_method": DeprMethod.PSC_DB,
            "decline_factor": 2,
            "sum_undepreciated_cost": True,
            "is_dmo_end_weighted": True,
            "tax_regime": TaxRegime.NAILED_DOWN,
            "effective_tax_rate": 0.22,
            "amortization": False,
            "sunk_cost_method": SunkCostMethod.DEPRECIATED_TANGIBLE,
            "regime": GrossSplitRegime.PERMEN_ESDM_8_2017,
            "initial_amortization_year": InitialYearAmortizationIncurred.ONSTREAM_YEAR,
        }

        # Assign "args_gross_split" as the class's attribute "self.contract_arguments"
        self.contract_arguments = args_gross_split

    def get_summary_arguments(self) -> None:
        """
        Set summary-level economic parameters for project evaluation.

        Initializes the dictionary of high-level financial settings—such as the
        discount rate, NPV mode, discounting convention, inflation profile, and
        profitability options—and assigns it to ``self.summary_arguments`` for use
        in summary and valuation calculations.

        Returns
        -------
        None
            Updates ``self.summary_arguments`` in place.
        """

        self.summary_arguments = {
            "discount_rate": 0.1,
            "npv_mode": NPVSelection.NPV_NOMINAL_TERMS,
            "discounting_mode": DiscountingMode.END_YEAR,
            "discount_rate_start_year": 2023,
            "inflation_rate": np.array(
                [
                    0.0, 0., 0.0, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
                    0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02
                ]
            ),
            "profitability_discounted": False,
        }

    def as_dict(self) -> dict:
        """
        Convert all contract data into a JSON-ready dictionary.

        This method transforms internal contract attributes—such as setup data,
        summary parameters, contract arguments, lifting profiles, and cost
        components—into a fully serializable dictionary representation. Objects are
        converted into primitive types using helper converters so the output can be
        safely exported (e.g., to JSON).

        Returns
        -------
        dict
            A dictionary containing JSON-ready representations of all contract
            components. The structure includes:

            - ``setup`` : dict
              Converted setup parameters.

            - ``summary_arguments`` : dict
              Summary-level arguments prepared for serialization.

            - ``contract_arguments`` : dict
              Contract-specific arguments.

            - ``grosssplit`` : dict or None
              Converted class-level arguments for Gross Split contracts; ``None``
              for all other contract types.

            - ``lifting`` : dict
              Converted lifting attributes.

            - ``capital`` : dict
              Capital cost data.

            - ``intangible`` : dict
              Intangible expenditure data.

            - ``opex`` : dict
              Operating expenditure data.

            - ``asr`` : dict
              Abandonment, site restoration, and related cost data.

            - ``lbt`` : dict
              Land and building tax cost data.
        """

        # Helper function to convert data stored in an argument dictionary
        def _converter(source: dict):
            return {key: convert_object(objects=val) for key, val in source.items()}

        # Convert data in "setup_arguments", "summary_arguments", and "contract_arguments"
        setup: dict = _converter(source=self.setup_arguments)
        summary_arguments: dict = _converter(source=self.summary_arguments)
        contract_arguments: dict = _converter(source=self.contract_arguments)

        # Convert data in "class_arguments"
        gs: dict = (
            _converter(source=self.class_arguments)
            if self.contract_type == ContractType.GROSS_SPLIT
            else None
        )

        # Convert data in "lifting"
        lifting: dict = construct_lifting_attr(
            lifting=tuple([Lifting(**lft) for lft in self.lifting.values()])
        )

        # Convert data in "capital", "intangible", "opex", and "asr"
        # Helper method to convert data associated with costs
        def _construct_cost_attributes(source: dict, Cls):
            items = tuple([Cls(**val) for val in source.values()])
            return construct_cost_attr(cost=items)

        cap: dict = _construct_cost_attributes(source=self.capital, Cls=CapitalCost)
        intang: dict = _construct_cost_attributes(source=self.intangible, Cls=Intangible)
        op: dict = _construct_cost_attributes(source=self.opex, Cls=OPEX)
        asr: dict = _construct_cost_attributes(source=self.asr, Cls=ASR)
        lbt: dict = _construct_cost_attributes(source=self.lbt, Cls=LBT)

        # Mapping converted data
        mapping: tuple = (
            ("setup", setup),
            ("summary_arguments", summary_arguments),
            ("contract_arguments", contract_arguments),
            ("grosssplit", gs),
            ("lifting", lifting),
            ("capital", cap),
            ("intangible", intang),
            ("opex", op),
            ("asr", asr),
            ("lbt", lbt),
        )

        # Return dictionary of json-ready data
        return {key: val for key, val in mapping}

    def as_class(self) -> GrossSplit:
        """
        Build and return a ``GrossSplit`` contract instance for the oil stream.

        Creates per-fluid component objects (lifting, capital, intangible,
        OPEX, ASR, and LBT), converts them into tuple-based arguments, and
        merges them with setup and class-level parameters to instantiate
        a ``GrossSplit`` contract.

        Returns
        -------
        GrossSplit
            A fully constructed GrossSplit contract populated with all
            lifting and cost components for the oil fluid.

        Notes
        -----
        - Only the ``"oil"`` fluid is processed.
        - Each cost category dictionary is expanded into keyword arguments
          when constructing component objects.
        - All merged arguments are passed directly to ``GrossSplit``.
        """

        fl: list = ["oil"]

        # Create per fluid instances for lifting and each cost category
        instances = {
            "lifting": {f: Lifting(**self.lifting[f]) for f in fl},
            "capital": {f: CapitalCost(**self.capital[f]) for f in fl},
            "intangible": {f: Intangible(**self.intangible[f]) for f in fl},
            "opex": {f: OPEX(**self.opex[f]) for f in fl},
            "asr": {f: ASR(**self.asr[f]) for f in fl},
            "lbt": {f: LBT(**self.lbt[f]) for f in fl},
        }

        # Construct tuples from the created instances
        instances_modified = {
            "lifting": tuple(instances["lifting"].values()),
            "capital_cost": tuple(instances["capital"].values()),
            "intangible_cost": tuple(instances["intangible"].values()),
            "opex": tuple(instances["opex"].values()),
            "asr_cost": tuple(instances["asr"].values()),
            "lbt_cost": tuple(instances["lbt"].values()),
        }

        # Merge keyword arguments to create an instance of contract
        kwargs_merged = {
            **self.setup_arguments,     # setup arguments
            **instances_modified,       # lifting and costs arguments
            **self.class_arguments,     # class's arguments
        }

        # Return an instance of GrossSplit
        return GrossSplit(**kwargs_merged)
