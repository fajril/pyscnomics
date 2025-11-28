"""
CASE_02
"""

import numpy as np
from datetime import date
from dataclasses import dataclass, field
# from functools import reduce
# from itertools import chain

# from pyscnomics.contracts.project import BaseProject
# from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.econ.selection import (
    FluidType,
    CostType,
    # VariableSplit082017,
    VariableSplit522017,
    VariableSplit132024,
    # TaxSplitTypeCR,
    ContractType,
    OtherRevenue,
    TaxRegime,
    # FTPTaxRegime,
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
    # CostOfSales,
)
from pyscnomics.io.getattr import (
    convert_object,
    construct_lifting_attr,
    construct_cost_attr,
)


@dataclass
class Case02:
    """
    A container class for building and managing contract data for CASE 04.

    This class initializes predefined datasets for lifting, capital, intangible,
    and other economic parameters for CASE 04.

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
        Arrange contract initialization after dataclass construction.

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

        # Only allows GrossSplit as the corresponding contract for CASE 04
        if self.contract_type is not ContractType.GROSS_SPLIT:
            raise ValueError(
                f"Contract type for CASE 04 must be ContractType.GROSS_SPLIT, "
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
        Prepare and store lifting data for the oil stream.

        Populates the ``self.lifting`` attribute with annual production,
        lifting rates, prices, and baseline production rates over the
        contract period.

        Returns
        -------
        None
            The lifting data are stored internally under ``self.lifting["oil"]``.
        """

        # Prepare lifting data: OIL
        lifting_oil = {
            "start_year": 2022,
            "end_year": 2059,
            "prod_year": np.array(
                [
                    2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033,
                    2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043,
                    2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053,
                    2054, 2055, 2056, 2057, 2058, 2059
                ]
            ),
            "lifting_rate": np.array(
                [
                    30.7496489541442, 91.3691250096681, 105.2117737416, 97.139262621418,
                    90.533729362489, 85.5792058078332, 82.1767593439732, 80.0882775790857,
                    79.879371564412, 79.4464749677659, 77.5683213452095, 71.638954251559,
                    65.1294562028971, 58.9095677329961, 53.6991479763489, 49.6650737250467,
                    46.6963360741145, 43.8233608711693, 41.471939632469, 37.0017482431865,
                    34.9082578255658, 35.2733186573718, 33.8063034227161, 31.9699848194289,
                    29.9664106553551, 28.030659603899, 26.4891113187661, 21.994061356818,
                    18.3657109734312, 15.5130066438716, 14.6478842083992, 14.1974383820702,
                    13.8538282850916, 13.434870745507, 13.0705202140086, 12.7837967284562
                ]
            ),
            "price": np.array(
                [
                    60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
                    60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
                    60, 60, 60, 60
                ]
            ),
            "fluid_type": FluidType.OIL,
            "prod_rate_baseline": np.array(
                [
                    0, 0, 0, 0, 0, 30000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    30000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
        }

        # Store lifting data as the class's attribute: "self.lifting"
        self.lifting = {"oil": lifting_oil}

    def get_capital(self) -> None:
        """
        Prepare and store capital cost data for the oil stream.

        Populates the ``self.capital`` attribute with annual expenditure,
        depreciation parameters, cost types, and tax-related attributes
        for all capital items.

        Returns
        -------
        None
            The capital cost data are stored internally under ``self.capital["oil"]``.
        """

        # Prepare capital data: OIL
        capital_oil = {
            "start_year": 2022,
            "end_year": 2059,
            "expense_year": np.array([2024, 2025, 2024, 2025]),
            "cost": np.array([428.144, 798.591, 345.992135389933, 528.104270779866]),
            "cost_allocation": [FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL],
            "pis_year": np.array([2024, 2025, 2024, 2025]),
            "useful_life": np.array([5, 5, 5, 5]),
            "depreciation_factor": np.array([0.25, 0.25, 0.25, 0.25]),
            "cost_type": [
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST
            ],
            "tax_portion": np.array([1.0, 1.0, 1.0, 1.0]),
            "description": ["Tang DWO", "Tang DWO", "PF", "PF"],
        }

        # Store capital costs as the class's attribute: "self.capital"
        self.capital = {"oil": capital_oil}

    def get_intangible(self) -> None:
        """
        Populate the intangible (non-capital) cost components for the OIL stream.

        Notes
        -----
        - Covers intangible post-onstream expenditures only.
        - All values are stored as raw arrays/lists to match downstream
          consumption in the Intangible cost model.
        - The resulting dictionary is assigned to ``self.intangible`` with the key
          ``"oil"``.
        """

        # Prepare intangible cost: OIL
        intangible_oil = {
            "start_year": 2022,
            "end_year": 2059,
            "expense_year": np.array([2024, 2025]),
            "cost": np.array([3176.383, 6605.103]),
            "cost_allocation": [FluidType.OIL, FluidType.OIL],
            "cost_type": [CostType.POST_ONSTREAM_COST, None],
            "tax_portion": np.array([1.0, 1.0]),
            "description": ["Intang DWO", "Intang DWO"],
        }

        # Store intangible cost as the class's attribute: "self.intangible"
        self.intangible = {"oil": intangible_oil}

    def get_opex(self) -> None:
        """
        Populate the operating expenditure (OPEX) components for the OIL stream.

        Notes
        -----
        - Covers routine post-onstream OPEX from 2024–2059.
        - Arrays must remain aligned by index across all fields
          (expense year, fixed cost, allocation, cost type, etc.).
        - The resulting dictionary is stored under ``self.opex["oil"]``.
        """

        # Prepare opex OIL
        opex_oil = {
            "start_year": 2022,
            "end_year": 2059,
            "expense_year": np.array(
                [
                    2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033,
                    2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043,
                    2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053,
                    2054, 2055, 2056, 2057, 2058, 2059
                ]
            ),
            "fixed_cost": np.array(
                [
                    284.471152404579, 1081.03004928944, 973.33516123829, 898.654746363262,
                    837.545637078258, 791.710348769426, 760.233636042965, 740.912673539638,
                    738.980042216689, 734.975229221796, 717.600054428802, 662.746293572023,
                    602.525625224242, 544.984193011494, 496.781557758799, 459.461530045152,
                    431.997144288848, 405.418676091361, 383.665207927897, 342.310573347367,
                    322.943274795874, 326.320525563078, 312.748874224231, 295.760723561501,
                    277.225258254821, 259.31723812759, 245.056066632169, 203.471460424194,
                    169.904865357406, 143.513927063785, 135.510506388743, 131.343341960208,
                    128.164536231039, 124.288676240834, 120.917996603837, 118.265460294294
                ]
            ),
            "cost_allocation": [
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL
            ],
            "cost_type": [
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
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST
            ],
            "tax_portion": np.array(
                [
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
                ]
            ),
            "description": [
                "OPEX Rutin", "OPEX Rutin", "OPEX Rutin", "OPEX Rutin", "OPEX Rutin",
                "OPEX Rutin", "OPEX Rutin", "OPEX Rutin", "OPEX Rutin", "OPEX Rutin",
                "OPEX Rutin", "OPEX Rutin", "OPEX Rutin", "OPEX Rutin", "OPEX Rutin",
                "OPEX Rutin", "OPEX Rutin", "OPEX Rutin", "OPEX Rutin", "OPEX Rutin",
                "OPEX Rutin", "OPEX Rutin", "OPEX Rutin", "OPEX Rutin", "OPEX Rutin",
                "OPEX Rutin", "OPEX Rutin", "OPEX Rutin", "OPEX Rutin", "OPEX Rutin",
                "OPEX Rutin", "OPEX Rutin", "OPEX Rutin", "OPEX Rutin", "OPEX Rutin",
                "OPEX Rutin"
            ],
        }

        # Store opex oil as class's attribute: "self.opex"
        self.opex = {"oil": opex_oil}

    def get_asr(self) -> None:
        """
        Populate the Abandonment and Site Restoration (ASR) cost components for OIL.

        Notes
        -----
        - Loads ASR costs for 2024–2059 with constant annual values.
        - ``final_year`` mirrors ``expense_year`` and indicates the ASR liability year.
        - ASR is treated as non-taxable (``tax_portion = 0``).
        - All fields must remain index-aligned across arrays.
        - Stored under ``self.asr["oil"]``.
        """

        # Prepare ASR: OIL
        asr_oil = {
            "start_year": 2022,
            "end_year": 2059,
            "expense_year": np.array(
                [
                    2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034,
                    2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045,
                    2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056,
                    2057, 2058, 2059
                ]
            ),
            "cost": np.array(
                [
                    28.1677060157542, 28.1677060157542, 28.1677060157542, 28.1677060157542,
                    28.1677060157542, 28.1677060157542, 28.1677060157542, 28.1677060157542,
                    28.1677060157542, 28.1677060157542, 28.1677060157542, 28.1677060157542,
                    28.1677060157542, 28.1677060157542, 28.1677060157542, 28.1677060157542,
                    28.1677060157542, 28.1677060157542, 28.1677060157542, 28.1677060157542,
                    28.1677060157542, 28.1677060157542, 28.1677060157542, 28.1677060157542,
                    28.1677060157542, 28.1677060157542, 28.1677060157542, 28.1677060157542,
                    28.1677060157542, 28.1677060157542, 28.1677060157542, 28.1677060157542,
                    28.1677060157542, 28.1677060157542, 28.1677060157542, 28.1677060157542
                ]
            ),
            "cost_allocation": [
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL
            ],
            "cost_type": [
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
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST
            ],
            "final_year": np.array(
                [
                    2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034,
                    2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045,
                    2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056,
                    2057, 2058, 2059
                ]
            ),
            "tax_portion": np.array(
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            ),
        }

        # Store ASR cost as class's attribute: "self.asr"
        self.asr = {"oil": asr_oil}

    def get_lbt(self) -> None:
        """
        Populate the Land and Building Tax (LBT) cost components for the OIL stream.

        Notes
        -----
        - Covers annual post-onstream LBT costs from 2024–2059.
        - All arrays must remain index-aligned across fields
          (expense year, cost, allocation, cost type, tax portion).
        - LBT is non-taxable in this configuration (``tax_portion = 0``).
        - The resulting dictionary is stored under ``self.lbt["oil"]``.
        """

        # Prepare LBT: OIL
        lbt_oil = {
            "start_year": 2022,
            "end_year": 2059,
            "expense_year": np.array(
                [
                    2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034,
                    2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045,
                    2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056,
                    2057, 2058, 2059
                ]
            ),
            "cost": np.array(
                [
                    37.0471770599529, 110.081521811648, 126.75914500388, 117.033383606284,
                    109.075037135927, 103.105827157277, 99.006559657619, 96.4903568272824,
                    96.2386668608036, 95.7171130411643, 93.4543135567084, 86.3106120822783,
                    78.4679688332504, 70.9742472047137, 64.6967334819052, 59.8364808239363,
                    56.2597457020932, 52.7983851775847, 49.9653928691986, 44.5797062833911,
                    42.0574690282417, 42.4972943184015, 40.7298343636884, 38.5174377104479,
                    36.1035315575718, 33.7713386907775, 31.9140813168494, 26.4984451226943,
                    22.1270085807899, 18.6900704045364, 17.6477708942794, 17.1050737627182,
                    16.6910923178783, 16.1863322741868, 15.7473627538376, 15.4019182984441
                ]
            ),
            "cost_allocation": [
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL
            ],
            "cost_type": [
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
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST
            ],
            "tax_portion": np.array(
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
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
            "end_date": date(year=2059, month=12, day=31),
            "oil_onstream_date": date(year=2024, month=1, day=1),
            "gas_onstream_date": None,
            "approval_year": None,
            "is_pod_1": False,
        }

    def get_class_arguments(self) -> None:
        """
        Initialize Gross Split–specific class-level arguments.

        Populates ``self.class_arguments`` with all contract parameters
        required under the 2017 Variable Split 52 scheme, including field
        characteristics, reservoir properties, ministry discretion, domestic
        market obligation (DMO) settings, and carry-forward depreciation.

        Returns
        -------
        None

        Notes
        -----
        - All parameter values are defined according to the
          ``VariableSplit522017`` regulatory framework.
        - The resulting dictionary is later merged when constructing the
          ``GrossSplit`` contract instance.
        """

        # Gross split regime
        VS_52 = VariableSplit522017
        VS_13 = VariableSplit132024

        kwargs_gross_split = {
            # Field and reservoir properties
            "field_status": VS_52.FieldStatus.NO_POD,
            "field_loc": VS_52.FieldLocation.ONSHORE,
            "res_depth": VS_52.ReservoirDepth.LESSEQUAL_2500,
            "infra_avail": VS_52.InfrastructureAvailability.WELL_DEVELOPED,
            "res_type": VS_52.ReservoirType.CONVENTIONAL,
            "api_oil": VS_52.APIOil.EQUALGREATERTHAN_25,
            "domestic_use": VS_52.DomesticUse.EQUAL_70_UNTIL_LESSTHAN_100,
            "prod_stage": VS_52.ProductionStage.SECONDARY,
            "co2_content": VS_52.CO2Content.LESSTHAN_5,
            "h2s_content": VS_52.H2SContent.LESSTHAN_100,
            "field_reserves_2024": VS_13.FieldReservesAmount.MEDIUM,
            "infra_avail_2024": VS_13.InfrastructureAvailability.PARTIALLY_AVAILABLE,
            "field_loc_2024": VS_13.FieldLocation.ONSHORE,

            # Ministry discretion
            "split_ministry_disc": 0.0,

            # DMO: None,
            "oil_dmo_volume_portion": 0.25,
            "oil_dmo_fee_portion": 1.0,
            "oil_dmo_holiday_duration": 60,
            "gas_dmo_volume_portion": 0.25,
            "gas_dmo_fee_portion": 1.0,
            "gas_dmo_holiday_duration": 60,

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
            "sulfur_revenue": OtherRevenue.ADDITION_TO_OIL_REVENUE,
            "electricity_revenue": OtherRevenue.ADDITION_TO_OIL_REVENUE,
            "co2_revenue": OtherRevenue.REDUCTION_TO_OIL_OPEX,
            "vat_rate": np.array(
                [
                    0.11, 0.11, 0.11, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12,
                    0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12,
                    0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12,
                    0.12, 0.12, 0.12, 0.12, 0.12
                ]
            ),
            "year_inflation": None,
            "inflation_rate": 0.0,
            "inflation_rate_applied_to": "None",
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
            "amortization": True,
            "sunk_cost_method": SunkCostMethod.DEPRECIATED_TANGIBLE,
            "regime": GrossSplitRegime.PERMEN_ESDM_52_2017,
            "reservoir_type_permen_2024": VariableSplit132024.ReservoirType.MK,
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
            "inflation_rate": 0.0,
            "profitability_discounted": True,
        }

    def as_dict(self) -> dict:
        """
        Serialize the full contract configuration into a JSON-ready dictionary.

        Converts all argument groups—setup, summary, contract, class-specific
        parameters, lifting data, and each cost category—into serializable
        dictionary structures. Internally instantiates the appropriate domain
        classes (e.g., ``Lifting``, ``CapitalCost``, ``Intangible``) before
        constructing their JSON-compatible representations.

        Notes
        -----
        - Class-specific arguments are included only when the contract type is
          ``GROSS_SPLIT``.
        - Lifting and all cost components are rebuilt as tuples of domain objects
          before being passed to helper constructors.
        - All nested objects are converted using ``convert_object`` to ensure
          JSON serialization compatibility.

        Returns
        -------
        dict
            A dictionary mapping all contract components (setup, summary,
            contract arguments, class arguments, lifting, and cost structures)
            to their JSON-ready representations.
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
        Assemble and return a ``GrossSplit`` contract.

        Initializes the lifting and cost components for oil, converts them into
        tuple-based inputs, merges all configuration arguments, and constructs
        the ``GrossSplit`` instance.
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
