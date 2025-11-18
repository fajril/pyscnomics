"""
CASE 1
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
        self.get_setup_arguments()
        self.get_class_arguments()
        self.get_contract_arguments()
        self.get_summary_arguments()

    def get_lifting(self) -> None:

        # Prepare lifting data: OIL
        lifting_oil = {
            "start_year": 2023,
            "end_year": 2039,
            "prod_year": np.array(
                [
                    2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031,
                    2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039
                ]
            ),
            "lifting_rate": np.array(
                [
                    18.98, 158.3033, 147.4048, 122.0714, 119.1029, 103.224,
                    90.7603, 78.0871, 82.4073, 70.117, 67.0701, 64.9807,
                    55.7928, 48.9258, 44.0986, 43.1454, 35.1856
                ]
            ),
            "price": np.array(
                [
                    75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
                    75, 75, 75
                ]
            ),
            "fluid_type": FluidType.OIL,
        }

        # Store lifting data as the class's attribute: "self.lifting"
        self.lifting = {"oil": lifting_oil}

    def get_capital(self) -> None:

        # Prepare capital data: OIL
        capital_oil = {
            "start_year": 2023,
            "end_year": 2039,
            "expense_year": np.array([2023, 2024, 2023, 2024, 2025]),
            "cost": np.array([555.3528, 248.4237, 591.4287, 4148.5932, 2197.7461]),
            "cost_allocation": [
                FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL
            ],
            "pis_year": np.array([2023, 2024, 2023, 2024, 2025]),
            "useful_life": np.array([5, 5, 5, 5, 5]),
            "depreciation_factor": np.array([0.25, 0.25, 0.25, 0.25, 0.25]),
            "cost_type": None,
            "tax_portion": None,
            "description": ["DWO", "DWO", "DWO", "PF", "PF"],
        }

        # Store capital costs as the class's attribute: "self.capital"
        self.capital = {"oil": capital_oil}

    def get_intangible(self) -> None:

        # Prepare intangible cost: OIL
        intangible_oil = {
            "start_year": 2023,
            "end_year": 2039,
            "expense_year": np.array([2023, 2024]),
            "cost": np.array([2314.5267, 1033.8725]),
            "cost_allocation": [FluidType.OIL, FluidType.OIL],
            "cost_type": None,
            "tax_portion": None,
            "description": ["DWO", "DWO"]
        }

        # Store intangible cost as the class's attribute: "self.intangible"
        self.intangible = {"oil": intangible_oil}

    def get_opex(self) -> None:

        # Prepare opex: OIL
        opex_oil = {
            "start_year": 2023,
            "end_year": 2039,
            "expense_year": np.array(
                [
                    2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031,
                    2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039
                ]
            ),
            "fixed_cost": np.array(
                [
                    2445.2836, 10178.3519, 7893.3926, 4325.1296, 4628.2841, 4653.1457,
                    4066.6783, 4574.1198, 4321.782, 3896.2821, 4329.2056, 4176.2252,
                    3778.0459, 3578.7603, 3432.025, 3320.0657, 3441.0413
                ]
            ),
            "cost_allocation": [
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL
            ],
            "cost_type": None,
            "tax_portion": None,
        }

        # Store opex as class's attribute: "self.opex"
        self.opex = {"oil": opex_oil}

    def get_asr(self) -> None:

        # Prepare ASR: OIL
        oil_asr = {
            "start_year": 2023,
            "end_year": 2039,
            "expense_year": np.array(
                [
                    2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031,
                    2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039
                ]
            ),
            "cost": np.array(
                [
                    0, 23.8613, 23.8613, 23.8613, 23.8613, 23.8613, 23.8613,
                    23.8613, 23.8613, 23.8613, 23.8613, 23.8613, 23.8613,
                    23.8613, 23.8613, 23.8613, 23.8613
                ]
            ),
            "cost_allocation": [
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL
            ],
            "cost_type": None,
            "tax_portion": None,
        }

        # Store ASR cost as class's attribute: "self.asr"
        self.asr = {"oil": oil_asr}

    def get_setup_arguments(self) -> None:
        self.setup_arguments = {
            "start_date": date(year=2023, month=1, day=1),
            "end_date": date(year=2039, month=12, day=31),
            "oil_onstream_date": date(year=2023, month=1, day=1),
            "gas_onstream_date": None,
            "approval_year": None,
            "is_pod_1": False,
        }

    def get_class_arguments(self) -> None:

        # Cost recovery
        kwargs_cost_recovery = {
            # FTP
            "oil_ftp_is_available": True,
            "oil_ftp_is_shared": True,
            "oil_ftp_portion": 0.1,
            "gas_ftp_is_available": True,
            "gas_ftp_is_shared": True,
            "gas_ftp_portion": 0.1,

            # Tax split
            "tax_split_type": TaxSplitTypeCR.CONVENTIONAL,
            "condition_dict": dict,
            "indicator_rc_icp_sliding": None,
            "oil_ctr_pretax_share": 0.599359,
            "gas_ctr_pretax_share": 0.0,

            # Investment credit
            "oil_ic_rate": 0.0,
            "gas_ic_rate": 0.0,
            "ic_is_available": False,
            "oil_cr_cap_rate": 1.0,
            "gas_cr_cap_rate": 0.0,

            # DMO
            "oil_dmo_volume_portion": 0.25,
            "oil_dmo_fee_portion": 1,
            "oil_dmo_holiday_duration": 0,
            "gas_dmo_volume_portion": 1.0,
            "gas_dmo_fee_portion": 1.0,
            "gas_dmo_holiday_duration": 0,

            # Carry forward depreciation
            "oil_carry_forward_depreciation": 0.0,
            "gas_carry_forward_depreciation": 0.0,
        }

        # Pooled class args
        class_args_map = {
            ContractType.COST_RECOVERY: kwargs_cost_recovery,
            ContractType.GROSS_SPLIT: {},
            ContractType.BASE_PROJECT: {},
        }

        try:
            self.class_arguments = class_args_map[self.contract_type]

        except KeyError:
            raise ValueError(f"Unrecognized contract type: {self.contract_type!r}")

    def get_contract_arguments(self) -> None:

        # Base project
        args_base_project = {
            "sulfur_revenue": OtherRevenue.REDUCTION_TO_OIL_OPEX,
            "electricity_revenue": OtherRevenue.REDUCTION_TO_OIL_OPEX,
            "co2_revenue": OtherRevenue.REDUCTION_TO_GAS_OPEX,
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
            "effective_tax_rate": 0.376,
            "ftp_tax_regime": FTPTaxRegime.DIRECT_MODE,
            "depr_method": DeprMethod.PSC_DB,
            "decline_factor": 2,
            "post_uu_22_year2001": True,
            "oil_cost_of_sales_applied": False,
            "gas_cost_of_sales_applied": False,
            "sum_undepreciated_cost": True,
            "sunk_cost_method": SunkCostMethod.DEPRECIATED_TANGIBLE,
        }

        # Pooled args
        ctr_args = {
            ContractType.COST_RECOVERY: args_cost_recovery,
            ContractType.GROSS_SPLIT: {},
            ContractType.BASE_PROJECT: args_base_project,
        }

        try:
            self.contract_arguments = ctr_args[self.contract_type]

        except KeyError:
            ValueError(f"Unrecognized contract type: {self.contract_type!r}")

    def get_summary_arguments(self):
        self.summary_arguments = {
            "discount_rate": 0.1,
            "npv_mode": NPVSelection.NPV_NOMINAL_TERMS,
            "discounting_mode": DiscountingMode.END_YEAR,
            "discount_rate_start_year": 2023,
            "inflation_rate": 0.0,
            "profitability_discounted": True,
        }

    def as_dict(self) -> dict:

        # Helper function to convert data stored in an argument dictionary
        def _converter(source: dict):
            return {key: convert_object(objects=val) for key, val in source.items()}

        # Convert data in "setup_arguments", "summary_arguments", and "contract_arguments"
        setup = _converter(source=self.setup_arguments)
        summary_arguments = _converter(source=self.summary_arguments)
        contract_arguments = _converter(source=self.contract_arguments)

        print('\t')
        print(f'Filetype: {type(contract_arguments)}')
        print(f'Length: {len(contract_arguments)}')
        print('contract_arguments = \n', contract_arguments)

    def as_class(self) -> CostRecovery | GrossSplit | BaseProject:

        fl = ["oil"]

        # Create per fluid instances for lifting and each cost category
        instances = {
            "lifting": {f: Lifting(**self.lifting[f]) for f in fl},
            "capital": {f: CapitalCost(**self.capital[f]) for f in fl},
            "intangible": {f: Intangible(**self.intangible[f]) for f in fl},
            "opex": {f: OPEX(**self.opex[f]) for f in fl},
            "asr": {f: ASR(**self.asr[f]) for f in fl},
        }

        # Construct tuples from the created instances
        instances_as_tuple = {
            "lifting": tuple(instances["lifting"].values()),
            "capital_cost": tuple(instances["capital"].values()),
            "intangible_cost": tuple(instances["intangible"].values()),
            "opex": tuple(instances["opex"].values()),
            "asr_cost": tuple(instances["asr"].values()),
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
