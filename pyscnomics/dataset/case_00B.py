"""
A collection of synthetic data to run the economic engine.
Only single fluid is involved.
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
    OptimizationParameter,
    # UncertaintyDistribution,
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
class Case00B:
    """
    A container class for building and managing contract data for a synthetic case.

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
    fluid: FluidType = field(default=None, init=False, repr=False)

    # An attribute representing project timeline
    start_year: int = field(default=2023, init=False, repr=False)
    end_year: int = field(default=2032, init=False, repr=False)
    project_duration: int = field(default=None, init=False, repr=False)
    project_time: dict = field(default_factory=lambda: {}, init=False, repr=False)

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

    # Attributes associated with sensitivity, uncertainty, and optimization arguments
    sensitivity_arguments: dict = field(default_factory=lambda: {}, init=False, repr=False)
    uncertainty_arguments: dict = field(default_factory=lambda: {}, init=False, repr=False)
    optimization_arguments: dict = field(default_factory=lambda: {}, init=False, repr=False)

    def __post_init__(self):
        """
        Initialize all contract components after object creation.

        This method automatically populates key contract attributes such as
        lifting, capital, operating costs, fiscal terms, and summary parameters
        immediately after class instantiation.
        """

        # Specify fluid type
        self.fluid = FluidType.OIL

        # Prepare attributes associated with project timeline
        self.project_duration = self.end_year - self.start_year + 1
        self.project_time = {
            "start_year": self.start_year,
            "end_year": self.end_year,
        }

        # Prepare the required attributes to execute the contract
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
        self.get_sensitivity_arguments()
        self.get_uncertainty_arguments()
        self.get_optimization_arguments()

    def get_lifting(self):
        """
        Initialize and store lifting data for a particular fluid type.
        The resulting data are stored in the class attribute `self.lifting`.
        """

        # Prepare lifting data
        lifting = {
            **self.project_time,
            "prod_year": np.array([2028, 2029, 2030, 2031, 2032]),
            "lifting_rate": np.array([0, 0, 100, 100, 100]),
            "price": np.array([120, 120, 120, 120, 120]),
            "fluid_type": self.fluid,
        }

        # Store lifting data as the class's attribute: "self.lifting"
        self.lifting = {
            self.fluid.value: lifting
        }

    def get_capital(self):
        """
        Initialize and store capital cost data for a particular fluid.

        This method defines yearly capital expenditures for a particular fluid,
        including expense years, cost values, allocation, cost types, and
        useful life, then stores them in the `self.capital` attribute.
        """

        # Prepare capital cost
        capital = {
            **self.project_time,
            "expense_year": np.array(
                [
                    2023,
                    2024,
                    2025,
                    2026,
                    2027,
                    2028,
                    2029,
                    2030,
                    2031,
                    2032,
                ]
            ),
            "cost": np.array(
                [
                    200,  # 2023
                    200,  # 2024
                    200,  # 2025
                    200,  # 2026
                    50,  # 2027
                    50,  # 2028
                    100,  # 2029
                    100,  # 2030
                    100,  # 2031
                    100,  # 2032
                ]
            ),
            "cost_allocation": [self.fluid for _ in range(self.project_duration)],
            "cost_type": (
                [
                    None,  # 2023
                    None,  # 2024
                    None,  # 2025
                    None,  # 2026
                    None,  # 2027
                    None,  # 2028
                    None,  # 2029
                    None,  # 2030
                    None,  # 2031
                    None,  # 2032
                ]
            ),
            "pis_year": np.array(
                [
                    2030,  # 2023
                    2030,  # 2024
                    2030,  # 2025
                    2030,  # 2026
                    2030,  # 2027
                    2030,  # 2028
                    2030,  # 2029
                    2030,  # 2030
                    2030,  # 2031
                    2030,  # 2032
                ]
            ),
            "useful_life": np.array(
                [
                    5,  # 2023
                    5,  # 2024
                    5,  # 2025
                    5,  # 2026
                    5,  # 2027
                    5,  # 2028
                    5,  # 2029
                    5,  # 2030
                    5,  # 2031
                    5,  # 2032
                ]
            ),
        }

        # Store capital cost as the class's attribute: "self.capital"
        self.capital = {
            self.fluid.value: capital
        }

    def get_intangible(self):
        """
        Initialize and store intangible cost data for a particular fluid.

        This method defines yearly intangible expenditures for a particular fluid,
        including expense years, cost values, allocation, and cost types, then
        stores them in the `self.intangible` attribute.
        """

        # Prepare intangible cost
        intangible = {
            **self.project_time,
            "expense_year": np.array(
                [
                    2023,
                    2024,
                    2025,
                    2026,
                    2027,
                    2028,
                    2029,
                    2030,
                    2031,
                    2032,
                ]
            ),
            "cost": np.array(
                [
                    200,
                    200,
                    200,
                    200,
                    50,
                    50,
                    100,
                    100,
                    100,
                    100,
                ]
            ),
            "cost_allocation": [self.fluid for _ in range(self.project_duration)],
            "cost_type": (
                [
                    None,  # 2023
                    None,  # 2024
                    None,  # 2025
                    None,  # 2026
                    None,  # 2027
                    None,  # 2028
                    None,  # 2029
                    None,  # 2030
                    None,  # 2031
                    None,  # 2032
                ]
            ),
        }

        # Store intangible cost as the class's attribute: "self.intangible"
        self.intangible = {
            self.fluid.value: intangible
        }

    def get_opex(self):
        """
        Initialize and store operating expenditure (OPEX) data for a particular
        fluid.

        Defines yearly OPEX information such as expense years, fixed costs,
        allocation, and cost type for a particular fluid type, then stores
        them in the `self.opex` attribute.
        """

        # Prepare OPEX
        opex = {
            **self.project_time,
            "expense_year": np.array(
                [
                    2023,
                    2024,
                    2025,
                    2026,
                    2027,
                    2028,
                    2029,
                    2030,
                    2031,
                    2032,
                ]
            ),
            "fixed_cost": np.array(
                [
                    200,
                    200,
                    200,
                    200,
                    50,
                    50,
                    100,
                    100,
                    100,
                    100,
                ]
            ),
            "cost_allocation": [self.fluid for _ in range(self.project_duration)],
            "cost_type": (
                [
                    None,  # 2023
                    None,  # 2024
                    None,  # 2025
                    None,  # 2026
                    None,  # 2027
                    None,  # 2028
                    None,  # 2029
                    None,  # 2030
                    None,  # 2031
                    None,  # 2032
                ]
            ),
        }

        # Store opex as class's attribute: "self.opex"
        self.opex = {
            self.fluid.value: opex
        }

    def get_asr(self):
        """
        Initialize and store Abandonment and Site Restoration (ASR) cost data
        for a particular fluid.

        Defines yearly ASR information such as expense years, cost values,
        cost allocation, and cost type for a particular fluid type, then
        stores them in the `self.asr` attribute.
        """

        # Prepare ASR
        asr = {
            **self.project_time,
            "expense_year": np.array(
                [
                    2023,
                    2024,
                    2025,
                    2026,
                    2027,
                    2028,
                    2029,
                    2030,
                    2031,
                    2032,
                ]
            ),
            "cost": np.array(
                [
                    200,  # 2023
                    200,  # 2024
                    200,  # 2025
                    200,  # 2026
                    50,  # 2027
                    50,  # 2028
                    100,  # 2029
                    100,  # 2030
                    100,  # 2031
                    100,  # 2032
                ]
            ),
            "cost_allocation": [self.fluid for _ in range(self.project_duration)],
            "cost_type": (
                [
                    None,  # 2023
                    None,  # 2024
                    None,  # 2025
                    None,  # 2026
                    None,  # 2027
                    None,  # 2028
                    None,  # 2029
                    None,  # 2030
                    None,  # 2031
                    None,  # 2032
                ]
            ),
        }

        # Store ASR cost as class's attribute: "self.asr"
        self.asr = {
            self.fluid.value: asr
        }

    def get_lbt(self):
        """
        Initialize and store Land and Building Tax (LBT) cost data for a
        particular fluid.

        Defines yearly LBT information including expense years, cost values,
        cost allocation, and cost type for a particular fluid type, then stores
        them in the `self.lbt` attribute.
        """

        # Prepare LBT
        lbt = {
            **self.project_time,
            "expense_year": np.array(
                [
                    2023,
                    2024,
                    2025,
                    2026,
                    2027,
                    2028,
                    2029,
                    2030,
                    2031,
                    2032,
                ]
            ),
            "cost": np.array(
                [
                    200,  # 2023
                    200,  # 2024
                    200,  # 2025
                    200,  # 2026
                    50,  # 2027
                    50,  # 2028
                    100,  # 2029
                    100,  # 2030
                    100,  # 2031
                    100,  # 2032
                ]
            ),
            "cost_allocation": [self.fluid for _ in range(self.project_duration)],
            "cost_type": (
                [
                    None,  # 2023
                    None,  # 2024
                    None,  # 2025
                    None,  # 2026
                    None,  # 2027
                    None,  # 2028
                    None,  # 2029
                    None,  # 2030
                    None,  # 2031
                    None,  # 2032
                ]
            ),
        }

        # Store LBT cost data in class's attribute "self.lbt"
        self.lbt = {
            self.fluid.value: lbt
        }

    def get_cos(self):
        """
        Initialize and store Cost of Sales (COS) data for a particular fluid.

        Defines yearly COS information including expense years, cost values,
        cost allocation, and cost type for a particular fluid type, then stores
        them in the `self.cos` attribute.
        """

        # Prepare cost of sales
        cos = {
            **self.project_time,
            "expense_year": np.array(
                [
                    2023,
                    2024,
                    2025,
                    2026,
                    2027,
                    2028,
                    2029,
                    2030,
                    2031,
                    2032,
                ]
            ),
            "cost": np.array(
                [
                    200,  # 2023
                    200,  # 2024
                    200,  # 2025
                    200,  # 2026
                    50,  # 2027
                    50,  # 2028
                    100,  # 2029
                    100,  # 2030
                    100,  # 2031
                    100,  # 2032
                ]
            ),
            "cost_allocation": [self.fluid for _ in range(self.project_duration)],
            "cost_type": (
                [
                    None,  # 2023
                    None,  # 2024
                    None,  # 2025
                    None,  # 2026
                    None,  # 2027
                    None,  # 2028
                    None,  # 2029
                    None,  # 2030
                    None,  # 2031
                    None,  # 2032
                ]
            ),
        }

        # Store cost of sales data in class's attribute "self.cos"
        self.cos = {
            self.fluid.value: cos
        }

    def get_setup_arguments(self):
        """
        Define and store general setup arguments for the project.
        """

        self.setup_arguments = {
            "start_date": date(year=2023, month=1, day=1),
            "end_date": date(year=2032, month=12, day=31),
            "oil_onstream_date": date(year=2028, month=1, day=1),
            "gas_onstream_date": None,
            "approval_year": 2026,
            "is_pod_1": True,
            "is_strict": False,
        }

    def get_class_arguments(self):
        """
        Define and store contract-specific class arguments.
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

    def get_contract_arguments(self):
        """
        Define and store contract-level configuration arguments.
        """

        # Base project
        kwargs_base_project = {
            "sulfur_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
            "electricity_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
            "co2_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
            # "vat_rate": 0.0,
            "vat_rate": np.array(
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=float
            ),
            "year_inflation": None,
            "inflation_rate": 0.0,
            "inflation_rate_applied_to": None,
        }

        # Cost recovery
        kwargs_cost_recovery = {
            **kwargs_base_project,
            "is_dmo_end_weighted": False,
            "tax_regime": TaxRegime.NAILED_DOWN,
            "effective_tax_rate": 0.22,
            "ftp_tax_regime": FTPTaxRegime.PDJP_20_2017,
            "depr_method": DeprMethod.PSC_DB,
            "decline_factor": 2,
            "post_uu_22_year2001": True,
            "oil_cost_of_sales_applied": False,
            "gas_cost_of_sales_applied": False,
            "sum_undepreciated_cost": False,
        }

        # Gross split
        kwargs_gross_split = {
            **kwargs_base_project,
            "cum_production_split_offset": 0.0,
            "depr_method": DeprMethod.PSC_DB,
            "decline_factor": 2,
            "sum_undepreciated_cost": False,
            "is_dmo_end_weighted": False,
            "tax_regime": TaxRegime.NAILED_DOWN,
            "effective_tax_rate": 0.22,
            "amortization": False,
            "regime": GrossSplitRegime.PERMEN_ESDM_13_2024,
            "reservoir_type_permen_2024": VariableSplit132024.ReservoirType.MK,
        }

        # Pooled args
        ctr_kwargs = {
            ContractType.COST_RECOVERY: kwargs_cost_recovery,
            ContractType.GROSS_SPLIT: kwargs_gross_split,
            ContractType.BASE_PROJECT: kwargs_base_project,
        }

        try:
            self.contract_arguments = ctr_kwargs[self.contract_type]
        except KeyError:
            raise ValueError(f"Unrecognized contract type: {self.contract_type!r}")

    def get_summary_arguments(self):
        """
        Define and store economic summary configuration parameters.
        """

        self.summary_arguments = {
            "discount_rate": 0.1,
            "npv_mode": NPVSelection.NPV_POINT_FORWARD,
            "discounting_mode": DiscountingMode.END_YEAR,
            "discount_rate_start_year": 2023,
            "inflation_rate": 0.0,
            "profitability_discounted": False,
        }

    def get_sensitivity_arguments(self) -> None:
        """
        Set default sensitivity-analysis parameters.

        Returns
        -------
        None
            Updates ``self.sensitivity_arguments`` with:
            ``min_deviation``, ``max_deviation``, ``base_value``, and ``step``.
        """

        self.sensitivity_arguments = {
            "min_deviation": 0.25,
            "max_deviation": 0.25,
            "base_value": 1,
            "step": 10,
        }

    def get_uncertainty_arguments(self) -> None:
        """
        Set default uncertainty-analysis parameters.

        Returns
        -------
        None
            Populates ``self.uncertainty_arguments`` with defaults for
            price, cost, lifting ranges, standard deviations, distributions,
            and the number of simulation runs.
        """

        self.uncertainty_arguments = {
            "run_number": 100,
            "min_oil_price": None,
            "mean_oil_price": None,
            "max_oil_price": None,
            "min_gas_price": None,
            "mean_gas_price": None,
            "max_gas_price": None,
            "min_opex": None,
            "mean_opex": None,
            "max_opex": None,
            "min_capex": None,
            "mean_capex": None,
            "max_capex": None,
            "min_lifting": None,
            "mean_lifting": None,
            "max_lifting": None,
            "oil_price_stddev": 1.25,
            "gas_price_stddev": 1.25,
            "opex_stddev": 1.25,
            "capex_stddev": 1.25,
            "lifting_stddev": 1.25,
            "oil_price_distribution": "Normal",
            "gas_price_distribution": "Normal",
            "opex_distribution": "Uniform",
            "capex_distribution": "LogNormal",
            "lifting_distribution": "Triangular",
        }

    def get_optimization_arguments(self) -> None:
        """
        Set default optimization parameters.

        Returns
        -------
        None
            Populates ``self.optimization_arguments`` with defaults for the
            optimization target (e.g., IRR), target value, and parameter
            bounds used in the economic optimization routine.
        """

        self.optimization_arguments = {
            "target_parameter": "IRR",
            "target_optimization": 0.30,
            "dict_optimization": {
                "parameter": [
                    "Ministerial Discretion",
                    "VAT Rate",
                ],
                "min": [
                    0.01,
                    0.005,
                ],
                "max": [
                    0.1,
                    0.2,
                ],
            },
            # "parameter": [
            #     OptimizationParameter.MINISTERIAL_DISCRETION,
            #     OptimizationParameter.VAT_RATE,
            # ],
            # "min": [
            #     0.01,
            #     0.005,
            # ],
            # "max": [
            #     0.1,
            #     0.2,
            # ],
        }

    def as_class(self):
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

        # Create lifting and costs-related instances
        instances = {
            "lifting": Lifting(**self.lifting[self.fluid.value]),
            "capital_cost": CapitalCost(**self.capital[self.fluid.value]),
            "intangible_cost": Intangible(**self.intangible[self.fluid.value]),
            "opex": OPEX(**self.opex[self.fluid.value]),
            "asr_cost": ASR(**self.asr[self.fluid.value]),
            "lbt_cost": LBT(**self.lbt[self.fluid.value]),
            "cost_of_sales": CostOfSales(**self.cos[self.fluid.value]),
        }

        # Construct tuples from "instances"
        instances_as_tuple = {
            key: (value,)
            for key, value in instances.items()
        }

        # Merge keyword arguments to create an instance of contract
        kwargs_merged = {
            **self.setup_arguments,   # setup arguments
            **instances_as_tuple,     # lifting and costs arguments
            **self.class_arguments,   # class's arguments
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
        setup_args = _converter(source=self.setup_arguments)
        summary_args = _converter(source=self.summary_arguments)
        contract_args = _converter(source=self.contract_arguments)

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

        # Convert lifting data
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
        mapping_converted_data = [
            ("setup", setup_args),
            ("summary_arguments", summary_args),
            ("contract_arguments", contract_args),
            ("grosssplit", gs),
            ("costrecovery", cr),
            ("lifting", lifting),
            ("capital", cap),
            ("intangible", itg),
            ("opex", op),
            ("asr", asr),
            ("lbt", lbt),
            ("cost_of_sales", cos),
            ("sensitivity_arguments", self.sensitivity_arguments),
            ("uncertainty_arguments", self.uncertainty_arguments),
            ("optimization_arguments", self.optimization_arguments),
        ]

        return {key: val for key, val in mapping_converted_data}


"""
FORMER APPROACH
---------------
    def get_optimization_arguments(self) -> None:
        self.optimization_arguments = {
            "target_parameter": "IRR",
            "target_optimization": 0.35,
            "dict_optimization": {
                "parameter": [
                    "Ministerial Discretion",
                    "Effective Tax Rate"
                ],
                "min": [
                    0.08,
                    0.4
                ],
                "max": [
                    0.1,
                    0.44
                ],
            },
        }
"""
