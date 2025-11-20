"""
CASE 2
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
class Case2:
    """
    A container class for building and managing contract data for CASE 2.

    This class initializes predefined datasets for lifting, capital, intangible,
    and other economic parameters for CASE 2.

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
        CASE 2 (i.e., ``ContractType.GROSS_SPLIT``), and initializes all core
        contract components immediately after object creation.

        It automatically computes lifting, expenditures, fiscal terms, and contract
        metadata so no additional setup calls are required.

        Notes
        -----
        - Raises a ``ValueError`` if ``contract_type`` is not ``ContractType.GROSS_SPLIT``.
        - Ensures consistent initialization of all cost components and parameters.
        - Executed automatically after dataclass instantiation.
        """

        # Only allows GrossSplit as the corresponding contract for CASE 2
        if self.contract_type is not ContractType.GROSS_SPLIT:
            raise ValueError(
                f"Contract type for CASE 2 must be ContractType.GROSS_SPLIT, "
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
        Prepare and store oil lifting data.

        Assembles yearly oil lifting information (production years, rates, and
        prices) and stores it in the ``self.lifting`` attribute under the key
        ``"oil"``. This method does not return a value.
        """

        # Prepare lifting data: OIL
        lifting_oil = {
            "start_year": 2022,
            "end_year": 2035,
            "prod_year": np.array(
                [2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035]
            ),
            "lifting_rate": np.array(
                [
                    1003.01994465298, 1752.43214595637, 1660.63017314197, 1646.06277634477,
                    1303.30419844135, 1206.56398735216, 715.464597247899, 575.351449953491,
                    385.408188212388, 177.896153424466, 95.5608132946032, 53.0868811432885
                ]
            ),
            "price": np.array([75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75]),
            "fluid_type": FluidType.OIL,
            "prod_rate_baseline": np.array(
                [
                    351_000, 351_000, 351_000, 351_000, 351_000, 351_000, 351_000,
                    351_000, 351_000, 351_000, 351_000, 351_000
                ]
            ),
        }

        # Store lifting data as the class's attribute: "self.lifting"
        self.lifting = {"oil": lifting_oil}

    def get_capital(self) -> None:
        """
        Prepare and store oil capital cost data.

        Assembles predefined capital expenditure information (years, costs,
        depreciation attributes, and metadata) and stores it in the
        ``self.capital`` attribute under the key ``"oil"``.

        Notes
        -----
        - Covers expense years 2024–2028 with a project span of 2022–2035.
        - All costs are allocated to ``FluidType.OIL`` and marked as
          ``CostType.POST_ONSTREAM_COST``.
        - Depreciation factors assume a 5-year useful life at 25% per year.
        """

        # Prepare capital data: OIL
        capital_oil = {
            "start_year": 2022,
            "end_year": 2035,
            "expense_year": np.array([2024, 2025, 2026, 2027, 2028]),
            "cost": np.array(
                [
                    46770.5618086779, 39367.6743830341, 14092.6047468454,
                    146.200209593186, 205.10626669388
                ]
            ),
            "cost_allocation": [
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL
            ],
            "pis_year": np.array([2024, 2025, 2026, 2027, 2028]),
            "useful_life": np.array([5, 5, 5, 5, 5]),
            "depreciation_factor": np.array([0.25, 0.25, 0.25, 0.25, 0.25]),
            "cost_type": [
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST
            ],
            "tax_portion": np.array([0, 0, 0, 0, 0]),
            "description": [
                "Tang DWO", "Tang DWO", "Tang DWO", "Tang DWO", "Tang DWO"
            ],
        }

        # Store capital costs as the class's attribute: "self.capital"
        self.capital = {"oil": capital_oil}

    def get_intangible(self) -> None:
        """
        Prepare and store oil intangible cost data.

        Assembles predefined intangible expenditures (expense years, costs,
        allocation, cost types, and metadata) and assigns them to the
        ``self.intangible`` attribute under the key ``"oil"``.

        Notes
        -----
        - Covers expense years 2024–2028 within a 2022–2035 project period.
        - All costs are allocated to ``FluidType.OIL`` and categorized as
          ``CostType.POST_ONSTREAM_COST``.
        - Tax portion is fixed at zero for all items.
        """

        # Prepare intangible cost: OIL
        intangible_oil = {
            "start_year": 2022,
            "end_year": 2035,
            "expense_year": np.array([2024, 2025, 2026, 2027, 2028]),
            "cost": np.array(
                [
                    138_789.463632672, 64_991.8968456878, 1_017.66466282672,
                    1_109.75384628443, 1_724.02172015002
                ]
            ),
            "cost_allocation": [
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL
            ],
            "cost_type": [
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST
            ],
            "tax_portion": np.array([0, 0, 0, 0, 0]),
            "description": [
                "Intang DWO", "Intang DWO", "Intang DWO", "Intang DWO", "Intang DWO"
            ],
        }

        # Store intangible cost as the class's attribute: "self.intangible"
        self.intangible = {"oil": intangible_oil}

    def get_opex(self) -> None:
        """
        Prepare and store oil operating expenditure (OPEX) data.

        Assembles predefined OPEX information (expense years, fixed costs,
        allocation, cost types, and metadata) and stores it under the
        ``self.opex`` attribute with the key ``"oil"``.

        Notes
        -----
        - Covers expense years 2023–2035 within a 2022–2035 project timeline.
        - Allocation is fully assigned to ``FluidType.OIL``.
        - Cost types include both ``CostType.PRE_ONSTREAM_COST`` and
          ``CostType.POST_ONSTREAM_COST``.
        - Tax portion is zero for all cost entries.
        """

        # Prepare opex: OIL
        opex_oil = {
            "start_year": 2022,
            "end_year": 2035,
            "expense_year": np.array(
                [
                    2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030,
                    2031, 2032, 2033, 2034, 2035
                ]
            ),
            "fixed_cost": np.array(
                [
                    0, 971.292234114079, 2457.37777867217, 3101.47410075184, 3499.0467473904,
                    3643.77149457785, 4144.81094457935, 4557.10420220148, 3879.78558891758,
                    3433.61380883011, 3469.40440315405, 1683.43593564488, 491.387568792144
                ]
            ),
            "cost_allocation": [
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL
            ],
            "cost_type": [
                CostType.PRE_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST
            ],
            "tax_portion": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "description": [
                "all OPEX", "all OPEX", "all OPEX", "all OPEX", "all OPEX", "all OPEX",
                "all OPEX", "all OPEX", "all OPEX", "all OPEX", "all OPEX", "all OPEX",
                "all OPEX"
            ],
        }

        # Store opex as class's attribute: "self.opex"
        self.opex = {"oil": opex_oil}

    def get_asr(self) -> None:
        """
        Prepare and store abandonment and site restoration (ASR) costs.

        Constructs ASR data for oil — including timing, yearly costs,
        allocations, and cost-type classification — and stores it under
        ``self.asr`` with the key ``"oil"``.

        Notes
        -----
        - Timeline covers 2022–2035; expenses occur from 2023 onward.
        - All costs are allocated to ``FluidType.OIL``.
        - The first cost entry is pre-onstream; subsequent entries are classified
          as ``CostType.POST_ONSTREAM_COST``.
        - Cost data is provided as NumPy arrays for vectorized operations.
        """

        # Prepare ASR: OIL
        asr_oil = {
            "start_year": 2022,
            "end_year": 2035,
            "expense_year": np.array(
                [
                    2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031,
                    2032, 2033, 2034, 2035
                ]
            ),
            "cost": np.array(
                [
                    0, 478.57133137928, 838.67985905111, 1136.99948801868,
                    1136.99948801868, 1136.99948801868, 1136.99948801868,
                    1136.99948801868, 1136.99948801868, 1136.99948801868,
                    1136.99948801868, 1136.99948801868, 1136.99948801868
                ]
            ),
            "cost_allocation": [
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL
            ],
            "cost_type": [
                CostType.PRE_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST
            ],
        }

        # Store ASR cost as class's attribute: "self.asr"
        self.asr = {"oil": asr_oil}

    def get_lbt(self) -> None:
        """
        Prepare and store land and building tax (LBT) costs.

        Builds LBT cost data for oil — including cost timeline, yearly
        values, allocation, and cost-type classification — and stores it
        under ``self.lbt`` with the key ``"oil"``.

        Notes
        -----
        - Cost timeline spans 2022–2035; expenses occur from 2023 onward.
        - All entries are allocated to ``FluidType.OIL``.
        - First entry uses ``CostType.PRE_ONSTREAM_COST``; remaining entries
          are ``CostType.POST_ONSTREAM_COST``.
        - ``cost`` and ``expense_year`` are NumPy arrays for vectorized use
          in downstream calculations.
        """

        # Prepare LBT: OIL
        lbt_oil = {
            "start_year": 2022,
            "end_year": 2035,
            "expense_year": np.array(
                [
                    2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031,
                    2032, 2033, 2034, 2035
                ]
            ),
            "cost": np.array(
                [
                    0, 0, 1718.30476967245, 3752.23099766397, 3812.96275956648,
                    2746.88736465813, 1971.62024928616, 1464.00914775356,
                    1000.50006761995, 814.840751988436, 457.13115224711,
                    356.605652566246, 191.296236384866
                ]
            ),
            "cost_allocation": [
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL,
                FluidType.OIL
            ],
            "cost_type": [
                CostType.PRE_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST, CostType.POST_ONSTREAM_COST,
                CostType.POST_ONSTREAM_COST
            ],
        }

        # Store LBT cost data as class's attribute "self.lbt"
        self.lbt = {"oil": lbt_oil}

    def get_setup_arguments(self) -> None:
        """
        Prepare and store general setup arguments for the contract.

        Defines project-wide configuration parameters (dates, approvals, and POD
        status) and stores them in the ``self.setup_arguments`` attribute.
        """

        self.setup_arguments = {
            "start_date": date(year=2022, month=1, day=1),
            "end_date": date(year=2035, month=1, day=1),
            "oil_onstream_date": date(year=2024, month=1, day=1),
            "gas_onstream_date": None,
            "approval_year": None,
            "is_pod_1": False,
        }

    def get_class_arguments(self) -> None:
        """
        Set class-level arguments based on the selected contract type.

        Constructs a mapping of arguments required by each contract class:
        cost-recovery, gross-split, and base-project. For gross-split, the
        method prepares a full set of reservoir, field, DMO, and fiscal
        parameters (using the VariableSplit522017 model). The appropriate
        argument dictionary is then selected based on ``self.contract_type``
        and stored in ``self.class_arguments``.

        Notes
        -----
        - Only ``ContractType.GROSS_SPLIT`` requires predefined parameters.
        - ``ContractType.COST_RECOVERY`` and ``ContractType.BASE_PROJECT``
          currently use empty argument mappings.
        - Raises a ``ValueError`` if ``self.contract_type`` is not recognized.
        """

        # Gross split
        VS_52 = VariableSplit522017

        kwargs_gross_split = {
            # Field and reservoir properties
            "field_status": VS_52.FieldStatus.NO_POD,
            "field_loc": VS_52.FieldLocation.ONSHORE,
            "res_depth": VS_52.ReservoirDepth.LESSEQUAL_2500,
            "infra_avail": VS_52.InfrastructureAvailability.WELL_DEVELOPED,
            "res_type": VS_52.ReservoirType.CONVENTIONAL,
            "api_oil": VS_52.APIOil.LESSTHAN_25,
            "domestic_use": VS_52.DomesticUse.EQUAL_50_UNTIL_LESSTHAN_70,
            "prod_stage": VS_52.ProductionStage.SECONDARY,
            "co2_content": VS_52.CO2Content.LESSTHAN_5,
            "h2s_content": VS_52.H2SContent.LESSTHAN_100,

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

        # Class argument's mapping
        class_args_map = {
            ContractType.COST_RECOVERY: {},
            ContractType.GROSS_SPLIT: kwargs_gross_split,
            ContractType.BASE_PROJECT: {},
        }

        try:
            self.class_arguments = class_args_map[self.contract_type]

        except KeyError:
            raise ValueError(f"Unrecognized contract type: {self.contract_type!r}")

    def get_contract_arguments(self) -> None:
        """
        Assign contract-level arguments based on contract type.

        Selects predefined parameter dictionaries for Base Project,
        Gross Split, or Cost Recovery contracts and stores the selected
        mapping in ``self.contract_arguments``.

        Notes
        -----
        - Base Project and Gross Split share core revenue and inflation
          settings; Gross Split extends these with depreciation, tax,
          amortization, and regime parameters.
        - Raises a ``ValueError`` if ``self.contract_type`` is invalid.
        """

        # Base project
        args_base_project = {
            "sulfur_revenue": OtherRevenue.ADDITION_TO_OIL_REVENUE,
            "electricity_revenue": OtherRevenue.ADDITION_TO_OIL_REVENUE,
            "co2_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
            "vat_rate": 0.0,
            "year_inflation": None,
            "inflation_rate": 0.0,
            "inflation_rate_applied_to": None,
        }

        # Gross split
        args_gross_split = {
            **args_base_project,
            "cum_production_split_offset": 0.0,
            "depr_method": DeprMethod.PSC_DB,
            "decline_factor": 2,
            "sum_undepreciated_cost": True,
            "is_dmo_end_weighted": True,
            "tax_regime": TaxRegime.PREVAILING,
            "effective_tax_rate": 0.2,
            "amortization": True,
            "sunk_cost_method": SunkCostMethod.DEPRECIATED_TANGIBLE,
            "regime": GrossSplitRegime.PERMEN_ESDM_12_2020,
            "initial_amortization_year": InitialYearAmortizationIncurred.ONSTREAM_YEAR,
        }

        # Pooled args
        ctr_args = {
            ContractType.COST_RECOVERY: {},
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
            "npv_mode": NPVSelection.NPV_POINT_FORWARD,
            "discounting_mode": DiscountingMode.MID_YEAR,
            "discount_rate_start_year": 2022,
            "inflation_rate": 0.0,
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
        Build and return a contract class instance.

        Creates per-fluid objects for lifting and cost components, merges them
        with setup and contract-level arguments, and initializes a ``GrossSplit``
        contract instance.

        Notes
        -----
        - Currently supports only ``ContractType.GROSS_SPLIT``; raises
          ``ValueError`` for other types.
        - Lifting, capital, intangible, OPEX, ASR, and LBT data are converted
          into their respective component classes before assembly.
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
