"""
CASE 04
"""

import numpy as np
from datetime import date
from dataclasses import dataclass, field
from functools import reduce
from itertools import chain

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
class Case04:

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

        lifting_oil = {
            "start_year": 2022,
            "end_year": 2059,
            "prod_year": None,
            "lifting_rate": None,
            "price": None,
            "fluid_type": None,
            "prod_rate_baseline": None,
        }

        self.lifting = {"oil": lifting_oil}

    def get_capital(self):
        pass

    def get_intangible(self):
        pass

    def get_opex(self):
        pass

    def get_asr(self):
        pass

    def get_lbt(self):
        pass

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

            # Ministry discretion
            "split_ministry_disc": 0.0,

            # DMO: None,
            "oil_dmo_volume_portion": 0.25,
            "oil_dmo_fee_portion": 1.0,
            "oil_dmo_holiday_duration": 60,

            # Carry forward depreciation
            "oil_carry_forward_depreciation": 0.0,
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

    def as_dict(self):
        pass

    def as_class(self):
        pass
