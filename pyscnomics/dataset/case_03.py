"""
CASE 03
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
class Case03:

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
        pass

    def get_capital(self) -> None:
        pass

    def get_intangible(self) -> None:
        pass

    def get_opex(self) -> None:
        pass

    def get_asr(self) -> None:
        pass

    def get_lbt(self) -> None:
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
            "end_date": date(year=2041, month=12, day=31),
            "oil_onstream_date": date(year=2024, month=1, day=1),
            "gas_onstream_date": None,
            "approval_year": None,
            "is_pod_1": False,
        }

    def get_class_arguments(self) -> None:

        # Gross split regime
        VS_08 = VariableSplit082017

        kwargs_gross_split = {
            # Field and reservoir properties
            "field_status": None,
            "field_loc": None,
            "res_depth": None,
            "infra_avail": None,
            "res_type": None,
            "api_oil": None,
            "domestic_use": None,
            "prod_stage": None,
            "co2_content": None,
            "h2s_content": None,

            # Ministry discretion
            "split_ministry_disc": None,

            # DMO
            "oil_dmo_volume_portion": None,
            "oil_dmo_fee_portion": None,
            "oil_dmo_holiday_duration": None,
            "gas_dmo_volume_portion": None,
            "gas_dmo_fee_portion": None,
            "gas_dmo_holiday_duration": None,

            # Carry forward depreciation
            "oil_carry_forward_depreciation": None,
            "gas_carry_forward_depreciation": None,
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
        pass

    def get_summary_arguments(self) -> None:
        pass

    def as_dict(self) -> None:
        pass

    def as_class(self) -> None:
        pass
