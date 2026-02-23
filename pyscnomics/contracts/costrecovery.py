"""
Handles calculations associated with PSC Cost Recovery.
"""

# import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from pyscnomics.econ.costs import CapitalCost
from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts import psc_tools
from pyscnomics.econ.selection import (
    FluidType,
    InflationAppliedTo,
    TaxRegime,
    FTPTaxRegime,
    TaxSplitTypeCR,
    DeprMethod,
    OtherRevenue,
    # SunkCostMethod,
    # InitialYearDepreciationIncurred,
    NPVSelection,
    DiscountingMode,
    ContractType,
)
from pyscnomics.econ.indicator import (
    irr,
    npv_nominal_terms,
    npv_real_terms,
    npv_skk_nominal_terms,
    npv_skk_real_terms,
    npv_point_forward,
    pot_psc,
)

# Set display for pandas dataframe
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_colwidth", 100)


class SunkCostException(Exception):
    """Exception to raise for a misuse of Sunk Cost Method"""

    pass


class CostRecoveryException(Exception):
    """Exception to raise for a misuse of Cost Recovery attributes"""

    pass


class CostRecoverySummaryException(Exception):
    """ Exception to be raised for a misuse of get_summary() method """

    pass


@dataclass
class CostRecovery(BaseProject):
    """
    Dataclass that represents Cost Recovery (CR) contract.

    Parameters
    ----------
    oil_ftp_is_available: bool
        The availability of the Oil's First Tranche Petroleum (FTP).
    oil_ftp_is_shared: bool
        The condition whether the Oil FTP will be shared to the contractor or not.
    oil_ftp_portion: float
        The portion of the Oil FTP in fraction.
    gas_ftp_is_available: bool
        The availability of the Gas's First Tranche Petroleum (FTP).
    gas_ftp_is_shared: bool
        The condition whether the Gas FTP will be shared to the contractor or not.
    gas_ftp_portion: float
        The portion of the Gas FTP in fraction.
    tax_split_type: TaxSplitTypeCR
        The type of the contractor split.
    condition_dict: dict
        The condition dictionary of the split when the tax_split_type is not Conventional.
    indicator_rc_icp_sliding: np.ndarray
        The indicator used in the SLIDING_SCALE or R2C tax_split_type.
    oil_ctr_pretax_share: float | np.ndarray
        The Oil Contractor Pre-Tax Split or Oil Contractor Pre-Tax Split Share.
    gas_ctr_pretax_share: float | np.ndarray
        The Gas Contractor Pre-Tax Split or Oil Contractor Pre-Tax Split Share.
    oil_ic_rate: float
        The Oil's Investment Credit (IC) rate of the contract.
    gas_ic_rate: float
        The Gas's Investment Credit (IC) rate of the contract.
    ic_is_available: bool
        The condition whether if IC is available or not.
    oil_cr_cap_rate: float
        The Oil Cost Recovery cap rate.
    gas_cr_cap_rate: float
        The Gas Cost Recovery cap rate.
    oil_dmo_volume_portion: float
        The Oil's Domestic Market Obligation (DMO) volume portion.
    oil_dmo_fee_portion: float
        The Oil's DMO fee portion.
    oil_dmo_holiday_duration: int
        The duration of the Oil DMO Holiday in month unit.
    gas_dmo_volume_portion: float
        The Gas's Domestic Market Obligation (DMO) volume portion.
    gas_dmo_fee_portion: float
        The Gas's DMO fee portion.
    gas_dmo_holiday_duration: int
        The duration of the Gas's DMO Holiday in month unit.
    """

    # Arguments associated with FTP
    oil_ftp_is_available: bool = field(default=True)
    oil_ftp_is_shared: bool = field(default=True)
    oil_ftp_portion: float | np.ndarray = field(default=0.2)
    gas_ftp_is_available: bool = field(default=True)
    gas_ftp_is_shared: bool = field(default=True)
    gas_ftp_portion: float | np.ndarray = field(default=0.2)

    # Arguments associated with split
    tax_split_type: TaxSplitTypeCR = field(default=TaxSplitTypeCR.CONVENTIONAL)
    condition_dict: dict = field(default_factory=dict)
    indicator_rc_icp_sliding: np.ndarray = field(default=None)
    oil_ctr_pretax_share: float | np.ndarray = field(default=0.25)
    gas_ctr_pretax_share: float | np.ndarray = field(default=0.5)

    # Arguments associated with investment credit and cap rate
    oil_ic_rate: float | np.ndarray = field(default=0.0)
    gas_ic_rate: float | np.ndarray = field(default=0.0)
    ic_is_available: bool = field(default=False)
    oil_cr_cap_rate: float = field(default=1.0)
    gas_cr_cap_rate: float = field(default=1.0)

    # Arguments associated with DMO
    oil_dmo_volume_portion: float | np.ndarray = field(default=0.25)
    oil_dmo_fee_portion: float | np.ndarray = field(default=0.25)
    oil_dmo_holiday_duration: int = field(default=60)
    gas_dmo_volume_portion: float | np.ndarray = field(default=1.0)
    gas_dmo_fee_portion: float | np.ndarray = field(default=1.0)
    gas_dmo_holiday_duration: int = field(default=60)

    # Arguments associated with carry forward depreciation
    oil_carry_forward_depreciation: float | np.ndarray = field(default=0.0)
    gas_carry_forward_depreciation: float | np.ndarray = field(default=0.0)

    # Attributes associated with carry forward depreciation
    _oil_carry_forward_depreciation: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _gas_carry_forward_depreciation: np.ndarray = field(
        default=None, init=False, repr=False
    )

    # Attributes associated with depreciations and undepreciated assets
    _oil_depreciations: dict = field(default_factory=lambda: {}, init=False, repr=False)
    _gas_depreciations: dict = field(default_factory=lambda: {}, init=False, repr=False)
    _oil_undepreciated_assets: dict = field(default_factory=lambda: {}, init=False, repr=False)
    _gas_undepreciated_assets: dict = field(default_factory=lambda: {}, init=False, repr=False)
    _oil_depreciation: np.ndarray = field(default=None, init=False, repr=False)
    _gas_depreciation: np.ndarray = field(default=None, init=False, repr=False)
    _oil_sum_undepreciated_asset: np.ndarray = field(default=None, init=False, repr=False)
    _gas_sum_undepreciated_asset: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes associated with non-depreciable costs
    _oil_non_depreciables: dict = field(default_factory=lambda: {}, init=False, repr=False)
    _gas_non_depreciables: dict = field(default_factory=lambda: {}, init=False, repr=False)

    # Attributes associated with total depreciation and total nondepreciable
    _oil_depr_total: np.ndarray = field(default=None, init=False, repr=False)
    _gas_depr_total: np.ndarray = field(default=None, init=False, repr=False)
    _oil_non_depr_total: np.ndarray = field(default=None, init=False, repr=False)
    _gas_non_depr_total: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes associated with FTP
    _oil_ftp: np.ndarray = field(default=None, init=False, repr=False)
    _oil_ftp_ctr: np.ndarray = field(default=None, init=False, repr=False)
    _oil_ftp_gov: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ftp: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ftp_ctr: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ftp_gov: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes associated with investment credit
    _oil_ic: np.ndarray = field(default=None, init=False, repr=False)
    _oil_ic_unrecovered: np.ndarray = field(default=None, init=False, repr=False)
    _oil_ic_paid: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ic: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ic_unrecovered: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ic_paid: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes associated with core business logic
    _oil_revenue_allocation: np.ndarray = field(default=None, init=False, repr=False)
    _gas_revenue_allocation: np.ndarray = field(default=None, init=False, repr=False)
    _oil_unrecovered_before_transfer: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _gas_unrecovered_before_transfer: np.ndarray = field(
        default=None, init=False, repr=False
    )
    # _oil_cost_to_be_recovered: np.ndarray = field(default=None, init=False, repr=False)
    # _gas_cost_to_be_recovered: np.ndarray = field(default=None, init=False, repr=False)
    _oil_recoverable_cost: np.ndarray = field(default=None, init=False, repr=False)
    _gas_recoverable_cost: np.ndarray = field(default=None, init=False, repr=False)
    _oil_cost_recovery: np.ndarray = field(default=None, init=False, repr=False)
    _gas_cost_recovery: np.ndarray = field(default=None, init=False, repr=False)
    _oil_ets_before_transfer: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ets_before_transfer: np.ndarray = field(default=None, init=False, repr=False)
    _transfer_to_oil: np.ndarray = field(default=None, init=False, repr=False)
    _transfer_to_gas: np.ndarray = field(default=None, init=False, repr=False)

    _oil_unrecovered_after_transfer: np.ndarray = field(default=None, init=False, repr=False)
    _gas_unrecovered_after_transfer: np.ndarray = field(default=None, init=False, repr=False)
    # _oil_cost_to_be_recovered_after_tf: np.ndarray = field(default=None, init=False, repr=False)
    # _gas_cost_to_be_recovered_after_tf: np.ndarray = field(default=None, init=False, repr=False)
    _oil_cost_recovery_after_tf: np.ndarray = field(default=None, init=False, repr=False)
    _gas_cost_recovery_after_tf: np.ndarray = field(default=None, init=False, repr=False)
    _oil_ets_after_transfer: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ets_after_transfer: np.ndarray = field(default=None, init=False, repr=False)

    _oil_contractor_share: np.ndarray = field(default=None, init=False, repr=False)
    _oil_government_share: np.ndarray = field(default=None, init=False, repr=False)
    _gas_contractor_share: np.ndarray = field(default=None, init=False, repr=False)
    _gas_government_share: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes associated with DMO
    _oil_dmo_volume: np.ndarray = field(default=None, init=False, repr=False)
    _oil_dmo_fee: np.ndarray = field(default=None, init=False, repr=False)
    _oil_ddmo: np.ndarray = field(default=None, init=False, repr=False)
    _gas_dmo_volume: np.ndarray = field(default=None, init=False, repr=False)
    _gas_dmo_fee: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ddmo: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes associated with taxable income
    _oil_taxable_income: np.ndarray = field(default=None, init=False, repr=False)
    _gas_taxable_income: np.ndarray = field(default=None, init=False, repr=False)
    _tax_rate_arr: np.ndarray = field(default=None, init=False, repr=False)
    _oil_tax_payment: np.ndarray = field(default=None, init=False, repr=False)
    _gas_tax_payment: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes associated with shares
    _oil_ctr_net_share: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ctr_net_share: np.ndarray = field(default=None, init=False, repr=False)
    _oil_contractor_take: np.ndarray = field(default=None, init=False, repr=False)
    _gas_contractor_take: np.ndarray = field(default=None, init=False, repr=False)
    _oil_government_take: np.ndarray = field(default=None, init=False, repr=False)
    _gas_government_take: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes associated with consolidated profiles
    _consolidated_carry_forward_depreciation: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_depreciations: dict = field(
        default_factory=lambda: {}, init=False, repr=False
    )
    # _consolidated_undepreciated_assets: dict = field(
    #     default_factory=lambda: {}, init=False, repr=False
    # )
    _consolidated_non_depreciables: dict = field(
        default_factory=lambda: {}, init=False, repr=False
    )
    _consolidated_depreciation: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_sum_undepreciated_asset: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_depr_total: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_non_depr_total: np.ndarray = field(default=None, init=False, repr=False)

    _consolidated_ftp: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_ftp_ctr: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_ftp_gov: np.ndarray = field(default=None, init=False, repr=False)

    _consolidated_ic: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_ic_unrecovered: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_ic_paid: np.ndarray = field(default=None, init=False, repr=False)

    # _consolidated_cost_to_be_recovered_before_tf: np.ndarray = field(
    #    default=None, init=False, repr=False
    # )
    _consolidated_recoverable_cosdt: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_unrecovered_before_transfer: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_cost_recovery_before_transfer: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_ets_before_transfer: np.ndarray = field(
        default=None, init=False, repr=False
    )

    # _consolidated_cost_to_be_recovered_after_tf: np.ndarray = field(
    #    default=None, init=False, repr=False
    # )
    _consolidated_unrecovered_after_transfer: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_cost_recovery_after_tf: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_ets_after_transfer: np.ndarray = field(
        default=None, init=False, repr=False
    )

    _consolidated_contractor_share: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_government_share: np.ndarray = field(
        default=None, init=False, repr=False
    )

    _consolidated_dmo_volume: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_dmo_fee: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_ddmo: np.ndarray = field(default=None, init=False, repr=False)

    _consolidated_taxable_income: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_tax_due: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_unpaid_tax_balance: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_tax_payment: np.ndarray = field(default=None, init=False, repr=False)

    _consolidated_ctr_net_share: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_contractor_take: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_government_take: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """
        Handles the following operations/procedures:
        -   Perform initial check to several input arguments.
        -   Initialize attributes associated with OIL FTP.
        -   Initialize attributes associated with GAS FTP.
        """

        # Call parent's post init from class BaseProject
        super().__post_init__()

        # Perform initial check to several input arguments
        self._check_attributes()

        # Initialize attributes associated with OIL FTP
        self._oil_ftp = np.zeros_like(self.project_years, dtype=float)
        self._oil_ftp_ctr = np.zeros_like(self.project_years, dtype=float)
        self._oil_ftp_gov = np.zeros_like(self.project_years, dtype=float)

        # Initialize attributes associated with GAS FTP
        self._gas_ftp = np.zeros_like(self.project_years, dtype=float)
        self._gas_ftp_ctr = np.zeros_like(self.project_years, dtype=float)
        self._gas_ftp_gov = np.zeros_like(self.project_years, dtype=float)

    def _check_attributes(self) -> None:
        """
        Validate and normalize project attributes for fractions, DMO holidays,
        and carry-forward depreciation.

        This method performs consistency checks on attributes related to
        fractional values (e.g., FTP portion, pretax share, investment credit rate,
        cost recovery cap, and DMO portions), discrete values (DMO holiday durations),
        and carry-forward depreciation arrays. It ensures correct data types,
        value ranges, and dimensional consistency with the project duration.
        Carry-forward depreciation arrays are also standardized to match the
        length of project years.

        Raises
        ------
        CostRecoveryException
            If any attribute has an invalid type, contains values outside
            the allowed range, has inconsistent array length, or contains
            negative values.

        Returns
        -------
        None
            This method updates internal attributes in-place by normalizing
            carry-forward depreciation arrays. No return value.
        """

        # Prepare fractional attributes
        min_fraction = 0.0
        max_fraction = 1.0

        # Defining any attributes that in the form of fraction
        fraction_attributes = (
            ("oil_ftp_portion", self.oil_ftp_portion),
            ("gas_ftp_portion", self.gas_ftp_portion),
            ("oil_ctr_pretax_share", self.oil_ctr_pretax_share),
            ("gas_ctr_pretax_share", self.gas_ctr_pretax_share),
            ("oil_ic_rate", self.oil_ic_rate),
            ("gas_ic_rate", self.gas_ic_rate),
            ("oil_cr_cap_rate", self.oil_cr_cap_rate),
            ("gas_cr_cap_rate", self.gas_cr_cap_rate),
            ("oil_dmo_volume_portion", self.oil_dmo_volume_portion),
            ("oil_dmo_fee_portion", self.oil_dmo_fee_portion),
            ("gas_dmo_volume_portion", self.gas_dmo_volume_portion),
            ("gas_dmo_fee_portion", self.gas_dmo_fee_portion),
        )

        for attr_name, attr in fraction_attributes:
            if not isinstance(attr, (float, int, np.ndarray)):
                raise CostRecoveryException(
                    f"Attribute {attr_name} should be given as an int, float, "
                    f"or a numpy ndarray, not as a/an "
                    f"{attr.__class__.__qualname__}"
                )

            if isinstance(attr, float) or isinstance(attr, int):
                if not (min_fraction <= attr <= max_fraction):
                    range_type = "exceeding 1.0" if attr > max_fraction else "below 0"
                    raise CostRecoveryException(
                        f"The {attr_name} value ({attr}) is {range_type}. "
                        f"Allowed range: {min_fraction} - {max_fraction}."
                    )

            elif isinstance(attr, np.ndarray):
                if len(attr) != len(self.project_years):
                    raise CostRecoveryException(
                        f"The {attr_name} length is ({len(attr)}). "
                        f"The length of {attr_name} ({len(attr)}) is different from "
                        f"the length of the project {len(self.project_duration)}"
                    )

        # Prepare attributes associated with DMO holiday
        discrete_attributes = (
            ("oil_dmo_holiday_duration", self.oil_dmo_holiday_duration),
            ("gas_dmo_holiday_duration", self.gas_dmo_holiday_duration),
        )

        for attr_name, attr in discrete_attributes:
            if attr < 0:
                raise CostRecoveryException(
                    f"The {attr_name} value ({attr}) is below 0. "
                    f"The minimum value of this attribute is 0. "
                )

        # Prepare attributes associated with carry forward depreciation
        carward_depr = {
            "oil_carry_forward_depreciation": self.oil_carry_forward_depreciation,
            "gas_carry_forward_depreciation": self.gas_carry_forward_depreciation,
        }

        for attr_name, value in carward_depr.items():
            # Check for unsuitable data type
            if not isinstance(value, (np.ndarray, float, int)):
                raise CostRecoveryException(
                    f"Attribute {attr_name} must be given as a numpy ndarray, "
                    f"a float, or an integer, not as a/an "
                    f"{value.__class__.__qualname__}"
                )

            # Check for negative values
            if np.any(value < 0):
                raise CostRecoveryException(f"The {attr_name} contains negative values")

            # Check length exceeding project duration
            if isinstance(value, np.ndarray) and len(value) > self.project_duration:
                raise CostRecoveryException(
                    f"The {attr_name} length ({len(value)}) exceeds "
                    f"project duration ({self.project_duration})"
                )

            # Modify 'carward_depr' to array with length equals to project duration
            if isinstance(value, (np.ndarray, float, int)):
                arr = np.atleast_1d(value).astype(float)
                result = np.zeros_like(self.project_years, dtype=float)
                result[:len(arr)] = arr
                carward_depr[attr_name] = result

        self._oil_carry_forward_depreciation = carward_depr["oil_carry_forward_depreciation"]
        self._gas_carry_forward_depreciation = carward_depr["gas_carry_forward_depreciation"]

    def _check_invalid_capital_sunk_costs(
        self,
        sc_capital_object: CapitalCost,
        sc_capital_name: str,
        is_strict: bool,
        onstream_year: int,
    ) -> None:
        """
        Validate that sunk costs are not treated as capital costs under PSC Cost Recovery.

        This method checks a ``CapitalCost`` object for non-zero sunk costs. If such costs
        are found, strict mode raises an exception, while loose mode records a warning and
        additionally validates capital PIS years against the onstream year.

        Parameters
        ----------
        sc_capital_object : CapitalCost
            Capital cost object containing sunk cost values and expense years.
        sc_capital_name : str
            Descriptive name of the sunk cost item, used in messages.
        is_strict : bool
            If True, invalid sunk costs raise an exception; otherwise, a warning is stored.
        onstream_year : int
            Onstream year used for PIS-year validation in loose mode.

        Raises
        ------
        CostRecoveryException
            If non-zero sunk costs are found and ``is_strict`` is True.
        """

        # Identify non-zero sunk costs
        name = sc_capital_name
        nonzero_mask = (sc_capital_object.cost != 0.0)

        # Exit early if all sunk costs are zeros
        if not np.any(nonzero_mask):
            return

        # Extract invalid costs and their corresponding years
        invalid_costs = sc_capital_object.cost[nonzero_mask]
        invalid_years = sc_capital_object.expense_year[nonzero_mask]

        # Pair invalid expense years with invalid costs
        invalid = [f"{yr}: {cst}" for yr, cst in zip(invalid_years, invalid_costs)]

        # Specify messages to be displayed
        msg_error = (
            f"Cannot allow {name!r} ({invalid}) in PSC Cost Recovery (CR) contract. "
            f"Conceptually, all sunk costs in PSC CR should be classified as "
            f"intangible cost--with NO DEPRECIATION APPLIED TO THEM--, not capital cost."
        )

        msg_warning = (
            f"Found {name!r} ({invalid}) in PSC Cost Recovery (CR) contract. "
            f"Conceptually, all sunk costs in PSC CR should be classified as "
            f"intangible cost--with NO DEPRECIATION APPLIED TO THEM--, not capital cost. "
            f"Classifying sunk costs as capital cost in PSC CR contract may produce "
            f"an incorrect calculation, where depreciation will be applied to them."
        )

        # Strict mode: raise an error and stop execution
        if is_strict:
            raise CostRecoveryException(msg_error)

        # Loose mode: record a warning message, validate PIS years with onstream year,
        # and continue execution
        else:
            self.warning_messages.append(
                (ContractType.COST_RECOVERY.value, msg_warning)
            )

            self._check_capital_pis_years_before_onstream(
                obj_capital=sc_capital_object,
                obj_name=sc_capital_name,
                is_amortization=False,
                is_strict=False,
                onstream_year=onstream_year,
            )

    def _prepare_depreciation(self) -> None:
        """
        Prepare and validate capital depreciation inputs before calculation.

        This method determines the project onstream year, validates that sunk costs are
        not treated as capital costs, and checks that capital PIS years do not precede
        the onstream year. Validation behavior follows the project strictness setting.

        Raises
        ------
        BaseProjectException or CostRecoveryException
            If validation fails in strict mode.
        """

        # Specify onstream year
        onstream_yr = min(self.oil_onstream_date.year, self.gas_onstream_date.year)

        # Collection of mappings
        mapping_capital_sunk_costs = [
            (self._oil_capital_sunk_cost, "oil_capital_sunk_cost"),
            (self._gas_capital_sunk_cost, "gas_capital_sunk_cost"),
        ]

        mapping_capital_preonstreams = [
            (self._oil_capital_preonstream, "oil_capital_preonstream"),
            (self._gas_capital_preonstream, "gas_capital_preonstream"),
        ]

        # Check whether sunk costs are treated as capital costs
        for sc_obj, sc_name in mapping_capital_sunk_costs:
            self._check_invalid_capital_sunk_costs(
                sc_capital_object=sc_obj,
                sc_capital_name=sc_name,
                is_strict=self.is_strict,
                onstream_year=onstream_yr,
            )

        # Check capital preonstream costs: whether PIS years < onstream year
        for preos_obj, preos_name in mapping_capital_preonstreams:
            self._check_capital_pis_years_before_onstream(
                obj_capital=preos_obj,
                obj_name=preos_name,
                is_amortization=False,
                is_strict=self.is_strict,
                onstream_year=onstream_yr,
            )

    def _get_depreciation(
        self,
        depr_method: DeprMethod,
        decline_factor: float | int,
        year_inflation: np.ndarray,
        inflation_rate: np.ndarray | int | float,
        tax_rate: np.ndarray | float,
        inflation_rate_applied_to: InflationAppliedTo,
    ) -> None:
        """
        Compute and assign depreciation schedules for oil and gas capital costs.

        This method validates depreciation inputs, computes annual depreciation and
        undepreciated asset balances for oil and gas capital costs, and stores the
        results internally by cost type (``sunk_cost``, ``preonstream``,
        ``postonstream``).

        Parameters
        ----------
        depr_method : DeprMethod
            Depreciation method to apply.
        decline_factor : float or int
            Decline factor for declining-balance depreciation.
        year_inflation : np.ndarray
            Yearly inflation multipliers.
        inflation_rate : np.ndarray or float or int
            Inflation rate(s) applied in depreciation calculations.
        tax_rate : np.ndarray or float
            Applicable tax rate(s).
        inflation_rate_applied_to : InflationAppliedTo
            Specifies how inflation is applied in depreciation.

        Returns
        -------
        None
            Updates internal depreciation and undepreciated asset attributes for oil
            and gas.
        """

        # Conduct preliminary evaluations
        self._prepare_depreciation()

        # Define the mapping between fluids, cost types, and capital objects
        depr_mapping = {
            "oil": (
                ("sunk_cost", self._oil_capital_sunk_cost),
                ("preonstream", self._oil_capital_preonstream),
                ("postonstream", self._oil_capital_postonstream),
            ),
            "gas": (
                ("sunk_cost", self._gas_capital_sunk_cost),
                ("preonstream", self._gas_capital_preonstream),
                ("postonstream", self._gas_capital_postonstream),
            )
        }

        # Define intermediate attributes:
        # "depreciations": dict and "undepreciated_assets": dict
        cost_types = ["sunk_cost", "preonstream", "postonstream"]

        depreciations = {f: {c: None for c in cost_types} for f in depr_mapping.keys()}
        undepreciated_assets = {f: {c: None for c in cost_types} for f in depr_mapping.keys()}

        # Common arguments passed to the depreciation calculator
        kwargs_depr = {
            "depr_method": depr_method,
            "decline_factor": decline_factor,
            "year_inflation": year_inflation,
            "inflation_rate": inflation_rate,
            "tax_rate": tax_rate,
            "inflation_rate_applied_to": inflation_rate_applied_to,
        }

        # Compute depreciation per fluid per cost type
        for f, mapping in depr_mapping.items():
            for c, obj_cap in mapping:
                (
                    depreciations[f][c],
                    undepreciated_assets[f][c]
                ) = obj_cap.total_depreciation_rate(**kwargs_depr)

        # Store the results as class's attributes
        self._oil_depreciations = depreciations["oil"]
        self._gas_depreciations = depreciations["gas"]
        self._oil_undepreciated_assets = undepreciated_assets["oil"]
        self._gas_undepreciated_assets = undepreciated_assets["gas"]

    def _allocate_non_depreciable_sunk_cost(
        self, non_depreciable_sunk_cost: np.ndarray
    ) -> np.ndarray:
        """
        Allocate non-depreciable sunk costs to the project onstream year.

        All non-depreciable sunk costs are aggregated into a single bulk value
        and assigned entirely to the earliest onstream year (oil or gas) within
        the project timeline.

        Parameters
        ----------
        non_depreciable_sunk_cost : np.ndarray
            Array of non-depreciable sunk cost values across project years.

        Returns
        -------
        np.ndarray
            Array aligned with ``project_years``, where the total sunk cost is
            allocated at the onstream year and zeros elsewhere.

        Raises
        ------
        CostRecoveryException
            If the onstream year cannot be uniquely identified in
            ``project_years``.
        """

        # Identify onstream year in project years array
        onstream_yr = min([self.oil_onstream_date.year, self.gas_onstream_date.year])
        match_at_onstream = np.flatnonzero(self.project_years == onstream_yr)
        if match_at_onstream.size != 1:
            raise CostRecoveryException(
                f"Expected one onstream year match, got {match_at_onstream.size} instead."
            )

        # Configure index of onstream year in project years array
        onstream_id = int(match_at_onstream[0])

        # Calculate bulk value of non-depreciable_sunk_cost
        bulk = non_depreciable_sunk_cost.sum(dtype=float)

        # Allocate bulk value of non-depreciable sunk cost at the onstream year
        arr = np.zeros_like(self.project_years, dtype=float)
        arr[onstream_id] = bulk

        return arr

    def _get_non_depreciables(self) -> None:
        """
        Assemble non-depreciable cost arrays for oil and gas.

        This method constructs dictionaries of non-depreciable costs by cost
        category (sunk cost, pre-onstream, post-onstream) for oil and gas.
        Non-depreciable sunk costs are aggregated and allocated to the
        onstream year, while other components are taken directly from their
        respective arrays.

        Sets
        ----
        _oil_non_depreciables : dict[str, np.ndarray]
            Non-depreciable oil costs by cost category.
        _gas_non_depreciables : dict[str, np.ndarray]
            Non-depreciable gas costs by cost category.
        """

        # Construct intermediate variable "non_depreciables" which stores
        # non-depreciable costs for each fluids and each cost types
        non_depreciables = {
            "oil": {
                "sunk_cost": self._allocate_non_depreciable_sunk_cost(
                    non_depreciable_sunk_cost=self._oil_non_depreciable_sunk_cost
                ),
                "preonstream": self._oil_non_depreciable_preonstream,
                "postonstream": self._oil_non_depreciable_postonstream,
            },
            "gas": {
                "sunk_cost": self._allocate_non_depreciable_sunk_cost(
                    non_depreciable_sunk_cost=self._gas_non_depreciable_sunk_cost
                ),
                "preonstream": self._gas_non_depreciable_preonstream,
                "postonstream": self._gas_non_depreciable_postonstream,
            },
        }

        # Stores processed non-depreciable costs as class's attributes
        self._oil_non_depreciables = non_depreciables["oil"]
        self._gas_non_depreciables = non_depreciables["gas"]

    def _get_modified_depreciations(self, sum_undepreciated_cost: bool) -> None:
        """
        Adjust oil and gas depreciation and undepreciated asset schedules.

        This method performs in-place modifications to the undepreciated asset
        and depreciation arrays for both oil and gas by:
        (i) removing numerically insignificant undepreciated balances, and
        (ii) optionally transferring remaining undepreciated costs into the
        final year of the corresponding depreciation schedules.

        Parameters
        ----------
        sum_undepreciated_cost : bool
            True  → sum undepreciated by cost type → final depreciation year
            False → keep undepreciated (after tolerance cleanup)

        Returns
        -------
        None
            Updates the following attributes in place:
            ``_oil_depreciations``, ``_gas_depreciations``,
            ``_oil_undepreciated_assets``, and ``_gas_undepreciated_assets``.

        Notes
        -----
        - Cost types: sunk_cost | preonstream | postonstream
        - Final-year adjustment affects depreciation only
        """

        undepre_assets = [self._oil_undepreciated_assets, self._gas_undepreciated_assets]
        cost_types = ["sunk_cost", "preonstream", "postonstream"]

        # Treatment for small order of number (example 1e-5) in undepreciated assets
        tol = 1.0e-5
        for asset in undepre_assets:
            for c in cost_types:
                asset[c][asset[c] < tol] = 0.0

        # Treatment whether the undepreciated asset is summed up in
        # the last year of the contract
        if sum_undepreciated_cost:
            for depr, undepr in [
                (self._oil_depreciations, self._oil_undepreciated_assets),
                (self._gas_depreciations, self._gas_undepreciated_assets)
            ]:
                for c in cost_types:
                    depr[c][-1] += undepr[c].sum()
                    undepr[c] = np.zeros([1, 1], dtype=float)

    def _get_ftp(self) -> None:
        """
        Calculate and allocate First Tranche Petroleum (FTP) for oil and gas.

        This method determines the portion of oil and gas revenues allocated to
        First Tranche Petroleum (FTP) under Production Sharing Contract (PSC) terms.

        If FTP is available, the allocation is split between the contractor and the
        government based on whether FTP sharing is applicable. If FTP is not shared,
        the entire FTP portion is assigned to the government.

        Updates
        -------
        self._oil_ftp : np.ndarray
            Total oil FTP before split.
        self._gas_ftp : np.ndarray
            Total gas FTP before split.
        self._oil_ftp_ctr : np.ndarray
            Contractor's share of oil FTP.
        self._gas_ftp_ctr : np.ndarray
            Contractor's share of gas FTP.
        self._oil_ftp_gov : np.ndarray
            Government's share of oil FTP.
        self._gas_ftp_gov : np.ndarray
            Government's share of gas FTP.

        Notes
        -----
        - If ``oil_ftp_is_available`` is ``False``, all oil FTP-related arrays remain zero.
        - If ``gas_ftp_is_available`` is ``False``, all gas FTP-related arrays remain zero.
        - If FTP sharing is disabled, the contractor's share remains zero and
          the government receives the full FTP.
        - Computation depends on the attributes:
          ``oil_ftp_portion``, ``gas_ftp_portion``,
          ``oil_ctr_pretax_share``, and ``gas_ctr_pretax_share``.
        """

        # Sharing condition if OIL FTP is available.
        # Fill attributes "_oil_ftp", "_oil_ftp_portion", and "_oil_ftp_gov"
        if self.oil_ftp_is_available:
            self._oil_ftp = self.oil_ftp_portion * self._oil_revenue
            if self.oil_ftp_is_shared:
                self._oil_ftp_ctr = self.oil_ctr_pretax_share * self._oil_ftp
            self._oil_ftp_gov = self._oil_ftp - self._oil_ftp_ctr

        # Sharing condition if GAS FTP is available.
        # Fill attributes "_gas_ftp", "_gas_ftp_portion", and "_gas_ftp_gov"
        if self.gas_ftp_is_available:
            self._gas_ftp = self.gas_ftp_portion * self._gas_revenue
            if self.gas_ftp_is_shared:
                self._gas_ftp_ctr = self.gas_ctr_pretax_share * self._gas_ftp
            self._gas_ftp_gov = self._gas_ftp - self._gas_ftp_ctr

    def _apply_cost_of_sales(self, oil_applied: bool, gas_applied: bool) -> None:
        """
        Apply cost of sales deductions to oil and gas revenues.

        This method reduces oil and/or gas revenues by their corresponding
        cost of sales expenditures (after tax), provided that revenue
        is available. If no revenue is present for a given stream, a
        CostRecoveryException is raised.

        Parameters
        ----------
        oil_applied : bool
            Whether to apply cost of sales to oil revenue.
        gas_applied : bool
            Whether to apply cost of sales to gas revenue.

        Raises
        ------
        CostRecoveryException
            If oil or gas cost of sales is applied but the respective
            revenue is zero across all project years.

        Notes
        -----
        - Revenue arrays (``_oil_revenue`` and ``_gas_revenue``) are
          updated in place.
        - Negative revenues are not expected; if all values are zero,
          it indicates absence of sales, hence the exception.
        """

        if oil_applied:
            # Ensure there is at least some positive oil revenue
            if not np.any(self._oil_revenue > 0):
                raise CostRecoveryException(
                    "Oil revenue is zero throughout project years"
                )

            # Deduct cost of sales (post-tax) from oil revenue
            self._oil_revenue -= self._oil_cost_of_sales_expenditures_post_tax

        if gas_applied:
            # Ensure there is at least some positive gas revenue
            if not np.any(self._gas_revenue > 0):
                raise CostRecoveryException(
                    "Gas revenue is zero throughout project years"
                )

            # Deduct cost of sales (post-tax) from gas revenue
            self._gas_revenue -= self._gas_cost_of_sales_expenditures_post_tax

    def _get_rc_icp_pretax(self) -> None:
        """
        A Function to get the value of PreTax Split using Revenue over Cost (RC)
        or Indonesian Crude Price (ICP) sliding scale.

        Notes
        -------
        The structure of the dictionary is as the following:
        condition_dict =
        {
            'condition_1': {'bot_limit': 0,
                            'top_limit': 1,
                            'ctr_oil': 0.3137260,
                            'ctr_gas': 0.5498040,
                            'gov_oil': 0.686274,
                            'gov_gas': 0.45098,
                            'split_ctr_oil': 20,
                            'split_ctr_gas': 35,
                            'split_gov_oil': 80,
                            'split_gov_gas': 65},

            'condition_2': {'bot_limit': 1,
                            'top_limit': 1.2,
                            'ctr_oil': 0.27451,
                            'ctr_gas': 0.509804,
                            'gov_oil': 0.72549,
                            'gov_gas': 0.490196,
                            'split_ctr_oil': 17.5,
                            'split_ctr_gas': 32.5,
                            'split_gov_oil': 82.5,
                            'split_gov_gas': 67.5},

            'condition_3': {'bot_limit': 1.2,
                            'top_limit': 1.4,
                            'ctr_oil': 0.235295,
                            'ctr_gas': 0.470589,
                            'gov_oil': 0.764705,
                            'gov_gas': 0.529411,
                            'split_ctr_oil': 15,
                            'split_ctr_gas': 30,
                            'split_gov_oil': 85,
                            'split_gov_gas': 70}
        }

        Returns
        -------
        Modifying the following Cost Recovery attributes:
            self.oil_ctr_pretax_share
            self.gas_ctr_pretax_share
        """

        # Conditioning the given dictionary
        length_condition = len(self.condition_dict[list(self.condition_dict.keys())[0]])
        enum_value = list(range(1, length_condition + 1))
        condition_dict_new = {}
        for index, step in enumerate(enum_value):
            dict_isi = {
                "bot_limit": self.condition_dict["RC Bottom Limit"][index],
                "top_limit": self.condition_dict["RC Top Limit"][index],
                "ctr_oil": self.condition_dict["Pre Tax CTR Oil"][index],
                "ctr_gas": self.condition_dict["Pre Tax CTR Gas"][index],
            }

            key_string = "condition_" + str(step)
            condition_dict_new[key_string] = dict_isi

        self.condition_dict = condition_dict_new

        # Extract relevant values from the condition_dict dictionary
        bot_limits = np.array(
            [self.condition_dict[c]["bot_limit"] for c in self.condition_dict]
        )
        top_limits = np.array(
            [self.condition_dict[c]["top_limit"] for c in self.condition_dict]
        )
        ctr_oil_values = np.array(
            [self.condition_dict[c]["ctr_oil"] for c in self.condition_dict]
        )
        ctr_gas_values = np.array(
            [self.condition_dict[c]["ctr_gas"] for c in self.condition_dict]
        )

        # Create conditions using vectorized comparisons
        conditions = (bot_limits[:, np.newaxis] < self.indicator_rc_icp_sliding) & (
            self.indicator_rc_icp_sliding <= top_limits[:, np.newaxis]
        )

        # Calculate corresponding values using np.select()
        self.oil_ctr_pretax_share = np.select(
            conditions, ctr_oil_values[:, np.newaxis], default=np.nan
        )
        self.gas_ctr_pretax_share = np.select(
            conditions, ctr_gas_values[:, np.newaxis], default=np.nan
        )

    def _get_tax_by_regime(self, tax_regime) -> np.ndarray:
        """
        Determine the tax rate array based on the tax regime and project years.

        This method computes the tax rates for the project years depending on
        the specified `tax_regime`.

        It uses predefined tax configurations for certain years (2013, 2016, 2020)
        and allows for specific tax regimes that override these configurations.

        For tax regimes like `NAILED_DOWN`, the tax rate is fixed based on the
        start year of the project.

        Parameters
        ----------
        tax_regime : TaxRegime
            The tax regime to be applied. It determines how tax rates are selected
            and can be one of the following:
            -   `TaxRegime.UU_07_2021`
            -   `TaxRegime.UU_02_2020`
            -   `TaxRegime.UU_36_2008`
            -   `TaxRegime.NAILED_DOWN`

        Returns
        -------
        tax_rate_arr : np.ndarray
            A 1D array of tax rates for each project year. The tax rate is determined
            based on the tax regime and the project's starting year in relation to
            the predefined tax configurations.
        """
        tax_config = {
            2013: 0.44,
            2016: 0.42,
            2020: 0.40
        }

        tax_rate_arr = np.full_like(
            self.project_years, fill_value=tax_config[min(tax_config)], dtype=float
        )

        for year in tax_config:
            indices = np.array(np.where(self.project_years >= year)).ravel()
            tax_rate_arr[indices] = tax_config[year]

        if tax_regime == TaxRegime.UU_07_2021:
            tax_rate_arr = np.full_like(self.project_years, fill_value=0.40, dtype=float)

        if tax_regime == TaxRegime.UU_02_2020:
            tax_rate_arr = np.full_like(self.project_years, fill_value=0.42, dtype=float)

        if tax_regime == TaxRegime.UU_36_2008:
            tax_rate_arr = np.full_like(self.project_years, fill_value=0.44, dtype=float)

        if tax_regime == TaxRegime.NAILED_DOWN:
            if self.start_date.year >= max(tax_config):
                tax_rate_arr = np.full_like(
                    self.project_years, fill_value=tax_config[max(tax_config)], dtype=float
                )
            else:
                tax_rate_arr = np.full_like(
                    self.project_years, fill_value=tax_config[min(tax_config)], dtype=float
                )

        return tax_rate_arr

    def _get_ic(
        self,
        revenue: np.ndarray,
        ftp: np.ndarray,
        cost_alloc: FluidType,
        ic_rate: float | np.ndarray,
    ) -> tuple:
        """
        The function to apply the Investment Credit (IC) into the capital cost
        based on the given IC rate.

        Parameters
        ----------
        revenue: np.ndarray
            The array of revenue.
        ftp: np.ndarray
            The array of First Tranche Petroleum (FTP).
        cost_alloc: FluidType
            The cost allocation of the cost.
        ic_rate: float | np.ndarray
            The IC rate which will be applied to the cost.

        Returns
        -------
        out: tuple
        ic_total: The total of the IC
        ic_unrecovered: The unrecoverable IC
        ic_paid: The IC that has been paid
        """

        # Condition where fluid is Oil:
        if cost_alloc == FluidType.GAS:
            capital_class = self._gas_capital_postonstream
        else:
            capital_class = self._oil_capital_postonstream

        # Condition when the ic rate is float
        if isinstance(ic_rate, float):
            ic_rate = np.full_like(capital_class.cost, ic_rate)
        else:
            pass

        # Applying the IC calculation to only true value
        ic_arr = np.where(
            np.asarray(capital_class.is_ic_applied),
            capital_class.cost * ic_rate,
            0,
        )

        ic_expenses = np.bincount(
            capital_class.expense_year - capital_class.start_year, weights=ic_arr
        )
        zeros = np.zeros(self.project_duration - len(ic_expenses))
        ic_total = np.concatenate((ic_expenses, zeros))

        # Unrec IC
        ic_unrecovered = np.cumsum(ic_total) - np.cumsum(revenue - ftp)
        ic_unrecovered = np.where(ic_unrecovered > 0, ic_unrecovered, 0)

        ic_unrec_next = np.roll(ic_unrecovered, 1)
        ic_unrec_next[0] = 0
        ic_paid = np.minimum(revenue - ftp, ic_total + ic_unrec_next)

        return ic_total, ic_unrecovered, ic_paid

    @staticmethod
    def _get_cost_recovery(
        revenue: np.ndarray,
        ftp: np.ndarray,
        ic: np.ndarray,
        depreciation: np.ndarray,
        non_capital: np.ndarray,
        cost_to_be_recovered: np.ndarray,
        cr_cap_rate: float,
    ) -> np.ndarray:
        """
        A function to get the array of cost recovery.

        Parameters
        ----------
        revenue: np.ndarray
            The array containing the revenue.
        ftp: np.ndarray
            The array containing the First Tranche Petroleum (FTP).
        ic: np.ndarray
            The array containing the Paid Investment Credit (IC).
        depreciation: np.ndarray
            The array containing the depreciated expenditures.
        non_capital: np.ndarray
            The array containing the non-capital expenditures.
            non_capital expenditures is consisting of:
            Intangible, Operating Expenditures (OPEX) and Abandonment Site
            and Restoration (ASR) Expenditures.
        cost_to_be_recovered: np.ndarray
            The array containing the cost to be recovered.
        cr_cap_rate: float
            The cost recovery cap rate.

        Returns
        -------
        out: np.ndarray
            The array of cost recovery.
        """

        return np.minimum(
            revenue - ftp - ic,
            ((depreciation + non_capital + cost_to_be_recovered) * cr_cap_rate),
        )
        # There is possibility for a bug in Cap rate
        # return (depreciation + non_capital + cost_to_be_recovered) * cr_cap_rate

    @staticmethod
    def _get_ets_before_transfer(
        revenue: np.ndarray,
        ftp_ctr: np.ndarray,
        ftp_gov: np.ndarray,
        ic: np.ndarray,
        cost_recovery: np.ndarray,
    ) -> np.ndarray:
        """
        A function to get the Equity To be Shared (ETS) before transfer.

        Parameters
        ----------
        revenue: np.ndarray
            The array containing the revenue.
        ftp_ctr: np.ndarray
            The array containing the Contractor's First Tranche Petroleum (FTP).
        ftp_gov: np.ndarray
            The array containing the Government's FTP.
        ic: np.ndarray
            The array containing the Paid Investment Credit (IC).
        cost_recovery
            The array containing the cost recovery.

        Returns
        -------
        out: np.ndarray
            The array of ETS before transfer.
        """

        result = revenue - (ftp_ctr + ftp_gov) - ic - cost_recovery
        tol = np.full_like(result, fill_value=1.0e-5)

        return np.where(result < tol, 0, result)

    @staticmethod
    def _get_ets_after_transfer(
        ets_before_tf: np.ndarray,
        revenue: np.ndarray,
        ftp_ctr: np.ndarray,
        ftp_gov: np.ndarray,
        ic: np.ndarray,
        cost_recovery: np.ndarray,
        transferred_in: np.ndarray,
        transferred_out: np.ndarray,
    ) -> np.ndarray:
        """
        A function to get the Equity To be Shared (ETS) before transfer.

        Parameters
        ----------
        revenue: np.ndarray
            The array containing the revenue.
        ftp_ctr: np.ndarray
            The array containing the Contractor's First Tranche Petroleum (FTP).
        ftp_gov: np.ndarray
            The array containing the Government's FTP.
        ic: np.ndarray
            The array containing the Paid Investment Credit (IC).
        cost_recovery: np.ndarray
            The array containing the cost recovery.
        transferred_in: np.ndarray
            The transferred cost into the cashflow.
        transferred_out: np.ndarray
            The transferred cost out from the cashflow.

        Returns
        -------
        out: np.ndarray
            The array of ETS before transfer.
        """
        ets_after_transfer = np.where(
            ets_before_tf >= 0,
            revenue
            - (ftp_ctr + ftp_gov)
            - ic
            - cost_recovery
            - transferred_out
            + transferred_in,
            0,
        )

        tol = np.full_like(ets_after_transfer, fill_value=1.0e-5)

        return np.where(ets_after_transfer < tol, 0, ets_after_transfer)

    @staticmethod
    def _get_equity_share(ets, pretax_ctr) -> tuple:
        r"""
        Use to calculate equity share (ES).

        .. math::

            ES_{ctr \, (t)} &= pretax_ctr \times ETS_{(t)} \\
            ES_{gov \, (t)} &= (1 - pretax_ctr) \times ETS_{(t)}

        Parameters
        ----------
        ets : np.ndarray
            equity to be shared
        pretax_ctr: float
            pretax contractor

        Returns
        -------
        out: tup
            * ES_ctr : np.ndarray
                Equity share of contractor
            * ES_gov : np.ndarray
                Equity share of government
        """
        contractor_share = pretax_ctr * ets
        government_share = (1 - pretax_ctr) * ets

        return contractor_share, government_share

    @staticmethod
    def _get_tax_payment(
        ctr_share: np.ndarray,
        taxable_income: np.ndarray,
        tax_rate: np.ndarray,
        ftp_ctr: np.ndarray,
        unrec_cost: np.ndarray,
        ftp_tax_regime: FTPTaxRegime = FTPTaxRegime.PRE_PDJP_20_2017,
    ) -> np.ndarray:
        """
        The function to get the tax payment of the Cost Recovery contract based on
        the chosen regime.

        There are three existing regime, they are :
            - PRE_PDJP_20_2017
            - PDJP_20_2017
            - DIRECT_MODE

        Notes
        ----------
        [1] PRE_PDJP_20_2017
            In general, the tax will be applied after the contractor share is not zero.
            The taxable object that occurs at the time prior of the applied tax time
            will be accumulated and weighted on the time when the tax is applied.

        [2] PDJP_20_2017
            The tax will be applied when the cumulative First Tranche Petroleum (FTP)
            is greater than the unrecoverable cost

        [3] DIRECT_MODE
            The tax will be applied regardless the cumulative FTP, unrecoverable cost
            and Contractor Share condition.

        Parameters
        ----------
        ctr_share: np.ndarray
            The contractor equity to be split
        taxable_income: np.ndarray
            The contractor's Taxable Income (TI)
        tax_rate: float | np.ndarray
            The applied effective tax rate
        ftp_ctr: np.ndarray
            The contractor's First Tranche Petroleum (FTP)
        unrec_cost: np.ndarray
            The unrecoverable cost
        ftp_tax_regime: FTPTaxRegime
            The regime used to calculate the tax payment

        Returns
        -------
        ctr_tax: np.ndarray
            The contractor's tax payment
        """

        # Tax payment prior to PDJP 2017
        if ftp_tax_regime == FTPTaxRegime.PRE_PDJP_20_2017:
            applied_tax = np.where(ctr_share > 0, 1, 0)
            cum_ti = np.cumsum(taxable_income)

            applied_tax_prior = np.concatenate((np.zeros(1), applied_tax))[0:-1]

            ctr_tax = np.where(
                np.logical_and(applied_tax == 1, applied_tax_prior == 0),
                cum_ti * tax_rate,
                np.where(
                    np.logical_and(applied_tax == 1, applied_tax_prior == 1),
                    taxable_income * tax_rate,
                    0,
                ),
            )

        # Tax payment after PDJP 2017
        elif ftp_tax_regime == FTPTaxRegime.PDJP_20_2017:
            # Defining the array used in calculation of ftp tax payment
            ftp_cum_b2 = np.cumsum(ftp_ctr)
            ftp_prior_b3 = np.zeros_like(ftp_ctr, dtype=float)
            ftp_diff_b4 = np.zeros_like(ftp_ctr, dtype=float)
            ftp_considered_b5 = np.zeros_like(ftp_ctr, dtype=float)

            # Looping throughout the ftp array
            for i, value in enumerate(ftp_ctr):
                if i == 0:
                    ftp_prior_b3[i] = 0

                ftp_prior_b3[i] = ftp_prior_b3[i - 1] + ftp_considered_b5[i - 1]

                ftp_diff_b4[i] = np.where(
                    ftp_cum_b2[i] > ftp_prior_b3[i], ftp_cum_b2[i] - ftp_prior_b3[i], 0
                )
                ftp_considered_b5[i] = np.where(
                    ftp_diff_b4[i] > unrec_cost[i], ftp_diff_b4[i] - unrec_cost[i], 0
                )

            # Calculating the ftp tax payment
            ftp_tax_paid = ftp_considered_b5 * tax_rate

            # Calculating the tax from taxable income
            ti_tax = (taxable_income - ftp_ctr) * tax_rate

            # Calculating ctr_tax
            ctr_tax = ftp_tax_paid + ti_tax

        else:
            ctr_tax = taxable_income * tax_rate

        return ctr_tax

    @staticmethod
    def _get_tax_payment_pdjp(
        ftp_ctr: np.ndarray,
        unrec_cost: np.ndarray,
        tax_rate: np.ndarray,
        taxable_income: np.ndarray,
        ctr_share: np.ndarray,
        ddmo: np.ndarray,
    ):
        """
        Calculate the contractor's tax payment and unpaid tax under the PDJP
        (Profit Distribution and Joint Payment) mechanism.

        This method models the tax computation associated with FTP (First Tranche
        Petroleum) and contractor equity in a production sharing contract (PSC)
        setting.

        It sequentially evaluates cumulative FTP, determines recoverable and
        unrecoverable portions, computes tax from FTP and equity shares, and
        iteratively adjusts for unpaid tax carried over between periods.

        Parameters
        ----------
        ftp_ctr : np.ndarray
            Array of contractor's FTP share for each period.
        unrec_cost : np.ndarray
            Array of unrecoverable costs that reduce the taxable FTP portion.
        tax_rate : np.ndarray
            Array of applicable tax rates for each period (expressed as a fraction,
            e.g., 0.3 for 30%).
        taxable_income : np.ndarray
            Array of contractor's taxable income before FTP adjustment.
        ctr_share : np.ndarray
            Array of contractor's equity share of production for each period.
        ddmo : np.ndarray
            Array of domestic market obligation (DMO) deductions or similar
            non-taxable components.

        Returns
        -------
        tax_paid : np.ndarray
            Array of tax amounts actually paid by the contractor in each period,
            after accounting for unpaid tax carried forward and DMO adjustments.
        unpaid_tax : np.ndarray
            Array of unpaid tax amounts to be carried over to subsequent periods.

        Notes
        -----
        The calculation proceeds as follows:
        1.  Compute cumulative FTP (`ftp_cum_b2`) and the portion of FTP considered
            taxable after subtracting unrecoverable costs.
        2.  Calculate tax on FTP (`ftp_tax_paid`) and equity-based tax (`ctr_ets_tax`)
            depending on whether the contractor share is active.
        3.  Aggregate tax from FTP and contractor equity (`ctr_tax`).
        4.  Iteratively determine tax actually paid (`tax_paid`) and unpaid tax
            carried forward (`unpaid_tax`) per period.
        """

        # Defining the array used in calculation of ftp tax payment
        ftp_cum_b2 = np.cumsum(ftp_ctr)
        ftp_prior_b3 = np.zeros_like(ftp_ctr, dtype=float)
        ftp_diff_b4 = np.zeros_like(ftp_ctr, dtype=float)
        ftp_considered_b5 = np.zeros_like(ftp_ctr, dtype=float)

        # Looping throughout the ftp array
        for i, value in enumerate(ftp_ctr):
            if i == 0:
                ftp_prior_b3[i] = 0

            ftp_prior_b3[i] = ftp_prior_b3[i - 1] + ftp_considered_b5[i - 1]

            ftp_diff_b4[i] = np.where(
                ftp_cum_b2[i] > ftp_prior_b3[i], ftp_cum_b2[i] - ftp_prior_b3[i], 0
            )
            ftp_considered_b5[i] = np.where(
                ftp_diff_b4[i] > unrec_cost[i], ftp_diff_b4[i] - unrec_cost[i], 0
            )

        # Calculating the ftp tax payment
        ftp_tax_paid = ftp_considered_b5 * tax_rate

        # Calculating the taxable income without ftp
        taxable_income_wo_ftp = np.where(
            taxable_income - ftp_ctr < 0,
            taxable_income - ftp_ctr + ddmo,
            taxable_income - ftp_ctr,
        )

        # Defining Taxing flag
        applied_tax = np.where(ctr_share > 0, 1, 0)
        cum_ti = np.cumsum(taxable_income_wo_ftp)

        applied_tax_prior = np.concatenate((np.zeros(1), applied_tax))[0:-1]

        ctr_ets_tax = np.where(
            np.logical_and(applied_tax == 1, applied_tax_prior == 0),
            cum_ti * tax_rate,
            np.where(
                np.logical_and(applied_tax == 1, applied_tax_prior == 1),
                taxable_income_wo_ftp * tax_rate,
                0,
            ),
        )

        # Tax from FTP and Contractor Equity
        ctr_tax = ftp_tax_paid + ctr_ets_tax

        # Contractor Share without FTP
        ets_and_ftp_ctr = ctr_share + ftp_ctr

        # Initiating the array of unpaid_tax and tax_paid
        unpaid_tax = np.zeros_like(taxable_income, dtype=float)
        tax_paid = np.zeros_like(taxable_income, dtype=float)

        for index, i in enumerate(taxable_income):
            if index == 0:
                tax_paid[index] = np.where(
                    taxable_income[index] > ctr_tax[index] + 0,
                    ctr_tax[index] + 0,
                    ets_and_ftp_ctr[index] - ddmo[index],
                )
                unpaid_tax[index] = np.where(
                    ctr_tax[index] + 0 > tax_paid[index],
                    ctr_tax[index] + 0 - tax_paid[index],
                    0,
                )
            else:
                tax_paid[index] = np.where(
                    taxable_income[index] > ctr_tax[index] + unpaid_tax[index - 1],
                    ctr_tax[index] + unpaid_tax[index - 1],
                    ets_and_ftp_ctr[index] - ddmo[index],
                )
                unpaid_tax[index] = np.where(
                    ctr_tax[index] + unpaid_tax[index - 1] > tax_paid[index],
                    ctr_tax[index] + unpaid_tax[index - 1] - tax_paid[index],
                    0,
                )

        return tax_paid, unpaid_tax

    @staticmethod
    def _unpaid_and_tax_balance(tax_payment: np.ndarray, ets_ctr: np.ndarray):
        """
        The function to get the contractor's unpaid tax and tax payment.

        Parameters
        ----------
        tax_payment: np.ndarray
            The contractors tax payment
        ets_ctr: np.ndarray
            The contractor's share

        Returns
        -------
        unpaid_tax: np.ndarray
            The unpaid contractor's tax
        ctr_tax: np.ndarray
            The contractor's tax payment
        """

        unpaid_tax = np.zeros_like(ets_ctr, dtype=float)
        unpaid_tax[0] = 0
        ctr_tax = np.zeros_like(ets_ctr, dtype=float)

        for i in range(1, len(unpaid_tax)):
            unpaid_tax[i] = max(0, unpaid_tax[i - 1] + tax_payment[i] - ets_ctr[i])
            ctr_tax[i] = min(ets_ctr[i], unpaid_tax[i - 1] + tax_payment[i])

        return unpaid_tax, ctr_tax

    def _get_consolidated_profiles_cr(self, ftp_tax_regime: FTPTaxRegime) -> None:
        """
        Consolidate oil and gas profiles into combined attributes.

        This method aggregates oil and gas data for lifting, prices, revenue,
        sunk costs, expenditures (pre-tax, post-tax), indirect taxes,
        total expenses, depreciation/amortization, cost recovery logic,
        tax calculations, and cashflows into consolidated project-level
        attributes.

        Notes
        -----
        The consolidated attributes are computed as element-wise sums of the
        corresponding oil and gas attributes. No return value is produced;
        all results are stored in instance variables prefixed with
        ``_consolidated_``.
        """

        # Attributes associated with consolidated lifting
        self._consolidated_lifting = (
            self._oil_lifting.get_lifting_rate_ghv_arr()
            + self._gas_lifting.get_lifting_rate_ghv_arr()
        )
        self._consolidated_wap_price = self._oil_wap_price + self._gas_wap_price
        self._consolidated_revenue = self._oil_revenue + self._gas_revenue

        # Attributes associated with consolidated sunk cost
        self._consolidated_depreciable_sunk_cost = (
            self._oil_depreciable_sunk_cost + self._gas_depreciable_sunk_cost
        )
        self._consolidated_non_depreciable_sunk_cost = (
            self._oil_non_depreciable_sunk_cost + self._gas_non_depreciable_sunk_cost
        )
        self._consolidated_sunk_cost = self._oil_sunk_cost + self._gas_sunk_cost

        # Attributes associated with consolidated preonstream
        self._consolidated_depreciable_preonstream = (
            self._oil_depreciable_preonstream + self._gas_depreciable_preonstream
        )
        self._consolidated_non_depreciable_preonstream = (
            self._oil_non_depreciable_preonstream + self._gas_non_depreciable_preonstream
        )
        self._consolidated_preonstream = self._oil_preonstream + self._gas_preonstream

        # Attributes associated with consolidated postonstream
        self._consolidated_depreciable_postonstream = (
            self._oil_depreciable_postonstream + self._gas_depreciable_postonstream
        )
        self._consolidated_non_depreciable_postonstream = (
            self._oil_non_depreciable_postonstream + self._gas_non_depreciable_postonstream
        )
        self._consolidated_postonstream = self._oil_postonstream + self._gas_postonstream

        categories = [
            "capital",
            "intangible",
            "opex",
            "asr",
            "lbt",
            "cost_of_sales",
        ]

        # Attributes associated with consolidated expenditures pre tax,
        # consolidated indirect tax, and consolidated expenditures post tax
        for categ in categories:
            oil_pre_tax = getattr(self, f"_oil_{categ}_expenditures_pre_tax")
            gas_pre_tax = getattr(self, f"_gas_{categ}_expenditures_pre_tax")
            oil_indirect_tax = getattr(self, f"_oil_{categ}_indirect_tax")
            gas_indirect_tax = getattr(self, f"_gas_{categ}_indirect_tax")
            oil_post_tax = getattr(self, f"_oil_{categ}_expenditures_post_tax")
            gas_post_tax = getattr(self, f"_gas_{categ}_expenditures_post_tax")

            # Set attributes associated with consolidated expenditures pre tax
            setattr(
                self,
                f"_consolidated_{categ}_expenditures_pre_tax",
                oil_pre_tax + gas_pre_tax
            )

            # Set attributes associated with consolidated indirect taxes
            setattr(
                self,
                f"_consolidated_{categ}_indirect_tax",
                oil_indirect_tax + gas_indirect_tax
            )

            # Set attributes associated with consolidated expenditures post tax
            setattr(
                self,
                f"_consolidated_{categ}_expenditures_post_tax",
                oil_post_tax + gas_post_tax
            )

        self._consolidated_total_expenditures_pre_tax = (
            self._oil_total_expenditures_pre_tax + self._gas_total_expenditures_pre_tax
        )

        self._consolidated_total_indirect_tax = (
            self._oil_total_indirect_tax + self._gas_total_indirect_tax
        )

        self._consolidated_total_expenditures_post_tax = (
            self._oil_total_expenditures_post_tax + self._gas_total_expenditures_post_tax
        )

        # Attributes associated with carry forward depreciation
        self._consolidated_carry_forward_depreciation = (
            self._oil_carry_forward_depreciation + self._gas_carry_forward_depreciation
        )

        # Attributes associated with depreciations and undepreciated assets
        cost_types = ["sunk_cost", "preonstream", "postonstream"]

        self._consolidated_depreciations = {
            c: self._oil_depreciations[c] + self._gas_depreciations[c]
            for c in cost_types
        }

        # self._consolidated_undepreciated_assets = {
        #     c: self._oil_undepreciated_assets[c] + self._gas_undepreciated_assets[c]
        #     for c in cost_types
        # }

        self._consolidated_non_depreciables = {
            c: self._oil_non_depreciables[c] + self._gas_non_depreciables[c]
            for c in cost_types
        }

        self._consolidated_sum_undepreciated_asset = self._oil_sum_undepreciated_asset \
            + self._gas_sum_undepreciated_asset

        self._consolidated_depreciation = self._oil_depreciation + self._gas_depreciation

        # Attributes associated with total depreciation and total non depreciable
        self._consolidated_depr_total = self._oil_depr_total + self._gas_depr_total
        self._consolidated_non_depr_total = (
            self._oil_non_depr_total + self._gas_non_depr_total
        )

        # Attributes associated with core business logics
        self._consolidated_ftp = self._oil_ftp + self._gas_ftp
        self._consolidated_ftp_ctr = self._oil_ftp_ctr + self._gas_ftp_ctr
        self._consolidated_ftp_gov = self._oil_ftp_gov + self._gas_ftp_gov

        self._consolidated_ic = self._oil_ic + self._gas_ic
        self._consolidated_ic_unrecovered = (
            self._oil_ic_unrecovered + self._gas_ic_unrecovered
        )
        self._consolidated_ic_paid = self._oil_ic_paid + self._gas_ic_paid

#        self._consolidated_cost_to_be_recovered_before_tf = (
#            self._oil_cost_to_be_recovered + self._gas_cost_to_be_recovered
#        )
        self._consolidated_recoverable_cost = (
            self._oil_recoverable_cost + self._gas_recoverable_cost
        )
        self._consolidated_revenue_allocation = (
            self._oil_revenue_allocation + self._gas_revenue_allocation
        )

        self._consolidated_unrecovered_before_transfer = (
            self._oil_unrecovered_before_transfer + self._gas_unrecovered_before_transfer
        )
        self._consolidated_cost_recovery_before_transfer = (
            self._oil_cost_recovery + self._gas_cost_recovery
        )
        self._consolidated_ets_before_transfer = (
            self._oil_ets_before_transfer + self._gas_ets_before_transfer
        )

        # self._consolidated_cost_to_be_recovered_after_tf = (
        #    self._oil_cost_to_be_recovered_after_tf
        #    + self._gas_cost_to_be_recovered_after_tf
        # )
        self._consolidated_unrecovered_after_transfer = (
            self._oil_unrecovered_after_transfer + self._gas_unrecovered_after_transfer
        )
        self._consolidated_cost_recovery_after_tf = (
            self._oil_cost_recovery_after_tf + self._gas_cost_recovery_after_tf
        )
        self._consolidated_ets_after_transfer = (
            self._oil_ets_after_transfer + self._gas_ets_after_transfer
        )

        self._consolidated_contractor_share = (
            self._oil_contractor_share + self._gas_contractor_share
        )
        self._consolidated_government_share = (
            self._oil_government_share + self._gas_government_share
        )

        # Attributes associated with DMO
        self._consolidated_dmo_volume = self._oil_dmo_volume + self._gas_dmo_volume
        self._consolidated_dmo_fee = self._oil_dmo_fee + self._gas_dmo_fee
        self._consolidated_ddmo = self._oil_ddmo + self._gas_ddmo

        # Attributes associated with tax
        self._consolidated_taxable_income = (
            self._oil_taxable_income + self._gas_taxable_income
        )

        # Calculating consolidated tax based on the ftp tax payment regime
        if ftp_tax_regime is FTPTaxRegime.PDJP_20_2017:
            self._consolidated_tax_payment, self._consolidated_unpaid_tax_balance = (
                self._get_tax_payment_pdjp(
                    ftp_ctr=self._oil_ftp_ctr + self._gas_ftp_ctr,
                    unrec_cost=(
                        self._oil_unrecovered_after_transfer
                        + self._gas_unrecovered_after_transfer
                    ),
                    tax_rate=self._tax_rate_arr,
                    taxable_income=self._consolidated_taxable_income,
                    ctr_share=self._consolidated_contractor_share,
                    ddmo=self._consolidated_ddmo,
                )
            )

        elif ftp_tax_regime is FTPTaxRegime.PRE_PDJP_20_2017:
            self._consolidated_tax_due = self._get_tax_payment(
                ctr_share=self._consolidated_contractor_share,
                taxable_income=self._consolidated_taxable_income,
                tax_rate=self._tax_rate_arr,
                ftp_ctr=self._consolidated_ftp_ctr,
                unrec_cost=self._consolidated_unrecovered_after_transfer,
                ftp_tax_regime=ftp_tax_regime,
            )

            (
                self._consolidated_unpaid_tax_balance,
                self._consolidated_tax_payment,
            ) = self._unpaid_and_tax_balance(
                tax_payment=self._consolidated_tax_due,
                ets_ctr=self._consolidated_contractor_share,
            )

        else:
            self._consolidated_tax_payment = self._oil_tax_payment + self._gas_tax_payment

        # Attributes associated with consolidated shares
        self._consolidated_ctr_net_share = (
            self._consolidated_taxable_income - self._consolidated_tax_payment
        )
        self._consolidated_contractor_take = (
            self._consolidated_taxable_income
            - self._consolidated_tax_payment
            + self._consolidated_cost_recovery_after_tf
        )
        self._consolidated_government_take = (
            self._consolidated_ftp_gov
            + self._consolidated_government_share
            + self._consolidated_tax_payment
            + self._consolidated_ddmo
        )

        # Attributes associated with consolidated investments
        self._consolidated_capital = self._oil_capital + self._gas_capital
        self._consolidated_non_capital = self._oil_non_capital + self._gas_non_capital
        self._consolidated_total_expenses = self._oil_total_expenses + self._gas_total_expenses

        # Attribute associated with consolidated cashflow
        self._consolidated_cashflow = self._oil_cashflow + self._gas_cashflow

        # ==============================================================
        self._consolidated_opex = (
            self._oil_opex_expenditures_post_tax + self._gas_opex_expenditures_post_tax
        )
        self._consolidated_asr = (
            self._oil_asr_expenditures_post_tax + self._gas_asr_expenditures_post_tax
        )
        self._consolidated_lbt = (
            self._oil_lbt_expenditures_post_tax + self._gas_lbt_expenditures_post_tax
        )
        self._consolidated_non_capital = self._oil_non_capital + self._gas_non_capital

    def run(
        self,
        sulfur_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_GAS_REVENUE,
        electricity_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_OIL_REVENUE,
        co2_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_GAS_REVENUE,
        vat_rate: np.ndarray | float = 0.0,
        year_inflation: np.ndarray = None,
        inflation_rate: np.ndarray | float = 0.0,
        inflation_rate_applied_to: InflationAppliedTo | None = InflationAppliedTo.CAPEX,
        is_dmo_end_weighted: bool = False,
        tax_regime: TaxRegime = TaxRegime.NAILED_DOWN,
        effective_tax_rate: float | np.ndarray | None = None,
        ftp_tax_regime=FTPTaxRegime.PDJP_20_2017,
        depr_method: DeprMethod = DeprMethod.PSC_DB,
        decline_factor: float | int = 2,
        post_uu_22_year2001: bool = True,
        oil_cost_of_sales_applied: bool = False,
        gas_cost_of_sales_applied: bool = False,
        sum_undepreciated_cost: bool = False,
    ):
        """
        Execute the Cost Recovery contract cashflow calculation.

        Runs the full PSC cost recovery workflow: expenditures, depreciation,
        FTP, cost recovery and transfer, equity split, DMO, taxation, and
        contractor/government take. Results are stored as internal time-series
        attributes on the contract instance.

        Parameters
        ----------
        sulfur_revenue : OtherRevenue, optional
            Treatment of sulfur revenue.
        electricity_revenue : OtherRevenue, optional
            Treatment of electricity revenue.
        co2_revenue : OtherRevenue, optional
            Treatment of CO₂ revenue.
        vat_rate : float or np.ndarray, optional
            VAT rate applied to expenditures.
        year_inflation : np.ndarray, optional
            Year index for inflation application.
        inflation_rate : float or np.ndarray, optional
            Inflation rate.
        inflation_rate_applied_to : InflationAppliedTo or None, optional
            Cost components affected by inflation.
        is_dmo_end_weighted : bool, optional
            Apply end-weighted DMO calculation.
        tax_regime : TaxRegime, optional
            Applicable income tax regime.
        effective_tax_rate : float, np.ndarray, or None, optional
            Override tax rate (scalar or time series).
        ftp_tax_regime : FTPTaxRegime, optional
            FTP taxation regime.
        depr_method : DeprMethod, optional
            Depreciation method.
        decline_factor : float or int, optional
            Declining-balance factor (if applicable).
        post_uu_22_year2001 : bool, optional
            Apply post–UU No.22/2001 DMO rules.
        oil_cost_of_sales_applied : bool, optional
            Apply cost of sales to oil revenue.
        gas_cost_of_sales_applied : bool, optional
            Apply cost of sales to gas revenue.
        sum_undepreciated_cost : bool, optional
            Aggregate undepreciated assets across cost categories.

        Returns
        -------
        None
            Results are stored on the object as internal attributes.
        """

        # Calculate WAP (Weighted Average Price) for each produced fluid
        self._get_wap_price()

        # Validate sunk cost, pre-onstream, and post-onstream objects
        self._get_cost_objects_validation()

        # Calculate pre tax expenditures
        self._get_expenditures_pre_tax(
            year_inflation=year_inflation,
            inflation_rate=inflation_rate,
            inflation_rate_applied_to=inflation_rate_applied_to,
        )

        # Calculate indirect taxes
        self._get_indirect_taxes(tax_rate=vat_rate)

        # Calculate post tax expenditures
        self._get_expenditures_post_tax()

        # Calculate other revenue
        self._get_other_revenue(
            sulfur_revenue=sulfur_revenue,
            electricity_revenue=electricity_revenue,
            co2_revenue=co2_revenue,
        )

        # Total pre-tax expenditures
        self._oil_total_expenditures_pre_tax = (
            self._oil_capital_expenditures_pre_tax
            + self._oil_intangible_expenditures_pre_tax
            + self._oil_opex_expenditures_pre_tax
            + self._oil_asr_expenditures_pre_tax
            + self._oil_lbt_expenditures_pre_tax
            + self._oil_cost_of_sales_expenditures_pre_tax
        )

        self._gas_total_expenditures_pre_tax = (
            self._gas_capital_expenditures_pre_tax
            + self._gas_intangible_expenditures_pre_tax
            + self._gas_opex_expenditures_pre_tax
            + self._gas_asr_expenditures_pre_tax
            + self._gas_lbt_expenditures_pre_tax
            + self._gas_cost_of_sales_expenditures_pre_tax
        )

        # Total indirect taxes
        self._oil_total_indirect_tax = (
            self._oil_capital_indirect_tax
            + self._oil_intangible_indirect_tax
            + self._oil_opex_indirect_tax
            + self._oil_asr_indirect_tax
            + self._oil_lbt_indirect_tax
            + self._oil_cost_of_sales_indirect_tax
        )

        self._gas_total_indirect_tax = (
            self._gas_capital_indirect_tax
            + self._gas_intangible_indirect_tax
            + self._gas_opex_indirect_tax
            + self._gas_asr_indirect_tax
            + self._gas_lbt_indirect_tax
            + self._gas_cost_of_sales_indirect_tax
        )

        # Total expenditures post tax
        self._oil_total_expenditures_post_tax = (
            self._oil_capital_expenditures_post_tax
            + self._oil_intangible_expenditures_post_tax
            + self._oil_opex_expenditures_post_tax
            + self._oil_asr_expenditures_post_tax
            + self._oil_lbt_expenditures_post_tax
            + self._oil_cost_of_sales_expenditures_post_tax
        )

        self._gas_total_expenditures_post_tax = (
            self._gas_capital_expenditures_post_tax
            + self._gas_intangible_expenditures_post_tax
            + self._gas_opex_expenditures_post_tax
            + self._gas_asr_expenditures_post_tax
            + self._gas_lbt_expenditures_post_tax
            + self._gas_cost_of_sales_expenditures_post_tax
        )

        # Prepare sunk costs, preonstream costs, and postonstream costs
        self._get_sunkcost_array()
        self._get_preonstream_array()
        self._get_postonstream_array()

        # Calculate depreciations and undepreciated assets for capital
        # sunk costs, preonstream costs, and postonstream costs
        self._get_depreciation(
            depr_method=depr_method,
            decline_factor=decline_factor,
            year_inflation=year_inflation,
            inflation_rate=inflation_rate,
            tax_rate=vat_rate,
            inflation_rate_applied_to=inflation_rate_applied_to,
        )

        # Process non-depreciable costs for non-capital sunk costs,
        # non-capital preonstream costs, and non-capital postonstream costs
        self._get_non_depreciables()

        # Modify depreciations, accounting for various adjustments
        self._get_modified_depreciations(sum_undepreciated_cost=sum_undepreciated_cost)

        # Summation attributes: "sunk_cost" + "preonstream" + "postonstream",
        # for depreciations and undepreciated_assets
        self._oil_sum_undepreciated_asset = np.sum(
            [np.sum(v) for v in self._oil_undepreciated_assets.values()]
        )
        self._gas_sum_undepreciated_asset = np.sum(
            [np.sum(v) for v in self._gas_undepreciated_assets.values()]
        )

        self._oil_depreciation = np.sum([v for v in self._oil_depreciations.values()], axis=0)
        self._gas_depreciation = np.sum([v for v in self._gas_depreciations.values()], axis=0)

        # Calculate FTP
        self._get_ftp()

        # Condition when Cost of Sales for Oil is being applied,
        # which will modify oil or gas revenue
        self._apply_cost_of_sales(
            oil_applied=oil_cost_of_sales_applied, gas_applied=gas_cost_of_sales_applied
        )

        # Defining the PreTax Split, whether using conventional PreTax or Sliding
        if self.tax_split_type is not TaxSplitTypeCR.CONVENTIONAL:
            self._get_rc_icp_pretax()

        # Calculate capital and non-capital investments
        self._get_investments()

        # Investment credit
        self._oil_ic, self._oil_ic_unrecovered, self._oil_ic_paid = self._get_ic(
            revenue=self._oil_revenue,
            ftp=self._oil_ftp,
            cost_alloc=FluidType.OIL,
            ic_rate=self.oil_ic_rate,
        )

        self._gas_ic, self._gas_ic_unrecovered, self._gas_ic_paid = self._get_ic(
            revenue=self._gas_revenue,
            ftp=self._gas_ftp,
            cost_alloc=FluidType.GAS,
            ic_rate=self.gas_ic_rate,
        )

        # Configure total capital cost in the form of depreciations for OIL and GAS
        self._oil_depr_total = (
            np.sum([v for v in self._oil_depreciations.values()], axis=0)
            + self._oil_carry_forward_depreciation
        )

        self._gas_depr_total = (
            np.sum([v for v in self._gas_depreciations.values()], axis=0)
            + self._gas_carry_forward_depreciation
        )

        # Configure total non-capital cost in the form of non-depreciables for OIL and GAS
        self._oil_non_depr_total = np.sum(
            [v for v in self._oil_non_depreciables.values()], axis=0
        )
        self._gas_non_depr_total = np.sum(
            [v for v in self._gas_non_depreciables.values()], axis=0
        )

        # ===========================================================================================

        """
        # Calculate three parameters before transfer: "cost to be recovered",
        # "unrecovered cost before transfer", and "cost recovery"
        (
            self._oil_unrecovered_before_transfer,
            self._oil_cost_to_be_recovered,
            self._oil_cost_recovery,
        ) = psc_tools.get_unrec_cost_2b_recovered_costrec(
            project_years=self.project_years,
            depreciation=self._oil_depr_total,
            non_capital=self._oil_non_depr_total,
            # depreciation=self._oil_depreciation,
            # non_capital=self._oil_non_capital,
            revenue=self._oil_revenue,
            ftp_ctr=self._oil_ftp_ctr,
            ftp_gov=self._oil_ftp_gov,
            ic=self._oil_ic_paid,
            cr_cap_rate=self.oil_cr_cap_rate,
        )

        (
            self._gas_unrecovered_before_transfer,
            self._gas_cost_to_be_recovered,
            self._gas_cost_recovery,
        ) = psc_tools.get_unrec_cost_2b_recovered_costrec(
            project_years=self.project_years,
            depreciation=self._gas_depr_total,
            non_capital=self._gas_non_depr_total,
            # depreciation=self._gas_depreciation,
            # non_capital=self._gas_non_capital,
            revenue=self._gas_revenue,
            ftp_ctr=self._gas_ftp_ctr,
            ftp_gov=self._gas_ftp_gov,
            ic=self._gas_ic_paid,
            cr_cap_rate=self.gas_cr_cap_rate,
        )

        # ETS (Equity to be Split) before transfer/consolidation
        self._oil_ets_before_transfer = self._get_ets_before_transfer(
            revenue=self._oil_revenue,
            ftp_ctr=self._oil_ftp_ctr,
            ftp_gov=self._oil_ftp_gov,
            ic=self._oil_ic_paid,
            cost_recovery=self._oil_cost_recovery,
        )

        self._gas_ets_before_transfer = self._get_ets_before_transfer(
            revenue=self._gas_revenue,
            ftp_ctr=self._gas_ftp_ctr,
            ftp_gov=self._gas_ftp_gov,
            ic=self._gas_ic_paid,
            cost_recovery=self._gas_cost_recovery,
        )

        # Calculate transfer
        self._transfer_to_oil, self._transfer_to_gas = psc_tools.get_transfer(
            gas_unrecovered=self._gas_unrecovered_before_transfer,
            oil_unrecovered=self._oil_unrecovered_before_transfer,
            gas_ets_pretransfer=self._gas_ets_before_transfer,
            oil_ets_pretransfer=self._oil_ets_before_transfer,
        )

        # Re-calculate three parameters after transfer: "cost to be recovered",
        # "unrecovered cost before transfer", and "cost recovery"
        (
            self._oil_unrecovered_after_transfer,
            self._oil_cost_to_be_recovered_after_tf,
            self._oil_cost_recovery_after_tf,
        ) = psc_tools.get_unrec_cost_2b_recovered_costrec(
            project_years=self.project_years,
            depreciation=self._oil_depr_total,
            non_capital=self._oil_non_depr_total,
            # depreciation=self._oil_depreciation,
            # non_capital=self._oil_non_capital,
            revenue=self._oil_revenue + self._transfer_to_oil,
            ftp_ctr=self._oil_ftp_ctr,
            ftp_gov=self._oil_ftp_gov,
            ic=self._oil_ic_paid,
            cr_cap_rate=self.oil_cr_cap_rate,
        )

        (
            self._gas_unrecovered_after_transfer,
            self._gas_cost_to_be_recovered_after_tf,
            self._gas_cost_recovery_after_tf,
        ) = psc_tools.get_unrec_cost_2b_recovered_costrec(
            project_years=self.project_years,
            depreciation=self._gas_depr_total,
            non_capital=self._gas_non_depr_total,
            # depreciation=self._gas_depreciation,
            # non_capital=self._gas_non_capital,
            revenue=self._gas_revenue + self._transfer_to_gas,
            ftp_ctr=self._gas_ftp_ctr,
            ftp_gov=self._gas_ftp_gov,
            ic=self._gas_ic_paid,
            cr_cap_rate=self.gas_cr_cap_rate,
        )

        # ETS after transfer
        self._oil_ets_after_transfer = self._get_ets_after_transfer(
            ets_before_tf=self._oil_ets_before_transfer,
            revenue=self._oil_revenue,
            ftp_ctr=self._oil_ftp_ctr,
            ftp_gov=self._oil_ftp_gov,
            ic=self._oil_ic_paid,
            cost_recovery=self._oil_cost_recovery_after_tf,
            transferred_in=self._transfer_to_oil,
            transferred_out=self._transfer_to_gas,
        )

        self._gas_ets_after_transfer = self._get_ets_after_transfer(
            ets_before_tf=self._gas_ets_before_transfer,
            revenue=self._gas_revenue,
            ftp_ctr=self._gas_ftp_ctr,
            ftp_gov=self._gas_ftp_gov,
            ic=self._gas_ic_paid,
            cost_recovery=self._gas_cost_recovery_after_tf,
            transferred_in=self._transfer_to_gas,
            transferred_out=self._transfer_to_oil,
        )

        cols1 = [#'revenue', 'ftp', 'ic_paid',
            'allocation', 'non_capital', 'depreciation',
                'tobe_rec_pre','cost_rec_pre','unrec_pre', 'ets_pre',
                'trf_to_og','tobe_rec_post','cost_rec_post','unrec_post','ets_post']
        fluids = ['oil','gas']

        cr1_df = pd.DataFrame(index=self.project_years,
                              columns=pd.MultiIndex.from_product([cols1,fluids]))
        # cr1_df[('revenue','oil')] = self._oil_revenue
        # cr1_df[('revenue','gas')] = self._gas_revenue
        # cr1_df[('ftp','oil')] = self._oil_ftp
        # cr1_df[('ftp','gas')] = self._gas_ftp
        # cr1_df[('ic_paid','oil')] = self._oil_ic_paid
        # cr1_df[('ic_paid','gas')] = self._gas_ic_paid
        cr1_df[('non_capital','oil')] = self._oil_non_depr_total
        cr1_df[('non_capital','gas')] = self._gas_non_depr_total
        cr1_df[('depreciation','oil')] = self._oil_depr_total
        cr1_df[('depreciation','gas')] = self._gas_depr_total
        # cr1_df[('allocation','oil')] = (
        #     cr1_df[('revenue', 'oil')] - cr1_df[('ftp', 'oil')] - cr1_df[('ic_paid', 'oil')]
        # )
        # cr1_df[('allocation','gas')] = (
        #     cr1_df[('revenue', 'gas')] - cr1_df[('ftp', 'gas')] - cr1_df[('ic_paid', 'gas')]
        # )
        cr1_df[('allocation','oil')] = self._oil_revenue - self._oil_ftp - self._oil_ic_paid
        cr1_df[('allocation','gas')] = self._gas_revenue - self._gas_ftp - self._gas_ic_paid
        cr1_df[('tobe_rec_pre','oil')] = self._oil_cost_to_be_recovered
        cr1_df[('tobe_rec_pre','gas')] = self._gas_cost_to_be_recovered
        cr1_df[('cost_rec_pre','oil')] = self._oil_cost_recovery
        cr1_df[('cost_rec_pre','gas')] = self._gas_cost_recovery
        cr1_df[('unrec_pre','oil')] = self._oil_unrecovered_before_transfer
        cr1_df[('unrec_pre','gas')] = self._gas_unrecovered_before_transfer
        cr1_df[('ets_pre','oil')] = self._oil_ets_before_transfer
        cr1_df[('ets_pre','gas')] = self._gas_ets_before_transfer
        cr1_df[('trf_to_og','oil')] = self._transfer_to_oil
        cr1_df[('trf_to_og','gas')] = self._transfer_to_gas
        cr1_df[('tobe_rec_post','oil')] = self._oil_cost_to_be_recovered_after_tf
        cr1_df[('tobe_rec_post','gas')] = self._gas_cost_to_be_recovered_after_tf
        cr1_df[('cost_rec_post','oil')] = self._oil_cost_recovery_after_tf
        cr1_df[('cost_rec_post','gas')] = self._gas_cost_recovery_after_tf
        cr1_df[('unrec_post','oil')] = self._oil_unrecovered_after_transfer
        cr1_df[('unrec_post','gas')] = self._gas_unrecovered_after_transfer
        cr1_df[('ets_post','oil')] = self._oil_ets_after_transfer
        cr1_df[('ets_post','gas')] = self._gas_ets_after_transfer
        """

        # ===========================================================================================

        oil_allocation = self._oil_revenue - self._oil_ftp - self._oil_ic_paid
        gas_allocation = self._gas_revenue - self._gas_ftp - self._gas_ic_paid
        allocation = np.vstack((oil_allocation, gas_allocation)).T
        # non_capital = np.vstack((self._oil_non_capital, self._gas_non_capital)).T
        # depreciation = np.vstack((self._oil_depreciation, self._gas_depreciation)).T
        non_capital = np.vstack((self._oil_non_depr_total, self._gas_non_depr_total)).T
        depreciation = np.vstack((self._oil_depr_total, self._gas_depr_total)).T

        ets_pre_trf = np.zeros_like(allocation)
        ets_post_trf = np.zeros_like(allocation)
        unrec_pre_trf = np.zeros_like(allocation)
        unrec_post_trf = np.zeros_like(allocation)
        rec_pre_trf = np.zeros_like(allocation)
        rec_post_trf = np.zeros_like(allocation)
        recoverable_cost = np.zeros_like(allocation)
        trf_to_og = np.zeros_like(allocation)
        project_duration = len(self.project_years)
        prev_unrec = np.zeros(2)

        for i in range(project_duration):

            recoverable_cost[i] = non_capital[i] + depreciation[i] + prev_unrec
            ets_pre_trf[i] = np.maximum(allocation[i] - recoverable_cost[i], 0)
            unrec_pre_trf[i] = np.maximum(recoverable_cost[i] - allocation[i], 0)
            rec_pre_trf[i] = recoverable_cost[i] - unrec_pre_trf[i]

            trf_to = np.zeros(2)
            ets_after = ets_pre_trf[i].copy()
            unrec_after = unrec_pre_trf[i].copy()
            recov_after = rec_pre_trf[i].copy()

            if ets_pre_trf[i, 0] > 0 and unrec_pre_trf[i, 1] > 0:
                trf_to[1] = np.minimum(ets_pre_trf[i, 0], unrec_pre_trf[i, 1])
                ets_after[0] = ets_pre_trf[i, 0] - trf_to[1]
                unrec_after[1] = unrec_pre_trf[i, 1] - trf_to[1]
                recov_after[1] = rec_pre_trf[i, 1] + trf_to[1]

            elif ets_pre_trf[i, 1] > 0 and unrec_pre_trf[i, 0] > 0:
                trf_to[0] = np.minimum(ets_pre_trf[i, 1], unrec_pre_trf[i, 0])
                ets_after[1] = ets_pre_trf[i, 1] - trf_to[0]
                unrec_after[0] = unrec_pre_trf[i, 0] - trf_to[0]
                recov_after[0] = rec_pre_trf[i, 0] + trf_to[0]

            ets_post_trf[i] = ets_after
            unrec_post_trf[i] = unrec_after
            rec_post_trf[i] = recov_after
            trf_to_og[i] = trf_to

            prev_unrec = unrec_after

        self._oil_revenue_allocation = oil_allocation
        self._gas_revenue_allocation = gas_allocation
        self._oil_recoverable_cost = recoverable_cost[:, 0]
        self._gas_recoverable_cost = recoverable_cost[:, 1]
        self._oil_cost_recovery = rec_pre_trf[:, 0]
        self._gas_cost_recovery = rec_pre_trf[:, 1]
        self._oil_unrecovered_before_transfer = unrec_pre_trf[:, 0]
        self._gas_unrecovered_before_transfer = unrec_pre_trf[:, 1]
        self._oil_cost_recovery_after_tf = rec_post_trf[:, 0]
        self._gas_cost_recovery_after_tf = rec_post_trf[:, 1]
        self._oil_ets_before_transfer = ets_pre_trf[:, 0]
        self._gas_ets_before_transfer = ets_pre_trf[:, 1]

        self._transfer_to_oil = trf_to_og[:, 0]
        self._transfer_to_gas = trf_to_og[:, 1]
        self._oil_cost_recovery_after_tf = rec_post_trf[:, 0]
        self._gas_cost_recovery_after_tf = rec_post_trf[:, 1]
        self._oil_unrecovered_after_transfer = unrec_post_trf[:, 0]
        self._gas_unrecovered_after_transfer = unrec_post_trf[:, 1]
        self._oil_ets_after_transfer = ets_post_trf[:, 0]
        self._gas_ets_after_transfer = ets_post_trf[:, 1]

        # ========================================================================================

        """
        Debugging DataFrame Cost Recovery & Transfer

        cols2 = [#'revenue', 'ftp', 'ic_paid',
            'allocation', 'non_capital', 'depreciation',
                 'recoverable_cost','rec_pre_trf','unrec_pre_trf','ets_pre_trf',
                    'trf_to_og','rec_post_trf','unrec_post_trf','ets_post_trf']

        cr2_df = pd.DataFrame(index=self.project_years,
                              columns=pd.MultiIndex.from_product([cols2,fluids]))
        #cr2_df[('revenue','oil')] = self._oil_revenue
        #cr2_df[('revenue','gas')] = self._gas_revenue
        #cr2_df[('ftp','oil')] = self._oil_ftp
        #cr2_df[('ftp','gas')] = self._gas_ftp
        #cr2_df[('ic_paid','oil')] = self._oil_ic_paid
        #cr2_df[('ic_paid','gas')] = self._gas_ic_paid
        cr2_df[('non_capital','oil')] = self._oil_non_depr_total
        cr2_df[('non_capital','gas')] = self._gas_non_depr_total
        cr2_df[('depreciation','oil')] = self._oil_depr_total
        cr2_df[('depreciation','gas')] = self._gas_depr_total
        cr2_df[('allocation','oil')] = allocation[:,0]
        cr2_df[('allocation','gas')] = allocation[:,1]
        cr2_df[('recoverable_cost','oil')] = recoverable_cost[:,0]
        cr2_df[('recoverable_cost','gas')] = recoverable_cost[:,1]
        cr2_df[('rec_pre_trf','oil')] = rec_pre_trf[:,0]
        cr2_df[('rec_pre_trf','gas')] = rec_pre_trf[:,1]
        cr2_df[('unrec_pre_trf','oil')] = unrec_pre_trf[:,0]
        cr2_df[('unrec_pre_trf','gas')] = unrec_pre_trf[:,1]
        cr2_df[('ets_pre_trf','oil')] = ets_pre_trf[:,0]
        cr2_df[('ets_pre_trf','gas')] = ets_pre_trf[:,1]
        cr2_df[('trf_to_og','oil')] = trf_to_og[:,0]
        cr2_df[('trf_to_og','gas')] = trf_to_og[:,1]
        cr2_df[('rec_post_trf','oil')] = rec_post_trf[:,0]
        cr2_df[('rec_post_trf','gas')] = rec_post_trf[:,1]
        cr2_df[('unrec_post_trf','oil')] = unrec_post_trf[:,0]
        cr2_df[('unrec_post_trf','gas')] = unrec_post_trf[:,1]
        cr2_df[('ets_post_trf','oil')] = ets_post_trf[:,0]
        cr2_df[('ets_post_trf','gas')] = ets_post_trf[:,1]
        """

        # ========================================================================================

        # ES (Equity Share)
        self._oil_contractor_share, self._oil_government_share = self._get_equity_share(
            ets=self._oil_ets_after_transfer, pretax_ctr=self.oil_ctr_pretax_share
        )

        self._gas_contractor_share, self._gas_government_share = self._get_equity_share(
            ets=self._gas_ets_after_transfer, pretax_ctr=self.gas_ctr_pretax_share
        )

        # DMO
        self._oil_dmo_volume, self._oil_dmo_fee, self._oil_ddmo = psc_tools.get_dmo(
            onstream_date=self.oil_onstream_date,
            start_date=self.start_date,
            project_years=self.project_years,
            dmo_holiday_duration=self.oil_dmo_holiday_duration,
            dmo_volume_portion=self.oil_dmo_volume_portion,
            dmo_fee_portion=self.oil_dmo_fee_portion,
            lifting=self._oil_lifting,
            price=self._oil_wap_price,
            ctr_pretax_share=self.oil_ctr_pretax_share,
            unrecovered_cost=self._oil_unrecovered_after_transfer,
            is_dmo_end_weighted=is_dmo_end_weighted,
            ets=self._oil_ets_after_transfer,
            ctr_ets=self._oil_contractor_share,
            ctr_ftp=self._oil_ftp_ctr,
            post_uu_22_year2001=post_uu_22_year2001,
        )

        self._gas_dmo_volume, self._gas_dmo_fee, self._gas_ddmo = psc_tools.get_dmo(
            onstream_date=self.gas_onstream_date,
            start_date=self.start_date,
            project_years=self.project_years,
            dmo_holiday_duration=self.gas_dmo_holiday_duration,
            dmo_volume_portion=self.gas_dmo_volume_portion,
            dmo_fee_portion=self.gas_dmo_fee_portion,
            lifting=self._gas_lifting,
            price=self._gas_wap_price,
            ctr_pretax_share=self.gas_ctr_pretax_share,
            unrecovered_cost=self._gas_unrecovered_after_transfer,
            is_dmo_end_weighted=is_dmo_end_weighted,
            ets=self._gas_ets_after_transfer,
            ctr_ets=self._gas_contractor_share,
            ctr_ftp=self._gas_ftp_ctr,
            post_uu_22_year2001=post_uu_22_year2001,
        )

        # Adjusting DDMO for Pre PDJP
        if ftp_tax_regime is FTPTaxRegime.PRE_PDJP_20_2017:
            self._oil_ddmo = np.where(self._oil_contractor_share > 0, self._oil_ddmo, 0)
            self._gas_ddmo = np.where(self._gas_contractor_share > 0, self._gas_ddmo, 0)

        # Taxable income (also known as Net Contractor Share)
        self._oil_taxable_income = (
            self._oil_ftp_ctr
            + self._oil_contractor_share
            + self._oil_ic_paid
            - self._oil_ddmo
        )

        self._gas_taxable_income = (
            self._gas_ftp_ctr
            + self._gas_contractor_share
            + self._gas_ic_paid
            - self._gas_ddmo
        )

        # Tax Rate
        # Procedure if information about effective tax rate is not given
        if effective_tax_rate is None:
            tax_rate_arr = self._get_tax_by_regime(tax_regime=tax_regime)

        else:
            # If effective tax rate is given as an array
            if isinstance(effective_tax_rate, np.ndarray):
                tax_rate_arr = effective_tax_rate

            # If effective tax rate is given as a single value
            elif isinstance(effective_tax_rate, (float, int)):
                tax_rate_arr = np.full_like(
                    self.project_years, effective_tax_rate, dtype=float
                )
            else:
                tax_rate_arr = np.zeros_like(self.project_years, dtype=float)

        self._tax_rate_arr = tax_rate_arr

        # Tax Payment
        self._oil_tax_payment = self._get_tax_payment(
            ctr_share=self._oil_contractor_share,
            taxable_income=self._oil_taxable_income,
            tax_rate=self._tax_rate_arr,
            ftp_ctr=self._oil_ftp_ctr,
            unrec_cost=self._oil_unrecovered_after_transfer,
            ftp_tax_regime=ftp_tax_regime,
        )

        self._gas_tax_payment = self._get_tax_payment(
            ctr_share=self._gas_contractor_share,
            taxable_income=self._gas_taxable_income,
            tax_rate=self._tax_rate_arr,
            ftp_ctr=self._gas_ftp_ctr,
            unrec_cost=self._gas_unrecovered_after_transfer,
            ftp_tax_regime=ftp_tax_regime,
        )

        # Contractor Share
        self._oil_ctr_net_share = self._oil_taxable_income - self._oil_tax_payment
        self._gas_ctr_net_share = self._gas_taxable_income - self._gas_tax_payment

        # Contractor Take by Fluid
        self._oil_contractor_take = (
            self._oil_taxable_income
            - self._oil_tax_payment
            + self._oil_cost_recovery_after_tf
        )

        self._gas_contractor_take = (
            self._gas_taxable_income
            - self._gas_tax_payment
            + self._gas_cost_recovery_after_tf
        )

        # Contractor cashflow
        self._oil_cashflow = self._oil_contractor_take - self._oil_total_expenses
        self._gas_cashflow = self._gas_contractor_take - self._gas_total_expenses

        # Government Take by Fluid
        self._oil_government_take = (
            self._oil_ftp_gov
            + self._oil_government_share
            + self._oil_tax_payment
            + self._oil_ddmo
        )

        self._gas_government_take = (
            self._gas_ftp_gov
            + self._gas_government_share
            + self._gas_tax_payment
            + self._gas_ddmo
        )

        # Returning the gross revenue
        # self._oil_revenue = (
        #     self._oil_revenue + self._oil_cost_of_sales_expenditures_post_tax
        # )
        # self._gas_revenue = (
        #     self._gas_revenue + self._gas_cost_of_sales_expenditures_post_tax
        # )

        # Prepare consolidated profiles
        self._get_consolidated_profiles_cr(ftp_tax_regime=ftp_tax_regime)

        # Display warning messages as pandas DataFrame
        # self.warnings_to_dataframe()

    def get_summary(
        self,
        discount_rate: float = 0.1,
        npv_mode: NPVSelection = NPVSelection.NPV_SKK_REAL_TERMS,
        discounting_mode: DiscountingMode = DiscountingMode.END_YEAR,
        discount_rate_start_year: int | None = None,
        inflation_rate: np.ndarray | float = 0.0,
        profitability_discounted: bool = False,
    ) -> dict:
        """
        Generate a comprehensive project economic summary under the Cost Recovery
        PSC scheme.

        This method computes key production, cost, tax, and profitability indicators,
        and returns the overall project summary in dictionary form.

        The summary includes metrics such as lifting volumes, gross revenues,
        investment costs, NPV, IRR, PI, and government take.

        Parameters
        ----------
        discount_rate : float, default=0.1
            The discount rate applied for NPV calculations.

        npv_mode : NPVSelection, default=NPVSelection.NPV_SKK_REAL_TERMS
            The NPV calculation mode. Options include:
            - ``NPV_SKK_REAL_TERMS`` : SKK Migas real-term NPV.
            - ``NPV_SKK_NOMINAL_TERMS`` : SKK Migas nominal-term NPV.
            - ``NPV_NOMINAL_TERMS`` : Standard nominal-term NPV.
            - ``NPV_REAL_TERMS`` : Standard real-term NPV.
            - ``NPV_POINT_FORWARD`` : Point-forward NPV.

        discounting_mode : DiscountingMode, default=DiscountingMode.END_YEAR
            The timing convention for discounting cashflows (e.g., start-year or end-year).

        discount_rate_start_year : int or None, optional
            The reference year for discounting. If ``None``, the project start year is used.
            Must lie between project start and end years.

        inflation_rate : float or ndarray, default=0.0
            Inflation rate(s) applied for real-term NPV calculations. Can be a scalar or
            array of yearly inflation rates.

        profitability_discounted : bool, default=False
            If ``True``, the profitability index (PI) is computed using discounted
            investment (NPV-based). If ``False``, the PI uses undiscounted investment.

        Returns
        -------
        summary : dict
            A dictionary containing the key project summary indicators.

        Raises
        ------
        CostRecoverySummaryException
            If `discount_rate_start_year` is before the project start year or
            after the project end year.

        Notes
        -----
        - This method must run the full model first via :meth:`run` before generating
          the summary results.
        - Economic indicators are computed according to SKK Migas evaluation conventions.
        """

        # Prepare discount rate start year
        if discount_rate_start_year is None:
            discount_rate_start_year = self.start_date.year

        # Cannot have discount rate year before the start year of the project
        if discount_rate_start_year < self.start_date.year:
            raise CostRecoverySummaryException(
                f"The discounting reference year ({discount_rate_start_year}) "
                f"is before start year of the project ({self.start_date.year})."
            )

        # Cannot have discount rate year after the end year of the project
        if discount_rate_start_year > self.end_date.year:
            raise CostRecoverySummaryException(
                f"The discounting reference year ({discount_rate_start_year}) "
                f"is after the end year of the project ({self.end_date.year})."
            )

        # Prepare OIL lifting summary
        oil_lifting_ghv = self._oil_lifting.get_lifting_rate_ghv_arr()
        oil_lifting_ghv_sum = np.sum(oil_lifting_ghv, dtype=float)
        oil_wap_sum = self._calc_division(
            numerator=self._oil_revenue.sum(dtype=float), denominator=oil_lifting_ghv_sum
        )

        # Prepare GAS lifting summary
        gas_lifting_ghv = self._gas_lifting.get_lifting_rate_ghv_arr()
        gas_lifting_ghv_sum = np.sum(gas_lifting_ghv, dtype=float)
        gas_wap_sum = self._calc_division(
            numerator=np.sum(self._gas_wap_price * gas_lifting_ghv),
            denominator=gas_lifting_ghv_sum,
        )

        # Prepare gross revenue summary
        oil_gross_revenue_sum = self._oil_revenue.sum(dtype=float)
        gas_gross_revenue_sum = self._gas_revenue.sum(dtype=float)
        total_gross_revenue_sum = oil_gross_revenue_sum + gas_gross_revenue_sum

        # Prepare sunk cost summary
        sunk_cost_sum = np.sum(self._oil_sunk_cost + self._gas_sunk_cost, dtype=float)

        # Prepare preonstream costs
        preonstream_map = {
            "oil_intangible": self._oil_intangible_preonstream,
            "gas_intangible": self._gas_intangible_preonstream,
            "oil_opex": self._oil_opex_preonstream,
            "gas_opex": self._gas_opex_preonstream,
            "oil_asr": self._oil_asr_preonstream,
            "gas_asr": self._gas_asr_preonstream,
            "oil_lbt": self._oil_lbt_preonstream,
            "gas_lbt": self._gas_lbt_preonstream,
        }

        preonstream_costs = {
            key: val.expenditures_pre_tax() for key, val in preonstream_map.items()
        }

        # Prepare tangible and intangible cost summary
        tangible_cost = (
            self._oil_capital_expenditures_post_tax
            + self._gas_capital_expenditures_post_tax
            + self._oil_depreciable_preonstream
            + self._gas_depreciable_preonstream
        )

        intangible_cost = (
            self._oil_intangible_expenditures_post_tax
            + self._gas_intangible_expenditures_post_tax
            + preonstream_costs["oil_intangible"]
            + preonstream_costs["gas_intangible"]
        )

        tangible_sum = tangible_cost.sum(dtype=float)
        intangible_sum = intangible_cost.sum(dtype=float)
        investment_sum = tangible_sum + intangible_sum

        # Prepare OPEX summary
        opex_cost = (
            self._oil_opex_expenditures_post_tax
            + self._gas_opex_expenditures_post_tax
            + preonstream_costs["oil_opex"]
            + preonstream_costs["gas_opex"]
        )

        opex_sum = opex_cost.sum(dtype=float)

        # Prepare ASR summary
        asr_cost = (
            self._oil_asr_expenditures_post_tax
            + self._gas_asr_expenditures_post_tax
            + preonstream_costs["oil_asr"]
            + preonstream_costs["gas_asr"]
        )

        asr_sum = asr_cost.sum(dtype=float)

        # Prepare LBT summary
        lbt_cost = (
            self._oil_lbt_expenditures_post_tax
            + self._gas_lbt_expenditures_post_tax
            + preonstream_costs["oil_lbt"]
            + preonstream_costs["gas_lbt"]
        )

        lbt_sum = lbt_cost.sum(dtype=float)

        # Prepare indirect taxes summary
        oil_indirect_tax_sum = self._oil_total_indirect_tax.sum(dtype=float)
        gas_indirect_tax_sum = self._gas_total_indirect_tax.sum(dtype=float)

        # Prepare carry forward depreciation summary
        oil_carward_depreciation_sum = self._oil_carry_forward_depreciation.sum(dtype=float)
        gas_carward_depreciation_sum = self._gas_carry_forward_depreciation.sum(dtype=float)
        total_carward_depreciation_sum = (
            oil_carward_depreciation_sum + gas_carward_depreciation_sum
        )

        # Prepare undepreciated assets summary
        oil_undepreciated_asset_sum = np.sum(
            [undepr.sum() for undepr in self._oil_undepreciated_assets.values()]
        )
        gas_undepreciated_asset_sum = np.sum(
            [undepr.sum() for undepr in self._gas_undepreciated_assets.values()]
        )
        total_undepreciated_asset_sum = (
            oil_undepreciated_asset_sum + gas_undepreciated_asset_sum
        )

        # Prepare government DDMO summary
        gov_ddmo = self._consolidated_ddmo.sum(dtype=float)

        # Prepare government take summary
        gov_take_income_sum = self._consolidated_tax_payment.sum(dtype=float)
        gov_take_sum = self._consolidated_government_take.sum(dtype=float)
        gov_take_over_gross_rev = self._calc_division(
            numerator=gov_take_sum, denominator=total_gross_revenue_sum
        )

        # Calculate IRR
        ctr_irr = irr(cashflow=self._consolidated_cashflow)

        # Calculate NPV
        # NPV method: SKK real terms
        if npv_mode == NPVSelection.NPV_SKK_REAL_TERMS:

            # Contractor NPV
            ctr_npv = npv_skk_real_terms(
                cashflow=self._consolidated_cashflow,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                discounting_mode=discounting_mode,
            )

            # Contractor investment NPV
            investment_npv = npv_skk_real_terms(
                cashflow=tangible_cost + intangible_cost,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                discounting_mode=discounting_mode,
            )

            # Government take NPV
            gov_take_npv = npv_skk_real_terms(
                cashflow=self._consolidated_government_take,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                discounting_mode=discounting_mode,
            )

        # NPV method: SKK nominal terms
        elif npv_mode == NPVSelection.NPV_SKK_NOMINAL_TERMS:

            # Contractor NPV
            ctr_npv = npv_skk_nominal_terms(
                cashflow=self._consolidated_cashflow,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                discounting_mode=discounting_mode,
            )

            # Contractor investment NPV
            investment_npv = npv_skk_nominal_terms(
                cashflow=tangible_cost + intangible_cost,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                discounting_mode=discounting_mode,
            )

            # Government take NPV
            gov_take_npv = npv_skk_nominal_terms(
                cashflow=self._consolidated_government_take,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                discounting_mode=discounting_mode,
            )

        # NPV method: nominal terms
        elif npv_mode == NPVSelection.NPV_NOMINAL_TERMS:

            # Contractor NPV
            ctr_npv = npv_nominal_terms(
                cashflow=self._consolidated_cashflow,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                discounting_mode=discounting_mode,
            )

            # Contractor investment NPV
            investment_npv = npv_nominal_terms(
                cashflow=tangible_cost + intangible_cost,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                discounting_mode=discounting_mode,
            )

            # Government take NPV
            gov_take_npv = npv_nominal_terms(
                cashflow=self._consolidated_government_take,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                discounting_mode=discounting_mode,
            )

        # NPV method: real terms
        elif npv_mode == NPVSelection.NPV_REAL_TERMS:

            # Contractor NPV
            ctr_npv = npv_real_terms(
                cashflow=self._consolidated_cashflow,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                inflation_rate=inflation_rate,
                discounting_mode=discounting_mode,
            )

            # Contractor investment NPV
            investment_npv = npv_real_terms(
                cashflow=tangible_cost + intangible_cost,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                inflation_rate=inflation_rate,
                discounting_mode=discounting_mode,
            )

            # Government take NPV
            gov_take_npv = npv_real_terms(
                cashflow=self._consolidated_government_take,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                inflation_rate=inflation_rate,
                discounting_mode=discounting_mode,
            )

        # NPV method: point forward
        else:
            # Contractor NPV
            ctr_npv = npv_point_forward(
                cashflow=self._consolidated_cashflow,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                discounting_mode=discounting_mode,
            )

            # Contractor investment NPV
            investment_npv = npv_point_forward(
                cashflow=tangible_cost + intangible_cost,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                discounting_mode=discounting_mode,
            )

            # Government take NPV
            gov_take_npv = npv_point_forward(
                cashflow=self._consolidated_government_take,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                discounting_mode=discounting_mode,
            )

        # Contractor present value ratio to the investment NPV
        # Profitability Index is calculated using discounted investment
        if profitability_discounted:
            ctr_pv_ratio = self._calc_division(numerator=ctr_npv, denominator=investment_npv)

        else:
            investment_pi = np.sum(tangible_cost + intangible_cost)
            ctr_pv_ratio = self._calc_division(numerator=ctr_npv, denominator=investment_pi)

        ctr_pi = 1 + ctr_pv_ratio

        # Contractor POT
        ctr_pot = pot_psc(
            cashflow=self._consolidated_cashflow,
            cashflow_years=self.project_years,
            reference_year=discount_rate_start_year,
        )

        # Prepare government FTP share summary
        gov_ftp_share_sum = self._consolidated_ftp_gov.sum(dtype=float)

        # Prepare government equity share summary
        gov_equity_share_sum = self._consolidated_government_share.sum(dtype=float)

        # Prepare cost recovery summary
        cost_recovery_sum = self._consolidated_cost_recovery_after_tf.sum(dtype=float)
        cost_recovery_over_gross_rev = self._calc_division(
            numerator=cost_recovery_sum, denominator=total_gross_revenue_sum
        )

        # Prepare unrecoverable cost summary
        unrec_cost = self._consolidated_unrecovered_after_transfer[-1]
        unrec_over_costrec = self._calc_division(
            numerator=unrec_cost, denominator=cost_recovery_sum
        )
        unrec_over_gross_rev = self._calc_division(
            numerator=unrec_cost, denominator=total_gross_revenue_sum
        )

        # Prepare contractor gross share summary
        ctr_gross_share_sum = self._consolidated_contractor_share.sum(dtype=float)

        # Prepare government gross share summary
        gov_gross_share_sum = self._consolidated_government_share.sum(dtype=float)

        # Prepare contractor net share summary
        ctr_net_share_sum = self._consolidated_ctr_net_share.sum(dtype=float)
        ctr_net_share_over_gross_share = self._calc_division(
            numerator=ctr_net_share_sum, denominator=total_gross_revenue_sum
        )

        # Prepare contractor net cashflow summary
        # For NPV POINT FORWARD
        if npv_mode == NPVSelection.NPV_POINT_FORWARD:
            # Modify contractor net cashflow since the cashflow before discount rate
            # start year is neglected
            ref_year_arr = np.full(
                self._consolidated_cashflow.shape, fill_value=discount_rate_start_year
            )
            cashflow_point_forward = np.where(
                self.project_years >= ref_year_arr,
                self._consolidated_cashflow,
                0,
            )
            gross_revenue_point_forward = np.where(
                self.project_years >= ref_year_arr,
                self._consolidated_cashflow,
                0,
            )

            # Contractor net cashflow
            ctr_net_cashflow_sum = cashflow_point_forward.sum(dtype=float)
            ctr_net_cashflow_over_gross_rev = self._calc_division(
                numerator=ctr_net_cashflow_sum,
                denominator=gross_revenue_point_forward.sum(dtype=float),
            )

        # For other NPV calculation methods
        else:
            # Contractor net cashflow
            ctr_net_cashflow_sum = self._consolidated_cashflow.sum(dtype=float)
            ctr_net_cashflow_over_gross_rev = self._calc_division(
                numerator=ctr_net_cashflow_sum, denominator=total_gross_revenue_sum
            )

        return {
            "lifting_oil": oil_lifting_ghv_sum,
            "oil_wap": oil_wap_sum,
            "lifting_gas": gas_lifting_ghv_sum,
            "gas_wap": gas_wap_sum,
            "gross_revenue": total_gross_revenue_sum,
            "gross_revenue_oil": oil_gross_revenue_sum,
            "gross_revenue_gas": gas_gross_revenue_sum,
            "ctr_gross_share": ctr_gross_share_sum,
            "gov_gross_share": gov_gross_share_sum,
            "investment": investment_sum,
            "oil_capex": self._oil_capital.sum(),
            "gas_capex": self._gas_capital.sum(),
            "sunk_cost": sunk_cost_sum,
            "tangible": tangible_sum,
            "intangible": intangible_sum,
            "opex_asr_lbt": opex_sum + asr_sum + lbt_sum,
            "opex": opex_sum,
            "asr": asr_sum,
            "lbt": lbt_sum,
            "cost_recovery/deductible_cost": cost_recovery_sum,
            "cost_recovery_over_gross_rev": cost_recovery_over_gross_rev,
            "unrec_cost": unrec_cost,
            "unrec_over_costrec": unrec_over_costrec,
            "unrec_over_gross_rev": unrec_over_gross_rev,
            "ctr_net_share": ctr_net_share_sum,
            "ctr_net_share_over_gross_share": ctr_net_share_over_gross_share,
            "ctr_net_cashflow": ctr_net_cashflow_sum,
            "ctr_net_cashflow_over_gross_rev": ctr_net_cashflow_over_gross_rev,
            "ctr_npv": ctr_npv,
            "ctr_npv_sunk_cost_pooled": ctr_npv,
            "ctr_irr": ctr_irr,
            "ctr_irr_sunk_cost_pooled": ctr_irr,
            "ctr_pot": ctr_pot,
            "ctr_pv_ratio": ctr_pv_ratio,
            "ctr_pi": ctr_pi,
            "gov_ftp_share": gov_ftp_share_sum,
            "gov_equity_share": gov_equity_share_sum,
            "gov_ddmo": gov_ddmo,
            "gov_tax_income": gov_take_income_sum,
            "gov_take": gov_take_sum,
            "gov_take_over_gross_rev": gov_take_over_gross_rev,
            "gov_take_npv": gov_take_npv,
            "undepreciated_asset_oil": oil_undepreciated_asset_sum,
            "undepreciated_asset_gas": gas_undepreciated_asset_sum,
            "undepreciated_asset_total": total_undepreciated_asset_sum,
            "total_indirect_taxes": oil_indirect_tax_sum + gas_indirect_tax_sum,
            "oil_indirect_taxes": oil_indirect_tax_sum,
            "gas_indirect_taxes": gas_indirect_tax_sum,
            "total_carry_forward_depreciation": total_carward_depreciation_sum,
            "oil_carry_forward_depreciation": oil_carward_depreciation_sum,
            "gas_carry_forward_depreciation": gas_carward_depreciation_sum,
        }
