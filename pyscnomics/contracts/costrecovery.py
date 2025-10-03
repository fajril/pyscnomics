"""
Handles calculations associated with PSC Cost Recovery.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

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
    SunkCostMethod,
    InitialYearDepreciationIncurred,
)
from pyscnomics.econ.depreciation import unit_of_production_rate


class SunkCostException(Exception):
    """Exception to raise for a misuse of Sunk Cost Method"""

    pass


class CostRecoveryException(Exception):
    """Exception to raise for a misuse of Cost Recovery attributes"""

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

    # Attributes associated with total sunk cost
    # (sunk cost depreciation + sunk cost non depreciable)
    _oil_sunk_cost_depre_non_depre: np.ndarray = field(default=None, init=False, repr=False)
    _gas_sunk_cost_depre_non_depre: np.ndarray = field(default=None, init=False, repr=False)

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
    _oil_unrecovered_before_transfer: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _gas_unrecovered_before_transfer: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _oil_cost_to_be_recovered: np.ndarray = field(default=None, init=False, repr=False)
    _gas_cost_to_be_recovered: np.ndarray = field(default=None, init=False, repr=False)
    _oil_cost_recovery: np.ndarray = field(default=None, init=False, repr=False)
    _gas_cost_recovery: np.ndarray = field(default=None, init=False, repr=False)
    _oil_ets_before_transfer: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ets_before_transfer: np.ndarray = field(default=None, init=False, repr=False)
    _transfer_to_oil: np.ndarray = field(default=None, init=False, repr=False)
    _transfer_to_gas: np.ndarray = field(default=None, init=False, repr=False)
    _oil_unrecovered_after_transfer: np.ndarray = field(default=None, init=False, repr=False)
    _gas_unrecovered_after_transfer: np.ndarray = field(default=None, init=False, repr=False)
    _oil_cost_to_be_recovered_after_tf: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _gas_cost_to_be_recovered_after_tf: np.ndarray = field(
        default=None, init=False, repr=False
    )
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
    _consolidated_undepreciated_assets: dict = field(
        default_factory=lambda: {}, init=False, repr=False
    )
    _consolidated_ftp: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_ftp_ctr: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_ftp_gov: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_ic: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_ic_unrecovered: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_ic_paid: np.ndarray = field(default=None, init=False, repr=False)
    _consolidated_unrecovered_before_transfer: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_cost_recovery_before_transfer: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_ets_before_transfer: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_unrecovered_after_transfer: np.ndarray = field(
        default=None, init=False, repr=False
    )
    _consolidated_cost_to_be_recovered_after_tf: np.ndarray = field(
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

        # Initialize attributes associated with cumulative sunk cost treatment
        self._oil_sunk_cost_depre_non_depre = np.zeros_like(self.project_years, dtype=float)
        self._gas_sunk_cost_depre_non_depre = np.zeros_like(self.project_years, dtype=float)

    def _check_attributes(self):
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
        Compute and assign depreciation schedules and undepreciated assets
        for oil and gas cost types.

        This method calculates depreciation and undepreciated asset balances
        for ``postonstream``, ``preonstream``, and ``sunk_cost`` expenditures
        under both oil and gas. Depreciation is calculated using the specified
        method and decline factor, adjusted for inflation and tax effects.
        Results are aligned with the project years and stored in internal
        attributes.

        Parameters
        ----------
        depr_method : DeprMethod
            Depreciation method to apply (e.g., straight-line, declining balance).
        decline_factor : float or int
            Factor used for declining balance depreciation (e.g., 2 for double
            declining balance).
        year_inflation : np.ndarray
            Array of year-on-year inflation multipliers applied to cost recovery.
        inflation_rate : np.ndarray or int or float
            Inflation rate(s) applied to depreciation. Can be a scalar or an array.
        tax_rate : np.ndarray or float
            Applicable tax rate(s) used in depreciation calculation.
        inflation_rate_applied_to : InflationAppliedTo
            Specifies whether inflation is applied to CAPEX, OPEX, CAPEX and OPEX,
            or None.

        Returns
        -------
        None
            Updates internal attributes in place:
            - ``_oil_depreciations`` and ``_gas_depreciations``:
              dicts of per-year depreciation schedules.
            - ``_oil_undepreciated_assets`` and ``_gas_undepreciated_assets``:
              dicts of undepreciated balances.

        Notes
        -----
        - Cost types considered: ``postonstream``, ``preonstream``, ``sunk_cost``.
        - Onstream year alignment is determined by cost type:
          - ``postonstream`` → depreciation begins immediately (direct).
          - ``preonstream`` and ``sunk_cost`` → depreciation begins in the
            onstream year.
        - Depreciation and undepreciated assets are computed via
          ``total_depreciation_rate`` of the respective capital objects.
        - Inflation and tax are incorporated consistently with the
          ``inflation_rate_applied_to`` policy.
        """

        # Define the mapping between fluids, cost types, capital objects, and
        # the initial year of depreciation incurred
        depr_mapping = {
            "oil": (
                (
                    "postonstream",
                    self._oil_capital_postonstream,
                    InitialYearDepreciationIncurred.DIRECT,
                ),
                (
                    "preonstream",
                    self._oil_capital_preonstream,
                    InitialYearDepreciationIncurred.ONSTREAM_YEAR,
                ),
                (
                    "sunk_cost",
                    self._oil_capital_sunk_cost,
                    InitialYearDepreciationIncurred.ONSTREAM_YEAR,
                ),
            ),
            "gas": (
                (
                    "postonstream",
                    self._gas_capital_postonstream,
                    InitialYearDepreciationIncurred.DIRECT,
                ),
                (
                    "preonstream",
                    self._gas_capital_preonstream,
                    InitialYearDepreciationIncurred.ONSTREAM_YEAR,
                ),
                (
                    "sunk_cost",
                    self._gas_capital_sunk_cost,
                    InitialYearDepreciationIncurred.ONSTREAM_YEAR,
                ),
            )
        }

        # Define intermediate attributes:
        # "depreciations": dict and "undepreciated_assets": dict
        fluids = ["oil", "gas"]
        cost_types = ["postonstream", "preonstream", "sunk_cost"]

        depreciations = {f: {c: None for c in cost_types} for f in fluids}
        undepreciated_assets = {f: {c: None for c in cost_types} for f in fluids}

        for (f, mapping) in depr_mapping.items():
            for (c, obj, depr_type) in mapping:
                (
                    depreciations[f][c],
                    undepreciated_assets[f][c]
                ) = obj.total_depreciation_rate(
                    oil_onstream_year=self.oil_onstream_date.year,
                    gas_onstream_year=self.gas_onstream_date.year,
                    initial_depreciation_year=depr_type,
                    depr_method=depr_method,
                    decline_factor=decline_factor,
                    year_inflation=year_inflation,
                    inflation_rate=inflation_rate,
                    tax_rate=tax_rate,
                    inflation_rate_applied_to=inflation_rate_applied_to,
                )

        # Set initial values for attributes "_oil_depreciations" and "_gas_depreciations"
        self._oil_depreciations = {
            c: np.zeros_like(self.project_years, dtype=float) for c in cost_types
        }

        self._gas_depreciations = {
            c: np.zeros_like(self.project_years, dtype=float) for c in cost_types
        }

        # Set initial values for attributes "_oil_undepreciated_assets"
        # and "_gas_undepreciated_assets"
        self._oil_undepreciated_assets = {c: None for c in cost_types}
        self._gas_undepreciated_assets = {c: None for c in cost_types}

        # Set attributes "_oil_depreciations", "_gas_depreciations",
        # "_oil_undepreciated_assets", and "gas_undepreciated_assets"
        for f, f_depr, f_undepr in [
            ("oil", self._oil_depreciations, self._oil_undepreciated_assets),
            ("gas", self._gas_depreciations, self._gas_undepreciated_assets)
        ]:
            for c in cost_types:
                f_depr[c] = depreciations[f][c]
                f_undepr[c] = undepreciated_assets[f][c]

    def _get_modified_depreciations(self, sum_undepreciated_cost: bool) -> None:
        """
        Modify oil and gas depreciation and undepreciated asset schedules.

        This method applies three sequential adjustments to the internal
        depreciation and undepreciated asset arrays for both oil and gas:

        1. **Tolerance cleanup**: Very small undepreciated asset values
           (below a fixed tolerance of ``1e-5``) are set to zero to
           eliminate numerical noise.
        2. **Final-year adjustment** (optional): If
           ``sum_undepreciated_cost=True``, all remaining undepreciated
           balances are summed and transferred into the final year of the
           corresponding depreciation schedule. After this transfer,
           the undepreciated asset arrays are reset to zeros.
        3. **Depreciation reclassification**: All ``preonstream`` cost
           depreciations are combined into ``sunk_cost`` depreciations.
           Once transferred, the ``preonstream`` depreciation arrays are
           reset to zeros.

        Parameters
        ----------
        sum_undepreciated_cost : bool
            If True, any remaining undepreciated costs are added to the last
            project year’s depreciation value and the undepreciated assets
            are cleared. If False, undepreciated balances remain as-is
            (apart from tolerance cleanup).

        Returns
        -------
        None
            The method updates the following attributes in place:

            - ``_oil_depreciations``
            - ``_gas_depreciations``
            - ``_oil_undepreciated_assets``
            - ``_gas_undepreciated_assets``

        Notes
        -----
        - Affects the following cost types: ``postonstream``, ``preonstream``,
          and ``sunk_cost``.
        - The tolerance threshold is fixed at ``1e-5``.
        - Depreciations are updated only in the last year if
          ``sum_undepreciated_cost=True``.
        - After modification, all ``preonstream`` depreciation values are
          reclassified into ``sunk_cost`` and cleared from their original arrays.
        """

        undepre_assets = [self._oil_undepreciated_assets, self._gas_undepreciated_assets]
        cost_types = ["postonstream", "preonstream", "sunk_cost"]

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

        # Modify depreciations.
        # Combine sunk cost and preonstream cost depreciations, then assign
        # the results as the "modified" sunk cost depreciations
        self._oil_depreciations["sunk_cost"] += self._oil_depreciations["preonstream"]
        self._gas_depreciations["sunk_cost"] += self._gas_depreciations["preonstream"]

        # Assign preonstream cost depreciations as zeros
        self._oil_depreciations["preonstream"] = np.zeros_like(self.project_years, dtype=float)
        self._gas_depreciations["preonstream"] = np.zeros_like(self.project_years, dtype=float)

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

    def _get_rc_icp_pretax(self):
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

        This method computes the tax rates for the project years depending on the specified
        `tax_regime`. It uses predefined tax configurations for certain years (2013, 2016, 2020)
        and allows for specific tax regimes that override these configurations. For tax regimes
        like `NAILED_DOWN`, the tax rate is fixed based on the start year of the project.

        Parameters
        ----------
        tax_regime : TaxRegime
            The tax regime to be applied. It determines how tax rates are selected and can be
            one of the following:
            -   `TaxRegime.UU_07_2021`
            -   `TaxRegime.UU_02_2020`
            -   `TaxRegime.UU_36_2008`
            -   `TaxRegime.NAILED_DOWN`

        Returns
        -------
        tax_rate_arr : np.ndarray
            A 1D array of tax rates for each project year. The tax rate is determined based on
            the tax regime and the project's starting year in relation to the predefined tax
            configurations.
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
    ):
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

    def _allocate_sunk_cost(
        self, sunk_cost: np.ndarray, preonstream: np.ndarray
    ) -> np.ndarray:
        """
        Allocate sunk cost and preonstream expenditures to the onstream year.

        This method aggregates the total value of ``sunk_cost`` and
        ``preonstream`` arrays into a single bulk value and assigns it to
        the project year corresponding to the earlier of the oil or gas
        onstream dates. All other years are set to zero.

        This method will be used to support sunk cost treatment configuration.

        Parameters
        ----------
        sunk_cost : np.ndarray
            Array of sunk cost values across project years.
        preonstream : np.ndarray
            Array of preonstream cost values across project years.

        Returns
        -------
        np.ndarray
            A 1D array aligned with ``project_years`` where the summed bulk
            value of ``sunk_cost`` and ``preonstream`` is positioned at the
            onstream year, with zeros elsewhere.

        Raises
        ------
        ValueError
            If the onstream year does not uniquely match exactly one entry
            in ``project_years``.

        Notes
        -----
        - The onstream year is determined as the earlier of
          ``oil_onstream_date.year`` and ``gas_onstream_date.year``.
        - Both ``sunk_cost`` and ``preonstream`` arrays are expected to
          align with ``project_years`` in length.
        """

        # Calculate bulk value of (sunk_cost + preonstream)
        bulk_value = float(np.sum(sunk_cost + preonstream))

        # Determine the location of onstream year in project years array
        onstream_yr = min([self.oil_onstream_date.year, self.gas_onstream_date.year])
        match = np.flatnonzero(self.project_years == onstream_yr)

        # Expected only a single match; raise an exception if multiple matches occurs
        if match.size != 1:
            raise ValueError(f"Expected one onstream year match, got {match.size} instead")

        onstream_id = int(match[0])

        # Create a new array with bulk value positioned at the onstream year
        arr = np.zeros_like(self.project_years, dtype=float)
        arr[onstream_id] = bulk_value

        return arr

    def _apply_sunk_cost_treatment(self, sunk_cost_method: SunkCostMethod) -> None:
        """
        Apply sunk cost treatment for both oil and gas based on the selected method.

        This method handles the allocation of non-depreciable and depreciable sunk
        costs depending on the specified sunk cost treatment method.

        The treatment can either depreciate sunk costs as tangible assets or
        pool them in the onstream year.

        Parameters
        ----------
        sunk_cost_method : SunkCostMethod
            The method used for treating depreciable sunk cost.
            Supported options:
                - SunkCostMethod.DEPRECIATED_TANGIBLE
                    Adds depreciable sunk cost to the corresponding
                    depreciation schedules.
                - SunkCostMethod.POOLED_1ST_YEAR
                    Pools depreciable sunk cost in the onstream year
                    through allocation.

        Raises
        ------
        CostRecoveryException
            If the specified `sunk_cost_method` is not recognized.

        Notes
        -----
        - Non-depreciable sunk costs are always allocated directly
          to the pre-onstream period.
        - Depreciable sunk costs are treated differently depending
          on the selected method:
            * **Depreciated Tangible**: Incorporated into depreciation schedules.
            * **Pooled 1st Year**: Allocated entirely in the onstream year.
        """

        # Carry out treatment for non depreciable sunk cost
        oil_non_dep = self._allocate_sunk_cost(
            sunk_cost=self._oil_non_depreciable_sunk_cost,
            preonstream=self._oil_non_depreciable_preonstream,
        )
        gas_non_dep = self._allocate_sunk_cost(
            sunk_cost=self._gas_non_depreciable_sunk_cost,
            preonstream=self._gas_non_depreciable_preonstream,
        )

        # Specify treatment for depreciable sunk cost for OPTION 1: DEPRECIATED TANGIBLE
        if sunk_cost_method == SunkCostMethod.DEPRECIATED_TANGIBLE:
            oil_dep = (
                self._oil_depreciations["sunk_cost"] + self._oil_depreciations["preonstream"]
            )
            gas_dep = (
                self._gas_depreciations["sunk_cost"] + self._gas_depreciations["preonstream"]
            )

        # Specify treatment for depreciable sunk cost for OPTION 2: POOLED AT ONSTREAM YEAR
        elif sunk_cost_method == SunkCostMethod.POOLED_1ST_YEAR:
            oil_dep = self._allocate_sunk_cost(
                sunk_cost=self._oil_depreciable_sunk_cost,
                preonstream=self._oil_depreciable_preonstream,
            )
            gas_dep = self._allocate_sunk_cost(
                sunk_cost=self._gas_depreciable_sunk_cost,
                preonstream=self._gas_depreciable_preonstream,
            )

        else:
            raise CostRecoveryException(
                f"Sunk cost treatment method ({sunk_cost_method}) is unrecognized"
            )

        # Specify total depreciable + non depreciable sunk cost
        self._oil_sunk_cost_depre_non_depre = oil_non_dep + oil_dep
        self._gas_sunk_cost_depre_non_depre = gas_non_dep + gas_dep

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
            Intangible, Operating Expenditures (OPEX) and Abandonment Site and Restoration (ASR) Expenditures.
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
        The function to get the tax payment of the Cost Recovery contract based on the chosen regime.
        There are three existing regime, they are :
            - PRE_PDJP_20_2017
            - PDJP_20_2017
            - DIRECT_MODE

        Notes
        ----------
        [1] PRE_PDJP_20_2017
            In general, the tax will be applied after the contractor share is not zero.
            The taxable object that occurs at the time prior of the applied tax time will be accumulated and weighted
            on the time when the tax is applied.

        [2] PDJP_20_2017
            The tax will be applied when the cumulative First Tranche Petroleum (FTP) is greater than
            the unrecoverable cost

        [3] DIRECT_MODE
            The tax will be applied regardless the cumulative FTP, unrecoverable cost and Contractor Share condition.


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



    def _prepare_sunk_cost_contract(self, depr_method, decline_factor):
        # Get the revenue total
        all_revenue = self._oil_revenue + self._gas_revenue

        # Find the first index where b is non-zero
        start_idx = np.argmax(all_revenue != 0)

        # Slice a from that index to the end
        prod_year = self.project_years[start_idx:]

        # Prepare several attributes associated with sunk cost and preonstream cost
        self.fit_sunk_preonstream_cost(
            prod_year=prod_year,
            prod=self._oil_lifting.prod_rate_total,
            tax_rate=0.0,
            depr_method=depr_method,
            decline_factor=decline_factor,
            salvage_value=0.0,
        )

    def _get_consolidated_profiles_cr(self, ftp_tax_regime: FTPTaxRegime) -> None:

        # Attributes associated with consolidated lifting
        self._consolidated_lifting = (
            self._oil_lifting.get_lifting_rate_arr()
            + self._gas_lifting.get_lifting_rate_arr()
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

        # Attributes associated with consolidated expenditures pre tax
        self._consolidated_capital_expenditures_pre_tax = (
            self._oil_capital_expenditures_pre_tax
            + self._gas_capital_expenditures_pre_tax
        )
        self._consolidated_intangible_expenditures_pre_tax = (
            self._oil_intangible_expenditures_pre_tax
            + self._gas_intangible_expenditures_pre_tax
        )
        self._consolidated_opex_expenditures_pre_tax = (
            self._oil_opex_expenditures_pre_tax + self._gas_opex_expenditures_pre_tax
        )
        self._consolidated_asr_expenditures_pre_tax = (
            self._oil_asr_expenditures_pre_tax + self._gas_asr_expenditures_pre_tax
        )
        self._consolidated_lbt_expenditures_pre_tax = (
            self._oil_lbt_expenditures_pre_tax + self._gas_lbt_expenditures_pre_tax
        )
        self._consolidated_cost_of_sales_expenditures_pre_tax = (
            self._oil_cost_of_sales_expenditures_pre_tax
            + self._gas_cost_of_sales_expenditures_pre_tax
        )
        self._consolidated_expenditures_pre_tax = (
            self._oil_total_expenditures_pre_tax + self._gas_total_expenditures_pre_tax
        )

        # Attributes associated with consolidated indirect tax
        self._consolidated_capital_indirect_tax = (
            self._oil_capital_indirect_tax + self._gas_capital_indirect_tax
        )
        self._consolidated_intangible_indirect_tax = (
            self._oil_intangible_indirect_tax + self._gas_intangible_indirect_tax
        )
        self._consolidated_opex_indirect_tax = (
            self._oil_opex_indirect_tax + self._gas_opex_indirect_tax
        )
        self._consolidated_asr_indirect_tax = (
            self._oil_asr_indirect_tax + self._gas_asr_indirect_tax
        )
        self._consolidated_lbt_indirect_tax = (
            self._oil_lbt_indirect_tax + self._gas_lbt_indirect_tax
        )
        self._consolidated_cost_of_sales_indirect_tax = (
            self._oil_cost_of_sales_indirect_tax + self._gas_cost_of_sales_indirect_tax
        )
        self._consolidated_indirect_tax = (
            self._oil_total_indirect_tax + self._gas_total_indirect_tax
        )

        # Attributes associated with consolidated expenditures post tax
        self._consolidated_capital_expenditures_post_tax = (
            self._oil_capital_expenditures_post_tax
            + self._gas_capital_expenditures_post_tax
        )
        self._consolidated_intangible_expenditures_post_tax = (
            self._oil_intangible_expenditures_post_tax
            + self._gas_intangible_expenditures_post_tax
        )
        self._consolidated_opex_expenditures_post_tax = (
            self._oil_opex_expenditures_post_tax + self._gas_opex_expenditures_post_tax
        )
        self._consolidated_asr_expenditures_post_tax = (
            self._oil_asr_expenditures_post_tax + self._gas_asr_expenditures_post_tax
        )
        self._consolidated_lbt_expenditures_post_tax = (
            self._oil_lbt_expenditures_post_tax + self._gas_lbt_expenditures_post_tax
        )
        self._consolidated_cost_of_sales_expenditures_post_tax = (
            self._oil_cost_of_sales_expenditures_post_tax
            + self._gas_cost_of_sales_expenditures_post_tax
        )
        self._consolidated_expenditures_post_tax = (
            self._oil_total_expenditures_post_tax + self._gas_total_expenditures_post_tax
        )

        # Attribute associated with consolidated non capital
        self._consolidated_non_capital = self._oil_non_capital + self._gas_non_capital

        # Attributes associated with consolidated depreciation
        self._consolidated_depreciation = self._oil_depreciation + self._gas_depreciation
        self._consolidated_undepreciated_asset = (
            self._oil_undepreciated_asset + self._gas_undepreciated_asset
        )
        self._consolidated_carry_forward_depreciation = (
            self._oil_carry_forward_depreciation + self._gas_carry_forward_depreciation
        )

        # Attributes associated with core business logic consolidated profiles
        self._consolidated_ftp = self._oil_ftp + self._gas_ftp
        self._consolidated_ftp_ctr = self._oil_ftp_ctr + self._gas_ftp_ctr
        self._consolidated_ftp_gov = self._oil_ftp_gov + self._gas_ftp_gov
        self._consolidated_ic = self._oil_ic + self._gas_ic
        self._consolidated_ic_unrecovered = (
            self._oil_ic_unrecovered + self._gas_ic_unrecovered
        )
        self._consolidated_ic_paid = self._oil_ic_paid + self._gas_ic_paid
        self._consolidated_unrecovered_before_transfer = (
            self._oil_unrecovered_before_transfer + self._gas_unrecovered_before_transfer
        )
        self._consolidated_cost_recovery_before_transfer = (
            self._oil_cost_recovery + self._gas_cost_recovery
        )
        self._consolidated_ets_before_transfer = (
            self._oil_ets_before_transfer + self._gas_ets_before_transfer
        )
        self._consolidated_unrecovered_after_transfer = (
            self._oil_unrecovered_after_transfer + self._gas_unrecovered_after_transfer
        )
        self._consolidated_cost_to_be_recovered_after_tf = (
            self._oil_cost_to_be_recovered_after_tf
            + self._gas_cost_to_be_recovered_after_tf
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

        # Attributes associated with consolidated DMO
        self._consolidated_dmo_volume = self._oil_dmo_volume + self._gas_dmo_volume
        self._consolidated_dmo_fee = self._oil_dmo_fee + self._gas_dmo_fee
        self._consolidated_ddmo = self._oil_ddmo + self._gas_ddmo
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

        # self._consolidated_cashflow = self._consolidated_contractor_take - (
        #     self._consolidated_capital_cost
        #     + self._consolidated_non_capital
        #     + self._consolidated_cost_of_sales
        # )

        self._consolidated_cashflow = self._oil_cashflow + self._gas_cashflow

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
        sunk_cost_method: SunkCostMethod = SunkCostMethod.DEPRECIATED_TANGIBLE,
    ):

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

        # Prepare sunk costs and preonstream costs
        self._get_sunkcost_array()
        self._get_preonstream_array()
        self._modify_sunk_cost_preonstream()

        # Calculate depreciations and undepreciated assets
        self._get_depreciation(
            depr_method=depr_method,
            decline_factor=decline_factor,
            year_inflation=year_inflation,
            inflation_rate=inflation_rate,
            tax_rate=vat_rate,
            inflation_rate_applied_to=inflation_rate_applied_to,
        )

        # Modify depreciations, accounting for various adjustments
        self._get_modified_depreciations(sum_undepreciated_cost=sum_undepreciated_cost)

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

        # Apply sunk cost treatment
        self._apply_sunk_cost_treatment(sunk_cost_method=sunk_cost_method)

        print('\t')
        print(f'Filetype: {type(self._oil_sunk_cost_depre_non_depre)}')
        print(f'Length: {len(self._oil_sunk_cost_depre_non_depre)}')
        print('_oil_sunk_cost_depre_non_depre = \n', self._oil_sunk_cost_depre_non_depre)

        oil_capital_total = (
            self._oil_sunk_cost_depre_non_depre
            + self._oil_depreciations["postonstream"]
            + self._oil_carry_forward_depreciation
            + self._oil_intangible_expenditures_post_tax
            + self._oil_opex_expenditures_post_tax
            + self._oil_asr_expenditures_post_tax
            + self._oil_lbt_expenditures_post_tax
            + self._oil_cost_of_sales_expenditures_post_tax
        )

        print('\t')
        print(f'Filetype: {type(oil_capital_total)}')
        print(f'Length: {len(oil_capital_total)}')
        print('oil_depr_total = \n', oil_capital_total)

        # (
        #     self._oil_unrecovered_before_transfer,
        #     self._oil_cost_to_be_recovered,
        #     self._oil_cost_recovery,
        # ) = psc_tools.get_unrec_cost_2b_recovered_costrec(
        #     project_years=self.project_years,
        #     depreciation=self._oil_depreciation,
        #     non_capital=self._oil_non_capital,
        #     revenue=self._oil_revenue,
        #     ftp_ctr=self._oil_ftp_ctr,
        #     ftp_gov=self._oil_ftp_gov,
        #     ic=self._oil_ic_paid,
        #     cr_cap_rate=self.oil_cr_cap_rate,
        # )
        #
        # (
        #     self._gas_unrecovered_before_transfer,
        #     self._gas_cost_to_be_recovered,
        #     self._gas_cost_recovery,
        # ) = psc_tools.get_unrec_cost_2b_recovered_costrec(
        #     project_years=self.project_years,
        #     depreciation=self._gas_depreciation,
        #     non_capital=self._gas_non_capital,
        #     revenue=self._gas_revenue,
        #     ftp_ctr=self._gas_ftp_ctr,
        #     ftp_gov=self._gas_ftp_gov,
        #     ic=self._gas_ic_paid,
        #     cr_cap_rate=self.gas_cr_cap_rate,
        # )

        # # ETS (Equity to be Split) before transfer/consolidation
        # self._oil_ets_before_transfer = self._get_ets_before_transfer(
        #     revenue=self._oil_revenue,
        #     ftp_ctr=self._oil_ftp_ctr,
        #     ftp_gov=self._oil_ftp_gov,
        #     ic=self._oil_ic_paid,
        #     cost_recovery=self._oil_cost_recovery,
        # )
        #
        # self._gas_ets_before_transfer = self._get_ets_before_transfer(
        #     revenue=self._gas_revenue,
        #     ftp_ctr=self._gas_ftp_ctr,
        #     ftp_gov=self._gas_ftp_gov,
        #     ic=self._gas_ic_paid,
        #     cost_recovery=self._gas_cost_recovery,
        # )
        #
        # self._transfer_to_oil, self._transfer_to_gas = psc_tools.get_transfer(
        #     gas_unrecovered=self._gas_unrecovered_before_transfer,
        #     oil_unrecovered=self._oil_unrecovered_before_transfer,
        #     gas_ets_pretransfer=self._gas_ets_before_transfer,
        #     oil_ets_pretransfer=self._oil_ets_before_transfer,
        # )
        #
        # (
        #     self._oil_unrecovered_after_transfer,
        #     self._oil_cost_to_be_recovered_after_tf,
        #     self._oil_cost_recovery_after_tf,
        # ) = psc_tools.get_unrec_cost_2b_recovered_costrec(
        #     project_years=self.project_years,
        #     depreciation=self._oil_depreciation,
        #     non_capital=self._oil_non_capital,
        #     revenue=self._oil_revenue + self._transfer_to_oil,
        #     ftp_ctr=self._oil_ftp_ctr,
        #     ftp_gov=self._oil_ftp_gov,
        #     ic=self._oil_ic_paid,
        #     cr_cap_rate=self.oil_cr_cap_rate,
        # )
        #
        # (
        #     self._gas_unrecovered_after_transfer,
        #     self._gas_cost_to_be_recovered_after_tf,
        #     self._gas_cost_recovery_after_tf,
        # ) = psc_tools.get_unrec_cost_2b_recovered_costrec(
        #     project_years=self.project_years,
        #     depreciation=self._gas_depreciation,
        #     non_capital=self._gas_non_capital,
        #     revenue=self._gas_revenue + self._transfer_to_gas,
        #     ftp_ctr=self._gas_ftp_ctr,
        #     ftp_gov=self._gas_ftp_gov,
        #     ic=self._gas_ic_paid,
        #     cr_cap_rate=self.gas_cr_cap_rate,
        # )
        #
        # # ETS after Transfer
        # self._oil_ets_after_transfer = self._get_ets_after_transfer(
        #     ets_before_tf=self._oil_ets_before_transfer,
        #     revenue=self._oil_revenue,
        #     ftp_ctr=self._oil_ftp_ctr,
        #     ftp_gov=self._oil_ftp_gov,
        #     ic=self._oil_ic_paid,
        #     cost_recovery=self._oil_cost_recovery_after_tf,
        #     transferred_in=self._transfer_to_oil,
        #     transferred_out=self._transfer_to_gas,
        # )
        #
        # self._gas_ets_after_transfer = self._get_ets_after_transfer(
        #     ets_before_tf=self._gas_ets_before_transfer,
        #     revenue=self._gas_revenue,
        #     ftp_ctr=self._gas_ftp_ctr,
        #     ftp_gov=self._gas_ftp_gov,
        #     ic=self._gas_ic_paid,
        #     cost_recovery=self._gas_cost_recovery_after_tf,
        #     transferred_in=self._transfer_to_gas,
        #     transferred_out=self._transfer_to_oil,
        # )
        #
        # # ES (Equity Share)
        # self._oil_contractor_share, self._oil_government_share = self._get_equity_share(
        #     ets=self._oil_ets_after_transfer, pretax_ctr=self.oil_ctr_pretax_share
        # )
        #
        # self._gas_contractor_share, self._gas_government_share = self._get_equity_share(
        #     ets=self._gas_ets_after_transfer, pretax_ctr=self.gas_ctr_pretax_share
        # )
        #
        # # DMO
        # self._oil_dmo_volume, self._oil_dmo_fee, self._oil_ddmo = psc_tools.get_dmo(
        #     onstream_date=self.oil_onstream_date,
        #     start_date=self.start_date,
        #     project_years=self.project_years,
        #     dmo_holiday_duration=self.oil_dmo_holiday_duration,
        #     dmo_volume_portion=self.oil_dmo_volume_portion,
        #     dmo_fee_portion=self.oil_dmo_fee_portion,
        #     lifting=self._oil_lifting,
        #     price=self._oil_wap_price,
        #     ctr_pretax_share=self.oil_ctr_pretax_share,
        #     unrecovered_cost=self._oil_unrecovered_after_transfer,
        #     is_dmo_end_weighted=is_dmo_end_weighted,
        #     ets=self._oil_ets_after_transfer,
        #     ctr_ets=self._oil_contractor_share,
        #     ctr_ftp=self._oil_ftp_ctr,
        #     post_uu_22_year2001=post_uu_22_year2001,
        # )
        #
        # self._gas_dmo_volume, self._gas_dmo_fee, self._gas_ddmo = psc_tools.get_dmo(
        #     onstream_date=self.gas_onstream_date,
        #     start_date=self.start_date,
        #     project_years=self.project_years,
        #     dmo_holiday_duration=self.gas_dmo_holiday_duration,
        #     dmo_volume_portion=self.gas_dmo_volume_portion,
        #     dmo_fee_portion=self.gas_dmo_fee_portion,
        #     lifting=self._gas_lifting,
        #     price=self._gas_wap_price,
        #     ctr_pretax_share=self.gas_ctr_pretax_share,
        #     unrecovered_cost=self._gas_unrecovered_after_transfer,
        #     is_dmo_end_weighted=is_dmo_end_weighted,
        #     ets=self._gas_ets_after_transfer,
        #     ctr_ets=self._gas_contractor_share,
        #     ctr_ftp=self._gas_ftp_ctr,
        #     post_uu_22_year2001=post_uu_22_year2001,
        # )
        #
        # # Adjusting DDMO for Pre PDJP
        # if ftp_tax_regime is FTPTaxRegime.PRE_PDJP_20_2017:
        #     self._oil_ddmo = np.where(self._oil_contractor_share > 0, self._oil_ddmo, 0)
        #     self._gas_ddmo = np.where(self._gas_contractor_share > 0, self._gas_ddmo, 0)
        #
        # # Taxable income (also known as Net Contractor Share - NCS)
        # self._oil_taxable_income = (
        #     self._oil_ftp_ctr
        #     + self._oil_contractor_share
        #     + self._oil_ic_paid
        #     - self._oil_ddmo
        # )
        # self._gas_taxable_income = (
        #     self._gas_ftp_ctr
        #     + self._gas_contractor_share
        #     + self._gas_ic_paid
        #     - self._gas_ddmo
        # )
        #
        # # Tax Payment
        # # Generating Tax array if tax_rate argument is a single value not array
        # if isinstance(effective_tax_rate, float):
        #     self._tax_rate_arr = np.full_like(
        #         self.project_years, effective_tax_rate, dtype=float
        #     )
        #
        # # Generating Tax array based on the tax regime if tax_rate argument is None
        # elif effective_tax_rate is None:
        #     self._tax_rate_arr = self._get_tax_by_regime(tax_regime=tax_regime)
        # elif isinstance(effective_tax_rate, np.ndarray):
        #     self._tax_rate_arr = effective_tax_rate
        # else:
        #     self._tax_rate_arr = np.full_like(
        #         self.project_years, fill_value=0.0, dtype=float
        #     )
        #
        # self._oil_tax_payment = self._get_tax_payment(
        #     ctr_share=self._oil_contractor_share,
        #     taxable_income=self._oil_taxable_income,
        #     tax_rate=self._tax_rate_arr,
        #     ftp_ctr=self._oil_ftp_ctr,
        #     unrec_cost=self._oil_unrecovered_after_transfer,
        #     ftp_tax_regime=ftp_tax_regime,
        # )
        #
        # self._gas_tax_payment = self._get_tax_payment(
        #     ctr_share=self._gas_contractor_share,
        #     taxable_income=self._gas_taxable_income,
        #     tax_rate=self._tax_rate_arr,
        #     ftp_ctr=self._gas_ftp_ctr,
        #     unrec_cost=self._gas_unrecovered_after_transfer,
        #     ftp_tax_regime=ftp_tax_regime,
        # )
        #
        # # Contractor Share
        # self._oil_ctr_net_share = self._oil_taxable_income - self._oil_tax_payment
        # self._gas_ctr_net_share = self._gas_taxable_income - self._gas_tax_payment
        #
        # # Contractor Take by Fluid
        # self._oil_contractor_take = (
        #     self._oil_taxable_income
        #     - self._oil_tax_payment
        #     + self._oil_cost_recovery_after_tf
        # )
        #
        # self._gas_contractor_take = (
        #     self._gas_taxable_income
        #     - self._gas_tax_payment
        #     + self._gas_cost_recovery_after_tf
        # )
        #
        # # Government Take by Fluid
        # self._oil_government_take = (
        #     self._oil_ftp_gov
        #     + self._oil_government_share
        #     + self._oil_tax_payment
        #     + self._oil_ddmo
        # )
        #
        # self._gas_government_take = (
        #     self._gas_ftp_gov
        #     + self._gas_government_share
        #     + self._gas_tax_payment
        #     + self._gas_ddmo
        # )
        #
        # # Returning the gross revenue
        # self._oil_revenue = (
        #     self._oil_revenue + self._oil_cost_of_sales_expenditures_post_tax
        # )
        # self._gas_revenue = (
        #     self._gas_revenue + self._gas_cost_of_sales_expenditures_post_tax
        # )
        #
        # # Consolidated attributes
        # self._get_consolidated_profiles_cr(ftp_tax_regime=ftp_tax_regime)
