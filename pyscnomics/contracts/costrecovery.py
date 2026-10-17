"""
Handles calculations associated with PSC Cost Recovery.
"""

# import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from numbers import Real
import numpy as np
import numpy.typing as npt
import pandas as pd
from typing_extensions import override

from pyscnomics.econ._array import last_float64 as _last_float
from pyscnomics.econ._array import sum_float64 as _sum_float
from pyscnomics.econ.costs import CapitalCost
from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.summary_profile import PostRunSummaryProfile
import pyscnomics.contracts.psc_tools as psc_tools
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
from pyscnomics.input_validation import InputValidationError, engine_input_issue
from pyscnomics.validation_rules import (
    condition_dict_is_mixed_form,
    contains_negative,
    fraction_value_out_of_range,
    length_exceeds_project,
    length_mismatch,
    nonzero_indices,
)

# Set display for pandas dataframe
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_colwidth", 100)


# Dtype-only aliases match BaseProject; runtime helpers validate external shapes.
FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
YearArray = npt.NDArray[np.int64]
TableArray = FloatArray | YearArray
CostSchedules = dict[str, FloatArray]
RCICPBand = dict[str, float]
RCICPConditions = dict[str, RCICPBand]
RCICPInput = RCICPConditions | Mapping[str, npt.ArrayLike]


class SunkCostException(Exception):
    """Exception to raise for a misuse of Sunk Cost Method"""

    pass


class CostRecoveryException(Exception):
    """Exception to raise for a misuse of Cost Recovery attributes"""

    pass


class CostRecoverySummaryException(Exception):
    """Exception to be raised for a misuse of get_summary() method"""

    pass


class CostRecoveryInputError(  # pyright: ignore[reportUnsafeMultipleInheritance]
    CostRecoveryException, InputValidationError
):
    """Verified user-input Cost Recovery error retaining ``CostRecoveryException``."""

    _input_code: str = "COST_RECOVERY_INPUT_INVALID"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        InputValidationError.__init__(
            self,
            (engine_input_issue(code=code or self._input_code, message=message),),
        )


def _as_float_array(values: object, name: str) -> FloatArray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CostRecoveryException(f"{name} must be a numeric array") from exc
    if array.ndim != 1:
        raise CostRecoveryException(f"{name} must be a one-dimensional numeric array")
    return array


def _as_bool_array(values: object, name: str) -> npt.NDArray[np.bool_]:
    array = np.asarray(values, dtype=np.bool_)
    if array.ndim != 1:
        raise CostRecoveryException(f"{name} must be a one-dimensional boolean array")
    return array


def _as_float(value: object, name: str) -> float:
    if not isinstance(value, Real):
        raise CostRecoveryException(f"{name} must be numeric")
    return float(value)


def _array_float_at(values: FloatArray, index: int, name: str) -> float:
    """Read a NumPy scalar through an explicit numeric boundary."""
    if index < 0 or index >= values.size:
        raise IndexError(f"{name} index {index} out of range")
    flat_index = int(index) if isinstance(index, (bool, np.bool_)) else index
    return float(values.item(flat_index))


def _as_rc_icp_conditions(values: RCICPInput) -> RCICPConditions:
    if not isinstance(values, dict):
        raise CostRecoveryException("condition_dict must be a dictionary")

    conditions: RCICPConditions = {}
    for name, band in values.items():
        if not isinstance(band, dict):
            raise CostRecoveryException(
                "condition_dict must use normalized condition bands"
            )
        conditions[name] = {
            "bot_limit": _as_float(band.get("bot_limit"), "bot_limit"),
            "top_limit": _as_float(band.get("top_limit"), "top_limit"),
            "ctr_oil": _as_float(band.get("ctr_oil"), "ctr_oil"),
            "ctr_gas": _as_float(band.get("ctr_gas"), "ctr_gas"),
        }
    return conditions


def _require_float(value: float | None, name: str) -> float:
    if value is None:
        raise CostRecoveryException(f"{name} is unavailable before run()")
    return value


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
    condition_dict: RCICPConditions
        The condition dictionary of the split when the tax_split_type is not Conventional.
    indicator_rc_icp_sliding: FloatArray
        The indicator used in the SLIDING_SCALE or R2C tax_split_type.
    oil_ctr_pretax_share: float | FloatArray
        The Oil Contractor Pre-Tax Split or Oil Contractor Pre-Tax Split Share.
    gas_ctr_pretax_share: float | FloatArray
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
    oil_ftp_portion: float | FloatArray = field(default=0.2)
    gas_ftp_is_available: bool = field(default=True)
    gas_ftp_is_shared: bool = field(default=True)
    gas_ftp_portion: float | FloatArray = field(default=0.2)

    # Arguments associated with split
    tax_split_type: TaxSplitTypeCR = field(default=TaxSplitTypeCR.CONVENTIONAL)
    condition_dict: RCICPInput = field(default_factory=dict[str, RCICPBand])
    indicator_rc_icp_sliding: FloatArray | None = field(default=None)
    oil_ctr_pretax_share: float | FloatArray = field(default=0.25)
    gas_ctr_pretax_share: float | FloatArray = field(default=0.5)

    # Arguments associated with investment credit and cap rate
    oil_ic_rate: float | FloatArray = field(default=0.0)
    gas_ic_rate: float | FloatArray = field(default=0.0)
    ic_is_available: bool = field(default=False)
    oil_cr_cap_rate: float = field(default=1.0)
    gas_cr_cap_rate: float = field(default=1.0)

    # Arguments associated with DMO
    oil_dmo_volume_portion: float | FloatArray = field(default=0.25)
    oil_dmo_fee_portion: float | FloatArray = field(default=0.25)
    oil_dmo_holiday_duration: int = field(default=60)
    gas_dmo_volume_portion: float | FloatArray = field(default=1.0)
    gas_dmo_fee_portion: float | FloatArray = field(default=1.0)
    gas_dmo_holiday_duration: int = field(default=60)

    # Consolidated expenditure arrays are populated during ``run``.
    _consolidated_capital_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_capital_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_capital_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_intangible_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_intangible_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_intangible_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_opex_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_opex_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_opex_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_asr_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_asr_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_asr_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_lbt_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_lbt_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_lbt_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_cost_of_sales_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_cost_of_sales_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_cost_of_sales_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )

    # Arguments associated with carry forward depreciation
    oil_carry_forward_depreciation: float | FloatArray = field(default=0.0)
    gas_carry_forward_depreciation: float | FloatArray = field(default=0.0)

    # Attributes associated with carry forward depreciation
    _oil_carry_forward_depreciation: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _gas_carry_forward_depreciation: FloatArray | None = field(
        default=None, init=False, repr=False
    )

    # Attributes associated with depreciations and undepreciated assets
    _oil_depreciations: CostSchedules = field(
        default_factory=dict, init=False, repr=False
    )
    _gas_depreciations: CostSchedules = field(
        default_factory=dict, init=False, repr=False
    )
    _oil_undepreciated_assets: CostSchedules = field(
        default_factory=dict, init=False, repr=False
    )
    _gas_undepreciated_assets: CostSchedules = field(
        default_factory=dict, init=False, repr=False
    )
    _oil_depreciation: FloatArray | None = field(default=None, init=False, repr=False)
    _gas_depreciation: FloatArray | None = field(default=None, init=False, repr=False)
    _oil_sum_undepreciated_asset: float | None = field(
        default=None, init=False, repr=False
    )
    _gas_sum_undepreciated_asset: float | None = field(
        default=None, init=False, repr=False
    )

    # Attributes associated with non-depreciable costs
    _oil_non_depreciables: CostSchedules = field(
        default_factory=dict, init=False, repr=False
    )
    _gas_non_depreciables: CostSchedules = field(
        default_factory=dict, init=False, repr=False
    )

    # Attributes associated with total depreciation and total nondepreciable
    _oil_depr_total: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _gas_depr_total: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _oil_non_depr_total: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _gas_non_depr_total: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )

    # Attributes associated with FTP
    _oil_ftp: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _oil_ftp_ctr: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _oil_ftp_gov: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _gas_ftp: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _gas_ftp_ctr: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _gas_ftp_gov: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )

    # Attributes associated with investment credit
    _oil_ic: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _oil_ic_unrecovered: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _oil_ic_paid: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _gas_ic: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _gas_ic_unrecovered: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _gas_ic_paid: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )

    # Attributes associated with core business logic
    _oil_revenue_allocation: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _gas_revenue_allocation: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _oil_unrecovered_before_transfer: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _gas_unrecovered_before_transfer: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    # _oil_cost_to_be_recovered: FloatArray = field(default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False)
    # _gas_cost_to_be_recovered: FloatArray = field(default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False)
    _oil_recoverable_cost: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _gas_recoverable_cost: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _oil_cost_recovery: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _gas_cost_recovery: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _oil_ets_before_transfer: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _gas_ets_before_transfer: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _transfer_to_oil: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _transfer_to_gas: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )

    _oil_unrecovered_after_transfer: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _gas_unrecovered_after_transfer: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    # _oil_cost_to_be_recovered_after_tf: FloatArray = field(default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False)
    # _gas_cost_to_be_recovered_after_tf: FloatArray = field(default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False)
    _oil_cost_recovery_after_tf: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _gas_cost_recovery_after_tf: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _oil_ets_after_transfer: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _gas_ets_after_transfer: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )

    _oil_contractor_share: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _oil_government_share: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _gas_contractor_share: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _gas_government_share: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )

    # Attributes associated with DMO
    _oil_dmo_volume: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _oil_dmo_fee: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _oil_ddmo: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _gas_dmo_volume: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _gas_dmo_fee: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _gas_ddmo: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )

    # Attributes associated with taxable income
    _oil_taxable_income: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _gas_taxable_income: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _tax_rate_arr: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _oil_tax_payment: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _gas_tax_payment: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )

    # Attributes associated with shares
    _oil_ctr_net_share: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _gas_ctr_net_share: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _oil_contractor_take: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _gas_contractor_take: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _oil_government_take: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _gas_government_take: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )

    # Attributes associated with consolidated profiles
    _consolidated_carry_forward_depreciation: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_depreciations: CostSchedules = field(
        default_factory=lambda: {}, init=False, repr=False
    )
    _consolidated_lifting: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_wap_price: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_revenue: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_depreciable_sunk_cost: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_non_depreciable_sunk_cost: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_sunk_cost: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_depreciable_preonstream: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_non_depreciable_preonstream: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_preonstream: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_depreciable_postonstream: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_non_depreciable_postonstream: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_postonstream: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    # _consolidated_undepreciated_assets: dict = field(
    #     default_factory=lambda: {}, init=False, repr=False
    # )
    _consolidated_non_depreciables: CostSchedules = field(
        default_factory=lambda: {}, init=False, repr=False
    )
    _consolidated_total_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_total_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_total_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_depreciation: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _consolidated_sum_undepreciated_asset: float | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_depr_total: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _consolidated_non_depr_total: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )

    _consolidated_ftp: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _consolidated_ftp_ctr: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _consolidated_ftp_gov: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )

    _consolidated_ic: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _consolidated_ic_unrecovered: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_ic_paid: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )

    # _consolidated_cost_to_be_recovered_before_tf: FloatArray | None = field(
    #    default=None, init=False, repr=False
    # )
    _consolidated_recoverable_cosdt: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_recoverable_cost: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_revenue_allocation: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_unrecovered_before_transfer: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_cost_recovery_before_transfer: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_ets_before_transfer: FloatArray | None = field(
        default=None, init=False, repr=False
    )

    # _consolidated_cost_to_be_recovered_after_tf: FloatArray | None = field(
    #    default=None, init=False, repr=False
    # )
    _consolidated_unrecovered_after_transfer: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_cost_recovery_after_tf: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_ets_after_transfer: FloatArray | None = field(
        default=None, init=False, repr=False
    )

    _consolidated_contractor_share: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_government_share: FloatArray | None = field(
        default=None, init=False, repr=False
    )

    _consolidated_dmo_volume: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _consolidated_dmo_fee: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _consolidated_ddmo: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )

    _consolidated_taxable_income: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _consolidated_tax_due: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_unpaid_tax_balance: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_tax_payment: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )

    _consolidated_ctr_net_share: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _consolidated_contractor_take: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _consolidated_government_take: FloatArray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float64), init=False, repr=False
    )
    _consolidated_capital: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_non_capital: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_total_expenses: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_cashflow: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _consolidated_opex: FloatArray | None = field(default=None, init=False, repr=False)
    _consolidated_asr: FloatArray | None = field(default=None, init=False, repr=False)
    _consolidated_lbt: FloatArray | None = field(default=None, init=False, repr=False)
    _oil_total_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _gas_total_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _oil_total_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _gas_total_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _oil_total_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _gas_total_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    _oil_cashflow: FloatArray | None = field(default=None, init=False, repr=False)
    _gas_cashflow: FloatArray | None = field(default=None, init=False, repr=False)

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
            if isinstance(attr, (int, float)):
                value = _as_float(attr, attr_name)
                if fraction_value_out_of_range(value, min_fraction, max_fraction):
                    range_type = (
                        "exceeding 1.0" if value > max_fraction else "below 0.0"
                    )
                    raise CostRecoveryInputError(
                        f"The {attr_name} value ({value}) is {range_type}. Allowed range: {min_fraction} - {max_fraction}.",
                        code="CR_ATTRIBUTE_OUT_OF_RANGE",
                    )
            else:
                arr = _as_float_array(attr, attr_name)
                if length_mismatch(len(arr), len(self.project_years)):
                    raise CostRecoveryInputError(
                        f"The {attr_name} length is ({len(arr)}). The length of {attr_name} ({len(arr)}) is different from the length of the project {self.project_duration}",
                        code="CR_ATTRIBUTE_LENGTH_MISMATCH",
                    )

        # Prepare attributes associated with DMO holiday
        discrete_attributes = (
            ("oil_dmo_holiday_duration", self.oil_dmo_holiday_duration),
            ("gas_dmo_holiday_duration", self.gas_dmo_holiday_duration),
        )

        for attr_name, attr in discrete_attributes:
            if contains_negative([attr]):
                raise CostRecoveryInputError(
                    f"The {attr_name} value ({attr}) is below 0. The minimum value of this attribute is 0. ",
                    code="CR_ATTRIBUTE_NEGATIVE",
                )

        # Prepare attributes associated with carry forward depreciation
        def _normalize_carry_forward(
            value: float | FloatArray, attr_name: str
        ) -> FloatArray:
            if isinstance(value, (int, float)):
                arr = np.atleast_1d(_as_float(value, attr_name)).astype(float)
            else:
                arr = _as_float_array(value, attr_name)

            if contains_negative(arr):
                raise CostRecoveryInputError(
                    f"The {attr_name} contains negative values",
                    code="CR_CARRY_FORWARD_NEGATIVE",
                )

            if length_exceeds_project(len(arr), self.project_duration):
                raise CostRecoveryInputError(
                    f"The {attr_name} length ({len(arr)}) exceeds project duration ({self.project_duration})",
                    code="CR_CARRY_FORWARD_TOO_LONG",
                )

            result = np.zeros_like(self.project_years, dtype=float)
            result[: len(arr)] = arr
            return result

        self._oil_carry_forward_depreciation = _normalize_carry_forward(
            self.oil_carry_forward_depreciation, "oil_carry_forward_depreciation"
        )
        self._gas_carry_forward_depreciation = _normalize_carry_forward(
            self.gas_carry_forward_depreciation, "gas_carry_forward_depreciation"
        )

        # Normalize the sliding-scale / RC condition dictionary into a typed form
        self._normalize_condition_dict()

    def _normalize_condition_dict(self) -> None:
        """
        Normalize the user-supplied ``condition_dict`` (column-oriented sliding
        scale / RC table) into the row-oriented, typed ``RCICPConditions`` form.

        The input is expected to map each column name to an array-like of band
        values, e.g.::

            {
                "RC Bottom Limit": [...],
                "RC Top Limit": [...],
                "Pre Tax CTR Oil": [...],
                "Pre Tax CTR Gas": [...],
            }

        After normalization ``self.condition_dict`` becomes::

            {
                "condition_1": {"bot_limit": ..., "top_limit": ...,
                                "ctr_oil": ..., "ctr_gas": ...},
                ...
            }

        Returns
        -------
        None
        """
        raw = self.condition_dict
        if not raw:
            return

        first_key = next(iter(raw))
        first_value = raw[first_key]
        if condition_dict_is_mixed_form(raw):
            raise CostRecoveryInputError(
                "condition_dict cannot mix normalized and column forms",
                code="CR_CONDITION_DICT_MIXED_FORM",
            )
        if isinstance(first_value, dict):
            normalized: RCICPConditions = {}
            for name, band in raw.items():
                if not isinstance(band, dict):
                    # Unreachable after the shared mixed-form check; kept so the
                    # band value is narrowed for the type checker.
                    raise CostRecoveryInputError(
                        "condition_dict cannot mix normalized and column forms",
                        code="CR_CONDITION_DICT_MIXED_FORM",
                    )
                normalized[name] = {
                    "bot_limit": _as_float(band["bot_limit"], "bot_limit"),
                    "top_limit": _as_float(band["top_limit"], "top_limit"),
                    "ctr_oil": _as_float(band["ctr_oil"], "ctr_oil"),
                    "ctr_gas": _as_float(band["ctr_gas"], "ctr_gas"),
                }
            self.condition_dict = normalized
            return

        bot_limit = _as_float_array(raw.get("RC Bottom Limit"), "RC Bottom Limit")
        top_limit = _as_float_array(raw.get("RC Top Limit"), "RC Top Limit")
        ctr_oil = _as_float_array(raw.get("Pre Tax CTR Oil"), "Pre Tax CTR Oil")
        ctr_gas = _as_float_array(raw.get("Pre Tax CTR Gas"), "Pre Tax CTR Gas")
        n_bands = len(bot_limit)

        normalized_columns: RCICPConditions = {}
        for index in range(n_bands):
            normalized_columns[f"condition_{index + 1}"] = {
                "bot_limit": _as_float(bot_limit.item(index), "bot_limit"),
                "top_limit": _as_float(top_limit.item(index), "top_limit"),
                "ctr_oil": _as_float(ctr_oil.item(index), "ctr_oil"),
                "ctr_gas": _as_float(ctr_gas.item(index), "ctr_gas"),
            }

        self.condition_dict = normalized_columns

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
        cost = _as_float_array(sc_capital_object.cost, f"{name}.cost")
        expense_year = self._require_array(
            sc_capital_object.expense_year, f"{name}.expense_year"
        )
        # Use the shared per-row non-zero predicate so the engine and preflight
        # classify exactly the same rows.
        nonzero_mask = np.zeros(len(cost), dtype=np.bool_)
        for nonzero_index in nonzero_indices(cost):
            nonzero_mask[nonzero_index] = True

        # Exit early if all sunk costs are zeros
        if not np.any(nonzero_mask):
            return

        # Pair invalid expense years with invalid costs
        invalid = [
            f"{int(expense_year.item(index))}: {_as_float(cost.item(index), 'cost')}"
            for index in range(len(cost))
            if bool(nonzero_mask.item(index))
        ]

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
            raise CostRecoveryInputError(msg_error, code="CR_CAPITAL_SUNK_COST")

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

        oil_onstream_date, gas_onstream_date = self._require_onstream_dates()
        is_strict = self.is_strict
        if is_strict is None:
            raise CostRecoveryException("Attribute is_strict is missing")

        # Specify onstream year
        onstream_yr = min(oil_onstream_date.year, gas_onstream_date.year)

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
                is_strict=is_strict,
                onstream_year=onstream_yr,
            )

        # Check capital preonstream costs: whether PIS years < onstream year
        for preos_obj, preos_name in mapping_capital_preonstreams:
            self._check_capital_pis_years_before_onstream(
                obj_capital=preos_obj,
                obj_name=preos_name,
                is_amortization=False,
                is_strict=is_strict,
                onstream_year=onstream_yr,
            )

    def _get_depreciation(
        self,
        depr_method: DeprMethod,
        decline_factor: float | int,
        year_inflation: YearArray | None,
        inflation_rate: FloatArray | int | float,
        tax_rate: FloatArray | float,
        inflation_rate_applied_to: InflationAppliedTo | None,
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
        year_inflation : FloatArray
            Yearly inflation multipliers.
        inflation_rate : FloatArray or float or int
            Inflation rate(s) applied in depreciation calculations.
        tax_rate : FloatArray or float
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
            ),
        }

        # Define intermediate attributes:
        # "depreciations": dict and "undepreciated_assets": dict
        cost_types = ["sunk_cost", "preonstream", "postonstream"]

        depreciations: dict[str, CostSchedules] = {}
        undepreciated_assets: dict[str, CostSchedules] = {}
        for fluid in depr_mapping:
            depreciations[fluid] = {
                c: np.zeros_like(self.project_years, dtype=float) for c in cost_types
            }
            undepreciated_assets[fluid] = {
                c: np.zeros_like(self.project_years, dtype=float) for c in cost_types
            }

        # Compute depreciation per fluid per cost type
        for f, mapping in depr_mapping.items():
            for c, obj_cap in mapping:
                (depreciations[f][c], undepreciated_assets[f][c]) = (
                    obj_cap.total_depreciation_rate(
                        depr_method=depr_method,
                        decline_factor=decline_factor,
                        year_inflation=year_inflation,
                        inflation_rate=inflation_rate,
                        tax_rate=tax_rate,
                        inflation_rate_applied_to=inflation_rate_applied_to,
                    )
                )

        # Store the results as class's attributes
        self._oil_depreciations = depreciations["oil"]
        self._gas_depreciations = depreciations["gas"]
        self._oil_undepreciated_assets = undepreciated_assets["oil"]
        self._gas_undepreciated_assets = undepreciated_assets["gas"]

    def _allocate_non_depreciable_sunk_cost(
        self, non_depreciable_sunk_cost: FloatArray
    ) -> FloatArray:
        """
        Allocate non-depreciable sunk costs to the project onstream year.

        All non-depreciable sunk costs are aggregated into a single bulk value
        and assigned entirely to the earliest onstream year (oil or gas) within
        the project timeline.

        Parameters
        ----------
        non_depreciable_sunk_cost : FloatArray
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

        oil_onstream_date, gas_onstream_date = self._require_onstream_dates()

        # Identify onstream year in project years array
        onstream_yr = min([oil_onstream_date.year, gas_onstream_date.year])
        onstream_mask = _as_bool_array(
            np.equal(self.project_years, onstream_yr), "onstream year match"
        )
        match_at_onstream = np.flatnonzero(onstream_mask)
        if match_at_onstream.size != 1:
            raise CostRecoveryException(
                f"Expected one onstream year match, got {match_at_onstream.size} instead."
            )

        # Configure index of onstream year in project years array
        onstream_id = int(match_at_onstream.item(0))

        # Calculate bulk value of non-depreciable_sunk_cost
        bulk = _sum_float(non_depreciable_sunk_cost)

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

        oil_non_depreciable_sunk_cost = self._require_array(
            self._oil_non_depreciable_sunk_cost,
            "_oil_non_depreciable_sunk_cost",
        )
        gas_non_depreciable_sunk_cost = self._require_array(
            self._gas_non_depreciable_sunk_cost,
            "_gas_non_depreciable_sunk_cost",
        )
        oil_non_depreciable_preonstream = self._require_array(
            self._oil_non_depreciable_preonstream,
            "_oil_non_depreciable_preonstream",
        )
        gas_non_depreciable_preonstream = self._require_array(
            self._gas_non_depreciable_preonstream,
            "_gas_non_depreciable_preonstream",
        )
        oil_non_depreciable_postonstream = self._require_array(
            self._oil_non_depreciable_postonstream,
            "_oil_non_depreciable_postonstream",
        )
        gas_non_depreciable_postonstream = self._require_array(
            self._gas_non_depreciable_postonstream,
            "_gas_non_depreciable_postonstream",
        )

        # Construct intermediate variable "non_depreciables" which stores
        # non-depreciable costs for each fluids and each cost types
        non_depreciables = {
            "oil": {
                "sunk_cost": self._allocate_non_depreciable_sunk_cost(
                    non_depreciable_sunk_cost=oil_non_depreciable_sunk_cost
                ),
                "preonstream": oil_non_depreciable_preonstream,
                "postonstream": oil_non_depreciable_postonstream,
            },
            "gas": {
                "sunk_cost": self._allocate_non_depreciable_sunk_cost(
                    non_depreciable_sunk_cost=gas_non_depreciable_sunk_cost
                ),
                "preonstream": gas_non_depreciable_preonstream,
                "postonstream": gas_non_depreciable_postonstream,
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

        undepre_assets = [
            self._oil_undepreciated_assets,
            self._gas_undepreciated_assets,
        ]
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
                (self._gas_depreciations, self._gas_undepreciated_assets),
            ]:
                for c in cost_types:
                    depr[c][-1] += undepr[c].sum()
                    undepr[c] = np.zeros(self.project_duration, dtype=float)

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
        self._oil_ftp : FloatArray
            Total oil FTP before split.
        self._gas_ftp : FloatArray
            Total gas FTP before split.
        self._oil_ftp_ctr : FloatArray
            Contractor's share of oil FTP.
        self._gas_ftp_ctr : FloatArray
            Contractor's share of gas FTP.
        self._oil_ftp_gov : FloatArray
            Government's share of oil FTP.
        self._gas_ftp_gov : FloatArray
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
            self._oil_ftp_ctr = (
                self.oil_ctr_pretax_share * self._oil_ftp
                if self.oil_ftp_is_shared
                else np.zeros_like(self._oil_ftp, dtype=float)
            )
            self._oil_ftp_gov = self._oil_ftp - self._oil_ftp_ctr

        # Sharing condition if GAS FTP is available.
        # Fill attributes "_gas_ftp", "_gas_ftp_portion", and "_gas_ftp_gov"
        if self.gas_ftp_is_available:
            self._gas_ftp = self.gas_ftp_portion * self._gas_revenue
            self._gas_ftp_ctr = (
                self.gas_ctr_pretax_share * self._gas_ftp
                if self.gas_ftp_is_shared
                else np.zeros_like(self._gas_ftp, dtype=float)
            )
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

        oil_revenue: FloatArray = self._oil_revenue
        gas_revenue: FloatArray = self._gas_revenue

        if oil_applied:
            oil_cost_of_sales_expenditures_post_tax = self._require_array(
                self._oil_cost_of_sales_expenditures_post_tax,
                "_oil_cost_of_sales_expenditures_post_tax",
            )
            # Ensure there is at least some positive oil revenue
            if not np.any(oil_revenue > 0):
                raise CostRecoveryException(
                    "Oil revenue is zero throughout project years"
                )

            # Deduct cost of sales (post-tax) from oil revenue
            oil_revenue -= oil_cost_of_sales_expenditures_post_tax

        if gas_applied:
            gas_cost_of_sales_expenditures_post_tax = self._require_array(
                self._gas_cost_of_sales_expenditures_post_tax,
                "_gas_cost_of_sales_expenditures_post_tax",
            )
            # Ensure there is at least some positive gas revenue
            if not np.any(gas_revenue > 0):
                raise CostRecoveryException(
                    "Gas revenue is zero throughout project years"
                )

            # Deduct cost of sales (post-tax) from gas revenue
            gas_revenue -= gas_cost_of_sales_expenditures_post_tax

        self._oil_revenue: FloatArray = oil_revenue
        self._gas_revenue: FloatArray = gas_revenue

    def _get_rc_icp_pretax(self) -> None:
        """
        Compute the Pre-Tax Split using Revenue over Cost (RC) or Indonesian
        Crude Price (ICP) sliding scale and store the resulting oil and gas
        contractor pre-tax shares.

        The method expects ``condition_dict`` to be already normalized (see
        ``_normalize_condition_dict``) into the form::

            {
                "condition_1": {"bot_limit": ..., "top_limit": ...,
                                "ctr_oil": ..., "ctr_gas": ...},
                ...
            }

        Returns
        -------
        None
            Updates ``oil_ctr_pretax_share`` and ``gas_ctr_pretax_share``.
        """

        sliding = self.indicator_rc_icp_sliding
        if sliding is None:
            raise CostRecoveryException(
                "indicator_rc_icp_sliding must be provided when using a non-conventional tax split type"
            )
        sliding_arr = _as_float_array(sliding, "indicator_rc_icp_sliding")

        bands: list[RCICPBand] = list(
            _as_rc_icp_conditions(self.condition_dict).values()
        )
        bot_limit = _as_float_array([band["bot_limit"] for band in bands], "bot_limit")
        top_limit = _as_float_array([band["top_limit"] for band in bands], "top_limit")
        ctr_oil_values = _as_float_array([band["ctr_oil"] for band in bands], "ctr_oil")
        ctr_gas_values = _as_float_array([band["ctr_gas"] for band in bands], "ctr_gas")

        # Create conditions using vectorized comparisons
        conditions = (bot_limit[:, np.newaxis] < sliding_arr) & (
            sliding_arr <= top_limit[:, np.newaxis]
        )

        # Calculate corresponding values using np.select()
        condition_rows = [conditions[index] for index in range(len(conditions))]
        oil_choices = [
            np.full_like(sliding_arr, _as_float(ctr_oil_values.item(index), "ctr_oil"))
            for index in range(len(ctr_oil_values))
        ]
        gas_choices = [
            np.full_like(sliding_arr, _as_float(ctr_gas_values.item(index), "ctr_gas"))
            for index in range(len(ctr_gas_values))
        ]
        self.oil_ctr_pretax_share = _as_float_array(
            np.select(condition_rows, oil_choices, default=np.nan),
            "oil_ctr_pretax_share",
        )
        self.gas_ctr_pretax_share = _as_float_array(
            np.select(condition_rows, gas_choices, default=np.nan),
            "gas_ctr_pretax_share",
        )

    def _get_tax_by_regime(self, tax_regime: TaxRegime) -> FloatArray:
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
        tax_rate_arr : FloatArray
            A 1D array of tax rates for each project year. The tax rate is determined
            based on the tax regime and the project's starting year in relation to
            the predefined tax configurations.
        """
        tax_config = {2013: 0.44, 2016: 0.42, 2020: 0.40}

        tax_rate_arr = np.full_like(
            self.project_years, fill_value=tax_config[min(tax_config)], dtype=float
        )

        for year in tax_config:
            indices = np.array(np.where(self.project_years >= year)).ravel()
            tax_rate_arr[indices] = tax_config[year]

        if tax_regime == TaxRegime.UU_07_2021:
            tax_rate_arr = np.full_like(
                self.project_years, fill_value=0.40, dtype=float
            )

        if tax_regime == TaxRegime.UU_02_2020:
            tax_rate_arr = np.full_like(
                self.project_years, fill_value=0.42, dtype=float
            )

        if tax_regime == TaxRegime.UU_36_2008:
            tax_rate_arr = np.full_like(
                self.project_years, fill_value=0.44, dtype=float
            )

        if tax_regime == TaxRegime.NAILED_DOWN:
            if self.start_date.year >= max(tax_config):
                tax_rate_arr = np.full_like(
                    self.project_years,
                    fill_value=tax_config[max(tax_config)],
                    dtype=float,
                )
            else:
                tax_rate_arr = np.full_like(
                    self.project_years,
                    fill_value=tax_config[min(tax_config)],
                    dtype=float,
                )

        return tax_rate_arr

    def _get_ic(
        self,
        revenue: FloatArray,
        ftp: FloatArray,
        cost_alloc: FluidType,
        ic_rate: float | FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """
        The function to apply the Investment Credit (IC) into the capital cost
        based on the given IC rate.

        Parameters
        ----------
        revenue: FloatArray
            The array of revenue.
        ftp: FloatArray
            The array of First Tranche Petroleum (FTP).
        cost_alloc: FluidType
            The cost allocation of the cost.
        ic_rate: float | FloatArray
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
        capital_cost = _as_float_array(capital_class.cost, "capital cost")
        expense_year = self._require_array(
            capital_class.expense_year, "capital expense_year"
        )
        ic_applied = _as_bool_array(
            capital_class.is_ic_applied, "capital is_ic_applied"
        )
        if isinstance(ic_rate, float):
            ic_rate = np.full_like(capital_cost, ic_rate)

        # Applying the IC calculation to only true value
        ic_arr = np.where(
            ic_applied,
            capital_cost * ic_rate,
            0,
        )

        ic_expenses = np.bincount(
            expense_year - capital_class.start_year, weights=ic_arr
        )
        zeros = np.zeros(self.project_duration - len(ic_expenses))
        ic_total = np.concatenate((ic_expenses, zeros))

        # Unrec IC
        ic_unrecovered = np.cumsum(ic_total) - np.cumsum(revenue - ftp)
        ic_unrecovered = np.where(ic_unrecovered > 0, ic_unrecovered, 0.0)

        ic_unrec_next = np.roll(ic_unrecovered, 1)
        ic_unrec_next[0] = 0.0
        ic_paid = np.minimum(revenue - ftp, ic_total + ic_unrec_next)

        return (
            _as_float_array(ic_total, "ic_total"),
            _as_float_array(ic_unrecovered, "ic_unrecovered"),
            _as_float_array(ic_paid, "ic_paid"),
        )

    @staticmethod
    def calculate_cost_recovery(
        revenue: FloatArray,
        ftp: FloatArray,
        ic: FloatArray,
        depreciation: FloatArray,
        non_capital: FloatArray,
        cost_to_be_recovered: FloatArray,
        cr_cap_rate: float,
    ) -> FloatArray:
        """
        A function to get the array of cost recovery.

        Parameters
        ----------
        revenue: FloatArray
            The array containing the revenue.
        ftp: FloatArray
            The array containing the First Tranche Petroleum (FTP).
        ic: FloatArray
            The array containing the Paid Investment Credit (IC).
        depreciation: FloatArray
            The array containing the depreciated expenditures.
        non_capital: FloatArray
            The array containing the non-capital expenditures.
            non_capital expenditures is consisting of:
            Intangible, Operating Expenditures (OPEX) and Abandonment Site
            and Restoration (ASR) Expenditures.
        cost_to_be_recovered: FloatArray
            The array containing the cost to be recovered.
        cr_cap_rate: float
            The cost recovery cap rate.

        Returns
        -------
        out: FloatArray
            The array of cost recovery.
        """

        return np.minimum(
            revenue - ftp - ic,
            ((depreciation + non_capital + cost_to_be_recovered) * cr_cap_rate),
        )
        # There is possibility for a bug in Cap rate
        # return (depreciation + non_capital + cost_to_be_recovered) * cr_cap_rate

    @staticmethod
    def calculate_ets_before_transfer(
        revenue: FloatArray,
        ftp_ctr: FloatArray,
        ftp_gov: FloatArray,
        ic: FloatArray,
        cost_recovery: FloatArray,
    ) -> FloatArray:
        """
        A function to get the Equity To be Shared (ETS) before transfer.

        Parameters
        ----------
        revenue: FloatArray
            The array containing the revenue.
        ftp_ctr: FloatArray
            The array containing the Contractor's First Tranche Petroleum (FTP).
        ftp_gov: FloatArray
            The array containing the Government's FTP.
        ic: FloatArray
            The array containing the Paid Investment Credit (IC).
        cost_recovery
            The array containing the cost recovery.

        Returns
        -------
        out: FloatArray
            The array of ETS before transfer.
        """

        result = revenue - (ftp_ctr + ftp_gov) - ic - cost_recovery
        tol = np.full_like(result, fill_value=1.0e-5)

        return np.where(result < tol, 0, result)

    @staticmethod
    def _get_ets_after_transfer(
        ets_before_tf: FloatArray,
        revenue: FloatArray,
        ftp_ctr: FloatArray,
        ftp_gov: FloatArray,
        ic: FloatArray,
        cost_recovery: FloatArray,
        transferred_in: FloatArray,
        transferred_out: FloatArray,
    ) -> FloatArray:
        """
        A function to get the Equity To be Shared (ETS) before transfer.

        Parameters
        ----------
        revenue: FloatArray
            The array containing the revenue.
        ftp_ctr: FloatArray
            The array containing the Contractor's First Tranche Petroleum (FTP).
        ftp_gov: FloatArray
            The array containing the Government's FTP.
        ic: FloatArray
            The array containing the Paid Investment Credit (IC).
        cost_recovery: FloatArray
            The array containing the cost recovery.
        transferred_in: FloatArray
            The transferred cost into the cashflow.
        transferred_out: FloatArray
            The transferred cost out from the cashflow.

        Returns
        -------
        out: FloatArray
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
    def _get_equity_share(
        ets: FloatArray, pretax_ctr: float | FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        r"""
        Use to calculate equity share (ES).

        .. math::

            ES_{ctr \, (t)} &= pretax_ctr \times ETS_{(t)} \\
            ES_{gov \, (t)} &= (1 - pretax_ctr) \times ETS_{(t)}

        Parameters
        ----------
        ets : FloatArray
            equity to be shared
        pretax_ctr: float
            pretax contractor

        Returns
        -------
        out: tup
            * ES_ctr : FloatArray
                Equity share of contractor
            * ES_gov : FloatArray
                Equity share of government
        """
        contractor_share = pretax_ctr * ets
        government_share = (1 - pretax_ctr) * ets

        return contractor_share, government_share

    @staticmethod
    def _get_tax_payment(
        ctr_share: FloatArray,
        taxable_income: FloatArray,
        tax_rate: FloatArray,
        ftp_ctr: FloatArray,
        unrec_cost: FloatArray,
        ftp_tax_regime: FTPTaxRegime = FTPTaxRegime.PRE_PDJP_20_2017,
    ) -> FloatArray:
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
        ctr_share: FloatArray
            The contractor equity to be split
        taxable_income: FloatArray
            The contractor's Taxable Income (TI)
        tax_rate: float | FloatArray
            The applied effective tax rate
        ftp_ctr: FloatArray
            The contractor's First Tranche Petroleum (FTP)
        unrec_cost: FloatArray
            The unrecoverable cost
        ftp_tax_regime: FTPTaxRegime
            The regime used to calculate the tax payment

        Returns
        -------
        ctr_tax: FloatArray
            The contractor's tax payment
        """

        # Tax payment prior to PDJP 2017
        if ftp_tax_regime == FTPTaxRegime.PRE_PDJP_20_2017:
            applied_tax = _as_float_array(
                np.where(ctr_share > 0, 1.0, 0.0), "applied_tax"
            )
            cum_ti = _as_float_array(np.cumsum(taxable_income), "cum_ti")

            applied_tax_prior = _as_float_array(
                np.concatenate((np.zeros(1), applied_tax))[0:-1],
                "applied_tax_prior",
            )

            applied_tax_bool: BoolArray = applied_tax.astype(np.bool_)
            applied_tax_prior_bool: BoolArray = applied_tax_prior.astype(np.bool_)
            first_tax_year: BoolArray = np.logical_and(
                applied_tax_bool, ~applied_tax_prior_bool
            )
            subsequent_tax_year: BoolArray = np.logical_and(
                applied_tax_bool, applied_tax_prior_bool
            )
            accumulated_tax = _as_float_array(cum_ti * tax_rate, "accumulated_tax")
            current_tax = _as_float_array(taxable_income * tax_rate, "current_tax")
            ctr_tax = _as_float_array(
                np.where(
                    first_tax_year,
                    accumulated_tax,
                    np.where(subsequent_tax_year, current_tax, 0),
                ),
                "ctr_tax",
            )

        # Tax payment after PDJP 2017
        elif ftp_tax_regime == FTPTaxRegime.PDJP_20_2017:
            # Defining the array used in calculation of ftp tax payment
            ftp_cum_b2 = _as_float_array(np.cumsum(ftp_ctr), "ftp_cum_b2")
            ftp_prior_b3 = np.zeros_like(ftp_ctr, dtype=float)
            ftp_diff_b4 = np.zeros_like(ftp_ctr, dtype=float)
            ftp_considered_b5 = np.zeros_like(ftp_ctr, dtype=float)

            # Looping throughout the ftp array
            for i in range(len(ftp_ctr)):
                if i == 0:
                    ftp_prior_b3[i] = 0

                ftp_prior_b3[i] = ftp_prior_b3[i - 1] + ftp_considered_b5[i - 1]

                ftp_cumulative = _array_float_at(ftp_cum_b2, i, "ftp_cum_b2")
                prior = _array_float_at(ftp_prior_b3, i, "ftp_prior_b3")
                ftp_diff_b4[i] = np.where(
                    ftp_cumulative > prior, ftp_cumulative - prior, 0
                )
                unrecovered = _array_float_at(unrec_cost, i, "unrec_cost")
                ftp_difference = _array_float_at(ftp_diff_b4, i, "ftp_diff_b4")
                ftp_considered_b5[i] = np.where(
                    ftp_difference > unrecovered,
                    ftp_difference - unrecovered,
                    0,
                )

            # Calculating the ftp tax payment
            ftp_tax_paid = ftp_considered_b5 * tax_rate

            # Calculating the tax from taxable income
            ti_tax = (taxable_income - ftp_ctr) * tax_rate

            # Calculating ctr_tax
            ctr_tax = _as_float_array(ftp_tax_paid + ti_tax, "ctr_tax")

        else:
            ctr_tax = _as_float_array(taxable_income * tax_rate, "ctr_tax")

        return ctr_tax

    @staticmethod
    def _get_tax_payment_pdjp(
        ftp_ctr: FloatArray,
        unrec_cost: FloatArray,
        tax_rate: FloatArray,
        taxable_income: FloatArray,
        ctr_share: FloatArray,
        ddmo: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
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
        ftp_ctr : FloatArray
            Array of contractor's FTP share for each period.
        unrec_cost : FloatArray
            Array of unrecoverable costs that reduce the taxable FTP portion.
        tax_rate : FloatArray
            Array of applicable tax rates for each period (expressed as a fraction,
            e.g., 0.3 for 30%).
        taxable_income : FloatArray
            Array of contractor's taxable income before FTP adjustment.
        ctr_share : FloatArray
            Array of contractor's equity share of production for each period.
        ddmo : FloatArray
            Array of domestic market obligation (DMO) deductions or similar
            non-taxable components.

        Returns
        -------
        tax_paid : FloatArray
            Array of tax amounts actually paid by the contractor in each period,
            after accounting for unpaid tax carried forward and DMO adjustments.
        unpaid_tax : FloatArray
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
        ftp_cum_b2 = _as_float_array(np.cumsum(ftp_ctr), "ftp_cum_b2")
        ftp_prior_b3 = np.zeros_like(ftp_ctr, dtype=float)
        ftp_diff_b4 = np.zeros_like(ftp_ctr, dtype=float)
        ftp_considered_b5 = np.zeros_like(ftp_ctr, dtype=float)

        # Looping throughout the ftp array
        for i in range(len(ftp_ctr)):
            if i == 0:
                ftp_prior_b3[i] = 0

            ftp_prior_b3[i] = ftp_prior_b3[i - 1] + ftp_considered_b5[i - 1]

            ftp_cumulative = _array_float_at(ftp_cum_b2, i, "ftp_cum_b2")
            prior = _array_float_at(ftp_prior_b3, i, "ftp_prior_b3")
            ftp_diff_b4[i] = np.where(ftp_cumulative > prior, ftp_cumulative - prior, 0)
            unrecovered = _array_float_at(unrec_cost, i, "unrec_cost")
            ftp_difference = _array_float_at(ftp_diff_b4, i, "ftp_diff_b4")
            ftp_considered_b5[i] = np.where(
                ftp_difference > unrecovered,
                ftp_difference - unrecovered,
                0,
            )

        # Calculating the ftp tax payment
        ftp_tax_paid = ftp_considered_b5 * tax_rate

        # Calculating the taxable income without ftp
        taxable_income_wo_ftp = _as_float_array(
            np.where(
                taxable_income - ftp_ctr < 0,
                taxable_income - ftp_ctr + ddmo,
                taxable_income - ftp_ctr,
            ),
            "taxable_income_wo_ftp",
        )

        # Defining Taxing flag
        applied_tax = _as_float_array(np.where(ctr_share > 0, 1.0, 0.0), "applied_tax")
        cum_ti = _as_float_array(np.cumsum(taxable_income_wo_ftp), "cum_ti")

        applied_tax_prior = _as_float_array(
            np.concatenate((np.zeros(1), applied_tax))[0:-1],
            "applied_tax_prior",
        )

        applied_tax_bool: BoolArray = applied_tax.astype(np.bool_)
        applied_tax_prior_bool: BoolArray = applied_tax_prior.astype(np.bool_)
        first_tax_year: BoolArray = np.logical_and(
            applied_tax_bool, ~applied_tax_prior_bool
        )
        subsequent_tax_year: BoolArray = np.logical_and(
            applied_tax_bool, applied_tax_prior_bool
        )
        accumulated_tax = _as_float_array(cum_ti * tax_rate, "accumulated_tax")
        current_tax = _as_float_array(taxable_income_wo_ftp * tax_rate, "current_tax")
        ctr_ets_tax = _as_float_array(
            np.where(
                first_tax_year,
                accumulated_tax,
                np.where(subsequent_tax_year, current_tax, 0),
            ),
            "ctr_ets_tax",
        )

        # Tax from FTP and Contractor Equity
        ctr_tax = ftp_tax_paid + ctr_ets_tax

        # Contractor Share without FTP
        ets_and_ftp_ctr = ctr_share + ftp_ctr

        # Initiating the array of unpaid_tax and tax_paid
        unpaid_tax = np.zeros_like(taxable_income, dtype=float)
        tax_paid = np.zeros_like(taxable_income, dtype=float)

        for index in range(len(taxable_income)):
            if index == 0:
                due = _array_float_at(ctr_tax, index, "ctr_tax")
                income = _array_float_at(taxable_income, index, "taxable_income")
                take = _array_float_at(ets_and_ftp_ctr, index, "ets_and_ftp_ctr")
                dmo = _array_float_at(ddmo, index, "ddmo")
                tax_paid[index] = np.where(
                    income > due,
                    due,
                    take - dmo,
                )
                paid = _array_float_at(tax_paid, index, "tax_paid")
                unpaid_tax[index] = np.where(
                    due > paid,
                    due - paid,
                    0,
                )
            else:
                due = _array_float_at(ctr_tax, index, "ctr_tax") + _array_float_at(
                    unpaid_tax, index - 1, "unpaid_tax"
                )
                income = _array_float_at(taxable_income, index, "taxable_income")
                take = _array_float_at(ets_and_ftp_ctr, index, "ets_and_ftp_ctr")
                dmo = _array_float_at(ddmo, index, "ddmo")
                tax_paid[index] = np.where(
                    income > due,
                    due,
                    take - dmo,
                )
                paid = _array_float_at(tax_paid, index, "tax_paid")
                unpaid_tax[index] = np.where(
                    due > paid,
                    due - paid,
                    0,
                )

        return _as_float_array(tax_paid, "tax_paid"), _as_float_array(
            unpaid_tax, "unpaid_tax"
        )

    @staticmethod
    def _unpaid_and_tax_balance(
        tax_payment: FloatArray, ets_ctr: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """
        The function to get the contractor's unpaid tax and tax payment.

        Parameters
        ----------
        tax_payment: FloatArray
            The contractors tax payment
        ets_ctr: FloatArray
            The contractor's share

        Returns
        -------
        unpaid_tax: FloatArray
            The unpaid contractor's tax
        ctr_tax: FloatArray
            The contractor's tax payment
        """

        unpaid_tax = np.zeros_like(ets_ctr, dtype=float)
        unpaid_tax[0] = 0
        ctr_tax = np.zeros_like(ets_ctr, dtype=float)

        for i in range(1, len(unpaid_tax)):
            due = _array_float_at(unpaid_tax, i - 1, "unpaid_tax") + _array_float_at(
                tax_payment, i, "tax_payment"
            )
            available = _array_float_at(ets_ctr, i, "ets_ctr")
            unpaid_tax[i] = max(0, due - available)
            ctr_tax[i] = min(available, due)

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

        oil_wap_price = self._require_array(self._oil_wap_price, "_oil_wap_price")
        gas_wap_price = self._require_array(self._gas_wap_price, "_gas_wap_price")
        oil_depreciable_sunk_cost = self._require_array(
            self._oil_depreciable_sunk_cost, "_oil_depreciable_sunk_cost"
        )
        gas_depreciable_sunk_cost = self._require_array(
            self._gas_depreciable_sunk_cost, "_gas_depreciable_sunk_cost"
        )
        oil_non_depreciable_sunk_cost = self._require_array(
            self._oil_non_depreciable_sunk_cost,
            "_oil_non_depreciable_sunk_cost",
        )
        gas_non_depreciable_sunk_cost = self._require_array(
            self._gas_non_depreciable_sunk_cost,
            "_gas_non_depreciable_sunk_cost",
        )
        oil_sunk_cost = self._require_array(self._oil_sunk_cost, "_oil_sunk_cost")
        gas_sunk_cost = self._require_array(self._gas_sunk_cost, "_gas_sunk_cost")
        oil_depreciable_preonstream = self._require_array(
            self._oil_depreciable_preonstream, "_oil_depreciable_preonstream"
        )
        gas_depreciable_preonstream = self._require_array(
            self._gas_depreciable_preonstream, "_gas_depreciable_preonstream"
        )
        oil_non_depreciable_preonstream = self._require_array(
            self._oil_non_depreciable_preonstream,
            "_oil_non_depreciable_preonstream",
        )
        gas_non_depreciable_preonstream = self._require_array(
            self._gas_non_depreciable_preonstream,
            "_gas_non_depreciable_preonstream",
        )
        oil_preonstream = self._require_array(self._oil_preonstream, "_oil_preonstream")
        gas_preonstream = self._require_array(self._gas_preonstream, "_gas_preonstream")
        oil_depreciable_postonstream = self._require_array(
            self._oil_depreciable_postonstream, "_oil_depreciable_postonstream"
        )
        gas_depreciable_postonstream = self._require_array(
            self._gas_depreciable_postonstream, "_gas_depreciable_postonstream"
        )
        oil_non_depreciable_postonstream = self._require_array(
            self._oil_non_depreciable_postonstream,
            "_oil_non_depreciable_postonstream",
        )
        gas_non_depreciable_postonstream = self._require_array(
            self._gas_non_depreciable_postonstream,
            "_gas_non_depreciable_postonstream",
        )
        oil_postonstream = self._require_array(
            self._oil_postonstream, "_oil_postonstream"
        )
        gas_postonstream = self._require_array(
            self._gas_postonstream, "_gas_postonstream"
        )

        # Attributes associated with consolidated lifting
        self._consolidated_lifting = (
            self._oil_lifting.get_lifting_rate_ghv_arr()
            + self._gas_lifting.get_lifting_rate_ghv_arr()
        )
        self._consolidated_wap_price = oil_wap_price + gas_wap_price
        self._consolidated_revenue = self._oil_revenue + self._gas_revenue

        # Attributes associated with consolidated sunk cost
        self._consolidated_depreciable_sunk_cost = (
            oil_depreciable_sunk_cost + gas_depreciable_sunk_cost
        )
        self._consolidated_non_depreciable_sunk_cost = (
            oil_non_depreciable_sunk_cost + gas_non_depreciable_sunk_cost
        )
        self._consolidated_sunk_cost = oil_sunk_cost + gas_sunk_cost

        # Attributes associated with consolidated preonstream
        self._consolidated_depreciable_preonstream = (
            oil_depreciable_preonstream + gas_depreciable_preonstream
        )
        self._consolidated_non_depreciable_preonstream = (
            oil_non_depreciable_preonstream + gas_non_depreciable_preonstream
        )
        self._consolidated_preonstream = oil_preonstream + gas_preonstream

        # Attributes associated with consolidated postonstream
        self._consolidated_depreciable_postonstream = (
            oil_depreciable_postonstream + gas_depreciable_postonstream
        )
        self._consolidated_non_depreciable_postonstream = (
            oil_non_depreciable_postonstream + gas_non_depreciable_postonstream
        )
        self._consolidated_postonstream = oil_postonstream + gas_postonstream

        # Consolidate each typed expenditure category explicitly. These arrays
        # are consumed by public tables and summaries in this exact order.
        self._consolidated_capital_expenditures_pre_tax = self._require_array(
            self._oil_capital_expenditures_pre_tax, "_oil_capital_expenditures_pre_tax"
        ) + self._require_array(
            self._gas_capital_expenditures_pre_tax, "_gas_capital_expenditures_pre_tax"
        )
        self._consolidated_capital_indirect_tax = self._require_array(
            self._oil_capital_indirect_tax, "_oil_capital_indirect_tax"
        ) + self._require_array(
            self._gas_capital_indirect_tax, "_gas_capital_indirect_tax"
        )
        self._consolidated_capital_expenditures_post_tax = self._require_array(
            self._oil_capital_expenditures_post_tax,
            "_oil_capital_expenditures_post_tax",
        ) + self._require_array(
            self._gas_capital_expenditures_post_tax,
            "_gas_capital_expenditures_post_tax",
        )
        self._consolidated_intangible_expenditures_pre_tax = self._require_array(
            self._oil_intangible_expenditures_pre_tax,
            "_oil_intangible_expenditures_pre_tax",
        ) + self._require_array(
            self._gas_intangible_expenditures_pre_tax,
            "_gas_intangible_expenditures_pre_tax",
        )
        self._consolidated_intangible_indirect_tax = self._require_array(
            self._oil_intangible_indirect_tax, "_oil_intangible_indirect_tax"
        ) + self._require_array(
            self._gas_intangible_indirect_tax, "_gas_intangible_indirect_tax"
        )
        self._consolidated_intangible_expenditures_post_tax = self._require_array(
            self._oil_intangible_expenditures_post_tax,
            "_oil_intangible_expenditures_post_tax",
        ) + self._require_array(
            self._gas_intangible_expenditures_post_tax,
            "_gas_intangible_expenditures_post_tax",
        )
        self._consolidated_opex_expenditures_pre_tax = self._require_array(
            self._oil_opex_expenditures_pre_tax, "_oil_opex_expenditures_pre_tax"
        ) + self._require_array(
            self._gas_opex_expenditures_pre_tax, "_gas_opex_expenditures_pre_tax"
        )
        self._consolidated_opex_indirect_tax = self._require_array(
            self._oil_opex_indirect_tax, "_oil_opex_indirect_tax"
        ) + self._require_array(self._gas_opex_indirect_tax, "_gas_opex_indirect_tax")
        self._consolidated_opex_expenditures_post_tax = self._require_array(
            self._oil_opex_expenditures_post_tax, "_oil_opex_expenditures_post_tax"
        ) + self._require_array(
            self._gas_opex_expenditures_post_tax, "_gas_opex_expenditures_post_tax"
        )
        self._consolidated_asr_expenditures_pre_tax = self._require_array(
            self._oil_asr_expenditures_pre_tax, "_oil_asr_expenditures_pre_tax"
        ) + self._require_array(
            self._gas_asr_expenditures_pre_tax, "_gas_asr_expenditures_pre_tax"
        )
        self._consolidated_asr_indirect_tax = self._require_array(
            self._oil_asr_indirect_tax, "_oil_asr_indirect_tax"
        ) + self._require_array(self._gas_asr_indirect_tax, "_gas_asr_indirect_tax")
        self._consolidated_asr_expenditures_post_tax = self._require_array(
            self._oil_asr_expenditures_post_tax, "_oil_asr_expenditures_post_tax"
        ) + self._require_array(
            self._gas_asr_expenditures_post_tax, "_gas_asr_expenditures_post_tax"
        )
        self._consolidated_lbt_expenditures_pre_tax = self._require_array(
            self._oil_lbt_expenditures_pre_tax, "_oil_lbt_expenditures_pre_tax"
        ) + self._require_array(
            self._gas_lbt_expenditures_pre_tax, "_gas_lbt_expenditures_pre_tax"
        )
        self._consolidated_lbt_indirect_tax = self._require_array(
            self._oil_lbt_indirect_tax, "_oil_lbt_indirect_tax"
        ) + self._require_array(self._gas_lbt_indirect_tax, "_gas_lbt_indirect_tax")
        self._consolidated_lbt_expenditures_post_tax = self._require_array(
            self._oil_lbt_expenditures_post_tax, "_oil_lbt_expenditures_post_tax"
        ) + self._require_array(
            self._gas_lbt_expenditures_post_tax, "_gas_lbt_expenditures_post_tax"
        )
        self._consolidated_cost_of_sales_expenditures_pre_tax = self._require_array(
            self._oil_cost_of_sales_expenditures_pre_tax,
            "_oil_cost_of_sales_expenditures_pre_tax",
        ) + self._require_array(
            self._gas_cost_of_sales_expenditures_pre_tax,
            "_gas_cost_of_sales_expenditures_pre_tax",
        )
        self._consolidated_cost_of_sales_indirect_tax = self._require_array(
            self._oil_cost_of_sales_indirect_tax, "_oil_cost_of_sales_indirect_tax"
        ) + self._require_array(
            self._gas_cost_of_sales_indirect_tax, "_gas_cost_of_sales_indirect_tax"
        )
        self._consolidated_cost_of_sales_expenditures_post_tax = self._require_array(
            self._oil_cost_of_sales_expenditures_post_tax,
            "_oil_cost_of_sales_expenditures_post_tax",
        ) + self._require_array(
            self._gas_cost_of_sales_expenditures_post_tax,
            "_gas_cost_of_sales_expenditures_post_tax",
        )

        oil_total_expenditures_pre_tax = self._require_array(
            self._oil_total_expenditures_pre_tax,
            "_oil_total_expenditures_pre_tax",
        )
        gas_total_expenditures_pre_tax = self._require_array(
            self._gas_total_expenditures_pre_tax,
            "_gas_total_expenditures_pre_tax",
        )
        oil_total_indirect_tax = self._require_array(
            self._oil_total_indirect_tax, "_oil_total_indirect_tax"
        )
        gas_total_indirect_tax = self._require_array(
            self._gas_total_indirect_tax, "_gas_total_indirect_tax"
        )
        oil_total_expenditures_post_tax = self._require_array(
            self._oil_total_expenditures_post_tax,
            "_oil_total_expenditures_post_tax",
        )
        gas_total_expenditures_post_tax = self._require_array(
            self._gas_total_expenditures_post_tax,
            "_gas_total_expenditures_post_tax",
        )

        self._consolidated_total_expenditures_pre_tax = (
            oil_total_expenditures_pre_tax + gas_total_expenditures_pre_tax
        )

        self._consolidated_total_indirect_tax = (
            oil_total_indirect_tax + gas_total_indirect_tax
        )

        self._consolidated_total_expenditures_post_tax = (
            oil_total_expenditures_post_tax + gas_total_expenditures_post_tax
        )

        # Attributes associated with carry forward depreciation
        oil_carry_forward_depreciation = self._require_array(
            self._oil_carry_forward_depreciation,
            "_oil_carry_forward_depreciation",
        )
        gas_carry_forward_depreciation = self._require_array(
            self._gas_carry_forward_depreciation,
            "_gas_carry_forward_depreciation",
        )
        self._consolidated_carry_forward_depreciation = (
            oil_carry_forward_depreciation + gas_carry_forward_depreciation
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

        self._consolidated_sum_undepreciated_asset = _require_float(
            self._oil_sum_undepreciated_asset,
            "_oil_sum_undepreciated_asset",
        ) + _require_float(
            self._gas_sum_undepreciated_asset,
            "_gas_sum_undepreciated_asset",
        )

        self._consolidated_depreciation = self._require_array(
            self._oil_depreciation, "_oil_depreciation"
        ) + self._require_array(self._gas_depreciation, "_gas_depreciation")

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

        self._consolidated_unrecovered_before_transfer = self._require_array(
            self._oil_unrecovered_before_transfer,
            "_oil_unrecovered_before_transfer",
        ) + self._require_array(
            self._gas_unrecovered_before_transfer,
            "_gas_unrecovered_before_transfer",
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
            consolidated_tax_due = self._get_tax_payment(
                ctr_share=self._consolidated_contractor_share,
                taxable_income=self._consolidated_taxable_income,
                tax_rate=self._tax_rate_arr,
                ftp_ctr=self._consolidated_ftp_ctr,
                unrec_cost=self._consolidated_unrecovered_after_transfer,
                ftp_tax_regime=ftp_tax_regime,
            )
            self._consolidated_tax_due = consolidated_tax_due

            (
                self._consolidated_unpaid_tax_balance,
                self._consolidated_tax_payment,
            ) = self._unpaid_and_tax_balance(
                tax_payment=consolidated_tax_due,
                ets_ctr=self._consolidated_contractor_share,
            )

        else:
            self._consolidated_tax_payment = (
                self._oil_tax_payment + self._gas_tax_payment
            )

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

        oil_capital = self._require_array(self._oil_capital, "_oil_capital")
        gas_capital = self._require_array(self._gas_capital, "_gas_capital")
        oil_non_capital = self._require_array(self._oil_non_capital, "_oil_non_capital")
        gas_non_capital = self._require_array(self._gas_non_capital, "_gas_non_capital")
        oil_total_expenses = self._require_array(
            self._oil_total_expenses, "_oil_total_expenses"
        )
        gas_total_expenses = self._require_array(
            self._gas_total_expenses, "_gas_total_expenses"
        )
        oil_cashflow = self._require_array(self._oil_cashflow, "_oil_cashflow")
        gas_cashflow = self._require_array(self._gas_cashflow, "_gas_cashflow")
        oil_opex_expenditures_post_tax = self._require_array(
            self._oil_opex_expenditures_post_tax,
            "_oil_opex_expenditures_post_tax",
        )
        gas_opex_expenditures_post_tax = self._require_array(
            self._gas_opex_expenditures_post_tax,
            "_gas_opex_expenditures_post_tax",
        )
        oil_asr_expenditures_post_tax = self._require_array(
            self._oil_asr_expenditures_post_tax,
            "_oil_asr_expenditures_post_tax",
        )
        gas_asr_expenditures_post_tax = self._require_array(
            self._gas_asr_expenditures_post_tax,
            "_gas_asr_expenditures_post_tax",
        )
        oil_lbt_expenditures_post_tax = self._require_array(
            self._oil_lbt_expenditures_post_tax,
            "_oil_lbt_expenditures_post_tax",
        )
        gas_lbt_expenditures_post_tax = self._require_array(
            self._gas_lbt_expenditures_post_tax,
            "_gas_lbt_expenditures_post_tax",
        )

        # Attributes associated with consolidated investments
        self._consolidated_capital = oil_capital + gas_capital
        self._consolidated_non_capital = oil_non_capital + gas_non_capital
        self._consolidated_total_expenses = oil_total_expenses + gas_total_expenses

        # Attribute associated with consolidated cashflow
        self._consolidated_cashflow = oil_cashflow + gas_cashflow

        # ==============================================================
        self._consolidated_opex = (
            oil_opex_expenditures_post_tax + gas_opex_expenditures_post_tax
        )
        self._consolidated_asr = (
            oil_asr_expenditures_post_tax + gas_asr_expenditures_post_tax
        )
        self._consolidated_lbt = (
            oil_lbt_expenditures_post_tax + gas_lbt_expenditures_post_tax
        )
        self._consolidated_non_capital = oil_non_capital + gas_non_capital

    def get_transition_fiscal_inputs(self) -> tuple[float, float, FloatArray]:
        """Return the completed-run fiscal values needed by a transition."""

        self._require_run()
        return (
            _as_float(
                self._oil_sum_undepreciated_asset,
                "_oil_sum_undepreciated_asset",
            ),
            _as_float(
                self._gas_sum_undepreciated_asset,
                "_gas_sum_undepreciated_asset",
            ),
            self._tax_rate_arr,
        )

    def apply_transition_adjustment(
        self,
        *,
        contractor_net_share: FloatArray,
        cashflow: FloatArray,
        government_take: FloatArray,
        tax_payment: FloatArray,
    ) -> None:
        """Apply the consolidated fiscal profiles calculated by a transition."""

        self._consolidated_ctr_net_share = contractor_net_share
        self._consolidated_cashflow = cashflow
        self._consolidated_government_take = government_take
        self._consolidated_tax_payment = tax_payment

    @override
    def get_oil_table_columns(self) -> dict[str, TableArray]:
        """Construct the oil Cost Recovery table columns on demand."""

        # Preserve the established pre-run KeyError from empty schedules.
        depreciations_sunk_cost = self._oil_depreciations["sunk_cost"]
        depreciations_preonstream = self._oil_depreciations["preonstream"]
        depreciations_postonstream = self._oil_depreciations["postonstream"]
        non_depreciables_sunk_cost = self._oil_non_depreciables["sunk_cost"]
        non_depreciables_preonstream = self._oil_non_depreciables["preonstream"]
        non_depreciables_postonstream = self._oil_non_depreciables["postonstream"]
        cashflow = self._require_array(self._oil_cashflow, "_oil_cashflow")

        return {
            "years": self.project_years,
            "lifting": self._oil_lifting.get_lifting_rate_ghv_arr(),
            "price": self._require_array(
                self._oil_wap_price,
                "_oil_wap_price",
            ),
            "revenue": self._oil_revenue,
            "lifting_sulfur": self._sulfur_lifting.get_lifting_rate_ghv_arr(),
            "price_sulfur": self._require_array(
                self._sulfur_wap_price,
                "_sulfur_wap_price",
            ),
            "revenue_sulfur": self._sulfur_revenue,
            "lifting_electricity": (
                self._electricity_lifting.get_lifting_rate_ghv_arr()
            ),
            "price_electricity": self._require_array(
                self._electricity_wap_price,
                "_electricity_wap_price",
            ),
            "revenue_electricity": self._electricity_revenue,
            "lifting_co2": self._co2_lifting.get_lifting_rate_ghv_arr(),
            "price_co2": self._require_array(
                self._co2_wap_price,
                "_co2_wap_price",
            ),
            "revenue_co2": self._co2_revenue,
            "sunk_cost_depreciable": self._require_array(
                self._oil_depreciable_sunk_cost,
                "_oil_depreciable_sunk_cost",
            ),
            "sunk_cost_non_depreciable": self._require_array(
                self._oil_non_depreciable_sunk_cost,
                "_oil_non_depreciable_sunk_cost",
            ),
            "sunk_cost": self._require_array(
                self._oil_sunk_cost,
                "_oil_sunk_cost",
            ),
            "preonstream_depreciable": self._require_array(
                self._oil_depreciable_preonstream,
                "_oil_depreciable_preonstream",
            ),
            "preonstream_non_depreciable": self._require_array(
                self._oil_non_depreciable_preonstream,
                "_oil_non_depreciable_preonstream",
            ),
            "preonstream": self._require_array(
                self._oil_preonstream,
                "_oil_preonstream",
            ),
            "postonstream_depreciable": self._require_array(
                self._oil_depreciable_postonstream,
                "_oil_depreciable_postonstream",
            ),
            "postonstream_non_depreciable": self._require_array(
                self._oil_non_depreciable_postonstream,
                "_oil_non_depreciable_postonstream",
            ),
            "postonstream": self._require_array(
                self._oil_postonstream,
                "_oil_postonstream",
            ),
            "capital_expenditures_pre_tax": self._require_array(
                self._oil_capital_expenditures_pre_tax,
                "_oil_capital_expenditures_pre_tax",
            ),
            "intangible_expenditures_pre_tax": self._require_array(
                self._oil_intangible_expenditures_pre_tax,
                "_oil_intangible_expenditures_pre_tax",
            ),
            "opex_expenditures_pre_tax": self._require_array(
                self._oil_opex_expenditures_pre_tax,
                "_oil_opex_expenditures_pre_tax",
            ),
            "asr_expenditures_pre_tax": self._require_array(
                self._oil_asr_expenditures_pre_tax,
                "_oil_asr_expenditures_pre_tax",
            ),
            "lbt_expenditures_pre_tax": self._require_array(
                self._oil_lbt_expenditures_pre_tax,
                "_oil_lbt_expenditures_pre_tax",
            ),
            "cost_of_sales_expenditures_pre_tax": self._require_array(
                self._oil_cost_of_sales_expenditures_pre_tax,
                "_oil_cost_of_sales_expenditures_pre_tax",
            ),
            "capital_indirect_tax": self._require_array(
                self._oil_capital_indirect_tax,
                "_oil_capital_indirect_tax",
            ),
            "intangible_indirect_tax": self._require_array(
                self._oil_intangible_indirect_tax,
                "_oil_intangible_indirect_tax",
            ),
            "opex_indirect_tax": self._require_array(
                self._oil_opex_indirect_tax,
                "_oil_opex_indirect_tax",
            ),
            "asr_indirect_tax": self._require_array(
                self._oil_asr_indirect_tax,
                "_oil_asr_indirect_tax",
            ),
            "lbt_indirect_tax": self._require_array(
                self._oil_lbt_indirect_tax,
                "_oil_lbt_indirect_tax",
            ),
            "cost_of_sales_indirect_tax": self._require_array(
                self._oil_cost_of_sales_indirect_tax,
                "_oil_cost_of_sales_indirect_tax",
            ),
            "capital_postonstream": self._require_array(
                self._oil_capital_expenditures_post_tax,
                "_oil_capital_expenditures_post_tax",
            ),
            "intangible_postonstream": self._require_array(
                self._oil_intangible_expenditures_post_tax,
                "_oil_intangible_expenditures_post_tax",
            ),
            "opex_postonstream": self._require_array(
                self._oil_opex_expenditures_post_tax,
                "_oil_opex_expenditures_post_tax",
            ),
            "asr_postonstream": self._require_array(
                self._oil_asr_expenditures_post_tax,
                "_oil_asr_expenditures_post_tax",
            ),
            "lbt_postonstream": self._require_array(
                self._oil_lbt_expenditures_post_tax,
                "_oil_lbt_expenditures_post_tax",
            ),
            "cost_of_sales_postonstream": self._require_array(
                self._oil_cost_of_sales_expenditures_post_tax,
                "_oil_cost_of_sales_expenditures_post_tax",
            ),
            "expenses_capital": self._require_array(
                self._oil_capital,
                "_oil_capital",
            ),
            "expenses_non_capital": self._require_array(
                self._oil_non_capital,
                "_oil_non_capital",
            ),
            "expenses_total": self._require_array(
                self._oil_total_expenses,
                "_oil_total_expenses",
            ),
            "depreciations_sunk_cost": depreciations_sunk_cost,
            "depreciations_preonstream": depreciations_preonstream,
            "depreciations_postonstream": depreciations_postonstream,
            "non_depreciables_sunk_cost": non_depreciables_sunk_cost,
            "non_depreciables_preonstream": non_depreciables_preonstream,
            "non_depreciables_postonstream": non_depreciables_postonstream,
            "ftp": self._oil_ftp,
            "ftp_ctr": self._oil_ftp_ctr,
            "ftp_gov": self._oil_ftp_gov,
            "investment_credit": self._oil_ic_paid,
            "recoverable_cost": self._oil_recoverable_cost,
            "unrecovered_cost": self._require_array(
                self._oil_unrecovered_before_transfer,
                "_oil_unrecovered_before_transfer",
            ),
            "cost_recovery": self._oil_cost_recovery,
            "ets_before_transfer": self._oil_ets_before_transfer,
            "transfer_to_gas": self._transfer_to_gas,
            "unrec_after_transfer": self._oil_unrecovered_after_transfer,
            "cost_recovery_after_tf": self._oil_cost_recovery_after_tf,
            "ets_after_transfer": self._oil_ets_after_transfer,
            "contractor_share": self._oil_contractor_share,
            "government_share": self._oil_government_share,
            "dmo_volume": self._oil_dmo_volume,
            "dmo_fee": self._oil_dmo_fee,
            "ddmo": self._oil_ddmo,
            "taxable_income": self._oil_taxable_income,
            "tax_payment": self._oil_tax_payment,
            "contractor_net_share": self._oil_ctr_net_share,
            "government_take": self._oil_government_take,
            "cashflow": cashflow,
            "cum_cashflow": np.cumsum(cashflow),
        }

    @override
    def get_gas_table_columns(self) -> dict[str, TableArray]:
        """Construct the gas Cost Recovery table columns on demand."""

        # Preserve the established pre-run KeyError from empty schedules.
        depreciations_sunk_cost = self._gas_depreciations["sunk_cost"]
        depreciations_preonstream = self._gas_depreciations["preonstream"]
        depreciations_postonstream = self._gas_depreciations["postonstream"]
        non_depreciables_sunk_cost = self._gas_non_depreciables["sunk_cost"]
        non_depreciables_preonstream = self._gas_non_depreciables["preonstream"]
        non_depreciables_postonstream = self._gas_non_depreciables["postonstream"]
        cashflow = self._require_array(self._gas_cashflow, "_gas_cashflow")

        return {
            "years": self.project_years,
            "lifting": self._gas_lifting.get_lifting_rate_ghv_arr(),
            "price": self._require_array(
                self._gas_wap_price,
                "_gas_wap_price",
            ),
            "revenue": self._gas_revenue,
            "lifting_sulfur": self._sulfur_lifting.get_lifting_rate_ghv_arr(),
            "price_sulfur": self._require_array(
                self._sulfur_wap_price,
                "_sulfur_wap_price",
            ),
            "revenue_sulfur": self._sulfur_revenue,
            "lifting_electricity": (
                self._electricity_lifting.get_lifting_rate_ghv_arr()
            ),
            "price_electricity": self._require_array(
                self._electricity_wap_price,
                "_electricity_wap_price",
            ),
            "revenue_electricity": self._electricity_revenue,
            "lifting_co2": self._co2_lifting.get_lifting_rate_ghv_arr(),
            "price_co2": self._require_array(
                self._co2_wap_price,
                "_co2_wap_price",
            ),
            "revenue_co2": self._co2_revenue,
            "sunk_cost_depreciable": self._require_array(
                self._gas_depreciable_sunk_cost,
                "_gas_depreciable_sunk_cost",
            ),
            "sunk_cost_non_depreciable": self._require_array(
                self._gas_non_depreciable_sunk_cost,
                "_gas_non_depreciable_sunk_cost",
            ),
            "sunk_cost": self._require_array(
                self._gas_sunk_cost,
                "_gas_sunk_cost",
            ),
            "preonstream_depreciable": self._require_array(
                self._gas_depreciable_preonstream,
                "_gas_depreciable_preonstream",
            ),
            "preonstream_non_depreciable": self._require_array(
                self._gas_non_depreciable_preonstream,
                "_gas_non_depreciable_preonstream",
            ),
            "preonstream": self._require_array(
                self._gas_preonstream,
                "_gas_preonstream",
            ),
            "postonstream_depreciable": self._require_array(
                self._gas_depreciable_postonstream,
                "_gas_depreciable_postonstream",
            ),
            "postonstream_non_depreciable": self._require_array(
                self._gas_non_depreciable_postonstream,
                "_gas_non_depreciable_postonstream",
            ),
            "postonstream": self._require_array(
                self._gas_postonstream,
                "_gas_postonstream",
            ),
            "capital_expenditures_pre_tax": self._require_array(
                self._gas_capital_expenditures_pre_tax,
                "_gas_capital_expenditures_pre_tax",
            ),
            "intangible_expenditures_pre_tax": self._require_array(
                self._gas_intangible_expenditures_pre_tax,
                "_gas_intangible_expenditures_pre_tax",
            ),
            "opex_expenditures_pre_tax": self._require_array(
                self._gas_opex_expenditures_pre_tax,
                "_gas_opex_expenditures_pre_tax",
            ),
            "asr_expenditures_pre_tax": self._require_array(
                self._gas_asr_expenditures_pre_tax,
                "_gas_asr_expenditures_pre_tax",
            ),
            "lbt_expenditures_pre_tax": self._require_array(
                self._gas_lbt_expenditures_pre_tax,
                "_gas_lbt_expenditures_pre_tax",
            ),
            "cost_of_sales_expenditures_pre_tax": self._require_array(
                self._gas_cost_of_sales_expenditures_pre_tax,
                "_gas_cost_of_sales_expenditures_pre_tax",
            ),
            "capital_indirect_tax": self._require_array(
                self._gas_capital_indirect_tax,
                "_gas_capital_indirect_tax",
            ),
            "intangible_indirect_tax": self._require_array(
                self._gas_intangible_indirect_tax,
                "_gas_intangible_indirect_tax",
            ),
            "opex_indirect_tax": self._require_array(
                self._gas_opex_indirect_tax,
                "_gas_opex_indirect_tax",
            ),
            "asr_indirect_tax": self._require_array(
                self._gas_asr_indirect_tax,
                "_gas_asr_indirect_tax",
            ),
            "lbt_indirect_tax": self._require_array(
                self._gas_lbt_indirect_tax,
                "_gas_lbt_indirect_tax",
            ),
            "cost_of_sales_indirect_tax": self._require_array(
                self._gas_cost_of_sales_indirect_tax,
                "_gas_cost_of_sales_indirect_tax",
            ),
            "capital_postonstream": self._require_array(
                self._gas_capital_expenditures_post_tax,
                "_gas_capital_expenditures_post_tax",
            ),
            "intangible_postonstream": self._require_array(
                self._gas_intangible_expenditures_post_tax,
                "_gas_intangible_expenditures_post_tax",
            ),
            "opex_postonstream": self._require_array(
                self._gas_opex_expenditures_post_tax,
                "_gas_opex_expenditures_post_tax",
            ),
            "asr_postonstream": self._require_array(
                self._gas_asr_expenditures_post_tax,
                "_gas_asr_expenditures_post_tax",
            ),
            "lbt_postonstream": self._require_array(
                self._gas_lbt_expenditures_post_tax,
                "_gas_lbt_expenditures_post_tax",
            ),
            "cost_of_sales_postonstream": self._require_array(
                self._gas_cost_of_sales_expenditures_post_tax,
                "_gas_cost_of_sales_expenditures_post_tax",
            ),
            "expenses_capital": self._require_array(
                self._gas_capital,
                "_gas_capital",
            ),
            "expenses_non_capital": self._require_array(
                self._gas_non_capital,
                "_gas_non_capital",
            ),
            "expenses_total": self._require_array(
                self._gas_total_expenses,
                "_gas_total_expenses",
            ),
            "depreciations_sunk_cost": depreciations_sunk_cost,
            "depreciations_preonstream": depreciations_preonstream,
            "depreciations_postonstream": depreciations_postonstream,
            "non_depreciables_sunk_cost": non_depreciables_sunk_cost,
            "non_depreciables_preonstream": non_depreciables_preonstream,
            "non_depreciables_postonstream": non_depreciables_postonstream,
            "ftp": self._gas_ftp,
            "ftp_ctr": self._gas_ftp_ctr,
            "ftp_gov": self._gas_ftp_gov,
            "investment_credit": self._gas_ic_paid,
            "recoverable_cost": self._gas_recoverable_cost,
            "unrecovered_cost": self._require_array(
                self._gas_unrecovered_before_transfer,
                "_gas_unrecovered_before_transfer",
            ),
            "cost_recovery": self._gas_cost_recovery,
            "ets_before_transfer": self._gas_ets_before_transfer,
            "transfer_to_oil": self._transfer_to_oil,
            "unrec_after_transfer": self._gas_unrecovered_after_transfer,
            "cost_recovery_after_tf": self._gas_cost_recovery_after_tf,
            "ets_after_transfer": self._gas_ets_after_transfer,
            "contractor_share": self._gas_contractor_share,
            "government_share": self._gas_government_share,
            "dmo_volume": self._gas_dmo_volume,
            "dmo_fee": self._gas_dmo_fee,
            "ddmo": self._gas_ddmo,
            "taxable_income": self._gas_taxable_income,
            "tax_payment": self._gas_tax_payment,
            "contractor_net_share": self._gas_ctr_net_share,
            "government_take": self._gas_government_take,
            "cashflow": cashflow,
            "cum_cashflow": np.cumsum(cashflow),
        }

    @override
    def get_consolidated_table_columns(self) -> dict[str, TableArray]:
        """Construct the consolidated Cost Recovery table columns on demand."""

        # Preserve the established pre-run KeyError from empty schedules.
        depreciations_sunk_cost = self._consolidated_depreciations["sunk_cost"]
        depreciations_preonstream = self._consolidated_depreciations["preonstream"]
        depreciations_postonstream = self._consolidated_depreciations["postonstream"]
        non_depreciables_sunk_cost = self._consolidated_non_depreciables["sunk_cost"]
        non_depreciables_preonstream = self._consolidated_non_depreciables[
            "preonstream"
        ]
        non_depreciables_postonstream = self._consolidated_non_depreciables[
            "postonstream"
        ]
        cashflow = self._require_array(
            self._consolidated_cashflow, "_consolidated_cashflow"
        )

        return {
            "years": self.project_years,
            "lifting": self._require_array(
                self._consolidated_lifting,
                "_consolidated_lifting",
            ),
            "price": self._require_array(
                self._consolidated_wap_price,
                "_consolidated_wap_price",
            ),
            "revenue": self._require_array(
                self._consolidated_revenue,
                "_consolidated_revenue",
            ),
            "lifting_sulfur": self._sulfur_lifting.get_lifting_rate_ghv_arr(),
            "price_sulfur": self._require_array(
                self._sulfur_wap_price,
                "_sulfur_wap_price",
            ),
            "revenue_sulfur": self._sulfur_revenue,
            "lifting_electricity": self._electricity_lifting.get_lifting_rate_ghv_arr(),
            "price_electricity": self._require_array(
                self._electricity_wap_price,
                "_electricity_wap_price",
            ),
            "revenue_electricity": self._electricity_revenue,
            "lifting_co2": self._co2_lifting.get_lifting_rate_ghv_arr(),
            "price_co2": self._require_array(
                self._co2_wap_price,
                "_co2_wap_price",
            ),
            "revenue_co2": self._co2_revenue,
            "sunk_cost_depreciable": self._require_array(
                self._consolidated_depreciable_sunk_cost,
                "_consolidated_depreciable_sunk_cost",
            ),
            "sunk_cost_non_depreciable": self._require_array(
                self._consolidated_non_depreciable_sunk_cost,
                "_consolidated_non_depreciable_sunk_cost",
            ),
            "sunk_cost": self._require_array(
                self._consolidated_sunk_cost,
                "_consolidated_sunk_cost",
            ),
            "preonstream_depreciable": self._require_array(
                self._consolidated_depreciable_preonstream,
                "_consolidated_depreciable_preonstream",
            ),
            "preonstream_non_depreciable": self._require_array(
                self._consolidated_non_depreciable_preonstream,
                "_consolidated_non_depreciable_preonstream",
            ),
            "preonstream": self._require_array(
                self._consolidated_preonstream,
                "_consolidated_preonstream",
            ),
            "postonstream_depreciable": self._require_array(
                self._consolidated_depreciable_postonstream,
                "_consolidated_depreciable_postonstream",
            ),
            "postonstream_non_depreciable": self._require_array(
                self._consolidated_non_depreciable_postonstream,
                "_consolidated_non_depreciable_postonstream",
            ),
            "postonstream": self._require_array(
                self._consolidated_postonstream,
                "_consolidated_postonstream",
            ),
            "capital_expenditures_pre_tax": self._require_array(
                self._consolidated_capital_expenditures_pre_tax,
                "_consolidated_capital_expenditures_pre_tax",
            ),
            "intangible_expenditures_pre_tax": self._require_array(
                self._consolidated_intangible_expenditures_pre_tax,
                "_consolidated_intangible_expenditures_pre_tax",
            ),
            "opex_expenditures_pre_tax": self._require_array(
                self._consolidated_opex_expenditures_pre_tax,
                "_consolidated_opex_expenditures_pre_tax",
            ),
            "asr_expenditures_pre_tax": self._require_array(
                self._consolidated_asr_expenditures_pre_tax,
                "_consolidated_asr_expenditures_pre_tax",
            ),
            "lbt_expenditures_pre_tax": self._require_array(
                self._consolidated_lbt_expenditures_pre_tax,
                "_consolidated_lbt_expenditures_pre_tax",
            ),
            "cost_of_sales_expenditures_pre_tax": self._require_array(
                self._consolidated_cost_of_sales_expenditures_pre_tax,
                "_consolidated_cost_of_sales_expenditures_pre_tax",
            ),
            "capital_indirect_tax": self._require_array(
                self._consolidated_capital_indirect_tax,
                "_consolidated_capital_indirect_tax",
            ),
            "intangible_indirect_tax": self._require_array(
                self._consolidated_intangible_indirect_tax,
                "_consolidated_intangible_indirect_tax",
            ),
            "opex_indirect_tax": self._require_array(
                self._consolidated_opex_indirect_tax,
                "_consolidated_opex_indirect_tax",
            ),
            "asr_indirect_tax": self._require_array(
                self._consolidated_asr_indirect_tax,
                "_consolidated_asr_indirect_tax",
            ),
            "lbt_indirect_tax": self._require_array(
                self._consolidated_lbt_indirect_tax,
                "_consolidated_lbt_indirect_tax",
            ),
            "cost_of_sales_indirect_tax": self._require_array(
                self._consolidated_cost_of_sales_indirect_tax,
                "_consolidated_cost_of_sales_indirect_tax",
            ),
            "capital_postonstream": self._require_array(
                self._consolidated_capital_expenditures_post_tax,
                "_consolidated_capital_expenditures_post_tax",
            ),
            "intangible_postonstream": self._require_array(
                self._consolidated_intangible_expenditures_post_tax,
                "_consolidated_intangible_expenditures_post_tax",
            ),
            "opex_postonstream": self._require_array(
                self._consolidated_opex_expenditures_post_tax,
                "_consolidated_opex_expenditures_post_tax",
            ),
            "asr_postonstream": self._require_array(
                self._consolidated_asr_expenditures_post_tax,
                "_consolidated_asr_expenditures_post_tax",
            ),
            "lbt_postonstream": self._require_array(
                self._consolidated_lbt_expenditures_post_tax,
                "_consolidated_lbt_expenditures_post_tax",
            ),
            "cost_of_sales_postonstream": self._require_array(
                self._consolidated_cost_of_sales_expenditures_post_tax,
                "_consolidated_cost_of_sales_expenditures_post_tax",
            ),
            "expenses_capital": self._require_array(
                self._consolidated_capital,
                "_consolidated_capital",
            ),
            "expenses_non_capital": self._require_array(
                self._consolidated_non_capital,
                "_consolidated_non_capital",
            ),
            "expenses_total": self._require_array(
                self._consolidated_total_expenses,
                "_consolidated_total_expenses",
            ),
            "depreciations_sunk_cost": depreciations_sunk_cost,
            "depreciations_preonstream": depreciations_preonstream,
            "depreciations_postonstream": depreciations_postonstream,
            "non_depreciables_sunk_cost": non_depreciables_sunk_cost,
            "non_depreciables_preonstream": non_depreciables_preonstream,
            "non_depreciables_postonstream": non_depreciables_postonstream,
            "ftp": self._consolidated_ftp,
            "ftp_ctr": self._consolidated_ftp_ctr,
            "ftp_gov": self._consolidated_ftp_gov,
            "investment_credit": self._consolidated_ic_paid,
            "recoverable_cost": self._require_array(
                self._consolidated_recoverable_cost,
                "_consolidated_recoverable_cost",
            ),
            "unrecovered_cost": self._require_array(
                self._consolidated_unrecovered_before_transfer,
                "_consolidated_unrecovered_before_transfer",
            ),
            "cost_recovery": self._require_array(
                self._consolidated_cost_recovery_before_transfer,
                "_consolidated_cost_recovery_before_transfer",
            ),
            "ets_before_transfer": self._require_array(
                self._consolidated_ets_before_transfer,
                "_consolidated_ets_before_transfer",
            ),
            "unrec_after_transfer": self._require_array(
                self._consolidated_unrecovered_after_transfer,
                "_consolidated_unrecovered_after_transfer",
            ),
            "cost_recovery_after_tf": self._require_array(
                self._consolidated_cost_recovery_after_tf,
                "_consolidated_cost_recovery_after_tf",
            ),
            "ets_after_transfer": self._require_array(
                self._consolidated_ets_after_transfer,
                "_consolidated_ets_after_transfer",
            ),
            "contractor_share": self._require_array(
                self._consolidated_contractor_share,
                "_consolidated_contractor_share",
            ),
            "government_share": self._require_array(
                self._consolidated_government_share,
                "_consolidated_government_share",
            ),
            "dmo_volume": self._consolidated_dmo_volume,
            "dmo_fee": self._consolidated_dmo_fee,
            "ddmo": self._consolidated_ddmo,
            "taxable_income": self._consolidated_taxable_income,
            "tax_payment": self._consolidated_tax_payment,
            "contractor_net_share": self._consolidated_ctr_net_share,
            "contractor_take": self._consolidated_contractor_take,
            "government_take": self._consolidated_government_take,
            "cashflow": cashflow,
            "cum_cashflow": np.cumsum(cashflow),
        }

    @override
    def run(
        self,
        sulfur_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_GAS_REVENUE,
        electricity_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_OIL_REVENUE,
        co2_revenue: OtherRevenue = OtherRevenue.ADDITION_TO_GAS_REVENUE,
        vat_rate: FloatArray | float = 0.0,
        year_inflation: YearArray | None = None,
        inflation_rate: FloatArray | float = 0.0,
        inflation_rate_applied_to: InflationAppliedTo | None = InflationAppliedTo.CAPEX,
        is_dmo_end_weighted: bool = False,
        tax_regime: TaxRegime = TaxRegime.NAILED_DOWN,
        effective_tax_rate: object = None,
        ftp_tax_regime: FTPTaxRegime = FTPTaxRegime.PDJP_20_2017,
        depr_method: DeprMethod = DeprMethod.PSC_DB,
        decline_factor: float | int = 2,
        post_uu_22_year2001: bool = True,
        oil_cost_of_sales_applied: bool = False,
        gas_cost_of_sales_applied: bool = False,
        sum_undepreciated_cost: bool = False,
    ) -> None:
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
        year_inflation : FloatArray, optional
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

        self._has_run: bool = False

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

        oil_capital_pre_tax = self._require_array(
            self._oil_capital_expenditures_pre_tax,
            "_oil_capital_expenditures_pre_tax",
        )
        oil_intangible_pre_tax = self._require_array(
            self._oil_intangible_expenditures_pre_tax,
            "_oil_intangible_expenditures_pre_tax",
        )
        oil_opex_pre_tax = self._require_array(
            self._oil_opex_expenditures_pre_tax,
            "_oil_opex_expenditures_pre_tax",
        )
        oil_asr_pre_tax = self._require_array(
            self._oil_asr_expenditures_pre_tax, "_oil_asr_expenditures_pre_tax"
        )
        oil_lbt_pre_tax = self._require_array(
            self._oil_lbt_expenditures_pre_tax, "_oil_lbt_expenditures_pre_tax"
        )
        oil_sales_pre_tax = self._require_array(
            self._oil_cost_of_sales_expenditures_pre_tax,
            "_oil_cost_of_sales_expenditures_pre_tax",
        )
        gas_capital_pre_tax = self._require_array(
            self._gas_capital_expenditures_pre_tax,
            "_gas_capital_expenditures_pre_tax",
        )
        gas_intangible_pre_tax = self._require_array(
            self._gas_intangible_expenditures_pre_tax,
            "_gas_intangible_expenditures_pre_tax",
        )
        gas_opex_pre_tax = self._require_array(
            self._gas_opex_expenditures_pre_tax,
            "_gas_opex_expenditures_pre_tax",
        )
        gas_asr_pre_tax = self._require_array(
            self._gas_asr_expenditures_pre_tax, "_gas_asr_expenditures_pre_tax"
        )
        gas_lbt_pre_tax = self._require_array(
            self._gas_lbt_expenditures_pre_tax, "_gas_lbt_expenditures_pre_tax"
        )
        gas_sales_pre_tax = self._require_array(
            self._gas_cost_of_sales_expenditures_pre_tax,
            "_gas_cost_of_sales_expenditures_pre_tax",
        )

        # Total pre-tax expenditures
        self._oil_total_expenditures_pre_tax = (
            oil_capital_pre_tax
            + oil_intangible_pre_tax
            + oil_opex_pre_tax
            + oil_asr_pre_tax
            + oil_lbt_pre_tax
            + oil_sales_pre_tax
        )

        self._gas_total_expenditures_pre_tax = (
            gas_capital_pre_tax
            + gas_intangible_pre_tax
            + gas_opex_pre_tax
            + gas_asr_pre_tax
            + gas_lbt_pre_tax
            + gas_sales_pre_tax
        )

        oil_capital_tax = self._require_array(
            self._oil_capital_indirect_tax, "_oil_capital_indirect_tax"
        )
        oil_intangible_tax = self._require_array(
            self._oil_intangible_indirect_tax, "_oil_intangible_indirect_tax"
        )
        oil_opex_tax = self._require_array(
            self._oil_opex_indirect_tax, "_oil_opex_indirect_tax"
        )
        oil_asr_tax = self._require_array(
            self._oil_asr_indirect_tax, "_oil_asr_indirect_tax"
        )
        oil_lbt_tax = self._require_array(
            self._oil_lbt_indirect_tax, "_oil_lbt_indirect_tax"
        )
        oil_sales_tax = self._require_array(
            self._oil_cost_of_sales_indirect_tax,
            "_oil_cost_of_sales_indirect_tax",
        )
        gas_capital_tax = self._require_array(
            self._gas_capital_indirect_tax, "_gas_capital_indirect_tax"
        )
        gas_intangible_tax = self._require_array(
            self._gas_intangible_indirect_tax, "_gas_intangible_indirect_tax"
        )
        gas_opex_tax = self._require_array(
            self._gas_opex_indirect_tax, "_gas_opex_indirect_tax"
        )
        gas_asr_tax = self._require_array(
            self._gas_asr_indirect_tax, "_gas_asr_indirect_tax"
        )
        gas_lbt_tax = self._require_array(
            self._gas_lbt_indirect_tax, "_gas_lbt_indirect_tax"
        )
        gas_sales_tax = self._require_array(
            self._gas_cost_of_sales_indirect_tax,
            "_gas_cost_of_sales_indirect_tax",
        )

        # Total indirect taxes
        self._oil_total_indirect_tax = (
            oil_capital_tax
            + oil_intangible_tax
            + oil_opex_tax
            + oil_asr_tax
            + oil_lbt_tax
            + oil_sales_tax
        )

        self._gas_total_indirect_tax = (
            gas_capital_tax
            + gas_intangible_tax
            + gas_opex_tax
            + gas_asr_tax
            + gas_lbt_tax
            + gas_sales_tax
        )

        oil_capital_post_tax = self._require_array(
            self._oil_capital_expenditures_post_tax,
            "_oil_capital_expenditures_post_tax",
        )
        oil_intangible_post_tax = self._require_array(
            self._oil_intangible_expenditures_post_tax,
            "_oil_intangible_expenditures_post_tax",
        )
        oil_opex_post_tax = self._require_array(
            self._oil_opex_expenditures_post_tax,
            "_oil_opex_expenditures_post_tax",
        )
        oil_asr_post_tax = self._require_array(
            self._oil_asr_expenditures_post_tax, "_oil_asr_expenditures_post_tax"
        )
        oil_lbt_post_tax = self._require_array(
            self._oil_lbt_expenditures_post_tax, "_oil_lbt_expenditures_post_tax"
        )
        oil_sales_post_tax = self._require_array(
            self._oil_cost_of_sales_expenditures_post_tax,
            "_oil_cost_of_sales_expenditures_post_tax",
        )
        gas_capital_post_tax = self._require_array(
            self._gas_capital_expenditures_post_tax,
            "_gas_capital_expenditures_post_tax",
        )
        gas_intangible_post_tax = self._require_array(
            self._gas_intangible_expenditures_post_tax,
            "_gas_intangible_expenditures_post_tax",
        )
        gas_opex_post_tax = self._require_array(
            self._gas_opex_expenditures_post_tax,
            "_gas_opex_expenditures_post_tax",
        )
        gas_asr_post_tax = self._require_array(
            self._gas_asr_expenditures_post_tax, "_gas_asr_expenditures_post_tax"
        )
        gas_lbt_post_tax = self._require_array(
            self._gas_lbt_expenditures_post_tax, "_gas_lbt_expenditures_post_tax"
        )
        gas_sales_post_tax = self._require_array(
            self._gas_cost_of_sales_expenditures_post_tax,
            "_gas_cost_of_sales_expenditures_post_tax",
        )

        # Total expenditures post tax
        self._oil_total_expenditures_post_tax = (
            oil_capital_post_tax
            + oil_intangible_post_tax
            + oil_opex_post_tax
            + oil_asr_post_tax
            + oil_lbt_post_tax
            + oil_sales_post_tax
        )

        self._gas_total_expenditures_post_tax = (
            gas_capital_post_tax
            + gas_intangible_post_tax
            + gas_opex_post_tax
            + gas_asr_post_tax
            + gas_lbt_post_tax
            + gas_sales_post_tax
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
        self._oil_sum_undepreciated_asset = _as_float(
            np.sum([np.sum(v) for v in self._oil_undepreciated_assets.values()]),
            "_oil_sum_undepreciated_asset",
        )
        self._gas_sum_undepreciated_asset = _as_float(
            np.sum([np.sum(v) for v in self._gas_undepreciated_assets.values()]),
            "_gas_sum_undepreciated_asset",
        )

        self._oil_depreciation = np.sum(
            [v for v in self._oil_depreciations.values()], axis=0
        )
        self._gas_depreciation = np.sum(
            [v for v in self._gas_depreciations.values()], axis=0
        )

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
        self._oil_ets_before_transfer = self.calculate_ets_before_transfer(
            revenue=self._oil_revenue,
            ftp_ctr=self._oil_ftp_ctr,
            ftp_gov=self._oil_ftp_gov,
            ic=self._oil_ic_paid,
            cost_recovery=self._oil_cost_recovery,
        )

        self._gas_ets_before_transfer = self.calculate_ets_before_transfer(
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
        allocation: FloatArray = np.vstack((oil_allocation, gas_allocation)).T
        # non_capital = np.vstack((self._oil_non_capital, self._gas_non_capital)).T
        # depreciation = np.vstack((self._oil_depreciation, self._gas_depreciation)).T
        non_capital: FloatArray = np.vstack(
            (self._oil_non_depr_total, self._gas_non_depr_total)
        ).T
        depreciation: FloatArray = np.vstack(
            (self._oil_depr_total, self._gas_depr_total)
        ).T

        ets_pre_trf: FloatArray = np.zeros_like(allocation)
        ets_post_trf: FloatArray = np.zeros_like(allocation)
        unrec_pre_trf: FloatArray = np.zeros_like(allocation)
        unrec_post_trf: FloatArray = np.zeros_like(allocation)
        rec_pre_trf: FloatArray = np.zeros_like(allocation)
        rec_post_trf: FloatArray = np.zeros_like(allocation)
        recoverable_cost: FloatArray = np.zeros_like(allocation)
        trf_to_og: FloatArray = np.zeros_like(allocation)
        project_duration = len(self.project_years)
        prev_unrec: FloatArray = np.zeros(2)

        for i in range(project_duration):
            allocation_row: FloatArray = np.asarray(allocation[i], dtype=np.float64)
            non_capital_row: FloatArray = np.asarray(non_capital[i], dtype=np.float64)
            depreciation_row: FloatArray = np.asarray(depreciation[i], dtype=np.float64)
            recoverable_row: FloatArray = (
                non_capital_row + depreciation_row + prev_unrec
            )
            ets_row: FloatArray = np.maximum(allocation_row - recoverable_row, 0)
            unrec_row: FloatArray = np.maximum(recoverable_row - allocation_row, 0)
            rec_row: FloatArray = recoverable_row - unrec_row

            recoverable_cost[i] = recoverable_row
            ets_pre_trf[i] = ets_row
            unrec_pre_trf[i] = unrec_row
            rec_pre_trf[i] = rec_row

            trf_to: FloatArray = np.zeros(2)
            ets_after = ets_row.copy()
            unrec_after = unrec_row.copy()
            recov_after = rec_row.copy()

            oil_ets = _array_float_at(ets_row, 0, "oil_ets")
            gas_unrec = _array_float_at(unrec_row, 1, "gas_unrec")
            gas_ets = _array_float_at(ets_row, 1, "gas_ets")
            oil_unrec = _array_float_at(unrec_row, 0, "oil_unrec")
            if oil_ets > 0 and gas_unrec > 0:
                trf_to[1] = np.minimum(oil_ets, gas_unrec)
                ets_after[0] = oil_ets - trf_to[1]
                unrec_after[1] = gas_unrec - trf_to[1]
                recov_after[1] = rec_row[1] + trf_to[1]

            elif gas_ets > 0 and oil_unrec > 0:
                trf_to[0] = np.minimum(gas_ets, oil_unrec)
                ets_after[1] = gas_ets - trf_to[0]
                unrec_after[0] = oil_unrec - trf_to[0]
                recov_after[0] = rec_row[0] + trf_to[0]

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

        oil_onstream_date, gas_onstream_date = self._require_onstream_dates()
        oil_wap_price = self._require_array(self._oil_wap_price, "_oil_wap_price")
        gas_wap_price = self._require_array(self._gas_wap_price, "_gas_wap_price")

        # DMO
        self._oil_dmo_volume, self._oil_dmo_fee, self._oil_ddmo = psc_tools.get_dmo(
            onstream_date=oil_onstream_date,
            start_date=self.start_date,
            project_years=self.project_years,
            dmo_holiday_duration=self.oil_dmo_holiday_duration,
            dmo_volume_portion=self.oil_dmo_volume_portion,
            dmo_fee_portion=self.oil_dmo_fee_portion,
            lifting=self._oil_lifting,
            price=oil_wap_price,
            ctr_pretax_share=self.oil_ctr_pretax_share,
            unrecovered_cost=self._oil_unrecovered_after_transfer,
            is_dmo_end_weighted=is_dmo_end_weighted,
            ets=self._oil_ets_after_transfer,
            ctr_ets=self._oil_contractor_share,
            ctr_ftp=self._oil_ftp_ctr,
            post_uu_22_year2001=post_uu_22_year2001,
        )

        self._gas_dmo_volume, self._gas_dmo_fee, self._gas_ddmo = psc_tools.get_dmo(
            onstream_date=gas_onstream_date,
            start_date=self.start_date,
            project_years=self.project_years,
            dmo_holiday_duration=self.gas_dmo_holiday_duration,
            dmo_volume_portion=self.gas_dmo_volume_portion,
            dmo_fee_portion=self.gas_dmo_fee_portion,
            lifting=self._gas_lifting,
            price=gas_wap_price,
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
                tax_rate_arr = _as_float_array(
                    np.asarray(effective_tax_rate, dtype=np.float64),
                    "effective_tax_rate",
                )

            # A scalar rate applies uniformly across project years.
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
        self._oil_cashflow = self._oil_contractor_take - self._require_array(
            self._oil_total_expenses, "_oil_total_expenses"
        )
        self._gas_cashflow = self._gas_contractor_take - self._require_array(
            self._gas_total_expenses, "_gas_total_expenses"
        )

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

        self._has_run = True

        # Display warning messages as pandas DataFrame
        # self.warnings_to_dataframe()

    @override
    def _require_run(self) -> None:
        if not self._has_run:
            raise CostRecoverySummaryException(
                "Call run() successfully before requesting a cost recovery summary."
            )

    @override
    def get_post_run_summary_profile(self) -> PostRunSummaryProfile:
        """Return named reporting data from a completed contract run."""
        return replace(
            self._post_run_common_summary_profile(),
            contract_type="cost_recovery",
            oil_cost_recovery_after_tf=self._require_array(
                self._oil_cost_recovery_after_tf, "_oil_cost_recovery_after_tf"
            ),
            gas_cost_recovery_after_tf=self._require_array(
                self._gas_cost_recovery_after_tf, "_gas_cost_recovery_after_tf"
            ),
            oil_unrecovered_after_transfer=self._require_array(
                self._oil_unrecovered_after_transfer,
                "_oil_unrecovered_after_transfer",
            ),
            gas_unrecovered_after_transfer=self._require_array(
                self._gas_unrecovered_after_transfer,
                "_gas_unrecovered_after_transfer",
            ),
            oil_contractor_share=self._require_array(
                self._oil_contractor_share, "_oil_contractor_share"
            ),
            gas_contractor_share=self._require_array(
                self._gas_contractor_share, "_gas_contractor_share"
            ),
            oil_government_share=self._require_array(
                self._oil_government_share, "_oil_government_share"
            ),
            gas_government_share=self._require_array(
                self._gas_government_share, "_gas_government_share"
            ),
            oil_contractor_take=self._require_array(
                self._oil_contractor_take, "_oil_contractor_take"
            ),
            gas_contractor_take=self._require_array(
                self._gas_contractor_take, "_gas_contractor_take"
            ),
            oil_government_take=self._require_array(
                self._oil_government_take, "_oil_government_take"
            ),
            gas_government_take=self._require_array(
                self._gas_government_take, "_gas_government_take"
            ),
            oil_ftp_ctr=self._require_array(self._oil_ftp_ctr, "_oil_ftp_ctr"),
            gas_ftp_ctr=self._require_array(self._gas_ftp_ctr, "_gas_ftp_ctr"),
            oil_ftp_gov=self._require_array(self._oil_ftp_gov, "_oil_ftp_gov"),
            gas_ftp_gov=self._require_array(self._gas_ftp_gov, "_gas_ftp_gov"),
            oil_ddmo=self._require_array(self._oil_ddmo, "_oil_ddmo"),
            gas_ddmo=self._require_array(self._gas_ddmo, "_gas_ddmo"),
            oil_tax_payment=self._require_array(
                self._oil_tax_payment, "_oil_tax_payment"
            ),
            gas_tax_payment=self._require_array(
                self._gas_tax_payment, "_gas_tax_payment"
            ),
            oil_depreciation=self._require_array(
                self._oil_depreciation, "_oil_depreciation"
            ),
            gas_depreciation=self._require_array(
                self._gas_depreciation, "_gas_depreciation"
            ),
            oil_carry_forward_depreciation=self._require_array(
                self._oil_carry_forward_depreciation,
                "_oil_carry_forward_depreciation",
            ),
            gas_carry_forward_depreciation=self._require_array(
                self._gas_carry_forward_depreciation,
                "_gas_carry_forward_depreciation",
            ),
            consolidated_carry_forward_depreciation=self._require_array(
                self._consolidated_carry_forward_depreciation,
                "_consolidated_carry_forward_depreciation",
            ),
            oil_sum_undepreciated_asset=_require_float(
                self._oil_sum_undepreciated_asset, "_oil_sum_undepreciated_asset"
            ),
            gas_sum_undepreciated_asset=_require_float(
                self._gas_sum_undepreciated_asset, "_gas_sum_undepreciated_asset"
            ),
            consolidated_cost_recovery_after_tf=self._require_array(
                self._consolidated_cost_recovery_after_tf,
                "_consolidated_cost_recovery_after_tf",
            ),
            consolidated_unrecovered_after_transfer=self._require_array(
                self._consolidated_unrecovered_after_transfer,
                "_consolidated_unrecovered_after_transfer",
            ),
            consolidated_contractor_share=self._require_array(
                self._consolidated_contractor_share,
                "_consolidated_contractor_share",
            ),
            consolidated_government_share=self._require_array(
                self._consolidated_government_share,
                "_consolidated_government_share",
            ),
            consolidated_contractor_take=self._require_array(
                self._consolidated_contractor_take, "_consolidated_contractor_take"
            ),
            consolidated_ftp_gov=self._require_array(
                self._consolidated_ftp_gov, "_consolidated_ftp_gov"
            ),
            consolidated_ctr_net_share=self._require_array(
                self._consolidated_ctr_net_share, "_consolidated_ctr_net_share"
            ),
            consolidated_ddmo=self._require_array(
                self._consolidated_ddmo, "_consolidated_ddmo"
            ),
            consolidated_tax_payment=self._require_array(
                self._consolidated_tax_payment, "_consolidated_tax_payment"
            ),
            consolidated_government_take=self._require_array(
                self._consolidated_government_take, "_consolidated_government_take"
            ),
        )

    @override
    def get_summary(
        self,
        discount_rate: float = 0.1,
        npv_mode: NPVSelection = NPVSelection.NPV_SKK_REAL_TERMS,
        discounting_mode: DiscountingMode = DiscountingMode.END_YEAR,
        discount_rate_start_year: int | None = None,
        inflation_rate: float = 0.0,
        profitability_discounted: bool = False,
    ) -> dict[str, float | np.float64]:
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

        self._require_run()
        gas_wap_price = self._require_array(self._gas_wap_price, "_gas_wap_price")
        oil_sunk_cost = self._require_array(self._oil_sunk_cost, "_oil_sunk_cost")
        gas_sunk_cost = self._require_array(self._gas_sunk_cost, "_gas_sunk_cost")
        oil_capital_post_tax = self._require_array(
            self._oil_capital_expenditures_post_tax,
            "_oil_capital_expenditures_post_tax",
        )
        gas_capital_post_tax = self._require_array(
            self._gas_capital_expenditures_post_tax,
            "_gas_capital_expenditures_post_tax",
        )
        oil_depreciable_preonstream = self._require_array(
            self._oil_depreciable_preonstream, "_oil_depreciable_preonstream"
        )
        gas_depreciable_preonstream = self._require_array(
            self._gas_depreciable_preonstream, "_gas_depreciable_preonstream"
        )
        oil_intangible_post_tax = self._require_array(
            self._oil_intangible_expenditures_post_tax,
            "_oil_intangible_expenditures_post_tax",
        )
        gas_intangible_post_tax = self._require_array(
            self._gas_intangible_expenditures_post_tax,
            "_gas_intangible_expenditures_post_tax",
        )
        oil_opex_post_tax = self._require_array(
            self._oil_opex_expenditures_post_tax,
            "_oil_opex_expenditures_post_tax",
        )
        gas_opex_post_tax = self._require_array(
            self._gas_opex_expenditures_post_tax,
            "_gas_opex_expenditures_post_tax",
        )
        oil_asr_post_tax = self._require_array(
            self._oil_asr_expenditures_post_tax, "_oil_asr_expenditures_post_tax"
        )
        gas_asr_post_tax = self._require_array(
            self._gas_asr_expenditures_post_tax, "_gas_asr_expenditures_post_tax"
        )
        oil_lbt_post_tax = self._require_array(
            self._oil_lbt_expenditures_post_tax, "_oil_lbt_expenditures_post_tax"
        )
        gas_lbt_post_tax = self._require_array(
            self._gas_lbt_expenditures_post_tax, "_gas_lbt_expenditures_post_tax"
        )
        oil_total_indirect_tax = self._require_array(
            self._oil_total_indirect_tax, "_oil_total_indirect_tax"
        )
        gas_total_indirect_tax = self._require_array(
            self._gas_total_indirect_tax, "_gas_total_indirect_tax"
        )
        consolidated_cashflow = self._require_array(
            self._consolidated_cashflow, "_consolidated_cashflow"
        )
        oil_carry_forward_depreciation = self._require_array(
            self._oil_carry_forward_depreciation,
            "_oil_carry_forward_depreciation",
        )
        gas_carry_forward_depreciation = self._require_array(
            self._gas_carry_forward_depreciation,
            "_gas_carry_forward_depreciation",
        )
        consolidated_contractor_share = self._require_array(
            self._consolidated_contractor_share,
            "_consolidated_contractor_share",
        )
        consolidated_government_share = self._require_array(
            self._consolidated_government_share,
            "_consolidated_government_share",
        )
        consolidated_cost_recovery_after_transfer = self._require_array(
            self._consolidated_cost_recovery_after_tf,
            "_consolidated_cost_recovery_after_tf",
        )
        consolidated_unrecovered_after_transfer = self._require_array(
            self._consolidated_unrecovered_after_transfer,
            "_consolidated_unrecovered_after_transfer",
        )
        oil_capital = self._require_array(self._oil_capital, "_oil_capital")
        gas_capital = self._require_array(self._gas_capital, "_gas_capital")

        # Prepare discount rate start year
        if discount_rate_start_year is None:
            discount_rate_start_year = self.start_date.year

        # Cannot have discount rate year before the start year of the project
        if discount_rate_start_year < self.start_date.year:
            raise CostRecoverySummaryException(
                f"The discounting reference year ({discount_rate_start_year}) is before start year of the project ({self.start_date.year})."
            )

        # Cannot have discount rate year after the end year of the project
        if discount_rate_start_year > self.end_date.year:
            raise CostRecoverySummaryException(
                f"The discounting reference year ({discount_rate_start_year}) is after the end year of the project ({self.end_date.year})."
            )

        # Prepare OIL lifting summary
        oil_lifting_ghv = _as_float_array(
            self._oil_lifting.get_lifting_rate_ghv_arr(), "oil_lifting_ghv"
        )
        oil_lifting_ghv_sum = _sum_float(oil_lifting_ghv)
        oil_wap_sum = self._calc_division(
            numerator=_sum_float(self._oil_revenue),
            denominator=oil_lifting_ghv_sum,
        )

        # Prepare GAS lifting summary
        gas_lifting_ghv = _as_float_array(
            self._gas_lifting.get_lifting_rate_ghv_arr(), "gas_lifting_ghv"
        )
        gas_lifting_ghv_sum = _sum_float(gas_lifting_ghv)
        gas_wap_sum = self._calc_division(
            numerator=_sum_float(gas_wap_price * gas_lifting_ghv),
            denominator=gas_lifting_ghv_sum,
        )

        # Prepare gross revenue summary
        oil_gross_revenue_sum = _sum_float(self._oil_revenue)
        gas_gross_revenue_sum = _sum_float(self._gas_revenue)
        total_gross_revenue_sum = oil_gross_revenue_sum + gas_gross_revenue_sum

        # Prepare sunk cost summary
        sunk_cost_sum = _sum_float(oil_sunk_cost + gas_sunk_cost)

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

        preonstream_costs: dict[str, FloatArray] = {
            key: _as_float_array(
                val.expenditures_pre_tax(), f"{key}.expenditures_pre_tax"
            )
            for key, val in preonstream_map.items()
        }

        # Prepare tangible and intangible cost summary
        tangible_cost = (
            oil_capital_post_tax
            + gas_capital_post_tax
            + oil_depreciable_preonstream
            + gas_depreciable_preonstream
        )

        intangible_cost = (
            oil_intangible_post_tax
            + gas_intangible_post_tax
            + preonstream_costs["oil_intangible"]
            + preonstream_costs["gas_intangible"]
        )
        investment_cashflow = (tangible_cost + intangible_cost).astype(
            np.float64, copy=False
        )

        tangible_sum = _sum_float(tangible_cost)
        intangible_sum = _sum_float(intangible_cost)
        investment_sum = tangible_sum + intangible_sum

        # Prepare OPEX summary
        opex_cost = (
            oil_opex_post_tax
            + gas_opex_post_tax
            + preonstream_costs["oil_opex"]
            + preonstream_costs["gas_opex"]
        )

        opex_sum = _sum_float(opex_cost)

        # Prepare ASR summary
        asr_cost = (
            oil_asr_post_tax
            + gas_asr_post_tax
            + preonstream_costs["oil_asr"]
            + preonstream_costs["gas_asr"]
        )

        asr_sum = _sum_float(asr_cost)

        # Prepare LBT summary
        lbt_cost = (
            oil_lbt_post_tax
            + gas_lbt_post_tax
            + preonstream_costs["oil_lbt"]
            + preonstream_costs["gas_lbt"]
        )

        lbt_sum = _sum_float(lbt_cost)

        # Prepare indirect taxes summary
        oil_indirect_tax_sum = _sum_float(oil_total_indirect_tax)
        gas_indirect_tax_sum = _sum_float(gas_total_indirect_tax)

        # Prepare carry forward depreciation summary
        oil_carward_depreciation_sum = _sum_float(oil_carry_forward_depreciation)
        gas_carward_depreciation_sum = _sum_float(gas_carry_forward_depreciation)
        total_carward_depreciation_sum = (
            oil_carward_depreciation_sum + gas_carward_depreciation_sum
        )

        # Prepare undepreciated assets summary
        oil_undepreciated_asset_sum = sum(
            _sum_float(undepr) for undepr in self._oil_undepreciated_assets.values()
        )
        gas_undepreciated_asset_sum = sum(
            _sum_float(undepr) for undepr in self._gas_undepreciated_assets.values()
        )
        total_undepreciated_asset_sum = (
            oil_undepreciated_asset_sum + gas_undepreciated_asset_sum
        )

        # Prepare government DDMO summary
        gov_ddmo = _sum_float(self._consolidated_ddmo)

        # Prepare government take summary
        gov_take_income_sum = _sum_float(self._consolidated_tax_payment)
        gov_take_sum = _sum_float(self._consolidated_government_take)
        gov_take_over_gross_rev = self._calc_division(
            numerator=gov_take_sum, denominator=total_gross_revenue_sum
        )

        # Point-forward mode uses the cash flow from the reference year
        # onward; full-cycle modes use the full project cash flow.
        if npv_mode == NPVSelection.NPV_POINT_FORWARD:
            mask = self.project_years >= discount_rate_start_year
            ctr_irr = irr(cashflow=consolidated_cashflow[mask])
        else:
            ctr_irr = irr(cashflow=consolidated_cashflow)

        # Calculate NPV
        # NPV method: SKK real terms
        if npv_mode == NPVSelection.NPV_SKK_REAL_TERMS:
            # Contractor NPV
            ctr_npv = npv_skk_real_terms(
                cashflow=consolidated_cashflow,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                discounting_mode=discounting_mode,
            )

            # Contractor investment NPV
            investment_npv = npv_skk_real_terms(
                cashflow=investment_cashflow,
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
                cashflow=consolidated_cashflow,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                discounting_mode=discounting_mode,
            )

            # Contractor investment NPV
            investment_npv = npv_skk_nominal_terms(
                cashflow=investment_cashflow,
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
                cashflow=consolidated_cashflow,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                discounting_mode=discounting_mode,
            )

            # Contractor investment NPV
            investment_npv = npv_nominal_terms(
                cashflow=investment_cashflow,
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
                cashflow=consolidated_cashflow,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                inflation_rate=inflation_rate,
                discounting_mode=discounting_mode,
            )

            # Contractor investment NPV
            investment_npv = npv_real_terms(
                cashflow=investment_cashflow,
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
                cashflow=consolidated_cashflow,
                cashflow_years=self.project_years,
                discount_rate=discount_rate,
                reference_year=discount_rate_start_year,
                discounting_mode=discounting_mode,
            )

            # Contractor investment NPV
            investment_npv = npv_point_forward(
                cashflow=investment_cashflow,
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
            ctr_pv_ratio = self._calc_division(
                numerator=ctr_npv, denominator=investment_npv
            )

        else:
            investment_pi = _sum_float(tangible_cost + intangible_cost)
            ctr_pv_ratio = self._calc_division(
                numerator=ctr_npv, denominator=investment_pi
            )

        ctr_pi = 1 + ctr_pv_ratio

        # Contractor POT
        ctr_pot = pot_psc(
            cashflow=consolidated_cashflow,
            cashflow_years=self.project_years,
            reference_year=discount_rate_start_year,
        )

        # Prepare government FTP share summary
        gov_ftp_share_sum = _sum_float(self._consolidated_ftp_gov)

        # Prepare government equity share summary
        gov_equity_share_sum = _sum_float(consolidated_government_share)

        # Prepare cost recovery summary
        cost_recovery_sum = _sum_float(
            consolidated_cost_recovery_after_transfer,
        )
        cost_recovery_over_gross_rev = self._calc_division(
            numerator=cost_recovery_sum, denominator=total_gross_revenue_sum
        )

        # Prepare unrecoverable cost summary
        unrec_cost = _last_float(consolidated_unrecovered_after_transfer)
        unrec_over_costrec = self._calc_division(
            numerator=unrec_cost, denominator=cost_recovery_sum
        )
        unrec_over_gross_rev = self._calc_division(
            numerator=unrec_cost, denominator=total_gross_revenue_sum
        )

        # Prepare contractor gross share summary
        ctr_gross_share_sum = _sum_float(consolidated_contractor_share)

        # Prepare government gross share summary
        gov_gross_share_sum = _sum_float(consolidated_government_share)

        # Prepare contractor net share summary
        ctr_net_share_sum = _sum_float(self._consolidated_ctr_net_share)
        ctr_net_share_over_gross_share = self._calc_division(
            numerator=ctr_net_share_sum, denominator=total_gross_revenue_sum
        )

        # Net cash flow is a full-lifecycle project metric. Point-forward
        # selection applies to valuation metrics, not this reported total.
        ctr_net_cashflow_sum = _sum_float(consolidated_cashflow)
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
            "oil_capex": oil_capital.sum(),
            "gas_capex": gas_capital.sum(),
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
