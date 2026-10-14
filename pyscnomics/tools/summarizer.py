from dataclasses import dataclass, field
from typing import cast
import numpy as np
import numpy.typing as npt

from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.summary_profile import PostRunSummaryProfile
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.gross_split import GrossSplit
from pyscnomics.contracts.transition import Transition
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.selection import NPVSelection, DiscountingMode
from pyscnomics.econ.indicator import (
    irr,
    npv_nominal_terms,
    npv_real_terms,
    npv_skk_nominal_terms,
    npv_skk_real_terms,
    npv_point_forward,
    pot_psc,
)

FloatArray = npt.NDArray[np.float64]
YearArray = npt.NDArray[np.int64]
SummaryArray = FloatArray | YearArray
SummaryValue = SummaryArray | float | np.float64 | None
ContractSummary = dict[str, SummaryValue]
ContractSummaryMap = dict[int, ContractSummary]
ParsedSummary = dict[str, SummaryArray]
ParsedSummaryMap = dict[int, ParsedSummary]


def _require_array(value: FloatArray | None, name: str) -> FloatArray:
    """Require a named profile array from a completed contract run."""
    if value is None:
        raise RuntimeError(f"{name} is unavailable before the contract is run")
    return value


def _require_lifting(value: Lifting | None, name: str) -> Lifting:
    """Require named lifting data from a completed contract run."""
    if value is None:
        raise RuntimeError(f"{name} is unavailable before the contract is run")
    return value


def _require_years(value: YearArray | None) -> YearArray:
    """Require the typed time axis from a completed contract run."""
    if value is None:
        raise RuntimeError("project_years is unavailable before the contract is run")
    return value


def _float_array(value: SummaryValue, name: str = "summary value") -> FloatArray:
    """Normalize an internal numeric profile to the summary's float-array shape."""
    if not isinstance(value, np.ndarray) or value.ndim != 1:
        raise TypeError(f"{name} must be a one-dimensional numeric array")
    return np.asarray(value, dtype=np.float64)


def _year_array(value: SummaryValue) -> YearArray:
    """Require a one-dimensional project-year axis from an internal profile."""
    if not isinstance(value, np.ndarray) or value.ndim != 1:
        raise TypeError("years must be a one-dimensional array")
    return np.asarray(value, dtype=np.int64)


def _summary_scalar(value: SummaryValue, name: str) -> float:
    """Require a scalar summary value where a terminal balance is expected."""
    if isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a scalar")
    if value is None:
        raise TypeError(f"{name} is unavailable")
    return float(value)


def _sum_float(values: FloatArray) -> np.float64:
    """Sum a normalized float array while retaining NumPy reduction semantics."""
    return np.sum(values)


def _ratio_float(
    numerator: float | np.float64, denominator: float | np.float64
) -> float:
    """Return a plain-scalar ratio, using 0.0 for a zero denominator.

    ``np.divide`` with a scalar ``out`` array returns a zero-dimensional array,
    which the JSON response boundary cannot serialize. The executive-summary
    ratios are scalar, so they are reduced to a Python ``float`` here.
    """
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


@dataclass
class ExecutiveSummary:
    lifting_oil: float | None = field(default=None, init=False, repr=False)
    oil_wap: float | None = field(default=None, init=False, repr=False)
    lifting_gas: float | None = field(default=None, init=False, repr=False)
    gas_wap: float | None = field(default=None, init=False, repr=False)
    gross_revenue: float | None = field(default=None, init=False, repr=False)
    gross_revenue_oil: float | None = field(default=None, init=False, repr=False)
    gross_revenue_gas: float | None = field(default=None, init=False, repr=False)
    ctr_gross_share: float | None = field(default=None, init=False, repr=False)
    gov_gross_share: float | None = field(default=None, init=False, repr=False)
    investment: float | None = field(default=None, init=False, repr=False)
    oil_capex: float | None = field(default=None, init=False, repr=False)
    gas_capex: float | None = field(default=None, init=False, repr=False)
    sunk_cost: float | None = field(default=None, init=False, repr=False)
    tangible: float | None = field(default=None, init=False, repr=False)
    intangible: float | None = field(default=None, init=False, repr=False)
    opex_asr_lbt: float | None = field(default=None, init=False, repr=False)
    opex: float | None = field(default=None, init=False, repr=False)
    asr: float | None = field(default=None, init=False, repr=False)
    lbt: float | None = field(default=None, init=False, repr=False)
    cost_recovery_deductible_cost: float | None = field(
        default=None, init=False, repr=False
    )
    cost_recovery_over_gross_rev: float | None = field(
        default=None, init=False, repr=False
    )
    unrec_cost: float | None = field(default=None, init=False, repr=False)
    unrec_over_costrec: float | None = field(default=None, init=False, repr=False)
    unrec_over_gross_rev: float | None = field(default=None, init=False, repr=False)
    ctr_net_share: float | None = field(default=None, init=False, repr=False)
    ctr_net_share_over_gross_share: float | None = field(
        default=None, init=False, repr=False
    )
    ctr_net_cashflow: float | None = field(default=None, init=False, repr=False)
    ctr_net_cashflow_over_gross_rev: float | None = field(
        default=None, init=False, repr=False
    )
    ctr_npv: float | None = field(default=None, init=False, repr=False)
    ctr_npv_sunk_cost_pooled: float | None = field(default=None, init=False, repr=False)
    ctr_irr: float | None = field(default=None, init=False, repr=False)
    ctr_irr_sunk_cost_pooled: float | None = field(default=None, init=False, repr=False)
    ctr_pot: float | None = field(default=None, init=False, repr=False)
    ctr_pv_ratio: float | None = field(default=None, init=False, repr=False)
    ctr_pi: float | None = field(default=None, init=False, repr=False)
    gov_ftp_share: float | None = field(default=None, init=False, repr=False)
    gov_equity_share: float | None = field(default=None, init=False, repr=False)
    gov_ddmo: float | None = field(default=None, init=False, repr=False)
    gov_tax_income: float | None = field(default=None, init=False, repr=False)
    gov_take: float | None = field(default=None, init=False, repr=False)
    gov_take_over_gross_rev: float | None = field(default=None, init=False, repr=False)
    gov_take_npv: float | None = field(default=None, init=False, repr=False)
    undepreciated_asset_oil: float | None = field(default=None, init=False, repr=False)
    undepreciated_asset_gas: float | None = field(default=None, init=False, repr=False)
    undepreciated_asset_total: float | None = field(
        default=None, init=False, repr=False
    )
    total_indirect_taxes: float | None = field(default=None, init=False, repr=False)
    oil_indirect_taxes: float | None = field(default=None, init=False, repr=False)
    gas_indirect_taxes: float | None = field(default=None, init=False, repr=False)


@dataclass
class Summary:
    contract: tuple[BaseProject | CostRecovery | GrossSplit | Transition, ...]
    _profiles: tuple[PostRunSummaryProfile, ...] = field(
        default=(), init=False, repr=False
    )
    reference_year: int | None = field(default=None)
    inflation_rate: float = field(default=0.0)
    discount_rate: float = field(default=0.1)
    npv_mode: NPVSelection = field(default=NPVSelection.NPV_SKK_REAL_TERMS)
    discounting_mode: DiscountingMode = field(default=DiscountingMode.END_YEAR)
    profitability_discounted: bool = True

    # Executive Summary
    executive_summary: ExecutiveSummary | None = field(
        default=None, init=False, repr=False
    )

    # Economic Indicator
    ctr_npv: float | None = field(default=None, init=False, repr=False)
    investment_npv: float | None = field(default=None, init=False, repr=False)
    gov_take_npv: float | None = field(default=None, init=False, repr=False)
    pot: float | None = field(default=None, init=False, repr=False)
    pv_ratio: float | None = field(default=None, init=False, repr=False)
    pi: float | None = field(default=None, init=False, repr=False)
    irr: float | None = field(default=None, init=False, repr=False)

    # Years
    years: YearArray | None = field(default=None, init=False, repr=False)

    # Lifting Attributes
    oil_lifting: FloatArray | None = field(default=None, init=False, repr=False)
    gas_lifting: FloatArray | None = field(default=None, init=False, repr=False)
    sulfur_lifting: FloatArray | None = field(default=None, init=False, repr=False)
    electricity_lifting: FloatArray | None = field(default=None, init=False, repr=False)
    co2_lifting: FloatArray | None = field(default=None, init=False, repr=False)

    # Revenue
    oil_revenue: FloatArray | None = field(default=None, init=False, repr=False)
    gas_revenue: FloatArray | None = field(default=None, init=False, repr=False)
    sulfur_revenue: FloatArray | None = field(default=None, init=False, repr=False)
    electricity_revenue: FloatArray | None = field(default=None, init=False, repr=False)
    co2_revenue: FloatArray | None = field(default=None, init=False, repr=False)

    # Cost Expenditures Pre Tax
    oil_capital_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    oil_intangible_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    oil_opex_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    oil_asr_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    oil_lbt_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    oil_cost_of_sales_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    gas_capital_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    gas_intangible_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    gas_opex_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    gas_asr_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    gas_lbt_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    gas_cost_of_sales_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_capital_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_intangible_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_opex_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_asr_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_lbt_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_cost_of_sales_expenditures_pre_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )

    # Cost Indirect Tax
    oil_capital_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    oil_intangible_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    oil_opex_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    oil_asr_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    oil_lbt_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    oil_cost_of_sales_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    gas_capital_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    gas_intangible_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    gas_opex_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    gas_asr_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    gas_lbt_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    gas_cost_of_sales_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_capital_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_intangible_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_opex_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_asr_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_lbt_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_cost_of_sales_indirect_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )

    # Cost Post Tax
    oil_capital_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    oil_intangible_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    oil_opex_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    oil_asr_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    oil_lbt_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    oil_cost_of_sales_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    gas_capital_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    gas_intangible_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    gas_opex_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    gas_asr_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    gas_lbt_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    gas_cost_of_sales_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_capital_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_intangible_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_opex_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_asr_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_lbt_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_cost_of_sales_expenditures_post_tax: FloatArray | None = field(
        default=None, init=False, repr=False
    )

    # Contract Terms Cost
    oil_depreciable: FloatArray | None = field(default=None, init=False, repr=False)
    oil_intangible: FloatArray | None = field(default=None, init=False, repr=False)
    oil_opex: FloatArray | None = field(default=None, init=False, repr=False)
    oil_asr: FloatArray | None = field(default=None, init=False, repr=False)
    oil_lbt: FloatArray | None = field(default=None, init=False, repr=False)
    oil_depreciation: FloatArray | None = field(default=None, init=False, repr=False)
    oil_non_capital: FloatArray | None = field(default=None, init=False, repr=False)
    oil_sunk_cost: FloatArray | None = field(default=None, init=False, repr=False)
    oil_undepreciated_asset: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    oil_carry_forward_depreciation: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    gas_depreciable: FloatArray | None = field(default=None, init=False, repr=False)
    gas_intangible: FloatArray | None = field(default=None, init=False, repr=False)
    gas_opex: FloatArray | None = field(default=None, init=False, repr=False)
    gas_asr: FloatArray | None = field(default=None, init=False, repr=False)
    gas_lbt: FloatArray | None = field(default=None, init=False, repr=False)
    gas_depreciation: FloatArray | None = field(default=None, init=False, repr=False)
    gas_non_capital: FloatArray | None = field(default=None, init=False, repr=False)
    gas_sunk_cost: FloatArray | None = field(default=None, init=False, repr=False)
    gas_undepreciated_asset: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    gas_carry_forward_depreciation: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_depreciable: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_intangible: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_opex: FloatArray | None = field(default=None, init=False, repr=False)
    consolidated_asr: FloatArray | None = field(default=None, init=False, repr=False)
    consolidated_lbt: FloatArray | None = field(default=None, init=False, repr=False)
    consolidated_depreciation: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_non_capital: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_sunk_cost: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_undepreciated_asset: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_carry_forward_depreciation: FloatArray | None = field(
        default=None, init=False, repr=False
    )

    # PSC terms
    oil_costrecovery_or_deductible_cost: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    oil_unrecoverable_cost_or_carryforward_cost: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    oil_ctr_share: FloatArray | None = field(default=None, init=False, repr=False)
    oil_gov_share: FloatArray | None = field(default=None, init=False, repr=False)
    oil_effective_tax_payment: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    oil_ddmo: FloatArray | None = field(default=None, init=False, repr=False)
    oil_ctr_take: FloatArray | None = field(default=None, init=False, repr=False)
    oil_government_take: FloatArray | None = field(default=None, init=False, repr=False)
    oil_ctr_cashflow: FloatArray | None = field(default=None, init=False, repr=False)
    gas_costrecovery_or_deductible_cost: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    gas_unrecoverable_cost_or_carryforward_cost: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    gas_ctr_share: FloatArray | None = field(default=None, init=False, repr=False)
    gas_gov_share: FloatArray | None = field(default=None, init=False, repr=False)
    gas_effective_tax_payment: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    gas_ddmo: FloatArray | None = field(default=None, init=False, repr=False)
    gas_ctr_take: FloatArray | None = field(default=None, init=False, repr=False)
    gas_government_take: FloatArray | None = field(default=None, init=False, repr=False)
    gas_ctr_cashflow: FloatArray | None = field(default=None, init=False, repr=False)
    consolidated_costrecovery_or_deductible_cost: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_unrecoverable_cost_or_carryforward_cost: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_ctr_share: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_ctr_net_share: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_gov_share: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_effective_tax_payment: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_ddmo: FloatArray | None = field(default=None, init=False, repr=False)
    consolidated_ctr_take: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_government_take: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_ctr_cashflow: FloatArray | None = field(
        default=None, init=False, repr=False
    )

    # Cost Recovery Terms
    oil_ftp_ctr: FloatArray | None = field(default=None, init=False, repr=False)
    gas_ftp_ctr: FloatArray | None = field(default=None, init=False, repr=False)
    oil_ftp_gov: FloatArray | None = field(default=None, init=False, repr=False)
    gas_ftp_gov: FloatArray | None = field(default=None, init=False, repr=False)
    consolidated_ctr_ftp: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    consolidated_gov_ftp: FloatArray | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self):
        # Defining overall project years
        self._profiles = tuple(
            contract.get_post_run_summary_profile() for contract in self.contract
        )
        project_years = np.concatenate(
            [_require_years(profile.project_years) for profile in self._profiles]
        )
        min_year = int(np.min(project_years))
        max_year = int(np.max(project_years))

        self.years = np.arange(min_year, max_year + 1, 1)

        # Condition when the reference year is None
        if self.reference_year is None:
            self.reference_year = min_year
        else:
            pass

    @staticmethod
    def _array_or_zeros(value: FloatArray | None, years: YearArray) -> FloatArray:
        """Return a profile array, substituting a documented zero profile when absent."""
        if value is None:
            return np.zeros_like(years, dtype=np.float64)
        return np.asarray(value, dtype=np.float64)

    @staticmethod
    def _scalar_or_zero(value: float | np.float64 | None) -> float:
        """Return a profile scalar, treating absence as the documented zero value."""
        return 0.0 if value is None else float(value)

    @staticmethod
    def _fluid_non_capital(profile: PostRunSummaryProfile, fluid: str) -> FloatArray:
        """Rebuild non-capital spend from the per-case post-tax cost components.

        A contract's non-capital spend is the sum of its post-tax intangible,
        opex, ASR, LBT and cost-of-sales components (the same components the
        per-case summary and the cost-recovery/gross-split tables expose). The
        multi-contract profile does not snapshot the aggregate, so it is derived
        here rather than read from a missing field.
        """
        total = np.zeros_like(profile.project_years, dtype=np.float64)
        for category in ("intangible", "opex", "asr", "lbt", "cost_of_sales"):
            field_name = f"{fluid}_{category}_expenditures_post_tax"
            value = cast("FloatArray | None", getattr(profile, field_name))
            total = total + _require_array(value, f"_{field_name}")
        return total

    def _get_lifting_dict(self) -> ContractSummaryMap:
        return {
            idx: {
                "years": contract.project_years,
                "oil_lifting": _require_lifting(
                    contract.oil_lifting, "_oil_lifting"
                ).get_lifting_rate_ghv_arr(),
                "gas_lifting": _require_lifting(
                    contract.gas_lifting, "_gas_lifting"
                ).get_lifting_rate_ghv_arr(),
                "sulfur_lifting": _require_lifting(
                    contract.sulfur_lifting, "_sulfur_lifting"
                ).get_lifting_rate_ghv_arr(),
                "electricity_lifting": _require_lifting(
                    contract.electricity_lifting, "_electricity_lifting"
                ).get_lifting_rate_ghv_arr(),
                "co2_lifting": _require_lifting(
                    contract.co2_lifting, "_co2_lifting"
                ).get_lifting_rate_ghv_arr(),
            }
            for idx, contract in enumerate(self._profiles)
        }

    def _get_revenue_dict(self) -> ContractSummaryMap:
        return {
            idx: {
                "years": contract.project_years,
                "oil_revenue": _require_array(contract.oil_revenue, "_oil_revenue"),
                "gas_revenue": _require_array(contract.gas_revenue, "_gas_revenue"),
                "sulfur_revenue": _require_array(
                    _require_array(contract.sulfur_revenue, "_sulfur_revenue"),
                    "_sulfur_revenue",
                ),
                "electricity_revenue": _require_array(
                    _require_array(
                        contract.electricity_revenue, "_electricity_revenue"
                    ),
                    "_electricity_revenue",
                ),
                "co2_revenue": _require_array(contract.co2_revenue, "_co2_revenue"),
            }
            for idx, contract in enumerate(self._profiles)
        }

    def _get_cost_expenditure_pretax_dict(self) -> ContractSummaryMap:
        return {
            idx: {
                "years": contract.project_years,
                "oil_capital_expenditures_pre_tax": _require_array(
                    _require_array(
                        contract.oil_capital_expenditures_pre_tax,
                        "_oil_capital_expenditures_pre_tax",
                    ),
                    "_oil_capital_expenditures_pre_tax",
                ),
                "oil_intangible_expenditures_pre_tax": _require_array(
                    _require_array(
                        contract.oil_intangible_expenditures_pre_tax,
                        "_oil_intangible_expenditures_pre_tax",
                    ),
                    "_oil_intangible_expenditures_pre_tax",
                ),
                "oil_opex_expenditures_pre_tax": _require_array(
                    _require_array(
                        contract.oil_opex_expenditures_pre_tax,
                        "_oil_opex_expenditures_pre_tax",
                    ),
                    "_oil_opex_expenditures_pre_tax",
                ),
                "oil_asr_expenditures_pre_tax": _require_array(
                    _require_array(
                        contract.oil_asr_expenditures_pre_tax,
                        "_oil_asr_expenditures_pre_tax",
                    ),
                    "_oil_asr_expenditures_pre_tax",
                ),
                "oil_lbt_expenditures_pre_tax": _require_array(
                    _require_array(
                        contract.oil_lbt_expenditures_pre_tax,
                        "_oil_lbt_expenditures_pre_tax",
                    ),
                    "_oil_lbt_expenditures_pre_tax",
                ),
                "oil_cost_of_sales_expenditures_pre_tax": _require_array(
                    _require_array(
                        contract.oil_cost_of_sales_expenditures_pre_tax,
                        "_oil_cost_of_sales_expenditures_pre_tax",
                    ),
                    "_oil_cost_of_sales_expenditures_pre_tax",
                ),
                "gas_capital_expenditures_pre_tax": _require_array(
                    _require_array(
                        contract.gas_capital_expenditures_pre_tax,
                        "_gas_capital_expenditures_pre_tax",
                    ),
                    "_gas_capital_expenditures_pre_tax",
                ),
                "gas_intangible_expenditures_pre_tax": _require_array(
                    _require_array(
                        contract.gas_intangible_expenditures_pre_tax,
                        "_gas_intangible_expenditures_pre_tax",
                    ),
                    "_gas_intangible_expenditures_pre_tax",
                ),
                "gas_opex_expenditures_pre_tax": _require_array(
                    _require_array(
                        contract.gas_opex_expenditures_pre_tax,
                        "_gas_opex_expenditures_pre_tax",
                    ),
                    "_gas_opex_expenditures_pre_tax",
                ),
                "gas_asr_expenditures_pre_tax": _require_array(
                    _require_array(
                        contract.gas_asr_expenditures_pre_tax,
                        "_gas_asr_expenditures_pre_tax",
                    ),
                    "_gas_asr_expenditures_pre_tax",
                ),
                "gas_lbt_expenditures_pre_tax": _require_array(
                    _require_array(
                        contract.gas_lbt_expenditures_pre_tax,
                        "_gas_lbt_expenditures_pre_tax",
                    ),
                    "_gas_lbt_expenditures_pre_tax",
                ),
                "gas_cost_of_sales_expenditures_pre_tax": _require_array(
                    _require_array(
                        contract.gas_cost_of_sales_expenditures_pre_tax,
                        "_gas_cost_of_sales_expenditures_pre_tax",
                    ),
                    "_gas_cost_of_sales_expenditures_pre_tax",
                ),
                "consolidated_capital_expenditures_pre_tax": _require_array(
                    _require_array(
                        contract.oil_capital_expenditures_pre_tax,
                        "_oil_capital_expenditures_pre_tax",
                    ),
                    "_oil_capital_expenditures_pre_tax",
                )
                + _require_array(
                    _require_array(
                        contract.gas_capital_expenditures_pre_tax,
                        "_gas_capital_expenditures_pre_tax",
                    ),
                    "_gas_capital_expenditures_pre_tax",
                ),
                "consolidated_intangible_expenditures_pre_tax": _require_array(
                    _require_array(
                        contract.oil_intangible_expenditures_pre_tax,
                        "_oil_intangible_expenditures_pre_tax",
                    ),
                    "_oil_intangible_expenditures_pre_tax",
                )
                + _require_array(
                    _require_array(
                        contract.gas_intangible_expenditures_pre_tax,
                        "_gas_intangible_expenditures_pre_tax",
                    ),
                    "_gas_intangible_expenditures_pre_tax",
                ),
                "consolidated_opex_expenditures_pre_tax": _require_array(
                    _require_array(
                        contract.oil_opex_expenditures_pre_tax,
                        "_oil_opex_expenditures_pre_tax",
                    ),
                    "_oil_opex_expenditures_pre_tax",
                )
                + _require_array(
                    _require_array(
                        contract.gas_opex_expenditures_pre_tax,
                        "_gas_opex_expenditures_pre_tax",
                    ),
                    "_gas_opex_expenditures_pre_tax",
                ),
                "consolidated_asr_expenditures_pre_tax": _require_array(
                    _require_array(
                        contract.oil_asr_expenditures_pre_tax,
                        "_oil_asr_expenditures_pre_tax",
                    ),
                    "_oil_asr_expenditures_pre_tax",
                )
                + _require_array(
                    _require_array(
                        contract.gas_asr_expenditures_pre_tax,
                        "_gas_asr_expenditures_pre_tax",
                    ),
                    "_gas_asr_expenditures_pre_tax",
                ),
                "consolidated_lbt_expenditures_pre_tax": _require_array(
                    _require_array(
                        contract.oil_lbt_expenditures_pre_tax,
                        "_oil_lbt_expenditures_pre_tax",
                    ),
                    "_oil_lbt_expenditures_pre_tax",
                )
                + _require_array(
                    _require_array(
                        contract.gas_lbt_expenditures_pre_tax,
                        "_gas_lbt_expenditures_pre_tax",
                    ),
                    "_gas_lbt_expenditures_pre_tax",
                ),
                "consolidated_cost_of_sales_expenditures_pre_tax": _require_array(
                    _require_array(
                        contract.oil_cost_of_sales_expenditures_pre_tax,
                        "_oil_cost_of_sales_expenditures_pre_tax",
                    ),
                    "_oil_cost_of_sales_expenditures_pre_tax",
                )
                + _require_array(
                    _require_array(
                        contract.gas_cost_of_sales_expenditures_pre_tax,
                        "_gas_cost_of_sales_expenditures_pre_tax",
                    ),
                    "_gas_cost_of_sales_expenditures_pre_tax",
                ),
            }
            for idx, contract in enumerate(self._profiles)
        }

    def _get_cost_indirect_tax_dict(self) -> ContractSummaryMap:
        return {
            idx: {
                "years": contract.project_years,
                "oil_capital_indirect_tax": _require_array(
                    _require_array(
                        contract.oil_capital_indirect_tax, "_oil_capital_indirect_tax"
                    ),
                    "_oil_capital_indirect_tax",
                ),
                "oil_intangible_indirect_tax": _require_array(
                    _require_array(
                        contract.oil_intangible_indirect_tax,
                        "_oil_intangible_indirect_tax",
                    ),
                    "_oil_intangible_indirect_tax",
                ),
                "oil_opex_indirect_tax": _require_array(
                    _require_array(
                        contract.oil_opex_indirect_tax, "_oil_opex_indirect_tax"
                    ),
                    "_oil_opex_indirect_tax",
                ),
                "oil_asr_indirect_tax": _require_array(
                    _require_array(
                        contract.oil_asr_indirect_tax, "_oil_asr_indirect_tax"
                    ),
                    "_oil_asr_indirect_tax",
                ),
                "oil_lbt_indirect_tax": _require_array(
                    _require_array(
                        contract.oil_lbt_indirect_tax, "_oil_lbt_indirect_tax"
                    ),
                    "_oil_lbt_indirect_tax",
                ),
                "oil_cost_of_sales_indirect_tax": _require_array(
                    _require_array(
                        contract.oil_cost_of_sales_indirect_tax,
                        "_oil_cost_of_sales_indirect_tax",
                    ),
                    "_oil_cost_of_sales_indirect_tax",
                ),
                "gas_capital_indirect_tax": _require_array(
                    _require_array(
                        contract.gas_capital_indirect_tax, "_gas_capital_indirect_tax"
                    ),
                    "_gas_capital_indirect_tax",
                ),
                "gas_intangible_indirect_tax": _require_array(
                    _require_array(
                        contract.gas_intangible_indirect_tax,
                        "_gas_intangible_indirect_tax",
                    ),
                    "_gas_intangible_indirect_tax",
                ),
                "gas_opex_indirect_tax": _require_array(
                    _require_array(
                        contract.gas_opex_indirect_tax, "_gas_opex_indirect_tax"
                    ),
                    "_gas_opex_indirect_tax",
                ),
                "gas_asr_indirect_tax": _require_array(
                    _require_array(
                        contract.gas_asr_indirect_tax, "_gas_asr_indirect_tax"
                    ),
                    "_gas_asr_indirect_tax",
                ),
                "gas_lbt_indirect_tax": _require_array(
                    _require_array(
                        contract.gas_lbt_indirect_tax, "_gas_lbt_indirect_tax"
                    ),
                    "_gas_lbt_indirect_tax",
                ),
                "gas_cost_of_sales_indirect_tax": _require_array(
                    _require_array(
                        contract.gas_cost_of_sales_indirect_tax,
                        "_gas_cost_of_sales_indirect_tax",
                    ),
                    "_gas_cost_of_sales_indirect_tax",
                ),
                "consolidated_capital_indirect_tax": _require_array(
                    _require_array(
                        contract.oil_capital_indirect_tax, "_oil_capital_indirect_tax"
                    ),
                    "_oil_capital_indirect_tax",
                )
                + _require_array(
                    _require_array(
                        contract.gas_capital_indirect_tax, "_gas_capital_indirect_tax"
                    ),
                    "_gas_capital_indirect_tax",
                ),
                "consolidated_intangible_indirect_tax": _require_array(
                    _require_array(
                        contract.oil_intangible_indirect_tax,
                        "_oil_intangible_indirect_tax",
                    ),
                    "_oil_intangible_indirect_tax",
                )
                + _require_array(
                    _require_array(
                        contract.gas_intangible_indirect_tax,
                        "_gas_intangible_indirect_tax",
                    ),
                    "_gas_intangible_indirect_tax",
                ),
                "consolidated_opex_indirect_tax": _require_array(
                    _require_array(
                        contract.oil_opex_indirect_tax, "_oil_opex_indirect_tax"
                    ),
                    "_oil_opex_indirect_tax",
                )
                + _require_array(
                    _require_array(
                        contract.gas_opex_indirect_tax, "_gas_opex_indirect_tax"
                    ),
                    "_gas_opex_indirect_tax",
                ),
                "consolidated_asr_indirect_tax": _require_array(
                    _require_array(
                        contract.oil_asr_indirect_tax, "_oil_asr_indirect_tax"
                    ),
                    "_oil_asr_indirect_tax",
                )
                + _require_array(
                    _require_array(
                        contract.gas_asr_indirect_tax, "_gas_asr_indirect_tax"
                    ),
                    "_gas_asr_indirect_tax",
                ),
                "consolidated_lbt_indirect_tax": _require_array(
                    _require_array(
                        contract.oil_lbt_indirect_tax, "_oil_lbt_indirect_tax"
                    ),
                    "_oil_lbt_indirect_tax",
                )
                + _require_array(
                    _require_array(
                        contract.gas_lbt_indirect_tax, "_gas_lbt_indirect_tax"
                    ),
                    "_gas_lbt_indirect_tax",
                ),
                "consolidated_cost_of_sales_indirect_tax": _require_array(
                    _require_array(
                        contract.oil_cost_of_sales_indirect_tax,
                        "_oil_cost_of_sales_indirect_tax",
                    ),
                    "_oil_cost_of_sales_indirect_tax",
                )
                + _require_array(
                    _require_array(
                        contract.gas_cost_of_sales_indirect_tax,
                        "_gas_cost_of_sales_indirect_tax",
                    ),
                    "_gas_cost_of_sales_indirect_tax",
                ),
            }
            for idx, contract in enumerate(self._profiles)
        }

    def _get_cost_expenditure_posttax_dict(self) -> ContractSummaryMap:
        return {
            idx: {
                "years": contract.project_years,
                "oil_capital_expenditures_post_tax": _require_array(
                    _require_array(
                        contract.oil_capital_expenditures_post_tax,
                        "_oil_capital_expenditures_post_tax",
                    ),
                    "_oil_capital_expenditures_post_tax",
                ),
                "oil_intangible_expenditures_post_tax": _require_array(
                    _require_array(
                        contract.oil_intangible_expenditures_post_tax,
                        "_oil_intangible_expenditures_post_tax",
                    ),
                    "_oil_intangible_expenditures_post_tax",
                ),
                "oil_opex_expenditures_post_tax": _require_array(
                    _require_array(
                        contract.oil_opex_expenditures_post_tax,
                        "_oil_opex_expenditures_post_tax",
                    ),
                    "_oil_opex_expenditures_post_tax",
                ),
                "oil_asr_expenditures_post_tax": _require_array(
                    _require_array(
                        contract.oil_asr_expenditures_post_tax,
                        "_oil_asr_expenditures_post_tax",
                    ),
                    "_oil_asr_expenditures_post_tax",
                ),
                "oil_lbt_expenditures_post_tax": _require_array(
                    _require_array(
                        contract.oil_lbt_expenditures_post_tax,
                        "_oil_lbt_expenditures_post_tax",
                    ),
                    "_oil_lbt_expenditures_post_tax",
                ),
                "oil_cost_of_sales_expenditures_post_tax": _require_array(
                    _require_array(
                        contract.oil_cost_of_sales_expenditures_post_tax,
                        "_oil_cost_of_sales_expenditures_post_tax",
                    ),
                    "_oil_cost_of_sales_expenditures_post_tax",
                ),
                "gas_capital_expenditures_post_tax": _require_array(
                    _require_array(
                        contract.gas_capital_expenditures_post_tax,
                        "_gas_capital_expenditures_post_tax",
                    ),
                    "_gas_capital_expenditures_post_tax",
                ),
                "gas_intangible_expenditures_post_tax": _require_array(
                    _require_array(
                        contract.gas_intangible_expenditures_post_tax,
                        "_gas_intangible_expenditures_post_tax",
                    ),
                    "_gas_intangible_expenditures_post_tax",
                ),
                "gas_opex_expenditures_post_tax": _require_array(
                    _require_array(
                        contract.gas_opex_expenditures_post_tax,
                        "_gas_opex_expenditures_post_tax",
                    ),
                    "_gas_opex_expenditures_post_tax",
                ),
                "gas_asr_expenditures_post_tax": _require_array(
                    _require_array(
                        contract.gas_asr_expenditures_post_tax,
                        "_gas_asr_expenditures_post_tax",
                    ),
                    "_gas_asr_expenditures_post_tax",
                ),
                "gas_lbt_expenditures_post_tax": _require_array(
                    _require_array(
                        contract.gas_lbt_expenditures_post_tax,
                        "_gas_lbt_expenditures_post_tax",
                    ),
                    "_gas_lbt_expenditures_post_tax",
                ),
                "gas_cost_of_sales_expenditures_post_tax": _require_array(
                    _require_array(
                        contract.gas_cost_of_sales_expenditures_post_tax,
                        "_gas_cost_of_sales_expenditures_post_tax",
                    ),
                    "_gas_cost_of_sales_expenditures_post_tax",
                ),
                "consolidated_capital_expenditures_post_tax": _require_array(
                    _require_array(
                        contract.oil_capital_expenditures_post_tax,
                        "_oil_capital_expenditures_post_tax",
                    ),
                    "_oil_capital_expenditures_post_tax",
                )
                + _require_array(
                    _require_array(
                        contract.gas_capital_expenditures_post_tax,
                        "_gas_capital_expenditures_post_tax",
                    ),
                    "_gas_capital_expenditures_post_tax",
                ),
                "consolidated_intangible_expenditures_post_tax": _require_array(
                    _require_array(
                        contract.oil_intangible_expenditures_post_tax,
                        "_oil_intangible_expenditures_post_tax",
                    ),
                    "_oil_intangible_expenditures_post_tax",
                )
                + _require_array(
                    _require_array(
                        contract.gas_intangible_expenditures_post_tax,
                        "_gas_intangible_expenditures_post_tax",
                    ),
                    "_gas_intangible_expenditures_post_tax",
                ),
                "consolidated_opex_expenditures_post_tax": _require_array(
                    _require_array(
                        contract.oil_opex_expenditures_post_tax,
                        "_oil_opex_expenditures_post_tax",
                    ),
                    "_oil_opex_expenditures_post_tax",
                )
                + _require_array(
                    _require_array(
                        contract.gas_opex_expenditures_post_tax,
                        "_gas_opex_expenditures_post_tax",
                    ),
                    "_gas_opex_expenditures_post_tax",
                ),
                "consolidated_asr_expenditures_post_tax": _require_array(
                    _require_array(
                        contract.oil_asr_expenditures_post_tax,
                        "_oil_asr_expenditures_post_tax",
                    ),
                    "_oil_asr_expenditures_post_tax",
                )
                + _require_array(
                    _require_array(
                        contract.gas_asr_expenditures_post_tax,
                        "_gas_asr_expenditures_post_tax",
                    ),
                    "_gas_asr_expenditures_post_tax",
                ),
                "consolidated_lbt_expenditures_post_tax": _require_array(
                    _require_array(
                        contract.oil_lbt_expenditures_post_tax,
                        "_oil_lbt_expenditures_post_tax",
                    ),
                    "_oil_lbt_expenditures_post_tax",
                )
                + _require_array(
                    _require_array(
                        contract.gas_lbt_expenditures_post_tax,
                        "_gas_lbt_expenditures_post_tax",
                    ),
                    "_gas_lbt_expenditures_post_tax",
                ),
                "consolidated_cost_of_sales_expenditures_post_tax": _require_array(
                    _require_array(
                        contract.oil_cost_of_sales_expenditures_post_tax,
                        "_oil_cost_of_sales_expenditures_post_tax",
                    ),
                    "_oil_cost_of_sales_expenditures_post_tax",
                )
                + _require_array(
                    _require_array(
                        contract.gas_cost_of_sales_expenditures_post_tax,
                        "_gas_cost_of_sales_expenditures_post_tax",
                    ),
                    "_gas_cost_of_sales_expenditures_post_tax",
                ),
            }
            for idx, contract in enumerate(self._profiles)
        }

    def _get_cost_contract_dict(self) -> ContractSummaryMap:
        return {
            idx: {
                "years": contract.project_years,
                "oil_depreciable": _require_array(
                    _require_array(
                        contract.oil_capital_expenditures_post_tax,
                        "_oil_capital_expenditures_post_tax",
                    ),
                    "_oil_capital_expenditures_post_tax",
                ),
                "oil_intangible": _require_array(
                    _require_array(
                        contract.oil_intangible_expenditures_post_tax,
                        "_oil_intangible_expenditures_post_tax",
                    ),
                    "_oil_intangible_expenditures_post_tax",
                ),
                "oil_opex": _require_array(
                    _require_array(
                        contract.oil_opex_expenditures_post_tax,
                        "_oil_opex_expenditures_post_tax",
                    ),
                    "_oil_opex_expenditures_post_tax",
                ),
                "oil_asr": _require_array(
                    _require_array(
                        contract.oil_asr_expenditures_post_tax,
                        "_oil_asr_expenditures_post_tax",
                    ),
                    "_oil_asr_expenditures_post_tax",
                ),
                "oil_lbt": _require_array(
                    _require_array(
                        contract.oil_lbt_expenditures_post_tax,
                        "_oil_lbt_expenditures_post_tax",
                    ),
                    "_oil_lbt_expenditures_post_tax",
                ),
                "oil_depreciation": self._array_or_zeros(
                    contract.oil_depreciation, contract.project_years
                ),
                "oil_non_capital": self._fluid_non_capital(contract, "oil"),
                "oil_sunk_cost": _require_array(
                    _require_array(contract.oil_sunk_cost, "_oil_sunk_cost"),
                    "_oil_sunk_cost",
                ),
                "oil_undepreciated_asset": self._scalar_or_zero(
                    contract.oil_sum_undepreciated_asset,
                ),
                "oil_carry_forward_depreciation": self._array_or_zeros(
                    contract.oil_carry_forward_depreciation, contract.project_years
                ),
                "gas_depreciable": _require_array(
                    _require_array(
                        contract.gas_capital_expenditures_post_tax,
                        "_gas_capital_expenditures_post_tax",
                    ),
                    "_gas_capital_expenditures_post_tax",
                ),
                "gas_intangible": _require_array(
                    _require_array(
                        contract.gas_intangible_expenditures_post_tax,
                        "_gas_intangible_expenditures_post_tax",
                    ),
                    "_gas_intangible_expenditures_post_tax",
                ),
                "gas_opex": _require_array(
                    _require_array(
                        contract.gas_opex_expenditures_post_tax,
                        "_gas_opex_expenditures_post_tax",
                    ),
                    "_gas_opex_expenditures_post_tax",
                ),
                "gas_asr": _require_array(
                    _require_array(
                        contract.gas_asr_expenditures_post_tax,
                        "_gas_asr_expenditures_post_tax",
                    ),
                    "_gas_asr_expenditures_post_tax",
                ),
                "gas_lbt": _require_array(
                    _require_array(
                        contract.gas_lbt_expenditures_post_tax,
                        "_gas_lbt_expenditures_post_tax",
                    ),
                    "_gas_lbt_expenditures_post_tax",
                ),
                "gas_depreciation": self._array_or_zeros(
                    contract.gas_depreciation, contract.project_years
                ),
                "gas_non_capital": self._fluid_non_capital(contract, "gas"),
                "gas_sunk_cost": _require_array(
                    _require_array(contract.gas_sunk_cost, "_gas_sunk_cost"),
                    "_gas_sunk_cost",
                ),
                "gas_undepreciated_asset": self._scalar_or_zero(
                    contract.gas_sum_undepreciated_asset,
                ),
                "gas_carry_forward_depreciation": self._array_or_zeros(
                    contract.gas_carry_forward_depreciation, contract.project_years
                ),
                "consolidated_depreciable": _require_array(
                    _require_array(
                        contract.oil_capital_expenditures_post_tax,
                        "_oil_capital_expenditures_post_tax",
                    ),
                    "_oil_capital_expenditures_post_tax",
                )
                + _require_array(
                    _require_array(
                        contract.gas_capital_expenditures_post_tax,
                        "_gas_capital_expenditures_post_tax",
                    ),
                    "_gas_capital_expenditures_post_tax",
                ),
                "consolidated_intangible": _require_array(
                    _require_array(
                        contract.oil_intangible_expenditures_post_tax,
                        "_oil_intangible_expenditures_post_tax",
                    ),
                    "_oil_intangible_expenditures_post_tax",
                )
                + _require_array(
                    _require_array(
                        contract.gas_intangible_expenditures_post_tax,
                        "_gas_intangible_expenditures_post_tax",
                    ),
                    "_gas_intangible_expenditures_post_tax",
                ),
                "consolidated_opex": _require_array(
                    _require_array(
                        contract.oil_opex_expenditures_post_tax,
                        "_oil_opex_expenditures_post_tax",
                    ),
                    "_oil_opex_expenditures_post_tax",
                )
                + _require_array(
                    _require_array(
                        contract.gas_opex_expenditures_post_tax,
                        "_gas_opex_expenditures_post_tax",
                    ),
                    "_gas_opex_expenditures_post_tax",
                ),
                "consolidated_asr": _require_array(
                    _require_array(
                        contract.oil_asr_expenditures_post_tax,
                        "_oil_asr_expenditures_post_tax",
                    ),
                    "_oil_asr_expenditures_post_tax",
                )
                + _require_array(
                    _require_array(
                        contract.gas_asr_expenditures_post_tax,
                        "_gas_asr_expenditures_post_tax",
                    ),
                    "_gas_asr_expenditures_post_tax",
                ),
                "consolidated_lbt": _require_array(
                    _require_array(
                        contract.oil_lbt_expenditures_post_tax,
                        "_oil_lbt_expenditures_post_tax",
                    ),
                    "_oil_lbt_expenditures_post_tax",
                )
                + _require_array(
                    _require_array(
                        contract.gas_lbt_expenditures_post_tax,
                        "_gas_lbt_expenditures_post_tax",
                    ),
                    "_gas_lbt_expenditures_post_tax",
                ),
                "consolidated_depreciation": self._array_or_zeros(
                    contract.oil_depreciation, contract.project_years
                )
                + self._array_or_zeros(
                    contract.gas_depreciation, contract.project_years
                ),
                "consolidated_non_capital": self._fluid_non_capital(contract, "oil")
                + self._fluid_non_capital(contract, "gas"),
                "consolidated_sunk_cost": _require_array(
                    _require_array(contract.oil_sunk_cost, "_oil_sunk_cost"),
                    "_oil_sunk_cost",
                )
                + _require_array(contract.gas_sunk_cost, "_gas_sunk_cost"),
                "consolidated_undepreciated_asset": self._scalar_or_zero(
                    contract.oil_sum_undepreciated_asset,
                )
                + self._scalar_or_zero(
                    contract.gas_sum_undepreciated_asset,
                ),
                "consolidated_carry_forward_depreciation": self._array_or_zeros(
                    contract.consolidated_carry_forward_depreciation,
                    contract.project_years,
                ),
            }
            for idx, contract in enumerate(self._profiles)
        }

    def _get_psc_terms_dict(self) -> ContractSummaryMap:
        result: ContractSummaryMap = {}
        for idx, contract in enumerate(self._profiles):
            if (
                contract.contract_type == "cost_recovery"
                or contract.contract_type == "gross_split"
            ):
                result[idx] = {
                    "years": _require_years(contract.project_years),
                    "oil_costrecovery_or_deductible_cost": _require_array(
                        _require_array(
                            contract.oil_cost_recovery_after_tf,
                            "_oil_cost_recovery_after_tf",
                        ),
                        "_oil_cost_recovery_after_tf",
                    )
                    if contract.contract_type == "cost_recovery"
                    else _require_array(
                        _require_array(
                            contract.oil_deductible_cost, "_oil_deductible_cost"
                        ),
                        "_oil_deductible_cost",
                    ),
                    "oil_unrecoverable_cost_or_carryforward_cost": _require_array(
                        _require_array(
                            contract.oil_unrecovered_after_transfer,
                            "_oil_unrecovered_after_transfer",
                        ),
                        "_oil_unrecovered_after_transfer",
                    )
                    if contract.contract_type == "cost_recovery"
                    else _require_array(
                        _require_array(
                            contract.oil_carward_cost_aftertf,
                            "_oil_carward_cost_aftertf",
                        ),
                        "_oil_carward_cost_aftertf",
                    ),
                    # patch 12/03/2025
                    "oil_ctr_share": _require_array(
                        _require_array(
                            contract.oil_contractor_share, "_oil_contractor_share"
                        ),
                        "_oil_contractor_share",
                    )
                    if contract.contract_type == "cost_recovery"
                    else _require_array(
                        _require_array(
                            contract.oil_ctr_net_share, "_oil_ctr_net_share"
                        ),
                        "_oil_ctr_net_share",
                    ),
                    # 'oil_ctr_share': _require_array(contract.oil_contractor_share, "_oil_contractor_share") if contract.contract_type == "cost_recovery" else _require_array(contract.oil_ctr_share_after_transfer, "_oil_ctr_share_after_transfer"),
                    "oil_gov_share": _require_array(
                        _require_array(
                            contract.oil_government_share, "_oil_government_share"
                        ),
                        "_oil_government_share",
                    )
                    if contract.contract_type == "cost_recovery"
                    else _require_array(contract.oil_gov_share, "_oil_gov_share"),
                    "oil_effective_tax_payment": _require_array(
                        _require_array(contract.oil_tax_payment, "_oil_tax_payment"),
                        "_oil_tax_payment",
                    )
                    if contract.contract_type == "cost_recovery"
                    else _require_array(contract.oil_tax, "_oil_tax"),
                    "oil_ddmo": _require_array(contract.oil_ddmo, "_oil_ddmo"),
                    "oil_ctr_take": _require_array(
                        _require_array(
                            contract.oil_contractor_take, "_oil_contractor_take"
                        ),
                        "_oil_contractor_take",
                    )
                    if contract.contract_type == "cost_recovery"
                    else _require_array(
                        _require_array(
                            contract.oil_ctr_net_share, "_oil_ctr_net_share"
                        ),
                        "_oil_ctr_net_share",
                    ),
                    "oil_government_take": _require_array(
                        _require_array(
                            contract.oil_government_take, "_oil_government_take"
                        ),
                        "_oil_government_take",
                    ),
                    "oil_ctr_cashflow": _require_array(
                        _require_array(contract.oil_cashflow, "_oil_cashflow"),
                        "_oil_cashflow",
                    )
                    if contract.contract_type == "cost_recovery"
                    else _require_array(contract.oil_ctr_cashflow, "_oil_ctr_cashflow"),
                    "gas_costrecovery_or_deductible_cost": _require_array(
                        _require_array(
                            contract.gas_cost_recovery_after_tf,
                            "_gas_cost_recovery_after_tf",
                        ),
                        "_gas_cost_recovery_after_tf",
                    )
                    if contract.contract_type == "cost_recovery"
                    else _require_array(
                        _require_array(
                            contract.gas_deductible_cost, "_gas_deductible_cost"
                        ),
                        "_gas_deductible_cost",
                    ),
                    "gas_unrecoverable_cost_or_carryforward_cost": _require_array(
                        _require_array(
                            contract.gas_unrecovered_after_transfer,
                            "_gas_unrecovered_after_transfer",
                        ),
                        "_gas_unrecovered_after_transfer",
                    )
                    if contract.contract_type == "cost_recovery"
                    else _require_array(
                        _require_array(
                            contract.gas_carward_cost_aftertf,
                            "_gas_carward_cost_aftertf",
                        ),
                        "_gas_carward_cost_aftertf",
                    ),
                    # patch 12/03/2025
                    "gas_ctr_share": _require_array(
                        _require_array(
                            contract.gas_contractor_share, "_gas_contractor_share"
                        ),
                        "_gas_contractor_share",
                    )
                    if contract.contract_type == "cost_recovery"
                    else _require_array(
                        _require_array(
                            contract.gas_ctr_net_share, "_gas_ctr_net_share"
                        ),
                        "_gas_ctr_net_share",
                    ),
                    # 'gas_ctr_share': _require_array(contract.gas_contractor_share, "_gas_contractor_share") if isinstance(contract,CostRecovery) else _require_array(contract.gas_ctr_share_after_transfer, "_gas_ctr_share_after_transfer"),
                    "gas_gov_share": _require_array(
                        _require_array(
                            contract.gas_government_share, "_gas_government_share"
                        ),
                        "_gas_government_share",
                    )
                    if contract.contract_type == "cost_recovery"
                    else _require_array(contract.gas_gov_share, "_gas_gov_share"),
                    "gas_effective_tax_payment": _require_array(
                        _require_array(contract.gas_tax_payment, "_gas_tax_payment"),
                        "_gas_tax_payment",
                    )
                    if contract.contract_type == "cost_recovery"
                    else _require_array(contract.gas_tax, "_gas_tax"),
                    "gas_ddmo": _require_array(contract.gas_ddmo, "_gas_ddmo"),
                    "gas_ctr_take": _require_array(
                        _require_array(
                            contract.gas_contractor_take, "_gas_contractor_take"
                        ),
                        "_gas_contractor_take",
                    )
                    if contract.contract_type == "cost_recovery"
                    else _require_array(
                        _require_array(
                            contract.gas_ctr_net_share, "_gas_ctr_net_share"
                        ),
                        "_gas_ctr_net_share",
                    ),
                    "gas_government_take": _require_array(
                        _require_array(
                            contract.gas_government_take, "_gas_government_take"
                        ),
                        "_gas_government_take",
                    ),
                    "gas_ctr_cashflow": _require_array(
                        _require_array(contract.gas_cashflow, "_gas_cashflow"),
                        "_gas_cashflow",
                    )
                    if contract.contract_type == "cost_recovery"
                    else _require_array(contract.gas_ctr_cashflow, "_gas_ctr_cashflow"),
                    "consolidated_costrecovery_or_deductible_cost": _require_array(
                        _require_array(
                            contract.consolidated_cost_recovery_after_tf,
                            "_consolidated_cost_recovery_after_tf",
                        ),
                        "_consolidated_cost_recovery_after_tf",
                    )
                    if contract.contract_type == "cost_recovery"
                    else _require_array(
                        _require_array(
                            contract.consolidated_deductible_cost,
                            "_consolidated_deductible_cost",
                        ),
                        "_consolidated_deductible_cost",
                    ),
                    "consolidated_unrecoverable_cost_or_carryforward_cost": _require_array(
                        _require_array(
                            contract.consolidated_unrecovered_after_transfer,
                            "_consolidated_unrecovered_after_transfer",
                        ),
                        "_consolidated_unrecovered_after_transfer",
                    )
                    if contract.contract_type == "cost_recovery"
                    else _require_array(
                        _require_array(
                            contract.consolidated_carward_cost_aftertf,
                            "_consolidated_carward_cost_aftertf",
                        ),
                        "_consolidated_carward_cost_aftertf",
                    ),
                    # patch 12/03/2025
                    "consolidated_ctr_share": _require_array(
                        _require_array(
                            contract.consolidated_contractor_share,
                            "_consolidated_contractor_share",
                        ),
                        "_consolidated_contractor_share",
                    )
                    if contract.contract_type == "cost_recovery"
                    else _require_array(
                        _require_array(
                            contract.consolidated_ctr_share_before_tf,
                            "_consolidated_ctr_share_before_tf",
                        ),
                        "_consolidated_ctr_share_before_tf",
                    ),
                    # 'consolidated_ctr_share': _require_array(contract.consolidated_contractor_share, "_consolidated_contractor_share") if contract.contract_type == "cost_recovery" else _require_array(contract.consolidated_ctr_share_after_transfer, "_consolidated_ctr_share_after_transfer"),
                    "consolidated_ctr_net_share": _require_array(
                        _require_array(
                            contract.consolidated_ctr_net_share,
                            "_consolidated_ctr_net_share",
                        ),
                        "_consolidated_ctr_net_share",
                    ),
                    "consolidated_gov_share": _require_array(
                        _require_array(
                            contract.consolidated_government_share,
                            "_consolidated_government_share",
                        ),
                        "_consolidated_government_share",
                    )
                    if contract.contract_type == "cost_recovery"
                    else _require_array(
                        _require_array(
                            contract.consolidated_gov_share_before_tf,
                            "_consolidated_gov_share_before_tf",
                        ),
                        "_consolidated_gov_share_before_tf",
                    ),
                    "consolidated_effective_tax_payment": _require_array(
                        _require_array(
                            contract.consolidated_tax_payment,
                            "_consolidated_tax_payment",
                        ),
                        "_consolidated_tax_payment",
                    )
                    if contract.contract_type == "cost_recovery"
                    else _require_array(
                        _require_array(
                            contract.consolidated_tax_payment,
                            "_consolidated_tax_payment",
                        ),
                        "_consolidated_tax_payment",
                    ),
                    "consolidated_ddmo": _require_array(
                        _require_array(
                            contract.consolidated_ddmo, "_consolidated_ddmo"
                        ),
                        "_consolidated_ddmo",
                    ),
                    "consolidated_ctr_take": _require_array(
                        _require_array(
                            contract.consolidated_contractor_take,
                            "_consolidated_contractor_take",
                        ),
                        "_consolidated_contractor_take",
                    )
                    if contract.contract_type == "cost_recovery"
                    else _require_array(
                        _require_array(
                            contract.consolidated_ctr_net_share,
                            "_consolidated_ctr_net_share",
                        ),
                        "_consolidated_ctr_net_share",
                    ),
                    "consolidated_government_take": _require_array(
                        _require_array(
                            contract.consolidated_government_take,
                            "_consolidated_government_take",
                        ),
                        "_consolidated_government_take",
                    ),
                    "consolidated_ctr_cashflow": _require_array(
                        _require_array(
                            contract.consolidated_cashflow, "_consolidated_cashflow"
                        ),
                        "_consolidated_cashflow",
                    ),
                }

            elif contract.contract_type == "transition":
                result[idx] = {
                    "years": _require_years(contract.project_years),
                    "oil_costrecovery_or_deductible_cost": _require_array(
                        _require_array(
                            contract.oil_deductible_cost, "_oil_deductible_cost"
                        ),
                        "_oil_deductible_cost",
                    ),
                    "oil_unrecoverable_cost_or_carryforward_cost": _require_array(
                        _require_array(contract.oil_unrec_cost, "_oil_unrec_cost"),
                        "_oil_unrec_cost",
                    ),
                    "oil_ctr_share": _require_array(
                        _require_array(contract.oil_ctr_ets, "_oil_ctr_ets"),
                        "_oil_ctr_ets",
                    ),
                    "oil_gov_share": _require_array(
                        _require_array(contract.oil_gov_ets, "_oil_gov_ets"),
                        "_oil_gov_ets",
                    ),
                    "oil_effective_tax_payment": _require_array(
                        _require_array(
                            contract.oil_effective_tax_payment,
                            "_oil_effective_tax_payment",
                        ),
                        "_oil_effective_tax_payment",
                    ),
                    "oil_ddmo": _require_array(contract.oil_ddmo, "_oil_ddmo"),
                    "oil_ctr_take": _require_array(
                        _require_array(contract.oil_ctr_take, "_oil_ctr_take"),
                        "_oil_ctr_take",
                    ),
                    "oil_government_take": _require_array(
                        _require_array(
                            contract.oil_government_take, "_oil_government_take"
                        ),
                        "_oil_government_take",
                    ),
                    "oil_ctr_cashflow": _require_array(
                        _require_array(contract.oil_cashflow, "_oil_cashflow"),
                        "_oil_cashflow",
                    ),
                    "gas_costrecovery_or_deductible_cost": _require_array(
                        _require_array(
                            contract.gas_deductible_cost, "_gas_deductible_cost"
                        ),
                        "_gas_deductible_cost",
                    ),
                    "gas_unrecoverable_cost_or_carryforward_cost": _require_array(
                        _require_array(contract.gas_unrec_cost, "_gas_unrec_cost"),
                        "_gas_unrec_cost",
                    ),
                    "gas_ctr_share": _require_array(
                        _require_array(contract.gas_ctr_ets, "_gas_ctr_ets"),
                        "_gas_ctr_ets",
                    ),
                    "gas_gov_share": _require_array(
                        _require_array(contract.gas_gov_ets, "_gas_gov_ets"),
                        "_gas_gov_ets",
                    ),
                    "gas_effective_tax_payment": _require_array(
                        _require_array(
                            contract.gas_effective_tax_payment,
                            "_gas_effective_tax_payment",
                        ),
                        "_gas_effective_tax_payment",
                    ),
                    "gas_ddmo": _require_array(contract.gas_ddmo, "_gas_ddmo"),
                    "gas_ctr_take": _require_array(
                        _require_array(contract.gas_ctr_take, "_gas_ctr_take"),
                        "_gas_ctr_take",
                    ),
                    "gas_government_take": _require_array(
                        _require_array(
                            contract.gas_government_take, "_gas_government_take"
                        ),
                        "_gas_government_take",
                    ),
                    "gas_ctr_cashflow": _require_array(
                        _require_array(contract.gas_cashflow, "_gas_cashflow"),
                        "_gas_cashflow",
                    ),
                    "consolidated_costrecovery_or_deductible_cost": _require_array(
                        _require_array(
                            contract.oil_deductible_cost, "_oil_deductible_cost"
                        ),
                        "_oil_deductible_cost",
                    )
                    + _require_array(
                        _require_array(
                            contract.gas_deductible_cost, "_gas_deductible_cost"
                        ),
                        "_gas_deductible_cost",
                    ),
                    "consolidated_unrecoverable_cost_or_carryforward_cost": _require_array(
                        _require_array(contract.oil_unrec_cost, "_oil_unrec_cost"),
                        "_oil_unrec_cost",
                    )
                    + _require_array(contract.gas_unrec_cost, "_gas_unrec_cost"),
                    "consolidated_ctr_share": _require_array(
                        _require_array(contract.oil_ctr_ets, "_oil_ctr_ets"),
                        "_oil_ctr_ets",
                    )
                    + _require_array(contract.gas_ctr_ets, "_gas_ctr_ets"),
                    "consolidated_gov_share": _require_array(
                        _require_array(contract.oil_gov_ets, "_oil_gov_ets"),
                        "_oil_gov_ets",
                    )
                    + _require_array(contract.gas_gov_ets, "_gas_gov_ets"),
                    "consolidated_effective_tax_payment": _require_array(
                        _require_array(
                            contract.oil_effective_tax_payment,
                            "_oil_effective_tax_payment",
                        ),
                        "_oil_effective_tax_payment",
                    )
                    + _require_array(
                        _require_array(
                            contract.gas_effective_tax_payment,
                            "_gas_effective_tax_payment",
                        ),
                        "_gas_effective_tax_payment",
                    ),
                    "consolidated_ddmo": _require_array(contract.oil_ddmo, "_oil_ddmo")
                    + _require_array(contract.gas_ddmo, "_gas_ddmo"),
                    "consolidated_ctr_take": _require_array(
                        _require_array(contract.oil_ctr_take, "_oil_ctr_take"),
                        "_oil_ctr_take",
                    )
                    + _require_array(contract.gas_ctr_take, "_gas_ctr_take"),
                    "consolidated_government_take": _require_array(
                        _require_array(
                            contract.oil_government_take, "_oil_government_take"
                        ),
                        "_oil_government_take",
                    )
                    + _require_array(
                        _require_array(
                            contract.gas_government_take, "_gas_government_take"
                        ),
                        "_gas_government_take",
                    ),
                    "consolidated_ctr_cashflow": _require_array(
                        _require_array(contract.oil_cashflow, "_oil_cashflow"),
                        "_oil_cashflow",
                    )
                    + _require_array(contract.gas_cashflow, "_gas_cashflow"),
                }
            elif contract.contract_type == "base_project":
                # A base project has no production-sharing terms: the per-case
                # summary reports zero government take and treats the gross
                # revenue as the contractor gross share, with the consolidated
                # cashflow as the contractor cashflow. These are the documented
                # ``BaseProject.get_summary`` semantics, materialized as profile
                # arrays so the combine/increment merge stays additive.
                years = _require_years(contract.project_years)
                zeros = np.zeros_like(years, dtype=np.float64)
                oil_revenue = _require_array(contract.oil_revenue, "_oil_revenue")
                gas_revenue = _require_array(contract.gas_revenue, "_gas_revenue")
                consolidated_revenue = _require_array(
                    contract.consolidated_revenue, "_consolidated_revenue"
                )
                oil_cashflow = _require_array(contract.oil_cashflow, "_oil_cashflow")
                gas_cashflow = _require_array(contract.gas_cashflow, "_gas_cashflow")
                consolidated_cashflow = _require_array(
                    contract.consolidated_cashflow, "_consolidated_cashflow"
                )
                # Documented base-project net share: gross revenue less the
                # tangible/intangible investment and the opex/ASR/LBT charges
                # (BaseProject.get_summary). No production-sharing terms apply.
                ctr_net_share = (
                    consolidated_revenue
                    - _require_array(
                        contract.oil_capital_expenditures_post_tax,
                        "_oil_capital_expenditures_post_tax",
                    )
                    - _require_array(
                        contract.gas_capital_expenditures_post_tax,
                        "_gas_capital_expenditures_post_tax",
                    )
                    - _require_array(
                        contract.oil_intangible_expenditures_post_tax,
                        "_oil_intangible_expenditures_post_tax",
                    )
                    - _require_array(
                        contract.gas_intangible_expenditures_post_tax,
                        "_gas_intangible_expenditures_post_tax",
                    )
                    - _require_array(
                        contract.oil_opex_expenditures_post_tax,
                        "_oil_opex_expenditures_post_tax",
                    )
                    - _require_array(
                        contract.gas_opex_expenditures_post_tax,
                        "_gas_opex_expenditures_post_tax",
                    )
                    - _require_array(
                        contract.oil_asr_expenditures_post_tax,
                        "_oil_asr_expenditures_post_tax",
                    )
                    - _require_array(
                        contract.gas_asr_expenditures_post_tax,
                        "_gas_asr_expenditures_post_tax",
                    )
                    - _require_array(
                        contract.oil_lbt_expenditures_post_tax,
                        "_oil_lbt_expenditures_post_tax",
                    )
                    - _require_array(
                        contract.gas_lbt_expenditures_post_tax,
                        "_gas_lbt_expenditures_post_tax",
                    )
                )
                result[idx] = {
                    "years": years,
                    "oil_costrecovery_or_deductible_cost": zeros,
                    "oil_unrecoverable_cost_or_carryforward_cost": zeros,
                    "oil_ctr_share": oil_revenue,
                    "oil_gov_share": zeros,
                    "oil_effective_tax_payment": zeros,
                    "oil_ddmo": zeros,
                    "oil_ctr_take": oil_revenue,
                    "oil_government_take": zeros,
                    "oil_ctr_cashflow": oil_cashflow,
                    "gas_costrecovery_or_deductible_cost": zeros,
                    "gas_unrecoverable_cost_or_carryforward_cost": zeros,
                    "gas_ctr_share": gas_revenue,
                    "gas_gov_share": zeros,
                    "gas_effective_tax_payment": zeros,
                    "gas_ddmo": zeros,
                    "gas_ctr_take": gas_revenue,
                    "gas_government_take": zeros,
                    "gas_ctr_cashflow": gas_cashflow,
                    "consolidated_costrecovery_or_deductible_cost": zeros,
                    "consolidated_unrecoverable_cost_or_carryforward_cost": zeros,
                    "consolidated_ctr_share": consolidated_revenue,
                    "consolidated_ctr_net_share": ctr_net_share,
                    "consolidated_gov_share": zeros,
                    "consolidated_effective_tax_payment": zeros,
                    "consolidated_ddmo": zeros,
                    "consolidated_ctr_take": consolidated_revenue,
                    "consolidated_government_take": zeros,
                    "consolidated_ctr_cashflow": consolidated_cashflow,
                }
            else:
                pass

        return result

    def _get_cr_terms_dict(self) -> ContractSummaryMap:
        return {
            idx: {
                "years": contract.project_years,
                "oil_ftp_ctr": np.zeros_like(contract.project_years, dtype=float)
                if contract.contract_type in {"gross_split", "base_project"}
                else _require_array(contract.oil_ftp_ctr, "_oil_ftp_ctr"),
                "oil_ftp_gov": np.zeros_like(contract.project_years, dtype=float)
                if contract.contract_type in {"gross_split", "base_project"}
                else _require_array(contract.oil_ftp_gov, "_oil_ftp_gov"),
                "gas_ftp_ctr": np.zeros_like(contract.project_years, dtype=float)
                if contract.contract_type in {"gross_split", "base_project"}
                else _require_array(contract.gas_ftp_ctr, "_gas_ftp_ctr"),
                "gas_ftp_gov": np.zeros_like(contract.project_years, dtype=float)
                if contract.contract_type in {"gross_split", "base_project"}
                else _require_array(contract.gas_ftp_gov, "_gas_ftp_gov"),
                "consolidated_ctr_ftp": np.zeros_like(
                    contract.project_years, dtype=float
                )
                if contract.contract_type in {"gross_split", "base_project"}
                else (
                    _require_array(contract.oil_ftp_ctr, "_oil_ftp_ctr")
                    + _require_array(contract.gas_ftp_ctr, "_gas_ftp_ctr")
                ),
                "consolidated_gov_ftp": np.zeros_like(
                    contract.project_years, dtype=float
                )
                if contract.contract_type in {"gross_split", "base_project"}
                else (
                    _require_array(contract.oil_ftp_gov, "_oil_ftp_gov")
                    + _require_array(contract.gas_ftp_gov, "_gas_ftp_gov")
                ),
            }
            for idx, contract in enumerate(self._profiles)
        }

    @staticmethod
    def _map_to_ref_years(
        ref_years: YearArray, project_years: YearArray, values: FloatArray
    ) -> FloatArray:
        # Get the indices where ref_years would be inserted into project_years
        indices = np.searchsorted(project_years, ref_years, side="right") - 1

        # Ensure the indices are valid (i.e., within bounds)
        indices = np.clip(indices, 0, len(project_years) - 1)

        return values[indices]

    def _parse_dict(self, source: ContractSummaryMap) -> ParsedSummaryMap:
        parsed: ParsedSummaryMap = {}
        for index, contract_values in source.items():
            project_years = _year_array(contract_values["years"])
            parsed_values: ParsedSummary = {"years": self._years()}
            for key, value in contract_values.items():
                if key == "years":
                    continue
                if key in {
                    "oil_undepreciated_asset",
                    "gas_undepreciated_asset",
                    "consolidated_undepreciated_asset",
                }:
                    undepreciated = np.zeros_like(self._years(), dtype=np.float64)
                    undepreciated[len(project_years) - 1] = _summary_scalar(value, key)
                    parsed_values[key] = undepreciated
                else:
                    parsed_values[key] = self._map_to_ref_years(
                        ref_years=self._years(),
                        project_years=project_years,
                        values=_float_array(value, key),
                    )
            parsed[index] = parsed_values
        return parsed

    @staticmethod
    def _merge_dicts(data: ParsedSummaryMap, mode: str) -> ParsedSummary:
        # Convert to list of dictionaries
        dicts = tuple(data.values())

        # Copy first dictionary
        merged = {key: value.copy() for key, value in dicts[0].items()}

        for other in dicts[1:]:
            for key, value in other.items():
                if key != "years":
                    # Element-wise addition
                    if mode == "combine":
                        merged[key] += value
                    elif mode == "incremental":
                        merged[key] -= value
                    else:
                        pass

        return merged

    def _parsing_attributes(
        self,
        lifting: ParsedSummary,
        revenue: ParsedSummary,
        cost_expenditure_pretax: ParsedSummary,
        cost_indirect_tax: ParsedSummary,
        cost_expenditure_posttax: ParsedSummary,
        cost_contract: ParsedSummary,
        psc_terms: ParsedSummary,
        cr_terms: ParsedSummary,
    ):
        # Lifting Attributes
        self.oil_lifting = _float_array(lifting["oil_lifting"])
        self.gas_lifting = _float_array(lifting["gas_lifting"])
        self.sulfur_lifting = _float_array(lifting["sulfur_lifting"])
        self.electricity_lifting = _float_array(lifting["electricity_lifting"])
        self.co2_lifting = _float_array(lifting["co2_lifting"])

        # Revenue
        self.oil_revenue = _float_array(revenue["oil_revenue"])
        self.gas_revenue = _float_array(revenue["gas_revenue"])
        self.sulfur_revenue = _float_array(revenue["sulfur_revenue"])
        self.electricity_revenue = _float_array(revenue["electricity_revenue"])
        self.co2_revenue = _float_array(revenue["co2_revenue"])

        # Cost Expenditures Pre Tax
        self.oil_capital_expenditures_pre_tax = _float_array(
            cost_expenditure_pretax["oil_capital_expenditures_pre_tax"]
        )
        self.oil_intangible_expenditures_pre_tax = _float_array(
            cost_expenditure_pretax["oil_intangible_expenditures_pre_tax"]
        )
        self.oil_opex_expenditures_pre_tax = _float_array(
            cost_expenditure_pretax["oil_opex_expenditures_pre_tax"]
        )
        self.oil_asr_expenditures_pre_tax = _float_array(
            cost_expenditure_pretax["oil_asr_expenditures_pre_tax"]
        )
        self.oil_lbt_expenditures_pre_tax = _float_array(
            cost_expenditure_pretax["oil_lbt_expenditures_pre_tax"]
        )
        self.oil_cost_of_sales_expenditures_pre_tax = _float_array(
            cost_expenditure_pretax["oil_cost_of_sales_expenditures_pre_tax"]
        )
        self.gas_capital_expenditures_pre_tax = _float_array(
            cost_expenditure_pretax["gas_capital_expenditures_pre_tax"]
        )
        self.gas_intangible_expenditures_pre_tax = _float_array(
            cost_expenditure_pretax["gas_intangible_expenditures_pre_tax"]
        )
        self.gas_opex_expenditures_pre_tax = _float_array(
            cost_expenditure_pretax["gas_opex_expenditures_pre_tax"]
        )
        self.gas_asr_expenditures_pre_tax = _float_array(
            cost_expenditure_pretax["gas_asr_expenditures_pre_tax"]
        )
        self.gas_lbt_expenditures_pre_tax = _float_array(
            cost_expenditure_pretax["gas_lbt_expenditures_pre_tax"]
        )
        self.gas_cost_of_sales_expenditures_pre_tax = _float_array(
            cost_expenditure_pretax["gas_cost_of_sales_expenditures_pre_tax"]
        )
        self.consolidated_capital_expenditures_pre_tax = _float_array(
            cost_expenditure_pretax["consolidated_capital_expenditures_pre_tax"]
        )
        self.consolidated_intangible_expenditures_pre_tax = _float_array(
            cost_expenditure_pretax["consolidated_intangible_expenditures_pre_tax"]
        )
        self.consolidated_opex_expenditures_pre_tax = _float_array(
            cost_expenditure_pretax["consolidated_opex_expenditures_pre_tax"]
        )
        self.consolidated_asr_expenditures_pre_tax = _float_array(
            cost_expenditure_pretax["consolidated_asr_expenditures_pre_tax"]
        )
        self.consolidated_lbt_expenditures_pre_tax = _float_array(
            cost_expenditure_pretax["consolidated_lbt_expenditures_pre_tax"]
        )
        self.consolidated_cost_of_sales_expenditures_pre_tax = _float_array(
            cost_expenditure_pretax["consolidated_cost_of_sales_expenditures_pre_tax"]
        )

        # Cost Indirect Tax
        self.oil_capital_indirect_tax = _float_array(
            cost_indirect_tax["oil_capital_indirect_tax"]
        )
        self.oil_intangible_indirect_tax = _float_array(
            cost_indirect_tax["oil_intangible_indirect_tax"]
        )
        self.oil_opex_indirect_tax = _float_array(
            cost_indirect_tax["oil_opex_indirect_tax"]
        )
        self.oil_asr_indirect_tax = _float_array(
            cost_indirect_tax["oil_asr_indirect_tax"]
        )
        self.oil_lbt_indirect_tax = _float_array(
            cost_indirect_tax["oil_lbt_indirect_tax"]
        )
        self.oil_cost_of_sales_indirect_tax = _float_array(
            cost_indirect_tax["oil_cost_of_sales_indirect_tax"]
        )
        self.gas_capital_indirect_tax = _float_array(
            cost_indirect_tax["gas_capital_indirect_tax"]
        )
        self.gas_intangible_indirect_tax = _float_array(
            cost_indirect_tax["gas_intangible_indirect_tax"]
        )
        self.gas_opex_indirect_tax = _float_array(
            cost_indirect_tax["gas_opex_indirect_tax"]
        )
        self.gas_asr_indirect_tax = _float_array(
            cost_indirect_tax["gas_asr_indirect_tax"]
        )
        self.gas_lbt_indirect_tax = _float_array(
            cost_indirect_tax["gas_lbt_indirect_tax"]
        )
        self.gas_cost_of_sales_indirect_tax = _float_array(
            cost_indirect_tax["gas_cost_of_sales_indirect_tax"]
        )
        self.consolidated_capital_indirect_tax = _float_array(
            cost_indirect_tax["consolidated_capital_indirect_tax"]
        )
        self.consolidated_intangible_indirect_tax = _float_array(
            cost_indirect_tax["consolidated_intangible_indirect_tax"]
        )
        self.consolidated_opex_indirect_tax = _float_array(
            cost_indirect_tax["consolidated_opex_indirect_tax"]
        )
        self.consolidated_asr_indirect_tax = _float_array(
            cost_indirect_tax["consolidated_asr_indirect_tax"]
        )
        self.consolidated_lbt_indirect_tax = _float_array(
            cost_indirect_tax["consolidated_lbt_indirect_tax"]
        )
        self.consolidated_cost_of_sales_indirect_tax = _float_array(
            cost_indirect_tax["consolidated_cost_of_sales_indirect_tax"]
        )

        # Cost Post Tax
        self.oil_capital_expenditures_post_tax = _float_array(
            cost_expenditure_posttax["oil_capital_expenditures_post_tax"]
        )
        self.oil_intangible_expenditures_post_tax = _float_array(
            cost_expenditure_posttax["oil_intangible_expenditures_post_tax"]
        )
        self.oil_opex_expenditures_post_tax = _float_array(
            cost_expenditure_posttax["oil_opex_expenditures_post_tax"]
        )
        self.oil_asr_expenditures_post_tax = _float_array(
            cost_expenditure_posttax["oil_asr_expenditures_post_tax"]
        )
        self.oil_lbt_expenditures_post_tax = _float_array(
            cost_expenditure_posttax["oil_lbt_expenditures_post_tax"]
        )
        self.oil_cost_of_sales_expenditures_post_tax = _float_array(
            cost_expenditure_posttax["oil_cost_of_sales_expenditures_post_tax"]
        )
        self.gas_capital_expenditures_post_tax = _float_array(
            cost_expenditure_posttax["gas_capital_expenditures_post_tax"]
        )
        self.gas_intangible_expenditures_post_tax = _float_array(
            cost_expenditure_posttax["gas_intangible_expenditures_post_tax"]
        )
        self.gas_opex_expenditures_post_tax = _float_array(
            cost_expenditure_posttax["gas_opex_expenditures_post_tax"]
        )
        self.gas_asr_expenditures_post_tax = _float_array(
            cost_expenditure_posttax["gas_asr_expenditures_post_tax"]
        )
        self.gas_lbt_expenditures_post_tax = _float_array(
            cost_expenditure_posttax["gas_lbt_expenditures_post_tax"]
        )
        self.gas_cost_of_sales_expenditures_post_tax = _float_array(
            cost_expenditure_posttax["gas_cost_of_sales_expenditures_post_tax"]
        )
        self.consolidated_capital_expenditures_post_tax = _float_array(
            cost_expenditure_posttax["consolidated_capital_expenditures_post_tax"]
        )
        self.consolidated_intangible_expenditures_post_tax = _float_array(
            cost_expenditure_posttax["consolidated_intangible_expenditures_post_tax"]
        )
        self.consolidated_opex_expenditures_post_tax = _float_array(
            cost_expenditure_posttax["consolidated_opex_expenditures_post_tax"]
        )
        self.consolidated_asr_expenditures_post_tax = _float_array(
            cost_expenditure_posttax["consolidated_asr_expenditures_post_tax"]
        )
        self.consolidated_lbt_expenditures_post_tax = _float_array(
            cost_expenditure_posttax["consolidated_lbt_expenditures_post_tax"]
        )
        self.consolidated_cost_of_sales_expenditures_post_tax = _float_array(
            cost_expenditure_posttax["consolidated_cost_of_sales_expenditures_post_tax"]
        )

        # Contract Terms Cost
        self.oil_depreciable = _float_array(cost_contract["oil_depreciable"])
        self.oil_intangible = _float_array(cost_contract["oil_intangible"])
        self.oil_opex = _float_array(cost_contract["oil_opex"])
        self.oil_asr = _float_array(cost_contract["oil_asr"])
        self.oil_lbt = _float_array(cost_contract["oil_lbt"])
        self.oil_depreciation = _float_array(cost_contract["oil_depreciation"])
        self.oil_non_capital = _float_array(cost_contract["oil_non_capital"])
        self.oil_sunk_cost = _float_array(cost_contract["oil_sunk_cost"])
        self.oil_undepreciated_asset = _float_array(
            cost_contract["oil_undepreciated_asset"]
        )
        self.oil_carry_forward_depreciation = _float_array(
            cost_contract["oil_carry_forward_depreciation"]
        )
        self.gas_depreciable = _float_array(cost_contract["gas_depreciable"])
        self.gas_intangible = _float_array(cost_contract["gas_intangible"])
        self.gas_opex = _float_array(cost_contract["gas_opex"])
        self.gas_asr = _float_array(cost_contract["gas_asr"])
        self.gas_lbt = _float_array(cost_contract["gas_lbt"])
        self.gas_depreciation = _float_array(cost_contract["gas_depreciation"])
        self.gas_non_capital = _float_array(cost_contract["gas_non_capital"])
        self.gas_sunk_cost = _float_array(cost_contract["gas_sunk_cost"])
        self.gas_undepreciated_asset = _float_array(
            cost_contract["gas_undepreciated_asset"]
        )
        self.gas_carry_forward_depreciation = _float_array(
            cost_contract["gas_carry_forward_depreciation"]
        )
        self.consolidated_depreciable = _float_array(
            cost_contract["consolidated_depreciable"]
        )
        self.consolidated_intangible = _float_array(
            cost_contract["consolidated_intangible"]
        )
        self.consolidated_opex = _float_array(cost_contract["consolidated_opex"])
        self.consolidated_asr = _float_array(cost_contract["consolidated_asr"])
        self.consolidated_lbt = _float_array(cost_contract["consolidated_lbt"])
        self.consolidated_depreciation = _float_array(
            cost_contract["consolidated_depreciation"]
        )
        self.consolidated_non_capital = _float_array(
            cost_contract["consolidated_non_capital"]
        )
        self.consolidated_sunk_cost = _float_array(
            cost_contract["consolidated_sunk_cost"]
        )
        self.consolidated_undepreciated_asset = _float_array(
            cost_contract["consolidated_undepreciated_asset"]
        )
        self.consolidated_carry_forward_depreciation = _float_array(
            cost_contract["consolidated_carry_forward_depreciation"]
        )

        # PSC terms
        self.oil_costrecovery_or_deductible_cost = _float_array(
            psc_terms["oil_costrecovery_or_deductible_cost"]
        )
        self.oil_unrecoverable_cost_or_carryforward_cost = _float_array(
            psc_terms["oil_unrecoverable_cost_or_carryforward_cost"]
        )
        self.oil_ctr_share = _float_array(psc_terms["oil_ctr_share"])
        self.oil_gov_share = _float_array(psc_terms["oil_gov_share"])
        self.oil_effective_tax_payment = _float_array(
            psc_terms["oil_effective_tax_payment"]
        )
        self.oil_ddmo = _float_array(psc_terms["oil_ddmo"])
        self.oil_ctr_take = _float_array(psc_terms["oil_ctr_take"])
        self.oil_government_take = _float_array(psc_terms["oil_government_take"])
        self.oil_ctr_cashflow = _float_array(psc_terms["oil_ctr_cashflow"])
        self.gas_costrecovery_or_deductible_cost = _float_array(
            psc_terms["gas_costrecovery_or_deductible_cost"]
        )
        self.gas_unrecoverable_cost_or_carryforward_cost = _float_array(
            psc_terms["gas_unrecoverable_cost_or_carryforward_cost"]
        )
        self.gas_ctr_share = _float_array(psc_terms["gas_ctr_share"])
        self.gas_gov_share = _float_array(psc_terms["gas_gov_share"])
        self.gas_effective_tax_payment = _float_array(
            psc_terms["gas_effective_tax_payment"]
        )
        self.gas_ddmo = _float_array(psc_terms["gas_ddmo"])
        self.gas_ctr_take = _float_array(psc_terms["gas_ctr_take"])
        self.gas_government_take = _float_array(psc_terms["gas_government_take"])
        self.gas_ctr_cashflow = _float_array(psc_terms["gas_ctr_cashflow"])
        self.consolidated_costrecovery_or_deductible_cost = _float_array(
            psc_terms["consolidated_costrecovery_or_deductible_cost"]
        )
        self.consolidated_unrecoverable_cost_or_carryforward_cost = _float_array(
            psc_terms["consolidated_unrecoverable_cost_or_carryforward_cost"]
        )
        self.consolidated_ctr_share = _float_array(psc_terms["consolidated_ctr_share"])
        self.consolidated_ctr_net_share = _float_array(
            psc_terms["consolidated_ctr_net_share"]
        )
        self.consolidated_gov_share = _float_array(psc_terms["consolidated_gov_share"])
        self.consolidated_effective_tax_payment = _float_array(
            psc_terms["consolidated_effective_tax_payment"]
        )
        self.consolidated_ddmo = _float_array(psc_terms["consolidated_ddmo"])
        self.consolidated_ctr_take = _float_array(psc_terms["consolidated_ctr_take"])
        self.consolidated_government_take = _float_array(
            psc_terms["consolidated_government_take"]
        )
        self.consolidated_ctr_cashflow = _float_array(
            psc_terms["consolidated_ctr_cashflow"]
        )

        # Cost Recovery Terms
        self.oil_ftp_ctr = _float_array(cr_terms["oil_ftp_ctr"])
        self.gas_ftp_ctr = _float_array(cr_terms["gas_ftp_ctr"])
        self.oil_ftp_gov = _float_array(cr_terms["oil_ftp_gov"])
        self.gas_ftp_gov = _float_array(cr_terms["gas_ftp_gov"])
        self.consolidated_ctr_ftp = _float_array(cr_terms["consolidated_ctr_ftp"])
        self.consolidated_gov_ftp = _float_array(cr_terms["consolidated_gov_ftp"])

    def run(self, mode: str):
        # Get the dictionary of each element
        lifting = self._parse_dict(source=self._get_lifting_dict())
        revenue = self._parse_dict(source=self._get_revenue_dict())
        cost_expenditure_pretax = self._parse_dict(
            source=self._get_cost_expenditure_pretax_dict()
        )
        cost_indirect_tax = self._parse_dict(source=self._get_cost_indirect_tax_dict())
        cost_expenditure_posttax = self._parse_dict(
            source=self._get_cost_expenditure_posttax_dict()
        )
        cost_contract = self._parse_dict(source=self._get_cost_contract_dict())
        psc_terms = self._parse_dict(source=self._get_psc_terms_dict())
        cr_terms = self._parse_dict(source=self._get_cr_terms_dict())

        # Calculate based on the chosen method, combine or incremental
        lifting_merged = self._merge_dicts(data=lifting, mode=mode)
        revenue_merged = self._merge_dicts(data=revenue, mode=mode)
        cost_expenditure_pretax_merged = self._merge_dicts(
            data=cost_expenditure_pretax, mode=mode
        )
        cost_indirect_tax_merged = self._merge_dicts(data=cost_indirect_tax, mode=mode)
        cost_expenditure_posttax_merged = self._merge_dicts(
            data=cost_expenditure_posttax, mode=mode
        )
        cost_contract_merged = self._merge_dicts(data=cost_contract, mode=mode)
        psc_terms_merged = self._merge_dicts(data=psc_terms, mode=mode)
        cr_terms_merged = self._merge_dicts(data=cr_terms, mode=mode)

        # Set the dataclass attributes
        self._parsing_attributes(
            lifting=lifting_merged,
            revenue=revenue_merged,
            cost_expenditure_pretax=cost_expenditure_pretax_merged,
            cost_indirect_tax=cost_indirect_tax_merged,
            cost_expenditure_posttax=cost_expenditure_posttax_merged,
            cost_contract=cost_contract_merged,
            psc_terms=psc_terms_merged,
            cr_terms=cr_terms_merged,
        )

    def _to_dataframe(self) -> dict[str, object]:
        """Convert selected dataclass attributes into a pandas DataFrame."""
        exclude: list[str] = [
            "contract",
            "reference_year",
            "inflation_rate",
            "discount_rate",
            "npv_mode",
            "discounting_mode",
            "profitability_discounted",
        ]
        attributes: dict[str, object] = vars(self)
        data_dict: dict[str, object] = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in attributes.items()
            if key not in exclude
        }
        # return pd.DataFrame(data_dict)  # Convert filtered dictionary to DataFrame
        return data_dict

    def case_combine(self) -> dict[str, object]:
        self.run(mode="combine")
        return self._to_dataframe()

    def case_incremental(self) -> dict[str, object]:
        self.run(mode="incremental")
        return self._to_dataframe()

    @staticmethod
    def _require_array(value: FloatArray | None) -> FloatArray:
        if value is None:
            raise RuntimeError("Summary data is unavailable before run()")
        return value

    def _years(self) -> YearArray:
        if self.years is None:
            raise RuntimeError("Summary years are unavailable before initialization")
        return self.years

    def _reference_year(self) -> int:
        if self.reference_year is None:
            raise RuntimeError("Summary reference year is unavailable")
        return self.reference_year

    def _discounting_mode(self) -> DiscountingMode:
        return self.discounting_mode

    @staticmethod
    def _require_scalar(value: float | None) -> float:
        if value is None:
            raise RuntimeError("Summary indicator is unavailable before calculation")
        return value

    def _calc_npv(self):
        investment_cashflow = (
            self._require_array(self.consolidated_capital_expenditures_post_tax)
            + self._require_array(self.consolidated_intangible_expenditures_post_tax)
        ).astype(np.float64, copy=False)

        # NPV Calculation for SKK Real Terms
        if self.npv_mode == NPVSelection.NPV_SKK_REAL_TERMS:
            # Contractor NPV
            self.ctr_npv = npv_skk_real_terms(
                cashflow=self._require_array(self.consolidated_ctr_cashflow),
                cashflow_years=self._years(),
                discount_rate=self.discount_rate,
                reference_year=self._reference_year(),
                discounting_mode=self._discounting_mode(),
            )

            # Contractor Investment NPV
            self.investment_npv = npv_skk_real_terms(
                cashflow=investment_cashflow,
                cashflow_years=self._years(),
                discount_rate=self.discount_rate,
                reference_year=self._reference_year(),
                discounting_mode=self._discounting_mode(),
            )

            # Government Take Net Present Value
            self.gov_take_npv = npv_skk_real_terms(
                cashflow=self._require_array(self.consolidated_government_take),
                cashflow_years=self._years(),
                discount_rate=self.discount_rate,
                reference_year=self._reference_year(),
                discounting_mode=self._discounting_mode(),
            )

        # NPV Calculation for SKK Nominal Terms
        elif self.npv_mode == NPVSelection.NPV_SKK_NOMINAL_TERMS:
            # Contractor NPV
            self.ctr_npv = npv_skk_nominal_terms(
                cashflow=self._require_array(self.consolidated_ctr_cashflow),
                cashflow_years=self._years(),
                discount_rate=self.discount_rate,
                discounting_mode=self._discounting_mode(),
            )

            # Contractor Investment NPV
            self.investment_npv = npv_skk_nominal_terms(
                cashflow=investment_cashflow,
                cashflow_years=self._years(),
                discount_rate=self.discount_rate,
                discounting_mode=self._discounting_mode(),
            )

            # Government Take Net Present Value
            self.gov_take_npv = npv_skk_nominal_terms(
                cashflow=self._require_array(self.consolidated_government_take),
                cashflow_years=self._years(),
                discount_rate=self.discount_rate,
                discounting_mode=self._discounting_mode(),
            )

        # NPV Calculation for Nominal Terms
        elif self.npv_mode == NPVSelection.NPV_NOMINAL_TERMS:
            # Contractor NPV
            self.ctr_npv = npv_nominal_terms(
                cashflow=self._require_array(self.consolidated_ctr_cashflow),
                cashflow_years=self._years(),
                discount_rate=self.discount_rate,
                reference_year=self._reference_year(),
                discounting_mode=self._discounting_mode(),
            )

            # Contractor Investment NPV
            self.investment_npv = npv_nominal_terms(
                cashflow=investment_cashflow,
                cashflow_years=self._years(),
                discount_rate=self.discount_rate,
                reference_year=self._reference_year(),
                discounting_mode=self._discounting_mode(),
            )

            # Government Take Net Present Value
            self.gov_take_npv = npv_nominal_terms(
                cashflow=self._require_array(self.consolidated_government_take),
                cashflow_years=self._years(),
                discount_rate=self.discount_rate,
                reference_year=self._reference_year(),
                discounting_mode=self._discounting_mode(),
            )

        # NPV Calculation for Real Terms
        elif self.npv_mode == NPVSelection.NPV_REAL_TERMS:
            # Contractor NPV
            self.ctr_npv = npv_real_terms(
                cashflow=self._require_array(self.consolidated_ctr_cashflow),
                cashflow_years=self._years(),
                discount_rate=self.discount_rate,
                reference_year=self._reference_year(),
                inflation_rate=self.inflation_rate,
                discounting_mode=self._discounting_mode(),
            )

            # Contractor Investment NPV
            self.investment_npv = npv_real_terms(
                cashflow=investment_cashflow,
                cashflow_years=self._years(),
                discount_rate=self.discount_rate,
                reference_year=self._reference_year(),
                inflation_rate=self.inflation_rate,
                discounting_mode=self._discounting_mode(),
            )

            # Government Take Net Present Value
            self.gov_take_npv = npv_real_terms(
                cashflow=self._require_array(self.consolidated_government_take),
                cashflow_years=self._years(),
                discount_rate=self.discount_rate,
                reference_year=self._reference_year(),
                inflation_rate=self.inflation_rate,
                discounting_mode=self._discounting_mode(),
            )

        # NPV Calculation for Point Forwards
        else:
            # Contractor NPV
            self.ctr_npv = npv_point_forward(
                cashflow=self._require_array(self.consolidated_ctr_cashflow),
                cashflow_years=self._years(),
                discount_rate=self.discount_rate,
                reference_year=self._reference_year(),
                discounting_mode=self._discounting_mode(),
            )

            # Contractor Investment NPV
            self.investment_npv = npv_point_forward(
                cashflow=investment_cashflow,
                cashflow_years=self._years(),
                discount_rate=self.discount_rate,
                reference_year=self._reference_year(),
                discounting_mode=self._discounting_mode(),
            )

            # Government Take Net Present Value
            self.gov_take_npv = npv_point_forward(
                cashflow=self._require_array(self.consolidated_government_take),
                cashflow_years=self._years(),
                discount_rate=self.discount_rate,
                reference_year=self._reference_year(),
                discounting_mode=self._discounting_mode(),
            )

    def _calc_pot(self):
        # Contractor POT
        self.pot = pot_psc(
            cashflow=self._require_array(self.consolidated_ctr_cashflow),
            cashflow_years=self._years(),
            reference_year=self._reference_year(),
        )

    def _calc_pv_ratio(self):
        if self.profitability_discounted:
            numerator = self._require_scalar(self.ctr_npv)
            denominator = self._require_scalar(self.investment_npv)
            self.pv_ratio = numerator / denominator if denominator != 0 else 0.0

        # Condition when the Profitability Index is calculated using un-discounted investment
        else:
            investment = _sum_float(
                self._require_array(self.consolidated_capital_expenditures_post_tax)
                + self._require_array(
                    self.consolidated_intangible_expenditures_post_tax
                )
                + self._require_array(self.consolidated_sunk_cost)
            )
            numerator = self._require_scalar(self.ctr_npv)
            self.pv_ratio = float(numerator / investment) if investment != 0 else 0.0

        self.pi = 1 + self._require_scalar(self.pv_ratio)

    def _calc_irr(self):
        self.irr = irr(cashflow=self._require_array(self.consolidated_ctr_cashflow))

    def get_executive_summary(self) -> dict[str, SummaryValue]:
        # Lifting Group
        lifting_oil = _sum_float(self._require_array(self.oil_lifting))
        lifting_gas = _sum_float(self._require_array(self.gas_lifting))

        # Gross Revenue Group
        gross_revenue_oil = _sum_float(self._require_array(self.oil_revenue))
        gross_revenue_gas = _sum_float(self._require_array(self.gas_revenue))
        gross_revenue = gross_revenue_oil + gross_revenue_gas

        # WAP Group
        oil_wap = _ratio_float(gross_revenue_oil, lifting_oil)
        gas_wap = _ratio_float(gross_revenue_gas, lifting_gas)

        # Contractor and Government Gross share
        ctr_gross_share = _sum_float(self._require_array(self.consolidated_ctr_share))
        gov_gross_share = _sum_float(self._require_array(self.consolidated_gov_share))

        # Investment Group
        sunk_cost = _sum_float(self._require_array(self.consolidated_sunk_cost))
        tangible = _sum_float(
            self._require_array(self.consolidated_capital_expenditures_post_tax)
        )
        intangible = _sum_float(
            self._require_array(self.consolidated_intangible_expenditures_post_tax)
        )
        investment = tangible + intangible

        # OPEX and ASR Group
        opex = _sum_float(
            self._require_array(self.consolidated_opex_expenditures_post_tax)
        )
        asr = _sum_float(
            self._require_array(self.consolidated_asr_expenditures_post_tax)
        )
        lbt = _sum_float(
            self._require_array(self.consolidated_lbt_expenditures_post_tax)
        )
        opex_asr_lbt = opex + asr + lbt

        # Cost recovery or Deductible Cost
        costrecovery_or_deductible_cost = _sum_float(
            self._require_array(self.consolidated_costrecovery_or_deductible_cost)
        )
        cost_recovery_over_gross_rev = _ratio_float(
            costrecovery_or_deductible_cost, gross_revenue
        )

        # Unrecoverable Cost (terminal balance, matching the per-case summary)
        unrec_cost = _sum_float(
            self._require_array(
                self.consolidated_unrecoverable_cost_or_carryforward_cost
            )[-1:]
        )
        unrec_over_gross_rev = _ratio_float(unrec_cost, gross_revenue)

        # Contractor Net Share
        ctr_net_share = _sum_float(self._require_array(self.consolidated_ctr_net_share))
        ctr_net_share_over_gross_share = _ratio_float(ctr_net_share, gross_revenue)

        # Contractor Net Cashflow
        ctr_net_cashflow = _sum_float(
            self._require_array(self.consolidated_ctr_cashflow)
        )
        ctr_net_cashflow_over_gross_rev = _ratio_float(ctr_net_cashflow, gross_revenue)

        # Contractor NPV
        self._calc_npv()
        ctr_npv = self._require_scalar(self.ctr_npv)

        # Investment NPV
        _investment_npv = self._require_scalar(self.investment_npv)

        # Government Take NPV
        gov_take_npv = self._require_scalar(self.gov_take_npv)

        # POT
        self._calc_pot()
        pot = self._require_scalar(self.pot)

        # PV_Ratio
        self._calc_pv_ratio()
        pv_ratio = self._require_scalar(self.pv_ratio)
        pi = self.pi

        # Contractor IRR
        self._calc_irr()
        ctr_irr = self._require_scalar(self.irr)

        # Government Indicator
        gov_ftp_share = _sum_float(self._require_array(self.consolidated_gov_ftp))
        _gov_equity_share = _sum_float(self._require_array(self.consolidated_gov_share))
        gov_ddmo = _sum_float(self._require_array(self.consolidated_ddmo))
        gov_tax_income = _sum_float(
            self._require_array(self.consolidated_effective_tax_payment)
        )
        gov_take = _sum_float(self._require_array(self.consolidated_government_take))
        gov_take_over_gross_rev = _ratio_float(gov_take, gross_revenue)

        oil_indirect_taxes = _sum_float(
            self._require_array(self.oil_capital_indirect_tax)
            + self._require_array(self.oil_intangible_indirect_tax)
            + self._require_array(self.oil_opex_indirect_tax)
            + self._require_array(self.oil_asr_indirect_tax)
            + self._require_array(self.oil_lbt_indirect_tax)
            + self._require_array(self.oil_cost_of_sales_indirect_tax)
        )
        gas_indirect_taxes = _sum_float(
            self._require_array(self.gas_capital_indirect_tax)
            + self._require_array(self.gas_intangible_indirect_tax)
            + self._require_array(self.gas_opex_indirect_tax)
            + self._require_array(self.gas_asr_indirect_tax)
            + self._require_array(self.gas_lbt_indirect_tax)
            + self._require_array(self.gas_cost_of_sales_indirect_tax)
        )
        total_indirect_taxes = oil_indirect_taxes + gas_indirect_taxes

        # Undepreciated Asset
        undepreciated_asset_oil = _sum_float(
            self._require_array(self.oil_undepreciated_asset)
        )
        undepreciated_asset_gas = _sum_float(
            self._require_array(self.gas_undepreciated_asset)
        )
        undepreciated_asset_total = undepreciated_asset_oil + undepreciated_asset_gas

        # Carry Forward Depreciation
        oil_carry_forward_depreciation = _sum_float(
            self._require_array(self.oil_carry_forward_depreciation)
        )
        gas_carry_forward_depreciation = _sum_float(
            self._require_array(self.gas_carry_forward_depreciation)
        )
        total_carry_forward_depreciation = (
            oil_carry_forward_depreciation + gas_carry_forward_depreciation
        )

        return {
            "lifting_oil": lifting_oil,
            "oil_wap": oil_wap,
            "lifting_gas": lifting_gas,
            "gas_wap": gas_wap,
            "gross_revenue": gross_revenue,
            "ctr_gross_share": ctr_gross_share,
            "sunk_cost": sunk_cost,
            "investment": investment,
            "tangible": tangible,
            "intangible": intangible,
            "opex_asr_lbt": opex_asr_lbt,
            "opex": opex,
            "asr": asr,
            "cost_recovery/deductible_cost": costrecovery_or_deductible_cost,
            "cost_recovery_over_gross_rev": cost_recovery_over_gross_rev,
            "unrec_cost": unrec_cost,
            "unrec_over_gross_rev": unrec_over_gross_rev,
            "ctr_net_share": ctr_net_share,
            "ctr_net_share_over_gross_share": ctr_net_share_over_gross_share,
            "ctr_net_cashflow": ctr_net_cashflow,
            "ctr_net_cashflow_over_gross_rev": ctr_net_cashflow_over_gross_rev,
            "ctr_npv": ctr_npv,
            "ctr_irr": ctr_irr,
            "ctr_pot": pot,
            "ctr_pv_ratio": pv_ratio,
            "ctr_pi": pi,
            "gov_gross_share": gov_gross_share,
            "gov_ftp_share": gov_ftp_share,
            "gov_ddmo": gov_ddmo,
            "gov_tax_income": gov_tax_income,
            "gov_take": gov_take,
            "gov_take_over_gross_rev": gov_take_over_gross_rev,
            "gov_take_npv": gov_take_npv,
            "undepreciated_asset_oil": undepreciated_asset_oil,
            "undepreciated_asset_gas": undepreciated_asset_gas,
            "undepreciated_asset_total": undepreciated_asset_total,
            "total_indirect_taxes": total_indirect_taxes,
            "oil_indirect_taxes": oil_indirect_taxes,
            "gas_indirect_taxes": gas_indirect_taxes,
            "total_carry_forward_depreciation": total_carry_forward_depreciation,
            "oil_carry_forward_depreciation": oil_carry_forward_depreciation,
            "gas_carry_forward_depreciation": gas_carry_forward_depreciation,
        }

    def get_cashflow_table(self):
        return self._to_dataframe()
