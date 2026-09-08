"""Named post-run data consumed by legacy summary reporting."""

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from pyscnomics.econ.revenue import Lifting

FloatArray: TypeAlias = npt.NDArray[np.float64]
YearArray: TypeAlias = npt.NDArray[np.int64]


@dataclass(frozen=True)
class PostRunSummaryProfile:
    """Flat compatibility snapshot for reporting generated after ``run()``."""

    contract_type: Literal["base_project", "cost_recovery", "gross_split", "transition"]
    project_years: YearArray
    co2_lifting: Lifting | None = None
    co2_revenue: FloatArray | None = None
    consolidated_carry_forward_depreciation: FloatArray | None = None
    consolidated_carward_cost_aftertf: FloatArray | None = None
    consolidated_carward_deduct_cost: FloatArray | None = None
    consolidated_cashflow: FloatArray | None = None
    consolidated_contractor_share: FloatArray | None = None
    consolidated_contractor_take: FloatArray | None = None
    consolidated_cost_recovery_after_tf: FloatArray | None = None
    consolidated_ctr_net_share: FloatArray | None = None
    consolidated_ctr_share_after_transfer: FloatArray | None = None
    consolidated_ctr_share_before_tf: FloatArray | None = None
    consolidated_ddmo: FloatArray | None = None
    consolidated_deductible_cost: FloatArray | None = None
    consolidated_ftp_gov: FloatArray | None = None
    consolidated_gov_share_before_tf: FloatArray | None = None
    consolidated_government_share: FloatArray | None = None
    consolidated_government_take: FloatArray | None = None
    consolidated_revenue: FloatArray | None = None
    consolidated_sunk_cost: FloatArray | None = None
    consolidated_tax_payment: FloatArray | None = None
    consolidated_unrecovered_after_transfer: FloatArray | None = None
    electricity_lifting: Lifting | None = None
    electricity_revenue: FloatArray | None = None
    gas_asr_expenditures_post_tax: FloatArray | None = None
    gas_asr_expenditures_pre_tax: FloatArray | None = None
    gas_asr_indirect_tax: FloatArray | None = None
    gas_capital_expenditures_post_tax: FloatArray | None = None
    gas_capital_expenditures_pre_tax: FloatArray | None = None
    gas_capital_indirect_tax: FloatArray | None = None
    gas_carry_forward_depreciation: FloatArray | None = None
    gas_carward_cost_aftertf: FloatArray | None = None
    gas_cashflow: FloatArray | None = None
    gas_contractor_share: FloatArray | None = None
    gas_contractor_take: FloatArray | None = None
    gas_cost_of_sales_expenditures_post_tax: FloatArray | None = None
    gas_cost_of_sales_expenditures_pre_tax: FloatArray | None = None
    gas_cost_of_sales_indirect_tax: FloatArray | None = None
    gas_cost_recovery_after_tf: FloatArray | None = None
    gas_ctr_cashflow: FloatArray | None = None
    gas_ctr_ets: FloatArray | None = None
    gas_ctr_net_share: FloatArray | None = None
    gas_ctr_share_after_transfer: FloatArray | None = None
    gas_ctr_take: FloatArray | None = None
    gas_ddmo: FloatArray | None = None
    gas_deductible_cost: FloatArray | None = None
    gas_depreciation: FloatArray | None = None
    gas_effective_tax_payment: FloatArray | None = None
    gas_ftp_ctr: FloatArray | None = None
    gas_ftp_gov: FloatArray | None = None
    gas_gov_ets: FloatArray | None = None
    gas_gov_share: FloatArray | None = None
    gas_government_share: FloatArray | None = None
    gas_government_take: FloatArray | None = None
    gas_intangible_expenditures_post_tax: FloatArray | None = None
    gas_intangible_expenditures_pre_tax: FloatArray | None = None
    gas_intangible_indirect_tax: FloatArray | None = None
    gas_lbt_expenditures_post_tax: FloatArray | None = None
    gas_lbt_expenditures_pre_tax: FloatArray | None = None
    gas_lbt_indirect_tax: FloatArray | None = None
    gas_lifting: Lifting | None = None
    gas_non_capital: FloatArray | None = None
    gas_opex_expenditures_post_tax: FloatArray | None = None
    gas_opex_expenditures_pre_tax: FloatArray | None = None
    gas_opex_indirect_tax: FloatArray | None = None
    gas_revenue: FloatArray | None = None
    gas_sum_undepreciated_asset: float | np.float64 | None = None
    gas_sunk_cost: FloatArray | None = None
    gas_tax: FloatArray | None = None
    gas_tax_payment: FloatArray | None = None
    gas_total_indirect_tax: FloatArray | None = None
    gas_unrec_cost: FloatArray | None = None
    gas_undepreciated_asset: FloatArray | None = None
    gas_unrecovered_after_transfer: FloatArray | None = None
    gas_wap_price: FloatArray | None = None
    oil_asr_expenditures_post_tax: FloatArray | None = None
    oil_asr_expenditures_pre_tax: FloatArray | None = None
    oil_asr_indirect_tax: FloatArray | None = None
    oil_capital_expenditures_post_tax: FloatArray | None = None
    oil_capital_expenditures_pre_tax: FloatArray | None = None
    oil_capital_indirect_tax: FloatArray | None = None
    oil_carry_forward_depreciation: FloatArray | None = None
    oil_carward_cost_aftertf: FloatArray | None = None
    oil_cashflow: FloatArray | None = None
    oil_contractor_share: FloatArray | None = None
    oil_contractor_take: FloatArray | None = None
    oil_cost_of_sales_expenditures_post_tax: FloatArray | None = None
    oil_cost_of_sales_expenditures_pre_tax: FloatArray | None = None
    oil_cost_of_sales_indirect_tax: FloatArray | None = None
    oil_cost_recovery_after_tf: FloatArray | None = None
    oil_ctr_cashflow: FloatArray | None = None
    oil_ctr_ets: FloatArray | None = None
    oil_ctr_net_share: FloatArray | None = None
    oil_ctr_share_after_transfer: FloatArray | None = None
    oil_ctr_take: FloatArray | None = None
    oil_ddmo: FloatArray | None = None
    oil_deductible_cost: FloatArray | None = None
    oil_depreciation: FloatArray | None = None
    oil_effective_tax_payment: FloatArray | None = None
    oil_ftp_ctr: FloatArray | None = None
    oil_ftp_gov: FloatArray | None = None
    oil_gov_ets: FloatArray | None = None
    oil_gov_share: FloatArray | None = None
    oil_government_share: FloatArray | None = None
    oil_government_take: FloatArray | None = None
    oil_intangible_expenditures_post_tax: FloatArray | None = None
    oil_intangible_expenditures_pre_tax: FloatArray | None = None
    oil_intangible_indirect_tax: FloatArray | None = None
    oil_lbt_expenditures_post_tax: FloatArray | None = None
    oil_lbt_expenditures_pre_tax: FloatArray | None = None
    oil_lbt_indirect_tax: FloatArray | None = None
    oil_lifting: Lifting | None = None
    oil_non_capital: FloatArray | None = None
    oil_opex_expenditures_post_tax: FloatArray | None = None
    oil_opex_expenditures_pre_tax: FloatArray | None = None
    oil_opex_indirect_tax: FloatArray | None = None
    oil_revenue: FloatArray | None = None
    oil_sum_undepreciated_asset: float | np.float64 | None = None
    oil_sunk_cost: FloatArray | None = None
    oil_tax: FloatArray | None = None
    oil_tax_payment: FloatArray | None = None
    oil_total_indirect_tax: FloatArray | None = None
    oil_unrec_cost: FloatArray | None = None
    oil_undepreciated_asset: FloatArray | None = None
    oil_unrecovered_after_transfer: FloatArray | None = None
    oil_wap_price: FloatArray | None = None
    sulfur_lifting: Lifting | None = None
    sulfur_revenue: FloatArray | None = None
    ctr_ftp: FloatArray | None = None
    ctr_net_cashflow: FloatArray | None = None
    gov_ftp: FloatArray | None = None
    net_operating_profit: FloatArray | None = None
    oil_preonstream_cost_array: dict[str, FloatArray] | None = None
    oil_preonstream_cost_bulk: dict[str, FloatArray] | None = None
    oil_sunk_cost_array: dict[str, FloatArray] | None = None
    oil_sunk_cost_bulk: dict[str, FloatArray] | None = None
    gas_preonstream_cost_array: dict[str, FloatArray] | None = None
    gas_preonstream_cost_bulk: dict[str, FloatArray] | None = None
    gas_sunk_cost_array: dict[str, FloatArray] | None = None
    gas_sunk_cost_bulk: dict[str, FloatArray] | None = None
