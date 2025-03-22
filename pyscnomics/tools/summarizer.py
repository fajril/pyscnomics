from typing import Tuple, Union
from dataclasses import dataclass, field
import numpy as np

from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts.transition import Transition
from pyscnomics.econ.selection import NPVSelection, DiscountingMode
from pyscnomics.econ.indicator import (irr,
                                       npv_nominal_terms,
                                       npv_real_terms,
                                       npv_skk_nominal_terms,
                                       npv_skk_real_terms,
                                       npv_point_forward,
                                       pot_psc)

@dataclass
class ExecutiveSummary:
    lifting_oil: float = field(default=None, init=False, repr=False)
    oil_wap: float = field(default=None, init=False, repr=False)
    lifting_gas: float = field(default=None, init=False, repr=False)
    gas_wap: float = field(default=None, init=False, repr=False)
    gross_revenue: float = field(default=None, init=False, repr=False)
    gross_revenue_oil: float = field(default=None, init=False, repr=False)
    gross_revenue_gas: float = field(default=None, init=False, repr=False)
    ctr_gross_share: float = field(default=None, init=False, repr=False)
    gov_gross_share: float = field(default=None, init=False, repr=False)
    investment: float = field(default=None, init=False, repr=False)
    oil_capex: float = field(default=None, init=False, repr=False)
    gas_capex: float = field(default=None, init=False, repr=False)
    sunk_cost: float = field(default=None, init=False, repr=False)
    tangible: float = field(default=None, init=False, repr=False)
    intangible: float = field(default=None, init=False, repr=False)
    opex_asr_lbt: float = field(default=None, init=False, repr=False)
    opex: float = field(default=None, init=False, repr=False)
    asr: float = field(default=None, init=False, repr=False)
    lbt: float = field(default=None, init=False, repr=False)
    cost_recovery_deductible_cost: float = field(default=None, init=False, repr=False)
    cost_recovery_over_gross_rev: float = field(default=None, init=False, repr=False)
    unrec_cost: float = field(default=None, init=False, repr=False)
    unrec_over_costrec: float = field(default=None, init=False, repr=False)
    unrec_over_gross_rev: float = field(default=None, init=False, repr=False)
    ctr_net_share: float = field(default=None, init=False, repr=False)
    ctr_net_share_over_gross_share: float = field(default=None, init=False, repr=False)
    ctr_net_cashflow: float = field(default=None, init=False, repr=False)
    ctr_net_cashflow_over_gross_rev: float = field(default=None, init=False, repr=False)
    ctr_npv: float = field(default=None, init=False, repr=False)
    ctr_npv_sunk_cost_pooled: float = field(default=None, init=False, repr=False)
    ctr_irr: float = field(default=None, init=False, repr=False)
    ctr_irr_sunk_cost_pooled: float = field(default=None, init=False, repr=False)
    ctr_pot: float = field(default=None, init=False, repr=False)
    ctr_pv_ratio: float = field(default=None, init=False, repr=False)
    ctr_pi: float = field(default=None, init=False, repr=False)
    gov_ftp_share: float = field(default=None, init=False, repr=False)
    gov_equity_share: float = field(default=None, init=False, repr=False)
    gov_ddmo: float = field(default=None, init=False, repr=False)
    gov_tax_income: float = field(default=None, init=False, repr=False)
    gov_take: float = field(default=None, init=False, repr=False)
    gov_take_over_gross_rev: float = field(default=None, init=False, repr=False)
    gov_take_npv: float = field(default=None, init=False, repr=False)
    undepreciated_asset_oil: float = field(default=None, init=False, repr=False)
    undepreciated_asset_gas: float = field(default=None, init=False, repr=False)
    undepreciated_asset_total: float = field(default=None, init=False, repr=False)
    total_indirect_taxes: float = field(default=None, init=False, repr=False)
    oil_indirect_taxes: float = field(default=None, init=False, repr=False)
    gas_indirect_taxes: float = field(default=None, init=False, repr=False)


@dataclass
class Summary:
    contract: Tuple[Union[BaseProject, CostRecovery, GrossSplit, Transition]]
    reference_year: int | None = field(default=None)
    inflation_rate: float = field(default=0.0)
    discount_rate: float = field(default=0.1)
    npv_mode: NPVSelection = field(default=NPVSelection.NPV_SKK_REAL_TERMS)
    discounting_mode: DiscountingMode = field(default=DiscountingMode)
    profitability_discounted: bool = True

    # Executive Summary
    executive_summary: ExecutiveSummary = field(default=None, init=False, repr=False)

    # Economic Indicator
    ctr_npv: float = field(default=None, init=False, repr=False)
    investment_npv: float = field(default=None, init=False, repr=False)
    gov_take_npv: float = field(default=None, init=False, repr=False)
    pot: float = field(default=None, init=False, repr=False)
    pv_ratio: float = field(default=None, init=False, repr=False)
    pi: float = field(default=None, init=False, repr=False)
    irr: float = field(default=None, init=False, repr=False)

    # Years
    years: np.ndarray = field(default=None, init=False, repr=False)

    # Lifting Attributes
    oil_lifting: np.ndarray = field(default=None, init=False, repr=False)
    gas_lifting: np.ndarray = field(default=None, init=False, repr=False)
    sulfur_lifting: np.ndarray = field(default=None, init=False, repr=False)
    electricity_lifting: np.ndarray = field(default=None, init=False, repr=False)
    co2_lifting: np.ndarray = field(default=None, init=False, repr=False)

    # Revenue
    oil_revenue: np.ndarray = field(default=None, init=False, repr=False)
    gas_revenue: np.ndarray = field(default=None, init=False, repr=False)
    sulfur_revenue: np.ndarray = field(default=None, init=False, repr=False)
    electricity_revenue: np.ndarray = field(default=None, init=False, repr=False)
    co2_revenue: np.ndarray = field(default=None, init=False, repr=False)

    # Cost Expenditures Pre Tax
    oil_capital_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    oil_intangible_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    oil_opex_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    oil_asr_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    oil_lbt_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    oil_cost_of_sales_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    gas_capital_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    gas_intangible_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    gas_opex_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    gas_asr_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    gas_lbt_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    gas_cost_of_sales_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_capital_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_intangible_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_opex_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_asr_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_lbt_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_cost_of_sales_expenditures_pre_tax: np.ndarray = field(default=None, init=False, repr=False)

    # Cost Indirect Tax
    oil_capital_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    oil_intangible_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    oil_opex_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    oil_asr_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    oil_lbt_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    oil_cost_of_sales_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    gas_capital_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    gas_intangible_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    gas_opex_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    gas_asr_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    gas_lbt_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    gas_cost_of_sales_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_capital_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_intangible_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_opex_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_asr_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_lbt_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_cost_of_sales_indirect_tax: np.ndarray = field(default=None, init=False, repr=False)

    # Cost Post Tax
    oil_capital_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    oil_intangible_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    oil_opex_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    oil_asr_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    oil_lbt_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    oil_cost_of_sales_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    gas_capital_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    gas_intangible_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    gas_opex_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    gas_asr_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    gas_lbt_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    gas_cost_of_sales_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_capital_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_intangible_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_opex_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_asr_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_lbt_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_cost_of_sales_expenditures_post_tax: np.ndarray = field(default=None, init=False, repr=False)

    # Contract Terms Cost
    oil_depreciable: np.ndarray = field(default=None, init=False, repr=False)
    oil_intangible: np.ndarray = field(default=None, init=False, repr=False)
    oil_opex: np.ndarray = field(default=None, init=False, repr=False)
    oil_asr: np.ndarray = field(default=None, init=False, repr=False)
    oil_lbt: np.ndarray = field(default=None, init=False, repr=False)
    oil_depreciation: np.ndarray = field(default=None, init=False, repr=False)
    oil_non_capital: np.ndarray = field(default=None, init=False, repr=False)
    oil_sunk_cost: np.ndarray = field(default=None, init=False, repr=False)
    oil_undepreciated_asset: np.ndarray = field(default=None, init=False, repr=False)
    gas_depreciable: np.ndarray = field(default=None, init=False, repr=False)
    gas_intangible: np.ndarray = field(default=None, init=False, repr=False)
    gas_opex: np.ndarray = field(default=None, init=False, repr=False)
    gas_asr: np.ndarray = field(default=None, init=False, repr=False)
    gas_lbt: np.ndarray = field(default=None, init=False, repr=False)
    gas_depreciation: np.ndarray = field(default=None, init=False, repr=False)
    gas_non_capital: np.ndarray = field(default=None, init=False, repr=False)
    gas_sunk_cost: np.ndarray = field(default=None, init=False, repr=False)
    gas_undepreciated_asset: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_depreciable: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_intangible: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_opex: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_asr: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_lbt: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_depreciation: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_non_capital: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_sunk_cost: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_undepreciated_asset: np.ndarray = field(default=None, init=False, repr=False)

    # PSC terms
    oil_costrecovery_or_deductible_cost: np.ndarray = field(default=None, init=False, repr=False)
    oil_unrecoverable_cost_or_carryforward_cost: np.ndarray = field(default=None, init=False, repr=False)
    oil_ctr_share: np.ndarray = field(default=None, init=False, repr=False)
    oil_gov_share: np.ndarray = field(default=None, init=False, repr=False)
    oil_effective_tax_payment: np.ndarray = field(default=None, init=False, repr=False)
    oil_ddmo: np.ndarray = field(default=None, init=False, repr=False)
    oil_ctr_take: np.ndarray = field(default=None, init=False, repr=False)
    oil_government_take: np.ndarray = field(default=None, init=False, repr=False)
    oil_ctr_cashflow: np.ndarray = field(default=None, init=False, repr=False)
    gas_costrecovery_or_deductible_cost: np.ndarray = field(default=None, init=False, repr=False)
    gas_unrecoverable_cost_or_carryforward_cost: np.ndarray = field(default=None, init=False, repr=False)
    gas_ctr_share: np.ndarray = field(default=None, init=False, repr=False)
    gas_gov_share: np.ndarray = field(default=None, init=False, repr=False)
    gas_effective_tax_payment: np.ndarray = field(default=None, init=False, repr=False)
    gas_ddmo: np.ndarray = field(default=None, init=False, repr=False)
    gas_ctr_take: np.ndarray = field(default=None, init=False, repr=False)
    gas_government_take: np.ndarray = field(default=None, init=False, repr=False)
    gas_ctr_cashflow: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_costrecovery_or_deductible_cost: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_unrecoverable_cost_or_carryforward_cost: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_ctr_share: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_gov_share: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_effective_tax_payment: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_ddmo: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_ctr_take: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_government_take: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_ctr_cashflow: np.ndarray = field(default=None, init=False, repr=False)

    # Cost Recovery Terms
    oil_ftp_ctr: np.ndarray = field(default=None, init=False, repr=False)
    gas_ftp_ctr: np.ndarray = field(default=None, init=False, repr=False)
    oil_ftp_gov: np.ndarray = field(default=None, init=False, repr=False)
    gas_ftp_gov: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_ctr_ftp: np.ndarray = field(default=None, init=False, repr=False)
    consolidated_gov_ftp: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # Defining overall project years
        project_years = np.concatenate([contract.project_years for contract in self.contract])
        min_year = min(project_years)
        max_year = max(project_years)

        self.years = np.arange(
            min_year, max_year + 1, 1
        )

    def _get_lifting_dict(self):
        return {
            idx: {
                'years': contract.project_years,
                'oil_lifting': contract._oil_lifting.get_lifting_rate_arr(),
                'gas_lifting': contract._gas_lifting.get_lifting_rate_arr(),
                'sulfur_lifting': contract._sulfur_lifting.get_lifting_rate_arr(),
                'electricity_lifting': contract._electricity_lifting.get_lifting_rate_arr(),
                'co2_lifting': contract._co2_lifting.get_lifting_rate_arr(),
            }
            for idx, contract in enumerate(self.contract)
        }

    def _get_revenue_dict(self):
        return {
            idx: {
                'years': contract.project_years,
                'oil_revenue': contract._oil_revenue,
                'gas_revenue': contract._gas_revenue,
                'sulfur_revenue': contract._sulfur_revenue,
                'electricity_revenue': contract._electricity_revenue,
                'co2_revenue': contract._co2_revenue,
            }
            for idx, contract in enumerate(self.contract)
        }

    def _get_cost_expenditure_pretax_dict(self):
        return {
            idx: {
                'years': contract.project_years,
                'oil_capital_expenditures_pre_tax': contract._oil_capital_expenditures_pre_tax,
                'oil_intangible_expenditures_pre_tax': contract._oil_intangible_expenditures_pre_tax,
                'oil_opex_expenditures_pre_tax': contract._oil_opex_expenditures_pre_tax,
                'oil_asr_expenditures_pre_tax': contract._oil_asr_expenditures_pre_tax,
                'oil_lbt_expenditures_pre_tax': contract._oil_lbt_expenditures_pre_tax,
                'oil_cost_of_sales_expenditures_pre_tax': contract._oil_cost_of_sales_expenditures_pre_tax,
                'gas_capital_expenditures_pre_tax': contract._gas_capital_expenditures_pre_tax,
                'gas_intangible_expenditures_pre_tax': contract._gas_intangible_expenditures_pre_tax,
                'gas_opex_expenditures_pre_tax': contract._gas_opex_expenditures_pre_tax,
                'gas_asr_expenditures_pre_tax': contract._gas_asr_expenditures_pre_tax,
                'gas_lbt_expenditures_pre_tax': contract._gas_lbt_expenditures_pre_tax,
                'gas_cost_of_sales_expenditures_pre_tax': contract._gas_cost_of_sales_expenditures_pre_tax,
                'consolidated_capital_expenditures_pre_tax': contract._oil_capital_expenditures_pre_tax + contract._gas_capital_expenditures_pre_tax,
                'consolidated_intangible_expenditures_pre_tax': contract._oil_intangible_expenditures_pre_tax + contract._gas_intangible_expenditures_pre_tax,
                'consolidated_opex_expenditures_pre_tax': contract._oil_opex_expenditures_pre_tax + contract._gas_opex_expenditures_pre_tax,
                'consolidated_asr_expenditures_pre_tax': contract._oil_asr_expenditures_pre_tax + contract._gas_asr_expenditures_pre_tax,
                'consolidated_lbt_expenditures_pre_tax': contract._oil_lbt_expenditures_pre_tax + contract._gas_lbt_expenditures_pre_tax,
                'consolidated_cost_of_sales_expenditures_pre_tax': contract._oil_cost_of_sales_expenditures_pre_tax + contract._gas_cost_of_sales_expenditures_pre_tax,
            }
            for idx, contract in enumerate(self.contract)
        }

    def _get_cost_indirect_tax_dict(self):
        return {
            idx: {
                'years': contract.project_years,
                'oil_capital_indirect_tax': contract._oil_capital_indirect_tax,
                'oil_intangible_indirect_tax': contract._oil_intangible_indirect_tax,
                'oil_opex_indirect_tax': contract._oil_opex_indirect_tax,
                'oil_asr_indirect_tax': contract._oil_asr_indirect_tax,
                'oil_lbt_indirect_tax': contract._oil_lbt_indirect_tax,
                'oil_cost_of_sales_indirect_tax': contract._oil_cost_of_sales_indirect_tax,
                'gas_capital_indirect_tax': contract._gas_capital_indirect_tax,
                'gas_intangible_indirect_tax': contract._gas_intangible_indirect_tax,
                'gas_opex_indirect_tax': contract._gas_opex_indirect_tax,
                'gas_asr_indirect_tax': contract._gas_asr_indirect_tax,
                'gas_lbt_indirect_tax': contract._gas_lbt_indirect_tax,
                'gas_cost_of_sales_indirect_tax': contract._gas_cost_of_sales_indirect_tax,
                'consolidated_capital_indirect_tax': contract._oil_capital_indirect_tax + contract._gas_capital_indirect_tax,
                'consolidated_intangible_indirect_tax': contract._oil_intangible_indirect_tax + contract._gas_intangible_indirect_tax,
                'consolidated_opex_indirect_tax': contract._oil_opex_indirect_tax + contract._gas_opex_indirect_tax,
                'consolidated_asr_indirect_tax': contract._oil_asr_indirect_tax + contract._gas_asr_indirect_tax,
                'consolidated_lbt_indirect_tax': contract._oil_lbt_indirect_tax + contract._gas_lbt_indirect_tax,
                'consolidated_cost_of_sales_indirect_tax': contract._oil_cost_of_sales_indirect_tax + contract._gas_cost_of_sales_indirect_tax,
            }
            for idx, contract in enumerate(self.contract)
        }

    def _get_cost_expenditure_posttax_dict(self):
        return {
            idx: {
                'years': contract.project_years,
                'oil_capital_expenditures_post_tax': contract._oil_capital_expenditures_post_tax,
                'oil_intangible_expenditures_post_tax': contract._oil_intangible_expenditures_post_tax,
                'oil_opex_expenditures_post_tax': contract._oil_opex_expenditures_post_tax,
                'oil_asr_expenditures_post_tax': contract._oil_asr_expenditures_post_tax,
                'oil_lbt_expenditures_post_tax': contract._oil_lbt_expenditures_post_tax,
                'oil_cost_of_sales_expenditures_post_tax': contract._oil_cost_of_sales_expenditures_post_tax,
                'gas_capital_expenditures_post_tax': contract._gas_capital_expenditures_post_tax,
                'gas_intangible_expenditures_post_tax': contract._gas_intangible_expenditures_post_tax,
                'gas_opex_expenditures_post_tax': contract._gas_opex_expenditures_post_tax,
                'gas_asr_expenditures_post_tax': contract._gas_asr_expenditures_post_tax,
                'gas_lbt_expenditures_post_tax': contract._gas_lbt_expenditures_post_tax,
                'gas_cost_of_sales_expenditures_post_tax': contract._gas_cost_of_sales_expenditures_post_tax,
                'consolidated_capital_expenditures_post_tax': contract._oil_capital_expenditures_post_tax + contract._gas_capital_expenditures_post_tax,
                'consolidated_intangible_expenditures_post_tax': contract._oil_intangible_expenditures_post_tax + contract._gas_intangible_expenditures_post_tax,
                'consolidated_opex_expenditures_post_tax': contract._oil_opex_expenditures_post_tax + contract._gas_opex_expenditures_post_tax,
                'consolidated_asr_expenditures_post_tax': contract._oil_asr_expenditures_post_tax + contract._gas_asr_expenditures_post_tax,
                'consolidated_lbt_expenditures_post_tax': contract._oil_lbt_expenditures_post_tax + contract._gas_lbt_expenditures_post_tax,
                'consolidated_cost_of_sales_expenditures_post_tax': contract._oil_cost_of_sales_expenditures_post_tax + contract._gas_cost_of_sales_expenditures_post_tax,
            }
            for idx, contract in enumerate(self.contract)
        }

    def _get_cost_contract_dict(self):
        return {
            idx: {
                'years': contract.project_years,
                'oil_depreciable': contract._oil_capital_expenditures_post_tax,
                'oil_intangible': contract._oil_intangible_expenditures_post_tax,
                'oil_opex': contract._oil_opex_expenditures_post_tax,
                'oil_asr': contract._oil_asr_expenditures_post_tax,
                'oil_lbt': contract._oil_lbt_expenditures_post_tax,
                'oil_depreciation': contract._oil_depreciation,
                'oil_non_capital': contract._oil_non_capital,
                'oil_sunk_cost': contract._oil_sunk_cost,
                'oil_undepreciated_asset': contract._oil_undepreciated_asset,
                'gas_depreciable': contract._gas_capital_expenditures_post_tax,
                'gas_intangible': contract._gas_intangible_expenditures_post_tax,
                'gas_opex': contract._gas_opex_expenditures_post_tax,
                'gas_asr': contract._gas_asr_expenditures_post_tax,
                'gas_lbt': contract._gas_lbt_expenditures_post_tax,
                'gas_depreciation': contract._gas_depreciation,
                'gas_non_capital': contract._gas_non_capital,
                'gas_sunk_cost': contract._gas_sunk_cost,
                'gas_undepreciated_asset': contract._gas_undepreciated_asset,
                'consolidated_depreciable': contract._oil_capital_expenditures_post_tax + contract._gas_capital_expenditures_post_tax,
                'consolidated_intangible': contract._oil_intangible_expenditures_post_tax + contract._gas_intangible_expenditures_post_tax,
                'consolidated_opex': contract._oil_opex_expenditures_post_tax + contract._gas_opex_expenditures_post_tax,
                'consolidated_asr': contract._oil_asr_expenditures_post_tax + contract._gas_asr_expenditures_post_tax,
                'consolidated_lbt': contract._oil_lbt_expenditures_post_tax + contract._gas_lbt_expenditures_post_tax,
                'consolidated_depreciation': contract._oil_depreciation + contract._gas_depreciation,
                'consolidated_non_capital': contract._oil_non_capital + contract._gas_non_capital,
                'consolidated_sunk_cost': contract._oil_sunk_cost + contract._gas_sunk_cost,
                'consolidated_undepreciated_asset': contract._oil_undepreciated_asset + contract._gas_undepreciated_asset,
            }
            for idx, contract in enumerate(self.contract)
        }

    def _get_psc_terms_dict(self):
        result = {}
        for idx, contract in enumerate(self.contract):
            if isinstance(contract, CostRecovery) or isinstance(contract, GrossSplit):
                result[idx] = {
                    'years': contract.project_years,
                    'oil_costrecovery_or_deductible_cost': contract._oil_cost_recovery_after_tf if isinstance(contract, CostRecovery) else contract._oil_deductible_cost,
                    'oil_unrecoverable_cost_or_carryforward_cost': contract._oil_unrecovered_after_transfer if isinstance(contract, CostRecovery) else contract._oil_carward_cost_aftertf,
                    'oil_ctr_share': contract._oil_contractor_share if isinstance(contract, CostRecovery) else contract._oil_ctr_share_after_transfer,
                    'oil_gov_share': contract._oil_government_share if isinstance(contract, CostRecovery) else contract._oil_gov_share,
                    'oil_effective_tax_payment': contract._oil_tax_payment if isinstance(contract, CostRecovery) else contract._oil_tax,
                    'oil_ddmo': contract._oil_ddmo,
                    'oil_ctr_take': contract._oil_contractor_take if isinstance(contract, CostRecovery) else contract._oil_ctr_net_share,
                    'oil_government_take': contract._oil_government_take,
                    'oil_ctr_cashflow': contract._oil_cashflow if isinstance(contract, CostRecovery) else contract._oil_ctr_cashflow,
                    'gas_costrecovery_or_deductible_cost': contract._gas_cost_recovery_after_tf if isinstance(contract, CostRecovery) else contract._gas_deductible_cost,
                    'gas_unrecoverable_cost_or_carryforward_cost': contract._gas_unrecovered_after_transfer if isinstance(contract, CostRecovery) else contract._gas_carward_cost_aftertf,
                    'gas_ctr_share': contract._gas_contractor_share if isinstance(contract,CostRecovery) else contract._gas_ctr_share_after_transfer,
                    'gas_gov_share': contract._gas_government_share if isinstance(contract,CostRecovery) else contract._gas_gov_share,
                    'gas_effective_tax_payment': contract._gas_tax_payment if isinstance(contract,CostRecovery) else contract._gas_tax,
                    'gas_ddmo': contract._gas_ddmo,
                    'gas_ctr_take': contract._gas_contractor_take if isinstance(contract,CostRecovery) else contract._gas_ctr_net_share,
                    'gas_government_take': contract._gas_government_take,
                    'gas_ctr_cashflow': contract._gas_cashflow if isinstance(contract,CostRecovery) else contract._gas_ctr_cashflow,
                    'consolidated_costrecovery_or_deductible_cost': contract._consolidated_cost_recovery_after_tf if isinstance(contract, CostRecovery) else contract._consolidated_deductible_cost,
                    'consolidated_unrecoverable_cost_or_carryforward_cost': contract._consolidated_unrecovered_after_transfer if isinstance(contract, CostRecovery) else contract._consolidated_carward_cost_aftertf,
                    'consolidated_ctr_share': contract._consolidated_contractor_share if isinstance(contract, CostRecovery) else contract._consolidated_ctr_share_after_transfer,
                    'consolidated_gov_share': contract._consolidated_government_share if isinstance(contract,CostRecovery) else contract._consolidated_gov_share_before_tf,
                    'consolidated_effective_tax_payment': contract._consolidated_tax_payment if isinstance(contract,CostRecovery) else contract._consolidated_tax_payment,
                    'consolidated_ddmo': contract._consolidated_ddmo,
                    'consolidated_ctr_take': contract._consolidated_contractor_take if isinstance(contract,CostRecovery) else contract._consolidated_ctr_net_share,
                    'consolidated_government_take': contract._consolidated_government_take,
                    'consolidated_ctr_cashflow': contract._consolidated_cashflow,
                }

            elif isinstance(contract, Transition):
                result[idx] = {
                    'years': contract.project_years,
                    'oil_costrecovery_or_deductible_cost': contract._oil_deductible_cost,
                    'oil_unrecoverable_cost_or_carryforward_cost': contract._oil_unrec_cost,
                    'oil_ctr_share': contract._oil_ctr_ets,
                    'oil_gov_share': contract._oil_gov_ets,
                    'oil_effective_tax_payment': contract._oil_effective_tax_payment,
                    'oil_ddmo': contract._oil_ddmo,
                    'oil_ctr_take': contract._oil_ctr_take,
                    'oil_government_take': contract._oil_government_take,
                    'oil_ctr_cashflow': contract._oil_cashflow,
                    'gas_costrecovery_or_deductible_cost': contract._gas_deductible_cost,
                    'gas_unrecoverable_cost_or_carryforward_cost': contract._gas_unrec_cost,
                    'gas_ctr_share': contract._gas_ctr_ets,
                    'gas_gov_share': contract._gas_gov_ets,
                    'gas_effective_tax_payment': contract._gas_effective_tax_payment,
                    'gas_ddmo': contract._gas_ddmo,
                    'gas_ctr_take': contract._gas_ctr_take,
                    'gas_government_take': contract._gas_government_take,
                    'gas_ctr_cashflow': contract._gas_cashflow,
                    'consolidated_costrecovery_or_deductible_cost': contract._oil_deductible_cost + contract._gas_deductible_cost,
                    'consolidated_unrecoverable_cost_or_carryforward_cost': contract._oil_unrec_cost + contract._gas_unrec_cost,
                    'consolidated_ctr_share': contract._oil_ctr_ets + contract._gas_ctr_ets,
                    'consolidated_gov_share': contract._oil_gov_ets + contract._gas_gov_ets,
                    'consolidated_effective_tax_payment': contract._oil_effective_tax_payment + contract._gas_effective_tax_payment,
                    'consolidated_ddmo': contract._oil_ddmo + contract._gas_ddmo,
                    'consolidated_ctr_take': contract._oil_ctr_take + contract._gas_ctr_take,
                    'consolidated_government_take': contract._oil_government_take + contract._gas_government_take,
                    'consolidated_ctr_cashflow': contract._oil_cashflow + contract._gas_cashflow,
                }
            else:
                pass

        return result

    def _get_cr_terms_dict(self):
        return {
            idx: {
                'years': contract.project_years,
                'oil_ftp_ctr': np.zeros_like(contract.project_years) if isinstance(contract, GrossSplit) else contract._oil_ftp_ctr,
                'oil_ftp_gov': np.zeros_like(contract.project_years) if isinstance(contract, GrossSplit) else contract._oil_ftp_gov,
                'gas_ftp_ctr': np.zeros_like(contract.project_years) if isinstance(contract, GrossSplit) else contract._gas_ftp_ctr,
                'gas_ftp_gov': np.zeros_like(contract.project_years) if isinstance(contract, GrossSplit) else contract._gas_ftp_gov,
                'consolidated_ctr_ftp': np.zeros_like(contract.project_years) if isinstance(contract, GrossSplit) else (contract._oil_ftp_ctr + contract._gas_ftp_ctr),
                'consolidated_gov_ftp': np.zeros_like(contract.project_years) if isinstance(contract, GrossSplit) else (contract._oil_ftp_gov + contract._gas_ftp_gov),
            }
            for idx, contract in enumerate(self.contract)
        }

    @staticmethod
    def _map_to_ref_years(ref_years: np.array, project_years: np.array, values: np.array):
        # Get the indices where ref_years would be inserted into project_years
        indices = np.searchsorted(project_years, ref_years, side='right') - 1

        # Ensure the indices are valid (i.e., within bounds)
        indices = np.clip(indices, 0, len(project_years) - 1)

        return values[indices]


    def _parse_dict(self, source):
        for idx, cntrct in source.items():
            for key in cntrct:
                if key == 'years':
                    pass
                elif key == 'oil_undepreciated_asset' or key == 'gas_undepreciated_asset' or key == 'consolidated_undepreciated_asset':
                    undepreciated = np.zeros_like(cntrct['years'])
                    undepreciated[-1] = cntrct[key]
                    cntrct[key] = undepreciated
                else:
                    cntrct[key] = self._map_to_ref_years(
                        ref_years=self.years,
                        project_years=cntrct['years'],
                        values=cntrct[key]
                    )
            cntrct['years']= self.years

        return source

    @staticmethod
    def _merge_dicts(data: dict, mode: str):
        # Convert to list of dictionaries
        dicts = tuple(data.values())

        # Copy first dictionary
        merged = {key: value.copy() for key, value in dicts[0].items()}

        for other in dicts[1:]:
            for key, value in other.items():
                if key != "years":
                    # Element-wise addition
                    if mode == 'combine':
                        merged[key] += value
                    elif mode == 'incremental':
                        merged[key] -= value
                    else:
                        pass

        return merged

    def _parsing_attributes(
            self,
            lifting: dict,
            revenue: dict,
            cost_expenditure_pretax: dict,
            cost_indirect_tax: dict,
            cost_expenditure_posttax: dict,
            cost_contract: dict,
            psc_terms: dict,
            cr_terms: dict,
    ):
        # Lifting Attributes
        self.oil_lifting = lifting['oil_lifting']
        self.gas_lifting = lifting['gas_lifting']
        self.sulfur_lifting = lifting['sulfur_lifting']
        self.electricity_lifting = lifting['electricity_lifting']
        self.co2_lifting = lifting['co2_lifting']

        # Revenue
        self.oil_revenue = revenue['oil_revenue']
        self.gas_revenue = revenue['gas_revenue']
        self.sulfur_revenue = revenue['sulfur_revenue']
        self.electricity_revenue = revenue['electricity_revenue']
        self.co2_revenue = revenue['co2_revenue']

        # Cost Expenditures Pre Tax
        self.oil_capital_expenditures_pre_tax = cost_expenditure_pretax['oil_capital_expenditures_pre_tax']
        self.oil_intangible_expenditures_pre_tax = cost_expenditure_pretax['oil_intangible_expenditures_pre_tax']
        self.oil_opex_expenditures_pre_tax = cost_expenditure_pretax['oil_opex_expenditures_pre_tax']
        self.oil_asr_expenditures_pre_tax = cost_expenditure_pretax['oil_asr_expenditures_pre_tax']
        self.oil_lbt_expenditures_pre_tax = cost_expenditure_pretax['oil_lbt_expenditures_pre_tax']
        self.oil_cost_of_sales_expenditures_pre_tax = cost_expenditure_pretax['oil_cost_of_sales_expenditures_pre_tax']
        self.gas_capital_expenditures_pre_tax = cost_expenditure_pretax['gas_capital_expenditures_pre_tax']
        self.gas_intangible_expenditures_pre_tax = cost_expenditure_pretax['gas_intangible_expenditures_pre_tax']
        self.gas_opex_expenditures_pre_tax = cost_expenditure_pretax['gas_opex_expenditures_pre_tax']
        self.gas_asr_expenditures_pre_tax = cost_expenditure_pretax['gas_asr_expenditures_pre_tax']
        self.gas_lbt_expenditures_pre_tax = cost_expenditure_pretax['gas_lbt_expenditures_pre_tax']
        self.gas_cost_of_sales_expenditures_pre_tax = cost_expenditure_pretax['gas_cost_of_sales_expenditures_pre_tax']
        self.consolidated_capital_expenditures_pre_tax = cost_expenditure_pretax['consolidated_capital_expenditures_pre_tax']
        self.consolidated_intangible_expenditures_pre_tax = cost_expenditure_pretax['consolidated_intangible_expenditures_pre_tax']
        self.consolidated_opex_expenditures_pre_tax = cost_expenditure_pretax['consolidated_opex_expenditures_pre_tax']
        self.consolidated_asr_expenditures_pre_tax = cost_expenditure_pretax['consolidated_asr_expenditures_pre_tax']
        self.consolidated_lbt_expenditures_pre_tax = cost_expenditure_pretax['consolidated_lbt_expenditures_pre_tax']
        self.consolidated_cost_of_sales_expenditures_pre_tax = cost_expenditure_pretax['consolidated_cost_of_sales_expenditures_pre_tax']

        # Cost Indirect Tax
        self.oil_capital_indirect_tax = cost_indirect_tax['oil_capital_indirect_tax']
        self.oil_intangible_indirect_tax = cost_indirect_tax['oil_intangible_indirect_tax']
        self.oil_opex_indirect_tax = cost_indirect_tax['oil_opex_indirect_tax']
        self.oil_asr_indirect_tax = cost_indirect_tax['oil_asr_indirect_tax']
        self.oil_lbt_indirect_tax = cost_indirect_tax['oil_lbt_indirect_tax']
        self.oil_cost_of_sales_indirect_tax = cost_indirect_tax['oil_cost_of_sales_indirect_tax']
        self.gas_capital_indirect_tax = cost_indirect_tax['gas_capital_indirect_tax']
        self.gas_intangible_indirect_tax = cost_indirect_tax['gas_intangible_indirect_tax']
        self.gas_opex_indirect_tax = cost_indirect_tax['gas_opex_indirect_tax']
        self.gas_asr_indirect_tax = cost_indirect_tax['gas_asr_indirect_tax']
        self.gas_lbt_indirect_tax = cost_indirect_tax['gas_lbt_indirect_tax']
        self.gas_cost_of_sales_indirect_tax = cost_indirect_tax['gas_cost_of_sales_indirect_tax']
        self.consolidated_capital_indirect_tax = cost_indirect_tax['consolidated_capital_indirect_tax']
        self.consolidated_intangible_indirect_tax = cost_indirect_tax['consolidated_intangible_indirect_tax']
        self.consolidated_opex_indirect_tax = cost_indirect_tax['consolidated_opex_indirect_tax']
        self.consolidated_asr_indirect_tax = cost_indirect_tax['consolidated_asr_indirect_tax']
        self.consolidated_lbt_indirect_tax = cost_indirect_tax['consolidated_lbt_indirect_tax']
        self.consolidated_cost_of_sales_indirect_tax = cost_indirect_tax['consolidated_cost_of_sales_indirect_tax']

        # Cost Post Tax
        self.oil_capital_expenditures_post_tax = cost_expenditure_posttax['oil_capital_expenditures_post_tax']
        self.oil_intangible_expenditures_post_tax = cost_expenditure_posttax['oil_intangible_expenditures_post_tax']
        self.oil_opex_expenditures_post_tax = cost_expenditure_posttax['oil_opex_expenditures_post_tax']
        self.oil_asr_expenditures_post_tax = cost_expenditure_posttax['oil_asr_expenditures_post_tax']
        self.oil_lbt_expenditures_post_tax = cost_expenditure_posttax['oil_lbt_expenditures_post_tax']
        self.oil_cost_of_sales_expenditures_post_tax = cost_expenditure_posttax['oil_cost_of_sales_expenditures_post_tax']
        self.gas_capital_expenditures_post_tax = cost_expenditure_posttax['gas_capital_expenditures_post_tax']
        self.gas_intangible_expenditures_post_tax = cost_expenditure_posttax['gas_intangible_expenditures_post_tax']
        self.gas_opex_expenditures_post_tax = cost_expenditure_posttax['gas_opex_expenditures_post_tax']
        self.gas_asr_expenditures_post_tax = cost_expenditure_posttax['gas_asr_expenditures_post_tax']
        self.gas_lbt_expenditures_post_tax = cost_expenditure_posttax['gas_lbt_expenditures_post_tax']
        self.gas_cost_of_sales_expenditures_post_tax = cost_expenditure_posttax['gas_cost_of_sales_expenditures_post_tax']
        self.consolidated_capital_expenditures_post_tax = cost_expenditure_posttax['consolidated_capital_expenditures_post_tax']
        self.consolidated_intangible_expenditures_post_tax = cost_expenditure_posttax['consolidated_intangible_expenditures_post_tax']
        self.consolidated_opex_expenditures_post_tax = cost_expenditure_posttax['consolidated_opex_expenditures_post_tax']
        self.consolidated_asr_expenditures_post_tax = cost_expenditure_posttax['consolidated_asr_expenditures_post_tax']
        self.consolidated_lbt_expenditures_post_tax = cost_expenditure_posttax['consolidated_lbt_expenditures_post_tax']
        self.consolidated_cost_of_sales_expenditures_post_tax = cost_expenditure_posttax['consolidated_cost_of_sales_expenditures_post_tax']

        # Contract Terms Cost
        self.oil_depreciable = cost_contract['oil_depreciable']
        self.oil_intangible = cost_contract['oil_intangible']
        self.oil_opex = cost_contract['oil_opex']
        self.oil_asr = cost_contract['oil_asr']
        self.oil_lbt = cost_contract['oil_lbt']
        self.oil_depreciation = cost_contract['oil_depreciation']
        self.oil_non_capital = cost_contract['oil_non_capital']
        self.oil_sunk_cost = cost_contract['oil_sunk_cost']
        self.oil_undepreciated_asset = cost_contract['oil_undepreciated_asset']
        self.gas_depreciable = cost_contract['gas_depreciable']
        self.gas_intangible = cost_contract['gas_intangible']
        self.gas_opex = cost_contract['gas_opex']
        self.gas_asr = cost_contract['gas_asr']
        self.gas_lbt = cost_contract['gas_lbt']
        self.gas_depreciation = cost_contract['gas_depreciation']
        self.gas_non_capital = cost_contract['gas_non_capital']
        self.gas_sunk_cost = cost_contract['gas_sunk_cost']
        self.gas_undepreciated_asset = cost_contract['gas_undepreciated_asset']
        self.consolidated_depreciable = cost_contract['consolidated_depreciable']
        self.consolidated_intangible = cost_contract['consolidated_intangible']
        self.consolidated_opex = cost_contract['consolidated_opex']
        self.consolidated_asr = cost_contract['consolidated_asr']
        self.consolidated_lbt = cost_contract['consolidated_lbt']
        self.consolidated_depreciation = cost_contract['consolidated_depreciation']
        self.consolidated_non_capital = cost_contract['consolidated_non_capital']
        self.consolidated_sunk_cost = cost_contract['consolidated_sunk_cost']
        self.consolidated_undepreciated_asset = cost_contract['consolidated_undepreciated_asset']

        # PSC terms
        self.oil_costrecovery_or_deductible_cost = psc_terms['oil_costrecovery_or_deductible_cost']
        self.oil_unrecoverable_cost_or_carryforward_cost = psc_terms['oil_unrecoverable_cost_or_carryforward_cost']
        self.oil_ctr_share = psc_terms['oil_ctr_share']
        self.oil_gov_share = psc_terms['oil_gov_share']
        self.oil_effective_tax_payment = psc_terms['oil_effective_tax_payment']
        self.oil_ddmo = psc_terms['oil_ddmo']
        self.oil_ctr_take = psc_terms['oil_ctr_take']
        self.oil_government_take = psc_terms['oil_government_take']
        self.oil_ctr_cashflow = psc_terms['oil_ctr_cashflow']
        self.gas_costrecovery_or_deductible_cost = psc_terms['gas_costrecovery_or_deductible_cost']
        self.gas_unrecoverable_cost_or_carryforward_cost = psc_terms['gas_unrecoverable_cost_or_carryforward_cost']
        self.gas_ctr_share = psc_terms['gas_ctr_share']
        self.gas_gov_share = psc_terms['gas_gov_share']
        self.gas_effective_tax_payment = psc_terms['gas_effective_tax_payment']
        self.gas_ddmo = psc_terms['gas_ddmo']
        self.gas_ctr_take = psc_terms['gas_ctr_take']
        self.gas_government_take = psc_terms['gas_government_take']
        self.gas_ctr_cashflow = psc_terms['gas_ctr_cashflow']
        self.consolidated_costrecovery_or_deductible_cost = psc_terms['consolidated_costrecovery_or_deductible_cost']
        self.consolidated_unrecoverable_cost_or_carryforward_cost = psc_terms['consolidated_unrecoverable_cost_or_carryforward_cost']
        self.consolidated_ctr_share = psc_terms['consolidated_ctr_share']
        self.consolidated_gov_share = psc_terms['consolidated_gov_share']
        self.consolidated_effective_tax_payment = psc_terms['consolidated_effective_tax_payment']
        self.consolidated_ddmo = psc_terms['consolidated_ddmo']
        self.consolidated_ctr_take = psc_terms['consolidated_ctr_take']
        self.consolidated_government_take = psc_terms['consolidated_government_take']
        self.consolidated_ctr_cashflow = psc_terms['consolidated_ctr_cashflow']

        # Cost Recovery Terms
        self.oil_ftp_ctr = cr_terms['oil_ftp_ctr']
        self.gas_ftp_ctr = cr_terms['gas_ftp_ctr']
        self.oil_ftp_gov = cr_terms['oil_ftp_gov']
        self.gas_ftp_gov = cr_terms['gas_ftp_gov']
        self.consolidated_ctr_ftp = cr_terms['consolidated_ctr_ftp']
        self.consolidated_gov_ftp = cr_terms['consolidated_gov_ftp']

    def run(self, mode: str):
        # Get the dictionary of each element
        lifting = self._parse_dict(source=self._get_lifting_dict())
        revenue = self._parse_dict(source=self._get_revenue_dict())
        cost_expenditure_pretax = self._parse_dict(source=self._get_cost_expenditure_pretax_dict())
        cost_indirect_tax = self._parse_dict(source=self._get_cost_indirect_tax_dict())
        cost_expenditure_posttax = self._parse_dict(source=self._get_cost_expenditure_posttax_dict())
        cost_contract = self._parse_dict(source=self._get_cost_contract_dict())
        psc_terms = self._parse_dict(source=self._get_psc_terms_dict())
        cr_terms = self._parse_dict(source=self._get_cr_terms_dict())

        # Calculate based on the chosen method, combine or incremental
        lifting_merged = self._merge_dicts(data=lifting, mode=mode)
        revenue_merged = self._merge_dicts(data=revenue, mode=mode)
        cost_expenditure_pretax_merged = self._merge_dicts(data=cost_expenditure_pretax, mode=mode)
        cost_indirect_tax_merged = self._merge_dicts(data=cost_indirect_tax, mode=mode)
        cost_expenditure_posttax_merged = self._merge_dicts(data=cost_expenditure_posttax, mode=mode)
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

    def _to_dataframe(self):
        import pandas as pd
        """Convert selected dataclass attributes into a pandas DataFrame."""
        exclude: list = ['contract', 'reference_year', 'inflation_rate', 'discount_rate', 'npv_mode',
                         'discounting_mode', 'profitability_discounted', ]
        data_dict = {key: value for key, value in self.__dict__.items() if key not in exclude}
        return pd.DataFrame(data_dict)  # Convert filtered dictionary to DataFrame

    def case_combine(self):
        self.run(mode='combine')
        return self._to_dataframe()

    def case_incremental(self):
        self.run(mode='incremental')
        return self._to_dataframe()

    def _calc_npv(self):
        # NPV Calculation for SKK Real Terms
        if self.npv_mode == NPVSelection.NPV_SKK_REAL_TERMS:
            # Contractor NPV
            self.ctr_npv = npv_skk_real_terms(
                cashflow=self.consolidated_ctr_cashflow,
                cashflow_years=self.years,
                discount_rate=self.discount_rate,
                reference_year=self.reference_year,
                discounting_mode=self.discounting_mode
            )

            # Contractor Investment NPV
            self.investment_npv = npv_skk_real_terms(
                cashflow=(
                        self.consolidated_capital_expenditures_post_tax +
                        self.consolidated_intangible_expenditures_post_tax
                ),
                cashflow_years=self.years,
                discount_rate=self.discount_rate,
                reference_year=self.reference_year,
                discounting_mode=self.discounting_mode
            )

            # Government Take Net Present Value
            self.gov_take_npv = npv_skk_real_terms(
                cashflow=self.consolidated_government_take,
                cashflow_years=self.years,
                discount_rate=self.discount_rate,
                reference_year=self.reference_year,
                discounting_mode=self.discounting_mode
            )

        # NPV Calculation for SKK Nominal Terms
        elif self.npv_mode == NPVSelection.NPV_SKK_NOMINAL_TERMS:
            # Contractor NPV
            self.ctr_npv = npv_skk_nominal_terms(
                cashflow=self.consolidated_ctr_cashflow,
                cashflow_years=self.years,
                discount_rate=self.discount_rate,
                discounting_mode=self.discounting_mode
            )

            # Contractor Investment NPV
            self.investment_npv = npv_skk_nominal_terms(
                cashflow=(
                        self.consolidated_capital_expenditures_post_tax +
                        self.consolidated_intangible_expenditures_post_tax
                ),
                cashflow_years=self.years,
                discount_rate=self.discount_rate,
                discounting_mode=self.discounting_mode
            )

            # Government Take Net Present Value
            self.gov_take_npv = npv_skk_nominal_terms(
                cashflow=self.consolidated_government_take,
                cashflow_years=self.years,
                discount_rate=self.discount_rate,
                discounting_mode=self.discounting_mode
            )

        # NPV Calculation for Nominal Terms
        elif self.npv_mode == NPVSelection.NPV_NOMINAL_TERMS:
            # Contractor NPV
            self.ctr_npv = npv_nominal_terms(
                cashflow=self.consolidated_ctr_cashflow,
                cashflow_years=self.years,
                discount_rate=self.discount_rate,
                reference_year=self.reference_year,
                discounting_mode=self.discounting_mode
            )

            # Contractor Investment NPV
            self.investment_npv = npv_nominal_terms(
                cashflow=(
                    self.consolidated_capital_expenditures_post_tax +
                    self.consolidated_intangible_expenditures_post_tax
            ),
                cashflow_years=self.years,
                discount_rate=self.discount_rate,
                reference_year=self.reference_year,
                discounting_mode=self.discounting_mode
            )

            # Government Take Net Present Value
            self.gov_take_npv = npv_nominal_terms(
                cashflow=self.consolidated_government_take,
                cashflow_years=self.years,
                discount_rate=self.discount_rate,
                reference_year=self.reference_year,
                discounting_mode=self.discounting_mode
            )

        # NPV Calculation for Real Terms
        elif self.npv_mode == NPVSelection.NPV_REAL_TERMS:
            # Contractor NPV
            self.ctr_npv = npv_real_terms(
                cashflow=self.consolidated_ctr_cashflow,
                cashflow_years=self.years,
                discount_rate=self.discount_rate,
                reference_year=self.reference_year,
                inflation_rate=self.inflation_rate,
                discounting_mode=self.discounting_mode
            )

            # Contractor Investment NPV
            self.investment_npv = npv_real_terms(
                cashflow=(
                        self.consolidated_capital_expenditures_post_tax +
                        self.consolidated_intangible_expenditures_post_tax
                ),
                cashflow_years=self.years,
                discount_rate=self.discount_rate,
                reference_year=self.reference_year,
                inflation_rate=self.inflation_rate,
                discounting_mode=self.discounting_mode
            )

            # Government Take Net Present Value
            self.gov_take_npv = npv_real_terms(
                cashflow=self.consolidated_government_take,
                cashflow_years=self.years,
                discount_rate=self.discount_rate,
                reference_year=self.reference_year,
                inflation_rate=self.inflation_rate,
                discounting_mode=self.discounting_mode
            )

        # NPV Calculation for Point Forwards
        else:
            # Contractor NPV
            self.ctr_npv = npv_point_forward(
                cashflow=self.consolidated_ctr_cashflow,
                cashflow_years=self.years,
                discount_rate=self.discount_rate,
                reference_year=self.reference_year,
                discounting_mode=self.discounting_mode
            )

            # Contractor Investment NPV
            self.investment_npv = npv_point_forward(
                cashflow=(
                    self.consolidated_capital_expenditures_post_tax +
                    self.consolidated_intangible_expenditures_post_tax
            ),
                cashflow_years=self.years,
                discount_rate=self.discount_rate,
                reference_year=self.reference_year,
                discounting_mode=self.discounting_mode
            )

            # Government Take Net Present Value
            self.gov_take_npv = npv_point_forward(
                cashflow=self.consolidated_government_take,
                cashflow_years=self.years,
                discount_rate=self.discount_rate,
                reference_year=self.reference_year,
                discounting_mode=self.discounting_mode
            )

    def _calc_pot(self):
        # Contractor POT
        self.pot = pot_psc(
            cashflow=self.consolidated_ctr_cashflow,
            cashflow_years=self.years,
            reference_year=self.reference_year
        )

    def _calc_pv_ratio(self):
        if self.profitability_discounted:
            self.pv_ratio = np.divide(self.ctr_npv, self.investment_npv, where=self.investment_npv != 0)

        # Condition when the Profitability Index is calculated using un-discounted investment
        else:
            investment = np.sum(
                self.consolidated_capital_expenditures_post_tax +
                self.consolidated_intangible_expenditures_post_tax +
                self.consolidated_sunk_cost
            )
            self.pv_ratio = np.divide(self.ctr_npv, investment, where=investment != 0)

        self.ctr_pi = 1 + self.pv_ratio

    def _calc_irr(self):
        self.irr = irr(cashflow=self.consolidated_ctr_cashflow)

    def get_executive_summary(self):
        # Lifting Group
        lifting_oil = np.sum(self.oil_lifting)
        lifting_gas = np.sum(self.gas_lifting)

        # Gross Revenue Group
        gross_revenue_oil = np.sum(self.oil_revenue)
        gross_revenue_gas = np.sum(self.gas_revenue)
        gross_revenue = gross_revenue_oil + gross_revenue_gas

        # WAP Group
        oil_wap = gross_revenue_oil / lifting_oil
        gas_wap = gross_revenue_gas / lifting_gas

        # Contractor and Government Gross share
        ctr_gross_share = np.sum(self.consolidated_ctr_share)
        gov_gross_share = np.sum(self.consolidated_gov_share)

        # Investment Group
        sunk_cost = np.sum(self.consolidated_sunk_cost)
        tangible = np.sum(self.consolidated_capital_expenditures_post_tax)
        intangible = np.sum(self.consolidated_intangible_expenditures_post_tax)
        investment = tangible + intangible

        # OPEX and ASR Group
        opex = np.sum(self.consolidated_opex_expenditures_post_tax)
        asr = np.sum(self.consolidated_asr_expenditures_post_tax)
        lbt = np.sum(self.consolidated_lbt_expenditures_post_tax)
        opex_asr_lbt = opex + asr + lbt

        # Cost recovery or Deductible Cost
        costrecovery_or_deductible_cost = np.sum(self.consolidated_costrecovery_or_deductible_cost)
        cost_recovery_over_gross_rev = costrecovery_or_deductible_cost / gross_revenue

        # Unrecoverable Cost
        unrec_cost = np.sum(self.consolidated_unrecoverable_cost_or_carryforward_cost)
        unrec_over_gross_rev = unrec_cost/ gross_revenue

        # Contractor Net Share
        ctr_share = np.sum(self.consolidated_ctr_share)
        ctr_net_share_over_gross_share = ctr_share / gross_revenue

        # Contractor Net Cashflow
        ctr_net_cashflow = np.sum(self.consolidated_ctr_cashflow)
        ctr_net_cashflow_over_gross_rev = ctr_net_cashflow / gross_revenue

        # Contractor NPV
        self._calc_npv()
        ctr_npv = self.ctr_npv

        # Investment NPV
        investment_npv = self.investment_npv

        # Government Take NPV
        gov_take_npv = self.gov_take_npv

        # POT
        self._calc_pot()
        pot = self.pot

        # PV_Ratio
        self._calc_pv_ratio()
        pv_ratio = self.pv_ratio
        pi = self.pi

        # Contractor IRR
        self._calc_irr()
        ctr_irr = self.irr

        # Government Indicator
        gov_ftp_share = self.consolidated_gov_ftp
        gov_equity_share = self.consolidated_gov_share
        gov_ddmo = self.consolidated_ddmo
        gov_tax_income = self.consolidated_effective_tax_payment
        gov_take = self.consolidated_government_take
        gov_take_over_gross_rev = gov_take / gross_revenue
        oil_indirect_taxes = (
                self.oil_capital_indirect_tax +
                self.oil_intangible_indirect_tax +
                self.oil_opex_indirect_tax +
                self.oil_asr_indirect_tax +
                self.oil_lbt_indirect_tax +
                self.oil_cost_of_sales_indirect_tax
        )
        gas_indirect_taxes = (
                self.gas_capital_indirect_tax +
                self.gas_intangible_indirect_tax +
                self.gas_opex_indirect_tax +
                self.gas_asr_indirect_tax +
                self.gas_lbt_indirect_tax +
                self.gas_cost_of_sales_indirect_tax
        )
        total_indirect_taxes = oil_indirect_taxes + gas_indirect_taxes

        # Undepreciated Asset
        undepreciated_asset_oil = self.oil_undepreciated_asset
        undepreciated_asset_gas = self.gas_undepreciated_asset
        undepreciated_asset_total = undepreciated_asset_oil + undepreciated_asset_gas

        return {
            'lifting_oil':lifting_oil,
            'oil_wap':oil_wap,
            'lifting_gas':lifting_gas,
            'gas_wap':gas_wap,
            'gross_revenue':gross_revenue,
            'ctr_gross_share':ctr_gross_share,
            'sunk_cost':sunk_cost,
            'investment':investment,
            'tangible':tangible,
            'intangible':intangible,
            'opex_asr_lbt':opex_asr_lbt,
            'opex':opex,
            'asr':asr,
            'cost_recovery/deductible_cost':costrecovery_or_deductible_cost,
            'cost_recovery_over_gross_rev':cost_recovery_over_gross_rev,
            'unrec_cost':unrec_cost,
            'unrec_over_gross_rev':unrec_over_gross_rev,
            'ctr_net_share':ctr_share,
            'ctr_net_share_over_gross_share':ctr_net_share_over_gross_share,
            'ctr_net_cashflow':ctr_net_cashflow,
            'ctr_net_cashflow_over_gross_rev':ctr_net_cashflow_over_gross_rev,
            'ctr_npv':ctr_npv,
            'ctr_irr':ctr_irr,
            'ctr_pot':pot,
            'ctr_pv_ratio':pv_ratio,
            'ctr_pi':pi,
            'gov_gross_share':gov_gross_share,
            'gov_ftp_share':gov_ftp_share,
            'gov_ddmo':gov_ddmo,
            'gov_tax_income':gov_tax_income,
            'gov_take':gov_take,
            'gov_take_over_gross_rev':gov_take_over_gross_rev,
            'gov_take_npv':gov_take_npv,
            'undepreciated_asset_oil':undepreciated_asset_oil,
            'undepreciated_asset_gas':undepreciated_asset_gas,
            'undepreciated_asset_total':undepreciated_asset_total,
            'total_indirect_taxes':total_indirect_taxes,
            'oil_indirect_taxes':oil_indirect_taxes,
            'gas_indirect_taxes':gas_indirect_taxes,
        }

