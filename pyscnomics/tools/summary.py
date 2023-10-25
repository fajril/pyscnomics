from dataclasses import dataclass, field, is_dataclass

from pyscnomics.econ.selection import ContractType
import numpy as np


@dataclass
class Summary:
    lifting_oil: float = field(default=0.0)
    oil_wap: float = field(default=0.0)
    lifting_gas: float = field(default=0.0)
    gas_wap: float = field(default=0.0)
    lifting_sulfur: float = field(default=0.0)
    sulfur_wap: float = field(default=0.0)
    sales_electricity: float = field(default=0.0)
    electricity_wap: float = field(default=0.0)
    lifting_co2: float = field(default=0.0)
    co2_wap: float = field(default=0.0)
    gross_revenue: float = field(default=0.0)
    ctr_gross_share: float = field(default=0.0)
    gov_gross_share: float = field(default=0.0)
    sunk_cost: float = field(default=0.0)
    investment: float = field(default=0.0)
    tangible: float = field(default=0.0)
    intangible: float = field(default=0.0)
    opex_and_asr: float = field(default=0.0)
    opex: float = field(default=0.0)
    asr: float = field(default=0.0)
    cost_recovery_or_deductible_cost: float = field(default=0.0)
    gross_rev_cost_recovery_or_deductible_cost: float = field(default=0.0)
    unrec_or_carry_forward_deductible_cost: float = field(default=0.0)
    unrec_over_cost_recovery: float = field(default=0.0)
    ctr_net_share: float = field(default=0.0)
    ctr_net_share_over_gross_share: float = field(default=0.0)
    ctr_net_cashflow: float = field(default=0.0)
    gross_rev_net_cashflow: float = field(default=0.0)
    ctr_npv: float = field(default=0.0)
    ctr_irr: float = field(default=0.0)
    ctr_pot: float = field(default=0.0)
    ctr_pi: float = field(default=0.0)
    gov_profitability: float = field(default=0.0)
    net_dmo: float = field(default=0.0)
    tax: float = field(default=0.0)
    goi_take: float = field(default=0.0)
    gross_rev_goi_take: float = field(default=0.0)
    goi_npv: float = field(default=0.0)

