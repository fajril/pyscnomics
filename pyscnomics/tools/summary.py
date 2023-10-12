from dataclasses import dataclass, field, is_dataclass
from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit

import numpy as np


@dataclass
class Summary:
    contract: BaseProject | CostRecovery | GrossSplit

    # Attributes to be defined later
    oil_production: float = field(default=None)
    oil_wap: float = field(default=None)
    gross_revenue: float = field(default=None)
    gross_share: float = field(default=None)
    ctr_gross_share: float = field(default=None)
    investment: float = field(default=None)
    tangible: float = field(default=None)
    intangible: float = field(default=None)
    opex_and_asr: float = field(default=None)
    opex: float = field(default=None)
    asr: float = field(default=None)
    deductible_cost: float = field(default=None)
    gross_rev_deductible_cost: float = field(default=None)
    carry_forward_deductible_cost: float = field(default=None)
    gross_rev_carry_forward_cost: float = field(default=None)
    ctr_profitability: float = field(default=None)
    ctr_net_share: float = field(default=None)
    gross_rev_net_share: float = field(default=None)
    ctr_net_cashflow: float = field(default=None)
    gross_rev_net_cashflow: float = field(default=None)
    ctr_npv_onward: float = field(default=None)
    ctr_irr_onward: float = field(default=None)
    ctr_pot: float = field(default=None)
    ctr_pi: float = field(default=None)
    gov_gross_share: float = field(default=None)
    net_dmo: float = field(default=None)
    tax: float = field(default=None)
    goi_take: float = field(default=None)
    gross_rev_goi_take: float = field(default=None)
    goi_npv_onward: float = field(default=None)

    def __post_init__(self):
        if is_dataclass(self.contract) and isinstance(self.contract, GrossSplit):
            self.oil_production = float(np.sum(self.contract._oil_lifting.lifting_rate))
            self.oil_wap = float(np.sum(self.contract._oil_revenue))/self.oil_production
            self.gross_revenue = float(np.sum(self.contract._))
            self.gross_share = float()
            self.ctr_gross_share = float()
            self.investment = float()
            self.tangible = float()
            self.intangible = float()
            self.opex_and_asr = float()
            self.opex = float()
            self.asr = float()
            self.deductible_cost = float()
            self.gross_rev_deductible_cost = float()
            self.carry_forward_deductible_cost = float()
            self.gross_rev_carry_forward_cost = float()
            self.ctr_profitability = float()
            self.ctr_net_share = float()
            self.gross_rev_net_share = float()
            self.ctr_net_cashflow = float()
            self.gross_rev_net_cashflow = float()
            self.ctr_npv_onward = float()
            self.ctr_irr_onward = float()
            self.ctr_pot = float()
            self.ctr_pi = float()
            self.gov_gross_share = float()
            self.net_dmo = float()
            self.tax = float()
            self.goi_take = float()
            self.gross_rev_goi_take = float()
            self.goi_npv_onward = float()
