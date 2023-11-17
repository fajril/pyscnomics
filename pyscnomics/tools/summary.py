from dataclasses import dataclass, field

import numpy as np

from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts.transition import Transition
from pyscnomics.econ.selection import NPVSelection
from pyscnomics.econ.indicator import npv_skk_nominal_terms, npv_skk_real_terms, irr, pot


class NPVMethodException(Exception):
    """Please Select the NPV method based on the available selection"""

    pass


@dataclass
class Summary:
    contract: CostRecovery | GrossSplit | Transition

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
    cost_recovery_or_deductible_cost_over_gross_rev: float = field(default=0.0)
    unrec_or_carry_forward_deductible_cost: float = field(default=0.0)
    unrec_over_gross_rev: float = field(default=0.0)
    ctr_net_share: float = field(default=0.0)
    ctr_net_share_over_gross_share: float = field(default=0.0)
    ctr_net_cashflow: float = field(default=0.0)
    ctr_net_cashflow_over_gross_rev: float = field(default=0.0)
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


def get_summary(contract: CostRecovery | GrossSplit | Transition,
                reference_year: int,
                npv_mode: NPVSelection = NPVSelection.FULL_CYCLE_REAL_TERMS,
                discount_rate: float = 0.1) -> dict:

    # Defining the same summary parameters for any contract
    # Lifting
    lifting_oil = np.sum(contract._oil_lifting.get_lifting_rate_arr(), dtype=float)
    oil_wap = np.average(contract._oil_wap_price[np.where(contract._oil_wap_price > 0)])
    lifting_gas = np.sum(contract._gas_lifting.get_lifting_rate_arr(), dtype=float)
    gas_wap = np.average(contract._gas_wap_price[np.where(contract._gas_wap_price > 0)])
    gross_revenue = np.sum(contract._consolidated_revenue, dtype=float)
    gross_revenue_oil = np.sum(contract._oil_revenue, dtype=float)
    gross_revenue_gas = np.sum(contract._gas_revenue, dtype=float)

    # Sunk Cost
    sunk_cost = np.sum(contract._oil_sunk_cost + contract._gas_sunk_cost, dtype=float)

    # Investment (Capital Cost)
    tangible = np.sum(contract._oil_tangible.expenditures() + contract._gas_tangible.expenditures(), dtype=float)
    intangible = np.sum(contract._oil_intangible.expenditures() + contract._gas_intangible.expenditures(),
                        dtype=float)
    investment = tangible + intangible - sunk_cost

    # OPEX and ASR (Non-Capital Cost)
    opex = np.sum(contract._oil_opex.expenditures() + contract._gas_opex.expenditures(), dtype=float)
    asr = np.sum(contract._oil_asr.expenditures() + contract._gas_asr.expenditures(), dtype=float)

    # Cashflow sunk cost
    if sunk_cost == 0:
        cashflow_sunk_cost_pooled = contract._consolidated_cashflow
    else:
        cashflow_sunk_cost_pooled = np.concatenate(
            (np.array([-sunk_cost]),
             contract._consolidated_cashflow[len(contract._consolidated_sunk_cost):]))

    # Government DDMO
    gov_ddmo = np.sum(contract._consolidated_ddmo)

    # Government Take Income
    gov_tax_income = np.sum(contract._consolidated_tax_payment)

    # Government Take
    gov_take = np.sum(contract._consolidated_government_take)

    # Government Share
    gov_take_over_gross_rev = gov_take / gross_revenue

    # Calculation related to NPV Calculation which are depends on the NPV Mode
    if npv_mode == NPVSelection.FULL_CYCLE_REAL_TERMS:
        # Contractor NPV
        ctr_npv = npv_skk_real_terms(cashflow=contract._consolidated_cashflow,
                                     cashflow_years=contract.project_years,
                                     discount_rate=discount_rate,
                                     reference_year=reference_year)

        # Contractor NPV when the sunk cost is being pooled on the first year
        ctr_npv_sunk_cost_pooled = npv_skk_real_terms(cashflow=cashflow_sunk_cost_pooled,
                                                      cashflow_years=contract.project_years[
                                                                     len(contract._consolidated_sunk_cost) - 1:],
                                                      discount_rate=discount_rate,
                                                      reference_year=reference_year)

        # Contractor Investment NPV
        investment_npv = npv_skk_real_terms(cashflow=(contract._oil_tangible.expenditures() +
                                                      contract._gas_tangible.expenditures() +
                                                      contract._oil_intangible.expenditures() +
                                                      contract._gas_intangible.expenditures() +
                                                      contract._oil_opex.expenditures() +
                                                      contract._gas_opex.expenditures() +
                                                      contract._oil_asr.expenditures() +
                                                      contract._gas_asr.expenditures()
                                                      ),
                                            cashflow_years=contract.project_years,
                                            discount_rate=discount_rate,
                                            reference_year=reference_year)

        # Government Take Net Present Value
        gov_take_npv = npv_skk_real_terms(cashflow=contract._consolidated_government_take,
                                          cashflow_years=contract.project_years,
                                          discount_rate=discount_rate,
                                          reference_year=reference_year)

    elif npv_mode == NPVSelection.FULL_CYCLE_NOMINAL_TERMS:
        # Contractor NPV
        ctr_npv = npv_skk_nominal_terms(cashflow=contract._consolidated_cashflow,
                                        cashflow_years=contract.project_years,
                                        discount_rate=discount_rate)

        # Contractor NPV when the sunk cost is being pooled on the first year
        ctr_npv_sunk_cost_pooled = npv_skk_nominal_terms(cashflow=cashflow_sunk_cost_pooled,
                                                         cashflow_years=contract.project_years,
                                                         discount_rate=discount_rate)

        # Contractor Investment NPV
        investment_npv = npv_skk_nominal_terms(cashflow=(contract._consolidated_sunk_cost +
                                                         contract._oil_tangible.expenditures() +
                                                         contract._gas_tangible.expenditures() +
                                                         contract._oil_intangible.expenditures() +
                                                         contract._gas_intangible.expenditures() +
                                                         contract._oil_opex.expenditures() +
                                                         contract._gas_opex.expenditures() +
                                                         contract._oil_asr.expenditures() +
                                                         contract._gas_asr.expenditures()
                                                         ),
                                               cashflow_years=contract.project_years,
                                               discount_rate=discount_rate)

        # Government Take Net Present Value
        gov_take_npv = npv_skk_nominal_terms(cashflow=contract._consolidated_government_take,
                                             cashflow_years=contract.project_years,
                                             discount_rate=discount_rate)
    else:
        raise NPVMethodException

    # Contractor Present Value ratio to the investment npv
    ctr_pv_ratio = ctr_npv / investment_npv

    # Condition where the contract is Cost Recovery
    if isinstance(contract, CostRecovery):
        # Cost Recovery
        cost_recovery = np.sum(contract._consolidated_cost_recovery_after_tf, dtype=float)
        cost_recovery_over_gross_rev = cost_recovery / gross_revenue

        # Unrecoverable Cost
        unrec_cost = contract._consolidated_unrecovered_after_transfer[-1]
        unrec_over_gross_rev = unrec_cost / gross_revenue

        # Gross Share of Contractor and Government
        ctr_gross_share = np.sum(contract._consolidated_contractor_share, dtype=float)
        gov_gross_share = np.sum(contract._consolidated_government_share, dtype=float)

        # Contractor Net Share
        ctr_net_share = np.sum(contract._consolidated_ctr_net_share, dtype=float)
        ctr_net_share_over_gross_share = ctr_net_share / gross_revenue

        # Contractor Net Cashflow
        ctr_net_cashflow = np.sum(contract._consolidated_cashflow, dtype=float)
        ctr_net_cashflow_over_gross_rev = ctr_net_cashflow / gross_revenue

        # Government FTP Share
        gov_ftp_share = np.sum(contract._consolidated_ftp_gov)

        # Government Equity Share
        gov_equity_share = np.sum(contract._consolidated_government_share)

        # Contractor IRR
        ctr_irr = irr(cashflow=contract._consolidated_cashflow)

        # Contractor POT
        ctr_pot = pot(cashflow=contract._consolidated_cashflow[len(contract._consolidated_sunk_cost) - 2:])

        # Parsing the obtained variable into a dictionary, Summary
        summary = {'lifting_oil': lifting_oil,
                   'oil_wap': oil_wap,
                   'lifting_gas': lifting_gas,
                   'gas_wap': gas_wap,
                   'gross_revenue': gross_revenue,
                   'gross_revenue_oil': gross_revenue_oil,
                   'gross_revenue_gas': gross_revenue_gas,
                   'ctr_gross_share': ctr_gross_share,
                   'gov_gross_share': gov_gross_share,
                   'investment': investment,
                   'sunk_cost': sunk_cost,
                   'tangible': tangible,
                   'intangible': intangible,
                   'opex': opex,
                   'asr': asr,
                   'cost_recovery': cost_recovery,
                   'cost_recovery_over_gross_rev': cost_recovery_over_gross_rev,
                   'unrec_cost': unrec_cost,
                   'unrec_over_gross_rev': unrec_over_gross_rev,
                   'ctr_net_share': ctr_net_share,
                   'ctr_net_share_over_gross_share': ctr_net_share_over_gross_share,
                   'ctr_net_cashflow': ctr_net_cashflow,
                   'ctr_net_cashflow_over_gross_rev': ctr_net_cashflow_over_gross_rev,
                   'ctr_npv': ctr_npv,
                   'ctr_npv_sunk_cost_pooled': ctr_npv_sunk_cost_pooled,
                   'ctr_irr': ctr_irr,
                   'ctr_pot': ctr_pot,
                   'ctr_pv_ratio': ctr_pv_ratio,
                   'gov_ftp_share': gov_ftp_share,
                   'gov_equity_share': gov_equity_share,
                   'gov_ddmo': gov_ddmo,
                   'gov_tax_income': gov_tax_income,
                   'gov_take': gov_take,
                   'gov_take_over_gross_rev': gov_take_over_gross_rev,
                   'gov_take_npv': gov_take_npv}

    # Condition where the contract is Gross Split
    if isinstance(contract, GrossSplit):
        #  Deductible Cost
        deductible_cost = np.sum(contract._consolidated_deductible_cost)
        deductible_cost_over_gross_rev = deductible_cost / gross_revenue

        # Carry Forward Cost
        carry_forward_deductible_cost = contract._consolidated_carward_deduct_cost[-1]
        carry_forcost_over_gross_share = carry_forward_deductible_cost / gross_revenue

        # CTR Gross Share
        ctr_gross_share = np.sum(contract._consolidated_ctr_share_before_tf, dtype=float)

        # GOV GOV Gross Share
        gov_gross_share = np.sum(contract._consolidated_gov_share_before_tf, dtype=float)

        # CTR Net Share
        ctr_net_share = np.sum(contract._consolidated_ctr_net_share)
        ctr_net_share_over_gross_share = ctr_net_share / gross_revenue

        # CTR Net Cashflow
        ctr_net_cashflow = np.sum(contract._consolidated_cashflow)
        ctr_net_cashflow_over_gross_rev = ctr_net_cashflow / gross_revenue

        # Contractor IRR
        ctr_irr = irr(cashflow=contract._consolidated_cashflow)

        # Contractor POT
        ctr_pot = pot(cashflow=contract._consolidated_cashflow)

        # Parsing the obtained variable into a dictionary, Summary
        summary = {'lifting_oil': lifting_oil,
                   'oil_wap': oil_wap,
                   'lifting_gas': lifting_gas,
                   'gas_wap': gas_wap,
                   'gross_revenue': gross_revenue,
                   'gross_revenue_oil': gross_revenue_oil,
                   'gross_revenue_gas': gross_revenue_gas,
                   'ctr_gross_share': ctr_gross_share,
                   'gov_gross_share': gov_gross_share,
                   'investment': investment,
                   'sunk_cost': sunk_cost,
                   'tangible': tangible,
                   'intangible': intangible,
                   'opex': opex,
                   'asr': asr,
                   'deductible_cost': deductible_cost,
                   'cost_recovery_over_gross_rev': deductible_cost_over_gross_rev,
                   'unrec_cost': carry_forward_deductible_cost,
                   'unrec_over_gross_rev': carry_forcost_over_gross_share,
                   'ctr_net_share': ctr_net_share,
                   'ctr_net_share_over_gross_share': ctr_net_share_over_gross_share,
                   'ctr_net_cashflow': ctr_net_cashflow,
                   'ctr_net_cashflow_over_gross_rev': ctr_net_cashflow_over_gross_rev,
                   'ctr_npv': ctr_npv,
                   'ctr_npv_sunk_cost_pooled': ctr_npv_sunk_cost_pooled,
                   'ctr_irr': ctr_irr,
                   'ctr_pot': ctr_pot,
                   'ctr_pv_ratio': ctr_pv_ratio,
                   'gov_ddmo': gov_ddmo,
                   'gov_tax_income': gov_tax_income,
                   'gov_take': gov_take,
                   'gov_take_over_gross_rev': gov_take_over_gross_rev,
                   'gov_take_npv': gov_take_npv}

    return summary
