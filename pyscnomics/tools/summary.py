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


class SummaryException(Exception):
    """Exception to raise for a misuse of Summary Method"""

    pass


def get_summary(contract: BaseProject | CostRecovery | GrossSplit | Transition,
                reference_year: int,
                inflation_rate: float = 0,
                discount_rate: float = 0.1,
                npv_mode: NPVSelection = NPVSelection.NPV_SKK_REAL_TERMS,
                discounting_mode: DiscountingMode = DiscountingMode.END_YEAR,
                profitability_discounted: bool = False) -> dict:
    # Condition when the reference year is less than project start date year
    if isinstance(contract, Transition):
        contract_start_object = contract.contract1.start_date.year
        contract_end_object = contract.contract2.end_date.year

    else:
        contract_start_object = contract.start_date.year
        contract_end_object = contract.end_date.year

    if reference_year < contract_start_object:
        raise SummaryException(
            f"The Discounting Reference Year {reference_year} "
            f"is before the project years: {contract.start_date.year}"
        )

    # Condition when the reference year is after than project end date year
    if reference_year > contract_end_object:
        raise SummaryException(
            f"The Discounting Reference Year {reference_year} "
            f"is after the project years: {contract.end_date.year}"
        )

    # Defining the same summary parameters for any contract
    # Lifting
    lifting_oil = np.sum(contract._oil_lifting.get_lifting_rate_arr(), dtype=float)
    if np.sum(contract._oil_lifting.get_lifting_rate_arr()) == 0:
        oil_wap = 0.0
    else:
        oil_wap = np.divide(np.sum(contract._oil_revenue), np.sum(contract._oil_lifting.get_lifting_rate_arr()),
                            where=np.sum(contract._oil_lifting.get_lifting_rate_arr()) != 0)

    lifting_gas = np.sum(contract._gas_lifting.get_lifting_rate_arr(), dtype=float)
    if np.sum(contract._gas_lifting.get_lifting_rate_arr()) == 0:
        gas_wap = 0.0
    else:
        gas_wap = np.divide(np.sum(contract._gas_wap_price * contract._gas_lifting.get_lifting_rate_arr() * contract._gas_lifting.get_lifting_ghv_arr()), np.sum(
            contract._gas_lifting.get_lifting_rate_arr() * contract._gas_lifting.get_lifting_ghv_arr()), where=np.sum(
            contract._gas_lifting.get_lifting_rate_arr() * contract._gas_lifting.get_lifting_ghv_arr()) != 0)

    # Gross Revenue
    gross_revenue_oil = np.sum(contract._oil_revenue, dtype=float)
    gross_revenue_gas = np.sum(contract._gas_revenue, dtype=float)
    gross_revenue = np.sum(gross_revenue_oil + gross_revenue_gas, dtype=float)

    # Sunk Cost
    sunk_cost = np.sum(contract._oil_sunk_cost + contract._gas_sunk_cost, dtype=float)

    # Investment (Capital Cost)
    sunk_cost_extended = np.concatenate((contract._consolidated_sunk_cost,
                                         np.zeros(len(contract.project_years) - len(contract._consolidated_sunk_cost))))

    tangible = np.sum(contract._oil_capital_expenditures + contract._gas_capital_expenditures)

    intangible = np.sum(contract._oil_intangible_expenditures + contract._gas_intangible_expenditures - sunk_cost_extended,
                        dtype=float)
    investment = tangible + intangible

    # Undepreciated Asset
    undepreciated_asset_oil = np.sum(contract._oil_undepreciated_asset)
    undepreciated_asset_gas = np.sum(contract._gas_undepreciated_asset)
    undepreciated_asset_total = undepreciated_asset_oil + undepreciated_asset_gas

    # Capex
    oil_capex = np.sum(contract._oil_capital_expenditures)
    gas_capex = np.sum(contract._gas_capital_expenditures)

    # OPEX and ASR (Non-Capital Cost)
    opex = np.sum(contract._oil_opex_expenditures + contract._gas_opex_expenditures, dtype=float)
    asr = np.sum(contract._oil_asr_expenditures + contract._gas_asr_expenditures, dtype=float)

    # Cashflow sunk cost
    if sunk_cost == 0:
        cashflow_sunk_cost_pooled = contract._consolidated_cashflow
    else:
        cashflow_sunk_cost_pooled = np.concatenate(
            (np.array([-sunk_cost]),
             contract._consolidated_cashflow[len(contract._consolidated_sunk_cost):]))

    # Years sunk cost pooled
    years_sunk_cost_pooled = (contract.project_years[(len(contract.project_years) - len(cashflow_sunk_cost_pooled)):])

    if isinstance(contract, CostRecovery) or isinstance(contract, GrossSplit) or isinstance(contract, Transition):
        # Government DDMO
        gov_ddmo = np.sum(contract._consolidated_ddmo)

        # Government Take Income
        gov_tax_income = np.sum(contract._consolidated_tax_payment)

        # Government Take
        gov_take = np.sum(contract._consolidated_government_take)

        # Government Share
        gov_take_over_gross_rev = gov_take / gross_revenue

    elif isinstance(contract, BaseProject):
        gov_ddmo = 0
        gov_tax_income = 0
        gov_take = 0
        gov_take_over_gross_rev = 0

    else:
        gov_ddmo = 0
        gov_tax_income = 0
        gov_take = 0
        gov_take_over_gross_rev = 0

    # Contractor IRR
    ctr_irr = irr(cashflow=contract._consolidated_cashflow)
    ctr_irr_sunk_cost_pooled = irr(cashflow=cashflow_sunk_cost_pooled)

    # Calculation related to NPV Calculation which are depends on the NPV Mode
    # NPV Calculation for SKK Real Terms
    if npv_mode == NPVSelection.NPV_SKK_REAL_TERMS:
        # Contractor NPV
        ctr_npv = npv_skk_real_terms(cashflow=contract._consolidated_cashflow,
                                     cashflow_years=contract.project_years,
                                     discount_rate=discount_rate,
                                     reference_year=reference_year,
                                     discounting_mode=discounting_mode)

        # Contractor NPV when the sunk cost is being pooled on the first year
        ctr_npv_sunk_cost_pooled = npv_skk_real_terms(cashflow=cashflow_sunk_cost_pooled,
                                                      cashflow_years=years_sunk_cost_pooled,
                                                      discount_rate=discount_rate,
                                                      reference_year=reference_year,
                                                      discounting_mode=discounting_mode)

        # Contractor Investment NPV
        investment_npv = npv_skk_real_terms(cashflow=(
                contract._oil_capital_expenditures +
                contract._gas_capital_expenditures +
                contract._oil_intangible_expenditures +
                contract._gas_intangible_expenditures -
                sunk_cost_extended
        ),
            cashflow_years=contract.project_years,
            discount_rate=discount_rate,
            reference_year=reference_year,
            discounting_mode=discounting_mode)

        # Government Take Net Present Value
        gov_take_npv = npv_skk_real_terms(cashflow=contract._consolidated_government_take,
                                          cashflow_years=contract.project_years,
                                          discount_rate=discount_rate,
                                          reference_year=reference_year,
                                          discounting_mode=discounting_mode)

    # NPV Calculation for SKK Nominal Terms
    elif npv_mode == NPVSelection.NPV_SKK_NOMINAL_TERMS:
        # Contractor NPV
        ctr_npv = npv_skk_nominal_terms(cashflow=contract._consolidated_cashflow,
                                        cashflow_years=contract.project_years,
                                        discount_rate=discount_rate,
                                        discounting_mode=discounting_mode)

        # Contractor NPV when the sunk cost is being pooled on the first year
        ctr_npv_sunk_cost_pooled = npv_skk_nominal_terms(cashflow=cashflow_sunk_cost_pooled,
                                                         cashflow_years=years_sunk_cost_pooled,
                                                         discount_rate=discount_rate,
                                                         discounting_mode=discounting_mode)

        # Contractor Investment NPV
        investment_npv = npv_skk_nominal_terms(cashflow=(
                contract._oil_capital_expenditures +
                contract._gas_capital_expenditures +
                contract._oil_intangible_expenditures +
                contract._gas_intangible_expenditures -
                sunk_cost_extended
        ),
            cashflow_years=contract.project_years,
            discount_rate=discount_rate,
            discounting_mode=discounting_mode)

        # Government Take Net Present Value
        gov_take_npv = npv_skk_nominal_terms(cashflow=contract._consolidated_government_take,
                                             cashflow_years=contract.project_years,
                                             discount_rate=discount_rate,
                                             discounting_mode=discounting_mode)

    # NPV Calculation for Nominal Terms
    elif npv_mode == NPVSelection.NPV_NOMINAL_TERMS:
        # Contractor NPV
        ctr_npv = npv_nominal_terms(cashflow=contract._consolidated_cashflow,
                                    cashflow_years=contract.project_years,
                                    discount_rate=discount_rate,
                                    reference_year=reference_year,
                                    discounting_mode=discounting_mode)

        # Contractor NPV when the sunk cost is being pooled on the first year
        ctr_npv_sunk_cost_pooled = npv_nominal_terms(cashflow=cashflow_sunk_cost_pooled,
                                                     cashflow_years=years_sunk_cost_pooled,
                                                     discount_rate=discount_rate,
                                                     reference_year=reference_year,
                                                     discounting_mode=discounting_mode)

        # Contractor Investment NPV
        investment_npv = npv_nominal_terms(cashflow=(
                contract._oil_capital_expenditures +
                contract._gas_capital_expenditures +
                contract._oil_intangible_expenditures +
                contract._gas_intangible_expenditures -
                sunk_cost_extended
        ),
            cashflow_years=contract.project_years,
            discount_rate=discount_rate,
            reference_year=reference_year,
            discounting_mode=discounting_mode)

        # Government Take Net Present Value
        gov_take_npv = npv_nominal_terms(cashflow=contract._consolidated_government_take,
                                         cashflow_years=contract.project_years,
                                         discount_rate=discount_rate,
                                         reference_year=reference_year,
                                         discounting_mode=discounting_mode)

    # NPV Calculation for Real Terms
    elif npv_mode == NPVSelection.NPV_REAL_TERMS:
        # Contractor NPV
        ctr_npv = npv_real_terms(cashflow=contract._consolidated_cashflow,
                                 cashflow_years=contract.project_years,
                                 discount_rate=discount_rate,
                                 reference_year=reference_year,
                                 inflation_rate=inflation_rate,
                                 discounting_mode=discounting_mode)

        # Contractor NPV when the sunk cost is being pooled on the first year
        ctr_npv_sunk_cost_pooled = npv_real_terms(cashflow=cashflow_sunk_cost_pooled,
                                                  cashflow_years=years_sunk_cost_pooled,
                                                  discount_rate=discount_rate,
                                                  reference_year=reference_year,
                                                  inflation_rate=inflation_rate,
                                                  discounting_mode=discounting_mode)

        # Contractor Investment NPV
        investment_npv = npv_real_terms(cashflow=(
                contract._oil_capital_expenditures +
                contract._gas_capital_expenditures +
                contract._oil_intangible_expenditures +
                contract._gas_intangible_expenditures -
                sunk_cost_extended
        ),
            cashflow_years=contract.project_years,
            discount_rate=discount_rate,
            reference_year=reference_year,
            inflation_rate=inflation_rate,
            discounting_mode=discounting_mode)

        # Government Take Net Present Value
        gov_take_npv = npv_real_terms(cashflow=contract._consolidated_government_take,
                                      cashflow_years=contract.project_years,
                                      discount_rate=discount_rate,
                                      reference_year=reference_year,
                                      inflation_rate=inflation_rate,
                                      discounting_mode=discounting_mode)

    # NPV Calculation for Point Forwards
    else:
        # Contractor NPV
        ctr_npv = npv_point_forward(cashflow=contract._consolidated_cashflow,
                                    cashflow_years=contract.project_years,
                                    discount_rate=discount_rate,
                                    reference_year=reference_year,
                                    discounting_mode=discounting_mode)

        # Contractor NPV when the sunk cost is being pooled on the first year
        ctr_npv_sunk_cost_pooled = npv_point_forward(cashflow=cashflow_sunk_cost_pooled,
                                                     cashflow_years=years_sunk_cost_pooled,
                                                     discount_rate=discount_rate,
                                                     reference_year=reference_year,
                                                     discounting_mode=discounting_mode)

        # Contractor Investment NPV
        investment_npv = npv_point_forward(cashflow=(
                contract._oil_capital_expenditures +
                contract._gas_capital_expenditures +
                contract._oil_intangible_expenditures +
                contract._gas_intangible_expenditures -
                sunk_cost_extended
        ),
            cashflow_years=contract.project_years,
            discount_rate=discount_rate,
            reference_year=reference_year,
            discounting_mode=discounting_mode)

        # Government Take Net Present Value
        gov_take_npv = npv_point_forward(cashflow=contract._consolidated_government_take,
                                         cashflow_years=contract.project_years,
                                         discount_rate=discount_rate,
                                         reference_year=reference_year,
                                         discounting_mode=discounting_mode)

        #  Modifying the Contractor Net Cashflow since the cashflow before the reference year is neglected.
        ref_year_arr = np.full_like(contract._consolidated_cashflow, fill_value=reference_year)
        cashflow_point_forward = np.where(contract.project_years >= ref_year_arr,
                                          contract._consolidated_cashflow,
                                          0)
        gross_revenue_point_forward = np.where(contract.project_years >= ref_year_arr,
                                               contract._consolidated_revenue,
                                               0)
        ctr_net_cashflow = np.sum(cashflow_point_forward, dtype=float)
        gross_revenue_point_forward = np.sum(gross_revenue_point_forward, dtype=float)
        ctr_net_cashflow_over_gross_rev = ctr_net_cashflow / gross_revenue_point_forward

    # Contractor Present Value ratio to the investment npv
    # Condition when the Profitability Index is calculated using discounted investment
    if profitability_discounted:
        ctr_pv_ratio = np.divide(ctr_npv, investment_npv, where=investment_npv != 0)

    # Condition when the Profitability Index is calculated using un-discounted investment
    else:
        investment_pi = np.sum(
            contract._oil_capital_expenditures +
            contract._gas_capital_expenditures +
            contract._oil_intangible_expenditures +
            contract._gas_intangible_expenditures -
            sunk_cost_extended
        )
        ctr_pv_ratio = np.divide(ctr_npv, investment_pi, where=investment_pi != 0)

    ctr_pi = 1 + ctr_pv_ratio

    # Condition where the contract is Cost Recovery
    if isinstance(contract, CostRecovery):
        # Cost Recovery
        cost_recovery = np.sum(contract._consolidated_cost_recovery_after_tf, dtype=float)
        cost_recovery_over_gross_rev = cost_recovery / gross_revenue

        # Unrecoverable Cost
        unrec_cost = contract._consolidated_unrecovered_after_transfer[-1]
        unrec_over_costrec = unrec_cost / cost_recovery
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

        # Contractor POT
        ctr_pot = pot_psc(cashflow=contract._consolidated_cashflow,
                          cashflow_years=contract.project_years,
                          reference_year=reference_year)

        # Parsing the obtained variable into a dictionary, Summary
        return {'lifting_oil': lifting_oil,
                'oil_wap': oil_wap,
                'lifting_gas': lifting_gas,
                'gas_wap': gas_wap,
                'gross_revenue': gross_revenue,
                'gross_revenue_oil': gross_revenue_oil,
                'gross_revenue_gas': gross_revenue_gas,
                'ctr_gross_share': ctr_gross_share,
                'gov_gross_share': gov_gross_share,
                'investment': investment,
                'oil_capex': oil_capex,
                'gas_capex': gas_capex,
                'sunk_cost': sunk_cost,
                'tangible': tangible,
                'intangible': intangible,
                'opex_and_asr': opex + asr,
                'opex': opex,
                'asr': asr,
                'cost_recovery / deductible_cost': cost_recovery,
                'cost_recovery_over_gross_rev': cost_recovery_over_gross_rev,
                'unrec_cost': unrec_cost,
                'unrec_over_costrec': unrec_over_costrec,
                'unrec_over_gross_rev': unrec_over_gross_rev,
                'ctr_net_share': ctr_net_share,
                'ctr_net_share_over_gross_share': ctr_net_share_over_gross_share,
                'ctr_net_cashflow': ctr_net_cashflow,
                'ctr_net_cashflow_over_gross_rev': ctr_net_cashflow_over_gross_rev,
                'ctr_npv': ctr_npv,
                'ctr_npv_sunk_cost_pooled': ctr_npv_sunk_cost_pooled,
                'ctr_irr': ctr_irr,
                'ctr_irr_sunk_cost_pooled': ctr_irr_sunk_cost_pooled,
                'ctr_pot': ctr_pot,
                'ctr_pv_ratio': ctr_pv_ratio,
                'ctr_pi': ctr_pi,
                'gov_ftp_share': gov_ftp_share,
                'gov_equity_share': gov_equity_share,
                'gov_ddmo': gov_ddmo,
                'gov_tax_income': gov_tax_income,
                'gov_take': gov_take,
                'gov_take_over_gross_rev': gov_take_over_gross_rev,
                'gov_take_npv': gov_take_npv,
                'undepreciated_asset_oil': undepreciated_asset_oil,
                'undepreciated_asset_gas': undepreciated_asset_gas,
                'undepreciated_asset_total': undepreciated_asset_total
                }
    # Condition where the contract is Gross Split
    if isinstance(contract, GrossSplit):
        #  Deductible Cost
        deductible_cost = np.sum(contract._consolidated_deductible_cost)
        deductible_cost_over_gross_rev = deductible_cost / gross_revenue

        # Carry Forward Cost
        carry_forward_deductible_cost = contract._consolidated_carward_deduct_cost[-1]
        carry_forcost_over_gross_share = carry_forward_deductible_cost / gross_revenue
        carry_forcost_over_deductible_cost = carry_forward_deductible_cost / deductible_cost

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
        ctr_pot = pot_psc(cashflow=contract._consolidated_cashflow,
                          cashflow_years=contract.project_years,
                          reference_year=reference_year)

        # Parsing the obtained variable into a dictionary, Summary
        return {'lifting_oil': lifting_oil,
                'oil_wap': oil_wap,
                'lifting_gas': lifting_gas,
                'gas_wap': gas_wap,
                'gross_revenue': gross_revenue,
                'gross_revenue_oil': gross_revenue_oil,
                'gross_revenue_gas': gross_revenue_gas,
                'ctr_gross_share': ctr_gross_share,
                'gov_gross_share': gov_gross_share,
                'investment': investment,
                'oil_capex': oil_capex,
                'gas_capex': gas_capex,
                'sunk_cost': sunk_cost,
                'tangible': tangible,
                'intangible': intangible,
                'opex_and_asr': opex + asr,
                'opex': opex,
                'asr': asr,
                'cost_recovery / deductible_cost': deductible_cost,
                'cost_recovery_over_gross_rev': deductible_cost_over_gross_rev,
                'unrec_cost': carry_forward_deductible_cost,
                'unrec_over_costrec': carry_forcost_over_deductible_cost,
                'unrec_over_gross_rev': carry_forcost_over_gross_share,
                'ctr_net_share': ctr_net_share,
                'ctr_net_share_over_gross_share': ctr_net_share_over_gross_share,
                'ctr_net_cashflow': ctr_net_cashflow,
                'ctr_net_cashflow_over_gross_rev': ctr_net_cashflow_over_gross_rev,
                'ctr_npv': ctr_npv,
                'ctr_npv_sunk_cost_pooled': ctr_npv_sunk_cost_pooled,
                'ctr_irr': ctr_irr,
                'ctr_irr_sunk_cost_pooled': ctr_irr_sunk_cost_pooled,
                'ctr_pot': ctr_pot,
                'ctr_pv_ratio': ctr_pv_ratio,
                'ctr_pi': ctr_pi,
                'gov_ddmo': gov_ddmo,
                'gov_tax_income': gov_tax_income,
                'gov_take': gov_take,
                'gov_take_over_gross_rev': gov_take_over_gross_rev,
                'gov_take_npv': gov_take_npv,
                'gov_ftp_share': 0,
                'undepreciated_asset_oil': undepreciated_asset_oil,
                'undepreciated_asset_gas': undepreciated_asset_gas,
                'undepreciated_asset_total': undepreciated_asset_total
                }

    if isinstance(contract, Transition):
        ctr_ftp = np.sum(contract._ctr_ftp, dtype=float)
        gov_ftp = np.sum(contract._gov_ftp, dtype=float)
        ftp = ctr_ftp + gov_ftp

        deductible_cost = np.sum(contract._oil_deductible_cost+contract._gas_deductible_cost, dtype=float)
        deductible_cost_over_grossrev = deductible_cost / gross_revenue

        carry_forward_cost = (contract._oil_unrec_cost + contract._gas_unrec_cost)[-1]
        carry_forward_cost_over_grossrev = carry_forward_cost / gross_revenue

        ctr_equity_share = np.sum(contract._oil_ctr_ets + contract._gas_ctr_ets, dtype=float)
        gov_equity_share = np.sum(contract._oil_gov_ets + contract._gas_gov_ets, dtype=float)
        equity_to_be_split = ctr_equity_share + gov_equity_share

        ctr_net_operating_profit = np.sum(contract._net_operating_profit, dtype=float)
        ctr_net_operating_profit_over_grossrev = ctr_net_operating_profit / gross_revenue

        ctr_net_cash_flow = np.sum(contract._ctr_net_cashflow, dtype=float)
        ctr_net_cash_flow_over_grossrev = ctr_net_cash_flow / gross_revenue

        # Contractor POT
        ctr_pot = pot_psc(cashflow=contract._consolidated_cashflow,
                          cashflow_years=contract.project_years,
                          reference_year=reference_year)

        # Contractor Gross Share
        ctr_gross_share = np.sum(
            (
                contract.contract1._consolidated_ctr_share_before_tf
                if isinstance(contract.contract1, GrossSplit)
                else contract.contract1._consolidated_contractor_share
            ),
            dtype=float,
        ) + np.sum(
            (
                contract.contract2._consolidated_ctr_share_before_tf
                if isinstance(contract.contract2, GrossSplit)
                else contract.contract2._consolidated_contractor_share
            ),
            dtype=float,
        )

        # GOV GOV Gross Share
        gov_gross_share = np.sum(
            (
                contract.contract1._consolidated_gov_share_before_tf
                if isinstance(contract.contract1, GrossSplit)
                else contract.contract1._consolidated_government_share
            ),
            dtype=float,
        ) + np.sum(
            (
                contract.contract2._consolidated_gov_share_before_tf
                if isinstance(contract.contract2, GrossSplit)
                else contract.contract2._consolidated_government_share
            ),
            dtype=float,
        )

        # return {'lifting_oil': lifting_oil,
        #         'oil_wap': oil_wap,
        #         'lifting_gas': lifting_gas,
        #         'gas_wap': gas_wap,
        #         'gross_revenue': gross_revenue,
        #         'gross_revenue_oil': gross_revenue_oil,
        #         'gross_revenue_gas': gross_revenue_gas,
        #         'ftp': ftp,
        #         'gov_ftp': gov_ftp,
        #         'ctr_ftp': ctr_ftp,
        #         'sunk_cost': sunk_cost,
        #         'tangible': tangible,
        #         'intangible': intangible,
        #         'opex': opex,
        #         'asr': asr,
        #         'deductible_cost': deductible_cost,
        #         'deductible_cost_over_grossrev': deductible_cost_over_grossrev,
        #         'carry_forward_cost': carry_forward_cost,
        #         'carry_forward_cost_over_grossrev': carry_forward_cost_over_grossrev,
        #         'equity_to_be_split': equity_to_be_split,
        #         'ctr_equity_share': ctr_equity_share,
        #         'gov_equity_share': gov_equity_share,
        #         'ctr_net_operating_profit': ctr_net_operating_profit,
        #         'ctr_net_operating_profit_over_grossrev': ctr_net_operating_profit_over_grossrev,
        #         'ctr_net_cash_flow': ctr_net_cash_flow,
        #         'ctr_net_cash_flow_over_grossrev': ctr_net_cash_flow_over_grossrev,
        #         'ctr_npv': ctr_npv,
        #         'ctr_npv_sunk_cost_pooled': ctr_npv_sunk_cost_pooled,
        #         'ctr_irr': ctr_irr,
        #         'ctr_irr_sunk_cost_pooled': ctr_irr_sunk_cost_pooled,
        #         'ctr_pot': ctr_pot,
        #         'ctr_pv_ratio': ctr_pv_ratio,
        #         'gov_ddmo': gov_ddmo,
        #         'gov_tax_income': gov_tax_income,
        #         'gov_take': gov_take,
        #         'gov_take_over_gross_rev': gov_take_over_gross_rev,
        #         'gov_take_npv': gov_take_npv
        #         }

        # return {'lifting_oil': lifting_oil,
        #         'oil_wap': oil_wap,
        #         'lifting_gas': lifting_gas,
        #         'gas_wap': gas_wap,
        #         'gross_revenue': gross_revenue,
        #         'gross_revenue_oil': gross_revenue_oil,
        #         'gross_revenue_gas': gross_revenue_gas,
        #         'ftp': ftp,
        #         'gov_ftp': gov_ftp,
        #         'ctr_ftp': ctr_ftp,
        #         'sunk_cost': sunk_cost,
        #         'tangible': tangible,
        #         'intangible': intangible,
        #         'opex': opex,
        #         'asr': asr,
        #         'deductible_cost': deductible_cost,
        #         'deductible_cost_over_grossrev': deductible_cost_over_grossrev,
        #         'carry_forward_cost': carry_forward_cost,
        #         'carry_forward_cost_over_grossrev': carry_forward_cost_over_grossrev,
        #         'equity_to_be_split': equity_to_be_split,
        #         'ctr_equity_share': ctr_equity_share,
        #         'gov_equity_share': gov_equity_share,
        #         'ctr_net_operating_profit': ctr_net_operating_profit,
        #         'ctr_net_operating_profit_over_grossrev': ctr_net_operating_profit_over_grossrev,
        #         'ctr_net_cash_flow': ctr_net_cash_flow,
        #         'ctr_net_cash_flow_over_grossrev': ctr_net_cash_flow_over_grossrev,
        #         'ctr_npv': ctr_npv,
        #         'ctr_npv_sunk_cost_pooled': ctr_npv_sunk_cost_pooled,
        #         'ctr_irr': ctr_irr,
        #         'ctr_irr_sunk_cost_pooled': ctr_irr_sunk_cost_pooled,
        #         'ctr_pot': ctr_pot,
        #         'ctr_pv_ratio': ctr_pv_ratio,
        #         'gov_ddmo': gov_ddmo,
        #         'gov_tax_income': gov_tax_income,
        #         'gov_take': gov_take,
        #         'gov_take_over_gross_rev': gov_take_over_gross_rev,
        #         'gov_take_npv': gov_take_npv
        #         }

        return {'lifting_oil': lifting_oil,
                'oil_wap': oil_wap,
                'lifting_gas': lifting_gas,
                'gas_wap': gas_wap,
                'gross_revenue': gross_revenue,
                'gross_revenue_oil': gross_revenue_oil,
                'gross_revenue_gas': gross_revenue_gas,
                'ctr_gross_share': ctr_gross_share,
                'gov_gross_share': gov_gross_share,
                'ftp': ftp,
                'gov_ftp': gov_ftp,
                'ctr_ftp': ctr_ftp,
                'sunk_cost': sunk_cost,
                'investment': investment,
                'tangible': tangible,
                'intangible': intangible,
                'opex_and_asr': opex + asr,
                'opex': opex,
                'asr': asr,
                'cost_recovery / deductible_cost': deductible_cost,
                'cost_recovery_over_gross_rev': deductible_cost_over_grossrev,
                'unrec_cost': carry_forward_cost,
                'unrec_over_gross_rev': carry_forward_cost_over_grossrev,
                'equity_to_be_split': equity_to_be_split,
                'ctr_net_share': ctr_net_operating_profit,
                'ctr_net_share_over_gross_share': ctr_net_operating_profit_over_grossrev,
                'ctr_equity_share': ctr_equity_share,
                'gov_equity_share': gov_equity_share,
                'ctr_net_operating_profit': ctr_net_operating_profit,
                'ctr_net_operating_profit_over_grossrev': ctr_net_operating_profit_over_grossrev,
                'ctr_net_cashflow': ctr_net_cash_flow,
                'ctr_net_cashflow_over_gross_rev': ctr_net_cash_flow_over_grossrev,
                'ctr_npv': ctr_npv,
                'ctr_npv_sunk_cost_pooled': ctr_npv_sunk_cost_pooled,
                'ctr_irr': ctr_irr,
                'ctr_irr_sunk_cost_pooled': ctr_irr_sunk_cost_pooled,
                'ctr_pot': ctr_pot,
                'ctr_pv_ratio': ctr_pv_ratio,
                'ctr_pi': ctr_pi,
                'gov_ddmo': gov_ddmo,
                'gov_tax_income': gov_tax_income,
                'gov_take': gov_take,
                'gov_take_over_gross_rev': gov_take_over_gross_rev,
                'gov_take_npv': gov_take_npv,
                'gov_ftp_share': gov_ftp,
                'undepreciated_asset_oil': undepreciated_asset_oil,
                'undepreciated_asset_gas': undepreciated_asset_gas,
                'undepreciated_asset_total': undepreciated_asset_total
                }

    if isinstance(contract, BaseProject):
        # Contractor POT
        ctr_pot = pot_psc(cashflow=contract._consolidated_cashflow,
                          cashflow_years=contract.project_years,
                          reference_year=reference_year)

        ctr_net_share = gross_revenue - investment - opex - asr
        ctr_net_share_over_gross_share = ctr_net_share / gross_revenue

        ctr_net_cashflow = ctr_net_share
        ctr_net_cashflow_over_gross_rev = ctr_net_share_over_gross_share

        return {'lifting_oil': lifting_oil,
                'oil_wap': oil_wap,
                'lifting_gas': lifting_gas,
                'gas_wap': gas_wap,
                'gross_revenue': gross_revenue,
                'gross_revenue_oil': gross_revenue_oil,
                'gross_revenue_gas': gross_revenue_gas,
                'investment': investment,
                'oil_capex': oil_capex,
                'gas_capex': gas_capex,
                'sunk_cost': sunk_cost,
                'tangible': tangible,
                'intangible': intangible,
                'opex_and_asr': opex + asr,
                'opex': opex,
                'asr': asr,
                'ctr_npv': ctr_npv,
                'ctr_irr': ctr_irr,
                'ctr_pot': ctr_pot,
                'ctr_pv_ratio': ctr_pv_ratio,
                'ctr_pi': ctr_pi,
                'ctr_gross_share': gross_revenue,
                'ctr_net_share': ctr_net_share,
                'ctr_net_share_over_gross_share': ctr_net_share_over_gross_share,
                'ctr_net_cashflow': ctr_net_cashflow,
                'ctr_net_cashflow_over_gross_rev': ctr_net_cashflow_over_gross_rev,

                # Zero value for the psc terms
                'gov_gross_share': 0,
                'cost_recovery / deductible_cost': 0,
                'cost_recovery_over_gross_rev': 0,
                'unrec_cost': 0,
                'unrec_over_gross_rev': 0,
                'gov_ftp_share': 0,
                'gov_ddmo': 0,
                'gov_tax_income': 0,
                'gov_take': 0,
                'gov_take_over_gross_rev': 0,
                'gov_take_npv': 0,
                }


@dataclass
class Summary:
    """
    Dataclass to represent the economic indicator of a contract.

    Parameters
    ----------
    contract: BaseProject | CostRecovery | GrossSplit | Transition
        The contract which the summary of the economic indicator will be retrieved.
    reference_year: int
        The reference year for calculation of the economic indicator of the contract.
    inflation_rate: float
        The inflation rate used to calculate the economic indicator of the contract.
    discount_rate: float
        The discount rate used to calculate the Net Present Value (NPV).
    npv_mode: NPVSelection
        The calculation method to calculate the NPV related indicator.
    discounting_mode: DiscountingMode
        The discounting mode used in NPV calculation.
    profitability_discounted: bool
        The method used to calculate the Profitability Index.

    """
    contract: BaseProject | CostRecovery | GrossSplit | Transition
    reference_year: int | None = field(default=None)
    inflation_rate: float = field(default=0.0)
    discount_rate: float = field(default=0.1)
    npv_mode: NPVSelection = field(default=NPVSelection.NPV_SKK_REAL_TERMS)
    discounting_mode: DiscountingMode = field(default=DiscountingMode)
    profitability_discounted: bool = True

    # Attributes that will be defined later
    lifting_oil: float = field(default=0.0, init=False, repr=False)
    oil_wap: float = field(default=0.0, init=False, repr=False)
    lifting_gas: float = field(default=0.0, init=False, repr=False)
    gas_wap: float = field(default=0.0, init=False, repr=False)
    gross_revenue: float = field(default=0.0, init=False, repr=False)
    gross_revenue_oil: float = field(default=0.0, init=False, repr=False)
    gross_revenue_gas: float = field(default=0.0, init=False, repr=False)
    ctr_gross_share: float = field(default=0.0, init=False, repr=False)
    gov_gross_share: float = field(default=0.0, init=False, repr=False)
    investment: float = field(default=0.0, init=False, repr=False)
    oil_capex: float = field(default=0.0, init=False, repr=False)
    gas_capex: float = field(default=0.0, init=False, repr=False)
    sunk_cost: float = field(default=0.0, init=False, repr=False)
    capital_cost: float = field(default=0.0, init=False, repr=False)
    intangible: float = field(default=0.0, init=False, repr=False)
    opex_and_asr: float = field(default=0.0, init=False, repr=False)
    opex: float = field(default=0.0, init=False, repr=False)
    asr: float = field(default=0.0, init=False, repr=False)
    cost_recovery_deductible_cost: float = field(default=0.0, init=False, repr=False)
    cost_recovery_over_gross_rev: float = field(default=0.0, init=False, repr=False)
    unrec_cost: float = field(default=0.0, init=False, repr=False)
    unrec_over_costrec: float = field(default=0.0, init=False, repr=False)
    unrec_over_gross_rev: float = field(default=0.0, init=False, repr=False)
    ctr_net_share: float = field(default=0.0, init=False, repr=False)
    ctr_net_share_over_gross_share: float = field(default=0.0, init=False, repr=False)
    ctr_net_cashflow: float = field(default=0.0, init=False, repr=False)
    ctr_net_cashflow_over_gross_rev: float = field(default=0.0, init=False, repr=False)
    ctr_npv: float = field(default=0.0, init=False, repr=False)
    ctr_npv_sunk_cost_pooled: float = field(default=0.0, init=False, repr=False)
    ctr_irr: float = field(default=0.0, init=False, repr=False)
    ctr_irr_sunk_cost_pooled: float = field(default=0.0, init=False, repr=False)
    ctr_pot: float = field(default=0.0, init=False, repr=False)
    ctr_pv_ratio: float = field(default=0.0, init=False, repr=False)
    ctr_pi: float = field(default=0.0, init=False, repr=False)
    gov_ftp_share: float = field(default=0.0, init=False, repr=False)
    ctr_equity_share: float = field(default=0.0, init=False, repr=False)
    gov_equity_share: float = field(default=0.0, init=False, repr=False)
    gov_ddmo: float = field(default=0.0, init=False, repr=False)
    gov_tax_income: float = field(default=0.0, init=False, repr=False)
    gov_take: float = field(default=0.0, init=False, repr=False)
    gov_take_over_gross_rev: float = field(default=0.0, init=False, repr=False)
    gov_take_npv: float = field(default=0.0, init=False, repr=False)

    sunk_cost_extended: np.ndarray = field(default=0.0, init=False, repr=False)
    cashflow_sunk_cost_pooled: np.ndarray = field(default=0.0, init=False, repr=False)
    years_sunk_cost_pooled: np.ndarray = field(default=0.0, init=False, repr=False)

    def __post_init__(self):
        # Condition when the reference year is less than project start date year
        if isinstance(self.contract, Transition):
            contract_start_object = self.contract.contract1.start_date.year
            contract_end_object = self.contract.contract2.end_date.year

        else:
            contract_start_object = self.contract.start_date.year
            contract_end_object = self.contract.end_date.year

        if self.reference_year < contract_start_object:
            raise SummaryException(
                f"The Discounting Reference Year {self.reference_year} "
                f"is before the project years: {self.contract.start_date.year}"
            )

        # Condition when the reference year is after than project end date year
        if self.reference_year > contract_end_object:
            raise SummaryException(
                f"The Discounting Reference Year {self.reference_year} "
                f"is after the project years: {self.contract.end_date.year}"
            )

    def _get_sum_lifting(self):
        """
        Function to retrieve the lifting indicator into the corresponding class attributes.
        Returns
        -------
        Parsing the lifting data to the lifting attributes
        """
        # Lifting Oil
        self.lifting_oil = float(np.sum(self.contract._oil_lifting.get_lifting_rate_arr(), dtype=float))
        if np.sum(self.contract._oil_lifting.get_lifting_rate_arr()) == 0:
            self.oil_wap = 0.0
        else:
            self.oil_wap = np.divide(np.sum(self.contract._oil_revenue),
                                np.sum(self.contract._oil_lifting.get_lifting_rate_arr()),
                                where=np.sum(self.contract._oil_lifting.get_lifting_rate_arr()) != 0)

        self.lifting_gas = float(np.sum(self.contract._gas_lifting.get_lifting_rate_arr(), dtype=float))

        # Lifting Gas
        if np.sum(self.contract._gas_lifting.get_lifting_rate_arr()) == 0:
            self.gas_wap = 0.0
        else:
            self.gas_wap = np.divide(
                np.sum(self.contract._gas_wap_price * self.contract._gas_lifting.get_lifting_rate_arr()),
                np.sum(self.contract._gas_lifting.get_lifting_rate_arr()),
                where=np.sum(self.contract._gas_lifting.get_lifting_rate_arr()) != 0)

    def _get_gross_revenue(self):
        """
        Function to retrieve the gross revenue indicator into the corresponding class attributes.

        Returns
        -------
        Parsing the gross revenue data to the corresponding attributes
        """
        # Gross Revenue
        self.gross_revenue_oil = float(np.sum(self.contract._oil_revenue, dtype=float))
        self.gross_revenue_gas = float(np.sum(self.contract._gas_revenue, dtype=float))
        self.gross_revenue = float(np.sum(self.gross_revenue_oil + self.gross_revenue_gas, dtype=float))

    def _get_sunk_cost(self):
        """
        Function to retrieve the sunk cost indicator into the corresponding class attributes.

        Returns
        -------
        Parsing the sunk cost data to the lifting attributes

        """
        # Sunk Cost
        self.sunk_cost = float(np.sum(self.contract._oil_sunk_cost + self.contract._gas_sunk_cost, dtype=float))
        self.sunk_cost_extended = np.concatenate(
            (
                self.contract._consolidated_sunk_cost,
                np.zeros(
                    len(self.contract.project_years) - len(self.contract._consolidated_sunk_cost)
                )
            )
        )

    def _get_capital_cost(self):
        """
        Function to retrieve the capital cost indicator into the corresponding class attributes.

        Returns
        -------
        Parsing the lifting data to the corresponding attributes
        """
        # Investment Capital Cost
        self.capital_cost = float(np.sum(self.contract._oil_capital_expenditures + self.contract._gas_capital_expenditures))

    def _get_intangible_cost(self):
        """
        Function to retrieve the intangible cost indicator into the corresponding class attributes.

        Returns
        -------
        Parsing the capital cost data to the lifting attributes
        """
        # Investment Intangible Cost
        self.intangible = float(
            np.sum(
                self.contract._oil_intangible_expenditures +
                self.contract._gas_intangible_expenditures -
                self.sunk_cost_extended,
                dtype=float)
        )

    def _get_investment(self):
        """
        Function to retrieve the investment indicator into the corresponding class attributes.

        Returns
        -------
        Parsing the investment data to the corresponding attributes

        """
        # Investment total
        self.investment = self.capital_cost + self.intangible

    def _get_capex(self):
        """
        Function to retrieve the capital expenditures indicator into the corresponding class attributes.

        Returns
        -------
        Parsing the capital expenditures data to the corresponding attributes

        """
        # Capex
        self.oil_capex = float(np.sum(self.contract._oil_capital_expenditures))
        self.gas_capex = float(np.sum(self.contract._gas_capital_expenditures))

    def _get_opex(self):
        """
        Function to retrieve the OPEX indicator into the corresponding class attributes.

        Returns
        -------
        Parsing the opex data to the corresponding attributes

        """
        # OPEX
        self.opex = float(
            np.sum(
                self.contract._oil_opex_expenditures +
                self.contract._gas_opex_expenditures,
                dtype=float)
        )

    def _get_asr(self):
        """
        Function to retrieve the ASR indicator into the corresponding class attributes.

        Returns
        -------
        Parsing the ASR data to the corresponding attributes

        """
        self.asr = float(
            np.sum(
                self.contract._oil_asr_expenditures +
                self.contract._gas_asr_expenditures,
                dtype=float))

    def _get_cashflow_sunkcost_pooled(self):
        """
        Function to retrieve the Cash Flow if the sunk cost is pooled indicator into the corresponding class attributes.

        Returns
        -------
        Parsing the Cash Flow if the sunk cost is pooled data to the corresponding attributes

        """
        # Cashflow sunk cost
        if self.sunk_cost == 0:
            self.cashflow_sunk_cost_pooled = self.contract._consolidated_cashflow
        else:
            self.cashflow_sunk_cost_pooled = np.concatenate(
                (
                    np.array([-self.sunk_cost]),
                    self.contract._consolidated_cashflow[len(self.contract._consolidated_sunk_cost):]
                )
            )

        # Years sunk cost pooled
        self.years_sunk_cost_pooled = (
        self.contract.project_years[(len(self.contract.project_years) - len(self.cashflow_sunk_cost_pooled)):])

    def _get_ddmo(self):
        """
        Function to retrieve the DDMO indicator into the corresponding class attributes.

        Returns
        -------
        Parsing the DDMO data to the corresponding attributes

        """
        # Government DDMO
        if isinstance(
                self.contract, CostRecovery) or  isinstance(self.contract, GrossSplit) or isinstance(self.contract,
                                                                                                     Transition):
            self.gov_ddmo = float(np.sum(self.contract._consolidated_ddmo))
        elif isinstance(self.contract, BaseProject):
            self.gov_ddmo = 0.0
        else:
            self.gov_ddmo = 0.0

    def _get_gov_tax_income(self):
        """
        Function to retrieve the Government Tax indicator into the corresponding class attributes.

        Returns
        -------
        Parsing the Government Tax data to the corresponding attributes

        """
        # Government Tax Income
        if isinstance(
                self.contract, CostRecovery) or isinstance(self.contract, GrossSplit) or isinstance(self.contract,
                                                                                                    Transition):
            self.gov_tax_income = float(np.sum(self.contract._consolidated_tax_payment))
        elif isinstance(self.contract, BaseProject):
            self.gov_tax_income = 0.0
        else:
            self.gov_tax_income = 0.0

    def _get_gov_take(self):
        """
        Function to retrieve the Government Take indicator into the corresponding class attributes.

        Returns
        -------
        Parsing the Government Take data to the corresponding attributes.

        """
        # Government Take and Government Take over Gross Share
        if isinstance(
                self.contract, CostRecovery) or isinstance(self.contract, GrossSplit) or isinstance(self.contract,
                                                                                                    Transition):
            self.gov_take = float(np.sum(self.contract._consolidated_government_take))
            self.gov_take_over_gross_rev = self.gov_take / self.gross_revenue
        elif isinstance(self.contract, BaseProject):
            self.gov_take = 0.0
            self.gov_take_over_gross_rev = 0.0
        else:
            self.gov_take = 0.0
            self.gov_take_over_gross_rev = 0.0

    def _get_contractor_irr(self):
        """
        Function to retrieve the Contractor Take indicator into the corresponding class attributes.

        Returns
        -------
        Parsing the Government Take data to the corresponding attributes

        """
        self.ctr_irr = irr(cashflow=self.contract._consolidated_cashflow)
        self.ctr_irr_sunk_cost_pooled = irr(cashflow=self.cashflow_sunk_cost_pooled)

    def _npv_skk_real_terms(self):
        """
        Function to retrieve the NPV related indicator into the corresponding class attributes.
        The method used within this module is SKK Real Terms.

        Returns
        -------
        Parsing the NPV Related data to the corresponding attributes.

        """
        # Contractor NPV
        self.ctr_npv = npv_skk_real_terms(
            cashflow=self.contract._consolidated_cashflow,
            cashflow_years=self.contract.project_years,
            discount_rate=self.discount_rate,
            reference_year=self.reference_year,
            discounting_mode=self.discounting_mode)

        # Contractor NPV when the sunk cost is being pooled on the first year
        self.ctr_npv_sunk_cost_pooled = npv_skk_real_terms(
            cashflow=self.cashflow_sunk_cost_pooled,
            cashflow_years=self.years_sunk_cost_pooled,
            discount_rate=self.discount_rate,
            reference_year=self.reference_year,
            discounting_mode=self.discounting_mode)

        # Contractor Investment NPV
        self.investment_npv = npv_skk_real_terms(
            cashflow=(
                    self.contract._oil_capital_expenditures +
                    self.contract._gas_capital_expenditures +
                    self.contract._oil_intangible_expenditures +
                    self.contract._gas_intangible_expenditures -
                    self.sunk_cost_extended
            ),
            cashflow_years=self.contract.project_years,
            discount_rate=self.discount_rate,
            reference_year=self.reference_year,
            discounting_mode=self.discounting_mode)

        # Government Take Net Present Value
        self.gov_take_npv = npv_skk_real_terms(
            cashflow=self.contract._consolidated_government_take,
            cashflow_years=self.contract.project_years,
            discount_rate=self.discount_rate,
            reference_year=self.reference_year,
            discounting_mode=self.discounting_mode)

    def _npv_skk_nominal_terms(self):
        """
        Function to retrieve the NPV related indicator into the corresponding class attributes.
        The method used within this module is SKK Nominal Terms.

        Returns
        -------
        Parsing the NPV Related data to the corresponding attributes.

        """
        # Contractor NPV
        self.ctr_npv = npv_skk_nominal_terms(
            cashflow=self.contract._consolidated_cashflow,
            cashflow_years=self.contract.project_years,
            discount_rate=self.discount_rate,
            discounting_mode=self.discounting_mode)

        # Contractor NPV when the sunk cost is being pooled on the first year
        self.ctr_npv_sunk_cost_pooled = npv_skk_nominal_terms(
            cashflow=self.cashflow_sunk_cost_pooled,
            cashflow_years=self.years_sunk_cost_pooled,
            discount_rate=self.discount_rate,
            discounting_mode=self.discounting_mode)

        # Contractor Investment NPV
        self.investment_npv = npv_skk_nominal_terms(
            cashflow=(
                    self.contract._oil_capital_expenditures +
                    self.contract._gas_capital_expenditures +
                    self.contract._oil_intangible_expenditures +
                    self.contract._gas_intangible_expenditures -
                    self.sunk_cost_extended
            ),
            cashflow_years=self.contract.project_years,
            discount_rate=self.discount_rate,
            discounting_mode=self.discounting_mode)

        # Government Take Net Present Value
        self.gov_take_npv = npv_skk_nominal_terms(
            cashflow=self.contract._consolidated_government_take,
            cashflow_years=self.contract.project_years,
            discount_rate=self.discount_rate,
            discounting_mode=self.discounting_mode)

    def _npv_nominal_terms(self):
        """
        Function to retrieve the NPV related indicator into the corresponding class attributes.
        The method used within this module is Nominal Terms.

        Returns
        -------
        Parsing the NPV Related data to the corresponding attributes.

        """
        # Contractor NPV
        self.ctr_npv = npv_nominal_terms(
            cashflow=self.contract._consolidated_cashflow,
            cashflow_years=self.contract.project_years,
            discount_rate=self.discount_rate,
            reference_year=self.reference_year,
            discounting_mode=self.discounting_mode)

        # Contractor NPV when the sunk cost is being pooled on the first year
        self.ctr_npv_sunk_cost_pooled = npv_nominal_terms(
            cashflow=self.cashflow_sunk_cost_pooled,
            cashflow_years=self.years_sunk_cost_pooled,
            discount_rate=self.discount_rate,
            reference_year=self.reference_year,
            discounting_mode=self.discounting_mode)

        # Contractor Investment NPV
        self.investment_npv = npv_nominal_terms(
            cashflow=(
                    self.contract._oil_capital_expenditures +
                    self.contract._gas_capital_expenditures +
                    self.contract._oil_intangible_expenditures +
                    self.contract._gas_intangible_expenditures -
                    self.sunk_cost_extended
            ),
            cashflow_years=self.contract.project_years,
            discount_rate=self.discount_rate,
            reference_year=self.reference_year,
            discounting_mode=self.discounting_mode)

        # Government Take Net Present Value
        self.gov_take_npv = npv_nominal_terms(
            cashflow=self.contract._consolidated_government_take,
            cashflow_years=self.contract.project_years,
            discount_rate=self.discount_rate,
            reference_year=self.reference_year,
            discounting_mode=self.discounting_mode)

    def _npv_real_terms(self):
        """
        Function to retrieve the NPV related indicator into the corresponding class attributes.
        The method used within this module is Real Terms.

        Returns
        -------
        Parsing the NPV Related data to the corresponding attributes.

        """
        # Contractor NPV
        self.ctr_npv = npv_real_terms(
            cashflow=self.contract._consolidated_cashflow,
            cashflow_years=self.contract.project_years,
            discount_rate=self.discount_rate,
            reference_year=self.reference_year,
            inflation_rate=self.inflation_rate,
            discounting_mode=self.discounting_mode)

        # Contractor NPV when the sunk cost is being pooled on the first year
        self.ctr_npv_sunk_cost_pooled = npv_real_terms(
            cashflow=self.cashflow_sunk_cost_pooled,
            cashflow_years=self.years_sunk_cost_pooled,
            discount_rate=self.discount_rate,
            reference_year=self.reference_year,
            inflation_rate=self.inflation_rate,
            discounting_mode=self.discounting_mode)

        # Contractor Investment NPV
        self.investment_npv = npv_real_terms(
            cashflow=(
                    self.contract._oil_capital_expenditures +
                    self.contract._gas_capital_expenditures +
                    self.contract._oil_intangible_expenditures +
                    self.contract._gas_intangible_expenditures -
                    self.sunk_cost_extended
            ),
            cashflow_years=self.contract.project_years,
            discount_rate=self.discount_rate,
            reference_year=self.reference_year,
            inflation_rate=self.inflation_rate,
            discounting_mode=self.discounting_mode)

        # Government Take Net Present Value
        self.gov_take_npv = npv_real_terms(
            cashflow=self.contract._consolidated_government_take,
            cashflow_years=self.contract.project_years,
            discount_rate=self.discount_rate,
            reference_year=self.reference_year,
            inflation_rate=self.inflation_rate,
            discounting_mode=self.discounting_mode)

    def _npv_pointforward_terms(self):
        """
        Function to retrieve the NPV related indicator into the corresponding class attributes.
        The method used within this module is Point Forward.

        Returns
        -------
        Parsing the NPV Related data to the corresponding attributes.

        """
        # Contractor NPV
        self.ctr_npv = npv_point_forward(
            cashflow=self.contract._consolidated_cashflow,
            cashflow_years=self.contract.project_years,
            discount_rate=self.discount_rate,
            reference_year=self.reference_year,
            discounting_mode=self.discounting_mode)

        # Contractor NPV when the sunk cost is being pooled on the first year
        self.ctr_npv_sunk_cost_pooled = npv_point_forward(
            cashflow=self.cashflow_sunk_cost_pooled,
            cashflow_years=self.years_sunk_cost_pooled,
            discount_rate=self.discount_rate,
            reference_year=self.reference_year,
            discounting_mode=self.discounting_mode)

        # Contractor Investment NPV
        investment_npv = npv_point_forward(
            cashflow=(
                    self.contract._oil_capital_expenditures +
                    self.contract._gas_capital_expenditures +
                    self.contract._oil_intangible_expenditures +
                    self.contract._gas_intangible_expenditures -
                    self.sunk_cost_extended
            ),
            cashflow_years=self.contract.project_years,
            discount_rate=self.discount_rate,
            reference_year=self.reference_year,
            discounting_mode=self.discounting_mode)

        # Government Take Net Present Value
        self.gov_take_npv = npv_point_forward(
            cashflow=self.contract._consolidated_government_take,
            cashflow_years=self.contract.project_years,
            discount_rate=self.discount_rate,
            reference_year=self.reference_year,
            discounting_mode=self.discounting_mode)

        #  Modifying the Contractor Net Cashflow since the cashflow before the reference year is neglected.
        ref_year_arr = np.full_like(
            self.contract._consolidated_cashflow,
            fill_value=self.reference_year)

        cashflow_point_forward = np.where(
            self.contract.project_years >= ref_year_arr,
            self.contract._consolidated_cashflow,
            0)

        gross_revenue_point_forward = np.where(
            self.contract.project_years >= ref_year_arr,
            self.contract._consolidated_revenue,
            0)

        self.ctr_net_cashflow = float(np.sum(cashflow_point_forward, dtype=float))
        gross_revenue_point_forward = np.sum(gross_revenue_point_forward, dtype=float)
        self.ctr_net_cashflow_over_gross_rev = self.ctr_net_cashflow / gross_revenue_point_forward

    def _get_npv_indicator(self):
        """
        Function to utilized as wrapper of NPV related indicator.

        Returns
        -------
        Wrapper of the NPV Related data to the corresponding attributes.

        """
        if self.npv_mode == NPVSelection.NPV_SKK_REAL_TERMS:
            self._npv_skk_real_terms()
        elif self.npv_mode == NPVSelection.NPV_SKK_NOMINAL_TERMS:
            self._npv_skk_real_terms()
        elif self.npv_mode == NPVSelection.NPV_NOMINAL_TERMS:
            self._npv_nominal_terms()
        elif self.npv_mode == NPVSelection.NPV_REAL_TERMS:
            self._npv_real_terms()
        elif self.npv_mode == NPVSelection.NPV_POINT_FORWARD:
            self._npv_pointforward_terms()
        else:
            raise SummaryException(
                f"The NPV Method {self.npv_mode} , is not recognized"
            )

    def _get_profitability_index(self):
        """
        Function to retrieve the Profitability Index into the corresponding class attributes.

        Returns
        -------
        Parsing the Profitability data to the corresponding attributes.

        """
        if self.profitability_discounted:
            self.ctr_pv_ratio = np.divide(self.ctr_npv, self.investment_npv, where=self.investment_npv != 0)

        # Condition when the Profitability Index is calculated using un-discounted investment
        else:
            self.investment_pi = np.sum(
                self.contract._oil_capital_expenditures +
                self.contract._gas_capital_expenditures +
                self.contract._oil_intangible_expenditures +
                self.contract._gas_intangible_expenditures -
                self.sunk_cost_extended
            )
            self.ctr_pv_ratio = np.divide(self.ctr_npv, self.investment_pi, where=self.investment_pi != 0)

        self.ctr_pi = 1 + self.ctr_pv_ratio

    def _get_costrec_deductible_cost(self):
        """
        Function to retrieve the Cost Recovery or Deductible Cost indicator into the corresponding class attributes.

        Returns
        -------
        Parsing the Cost Recovery or Deductible Cost data to the corresponding attributes.

        Notes
        -----
        [1] The Cost Recovery indicator will be filled if the contract type is Cost Recovery.
        [2] The Deductible Cost indicator will be filled if the contract type is Gross Split.

        """
        if isinstance(self.contract, CostRecovery):
            # Cost Recovery
            self.cost_recovery = np.sum(self.contract._consolidated_cost_recovery_after_tf, dtype=float)
            self.cost_recovery_over_gross_rev = self.cost_recovery / self.gross_revenue

        elif isinstance(self.contract, GrossSplit):
            #  Deductible Cost
            self.deductible_cost = np.sum(self.contract._consolidated_deductible_cost)
            self.deductible_cost_over_gross_rev = self.deductible_cost / self.gross_revenue
        else:
            pass

    def _get_unrecoverable_carryforward_cost(self):
        """
        Function to retrieve the Unrecoverable Cost or Carry Forward Cost indicator into
        the corresponding class attributes.

        Returns
        -------
        Parsing the Unrecoverable Cost or Carry Forward Cost data to the corresponding attributes.

        Notes
        -----
        [1] The Unrecoverable cost indicator will be filled if the contract type is Cost Recovery.
        [2] The Carry Forward indicator will be filled if the contract type is Gross Split.

        """

        if isinstance(self.contract, CostRecovery):
            # Unrecoverable Cost
            self.unrec_cost = self.contract._consolidated_unrecovered_after_transfer[-1]
            self.unrec_over_costrec = self.unrec_cost / self.cost_recovery
            self.unrec_over_gross_rev = self.unrec_cost / self.gross_revenue

        elif isinstance(self.contract, GrossSplit):
            # Carry Forward Cost
            self.carry_forward_deductible_cost = self.contract._consolidated_carward_deduct_cost[-1]
            self.carry_forcost_over_gross_share = self.carry_forward_deductible_cost / self.gross_revenue
            self.carry_forcost_over_deductible_cost = self.carry_forward_deductible_cost / self.deductible_cost

        else:
            pass

    def _get_gross_share(self):
        """
        Function to retrieve the Contractor and Government Gross Share into the corresponding class attributes.

        Returns
        -------
        Parsing the Gross Share data to the corresponding attributes.

        """

        if isinstance(self.contract, CostRecovery):
            # Cost Recovery Gross Share
            self.ctr_gross_share = float(np.sum(self.contract._consolidated_contractor_share, dtype=float))
            self.gov_gross_share = float(np.sum(self.contract._consolidated_government_share, dtype=float))

        elif isinstance(self.contract, GrossSplit):
            # Gross Split Gross Share
            self.ctr_gross_share = float(np.sum(self.contract._consolidated_ctr_share_before_tf, dtype=float))
            self.gov_gross_share = float(np.sum(self.contract._consolidated_gov_share_before_tf, dtype=float))

        else:
            pass

    def _get_net_share(self):
        """
        Function to retrieve the Net Share into the corresponding class attributes.

        Returns
        -------
        Parsing the Net Share data to the corresponding attributes.

        """
        # Contractor Net Share
        self.ctr_net_share = float(np.sum(self.contract._consolidated_ctr_net_share, dtype=float))
        self.ctr_net_share_over_gross_share = self.ctr_net_share / self.gross_revenue

    def _get_netcashfflow(self):
        """
        Function to retrieve the Net Cashflow of contractor into the corresponding class attributes.

        Returns
        -------
        Parsing the Profitability data to the corresponding attributes.

        """
        # CTR Net Cashflow
        self.ctr_net_cashflow = float(np.sum(self.contract._consolidated_cashflow))
        self.ctr_net_cashflow_over_gross_rev = self.ctr_net_cashflow / self.gross_revenue

    def _get_ftp_share(self):
        """
        Function to retrieve the First Tranche Petroleum (FTP) into the corresponding class attributes.

        Returns
        -------
        Parsing the First Tranche Petroleum (FTP) data to the corresponding attributes.

        """
        # Government FTP Share
        self.gov_ftp_share = float(np.sum(self.contract._consolidated_ftp_gov))

    def _get_equity_share(self):
        """
        Function to retrieve the Government Equity Share into the corresponding class attributes.

        Returns
        -------
        Parsing the Government Equity Share data to the corresponding attributes.

        """
        self.gov_equity_share = float(np.sum(self.contract._consolidated_government_share))

    def _get_irr(self):
        """
        Function to retrieve the Government Equity Share into the corresponding class attributes.

        Returns
        -------
        Parsing the Government Equity Share data to the corresponding attributes.

        """
        # Contractor IRR
        self.ctr_irr = irr(
            cashflow=self.contract._consolidated_cashflow
        )

    def _get_pot(self):
        """
        Function to retrieve the Pay Out Time (POT) into the corresponding class attributes.

        Returns
        -------
        Parsing the POT data to the corresponding attributes.

        """
        self.ctr_pot = pot_psc(
            cashflow=self.contract._consolidated_cashflow,
            cashflow_years=self.contract.project_years,
            reference_year=self.reference_year
        )

    def _get_psc_indicator(self):
        """
        Function utilized as wrapper to retrieve the indicator of Production Sharing Contract.

        Returns
        -------
        Parsing the PSC data to the corresponding attributes.

        """
        # Retrieving Cost Recovery or Deductible Cost
        self._get_costrec_deductible_cost()

        # Retrieving Unrecoverable Cost or Carry Forward Cost
        self._get_unrecoverable_carryforward_cost()

        # Retrieving Gross Share
        self._get_gross_share()

        # Retrieving Net Share
        self._get_net_share()

        # Retrieving Net Cash Flow
        self._get_netcashfflow()

        # Retrieving FTP Share (Cost Recovery)
        self._get_ftp_share()

        # Retrieving Government Equity Share
        self._get_equity_share()

        # Retrieving POT
        self._get_pot()

    def _get_baseproject_indicator(self):
        # Retrieving POT
        self._get_pot()

        # Retrieving Net Share
        self._get_net_share()

        # Retrieving Net Cash Flow
        self._get_netcashfflow()

    def _get_transition_indicator(self):
        """
        Function to retrieve the economic indicator of the contract if the contract type is Transition.

        Returns
        -------
        Parsing the economic indicator of Transition contract.

        """
        self.ctr_ftp = np.sum(self.contract._ctr_ftp, dtype=float)
        self.gov_ftp = np.sum(self.contract._gov_ftp, dtype=float)
        self.ftp = self.ctr_ftp + self.gov_ftp

        self.deductible_cost = np.sum(self.contract._oil_deductible_cost + self.contract._gas_deductible_cost, dtype=float)
        self.deductible_cost_over_grossrev = self.deductible_cost / self.gross_revenue

        self.carry_forward_cost = (self.contract._oil_unrec_cost + self.contract._gas_unrec_cost)[-1]
        self.carry_forward_cost_over_grossrev = self.carry_forward_cost / self.gross_revenue

        self.ctr_equity_share = float(np.sum(self.contract._oil_ctr_ets + self.contract._gas_ctr_ets, dtype=float))
        self.gov_equity_share = float(np.sum(self.contract._oil_gov_ets + self.contract._gas_gov_ets, dtype=float))
        self.equity_to_be_split = self.ctr_equity_share + self.gov_equity_share

        self.ctr_net_operating_profit = np.sum(self.contract._net_operating_profit, dtype=float)
        self.ctr_net_operating_profit_over_grossrev = self.ctr_net_operating_profit / self.gross_revenue

        self.ctr_net_cash_flow = np.sum(self.contract._ctr_net_cashflow, dtype=float)
        self.ctr_net_cash_flow_over_grossrev = self.ctr_net_cash_flow / self.gross_revenue

        # Contractor POT
        self.ctr_pot = pot_psc(
            cashflow=self.contract._consolidated_cashflow,
            cashflow_years=self.contract.project_years,
            reference_year=self.reference_year)

        # Contractor Gross Share
        self.ctr_gross_share = np.sum(
            (
                self.contract.contract1._consolidated_ctr_share_before_tf
                if isinstance(self.contract.contract1, GrossSplit)
                else self.contract.contract1._consolidated_contractor_share
            ),
            dtype=float,
        ) + np.sum(
            (
                self.contract.contract2._consolidated_ctr_share_before_tf
                if isinstance(self.contract.contract2, GrossSplit)
                else self.contract.contract2._consolidated_contractor_share
            ),
            dtype=float,
        )

        # GOV GOV Gross Share
        self.gov_gross_share = np.sum(
            (
                self.contract.contract1._consolidated_gov_share_before_tf
                if isinstance(self.contract.contract1, GrossSplit)
                else self.contract.contract1._consolidated_gov_share_before_tf
            ),
            dtype=float,
        ) + np.sum(
            (
                self.contract.contract2._consolidated_gov_share_before_tf
                if isinstance(self.contract.contract2, GrossSplit)
                else self.contract.contract2._consolidated_gov_share_before_tf
            ),
            dtype=float,
        )

    def run(self):
        """
        Function utilized as the function wrapper which will retrieve the economic indicators of the contract.

        Returns
        -------
        Parsing the contract economic indicator to each corresponding attributes.

        """
        # Retrieving Lifting Indicator
        self._get_sum_lifting()

        # Retrieving Gross Revenue
        self._get_gross_revenue()

        # Retrieving Sunk Cost
        self._get_sunk_cost()

        # Retrieving Capital Cost Investment
        self._get_capital_cost()

        # Retrieving Intangible Investment
        self._get_intangible_cost()

        # Retrieving Investment Total
        self._get_investment()

        # Retrieving Capital Expenditure
        self._get_capex()

        # Retrieving OPEX
        self._get_opex()

        # Retrieving ASR
        self._get_asr()

        # Retrieving Cashflow Sunk Cost Pooled
        self._get_cashflow_sunkcost_pooled()

        # Retrieving DDMO
        self._get_ddmo()

        # Retrieving Government Tax Income
        self._get_gov_tax_income()

        # Retrieving Government Take
        self._get_gov_take()

        # Retrieving IRR
        self._get_contractor_irr()

        # Retrieving NPV Related Indicator
        self._get_npv_indicator()

        # Retrieving Profitability Index
        self._get_profitability_index()

        # Condition if the contract type is Cost Recovery or Gross Split
        if isinstance(self.contract, CostRecovery) or isinstance(self.contract, GrossSplit):
            self._get_psc_indicator()
        elif isinstance(self.contract, Transition):
            self._get_transition_indicator()
        else:
            self._get_baseproject_indicator()

    def get_summary_dict(self) -> dict:
        """
        Function to returning the contract economic indicator into a form of a dictionary.

        Returns
        -------
        out: dict
            The dictionary containing the economic indicator summary of the contract.


        Notes
        -----
        This function is made in this form maintain the chain of the previous version.

        """
        # Condition if the contract type is Cost Recovery or Gross Split
        if isinstance(self.contract, CostRecovery):
            return {
                'lifting_oil': self.lifting_oil,
                'oil_wap': self.oil_wap,
                'lifting_gas': self.lifting_gas,
                'gas_wap': self.gas_wap,
                'gross_revenue': self.gross_revenue,
                'gross_revenue_oil': self.gross_revenue_oil,
                'gross_revenue_gas': self.gross_revenue_gas,
                'ctr_gross_share': self.ctr_gross_share,
                'gov_gross_share': self.gov_gross_share,
                'investment': self.investment,
                'oil_capex': self.oil_capex,
                'gas_capex': self.gas_capex,
                'sunk_cost': self.sunk_cost,
                'tangible': self.capital_cost,
                'intangible': self.intangible,
                'opex_and_asr': self.opex + self.asr,
                'opex': self.opex,
                'asr': self.asr,
                'cost_recovery / deductible_cost': self.cost_recovery,
                'cost_recovery_over_gross_rev': self.cost_recovery_over_gross_rev,
                'unrec_cost': self.unrec_cost,
                'unrec_over_costrec': self.unrec_over_costrec,
                'unrec_over_gross_rev': self.unrec_over_gross_rev,
                'ctr_net_share': self.ctr_net_share,
                'ctr_net_share_over_gross_share': self.ctr_net_share_over_gross_share,
                'ctr_net_cashflow': self.ctr_net_cashflow,
                'ctr_net_cashflow_over_gross_rev': self.ctr_net_cashflow_over_gross_rev,
                'ctr_npv': self.ctr_npv,
                'ctr_npv_sunk_cost_pooled': self.ctr_npv_sunk_cost_pooled,
                'ctr_irr': self.ctr_irr,
                'ctr_irr_sunk_cost_pooled': self.ctr_irr_sunk_cost_pooled,
                'ctr_pot': self.ctr_pot,
                'ctr_pv_ratio': self.ctr_pv_ratio,
                'ctr_pi': self.ctr_pi,
                'gov_ftp_share': self.gov_ftp_share,
                'gov_equity_share': self.gov_equity_share,
                'gov_ddmo': self.gov_ddmo,
                'gov_tax_income': self.gov_tax_income,
                'gov_take': self.gov_take,
                'gov_take_over_gross_rev': self.gov_take_over_gross_rev,
                'gov_take_npv': self.gov_take_npv
            }

        elif isinstance(self.contract, GrossSplit):
            return {
                'lifting_oil': self.lifting_oil,
                'oil_wap': self.oil_wap,
                'lifting_gas': self.lifting_gas,
                'gas_wap': self.gas_wap,
                'gross_revenue': self.gross_revenue,
                'gross_revenue_oil': self.gross_revenue_oil,
                'gross_revenue_gas': self.gross_revenue_gas,
                'ctr_gross_share': self.ctr_gross_share,
                'gov_gross_share': self.gov_gross_share,
                'investment': self.investment,
                'oil_capex': self.oil_capex,
                'gas_capex': self.gas_capex,
                'sunk_cost': self.sunk_cost,
                'tangible': self.capital_cost,
                'intangible': self.intangible,
                'opex_and_asr': self.opex + self.asr,
                'opex': self.opex,
                'asr': self.asr,
                'cost_recovery / deductible_cost': self.deductible_cost,
                'cost_recovery_over_gross_rev': self.deductible_cost_over_gross_rev,
                'unrec_cost': self.carry_forward_deductible_cost,
                'unrec_over_costrec': self.carry_forcost_over_deductible_cost,
                'unrec_over_gross_rev': self.carry_forcost_over_gross_share,
                'ctr_net_share': self.ctr_net_share,
                'ctr_net_share_over_gross_share': self.ctr_net_share_over_gross_share,
                'ctr_net_cashflow': self.ctr_net_cashflow,
                'ctr_net_cashflow_over_gross_rev': self.ctr_net_cashflow_over_gross_rev,
                'ctr_npv': self.ctr_npv,
                'ctr_npv_sunk_cost_pooled': self.ctr_npv_sunk_cost_pooled,
                'ctr_irr': self.ctr_irr,
                'ctr_irr_sunk_cost_pooled': self.ctr_irr_sunk_cost_pooled,
                'ctr_pot': self.ctr_pot,
                'ctr_pv_ratio': self.ctr_pv_ratio,
                'ctr_pi': self.ctr_pi,
                'gov_ddmo': self.gov_ddmo,
                'gov_tax_income': self.gov_tax_income,
                'gov_take': self.gov_take,
                'gov_take_over_gross_rev': self.gov_take_over_gross_rev,
                'gov_take_npv': self.gov_take_npv,
                'gov_ftp_share': 0
            }

        elif isinstance(self.contract, Transition):
            return {
                'lifting_oil': self.lifting_oil,
                'oil_wap': self.oil_wap,
                'lifting_gas': self.lifting_gas,
                'gas_wap': self.gas_wap,
                'gross_revenue': self.gross_revenue,
                'gross_revenue_oil': self.gross_revenue_oil,
                'gross_revenue_gas': self.gross_revenue_gas,
                'ctr_gross_share': self.ctr_gross_share,
                'gov_gross_share': self.gov_gross_share,
                'ftp': self.ftp,
                'gov_ftp': self.gov_ftp,
                'ctr_ftp': self.ctr_ftp,
                'sunk_cost': self.sunk_cost,
                'investment': self.investment,
                'tangible': self.capital_cost,
                'intangible': self.intangible,
                'opex_and_asr': self.opex + self.asr,
                'opex': self.opex,
                'asr': self.asr,
                'cost_recovery / deductible_cost': self.deductible_cost,
                'cost_recovery_over_gross_rev': self.deductible_cost_over_grossrev,
                'unrec_cost': self.carry_forward_cost,
                'unrec_over_gross_rev': self.carry_forward_cost_over_grossrev,
                'equity_to_be_split': self.equity_to_be_split,
                'ctr_net_share': self.ctr_net_operating_profit,
                'ctr_net_share_over_gross_share': self.ctr_net_operating_profit_over_grossrev,
                'ctr_equity_share': self.ctr_equity_share,
                'gov_equity_share': self.gov_equity_share,
                'ctr_net_operating_profit': self.ctr_net_operating_profit,
                'ctr_net_operating_profit_over_grossrev': self.ctr_net_operating_profit_over_grossrev,
                'ctr_net_cashflow': self.ctr_net_cash_flow,
                'ctr_net_cashflow_over_gross_rev': self.ctr_net_cash_flow_over_grossrev,
                'ctr_npv': self.ctr_npv,
                'ctr_npv_sunk_cost_pooled': self.ctr_npv_sunk_cost_pooled,
                'ctr_irr': self.ctr_irr,
                'ctr_irr_sunk_cost_pooled': self.ctr_irr_sunk_cost_pooled,
                'ctr_pot': self.ctr_pot,
                'ctr_pv_ratio': self.ctr_pv_ratio,
                'ctr_pi': self.ctr_pi,
                'gov_ddmo': self.gov_ddmo,
                'gov_tax_income': self.gov_tax_income,
                'gov_take': self.gov_take,
                'gov_take_over_gross_rev': self.gov_take_over_gross_rev,
                'gov_take_npv': self.gov_take_npv,
                'gov_ftp_share': self.gov_ftp
            }
        else:
            return {
                'lifting_oil': self.lifting_oil,
                'oil_wap': self.oil_wap,
                'lifting_gas': self.lifting_gas,
                'gas_wap': self.gas_wap,
                'gross_revenue': self.gross_revenue,
                'gross_revenue_oil': self.gross_revenue_oil,
                'gross_revenue_gas': self.gross_revenue_gas,
                'investment': self.investment,
                'oil_capex': self.oil_capex,
                'gas_capex': self.gas_capex,
                'sunk_cost': self.sunk_cost,
                'tangible': self.capital_cost,
                'intangible': self.intangible,
                'opex_and_asr': self.opex + self.asr,
                'opex': self.opex,
                'asr': self.asr,
                'ctr_npv': self.ctr_npv,
                'ctr_irr': self.ctr_irr,
                'ctr_pot': self.ctr_pot,
                'ctr_pv_ratio': self.ctr_pv_ratio,
                'ctr_pi': self.ctr_pi,
                'ctr_gross_share': self.gross_revenue,
                'ctr_net_share': self.ctr_net_share,
                'ctr_net_share_over_gross_share': self.ctr_net_share_over_gross_share,
                'ctr_net_cashflow': self.ctr_net_cashflow,
                'ctr_net_cashflow_over_gross_rev': self.ctr_net_cashflow_over_gross_rev,

                # Zero value for the psc terms
                'gov_gross_share': 0,
                'cost_recovery / deductible_cost': 0,
                'cost_recovery_over_gross_rev': 0,
                'unrec_cost': 0,
                'unrec_over_gross_rev': 0,
                'gov_ftp_share': 0,
                'gov_ddmo': 0,
                'gov_tax_income': 0,
                'gov_take': 0,
                'gov_take_over_gross_rev': 0,
                'gov_take_npv': 0,
            }