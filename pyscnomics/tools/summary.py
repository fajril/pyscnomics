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


def get_summary(contract: BaseProject | CostRecovery | GrossSplit | Transition,
                reference_year: int,
                inflation_rate: float = 0,
                discount_rate: float = 0.1,
                npv_mode: NPVSelection = NPVSelection.NPV_SKK_REAL_TERMS,
                discounting_mode: DiscountingMode = DiscountingMode.END_YEAR) -> dict:
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
        gas_wap = np.divide(np.sum(contract._gas_wap_price * contract._gas_lifting.get_lifting_rate_arr()), np.sum(
            contract._gas_lifting.get_lifting_rate_arr()), where=np.sum(
            contract._gas_lifting.get_lifting_rate_arr()) != 0)

    # Gross Revenue
    gross_revenue_oil = np.sum(contract._oil_revenue, dtype=float)
    gross_revenue_gas = np.sum(contract._gas_revenue, dtype=float)
    gross_revenue = np.sum(gross_revenue_oil + gross_revenue_gas, dtype=float)

    # Sunk Cost
    sunk_cost = np.sum(contract._oil_sunk_cost + contract._gas_sunk_cost, dtype=float)

    # Investment (Capital Cost)
    sunk_cost_extended = np.concatenate((contract._consolidated_sunk_cost,
                                         np.zeros(len(contract.project_years) - len(contract._consolidated_sunk_cost))))

    tangible = np.sum(contract._oil_tangible_expenditures + contract._gas_tangible_expenditures)

    intangible = np.sum(contract._oil_intangible_expenditures + contract._gas_intangible_expenditures - sunk_cost_extended,
                        dtype=float)
    investment = tangible + intangible

    # Capex
    oil_capex = np.sum(contract._oil_tangible_expenditures)
    gas_capex = np.sum(contract._gas_tangible_expenditures)

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

    if isinstance(contract, BaseProject):
        gov_ddmo = 0
        gov_tax_income = 0
        gov_take = 0
        gov_take_over_gross_rev = 0

    if isinstance(contract, CostRecovery) or isinstance(contract, GrossSplit) or isinstance(contract, Transition):
        # Government DDMO
        gov_ddmo = np.sum(contract._consolidated_ddmo)

        # Government Take Income
        gov_tax_income = np.sum(contract._consolidated_tax_payment)

        # Government Take
        gov_take = np.sum(contract._consolidated_government_take)

        # Government Share
        gov_take_over_gross_rev = gov_take / gross_revenue

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
                contract._oil_tangible_expenditures +
                contract._gas_tangible_expenditures +
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
                contract._oil_tangible_expenditures +
                contract._gas_tangible_expenditures +
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
                contract._oil_tangible_expenditures +
                contract._gas_tangible_expenditures +
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
                contract._oil_tangible_expenditures +
                contract._gas_tangible_expenditures +
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
                contract._oil_tangible_expenditures +
                contract._gas_tangible_expenditures +
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
    ctr_pv_ratio = np.divide(ctr_npv, investment_npv, where=investment_npv != 0)
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
                'gov_take_npv': gov_take_npv}

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
                'gov_ftp_share': 0}

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
                'ctr_gross_share': '---',
                'gov_gross_share': '---',
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
                'gov_ftp_share': gov_ftp
                }

    if isinstance(contract, BaseProject):
        # Contractor POT
        ctr_pot = pot_psc(cashflow=contract._consolidated_cashflow,
                          cashflow_years=contract.project_years,
                          reference_year=reference_year)

        ctr_net_share = gross_revenue - investment + opex + asr
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
