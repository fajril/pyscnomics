"""
This code is used to generate the cashflow dataframe of a contract.
"""

import pandas as pd
import numpy as np
from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts.transition import Transition


def get_table(contract: BaseProject | CostRecovery | GrossSplit | Transition) -> tuple[
                                                                                     pd.DataFrame,
                                                                                     pd.DataFrame,
                                                                                     pd.DataFrame] | \
                                                                                 tuple[list[pd.DataFrame],list[
                                                                                     pd.DataFrame], list[pd.DataFrame]]:
    """
    A function to get the dataframe of the executed PSC object.

    Parameters
    ----------
    contract: CostRecovery | GrossSplit | Transition
        the psc object that its dataframe will be generated

    Returns
    -------
    out: tuple
        Dataframe of Oil Cashflow
        Dataframe of Gas Cashflow
        Dataframe of Consolidated Cashflow


    """
    if isinstance(contract, CostRecovery):
        psc_table_oil = pd.DataFrame()
        psc_table_oil['Year'] = contract.project_years
        psc_table_oil['Lifting'] = contract._oil_lifting.get_lifting_rate_arr()
        psc_table_oil['Price'] = contract._oil_wap_price
        psc_table_oil['Revenue'] = contract._oil_revenue
        psc_table_oil['Depreciable'] = contract._oil_capital_expenditures_post_tax
        psc_table_oil['Intangible'] = contract._oil_intangible_expenditures_post_tax
        psc_table_oil['Opex'] = contract._oil_opex_expenditures_post_tax
        psc_table_oil['ASR'] = contract._oil_asr_expenditures_post_tax
        psc_table_oil['Depreciation'] = contract._oil_depreciation
        psc_table_oil['Non_Capital'] = contract._oil_non_capital
        psc_table_oil['FTP'] = contract._oil_ftp
        psc_table_oil['FTP_CTR'] = contract._oil_ftp_ctr
        psc_table_oil['FTP_GOV'] = contract._oil_ftp_gov
        psc_table_oil['Investment_Credit'] = contract._oil_ic_paid
        psc_table_oil['Unrecovered_Cost'] = contract._oil_unrecovered_before_transfer
        psc_table_oil['Cost_to_Be_Recovered'] = contract._oil_cost_to_be_recovered
        psc_table_oil['Cost_Recovery'] = contract._oil_cost_recovery
        psc_table_oil['ETS_Before_Transfer'] = contract._oil_ets_before_transfer
        psc_table_oil['Transfer_to_Gas'] = contract._transfer_to_gas
        psc_table_oil['Unrec_after_Transfer'] = contract._oil_unrecovered_after_transfer
        psc_table_oil['Cost_To_Be_Recovered_After_TF'] = contract._oil_cost_to_be_recovered_after_tf
        psc_table_oil['Cost_Recovery_After_TF'] = contract._oil_cost_recovery_after_tf
        psc_table_oil['ETS_After_Transfer'] = contract._oil_ets_after_transfer
        psc_table_oil['Contractor_Share'] = contract._oil_contractor_share
        psc_table_oil['Government_Share'] = contract._oil_government_share
        psc_table_oil['DMO_Volume'] = contract._oil_dmo_volume
        psc_table_oil['DMO_Fee'] = contract._oil_dmo_fee
        psc_table_oil['DDMO'] = contract._oil_ddmo
        psc_table_oil['Taxable_Income'] = contract._oil_taxable_income
        psc_table_oil['Tax_Payment'] = contract._oil_tax_payment
        psc_table_oil['Contractor_Net_Share'] = contract._oil_ctr_net_share
        psc_table_oil['Cashflow'] = contract._oil_cashflow
        psc_table_oil['Cum_Cashflow'] = np.cumsum(contract._oil_cashflow)
        psc_table_oil['Government_Take'] = contract._oil_government_take
        # psc_table_oil.loc['Column_Total'] = psc_table_oil.sum(numeric_only=True, axis=0)

        psc_table_gas = pd.DataFrame()
        psc_table_gas['Year'] = contract.project_years
        psc_table_gas['Lifting'] = contract._gas_lifting.get_lifting_rate_arr()
        psc_table_gas['Price'] = contract._gas_wap_price
        psc_table_gas['Revenue'] = contract._gas_revenue
        psc_table_gas['Depreciable'] = contract._gas_capital_expenditures_post_tax
        psc_table_gas['Intangible'] = contract._gas_intangible_expenditures_post_tax
        psc_table_gas['Opex'] = contract._gas_opex_expenditures_post_tax
        psc_table_gas['ASR'] = contract._gas_asr_expenditures_post_tax
        psc_table_gas['Depreciation'] = contract._gas_depreciation
        psc_table_gas['Non_Capital'] = contract._gas_non_capital
        psc_table_gas['FTP'] = contract._gas_ftp
        psc_table_gas['FTP_CTR'] = contract._gas_ftp_ctr
        psc_table_gas['FTP_GOV'] = contract._gas_ftp_gov
        psc_table_gas['Investment_Credit'] = contract._gas_ic_paid
        psc_table_gas['Unrecovered_Cost'] = contract._gas_unrecovered_before_transfer
        psc_table_gas['Cost_to_Be_Recovered'] = contract._gas_cost_to_be_recovered
        psc_table_gas['Cost_Recovery'] = contract._gas_cost_recovery
        psc_table_gas['ETS_Before_Transfer'] = contract._gas_ets_before_transfer
        psc_table_gas['Transfer_to_Oil'] = contract._transfer_to_oil
        psc_table_gas['Unrec_after_Transfer'] = contract._gas_unrecovered_after_transfer
        psc_table_gas['Cost_To_Be_Recovered_After_TF'] = contract._gas_cost_to_be_recovered_after_tf
        psc_table_gas['Cost_Recovery_After_TF'] = contract._gas_cost_recovery_after_tf
        psc_table_gas['ETS_After_Transfer'] = contract._gas_ets_after_transfer
        psc_table_gas['Contractor_Share'] = contract._gas_contractor_share
        psc_table_gas['Government_Share'] = contract._gas_government_share
        psc_table_gas['DMO_Volume'] = contract._gas_dmo_volume
        psc_table_gas['DMO_Fee'] = contract._gas_dmo_fee
        psc_table_gas['DDMO'] = contract._gas_ddmo
        psc_table_gas['Taxable_Income'] = contract._gas_taxable_income
        psc_table_gas['Tax_Payment'] = contract._gas_tax_payment
        psc_table_gas['Contractor_Net_Share'] = contract._gas_ctr_net_share
        psc_table_gas['Cashflow'] = contract._gas_cashflow
        psc_table_gas['Cum_Cashflow'] = np.cumsum(contract._gas_cashflow)
        psc_table_gas['Government_Take'] = contract._gas_government_take
        # psc_table_gas.loc['Column_Total'] = psc_table_gas.sum(numeric_only=True, axis=0)

        psc_table_consolidated = pd.DataFrame()
        psc_table_consolidated['Year'] = contract.project_years
        psc_table_consolidated['Lifting_oil'] = contract._oil_lifting.get_lifting_rate_arr()
        psc_table_consolidated['Lifting_gas'] = contract._gas_lifting.get_lifting_rate_arr()
        psc_table_consolidated['C_Revenue'] = contract._consolidated_revenue
        psc_table_consolidated['C_Depreciable'] = contract._consolidated_capital_cost
        psc_table_consolidated['C_Intangible'] = contract._consolidated_intangible
        psc_table_consolidated['C_Opex'] = contract._consolidated_opex
        psc_table_consolidated['C_ASR'] = contract._consolidated_asr
        psc_table_consolidated['C_Depreciation'] = contract._consolidated_depreciation
        psc_table_consolidated['C_Non_Capital'] = contract._consolidated_non_capital
        psc_table_consolidated['C_FTP'] = contract._consolidated_ftp
        psc_table_consolidated['C_FTP_CTR'] = contract._consolidated_ftp_ctr
        psc_table_consolidated['C_FTP_GOV'] = contract._consolidated_ftp_gov
        psc_table_consolidated['C_IC'] = contract._consolidated_ic
        psc_table_consolidated[
            'C_Unrecovered_before_TF'] = contract._consolidated_unrecovered_before_transfer
        psc_table_consolidated['C_Cost_Recovery'] = contract._consolidated_cost_recovery_before_transfer
        psc_table_consolidated['C_ETS_before_TF'] = contract._consolidated_ets_before_transfer
        psc_table_consolidated['C_Unrecovered_after_TF'] = contract._consolidated_unrecovered_after_transfer
        psc_table_consolidated[
            'C_Cost_to_be_Recovered_after_TF'] = contract._consolidated_cost_to_be_recovered_after_tf
        psc_table_consolidated['C_Cost_Recovery_after_TF'] = contract._consolidated_cost_recovery_after_tf
        psc_table_consolidated['C_ETS_after_TF'] = contract._consolidated_ets_after_transfer
        psc_table_consolidated['C_Contractor_Share'] = contract._consolidated_contractor_share
        psc_table_consolidated['C_Government_Share'] = contract._consolidated_government_share
        psc_table_consolidated['C_DMO_Volume'] = contract._consolidated_dmo_volume
        psc_table_consolidated['C_DMO_Fee'] = contract._consolidated_dmo_fee
        psc_table_consolidated['C_DDMO'] = contract._consolidated_ddmo
        psc_table_consolidated['C_Taxable_Income'] = contract._consolidated_taxable_income
        psc_table_consolidated['C_Tax_Due'] = contract._consolidated_tax_due
        psc_table_consolidated['C_Unpaid_Tax_Balance'] = contract._consolidated_unpaid_tax_balance
        psc_table_consolidated['C_Tax_Payment'] = contract._consolidated_tax_payment
        psc_table_consolidated['C_CTR_Net_Share'] = contract._consolidated_ctr_net_share
        psc_table_consolidated['C_Contractor_Take'] = contract._consolidated_contractor_take
        psc_table_consolidated['C_Cashflow'] = contract._consolidated_cashflow
        psc_table_consolidated['cum_C_Cashflow'] = np.cumsum(contract._consolidated_cashflow)
        psc_table_consolidated['C_Government_take'] = contract._consolidated_government_take
        # psc_table_consolidated.loc['Column_Total'] = psc_table_consolidated.sum(numeric_only=True, axis=0)

        return psc_table_oil, psc_table_gas, psc_table_consolidated

    elif isinstance(contract, GrossSplit):
        psc_table_oil = pd.DataFrame()
        psc_table_oil['Years'] = contract.project_years
        psc_table_oil['Lifting'] = contract._oil_lifting.get_lifting_rate_arr()
        psc_table_oil['Price'] = contract._oil_wap_price
        psc_table_oil['Depreciable'] = contract._oil_capital_expenditures_post_tax
        psc_table_oil['Intangible'] = contract._oil_intangible_expenditures_post_tax
        psc_table_oil['Opex'] = contract._oil_opex_expenditures_post_tax
        psc_table_oil['ASR'] = contract._oil_asr_expenditures_post_tax
        psc_table_oil['Revenue'] = contract._oil_revenue
        psc_table_oil['Base_Split'] = contract._oil_base_split
        psc_table_oil['Variable_Split'] = contract._var_split_array
        psc_table_oil['Progressive_Price_Split'] = contract._oil_prog_price_split
        psc_table_oil['Progressive_Cumulative_Production_Split'] = contract._oil_prog_cum_split
        psc_table_oil['Progressive_Split'] = contract._oil_prog_split
        psc_table_oil['Contractor_Split'] = contract._oil_ctr_split
        psc_table_oil['Contractor_Share'] = contract._oil_ctr_share_before_transfer
        psc_table_oil['Government_Share'] = contract._oil_gov_share
        psc_table_oil['Depreciation'] = contract._oil_depreciation
        psc_table_oil['Non_Capital'] = contract._oil_non_capital
        psc_table_oil['Total_Expenses'] = contract._oil_total_expenses
        psc_table_oil['Cost_To_Be_Deducted'] = contract._oil_cost_tobe_deducted
        psc_table_oil['Carry_Forward_Cost'] = contract._oil_carward_deduct_cost
        psc_table_oil['Deductible_Cost'] = contract._oil_deductible_cost
        psc_table_oil['Transfer_To_Gas'] = contract._transfer_to_gas
        psc_table_oil['Carry_Forward_Cost_after_TF'] = contract._oil_carward_cost_aftertf
        psc_table_oil['CTR_Share_After_TF'] = contract._oil_ctr_share_after_transfer
        psc_table_oil['CTR_Net_Operating_Profit'] = contract._oil_net_operating_profit
        psc_table_oil['DMO_Volume'] = contract._oil_dmo_volume
        psc_table_oil['DMO_Fee'] = contract._oil_dmo_fee
        psc_table_oil['DDMO'] = contract._oil_ddmo
        psc_table_oil['Taxable_Income'] = contract._oil_taxable_income
        psc_table_oil['Tax'] = contract._oil_tax
        psc_table_oil['Net_CTR_Share'] = contract._oil_ctr_net_share
        psc_table_oil['CTR_Cash_Flow'] = contract._oil_ctr_cashflow
        psc_table_oil['Cum_Cash_Flow'] = np.cumsum(contract._oil_ctr_cashflow)
        psc_table_oil['Government_Take'] = contract._oil_government_take
        # psc_table_oil.loc['Column_Total'] = psc_table_oil.sum(numeric_only=True, axis=0)

        psc_table_gas = pd.DataFrame()
        psc_table_gas['Years'] = contract.project_years
        psc_table_gas['Lifting'] = contract._gas_lifting.get_lifting_rate_arr()
        psc_table_gas['Price'] = contract._gas_wap_price
        psc_table_gas['Depreciable'] = contract._gas_capital_expenditures_post_tax
        psc_table_gas['Intangible'] = contract._gas_intangible_expenditures_post_tax
        psc_table_gas['Opex'] = contract._gas_opex_expenditures_post_tax
        psc_table_gas['ASR'] = contract._gas_asr_expenditures_post_tax
        psc_table_gas['Revenue'] = contract._gas_revenue
        psc_table_gas['Base_Split'] = contract._gas_base_split
        psc_table_gas['Progressive_Price_Split'] = contract._gas_prog_price_split
        psc_table_gas['Progressive_Cumulative_Production_Split'] = contract._gas_prog_cum_split
        psc_table_gas['Variable_Split'] = contract._var_split_array
        psc_table_gas['Progressive_Split'] = contract._gas_prog_split
        psc_table_gas['Contractor_Split'] = contract._gas_ctr_split
        psc_table_gas['Contractor_Share'] = contract._gas_ctr_share_before_transfer
        psc_table_gas['Government_Share'] = contract._gas_gov_share
        psc_table_gas['Depreciation'] = contract._gas_depreciation
        psc_table_gas['Non_Capital'] = contract._gas_non_capital
        psc_table_gas['Total_Expenses'] = contract._gas_total_expenses
        psc_table_gas['Cost_To_Be_Deducted'] = contract._gas_cost_tobe_deducted
        psc_table_gas['Carry_Forward_Cost'] = contract._gas_carward_deduct_cost
        psc_table_gas['Deductible_Cost'] = contract._gas_deductible_cost
        psc_table_gas['Transfer_to_Oil'] = contract._transfer_to_oil
        psc_table_gas['Carry_Forward_Cost_after_TF'] = contract._gas_carward_cost_aftertf
        psc_table_gas['CTR_Share_After_TF'] = contract._gas_ctr_share_after_transfer
        psc_table_gas['CTR_Net_Operating_Profit'] = contract._gas_net_operating_profit
        psc_table_gas['DMO_Volume'] = contract._gas_dmo_volume
        psc_table_gas['DMO_Fee'] = contract._gas_dmo_fee
        psc_table_gas['DDMO'] = contract._gas_ddmo
        psc_table_gas['Taxable_Income'] = contract._gas_taxable_income
        psc_table_gas['Tax'] = contract._gas_tax
        psc_table_gas['Net_CTR_Share'] = contract._gas_ctr_net_share
        psc_table_gas['CTR_Cash_Flow'] = contract._gas_ctr_cashflow
        psc_table_gas['Cum_Cashflow'] = np.cumsum(contract._gas_ctr_cashflow)
        psc_table_gas['Government_Take'] = contract._gas_government_take
        # psc_table_gas.loc['Column_Total'] = psc_table_gas.sum(numeric_only=True, axis=0)

        psc_table_consolidated = pd.DataFrame()
        psc_table_consolidated['Years'] = contract.project_years
        psc_table_consolidated['C_Lifting_Oil'] = contract._oil_lifting.get_lifting_rate_arr()
        psc_table_consolidated['C_Lifting_Gas'] = contract._gas_lifting.get_lifting_rate_arr()
        psc_table_consolidated['C_Revenue'] = contract._consolidated_revenue
        psc_table_consolidated['C_Government_Share'] = contract._consolidated_ctr_share_before_tf
        psc_table_consolidated['C_Contractor_Share'] = contract._consolidated_gov_share_before_tf
        psc_table_consolidated['C_Depreciation'] = contract._consolidated_depreciation
        psc_table_consolidated['C_Opex'] = contract._consolidated_opex
        psc_table_consolidated['C_ASR'] = contract._consolidated_asr
        psc_table_consolidated['C_Non_Capital'] = contract._consolidated_non_capital
        psc_table_consolidated['C_Total_Expenses'] = contract._consolidated_total_expenses
        psc_table_consolidated['C_Cost_To_Be_Deducted'] = contract._consolidated_cost_tobe_deducted
        psc_table_consolidated['C_Carry_Forward_Cost'] = contract._consolidated_carward_deduct_cost
        psc_table_consolidated['C_Deductible_Cost'] = contract._consolidated_deductible_cost
        psc_table_consolidated['C_Carry_Forward_Cost_after_TF'] = contract._consolidated_carward_cost_aftertf
        psc_table_consolidated['C_CTR_Share_After'] = contract._consolidated_ctr_share_after_transfer
        psc_table_consolidated['C_CTR_Net_Operating_Profit'] = contract._consolidated_net_operating_profit
        psc_table_consolidated['C_DMO_Volume'] = contract._consolidated_dmo_volume
        psc_table_consolidated['C_DMO_Fee'] = contract._consolidated_dmo_fee
        psc_table_consolidated['C_DDMO'] = contract._consolidated_ddmo
        psc_table_consolidated['C_Taxable_Income'] = contract._consolidated_taxable_income
        psc_table_consolidated['C_Tax'] = contract._consolidated_tax_payment
        psc_table_consolidated['C_Net_CTR_Share'] = contract._consolidated_ctr_net_share
        psc_table_consolidated['C_CashFlow'] = contract._consolidated_cashflow
        psc_table_consolidated['C_Government_Take'] = contract._consolidated_government_take
        psc_table_consolidated['cum_C_CashFlow'] = np.cumsum(contract._consolidated_cashflow)
        # psc_table_consolidated.loc['Column_Total'] = psc_table_consolidated.sum(numeric_only=True, axis=0)

        return psc_table_oil, psc_table_gas, psc_table_consolidated

    elif isinstance(contract, Transition):
        psc_table_oil_1 = pd.DataFrame()
        psc_table_oil_2 = pd.DataFrame()
        psc_table_gas_1 = pd.DataFrame()
        psc_table_gas_2 = pd.DataFrame()
        psc_table_consolidated_1 = pd.DataFrame()
        psc_table_consolidated_2 = pd.DataFrame()

        if isinstance(contract.contract1, CostRecovery):
            psc_table_oil_1['Year'] = contract._contract1_transitioned.project_years
            psc_table_oil_1['Lifting'] = contract._contract1_transitioned._oil_lifting.get_lifting_rate_arr()
            psc_table_oil_1['Price'] = contract._contract1_transitioned._oil_wap_price
            psc_table_oil_1['Revenue'] = contract._contract1_transitioned._oil_revenue
            psc_table_oil_1['Depreciable'] = contract._contract1_transitioned._oil_capital_expenditures_post_tax
            psc_table_oil_1['Intangible'] = contract._contract1_transitioned._oil_intangible_expenditures_post_tax
            psc_table_oil_1['Opex'] = contract._contract1_transitioned._oil_opex_expenditures_post_tax
            psc_table_oil_1['ASR'] = contract._contract1_transitioned._oil_asr_expenditures_post_tax
            psc_table_oil_1['Depreciation'] = contract._contract1_transitioned._oil_depreciation
            psc_table_oil_1['Non_Capital'] = contract._contract1_transitioned._oil_non_capital
            psc_table_oil_1['FTP'] = contract._contract1_transitioned._oil_ftp
            psc_table_oil_1['FTP_CTR'] = contract._contract1_transitioned._oil_ftp_ctr
            psc_table_oil_1['FTP_GOV'] = contract._contract1_transitioned._oil_ftp_gov
            psc_table_oil_1['Investment_Credit'] = contract._contract1_transitioned._oil_ic_paid
            psc_table_oil_1['Unrecovered_Cost'] = contract._contract1_transitioned._oil_unrecovered_before_transfer
            psc_table_oil_1['Cost_to_Be_Recovered'] = contract._contract1_transitioned._oil_cost_to_be_recovered
            psc_table_oil_1['Cost_Recovery'] = contract._contract1_transitioned._oil_cost_recovery
            psc_table_oil_1['ETS_Before_Transfer'] = contract._contract1_transitioned._oil_ets_before_transfer
            psc_table_oil_1['Transfer_to_Gas'] = contract._contract1_transitioned._transfer_to_gas
            psc_table_oil_1['Unrec_after_Transfer'] = contract._contract1_transitioned._oil_unrecovered_after_transfer
            psc_table_oil_1[
                'Cost_To_Be_Recovered_After_TF'] = contract._contract1_transitioned._oil_cost_to_be_recovered_after_tf
            psc_table_oil_1['Cost_Recovery_After_TF'] = contract._contract1_transitioned._oil_cost_recovery_after_tf
            psc_table_oil_1['ETS_After_Transfer'] = contract._contract1_transitioned._oil_ets_after_transfer
            psc_table_oil_1['Contractor_Share'] = contract._contract1_transitioned._oil_contractor_share
            psc_table_oil_1['Government_Share'] = contract._contract1_transitioned._oil_government_share
            psc_table_oil_1['DMO_Volume'] = contract._contract1_transitioned._oil_dmo_volume
            psc_table_oil_1['DMO_Fee'] = contract._contract1_transitioned._oil_dmo_fee
            psc_table_oil_1['DDMO'] = contract._contract1_transitioned._oil_ddmo
            psc_table_oil_1['Taxable_Income'] = contract._contract1_transitioned._oil_taxable_income
            psc_table_oil_1['Tax_Payment'] = contract._contract1_transitioned._oil_tax_payment
            psc_table_oil_1['Contractor_Net_Share'] = contract._contract1_transitioned._oil_ctr_net_share
            psc_table_oil_1['Cashflow'] = contract._contract1_transitioned._oil_cashflow
            psc_table_oil_1['Cum._Cashflow'] = np.cumsum(contract._contract1_transitioned._oil_cashflow)
            psc_table_oil_1['Government_Take'] = contract._contract1_transitioned._oil_government_take
            # psc_table_oil.loc['Column_Total'] = psc_table_oil.sum(numeric_only=True, axis=0)

            psc_table_gas_1['Year'] = contract._contract1_transitioned.project_years
            psc_table_gas_1['Lifting'] = contract._contract1_transitioned._gas_lifting.get_lifting_rate_arr()
            psc_table_gas_1['Price'] = contract._contract1_transitioned._gas_wap_price
            psc_table_gas_1['Revenue'] = contract._contract1_transitioned._gas_revenue
            psc_table_gas_1['Depreciable'] = contract._contract1_transitioned._gas_capital_expenditures_post_tax
            psc_table_gas_1['Intangible'] = contract._contract1_transitioned._gas_intangible_expenditures_post_tax
            psc_table_gas_1['Opex'] = contract._contract1_transitioned._gas_opex_expenditures_post_tax
            psc_table_gas_1['ASR'] = contract._contract1_transitioned._gas_asr_expenditures_post_tax
            psc_table_gas_1['Depreciation'] = contract._contract1_transitioned._gas_depreciation
            psc_table_gas_1['Non_Capital'] = contract._contract1_transitioned._gas_non_capital
            psc_table_gas_1['FTP'] = contract._contract1_transitioned._gas_ftp
            psc_table_gas_1['FTP_CTR'] = contract._contract1_transitioned._gas_ftp_ctr
            psc_table_gas_1['FTP_GOV'] = contract._contract1_transitioned._gas_ftp_gov
            psc_table_gas_1['Investment_Credit'] = contract._contract1_transitioned._gas_ic_paid
            psc_table_gas_1['Unrecovered_Cost'] = contract._contract1_transitioned._gas_unrecovered_before_transfer
            psc_table_gas_1['Cost_to_Be_Recovered'] = contract._contract1_transitioned._gas_cost_to_be_recovered
            psc_table_gas_1['Cost_Recovery'] = contract._contract1_transitioned._gas_cost_recovery
            psc_table_gas_1['ETS_Before_Transfer'] = contract._contract1_transitioned._gas_ets_before_transfer
            psc_table_gas_1['Transfer_to_Oil'] = contract._contract1_transitioned._transfer_to_oil
            psc_table_gas_1['Unrec_after_Transfer'] = contract._contract1_transitioned._gas_unrecovered_after_transfer
            psc_table_gas_1[
                'Cost_To_Be_Recovered_After_TF'] = contract._contract1_transitioned._gas_cost_to_be_recovered_after_tf
            psc_table_gas_1['Cost_Recovery_After_TF'] = contract._contract1_transitioned._gas_cost_recovery_after_tf
            psc_table_gas_1['ETS_After_Transfer'] = contract._contract1_transitioned._gas_ets_after_transfer
            psc_table_gas_1['Contractor_Share'] = contract._contract1_transitioned._gas_contractor_share
            psc_table_gas_1['Government_Share'] = contract._contract1_transitioned._gas_government_share
            psc_table_gas_1['DMO_Volume'] = contract._contract1_transitioned._gas_dmo_volume
            psc_table_gas_1['DMO_Fee'] = contract._contract1_transitioned._gas_dmo_fee
            psc_table_gas_1['DDMO'] = contract._contract1_transitioned._gas_ddmo
            psc_table_gas_1['Taxable_Income'] = contract._contract1_transitioned._gas_taxable_income
            psc_table_gas_1['Tax_Payment'] = contract._contract1_transitioned._gas_tax_payment
            psc_table_gas_1['Contractor_Net_Share'] = contract._contract1_transitioned._gas_ctr_net_share
            psc_table_gas_1['Cashflow'] = contract._contract1_transitioned._gas_cashflow
            psc_table_gas_1['Cum._Cashflow'] = np.cumsum(contract._contract1_transitioned._gas_cashflow)
            psc_table_gas_1['Government_Take'] = contract._contract1_transitioned._gas_government_take
            # psc_table_gas.loc['Column_Total'] = psc_table_gas.sum(numeric_only=True, axis=0)

            psc_table_consolidated_1['Year'] = contract._contract1_transitioned.project_years
            psc_table_consolidated_1[
                'Lifting_oil'] = contract._contract1_transitioned._oil_lifting.get_lifting_rate_arr()
            psc_table_consolidated_1[
                'Lifting_gas'] = contract._contract1_transitioned._gas_lifting.get_lifting_rate_arr()
            psc_table_consolidated_1['C_Revenue'] = contract._contract1_transitioned._consolidated_revenue
            psc_table_consolidated_1['C_Depreciable'] = contract._contract1_transitioned._consolidated_capital_cost
            psc_table_consolidated_1['C_Intangible'] = contract._contract1_transitioned._consolidated_intangible
            psc_table_consolidated_1['C_Opex'] = contract._contract1_transitioned._consolidated_opex
            psc_table_consolidated_1['C_ASR'] = contract._contract1_transitioned._consolidated_asr
            psc_table_consolidated_1['C_Depreciation'] = contract._contract1_transitioned._consolidated_depreciation
            psc_table_consolidated_1['C_Non_Capital'] = contract._contract1_transitioned._consolidated_non_capital
            psc_table_consolidated_1['C_FTP'] = contract._contract1_transitioned._consolidated_ftp
            psc_table_consolidated_1['C_FTP_CTR'] = contract._contract1_transitioned._consolidated_ftp_ctr
            psc_table_consolidated_1['C_FTP_GOV'] = contract._contract1_transitioned._consolidated_ftp_gov
            psc_table_consolidated_1['C_IC'] = contract._contract1_transitioned._consolidated_ic
            psc_table_consolidated_1[
                'C_Unrecovered_before_TF'] = contract._contract1_transitioned._consolidated_unrecovered_before_transfer
            psc_table_consolidated_1[
                'C_Cost_Recovery'] = contract._contract1_transitioned._consolidated_cost_recovery_before_transfer
            psc_table_consolidated_1[
                'C_ETS_before_TF'] = contract._contract1_transitioned._consolidated_ets_before_transfer
            psc_table_consolidated_1[
                'C_Unrecovered_after_TF'] = contract._contract1_transitioned._consolidated_unrecovered_after_transfer
            psc_table_consolidated_1[
                'C_Cost_to_be_Recovered_after_TF'] = contract._contract1_transitioned._consolidated_cost_to_be_recovered_after_tf
            psc_table_consolidated_1[
                'C_Cost_Recovery_after_TF'] = contract._contract1_transitioned._consolidated_cost_recovery_after_tf
            psc_table_consolidated_1[
                'C_ETS_after_TF'] = contract._contract1_transitioned._consolidated_ets_after_transfer
            psc_table_consolidated_1[
                'C_Contractor_Share'] = contract._contract1_transitioned._consolidated_contractor_share
            psc_table_consolidated_1[
                'C_Government_Share'] = contract._contract1_transitioned._consolidated_government_share
            psc_table_consolidated_1['C_DMO_Volume'] = contract._contract1_transitioned._consolidated_dmo_volume
            psc_table_consolidated_1['C_DMO_Fee'] = contract._contract1_transitioned._consolidated_dmo_fee
            psc_table_consolidated_1['C_DDMO'] = contract._contract1_transitioned._consolidated_ddmo
            psc_table_consolidated_1['C_Taxable_Income'] = contract._contract1_transitioned._consolidated_taxable_income
            psc_table_consolidated_1['C_Tax_Due'] = contract._contract1_transitioned._consolidated_tax_due
            psc_table_consolidated_1[
                'C_Unpaid_Tax_Balance'] = contract._contract1_transitioned._consolidated_unpaid_tax_balance
            psc_table_consolidated_1['C_Tax_Payment'] = contract._contract1_transitioned._consolidated_tax_payment
            psc_table_consolidated_1['C_CTR_Net_Share'] = contract._contract1_transitioned._consolidated_ctr_net_share
            psc_table_consolidated_1[
                'C_Contractor_Take'] = contract._contract1_transitioned._consolidated_contractor_take
            psc_table_consolidated_1['C_Cashflow'] = contract._contract1_transitioned._consolidated_cashflow
            psc_table_consolidated_1['cum. C_Cashflow'] = np.cumsum(
                contract._contract1_transitioned._consolidated_cashflow)
            psc_table_consolidated_1[
                'C_Government_take'] = contract._contract1_transitioned._consolidated_government_take
            # psc_table_consolidated.loc['Column_Total'] = psc_table_consolidated.sum(numeric_only=True, axis=0)

        elif isinstance(contract.contract1, GrossSplit):
            psc_table_oil_1['Years'] = contract._contract1_transitioned.project_years
            psc_table_oil_1['Lifting'] = contract._contract1_transitioned._oil_lifting.get_lifting_rate_arr()
            psc_table_oil_1['Price'] = contract._contract1_transitioned._oil_wap_price
            psc_table_oil_1['Depreciable'] = contract._contract1_transitioned._oil_capital_expenditures_post_tax
            psc_table_oil_1['Intangible'] = contract._contract1_transitioned._oil_intangible_expenditures_post_tax
            psc_table_oil_1['Opex'] = contract._contract1_transitioned._oil_opex_expenditures_post_tax
            psc_table_oil_1['ASR'] = contract._contract1_transitioned._oil_asr_expenditures_post_tax
            psc_table_oil_1['Revenue'] = contract._contract1_transitioned._oil_revenue
            psc_table_oil_1['Base_Split'] = contract._contract1_transitioned._oil_base_split
            psc_table_oil_1['Variable_Split'] = contract._contract1_transitioned._var_split_array
            psc_table_oil_1['Base_Split'] = contract._contract1_transitioned._oil_base_split
            psc_table_oil_1['Progressive_Price_Split'] = contract._contract1_transitioned._oil_prog_price_split
            psc_table_oil_1['Progressive_Cumulative_Production_Split'] = contract._contract1_transitioned._oil_prog_cum_split
            psc_table_oil_1['Progressive_Split'] = contract._contract1_transitioned._oil_prog_split
            psc_table_oil_1['Contractor_Split'] = contract._contract1_transitioned._oil_ctr_split
            psc_table_oil_1['Contractor_Share'] = contract._contract1_transitioned._oil_ctr_share_before_transfer
            psc_table_oil_1['Government_Share'] = contract._contract1_transitioned._oil_gov_share
            psc_table_oil_1['Depreciation'] = contract._contract1_transitioned._oil_depreciation
            psc_table_oil_1['Non_Capital'] = contract._contract1_transitioned._oil_non_capital
            psc_table_oil_1['Total_Expenses'] = contract._contract1_transitioned._oil_total_expenses
            psc_table_oil_1['Cost_To_Be_Deducted'] = contract._contract1_transitioned._oil_cost_tobe_deducted
            psc_table_oil_1['Carry_Forward_Cost'] = contract._contract1_transitioned._oil_carward_deduct_cost
            psc_table_oil_1['Deductible_Cost'] = contract._contract1_transitioned._oil_deductible_cost
            psc_table_oil_1['Transfer_To_Gas'] = contract._contract1_transitioned._transfer_to_gas
            psc_table_oil_1['Carry_Forward_Cost_after_TF'] = contract._contract1_transitioned._oil_carward_cost_aftertf
            psc_table_oil_1['CTR_Share_After_TF'] = contract._contract1_transitioned._oil_ctr_share_after_transfer
            psc_table_oil_1['CTR_Net_Operating_Profit'] = contract._contract1_transitioned._oil_net_operating_profit
            psc_table_oil_1['DMO_Volume'] = contract._contract1_transitioned._oil_dmo_volume
            psc_table_oil_1['DMO_Fee'] = contract._contract1_transitioned._oil_dmo_fee
            psc_table_oil_1['DDMO'] = contract._contract1_transitioned._oil_ddmo
            psc_table_oil_1['Taxable_Income'] = contract._contract1_transitioned._oil_taxable_income
            psc_table_oil_1['Tax'] = contract._contract1_transitioned._oil_tax
            psc_table_oil_1['Net_CTR_Share'] = contract._contract1_transitioned._oil_ctr_net_share
            psc_table_oil_1['CTR_Cash_Flow'] = contract._contract1_transitioned._oil_ctr_cashflow
            psc_table_oil_1['Cum_Cash_Flow'] = np.cumsum(contract._contract1_transitioned._oil_ctr_cashflow)
            psc_table_oil_1['Government_Take'] = contract._contract1_transitioned._oil_government_take
            # psc_table_oil.loc['Column_Total'] = psc_table_oil.sum(numeric_only=True, axis=0)

            psc_table_gas_1['Years'] = contract.project_years
            psc_table_gas_1['Lifting'] = contract._contract1_transitioned._gas_lifting.get_lifting_rate_arr()
            psc_table_gas_1['Price'] = contract._contract1_transitioned._gas_wap_price
            psc_table_gas_1['Depreciable'] = contract._contract1_transitioned._gas_capital_expenditures_post_tax
            psc_table_gas_1['Intangible'] = contract._contract1_transitioned._gas_intangible_expenditures_post_tax
            psc_table_gas_1['Opex'] = contract._contract1_transitioned._gas_opex_expenditures_post_tax
            psc_table_gas_1['ASR'] = contract._contract1_transitioned._gas_asr_expenditures_post_tax
            psc_table_gas_1['Revenue'] = contract._contract1_transitioned._gas_revenue
            psc_table_gas_1['Base_Split'] = contract._contract1_transitioned._gas_base_split
            psc_table_gas_1['Variable_Split'] = contract._contract1_transitioned._var_split_array
            psc_table_gas_1['Base_Split'] = contract._contract1_transitioned._gas_base_split
            psc_table_gas_1['Progressive_Price_Split'] = contract._contract1_transitioned._gas_prog_price_split
            psc_table_gas_1['Progressive_Cumulative_Production_Split'] = contract._contract1_transitioned._gas_prog_cum_split
            psc_table_gas_1['Progressive_Split'] = contract._contract1_transitioned._gas_prog_split
            psc_table_gas_1['Contractor_Split'] = contract._contract1_transitioned._gas_ctr_split
            psc_table_gas_1['Contractor_Share'] = contract._contract1_transitioned._gas_ctr_share_before_transfer
            psc_table_gas_1['Government_Share'] = contract._contract1_transitioned._gas_gov_share
            psc_table_gas_1['Depreciation'] = contract._contract1_transitioned._gas_depreciation
            psc_table_gas_1['Non_Capital'] = contract._contract1_transitioned._gas_non_capital
            psc_table_gas_1['Total_Expenses'] = contract._contract1_transitioned._gas_total_expenses
            psc_table_gas_1['Cost_To_Be_Deducted'] = contract._contract1_transitioned._gas_cost_tobe_deducted
            psc_table_gas_1['Carry_Forward_Cost'] = contract._contract1_transitioned._gas_carward_deduct_cost
            psc_table_gas_1['Deductible_Cost'] = contract._contract1_transitioned._gas_deductible_cost
            psc_table_gas_1['Transfer_to_Oil'] = contract._contract1_transitioned._transfer_to_oil
            psc_table_gas_1['Carry_Forward_Cost_after_TF'] = contract._contract1_transitioned._gas_carward_cost_aftertf
            psc_table_gas_1['CTR_Share_After_TF'] = contract._contract1_transitioned._gas_ctr_share_after_transfer
            psc_table_gas_1['CTR_Net_Operating_Profit'] = contract._contract1_transitioned._gas_net_operating_profit
            psc_table_gas_1['DMO_Volume'] = contract._contract1_transitioned._gas_dmo_volume
            psc_table_gas_1['DMO_Fee'] = contract._contract1_transitioned._gas_dmo_fee
            psc_table_gas_1['DDMO'] = contract._contract1_transitioned._gas_ddmo
            psc_table_gas_1['Taxable_Income'] = contract._contract1_transitioned._gas_taxable_income
            psc_table_gas_1['Tax'] = contract._contract1_transitioned._gas_tax
            psc_table_gas_1['Net_CTR_Share'] = contract._contract1_transitioned._gas_ctr_net_share
            psc_table_gas_1['CTR_Cash_Flow'] = contract._contract1_transitioned._gas_ctr_cashflow
            psc_table_gas_1['Cum_Cash_Flow'] = np.cumsum(contract._contract2_transitioned._gas_ctr_cashflow)
            psc_table_gas_1['Government_Take'] = contract._contract2_transitioned._gas_government_take
            # psc_table_gas.loc['Column_Total'] = psc_table_gas.sum(numeric_only=True, axis=0)

            psc_table_consolidated_1['Years'] = contract._contract1_transitioned.project_years
            psc_table_consolidated_1['C_Lifting_Oil'] = contract._contract1_transitioned._oil_lifting.get_lifting_rate_arr()
            psc_table_consolidated_1['C_Lifting_Gas'] = contract._contract1_transitioned._gas_lifting.get_lifting_rate_arr()
            psc_table_consolidated_1['C_Revenue'] = contract._contract1_transitioned._consolidated_revenue
            psc_table_consolidated_1['C_Government_Share'] = contract._contract1_transitioned._consolidated_ctr_share_before_tf
            psc_table_consolidated_1['C_Contractor_Share'] = contract._contract1_transitioned._consolidated_gov_share_before_tf
            psc_table_consolidated_1['C_Depreciation'] = contract._contract1_transitioned._consolidated_depreciation
            psc_table_consolidated_1['C_Opex'] = contract._contract1_transitioned._consolidated_opex
            psc_table_consolidated_1['C_ASR'] = contract._contract1_transitioned._consolidated_asr
            psc_table_consolidated_1['C_Non_Capital'] = contract._contract1_transitioned._consolidated_non_capital
            psc_table_consolidated_1['C_Total_Expenses'] = contract._contract1_transitioned._consolidated_total_expenses
            psc_table_consolidated_1['C_Cost_To_Be_Deducted'] = contract._contract1_transitioned._consolidated_cost_tobe_deducted
            psc_table_consolidated_1['C_Carry_Forward_Cost'] = contract._contract1_transitioned._consolidated_carward_deduct_cost
            psc_table_consolidated_1['C_Deductible_Cost'] = contract._contract1_transitioned._consolidated_deductible_cost
            psc_table_consolidated_1['C_Carry_Forward_Cost_after_TF'] = contract._contract1_transitioned._consolidated_carward_cost_aftertf
            psc_table_consolidated_1['C_CTR_Share_After'] = contract._contract1_transitioned._consolidated_ctr_share_after_transfer
            psc_table_consolidated_1['C_CTR_Net_Operating_Profit'] = contract._contract1_transitioned._consolidated_net_operating_profit
            psc_table_consolidated_1['C_DMO_Volume'] = contract._contract1_transitioned._consolidated_dmo_volume
            psc_table_consolidated_1['C_DMO_Fee'] = contract._contract1_transitioned._consolidated_dmo_fee
            psc_table_consolidated_1['C_DDMO'] = contract._contract1_transitioned._consolidated_ddmo
            psc_table_consolidated_1['C_Taxable_Income'] = contract._contract1_transitioned._consolidated_taxable_income
            psc_table_consolidated_1['C_Tax'] = contract._contract1_transitioned._consolidated_tax_payment
            psc_table_consolidated_1['C_Net_CTR_Share'] = contract._contract1_transitioned._consolidated_ctr_net_share
            psc_table_consolidated_1['C_CashFlow'] = contract._contract1_transitioned._consolidated_cashflow
            psc_table_consolidated_1['C_Government_Take'] = contract._contract1_transitioned._consolidated_government_take
            psc_table_consolidated_1['cum_C_CashFlow'] = np.cumsum(
                contract._contract2_transitioned._consolidated_cashflow)
            # psc_table_consolidated.loc['Column_Total'] = psc_table_consolidated.sum(numeric_only=True, axis=0)

        if isinstance(contract.contract2, CostRecovery):
            psc_table_oil_2['Year'] = contract._contract2_transitioned.project_years
            psc_table_oil_2['Lifting'] = contract._contract2_transitioned._oil_lifting.get_lifting_rate_arr()
            psc_table_oil_2['Price'] = contract._contract2_transitioned._oil_wap_price
            psc_table_oil_2['Revenue'] = contract._contract2_transitioned._oil_revenue
            psc_table_oil_2['Depreciable'] = contract._contract2_transitioned._oil_capital_expenditures_post_tax
            psc_table_oil_2['Intangible'] = contract._contract2_transitioned._oil_intangible_expenditures_post_tax
            psc_table_oil_2['Opex'] = contract._contract2_transitioned._oil_opex_expenditures_post_tax
            psc_table_oil_2['ASR'] = contract._contract2_transitioned._oil_asr_expenditures_post_tax
            psc_table_oil_2['Depreciation'] = contract._contract2_transitioned._oil_depreciation
            psc_table_oil_2['Non_Capital'] = contract._contract2_transitioned._oil_non_capital
            psc_table_oil_2['FTP'] = contract._contract2_transitioned._oil_ftp
            psc_table_oil_2['FTP_CTR'] = contract._contract2_transitioned._oil_ftp_ctr
            psc_table_oil_2['FTP_GOV'] = contract._contract2_transitioned._oil_ftp_gov
            psc_table_oil_2['Investment_Credit'] = contract._contract2_transitioned._oil_ic_paid
            psc_table_oil_2['Unrecovered_Cost'] = contract._contract2_transitioned._oil_unrecovered_before_transfer
            psc_table_oil_2['Cost_to_Be_Recovered'] = contract._contract2_transitioned._oil_cost_to_be_recovered
            psc_table_oil_2['Cost_Recovery'] = contract._contract2_transitioned._oil_cost_recovery
            psc_table_oil_2['ETS_Before_Transfer'] = contract._contract2_transitioned._oil_ets_before_transfer
            psc_table_oil_2['Transfer_to_Gas'] = contract._contract2_transitioned._transfer_to_gas
            psc_table_oil_2['Unrec_after_Transfer'] = contract._contract2_transitioned._oil_unrecovered_after_transfer
            psc_table_oil_2[
                'Cost_To_Be_Recovered_After_TF'] = contract._contract2_transitioned._oil_cost_to_be_recovered_after_tf
            psc_table_oil_2['Cost_Recovery_After_TF'] = contract._contract2_transitioned._oil_cost_recovery_after_tf
            psc_table_oil_2['ETS_After_Transfer'] = contract._contract2_transitioned._oil_ets_after_transfer
            psc_table_oil_2['Contractor_Share'] = contract._contract2_transitioned._oil_contractor_share
            psc_table_oil_2['Government_Share'] = contract._contract2_transitioned._oil_government_share
            psc_table_oil_2['DMO_Volume'] = contract._contract2_transitioned._oil_dmo_volume
            psc_table_oil_2['DMO_Fee'] = contract._contract2_transitioned._oil_dmo_fee
            psc_table_oil_2['DDMO'] = contract._contract2_transitioned._oil_ddmo
            psc_table_oil_2['Taxable_Income'] = contract._contract2_transitioned._oil_taxable_income
            psc_table_oil_2['Tax_Payment'] = contract._contract2_transitioned._oil_tax_payment
            psc_table_oil_2['Contractor_Net_Share'] = contract._contract2_transitioned._oil_ctr_net_share
            psc_table_oil_2['Cashflow'] = contract._contract2_transitioned._oil_cashflow
            psc_table_oil_2['Cum._Cashflow'] = np.cumsum(contract._contract2_transitioned._oil_cashflow)
            psc_table_oil_2['Government_Take'] = contract._contract2_transitioned._oil_government_take
            # psc_table_oil.loc['Column_Total'] = psc_table_oil.sum(numeric_only=True, axis=0)

            psc_table_gas_2['Year'] = contract._contract2_transitioned.project_years
            psc_table_gas_2['Lifting'] = contract._contract2_transitioned._gas_lifting.get_lifting_rate_arr()
            psc_table_gas_2['Price'] = contract._contract2_transitioned._gas_wap_price
            psc_table_gas_2['Revenue'] = contract._contract2_transitioned._gas_revenue
            psc_table_gas_2['Depreciable'] = contract._contract2_transitioned._gas_capital_expenditures_post_tax
            psc_table_gas_2['Intangible'] = contract._contract2_transitioned._gas_intangible_expenditures_post_tax
            psc_table_gas_2['Opex'] = contract._contract2_transitioned._gas_opex_expenditures_post_tax
            psc_table_gas_2['ASR'] = contract._contract2_transitioned._gas_asr_expenditures_post_tax
            psc_table_gas_2['Depreciation'] = contract._contract2_transitioned._gas_depreciation
            psc_table_gas_2['Non_Capital'] = contract._contract2_transitioned._gas_non_capital
            psc_table_gas_2['FTP'] = contract._contract2_transitioned._gas_ftp
            psc_table_gas_2['FTP_CTR'] = contract._contract2_transitioned._gas_ftp_ctr
            psc_table_gas_2['FTP_GOV'] = contract._contract2_transitioned._gas_ftp_gov
            psc_table_gas_2['Investment_Credit'] = contract._contract2_transitioned._gas_ic_paid
            psc_table_gas_2['Unrecovered_Cost'] = contract._contract2_transitioned._gas_unrecovered_before_transfer
            psc_table_gas_2['Cost_to_Be_Recovered'] = contract._contract2_transitioned._gas_cost_to_be_recovered
            psc_table_gas_2['Cost_Recovery'] = contract._contract2_transitioned._gas_cost_recovery
            psc_table_gas_2['ETS_Before_Transfer'] = contract._contract2_transitioned._gas_ets_before_transfer
            psc_table_gas_2['Transfer_to_Oil'] = contract._contract2_transitioned._transfer_to_oil
            psc_table_gas_2['Unrec_after_Transfer'] = contract._contract2_transitioned._gas_unrecovered_after_transfer
            psc_table_gas_2[
                'Cost_To_Be_Recovered_After_TF'] = contract._contract2_transitioned._gas_cost_to_be_recovered_after_tf
            psc_table_gas_2['Cost_Recovery_After_TF'] = contract._contract2_transitioned._gas_cost_recovery_after_tf
            psc_table_gas_2['ETS_After_Transfer'] = contract._contract2_transitioned._gas_ets_after_transfer
            psc_table_gas_2['Contractor_Share'] = contract._contract2_transitioned._gas_contractor_share
            psc_table_gas_2['Government_Share'] = contract._contract2_transitioned._gas_government_share
            psc_table_gas_2['DMO_Volume'] = contract._contract2_transitioned._gas_dmo_volume
            psc_table_gas_2['DMO_Fee'] = contract._contract2_transitioned._gas_dmo_fee
            psc_table_gas_2['DDMO'] = contract._contract2_transitioned._gas_ddmo
            psc_table_gas_2['Taxable_Income'] = contract._contract2_transitioned._gas_taxable_income
            psc_table_gas_2['Tax_Payment'] = contract._contract2_transitioned._gas_tax_payment
            psc_table_gas_2['Contractor_Net_Share'] = contract._contract2_transitioned._gas_ctr_net_share
            psc_table_gas_2['Cashflow'] = contract._contract2_transitioned._gas_cashflow
            psc_table_gas_2['Cum._Cashflow'] = np.cumsum(contract._contract2_transitioned._gas_cashflow)
            psc_table_gas_2['Government_Take'] = contract._contract2_transitioned._gas_government_take
            # psc_table_gas.loc['Column_Total'] = psc_table_gas.sum(numeric_only=True, axis=0)

            psc_table_consolidated_2['Year'] = contract._contract2_transitioned.project_years
            psc_table_consolidated_2[
                'Lifting_oil'] = contract._contract2_transitioned._oil_lifting.get_lifting_rate_arr()
            psc_table_consolidated_2[
                'Lifting_gas'] = contract._contract2_transitioned._gas_lifting.get_lifting_rate_arr()
            psc_table_consolidated_2['C_Revenue'] = contract._contract2_transitioned._consolidated_revenue
            psc_table_consolidated_2['C_Depreciable'] = contract._contract2_transitioned._consolidated_capital_cost
            psc_table_consolidated_2['C_Intangible'] = contract._contract2_transitioned._consolidated_intangible
            psc_table_consolidated_2['C_Opex'] = contract._contract2_transitioned._consolidated_opex
            psc_table_consolidated_2['C_ASR'] = contract._contract2_transitioned._consolidated_asr
            psc_table_consolidated_2['C_Depreciation'] = contract._contract2_transitioned._consolidated_depreciation
            psc_table_consolidated_2['C_Non_Capital'] = contract._contract2_transitioned._consolidated_non_capital
            psc_table_consolidated_2['C_FTP'] = contract._contract2_transitioned._consolidated_ftp
            psc_table_consolidated_2['C_FTP_CTR'] = contract._contract2_transitioned._consolidated_ftp_ctr
            psc_table_consolidated_2['C_FTP_GOV'] = contract._contract2_transitioned._consolidated_ftp_gov
            psc_table_consolidated_2['C_IC'] = contract._contract2_transitioned._consolidated_ic
            psc_table_consolidated_2[
                'C_Unrecovered_before_TF'] = contract._contract2_transitioned._consolidated_unrecovered_before_transfer
            psc_table_consolidated_2[
                'C_Cost_Recovery'] = contract._contract2_transitioned._consolidated_cost_recovery_before_transfer
            psc_table_consolidated_2[
                'C_ETS_before_TF'] = contract._contract2_transitioned._consolidated_ets_before_transfer
            psc_table_consolidated_2[
                'C_Unrecovered_after_TF'] = contract._contract2_transitioned._consolidated_unrecovered_after_transfer
            psc_table_consolidated_2[
                'C_Cost_to_be_Recovered_after_TF'] = contract._contract2_transitioned._consolidated_cost_to_be_recovered_after_tf
            psc_table_consolidated_2[
                'C_Cost_Recovery_after_TF'] = contract._contract2_transitioned._consolidated_cost_recovery_after_tf
            psc_table_consolidated_2[
                'C_ETS_after_TF'] = contract._contract2_transitioned._consolidated_ets_after_transfer
            psc_table_consolidated_2[
                'C_Contractor_Share'] = contract._contract2_transitioned._consolidated_contractor_share
            psc_table_consolidated_2[
                'C_Government_Share'] = contract._contract2_transitioned._consolidated_government_share
            psc_table_consolidated_2['C_DMO_Volume'] = contract._contract2_transitioned._consolidated_dmo_volume
            psc_table_consolidated_2['C_DMO_Fee'] = contract._contract2_transitioned._consolidated_dmo_fee
            psc_table_consolidated_2['C_DDMO'] = contract._contract2_transitioned._consolidated_ddmo
            psc_table_consolidated_2['C_Taxable_Income'] = contract._contract2_transitioned._consolidated_taxable_income
            psc_table_consolidated_2['C_Tax_Due'] = contract._contract2_transitioned._consolidated_tax_due
            psc_table_consolidated_2[
                'C_Unpaid_Tax_Balance'] = contract._contract2_transitioned._consolidated_unpaid_tax_balance
            psc_table_consolidated_2['C_Tax_Payment'] = contract._contract2_transitioned._consolidated_tax_payment
            psc_table_consolidated_2['C_CTR_Net_Share'] = contract._contract2_transitioned._consolidated_ctr_net_share
            psc_table_consolidated_2[
                'C_Contractor_Take'] = contract._contract2_transitioned._consolidated_contractor_take
            psc_table_consolidated_2['C_Cashflow'] = contract._contract2_transitioned._consolidated_cashflow
            psc_table_consolidated_2['cum. C_Cashflow'] = np.cumsum(
                contract._contract2_transitioned._consolidated_cashflow)
            psc_table_consolidated_2[
                'C_Government_take'] = contract._contract2_transitioned._consolidated_government_take
            # psc_table_consolidated.loc['Column_Total'] = psc_table_consolidated.sum(numeric_only=True, axis=0)

        elif isinstance(contract.contract2, GrossSplit):
            psc_table_oil_2['Years'] = contract._contract2_transitioned.project_years
            psc_table_oil_2['Lifting'] = contract._contract2_transitioned._oil_lifting.get_lifting_rate_arr()
            psc_table_oil_2['Price'] = contract._contract2_transitioned._oil_wap_price
            psc_table_oil_2['Depreciable'] = contract._contract2_transitioned._oil_capital_expenditures_post_tax
            psc_table_oil_2['Intangible'] = contract._contract2_transitioned._oil_intangible_expenditures_post_tax
            psc_table_oil_2['Opex'] = contract._contract2_transitioned._oil_opex_expenditures_post_tax
            psc_table_oil_2['ASR'] = contract._contract2_transitioned._oil_asr_expenditures_post_tax
            psc_table_oil_2['Revenue'] = contract._contract2_transitioned._oil_revenue
            psc_table_oil_2['Base_Split'] = contract._contract2_transitioned._oil_base_split
            psc_table_oil_2['Variable_Split'] = contract._contract2_transitioned._var_split_array
            psc_table_oil_2['Progressive_Price_Split'] = contract._contract2_transitioned._oil_prog_price_split
            psc_table_oil_2['Progressive_Cumulative_Production_Split'] = contract._contract2_transitioned._oil_prog_cum_split
            psc_table_oil_2['Progressive_Split'] = contract._contract2_transitioned._oil_prog_split
            psc_table_oil_2['Contractor_Split'] = contract._contract2_transitioned._oil_ctr_split
            psc_table_oil_2['Contractor_Share'] = contract._contract2_transitioned._oil_ctr_share_before_transfer
            psc_table_oil_2['Government_Share'] = contract._contract2_transitioned._oil_gov_share
            psc_table_oil_2['Depreciation'] = contract._contract2_transitioned._oil_depreciation
            psc_table_oil_2['Non_Capital'] = contract._contract2_transitioned._oil_non_capital
            psc_table_oil_2['Total_Expenses'] = contract._contract2_transitioned._oil_total_expenses
            psc_table_oil_2['Cost_To_Be_Deducted'] = contract._contract2_transitioned._oil_cost_tobe_deducted
            psc_table_oil_2['Carry_Forward_Cost'] = contract._contract2_transitioned._oil_carward_deduct_cost
            psc_table_oil_2['Deductible_Cost'] = contract._contract2_transitioned._oil_deductible_cost
            psc_table_oil_2['Transfer_To_Gas'] = contract._contract2_transitioned._transfer_to_gas
            psc_table_oil_2['Carry_Forward_Cost_after_TF'] = contract._contract2_transitioned._oil_carward_cost_aftertf
            psc_table_oil_2['CTR_Share_After_TF'] = contract._contract2_transitioned._oil_ctr_share_after_transfer
            psc_table_oil_2['CTR_Net_Operating_Profit'] = contract._contract2_transitioned._oil_net_operating_profit
            psc_table_oil_2['DMO_Volume'] = contract._contract2_transitioned._oil_dmo_volume
            psc_table_oil_2['DMO_Fee'] = contract._contract2_transitioned._oil_dmo_fee
            psc_table_oil_2['DDMO'] = contract._contract2_transitioned._oil_ddmo
            psc_table_oil_2['Taxable_Income'] = contract._contract2_transitioned._oil_taxable_income
            psc_table_oil_2['Tax'] = contract._contract2_transitioned._oil_tax
            psc_table_oil_2['Net_CTR_Share'] = contract._contract2_transitioned._oil_ctr_net_share
            psc_table_oil_2['CTR_Cash_Flow'] = contract._contract2_transitioned._oil_ctr_cashflow
            psc_table_oil_2['Cum_Cash_Flow'] = np.cumsum(contract._contract2_transitioned._oil_ctr_cashflow)
            psc_table_oil_2['Government_Take'] = contract._contract2_transitioned._oil_government_take
            # psc_table_oil.loc['Column_Total'] = psc_table_oil.sum(numeric_only=True, axis=0)

            psc_table_gas_2['Years'] = contract.project_years
            psc_table_gas_2['Lifting'] = contract._contract2_transitioned._gas_lifting.get_lifting_rate_arr()
            psc_table_gas_2['Price'] = contract._contract2_transitioned._gas_wap_price
            psc_table_gas_2['Depreciable'] = contract._contract2_transitioned._gas_capital_expenditures_post_tax
            psc_table_gas_2['Intangible'] = contract._contract2_transitioned._gas_intangible_expenditures_post_tax
            psc_table_gas_2['Opex'] = contract._contract2_transitioned._gas_opex_expenditures_post_tax
            psc_table_gas_2['ASR'] = contract._contract2_transitioned._gas_asr_expenditures_post_tax
            psc_table_gas_2['Revenue'] = contract._contract2_transitioned._gas_revenue
            psc_table_gas_2['Base_Split'] = contract._contract2_transitioned._gas_base_split
            psc_table_gas_2['Variable_Split'] = contract._contract2_transitioned._var_split_array
            psc_table_gas_2['Progressive_Price_Split'] = contract._contract2_transitioned._gas_prog_price_split
            psc_table_gas_2['Progressive_Cumulative_Production_Split'] = contract._contract2_transitioned._gas_prog_cum_split
            psc_table_gas_2['Progressive_Split'] = contract._contract2_transitioned._gas_prog_split
            psc_table_gas_2['Contractor_Split'] = contract._contract2_transitioned._gas_ctr_split
            psc_table_gas_2['Contractor_Share'] = contract._contract2_transitioned._gas_ctr_share_before_transfer
            psc_table_gas_2['Government_Share'] = contract._contract2_transitioned._gas_gov_share
            psc_table_gas_2['Depreciation'] = contract._contract2_transitioned._gas_depreciation
            psc_table_gas_2['Non_Capital'] = contract._contract2_transitioned._gas_non_capital
            psc_table_gas_2['Total_Expenses'] = contract._contract2_transitioned._gas_total_expenses
            psc_table_gas_2['Cost_To_Be_Deducted'] = contract._contract2_transitioned._gas_cost_tobe_deducted
            psc_table_gas_2['Carry_Forward_Cost'] = contract._contract2_transitioned._gas_carward_deduct_cost
            psc_table_gas_2['Deductible_Cost'] = contract._contract2_transitioned._gas_deductible_cost
            psc_table_gas_2['Transfer_to_Oil'] = contract._contract2_transitioned._transfer_to_oil
            psc_table_gas_2['Carry_Forward_Cost_after_TF'] = contract._contract2_transitioned._gas_carward_cost_aftertf
            psc_table_gas_2['CTR_Share_After_TF'] = contract._contract2_transitioned._gas_ctr_share_after_transfer
            psc_table_gas_2['CTR_Net_Operating_Profit'] = contract._contract2_transitioned._gas_net_operating_profit
            psc_table_gas_2['DMO_Volume'] = contract._contract2_transitioned._gas_dmo_volume
            psc_table_gas_2['DMO_Fee'] = contract._contract2_transitioned._gas_dmo_fee
            psc_table_gas_2['DDMO'] = contract._contract2_transitioned._gas_ddmo
            psc_table_gas_2['Taxable_Income'] = contract._contract2_transitioned._gas_taxable_income
            psc_table_gas_2['Tax'] = contract._contract2_transitioned._gas_tax
            psc_table_gas_2['Net_CTR_Share'] = contract._contract2_transitioned._gas_ctr_net_share
            psc_table_gas_2['CTR_Cash_Flow'] = contract._contract2_transitioned._gas_ctr_cashflow
            psc_table_gas_2['Cum_Cash_Flow'] = np.cumsum(contract._contract2_transitioned._gas_ctr_cashflow)
            psc_table_gas_2['Government_Take'] = contract._contract2_transitioned._gas_government_take
            # psc_table_gas.loc['Column_Total'] = psc_table_gas.sum(numeric_only=True, axis=0)

            psc_table_consolidated_2['Years'] = contract._contract2_transitioned.project_years
            psc_table_consolidated_2[
                'C_Lifting_Oil'] = contract._contract2_transitioned._oil_lifting.get_lifting_rate_arr()
            psc_table_consolidated_2[
                'C_Lifting_Gas'] = contract._contract2_transitioned._gas_lifting.get_lifting_rate_arr()
            psc_table_consolidated_2['C_Revenue'] = contract._contract2_transitioned._consolidated_revenue
            psc_table_consolidated_2[
                'C_Government_Share'] = contract._contract2_transitioned._consolidated_ctr_share_before_tf
            psc_table_consolidated_2[
                'C_Contractor_Share'] = contract._contract2_transitioned._consolidated_gov_share_before_tf
            psc_table_consolidated_2['C_Depreciation'] = contract._contract2_transitioned._consolidated_depreciation
            psc_table_consolidated_2['C_Opex'] = contract._contract2_transitioned._consolidated_opex
            psc_table_consolidated_2['C_ASR'] = contract._contract2_transitioned._consolidated_asr
            psc_table_consolidated_2['C_Non_Capital'] = contract._contract2_transitioned._consolidated_non_capital
            psc_table_consolidated_2['C_Total_Expenses'] = contract._contract2_transitioned._consolidated_total_expenses
            psc_table_consolidated_2[
                'C_Cost_To_Be_Deducted'] = contract._contract2_transitioned._consolidated_cost_tobe_deducted
            psc_table_consolidated_2[
                'C_Carry_Forward_Cost'] = contract._contract2_transitioned._consolidated_carward_deduct_cost
            psc_table_consolidated_2[
                'C_Deductible_Cost'] = contract._contract2_transitioned._consolidated_deductible_cost
            psc_table_consolidated_2[
                'C_Carry_Forward_Cost_after_TF'] = contract._contract2_transitioned._consolidated_carward_cost_aftertf
            psc_table_consolidated_2[
                'C_CTR_Share_After'] = contract._contract2_transitioned._consolidated_ctr_share_after_transfer
            psc_table_consolidated_2[
                'C_CTR_Net_Operating_Profit'] = contract._contract2_transitioned._consolidated_net_operating_profit
            psc_table_consolidated_2['C_DMO_Volume'] = contract._contract2_transitioned._consolidated_dmo_volume
            psc_table_consolidated_2['C_DMO_Fee'] = contract._contract2_transitioned._consolidated_dmo_fee
            psc_table_consolidated_2['C_DDMO'] = contract._contract2_transitioned._consolidated_ddmo
            psc_table_consolidated_2['C_Taxable_Income'] = contract._contract2_transitioned._consolidated_taxable_income
            psc_table_consolidated_2['C_Tax'] = contract._contract2_transitioned._consolidated_tax_payment
            psc_table_consolidated_2['C_Net_CTR_Share'] = contract._contract2_transitioned._consolidated_ctr_net_share
            psc_table_consolidated_2['C_CashFlow'] = contract._contract2_transitioned._consolidated_cashflow
            psc_table_consolidated_2[
                'C_Government_Take'] = contract._contract2_transitioned._consolidated_government_take
            psc_table_consolidated_2['cum_C_CashFlow'] = np.cumsum(
                contract._contract2_transitioned._consolidated_cashflow)
            # psc_table_consolidated.loc['Column_Total'] = psc_table_consolidated.sum(numeric_only=True, axis=0)

        psc_table_oil = [psc_table_oil_1, psc_table_oil_2]
        psc_table_gas = [psc_table_gas_1, psc_table_gas_2]
        psc_table_consolidated = [psc_table_consolidated_1, psc_table_consolidated_2]

        return psc_table_oil, psc_table_gas, psc_table_consolidated

    elif isinstance(contract, BaseProject):
        psc_table_oil = pd.DataFrame()
        psc_table_oil['Years'] = contract.project_years
        psc_table_oil['Lifting'] = contract._oil_lifting.get_lifting_rate_arr()
        psc_table_oil['Price'] = contract._oil_wap_price
        psc_table_oil['Revenue'] = contract._oil_revenue
        psc_table_oil['Tangible'] = contract._oil_capital_expenditures_post_tax
        psc_table_oil['Intangible'] = contract._oil_intangible_expenditures_post_tax
        psc_table_oil['Opex'] = contract._oil_opex_expenditures_post_tax
        psc_table_oil['ASR'] = contract._oil_asr_expenditures_post_tax
        psc_table_oil['Cashflow'] = contract._oil_cashflow

        psc_table_gas = pd.DataFrame()
        psc_table_gas['Years'] = contract.project_years
        psc_table_gas['Lifting'] = contract._gas_lifting.get_lifting_rate_arr()
        psc_table_gas['Price'] = contract._gas_wap_price
        psc_table_gas['Revenue'] = contract._gas_revenue
        psc_table_gas['Tangible'] = contract._gas_capital_expenditures_post_tax
        psc_table_gas['Intangible'] = contract._gas_intangible_expenditures_post_tax
        psc_table_gas['Opex'] = contract._gas_opex_expenditures_post_tax
        psc_table_gas['ASR'] = contract._gas_asr_expenditures_post_tax
        psc_table_gas['Cashflow'] = contract._gas_cashflow

        psc_table_consolidated = pd.DataFrame()
        psc_table_consolidated['Years'] = contract.project_years
        psc_table_consolidated['Lifting'] = (contract._oil_lifting.get_lifting_rate_arr() +
                                             contract._gas_lifting.get_lifting_rate_arr())
        psc_table_consolidated['Price'] = (contract._oil_wap_price +
                                           contract._gas_wap_price)
        psc_table_consolidated['Revenue'] = (contract._oil_revenue +
                                             contract._gas_revenue)
        psc_table_consolidated['Tangible'] = (contract._oil_capital_expenditures_post_tax +
                                              contract._gas_capital_expenditures_post_tax)
        psc_table_consolidated['Intangible'] = (contract._oil_intangible_expenditures_post_tax +
                                                contract._gas_intangible_expenditures_post_tax)
        psc_table_consolidated['Opex'] = (contract._oil_opex_expenditures_post_tax +
                                          contract._gas_opex_expenditures_post_tax)
        psc_table_consolidated['ASR'] = (contract._oil_asr_expenditures_post_tax +
                                         contract._gas_asr_expenditures_post_tax)
        psc_table_consolidated['Cashflow'] = (contract._oil_cashflow +
                                              contract._gas_cashflow)

        return psc_table_oil, psc_table_gas, psc_table_consolidated
