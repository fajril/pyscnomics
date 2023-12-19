"""
Python Script to parse data from Spreadsheet
"""

from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts.transition import Transition

from pyscnomics.io.spreadsheet import Spreadsheet
from pyscnomics.io.aggregator import Aggregate


def initiate_contract(workbook_path: str):
    """
    The function to generate psc object, psc arguments and summary arguments.

    Parameters
    ----------
    workbook_path: str
        The directory path of the Excel file

    Returns
    -------
    out: tuple
    """
    # Generate PSC Object, using Spreadsheet modul and populating its attributes using .fit()
    data = Aggregate(workbook_to_read=workbook_path)
    data.fit()

    # Generate Summary Arguments
    summary_arguments = {'reference_year': data.general_config_data.discount_rate_start_year,
                         'inflation_rate': data.fiscal_config_data.inflation_rate,
                         'discount_rate': data.general_config_data.discount_rate,
                         'npv_mode': data.fiscal_config_data.npv_mode,
                         'discounting_mode': data.fiscal_config_data.discounting_mode, }

    psc = None
    psc_arguments = {}

    # Condition when the contract is Cost Recovery
    if data.general_config_data.type_of_contract == 'PSC Cost Recovery (CR)':
        psc = CostRecovery(start_date=data.general_config_data.start_date_project,
                           end_date=data.general_config_data.end_date_project,
                           oil_onstream_date=data.general_config_data.oil_onstream_date,
                           gas_onstream_date=data.general_config_data.gas_onstream_date,
                           lifting=data.oil_lifting_aggregate_total +
                                   data.gas_lifting_aggregate_total +
                                   data.co2_lifting_aggregate +
                                   data.sulfur_lifting_aggregate +
                                   data.electricity_lifting_aggregate,
                           tangible_cost=data.tangible_cost_aggregate,
                           intangible_cost=data.intangible_cost_aggregate,
                           opex=data.opex_aggregate,
                           asr_cost=data.asr_cost_aggregate,
                           oil_ftp_is_available=data.psc_cr_data.oil_ftp_availability,
                           oil_ftp_is_shared=data.psc_cr_data.oil_ftp_is_shared,
                           oil_ftp_portion=data.psc_cr_data.oil_ftp_portion,
                           gas_ftp_is_available=data.psc_cr_data.gas_ftp_availability,
                           gas_ftp_is_shared=data.psc_cr_data.gas_ftp_is_shared,
                           gas_ftp_portion=data.psc_cr_data.gas_ftp_portion,
                           tax_split_type=data.psc_cr_data.split_type,
                           condition_dict=data.psc_cr_data.icp_sliding_scale,
                           indicator_rc_icp_sliding=data.psc_cr_data.indicator_rc_split_sliding_scale,
                           oil_ctr_pretax_share=data.psc_cr_data.oil_ctr_pretax,
                           gas_ctr_pretax_share=data.psc_cr_data.gas_ctr_pretax,
                           oil_ic_rate=data.psc_cr_data.ic_oil,
                           gas_ic_rate=data.psc_cr_data.ic_gas,
                           ic_is_available=data.psc_cr_data.ic_availability,
                           oil_cr_cap_rate=data.psc_cr_data.oil_cr_cap_rate,
                           gas_cr_cap_rate=data.psc_cr_data.gas_cr_cap_rate,
                           oil_dmo_volume_portion=data.psc_cr_data.oil_dmo_volume,
                           oil_dmo_fee_portion=data.psc_cr_data.oil_dmo_fee,
                           oil_dmo_holiday_duration=data.psc_cr_data.oil_dmo_period,
                           gas_dmo_volume_portion=data.psc_cr_data.gas_dmo_volume,
                           gas_dmo_fee_portion=data.psc_cr_data.gas_dmo_fee,
                           gas_dmo_holiday_duration=data.psc_cr_data.gas_dmo_period, )

        psc_arguments = {'sulfur_revenue': data.fiscal_config_data.sulfur_revenue_config,
                         'electricity_revenue': data.fiscal_config_data.electricity_revenue_config,
                         'co2_revenue': data.fiscal_config_data.co2_revenue_config,
                         'is_dmo_end_weighted': data.psc_cr_data.dmo_is_weighted,
                         'tax_regime': data.fiscal_config_data.tax_mode,
                         'tax_rate': data.fiscal_config_data.tax_rate,
                         'ftp_tax_regime': data.fiscal_config_data.tax_payment_config,
                         'sunk_cost_reference_year': data.fiscal_config_data.sunk_cost_reference_year,
                         'depr_method': data.fiscal_config_data.depreciation_method,
                         'decline_factor': data.fiscal_config_data.decline_factor,
                         # 'year_ref': None,
                         # 'tax_type': None,
                         'vat_rate': data.fiscal_config_data.vat_rate,
                         'lbt_rate': data.fiscal_config_data.lbt_rate,
                         'inflation_rate': data.fiscal_config_data.inflation_rate,
                         'future_rate': float(data.fiscal_config_data.asr_future_rate),
                         'inflation_rate_applied_to': data.general_config_data.inflation_rate_applied_to, }

        # Filling the summary contract argument
        summary_arguments['contract'] = psc

        return psc, psc_arguments, summary_arguments

    # Condition when the contract is Gross Split
    elif data.general_config_data.type_of_contract == 'PSC Gross Split (GS)':
        psc = GrossSplit(start_date=data.general_config_data.start_date_project,
                         end_date=data.general_config_data.end_date_project,
                         oil_onstream_date=data.general_config_data.oil_onstream_date,
                         gas_onstream_date=data.general_config_data.gas_onstream_date,
                         lifting=data.oil_lifting_aggregate_total +
                                 data.gas_lifting_aggregate_total +
                                 data.co2_lifting_aggregate +
                                 data.sulfur_lifting_aggregate +
                                 data.electricity_lifting_aggregate,
                         tangible_cost=data.tangible_cost_aggregate,
                         intangible_cost=data.intangible_cost_aggregate,
                         opex=data.opex_aggregate,
                         asr_cost=data.asr_cost_aggregate,
                         field_status=data.psc_gs_data.field_status,
                         field_loc=data.psc_gs_data.field_location,
                         res_depth=data.psc_gs_data.reservoir_depth,
                         infra_avail=data.psc_gs_data.infrastructure_availability,
                         res_type=data.psc_gs_data.reservoir_type,
                         api_oil=data.psc_gs_data.oil_api,
                         domestic_use=data.psc_gs_data.domestic_content_use,
                         prod_stage=data.psc_gs_data.production_stage,
                         co2_content=data.psc_gs_data.co2_content,
                         h2s_content=data.psc_gs_data.h2s_content,
                         base_split_ctr_oil=data.psc_gs_data.oil_base_split,
                         base_split_ctr_gas=data.psc_gs_data.gas_base_split,
                         split_ministry_disc=data.psc_gs_data.ministry_discretion_split,
                         oil_dmo_volume_portion=data.psc_gs_data.oil_dmo_volume,
                         oil_dmo_fee_portion=data.psc_gs_data.oil_dmo_fee,
                         oil_dmo_holiday_duration=data.psc_gs_data.oil_dmo_period,
                         gas_dmo_volume_portion=data.psc_gs_data.gas_dmo_volume,
                         gas_dmo_fee_portion=data.psc_gs_data.gas_dmo_fee,
                         gas_dmo_holiday_duration=data.psc_gs_data.gas_dmo_period,
                         )

        psc_arguments = {'sulfur_revenue': data.fiscal_config_data.sulfur_revenue_config,
                         'electricity_revenue': data.fiscal_config_data.electricity_revenue_config,
                         'co2_revenue': data.fiscal_config_data.co2_revenue_config,
                         'is_dmo_end_weighted': data.psc_gs_data.dmo_is_weighted,
                         # 'regime': None,
                         'tax_regime': data.fiscal_config_data.tax_mode,
                         'tax_rate': data.fiscal_config_data.tax_rate,
                         'sunk_cost_reference_year': data.fiscal_config_data.sunk_cost_reference_year,
                         'depr_method': data.fiscal_config_data.depreciation_method,
                         'decline_factor': data.fiscal_config_data.decline_factor,
                         # 'year_ref': None,
                         # 'tax_type': None,
                         'vat_rate': data.fiscal_config_data.vat_rate,
                         'lbt_rate': data.fiscal_config_data.lbt_rate,
                         'inflation_rate': data.fiscal_config_data.inflation_rate,
                         'future_rate': float(data.fiscal_config_data.asr_future_rate),
                         'inflation_rate_applied_to': data.general_config_data.inflation_rate_applied_to, }

        # Filling the summary contract argument
        summary_arguments['contract'] = psc

        return psc, psc_arguments, summary_arguments

    # Condition when the contract is Transition CR - GS
    elif data.general_config_data.type_of_contract == 'Transition CR - GS':
        psc_cr = CostRecovery(start_date=data.general_config_data.start_date_project,
                              end_date=data.general_config_data.end_date_project,
                              oil_onstream_date=data.general_config_data.oil_onstream_date,
                              gas_onstream_date=data.general_config_data.gas_onstream_date,
                              lifting=data.oil_lifting_aggregate_total['PSC 1'] +
                                      data.gas_lifting_aggregate_total['PSC 1'] +
                                      data.co2_lifting_aggregate['PSC 1'] +
                                      data.sulfur_lifting_aggregate['PSC 1'] +
                                      data.electricity_lifting_aggregate['PSC 1'],
                              tangible_cost=data.tangible_cost_aggregate['PSC 1'],
                              intangible_cost=data.intangible_cost_aggregate['PSC 1'],
                              opex=data.opex_aggregate['PSC 1'],
                              asr_cost=data.asr_cost_aggregate['PSC 1'],
                              oil_ftp_is_available=data.psc_transition_cr_to_gs[
                                  'Cost Recovery Config'].oil_ftp_availability,
                              oil_ftp_is_shared=data.psc_transition_cr_to_gs['Cost Recovery Config'].oil_ftp_is_shared,
                              oil_ftp_portion=data.psc_transition_cr_to_gs['Cost Recovery Config'].oil_ftp_portion,
                              gas_ftp_is_available=data.psc_transition_cr_to_gs[
                                  'Cost Recovery Config'].gas_ftp_availability,
                              gas_ftp_is_shared=data.psc_transition_cr_to_gs['Cost Recovery Config'].gas_ftp_is_shared,
                              gas_ftp_portion=data.psc_transition_cr_to_gs['Cost Recovery Config'].gas_ftp_portion,
                              tax_split_type=data.psc_transition_cr_to_gs['Cost Recovery Config'].split_type,
                              condition_dict=data.psc_transition_cr_to_gs['Cost Recovery Config'].icp_sliding_scale,
                              indicator_rc_icp_sliding=data.psc_transition_cr_to_gs[
                                  'Cost Recovery Config'].indicator_rc_split_sliding_scale,
                              oil_ctr_pretax_share=data.psc_transition_cr_to_gs['Cost Recovery Config'].oil_ctr_pretax,
                              gas_ctr_pretax_share=data.psc_transition_cr_to_gs['Cost Recovery Config'].gas_ctr_pretax,
                              oil_ic_rate=data.psc_transition_cr_to_gs['Cost Recovery Config'].ic_oil,
                              gas_ic_rate=data.psc_transition_cr_to_gs['Cost Recovery Config'].ic_gas,
                              ic_is_available=data.psc_transition_cr_to_gs['Cost Recovery Config'].ic_availability,
                              oil_cr_cap_rate=data.psc_transition_cr_to_gs['Cost Recovery Config'].oil_cr_cap_rate,
                              gas_cr_cap_rate=data.psc_transition_cr_to_gs['Cost Recovery Config'].gas_cr_cap_rate,
                              oil_dmo_volume_portion=data.psc_transition_cr_to_gs[
                                  'Cost Recovery Config'].oil_dmo_volume,
                              oil_dmo_fee_portion=data.psc_transition_cr_to_gs['Cost Recovery Config'].oil_dmo_fee,
                              oil_dmo_holiday_duration=data.psc_transition_cr_to_gs[
                                  'Cost Recovery Config'].oil_dmo_period,
                              gas_dmo_volume_portion=data.psc_transition_cr_to_gs[
                                  'Cost Recovery Config'].gas_dmo_volume,
                              gas_dmo_fee_portion=data.psc_transition_cr_to_gs['Cost Recovery Config'].gas_dmo_fee,
                              gas_dmo_holiday_duration=data.psc_transition_cr_to_gs[
                                  'Cost Recovery Config'].gas_dmo_period, )

        psc_gs = GrossSplit(start_date=data.general_config_data.start_date_project_second,
                            end_date=data.general_config_data.end_date_project_second,
                            oil_onstream_date=data.general_config_data.oil_onstream_date_second,
                            gas_onstream_date=data.general_config_data.gas_onstream_date_second,
                            lifting=data.oil_lifting_aggregate_total['PSC 2'] +
                                    data.gas_lifting_aggregate_total['PSC 2'] +
                                    data.co2_lifting_aggregate['PSC 2'] +
                                    data.sulfur_lifting_aggregate['PSC 2'] +
                                    data.electricity_lifting_aggregate['PSC 2'],
                            tangible_cost=data.tangible_cost_aggregate['PSC 2'],
                            intangible_cost=data.intangible_cost_aggregate['PSC 2'],
                            opex=data.opex_aggregate['PSC 2'],
                            asr_cost=data.asr_cost_aggregate['PSC 2'],
                            field_status=data.psc_transition_cr_to_gs['Gross Split Config'].field_status,
                            field_loc=data.psc_transition_cr_to_gs['Gross Split Config'].field_location,
                            res_depth=data.psc_transition_cr_to_gs['Gross Split Config'].reservoir_depth,
                            infra_avail=data.psc_transition_cr_to_gs['Gross Split Config'].infrastructure_availability,
                            res_type=data.psc_transition_cr_to_gs['Gross Split Config'].reservoir_type,
                            api_oil=data.psc_transition_cr_to_gs['Gross Split Config'].oil_api,
                            domestic_use=data.psc_transition_cr_to_gs['Gross Split Config'].domestic_content_use,
                            prod_stage=data.psc_transition_cr_to_gs['Gross Split Config'].production_stage,
                            co2_content=data.psc_transition_cr_to_gs['Gross Split Config'].co2_content,
                            h2s_content=data.psc_transition_cr_to_gs['Gross Split Config'].h2s_content,
                            base_split_ctr_oil=data.psc_transition_cr_to_gs['Gross Split Config'].oil_base_split,
                            base_split_ctr_gas=data.psc_transition_cr_to_gs['Gross Split Config'].gas_base_split,
                            split_ministry_disc=data.psc_transition_cr_to_gs[
                                'Gross Split Config'].ministry_discretion_split,
                            oil_dmo_volume_portion=data.psc_transition_cr_to_gs['Gross Split Config'].oil_dmo_volume,
                            oil_dmo_fee_portion=data.psc_transition_cr_to_gs['Gross Split Config'].oil_dmo_fee,
                            oil_dmo_holiday_duration=data.psc_transition_cr_to_gs['Gross Split Config'].oil_dmo_period,
                            gas_dmo_volume_portion=data.psc_transition_cr_to_gs['Gross Split Config'].gas_dmo_volume,
                            gas_dmo_fee_portion=data.psc_transition_cr_to_gs['Gross Split Config'].gas_dmo_fee,
                            gas_dmo_holiday_duration=data.psc_transition_cr_to_gs['Gross Split Config'].gas_dmo_period,
                            )

        psc_cr_arguments = {'sulfur_revenue': data.fiscal_config_data.sulfur_revenue_config['PSC 1'],
                            'electricity_revenue': data.fiscal_config_data.electricity_revenue_config['PSC 1'],
                            'co2_revenue': data.fiscal_config_data.co2_revenue_config['PSC 1'],
                            'is_dmo_end_weighted': data.psc_cr_data.dmo_is_weighted,
                            'tax_regime': data.fiscal_config_data.tax_mode['PSC 1'],
                            'tax_rate': data.fiscal_config_data.tax_rate['PSC 1'],
                            'ftp_tax_regime': data.fiscal_config_data.tax_payment_config['PSC 1'],
                            'sunk_cost_reference_year': data.fiscal_config_data.sunk_cost_reference_year,
                            'depr_method': data.fiscal_config_data.depreciation_method['PSC 1'],
                            'decline_factor': data.fiscal_config_data.decline_factor['PSC 1'],
                            # 'year_ref': None,
                            # 'tax_type': None,
                            'vat_rate': data.fiscal_config_data.vat_rate['PSC 1'],
                            'lbt_rate': data.fiscal_config_data.lbt_rate['PSC 1'],
                            'inflation_rate': data.fiscal_config_data.inflation_rate['PSC 1'],
                            'future_rate': float(data.fiscal_config_data.asr_future_rate['PSC 1']),
                            'inflation_rate_applied_to': data.general_config_data.inflation_rate_applied_to, }

        psc_gs_arguments = {'sulfur_revenue': data.fiscal_config_data.sulfur_revenue_config['PSC 2'],
                            'electricity_revenue': data.fiscal_config_data.electricity_revenue_config['PSC 2'],
                            'co2_revenue': data.fiscal_config_data.co2_revenue_config['PSC 2'],
                            'is_dmo_end_weighted': data.psc_gs_data.dmo_is_weighted,
                            # 'regime': None,
                            'tax_regime': data.fiscal_config_data.tax_mode['PSC 2'],
                            'tax_rate': data.fiscal_config_data.tax_rate['PSC 2'],
                            'sunk_cost_reference_year': data.fiscal_config_data.sunk_cost_reference_year,
                            'depr_method': data.fiscal_config_data.depreciation_method['PSC 2'],
                            'decline_factor': data.fiscal_config_data.decline_factor['PSC 2'],
                            # 'year_ref': None,
                            # 'tax_type': None,
                            'vat_rate': data.fiscal_config_data.vat_rate['PSC 2'],
                            'lbt_rate': data.fiscal_config_data.lbt_rate['PSC 2'],
                            'inflation_rate': data.fiscal_config_data.inflation_rate['PSC 2'],
                            'future_rate': float(data.fiscal_config_data.asr_future_rate['PSC 2']),
                            'inflation_rate_applied_to': data.general_config_data.inflation_rate_applied_to, }

        # Defining Transition object
        psc = Transition(contract1=psc_cr,
                         contract2=psc_gs,
                         argument_contract1=psc_cr_arguments,
                         argument_contract2=psc_gs_arguments, )

        # Defining arguments for transition
        psc_arguments = {'unrec_portion': data.fiscal_config_data.transferred_unrec_cost}

        return psc, psc_arguments, summary_arguments
