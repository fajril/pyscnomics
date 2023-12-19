"""
Python Script to parse data from Spreadsheet
"""

from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts.transition import Transition


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
    # Generate PSC Object, using Spreadsheet modul
    data = NotImplemented

    # Generate Summary Arguments
    summary_arguments = {'reference_year': None,
                         'inflation_rate': None,
                         'discount_rate': None,
                         'npv_mode': None,
                         'discounting_mode': None, }

    psc = None
    psc_arguments = {}

    # Condition when the contract is Cost Recovery
    if data[None] == 'PSC Cost Recovery (CR)':
        psc = CostRecovery(start_date=None,
                           end_date=None,
                           oil_onstream_date=None,
                           gas_onstream_date=None,
                           lifting=None,
                           tangible_cost=None,
                           intangible_cost=None,
                           opex=None,
                           asr_cost=None,
                           oil_ftp_is_available=None,
                           oil_ftp_is_shared=None,
                           oil_ftp_portion=None,
                           gas_ftp_is_available=None,
                           gas_ftp_is_shared=None,
                           gas_ftp_portion=None,
                           tax_split_type=None,
                           condition_dict=None,
                           indicator_rc_icp_sliding=None,
                           oil_ctr_pretax_share=None,
                           gas_ctr_pretax_share=None,
                           oil_ic_rate=None,
                           gas_ic_rate=None,
                           ic_is_available=None,
                           oil_cr_cap_rate=None,
                           gas_cr_cap_rate=None,
                           oil_dmo_volume_portion=None,
                           oil_dmo_fee_portion=None,
                           oil_dmo_holiday_duration=None,
                           gas_dmo_volume_portion=None,
                           gas_dmo_fee_portion=None,
                           gas_dmo_holiday_duration=None, )

        psc_arguments = {'sulfur_revenue': None,
                         'electricity_revenue': None,
                         'co2_revenue': None,
                         'is_dmo_end_weighted': None,
                         'tax_regime': None,
                         'tax_rate': None,
                         'ftp_tax_regime': None,
                         'sunk_cost_reference_year': None,
                         'depr_method': None,
                         'decline_factor': None,
                         'year_ref': None,
                         'tax_type': None,
                         'vat_rate': None,
                         'lbt_rate': None,
                         'inflation_rate': None,
                         'future_rate': None,
                         'inflation_rate_applied_to': None, }

        # Filling the summary contract argument
        summary_arguments['contract'] = psc

    # Condition when the contract is Gross Split
    elif data[None] == 'PSC Gross Split (GS)':
        psc = GrossSplit(start_date=None,
                         end_date=None,
                         oil_onstream_date=None,
                         gas_onstream_date=None,
                         lifting=None,
                         tangible_cost=None,
                         intangible_cost=None,
                         opex=None,
                         asr_cost=None,
                         field_status=None,
                         field_loc=None,
                         res_depth=None,
                         infra_avail=None,
                         res_type=None,
                         api_oil=None,
                         domestic_use=None,
                         prod_stage=None,
                         co2_content=None,
                         h2s_content=None,
                         base_split_ctr_oil=None,
                         base_split_ctr_gas=None,
                         split_ministry_disc=None,
                         oil_dmo_volume_portion=None,
                         oil_dmo_fee_portion=None,
                         oil_dmo_holiday_duration=None,
                         gas_dmo_volume_portion=None,
                         gas_dmo_fee_portion=None,
                         gas_dmo_holiday_duration=None,
                         )
        psc_arguments = {}

        # Filling the summary contract argument
        summary_arguments['contract'] = psc

        # Condition when the contract is Gross Split
    elif data[None] == 'Transition CR - CR':
        psc = GrossSplit(
            start_date=None,
            end_date=None,
            oil_onstream_date=None,
            gas_onstream_date=None,
            lifting=None,
            tangible_cost=None,
            intangible_cost=None,
            opex=None,
            asr_cost=None,
            field_status=None,
            field_loc=None,
            res_depth=None,
            infra_avail=None,
            res_type=None,
            api_oil=None,
            domestic_use=None,
            prod_stage=None,
            co2_content=None,
            h2s_content=None,
            base_split_ctr_oil=None,
            base_split_ctr_gas=None,
            split_ministry_disc=None,
            oil_dmo_volume_portion=None,
            oil_dmo_fee_portion=None,
            oil_dmo_holiday_duration=None,
            gas_dmo_volume_portion=None,
            gas_dmo_fee_portion=None,
            gas_dmo_holiday_duration=None,
        )
        psc_arguments = {'sulfur_revenue': None,
                         'electricity_revenue': None,
                         'co2_revenue': None,
                         'is_dmo_end_weighted': None,
                         'regime': None,
                         'tax_regime': None,
                         'tax_rate': None,
                         'sunk_cost_reference_year': None,
                         'depr_method': None,
                         'decline_factor': None,
                         'year_ref': None,
                         'tax_type': None,
                         'vat_rate': None,
                         'lbt_rate': None,
                         'inflation_rate': None,
                         'future_rate': None,
                         'inflation_rate_applied_to': None, }

        # Filling the summary contract argument
        summary_arguments['contract'] = psc

    return psc, psc_arguments, summary_arguments
