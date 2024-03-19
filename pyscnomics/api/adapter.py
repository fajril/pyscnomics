"""
This file is utilized to be the adapter of the router into the core codes.
"""

from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.tools.summary import get_summary
from pyscnomics.api.converter import (convert_str_to_date,
                                      convert_list_to_array_float_or_array,
                                      convert_dict_to_lifting,
                                      convert_dict_to_tangible,
                                      convert_dict_to_intangible,
                                      convert_dict_to_opex,
                                      convert_dict_to_asr,
                                      convert_list_to_array_float,
                                      convert_list_to_array_float_or_array_or_none,
                                      convert_str_to_taxsplit,
                                      convert_str_to_npvmode,
                                      convert_str_to_discountingmode,
                                      convert_str_to_otherrevenue,
                                      convert_str_to_taxregime,
                                      convert_str_to_ftptaxregime,
                                      convert_str_to_depremethod,
                                      convert_str_to_inflationappliedto)


def get_contract_costrecovery(data: dict):
    # Filling the attributes of the contract with the input data
    start_date = convert_str_to_date(str_object=data['setup']['start_date'])
    end_date = convert_str_to_date(str_object=data['setup']['end_date'])
    oil_onstream_date = convert_str_to_date(str_object=data['setup']['oil_onstream_date'])
    gas_onstream_date = convert_str_to_date(str_object=data['setup']['gas_onstream_date'])
    lifting = convert_dict_to_lifting(data_raw=data)
    tangible = convert_dict_to_tangible(data_raw=data['tangible'])
    intangible = convert_dict_to_intangible(data_raw=data['intangible'])
    opex = convert_dict_to_opex(data_raw=data['opex'])
    asr = convert_dict_to_asr(data_raw=data['asr'])
    oil_ftp_is_available = data['costrecovery']['oil_ftp_is_available']
    oil_ftp_is_shared = data['costrecovery']['oil_ftp_is_shared']
    oil_ftp_portion = data['costrecovery']['oil_ftp_portion']
    gas_ftp_is_available = data['costrecovery']['gas_ftp_is_available']
    gas_ftp_is_shared = data['costrecovery']['gas_ftp_is_shared']
    gas_ftp_portion = data['costrecovery']['gas_ftp_portion']
    tax_split_type = convert_str_to_taxsplit(str_object=data['costrecovery']['tax_split_type'])
    condition_dict = data['costrecovery']['condition_dict']
    indicator_rc_icp_sliding = convert_list_to_array_float(data_list=data['costrecovery']['indicator_rc_icp_sliding'])
    oil_ctr_pretax_share = data['costrecovery']['oil_ctr_pretax_share']
    gas_ctr_pretax_share = data['costrecovery']['gas_ctr_pretax_share']
    oil_ic_rate = data['costrecovery']['oil_ic_rate']
    gas_ic_rate = data['costrecovery']['gas_ic_rate']
    ic_is_available = data['costrecovery']['ic_is_available']
    oil_cr_cap_rate = data['costrecovery']['oil_cr_cap_rate']
    gas_cr_cap_rate = data['costrecovery']['gas_cr_cap_rate']
    oil_dmo_volume_portion = data['costrecovery']['oil_dmo_volume_portion']
    oil_dmo_fee_portion = data['costrecovery']['oil_dmo_fee_portion']
    oil_dmo_holiday_duration = data['costrecovery']['oil_dmo_holiday_duration']
    gas_dmo_volume_portion = data['costrecovery']['gas_dmo_volume_portion']
    gas_dmo_fee_portion = data['costrecovery']['gas_dmo_fee_portion']
    gas_dmo_holiday_duration = data['costrecovery']['gas_dmo_holiday_duration']

    contract = CostRecovery(start_date=start_date,
                            end_date=end_date,
                            oil_onstream_date=oil_onstream_date,
                            gas_onstream_date=gas_onstream_date,
                            lifting=lifting,
                            tangible_cost=tangible,
                            intangible_cost=intangible,
                            opex=opex,
                            asr_cost=asr,
                            oil_ftp_is_available=oil_ftp_is_available,
                            oil_ftp_is_shared=oil_ftp_is_shared,
                            oil_ftp_portion=oil_ftp_portion,
                            gas_ftp_is_available=gas_ftp_is_available,
                            gas_ftp_is_shared=gas_ftp_is_shared,
                            gas_ftp_portion=gas_ftp_portion,
                            tax_split_type=tax_split_type,
                            condition_dict=condition_dict,
                            indicator_rc_icp_sliding=indicator_rc_icp_sliding,
                            oil_ctr_pretax_share=oil_ctr_pretax_share,
                            gas_ctr_pretax_share=gas_ctr_pretax_share,
                            oil_ic_rate=oil_ic_rate,
                            gas_ic_rate=gas_ic_rate,
                            ic_is_available=ic_is_available,
                            oil_cr_cap_rate=oil_cr_cap_rate,
                            gas_cr_cap_rate=gas_cr_cap_rate,
                            oil_dmo_volume_portion=oil_dmo_volume_portion,
                            oil_dmo_fee_portion=oil_dmo_fee_portion,
                            oil_dmo_holiday_duration=oil_dmo_holiday_duration,
                            gas_dmo_volume_portion=gas_dmo_volume_portion,
                            gas_dmo_fee_portion=gas_dmo_fee_portion,
                            gas_dmo_holiday_duration=gas_dmo_holiday_duration,)

    # Filling the arguments of the contract with the data input
    sulfur_revenue = convert_str_to_otherrevenue(str_object=data['contract_setup']['sulfur_revenue'])
    electricity_revenue = convert_str_to_otherrevenue(str_object=data['contract_setup']['electricity_revenue'])
    co2_revenue = convert_str_to_otherrevenue(str_object=data['contract_setup']['co2_revenue'])
    is_dmo_end_weighted = data['contract_setup']['is_dmo_end_weighted']
    tax_regime = convert_str_to_taxregime(str_object=data['contract_setup']['tax_regime'])
    tax_rate = convert_list_to_array_float_or_array_or_none(data_list=data['contract_setup']['tax_rate'])
    ftp_tax_regime = convert_str_to_ftptaxregime(str_object=data['contract_setup']['ftp_tax_regime'])
    sunk_cost_reference_year = data['contract_setup']['sunk_cost_reference_year']
    depr_method = convert_str_to_depremethod(str_object=data['contract_setup']['depr_method'])
    decline_factor = data['contract_setup']['decline_factor']
    vat_rate = convert_list_to_array_float_or_array(data_input=data['contract_setup']['vat_rate'])
    lbt_rate = convert_list_to_array_float_or_array(data_input=data['contract_setup']['lbt_rate'])
    inflation_rate = convert_list_to_array_float_or_array(data_input=data['contract_setup']['inflation_rate'])
    future_rate = data['contract_setup']['future_rate']
    inflation_rate_applied_to = convert_str_to_inflationappliedto(
        str_object=data['contract_setup']['inflation_rate_applied_to'])

    arguments_dict = {
        "sulfur_revenue": sulfur_revenue,
        "electricity_revenue": electricity_revenue,
        "co2_revenue": co2_revenue,
        "is_dmo_end_weighted": is_dmo_end_weighted,
        "tax_regime": tax_regime,
        "tax_rate": tax_rate,
        "ftp_tax_regime": ftp_tax_regime,
        "sunk_cost_reference_year": sunk_cost_reference_year,
        "depr_method": depr_method,
        "decline_factor": decline_factor,
        "vat_rate": vat_rate,
        "lbt_rate": lbt_rate,
        "inflation_rate": inflation_rate,
        "future_rate": future_rate,
        "inflation_rate_applied_to": inflation_rate_applied_to,
    }

    # Running the contract
    contract.run(**arguments_dict)

    # Filling the argument with the input data
    reference_year = data['summary_setup']['reference_year']
    inflation_rate = data['summary_setup']['inflation_rate']
    discount_rate = data['summary_setup']['discount_rate']
    npv_mode = convert_str_to_npvmode(str_object=data['summary_setup']['npv_mode'])
    discounting_mode = convert_str_to_discountingmode(str_object=data['summary_setup']['discounting_mode'])

    summary = get_summary(contract=contract,
                          reference_year=reference_year,
                          inflation_rate=inflation_rate,
                          discount_rate=discount_rate,
                          npv_mode=npv_mode,
                          discounting_mode=discounting_mode,)

    return summary

