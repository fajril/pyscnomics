"""
This file containing the tools which utilized by API adapter.
"""
from datetime import datetime, date

import numpy as np

from pyscnomics.econ.costs import Tangible, Intangible, OPEX, ASR
from pyscnomics.dataset.sample import assign_lifting, read_fluid_type
from pyscnomics.econ.selection import TaxRegime, TaxType
from pyscnomics.tools.helper import (get_inflation_applied_converter,
                                     get_tax_regime_converter,
                                     get_npv_mode_converter,
                                     get_discounting_mode_converter,
                                     get_depreciation_method_converter,
                                     get_other_revenue_converter,
                                     get_split_type_converter,
                                     get_optimization_target_converter,
                                     get_optimization_parameter_converter)


def convert_str_to_date(str_object: str) -> date | None:
    if str_object is None:
        return None
    else:
        return datetime.strptime(str_object, '%d/%m/%Y').date()


def convert_list_to_array_float(data_list: list) -> np.ndarray:
    return np.array(data_list, dtype=float)


def convert_list_to_array_float_or_array(data_input: list | float) -> np.ndarray | float:
    if isinstance(data_input, float):
        return data_input

    else:
        return np.array(data_input, dtype=float)


def convert_list_to_array_float_or_array_or_none(data_list: list | float | None) -> np.ndarray | float | None:
    if isinstance(data_list, float):
        return data_list
    elif data_list is None:
        return None
    else:
        return np.array(data_list, dtype=float)


def convert_list_to_array_int(data_list: list) -> np.ndarray:
    return np.array(data_list, dtype=int)


def convert_dict_to_lifting(data_raw: dict) -> tuple:
    return assign_lifting(data_raw=data_raw)


def convert_dict_to_tangible(data_raw: dict) -> tuple:
    tangible_list = [
        Tangible(
            start_year=data_raw[key]['start_year'],
            end_year=data_raw[key]['end_year'],
            cost=np.array(data_raw[key]['cost']),
            expense_year=np.array(data_raw[key]['expense_year'], dtype=int),
            cost_allocation=read_fluid_type(fluid=data_raw[key]['cost_allocation']),
            description=data_raw[key]['description'],
            vat_portion=np.array(data_raw[key]['vat_portion']),
            vat_discount=np.array(data_raw[key]['vat_discount']),
            lbt_portion=np.array(data_raw[key]['lbt_portion']),
            lbt_discount=np.array(data_raw[key]['lbt_discount']),
            pis_year=np.array(data_raw[key]['pis_year']),
            salvage_value=np.array(data_raw[key]['salvage_value']),
            useful_life=np.array(data_raw[key]['useful_life']),
            depreciation_factor=np.array(data_raw[key]['depreciation_factor']),
            is_ic_applied=data_raw[key]['is_ic_applied'],
        )
        for key in data_raw.keys()
    ]

    return tuple(tangible_list)


def convert_dict_to_intangible(data_raw: dict) -> tuple:
    intangible_list = [
        Intangible(
            start_year=data_raw[key]['start_year'],
            end_year=data_raw[key]['end_year'],
            cost=np.array(data_raw[key]['cost'], dtype=float),
            expense_year=np.array(data_raw[key]['expense_year'], dtype=int),
            cost_allocation=read_fluid_type(fluid=data_raw[key]['cost_allocation']),
            description=data_raw[key]['description'],
            vat_portion=np.array(data_raw[key]['vat_portion'], dtype=float),
            vat_discount=np.array(data_raw[key]['vat_discount'], dtype=float),
            lbt_portion=np.array(data_raw[key]['lbt_portion'], dtype=float),
            lbt_discount=np.array(data_raw[key]['lbt_discount'], dtype=float),
        )
        for key in data_raw.keys()
    ]

    return tuple(intangible_list)


def convert_dict_to_opex(data_raw: dict) -> tuple:
    opex_list = [
        OPEX(
            start_year=data_raw[key]['start_year'],
            end_year=data_raw[key]['end_year'],
            expense_year=np.array(data_raw[key]['expense_year'], dtype=int),
            cost_allocation=read_fluid_type(fluid=data_raw[key]['cost_allocation']),
            description=data_raw[key]['description'],
            vat_portion=np.array(data_raw[key]['vat_portion'], dtype=float),
            vat_discount=np.array(data_raw[key]['vat_discount'], dtype=float),
            lbt_portion=np.array(data_raw[key]['lbt_portion'], dtype=float),
            lbt_discount=np.array(data_raw[key]['lbt_discount'], dtype=float),
            fixed_cost=np.array(data_raw[key]['fixed_cost'], dtype=float),
            prod_rate=np.array(data_raw[key]['prod_rate'], dtype=float),
            cost_per_volume=np.array(data_raw[key]['cost_per_volume'], dtype=float),
        )
        for key in data_raw.keys()
    ]

    return tuple(opex_list)


def convert_dict_to_asr(data_raw: dict) -> tuple:
    asr_list = [
        ASR(
            start_year=data_raw[key]['start_year'],
            end_year=data_raw[key]['end_year'],
            cost=np.array(data_raw[key]['cost'],dtype=float),
            expense_year=np.array(data_raw[key]['expense_year'], dtype=int),
            cost_allocation=read_fluid_type(fluid=data_raw[key]['cost_allocation']),
            description=data_raw[key]['description'],
            vat_portion=np.array(data_raw[key]['vat_portion'], dtype=float),
            vat_discount=np.array(data_raw[key]['vat_discount'],dtype=float),
            lbt_portion=np.array(data_raw[key]['lbt_portion'], dtype=float),
            lbt_discount=np.array(data_raw[key]['lbt_discount'], dtype=float),
        )
        for key in data_raw.keys()
    ]

    return tuple(asr_list)


def convert_str_to_taxsplit(str_object: str):
    return get_split_type_converter(target=str_object)


def convert_str_to_npvmode(str_object: str):
    return get_npv_mode_converter(target=str_object)


def convert_str_to_discountingmode(str_object: str):
    return get_discounting_mode_converter(target=str_object)


def convert_str_to_otherrevenue(str_object: str):
    return get_other_revenue_converter(target=str_object)


def convert_str_to_taxregime(str_object: str):
    dict_tax_regime = {i.value: i for i in TaxRegime}

    for key in dict_tax_regime.keys():
        if str_object == key:
            return dict_tax_regime[key]
        else:
            return None


def convert_str_to_ftptaxregime(str_object: str):
    return get_tax_regime_converter(target=str_object)


def convert_str_to_depremethod(str_object: str):
    return get_depreciation_method_converter(target=str_object)


def convert_str_to_taxtype(str_object: str):
    dict_tax_regime = {i.value: i for i in TaxType}

    for key in dict_tax_regime.keys():
        if str_object == key:
            return dict_tax_regime[key]
        else:
            return None


def convert_str_to_inflationappliedto(str_object: str):
    return get_inflation_applied_converter(target=str_object)


def convert_summary_to_dict(dict_object: dict):
    summary_skk_format = {
        'lifting_oil': dict_object['lifting_oil'],
        'oil_wap': dict_object['oil_wap'],
        'lifting_gas': dict_object['lifting_gas'],
        'gas_wap': dict_object['gas_wap'],
        'gross_revenue': dict_object['gross_revenue'],
        'ctr_gross_share': dict_object['ctr_gross_share'],
        'sunk_cost': dict_object['sunk_cost'],
        'investment': dict_object['investment'],
        'tangible': dict_object['tangible'],
        'intangible': dict_object['intangible'],
        'opex_and_asr': dict_object['opex_and_asr'],
        'opex': dict_object['opex'],
        'asr': dict_object['asr'],
        'cost_recovery/deductible_cost': dict_object['cost_recovery / deductible_cost'],
        'cost_recovery_over_gross_rev': dict_object['cost_recovery_over_gross_rev'],
        'unrec_cost': dict_object['unrec_cost'],
        'unrec_over_gross_rev': dict_object['unrec_over_gross_rev'],
        'ctr_net_share': dict_object['ctr_net_share'],
        'ctr_net_share_over_gross_share': dict_object['ctr_net_share_over_gross_share'],
        'ctr_net_cashflow': dict_object['ctr_net_cashflow'],
        'ctr_net_cashflow_over_gross_rev': dict_object['ctr_net_cashflow_over_gross_rev'],
        'ctr_npv': dict_object['ctr_npv'],
        'ctr_irr': dict_object['ctr_irr'],
        'ctr_pot': dict_object['ctr_pot'],
        'ctr_pv_ratio': dict_object['ctr_pv_ratio'],
        'ctr_pi': dict_object['ctr_pi'],
        'gov_gross_share': dict_object['gov_gross_share'],
        'gov_ftp_share': dict_object['gov_ftp_share'],
        'gov_ddmo': dict_object['gov_ddmo'],
        'gov_tax_income': dict_object['gov_tax_income'],
        'gov_take': dict_object['gov_take'],
        'gov_take_over_gross_rev': dict_object['gov_take_over_gross_rev'],
        'gov_take_npv': dict_object['gov_take_npv'],
    }
    return summary_skk_format


def convert_str_to_optimization_targetparameter(str_object: str):
    return get_optimization_target_converter(target=str_object)


def convert_str_to_optimization_parameters(str_object: str):
    return get_optimization_parameter_converter(target=str_object)
