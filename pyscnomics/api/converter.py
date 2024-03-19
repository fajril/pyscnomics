"""
This file containing the tools which utilized by API adapter.
"""
from datetime import datetime, date

import numpy as np

from pyscnomics.econ.costs import Tangible, Intangible, OPEX, ASR
from pyscnomics.dataset.sample import assign_lifting, read_fluid_type


def convert_str_to_date(str_object: str) -> date:
    return datetime.strptime(str_object, '%Y/%m/%d').date()


def convert_dict_to_lifting(data_raw: dict) -> tuple:
    return assign_lifting(data_raw=data_raw)


def convert_dict_to_tangible(data_raw: dict) -> tuple:
    tangible_list = [
        Tangible(
            start_year=data_raw[key]['start_year'],
            end_year=data_raw[key]['end_year'],
            cost=np.array(data_raw[key]['cost']),
            expense_year=np.array(data_raw[key]['expense_year']),
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
            cost=data_raw[key]['cost'],
            expense_year=data_raw[key]['expense_year'],
            cost_allocation=data_raw[key]['cost_allocation'],
            description=data_raw[key]['description'],
            vat_portion=data_raw[key]['vat_portion'],
            vat_discount=data_raw[key]['vat_discount'],
            lbt_portion=data_raw[key]['lbt_portion'],
            lbt_discount=data_raw[key]['lbt_discount'],
        )
        for key in data_raw.keys()
    ]

    return tuple(intangible_list)


def convert_dict_to_opex(data_raw: dict) -> tuple:
    opex_list = [
        OPEX(
            start_year=data_raw[key]['start_year'],
            end_year=data_raw[key]['end_year'],
            expense_year=data_raw[key]['expense_year'],
            cost_allocation=data_raw[key]['cost_allocation'],
            description=data_raw[key]['description'],
            vat_portion=data_raw[key]['vat_portion'],
            vat_discount=data_raw[key]['vat_discount'],
            lbt_portion=data_raw[key]['lbt_portion'],
            lbt_discount=data_raw[key]['lbt_discount'],
            fixed_cost=data_raw[key]['fixed_cost'],
            prod_rate=data_raw[key]['prod_rate'],
            cost_per_volume=data_raw[key]['cost_per_volume'],
        )
        for key in data_raw.keys()
    ]

    return tuple(opex_list)


def convert_dict_to_asr(data_raw: dict) -> tuple:
    asr_list = [
        ASR(
            start_year=data_raw[key]['start_year'],
            end_year=data_raw[key]['end_year'],
            cost=data_raw[key]['cost'],
            expense_year=data_raw[key]['expense_year'],
            cost_allocation=data_raw[key]['cost_allocation'],
            description=data_raw[key]['description'],
            vat_portion=data_raw[key]['vat_portion'],
            vat_discount=data_raw[key]['vat_discount'],
            lbt_portion=data_raw[key]['lbt_portion'],
            lbt_discount=data_raw[key]['lbt_discount'],
        )
        for key in data_raw.keys()
    ]

    return tuple(asr_list)


def convert_list_to_array_float(data_list: list) -> np.ndarray:
    return np.array(data_list, dtype=float)


def convert_list_to_array_int(data_list: list) -> np.ndarray:
    return np.array(data_list, dtype=int)







