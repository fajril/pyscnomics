"""
This file containing the tools which utilized by API adapter.
"""
from datetime import datetime, date
from typing import Dict

from pydantic import BaseModel
import numpy as np

from pyscnomics.econ.costs import Tangible, Intangible, OPEX, ASR
from pyscnomics.dataset.sample import assign_lifting, read_fluid_type
from pyscnomics.econ.selection import TaxRegime, TaxType, FTPTaxRegime
from pyscnomics.tools.helper import (get_inflation_applied_converter,
                                     get_npv_mode_converter,
                                     get_discounting_mode_converter,
                                     get_depreciation_method_converter,
                                     get_other_revenue_converter,
                                     get_split_type_converter,
                                     get_optimization_target_converter,
                                     get_optimization_parameter_converter)


class SetupBM(BaseModel):
    start_date: str = "01/01/2010"
    end_date: str = "31/12/2045"
    oil_onstream_date: str = "01/01/2023"
    gas_onstream_date: str = "01/01/2023"


class SummaryArgumentsBM(BaseModel):
    reference_year: int = 2022
    inflation_rate: float = 0.1
    discount_rate: float = 0.1
    npv_mode: str = "SKK Full Cycle Nominal Terms"
    discounting_mode: str = "End Year"


class CostRecoveryBM(BaseModel):
    oil_ftp_is_available: bool = True
    oil_ftp_is_shared: bool = True
    oil_ftp_portion: float = 0.2
    gas_ftp_is_available: bool = True
    gas_ftp_is_shared: bool = True
    gas_ftp_portion: float = 0.2
    tax_split_type: str = "Conventional"
    condition_dict: dict = {}
    indicator_rc_icp_sliding: list[float] = []
    oil_ctr_pretax_share: float = 0.34722220
    gas_ctr_pretax_share: float = 0.5208330
    oil_ic_rate: float = 0
    gas_ic_rate: float = 0
    ic_is_available: bool = False
    oil_cr_cap_rate: float = 1
    gas_cr_cap_rate: float = 1
    oil_dmo_volume_portion: float = 0.25
    oil_dmo_fee_portion: float = 0.25
    oil_dmo_holiday_duration: float = 60
    gas_dmo_volume_portion: float = 0.25
    gas_dmo_fee_portion: float = 1
    gas_dmo_holiday_duration: float = 60


class GrossSplitBM(BaseModel):
    field_status: str = "No POD"
    field_loc: str = "Onshore"
    res_depth: str = "<=2500"
    infra_avail: str = "Well Developed"
    res_type: str = "Conventional"
    api_oil: str = "<25"
    domestic_use: str = "50<=x<70"
    prod_stage: str = "Secondary"
    co2_content: str = "<5"
    h2s_content: str = "<100"
    base_split_ctr_oil: float = 0.43
    base_split_ctr_gas: float = 0.48
    split_ministry_disc: float = 0.08
    oil_dmo_volume_portion: float = 0.25
    oil_dmo_fee_portion: float = 1.0
    oil_dmo_holiday_duration: float = 60
    gas_dmo_volume_portion: float = 1.0
    gas_dmo_fee_portion: float = 1.0
    gas_dmo_holiday_duration: float = 60


class ContractArgumentsBM(BaseModel):
    sulfur_revenue: str = "Addition to Oil Revenue"
    electricity_revenue: str = "Addition to Oil Revenue"
    co2_revenue: str = "Addition to Oil Revenue"
    is_dmo_end_weighted: bool = False
    tax_regime: str = "nailed down"
    tax_rate: float | list | None = 0.424
    ftp_tax_regime: str = "Pre PDJP No.20 Tahun 2017"
    sunk_cost_reference_year: int = 2021
    depr_method: str = "PSC Declining Balance"
    decline_factor: float | int = 2
    vat_rate: list | float = 0.0
    lbt_rate: list | float = 0.0
    inflation_rate: list | float = 0.0
    future_rate: float = 0.02
    inflation_rate_applied_to: str = "CAPEX"


class ContractArgumentsTransitionBM(BaseModel):
    unrec_portion: float


class LiftingBM(BaseModel):
    start_year: int
    end_year: int
    lifting_rate: list[float]
    price: list[float]
    prod_year: list[int]
    fluid_type: str
    ghv: list[float] | None
    prod_rate: list[float] | None


class TangibleBM(BaseModel):
    start_year: int
    end_year: int
    cost: list[float]
    expense_year: list[int]
    cost_allocation: list[str]
    description: list[str]
    vat_portion: list[float]
    vat_discount: list[float]
    lbt_portion: list[float]
    lbt_discount: list[float]
    pis_year: list[int]
    salvage_value: list[float]
    useful_life: list[int]
    depreciation_factor: list[float]
    is_ic_applied: list[bool]


class IntangibleBM(BaseModel):
    start_year: int
    end_year: int
    cost: list[float]
    expense_year: list[int]
    cost_allocation: list[str]
    description: list[str]
    vat_portion: list[float]
    vat_discount: list[float]
    lbt_portion: list[float]
    lbt_discount: list[float]


class OpexBM(BaseModel):
    start_year: int
    end_year: int
    expense_year: list[int]
    cost_allocation: list[str]
    description: list[str]
    fixed_cost: list[float]
    prod_rate: list[float]
    cost_per_volume: list[float]
    vat_portion: list[float]
    vat_discount: list[float]
    lbt_portion: list[float]
    lbt_discount: list[float]


class AsrBM(BaseModel):
    start_year: int
    end_year: int
    cost: list[float]
    expense_year: list[int]
    cost_allocation: list[str]
    description: list[str]
    vat_portion: list[float]
    vat_discount: list[float]
    lbt_portion: list[float]
    lbt_discount: list[float]


class OptimizationDictBM(BaseModel):
    parameter: list[str]
    min: list[float]
    max: list[float]


class OptimizationBM(BaseModel):
    dict_optimization: OptimizationDictBM
    target_optimization: float
    target_parameter: str


class SensitivityBM(BaseModel):
    min: float
    max: float


class UncertaintyBM(BaseModel):
    number_of_simulation: int
    min: list[float]
    max: list[float]
    std_dev: list[float]


class Data(BaseModel):
    setup: SetupBM
    summary_arguments: SummaryArgumentsBM
    contract_arguments: ContractArgumentsBM
    lifting: Dict[str, LiftingBM]
    tangible: Dict[str, TangibleBM]
    intangible: Dict[str, IntangibleBM]
    opex: Dict[str, OpexBM]
    asr: Dict[str, AsrBM]
    optimization_arguments: OptimizationBM
    sensitivity_arguments: SensitivityBM
    uncertainty_arguments: UncertaintyBM
    costrecovery: CostRecoveryBM = None
    grosssplit: GrossSplitBM = None
    result: dict = None


class DataTransition(BaseModel):
    contract_1: CostRecoveryBM | GrossSplitBM
    contract_2: CostRecoveryBM | GrossSplitBM
    contract_arguments: ContractArgumentsTransitionBM
    summary_arguments: SummaryArgumentsBM
    result: dict = None


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
    attrs = {
        "PDJP No.20 Tahun 2017": FTPTaxRegime.PDJP_20_2017,
        "Pre PDJP No.20 Tahun 2017": FTPTaxRegime.PRE_PDJP_20_2017,
        "Direct": FTPTaxRegime.DIRECT_MODE
    }

    for key in attrs.keys():
        if str_object == key:
            return attrs[key]


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
