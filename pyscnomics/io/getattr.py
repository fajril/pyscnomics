"""
Handles the routine to get the attributes of a contract object.
"""
import numpy as np

from datetime import date
from enum import Enum

from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts.transition import Transition
from pyscnomics.econ import TaxSplitTypeCR, NPVSelection, DiscountingMode, OtherRevenue, DeprMethod, FTPTaxRegime, \
    TaxRegime, InflationAppliedTo, GrossSplitRegime
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.costs import CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales
from pyscnomics.econ.selection import FluidType, VariableSplit082017, VariableSplit522017


class GetAttrException(Exception):
    """Exception to raise for a misuse of Get Attributes Method"""

    pass


def convert_enum_fluid(objects: FluidType):
    if objects is FluidType.OIL:
        result = 'Oil'
    elif objects is FluidType.GAS:
        result = 'Gas'
    elif objects is FluidType.SULFUR:
        result = 'Sulfur'
    elif objects is FluidType.ELECTRICITY:
        result = 'Electricity'
    elif objects is FluidType.CO2:
        result = 'CO2'
    else:
        raise GetAttrException(
            f"{objects} is not recognized"
        )

    return result

def convert_enum_taxsplit(objects: TaxSplitTypeCR):
    if objects is TaxSplitTypeCR.CONVENTIONAL:
        result = 'Conventional'
    elif objects is TaxSplitTypeCR.SLIDING_SCALE:
        result = 'ICP Sliding Scale'
    elif objects is TaxSplitTypeCR.R2C:
        result = 'R/C'
    else:
        raise GetAttrException(
            f"{objects} is not recognized"
        )
    return result

def convert_enum_npv(objects: NPVSelection):
    attrs = {
        NPVSelection.NPV_SKK_REAL_TERMS:"SKK Full Cycle Real Terms",
        NPVSelection.NPV_SKK_NOMINAL_TERMS:"SKK Full Cycle Nominal Terms",
        NPVSelection.NPV_REAL_TERMS:"Full Cycle Real Terms",
        NPVSelection.NPV_NOMINAL_TERMS:"Full Cycle Nominal Terms",
        NPVSelection.NPV_POINT_FORWARD:"Point Forward",
    }

    for key in attrs.keys():
        if objects == key:
            return attrs[key]

def convert_enum_discountingmode(objects: DiscountingMode):
    attrs = {
        DiscountingMode.END_YEAR: "End Year",
        DiscountingMode.MID_YEAR: "Mid Year",
    }

    for key in attrs.keys():
        if objects == key:
            return attrs[key]

def convert_enum_otherrevenue(objects: OtherRevenue):
    attrs = {
        OtherRevenue.ADDITION_TO_OIL_REVENUE: "Addition to Oil Revenue",
        OtherRevenue.ADDITION_TO_GAS_REVENUE: "Addition to Gas Revenue",
        OtherRevenue.REDUCTION_TO_OIL_OPEX: "Reduction to Oil OPEX",
        OtherRevenue.REDUCTION_TO_GAS_OPEX: "Reduction to Gas OPEX",
    }

    for key in attrs.keys():
        if objects == key:
            return attrs[key]

def convert_enum_depreciationmethod(objects: DeprMethod):
    attrs = {
        DeprMethod.PSC_DB: "PSC Declining Balance",
        DeprMethod.DB: "Declining Balance",
        DeprMethod.UOP: "Unit Of Production",
        DeprMethod.SL: "Straight Line",
    }

    for key in attrs.keys():
        if objects == key:
            return attrs[key]

def convert_enum_ftptaxregime(objects: FTPTaxRegime):
    attrs = {
        FTPTaxRegime.PDJP_20_2017: "PDJP No.20 Tahun 2017",
        FTPTaxRegime.PRE_PDJP_20_2017: "Pre PDJP No.20 Tahun 2017",
    }

    for key in attrs.keys():
        if objects == key:
            return attrs[key]

def convert_enum_taxregime(objects: TaxRegime):
    attrs = {
        TaxRegime.NAILED_DOWN: "nailed down",
        TaxRegime.PREVAILING: "prevailing",
        TaxRegime.UU_36_2008: "UU No.36 Tahun 2008",
        TaxRegime.UU_02_2020: "UU No.02 Tahun 2020",
        TaxRegime.UU_07_2021: "UU No.07 Tahun 2021",
    }

    for key in attrs.keys():
        if objects == key:
            return attrs[key]

def convert_enum_inflationappliedto(objects: InflationAppliedTo):
    attrs = {
        InflationAppliedTo.CAPEX: "CAPEX",
        InflationAppliedTo.OPEX: "OPEX",
        InflationAppliedTo.CAPEX_AND_OPEX: "CAPEX AND OPEX",
    }

    for key in attrs.keys():
        if objects == key:
            return attrs[key]

def convert_enum_gsregime(objects: GrossSplitRegime):
    attrs = {
        GrossSplitRegime.PERMEN_ESDM_8_2017: "PERMEN_ESDM_8_2017",
        GrossSplitRegime.PERMEN_ESDM_52_2017: "PERMEN_ESDM_52_2017",
        GrossSplitRegime.PERMEN_ESDM_20_2019: "PERMEN_ESDM_20_2019",
        GrossSplitRegime.PERMEN_ESDM_12_2020: "PERMEN_ESDM_12_2020",
    }

    for key in attrs.keys():
        if objects == key:
            return attrs[key]

def convert_object(objects):
    if isinstance(objects, date):
        return objects.strftime('%Y-%m-%d')

    elif isinstance(objects, np.ndarray):
        return objects.tolist()

    elif isinstance(objects, FluidType):
        return convert_enum_fluid(objects=objects)

    elif isinstance(objects, TaxSplitTypeCR):
        return convert_enum_taxsplit(objects=objects)

    elif isinstance(objects, NPVSelection):
        return convert_enum_npv(objects=objects)

    elif isinstance(objects, DiscountingMode):
        return convert_enum_discountingmode(objects=objects)

    elif isinstance(objects, OtherRevenue):
        return convert_enum_otherrevenue(objects=objects)

    elif isinstance(objects, DeprMethod):
        return convert_enum_depreciationmethod(objects=objects)

    elif isinstance(objects, FTPTaxRegime):
        return convert_enum_ftptaxregime(objects=objects)

    elif isinstance(objects, TaxRegime):
        return convert_enum_taxregime(objects=objects)

    elif isinstance(objects, InflationAppliedTo):
        return convert_enum_inflationappliedto(objects=objects)

    elif isinstance(objects, GrossSplitRegime):
        return convert_enum_gsregime(objects=objects)

    elif isinstance(objects, (
            VariableSplit522017.FieldStatus,
            VariableSplit522017.FieldLocation,
            VariableSplit522017.ReservoirDepth,
            VariableSplit522017.InfrastructureAvailability,
            VariableSplit522017.ReservoirType,
            VariableSplit522017.CO2Content,
            VariableSplit522017.H2SContent,
            VariableSplit522017.APIOil,
            VariableSplit522017.DomesticUse,
            VariableSplit522017.ProductionStage,
    )):
        return objects.value

    elif isinstance(objects, (
            VariableSplit082017.FieldStatus,
            VariableSplit082017.FieldLocation,
            VariableSplit082017.ReservoirDepth,
            VariableSplit082017.InfrastructureAvailability,
            VariableSplit082017.ReservoirType,
            VariableSplit082017.CO2Content,
            VariableSplit082017.H2SContent,
            VariableSplit082017.APIOil,
            VariableSplit082017.DomesticUse,
            VariableSplit082017.ProductionStage,
    )):
        return objects.value

    else:
        return objects

def construct_lifting_attr(lifting: tuple[Lifting]):
    fluid_types = [(str(lift.fluid_type.value) + ' ' + str(index)).capitalize() for index, lift in enumerate(lifting)]
    liftings = [vars(lift) for lift in lifting]

    for lift in liftings:
        for key, item in lift.items():
            lift[key] = convert_object(objects=item)

    return dict(zip(fluid_types, liftings))

def construct_cost_attr(cost: tuple[CapitalCost] | tuple[Intangible] | tuple[OPEX] | tuple[ASR] | tuple[LBT] | tuple[CostOfSales]):
    cost_key = ['Cost ' + str(index) for index, _ in enumerate(cost)]
    costs = [vars(cst) for cst in cost]

    for cst in costs:
        for key, item in cst.items():
            if key == 'cost_allocation':
                cst[key] = [convert_enum_fluid(objects=fluid) for fluid in cst[key]]
            else:
                cst[key] = convert_object(objects=item)
    return dict(zip(cost_key, costs))

def construct_setup_attr(contract: BaseProject | CostRecovery | GrossSplit | Transition):
    return {
        "start_date": contract.start_date.strftime("%d/%m/%Y"),
        "end_date": contract.end_date.strftime("%d/%m/%Y"),
        "oil_onstream_date": contract.oil_onstream_date.strftime("%d/%m/%Y"),
        "gas_onstream_date": contract.gas_onstream_date.strftime("%d/%m/%Y"),
    }

def construct_summary_arguments_attr(summary_arguments: dict):
    for key, value in summary_arguments.items():
        summary_arguments[key] = convert_object(objects=value)
    return summary_arguments

def construct_costrecovery_attr(contract: CostRecovery):
    cr_setup =  {
        'oil_ftp_is_available': contract.oil_ftp_is_available,
        'oil_ftp_is_shared': contract.oil_ftp_is_shared,
        'oil_ftp_portion': contract.oil_ftp_portion,
        'gas_ftp_is_available': contract.gas_ftp_is_available,
        'gas_ftp_is_shared': contract.gas_ftp_is_shared,
        'gas_ftp_portion': contract.gas_ftp_portion,
        'tax_split_type': contract.tax_split_type,
        'condition_dict': contract.condition_dict,
        'indicator_rc_icp_sliding': contract.indicator_rc_icp_sliding,
        'oil_ctr_pretax_share': contract.oil_ctr_pretax_share,
        'gas_ctr_pretax_share': contract.gas_ctr_pretax_share,
        'oil_ic_rate': contract.oil_ic_rate,
        'gas_ic_rate': contract.gas_ic_rate,
        'ic_is_available': contract.ic_is_available,
        'oil_cr_cap_rate': contract.oil_cr_cap_rate,
        'gas_cr_cap_rate': contract.gas_cr_cap_rate,
        'oil_dmo_volume_portion': contract.oil_dmo_volume_portion,
        'oil_dmo_fee_portion': contract.oil_dmo_fee_portion,
        'oil_dmo_holiday_duration': contract.oil_dmo_holiday_duration,
        'gas_dmo_volume_portion': contract.gas_dmo_volume_portion,
        'gas_dmo_fee_portion': contract.gas_dmo_fee_portion,
        'gas_dmo_holiday_duration': contract.gas_dmo_holiday_duration,
    }

    for key, value in cr_setup.items():
        cr_setup[key] = convert_object(objects=value)

    return cr_setup

def construct_costrecovery_arguments_attr(contract_arguments: dict):
    cr_arguments = {
        "sulfur_revenue": contract_arguments.get("sulfur_revenue", "Addition to Oil Revenue"),
        "electricity_revenue": contract_arguments.get("electricity_revenue", "Addition to Oil Revenue"),
        "co2_revenue": contract_arguments.get("co2_revenue", "Addition to Oil Revenue"),
        "is_dmo_end_weighted": contract_arguments.get("is_dmo_end_weighted", False),
        "tax_regime": contract_arguments.get("tax_regime", "nailed down"),
        "effective_tax_rate": contract_arguments.get("effective_tax_rate", None),
        "ftp_tax_regime": contract_arguments.get("ftp_tax_regime", "PDJP No.20 Tahun 2017"),
        "sunk_cost_reference_year": contract_arguments.get("sunk_cost_reference_year", None),
        "depr_method": contract_arguments.get("depr_method", "PSC Declining Balance"),
        "decline_factor": contract_arguments.get("decline_factor", 2),
        "vat_rate": contract_arguments.get("vat_rate", 0.0),
        "inflation_rate": contract_arguments.get("inflation_rate", 0.0),
        "inflation_rate_applied_to": contract_arguments.get("inflation_rate_applied_to", None),
        "post_uu_22_year2001": contract_arguments.get("post_uu_22_year2001", True),
        "sum_undepreciated_cost": contract_arguments.get("sum_undepreciated_cost", True),
        "oil_cost_of_sales_applied": contract_arguments.get("oil_cost_of_sales_applied", False),
        "gas_cost_of_sales_applied": contract_arguments.get("gas_cost_of_sales_applied", False),
    }

    for key, value in cr_arguments.items():
        cr_arguments[key] = convert_object(objects=value)

    return cr_arguments

def construct_grosssplit_attr(contract: GrossSplit):
    gs_setup = {
        "field_status": contract.field_status,
        "field_loc": contract.field_loc,
        "res_depth": contract.res_depth,
        "infra_avail": contract.infra_avail,
        "res_type": contract.res_type,
        "api_oil": contract.api_oil,
        "domestic_use": contract.domestic_use,
        "prod_stage": contract.prod_stage,
        "co2_content": contract.co2_content,
        "h2s_content": contract.h2s_content,
        "base_split_ctr_oil": contract.base_split_ctr_oil,
        "base_split_ctr_gas": contract.base_split_ctr_gas,
        "split_ministry_disc": contract.split_ministry_disc,
        "oil_dmo_volume_portion": contract.oil_dmo_volume_portion,
        "oil_dmo_fee_portion": contract.oil_dmo_fee_portion,
        "oil_dmo_holiday_duration": contract.oil_dmo_holiday_duration,
        "gas_dmo_volume_portion": contract.gas_dmo_volume_portion,
        "gas_dmo_fee_portion": contract.gas_dmo_fee_portion,
        "gas_dmo_holiday_duration": contract.gas_dmo_holiday_duration,
    }

    for key, value in gs_setup.items():
        gs_setup[key] = convert_object(objects=value)

    return gs_setup

def construct_grosssplit_arguments_attr(contract_arguments: dict):
    gs_arguments = {
        "sulfur_revenue": contract_arguments.get("sulfur_revenue", "Addition to Oil Revenue"),
        "electricity_revenue": contract_arguments.get("electricity_revenue", "Addition to Oil Revenue"),
        "co2_revenue": contract_arguments.get("co2_revenue", "Addition to Oil Revenue"),
        "is_dmo_end_weighted": contract_arguments.get("is_dmo_end_weighted", False),
        "tax_regime": contract_arguments.get("tax_regime", "nailed down"),
        "effective_tax_rate": contract_arguments.get("effective_tax_rate", 0.22),
        "sunk_cost_reference_year": contract_arguments.get("sunk_cost_reference_year", None),
        "depr_method": contract_arguments.get("depr_method", "PSC Declining Balance"),
        "decline_factor": contract_arguments.get("decline_factor", 2),
        "vat_rate": contract_arguments.get("vat_rate", 0.0),
        "inflation_rate": contract_arguments.get("inflation_rate", 0.0),
        "inflation_rate_applied_to": contract_arguments.get("inflation_rate_applied_to", None),
        "cum_production_split_offset": contract_arguments.get("cum_production_split_offset", None),
        "amortization": contract_arguments.get("amortization", False),
        "regime": contract_arguments.get("regime", "PERMEN_ESDM_12_2020"),
        "sum_undepreciated_cost": contract_arguments.get("sum_undepreciated_cost", True),
    }

    for key, value in gs_arguments.items():
        gs_arguments[key] = convert_object(objects=value)

    return gs_arguments

def construct_transition_attr(contract: Transition):
    if isinstance(contract.contract1, CostRecovery):
        contract_1 = construct_costrecovery_attr(contract=contract.contract1)
    elif isinstance(contract.contract1, GrossSplit):
        contract_1 = construct_grosssplit_attr(contract=contract.contract1)
    else:
        raise GetAttrException(
            f"The contract: {type(contract.contract1)} type is not recognized"
        )

    if isinstance(contract.contract1, CostRecovery):
        contract_2 = construct_costrecovery_attr(contract=contract.contract2)
    elif isinstance(contract.contract1, GrossSplit):
        contract_2 = construct_grosssplit_attr(contract=contract.contract2)
    else:
        raise GetAttrException(
            f"The contract: {type(contract.contract2)} type is not recognized"
        )

    return contract_1, contract_2

def construct_transition_arguments(contract_arguments: dict):
    trans_arguments = {
        "unrec_portion": contract_arguments["unrec_portion"],
    }

    for key, value in trans_arguments.items():
        trans_arguments[key] = convert_object(objects=value)

    return trans_arguments

def construct_baseproject_arguments_attr(contract_arguments: dict):
    bp_arguments = {
        "sulfur_revenue": contract_arguments.get("sulfur_revenue", "Addition to Oil Revenue"),
        "electricity_revenue": contract_arguments.get("electricity_revenue", "Addition to Oil Revenue"),
        "co2_revenue": contract_arguments.get("co2_revenue", "Addition to Oil Revenue"),
        "sunk_cost_reference_year": contract_arguments.get("sunk_cost_reference_year", None),
        "year_inflation": contract_arguments.get("year_inflation", None),
        "inflation_rate": contract_arguments.get("inflation_rate", 0.0),
        "tax_rate": contract_arguments.get("tax_rate", 0.0),
        "inflation_rate_applied_to": contract_arguments.get("inflation_rate_applied_to", None),
    }

    for key, value in bp_arguments.items():
        bp_arguments[key] = convert_object(objects=value)

    return bp_arguments


def get_contract_attributes(
        contract: BaseProject | CostRecovery | GrossSplit | Transition,
        contract_arguments: dict,
        summary_arguments: dict,
) -> dict:
    """
    Function to get the attributes of a contract in a defined json compatible format.

    Parameters
    ----------
    contract: BaseProject | CostRecovery | GrossSplit | Transition
        The contract which the attributes will be retrieved in compatible json format.
    contract_arguments: dict
        The contract arguments
    summary_arguments: dict
        The summary arguments of the contract.

    Returns
    -------
    out: dict
        The dictionary containing the contract attributes in json compatible format.
    """

    # Constructing the setup and summary arguments key
    attr = {
        'setup': construct_setup_attr(contract=contract),
        'summary_arguments': construct_summary_arguments_attr(summary_arguments=summary_arguments),
    }

    # Constructing the contract config key
    if isinstance(contract, CostRecovery):
        attr['costrecovery'] = construct_costrecovery_attr(contract=contract)
        attr['contract_arguments'] = construct_costrecovery_arguments_attr(contract_arguments=contract_arguments)
    elif isinstance(contract, GrossSplit):
        attr['grosssplit'] = construct_grosssplit_attr(contract=contract)
        attr['contract_arguments'] = construct_grosssplit_arguments_attr(contract_arguments=contract_arguments)
    elif isinstance(contract, Transition):
        attr['contract_1'], attr['contract_2'] = construct_transition_attr(contract=contract)
        attr['contract_arguments'] = construct_transition_arguments(contract_arguments=contract_arguments)

    # Constructing the lifting key
    attr['lifting'] = construct_lifting_attr(lifting=contract.lifting)

    # Constructing the capital key
    attr['capital'] = construct_cost_attr(cost=contract.capital_cost)

    # Constructing the intangible key
    attr['intangible'] = construct_cost_attr(cost=contract.intangible_cost)

    # Constructing the opex key
    attr['opex'] = construct_cost_attr(cost=contract.opex)

    # Constructing the asr key
    attr['asr'] = construct_cost_attr(cost=contract.asr_cost)

    # Constructing the lbt key
    attr['lbt'] = construct_cost_attr(cost=contract.lbt_cost)

    # Constructing the cost of sales key
    attr['cost_of_sales'] = construct_cost_attr(cost=contract.cost_of_sales)

    return attr







