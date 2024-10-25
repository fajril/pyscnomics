"""
This file containing the tools which utilized by API adapter.
"""
from datetime import datetime, date
from typing import Dict, Union, Optional, List

from pydantic import BaseModel, Field
import numpy as np

from pyscnomics.econ.costs import CapitalCost, Intangible, OPEX, ASR, CostOfSales, LBT
from pyscnomics.dataset.sample import assign_lifting, read_fluid_type
from pyscnomics.econ.selection import TaxRegime, TaxType, FTPTaxRegime, GrossSplitRegime, LimitMethod
from pyscnomics.tools.helper import (get_inflation_applied_converter,
                                     get_npv_mode_converter,
                                     get_discounting_mode_converter,
                                     get_depreciation_method_converter,
                                     get_other_revenue_converter,
                                     get_split_type_converter,
                                     get_optimization_target_converter,
                                     get_optimization_parameter_converter)


class SetupBM(BaseModel):
    """
    The BaseModel to validate the API setup input data.

    Parameters
    ----------
    start_date : str | int
        The start date of the project.
    end_date : str | int
        The end date of the project.
    oil_onstream_date: str | int | None
        The start date of oil production.
    gas_onstream_date: str | int | None
        The start date of gas production.
    """
    start_date: str | int = "01/01/2010"
    end_date: str | int = "31/12/2045"
    oil_onstream_date: str | int | None = "01/01/2023"
    gas_onstream_date: str | int | None = "01/01/2023"


class SummaryArgumentsBM(BaseModel):
    """
    BaseModel to validate the API summary arguments input data.

    Parameters
    ----------
    reference_year: int
        The reference year used to calculate the economic indicator.
    inflation_rate: float
        The inflation rate used to calculate the economic indicator and cost inflation.
    discount_rate: float
        The discount rate used to calculate the discounted economic indicator.
    npv_mode: str
        The mode of the NPV used in the NPV calculation. The available option are:
        [SKK Full Cycle Real Terms, SKK Full Cycle Nominal Terms,
        Full Cycle Real Terms, Full Cycle Nominal Terms, Point Forward]
    discounting_mode: str
        The discounting mode used in the NPV calculation. The available option are: [End Year, Mid Year]

    """
    reference_year: int = 2022
    inflation_rate: float | int | list = 0.1
    discount_rate: float | int = 0.1
    npv_mode: str = "SKK Full Cycle Nominal Terms"
    discounting_mode: str = "End Year"
    profitability_discounted: bool = False


class CostRecoveryBM(BaseModel):
    """
    The BaseModel to validate the API Cost Recovery input data.

    Parameters
    ----------
    oil_ftp_is_available: bool
        The oil ftp availability of the cost recovery contract.
    oil_ftp_is_shared: bool
        The oil shared ftp condition of the cost recovery contract.
    oil_ftp_portion: float
        The oil ftp portion.
    gas_ftp_is_available: bool
        The gas ftp availability of the cost recovery contract.
    gas_ftp_is_shared: bool
        The gas shared ftp condition of the cost recovery contract.
    gas_ftp_portion: float
        The gas ftp portion.
    tax_split_type: str
        The tax split type used in the contract. The available option are: [Conventional, RC Split, ICP Sliding Scale].
    condition_dict: dict
        The condition input for the tax split type except for the conventional split.
    indicator_rc_icp_sliding: list
        The indicator input for the tax split type except for the conventional split.
    oil_ctr_pretax_share: float
        The oil contractor pre-tax share.
    gas_ctr_pretax_share: float
        The gas contractor pre-tax share.
    oil_ic_rate: float
        The oil Investment Credit (IC) rate.
    gas_ic_rate: float
        The gas Investment Credit (IC) rate.
    ic_is_available: bool
        The Investment Credit (IC) availability.
    oil_cr_cap_rate: float
        The oil cost recovery cap rate
    gas_cr_cap_rate: float
        The gas cost recovery cap rate
    oil_dmo_volume_portion: float
        The oil Domestic Market Obligation (DMO) volume portion.
    oil_dmo_fee_portion: float
        The oil Domestic Market Obligation (DMO) fee portion.
    oil_dmo_holiday_duration: float
        The oil Domestic Market Obligation (DMO) Holiday duration.
    gas_dmo_volume_portion: float
        The gas Domestic Market Obligation (DMO) volume portion.
    gas_dmo_fee_portion: float
        The gas Domestic Market Obligation (DMO) fee portion.
    gas_dmo_holiday_duration: float
        The gas Domestic Market Obligation (DMO) Holiday duration.
    """
    oil_ftp_is_available: bool = True
    oil_ftp_is_shared: bool = True
    oil_ftp_portion: float | int = 0.2
    gas_ftp_is_available: bool = True
    gas_ftp_is_shared: bool = True
    gas_ftp_portion: float | int = 0.2
    tax_split_type: str = "Conventional"
    condition_dict: dict = {}
    indicator_rc_icp_sliding: list[float] = []
    oil_ctr_pretax_share: float | int = 0.34722220
    gas_ctr_pretax_share: float | int = 0.5208330
    oil_ic_rate: float | int = 0
    gas_ic_rate: float | int = 0
    ic_is_available: bool = False
    oil_cr_cap_rate: float | int = 1
    gas_cr_cap_rate: float | int = 1
    oil_dmo_volume_portion: float | int = 0.25
    oil_dmo_fee_portion: float | int = 0.25
    oil_dmo_holiday_duration: float | int = 60
    gas_dmo_volume_portion: float | int = 0.25
    gas_dmo_fee_portion: float | int = 1
    gas_dmo_holiday_duration: float | int = 60


class GrossSplitBM(BaseModel):
    """
    The BaseModel to validate the API Gross Split input data.

    Parameters
    ----------
    field_status: str
        The field status. The available options are:
        [POD I, POD II, POFD, No POD].
    field_loc: str
        The field location. The available options are:
        [Onshore, Offshore (0<h<=20), Offshore (20<h<=50),
        Offshore (50<h<=150), Offshore (150<h<=1000), Offshore (h>1000)].
    res_depth: str
        The reservoir depth of the field. The available options are:
        [<=2500, >2500]
    infra_avail: str
        The infrastructure availability of the field. The available options are:
        [Well Developed, New Frontier].
    res_type: str
        The reservoir type of the field. The available options are:
        [Conventional, Non-Conventional]
    api_oil: str
        The oil API classification. The available options are:
        [<25, >=25].
    domestic_use: str
        The local content of the contract. The available options are:
        [<30, 30<=x<50, 50<=x<70, 70<=x<100]
    prod_stage: str
        The production stage of the contract. The available options are:
        [Primary, Secondary, Tertiary]
    co2_content: str
        The CO2 content of the field produced fluid. The available options are:
        [<5, 5<=x<10, 10<=x<20, 20<=x<40, 40<=x<60, x>=60]
    h2s_content: str
        The H2S content of the field produced fluid. The available options are:
        [<100, 100<=x<300, 300<=x<500, x>=500]
    base_split_ctr_oil: float
        The contractor base split for oil.
    base_split_ctr_gas: float
        The contractor base split for gas.
    split_ministry_disc: float
        The ministerial discretion split.
    oil_dmo_volume_portion: float
        The oil Domestic Market Obligation (DMO) volume portion.
    oil_dmo_fee_portion: float
        The oil Domestic Market Obligation (DMO) fee portion.
    oil_dmo_holiday_duration: float
        The oil Domestic Market Obligation (DMO) Holiday duration.
    gas_dmo_volume_portion: float
        The gas Domestic Market Obligation (DMO) volume portion.
    gas_dmo_fee_portion: float
        The gas Domestic Market Obligation (DMO) fee portion.
    gas_dmo_holiday_duration: float
        The gas Domestic Market Obligation (DMO) Holiday duration.

    """
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
    base_split_ctr_oil: float | int = 0.43
    base_split_ctr_gas: float | int = 0.48
    split_ministry_disc: float | int = 0.08
    oil_dmo_volume_portion: float | int = 0.25
    oil_dmo_fee_portion: float | int = 1.0
    oil_dmo_holiday_duration: float | int = 60
    gas_dmo_volume_portion: float | int = 1.0
    gas_dmo_fee_portion: float | int = 1.0
    gas_dmo_holiday_duration: float | int = 60


class ContractArgumentsBM(BaseModel):
    """
    The BaseModel to validate the Contract Arguments input data.

    Parameters
    ----------
    sulfur_revenue: str
        The treatment of the sulfur revenue. The options are:
        [Addition to Oil Revenue, Addition to Gas Revenue, Reduction to Oil OPEX, Reduction to Gas OPEX]
    electricity_revenue: str
        The treatment of the electricity revenue. The options are:
        [Addition to Oil Revenue, Addition to Gas Revenue, Reduction to Oil OPEX, Reduction to Gas OPEX]
    co2_revenue: str
        The treatment of the CO2 revenue. The options are:
        [Addition to Oil Revenue, Addition to Gas Revenue, Reduction to Oil OPEX, Reduction to Gas OPEX]
    is_dmo_end_weighted: bool
        The option treatment of the DMO.
    tax_regime: str
        The tax regime used in tax calculation. The options are:
        [nailed, prevailing, UU No.36 Tahun 2008, UU No.02 Tahun 2020, UU No.07 Tahun 2021]
    effective_tax_rate: float
        The tax rate used for the tax payment.
    ftp_tax_regime: str
        The tax scheme used to calculate the tax payment. The options are:
        [PDJP No.20 Tahun 2017, Pre PDJP No.20 Tahun 2017, Direct]
    sunk_cost_reference_year: int
        The sunk cost reference year.
    depr_method: str
        The depreciation method used for calculating the depreciation. The options are:
        [PSC Declining Balance, Declining Balance, Unit Of Production, Straight Line]
    decline_factor: float
        The decline factor for the depreciation if the depreciation method is Declining Balance.
    vat_rate: list
        The Value Added Tax (VAT) rate used in the calculation.
    lbt_rate: list
        The Land and Building (LBT) rate used in the calculation.
    inflation_rate: list
        The inflation rate used in the calculation.
    inflation_rate_applied_to: str
        The option of where the inflation will be applied to. The options are:
        [CAPEX, OPEX, CAPEX AND OPEX]

    """
    sulfur_revenue: str = "Addition to Oil Revenue"
    electricity_revenue: str = "Addition to Oil Revenue"
    co2_revenue: str = "Addition to Oil Revenue"
    is_dmo_end_weighted: bool = False
    tax_regime: str = "nailed down"
    effective_tax_rate: float | list | int | None = 0.424
    ftp_tax_regime: str = "Pre PDJP No.20 Tahun 2017"
    sunk_cost_reference_year: int | None = 2021
    depr_method: str = "PSC Declining Balance"
    decline_factor: float | int = 2
    vat_rate: list | float | int = 0.0
    lbt_rate: list | float | int = 0.0
    inflation_rate: list | float | int = 0.0
    inflation_rate_applied_to: str = "CAPEX"
    post_uu_22_year2001: bool = True
    cum_production_split_offset: list | float | int | None = None
    amortization: bool = False
    regime: str = "PERMEN_ESDM_12_2020"
    oil_cost_of_sales_applied: bool = False
    gas_cost_of_sales_applied: bool = False
    sum_undepreciated_cost: bool = False


class ContractArgumentsTransitionBM(BaseModel):
    """
    The BaseModel to validate the Contract Arguments of Transition input data.

    Parameters
    ----------
    unrec_portion: float
        The unrec portion that will be transferred into second contract.
    """
    unrec_portion: float | int


class LiftingBM(BaseModel):
    """
    The BaseModel to validate the Lifting input data.

    Parameters
    ----------
    start_year: int
        The start year of the lifting.
    end_year: int
        The end year of the lifting.
    lifting_rate: list[float]
        The list containing the lifting rate.
    price: list[float]
        The list containing the price.
    prod_year: list[int]
        The list containing the production year of the corresponding lifting.
    fluid_type: str
        The type of the corresponding fluid.
    ghv: list[float] | None
        The list containing the Gross Heating Value (GHV) of the corresponding fluid.
    prod_rate: list[float] | None
        The list containing the production rate of the corresponding lifting.
    prod_rate_baseline: list[float] | list[int] | None
        The list containing the production rate baseline of the corresponding lifting.
    """
    start_year: int
    end_year: int
    lifting_rate: list[float] | list[int]
    price: list[float] | list[int]
    prod_year: list[int]
    fluid_type: str
    ghv: list[float] | list[int] | None = None
    prod_rate: list[float] | list[int] | None = None
    prod_rate_baseline: list[float] | list[int] | None = None

class GeneralCostBM(BaseModel):
    start_year: int
    end_year: int
    expense_year: List[int]
    cost: Union[List[float], List[int]]
    cost_allocation: Optional[List[str]] = Field(default=None)
    description: Optional[List[str]] = Field(default=None)
    tax_portion: Optional[Union[List[float], List[int]]] = Field(default=None)
    tax_discount: Optional[Union[List[float], List[int]]] = Field(default=None)


class CapitalCostBM(GeneralCostBM):
    """
    The BaseModel to validate the Capital Cost input data.

    Parameters
    ----------
    pis_year: list[int]
        The list representing the PIS year of a capital asset.
    salvage_value: list[float]
        The list representing the salvage value of a capital asset.
    useful_life: list[int]
        The list representing the useful life of a capital asset.
    depreciation_factor: list[float]
        The value of depreciation factor to be used in PSC DB depreciation method.
        Default value is 0.5 for the entire project duration.
    is_ic_applied: list[bool]
        The condition of will the IC applied or not.
    """

    pis_year: Optional[list[int]] = Field(default=None)
    salvage_value: Optional[Union[list[int], list[float]]] = Field(default=None)
    useful_life: Optional[list[int]] = Field(default=None)
    depreciation_factor: Optional[list[float]] = Field(default=None)
    is_ic_applied: Optional[list[bool]] = Field(default=None)


class IntangibleBM(GeneralCostBM):
    """
    The BaseModel to validate the Intangible input data.
    """

class OpexBM(GeneralCostBM):
    """
    The BaseModel to validate the Opex input data.

    Parameters
    ----------
    fixed_cost : list
        An list representing the fixed cost of an opex asset.
    prod_rate: list
        The production rate of a particular fluid type.
    cost_per_volume: list
        Cost associated with production of a particular fluid type.
    """
    fixed_cost: list[float] | list[int]
    prod_rate: list[float] | list[int]
    cost_per_volume: list[float] | list[int]

    cost: list[float] | list[int] = Field(default=None, exclude=True)


class AsrBM(GeneralCostBM):
    """
    The BaseModel to validate the ASR input data.

    Parameters
    ----------
    final_year
        A list representing the final year of an ASR asset.
    future_rate
        A list representing the future rate of an ASR asset.
    """
    final_year: Optional[Union[List[float], List[int]]] = Field(default=None)
    future_rate: Optional[Union[List[float], List[int], float, int]] = Field(default=None)

class CostOfSalesBM(GeneralCostBM):
    """
    The BaseModel to validate the cost of sales input data.
    """

class LbtBM(GeneralCostBM):
    """
    The BaseModel to validate the Land and Building Tax (LBT) input data.

    Parameters
    ----------
    final_year
        The year in which cost distribution ends for each cost element.
    utilized_land_area
        The utilized land area.
    utilized_building_area
        The utilized building area.
    njop_land
        NJOP related to land.
    njop_building
        NJOP related to building.
    gross_revenue
        The array of gross revenues.

    """
    final_year: Optional[Union[List[float], List[int]]] = Field(default=None)
    utilized_land_area: Optional[Union[List[float], List[int]]] = Field(default=None)
    utilized_building_area: Optional[Union[List[float], List[int]]] = Field(default=None)
    njop_land: Optional[Union[List[float], List[int]]] = Field(default=None)
    njop_building: Optional[Union[List[float], List[int]]] = Field(default=None)
    gross_revenue: Optional[Union[List[float], List[int]]] = Field(default=None)

class OptimizationDictBM(BaseModel):
    """
    The BaseModel to validate the Optimization Dictionary input data.

    Parameters
    ----------
    parameter: list[str]
        The list of string containing the parameter which will be obtained.
    min: list[float]
        The list of float containing the minimum boundary for the optimization variable.
    max: list[float]
        The list of float containing the maximum boundary for the optimization variable.

    """
    parameter: list[str]
    min: list[float] | list[int]
    max: list[float] | list[int]


class OptimizationBM(BaseModel):
    """
    The BaseModel to validate the Optimization input data.

    Parameters
    ----------
    dict_optimization: OptimizationDictBM
        The dictionary containing the optimization parameter
    target_optimization: float
        The value of the optimization target.
    target_parameter: str
        The targeted optimization parameter.
    """
    dict_optimization: OptimizationDictBM
    target_optimization: float | int
    target_parameter: str


class SensitivityBM(BaseModel):
    """
    The BaseModel to validate the Sensitivity input data.

    Parameters
    ----------
    min: float
        The minimum value for the sensitivity
    max: float
        The maximum value for the sensitivity
    """
    min: float | int
    max: float | int


class UncertaintyBM(BaseModel):
    """
    The BaseModel to validate the uncertainty input data.

    Parameters
    ----------
    number_of_simulation: int
        The number of the simulation.
    min: list[float]
        The minimum value for uncertainty model
    max: list[float]
        The maximum value for uncertainty model
    std_dev: list[float]
        The standard deviation for the uncertainty model
    """
    number_of_simulation: int
    min: list[float] | list[int]
    max: list[float] | list[int]
    std_dev: list[float] | list[int]


class LtpBM(BaseModel):
    """
    The BaseModel to validate the LTP data input .

    Parameters
    ----------
    volume: float | int
        The volume of the reserves.
    start_year: int
        The start year.
    end_year: int
        The end year.
    fluid_type: str
        The fluid type of the corresponding volume. Should be "Oil" or "Gas".

    """
    volume: float | int
    start_year: int
    end_year: int
    fluid_type: str


class RpdBM(BaseModel):
    """
    The BaseModel to validate the RPD data input .

    Parameters
    ----------
    year_rampup: int
        Number of year from onstream to peak/plateau rate (yr).
    drate: float
        Arps yearly decline rate (1/yr).
    q_plateau_ratio: float
        Ratio of plateau rate and volume (1/yr).
    q_min_ratio: float
        Ratio of minimum rate at abandoned year and volume (1/yr).
    volume: float | int
        The volume of the reserves.
    start_year: int
        The start year.
    end_year: int
        The end year.
    """
    year_rampup: int
    drate: float | int
    q_plateau_ratio: float | int
    q_min_ratio: float | int
    volume: float | int
    start_year: int
    end_year: int


class Data(BaseModel):
    """
    The BaseModel to validate the Data input.

    Parameters
    ----------
    setup: SetupBM
        The setup input in form of SetupBM BaseModel.
    summary_arguments: SummaryArgumentsBM
        The summary arguments in form of SummaryArgumentsBM BaseModel.
    contract_arguments: ContractArgumentsBM
        The contract arguments in form of ContractArgumentsBM BaseModel.
    lifting: Dict[str, LiftingBM]
        The lifting input in form of LiftingBM BaseModel.
    capital: Dict[str, CapitalCostBM]
        The capital input in form of TangibleBM BaseModel.
    intangible: Dict[str, IntangibleBM]
        The intangible input in form of IntangibleBM BaseModel.
    opex: Dict[str, OpexBM]
        The opex input in form of OpexBM BaseModel.
    asr: Dict[str, AsrBM]
        The asr input in form of AsrBM BaseModel.
    lbt: Dict[str, AsrBM]
        The lbt input in form of LBT BaseModel.
    cost_of_sales: Dict[str, CosOfSalesBM]
        The asr input in form of Cost of Sales BaseModel.
    optimization_arguments: OptimizationBM = None
        The optimization arguments in form of OptimizationBM BaseModel.
    sensitivity_arguments: SensitivityBM = None
        The sensitivity arguments in form of SensitivityBM BaseModel.
    uncertainty_arguments: UncertaintyBM = None
        The uncertainty arguments in form of UncertaintyBM BaseModel.
    costrecovery: CostRecoveryBM = None
        The cost recovery object in form of CostRecoveryBM BaseModel.
    grosssplit: GrossSplitBM = None
        The gross split object in form of GrossSplit BaseModel.
    result: dict = None
        The result of the running contract.
    """
    setup: SetupBM
    summary_arguments: SummaryArgumentsBM
    contract_arguments: ContractArgumentsBM
    lifting: Dict[str, LiftingBM] = Field(default=None)
    capital: Dict[str, CapitalCostBM] = Field(default=None)
    intangible: Dict[str, IntangibleBM] = Field(default=None)
    opex: Dict[str, OpexBM] = Field(default=None)
    asr: Dict[str, AsrBM] = Field(default=None)
    lbt: Optional[Dict[str, LbtBM]] = Field(default=None)
    cost_of_sales: Optional[Dict[str, CostOfSalesBM]] = Field(default=None)
    optimization_arguments: OptimizationBM = Field(default=None)
    sensitivity_arguments: SensitivityBM = Field(default=None)
    uncertainty_arguments: UncertaintyBM = Field(default=None)
    costrecovery: CostRecoveryBM = Field(default=None)
    grosssplit: GrossSplitBM = Field(default=None)
    result: dict = Field(default=None)


class TransitionBM(BaseModel):
    """
    The BaseModel to validate the contract in the Transition input data.

    Parameters
    ----------
    setup: SetupBM
        The setup input in form of SetupBM BaseModel.
    contract_arguments: ContractArgumentsBM
        The contract arguments in form of ContractArgumentsBM BaseModel.
    lifting: Dict[str, LiftingBM]
        The lifting input in form of LiftingBM BaseModel.
    capital: Dict[str, CapitalCostBM]
        The tangible input in form of TangibleBM BaseModel.
    intangible: Dict[str, IntangibleBM]
        The intangible input in form of IntangibleBM BaseModel.
    opex: Dict[str, OpexBM]
        The opex input in form of OpexBM BaseModel.
    asr: Dict[str, AsrBM]
        The asr input in form of AsrBM BaseModel.
    costrecovery: CostRecoveryBM = None
        The cost recovery object in form of CostRecoveryBM BaseModel.
    grosssplit: GrossSplitBM = None
        The gross split object in form of GrossSplit BaseModel.
    """
    setup: SetupBM
    contract_arguments: ContractArgumentsBM
    lifting: Dict[str, LiftingBM]
    capital: Dict[str, CapitalCostBM]
    intangible: Dict[str, IntangibleBM]
    opex: Dict[str, OpexBM]
    asr: Dict[str, AsrBM]
    lbt: Dict[str, LbtBM]
    cost_of_sales: Dict[str, CostOfSalesBM] = None
    costrecovery: CostRecoveryBM | None = None
    grosssplit: GrossSplitBM | None = None


class DataTransition(BaseModel):
    """
    The BaseModel to validate the Transition Data input.

    Parameters
    ----------
    contract_1: TransitionBM
        The first contract object in form of TransitionBM
    contract_2: TransitionBM
        The second contract object in form of TransitionBM
    contract_arguments: ContractArgumentsTransitionBM
        The contract arguments in form of ContractArgumentsBM BaseModel.
    summary_arguments: SummaryArgumentsBM
        The summary arguments in form of SummaryArgumentsBM BaseModel.
    result: dict = None
        The result of the running contract.
    """
    contract_1: TransitionBM
    contract_2: TransitionBM
    contract_arguments: ContractArgumentsTransitionBM
    summary_arguments: SummaryArgumentsBM
    result: dict = None
    optimization_arguments: OptimizationBM = None


class EconLimit(BaseModel):
    """
    The BaseModel to validate the Economic Limit.

    Parameters
    ----------
    """
    years: list[int]
    cash_flow: list[int] | list[float]
    method: str

class ASRExpendituresBM(BaseModel):
    """
    The BaseModel to validate the ASR Expenditures Data.

    Parameters
    ----------
    start_date : str | int
        The start date of the project.
    end_date : str | int
        The end date of the project.
    asr : AsrBM
        The asr input in form of AsrBM BaseModel.
    """
    start_date: str | int = "01/01/2010"
    end_date: str | int = "01/01/2010"
    asr: AsrBM


def convert_str_to_date(str_object: str | int) -> date | None:
    """
    The function to convert string or integer unix timestamp format object into dateformat
    Parameters
    ----------
    str_object: str | int
        The object which will be converted into date format

    Returns
    -------
    date

    """
    if str_object is None:
        return None
    else:
        if isinstance(str_object, str):
            return datetime.strptime(str_object, '%d/%m/%Y').date()
        elif isinstance(str_object, int):
            value = datetime.fromtimestamp(str_object)
            date_value = value.strftime('%d/%m/%Y')
            return datetime.strptime(date_value, '%d/%m/%Y').date()


def convert_list_to_array_float(data_list: list) -> np.ndarray:
    """
    The function to convert list into array of float.

    Parameters
    ----------
    data_list: list
        The list which will be converted into numpy array.

    Returns
    -------
    array
        np.ndarray
    """
    return np.array(data_list, dtype=float)


def convert_list_to_array_float_or_array(data_input: list | float | int | str) -> np.ndarray | float:
    """
    The function to convert list or float into float or array of float.

    Parameters
    ----------
    data_input: list | float
        The list or float which will be converted

    Returns
    -------
    output
        float | np.ndarray
    """
    if isinstance(data_input, float):
        return data_input

    elif isinstance(data_input, int) or isinstance(data_input, str):
        return (
            float(data_input)
            if isinstance(data_input, int)
               or (isinstance(data_input, str) and data_input.strip() != "")
            else 0
        )
    else:
        return np.array(data_input, dtype=float)


def convert_list_to_array_float_or_array_or_none(data_list: list | float | int | str | None
                                                 ) -> np.ndarray | float | None:
    """
    Function to convert list into array of float, None or array.

    Parameters
    ----------
    data_list
        The list which will be converted.

    Returns
    -------
    out
        np.ndarray | float | None

    """
    if isinstance(data_list, float):
        return data_list
    elif isinstance(data_list, int) or isinstance(data_list, str):
        return (
            float(data_list)
            if isinstance(data_list, int)
               or (isinstance(data_list, str) and data_list.strip() != "")
            else None
        )
    elif data_list is None:
        return None
    else:
        return np.array(data_list, dtype=float)


def convert_list_to_array_int(data_list: list) -> np.ndarray:
    """
    The function to convert list into array of integer.

    Parameters
    ----------
    data_list: list
        The list which will be converted into numpy array of integer.

    Returns
    -------
    out
        np. array

    """
    return np.array(data_list, dtype=int)


def convert_dict_to_lifting(data_raw: dict) -> tuple:
    """
    The function to convert dictionary into tuple of Lifting dataclass.

    Parameters
    ----------
    data_raw: dict
        The dictionary which will be converted into tuple of Lifting


    Returns
    -------
    out:
        tuple[Lifting]
    """
    return assign_lifting(data_raw=data_raw)


def convert_dict_to_capital(data_raw: dict) -> tuple | None:
    """
    The function to convert a dictionary into tuple of CapitalCost dataclass.

    Parameters
    ----------
    data_raw
        The dictionary which will be converted into tuple of CapitalCost

    Returns
    -------
    out:
        tuple[CapitalCost] | None
    """
    capital = tuple([
        CapitalCost(
            start_year=data_raw[key]['start_year'],
            end_year=data_raw[key]['end_year'],
            cost=np.array(data_raw[key]['cost']),
            expense_year=np.array(data_raw[key]['expense_year'], dtype=int),
            cost_allocation=read_fluid_type(fluid=data_raw[key]['cost_allocation']),
            description=data_raw[key]['description'],
            tax_portion=np.array(data_raw[key]['tax_portion'], dtype=float),
            tax_discount=np.array(data_raw[key]['tax_discount'], dtype=float),
            pis_year=np.array(data_raw[key]['pis_year']),
            salvage_value=np.array(data_raw[key]['salvage_value']),
            useful_life=np.array(data_raw[key]['useful_life']),
            depreciation_factor=np.array(data_raw[key]['depreciation_factor']),
            is_ic_applied=data_raw[key]['is_ic_applied'],
        )
        for key in data_raw.keys()
    ]) if data_raw is not None else None

    return capital


def convert_dict_to_intangible(data_raw: dict) -> tuple | None:
    """
    The function to convert dictionary into tuple of Intangible dataclass.

    Parameters
    ----------
    data_raw: dict
        The dictionary which will be converted into tuple of Intangible

    Returns
    -------
    out:
        tuple[Intangible] | None
    """
    intangible = tuple([
        Intangible(
            start_year=data_raw[key]['start_year'],
            end_year=data_raw[key]['end_year'],
            cost=np.array(data_raw[key]['cost'], dtype=float),
            expense_year=np.array(data_raw[key]['expense_year'], dtype=int),
            cost_allocation=read_fluid_type(fluid=data_raw[key]['cost_allocation']),
            description=data_raw[key]['description'],
            tax_portion=np.array(data_raw[key]['tax_portion'], dtype=float),
            tax_discount=np.array(data_raw[key]['tax_discount'], dtype=float),
        )
        for key in data_raw.keys()
    ]) if data_raw is not None else None

    return intangible


def convert_dict_to_opex(data_raw: dict) -> tuple | None:
    """
    The function to convert dictionary into tuple of OPEX dataclass.

    Parameters
    ----------
    data_raw: dict
        The dictionary which will be converted into tuple of OPEX

    Returns
    -------
    out:
        tuple[OPEX] | None
    """
    opex = tuple([
        OPEX(
            start_year=data_raw[key]['start_year'],
            end_year=data_raw[key]['end_year'],
            expense_year=np.array(data_raw[key]['expense_year'], dtype=int),
            cost_allocation=read_fluid_type(fluid=data_raw[key]['cost_allocation']),
            description=data_raw[key]['description'],
            tax_portion=np.array(data_raw[key]['tax_portion'], dtype=float),
            tax_discount=np.array(data_raw[key]['tax_discount'], dtype=float),
            fixed_cost=np.array(data_raw[key]['fixed_cost'], dtype=float),
            prod_rate=np.array(data_raw[key]['prod_rate'], dtype=float),
            cost_per_volume=np.array(data_raw[key]['cost_per_volume'], dtype=float),
        )
        for key in data_raw.keys()
    ]) if data_raw is not None else None

    return opex


def convert_dict_to_asr(data_raw: dict) -> tuple:
    """
    The function to convert dictionary into tuple of ASR dataclass.

    Parameters
    ----------
    data_raw: dict
        The dictionary which will be converted into tuple of ASR

    Returns
    -------
    out:
        tuple[ASR]
    """
    asr = tuple([
        ASR(
            start_year=data_raw[key]['start_year'],
            end_year=data_raw[key]['end_year'],
            cost=np.array(data_raw[key]['cost'], dtype=float),
            expense_year=np.array(data_raw[key]['expense_year'], dtype=int),
            cost_allocation=read_fluid_type(fluid=data_raw[key]['cost_allocation']),
            description=data_raw[key]['description'],
            tax_portion=np.array(data_raw[key]['tax_portion'], dtype=float),
            tax_discount=np.array(data_raw[key]['tax_discount'], dtype=float),
            final_year=np.array(data_raw[key]['final_year'], dtype=float) if data_raw[key]['final_year'] is not None else None,
            future_rate=np.array(
                data_raw[key]['future_rate'],
                dtype=float) if isinstance(data_raw[key]['future_rate'], list) else data_raw[key]['future_rate'],
        )
        for key in data_raw.keys()
    ]) if data_raw is not None else None


    return asr

def convert_dict_to_lbt(data_raw: dict) -> tuple:
    """
    The function to convert dictionary into tuple of LBT dataclass.

    Parameters
    ----------
    data_raw: dict
        The dictionary which will be converted into tuple of Land and Building Tax

    Returns
    -------
    out:
        tuple[Intangible]
    """
    lbt = tuple([
        LBT(
            start_year=data_raw[key]['start_year'],
            end_year=data_raw[key]['end_year'],
            cost=np.array(data_raw[key]['cost'], dtype=float),
            expense_year=np.array(data_raw[key]['expense_year'], dtype=int),
            cost_allocation=read_fluid_type(fluid=data_raw[key]['cost_allocation']),
            description=data_raw[key]['description'],
            tax_portion=np.array(data_raw[key]['tax_portion'], dtype=float),
            tax_discount=np.array(data_raw[key]['tax_discount'], dtype=float),
            final_year=np.array(data_raw[key]['final_year'], dtype=float) if data_raw[key]['final_year'] is not None else None,
            utilized_land_area=np.array(data_raw[key]['utilized_land_area'], dtype=float) if data_raw[key]['utilized_land_area'] is not None else None,
            utilized_building_area=np.array(data_raw[key]['utilized_building_area'], dtype=float) if data_raw[key]['utilized_building_area'] is not None else None,
            njop_land=np.array(data_raw[key]['njop_land'], dtype=float) if data_raw[key]['njop_land'] is not None else None,
            njop_building=np.array(data_raw[key]['njop_building'], dtype=float) if data_raw[key]['njop_building'] is not None else None,
            gross_revenue=np.array(data_raw[key]['gross_revenue'], dtype=float) if data_raw[key]['gross_revenue'] is not None else None,
        )
        for key in data_raw.keys()
    ]) if data_raw is not None else None

    return lbt


def convert_dict_to_cost_of_sales(data_raw: dict) -> tuple:
    """
    The function to convert dictionary into tuple of CostOfSales dataclass.

    Parameters
    ----------
    data_raw: dict
        The dictionary which will be converted into tuple of Cost Of Sales

    Returns
    -------
    out:
        tuple[CostOfSales]
    """
    cos = tuple([
        CostOfSales(
            start_year=data_raw[key]['start_year'],
            end_year=data_raw[key]['end_year'],
            cost=np.array(data_raw[key]['cost'], dtype=float),
            expense_year=np.array(data_raw[key]['expense_year'], dtype=int),
            cost_allocation=read_fluid_type(fluid=data_raw[key]['cost_allocation']),
            description=data_raw[key]['description'],
            tax_portion=np.array(data_raw[key]['tax_portion'], dtype=float),
            tax_discount=np.array(data_raw[key]['tax_discount'], dtype=float),
        )
        for key in data_raw.keys()
    ]) if data_raw is not None else None

    return cos


def convert_str_to_taxsplit(str_object: str):
    """
    The function to converts a string representing a tax split type to its corresponding
    enum value from the TaxSplitTypeCR class.

    Parameters
    ----------
    str_object: str
        The string representation of the tax split type.

    Returns
    -------
    Union: TaxSplitTypeCR, None
        The corresponding enum value if the target matches one of the predefined split types.
        Returns None if no match is found.

    Example
    -------
    >>> result = get_split_type_converter("RC Split")
    >>> print(result)
    TaxSplitTypeCR.R2C
    """
    return get_split_type_converter(target=str_object)


def convert_str_to_npvmode(str_object: str):
    """
    The function to converts a string representing an NPV mode to its corresponding
    enum value from the NPVSelection class.

    Parameters
    ----------
    str_object: str
        The string representation of the NPV mode.

    Returns
    -------
    Union[NPVSelection, None]
        The corresponding enum value if the target matches one of the predefined NPV modes.
        Returns None if no match is found.

    Example
    -------
    >>> result = get_npv_mode_converter("SKK Full Cycle Real Terms")
    >>> print(result)
    NPVSelection.NPV_SKK_REAL_TERMS
    """
    return get_npv_mode_converter(target=str_object)


def convert_str_to_discountingmode(str_object: str):
    """
    The function to converts a string representing a discounting mode to its corresponding
    enum value from the DiscountingMode class.

    Parameters
    ----------
    str_object: str
        The string representation of the discounting mode.

    Returns
    -------
        Union[DiscountingMode, None]
        The corresponding enum value if the target matches one of the predefined discounting modes.
        Returns None if no match is found.

    Example
    -------
    >>> result = get_discounting_mode_converter("End Year")
    >>> print(result)
    DiscountingMode.END_YEAR

    """
    return get_discounting_mode_converter(target=str_object)


def convert_str_to_otherrevenue(str_object: str):
    """
    The function to convert a string representation of other revenue types to the corresponding enum value.

    Parameters
    ----------
    str_object: str
        The string representation of the other revenue type.

    Returns
    -------
    OtherRevenue or None
        The enum value corresponding to the provided string representation of other revenue.
        Returns None if the string does not match any known other revenue types.

    Notes
    -----
    - The function uses a mapping of string representations to OtherRevenue enum values.
    - If the target string matches a known other revenue type, the corresponding enum value is returned.
    - If the target string does not match any known types, the function returns None.
    """
    return get_other_revenue_converter(target=str_object)


def convert_str_to_taxregime(str_object: str) -> TaxRegime | None:
    """
    The function to convert string into tax regime selection.

    Parameters
    ----------
    str_object: str
        The string object which will be converted

    Returns
    -------
    out:
        TaxRegime | None
    """
    dict_tax_regime = {i.value: i for i in TaxRegime}

    for key in dict_tax_regime.keys():
        if str_object == key:
            return dict_tax_regime[key]
        else:
            return None


def convert_str_to_ftptaxregime(str_object: str):
    """
    The functon to converts a string representing a tax regime to its corresponding
    enum value from the FTPTaxRegime class.

    Parameters
    ----------
    str_object: str
        The string representation of the tax regime.


    Returns
    -------
    Union[FTPTaxRegime, None]
        The corresponding enum value if the target matches one of the predefined tax regimes.
        Returns None if no match is found.

    """
    attrs = {
        "PDJP No.20 Tahun 2017": FTPTaxRegime.PDJP_20_2017,
        "Pre PDJP No.20 Tahun 2017": FTPTaxRegime.PRE_PDJP_20_2017,
        "Direct Mode": FTPTaxRegime.DIRECT_MODE
    }

    for key in attrs.keys():
        if str_object == key:
            return attrs[key]


def convert_str_to_depremethod(str_object: str):
    """
    The function to converts a string representing a depreciation method to its corresponding
    enum value from the DeprMethod class.

    Parameters
    ----------
    str_object: str
        The string representation of the depreciation method.

    Returns
    -------
    Union[DeprMethod, None]
        The corresponding enum value if the target matches one of the predefined depreciation methods.
        Returns None if no match is found.

    Example
    -------
    >>> result = get_depreciation_method_converter("PSC Declining Balance")
    >>> print(result)
    DeprMethod.PSC_DB

    """
    return get_depreciation_method_converter(target=str_object)


def convert_str_to_taxtype(str_object: str):
    """
    The function to convert a string object into tax type selection.

    Parameters
    ----------
    str_object: str
        The string representation of tax type.

    Returns
    -------
    out:
        TaxType
    """
    dict_tax_regime = {i.value: i for i in TaxType}

    for key in dict_tax_regime.keys():
        if str_object == key:
            return dict_tax_regime[key]
        else:
            return None


def convert_str_to_inflationappliedto(str_object: str):
    """
    The function to converts a string representing the application of inflation to its
    corresponding enum value from the InflationAppliedTo class.

    Parameters
    ----------
    str_object: str
        The string representation of the inflation application.

    Returns
    -------
    Union[InflationAppliedTo, None]
        The corresponding enum value if the target matches one of the predefined options.
        Returns None if no match is found.

    Example:
    >>> result = get_inflation_applied_converter("OPEX")
    >>> print(result)
    InflationAppliedTo.OPEX

    """
    return get_inflation_applied_converter(target=str_object)


def convert_grosssplitregime_to_enum(target: str) -> GrossSplitRegime:
    """
    Converts a string representing the Gross Split Regime to its
    corresponding enum value from the GrossSplitRegime enum class.

    Parameters
    ----------
    target: str
        The string representation of the gross split regime.

    Returns
    -------

    """
    attrs = {
        "PERMEN_ESDM_8_2017": GrossSplitRegime.PERMEN_ESDM_8_2017,
        "PERMEN_ESDM_52_2017": GrossSplitRegime.PERMEN_ESDM_52_2017,
        "PERMEN_ESDM_20_2019": GrossSplitRegime.PERMEN_ESDM_20_2019,
        "PERMEN_ESDM_12_2020": GrossSplitRegime.PERMEN_ESDM_12_2020,
    }

    for key in attrs.keys():
        if target == key:
            return attrs[key]


def convert_summary_to_dict(dict_object: dict):
    """
    The function to convert the summary into skk executive summary format.
    Parameters
    ----------
    dict_object: dict
        The dictionary representation of the summary.

    Returns
    -------
    out: dict
        The dictionary of skk executive summary.

    """
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
        'opex_asr_lbt': dict_object['opex_asr_lbt'],
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
        'undepreciated_asset_oil': dict_object['undepreciated_asset_oil'],
        'undepreciated_asset_gas': dict_object['undepreciated_asset_gas'],
        'undepreciated_asset_total': dict_object['undepreciated_asset_total'],
        'total_indirect_taxes': dict_object['total_indirect_taxes'],
        'oil_indirect_taxes': dict_object['oil_indirect_taxes'],
        'gas_indirect_taxes': dict_object['gas_indirect_taxes'],


    }
    return summary_skk_format


def convert_str_to_optimization_targetparameter(str_object: str):
    """
    The function to converts a string representation of an optimization target to the corresponding
    OptimizationTarget enum value.

    Parameters
    ----------
    str_object: str
        The string representation of the optimization target.

    Returns
    -------
    Optional[OptimizationTarget]
        The OptimizationTarget enum value if the target is valid, or None if the target is not found.

    Example
    -------
    >>> get_optimization_target_converter("NPV")
    OptimizationTarget.NPV
    >>> get_optimization_target_converter("IRR")
    OptimizationTarget.IRR
    >>> get_optimization_target_converter("PI")
    OptimizationTarget.PI
    >>> get_optimization_target_converter("InvalidTarget")
    None
    """
    return get_optimization_target_converter(target=str_object)


def convert_str_to_optimization_parameters(str_object: str):
    """
    The function to converts a string representation of an optimization parameter to the corresponding
    OptimizationParameter enum value.

    Parameters
    ----------
    str_object: str
        The string representation of the optimization parameter.

    Returns
    -------
    Union[OptimizationParameter, None]
        The OptimizationParameter enum value if the target is valid,
        or None if the target is not found.

    Example
    -------
    >>> get_optimization_parameter_converter("Oil Contractor Pre Tax")
    OptimizationParameter.OIL_CTR_PRETAX
    >>> get_optimization_parameter_converter("Gas FTP Portion")
    OptimizationParameter.GAS_FTP_PORTION
    >>> get_optimization_parameter_converter("VAT Rate")
    OptimizationParameter.VAT_RATE
    >>> get_optimization_parameter_converter("InvalidParameter")
    None

    """
    return get_optimization_parameter_converter(target=str_object)


def convert_to_float(target=int):
    """
    Function to convert integer into float.

    Parameters
    ----------
    target: int
        The target that will be converted.

    Returns
    -------
    float
    """
    return float(target)


def convert_to_method_limit(target:str):
    """
    Function to convert string into LimitMethod selection.

    Parameters
    ----------
    target: dict
        The target that will be converted.

    Returns
    -------
    LimitMethod

    """
    if target == 'Maximum Cumulative Cashflow':
        return LimitMethod.MAX_CUM_CASHFLOW
    elif target == 'Maximum NPV':
        return LimitMethod.MAX_NPV
    elif target == 'Negative Cashflow':
        return LimitMethod.NEGATIVE_CASHFLOW
    else:
        return ValueError("Invalid LimitMethod provided.")