"""
Collection of functions to administer Monte Carlo simulation.
The code below is the modification of the code from PSCnomics. The routine and result of this module is maintaining the
requirements of the PSCnomics.
"""
import copy
import numpy as np
from scipy.stats import uniform, triang, truncnorm

from pyscnomics.tools.summary import get_summary
from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts.transition import Transition
from pyscnomics.econ import FluidType
from pyscnomics.econ.selection import UncertaintyDistribution
from pyscnomics.io.getattr import get_contract_attributes
from pyscnomics.api.converter import (
    convert_to_float,
    convert_str_to_taxsplit,
    convert_list_to_array_float,
    convert_str_to_otherrevenue,
    convert_str_to_taxregime,
    convert_list_to_array_float_or_array_or_none,
    convert_str_to_ftptaxregime,
    convert_str_to_depremethod,
    convert_list_to_array_float_or_array,
    convert_str_to_inflationappliedto,
    convert_str_to_npvmode,
    convert_str_to_discountingmode,
    convert_str_to_date,
    convert_dict_to_lifting,
    convert_dict_to_capital,
    convert_dict_to_intangible,
    convert_dict_to_opex,
    convert_dict_to_asr,
    convert_dict_to_lbt,
    convert_dict_to_cost_of_sales,
    convert_grosssplitregime_to_enum
)


class MonteCarloException(Exception):
    """ Exception to be raised for a misuse of MonteCarlo class """

    pass

################################################ Uncertainty Detached ################################################
def get_setup_dict(data: dict) -> tuple:
    """
    Function to get conversion of the setup input from dictionary into acceptable core engine data format.

    Parameters
    ----------
    data: dict
        The dictionary of the data input

    Returns
    -------
    start_date: date
        The start date of the project.
    end_date: date
        The end date of the project.
    oil_onstream_date: date
        The oil onstream date.
    gas_onstream_date: date
        The gas onstream date.
    lifting: Lifting
        The lifting of the project, in Lifting Dataclass format.
    capital: Tangible
        The capital cost of the project, in Tangible Dataclass format.
    intangible: Intangible
        The intangible cost of the project, in Intangible Dataclass format.
    opex: OPEX
        The opex cost of the project, in OPEX Dataclass format.
    lbt: LBT
        The land and building tax of the project, in LBT Dataclass format.
    cost_of_sales: CostOfSales
        The opex cost of the project, in CostOfSales Dataclass format.
    asr: ASR
        The asr cost of the project, in ASR Dataclass format.

    """
    # Parsing the contract setup into each corresponding variables
    start_date = convert_str_to_date(str_object=data['setup']['start_date'])
    end_date = convert_str_to_date(str_object=data['setup']['end_date'])
    oil_onstream_date = convert_str_to_date(str_object=data['setup']['oil_onstream_date'])
    gas_onstream_date = convert_str_to_date(str_object=data['setup']['gas_onstream_date'])
    lifting = convert_dict_to_lifting(data_raw=data)
    capital = convert_dict_to_capital(data_raw=data['capital'])
    intangible = convert_dict_to_intangible(data_raw=data['intangible'])
    opex = convert_dict_to_opex(data_raw=data['opex'])
    asr = convert_dict_to_asr(data_raw=data['asr'])
    lbt = convert_dict_to_lbt(data_raw=data['lbt'])
    cost_of_sales = convert_dict_to_cost_of_sales(data_raw=data['cost_of_sales'])
    return start_date, end_date, oil_onstream_date, gas_onstream_date, lifting, capital, intangible, opex, asr, lbt, cost_of_sales

def get_summary_dict(data: dict) -> dict:
    """
    Function to get the summary arguments from the dictionary data input.
    Parameters
    ----------
    data: dict
        The dictionary of the data input

    Returns
    -------
    summary_arguments_dict: dict
        The summary argument in the core engine acceptable format.
    """
    # Filling the argument with the input data
    reference_year = data['summary_arguments'].get('reference_year', None)
    inflation_rate = data['summary_arguments'].get('inflation_rate', None)
    discount_rate = data['summary_arguments'].get('discount_rate', 0.1)
    npv_mode = convert_str_to_npvmode(str_object=data['summary_arguments'].get('npv_mode', "Full Cycle Nominal Terms"))
    discounting_mode = convert_str_to_discountingmode(str_object=data['summary_arguments'].get('discounting_mode', 'discounting_mode'))
    profitability_discounted = data['summary_arguments'].get('profitability_discounted', False)

    summary_arguments_dict = {
        'reference_year': reference_year,
        'inflation_rate': inflation_rate,
        'discount_rate': discount_rate,
        'npv_mode': npv_mode,
        'discounting_mode': discounting_mode,
        'profitability_discounted': profitability_discounted
    }

    return summary_arguments_dict

def get_baseproject(data: dict):
    """
    The function to get the Summary, Base Project object, contract arguments, and summary arguments used.

    Parameters
    ----------
    data: dict
        The dictionary of the data input.

    Returns
    -------
    summary_skk: dict
        The executive summary of the contract.
    """
    start_date, end_date, oil_onstream_date, gas_onstream_date, lifting, tangible, intangible, opex, asr, lbt, cost_of_sales = (
        get_setup_dict(data=data))

    contract = BaseProject(start_date=start_date,
                           end_date=end_date,
                           oil_onstream_date=oil_onstream_date,
                           gas_onstream_date=gas_onstream_date,
                           lifting=lifting,
                           capital_cost=tangible,
                           intangible_cost=intangible,
                           opex=opex,
                           asr_cost=asr,
                           lbt_cost=lbt)

    contract_arguments_dict = {
        "sulfur_revenue": convert_str_to_otherrevenue(str_object=data['contract_arguments']['sulfur_revenue']),
        "electricity_revenue": convert_str_to_otherrevenue(
            str_object=data['contract_arguments']['electricity_revenue']),
        "co2_revenue": convert_str_to_otherrevenue(str_object=data['contract_arguments']['co2_revenue']),
        "sunk_cost_reference_year": data['contract_arguments']['sunk_cost_reference_year'],
        "tax_rate": convert_list_to_array_float_or_array(data_input=data['contract_arguments']['tax_rate']),
        "inflation_rate": convert_list_to_array_float_or_array(data_input=data['contract_arguments']['inflation_rate']),
        "inflation_rate_applied_to": convert_str_to_inflationappliedto(str_object=data['contract_arguments']['inflation_rate_applied_to']),
    }

    contract.run(**contract_arguments_dict)

    summary_arguments_dict = get_summary_dict(data=data)
    summary_arguments_dict['contract'] = contract
    return get_summary(**summary_arguments_dict)

def get_costrecovery(data: dict):
    """
    The function to get the Summary, Cost Recovery object, contract arguments, and summary arguments used.

    Parameters
    ----------
    data: dict
        The dictionary of the data input.

    Returns
    -------
    summary_skk: dict
        The executive summary of the contract.
    """
    start_date, end_date, oil_onstream_date, gas_onstream_date, lifting, tangible, intangible, opex, asr, lbt, cost_of_sales = get_setup_dict(data=data)

    contract = CostRecovery(
        start_date=start_date,
        end_date=end_date,
        oil_onstream_date=oil_onstream_date,
        gas_onstream_date=gas_onstream_date,
        lifting=lifting,
        capital_cost=tangible,
        intangible_cost=intangible,
        opex=opex,
        asr_cost=asr,
        lbt_cost=lbt,
        cost_of_sales=cost_of_sales,
        oil_ftp_is_available=data['costrecovery']['oil_ftp_is_available'],
        oil_ftp_is_shared=data['costrecovery']['oil_ftp_is_shared'],
        oil_ftp_portion=convert_to_float(target=data['costrecovery']['oil_ftp_portion']),
        gas_ftp_is_available=data['costrecovery']['gas_ftp_is_available'],
        gas_ftp_is_shared=data['costrecovery']['gas_ftp_is_shared'],
        gas_ftp_portion=convert_to_float(target=data['costrecovery']['gas_ftp_portion']),
        tax_split_type=convert_str_to_taxsplit(str_object=data['costrecovery']['tax_split_type']),
        condition_dict=data['costrecovery']['condition_dict'],
        indicator_rc_icp_sliding=convert_list_to_array_float(
            data_list=data['costrecovery']['indicator_rc_icp_sliding']),
        oil_ctr_pretax_share=convert_to_float(target=data['costrecovery']['oil_ctr_pretax_share']),
        gas_ctr_pretax_share=convert_to_float(target=data['costrecovery']['gas_ctr_pretax_share']),
        oil_ic_rate=convert_to_float(target=data['costrecovery']['oil_ic_rate']),
        gas_ic_rate=convert_to_float(target=data['costrecovery']['gas_ic_rate']),
        ic_is_available=data['costrecovery']['ic_is_available'],
        oil_cr_cap_rate=convert_to_float(target=data['costrecovery']['oil_cr_cap_rate']),
        gas_cr_cap_rate=convert_to_float(target=data['costrecovery']['gas_cr_cap_rate']),
        oil_dmo_volume_portion=convert_to_float(target=data['costrecovery']['oil_dmo_volume_portion']),
        oil_dmo_fee_portion=convert_to_float(target=data['costrecovery']['oil_dmo_fee_portion']),
        oil_dmo_holiday_duration=data['costrecovery']['oil_dmo_holiday_duration'],
        gas_dmo_volume_portion=convert_to_float(target=data['costrecovery']['gas_dmo_volume_portion']),
        gas_dmo_fee_portion=convert_to_float(target=data['costrecovery']['gas_dmo_fee_portion']),
        gas_dmo_holiday_duration=data['costrecovery']['gas_dmo_holiday_duration'],
        oil_carry_forward_depreciation=data['costrecovery']['oil_carry_forward_depreciation'],
        gas_carry_forward_depreciation=data['costrecovery']['gas_carry_forward_depreciation'],
    )

    # Filling the arguments of the contract with the data input
    contract_arguments_dict = {
        "sulfur_revenue": convert_str_to_otherrevenue(str_object=data['contract_arguments']['sulfur_revenue']),
        "electricity_revenue": convert_str_to_otherrevenue(
            str_object=data['contract_arguments']['electricity_revenue']),
        "co2_revenue": convert_str_to_otherrevenue(str_object=data['contract_arguments']['co2_revenue']),
        "is_dmo_end_weighted": data['contract_arguments']['is_dmo_end_weighted'],
        "tax_regime": convert_str_to_taxregime(str_object=data['contract_arguments']['tax_regime']),
        "effective_tax_rate": convert_list_to_array_float_or_array_or_none(data_list=data['contract_arguments']['effective_tax_rate']),
        "ftp_tax_regime": convert_str_to_ftptaxregime(str_object=data['contract_arguments']['ftp_tax_regime']),
        "sunk_cost_reference_year": data['contract_arguments']['sunk_cost_reference_year'],
        "depr_method": convert_str_to_depremethod(str_object=data['contract_arguments']['depr_method']),
        "decline_factor": data['contract_arguments']['decline_factor'],
        "vat_rate": convert_list_to_array_float_or_array(data_input=data['contract_arguments']['vat_rate']),
        "inflation_rate": convert_list_to_array_float_or_array(data_input=data['contract_arguments']['inflation_rate']),
        "inflation_rate_applied_to": convert_str_to_inflationappliedto(str_object=data['contract_arguments']['inflation_rate_applied_to']),
        "post_uu_22_year2001": True if 'post_uu_22_year2001' not in data['contract_arguments'] else
        data['contract_arguments']['post_uu_22_year2001'],
        "oil_cost_of_sales_applied": False if "oil_cost_of_sales_applied" not in data["contract_arguments"] else
        data["contract_arguments"]["oil_cost_of_sales_applied"],
        "gas_cost_of_sales_applied": False if "gas_cost_of_sales_applied" not in data["contract_arguments"] else
        data["contract_arguments"]["gas_cost_of_sales_applied"],
        "sum_undepreciated_cost": False if 'sum_undepreciated_cost' not in data['contract_arguments'] else
        data['contract_arguments']['sum_undepreciated_cost'],
    }

    # Running the contract
    contract.run(**contract_arguments_dict)

    # Filling the summary arguments
    summary_arguments_dict = get_summary_dict(data=data)
    summary_arguments_dict['contract'] = contract
    return get_summary(**summary_arguments_dict)

def get_grosssplit(data: dict):
    """
    The function to get the Summary, Gross Split object, contract arguments, and summary arguments used.

    Parameters
    ----------
    data: dict
        The dictionary of the data input.

    Returns
    -------
    summary_skk: dict
        The executive summary of the contract.
    """
    start_date, end_date, oil_onstream_date, gas_onstream_date, lifting, tangible, intangible, opex, asr, lbt, cost_of_sales = (
        get_setup_dict(data=data))

    contract = GrossSplit(
        start_date=start_date,
        end_date=end_date,
        oil_onstream_date=oil_onstream_date,
        gas_onstream_date=gas_onstream_date,
        lifting=lifting,
        capital_cost=tangible,
        intangible_cost=intangible,
        opex=opex,
        asr_cost=asr,
        lbt_cost=lbt,
        field_status=data['grosssplit']['field_status'],
        field_loc=data['grosssplit']['field_loc'],
        res_depth=data['grosssplit']['res_depth'],
        infra_avail=data['grosssplit']['infra_avail'],
        res_type=data['grosssplit']['res_type'],
        api_oil=data['grosssplit']['api_oil'],
        domestic_use=data['grosssplit']['domestic_use'],
        prod_stage=data['grosssplit']['prod_stage'],
        co2_content=data['grosssplit']['co2_content'],
        h2s_content=data['grosssplit']['h2s_content'],
        base_split_ctr_oil=convert_to_float(target=data['grosssplit']['base_split_ctr_oil']),
        base_split_ctr_gas=convert_to_float(target=data['grosssplit']['base_split_ctr_gas']),
        split_ministry_disc=convert_to_float(target=data['grosssplit']['split_ministry_disc']),
        oil_dmo_volume_portion=convert_to_float(target=data['grosssplit']['oil_dmo_volume_portion']),
        oil_dmo_fee_portion=convert_to_float(target=data['grosssplit']['oil_dmo_fee_portion']),
        oil_dmo_holiday_duration=data['grosssplit']['oil_dmo_holiday_duration'],
        gas_dmo_volume_portion=convert_to_float(target=data['grosssplit']['gas_dmo_volume_portion']),
        gas_dmo_fee_portion=convert_to_float(target=data['grosssplit']['gas_dmo_fee_portion']),
        gas_dmo_holiday_duration=data['grosssplit']['gas_dmo_holiday_duration'],
        oil_carry_forward_depreciation=data['grosssplit']['oil_carry_forward_depreciation'],
        gas_carry_forward_depreciation=data['grosssplit']['gas_carry_forward_depreciation'],
    )

    # Filling the arguments of the contract with the data input
    contract_arguments_dict = {
        "sulfur_revenue": convert_str_to_otherrevenue(str_object=data['contract_arguments']['sulfur_revenue']),
        "electricity_revenue": convert_str_to_otherrevenue(
            str_object=data['contract_arguments']['electricity_revenue']),
        "co2_revenue": convert_str_to_otherrevenue(str_object=data['contract_arguments']['co2_revenue']),
        "is_dmo_end_weighted": data['contract_arguments']['is_dmo_end_weighted'],
        "tax_regime": convert_str_to_taxregime(str_object=data['contract_arguments']['tax_regime']),
        "effective_tax_rate": convert_list_to_array_float_or_array_or_none(data_list=data['contract_arguments']['effective_tax_rate']),
        "sunk_cost_reference_year": data['contract_arguments']['sunk_cost_reference_year'],
        "depr_method": convert_str_to_depremethod(str_object=data['contract_arguments']['depr_method']),
        "decline_factor": data['contract_arguments']['decline_factor'],
        "vat_rate": convert_list_to_array_float_or_array(data_input=data['contract_arguments']['vat_rate']),
        "inflation_rate": convert_list_to_array_float_or_array(data_input=data['contract_arguments']['inflation_rate']),
        "inflation_rate_applied_to": convert_str_to_inflationappliedto(
            str_object=data['contract_arguments']['inflation_rate_applied_to']),
        "cum_production_split_offset": convert_list_to_array_float_or_array(data_input=data["contract_arguments"]["cum_production_split_offset"]),
        "amortization": data["contract_arguments"]["amortization"],
        "regime": convert_grosssplitregime_to_enum(target=data["contract_arguments"]["regime"]),
        "sum_undepreciated_cost": False if 'sum_undepreciated_cost' not in data['contract_arguments'] else
        data['contract_arguments']['sum_undepreciated_cost'],
    }

    # Running the contract
    contract.run(**contract_arguments_dict)

    # Filling the summary arguments
    summary_arguments_dict = get_summary_dict(data=data)
    summary_arguments_dict['contract'] = contract
    return get_summary(**summary_arguments_dict)

def get_transition(data: dict):
    """
    The function to get the Summary, Transition object, contract arguments, and summary arguments used.

    Parameters
    ----------
    data: dict
        The dictionary of the data input.

    Returns
    -------
    summary_skk: dict
        The executive summary of the contract.
    """
    # Defining contract_1
    if data['contract_1']['costrecovery'] is not None and data['contract_1']['grosssplit'] is None:
        _, contract_1, contract_arguments_1, _ = get_costrecovery(data=data['contract_1'], summary_result=False)

    elif data['contract_1']['grosssplit'] is not None and data['contract_1']['costrecovery'] is None:
        _, contract_1, contract_arguments_1, _ = get_grosssplit(data=data['contract_1'], summary_result=False)

    else:
        raise MonteCarloException("Contract type is not recognized")

    # Defining contract_2
    if data['contract_2']['costrecovery'] is not None and data['contract_2']['grosssplit'] is None:
        _, contract_2, contract_arguments_2, _ = get_costrecovery(data=data['contract_2'], summary_result=False)

    elif data['contract_2']['grosssplit'] is not None and data['contract_2']['costrecovery'] is None:
        _, contract_2, contract_arguments_2, _ = get_grosssplit(data=data['contract_2'], summary_result=False)

    else:
        raise MonteCarloException("Contract type is not recognized")

    # generating the transition contract object
    contract = Transition(contract1=contract_1,
                          contract2=contract_2,
                          argument_contract1=contract_arguments_1,
                          argument_contract2=contract_arguments_2, )

    # Generating the transition contract arguments
    contract_arguments_dict = data['contract_arguments']

    # Running the transition contract
    contract.run(**contract_arguments_dict)

    # Filling the summary arguments
    summary_arguments_dict = get_summary_dict(data=data)
    summary_arguments_dict['contract'] = contract

    return get_summary(**summary_arguments_dict)


def get_multipliers_montecarlo(
    run_number: int,
    distribution: str,
    min_value: float,
    mean_value: float,
    max_value: float,
    std_dev: float,
) -> np.ndarray:
    """
    Generate an array of multipliers for Monte Carlo simulation based on the specified distribution.

    Parameters
    ----------
    run_number: int
        Number of runs.
    distribution: str
        Type of distribution ("Uniform", "Triangular", or "Normal").
    min_value: float
        Minimum value for the distribution.
    mean_value: float
        Mean (or central value) for the distribution.
    max_value: float
        Maximum value for the distribution.
    std_dev: float
        Standard deviation for the normal distribution.

    Returns
    -------
    multipliers: np.ndarray
        Array of multipliers generated using Monte Carlo simulation.

    Notes
    -----
    - For "Uniform" distribution, the function uses a uniform random variable.
    - For "Triangular" distribution, the function uses a triangular random variable.
    - For "Normal" distribution, the function uses a truncated normal random variable.
    """
    # Uniform distribution
    if distribution == "Uniform":
        # Modify minimum and maximum values
        params = [min_value, max_value]
        attrs_updated = {
            key: float(params[i] / mean_value)
            for i, key in enumerate(["min_value", "max_value"])
        }

        # Determine multipliers
        multipliers = uniform.rvs(
            loc=attrs_updated["min_value"],
            scale=attrs_updated["max_value"] - attrs_updated["min_value"],
            size=run_number,
        )

    # Triangular distribution
    elif distribution == "Triangular":
        # Modify minimum and maximum values
        params = [min_value, mean_value, max_value]
        attrs_updated = {
            key: float(params[i] / mean_value)
            for i, key in enumerate(["min_value", "mean_value", "max_value"])
        }

        # Determine mode (central point)
        c = (
            (attrs_updated["mean_value"] - attrs_updated["min_value"]) /
            (attrs_updated["max_value"] - attrs_updated["min_value"])
        )

        # Determine multipliers
        multipliers = triang.rvs(
            c=c,
            loc=attrs_updated["min_value"],
            scale=attrs_updated["max_value"] - attrs_updated["min_value"],
            size=run_number,
        )

    # Normal distribution
    elif distribution == "Normal":
        # Modify minimum and maximum values
        params = [min_value, mean_value, max_value, std_dev]
        attrs_updated = {
            key: float(params[i] / mean_value)
            for i, key in enumerate(["min_value", "mean_value", "max_value", "std_dev"])
        }

        # Determine z-values
        zvalues = {
            key: float(
                (attrs_updated[key] - attrs_updated["mean_value"]) / attrs_updated["std_dev"]
            )
            for key in ["min_value", "mean_value", "max_value"]
        }

        # Determine multipliers
        multipliers_init = truncnorm.rvs(
            a=zvalues["min_value"],
            b=zvalues["max_value"],
            loc=zvalues["mean_value"],
            scale=1,
            size=run_number,
        )

        multipliers = (multipliers_init * attrs_updated["std_dev"]) + attrs_updated["mean_value"]

    else:
        raise MonteCarloException(
            f"The type of distribution is unavailable. "
            f"Please select type of distribution between: "
            f"(i) Uniform, (ii) Triangular, or (iii) Normal."
        )

    return multipliers

def min_mean_max_retriever(
        contract: BaseProject | CostRecovery | GrossSplit | Transition,
        verbose: bool = False
):
    """
    Function to get the value of min, max and stddev of each montecarlo elements.

    Parameters
    ----------
    contract : BaseProject | CostRecovery | GrossSplit | Transition
        The observed contract object
    verbose: bool
        The option to print the min, mean and max.

    Returns
    -------
    out: dict
    """
    # Retrieve Min, Mean, and Max for CAPEX
    min_capex = np.min(np.concatenate(
        (contract.capital_cost_total.cost, contract.intangible_cost_total.cost)))
    mean_capex = np.average(np.concatenate(
        (contract.capital_cost_total.cost, contract.intangible_cost_total.cost)))
    max_capex = np.max(np.concatenate(
        (contract.capital_cost_total.cost, contract.intangible_cost_total.cost)))

    # Retrieve Min, Mean, and Max for OPEX
    min_opex = np.min(np.concatenate(
        (contract.opex_total.cost, contract.asr_cost_total.cost,
         contract.lbt_cost_total.cost, contract.cost_of_sales_total.cost)))
    mean_opex = np.average(np.concatenate(
        (contract.opex_total.cost, contract.asr_cost_total.cost,
         contract.lbt_cost_total.cost, contract.cost_of_sales_total.cost)))
    max_opex = np.max(np.concatenate(
        (contract.opex_total.cost, contract.asr_cost_total.cost,
         contract.lbt_cost_total.cost, contract.cost_of_sales_total.cost)))

    # Retrieve Min, Mean, and Max for Oil Price
    min_oil_price = np.min(contract._oil_lifting.price)
    mean_oil_price = np.average(contract._oil_lifting.price)
    max_oil_price = np.max(contract._oil_lifting.price)

    # Retrieve Min, Mean, Max for Gas Price
    min_gas_price = np.min(contract._gas_lifting.price)
    mean_gas_price = np.average(contract._gas_lifting.price)
    max_gas_price = np.max(contract._gas_lifting.price)

    # Retrieve Min, Mean, Max for Oil Lifting
    min_lifting = np.min([
        np.min(contract._oil_lifting.lifting_rate),
        np.min(contract._gas_lifting.lifting_rate)
    ])
    mean_lifting = np.mean([
        np.mean(contract._oil_lifting.lifting_rate),
        np.mean(contract._gas_lifting.lifting_rate)
    ])
    max_lifting = np.max([
        np.max(contract._oil_lifting.lifting_rate),
        np.max(contract._gas_lifting.lifting_rate)
    ])

    result =  {
        'min_capex': min_capex,
        'mean_capex': mean_capex,
        'max_capex': max_capex,
        'min_opex': min_opex,
        'mean_opex': mean_opex,
        'max_opex': max_opex,
        'min_oil_price': min_oil_price,
        'mean_oil_price': mean_oil_price,
        'max_oil_price': max_oil_price,
        'min_gas_price': min_gas_price,
        'mean_gas_price': mean_gas_price,
        'max_gas_price': max_gas_price,
        'min_lifting': min_lifting,
        'mean_lifting': mean_lifting,
        'max_lifting': max_lifting,
    }

    if verbose is True:
        print('Parameter used:')
        for key, value in result.items():
            print(key,': ', value)
        print('')

    return result



class ProcessMonte:
    target = ["npv", "irr", "pi", "pot", "gov_take", "ctr_net_share"]

    def __init__(self, type, contract, numSim, params):
        self.type = type
        self.numSim = numSim
        self.baseContract = contract
        self.parameter = params
        self.hasGas = False
        for i in range(len(self.parameter)):
            if self.parameter[i]["id"] == 1:
                self.hasGas = True
                break

        # Get multipliers
        self.multipliers = np.ones([self.numSim, len(self.parameter)], dtype=np.float64)

        for i in range(len(self.parameter)):
            self.multipliers[:, i] = get_multipliers_montecarlo(
                run_number=self.numSim,
                distribution=(
                    self.parameter[i]["dist"].value
                ),
                min_value=self.parameter[i]["min"],
                mean_value=self.parameter[i]["base"],
                max_value=self.parameter[i]["max"],
                std_dev=self.parameter[i]["stddev"],
            )

    def Adjust_Data(self, multipliers: np.ndarray):
        Adj_Contract = copy.deepcopy(self.baseContract)

        def Adj_Partial_Data(
            contract_: dict, par: str, key: str, multiplier: float, datakeys: list = []
        ):
            for item_key in contract_[key].keys():
                item = contract_[key][item_key]
                if (
                    par == "Lifting"
                    and key == "lifting"
                    and item["fluid_type"] == "Gas"
                ):
                    continue
                if key == "lifting":
                    if (
                        (par == "Oil Price" and item["fluid_type"] == "Oil")
                        or (par == "Gas Price" and item["fluid_type"] == "Gas")
                        or par == "Lifting"
                    ):
                        lifting_key = (
                            ["price"]
                            if par == "Oil Price" or par == "Gas Price"
                            else ["lifting_rate", "prod_rate"]
                        )
                        for lift_keys in lifting_key:
                            item[lift_keys] = (
                                np.array(item[lift_keys]) * multiplier
                            ).tolist()
                else:
                    for data_key in datakeys:
                        item[data_key] = (
                            np.array(item[data_key]) * multiplier
                        ).tolist()

        # for iloop in range(2 if self.type >= 3 else 1):
        contract_ = (
            # Adj_Contract if self.type < 3 else Adj_Contract[f"contract_{iloop+1}"]
            Adj_Contract if self.type < 3 else Adj_Contract[f"contract_{2}"]
        )
        for i in range(len(self.parameter)):
            # OIl
            if self.parameter[i]["id"] == 0:
                Adj_Partial_Data(contract_, "Oil Price", "lifting", multipliers[i])
            elif self.parameter[i]["id"] == 1:
                Adj_Partial_Data(contract_, "Gas Price", "lifting", multipliers[i])
            elif self.parameter[i]["id"] == 2:
                Adj_Partial_Data(
                    contract_,
                    "OPEX",
                    "opex",
                    multipliers[i],
                    ["fixed_cost", "cost_per_volume"],
                )
                Adj_Partial_Data(
                    contract_,
                    "ASR",
                    "asr",
                    multipliers[i],
                    ["cost"],
                )
                Adj_Partial_Data(
                    contract_,
                    "LBT",
                    "lbt",
                    multipliers[i],
                    ["cost"],
                )
                Adj_Partial_Data(
                    contract_,
                    "COS",
                    "cost_of_sales",
                    multipliers[i],
                    ["cost"],
                )
            elif self.parameter[i]["id"] == 3:
                Adj_Partial_Data(
                    contract_, "CAPEX", "capital", multipliers[i], ["cost"]
                )
                Adj_Partial_Data(
                    contract_, "CAPEX", "intangible", multipliers[i], ["cost"]
                )
            elif self.parameter[i]["id"] == 4:
                Adj_Partial_Data(contract_, "Lifting", "lifting", multipliers[i])

        return Adj_Contract

    def calcContract(self, n: int):
        try:
            print(f"Monte Progress:{n}", flush=True)
            # time.sleep(100)

            dataAdj = self.Adjust_Data(self.multipliers[n, :])
            csummary = (
                get_costrecovery(data=dataAdj)
                if self.type == 1
                else (
                    get_grosssplit(data=dataAdj)
                    if self.type == 2
                    else (
                        get_transition(data=dataAdj)
                        if self.type >= 3
                        else get_baseproject(data=dataAdj)
                    )
                )
            )
            del dataAdj
            return {
                "n": n,
                "output": (
                    csummary["ctr_npv"],
                    csummary["ctr_irr"],
                    csummary["ctr_pi"],
                    csummary["ctr_pot"],
                    csummary["gov_take"],
                    csummary["ctr_net_share"],
                ),
            }
        except Exception as err:
            print(f"Error: {err}")
            return {
                "n": n,
                "output": (
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            }

    def calculate(self):
        results = np.zeros(
            [self.numSim, len(self.target) + len(self.parameter)], dtype=np.float64
        )

        # # Execute MonteCarlo simulation
        # client = Client()
        # b = db.from_sequence(range(self.numSim), partition_size=100)
        # futures = b.map(self.calcContract).compute()
        # # print(futures)
        # for res in futures:
        #     # for res in outcalcmonte.get():
        #     results[res["n"], 0 : len(self.target)] = res["output"]
        #     results[res["n"], len(self.target) :] = [
        #         self.multipliers[res["n"], index] * item["base"]
        #         for index, item in enumerate(self.parameter)
        #     ]
        #
        # client.close()

        # Use ProcessPoolExecutor for parallel execution
        # import concurrent.futures
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     futures = [executor.submit(self.calcContract, n) for n in range(self.numSim)]
        #     for future in concurrent.futures.as_completed(futures):
        #         res = future.result()
        #         results[res["n"], 0: len(self.target)] = res["output"]
        #         results[res["n"], len(self.target):] = [
        #             self.multipliers[res["n"], index] * item["base"]
        #             for index, item in enumerate(self.parameter)
        #         ]

        # Execute MonteCarlo simulation using pathos multiprocessing
        from pathos.multiprocessing import ProcessingPool as Pool
        with Pool() as pool:
            futures = pool.map(self.calcContract, range(self.numSim))

        for res in futures:
            results[res["n"], 0: len(self.target)] = res["output"]
            results[res["n"], len(self.target):] = [
                self.multipliers[res["n"], index] * item["base"]
                for index, item in enumerate(self.parameter)
            ]

        # Sorted the results
        results_sorted = np.take_along_axis(
            arr=results,
            indices=np.argsort(results, axis=0),
            axis=0,
        )
        # Specify probability
        prob = np.arange(1, self.numSim + 1, dtype=np.float64) / self.numSim

        # Arrange the results
        results_arranged = np.concatenate((prob[:, np.newaxis], results_sorted), axis=1)

        # Calculate P10, P50, P90
        percentiles = np.percentile(
            a=results_arranged,
            q=[10, 50, 90],
            method="higher",
            axis=0,
        )

        # Determine indices of data
        indices = np.linspace(0, self.numSim, 101)[0:-1].astype(int)
        indices[0] = 1
        if indices[-1] != self.numSim - 1:
            indices = np.append(indices, int(self.numSim - 1))

        # Final outcomes
        outcomes = {
            "params": (
                ["Oil Price", "Gas Price", "Opex", "Capex", "Cum. prod."]
                if self.hasGas
                else ["Oil Price", "Opex", "Capex", "Cum. prod."]
            ),
            "results": results_arranged[indices, :].tolist(),
            "P10": percentiles[0, :].tolist(),
            "P50": percentiles[1, :].tolist(),
            "P90": percentiles[2, :].tolist(),
        }

        return outcomes

def uncertainty_psc(
        contract: BaseProject | CostRecovery | GrossSplit | Transition,
        contract_arguments: dict,
        summary_arguments: dict,
        run_number: int = 10,
        min_oil_price: float = None,
        mean_oil_price: float = None,
        max_oil_price: float = None,
        min_gas_price: float = None,
        mean_gas_price: float = None,
        max_gas_price: float = None,
        min_opex: float = None,
        mean_opex: float = None,
        max_opex: float = None,
        min_capex: float = None,
        mean_capex: float = None,
        max_capex: float = None,
        min_lifting: float = None,
        mean_lifting: float = None,
        max_lifting: float = None,
        oil_price_stddev: float = 1.25,
        gas_price_stddev: float = 1.25,
        opex_stddev: float = 1.25,
        capex_stddev: float = 1.25,
        lifting_stddev: float = 1.25,
        oil_price_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL,
        gas_price_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL,
        opex_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL,
        capex_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL,
        lifting_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL,
        verbose: bool = True,
):
    # Translating the contract type before parsing into ProcessMonte class
    if isinstance(contract, CostRecovery):
        contract_type = 1
    elif isinstance(contract, GrossSplit):
        contract_type = 2
    elif isinstance(contract, Transition):
        contract_type = 3
    else:
        contract_type = 0

    # Retrieve the Min Max and Average Values
    min_max_mean_original = min_mean_max_retriever(contract=contract, verbose=verbose)

    min_max_mean_std = {
        'min_oil_price': min_oil_price,
        'mean_oil_price': mean_oil_price,
        'max_oil_price': max_oil_price,
        'min_gas_price': min_gas_price,
        'mean_gas_price': mean_gas_price,
        'max_gas_price': max_gas_price,
        'min_opex': min_opex,
        'mean_opex': mean_opex,
        'max_opex': max_opex,
        'min_capex': min_capex,
        'mean_capex': mean_capex,
        'max_capex': max_capex,
        'min_lifting': min_lifting,
        'mean_lifting': mean_lifting,
        'max_lifting': max_lifting,
    }

    # If the element is not being input from argument, fill with original value
    for element in min_max_mean_std.keys():
        if min_max_mean_std[element] is None:
            min_max_mean_std[element] = min_max_mean_original[element]

    # Iterate over the dictionary
    for key in list(min_max_mean_std.keys()):
        if key.startswith("min_"):
            # Base key (e.g., 'capex', 'opex', etc.)
            base = key[4:]
            min_key = f"min_{base}"
            mean_key = f"mean_{base}"
            max_key = f"max_{base}"

            # Check if min, mean, and max values are the same
            if (
                    min_key in min_max_mean_std and mean_key in min_max_mean_std and max_key in min_max_mean_std and
                    min_max_mean_std[min_key] == min_max_mean_std[mean_key] == min_max_mean_std[max_key]
            ):
                # Adjust min and max values
                min_max_mean_std[min_key] = 0.8 * min_max_mean_std[min_key]  # Set min to 0.8 of the min
                min_max_mean_std[max_key] = 1.2 * min_max_mean_std[max_key]  # Set max to 1.2 of the max

    parameter= [
        {"id": 0, "dist": oil_price_distribution, "min": min_max_mean_std['min_oil_price'], "base": min_max_mean_std['mean_oil_price'], "max": min_max_mean_std['max_oil_price'], "stddev": oil_price_stddev},
        {"id": 1, "dist": gas_price_distribution, "min": min_max_mean_std['min_gas_price'], "base": min_max_mean_std['mean_gas_price'], "max": min_max_mean_std['max_gas_price'],"stddev": gas_price_stddev},
        {"id": 2, "dist": opex_distribution, "min": min_max_mean_std['min_opex'], "base": min_max_mean_std['mean_opex'], "max": min_max_mean_std['max_opex'],"stddev": opex_stddev},
        {"id": 3, "dist": capex_distribution, "min": min_max_mean_std['min_capex'], "base": min_max_mean_std['mean_capex'], "max": min_max_mean_std['max_capex'],"stddev": capex_stddev},
        {"id": 4, "dist": lifting_distribution, "min": min_max_mean_std['min_lifting'], "base": min_max_mean_std['mean_lifting'], "max": min_max_mean_std['max_lifting'],"stddev": lifting_stddev}
    ]

    # Condition when there is no gas produced
    fluid_produced = [lift.fluid_type.value for lift in contract.lifting]
    if FluidType.GAS not in fluid_produced:
        del parameter[1]

    # Constructing the contract key
    contract_dict = get_contract_attributes(
        contract=contract,
        contract_arguments=contract_arguments,
        summary_arguments=summary_arguments
    )

    # Executing the montecarlo
    monte = ProcessMonte(
        contract_type,
        contract_dict,
        run_number,
        parameter,
    )

    return monte.calculate()
