"""
Collection of functions to administer sensitivity analysis.
"""

import numpy as np

from pyscnomics.econ import CostOfSales, Lifting, FluidType
from pyscnomics.io.aggregator import Aggregate
from pyscnomics.optimize.adjuster import AdjustData

from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts.transition import Transition


def get_multipliers_sensitivity(
    min_deviation: float,
    max_deviation: float,
    base_value: float = 1.0,
    step: int = 10,
    number_of_params: int = 5,
) -> np.ndarray:
    """
    Generate multipliers for different economic parameters within a specified range
    for sensitivity study.

    Parameters
    ----------
    min_deviation: float
        The minimum deviation from the base value.
    max_deviation: float
        The maximum deviation from the base value.
    base_value: float, optional
        The base value for the multipliers. Default is 1.0.
    step: int, optional
        The number of steps to create multipliers. Default is 10.
    number_of_params: int, optional
        The number of parameters to vary in sensitivity analysis. Default is 5.

    Returns
    -------
    multipliers: np.ndarray
        A 3D NumPy array containing multipliers for different economic factors.
    """
    # Specify the minimum and maximum values
    min_val = base_value - min_deviation
    max_val = base_value + max_deviation

    min_multipliers = np.linspace(min_val, base_value, step + 1)
    max_multipliers = np.linspace(base_value, max_val, step + 1)
    tot_multipliers = np.concatenate((min_multipliers, max_multipliers[1:]))

    # Specify array multipliers
    multipliers = (
        np.ones(
            [number_of_params, len(tot_multipliers), number_of_params],
            dtype=np.float_,
        )
    )

    for i in range(number_of_params):
        multipliers[i, :, i] = tot_multipliers.copy()

    # Specify total number of run
    total_run = len(tot_multipliers) * number_of_params

    return multipliers, total_run


def run_sensitivity(
    contract: BaseProject | CostRecovery | GrossSplit | Transition,
    contract_arguments: dict,
    summary_arguments: dict,
) -> tuple:
    """
    Function to run simulation in Sensitivity mode.

    Parameters
    ----------
    contract: BaseProject | CostRecovery | GrossSplit | Transition
        The contract object that will be run.
    contract_arguments: dict
        The contract arguments.
    summary_arguments: dict
        The dictionary of the summary arguments.

    Returns
    -------
    tuple:
        A tuple containing the following project metrics:
        - NPV (Net Present Value)
        - IRR (Internal Rate of Return)
        - PI (Profitability Index)
        - POT (Pay Out Time)
        - Gov Take (Government Take)
        - Net Share (Contractor's Net Share)
    """
    contract.run(**contract_arguments)
    contract_summary = get_summary(**summary_arguments)

    return (
        contract_summary["ctr_npv"],
        contract_summary["ctr_irr"],
        contract_summary["ctr_pi"],
        contract_summary["ctr_pot"],
        contract_summary["gov_take"],
        contract_summary["ctr_net_share"],
    )


def execute_sensitivity(
    data: Aggregate,
    target: list,
    multipliers: np.ndarray,
    workbook_path: str,
) -> dict[str, np.ndarray]:
    """
    Run sensitivity analysis in a serial manner.

    Parameters
    ----------
    data: Aggregate
        The aggregate data for the analysis.
    target: list
        The target objective variable.
    workbook_path: str
        The path location of the associated Excel file.
    multipliers: np.ndarray
        A 3D array of multipliers for sensitivity analysis.

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary containing sensitivity analysis results. The keys correspond to
        different parameters, and the values are arrays of results for each multiplier.
    """
    args = ["psc", "psc_arguments", "summary_arguments"]

    # Create containers for the generated project instances
    params = {
        par: {
            arg: [0] * multipliers.shape[1] for arg in args
        }
        for par in data.sensitivity_data.parameter
    }

    # Fill "params" with the associated contract instances
    for i, par in enumerate(data.sensitivity_data.parameter):
        for j in range(multipliers.shape[1]):
            (
                params[par]["psc"][j],
                params[par]["psc_arguments"][j],
                params[par]["summary_arguments"][j],
            ) = AdjustData(
                data=data,
                workbook_path=workbook_path,
                multipliers=multipliers[i, j, :],
            ).activate()

    # Create a container for calculation results
    results = {
        par: np.zeros([multipliers.shape[1], len(target)], dtype=np.float_)
        for par in data.sensitivity_data.parameter
    }

    # Run the simulations to obtain the results (NPV, IRR, PI, POT, GOV_TAKE, CTR_NET_SHARE)
    for par in data.sensitivity_data.parameter:
        for j in range(multipliers.shape[1]):
            results[par][j, :] = run_sensitivity(
                contract=params[par]["psc"][j],
                contract_arguments=params[par]["psc_arguments"][j],
                summary_arguments=params[par]["summary_arguments"][j],
            )

    return results

################################################ Sensitivity Detached ################################################
"""
This Code is made to detached the sensitivity module in the previous version which depended with the excel 
into fully able to be run in python. 
"""
from pyscnomics.econ.costs import CapitalCost, Intangible, OPEX, ASR, LBT
from pyscnomics.tools.summary import get_summary

import pandas as pd


class SensitivityException(Exception):
    """Exception to raise for a misuse of Sensitivity Method"""
    pass


def _get_multipliers(
        min_deviation: float,
        max_deviation: float,
        base_value: float = 1.0,
        step: int = 10,
) -> np.ndarray:
    """
    Generate multipliers for different economic parameters within a specified range
    for sensitivity study.

    Parameters
    ----------
    min_deviation: float
        The minimum deviation from the base value.
    max_deviation: float
        The maximum deviation from the base value.
    base_value: float, optional
        The base value for the multipliers. Default is 1.0.
    step: int, optional
        The number of steps to create multipliers. Default is 10.
    Returns
    -------
    multipliers: np.ndarray
        A NumPy array containing multipliers for different economic factors.
    """
    # Specify the minimum and maximum values
    min_val = base_value - min_deviation
    max_val = base_value + max_deviation

    min_multipliers = np.linspace(min_val, base_value, step + 1)
    max_multipliers = np.linspace(base_value, max_val, step + 1)
    return np.concatenate((min_multipliers, max_multipliers[1:]))

def _adjust_element_single_contract(
        contract: BaseProject | CostRecovery | GrossSplit | Transition,
        contract_arguments: dict,
        element: str,
        adjustment_value: float,
) -> CostRecovery | GrossSplit | Transition:
    """
    The function to adjust the element of the contract based on the adjustment_value.

    Parameters
    ----------
    contract: BaseProject | CostRecovery | GrossSplit | Transition
        The contract which will be adjusted.
    contract_arguments: dict
        The contract arguments.
    element: str
        The element of the contract that will be adjusted.
    adjustment_value: float
        The adjustment value which will be multiplied by the corresponding element.

    Returns
    -------
    contract_adjusted: CostRecovery | GrossSplit | Transition
        The contract that has been adjusted.
    """

    capital_adjusted = contract.capital_cost
    intangible_adjusted = contract.intangible_cost
    opex_adjusted = contract.opex
    asr_adjusted = contract.asr_cost
    lbt_adjusted = contract.lbt_cost
    cos_adjusted = contract.cost_of_sales
    lifting_adjusted = contract.lifting

    if element == 'CAPEX':
        # Adjusting the Capital Cost of the contract
        capital_adjusted = tuple([
            CapitalCost(
                start_year=tan.start_year,
                end_year=tan.end_year,
                cost=tan.cost * adjustment_value,
                expense_year=tan.expense_year,
                cost_allocation=tan.cost_allocation,
                description=tan.description,
                tax_portion=tan.tax_portion,
                tax_discount=tan.tax_discount,
                pis_year=tan.pis_year,
                salvage_value=tan.salvage_value,
                useful_life=tan.useful_life,
                depreciation_factor=tan.depreciation_factor,
                is_ic_applied=tan.is_ic_applied,
            ) for tan in contract.capital_cost
        ])

        # Adjusting the Intangible cost of the contract
        intangible_adjusted = tuple([
            Intangible(
                start_year=intang.start_year,
                end_year=intang.end_year,
                cost=intang.cost * adjustment_value,
                expense_year=intang.expense_year,
                cost_allocation=intang.cost_allocation,
                description=intang.description,
                tax_portion=intang.tax_portion,
                tax_discount=intang.tax_discount,

            ) for intang in contract.intangible_cost
        ])

    elif element == 'OPEX':
        # Adjusting the OPEX cost of the contract
        opex_adjusted = tuple([
            OPEX(
                start_year=opx.start_year,
                end_year=opx.end_year,
                expense_year=opx.expense_year,
                cost_allocation=opx.cost_allocation,
                description=opx.description,
                tax_portion=opx.tax_portion,
                tax_discount=opx.tax_discount,
                fixed_cost=opx.fixed_cost * adjustment_value,
                prod_rate=opx.prod_rate,
                cost_per_volume=opx.cost_per_volume,
            ) for opx in contract.opex
        ])

        # Adjusting the ASR cost of the contract
        asr_adjusted = tuple([
            ASR(
                start_year=asr.start_year,
                end_year=asr.end_year,
                cost=asr.cost * adjustment_value,
                expense_year=asr.expense_year,
                cost_allocation=asr.cost_allocation,
                description=asr.description,
                tax_portion=asr.tax_portion,
                tax_discount=asr.tax_discount,
                final_year=asr.final_year,
                future_rate=asr.future_rate,
            ) for asr in contract.asr_cost
        ])

        # Adjusting the LBT of the contract
        lbt_adjusted = tuple([
            LBT(
                start_year=bt.start_year,
                end_year=bt.end_year,
                expense_year=bt.expense_year,
                cost=bt.cost * adjustment_value,
                cost_allocation=bt.cost_allocation,
                description=bt.description,
                tax_portion=bt.tax_portion,
                tax_discount=bt.tax_discount,
                final_year=bt.final_year,
                utilized_land_area=bt.utilized_land_area,
                utilized_building_area=bt.utilized_building_area,
                njop_land=bt.njop_land,
                njop_building=bt.njop_building,
                gross_revenue=bt.gross_revenue,
            ) for bt in contract.lbt_cost
        ])

        # Adjusting the Intangible cost of the contract
        cos_adjusted = tuple([
            CostOfSales(
                start_year=cos.start_year,
                end_year=cos.end_year,
                cost=cos.cost * adjustment_value,
                expense_year=cos.expense_year,
                cost_allocation=cos.cost_allocation,
                description=cos.description,
                tax_portion=cos.tax_portion,
                tax_discount=cos.tax_discount,

            ) for cos in contract.cost_of_sales
        ])

    elif element == 'OILPRICE':
        # Adjusting the oil price
        lifting_adjusted = tuple([
            Lifting(
                start_year=lift.start_year,
                end_year=lift.end_year,
                lifting_rate=lift.lifting_rate,
                price=lift.price * adjustment_value if lift.fluid_type is FluidType.OIL else lift.price,
                prod_year=lift.prod_year,
                fluid_type=lift.fluid_type,
                ghv=lift.ghv,
                prod_rate=lift.prod_rate,
                prod_rate_baseline=lift.prod_rate_baseline,
            )
            for lift in contract.lifting
        ])

    elif element == 'GASPRICE':
        # Adjusting the oil price
        lifting_adjusted = tuple([
            Lifting(
                start_year=lift.start_year,
                end_year=lift.end_year,
                lifting_rate=lift.lifting_rate,
                price=lift.price * adjustment_value if lift.fluid_type is FluidType.GAS else lift.price,
                prod_year=lift.prod_year,
                fluid_type=lift.fluid_type,
                ghv=lift.ghv,
                prod_rate=lift.prod_rate,
                prod_rate_baseline=lift.prod_rate_baseline,
            )
            for lift in contract.lifting
        ])

    elif element == 'LIFTING':
        # Adjusting the lifting
        lifting_adjusted = tuple([
            Lifting(
                start_year=lift.start_year,
                end_year=lift.end_year,
                lifting_rate=lift.lifting_rate * adjustment_value,
                price=lift.price,
                prod_year=lift.prod_year,
                fluid_type=lift.fluid_type,
                ghv=lift.ghv,
                # prod_rate=lift.prod_rate,
                # prod_rate is being by passed in the routine of sensitivity, thus it will be filled with the lifting value
                prod_rate_baseline=lift.prod_rate_baseline,
            )
            for lift in contract.lifting
        ])

    else:
        raise SensitivityException(
            f"The element value, {element}, is not recognized."
        )

    # Adjust the contract with the adjusted element
    # On stream date treatment
    if np.sum(contract._oil_revenue) == 0:
        oil_onstream_date = None
    else:
        oil_onstream_date = contract.oil_onstream_date

    if np.sum(contract._gas_revenue) == 0:
        gas_onstream_date = None
    else:
        gas_onstream_date = contract.gas_onstream_date

    # When the contract is CostRecovery, parsing back the adjusted cost elements to the cost recovery contract
    if isinstance(contract, CostRecovery):
        contract_adjusted = CostRecovery(
            start_date=contract.start_date,
            end_date=contract.end_date,
            oil_onstream_date=oil_onstream_date,
            gas_onstream_date=gas_onstream_date,
            lifting=lifting_adjusted,
            capital_cost=capital_adjusted,
            intangible_cost=intangible_adjusted,
            opex=opex_adjusted,
            asr_cost=asr_adjusted,
            lbt_cost=lbt_adjusted,
            cost_of_sales=cos_adjusted,
            oil_ftp_is_available=contract.oil_ftp_is_available,
            oil_ftp_is_shared=contract.oil_ftp_is_shared,
            oil_ftp_portion=contract.oil_ftp_portion,
            gas_ftp_is_available=contract.gas_ftp_is_available,
            gas_ftp_is_shared=contract.gas_ftp_is_shared,
            gas_ftp_portion=contract.gas_ftp_portion,
            tax_split_type=contract.tax_split_type,
            condition_dict=contract.condition_dict,
            indicator_rc_icp_sliding=contract.indicator_rc_icp_sliding,
            oil_ctr_pretax_share=contract.oil_ctr_pretax_share,
            gas_ctr_pretax_share=contract.gas_ctr_pretax_share,
            oil_ic_rate=contract.oil_ic_rate,
            gas_ic_rate=contract.gas_ic_rate,
            ic_is_available=contract.ic_is_available,
            oil_cr_cap_rate=contract.oil_cr_cap_rate,
            gas_cr_cap_rate=contract.gas_cr_cap_rate,
            oil_dmo_volume_portion=contract.oil_dmo_volume_portion,
            oil_dmo_fee_portion=contract.oil_dmo_fee_portion,
            oil_dmo_holiday_duration=contract.oil_dmo_holiday_duration,
            gas_dmo_volume_portion=contract.gas_dmo_volume_portion,
            gas_dmo_fee_portion=contract.gas_dmo_fee_portion,
            gas_dmo_holiday_duration=contract.gas_dmo_holiday_duration,
        )

    # When the contract is GrossSplit, parsing back the adjusted cost elements to the gross split contract
    elif isinstance(contract, GrossSplit):
        contract_adjusted = GrossSplit(
            start_date=contract.start_date,
            end_date=contract.end_date,
            oil_onstream_date=oil_onstream_date,
            gas_onstream_date=gas_onstream_date,
            lifting=contract.lifting,
            capital_cost=capital_adjusted,
            intangible_cost=intangible_adjusted,
            opex=opex_adjusted,
            asr_cost=asr_adjusted,
            lbt_cost=lbt_adjusted,
            cost_of_sales=cos_adjusted,
            field_status=contract.field_status,
            field_loc=contract.field_loc,
            res_depth=contract.res_depth,
            infra_avail=contract.infra_avail,
            res_type=contract.res_type,
            api_oil=contract.api_oil,
            domestic_use=contract.domestic_use,
            prod_stage=contract.prod_stage,
            co2_content=contract.co2_content,
            h2s_content=contract.h2s_content,
            base_split_ctr_oil=contract.base_split_ctr_oil,
            base_split_ctr_gas=contract.base_split_ctr_gas,
            split_ministry_disc=contract.split_ministry_disc,
            oil_dmo_volume_portion=contract.oil_dmo_volume_portion,
            oil_dmo_fee_portion=contract.oil_dmo_fee_portion,
            oil_dmo_holiday_duration=contract.oil_dmo_holiday_duration,
            gas_dmo_volume_portion=contract.gas_dmo_volume_portion,
            gas_dmo_fee_portion=contract.gas_dmo_fee_portion,
            gas_dmo_holiday_duration=contract.gas_dmo_holiday_duration,
        )

    elif isinstance(contract, BaseProject):
        contract_adjusted = BaseProject(
            start_date=contract.start_date,
            end_date=contract.end_date,
            oil_onstream_date=oil_onstream_date,
            gas_onstream_date=gas_onstream_date,
            lifting=contract.lifting,
            capital_cost=capital_adjusted,
            intangible_cost=intangible_adjusted,
            opex=opex_adjusted,
            asr_cost=asr_adjusted,
            lbt_cost=lbt_adjusted,
            cost_of_sales=cos_adjusted,
        )

    # When the contract is not recognized, raise an exception
    else:
        raise SensitivityException(
            f"The contract type, {type(contract)}, is not recognized."
        )

    contract_adjusted.run(**contract_arguments)
    return contract_adjusted


def sensitivity_psc(
        contract: BaseProject | CostRecovery | GrossSplit | Transition,
        contract_arguments: dict,
        summary_arguments: dict,
        min_deviation: float,
        max_deviation: float,
        base_value: float = 1.0,
        step: int = 10,
        dataframe_output: bool = True
)-> dict | pd.DataFrame:
    """
    The function to get the sensitivity analysis of a contract.

    Parameters
    ----------
    contract: BaseProject | CostRecovery | Gross Split | Transition
        The contract that the sensitivity will be retrieved.
    contract_arguments: dict
        The contract arguments of the contract.
    summary_arguments: dict
        The summary arguments.
    min_deviation: float
        The minimum deviation from the base value.
    max_deviation: float
        The maximum deviation from the base value.
    base_value: float, optional
        The base value for the multipliers. Default is 1.0.
    step: int, optional
        The number of steps to create multipliers. Default is 10.
    dataframe_output: bool
        The option whether the output in a dataframe form or dictionary.
    Returns
    -------
    out: dict | pd.DataFrame
        The sensitivity result
    """
    # Get the multipliers
    multipliers = _get_multipliers(
        min_deviation=min_deviation,
        max_deviation=max_deviation,
        base_value=base_value,
        step=step,
    )

    if isinstance(contract, Transition):
        raise NotImplementedError
    else:
        pass

    # Adjust the element of the contract and contain it in a list
    psc_adjusted_dict = {
        element: {
            mul: _adjust_element_single_contract(
                contract=contract,
                contract_arguments=contract_arguments,
                element=element,
                adjustment_value=mul,
            )
            for mul in multipliers
        }
        for element in ['CAPEX', 'OPEX', 'OILPRICE', 'GASPRICE', 'LIFTING']
    }

    # Get the summary of each contract in psc_adjusted_dict and contain it in a dictionary
    summary_adjusted_dict = {
        element: {
            mul: get_summary(
                **{**summary_arguments, 'contract': psc_adjusted_dict[element][mul]}
            )
            for mul in psc_adjusted_dict[element]
        }
        for element in psc_adjusted_dict.keys()
    }

    # Retrieve the value for NPV, IRR, PI, POT, Gov Take, and Contractor Share
    indicator_list = ['ctr_npv', 'ctr_irr', 'ctr_pi', 'ctr_pot', 'gov_take', 'ctr_net_share',]

    # Transform summary_adjusted_dict into the following structure
    sensitivity_result = {
        indicator: {
            multiplier: {
                element: summary_adjusted_dict[element][multiplier].get(indicator, None)
                for element in summary_adjusted_dict
            }
            for multiplier in summary_adjusted_dict[next(iter(summary_adjusted_dict))]
        }
        for indicator in indicator_list
    }

    # Transform result dictionary into DataFrames
    sensitivity_result_df = {
        indicator: pd.DataFrame.from_dict(
            data, orient='index'
        ).reset_index().rename(columns={'index': 'Factor'}).set_index('Factor')
        for indicator, data in sensitivity_result.items()
    }

    if dataframe_output is True:
        return sensitivity_result_df
    else:
        return sensitivity_result