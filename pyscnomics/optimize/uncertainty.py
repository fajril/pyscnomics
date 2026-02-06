"""
Collection of functions to administer Monte Carlo simulation.
The code below is the modification of the code from PSCnomics.

The routine and result of this module is maintaining the
requirements of the PSCnomics.
"""

import copy
import os
# import traceback
import numpy as np
from scipy.stats import uniform, triang, truncnorm
from pathos.pools import ProcessPool as Pool
from pathos.helpers import mp

from pyscnomics.tools.summary import get_summary
from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts.transition import Transition
from pyscnomics.econ import FluidType
from pyscnomics.econ.selection import UncertaintyDistribution, SunkCostMethod
from pyscnomics.io.getattr import get_contract_attributes
from pyscnomics.api.converter import (
    convert_str_to_date,
    convert_str_to_int,
    convert_list_to_array_float_or_array,
    convert_dict_to_lifting,
    convert_dict_to_capital,
    convert_dict_to_intangible,
    convert_dict_to_opex,
    convert_dict_to_asr,
    convert_dict_to_lbt,
    convert_dict_to_cost_of_sales,
    convert_list_to_array_float,
    convert_list_to_array_float_or_array_or_none,
    convert_str_to_taxsplit,
    convert_str_to_npvmode,
    convert_str_to_discountingmode,
    convert_str_to_otherrevenue,
    convert_str_to_taxregime,
    convert_str_to_ftptaxregime,
    convert_str_to_depremethod,
    convert_str_to_inflationappliedto,
    convert_grosssplitregime_to_enum,
    convert_to_float,
    converter_sunk_cost_method,
    converter_reservoir_type_permen_2024,
    converter_initial_amortization_year,
    # read_fluid_type,
    # convert_to_method_limit,
    # convert_to_uncertainty_distribution,
    # convert_to_skk_summary_baseproject,
    # convert_summary_to_dict,
    # convert_str_to_optimization_parameters,
    # convert_str_to_optimization_targetparameter,
)
np.set_printoptions(suppress=True, precision=4)


class MonteCarloException(Exception):
    """ Exception to be raised for a misuse of MonteCarlo class """

    pass


def _extract_from_dict(target_key: str, source: dict, default=None):
    """
    Check whether a key exists in a dictionary and return a safe value.

    Parameters
    ----------
    target_key : str
        Key to look for in `data`.
    source : dict
        Input dictionary.
    default : any, optional
        Fallback value if key is missing or its value is None.

    Returns
    -------
    any
        `data[target_key]` if present and not None, otherwise `default`.
    """
    return (
        default if (target_key not in source) or (source[target_key] is None)
        else source[target_key]
    )


def get_setup_dict(data: dict) -> tuple:
    """
    Parse project setup and optional cost components into internal objects.

    Parameters
    ----------
    data : dict
        Project configuration dictionary containing a required ``setup`` section
        and optional cost-related sections.

    Returns
    -------
    tuple
        Parsed setup values and converted cost objects, returned in a fixed order
        expected by the economic engine.
    """

    # Check whether "setup" exist in "data" and prepare it accordingly
    se = _extract_from_dict(target_key="setup", source=data)

    # Parsing the contract setup into each corresponding variables
    start_date = convert_str_to_date(str_object=se["start_date"])
    end_date = convert_str_to_date(str_object=se["end_date"])
    approval_year = convert_str_to_int(str_object=se["approval_year"])
    oil_onstream_date = convert_str_to_date(str_object=se.get("oil_onstream_date", None))
    gas_onstream_date = convert_str_to_date(str_object=se.get("gas_onstream_date", None))
    is_pod_1 = se.get("is_pod_1", None)
    is_strict = se.get("is_strict", None)
    lifting = convert_dict_to_lifting(data_raw=data) if "lifting" in data else None
    capital = (
        convert_dict_to_capital(data_raw=data["capital"]) if "capital" in data else None
    )
    intangible = (
        convert_dict_to_intangible(data_raw=data["intangible"]) if "intangible" in data else None
    )
    opex = convert_dict_to_opex(data_raw=data["opex"]) if "opex" in data else None
    asr = convert_dict_to_asr(data_raw=data["asr"]) if "asr" in data else None
    lbt = convert_dict_to_lbt(data_raw=data["lbt"]) if "lbt" in data else None
    cost_of_sales = (
        convert_dict_to_cost_of_sales(data_raw=data["cost_of_sales"])
        if "cost_of_sales" in data else None
    )

    return (
        start_date,
        end_date,
        approval_year,
        oil_onstream_date,
        gas_onstream_date,
        is_pod_1,
        is_strict,
        lifting,
        capital,
        intangible,
        opex,
        asr,
        lbt,
        cost_of_sales,
    )


def get_summary_dict(data: dict) -> dict:
    """
    Extract and normalize summary-related arguments.

    Parameters
    ----------
    data : dict
        Input configuration containing a required ``summary_arguments`` section.

    Returns
    -------
    dict
        Summary arguments with defaults applied and enums converted:
        - discount_rate_start_year
        - inflation_rate
        - discount_rate
        - npv_mode
        - discounting_mode
        - profitability_discounted
    """

    # Check whether "summary_arguments" exists in "data" and prepare it accordingly
    sa = _extract_from_dict(target_key="summary_arguments", source=data)

    # Fill get_summary() argument with input data
    discount_rate_start_year = sa.get("discount_rate_start_year", None)
    inflation_rate = sa.get("inflation_rate", None)
    discount_rate = sa.get("discount_rate", 0.1)
    npv_mode = convert_str_to_npvmode(str_object=sa.get("npv_mode", "Full Cycle Nominal Terms"))
    discounting_mode = convert_str_to_discountingmode(
        str_object=sa.get("discounting_mode", "End Year Discounting")
    )
    profitability_discounted = sa.get("profitability_discounted", False)

    summary_arguments_dict = {
        "discount_rate_start_year": discount_rate_start_year,
        "inflation_rate": inflation_rate,
        "discount_rate": discount_rate,
        "npv_mode": npv_mode,
        "discounting_mode": discounting_mode,
        "profitability_discounted": profitability_discounted,
    }

    return summary_arguments_dict


def build_baseproject_instance(data: dict) -> BaseProject:
    """
    Build a ``BaseProject`` instance from input configuration data.

    Parameters
    ----------
    data : dict
        Project configuration containing ``setup`` and optional cost/lifting sections.

    Returns
    -------
    BaseProject
        Initialized BaseProject with setup, lifting, and cost components attached.

    Notes
    -----
    - Internally relies on ``get_setup_dict()`` for parsing & conversion.
    - Missing optional sections → corresponding attributes set to ``None``.
    """

    # Specify base arguments
    (
        start_date,
        end_date,
        approval_year,
        oil_onstream_date,
        gas_onstream_date,
        is_pod_1,
        is_strict,
        lifting,
        capital,
        intangible,
        opex,
        asr,
        lbt,
        cost_of_sales,
    ) = get_setup_dict(data=data)

    # Prepare contract attributes for BaseProject
    contract_kwargs = {
        # Base parameters
        "start_date": start_date,
        "end_date": end_date,
        "approval_year": approval_year,
        "oil_onstream_date": oil_onstream_date,
        "gas_onstream_date": gas_onstream_date,
        "is_pod_1": is_pod_1,
        "is_strict": is_strict,

        # Lifting and costs
        "lifting": lifting,
        "capital_cost": capital,
        "intangible_cost": intangible,
        "opex": opex,
        "asr_cost": asr,
        "lbt_cost": lbt,
        "cost_of_sales": cost_of_sales,
    }

    return BaseProject(**contract_kwargs)


def build_baseproject_arguments(data: dict) -> dict:
    """
    Build BaseProject contract arguments from input data.

    Parameters
    ----------
    data : dict
        Project input dictionary containing ``contract_arguments``.

    Returns
    -------
    dict
        Parsed BaseProject arguments, including:
        - other revenues → sulfur | electricity | CO₂
        - VAT & inflation settings

    Notes
    -----
    - ``contract_arguments`` must exist in ``data``
    - Values are converted to engine-ready types
    """

    # Check whether "contract_arguments" exist in "data", and prepare it accordingly
    ca = _extract_from_dict(target_key="contract_arguments", source=data)

    return {
        # Other revenues
        "sulfur_revenue": convert_str_to_otherrevenue(str_object=ca["sulfur_revenue"]),
        "electricity_revenue": convert_str_to_otherrevenue(str_object=ca["electricity_revenue"]),
        "co2_revenue": convert_str_to_otherrevenue(str_object=ca["co2_revenue"]),

        # VAT and inflation
        "vat_rate": convert_list_to_array_float_or_array(data_input=ca["vat_rate"]),
        "inflation_rate": convert_list_to_array_float_or_array(data_input=ca["inflation_rate"]),
        "inflation_rate_applied_to": convert_str_to_inflationappliedto(
            str_object=ca["inflation_rate_applied_to"]
        ),
    }


def get_baseproject(data: dict) -> dict:
    """
    Build, execute, and summarize a Base Project contract evaluation.

    This function constructs a :class:`BaseProject` instance using the provided
    input data, executes its economic evaluation, and generates a summary of the
    results.

    Parameters
    ----------
    data : dict
        Dictionary containing setup information, contract parameters, and summary
        configuration.

    Returns
    -------
    dict
        Summary of the Base Project contract evaluation.
    """

    # Specify contract and contract arguments
    contract = build_baseproject_instance(data=data)
    contract_arguments_dict = build_baseproject_arguments(data=data)

    # Execute BaseProject instance
    contract.run(**contract_arguments_dict)

    # Fill summary arguments
    summary_arguments_dict = get_summary_dict(data=data)

    return contract.get_summary(**summary_arguments_dict)


def build_costrecovery_instance(data: dict) -> CostRecovery:
    """
    Build a CostRecovery contract instance from input data.

    Parameters
    ----------
    data : dict
        Project configuration dictionary containing:
        - ``setup`` (base project info)
        - ``costrecovery`` (CR-specific terms)

    Returns
    -------
    CostRecovery
        Initialized CostRecovery contract object.

    Notes
    -----
    - Base attributes are parsed via ``get_setup_dict()``
    - CR terms include FTP, split, IC, cap rate, and DMO settings
    - Inputs are converted to engine-ready types before instantiation
    """

    # Specify base arguments
    (
        start_date,
        end_date,
        approval_year,
        oil_onstream_date,
        gas_onstream_date,
        is_pod_1,
        is_strict,
        lifting,
        capital,
        intangible,
        opex,
        asr,
        lbt,
        cost_of_sales,
    ) = get_setup_dict(data=data)

    # Check whether "costrecovery" exist in "data", and prepare it accordingly
    cr = _extract_from_dict(target_key="costrecovery", source=data)

    # Prepare contract attributes for CostRecovery
    contract_kwargs = {
        # Base parameters
        "start_date": start_date,
        "end_date": end_date,
        "approval_year": approval_year,
        "oil_onstream_date": oil_onstream_date,
        "gas_onstream_date": gas_onstream_date,
        "is_pod_1": is_pod_1,
        "is_strict": is_strict,

        # Lifting and costs
        "lifting": lifting,
        "capital_cost": capital,
        "intangible_cost": intangible,
        "opex": opex,
        "asr_cost": asr,
        "lbt_cost": lbt,
        "cost_of_sales": cost_of_sales,

        # FTP
        "oil_ftp_is_available": cr["oil_ftp_is_available"],
        "oil_ftp_is_shared": cr["oil_ftp_is_shared"],
        "oil_ftp_portion": convert_list_to_array_float_or_array(data_input=cr["oil_ftp_portion"]),
        "gas_ftp_is_available": cr["gas_ftp_is_available"],
        "gas_ftp_is_shared": cr["gas_ftp_is_shared"],
        "gas_ftp_portion": convert_list_to_array_float_or_array(data_input=cr["gas_ftp_portion"]),

        # Split
        "tax_split_type": convert_str_to_taxsplit(str_object=cr["tax_split_type"]),
        "condition_dict": cr["condition_dict"],
        "indicator_rc_icp_sliding": convert_list_to_array_float(
            data_list=cr["indicator_rc_icp_sliding"]
        ),
        "oil_ctr_pretax_share": convert_list_to_array_float_or_array(
            data_input=cr["oil_ctr_pretax_share"]
        ),
        "gas_ctr_pretax_share": convert_list_to_array_float_or_array(
            data_input=cr["gas_ctr_pretax_share"]
        ),

        # Investment credit and cap rate
        "oil_ic_rate": convert_to_float(target=cr["oil_ic_rate"]),
        "gas_ic_rate": convert_to_float(target=cr["gas_ic_rate"]),
        "ic_is_available": cr["ic_is_available"],
        "oil_cr_cap_rate": convert_to_float(target=cr["oil_cr_cap_rate"]),
        "gas_cr_cap_rate": convert_to_float(target=cr["gas_cr_cap_rate"]),

        # DMO
        "oil_dmo_volume_portion": convert_list_to_array_float_or_array(
            data_input=cr["oil_dmo_volume_portion"]
        ),
        "oil_dmo_fee_portion": convert_list_to_array_float_or_array(
            data_input=cr["oil_dmo_fee_portion"]
        ),
        "oil_dmo_holiday_duration": cr["oil_dmo_holiday_duration"],
        "gas_dmo_volume_portion": convert_list_to_array_float_or_array(
            data_input=cr["gas_dmo_volume_portion"]
        ),
        "gas_dmo_fee_portion": convert_list_to_array_float_or_array(
            data_input=cr["gas_dmo_fee_portion"]
        ),
        "gas_dmo_holiday_duration": cr["gas_dmo_holiday_duration"],

        # Carry forward depreciation
        "oil_carry_forward_depreciation": 0.0,
        "gas_carry_forward_depreciation": 0.0,
    }

    return CostRecovery(**contract_kwargs)


def build_costrecovery_arguments(data: dict) -> dict:
    """
    Build normalized cost-recovery arguments from raw input data.

    Extracts `contract_arguments` from `data`, applies defaults where needed,
    and converts string/list inputs into internal enum/array representations
    required by the cost-recovery model.

    Parameters
    ----------
    data : dict
        Raw input dictionary containing `contract_arguments`.

    Returns
    -------
    dict
        Sanitized and converted cost-recovery arguments ready for model use.
    """

    # Check whether "contract_arguments" exist in "data", and prepare it accordingly
    ca = _extract_from_dict(target_key="contract_arguments", source=data)

    return {
        # Other revenues
        "sulfur_revenue": convert_str_to_otherrevenue(str_object=ca["sulfur_revenue"]),
        "electricity_revenue": convert_str_to_otherrevenue(str_object=ca["electricity_revenue"]),
        "co2_revenue": convert_str_to_otherrevenue(str_object=ca["co2_revenue"]),

        # FTP
        "ftp_tax_regime": convert_str_to_ftptaxregime(str_object=ca["ftp_tax_regime"]),

        # VAT and inflation
        "vat_rate": convert_list_to_array_float_or_array(data_input=ca["vat_rate"]),
        "inflation_rate": convert_list_to_array_float_or_array(data_input=ca["inflation_rate"]),
        "inflation_rate_applied_to": convert_str_to_inflationappliedto(
            str_object=ca["inflation_rate_applied_to"]
        ),

        # DMO and tax
        "is_dmo_end_weighted": ca["is_dmo_end_weighted"],
        "tax_regime": convert_str_to_taxregime(str_object=ca["tax_regime"]),
        "effective_tax_rate": convert_list_to_array_float_or_array_or_none(
            data_list=ca["effective_tax_rate"]
        ),
        "post_uu_22_year2001": _extract_from_dict(
            target_key="post_uu_22_year2001", source=ca, default=True
        ),

        # Depreciation
        "depr_method": convert_str_to_depremethod(str_object=ca["depr_method"]),
        "decline_factor": ca["decline_factor"],
        "sum_undepreciated_cost": _extract_from_dict(
            target_key="sum_undepreciated_cost", source=ca, default=False
        ),

        # Cost of sales
        "oil_cost_of_sales_applied": _extract_from_dict(
            target_key="oil_cost_of_sales_applied", source=ca, default=False
        ),
        "gas_cost_of_sales_applied": _extract_from_dict(
            target_key="gas_cost_of_sales_applied", source=ca, default=False
        ),
    }


def get_costrecovery(data: dict) -> dict:
    """
    Run cost-recovery calculation and return its summary.

    Builds the cost-recovery contract instance and arguments from `data`,
    executes the model, and extracts the final summary output.

    Parameters
    ----------
    data : dict
        Raw input dictionary containing contract setup, arguments,
        and summary configuration.

    Returns
    -------
    dict
        Cost-recovery summary results.
    """

    # Specify contract and contract arguments
    contract = build_costrecovery_instance(data=data)
    contract_arguments_dict = build_costrecovery_arguments(data=data)

    # Execute CostRecovery instance
    contract.run(**contract_arguments_dict)

    # Fill summary arguments
    summary_arguments_dict = get_summary_dict(data=data)

    return contract.get_summary(**summary_arguments_dict)


def build_grosssplit_instance(data: dict) -> GrossSplit:
    """
    Build and initialize a GrossSplit contract instance.

    Extracts setup, lifting, cost, and Gross Split–specific parameters
    from `data`, prepares contract arguments, and returns a ready-to-run
    `GrossSplit` object.

    Parameters
    ----------
    data : dict
        Raw input dictionary containing project setup and gross split
        contract configuration.

    Returns
    -------
    GrossSplit
        Initialized GrossSplit contract instance.
    """

    # Specify base arguments
    (
        start_date,
        end_date,
        approval_year,
        oil_onstream_date,
        gas_onstream_date,
        is_pod_1,
        is_strict,
        lifting,
        capital,
        intangible,
        opex,
        asr,
        lbt,
        cost_of_sales,
    ) = get_setup_dict(data=data)

    # Check whether "grosssplit" exist in "data", and prepare it accordingly
    gs = _extract_from_dict(target_key="grosssplit", source=data)

    # Prepare contract attributes for GrossSplit
    contract_kwargs = {
        # Base parameters
        "start_date": start_date,
        "end_date": end_date,
        "approval_year": approval_year,
        "oil_onstream_date": oil_onstream_date,
        "gas_onstream_date": gas_onstream_date,
        "is_pod_1": is_pod_1,
        "is_strict": is_strict,

        # Lifting and costs
        "lifting": lifting,
        "capital_cost": capital,
        "intangible_cost": intangible,
        "opex": opex,
        "asr_cost": asr,
        "lbt_cost": lbt,
        "cost_of_sales": cost_of_sales,

        # Field and reservoir properties
        "field_status": _extract_from_dict(target_key="field_status", source=gs),
        "field_loc": _extract_from_dict(target_key="field_loc", source=gs),
        "res_depth": _extract_from_dict(target_key="res_depth", source=gs),
        "infra_avail": _extract_from_dict(target_key="infra_avail", source=gs),
        "res_type": _extract_from_dict(target_key="res_type", source=gs),
        "api_oil": _extract_from_dict(target_key="api_oil", source=gs),
        "domestic_use": _extract_from_dict(target_key="domestic_use", source=gs),
        "prod_stage": _extract_from_dict(target_key="prod_stage", source=gs),
        "co2_content": _extract_from_dict(target_key="co2_content", source=gs),
        "h2s_content": _extract_from_dict(target_key="h2s_content", source=gs),
        "field_reserves_2024": _extract_from_dict(target_key="field_reserves_2024", source=gs),
        "infra_avail_2024": _extract_from_dict(target_key="infra_avail_2024", source=gs),
        "field_loc_2024": _extract_from_dict(target_key="field_loc_2024", source=gs),
        "split_ministry_disc": convert_to_float(target=gs["split_ministry_disc"]),

        # DMO parameters
        "oil_dmo_volume_portion": convert_list_to_array_float_or_array(
            data_input=gs["oil_dmo_volume_portion"]
        ),
        "oil_dmo_fee_portion": convert_list_to_array_float_or_array(
            data_input=gs["oil_dmo_fee_portion"]
        ),
        "oil_dmo_holiday_duration": _extract_from_dict(
            target_key="oil_dmo_holiday_duration", source=gs
        ),
        "gas_dmo_volume_portion": convert_list_to_array_float_or_array(
            data_input=gs["gas_dmo_volume_portion"]
        ),
        "gas_dmo_fee_portion": convert_list_to_array_float_or_array(
            data_input=gs["gas_dmo_fee_portion"]
        ),
        "gas_dmo_holiday_duration": _extract_from_dict(
            target_key="gas_dmo_holiday_duration", source=gs
        ),

        # Carry forward depreciation
        "oil_carry_forward_depreciation": 0.0,
        "gas_carry_forward_depreciation": 0.0,
    }

    return GrossSplit(**contract_kwargs)


def build_grosssplit_arguments(data: dict) -> dict:
    """
    Build Gross Split contract argument dictionary.

    Extracts `contract_arguments` from `data`, applies defaults where needed,
    performs type/enum conversions, and prepares keyword arguments for
    Gross Split contract execution.

    Parameters
    ----------
    data : dict
        Raw input dictionary containing contract arguments.

    Returns
    -------
    dict
        Parsed and converted Gross Split contract arguments.
    """

    # Check whether "contract_arguments" exist in "data", and prepare it accordingly
    ca = _extract_from_dict(target_key="contract_arguments", source=data)

    return {
        # Other revenues
        "sulfur_revenue": convert_str_to_otherrevenue(str_object=ca["sulfur_revenue"]),
        "electricity_revenue": convert_str_to_otherrevenue(str_object=ca["electricity_revenue"]),
        "co2_revenue": convert_str_to_otherrevenue(str_object=ca["co2_revenue"]),

        # VAT and inflation
        "vat_rate": convert_list_to_array_float_or_array(data_input=ca["vat_rate"]),
        "inflation_rate": convert_list_to_array_float_or_array(data_input=ca["inflation_rate"]),
        "inflation_rate_applied_to": convert_str_to_inflationappliedto(
            str_object=ca["inflation_rate_applied_to"]
        ),

        # Production offset
        "cum_production_split_offset": convert_list_to_array_float_or_array(
            data_input=ca["cum_production_split_offset"]
        ),

        # Depreciation
        "depr_method": convert_str_to_depremethod(str_object=ca["depr_method"]),
        "decline_factor": _extract_from_dict(target_key="decline_factor", source=ca, default=2),
        "sum_undepreciated_cost": _extract_from_dict(
            target_key="sum_undepreciated_cost", source=ca, default=False
        ),

        # DMO and tax
        "is_dmo_end_weighted": _extract_from_dict(
            target_key="is_dmo_end_weighted", source=ca, default=False
        ),
        "tax_regime": convert_str_to_taxregime(str_object=ca["tax_regime"]),
        "effective_tax_rate": convert_list_to_array_float_or_array_or_none(
            data_list=ca["effective_tax_rate"]
        ),

        # Sunk cost
        "amortization": _extract_from_dict(target_key="amortization", source=ca, default=False),

        # Fiscal regime
        "regime": convert_grosssplitregime_to_enum(target=ca["regime"]),
        "reservoir_type_permen_2024": converter_reservoir_type_permen_2024(
            target_str=ca["reservoir_type_permen_2024"]
        ),
    }


def get_grosssplit(data: dict) -> dict:
    """
    Build, execute, and summarize a Gross Split contract evaluation.

    This function constructs a :class:`GrossSplit` instance using the provided
    input data, executes its economic evaluation, and generates a summary of the
    results.

    Parameters
    ----------
    data : dict
        Dictionary containing setup information, contract parameters, and summary
        configuration.

    Returns
    -------
    dict
        Summary of the Gross Split contract evaluation.
    """

    # Specify contract and contract arguments
    contract = build_grosssplit_instance(data=data)
    contract_arguments_dict = build_grosssplit_arguments(data=data)

    # Execute GrossSplit instance
    contract.run(**contract_arguments_dict)

    # Fill summary arguments
    summary_arguments_dict = get_summary_dict(data=data)

    return contract.get_summary(**summary_arguments_dict)


def get_transition(data: dict):
    """
    The function to get the Summary, Transition object, contract arguments,
    and summary arguments used.

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
    if (
        data["contract_1"]["costrecovery"] is not None
        and data["contract_1"]["grosssplit"] is None
    ):
        _, contract_1, contract_arguments_1, _ = get_costrecovery(
            data=data["contract_1"], summary_result=False
        )

    elif (
        data["contract_1"]["grosssplit"] is not None
        and data["contract_1"]["costrecovery"] is None
    ):
        _, contract_1, contract_arguments_1, _ = get_grosssplit(
            data=data["contract_1"], summary_result=False
        )

    else:
        raise MonteCarloException("Contract type is not recognized")

    # Defining contract_2
    if (
        data["contract_2"]["costrecovery"] is not None
        and data["contract_2"]["grosssplit"] is None
    ):
        _, contract_2, contract_arguments_2, _ = get_costrecovery(
            data=data["contract_2"], summary_result=False
        )

    elif (
        data["contract_2"]["grosssplit"] is not None
        and data["contract_2"]["costrecovery"] is None
    ):
        _, contract_2, contract_arguments_2, _ = get_grosssplit(
            data=data["contract_2"], summary_result=False
        )

    else:
        raise MonteCarloException("Contract type is not recognized")

    # generating the transition contract object
    contract = Transition(
        contract1=contract_1,
        contract2=contract_2,
        argument_contract1=contract_arguments_1,
        argument_contract2=contract_arguments_2,
    )

    # Generating the transition contract arguments
    contract_arguments_dict = data["contract_arguments"]

    # Running the transition contract
    contract.run(**contract_arguments_dict)

    # Filling the summary arguments
    summary_arguments_dict = get_summary_dict(data=data)
    summary_arguments_dict["contract"] = contract

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
    Generate an array of multipliers for Monte Carlo simulation based on
    the specified probability distribution.

    All distributions are generated in *ratio space* (value / mean_value),
    so the resulting multipliers are centered around 1.0.

    Parameters
    ----------
    run_number : int
        Number of Monte Carlo runs.
    distribution : str
        Distribution type (case-insensitive):
        {"uniform", "triangular", "normal", "lognormal"}.
    min_value : float
        Minimum value of the distribution.
    mean_value : float
        Mean (central) value of the distribution.
    max_value : float
        Maximum value of the distribution.
    std_dev : float
        Standard deviation (used for Normal and LogNormal distributions).

    Returns
    -------
    np.ndarray
        Array of Monte Carlo multipliers.

    Raises
    ------
    ValueError
        If input values are invalid.
    MonteCarloException
        If the distribution type is unsupported.

    TL;DR:
    Generate mean-normalized Monte Carlo multipliers using Uniform, Triangular,
    Normal (truncated), or LogNormal (truncated) distributions.
    """

    # Return an array of ones if min = mean = max
    if min_value == mean_value == max_value:
        return np.ones(run_number, dtype=np.float64)

    # Validate input: cannot have zero mean value
    if mean_value == 0:
        raise ValueError(f"Cannot have zero mean value")

    # Validate input: filter incorrect assignment of `max_value` and/or `min_value`
    if max_value <= min_value:
        raise ValueError(
            f"Paramater max_value must be greater than min_value"
        )

    # Normalize distribution name
    distribution = distribution.lower()

    # Uniform distribution
    if distribution == "uniform":
        # Normalize parameters
        min_ratio, max_ratio = [float(v / mean_value) for v in (min_value, max_value)]

        # Draw samples, then assign them as `multipliers`
        multipliers = uniform.rvs(
            loc=min_ratio,
            scale=max_ratio - min_ratio,
            size=run_number,
        )

    # Triangular distribution
    elif distribution == "triangular":
        # Normalize parameters
        (
            min_ratio,
            mean_ratio,
            max_ratio
        ) = [float(v / mean_value) for v in (min_value, mean_value, max_value)]

        # Determine mode (central point)
        c = (mean_ratio - min_ratio) / (max_ratio - min_ratio)

        # Draw samples, then assign them as `multipliers`
        multipliers = triang.rvs(
            c=c,
            loc=min_ratio,
            scale=max_ratio - min_ratio,
            size=run_number,
        )

    # Normal distribution
    elif distribution == "normal":
        # Validate input: cannot have zero standard deviation
        if std_dev == 0:
            raise ValueError(f"Cannot have zero standard deviation")

        # Normalize parameters
        (
            min_ratio,
            mean_ratio,
            max_ratio,
            std_ratio
        ) = [float(v / mean_value) for v in (min_value, mean_value, max_value, std_dev)]

        # Determine z-values
        z_min = (min_ratio - mean_ratio) / std_ratio
        z_max = (max_ratio - mean_ratio) / std_ratio

        # Draw samples, then assign them as `multipliers`
        multipliers_init = truncnorm.rvs(
            a=z_min,
            b=z_max,
            loc=0,
            scale=1,
            size=run_number,
        )

        multipliers = (multipliers_init * std_ratio) + mean_ratio

    # Log normal distribution
    elif distribution == "lognormal":
        # Validate input: cannot have zero standard deviation
        if std_dev == 0:
            raise ValueError(f"Cannot have zero standard deviation")

        # Validate input
        if min_value <= 0 or mean_value <= 0 or max_value <= 0:
            raise ValueError("All input must be positive for a log-normal distribution")

        # Normalize parameters
        (
            min_ratio,
            mean_ratio,
            max_ratio,
            std_ratio
        ) = [float(v / mean_value) for v in (min_value, mean_value, max_value, std_dev)]

        # Convert to log-space
        log_min = np.log(min_ratio)
        log_mean = np.log(mean_ratio)
        log_max = np.log(max_ratio)
        log_std = std_ratio / mean_ratio

        # Determine z-values in log-space
        z_min = (log_min - log_mean) / log_std
        z_max = (log_max - log_mean) / log_std

        if z_min > z_max:
            raise ValueError(f"Invalid truncation bounds: z_min = {z_min}, z_max = {z_max}")

        # Draw truncated normal samples in log-space
        log_samples = truncnorm.rvs(
            a=z_min,
            b=z_max,
            loc=0,
            scale=1,
            size=run_number,
        )

        # Transform back
        multipliers = np.exp((log_samples * log_std) + log_mean)

    else:
        raise MonteCarloException(
            f"The type of distribution is unrecognized. "
            f"Please select type of distribution between: "
            f"Uniform, Triangular, Normal, and LogNormal."
        )

    return multipliers


def min_mean_max_retriever(
    contract: BaseProject | CostRecovery | GrossSplit | Transition,
    verbose: bool = False,
):
    """
    Retrieve minimum, mean, and maximum values for key economic parameters.

    This function computes summary statistics (min, mean, max) for CAPEX,
    OPEX, lifting rates, and oil/gas prices based on data stored in the
    contract object.

    Parameters
    ----------
    contract : BaseProject | CostRecovery | GrossSplit | Transition
        Contract instance containing cost, lifting, and price data.
    verbose : bool, default False
        If True, print computed statistics to stdout.

    Returns
    -------
    dict
        Dictionary containing min, mean, and max values for each parameter.
    """

    # Compute global min/mean/max across multiple arrays
    def _get_statistics(arrays_list: list):
        min_value = min([np.min(arr) for arr in arrays_list])
        mean_value = np.mean([np.mean(arr) for arr in arrays_list])
        max_value = max([np.max(arr) for arr in arrays_list])
        return min_value, mean_value, max_value

    # Compute min/mean/max for a single array (e.g., prices)
    def _get_statistics_price(array: np.ndarray):
        return array.min(), array.mean(), array.max()

    # Calculate statistics (min, mean, max) for CAPEX.
    # --- CAPEX: capital + intangible costs ---
    (min_capex, mean_capex, max_capex) = _get_statistics(
        arrays_list=[
            getattr(contract, "capital_cost_total").cost,
            getattr(contract, "intangible_cost_total").cost,
        ]
    )

    # Calculate statistics (min, mean, max) for OPEX.
    # --- OPEX: operating, ASR, LBT, and cost of sales ---
    (min_opex, mean_opex, max_opex) = _get_statistics(
        arrays_list=[
            getattr(contract, "opex_total").cost,
            getattr(contract, "asr_cost_total").cost,
            getattr(contract, "lbt_cost_total").cost,
            getattr(contract, "cost_of_sales_total").cost,
        ]
    )

    # Calculate statistics (min, mean, max) for LIFTING.
    # --- LIFTING: oil and gas lifting rates ---
    (min_lifting, mean_lifting, max_lifting) = _get_statistics(
        arrays_list=[
            getattr(contract, "_oil_lifting").lifting_rate,
            getattr(contract, "_gas_lifting").lifting_rate,
        ]
    )

    # Calculate statistics (min, mean, max) for OIL PRICE
    (min_oil_price, mean_oil_price, max_oil_price) = _get_statistics_price(
        array=getattr(contract, "_oil_lifting").price
    )

    # Calculate statistics (min, mean, max) for GAS PRICE
    (min_gas_price, mean_gas_price, max_gas_price) = _get_statistics_price(
        array=getattr(contract, "_gas_lifting").price
    )

    # Collect all statistics into a dictionary
    results = {
        "min_capex": min_capex,
        "mean_capex": mean_capex,
        "max_capex": max_capex,
        "min_opex": min_opex,
        "mean_opex": mean_opex,
        "max_opex": max_opex,
        "min_oil_price": min_oil_price,
        "mean_oil_price": mean_oil_price,
        "max_oil_price": max_oil_price,
        "min_gas_price": min_gas_price,
        "mean_gas_price": mean_gas_price,
        "max_gas_price": max_gas_price,
        "min_lifting": min_lifting,
        "mean_lifting": mean_lifting,
        "max_lifting": max_lifting,
    }

    # Optional debug output
    if verbose is True:
        print("Parameter used:")
        for key, value in results.items():
            print(key, ": ", value)
        print("")

    return results


class ProcessMonte:
    """
    Prepare and execute Monte Carlo simulation for uncertainty assessment.
    """
    target = ["npv", "irr", "pi", "pot", "gov_take", "ctr_net_share"]

    def __init__(self, contract_type, contract, numSim, params):

        self.type = contract_type
        self.numSim = numSim
        self.baseContract = contract
        self.parameter = params
        self.hasOil = any([p["id"] == 0 for p in self.parameter])
        self.hasGas = any([p["id"] == 1 for p in self.parameter])

        mults = np.array([1, 0.75, 0.5, 0.25, 0.125])
        self.multipliers = np.repeat(mults[:, np.newaxis], len(self.parameter), axis=1)

        # # Prepare multipliers
        # self.multipliers = np.ones(
        #     [self.numSim, len(self.parameter)], dtype=np.float64
        # )
        #
        # for i, param in enumerate(self.parameter):
        #     self.multipliers[:, i] = get_multipliers_montecarlo(
        #         run_number=self.numSim,
        #         distribution=param["dist"].value,
        #         min_value=param["min"],
        #         mean_value=param["base"],
        #         max_value=param["max"],
        #         std_dev=param["stddev"],
        #     )

        print('\t')
        print(f'Filetype: {type(self.multipliers)}, Shape: {self.multipliers.shape}')
        print('multipliers = \n', self.multipliers)

    def Adjust_Data(self, multipliers: np.ndarray) -> dict:
        """
        Apply parameter-based multipliers to adjust contract economic data.

        Creates a deep copy of the base contract and scales relevant fields
        (price, cost, lifting rate) according to the provided multipliers.

        Parameters
        ----------
        multipliers : np.ndarray
            Array of scaling factors for target parameters
            (Oil Price, Gas Price, OPEX, CAPEX, Lifting).

        Returns
        -------
        dict
            Deep-copied contract with adjusted economic attributes.

        Notes
        -----
        The adjustment is performed using an internal helper function and
        does not modify the original base contract.
        """

        contract_adjusted: dict = copy.deepcopy(self.baseContract)

        def _adjust_partial_data(
            contract_: dict,
            target_param: str,
            key: str,
            multiplier: float,
            datakeys: list | None = None,
        ):
            """
            Helper function to apply partial data adjustment to selected contract
            attributes.

            Scales numeric fields in the specified section (e.g., 'lifting', 'opex',
            'capex') based on the target parameter and multiplier.

            Parameters
            ----------
            contract_ : dict
                Contract data structure to modify.
            target_param : str
                Parameter to adjust (e.g., "Oil Price", "Gas Price", "Lifting").
            key : str
                Section name containing target data.
            multiplier : float
                Scaling factor applied to numeric fields.
            datakeys : list of str, optional
                Field names to adjust for non-lifting sections.

            Notes
            -----
            Lifting adjustments depend on `target_param` and `fluid_type`;
            other sections use `datakeys`.
            """

            if datakeys is None:
                datakeys = []

            for item_key, item in contract_[key].items():
                # Specify target keys for lifting-related target
                if key == "lifting":
                    if target_param == "Oil Price" and item["fluid_type"] == "Oil":
                        target_keys = ["price"]
                    elif target_param == "Gas Price" and item["fluid_type"] == "Gas":
                        target_keys = ["price"]
                    elif target_param == "Lifting":
                        target_keys = ["lifting_rate", "prod_rate"]
                    else:
                        continue

                # Specify target keys for non-lifting targets
                else:
                    target_keys = datakeys

                # Adjust target attributes by multiplication with the prescribed multipliers
                for k in target_keys:
                    if k in item:
                        item[k] = (np.array(item[k]) * multiplier).tolist()

        # Specify attribute `contract_` based on `contract_type`
        contract_ = (
            contract_adjusted if self.type < 3 else contract_adjusted[f"contract_{2}"]
        )

        # Contract adjustments per single run simulation (i.e., per row)
        for i, val in enumerate(self.parameter):

            # Target parameter: Oil Price
            if val["id"] == 0:
                _adjust_partial_data(
                    contract_=contract_,
                    target_param="Oil Price",
                    key="lifting",
                    multiplier=multipliers[i]
                )

            # Target parameter: Gas Price
            elif val["id"] == 1:
                _adjust_partial_data(
                    contract_=contract_,
                    target_param="Gas Price",
                    key="lifting",
                    multiplier=multipliers[i],
                )

            # Target parameter: OPEX
            elif val["id"] == 2:
                # Adjust class OPEX
                _adjust_partial_data(
                    contract_=contract_,
                    target_param="OPEX",
                    key="opex",
                    multiplier=multipliers[i],
                    datakeys=["fixed_cost", "cost_per_volume", "cost"],
                )

                # Adjust class ASR
                _adjust_partial_data(
                    contract_=contract_,
                    target_param="ASR",
                    key="asr",
                    multiplier=multipliers[i],
                    datakeys=["cost"],
                )

                # Adjust class LBT
                _adjust_partial_data(
                    contract_=contract_,
                    target_param="LBT",
                    key="lbt",
                    multiplier=multipliers[i],
                    datakeys=["cost"],
                )

                # Adjust class CostOfSales
                _adjust_partial_data(
                    contract_=contract_,
                    target_param="COS",
                    key="cost_of_sales",
                    multiplier=multipliers[i],
                    datakeys=["cost"],
                )

            # Target parameter: CAPEX
            elif val["id"] == 3:
                # Adjust class CapitalCost
                _adjust_partial_data(
                    contract_=contract_,
                    target_param="CAPEX",
                    key="capital",
                    multiplier=multipliers[i],
                    datakeys=["cost"],
                )

                # Adjust class Intangible
                _adjust_partial_data(
                    contract_=contract_,
                    target_param="CAPEX",
                    key="intangible",
                    multiplier=multipliers[i],
                    datakeys=["cost"],
                )

            # Target parameter: Lifting
            elif val["id"] == 4:
                _adjust_partial_data(
                    contract_=contract_,
                    target_param="Lifting",
                    key="lifting",
                    multiplier=multipliers[i],
                )

        return contract_

    def calcContract(self, n: int) -> dict:
        """
        Execute a Monte Carlo simulation for the selected contract type and
        return key economic metrics.

        Parameters
        ----------
        n : int
            Simulation index used to select the corresponding multiplier set
            for data adjustment.

        Returns
        -------
        dict
            A dictionary with keys:
            - 'n': int, the simulation index.
            - 'output': tuple of float, containing
              (ctr_npv, ctr_irr, ctr_pi, ctr_pot, gov_take, ctr_net_share).

        Notes
        -----
        The function automatically selects and runs the appropriate contract evaluation
        (Cost Recovery, Gross Split, Transition, or Base Project) based on ``self.type``.
        In case of an error, zeros are returned for all output metrics.
        """
        try:
            # print(f"Monte Progress: {n}", flush=True)
            # time.sleep(100)

            # Specify adjusted data by calling the "Adjust_Data()" method
            dataAdj: dict = self.Adjust_Data(self.multipliers[n, :])

            # Execute the corresponding contract and return the result in terms of summary
            mapping_summary = {
                1: get_costrecovery,
                2: get_grosssplit,
                3: get_transition,
            }

            csummary = mapping_summary.get(self.type, get_baseproject)(data=dataAdj)

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

    def get_outcomes(self, results: np.ndarray) -> dict:
        """
        Compute and summarize probabilistic outcomes from Monte Carlo results.

        The function sorts simulation results, assigns cumulative probabilities,
        and computes key percentiles (P10, P50, P90) to summarize uncertainty.

        Parameters
        ----------
        results : np.ndarray
            Array of simulation results with shape (numSim, n_variables).

        Returns
        -------
        dict
            Dictionary containing:
            - **params** : list of str
                Names of economic parameters considered.
            - **results** : list of list
                Sorted results with corresponding cumulative probabilities.
            - **P10**, **P50**, **P90** : list of float
                Percentile values representing optimistic, median, and
                conservative outcomes.

        Notes
        -----
        If `self.hasGas` is True, gas-related parameters are included.
        """

        row_number = self.numSim

        # Sort results in ascending manner
        results_sorted = np.take_along_axis(
            arr=results,
            indices=np.argsort(results, axis=0),
            axis=0
        )

        # Specify probability
        prob = np.arange(1, row_number + 1, dtype=float) / row_number

        # Arrange the results
        results_arranged = np.concatenate(
            (prob[:, np.newaxis], results_sorted), axis=1
        )

        print('\t')
        print(f'Filetype: {type(results_arranged)}')
        print(f'Shape: {results_arranged.shape}')
        print('results_arranged = \n', results_arranged[:, :4])

        # Calculate P10, P50, P90
        percentiles = np.percentile(
            a=results_arranged,
            q=[10, 50, 90],
            method="higher",
            axis=0,
        )

        print('\t')
        print(f'Filetype: {type(percentiles)}')
        print(f'Shape: {percentiles.shape}')
        print('percentiles = \n', percentiles[:, :4])

        # Determine indices of data
        indices = np.linspace(0, row_number, 101)[0:-1].astype(int)
        indices[0] = 1
        if indices[-1] != row_number - 1:
            indices = np.append(indices, int(row_number - 1))

        print('\t')
        print(f'Filetype: {type(indices)}')
        print(f'Shape: {indices.shape}')
        print('indices = ', indices)

        print('\t')
        print(f'Filetype: {type(self.parameter)}')
        print(f'Length: {len(self.parameter)}')
        print('self.parameter = \n', self.parameter)

        target_indices = np.array([val["id"] for _, val in enumerate(self.parameter)])
        param_mapping = {
            0: "a",
            1: "b",
            2: "c",
            3: "d",
            4: "e",
        }
        param_names = param_mapping[target_indices]

        print('\t')
        print(f'Filetype: {type(target_indices)}')
        print(f'Length: {len(target_indices)}')
        print('target_indices = ', target_indices)

        print('\t')
        print(f'Filetype: {type(param_names)}')
        print(f'Length: {len(param_names)}')
        print('param_names = ', param_names)

        # # Final outcomes
        # outcomes = {
        #     "params": (
        #         ["Oil Price", "Gas Price", "Opex", "Capex", "Cum. prod."]
        #         if self.hasGas
        #         else ["Oil Price", "Opex", "Capex", "Cum. prod."]
        #     ),
        #     "results": results_arranged[indices, :].tolist(),
        #     "P10": percentiles[0, :].tolist(),
        #     "P50": percentiles[1, :].tolist(),
        #     "P90": percentiles[2, :].tolist(),
        # }
        #
        # return outcomes

    def calculate_single_core(self) -> dict:
        """
        Run Monte Carlo simulation sequentially on a single CPU core.

        Executes multiple contract evaluations in series, storing key
        economic indicators and parameter multipliers for each iteration.

        Returns
        -------
        dict
            Dictionary of summarized outcomes from all simulations, including:
            - **params** : list of str
                Names of economic parameters.
            - **results** : list of list
                Simulation results with cumulative probabilities.
            - **P10**, **P50**, **P90** : list of float
                Key percentiles summarizing uncertainty in results.
        """

        row_number = self.numSim

        # Designate a container to store Monte Carlo simulation result
        results: np.ndarray = np.zeros(
            [row_number, len(self.target) + len(self.parameter)], dtype=float
        )

        for n in range(row_number):

            # Execute Monte Carlo simulation by calling function calcContract()
            out = self.calcContract(n)
            rnum = out["n"]
            output = out["output"]

            # Columns 0 until 5:
            # Fill with NPV, IRR, PI, POT, gov_take, and ctr_net_share, respectively
            results[rnum, 0:len(self.target)] = output

            # Column 6 until end:
            # Fill with the multipliers associated with:
            # oil price, gas price, opex, capex, and lifting, accordingly
            results[rnum, len(self.target):] = [
                self.multipliers[rnum, idx] * items["base"]
                for idx, items in enumerate(self.parameter)
            ]

        # Get the outcomes of Monte Carlo simulation
        return self.get_outcomes(results=results)

    def calculate_multi_cores(self) -> dict:
        """
        Run Monte Carlo simulation in parallel using multiple CPU cores.

        Distributes contract evaluations across available processors and
        aggregates results for percentile-based outcome analysis.

        Returns
        -------
        dict
            Dictionary of summarized outcomes including:
            - **params** : list of str
                Economic parameter names.
            - **results** : list of list
                Combined simulation results with cumulative probabilities.
            - **P10**, **P50**, **P90** : list of float
                Key percentiles representing uncertainty ranges.
        """

        mp.freeze_support()

        row_number = self.numSim

        # Specify the number of active CPU
        n_processes = max(1, int(os.cpu_count() / 2))

        # Designate a container to store Monte Carlo simulation results
        results: np.ndarray = np.zeros(
            [row_number, len(self.target) + len(self.parameter)], dtype=float
        )

        # Specify a helper function
        def worker(n):
            """
            Run a single Monte Carlo iteration.

            Parameters
            ----------
            n : int
                Simulation index.

            Returns
            -------
            tuple
                (rnum, output, multipliers) containing the run index,
                economic results, and parameter multipliers.
            """
            out = self.calcContract(n)
            rnum = out["n"]
            output = out["output"]
            multipliers = [
                self.multipliers[rnum, idx] * items["base"]
                for idx, items in enumerate(self.parameter)
            ]

            return rnum, output, multipliers

        # Instantiate pathos pool
        pl = Pool(nodes=n_processes)

        # Map each Monte Carlo run in parallel
        results_list = pl.map(worker, range(row_number))

        # Close the pool instance
        pl.close()
        pl.join()

        for rnum, output, mult in results_list:

            # Columns 0 until 5:
            # Fill with NPV, IRR, PI, POT, gov_take, and ctr_net_share, respectively
            results[rnum, 0:len(self.target)] = output

            # Column 6 until 11:
            # Fill with the multipliers associated with:
            # oil price, gas price, opex, capex, and lifting, accordingly
            results[rnum, len(self.target):] = mult

        return self.get_outcomes(results=results)


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
    verbose: bool = False,
) -> dict:
    """
    Perform Monte Carlo uncertainty analysis for a PSC economic model.

    Generates probabilistic outcomes (P10, P50, P90) by varying key
    parameters—oil price, gas price, OPEX, CAPEX, and lifting—according
    to defined distributions and standard deviations.

    Returns
    -------
    dict
        Summary of probabilistic outcomes including P10, P50, P90, and
        associated simulation results.

    Notes
    -----
    The function builds and executes a PSC contract (BaseProject,
    CostRecovery, GrossSplit, or Transition) across multiple simulation
    runs. Each run applies random multipliers to key parameters,
    evaluates the economic results, and aggregates them to estimate
    uncertainty ranges.

    Automatically switches to parallel mode when `run_number > 400`.
    """

    # Translating the contract type before parsing into ProcessMonte class
    if isinstance(contract, CostRecovery):
        contract_type = 1

    elif isinstance(contract, GrossSplit):
        contract_type = 2

    elif isinstance(contract, Transition):
        contract_type = 3

    else:
        contract_type = 0

    # Extract original statistics (min, mean, max) of several input parameters
    min_max_mean_original = min_mean_max_retriever(contract=contract, verbose=verbose)

    # Organize the statistics of several input parameters (from user's input)
    min_max_mean_std = {
        "min_oil_price": min_oil_price,
        "mean_oil_price": mean_oil_price,
        "max_oil_price": max_oil_price,
        "min_gas_price": min_gas_price,
        "mean_gas_price": mean_gas_price,
        "max_gas_price": max_gas_price,
        "min_opex": min_opex,
        "mean_opex": mean_opex,
        "max_opex": max_opex,
        "min_capex": min_capex,
        "mean_capex": mean_capex,
        "max_capex": max_capex,
        "min_lifting": min_lifting,
        "mean_lifting": mean_lifting,
        "max_lifting": max_lifting,
    }

    # When user does not provide statistics input for the required parameters, then
    # fill them with the original values taken from variable "min_max_mean_original"
    for element in min_max_mean_std.keys():
        if min_max_mean_std[element] is None:
            min_max_mean_std[element] = min_max_mean_original[element]

    # Specify default values for multipliers,
    # (in the form of "min_multipliers" and "max_multipliers")
    # min_factor, max_factor = (0.8, 1.2)

    # Check for equal values of min, mean, and max, then specify adjustments.
    # =========================================================================
    # Extract keywords: "oil_price", "gas_price", "opex", "capex", and "lifting"
    # from variable "min_max_mean_std".
    # bases = [key[4:] for key in min_max_mean_std if key.startswith("min_")]

    # for base in bases:
    #     min_key, mean_key, max_key = f"min_{base}", f"mean_{base}", f"max_{base}"
    #
    #     # Exit loop if at least one of ["min_key", "mean_key", "max_key"] is not
    #     # available in variable "min_max_mean_std".
    #     if not all([k in min_max_mean_std for k in [min_key, mean_key, max_key]]):
    #         continue
    #
    #     # Extract statistics (min, mean, max) from variable "min_max_mean_std"
    #     min_val, mean_val, max_val = (
    #         min_max_mean_std[min_key],
    #         min_max_mean_std[mean_key],
    #         min_max_mean_std[max_key],
    #     )
    #
    #     # Carry out adjustment if min_val == mean_val == max_val.
    #     # Adjustment is invoked on "min" and "max" values of a particular keyword
    #     # in variable "min_max_mean_std" by applying default multipliers.
    #     if min_val == mean_val == max_val:
    #         min_max_mean_std[min_key] = min_factor * min_val
    #         min_max_mean_std[max_key] = max_factor * max_val

    # print('\t')
    # print(f'Filetype: {type(min_max_mean_std)}')
    # print(f'Length: {len(min_max_mean_std)}')
    # print('min_max_mean_std = \n', min_max_mean_std)

    # Create a list of input arguments' configuration for each uncertainty parameters.
    # --- Uncertainty parameter:
    # --- (1) OIL PRICE, (2) GAS PRICE, (3) OPEX, (4) CAPEX, (5) Lifting
    parameter_initial = [
        {
            "id": 0,
            "fluid": FluidType.OIL,
            "dist": oil_price_distribution,
            "min": min_max_mean_std["min_oil_price"],
            "base": min_max_mean_std["mean_oil_price"],
            "max": min_max_mean_std["max_oil_price"],
            "stddev": oil_price_stddev,
        },
        {
            "id": 1,
            "fluid": FluidType.GAS,
            "dist": gas_price_distribution,
            "min": min_max_mean_std["min_gas_price"],
            "base": min_max_mean_std["mean_gas_price"],
            "max": min_max_mean_std["max_gas_price"],
            "stddev": gas_price_stddev,
        },
        {
            "id": 2,
            "fluid": None,
            "dist": opex_distribution,
            "min": min_max_mean_std["min_opex"],
            "base": min_max_mean_std["mean_opex"],
            "max": min_max_mean_std["max_opex"],
            "stddev": opex_stddev,
        },
        {
            "id": 3,
            "fluid": None,
            "dist": capex_distribution,
            "min": min_max_mean_std["min_capex"],
            "base": min_max_mean_std["mean_capex"],
            "max": min_max_mean_std["max_capex"],
            "stddev": capex_stddev,
        },
        {
            "id": 4,
            "fluid": None,
            "dist": lifting_distribution,
            "min": min_max_mean_std["min_lifting"],
            "base": min_max_mean_std["mean_lifting"],
            "max": min_max_mean_std["max_lifting"],
            "stddev": lifting_stddev,
        },
    ]

    # Leave out configuration for OIL price or GAS price based on the available fluid
    fluid_produced = [lft.fluid_type for lft in contract.lifting]
    parameter = [
        p for p in parameter_initial
        if p["fluid"] is None or p["fluid"] in fluid_produced
    ]

    # Constructing the contract key
    contract_dict: dict = get_contract_attributes(
        contract=contract,
        contract_arguments=contract_arguments,
        summary_arguments=summary_arguments,
    )

    # Executing the montecarlo
    kwargs_monte = {
        "contract_type": contract_type,
        "contract": contract_dict,
        "params": parameter,
        "numSim": run_number,
    }

    monte = ProcessMonte(**kwargs_monte)

    # mults = np.array([1, 0.75, 0.5, 0.25, 0.125])
    # multipliers = np.repeat(mults[:, np.newaxis], len(parameter), axis=1)
    #
    # print('\t')
    # print(f'Filetype: {type(multipliers)}')
    # print(f'Shape: {multipliers.shape}')
    # print('multipliers = \n', multipliers)

    # monte.Adjust_Data(multipliers=multipliers[0, :])
    # monte.calcContract(n=2)
    monte.calculate_single_core()


    # # Use multiprocessing if run_number is large (i.e. larger than 400)
    # try:
    #     if run_number <= 400:
    #         return monte.calculate_single_core()
    #     else:
    #         return monte.calculate_multi_cores()
    #
    # # Should error occurs, return back to serial processing
    # except Exception as e:
    #     print(f"[Warning] Parallel mode failed: {e}")
    #     # print(traceback.format_exc(limit=2))
    #
    #     return monte.calculate_single_core()


"""
FORMER APPROACH
===============

def get_setup_dict(data: dict) -> tuple:
    # Parsing the contract setup into each corresponding variables
    start_date = convert_str_to_date(str_object=data["setup"]["start_date"])
    end_date = convert_str_to_date(str_object=data["setup"]["end_date"])
    oil_onstream_date = convert_str_to_date(
        str_object=data["setup"]["oil_onstream_date"]
    )
    gas_onstream_date = convert_str_to_date(
        str_object=data["setup"]["gas_onstream_date"]
    )
    lifting = convert_dict_to_lifting(data_raw=data)
    capital = convert_dict_to_capital(data_raw=data["capital"])
    intangible = convert_dict_to_intangible(data_raw=data["intangible"])
    opex = convert_dict_to_opex(data_raw=data["opex"])
    asr = convert_dict_to_asr(data_raw=data["asr"])
    lbt = convert_dict_to_lbt(data_raw=data["lbt"])
    cost_of_sales = convert_dict_to_cost_of_sales(data_raw=data["cost_of_sales"])
    return (
        start_date,
        end_date,
        oil_onstream_date,
        gas_onstream_date,
        lifting,
        capital,
        intangible,
        opex,
        asr,
        lbt,
        cost_of_sales,
    )
    
=========================================================================================
    
def get_summary_dict(data: dict) -> dict:
    # Filling the argument with the input data
    reference_year = data["summary_arguments"].get("reference_year", None)
    inflation_rate = data["summary_arguments"].get("inflation_rate", None)
    discount_rate = data["summary_arguments"].get("discount_rate", 0.1)
    npv_mode = convert_str_to_npvmode(
        str_object=data["summary_arguments"].get("npv_mode", "Full Cycle Nominal Terms")
    )
    discounting_mode = convert_str_to_discountingmode(
        str_object=data["summary_arguments"].get("discounting_mode", "discounting_mode")
    )
    profitability_discounted = data["summary_arguments"].get(
        "profitability_discounted", False
    )

    summary_arguments_dict = {
        "reference_year": reference_year,
        "inflation_rate": inflation_rate,
        "discount_rate": discount_rate,
        "npv_mode": npv_mode,
        "discounting_mode": discounting_mode,
        "profitability_discounted": profitability_discounted,
    }

    return summary_arguments_dict
    
=========================================================================================
    
def get_baseproject(data: dict):
    (
        start_date,
        end_date,
        oil_onstream_date,
        gas_onstream_date,
        lifting,
        tangible,
        intangible,
        opex,
        asr,
        lbt,
        cost_of_sales,
    ) = get_setup_dict(data=data)

    contract = BaseProject(
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
    )

    contract_arguments_dict = {
        "sulfur_revenue": convert_str_to_otherrevenue(
            str_object=data["contract_arguments"]["sulfur_revenue"]
        ),
        "electricity_revenue": convert_str_to_otherrevenue(
            str_object=data["contract_arguments"]["electricity_revenue"]
        ),
        "co2_revenue": convert_str_to_otherrevenue(
            str_object=data["contract_arguments"]["co2_revenue"]
        ),
        "sunk_cost_reference_year": data["contract_arguments"][
            "sunk_cost_reference_year"
        ],
        "tax_rate": convert_list_to_array_float_or_array(
            data_input=data["contract_arguments"]["tax_rate"]
        ),
        "inflation_rate": convert_list_to_array_float_or_array(
            data_input=data["contract_arguments"]["inflation_rate"]
        ),
        "inflation_rate_applied_to": convert_str_to_inflationappliedto(
            str_object=data["contract_arguments"]["inflation_rate_applied_to"]
        ),
    }

    contract.run(**contract_arguments_dict)

    summary_arguments_dict = get_summary_dict(data=data)
    summary_arguments_dict["contract"] = contract
    return get_summary(**summary_arguments_dict)
    
=========================================================================================
    
def get_costrecovery(data: dict):
    (
        start_date,
        end_date,
        oil_onstream_date,
        gas_onstream_date,
        lifting,
        tangible,
        intangible,
        opex,
        asr,
        lbt,
        cost_of_sales,
    ) = get_setup_dict(data=data)

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
        oil_ftp_is_available=data["costrecovery"]["oil_ftp_is_available"],
        oil_ftp_is_shared=data["costrecovery"]["oil_ftp_is_shared"],
        oil_ftp_portion=convert_to_float(
            target=data["costrecovery"]["oil_ftp_portion"]
        ),
        gas_ftp_is_available=data["costrecovery"]["gas_ftp_is_available"],
        gas_ftp_is_shared=data["costrecovery"]["gas_ftp_is_shared"],
        gas_ftp_portion=convert_to_float(
            target=data["costrecovery"]["gas_ftp_portion"]
        ),
        tax_split_type=convert_str_to_taxsplit(
            str_object=data["costrecovery"]["tax_split_type"]
        ),
        condition_dict=data["costrecovery"]["condition_dict"],
        indicator_rc_icp_sliding=convert_list_to_array_float(
            data_list=data["costrecovery"]["indicator_rc_icp_sliding"]
        ),
        oil_ctr_pretax_share=convert_to_float(
            target=data["costrecovery"]["oil_ctr_pretax_share"]
        ),
        gas_ctr_pretax_share=convert_to_float(
            target=data["costrecovery"]["gas_ctr_pretax_share"]
        ),
        oil_ic_rate=convert_to_float(target=data["costrecovery"]["oil_ic_rate"]),
        gas_ic_rate=convert_to_float(target=data["costrecovery"]["gas_ic_rate"]),
        ic_is_available=data["costrecovery"]["ic_is_available"],
        oil_cr_cap_rate=convert_to_float(
            target=data["costrecovery"]["oil_cr_cap_rate"]
        ),
        gas_cr_cap_rate=convert_to_float(
            target=data["costrecovery"]["gas_cr_cap_rate"]
        ),
        oil_dmo_volume_portion=convert_to_float(
            target=data["costrecovery"]["oil_dmo_volume_portion"]
        ),
        oil_dmo_fee_portion=convert_to_float(
            target=data["costrecovery"]["oil_dmo_fee_portion"]
        ),
        oil_dmo_holiday_duration=data["costrecovery"]["oil_dmo_holiday_duration"],
        gas_dmo_volume_portion=convert_to_float(
            target=data["costrecovery"]["gas_dmo_volume_portion"]
        ),
        gas_dmo_fee_portion=convert_to_float(
            target=data["costrecovery"]["gas_dmo_fee_portion"]
        ),
        gas_dmo_holiday_duration=data["costrecovery"]["gas_dmo_holiday_duration"],
        oil_carry_forward_depreciation=data["costrecovery"][
            "oil_carry_forward_depreciation"
        ],
        gas_carry_forward_depreciation=data["costrecovery"][
            "gas_carry_forward_depreciation"
        ],
    )

    # Filling the arguments of the contract with the data input
    contract_arguments_dict = {
        "sulfur_revenue": convert_str_to_otherrevenue(
            str_object=data["contract_arguments"]["sulfur_revenue"]
        ),
        "electricity_revenue": convert_str_to_otherrevenue(
            str_object=data["contract_arguments"]["electricity_revenue"]
        ),
        "co2_revenue": convert_str_to_otherrevenue(
            str_object=data["contract_arguments"]["co2_revenue"]
        ),
        "is_dmo_end_weighted": data["contract_arguments"]["is_dmo_end_weighted"],
        "tax_regime": convert_str_to_taxregime(
            str_object=data["contract_arguments"]["tax_regime"]
        ),
        "effective_tax_rate": convert_list_to_array_float_or_array_or_none(
            data_list=data["contract_arguments"]["effective_tax_rate"]
        ),
        "ftp_tax_regime": convert_str_to_ftptaxregime(
            str_object=data["contract_arguments"]["ftp_tax_regime"]
        ),
        "sunk_cost_reference_year": data["contract_arguments"][
            "sunk_cost_reference_year"
        ],
        "depr_method": convert_str_to_depremethod(
            str_object=data["contract_arguments"]["depr_method"]
        ),
        "decline_factor": data["contract_arguments"]["decline_factor"],
        "vat_rate": convert_list_to_array_float_or_array(
            data_input=data["contract_arguments"]["vat_rate"]
        ),
        "inflation_rate": convert_list_to_array_float_or_array(
            data_input=data["contract_arguments"]["inflation_rate"]
        ),
        "inflation_rate_applied_to": convert_str_to_inflationappliedto(
            str_object=data["contract_arguments"]["inflation_rate_applied_to"]
        ),
        "post_uu_22_year2001": (
            True
            if "post_uu_22_year2001" not in data["contract_arguments"]
            else data["contract_arguments"]["post_uu_22_year2001"]
        ),
        "oil_cost_of_sales_applied": (
            False
            if "oil_cost_of_sales_applied" not in data["contract_arguments"]
            else data["contract_arguments"]["oil_cost_of_sales_applied"]
        ),
        "gas_cost_of_sales_applied": (
            False
            if "gas_cost_of_sales_applied" not in data["contract_arguments"]
            else data["contract_arguments"]["gas_cost_of_sales_applied"]
        ),
        "sum_undepreciated_cost": (
            False
            if "sum_undepreciated_cost" not in data["contract_arguments"]
            else data["contract_arguments"]["sum_undepreciated_cost"]
        ),
    }

    # Running the contract
    contract.run(**contract_arguments_dict)

    # Filling the summary arguments
    summary_arguments_dict = get_summary_dict(data=data)
    summary_arguments_dict["contract"] = contract
    return get_summary(**summary_arguments_dict)
    
=========================================================================================
    
def get_grosssplit(data: dict):
    (
        start_date,
        end_date,
        oil_onstream_date,
        gas_onstream_date,
        lifting,
        tangible,
        intangible,
        opex,
        asr,
        lbt,
        cost_of_sales,
    ) = get_setup_dict(data=data)

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
        field_status=data["grosssplit"]["field_status"],
        field_loc=data["grosssplit"]["field_loc"],
        res_depth=data["grosssplit"]["res_depth"],
        infra_avail=data["grosssplit"]["infra_avail"],
        res_type=data["grosssplit"]["res_type"],
        api_oil=data["grosssplit"]["api_oil"],
        domestic_use=data["grosssplit"]["domestic_use"],
        prod_stage=data["grosssplit"]["prod_stage"],
        co2_content=data["grosssplit"]["co2_content"],
        h2s_content=data["grosssplit"]["h2s_content"],
        base_split_ctr_oil=convert_to_float(
            target=data["grosssplit"]["base_split_ctr_oil"]
        ),
        base_split_ctr_gas=convert_to_float(
            target=data["grosssplit"]["base_split_ctr_gas"]
        ),
        split_ministry_disc=convert_to_float(
            target=data["grosssplit"]["split_ministry_disc"]
        ),
        oil_dmo_volume_portion=convert_to_float(
            target=data["grosssplit"]["oil_dmo_volume_portion"]
        ),
        oil_dmo_fee_portion=convert_to_float(
            target=data["grosssplit"]["oil_dmo_fee_portion"]
        ),
        oil_dmo_holiday_duration=data["grosssplit"]["oil_dmo_holiday_duration"],
        gas_dmo_volume_portion=convert_to_float(
            target=data["grosssplit"]["gas_dmo_volume_portion"]
        ),
        gas_dmo_fee_portion=convert_to_float(
            target=data["grosssplit"]["gas_dmo_fee_portion"]
        ),
        gas_dmo_holiday_duration=data["grosssplit"]["gas_dmo_holiday_duration"],
        oil_carry_forward_depreciation=data["grosssplit"][
            "oil_carry_forward_depreciation"
        ],
        gas_carry_forward_depreciation=data["grosssplit"][
            "gas_carry_forward_depreciation"
        ],
    )

    # Filling the arguments of the contract with the data input
    contract_arguments_dict = {
        "sulfur_revenue": convert_str_to_otherrevenue(
            str_object=data["contract_arguments"]["sulfur_revenue"]
        ),
        "electricity_revenue": convert_str_to_otherrevenue(
            str_object=data["contract_arguments"]["electricity_revenue"]
        ),
        "co2_revenue": convert_str_to_otherrevenue(
            str_object=data["contract_arguments"]["co2_revenue"]
        ),
        "is_dmo_end_weighted": data["contract_arguments"]["is_dmo_end_weighted"],
        "tax_regime": convert_str_to_taxregime(
            str_object=data["contract_arguments"]["tax_regime"]
        ),
        "effective_tax_rate": convert_list_to_array_float_or_array_or_none(
            data_list=data["contract_arguments"]["effective_tax_rate"]
        ),
        "sunk_cost_reference_year": data["contract_arguments"][
            "sunk_cost_reference_year"
        ],
        "depr_method": convert_str_to_depremethod(
            str_object=data["contract_arguments"]["depr_method"]
        ),
        "decline_factor": data["contract_arguments"]["decline_factor"],
        "vat_rate": convert_list_to_array_float_or_array(
            data_input=data["contract_arguments"]["vat_rate"]
        ),
        "inflation_rate": convert_list_to_array_float_or_array(
            data_input=data["contract_arguments"]["inflation_rate"]
        ),
        "inflation_rate_applied_to": convert_str_to_inflationappliedto(
            str_object=data["contract_arguments"]["inflation_rate_applied_to"]
        ),
        "cum_production_split_offset": convert_list_to_array_float_or_array(
            data_input=data["contract_arguments"]["cum_production_split_offset"]
        ),
        "amortization": data["contract_arguments"]["amortization"],
        "regime": convert_grosssplitregime_to_enum(
            target=data["contract_arguments"]["regime"]
        ),
        "sum_undepreciated_cost": (
            False
            if "sum_undepreciated_cost" not in data["contract_arguments"]
            else data["contract_arguments"]["sum_undepreciated_cost"]
        ),
    }

    # Running the contract
    contract.run(**contract_arguments_dict)

    # Filling the summary arguments
    summary_arguments_dict = get_summary_dict(data=data)
    summary_arguments_dict["contract"] = contract
    return get_summary(**summary_arguments_dict)
    
=========================================================================================
    
csummary = (
                get_costrecovery(data=dataAdj) if self.type == 1
                else (
                    get_grosssplit(data=dataAdj) if self.type == 2
                    else (
                        get_transition(data=dataAdj) if self.type >= 3
                        else get_baseproject(data=dataAdj)
                    )
                )
            )
        
=========================================================================================    
            
def Adjust_Data(self, multipliers: np.ndarray):

        Adj_Contract = copy.deepcopy(self.baseContract)

        def Adj_Partial_Data(
            contract_: dict,
            par: str,
            key: str,
            multiplier: float,
            datakeys: list = [],
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

        # Specify contract based on contract type
        # for iloop in range(2 if self.type >= 3 else 1):
        contract_ = (
            # Adj_Contract if self.type < 3 else Adj_Contract[f"contract_{iloop+1}"]
            Adj_Contract
            if self.type < 3
            else Adj_Contract[f"contract_{2}"]
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
        
=========================================================================================
        
def calculate(self):

        # Designate a container to store Monte Carlo simulation results
        results = np.zeros(
            [self.numSim, len(self.target) + len(self.parameter)], dtype=np.float64
        )

        # Execute MonteCarlo simulation using pathos multiprocessing
        from pathos.multiprocessing import ProcessingPool as Pool

        with Pool() as pool:
            futures = pool.map(self.calcContract, range(self.numSim))

        for res in futures:
            results[res["n"], 0 : len(self.target)] = res["output"]
            results[res["n"], len(self.target) :] = [
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
        
=========================================================================================
        
# Iterate over the dictionary
    for key in list(min_max_mean_std.keys()):
        if key.startswith("min_"):
            base = key[4:]  # Base key (e.g., 'capex', 'opex', etc.)

            min_key = f"min_{base}"
            mean_key = f"mean_{base}"
            max_key = f"max_{base}"

            # Check if min, mean, and max values are the same
            if (
                min_key in min_max_mean_std
                and mean_key in min_max_mean_std
                and max_key in min_max_mean_std
                and min_max_mean_std[min_key]
                == min_max_mean_std[mean_key]
                == min_max_mean_std[max_key]
            ):
                # Adjust min and max values
                # +++ Set min to 0.8 of the min
                min_max_mean_std[min_key] = (0.8 * min_max_mean_std[min_key])

                # +++ Set max to 1.2 of the max
                min_max_mean_std[max_key] = (1.2 * min_max_mean_std[max_key])
                
==========================================================================================

    for base in bases:
        min_key, mean_key, max_key = f"min_{base}", f"mean_{base}", f"max_{base}"

        # Exit loop if at least one of ["min_key", "mean_key", "max_key"] is not
        # available in variable "min_max_mean_std".
        if not all([k in min_max_mean_std for k in [min_key, mean_key, max_key]]):
            continue

        # Extract statistics (min, mean, max) from variable "min_max_mean_std"
        min_val, mean_val, max_val = (
            min_max_mean_std[min_key],
            min_max_mean_std[mean_key],
            min_max_mean_std[max_key],
        )

        # Carry out adjustment if min_val == mean_val == max_val.
        # Adjustment is invoked on "min" and "max" values of a particular keyword
        # in variable "min_max_mean_std" by applying default multipliers.
        if min_val == mean_val == max_val:
            min_max_mean_std[min_key] = min_factor * min_val
            min_max_mean_std[max_key] = max_factor * max_val
            
==========================================================================================

# Modify attribute `hasGas` if GAS is present as a lifting commodity
        for i, _ in enumerate(self.parameter):
            if self.parameter[i]["id"] == 1:
                self.hasGas = True
                break     
                
==========================================================================================

        def _adjust_partial_data(
            contract_: dict,
            target_param: str,
            key: str,
            multiplier: float,
            datakeys: list | None = None,
        ):
            if datakeys is None:
                datakeys = []

            for item_key, item in contract_[key].items():

                # Skip GAS lifting adjustment for "Lifting" target parameter
                if (
                    target_param == "Lifting" and key == "lifting"
                    and item["fluid_type"] == "Gas"
                ):
                    continue

                # Specify target keys for lifting-related target
                if key == "lifting":
                    if target_param == "Oil Price" and item["fluid_type"] == "Oil":
                        target_keys = ["price"]
                    elif target_param == "Gas Price" and item["fluid_type"] == "Gas":
                        target_keys = ["price"]
                    elif target_param == "Lifting":
                        target_keys = ["lifting_rate", "prod_rate"]
                    else:
                        continue

                # Specify target keys for non-lifting targets
                else:
                    target_keys = datakeys

                # Adjust target attributes by multiplication with the prescribed multipliers
                for k in target_keys:
                    if k in item:
                        item[k] = (np.array(item[k]) * multiplier).tolist()

        # Specify attribute `contract_` based on `contract_type`
        contract_ = (
            contract_adjusted if self.type < 3 else contract_adjusted[f"contract_{2}"]
        )
        
        # Contract adjustments per single run simulation (i.e., per row)
        for i, val in enumerate(self.parameter):

            # Target parameter: Oil Price
            if val["id"] == 0:
                _adjust_partial_data(
                    contract_=contract_,
                    target_param="Oil Price",
                    key="lifting",
                    multiplier=multipliers[i]
                )

            # Target parameter: Gas Price
            elif val["id"] == 1:
                _adjust_partial_data(
                    contract_=contract_,
                    target_param="Gas Price",
                    key="lifting",
                    multiplier=multipliers[i],
                )

            # Target parameter: OPEX
            elif val["id"] == 2:
                # Adjust class OPEX
                _adjust_partial_data(
                    contract_=contract_,
                    target_param="OPEX",
                    key="opex",
                    multiplier=multipliers[i],
                    datakeys=["fixed_cost", "cost_per_volume", "cost"],
                )

                # Adjust class ASR
                _adjust_partial_data(
                    contract_=contract_,
                    target_param="ASR",
                    key="asr",
                    multiplier=multipliers[i],
                    datakeys=["cost"],
                )

                # Adjust class LBT
                _adjust_partial_data(
                    contract_=contract_,
                    target_param="LBT",
                    key="lbt",
                    multiplier=multipliers[i],
                    datakeys=["cost"],
                )

                # Adjust class CostOfSales
                _adjust_partial_data(
                    contract_=contract_,
                    target_param="COS",
                    key="cost_of_sales",
                    multiplier=multipliers[i],
                    datakeys=["cost"],
                )

            # Target parameter: CAPEX
            elif val["id"] == 3:
                # Adjust class CapitalCost
                _adjust_partial_data(
                    contract_=contract_,
                    target_param="CAPEX",
                    key="capital",
                    multiplier=multipliers[i],
                    datakeys=["cost"],
                )

                # Adjust class Intangible
                _adjust_partial_data(
                    contract_=contract_,
                    target_param="CAPEX",
                    key="intangible",
                    multiplier=multipliers[i],
                    datakeys=["cost"],
                )

            # Target parameter: Lifting
            elif val["id"] == 4:
                _adjust_partial_data(
                    contract_=contract_,
                    target_param="Lifting",
                    key="lifting",
                    multiplier=multipliers[i],
                )

        return contract_
        
==========================================================================================
"""
