"""
This file is utilized to be the adapter of the router into the core codes.
"""

import numpy as np
import pandas as pd
from datetime import datetime

from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts.transition import Transition
from pyscnomics.optimize import sensitivity_psc, uncertainty_psc
# from pyscnomics.tools.summary import get_summary
from pyscnomics.tools.table import get_table
from pyscnomics.optimize.optimization import optimize_psc
from pyscnomics.optimize.optimization_transition import (
    optimize_psc_core as optimize_psc_trans,
)
from pyscnomics.econ.selection import (
    OptimizationParameter,
    FluidType,
    SunkCostMethod,
)
from pyscnomics.tools.ltp import oil_ltp_predict, gas_ltp_predict
from pyscnomics.tools.rpd import RPDModel
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
    convert_summary_to_dict,
    convert_str_to_optimization_parameters,
    convert_str_to_optimization_targetparameter,
    convert_grosssplitregime_to_enum,
    convert_to_float,
    read_fluid_type,
    convert_to_method_limit,
    convert_to_uncertainty_distribution,
    convert_to_skk_summary_baseproject,
    converter_sunk_cost_method,
    converter_reservoir_type_permen_2024,
    converter_initial_amortization_year,
)
from pyscnomics.econ.limit import econ_limit


class ContractException(Exception):
    """ Exception to be raised if contract type is misused """

    pass


class LTPModelException(Exception):
    """ Exception to be raised for an incorrect LTP configurations """

    pass


class RDPModelException(Exception):
    """ Exception to be raised for an incorrect RDP configurations """

    pass


def get_setup_dict(data: dict) -> tuple:
    """
    Convert the setup section of the input dictionary into structured core engine data.

    This function parses the setup information and cost-related sections
    (e.g., capital, intangible, opex, ASR, LBT, and cost of sales)
    from the input dictionary into standardized dataclass-based objects.
    It ensures each section conforms to the expected internal data structure
    used by the core economic engine.

    Parameters
    ----------
    data : dict
        The full project setup dictionary containing general setup information
        (e.g., start/end dates, approval year, POD status) and financial data
        such as capital, intangible, opex, ASR, LBT, and cost-of-sales details.

        Expected structure example:
            {
                "setup": {
                    "start_date": "2020-01-01",
                    "end_date": "2035-12-31",
                    "approval_year": "2020",
                    "is_pod_1": True,
                    "oil_onstream_date": "2022-06-01",
                    "gas_onstream_date": "2023-01-01"
                },
                "capital": {...},
                "intangible": {...},
                "opex": {...},
                "asr": {...},
                "lbt": {...},
                "cost_of_sales": {...},
                "lifting": {...}
            }

    Returns
    -------
    tuple
        A tuple containing parsed and converted project setup components:

        - **start_date** : `datetime.date`
          Project start date.
        - **end_date** : `datetime.date`
          Project end date.
        - **oil_onstream_date** : `datetime.date` or `None`
          Oil production start date, if available.
        - **gas_onstream_date** : `datetime.date` or `None`
          Gas production start date, if available.
        - **approval_year** : `int` or `None`
          Project approval year.
        - **is_pod_1** : `bool`
          Indicator whether the project is POD-1.
        - **lifting** : `Lifting` or `None`
          Lifting configuration, converted using `convert_dict_to_lifting()`.
        - **capital** : `CapitalCost` or `None`
          Capital expenditure data, converted using `convert_dict_to_capital()`.
        - **intangible** : `Intangible` or `None`
          Intangible expenditure data, converted using `convert_dict_to_intangible()`.
        - **opex** : `OPEX` or `None`
          Operating expenditure data, converted using `convert_dict_to_opex()`.
        - **asr** : `ASR` or `None`
          Abandonment and Site Restoration data, converted using `convert_dict_to_asr()`.
        - **lbt** : `LBT` or `None`
          Land and Building Tax data, converted using `convert_dict_to_lbt()`.
        - **cost_of_sales** : `CostOfSales` or `None`
          Cost of sales data, converted using `convert_dict_to_cost_of_sales()`.
    """

    # Parsing the contract setup into each corresponding variables
    start_date = convert_str_to_date(str_object=data["setup"]["start_date"])
    end_date = convert_str_to_date(str_object=data["setup"]["end_date"])
    oil_onstream_date = convert_str_to_date(
        str_object=data["setup"].get("oil_onstream_date", None)
    )
    gas_onstream_date = convert_str_to_date(
        str_object=data["setup"].get("gas_onstream_date", None)
    )
    approval_year = convert_str_to_int(str_object=data["setup"]["approval_year"])
    is_pod_1 = data["setup"]["is_pod_1"]
    lifting = convert_dict_to_lifting(data_raw=data) if "lifting" in data else None
    capital = convert_dict_to_capital(
        data_raw=data["capital"] if "capital" in data else None
    )
    intangible = convert_dict_to_intangible(
        data_raw=data["intangible"] if "intangible" in data else None
    )
    opex = convert_dict_to_opex(data_raw=data["opex"]) if "opex" in data else None
    asr = convert_dict_to_asr(data_raw=data["asr"]) if "asr" in data else None
    lbt = convert_dict_to_lbt(data_raw=data["lbt"]) if "lbt" in data else None
    cost_of_sales = convert_dict_to_cost_of_sales(
        data_raw=data["cost_of_sales"] if "cost_of_sales" in data else None
    )

    return (
        start_date,
        end_date,
        oil_onstream_date,
        gas_onstream_date,
        approval_year,
        is_pod_1,
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
    Extract and convert summary-related parameters from the input dictionary
    into a standardized format accepted by the core engine.

    Parameters
    ----------
    data : dict
        The input data dictionary containing the `"summary_arguments"` key,
        which stores summary-level project parameters.

    Returns
    -------
    summary_arguments_dict : dict
        A dictionary containing the processed summary arguments in the
        core engine–compatible format, with the following keys:

        - **discount_rate_start_year** : int or None
          The project year at which the discount rate becomes effective.
        - **inflation_rate** : float or None
          The annual inflation rate applied to cost and revenue projections.
        - **discount_rate** : float
          The discount rate used in NPV calculation (default is 0.1 if unspecified).
        - **npv_mode** : NPVMode
          The NPV mode converted from string representation
          (default is `"Full Cycle Nominal Terms"`).
        - **discounting_mode** : DiscountingMode
          The discounting mode converted from string representation
          (default is `"discounting_mode"` if unspecified).
        - **profitability_discounted** : bool
          Flag indicating whether profitability metrics should be discounted
          (default is `False`).

    Notes
    -----
    - Missing keys in the `"summary_arguments"` section of the input are
      replaced with predefined default values where applicable.
    - String-based parameters such as `"npv_mode"` and `"discounting_mode"`
      are converted into their corresponding enumeration types via helper
      functions (e.g., `convert_str_to_npvmode()`).
    """

    # Fill get_summary() argument with input data
    discount_rate_start_year = data["summary_arguments"].get("discount_rate_start_year", None)
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
    Build and initialize a Base Project contract instance.

    This function extracts fundamental parameters from the input data,
    prepares the required keyword arguments, and returns a fully
    constructed :class:`BaseProject` instance for economic evaluation.

    Parameters
    ----------
    data : dict
        Input dictionary containing all parameters necessary to
        configure a Base Project contract.

    Returns
    -------
    BaseProject
        Initialized :class:`BaseProject` object ready for evaluation.
    """

    # Specify base arguments
    (
        start_date,
        end_date,
        oil_onstream_date,
        gas_onstream_date,
        approval_year,
        is_pod_1,
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
        "oil_onstream_date": oil_onstream_date,
        "gas_onstream_date": gas_onstream_date,
        "approval_year": approval_year,
        "is_pod_1": is_pod_1,

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
    Build the argument dictionary for a Base Project contract.

    This function extracts and converts key economic parameters such as
    revenues, tax rate, and inflation information from the input data,
    returning a dictionary suitable for initializing or executing a
    :class:`BaseProject` instance.

    Parameters
    ----------
    data : dict
        Input dictionary containing contract arguments and economic
        parameters for the Base Project.

    Returns
    -------
    dict
        Dictionary of processed Base Project arguments, ready for use
        in model evaluation.
    """

    # Specify abbreviations
    ca = data["contract_arguments"]
    f_rev = convert_str_to_otherrevenue
    f_rate = convert_list_to_array_float_or_array
    f_infl = convert_str_to_inflationappliedto

    return {
        # Other revenues
        "sulfur_revenue": f_rev(str_object=ca["sulfur_revenue"]),
        "electricity_revenue": f_rev(str_object=ca["electricity_revenue"]),
        "co2_revenue": f_rev(str_object=ca["co2_revenue"]),

        # VAT and inflation
        "vat_rate": f_rate(data_input=ca["vat_rate"]),
        "inflation_rate": f_rate(data_input=ca["inflation_rate"]),
        "inflation_rate_applied_to": f_infl(str_object=ca["inflation_rate_applied_to"]),
    }


def get_baseproject(data: dict, summary_result: bool = True):
    """
    Build, execute, and optionally summarize a Base Project contract evaluation.

    This function prepares all necessary inputs, constructs a :class:`BaseProject`
    instance, executes its economic evaluation, and optionally generates an
    executive summary formatted according to SKK Migas standards.

    Parameters
    ----------
    data : dict
        Input dictionary containing setup information, contract arguments,
        and summary parameters required for Base Project evaluation.
    summary_result : bool, default=True
        If ``True``, generate and return the SKK Migas–formatted summary.
        If ``False``, return only the contract instance and its arguments.

    Returns
    -------
    tuple
        A 4-element tuple containing:

        - **summary_skk** : dict or None
          Executive summary in SKK-compatible format, or ``None`` if
          ``summary_result=False``.
        - **contract** : BaseProject
          Executed :class:`BaseProject` instance.
        - **contract_arguments_dict** : dict
          Dictionary of arguments passed to :meth:`BaseProject.run`.
        - **summary_arguments_dict** : dict or None
          Summary argument dictionary, or ``None`` if summary generation
          was skipped.

    Notes
    -----
    The function performs the following key steps:

    1. Instantiates the :class:`BaseProject` object via
       :func:`build_baseproject_instance`.
    2. Prepares contract arguments using :func:`build_baseproject_arguments`.
    3. Executes the contract with :meth:`BaseProject.run`.
    4. Optionally generates an SKK Migas–formatted summary using
       :func:`convert_to_skk_summary_baseproject` and appends execution info.
    """

    # Specify contract and contract arguments
    contract = build_baseproject_instance(data=data)
    contract_arguments_dict = build_baseproject_arguments(data=data)

    # Execute BaseProject instance
    contract.run(**contract_arguments_dict)

    # Display (or undisplay) summary
    if summary_result is True:
        # Fill summary arguments
        summary_arguments_dict = get_summary_dict(data=data)
        summary = contract.get_summary(**summary_arguments_dict)

        # Display summary using SKK Migas format
        summary_skk = convert_to_skk_summary_baseproject(dict_object=summary)

        # Add execution info
        summary_skk = add_execution_info(data=summary_skk)

    # Since the required object is only the contract, it will return None for the summary
    else:
        summary_skk = None
        summary_arguments_dict = None

    return summary_skk, contract, contract_arguments_dict, summary_arguments_dict


def build_costrecovery_instance(data: dict) -> CostRecovery:
    """
    Build and return a fully configured `CostRecovery` contract instance.

    This function extracts setup parameters and cost recovery–specific
    attributes from the input data dictionary, applies the necessary
    type conversions, and initializes the `CostRecovery` object with
    the prepared keyword arguments.

    Parameters
    ----------
    data : dict
        Input data dictionary containing setup parameters and
        `costrecovery`-specific contract attributes. Must include
        sections such as `lifting`, `capital`, `opex`, and
        `costrecovery`.

    Returns
    -------
    CostRecovery
        The instantiated `CostRecovery` contract object, ready for
        execution using its `run()` method.

    Notes
    -----
    - Setup parameters (dates, lifting, and costs) are parsed via
      :func:`get_setup_dict`.
    - Conversion utilities like :func:`convert_list_to_array_float_or_array`
      and :func:`convert_str_to_taxsplit` are used for type coercion.
    - The function prepares all base, FTP, split, investment credit,
      DMO, and depreciation parameters required for initialization.
    """

    # Specify base arguments
    (
        start_date,
        end_date,
        oil_onstream_date,
        gas_onstream_date,
        approval_year,
        is_pod_1,
        lifting,
        capital,
        intangible,
        opex,
        asr,
        lbt,
        cost_of_sales,
    ) = get_setup_dict(data=data)

    # Specify abbreviations
    cr = data["costrecovery"]
    f_rate = convert_list_to_array_float_or_array
    f_split = convert_str_to_taxsplit
    f_icp = convert_list_to_array_float
    f_float = convert_to_float

    # Prepare contract attributes for CostRecovery
    contract_kwargs = {
        # Base parameters
        "start_date": start_date,
        "end_date": end_date,
        "oil_onstream_date": oil_onstream_date,
        "gas_onstream_date": gas_onstream_date,
        "approval_year": approval_year,
        "is_pod_1": is_pod_1,

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
        "oil_ftp_portion": f_rate(data_input=cr["oil_ftp_portion"]),
        "gas_ftp_is_available": cr["gas_ftp_is_available"],
        "gas_ftp_is_shared": cr["gas_ftp_is_shared"],
        "gas_ftp_portion": f_rate(data_input=cr["gas_ftp_portion"]),

        # Split
        "tax_split_type": f_split(str_object=cr["tax_split_type"]),
        "condition_dict": cr["condition_dict"],
        "indicator_rc_icp_sliding": f_icp(data_list=cr["indicator_rc_icp_sliding"]),
        "oil_ctr_pretax_share": f_rate(data_input=cr["oil_ctr_pretax_share"]),
        "gas_ctr_pretax_share": f_rate(data_input=cr["gas_ctr_pretax_share"]),

        # Investment credit and cap rate
        "oil_ic_rate": f_float(target=cr["oil_ic_rate"]),
        "gas_ic_rate": f_float(target=cr["gas_ic_rate"]),
        "ic_is_available": cr["ic_is_available"],
        "oil_cr_cap_rate": f_float(target=cr["oil_cr_cap_rate"]),
        "gas_cr_cap_rate": f_float(target=cr["gas_cr_cap_rate"]),

        # DMO
        "oil_dmo_volume_portion": f_rate(data_input=cr["oil_dmo_volume_portion"]),
        "oil_dmo_fee_portion": f_rate(data_input=cr["oil_dmo_fee_portion"]),
        "oil_dmo_holiday_duration": cr["oil_dmo_holiday_duration"],
        "gas_dmo_volume_portion": f_rate(data_input=cr["gas_dmo_volume_portion"]),
        "gas_dmo_fee_portion": f_rate(data_input=cr["gas_dmo_fee_portion"]),
        "gas_dmo_holiday_duration": cr["gas_dmo_holiday_duration"],

        # Carry forward depreciation
        "oil_carry_forward_depreciation": 0.0,
        "gas_carry_forward_depreciation": 0.0,
    }

    return CostRecovery(**contract_kwargs)


def build_costrecovery_arguments(data: dict) -> dict:
    """
    Build and return the argument dictionary required to execute a
    `CostRecovery` contract.

    This function extracts contract-level parameters from the input
    data dictionary, performs necessary type conversions, and prepares
    all supporting arguments related to revenue, FTP, VAT, inflation,
    tax, depreciation, and cost of sales. The resulting dictionary can
    be directly passed to the `CostRecovery.run()` method.

    Parameters
    ----------
    data : dict
        Input data dictionary containing the section `contract_arguments`
        with all parameter definitions required for the cost recovery
        contract setup.

    Returns
    -------
    dict
        Dictionary of processed arguments to initialize or execute
        the `CostRecovery` contract.

    Notes
    -----
    - Type conversion functions such as
      :func:`convert_str_to_otherrevenue`, :func:`convert_str_to_taxregime`,
      and :func:`convert_str_to_depremethod` are used for input parsing.
    - Internal helper function `_get_value()` safely retrieves dictionary
      values and applies optional conversion or defaults when keys are
      missing or `None`.
    - Covers the following parameter groups:
        * Other revenues (e.g., sulfur, electricity, CO₂)
        * FTP and tax regimes
        * VAT and inflation parameters
        * Depreciation configuration
        * DMO weighting and cost-of-sales flags
        * Sunk cost handling
    """

    # Specify abbreviations and helper method
    ca = data["contract_arguments"]
    f_rev = convert_str_to_otherrevenue
    f_rate = convert_list_to_array_float_or_array
    f_infl = convert_str_to_inflationappliedto
    f_tax = convert_str_to_taxregime
    f_tax_rate = convert_list_to_array_float_or_array_or_none
    f_ftp = convert_str_to_ftptaxregime
    f_depr = convert_str_to_depremethod

    def _get_value(key: str, source: dict = ca, default=True, converter=None):
        """
        Helper function:
        Safely retrieve a value from a dictionary with an optional converter.

        Parameters
        ----------
        key : str
            Key to look up in the dictionary.
        source : dict, optional
            Source dictionary. Defaults to ``ca``.
        default : any, optional
            Value to return if key is missing or ``None``.
        converter : callable, optional
            Function to convert the retrieved value.
        """
        if converter is None:
            return (
                default if (key not in source) or (source[key] is None)
                else source[key]
            )
        else:
            return (
                default if (key not in source) or (source[key] is None)
                else converter(source[key])
            )

    return {
        # Other revenues
        "sulfur_revenue": f_rev(str_object=ca["sulfur_revenue"]),
        "electricity_revenue": f_rev(str_object=ca["electricity_revenue"]),
        "co2_revenue": f_rev(str_object=ca["co2_revenue"]),

        # FTP
        "ftp_tax_regime": f_ftp(str_object=ca["ftp_tax_regime"]),

        # VAT and inflation
        "vat_rate": f_rate(data_input=ca["vat_rate"]),
        "inflation_rate": f_rate(data_input=ca["inflation_rate"]),
        "inflation_rate_applied_to": f_infl(str_object=ca["inflation_rate_applied_to"]),

        # DMO and tax
        "is_dmo_end_weighted": ca["is_dmo_end_weighted"],
        "tax_regime": f_tax(str_object=ca["tax_regime"]),
        "effective_tax_rate": f_tax_rate(data_list=ca["effective_tax_rate"]),
        "post_uu_22_year2001": _get_value(key="post_uu_22_year2001"),

        # Depreciation
        "depr_method": f_depr(str_object=ca["depr_method"]),
        "decline_factor": ca["decline_factor"],
        "sum_undepreciated_cost": _get_value(key="sum_undepreciated_cost", default=False),

        # Cost of sales
        "oil_cost_of_sales_applied": _get_value(key="oil_cost_of_sales_applied", default=False),
        "gas_cost_of_sales_applied": _get_value(key="gas_cost_of_sales_applied", default=False),

        # Sunk cost
        "sunk_cost_method": _get_value(
            key="sunk_cost_method",
            default=SunkCostMethod.DEPRECIATED_TANGIBLE,
            converter=converter_sunk_cost_method,
        ),
    }


def get_costrecovery(data: dict, summary_result: bool = True):
    """
    Execute a Cost Recovery PSC evaluation and optionally return the summary results.

    This function builds the contract instance, prepares input arguments, runs
    the Cost Recovery model, and optionally generates the SKK Migas–formatted
    summary.

    Parameters
    ----------
    data : dict
        Input data containing all parameters required for the Cost Recovery evaluation.
    summary_result : bool, default=True
        If True, return the SKK Migas–formatted summary; otherwise, omit it.

    Returns
    -------
    tuple
        (summary_skk, contract, contract_arguments_dict, summary_arguments_dict)

        - **summary_skk** : dict or None
          Summary of results in SKK Migas format, or None if not requested.
        - **contract** : CostRecovery
          Executed Cost Recovery contract instance.
        - **contract_arguments_dict** : dict
          Arguments used in :meth:`CostRecovery.run`.
        - **summary_arguments_dict** : dict or None
          Arguments used for summary generation, or None if skipped.
    """

    # Specify contract and contract arguments
    contract = build_costrecovery_instance(data=data)
    contract_arguments_dict = build_costrecovery_arguments(data=data)

    # Execute CostRecovery instance
    contract.run(**contract_arguments_dict)

    # Display (or undisplay) summary
    if summary_result is True:

        # Fill summary arguments
        summary_arguments_dict = get_summary_dict(data=data)
        # ==== IGNORED IN 1.4.0 =====
        # summary_arguments_dict["contract"] = contract
        # summary = get_summary(**summary_arguments_dict)
        summary = contract.get_summary(**summary_arguments_dict)

        # Display summary using SKK Migas format
        summary_skk = convert_summary_to_dict(dict_object=summary)

        # Add execution info
        summary_skk = add_execution_info(data=summary_skk)

    # Since the required object is only the contract, it will return None for the summary
    else:
        summary_skk = None
        summary_arguments_dict = None

    return summary_skk, contract, contract_arguments_dict, summary_arguments_dict


def build_grosssplit_instance(data: dict) -> GrossSplit:
    """
    Build and return an initialized :class:`GrossSplit` contract instance.

    This function extracts and transforms relevant input parameters from the
    provided ``data`` dictionary to create a fully configured instance of the
    :class:`GrossSplit` class.

    It handles type conversion, optional attribute retrieval, nd ensures missing
    values are safely replaced with defaults when applicable. The returned object
    is ready for execution through the :meth:`GrossSplit.run` method.

    Parameters
    ----------
    data : dict
        Dictionary containing the full dataset required to construct a
        Gross Split PSC (Production Sharing Contract) instance.

    Returns
    -------
    GrossSplit
        A fully configured :class:`GrossSplit` instance initialized with
        project setup parameters, fiscal settings, and DMO-related attributes.
        The instance is ready to run the Gross Split PSC economic model.

    Notes
    -----
    - The helper function ``_get_value()`` ensures missing or ``None`` attributes
      from the input dictionary are replaced with a default value (``None`` by default).
    - List-type fiscal inputs such as DMO portions are converted to NumPy arrays
      using :func:`convert_list_to_array_float_or_array`.
    """

    # Specify base arguments
    (
        start_date,
        end_date,
        oil_onstream_date,
        gas_onstream_date,
        approval_year,
        is_pod_1,
        lifting,
        capital,
        intangible,
        opex,
        asr,
        lbt,
        cost_of_sales,
    ) = get_setup_dict(data=data)

    # Specify abbreviations and helper method to be used in instance preparation
    gs = data["grosssplit"]
    func = convert_list_to_array_float_or_array

    def _get_value(key: str, source: dict = gs, default=None):
        """
        Helper function: safely retrieve a value from a dictionary.

        Parameters
        ----------
        key : str
            Key to look up in the dictionary.
        source : dict, optional
            Source dictionary. Defaults to ``ca``.
        default : any, optional
            Value to return if key is missing or ``None``.
        """
        return default if (key not in source) or (source[key] is None) else source[key]

    # Prepare contract attributes for GrossSplit
    contract_kwargs = {
        # Base parameters
        "start_date": start_date,
        "end_date": end_date,
        "oil_onstream_date": oil_onstream_date,
        "gas_onstream_date": gas_onstream_date,
        "approval_year": approval_year,
        "is_pod_1": is_pod_1,

        # Lifting and costs
        "lifting": lifting,
        "capital_cost": capital,
        "intangible_cost": intangible,
        "opex": opex,
        "asr_cost": asr,
        "lbt_cost": lbt,

        # Field and reservoir properties
        "field_status": _get_value(key="field_status"),
        "field_loc": _get_value(key="field_loc"),
        "res_depth": _get_value(key="res_depth"),
        "infra_avail": _get_value(key="infra_avail"),
        "res_type": _get_value(key="res_type"),
        "api_oil": _get_value(key="api_oil"),
        "domestic_use": _get_value(key="domestic_use"),
        "prod_stage": _get_value(key="prod_stage"),
        "co2_content": _get_value(key="co2_content"),
        "h2s_content": _get_value(key="h2s_content"),
        "field_reserves_2024": _get_value(key="field_reserves_2024"),
        "infra_avail_2024": _get_value(key="infra_avail_2024"),
        "field_loc_2024": _get_value(key="field_loc_2024"),
        "split_ministry_disc": convert_to_float(target=gs["split_ministry_disc"]),

        # DMO parameters
        "oil_dmo_volume_portion": func(data_input=gs["oil_dmo_volume_portion"]),
        "oil_dmo_fee_portion": func(data_input=gs["oil_dmo_fee_portion"]),
        "oil_dmo_holiday_duration": _get_value(key="oil_dmo_holiday_duration"),
        "gas_dmo_volume_portion": func(data_input=gs["gas_dmo_volume_portion"]),
        "gas_dmo_fee_portion": func(data_input=gs["gas_dmo_fee_portion"]),
        "gas_dmo_holiday_duration": _get_value(key="gas_dmo_holiday_duration"),

        # Carry forward depreciation
        "oil_carry_forward_depreciation": 0.0,
        "gas_carry_forward_depreciation": 0.0,
    }

    return GrossSplit(**contract_kwargs)


def build_grosssplit_arguments(data: dict) -> dict:
    """
    Build a dictionary of contract arguments for the Gross Split PSC scheme.

    This function extracts and converts values from the input ``data`` dictionary,
    applying type conversions and default values as necessary. It prepares a
    consistent set of parameters to initialize a ``GrossSplit`` instance or to
    perform economic evaluations under the Gross Split PSC regime.

    Parameters
    ----------
    data : dict
        Input dictionary containing contract-related information. Must include
        the key ``"contract_arguments"`` that holds all argument values. Expected
        structure:

        - ``data["contract_arguments"]["sulfur_revenue"]`` : str
        - ``data["contract_arguments"]["inflation_rate"]`` : list or float
        - ``data["contract_arguments"]["tax_regime"]`` : str
        - and so on.

    Returns
    -------
    dict
        A dictionary of processed contract arguments ready for initializing a
        ``GrossSplit`` instance.

    Notes
    -----
    - Missing or ``None`` values are replaced with the specified default in
      each converter call.
    - Optional keys are safely accessed using the internal helper method
      ``_get_value()``, which applies type conversion if a converter is provided.
    - The returned dictionary is intended for use as keyword arguments in
      creating a ``GrossSplit`` contract object.
    """

    # Specify abbreviations and helper method
    ca = data["contract_arguments"]
    f_rev = convert_str_to_otherrevenue
    f_rate = convert_list_to_array_float_or_array
    f_infl = convert_str_to_inflationappliedto
    f_depr = convert_str_to_depremethod
    f_tax = convert_str_to_taxregime
    f_tax_rate = convert_list_to_array_float_or_array_or_none
    f_regime = convert_grosssplitregime_to_enum
    f_2024 = converter_reservoir_type_permen_2024
    f_amor = converter_initial_amortization_year

    def _get_value(key: str, source: dict = ca, default=None, converter=None):
        """
        Helper function:
        Safely retrieve a value from a dictionary with an optional converter.

        Parameters
        ----------
        key : str
            Key to look up in the dictionary.
        source : dict, optional
            Source dictionary. Defaults to ``ca``.
        default : any, optional
            Value to return if key is missing or ``None``.
        converter : callable, optional
            Function to convert the retrieved value.
        """

        if converter is None:
            return (
                default if (key not in source) or (source[key] is None)
                else source[key]
            )
        else:
            return (
                default if (key not in source) or (source[key] is None)
                else converter(source[key])
            )

    return {
        # Other revenues
        "sulfur_revenue": f_rev(str_object=ca["sulfur_revenue"]),
        "electricity_revenue": f_rev(str_object=ca["electricity_revenue"]),
        "co2_revenue": f_rev(str_object=ca["co2_revenue"]),

        # VAT and inflation
        "vat_rate": f_rate(data_input=ca["vat_rate"]),
        "inflation_rate": f_rate(data_input=ca["inflation_rate"]),
        "inflation_rate_applied_to": f_infl(str_object=ca["inflation_rate_applied_to"]),

        # Production offset
        "cum_production_split_offset": f_rate(data_input=ca["cum_production_split_offset"]),

        # Depreciation
        "depr_method": f_depr(str_object=ca["depr_method"]),
        "decline_factor": ca["decline_factor"],
        "sum_undepreciated_cost": _get_value(key="sum_undepreciated_cost", default=False),

        # DMO and tax
        "is_dmo_end_weighted": ca["is_dmo_end_weighted"],
        "tax_regime": f_tax(str_object=ca["tax_regime"]),
        "effective_tax_rate": f_tax_rate(data_list=ca["effective_tax_rate"]),

        # Sunk cost
        "amortization": _get_value(key="amortization", default=False),
        "sunk_cost_method": _get_value(
            key="sunk_cost_method",
            default=SunkCostMethod.DEPRECIATED_TANGIBLE,
            converter=converter_sunk_cost_method,
        ),

        # Fiscal regime
        "regime": f_regime(target=ca["regime"]),
        "reservoir_type_permen_2024": f_2024(target_str=ca["reservoir_type_permen_2024"]),
        "initial_amortization_year": f_amor(target_str=ca["initial_amortization_year"]),
    }


def get_grosssplit(data: dict, summary_result: bool = True) -> tuple:
    """
    Execute a Gross Split PSC evaluation and optionally return the summary results.

    This function builds the contract instance, prepares input arguments, runs
    the Gross Split model, and optionally generates the SKK Migas–formatted
    summary.

    Parameters
    ----------
    data : dict
        Input data containing all parameters required for the Gross Split evaluation.
    summary_result : bool, default=True
        If True, return the SKK Migas–formatted summary; otherwise, omit it.

    Returns
    -------
    tuple
        (summary_skk, contract, contract_arguments_dict, summary_arguments_dict)

        - **summary_skk** : dict or None
          Summary of results in SKK Migas format, or None if not requested.
        - **contract** : GrossSplit
          Executed Gross Split contract instance.
        - **contract_arguments_dict** : dict
          Arguments used in :meth:`GrossSplit.run`.
        - **summary_arguments_dict** : dict or None
          Arguments used for summary generation, or None if skipped.
    """

    # Specify contract and contract arguments
    contract: GrossSplit = build_grosssplit_instance(data=data)
    contract_arguments_dict: dict = build_grosssplit_arguments(data=data)

    # Execute GrossSplit instance
    contract.run(**contract_arguments_dict)

    # Display (or undisplay) summary
    if summary_result is True:
        # Fill summary arguments
        summary_arguments_dict = get_summary_dict(data=data)
        summary = contract.get_summary(**summary_arguments_dict)

        # Display summary using SKK Migas format
        summary_skk = convert_summary_to_dict(dict_object=summary)

        # Add execution info
        summary_skk = add_execution_info(data=summary_skk)

    else:
        summary_skk = None
        summary_arguments_dict = None

    return summary_skk, contract, contract_arguments_dict, summary_arguments_dict


def get_grosssplit_split(data: dict) -> dict:
    """
    Retrieve contractor split information from the Gross Split PSC scheme.

    This function builds and executes a :class:`GrossSplit` contract instance
    based on the provided input data, then extracts yearly oil and gas
    split components, including base, variable, progressive, and total splits,
    as well as the years with maximum contractor split.

    Parameters
    ----------
    data : dict
        Input data containing parameters required for the Gross Split
        contract evaluation.

    Returns
    -------
    dict
        Dictionary with two main elements:

        - **contractor_split** : dict
          Yearly contractor split information for oil and gas, including base,
          variable, and progressive components, indexed by project years.

        - **years_of_maximum_split** : dict
          Years when the contractor achieved the maximum split for oil and gas.
    """

    # Specify contract and contract arguments
    contract = build_grosssplit_instance(data=data)
    contract_arguments_dict = build_grosssplit_arguments(data=data)

    # Execute GrossSplit intance
    contract.run(**contract_arguments_dict)

    # Convert contract to a dictionary using vars() --> cad = contract as dictionary
    cad = vars(contract)

    # Retrieving the split information
    contractor_split = (
        pd.DataFrame(
            {
                "project_years": contract.project_years.tolist(),
                "oil_base_split": cad["_oil_base_split"].tolist(),
                "gas_base_split": cad["_gas_base_split"].tolist(),
                "var_split_array": cad["_var_split_array"].tolist(),
                "oil_prog_price_split": cad["_oil_prog_price_split"].tolist(),
                "gas_prog_price_split": cad["_gas_prog_price_split"].tolist(),
                "oil_prog_cumulative_production_split": cad["_oil_prog_cum_split"].tolist(),
                "gas_prog_cumulative_production_split": cad["_gas_prog_cum_split"].tolist(),
                "oil_prog_total_split": cad["_oil_prog_split"].tolist(),
                "gas_prog_total_split": cad["_gas_prog_split"].tolist(),
                "oil_ctr_split": cad["_oil_ctr_split_prior_bracket"].tolist(),
                "gas_ctr_split": cad["_gas_ctr_split_prior_bracket"].tolist(),
            }
        )
        .set_index("project_years")
        .to_dict()
    )

    years_of_maximum_split = {
        "oil": cad["_oil_year_maximum_ctr_split"].tolist(),
        "gas": cad["_gas_year_maximum_ctr_split"].tolist(),
    }

    return {
        "contractor_split": contractor_split,
        "years_of_maximum_split": years_of_maximum_split,
    }


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
    contract:
        The Transition contract object.
    contract_arguments_dict: dict
        The contract arguments used in running the contract calculation.
    summary_arguments_dic: dict
        The summary arguments used in retrieving the executive summary of the contract.

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
        raise ContractException("Contract type is not recognized")

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
        raise ContractException("Contract type is not recognized")

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

    summary = get_summary(**summary_arguments_dict)

    # Converting the summary format to skk summary format
    summary_skk = convert_summary_to_dict(dict_object=summary)

    # Adding the execution info
    summary_skk = add_execution_info(data=summary_skk)

    return summary_skk, contract, contract_arguments_dict, summary_arguments_dict


def add_execution_info(data: dict) -> dict:
    """
    Function to adding the execution info into a dictionary.

    Parameters
    ----------
    data: dict
        The dictionary which will added with execution info

    Returns
    -------
    out: dict
        Dictionary containing the execution info
    """
    # Defining the execution date
    execution_date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    # Defining the PySCnomics Version
    from importlib.metadata import version, PackageNotFoundError

    try:
        package_version = version("pyscnomics")

    except PackageNotFoundError:
        package_version = "unknown"

    # The following codes are replaced in version 1.4.0
    # package_version = " "
    # try:
    #     from importlib.metadata import version
    #     package_version = version("pyscnomics")
    # except:
    #     pass

    # Parsing the data into the data output
    execution_info = {
        "execution_datetime": execution_date,
        "package_version": package_version,
    }

    data["execution_info"] = execution_info

    return data


def get_contract_table(data: dict, contract_type: str = "Cost Recovery") -> dict:
    """
    Function to get the cash flow table of the contract that has been run.

    Parameters
    ----------
    data: dict
        The dictionary of the data input.
    contract_type: str
        The option for the contract type.
        The available option are: ['Cost Recovery', 'Gross Split']

    Returns
    -------
    table_all_dict: dict
        The dictionary containing the oil, gas and consolidated cash flow table.
    """

    # Adjusting the variable to the corresponding contract type
    if contract_type == "Cost Recovery":
        contract = get_costrecovery(data=data)[1]
        year_column = "years"
        # year_column = "Year"

    elif contract_type == "Gross Split":
        contract = get_grosssplit(data=data)[1]
        year_column = "years"
        # year_column = "Years"

    elif contract_type == "Base Project":
        contract = get_baseproject(data=data)[1]
        year_column = "years"
        # year_column = "Years"

    else:
        contract = get_transition(data=data)[1]
        year_column = "years"
        # year_column = "Year"

    # Condition when the contract is Transition
    if contract_type == "Transition":
        # Retrieving the table
        table_oil, table_gas, table_consolidated = get_table(contract=contract)

        # Forming the table dictionary as the output
        table_all_dict = {
            "contract_1": {
                "oil": table_oil[0].set_index(table_oil[0].columns[0]).to_dict(),
                "gas": table_gas[0].set_index(table_gas[0].columns[0]).to_dict(),
                "consolidated": table_consolidated[0]
                .set_index(table_consolidated[0].columns[0])
                .to_dict(),
            },
            "contract_2": {
                "oil": table_oil[1].set_index(table_oil[1].columns[0]).to_dict(),
                "gas": table_gas[1].set_index(table_gas[1].columns[0]).to_dict(),
                "consolidated": table_consolidated[1]
                .set_index(table_consolidated[1].columns[0])
                .to_dict(),
            },
        }

    else:
        # Retrieving the table
        table_oil, table_gas, table_consolidated = get_table(contract=contract)

        # Forming the table dictionary as the output
        table_all_dict = {
            "oil": table_oil.set_index(year_column).to_dict(),
            "gas": table_gas.set_index(year_column).to_dict(),
            "consolidated": table_consolidated.set_index(year_column).to_dict(),
        }

    t1 = table_all_dict["oil"]["sunk_cost"]
    print('\t')
    print(f'Filetype: {type(t1)}')
    print(f'Length: {len(t1)}')
    print('t1 = \n', t1)


    # Adding the execution info
    table_all_dict = add_execution_info(data=table_all_dict)

    return table_all_dict


def get_detailed_summary(data: dict, contract_type: str) -> dict:
    """
    Generate a detailed economic summary for a given PSC (Production Sharing Contract)
    type.

    This method dynamically selects and executes the appropriate PSC computation
    function (e.g., Cost Recovery, Gross Split, Transition, or Base Project) based
    on the specified contract type.

    The function returns a detailed economic summary generated from the selected
    contract model.

    Parameters
    ----------
    data : dict
        Input data containing the economic parameters and production profile required
        for the PSC evaluation.
        The expected structure depends on the specific contract type.
    contract_type : str
        Type of PSC contract. Must be one of the following:
            - "Cost Recovery"
            - "Gross Split"
            - "Transition"
            - "Base Project"

    Returns
    -------
    dict
        A dictionary representing the detailed economic summary generated by the
        selected PSC model. The contents typically include key financial and fiscal
        indicators such as government take, contractor take, NPV, and other
        economic metrics derived from the contract's `get_summary` method.
    """

    contract_map = {
        "Cost Recovery": get_costrecovery,
        "Gross Split": get_grosssplit,
        "Transition": get_transition,
        "Base Project": get_baseproject,
    }

    if contract_type not in contract_map:
        raise ValueError(f"Invalid contract type: {contract_type!r}")

    func = contract_map[contract_type]
    _, contract, _, summary_args = func(data=data, summary_result=True)

    return contract.get_summary(**summary_args)


def get_ltp(
    volume: float, start_year: int, end_year: int, fluid_type: FluidType
) -> np.ndarray:
    """
    Function to get the ltp array.

    Parameters
    ----------
    volume:float
        The volume of the reserves.
    start_year:int
        The start year.
    end_year:int
        The end year.
    fluid_type: FluidType
        The FluidType of the corresponding volume.

    Returns
    -------
    out: np.ndarray
        The array of ltp

    """
    # Condition checking for the fluid type for initiating the array of ltp
    if fluid_type == FluidType.OIL:
        rate_ltp = oil_ltp_predict(volume=volume)

    elif fluid_type == FluidType.GAS:
        rate_ltp = gas_ltp_predict(volume=volume)

    else:
        raise LTPModelException(f"Unsupported Fluid Type {fluid_type} ")

    # Initiating the array of years
    year_arr = np.arange(start_year, end_year + 1)
    gap = len(year_arr) - len(rate_ltp)

    # Checking condition for the given years gap
    if gap < 0:
        raise LTPModelException(
            f"Forecast years from {start_year} to {end_year} is too short. "
            f"Please set the end_year at least until {end_year + (-1 * gap)}"
        )

    rate_gap = np.zeros(gap)
    rate_adjusted = np.concatenate([rate_gap, rate_ltp])

    return rate_adjusted


def get_ltp_dict(data: dict):
    """
    The function to get the list of LTP from the given reserves volume.

    Parameters
    ----------
    data: dict
        The dictionary of the data input.

    Returns
    -------
    """

    volume = data["volume"]
    start_year = data["start_year"]
    end_year = data["end_year"]
    fluid_type = read_fluid_type(fluid=data["fluid_type"])

    ltp_array = get_ltp(
        volume=volume,
        start_year=start_year,
        end_year=end_year,
        fluid_type=fluid_type,
    )

    return (
        pd.DataFrame(
            {
                "year": np.arange(start_year, end_year + 1).tolist(),
                "ltp": ltp_array.tolist(),
            }
        )
        .set_index("year")
        .to_dict()
    )


def get_rdp(
    start_year: int,
    end_year: int,
    year_rampup: int,
    drate: float,
    q_plateau_ratio: float,
    q_min_ratio: float,
    volume: float,
) -> np.ndarray:
    """
    The function to get the RDP array.

    Parameters
    ----------
    start_year
    end_year
    year_rampup
    drate
    q_plateau_ratio
    q_min_ratio
    volume

    Returns
    -------
    out: np.ndarray
        The array of rdp
    """

    # Initiating Model
    model = RPDModel(
        year_rampup=year_rampup,
        drate=drate,
        q_plateau_ratio=q_plateau_ratio,
        q_min_ratio=q_min_ratio,
    )

    rate_rdp = model.predict(volume)

    # Initiating the array of years
    year_arr = np.arange(start_year, end_year + 1)
    gap = len(year_arr) - len(rate_rdp)

    # Checking condition for the given years gap
    if gap < 0:
        raise RDPModelException(
            f"Forecast years from {start_year} to {end_year} is too short. "
            f"Please set the end_year at least until {end_year + (-1 * gap)}"
        )

    rate_gap = np.zeros(gap)
    rate_adjusted = np.concatenate([rate_gap, rate_rdp])

    return rate_adjusted


def get_rpd_dict(data: dict):
    """
    The function to get the list of RPD from the given reserves volume.

    Parameters
    ----------
    data: dict
        The dictionary of the data input.

    Returns
    -------
    """

    # Initiating the data input
    year_rampup = data["year_rampup"]
    drate = data["drate"]
    q_plateau_ratio = data["q_plateau_ratio"]
    q_min_ratio = data["q_min_ratio"]
    volume = data["volume"]
    start_year = data["start_year"]
    end_year = data["end_year"]

    # Get the array of rdp
    rdp_array = get_rdp(
        start_year=start_year,
        end_year=end_year,
        year_rampup=year_rampup,
        drate=drate,
        q_plateau_ratio=q_plateau_ratio,
        q_min_ratio=q_min_ratio,
        volume=volume,
    )

    return (
        pd.DataFrame(
            {
                "year": np.arange(start_year, end_year + 1).tolist(),
                "rpd": rdp_array.tolist(),
            }
        )
        .set_index("year")
        .to_dict()
    )


def get_transition_split(data: dict):
    """
    The function to get the contractor split information from Transition Contract Scheme.

    Parameters
    ----------
    data: dict
        The dictionary of the data input.

    Returns
    -------
    dict
        The dictionary containing the information of the contractor split.

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
        raise ContractException("Contract type is not recognized")

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
        raise ContractException("Contract type is not recognized")

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

    # Making the base for the loops and container of the result
    result = {}

    # Retrieving the split information from first contract
    for index, contract in enumerate([contract_1, contract_2]):
        if isinstance(contract, GrossSplit):
            # Retrieving the split information
            contractor_split = (
                pd.DataFrame(
                    {
                        "project_years": contract.project_years.tolist(),
                        "oil_base_split": contract._oil_base_split.tolist(),
                        "gas_base_split": contract._gas_base_split.tolist(),
                        "var_split_array": contract._var_split_array.tolist(),
                        "oil_prog_split": contract._oil_prog_split.tolist(),
                        "gas_prog_split": contract._gas_prog_split.tolist(),
                        "oil_ctr_split": contract._oil_ctr_split_prior_bracket.tolist(),
                        "gas_ctr_split": contract._gas_ctr_split_prior_bracket.tolist(),
                    }
                )
                .set_index("project_years")
                .to_dict()
            )

            years_of_maximum_split = {
                "oil": contract._oil_year_maximum_ctr_split.tolist(),
                "gas": contract._gas_year_maximum_ctr_split.tolist(),
            }

            result["contract_" + str(index + 1)] = {
                "contractor_split": contractor_split,
                "years_of_maximum_split": years_of_maximum_split,
            }
        else:
            result["contract_" + str(index + 1)] = {}
            pass

    return result


def get_economic_limit(data: dict):
    """
    The function to get the information of economic limit years from
    a cashflow using selected method.

    Parameters
    ----------
    data: dict

    Returns
    -------
    int
        The index
    """

    years = np.array(data["years"], dtype=int)
    cash_flow = np.array(data["cash_flow"], dtype=float)
    method = convert_to_method_limit(target=data["method"])
    index_limit = econ_limit(cashflow=cash_flow, method=method)

    return years[index_limit]


def get_asr_expenditures(data: dict) -> dict:
    """
    Compute and return the post-tax ASR (Abandonment, Site Restoration)
    expenditures from the given input data.

    This function constructs a pseudo Base Project environment using
    the provided ASR and setup data, runs the project simulation,
    and extracts the ASR expenditures for oil and gas over project years.

    Parameters
    ----------
    data : dict
        Input data dictionary containing at least the following keys:
        - ``asr`` : ASR-related cost data.
        - ``setup`` : Dictionary with setup information
          (e.g., start/end date, onstream dates).
        - ``lifting`` : Lifting profile or data for oil and gas.

    Returns
    -------
    dict
        Dictionary of ASR expenditures with project years as keys and
        sub-dictionaries containing:
        - ``oil_asr_expenditures`` : Post-tax ASR expenditures for oil.
        - ``gas_asr_expenditures`` : Post-tax ASR expenditures for gas.

    Notes
    -----
    - The function internally mimics a Base Project execution by
      constructing a minimal pseudo-input structure compatible with
      :func:`get_baseproject`.
    - ASR expenditures are extracted from the executed contract object
      and converted into a dictionary using a Pandas DataFrame.
    - No profitability or inflation adjustments are applied.
    """

    # Initiating the asr data
    asr_pseudo = {"asr": data["asr"]}

    # Mimics the baseproject data
    _setup = data["setup"]

    data_pseudo = {
        "setup": {
            "start_date": _setup["start_date"],
            "end_date": _setup["end_date"],
            "oil_onstream_date": _setup["oil_onstream_date"],
            "gas_onstream_date": _setup["gas_onstream_date"],
            "approval_year": None,
            "is_pod_1": False,
        },
        "summary_arguments": {
            "discount_rate_start_year": None,
            "inflation_rate": 0.0,
            "discount_rate": 0.1,
            "npv_mode": "Full Cycle Nominal Terms",
            "discounting_mode": "Mid Year",
            "profitability_discounted": False,
        },
        "contract_arguments": {
            "sulfur_revenue": "Addition to Gas Revenue",
            "electricity_revenue": "Addition to Oil Revenue",
            "co2_revenue": "Addition to Gas Revenue",
            # "sunk_cost_reference_year": None,
            # "year_inflation": 0,
            "inflation_rate": 0,
            "vat_rate": 0,
            "inflation_rate_applied_to": "CAPEX",
        },
        "lifting": data["lifting"],
        "capital": None,
        "intangible": None,
        "opex": None,
        "asr": asr_pseudo["asr"],
        "lbt": None,
        "cost_of_sales": None,
    }

    # Parsing the data into base project dataclass
    # contract = get_baseproject(data=data_pseudo, summary_result=False)[1]
    _, contract, contract_arguments, _ = get_baseproject(
        data=data_pseudo, summary_result=False
    )

    # Convert BaseProject instance into dictionary using method `vars()`
    contract_as_dict = vars(contract)

    # Returning the ASR Expenditures
    df = pd.DataFrame(
        {
            "project_years": contract.project_years,
            "oil_asr_expenditures": contract_as_dict["_oil_asr_expenditures_post_tax"],
            "gas_asr_expenditures": contract_as_dict["_gas_asr_expenditures_post_tax"],
            # "oil_asr_expenditures": contract._oil_asr_expenditures_post_tax,
            # "gas_asr_expenditures": contract._gas_asr_expenditures_post_tax,
        }
    )

    df = df.set_index("project_years").to_dict()

    return df


def get_lbt_expenditures(data: dict) -> dict:
    """
    Compute and return the post-tax ASR (Abandonment, Site Restoration)
    expenditures from the given input data.

    This function constructs a pseudo Base Project environment using
    the provided ASR and setup data, runs the project simulation,
    and extracts the ASR expenditures for oil and gas over project years.

    Parameters
    ----------
    data : dict
        Input data dictionary containing at least the following keys:
        - ``asr`` : ASR-related cost data.
        - ``setup`` : Dictionary with setup information
          (e.g., start/end date, onstream dates).
        - ``lifting`` : Lifting profile or data for oil and gas.

    Returns
    -------
    dict
        Dictionary of ASR expenditures with project years as keys and
        sub-dictionaries containing:
        - ``oil_asr_expenditures`` : Post-tax ASR expenditures for oil.
        - ``gas_asr_expenditures`` : Post-tax ASR expenditures for gas.

    Notes
    -----
    - The function internally mimics a Base Project execution by
      constructing a minimal pseudo-input structure compatible with
      :func:`get_baseproject`.
    - ASR expenditures are extracted from the executed contract object
      and converted into a dictionary using a Pandas DataFrame.
    - No profitability or inflation adjustments are applied.
    """

    # Initiating the LBT data
    lbt_pseudo = {"lbt": data["lbt"]}

    # Mimics the baseproject data
    _setup = data["setup"]

    # Mimics the baseproject data
    data_pseudo = {
        "setup": {
            "start_date": _setup["start_date"],
            "end_date": _setup["end_date"],
            "oil_onstream_date": _setup["oil_onstream_date"],
            "gas_onstream_date": _setup["gas_onstream_date"],
            "approval_year": None,
            "is_pod_1": False,
        },
        "summary_arguments": {
            "discount_rate_start_year": None,
            "inflation_rate": 0.0,
            "discount_rate": 0.1,
            "npv_mode": "Full Cycle Nominal Terms",
            "discounting_mode": "Mid Year",
            "profitability_discounted": False,
        },
        "contract_arguments": {
            "sulfur_revenue": "Addition to Gas Revenue",
            "electricity_revenue": "Addition to Oil Revenue",
            "co2_revenue": "Addition to Gas Revenue",
            # "sunk_cost_reference_year": None,
            # "year_inflation": 0,
            "inflation_rate": 0,
            "vat_rate": 0,
            "inflation_rate_applied_to": "CAPEX",
        },
        "lifting": data["lifting"],
        "capital": None,
        "intangible": None,
        "opex": None,
        "asr": None,
        "lbt": lbt_pseudo["lbt"],
        "cost_of_sales": None,
    }

    # Parsing the data into base project dataclass
    # contract = get_baseproject(data=data_pseudo, summary_result=False)[1]
    _, contract, contract_arguments, _ = get_baseproject(
        data=data_pseudo, summary_result=False
    )

    # Convert BaseProject instance into dictionary using method `vars()`
    contract_as_dict = vars(contract)

    # Returning the LBT Expenditures
    df = pd.DataFrame(
        {
            "project_years": contract.project_years,
            "oil_lbt_expenditures": contract_as_dict["_oil_lbt_expenditures_post_tax"],
            "gas_lbt_expenditures": contract_as_dict["_gas_lbt_expenditures_post_tax"],
            # "oil_lbt_expenditures": contract._oil_lbt_expenditures_post_tax,
            # "gas_lbt_expenditures": contract._gas_lbt_expenditures_post_tax,
        }
    )

    df = df.set_index("project_years").to_dict()

    return df


def get_contract_optimization(data: dict, contract_type: str = "Cost Recovery") -> dict:
    """
    The function to run contract optimization. Resulting optimization result
    in dictionary format.

    Parameters
    ----------
    data: dict
        The dictionary of the data input.
    contract_type: str
        The option for the contract type. The available option are:
        ['Cost Recovery', 'Gross Split']

    Returns
    -------
    result_parameters: dict
        The result of the optimization in dictionary format
    """

    if "optimization_arguments" not in data:
        raise ContractException(
            "The payload does not have the optimization_arguments key"
        )

    if data["optimization_arguments"] is None:
        raise ContractException(
            "The payload optimization_arguments does not have any values"
        )

    # Converting the parameters in dict_optimization to the corresponding enum
    optimization_parameters = [
        convert_str_to_optimization_parameters(str_object=i)
        for i in data["optimization_arguments"]["dict_optimization"]["parameter"]
    ]

    # Generating the dictionary of the optimization arguments
    dict_optimization = {
        "parameter": optimization_parameters,
        "min": convert_list_to_array_float(
            data_list=data["optimization_arguments"]["dict_optimization"]["min"]
        ),
        "max": convert_list_to_array_float(
            data_list=data["optimization_arguments"]["dict_optimization"]["max"]
        ),
    }

    # Filling the optimization arguments with target_optimization and target_parameter,
    target_optimization_value = data["optimization_arguments"]["target_optimization"]
    target_parameter = convert_str_to_optimization_targetparameter(
        str_object=data["optimization_arguments"]["target_parameter"]
    )

    # Retrieving the contract, contract_arguments_dict,
    # summary_arguments_dict based on the contract type
    if contract_type == "Cost Recovery":
        contract = get_costrecovery(data=data)[1]
        contract_arguments = get_costrecovery(data=data)[2]
        summary_argument = get_costrecovery(data=data)[3]

    elif contract_type == "Gross Split":
        contract = get_grosssplit(data=data)[1]
        contract_arguments = get_grosssplit(data=data)[2]
        summary_argument = get_grosssplit(data=data)[3]

    elif contract_type == "Transition":
        contract = get_transition(data=data)[1]
        contract_arguments = get_transition(data=data)[2]
        summary_argument = get_transition(data=data)[3]

    else:
        contract = NotImplemented
        contract_arguments = NotImplemented
        summary_argument = NotImplemented

    if contract_type == "Transition":
        # Retrieve the original useful life of the capital cost
        useful_life_original = (
            contract.contract2.capital_cost_total.useful_life.tolist()
        )

        list_str, list_params_value, result_optimization, list_executed_contract = (
            optimize_psc_trans(
                dict_optimization=dict_optimization,
                contract=contract,
                contract_arguments=contract_arguments,
                target_optimization_value=target_optimization_value,
                summary_argument=summary_argument,
                target_parameter=target_parameter,
            )
        )

    else:
        # Retrieve the original useful life of the capital cost
        useful_life_original = contract.capital_cost_total.useful_life.tolist()
        list_str, list_params_value, result_optimization, list_executed_contract = (
            optimize_psc(
                dict_optimization=dict_optimization,
                contract=contract,
                contract_arguments=contract_arguments,
                target_optimization_value=target_optimization_value,
                summary_argument=summary_argument,
                target_parameter=target_parameter,
            )
        )

    # Treatment to add the useful life of optimization into the result
    def get_enum_index(enum_list: list, element: any) -> int | None:
        """
        Function to get the index of the OptimizationParameter.DEPRECIATION_ACCELERATION.

        Parameters
        ----------
        enum_list: list
            The source of list.
        element: any
            The corresponding element
        Returns
        -------
        out : int | None
        """
        try:
            return enum_list.index(element)
        except ValueError:
            return None

    # Get the index of the depreciation optimization parameter
    index_depreciation = get_enum_index(
        enum_list=optimization_parameters,
        element=OptimizationParameter.DEPRECIATION_ACCELERATION,
    )

    # Adding condition of the contract type for retrieving the optimized contract
    if contract_type == "Transition":
        contract_optimized = list_executed_contract[-1].contract2
    else:
        contract_optimized = list_executed_contract[-1]

    # Adding the information of optimized useful life into the list_params_value
    if index_depreciation is not None:
        optimized_capital_cost = {
            "year": contract_optimized.capital_cost_total.expense_year.tolist(),
            "cost_allocation": contract_optimized.capital_cost_total.cost_allocation,
            "cost": contract_optimized.capital_cost_total.cost.tolist(),
            "pis_year": contract_optimized.capital_cost_total.pis_year.tolist(),
            "useful_life_original": useful_life_original,
            "useful_life_optimized": contract_optimized.capital_cost_total.useful_life.tolist(),
            "description": contract_optimized.capital_cost_total.description,
        }

        # Adding optimized_capital_cost into the result of the optimization
        list_params_value[index_depreciation] = {
            "depreciation acceleration": list_params_value[index_depreciation],
            "optimized_useful_life": optimized_capital_cost,
        }

    # Forming the optimization result into a dictionary object
    optimization_result = {
        "list_str": list_str,
        "list_params_value": list_params_value,
    }

    # Converting the result into dictionary format
    result_parameters = (
        pd.DataFrame(optimization_result).set_index("list_str").to_dict()
    )
    result_parameters["optimization_result"] = result_optimization

    # Adding the execution info
    result_parameters = add_execution_info(data=result_parameters)

    return result_parameters


def get_sensitivity(data: dict, contract_type: str):
    """
    Run sensitivity analysis for a given PSC contract type.

    This function retrieves the appropriate contract instance, contract arguments,
    and summary arguments based on the specified `contract_type`, and then performs
    a sensitivity analysis using the `sensitivity_psc` function.

    The analysis adjusts selected economic parameters (CAPEX, OPEX, OILPRICE, GASPRICE,
    ILLIFTING, GASLIFTING) according to the sensitivity configuration provided in the
    input dictionary.

    Parameters
    ----------
    data : dict
        A dictionary containing the full project input payload.
    contract_type : str
        The type of PSC contract to be evaluated. Accepted values include:
        - `"Base Project"`
        - `"Cost Recovery"`
        - `"Gross Split"`
        - `"Transition"`

    Returns
    -------
    dict
        A nested dictionary containing sensitivity results for several key indicators.

    Notes
    -----
    - Internally, this function calls one of the following constructors based on
      `contract_type`:
        * `get_baseproject(data)`
        * `get_costrecovery(data)`
        * `get_grosssplit(data)`
        * `get_transition(data)`
    - Each constructor is assumed to return a tuple whose elements include:
        `(status, contract_object, contract_arguments, summary_arguments)`.
    - The sensitivity analysis is performed by calling `sensitivity_psc(...)` with
      `dataframe_output=False`, so the result is returned as a dictionary rather than
      a collection of DataFrames.
    """

    if "sensitivity_arguments" not in data:
        raise ContractException(
            "The payload does not have the sensitivity_arguments key"
        )

    if data["sensitivity_arguments"] is None:
        raise ContractException(
            "The payload sensitivity_arguments does not have any values"
        )

    # Retrieving the contract, contract_arguments_dict,
    # summary_arguments_dict based on the contract type
    # CostRecovery
    if contract_type == "Cost Recovery":
        contract = get_costrecovery(data=data)[1]
        contract_arguments = get_costrecovery(data=data)[2]
        summary_argument = get_costrecovery(data=data)[3]

    # GrossSplit
    elif contract_type == "Gross Split":
        contract = get_grosssplit(data=data)[1]
        contract_arguments = get_grosssplit(data=data)[2]
        summary_argument = get_grosssplit(data=data)[3]

    # Transition
    elif contract_type == "Transition":
        contract = get_transition(data=data)[1]
        contract_arguments = get_transition(data=data)[2]
        summary_argument = get_transition(data=data)[3]

    # BaseProject
    else:
        contract = get_baseproject(data=data)[1]
        contract_arguments = get_baseproject(data=data)[2]
        summary_argument = get_baseproject(data=data)[3]

    # Constructing the sensitivity arguments
    sensitivity_result = sensitivity_psc(
        contract=contract,
        contract_arguments=contract_arguments,
        summary_arguments=summary_argument,
        min_deviation=data["sensitivity_arguments"]["min_deviation"],
        max_deviation=data["sensitivity_arguments"]["max_deviation"],
        base_value=data["sensitivity_arguments"]["base_value"],
        step=data["sensitivity_arguments"]["step"],
        dataframe_output=False,
    )

    return sensitivity_result


def get_uncertainty(data: dict, contract_type: str):
    """
    Run Monte Carlo uncertainty analysis for a specified PSC contract type.

    This function extracts contract setup, statistical parameters, and
    uncertainty configurations from the input data, then executes
    probabilistic simulations through `uncertainty_psc()`.

    Returns
    -------
    dict
        Dictionary containing probabilistic results (P10, P50, P90) and
        simulation details.

    Notes
    -----
    - Supported contract types: "Base Project", "Cost Recovery",
      "Gross Split", and "Transition".
    - Each uncertainty parameter (e.g., oil price, gas price, CAPEX, OPEX,
      lifting) is simulated using its mean, range, standard deviation,
      and selected probability distribution.
    - Raises `ContractException` if `uncertainty_arguments` are missing or invalid.
    """

    # Filter unwanted inputs
    if "uncertainty_arguments" not in data:
        raise ContractException(
            "The payload does not have the uncertainty_arguments key"
        )

    if data["uncertainty_arguments"] is None:
        raise ContractException(
            "The payload uncertainty_arguments does not have any values"
        )

    # Retrieving the contract, contract_arguments_dict,
    # summary_arguments_dict based on the contract type
    if contract_type == "Cost Recovery":
        contract = get_costrecovery(data=data)[1]
        contract_arguments = get_costrecovery(data=data)[2]
        summary_argument = get_costrecovery(data=data)[3]

    elif contract_type == "Gross Split":
        contract = get_grosssplit(data=data)[1]
        contract_arguments = get_grosssplit(data=data)[2]
        summary_argument = get_grosssplit(data=data)[3]

    elif contract_type == "Transition":
        contract = get_transition(data=data)[1]
        contract_arguments = get_transition(data=data)[2]
        summary_argument = get_transition(data=data)[3]

    else:
        contract = get_baseproject(data=data)[1]
        contract_arguments = get_baseproject(data=data)[2]
        summary_argument = get_baseproject(data=data)[3]

    # Helper function
    def _convert_distribution_enum_to_str(target: str):
        return convert_to_uncertainty_distribution(
            target=data["uncertainty_arguments"][target]
        )

    # Abbreviations
    ua = data["uncertainty_arguments"]
    to_str = _convert_distribution_enum_to_str

    # Constructing the sensitivity arguments
    uncertainty_kwargs = {
        # Base parameters
        "contract": contract,
        "contract_arguments": contract_arguments,
        "summary_arguments": summary_argument,
        "run_number": ua["run_number"],

        # Statistics parameters for OIL PRICE
        "min_oil_price": ua["min_oil_price"],
        "mean_oil_price": ua["mean_oil_price"],
        "max_oil_price": ua["max_oil_price"],

        # Statistics parameters for GAS PRICE
        "min_gas_price": ua["min_gas_price"],
        "mean_gas_price": ua["mean_gas_price"],
        "max_gas_price": ua["max_gas_price"],

        # Statistics parameters for OPEX
        "min_opex": ua["min_opex"],
        "mean_opex": ua["mean_opex"],
        "max_opex": ua["max_opex"],

        # Statistics parameters for CAPEX
        "min_capex": ua["min_capex"],
        "mean_capex": ua["mean_capex"],
        "max_capex": ua["max_capex"],

        # Statistics parameters for Lifting
        "min_lifting": ua["min_lifting"],
        "mean_lifting": ua["mean_lifting"],
        "max_lifting": ua["max_lifting"],

        # Standard deviations
        "oil_price_stddev": ua["oil_price_stddev"],
        "gas_price_stddev": ua["gas_price_stddev"],
        "opex_stddev": ua["opex_stddev"],
        "capex_stddev": ua["capex_stddev"],
        "lifting_stddev": ua["lifting_stddev"],

        # Distribution
        "oil_price_distribution": to_str("oil_price_distribution"),
        "gas_price_distribution": to_str("gas_price_distribution"),
        "opex_distribution": to_str("opex_distribution"),
        "capex_distribution": to_str("capex_distribution"),
        "lifting_distribution": to_str("lifting_distribution"),
    }

    return uncertainty_psc(**uncertainty_kwargs)


# def get_summary_object(
#     data: dict, contract: CostRecovery | GrossSplit | Transition
# ) -> (dict, dict):
#     """
#     The function to get the summary dictionary object from the data and contract input.
#
#     Parameters
#     ----------
#     data: dict
#         The dictionary of the data input
#     contract: CostRecovery | GrossSplit | Transition
#         The contract object.
#
#     Returns
#     -------
#     summary: dict
#         The summary of the contract ini dictionary format.
#
#     summary_arguments_dict: dict
#         The summary arguments used in retrieving the summary of the contract.
#
#     """
#     if contract is Transition:
#         summary_arguments_dict = get_summary_dict(data=data)
#         summary_arguments_dict["contract"] = contract
#         summary = get_summary(**summary_arguments_dict)
#
#     else:
#         summary_arguments_dict = get_summary_dict(data=data)
#         summary_arguments_dict["contract"] = contract
#         summary = get_summary(**summary_arguments_dict)
#
#     return summary, summary_arguments_dict
