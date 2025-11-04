"""
A collection of operations to run optimization module.
"""

# import numpy as np
from datetime import date
from pyscnomics.example import ExampleCase
from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.econ.selection import (
    TaxSplitTypeCR,
    OtherRevenue,
    InflationAppliedTo,
    TaxRegime,
    FTPTaxRegime,
    DeprMethod,
    SunkCostMethod,
    InitialYearAmortizationIncurred,
    GrossSplitRegime,
    VariableSplit082017,
    VariableSplit132024,
    NPVSelection,
    DiscountingMode,
    PoolData,
)
from pyscnomics.dataset.poolDataAsClass import PrepareLiftingCostsAsClass


# Create an instance of synthetic data
lft_cst = PrepareLiftingCostsAsClass()

target = PoolData.BENUANG

def get_lifting_costs_map(target: PoolData):
    return {
        "lifting": {
            "oil": lft_cst.lifting["oil"][target],
            "gas": lft_cst.lifting["gas"][target],
        },
        "capital": {
            "oil": lft_cst.capital_cost["oil"][target],
            "gas": lft_cst.capital_cost["gas"][target],
        },
        "intangible": {
            "oil": lft_cst.intangible_cost["oil"][target],
            "gas": lft_cst.intangible_cost["gas"][target],
        },
        "opex": {
            "oil": lft_cst.opex["oil"][target],
            "gas": lft_cst.opex["gas"][target],
        },
        "asr": {
            "oil": lft_cst.asr_cost["oil"][target],
            "gas": lft_cst.asr_cost["gas"][target],
        },
        "lbt": {
            "oil": lft_cst.lbt_cost["oil"][target],
            "gas": lft_cst.lbt_cost["gas"][target],
        },
        "cos": {
            "oil": lft_cst.cost_of_sales["oil"][target],
            "gas": lft_cst.cost_of_sales["gas"][target],
        },
    }


# Synthetic data: class format
def get_lifting_costs_class(target: PoolData) -> dict:
    """
    Collect and group cost components for multiple project cases.

    Returns
    -------
    dict
        Dictionary containing tuples of cost elements for each field, including
        lifting, capital, intangible, opex, ASR, LBT, and cost of sales.
    """

    data = get_lifting_costs_map(target=target)

    lifting = tuple([data["lifting"]["oil"], data["lifting"]["gas"]])
    capital_cost = tuple([data["capital"]["oil"], data["capital"]["gas"]])
    intangible_cost = tuple([data["intangible"]["oil"], data["intangible"]["gas"]])
    opex = tuple([data["opex"]["oil"], data["opex"]["gas"]])
    asr_cost = tuple([data["asr"]["oil"], data["asr"]["gas"]])
    lbt_cost = tuple([data["lbt"]["oil"]  ])


    lifting = tuple([case.lifting_mangga, case.lifting_apel])
    capital_cost = tuple([case.capital_mangga, case.capital_apel])
    intangible_cost = tuple([case.intangible_mangga, case.intangible_apel])
    opex = tuple([case.opex_mangga, case.opex_apel])
    asr_cost = tuple([case.asr_mangga, case.asr_apel])
    lbt_cost = tuple([case.lbt_mangga, case.lbt_apel])
    cost_of_sales = tuple([case.cos_mangga, case.cos_apel])

    return {
        "lifting": lifting,
        "capital_cost": capital_cost,
        "intangible_cost": intangible_cost,
        "opex": opex,
        "asr_cost": asr_cost,
        "lbt_cost": lbt_cost,
        "cost_of_sales": cost_of_sales,
    }


def get_kwargs_class(contract_type: str) -> dict:
    """
    Return default argument mappings for the given PSC contract type.

    Parameters
    ----------
    contract_type : str
        One of {"base_project", "cost_recovery", "gross_split"}.

    Returns
    -------
    dict
        Dictionary of keyword arguments for initializing the contract class.

    Raises
    ------
    ValueError
        If the contract type is unrecognized.
    """

    # Base Project
    kwargs_base_project = {
        # Base parameters
        "start_date": date(year=2023, month=1, day=1),
        "end_date": date(year=2032, month=12, day=31),
        "oil_onstream_date": date(year=2030, month=1, day=1),
        "gas_onstream_date": date(year=2029, month=1, day=1),
        "approval_year": 2026,
        "is_pod_1": False,
    }

    # Cost recovery
    kwargs_cost_recovery = {
        # Base parameters
        **kwargs_base_project,

        # FTP
        "oil_ftp_is_available": True,
        "oil_ftp_is_shared": True,
        "oil_ftp_portion": 0.2,
        "gas_ftp_is_available": True,
        "gas_ftp_is_shared": True,
        "gas_ftp_portion": 0.2,

        # Tax split
        "tax_split_type": TaxSplitTypeCR.CONVENTIONAL,
        "condition_dict": dict,
        "indicator_rc_icp_sliding": None,
        "oil_ctr_pretax_share": 0.25,
        "gas_ctr_pretax_share": 0.5,

        # Investment credit
        "oil_ic_rate": 0.0,
        "gas_ic_rate": 0.0,
        "ic_is_available": False,
        "oil_cr_cap_rate": 1.0,
        "gas_cr_cap_rate": 1.0,

        # DMO
        "oil_dmo_volume_portion": 0.25,
        "oil_dmo_fee_portion": 0.25,
        "oil_dmo_holiday_duration": 60,
        "gas_dmo_volume_portion": 1.0,
        "gas_dmo_fee_portion": 1.0,
        "gas_dmo_holiday_duration": 60,

        # Carry forward depreciation
        "oil_carry_forward_depreciation": 0.0,
        "gas_carry_forward_depreciation": 0.0,
    }

    # Gross Split
    VS_08 = VariableSplit082017
    VS_13 = VariableSplit132024

    kwargs_gross_split = {
        # Base Parameters
        **kwargs_base_project,

        # Field and reservoir properties
        "field_status": VS_08.FieldStatus.NO_POD,
        "field_loc": VS_08.FieldLocation.ONSHORE,
        "res_depth": VS_08.ReservoirDepth.LESSEQUAL_2500,
        "infra_avail": VS_08.InfrastructureAvailability.WELL_DEVELOPED,
        "res_type": VS_08.ReservoirType.CONVENTIONAL,
        "api_oil": VS_08.APIOil.LESSTHAN_25,
        "domestic_use": VS_08.DomesticUse.EQUAL_50_UNTIL_LESSTHAN_70,
        "prod_stage": VS_08.ProductionStage.SECONDARY,
        "co2_content": VS_08.CO2Content.LESSTHAN_5,
        "h2s_content": VS_08.H2SContent.LESSTHAN_100,
        "field_reserves_2024": VS_13.FieldReservesAmount.MEDIUM,
        "infra_avail_2024": VS_13.InfrastructureAvailability.PARTIALLY_AVAILABLE,
        "field_loc_2024": VS_13.FieldLocation.ONSHORE,

        # Ministry discretion
        "split_ministry_disc": 0.08,

        # DMO
        "oil_dmo_volume_portion": 0.25,
        "oil_dmo_fee_portion": 1.0,
        "gas_dmo_volume_portion": 1.0,
        "gas_dmo_fee_portion": 1.0,
        "oil_dmo_holiday_duration": 60,
        "gas_dmo_holiday_duration": 60,

        # Carry forward depreciation
        "oil_carry_forward_depreciation": 0.0,
        "gas_carry_forward_depreciation": 0.0,
    }

    # Pooled kwargs
    kwargs_contract = {
        "base_project": kwargs_base_project,
        "cost_recovery": kwargs_cost_recovery,
        "gross_split": kwargs_gross_split,
    }

    try:
        return kwargs_contract[contract_type]

    except KeyError:
        raise ValueError(f"Unrecognized contract type: {contract_type!r}")


def get_contract_arguments_class(contract_type: str) -> dict:
    """
    Retrieve contract-specific parameter settings for PSC evaluation.

    Parameters
    ----------
    contract_type : str
        One of {"base_project", "cost_recovery", "gross_split"}.

    Returns
    -------
    dict
        Parameter dictionary defining fiscal, revenue, tax, and cost
        assumptions for the specified contract type.

    Notes
    -----
    The function selects from predefined parameter sets aligned with
    Indonesian PSC schemes. Raises a ValueError if the contract type
    is unrecognized.
    """

    # Base Project
    params_base_project = {
        "sulfur_revenue": OtherRevenue.REDUCTION_TO_GAS_OPEX,
        "electricity_revenue": OtherRevenue.ADDITION_TO_OIL_REVENUE,
        "co2_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
        "tax_rate": 0.0,
        "year_inflation": None,
        "inflation_rate": 0.0,
        "inflation_rate_applied_to": None,
    }

    # Cost Recovery
    params_cost_recovery = {
        "sulfur_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
        "electricity_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
        "co2_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
        "vat_rate": 0.0,
        "year_inflation": None,
        "inflation_rate": 0.0,
        "inflation_rate_applied_to": None,
        "is_dmo_end_weighted": False,
        "tax_regime": TaxRegime.NAILED_DOWN,
        "effective_tax_rate": None,
        "ftp_tax_regime": FTPTaxRegime.PDJP_20_2017,
        "depr_method": DeprMethod.PSC_DB,
        "decline_factor": 2,
        "post_uu_22_year2001": True,
        "oil_cost_of_sales_applied": False,
        "gas_cost_of_sales_applied": False,
        "sum_undepreciated_cost": False,
        "sunk_cost_method": SunkCostMethod.DEPRECIATED_TANGIBLE,
    }

    # Gross Split
    params_gross_split = {
        "sulfur_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
        "electricity_revenue": OtherRevenue.ADDITION_TO_OIL_REVENUE,
        "co2_revenue": OtherRevenue.ADDITION_TO_GAS_REVENUE,
        "vat_rate": 0.0,
        "inflation_rate": 0.0,
        "inflation_rate_applied_to": InflationAppliedTo.CAPEX,
        "cum_production_split_offset": 0.0,
        "depr_method": DeprMethod.PSC_DB,
        "decline_factor": 2,
        "sum_undepreciated_cost": False,
        "is_dmo_end_weighted": False,
        "tax_regime": TaxRegime.NAILED_DOWN,
        "effective_tax_rate": 0.22,
        "amortization": False,
        "sunk_cost_method": SunkCostMethod.DEPRECIATED_TANGIBLE,
        "regime": GrossSplitRegime.PERMEN_ESDM_13_2024,
        "reservoir_type_permen_2024": VariableSplit132024.ReservoirType.MK,
        "initial_amortization_year": InitialYearAmortizationIncurred.ONSTREAM_YEAR,
    }

    # Pooled params
    params = {
        "base_project": params_base_project,
        "cost_recovery": params_cost_recovery,
        "gross_split": params_gross_split,
    }

    try:
        return params[contract_type]

    except KeyError:
        raise ValueError(f"Unrecognized contract type: {contract_type!r}")


def get_summary_arguments_class() -> dict:
    """
    Define default summary arguments for PSC economic evaluation.

    Returns
    -------
    dict
        Dictionary containing default settings for NPV calculation,
        discounting, and inflation parameters.

    Notes
    -----
    The returned arguments are typically used in project summary
    and economic indicator computations.
    """

    return {
        "discount_rate": 0.1,
        "npv_mode": NPVSelection.NPV_SKK_NOMINAL_TERMS,
        "discounting_mode": DiscountingMode.END_YEAR,
        "discount_rate_start_year": 2023,
        "inflation_rate": 0.0,
        "profitability_discounted": False,
    }


def synthetic_data_class(
    contract_type: str
) -> CostRecovery | GrossSplit | BaseProject:
    """
    Generate a synthetic contract instance for testing or benchmarking.

    Parameters
    ----------
    contract_type : str
        Type of contract to instantiate. Must be one of:
        ``"cost_recovery"``, ``"gross_split"``, or ``"base_project"``.

    Returns
    -------
    CostRecovery or GrossSplit or BaseProject
        A fully initialized contract instance with predefined arguments,
        parameters, and lifting cost settings.

    Notes
    -----
    The function combines default arguments and parameters to
    create representative PSC contract objects for simulation or validation.
    """

    mapping = {
        "cost_recovery": (
            CostRecovery,
            get_kwargs_class("cost_recovery"),
        ),
        "gross_split": (
            GrossSplit,
            get_kwargs_class("gross_split"),
        ),
        "base_project": (
            BaseProject,
            get_kwargs_class("base_project"),
        ),
    }

    ctr = mapping[contract_type]

    return ctr[0](**ctr[1], **get_lifting_costs_class())


# Synthetic data: dictionary format
def contract_arguments_dict(contract_type: str) -> dict:
    """
    Return default argument settings for a given PSC contract type.

    Parameters
    ----------
    contract_type : str
        One of {"base_project", "cost_recovery", "gross_split"}.

    Returns
    -------
    dict
        Default argument mapping for the specified contract.

    Raises
    ------
    ValueError
        If the contract type is unrecognized.
    """

    contract_arguments_mapping = {
        "base_project": {
            "sulfur_revenue": "Addition to Gas Revenue",
            "electricity_revenue": "Addition to Oil Revenue",
            "co2_revenue": "Addition to Gas Revenue",
            "vat_rate": 0.0,
            "inflation_rate": 0.0,
            "inflation_rate_applied_to": "None",
        },
        "cost_recovery": {
            "sulfur_revenue": "Addition to Gas Revenue",
            "electricity_revenue": "Addition to Oil Revenue",
            "co2_revenue": "Addition to Gas Revenue",
            "vat_rate": 0.0,
            "inflation_rate": 0.0,
            "inflation_rate_applied_to": "None",
            "is_dmo_end_weighted": False,
            "tax_regime": "nailed down",
            "effective_tax_rate": 0.376,
            "ftp_tax_regime": "Direct Mode",
            "depr_method": "PSC Declining Balance",
            "decline_factor": 2,
            "post_uu_22_year2001": True,
            "oil_cost_of_sales_applied": False,
            "gas_cost_of_sales_applied": False,
            "sum_undepreciated_cost": False,
            "sunk_cost_method": "depreciated_tangible",
        },
        "gross_split": {
            "sulfur_revenue": "Addition to Gas Revenue",
            "electricity_revenue": "Addition to Oil Revenue",
            "co2_revenue": "Addition to Gas Revenue",
            "vat_rate": 0.0,
            "inflation_rate": 0.0,
            "inflation_rate_applied_to": "None",
            "cum_production_split_offset": 0.0,
            "depr_method": "PSC Declining Balance",
            "decline_factor": 2,
            "sum_undepreciated_cost": False,
            "is_dmo_end_weighted": False,
            "tax_regime": "nailed down",
            "effective_tax_rate": 0.22,
            "amortization": False,
            "sunk_cost_method": "depreciated_tangible",
            "regime": "PERMEN_ESDM_13_2024",
            "reservoir_type_permen_2024": "conventional",
            "initial_amortization_year": "onstream_year",
        },
    }

    try:
        return contract_arguments_mapping[contract_type]

    except KeyError:
        raise ValueError(f"Unrecognized contract type: {contract_type!r}")


def optimization_arguments_dict():
    """
    Returns a dictionary containing default optimization settings.

    Returns
    -------
    dict
        Dictionary with target parameter, target value, and optimization
        variable configuration (name, minimum, and maximum limits).
    """
    return {
        "target_parameter": "IRR",
        "target_value": 0.3,
        "dict_optimization": {
            "parameter": "VAT Discount",
            "min": 0.1,
            "max": 0.6,
        },
    }


def synthetic_data_dict(contract_type: str) -> dict:
    """
    Generate synthetic input data for PSC economic evaluation.

    Builds mock datasets for setup, contract arguments, lifting profiles,
    and cost components under the specified contract type.

    Parameters
    ----------
    contract_type : str
        One of {"base_project", "cost_recovery", "gross_split"}.

    Returns
    -------
    dict
        Synthetic input data structured for PSC economic evaluation.
    """

    # Synthetic setup
    setup = {
        "start_date": "01/01/2022",
        "end_date": "31/12/2035",
        "oil_onstream_date": "01/01/2024",
        "gas_onstream_date": None,
        "approval_year": "2024",
        "is_pod_1": True,
    }

    # Synthetic summary arguments
    summary_arguments = {
        "discount_rate_start_year": 2023,
        "inflation_rate": 0.0,
        "discount_rate": 0.1,
        "npv_mode": "SKK Full Cycle Nominal Terms",
        "discounting_mode": "End Year",
        "profitability_discounted": False,
    }

    # Synthetic grosssplit arguments
    grosssplit = {
        "field_status": "No POD",
        "field_loc": "Onshore",
        "res_depth": "<=2500",
        "infra_avail": "Well Developed",
        "res_type": "Conventional",
        "api_oil": "<25>",
        "domestic_use": "50<=x<70",
        "prod_stage": "Secondary",
        "co2_content": "<5",
        "h2s_content": "<100",
        "field_reserves_2024": "medium",
        "infra_avail_2024": "partially_available",
        "field_loc_2024": "Onshore",
        "split_ministry_disc": 0.08,
        "oil_dmo_volume_portion": 0.25,
        "oil_dmo_fee_portion": 1,
        "oil_dmo_holiday_duration": 60,
        "gas_dmo_volume_portion": 0.25,
        "gas_dmo_fee_portion": 0.25,
        "gas_dmo_holiday_duration": 60,
        "oil_carry_forward_depreciation": [50, 20, 100],
        "gas_carry_forward_depreciation": 0.0,
    }

    # Synthetic costrecovery arguments
    costrecovery = {
        "oil_ftp_is_available": True,
        "oil_ftp_is_shared": True,
        "oil_ftp_portion": 0.1,
        "gas_ftp_is_available": True,
        "gas_ftp_is_shared": True,
        "gas_ftp_portion": 0.1,
        "tax_split_type": "Conventional",
        "condition_dict": {},
        "indicator_rc_icp_sliding": [],
        "oil_ctr_pretax_share": 0.599359,
        "gas_ctr_pretax_share": 0.0,
        "oil_ic_rate": 0.0,
        "gas_ic_rate": 0.0,
        "ic_is_available": False,
        "oil_cr_cap_rate": 1,
        "gas_cr_cap_rate": 0,
        "oil_dmo_volume_portion": 0.25,
        "oil_dmo_fee_portion": 1,
        "oil_dmo_holiday_duration": 0,
        "gas_dmo_volume_portion": 1,
        "gas_dmo_fee_portion": 0.25,
        "gas_dmo_holiday_duration": 0,
        "oil_carry_forward_depreciation": 0,
        "gas_carry_forward_depreciation": 0,
    }

    # Synthetic contract arguments
    contract_arguments = contract_arguments_dict(contract_type)

    # Synthetic lifting
    lifting = {
        "Oil Oil sources #1": {
            "start_year": 2022,
            "end_year": 2035,
            "prod_rate_baseline": [
                351000,
                351000,
                351000,
                351000,
                351000,
                351000,
                351000,
                351000,
                351000,
                351000,
                351000,
                351000,
            ],
            "prod_rate": None,
            "lifting_rate": [
                1003.01994465298,
                1752.43214595637,
                1660.63017314197,
                1646.06277634477,
                1303.30419844135,
                1206.56398735216,
                715.464597247899,
                575.351449953491,
                385.408188212388,
                177.896153424466,
                95.5608132946032,
                53.0868811432885
            ],
            "price": [
                75,
                75,
                75,
                75,
                75,
                75,
                75,
                75,
                75,
                75,
                75,
                75
            ],
            "prod_year": [
                2024,
                2025,
                2026,
                2027,
                2028,
                2029,
                2030,
                2031,
                2032,
                2033,
                2034,
                2035
            ],
            "fluid_type": "Oil"
        }
    }

    # Synthetic capital cost
    capital = {
        "GS Dummy-Optimized": {
            "start_year": 2022,
            "end_year": 2035,
            "expense_year": [
                2024,
                2025,
                2026,
                2027,
                2028
            ],
            "cost": [
                46770.5618086779,
                39367.6743830341,
                14092.6047468454,
                146.200209593186,
                205.10626669388
            ],
            "cost_allocation": [
                "Oil",
                "Oil",
                "Oil",
                "Oil",
                "Oil"
            ],
            "cost_type": [
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
            ],
            "description": [
                "Tang DWO",
                "Tang DWO",
                "Tang DWO",
                "Tang DWO",
                "Tang DWO"
            ],
            "tax_portion": [
                0,
                0,
                0,
                0,
                0,
            ],
            "tax_discount": [
                0,
                0,
                0,
                0,
                0,
            ],
            "pis_year": [
                2024,
                2025,
                2026,
                2027,
                2028
            ],
            "salvage_value": [
                0,
                0,
                0,
                0,
                0,
            ],
            "useful_life": [
                5,
                5,
                5,
                5,
                5,
            ],
            "depreciation_factor": [
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
            ],
            "is_ic_applied": [
                False,
                False,
                False,
                False,
                False,
            ],
        }
    }

    # Synthetic intangible cost
    intangible = {
        "GS Dummy-Optimized": {
            "start_year": 2022,
            "end_year": 2035,
            "expense_year": [
                2024,
                2025,
                2026,
                2027,
                2028
            ],
            "cost": [
                138789.463632672,
                64991.8968456878,
                1017.66466282672,
                1109.75384628443,
                1724.02172015002,
            ],
            "cost_allocation": [
                "Oil",
                "Oil",
                "Oil",
                "Oil",
                "Oil"
            ],
            "cost_type": [
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
            ],
            "description": [
                "Intang DWO",
                "Intang DWO",
                "Intang DWO",
                "Intang DWO",
                "Intang DWO"
            ],
            "tax_portion": [
                0,
                0,
                0,
                0,
                0
            ],
            "tax_discount": [
                0,
                0,
                0,
                0,
                0
            ],
        }
    }

    # Synthetic opex
    opex = {
        "GS Dummy-Optimized": {
            "start_year": 2022,
            "end_year": 2035,
            "expense_year": [
                2023,
                2024,
                2025,
                2026,
                2027,
                2028,
                2029,
                2030,
                2031,
                2032,
                2033,
                2034,
                2035
            ],
            "fixed_cost": [
                0,
                971.292234114079,
                2457.37777867217,
                3101.47410075184,
                3499.0467473904,
                3643.77149457785,
                4144.81094457935,
                4557.10420220148,
                3879.78558891758,
                3433.61380883011,
                3469.40440315405,
                1683.43593564488,
                491.387568792144,
            ],
            "cost_allocation": [
                "Oil",
                "Oil",
                "Oil",
                "Oil",
                "Oil",
                "Oil",
                "Oil",
                "Oil",
                "Oil",
                "Oil",
                "Oil",
                "Oil",
                "Oil",
            ],
            "cost_type": [
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
            ],
            "description": [
                "all OPEX",
                "all OPEX",
                "all OPEX",
                "all OPEX",
                "all OPEX",
                "all OPEX",
                "all OPEX",
                "all OPEX",
                "all OPEX",
                "all OPEX",
                "all OPEX",
                "all OPEX",
                "all OPEX"
            ],
            "tax_portion": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
            ],
            "tax_discount": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
            ],
            "prod_rate": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
            ],
            "cost_per_volume": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
            ],
        }
    }

    # Synthetic asr
    asr = {
        "GS Dummy-Optimized": {
            "start_year": 2022,
            "end_year": 2035,
            "expense_year": [
                2023,
                2024,
                2025,
                2026,
                2027,
                2028,
                2029,
                2030,
                2031,
                2032,
                2033,
                2034,
                2035
            ],
            "cost": [
                0,
                478.57133137928,
                838.67985905111,
                1136.99948801868,
                1136.99948801868,
                1136.99948801868,
                1136.99948801868,
                1136.99948801868,
                1136.99948801868,
                1136.99948801868,
                1136.99948801868,
                1136.99948801868,
                1136.99948801868
            ],
            "cost_allocation": [
                "Oil",
                "Oil",
                "Oil",
                "Oil",
                "Oil",
                "Oil",
                "Oil",
                "Oil",
                "Oil",
                "Oil",
                "Oil",
                "Oil",
                "Oil"
            ],
            "cost_type": [
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost"
            ],
            "description": [
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-"
            ],
            "tax_portion": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
            ],
            "tax_discount": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
            ],
            "final_year": [
                2023,
                2024,
                2025,
                2026,
                2027,
                2028,
                2029,
                2030,
                2031,
                2032,
                2033,
                2034,
                2035
            ],
            "future_rate": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
            ],
        }
    }

    # Synthetic lbt
    lbt = {
        "GS Dummy-Optimized": {
            "start_year": 2022,
            "end_year": 2035,
            "expense_year": [
                2023,
                2024,
                2025,
                2026,
                2027,
                2028,
                2029,
                2030,
                2031,
                2032,
                2033,
                2034,
                2035
            ],
            "cost": [
                0,
                0,
                1718.30476967245,
                3752.23099766397,
                3812.96275956648,
                2746.88736465813,
                1971.62024928616,
                1464.00914775356,
                1000.50006761995,
                814.840751988436,
                457.13115224711,
                356.605652566246,
                191.296236384866
            ],
            "cost_allocation": [
                "Oil",
                "Oil",
                "Oil",
                "Oil",
                "Oil",
                "Oil",
                "Oil",
                "Oil",
                "Oil",
                "Oil",
                "Oil",
                "Oil",
                "Oil"
            ],
            "cost_type": [
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost",
                "preonstream_cost"
            ],
            "description": [
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-"
            ],
            "tax_portion": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
            ],
            "tax_discount": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
            ],
            "final_year": None,
            "utilized_land_area": None,
            "utilized_building_area": None,
            "njop_land": None,
            "njop_building": None,
            "gross_revenue": None,
        }
    }

    # Synthetic cost of sales
    cos = {
        "GS Dummy-Optimized": {
            "start_year": 2022,
            "end_year": 2035,
            "expense_year": [2022],
            "cost": [0],
            "cost_allocation": ["Oil"],
            "cost_type": ["preonstream_cost"],
            "description": ["-"],
            "tax_portion": [0],
            "tax_discount": [0],
        }
    }

    # Synthetic optimization
    opt = {}

    # Mapping dict
    mapping_data = (
        ("setup", setup),
        ("summary_arguments", summary_arguments),
        ("grosssplit", grosssplit),
        ("costrecovery", costrecovery),
        ("contract_arguments", contract_arguments),
        ("lifting", lifting),
        ("capital", capital),
        ("intangible", intangible),
        ("opex", opex),
        ("asr", asr),
        ("lbt", lbt),
        ("cos", cos),
        ("opt", opt),
    )

    return {key: val for key, val in mapping_data}
