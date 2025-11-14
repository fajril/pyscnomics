"""
A collection of procedures to generate cashflow of a contract in the form of DataFrame.
"""

import pandas as pd
import numpy as np
from dataclasses import asdict
from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts.transition import Transition


def _assign_attr(
    attr: str,
    contract: CostRecovery | GrossSplit | BaseProject,
    is_lifting: bool = False
):
    """
    Retrieve an attribute from a contract object and optionally extract its lifting rate.

    Parameters
    ----------
    attr : str
        Name of the attribute to retrieve from the contract object.
    contract : CostRecovery or GrossSplit or BaseProject
        Contract instance containing the target attribute.
    is_lifting : bool, default=False
        If True, return the lifting rate array using
        `value.get_lifting_rate_ghv_arr()`; otherwise, return the attribute value itself.

    Returns
    -------
    Any
        The retrieved attribute or its lifting rate array if `is_lifting` is True.

    Notes
    -----
    This function simplifies attribute access from contract objects while
    handling lifting-related attributes consistently.
    """

    value = getattr(contract, attr)

    return value.get_lifting_rate_ghv_arr() if is_lifting else value


def get_table_costrecovery_oil(contract: CostRecovery) -> pd.DataFrame:

    cr = contract

    # Specify postonstream attributes for OIL
    oil_depreciable_postonstream = _assign_attr(
        "_oil_capital_expenditures_post_tax", cr
    )

    oil_non_depreciable_postonstream = np.array(
        [
            _assign_attr(at, cr) for at in [
                "_oil_intangible_expenditures_post_tax",
                "_oil_opex_expenditures_post_tax",
                "_oil_asr_expenditures_post_tax",
                "_oil_lbt_expenditures_post_tax",
                "_oil_cost_of_sales_expenditures_post_tax",
            ]
        ]
    ).sum(axis=0)

    oil_postonstream = oil_depreciable_postonstream + oil_non_depreciable_postonstream

    # A list of cost categories
    categories = [
        "capital",
        "intangible",
        "opex",
        "asr",
        "lbt",
        "cost_of_sales"
    ]

    # Assign attributes associated with expenditures pre tax
    pre_tax = {
        f"{cat}_expenditures_pre_tax": _assign_attr(
            f"_oil_{cat}_expenditures_pre_tax", cr
        )
        for cat in categories
    }

    # Assign attributes associated with indirect tax
    indirect_tax = {
        f"{cat}_indirect_tax": _assign_attr(f"_oil_{cat}_indirect_tax", cr)
        for cat in categories
    }

    # Assign attributes associated with postonstream costs (or expenditures post tax)
    post_tax = {
        f"{cat}_postonstream": _assign_attr(f"_oil_{cat}_expenditures_post_tax", cr)
        for cat in categories
    }

    # Assign attribute associated with depreciations
    depreciations = _assign_attr(f"_oil_depreciations", cr)

    # Specify cashflow table for OIL
    table_oil: dict = {
        # Basic attributes
        "years": cr.project_years,
        "lifting": _assign_attr("_oil_lifting", cr, True),
        "price": _assign_attr("_oil_wap_price", cr),
        "revenue": _assign_attr("_oil_revenue", cr),

        # Attributes associated with sulfur commodity
        "lifting_sulfur": _assign_attr("_sulfur_lifting", cr, True),
        "price_sulfur": _assign_attr("_sulfur_wap_price", cr),
        "revenue_sulfur": _assign_attr("_sulfur_revenue", cr),

        # Attributes associated with electricity commodity
        "lifting_electricity": _assign_attr("_electricity_lifting", cr, True),
        "price_electricity": _assign_attr("_electricity_wap_price", cr),
        "revenue_electricity": _assign_attr("_electricity_revenue", cr),

        # Attributes associated with CO2 commodity
        "lifting_co2": _assign_attr("_co2_lifting", cr, True),
        "price_co2": _assign_attr("_co2_wap_price", cr),
        "revenue_co2": _assign_attr("_co2_revenue", cr),

        # Attributes associated with sunk cost
        "sunk_cost_depreciable": _assign_attr("_oil_depreciable_sunk_cost", cr),
        "sunk_cost_non_depreciable": _assign_attr(
            "_oil_non_depreciable_sunk_cost", cr
        ),
        "sunk_cost": _assign_attr("_oil_sunk_cost", cr),

        # Attributes associated with preonstream cost
        "preonstream_depreciable": _assign_attr("_oil_depreciable_preonstream", cr),
        "preonstream_non_depreciable": _assign_attr(
            "_oil_non_depreciable_preonstream", cr
        ),
        "preonstream": _assign_attr("_oil_preonstream", cr),

        # Attributes associated with postonstream cost
        "postonstream_depreciable": oil_depreciable_postonstream,
        "postonstream_non_depreciable": oil_non_depreciable_postonstream,
        "postonstream": oil_postonstream,

        # Attributes associated with expenditures pre tax
        **pre_tax,

        # Attributes associated with indirect tax
        **indirect_tax,

        # Attributes associated with expenditures post tax
        **post_tax,

        # Attributes associated with expenses
        "expenses_capital": _assign_attr("_oil_capital", cr),
        "expenses_non_capital": _assign_attr("_oil_non_capital", cr),
        "expenses_total": _assign_attr("_oil_total_expenses", cr),

        # Attributes associated with depreciations
        "depreciations_sunk_cost": depreciations["sunk_cost"],
        "depreciations_preonstream": depreciations["preonstream"],
        "depreciations_postonstream": depreciations["postonstream"],

        # Attributes associated with FTP
        "ftp": _assign_attr("_oil_ftp", cr),
        "ftp_ctr": _assign_attr("_oil_ftp_ctr", cr),
        "ftp_gov": _assign_attr("_oil_ftp_gov", cr),

        # Attributes associated with core business logic
        "investment_credit": _assign_attr("_oil_ic_paid", cr),
        "unrecovered_cost": _assign_attr("_oil_unrecovered_before_transfer", cr),
        "recoverable_cost": _assign_attr("_oil_recoverable_cost", cr),
        "cost_recovery": _assign_attr("_oil_cost_recovery", cr),
        "ets_before_transfer": _assign_attr("_oil_ets_before_transfer", cr),
        "transfer_to_gas": _assign_attr("_transfer_to_gas", cr),
        "unrec_after_transfer": _assign_attr("_oil_unrecovered_after_transfer", cr),
        "cost_recovery_after_tf": _assign_attr("_oil_cost_recovery_after_tf", cr),
        "ets_after_transfer": _assign_attr("_oil_ets_after_transfer", cr),
        "contractor_share": _assign_attr("_oil_contractor_share", cr),
        "government_share": _assign_attr("_oil_government_share", cr),

        # Attributes associated with DMO
        "dmo_volume": _assign_attr("_oil_dmo_volume", cr),
        "dmo_fee": _assign_attr("_oil_dmo_fee", cr),
        "ddmo": _assign_attr("_oil_ddmo", cr),

        # Attributes associated with taxable income
        "taxable_income": _assign_attr("_oil_taxable_income", cr),
        "tax_payment": _assign_attr("_oil_tax_payment", cr),

        # Attributes associated with government and contractor shares
        "contractor_net_share": _assign_attr("_oil_ctr_net_share", cr),
        "government_take": _assign_attr("_oil_government_take", cr),

        # Attributes associated with cashflow
        "cashflow": _assign_attr("_oil_cashflow", cr),
        "cum_cashflow": np.cumsum(_assign_attr("_oil_cashflow", cr)),
    }

    # Convert OIL cashflow table into pandas DataFrame
    return pd.DataFrame(table_oil)


def get_table_costrecovery_gas(contract: CostRecovery) -> pd.DataFrame:

    cr = contract

    # Specify postonstream attributes for GAS
    gas_depreciable_postonstream = _assign_attr(
        "_gas_capital_expenditures_post_tax", cr
    )




def get_table_costrecovery_consolidated(contract: CostRecovery) -> pd.DataFrame:
    pass


def get_table_baseproject_oil(contract: BaseProject) -> pd.DataFrame:
    """
    Construct the base project cashflow table for oil in a BaseProject contract.

    The function extracts key oil-related attributes from the contract, including
    lifting, revenue, expenditures, indirect taxes, sunk costs, and pre-/post-onstream
    components. It consolidates them into a structured pandas DataFrame for
    downstream economic analysis.

    Parameters
    ----------
    contract : BaseProject
        The contract instance containing all oil-related financial and production
        attributes.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the oil cashflow table with columns for project years,
        lifting, prices, revenues, sunk costs, pre-/post-onstream costs, expenditures
        (pre-tax, indirect tax, post-tax), expenses, and net cashflow.

    Notes
    -----
    - The function internally uses `_assign_attr` to fetch attributes from the contract.
    - Post-onstream costs are split into depreciable and non-depreciable components.
    - Expenditure categories include: capital, intangible, opex, ASR, LBT, and
      cost of sales.
    - The resulting DataFrame serves as a base input for project-level
      economic evaluations.
    """

    bp = contract

    # Specify postonstream attributes for OIL
    oil_depreciable_postonstream = _assign_attr(
        "_oil_capital_expenditures_post_tax", bp
    )

    oil_non_depreciable_postonstream = np.array(
        [
            _assign_attr(at, bp) for at in [
                "_oil_intangible_expenditures_post_tax",
                "_oil_opex_expenditures_post_tax",
                "_oil_asr_expenditures_post_tax",
                "_oil_lbt_expenditures_post_tax",
                "_oil_cost_of_sales_expenditures_post_tax",
            ]
        ]
    ).sum(axis=0)

    oil_postonstream = oil_depreciable_postonstream + oil_non_depreciable_postonstream

    # A list of cost categories
    categories = [
        "capital",
        "intangible",
        "opex",
        "asr",
        "lbt",
        "cost_of_sales"
    ]

    # Assign attributes associated with expenditures pre tax
    pre_tax = {
        f"{cat}_expenditures_pre_tax": _assign_attr(
            f"_oil_{cat}_expenditures_pre_tax", bp
        )
        for cat in categories
    }

    # Assign attributes associated with indirect tax
    indirect_tax = {
        f"{cat}_indirect_tax": _assign_attr(f"_oil_{cat}_indirect_tax", bp)
        for cat in categories
    }

    # Assign attributes associated with postonstream costs (or expenditures post tax)
    post_tax = {
        f"{cat}_postonstream": _assign_attr(
            f"_oil_{cat}_expenditures_post_tax", bp
        )
        for cat in categories
    }

    # Specify cashflow table for OIL
    table_oil: dict = {
        # Basic attributes
        "years": bp.project_years,
        "lifting": _assign_attr("_oil_lifting", bp, True),
        "price":  _assign_attr("_oil_wap_price", bp),
        "revenue": _assign_attr("_oil_revenue", bp),

        # Attributes associated with sunk cost
        "sunk_cost_depreciable": _assign_attr("_oil_depreciable_sunk_cost", bp),
        "sunk_cost_non_depreciable": _assign_attr(
            "_oil_non_depreciable_sunk_cost", bp
        ),
        "sunk_cost": _assign_attr("_oil_sunk_cost", bp),

        # Attributes associated with preonstream cost
        "preonstream_depreciable": _assign_attr(
            "_oil_depreciable_preonstream", bp
        ),
        "preonstream_non_depreciable": _assign_attr(
            "_oil_non_depreciable_preonstream", bp
        ),
        "preonstream": _assign_attr("_oil_preonstream", bp),

        # Attributes associated with postonstream cost
        "postonstream_depreciable": oil_depreciable_postonstream,
        "postonstream_non_depreciable": oil_non_depreciable_postonstream,
        "postonstream": oil_postonstream,

        # Attributes associated with expenditures pre tax
        **pre_tax,

        # Attributes associated with indirect tax
        **indirect_tax,

        # Attributes associated with expenditures post tax
        **post_tax,

        # Attributes associated with expenses
        "expenses_capital": _assign_attr("_oil_capital", bp),
        "expenses_non_capital": _assign_attr("_oil_non_capital", bp),
        "expenses_total": _assign_attr("_oil_total_expenses", bp),

        # Attribute associated with cashflow
        "cashflow": _assign_attr("_oil_cashflow", bp),
    }

    # Convert OIL cashflow table into pandas DataFrame
    return pd.DataFrame(table_oil)


def get_table_baseproject_gas(contract: BaseProject) -> pd.DataFrame:
    """
    Construct the base project cashflow table for gas in a BaseProject contract.

    This function extracts gas-related economic attributes from the contract,
    including lifting, price, revenue, sunk costs, pre-/post-onstream expenditures,
    indirect taxes, and total cashflow. The extracted data are compiled into a
    structured pandas DataFrame suitable for economic analysis and reporting.

    Parameters
    ----------
    contract : BaseProject
        The BaseProject contract instance containing gas production and
        financial attributes.

    Returns
    -------
    pandas.DataFrame
        A DataFrame representing the gas cashflow table with columns for
        project years, lifting, prices, revenues, sunk costs, pre-/post-onstream
        costs, expenditures (pre-tax, indirect tax, post-tax), expenses, and
        overall cashflow.

    Notes
    -----
    - `_assign_attr` is used internally to retrieve contract attributes.
    - Post-onstream costs are separated into depreciable and non-depreciable components.
    - Expenditure categories include: capital, intangible, opex, ASR, LBT,
      and cost of sales.
    - The resulting DataFrame provides the core structure for gas-related
      project economic evaluation.
    """

    bp = contract

    # Specify postonstream attributes for GAS
    gas_depreciable_postonstream = _assign_attr(
        "_gas_capital_expenditures_post_tax", bp
    )

    gas_non_depreciable_postonstream = np.array(
        [
            _assign_attr(at, bp) for at in [
                "_gas_intangible_expenditures_post_tax",
                "_gas_opex_expenditures_post_tax",
                "_gas_asr_expenditures_post_tax",
                "_gas_lbt_expenditures_post_tax",
                "_gas_cost_of_sales_expenditures_post_tax",
            ]
        ]
    ).sum(axis=0)

    gas_postonstream = gas_depreciable_postonstream + gas_non_depreciable_postonstream

    # A list of cost categories
    categories = [
        "capital",
        "intangible",
        "opex",
        "asr",
        "lbt",
        "cost_of_sales"
    ]

    # Assign attributes associated with expenditures pre tax
    pre_tax = {
        f"{cat}_expenditures_pre_tax": _assign_attr(
            f"_gas_{cat}_expenditures_pre_tax", bp
        )
        for cat in categories
    }

    # Assign attributes associated with indirect tax
    indirect_tax = {
        f"{cat}_indirect_tax": _assign_attr(f"_gas_{cat}_indirect_tax", bp)
        for cat in categories
    }

    # Assign attributes associated with postonstream costs (or expenditures post tax)
    post_tax = {
        f"{cat}_postonstream": _assign_attr(f"_gas_{cat}_expenditures_post_tax", bp)
        for cat in categories
    }

    # Specify cashflow table for GAS
    table_gas: dict = {
        # Basic attributes
        "years": bp.project_years,
        "lifting": _assign_attr("_gas_lifting", bp, True),
        "price": _assign_attr("_gas_wap_price", bp),
        "revenue": _assign_attr("_gas_revenue", bp),

        # Attributes associated with sunk cost
        "sunk_cost_depreciable": _assign_attr("_gas_depreciable_sunk_cost", bp),
        "sunk_cost_non_depreciable": _assign_attr(
            "_gas_non_depreciable_sunk_cost", bp
        ),
        "sunk_cost": _assign_attr("_gas_sunk_cost", bp),

        # Attributes associated with preonstream cost
        "preonstream_depreciable": _assign_attr(
            "_gas_depreciable_preonstream", bp
        ),
        "preonstream_non_depreciable": _assign_attr(
            "_gas_non_depreciable_preonstream", bp
        ),
        "preonstream": _assign_attr("_gas_preonstream", bp),

        # Attributes associated with postonstream cost
        "postonstream_depreciable": gas_depreciable_postonstream,
        "postonstream_non_depreciable": gas_non_depreciable_postonstream,
        "postonstream": gas_postonstream,

        # Attributes associated with expenditures pre tax
        **pre_tax,

        # Attributes associated with indirect tax
        **indirect_tax,

        # Attributes associated with postonstream costs
        **post_tax,

        # Attributes associated with expenses
        "expenses_capital": _assign_attr("_gas_capital", bp),
        "expenses_non_capital": _assign_attr("_gas_non_capital", bp),
        "expenses_total": _assign_attr("_gas_total_expenses", bp),

        # Attribute associated with cashflow
        "cashflow": _assign_attr("_gas_cashflow", bp),
    }

    # Convert GAS cashflow table into pandas DataFrame
    return pd.DataFrame(table_gas)


def get_table_baseproject_consolidated(contract) -> pd.DataFrame:
    """
    Construct the consolidated base project cashflow table for a BaseProject contract.

    This function compiles consolidated financial and production data from the
    contract, including lifting, prices, revenues, sunk costs, pre-/post-onstream
    expenditures, indirect taxes, and total cashflow. The resulting data are
    structured into a pandas DataFrame for overall project-level economic analysis.

    Parameters
    ----------
    contract : BaseProject
        The BaseProject contract instance containing consolidated production
        and financial attributes.

    Returns
    -------
    pandas.DataFrame
        A DataFrame representing the consolidated cashflow table, including
        project years, lifting, prices, revenues, sunk costs, pre-/post-onstream
        costs, expenditures (pre-tax, indirect tax, post-tax), expenses, and
        total cashflow.

    Notes
    -----
    - `_assign_attr` is used internally to extract attributes from the contract.
    - Post-onstream costs are divided into depreciable and non-depreciable parts.
    - Expenditure categories include: capital, intangible, opex, ASR, LBT,
      and cost of sales.
    - The consolidated table integrates oil and gas components into a unified
      cashflow representation for the entire project.
    """

    bp = contract

    # Specify postonstream attributes for CONSOLIDATED
    consolidated_depreciable_postonstream = _assign_attr(
        "_consolidated_capital_expenditures_post_tax", bp
    )

    consolidated_non_depreciable_postonstream = np.array(
        [
            _assign_attr(at, bp) for at in [
                "_consolidated_intangible_expenditures_post_tax",
                "_consolidated_opex_expenditures_post_tax",
                "_consolidated_asr_expenditures_post_tax",
                "_consolidated_lbt_expenditures_post_tax",
                "_consolidated_cost_of_sales_expenditures_post_tax",
            ]
        ]
    ).sum(axis=0)

    consolidated_postonstream = (
        consolidated_depreciable_postonstream + consolidated_non_depreciable_postonstream
    )

    # A list of cost categories
    categories = [
        "capital",
        "intangible",
        "opex",
        "asr",
        "lbt",
        "cost_of_sales"
    ]

    # Assign attributes associated with expenditures pre tax
    pre_tax = {
        f"{cat}_expenditures_pre_tax": _assign_attr(
            f"_consolidated_{cat}_expenditures_pre_tax", bp
        )
        for cat in categories
    }

    # Assign attributes associated with indirect tax
    indirect_tax = {
        f"{cat}_indirect_tax": _assign_attr(f"_consolidated_{cat}_indirect_tax", bp)
        for cat in categories
    }

    # Assign attributes associated with postonstream costs (or expenditures post tax)
    post_tax = {
        f"{cat}_postonstream": _assign_attr(
            f"_consolidated_{cat}_expenditures_post_tax", bp
        )
        for cat in categories
    }

    # Specify cashflow table for CONSOLIDATED
    table_consolidated: dict = {
        # Basic attributes
        "years": bp.project_years,
        "lifting": _assign_attr("_consolidated_lifting", bp),
        "price": _assign_attr("_consolidated_wap_price", bp),
        "revenue": _assign_attr("_consolidated_revenue", bp),

        # Attributes associated with sunk cost
        "sunk_cost_depreciable": _assign_attr(
            "_consolidated_depreciable_sunk_cost", bp
        ),
        "sunk_cost_non_depreciable": _assign_attr(
            "_consolidated_non_depreciable_sunk_cost", bp
        ),
        "sunk_cost": _assign_attr("_consolidated_sunk_cost", bp),

        # Attributes associated with preonstream cost
        "preonstream_depreciable": _assign_attr(
            "_consolidated_depreciable_preonstream", bp
        ),
        "preonstream_non_depreciable": _assign_attr(
            "_consolidated_non_depreciable_preonstream", bp
        ),
        "preonstream": _assign_attr("_consolidated_preonstream", bp),

        # Attributes associated with postonstream cost
        "postonstream_depreciable": consolidated_depreciable_postonstream,
        "postonstream_non_depreciable": consolidated_non_depreciable_postonstream,
        "postonstream": consolidated_postonstream,

        # Attributes associated with expenditures pre tax
        **pre_tax,

        # Attributes associated with indirect tax
        **indirect_tax,

        # Attributes associated with postonstream costs
        **post_tax,

        # Attributes associated with expenses
        "expenses_capital": _assign_attr("_consolidated_capital", bp),
        "expenses_non_capital": _assign_attr("_consolidated_non_capital", bp),
        "expenses_total": _assign_attr("_consolidated_total_expenses", bp),

        # Attribute associated with cashflow
        "cashflow": _assign_attr("_consolidated_cashflow", bp),
    }

    # Convert CONSOLIDATED cashflow table into pandas DataFrame
    return pd.DataFrame(table_consolidated)


def get_table(
    contract: CostRecovery | GrossSplit | BaseProject | Transition,
) -> tuple:

    if isinstance(contract, CostRecovery):
        psc_table_oil = get_table_costrecovery_oil(contract=contract)
        psc_table_gas = get_table_costrecovery_gas(contract=contract)
        psc_table_consolidated = get_table_costrecovery_consolidated(contract=contract)

        print('\t')
        print(f'Filetype: {type(psc_table_oil)}')
        print(f'Length: {len(psc_table_oil)}')
        print('psc_table_oil = \n', psc_table_oil)


    elif isinstance(contract, GrossSplit):
        pass

    elif isinstance(contract, Transition):
        pass

    elif isinstance(contract, BaseProject):
        psc_table_oil = get_table_baseproject_oil(contract=contract)
        psc_table_gas = get_table_baseproject_gas(contract=contract)
        psc_table_consolidated = get_table_baseproject_consolidated(contract=contract)

        return psc_table_oil, psc_table_gas, psc_table_consolidated

