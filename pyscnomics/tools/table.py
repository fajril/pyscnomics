"""
A collection of procedures to generate cashflow of a contract in the form of DataFrame.
"""

import pandas as pd
import numpy as np
from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts.transition import Transition

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 150)


# Define sulfur, electricity, and CO2 as non-petroleum commodities
non_petroleum_commodities = ["sulfur", "electricity", "co2"]


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


def get_non_petroleum_commodity(
    commodity: str,
    contract: CostRecovery | GrossSplit | BaseProject | Transition,
) -> dict:
    """
    Retrieve lifting, price, and revenue data for a non-petroleum commodity.

    Validates the commodity name and extracts its associated economic
    attributes from the contract object using standardized attribute naming
    (e.g., ``_sulfur_lifting``, ``_sulfur_wap_price``, ``_sulfur_revenue``).

    Parameters
    ----------
    commodity : str
        Name of the non-petroleum commodity (e.g., ``"sulfur"``,
        ``"electricity"``, ``"co2"``). Must be listed in
        ``non_petroleum_commodities``.
    contract : CostRecovery or GrossSplit or BaseProject or Transition
        Contract instance providing commodity-specific attributes.

    Returns
    -------
    dict
        Dictionary containing lifting, price, and revenue arrays for the
        specified commodity.

    Raises
    ------
    ValueError
        If the commodity name is not recognized.
    """

    if commodity not in non_petroleum_commodities:
        raise ValueError(f"Invalid non-petroleum commodity: {commodity!r}")

    return {
        f"lifting_{commodity}": _assign_attr(
            attr=f"_{commodity}_lifting", contract=contract, is_lifting=True
        ),
        f"price_{commodity}": _assign_attr(
            attr=f"_{commodity}_wap_price", contract=contract
        ),
        f"revenue_{commodity}": _assign_attr(
            attr=f"_{commodity}_revenue", contract=contract
        )
    }


def get_table_costrecovery_oil(contract: CostRecovery) -> pd.DataFrame:
    """
    Build the oil cashflow table for a CostRecovery contract.

    Compiles yearly oil-related economic attributes—including lifting,
    revenues, expenditures (pre-tax, post-tax, indirect tax), sunk costs,
    depreciations, FTP components, cost recovery mechanics, DMO, taxation,
    government/contractor shares, and cashflow—into a pandas DataFrame.

    Parameters
    ----------
    contract : CostRecovery
        Contract instance containing all oil economic attributes.

    Returns
    -------
    pandas.DataFrame
        Tabular oil cashflow data indexed by project years.
    """

    cr = contract

    # Prepare non-petroleum commodities data
    sulfur, electricity, co2 = [
        get_non_petroleum_commodity(com, cr) for com in non_petroleum_commodities
    ]

    # Specify a list of cost categories
    categories = [
        "capital",
        "intangible",
        "opex",
        "asr",
        "lbt",
        "cost_of_sales"
    ]

    # Prepare attributes associated with expenditures pre tax
    pre_tax = {
        f"{cat}_expenditures_pre_tax": _assign_attr(
            f"_oil_{cat}_expenditures_pre_tax", cr
        )
        for cat in categories
    }

    # Prepare attributes associated with indirect tax
    indirect_tax = {
        f"{cat}_indirect_tax": _assign_attr(f"_oil_{cat}_indirect_tax", cr)
        for cat in categories
    }

    # Prepare attributes associated with postonstream costs (or expenditures post tax)
    post_tax = {
        f"{cat}_postonstream": _assign_attr(f"_oil_{cat}_expenditures_post_tax", cr)
        for cat in categories
    }

    # Prepare attributes associated with depreciations and non-depreciables
    depreciations = _assign_attr("_oil_depreciations", cr)
    non_depreciables = _assign_attr("_oil_non_depreciables", cr)

    # Specify cashflow table for OIL
    table_oil: dict = {
        # Basic attributes
        "years": cr.project_years,
        "lifting": _assign_attr("_oil_lifting", cr, True),
        "price": _assign_attr("_oil_wap_price", cr),
        "revenue": _assign_attr("_oil_revenue", cr),

        # Attributes associated with sulfur, electricity, and CO2 commodities
        **sulfur,
        **electricity,
        **co2,

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
        "postonstream_depreciable": _assign_attr("_oil_depreciable_postonstream", cr),
        "postonstream_non_depreciable": _assign_attr("_oil_non_depreciable_postonstream", cr),
        "postonstream": _assign_attr("_oil_postonstream", cr),

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

        # Attributes associated with non-depreciables
        "non_depreciables_sunk_cost": non_depreciables["sunk_cost"],
        "non_depreciables_preonstream": non_depreciables["preonstream"],
        "non_depreciables_postonstream": non_depreciables["postonstream"],

        # Attributes associated with FTP
        "ftp": _assign_attr("_oil_ftp", cr),
        "ftp_ctr": _assign_attr("_oil_ftp_ctr", cr),
        "ftp_gov": _assign_attr("_oil_ftp_gov", cr),

        # Attributes associated with core business logic
        "investment_credit": _assign_attr("_oil_ic_paid", cr),
        "recoverable_cost": _assign_attr("_oil_recoverable_cost", cr),
        "unrecovered_cost": _assign_attr("_oil_unrecovered_before_transfer", cr),
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
    """
    Build the gas cashflow table for a CostRecovery contract.

    Collects yearly gas-related economic attributes—including lifting,
    revenues, expenditures (pre-tax, post-tax, indirect tax), sunk costs,
    depreciations, FTP components, cost recovery mechanics, DMO, taxation,
    government/contractor shares, and cashflow—and compiles them into a
    pandas DataFrame.

    Parameters
    ----------
    contract : CostRecovery
        Contract instance containing all gas economic attributes.

    Returns
    -------
    pandas.DataFrame
        Tabular gas cashflow data indexed by project years.
    """

    cr = contract

    # Prepare non-petroleum commodities data
    sulfur, electricity, co2 = [
        get_non_petroleum_commodity(com, cr) for com in non_petroleum_commodities
    ]

    # Specify a list of cost categories
    categories = [
        "capital",
        "intangible",
        "opex",
        "asr",
        "lbt",
        "cost_of_sales",
    ]

    # Prepare attributes associated with expenditures pre tax
    pre_tax = {
        f"{cat}_expenditures_pre_tax": _assign_attr(
            f"_gas_{cat}_expenditures_pre_tax", cr
        )
        for cat in categories
    }

    # Prepare attributes associated with indirect tax
    indirect_tax = {
        f"{cat}_indirect_tax": _assign_attr(f"_gas_{cat}_indirect_tax", cr)
        for cat in categories
    }

    # Prepare attributes associated with postonstream costs (or expenditures post tax)
    post_tax = {
        f"{cat}_postonstream": _assign_attr(f"_gas_{cat}_expenditures_post_tax", cr)
        for cat in categories
    }

    # Prepare attributes associated with depreciations and non-depreciables
    depreciations = _assign_attr("_gas_depreciations", cr)
    non_depreciables = _assign_attr("_gas_non_depreciables", cr)

    # Specify cashflow table for GAS
    table_gas: dict = {
        # Basic attributes
        "years": cr.project_years,
        "lifting": _assign_attr("_gas_lifting", cr, True),
        "price": _assign_attr("_gas_wap_price", cr),
        "revenue": _assign_attr("_gas_revenue", cr),

        # Attributes associated with sulfur, electricity, and CO2 commodities
        **sulfur,
        **electricity,
        **co2,

        # Attributes associated with sunk cost
        "sunk_cost_depreciable": _assign_attr("_gas_depreciable_sunk_cost", cr),
        "sunk_cost_non_depreciable": _assign_attr("_gas_non_depreciable_sunk_cost", cr),
        "sunk_cost": _assign_attr("_gas_sunk_cost", cr),

        # Attributes associated with preonstream cost
        "preonstream_depreciable": _assign_attr("_gas_depreciable_preonstream", cr),
        "preonstream_non_depreciable": _assign_attr(
            "_gas_non_depreciable_preonstream", cr
        ),
        "preonstream": _assign_attr("_gas_preonstream", cr),

        # Attribute associated with postonstream cost
        "postonstream_depreciable": _assign_attr("_gas_depreciable_postonstream", cr),
        "postonstream_non_depreciable": _assign_attr("_gas_non_depreciable_postonstream", cr),
        "postonstream": _assign_attr("_gas_postonstream", cr),

        # Attributes associated with expenditures pre tax
        **pre_tax,

        # Attributes associated with indirect tax
        **indirect_tax,

        # Attributes associated with expenditures post tax
        **post_tax,

        # Attributes associated with expenses
        "expenses_capital": _assign_attr("_gas_capital", cr),
        "expenses_non_capital": _assign_attr("_gas_non_capital", cr),
        "expenses_total": _assign_attr("_gas_total_expenses", cr),

        # Attributes associated with depreciations
        "depreciations_sunk_cost": depreciations["sunk_cost"],
        "depreciations_preonstream": depreciations["preonstream"],
        "depreciations_postonstream": depreciations["postonstream"],

        # Attributes associated with non-depreciables
        "non_depreciables_sunk_cost": non_depreciables["sunk_cost"],
        "non_depreciables_preonstream": non_depreciables["preonstream"],
        "non_depreciables_postonstream": non_depreciables["postonstream"],

        # Attributes associated with FTP
        "ftp": _assign_attr("_gas_ftp", cr),
        "ftp_ctr": _assign_attr("_gas_ftp_ctr", cr),
        "ftp_gov": _assign_attr("_gas_ftp_gov", cr),

        # Attributes associated with core business logic
        "investment_credit": _assign_attr("_gas_ic_paid", cr),
        "recoverable_cost": _assign_attr("_gas_recoverable_cost", cr),
        "unrecovered_cost": _assign_attr("_gas_unrecovered_before_transfer", cr),
        "cost_recovery": _assign_attr("_gas_cost_recovery", cr),
        "ets_before_transfer": _assign_attr("_gas_ets_before_transfer", cr),
        "transfer_to_oil": _assign_attr("_transfer_to_oil", cr),
        "unrec_after_transfer": _assign_attr("_gas_unrecovered_after_transfer", cr),
        "cost_recovery_after_tf": _assign_attr("_gas_cost_recovery_after_tf", cr),
        "ets_after_transfer": _assign_attr("_gas_ets_after_transfer", cr),
        "contractor_share": _assign_attr("_gas_contractor_share", cr),
        "government_share": _assign_attr("_gas_government_share", cr),

        # Attributes associated with DMO
        "dmo_volume": _assign_attr("_gas_dmo_volume", cr),
        "dmo_fee": _assign_attr("_gas_dmo_fee", cr),
        "ddmo": _assign_attr("_gas_ddmo", cr),

        # Attributes associated with taxable income
        "taxable_income": _assign_attr("_gas_taxable_income", cr),
        "tax_payment": _assign_attr("_gas_tax_payment", cr),

        # Attributes associated with government and contractor shares
        "contractor_net_share": _assign_attr("_gas_ctr_net_share", cr),
        "government_take": _assign_attr("_gas_government_take", cr),

        # Attributes associated with cashflow
        "cashflow": _assign_attr("_gas_cashflow", cr),
        "cum_cashflow": np.cumsum(_assign_attr("_gas_cashflow", cr)),
    }

    # Convert GAS cashflow table into pandas DataFrame
    return pd.DataFrame(table_gas)


def get_table_costrecovery_consolidated(contract: CostRecovery) -> pd.DataFrame:
    """
    Build the consolidated cashflow table for a CostRecovery contract.

    Aggregates oil and gas economic components into a unified consolidated
    table, including lifting, revenues, expenditures (pre-tax, post-tax,
    indirect tax), sunk and preonstream costs, depreciations, FTP components,
    cost recovery mechanics, DMO, taxation, government/contractor shares,
    and cashflow.

    Parameters
    ----------
    contract : CostRecovery
        Contract instance containing consolidated economic attributes.

    Returns
    -------
    pandas.DataFrame
        Tabular consolidated cashflow data indexed by project years.
    """

    cr = contract

    # Prepare non-petroleum commodities data
    sulfur, electricity, co2 = [
        get_non_petroleum_commodity(com, cr) for com in non_petroleum_commodities
    ]

    # Specify a list of cost categories
    categories = [
        "capital",
        "intangible",
        "opex",
        "asr",
        "lbt",
        "cost_of_sales"
    ]

    # Prepare attributes associated with expenditures pre tax
    pre_tax = {
        f"{cat}_expenditures_pre_tax": _assign_attr(
            f"_consolidated_{cat}_expenditures_pre_tax", cr
        )
        for cat in categories
    }

    # Prepare attributes associated with indirect tax
    indirect_tax = {
        f"{cat}_indirect_tax": _assign_attr(
            f"_consolidated_{cat}_indirect_tax", cr
        )
        for cat in categories
    }

    # Prepare attributes associated with postonstream costs (or expenditures post tax)
    post_tax = {
        f"{cat}_postonstream": _assign_attr(
            f"_consolidated_{cat}_expenditures_post_tax", cr
        )
        for cat in categories
    }

    # Prepare attributes associated with depreciations and non-depreciables
    depreciations = _assign_attr("_consolidated_depreciations", cr)
    non_depreciables = _assign_attr("_consolidated_non_depreciables", cr)

    # Specify cashflow table for CONSOLIDATED
    table_consolidated: dict = {
        # Basic attributes
        "years": cr.project_years,
        "lifting": _assign_attr("_consolidated_lifting", cr),
        "price": _assign_attr("_consolidated_wap_price", cr),
        "revenue": _assign_attr("_consolidated_revenue", cr),

        # Attributes associated with sulfur, electricity, and CO2 commodities
        **sulfur,
        **electricity,
        **co2,

        # Attributes associated with sunk cost
        "sunk_cost_depreciable": _assign_attr("_consolidated_depreciable_sunk_cost", cr),
        "sunk_cost_non_depreciable": _assign_attr(
            "_consolidated_non_depreciable_sunk_cost", cr
        ),
        "sunk_cost": _assign_attr("_consolidated_sunk_cost", cr),

        # Attributes associated with preonstream cost
        "preonstream_depreciable": _assign_attr(
            "_consolidated_depreciable_preonstream", cr
        ),
        "preonstream_non_depreciable": _assign_attr(
            "_consolidated_non_depreciable_preonstream", cr
        ),
        "preonstream": _assign_attr("_consolidated_preonstream", cr),

        # Attributes associated with postonstream cost
        "postonstream_depreciable": _assign_attr(
            "_consolidated_depreciable_postonstream", cr
        ),
        "postonstream_non_depreciable": _assign_attr(
            "_consolidated_non_depreciable_postonstream", cr
        ),
        "postonstream": _assign_attr("_consolidated_postonstream", cr),

        # Attributes associated with expenditures pre tax
        **pre_tax,

        # Attributes associated with indirect tax
        **indirect_tax,

        # Attributes associated with expenditures post tax
        **post_tax,

        # Attributes associated with expenses
        "expenses_capital": _assign_attr("_consolidated_capital", cr),
        "expenses_non_capital": _assign_attr("_consolidated_non_capital", cr),
        "expenses_total": _assign_attr("_consolidated_total_expenses", cr),

        # Attributes associated with depreciations
        "depreciations_sunk_cost": depreciations["sunk_cost"],
        "depreciations_preonstream": depreciations["preonstream"],
        "depreciations_postonstream": depreciations["postonstream"],

        # Attributes associated with non-depreciables
        "non_depreciables_sunk_cost": non_depreciables["sunk_cost"],
        "non_depreciables_preonstream": non_depreciables["preonstream"],
        "non_depreciables_postonstream": non_depreciables["postonstream"],

        # Attributes associated with FTP
        "ftp": _assign_attr("_consolidated_ftp", cr),
        "ftp_ctr": _assign_attr("_consolidated_ftp_ctr", cr),
        "ftp_gov": _assign_attr("_consolidated_ftp_gov", cr),

        # Attributes associated with core business logic
        "investment_credit": _assign_attr("_consolidated_ic_paid", cr),
        "recoverable_cost": _assign_attr("_consolidated_recoverable_cost", cr),
        "unrecovered_cost": _assign_attr(
            "_consolidated_unrecovered_before_transfer", cr
        ),
        "cost_recovery": _assign_attr(
            "_consolidated_cost_recovery_before_transfer", cr
        ),
        "ets_before_transfer": _assign_attr("_consolidated_ets_before_transfer", cr),
        "unrec_after_transfer": _assign_attr(
            "_consolidated_unrecovered_after_transfer", cr
        ),
        "cost_recovery_after_tf": _assign_attr(
            "_consolidated_cost_recovery_after_tf", cr
        ),
        "ets_after_transfer": _assign_attr("_consolidated_ets_after_transfer", cr),
        "contractor_share": _assign_attr("_consolidated_contractor_share", cr),
        "government_share": _assign_attr("_consolidated_government_share", cr),

        # Attributes associated with DMO
        "dmo_volume": _assign_attr("_consolidated_dmo_volume", cr),
        "dmo_fee": _assign_attr("_consolidated_dmo_fee", cr),
        "ddmo": _assign_attr("_consolidated_ddmo", cr),

        # Attributes associated with taxable income
        "taxable_income": _assign_attr("_consolidated_taxable_income", cr),
        "tax_payment": _assign_attr("_consolidated_tax_payment", cr),

        # Attributes associated with government and contractor shares
        "contractor_net_share": _assign_attr("_consolidated_ctr_net_share", cr),
        "contractor_take": _assign_attr("_consolidated_contractor_take", cr),
        "government_take": _assign_attr("_consolidated_government_take", cr),

        # Attributes associated with cashflow
        "cashflow": _assign_attr("_consolidated_cashflow", cr),
        "cum_cashflow": np.cumsum(_assign_attr("_consolidated_cashflow", cr)),
    }

    # Convert CONSOLIDATED cashflow table into pandas DataFrame
    return pd.DataFrame(table_consolidated)


def get_table_grosssplit_oil(contract: GrossSplit) -> pd.DataFrame:
    """
    Build the OIL cashflow table for a GrossSplit contract.

    Assembles lifting, revenue, cost components, taxes, depreciation,
    amortization, progressive/variable splits, government–contractor shares,
    and cashflow values. Non-petroleum commodity data (sulfur, electricity,
    CO₂) is included using `get_non_petroleum_commodity`.

    Parameters
    ----------
    contract : GrossSplit
        Contract instance providing OIL-related economic attributes.

    Returns
    -------
    pandas.DataFrame
        Cashflow table for OIL, containing prices, revenues, costs,
        taxes, splits, shares, and cashflow metrics.

    Notes
    -----
    Attribute access is standardized through ``_assign_attr`` and category-based
    dictionary expansions for cost components.
    """

    gs = contract

    # Prepare non-petroleum commodities data
    sulfur, electricity, co2 = [
        get_non_petroleum_commodity(com, gs) for com in non_petroleum_commodities
    ]

    # Prepare postonstream attributes for OIL
    oil_depreciable_postonstream = _assign_attr(
        "_oil_capital_expenditures_post_tax", gs
    )

    oil_non_depreciable_postonstream = np.array(
        [
            _assign_attr(at, gs) for at in [
                "_oil_intangible_expenditures_post_tax",
                "_oil_opex_expenditures_post_tax",
                "_oil_asr_expenditures_post_tax",
                "_oil_lbt_expenditures_post_tax",
                "_oil_cost_of_sales_expenditures_post_tax",
            ]
        ]
    ).sum(axis=0)

    oil_postonstream = oil_depreciable_postonstream + oil_non_depreciable_postonstream

    # Specify a list of cost categories
    categories = [
        "capital",
        "intangible",
        "opex",
        "asr",
        "lbt",
        "cost_of_sales"
    ]

    # Prepare attributes associated with expenditures pre tax
    pre_tax = {
        f"{cat}_expenditures_pre_tax": _assign_attr(
            f"_oil_{cat}_expenditures_pre_tax", gs
        )
        for cat in categories
    }

    # Prepare attributes associated with indirect tax
    indirect_tax = {
        f"{cat}_indirect_tax": _assign_attr(f"_oil_{cat}_indirect_tax", gs)
        for cat in categories
    }

    # Prepare attributes associated with postonstream costs (or expenditures post tax)
    post_tax = {
        f"{cat}_postonstream": _assign_attr(f"_oil_{cat}_expenditures_post_tax", gs)
        for cat in categories
    }

    # Prepare attribute associated with depreciations
    depreciations = _assign_attr("_oil_depreciations", gs)

    # Prepare attribute associated with amortizations
    amortizations = _assign_attr("_oil_amortizations", gs)

    # Specify cashflow table for OIL
    table_oil: dict = {
        # Basic attributes
        "years": gs.project_years,
        "lifting": _assign_attr("_oil_lifting", gs, True),
        "price": _assign_attr("_oil_wap_price", gs),
        "revenue": _assign_attr("_oil_revenue", gs),

        # Attributes associated with sulfur, electricity, and CO2 commodities
        **sulfur,
        **electricity,
        **co2,

        # Attributes associated with sunk cost
        "sunk_cost_depreciable": _assign_attr("_oil_depreciable_sunk_cost", gs),
        "sunk_cost_non_depreciable": _assign_attr("_oil_non_depreciable_sunk_cost", gs),
        "sunk_cost": _assign_attr("_oil_sunk_cost", gs),

        # Attributes associated with preonstream cost
        "preonstream_depreciable": _assign_attr("_oil_depreciable_preonstream", gs),
        "preonstream_non_depreciable": _assign_attr(
            "_oil_non_depreciable_preonstream", gs
        ),
        "preonstream": _assign_attr("_oil_preonstream", gs),

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
        "expenses_capital": _assign_attr("_oil_capital", gs),
        "expenses_non_capital": _assign_attr("_oil_non_capital", gs),
        "expenses_total": _assign_attr("_oil_total_expenses", gs),

        # Attributes associated with depreciations
        "depreciations_sunk_cost": depreciations["sunk_cost"],
        "depreciations_preonstream": depreciations["preonstream"],
        "depreciations_postonstream": depreciations["postonstream"],

        # Attributes associated with amortizations
        "amortizations_sunk_cost": amortizations["sunk_cost"],
        "amortizations_preonstream": amortizations["preonstream"],
        "amortizations_postonstream": amortizations["postonstream"],

        # Attributes associated with splits
        "base_split": _assign_attr("_oil_base_split", gs),
        "variable_split": _assign_attr("_var_split_array", gs),
        "progressive_price_split": _assign_attr("_oil_prog_price_split", gs),
        "progressive_cum_prod_split": _assign_attr("_oil_prog_cum_split", gs),
        "progressive_split": _assign_attr("_oil_prog_split", gs),
        "contractor_split": _assign_attr("_oil_ctr_split", gs),

        # Attributes associated with shares
        "contractor_share": _assign_attr("_oil_ctr_share_before_transfer", gs),
        "government_share": _assign_attr("_oil_gov_share", gs),

        # Attributes associated with business logic
        "cost_to_be_deducted": _assign_attr("_oil_cost_tobe_deducted", gs),
        "carry_forward_cost": _assign_attr("_oil_carward_deduct_cost", gs),
        "deductible_cost": _assign_attr("_oil_deductible_cost", gs),
        "transfer_to_gas": _assign_attr("_transfer_to_gas", gs),
        "carry_forward_cost_after_tf": _assign_attr("_oil_carward_cost_aftertf", gs),
        "profit_pre_transfer": _assign_attr("_oil_profit_pre_transfer", gs),
        "ctr_net_operating_profit": _assign_attr("_oil_net_operating_profit", gs),

        # Attributes associated with DMO
        "dmo_volume": _assign_attr("_oil_dmo_volume", gs),
        "dmo_fee": _assign_attr("_oil_dmo_fee", gs),
        "ddmo": _assign_attr("_oil_ddmo", gs),

        # Attributes associated with taxable income
        "taxable_income": _assign_attr("_oil_taxable_income", gs),
        "tax": _assign_attr("_oil_tax", gs),

        # Attributes associated with government and contractor shares
        "net_ctr_share": _assign_attr("_oil_ctr_net_share", gs),
        "government_take": _assign_attr("_oil_government_take", gs),

        # Attribute associated with cashflow
        "ctr_cashflow": _assign_attr("_oil_ctr_cashflow", gs),
        "cum_cashflow": np.cumsum(_assign_attr("_oil_ctr_cashflow", gs)),
    }

    # Convert OIL cashflow table into pandas DataFrame
    return pd.DataFrame(table_oil)


def get_table_grosssplit_gas(contract: GrossSplit) -> pd.DataFrame:
    """
    Build the GAS cashflow table for a GrossSplit contract.

    Collects lifting, revenue, cost components, indirect taxes,
    depreciations, amortizations, progressive/variable splits, contractor
    and government shares, and cashflow-related values. Non-petroleum
    commodity data (sulfur, electricity, CO₂) is included via
    `get_non_petroleum_commodity`.

    Parameters
    ----------
    contract : GrossSplit
        Contract instance supplying GAS-related economic attributes.

    Returns
    -------
    pandas.DataFrame
        Cashflow table for GAS, including prices, revenues, cost categories,
        taxes, splits, shares, and contractor cashflow metrics.

    Notes
    -----
    Attributes are populated through ``_assign_attr`` and expanded using
    category-based dictionaries for pre-tax, indirect-tax, and post-tax costs.
    """

    gs = contract

    # Prepare non-petroleum commodities data
    sulfur, electricity, co2 = [
        get_non_petroleum_commodity(com, gs) for com in non_petroleum_commodities
    ]

    # Prepare postonstream attributes for GAS
    gas_depreciable_postonstream = _assign_attr(
        "_gas_capital_expenditures_post_tax", gs
    )

    gas_non_depreciable_postonstream = np.array(
        [
            _assign_attr(at, gs) for at in [
                "_gas_intangible_expenditures_post_tax",
                "_gas_opex_expenditures_post_tax",
                "_gas_asr_expenditures_post_tax",
                "_gas_lbt_expenditures_post_tax",
                "_gas_cost_of_sales_expenditures_post_tax",
            ]
        ]
    ).sum(axis=0)

    gas_postonstream = gas_depreciable_postonstream + gas_non_depreciable_postonstream

    # Specify a list of cost categories
    categories = [
        "capital",
        "intangible",
        "opex",
        "asr",
        "lbt",
        "cost_of_sales"
    ]

    # Prepare attributes associated with expenditures pre tax
    pre_tax = {
        f"{cat}_expenditures_pre_tax": _assign_attr(
            f"_gas_{cat}_expenditures_pre_tax", gs
        )
        for cat in categories
    }

    # Prepare attributes associated with indirect tax
    indirect_tax = {
        f"{cat}_indirect_tax": _assign_attr(f"_gas_{cat}_indirect_tax", gs)
        for cat in categories
    }

    # Prepare attributes associated with postonstream costs (or expenditures post tax)
    post_tax = {
        f"{cat}_postonstream": _assign_attr(f"_gas_{cat}_expenditures_post_tax", gs)
        for cat in categories
    }

    # Prepare attribute associated with depreciations
    depreciations = _assign_attr("_gas_depreciations", gs)

    # Prepare attribute associated with amortizations
    amortizations = _assign_attr("_gas_amortizations", gs)

    # Specify cashflow table for GAS
    table_gas: dict = {
        # Basic attributes
        "years": gs.project_years,
        "lifting": _assign_attr("_gas_lifting", gs, True),
        "price": _assign_attr("_gas_wap_price", gs),
        "revenue": _assign_attr("_gas_revenue", gs),

        # Attributes associated with sulfur, electricity, and CO2 commodities
        **sulfur,
        **electricity,
        **co2,

        # Attributes associated with sunk cost
        "sunk_cost_depreciable": _assign_attr("_gas_depreciable_sunk_cost", gs),
        "sunk_cost_non_depreciable": _assign_attr("_gas_non_depreciable_sunk_cost", gs),
        "sunk_cost": _assign_attr("_gas_sunk_cost", gs),

        # Attributes associated with preonstream cost
        "preonstream_depreciable": _assign_attr("_gas_depreciable_preonstream", gs),
        "preonstream_non_depreciable": _assign_attr(
            "_gas_non_depreciable_preonstream", gs
        ),
        "preonstream": _assign_attr("_gas_preonstream", gs),

        # Attributes associated with postonstream cost
        "postonstream_depreciable": gas_depreciable_postonstream,
        "postonstream_non_depreciable": gas_non_depreciable_postonstream,
        "postonstream": gas_postonstream,

        # Attributes associated with expenditures pre tax
        **pre_tax,

        # Attributes associated with indirect tax
        **indirect_tax,

        # Attributes associated with expenditures post tax
        **post_tax,

        # Attributes associated with expenses
        "expenses_capital": _assign_attr("_gas_capital", gs),
        "expenses_non_capital": _assign_attr("_gas_non_capital", gs),
        "expenses_total": _assign_attr("_gas_total_expenses", gs),

        # Attributes associated with depreciations
        "depreciations_sunk_cost": depreciations["sunk_cost"],
        "depreciations_preonstream": depreciations["preonstream"],
        "depreciations_postonstream": depreciations["postonstream"],

        # Attributes associated with amortizations
        "amortizations_sunk_cost": amortizations["sunk_cost"],
        "amortizations_preonstream": amortizations["preonstream"],
        "amortizations_postonstream": amortizations["postonstream"],

        # Attributes associated with splits
        "base_split": _assign_attr("_gas_base_split", gs),
        "variable_split": _assign_attr("_var_split_array", gs),
        "progressive_price_split": _assign_attr("_gas_prog_price_split", gs),
        "progressive_cum_prod_split": _assign_attr("_gas_prog_cum_split", gs),
        "progressive_split": _assign_attr("_gas_prog_split", gs),
        "contractor_split": _assign_attr("_gas_ctr_split", gs),

        # Attributes associated with shares
        "contractor_share": _assign_attr("_gas_ctr_share_before_transfer", gs),
        "government_share": _assign_attr("_gas_gov_share", gs),

        # Attributes associated with business logic
        "cost_to_be_deducted": _assign_attr("_gas_cost_tobe_deducted", gs),
        "carry_forward_cost": _assign_attr("_gas_carward_deduct_cost", gs),
        "deductible_cost": _assign_attr("_gas_deductible_cost", gs),
        "transfer_to_oil": _assign_attr("_transfer_to_oil", gs),
        "carry_forward_cost_after_tf": _assign_attr("_gas_carward_cost_aftertf", gs),
        "profit_pre_transfer": _assign_attr("_gas_profit_pre_transfer", gs),
        "ctr_net_operating_profit": _assign_attr("_gas_net_operating_profit", gs),

        # Attributes associated with DMO
        "dmo_volume": _assign_attr("_gas_dmo_volume", gs),
        "dmo_fee": _assign_attr("_gas_dmo_fee", gs),
        "ddmo": _assign_attr("_gas_ddmo", gs),

        # Attributes associated with taxable income
        "taxable_income": _assign_attr("_gas_taxable_income", gs),
        "tax": _assign_attr("_gas_tax", gs),

        # Attributes associated with government and contractor shares
        "net_ctr_share": _assign_attr("_gas_ctr_net_share", gs),
        "government_take": _assign_attr("_gas_government_take", gs),

        # Attributes associated with cashflow
        "ctr_cashflow": _assign_attr("_gas_ctr_cashflow", gs),
        "cum_cashflow": np.cumsum(_assign_attr("_gas_ctr_cashflow", gs)),
    }

    # Convert GAS cashflow table into pandas DataFrame
    return pd.DataFrame(table_gas)


def get_table_grosssplit_consolidated(contract: GrossSplit) -> pd.DataFrame:
    """
    Build the CONSOLIDATED cashflow table for a GrossSplit contract.

    Aggregates lifting, revenue, cost components (pre-tax, indirect-tax,
    post-tax), depreciations, amortizations, contractor/government shares,
    and cashflow metrics. Includes sulfur, electricity, and CO₂ data via
    `get_non_petroleum_commodity`.

    Parameters
    ----------
    contract : GrossSplit
        Contract instance providing consolidated economic attributes.

    Returns
    -------
    pandas.DataFrame
        Consolidated cashflow table containing revenues, costs, taxes,
        shares, and cashflow values.

    Notes
    -----
    Cost components are expanded using category-based dictionaries, and all
    economic attributes are retrieved through ``_assign_attr`` for consistency.
    """

    gs = contract

    # Prepare non-petroleum commodities data
    sulfur, electricity, co2 = [
        get_non_petroleum_commodity(com, gs) for com in non_petroleum_commodities
    ]

    # Specify postonstream attributes for CONSOLIDATED
    consolidated_depreciable_postonstream = _assign_attr(
        "_consolidated_capital_expenditures_post_tax", gs
    )

    consolidated_non_depreciable_postonstream = np.array(
        [
            _assign_attr(at, gs) for at in [
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

    # Specify a list of cost categories
    categories = [
        "capital",
        "intangible",
        "opex",
        "asr",
        "lbt",
        "cost_of_sales"
    ]

    # Prepare attributes associated with expenditures pre tax
    pre_tax = {
        f"{cat}_expenditures_pre_tax": _assign_attr(
            f"_consolidated_{cat}_expenditures_pre_tax", gs
        )
        for cat in categories
    }

    # Prepare attributes associated with indirect tax
    indirect_tax = {
        f"{cat}_indirect_tax": _assign_attr(
            f"_consolidated_{cat}_indirect_tax", gs
        )
        for cat in categories
    }

    # Prepare attributes associated with postonstream costs (or expenditures post tax)
    post_tax = {
        f"{cat}_postonstream": _assign_attr(
            f"_consolidated_{cat}_expenditures_post_tax", gs
        )
        for cat in categories
    }

    # Prepare attribute associated with depreciations
    depreciations = _assign_attr("_consolidated_depreciations", gs)

    # Prepare attribute associated with amortizations
    amortizations = _assign_attr("_consolidated_amortizations", gs)

    # Specify cashflow table for CONSOLIDATED
    table_consolidated: dict = {
        # Basic attributes
        "years": gs.project_years,
        "lifting": _assign_attr("_consolidated_lifting", gs),
        "price": _assign_attr("_consolidated_wap_price", gs),
        "revenue": _assign_attr("_consolidated_revenue", gs),

        # Attributes associated with sulfur, electricity, and CO2
        **sulfur,
        **electricity,
        **co2,

        # Attributes associated with sunk cost
        "sunk_cost_depreciable": _assign_attr("_consolidated_depreciable_sunk_cost", gs),
        "sunk_cost_non_depreciable": _assign_attr(
            "_consolidated_non_depreciable_sunk_cost", gs
        ),
        "sunk_cost": _assign_attr("_consolidated_sunk_cost", gs),

        # Attributes associated with preonstream cost
        "preonstream_depreciable": _assign_attr(
            "_consolidated_depreciable_preonstream", gs
        ),
        "preonstream_non_depreciable": _assign_attr(
            "_consolidated_non_depreciable_preonstream", gs
        ),
        "preonstream": _assign_attr("_consolidated_preonstream", gs),

        # Attributes associated with postonstream cost
        "postonstream_depreciable": consolidated_depreciable_postonstream,
        "postonstream_non_depreciable": consolidated_non_depreciable_postonstream,
        "postonstream": consolidated_postonstream,

        # Attributes associated with expenditures pre tax
        **pre_tax,

        # Attributes associated with indirect tax
        **indirect_tax,

        # Attributes associated with expenditures post tax
        **post_tax,

        # Attributes associated with expenses
        "expenses_capital": _assign_attr("_consolidated_capital", gs),
        "expenses_non_capital": _assign_attr("_consolidated_non_capital", gs),
        "expenses_total": _assign_attr("_consolidated_total_expenses", gs),

        # Attributes associated with depreciations
        "depreciations_sunk_cost": depreciations["sunk_cost"],
        "depreciations_preonstream": depreciations["preonstream"],
        "depreciations_postonstream": depreciations["postonstream"],

        # Attributes associated with amortizations
        "amortizations_sunk_cost": amortizations["sunk_cost"],
        "amortizations_preonstream": amortizations["preonstream"],
        "amortizations_postonstream": amortizations["postonstream"],

        # Attributes associated with shares
        "contractor_share": _assign_attr("_consolidated_ctr_share_before_tf", gs),
        "government_share": _assign_attr("_consolidated_gov_share_before_tf", gs),

        # Attributes associated with business logic
        "cost_to_be_deducted": _assign_attr("_consolidated_cost_tobe_deducted", gs),
        "carry_forward_cost": _assign_attr("_consolidated_carward_deduct_cost", gs),
        "deductible_cost": _assign_attr("_consolidated_deductible_cost", gs),
        "carry_forward_cost_after_tf": _assign_attr(
            "_consolidated_carward_cost_aftertf", gs
        ),
        "ctr_net_operating_profit": _assign_attr(
            "_consolidated_net_operating_profit", gs
        ),

        # Attributes associated with DMO
        "dmo_volume": _assign_attr("_consolidated_dmo_volume", gs),
        "dmo_fee": _assign_attr("_consolidated_dmo_fee", gs),
        "ddmo": _assign_attr("_consolidated_ddmo", gs),

        # Attributes associated with taxable income
        "taxable_income": _assign_attr("_consolidated_taxable_income", gs),
        "tax": _assign_attr("_consolidated_tax_payment", gs),

        # Attributes associated with government and contractor shares
        "net_ctr_share": _assign_attr("_consolidated_ctr_net_share", gs),
        "government_take": _assign_attr("_consolidated_government_take", gs),

        # Attributes associated with cashflow
        "ctr_cashflow": _assign_attr("_consolidated_cashflow", gs),
        "cum_cashflow": np.cumsum(_assign_attr("_consolidated_cashflow", gs)),
    }

    # Convert CONSOLIDATED cashflow table into pandas DataFrame
    return pd.DataFrame(table_consolidated)


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

    # Prepare non-petroleum commodities data
    sulfur, electricity, co2 = [
        get_non_petroleum_commodity(com, bp) for com in non_petroleum_commodities
    ]

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

        # Attributes associated with sulfur, electricity, and CO2 commodities
        **sulfur,
        **electricity,
        **co2,

        # Attributes associated with sunk cost
        "sunk_cost_depreciable": _assign_attr("_oil_depreciable_sunk_cost", bp),
        "sunk_cost_non_depreciable": _assign_attr("_oil_non_depreciable_sunk_cost", bp),
        "sunk_cost": _assign_attr("_oil_sunk_cost", bp),

        # Attributes associated with preonstream cost
        "preonstream_depreciable": _assign_attr("_oil_depreciable_preonstream", bp),
        "preonstream_non_depreciable": _assign_attr(
            "_oil_non_depreciable_preonstream", bp
        ),
        "preonstream": _assign_attr("_oil_preonstream", bp),

        # Attributes associated with postonstream cost
        "postonstream_depreciable": _assign_attr("_oil_depreciable_postonstream", bp),
        "postonstream_non_depreciable": _assign_attr(
            "_oil_non_depreciable_postonstream", bp
        ),
        "postonstream": _assign_attr("_oil_postonstream", bp),

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

    # Prepare non-petroleum commodities data
    sulfur, electricity, co2 = [
        get_non_petroleum_commodity(com, bp) for com in non_petroleum_commodities
    ]

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

        # Attributes associated with sulfur, electricity, and CO2 commodities
        **sulfur,
        **electricity,
        **co2,

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
        "postonstream_depreciable": _assign_attr("_gas_depreciable_postonstream", bp),
        "postonstream_non_depreciable": _assign_attr(
            "_gas_non_depreciable_postonstream", bp
        ),
        "postonstream": _assign_attr("_gas_postonstream", bp),

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


def get_table_baseproject_consolidated(contract: BaseProject) -> pd.DataFrame:
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

    # Prepare non-petroleum commodities data
    sulfur, electricity, co2 = [
        get_non_petroleum_commodity(com, bp) for com in non_petroleum_commodities
    ]

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

        # Attributes associated with sulfur, electricity, and CO2 commodities
        **sulfur,
        **electricity,
        **co2,

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
        "postonstream_depreciable": _assign_attr(
            "_consolidated_depreciable_postonstream", bp
        ),
        "postonstream_non_depreciable": _assign_attr(
            "_consolidated_non_depreciable_postonstream", bp
        ),
        "postonstream": _assign_attr("_consolidated_postonstream", bp),

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
    """
    Generate OIL, GAS, and CONSOLIDATED cashflow tables for a PSC contract.

    Depending on the contract type (CostRecovery, GrossSplit, BaseProject, or
    Transition), this function dispatches to the appropriate table-construction
    routines and returns the three corresponding cashflow tables.

    Parameters
    ----------
    contract : CostRecovery or GrossSplit or BaseProject or Transition
        PSC contract instance for which cashflow tables are generated.

    Returns
    -------
    tuple of pandas.DataFrame
        A tuple containing:
        - oil_table : DataFrame
            Cashflow table for OIL stream.
        - gas_table : DataFrame
            Cashflow table for GAS stream.
        - consolidated_table : DataFrame
            Consolidated cashflow table combining OIL and GAS.
    """

    # Construct OIL, GAS, and CONSOLIDATED cashflow tables for CR contract
    if isinstance(contract, CostRecovery):
        psc_table_oil = get_table_costrecovery_oil(contract=contract)
        psc_table_gas = get_table_costrecovery_gas(contract=contract)
        psc_table_consolidated = get_table_costrecovery_consolidated(contract=contract)
        return psc_table_oil, psc_table_gas, psc_table_consolidated

    # Construct OIL, GAS, and CONSOLIDATED cashflow tables for GS contract
    elif isinstance(contract, GrossSplit):
        psc_table_oil = get_table_grosssplit_oil(contract=contract)
        psc_table_gas = get_table_grosssplit_gas(contract=contract)
        psc_table_consolidated = get_table_grosssplit_consolidated(contract=contract)
        return psc_table_oil, psc_table_gas, psc_table_consolidated

    # Construct OIL, GAS, and CONSOLIDATED cashflow tables for transition contract
    elif isinstance(contract, Transition):
        pass

    # Construct OIL, GAS, and CONSOLIDATED cashflow tables for base project contract
    elif isinstance(contract, BaseProject):
        psc_table_oil = get_table_baseproject_oil(contract=contract)
        psc_table_gas = get_table_baseproject_gas(contract=contract)
        psc_table_consolidated = get_table_baseproject_consolidated(contract=contract)
        return psc_table_oil, psc_table_gas, psc_table_consolidated
