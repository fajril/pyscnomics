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
    value = getattr(contract, attr)

    return value.get_lifting_rate_ghv_arr() if is_lifting else value


def get_table_baseproject_oil(contract) -> pd.DataFrame:

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

    # Specify cashflow table for OIL
    table_oil: dict = {
        # Attribute associated with project years
        "years": bp.project_years,

        # Attributes associated with lifting
        "lifting_oil": _assign_attr("_oil_lifting", bp, True),
        "lifting_sulfur": _assign_attr("_sulfur_lifting", bp, True),
        "lifting_electricity": _assign_attr(
            "_electricity_lifting", bp, True
        ),
        "lifting_co2": _assign_attr("_co2_lifting", bp, True),

        # Attributes associated with price
        "price_oil": _assign_attr("_oil_wap_price", bp),
        "price_sulfur": _assign_attr("_sulfur_wap_price", bp),
        "price_electricity": _assign_attr("_electricity_wap_price", bp),
        "price_co2": _assign_attr("_co2_wap_price", bp),

        # Attributes associated with revenue
        "revenue_oil": _assign_attr("_oil_revenue", bp),
        "revenue_sulfur": _assign_attr("_sulfur_revenue", bp),
        "revenue_electricity": _assign_attr("_electricity_revenue", bp),
        "revenue_co2": _assign_attr("_co2_revenue", bp),

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
        "capital_expenditures_pre_tax": _assign_attr(
            "_oil_capital_expenditures_pre_tax", bp
        ),
        "intangible_expenditures_pre_tax": _assign_attr(
            "_oil_intangible_expenditures_pre_tax", bp
        ),
        "opex_expenditures_pre_tax": _assign_attr("_oil_opex_expenditures_pre_tax", bp),
        "asr_expenditures_pre_tax": _assign_attr("_oil_asr_expenditures_pre_tax", bp),
        "lbt_expenditures_pre_tax": _assign_attr("_oil_lbt_expenditures_pre_tax", bp),
        "cost_of_sales_expenditures_pre_tax": _assign_attr(
            "_oil_cost_of_sales_expenditures_pre_tax", bp
        ),

        # Attributes associated with indirect tax
        "capital_indirect_tax": _assign_attr("_oil_capital_indirect_tax", bp),
        "intangible_indirect_tax": _assign_attr("_oil_intangible_indirect_tax", bp),
        "opex_indirect_tax": _assign_attr("_oil_opex_indirect_tax", bp),
        "asr_indirect_tax": _assign_attr("_oil_asr_indirect_tax", bp),
        "lbt_indirect_tax": _assign_attr("_oil_lbt_indirect_tax", bp),
        "cost_of_sales_indirect_tax": _assign_attr("_oil_cost_of_sales_indirect_tax", bp),

        # Attributes associated with expenditures post tax
        "capital_expenditures_post_tax": _assign_attr(
            "_oil_capital_expenditures_post_tax", bp
        ),
        "intangible_expenditures_post_tax": _assign_attr(
            "_oil_intangible_expenditures_post_tax", bp
        ),
        "opex_expenditures_post_tax": _assign_attr(
            "_oil_opex_expenditures_post_tax", bp
        ),
        "asr_expenditures_post_tax": _assign_attr(
            "_oil_asr_expenditures_post_tax", bp
        ),
        "lbt_expenditures_post_tax": _assign_attr(
            "_oil_lbt_expenditures_post_tax", bp
        ),
        "cost_of_sales_expenditures_post_tax": _assign_attr(
            "_oil_cost_of_sales_expenditures_post_tax", bp
        ),

        # Attributes associated with expenses
        "expenses_capital": _assign_attr("_oil_capital", bp),
        "expenses_non_capital": _assign_attr("_oil_non_capital", bp),
        "expenses_total": _assign_attr("_oil_total_expenses", bp),

        # Attribute associated with cashflow
        "cashflow": _assign_attr("_oil_cashflow", bp),
    }

    # Convert OIL cashflow table into pandas DataFrame
    return pd.DataFrame(table_oil)


def get_table_baseproject_gas(contract) -> pd.DataFrame:

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

    # Specify cashflow table for GAS
    table_gas: dict = {
        # Attribute associated with project years
        "years": bp.project_years,

        # Attributes associated with lifting
        "lifting_gas": _assign_attr("_gas_lifting", bp, True),
        "lifting_sulfur": _assign_attr("_sulfur_lifting", bp, True),
        "lifting_electricity": _assign_attr(
            "_electricity_lifting", bp, True
        ),
        "lifting_co2": _assign_attr("_co2_lifting", bp, True),

        # Attributes associated with price
        "price_gas": _assign_attr("_gas_wap_price", bp),
        "price_sulfur": _assign_attr("_sulfur_wap_price", bp),
        "price_electricity": _assign_attr("_electricity_wap_price", bp),
        "price_co2": _assign_attr("_co2_wap_price", bp),

        # Attributes associated with revenue
        "revenue_gas": _assign_attr("_gas_revenue", bp),
        "revenue_sulfur": _assign_attr("_sulfur_revenue", bp),
        "revenue_electricity": _assign_attr("_electricity_revenue", bp),
        "revenue_co2": _assign_attr("_co2_revenue", bp),

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
        "capital_expenditures_pre_tax": _assign_attr(
            "_gas_capital_expenditures_pre_tax", bp
        ),
        "intangible_expenditures_pre_tax": _assign_attr(
            "_gas_intangible_expenditures_pre_tax", bp
        ),
        "opex_expenditures_pre_tax": _assign_attr("_gas_opex_expenditures_pre_tax", bp),
        "asr_expenditures_pre_tax": _assign_attr("_gas_asr_expenditures_pre_tax", bp),
        "lbt_expenditures_pre_tax": _assign_attr("_gas_lbt_expenditures_pre_tax", bp),
        "cost_of_sales_expenditures_pre_tax": _assign_attr(
            "_gas_cost_of_sales_expenditures_pre_tax", bp
        ),

        # Attributes associated with indirect tax
        "capital_indirect_tax": _assign_attr("_gas_capital_indirect_tax", bp),
        "intangible_indirect_tax": _assign_attr("_gas_intangible_indirect_tax", bp),
        "opex_indirect_tax": _assign_attr("_gas_opex_indirect_tax", bp),
        "asr_indirect_tax": _assign_attr("_gas_asr_indirect_tax", bp),
        "lbt_indirect_tax": _assign_attr("_gas_lbt_indirect_tax", bp),
        "cost_of_sales_indirect_tax": _assign_attr("_gas_cost_of_sales_indirect_tax", bp),

        # Attributes associated with postonstream costs
        "capital_postonstream": _assign_attr(
            "_gas_capital_expenditures_post_tax", bp
        ),
        "intangible_postonstream": _assign_attr(
            "_gas_intangible_expenditures_post_tax", bp
        ),
        "opex_postonstream": _assign_attr(
            "_gas_opex_expenditures_post_tax", bp
        ),
        "asr_postonstream": _assign_attr(
            "_gas_asr_expenditures_post_tax", bp
        ),
        "lbt_postonstream": _assign_attr(
            "_gas_lbt_expenditures_post_tax", bp
        ),
        "cost_of_sales_postonstream": _assign_attr(
            "_gas_cost_of_sales_expenditures_post_tax", bp
        ),

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
    pass


def get_table(
    contract: CostRecovery | GrossSplit | BaseProject | Transition,
) -> tuple:

    if isinstance(contract, CostRecovery):
        pass

    elif isinstance(contract, GrossSplit):
        pass

    elif isinstance(contract, Transition):
        pass

    elif isinstance(contract, BaseProject):
        psc_table_oil = get_table_baseproject_oil(contract=contract)
        psc_table_gas = get_table_baseproject_gas(contract=contract)

        print('\t')
        print(f'Filetype: {type(psc_table_gas)}')
        print(f'Shape: {psc_table_gas.shape}')
        print('psc_table_gas = \n', psc_table_gas)
