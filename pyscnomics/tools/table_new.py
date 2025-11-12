"""
A collection of procedures to generate cashflow of a contract in the form of DataFrame
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
    return value.get_lifting_rate_arr() if is_lifting else value


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

        bp = contract

        postonstream_non_depreciable = np.sum(
            [
                _assign_attr(at, bp) for at in [
                    "_oil_intangible_expenditures_post_tax",
                    "_oil_opex_expenditures_post_tax",
                    "_oil_asr_expenditures_post_tax",
                    "_oil_lbt_expenditures_post_tax",
                    "_oil_cost_of_sales_expenditures_post_tax",
                ]
            ],
            axis=0
        )

        print('\t')
        print(f'Filetype: {type(postonstream_non_depreciable)}')
        print(f'Length: {len(postonstream_non_depreciable)}')
        print('postonstream_non_depreciable = \n', postonstream_non_depreciable)

        table_oil: dict = {
            # Attribute associated with project years
            "years": bp.project_years,

            # Attributes associated with lifting
            "lifting_oil": _assign_attr("_oil_lifting", bp, True),
            "lifting_sulfur": _assign_attr("_sulfur_lifting", bp, True),
            "lifting_electricity": _assign_attr("_electricity_lifting", bp, True),
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
            "sc_depreciable": _assign_attr("_oil_depreciable_sunk_cost", bp),
            "sc_non_depreciable": _assign_attr("_oil_non_depreciable_sunk_cost", bp),
            "sc": _assign_attr("_oil_sunk_cost", bp),

            # Attributes associated with preonstream cost
            "pre_os_depreciable": _assign_attr("_oil_depreciable_preonstream", bp),
            "pre_os_non_depreciable": _assign_attr("_oil_non_depreciable_preonstream", bp),
            "pre_os": _assign_attr("_oil_preonstream", bp),

            # Attributes associated with postomstream cost
            "post_os_depreciable": _assign_attr("_oil_capital_expenditures_post_tax", bp),
            "post_os_non_depreciable": postonstream_non_depreciable,
            "post_os": _assign_attr("_oil_total_expenditures_post_tax", bp),
        }

        print('\t')
        print(f'Filetype: {type(table_oil)}')
        print(f'Length: {len(table_oil)}')
        print('table_oil = \n', table_oil)


