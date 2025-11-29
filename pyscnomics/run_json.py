"""
A collection of procedures to run a standard PSC contract from JSON data format
"""

import json
import importlib.resources as resources
import pandas as pd
from pyscnomics.econ.selection import (
    OptimizationParameter,
    OptimizationTarget,
    ContractType,
    VariableSplit082017,
    VariableSplit522017,
    VariableSplit132024,
)
from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.optimize.optimization import (
    adjust_cost_element,
    adjust_contract,
    optimize_psc_core,
    optimize_psc,
)
from pyscnomics.api.adapter import (
    get_costrecovery,
    get_grosssplit,
    get_baseproject,
    get_contract_table,
)
from pyscnomics.dataset.sample import read_cost_type
from pyscnomics.tools.table import get_table
from pyscnomics.dataset.case_00A import Case00A
from pyscnomics.dataset.case_01 import Case01
from pyscnomics.dataset.case_02 import Case02


def execute_json(case, contract_type):

    contract = case

    if contract_type == ContractType.COST_RECOVERY:
        cr = get_costrecovery(data=contract, summary_result=True)
        return {
            "data": contract,
            "contract": cr[1],
            "contract_arguments": cr[2],
            "summary_arguments": cr[3],
            "summary": cr[0],
        }

    elif contract_type == ContractType.GROSS_SPLIT:
        gs = get_grosssplit(data=contract, summary_result=True)
        return {
            "data": contract,
            "contract": gs[1],
            "contract_arguments": gs[2],
            "summary_arguments": gs[3],
            "summary": gs[0],
        }

    elif contract_type == ContractType.BASE_PROJECT:
        bp = get_baseproject(data=contract, summary_result=True)
        return {
            "data": contract,
            "contract": bp[1],
            "contract_arguments": bp[2],
            "summary_arguments": bp[3],
            "summary": bp[0],
        }

    else:
        raise ValueError(f"Invalid contract type: {contract_type!r}")


def load_json(target_json: str):

    with resources.open_text("pyscnomics.dataset", target_json) as f:
        data = json.load(f)

    return data


if __name__ == "__main__":

    case = load_json(target_json="duri_field.json")

    kwargs_execute = {
        "case": case,
        "contract_type": ContractType.GROSS_SPLIT,
    }

    ctr = execute_json(**kwargs_execute)

    # Configure results
    data: dict = ctr["data"]
    contract: CostRecovery | GrossSplit | BaseProject = ctr["contract"]
    contract_arguments: dict = ctr["contract_arguments"]
    summary_arguments: dict = ctr["summary_arguments"]
    summary: dict = ctr["summary"]

    # Configure cashflow table
    cshflow: tuple = get_table(contract=contract)
    cashflow_table: dict = {
        "oil": cshflow[0],
        "gas": cshflow[1],
        "consolidated": cshflow[2],
    }

    t1 = cashflow_table["oil"]
    print('\t')
    print(f'Filetype: {type(t1)}')
    print(f'Length: {len(t1)}')
    print('t1 = \n', t1)

    # print('\t')
    # print(f'Filetype: {type()}')
    # print(f'Length: {len()}')
    # print()

    # print('\t')
    # print(f'Filetype: {type(t1)}')
    # print(f'Length: {len(t1)}')
    # print('t1 = \n', t1)
