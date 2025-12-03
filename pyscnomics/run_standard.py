"""
A collection of procedures to run standard PSC contract
"""

import os
import json
import importlib.resources as resources

from pyscnomics.econ.selection import ContractType
from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.api.adapter import (
    get_costrecovery,
    get_grosssplit,
    get_baseproject,
)
from pyscnomics.validation.helper_validation import (
    convert_to_json,
    convert_to_dict,
    execute_contract,
)

from pyscnomics.tools.table import get_table
from pyscnomics.dataset.case_00A import Case00A
from pyscnomics.dataset.case_01 import Case01
from pyscnomics.dataset.case_02 import Case02


if __name__ == "__main__":

    # Specify arguments to run function "execute_contract()"
    kwargs_execute = {
        "case": Case01,
        "contract_type": ContractType.COST_RECOVERY,
    }

    # Run the contract using function "execute_contract()"
    ctr = execute_contract(**kwargs_execute)

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
