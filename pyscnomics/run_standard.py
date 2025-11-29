"""
A collection of procedures to run standard PSC contract
"""

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
from pyscnomics.api.adapter import (
    get_costrecovery,
    get_grosssplit,
    get_baseproject,
    get_contract_table,
)

from pyscnomics.tools.table import get_table
from pyscnomics.dataset.case_00A import Case00A
from pyscnomics.dataset.case_01 import Case01
from pyscnomics.dataset.case_02 import Case02


def execute_contract(case, contract_type, run_as_dict):
    """
    Execute a contract simulation either as a dictionary or as a class instance.

    Parameters
    ----------
    case : type
        The class constructor used to initialize the contract object.
    contract_type : ContractType
        Type of the contract to execute, e.g., COST_RECOVERY, GROSS_SPLIT, or BASE_PROJECT.
    run_as_dict : bool
        If True, execute the contract using dictionary-based inputs;
        otherwise, execute using the class instance.

    Returns
    -------
    dict
        Summary results of the contract execution, including key performance indicators.

    Raises
    ------
    ValueError
        If an invalid contract type is provided.
    """

    data = case(contract_type)

    # Run contract as dictionary
    if run_as_dict:
        # Prepare contract, contract_arguments, and summary_arguments
        contract = data.as_dict()
        # contract_arguments = data.contract_arguments
        # summary_arguments = data.summary_arguments

        # Execute the contract and return the results in terms of a dictionary
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

    # Run contract as instance
    else:
        # Prepare contract, contract_arguments, and summary_arguments
        contract = data.as_class()
        contract_arguments = data.contract_arguments
        summary_arguments = data.summary_arguments

        # Run the contract
        contract.run(**contract_arguments)

        # Return the results in terms of summary
        return {
            "contract": contract,
            "contract_arguments": contract_arguments,
            "summary_arguments": summary_arguments,
            "summary": contract.get_summary(**summary_arguments),
        }


if __name__ == "__main__":

    # Specify arguments to run function "execute_contract()"
    kwargs_execute = {
        "case": Case02,
        "contract_type": ContractType.GROSS_SPLIT,
        "run_as_dict": True,
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
