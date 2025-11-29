"""
A collection of procedures to run standard PSC contract
"""

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

from pyscnomics.tools.table import get_table
from pyscnomics.dataset.case_00A import Case00A
from pyscnomics.dataset.case_01 import Case01
from pyscnomics.dataset.case_02 import Case02


def execute(case, contract_type, run_as_dict=True):
    """
    Execute an economic evaluation case under a specified contract type.

    Parameters
    ----------
    case : callable or dict
        A case definition. If callable, it must return an object initialized
        with ``contract_type`` and providing ``as_dict()``, ``as_class()``,
        ``contract_arguments``, and ``summary_arguments``. If a dict, it is
        treated as raw input data.
    contract_type : ContractType
        Contract model to evaluate (e.g., COST_RECOVERY, GROSS_SPLIT, BASE_PROJECT).
    run_as_dict : bool, default=True
        If True, evaluate using dict-based engine functions. If False,
        instantiate the case class and execute through its API.

    Returns
    -------
    dict
        A dictionary containing evaluation results. Keys include:
        ``data`` (dict input), ``contract`` (contract engine or instance),
        ``contract_arguments`` (dict), ``summary_arguments`` (dict),
        and ``summary`` (evaluation summary).

    Raises
    ------
    ValueError
        If ``contract_type`` is not supported.

    Notes
    -----
    The function dispatches to the appropriate engine based on contract type
    and supports both dict-based and class-based execution flows.
    """

    # Choose which engine to call based on "contract_type"
    engines = {
        ContractType.COST_RECOVERY: get_costrecovery,
        ContractType.GROSS_SPLIT: get_grosssplit,
        ContractType.BASE_PROJECT: get_baseproject,
    }

    if contract_type not in engines:
        raise ValueError(f"Invalid contract type: {contract_type!r}")

    engine = engines[contract_type]

    # Run as dictionary
    if run_as_dict:
        # If case is provided as a class/instance
        if callable(case):
            obj = case(contract_type)
            data = obj.as_dict()

        # If case is provided as a JSON data
        else:
            data = case

        results = engine(data=data, summary_result=True)

        return {
            "data": data,
            "contract": results[1],
            "contract_arguments": results[2],
            "summary_arguments": results[3],
            "summary": results[0],
        }

    # Run as a class's instance
    obj = case(contract_type)
    contract = obj.as_class()
    contract_arguments = obj.contract_arguments
    summary_arguments = obj.summary_arguments

    contract.run(**contract_arguments)

    return {
        "contract": contract,
        "contract_arguments": contract_arguments,
        "summary_arguments": summary_arguments,
        "summary": contract.get_summary(**summary_arguments),
    }


def load_json(target_json: str):
    """
    Load a JSON file from the packaged dataset.

    Parameters
    ----------
    target_json : str
        Name of the JSON file to load from the ``pyscnomics.dataset`` package
        resources.

    Returns
    -------
    dict or list
        Parsed JSON content.

    Notes
    -----
    This function reads JSON data bundled within the package using
    ``importlib.resources``.
    """

    with resources.open_text("pyscnomics.dataset", target_json) as f:
        data = json.load(f)

    return data


if __name__ == "__main__":

    # Specify arguments to run function "execute()"
    kwargs_execute = {
        "case": Case00A,
        "contract_type": ContractType.BASE_PROJECT,
        "run_as_dict": True,
    }

    # Run the contract using function "execute()"
    ctr = execute(**kwargs_execute)

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

    # t1 = contract_arguments
    # print('\t')
    # print(f'Filetype: {type(t1)}')
    # print(f'Length: {len(t1)}')
    # print('t1 = \n', t1)

    t1 = data
    print('\t')
    print(f'Filetype: {type(t1["lifting"])}')
    print(f'Length: {len(t1["lifting"])}')
    print('t1 = \n', t1["lifting"])

    # print('\t')
    # print(f'Filetype: {type()}')
    # print(f'Length: {len()}')
    # print()

    # print('\t')
    # print(f'Filetype: {type(t1)}')
    # print(f'Length: {len(t1)}')
    # print('t1 = \n', t1)
