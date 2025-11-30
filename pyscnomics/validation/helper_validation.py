"""
A collection of procedures to prepare case 0
"""

import os
import json
from pyscnomics.econ.selection import ContractType
from pyscnomics.api.adapter import (
    get_costrecovery,
    get_grosssplit,
    get_baseproject,
)


def convert_to_json(
    case,
    contract_type,
    target_dir=(
        "E:/1009_My Journal_PSC Migas/26_20230707_PSCEconomic/"
        "pyscnomics/pyscnomics/dataset/"
    )
) -> None:
    """
    Convert a case class into JSON format and save it to a file.

    This function instantiates a case class using the specified
    ``contract_type``, converts the resulting object into a Python
    dictionary via its ``as_dict`` method, serializes it into a JSON
    string, and saves the JSON output to the specified directory.
    The JSON filename is generated dynamically based on the class name
    of the input ``case`` argument.

    Parameters
    ----------
    case : type
        A class (not an instance) representing a PSC economic case.
        The class must accept ``contract_type`` as its sole initialization
        argument and implement an ``as_dict`` method that returns a
        JSON-serializable dictionary.
    contract_type : str or Any
        The type of PSC contract used to instantiate ``case``.
        This value is passed directly to ``case(contract_type)``.
    target_dir : str, optional
        Path to the directory where the resulting JSON file will be saved.
        Must be an existing directory. Default points to the PySCnomics
        dataset folder.

    Returns
    -------
    None
        The function writes a JSON file to ``target_dir`` but does not
        return any value.

    Notes
    -----
    The output JSON filename is derived from the class name of ``case``.
    For example, if ``case.__name__`` is ``"PSC2021"``, the resulting
    file will be named ``"PSC2021.json"``.
    """

    # Create an instance of case based on the prescribed contract_type
    obj = case(contract_type)

    # Convert the case's instance into a dictionary
    data = obj.as_dict()

    # Convert the dictionary into a json string
    data_as_json = json.dumps(data, indent=4)

    # Save json data into a target directory
    filename = f"{case.__name__}.json"
    filepath = os.path.join(target_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(data_as_json)


def convert_to_dict(
    filename: str,
    target_dir=(
        "E:/1009_My Journal_PSC Migas/26_20230707_PSCEconomic/"
        "pyscnomics/pyscnomics/dataset/"
    )
) -> dict:
    """
    Load a JSON file and convert its contents into a Python dictionary.

    This function opens a JSON file located in the specified directory,
    parses its contents using :func:`json.load`, and returns the resulting
    Python dictionary. The function assumes that the JSON file exists in
    ``target_dir`` and that its contents are valid JSON.

    Parameters
    ----------
    filename : str
        Name of the JSON file to read (e.g., ``"PSC2021.json"``).
        This should be the base filename only, not a full path.
    target_dir : str, optional
        Path to the directory containing the JSON file.
        Must be an existing directory.
        Default points to the PySCnomics dataset folder.

    Returns
    -------
    dict
        A Python dictionary representing the JSON contents of the file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the constructed path.
    json.JSONDecodeError
        If the file is not valid JSON.
    OSError
        If an error occurs while opening the file.
    """

    filepath = os.path.join(target_dir, filename)

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def execute_contract(case, contract_type) -> dict:
    """
    Execute a PSC contract evaluation and return structured results.

    Parameters
    ----------
    case : dict or callable
        Input case configuration. If a callable (e.g., a case class),
        it will be instantiated as ``case(contract_type)`` and converted
        to a dictionary via ``as_dict()``. Otherwise, it must be a
        dictionary containing all required contract inputs.
    contract_type : ContractType
        The contract scheme to evaluate. Must be one of
        ``ContractType.COST_RECOVERY``, ``ContractType.GROSS_SPLIT``,
        or ``ContractType.BASE_PROJECT``.

    Returns
    -------
    dict
        A dictionary containing:
        - ``data`` : dict
          The input case data used for evaluation.
        - ``contract`` : dict
          Detailed contract-level results.
        - ``contract_arguments`` : dict
          Arguments used in contract computation.
        - ``summary_arguments`` : dict
          Arguments used for summary calculations.
        - ``summary`` : dict
          Summary-level economic outputs.

    Raises
    ------
    ValueError
        If `contract_type` is not a supported PSC scheme.
    """

    # Choose which engine to call based on "contract_type"
    engines = {
        ContractType.COST_RECOVERY: get_costrecovery,
        ContractType.GROSS_SPLIT: get_grosssplit,
        ContractType.BASE_PROJECT: get_baseproject
    }

    if contract_type not in engines:
        raise ValueError(f"Invalid contract type: {contract_type!r}")

    engine = engines[contract_type]

    # If "case" is provided as a class's instance
    if callable(case):
        obj = case(contract_type)
        data = obj.as_dict()

    # If "case" is provided as a dictionary
    else:
        data = case

    # Execute contract using the appropriate engine
    results = engine(data=data, summary_result=True)

    return {
        "data": data,
        "contract": results[1],
        "contract_arguments": results[2],
        "summary_arguments": results[3],
        "summary": results[0],
    }
