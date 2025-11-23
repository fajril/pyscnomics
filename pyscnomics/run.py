"""
Execute calculations
"""

from pyscnomics.econ.selection import (
    OptimizationParameter,
    OptimizationTarget,
    ContractType,
)
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
from pyscnomics.tools.table import get_table
from pyscnomics.dataset.case_00 import Case00
from pyscnomics.dataset.case_01 import Case01
from pyscnomics.dataset.case_02 import Case02
from pyscnomics.dataset.case_03 import Case03


def execute_contract(cls, contract_type, run_as_dict):
    """
    Execute a contract simulation either as a dictionary or as a class instance.

    Parameters
    ----------
    cls : type
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

    data = cls(contract_type)

    # Run contract as dictionary
    if run_as_dict:
        # Prepare contract, contract_arguments, and summary_arguments
        contract = data.as_dict()
        # contract_arguments = data.contract_arguments
        # summary_arguments = data.summary_arguments

        # Execute the contract and return the results in terms of a dictionary
        if contract_type == ContractType.COST_RECOVERY:
            cr = get_costrecovery(data=contract, summary_result=True)
            return {"contract": contract, "summary": cr[0]}

        elif contract_type == ContractType.GROSS_SPLIT:
            gs = get_grosssplit(data=contract, summary_result=True)
            return {"contract": contract, "summary": gs[0]}

        elif contract_type == ContractType.BASE_PROJECT:
            bp = get_baseproject(data=contract, summary_result=True)
            return {"contract": contract, "summary": bp[0]}

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
            "summary": contract.get_summary(**summary_arguments)
        }


if __name__ == "__main__":

    data = Case03()

    t1 = data.as_dict()
    print('\t')
    print(f'Filetype: {type(t1)}')
    print(f'Length: {len(t1)}')
    print('t1 = \n', t1)

    # # Specify arguments to run "execute_contract()"
    # kwargs_execute = {
    #     "cls": Case1,
    #     "contract_type": ContractType.COST_RECOVERY,
    #     "run_as_dict": False,
    # }
    #
    # # Results in terms of "contract" and "summary"
    # ctr = execute_contract(**kwargs_execute)
    # contract = ctr["contract"]
    # summary = ctr["summary"]

    # get_table(contract=contract)

    # # Run case as class's instance
    # contract = data.as_class()
    # contract_arguments = data.contract_arguments
    # summary_arguments = data.summary_arguments
    #
    # contract.run(**contract_arguments)
    # results = contract.get_summary(**summary_arguments)

    # print('\t')
    # print(f'Filetype: {type()}')
    # print(f'Length: {len()}')
    # print()

    # print('\t')
    # print(f'Filetype: {type(t1)}')
    # print(f'Length: {len(t1)}')
    # print('t1 = \n', t1)
