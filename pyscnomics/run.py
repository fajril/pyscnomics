"""
Execute calculations
"""

from pyscnomics.poolData import (
    get_contract_arguments_class,
    get_summary_arguments_class,
    synthetic_data_dict,
    optimization_arguments_dict,
)
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
)

from pyscnomics.dataset.case0 import Case0
from pyscnomics.dataset.case1 import Case1


if __name__ == "__main__":

    ctr = ContractType.BASE_PROJECT
    data = Case0(contract_type=ctr)

    # Run case as class's instance
    contract = data.as_class()
    contract_arguments = data.contract_arguments
    summary_arguments = data.summary_arguments

    contract.run(**contract_arguments)
    results = contract.get_summary(**summary_arguments)

    print('\t')
    print(f'Filetype: {type(results)}')
    print(f'Length: {len(results)}')
    print('results = \n', results)

    # Run case as dictionary
    # mapping_run_dict = {
    #     ContractType.COST_RECOVERY: get_costrecovery,
    #     ContractType.GROSS_SPLIT: get_grosssplit,
    #     ContractType.BASE_PROJECT: get_baseproject,
    # }
    #
    # run_as_dict = mapping_run_dict[ctr](data=data)

    # cr, gs, bp = "cost_recovery", "gross_split", "base_project"
    #
    # Specify contract type
    # contract = bp
    # contract_arguments = get_contract_arguments_class(contract)
    # summary_arguments = get_summary_arguments_class()
    # # optimization_arguments = optimization_arguments_dict()
    #
    # # Generate synthetic data
    # contract_as_class = synthetic_data_class(contract)
    # contract_as_dict = synthetic_data_dict(contract)

    # print('\t')
    # print(f'Filetype: {type()}')
    # print(f'Length: {len()}')
    # print()

    # print('\t')
    # print(f'Filetype: {type(t1)}')
    # print(f'Length: {len(t1)}')
    # print('t1 = \n', t1)
