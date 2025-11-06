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
from pyscnomics.dataset.case1 import Case1


if __name__ == "__main__":

    ctr = ContractType.BASE_PROJECT

    data = Case1(contract_type=ctr)
    t1 = data.capital

    print('\t')
    print(f'Filetype: {type(t1["class"])}')
    print(f'Length: {len(t1["class"])}')
    print('t1["class"] = \n', t1["class"])

    print('\t')
    print(f'Filetype: {type(t1["dict"])}')
    print(f'Length: {len(t1["dict"])}')
    print('t1["dict] = \n', t1["dict"])

    # cr, gs, bp = "cost_recovery", "gross_split", "base_project"
    #
    # # Specify contract type
    # contract = bp
    # contract_arguments = get_contract_arguments_class(contract)
    # summary_arguments = get_summary_arguments_class()
    # # optimization_arguments = optimization_arguments_dict()
    #
    # # Generate synthetic data
    # contract_as_class = synthetic_data_class(contract)
    # contract_as_dict = synthetic_data_dict(contract)
    #
    # data = PrepareLiftingCostsAsClass()
    #
    # t1 = data.asr_cost

    # print('\t')
    # print(f'Filetype: {type()}')
    # print(f'Length: {len()}')
    # print()

    # print('\t')
    # print(f'Filetype: {type(t1)}')
    # print(f'Length: {len(t1)}')
    # print('t1 = \n', t1)
