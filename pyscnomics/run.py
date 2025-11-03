"""
Execute calculations
"""

from pyscnomics.poolData import (
    synthetic_data_class,
    get_contract_arguments_class,
    get_summary_arguments_class,
    synthetic_data_dict,
    optimization_arguments_dict,
)
from pyscnomics.dataset.poolData1 import PrepareLiftingCostsAsClass
from pyscnomics.econ.selection import (
    OptimizationParameter,
    OptimizationTarget,
)
from pyscnomics.optimize.optimization import (
    adjust_cost_element,
    adjust_contract,
    optimize_psc_core,
    optimize_psc,
)


if __name__ == "__main__":

    # cr, gs, bp = "cost_recovery", "gross_split", "base_project"
    #
    # # Specify contract type
    # contract = gs
    # contract_arguments = get_contract_arguments_class(contract)
    # summary_arguments = get_summary_arguments_class()
    # optimization_arguments = optimization_arguments_dict()
    #
    # # Generate synthetic data
    # contract_as_class = synthetic_data_class(contract)
    # contract_as_dict = synthetic_data_dict(contract)

    data = PrepareLiftingCostsAsClass()

    t1 = data.cost_of_sales

    # print('\t')
    # print(f'Filetype: {type()}')
    # print(f'Length: {len()}')
    # print()

    print('\t')
    print(f'Filetype: {type(t1)}')
    print(f'Length: {len(t1)}')
    print('t1 = \n', t1)
