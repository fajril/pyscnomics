"""
Procedures to carry out sensitivity analysis
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
from pyscnomics.api.adapter import get_sensitivity, get_uncertainty, get_contract_optimization
from pyscnomics.dataset.case_00A import Case00A


if __name__ == "__main__":

    case = Case00A(contract_type=ContractType.GROSS_SPLIT)
    data = case.as_dict()

    # t1 = get_uncertainty(data=data, contract_type="Base Project")
    # t1 = get_sensitivity(data=data, contract_type="Base Project")
    t1 = get_contract_optimization(data=data, contract_type="Gross Split")
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
