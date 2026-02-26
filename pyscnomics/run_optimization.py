"""
Procedures to carry out optimization study
"""
import numpy as np

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
    adjust_contract,
    optimize_psc_core,
    adjust_cost_element,
    adjust_useful_life_years,
    optimize_psc,
)
from pyscnomics.api.adapter import (
    get_costrecovery,
    get_grosssplit,
    get_baseproject,
    get_contract_table,
)
from pyscnomics.tools.table import get_table
from pyscnomics.api.adapter import get_contract_optimization
from pyscnomics.dataset.case_00A import Case00A
from pyscnomics.dataset.case_00B import Case00B


if __name__ == "__main__":

    case = Case00B(contract_type=ContractType.GROSS_SPLIT)
    contract = case.as_class()
    contract_arguments = case.contract_arguments
    summary_arguments = case.summary_arguments
    optimization_arguments = case.optimization_arguments

    # # Execute adjust contract
    # kwargs_adjust_contract = {
    #     "contract": contract,
    #     "contract_arguments": contract_arguments,
    #     "summary_argument": summary_arguments,
    #     "variable": OptimizationParameter.MINISTERIAL_DISCRETION,
    #     "value": 0.1,
    #     "target_parameter": "ctr_irr",
    # }
    #
    # adjust_contract(**kwargs_adjust_contract)

    # Execute optimize psc core
    kwargs_psc_core = {
        "contract": contract,
        "contract_arguments": contract_arguments,
        "summary_argument": summary_arguments,
        "target_optimization_value": 0.3,
        "dict_optimization": optimization_arguments,
        "target_parameter": OptimizationTarget.IRR,
    }

    t1 = optimize_psc_core(**kwargs_psc_core)
    print('\t')
    print(f'Filetype: {type(t1)}')
    print(f'Length: {len(t1)}')
    print('t1 = \n', t1[3])

    # Execute optimize_psc
    # kwargs_optim = {
    #     "contract": contract,
    #     "contract_arguments": contract_arguments,
    #     "summary_arguments": summary_arguments,
    #     "dict_optimization": optimization_arguments,
    #     "target_parameter": OptimizationTarget.IRR,
    #     "target_optimization_value": 0.15,
    # }
    #
    # optimize_psc(**kwargs_optim)

    # case = Case00A(contract_type=ctr_type)
    # data = case.as_dict()
    # t1 = get_contract_optimization(data=data, contract_type=ctr_type.value)
    # print('\t')
    # print(f'Filetype: {type(t1)}')
    # print(f'Length: {len(t1)}')
    # print('t1 = \n', t1)

    # print('\t')
    # print(f'Filetype: {type()}')
    # print(f'Length: {len()}')
    # print()

    # print('\t')
    # print(f'Filetype: {type(t1)}')
    # print(f'Length: {len(t1)}')
    # print('t1 = \n', t1)


