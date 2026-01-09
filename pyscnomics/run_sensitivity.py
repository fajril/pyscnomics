"""
Procedures to carry out sensitivity analysis
"""

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
from pyscnomics.optimize.sensitivity import (
    _get_multipliers,
    _prepare_adjusted_parameters_single_contract
)
from pyscnomics.api.adapter import get_sensitivity
from pyscnomics.dataset.case_00A import Case00A


if __name__ == "__main__":

    case = Case00A(contract_type=ContractType.COST_RECOVERY)
    contract = case.as_class()

    _prepare_adjusted_parameters_single_contract(
        contract=contract,
        adjustment_value=0.5,
        element="GASLIFTING"
    )

    # t1 = contract
    # print('\t')
    # print(f'Filetype: {type(t1)}')
    # print(f'Length: {len(t1)}')
    # print('t1 = \n', t1)

    # ctr_type = ContractType.GROSS_SPLIT
    #
    # case = Case00A(contract_type=ctr_type)
    # data = case.as_dict()
    #
    # t1 = get_sensitivity(data=data, contract_type=ctr_type.value)
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
