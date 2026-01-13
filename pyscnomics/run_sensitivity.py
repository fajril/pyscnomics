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
    _prepare_adjusted_parameters_single_contract,
    _adjust_element_single_contract,
    sensitivity_psc,
)
from pyscnomics.api.adapter import get_sensitivity
from pyscnomics.dataset.case_00A import Case00A


if __name__ == "__main__":

    # case = Case00A(contract_type=ContractType.BASE_PROJECT)
    # contract = case.as_class()
    # contract_arguments = case.contract_arguments
    # summary_arguments = case.summary_arguments
    #
    # elements = {
    #     0: "CAPEX",
    #     1: "OPEX",
    #     2: "OILPRICE",
    #     3: "GASPRICE",
    #     4: "OILLIFTING",
    #     5: "GASLIFTING",
    # }
    #
    # kwargs_sensitivity = {
    #     "contract": contract,
    #     "contract_arguments": contract_arguments,
    #     "summary_arguments": summary_arguments,
    #     "min_deviation": 0.5,
    #     "max_deviation": 0.5,
    #     "dataframe_output": True,
    # }
    #
    # t1 = sensitivity_psc(**kwargs_sensitivity)
    # print('\t')
    # print(f'Filetype: {type(t1)}')
    # print(f'Length: {len(t1)}')
    # print('t1 = \n', t1)

    ctr_type = ContractType.GROSS_SPLIT

    case = Case00A(contract_type=ctr_type)
    data = case.as_dict()

    t1 = get_sensitivity(data=data, contract_type=ctr_type.value)

    # print('\t')
    # print(f'Filetype: {type()}')
    # print(f'Length: {len()}')
    # print()

    print('\t')
    print(f'Filetype: {type(t1)}')
    print(f'Length: {len(t1)}')
    print('t1 = \n', t1)
