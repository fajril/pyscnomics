"""
Procedures to carry out uncertainty analysis
"""

import pandas as pd
from pyscnomics.econ.selection import (
    OptimizationParameter,
    OptimizationTarget,
    ContractType,
    VariableSplit082017,
    VariableSplit522017,
    VariableSplit132024,
    UncertaintyDistribution,
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
# from pyscnomics.api.adapter import (
#     get_costrecovery,
#     get_grosssplit,
#     get_baseproject,
#     get_contract_table,
# )
from pyscnomics.optimize.uncertainty import (
    get_setup_dict,
    get_summary_dict,
    ProcessMonte,
    uncertainty_psc,
)
from pyscnomics.tools.table import get_table
from pyscnomics.api.adapter import get_uncertainty
from pyscnomics.dataset.case_00A import Case00A
from pyscnomics.dataset.case_00B import Case00B
from pyscnomics.io.getattr import get_contract_attributes


if __name__ == "__main__":

    case = Case00B(contract_type=ContractType.COST_RECOVERY)
    contract = case.as_class()
    contract_arguments = case.contract_arguments
    summary_arguments = case.summary_arguments

    kwargs_uncertainty = {
        "contract": contract,
        "contract_arguments": contract_arguments,
        "summary_arguments": summary_arguments,
        "run_number": 5,
        "oil_price_stddev": 1,
        "gas_price_stddev": 1,
        "opex_stddev": 1,
        "capex_stddev": 1,
        "lifting_stddev": 1,
        "oil_price_distribution": UncertaintyDistribution.UNIFORM,
        "gas_price_distribution": UncertaintyDistribution.UNIFORM,
        "opex_distribution": UncertaintyDistribution.UNIFORM,
        "capex_distribution": UncertaintyDistribution.UNIFORM,
        "lifting_distribution": UncertaintyDistribution.UNIFORM,
    }

    uncertainty_psc(**kwargs_uncertainty)

    # data = case.as_dict()
    # get_summary_dict(data=data)

    # t1 = get_uncertainty(data=data, contract_type=ctr_type.value)
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
