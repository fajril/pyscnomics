"""
Execute calculations
"""

import numpy as np

from pyscnomics.econ.costs import CapitalCost
from pyscnomics.econ.selection import ContractType, FluidType, CostType
from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.tools.table import get_table

from pyscnomics.dataset.case_00A import Case00A
from pyscnomics.dataset.case_01 import Case01
from pyscnomics.dataset.case_02 import Case02


kwargs_capital_oil = {
    "start_year": 2023,
    "end_year": 2032,
    "expense_year": np.array(
        [
            2023,
            2024,
            2025,
            2026,
            2027,
            2028,
            2029,
            2030,
            2031,
            2032,
        ]
    ),
    "cost": np.array(
        [
            200,
            200,
            200,
            200,
            50,
            50,
            100,
            100,
            100,
            100,
        ]
    ),
    "cost_allocation": [
        FluidType.OIL,
        FluidType.OIL,
        FluidType.OIL,
        FluidType.OIL,
        FluidType.OIL,
        FluidType.OIL,
        FluidType.OIL,
        FluidType.OIL,
        FluidType.OIL,
        FluidType.OIL,
    ],
    "cost_type": [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        np.nan,
    ],
}


if __name__ == "__main__":

    case = Case00A(contract_type=ContractType.GROSS_SPLIT)

    contract = case.as_class()
    contract_arguments = case.contract_arguments
    summary_arguments = case.summary_arguments
    contract.run(**contract_arguments)

    print('\t')
    print(f'Filetype: {type(contract.warning_messages)}')
    print(f'Length: {len(contract.warning_messages)}')
    print('warning_messages = \n', contract.warning_messages)

    # print('\t')
    # print(f'Filetype: {type()}')
    # print(f'Length: {len()}')
    # print()

    # print('\t')
    #     # print(f'Filetype: {type(t1)}')
    #     # print(f'Length: {len(t1)}')
    #     # print(t1)
