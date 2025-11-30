"""
Execute calculations
"""

from pyscnomics.econ.selection import ContractType
from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.tools.table import get_table
from pyscnomics.dataset.case_00A import Case00A
from pyscnomics.dataset.case_01 import Case01
from pyscnomics.dataset.case_02 import Case02


if __name__ == "__main__":

    case = Case00A(contract_type=ContractType.BASE_PROJECT)

    contract = case.as_class()
    contract_arguments = case.contract_arguments

    contract.run(**contract_arguments)


    # print('\t')
    # print(f'Filetype: {type(contract)}')
    # print(f'Length: {len(contract)}')
    # print('contract = \n', contract)

    # print('\t')
    # print(f'Filetype: {type()}')
    # print(f'Length: {len()}')
    # print()

    # print('\t')
    # print(f'Filetype: {type(t1)}')
    # print(f'Length: {len(t1)}')
    # print('t1 = \n', t1)
