"""
A collection of unit testings for CASE 04
"""

# import pytest
import numpy as np
from pyscnomics.econ.selection import ContractType
from pyscnomics.dataset.case_04 import Case04
from pyscnomics.validation.preparation import execute_contract
from pyscnomics.tools.table import get_table


# Specify cost types
cost_types = [
    "capital",
    "intangible",
    "opex",
    "asr",
    "lbt",
]

# Specify arguments to run function "execute_contract()"
kwargs_execute = {
    "cls": Case04,
    "contract_type": ContractType.GROSS_SPLIT,
    "run_as_dict": True,
}

# Run the contract using function "execute_contract()"
ctr = execute_contract(**kwargs_execute)


def test_project_years():

    t1 = ctr
    print('\t')
    print(f'Filetype: {type(t1)}')
    print(f'Length: {len(t1)}')
    print('t1 = \n', t1)

