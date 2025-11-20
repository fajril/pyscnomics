"""
A collection of unit testings for CASE 01
"""

import numpy as np
from pyscnomics.econ.selection import ContractType
from pyscnomics.dataset.case_01 import Case01
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
    "cls": Case01,
    "contract_type": ContractType.COST_RECOVERY,
    "run_as_dict": False,
}

# Run the contract using function "execute_contract()"
ctr = execute_contract(**kwargs_execute)

# Results of run in terms of "contract" and "summary"
contract = ctr["contract"]
summary = ctr["summary"]

# Generate OIL cashflow
cashflow_oil = get_table(contract=contract)[0]


def _get_attr(attr: str) -> np.ndarray:
    return cashflow_oil[attr].to_numpy()


def test_project_years():

    project_yrs = {
        "calculated": _get_attr(attr="years"),
        "expected": np.array(
            [
                2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032,
                2033, 2034, 2035, 2036, 2037, 2038, 2039,
            ]
        ),
    }

    # Execute testing
    np.testing.assert_allclose(project_yrs["calculated"], project_yrs["expected"])


def test_lifting():
    pass


def test_price():
    pass


def test_revenue():
    pass


def test_sulfur():
    pass


def test_electricity():
    pass


def test_co2():
    pass


