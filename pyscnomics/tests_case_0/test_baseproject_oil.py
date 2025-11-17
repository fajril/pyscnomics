"""
A collection of unit testings for CASE 0: Dummy
"""

import pytest
import numpy as np
from pyscnomics.econ.selection import ContractType
from pyscnomics.dataset.case0 import Case0
from pyscnomics.tests_case_0.preparation import execute_contract
from pyscnomics.tools.table import get_table


# Specify arguments to run function "execute_contract()"
kwargs_execute = {
    "cls": Case0,
    "contract_type": ContractType.BASE_PROJECT,
    "run_as_dict": True,
}

# Run the contract using function "execute_contract()"
ctr = execute_contract(**kwargs_execute)

# Results of run in terms of "data", "contract", and "summary"
data = ctr["data"]
contract = ctr["contract"]
summary = ctr["summary"]

# Generate OIL cashflow
cashflow_oil = get_table(contract=contract)[0]


def test_project_years():
    """
    Test that the project year sequence matches the expected values.

    Verifies
    --------
    The ``years`` column in ``cashflow_oil`` matches the expected 2023–2032 range.
    """

    # Calculated
    calculated = cashflow_oil["years"].to_numpy()

    # Expected
    expected = np.array([2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032])

    # Execute testing
    np.testing.assert_allclose(calculated, expected)


def test_lifting():
    """
    Test that the oil lifting volumes match expected values.

    Verifies
    --------
    The ``lifting`` column in ``cashflow_oil`` equals the expected
    array of zeros followed by three 100s.
    """

    # Calculated
    calculated = cashflow_oil["lifting"].to_numpy()

    # Expected
    expected = np.array([0, 0, 0, 0, 0, 0, 0, 100, 100, 100])

    np.testing.assert_allclose(calculated, expected)

def test_price():
    oass



