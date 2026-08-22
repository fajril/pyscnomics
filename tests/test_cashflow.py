from datetime import date

import numpy as np

from pyscnomics.contracts.project import BaseProject
from pyscnomics.econ.cashflow import Cashflow


def test_multiple_cashflow_aligns_contract_years() -> None:
    first = BaseProject(
        start_date=date(2020, 1, 1),
        end_date=date(2022, 12, 31),
        approval_year=2020,
    )
    second = BaseProject(
        start_date=date(2021, 1, 1),
        end_date=date(2023, 12, 31),
        approval_year=2021,
    )
    object.__setattr__(first, "_consolidated_cashflow", np.array([1.0, 2.0, 3.0]))
    object.__setattr__(second, "_consolidated_cashflow", np.array([10.0, 20.0, 30.0]))

    cashflow = Cashflow(contract=(first, second))
    cashflow.get_cashflow_all()

    np.testing.assert_array_equal(cashflow.project_years, np.array([2020, 2021, 2022]))
    assert cashflow.cashflow is not None
    np.testing.assert_array_equal(cashflow.cashflow, np.array([1.0, 12.0, 23.0]))
