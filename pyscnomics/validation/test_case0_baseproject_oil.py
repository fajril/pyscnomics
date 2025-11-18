"""
A collection of unit testings for CASE 0: Dummy
"""

import pytest
import numpy as np
from pyscnomics.econ.selection import ContractType
from pyscnomics.dataset.case0 import Case0
from pyscnomics.validation.preparation import execute_contract
from pyscnomics.tools.table import get_table


# Specify cost types
cost_types = [
    "capital",
    "intangible",
    "opex",
    "asr",
    "lbt",
    "cost_of_sales",
]

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


def _calc_attr(attr: str) -> np.ndarray:
    return cashflow_oil[attr].to_numpy()


def test_project_years():
    """
    Validate that project years match the expected sequence.

    Verifies
    --------
    ``_calc_attr("years")`` produces the range 2023–2032.
    """

    project_yrs = {
        "calculated": _calc_attr(attr="years"),
        "expected": np.array([2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032]),
    }

    # Execute testing
    np.testing.assert_allclose(project_yrs["calculated"], project_yrs["expected"])


def test_lifting():
    """
    Validate that oil lifting volumes match expected values.

    Verifies
    --------
    ``_calc_attr("lifting")`` returns seven zeros followed by three 100s.
    """

    lft = {
        "calculated": _calc_attr(attr="lifting"),
        "expected": np.array([0, 0, 0, 0, 0, 0, 0, 100, 100, 100]),
    }

    # Execute testing
    np.testing.assert_allclose(lft["calculated"], lft["expected"])


def test_price():
    """
    Validate that oil prices match expected values.

    Verifies
    --------
    ``_calc_attr("price")`` returns seven zeros followed by three 120s.
    """

    price = {
        "calculated": _calc_attr("price"),
        "expected": np.array([0, 0, 0, 0, 0, 0, 0, 120, 120, 120])
    }

    np.testing.assert_allclose(price["calculated"], price["expected"])


def test_revenue():
    """
    Validate that oil revenues match expected values.

    Verifies
    --------
    ``_calc_attr("revenue")`` returns seven zeros followed by three 12,000s.
    """

    revs = {
        "calculated": _calc_attr(attr="revenue"),
        "expected": np.array([0, 0, 0, 0, 0, 0, 0, 12_000, 12_000, 12_000]),
    }

    np.testing.assert_allclose(revs["calculated"], revs["expected"])


def test_sulfur():
    """
    Validate sulfur-related lifting, price, and revenue values.

    Verifies
    --------
    - ``_calc_attr("lifting_sulfur")`` matches the expected lifting array.
    - ``_calc_attr("price_sulfur")`` matches the expected price array.
    - ``_calc_attr("revenue_sulfur")`` matches the expected revenue array.
    """

    sulfur = {
        "calculated": {
            "lifting": _calc_attr(attr="lifting_sulfur"),
            "price": _calc_attr(attr="price_sulfur"),
            "revenue": _calc_attr(attr="revenue_sulfur"),
        },
        "expected": {
            "lifting": np.array([0, 0, 0, 0, 0, 0, 10, 0, 10, 10]),
            "price": np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1]),
            "revenue": np.array([0, 0, 0, 0, 0, 0, 10, 0, 10, 10]),
        }
    }

    calc = sulfur["calculated"]
    expected = sulfur["expected"]

    # Execute testings
    np.testing.assert_allclose(calc["lifting"], expected["lifting"])
    np.testing.assert_allclose(calc["price"], expected["price"])
    np.testing.assert_allclose(calc["revenue"], expected["revenue"])


def test_electricity():
    """
    Validate electricity-related lifting, price, and revenue values.

    Verifies
    --------
    - ``_calc_attr("lifting_electricity")`` matches the expected lifting array.
    - ``_calc_attr("price_electricity")`` matches the expected price array.
    - ``_calc_attr("revenue_electricity")`` matches the expected revenue array.
    """

    elec = {
        "calculated": {
            "lifting": _calc_attr(attr="lifting_electricity"),
            "price": _calc_attr(attr="price_electricity"),
            "revenue": _calc_attr(attr="revenue_electricity"),
        },
        "expected": {
            "lifting": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "price": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "revenue": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        },
    }

    calc = elec["calculated"]
    expected = elec["expected"]

    # Execute testings
    np.testing.assert_allclose(calc["lifting"], expected["lifting"])
    np.testing.assert_allclose(calc["price"], expected["price"])
    np.testing.assert_allclose(calc["revenue"], expected["revenue"])


def test_co2():
    """
    Validate co2-related lifting, price, and revenue values.

    Verifies
    --------
    - ``_calc_attr("lifting_co2")`` matches the expected lifting array.
    - ``_calc_attr("price_co2")`` matches the expected price array.
    - ``_calc_attr("revenue_co2")`` matches the expected revenue array.
    """
    co2 = {
        "calculated": {
            "lifting": _calc_attr(attr="lifting_co2"),
            "price": _calc_attr(attr="price_co2"),
            "revenue": _calc_attr(attr="revenue_co2"),
        },
        "expected": {
            "lifting": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "price": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "revenue": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        },
    }

    calc = co2["calculated"]
    expected = co2["expected"]

    # Execute testings
    np.testing.assert_allclose(calc["lifting"], expected["lifting"])
    np.testing.assert_allclose(calc["price"], expected["price"])
    np.testing.assert_allclose(calc["revenue"], expected["revenue"])


def test_sunk_cost():
    """
    Validate depreciable, non-depreciable, and total sunk costs.

    Verifies
    --------
    -   ``_calc_attr("sunk_cost_depreciable")`` matches the expected depreciable array.
    -   ``_calc_attr("sunk_cost_non_depreciable")`` matches the expected
        non-depreciable array.
    -   ``_calc_attr("sunk_cost")`` matches the expected total array.
    """

    sunk_cost = {
        # Calculated attributes
        "calculated": {
            "depreciable": _calc_attr(attr="sunk_cost_depreciable"),
            "non_depreciable": _calc_attr(attr="sunk_cost_non_depreciable"),
            "total": _calc_attr(attr="sunk_cost"),
        },

        # Expected attributes
        "expected": {
            "depreciable": np.array([200, 200, 200, 200, 50, 50, 0, 0, 0, 0]),
            "non_depreciable": np.array([1000, 1000, 1000, 1000, 250, 250, 0, 0, 0, 0]),
            "total": np.array([1200, 1200, 1200, 1200, 300, 300, 0, 0, 0, 0]),
        },
    }

    calc = sunk_cost["calculated"]
    expected = sunk_cost["expected"]

    # Execute testings
    np.testing.assert_allclose(calc["depreciable"], expected["depreciable"])
    np.testing.assert_allclose(calc["non_depreciable"], expected["non_depreciable"])
    np.testing.assert_allclose(calc["total"], expected["total"])


def test_preonstream_cost():
    """
    Validate depreciable, non-depreciable, and total pre-onstream costs.

    Verifies
    --------
    - ``_calc_attr("preonstream_depreciable")`` matches the expected zero array.
    - ``_calc_attr("preonstream_non_depreciable")`` matches the expected zero array.
    - ``_calc_attr("preonstream")`` matches the expected total zero array.
    """

    preonstream = {
        # Calculated attributes
        "calculated": {
            "depreciable": _calc_attr(attr="preonstream_depreciable"),
            "non_depreciable": _calc_attr(attr="preonstream_non_depreciable"),
            "total": _calc_attr(attr="preonstream"),
        },

        # Expected attributes
        "expected": {
            "depreciable": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "non_depreciable": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "total": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        },
    }

    calc = preonstream["calculated"]
    expected = preonstream["expected"]

    # Execute testings
    np.testing.assert_allclose(calc["depreciable"], expected["depreciable"])
    np.testing.assert_allclose(calc["non_depreciable"], expected["non_depreciable"])
    np.testing.assert_allclose(calc["total"], expected["total"])


def test_postonstream_cost():
    """
    Test the correctness of postonstream cost calculations.

    Verifies that calculated depreciable, non-depreciable, and total
    postonstream costs match their expected NumPy arrays using
    ``np.testing.assert_allclose``.
    """

    postonstream = {
        "calculated": {
            "depreciable": _calc_attr(attr="postonstream_depreciable"),
            "non_depreciable": _calc_attr(attr="postonstream_non_depreciable"),
            "total": _calc_attr(attr="postonstream"),
        },
        "expected": {
            "depreciable": np.array([0, 0, 0, 0, 0, 0, 100, 100, 100, 100]),
            "non_depreciable": np.array([0, 0, 0, 0, 0, 0, 500, 500, 500, 500]),
            "total": np.array([0, 0, 0, 0, 0, 0, 600, 600, 600, 600]),
        },
    }

    calc = postonstream["calculated"]
    expected = postonstream["expected"]

    # Execute testings
    np.testing.assert_allclose(calc["depreciable"], expected["depreciable"])
    np.testing.assert_allclose(calc["non_depreciable"], expected["non_depreciable"])
    np.testing.assert_allclose(calc["total"], expected["total"])


def test_expenditures_pre_tax():
    """
    Test the correctness of pre-tax expenditure calculations.

    Ensures that calculated pre-tax expenditures for all cost types
    (capital, intangible, opex, asr, lbt, cost_of_sales) match their
    expected NumPy arrays using ``np.testing.assert_allclose``.
    """

    pre_tax = {
        "calculated": {
            ct: _calc_attr(attr=f"{ct}_expenditures_pre_tax") for ct in cost_types
        },
        "expected": {
            ct: np.array([0, 0, 0, 0, 0, 0, 100, 100, 100, 100]) for ct in cost_types
        },
    }

    calc = pre_tax["calculated"]
    expected = pre_tax["expected"]

    # Execute testings
    np.testing.assert_allclose(calc["capital"], expected["capital"])
    np.testing.assert_allclose(calc["intangible"], expected["intangible"])
    np.testing.assert_allclose(calc["opex"], expected["opex"])
    np.testing.assert_allclose(calc["asr"], expected["asr"])
    np.testing.assert_allclose(calc["lbt"], expected["lbt"])
    np.testing.assert_allclose(calc["cost_of_sales"], expected["cost_of_sales"])


def test_indirect_tax():
    """
    Test the correctness of indirect tax calculations.

    Verifies that calculated indirect tax values for all cost types
    (capital, intangible, opex, asr, lbt, cost_of_sales) match the
    expected zero-valued NumPy arrays using ``np.testing.assert_allclose``.

    Notes
    -----
    - ``_calc_attr`` retrieves each ``<cost_type>_indirect_tax`` attribute.
    - Expected arrays are all zeros, indicating no indirect tax applied.
    """

    indirect_tax = {
        "calculated": {ct: _calc_attr(attr=f"{ct}_indirect_tax") for ct in cost_types},
        "expected": {ct: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) for ct in cost_types},
    }

    calc = indirect_tax["calculated"]
    expected = indirect_tax["expected"]

    # Execute testings
    np.testing.assert_allclose(calc["capital"], expected["capital"])
    np.testing.assert_allclose(calc["intangible"], expected["intangible"])
    np.testing.assert_allclose(calc["opex"], expected["opex"])
    np.testing.assert_allclose(calc["asr"], expected["asr"])
    np.testing.assert_allclose(calc["lbt"], expected["lbt"])
    np.testing.assert_allclose(calc["cost_of_sales"], expected["cost_of_sales"])


def test_expeditures_post_tax():
    """
    Test the correctness of post-tax expenditure calculations.

    Verifies that calculated post-tax expenditures for all cost types
    (capital, intangible, opex, asr, lbt, cost_of_sales) match the
    expected NumPy arrays using ``np.testing.assert_allclose``.

    Notes
    -----
    - ``_calc_attr`` retrieves each ``<cost_type>_postonstream`` attribute.
    - Expected arrays contain a uniform post-tax expenditure profile.
    """

    post_tax = {
        "calculated": {ct: _calc_attr(attr=f"{ct}_postonstream") for ct in cost_types},
        "expected": {
            ct: np.array([0, 0, 0, 0, 0, 0, 100, 100, 100, 100]) for ct in cost_types
        },
    }

    calc = post_tax["calculated"]
    expected = post_tax["expected"]

    # Execute testings
    np.testing.assert_allclose(calc["capital"], expected["capital"])
    np.testing.assert_allclose(calc["intangible"], expected["intangible"])
    np.testing.assert_allclose(calc["opex"], expected["opex"])
    np.testing.assert_allclose(calc["asr"], expected["asr"])
    np.testing.assert_allclose(calc["lbt"], expected["lbt"])
    np.testing.assert_allclose(calc["cost_of_sales"], expected["cost_of_sales"])


def test_expenses():
    """
    Test the correctness of capital, non-capital, and total expense calculations.

    Verifies that calculated expense arrays match their expected values for
    capital, non-capital, and total expenses using ``np.testing.assert_allclose``.

    Notes
    -----
    - ``_calc_attr`` retrieves ``expenses_capital``, ``expenses_non_capital``,
      and ``expenses_total``.
    - Expected arrays encode the benchmark expense profiles.
    """

    expenses = {
        "calculated": {
            "capital": _calc_attr(attr="expenses_capital"),
            "non_capital": _calc_attr(attr="expenses_non_capital"),
            "total": _calc_attr(attr="expenses_total"),
        },
        "expected": {
            "capital": np.array([0, 0, 0, 0, 0, 0, 100, 100, 100, 100]),
            "non_capital": np.array([0, 0, 0, 0, 0, 0, 500, 500, 500, 500]),
            "total": np.array([0, 0, 0, 0, 0, 0, 600, 600, 600, 600]),
        },
    }

    calc = expenses["calculated"]
    expected = expenses["expected"]

    # Execute testings
    np.testing.assert_allclose(calc["capital"], expected["capital"])
    np.testing.assert_allclose(calc["non_capital"], expected["non_capital"])
    np.testing.assert_allclose(calc["total"], expected["total"])


def test_cashflow():
    """
    Test the correctness of total project cashflow computation.

    Ensures that the calculated cashflow array matches the expected
    values using ``np.testing.assert_allclose``.

    Notes
    -----
    - ``_calc_attr("cashflow")`` retrieves the model-generated cashflow.
    - Expected values represent the predefined project cashflow profile.
    """

    cashfl = {
        "calculated": _calc_attr(attr="cashflow"),
        "expected": np.array(
            [-1_200, -1_200, -1_200, -1_200, -300, -300, -600, 11_400, 11_400, 11_400]
        ),
    }

    # Execute testing
    np.testing.assert_allclose(cashfl["calculated"], cashfl["expected"])
