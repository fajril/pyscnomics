"""
A collection of unit testings for module project.py
"""
from typing import final

import pytest
import numpy as np
from datetime import date

from pyscnomics.econ.selection import FluidType, TaxType
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.costs import CapitalCost, Intangible, OPEX, ASR
from pyscnomics.contracts.project import BaseProject, BaseProjectException


# Create an example of Lifting data
lifting_mangga = Lifting(
    start_year=2023,
    end_year=2030,
    lifting_rate=np.array([100, 100, 100]),
    price=np.array([10, 10, 10]),
    prod_year=np.array([2023, 2024, 2025]),
    fluid_type=FluidType.OIL,
)

lifting_apel = Lifting(
    start_year=2023,
    end_year=2030,
    lifting_rate=np.array([50, 50, 50]),
    price=np.array([10, 10, 10]),
    prod_year=np.array([2027, 2028, 2029]),
    fluid_type=FluidType.OIL,
)

lifting_jeruk = Lifting(
    start_year=2023,
    end_year=2030,
    lifting_rate=np.array([100, 100, 100]),
    price=np.array([1, 1, 1]),
    prod_year=np.array([2024, 2025, 2026]),
    ghv=np.array([0.1, 0.1, 0.1]),
    fluid_type=FluidType.GAS,
)

lifting_nanas = Lifting(
    start_year=2023,
    end_year=2030,
    lifting_rate=np.array([50, 50, 50]),
    price=np.array([1, 1, 1]),
    prod_year=np.array([2028, 2029, 2030]),
    ghv=np.array([0.1, 0.1, 0.1]),
    fluid_type=FluidType.GAS,
)


def test_incorrect_data_input():
    """A unit testing for incorrect data input"""

    # Incorrect start date
    with pytest.raises(BaseProjectException):
        BaseProject(
            start_date=date(2032, 1, 1),
            end_date=date(2030, 12, 31),
        )

    # Oil onstream year is after the end project year
    with pytest.raises(BaseProjectException):
        BaseProject(
            start_date=date(2023, 1, 1),
            end_date=date(2030, 12, 31),
            oil_onstream_date=date(2031, 1, 1),
            lifting=(lifting_mangga, lifting_apel),
        )

    # Oil onstream year is before the start year of the project
    with pytest.raises(BaseProjectException):
        BaseProject(
            start_date=date(2023, 1, 1),
            end_date=date(2030, 12, 31),
            oil_onstream_date=date(2020, 1, 1),
            lifting=(lifting_mangga, lifting_apel),
        )

    # Oil onstream year is inconsistent with prod_year
    with pytest.raises(BaseProjectException):
        BaseProject(
            start_date=date(2023, 1, 1),
            end_date=date(2030, 1, 1),
            oil_onstream_date=date(2026, 1, 1),
            lifting=(lifting_mangga, lifting_apel),
        )


def test_base_project_lifting():
    """A unit testing for base project methods: _get_oil_lifting() and _get_gas_lifting() """

    # Expected results
    oil_lifting_rate = np.array([100, 100, 100, 50, 50, 50])
    oil_revenue = np.array([1000, 1000, 1000, 0, 500, 500, 500, 0])
    gas_lifting_rate = np.array([100, 100, 100, 50, 50, 50])
    gas_revenue = np.array([0, 10, 10, 10, 0, 5, 5, 5])

    # Calculated results
    base_case = BaseProject(
        start_date=date(2023, 1, 1),
        end_date=date(2030, 12, 31),
        lifting=(
            lifting_mangga,
            lifting_apel,
            lifting_jeruk,
            lifting_nanas,
        )
    )

    results = vars(base_case)

    oil_lifting_rate_calc = results["_oil_lifting"].lifting_rate
    oil_revenue_calc = results["_oil_revenue"]
    gas_lifting_rate_calc = results["_gas_lifting"].lifting_rate
    gas_revenue_calc = results["_gas_revenue"]

    # Execute testing (expected == calculated)
    np.testing.assert_allclose(oil_lifting_rate, oil_lifting_rate_calc)
    np.testing.assert_allclose(oil_revenue, oil_revenue_calc)
    np.testing.assert_allclose(gas_lifting_rate, gas_lifting_rate_calc)
    np.testing.assert_allclose(gas_revenue, gas_revenue_calc)


def test_base_project_tangible_expenditures():
    """ A unit testing to calculate tangible expenditures in BaseProject class """

    # Expected results
    oil_tangible_expenditures = np.array([25, 25, 50, 50, 25, 0, 0, 0])
    gas_tangible_expenditures = np.array([0, 10, 20, 20, 0, 0, 0, 0])

    # Calculated results
    oil_mangga_tangible = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([25, 25, 25, 25]),
        expense_year=np.array([2023, 2024, 2025, 2026]),
        cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL],
    )

    oil_apel_tangible = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([25, 25, 25]),
        expense_year=np.array([2025, 2026, 2027]),
        cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL],
    )

    gas_mangga_tangible = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([10, 10, 10]),
        expense_year=np.array([2024, 2025, 2026]),
        cost_allocation=[FluidType.GAS, FluidType.GAS, FluidType.GAS],
    )

    gas_apel_tangible = CapitalCost(
        start_year=2023,
        end_year=2030,
        cost=np.array([10, 10]),
        expense_year=np.array([2025, 2026]),
        cost_allocation=[FluidType.GAS, FluidType.GAS],
    )

    base_case = BaseProject(
        start_date=date(2023, 1, 1),
        end_date=date(2030, 1, 1),
        capital_cost=(
            oil_mangga_tangible,
            oil_apel_tangible,
            gas_mangga_tangible,
            gas_apel_tangible
        )
    )

    base_case._get_expenditures_post_tax()
    results = vars(base_case)

    oil_tangible_expenditures_calc = results["_oil_capital_expenditures_post_tax"]
    gas_tangible_expenditures_calc = results["_gas_capital_expenditures_post_tax"]

    # Execute testing (expected == calculated)
    np.testing.assert_allclose(oil_tangible_expenditures, oil_tangible_expenditures_calc)
    np.testing.assert_allclose(gas_tangible_expenditures, gas_tangible_expenditures_calc)


def test_base_project_intangible_expenditures():
    """ A unit testing to calculate intangible expenditures in BaseProject class """

    # Expected results
    oil_case1 = np.array([100, 0, 100, 0, 100, 0, 100, 0])
    gas_case1 = np.array([0, 50, 0, 50, 0, 50, 0, 50])
    oil_case2 = np.array([101, 0, 103, 0, 105, 0, 107, 0])
    gas_case2 = np.array([0, 51, 0, 52, 0, 53, 0, 54])

    # Calculated results
    intangible_mangga = Intangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 100, 100, 100]),
        expense_year=np.array([2023, 2025, 2027, 2029]),
        cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL],
        tax_portion=np.array([1, 1, 1, 1]),
    )

    intangible_apel = Intangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([50, 50, 50, 50]),
        expense_year=np.array([2024, 2026, 2028, 2030]),
        cost_allocation=[FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS],
        tax_portion=np.array([1, 1, 1, 1]),
    )

    case1 = BaseProject(
        start_date=date(2023, 1, 1),
        end_date=date(2030, 1, 1),
        intangible_cost=(intangible_mangga, intangible_apel),
    )

    case2 = BaseProject(
        start_date=date(2023, 1, 1),
        end_date=date(2030, 1, 1),
        intangible_cost=(intangible_mangga, intangible_apel),
    )

    case1._get_expenditures_post_tax()
    case2._get_expenditures_post_tax(
        tax_rate=np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]),
        inflation_rate=0.05,
    )

    results1 = vars(case1)
    results2 = vars(case2)

    oil_case1_calc = results1["_oil_intangible_expenditures_post_tax"]
    gas_case1_calc = results1["_gas_intangible_expenditures_post_tax"]
    oil_case2_calc = results2["_oil_intangible_expenditures_post_tax"]
    gas_case2_calc = results2["_gas_intangible_expenditures_post_tax"]

    # Testing (expected == calculated)
    np.testing.assert_allclose(oil_case1, oil_case1_calc)
    np.testing.assert_allclose(gas_case1, gas_case1_calc)
    np.testing.assert_allclose(oil_case2, oil_case2_calc)
    np.testing.assert_allclose(gas_case2, gas_case2_calc)


def test_base_project_opex_expenditures():
    """ A unit testing to calculate opex expenditures in BaseProject class """

    # Expected results
    oil_case1 = np.array([100, 100, 100, 100, 0, 0, 0, 0])
    gas_case1 = np.array([0, 0, 0, 0, 0, 50, 50, 50])

    # Calculated results
    opex_mangga = OPEX(
        start_year=2023,
        end_year=2030,
        fixed_cost=np.array([100, 100, 100, 100]),
        expense_year=np.array([2023, 2024, 2025, 2026]),
        cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL],
    )

    opex_apel = OPEX(
        start_year=2023,
        end_year=2030,
        fixed_cost=np.array([50, 50, 50]),
        expense_year=np.array([2028, 2029, 2030]),
        cost_allocation=[FluidType.GAS, FluidType.GAS, FluidType.GAS],
    )

    case1 = BaseProject(
        start_date=date(2023, 1, 1),
        end_date=date(2030, 1, 1),
        opex=(opex_mangga, opex_apel),
    )

    case1._get_expenditures_post_tax()

    results1 = vars(case1)

    oil_case1_calc = results1["_oil_opex_expenditures_post_tax"]
    gas_case1_calc = results1["_gas_opex_expenditures_post_tax"]

    # Testing (expected == calculated)
    np.testing.assert_allclose(oil_case1, oil_case1_calc)
    np.testing.assert_allclose(gas_case1, gas_case1_calc)


def test_base_project_asr_expenditures():
    """ A unit testing to calculate asr expenditures in BaseProject class """

    oil_case1 = np.array([100.0, 100.0, 100.0, 100.0, 0, 0, 0, 0])
    gas_case1 = np.array([0, 0, 0, 0, 50, 50, 50, 50])
    oil_case2 = np.array([106.8, 105.95, 105.1, 0, 0, 0, 101.7, 0])
    gas_case2 = np.array([0, 0, 0, 0, 51.7, 51.275, 0, 50.425])

    # Calculated results
    asr_mangga = ASR(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 100, 100, 100]),
        expense_year=np.array([2023, 2024, 2025, 2026]),
        final_year=np.array([2023, 2024, 2025, 2026]),
        cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL],
    )

    asr_apel = ASR(
        start_year=2023,
        end_year=2030,
        cost=np.array([50, 50, 50, 50]),
        expense_year=np.array([2027, 2028, 2029, 2030]),
        final_year=np.array([2027, 2028, 2029, 2030]),
        cost_allocation=[FluidType.GAS, FluidType.GAS, FluidType.GAS, FluidType.GAS],
    )

    asr_jeruk = ASR(
        start_year=2023,
        end_year=2030,
        cost=np.array([100, 100, 100, 100]),
        expense_year=np.array([2023, 2024, 2025, 2029]),
        final_year=np.array([2023, 2024, 2025, 2029]),
        cost_allocation=[FluidType.OIL, FluidType.OIL, FluidType.OIL, FluidType.OIL],
        tax_portion=np.array([0.85, 0.85, 0.85, 0.85]),
    )

    asr_nanas = ASR(
        start_year=2023,
        end_year=2030,
        cost=np.array([50, 50, 50]),
        expense_year=np.array([2027, 2028, 2030]),
        final_year=np.array([2027, 2028, 2030]),
        cost_allocation=[FluidType.GAS, FluidType.GAS, FluidType.GAS],
        tax_portion=np.array([0.85, 0.85, 0.85]),
    )

    case1 = BaseProject(
        start_date=date(2023, 1, 1),
        end_date=date(2030, 1, 1),
        asr_cost=(asr_mangga, asr_apel),
    )

    case2 = BaseProject(
        start_date=date(2023, 1, 1),
        end_date=date(2030, 1, 1),
        asr_cost=(asr_jeruk, asr_nanas),
    )

    case1._get_expenditures_post_tax()
    case2._get_expenditures_post_tax(
        tax_rate=np.array([0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]),
        inflation_rate=0.03,
    )

    results1 = vars(case1)
    results2 = vars(case2)

    oil_case1_calc = results1["_oil_asr_expenditures_post_tax"]
    gas_case1_calc = results1["_gas_asr_expenditures_post_tax"]
    oil_case2_calc = results2["_oil_asr_expenditures_post_tax"]
    gas_case2_calc = results2["_gas_asr_expenditures_post_tax"]

    # Testing (expected == calculated)
    np.testing.assert_allclose(oil_case1, oil_case1_calc)
    np.testing.assert_allclose(gas_case1, gas_case1_calc)
    np.testing.assert_allclose(oil_case2, oil_case2_calc)
    np.testing.assert_allclose(gas_case2, gas_case2_calc)
