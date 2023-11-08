"""
A collection of unit testings for module project.py
"""

import pytest
import numpy as np
from datetime import date

from pyscnomics.econ.selection import FluidType
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.costs import Tangible
from pyscnomics.contracts.project import BaseProject, BaseProjectException


# Create an example of Lifting data
oil_mangga_lifting = Lifting(
    start_year=2023,
    end_year=2030,
    lifting_rate=np.array([100, 100, 100]),
    price=np.array([10, 10, 10]),
    prod_year=np.array([2025, 2026, 2027]),
    fluid_type=FluidType.OIL,
)

oil_apel_lifting = Lifting(
    start_year=2023,
    end_year=2030,
    lifting_rate=np.array([50, 50, 50]),
    price=np.array([10, 10, 10]),
    prod_year=np.array([2027, 2028, 2029]),
    fluid_type=FluidType.OIL,
)

oil_nanas_lifting = Lifting(
    start_year=2023,
    end_year=2030,
    lifting_rate=np.array([100, 100, 100]),
    price=np.array([10, 10, 10]),
    prod_year=np.array([2025, 2026, 2027]),
    fluid_type=FluidType.OIL,
)

gas_mangga_lifting = Lifting(
    start_year=2023,
    end_year=2030,
    lifting_rate=np.array([100, 100, 100]),
    price=np.array([1, 1, 1]),
    prod_year=np.array([2024, 2025, 2026]),
    ghv=np.array([0.1, 0.1, 0.1]),
    fluid_type=FluidType.GAS,
)

gas_apel_lifting = Lifting(
    start_year=2023,
    end_year=2030,
    lifting_rate=np.array([100, 100, 100]),
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
        )

    # Oil onstream year is before the start year of the project
    with pytest.raises(BaseProjectException):
        BaseProject(
            start_date=date(2023, 1, 1),
            end_date=date(2030, 12, 31),
            oil_onstream_date=date(2020, 1, 1),
        )

    # Oil onstream year is inconsistent with prod_year
    with pytest.raises(BaseProjectException):
        BaseProject(
            start_date=date(2023, 1, 1),
            end_date=date(2030, 1, 1),
            oil_onstream_date=date(2026, 1, 1),
            lifting=(oil_mangga_lifting, oil_apel_lifting)
        )


def test_base_project_lifting():
    """A unit testing for base project methods: _get_oil_lifting() and _get_gas_lifting() """

    # Expected results
    oil_lifting_rate = [100, 100, 100, 50, 50, 50]
    oil_revenue = [0, 0, 1000, 1000, 1500, 500, 500, 0]
    gas_lifting_rate = [100, 100, 100, 100, 100, 100]
    gas_revenue = [0, 10, 10, 10, 0, 10, 10, 10]

    # Calculated results
    base_case = BaseProject(
        start_date=date(2023, 1, 1),
        end_date=date(2030, 12, 31),
        lifting=(
            oil_mangga_lifting,
            oil_apel_lifting,
            gas_mangga_lifting,
            gas_apel_lifting,
        )
    )

    results = vars(base_case)

    oil_lifting_rate_calc = results["_oil_lifting"].lifting_rate
    # calc_oil_revenue = summary["_oil_revenue"]
    # calc_gas_lifting_rate = summary["_gas_lifting"].lifting_rate
    # calc_gas_revenue = summary["_gas_revenue"]

    # Execute testing (expected == calculated)
    np.testing.assert_allclose(oil_lifting_rate, oil_lifting_rate_calc)
    # np.testing.assert_allclose(oil_revenue, calc_oil_revenue)
    # np.testing.assert_allclose(gas_lifting_rate, calc_gas_lifting_rate)
    # np.testing.assert_allclose(gas_revenue, calc_gas_revenue)


def test_base_project_tangible():
    """ A unit testing for BaseProject class' methods: _get_oil_tangible() and _get_gas_tangible() """

    # Expected results
    oil_tangible_expenditures = [25, 25, 50, 50, 25, 0, 0, 0]
    oil_tangible_depreciation = [12.5, 18.75, 34.375, 42.1875, 34.375, 17.1875, 9.375, 4.6875]
    oil_tangible_undepreciated = 1.5625

    gas_tangible_expenditures = [0, 10, 20, 20, 0, 0, 0, 0]
    gas_tangible_depreciation = [0, 5, 12.5, 16.25, 8.125, 4.375, 2.5, 1.25]
    gas_tangible_undepreciated = 0.

    # Calculated results
    oil_mangga_tangible = Tangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([25, 25, 25, 25]),
        expense_year=np.array([2023, 2024, 2025, 2026]),
        cost_allocation=FluidType.OIL,
        depreciation_factor=np.array([0.5, 0.5, 0.5, 0.5])
    )

    oil_apel_tangible = Tangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([25, 25, 25]),
        expense_year=np.array([2025, 2026, 2027]),
        cost_allocation=FluidType.OIL,
        depreciation_factor=np.array([0.5, 0.5, 0.5])
    )

    gas_mangga_tangible = Tangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([10, 10, 10]),
        expense_year=np.array([2024, 2025, 2026]),
        cost_allocation=FluidType.GAS,
        depreciation_factor=np.array([0.5, 0.5, 0.5])
    )

    gas_apel_tangible = Tangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([10, 10]),
        expense_year=np.array([2025, 2026]),
        cost_allocation=FluidType.GAS,
        depreciation_factor=np.array([0.5, 0.5])
    )

    base_case = BaseProject(
        start_date=date(2023, 1, 1),
        end_date=date(2030, 1, 1),
        tangible_cost=(oil_mangga_tangible, oil_apel_tangible, gas_mangga_tangible, gas_apel_tangible)
    )

    base_case._get_costpool()

    summary = vars(base_case)

    calc_oil_tangible_expenditures = summary["_oil_tangible_expenditures"]
    calc_oil_tangible_depreciation = summary["_oil_depreciation"]
    calc_oil_tangible_undepreciated = summary["_oil_undepreciated_asset"]
    calc_gas_tangible_expenditures = summary["_gas_tangible_expenditures"]
    calc_gas_tangible_depreciation = summary["_gas_depreciation"]
    calc_gas_tangible_undepreciated = summary["_gas_undepreciated_asset"]

    # Execute testing (expected == calculated)
    np.testing.assert_allclose(oil_tangible_expenditures, calc_oil_tangible_expenditures)
    np.testing.assert_allclose(oil_tangible_depreciation, calc_oil_tangible_depreciation)
    np.testing.assert_allclose(oil_tangible_undepreciated, calc_oil_tangible_undepreciated)
    np.testing.assert_allclose(gas_tangible_expenditures, calc_gas_tangible_expenditures)
    np.testing.assert_allclose(gas_tangible_depreciation, calc_gas_tangible_depreciation)
    np.testing.assert_allclose(gas_tangible_undepreciated, calc_gas_tangible_undepreciated)


def test_base_project_comparison():
    """ A unit test for comparison involving instances of BaseProject """

    oil_mangga_tangible = Tangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([25, 25, 25, 25]),
        expense_year=np.array([2023, 2024, 2025, 2026]),
        cost_allocation=FluidType.OIL,
        depreciation_factor=np.array([0.5, 0.5, 0.5, 0.5]),
        vat_portion=0.8,
    )

    oil_apel_tangible = Tangible(
        start_year=2023,
        end_year=2030,
        cost=np.array([25, 25, 25]),
        expense_year=np.array([2025, 2026, 2027]),
        cost_allocation=FluidType.OIL,
        depreciation_factor=np.array([0.5, 0.5, 0.5]),
        vat_portion=0.5,
    )

    case1 = BaseProject(
        start_date=date(2023, 1, 1),
        end_date=date(2030, 12, 31),
        lifting=(oil_mangga_lifting, oil_apel_lifting),
        tangible_cost=(oil_mangga_tangible, oil_apel_tangible),
    )

    case2 = BaseProject(
        start_date=date(2023, 1, 1),
        end_date=date(2030, 12, 31),
        lifting=(oil_mangga_lifting, oil_apel_lifting),
        tangible_cost=(oil_mangga_tangible, oil_apel_tangible),
    )

    case3 = BaseProject(
        start_date=date(2023, 1, 1),
        end_date=date(2030, 12, 31),
        lifting=([oil_apel_lifting]),
        tangible_cost=([oil_apel_tangible]),
    )

    assert case1 == case2
    assert case3 < case1
    assert case3 <= case1
    assert case2 > case3
    assert case2 >= case3
    assert case1 != case3

