"""
A collection of unit testings to validate operations occurred
in class BaseProject: sunk cost
"""

from datetime import date
import numpy as np

from pyscnomics.econ.selection import OtherRevenue, FluidType, CostType
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.costs import CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales
from pyscnomics.contracts.project import BaseProject


def test_base_project_get_commodity_lifting():

    # Synthetic Data Lifting
    oil_lifting = Lifting(
        start_year=2023,
        end_year=2034,
        prod_year=np.array([2030, 2031, 2032, 2033, 2034]),
        lifting_rate=np.array([100, 100, 100, 100, 100]),
        price=np.array([10, 10, 10, 10, 10]),
        fluid_type=FluidType.OIL,
    )

    gas_lifting = Lifting(
        start_year=2023,
        end_year=2034,
        prod_year=np.array([2030, 2031, 2032, 2033, 2029]),
        lifting_rate=np.array([10, 10, 10, 10, 10]),
        price=np.array([1, 1, 1, 1, 1]),
        fluid_type=FluidType.GAS,
    )

    sulfur_lifting = Lifting(
        start_year=2023,
        end_year=2034,
        prod_year=np.array([2030, 2031, 2032, 2033, 2029]),
        lifting_rate=np.array([5, 5, 5, 5, 5]),
        price=np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
        fluid_type=FluidType.SULFUR,
    )

    total_lifting = tuple([oil_lifting, gas_lifting, sulfur_lifting])

    pr = BaseProject(
        start_date=date(year=2023, month=1, day=1),
        end_date=date(year=2034, month=12, day=31),
        oil_onstream_date=date(year=2030, month=1, day=1),
        gas_onstream_date=date(year=2029, month=1, day=1),
        approval_year=2027,
        lifting=total_lifting,
    )

    # Generate calculated results
    calc_oil = pr._get_oil_lifting()
    calc_gas = pr._get_gas_lifting()
    calc_sulfur = pr._get_sulfur_lifting()

    calculated = {
        "oil_lifting_rate": calc_oil.lifting_rate,
        "gas_lifting_rate": calc_gas.lifting_rate,
        "sulfur_lifting_rate": calc_sulfur.lifting_rate,
        "oil_price": calc_oil.price,
        "gas_price": calc_gas.price,
        "sulfur_price": calc_sulfur.price,
    }

    # Generate expected results
    expected = {
        "oil_lifting_rate": np.array([100, 100, 100, 100, 100]),
        "gas_lifting_rate": np.array([10, 10, 10, 10, 10]),
        "sulfur_lifting_rate": np.array([5, 5, 5, 5, 5]),
        "oil_price": np.array([10, 10, 10, 10, 10]),
        "gas_price": np.array([1, 1, 1, 1, 1]),
        "sulfur_price": np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
    }

    # Execute testings
    np.testing.assert_allclose(calculated["oil_lifting_rate"], expected["oil_lifting_rate"])
    np.testing.assert_allclose(calculated["gas_lifting_rate"], expected["gas_lifting_rate"])
    np.testing.assert_allclose(
        calculated["sulfur_lifting_rate"], expected["sulfur_lifting_rate"]
    )
    np.testing.assert_allclose(calculated["oil_price"], expected["oil_price"])
    np.testing.assert_allclose(calculated["gas_price"], expected["gas_price"])
    np.testing.assert_allclose(calculated["sulfur_price"], expected["sulfur_price"])


def test_base_project_commodity_revenue():

    # Synthetic Data Lifting
    oil_lifting = Lifting(
        start_year=2023,
        end_year=2034,
        prod_year=np.array([2030, 2031, 2032, 2033, 2034]),
        lifting_rate=np.array([100, 100, 100, 100, 100]),
        price=np.array([10, 10, 10, 10, 10]),
        fluid_type=FluidType.OIL,
    )

    gas_lifting = Lifting(
        start_year=2023,
        end_year=2034,
        prod_year=np.array([2030, 2031, 2032, 2033, 2029]),
        lifting_rate=np.array([10, 10, 10, 10, 10]),
        price=np.array([1, 1, 1, 1, 1]),
        fluid_type=FluidType.GAS,
    )

    sulfur_lifting = Lifting(
        start_year=2023,
        end_year=2034,
        prod_year=np.array([2030, 2031, 2032, 2033, 2029]),
        lifting_rate=np.array([5, 5, 5, 5, 5]),
        price=np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
        fluid_type=FluidType.SULFUR,
    )

    total_lifting = tuple([oil_lifting, gas_lifting, sulfur_lifting])

    pr = BaseProject(
        start_date=date(year=2023, month=1, day=1),
        end_date=date(year=2034, month=12, day=31),
        oil_onstream_date=date(year=2030, month=1, day=1),
        gas_onstream_date=date(year=2029, month=1, day=1),
        approval_year=2027,
        lifting=total_lifting,
    )

    # Generate calculated results
    calc_oil = pr._get_oil_lifting()
    calc_gas = pr._get_gas_lifting()
    calc_sulfur = pr._get_sulfur_lifting()

    calculated = {
        "oil_revenue": calc_oil.revenue(),
        "gas_revenue": calc_gas.revenue(),
        "sulfur_revenue": calc_sulfur.revenue(),
    }

    # Generate expected results
    expected = {
        "oil_revenue": np.array([0, 0, 0, 0, 0, 0, 0, 1_000, 1_000, 1_000, 1_000, 1_000]),
        "gas_revenue": np.array([0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 0]),
        "sulfur_revenue": np.array([0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0]),
    }

    np.testing.assert_allclose(calculated["oil_revenue"], expected["oil_revenue"])
    np.testing.assert_allclose(calculated["gas_revenue"], expected["gas_revenue"])
    np.testing.assert_allclose(calculated["sulfur_revenue"], expected["sulfur_revenue"])
