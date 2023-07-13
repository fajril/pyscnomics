import numpy as np
import pyscnomics as psc


def test_tangible_comparison():
    mangga_tangible = psc.Tangible(
        start_year=2023,
        end_year=2033,
        cost=np.array([200_000, 200_000]),
        expense_year=np.array([2023, 2024]),
        salvage_value=np.array([25_000, 25_000]),
        useful_life=np.array([5, 5]),
        cost_allocation=[psc.FluidType.OIL, psc.FluidType.OIL],
    )

    jeruk_tangible = psc.Tangible(
        start_year=2023,
        end_year=2033,
        cost=np.array([205_001, 200_000]),
        expense_year=np.array([2023, 2024]),
        salvage_value=np.array([25_000, 25_000]),
        useful_life=np.array([5, 5]),
        cost_allocation=[psc.FluidType.OIL, psc.FluidType.OIL],
    )

    hiu_tangible = psc.Tangible(
        start_year=2023,
        end_year=2033,
        cost=np.array([200_000, 250_000]),
        expense_year=np.array([2023, 2024]),
        salvage_value=np.array([25_000, 25_000]),
        useful_life=np.array([5, 5]),
        cost_allocation=[psc.FluidType.OIL, psc.FluidType.OIL],
    )

    assert mangga_tangible == jeruk_tangible
    assert mangga_tangible != hiu_tangible
    assert mangga_tangible < hiu_tangible
    assert mangga_tangible <= jeruk_tangible
    assert hiu_tangible > jeruk_tangible
    assert jeruk_tangible >= mangga_tangible


def test_tangible_arithmetics():
    mangga_tangible = psc.Tangible(
        start_year=2023,
        end_year=2033,
        cost=np.array([200_000, 200_000]),
        expense_year=np.array([2023, 2024]),
        salvage_value=np.array([25_000, 25_000]),
        useful_life=np.array([5, 5]),
        cost_allocation=[psc.FluidType.OIL, psc.FluidType.OIL],
    )

    cost_mul_by_two = np.array([400_000, 400_000])
    cost_add_tangible = np.array([200_000, 200_000, 200_000, 200_000])

    mangga_tangible_mult_by_two = mangga_tangible * 2
    mangga_tangible_add = mangga_tangible + mangga_tangible
    mangga_tangible_div = mangga_tangible / mangga_tangible_mult_by_two
    mangga_tangible_div_by_scalar = mangga_tangible_mult_by_two / 2

    np.testing.assert_array_almost_equal(cost_mul_by_two, mangga_tangible_mult_by_two.cost)
    np.testing.assert_array_almost_equal(cost_add_tangible, mangga_tangible_add.cost)
    np.testing.assert_almost_equal(mangga_tangible_div, 0.5)
    np.testing.assert_array_almost_equal(mangga_tangible.cost, mangga_tangible_div_by_scalar.cost)


def test_tangible():
    """ Test Tangible class
    """
    depreciation_charge = [
        80000.0,
        128000.0,
        76800.0,
        46080.0,
        18200.0,
        920,
        0,
        0,
        0,
        0,
        0
    ]

    mangga_tangible = psc.Tangible(
        start_year=2023,
        end_year=2033,
        cost=np.array([200_000, 200_000]),
        expense_year=np.array([2023, 2024]),
        salvage_value=np.array([25_000, 25_000]),
        useful_life=np.array([5, 5]),
        cost_allocation=[psc.FluidType.OIL, psc.FluidType.OIL],
    )
    depreciation_charge_calc = mangga_tangible.total_depreciation_rate(
        depr_method=psc.DeprMethod.DB, fluid_type=psc.FluidType.OIL
    )
    np.testing.assert_array_almost_equal(depreciation_charge, depreciation_charge_calc)
