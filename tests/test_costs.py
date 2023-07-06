import numpy as np
import pyscnomics as psc


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
        project_name="Mangga Field",
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
