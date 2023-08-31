"""
A unit testing for module project.py.
"""

import numpy as np

from pyscnomics.econ import revenue, selection, costs
from pyscnomics.contracts import project, costrecovery


# Data input
start_year = 2023
end_year = 2030

# Lifting data
oil_lifting = [100, 100, 100]
oil_price = [10, 10, 10]
oil_prod_year = [2025, 2026, 2027]

gas_lifting = [10, 10, 10, 10]
gas_price = [1, 1, 1, 1]
gas_prod_year = [2027, 2028, 2029, 2030]

liftingOil = revenue.Lifting(
    start_year=start_year,
    end_year=end_year,
    lifting_rate=np.array(oil_lifting),
    price=np.array(oil_price),
    prod_year=np.array(oil_prod_year),
    fluid_type=selection.FluidType.OIL
)

liftingGas = revenue.Lifting(
    start_year=start_year,
    end_year=end_year,
    lifting_rate=np.array(gas_lifting),
    price=np.array(gas_price),
    prod_year=np.array(gas_prod_year),
    fluid_type=selection.FluidType.GAS
)

#





