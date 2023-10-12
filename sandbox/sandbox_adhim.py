import numpy as np
from pyscnomics.econ.costs import Tangible
from pyscnomics.econ.selection import FluidType


oil_mangga1_tangible = Tangible(
    start_year=2023,
    end_year=2030,
    cost=np.array([50, 50, 50]),
    expense_year=np.array([2023, 2024, 2025]),
    cost_allocation=FluidType.OIL,
    depreciation_factor=np.array([0.5, 0.5, 0.5]),
    vat_portion=1.0,
    pdri_portion=1.0,
)

oil_mangga2_tangible = Tangible(
    start_year=2023,
    end_year=2030,
    cost=np.array([50, 50, 50]),
    expense_year=np.array([2023, 2024, 2025]),
    cost_allocation=FluidType.OIL,
    depreciation_factor=np.array([0.5, 0.5, 0.5]),
    vat_portion=1.0,
    pdri_portion=1.0,
)

oil_apel1_tangible = Tangible(
    start_year=2023,
    end_year=2030,
    cost=np.array([25, 25, 25]),
    expense_year=np.array([2027, 2028, 2029]),
    cost_allocation=FluidType.OIL,
    vat_portion=0.5,
    pdri_portion=0.5,
)

oil_apel2_tangible = Tangible(
    start_year=2023,
    end_year=2030,
    cost=np.array([25, 25, 25]),
    expense_year=np.array([2027, 2028, 2029]),
    cost_allocation=FluidType.OIL,
    vat_portion=0.5,
    pdri_portion=0.5,
)

oil_nanas1_tangible = Tangible(
    start_year=2023,
    end_year=2030,
    cost=np.array([10, 10, 10]),
    expense_year=np.array([2027, 2028, 2029]),
    cost_allocation=FluidType.OIL,
    vat_portion=0.5,
    pdri_portion=1.0,
)

oil_nanas2_tangible = Tangible(
    start_year=2023,
    end_year=2030,
    cost=np.array([10, 10, 10]),
    expense_year=np.array([2027, 2028, 2029]),
    cost_allocation=FluidType.OIL,
    vat_portion=0.5,
    pdri_portion=1.0,
)

tangible_list = [
    oil_mangga1_tangible,
    oil_mangga2_tangible,
    oil_apel1_tangible,
    oil_apel2_tangible,
    oil_nanas1_tangible,
    oil_nanas2_tangible,
]


from itertools import groupby
from operator import attrgetter

# Sort the list by the attributes you want to group by
tangible_list.sort(key=attrgetter("vat_portion", "pdri_portion"))

grouped_objects = []
for key, group in groupby(tangible_list, key=attrgetter("vat_portion", "pdri_portion")):
    grouped_objects.append((key, list(group)))

print(grouped_objects)
