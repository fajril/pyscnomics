"""
A module to execute preliminary evaluations
"""

import numpy as np
from datetime import date
from dataclasses import asdict

from pyscnomics.econ.selection import DeprMethod, FluidType
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.costs import Tangible, ASR
from pyscnomics.contracts.project import BaseProject


# Lifting data
oil_mangga_lifting = Lifting(
        start_year=2023,
        end_year=2030,
        lifting_rate=np.array([100, 100, 100]),
        price=np.array([10, 10, 10]),
        prod_year=np.array([2026, 2027, 2028]),
        fluid_type=FluidType.OIL
)

oil_apel_lifting = Lifting(
        start_year=2023,
        end_year=2030,
        lifting_rate=np.array([50, 50, 50]),
        price=np.array([10, 10, 10]),
        prod_year=np.array([2027, 2028, 2029]),
        fluid_type=FluidType.OIL
)

gas_mangga_lifting = Lifting(
        start_year=2023,
        end_year=2030,
        lifting_rate=np.array([100, 100, 100]),
        price=np.array([1, 1, 1]),
        prod_year=np.array([2025, 2026, 2027]),
        ghv=np.array([0.1, 0.1, 0.1]),
        fluid_type=FluidType.GAS
)

gas_apel_lifting = Lifting(
        start_year=2023,
        end_year=2030,
        lifting_rate=np.array([100, 100, 100]),
        price=np.array([1, 1, 1]),
        prod_year=np.array([2028, 2029, 2030]),
        ghv=np.array([0.1, 0.1, 0.1]),
        fluid_type=FluidType.GAS
)

# Tangible data
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


base_project = BaseProject(
        start_date=date(2023, 1, 1),
        end_date=date(2030, 12, 31),
        oil_onstream_date=date(2026, 1, 1),
        gas_onstream_date=date(2025, 1, 1),
        lifting=(oil_mangga_lifting, oil_apel_lifting, gas_mangga_lifting, gas_apel_lifting)
)

base_project.run()

# create_dict = vars(base_project)
create_dict = asdict(base_project)

print('\t')
print(create_dict.keys())
print('create dict = \n', create_dict)

# print('\t')
# print('base project = ', base_project.lifting)

# print('\t')
# print(f'Filetype: {type(mangga_project.lifting)}, Length: {len(mangga_project.lifting)}')
# print('lifting = \n', mangga_project.lifting)
#
# print('\t')
# print(f'Filetype: {type(mangga_project.tangible_cost)}, Length: {len(mangga_project.tangible_cost)}')
# print('tangible cost = \n', mangga_project.tangible_cost)
#
# print('\t')
# print(f'Filetype: {type(mangga_project.intangible_cost)}')
# print(f'Length: {len(mangga_project.intangible_cost)}')
# print('intangible cost = \n', mangga_project.intangible_cost)
#
# print('\t')
# print(f'Filetype: {type(mangga_project.opex)}')
# print(f'Length: {len(mangga_project.opex)}')
# print('opex = \n', mangga_project.opex)
#
# print('\t')
# print(f'Filetype: {type(mangga_project.asr_cost)}')
# print(f'Length: {len(mangga_project.asr_cost)}')
# print('asr cost = \n', mangga_project.asr_cost)
