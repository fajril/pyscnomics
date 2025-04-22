
import numpy as np
from pyscnomics.econ.costs import SunkCost
from pyscnomics.econ.selection import SunkCostEndPoint


posc = SunkCost(
    start_year=2023,
    end_year=2032,
    expense_year=np.array([2023, 2024, 2025, 2026, 2027]),
    cost=np.array([100, 100, 100, 50, 50]),
    # end_point_reference=SunkCostEndPoint.POD_I,
    onstream_year=2027,
    pod1_year=2025,
)

t1 = posc.sunk_cost_total
# t1 = posc.pre_onstream_cost_total

# print('\t')
# print(f'Filetype: {type()}')
# print(f'Length: {len()}')
# print()

print('\t')
print(f'Filetype: {type(t1)}')
# print(f'Length: {len(t1)}')
print('t1 = ', t1)

