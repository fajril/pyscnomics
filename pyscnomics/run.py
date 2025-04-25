
import numpy as np
from pyscnomics.econ.costs import SunkCost
from pyscnomics.econ.selection import FluidType


posc = SunkCost(
    start_year=2023,
    end_year=2030,
    onstream_year=2027,
    pod1_year=2025,
    expense_year=np.array([2023, 2024, 2024, 2026, 2025]),
    cost=np.array([100, 100, 100, 50, 50]),
    cost_allocation=[FluidType.GAS, FluidType.OIL, FluidType.GAS, FluidType.OIL, FluidType.GAS],
    tax_portion=np.array([1, 1, 1, 1, 1]),
)

t1 = posc.get_pre_onstream_cost_oil_amortization_charge(
    tax_rate=0.0,
    prod=np.array([50, 1_000]),
    prod_year=np.array([2027, 2028]),
    salvage_value=0.0,
    amortization_len=8,
)


# print('\t')
# print(f'Filetype: {type()}')
# print(f'Length: {len()}')
# print()

print('\t')
print(f'Filetype: {type(t1)}')
print(f'Length: {len(t1)}')
print('t1 = \n', t1)

# print('\t')
# print(f'Filetype: {type(t1)}')
# # print(f'Length: {len(t1)}')
# print('t1 = ', t1)
#
# print('\t')
# print(f'Filetype: {type(t2)}')
# # print(f'Length: {len(t2)}')
# print('t2 = ', t2)

# # Specify condition for end_point_reference equals to onstream year
# if self.end_point_reference == SunkCostEndPoint.ONSTREAM:
#
#     if self.onstream_year == self.start_year:
#         self.expense_year = np.full_like(self.expense_year, self.start_year, dtype=int)
#
#         print('\t')
#         print(f'Filetype: {type(self.expense_year)}')
#         print(f'Length: {len(self.expense_year)}')
#         print(self.expense_year)
#
#     else:
#
#         # Prepare attribute onstream_year
#         if not isinstance(self.onstream_year, int):
#             raise SunkCostException(
#                 f"Attribute onstream_year must be provided as an int, "
#                 f"not as a/an {self.onstream_year.__class__.__qualname__}"
#             )
#
#         if self.onstream_year < self.start_year:
#             raise SunkCostException(
#                 f"Onstream year ({self.onstream_year}) is before the start "
#                 f"year of the project ({self.start_year})"
#             )
#
#         if self.onstream_year > self.end_year:
#             raise SunkCostException(
#                 f"Onstream year ({self.onstream_year}) is after the end year "
#                 f"of the project ({self.end_year})"
#             )
#
#         # Filter expense_year array
#         expense_year_large_sum = np.sum(self.expense_year > self.onstream_year, dtype=int)
#         if expense_year_large_sum > 0:
#             raise SunkCostException(
#                 f"Onstream year ({self.onstream_year}) is before the "
#                 f"expense_year ({self.expense_year})"
#             )
#
#         # Fill attribute sunk_cost_total
#         ids = np.argwhere(self.expense_year <= self.onstream_year).ravel()
#         self.sunk_cost_total = np.sum(self.cost[ids], dtype=np.float64)
#
# # Specify condition for end_point_reference equals to POD I year
# elif self.end_point_reference == SunkCostEndPoint.POD_I:
#
#     # Prepare attribute onstream year
#     if self.onstream_year is None:
#         raise SunkCostException(
#             f"Missing data for onstream_year: ({self.onstream_year})"
#         )
#
#     else:
#         if not isinstance(self.onstream_year, int):
#             raise SunkCostException(
#                 f"Attribute onstream_year must be provided as an int, "
#                 f"not as a/an {self.onstream_year.__class__.__qualname__}"
#             )
#
#         if self.onstream_year < self.start_year:
#             raise SunkCostException(
#                 f"Onstream year ({self.onstream_year}) is before the start "
#                 f"year of the project ({self.start_year})"
#             )
#
#         if self.onstream_year > self.end_year:
#             raise SunkCostException(
#                 f"Onstream year ({self.onstream_year}) is after the end year "
#                 f"of the project ({self.end_year})"
#             )
#
#     # Prepare attribute POD I year
#     if self.pod1_year is None:
#         raise SunkCostException(
#             f"Missing data for POD I year: {self.pod1_year}"
#         )
#
#     else:
#         if not isinstance(self.pod1_year, int):
#             raise SunkCostException(
#                 f"Attribute pod1_year must be provided as an int, "
#                 f"not as a/an {self.pod1_year.__class__.__qualname__}"
#             )
#
#         if self.pod1_year < self.start_year:
#             raise SunkCostException(
#                 f"POD I year ({self.pod1_year}) is before the start year "
#                 f"of the project ({self.start_year})"
#             )
#
#         if self.pod1_year > self.end_year:
#             raise SunkCostException(
#                 f"POD I year ({self.pod1_year}) is after the end year "
#                 f"of the project ({self.end_year})"
#             )
#
#         if self.pod1_year > self.onstream_year:
#             raise SunkCostException(
#                 f"POD I year ({self.pod1_year}) is after the onstream year "
#                 f"({self.onstream_year})"
#             )
#
#     # Filter expense_year array
#     expense_year_large_sum = np.sum(self.expense_year > self.onstream_year, dtype=int)
#     if expense_year_large_sum > 0:
#         raise SunkCostException(
#             f"Onstream year ({self.onstream_year}) is before the "
#             f"expense_year ({self.expense_year})"
#         )
#
#     # Fill attribute sunk cost total and pre-onstream cost total
#     if self.pod1_year == self.start_year:
#         self.sunk_cost_total = 0.0
#         ids_pre_onstream_cost = np.argwhere(self.expense_year >= self.pod1_year).ravel()
#         self.pre_onstream_cost_total = (
#             np.sum(self.cost[ids_pre_onstream_cost], dtype=np.float64)
#         )
#
#     elif self.pod1_year == self.onstream_year:
#         ids_sunk_cost = np.argwhere(self.expense_year <= self.pod1_year).ravel()
#         self.sunk_cost_total = np.sum(self.cost[ids_sunk_cost], dtype=np.float64)
#         self.pre_onstream_cost_total = 0.0
#
#     else:
#         ids_sunk_cost = np.argwhere(self.expense_year <= self.pod1_year).ravel()
#         self.sunk_cost_total = np.sum(self.cost[ids_sunk_cost], dtype=np.float64)
#         ids_pre_onstream_cost = np.argwhere(self.expense_year > self.pod1_year).ravel()
#         self.pre_onstream_cost_total = (
#             np.sum(self.cost[ids_pre_onstream_cost], dtype=np.float64)
#         )
#
# else:
#     raise SunkCostException(
#         f"Sunk cost end-point reference is not recognized "
#         f"({self.end_point_reference})"
#     )
#
# self.sunk_cost_total = np.float64(self.sunk_cost_total)
# self.pre_onstream_cost_total = np.float64(self.pre_onstream_cost_total)
