
import numpy as np
from pyscnomics.econ.costs import SunkCost
from pyscnomics.econ.selection import FluidType, SunkCostInvestmentType


posc = SunkCost(
    start_year=2023,
    end_year=2030,
    onstream_year=2027,
    pod1_year=2025,
    expense_year=np.array([2023, 2024, 2024, 2026, 2025, 2027]),
    cost=np.array([90, 100, 100, 40, 50, 10]),
    cost_allocation=[
        FluidType.OIL,
        FluidType.OIL,
        FluidType.OIL,
        FluidType.GAS,
        FluidType.GAS,
        FluidType.GAS,
    ],
    investment_type=[
        SunkCostInvestmentType.TANGIBLE,
        SunkCostInvestmentType.TANGIBLE,
        SunkCostInvestmentType.TANGIBLE,
        SunkCostInvestmentType.INTANGIBLE,
        SunkCostInvestmentType.INTANGIBLE,
        SunkCostInvestmentType.INTANGIBLE,
    ],
    tax_portion=np.array([1, 1, 1, 1, 1, 1]),
)

print('\t')
print('cost = ', posc.cost)
print('expense_year = ', posc.expense_year)
print('project_years = ', posc.project_years)

t1 = posc.preonstream_cost_amortization_book_value(
    tax_rate=0.0,
    fluid_type=FluidType.GAS,
    investment_config=SunkCostInvestmentType.INTANGIBLE,
    prod=np.array([50, 100]),
    prod_year=np.array([2027, 2028]),
    amortization_len=8,
)

# t1 = posc.get_preonstream_cost_amortization_charge(
#     tax_rate=0.0,
#     fluid_type=FluidType.GAS,
#     investment_config=SunkCostInvestmentType.INTANGIBLE,
#     prod=np.array([50, 100]),
#     prod_year=np.array([2027, 2028]),
#     amortization_len=8,
# )

# t1 = posc.get_sunk_cost_amortization_charge(
#     tax_rate=0.0,
#     fluid_type=FluidType.GAS,
#     investment_config=SunkCostInvestmentType.INTANGIBLE,
#     prod=np.array([50, 100]),
#     prod_year=np.array([2027, 2028]),
#     amortization_len=8,
# )

# print('\t')
# print(f'Filetype: {type()}')
# print(f'Length: {len()}')
# print()

# print('\t')
# print(f'Filetype: {type(t1)}')
# print(f'Length: {len(t1)}')
# print('t1 = \n', t1)

# print('\t')
# print(f'Filetype: {type(t1)}')
# print(f'Length: {len(t1)}')
# print('t1 = ', t1)
#
# print('\t')
# print(f'Filetype: {type(t2)}')
# print(f'Length: {len(t2)}')
# print('t2 = ', t2)


# def _get_fluid_type_id(self, fluid_type: FluidType) -> tuple:
#     # Year of POD I approval equals to onstream year
#     if self.pod1_year == self.onstream_year:
#
#         # Determine sunk cost ID for a particular fluid type (OIL or GAS)
#         sc_id = np.argwhere(self.expense_year <= self.onstream_year).ravel()
#         sc_cost_allocation_id = np.array(
#             [self.cost_allocation[val] for _, val in enumerate(sc_id)]
#         )
#         sc_fluid_type_id = np.array(
#             [sc_id[i] for i, val in enumerate(sc_cost_allocation_id) if val == fluid_type]
#         )
#
#         # Determine pre-onstream cost ID for a particular fluid type (OIL or GAS)
#         poc_fluid_type_id = np.array([])
#
#     # Year of POD I approval is before the onstream year
#     elif self.pod1_year < self.onstream_year:
#
#         # Determine sunk cost ID for a particular fluid type (OIL or GAS)
#         sc_id = np.argwhere(self.expense_year <= self.pod1_year).ravel()
#         sc_cost_allocation_id = np.array(
#             [self.cost_allocation[val] for _, val in enumerate(sc_id)]
#         )
#         sc_fluid_type_id = np.array(
#             [sc_id[i] for i, val in enumerate(sc_cost_allocation_id) if val == fluid_type]
#         )
#
#         # Determine pre-onstream cost ID for a particular fluid type (OIL or GAS)
#         poc_id = np.argwhere(self.expense_year >= self.pod1_year).ravel()
#         poc_cost_allocation_id = np.array(
#             [self.cost_allocation[val] for _, val in enumerate(poc_id)]
#         )
#         poc_fluid_type_id = np.array(
#             [poc_id[i] for i, val in enumerate(poc_cost_allocation_id) if val == fluid_type]
#         )
#
#     else:
#         raise SunkCostException(f"Cannot have POD I year after the onstream year")
#
#     return sc_fluid_type_id, poc_fluid_type_id
