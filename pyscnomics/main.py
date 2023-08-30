import numpy as np
import datetime as dt

# import pyscnomics.econ.depreciation as depr
# from pyscnomics.contracts.project import BaseProject
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.selection import FluidType
from pyscnomics.econ.costs import (
    Tangible,
    Intangible,
    OPEX,
    ASR,
)

# Data
project_start_date = dt.date(year=2023, month=1, day=1)
project_end_date = dt.date(year=2032, month=12, day=31)
onstream_date = dt.date(year=2028, month=1, day=1)
dmo_start_production = dt.date(year=2025, month=1, day=1)
dmo_holiday = 60

# Operation
project_years = np.arange(project_start_date.year, project_end_date.year + 1, 1)
start_prod_year = dmo_start_production.year
start_dmo_year = start_prod_year + int(dmo_holiday / 12)
indices = np.argwhere(project_years == start_dmo_year).ravel()


print('\t')
print('project years = ', project_years)
print('start prod year = ', start_prod_year)
print('start dmo year = ', start_dmo_year)

print('\t')
print(f'Shape: {indices.shape}')
print('indices = ', indices)

# a = np.datetime64('2023-08-30')
# year = a.astype('datetime64[Y]').tolist().year
#
# print('\t')
# print(a('Y'))

# lifting = Lifting(
#     start_year=2023,
#     end_year=2030,
#     lifting_rate=np.array([100, 100, 100, 100]),
#     price=np.array([1, 1, 1, 1]),
#     prod_year=np.array([2027, 2028, 2029, 2030]),
# )
#
# print('\t')
# print(lifting)


