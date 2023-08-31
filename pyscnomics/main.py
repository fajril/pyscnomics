import numpy as np
import datetime as dt
import dateutils

import pyscnomics.econ.depreciation as depr
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


# print('\t')
# print('project years = ', project_years)
# print('start prod year = ', start_prod_year)
# print('start dmo year = ', start_dmo_year)
#
# print('\t')
# print(f'Shape: {indices.shape}')
# print('indices = ', indices)


# depr_db = depr.declining_balance_depreciation_rate(
#     cost=100,
#     salvage_value=50,
#     useful_life=10,
#     decline_factor=2,
#     depreciation_len=5
# )
#
# print('\t')
# print('depr_db = ', depr_db)

lifting = Lifting(
    start_year=2023,
    end_year=2030,

)

def get_dmo(
        # dmo_volume_portion: float,
        # dmo_fee_portion: float,
        dmo_start_production: dt,
        # dmo_holiday: int,
        # lifting: Lifting,
        # ctr_pretax_share: float,
        # ES_ctr: np.ndarray
) -> np.ndarray:

    # # Instantiate dmo_volume array
    # dmo_volume = np.zeros_like(lifting.revenue())
    # dmo_fee = np.zeros_like(lifting.revenue())
    # ddmo = np.zeros_like(lifting.revenue())

    dmo_end_date = dmo_start_production + dateutils.relativedelta(months=60)

    print('\t')
    print('dmo end date = ', dmo_end_date)

    for i in [dmo_end_date.year, dmo_end_date.month, dmo_end_date.day]:
        print(i)


dmo = get_dmo(dmo_start_production=dt.date(year=2023, month=5, day=14))








