"""
Create Lifting and Cost instances to be passed on to the main executable.
"""

import os as os
import numpy as np
import pandas as pd
import time as tm

from pyscnomics.io.spreadsheet import Spreadsheet
from pyscnomics.econ.selection import FluidType
from pyscnomics.econ.revenue import Lifting


def get_lifting_instances_from_data(
    start_year: int,
    end_year: int,
    lifting_rate: dict,
    price: dict,
    prod_year: dict,
    fluid_type: FluidType,

):
    if fluid_type != FluidType.GAS:
        lifting = tuple(
            Lifting(
                start_year=start_year,
                end_year=end_year,
                lifting_rate=lifting_rate[key],
                price=price[key],
                prod_year=prod_year[key],
                fluid_type=fluid_type,
            )
            for key in prod_year.keys()
        )

    return lifting


if __name__ == "__main__":

    os.chdir("..")

    # Load and prepare the data
    data = Spreadsheet()
    data.prepare_data()

    # Specify instance: lifting_oil
    if "Transition" in data.general_config_data.type_of_contract:
        pass

    else:
        lifting_oil = get_lifting_instances_from_data(
            start_year=data.general_config_data.start_date_project.year,
            end_year=data.general_config_data.end_date_project.year,
            lifting_rate=data.oil_lifting_data.oil_lifting_rate,
            price=data.oil_lifting_data.oil_price,
            prod_year=data.oil_lifting_data.prod_year,
            fluid_type=FluidType.OIL,
        )

    print('\t')
    print(f'Filetype: {type(lifting_oil)}')
    print('lifting_oil = \n', lifting_oil)
