# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 08:57:18 2023

Check json validation: https://jsonlint.com/
"""
import json
import numpy as np
import os

from pyscnomics.econ.selection import FluidType
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.costs import Tangible, Intangible, OPEX, ASR

def load_data(data_name):
    """


    Parameters
    ----------
    data_name : str
        dataset name such as "CR_Gas"

    Raises
    ------
    FileNotFoundError
        DESCRIPTION.

    Returns
    -------
    data_dict : TYPE
        DESCRIPTION.

    """
    json_path = os.getcwd() + "\\pyscnomics\\dataset\\" + data_name + ".json"

    # check if path exist
    if os.path.exists(json_path):
        print(json_path)

        # Open the JSON file and load its content
        with open(json_path, "r") as file:

            data_dict = json.load(file)

            dict_obj = {}

            # Create Object
            start_year = data_dict["Year Lifting"][0]
            end_year = data_dict["Year Lifting"][-1]

            for phase in ["Gas", "Oil"]:
                if phase == "Gas":
                    fluid_type_used = FluidType.GAS
                else:
                    fluid_type_used = FluidType.OIL

                # Create Lifting Object
                lifting_rate = np.array(data_dict[phase + " Lifting"])
                price = np.array(data_dict[phase + " Price"])

                dict_obj[phase + " Lifting"] = \
                    Lifting(start_year, end_year, lifting_rate,
                            price, fluid_type=fluid_type_used)

                # Create Tangible Object

                # Create Intangible Object

                # Create OPEX Object
                fixed_cost = data_dict[phase + " OPEX"]
                dict_obj[phase + " OPEX"] = \
                    OPEX(start_year, end_year, fixed_cost,
                         cost_allocation=fluid_type_used)

                # Create ASR Object

            return dict_obj
    else:
        raise FileNotFoundError(f"File not found: {json_path}")

if __name__ == "__main__":
    data_dict = load_data("CR_Gas")