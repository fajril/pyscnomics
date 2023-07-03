import numpy as np

from datetime import date
from dataclasses import dataclass, field
from enum import Enum

import pyxirr


class DeprMethod(Enum):
    SL = "sl"
    DB = "db"
    UOP = "uop"


class FluidType(Enum):
    OIL = "oil"
    GAS = "gas"


@dataclass
class Tangible:
    start_date: date
    cost: np.ndarray
    salvage_value: np.ndarray
    expense_year: list[int]
    pis_year: list[int]
    cost_allocation: list[FluidType]
    depr_method: DeprMethod

    def __post_init__(self):
        arr_length = len(self.cost)
        if not all(
            [
                len(arr) == arr_length
                for arr in [
                    self.salvage_value,
                    self.expense_year,
                    self.pis_year,
                    self.cost_allocation,
                ]
            ]
        ):
            raise ValueError(
                f"Inequal length of array: cost: {len(self.cost)}, salvage_value: {len(self.salvage_value)} \
                    expense_year: {len(self.expense_year)}, pis_year: {len(self.cost_allocation)}"
            )
