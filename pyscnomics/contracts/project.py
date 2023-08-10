from dataclasses import dataclass, field
from datetime import date

import numpy as np

from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.costs import (
    Tangible,
    Intangible,
    OPEX,
    ASR,
)
from pyscnomics.econ.results import CashFlow

@dataclass
class BaseProject:
    start_date: date
    end_date: date
    lifting: tuple[Lifting]
    tangible_cost: tuple[Tangible]
    intangible_cost: tuple[Intangible] = field(default=None, repr=False)
    opex: tuple[OPEX] = field(default=None, repr=False)
    asr_cost: tuple[ASR] = field(default=None, repr=False)

    def __post_init__(self):
        if self.start_date < self.end_date:
            self.duration = self.end_date.year() - self.start_date.year() + 1
        else:
            raise ValueError(
                f"start date {self.start_date} "
                f"is after the end date: {self.end_date}"
            )
        #TODO: write validation for costs date



    def run(self):
        revenues = np.zeros(self.duration)
        for lift in self.lifting:
            revenues = revenues + lift.revenue()

        expenditures = np.zeros(self.duration)

        for cost in self.tangible_cost:
            expenditures = expenditures + cost.expenditures()

        for cost in self.intangible_cost:
            expenditures = expenditures + cost.expenditures()

        for cost in self.opex:
            expenditures = expenditures + cost.expenditures()

        for cost in self.asr_cost:
            expenditures = expenditures + cost.expenditures()

        return CashFlow(start_date=self.start_date,
                        end_date=self.end_date,
                        cash=revenues - expenditures)
