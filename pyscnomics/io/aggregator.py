"""
Create Lifting and Cost instances to be passed on to the main executable.
"""

import os as os
import numpy as np
import pandas as pd
import time as tm
from dataclasses import dataclass, field, InitVar
from functools import reduce

from pyscnomics.io.spreadsheet import Spreadsheet
from pyscnomics.econ.selection import FluidType
from pyscnomics.econ.revenue import Lifting


class SpreadsheetException(Exception):
    """Exception to raise for a misuse of Spreadsheet class"""

    pass


@dataclass
class Aggregate(Spreadsheet):
    """
    Prepares an aggregate of lifting data (oil, condensate, gas, LPG propane, LPG butane,
    sulfur, electricity, and CO2) and cost data (tangible, intangible, opex, ASR) to be
    passed on to the subsequent economic analysis.
    """

    # Attributes associated with aggregates of lifting data
    oil_lifting_aggregate: dict | tuple[Lifting] = field(default=None, init=False)
    condensate_lifting_aggregate: dict | tuple[Lifting] = field(default=None, init=False)

    # Attributes associated with PSC transition
    psc_regimes: list = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # Configure attribute workbook_to_read
        if self.workbook_to_read is None:
            self.workbook_to_read = "Workbook.xlsb"

        if self.workbook_to_read is not None:
            if not isinstance(self.workbook_to_read, str):
                raise SpreadsheetException(
                    f"Excel filename must be provided in str format. "
                    f"{self.workbook_to_read} ({self.workbook_to_read.__class__.__qualname__}) "
                    f"is not a str format."
                )

        # Configure attribute directory_location
        if self.directory_location is None:
            self.load_dir = os.path.join(os.getcwd(), self.workbook_to_read)

        if self.directory_location is not None:
            self.load_dir = os.path.join(self.directory_location, self.workbook_to_read)

        # Arrange the loaded data
        self.prepare_data()

        # Prepare attributes associated with PSC transition
        self.psc_regimes = ["PSC 1", "PSC 2"]

    def _get_oil_lifting_aggregate(self) -> dict | tuple[Lifting]:
        """
        Retrieves the oil lifting aggregate based on the Production Sharing Contract (PSC) type.

        Returns
        -------
        Dict[str, Tuple[Lifting, ...]], Tuple[Lifting, ...]:
            -   If the type_of_contract is "Transition," returns a dictionary with
                PSC regimes as keys and a tuple of Lifting instances as values.
            -   If the type_of_contract is a single PSC (CR or GS), returns a tuple
                of Lifting instances.

        Notes
        -----
        (1) For single PSC case, the aggregate of oil lifting data is generated using
            a tuple comprehension of all available oil lifting data stored in
            parameter 'self.oil_lifting_data',
        (2) For PSC transition case, the aggregate of oil lifting data is stored in
            a dictionary with keys: ['PSC 1', 'PSC 2']. The value of each keys is a
            tuple of all available oil lifting data stored in parameter 'self.oil_lifting_data'
            for each corresponding PSC regimes.
        """
        # For PSC transition
        if "Transition" in self.oil_lifting_data.type_of_contract:
            start_year_combined = [
                self.general_config_data.start_date_project.year,
                self.general_config_data.start_date_project_second.year,
            ]

            end_year_combined = [
                self.general_config_data.end_date_project.year,
                self.general_config_data.end_date_project_second.year,
            ]

            oil_lifting_aggr = {
                psc: tuple(
                    [
                        Lifting(
                            start_year=start_year_combined[i],
                            end_year=end_year_combined[i],
                            lifting_rate=self.oil_lifting_data.oil_lifting_rate[ws][psc],
                            price=self.oil_lifting_data.oil_price[ws][psc],
                            prod_year=self.oil_lifting_data.prod_year[ws][psc],
                            fluid_type=FluidType.OIL,
                        ) for ws in self.oil_lifting_data.prod_year.keys()
                    ]
                )
                for i, psc in enumerate(self.psc_regimes)
            }

        # For single PSC (CR or GS)
        else:
            oil_lifting_aggr = tuple(
                [
                    Lifting(
                        start_year=self.general_config_data.start_date_project.year,
                        end_year=self.general_config_data.end_date_project.year,
                        lifting_rate=self.oil_lifting_data.oil_lifting_rate[ws],
                        price=self.oil_lifting_data.oil_price[ws],
                        prod_year=self.oil_lifting_data.prod_year[ws],
                        fluid_type=FluidType.OIL,
                    ) for ws in self.oil_lifting_data.prod_year.keys()
                ]
            )

        return oil_lifting_aggr

    def _get_condensate_lifting_aggregate(self) -> dict | tuple[Lifting]:

        if "Transition" in self.oil_lifting_data.type_of_contract:
            pass

        else:
            condensate_lifting_aggr = tuple(
                [
                    Lifting(
                        start_year=self.general_config_data.start_date_project.year,
                        end_year=self.general_config_data.end_date_project.year,
                        lifting_rate=self.oil_lifting_data.condensate_lifting_rate[ws],
                        price=self.oil_lifting_data.condensate_price[ws],
                        prod_year=self.oil_lifting_data.prod_year[ws],
                        fluid_type=FluidType.OIL,
                    ) for ws in self.oil_lifting_data.prod_year.keys()
                ]
            )

        print('\t')
        print(f'Filetype: {type(condensate_lifting_aggr)}')
        print('condensate_lifting_aggr = \n', condensate_lifting_aggr)

    def _get_gas_lifting_aggregate(self):
        pass

    def _get_lpg_propane_lifting_aggregare(self):
        pass

    def _get_lpg_butane_lifting_aggregate(self):
        pass

    def _get_sulfur_lifting_aggregate(self):
        pass

    def _get_electricity_lifting_aggregate(self):
        pass

    def _get_co2_lifting_aggregate(self):
        pass

    def fit(self):
        self.oil_lifting_aggregate = self._get_oil_lifting_aggregate()
        self.conden

        print('\t')
        print(f'Filetype: {type(self.oil_lifting_aggregate)}')
        print('oil_lifting_aggregate = \n', self.oil_lifting_aggregate)

