"""
Create aggregate of Lifting and Cost instances to be passed on to the main executable.
"""

import os as os
from dataclasses import dataclass, field

from pyscnomics.io.spreadsheet import Spreadsheet
from pyscnomics.econ.selection import FluidType
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.costs import Tangible, Intangible, OPEX, ASR


class SpreadsheetException(Exception):
    """ Exception to raise for a misuse of Spreadsheet class """

    pass


class AggregateException(Exception):
    """ Exception to raise for a misuse of Aggregate class """

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
    gas_lifting_aggregate: dict | tuple[Lifting] = field(default=None, init=False)
    lpg_propane_lifting_aggregate: dict | tuple[Lifting] = field(default=None, init=False)
    lpg_butane_lifting_aggregate: dict | tuple[Lifting] = field(default=None, init=False)
    sulfur_lifting_aggregate: dict | tuple[Lifting] = field(default=None, init=False)
    electricity_lifting_aggregate: dict | tuple[Lifting] = field(default=None, init=False)
    co2_lifting_aggregate: dict | tuple[Lifting] = field(default=None, init=False)

    # Attributes associated with aggregates of cost data
    tangible_cost_aggregate: dict | tuple[Tangible] = field(default=None, init=False)
    intangible_cost_aggregate: dict | tuple[Intangible] = field(default=None, init=False)
    opex_aggregate: dict | tuple[OPEX] = field(default=None, init=False)
    asr_cost_aggregate: dict | tuple[ASR] = field(default=None, init=False)

    # Attributes associated with aggregate of total lifting data
    oil_lifting_aggregate_total: dict | tuple[Lifting] = field(default=None, init=False)
    gas_lifting_aggregate_total: dict | tuple[Lifting] = field(default=None, init=False)

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
            -   If the type_of_contract is "Transition", returns a dictionary with
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
        """
        Retrieves the condensate lifting aggregate based on the Production
        Sharing Contract (PSC) type.

        Returns
        -------
        Dict[str, Tuple[Lifting, ...]], Tuple[Lifting, ...]:
            -   If the type_of_contract is 'Transition', returns a dictionary with
                PSC regimes as keys and a tuple of Lifting instances as values.
            -   If the type_of_contract is a single PSC (CR or GS), returns a tuple
                of Lifting instances.

        Notes
        -----
        (1) For single PSC case, the aggregate of condensate lifting data is generated using
            a tuple comprehension of all available condensate lifting data stored in
            parameter 'self.oil_lifting_data',
        (2) For PSC transition case, the aggregate of condensate lifting data is stored in
            a dictionary with keys: ['PSC 1', 'PSC 2']. The value of each keys is a
            tuple of all available condensate lifting data stored in parameter 'self.oil_lifting_data'
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

            condensate_lifting_aggr = {
                psc: tuple(
                    [
                        Lifting(
                            start_year=start_year_combined[i],
                            end_year=end_year_combined[i],
                            lifting_rate=self.oil_lifting_data.condensate_lifting_rate[ws][psc],
                            price=self.oil_lifting_data.condensate_price[ws][psc],
                            prod_year=self.oil_lifting_data.prod_year[ws][psc],
                            fluid_type=FluidType.OIL,
                        )
                        for ws in self.oil_lifting_data.prod_year.keys()
                    ]
                )
                for i, psc in enumerate(self.psc_regimes)
            }

        # For single PSC (CR or GS)
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

        return condensate_lifting_aggr

    def _get_gas_lifting_aggregate(self) -> dict | tuple[Lifting]:
        """
        Retrieves the gas lifting aggregate based on the Production Sharing Contract (PSC) type.

        Returns
        -------
        Dict[str, Tuple[Lifting, ...]], Tuple[Lifting, ...]:
            -   If the type_of_contract is "Transition," returns a dictionary with
                PSC regimes as keys and a tuple of Lifting instances as values.
            -   If the type_of_contract is a single PSC (CR or GS), returns a tuple
                of Lifting instances.

        Notes
        -----
        (1) For single PSC case, the aggregate of gas lifting data is generated using
            a tuple comprehension of all available gas lifting data stored in
            parameter 'self.gas_lifting_data',
        (2) For PSC transition case, the aggregate of gas lifting data is stored in
            a dictionary with keys: ['PSC 1', 'PSC 2']. The value of each keys is a
            tuple of all available gas lifting data stored in parameter 'self.gas_lifting_data'
            for each corresponding PSC regimes.
        """
        gsa_number = self.general_config_data.gsa_number
        gsa_variables = ["GSA {0}".format(i + 1) for i in range(gsa_number)]

        # For PSC transition
        if "Transition" in self.gas_lifting_data.type_of_contract:
            start_year_combined = [
                self.general_config_data.start_date_project.year,
                self.general_config_data.start_date_project_second.year,
            ]

            end_year_combined = [
                self.general_config_data.end_date_project.year,
                self.general_config_data.end_date_project_second.year,
            ]

            gas_lifting_aggr = {
                psc: tuple(
                    [
                        Lifting(
                            start_year=start_year_combined[i],
                            end_year=end_year_combined[i],
                            lifting_rate=self.gas_lifting_data.lifting_rate[ws][gsa][psc],
                            price=self.gas_lifting_data.price[ws][gsa][psc],
                            prod_year=self.gas_lifting_data.prod_year[ws][psc],
                            fluid_type=FluidType.GAS,
                            ghv=self.gas_lifting_data.ghv[ws][gsa][psc],
                            prod_rate=self.gas_lifting_data.prod_rate[ws][psc],
                        )
                        for ws in self.gas_lifting_data.prod_year.keys()
                        for gsa in gsa_variables
                    ]
                )
                for i, psc in enumerate(self.psc_regimes)
            }

        # For single PSC (CR or GS)
        else:
            gas_lifting_aggr = tuple(
                [
                    Lifting(
                        start_year=self.general_config_data.start_date_project.year,
                        end_year=self.general_config_data.end_date_project.year,
                        lifting_rate=self.gas_lifting_data.lifting_rate[ws][gsa],
                        price=self.gas_lifting_data.price[ws][gsa],
                        prod_year=self.gas_lifting_data.prod_year[ws],
                        fluid_type=FluidType.GAS,
                        ghv=self.gas_lifting_data.ghv[ws][gsa],
                        prod_rate=self.gas_lifting_data.prod_rate[ws],
                    )
                    for ws in self.gas_lifting_data.prod_year.keys()
                    for gsa in gsa_variables
                ]
            )

        return gas_lifting_aggr

    def _get_lpg_propane_lifting_aggregate(self) -> dict | tuple[Lifting]:
        """
        Retrieves the LPG propane lifting aggregate based on the Production
        Sharing Contract (PSC) type.

        Returns
        -------
        Dict[str, Tuple[Lifting, ...]], Tuple[Lifting, ...]:
            -   If the type_of_contract is 'Transition', returns a dictionary with
                PSC regimes as keys and a tuple of Lifting instances as values.
            -   If the type_of_contract is a single PSC (CR or GS), returns a tuple
                of Lifting instances.

        Notes
        -----
        (1) For a single PSC case, the aggregate of LPG propane lifting data is generated using
            a tuple comprehension of all available LPG propane lifting data stored in
            parameter 'self.lpg_propane_lifting_data'.
        (2) For a PSC transition case, the aggregate of LPG propane lifting data is stored in
            a dictionary with keys: ['PSC 1', 'PSC 2']. The value of each key is a
            tuple of all available LPG propane lifting data stored in parameter
            'self.lpg_propane_lifting_data' for each corresponding PSC regime.
        """
        # For PSC transition
        if "Transition" in self.lpg_propane_lifting_data.type_of_contract:
            start_year_combined = [
                self.general_config_data.start_date_project.year,
                self.general_config_data.start_date_project_second.year,
            ]

            end_year_combined = [
                self.general_config_data.end_date_project.year,
                self.general_config_data.end_date_project_second.year,
            ]

            lpg_propane_lifting_aggr = {
                psc: tuple(
                    [
                        Lifting(
                            start_year=start_year_combined[i],
                            end_year=end_year_combined[i],
                            lifting_rate=self.lpg_propane_lifting_data.lifting_rate[ws][psc],
                            price=self.lpg_propane_lifting_data.price[ws][psc],
                            prod_year=self.lpg_propane_lifting_data.prod_year[ws][psc],
                            fluid_type=FluidType.GAS,
                        )
                        for ws in self.lpg_propane_lifting_data.prod_year.keys()
                    ]
                )
                for i, psc in enumerate(self.psc_regimes)
            }

        # For single PSC (CR or GS)
        else:
            lpg_propane_lifting_aggr = tuple(
                [
                    Lifting(
                        start_year=self.general_config_data.start_date_project.year,
                        end_year=self.general_config_data.end_date_project.year,
                        lifting_rate=self.lpg_propane_lifting_data.lifting_rate[ws],
                        price=self.lpg_propane_lifting_data.price[ws],
                        prod_year=self.lpg_propane_lifting_data.prod_year[ws],
                        fluid_type=FluidType.GAS,
                    )
                    for ws in self.lpg_propane_lifting_data.prod_year.keys()
                ]
            )

        return lpg_propane_lifting_aggr

    def _get_lpg_butane_lifting_aggregate(self) -> dict | tuple[Lifting]:
        """
        Retrieves the LPG butane lifting aggregate based on the Production
        Sharing Contract (PSC) type.

        Returns
        -------
        Dict[str, Tuple[Lifting, ...]], Tuple[Lifting, ...]:
            -   If the type_of_contract is 'Transition', returns a dictionary with
                PSC regimes as keys and a tuple of Lifting instances as values.
            -   If the type_of_contract is a single PSC (CR or GS), returns a tuple
                of Lifting instances.

        Notes
        -----
        (1) For a single PSC case, the aggregate of LPG butane lifting data is generated using
            a tuple comprehension of all available LPG butane lifting data stored in
            parameter 'self.lpg_butane_lifting_data'.
        (2) For a PSC transition case, the aggregate of LPG butane lifting data is stored in
            a dictionary with keys: ['PSC 1', 'PSC 2']. The value of each key is a
            tuple of all available LPG butane lifting data stored in parameter
            'self.lpg_butane_lifting_data' for each corresponding PSC regime.
        """
        # For PSC transition
        if "Transition" in self.lpg_butane_lifting_data.type_of_contract:
            start_year_combined = [
                self.general_config_data.start_date_project.year,
                self.general_config_data.start_date_project_second.year,
            ]

            end_year_combined = [
                self.general_config_data.end_date_project.year,
                self.general_config_data.end_date_project_second.year,
            ]

            lpg_butane_lifting_aggr = {
                psc: tuple(
                    [
                        Lifting(
                            start_year=start_year_combined[i],
                            end_year=end_year_combined[i],
                            lifting_rate=self.lpg_butane_lifting_data.lifting_rate[ws][psc],
                            price=self.lpg_butane_lifting_data.price[ws][psc],
                            prod_year=self.lpg_butane_lifting_data.prod_year[ws][psc],
                            fluid_type=FluidType.GAS,
                        )
                        for ws in self.lpg_butane_lifting_data.prod_year.keys()
                    ]
                )
                for i, psc in enumerate(self.psc_regimes)
            }

        # For single PSC (CR or GS)
        else:
            lpg_butane_lifting_aggr = tuple(
                [
                    Lifting(
                        start_year=self.general_config_data.start_date_project.year,
                        end_year=self.general_config_data.end_date_project.year,
                        lifting_rate=self.lpg_butane_lifting_data.lifting_rate[ws],
                        price=self.lpg_butane_lifting_data.price[ws],
                        prod_year=self.lpg_butane_lifting_data.prod_year[ws],
                        fluid_type=FluidType.GAS,
                    )
                    for ws in self.lpg_butane_lifting_data.prod_year.keys()
                ]
            )

        return lpg_butane_lifting_aggr

    def _get_sulfur_lifting_aggregate(self) -> dict | tuple[Lifting]:
        """
        Retrieves the sulfur lifting aggregate based on the Production
        Sharing Contract (PSC) type.

        Returns
        -------
        Dict[str, Tuple[Lifting, ...]], Tuple[Lifting, ...]:
            -   If the type_of_contract is 'Transition', returns a dictionary with
                PSC regimes as keys and a tuple of Lifting instances as values.
            -   If the type_of_contract is a single PSC (CR or GS), returns a tuple
                of Lifting instances.

        Notes
        -----
        (1) For a single PSC case, the aggregate of sulfur lifting data is generated using
            a tuple comprehension of all available sulfur lifting data stored in
            parameter 'self.sulfur_lifting_data'.
        (2) For a PSC transition case, the aggregate of sulfur lifting data is stored in
            a dictionary with keys: ['PSC 1', 'PSC 2']. The value of each key is a
            tuple of all available sulfur lifting data stored in parameter
            'self.sulfur_lifting_data' for each corresponding PSC regime.
        """
        # For PSC transition
        if "Transition" in self.sulfur_lifting_data.type_of_contract:
            start_year_combined = [
                self.general_config_data.start_date_project.year,
                self.general_config_data.start_date_project_second.year,
            ]

            end_year_combined = [
                self.general_config_data.end_date_project.year,
                self.general_config_data.end_date_project_second.year,
            ]

            sulfur_lifting_aggr = {
                psc: tuple(
                    [
                        Lifting(
                            start_year=start_year_combined[i],
                            end_year=end_year_combined[i],
                            lifting_rate=self.sulfur_lifting_data.lifting_rate[ws][psc],
                            price=self.sulfur_lifting_data.price[ws][psc],
                            prod_year=self.sulfur_lifting_data.prod_year[ws][psc],
                            fluid_type=FluidType.SULFUR,
                        )
                        for ws in self.sulfur_lifting_data.prod_year.keys()
                    ]
                )
                for i, psc in enumerate(self.psc_regimes)
            }

        # For single PSC (CR or GS)
        else:
            sulfur_lifting_aggr = tuple(
                [
                    Lifting(
                        start_year=self.general_config_data.start_date_project.year,
                        end_year=self.general_config_data.end_date_project.year,
                        lifting_rate=self.sulfur_lifting_data.lifting_rate[ws],
                        price=self.sulfur_lifting_data.price[ws],
                        prod_year=self.sulfur_lifting_data.prod_year[ws],
                        fluid_type=FluidType.SULFUR,
                    )
                    for ws in self.sulfur_lifting_data.prod_year.keys()
                ]
            )

        return sulfur_lifting_aggr

    def _get_electricity_lifting_aggregate(self) -> dict | tuple[Lifting]:
        """
        Retrieves the electricity lifting aggregate based on the Production
        Sharing Contract (PSC) type.

        Returns
        -------
        Dict[str, Tuple[Lifting, ...]], Tuple[Lifting, ...]:
            -   If the type_of_contract is 'Transition', returns a dictionary with
                PSC regimes as keys and a tuple of Lifting instances as values.
            -   If the type_of_contract is a single PSC (CR or GS), returns a tuple
                of Lifting instances.

        Notes
        -----
        (1) For a single PSC case, the aggregate of electricity lifting data is generated using
            a tuple comprehension of all available electricity lifting data stored in
            parameter 'self.electricity_lifting_data'.
        (2) For a PSC transition case, the aggregate of electricity lifting data is stored in
            a dictionary with keys: ['PSC 1', 'PSC 2']. The value of each key is a
            tuple of all available electricity lifting data stored in parameter
            'self.electricity_lifting_data' for each corresponding PSC regime.
        """
        # For PSC transition
        if "Transition" in self.electricity_lifting_data.type_of_contract:
            start_year_combined = [
                self.general_config_data.start_date_project.year,
                self.general_config_data.start_date_project_second.year,
            ]

            end_year_combined = [
                self.general_config_data.end_date_project.year,
                self.general_config_data.end_date_project_second.year,
            ]

            electricity_lifting_aggr = {
                psc: tuple(
                    [
                        Lifting(
                            start_year=start_year_combined[i],
                            end_year=end_year_combined[i],
                            lifting_rate=self.electricity_lifting_data.lifting_rate[ws][psc],
                            price=self.electricity_lifting_data.price[ws][psc],
                            prod_year=self.electricity_lifting_data.prod_year[ws][psc],
                            fluid_type=FluidType.ELECTRICITY,
                        )
                        for ws in self.electricity_lifting_data.prod_year.keys()
                    ]
                )
                for i, psc in enumerate(self.psc_regimes)
            }

        # For single PSC (CR or GS)
        else:
            electricity_lifting_aggr = tuple(
                [
                    Lifting(
                        start_year=self.general_config_data.start_date_project.year,
                        end_year=self.general_config_data.end_date_project.year,
                        lifting_rate=self.electricity_lifting_data.lifting_rate[ws],
                        price=self.electricity_lifting_data.price[ws],
                        prod_year=self.electricity_lifting_data.prod_year[ws],
                        fluid_type=FluidType.ELECTRICITY,
                    )
                    for ws in self.electricity_lifting_data.prod_year.keys()
                ]
            )

        return electricity_lifting_aggr

    def _get_co2_lifting_aggregate(self) -> dict | tuple[Lifting]:
        """
        Retrieves the co2 lifting aggregate based on the Production
        Sharing Contract (PSC) type.

        Returns
        -------
        Dict[str, Tuple[Lifting, ...]], Tuple[Lifting, ...]:
            -   If the type_of_contract is 'Transition', returns a dictionary with
                PSC regimes as keys and a tuple of Lifting instances as values.
            -   If the type_of_contract is a single PSC (CR or GS), returns a tuple
                of Lifting instances.

        Notes
        -----
        (1) For a single PSC case, the aggregate of co2 lifting data is generated using
            a tuple comprehension of all available co2 lifting data stored in
            parameter 'self.co2_lifting_data'.
        (2) For a PSC transition case, the aggregate of co2 lifting data is stored in
            a dictionary with keys: ['PSC 1', 'PSC 2']. The value of each key is a
            tuple of all available co2 lifting data stored in parameter
            'self.co2_lifting_data' for each corresponding PSC regime.
        """
        # For PSC transition
        if "Transition" in self.co2_lifting_data.type_of_contract:
            start_year_combined = [
                self.general_config_data.start_date_project.year,
                self.general_config_data.start_date_project_second.year,
            ]

            end_year_combined = [
                self.general_config_data.end_date_project.year,
                self.general_config_data.end_date_project_second.year,
            ]

            co2_lifting_aggr = {
                psc: tuple(
                    [
                        Lifting(
                            start_year=start_year_combined[i],
                            end_year=end_year_combined[i],
                            lifting_rate=self.co2_lifting_data.lifting_rate[ws][psc],
                            price=self.co2_lifting_data.price[ws][psc],
                            prod_year=self.co2_lifting_data.prod_year[ws][psc],
                            fluid_type=FluidType.CO2,
                        )
                        for ws in self.co2_lifting_data.prod_year.keys()
                    ]
                )
                for i, psc in enumerate(self.psc_regimes)
            }

        # For single PSC (CR or GS)
        else:
            co2_lifting_aggr = tuple(
                [
                    Lifting(
                        start_year=self.general_config_data.start_date_project.year,
                        end_year=self.general_config_data.end_date_project.year,
                        lifting_rate=self.co2_lifting_data.lifting_rate[ws],
                        price=self.co2_lifting_data.price[ws],
                        prod_year=self.co2_lifting_data.prod_year[ws],
                        fluid_type=FluidType.CO2,
                    )
                    for ws in self.co2_lifting_data.prod_year.keys()
                ]
            )

        return co2_lifting_aggr

    def _get_oil_lifting_aggregate_total(
        self,
        oil_lifting_aggregate: dict | tuple,
        condensate_lifting_aggregate: dict | tuple,
    ) -> dict | tuple[Lifting]:
        """
        Calculate the total oil lifting aggregate for a Production Sharing Contract (PSC).

        Parameters
        ----------
        oil_lifting_aggregate: dict | tuple
            Dictionary or tuple containing oil lifting data.
        condensate_lifting_aggregate: dict | tuple
            Dictionary or tuple containing condensate lifting data.

        Returns
        -------
        -   If the type_of_contract is 'Transition', returns a dictionary with
            PSC regimes as keys and a tuple of Lifting instances as values.
        -   If the type_of_contract is a single PSC (CR or GS), returns a tuple
            of Lifting instances.

        Notes
        -----
        -   For PSC transition, the function sums the oil and condensate lifting
            aggregates for each PSC regime.
        -   For a single PSC (CR or GS), the function simply concatenates the oil
            and condensate lifting aggregates.
        """
        # For PSC transition
        if "Transition" in self.oil_lifting_data.type_of_contract:
            oil_lifting_aggr_tot = {
                psc: oil_lifting_aggregate[psc] + condensate_lifting_aggregate[psc]
                for psc in self.psc_regimes
            }

        # For single PSC (CR or GS)
        else:
            oil_lifting_aggr_tot = oil_lifting_aggregate + condensate_lifting_aggregate

        return oil_lifting_aggr_tot

    def _get_gas_lifting_aggregate_total(
        self,
        gas_lifting_aggregate: dict | tuple,
        lpg_propane_lifting_aggregate: dict | tuple,
        lpg_butane_lifting_aggregate: dict | tuple,
    ) -> dict | tuple | AggregateException:
        """
        Calculate the total gas lifting aggregate for a Production Sharing Contract (PSC).

        Parameters
        ----------
        gas_lifting_aggregate: dict | tuple
            Dictionary or tuple containing gas lifting data.
        lpg_propane_lifting_aggregate: dict | tuple
            Dictionary or tuple containing LPG propane lifting data.
        lpg_butane_lifting_aggregate: dict | tuple
            Dictionary or tuple containing LPG butane lifting data.

        Returns
        -------
        dict | tuple | AggregateException
            -   If all contracts are transition (contain "Transition" in their type),
                returns a dictionary representing the total gas lifting aggregate for
                each PSC regime.
            -   If all contracts are single PSCs (CR or GS, without "Transition" in their type),
                returns a tuple containing the aggregated gas lifting data.
            -   If contracts mix both transition and non-transition types,
                returns an Exception.

        Notes
        -----
        -   For PSC transition, the function sums the gas, LPG propane, and LPG butane
            lifting aggregates for each PSC regime.
        -   For a single PSC (CR or GS), the function simply concatenates the gas,
            LPG propane, and LPG butane lifting aggregates.
        """
        contracts = [
            self.gas_lifting_data.type_of_contract,
            self.lpg_propane_lifting_data.type_of_contract,
            self.lpg_butane_lifting_data.type_of_contract,
        ]

        # For PSC transition
        if all(["Transition" in i for i in contracts]):
            return {
                psc: (
                    gas_lifting_aggregate[psc]
                    + lpg_propane_lifting_aggregate[psc]
                    + lpg_butane_lifting_aggregate[psc]
                )
                for psc in self.psc_regimes
            }

        # For single PSC (CR or GS)
        elif all(["Transition" not in i for i in contracts]):
            return (
                gas_lifting_aggregate
                + lpg_propane_lifting_aggregate
                + lpg_butane_lifting_aggregate
            )

        else:
            return AggregateException(
                f"Unequal type of PSC contracts for Gas, LPG propane, and LPG butane. "
                f"Gas: {self.gas_lifting_data.type_of_contract}, "
                f"LPG Propane: {self.lpg_propane_lifting_data.type_of_contract}, "
                f"LPG Butane: {self.lpg_butane_lifting_data.type_of_contract}."
            )

    def _get_tangible_cost_aggregate(self) -> dict | tuple[Tangible]:
        """
        Retrieves the tangible cost aggregate based on the Production
        Sharing Contract (PSC) type.

        Returns
        -------
        -   If PSC transition: dict
                A dictionary containing PSC regimes as keys and tuples of Tangible objects as values.
        -   If single PSC (CR or GS): tuple[Tangible]
                A tuple containing a single Tangible object.

        Notes
        -----
        For a PSC transition case, the aggregate of tangible cost data is stored in
        a dictionary with keys: ['PSC 1', 'PSC 2']. The value of each key is a tuple
        of tangible cost data stored in parameter 'self.tangible_cost_data' for each
        corresponding PSC regime.
        """
        # For PSC transition
        if "Transition" in self.tangible_cost_data.type_of_contract:
            start_year_combined = [
                self.general_config_data.start_date_project.year,
                self.general_config_data.start_date_project_second.year,
            ]

            end_year_combined = [
                self.general_config_data.end_date_project.year,
                self.general_config_data.end_date_project_second.year,
            ]

            tangible_cost_aggr = {
                psc: tuple(
                    [
                        Tangible(
                            start_year=start_year_combined[i],
                            end_year=end_year_combined[i],
                            cost=self.tangible_cost_data.cost[psc],
                            expense_year=self.tangible_cost_data.expense_year[psc],
                            cost_allocation=self.tangible_cost_data.cost_allocation[psc],
                            vat_portion=self.tangible_cost_data.vat_portion[psc],
                            vat_discount=self.fiscal_config_data.vat_discount[psc],
                            lbt_portion=self.tangible_cost_data.lbt_portion[psc],
                            lbt_discount=self.fiscal_config_data.lbt_discount[psc],
                            description=self.tangible_cost_data.description[psc],
                            pis_year=self.tangible_cost_data.pis_year[psc],
                            salvage_value=self.tangible_cost_data.salvage_value[psc],
                            useful_life=self.tangible_cost_data.useful_life[psc],
                            depreciation_factor=self.tangible_cost_data.depreciation_factor[psc],
                            is_ic_applied=self.tangible_cost_data.is_ic_applied[psc],
                        )
                    ]
                )
                for i, psc in enumerate(self.psc_regimes)
            }

        # For single PSC (CR or GS)
        else:
            tangible_cost_aggr = tuple(
                [
                    Tangible(
                        start_year=self.general_config_data.start_date_project.year,
                        end_year=self.general_config_data.end_date_project.year,
                        cost=self.tangible_cost_data.cost,
                        expense_year=self.tangible_cost_data.expense_year,
                        cost_allocation=self.tangible_cost_data.cost_allocation,
                        vat_portion=self.tangible_cost_data.vat_portion,
                        vat_discount=self.fiscal_config_data.vat_discount,
                        lbt_portion=self.tangible_cost_data.lbt_portion,
                        lbt_discount=self.fiscal_config_data.lbt_discount,
                        description=self.tangible_cost_data.description,
                        pis_year=self.tangible_cost_data.pis_year,
                        salvage_value=self.tangible_cost_data.salvage_value,
                        useful_life=self.tangible_cost_data.useful_life,
                        depreciation_factor=self.tangible_cost_data.depreciation_factor,
                        is_ic_applied=self.tangible_cost_data.is_ic_applied,
                    )
                ]
            )

        return tangible_cost_aggr

    def _get_intangible_cost_aggregate(self) -> dict | tuple[Intangible]:
        """
        Retrieves the intangible cost aggregate based on the Production
        Sharing Contract (PSC) type.

        Returns
        -------
        -   If PSC transition: dict
                A dictionary containing PSC regimes as keys and tuples of Intangible objects as values.
        -   If single PSC (CR or GS): tuple[Intangible]
                A tuple containing a single Intangible object.

        Notes
        -----
        For a PSC transition case, the aggregate of intangible cost data is stored in
        a dictionary with keys: ['PSC 1', 'PSC 2']. The value of each key is a tuple
        of intangible cost data stored in parameter 'self.intangible_cost_data' for each
        corresponding PSC regime.
        """
        # For PSC transition
        if "Transition" in self.intangible_cost_data.type_of_contract:
            start_year_combined = [
                self.general_config_data.start_date_project.year,
                self.general_config_data.start_date_project_second.year,
            ]

            end_year_combined = [
                self.general_config_data.end_date_project.year,
                self.general_config_data.end_date_project_second.year,
            ]

            intangible_cost_aggr = {
                psc: tuple(
                    [
                        Intangible(
                            start_year=start_year_combined[i],
                            end_year=end_year_combined[i],
                            cost=self.intangible_cost_data.cost[psc],
                            expense_year=self.intangible_cost_data.expense_year[psc],
                            cost_allocation=self.intangible_cost_data.cost_allocation[psc],
                            description=self.intangible_cost_data.description[psc],
                            vat_portion=self.intangible_cost_data.vat_portion[psc],
                            vat_discount=self.fiscal_config_data.vat_discount[psc],
                            lbt_portion=self.intangible_cost_data.lbt_portion[psc],
                            lbt_discount=self.fiscal_config_data.lbt_discount[psc],
                        )
                    ]
                )
                for i, psc in enumerate(self.psc_regimes)
            }

        # For single PSC (CR or GS)
        else:
            intangible_cost_aggr = tuple(
                [
                    Intangible(
                        start_year=self.general_config_data.start_date_project.year,
                        end_year=self.general_config_data.end_date_project.year,
                        cost=self.intangible_cost_data.cost,
                        expense_year=self.intangible_cost_data.expense_year,
                        cost_allocation=self.intangible_cost_data.cost_allocation,
                        description=self.intangible_cost_data.description,
                        vat_portion=self.intangible_cost_data.vat_portion,
                        vat_discount=self.fiscal_config_data.vat_discount,
                        lbt_portion=self.intangible_cost_data.lbt_portion,
                        lbt_discount=self.fiscal_config_data.lbt_discount,
                    )
                ]
            )

        return intangible_cost_aggr

    def _get_opex_aggregate(self) -> dict | tuple[OPEX]:
        """
        Retrieves the OPEX aggregate based on the Production
        Sharing Contract (PSC) type.

        Returns
        -------
        -   If PSC transition: dict
                A dictionary containing PSC regimes as keys and tuples of OPEX objects as values.
        -   If single PSC (CR or GS): tuple[OPEX]
                A tuple containing a single OPEX object.

        Notes
        -----
        For a PSC transition case, the aggregate of OPEX data is stored in
        a dictionary with keys: ['PSC 1', 'PSC 2']. The value of each key is a tuple
        of OPEX data stored in parameter 'self.opex_data' for each
        corresponding PSC regime.
        """
        # For PSC transition
        if "Transition" in self.opex_data.type_of_contract:
            start_year_combined = [
                self.general_config_data.start_date_project.year,
                self.general_config_data.start_date_project_second.year,
            ]

            end_year_combined = [
                self.general_config_data.end_date_project.year,
                self.general_config_data.end_date_project_second.year,
            ]

            opex_aggr = {
                psc: tuple(
                    [
                        OPEX(
                            start_year=start_year_combined[i],
                            end_year=end_year_combined[i],
                            expense_year=self.opex_data.expense_year[psc],
                            cost_allocation=self.opex_data.cost_allocation[psc],
                            description=self.opex_data.description[psc],
                            vat_portion=self.opex_data.vat_portion[psc],
                            vat_discount=self.fiscal_config_data.vat_discount[psc],
                            lbt_portion=self.opex_data.lbt_portion[psc],
                            lbt_discount=self.fiscal_config_data.lbt_discount[psc],
                            fixed_cost=self.opex_data.fixed_cost[psc],
                            prod_rate=self.opex_data.prod_rate[psc],
                            cost_per_volume=self.opex_data.cost_per_volume[psc],
                        )
                    ]
                )
                for i, psc in enumerate(self.psc_regimes)
            }

        # For single PSC (CR or GS)
        else:
            opex_aggr = tuple(
                [
                    OPEX(
                        start_year=self.general_config_data.start_date_project.year,
                        end_year=self.general_config_data.end_date_project.year,
                        expense_year=self.opex_data.expense_year,
                        cost_allocation=self.opex_data.cost_allocation,
                        description=self.opex_data.description,
                        vat_portion=self.opex_data.vat_portion,
                        vat_discount=self.fiscal_config_data.vat_discount,
                        lbt_portion=self.opex_data.lbt_portion,
                        lbt_discount=self.fiscal_config_data.lbt_discount,
                        fixed_cost=self.opex_data.fixed_cost,
                        prod_rate=self.opex_data.prod_rate,
                        cost_per_volume=self.opex_data.cost_per_volume,
                    )
                ]
            )

        return opex_aggr

    def _get_asr_cost_aggregate(self) -> dict | tuple[ASR]:
        """
        Retrieves the ASR cost aggregate based on the Production
        Sharing Contract (PSC) type.

        Returns
        -------
        -   If PSC transition: dict
                A dictionary containing PSC regimes as keys and tuples of ASR objects as values.
        -   If single PSC (CR or GS): tuple[ASR]
                A tuple containing a single ASR object.

        Notes
        -----
        For a PSC transition case, the aggregate of ASR cost data is stored in
        a dictionary with keys: ['PSC 1', 'PSC 2']. The value of each key is a tuple
        of ASR cost data stored in parameter 'self.asr_cost_data' for each
        corresponding PSC regime.
        """
        # For PSC transition
        if "Transition" in self.asr_cost_data.type_of_contract:
            start_year_combined = [
                self.general_config_data.start_date_project.year,
                self.general_config_data.start_date_project_second.year,
            ]

            end_year_combined = [
                self.general_config_data.end_date_project.year,
                self.general_config_data.end_date_project_second.year,
            ]

            asr_cost_aggr = {
                psc: tuple(
                    [
                        ASR(
                            start_year=start_year_combined[i],
                            end_year=end_year_combined[i],
                            cost=self.asr_cost_data.cost[psc],
                            expense_year=self.asr_cost_data.expense_year[psc],
                            cost_allocation=self.asr_cost_data.cost_allocation[psc],
                            description=self.asr_cost_data.description[psc],
                            vat_portion=self.asr_cost_data.vat_portion[psc],
                            vat_discount=self.fiscal_config_data.vat_discount[psc],
                            lbt_portion=self.asr_cost_data.lbt_portion[psc],
                            lbt_discount=self.fiscal_config_data.lbt_discount[psc],
                        )
                    ]
                )
                for i, psc in enumerate(self.psc_regimes)
            }

        # For single PSC (CR or GS)
        else:
            asr_cost_aggr = tuple(
                [
                    ASR(
                        start_year=self.general_config_data.start_date_project.year,
                        end_year=self.general_config_data.end_date_project.year,
                        cost=self.asr_cost_data.cost,
                        expense_year=self.asr_cost_data.expense_year,
                        cost_allocation=self.asr_cost_data.cost_allocation,
                        description=self.asr_cost_data.description,
                        vat_portion=self.asr_cost_data.vat_portion,
                        vat_discount=self.fiscal_config_data.vat_discount,
                        lbt_portion=self.asr_cost_data.lbt_portion,
                        lbt_discount=self.fiscal_config_data.lbt_discount,
                    )
                ]
            )

        return asr_cost_aggr

    def fit(self) -> None:
        """
        Fit method for the Aggregate class.
        This method assigns aggregates associated with lifting data and costs data.

        Returns
        -------
        None
            This method modifies the instance attributes of Aggregate class.

        Notes
        -----
        This method serves as a comprehensive data preparation step, calling specific
        private methods to obtain and assign data based on their category.

        The attributes include lifting and costs aggregates for all available components.
        """
        # Aggregates associated with lifting data
        self.oil_lifting_aggregate = self._get_oil_lifting_aggregate()
        self.gas_lifting_aggregate = self._get_gas_lifting_aggregate()
        self.condensate_lifting_aggregate = self._get_condensate_lifting_aggregate()
        self.lpg_propane_lifting_aggregate = self._get_lpg_propane_lifting_aggregate()
        self.lpg_butane_lifting_aggregate = self._get_lpg_butane_lifting_aggregate()
        self.sulfur_lifting_aggregate = self._get_sulfur_lifting_aggregate()
        self.electricity_lifting_aggregate = self._get_electricity_lifting_aggregate()
        self.co2_lifting_aggregate = self._get_co2_lifting_aggregate()

        # Aggregates associated with total lifting data
        self.oil_lifting_aggregate_total = self._get_oil_lifting_aggregate_total(
            oil_lifting_aggregate=self.oil_lifting_aggregate,
            condensate_lifting_aggregate=self.condensate_lifting_aggregate,
        )

        self.gas_lifting_aggregate_total = self._get_gas_lifting_aggregate_total(
            gas_lifting_aggregate=self.gas_lifting_aggregate,
            lpg_propane_lifting_aggregate=self.lpg_propane_lifting_aggregate,
            lpg_butane_lifting_aggregate=self.lpg_butane_lifting_aggregate,
        )

        # Aggregates associated with costs data
        self.tangible_cost_aggregate = self._get_tangible_cost_aggregate()
        self.intangible_cost_aggregate = self._get_intangible_cost_aggregate()
        self.opex_aggregate = self._get_opex_aggregate()
        self.asr_cost_aggregate = self._get_asr_cost_aggregate()
