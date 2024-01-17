"""
Manage input-output data from and to a target Excel file.
"""

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from pyscnomics.io.config import (
    GeneralConfigData,
    FiscalConfigData,
    OilLiftingData,
    GasLiftingData,
    LPGPropaneLiftingData,
    LPGButaneLiftingData,
    SulfurLiftingData,
    ElectricityLiftingData,
    CO2LiftingData,
    TangibleCostData,
    IntangibleCostData,
    OPEXData,
    ASRCostData,
    PSCCostRecoveryData,
    PSCGrossSplitData,
    SensitivityData,
    MonteCarloData,
    OptimizationData,
)


class SpreadsheetException(Exception):
    """Exception to raise for a misuse of Spreadsheet class"""

    pass


@dataclass
class Spreadsheet:
    """
    Load and prepare data from a target Excel file.

    Attributes
    ----------
    workbook_to_read: str
        The name of the target Excel file. Must be given in '.xlsb' format.
        Defaults to None.
    directory_location: str
        The directory location of workbook.
    """

    workbook_to_read: str = field(default=None)
    directory_location: str = field(default=None)

    # Attribute associated with directory location
    load_dir: str = field(default=None, init=False)

    # Attributes associated with loading data from a target Excel file
    sheets_name: list = field(default=None, init=False, repr=False)
    sheets_loaded: list = field(default=None, init=False, repr=False)
    data_loaded: dict = field(default=None, init=False, repr=False)

    # Attributes associated with config data
    general_config_data: GeneralConfigData = field(default=None, init=False)
    fiscal_config_data: FiscalConfigData = field(default=None, init=False)

    # Attributes associated with lifting data
    oil_lifting_data: OilLiftingData = field(default=None, init=False)
    gas_lifting_data: GasLiftingData = field(default=None, init=False)
    lpg_propane_lifting_data: LPGPropaneLiftingData = field(default=None, init=False)
    lpg_butane_lifting_data: LPGButaneLiftingData = field(default=None, init=False)
    sulfur_lifting_data: SulfurLiftingData = field(default=None, init=False)
    electricity_lifting_data: ElectricityLiftingData = field(default=None, init=False)
    co2_lifting_data: CO2LiftingData = field(default=None, init=False)

    # Attributes associated with cost data
    tangible_cost_data: TangibleCostData = field(default=None, init=False)
    intangible_cost_data: IntangibleCostData = field(default=None, init=False)
    opex_data: OPEXData = field(default=None, init=False)
    asr_cost_data: ASRCostData = field(default=None, init=False)

    # Attributes associated with contract data
    psc_cr_data: PSCCostRecoveryData = field(default=None, init=False)
    psc_gs_data: PSCGrossSplitData = field(default=None, init=False)
    psc_transition_cr_to_cr: dict = field(default=None, init=False)
    psc_transition_cr_to_gs: dict = field(default=None, init=False)
    psc_transition_gs_to_gs: dict = field(default=None, init=False)
    psc_transition_gs_to_cr: dict = field(default=None, init=False)

    # Fill in the attributes associated with additional functionality
    optimization_data: OptimizationData = field(default=None, init=False)
    sensitivity_data: SensitivityData = field(default=None, init=False)
    montecarlo_data: MonteCarloData = field(default=None, init=False)

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

    def read_from_excel(self) -> None:
        """
        Reads data from a target Excel file specified by 'workbook_to_read' attribute.

        Returns
        -------
        None
            This method updates class attributes by storing information
            associated with the loaded data.

        Notes
        -----
        The core procedures are as follows:
        (1) Identify the directory location of the target Excel file,
        (2) From the target Excel file, identify the worksheets,
        (3) Load data from the target Excel file, capturing all necessary worksheets.
        """
        # Identify worksheets in the target Excel file
        excel = pd.ExcelFile(self.load_dir)
        self.sheets_name = excel.sheet_names

        # # self.sheets_name = [sh.title for sh in sheets]
        self.sheets_loaded = [
            i for i in self.sheets_name
            if i not in [
                "Cover",
                "UserGuide",
                "References",
                "ChartDATA",
                "ORETZS",
                "Summary",
                "Results Table CR",
                "Results Table GS",
                "Results Table Base Project",
                "Results Table GS (2)",
                "Results Table CR (2)",
            ]
        ]

        # Load data from all worksheets
        self.data_loaded = {
            key: pd.read_excel(
                excel,
                sheet_name=key,
                skiprows=1,
                index_col=None,
                header=None,
                engine="pyxlsb",
            )
            for key in self.sheets_loaded
        }

    def _get_general_config_data(self) -> GeneralConfigData:
        """
        Extracts and prepares general configuration data from the 'General Config' sheet.

        Returns
        -------
        GeneralConfigData
            An instance of the GeneralConfigData class containing the extracted data
            stored as its attributes.

        Notes
        -----
        The procedures are as follows:
        (1) In the associated dataframe, convert NaN values to None,
        (2) Assign data to their associated attributes,
        (3) Return a new instance of GeneralConfigData with filled attributes.
        """
        # Step #1 (See 'Notes' section in the docstring)
        general_config_data_loaded = self.data_loaded["General Config"].replace(np.nan, None)

        # Step #2 - Step #3 (See 'Notes' section in the docstring)
        type_of_contract = general_config_data_loaded.iloc[0, 2]
        discount_rate_start_year = general_config_data_loaded.iloc[1, 2]
        discount_rate = general_config_data_loaded.iloc[2, 2]
        inflation_rate_applied_to = general_config_data_loaded.iloc[3, 2]
        start_date_project = general_config_data_loaded.iloc[4, 2]
        end_date_project = general_config_data_loaded.iloc[5, 2]
        start_date_project_second = general_config_data_loaded.iloc[6, 2]
        end_date_project_second = general_config_data_loaded.iloc[7, 2]
        oil_onstream_date = general_config_data_loaded.iloc[8, 2]
        gas_onstream_date = general_config_data_loaded.iloc[9, 2]
        oil_onstream_date_second = general_config_data_loaded.iloc[10, 2]
        gas_onstream_date_second = general_config_data_loaded.iloc[11, 2]
        gsa_number = general_config_data_loaded.iloc[13, 7]

        return GeneralConfigData(
            start_date_project=start_date_project,
            end_date_project=end_date_project,
            start_date_project_second=start_date_project_second,
            end_date_project_second=end_date_project_second,
            type_of_contract=type_of_contract,
            oil_onstream_date=oil_onstream_date,
            gas_onstream_date=gas_onstream_date,
            oil_onstream_date_second=oil_onstream_date_second,
            gas_onstream_date_second=gas_onstream_date_second,
            discount_rate_start_year=discount_rate_start_year,
            discount_rate=discount_rate,
            inflation_rate_applied_to=inflation_rate_applied_to,
            gsa_number=gsa_number,
        )

    def _get_fiscal_config_data(self) -> FiscalConfigData:
        """
        Extracts and prepares fiscal configuration data from the 'Fiscal Config' sheet.

        Returns
        -------
        FiscalConfigData
            An instance of the FiscalConfigData class containing the extracted data
            stored as its attributes.

        Notes
        -----
        The procedures are as follows:
        (1) Slice the data, only capture columns that contain necessary data,
        (2) Assign data to their associated attributes,
        (3) Return a new instance of FiscalConfigData with filled attributes.
        """
        # Step #1 (See 'Notes' section in the docstring)
        fiscal_config_data_loaded = self.data_loaded["Fiscal Config"].iloc[:, 1:]

        # Step #2 (See 'Notes' section in the docstring)
        # Prepare attributes associated with tax
        tax_mode = {
            "PSC 1": fiscal_config_data_loaded.iloc[0, 1],
            "PSC 2": fiscal_config_data_loaded.iloc[118, 1],
        }

        tax_rate_init = {
            "PSC 1": fiscal_config_data_loaded.iloc[1, 1],
            "PSC 2": fiscal_config_data_loaded.iloc[119, 1],
        }

        multi_tax_init = {
            "PSC 1": {
                "year": fiscal_config_data_loaded.iloc[3:10, 0].to_numpy(),
                "rate": fiscal_config_data_loaded.iloc[3:10, 1].to_numpy(),
            },
            "PSC 2": {
                "year": fiscal_config_data_loaded.iloc[122:128, 0].to_numpy(),
                "rate": fiscal_config_data_loaded.iloc[122:128, 1].to_numpy(),
            }
        }

        tax_payment_config = {
            "PSC 1": fiscal_config_data_loaded.iloc[12, 1],
            "PSC 2": fiscal_config_data_loaded.iloc[130, 1],
        }

        # Prepare attribute associated with ASR config
        asr_future_rate = {
            "PSC 1": fiscal_config_data_loaded.iloc[15, 1],
            "PSC 2": fiscal_config_data_loaded.iloc[133, 1],
        }

        # Prepare attributes associated with depreciation config
        depreciation_method = {
            "PSC 1": fiscal_config_data_loaded.iloc[18, 1],
            "PSC 2": fiscal_config_data_loaded.iloc[136, 1],
        }

        decline_factor = {
            "PSC 1": fiscal_config_data_loaded.iloc[19, 1],
            "PSC 2": fiscal_config_data_loaded.iloc[137, 1],
        }

        # Prepare attributes associated with inflation config
        inflation_rate_mode = {
            "PSC 1": fiscal_config_data_loaded.iloc[22, 1],
            "PSC 2": fiscal_config_data_loaded.iloc[140, 1],
        }

        inflation_rate_init = {
            "PSC 1": fiscal_config_data_loaded.iloc[23, 1],
            "PSC 2": fiscal_config_data_loaded.iloc[141, 1],
        }

        multi_inflation_init = {
            "PSC 1": {
                "year": fiscal_config_data_loaded.iloc[26:46, 0].to_numpy(),
                "rate": fiscal_config_data_loaded.iloc[26:46, 1].to_numpy(),
            },
            "PSC 2": {
                "year": fiscal_config_data_loaded.iloc[144:164, 0].to_numpy(),
                "rate": fiscal_config_data_loaded.iloc[144:164, 1].to_numpy(),
            }
        }

        # Prepare attributes associated with VAT config
        vat_mode = {
            "PSC 1": fiscal_config_data_loaded.iloc[48, 1],
            "PSC 2": fiscal_config_data_loaded.iloc[166, 1],
        }

        vat_rate_init = {
            "PSC 1": fiscal_config_data_loaded.iloc[49, 1],
            "PSC 2": fiscal_config_data_loaded.iloc[167, 1],
        }

        multi_vat_init = {
            "PSC 1": {
                "year": fiscal_config_data_loaded.iloc[52:72, 0].to_numpy(),
                "rate": fiscal_config_data_loaded.iloc[52:72, 1].to_numpy(),
            },
            "PSC 2": {
                "year": fiscal_config_data_loaded.iloc[170:190, 0].to_numpy(),
                "rate": fiscal_config_data_loaded.iloc[170:190, 1].to_numpy(),
            }
        }

        # Prepare attributes associated with LBT config
        lbt_mode = {
            "PSC 1": fiscal_config_data_loaded.iloc[74, 1],
            "PSC 2": fiscal_config_data_loaded.iloc[192, 1],
        }

        lbt_rate_init = {
            "PSC 1": fiscal_config_data_loaded.iloc[75, 1],
            "PSC 2": fiscal_config_data_loaded.iloc[193, 1],
        }

        multi_lbt_init = {
            "PSC 1": {
                "year": fiscal_config_data_loaded.iloc[78:98, 0].to_numpy(),
                "rate": fiscal_config_data_loaded.iloc[78:98, 1].to_numpy(),
            },
            "PSC 2": {
                "year": fiscal_config_data_loaded.iloc[196:216, 0].to_numpy(),
                "rate": fiscal_config_data_loaded.iloc[196:216, 1].to_numpy(),
            }
        }

        # Prepare attributes associated with discount config
        vat_discount = {
            "PSC 1": fiscal_config_data_loaded.iloc[100, 1],
            "PSC 2": fiscal_config_data_loaded.iloc[218, 1],
        }

        lbt_discount = {
            "PSC 1": fiscal_config_data_loaded.iloc[101, 1],
            "PSC 2": fiscal_config_data_loaded.iloc[219, 1],
        }

        # Prepare attributes associated with NPV config
        npv_mode = fiscal_config_data_loaded.iloc[104, 1]
        discounting_mode = fiscal_config_data_loaded.iloc[105, 1]

        # Prepare attributes associated with other revenue config
        sulfur_revenue_config = {
            "PSC 1": fiscal_config_data_loaded.iloc[108, 1],
            "PSC 2": fiscal_config_data_loaded.iloc[222, 1],
        }

        electricity_revenue_config = {
            "PSC 1": fiscal_config_data_loaded.iloc[109, 1],
            "PSC 2": fiscal_config_data_loaded.iloc[223, 1],
        }

        co2_revenue_config = {
            "PSC 1": fiscal_config_data_loaded.iloc[110, 1],
            "PSC 2": fiscal_config_data_loaded.iloc[224, 1],
        }

        # Prepare attribute associated with transition config
        transferred_unrec_cost = fiscal_config_data_loaded.iloc[115, 1]

        # Prepare attribute associated with sunk cost config
        sunk_cost_reference_year = fiscal_config_data_loaded.iloc[230, 1]

        # Step #3 (See 'Notes' section in the docstring)
        return FiscalConfigData(
            tax_mode=tax_mode,
            tax_rate_init=tax_rate_init,
            multi_tax_init=multi_tax_init,
            tax_payment_config=tax_payment_config,
            asr_future_rate=asr_future_rate,
            depreciation_method=depreciation_method,
            decline_factor=decline_factor,
            inflation_rate_mode=inflation_rate_mode,
            inflation_rate_init=inflation_rate_init,
            multi_inflation_init=multi_inflation_init,
            vat_mode=vat_mode,
            vat_rate_init=vat_rate_init,
            multi_vat_init=multi_vat_init,
            lbt_mode=lbt_mode,
            lbt_rate_init=lbt_rate_init,
            multi_lbt_init=multi_lbt_init,
            vat_discount=vat_discount,
            lbt_discount=lbt_discount,
            npv_mode=npv_mode,
            discounting_mode=discounting_mode,
            sulfur_revenue_config=sulfur_revenue_config,
            electricity_revenue_config=electricity_revenue_config,
            co2_revenue_config=co2_revenue_config,
            transferred_unrec_cost=transferred_unrec_cost,
            sunk_cost_reference_year=sunk_cost_reference_year,
            project_years=self.general_config_data.project_years,
            type_of_contract=self.general_config_data.type_of_contract,
        )

    def _get_oil_lifting_data(self) -> OilLiftingData:
        """
        Retrieves oil lifting data based on available sheets.

        Returns
        -------
        OilLiftingData
            An instance of the OilLiftingData class.

        Notes
        -----
        The undertaken procedures are as follows:
        (1) Filter attribute self.sheets_loaded for sheets that contain 'Prod Oil' data,
            then assigned it as local variable named 'oil_data_available',
        (2) Load the data associated with oil_data, then store it in the variable
            named 'oil_data_loaded_init',
        (3) Undertake data cleansing: remove all rows which column 'prod_year' is NaN.
            Store the results in the variable named 'oil_data_loaded',
        (4) Create a dictionary named 'oil_data' to store the necessary data from 'oil_data_loaded',
        (5) Return an instance of OilLiftingData to store the oil data appropriately
            as its attributes.
        """
        # Step #1 (See 'Notes' section in the docstring)
        oil_data_available = list(filter(lambda i: "Prod Oil" in i, self.sheets_loaded))

        # Step #2 (See 'Notes' section in the docstring)
        oil_data_loaded_init = {ws: self.data_loaded[ws] for ws in oil_data_available}

        # Step #3 (See 'Notes' section in the docstring)
        oil_data_loaded = {
            ws: oil_data_loaded_init[ws] if oil_data_loaded_init[ws].empty
            else oil_data_loaded_init[ws][~pd.isna(oil_data_loaded_init[ws].iloc[:, 0])].copy()
            for ws in oil_data_available
        }

        # Step #4 (See 'Notes' section in the docstring)
        oil_attrs = [
            "prod_year",
            "oil_lifting_rate",
            "oil_price",
            "condensate_lifting_rate",
            "condensate_price",
        ]

        oil_data = {
            key: {
                ws: None if oil_data_loaded[ws].empty
                else oil_data_loaded[ws].iloc[:, i].to_numpy()
                for ws in oil_data_available
            }
            for i, key in enumerate(oil_attrs)
        }

        # Step #5 (See 'Notes' section in the docstring)
        return OilLiftingData(
            prod_year_init=oil_data["prod_year"],
            oil_lifting_rate=oil_data["oil_lifting_rate"],
            oil_price=oil_data["oil_price"],
            condensate_lifting_rate=oil_data["condensate_lifting_rate"],
            condensate_price=oil_data["condensate_price"],
            project_duration=self.general_config_data.project_duration,
            project_years=self.general_config_data.project_years,
            type_of_contract=self.general_config_data.type_of_contract,
            end_date_project=self.general_config_data.end_date_project,
            start_date_project_second=self.general_config_data.start_date_project_second,
        )

    def _get_gas_lifting_data(self) -> GasLiftingData:
        """
        Retrieves gas lifting data based on available sheets.

        Returns
        -------
        GasLiftingData
            An instance of GasLiftingData class.

        Notes
        -----
        The undertaken procedures are as follows:
        (1) Filter attribute self.sheets_loaded for sheets that contain 'Prod Gas' data,
            then assigned it as local variable named 'gas_data_available',
        (2) Load the data associated with gas, then store it in the variable
            named 'gas_data_loaded_init',
        (3) Undertake data cleansing: remove all rows which column 'prod_year' is NaN.
            Store the results in the variable named 'gas_data_loaded',
        (4) Create a dictionary named 'gas_data_general' to store information related to
            'prod_year' and 'prod_rate' from 'gas_data_loaded',
        (5) -   Define variable named 'gsa_id'. This variable stores information regarding
                the associated column indices in 'gas_data_loaded' that corresponds to
                'lifting_rate', 'ghv', and 'price'. Variable 'gsa_id' is a 2D array: the
                zeroth axis depicts indices of 'lifting_rate', 'ghv', and 'price' while the
                first axis manifest the number of available GSA,
            -   Create a dictionary named 'gas_data_gsa' to store information related to
                'lifting_rate', 'ghv', and 'price',
        (6) Return an instance of GasLiftingData to store the gas data appropriately
            as its attributes.
        """
        # Step #1 (See 'Notes' section in the docstring)
        gas_data_available = list(filter(lambda i: "Prod Gas" in i, self.sheets_loaded))

        # Step #2 (See 'Notes' section in the docstring)
        gas_data_loaded_init = {ws: self.data_loaded[ws] for ws in gas_data_available}

        # Step #3 (See 'Notes' section in the docstring)
        gas_data_loaded = {
            ws: gas_data_loaded_init[ws] if gas_data_loaded_init[ws].empty
            else (
                gas_data_loaded_init[ws]
                [~pd.isna(gas_data_loaded_init[ws].iloc[:, 0])]
                .copy()
            )
            for ws in gas_data_available
        }

        # Step #4 (See 'Notes' section in the docstring)
        gas_attrs_general = ["prod_year", "prod_rate"]
        gas_data_general = {
            key: {
                ws: None if gas_data_loaded[ws].empty
                else gas_data_loaded[ws].iloc[:, i].to_numpy()
                for ws in gas_data_available
            }
            for i, key in enumerate(gas_attrs_general)
        }

        # Step #5 (See 'Notes' section in the docstring)
        gsa_number = self.general_config_data.gsa_number
        gsa_variables = ["GSA {0}".format(i + 1) for i in range(gsa_number)]
        gas_attrs_gsa = ["lifting_rate", "ghv", "price"]

        gsa_id = (
            np.repeat(np.arange(len(gas_attrs_gsa))[:, np.newaxis] + 2, gsa_number, axis=1)
            + (np.arange(gsa_number) * len(gas_attrs_gsa))[np.newaxis, :]
        )

        gas_data_gsa = {
            key: {
                ws: {
                    gsa: None if gas_data_loaded[ws].empty
                    else gas_data_loaded[ws].iloc[:, gsa_id[i, j]].to_numpy()
                    for gsa, j in zip(gsa_variables, range(gsa_number))
                }
                for ws in gas_data_available
            }
            for i, key in enumerate(gas_attrs_gsa)
        }

        # Step #6 (See 'Notes' section in the docstring)
        return GasLiftingData(
            gsa_number=gsa_number,
            prod_year_init=gas_data_general["prod_year"],
            prod_rate=gas_data_general["prod_rate"],
            lifting_rate=gas_data_gsa["lifting_rate"],
            ghv=gas_data_gsa["ghv"],
            price=gas_data_gsa["price"],
            project_duration=self.general_config_data.project_duration,
            project_years=self.general_config_data.project_years,
            type_of_contract=self.general_config_data.type_of_contract,
            end_date_project=self.general_config_data.end_date_project,
            start_date_project_second=self.general_config_data.start_date_project_second,
        )

    def _get_lpg_propane_lifting_data(self) -> LPGPropaneLiftingData:
        """
        Retrieves LPG propane lifting data based on available sheets.

        Returns
        -------
        LPGPropaneLiftingData
            An instance of the LPGPropaneLiftingData class.

        Notes
        -----
        The undertaken procedures are as follows:
        (1) Filter attribute self.sheets_loaded for sheets that contain 'Prod LPG Propane'
            data, then assigned it as local variable named 'lpg_propane_data_available',
        (2) Load the data associated with lpg_propane_data, then store it in the variable
            named 'lpg_propane_data_loaded_init',
        (3) Undertake data cleansing: remove all rows which column 'prod_year' is NaN.
            Store the results in the variable named 'lpg_propane_data_loaded',
        (4) Create a dictionary named 'lpg_propane_data' to store the necessary data from
            'lpg_propane_data_loaded',
        (5) Return an instance of LPGPropaneLiftingData to store the lpg propane data
            appropriately as its attributes.
        """
        # Step #1 (See 'Notes' section in the docstring)
        lpg_propane_data_available = list(
            filter(lambda i: "Prod LPG Propane" in i, self.sheets_loaded)
        )

        # Step #2 (See 'Notes' section in the docstring)
        lpg_propane_data_loaded_init = {
            ws: self.data_loaded[ws] for ws in lpg_propane_data_available
        }

        # Step #3 (See 'Notes' section in the docstring)
        lpg_propane_data_loaded = {
            ws: lpg_propane_data_loaded_init[ws] if lpg_propane_data_loaded_init[ws].empty
            else (
                lpg_propane_data_loaded_init[ws]
                [~pd.isna(lpg_propane_data_loaded_init[ws].iloc[:, 0])]
                .copy()
            )
            for ws in lpg_propane_data_available
        }

        # Step #4 (See 'Notes' section in the docstring)
        lpg_propane_attrs = ["prod_year", "lifting_rate", "price"]

        lpg_propane_data = {
            key: {
                ws: None if lpg_propane_data_loaded[ws].empty
                else lpg_propane_data_loaded[ws].iloc[:, i].to_numpy()
                for ws in lpg_propane_data_available
            }
            for i, key in enumerate(lpg_propane_attrs)
        }

        # Step #5 (See 'Notes' section in the docstring)
        return LPGPropaneLiftingData(
            prod_year_init=lpg_propane_data["prod_year"],
            lifting_rate=lpg_propane_data["lifting_rate"],
            price=lpg_propane_data["price"],
            project_duration=self.general_config_data.project_duration,
            project_years=self.general_config_data.project_years,
            type_of_contract=self.general_config_data.type_of_contract,
            end_date_project=self.general_config_data.end_date_project,
            start_date_project_second=self.general_config_data.start_date_project_second,
        )

    def _get_lpg_butane_lifting_data(self) -> LPGButaneLiftingData:
        """
        Retrieves LPG butane lifting data based on available sheets.

        Returns
        -------
        LPGButaneLiftingData
            An instance of the LPGButaneLiftingData class.

        Notes
        -----
        The undertaken procedures are as follows:
        (1) Filter attribute self.sheets_loaded for sheets that contain 'Prod LPG Butane'
            data, then assigned it as local variable named 'lpg_butane_data_available',
        (2) Load the data associated with lpg_butane_data, then store it in the variable
            named 'lpg_butane_data_loaded_init',
        (3) Undertake data cleansing: remove all rows which column 'prod_year' is NaN.
            Store the results in the variable named 'lpg_butane_data_loaded',
        (4) Create a dictionary named 'lpg_butane_data' to store the necessary data from
            'lpg_butane_data_loaded',
        (5) Return an instance of LPGButaneLiftingData to store the lpg butane data
            appropriately as its attributes.
        """
        # Step #1 (See 'Notes' section in the docstring)
        lpg_butane_data_available = list(
            filter(lambda i: "Prod LPG Butane" in i, self.sheets_loaded)
        )

        # Step #2 (See 'Notes' section in the docstring)
        lpg_butane_data_loaded_init = {
            ws: self.data_loaded[ws] for ws in lpg_butane_data_available
        }

        # Step #3 (See 'Notes' section in the docstring)
        lpg_butane_data_loaded = {
            ws: lpg_butane_data_loaded_init[ws] if lpg_butane_data_loaded_init[ws].empty
            else (
                lpg_butane_data_loaded_init[ws]
                [~pd.isna(lpg_butane_data_loaded_init[ws].iloc[:, 0])]
                .copy()
            )
            for ws in lpg_butane_data_available
        }

        # Step #4 (See 'Notes' section in the docstring)
        lpg_butane_attrs = ["prod_year", "lifting_rate", "price"]

        lpg_butane_data = {
            key: {
                ws: None if lpg_butane_data_loaded[ws].empty
                else lpg_butane_data_loaded[ws].iloc[:, i].to_numpy()
                for ws in lpg_butane_data_available
            }
            for i, key in enumerate(lpg_butane_attrs)
        }

        # Step #5 (See 'Notes' section in the docstring)
        return LPGButaneLiftingData(
            prod_year_init=lpg_butane_data["prod_year"],
            lifting_rate=lpg_butane_data["lifting_rate"],
            price=lpg_butane_data["price"],
            project_duration=self.general_config_data.project_duration,
            project_years=self.general_config_data.project_years,
            type_of_contract=self.general_config_data.type_of_contract,
            end_date_project=self.general_config_data.end_date_project,
            start_date_project_second=self.general_config_data.start_date_project_second,
        )

    def _get_sulfur_lifting_data(self) -> SulfurLiftingData:
        """
        Retrieves sulfur lifting data based on available sheets.

        Returns
        -------
        SulfurLiftingData
            An instance of the SulfurLiftingData class.

        Notes
        -----
        The undertaken procedures are as follows:
        (1) Filter attribute self.sheets_loaded for sheets that contain 'Prod Sulfur'
            data, then assigned it as local variable named 'sulfur_data_available',
        (2) Load the data associated with sulfur data, then store it in the variable
            named 'sulfur_data_loaded_init',
        (3) Undertake data cleansing: remove all rows which column 'prod_year' is NaN.
            Store the results in the variable named 'sulfur_data_loaded',
        (4) Create a dictionary named 'sulfur_data' to store the necessary data from
            'sulfur_data_loaded',
        (5) Return an instance of SulfurLiftingData to store the sulfur data
            appropriately as its attributes.
        """
        # Step #1 (See 'Notes' section in the docstring)
        sulfur_data_available = list(filter(lambda i: "Prod Sulfur" in i, self.sheets_loaded))

        # Step #2 (See 'Notes' section in the docstring)
        sulfur_data_loaded_init = {ws: self.data_loaded[ws] for ws in sulfur_data_available}

        # Step #3 (See 'Notes' section in the docstring)
        sulfur_data_loaded = {
            ws: sulfur_data_loaded_init[ws] if sulfur_data_loaded_init[ws].empty
            else (
                sulfur_data_loaded_init[ws]
                [~pd.isna(sulfur_data_loaded_init[ws].iloc[:, 0])]
                .copy()
            )
            for ws in sulfur_data_available
        }

        # Step #4 (See 'Notes' section in the docstring)
        sulfur_data_attrs = ["prod_year", "lifting_rate", "price"]

        sulfur_data = {
            key: {
                ws: None if sulfur_data_loaded[ws].empty
                else sulfur_data_loaded[ws].iloc[:, i].to_numpy()
                for ws in sulfur_data_available
            }
            for i, key in enumerate(sulfur_data_attrs)
        }

        # Step #5 (See 'Notes' section in the docstring)
        return SulfurLiftingData(
            prod_year_init=sulfur_data["prod_year"],
            lifting_rate=sulfur_data["lifting_rate"],
            price=sulfur_data["price"],
            project_duration=self.general_config_data.project_duration,
            project_years=self.general_config_data.project_years,
            type_of_contract=self.general_config_data.type_of_contract,
            end_date_project=self.general_config_data.end_date_project,
            start_date_project_second=self.general_config_data.start_date_project_second,
        )

    def _get_electricity_lifting_data(self) -> ElectricityLiftingData:
        """
        Retrieves electricity lifting data based on available sheets.

        Returns
        -------
        ElectricityLiftingData
            An instance of the ElectricityLiftingData class.

        Notes
        -----
        The undertaken procedures are as follows:
        (1) Filter attribute self.sheets_loaded for sheets that contain 'Prod Electricity'
            data, then assigned it as local variable named 'electricity_data_available',
        (2) Load the data associated with electricity data, then store it in the variable
            named 'electricity_data_loaded_init',
        (3) Undertake data cleansing: remove all rows which column 'prod_year' is NaN.
            Store the results in the variable named 'electricity_data_loaded',
        (4) Create a dictionary named 'electricity_data' to store the necessary data from
            'electricity_data_loaded',
        (5) Return an instance of ElectricityLiftingData to store the electricity data
            appropriately as its attributes.
        """
        # Step #1 (See 'Notes' section in the docstring)
        electricity_data_available = list(filter(lambda i: "Prod Electricity" in i, self.sheets_loaded))

        # Step #2 (See 'Notes' section in the docstring)
        electricity_data_loaded_init = {
            ws: self.data_loaded[ws] for ws in electricity_data_available
        }

        # Step #3 (See 'Notes' section in the docstring)
        electricity_data_loaded = {
            ws: electricity_data_loaded_init[ws] if electricity_data_loaded_init[ws].empty
            else (
                electricity_data_loaded_init[ws]
                [~pd.isna(electricity_data_loaded_init[ws].iloc[:, 0])]
                .copy()
            )
            for ws in electricity_data_available
        }

        # Step #4 (See 'Notes' section in the docstring)
        electricity_data_attrs = ["prod_year", "lifting_rate", "price"]

        electricity_data = {
            key: {
                ws: None if electricity_data_loaded[ws].empty
                else electricity_data_loaded[ws].iloc[:, i].to_numpy()
                for ws in electricity_data_available
            }
            for i, key in enumerate(electricity_data_attrs)
        }

        # Step #5 (See 'Notes' section in the docstring)
        return ElectricityLiftingData(
            prod_year_init=electricity_data["prod_year"],
            lifting_rate=electricity_data["lifting_rate"],
            price=electricity_data["price"],
            project_duration=self.general_config_data.project_duration,
            project_years=self.general_config_data.project_years,
            type_of_contract=self.general_config_data.type_of_contract,
            end_date_project=self.general_config_data.end_date_project,
            start_date_project_second=self.general_config_data.start_date_project_second,
        )

    def _get_co2_lifting_data(self) -> CO2LiftingData:
        """
        Retrieves CO2 lifting data based on available sheets.

        Returns
        -------
        CO2LiftingData
            An instance of the CO2LiftingData class.

        Notes
        -----
        The undertaken procedures are as follows:
        (1) Filter attribute self.sheets_loaded for sheets that contain 'Prod CO2'
            data, then assigned it as local variable named 'co2_data_available',
        (2) Load the data associated with co2 data, then store it in the variable
            named 'co2_data_loaded_init',
        (3) Undertake data cleansing: remove all rows which column 'prod_year' is NaN.
            Store the results in the variable named 'co2_data_loaded',
        (4) Create a dictionary named 'co2_data' to store the necessary data from
            'co2_data_loaded',
        (5) Return an instance of CO2LiftingData to store the co2 data
            appropriately as its attributes.
        """
        # Step #1 (See 'Notes' section in the docstring)
        co2_data_available = list(filter(lambda i: "Prod CO2" in i, self.sheets_loaded))

        # Step #2 (See 'Notes' section in the docstring)
        co2_data_loaded_init = {ws: self.data_loaded[ws] for ws in co2_data_available}

        # Step #3 (See 'Notes' section in the docstring)
        co2_data_loaded = {
            ws: co2_data_loaded_init[ws] if co2_data_loaded_init[ws].empty
            else (
                co2_data_loaded_init[ws]
                [~pd.isna(co2_data_loaded_init[ws].iloc[:, 0])]
                .copy()
            )
            for ws in co2_data_available
        }

        # Step #4 (See 'Notes' section in the docstring)
        co2_data_attrs = ["prod_year", "lifting_rate", "price"]

        co2_data = {
            key: {
                ws: None if co2_data_loaded[ws].empty
                else co2_data_loaded[ws].iloc[:, i].to_numpy()
                for ws in co2_data_available
            }
            for i, key in enumerate(co2_data_attrs)
        }

        # Step #5 (See 'Notes' section in the docstring)
        return CO2LiftingData(
            prod_year_init=co2_data["prod_year"],
            lifting_rate=co2_data["lifting_rate"],
            price=co2_data["price"],
            project_duration=self.general_config_data.project_duration,
            project_years=self.general_config_data.project_years,
            type_of_contract=self.general_config_data.type_of_contract,
            end_date_project=self.general_config_data.end_date_project,
            start_date_project_second=self.general_config_data.start_date_project_second,
        )

    def _get_tangible_cost_data(self) -> TangibleCostData:
        """
        Retrieves tangible cost data based on available sheets.

        Returns
        -------
        TangibleCostData
            An instance of TangibleCostData class.

        Notes
        -----
        The core operations are as follows:
        (1) Capture tangible cost data from attribute self.data_loaded, then perform the
            necessary adjustment,
        (2) Undertake data cleansing: remove all rows which column 'expense_year' is NaN.
            Store the results in a variable named 'tangible_data_loaded',
        (3) Create a dictionary named 'tangible_data' to store the necessary data from
            'tangible_data_loaded',
        (4) Return an instance of TangibleCostData to store the tangible data appropriately
            as its attributes.
        """
        # Step #1 (See 'Notes' section in the docstring)
        tangible_data_loaded_init = self.data_loaded["Cost Tangible"].dropna(axis=0, how="all")

        # Step #2 (See 'Notes' section in the docstring)
        tangible_data_loaded = (
            tangible_data_loaded_init if tangible_data_loaded_init.empty
            else tangible_data_loaded_init[~pd.isna(tangible_data_loaded_init.iloc[:, 0])].copy()
        )

        # Step #3 (See 'Notes' section in the docstring)
        tangible_data_attrs = [
            "expense_year",
            "cost_allocation",
            "cost",
            "pis_year",
            "useful_life",
            "depreciation_factor",
            "salvage_value",
            "is_ic_applied",
            "vat_portion",
            "lbt_portion",
            "description",
        ]

        tangible_data = {
            key: None if tangible_data_loaded.empty
            else tangible_data_loaded.iloc[:, i].to_numpy()
            for i, key in enumerate(tangible_data_attrs)
        }

        # Step #4 (See 'Notes' section in the docstring)
        return TangibleCostData(
            expense_year_init=tangible_data["expense_year"],
            cost_allocation=tangible_data["cost_allocation"],
            cost=tangible_data["cost"],
            pis_year=tangible_data["pis_year"],
            useful_life=tangible_data["useful_life"],
            depreciation_factor=tangible_data["depreciation_factor"],
            salvage_value=tangible_data["salvage_value"],
            is_ic_applied=tangible_data["is_ic_applied"],
            vat_portion=tangible_data["vat_portion"],
            lbt_portion=tangible_data["lbt_portion"],
            description=tangible_data["description"],
            project_years=self.general_config_data.project_years,
            type_of_contract=self.general_config_data.type_of_contract,
            end_date_project=self.general_config_data.end_date_project,
            start_date_project_second=self.general_config_data.start_date_project_second,
        )

    def _get_intangible_cost_data(self) -> IntangibleCostData:
        """
        Retrieves intangible cost data based on available sheets.

        Returns
        -------
        IntangibleCostData
            An instance of IntangibleCostData class.

        Notes
        -----
        The core operations are as follows:
        (1) Capture intangible cost data from attribute self.data_loaded, then perform the
            necessary adjustment,
        (2) Undertake data cleansing: remove all rows which column 'expense_year' is NaN.
            Store the results in a variable named 'intangible_data_loaded',
        (3) Create a dictionary named 'intangible_data' to store the necessary data from
            'intangible_data_loaded',
        (4) Return an instance of IntangibleCostData to store the intangible data appropriately
            as its attributes.
        """
        # Step #1 (See 'Notes' section in the docstring)
        intangible_data_loaded_init = self.data_loaded["Cost Intangible"].dropna(axis=0, how="all")

        # Step #2 (See 'Notes' section in the docstring)
        intangible_data_loaded = (
            intangible_data_loaded_init if intangible_data_loaded_init.empty
            else intangible_data_loaded_init[~pd.isna(intangible_data_loaded_init.iloc[:, 0])].copy()
        )

        # Step #3 (See 'Notes' section in the docstring)
        intangible_data_attrs = [
            "expense_year",
            "cost_allocation",
            "cost",
            "vat_portion",
            "lbt_portion",
            "description",
        ]

        intangible_data = {
            key: None if intangible_data_loaded.empty
            else intangible_data_loaded.iloc[:, i].to_numpy()
            for i, key in enumerate(intangible_data_attrs)
        }

        # Step #4 (See 'Notes' section in the docstring)
        return IntangibleCostData(
            expense_year_init=intangible_data["expense_year"],
            cost_allocation=intangible_data["cost_allocation"],
            cost=intangible_data["cost"],
            vat_portion=intangible_data["vat_portion"],
            lbt_portion=intangible_data["lbt_portion"],
            description=intangible_data["description"],
            project_years=self.general_config_data.project_years,
            type_of_contract=self.general_config_data.type_of_contract,
            end_date_project=self.general_config_data.end_date_project,
            start_date_project_second=self.general_config_data.start_date_project_second,
        )

    def _get_opex_data(self) -> OPEXData:
        """
        Retrieves opex data based on available sheets.

        Returns
        -------
        OPEXData
            An instance of OPEXData class.

        Notes
        -----
        The core operations are as follows:
        (1) Capture opex data from attribute self.data_loaded, then perform the
            necessary adjustment,
        (2) Undertake data cleansing: remove all rows which column 'expense_year' is NaN.
            Store the results in a variable named 'opex_data_loaded',
        (3) Create a dictionary named 'opex_data' to store the necessary data from
            'opex_data_loaded',
        (4) Return an instance of OPEXData to store the asr data appropriately
            as its attributes.
        """
        # Step #1 (See 'Notes' section in the docstring)
        opex_data_loaded_init = self.data_loaded["Cost OPEX"].dropna(axis=0, how="all")

        # Step #2 (See 'Notes' section in the docstring)
        opex_data_loaded = (
            opex_data_loaded_init if opex_data_loaded_init.empty
            else opex_data_loaded_init[~pd.isna(opex_data_loaded_init.iloc[:, 0])].copy()
        )

        # Step #3 (See 'Notes' section in the docstring)
        opex_data_attrs = [
            "expense_year",
            "cost_allocation",
            "fixed_cost",
            "prod_rate",
            "cost_per_volume",
            "vat_portion",
            "lbt_portion",
            "description",
        ]

        opex_data = {
            key: None if opex_data_loaded.empty
            else opex_data_loaded.iloc[:, i].to_numpy()
            for i, key in enumerate(opex_data_attrs)
        }

        # Step #4 (See 'Notes' section in the docstring)
        return OPEXData(
            expense_year_init=opex_data["expense_year"],
            cost_allocation=opex_data["cost_allocation"],
            fixed_cost=opex_data["fixed_cost"],
            prod_rate=opex_data["prod_rate"],
            cost_per_volume=opex_data["cost_per_volume"],
            vat_portion=opex_data["vat_portion"],
            lbt_portion=opex_data["lbt_portion"],
            description=opex_data["description"],
            project_years=self.general_config_data.project_years,
            type_of_contract=self.general_config_data.type_of_contract,
            end_date_project=self.general_config_data.end_date_project,
            start_date_project_second=self.general_config_data.start_date_project_second,
        )

    def _get_asr_cost_data(self) -> ASRCostData:
        """
        Retrieves ASR cost data based on available sheets.

        Returns
        -------
        ASRCostData
            An instance of ASRCostData class.

        Notes
        -----
        The core operations are as follows:
        (1) Capture asr cost data from attribute self.data_loaded, then perform the
            necessary adjustment,
        (2) Undertake data cleansing: remove all rows which column 'expense_year' is NaN.
            Store the results in a variable named 'asr_data_loaded',
        (3) Create a dictionary named 'asr_data' to store the necessary data from
            'asr_data_loaded',
        (4) Return an instance of ASRCostData to store the asr data appropriately
            as its attributes.
        """
        # Step #1 (See 'Notes' section in the docstring)
        asr_data_loaded_init = self.data_loaded["Cost ASR"].dropna(axis=0, how="all")

        # Step #2 (See 'Notes' section in the docstring)
        asr_data_loaded = (
            asr_data_loaded_init if asr_data_loaded_init.empty
            else asr_data_loaded_init[~pd.isna(asr_data_loaded_init.iloc[:, 0])].copy()
        )

        # Step #3 (See 'Notes' section in the docstring)
        asr_data_attrs = [
            "expense_year",
            "cost_allocation",
            "cost",
            "vat_portion",
            "lbt_portion",
            "description",
        ]

        asr_data = {
            key: None if asr_data_loaded.empty
            else asr_data_loaded.iloc[:, i].to_numpy()
            for i, key in enumerate(asr_data_attrs)
        }

        # Step #4 (See 'Notes' section in the docstring)
        return ASRCostData(
            expense_year_init=asr_data["expense_year"],
            cost_allocation=asr_data["cost_allocation"],
            cost=asr_data["cost"],
            vat_portion=asr_data["vat_portion"],
            lbt_portion=asr_data["lbt_portion"],
            description=asr_data["description"],
            project_years=self.general_config_data.project_years,
            type_of_contract=self.general_config_data.type_of_contract,
            end_date_project=self.general_config_data.end_date_project,
            start_date_project_second=self.general_config_data.start_date_project_second,
        )

    def _get_psc_cr_data(self) -> PSCCostRecoveryData:
        """
        Retrieves Production Sharing Contract (PSC) Cost Recovery data
        based on available sheets.

        Returns
        -------
        PSCCostRecoveryData
            An instance of PSCCostRecoveryData representing the extracted data
            stored as its attributes.

        Notes
        -----
        The procedures are as follows:
        (1) Slice the data, only capture the columns that contain necessary data,
        (2) Convert the remaining NaN values to None,
        (3) Assign data to their associated attributes,
        (4) Return a new instance of PSCCostRecoveryData with filled attributes.
        """
        # Step #1 - Step #2 (See 'Notes' section in the docstring)
        psc_cr_data_loaded = self.data_loaded["Cost Recovery Config"].replace(np.nan, None)

        # Step #3 - Step #4 (See 'Notes' section in the docstring)
        return PSCCostRecoveryData(
            oil_ftp_availability=psc_cr_data_loaded.iloc[0, 2],
            oil_ftp_is_shared=psc_cr_data_loaded.iloc[1, 2],
            oil_ftp_portion=psc_cr_data_loaded.iloc[2, 2],
            gas_ftp_availability=psc_cr_data_loaded.iloc[3, 2],
            gas_ftp_is_shared=psc_cr_data_loaded.iloc[4, 2],
            gas_ftp_portion=psc_cr_data_loaded.iloc[5, 2],
            split_type=psc_cr_data_loaded.iloc[8, 2],
            oil_ctr_pretax=psc_cr_data_loaded.iloc[9, 2],
            gas_ctr_pretax=psc_cr_data_loaded.iloc[10, 2],
            ic_availability=psc_cr_data_loaded.iloc[13, 2],
            ic_oil=psc_cr_data_loaded.iloc[14, 2],
            ic_gas=psc_cr_data_loaded.iloc[15, 2],
            oil_cr_cap_rate=psc_cr_data_loaded.iloc[18, 2],
            gas_cr_cap_rate=psc_cr_data_loaded.iloc[19, 2],
            dmo_is_weighted=psc_cr_data_loaded.iloc[22, 2],
            oil_dmo_holiday=psc_cr_data_loaded.iloc[25, 2],
            oil_dmo_period=psc_cr_data_loaded.iloc[26, 2],
            oil_dmo_start_production=psc_cr_data_loaded.iloc[27, 2],
            oil_dmo_volume=psc_cr_data_loaded.iloc[28, 2],
            oil_dmo_fee=psc_cr_data_loaded.iloc[29, 2],
            gas_dmo_holiday=psc_cr_data_loaded.iloc[32, 2],
            gas_dmo_period=psc_cr_data_loaded.iloc[33, 2],
            gas_dmo_start_production=psc_cr_data_loaded.iloc[34, 2],
            gas_dmo_volume=psc_cr_data_loaded.iloc[35, 2],
            gas_dmo_fee=psc_cr_data_loaded.iloc[36, 2],
            rc_split_init=psc_cr_data_loaded.iloc[41:48, 1:].to_numpy(),
            icp_sliding_scale_init=psc_cr_data_loaded.iloc[52:60, 1:].to_numpy(),
            indicator_rc_split_sliding_scale_init=psc_cr_data_loaded.iloc[63:83, 1:3].to_numpy(),
        )

    def _get_psc_gs_data(self) -> PSCGrossSplitData:
        """
        Retrieves Production Sharing Contract (PSC) Gross Split data
        based on available sheets.

        Returns
        -------
        PSCGrossSplitData
            An instance of PSCGrossSplitData representing the extracted data
            stored as its attributes.

        Notes
        -----
        The procedures are as follows:
        (1) Slice the data, only capture the columns that contain necessary data,
        (2) Convert the remaining NaN values to None,
        (3) Assign data to their associated attributes,
        (4) Return a new instance of PSCGrossSplitData with filled attributes.
        """
        # Step #1 - Step #2 (See 'Notes' section in the docstring)
        psc_gs_data_loaded = (
            self.data_loaded["Gross Split Config"]
            .iloc[:, 2]
            .replace(np.nan, None)
        )

        # Step #3 - Step #4 (See 'Notes' section in the docstring)
        return PSCGrossSplitData(
            field_status=psc_gs_data_loaded.iloc[0],
            field_location=psc_gs_data_loaded.iloc[1],
            reservoir_depth=psc_gs_data_loaded.iloc[2],
            infrastructure_availability=psc_gs_data_loaded.iloc[3],
            reservoir_type=psc_gs_data_loaded.iloc[4],
            co2_content=psc_gs_data_loaded.iloc[5],
            h2s_content=psc_gs_data_loaded.iloc[6],
            oil_api=psc_gs_data_loaded.iloc[7],
            domestic_content_use=psc_gs_data_loaded.iloc[8],
            production_stage=psc_gs_data_loaded.iloc[9],
            ministry_discretion_split=psc_gs_data_loaded.iloc[11],
            oil_base_split=psc_gs_data_loaded.iloc[13],
            gas_base_split=psc_gs_data_loaded.iloc[14],
            dmo_is_weighted=psc_gs_data_loaded.iloc[17],
            oil_dmo_holiday=psc_gs_data_loaded.iloc[20],
            oil_dmo_period=psc_gs_data_loaded.iloc[21],
            oil_dmo_start_production=psc_gs_data_loaded.iloc[22],
            oil_dmo_volume=psc_gs_data_loaded.iloc[23],
            oil_dmo_fee=psc_gs_data_loaded.iloc[24],
            gas_dmo_holiday=psc_gs_data_loaded.iloc[27],
            gas_dmo_period=psc_gs_data_loaded.iloc[28],
            gas_dmo_start_production=psc_gs_data_loaded.iloc[29],
            gas_dmo_volume=psc_gs_data_loaded.iloc[30],
            gas_dmo_fee=psc_gs_data_loaded.iloc[31],
        )

    def _get_psc_transition_cr_to_cr(self) -> dict:
        """
        Extract and process the associated data for PSC transition CR to CR.

        Returns
        -------
        dict
            A dictionary containing PSCCostRecoveryData objects.

        Notes
        -----
        The undertaken procedures are as follows:
        (1) Filter attribute self.sheets_loaded for sheets that contain 'Cost Recovery Config' data,
            then assigned it as local variable named 'psc_cr_to_cr_available',
        (2) Capture psc_cr_to_cr data from attribute self.data_loaded, then perform the
            necessary adjustment. Stored the result in a variable named 'psc_cr_to_cr_data_loaded',
        (3) Define a variable named 'psc_cr_to_cr_data' as a dictionary that contain two
            keys: (1) 'Cost Recovery Config', and (2) 'Cost Recovery Config (2)'. Each key is
            an instance of PSCCostRecoveryData with the associated attributes loaded from
            'psc_cr_to_cr_data_loaded',
        (4) Return the variable 'psc_cr_to_cr_data'.

        Example
        -------
        transition_cr_to_cr_data = instance._get_psc_transition_cr_to_cr()
        config_1 = transition_cr_to_cr_data['Cost Recovery Config']
        config_2 = transition_cr_to_cr_data['Cost Recovery Config (2)']
        """
        # Step #1 (See 'Notes' section in the docstring)
        psc_cr_to_cr_available = list(filter(lambda i: "Cost Recovery Config" in i, self.sheets_loaded))

        # Step #2 (See 'Notes' section in the docstring)
        psc_cr_to_cr_data_loaded = {
            key: self.data_loaded[key].replace(np.nan, None) for key in psc_cr_to_cr_available
        }

        # Step #3 - Step #4 (See 'Notes' section in the docstring)
        psc_cr_to_cr_data = {
            key: PSCCostRecoveryData(
                oil_ftp_availability=psc_cr_to_cr_data_loaded[key].iloc[0, 2],
                oil_ftp_is_shared=psc_cr_to_cr_data_loaded[key].iloc[1, 2],
                oil_ftp_portion=psc_cr_to_cr_data_loaded[key].iloc[2, 2],
                gas_ftp_availability=psc_cr_to_cr_data_loaded[key].iloc[3, 2],
                gas_ftp_is_shared=psc_cr_to_cr_data_loaded[key].iloc[4, 2],
                gas_ftp_portion=psc_cr_to_cr_data_loaded[key].iloc[5, 2],
                split_type=psc_cr_to_cr_data_loaded[key].iloc[8, 2],
                oil_ctr_pretax=psc_cr_to_cr_data_loaded[key].iloc[9, 2],
                gas_ctr_pretax=psc_cr_to_cr_data_loaded[key].iloc[10, 2],
                ic_availability=psc_cr_to_cr_data_loaded[key].iloc[13, 2],
                ic_oil=psc_cr_to_cr_data_loaded[key].iloc[14, 2],
                ic_gas=psc_cr_to_cr_data_loaded[key].iloc[15, 2],
                oil_cr_cap_rate=psc_cr_to_cr_data_loaded[key].iloc[18, 2],
                gas_cr_cap_rate=psc_cr_to_cr_data_loaded[key].iloc[19, 2],
                dmo_is_weighted=psc_cr_to_cr_data_loaded[key].iloc[22, 2],
                oil_dmo_holiday=psc_cr_to_cr_data_loaded[key].iloc[25, 2],
                oil_dmo_period=psc_cr_to_cr_data_loaded[key].iloc[26, 2],
                oil_dmo_start_production=psc_cr_to_cr_data_loaded[key].iloc[27, 2],
                oil_dmo_volume=psc_cr_to_cr_data_loaded[key].iloc[28, 2],
                oil_dmo_fee=psc_cr_to_cr_data_loaded[key].iloc[29, 2],
                gas_dmo_holiday=psc_cr_to_cr_data_loaded[key].iloc[32, 2],
                gas_dmo_period=psc_cr_to_cr_data_loaded[key].iloc[33, 2],
                gas_dmo_start_production=psc_cr_to_cr_data_loaded[key].iloc[34, 2],
                gas_dmo_volume=psc_cr_to_cr_data_loaded[key].iloc[35, 2],
                gas_dmo_fee=psc_cr_to_cr_data_loaded[key].iloc[36, 2],
                rc_split_init=psc_cr_to_cr_data_loaded[key].iloc[41:48, 1:].to_numpy(),
                icp_sliding_scale_init=psc_cr_to_cr_data_loaded[key].iloc[52:60, 1:].to_numpy(),
                indicator_rc_split_sliding_scale_init=(
                    psc_cr_to_cr_data_loaded[key].iloc[63:83, 1:3].to_numpy()
                ),
            )
            for key in psc_cr_to_cr_available
        }

        return psc_cr_to_cr_data

    def _get_psc_transition_cr_to_gs(self) -> dict:
        """
        Extract and process the associated data for PSC transition CR to GS.

        Returns
        -------
        dict
            A dictionary containing PSCCostRecoveryData and PSCGrossSplitData objects.

        Notes
        -----
        The undertaken procedures are as follows:
        (1) Load the PSC CR and PSC GS data, respectively,
        (2) Combine both data into a dictionary called 'psc_cr_to_gs_data_loaded',
        (3) Define a variable named 'psc_cr_to_gs_data' as a dictionary that contain two
            keys: (1) 'Cost Recovery Config', and (2) 'Gross Split Config'. The value of each
            key is an instance of PSCCostRecoveryData and PSCGrossSplitData, respectively, with
            the associated attributes loaded from 'psc_cr_to_gs_data_loaded',
        (4) Return the variable 'psc_cr_to_gs_data'.

        Example
        -------
        transition_cr_to_gs_data = instance._get_psc_transition_cr_to_gs()
        config_1 = transition_cr_to_gs_data['Cost Recovery Config']
        config_2 = transition_cr_to_gs_data['Gross Split Config']
        """
        # Step #1 (See 'Notes' section in the docstring)
        psc_cr_data_loaded = self.data_loaded["Cost Recovery Config"].replace(np.nan, None)
        psc_gs_data_loaded = (
            self.data_loaded["Gross Split Config"]
            .iloc[:, 2]
            .replace(np.nan, None)
        )

        # Step #2 (See 'Notes' section in the docstring)
        sheets_target = ["Cost Recovery Config", "Gross Split Config"]
        data_loaded_target = [psc_cr_data_loaded, psc_gs_data_loaded]
        psc_cr_to_gs_data_loaded = {key: data_loaded_target[i] for i, key in enumerate(sheets_target)}

        # Step #3 - Step #4 (See 'Notes' section in the docstring)
        psc_cr_to_gs_data = {
            sheets_target[0]: PSCCostRecoveryData(
                oil_ftp_availability=psc_cr_to_gs_data_loaded[sheets_target[0]].iloc[0, 2],
                oil_ftp_is_shared=psc_cr_to_gs_data_loaded[sheets_target[0]].iloc[1, 2],
                oil_ftp_portion=psc_cr_to_gs_data_loaded[sheets_target[0]].iloc[2, 2],
                gas_ftp_availability=psc_cr_to_gs_data_loaded[sheets_target[0]].iloc[3, 2],
                gas_ftp_is_shared=psc_cr_to_gs_data_loaded[sheets_target[0]].iloc[4, 2],
                gas_ftp_portion=psc_cr_to_gs_data_loaded[sheets_target[0]].iloc[5, 2],
                split_type=psc_cr_to_gs_data_loaded[sheets_target[0]].iloc[8, 2],
                oil_ctr_pretax=psc_cr_to_gs_data_loaded[sheets_target[0]].iloc[9, 2],
                gas_ctr_pretax=psc_cr_to_gs_data_loaded[sheets_target[0]].iloc[10, 2],
                ic_availability=psc_cr_to_gs_data_loaded[sheets_target[0]].iloc[13, 2],
                ic_oil=psc_cr_to_gs_data_loaded[sheets_target[0]].iloc[14, 2],
                ic_gas=psc_cr_to_gs_data_loaded[sheets_target[0]].iloc[15, 2],
                oil_cr_cap_rate=psc_cr_to_gs_data_loaded[sheets_target[0]].iloc[18, 2],
                gas_cr_cap_rate=psc_cr_to_gs_data_loaded[sheets_target[0]].iloc[19, 2],
                dmo_is_weighted=psc_cr_to_gs_data_loaded[sheets_target[0]].iloc[22, 2],
                oil_dmo_holiday=psc_cr_to_gs_data_loaded[sheets_target[0]].iloc[25, 2],
                oil_dmo_period=psc_cr_to_gs_data_loaded[sheets_target[0]].iloc[26, 2],
                oil_dmo_start_production=psc_cr_to_gs_data_loaded[sheets_target[0]].iloc[27, 2],
                oil_dmo_volume=psc_cr_to_gs_data_loaded[sheets_target[0]].iloc[28, 2],
                oil_dmo_fee=psc_cr_to_gs_data_loaded[sheets_target[0]].iloc[29, 2],
                gas_dmo_holiday=psc_cr_to_gs_data_loaded[sheets_target[0]].iloc[32, 2],
                gas_dmo_period=psc_cr_to_gs_data_loaded[sheets_target[0]].iloc[33, 2],
                gas_dmo_start_production=psc_cr_to_gs_data_loaded[sheets_target[0]].iloc[34, 2],
                gas_dmo_volume=psc_cr_to_gs_data_loaded[sheets_target[0]].iloc[35, 2],
                gas_dmo_fee=psc_cr_to_gs_data_loaded[sheets_target[0]].iloc[36, 2],
                rc_split_init=psc_cr_to_gs_data_loaded[sheets_target[0]].iloc[41:48, 1:].to_numpy(),
                icp_sliding_scale_init=(
                    psc_cr_to_gs_data_loaded[sheets_target[0]].iloc[52:60, 1:].to_numpy()
                ),
                indicator_rc_split_sliding_scale_init=(
                    psc_cr_to_gs_data_loaded[sheets_target[0]].iloc[63:83, 1:3].to_numpy()
                ),
            ),
            sheets_target[1]: PSCGrossSplitData(
                field_status=psc_cr_to_gs_data_loaded[sheets_target[1]].iloc[0],
                field_location=psc_cr_to_gs_data_loaded[sheets_target[1]].iloc[1],
                reservoir_depth=psc_cr_to_gs_data_loaded[sheets_target[1]].iloc[2],
                infrastructure_availability=psc_cr_to_gs_data_loaded[sheets_target[1]].iloc[3],
                reservoir_type=psc_cr_to_gs_data_loaded[sheets_target[1]].iloc[4],
                co2_content=psc_cr_to_gs_data_loaded[sheets_target[1]].iloc[5],
                h2s_content=psc_cr_to_gs_data_loaded[sheets_target[1]].iloc[6],
                oil_api=psc_cr_to_gs_data_loaded[sheets_target[1]].iloc[7],
                domestic_content_use=psc_cr_to_gs_data_loaded[sheets_target[1]].iloc[8],
                production_stage=psc_cr_to_gs_data_loaded[sheets_target[1]].iloc[9],
                ministry_discretion_split=psc_cr_to_gs_data_loaded[sheets_target[1]].iloc[11],
                oil_base_split=psc_cr_to_gs_data_loaded[sheets_target[1]].iloc[13],
                gas_base_split=psc_cr_to_gs_data_loaded[sheets_target[1]].iloc[14],
                dmo_is_weighted=psc_cr_to_gs_data_loaded[sheets_target[1]].iloc[17],
                oil_dmo_holiday=psc_cr_to_gs_data_loaded[sheets_target[1]].iloc[20],
                oil_dmo_period=psc_cr_to_gs_data_loaded[sheets_target[1]].iloc[21],
                oil_dmo_start_production=psc_cr_to_gs_data_loaded[sheets_target[1]].iloc[22],
                oil_dmo_volume=psc_cr_to_gs_data_loaded[sheets_target[1]].iloc[23],
                oil_dmo_fee=psc_cr_to_gs_data_loaded[sheets_target[1]].iloc[24],
                gas_dmo_holiday=psc_cr_to_gs_data_loaded[sheets_target[1]].iloc[27],
                gas_dmo_period=psc_cr_to_gs_data_loaded[sheets_target[1]].iloc[28],
                gas_dmo_start_production=psc_cr_to_gs_data_loaded[sheets_target[1]].iloc[29],
                gas_dmo_volume=psc_cr_to_gs_data_loaded[sheets_target[1]].iloc[30],
                gas_dmo_fee=psc_cr_to_gs_data_loaded[sheets_target[1]].iloc[31],
            )
        }

        return psc_cr_to_gs_data

    def _get_psc_transition_gs_to_gs(self) -> dict:
        """
        Extract and process the associated data for PSC transition GS to GS.

        Returns
        -------
        dict
            A dictionary containing PSCGrossSplitData objects.

        Notes
        -----
        The undertaken procedures are as follows:
        (1) Filter attribute self.sheets_loaded for sheets that contain 'Gross Split Config' data,
            then assigned it as local variable named 'psc_gs_to_gs_available',
        (2) Capture psc_gs_to_gs data from attribute self.data_loaded, then perform the
            necessary adjustment. Stored the result in a variable named 'psc_gs_to_gs_data_loaded',
        (3) Define a variable named 'psc_gs_to_gs_data' as a dictionary that contain two
            keys: (1) 'Gross Split Config', and (2) 'Gross Split Config (2)'. Each keys is
            an instance of PSCGrossSplitData with the associated attributes loaded from
            'psc_gs_to_gs_data_loaded',
        (4) Return the variable 'psc_gs_to_gs_data'.

        Example
        -------
        transition_gs_to_gs_data = instance._get_psc_transition_gs_to_gs()
        config_1 = transition_gs_to_gs_data['Gross Split Config']
        config_2 = transition_gs_to_gs_data['Gross Split Config (2)']
        """
        # Step #1 (See 'Notes' section in the docstring)
        psc_gs_to_gs_available = list(filter(lambda i: "Gross Split Config" in i, self.sheets_loaded))

        # Step #2 (See 'Notes' section in the docstring)
        psc_gs_to_gs_data_loaded = {
            key: self.data_loaded[key].iloc[:, 2].replace(np.nan, None)
            for key in psc_gs_to_gs_available
        }

        # Step #3 - Step #4 (See 'Notes' section in the docstring)
        psc_gs_to_gs_data = {
            key: PSCGrossSplitData(
                field_status=psc_gs_to_gs_data_loaded[key].iloc[0],
                field_location=psc_gs_to_gs_data_loaded[key].iloc[1],
                reservoir_depth=psc_gs_to_gs_data_loaded[key].iloc[2],
                infrastructure_availability=psc_gs_to_gs_data_loaded[key].iloc[3],
                reservoir_type=psc_gs_to_gs_data_loaded[key].iloc[4],
                co2_content=psc_gs_to_gs_data_loaded[key].iloc[5],
                h2s_content=psc_gs_to_gs_data_loaded[key].iloc[6],
                oil_api=psc_gs_to_gs_data_loaded[key].iloc[7],
                domestic_content_use=psc_gs_to_gs_data_loaded[key].iloc[8],
                production_stage=psc_gs_to_gs_data_loaded[key].iloc[9],
                ministry_discretion_split=psc_gs_to_gs_data_loaded[key].iloc[11],
                oil_base_split=psc_gs_to_gs_data_loaded[key].iloc[13],
                gas_base_split=psc_gs_to_gs_data_loaded[key].iloc[14],
                dmo_is_weighted=psc_gs_to_gs_data_loaded[key].iloc[17],
                oil_dmo_holiday=psc_gs_to_gs_data_loaded[key].iloc[20],
                oil_dmo_period=psc_gs_to_gs_data_loaded[key].iloc[21],
                oil_dmo_start_production=psc_gs_to_gs_data_loaded[key].iloc[22],
                oil_dmo_volume=psc_gs_to_gs_data_loaded[key].iloc[23],
                oil_dmo_fee=psc_gs_to_gs_data_loaded[key].iloc[24],
                gas_dmo_holiday=psc_gs_to_gs_data_loaded[key].iloc[27],
                gas_dmo_period=psc_gs_to_gs_data_loaded[key].iloc[28],
                gas_dmo_start_production=psc_gs_to_gs_data_loaded[key].iloc[29],
                gas_dmo_volume=psc_gs_to_gs_data_loaded[key].iloc[30],
                gas_dmo_fee=psc_gs_to_gs_data_loaded[key].iloc[31],
            )
            for key in psc_gs_to_gs_available
        }

        return psc_gs_to_gs_data

    def _get_psc_transition_gs_to_cr(self) -> dict:
        """
        Extract and process the associated data for PSC transition GS to CR.

        Returns
        -------
        dict
            A dictionary containing PSCGrossSplitData and PSCCostRecoveryData objects.

        Notes
        -----
        The undertaken procedures are as follows:
        (1) Load the PSC GS and PSC CR data, respectively,
        (2) Combine both data into a dictionary called 'psc_gs_to_cr_data_loaded',
        (3) Define a variable named 'psc_gs_to_cr_data' as a dictionary that contain two
            keys: (1) 'Gross Split Config', and (2) 'Cost Recovery Config'. The value of each
            key is an instance of PSCGrossSplitData and PSCCostRecoveryData, respectively, with
            the associated attributes loaded from 'psc_gs_to_cr_data_loaded',
        (4) Return the variable 'psc_gs_to_cr_data'.

        Example
        -------
        transition_gs_to_cr_data = instance._get_psc_transition_gs_to_cr()
        config_1 = transition_gs_to_cr_data['Gross Split Config']
        config_2 = transition_gs_to_cr_data['Cost Recovery Config']
        """
        # Step #1 (See 'Notes' section in the docstring)
        psc_gs_data_loaded = (
            self.data_loaded["Gross Split Config"]
            .iloc[:, 2]
            .replace(np.nan, None)
        )
        psc_cr_data_loaded = self.data_loaded["Cost Recovery Config"].replace(np.nan, None)

        # Step #2 (See 'Notes' section in the docstring)
        sheets_target = ["Gross Split Config", "Cost Recovery Config"]
        data_loaded_target = [psc_gs_data_loaded, psc_cr_data_loaded]
        psc_gs_to_cr_data_loaded = {key: data_loaded_target[i] for i, key in enumerate(sheets_target)}

        # Step #3 - Step #4 (See 'Notes' section in the docstring)
        psc_gs_to_cr_data = {
            sheets_target[0]: PSCGrossSplitData(
                field_status=psc_gs_to_cr_data_loaded[sheets_target[0]].iloc[0],
                field_location=psc_gs_to_cr_data_loaded[sheets_target[0]].iloc[1],
                reservoir_depth=psc_gs_to_cr_data_loaded[sheets_target[0]].iloc[2],
                infrastructure_availability=psc_gs_to_cr_data_loaded[sheets_target[0]].iloc[3],
                reservoir_type=psc_gs_to_cr_data_loaded[sheets_target[0]].iloc[4],
                co2_content=psc_gs_to_cr_data_loaded[sheets_target[0]].iloc[5],
                h2s_content=psc_gs_to_cr_data_loaded[sheets_target[0]].iloc[6],
                oil_api=psc_gs_to_cr_data_loaded[sheets_target[0]].iloc[7],
                domestic_content_use=psc_gs_to_cr_data_loaded[sheets_target[0]].iloc[8],
                production_stage=psc_gs_to_cr_data_loaded[sheets_target[0]].iloc[9],
                ministry_discretion_split=psc_gs_to_cr_data_loaded[sheets_target[0]].iloc[11],
                oil_base_split=psc_gs_to_cr_data_loaded[sheets_target[0]].iloc[13],
                gas_base_split=psc_gs_to_cr_data_loaded[sheets_target[0]].iloc[14],
                dmo_is_weighted=psc_gs_to_cr_data_loaded[sheets_target[0]].iloc[17],
                oil_dmo_holiday=psc_gs_to_cr_data_loaded[sheets_target[0]].iloc[20],
                oil_dmo_period=psc_gs_to_cr_data_loaded[sheets_target[0]].iloc[21],
                oil_dmo_start_production=psc_gs_to_cr_data_loaded[sheets_target[0]].iloc[22],
                oil_dmo_volume=psc_gs_to_cr_data_loaded[sheets_target[0]].iloc[23],
                oil_dmo_fee=psc_gs_to_cr_data_loaded[sheets_target[0]].iloc[24],
                gas_dmo_holiday=psc_gs_to_cr_data_loaded[sheets_target[0]].iloc[27],
                gas_dmo_period=psc_gs_to_cr_data_loaded[sheets_target[0]].iloc[28],
                gas_dmo_start_production=psc_gs_to_cr_data_loaded[sheets_target[0]].iloc[29],
                gas_dmo_volume=psc_gs_to_cr_data_loaded[sheets_target[0]].iloc[30],
                gas_dmo_fee=psc_gs_to_cr_data_loaded[sheets_target[0]].iloc[31],
            ),
            sheets_target[1]: PSCCostRecoveryData(
                oil_ftp_availability=psc_gs_to_cr_data_loaded[sheets_target[1]].iloc[0, 2],
                oil_ftp_is_shared=psc_gs_to_cr_data_loaded[sheets_target[1]].iloc[1, 2],
                oil_ftp_portion=psc_gs_to_cr_data_loaded[sheets_target[1]].iloc[2, 2],
                gas_ftp_availability=psc_gs_to_cr_data_loaded[sheets_target[1]].iloc[3, 2],
                gas_ftp_is_shared=psc_gs_to_cr_data_loaded[sheets_target[1]].iloc[4, 2],
                gas_ftp_portion=psc_gs_to_cr_data_loaded[sheets_target[1]].iloc[5, 2],
                split_type=psc_gs_to_cr_data_loaded[sheets_target[1]].iloc[8, 2],
                oil_ctr_pretax=psc_gs_to_cr_data_loaded[sheets_target[1]].iloc[9, 2],
                gas_ctr_pretax=psc_gs_to_cr_data_loaded[sheets_target[1]].iloc[10, 2],
                ic_availability=psc_gs_to_cr_data_loaded[sheets_target[1]].iloc[13, 2],
                ic_oil=psc_gs_to_cr_data_loaded[sheets_target[1]].iloc[14, 2],
                ic_gas=psc_gs_to_cr_data_loaded[sheets_target[1]].iloc[15, 2],
                oil_cr_cap_rate=psc_gs_to_cr_data_loaded[sheets_target[1]].iloc[18, 2],
                gas_cr_cap_rate=psc_gs_to_cr_data_loaded[sheets_target[1]].iloc[19, 2],
                dmo_is_weighted=psc_gs_to_cr_data_loaded[sheets_target[1]].iloc[22, 2],
                oil_dmo_holiday=psc_gs_to_cr_data_loaded[sheets_target[1]].iloc[25, 2],
                oil_dmo_period=psc_gs_to_cr_data_loaded[sheets_target[1]].iloc[26, 2],
                oil_dmo_start_production=psc_gs_to_cr_data_loaded[sheets_target[1]].iloc[27, 2],
                oil_dmo_volume=psc_gs_to_cr_data_loaded[sheets_target[1]].iloc[28, 2],
                oil_dmo_fee=psc_gs_to_cr_data_loaded[sheets_target[1]].iloc[29, 2],
                gas_dmo_holiday=psc_gs_to_cr_data_loaded[sheets_target[1]].iloc[32, 2],
                gas_dmo_period=psc_gs_to_cr_data_loaded[sheets_target[1]].iloc[33, 2],
                gas_dmo_start_production=psc_gs_to_cr_data_loaded[sheets_target[1]].iloc[34, 2],
                gas_dmo_volume=psc_gs_to_cr_data_loaded[sheets_target[1]].iloc[35, 2],
                gas_dmo_fee=psc_gs_to_cr_data_loaded[sheets_target[1]].iloc[36, 2],
                rc_split_init=psc_gs_to_cr_data_loaded[sheets_target[1]].iloc[41:48, 1:].to_numpy(),
                icp_sliding_scale_init=(
                    psc_gs_to_cr_data_loaded[sheets_target[1]].iloc[52:60, 1:].to_numpy()
                ),
                indicator_rc_split_sliding_scale_init=(
                    psc_gs_to_cr_data_loaded[sheets_target[1]].iloc[63:83, 1:3].to_numpy()
                ),
            )
        }

        return psc_gs_to_cr_data

    def _get_sensitivity_data(self) -> SensitivityData:
        """
        Extract and process the associated data for sensitivity analysis.

        Returns
        -------
        SensitivityData
            An instance of SensitivityData.

        Notes
        -----
        The undertaken procedures are as follows:
        (1) Extract sensitivity data from 'self.data_loaded',
        (2) Create a new instance of SensitivityData with the necessary information
            stored as the corresponding attributes.
        """
        sensitivity_data_loaded = self.data_loaded["Sensitivity"].iloc[0:11, 1:3]

        return SensitivityData(
            parameter=sensitivity_data_loaded.iloc[6:, 0].to_numpy().tolist(),
            percentage_min=sensitivity_data_loaded.iloc[0, 1],
            percentage_max=sensitivity_data_loaded.iloc[1, 1],
            step=sensitivity_data_loaded.iloc[2, 1],
        )

    def _get_montecarlo_data(self) -> MonteCarloData:
        """
        Extract and process the associated data for montecarlo analysis.

        Returns
        -------
        MonteCarloData
            An instance of MonteCarloData.

        Notes
        -----
        The undertaken procedures are as follows:
        (1) Extract montecarlo data from 'self.data_loaded',
        (2) Create a new instance of MonteCarloData with the necessary information
            stored as the corresponding attributes.
        """
        mc_data_loaded = self.data_loaded["Uncertainty"].iloc[1:10, np.r_[1:3, 4:8]]

        return MonteCarloData(
            run_number=mc_data_loaded.iloc[0, 1],
            parameter=mc_data_loaded.iloc[3:8, 0].to_numpy().tolist(),
            distribution=mc_data_loaded.iloc[3:8, 1].to_numpy().tolist(),
            min_values=mc_data_loaded.iloc[3:8, 2].to_numpy(),
            mean_values=mc_data_loaded.iloc[3:8, 3].to_numpy(),
            max_values=mc_data_loaded.iloc[3:8, 4].to_numpy(),
            std_dev=mc_data_loaded.iloc[3:8, 5].to_numpy(),
        )

    def _get_optimization_data(self) -> OptimizationData:
        """
        Extract and process the associated data for optimization study.

        Returns
        -------
        OptimizationData
            An instance of OptimizationData.

        Notes
        -----
        The undertaken operations are as follows:
        (1) Extract optimization data from 'self.data_loaded' and perform the necessary adjustments,
        (2) Configure the data associated with PSC Cost Recovery optimization,
        (3) Configure the data associated with PSC Gross Split optimization,
        (4) Configure the data associated with objective function,
        (5) Create an instance of OptimizationData and store the above data as its attributes.
        """
        # Step #1 (See 'Notes' section in the docstring)
        optimization_data_loaded = (
            self.data_loaded["Optimization"]
            .iloc[:, np.r_[1, 4, 6:8, 3, 5]]
            # .replace(np.nan, None)
        )

        # Step #2 (See 'Notes' section in the docstring)
        data_cr_init = {
            key: optimization_data_loaded.iloc[3:13, i].to_numpy()
            for i, key in enumerate(["parameter", "priority", "min", "max"])
        }

        # Step #3 (See 'Notes' section in the docstring)
        data_gs_init = {
            key: optimization_data_loaded.iloc[15:, i].to_numpy()
            for i, key in enumerate(["parameter", "priority", "min", "max"])
        }

        # Step #4 (See 'Notes' section in the docstring)
        target = {
            key: optimization_data_loaded.iloc[0, i + 4]
            for i, key in enumerate(["parameter", "value"])
        }

        # Step #5 (See 'Notes' section in the docstring)
        return OptimizationData(
            target_init=target,
            data_cr_init=data_cr_init,
            data_gs_init=data_gs_init,
            type_of_contract=self.general_config_data.type_of_contract,
        )

    def prepare_data(self) -> None:
        """
        Method to prepare data of Spreadsheet class.

        Reads and prepare data from a target Excel file and fills in attributes
        associated with various categories, including configuration, lifting, cost,
        contract, and additional functionality.

        Returns
        -------
        None
            This method modifies the instance attributes of Spreadsheet class.

        Notes
        -----
        This method serves as a comprehensive data preparation step, calling specific
        private methods to obtain and assign data based on their category.

        The attributes include general configuration data, fiscal configuration data,
        lifting data for oil, gas, LPG propane, LPG butane, sulfur, electricity, and CO2, cost data
        including tangible and intangible costs, opex, and ASR costs, contract data including
        PSC Cost Recovery (CR), Gross Split (GS), and transition cases, as well as additional
        functionality data such as sensitivity, Monte Carlo, and optimization data.
        """
        # Read data from a target Excel file
        self.read_from_excel()

        # Fill in the attributes associated with config data
        self.general_config_data = self._get_general_config_data()
        self.fiscal_config_data = self._get_fiscal_config_data()

        # Fill in the attributes associated with lifting data
        self.oil_lifting_data = self._get_oil_lifting_data()
        self.gas_lifting_data = self._get_gas_lifting_data()
        self.lpg_propane_lifting_data = self._get_lpg_propane_lifting_data()
        self.lpg_butane_lifting_data = self._get_lpg_butane_lifting_data()
        self.sulfur_lifting_data = self._get_sulfur_lifting_data()
        self.electricity_lifting_data = self._get_electricity_lifting_data()
        self.co2_lifting_data = self._get_co2_lifting_data()

        # Fill in the attributes associated with cost data
        self.tangible_cost_data = self._get_tangible_cost_data()
        self.intangible_cost_data = self._get_intangible_cost_data()
        self.opex_data = self._get_opex_data()
        self.asr_cost_data = self._get_asr_cost_data()

        # Fill in the attributes associated with contract data
        self.psc_cr_data = self._get_psc_cr_data()
        self.psc_gs_data = self._get_psc_gs_data()
        self.psc_transition_cr_to_cr = self._get_psc_transition_cr_to_cr()
        self.psc_transition_cr_to_gs = self._get_psc_transition_cr_to_gs()
        self.psc_transition_gs_to_gs = self._get_psc_transition_gs_to_gs()
        self.psc_transition_gs_to_cr = self._get_psc_transition_gs_to_cr()

        # Fill in the attributes associated with sensitivity and optimization
        self.optimization_data = self._get_optimization_data()
        self.sensitivity_data = self._get_sensitivity_data()
        self.montecarlo_data = self._get_montecarlo_data()
