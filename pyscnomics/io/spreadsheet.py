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
    sensitivity_data: SensitivityData = field(default=None, init=False)
    montecarlo_data: MonteCarloData = field(default=None, init=False)
    optimization_data: OptimizationData = field(default=None, init=False)

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
                "Summary",
                "Result Table CR",
                "Result Table GS"
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
        vat_discount = general_config_data_loaded.iloc[18, 2]
        lbt_discount = general_config_data_loaded.iloc[19, 2]
        gsa_number = general_config_data_loaded.iloc[11, 7]

        return GeneralConfigData(
            start_date_project=start_date_project,
            end_date_project=end_date_project,
            start_date_project_second=start_date_project_second,
            end_date_project_second=end_date_project_second,
            type_of_contract=type_of_contract,
            oil_onstream_date=oil_onstream_date,
            gas_onstream_date=gas_onstream_date,
            discount_rate_start_year=discount_rate_start_year,
            discount_rate=discount_rate,
            inflation_rate_applied_to=inflation_rate_applied_to,
            vat_discount=vat_discount,
            lbt_discount=lbt_discount,
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
        tax_mode = fiscal_config_data_loaded.iloc[0, 1]
        tax_rate_init = fiscal_config_data_loaded.iloc[1, 1]
        tax_payment_method = fiscal_config_data_loaded.iloc[11, 1]
        tax_ftp_regime = fiscal_config_data_loaded.iloc[14, 1]
        npv_mode = fiscal_config_data_loaded.iloc[17, 1]
        discounting_mode = fiscal_config_data_loaded.iloc[18, 1]
        future_rate_asr = fiscal_config_data_loaded.iloc[21, 1]
        depreciation_method = fiscal_config_data_loaded.iloc[24, 1]
        decline_factor = fiscal_config_data_loaded.iloc[25, 1]
        inflation_rate_mode = fiscal_config_data_loaded.iloc[28, 1]
        inflation_rate_init = fiscal_config_data_loaded.iloc[29, 1]
        multi_tax = {
            "year": fiscal_config_data_loaded.iloc[4:10, 0].to_numpy(),
            "rate": fiscal_config_data_loaded.iloc[4:10, 1].to_numpy(),
        }
        multi_inflation = {
            "year": fiscal_config_data_loaded.iloc[32:52, 0].to_numpy(),
            "rate": fiscal_config_data_loaded.iloc[32:52, 1].to_numpy(),
        }
        transferred_unrec_cost = fiscal_config_data_loaded.iloc[54, 1]
        tax_rate_second_contract = fiscal_config_data_loaded.iloc[55, 1]
        vat_mode = fiscal_config_data_loaded.iloc[58, 1]
        vat_rate_init = fiscal_config_data_loaded.iloc[59, 1]
        multi_vat = {
            "year": fiscal_config_data_loaded.iloc[62:82, 0].to_numpy(),
            "rate": fiscal_config_data_loaded.iloc[62:82, 1].to_numpy(),
        }
        lbt_mode = fiscal_config_data_loaded.iloc[84, 1]
        lbt_rate_init = fiscal_config_data_loaded.iloc[85, 1]
        multi_lbt = {
            "year": fiscal_config_data_loaded.iloc[87:108, 0].to_numpy(),
            "rate": fiscal_config_data_loaded.iloc[87:108, 1].to_numpy(),
        }

        # Step #3 (See 'Notes' section in the docstring)
        return FiscalConfigData(
            tax_payment_method=tax_payment_method,
            tax_ftp_regime=tax_ftp_regime,
            npv_mode=npv_mode,
            discounting_mode=discounting_mode,
            future_rate_asr=float(future_rate_asr),
            depreciation_method=depreciation_method,
            decline_factor=float(decline_factor),
            transferred_unrec_cost=float(transferred_unrec_cost),
            tax_rate_second_contract=float(tax_rate_second_contract),
            tax_mode=tax_mode,
            tax_rate_init=tax_rate_init,
            multi_tax=multi_tax,
            inflation_rate_mode=inflation_rate_mode,
            inflation_rate_init=inflation_rate_init,
            multi_inflation=multi_inflation,
            vat_mode=vat_mode,
            vat_rate_init=vat_rate_init,
            multi_vat=multi_vat,
            lbt_mode=lbt_mode,
            lbt_rate_init=lbt_rate_init,
            multi_lbt=multi_lbt,
            project_years=self.general_config_data.project_years,
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
        (2) Configure the number of active GSA in each element of 'gas_data_available',
        (3) If 'gas_data_available' is empty (length is zero), then return a new instance
            of GasLiftingData with the associated attributes set to None. Here, the associated
            operations are as follows:
            -   Create a dictionary named 'gas_data' where each corresponding attributes
                set to None,
            -   Create a new instance of GasLiftingData with attributes set accordingly
                based on information stored in 'gas_data',
        (4) If 'gas_data_available' is not empty, then return a new instance of GasLiftingData
            with the associated attributes loaded from the commensurate Excel worksheet. Here,
            the associated operations are as follows:
            -   Specify two variables, namely 'gas_attrs_general' and 'gas_attrs_gsa',
            -   Define variable named 'gas_id'. This variable stores information regarding
                the associated column indices in self.data_loaded that corresponds to
                'gas_gsa_lifting_rate', 'gas_gsa_ghv', and 'gas_gsa_price'. Variable 'gas_id'
                is a 2D array: the zeroth axis depicts indices of 'gas_gsa_lifting_rate',
                'gas_gsa_ghv', and 'gas_gsa_price' while the first axis manifest the number of
                GSA available,
            -   Rearrange self.loaded_data encompassing only gas lifting data; stores the
                information in a variable named 'gas_data_loaded',
            -   Instantiate an empty dictionary named 'gas_data', then fills in the variable
                by specifying the elements accordingly: (a) 'prod_year', (b) 'gas_lifting_rate',
                (c) 'gas_gsa_lifting_rate', (d) 'gas_gsa_ghv', and (e) gas_gsa_price,
            -   Create a new instance of GasLiftingData with attributes set accordingly
                based on information stored in 'gas_data'.
        """
        # Step #1 (See 'Notes' section in the docstring)
        gas_data_available = list(filter(lambda i: "Prod Gas" in i, self.sheets_loaded))

        # Step #2 (See 'Notes' section in the docstring)
        gas_gsa_number = self.general_config_data.gsa_number
        gas_gsa_variables = ["GSA {0}".format(i + 1) for i in range(gas_gsa_number)]

        # Step #3 (See 'Notes' section in the docstring)
        gas_attrs = [
            "prod_year",
            "gas_prod_rate",
            "gas_gsa_lifting_rate",
            "gas_gsa_ghv",
            "gas_gsa_price",
        ]

        if len(gas_data_available) == 0:
            gas_data = {
                key: {"Prod Gas": {i: None for i in gas_gsa_variables}}
                if key not in ["prod_year", "gas_prod_rate"]
                else {"Prod Gas": None}
                for key in gas_attrs
            }

            return GasLiftingData(
                gas_gsa_number=gas_gsa_number,
                prod_year_init=gas_data["prod_year"],
                gas_prod_rate=gas_data["gas_prod_rate"],
                gas_gsa_lifting_rate=gas_data["gas_gsa_lifting_rate"],
                gas_gsa_ghv=gas_data["gas_gsa_ghv"],
                gas_gsa_price=gas_data["gas_gsa_price"],
                project_duration=self.general_config_data.project_duration,
                project_years=self.general_config_data.project_years,
                type_of_contract=self.general_config_data.type_of_contract,
                end_date_project=self.general_config_data.end_date_project,
                start_date_project_second=self.general_config_data.start_date_project_second,
            )

        # Step #4 (See 'Notes' section in the docstring)
        else:
            gas_attrs_general = ["prod_year", "gas_prod_rate"]
            gas_attrs_gsa = ["gas_gsa_lifting_rate", "gas_gsa_ghv", "gas_gsa_price"]
            gas_id = (
                np.repeat(np.arange(len(gas_attrs_gsa))[:, np.newaxis] + 2, gas_gsa_number, axis=1)
                + np.arange(gas_gsa_number) * len(gas_attrs_gsa)
            )
            gas_data_loaded = {i: self.data_loaded[i].fillna(0) for i in gas_data_available}

            gas_data = {}

            for key, val_attr in enumerate(gas_attrs_general):
                gas_data[val_attr] = {}
                for i in gas_data_available:
                    if gas_data_loaded[i].empty:
                        gas_data[val_attr][i] = None
                    else:
                        gas_data[val_attr][i] = (
                            gas_data_loaded[i].iloc[:, key]
                            .to_numpy()
                        )

            for key, val_attr in enumerate(gas_attrs_gsa):
                gas_data[val_attr] = {}
                for i in gas_data_available:
                    gas_data[val_attr][i] = {}
                    if gas_data_loaded[i].empty:
                        for j in gas_gsa_variables:
                            gas_data[val_attr][i][j] = None
                    else:
                        for j, k in zip(gas_gsa_variables, range(gas_gsa_number)):
                            gas_data[val_attr][i][j] = (
                                gas_data_loaded[i].iloc[:, gas_id[key, k]]
                                .to_numpy()
                            )

            return GasLiftingData(
                gas_gsa_number=gas_gsa_number,
                prod_year_init=gas_data["prod_year"],
                gas_prod_rate=gas_data["gas_prod_rate"],
                gas_gsa_lifting_rate=gas_data["gas_gsa_lifting_rate"],
                gas_gsa_ghv=gas_data["gas_gsa_ghv"],
                gas_gsa_price=gas_data["gas_gsa_price"],
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
        (2) If 'sulfur_data_available' is empty (length is zero), then return a new
            instance of SulfurLiftingData with the associated attributes set to None,
        (3) If 'sulfur_data_available' is not empty, first check whether a particular
            self.data_loaded[sulfur_data_available] is an empty dataframe. If it is, then
            create a new instance of SulfurLiftingData with the associated attributes set
            to None. If a particular self.data_loaded[sulfur_data_available] is not an empty
            dataframe, then create a new instance of SulfurLiftingData with the associated
            attributes set as the loaded value from the corresponding Excel worksheet.
        """

        # Step #1 (See 'Notes' section in the docstring)
        sulfur_data_available = list(filter(lambda i: "Prod Sulfur" in i, self.sheets_loaded))

        # Step #2 (See 'Notes' section in the docstring)
        sulfur_data_attrs = ["prod_year", "lifting_rate", "price"]

        if len(sulfur_data_available) == 0:
            sulfur_data = {key: {"Prod Sulfur": None} for key in sulfur_data_attrs}

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

        # Step #3 (See 'Notes' section in the docstring)
        else:
            sulfur_data = {}
            sulfur_data_loaded = {i: self.data_loaded[i].fillna(0) for i in sulfur_data_available}

            for key, val_attr in enumerate(sulfur_data_attrs):
                sulfur_data[val_attr] = {}
                for i in sulfur_data_available:
                    if sulfur_data_loaded[i].empty:
                        sulfur_data[val_attr][i] = None
                    else:
                        sulfur_data[val_attr][i] = (
                            sulfur_data_loaded[i].iloc[:, key]
                            .to_numpy()
                        )

            return SulfurLiftingData(
                prod_year_init=sulfur_data["prod_year"],
                lifting_rate=sulfur_data["lifting_rate"],
                price=sulfur_data["price"],
                project_duration=self.general_config_data.project_duration,
                project_years=self.general_config_data.project_years,
                type_of_contract=self.general_config_data.type_of_contract,
                end_date_project=self.general_config_data.end_date_project,
                start_date_project_second=self.general_config_data.start_date_project_second
            )

    def _get_electricity_lifting_data(self) -> ElectricityLiftingData:
        """
        Retrieves electricity lifting data based on available sheets.

        Returns
        -------
        ElectricityLiftingData
            An instance of the ElectricityLiftingData class containing the following attributes:
                - prod_year: dict
                    Dictionary containing production years data.
                - lifting_rate: dict
                    Dictionary containing electricity lifting rate data.
                - price: dict
                    Dictionary containing electricity price data.
                - project_duration: int
                    The duration of the project.
                - project_years: numpy.ndarray
                    An array representing the project years.

        Notes
        -----
        The undertaken procedures are as follows:
        (1) Filter attribute self.sheets_loaded for sheets that contain 'Prod Electricity'
            data, then assigned it as local variable named 'electricity_data_available',
        (2) If 'electricity_data_available' is empty (length is zero), then return a new
            instance of ElectricityLiftingData with the associated attributes set to None,
        (3) If 'electricity_data_available' is not empty, first check whether a particular
            self.data_loaded[electricity_data_available] is an empty dataframe. If it is, then
            create a new instance of ElectricityLiftingData with the associated attributes set
            to None. If a particular self.data_loaded[electricity_data_available] is not an empty
            dataframe, then create a new instance of ElectricityLiftingData with the associated
            attributes set as the loaded value from the corresponding Excel worksheet.
        """

        # Step #1 (See 'Notes' section in the docstring)
        electricity_data_available = list(filter(lambda i: "Prod Electricity" in i, self.sheets_loaded))

        # Step #2 (See 'Notes' section in the docstring)
        electricity_data_attrs = ["prod_year", "lifting_rate", "price"]

        if len(electricity_data_available) == 0:
            electricity_data = {key: {"Prod Electricity": None} for key in electricity_data_attrs}

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

        # Step #3 (See 'Notes' section in the docstring)
        else:
            electricity_data = {}
            electricity_data_loaded = {
                i: self.data_loaded[i].fillna(0) for i in electricity_data_available
            }

            for key, val_attr in enumerate(electricity_data_attrs):
                electricity_data[val_attr] = {}
                for i in electricity_data_available:
                    if electricity_data_loaded[i].empty:
                        electricity_data[val_attr][i] = None
                    else:
                        electricity_data[val_attr][i] = (
                            electricity_data_loaded[i].iloc[:, key]
                            .to_numpy()
                        )

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
        (2) If 'co2_data_available' is empty (length is zero), then return a new
            instance of CO2LiftingData with the associated attributes set to None,
        (3) If 'co2_data_available' is not empty, first check whether a particular
            self.data_loaded[co2_data_available] is an empty dataframe. If it is, then
            create a new instance of CO2LiftingData with the associated attributes set
            to None. If a particular self.data_loaded[co2_data_available] is not an empty
            dataframe, then create a new instance of CO2LiftingData with the associated
            attributes set as the loaded value from the corresponding Excel worksheet.
        """

        # Step #1 (See 'Notes' section in the docstring)
        co2_data_available = list(filter(lambda i: "Prod CO2" in i, self.sheets_loaded))

        # Step #2 (See 'Notes' section in the docstring)
        co2_data_attrs = ["prod_year", "lifting_rate", "price"]

        if len(co2_data_available) == 0:
            co2_data = {key: {"Prod CO2": None} for key in co2_data_attrs}

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

        # Step #3 (See 'Notes' section in the docstring)
        else:
            co2_data = {}
            co2_data_loaded = {i: self.data_loaded[i].fillna(0) for i in co2_data_available}

            for key, val_attr in enumerate(co2_data_attrs):
                co2_data[val_attr] = {}
                for i in co2_data_available:
                    if co2_data_loaded[i].empty:
                        co2_data[val_attr][i] = None
                    else:
                        co2_data[val_attr][i] = (
                            co2_data_loaded[i].iloc[:, key]
                            .to_numpy()
                        )

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

    def _get_tangible_cost_data(self) -> None | TangibleCostData:
        """
        Retrieves tangible cost data based on available sheets.

        Returns
        -------
        None or TangibleCostData

        Notes
        -----
        The core operations are as follows:
        (1) Capture tangible cost data from attribute self.data_loaded, then perform the
            necessary adjustment,
        (2) If 'tangible_data_loaded' is an empty dataframe, then return None,
        (3) If 'tangible_data_loaded' is not an empty dataframe, then load and
            arrange the attributes of interest from variable 'tangible_data_loaded'.
            Afterwards, create a new instance of TangibleCostData to store them.
        """
        # Step #1 (See 'Notes' section in the docstring)
        tangible_data_loaded = self.data_loaded["Cost Tangible"].dropna(axis=0, how="all")

        # Step #2 (See 'Notes' section in the docstring)
        if tangible_data_loaded.empty:
            return None

        # Step #3 (See 'Notes' section in the docstring)
        else:
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
                key: tangible_data_loaded.iloc[:, i].to_numpy()
                for i, key in enumerate(tangible_data_attrs)
            }

            return TangibleCostData(
                expense_year_init=tangible_data["expense_year"],
                cost_allocation=tangible_data["cost_allocation"].tolist(),
                cost=tangible_data["cost"],
                pis_year=tangible_data["pis_year"],
                useful_life=tangible_data["useful_life"],
                depreciation_factor=tangible_data["depreciation_factor"],
                salvage_value=tangible_data["salvage_value"],
                is_ic_applied=tangible_data["is_ic_applied"].tolist(),
                vat_portion=tangible_data["vat_portion"],
                lbt_portion=tangible_data["lbt_portion"],
                description=tangible_data["description"].tolist(),
                data_length=tangible_data_loaded.shape[0],
                project_years=self.general_config_data.project_years,
                type_of_contract=self.general_config_data.type_of_contract,
                end_date_project=self.general_config_data.end_date_project,
                start_date_project_second=self.general_config_data.start_date_project_second,
            )

    def _get_intangible_cost_data(self) -> None | IntangibleCostData:
        """
        Retrieves intangible cost data based on available sheets.

        Returns
        -------
        None or IntangibleCostData

        Notes
        -----
        The core operations are as follows:
        (1) Capture intangible data from attribute self.data_loaded, then perform the
            necessary adjustment,
        (2) If 'intangible_data_loaded' is an empty dataframe, then return None,
        (3) If 'intangible_data_loaded' is not an empty dataframe, then load and
            arrange the attributes of interest from variable 'intangible_data_loaded'.
            Afterwards, create a new instance of IntangibleCostData to store them.
        """
        # Step #1 (See 'Notes' section in the docstring)
        intangible_data_loaded = self.data_loaded["Cost Intangible"].dropna(axis=0, how="all")

        # Step #2 (See 'Notes' section in the docstring)
        if intangible_data_loaded.empty:
            return None

        # Step #3 (See 'Notes' section in the docstring)
        else:
            intangible_data_attrs = [
                "expense_year",
                "cost_allocation",
                "cost",
                "vat_portion",
                "lbt_portion",
                "description",
            ]

            intangible_data = {
                key: intangible_data_loaded.iloc[:, i].to_numpy()
                for i, key in enumerate(intangible_data_attrs)
            }

            return IntangibleCostData(
                expense_year_init=intangible_data["expense_year"],
                cost_allocation=intangible_data["cost_allocation"].tolist(),
                cost=intangible_data["cost"],
                vat_portion=intangible_data["vat_portion"],
                lbt_portion=intangible_data["lbt_portion"],
                description=intangible_data["description"].tolist(),
                data_length=intangible_data_loaded.shape[0],
                project_years=self.general_config_data.project_years,
                type_of_contract=self.general_config_data.type_of_contract,
                end_date_project=self.general_config_data.end_date_project,
                start_date_project_second=self.general_config_data.start_date_project_second,
            )

    def _get_opex_data(self) -> None | OPEXData:
        """
        Retrieves opex data based on available sheets.

        Returns
        -------
        None or OPEXData

        Notes
        -----
        The core operations are as follows:
        (1) Capture opex data from attribute self.data_loaded, then perform the
            necessary adjustment,
        (2) If 'opex_data_loaded' is an empty dataframe, then return None,
        (3) If 'opex_data_loaded' is not an empty dataframe, then load and
            arrange the attributes of interest from variable 'opex_data_loaded'.
            Afterwards, create a new instance of OPEXData to store them.
        """
        # Step #1 (See 'Notes' section in the docstring)
        opex_data_loaded = self.data_loaded["Cost OPEX"].dropna(axis=0, how="all")

        # Step #2 (See 'Notes' section in the docstring)
        if opex_data_loaded.empty:
            return None

        # Step #3 (See 'Notes' section in the docstring)
        else:
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
                key: opex_data_loaded.iloc[:, i].to_numpy()
                for i, key in enumerate(opex_data_attrs)
            }

            return OPEXData(
                expense_year_init=opex_data["expense_year"],
                cost_allocation=opex_data["cost_allocation"].tolist(),
                fixed_cost=opex_data["fixed_cost"],
                prod_rate=opex_data["prod_rate"],
                cost_per_volume=opex_data["cost_per_volume"],
                vat_portion=opex_data["vat_portion"],
                lbt_portion=opex_data["lbt_portion"],
                description=opex_data["description"].tolist(),
                data_length=opex_data_loaded.shape[0],
                project_years=self.general_config_data.project_years,
                type_of_contract=self.general_config_data.type_of_contract,
                end_date_project=self.general_config_data.end_date_project,
                start_date_project_second=self.general_config_data.start_date_project_second,
            )

    def _get_asr_cost_data(self) -> None | ASRCostData:
        """
        Retrieves ASR cost data based on available sheets.

        Returns
        -------
        None or ASRCostData

        Notes
        -----
        The core operations are as follows:
        (1) Capture asr cost data from attribute self.data_loaded, then perform the
            necessary adjustment,
        (2) If 'asr_data_loaded' is an empty dataframe, then return None,
        (3) If 'asr_data_loaded' is not an empty dataframe, then load and
            arrange the attributes of interest from variable 'asr_data_loaded'.
            Afterwards, create a new instance of ASRCostData to store them.
        """
        # Step #1 (See 'Notes' section in the docstring)
        asr_data_loaded = self.data_loaded["Cost ASR"].dropna(axis=0, how="all")

        # Step #2 (See 'Notes' section in the docstring)
        if asr_data_loaded.empty:
            return None

        # Step #3 (See 'Notes' section in the docstring)
        else:
            asr_data_attrs = [
                "expense_year",
                "cost_allocation",
                "cost",
                "vat_portion",
                "lbt_portion",
                "description",
            ]

            asr_data = {
                key: asr_data_loaded.iloc[:, i].to_numpy()
                for i, key in enumerate(asr_data_attrs)
            }

            return ASRCostData(
                expense_year_init=asr_data["expense_year"],
                cost_allocation=asr_data["cost_allocation"].tolist(),
                cost=asr_data["cost"],
                vat_portion=asr_data["vat_portion"],
                lbt_portion=asr_data["lbt_portion"],
                description=asr_data["description"].tolist(),
                data_length=asr_data_loaded.shape[0],
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
        sensitivity_data_loaded = self.data_loaded["Sensitivity"].iloc[0:10, 1:3]

        return SensitivityData(
            parameter=sensitivity_data_loaded.iloc[5:, 0].to_numpy().tolist(),
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
        mc_data_loaded = self.data_loaded["Uncertainty"].iloc[1:6, np.r_[1:3, 4, 6:8]]

        return MonteCarloData(
            parameter=mc_data_loaded.iloc[:, 0].to_numpy(),
            distribution=mc_data_loaded.iloc[:, 1].to_numpy(),
            min_values=mc_data_loaded.iloc[:, 2].to_numpy(),
            max_values=mc_data_loaded.iloc[:, 3].to_numpy(),
            std_dev=mc_data_loaded.iloc[:, 4].to_numpy(),
            data_length=mc_data_loaded.shape[0],
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
            .replace(np.nan, None)
        )

        # Step #2 (See 'Notes' section in the docstring)
        data_cr_init = {
            key: optimization_data_loaded.iloc[3:12, i].to_numpy()
            for i, key in enumerate(["parameter", "priority", "min", "max"])
        }

        # Step #3 (See 'Notes' section in the docstring)
        data_gs_init = {
            key: optimization_data_loaded.iloc[14:, i].to_numpy()
            for i, key in enumerate(["parameter", "priority", "min", "max"])
        }

        # Step #4 (See 'Notes' section in the docstring)
        target = {
            key: optimization_data_loaded.iloc[0, i + 4]
            for i, key in enumerate(["parameter", "value"])
        }

        # Step #5 (See 'Notes' section in the docstring)
        return OptimizationData(
            target=target,
            data_cr_init=data_cr_init,
            data_gs_init=data_gs_init,
        )

    def prepare_data(self) -> None:
        """
        Prepare data for optimization.

        Reads and prepare data from a target Excel file and fills in attributes
        associated with various categories, including configuration, lifting, cost,
        contract, and additional functionality.

        Returns
        -------
        None
            This method modifies the instance attributes of the class.

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
        # self.gas_lifting_data = self._get_gas_lifting_data()
        self.lpg_propane_lifting_data = self._get_lpg_propane_lifting_data()
        self.lpg_butane_lifting_data = self._get_lpg_butane_lifting_data()
        # self.sulfur_lifting_data = self._get_sulfur_lifting_data()
        # self.electricity_lifting_data = self._get_electricity_lifting_data()
        # self.co2_lifting_data = self._get_co2_lifting_data()
        #
        # # Fill in the attributes associated with cost data
        # self.tangible_cost_data = self._get_tangible_cost_data()
        # self.intangible_cost_data = self._get_intangible_cost_data()
        # self.opex_data = self._get_opex_data()
        # self.asr_cost_data = self._get_asr_cost_data()
        #
        # # Fill in the attributes associated with contract data
        # self.psc_cr_data = self._get_psc_cr_data()
        # self.psc_gs_data = self._get_psc_gs_data()
        # self.psc_transition_cr_to_cr = self._get_psc_transition_cr_to_cr()
        # self.psc_transition_cr_to_gs = self._get_psc_transition_cr_to_gs()
        # self.psc_transition_gs_to_gs = self._get_psc_transition_gs_to_gs()
        # self.psc_transition_gs_to_cr = self._get_psc_transition_gs_to_cr()

        print('\t')
        print(f'Filetype: {type(self.lpg_butane_lifting_data)}')
        print('lpg_butane_lifting_data = \n', self.lpg_butane_lifting_data)

        print('\t')
        print('=========================================================================================')
