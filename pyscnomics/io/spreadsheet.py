"""
Manage input-output data from and to a target Excel file.
"""

import os
import numpy as np
import pandas as pd
from datetime import date
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
        The name of the target Excel file. Must be given in '.xlsm' format.
        Defaults to None.
    """

    workbook_to_read: str = field(default=None)

    # Attributes associated with loading data from a target Excel file
    sheets_name: list = field(default=None, init=False, repr=False)
    sheets_loaded: list = field(default=None, init=False, repr=False)
    data_loaded: dict = field(default=None, init=False, repr=False)

    # Attributes associated with data preparation
    general_config_data: GeneralConfigData = field(default=None, init=False)
    fiscal_config_data: FiscalConfigData = field(default=None, init=False)
    oil_lifting_data: OilLiftingData = field(default=None, init=False)
    gas_lifting_data: GasLiftingData = field(default=None, init=False)
    lpg_propane_lifting_data: LPGPropaneLiftingData = field(default=None, init=False)
    lpg_butane_lifting_data: LPGButaneLiftingData = field(default=None, init=False)
    sulfur_lifting_data: SulfurLiftingData = field(default=None, init=False)
    electricity_lifting_data: ElectricityLiftingData = field(default=None, init=False)
    co2_lifting_data: CO2LiftingData = field(default=None, init=False)
    tangible_cost_data: TangibleCostData = field(default=None, init=False)
    intangible_cost_data: IntangibleCostData = field(default=None, init=False)
    opex_data: OPEXData = field(default=None, init=False)
    asr_cost_data: ASRCostData = field(default=None, init=False)

    def __post_init__(self):
        # Configure attribute workbook_to_read
        if self.workbook_to_read is None:
            self.workbook_to_read = "Workbook.xlsm"

        if self.workbook_to_read is not None:
            if not isinstance(self.workbook_to_read, str):
                raise SpreadsheetException(
                    f"Excel filename must be provided in str format. "
                    f"{self.workbook_to_read} ({self.workbook_to_read.__class__.__qualname__}) "
                    f"is not a str format."
                )

            if ".xlsm" not in self.workbook_to_read:
                raise SpreadsheetException(
                    f"Excel filename must be provided in '.xlsm' format."
                )

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
        # Directory location of the target Excel file
        load_dir = os.path.join(os.getcwd(), self.workbook_to_read)

        # Identify worksheets in the target Excel file
        excel = pd.ExcelFile(load_dir)
        sheets = excel.book.worksheets
        self.sheets_name = [sh.title for sh in sheets]
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
                engine="openpyxl",
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
        (1) Slice the data, only capture the columns that contain necessary data,
        (2) Drop all-column-NaN values from the associated dataframe,
        (3) Convert the remaining NaN values to None,
        (4) Assign data to their associated attributes,
        (5) Return a new instance of GeneralConfigData with filled attributes.
        """
        # Step #1 - Step #3 (See 'Notes' section in the docstring)
        general_config_data_loaded = (
            self.data_loaded["General Config"]
            .iloc[:, np.r_[1:3, -1]]
            .dropna(axis=0, how="all")
            .replace(np.nan, None)
        )

        # Step #4 - Step #5 (See 'Notes' section in the docstring)
        type_of_contract = general_config_data_loaded.iloc[0, 1]
        discount_rate_start_year = general_config_data_loaded.iloc[1, 1]
        discount_rate = general_config_data_loaded.iloc[2, 1]
        inflation_rate_applied_to = general_config_data_loaded.iloc[3, 1]
        start_date_project = general_config_data_loaded.iloc[4, 1]
        end_date_project = general_config_data_loaded.iloc[5, 1]
        oil_onstream_date = general_config_data_loaded.iloc[6, 1]
        gas_onstream_date = general_config_data_loaded.iloc[7, 1]
        vat_discount = general_config_data_loaded.iloc[10, 1]
        lbt_discount = general_config_data_loaded.iloc[11, 1]
        gsa_number = general_config_data_loaded.iloc[9, 2]

        return GeneralConfigData(
            start_date_project=start_date_project,
            end_date_project=end_date_project,
            type_of_contract=type_of_contract,
            oil_onstream_date=oil_onstream_date,
            gas_onstream_date=gas_onstream_date,
            discount_rate_start_year=discount_rate_start_year,
            discount_rate=discount_rate,
            inflation_rate_applied_to=inflation_rate_applied_to,
            vat_discount=float(vat_discount),
            lbt_discount=float(lbt_discount),
            gsa_number=int(gsa_number),
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
        (2) Drop NaN values from the associated dataframe,
        (3) Convert the remaining NaN values to zero,
        (4) Assign data to their associated attributes,
        (5) Return a new instance of FiscalConfigData with filled attributes.
        """
        # Step #1 - Step #3 (See 'Notes' section in the docstring)
        fiscal_config_data_loaded = (
            self.data_loaded["Fiscal Config"]
            .iloc[:, 1:]
            # .dropna(axis=0, how="all")
            # .replace(np.nan, None)
        )

        # print('\t')
        # print(f'Filetype: {type(fiscal_config_data_loaded)}')
        # print(f'Length: {len(fiscal_config_data_loaded)}')
        # print('fiscal_config_data_loaded = ', fiscal_config_data_loaded)

        # Step #4 - Step #5 (See 'Notes' section in the docstring)
        tax_mode = fiscal_config_data_loaded.iloc[0, 1]
        tax_rate_input = fiscal_config_data_loaded.iloc[1, 1]
        tax_payment_method = fiscal_config_data_loaded.iloc[11, 1]
        tax_psc_cost_recovery = fiscal_config_data_loaded.iloc[14, 1]
        npv_mode = fiscal_config_data_loaded.iloc[17, 1]
        future_rate_asr = fiscal_config_data_loaded.iloc[20, 1]
        depreciation_method = fiscal_config_data_loaded.iloc[23, 1]
        tax_multi = {
            "year": fiscal_config_data_loaded.iloc[4:10, 0].to_numpy(),
            "tax_rate": fiscal_config_data_loaded.iloc[4:10, 1].to_numpy(),
        }

        # year_arr = fiscal_config_data_loaded.iloc[4:10, 0]
        # tax_arr = fiscal_config_data_loaded.iloc[4:10, 1]

        return FiscalConfigData(
            tax_mode=tax_mode,
            tax_rate_input=tax_rate_input,
            tax_payment_method=tax_payment_method,
            tax_psc_cost_recovery=tax_psc_cost_recovery,
            npv_mode=npv_mode,
            future_rate_asr=future_rate_asr,
            depreciation_method=depreciation_method,
            tax_multi=tax_multi,
            project_years=self.general_config_data.project_years,
        )

    def _get_oil_lifting_data(self) -> OilLiftingData:
        """
        Retrieves oil lifting data based on available sheets.

        Returns
        -------
        OilLiftingData
            An instance of the OilLiftingData class containing the following attributes:
                - project_duration: int
                    The duration of the project.
                - project_years: numpy.ndarray
                    An array representing the project years.
                - prod_year: dict
                    Dictionary containing production years data.
                - oil_lifting_rate: dict
                    Dictionary containing oil lifting rate data.
                - oil_price: dict
                    Dictionary containing oil price data.
                - condensate_lifting_rate: dict
                    Dictionary containing condensate lifting rate data.
                - condensate_price: dict
                    Dictionary containing condensate price data.

        Notes
        -----
        The undertaken procedures are as follows:
        (1) Filter attribute self.sheets_loaded for sheets that contain 'Prod Oil' data,
            then assigned it as local variable named 'oil_data_available',
        (2) If 'oil_data_available' is empty (length is zero), then return a new instance
            of OilLiftingData with the associated attributes set to None,
        (3) If 'oil_data_available' is not empty, first check whether a particular
            self.data_loaded[oil_data_available] is an empty dataframe. If it is, then
            create a new instance of OilLiftingData with the associated attributes set
            to None. If a particular self.data_loaded[oil_data_available] is not an empty
            dataframe, then create a new instance of OilLiftingData with the associated
            attributes set as the loaded value from the corresponding Excel worksheet.
        """

        # Step #1 (See 'Notes' section in the docstring)
        oil_data_available = list(filter(lambda i: "Prod Oil" in i, self.sheets_loaded))

        # Step #2 (See 'Notes' section in the docstring)
        oil_attrs = [
            "prod_year",
            "oil_lifting_rate",
            "oil_price",
            "condensate_lifting_rate",
            "condensate_price",
        ]

        if len(oil_data_available) == 0:
            oil_data = {key: {"Prod Oil": None}for key in oil_attrs}

            return OilLiftingData(
                prod_year=oil_data["prod_year"],
                oil_lifting_rate=oil_data["oil_lifting_rate"],
                oil_price=oil_data["oil_price"],
                condensate_lifting_rate=oil_data["condensate_lifting_rate"],
                condensate_price=oil_data["condensate_price"],
                project_duration=self.general_config_data.project_duration,
                project_years=self.general_config_data.project_years,
            )

        # Step #3 (See 'Notes' section in the docstring)
        else:
            oil_data = {}
            oil_data_loaded = {i: self.data_loaded[i].fillna(0) for i in oil_data_available}

            for key, val_attr in enumerate(oil_attrs):
                oil_data[val_attr] = {}
                for i in oil_data_available:
                    if oil_data_loaded[i].empty:
                        oil_data[val_attr][i] = None
                    else:
                        oil_data[val_attr][i] = (
                            oil_data_loaded[i].iloc[:, key]
                            .to_numpy()
                        )

            return OilLiftingData(
                prod_year=oil_data["prod_year"],
                oil_lifting_rate=oil_data["oil_lifting_rate"],
                oil_price=oil_data["oil_price"],
                condensate_lifting_rate=oil_data["condensate_lifting_rate"],
                condensate_price=oil_data["condensate_price"],
                project_duration=self.general_config_data.project_duration,
                project_years=self.general_config_data.project_years,
            )

    def _get_gas_lifting_data(self) -> GasLiftingData:
        """
        Retrieves gas lifting data based on available sheets.

        Returns
        -------
        GasLiftingData
            An instance of GasLiftingData class containing the following attributes:
                - gas_gsa_number: int
                    The number of GSA.
                - prod_year: dict
                    Dictionary containing production years data.
                - gas_lifting_rate: dict
                    Dictionary containing gas lifting rate data.
                - gas_gsa_lifting_rate: dict
                    Dictionary containing gas GSA lifting rate data.
                - gas_gsa_ghv: dict
                    Dictionary containing gas GSA ghv data.
                - gas_gsa_price: dict
                    Dictionary containing gas GSA price data.
                - project_duration: int
                    The duration of the project.
                - project_years: numpy.ndarray
                    An array representing the project years.

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
                prod_year=gas_data["prod_year"],
                gas_prod_rate=gas_data["gas_prod_rate"],
                gas_gsa_lifting_rate=gas_data["gas_gsa_lifting_rate"],
                gas_gsa_ghv=gas_data["gas_gsa_ghv"],
                gas_gsa_price=gas_data["gas_gsa_price"],
                project_duration=self.general_config_data.project_duration,
                project_years=self.general_config_data.project_years,
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
                prod_year=gas_data["prod_year"],
                gas_prod_rate=gas_data["gas_prod_rate"],
                gas_gsa_lifting_rate=gas_data["gas_gsa_lifting_rate"],
                gas_gsa_ghv=gas_data["gas_gsa_ghv"],
                gas_gsa_price=gas_data["gas_gsa_price"],
                project_duration=self.general_config_data.project_duration,
                project_years=self.general_config_data.project_years,
            )

    def _get_lpg_propane_lifting_data(self) -> LPGPropaneLiftingData:
        """
        Retrieves LPG propane lifting data based on available sheets.

        Returns
        -------
        LPGPropaneLiftingData
            An instance of the LPGPropaneLiftingData class containing the following attributes:
                - prod_year: dict
                    Dictionary containing production years data.
                - lifting_rate: dict
                    Dictionary containing LPG propane lifting rate data.
                - price: dict
                    Dictionary containing LPG propane price data.
                - project_duration: int
                    The duration of the project.
                - project_years: numpy.ndarray
                    An array representing the project years.

        Notes
        -----
        The undertaken procedures are as follows:
        (1) Filter attribute self.sheets_loaded for sheets that contain 'Prod LPG Propane'
            data, then assigned it as local variable named 'lpg_propane_data_available',
        (2) If 'lpg_propane_data_available' is empty (length is zero), then return a new
            instance of LPGPropaneLiftingData with the associated attributes set to None,
        (3) If 'lpg_propane_data_available' is not empty, first check whether a particular
            self.data_loaded[lpg_propane_data_available] is an empty dataframe. If it is, then
            create a new instance of LPGPropaneLiftingData with the associated attributes set
            to None. If a particular self.data_loaded[lpg_propane_data_available] is not an empty
            dataframe, then create a new instance of LPGPropaneLiftingData with the associated
            attributes set as the loaded value from the corresponding Excel worksheet.
        """

        # Step #1 (See 'Notes' section in the docstring)
        lpg_propane_data_available = list(
            filter(lambda i: "Prod LPG Propane" in i, self.sheets_loaded)
        )

        # Step #2 (See 'Notes' section in the docstring)
        lpg_propane_attrs = ["prod_year", "lifting_rate", "price"]

        if len(lpg_propane_data_available) == 0:
            lpg_propane_data = {key: {"Prod LPG Propane": None} for key in lpg_propane_attrs}

            return LPGPropaneLiftingData(
                prod_year=lpg_propane_data["prod_year"],
                lifting_rate=lpg_propane_data["lifting_rate"],
                price=lpg_propane_data["price"],
                project_duration=self.general_config_data.project_duration,
                project_years=self.general_config_data.project_years,
            )

        # Step #3 (See 'Notes' section in the docstring)
        else:
            lpg_propane_data = {}
            lpg_propane_data_loaded = {
                i: self.data_loaded[i].fillna(0) for i in lpg_propane_data_available
            }

            for key, val_attr in enumerate(lpg_propane_attrs):
                lpg_propane_data[val_attr] = {}
                for i in lpg_propane_data_available:
                    if lpg_propane_data_loaded[i].empty:
                        lpg_propane_data[val_attr][i] = None
                    else:
                        lpg_propane_data[val_attr][i] = (
                            lpg_propane_data_loaded[i].iloc[:, key]
                            .to_numpy()
                        )

            return LPGPropaneLiftingData(
                prod_year=lpg_propane_data["prod_year"],
                lifting_rate=lpg_propane_data["lifting_rate"],
                price=lpg_propane_data["price"],
                project_duration=self.general_config_data.project_duration,
                project_years=self.general_config_data.project_years,
            )

    def _get_lpg_butane_lifting_data(self) -> LPGButaneLiftingData:
        """
        Retrieves LPG butane lifting data based on available sheets.

        Returns
        -------
        LPGButaneLiftingData
            An instance of the LPGButaneLiftingData class containing the following attributes:
                - prod_year: dict
                    Dictionary containing production years data.
                - lifting_rate: dict
                    Dictionary containing LPG butane lifting rate data.
                - price: dict
                    Dictionary containing LPG butane price data.
                - project_duration: int
                    The duration of the project.
                - project_years: numpy.ndarray
                    An array representing the project years.

        Notes
        -----
        The undertaken procedures are as follows:
        (1) Filter attribute self.sheets_loaded for sheets that contain 'Prod LPG Butane'
            data, then assigned it as local variable named 'lpg_butane_data_available',
        (2) If 'lpg_butane_data_available' is empty (length is zero), then return a new
            instance of LPGButaneLiftingData with the associated attributes set to None,
        (3) If 'lpg_butane_data_available' is not empty, first check whether a particular
            self.data_loaded[lpg_butane_data_available] is an empty dataframe. If it is, then
            create a new instance of LPGButaneLiftingData with the associated attributes set
            to None. If a particular self.data_loaded[lpg_butane_data_available] is not an empty
            dataframe, then create a new instance of LPGButaneLiftingData with the associated
            attributes set as the loaded value from the corresponding Excel worksheet.
        """

        # Step #1 (See 'Notes' section in the docstring)
        lpg_butane_data_available = list(
            filter(lambda i: "Prod LPG Butane" in i, self.sheets_loaded)
        )

        # Step #2 (See 'Notes' section in the docstring)
        lpg_butane_attrs = ["prod_year", "lifting_rate", "price"]

        if len(lpg_butane_data_available) == 0:
            lpg_butane_data = {key: {"Prod LPG Butane": None} for key in lpg_butane_attrs}

            return LPGButaneLiftingData(
                prod_year=lpg_butane_data["prod_year"],
                lifting_rate=lpg_butane_data["lifting_rate"],
                price=lpg_butane_data["price"],
                project_duration=self.general_config_data.project_duration,
                project_years=self.general_config_data.project_years,
            )

        # Step #3 (See 'Notes' section in the docstring)
        else:
            lpg_butane_data = {}
            lpg_butane_data_loaded = {
                i: self.data_loaded[i].fillna(0) for i in lpg_butane_data_available
            }

            for key, val_attr in enumerate(lpg_butane_attrs):
                lpg_butane_data[val_attr] = {}
                for i in lpg_butane_data_available:
                    if lpg_butane_data_loaded[i].empty:
                        lpg_butane_data[val_attr][i] = None
                    else:
                        lpg_butane_data[val_attr][i] = (
                            lpg_butane_data_loaded[i].iloc[:, key]
                            .to_numpy()
                        )

            return LPGButaneLiftingData(
                prod_year=lpg_butane_data["prod_year"],
                lifting_rate=lpg_butane_data["lifting_rate"],
                price=lpg_butane_data["price"],
                project_duration=self.general_config_data.project_duration,
                project_years=self.general_config_data.project_years,
            )

    def _get_sulfur_lifting_data(self) -> SulfurLiftingData:
        """
        Retrieves sulfur lifting data based on available sheets.

        Returns
        -------
        SulfurLiftingData
            An instance of the SulfurLiftingData class containing the following attributes:
                - prod_year: dict
                    Dictionary containing production years data.
                - lifting_rate: dict
                    Dictionary containing sulfur lifting rate data.
                - price: dict
                    Dictionary containing sulfur price data.
                - project_duration: int
                    The duration of the project.
                - project_years: numpy.ndarray
                    An array representing the project years.

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
                prod_year=sulfur_data["prod_year"],
                lifting_rate=sulfur_data["lifting_rate"],
                price=sulfur_data["price"],
                project_duration=self.general_config_data.project_duration,
                project_years=self.general_config_data.project_years,
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
                prod_year=sulfur_data["prod_year"],
                lifting_rate=sulfur_data["lifting_rate"],
                price=sulfur_data["price"],
                project_duration=self.general_config_data.project_duration,
                project_years=self.general_config_data.project_years,
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
                prod_year=electricity_data["prod_year"],
                lifting_rate=electricity_data["lifting_rate"],
                price=electricity_data["price"],
                project_duration=self.general_config_data.project_duration,
                project_years=self.general_config_data.project_years,
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
                prod_year=electricity_data["prod_year"],
                lifting_rate=electricity_data["lifting_rate"],
                price=electricity_data["price"],
                project_duration=self.general_config_data.project_duration,
                project_years=self.general_config_data.project_years,
            )

    def _get_co2_lifting_data(self) -> CO2LiftingData:
        """
        Retrieves CO2 lifting data based on available sheets.

        Returns
        -------
        CO2LiftingData
            An instance of the CO2LiftingData class containing the following attributes:
                - prod_year: dict
                    Dictionary containing production years data.
                - lifting_rate: dict
                    Dictionary containing CO2 lifting rate data.
                - price: dict
                    Dictionary containing CO2 price data.
                - project_duration: int
                    The duration of the project.
                - project_years: numpy.ndarray
                    An array representing the project years.

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
                prod_year=co2_data["prod_year"],
                lifting_rate=co2_data["lifting_rate"],
                price=co2_data["price"],
                project_duration=self.general_config_data.project_duration,
                project_years=self.general_config_data.project_years,
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
            prod_year=co2_data["prod_year"],
            lifting_rate=co2_data["lifting_rate"],
            price=co2_data["price"],
            project_duration=self.general_config_data.project_duration,
            project_years=self.general_config_data.project_years,
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
                expense_year=tangible_data["expense_year"],
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
                expense_year=intangible_data["expense_year"],
                cost_allocation=intangible_data["cost_allocation"].tolist(),
                cost=intangible_data["cost"],
                vat_portion=intangible_data["vat_portion"],
                lbt_portion=intangible_data["lbt_portion"],
                description=intangible_data["description"].tolist(),
                data_length=intangible_data_loaded.shape[0],
                project_years=self.general_config_data.project_years,
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
                expense_year=opex_data["expense_year"],
                cost_allocation=opex_data["cost_allocation"].tolist(),
                fixed_cost=opex_data["fixed_cost"],
                prod_rate=opex_data["prod_rate"],
                cost_per_volume=opex_data["cost_per_volume"],
                vat_portion=opex_data["vat_portion"],
                lbt_portion=opex_data["lbt_portion"],
                description=opex_data["description"].tolist(),
                data_length=opex_data_loaded.shape[0],
                project_years=self.general_config_data.project_years,
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
                expense_year=asr_data["expense_year"],
                cost_allocation=asr_data["cost_allocation"].tolist(),
                cost=asr_data["cost"],
                vat_portion=asr_data["vat_portion"],
                lbt_portion=asr_data["lbt_portion"],
                description=asr_data["description"].tolist(),
                data_length=asr_data_loaded.shape[0],
                project_years=self.general_config_data.project_years,
            )

    def prepare_data(self):
        """123"""
        # Read data from a target Excel file
        self.read_from_excel()

        # print("\t")
        # print("data loaded = \n", self.data_loaded.keys())

        # print('\t')
        # print(f'Filetype: {self.data_loaded["Prod Oil"]}')
        # print('oil prod = ', self.data_loaded["Prod Oil"])

        # Fill in the attributes associated with data preparation
        self.general_config_data = self._get_general_config_data()
        self.fiscal_config_data = self._get_fiscal_config_data()
        # self.oil_lifting_data = self._get_oil_lifting_data()
        # self.gas_lifting_data = self._get_gas_lifting_data()
        # self.lpg_propane_lifting_data = self._get_lpg_propane_lifting_data()
        # self.lpg_butane_lifting_data = self._get_lpg_butane_lifting_data()
        # self.sulfur_lifting_data = self._get_sulfur_lifting_data()
        # self.electricity_lifting_data = self._get_electricity_lifting_data()
        # self.co2_lifting_data = self._get_co2_lifting_data()
        # self.tangible_cost_data = self._get_tangible_cost_data()
        # self.intangible_cost_data = self._get_intangible_cost_data()
        # self.opex_data = self._get_opex_data()
        # self.asr_cost_data = self._get_asr_cost_data()
        #
        # print("\t")
        # print(f"Filetype: {type(self.tangible_cost_data)}")
        # print("tangible_cost_data = \n", self.tangible_cost_data)
