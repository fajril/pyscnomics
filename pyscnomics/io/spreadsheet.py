"""
Manage input-output data from and to a target Excel file.
"""

import os
import numpy as np
import pandas as pd
from datetime import date
from dataclasses import dataclass, field, asdict

from pyscnomics.io.config import (
    GeneralConfigData,
    FiscalConfigData,
    OilLiftingData,
    GasLiftingData,
)


class SpreadsheetException(Exception):
    """Exception to raise for a misuse of Spreadsheet class"""

    pass


@dataclass
class Spreadsheet:
    """
    Parameters
    ----------
    workbook_to_read: str
        The name of the target Excel file. Must be given in '.xlsm' format.
        Defaults to None.
    """

    workbook_to_read: str = field(default=None)

    # Attributes associated with loading data from a target Excel file
    sheets_raw: dict = field(default=None, init=False, repr=False)
    sheets_visible: list = field(default=None, init=False, repr=False)
    sheets_loaded: list = field(default=None, init=False, repr=False)
    data_loaded: dict = field(default=None, init=False, repr=False)

    # Attributes associated with data preparation
    general_config_data: GeneralConfigData = field(default=None, init=False)
    fiscal_config_data: FiscalConfigData = field(default=None, init=False)
    oil_lifting_data: OilLiftingData = field(default=None, init=False)
    gas_lifting_data: GasLiftingData = field(default=None, init=False)

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
        (2) From the target Excel file, identify 'visible' worksheets,
        (3) Load data from the target Excel file, capturing the 'visible' worksheets only.
        """
        # Directory location of the target Excel file
        load_dir = os.path.join(os.getcwd(), self.workbook_to_read)

        # Identify 'visible' worksheets in the target Excel file
        excel = pd.ExcelFile(load_dir)
        sheets = excel.book.worksheets
        self.sheets_raw = dict([(sh.title, sh.sheet_state) for sh in sheets])
        self.sheets_visible = [
            key for key, val in self.sheets_raw.items() if val == "visible"
        ]
        self.sheets_loaded = self.sheets_visible[3 : len(self.sheets_visible) - 2]

        # Load data from 'visible' worksheets only
        self.data_loaded = {
            i: pd.read_excel(
                excel,
                sheet_name=i,
                skiprows=1,
                index_col=None,
                header=None,
                engine="openpyxl",
            )
            for i in self.sheets_loaded
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

        # Prepare data
        self.data_loaded["General Config"] = self.data_loaded["General Config"].iloc[:, np.r_[1:3, -1]]
        self.data_loaded["General Config"] = (
            self.data_loaded["General Config"]
            .dropna(axis=0, how="all")
            .replace(np.nan, None)
        )

        # Assign the prepared data to their associated attributes
        type_of_contract = self.data_loaded["General Config"].iloc[0, 1]
        discount_rate_start_year = self.data_loaded["General Config"].iloc[1, 1]
        discount_rate = self.data_loaded["General Config"].iloc[2, 1]
        inflation_rate_applied_to = self.data_loaded["General Config"].iloc[3, 1]
        start_date_project = self.data_loaded["General Config"].iloc[4, 1]
        end_date_project = self.data_loaded["General Config"].iloc[5, 1]
        oil_onstream_date = self.data_loaded["General Config"].iloc[6, 1]
        gas_onstream_date = self.data_loaded["General Config"].iloc[7, 1]
        vat_discount = self.data_loaded["General Config"].iloc[10, 1]
        lbt_discount = self.data_loaded["General Config"].iloc[11, 1]
        gsa_number = self.data_loaded["General Config"].iloc[9, 2]

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
        (3) Convert NaN to None,
        (4) Assign data to their associated attributes,
        (5) Return a new instance of GeneralConfigData with filled attributes.
        """

        # Prepare data
        self.data_loaded["Fiscal Config"] = self.data_loaded["Fiscal Config"].iloc[:, 1:]
        self.data_loaded["Fiscal Config"] = (
            self.data_loaded["Fiscal Config"]
            .dropna(axis=0, how="all")
            .replace(np.nan, 0.0)
        )

        # Assign the prepared data to their associated attributes
        tax_mode = self.data_loaded["Fiscal Config"].iloc[0, 1]
        tax_payment_method = self.data_loaded["Fiscal Config"].iloc[9, 1]
        tax_psc_cost_recovery = self.data_loaded["Fiscal Config"].iloc[10, 1]
        npv_mode = self.data_loaded["Fiscal Config"].iloc[11, 1]
        future_rate_asr = self.data_loaded["Fiscal Config"].iloc[12, 1]
        depreciation_factor = self.data_loaded["Fiscal Config"].iloc[13, 1]
        tax_rate = self.data_loaded["Fiscal Config"].iloc[1, 1]
        year_arr = self.data_loaded["Fiscal Config"].iloc[3:9, 0]
        tax_rate_arr = self.data_loaded["Fiscal Config"].iloc[3:9, 1]

        return FiscalConfigData(
            tax_mode=tax_mode,
            tax_payment_method=tax_payment_method,
            tax_psc_cost_recovery=tax_psc_cost_recovery,
            npv_mode=npv_mode,
            future_rate_asr=future_rate_asr,
            depreciation_method=depreciation_factor,
            tax_rate=tax_rate,
            year_arr=year_arr.to_numpy(),
            tax_rate_arr=tax_rate_arr.to_numpy(),
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
            of OilLiftingData with the associated attributes set as an array of zeros,
        (3) If 'oil_data_available' is not empty, first check whether a particular
            self.data_loaded[oil_data_available] is an empty dataframe. If it is, then
            create a new instance of OilLiftingData with the associated attributes set
            as an array of zeros. If a particular self.data_loaded[oil_data_available]
            is not an empty dataframe, then create a new instance of OilLiftingData with
            the associated attributes set as the loaded value from the corresponding
            Excel worksheet.
        """

        # Step #1 (See 'Notes' section in the docstring)
        oil_data_available = list(filter(lambda i: "Prod Oil" in i, self.sheets_loaded))

        # Step #2 (See 'Notes' section in the docstring)
        if len(oil_data_available) == 0:
            oil_data = {
                key: {"Prod Oil": np.zeros_like(self.general_config_data.project_years)}
                for key in [
                    "prod_year",
                    "oil_lifting_rate",
                    "oil_price",
                    "condensate_lifting_rate",
                    "condensate_price",
                ]
            }

            return OilLiftingData(
                project_duration=self.general_config_data.project_duration,
                project_years=self.general_config_data.project_years,
                prod_year=oil_data["prod_year"],
                oil_lifting_rate=oil_data["oil_lifting_rate"],
                oil_price=oil_data["oil_price"],
                condensate_lifting_rate=oil_data["condensate_lifting_rate"],
                condensate_price=oil_data["condensate_price"],
            )

        # Step #3 (See 'Notes' section in the docstring)
        else:
            oil_data = {}
            oil_attrs = [
                "prod_year",
                "oil_lifting_rate",
                "oil_price",
                "condensate_lifting_rate",
                "condensate_price",
            ]

            for i, val_attr in enumerate(oil_attrs):
                oil_data[val_attr] = {}
                for j in oil_data_available:
                    if self.data_loaded[j].empty:
                        oil_data[val_attr][j] = np.zeros_like(
                            self.general_config_data.project_years
                        )
                    else:
                        self.data_loaded[j] = self.data_loaded[j].fillna(0)
                        oil_data[val_attr][j] = self.data_loaded[j].iloc[:, i].to_numpy()

            return OilLiftingData(
                project_duration=self.general_config_data.project_duration,
                project_years=self.general_config_data.project_years,
                prod_year=oil_data["prod_year"],
                oil_lifting_rate=oil_data["oil_lifting_rate"],
                oil_price=oil_data["oil_price"],
                condensate_lifting_rate=oil_data["condensate_lifting_rate"],
                condensate_price=oil_data["condensate_price"],
            )

    def _get_gas_lifting_data(self) -> GasLiftingData:

        gas_data_available = list(filter(lambda i: "Prod Gas" in i, self.sheets_loaded))
        gas_gsa_number = self.general_config_data.gsa_number

        if len(gas_data_available) == 0:
            gas_data = {}

        print('\t')
        print(f'Filetype: {type(gas_data_available)}')
        print(f'Length: {len(gas_data_available)}')
        print('gas_data_available = \n', gas_data_available)

        print('\t')
        print(f'Filetype: {type(gas_gsa_number)}')
        print('gas_gsa_number = \n', gas_gsa_number)


    def _get_lpg_propane_lifting_data(self):
        raise NotImplementedError

    def _get_lpg_butane_lifting_data(self):
        raise NotImplementedError

    def _get_sulfur_lifting_data(self):
        raise NotImplementedError

    def _get_electricity_lifting_data(self):
        raise NotImplementedError

    def _get_co2_lifting_data(self):
        raise NotImplementedError

    def _get_tangible_cost_data(self):
        raise NotImplementedError

    def _get_intangible_cost_data(self):
        raise NotImplementedError

    def _get_opex_data(self):
        raise NotImplementedError

    def _get_asr_cost_data(self):
        raise NotImplementedError

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
        self.oil_lifting_data = self._get_oil_lifting_data()
        # self.gas_lifting_data = self._get_gas_lifting_data()

        print('\t')
        print(f'Filetype: {type(self.oil_lifting_data.oil_price)}')
        print(f'Keys: {self.oil_lifting_data.oil_price.keys()}')
        print('self.oil_lifting_data = \n', self.oil_lifting_data.oil_price)

        # print('\t')
        # print('fiscal_config_data = \n', self.data_loaded["Fiscal Config"].iloc[:, 1])
