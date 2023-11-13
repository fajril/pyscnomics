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
        self.sheets_visible = [key for key, val in self.sheets_raw.items() if val == "visible"]
        self.sheets_loaded = self.sheets_visible[3:len(self.sheets_visible) - 2]

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
        (1) Slice the data, only capture columns that contain necessary data,
        (2) Drop NaN values from the associated dataframe,
        (3) Convert NaN to None,
        (4) Assign data to their associated attributes,
        (5) Return a new instance of GeneralConfigData with filled attributes.
        """

        # Prepare data
        self.data_loaded["General Config"] = self.data_loaded["General Config"].iloc[:, 1:3]
        self.data_loaded["General Config"] = self.data_loaded["General Config"].dropna(axis=0, how="all")
        self.data_loaded["General Config"] = (
            self.data_loaded["General Config"].where(
                pd.notnull(self.data_loaded["General Config"]), None
            )
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
        vat_discount = self.data_loaded["General Config"].iloc[9, 1]
        lbt_discount = self.data_loaded["General Config"].iloc[10, 1]

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
        self.data_loaded["Fiscal Config"] = self.data_loaded["Fiscal Config"].dropna(axis=0, how="all")
        self.data_loaded["Fiscal Config"] = (
            self.data_loaded["Fiscal Config"].where(
                pd.notnull(self.data_loaded["Fiscal Config"]), None
            )
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

    def _get_oil_lifting_data(self):
        t1 = np.sum(np.char.count(self.sheets_loaded, "Prod Gas"))
        raise NotImplementedError

    def _get_gas_lifting_data(self):
        raise NotImplementedError

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
        """ 123
        """
        # Read data from a target Excel file
        self.read_from_excel()

        print('\t')
        print('data loaded = \n', self.data_loaded.keys())

        # Fill in the attributes associated with data preparation
        self.general_config_data = self._get_general_config_data()
        self.fiscal_config_data = self._get_fiscal_config_data()

        print('\t')
        print(f'Filetype: {type(self.fiscal_config_data)}')
        print('fiscal_config_data = \n', self.fiscal_config_data)

        # print('\t')
        # print('fiscal_config_data = \n', self.data_loaded["Fiscal Config"].iloc[:, 1])
