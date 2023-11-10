"""
Manage input-output data from and to a target Excel file.
"""

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict


class SpreadsheetException(Exception):
    """ Exception to raise for a misuse of Spreadsheet class """

    pass


@dataclass
class Spreadsheet:
    """ 123
    """
    workbook_to_read: str = field(default=None)

    def __post_init__(self):
        # Configure attribute workbook_to_read
        if self.workbook_to_read is None:
            self.workbook_to_read = 'Workbook.xlsm'

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

    def read_from_excel(self):
        """ 123
        """
        load_dir = os.path.join(os.getcwd(), self.workbook_to_read)

        print('\t')
        print(f'Filetype: {type(load_dir)}')
        print('load_dir = ', load_dir)

        excel = pd.ExcelFile(load_dir)
        sheets = excel.book.worksheets
        sheet_titles = [i.title for i in sheets]

        print('\t')
        print(f'Filetype: {type(sheets)}')
        print(f'Length: {len(sheets)}')
        print('sheets = ', sheets)

        print('\t')
        print(f'Filetype: {type(sheet_titles)}')
        print(f'Length: {len(sheet_titles)}')
        print('sheet_titles = ', sheet_titles)





