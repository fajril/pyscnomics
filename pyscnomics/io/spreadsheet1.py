"""
Manage input-output data from and to a target Excel workbook.
"""

import os as os
import pandas as pd
from dataclasses import dataclass, field
from ionomics.config import (
    GeneralConfig,
    CostRecData,
    GrossSplitData,
    CostData,
    LiftingData
)


@dataclass
class Spreadsheet:
    """
    Read and prepare data input from a target Excel file.

    Parameters
    ----------
    run_economic_limit: bool
        Whether economic limit is applied or not.
    """

    run_economic_limit: bool = field(default=False)

    # Attributes to be defined later on
    loaded_data: dict = field(default=None, init=False)
    named_keys: list = field(default=None, init=False)
    economic_limit: int = field(default=None, init=False)
    general: GeneralConfig = field(default=None, init=False)
    psc_cr_1: CostRecData = field(default=None, init=False)
    psc_gs_1: GrossSplitData = field(default=None, init=False)
    psc_cr_2: CostRecData = field(default=None, init=False)
    psc_gs_2: GrossSplitData = field(default=None, init=False)
    cost: CostData = field(default=None, init=False)
    lifting: LiftingData = field(default=None, init=False)

    def __post_init__(self):

        # Instantiate attribute loaded_data as an empty dictionary
        self.loaded_data = {}

        # Instantiate attribute named_keys
        self.named_keys = [
            "general",
            "psc_cr_1",
            "psc_gs_1",
            "psc_cr_2",
            "psc_gs_2",
            "depreciation",
            "vat",
            "inflation",
            "gas_conversion_factor",
            "lbt",
            "pdrd",
            "effective_tax_rate",
            "rc_split",
            "icp_sliding_scale",
            "cost",
            "oil_sales",
            "gas_sales",
            "lpg_sales",
            "other_sales"
        ]

    def read_from_excel(self, workbook_to_load: str) -> dict:
        """
        Reads and filters the input data from a target Excel file and store it in CPU's memory.

        Parameters
        ----------
        workbook_to_load: str
            The name of the target Excel file.

        Returns
        -------
        loaded_data: dict
            The loaded data from an Excel file that has been filtered
            and classified into its corresponding category.
        """

        # Specify the directory where the target Excel workbook is located
        # os.chdir('..')
        load_dir = os.path.join(os.getcwd(), workbook_to_load)

        # Load the Excel worksheets in the target workbook
        wb = pd.ExcelFile(load_dir)
        ws_name = wb.sheet_names

        for name in ws_name:
            self.loaded_data[name] = pd.read_excel(
                wb,
                sheet_name=name,
                skiprows=3,
                index_col=None,
                header=None,
                engine='openpyxl'
            )

        # Create a sub-function to classify data
        def _classify_data(
                loaded_data: pd.DataFrame,
                target_worksheet: str,
                target_data: tuple
        ):
            """
            Classify the (raw) loaded data into its corresponding category.

            Parameters
            ----------
            loaded_data: pd.DataFrame
                The (raw) loded data from an Excel file.

            target_worksheet: str
                The name of the target worksheet where target data is located.

            target_data: tuple
                A pair of row and column indices of the data of interest,
                given as (row_index, column_index).

            Returns
            -------
            classified_loaded_data
                The loaded data that has been classified into its own corresponding category.
            """

            loaded_data = (
                loaded_data[target_worksheet].iloc[target_data[0], target_data[1]]
                .dropna(axis=0, how="all")
                .fillna(0)
                .values
            )

            return loaded_data

        # Define the target data in each target worksheets. Here, we specify the target data as a pair of
        # row and column indices of the data of interest at the corresponding target worksheet.
        target_data = [0 for _ in range(len(ws_name))]

        # Target data in worksheet #0 ('INPUT')
        target_data[0] = [
            (slice(0, 11), slice(1, 3)),        # general
            (slice(None), slice(5, 8)),         # psc_cr_1
            (slice(1, None), slice(9, 12)),     # psc_gs_1
            (slice(None), slice(13, 16)),       # psc_cr_2
            (slice(1, None), slice(17, None)),  # psc_gs_2
        ]

        # Target data in worksheet #1 ('CONFIG')
        target_data[1] = [
            (slice(None), slice(2, 8)),         # depreciation
            (slice(None), slice(10, 13)),       # ppn
            (slice(None), slice(15, 18)),       # inflation
            (slice(None), slice(20, 23)),       # gas_conversion_factor
            (slice(None), slice(25, 28)),       # pbb
            (slice(None), slice(30, 33)),       # pdrd
            (slice(None), slice(35, 38)),       # effective_tax_rate
            (slice(2, None), slice(39, 51)),    # rc_split
            (slice(2, 9), slice(52, None)),     # icp_sliding_scale
        ]

        # Target data in worksheet #2 ('COST')
        target_data[2] = [(slice(None), slice(2, None))]

        # Target data in worksheet #3 ('OIL SALES')
        target_data[3] = [(slice(1, None), slice(1, None))]

        # Target data in worksheet #4 ('GAS SALES')
        target_data[4] = [(slice(2, None), slice(1, None))]

        # Target data in worksheet #5 ('LPG SALES')
        target_data[5] = [(slice(1, None), slice(1, None))]

        # Target data in worksheet #6 ('OTHER SALES')
        target_data[6] = [(slice(2, None), slice(1, None))]

        # Data classification is carried out repeatedly for every target data in its corresponding
        # target worksheet. As a consequence, the procedure is undertaken in an iterative process.
        # Index i -> iteration over the target worksheets
        # Index j -> iteration over the target data
        count = 0

        for i in range(len(ws_name)):
            for j in range(len(target_data[i])):
                self.loaded_data[self.named_keys[count]] = _classify_data(
                    loaded_data=self.loaded_data,
                    target_worksheet=ws_name[i],
                    target_data=target_data[i][j],
                )

                count += 1

    def _prepare_general_data(self) -> GeneralConfig:
        """
        Prepare general data input.

        Returns
        -------
        general: GeneralConfig
            Attribute 'general' as an instance of GeneralConfig class.
        """

        # Check run economic limit
        if self.run_economic_limit is False:
            self.economic_limit = 0

        else:
            self.economic_limit = self.loaded_data["general"][10, 1]

        # Prepare general config data input
        general = GeneralConfig(
            start_year=self.loaded_data["general"][0, 1],
            end_year=self.loaded_data["general"][1, 1],
            contract_type=self.loaded_data["general"][3, 1],
            start_new_contract=self.loaded_data["general"][5, 1],
            year_new_contract=self.loaded_data["general"][6, 1],
            discount_rate_start=self.loaded_data["general"][7, 1],
            contract_basis=self.loaded_data["general"][4, 1],
            inflation_rate_applied_to=self.loaded_data["general"][8, 1],
            economic_limit=self.economic_limit,
            depreciation_config=self.loaded_data["depreciation"],
            vat_config=self.loaded_data["vat"],
            inflation_config=self.loaded_data["inflation"],
            gas_conversion_factor=self.loaded_data["gas_conversion_factor"],
            lbt_config=self.loaded_data["lbt"],
            pdrd_config=self.loaded_data["pdrd"],
            etr_config=self.loaded_data["effective_tax_rate"],
            rc_split=self.loaded_data["rc_split"],
            icp_sliding=self.loaded_data["icp_sliding_scale"]
        )

        return general

    def _prepare_psc_cr_data(self, data_key: str) -> CostRecData:
        """
        Prepare PSC Cost Recovery data input.

        Parameters
        ----------
        data_key: str
            A particular target data key.

        Returns
        -------
        psc_cr: CostRecData
            Attribute 'psc_cr' as an instance of CostRecData class.
        """

        target_key = self.loaded_data[data_key]

        psc_cr = CostRecData(
            split_config=target_key[0, 2],
            goi_tax={
                "pre_tax_oil": target_key[7, 1],
                "pre_tax_gas": target_key[9, 1],
                "post_tax_oil": target_key[2, 1],
                "post_tax_gas": target_key[4, 1]
            },
            ctr_tax={
                "pre_tax_oil": target_key[8, 1],
                "pre_tax_gas": target_key[10, 1],
                "post_tax_oil": target_key[3, 1],
                "post_tax_gas": target_key[5, 1]
            },
            corporate_income=target_key[12, 1],
            branch_profit=target_key[13, 1],
            effective_rate=target_key[14, 1],
            dmo_holiday={
                "oil": target_key[23, 1],
                "gas": target_key[29, 1]
            },
            dmo_period={
                "oil": target_key[24, 1],
                "gas": target_key[30, 1]
            },
            dmo_start_production={
                "oil": target_key[25, 1],
                "gas": target_key[31, 1]
            },
            dmo_volume={
                "oil": target_key[26, 1],
                "gas": target_key[32, 1]
            },
            dmo_fee={
                "oil": target_key[27, 1],
                "gas": target_key[33, 1]
            },
            ic_available={
                "oil": target_key[35, 2],
                "gas": target_key[36, 2]
            },
            ic_value={
                "oil": target_key[35, 1],
                "gas": target_key[36, 1]
            },
            ftp_available=target_key[16, 1],
            ftp_ctr_shared=target_key[17, 1],
            ftp_portion=target_key[18, 1],
            is_capped={
                "oil": target_key[20, 2],
                "gas": target_key[21, 2]
            },
            cr={
                "oil": target_key[20, 1],
                "gas": target_key[21, 1]
            },
            vat_categ=target_key[38, 2],
            vat_percent=target_key[38, 1],
            vat_discount=target_key[39, 1],
            lbt_discount=target_key[40, 1],
            pdrd=target_key[42, 1],
            pdrd_discount=target_key[41, 1],
            pdri=target_key[44, 1],
            pdri_discount=target_key[43, 1],
            import_duty=target_key[45, 1],
            vat_import=target_key[46, 1],
            pph_import=target_key[47, 1],
            depre_accel=target_key[48, 1],
            depre_method=target_key[49, 1]
        )

        return psc_cr

    def _prepare_psc_gs_data(self, data_key: str) -> GrossSplitData:
        """
        Prepare PSC Gross Split data input.

        Parameters
        ----------
        data_key: str
            A particular target data key.

        Returns
        -------
        psc_gs: GrossSplitData
            Attribute 'psc_gs' as an instance of GrossSplitData class.
        """

        target_key = self.loaded_data[data_key]

        psc_gs = GrossSplitData(
            field_status={
                "param": target_key[0, 1],
                "val": target_key[0, 2]
            },
            field_loc={
                "param": target_key[1, 1],
                "val": target_key[1, 2]
            },
            res_depth={
                "param": target_key[2, 1],
                "val": target_key[2, 2]
            },
            infra_avail={
                "param": target_key[3, 1],
                "val": target_key[3, 2]
            },
            res_type={
                "param": target_key[4, 1],
                "val": target_key[4, 2]
            },
            api_oil={
                "param": target_key[5, 1],
                "val": target_key[5, 2]
            },
            domestic_use={
                "param": target_key[6, 1],
                "val": target_key[6, 2]
            },
            prod_stage={
                "param": target_key[7, 1],
                "val": target_key[7, 2]
            },
            co2_content={
                "param": target_key[8, 1],
                "val": target_key[8, 2]
            },
            h2s_content={
                "param": target_key[9, 1],
                "val": target_key[9, 2]
            },
            total_split=target_key[10, 2],
            base_split={
                "ctr_oil": target_key[12, 1],
                "ctr_gas": target_key[13, 1],
                "ministry": target_key[16, 1]
            },
            tax_rate=target_key[18, 1],
            branch_profit=target_key[19, 1],
            effective_rate=target_key[20, 1],
            dmo_holiday={
                "oil": target_key[22, 1],
                "gas": target_key[28, 1]
            },
            dmo_period={
                "oil": target_key[23, 1],
                "gas": target_key[29, 1]
            },
            dmo_start_production={
                "oil": target_key[24, 1],
                "gas": target_key[30, 1]
            },
            dmo_volume={
                "oil": target_key[25, 1],
                "gas": target_key[31, 1]
            },
            dmo_fee={
                "oil": target_key[26, 1],
                "gas": target_key[32, 1]
            },
            ic_available={
                "oil": target_key[34, 2],
                "gas": target_key[35, 2]
            },
            ic_value={
                "oil": target_key[34, 1],
                "gas": target_key[35, 1]
            },
            vat_categ=target_key[37, 2],
            vat_percent=target_key[37, 1],
            vat_discount=target_key[38, 1],
            lbt_discount=target_key[39, 1],
            pdrd=target_key[41, 1],
            pdrd_discount=target_key[40, 1],
            pdri=target_key[43, 1],
            pdri_discount=target_key[42, 1],
            import_duty=target_key[44, 1],
            vat_import=target_key[45, 1],
            pph_import=target_key[46, 1],
            depre_accel=target_key[47, 1],
            depre_method=target_key[48, 1]
        )

        return psc_gs

    def _prepare_cost_data(self) -> CostData:
        """
        Prepare cost data input.

        Returns
        -------
        cost: CostData
            Attribute 'cost' as an instance of CostData class.
        """

        cost = CostData(
            start_year=self.loaded_data["general"][0, 1],
            end_year=self.loaded_data["general"][1, 1],
            cost_data=self.loaded_data["cost"]
        )

        return cost

    def _prepare_lifting_data(self) -> LiftingData:
        """
        Prepare lifting data input.

        Returns
        -------
        lifting: LiftingData
            Attribute 'lifting' as an instance of LiftingData class.
        """

        lifting = LiftingData(
            start_year=self.loaded_data["general"][0, 1],
            end_year=self.loaded_data["general"][1, 1],
            prod_rate={
                "oil": self.loaded_data["oil_sales"][:, 1],
                "con": self.loaded_data["oil_sales"][:, 5],
                "gsa": {
                    "gsa1": self.loaded_data["gas_sales"][:, 4],
                    "gsa2": self.loaded_data["gas_sales"][:, 7],
                    "gsa3": self.loaded_data["gas_sales"][:, 10],
                    "gsa4": self.loaded_data["gas_sales"][:, 13],
                    "gsa5": self.loaded_data["gas_sales"][:, 16],
                    "gsa6": self.loaded_data["gas_sales"][:, 19],
                    "gsa7": self.loaded_data["gas_sales"][:, 22],
                    "gsa8": self.loaded_data["gas_sales"][:, 25],
                    "gsa9": self.loaded_data["gas_sales"][:, 28],
                    "gsa10": self.loaded_data["gas_sales"][:, 31],
                    "gsa11": self.loaded_data["gas_sales"][:, 34],
                    "gsa12": self.loaded_data["gas_sales"][:, 37],
                    "gsa13": self.loaded_data["gas_sales"][:, 40],
                    "gsa14": self.loaded_data["gas_sales"][:, 43],
                    "gsa15": self.loaded_data["gas_sales"][:, 46],
                    "gsa16": self.loaded_data["gas_sales"][:, 49],
                    "gsa17": self.loaded_data["gas_sales"][:, 52],
                    "gsa18": self.loaded_data["gas_sales"][:, 55],
                    "gsa19": self.loaded_data["gas_sales"][:, 58],
                    "gsa20": self.loaded_data["gas_sales"][:, 61]
                },
                "lpg": {
                    "propane": self.loaded_data["lpg_sales"][:, 1],
                    "butane": self.loaded_data["lpg_sales"][:, 4]
                },
                "sulfur": self.loaded_data["other_sales"][:, 2],
                "electricity": self.loaded_data["other_sales"][:, 5],
                "co2": self.loaded_data["other_sales"][:, 8],
                "others": {
                    "other1": self.loaded_data["other_sales"][:, 11],
                    "other2": self.loaded_data["other_sales"][:, 14]
                }
            },
            price={
                "oil": self.loaded_data["oil_sales"][:, 4],
                "con": self.loaded_data["oil_sales"][:, 8],
                "gsa": {
                    "gsa1": self.loaded_data["gas_sales"][:, 6],
                    "gsa2": self.loaded_data["gas_sales"][:, 9],
                    "gsa3": self.loaded_data["gas_sales"][:, 12],
                    "gsa4": self.loaded_data["gas_sales"][:, 15],
                    "gsa5": self.loaded_data["gas_sales"][:, 18],
                    "gsa6": self.loaded_data["gas_sales"][:, 21],
                    "gsa7": self.loaded_data["gas_sales"][:, 24],
                    "gsa8": self.loaded_data["gas_sales"][:, 27],
                    "gsa9": self.loaded_data["gas_sales"][:, 30],
                    "gsa10": self.loaded_data["gas_sales"][:, 33],
                    "gsa11": self.loaded_data["gas_sales"][:, 36],
                    "gsa12": self.loaded_data["gas_sales"][:, 39],
                    "gsa13": self.loaded_data["gas_sales"][:, 42],
                    "gsa14": self.loaded_data["gas_sales"][:, 45],
                    "gsa15": self.loaded_data["gas_sales"][:, 48],
                    "gsa16": self.loaded_data["gas_sales"][:, 51],
                    "gsa17": self.loaded_data["gas_sales"][:, 54],
                    "gsa18": self.loaded_data["gas_sales"][:, 57],
                    "gsa19": self.loaded_data["gas_sales"][:, 60],
                    "gsa20": self.loaded_data["gas_sales"][:, 63]
                },
                "lpg": {
                    "propane": self.loaded_data["lpg_sales"][:, 2],
                    "butane": self.loaded_data["lpg_sales"][:, 5]
                },
                "sulfur": self.loaded_data["other_sales"][:, 3],
                "electricity": self.loaded_data["other_sales"][:, 6],
                "co2": self.loaded_data["other_sales"][:, 9],
                "others": {
                    "other1": self.loaded_data["other_sales"][:, 12],
                    "other2": self.loaded_data["other_sales"][:, 15]
                }
            },
            ghv={
                "oil": None,
                "con": None,
                "gsa": {
                    "gsa1": self.loaded_data["gas_sales"][:, 5],
                    "gsa2": self.loaded_data["gas_sales"][:, 8],
                    "gsa3": self.loaded_data["gas_sales"][:, 11],
                    "gsa4": self.loaded_data["gas_sales"][:, 14],
                    "gsa5": self.loaded_data["gas_sales"][:, 17],
                    "gsa6": self.loaded_data["gas_sales"][:, 20],
                    "gsa7": self.loaded_data["gas_sales"][:, 23],
                    "gsa8": self.loaded_data["gas_sales"][:, 26],
                    "gsa9": self.loaded_data["gas_sales"][:, 29],
                    "gsa10": self.loaded_data["gas_sales"][:, 32],
                    "gsa11": self.loaded_data["gas_sales"][:, 35],
                    "gsa12": self.loaded_data["gas_sales"][:, 38],
                    "gsa13": self.loaded_data["gas_sales"][:, 41],
                    "gsa14": self.loaded_data["gas_sales"][:, 44],
                    "gsa15": self.loaded_data["gas_sales"][:, 47],
                    "gsa16": self.loaded_data["gas_sales"][:, 50],
                    "gsa17": self.loaded_data["gas_sales"][:, 53],
                    "gsa18": self.loaded_data["gas_sales"][:, 56],
                    "gsa19": self.loaded_data["gas_sales"][:, 59],
                    "gsa20": self.loaded_data["gas_sales"][:, 62]
                },
                "lpg": {
                    "propane": None,
                    "butane": None
                },
                "sulfur": None,
                "electricity": None,
                "co2": None,
                "others": {
                    "other1": None,
                    "other2": None
                }
            },
            days={
                "oil": self.loaded_data["oil_sales"][:, 2],
                "con": self.loaded_data["oil_sales"][:, 6],
                "gsa": {
                    "gsa1": None,
                    "gsa2": None,
                    "gsa3": None,
                    "gsa4": None,
                    "gsa5": None,
                    "gsa6": None,
                    "gsa7": None,
                    "gsa8": None,
                    "gsa9": None,
                    "gsa10": None,
                    "gsa11": None,
                    "gsa12": None,
                    "gsa13": None,
                    "gsa14": None,
                    "gsa15": None,
                    "gsa16": None,
                    "gsa17": None,
                    "gsa18": None,
                    "gsa19": None,
                    "gsa20": None
                },
                "lpg": {
                    "propane": None,
                    "butane": None
                },
                "sulfur": None,
                "electricity": None,
                "co2": None,
                "others": {
                    "other1": None,
                    "other2": None
                }
            }
        )

        return lifting

    def arrange_data(self, workbook_to_load: str):
        """
        Decompose the loaded data into their corresponding components in each associated category.

        Parameters
        ----------
        workbook_to_load: str
            Name of the target Excel workbook.

        Returns
        -------
        The associated data from each category, given as the attributes of the class.
        """

        # Read data input from Excel
        self.read_from_excel(workbook_to_load=workbook_to_load)

        # Arrange data input for each cost components
        self.general = self._prepare_general_data()
        self.psc_cr_1 = self._prepare_psc_cr_data(data_key="psc_cr_1")
        self.psc_cr_2 = self._prepare_psc_cr_data(data_key="psc_cr_2")
        self.psc_gs_1 = self._prepare_psc_gs_data(data_key="psc_gs_1")
        self.psc_gs_2 = self._prepare_psc_gs_data(data_key="psc_gs_2")
        self.cost = self._prepare_cost_data()
        self.lifting = self._prepare_lifting_data()

    def write_to_excel(self):
        """
        Transfer the results of the calculation to a target Excel file.

        Returns
        -------
        """

        raise NotImplementedError
