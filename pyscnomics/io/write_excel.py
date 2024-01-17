import pandas as pd
import numpy as np
import xlwings as xw

from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts.transition import Transition

from pyscnomics.tools.table import get_table


def write_cashflow(workbook_object: xw.Book,
                   sheet_name: str,
                   contract: BaseProject | CostRecovery | GrossSplit | Transition,
                   oil_starting_cell: str = 'B5',
                   gas_starting_cell: str = 'B59',
                   consolidated_starting_cell: str = 'B113'):
    """
    The function to write contract cashflow into Excel Workbook.

    Parameters
    ----------
    workbook_object: xw.Book
        The path of the Excel workbook
    sheet_name: str
        The sheet name of where the cashflow is located within the workbook
    contract: CostRecovery | GrossSplit | Transition
        The object of the contract
    oil_starting_cell: str
        The range cell where the oil table cashflow first row and first column is located
    gas_starting_cell: str
        The range cell where the gas table cashflow first row and first column is located
    consolidated_starting_cell: str
        The range cell where the consolidated table cashflow first row and first column is located

    """
    # Producing the table
    df_oil, df_gas, df_consolidated = get_table(contract=contract)

    # Defining the workbook and its sheet when the contract is Transition
    if sheet_name == 'Transition':
        # Defining contract type for frst and second contract
        if isinstance(contract.contract1, CostRecovery) and isinstance(contract.contract2, CostRecovery):
            sheet_cf_1 = 'Result Table CR'
            sheet_cf_2 = 'Result Table CR (2)'

        elif isinstance(contract.contract1, CostRecovery) and isinstance(contract.contract2, GrossSplit):
            sheet_cf_1 = 'Result Table CR'
            sheet_cf_2 = 'Result Table GS'

        elif isinstance(contract.contract1, GrossSplit) and isinstance(contract.contract2, GrossSplit):
            sheet_cf_1 = 'Result Table GS'
            sheet_cf_2 = 'Result Table GS (2)'

        else:
            sheet_cf_1 = 'Result Table GS'
            sheet_cf_2 = 'Result Table CR'

        # Looping the sheet name to write the cashflow into Excel file
        lst_of_sheet = [sheet_cf_1, sheet_cf_2]
        for index, sheet in enumerate(lst_of_sheet):
            # Defining the workbook and its sheet
            ws = workbook_object.sheets(sheet)

            # Writing oil df_oil
            ws.range(oil_starting_cell).value = df_oil[index].values

            # Writing oil df_gas
            ws.range(gas_starting_cell).value = df_gas[index].values

            # Writing oil df_consolidated
            ws.range(consolidated_starting_cell).value = df_consolidated[index].values

    # When the contract is Cost Recovery or Gross Split
    else:
        # Defining the workbook and its sheet
        ws = workbook_object.sheets(sheet_name)

        # Writing oil df_oil
        ws.range(oil_starting_cell).value = df_oil.values

        # Writing oil df_gas
        ws.range(gas_starting_cell).value = df_gas.values

        # Writing oil df_consolidated
        ws.range(consolidated_starting_cell).value = df_consolidated.values


def write_summary(summary_dict: dict,
                  workbook_object: xw.Book,
                  sheet_name: str = 'Summary',
                  range_cell: str = 'E5'):
    """
    A function to generate the summary and write it on the Microsoft Excel sheet.

    Parameters
    ----------
    summary_dict :  dict
        The dictionary containing the argument for generating the summary.
    workbook_object : xw.Book
        The directory path of the Microsoft Excel file.
    sheet_name : str
        The name of the worksheet
    range_cell : str
        The cell range of the summary in the Excel sheet

    Returns
    -------

    """

    # Defining the result that used in summary table
    result_list = [
        summary_dict['lifting_oil'],
        summary_dict['oil_wap'],
        summary_dict['lifting_gas'],
        summary_dict['gas_wap'],
        summary_dict['gross_revenue'],
        None,
        summary_dict['ctr_gross_share'],
        summary_dict['gov_gross_share'],
        summary_dict['sunk_cost'],
        summary_dict['investment'],
        summary_dict['tangible'],
        summary_dict['intangible'],
        summary_dict['opex_and_asr'],
        summary_dict['opex'],
        summary_dict['asr'],
        summary_dict['cost_recovery / deductible_cost'],
        summary_dict['cost_recovery_over_gross_rev'],
        summary_dict['unrec_cost'],
        summary_dict['unrec_over_gross_rev'],
        None,
        summary_dict['ctr_net_share'],
        summary_dict['ctr_net_share_over_gross_share'],
        summary_dict['ctr_net_cashflow'],
        summary_dict['ctr_net_cashflow_over_gross_rev'],
        summary_dict['ctr_npv'],
        summary_dict['ctr_irr'],
        summary_dict['ctr_pot'],
        summary_dict['ctr_pv_ratio'],
        summary_dict['ctr_pi'],
        None,
        summary_dict['gov_gross_share'],
        summary_dict['gov_ftp_share'],
        summary_dict['gov_ddmo'],
        summary_dict['gov_tax_income'],
        summary_dict['gov_take'],
        summary_dict['gov_take_over_gross_rev'],
        summary_dict['gov_take_npv']
    ]

    # Initiating the dataframe that containing the summary
    df_summary = pd.DataFrame(result_list, columns=['control'], dtype=str)

    # Replacing NAN with --- to fill the gap between Excel rows
    df_summary.fillna('---', inplace=True)

    # Selecting a sheet in Excel
    ws = workbook_object.sheets(sheet_name)

    # Test Write
    ws.range(range_cell).value = df_summary.values


def write_opt(list_str: list,
              list_params_value: list,
              result_optimization: float,
              workbook_object: xw.Book,
              sheet_name: str = 'Optimization',
              range_opt_result: str = 'P2',
              range_list_params: str = 'N5',
              range_list_value: str = 'P5',
              ):
    """
    The function to write optimization result into Excel workbook.

    Parameters
    ----------
    list_str: list
        The list of optimization parameter
    list_params_value: list
        The list of optimized parameter value
    result_optimization: float
        The value result of optimized target
    workbook_object: xw.Book
        The workbook object of the Excel file
    sheet_name: str
        The sheet name where the optimization result will be printed on
    range_opt_result: str
        The range cell where the optimization result will be printed on
    range_list_params: str
        The range cell where the optimized parameters will be printed on
    range_list_value: str
        The range cell where the optimized parameters value will be printed on

    """
    # Making dataframe of the optimization parameter list, optimization parameter values
    # df_optim = pd.DataFrame()
    # df_optim['list_parameter'] = list_str
    # df_optim['list_parameter_value'] = list_params_value

    # Defining the workbook and its sheet
    ws = workbook_object.sheets(sheet_name)

    # Writing result of optimization
    ws.range(range_opt_result).value = result_optimization

    # Deleting the row of the optimization result
    ws.range(range_list_params).options(transpose=True).value = [''] * 10
    ws.range(range_list_value).options(transpose=True).value = [''] * 10

    # Writing list of parameter optimization and parameter value of optimization
    ws.range(range_list_params).options(transpose=True).value = list_str
    ws.range(range_list_value).options(transpose=True).value = list_params_value


def write_table(workbook_object: xw.Book,
                sheet_name: str,
                starting_cell: str,
                table: pd.DataFrame | list[pd.DataFrame]):
    """
    The function to write a table into Excel file
    Parameters
    ----------
    workbook_object: xw.Book
        The workbook object of the Excel file
    sheet_name: str
        The sheet name where the table will be printed on
    starting_cell: str
        The starting cell range of where the table will be printed on
    table: pd.Dataframe
        The dataframe object of the table

    """
    # Defining the workbook and its sheet
    ws = workbook_object.sheets(sheet_name)

    # Writing the dataframe into workbook
    ws.range(starting_cell).value = table.values







