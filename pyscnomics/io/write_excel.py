import pandas as pd
import numpy as np
import xlwings as xw

from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts.transition import Transition

from pyscnomics.tools.table import get_table


def write_cashflow(workbook_path: str,
                   sheet_name: str,
                   contract: BaseProject | CostRecovery | GrossSplit | Transition,
                   oil_starting_cell: str = 'B5',
                   gas_starting_cell: str = 'B59',
                   consolidated_starting_cell: str = 'B113'):
    """
    The function to write contract cashflow into Excel Workbook.

    Parameters
    ----------
    workbook_path: str
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

    # Defining the workbook and its sheet
    ws = xw.Book(workbook_path).sheets(sheet_name)

    # Writing oil df_oil
    ws.range(oil_starting_cell).value = df_oil.value

    # Writing oil df_gas
    ws.range(gas_starting_cell).value = df_gas.value

    # Writing oil df_consolidated
    ws.range(consolidated_starting_cell).value = df_consolidated.value


def write_summary(summary_dict: dict,
                  workbook_path: str,
                  sheet_name: str = 'Summary',
                  range_cell: str = 'E5'):
    """
    A function to generate the summary and write it on the Microsoft Excel sheet.

    Parameters
    ----------
    summary_dict :  dict
        The dictionary containing the argument for generating the summary.
    workbook_path : str
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
    ws = xw.Book(workbook_path).sheets(sheet_name)

    # Test Write
    ws.range(range_cell).value = df_summary.values


def write_opt(list_str,
              list_params_value,
              result_optimization,
              workbook_path: str,
              sheet_name: str = 'Optimization',
              range_opt_result: str = 'P2',
              range_list_params: str = 'N5',
              range_list_value: str = 'P5',
              ):
    # Making dataframe of the optimization parameter list, optimization parameter values
    df_optim = pd.DataFrame()
    df_optim['list_parameter'] = list_str
    df_optim['list_parameter_value'] = list_params_value

    # Defining the workbook and its sheet
    ws = xw.Book(workbook_path).sheets(sheet_name)

    # Writing result of optimization
    ws.range(range_opt_result).value = result_optimization

    # Writing list of parameter optimization
    ws.range(range_list_params).value = df_optim['list_parameter'].values

    # Writing list of parameter value of optimization
    ws.range(range_list_value).value = df_optim['list_parameter_value'].values
