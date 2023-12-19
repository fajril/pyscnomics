"""
Python Script as the entry point of Excel Workbook
"""
import xlwings as xw
from pyscnomics.io.write_excel import write_cashflow, write_summary, write_opt
from pyscnomics.Optimize.optimization import optimize_psc
from pyscnomics.dataset.object_sample import generate_contract_sample


def initiate_contract(workbook_path: str):
    """
    The function to generate psc object, psc arguments and summary arguments.

    Parameters
    ----------
    workbook_path: str
        The directory path of the Excel file

    Returns
    -------
    out: tuple
    """
    # Generate PSC Object, using Spreadsheet modul
    psc = 'NotImplemented' + workbook_path

    # Generate PSC Arguments
    # Conditional arguments based on the contract type
    ## if NotImplemented

    psc_arguments = {'NotImplemented_1': NotImplemented,
                     'NotImplemented_2': NotImplemented}

    # Generate Summary Arguments
    summary_arguments = {'NotImplemented_1': NotImplemented,
                         'NotImplemented_2': NotImplemented}

    return psc, psc_arguments, summary_arguments


def main(workbook_path, mode):
    # Initiate the contract object
    psc, psc_argument, summary_arguments = initiate_contract(workbook_path=workbook_path)

    # Defining the workbook object
    ws = xw.Book(workbook_path).sheets('References')

    if mode == 'Standard':
        result = 'Standard Result'
    elif mode == 'Optimization':
        result = 'Optimization Result'
    elif mode == 'Sensitivity':
        result = 'Sensitivity Result'
    elif mode == 'Monte_Carlo':
        result = 'Monte Carlo Result'
    else:
        result = 'Other Result'

    # Test Write
    ws.range("N6").value = result

    # Giving the workbook execution status
    xw.Book(workbook_path).sheets('Cover').range("F17").value = 'Success'


def run_standard(contract,
                 contract_arguments: dict,
                 workbook_path: str,
                 summary_dict: dict
                 ):
    # Running standard contract
    contract.run(**contract_arguments)

    # Check the type of the contract
    ## if NotImplemented

    sheet_name = NotImplemented
    oil_starting_cell = NotImplemented
    gas_starting_cell = NotImplemented
    consolidated_starting_cell = NotImplemented

    # Check condition of contract type
    ##
    sheet_name = NotImplemented

    # Writing the result of the contract
    write_cashflow(workbook_path=workbook_path,
                   sheet_name=sheet_name,
                   contract=contract,
                   oil_starting_cell=oil_starting_cell,
                   gas_starting_cell=gas_starting_cell,
                   consolidated_starting_cell=consolidated_starting_cell, )

    # Writing the summary of the contract
    write_summary(summary_dict=summary_dict,
                  workbook_path=workbook_path,
                  sheet_name='Summary',
                  range_cell='E5', )


def run_optimization(contract,
                     data,
                     contract_arguments: dict,
                     summary_arguments: dict,
                     workbook_path: str,
                     ):
    # Running standard contract
    contract.run(**contract_arguments)

    # Running optimization
    optim_result = optimize_psc(dict_optimization=data['DictOptimization'],
                                contract=contract,
                                contract_arguments=contract_arguments,
                                target_optimization_value=data['Target'],
                                summary_argument=summary_arguments,
                                target_parameter=data['Target_Parameters'])

    list_str = optim_result[0]
    list_params_value = optim_result[1]
    result_optimization = optim_result[2]

    # Writing optimization result
    write_opt(list_str=list_str,
              list_params_value=list_params_value,
              result_optimization=result_optimization,
              workbook_path=workbook_path,
              sheet_name='Optimization')


def run_input_python(case: str,
                     workbook_path: str):
    from pyscnomics.contracts.costrecovery import CostRecovery
    from pyscnomics.contracts.grossplit import GrossSplit
    from pyscnomics.contracts.transition import Transition
    from pyscnomics.tools.summary import get_summary

    # Initiating and running the contract
    psc = generate_contract_sample(case=case)

    # Defining the type of contract
    sheet_name = None
    if isinstance(psc, CostRecovery):
        sheet_name = 'Result Table CR'
    elif isinstance(psc, GrossSplit):
        sheet_name = 'Result Table GS'
    elif isinstance(psc, Transition):
        if isinstance(psc.contract2, CostRecovery):
            sheet_name = 'Result Table CR'
        elif isinstance(psc.contract2, GrossSplit):
            sheet_name = 'Result Table GS'

    # Writing the cashflow table
    write_cashflow(workbook_path=workbook_path,
                   sheet_name=sheet_name,
                   contract=psc, )

    get_summary()
    # Get the summary of the contract


if __name__ == '__main__':
    import sys

    main(workbook_path=sys.argv[1], mode=sys.argv[2])
    #
    # workbook_path = 'Test Entry Point.xlsb'
    # run_mode = 'Standard'
    #
    # main(workbook_path=workbook_path, mode=run_mode)
