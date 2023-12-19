"""
Python Script as the entry point of Excel Workbook
"""
import xlwings as xw

from pyscnomics.io.parse import initiate_contract

from pyscnomics.io.write_excel import write_cashflow, write_summary, write_opt
from pyscnomics.Optimize.optimization import optimize_psc
from pyscnomics.dataset.object_sample import generate_contract_sample

from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts.transition import Transition

from pyscnomics.tools.summary import get_summary


def main(workbook_path, mode):
    # Initiate the contract object
    psc, psc_arguments, summary_arguments = initiate_contract(workbook_path)

    if mode == 'Standard':
        run_standard(contract=psc,
                     contract_arguments=psc_arguments,
                     workbook_path=workbook_path,
                     summary_argument=summary_arguments,)

    # Giving the workbook execution status
    xw.Book(workbook_path).sheets('Cover').range("F17").value = 'Success'


def run_standard(contract,
                 contract_arguments: dict,
                 workbook_path: str,
                 summary_argument: dict
                 ):
    # Running standard contract
    contract.run(**contract_arguments)

    # Define the summary object
    contract_summary = get_summary(**summary_argument)

    # Check the type of the contract
    oil_starting_cell = 'B5'
    gas_starting_cell = 'B59'
    consolidated_starting_cell = 'B113'

    sheet_name = None
    if isinstance(contract, CostRecovery):
        sheet_name = 'Result Table CR'

    elif isinstance(contract, GrossSplit):
        sheet_name = 'Result Table GS'

    # Writing the result of the contract
    write_cashflow(workbook_path=workbook_path,
                   sheet_name=sheet_name,
                   contract=contract,
                   oil_starting_cell=oil_starting_cell,
                   gas_starting_cell=gas_starting_cell,
                   consolidated_starting_cell=consolidated_starting_cell)

    # Writing the summary of the contract
    write_summary(summary_dict=contract_summary,
                  workbook_path=workbook_path,
                  sheet_name='Summary',
                  range_cell='E5',)


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

    # workbook_path = "Workbook.xlsb"
    # run_mode = 'Standard'
    #
    # main(workbook_path=workbook_path, mode=run_mode)
