"""
Python Script as the entry point of Excel Workbook
"""
import xlwings as xw

from pyscnomics.io.parse import initiate_contract

from pyscnomics.io.write_excel import write_cashflow, write_summary, write_opt
from pyscnomics.Optimize.optimization import optimize_psc

from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts.transition import Transition

from pyscnomics.tools.summary import get_summary
from pyscnomics.io.aggregator import Aggregate


def main(workbook_path, mode):
    """
    The entry point of the engine if the execution is using Excel workbook.

    Parameters
    ----------
    workbook_path: str
        The path of the workbook
    mode: str
        The mode of the simulation.
        The available mode are: 'Standard', 'Optimization', 'Sensitivity', 'Uncertainty'
    """
    # Initiate the contract object
    psc, psc_arguments, summary_arguments, data = initiate_contract(workbook_path)

    # Defining the workbook object
    workbook_object = xw.Book(workbook_path)

    if mode == 'Standard':
        run_standard(contract=psc,
                     contract_arguments=psc_arguments,
                     workbook_object=workbook_object,
                     summary_argument=summary_arguments, )

    elif mode == 'Optimization':
        run_optimization(contract=psc,
                         data=data,
                         contract_arguments=psc_arguments,
                         workbook_object=workbook_object,
                         summary_arguments=summary_arguments,
                         )

    # Giving the workbook execution status to show that execution is success
    xw.Book(workbook_path).sheets('Cover').range("F17").value = 'Success'


def run_standard(contract: CostRecovery | GrossSplit | Transition,
                 contract_arguments: dict,
                 workbook_object: xw.Book,
                 summary_argument: dict
                 ):
    """
    The function to run simulation in Standard mode.

    Parameters
    ----------
    contract: CostRecovery | GrossSplit | Transition
        The contract object that will be run
    contract_arguments: dict
        The contract arguments
    workbook_object: xw.Book
        The workbook object used in simulation
    summary_argument: dict
        The dictionary of the summary arguments
    """
    # Running standard contract
    contract.run(**contract_arguments)

    # Define the summary object
    contract_summary = get_summary(**summary_argument)

    sheet_name = None
    if isinstance(contract, CostRecovery):
        sheet_name = 'Result Table CR'

    elif isinstance(contract, GrossSplit):
        sheet_name = 'Result Table GS'

    elif isinstance(contract, Transition):
        sheet_name = NotImplemented

    # Writing the result of the contract
    write_cashflow(workbook_object=workbook_object,
                   sheet_name=sheet_name,
                   contract=contract, )

    # Writing the summary of the contract
    write_summary(summary_dict=contract_summary,
                  workbook_object=workbook_object,
                  sheet_name='Summary',
                  range_cell='E5', )


def run_optimization(contract: CostRecovery | GrossSplit | Transition,
                     data: Aggregate,
                     contract_arguments: dict,
                     summary_arguments: dict,
                     workbook_object: xw.Book,
                     ):
    # Running standard contract
    contract.run(**contract_arguments)

    # Condition to adjust the dict_optimization based on the contract type
    if isinstance(contract, CostRecovery):
        dict_optimization = data.optimization_data.data_cr
        range_list_params = 'N5'
        range_list_value = 'P5'
    elif isinstance(contract, GrossSplit):
        dict_optimization = data.optimization_data.data_gs
        range_list_params = 'N17'
        range_list_value = 'P17'
    else:
        if isinstance(contract.contract2, CostRecovery):
            dict_optimization = data.optimization_data.data_cr
            range_list_params = 'N5'
            range_list_value = 'P5'
        else:
            dict_optimization = data.optimization_data.data_gs
            range_list_params = 'N17'
            range_list_value = 'P17'

    # Defining the target_optimization_value and target_parameter
    target_optimization_value = data.optimization_data.target['parameter']
    target_parameter = data.optimization_data.target['value']

    # Running optimization
    optim_result = optimize_psc(dict_optimization=dict_optimization,
                                contract=contract,
                                contract_arguments=contract_arguments,
                                target_optimization_value=target_parameter,
                                summary_argument=summary_arguments,
                                target_parameter=target_optimization_value)

    list_str = optim_result[0]
    list_params_value = optim_result[1]
    result_optimization = optim_result[2]

    # print('list_str:', list_str, '\n')
    # print('list_params_value:', list_params_value, '\n')
    # print('result_optimization:', result_optimization, '\n')
    # input()

    # Writing optimization result
    write_opt(list_str=list_str,
              list_params_value=list_params_value,
              result_optimization=result_optimization,
              workbook_object=workbook_object,
              sheet_name='Optimization',
              range_opt_result='P2',
              range_list_params=range_list_params,
              range_list_value=range_list_value,
              )


if __name__ == '__main__':
    import sys

    main(workbook_path=sys.argv[1], mode=sys.argv[2])

    import time

    # workbook_path = "Workbook_Test Optimization_GS.xlsb"
    # run_mode = 'Optimization'
    #
    # start_time = time.time()
    # main(workbook_path=workbook_path, mode=run_mode)
    # end_time = time.time()
    # print('Execution Time:', end_time - start_time)
