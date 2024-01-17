"""
Python Script as the entry point of Excel Workbook
"""
import numpy as np
import pandas as pd
import xlwings as xw
import threading as th

from pyscnomics.io.parse import InitiateContract
from pyscnomics.optimize.adjuster import get_multipliers_sensitivity, AdjustData

from pyscnomics.io.write_excel import write_cashflow, write_summary, write_opt, write_table
from pyscnomics.optimize.optimization import optimize_psc

from pyscnomics.contracts.project import BaseProject
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
    # Defining the workbook object
    workbook_object = xw.Book(workbook_path)

    # Run standard economic analysis
    if mode == "Standard":
        (
            psc,
            psc_arguments,
            summary_arguments,
            data
        ) = InitiateContract(workbook_path=workbook_path).activate()

        run_standard(
            contract=psc,
            contract_arguments=psc_arguments,
            workbook_object=workbook_object,
            summary_argument=summary_arguments,
        )

    # Run optimization study
    elif mode == "Optimization":
        (
            psc,
            psc_arguments,
            summary_arguments,
            data
        ) = InitiateContract(workbook_path=workbook_path).activate()

        run_optimization(
            contract=psc,
            data=data,
            contract_arguments=psc_arguments,
            workbook_object=workbook_object,
            summary_arguments=summary_arguments,
        )

    # Run sensitivity study
    elif mode == "Sensitivity":
        # Prepare the loaded data
        data = Aggregate(workbook_to_read=workbook_path)
        data.fit()

        # Specify the multipliers
        multipliers, total_run = get_multipliers_sensitivity(
            min_deviation=data.sensitivity_data.percentage_min,
            max_deviation=data.sensitivity_data.percentage_max,
            number_of_params=len(data.sensitivity_data.parameter),
        )

        # Run sensitivity study
        target = ["npv", "irr", "pi", "pot", "gov_take", "ctr_net_share"]
        results = execute_sensitivity_serial(data=data,
                                             target=target,
                                             multipliers=multipliers,
                                             workbook_path=workbook_path)

        # Arrange the results into the desired output
        results_arranged = {
            key: (
                np.zeros(
                    [multipliers.shape[1], len(data.sensitivity_data.parameter)],
                    dtype=np.float_
                )
            )
            for key in target
        }

        for i, econ in enumerate(target):
            for j, param in enumerate(data.sensitivity_data.parameter):
                results_arranged[econ][:, j] = results[param][:, i]

        # Grouping the sensitivity result
        list_df = [pd.DataFrame] * len(target)
        for index, key in enumerate(target):
            df = pd.DataFrame()
            df['oil_price'] = results_arranged[key][:, 0]
            df['gas_price'] = results_arranged[key][:, 1]
            df['opex'] = results_arranged[key][:, 2]
            df['capex'] = results_arranged[key][:, 3]
            df['prod'] = results_arranged[key][:, 4]
            list_df[index] = df

        # Writing the sensitivity result into the workbook
        list_cell_sensitivity = ['M4', 'M29', 'M54', 'M79', 'M104']
        for index, cell in enumerate(list_cell_sensitivity):
            write_table(workbook_object=workbook_object,
                        sheet_name='Sensitivity',
                        starting_cell=cell,
                        table=list_df[index],)

    # Giving the workbook execution status to show that execution is success
    xw.Book(workbook_path).sheets('Cover').range("F17").value = 'Success'


def run_standard(
    contract: CostRecovery | GrossSplit | Transition,
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

    elif isinstance(contract, BaseProject):
        sheet_name = 'Result Table Base Project'

    elif isinstance(contract, Transition):
        sheet_name = 'Transition'

    # Writing the result of the contract
    write_cashflow(workbook_object=workbook_object,
                   sheet_name=sheet_name,
                   contract=contract, )

    # Writing the summary of the contract
    write_summary(summary_dict=contract_summary,
                  workbook_object=workbook_object,
                  sheet_name='Summary',
                  range_cell='E5', )


def run_optimization(
    contract: CostRecovery | GrossSplit | Transition,
    data: Aggregate,
    contract_arguments: dict,
    summary_arguments: dict,
    workbook_object: xw.Book,
):
    """
    The function to run simulation in Optimization mode.

    Parameters
    ----------
    """
    # Running standard contract
    contract.run(**contract_arguments)

    # Condition to adjust the dict_optimization based on the contract type
    if isinstance(contract, CostRecovery):
        dict_optimization = data.optimization_data.data_cr
        range_list_params = "N5"
        range_list_value = "P5"

    elif isinstance(contract, GrossSplit):
        dict_optimization = data.optimization_data.data_gs
        range_list_params = "N17"
        range_list_value = "P17"

    else:
        if isinstance(contract.contract2, CostRecovery):
            dict_optimization = data.optimization_data.data_cr
            range_list_params = "N5"
            range_list_value = "P5"
        else:
            dict_optimization = data.optimization_data.data_gs
            range_list_params = "N17"
            range_list_value = "P17"

    # Defining the target_optimization_value and target_parameter
    target_optimization_value = data.optimization_data.target["parameter"]
    target_parameter = data.optimization_data.target["value"]

    # Running optimization
    optim_result = optimize_psc(
        dict_optimization=dict_optimization,
        contract=contract,
        contract_arguments=contract_arguments,
        target_optimization_value=target_parameter,
        summary_argument=summary_arguments,
        target_parameter=target_optimization_value,
    )

    list_str = optim_result[0]
    list_params_value = optim_result[1]
    result_optimization = optim_result[2]

    # Writing optimization result
    write_opt(
        list_str=list_str,
        list_params_value=list_params_value,
        result_optimization=result_optimization,
        workbook_object=workbook_object,
        sheet_name='Optimization',
        range_opt_result='P2',
        range_list_params=range_list_params,
        range_list_value=range_list_value,
    )


def run_sensitivity(
    contract: CostRecovery | GrossSplit | Transition,
    contract_arguments: dict,
    summary_arguments: dict,
):
    """
    Function to run simulation in Sensitivity mode.

    Parameters
    ----------
    contract: CostRecovery | GrossSplit | Transition
        The contract object that will be run
    contract_arguments: dict
        The contract arguments
    summary_arguments: dict
        The dictionary of the summary arguments
    """
    contract.run(**contract_arguments)
    contract_summary = get_summary(**summary_arguments)

    return (
        contract_summary["ctr_npv"],
        contract_summary["ctr_irr"],
        contract_summary["ctr_pi"],
        contract_summary["ctr_pot"],
        contract_summary["gov_take"],
        contract_summary["ctr_net_share"],
    )


def execute_sensitivity_serial(
    data: Aggregate,
    target: list,
    multipliers: np.ndarray,
    workbook_path: str
) -> dict[str, np.ndarray]:
    """
    Perform sensitivity analysis in a serial manner.

    Parameters
    ----------
    data: Aggregate
        The aggregate data for the analysis.
    target: list
    multipliers: np.ndarray
        A 3D array of multipliers for sensitivity analysis.
    workbook_path: str
        The workbook path

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary containing sensitivity analysis results. The keys correspond to
        different parameters, and the values are arrays of results for each multiplier.
    """
    params = {}
    results = {
        key: np.zeros([multipliers.shape[1], len(target)], dtype=np.float_)
        for key in data.sensitivity_data.parameter
    }

    for i, key in enumerate(data.sensitivity_data.parameter):
        for j in range(multipliers.shape[1]):
            # Adjust data prior to simulation
            (
                params["psc"],
                params["psc_arguments"],
                params["summary_arguments"]
            ) = AdjustData(
                data=data,
                workbook_path=workbook_path,
                multipliers=multipliers[i, j, :],
            ).activate()

            # Collect the results
            results[key][j, :] = run_sensitivity(
                contract=params["psc"],
                contract_arguments=params["psc_arguments"],
                summary_arguments=params["summary_arguments"],
            )

    return results


if __name__ == '__main__':
    # import sys
    # main(workbook_path=sys.argv[1], mode=sys.argv[2])

    import time
    workbook_path = "Workbook.xlsb"
    run_mode = "Sensitivity"
    # workbook_path = "Workbook_Filled CR.xlsb"
    # run_mode = 'Standard'

    start_time = time.time()
    main(workbook_path=workbook_path, mode=run_mode)
    end_time = time.time()

    print('\t')
    print(f'Execution time: {end_time - start_time} seconds')
