"""
Python Script as the entry point of Excel Workbook
"""
import numpy as np
import xlwings as xw
import threading as th

from pyscnomics.io.parse import InitiateContract
from pyscnomics.optimize.sensitivity import get_multipliers, SensitivityData

from pyscnomics.io.write_excel import write_cashflow, write_summary, write_opt
from pyscnomics.optimize.optimization import optimize_psc

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
        data = Aggregate(workbook_to_read=workbook_path)
        data.fit()

        # print('\t')
        # print(f'Filetype: {type(data.oil_lifting_aggregate_total)}')
        # print('oil_lifting_aggregate_total = \n', data.oil_lifting_aggregate_total)

        # Specify the multipliers
        multipliers, total_run = get_multipliers(
            min_deviation=data.sensitivity_data.percentage_min,
            max_deviation=data.sensitivity_data.percentage_max,
            step=data.sensitivity_data.step,
            number_of_params=len(data.sensitivity_data.parameter),
        )

        t1 = SensitivityData(
            data=data,
            workbook_path=workbook_path,
            multipliers=multipliers[3, 0, :],
        )

        print('\t')
        print('===========================================================================================')

        print('\t')
        print(f'Filetype: {type(data.tangible_cost_aggregate)}')
        print('tangible_cost_aggregate = \n', data.tangible_cost_aggregate)

        # params1 = {}
        # (
        #     params1["psc"],
        #     params1["psc_arguments"],
        #     params1["summary_arguments"]
        # ) = SensitivityData(
        #     data=data,
        #     oil_lifting_aggregate_total=data.oil_lifting_aggregate_total,
        #     workbook_path=workbook_path,
        #     multipliers=multipliers[4, 0, :]
        # ).activate()
        #
        # results1 = run_sensitivity(
        #     contract=params1["psc"],
        #     contract_arguments=params1["psc_arguments"],
        #     summary_arguments=params1["summary_arguments"],
        # )
        #
        # print('\t')
        # print(f'Filetype: {type(results1)}')
        # print('results1 = \n', results1)
        #
        # print('\t')
        # print('===========================================================================================')

        # print('\t')
        # print('data = \n', data.oil_lifting_aggregate)
        #
        # params2 = {}
        # (
        #     params2["psc"],
        #     params2["psc_arguments"],
        #     params2["summary_arguments"]
        # ) = SensitivityData(
        #     data=data,
        #     oil_lifting_aggregate_total=data.oil_lifting_aggregate_total,
        #     workbook_path=workbook_path,
        #     multipliers=multipliers[4, 0, :]
        # ).activate()
        #
        # results2 = run_sensitivity(
        #     contract=params2["psc"],
        #     contract_arguments=params2["psc_arguments"],
        #     summary_arguments=params2["summary_arguments"],
        # )
        #
        # print('\t')
        # print(f'Filetype: {type(results2)}')
        # print('results2 = \n', results2)

        # params_name = ["psc", "psc_arguments", "summary_arguments"]
        # target = ["npv", "irr", "pi", "pot", "gov_take", "ctr_net_share"]
        # params2 = {}
        # results2 = np.zeros([multipliers.shape[1], len(target)], dtype=np.float_)
        #
        # for j in range(multipliers.shape[1]):
        #     (
        #         params2["psc"],
        #         params2["psc_arguments"],
        #         params2["summary_arguments"],
        #     ) = SensitivityData(
        #         data=data,
        #         workbook_path=workbook_path,
        #         multipliers=multipliers[4, j, :],
        #     ).activate()
        #
        #     results2[j, :] = run_sensitivity(
        #         contract=params2["psc"],
        #         contract_arguments=params2["psc_arguments"],
        #         summary_arguments=params2["summary_arguments"],
        #     )
        #
        # print('\t')
        # print(f'Filetype: {type(results2)}')
        # print('results2 = \n', results2)

        # print('\t')
        # print(f'Filetype: {type(params)}')
        # print(f'Keys: {params.keys()}')
        # print('params = \n', params)

        # params = {}
        # results = {
        #     key: np.zeros([multipliers.shape[1], len(target)], dtype=np.float_)
        #     for key in data.sensitivity_data.parameter
        # }

        # for i, val in enumerate(data.sensitivity_data.parameter):
        #     for j in range(multipliers.shape[1]):
        #         # Prepare simulation parameters
        #         (
        #             params["psc"],
        #             params["psc_arguments"],
        #             params["summary_arguments"],
        #         ) = SensitivityData(
        #             multipliers=multipliers[i, j, :],
        #             workbook_path=workbook_path,
        #             data=data,
        #         ).activate()
        #
        #         # Run the simulation
        #         results[val][j, :] = run_sensitivity(
        #             contract=params["psc"],
        #             contract_arguments=params["psc_arguments"],
        #             summary_arguments=params["summary_arguments"],
        #         )
        #
        # print('\t')
        # print(f'Filetype: {type(results)}')
        # print('results = \n', results)

    # Giving the workbook execution status to show that execution is success
    # xw.Book(workbook_path).sheets("Cover").range("F17").value = "Success"


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

    elif isinstance(contract, Transition):
        sheet_name = NotImplemented

    # Writing the result of the contract
    write_cashflow(
        workbook_object=workbook_object,
        sheet_name=sheet_name,
        contract=contract,
    )

    # Writing the summary of the contract
    write_summary(
        summary_dict=contract_summary,
        workbook_object=workbook_object,
        sheet_name='Summary',
        range_cell='E5',
    )


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
    print('Execution Time:', end_time - start_time)
