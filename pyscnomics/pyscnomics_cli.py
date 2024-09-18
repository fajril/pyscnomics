"""
Python Script as the entry point of Excel Workbook
"""

import click
import numpy as np
import pandas as pd
import xlwings as xw
import uvicorn

from pyscnomics.io.parse import InitiateContract
from pyscnomics.io.aggregator import Aggregate
from pyscnomics.io.plot_generator import get_uncertainty_plot

from pyscnomics.tools.summary import get_summary
from pyscnomics.io.write_excel import write_cashflow, write_summary, write_opt, write_table
from pyscnomics.optimize.optimization import optimize_psc

from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts.transition import Transition

from pyscnomics.optimize.sensitivity import (
    get_multipliers_sensitivity,
    execute_sensitivity,
)

from pyscnomics.optimize.uncertainty import (
    get_montecarlo_data,
    get_multipliers_montecarlo,
    execute_montecarlo,
)


# Click command for generating CLI command
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
        results = execute_sensitivity(
            data=data,
            target=target,
            multipliers=multipliers,
            workbook_path=workbook_path,
        )

        # Arrange the results into the desired output structure
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
        list_cell_sensitivity = ['M4', 'M29', 'M54', 'M79', 'M104', 'M129']
        for index, cell in enumerate(list_cell_sensitivity):
            write_table(workbook_object=workbook_object,
                        sheet_name='Sensitivity',
                        starting_cell=cell,
                        table=list_df[index], )

    # Run montecarlo simulation
    elif mode == "Uncertainty":
        # Prepare the loaded data
        data = Aggregate(workbook_to_read=workbook_path)
        data.fit()

        # Prepare MonteCarlo data
        uncertainty_data, mean_values_not_zero = get_montecarlo_data(data=data)

        # Get multipliers
        multipliers = np.ones(
            [uncertainty_data["run_number"], len(uncertainty_data["parameter"])],
            dtype=np.float_
        )

        for i in mean_values_not_zero:
            multipliers[:, i] = get_multipliers_montecarlo(
                run_number=uncertainty_data["run_number"],
                distribution=uncertainty_data["distribution"][i],
                min_value=uncertainty_data["min_values"][i],
                mean_value=uncertainty_data["mean_values"][i],
                max_value=uncertainty_data["max_values"][i],
                std_dev=uncertainty_data["std_dev"][i],
            )

        # Run MonteCarlo simulation
        target = ["npv", "irr", "pi", "pot", "gov_take", "ctr_net_share"]
        results = execute_montecarlo(
            data=data,
            workbook_path=workbook_path,
            uncertainty_data=uncertainty_data,
            target=target,
            mult=multipliers,
        )

        # Sorted the results
        results_sorted = np.take_along_axis(
            arr=results,
            indices=np.argsort(results, axis=0),
            axis=0,
        )

        # Specify probability
        prob = (
                np.arange(1, uncertainty_data["run_number"] + 1, dtype=np.float_)
                / uncertainty_data["run_number"]
        )

        # Arrange the results
        results_arranged = np.concatenate((prob[:, np.newaxis], results_sorted), axis=1)

        # Calculate P10, P50, P90
        percentiles = np.percentile(
            a=results_arranged,
            q=[10, 50, 90],
            method="lower",
            axis=0,
        )

        # Determine indices of data to be captured to Excel
        indices = (
            np.linspace(0, uncertainty_data["run_number"], 101)[0:-1].astype(int)
        )

        # Final outcomes to be captured to Excel
        outcomes = {
            "results": results_arranged[indices, :],
            "P10": percentiles[0, :],
            "P50": percentiles[1, :],
            "P90": percentiles[2, :],
        }

        # Making the dataframe to contain the result of percentile
        df_uncertainty_percentile = pd.DataFrame()
        df_uncertainty_percentile.index = ['Result', 'NPV', 'IRR', 'PI', 'POT', 'Gov_Take', 'CTR_Net_Share']
        df_uncertainty_percentile['P10'] = outcomes['P10']
        df_uncertainty_percentile['P50'] = outcomes['P50']
        df_uncertainty_percentile['P90'] = outcomes['P90']
        df_uncertainty_transposed = df_uncertainty_percentile.transpose()
        df_uncertainty_transposed.drop(['Result'], axis=1, inplace=True)

        # Grouping the data into each category in a dataframe
        df_uncertainty_result = pd.DataFrame()
        df_uncertainty_result['frequency'] = outcomes['results'][:, 0]
        df_uncertainty_result['npv'] = outcomes['results'][:, 1]
        df_uncertainty_result['irr'] = outcomes['results'][:, 2]
        df_uncertainty_result['pi'] = outcomes['results'][:, 3]
        df_uncertainty_result['pot'] = outcomes['results'][:, 4]
        df_uncertainty_result['gov_take'] = outcomes['results'][:, 5]
        df_uncertainty_result['ctr_net_share'] = outcomes['results'][:, 6]

        # Writing the percentile table into Excel workbook
        write_table(
            workbook_object=workbook_object,
            sheet_name='Uncertainty',
            starting_cell='L6',
            table=df_uncertainty_transposed,
        )

        # Writing the result table into Excel workbook
        write_table(
            workbook_object=workbook_object,
            sheet_name='Uncertainty',
            starting_cell='K49',
            table=df_uncertainty_result,
        )

        # Generating the uncertainty plot
        get_uncertainty_plot(uncertainty_outcomes=outcomes, plot_type='Stairway')

    # Giving the workbook execution status to show that execution is success
    xw.Book(workbook_path).sheets("References").range("N17").value = "Success"


@click.command()
@click.option(
    '-p',
    '--path',
    help='The path of the Microsoft Excel Workbook with PySCnomics template'
)
@click.option(
    '-m',
    '--mode',
    default='Standard',
    help='The mode of the simulation. They are: "Standard", "Sensitivity", "Optimization", "Uncertainty"'
)
@click.option(
    '-api',
    '--api',
    default=0,
    help='The command for running the API backend. '
         '0 for not activating the API backend. 1 for activating the API backend'
)
@click.option(
    '-port',
    '--port',
    default=8000,
    help='The port number for running the API backend. The default port is 8000'
)
def entry_point(**kwargs):
    """ Manages CLI """
    if kwargs['api'] == 1:
        body = """
                We welcome you to our library, PySCnomics. This package contains tailored functionalities for 
                assessing economic feasibility of oil and gas projects following the state-of-the-art Production 
                Sharing Contract (PSC) schemes in Indonesia.
                PySCnomics is the product of join research between Indonesia's Special Task Force for Upstream Oil 
                and Gas Business Activities (SKK Migas) and the Department of Petroleum Engineering, 
                Institut Teknologi Bandung (ITB)
                """
        print(body)
        port_number = kwargs['port']
        uvicorn.run("pyscnomics.api.main:app", port=int(port_number), reload=False)

    if kwargs['path'] is not None:
        # Defining the cli command for path
        if 'p' in kwargs:
            file_path = kwargs['p']
        elif 'path' in kwargs:
            file_path = kwargs['path']
        else:
            file_path = None

        # Defining the cli command for mode
        if 'm' in kwargs:
            mode = kwargs['m']
        elif 'mode' in kwargs:
            mode = kwargs['mode']
        else:
            mode = None

        # Running the code based on the given CLI input
        main(workbook_path=file_path, mode=mode)


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
    write_cashflow(
        workbook_object=workbook_object,
        sheet_name=sheet_name,
        contract=contract,
    )

    # Writing the summary of the contract
    write_summary(
        summary_dict=contract_summary,
        workbook_object=workbook_object,
        sheet_name='Executive Summary',
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
    optimized_contract = optim_result[3][-1]

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

    # Generating the summary of the optimized contract
    summary_arguments['contract'] = optimized_contract
    optimized_contract_summary = get_summary(**summary_arguments)

    # Writing optimized contract summary
    write_summary(
        summary_dict=optimized_contract_summary,
        workbook_object=workbook_object,
        sheet_name='Optimization Comparison',
        range_cell='J6',
    )


if __name__ == "__main__":
    # entry_point()

    from pyscnomics.econ.costs import CapitalCost, Intangible, LBT, OPEX, ASR, ASRCalculator
    from pyscnomics.econ.selection import FluidType

    asrcalc = ASRCalculator(
        start_year_project=2023,
        end_year_project=2040,
        cost_total=np.array([100, 100, 100]),
        begin_year_split=np.array([2025, 2027, 2030]),
        final_year_split=np.array([2028, 2030, 2033]),
        future_rate=np.array([0, 0, 0]),
        vat_portion=np.array([1, 1, 1]),
    )

    t1 = asrcalc.get_distributed_cost()

    print('\t')
    print(f'Filetype: {type(t1)}')
    print('t1 = ', t1)

    # future1 = asr1.get_future_values(
    #     year_ref=np.array([2025, 2027]),
    #     vat_rate=np.array([0.05, 0.0]),
    #     inflation_rate=np.array([0.0, 0.01]),
    # )

    # split1 = asr1.get_distributed_cost()
    #
    # print('\t')
    # print(f'Filetype: {type(split1)}, Length: {len(split1)}')
    # print('split1 = \n', split1)

    print('\t')
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
