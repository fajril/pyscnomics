"""
Collection of functions to administer sensitivity analysis.
"""

import numpy as np

from pyscnomics.tools.summary import get_summary
from pyscnomics.io.aggregator import Aggregate
from pyscnomics.optimize.adjuster import AdjustData

from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts.transition import Transition


def get_multipliers_sensitivity(
    min_deviation: float,
    max_deviation: float,
    base_value: float = 1.0,
    step: int = 10,
    number_of_params: int = 5,
) -> np.ndarray:
    """
    Generate multipliers for different economic parameters within a specified range
    for sensitivity study.

    Parameters
    ----------
    min_deviation: float
        The minimum deviation from the base value.
    max_deviation: float
        The maximum deviation from the base value.
    base_value: float, optional
        The base value for the multipliers. Default is 1.0.
    step: int, optional
        The number of steps to create multipliers. Default is 10.
    number_of_params: int, optional
        The number of parameters to vary in sensitivity analysis. Default is 5.

    Returns
    -------
    multipliers: np.ndarray
        A 3D NumPy array containing multipliers for different economic factors.
    """
    # Specify the minimum and maximum values
    min_val = base_value - min_deviation
    max_val = base_value + max_deviation

    min_multipliers = np.linspace(min_val, base_value, step + 1)
    max_multipliers = np.linspace(base_value, max_val, step + 1)
    tot_multipliers = np.concatenate((min_multipliers, max_multipliers[1:]))

    # Specify array multipliers
    multipliers = (
        np.ones(
            [number_of_params, len(tot_multipliers), number_of_params],
            dtype=np.float_,
        )
    )

    for i in range(number_of_params):
        multipliers[i, :, i] = tot_multipliers.copy()

    # Specify total number of run
    total_run = len(tot_multipliers) * number_of_params

    return multipliers, total_run


def run_sensitivity(
    contract: BaseProject | CostRecovery | GrossSplit | Transition,
    contract_arguments: dict,
    summary_arguments: dict,
) -> tuple:
    """
    Function to run simulation in Sensitivity mode.

    Parameters
    ----------
    contract: BaseProject | CostRecovery | GrossSplit | Transition
        The contract object that will be run.
    contract_arguments: dict
        The contract arguments.
    summary_arguments: dict
        The dictionary of the summary arguments.

    Returns
    -------
    tuple:
        A tuple containing the following project metrics:
        - NPV (Net Present Value)
        - IRR (Internal Rate of Return)
        - PI (Profitability Index)
        - POT (Pay Out Time)
        - Gov Take (Government Take)
        - Net Share (Contractor's Net Share)
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


def execute_sensitivity(
    data: Aggregate,
    target: list,
    multipliers: np.ndarray,
    workbook_path: str,
) -> dict[str, np.ndarray]:
    """
    Run sensitivity analysis in a serial manner.

    Parameters
    ----------
    data: Aggregate
        The aggregate data for the analysis.
    target: list
        The target objective variable.
    workbook_path: str
        The path location of the associated Excel file.
    multipliers: np.ndarray
        A 3D array of multipliers for sensitivity analysis.

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary containing sensitivity analysis results. The keys correspond to
        different parameters, and the values are arrays of results for each multiplier.
    """
    args = ["psc", "psc_arguments", "summary_arguments"]

    # Create containers for the generated project instances
    params = {
        par: {
            arg: [0] * multipliers.shape[1] for arg in args
        }
        for par in data.sensitivity_data.parameter
    }

    # Fill "params" with the associated contract instances
    for i, par in enumerate(data.sensitivity_data.parameter):
        for j in range(multipliers.shape[1]):
            (
                params[par]["psc"][j],
                params[par]["psc_arguments"][j],
                params[par]["summary_arguments"][j],
            ) = AdjustData(
                data=data,
                workbook_path=workbook_path,
                multipliers=multipliers[i, j, :],
            ).activate()

    # Create a container for calculation results
    results = {
        par: np.zeros([multipliers.shape[1], len(target)], dtype=np.float_)
        for par in data.sensitivity_data.parameter
    }

    # Run the simulations to obtain the results (NPV, IRR, PI, POT, GOV_TAKE, CTR_NET_SHARE)
    for par in data.sensitivity_data.parameter:
        for j in range(multipliers.shape[1]):
            results[par][j, :] = run_sensitivity(
                contract=params[par]["psc"][j],
                contract_arguments=params[par]["psc_arguments"][j],
                summary_arguments=params[par]["summary_arguments"][j],
            )

    return results
