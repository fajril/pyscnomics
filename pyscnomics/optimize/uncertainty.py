"""
Collection of functions to administer Monte Carlo simulation.
"""

import os as os
import numpy as np
import pandas as pd
import multiprocessing as mp
from scipy.stats import uniform, triang, truncnorm

from pyscnomics.tools.summary import get_summary
from pyscnomics.io.aggregator import Aggregate
from pyscnomics.optimize.adjuster import AdjustData

from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts.transition import Transition


class MonteCarloException(Exception):
    """ Exception to be raised for a misuse of MonteCarlo class """

    pass


def get_montecarlo_data(data: Aggregate) -> dict:
    """
    Prepares Monte Carlo simulation data from the given Aggregate object.

    Parameters
    ----------
    data: Aggregate
        The input data containing Monte Carlo simulation data.

    Returns
    -------
    dict
        A dictionary containing the prepared Monte Carlo simulation data.

    Raises
    ------
    MonteCarloException
        If any of the required data is missing or inconsistent.
    """
    # Prepare a container to store all the necessary data
    uncertainty_data = {}

    # Prepare attribute run_number
    if pd.isna(data.montecarlo_data.run_number):
        raise MonteCarloException(
            f"The data for run number is missing. Please check the input data."
        )

    uncertainty_data["run_number"] = int(data.montecarlo_data.run_number)

    # Prepare attribute parameter
    if not isinstance(data.montecarlo_data.parameter, list):
        raise MonteCarloException(
            f"Attribute parameter must be provided in a list datatype, "
            f"not {data.montecarlo_data.parameter.__class__.__qualname__}"
        )

    uncertainty_data["parameter"] = data.montecarlo_data.parameter

    # Prepare attribute distribution
    if not isinstance(data.montecarlo_data.distribution, list):
        raise MonteCarloException(
            f"Attribute distribution must be provided in a list datatype, "
            f"not {data.montecarlo_data.distribution.__class__.__qualname__}"
        )

    uncertainty_data["distribution"] = data.montecarlo_data.distribution

    # Prepare attribute min_values
    if not isinstance(data.montecarlo_data.min_values, np.ndarray):
        raise MonteCarloException(
            f"Attribute min_values must be provided in a numpy ndarray datatype, "
            f"not {data.montecarlo_data.min_values.__class__.__qualname__}"
        )

    min_values_nan = list(filter(lambda i: pd.isna(i), data.montecarlo_data.min_values))
    if len(min_values_nan) > 0:
        raise MonteCarloException(
            f"Minimum values data are incomplete. Please check the input data."
        )

    uncertainty_data["min_values"] = data.montecarlo_data.min_values.astype(np.float_)

    # Prepare attribute mean_values
    if not isinstance(data.montecarlo_data.mean_values, np.ndarray):
        raise MonteCarloException(
            f"Attribute mean_values must be provided in a numpy ndarray datatype, "
            f"not {data.montecarlo_data.mean_values.__class__.__qualname__}"
        )

    mean_values_nan = list(filter(lambda i: pd.isna(i), data.montecarlo_data.mean_values))
    if len(mean_values_nan) > 0:
        raise MonteCarloException(
            f"Mean values data are incomplete. Please check the input data."
        )

    uncertainty_data["mean_values"] = data.montecarlo_data.mean_values.astype(np.float_)

    # Prepare attribute max_values
    if not isinstance(data.montecarlo_data.max_values, np.ndarray):
        raise MonteCarloException(
            f"Attribute max_values must be provided in a numpy ndarray datatype, "
            f"not {data.montecarlo_data.max_values.__class__.__qualname__}"
        )

    max_values_nan = list(filter(lambda i: pd.isna(i), data.montecarlo_data.max_values))
    if len(max_values_nan) > 0:
        raise MonteCarloException(
            f"Maximum values data are incomplete. Please check the input data."
        )

    uncertainty_data["max_values"] = data.montecarlo_data.max_values.astype(np.float_)

    # Prepare attribute standard deviation
    if not isinstance(data.montecarlo_data.std_dev, np.ndarray):
        raise MonteCarloException(
            f"Attribute std_dev must be provided in a numpy ndarray datatype, "
            f"not {data.montecarlo_data.std_dev.__class__.__qualname__}"
        )

    for i, val in enumerate(uncertainty_data["distribution"]):
        if val == "Uniform" or val == "Triangular":
            data.montecarlo_data.std_dev[i] = 0.0

    std_dev_nan = list(filter(lambda i: pd.isna(i), data.montecarlo_data.std_dev))
    if len(std_dev_nan) > 0:
        raise MonteCarloException(
            f"Std deviation data are incomplete. Please check the input data."
        )

    uncertainty_data["std_dev"] = data.montecarlo_data.std_dev.astype(np.float_)

    # Throw exception for inappropriate conditions
    for i, val in enumerate(uncertainty_data["min_values"]):
        if (
            uncertainty_data["min_values"][i] > uncertainty_data["max_values"][i]
            or uncertainty_data["min_values"][i] > uncertainty_data["mean_values"][i]
            or uncertainty_data["mean_values"][i] > uncertainty_data["max_values"][i]
        ):
            raise MonteCarloException(
                f"Inappropriate minimum, mean, or maximum value data. "
                f"Please check the input data."
            )

    # Identify zero mean values
    mean_values_not_zero = np.argwhere(uncertainty_data["mean_values"] != 0.0).ravel()

    return uncertainty_data, mean_values_not_zero


def get_multipliers_montecarlo(
    run_number: int,
    distribution: str,
    min_value: float,
    mean_value: float,
    max_value: float,
    std_dev: float,
) -> np.ndarray:
    """
    Generate an array of multipliers for Monte Carlo simulation based on the specified distribution.

    Parameters
    ----------
    run_number: int
        Number of runs.
    distribution: str
        Type of distribution ("Uniform", "Triangular", or "Normal").
    min_value: float
        Minimum value for the distribution.
    mean_value: float
        Mean (or central value) for the distribution.
    max_value: float
        Maximum value for the distribution.
    std_dev: float
        Standard deviation for the normal distribution.

    Returns
    -------
    multipliers: np.ndarray
        Array of multipliers generated using Monte Carlo simulation.

    Notes
    -----
    - For "Uniform" distribution, the function uses a uniform random variable.
    - For "Triangular" distribution, the function uses a triangular random variable.
    - For "Normal" distribution, the function uses a truncated normal random variable.
    """
    # Uniform distribution
    if distribution == "Uniform":
        # Modify minimum and maximum values
        params = [min_value, max_value]
        attrs_updated = {
            key: float(params[i] / mean_value)
            for i, key in enumerate(["min_value", "max_value"])
        }

        # Determine multipliers
        multipliers = uniform.rvs(
            loc=attrs_updated["min_value"],
            scale=attrs_updated["max_value"] - attrs_updated["min_value"],
            size=run_number,
        )

    # Triangular distribution
    elif distribution == "Triangular":
        # Modify minimum and maximum values
        params = [min_value, mean_value, max_value]
        attrs_updated = {
            key: float(params[i] / mean_value)
            for i, key in enumerate(["min_value", "mean_value", "max_value"])
        }

        # Determine mode (central point)
        c = (
            (attrs_updated["mean_value"] - attrs_updated["min_value"]) /
            (attrs_updated["max_value"] - attrs_updated["min_value"])
        )

        # Determine multipliers
        multipliers = triang.rvs(
            c=c,
            loc=attrs_updated["min_value"],
            scale=attrs_updated["max_value"] - attrs_updated["min_value"],
            size=run_number,
        )

    # Normal distribution
    elif distribution == "Normal":
        # Modify minimum and maximum values
        params = [min_value, mean_value, max_value, std_dev]
        attrs_updated = {
            key: float(params[i] / mean_value)
            for i, key in enumerate(["min_value", "mean_value", "max_value", "std_dev"])
        }

        # Determine z-values
        zvalues = {
            key: float(
                (attrs_updated[key] - attrs_updated["mean_value"]) / attrs_updated["std_dev"]
            )
            for key in ["min_value", "mean_value", "max_value"]
        }

        # Determine multipliers
        multipliers_init = truncnorm.rvs(
            a=zvalues["min_value"],
            b=zvalues["max_value"],
            loc=zvalues["mean_value"],
            scale=1,
            size=run_number,
        )

        multipliers = (multipliers_init * attrs_updated["std_dev"]) + attrs_updated["mean_value"]

    else:
        raise MonteCarloException(
            f"The type of distribution is unavailable. "
            f"Please select type of distribution between: "
            f"(i) Uniform, (ii) Triangular, or (iii) Normal."
        )

    return multipliers


def run_montecarlo(
    contract: BaseProject | CostRecovery | GrossSplit | Transition,
    contract_arguments: dict,
    summary_arguments: dict,
) -> tuple:
    """
    Function to run Monte Carlo simulation.

    Parameters
    ----------
    contract: BaseProject | CostRecovery | GrossSplit | Transition
        The contract type for the project.
    contract_arguments: dict
        Arguments required for running the specified contract.
    summary_arguments: dict
        Arguments for generating the project summary.

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


def execute_montecarlo(
    data: Aggregate,
    workbook_path: str,
    uncertainty_data: dict,
    target: list,
    mult: np.ndarray,
    threshold_parallel: int = 500,
) -> np.ndarray:
    """
    A target function to run MonteCarlo simulation.

    Parameters
    ----------
    data: Aggregate
        The aggregate data for the simulation.
    workbook_path: str
        The path to the workbook containing data.
    uncertainty_data: dict
        Data associated with uncertainty analysis.
    target: list
        A list of targets for the simulation.
    mult: np.ndarray
        A numpy array containing multipliers.
    threshold_parallel: int
        A threshold value to activate parallel computation.

    Returns
    -------
    results: np.ndarray
        Results of the Monte Carlo simulation as a numpy array.

    Notes
    -----
    The engine executes parallel computing for number of simulation run larger than 1_000.
    Instead, serial computing is adopted.
    """
    # Generate project instances to be run in the simulation
    args = ["psc", "psc_arguments", "summary_arguments"]
    params = {arg: [0] * uncertainty_data["run_number"] for arg in args}

    for n in range(uncertainty_data["run_number"]):
        (
            params["psc"][n],
            params["psc_arguments"][n],
            params["summary_arguments"][n],
        ) = AdjustData(
            data=data,
            workbook_path=workbook_path,
            multipliers=mult[n, :],
        ).activate()

    # Execute MonteCarlo simulation
    results = np.zeros([uncertainty_data["run_number"], len(target)], dtype=np.float_)

    # Run serial for number of simulation less than 1000
    if uncertainty_data["run_number"] <= threshold_parallel:
        for n in range(uncertainty_data["run_number"]):
            results[n, :] = run_montecarlo(
                contract=params["psc"][n],
                contract_arguments=params["psc_arguments"][n],
                summary_arguments=params["summary_arguments"][n],
            )

    # Run parallel for number of simulation larger than 1000
    else:
        combined_iters = []
        for n in range(uncertainty_data["run_number"]):
            combined_iters.append(
                (
                    params["psc"][n],
                    params["psc_arguments"][n],
                    params["summary_arguments"][n],
                )
            )

        pl = mp.Pool(processes=int(os.cpu_count() - 1))
        results_init = pl.starmap(run_montecarlo, combined_iters)
        pl.close()
        pl.join()

        # Arrange the results into the appointed data structure
        for n in range(uncertainty_data["run_number"]):
            results[n, :] = results_init[n]

    return results
