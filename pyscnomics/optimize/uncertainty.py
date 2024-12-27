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



################################################ Uncertainty Detached ################################################
"""
This Code is made to detached the uncertainty module in the previous version which depended with the excel 
into fully able to be run in python. 
"""

from pyscnomics.econ.selection import UncertaintyDistribution
from pyscnomics.optimize.sensitivity import _adjust_element_single_contract

def get_multipliers_monte(
    run_number: int,
    distribution: UncertaintyDistribution,
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
    if distribution == UncertaintyDistribution.UNIFORM:
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
    elif distribution == UncertaintyDistribution.TRIANGULAR:
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
    elif distribution == UncertaintyDistribution.NORMAL:
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

def min_mean_max_retriever(
        contract: BaseProject | CostRecovery | GrossSplit | Transition,
):
    # Retrieve Min, Mean, and Max for CAPEX
    min_capex = np.min(np.concatenate(
        (contract.capital_cost_total.cost, contract.intangible_cost_total.cost)))
    mean_capex = np.average(np.concatenate(
        (contract.capital_cost_total.cost, contract.intangible_cost_total.cost)))
    max_capex = np.max(np.concatenate(
        (contract.capital_cost_total.cost, contract.intangible_cost_total.cost)))

    # Retrieve Min, Mean, and Max for OPEX
    min_opex = np.min(np.concatenate(
        (contract.opex_total.cost, contract.asr_cost_total.cost,
         contract.lbt_cost_total.cost, contract.cost_of_sales_total.cost)))
    mean_opex = np.average(np.concatenate(
        (contract.opex_total.cost, contract.asr_cost_total.cost,
         contract.lbt_cost_total.cost, contract.cost_of_sales_total.cost)))
    max_opex = np.max(np.concatenate(
        (contract.opex_total.cost, contract.asr_cost_total.cost,
         contract.lbt_cost_total.cost, contract.cost_of_sales_total.cost)))

    # Retrieve Min, Mean, and Max for Oil Price
    min_oil_price = np.min(contract._oil_lifting.price)
    mean_oil_price = np.average(contract._oil_lifting.price)
    max_oil_price = np.max(contract._oil_lifting.price)

    # Retrieve Min, Mean, Max for Gas Price
    min_gas_price = np.min(contract._gas_lifting.price)
    mean_gas_price = np.average(contract._gas_lifting.price)
    max_gas_price = np.max(contract._gas_lifting.price)

    # Retrieve Min, Mean, Max for Oil Lifting
    min_oil_lifting = np.min(contract._oil_lifting.lifting_rate)
    mean_oil_lifting = np.average(contract._oil_lifting.lifting_rate)
    max_oil_lifting = np.max(contract._oil_lifting.lifting_rate)

    # Retrieve Min, Mean, Max for Gas Lifting
    min_gas_lifting = np.min(contract._gas_lifting.lifting_rate)
    mean_gas_lifting = np.average(contract._gas_lifting.lifting_rate)
    max_gas_lifting = np.max(contract._gas_lifting.lifting_rate)

    return {
        'min_capex': min_capex,
        'mean_capex': mean_capex,
        'max_capex': max_capex,
        'min_opex': min_opex,
        'mean_opex': mean_opex,
        'max_opex': max_opex,
        'min_oil_price': min_oil_price,
        'mean_oil_price': mean_oil_price,
        'max_oil_price': max_oil_price,
        'min_gas_price': min_gas_price,
        'mean_gas_price': mean_gas_price,
        'max_gas_price': max_gas_price,
        'min_oil_lifting': min_oil_lifting,
        'mean_oil_lifting': mean_oil_lifting,
        'max_oil_lifting': max_oil_lifting,
        'min_gas_lifting': min_gas_lifting,
        'mean_gas_lifting': mean_gas_lifting,
        'max_gas_lifting': max_gas_lifting,
    }


def uncertainty_psc(
        contract: BaseProject | CostRecovery | GrossSplit | Transition,
        contract_arguments: dict,
        summary_arguments: dict,
        run_number: int = 100,
        min_capex: float = None,
        mean_capex: float = None,
        max_capex: float = None,
        min_opex: float = None,
        mean_opex: float = None,
        max_opex: float = None,
        min_oil_price: float = None,
        mean_oil_price: float = None,
        max_oil_price: float = None,
        min_gas_price: float = None,
        mean_gas_price: float = None,
        max_gas_price: float = None,
        min_oil_lifting: float = None,
        mean_oil_lifting: float = None,
        max_oil_lifting: float = None,
        min_gas_lifting: float = None,
        mean_gas_lifting: float = None,
        max_gas_lifting: float = None,
        capex_stddev: float = 1.25,
        opex_stddev: float = 1.25,
        oil_price_stddev: float = 1.25,
        gas_price_stddev: float = 1.25,
        oil_lifting_stddev: float = 1.25,
        gas_lifting_stddev: float = 1.25,
        capex_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL,
        opex_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL,
        oil_price_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL,
        gas_price_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL,
        oil_lifting_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL,
        gas_lifting_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL,
        dataframe_output: bool = True,
) -> tuple:
    """
    The function to get the uncertainty analysis (Monte Carlo) of a contract.

    Parameters
    ----------
    contract: BaseProject | CostRecovery | GrossSplit | Transition
        The contract that the uncertainty will be retrieved.
    contract_arguments: dict
        The contract arguments of the contract.
    summary_arguments: dict
        The summary arguments.
    run_number: int = 100
        The total run number for the monte carlo.
    min_capex: float = None
        Minimum value for the capital cost component.
    mean_capex: float = None
        Average value for the capital cost component.
    max_capex: float = None
        Maximum value for the capital cost component.
    min_opex: float = None
        Minimum value for the opex cost component.
    mean_opex: float = None
        Average value for the opex cost component.
    max_opex: float = None
        Maximum value for the opex cost component.
    min_oil_price: float = None
        Minimum value for the price of oil lifting.
    mean_oil_price: float = None
        Average value for the price of oil lifting.
    max_oil_price: float = None
        Maximum value for the price of oil lifting.
    min_gas_price: float = None
        Minimum value for the price of gas lifting.
    mean_gas_price: float = None
        Average value for the price of gas lifting.
    max_gas_price: float = None
        Maximum value for the price of gas lifting.
    min_oil_lifting: float = None
        Minimum value for the oil lifting.
    mean_oil_lifting: float = None
        Average value for the oil lifting.
    max_oil_lifting: float = None
        Maximum value for the oil lifting.
    min_gas_lifting: float = None
        Minimum value for the gas lifting.
    mean_gas_lifting: float = None
        Average value for the gas lifting.
    max_gas_lifting: float = None
        Maximum value for the gas lifting.
    capex_stddev: float = 1.25
        The standard deviation of the capital cost element which will be used in monte carlo
    opex_stddev: float = 1.25
        The standard deviation of the opex cost element which will be used in monte carlo
    oil_price_stddev: float = 1.25
        The standard deviation of the oil price element which will be used in monte carlo
    gas_price_stddev: float = 1.25
        The standard deviation of the gas price element which will be used in monte carlo
    oil_lifting_stddev: float = 1.25
        The standard deviation of the oil lifting element which will be used in monte carlo
    gas_lifting_stddev: float = 1.25
        The standard deviation of the gas lifting element which will be used in monte carlo
    capex_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL
        The distribution option which will be used in generating capex values.
    opex_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL
        The distribution option which will be used in generating opex values.
    oil_price_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL
        The distribution option which will be used in generating oil price values.
    gas_price_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL
        The distribution option which will be used in generating gas price values.
    oil_lifting_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL
        The distribution option which will be used in generating oil lifting values.
    gas_lifting_distribution: UncertaintyDistribution = UncertaintyDistribution.NORMAL
        The distribution option which will be used in generating gas lifting values.
    dataframe_output: bool = True
        The option whether the output in a dataframe form or dictionary.
    Returns
    -------
    out: tuple
        Uncertainty result,
        Group of the uncertainty result [P10, P50 AND P90]
    """
    # Retrieve the Min Max and Average Values
    min_max_mean_original = min_mean_max_retriever(contract=contract)

    min_max_mean_std = {
        'min_capex': min_capex,
        'mean_capex': mean_capex,
        'max_capex': max_capex,
        'min_opex': min_opex,
        'mean_opex': mean_opex,
        'max_opex': max_opex,
        'min_oil_price': min_oil_price,
        'mean_oil_price': mean_oil_price,
        'max_oil_price': max_oil_price,
        'min_gas_price': min_gas_price,
        'mean_gas_price': mean_gas_price,
        'max_gas_price': max_gas_price,
        'min_oil_lifting': min_oil_lifting,
        'mean_oil_lifting': mean_oil_lifting,
        'max_oil_lifting': max_oil_lifting,
        'min_gas_lifting': min_gas_lifting,
        'mean_gas_lifting': mean_gas_lifting,
        'max_gas_lifting': max_gas_lifting,
    }

    # If the element is not being input from argument, fill with original value
    for element in min_max_mean_std.keys():
        if min_max_mean_std[element] is None:
            min_max_mean_std[element] = min_max_mean_original[element]

    # Constructing the dictionary to contain the uncertainty data
    uncertainty_dict = {
        'CAPEX': {
            'min': min_max_mean_std['min_capex'],
            'mean': min_max_mean_std['mean_capex'],
            'max': min_max_mean_std['max_capex'],
            'std_dev': capex_stddev,
            'distribution': capex_distribution,
        },
        'OPEX': {
            'min': min_max_mean_std['min_opex'],
            'mean': min_max_mean_std['mean_opex'],
            'max': min_max_mean_std['max_opex'],
            'std_dev': opex_stddev,
            'distribution': opex_distribution,
        },
        'OILPRICE': {
            'min': min_max_mean_std['min_oil_price'],
            'mean': min_max_mean_std['mean_oil_price'],
            'max': min_max_mean_std['max_oil_price'],
            'std_dev': oil_price_stddev,
            'distribution': oil_price_distribution,
        },
        'GASPRICE': {
            'min': min_max_mean_std['min_gas_price'],
            'mean': min_max_mean_std['mean_gas_price'],
            'max': min_max_mean_std['max_gas_price'],
            'std_dev': gas_price_stddev,
            'distribution': gas_price_distribution,
        },
        'OILLIFTING': {
            'min': min_max_mean_std['min_oil_lifting'],
            'mean': min_max_mean_std['mean_oil_lifting'],
            'max': min_max_mean_std['max_oil_lifting'],
            'std_dev': oil_lifting_stddev,
            'distribution': oil_lifting_distribution,
        },
        'GASLIFTING': {
            'min': min_max_mean_std['min_gas_lifting'],
            'mean': min_max_mean_std['mean_gas_lifting'],
            'max': min_max_mean_std['max_gas_lifting'],
            'std_dev': gas_lifting_stddev,
            'distribution': gas_lifting_distribution,
        },
    }

    # Removing the same value or zeros value of min max and mean element
    uncertainty_dict = {
        key: value for key, value in uncertainty_dict.items() if np.array([value['min'], value['mean'], value['max']]).std() != 0.0
    }

    # Get the multipliers based on the constructed uncertainty dictionary
    for element in uncertainty_dict.keys():
        uncertainty_dict[element]['multipliers'] = get_multipliers_monte(
            run_number=run_number,
            distribution=uncertainty_dict[element]['distribution'],
            min_value=uncertainty_dict[element]['min'],
            mean_value=uncertainty_dict[element]['mean'],
            max_value=uncertainty_dict[element]['max'],
            std_dev=uncertainty_dict[element]['std_dev'],
        )

    # Make a container for the adjusted contract
    adjusted_contract_list = [contract] * run_number

    # Adjust the element of each contract based on each multiplier simultaneously
    for index in range(run_number):
        for element in uncertainty_dict.keys():
            adjusted_contract_list[index] = _adjust_element_single_contract(
                contract=adjusted_contract_list[index],
                contract_arguments=contract_arguments,
                element=element,
                adjustment_value=uncertainty_dict[element]['multipliers'][index],
                run_contract=False,
            )

    ## The alternative code using lambda
    # for index, contract in enumerate(adjusted_contract_list):
    #     adjusted_contract_list[index] = reduce(
    #         lambda psc, element: _adjust_element_single_contract(
    #             contract=psc,
    #             contract_arguments=contract_arguments,
    #             element=element,
    #             adjustment_value=uncertainty_dict[element]['multipliers'][index],
    #             run_contract=False,
    #         ),
    #         uncertainty_dict.keys(),
    #         contract
    #     )

    # Running all contract in the adjusted_contract_list
    for cntrct in adjusted_contract_list:
        cntrct.run(**contract_arguments)

    # Retrieving the summary of each contract in adjusted_contract_list
    summary_list = [get_summary(
                **{**summary_arguments, 'contract': contrct}
            ) for contrct in adjusted_contract_list]

    # Todo: separate the routine of adjust and running the contract to accommodate Multi Processing
    # Constructing a dictionary to contain the sorted result
    target = ["ctr_npv", "ctr_irr", "ctr_pi", "ctr_pot", "gov_take", "ctr_net_share"]
    filtered_result = {
        key :np.array([
            summary[key]
            for summary in summary_list
        ])
        for key in target
    }

    # Sorting the result
    for indicator in filtered_result.keys():
        filtered_result[indicator] = np.take_along_axis(
            arr=filtered_result[indicator],
            indices=np.argsort(filtered_result[indicator], axis=0),
            axis=0,
        )

    percentiles_dict = {
        key: np.percentile(
            a=filtered_result[key],
            q=[10, 50, 90],
            method="lower",
            axis=0,
        )
        for key in filtered_result.keys()
    }

    # Defining the array of probability
    filtered_result['prob'] = (
            np.arange(1, run_number + 1, dtype=np.float_)
            / run_number
    )

    if dataframe_output is True:
        return (pd.DataFrame(filtered_result).set_index('prob'),
                pd.DataFrame(percentiles_dict).set_index(pd.Index(['P10', 'P50', 'P90'])))

    else:
        return (filtered_result,
                percentiles_dict)
