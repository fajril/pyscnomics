"""
Create graphs and visualizations based on the cashflow.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts.transition import Transition
from pyscnomics.pyscnomics_cli import main


class VisualizationException(Exception):
    """ Exception to be raised for an inappropriate use of ecviz modules """

    pass


def _get_results(
    workbook_path: str,
    mode: str = "Standard",
) -> dict:
    """
    Perform a Standard Production Sharing Contract (PSC) calculation and return the results.

    Parameters
    ----------
    workbook_path : str
        The path to the workbook containing input data for the PSC calculation.
    mode : str, optional
        The mode of calculation. Default is 'Standard'.

    Returns
    -------
    dict
        A dictionary containing the results of the PSC calculation.

    Raises
    ------
    VisualizationException
        If mode is not 'Standard', indicating an invalid calculation mode.
    """
    if mode != "Standard":
        raise VisualizationException(
            f"Must execute a Standard PSC calculation."
        )

    return main(workbook_path=workbook_path, mode=mode)


def _get_plot(
    xdata: np.ndarray,
    ydata: np.ndarray,
    xlabel: str,
    ylabel: str,
    legend_title: str,
):
    # Create instances of Figure and Axes
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"facecolor": "white"})

    # Plot the data
    ax.bar(
        x=xdata,
        height=ydata,
        width=0.8,
        align="center",
        color="tab:green",
        label=legend_title,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend(loc="best")

    plt.show()


def plot_oil_revenue(results: dict):
    return _get_plot(
        xdata=results["project_years"],
        ydata=results["_oil_revenue"],
        xlabel="Project Years",
        ylabel="Oil Revenue (MMUSD)",
        legend_title="Oil Revenue"
    )


# Run the calculation to obtain results
results = _get_results(
    workbook_path=r"E:\1009_My Journal_PSC Migas\26_20230707_PSCEconomic\pyscnomics\Workbook_Filled CR.xlsb"
)

print('\t')
print(f'Filetype: {type(results["_oil_revenue"])}, Length: {len(results["_oil_revenue"])}')
print('oil revenue = \n', results["_oil_revenue"])

# Plot data
plot_oil_revenue(results=results)
