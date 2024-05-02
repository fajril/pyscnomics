"""
Create graphs and visualizations based on the cashflow.
"""

import numpy as np
import matplotlib.ticker as tk
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


def _get_bar_plot(
    xdata: np.ndarray,
    ydata: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    legend_label: str,
    bar_color: str,
):
    """
    Generate a bar plot.

    Parameters
    ----------
    xdata: np.ndarray
        Array of x-axis data.
    ydata: np.ndarray
        Array of y-axis data.
    title: str
        Title of the plot.
    xlabel: str
        Label for the x-axis.
    ylabel: str
        Label for the y-axis.
    legend_label: str
        Title for the legend.
    bar_color: str
        Color for the bars.

    Returns
    -------
    None
    """
    # Create instances of Figure and Axes
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"facecolor": "#e1e1e1"})

    # Construct bar plot
    ax.bar(
        x=xdata,
        height=ydata,
        width=0.8,
        align="center",
        color=bar_color,
        edgecolor="black",
        linewidth=0.5,
        label=legend_label,
    )

    # Set title
    ax.set_title(title, pad=15, fontdict={"family": "serif", "size": 14})

    # Set axis labels
    ax.set_xlabel(xlabel, labelpad=10, fontdict={"family": "serif", "size": 13})
    ax.set_ylabel(ylabel, labelpad=10, fontdict={"family": "serif", "size": 13})

    # Specify the limit of the axis
    ax.axis("tight")

    # Set legend
    ax.legend(loc="best", prop={"family": "serif", "size": 11}, frameon=False, fancybox=False)

    # Display grid
    ax.grid(
        color="grey",
        which="major",
        axis="y",
        linestyle=":",
        linewidth=0.5,
    )

    # Set tick position
    ax.xaxis.set_major_locator(tk.AutoLocator())
    ax.xaxis.set_minor_locator(tk.AutoMinorLocator())
    ax.yaxis.set_major_locator(tk.AutoLocator())
    ax.yaxis.set_minor_locator(tk.AutoMinorLocator())

    # Specify plot style
    plt.style.use("seaborn-v0_8-ticks")
    plt.show()


def plot_oil_lifting(
    results: BaseProject | CostRecovery | GrossSplit | Transition
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot oil lifting rate over project years.

    Parameters
    ----------
    results: Union[BaseProject, CostRecovery, GrossSplit, Transition]
        An instance of BaseProject, CostRecovery, GrossSplit, or Transition containing project results.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        A tuple containing matplotlib Figure and Axes objects representing the generated plot.
    """
    return _get_bar_plot(
        xdata=results.project_years,
        ydata=results._oil_lifting.get_lifting_rate_arr(),
        title="Profile of Oil and Condensate Lifting",
        xlabel="Project Years (Year)",
        ylabel="Oil + Condensate Lifting (MMSTB)",
        legend_label="Oil and Condensate Lifting",
        bar_color="tab:green",
    )


def plot_gas_lifting(
    results: BaseProject | CostRecovery | GrossSplit | Transition
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot gas lifting rate over project years.

    Parameters
    ----------
    results: Union[BaseProject, CostRecovery, GrossSplit, Transition]
        An instance of BaseProject, CostRecovery, GrossSplit, or Transition containing project results.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        A tuple containing matplotlib Figure and Axes objects representing the generated plot.
    """
    return _get_bar_plot(
        xdata=results.project_years,
        ydata=results._gas_lifting.get_lifting_rate_arr(),
        title="Gas and LPG Lifting Profile",
        xlabel="Project Years (Year)",
        ylabel="Gas and LPG Lifting (MMSCF)",
        legend_label="Gas Lifting",
        bar_color="tab:red",
    )


def plot_sulfur_lifting(
    results: BaseProject | CostRecovery | GrossSplit | Transition
) -> tuple[plt.Figure, plt.Axes]:
    return _get_bar_plot(
        xdata=results.project_years,
        ydata=results._sulfur_lifting.get_lifting_rate_arr(),
        title="Sulfur Lifting Profile",
        xlabel="Project Years (Year)",
        ylabel="Sulfur Lifting (Ton)",
        legend_label="Sulfur Lifting",
        bar_color="purple",
    )



# Run the calculation to obtain results
results = _get_results(
    workbook_path=r"E:\1009_My Journal_PSC Migas\26_20230707_PSCEconomic\pyscnomics\Workbook_Filled CR.xlsb"
)

print('\t')
print(f'Filetype: {type(results)}')

print('\t')
print(f'Filetype: {type(results._sulfur_lifting)}')
print('oil lifting = \n', results._sulfur_lifting)

print('\t')
print(f'Filetype: {type(results._sulfur_lifting.get_lifting_rate_arr())}')
print(f'Shape: {results._sulfur_lifting.get_lifting_rate_arr().shape}')
print('oil lifting = \n', results._sulfur_lifting.get_lifting_rate_arr())

# Plot data
plot_sulfur_lifting(results=results)
# plot_oil_lifting(results=results)
# plot_gas_lifting(results=results)


