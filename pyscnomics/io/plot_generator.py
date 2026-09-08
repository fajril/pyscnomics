from typing import TypedDict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from numpy.typing import NDArray
import pandas as pd


class _UncertaintyOutcomes(TypedDict):
    results: NDArray[np.float64]


def get_uncertainty_plot(
    uncertainty_outcomes: _UncertaintyOutcomes,
    plot_type: str = "Stairway",
) -> None:
    """
    Function to get the uncertainty plot.

    Parameters
    ----------
    uncertainty_outcomes: _UncertaintyOutcomes
        Mapping containing a 2-D float64 NumPy array under ``results``.
    plot_type: str
        The type of the plot that will be generated. the available options is
        ['Stairway', 'Histogram']

    Returns
    -------
    Generating an uncertainty plot.

    """
    # Grouping the result
    result_freq = uncertainty_outcomes["results"][:, 0]
    result_npv = uncertainty_outcomes["results"][:, 1]
    result_irr = uncertainty_outcomes["results"][:, 2]
    result_pi = uncertainty_outcomes["results"][:, 3]
    result_pot = uncertainty_outcomes["results"][:, 4]
    result_gov_take = uncertainty_outcomes["results"][:, 5]
    result_ctr_net_share = uncertainty_outcomes["results"][:, 6]

    if plot_type == "Stairway":
        # Create a figure and a grid of 2x3 subplots
        fig, axs = plt.subplots(2, 3, figsize=(18, 11))
        npv_axes = axs[0, 0]
        irr_axes = axs[0, 1]
        pi_axes = axs[0, 2]
        pot_axes = axs[1, 0]
        gov_take_axes = axs[1, 1]
        ctr_net_share_axes = axs[1, 2]

        # plot each subplot individually
        # Configuring NPV
        _ = npv_axes.set_title("NPV")
        _ = npv_axes.plot(result_npv, result_freq, drawstyle="steps-post")
        _ = npv_axes.set_xlabel("MUSD", fontsize=10, labelpad=0)
        _ = npv_axes.set_ylabel("Frequency", fontsize=10, labelpad=0)

        locator_npv = mticker.MaxNLocator(nbins="auto")
        npv_axes.xaxis.set_major_locator(locator_npv)
        npv_axes.grid(which="major", linewidth=1)
        npv_axes.grid(which="minor", linewidth=0.2)
        npv_axes.minorticks_on()

        # IRR
        # Configuring IRR
        _ = irr_axes.set_title("IRR")
        _ = irr_axes.plot(result_irr, result_freq, drawstyle="steps-post")
        _ = irr_axes.set_xlabel("Fraction", fontsize=10, labelpad=0)
        _ = irr_axes.set_ylabel("Frequency", fontsize=14, labelpad=8)

        locator_irr = mticker.MaxNLocator(nbins="auto")
        irr_axes.xaxis.set_major_locator(locator_irr)
        irr_axes.grid(which="major", linewidth=1)
        irr_axes.grid(which="minor", linewidth=0.2)
        irr_axes.minorticks_on()

        # PI
        # Configuring PI
        _ = pi_axes.set_title("PI")
        _ = pi_axes.plot(result_pi, result_freq, drawstyle="steps-post")
        _ = pi_axes.set_xlabel("Fraction", fontsize=10, labelpad=0)
        _ = pi_axes.set_ylabel("Frequency", fontsize=14, labelpad=8)

        locator_pi = mticker.MaxNLocator(nbins="auto")
        pi_axes.xaxis.set_major_locator(locator_pi)
        pi_axes.grid(which="major", linewidth=1)
        pi_axes.grid(which="minor", linewidth=0.2)
        pi_axes.minorticks_on()

        # POT
        # Configuring POT
        _ = pot_axes.set_title("POT")
        _ = pot_axes.plot(result_pot, result_freq, drawstyle="steps-post")
        _ = pot_axes.set_xlabel("Years", fontsize=10, labelpad=0)
        _ = pot_axes.set_ylabel("Frequency", fontsize=14, labelpad=8)

        locator_pot = mticker.MaxNLocator(nbins="auto")
        pot_axes.xaxis.set_major_locator(locator_pot)
        pot_axes.grid(which="major", linewidth=1)
        pot_axes.grid(which="minor", linewidth=0.2)
        pot_axes.minorticks_on()

        # Government Take
        # Configuring Gov Take
        _ = gov_take_axes.set_title("Gov. Take")
        _ = gov_take_axes.plot(result_gov_take, result_freq, drawstyle="steps-post")
        _ = gov_take_axes.set_xlabel("MUSD", fontsize=10, labelpad=0)
        _ = gov_take_axes.set_ylabel("Frequency", fontsize=14, labelpad=8)

        locator_govtake = mticker.MaxNLocator("auto")
        gov_take_axes.xaxis.set_major_locator(locator_govtake)
        gov_take_axes.grid(which="major", linewidth=1)
        gov_take_axes.grid(which="minor", linewidth=0.2)
        gov_take_axes.minorticks_on()

        # Contractor Net Share
        # Configuring Gov Take
        _ = ctr_net_share_axes.set_title("CTR Net Share")
        _ = ctr_net_share_axes.plot(
            result_ctr_net_share, result_freq, drawstyle="steps-post"
        )
        _ = ctr_net_share_axes.set_xlabel("MUSD", fontsize=10, labelpad=0)
        _ = ctr_net_share_axes.set_ylabel("Frequency", fontsize=14, labelpad=8)

        locator_netshare = mticker.MaxNLocator("auto")
        ctr_net_share_axes.xaxis.set_major_locator(locator_netshare)
        ctr_net_share_axes.grid(which="major", linewidth=1)
        ctr_net_share_axes.grid(which="minor", linewidth=0.2)
        ctr_net_share_axes.minorticks_on()

        # Adjust layout to prevent clipping of titles
        fig.subplots_adjust(wspace=10)
        plt.tight_layout(pad=5)
        fig.savefig("uncertainty_plot.png")
        # plt.show()

    elif plot_type == "Histogram":
        df = pd.DataFrame()
        df["npv"] = result_npv
        df["irr"] = result_irr
        df["pi"] = result_pi
        df["pot"] = result_pot
        df["gov_take"] = result_gov_take
        df["ctr_net_share"] = result_ctr_net_share

        _ = df.hist(figsize=(18, 10))
        plt.savefig("uncertainty_plot.png")
        # plt.show()
