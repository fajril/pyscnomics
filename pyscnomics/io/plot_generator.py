import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


def get_uncertainty_plot(uncertainty_outcomes: dict,
                         plot_type: str = 'Stairway'
                         ):
    """
    Function to get the uncertainty plot.

    Parameters
    ----------
    uncertainty_outcomes: dict
        The result of an uncertainty analyis in form of a dictionary.
    plot_type: str
        The type of the plot that will be generated. the available options is
        ['Stairway', 'Histogram']

    Returns
    -------
    Generating an uncertainty plot.

    """
    # Grouping the result
    result_freq = uncertainty_outcomes['results'][:, 0]
    result_npv = uncertainty_outcomes['results'][:, 1]
    result_irr = uncertainty_outcomes['results'][:, 2]
    result_pi = uncertainty_outcomes['results'][:, 3]
    result_pot = uncertainty_outcomes['results'][:, 4]
    result_gov_take = uncertainty_outcomes['results'][:, 5]
    result_ctr_net_share = uncertainty_outcomes['results'][:, 6]

    if plot_type == 'Stairway':
        # Create a figure and a grid of 2x3 subplots
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))

        # plot each subplot individually
        # Configuring NPV
        axs[0, 0].set_title('NPV')
        axs[0, 0].plot(result_npv, result_freq, drawstyle="steps-post")
        axs[0, 0].set_xlabel("MUSD", fontsize=10, labelpad=0)
        axs[0, 0].set_ylabel("Frequency", fontsize=10, labelpad=0)

        locator = mticker.MultipleLocator(5)
        axs[0, 0].xaxis.set_major_locator(locator)
        axs[0, 0].grid(which="major", linewidth=1)
        axs[0, 0].grid(which="minor", linewidth=0.2)
        axs[0, 0].minorticks_on()

        # IRR
        # Configuring IRR
        axs[0, 1].set_title('IRR')
        axs[0, 1].plot(result_irr, result_freq, drawstyle="steps-post")
        axs[0, 1].set_xlabel("Fraction", fontsize=10, labelpad=0)
        axs[0, 1].set_ylabel("Frequency", fontsize=14, labelpad=8)

        locator = mticker.MultipleLocator(0.05)
        axs[0, 1].xaxis.set_major_locator(locator)

        axs[0, 1].grid(which="major", linewidth=1)
        axs[0, 1].grid(which="minor", linewidth=0.2)
        axs[0, 1].minorticks_on()

        # PI
        # Configuring PI
        axs[0, 2].set_title('PI')
        axs[0, 2].plot(result_pi, result_freq, drawstyle="steps-post")
        axs[0, 2].set_xlabel("Fraction", fontsize=10, labelpad=0)
        axs[0, 2].set_ylabel("Frequency", fontsize=14, labelpad=8)

        locator = mticker.MultipleLocator(0.5)
        axs[0, 2].xaxis.set_major_locator(locator)

        axs[0, 2].grid(which="major", linewidth=1)
        axs[0, 2].grid(which="minor", linewidth=0.2)
        axs[0, 2].minorticks_on()

        # POT
        # Configuring POT
        axs[1, 0].set_title('POT')
        axs[1, 0].plot(result_pot, result_freq, drawstyle="steps-post")
        axs[1, 0].set_xlabel("Years", fontsize=10, labelpad=0)
        axs[1, 0].set_ylabel("Frequency", fontsize=14, labelpad=8)

        locator = mticker.MultipleLocator(2)
        axs[1, 0].xaxis.set_major_locator(locator)

        axs[1, 0].grid(which="major", linewidth=1)
        axs[1, 0].grid(which="minor", linewidth=0.2)
        axs[1, 0].minorticks_on()

        # Government Take
        # Configuring Gov Take
        axs[1, 1].set_title('Gov. Take')
        axs[1, 1].plot(result_gov_take, result_freq, drawstyle="steps-post")
        axs[1, 1].set_xlabel("MUSD", fontsize=10, labelpad=0)
        axs[1, 1].set_ylabel("Frequency", fontsize=14, labelpad=8)

        locator = mticker.MultipleLocator(100)
        axs[1, 1].xaxis.set_major_locator(locator)

        axs[1, 1].grid(which="major", linewidth=1)
        axs[1, 1].grid(which="minor", linewidth=0.2)
        axs[1, 1].minorticks_on()

        # Contractor Net Share
        # Configuring Gov Take
        axs[1, 2].set_title('CTR Net Share')
        axs[1, 2].plot(result_ctr_net_share, result_freq, drawstyle="steps-post")
        axs[1, 2].set_xlabel("MUSD", fontsize=10, labelpad=0)
        axs[1, 2].set_ylabel("Frequency", fontsize=14, labelpad=8)

        locator = mticker.MultipleLocator(20)
        axs[1, 2].xaxis.set_major_locator(locator)

        axs[1, 2].grid(which="major", linewidth=1)
        axs[1, 2].grid(which="minor", linewidth=0.2)
        axs[1, 2].minorticks_on()

        # Adjust layout to prevent clipping of titles
        fig.subplots_adjust(wspace=10)
        plt.tight_layout()
        plt.savefig('uncertainty_plot.png')
        # plt.show()

    elif plot_type == 'Histogram':
        df = pd.DataFrame()
        df['npv'] = result_npv
        df['irr'] = result_irr
        df['pi'] = result_pi
        df['pot'] = result_pot
        df['gov_take'] = result_gov_take
        df['ctr_net_share'] = result_ctr_net_share

        df.hist(figsize=(18, 10))
        plt.savefig('uncertainty_plot.png')
        # plt.show()



