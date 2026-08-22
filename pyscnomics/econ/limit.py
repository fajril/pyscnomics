import numpy as np
import numpy.typing as npt

from pyscnomics.econ.indicator import npv
from pyscnomics.econ.selection import LimitMethod

def econ_limit(
    cashflow: npt.NDArray[np.float64],
    method: LimitMethod = LimitMethod.MAX_CUM_CASHFLOW,
) -> int:
    """Determine the limit based on the specified method.

    Parameters
    ----------
    cashflow : np.ndarray
        An array of cash flows over time.
    method : LimitMethod, optional
        The method to use for determining the limit (default is MAX_CUM_CASHFLOW).

    Returns
    -------
    int
        The result based on the selected limit method.

    Raises
    ------
    ValueError
        If an invalid LimitMethod is provided.
    """
    func = {
        LimitMethod.MAX_CUM_CASHFLOW: _max_cum_cashflow,
        LimitMethod.NEGATIVE_CASHFLOW: _negative_cashflow,
        LimitMethod.MAX_NPV: _max_npv,
    }

    # Error handling for invalid method
    if method not in func:
        raise ValueError("Invalid LimitMethod provided.")
    if len(cashflow) == 0:
        raise ValueError("The cashflow is empty.")
    return func[method](cashflow)


def _max_cum_cashflow(cashflow: npt.NDArray[np.float64]) -> int:
    return int(np.argmax(np.cumsum(cashflow)))


def _negative_cashflow(cashflow: npt.NDArray[np.float64]) -> int:
    # Get indices of positive values
    positive_indices= np.where(cashflow > 0)[0]

    # Get indices of negative values
    negative_indices = np.where(cashflow < 0)[0]

    # Condition when the cashflow is all positive
    if np.all(cashflow >= 0):
        result = len(cashflow) - 1

    # Condition when the cashflow is all negative. It will return the first index
    elif np.all(cashflow < 0):
        result = 0

    # Condition when the cashflow is mixed
    else:
        # neglect_idx is the cashflow that valued negative i the early years due to the nature of the business
        first_positive = positive_indices.item(0)
        neglect_idx = np.flatnonzero(
            negative_indices < first_positive
        )

        # Condition when there are negative values in the early rows
        if len(neglect_idx) > 1:
            cf_pseudo = cashflow[neglect_idx.item(-1) + 1 :]
            return np.flatnonzero(cf_pseudo < 0).item(0) + len(neglect_idx) - 1

        # Condition when there is no negative value in the early rows but there are negative cashflow
        elif len(neglect_idx) == 0 and len(negative_indices) > 0:
            return negative_indices.item(0) - 1

        else:
            return 1

    return int(result)


def _max_npv(cashflow: npt.NDArray[np.float64]) -> int:
    npv_values= [
        npv(cashflow[: i + 1]) for i in range(len(cashflow))
    ]
    return int(np.argmax(npv_values))
