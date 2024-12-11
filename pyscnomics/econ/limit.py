from typing import Callable, Dict
import numpy as np
from pyscnomics.econ import npv
from pyscnomics.econ.selection import LimitMethod

FuncType = Callable[[np.ndarray], int]


def econ_limit(
    cashflow: np.ndarray, method: LimitMethod = LimitMethod.MAX_CUM_CASHFLOW
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
    func: Dict[LimitMethod, FuncType] = {
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


def _max_cum_cashflow(cashflow: np.ndarray) -> int:
    return int(np.argmax(np.cumsum(cashflow)))


def _negative_cashflow(cashflow: np.ndarray) -> int:
    if len(cashflow) == 0:
        raise ValueError("Cashflow array cannot be empty.")
    if np.all(cashflow < 0):
        return 0  # Return 0 for all negative values

    # Patch for negative cashflow at the first index
    if cashflow[0] < 0.0:
        return 0  # Return 0 for all negative values

    sign_changes = np.diff(np.sign(cashflow))
    if np.any(sign_changes):
        # Find the first negative cash flow
        first_negative_index = np.where(sign_changes < 0)[0][0]
        return first_negative_index  # Return the index before the first negative value

    return len(cashflow) - 1  # Return the last index if all values are positive



def _max_npv(cashflow: np.ndarray) -> int:
    npv_values = [npv(cashflow[: i + 1]) for i in range(len(cashflow))]
    return int(np.argmax(npv_values))
