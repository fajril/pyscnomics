from typing import Callable, Dict, TypeAlias

import numpy as np
import numpy.typing as npt

from pyscnomics.econ.indicator import npv
from pyscnomics.econ.selection import LimitMethod

_NumericScalar: TypeAlias = (
    np.bool_
    | np.int8
    | np.int16
    | np.int32
    | np.int64
    | np.uint8
    | np.uint16
    | np.uint32
    | np.uint64
    | np.float16
    | np.float32
    | np.float64
)
_NumericArray: TypeAlias = npt.NDArray[_NumericScalar]
FuncType = Callable[[_NumericArray], int]


def econ_limit(
    cashflow: _NumericArray,
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


def _max_cum_cashflow(cashflow: _NumericArray) -> int:
    return int(np.argmax(np.cumsum(cashflow)))


def _negative_cashflow(cashflow: _NumericArray) -> int:
    # Old Codes
    # if len(cashflow) == 0:
    #     raise ValueError("Cashflow array cannot be empty.")
    # if np.all(cashflow < 0):
    #     return 0  # Return 0 for all negative values
    #
    # # Patch for negative cashflow at the first index
    # if cashflow[0] < 0.0:
    #     return 0  # Return 0 for all negative values
    #
    # sign_changes = np.diff(np.sign(cashflow))
    # if np.any(sign_changes):
    #     # Find the first negative cash flow
    #     first_negative_index = np.where(sign_changes < 0)[0][0]
    #     return first_negative_index  # Return the index before the first negative value

    # New Codes
    # Get indices of positive values
    positive_indices: npt.NDArray[np.intp] = np.where(cashflow > 0)[0]

    # Get indices of negative values
    negative_indices: npt.NDArray[np.intp] = np.where(cashflow < 0)[0]

    # Condition when the cashflow is all positive
    if np.all(cashflow >= 0):
        result = len(cashflow) - 1

    # Condition when the cashflow is all negative. It will return the first index
    elif np.all(cashflow < 0):
        result = 0

    # Condition when the cashflow is mixed
    else:
        # neglect_idx is the cashflow that valued negative i the early years due to the nature of the business
        neglect_idx: npt.NDArray[np.intp] = np.where(
            negative_indices < positive_indices[0]
        )[0]

        # Condition when there are negative values in the early rows
        if len(neglect_idx) > 1:
            cf_pseudo: _NumericArray = cashflow[neglect_idx[-1] + 1 :]
            result = np.where(cf_pseudo < 0)[0][0] + len(neglect_idx) - 1

        # Condition when there is no negative value in the early rows but there are negative cashflow
        elif len(neglect_idx) == 0 and len(negative_indices) > 0:
            result = negative_indices[0] - 1

        else:
            result = 1

    return result


def _max_npv(cashflow: _NumericArray) -> int:
    npv_values: list[float] = [npv(cashflow[: i + 1]) for i in range(len(cashflow))]
    return int(np.argmax(npv_values))
