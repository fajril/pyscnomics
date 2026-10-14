"""
Prepare and classify cost data based on its components.
The associated cost components are:
(1) Capital (also known as Tangible),
(2) Intangible,
(3) OPEX,
(4) ASR,
(5) LBT,
(6) CostOfSales.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeGuard, TypeVar, final

from typing_extensions import override

import numpy as np
import numpy.typing as npt

import pyscnomics.econ.depreciation as depr
from pyscnomics.econ.selection import (
    FluidType,
    CostType,
    DeprMethod,
    InflationAppliedTo,
)
from pyscnomics.econ.costs_tools import (
    get_cost_adjustment_by_inflation,
    calc_indirect_tax,
    calc_distributed_cost,
)
from pyscnomics.validation_rules import (
    any_below,
    magnitude_exceeds,
    missing_value_indices,
    out_of_range_indices,
    period_is_reversed,
    unequal_lengths,
    year_above,
    year_below,
)
from pyscnomics.input_validation import InputValidationError, engine_input_issue


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(value != value)
    return False


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _is_ndarray(value: object) -> TypeGuard[npt.NDArray[np.generic]]:
    return isinstance(value, np.ndarray)


def _is_cost_type_or_none(value: object) -> TypeGuard[CostType | None]:
    return value is None or isinstance(value, CostType)


def _normalize_cost_type(
    value: object, exception_type: type[Exception]
) -> list[CostType | None]:
    if not _is_object_list(value):
        raise exception_type(
            "Attribute cost_type must be given as a list, "
            + f"not as a/an {value.__class__.__qualname__}"
        )
    items: list[object] = value
    result: list[CostType | None] = []
    for item in items:
        if _is_missing(item):
            result.append(None)
        elif _is_cost_type_or_none(item):
            result.append(item)
        else:
            raise exception_type("Must insert CostType in list cost_type")
    return result


def _normalize_fluid_allocation(
    value: object, exception_type: type[Exception]
) -> list[FluidType]:
    if not _is_object_list(value):
        raise exception_type("Attribute cost_allocation must be given as a list")
    items: list[object] = value
    result: list[FluidType] = []
    for item in items:
        if _is_missing(item):
            result.append(FluidType.OIL)
        elif isinstance(item, FluidType):
            result.append(item)
        else:
            raise exception_type("Must insert FluidType in list cost_allocation")
    return result


_T = TypeVar("_T")


def _require_present(
    value: _T | None,
    exception_type: type[Exception],
    attribute_name: str,
) -> _T:
    """Return a normalized value or raise the component's domain exception."""
    if value is None:
        raise exception_type(f"Attribute {attribute_name} is not initialized")
    return value


def _validate_inflation_scope(value: object) -> InflationAppliedTo | None:
    if value is not None and not isinstance(value, InflationAppliedTo):
        raise CapitalException(
            "Invalid inflation_rate_applied_to: "
            + f"({value.__class__.__qualname__}). "
            + f"Expected one of {list(InflationAppliedTo)} or None."
        )
    return value


def _require_capital_array(
    value: object, attribute_name: str
) -> npt.NDArray[np.float64]:
    if not isinstance(value, np.ndarray):
        raise CapitalException(
            f"Attribute {attribute_name} must be provided as a numpy.ndarray, "
            + f"not as a/an {value.__class__.__qualname__}"
        )
    result: npt.NDArray[np.float64] = np.asarray(value, dtype=np.float64)
    return result


def _require_capital_list(value: object, attribute_name: str) -> list[object]:
    if not _is_object_list(value):
        raise CapitalException(
            f"Attribute {attribute_name} must be given as a list, "
            + f"not as a/an {value.__class__.__qualname__}"
        )
    result: list[object] = [item for item in value]
    return result


def _require_capital_tax_discount(
    value: object,
) -> float | int | npt.NDArray[np.float64]:
    if not isinstance(value, (float, int, np.ndarray)):
        raise CapitalException(
            "Attribute tax_discount must be provided as a float/int or as a numpy.ndarray, "
            + f"not as a/an {value.__class__.__qualname__}"
        )
    if isinstance(value, np.ndarray):
        result: npt.NDArray[np.float64] = np.asarray(value, dtype=np.float64)
        return result
    return value


def _missing_indices(
    values: npt.NDArray[np.generic],
) -> npt.NDArray[np.intp]:
    missing = np.fromiter(
        (_is_missing(value) for value in values.flat),
        dtype=np.bool_,
        count=values.size,
    )
    return np.flatnonzero(missing)


class GeneralCostException(Exception):
    """Exception to be raised for an incorrect use of class GeneralCost"""

    pass


class CapitalException(Exception):
    """Exception to be raised for an incorrect use of class CapitalCost"""

    pass


class IntangibleException(Exception):
    """Exception to be raised for an incorrect use of class Intangible"""

    pass


class OPEXException(Exception):
    """Exception to be raised for an incorrect use of class OPEX"""

    pass


def _require_opex_array(value: object, attribute_name: str) -> npt.NDArray[np.generic]:
    if not isinstance(value, np.ndarray):
        raise OPEXException(
            f"Attribute {attribute_name} must be provided as a numpy.ndarray, "
            + f"not as a/an {value.__class__.__qualname__}"
        )
    result: npt.NDArray[np.float64] = np.asarray(value, dtype=np.float64)
    return result


def _require_opex_object_list(value: object, attribute_name: str) -> list[object]:
    if not _is_object_list(value):
        raise OPEXException(
            f"Attribute {attribute_name} must be given as a list, "
            + f"not as a/an {value.__class__.__qualname__}"
        )
    result: list[object] = [item for item in value]
    return result


def _require_opex_tax_discount(
    value: object,
) -> float | int | npt.NDArray[np.generic]:
    if not isinstance(value, (float, int, np.ndarray)):
        raise OPEXException(
            "Attribute tax_discount must be provided as a float/int or as a numpy.ndarray, "
            + f"not as a/an {value.__class__.__qualname__}"
        )
    if isinstance(value, np.ndarray):
        result: npt.NDArray[np.float64] = np.asarray(value, dtype=np.float64)
        return result
    return value


class ASRException(Exception):
    """Exception to be raised for an incorrect use of class ASR"""

    pass


def _require_asr_array(value: object, attribute_name: str) -> npt.NDArray[np.generic]:
    if not isinstance(value, np.ndarray):
        raise ASRException(
            f"Attribute {attribute_name} must be given as a numpy.ndarray, "
            + f"not as a/an {value.__class__.__qualname__}"
        )
    result: npt.NDArray[np.float64] = np.asarray(value, dtype=np.float64)
    return result


def _require_asr_object_list(value: object, attribute_name: str) -> list[object]:
    if not _is_object_list(value):
        raise ASRException(
            f"Attribute {attribute_name} must be given as a list, "
            + f"not as a/an {value.__class__.__qualname__}"
        )
    result: list[object] = [item for item in value]
    return result


def _require_asr_tax_discount(value: object) -> float | int | npt.NDArray[np.generic]:
    if not isinstance(value, (float, int, np.ndarray)):
        raise ASRException(
            "Attribute tax_discount must be provided as a float/int or as a numpy.ndarray, "
            + f"not as a/an {value.__class__.__qualname__}"
        )
    if isinstance(value, np.ndarray):
        result: npt.NDArray[np.float64] = np.asarray(value, dtype=np.float64)
        return result
    return value


def _require_asr_future_rate(value: object) -> float | npt.NDArray[np.generic]:
    if not isinstance(value, (float, np.ndarray)):
        raise ASRException(
            "Attribute future_rate must be provided as a numpy ndarray, "
            + f"not as an/a {value.__class__.__qualname__}"
        )
    if isinstance(value, np.ndarray):
        result: npt.NDArray[np.float64] = np.asarray(value, dtype=np.float64)
        return result
    return value


class LBTException(Exception):
    """Exception to be raised for an incorrect use of class LBT"""

    pass


class CostOfSalesException(Exception):
    """Exception to be raised for an incorrect use of class CostOfSales"""

    pass


class SunkCostException(Exception):
    """Exception to be raised for an incorrect procedure associated with sunkcost"""

    pass


class CapitalInputError(  # pyright: ignore[reportUnsafeMultipleInheritance]
    CapitalException, InputValidationError
):
    """Verified user-input capital error retaining ``CapitalException``."""

    _input_code: str = "CAPITAL_INPUT_INVALID"

    def __init__(self, message: str) -> None:
        InputValidationError.__init__(
            self, (engine_input_issue(code=self._input_code, message=message),)
        )


class IntangibleInputError(  # pyright: ignore[reportUnsafeMultipleInheritance]
    IntangibleException, InputValidationError
):
    """Verified user-input intangible error retaining ``IntangibleException``."""

    _input_code: str = "INTANGIBLE_INPUT_INVALID"

    def __init__(self, message: str) -> None:
        InputValidationError.__init__(
            self, (engine_input_issue(code=self._input_code, message=message),)
        )


class OPEXInputError(  # pyright: ignore[reportUnsafeMultipleInheritance]
    OPEXException, InputValidationError
):
    """Verified user-input OPEX error retaining ``OPEXException``."""

    _input_code: str = "OPEX_INPUT_INVALID"

    def __init__(self, message: str) -> None:
        InputValidationError.__init__(
            self, (engine_input_issue(code=self._input_code, message=message),)
        )


class ASRInputError(  # pyright: ignore[reportUnsafeMultipleInheritance]
    ASRException, InputValidationError
):
    """Verified user-input ASR error retaining ``ASRException``."""

    _input_code: str = "ASR_INPUT_INVALID"

    def __init__(self, message: str) -> None:
        InputValidationError.__init__(
            self, (engine_input_issue(code=self._input_code, message=message),)
        )


class LBTInputError(  # pyright: ignore[reportUnsafeMultipleInheritance]
    LBTException, InputValidationError
):
    """Verified user-input LBT error retaining ``LBTException``."""

    _input_code: str = "LBT_INPUT_INVALID"

    def __init__(self, message: str) -> None:
        InputValidationError.__init__(
            self, (engine_input_issue(code=self._input_code, message=message),)
        )


class CostOfSalesInputError(  # pyright: ignore[reportUnsafeMultipleInheritance]
    CostOfSalesException, InputValidationError
):
    """Verified user-input Cost of Sales error retaining ``CostOfSalesException``."""

    _input_code: str = "COST_OF_SALES_INPUT_INVALID"

    def __init__(self, message: str) -> None:
        InputValidationError.__init__(
            self, (engine_input_issue(code=self._input_code, message=message),)
        )


@dataclass
class GeneralCost:
    """
    Manages a GeneralCost asset.

    Parameters
    ----------
    start_year : int
        The start year of the project.
    end_year : int
        The end year of the project.
    expense_year : numpy.ndarray
        Array of years when the cost is incurred.
    cost : numpy.ndarray
        Array of cost values corresponding to `expense_year`.
    cost_allocation : list[FluidType], optional
        Allocation of each cost element to a specific fluid type
        (e.g., `FluidType.OIL`, `FluidType.GAS`). Default is None.
    cost_type : list[CostType], optional
        Type/category of each cost element (e.g. `CostType.SUNKCOST`,
        `CostType.PRE_ONSTREAM_COST`, `CostType.POST_ONSTREAM_COST`).
        Default is None.
    description : list[str], optional
        Descriptive labels or notes for each cost element. Default is None.
    tax_portion : numpy.ndarray, optional
        Portion of each cost element subject to tax. If not provided,
        defaults to an array of zeros with the same shape as `cost`.
    tax_discount : float | int | numpy.ndarray, optional
        Discount factor(s) applied to the taxable portion of costs,
        reducing the overall tax impact. Default is 0.0.

    Attributes
    ----------
    project_duration : int
        Total duration of the project in years, computed in `__post_init__`.
    project_years : numpy.ndarray
        Array of project years spanning from `start_year` to `end_year`,
        computed in `__post_init__`.
    """

    start_year: int
    end_year: int
    expense_year: npt.NDArray[np.int64] | None = None
    cost: npt.NDArray[np.float64] | None = None
    cost_allocation: list[FluidType] | None = None
    cost_type: list[CostType | None] | None = None
    description: list[str] | None = None
    tax_portion: npt.NDArray[np.float64] | None = None
    tax_discount: npt.NDArray[np.float64] | float | int = field(default=0.0)

    # Attributes to be defined in the __post_init__
    project_duration: int | None = field(default=None, init=False)
    project_years: npt.NDArray[np.int64] | None = field(default=None, init=False)

    def expenditures_pre_tax(
        self,
        year_inflation: npt.NDArray[np.int64] | None = None,
        inflation_rate: npt.NDArray[np.float64] | int | float = 0.0,
    ) -> npt.NDArray[np.float64]:
        """
        Calculate pre-tax expenditures adjusted by inflation.

        This function adjusts the project costs based on inflation rates and allocates
        the costs to their respective expense years, returning the pre-tax expenditures
        for the entire project duration. The function performs cost adjustments using
        inflation rates and expense years, ensuring that costs are properly distributed
        across project years.

        Parameters
        ----------
        year_inflation : np.ndarray, optional
            A NumPy array representing the years during which inflation is applied to the costs.
            If not provided, defaults to repeating the `start_year` for all costs.
        inflation_rate : np.ndarray or float, optional
            The inflation rate(s) to apply to the project costs. If provided as a float,
            a uniform inflation rate is applied. If provided as a NumPy array, different
            rates are applied based on the corresponding project years. Default is 0.0.

        Returns
        -------
        np.ndarray
            A NumPy array representing the pre-tax expenditures adjusted by inflation,
            allocated to the expense years, and extended to the full project duration.

        Notes
        -----
        -   The inflation adjustment is performed using the `calc_inflation` function,
            which computes the inflation-adjusted costs.
        -   Costs are allocated by their associated expense years, and any missing years
            are filled with zeros to match the total project duration.
        -   The indirect tax is allocated across expense years using `np.bincount`.
        """

        project_duration = _require_present(
            self.project_duration, GeneralCostException, "project_duration"
        )
        project_years = _require_present(
            self.project_years, GeneralCostException, "project_years"
        )
        expense_year, cost, _ = self._normalized_cost_arrays(GeneralCostException)

        # Cost adjustment by inflation scheme
        cost_adjusted_by_inflation = get_cost_adjustment_by_inflation(
            start_year=self.start_year,
            end_year=self.end_year,
            cost=cost,
            expense_year=expense_year,
            project_years=project_years,
            year_inflation=year_inflation,
            inflation_rate=inflation_rate,
        )

        # Allocate adjusted costs by their associated expense year
        expenses = np.bincount(
            expense_year - self.start_year, weights=cost_adjusted_by_inflation
        ).astype(np.float64, copy=False)
        zeros = np.zeros(project_duration - len(expenses), dtype=np.float64)

        return np.concatenate((expenses, zeros))

    def indirect_taxes(
        self, tax_rate: npt.NDArray[np.float64] | float = 0.0
    ) -> npt.NDArray[np.float64]:
        """
        Calculate and allocate indirect taxes on project costs.

        This function calculates the indirect tax by applying the specified tax portion,
        tax rates, and tax discount to the project costs. The calculated tax is then allocated
        by the associated expense year.

        Parameters
        ----------
        tax_rate : np.ndarray or float, optional
            A NumPy array or float representing the tax rate applied to the costs. If not provided,
            a default rate of 0.0 will be used. When provided as an array, it should match the
            length of the project years.

        Returns
        -------
        np.ndarray
            A NumPy array containing the allocated indirect taxes per year, adjusted by
            the tax portion, tax rates, and tax discount. The array has the length equal to
            the project duration.

        Notes
        -----
        -   The function computes the tax rate based on the expense year and `start_year`,
            and then applies the tax rate, portion, and discount to the project costs.
        -   The indirect tax is allocated across expense years using `np.bincount`.
        """

        project_duration = _require_present(
            self.project_duration, GeneralCostException, "project_duration"
        )
        project_years = _require_present(
            self.project_years, GeneralCostException, "project_years"
        )
        tax_portion = _require_present(
            self.tax_portion, GeneralCostException, "tax_portion"
        )
        expense_year, cost, tax_discount = self._normalized_cost_arrays(
            GeneralCostException
        )
        # Calculate indirect tax
        indirect_tax = calc_indirect_tax(
            start_year=self.start_year,
            cost=cost,
            expense_year=expense_year,
            project_years=project_years,
            tax_portion=tax_portion,
            tax_rate=tax_rate,
            tax_discount=tax_discount,
        )

        # Allocate indirect tax by their associated expense year
        expenses = np.bincount(
            expense_year - self.start_year, weights=indirect_tax
        ).astype(np.float64, copy=False)
        zeros = np.zeros(project_duration - len(expenses), dtype=np.float64)

        return np.concatenate((expenses, zeros))

    def expenditures_post_tax(
        self,
        year_inflation: npt.NDArray[np.int64] | None = None,
        inflation_rate: npt.NDArray[np.float64] | int | float = 0.0,
        tax_rate: npt.NDArray[np.float64] | float = 0.0,
    ) -> npt.NDArray[np.float64]:
        """
        Calculate post-tax expenditures, adjusted for inflation and indirect taxes.

        This function computes the total project expenditures after adjusting for inflation and
        applying indirect taxes. It first calculates pre-tax expenditures adjusted for inflation,
        and then adds the indirect taxes based on the tax portion, rate, and discount.

        Parameters
        ----------
        year_inflation : np.ndarray, optional
            An array of years representing when inflation impacts each cost. If not provided,
            it defaults to the `start_year` of the project for all costs. The array must have
            the same length as `self.cost`.
        inflation_rate : np.ndarray or float, optional
            The inflation rate(s) to apply. If a single float is provided, it is applied
            uniformly across all years. If an array is provided, each inflation rate
            corresponds to a specific project year (default is 0.0).
        tax_rate : np.ndarray or float, optional
            The tax rate to apply to the costs. If a float is provided, it applies uniformly
            across all project years. If a NumPy array is provided, the rate can vary by year
            (default is 0.0).

        Returns
        -------
        np.ndarray
            An array representing the post-tax expenditures for each project year, adjusted
            for inflation and indirect taxes.

        Notes
        -----
        This function combines two steps:
        1.  Calls `expenditures_pre_tax` to adjust the costs for inflation.
        2.  Calls `indirect_taxes` to compute indirect taxes on the costs based on the specified
            tax portion, rate, and discount.

        The formula used to calculate post-tax expenditures is:
            expenditures_post_tax = expenditures_pre_tax + indirect_tax
        """

        return self.expenditures_pre_tax(
            year_inflation=year_inflation, inflation_rate=inflation_rate
        ) + self.indirect_taxes(tax_rate=tax_rate)

    def __len__(self) -> int:
        expense_year, _, _ = self._normalized_cost_arrays(GeneralCostException)
        return len(expense_year)

    def _normalized_general_fields(
        self,
        exception_type: type[Exception],
    ) -> tuple[
        int,
        npt.NDArray[np.int64],
        list[FluidType],
        list[CostType | None],
        list[str],
        npt.NDArray[np.float64],
    ]:
        """Validate fields populated during component normalization."""
        return (
            _require_present(self.project_duration, exception_type, "project_duration"),
            _require_present(self.project_years, exception_type, "project_years"),
            _require_present(self.cost_allocation, exception_type, "cost_allocation"),
            _require_present(self.cost_type, exception_type, "cost_type"),
            _require_present(self.description, exception_type, "description"),
            _require_present(self.tax_portion, exception_type, "tax_portion"),
        )

    def _normalized_cost_arrays(
        self,
        exception_type: type[Exception],
    ) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Validate array fields that are concrete after component normalization."""
        expense_year = _require_present(
            self.expense_year, exception_type, "expense_year"
        )
        cost = _require_present(self.cost, exception_type, "cost")
        if not _is_ndarray(self.tax_discount):
            raise exception_type(
                "Attribute tax_discount is not initialized as an array"
            )
        return (
            expense_year,
            cost,
            self.tax_discount.astype(np.float64, copy=False),
        )


@dataclass
@final
class CapitalCost(GeneralCost):
    """
    Manages a capital asset.

    Parameters
    ----------
    The attributes are inherited from class GeneralCost. Local attributes associated
    with class CapitalCost are:

    pis_year : numpy.ndarray, optional
        An array representing the PIS year of a capital asset.
    salvage_value : numpy.ndarray, optional
        An array representing the salvage value of a capital asset.
    useful_life : numpy.ndarray, optional
        An array representing the useful life of a capital asset.
    depreciation_factor: np.ndarray, optional
        The value of depreciation factor to be used in PSC_DB depreciation method.
        Default value is 0.5 for the entire project duration.
    is_ic_applied: list
        Whether investment credit is applied for each corresponding capital assets.
    """

    pis_year: npt.NDArray[np.int64] | None = None
    salvage_value: npt.NDArray[np.float64] | None = None
    useful_life: npt.NDArray[np.float64] | None = None
    depreciation_factor: npt.NDArray[np.float64] | None = None
    is_ic_applied: list[bool] | None = None

    def _normalized_capital_fields(
        self,
    ) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        list[bool],
    ]:
        """Validate capital fields populated during component normalization."""
        return (
            _require_present(self.pis_year, CapitalException, "pis_year"),
            _require_present(self.salvage_value, CapitalException, "salvage_value"),
            _require_present(self.useful_life, CapitalException, "useful_life"),
            _require_present(
                self.depreciation_factor, CapitalException, "depreciation_factor"
            ),
            _require_present(self.is_ic_applied, CapitalException, "is_ic_applied"),
        )

    def __post_init__(self):
        """
        Handles the following operations/procedures:
        -   Prepare attributes project_duration and project_years,
        -   Prepare attribute expense_year,
        -   Prepare attribute cost,
        -   Prepare attribute cost_allocation,
        -   Prepare attribute cost_type,
        -   Prepare attribute description,
        -   Prepare attribute pis_year,
        -   Prepare attribute salvage_value,
        -   Prepare attribute useful_life,
        -   Prepare attribute depreciation_factor,
        -   Prepare attribute is_ic_applied,
        -   Prepare attribute tax_portion,
        -   Prepare attribute tax_discount,
        -   Check for unequal length of arrays data,
        -   Raise an exception: salvage_value is larger than the associated cost,
        -   Raise an error: expense_year is after the end year of the project,
        -   Raise an error: expense_year is before the start year of the project,
        """

        # Prepare attributes project_duration and project_years
        if not period_is_reversed(self.start_year, self.end_year):
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise CapitalInputError(
                f"start year {self.start_year} is after the end year {self.end_year}"
            )

        # Prepare attribute expense_year
        raw_expense_year: object = self.expense_year
        if not isinstance(raw_expense_year, np.ndarray):
            raise CapitalException(
                "Attribute expense_year must be provided as a numpy.ndarray, "
                + f"not as a/an {self.expense_year.__class__.__qualname__}"
            )

        else:
            expense_year = raw_expense_year
            if missing_value_indices(expense_year):
                raise CapitalInputError(
                    f"Missing values in array expense_year: {self.expense_year}"
                )

        self.expense_year = raw_expense_year.astype(np.int64)
        expense_year = self.expense_year

        # Prepare attribute cost
        raw_cost: object = self.cost
        if not isinstance(raw_cost, np.ndarray):
            raise CapitalException(
                "Attribute cost must be provided as a numpy.ndarray, "
                + f"not as a/an {self.cost.__class__.__qualname__}"
            )

        else:
            cost = raw_cost
            cost_nan_id = _missing_indices(cost)
            if len(cost_nan_id) > 0:
                cost[cost_nan_id] = np.zeros(len(cost_nan_id))

        self.cost = raw_cost.astype(np.float64)
        cost = self.cost

        # Prepare attribute cost_allocation
        raw_cost_allocation: object = self.cost_allocation
        if raw_cost_allocation is None:
            self.cost_allocation = [
                FluidType.OIL for _ in range(len(self.expense_year))
            ]

        else:
            if not _is_object_list(raw_cost_allocation):
                raise CapitalException(
                    "Attribute cost_allocation must be given as a list, "
                    + f"not as a/an {self.cost_allocation.__class__.__qualname__}"
                )

            self.cost_allocation = _normalize_fluid_allocation(
                raw_cost_allocation, CapitalException
            )

        # Prepare attribute cost_type
        raw_cost_type: object = self.cost_type
        if raw_cost_type is None:
            self.cost_type = [None for _ in range(len(self.expense_year))]

        else:
            if not _is_object_list(raw_cost_type):
                raise CapitalException(
                    "Attribute cost_type must be given as a list, not "
                    + f"as a/an {self.cost_type.__class__.__qualname__}"
                )

            self.cost_type = _normalize_cost_type(raw_cost_type, CapitalException)

        # Prepare attribute description
        if self.description is None:
            self.description = [" " for _ in range(len(self.expense_year))]

        else:
            raw_description = _require_capital_list(self.description, "description")
            description: list[str] = []
            for val in raw_description:
                if _is_missing(val):
                    description.append(" ")
                elif isinstance(val, str):
                    description.append(val)
                else:
                    raise CapitalException("Must insert string in list description")
            self.description = description

        # Prepare attribute pis_year
        if self.pis_year is None:
            self.pis_year = self.expense_year.copy()

        else:
            raw_pis_year = _require_capital_array(self.pis_year, "pis_year")

            pis_year_nan_id = _missing_indices(raw_pis_year)
            if len(pis_year_nan_id) > 0:
                raw_pis_year[pis_year_nan_id] = self.expense_year[
                    pis_year_nan_id
                ].copy()

        self.pis_year = self.pis_year.astype(np.int64)

        # Prepare attribute salvage_value
        if self.salvage_value is None:
            self.salvage_value = np.zeros_like(self.expense_year, dtype=np.float64)

        else:
            raw_salvage_value = _require_capital_array(
                self.salvage_value, "salvage_value"
            )

            salvage_value_nan_id = _missing_indices(raw_salvage_value)
            if len(salvage_value_nan_id) > 0:
                raw_salvage_value[salvage_value_nan_id] = np.zeros(
                    len(salvage_value_nan_id)
                )

        self.salvage_value = self.salvage_value.astype(np.float64)

        # Prepare attribute useful_life
        life = min(5.0, self.project_duration)

        if self.useful_life is None:
            self.useful_life = np.repeat(life, len(self.expense_year))

        else:
            raw_useful_life = _require_capital_array(self.useful_life, "useful_life")
            useful_life_nan_id = _missing_indices(raw_useful_life)
            if len(useful_life_nan_id) > 0:
                raw_useful_life[useful_life_nan_id] = np.repeat(
                    life, len(useful_life_nan_id)
                )

        self.useful_life = self.useful_life.astype(np.float64)

        # Prepare attribute depreciation_factor
        if self.depreciation_factor is None:
            self.depreciation_factor = np.repeat(0.5, len(self.expense_year))

        else:
            raw_depreciation_factor = _require_capital_array(
                self.depreciation_factor, "depreciation_factor"
            )

            depreciation_factor_nan_id = _missing_indices(raw_depreciation_factor)
            if len(depreciation_factor_nan_id) > 0:
                raw_depreciation_factor[depreciation_factor_nan_id] = np.repeat(
                    0.5, len(depreciation_factor_nan_id)
                )

        self.depreciation_factor = self.depreciation_factor.astype(np.float64)

        # Prepare attribute is_ic_applied
        if self.is_ic_applied is None:
            self.is_ic_applied = [False for _ in range(len(self.expense_year))]

        else:
            raw_is_ic_applied = _require_capital_list(
                self.is_ic_applied, "is_ic_applied"
            )
            is_ic_applied: list[bool] = []
            for val in raw_is_ic_applied:
                if _is_missing(val):
                    is_ic_applied.append(False)
                elif isinstance(val, bool):
                    is_ic_applied.append(val)
                else:
                    raise CapitalException("Must insert boolean in list is_ic_applied")
            self.is_ic_applied = is_ic_applied

        # Prepare attribute tax_portion
        if self.tax_portion is None:
            self.tax_portion = np.zeros_like(self.expense_year, dtype=np.float64)

        else:
            raw_tax_portion = _require_capital_array(self.tax_portion, "tax_portion")

            tax_portion_array = np.asarray(raw_tax_portion, dtype=np.float64)
            tax_portion_nan_id = _missing_indices(tax_portion_array)
            if len(tax_portion_nan_id) > 0:
                tax_portion_array[tax_portion_nan_id] = np.zeros(
                    len(tax_portion_nan_id)
                )

            if out_of_range_indices(tax_portion_array, 0.0, 1.0):
                raise CapitalInputError(
                    "The value of tax_portion must be: 0 < tax_portion < 1, "
                    + f"tax_portion: {tax_portion_array}"
                )

        self.tax_portion = np.asarray(self.tax_portion, dtype=np.float64)

        # Prepare attribute tax_discount
        raw_tax_discount = _require_capital_tax_discount(self.tax_discount)

        if isinstance(raw_tax_discount, (float, int)):
            if out_of_range_indices((raw_tax_discount,), 0.0, 1.0):
                raise CapitalInputError("Attribute tax_discount must be between 0 and 1")

            self.tax_discount = np.repeat(raw_tax_discount, len(self.cost))

        else:
            tax_discount_array = np.asarray(raw_tax_discount, dtype=np.float64)
            tax_discount_nan_id = _missing_indices(tax_discount_array)
            if len(tax_discount_nan_id) > 0:
                tax_discount_array[tax_discount_nan_id] = np.zeros(
                    len(tax_discount_nan_id)
                )

            if out_of_range_indices(tax_discount_array, 0.0, 1.0):
                raise CapitalInputError(
                    "The value of tax_discount must be: 0 < tax_discount < 1, "
                    + f"tax_discount: {tax_discount_array}"
                )
            self.tax_discount = tax_discount_array

        self.tax_discount = np.asarray(self.tax_discount, dtype=np.float64)

        # Check input data for unequal length
        expense_year = _require_present(
            self.expense_year, CapitalException, "expense_year"
        )
        cost = _require_present(self.cost, CapitalException, "cost")
        cost_allocation = _require_present(
            self.cost_allocation, CapitalException, "cost_allocation"
        )
        cost_type = _require_present(self.cost_type, CapitalException, "cost_type")
        description = _require_present(
            self.description, CapitalException, "description"
        )
        tax_portion = _require_present(
            self.tax_portion, CapitalException, "tax_portion"
        )
        tax_discount = _require_present(
            self.tax_discount, CapitalException, "tax_discount"
        )
        pis_year = _require_present(self.pis_year, CapitalException, "pis_year")
        salvage_value = _require_present(
            self.salvage_value, CapitalException, "salvage_value"
        )
        useful_life = _require_present(
            self.useful_life, CapitalException, "useful_life"
        )
        depreciation_factor = _require_present(
            self.depreciation_factor, CapitalException, "depreciation_factor"
        )
        is_ic_applied = _require_present(
            self.is_ic_applied, CapitalException, "is_ic_applied"
        )
        length_map = {
            "expense_year": len(expense_year),
            "cost": len(cost),
            "cost_allocation": len(cost_allocation),
            "cost_type": len(cost_type),
            "description": len(description),
            "tax_portion": len(tax_portion),
            "tax_discount": len(tax_discount),
            "pis_year": len(pis_year),
            "salvage_value": len(salvage_value),
            "useful_life": len(useful_life),
            "depreciation_factor": len(depreciation_factor),
            "is_ic_applied": len(is_ic_applied),
        }

        if unequal_lengths(length_map) is not None:
            raise CapitalInputError(
                "Unequal length of arrays: "
                + f"cost: {len(cost)}, "
                + f"expense_year: {len(expense_year)}, "
                + f"cost_allocation: {len(cost_allocation)}, "
                + f"cost_type: {len(cost_type)}, "
                + f"description: {len(description)}, "
                + f"tax_portion: {len(tax_portion)}, "
                + f"tax_discount: {len(tax_discount)}, "
                + f"pis_year: {len(pis_year)}, "
                + f"salvage_value: {len(salvage_value)}, "
                + f"useful_life: {len(useful_life)}, "
                + f"depreciation_factor: {len(depreciation_factor)}, "
                + f"is_ic_applied: {len(is_ic_applied)} "
            )

        # Raise an exception: salvage_value magnitude exceeds associated cost magnitude
        if magnitude_exceeds(self.salvage_value, self.cost):
            raise CapitalInputError(
                f"Salvage value magnitude ({np.abs(self.salvage_value)}) exceeds "
                + f"associated cost magnitude ({np.abs(self.cost)})."
            )

        # Raise an error: expense year is after the end year of the project
        above = year_above(self.expense_year, self.end_year)
        if above is not None:
            raise CapitalInputError(
                f"Expense year ({above}) "
                + f"is after the end year of the project ({self.end_year})"
            )

        # Raise an error: expense year is before the start year of the project
        below = year_below(self.expense_year, self.start_year)
        if below is not None:
            raise CapitalInputError(
                f"Expense year ({below}) "
                + f"is before the start year of the project ({self.start_year})"
            )

    def total_depreciation_rate(
        self,
        depr_method: DeprMethod = DeprMethod.PSC_DB,
        decline_factor: float | int = 2,
        year_inflation: npt.NDArray[np.int64] | None = None,
        inflation_rate: npt.NDArray[np.float64] | int | float = 0.0,
        tax_rate: npt.NDArray[np.float64] | float = 0.0,
        inflation_rate_applied_to: InflationAppliedTo | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Compute total annual depreciation charges for all assets, with optional
        inflation and indirect tax adjustments.

        Asset costs are first adjusted based on the specified inflation scope and
        indirect tax, then depreciated using the selected depreciation method.
        Assets whose useful lives extend beyond the project end are tracked as
        undepreciated (overdue) assets.

        Parameters
        ----------
        depr_method : DeprMethod, default=DeprMethod.PSC_DB
            Depreciation method to apply (straight-line, declining balance,
            or PSC declining balance).
        decline_factor : float or int, default=2
            Decline factor for the declining-balance method.
            Ignored unless ``depr_method`` is ``DeprMethod.DB``.
        year_inflation : np.ndarray, optional
            Project years to which inflation adjustments are applied.
            If None, inflation is applied relative to the project start year.
        inflation_rate : float or np.ndarray, default=0.0
            Inflation rate(s) applied according to ``inflation_rate_applied_to``.
        tax_rate : float or np.ndarray, default=0.0
            Indirect tax rate(s) applied to project costs.
        inflation_rate_applied_to : InflationAppliedTo or None, default=None
            Defines whether inflation is applied to CAPEX, OPEX, both, or not applied
            at all (tax only).

        Returns
        -------
        tuple of np.ndarray
            (1) ``total_depreciation_charge`` :
                Total depreciation charge per project year.
            (2) ``undepreciated_asset`` :
                Remaining depreciation beyond the project duration
                for assets not fully depreciated.

        Raises
        ------
        CapitalException
            If an invalid depreciation method or inflation application scope
            is provided.

        Notes
        -----
        - Depreciation charges are aligned to each asset’s ``pis_year``.
        - Inflation and indirect tax are applied before depreciation,
          based on ``inflation_rate_applied_to``.
        - Overdue depreciation (beyond project end) is reported separately
          as undepreciated assets.
        """

        project_duration = _require_present(
            self.project_duration, CapitalException, "project_duration"
        )
        project_years = _require_present(
            self.project_years, CapitalException, "project_years"
        )
        tax_portion = _require_present(
            self.tax_portion, CapitalException, "tax_portion"
        )
        pis_year = _require_present(self.pis_year, CapitalException, "pis_year")
        expense_year, cost, _tax_discount = self._normalized_cost_arrays(
            CapitalException
        )
        salvage_value = _require_present(
            self.salvage_value, CapitalException, "salvage_value"
        )
        useful_life = _require_present(
            self.useful_life, CapitalException, "useful_life"
        )
        depreciation_factor = _require_present(
            self.depreciation_factor, CapitalException, "depreciation_factor"
        )
        expense_year, cost, _tax_discount = self._normalized_cost_arrays(
            CapitalException
        )

        # Prepare parameter "inflation_rate_applied_to"
        raw_inflation_rate_applied_to = _validate_inflation_scope(
            inflation_rate_applied_to
        )

        # Specify "cost_adjusted" according to "inflation_rate_applied_to"
        cost_adjusted_by_inflation_and_tax = get_cost_adjustment_by_inflation(
            start_year=self.start_year,
            end_year=self.end_year,
            cost=cost,
            expense_year=expense_year,
            project_years=project_years,
            year_inflation=year_inflation,
            inflation_rate=inflation_rate,
        ) + calc_indirect_tax(
            start_year=self.start_year,
            cost=cost,
            expense_year=expense_year,
            project_years=project_years,
            tax_portion=tax_portion,
            tax_rate=tax_rate,
            tax_discount=self.tax_discount,
        )

        cost_adjusted_only_by_tax = cost + calc_indirect_tax(
            start_year=self.start_year,
            cost=cost,
            expense_year=expense_year,
            project_years=project_years,
            tax_portion=tax_portion,
            tax_rate=tax_rate,
            tax_discount=self.tax_discount,
        )

        cost_adjusted: npt.NDArray[np.float64] = np.asarray(
            {
                InflationAppliedTo.CAPEX: cost_adjusted_by_inflation_and_tax,
                InflationAppliedTo.CAPEX_AND_OPEX: cost_adjusted_by_inflation_and_tax,
                InflationAppliedTo.OPEX: cost_adjusted_only_by_tax,
                None: cost_adjusted_only_by_tax,
            }[raw_inflation_rate_applied_to],
            dtype=np.float64,
        )
        adjusted_costs: npt.NDArray[np.float64] = np.asarray(
            cost_adjusted, dtype=np.float64
        )
        salvage_values: npt.NDArray[np.float64] = np.asarray(
            salvage_value, dtype=np.float64
        )
        useful_lives: npt.NDArray[np.int64] = np.asarray(useful_life, dtype=np.int64)
        depreciation_factors: npt.NDArray[np.float64] = np.asarray(
            depreciation_factor, dtype=np.float64
        )
        # Calculate depreciation
        # Depreciation method is straight line
        if depr_method == DeprMethod.SL:
            depreciation_rates: list[npt.NDArray[np.float64]] = []
            for i in range(len(adjusted_costs)):
                cost = float(adjusted_costs.item(i))
                salvage_value = float(salvage_values.item(i))
                useful_life = int(useful_lives.item(i))
                depreciation_rates.append(
                    depr.straight_line_depreciation_rate(
                        cost=cost,
                        salvage_value=salvage_value,
                        useful_life=useful_life,
                        depreciation_len=project_duration,
                    )
                )
            depreciation_charge: npt.NDArray[np.float64] = np.asarray(
                depreciation_rates, dtype=np.float64
            )

        # Depreciation method is declining balance
        elif depr_method == DeprMethod.DB:
            depreciation_rates = []
            for i in range(len(adjusted_costs)):
                cost = float(adjusted_costs.item(i))
                salvage_value = float(salvage_values.item(i))
                useful_life = int(useful_lives.item(i))
                depreciation_rates.append(
                    depr.declining_balance_depreciation_rate(
                        cost=cost,
                        salvage_value=salvage_value,
                        useful_life=useful_life,
                        decline_factor=decline_factor,
                        depreciation_len=project_duration,
                    )
                )
            depreciation_charge = np.asarray(depreciation_rates, dtype=np.float64)

        # Depreciation method is PSC declining balance
        elif depr_method == DeprMethod.PSC_DB:
            depreciation_rates = []
            for i in range(len(adjusted_costs)):
                cost = float(adjusted_costs.item(i))
                depreciation_factor = float(depreciation_factors.item(i))
                useful_life = float(useful_lives.item(i))
                depreciation_rates.append(
                    depr.psc_declining_balance_depreciation_rate(
                        cost=cost,
                        depreciation_factor=depreciation_factor,
                        useful_life=useful_life,
                        depreciation_len=project_duration,
                    )
                )
            depreciation_charge = np.asarray(depreciation_rates, dtype=np.float64)

        else:
            raise CapitalException(
                f"Depreciation method ({depr_method}) is not recognized"
            )

        # Specify indices to place the first element of depreciation
        shift_indices: npt.NDArray[np.int64] = np.asarray(
            pis_year - self.start_year, dtype=np.int64
        )

        # Prepare assets with overdue depreciation, namely those that
        # have not been fully depreciated by the end of the project
        overdues: npt.NDArray[np.int64] = np.asarray(
            pis_year + useful_life - self.end_year - 1, dtype=np.int64
        )

        is_overdue = overdues > 0

        if np.any(is_overdue):
            # Some assets have not been fully depreciated by the end of the project.
            # These assets are overdue by {overdue[is_overdue]} years.
            max_overdue = int(np.max(overdues[is_overdue]))

            full_depr_charge: npt.NDArray[np.float64] = np.zeros(
                (
                    depreciation_charge.shape[0],
                    depreciation_charge.shape[1] + max_overdue,
                ),
                dtype=np.float64,
            )
            # useful_life = self.useful_life.astype(np.int64)

            for i in range(depreciation_charge.shape[0]):
                # (
                #     full_depr_charge[i, shift_indices[i]:shift_indices[i] + useful_life[i]]
                # ) = charge[useful_life[i]]

                shift_index = shift_indices.item(i)
                charge: npt.NDArray[np.float64] = np.asarray(
                    depreciation_charge[i], dtype=np.float64
                )
                if shift_index + charge.shape[0] > full_depr_charge.shape[1]:
                    max_idx = full_depr_charge.shape[1] - shift_index
                    charge = charge[:max_idx]

                full_depr_charge[i, shift_index : shift_index + charge.shape[0]] = (
                    charge
                )

            overdue_depr_charge = full_depr_charge[
                :, self.end_year - self.start_year + 1 :
            ]

        else:
            overdue_depr_charge = np.zeros((1, 1), dtype=np.float64)

        # Modify depreciation_charge so that expenditures are aligned with
        # the corresponding pis_year (or expense_year)
        depreciation_charge = np.asarray(
            [
                np.concatenate(
                    (
                        np.zeros(shift_indices.item(i)),
                        depreciation_charge[i][: -shift_indices.item(i)],
                    )
                )
                if shift_indices.item(i) > 0
                else depreciation_charge[i]
                for i in range(depreciation_charge.shape[0])
            ],
            dtype=np.float64,
        )

        # Calculate total depreciation charge and undepreciated asset
        total_depreciation_charge: npt.NDArray[np.float64] = np.asarray(
            depreciation_charge.sum(axis=0), dtype=np.float64
        )
        undepreciated_asset: npt.NDArray[np.float64] = np.asarray(
            overdue_depr_charge.sum(axis=0), dtype=np.float64
        )

        return total_depreciation_charge, undepreciated_asset

    def total_depreciation_book_value(
        self,
        depr_method: DeprMethod = DeprMethod.PSC_DB,
        decline_factor: float | int = 2,
        year_inflation: npt.NDArray[np.int64] | None = None,
        inflation_rate: npt.NDArray[np.float64] | int | float = 0.0,
        tax_rate: npt.NDArray[np.float64] | float = 0.0,
        inflation_rate_applied_to: InflationAppliedTo | None = None,
    ) -> npt.NDArray[np.float64]:
        """
        Compute the cumulative book value of project assets after depreciation.

        The book value is calculated as cumulative post-tax (and optionally
        inflation-adjusted) expenditures minus cumulative depreciation charges.
        Depreciation charges are obtained from ``total_depreciation_rate`` using
        the specified depreciation method and adjustment rules.

        Parameters
        ----------
        depr_method : DeprMethod, default=DeprMethod.PSC_DB
            Depreciation method applied to asset costs.
        decline_factor : float or int, default=2
            Decline factor for the declining-balance method.
            Ignored unless ``depr_method`` is ``DeprMethod.DB``.
        year_inflation : np.ndarray, optional
            Project years to which inflation adjustments are applied.
        inflation_rate : float or np.ndarray, default=0.0
            Inflation rate(s) applied according to ``inflation_rate_applied_to``.
        tax_rate : float or np.ndarray, default=0.0
            Indirect tax rate(s) applied to project costs.
        inflation_rate_applied_to : InflationAppliedTo or None, default=None
            Defines whether inflation is applied to CAPEX, OPEX, both, or not applied
            (tax only).

        Returns
        -------
        np.ndarray
            1D array of cumulative asset book values per project year.

        Notes
        -----
        - Book value is computed as:
              cumulative(post-tax expenditures) − cumulative(depreciation charges).
        - Inflation and indirect tax treatments are consistent with
          ``expenditures_post_tax`` and ``total_depreciation_rate``.
        """

        # Calculate total depreciation charge from method total_depreciation_rate()
        total_depreciation_charge = self.total_depreciation_rate(
            depr_method=depr_method,
            decline_factor=decline_factor,
            year_inflation=year_inflation,
            inflation_rate=inflation_rate,
            tax_rate=tax_rate,
            inflation_rate_applied_to=inflation_rate_applied_to,
        )[0]

        # Calculate total depreciation book value
        return np.cumsum(
            self.expenditures_post_tax(
                year_inflation=year_inflation,
                inflation_rate=inflation_rate,
                tax_rate=tax_rate,
            )
        ) - np.cumsum(total_depreciation_charge)

    def total_amortization_rate(
        self,
        prod: npt.NDArray[np.float64],
        year_inflation: npt.NDArray[np.int64],
        inflation_rate: npt.NDArray[np.float64] | int | float,
        tax_rate: npt.NDArray[np.float64] | float,
        inflation_rate_applied_to: InflationAppliedTo | None,
    ) -> npt.NDArray[np.float64]:
        """
        Compute total unit-of-production amortization rate for all capital elements.

        Adjusts capital costs for inflation and/or indirect tax according to
        ``inflation_rate_applied_to``, then computes unit-of-production
        amortization per cost element and aggregates them into a project-level
        amortization profile.

        Parameters
        ----------
        prod : np.ndarray
            Production profile aligned with ``self.project_years``.
        year_inflation : np.ndarray
            Year identifiers used for inflation adjustment.
        inflation_rate : np.ndarray or float or int
            Inflation rate(s) applied to eligible cost components.
        tax_rate : np.ndarray or float
            Indirect tax rate(s).
        inflation_rate_applied_to : InflationAppliedTo or None
            Specifies which cost components are adjusted by inflation.
            If None, only indirect tax is applied.

        Returns
        -------
        np.ndarray
            Total amortization charge per project year.
        """

        project_years = _require_present(
            self.project_years, CapitalException, "project_years"
        )
        tax_portion = _require_present(
            self.tax_portion, CapitalException, "tax_portion"
        )
        salvage_value = _require_present(
            self.salvage_value, CapitalException, "salvage_value"
        )
        pis_year = _require_present(self.pis_year, CapitalException, "pis_year")
        expense_year, cost, tax_discount = self._normalized_cost_arrays(
            CapitalException
        )

        # Prepare parameter "inflation_rate_applied_to"
        raw_inflation_rate_applied_to = _validate_inflation_scope(
            inflation_rate_applied_to
        )

        # Specify "cost_adjusted" according to "inflation_rate_applied_to"
        cost_adjusted_by_inflation_and_tax = get_cost_adjustment_by_inflation(
            start_year=self.start_year,
            end_year=self.end_year,
            cost=cost,
            expense_year=expense_year,
            project_years=project_years,
            year_inflation=year_inflation,
            inflation_rate=inflation_rate,
        ) + calc_indirect_tax(
            start_year=self.start_year,
            cost=cost,
            expense_year=expense_year,
            project_years=project_years,
            tax_portion=tax_portion,
            tax_rate=tax_rate,
            tax_discount=tax_discount,
        )

        cost_adjusted_only_by_tax = cost + calc_indirect_tax(
            start_year=self.start_year,
            cost=cost,
            expense_year=expense_year,
            project_years=project_years,
            tax_portion=tax_portion,
            tax_rate=tax_rate,
            tax_discount=tax_discount,
        )

        cost_adjusted: npt.NDArray[np.float64] = np.asarray(
            {
                InflationAppliedTo.CAPEX: cost_adjusted_by_inflation_and_tax,
                InflationAppliedTo.CAPEX_AND_OPEX: cost_adjusted_by_inflation_and_tax,
                InflationAppliedTo.OPEX: cost_adjusted_only_by_tax,
                None: cost_adjusted_only_by_tax,
            }[raw_inflation_rate_applied_to],
            dtype=np.float64,
        )
        adjusted_costs: np.ndarray[tuple[int], np.dtype[np.float64]] = np.asarray(
            cost_adjusted, dtype=np.float64
        )
        salvage_values: np.ndarray[tuple[int], np.dtype[np.float64]] = np.asarray(
            salvage_value, dtype=np.float64
        )
        pis_years: np.ndarray[tuple[int], np.dtype[np.int64]] = np.asarray(
            pis_year, dtype=np.int64
        )
        # Calculate amortization charges
        amortization_rates: list[npt.NDArray[np.float64]] = []
        for i in range(len(adjusted_costs)):
            cost = float(adjusted_costs.item(i))
            salvage_value = float(salvage_values.item(i))
            pis_year = int(pis_years.item(i))
            amortization_rates.append(
                depr.unit_of_production_rate(
                    project_years=project_years,
                    prod=prod,
                    cost=cost,
                    salvage_value=salvage_value,
                    pis_year=pis_year,
                )
            )
        amortization_charges: npt.NDArray[np.float64] = np.asarray(
            amortization_rates, dtype=np.float64
        )

        # Returns total amortization charge for all cost elements
        return np.asarray(amortization_charges.sum(axis=0), dtype=np.float64)

    @override
    def __eq__(self, other: object) -> bool:
        # Between two instances of CapitalCost
        (
            _project_duration,
            _project_years,
            cost_allocation,
            cost_type,
            _description,
            tax_portion,
        ) = self._normalized_general_fields(CapitalException)
        expense_year, cost, tax_discount = self._normalized_cost_arrays(
            CapitalException
        )
        if isinstance(other, CapitalCost):
            (
                _other_project_duration,
                _other_project_years,
                other_cost_allocation,
                other_cost_type,
                _other_description,
                other_tax_portion,
            ) = other._normalized_general_fields(CapitalException)
            other_expense_year, other_cost, other_tax_discount = (
                other._normalized_cost_arrays(CapitalException)
            )
            _, _, cost_allocation, cost_type, _, tax_portion = (
                self._normalized_general_fields(CapitalException)
            )
            _, _, other_cost_allocation, other_cost_type, _, other_tax_portion = (
                other._normalized_general_fields(CapitalException)
            )
            pis_year, salvage_value, useful_life, depreciation_factor, is_ic_applied = (
                self._normalized_capital_fields()
            )
            (
                other_pis_year,
                other_salvage_value,
                other_useful_life,
                other_depreciation_factor,
                other_is_ic_applied,
            ) = other._normalized_capital_fields()
            return all(
                (
                    self.start_year == other.start_year,
                    self.end_year == other.end_year,
                    np.allclose(expense_year, other_expense_year),
                    np.allclose(cost, other_cost),
                    np.allclose(pis_year, other_pis_year),
                    np.allclose(salvage_value, other_salvage_value),
                    np.allclose(useful_life, other_useful_life),
                    np.allclose(depreciation_factor, other_depreciation_factor),
                    np.allclose(tax_portion, other_tax_portion),
                    np.allclose(tax_discount, other_tax_discount),
                    cost_allocation == other_cost_allocation,
                    cost_type == other_cost_type,
                    is_ic_applied == other_is_ic_applied,
                )
            )

        # Between an instance of CapitalCost and an integer/float
        elif isinstance(other, (int, float)):
            return float(np.sum(cost)) == other

        else:
            return False

    def __lt__(self, other: object) -> bool:
        # Between an instance of CapitalCost with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales
        (
            _project_duration,
            _project_years,
            _cost_allocation,
            _cost_type,
            _description,
            _tax_portion,
        ) = self._normalized_general_fields(CapitalException)
        _expense_year, cost, _tax_discount = self._normalized_cost_arrays(
            CapitalException
        )
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales)):
            (
                _other_project_duration,
                _other_project_years,
                _other_cost_allocation,
                _other_cost_type,
                _other_description,
                _other_tax_portion,
            ) = other._normalized_general_fields(CapitalException)
            _other_expense_year, other_cost, _other_tax_discount = (
                other._normalized_cost_arrays(CapitalException)
            )
            return bool(np.sum(cost) < np.sum(other_cost))

        # Between an instance of CapitalCost and an integer/float
        elif isinstance(other, (int, float)):
            return bool(np.sum(cost) < other)

        else:
            raise CapitalException(
                "Must compare an instance of CapitalCost with another instance of "
                + "CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales, an integer, or a float."
            )

    def __le__(self, other: object) -> bool:
        # Between an instance of CapitalCost with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales
        (
            _project_duration,
            _project_years,
            _cost_allocation,
            _cost_type,
            _description,
            _tax_portion,
        ) = self._normalized_general_fields(CapitalException)
        _expense_year, cost, _tax_discount = self._normalized_cost_arrays(
            CapitalException
        )
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales)):
            (
                _other_project_duration,
                _other_project_years,
                _other_cost_allocation,
                _other_cost_type,
                _other_description,
                _other_tax_portion,
            ) = other._normalized_general_fields(CapitalException)
            _other_expense_year, other_cost, _other_tax_discount = (
                other._normalized_cost_arrays(CapitalException)
            )
            return bool(np.sum(cost) <= np.sum(other_cost))

        # Between an instance of CapitalCost and an integer/float
        elif isinstance(other, (int, float)):
            return bool(np.sum(cost) <= other)

        else:
            raise CapitalException(
                "Must compare an instance of CapitalCost with another instance of "
                + "CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales, an integer, or a float."
            )

    def __gt__(self, other: object) -> bool:
        # Between an instance of CapitalCost with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales
        (
            _project_duration,
            _project_years,
            _cost_allocation,
            _cost_type,
            _description,
            _tax_portion,
        ) = self._normalized_general_fields(CapitalException)
        _expense_year, cost, _tax_discount = self._normalized_cost_arrays(
            CapitalException
        )
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales)):
            (
                _other_project_duration,
                _other_project_years,
                _other_cost_allocation,
                _other_cost_type,
                _other_description,
                _other_tax_portion,
            ) = other._normalized_general_fields(CapitalException)
            _other_expense_year, other_cost, _other_tax_discount = (
                other._normalized_cost_arrays(CapitalException)
            )
            return bool(np.sum(cost) > np.sum(other_cost))

        # Between an instance of CapitalCost and an integer/float
        elif isinstance(other, (int, float)):
            return bool(np.sum(cost) > other)

        else:
            raise CapitalException(
                "Must compare an instance of CapitalCost with another instance of "
                + "CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales, an integer, or a float."
            )

    def __ge__(self, other: object) -> bool:
        # Between an instance of CapitalCost with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales
        (
            _project_duration,
            _project_years,
            _cost_allocation,
            _cost_type,
            _description,
            _tax_portion,
        ) = self._normalized_general_fields(CapitalException)
        _expense_year, cost, _tax_discount = self._normalized_cost_arrays(
            CapitalException
        )
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales)):
            (
                _other_project_duration,
                _other_project_years,
                _other_cost_allocation,
                _other_cost_type,
                _other_description,
                _other_tax_portion,
            ) = other._normalized_general_fields(CapitalException)
            _other_expense_year, other_cost, _other_tax_discount = (
                other._normalized_cost_arrays(CapitalException)
            )
            return bool(np.sum(cost) >= np.sum(other_cost))

        # Between an instance of CapitalCost and an integer/float
        elif isinstance(other, (int, float)):
            return bool(np.sum(cost) >= other)

        else:
            raise CapitalException(
                "Must compare an instance of CapitalCost with another instance of "
                + "CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales, an integer, or a float."
            )

    def __add__(self, other: object) -> CapitalCost:
        # Only allows addition between an instance of CapitalCost
        # and another instance of CapitalCost
        (
            _project_duration,
            _project_years,
            cost_allocation,
            cost_type,
            description,
            tax_portion,
        ) = self._normalized_general_fields(CapitalException)
        expense_year, cost, tax_discount = self._normalized_cost_arrays(
            CapitalException
        )
        if isinstance(other, CapitalCost):
            (
                _other_project_duration,
                _other_project_years,
                other_cost_allocation,
                other_cost_type,
                other_description,
                other_tax_portion,
            ) = other._normalized_general_fields(CapitalException)
            other_expense_year, other_cost, other_tax_discount = (
                other._normalized_cost_arrays(CapitalException)
            )
            _, _, cost_allocation, cost_type, description, tax_portion = (
                self._normalized_general_fields(CapitalException)
            )
            (
                _,
                _,
                other_cost_allocation,
                other_cost_type,
                other_description,
                other_tax_portion,
            ) = other._normalized_general_fields(CapitalException)
            pis_year, salvage_value, useful_life, depreciation_factor, is_ic_applied = (
                self._normalized_capital_fields()
            )
            (
                other_pis_year,
                other_salvage_value,
                other_useful_life,
                other_depreciation_factor,
                other_is_ic_applied,
            ) = other._normalized_capital_fields()
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            expense_year_combined = np.concatenate((expense_year, other_expense_year))
            cost_combined = np.concatenate((cost, other_cost))
            pis_year_combined = np.concatenate((pis_year, other_pis_year))
            salvage_value_combined = np.concatenate(
                (salvage_value, other_salvage_value)
            )
            useful_life_combined = np.concatenate((useful_life, other_useful_life))
            depreciation_factor_combined = np.concatenate(
                (depreciation_factor, other_depreciation_factor)
            )
            tax_portion_combined = np.concatenate((tax_portion, other_tax_portion))
            tax_discount_combined = np.concatenate((tax_discount, other_tax_discount))
            cost_allocation_combined = cost_allocation + other_cost_allocation
            cost_type_combined = cost_type + other_cost_type
            description_combined = description + other_description
            is_ic_applied_combined = is_ic_applied + other_is_ic_applied

            return CapitalCost(
                start_year=start_year_combined,
                end_year=end_year_combined,
                expense_year=expense_year_combined,
                cost=cost_combined,
                cost_allocation=cost_allocation_combined,
                cost_type=cost_type_combined,
                description=description_combined,
                tax_portion=tax_portion_combined,
                tax_discount=tax_discount_combined,
                pis_year=pis_year_combined,
                salvage_value=salvage_value_combined,
                useful_life=useful_life_combined,
                depreciation_factor=depreciation_factor_combined,
                is_ic_applied=is_ic_applied_combined,
            )

        else:
            raise CapitalException(
                "Must add an instance of CapitalCost with another instance of CapitalCost. "
                + f"{other}({other.__class__.__qualname__}) is not an instance of CapitalCost."
            )

    def __iadd__(self, other: object) -> CapitalCost:
        return self.__add__(other)

    def __sub__(self, other: object) -> CapitalCost:
        # Only allows subtraction between an instance of CapitalCost
        # and another instance of CapitalCost
        (
            _project_duration,
            _project_years,
            cost_allocation,
            cost_type,
            description,
            tax_portion,
        ) = self._normalized_general_fields(CapitalException)
        expense_year, cost, tax_discount = self._normalized_cost_arrays(
            CapitalException
        )
        if isinstance(other, CapitalCost):
            (
                _other_project_duration,
                _other_project_years,
                other_cost_allocation,
                other_cost_type,
                other_description,
                other_tax_portion,
            ) = other._normalized_general_fields(CapitalException)
            other_expense_year, other_cost, other_tax_discount = (
                other._normalized_cost_arrays(CapitalException)
            )
            _, _, cost_allocation, cost_type, description, tax_portion = (
                self._normalized_general_fields(CapitalException)
            )
            (
                _,
                _,
                other_cost_allocation,
                other_cost_type,
                other_description,
                other_tax_portion,
            ) = other._normalized_general_fields(CapitalException)
            pis_year, salvage_value, useful_life, depreciation_factor, is_ic_applied = (
                self._normalized_capital_fields()
            )
            (
                other_pis_year,
                other_salvage_value,
                other_useful_life,
                other_depreciation_factor,
                other_is_ic_applied,
            ) = other._normalized_capital_fields()
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            expense_year_combined = np.concatenate((expense_year, other_expense_year))
            cost_combined = np.concatenate((cost, -other_cost))
            pis_year_combined = np.concatenate((pis_year, other_pis_year))
            salvage_value_combined = np.concatenate(
                (salvage_value, -other_salvage_value)
            )
            useful_life_combined = np.concatenate((useful_life, other_useful_life))
            depreciation_factor_combined = np.concatenate(
                (depreciation_factor, other_depreciation_factor)
            )
            tax_portion_combined = np.concatenate((tax_portion, other_tax_portion))
            tax_discount_combined = np.concatenate((tax_discount, other_tax_discount))
            cost_allocation_combined = cost_allocation + other_cost_allocation
            cost_type_combined = cost_type + other_cost_type
            description_combined = description + other_description
            is_is_applied_combined = is_ic_applied + other_is_ic_applied

            return CapitalCost(
                start_year=start_year_combined,
                end_year=end_year_combined,
                expense_year=expense_year_combined,
                cost=cost_combined,
                cost_allocation=cost_allocation_combined,
                cost_type=cost_type_combined,
                description=description_combined,
                tax_portion=tax_portion_combined,
                tax_discount=tax_discount_combined,
                pis_year=pis_year_combined,
                salvage_value=salvage_value_combined,
                useful_life=useful_life_combined,
                depreciation_factor=depreciation_factor_combined,
                is_ic_applied=is_is_applied_combined,
            )

        else:
            raise CapitalException(
                "Must subtract between an instance of CapitalCost "
                + "with another instance of CapitalCost. "
                + f"{other}({other.__class__.__qualname__}) is not "
                + "an instance of CapitalCost."
            )

    def __rsub__(self, other: object) -> CapitalCost:
        return self.__sub__(other)

    def __mul__(self, other: object) -> CapitalCost:
        # Multiplication is allowed only with an integer/a float
        (
            _project_duration,
            _project_years,
            cost_allocation,
            cost_type,
            description,
            tax_portion,
        ) = self._normalized_general_fields(CapitalException)
        expense_year, cost, tax_discount = self._normalized_cost_arrays(
            CapitalException
        )
        if isinstance(other, (int, float)):
            _, _, cost_allocation, cost_type, description, tax_portion = (
                self._normalized_general_fields(CapitalException)
            )
            pis_year, salvage_value, useful_life, depreciation_factor, is_ic_applied = (
                self._normalized_capital_fields()
            )
            return CapitalCost(
                start_year=self.start_year,
                end_year=self.end_year,
                expense_year=expense_year,
                cost=cost * other,
                cost_allocation=cost_allocation,
                cost_type=cost_type,
                description=description,
                tax_portion=tax_portion,
                tax_discount=tax_discount,
                pis_year=pis_year,
                salvage_value=salvage_value * other,
                useful_life=useful_life,
                depreciation_factor=depreciation_factor,
                is_ic_applied=is_ic_applied,
            )

        else:
            raise CapitalException(
                "Must multiply with an integer or a float. "
                + f"{other}({other.__class__.__qualname__}) "
                + "is not an integer nor a float."
            )

    def __rmul__(self, other: object) -> CapitalCost:
        return self.__mul__(other)

    def __truediv__(self, other: object) -> CapitalCost | float:
        # Between an instance of CapitalCost with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales
        (
            _project_duration,
            _project_years,
            cost_allocation,
            cost_type,
            description,
            tax_portion,
        ) = self._normalized_general_fields(CapitalException)
        expense_year, cost, tax_discount = self._normalized_cost_arrays(
            CapitalException
        )
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales)):
            (
                _other_project_duration,
                _other_project_years,
                _other_cost_allocation,
                _other_cost_type,
                _other_description,
                _other_tax_portion,
            ) = other._normalized_general_fields(CapitalException)
            _other_expense_year, other_cost, _other_tax_discount = (
                other._normalized_cost_arrays(CapitalException)
            )
            return float(np.sum(cost) / np.sum(other_cost))

        # Between an instance of CapitalCost and an integer/a float
        elif isinstance(other, (int, float)):
            # Cannot divide with zero
            if other == 0:
                raise CapitalException("Cannot divide with zero")

            else:
                _, _, cost_allocation, cost_type, description, tax_portion = (
                    self._normalized_general_fields(CapitalException)
                )
                (
                    pis_year,
                    salvage_value,
                    useful_life,
                    depreciation_factor,
                    is_ic_applied,
                ) = self._normalized_capital_fields()
                return CapitalCost(
                    start_year=self.start_year,
                    end_year=self.end_year,
                    expense_year=expense_year,
                    cost=cost / other,
                    cost_allocation=cost_allocation,
                    cost_type=cost_type,
                    description=description,
                    tax_portion=tax_portion,
                    tax_discount=tax_discount,
                    pis_year=pis_year,
                    salvage_value=salvage_value / other,
                    useful_life=useful_life,
                    depreciation_factor=depreciation_factor,
                    is_ic_applied=is_ic_applied,
                )

        else:
            raise CapitalException(
                "Must divide with an instance of CapitalCost/Intangible/OPEX/ASR/LBT"
                + f"/CostOfSales, integer or a float. {other}({other.__class__.__qualname__}) "
                + "is not an instance of CapitalCost/Intangible/OPEX/ASR/LBT not an integer "
                + "nor a float."
            )


@dataclass
@final
class Intangible(GeneralCost):
    """
    Manages an intangible asset.

    Parameters
    ----------
    The attributes are inherited from class GeneralCost.
    """

    @staticmethod
    def validate_cost_types(
        cost_type: object,
        entry_count: int,
    ) -> list[CostType | None]:
        """Normalize and validate raw cost-type input."""
        if cost_type is None:
            return [None for _ in range(entry_count)]
        if not _is_object_list(cost_type):
            raise IntangibleException(
                f"Attribute cost_type must be given as a list, not as a/an {cost_type.__class__.__qualname__}"
            )

        normalized: list[CostType | None] = []
        for value in cost_type:
            if _is_missing(value):
                normalized.append(None)
            elif isinstance(value, CostType):
                normalized.append(value)
            else:
                raise IntangibleException("Must insert CostType in list cost_type")
        return normalized

    def __post_init__(self):
        """
        Handles the following operations/procedures:

        -   Prepare attributes project_duration and project_years,
        -   Prepare attribute expense_year,
        -   Prepare attribute cost,
        -   Prepare attribute cost_allocation,
        -   Prepare attribute cost_type,
        -   Prepare attribute description,
        -   Prepare attribute tax_portion,
        -   Prepare attribute tax_discount,
        -   Check for unequal length of input arrays,
        -   Raise an error: expense_year is after the end year of the project,
        -   Raise an error: expense_year is before the start year of the project,
        """

        # Prepare attributes project_duration and project_years
        if not period_is_reversed(self.start_year, self.end_year):
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise IntangibleInputError(
                f"start year {self.start_year} is after the end year {self.end_year}"
            )

        # Prepare attribute expense_year
        if not isinstance(self.expense_year, np.ndarray):
            raise IntangibleException(
                "Attribute expense_year must be provided as a numpy.ndarray, "
                + f"not as a/an {self.expense_year.__class__.__qualname__}"
            )

        else:
            if missing_value_indices(self.expense_year):
                raise IntangibleInputError(
                    f"Missing values in array expense_year: {self.expense_year}"
                )

        self.expense_year = self.expense_year.astype(np.int64)

        # Prepare attribute cost
        if not isinstance(self.cost, np.ndarray):
            raise IntangibleException(
                "Attribute cost must be provided as a numpy.ndarray, "
                + f"not as a/an {self.cost.__class__.__qualname__}"
            )

        else:
            cost_nan_id = _missing_indices(self.cost)
            if len(cost_nan_id) > 0:
                self.cost[cost_nan_id] = np.zeros(len(cost_nan_id))

        self.cost = self.cost.astype(np.float64)

        # Prepare attribute cost_allocation
        if self.cost_allocation is None:
            self.cost_allocation = [
                FluidType.OIL for _ in range(len(self.expense_year))
            ]

        else:
            raw_cost_allocation: object = self.cost_allocation
            if not _is_object_list(raw_cost_allocation):
                raise IntangibleException(
                    "Attribute cost_allocation must be given as a list, "
                    + f"not as a/an {self.cost_allocation.__class__.__qualname__}"
                )

            self.cost_allocation = _normalize_fluid_allocation(
                raw_cost_allocation, IntangibleException
            )

        # Prepare attribute cost_type
        self.cost_type = self.validate_cost_types(
            self.cost_type,
            len(self.expense_year),
        )

        # Prepare attribute description
        if self.description is None:
            self.description = [" " for _ in range(len(self.expense_year))]

        else:
            raw_description: object = self.description
            if not _is_object_list(raw_description):
                raise IntangibleException(
                    "Attribute description must be given as a list, "
                    + f"not as a/an {self.description.__class__.__qualname__}"
                )

            normalized_description: list[str] = []
            for val in raw_description:
                if _is_missing(val):
                    normalized_description.append(" ")
                elif isinstance(val, str):
                    normalized_description.append(val)
                else:
                    raise IntangibleException("Must insert string in list description")
            self.description = normalized_description

        # Prepare attribute tax_portion
        if self.tax_portion is None:
            self.tax_portion = np.zeros_like(self.expense_year, dtype=np.float64)

        else:
            raw_tax_portion: object = self.tax_portion
            if not _is_ndarray(raw_tax_portion):
                raise IntangibleException(
                    "Attribute tax_portion must be given as a numpy.ndarray, "
                    + f"not as a/an {self.tax_portion.__class__.__qualname__}"
                )

            tax_portion_array = np.asarray(raw_tax_portion, dtype=np.float64)
            tax_portion_nan_id = _missing_indices(tax_portion_array)
            if len(tax_portion_nan_id) > 0:
                tax_portion_array[tax_portion_nan_id] = np.zeros(
                    len(tax_portion_nan_id)
                )

            if out_of_range_indices(tax_portion_array, 0.0, 1.0):
                raise IntangibleInputError(
                    "The value of tax_portion must be: 0 < tax_portion < 1, "
                    + f"tax_portion: {tax_portion_array}"
                )

        self.tax_portion = np.asarray(self.tax_portion, dtype=np.float64)

        # Prepare attribute tax_discount
        raw_tax_discount: object = self.tax_discount
        if not (
            isinstance(raw_tax_discount, (float, int)) or _is_ndarray(raw_tax_discount)
        ):
            raise IntangibleException(
                "Attribute tax_discount must be provided as a float/int or as a numpy.ndarray, "
                + f"not as a/an {self.tax_discount.__class__.__qualname__}"
            )

        if isinstance(raw_tax_discount, (float, int)):
            if out_of_range_indices((raw_tax_discount,), 0.0, 1.0):
                raise IntangibleInputError(
                    "Attribute tax_discount must be between 0 and 1"
                )

            self.tax_discount = np.repeat(raw_tax_discount, len(self.cost))

        else:
            tax_discount_array = np.asarray(raw_tax_discount, dtype=np.float64)
            tax_discount_nan_id = _missing_indices(tax_discount_array)
            if len(tax_discount_nan_id) > 0:
                tax_discount_array[tax_discount_nan_id] = np.zeros(
                    len(tax_discount_nan_id)
                )

            if out_of_range_indices(tax_discount_array, 0.0, 1.0):
                raise IntangibleInputError(
                    "The value of tax_discount must be: 0 < tax_discount < 1, "
                    + f"tax_discount: {tax_discount_array}"
                )

        self.tax_discount = np.asarray(self.tax_discount, dtype=np.float64)

        # Check input data for unequal length
        expense_year = _require_present(
            self.expense_year, IntangibleException, "expense_year"
        )
        cost = _require_present(self.cost, IntangibleException, "cost")
        cost_allocation = _require_present(
            self.cost_allocation, IntangibleException, "cost_allocation"
        )
        cost_type = _require_present(self.cost_type, IntangibleException, "cost_type")
        description = _require_present(
            self.description, IntangibleException, "description"
        )
        tax_portion = _require_present(
            self.tax_portion, IntangibleException, "tax_portion"
        )
        tax_discount = _require_present(
            self.tax_discount, IntangibleException, "tax_discount"
        )
        length_map = {
            "expense_year": len(expense_year),
            "cost": len(cost),
            "cost_allocation": len(cost_allocation),
            "cost_type": len(cost_type),
            "description": len(description),
            "tax_portion": len(tax_portion),
            "tax_discount": len(tax_discount),
        }

        if unequal_lengths(length_map) is not None:
            raise IntangibleInputError(
                "Unequal length of arrays: "
                + f"cost: {len(cost)}, "
                + f"expense_year: {len(expense_year)}, "
                + f"cost_allocation: {len(cost_allocation)}, "
                + f"cost_type: {len(cost_type)}, "
                + f"description: {len(description)}, "
                + f"tax_portion: {len(tax_portion)}, "
                + f"tax_discount: {len(tax_discount)}, "
            )

        # Raise an error: expense year is after the end year of the project
        above = year_above(self.expense_year, self.end_year)
        if above is not None:
            raise IntangibleInputError(
                f"Expense year ({above}) "
                + f"is after the end year of the project ({self.end_year})"
            )

        below = year_below(self.expense_year, self.start_year)
        if below is not None:
            raise IntangibleInputError(
                f"Expense year ({below}) "
                + f"is before the start year of the project ({self.start_year})"
            )

    @override
    def __eq__(self, other: object) -> bool:
        # Between two instances of Intangible
        (
            _project_duration,
            _project_years,
            cost_allocation,
            cost_type,
            _description,
            tax_portion,
        ) = self._normalized_general_fields(IntangibleException)
        expense_year, cost, tax_discount = self._normalized_cost_arrays(
            IntangibleException
        )
        if isinstance(other, Intangible):
            (
                _other_project_duration,
                _other_project_years,
                other_cost_allocation,
                other_cost_type,
                _other_description,
                other_tax_portion,
            ) = other._normalized_general_fields(IntangibleException)
            other_expense_year, other_cost, other_tax_discount = (
                other._normalized_cost_arrays(IntangibleException)
            )
            _, _, cost_allocation, cost_type, _, tax_portion = (
                self._normalized_general_fields(IntangibleException)
            )
            _, _, other_cost_allocation, other_cost_type, _, other_tax_portion = (
                other._normalized_general_fields(IntangibleException)
            )
            return all(
                (
                    self.start_year == other.start_year,
                    self.end_year == other.end_year,
                    np.allclose(expense_year, other_expense_year),
                    np.allclose(cost, other_cost),
                    np.allclose(tax_portion, other_tax_portion),
                    np.allclose(tax_discount, other_tax_discount),
                    cost_allocation == other_cost_allocation,
                    cost_type == other_cost_type,
                )
            )

        # Between an instance of Intangible and an integer/a float
        elif isinstance(other, (int, float)):
            return float(np.sum(cost)) == other

        else:
            return False

    def __lt__(self, other: object) -> bool:
        # Between an instance of Intangible with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales
        (
            _project_duration,
            _project_years,
            _cost_allocation,
            _cost_type,
            _description,
            _tax_portion,
        ) = self._normalized_general_fields(IntangibleException)
        _expense_year, cost, _tax_discount = self._normalized_cost_arrays(
            IntangibleException
        )
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales)):
            (
                _other_project_duration,
                _other_project_years,
                _other_cost_allocation,
                _other_cost_type,
                _other_description,
                _other_tax_portion,
            ) = other._normalized_general_fields(IntangibleException)
            _other_expense_year, other_cost, _other_tax_discount = (
                other._normalized_cost_arrays(IntangibleException)
            )
            return bool(np.sum(cost) < np.sum(other_cost))

        # Between an instance of Intangible and an integer/a float
        elif isinstance(other, (int, float)):
            return bool(np.sum(cost) < other)

        else:
            raise IntangibleException(
                "Must compare an instance of Intangible with another instance of "
                + "CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales, an integer, or a float."
            )

    def __le__(self, other: object) -> bool:
        # Between an instance of Intangible with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales
        (
            _project_duration,
            _project_years,
            _cost_allocation,
            _cost_type,
            _description,
            _tax_portion,
        ) = self._normalized_general_fields(IntangibleException)
        _expense_year, cost, _tax_discount = self._normalized_cost_arrays(
            IntangibleException
        )
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales)):
            (
                _other_project_duration,
                _other_project_years,
                _other_cost_allocation,
                _other_cost_type,
                _other_description,
                _other_tax_portion,
            ) = other._normalized_general_fields(IntangibleException)
            _other_expense_year, other_cost, _other_tax_discount = (
                other._normalized_cost_arrays(IntangibleException)
            )
            return bool(np.sum(cost) <= np.sum(other_cost))

        # Between an instance of Intangible and an integer/a float
        elif isinstance(other, (int, float)):
            return bool(np.sum(cost) <= other)

        else:
            raise IntangibleException(
                "Must compare an instance of Intangible with another instance of "
                + "CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales, an integer, or a float."
            )

    def __gt__(self, other: object) -> bool:
        # Between an instance of Intangible with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales
        (
            _project_duration,
            _project_years,
            _cost_allocation,
            _cost_type,
            _description,
            _tax_portion,
        ) = self._normalized_general_fields(IntangibleException)
        _expense_year, cost, _tax_discount = self._normalized_cost_arrays(
            IntangibleException
        )
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales)):
            (
                _other_project_duration,
                _other_project_years,
                _other_cost_allocation,
                _other_cost_type,
                _other_description,
                _other_tax_portion,
            ) = other._normalized_general_fields(IntangibleException)
            _other_expense_year, other_cost, _other_tax_discount = (
                other._normalized_cost_arrays(IntangibleException)
            )
            return bool(np.sum(cost) > np.sum(other_cost))

        # Between an instance of Intangible and an integer/a float
        elif isinstance(other, (int, float)):
            return bool(np.sum(cost) > other)

        else:
            raise IntangibleException(
                "Must compare an instance of Intangible with another instance of "
                + "CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales, an integer, or a float."
            )

    def __ge__(self, other: object) -> bool:
        # Between an instance of Intangible with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales
        (
            _project_duration,
            _project_years,
            _cost_allocation,
            _cost_type,
            _description,
            _tax_portion,
        ) = self._normalized_general_fields(IntangibleException)
        _expense_year, cost, _tax_discount = self._normalized_cost_arrays(
            IntangibleException
        )
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales)):
            (
                _other_project_duration,
                _other_project_years,
                _other_cost_allocation,
                _other_cost_type,
                _other_description,
                _other_tax_portion,
            ) = other._normalized_general_fields(IntangibleException)
            _other_expense_year, other_cost, _other_tax_discount = (
                other._normalized_cost_arrays(IntangibleException)
            )
            return bool(np.sum(cost) >= np.sum(other_cost))

        # Between an instance of Intangible and an integer/a float
        elif isinstance(other, (int, float)):
            return bool(np.sum(cost) >= other)

        else:
            raise IntangibleException(
                "Must compare an instance of Intangible with another instance of "
                + "CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales, an integer, or a float."
            )

    def __add__(self, other: object) -> Intangible:
        # Only allows addition between an instance of Intangible
        # and another instance of Intangible
        (
            _project_duration,
            _project_years,
            cost_allocation,
            cost_type,
            description,
            tax_portion,
        ) = self._normalized_general_fields(IntangibleException)
        expense_year, cost, tax_discount = self._normalized_cost_arrays(
            IntangibleException
        )
        if isinstance(other, Intangible):
            (
                _other_project_duration,
                _other_project_years,
                other_cost_allocation,
                other_cost_type,
                other_description,
                other_tax_portion,
            ) = other._normalized_general_fields(IntangibleException)
            other_expense_year, other_cost, other_tax_discount = (
                other._normalized_cost_arrays(IntangibleException)
            )
            _, _, cost_allocation, cost_type, description, tax_portion = (
                self._normalized_general_fields(IntangibleException)
            )
            (
                _,
                _,
                other_cost_allocation,
                other_cost_type,
                other_description,
                other_tax_portion,
            ) = other._normalized_general_fields(IntangibleException)
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            expense_year_combined = np.concatenate((expense_year, other_expense_year))
            cost_combined = np.concatenate((cost, other_cost))
            tax_portion_combined = np.concatenate((tax_portion, other_tax_portion))
            tax_discount_combined = np.concatenate((tax_discount, other_tax_discount))
            cost_allocation_combined = cost_allocation + other_cost_allocation
            cost_type_combined = cost_type + other_cost_type
            description_combined = description + other_description

            return Intangible(
                start_year=start_year_combined,
                end_year=end_year_combined,
                expense_year=expense_year_combined,
                cost=cost_combined,
                cost_allocation=cost_allocation_combined,
                cost_type=cost_type_combined,
                description=description_combined,
                tax_portion=tax_portion_combined,
                tax_discount=tax_discount_combined,
            )

        else:
            raise IntangibleException(
                "Must add an instance of Intangible with another instance of Intangible. "
                + f"{other}({other.__class__.__qualname__}) is not an instance of Intangible."
            )

    def __iadd__(self, other: object) -> Intangible:
        return self.__add__(other)

    def __sub__(self, other: object) -> Intangible:
        # Only allows subtraction between an instance of Intangible
        # and another instance of Intangible
        (
            _project_duration,
            _project_years,
            cost_allocation,
            cost_type,
            description,
            tax_portion,
        ) = self._normalized_general_fields(IntangibleException)
        expense_year, cost, tax_discount = self._normalized_cost_arrays(
            IntangibleException
        )
        if isinstance(other, Intangible):
            (
                _other_project_duration,
                _other_project_years,
                other_cost_allocation,
                other_cost_type,
                other_description,
                other_tax_portion,
            ) = other._normalized_general_fields(IntangibleException)
            other_expense_year, other_cost, other_tax_discount = (
                other._normalized_cost_arrays(IntangibleException)
            )
            _, _, cost_allocation, cost_type, description, tax_portion = (
                self._normalized_general_fields(IntangibleException)
            )
            (
                _,
                _,
                other_cost_allocation,
                other_cost_type,
                other_description,
                other_tax_portion,
            ) = other._normalized_general_fields(IntangibleException)
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            expense_year_combined = np.concatenate((expense_year, other_expense_year))
            cost_combined = np.concatenate((cost, -other_cost))
            tax_portion_combined = np.concatenate((tax_portion, other_tax_portion))
            tax_discount_combined = np.concatenate((tax_discount, other_tax_discount))
            cost_allocation_combined = cost_allocation + other_cost_allocation
            cost_type_combined = cost_type + other_cost_type
            description_combined = description + other_description

            return Intangible(
                start_year=start_year_combined,
                end_year=end_year_combined,
                expense_year=expense_year_combined,
                cost=cost_combined,
                cost_allocation=cost_allocation_combined,
                cost_type=cost_type_combined,
                description=description_combined,
                tax_portion=tax_portion_combined,
                tax_discount=tax_discount_combined,
            )

        else:
            raise IntangibleException(
                "Must subtract an instance of Intangible with another instance of Intangible. "
                + f"{other}({other.__class__.__qualname__}) is not an instance of Intangible."
            )

    def __rsub__(self, other: object) -> Intangible:
        return self.__sub__(other)

    def __mul__(self, other: object) -> Intangible:
        # Multiplication is allowed only with an integer/a float
        (
            _project_duration,
            _project_years,
            cost_allocation,
            cost_type,
            description,
            tax_portion,
        ) = self._normalized_general_fields(IntangibleException)
        expense_year, cost, tax_discount = self._normalized_cost_arrays(
            IntangibleException
        )
        if isinstance(other, (int, float)):
            _, _, cost_allocation, cost_type, description, tax_portion = (
                self._normalized_general_fields(IntangibleException)
            )
            return Intangible(
                start_year=self.start_year,
                end_year=self.end_year,
                expense_year=expense_year,
                cost=cost * other,
                cost_allocation=cost_allocation,
                cost_type=cost_type,
                description=description,
                tax_portion=tax_portion,
                tax_discount=tax_discount,
            )

        else:
            raise IntangibleException(
                "Must multiply with an integer or a float. "
                + f"{other}({other.__class__.__qualname__}) "
                + "is not an integer nor a float."
            )

    def __rmul__(self, other: object) -> Intangible:
        return self.__mul__(other)

    def __truediv__(self, other: object) -> Intangible | float:
        # Between an instance of Intangible with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales
        (
            _project_duration,
            _project_years,
            cost_allocation,
            cost_type,
            description,
            tax_portion,
        ) = self._normalized_general_fields(IntangibleException)
        expense_year, cost, tax_discount = self._normalized_cost_arrays(
            IntangibleException
        )
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales)):
            (
                _other_project_duration,
                _other_project_years,
                _other_cost_allocation,
                _other_cost_type,
                _other_description,
                _other_tax_portion,
            ) = other._normalized_general_fields(IntangibleException)
            _other_expense_year, other_cost, _other_tax_discount = (
                other._normalized_cost_arrays(IntangibleException)
            )
            return float(np.sum(cost) / np.sum(other_cost))

        # Between an instance of Intangible and an integer/a float
        elif isinstance(other, (int, float)):
            # Cannot divide by zero
            if other == 0:
                raise IntangibleException("Cannot divide by zero")

            else:
                _, _, cost_allocation, cost_type, description, tax_portion = (
                    self._normalized_general_fields(IntangibleException)
                )
                return Intangible(
                    start_year=self.start_year,
                    end_year=self.end_year,
                    expense_year=expense_year,
                    cost=cost / other,
                    cost_allocation=cost_allocation,
                    cost_type=cost_type,
                    description=description,
                    tax_portion=tax_portion,
                    tax_discount=tax_discount,
                )

        else:
            raise IntangibleException(
                "Must divide with an instance of CapitalCost/Intangible/OPEX/ASR/LBT"
                + f"/CostOfSales, integer or a float. {other}({other.__class__.__qualname__}) "
                + "is not an instance of CapitalCost/Intangible/OPEX/ASR/LBT nor an integer "
                + "nor a float."
            )


@dataclass
@final
class OPEX(GeneralCost):
    """
    Manages an OPEX asset.

    Parameters
    ----------
    The attributes are inherited from class GeneralCost. Local attributes associated
    with class OPEX are:

    fixed_cost : np.ndarray
        An array representing the fixed cost of an OPEX asset.
    prod_rate: np.ndarray
        The production rate of a particular fluid type.
    cost_per_volume: np.ndarray
        Cost associated with production of a particular fluid type.
    """

    fixed_cost: npt.NDArray[np.float64] | None = None
    prod_rate: npt.NDArray[np.float64] | None = None
    cost_per_volume: npt.NDArray[np.float64] | None = None

    # Attributes to be defined later
    variable_cost: npt.NDArray[np.float64] | None = field(default=None, init=False)
    cost: npt.NDArray[np.float64] | None = field(default=None, init=False)

    def _normalized_opex_fields(
        self,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Return OPEX arrays after constructor normalization."""
        return (
            _require_present(self.fixed_cost, OPEXException, "fixed_cost"),
            _require_present(self.prod_rate, OPEXException, "prod_rate"),
            _require_present(self.cost_per_volume, OPEXException, "cost_per_volume"),
            _require_present(self.variable_cost, OPEXException, "variable_cost"),
        )

    def __post_init__(self):
        """
        Handles the following operations/procedures:
        -   Prepare attributes project_duration and project_years,
        -   Prepare attribute expense_year,
        -   Prepare attribute fixed_cost,
        -   Prepare attribute cost_allocation,
        -   Prepare attribute cost_type,
        -   Prepare attribute description,
        -   Prepare attribute tax_portion,
        -   Prepare attribute tax_discount,
        -   Prepare attribute prod_rate,
        -   Prepare attribute cost_per_volume,
        -   Check input data for unequal length of arrays,
        -   Prepare attribute variable_cost,
        -   Prepare attribute cost,
        -   Raise an error: expense_year is after the end year of the project,
        -   Raise an error: expense_year is before the start year of the project,
        """

        # Prepare attributes project_duration and project_years
        if not period_is_reversed(self.start_year, self.end_year):
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise OPEXInputError(
                f"start year {self.start_year} is after the end year {self.end_year}"
            )

        # Prepare attribute expense_year
        expense_year = _require_opex_array(self.expense_year, "expense_year")
        if missing_value_indices(expense_year):
            raise OPEXInputError(f"Missing values in array expense_year: {expense_year}")

        self.expense_year = expense_year.astype(np.int64)

        # Prepare attribute fixed_cost
        fixed_cost = self.fixed_cost
        fixed_cost_array: npt.NDArray[np.generic] = np.zeros_like(
            self.expense_year, dtype=np.float64
        )
        if fixed_cost is None:
            self.fixed_cost = np.zeros_like(self.expense_year, dtype=np.float64)

        else:
            fixed_cost_array = _require_opex_array(fixed_cost, "fixed_cost")
            fixed_cost_nan_id = _missing_indices(fixed_cost_array)
            if len(fixed_cost_nan_id) > 0:
                fixed_cost_array[fixed_cost_nan_id] = np.zeros(len(fixed_cost_nan_id))

        self.fixed_cost = (
            np.zeros_like(self.expense_year, dtype=np.float64)
            if fixed_cost is None
            else fixed_cost_array.astype(np.float64)
        )

        # Prepare attribute cost_allocation
        cost_allocation = self.cost_allocation
        if cost_allocation is None:
            normalized_cost_allocation = [
                FluidType.OIL for _ in range(len(self.expense_year))
            ]

        else:
            cost_allocation = _require_opex_object_list(
                cost_allocation, "cost_allocation"
            )

            normalized_cost_allocation: list[FluidType] = []
            for val in cost_allocation:
                if _is_missing(val):
                    normalized_cost_allocation.append(FluidType.OIL)
                elif isinstance(val, FluidType):
                    normalized_cost_allocation.append(val)
                else:
                    raise OPEXException("Must insert FluidType in list cost_allocation")
        self.cost_allocation = normalized_cost_allocation

        # Prepare attribute cost_type
        cost_type = self.cost_type
        if cost_type is None:
            normalized_cost_type: list[CostType | None] = [
                None for _ in range(len(self.expense_year))
            ]

        else:
            cost_type = _require_opex_object_list(cost_type, "cost_type")

            normalized_cost_type = []
            for val in cost_type:
                if _is_missing(val) or val is None:
                    normalized_cost_type.append(None)
                elif isinstance(val, CostType):
                    normalized_cost_type.append(val)
                else:
                    raise OPEXException("Must insert CostType in list cost_type")
        self.cost_type = normalized_cost_type

        # Prepare attribute description
        description = self.description
        if description is None:
            normalized_description: list[str] = [
                " " for _ in range(len(self.expense_year))
            ]

        else:
            description = _require_opex_object_list(description, "description")

            normalized_description = []
            for val in description:
                if _is_missing(val):
                    normalized_description.append(" ")
                elif isinstance(val, str):
                    normalized_description.append(val)
                else:
                    raise OPEXException("Must insert str in list description")
        self.description = normalized_description

        # Prepare attribute tax_portion
        if self.tax_portion is None:
            self.tax_portion = np.zeros_like(self.expense_year, dtype=np.float64)

        else:
            if type(self.tax_portion) is not np.ndarray:
                raise OPEXException(
                    "Attribute tax_portion must be given as a numpy.ndarray, "
                    + f"not as a/an {self.tax_portion.__class__.__qualname__}"
                )

            tax_portion_array = np.asarray(self.tax_portion)
            tax_portion_nan_id = _missing_indices(tax_portion_array)
            if len(tax_portion_nan_id) > 0:
                tax_portion_array[tax_portion_nan_id] = np.zeros(
                    len(tax_portion_nan_id)
                )

            if out_of_range_indices(tax_portion_array, 0.0, 1.0):
                raise OPEXInputError(
                    "The value of tax_portion must be: 0 < tax_portion < 1, "
                    + f"tax_portion: {tax_portion_array}"
                )

        self.tax_portion = np.asarray(self.tax_portion, dtype=np.float64)

        # Prepare attribute tax_discount
        tax_discount = _require_opex_tax_discount(self.tax_discount)

        if isinstance(tax_discount, (float, int)):
            if out_of_range_indices((tax_discount,), 0.0, 1.0):
                raise OPEXInputError("Attribute tax_discount must be between 0 and 1")

            normalized_tax_discount = np.repeat(tax_discount, len(self.expense_year))

        else:
            tax_discount_array = tax_discount.astype(np.float64)
            tax_discount_nan_id = _missing_indices(tax_discount_array)
            if len(tax_discount_nan_id) > 0:
                tax_discount_array[tax_discount_nan_id] = np.zeros(
                    len(tax_discount_nan_id)
                )

            if out_of_range_indices(tax_discount_array, 0.0, 1.0):
                raise OPEXInputError(
                    "The value of tax_discount must be: 0 < tax_discount < 1, "
                    + f"tax_discount: {tax_discount}"
                )

            normalized_tax_discount = tax_discount_array

        self.tax_discount = normalized_tax_discount.astype(np.float64)

        # Prepare attribute prod_rate
        if self.prod_rate is None:
            self.prod_rate = np.zeros_like(self.expense_year, dtype=np.float64)

        else:
            if type(self.prod_rate) is not np.ndarray:
                raise OPEXException(
                    "Attribute prod_rate must be provided as a numpy.ndarray, "
                    + f"not as a/an {self.prod_rate.__class__.__qualname__}"
                )

            else:
                prod_rate_nan_id = _missing_indices(self.prod_rate)
                if len(prod_rate_nan_id) > 0:
                    self.prod_rate[prod_rate_nan_id] = np.zeros(len(prod_rate_nan_id))

        self.prod_rate = self.prod_rate.astype(np.float64)

        # Prepare attribute cost_per_volume
        if self.cost_per_volume is None:
            self.cost_per_volume = np.zeros_like(self.expense_year, dtype=np.float64)

        else:
            if type(self.cost_per_volume) is not np.ndarray:
                raise OPEXException(
                    "Attribute cost_per_volume must be provided as a numpy.ndarray, "
                    + f"not as a/an {self.cost_per_volume.__class__.__qualname__}"
                )

            else:
                cost_per_volume_nan_id = _missing_indices(self.cost_per_volume)
                if len(cost_per_volume_nan_id) > 0:
                    self.cost_per_volume[cost_per_volume_nan_id] = np.zeros(
                        len(cost_per_volume_nan_id)
                    )

        self.cost_per_volume = self.cost_per_volume.astype(np.float64)

        # Check input data for unequal length
        length_map = {
            "expense_year": len(self.expense_year),
            "fixed_cost": len(self.fixed_cost),
            "cost_allocation": len(self.cost_allocation),
            "cost_type": len(self.cost_type),
            "description": len(self.description),
            "tax_portion": len(self.tax_portion),
            "tax_discount": len(self.tax_discount),
            "prod_rate": len(self.prod_rate),
            "cost_per_volume": len(self.cost_per_volume),
        }

        if unequal_lengths(length_map) is not None:
            raise OPEXInputError(
                "Unequal length of arrays: "
                + f"expense_year: {len(self.expense_year)}, "
                + f"fixed_cost: {len(self.fixed_cost)}, "
                + f"cost_allocation: {len(self.cost_allocation)}, "
                + f"cost_type: {len(self.cost_type)}, "
                + f"description: {len(self.description)}, "
                + f"tax_portion: {len(self.tax_portion)}, "
                + f"tax_discount: {len(self.tax_discount)}, "
                + f"prod_rate: {len(self.prod_rate)}, "
                + f"cost_per_volume: {len(self.cost_per_volume)}, "
            )

        # Prepare attribute variable_cost
        self.variable_cost = self.prod_rate * self.cost_per_volume

        # Prepare attribute cost
        self.cost = self.fixed_cost + self.variable_cost

        # Raise an error: expense year is after the end year of the project
        above = year_above(self.expense_year, self.end_year)
        if above is not None:
            raise OPEXInputError(
                f"Expense year ({above}) "
                + f"is after the end year of the project ({self.end_year})"
            )

        # Raise an error: expense year is before the start year of the project
        below = year_below(self.expense_year, self.start_year)
        if below is not None:
            raise OPEXInputError(
                f"Expense year ({below}) "
                + f"is before the start year of the project ({self.start_year})"
            )

    @override
    def __eq__(self, other: object) -> bool:
        # Between two instances of OPEX
        fixed_cost, _prod_rate, _cost_per_volume, variable_cost = (
            self._normalized_opex_fields()
        )
        (
            _project_duration,
            _project_years,
            cost_allocation,
            cost_type,
            _description,
            tax_portion,
        ) = self._normalized_general_fields(OPEXException)
        expense_year, cost, tax_discount = self._normalized_cost_arrays(OPEXException)
        if isinstance(other, OPEX):
            (
                other_fixed_cost,
                _other_prod_rate,
                _other_cost_per_volume,
                other_variable_cost,
            ) = other._normalized_opex_fields()
            (
                _other_project_duration,
                _other_project_years,
                other_cost_allocation,
                other_cost_type,
                _other_description,
                other_tax_portion,
            ) = other._normalized_general_fields(OPEXException)
            other_expense_year, other_cost, other_tax_discount = (
                other._normalized_cost_arrays(OPEXException)
            )
            return all(
                (
                    self.start_year == other.start_year,
                    self.end_year == other.end_year,
                    np.allclose(expense_year, other_expense_year),
                    np.allclose(fixed_cost, other_fixed_cost),
                    np.allclose(tax_portion, other_tax_portion),
                    np.allclose(tax_discount, other_tax_discount),
                    np.allclose(variable_cost, other_variable_cost),
                    np.allclose(cost, other_cost),
                    cost_allocation == other_cost_allocation,
                    cost_type == other_cost_type,
                )
            )

        # Between an instance of OPEX and an integer/a float
        elif isinstance(other, (int, float)):
            return float(np.sum(cost)) == other

        else:
            return False

    def __lt__(self, other: object) -> bool:
        # Between an instance of OPEX with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales
        (
            _project_duration,
            _project_years,
            _cost_allocation,
            _cost_type,
            _description,
            _tax_portion,
        ) = self._normalized_general_fields(OPEXException)
        _expense_year, cost, _tax_discount = self._normalized_cost_arrays(OPEXException)
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales)):
            (
                _other_project_duration,
                _other_project_years,
                _other_cost_allocation,
                _other_cost_type,
                _other_description,
                _other_tax_portion,
            ) = other._normalized_general_fields(OPEXException)
            _other_expense_year, other_cost, _other_tax_discount = (
                other._normalized_cost_arrays(OPEXException)
            )
            return bool(np.sum(cost) < np.sum(other_cost))

        # Between an instance of OPEX and an integer/float
        elif isinstance(other, (int, float)):
            return bool(np.sum(cost) < other)

        else:
            raise OPEXException(
                "Must compare an instance of OPEX with another instance of "
                + "CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales, an integer, or a float."
            )

    def __le__(self, other: object) -> bool:
        # Between an instance of OPEX with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales
        (
            _project_duration,
            _project_years,
            _cost_allocation,
            _cost_type,
            _description,
            _tax_portion,
        ) = self._normalized_general_fields(OPEXException)
        _expense_year, cost, _tax_discount = self._normalized_cost_arrays(OPEXException)
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales)):
            (
                _other_project_duration,
                _other_project_years,
                _other_cost_allocation,
                _other_cost_type,
                _other_description,
                _other_tax_portion,
            ) = other._normalized_general_fields(OPEXException)
            _other_expense_year, other_cost, _other_tax_discount = (
                other._normalized_cost_arrays(OPEXException)
            )
            return bool(np.sum(cost) <= np.sum(other_cost))

        # Between an instance of OPEX and an integer/float
        elif isinstance(other, (int, float)):
            return bool(np.sum(cost) <= other)

        else:
            raise OPEXException(
                "Must compare an instance of OPEX with another instance of "
                + "CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales, an integer, or a float."
            )

    def __gt__(self, other: object) -> bool:
        # Between an instance of OPEX with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales
        (
            _project_duration,
            _project_years,
            _cost_allocation,
            _cost_type,
            _description,
            _tax_portion,
        ) = self._normalized_general_fields(OPEXException)
        _expense_year, cost, _tax_discount = self._normalized_cost_arrays(OPEXException)
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales)):
            (
                _other_project_duration,
                _other_project_years,
                _other_cost_allocation,
                _other_cost_type,
                _other_description,
                _other_tax_portion,
            ) = other._normalized_general_fields(OPEXException)
            _other_expense_year, other_cost, _other_tax_discount = (
                other._normalized_cost_arrays(OPEXException)
            )
            return bool(np.sum(cost) > np.sum(other_cost))

        # Between an instance of OPEX and an integer/float
        elif isinstance(other, (int, float)):
            return bool(np.sum(cost) > other)

        else:
            raise OPEXException(
                "Must compare an instance of OPEX with another instance of "
                + "CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales, an integer, or a float."
            )

    def __ge__(self, other: object) -> bool:
        # Between an instance of OPEX with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales
        (
            _project_duration,
            _project_years,
            _cost_allocation,
            _cost_type,
            _description,
            _tax_portion,
        ) = self._normalized_general_fields(OPEXException)
        _expense_year, cost, _tax_discount = self._normalized_cost_arrays(OPEXException)
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales)):
            (
                _other_project_duration,
                _other_project_years,
                _other_cost_allocation,
                _other_cost_type,
                _other_description,
                _other_tax_portion,
            ) = other._normalized_general_fields(OPEXException)
            _other_expense_year, other_cost, _other_tax_discount = (
                other._normalized_cost_arrays(OPEXException)
            )
            return bool(np.sum(cost) >= np.sum(other_cost))

        # Between an instance of OPEX and an integer/float
        elif isinstance(other, (int, float)):
            return bool(np.sum(cost) >= other)

        else:
            raise OPEXException(
                "Must compare an instance of OPEX with another instance of "
                + "CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales, an integer, or a float."
            )

    def __add__(self, other: object) -> OPEX:
        # Only allows addition between an instance of OPEX
        # and another instance of OPEX
        fixed_cost, prod_rate, cost_per_volume, _variable_cost = (
            self._normalized_opex_fields()
        )
        (
            _project_duration,
            _project_years,
            cost_allocation,
            cost_type,
            description,
            tax_portion,
        ) = self._normalized_general_fields(OPEXException)
        expense_year, _cost, tax_discount = self._normalized_cost_arrays(OPEXException)
        if isinstance(other, OPEX):
            (
                other_fixed_cost,
                other_prod_rate,
                other_cost_per_volume,
                _other_variable_cost,
            ) = other._normalized_opex_fields()
            (
                _other_project_duration,
                _other_project_years,
                other_cost_allocation,
                other_cost_type,
                other_description,
                other_tax_portion,
            ) = other._normalized_general_fields(OPEXException)
            other_expense_year, _other_cost, other_tax_discount = (
                other._normalized_cost_arrays(OPEXException)
            )
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            expense_year_combined = np.concatenate((expense_year, other_expense_year))
            fixed_cost_combined = np.concatenate((fixed_cost, other_fixed_cost))
            tax_portion_combined = np.concatenate((tax_portion, other_tax_portion))
            tax_discount_combined = np.concatenate((tax_discount, other_tax_discount))
            prod_rate_combined = np.concatenate((prod_rate, other_prod_rate))
            cost_per_volume_combined = np.concatenate(
                (cost_per_volume, other_cost_per_volume)
            )
            cost_allocation_combined = cost_allocation + other_cost_allocation
            cost_type_combined = cost_type + other_cost_type
            description_combined = description + other_description

            return OPEX(
                start_year=start_year_combined,
                end_year=end_year_combined,
                expense_year=expense_year_combined,
                fixed_cost=fixed_cost_combined,
                cost_allocation=cost_allocation_combined,
                cost_type=cost_type_combined,
                description=description_combined,
                tax_portion=tax_portion_combined,
                tax_discount=tax_discount_combined,
                prod_rate=prod_rate_combined,
                cost_per_volume=cost_per_volume_combined,
            )

        else:
            raise OPEXException(
                "Must add an instance of OPEX with another instance of OPEX. "
                + f"{other}({other.__class__.__qualname__}) is not an instance of OPEX."
            )

    def __iadd__(self, other: object) -> OPEX:
        return self.__add__(other)

    def __sub__(self, other: object) -> OPEX:
        # Only allows subtraction between an instance of OPEX
        # and another instance of OPEX
        fixed_cost, prod_rate, cost_per_volume, _variable_cost = (
            self._normalized_opex_fields()
        )
        (
            _project_duration,
            _project_years,
            cost_allocation,
            cost_type,
            description,
            tax_portion,
        ) = self._normalized_general_fields(OPEXException)
        expense_year, _cost, tax_discount = self._normalized_cost_arrays(OPEXException)
        if isinstance(other, OPEX):
            (
                other_fixed_cost,
                other_prod_rate,
                other_cost_per_volume,
                _other_variable_cost,
            ) = other._normalized_opex_fields()
            (
                _other_project_duration,
                _other_project_years,
                other_cost_allocation,
                other_cost_type,
                other_description,
                other_tax_portion,
            ) = other._normalized_general_fields(OPEXException)
            other_expense_year, _other_cost, other_tax_discount = (
                other._normalized_cost_arrays(OPEXException)
            )
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            expense_year_combined = np.concatenate((expense_year, other_expense_year))
            fixed_cost_combined = np.concatenate((fixed_cost, -other_fixed_cost))
            tax_portion_combined = np.concatenate((tax_portion, other_tax_portion))
            tax_discount_combined = np.concatenate((tax_discount, other_tax_discount))
            prod_rate_combined = np.concatenate((prod_rate, other_prod_rate))
            cost_per_volume_combined = np.concatenate(
                (cost_per_volume, -other_cost_per_volume)
            )
            cost_allocation_combined = cost_allocation + other_cost_allocation
            cost_type_combined = cost_type + other_cost_type
            description_combined = description + other_description

            return OPEX(
                start_year=start_year_combined,
                end_year=end_year_combined,
                expense_year=expense_year_combined,
                fixed_cost=fixed_cost_combined,
                cost_allocation=cost_allocation_combined,
                cost_type=cost_type_combined,
                description=description_combined,
                tax_portion=tax_portion_combined,
                tax_discount=tax_discount_combined,
                prod_rate=prod_rate_combined,
                cost_per_volume=cost_per_volume_combined,
            )

        else:
            raise OPEXException(
                "Must subtract an instance of OPEX with another instance of OPEX. "
                + f"{other}({other.__class__.__qualname__}) is not an instance of OPEX."
            )

    def __rsub__(self, other: object) -> OPEX:
        return self.__sub__(other)

    def __mul__(self, other: object) -> OPEX:
        # Multiplication is only allowed with an integer/a float
        fixed_cost, prod_rate, cost_per_volume, _variable_cost = (
            self._normalized_opex_fields()
        )
        (
            _project_duration,
            _project_years,
            cost_allocation,
            cost_type,
            description,
            tax_portion,
        ) = self._normalized_general_fields(OPEXException)
        expense_year, _cost, tax_discount = self._normalized_cost_arrays(OPEXException)
        if isinstance(other, (int, float)):
            return OPEX(
                start_year=self.start_year,
                end_year=self.end_year,
                expense_year=expense_year,
                fixed_cost=fixed_cost * other,
                cost_allocation=cost_allocation,
                cost_type=cost_type,
                description=description,
                tax_portion=tax_portion,
                tax_discount=tax_discount,
                prod_rate=prod_rate,
                cost_per_volume=cost_per_volume * other,
            )

        else:
            raise OPEXException(
                "Must multiply with an integer or a float. "
                + f"{other}({other.__class__.__qualname__}) "
                + "is not an integer nor a float."
            )

    def __rmul__(self, other: object) -> OPEX:
        return self.__mul__(other)

    def __truediv__(self, other: object) -> OPEX | float:
        # Between an instance of OPEX with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales
        fixed_cost, prod_rate, cost_per_volume, _variable_cost = (
            self._normalized_opex_fields()
        )
        (
            _project_duration,
            _project_years,
            cost_allocation,
            cost_type,
            description,
            tax_portion,
        ) = self._normalized_general_fields(OPEXException)
        expense_year, cost, tax_discount = self._normalized_cost_arrays(OPEXException)
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales)):
            (
                _other_project_duration,
                _other_project_years,
                _other_cost_allocation,
                _other_cost_type,
                _other_description,
                _other_tax_portion,
            ) = other._normalized_general_fields(OPEXException)
            _other_expense_year, other_cost, _other_tax_discount = (
                other._normalized_cost_arrays(OPEXException)
            )
            return float(np.sum(cost) / np.sum(other_cost))

        # Between an instance of OPEX and an integer/a float
        elif isinstance(other, (int, float)):
            # Cannot divide by zero
            if other == 0:
                raise OPEXException("Cannot divide with zero")

            else:
                return OPEX(
                    start_year=self.start_year,
                    end_year=self.end_year,
                    expense_year=expense_year,
                    fixed_cost=fixed_cost / other,
                    cost_allocation=cost_allocation,
                    cost_type=cost_type,
                    description=description,
                    tax_portion=tax_portion,
                    tax_discount=tax_discount,
                    prod_rate=prod_rate,
                    cost_per_volume=cost_per_volume / other,
                )

        else:
            raise OPEXException(
                "Must divide with an instance of CapitalCost/Intangible/OPEX/ASR/LBT"
                + f"/CostOfSales, integer or a float. {other}({other.__class__.__qualname__}) "
                + "is not an instance of CapitalCost/Intangible/OPEX/ASR/LBT not an integer "
                + "nor a float."
            )


@dataclass
@final
class ASR(GeneralCost):
    """
    Manages an ASR asset.

    Parameters
    ----------
    The attributes are inherited from class GeneralCost.
    Local attributes associated with class ASR are:

    final_year : np.ndarray
        The year in which cost distribution ends for each cost element.
    future_rate : float or np.ndarray, optional
        Future rates applied to costs for each year, defaulting to a rate of 0%.
    """

    final_year: npt.NDArray[np.int64] | None = None
    future_rate: float | npt.NDArray[np.float64] | None = None

    def _normalized_asr_fields(
        self,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        """Return ASR arrays after constructor normalization."""
        final_year = _require_present(self.final_year, ASRException, "final_year")
        if not _is_ndarray(self.future_rate):
            raise ASRException("Attribute future_rate is not initialized as an array")
        return final_year, self.future_rate.astype(np.float64, copy=False)

    def __post_init__(self):
        """
        Handles the following operations/procedures:
        -   Prepare attributes project_duration and project_years,
        -   Prepare attribute expense_year,
        -   Prepare attribute cost,
        -   Prepare attribute cost_allocation,
        -   Prepare attribute cost_type,
        -   Prepare attribute description,
        -   Prepare attribute tax_portion,
        -   Prepare attribute tax_discount,
        -   Prepare attribute future_rate,
        -   Prepare attribute final_year,
        -   Check input data for unequal length,
        -   Raise an exception: expense year is after the end year of the project,
        -   Raise an exception: expense year is before the start year of the project,
        """

        # Prepare attributes project_duration and project_years
        if not period_is_reversed(self.start_year, self.end_year):
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise ASRInputError(
                f"start year {self.start_year} is after the end year {self.end_year}"
            )

        # Prepare attribute expense_year
        if not isinstance(self.expense_year, np.ndarray):
            raise ASRException(
                "Attribute expense_year must be provided as a numpy.ndarray, "
                + f"not as a/an {self.expense_year.__class__.__qualname__}"
            )

        else:
            if missing_value_indices(self.expense_year):
                raise ASRInputError(
                    f"Missing values in array expense_year: {self.expense_year}"
                )

        self.expense_year = self.expense_year.astype(np.int64)

        # Prepare attribute cost
        if not isinstance(self.cost, np.ndarray):
            raise ASRException(
                "Attribute cost must be provided as a numpy.ndarray, "
                + f"not as a/an {self.cost.__class__.__qualname__}"
            )

        else:
            cost_nan_id = _missing_indices(self.cost)
            if len(cost_nan_id) > 0:
                self.cost[cost_nan_id] = np.zeros(len(cost_nan_id))

        self.cost = self.cost.astype(np.float64)

        # Prepare attribute cost_allocation
        if self.cost_allocation is None:
            self.cost_allocation = [
                FluidType.OIL for _ in range(len(self.expense_year))
            ]

        else:
            raw_cost_allocation = _require_asr_object_list(
                self.cost_allocation, "cost_allocation"
            )
            normalized_allocation: list[FluidType] = []
            for val in raw_cost_allocation:
                if _is_missing(val):
                    normalized_allocation.append(FluidType.OIL)
                elif isinstance(val, FluidType):
                    normalized_allocation.append(val)
                else:
                    raise ASRException("Must insert FluidType in list cost_allocation")
            self.cost_allocation = normalized_allocation

        # Prepare attribute cost_type
        raw_cost_type: object = self.cost_type
        if raw_cost_type is None:
            self.cost_type = [None for _ in range(len(self.expense_year))]

        else:
            if not _is_object_list(raw_cost_type):
                raise ASRException(
                    "Attribute cost_type must be given as a list, not "
                    + f"as a/an {self.cost_type.__class__.__qualname__}"
                )

            self.cost_type = _normalize_cost_type(raw_cost_type, ASRException)

        # Prepare attribute description
        raw_description: object = self.description
        if raw_description is None:
            self.description = [" " for _ in range(len(self.expense_year))]

        else:
            raw_description_list = _require_asr_object_list(
                raw_description, "description"
            )
            normalized_description: list[str] = []
            for val in raw_description_list:
                if _is_missing(val):
                    normalized_description.append(" ")
                elif isinstance(val, str):
                    normalized_description.append(val)
                else:
                    raise ASRException("Description values must be strings")
            self.description = normalized_description

        # Prepare attribute tax_portion
        raw_tax_portion: object = self.tax_portion
        if raw_tax_portion is None:
            self.tax_portion = np.zeros_like(self.cost, dtype=np.float64)

        else:
            tax_portion_array = np.asarray(
                _require_asr_array(raw_tax_portion, "tax_portion"), dtype=np.float64
            )
            tax_portion_nan_id = _missing_indices(tax_portion_array)
            if len(tax_portion_nan_id) > 0:
                tax_portion_array[tax_portion_nan_id] = np.zeros(
                    len(tax_portion_nan_id)
                )

            if out_of_range_indices(tax_portion_array, 0.0, 1.0):
                raise ASRInputError(
                    "The value of tax_portion must be: 0 < tax_portion < 1, "
                    + f"tax_portion: {tax_portion_array}"
                )

        self.tax_portion = np.asarray(
            _require_present(self.tax_portion, ASRException, "tax_portion"),
            dtype=np.float64,
        )

        # Prepare attribute tax_discount
        raw_tax_discount: object = self.tax_discount
        raw_tax_discount = _require_asr_tax_discount(raw_tax_discount)

        if isinstance(raw_tax_discount, (float, int)):
            if out_of_range_indices((raw_tax_discount,), 0.0, 1.0):
                raise ASRInputError("Attribute tax_discount must be between 0 and 1")

            self.tax_discount = np.repeat(raw_tax_discount, len(self.cost))

        else:
            tax_discount_nan_id = _missing_indices(raw_tax_discount)
            if len(tax_discount_nan_id) > 0:
                raw_tax_discount[tax_discount_nan_id] = np.zeros(
                    len(tax_discount_nan_id)
                )

            tax_discount_array = np.asarray(raw_tax_discount, dtype=np.float64)
            if out_of_range_indices(tax_discount_array, 0.0, 1.0):
                raise ASRInputError(
                    "The value of tax_discount must be: 0 < tax_discount < 1, "
                    + f"tax_discount: {tax_discount_array}"
                )

        self.tax_discount = np.asarray(self.tax_discount, dtype=np.float64)

        # Prepare attribute future_rate
        if self.future_rate is None:
            self.future_rate = np.zeros_like(self.cost, dtype=np.float64)

        else:
            raw_future_rate = _require_asr_future_rate(self.future_rate)

            if isinstance(raw_future_rate, float):
                if out_of_range_indices((raw_future_rate,), 0.0, 1.0):
                    raise ASRInputError("Attribute future_rate must be between 0 and 1")
                else:
                    self.future_rate = np.repeat(raw_future_rate, len(self.cost))

            elif isinstance(raw_future_rate, np.ndarray):
                future_rate_nan_id = _missing_indices(raw_future_rate)
                if len(future_rate_nan_id) > 0:
                    raw_future_rate[future_rate_nan_id] = np.zeros(
                        len(future_rate_nan_id)
                    )

                future_rate_array = np.asarray(raw_future_rate, dtype=np.float64)
                if out_of_range_indices(future_rate_array, 0.0, 1.0):
                    raise ASRInputError(
                        "The value of future_rate must fall within the following "
                        + f"interval: 0 < future_rate < 1 (future_rate: {self.future_rate})"
                    )

        self.future_rate = np.asarray(self.future_rate, dtype=np.float64)

        # Prepare attribute final_year
        if self.final_year is None:
            self.final_year = self.expense_year.copy()

        else:
            final_year_array = _require_asr_array(self.final_year, "final_year")
            final_year_numeric = np.asarray(final_year_array, dtype=np.float64)
            final_year_nan = _missing_indices(final_year_numeric)
            if len(final_year_nan) > 0:
                final_year_numeric[final_year_nan] = self.expense_year[
                    final_year_nan
                ].copy()

            # Exception: final_year is before expense_year
            if any_below(final_year_numeric, self.expense_year):
                raise ASRInputError(
                    f"Attribute final_year ({final_year_numeric}) is before "
                    + f"the expense_year ({self.expense_year})"
                )

            # Exception: final_year is after the end year of the project
            above = year_above(final_year_numeric, self.end_year)
            if above is not None:
                raise ASRInputError(
                    f"Final year ({above}) "
                    + f"is after the end year of the project ({self.end_year})"
                )

        self.final_year = self.final_year.astype(np.int64)

        # Check input data for unequal length
        expense_year = _require_present(self.expense_year, ASRException, "expense_year")
        cost = _require_present(self.cost, ASRException, "cost")
        cost_allocation = _require_present(
            self.cost_allocation, ASRException, "cost_allocation"
        )
        cost_type = _require_present(self.cost_type, ASRException, "cost_type")
        description = _require_present(self.description, ASRException, "description")
        tax_portion = _require_present(self.tax_portion, ASRException, "tax_portion")
        tax_discount = _require_present(self.tax_discount, ASRException, "tax_discount")
        final_year = _require_present(self.final_year, ASRException, "final_year")
        future_rate = _require_present(self.future_rate, ASRException, "future_rate")
        length_map = {
            "expense_year": len(expense_year),
            "cost": len(cost),
            "cost_allocation": len(cost_allocation),
            "cost_type": len(cost_type),
            "description": len(description),
            "tax_portion": len(tax_portion),
            "tax_discount": len(tax_discount),
            "final_year": len(final_year),
            "future_rate": len(future_rate),
        }

        if unequal_lengths(length_map) is not None:
            raise ASRInputError(
                "Unequal length of array: "
                + f"cost: {len(cost)}, expense_year: {len(expense_year)}, "
                + f"cost_allocation: {len(cost_allocation)}, cost_type: {len(cost_type)}, "
                + f"description: {len(description)}, final_year: {len(final_year)}, "
                + f"future_rate: {len(future_rate)}, tax_portion: {len(tax_portion)}, "
                + f"tax_discount: {len(tax_discount)}. "
            )

        # Raise an exception: expense year is after the end year of the project
        above = year_above(self.expense_year, self.end_year)
        if above is not None:
            raise ASRInputError(
                f"Expense year ({above}) "
                + f"is after the end year of the project ({self.end_year})"
            )

        # Raise an exception: expense year is before the start year of the project
        below = year_below(self.expense_year, self.start_year)
        if below is not None:
            raise ASRInputError(
                f"Expense year ({below}) "
                + f"is before the start year of the project ({self.start_year})"
            )

    def _calc_future_cost(self) -> npt.NDArray[np.float64]:
        """
        Calculate the future cost of each element by applying a growth rate
        over the project's timeline.

        This method computes the future cost of each element based on the time difference
        between the project's end year and the expense year, adjusting for a specified
        future rate.

        Returns
        -------
        np.ndarray
            A 1D array of future costs for each element, adjusted by the future growth rate
            over the time difference between the expense year and the project end year.

        Notes
        -----
        The future cost is calculated by multiplying the current cost by
        `(1 + future_rate) ** year_diff`, where `year_diff` is the number
        of years between the `expense_year` and the `end_year`.
        """

        expense_year, cost, _tax_discount = self._normalized_cost_arrays(ASRException)
        _final_year, future_rate = self._normalized_asr_fields()

        # Distance between the end year of the project with expense year
        year_diff = self.end_year - expense_year + 1

        # Calculate future value of cost
        return cost * np.power((1.0 + future_rate), year_diff)

    @override
    def expenditures_pre_tax(
        self,
        year_inflation: npt.NDArray[np.int64] | None = None,
        inflation_rate: npt.NDArray[np.float64] | int | float = 0.0,
    ) -> npt.NDArray[np.float64]:
        """
        Calculate the pre-tax expenditures by adjusting future costs for inflation and
        distributing them over the project duration.

        This method first calculates the future cost of each element, adjusts these costs
        for inflation, and then distributes the adjusted costs over the project duration.
        The total pre-tax expenditures are summed for each project year.

        Parameters
        ----------
        year_inflation : np.ndarray, optional
            A NumPy array representing the years during which inflation is applied to the costs.
            If not provided, defaults to repeating the `start_year` for all costs.
        inflation_rate : np.ndarray or float, optional
            The inflation rate(s) to apply to the project costs. If provided as a float,
            a uniform inflation rate is applied. If provided as a NumPy array, different
            rates are applied based on the corresponding project years. Default is 0.0.

        Returns
        -------
        np.ndarray
            A 1D array representing the total pre-tax expenditures for each project year
            after adjusting for inflation and distributing the future costs.

        Notes
        -----
        -   The method first calculates the future cost using the `_calc_future_cost` method.
        -   It adjusts the future cost for inflation using `get_cost_adjustment_by_inflation`,
            which accounts for inflation rates across the project's duration.
        -   Finally, it distributes the adjusted cost using `calc_distributed_cost`, summing
            the distributed costs for each year to provide the total pre-tax expenditures.
        """

        # Calculate future cost
        (
            project_duration,
            project_years,
            _cost_allocation,
            _cost_type,
            _description,
            _tax_portion,
        ) = self._normalized_general_fields(ASRException)
        expense_year, _cost, _tax_discount = self._normalized_cost_arrays(ASRException)
        final_year, _future_rate = self._normalized_asr_fields()
        cost_future = self._calc_future_cost()

        # Cost adjustment due to inflation
        cost_adjusted_by_inflation = get_cost_adjustment_by_inflation(
            start_year=self.start_year,
            end_year=self.end_year,
            cost=cost_future,
            expense_year=expense_year,
            project_years=project_years,
            year_inflation=year_inflation,
            inflation_rate=inflation_rate,
        )

        # Calculate distributed _cost
        distributed_cost: npt.NDArray[np.float64] = np.asarray(
            calc_distributed_cost(
                cost=cost_adjusted_by_inflation,
                expense_year=expense_year,
                final_year=final_year,
                project_years=project_years,
                project_duration=project_duration,
            ),
            dtype=np.float64,
        )

        return np.asarray(distributed_cost.sum(axis=1), dtype=np.float64)

    @override
    def indirect_taxes(
        self, tax_rate: npt.NDArray[np.float64] | float = 0.0
    ) -> npt.NDArray[np.float64]:
        """
        Calculate and distribute indirect taxes over the project duration.

        This method calculates indirect taxes based on future costs, tax portions, tax rates,
        and any applicable tax discounts. The taxes are then distributed across the project's
        timeline, and the total indirect taxes are summed for each project year.

        Parameters
        ----------
        tax_rate : np.ndarray or float, optional
            A NumPy array or float representing the tax rate applied to the costs. If not provided,
            a default rate of 0.0 will be used. When provided as an array, it should match
            the project years.

        Returns
        -------
        np.ndarray
            A 1D array representing the total distributed indirect taxes for each project year.

        Notes
        -----
        -   The method first calculates the future cost using the `_calc_future_cost` method.
        -   It calculates the indirect tax using `calc_indirect_tax`, which applies the
            tax portion, tax rate, and tax discount.
        -   The method then distributes the calculated taxes across the project's timeline
            using `calc_distributed_cost`.
        -   The final result is the sum of the distributed taxes for each project year.
        """

        # Calculate future cost
        (
            project_duration,
            project_years,
            _cost_allocation,
            _cost_type,
            _description,
            tax_portion,
        ) = self._normalized_general_fields(ASRException)
        expense_year, _cost, tax_discount = self._normalized_cost_arrays(ASRException)
        final_year, _future_rate = self._normalized_asr_fields()
        cost_future = self._calc_future_cost()

        # Calculate indirect tax
        indirect_tax = calc_indirect_tax(
            start_year=self.start_year,
            cost=cost_future,
            expense_year=expense_year,
            project_years=project_years,
            tax_portion=tax_portion,
            tax_rate=tax_rate,
            tax_discount=tax_discount,
        )

        # Calculate distributed indirect taxes
        distributed_taxes: npt.NDArray[np.float64] = np.asarray(
            calc_distributed_cost(
                cost=indirect_tax,
                expense_year=expense_year,
                final_year=final_year,
                project_years=project_years,
                project_duration=project_duration,
            ),
            dtype=np.float64,
        )

        return np.asarray(distributed_taxes.sum(axis=1), dtype=np.float64)

    @override
    def __eq__(self, other: object) -> bool:
        # Between two instances of ASR
        (
            _project_duration,
            _project_years,
            cost_allocation,
            cost_type,
            _description,
            tax_portion,
        ) = self._normalized_general_fields(ASRException)
        expense_year, cost, tax_discount = self._normalized_cost_arrays(ASRException)
        final_year, _future_rate = self._normalized_asr_fields()
        final_year, _future_rate = self._normalized_asr_fields()
        if isinstance(other, ASR):
            (
                _other_project_duration,
                _other_project_years,
                other_cost_allocation,
                other_cost_type,
                _other_description,
                other_tax_portion,
            ) = other._normalized_general_fields(ASRException)
            other_expense_year, other_cost, other_tax_discount = (
                other._normalized_cost_arrays(ASRException)
            )
            other_final_year, other_future_rate = other._normalized_asr_fields()
            return all(
                (
                    self.start_year == other.start_year,
                    self.end_year == other.end_year,
                    np.allclose(expense_year, other_expense_year),
                    np.allclose(cost, other_cost),
                    np.allclose(tax_portion, other_tax_portion),
                    np.allclose(tax_discount, other_tax_discount),
                    np.allclose(final_year, other_final_year),
                    np.allclose(_future_rate, other_future_rate),
                    cost_allocation == other_cost_allocation,
                    cost_type == other_cost_type,
                )
            )

        # Between an instance of ASR and integer/float
        elif isinstance(other, (int, float)):
            return float(np.sum(cost)) == other

        else:
            return False

    def __lt__(self, other: object) -> bool:
        # Between an instance of ASR with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales
        (
            _project_duration,
            _project_years,
            _cost_allocation,
            _cost_type,
            _description,
            _tax_portion,
        ) = self._normalized_general_fields(ASRException)
        _expense_year, cost, _tax_discount = self._normalized_cost_arrays(ASRException)
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales)):
            (
                _other_project_duration,
                _other_project_years,
                _other_cost_allocation,
                _other_cost_type,
                _other_description,
                _other_tax_portion,
            ) = other._normalized_general_fields(ASRException)
            _other_expense_year, other_cost, _other_tax_discount = (
                other._normalized_cost_arrays(ASRException)
            )
            return bool(np.sum(cost) < np.sum(other_cost))

        # Between an instance of ASR and an integer/float
        elif isinstance(other, (int, float)):
            return bool(np.sum(cost) < other)

        else:
            raise ASRException(
                "Must compare an instance of ASR with another instance of "
                + "CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales, an integer or a float. "
            )

    def __le__(self, other: object) -> bool:
        # Between an instance of ASR with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales
        (
            _project_duration,
            _project_years,
            _cost_allocation,
            _cost_type,
            _description,
            _tax_portion,
        ) = self._normalized_general_fields(ASRException)
        _expense_year, cost, _tax_discount = self._normalized_cost_arrays(ASRException)
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales)):
            (
                _other_project_duration,
                _other_project_years,
                _other_cost_allocation,
                _other_cost_type,
                _other_description,
                _other_tax_portion,
            ) = other._normalized_general_fields(ASRException)
            _other_expense_year, other_cost, _other_tax_discount = (
                other._normalized_cost_arrays(ASRException)
            )
            return bool(np.sum(cost) <= np.sum(other_cost))

        # Between an instance of ASR and an integer/float
        elif isinstance(other, (int, float)):
            return bool(np.sum(cost) <= other)

        else:
            raise ASRException(
                "Must compare an instance of ASR with another instance of "
                + "CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales, an integer or a float. "
            )

    def __gt__(self, other: object) -> bool:
        # Between an instance of ASR with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales
        (
            _project_duration,
            _project_years,
            _cost_allocation,
            _cost_type,
            _description,
            _tax_portion,
        ) = self._normalized_general_fields(ASRException)
        _expense_year, cost, _tax_discount = self._normalized_cost_arrays(ASRException)
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales)):
            (
                _other_project_duration,
                _other_project_years,
                _other_cost_allocation,
                _other_cost_type,
                _other_description,
                _other_tax_portion,
            ) = other._normalized_general_fields(ASRException)
            _other_expense_year, other_cost, _other_tax_discount = (
                other._normalized_cost_arrays(ASRException)
            )
            return bool(np.sum(cost) > np.sum(other_cost))

        # Between an instance of ASR and an integer/float
        elif isinstance(other, (int, float)):
            return bool(np.sum(cost) > other)

        else:
            raise ASRException(
                "Must compare an instance of ASR with another instance of "
                + "CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales, an integer, or a float."
            )

    def __ge__(self, other: object) -> bool:
        # Between an instance of ASR with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales
        (
            _project_duration,
            _project_years,
            _cost_allocation,
            _cost_type,
            _description,
            _tax_portion,
        ) = self._normalized_general_fields(ASRException)
        _expense_year, cost, _tax_discount = self._normalized_cost_arrays(ASRException)
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales)):
            (
                _other_project_duration,
                _other_project_years,
                _other_cost_allocation,
                _other_cost_type,
                _other_description,
                _other_tax_portion,
            ) = other._normalized_general_fields(ASRException)
            _other_expense_year, other_cost, _other_tax_discount = (
                other._normalized_cost_arrays(ASRException)
            )
            return bool(np.sum(cost) >= np.sum(other_cost))

        # Between an instance of ASR and an integer/float
        elif isinstance(other, (int, float)):
            return bool(np.sum(cost) >= other)

        else:
            raise ASRException(
                "Must compare an instance of ASR with another instance of "
                + "CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales, an integer, or a float."
            )

    def __add__(self, other: object) -> ASR:
        # Only allows addition between an instance of ASR
        # and another instance of ASR
        (
            _project_duration,
            _project_years,
            cost_allocation,
            cost_type,
            description,
            tax_portion,
        ) = self._normalized_general_fields(ASRException)
        expense_year, cost, tax_discount = self._normalized_cost_arrays(ASRException)
        final_year, _future_rate = self._normalized_asr_fields()
        if isinstance(other, ASR):
            (
                _other_project_duration,
                _other_project_years,
                other_cost_allocation,
                other_cost_type,
                other_description,
                other_tax_portion,
            ) = other._normalized_general_fields(ASRException)
            other_expense_year, other_cost, other_tax_discount = (
                other._normalized_cost_arrays(ASRException)
            )
            _other_final_year, other_future_rate = other._normalized_asr_fields()
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            expense_year_combined = np.concatenate((expense_year, other_expense_year))
            cost_combined = np.concatenate((cost, other_cost))
            tax_portion_combined = np.concatenate((tax_portion, other_tax_portion))
            tax_discount_combined = np.concatenate((tax_discount, other_tax_discount))
            final_year_combined = np.concatenate((final_year, _other_final_year))
            future_rate_combined = np.concatenate((_future_rate, other_future_rate))
            cost_allocation_combined = cost_allocation + other_cost_allocation
            cost_type_combined = cost_type + other_cost_type
            description_combined = description + other_description

            return ASR(
                start_year=start_year_combined,
                end_year=end_year_combined,
                expense_year=expense_year_combined,
                cost=cost_combined,
                cost_allocation=cost_allocation_combined,
                cost_type=cost_type_combined,
                description=description_combined,
                tax_portion=tax_portion_combined,
                tax_discount=tax_discount_combined,
                final_year=final_year_combined,
                future_rate=future_rate_combined,
            )

        else:
            raise ASRException(
                "Must add an instance of ASR with another instance of ASR. "
                + f"{other}({other.__class__.__qualname__}) is not an instance of ASR."
            )

    def __iadd__(self, other: object) -> ASR:
        return self.__add__(other)

    def __sub__(self, other: object) -> ASR:
        # Only allows subtraction between an instance of ASR
        # and another instance of ASR
        (
            _project_duration,
            _project_years,
            cost_allocation,
            cost_type,
            description,
            tax_portion,
        ) = self._normalized_general_fields(ASRException)
        expense_year, cost, tax_discount = self._normalized_cost_arrays(ASRException)
        final_year, _future_rate = self._normalized_asr_fields()
        if isinstance(other, ASR):
            (
                _other_project_duration,
                _other_project_years,
                other_cost_allocation,
                other_cost_type,
                other_description,
                other_tax_portion,
            ) = other._normalized_general_fields(ASRException)
            other_expense_year, other_cost, other_tax_discount = (
                other._normalized_cost_arrays(ASRException)
            )
            _other_final_year, other_future_rate = other._normalized_asr_fields()
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            expense_year_combined = np.concatenate((expense_year, other_expense_year))
            cost_combined = np.concatenate((cost, -other_cost))
            tax_portion_combined = np.concatenate((tax_portion, other_tax_portion))
            tax_discount_combined = np.concatenate((tax_discount, other_tax_discount))
            final_year_combined = np.concatenate((final_year, _other_final_year))
            future_rate_combined = np.concatenate((_future_rate, other_future_rate))
            cost_allocation_combined = cost_allocation + other_cost_allocation
            cost_type_combined = cost_type + other_cost_type
            description_combined = description + other_description

            return ASR(
                start_year=start_year_combined,
                end_year=end_year_combined,
                expense_year=expense_year_combined,
                cost=cost_combined,
                cost_allocation=cost_allocation_combined,
                cost_type=cost_type_combined,
                description=description_combined,
                tax_portion=tax_portion_combined,
                tax_discount=tax_discount_combined,
                final_year=final_year_combined,
                future_rate=future_rate_combined,
            )

        else:
            raise ASRException(
                "Must subtract between an instance of ASR "
                + "with another instance of ASR. "
                + f"{other}({other.__class__.__qualname__}) is not "
                + "an instance of ASR."
            )

    def __rsub__(self, other: object) -> ASR:
        return self.__sub__(other)

    def __mul__(self, other: object) -> ASR:
        # Multiplication is allowed only with an integer/a float
        (
            _project_duration,
            _project_years,
            cost_allocation,
            cost_type,
            description,
            tax_portion,
        ) = self._normalized_general_fields(ASRException)
        expense_year, cost, tax_discount = self._normalized_cost_arrays(ASRException)
        final_year, future_rate = self._normalized_asr_fields()
        if isinstance(other, (int, float)):
            return ASR(
                start_year=self.start_year,
                end_year=self.end_year,
                expense_year=expense_year,
                cost=cost * other,
                cost_allocation=cost_allocation,
                cost_type=cost_type,
                description=description,
                tax_portion=tax_portion,
                tax_discount=tax_discount,
                final_year=final_year,
                future_rate=future_rate,
            )

        else:
            raise ASRException(
                "Must multiply with an integer or a float. "
                + f"{other}({other.__class__.__qualname__}) "
                + "is not an integer nor a float."
            )

    def __rmul__(self, other: object) -> ASR:
        return self.__mul__(other)

    def __truediv__(self, other: object) -> ASR | float:
        # Between an instance of ASR with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales
        (
            _project_duration,
            _project_years,
            cost_allocation,
            cost_type,
            description,
            tax_portion,
        ) = self._normalized_general_fields(ASRException)
        expense_year, cost, tax_discount = self._normalized_cost_arrays(ASRException)
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales)):
            (
                _other_project_duration,
                _other_project_years,
                _other_cost_allocation,
                _other_cost_type,
                _other_description,
                _other_tax_portion,
            ) = other._normalized_general_fields(ASRException)
            _other_expense_year, other_cost, _other_tax_discount = (
                other._normalized_cost_arrays(ASRException)
            )
            return float(np.sum(cost) / np.sum(other_cost))

        # Between an instance of ASR and an integer/a float
        elif isinstance(other, (int, float)):
            if other == 0:
                raise ASRException("Cannot divide with zero")

            else:
                return ASR(
                    start_year=self.start_year,
                    end_year=self.end_year,
                    expense_year=expense_year,
                    cost=cost / other,
                    cost_allocation=cost_allocation,
                    cost_type=cost_type,
                    description=description,
                    tax_portion=tax_portion,
                    tax_discount=tax_discount,
                    final_year=self.final_year,
                    future_rate=self.future_rate,
                )

        else:
            raise ASRException(
                "Must divide with an instance of CapitalCost/Intangible/OPEX/ASR/LBT"
                + f"/CostOfSales, integer or a float. {other}({other.__class__.__qualname__}) "
                + "is not an instance of CapitalCost/Intangible/OPEX/ASR/LBT not an integer "
                + "nor a float."
            )


@dataclass
@final
class LBT(GeneralCost):
    """
    Manages LBT asset.

    Parameters
    ----------
    final_year : np.ndarray
        The year in which cost distribution ends for each cost element.
    utilized_land_area : np.ndarray
        The utilized land area.
    utilized_building_area : np.ndarray
        The utilized building area.
    njop_land : np.ndarray
        NJOP related to land.
    njop_building : np.ndarray
        NJOP related to building.
    gross_revenue : np.ndarray
        The array of gross revenues.
    """

    final_year: npt.NDArray[np.generic] | None = None
    utilized_land_area: npt.NDArray[np.generic] | None = None
    utilized_building_area: npt.NDArray[np.generic] | None = None
    njop_land: npt.NDArray[np.generic] | None = None
    njop_building: npt.NDArray[np.generic] | None = None
    gross_revenue: npt.NDArray[np.generic] | None = None

    # Override attribute

    # Attributes to be defined later
    # (associated with surface and subsurface LBT components)
    _surface_lbt_cost: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _subsurface_lbt_cost: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )

    def _normalized_lbt_fields(
        self,
    ) -> tuple[npt.NDArray[np.generic], ...]:
        """Return LBT arrays after constructor normalization."""
        values = (
            self.final_year,
            self.utilized_land_area,
            self.utilized_building_area,
            self.njop_land,
            self.njop_building,
            self.gross_revenue,
        )
        arrays: list[npt.NDArray[np.generic]] = []
        for value, name in zip(
            values,
            (
                "final_year",
                "utilized_land_area",
                "utilized_building_area",
                "njop_land",
                "njop_building",
                "gross_revenue",
            ),
        ):
            if not _is_ndarray(value):
                raise LBTException(f"Attribute {name} is not initialized as an array")
            arrays.append(value)
        return tuple(arrays)

    def __post_init__(self):
        """
        Handles the following operations/procedures:
        -   Prepare attributes project_duration and project_years,
        -   Prepare attribute expense_year,
        -   Prepare attribute cost_allocation,
        -   Prepare attribute cost_type,
        -   Prepare attribute description,
        -   Prepare attribute tax_portion,
        -   Prepare attribute tax_discount,
        -   Prepare attribute final_year,
        -   Prepare attribute utilized_land_area,
        -   Prepare attribute utilized_building_area,
        -   Prepare attribute njop_land,
        -   Prepare attribute njop_building,
        -   Prepare attribute gross_revenue,
        -   Prepare attribute _surface_lbt_cost,
        -   Prepare attribute _subsurface_lbt_cost,
        -   Prepare attribute cost,
        -   Check for unequal length of array data,
        -   Raise an exception: expense_year is after the end_year of the project,
        -   Raise an exception: expense_year is before the start_year of the project,
        """

        # Prepare attributes project_duration and project_years
        if not period_is_reversed(self.start_year, self.end_year):
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise LBTInputError(
                f"start year {self.start_year} is after the end year {self.end_year}"
            )

        # Prepare attribute expense_year
        if not isinstance(self.expense_year, np.ndarray):
            raise LBTException(
                "Attribute expense_year must be provided as a numpy.ndarray, "
                + f"not as a/an {self.expense_year.__class__.__qualname__}"
            )

        else:
            if missing_value_indices(self.expense_year):
                raise LBTInputError(
                    f"Missing values in array expense_year: {self.expense_year}"
                )

        self.expense_year = self.expense_year.astype(np.int64)

        # Prepare attribute cost_allocation
        if self.cost_allocation is None:
            self.cost_allocation = [
                FluidType.OIL for _ in range(len(self.expense_year))
            ]

        else:
            raw_cost_allocation: object = self.cost_allocation
            if not _is_object_list(raw_cost_allocation):
                raise LBTException(
                    "Attribute cost_allocation must be given as a list, "
                    + f"not as a/an {self.cost_allocation.__class__.__qualname__}"
                )

            normalized_cost_allocation: list[FluidType] = []
            for val in raw_cost_allocation:
                if _is_missing(val):
                    normalized_cost_allocation.append(FluidType.OIL)
                elif isinstance(val, FluidType):
                    normalized_cost_allocation.append(val)
                else:
                    raise LBTException("Must insert FluidType in list cost_allocation")
            self.cost_allocation = normalized_cost_allocation

        # Prepare attribute cost_type
        raw_cost_type: object = self.cost_type
        if raw_cost_type is None:
            self.cost_type = [None for _ in range(len(self.expense_year))]

        else:
            if not _is_object_list(raw_cost_type):
                raise LBTException(
                    "Attribute cost_type must be given as a list, not "
                    + f"as a/an {self.cost_type.__class__.__qualname__}"
                )

            self.cost_type = _normalize_cost_type(raw_cost_type, LBTException)

        # Prepare attribute description
        if self.description is None:
            self.description = [" " for _ in range(len(self.expense_year))]

        else:
            raw_description: object = self.description
            if not _is_object_list(raw_description):
                raise LBTException(
                    "Attribute description must be given as a list, "
                    + f"not as a/an {self.description.__class__.__qualname__}"
                )

            self.description = [
                " " if _is_missing(val) else val
                for _, val in enumerate(self.description)
            ]

        # Prepare attribute tax_portion
        if self.tax_portion is None:
            self.tax_portion = np.zeros_like(self.expense_year, dtype=np.float64)

        else:
            raw_tax_portion: object = self.tax_portion
            if not _is_ndarray(raw_tax_portion):
                raise LBTException(
                    "Attribute tax_portion must be given as a numpy.ndarray, "
                    + f"not as a/an {self.tax_portion.__class__.__qualname__}"
                )

            tax_portion_nan_id = _missing_indices(self.tax_portion)
            if len(tax_portion_nan_id) > 0:
                self.tax_portion[tax_portion_nan_id] = np.zeros(len(tax_portion_nan_id))

            if out_of_range_indices(self.tax_portion, 0.0, 1.0):
                raise LBTInputError(
                    "The value of tax_portion must be: 0 < tax_portion < 1, "
                    + f"tax_portion: {self.tax_portion}"
                )

        self.tax_portion = self.tax_portion.astype(np.float64)

        # Prepare attribute tax_discount
        raw_tax_discount: object = self.tax_discount
        if not (
            isinstance(raw_tax_discount, float)
            or isinstance(raw_tax_discount, int)
            or _is_ndarray(raw_tax_discount)
        ):
            raise LBTException(
                "Attribute tax_discount must be provided as a float/int or as a numpy.ndarray, "
                + f"not as a/an {self.tax_discount.__class__.__qualname__}"
            )

        if isinstance(raw_tax_discount, (float, int)):
            if out_of_range_indices((raw_tax_discount,), 0.0, 1.0):
                raise LBTInputError("Attribute tax_discount must be between 0 and 1")

            self.tax_discount = np.repeat(self.tax_discount, len(self.expense_year))

        else:
            tax_discount_array = np.asarray(raw_tax_discount, dtype=np.float64)
            tax_discount_nan_id = _missing_indices(tax_discount_array)
            if len(tax_discount_nan_id) > 0:
                tax_discount_array[tax_discount_nan_id] = np.zeros(
                    len(tax_discount_nan_id)
                )

            if out_of_range_indices(tax_discount_array, 0.0, 1.0):
                raise LBTInputError(
                    "The value of tax_discount must be: 0 < tax_discount < 1, "
                    + f"tax_discount: {tax_discount_array}"
                )

        self.tax_discount = np.asarray(self.tax_discount, dtype=np.float64)

        # Prepare attribute final_year
        if self.final_year is None:
            self.final_year = self.expense_year.copy()

        else:
            raw_final_year: object = self.final_year
            if not _is_ndarray(raw_final_year):
                raise LBTException(
                    "Attribute final_year must be given as a numpy.ndarray, "
                    + f"not as a/an {self.final_year.__class__.__qualname__}"
                )

            final_year_nan_id = _missing_indices(self.final_year)
            if len(final_year_nan_id) > 0:
                self.final_year[final_year_nan_id] = self.expense_year[
                    final_year_nan_id
                ].copy()

            # Exception: final_year is before expense_year
            if any_below(
                np.asarray(self.final_year, dtype=np.int64), self.expense_year
            ):
                raise LBTInputError(
                    f"Attribute final_year ({self.final_year}) is before "
                    + f"the expense_year ({self.expense_year})"
                )

            # Exception: final_year is after the end year of the project
            above = year_above(np.asarray(self.final_year, dtype=np.int64), self.end_year)
            if above is not None:
                raise LBTInputError(
                    f"Final year ({above}) "
                    + f"is after the end year of the project ({self.end_year})"
                )

        self.final_year = self.final_year.astype(np.int64)

        # Prepare attribute utilized_land_area
        if self.utilized_land_area is None:
            self.utilized_land_area = np.zeros_like(self.expense_year, dtype=np.float64)

        else:
            raw_land: object = self.utilized_land_area
            if not _is_ndarray(raw_land):
                raise LBTException(
                    "Attribute utilized_land_area must be given as a numpy.ndarray, "
                    + f"not as a/an {self.utilized_land_area.__class__.__qualname__}"
                )

            # Missing data in array utilized_land_area
            utilized_land_nan_id = _missing_indices(self.utilized_land_area)
            if len(utilized_land_nan_id) > 0:
                self.utilized_land_area[utilized_land_nan_id] = np.zeros(
                    len(utilized_land_nan_id)
                )

            # Exception: negative values in array utilized_land_area
            if out_of_range_indices(self.utilized_land_area, 0.0, float("inf")):
                raise LBTInputError(
                    f"Negative values in array utilized_land_area: {self.utilized_land_area}"
                )

        self.utilized_land_area = self.utilized_land_area.astype(np.float64)

        # Prepare attribute utilized_building_area
        if self.utilized_building_area is None:
            self.utilized_building_area = np.zeros_like(
                self.expense_year, dtype=np.float64
            )

        else:
            raw_building: object = self.utilized_building_area
            if not _is_ndarray(raw_building):
                raise LBTException(
                    "Attribute utilized_building_area must be given as a numpy.ndarray, "
                    + f"not as an/a {self.utilized_building_area.__class__.__qualname__}"
                )

            # Missing data in array utilized_building_area
            utilized_building_nan_id = _missing_indices(self.utilized_building_area)
            if len(utilized_building_nan_id) > 0:
                self.utilized_building_area[utilized_building_nan_id] = np.zeros(
                    len(utilized_building_nan_id)
                )

            # Exception: negative values in array utilized_building_area
            if out_of_range_indices(self.utilized_building_area, 0.0, float("inf")):
                raise LBTInputError(
                    f"Negative values in array utilized_building_area: {self.utilized_building_area}"
                )

        self.utilized_building_area = self.utilized_building_area.astype(np.float64)

        # Prepare attribute njop_land
        if self.njop_land is None:
            self.njop_land = np.zeros_like(self.expense_year, dtype=np.float64)

        else:
            raw_njop_land: object = self.njop_land
            if not _is_ndarray(raw_njop_land):
                raise LBTException(
                    "Attribute njop_land must be given as a numpy.ndarray, "
                    + f"not as a/an {self.njop_land.__class__.__qualname__}"
                )

            # Missing data of array njop_land
            njop_land_nan_id = _missing_indices(self.njop_land)
            if len(njop_land_nan_id) > 0:
                self.njop_land[njop_land_nan_id] = np.zeros(len(njop_land_nan_id))

        self.njop_land = self.njop_land.astype(np.float64)

        # Prepare attribute njop_building
        if self.njop_building is None:
            self.njop_building = np.zeros_like(self.expense_year, dtype=np.float64)

        else:
            raw_njop_building: object = self.njop_building
            if not _is_ndarray(raw_njop_building):
                raise LBTException(
                    "Attribute njop_building must be given as a numpy.ndarray, "
                    + f"not as a/an {self.njop_building.__class__.__qualname__}"
                )

            # Missing data in array njop_building
            njop_building_nan_id = _missing_indices(self.njop_building)
            if len(njop_building_nan_id) > 0:
                self.njop_building[njop_building_nan_id] = np.zeros(
                    len(njop_building_nan_id)
                )

        self.njop_building = self.njop_building.astype(np.float64)

        # Prepare attribute gross_revenue
        if self.gross_revenue is None:
            self.gross_revenue = np.zeros_like(self.expense_year, dtype=np.float64)

        else:
            raw_revenue: object = self.gross_revenue
            if not _is_ndarray(raw_revenue):
                raise LBTException(
                    "Attribute gross_revenue must be given as a numpy.ndarray, "
                    + f"not as a/an {self.gross_revenue.__class__.__qualname__}"
                )

            # Missing data in array gross_revenue
            gross_revenue_nan_id = _missing_indices(self.gross_revenue)
            if len(gross_revenue_nan_id) > 0:
                self.gross_revenue[gross_revenue_nan_id] = np.zeros(
                    len(gross_revenue_nan_id)
                )

        self.gross_revenue = self.gross_revenue.astype(np.float64)

        # Prepare attribute _surface_lbt_cost
        surface_land = self.utilized_land_area * self.njop_land
        surface_building = self.utilized_building_area * self.njop_building
        self._surface_lbt_cost = (
            (0.5 / 100) * (40 / 100) * 1.0 * (surface_land + surface_building)
        )

        # Prepare attribute _subsurface_lbt_cost
        self._subsurface_lbt_cost = (
            (0.5 / 100) * (40 / 100) * 10.04 * self.gross_revenue
        )

        # Prepare attribute cost
        if self.cost is None:
            self.cost = self._surface_lbt_cost + self._subsurface_lbt_cost

        else:
            raw_cost: object = self.cost
            if not _is_ndarray(raw_cost):
                raise LBTException(
                    "Attribute cost must be given as a numpy.ndarray, "
                    + f"not as a/an {self.cost.__class__.__qualname__}"
                )

            else:
                cost_nan_id = _missing_indices(self.cost)
                if len(cost_nan_id) > 0:
                    self.cost[cost_nan_id] = np.zeros(len(cost_nan_id))

        self.cost = self.cost.astype(np.float64)

        # Check for unequal length of array data
        arr_reference = len(self.expense_year)
        normalized_lbt = self._normalized_lbt_fields()
        (
            _,
            _,
            normalized_cost_allocation,
            normalized_cost_type,
            normalized_description,
            normalized_tax_portion,
        ) = self._normalized_general_fields(LBTException)

        length_map = {
            "expense_year": arr_reference,
            "cost_allocation": len(normalized_cost_allocation),
            "cost_type": len(normalized_cost_type),
            "description": len(normalized_description),
            "tax_portion": len(normalized_tax_portion),
            "tax_discount": len(np.asarray(self.tax_discount)),
            "final_year": len(normalized_lbt[0]),
            "utilized_land_area": len(self.utilized_land_area),
            "utilized_building_area": len(self.utilized_building_area),
            "njop_land": len(self.njop_land),
            "njop_building": len(self.njop_building),
            "gross_revenue": len(self.gross_revenue),
            "cost": len(_require_present(self.cost, LBTException, "cost")),
        }

        if unequal_lengths(length_map) is not None:
            raise LBTInputError(
                "Unequal length of arrays: "
                + f"expense_year: {len(self.expense_year)}, "
                + f"cost_allocation: {len(normalized_cost_allocation)}, "
                + f"cost_type: {len(normalized_cost_type)}, "
                + f"description: {len(normalized_description)}, "
                + f"tax_portion: {len(normalized_tax_portion)}, "
                + f"tax_discount: {len(np.asarray(self.tax_discount))}, "
                + f"final_year: {len(normalized_lbt[0])}, "
                + f"utilized_land_area: {len(self.utilized_land_area)}, "
                + f"utilized_building_area: {len(self.utilized_building_area)}, "
                + f"njop_land: {len(self.njop_land)}, "
                + f"njop_building: {len(self.njop_building)}, "
                + f"gross_revenue: {len(self.gross_revenue)}, "
                + f"cost: {len(self.cost)}"
            )

        # Exception: expense year is after the end year of the project
        above = year_above(self.expense_year, self.end_year)
        if above is not None:
            raise LBTInputError(
                f"Expense year ({above}) "
                + f"is after the end year of the project ({self.end_year})"
            )

        # Exception: expense_year is before the start year of the project
        below = year_below(self.expense_year, self.start_year)
        if below is not None:
            raise LBTInputError(
                f"Expense year ({below}) "
                + f"is before the start year of the project ({self.start_year})"
            )

    @override
    def expenditures_pre_tax(
        self,
        year_inflation: npt.NDArray[np.int64] | None = None,
        inflation_rate: npt.NDArray[np.float64] | int | float = 0.0,
    ) -> npt.NDArray[np.float64]:
        """
        Calculate pre-tax expenditures adjusted for inflation.

        This method calculates the total pre-tax expenditures for each project year
        by adjusting the cost for inflation and distributing the adjusted costs
        over the project's timeline.

        Parameters
        ----------
        year_inflation : np.ndarray, optional
            A NumPy array representing the years during which inflation is applied to the costs.
            If not provided, defaults to repeating the `start_year` for all costs.
        inflation_rate : np.ndarray or float, optional
            The inflation rate(s) to apply to the project costs. If provided as a float,
            a uniform inflation rate is applied. If provided as a NumPy array, different
            rates are applied based on the corresponding project years. Default is 0.0.

        Returns
        -------
        np.ndarray
            A 1D array representing the total pre-tax expenditures for each project year,
            adjusted for inflation.

        Notes
        -----
        -   The method first adjusts future costs using the `get_cost_adjustment_by_inflation`
            function, which applies inflation rates based on the provided parameters.
        -   It then calculates the distributed costs across the project's timeline using
            `calc_distributed_cost`.
        -   The final result is the sum of the distributed costs for each project year.
        """

        # Cost adjustment due to inflation
        (
            project_duration,
            project_years,
            _cost_allocation,
            _cost_type,
            _description,
            _tax_portion,
        ) = self._normalized_general_fields(LBTException)
        expense_year, cost, _tax_discount = self._normalized_cost_arrays(LBTException)
        final_year = np.asarray(self._normalized_lbt_fields()[0], dtype=np.int64)
        cost_adjusted_by_inflation = get_cost_adjustment_by_inflation(
            start_year=self.start_year,
            end_year=self.end_year,
            cost=cost,
            expense_year=expense_year,
            project_years=project_years,
            year_inflation=year_inflation,
            inflation_rate=inflation_rate,
        )

        # Calculate distributed cost
        distributed_cost: npt.NDArray[np.float64] = np.asarray(
            calc_distributed_cost(
                cost=cost_adjusted_by_inflation,
                expense_year=expense_year,
                final_year=final_year,
                project_years=project_years,
                project_duration=project_duration,
            ),
            dtype=np.float64,
        )

        return np.asarray(distributed_cost.sum(axis=1), dtype=np.float64)

    @override
    def indirect_taxes(
        self, tax_rate: npt.NDArray[np.float64] | float = 0.0
    ) -> npt.NDArray[np.float64]:
        """
        Calculate and distribute indirect taxes over the project duration.

        This method computes indirect taxes by applying the provided tax rates, portions, and
        discounts to the associated costs. The calculated taxes are then distributed over the
        project's timeline, and the total indirect taxes are summed for each project year.

        Parameters
        ----------
        tax_rate : np.ndarray or float, optional
            A NumPy array or float representing the tax rate applied to the costs. If not provided,
            a default rate of 0.0 will be used. When provided as an array, it should match
            the project years.

        Returns
        -------
        np.ndarray
            A 1D array representing the total distributed indirect taxes for each project year.

        Notes
        -----
        -   The indirect tax is first calculated using the `calc_indirect_tax` function, which
            applies the specified tax rate, portion, and discount to the project's costs.
        -   The calculated taxes are then distributed over the project timeline using the
            `calc_distributed_cost` function.
        -   The final result is the sum of distributed indirect taxes for each year in
            the project timeline.
        """

        # Calculate indirect tax
        (
            project_duration,
            project_years,
            _cost_allocation,
            _cost_type,
            _description,
            tax_portion,
        ) = self._normalized_general_fields(LBTException)
        expense_year, cost, tax_discount = self._normalized_cost_arrays(LBTException)
        final_year = np.asarray(self._normalized_lbt_fields()[0], dtype=np.int64)
        indirect_tax = calc_indirect_tax(
            start_year=self.start_year,
            cost=cost,
            expense_year=expense_year,
            project_years=project_years,
            tax_portion=tax_portion,
            tax_rate=tax_rate,
            tax_discount=tax_discount,
        )

        # Calculate distributed indirect taxes
        distributed_taxes: npt.NDArray[np.float64] = np.asarray(
            calc_distributed_cost(
                cost=indirect_tax,
                expense_year=expense_year,
                final_year=final_year,
                project_years=project_years,
                project_duration=project_duration,
            ),
            dtype=np.float64,
        )

        return np.asarray(distributed_taxes.sum(axis=1), dtype=np.float64)

    @override
    def __eq__(self, other: object) -> bool:
        # Between two instances of LBT
        (
            _project_duration,
            _project_years,
            cost_allocation,
            cost_type,
            _description,
            tax_portion,
        ) = self._normalized_general_fields(LBTException)
        expense_year, cost, tax_discount = self._normalized_cost_arrays(LBTException)
        final_year, land, building, njop_land, njop_building, revenue = (
            self._normalized_lbt_fields()
        )
        if isinstance(other, LBT):
            (
                _other_project_duration,
                _other_project_years,
                other_cost_allocation,
                other_cost_type,
                _other_description,
                other_tax_portion,
            ) = other._normalized_general_fields(LBTException)
            other_expense_year, other_cost, other_tax_discount = (
                other._normalized_cost_arrays(LBTException)
            )
            (
                other_final_year,
                other_land,
                other_building,
                other_njop_land,
                other_njop_building,
                other_revenue,
            ) = other._normalized_lbt_fields()
            return all(
                (
                    self.start_year == other.start_year,
                    self.end_year == other.end_year,
                    np.allclose(expense_year, other_expense_year),
                    np.allclose(tax_portion, other_tax_portion),
                    np.allclose(tax_discount, other_tax_discount),
                    np.allclose(final_year, other_final_year),
                    np.allclose(land, other_land),
                    np.allclose(building, other_building),
                    np.allclose(njop_land, other_njop_land),
                    np.allclose(njop_building, other_njop_building),
                    np.allclose(revenue, other_revenue),
                    np.allclose(cost, other_cost),
                    cost_allocation == other_cost_allocation,
                    cost_type == other_cost_type,
                )
            )

        # Between an instance of LBT and an integer/a float
        elif isinstance(other, (int, float)):
            return float(np.sum(cost)) == other

        else:
            return False

    def __lt__(self, other: object) -> bool:
        # Between an instance of LBT with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales
        (
            _project_duration,
            _project_years,
            _cost_allocation,
            _cost_type,
            _description,
            _tax_portion,
        ) = self._normalized_general_fields(LBTException)
        _expense_year, cost, _tax_discount = self._normalized_cost_arrays(LBTException)
        _, _, _, _, _, _ = self._normalized_lbt_fields()
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales)):
            (
                _other_project_duration,
                _other_project_years,
                _other_cost_allocation,
                _other_cost_type,
                _other_description,
                _other_tax_portion,
            ) = other._normalized_general_fields(LBTException)
            _other_expense_year, other_cost, _other_tax_discount = (
                other._normalized_cost_arrays(LBTException)
            )
            return bool(np.sum(cost) < np.sum(other_cost))

        # Between an instance of LBT and an integer/a float
        elif isinstance(other, (int, float)):
            return bool(np.sum(cost) < other)

        else:
            raise LBTException(
                "Must compare an instance of LBT with another instance of "
                + "CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales, an integer, or a float."
            )

    def __le__(self, other: object) -> bool:
        # Between an instance of LBT with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales
        (
            _project_duration,
            _project_years,
            _cost_allocation,
            _cost_type,
            _description,
            _tax_portion,
        ) = self._normalized_general_fields(LBTException)
        _expense_year, cost, _tax_discount = self._normalized_cost_arrays(LBTException)
        _, _, _, _, _, _ = self._normalized_lbt_fields()
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales)):
            (
                _other_project_duration,
                _other_project_years,
                _other_cost_allocation,
                _other_cost_type,
                _other_description,
                _other_tax_portion,
            ) = other._normalized_general_fields(LBTException)
            _other_expense_year, other_cost, _other_tax_discount = (
                other._normalized_cost_arrays(LBTException)
            )
            return bool(np.sum(cost) <= np.sum(other_cost))

        # Between an instance of LBT and an integer/a float
        elif isinstance(other, (int, float)):
            return bool(np.sum(cost) <= other)

        else:
            raise LBTException(
                "Must compare an instance of LBT with another instance of "
                + "CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales, an integer, or a float."
            )

    def __gt__(self, other: object) -> bool:
        # Between an instance of LBT with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales
        (
            _project_duration,
            _project_years,
            _cost_allocation,
            _cost_type,
            _description,
            _tax_portion,
        ) = self._normalized_general_fields(LBTException)
        _expense_year, cost, _tax_discount = self._normalized_cost_arrays(LBTException)
        _, _, _, _, _, _ = self._normalized_lbt_fields()
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales)):
            (
                _other_project_duration,
                _other_project_years,
                _other_cost_allocation,
                _other_cost_type,
                _other_description,
                _other_tax_portion,
            ) = other._normalized_general_fields(LBTException)
            _other_expense_year, other_cost, _other_tax_discount = (
                other._normalized_cost_arrays(LBTException)
            )
            return bool(np.sum(cost) > np.sum(other_cost))

        # Between an instance of ASR and an integer/float
        elif isinstance(other, (int, float)):
            return bool(np.sum(cost) > other)

        else:
            raise LBTException(
                "Must compare an instance of ASR with another instance of "
                + "CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales, an integer, or a float."
            )

    def __ge__(self, other: object) -> bool:
        # Between an instance of LBT with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales
        (
            _project_duration,
            _project_years,
            _cost_allocation,
            _cost_type,
            _description,
            _tax_portion,
        ) = self._normalized_general_fields(LBTException)
        _expense_year, cost, _tax_discount = self._normalized_cost_arrays(LBTException)
        _, _, _, _, _, _ = self._normalized_lbt_fields()
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales)):
            (
                _other_project_duration,
                _other_project_years,
                _other_cost_allocation,
                _other_cost_type,
                _other_description,
                _other_tax_portion,
            ) = other._normalized_general_fields(LBTException)
            _other_expense_year, other_cost, _other_tax_discount = (
                other._normalized_cost_arrays(LBTException)
            )
            return bool(np.sum(cost) >= np.sum(other_cost))

        # Between an instance of LBT and an integer/a float
        elif isinstance(other, (int, float)):
            return bool(np.sum(cost) >= other)

        else:
            raise LBTException(
                "Must compare an instance of LBT with another instance of "
                + "CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales, an integer, or a float."
            )

    def __add__(self, other: object) -> LBT:
        # Only allows addition between an instance of LBT
        # and another instance of LBT
        (
            _project_duration,
            _project_years,
            cost_allocation,
            cost_type,
            description,
            tax_portion,
        ) = self._normalized_general_fields(LBTException)
        expense_year, cost, tax_discount = self._normalized_cost_arrays(LBTException)
        final_year, land, building, njop_land, njop_building, revenue = (
            self._normalized_lbt_fields()
        )
        if isinstance(other, LBT):
            (
                _other_project_duration,
                _other_project_years,
                other_cost_allocation,
                other_cost_type,
                other_description,
                other_tax_portion,
            ) = other._normalized_general_fields(LBTException)
            other_expense_year, other_cost, other_tax_discount = (
                other._normalized_cost_arrays(LBTException)
            )
            (
                other_final_year,
                other_land,
                other_building,
                other_njop_land,
                other_njop_building,
                other_revenue,
            ) = other._normalized_lbt_fields()
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            expense_year_combined = np.concatenate((expense_year, other_expense_year))
            tax_portion_combined = np.concatenate((tax_portion, other_tax_portion))
            tax_discount_combined = np.concatenate((tax_discount, other_tax_discount))
            final_year_combined = np.concatenate((final_year, other_final_year))
            utilized_land_area_combined = np.concatenate((land, other_land))
            utilized_building_area_combined = np.concatenate((building, other_building))
            njop_land_combined = np.concatenate((njop_land, other_njop_land))
            njop_building_combined = np.concatenate(
                (njop_building, other_njop_building)
            )
            gross_revenue_combined = np.concatenate((revenue, other_revenue))
            cost_combined = np.concatenate((cost, other_cost))
            cost_allocation_combined = cost_allocation + other_cost_allocation
            cost_type_combined = cost_type + other_cost_type
            description_combined = description + other_description

            return LBT(
                start_year=start_year_combined,
                end_year=end_year_combined,
                expense_year=expense_year_combined,
                cost_allocation=cost_allocation_combined,
                cost_type=cost_type_combined,
                description=description_combined,
                tax_portion=tax_portion_combined,
                tax_discount=tax_discount_combined,
                final_year=final_year_combined,
                utilized_land_area=utilized_land_area_combined,
                utilized_building_area=utilized_building_area_combined,
                njop_land=njop_land_combined,
                njop_building=njop_building_combined,
                gross_revenue=gross_revenue_combined,
                cost=cost_combined,
            )

        else:
            raise LBTException(
                "Must add between an instance of LBT with another instance of LBT. "
                + f"{other}({other.__class__.__qualname__}) is not an instance of LBT."
            )

    def __iadd__(self, other: object) -> LBT:
        return self.__add__(other)

    def __sub__(self, other: object) -> LBT:
        # Only allows subtraction between an instance of LBT
        # and another instance of LBT
        (
            _project_duration,
            _project_years,
            cost_allocation,
            cost_type,
            description,
            tax_portion,
        ) = self._normalized_general_fields(LBTException)
        expense_year, cost, tax_discount = self._normalized_cost_arrays(LBTException)
        final_year, land, building, njop_land, njop_building, revenue = (
            self._normalized_lbt_fields()
        )
        if isinstance(other, LBT):
            (
                _other_project_duration,
                _other_project_years,
                other_cost_allocation,
                other_cost_type,
                other_description,
                other_tax_portion,
            ) = other._normalized_general_fields(LBTException)
            other_expense_year, other_cost, other_tax_discount = (
                other._normalized_cost_arrays(LBTException)
            )
            (
                other_final_year,
                other_land,
                other_building,
                other_njop_land,
                other_njop_building,
                other_revenue,
            ) = other._normalized_lbt_fields()
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            expense_year_combined = np.concatenate((expense_year, other_expense_year))
            tax_portion_combined = np.concatenate((tax_portion, other_tax_portion))
            tax_discount_combined = np.concatenate((tax_discount, other_tax_discount))
            final_year_combined = np.concatenate((final_year, other_final_year))
            utilized_land_area_combined = np.concatenate((land, other_land))
            utilized_building_area_combined = np.concatenate((building, other_building))
            njop_land_combined = np.concatenate(
                (njop_land, -np.asarray(other_njop_land, dtype=np.float64))
            )
            njop_building_combined = np.concatenate(
                (njop_building, -np.asarray(other_njop_building, dtype=np.float64))
            )
            gross_revenue_combined = np.concatenate(
                (revenue, -np.asarray(other_revenue, dtype=np.float64))
            )
            cost_combined = np.concatenate((cost, -other_cost))
            cost_allocation_combined = cost_allocation + other_cost_allocation
            cost_type_combined = cost_type + other_cost_type
            description_combined = description + other_description

            return LBT(
                start_year=start_year_combined,
                end_year=end_year_combined,
                expense_year=expense_year_combined,
                cost_allocation=cost_allocation_combined,
                cost_type=cost_type_combined,
                description=description_combined,
                tax_portion=tax_portion_combined,
                tax_discount=tax_discount_combined,
                final_year=final_year_combined,
                utilized_land_area=utilized_land_area_combined,
                utilized_building_area=utilized_building_area_combined,
                njop_land=njop_land_combined,
                njop_building=njop_building_combined,
                gross_revenue=gross_revenue_combined,
                cost=cost_combined,
            )

        else:
            raise LBTException(
                "Must subtract between an instance of LBT with another instance of LBT. "
                + f"{other}({other.__class__.__qualname__}) is not an instance of LBT."
            )

    def __rsub__(self, other: object) -> LBT:
        return self.__sub__(other)

    def __mul__(self, other: object) -> LBT:
        # Multiplication is allowed only with an integer/a float
        (
            _project_duration,
            _project_years,
            cost_allocation,
            cost_type,
            description,
            tax_portion,
        ) = self._normalized_general_fields(LBTException)
        expense_year, _cost, tax_discount = self._normalized_cost_arrays(LBTException)
        final_year, land, building, njop_land, njop_building, revenue = (
            self._normalized_lbt_fields()
        )
        if isinstance(other, (int, float)):
            return LBT(
                start_year=self.start_year,
                end_year=self.end_year,
                expense_year=expense_year,
                cost_allocation=cost_allocation,
                cost_type=cost_type,
                description=description,
                tax_portion=tax_portion,
                tax_discount=tax_discount,
                final_year=final_year,
                utilized_land_area=land,
                utilized_building_area=building,
                njop_land=np.asarray(njop_land, dtype=np.float64) * other,
                njop_building=np.asarray(njop_building, dtype=np.float64) * other,
                gross_revenue=np.asarray(revenue, dtype=np.float64) * other,
            )

        else:
            raise LBTException(
                "Must multiply with an integer or a float. "
                + f"{other}({other.__class__.__qualname__}) "
                + "is not an integer nor a float."
            )

    def __rmul__(self, other: object) -> LBT:
        return self.__mul__(other)

    def __truediv__(self, other: object) -> LBT | float:
        # Between an instance of LBT with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales
        (
            _project_duration,
            _project_years,
            cost_allocation,
            cost_type,
            description,
            tax_portion,
        ) = self._normalized_general_fields(LBTException)
        expense_year, cost, tax_discount = self._normalized_cost_arrays(LBTException)
        final_year, land, building, njop_land, njop_building, revenue = (
            self._normalized_lbt_fields()
        )
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales)):
            (
                _other_project_duration,
                _other_project_years,
                _other_cost_allocation,
                _other_cost_type,
                _other_description,
                _other_tax_portion,
            ) = other._normalized_general_fields(LBTException)
            _other_expense_year, other_cost, _other_tax_discount = (
                other._normalized_cost_arrays(LBTException)
            )
            return float(np.sum(cost) / np.sum(other_cost))

        # Between an instance of LBT and an integer/a float
        elif isinstance(other, (int, float)):
            # Cannot divide with zero
            if other == 0:
                raise LBTException("Cannot divide with zero")

            else:
                return LBT(
                    start_year=self.start_year,
                    end_year=self.end_year,
                    expense_year=expense_year,
                    cost_allocation=cost_allocation,
                    cost_type=cost_type,
                    description=description,
                    final_year=final_year,
                    utilized_land_area=land,
                    utilized_building_area=building,
                    njop_land=np.asarray(njop_land, dtype=np.float64) / other,
                    njop_building=np.asarray(njop_building, dtype=np.float64) / other,
                    gross_revenue=np.asarray(revenue, dtype=np.float64) / other,
                    tax_portion=tax_portion,
                    tax_discount=tax_discount,
                )

        else:
            raise LBTException(
                "Must divide with an instance of CapitalCost/Intangible/OPEX/ASR/LBT"
                + f"/CostOfSales, integer or a float. {other}({other.__class__.__qualname__}) "
                + "is not an instance  of CapitalCost/Intangible/OPEX/ASR/LBT nor an integer "
                + "nor a float."
            )


@dataclass
@final
class CostOfSales(GeneralCost):
    """
    Manages a cost of sales.

    Parameters
    ----------
    Parameters are inherited from class GeneralCost

    Notes
    -----
    Overridden attributes are: expense_year and cost
    """

    expense_year: npt.NDArray[np.int64] | None = None
    cost: npt.NDArray[np.float64] | None = None

    @staticmethod
    def normalize_expense_year(
        expense_year: object,
        project_years: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int64]:
        """Validate raw expense-year input and return integer years."""
        if expense_year is None:
            return project_years.copy()
        if not _is_ndarray(expense_year):
            raise CostOfSalesException(
                f"Attribute expense_year must be provided as a numpy.ndarray, not as a/an {expense_year.__class__.__qualname__}"
            )
        if missing_value_indices(expense_year):
            raise CostOfSalesInputError(
                f"Missing values in array expense_year: {expense_year}"
            )
        return expense_year.astype(np.int64)

    @staticmethod
    def normalize_cost(
        cost: object,
        expense_year: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float64]:
        """Validate raw cost input and return floating-point costs."""
        if cost is None:
            return np.zeros_like(expense_year, dtype=np.float64)
        if not _is_ndarray(cost):
            raise CostOfSalesException(
                f"Attribute cost must be provided as a numpy.ndarray, not as an/a {cost.__class__.__qualname__}"
            )
        missing_indices = _missing_indices(cost)
        if len(missing_indices) > 0:
            cost[missing_indices] = np.zeros(len(missing_indices))
        return cost.astype(np.float64)

    @staticmethod
    def normalize_cost_allocation(
        cost_allocation: object,
        entry_count: int,
    ) -> list[FluidType]:
        """Validate raw cost allocation and normalize missing values to oil."""
        if cost_allocation is None:
            return [FluidType.OIL for _ in range(entry_count)]
        if not _is_object_list(cost_allocation):
            raise CostOfSalesException(
                f"Attribute cost allocation must be provided as a list, not as an/a {cost_allocation.__class__.__qualname__}"
            )

        normalized: list[FluidType] = []
        for value in cost_allocation:
            if _is_missing(value):
                normalized.append(FluidType.OIL)
            elif isinstance(value, FluidType):
                normalized.append(value)
            else:
                raise CostOfSalesException(
                    "Attribute cost allocation entries must be FluidType values"
                )
        return normalized

    def __post_init__(self):
        """
        Handles the following operations/procedures:
        -   Prepare attributes project_duration and project_years,
        -   Prepare attribute expense_year,
        -   Prepare attribute cost,
        -   Prepare attribute cost_allocation,
        -   Prepare attribute cost_type,
        -   Prepare attribute description,
        -   Prepare attribute tax_portion,
        -   Prepare attribute tax_discount,
        -   Check input data for unequal length,
        -   Raise an exception: expense_year is after the end year of the project,
        -   Raise an exception: expense_year is before the start_year of the project
        """

        # Prepare attributes project_duration and project_years
        if not period_is_reversed(self.start_year, self.end_year):
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise CostOfSalesInputError(
                f"start year {self.start_year} is after the end year {self.end_year}"
            )

        # Prepare attribute expense_year
        self.expense_year = self.normalize_expense_year(
            self.expense_year,
            self.project_years,
        )

        # Prepare attribute cost
        self.cost = self.normalize_cost(self.cost, self.expense_year)

        # Prepare attribute cost_allocation
        self.cost_allocation = self.normalize_cost_allocation(
            self.cost_allocation,
            len(self.expense_year),
        )

        # Prepare attribute cost_type
        raw_cost_type: object = self.cost_type
        if raw_cost_type is None:
            self.cost_type = [None for _ in range(len(self.expense_year))]

        else:
            if not _is_object_list(raw_cost_type):
                raise CostOfSalesException(
                    "Attribute cost_type must be given as a list, not "
                    + f"as a/an {self.cost_type.__class__.__qualname__}"
                )

            self.cost_type = _normalize_cost_type(raw_cost_type, CostOfSalesException)

        # Prepare attribute description
        if self.description is None:
            self.description = [" " for _ in range(len(self.expense_year))]

        else:
            raw_description: object = self.description
            if not _is_object_list(raw_description):
                raise CostOfSalesException(
                    "Attribute description must be provided as a list, "
                    + f"not as an/a {self.description.__class__.__qualname__}"
                )

            normalized_description: list[str] = []
            for val in raw_description:
                if _is_missing(val):
                    normalized_description.append(" ")
                elif isinstance(val, str):
                    normalized_description.append(val)
                else:
                    raise CostOfSalesException("Description values must be strings")
            self.description = normalized_description

        # Prepare attribute tax_portion
        if self.tax_portion is None:
            self.tax_portion = np.zeros_like(self.expense_year, dtype=np.float64)

        else:
            raw_tax_portion: object = self.tax_portion
            if not _is_ndarray(raw_tax_portion):
                raise CostOfSalesException(
                    "Attribute tax_portion must be given as a numpy.ndarray, "
                    + f"not as a/an {self.tax_portion.__class__.__qualname__}"
                )

            tax_portion_array = np.asarray(raw_tax_portion, dtype=np.float64)
            tax_portion_nan_id = _missing_indices(tax_portion_array)
            if len(tax_portion_nan_id) > 0:
                tax_portion_array[tax_portion_nan_id] = np.zeros(
                    len(tax_portion_nan_id)
                )

            if out_of_range_indices(tax_portion_array, 0.0, 1.0):
                raise CostOfSalesInputError(
                    "The value of tax_portion must be: 0 < tax_portion < 1, "
                    + f"tax_portion: {tax_portion_array}"
                )

        self.tax_portion = self.tax_portion.astype(np.float64)

        # Prepare attribute tax_discount
        raw_tax_discount: object = self.tax_discount
        if not (
            isinstance(raw_tax_discount, (float, int)) or _is_ndarray(raw_tax_discount)
        ):
            raise CostOfSalesException(
                "Attribute tax_discount must be provided as a float/int or as a numpy.ndarray, "
                + f"not as a/an {self.tax_discount.__class__.__qualname__}"
            )

        if isinstance(raw_tax_discount, (float, int)):
            if out_of_range_indices((raw_tax_discount,), 0.0, 1.0):
                raise CostOfSalesInputError(
                    "Attribute tax_discount must be between 0 and 1"
                )

            self.tax_discount = np.repeat(raw_tax_discount, len(self.expense_year))

        else:
            tax_discount_array = np.asarray(raw_tax_discount, dtype=np.float64)
            tax_discount_nan_id = _missing_indices(tax_discount_array)
            if len(tax_discount_nan_id) > 0:
                tax_discount_array[tax_discount_nan_id] = np.zeros(
                    len(tax_discount_nan_id)
                )

            if out_of_range_indices(tax_discount_array, 0.0, 1.0):
                raise CostOfSalesInputError(
                    "The value of tax_discount must be: 0 < tax_discount < 1, "
                    + f"tax_discount: {tax_discount_array}"
                )

        self.tax_discount = np.asarray(self.tax_discount, dtype=np.float64)

        # Check input data for unequal length
        expense_year = _require_present(
            self.expense_year, CostOfSalesException, "expense_year"
        )
        cost = _require_present(self.cost, CostOfSalesException, "cost")
        cost_allocation = _require_present(
            self.cost_allocation, CostOfSalesException, "cost_allocation"
        )
        cost_type = _require_present(self.cost_type, CostOfSalesException, "cost_type")
        description = _require_present(
            self.description, CostOfSalesException, "description"
        )
        tax_portion = _require_present(
            self.tax_portion, CostOfSalesException, "tax_portion"
        )
        tax_discount = _require_present(
            self.tax_discount, CostOfSalesException, "tax_discount"
        )
        length_map = {
            "expense_year": len(expense_year),
            "cost": len(cost),
            "cost_allocation": len(cost_allocation),
            "cost_type": len(cost_type),
            "description": len(description),
            "tax_portion": len(tax_portion),
            "tax_discount": len(tax_discount),
        }

        if unequal_lengths(length_map) is not None:
            raise CostOfSalesInputError(
                "Unequal length of arrays: "
                + f"expense_year: {len(expense_year)}, "
                + f"cost: {len(cost)}, "
                + f"cost_allocation: {len(cost_allocation)}, "
                + f"cost_type: {len(cost_type)}, "
                + f"description: {len(description)}, "
                + f"tax_portion: {len(tax_portion)}, "
                + f"tax_discount: {len(tax_discount)} "
            )

        # Raise an error message: expense_year is after the end year of the project
        above = year_above(self.expense_year, self.end_year)
        if above is not None:
            raise CostOfSalesInputError(
                f"Expense year ({above}) "
                + f"is after the end year of the project ({self.end_year})"
            )

        # Raise an error message: expense_year is before the start year of the project
        below = year_below(self.expense_year, self.start_year)
        if below is not None:
            raise CostOfSalesInputError(
                f"Expense year ({below}) "
                + f"is before the start year of the project ({self.start_year})"
            )

    @override
    def __eq__(self, other: object) -> bool:
        # Between two instances of CostOfSales
        (
            _project_duration,
            _project_years,
            cost_allocation,
            cost_type,
            _description,
            tax_portion,
        ) = self._normalized_general_fields(CostOfSalesException)
        expense_year, cost, tax_discount = self._normalized_cost_arrays(
            CostOfSalesException
        )
        if isinstance(other, CostOfSales):
            (
                _other_project_duration,
                _other_project_years,
                other_cost_allocation,
                other_cost_type,
                _other_description,
                other_tax_portion,
            ) = other._normalized_general_fields(CostOfSalesException)
            other_expense_year, other_cost, other_tax_discount = (
                other._normalized_cost_arrays(CostOfSalesException)
            )
            return all(
                (
                    self.start_year == other.start_year,
                    self.end_year == other.end_year,
                    np.allclose(expense_year, other_expense_year),
                    np.allclose(cost, other_cost),
                    np.allclose(tax_portion, other_tax_portion),
                    np.allclose(tax_discount, other_tax_discount),
                    cost_allocation == other_cost_allocation,
                    cost_type == other_cost_type,
                )
            )

        # Between an instance of CostOfSales and an integer/a float
        elif isinstance(other, (int, float)):
            return float(np.sum(cost)) == other

        else:
            return False

    def __lt__(self, other: object) -> bool:
        # Between an instance of CostOfSales with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales
        (
            _project_duration,
            _project_years,
            _cost_allocation,
            _cost_type,
            _description,
            _tax_portion,
        ) = self._normalized_general_fields(CostOfSalesException)
        _expense_year, cost, _tax_discount = self._normalized_cost_arrays(
            CostOfSalesException
        )
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales)):
            (
                _other_project_duration,
                _other_project_years,
                _other_cost_allocation,
                _other_cost_type,
                _other_description,
                _other_tax_portion,
            ) = other._normalized_general_fields(CostOfSalesException)
            _other_expense_year, other_cost, _other_tax_discount = (
                other._normalized_cost_arrays(CostOfSalesException)
            )
            return bool(np.sum(cost) < np.sum(other_cost))

        # Between an instance of CostOfSales and an integer/a float
        elif isinstance(other, (int, float)):
            return bool(np.sum(cost) < other)

        else:
            raise CostOfSalesException(
                "Must compare an instance of CostOfSales with another instance of "
                + "CapitalCost/Instangible/OPEX/ASR/LBT/CostOfSales, an integer, or a float."
            )

    def __le__(self, other: object) -> bool:
        # Between an instance of CostOfSales with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales
        (
            _project_duration,
            _project_years,
            _cost_allocation,
            _cost_type,
            _description,
            _tax_portion,
        ) = self._normalized_general_fields(CostOfSalesException)
        _expense_year, cost, _tax_discount = self._normalized_cost_arrays(
            CostOfSalesException
        )
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales)):
            (
                _other_project_duration,
                _other_project_years,
                _other_cost_allocation,
                _other_cost_type,
                _other_description,
                _other_tax_portion,
            ) = other._normalized_general_fields(CostOfSalesException)
            _other_expense_year, other_cost, _other_tax_discount = (
                other._normalized_cost_arrays(CostOfSalesException)
            )
            return bool(np.sum(cost) <= np.sum(other_cost))

        # Between an instance of CostOfSales and an integer/a float
        elif isinstance(other, (int, float)):
            return bool(np.sum(cost) <= other)

        else:
            raise CostOfSalesException(
                "Must compare an instance of CostOfSales with another instance of "
                + "CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales, an integer, or a float."
            )

    def __gt__(self, other: object) -> bool:
        # Between an instance of CostOfSales with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales
        (
            _project_duration,
            _project_years,
            _cost_allocation,
            _cost_type,
            _description,
            _tax_portion,
        ) = self._normalized_general_fields(CostOfSalesException)
        _expense_year, cost, _tax_discount = self._normalized_cost_arrays(
            CostOfSalesException
        )
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales)):
            (
                _other_project_duration,
                _other_project_years,
                _other_cost_allocation,
                _other_cost_type,
                _other_description,
                _other_tax_portion,
            ) = other._normalized_general_fields(CostOfSalesException)
            _other_expense_year, other_cost, _other_tax_discount = (
                other._normalized_cost_arrays(CostOfSalesException)
            )
            return bool(np.sum(cost) > np.sum(other_cost))

        # Between an instance of CostOfSales and an integer/a float
        elif isinstance(other, (int, float)):
            return bool(np.sum(cost) > other)

        else:
            raise CostOfSalesException(
                "Must compare an instance of CostOfSales with another instance of "
                + "CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales, an integer, or a float."
            )

    def __ge__(self, other: object) -> bool:
        # Between an instance of CostOfSales with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales
        (
            _project_duration,
            _project_years,
            _cost_allocation,
            _cost_type,
            _description,
            _tax_portion,
        ) = self._normalized_general_fields(CostOfSalesException)
        _expense_year, cost, _tax_discount = self._normalized_cost_arrays(
            CostOfSalesException
        )
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales)):
            (
                _other_project_duration,
                _other_project_years,
                _other_cost_allocation,
                _other_cost_type,
                _other_description,
                _other_tax_portion,
            ) = other._normalized_general_fields(CostOfSalesException)
            _other_expense_year, other_cost, _other_tax_discount = (
                other._normalized_cost_arrays(CostOfSalesException)
            )
            return bool(np.sum(cost) >= np.sum(other_cost))

        # Between an instance of CostOfSales and an integer/a float
        elif isinstance(other, (int, float)):
            return bool(np.sum(cost) >= other)

        else:
            raise CostOfSalesException(
                "Must compare an instance of CostOfSales with another instance of "
                + "CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales, an integer, or a float."
            )

    def __add__(self, other: object) -> CostOfSales:
        # Only allows addition between an instance of CostOfSales
        # and another instance of CostOfSales
        (
            _project_duration,
            _project_years,
            cost_allocation,
            cost_type,
            description,
            tax_portion,
        ) = self._normalized_general_fields(CostOfSalesException)
        expense_year, cost, tax_discount = self._normalized_cost_arrays(
            CostOfSalesException
        )
        if isinstance(other, CostOfSales):
            (
                _other_project_duration,
                _other_project_years,
                other_cost_allocation,
                other_cost_type,
                other_description,
                other_tax_portion,
            ) = other._normalized_general_fields(CostOfSalesException)
            other_expense_year, other_cost, other_tax_discount = (
                other._normalized_cost_arrays(CostOfSalesException)
            )
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            cost_allocation_combined = cost_allocation + other_cost_allocation
            cost_type_combined = cost_type + other_cost_type
            description_combined = description + other_description
            tax_portion_combined = np.concatenate((tax_portion, other_tax_portion))
            tax_discount_combined = np.concatenate((tax_discount, other_tax_discount))
            expense_year_combined = np.concatenate((expense_year, other_expense_year))
            cost_combined = np.concatenate((cost, other_cost))

            return CostOfSales(
                start_year=start_year_combined,
                end_year=end_year_combined,
                cost_allocation=cost_allocation_combined,
                cost_type=cost_type_combined,
                description=description_combined,
                tax_portion=tax_portion_combined,
                tax_discount=tax_discount_combined,
                expense_year=expense_year_combined,
                cost=cost_combined,
            )

        else:
            raise CostOfSalesException(
                "Must add between an instance of CostOfSales with another instance "
                + f"of CostOfSales. {other}({other.__class__.__qualname__}) is not "
                + "an instance of CostOfSales."
            )

    def __iadd__(self, other: object) -> CostOfSales:
        return self.__add__(other)

    def __sub__(self, other: object) -> CostOfSales:
        # Only allows subtraction between an instance of CostOfSales
        # and another instance of CostOfSales
        (
            _project_duration,
            _project_years,
            cost_allocation,
            cost_type,
            description,
            tax_portion,
        ) = self._normalized_general_fields(CostOfSalesException)
        expense_year, cost, tax_discount = self._normalized_cost_arrays(
            CostOfSalesException
        )
        if isinstance(other, CostOfSales):
            (
                _other_project_duration,
                _other_project_years,
                other_cost_allocation,
                other_cost_type,
                other_description,
                other_tax_portion,
            ) = other._normalized_general_fields(CostOfSalesException)
            other_expense_year, other_cost, other_tax_discount = (
                other._normalized_cost_arrays(CostOfSalesException)
            )
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            cost_allocation_combined = cost_allocation + other_cost_allocation
            cost_type_combined = cost_type + other_cost_type
            description_combined = description + other_description
            tax_portion_combined = np.concatenate((tax_portion, other_tax_portion))
            tax_discount_combined = np.concatenate((tax_discount, other_tax_discount))
            expense_year_combined = np.concatenate((expense_year, other_expense_year))
            cost_combined = np.concatenate((cost, -other_cost))

            return CostOfSales(
                start_year=start_year_combined,
                end_year=end_year_combined,
                cost_allocation=cost_allocation_combined,
                cost_type=cost_type_combined,
                description=description_combined,
                tax_portion=tax_portion_combined,
                tax_discount=tax_discount_combined,
                expense_year=expense_year_combined,
                cost=cost_combined,
            )

        else:
            raise CostOfSalesException(
                "Must subtract between an instance of CostOfSales with another instance "
                + f"of CostOfSales. {other}({other.__class__.__qualname__}) is not "
                + "an instance of CostOfSales."
            )

    def __rsub__(self, other: object) -> CostOfSales:
        return self.__sub__(other)

    def __mul__(self, other: object) -> CostOfSales:
        # Multiplication is allowed only with an integer/a float
        (
            _project_duration,
            _project_years,
            cost_allocation,
            cost_type,
            description,
            tax_portion,
        ) = self._normalized_general_fields(CostOfSalesException)
        expense_year, cost, tax_discount = self._normalized_cost_arrays(
            CostOfSalesException
        )
        if isinstance(other, (int, float)):
            return CostOfSales(
                start_year=self.start_year,
                end_year=self.end_year,
                cost_allocation=cost_allocation,
                cost_type=cost_type,
                description=description,
                tax_portion=tax_portion,
                tax_discount=tax_discount,
                expense_year=expense_year,
                cost=cost * other,
            )

        else:
            raise CostOfSalesException(
                "Must multiply with an integer or a float. "
                + f"{other}({other.__class__.__qualname__}) "
                + "is not an integer nor a float."
            )

    def __rmul__(self, other: object) -> CostOfSales:
        return self.__mul__(other)

    def __truediv__(self, other: object) -> CostOfSales | float:
        # Between an instance of CostOfSales with another instance of
        # CapitalCost/Intangible/OPEX/ASR/LBT/CostOfSales
        (
            _project_duration,
            _project_years,
            cost_allocation,
            cost_type,
            description,
            tax_portion,
        ) = self._normalized_general_fields(CostOfSalesException)
        expense_year, cost, tax_discount = self._normalized_cost_arrays(
            CostOfSalesException
        )
        if isinstance(other, (CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales)):
            (
                _other_project_duration,
                _other_project_years,
                _other_cost_allocation,
                _other_cost_type,
                _other_description,
                _other_tax_portion,
            ) = other._normalized_general_fields(CostOfSalesException)
            _other_expense_year, other_cost, _other_tax_discount = (
                other._normalized_cost_arrays(CostOfSalesException)
            )
            return float(np.sum(cost) / np.sum(other_cost))

        # Between an instance of CostOfSales and an integer/a float
        elif isinstance(other, (int, float)):
            # Cannot divide by zero
            if other == 0:
                raise CostOfSalesException("Cannot divide with zero.")

            else:
                return CostOfSales(
                    start_year=self.start_year,
                    end_year=self.end_year,
                    cost_allocation=cost_allocation,
                    cost_type=cost_type,
                    description=description,
                    tax_portion=tax_portion,
                    tax_discount=tax_discount,
                    expense_year=expense_year,
                    cost=cost / other,
                )

        else:
            raise CostOfSalesException(
                "Must divide with an instance of CapitalCost/Intangible/OPEX/ASR/LBT"
                + f"/CostOfSales, an integer, or a float. {other}({other.__class__.__qualname__}) "
                + "is not an instance of CapitalCost/Intangible/OPEX/ASR/LBT nor an integer "
                + "nor a float."
            )
