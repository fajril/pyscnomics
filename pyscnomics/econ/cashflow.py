from dataclasses import dataclass, field
from typing import TypeGuard

import numpy as np
import numpy.typing as npt

from pyscnomics.contracts.project import BaseProject, BaseProjectSummaryException
from pyscnomics.contracts.summary_profile import PostRunSummaryProfile
from pyscnomics.contracts.transition import Transition, TransitionException
from pyscnomics.econ.indicator import (
    irr,
    npv_nominal_terms,
    npv_point_forward,
    npv_real_terms,
    npv_skk_nominal_terms,
    npv_skk_real_terms,
    pot_psc,
)
from pyscnomics.econ.selection import DiscountingMode, NPVSelection
from pyscnomics.input_validation import InputValidationError, engine_input_issue


class CashflowException(Exception):
    """Exception to raise for a misuse of Cashflow Exception"""

    pass


class CashflowInputError(  # pyright: ignore[reportUnsafeMultipleInheritance]
    CashflowException, InputValidationError
):
    """Verified user-input cashflow error retaining ``CashflowException``."""

    _input_code: str = "CASHFLOW_INPUT_INVALID"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        InputValidationError.__init__(
            self,
            (engine_input_issue(code=code or self._input_code, message=message),),
        )


def _require_integer_value(value: object) -> int:
    """Validate one integer scalar returned by a NumPy array."""
    if isinstance(value, int):
        return value
    if isinstance(value, np.integer):
        scalar_value: object = value.item()
        return _require_integer_value(scalar_value)
    raise CashflowException("Cashflow project years must contain integer values.")


def _require_float_value(value: object) -> float:
    """Validate one numeric scalar returned by a NumPy array."""
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    if isinstance(value, np.floating):
        scalar_value: object = value.item()
        return _require_float_value(scalar_value)
    if isinstance(value, np.integer):
        scalar_value = value.item()
        return _require_float_value(scalar_value)
    raise CashflowException("Cashflow values must contain numeric values.")


def _require_integer_item(values: npt.NDArray[np.int64], index: int) -> int:
    """Return a validated integer array element."""
    value: object = values.item(index)
    return _require_integer_value(value)


def _require_float_item(values: npt.NDArray[np.float64], index: int) -> float:
    """Return a validated floating array element."""
    value: object = values.item(index)
    return _require_float_value(value)


def _is_contract_tuple(value: object) -> TypeGuard[tuple[object, ...]]:
    """Identify tuple inputs before validating every public contract element."""
    return isinstance(value, tuple)


def _normalize_contract_input(
    contract_input: object,
) -> BaseProject | Transition | tuple[BaseProject | Transition, ...]:
    """Validate the public contract input before cashflow processing."""
    if isinstance(contract_input, (BaseProject, Transition)):
        return contract_input
    if not _is_contract_tuple(contract_input):
        raise CashflowException(
            f"The contract type: {type(contract_input)} , is not recognized."
        )

    contracts: list[BaseProject | Transition] = []
    for item in contract_input:
        contract: object = item
        if not isinstance(contract, (BaseProject, Transition)):
            raise CashflowException("Cashflow dataclass only receive psc contract/s.")
        contracts.append(contract)
    return tuple(contracts)


def _require_post_run_summary_profile(
    contract: BaseProject | Transition,
) -> PostRunSummaryProfile:
    """Return public output from a completed contract run."""
    try:
        return contract.get_post_run_summary_profile()
    except (BaseProjectSummaryException, TransitionException) as error:
        message = "Cashflow dataclass only receive contract that has been run. Please run the contract before parse it to the Cashflow dataclass."
        raise CashflowException(message) from error


def _require_project_years(
    contract: BaseProject | Transition,
) -> npt.NDArray[np.int64]:
    """Return the completed contract timeline used by its cashflow."""
    project_years = _require_post_run_summary_profile(contract).project_years
    if project_years.ndim != 1 or project_years.size == 0:
        raise CashflowException(
            "Cashflow requires a completed contract with a non-empty project timeline."
        )
    return project_years


def _require_consolidated_cashflow(
    contract: BaseProject | Transition,
) -> npt.NDArray[np.float64]:
    """Return the completed consolidated cashflow for one contract."""
    cashflow = _require_post_run_summary_profile(contract).consolidated_cashflow
    if cashflow is None:
        raise CashflowException(
            "Cashflow dataclass only receive contract that has been run. Please run the contract before parse it to the Cashflow dataclass."
        )
    return cashflow


@dataclass
class Cashflow:
    """
    A dataclass used to represent the cashflow for a project.

    Attributes
    ----------
    contract : (CostRecovery | GrossSplit | Transition | Tuple[Union[CostRecovery, GrossSplit, Transition], ...])
        The type of contract or a tuple of different contract types.
    reference_year : int, optional
        The reference year for the economic indicator (default is None).
    inflation_rate : float, optional
        The inflation rate applied to calculate the economic indicator (default is 0.0).
    discount_rate : float, optional
        The discount rate used for calculating NPV (default is 0.1).
    npv_mode : NPVSelection, optional
        The mode for calculating NPV, default is NPVSelection.NPV_SKK_REAL_TERMS.
    discounting_mode : DiscountingMode, optional
        The mode for discounting, default is DiscountingMode.END_YEAR.
    start_year : int, optional
        The start year of the cashflow, defined later (default is None, not included in initialization or representation).
    end_year : int, optional
        The end year of the cashflow, defined later (default is None, not included in initialization or representation).
    project_years : np.ndarray, optional
        The years of the project, defined later (default is None, not included in initialization or representation).
    cashflow : np.ndarray, optional
        The cashflow values for the project, defined later (default is None, not included in initialization or representation).
    npv : float, optional
        The Net Present Value (NPV) of the cashflow, defined later (default is None, not included in initialization or representation).
    irr : float, optional
        The Internal Rate of Return (IRR) of the cashflow, defined later (default is None, not included in initialization or representation).
    pot : float, optional
        The Payback on Time (POT) of the cashflow, defined later (default is None, not included in initialization or representation).
    """

    contract: BaseProject | Transition | tuple[BaseProject | Transition, ...]

    reference_year: int | None = field(default=None)
    inflation_rate: float = field(default=0.0)
    discount_rate: float = field(default=0.1)
    npv_mode: NPVSelection = field(default=NPVSelection.NPV_SKK_REAL_TERMS)
    discounting_mode: DiscountingMode = field(default=DiscountingMode.END_YEAR)

    # Attributes associated with the date of the cashflow
    start_year: int = field(init=False, repr=False)
    end_year: int = field(init=False, repr=False)

    # Attributes associated with the years which will be defined later
    project_years: npt.NDArray[np.int64] = field(init=False, repr=False)

    # Attributes associated with the economic indicator which will be defined later
    cashflow: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    npv: float | None = field(default=None, init=False, repr=False)
    irr: float | None = field(default=None, init=False, repr=False)
    pot: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        contract_input: object = self.contract
        self.contract = _normalize_contract_input(contract_input)

        # Check if the contract/s has been run
        self.check_run_status()

        # Filling the start_year, end_year, and project_years
        if isinstance(self.contract, tuple):
            self.get_multiple_project_start_end()
            self.get_multiple_project_years()
        else:
            self.get_single_project_start_end()
            self.get_single_project_years()

        # Check the reference year
        if self.reference_year is None:
            # Retrieving the first year for the contract's project year as the reference year
            self.reference_year = int(np.min(self.project_years))
        else:
            self.check_reference_year()

    def check_run_status(self) -> None:
        """
        Function to check whether the contract has been run or not.
        """
        contract = self.contract
        if not isinstance(contract, tuple):
            _ = _require_consolidated_cashflow(contract)
            return

        for index, item in enumerate(contract):
            try:
                _ = _require_consolidated_cashflow(item)
            except CashflowException as error:
                message = f"Cashflow dataclass only receive contracts that has been run. Contract in index {index} has not been run, please run the contract before parse it to the Cashflow dataclass"
                raise CashflowException(message) from error

    def get_single_project_start_end(self) -> None:
        """
        Function to parse the self.start_year and self.end_year
        """
        contract = self.contract
        if isinstance(contract, tuple):
            raise CashflowException("Cashflow dataclass only receive psc contract/s.")
        if isinstance(contract, BaseProject):
            self.start_year = contract.start_date.year
            self.end_year = contract.end_date.year
        else:
            project_years = _require_project_years(contract)
            self.start_year = _require_integer_item(project_years, 0)
            self.end_year = _require_integer_item(project_years, -1)

    def get_multiple_project_start_end(self) -> None:
        """
        Function to parse the self.start_year and self.end_year from the given contracts.
        """
        contracts = self.contract
        if not isinstance(contracts, tuple):
            raise CashflowException("Cashflow dataclass only receive psc contract/s.")
        start_years: list[int] = []
        end_years: list[int] = []
        for contract in contracts:
            if isinstance(contract, BaseProject):
                start_years.append(contract.start_date.year)
                end_years.append(contract.end_date.year)
            else:
                project_years = _require_project_years(contract)
                start_years.append(_require_integer_item(project_years, 0))
                end_years.append(_require_integer_item(project_years, -1))
        self.start_year = min(start_years)
        self.end_year = max(end_years)

    def get_single_project_years(self) -> None:
        """
        Function to parse the self.start_year and self.end_year from the given contract.
        """
        contract = self.contract
        if isinstance(contract, tuple):
            raise CashflowException("Cashflow dataclass only receive psc contract/s.")
        self.project_years = _require_project_years(contract)

    def get_multiple_project_years(self) -> None:
        """
        Function to parse the self.project_years from the given contracts.
        """
        contracts = self.contract
        if not isinstance(contracts, tuple):
            raise CashflowException("Cashflow dataclass only receive psc contract/s.")

        start_year_arr = self.start_year
        end_year_arr = self.end_year
        self.project_years = np.arange(
            start=start_year_arr, stop=end_year_arr, step=1, dtype=np.int64
        )

    def check_reference_year(self) -> None:
        """
        Function to check the input of reference year.
        """
        if self.reference_year is not None:
            if self.reference_year < self.start_year:
                raise CashflowInputError(
                    f"The reference year: {self.reference_year}, is less than start_year {self.start_year}. The reference year should be in the range of the start_year and end_year",
                    code="CASHFLOW_REFERENCE_YEAR_OUTSIDE_PROJECT",
                )
            elif self.reference_year > self.end_year:
                raise CashflowInputError(
                    f"The reference year: {self.reference_year}, is greater than end_year {self.end_year}. The reference year should be in the range of the start_year and end_year",
                    code="CASHFLOW_REFERENCE_YEAR_OUTSIDE_PROJECT",
                )

    def get_cashflow_all(self) -> None:
        """
        Function to parse self.cashflow from the input of contract/s.
        """
        if isinstance(self.contract, tuple):
            self.parse_multiple_cashflow()
        else:
            self.cashflow = _require_consolidated_cashflow(self.contract)

    def parse_multiple_cashflow(self) -> None:
        """
        Function to retrieve the cashflow of the given contracts.
        """
        contracts = self.contract
        if not isinstance(contracts, tuple):
            raise CashflowException("Cashflow dataclass only receive psc contract/s.")

        # Initiating the contract_years
        corresponding_year = self.project_years

        # Initiating the cashflow array
        total_cashflow = np.zeros_like(corresponding_year, dtype=np.float64)
        self.cashflow = total_cashflow

        # Iterate through each contract
        for contract in contracts:
            # Iterate through each year and cashflow in the contract
            contract_years = _require_project_years(contract)
            contract_cashflow = _require_consolidated_cashflow(contract)
            for index in range(contract_years.size):
                year = _require_integer_item(contract_years, index)
                for cashflow_index in range(corresponding_year.size):
                    if (
                        _require_integer_item(corresponding_year, cashflow_index)
                        == year
                    ):
                        total_cashflow[cashflow_index] += _require_float_item(
                            contract_cashflow, index
                        )
                        break

    def get_contract_npv(self) -> float:
        """
        Function to get the Net Present Value (NPV) of the obtained cashflow.
        """
        reference_year = self.reference_year
        if reference_year is None:
            raise CashflowException(
                "Reference year must be initialized before NPV calculation."
            )
        cashflow = self.cashflow
        if cashflow is None:
            raise CashflowException(
                "Cashflow must be initialized before NPV calculation."
            )

        # NPV Calculation for SKK Real Terms
        if self.npv_mode == NPVSelection.NPV_SKK_REAL_TERMS:
            ctr_npv = npv_skk_real_terms(
                cashflow=cashflow,
                cashflow_years=self.project_years,
                discount_rate=self.discount_rate,
                reference_year=reference_year,
                discounting_mode=self.discounting_mode,
            )

        # NPV Calculation for SKK Nominal Terms
        elif self.npv_mode == NPVSelection.NPV_SKK_NOMINAL_TERMS:
            ctr_npv = npv_skk_nominal_terms(
                cashflow=cashflow,
                cashflow_years=self.project_years,
                discount_rate=self.discount_rate,
                discounting_mode=self.discounting_mode,
            )

        # NPV Calculation for Nominal Terms
        elif self.npv_mode == NPVSelection.NPV_NOMINAL_TERMS:
            ctr_npv = npv_nominal_terms(
                cashflow=cashflow,
                cashflow_years=self.project_years,
                discount_rate=self.discount_rate,
                reference_year=reference_year,
                discounting_mode=self.discounting_mode,
            )

        # NPV Calculation for Real Terms
        elif self.npv_mode == NPVSelection.NPV_REAL_TERMS:
            ctr_npv = npv_real_terms(
                cashflow=cashflow,
                cashflow_years=self.project_years,
                discount_rate=self.discount_rate,
                reference_year=reference_year,
                inflation_rate=self.inflation_rate,
                discounting_mode=self.discounting_mode,
            )

        # NPV Calculation for Point Forwards
        elif self.npv_mode == NPVSelection.NPV_POINT_FORWARD:
            ctr_npv = npv_point_forward(
                cashflow=cashflow,
                cashflow_years=self.project_years,
                discount_rate=self.discount_rate,
                reference_year=reference_year,
                discounting_mode=self.discounting_mode,
            )

        # Condition when the npv_mode is not recognized
        else:
            raise CashflowInputError(
                f"The npv_mode: {self.npv_mode} , is not recognized.The npv_mode input should be in one of NPVSelection.",
                code="CASHFLOW_NPV_MODE_UNRECOGNIZED",
            )

        return ctr_npv

    def get_contract_irr(self) -> float | None:
        """
        Function to get the Internal Rate of Return (IRR) of the obtained cashflow.
        """
        cashflow = self.cashflow
        if cashflow is None:
            raise CashflowException(
                "Cashflow must be initialized before IRR calculation."
            )
        return irr(cashflow=cashflow)

    def get_contract_pot(self) -> float:
        """
        Function to get the Pay-Out Time (POT) of the obtained cashflow.
        """
        reference_year = self.reference_year
        if reference_year is None:
            raise CashflowException(
                "Reference year must be initialized before POT calculation."
            )
        cashflow = self.cashflow
        if cashflow is None:
            raise CashflowException(
                "Cashflow must be initialized before POT calculation."
            )

        return pot_psc(
            cashflow=cashflow,
            cashflow_years=self.project_years,
            reference_year=reference_year,
        )

    def run(self) -> None:
        """
        Function to get the cashflow and economic indicator of the given contract/s.
        """
        # Retrieve the cashflow of the contract/s
        self.get_cashflow_all()

        # Retrieve the economic indicator
        self.npv = self.get_contract_npv()
        self.irr = self.get_contract_irr()
        self.pot = self.get_contract_pot()
