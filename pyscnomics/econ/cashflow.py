from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from typing import TypeGuard, cast

import numpy as np
import numpy.typing as npt

from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.transition import Transition
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


class CashflowException(Exception):
    """Exception to raise for a misuse of Cashflow Exception"""

    pass


def _is_contract_tuple(
    contract: object,
) -> TypeGuard[tuple[BaseProject | Transition, ...]]:
    return isinstance(contract, tuple) and all(
        isinstance(item, (BaseProject, Transition))
        for item in cast(tuple[object, ...], contract)
    )


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
        # Check if the contract/s has been run
        self.check_run_status()

        # Filling the start_year, end_year, and project_years
        if isinstance(self.contract, (BaseProject, Transition)):
            self.get_single_project_start_end()
            self.get_single_project_years()

        else:
            if _is_contract_tuple(self.contract):
                self.get_multiple_project_start_end()
                self.get_multiple_project_years()
            else:
                raise CashflowException(
                    f"The contract type: {type(self.contract)} , is not recognized."
                )

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
        # Condition when the contract is a single-value
        if isinstance(self.contract, (BaseProject, Transition)):
            if getattr(self.contract, "_consolidated_cashflow") is None:
                raise CashflowException(
                    "Cashflow dataclass only receive contract that has been run. Please run the contract before parse it to the Cashflow dataclass."
                )

        # Condition when the contract is a multi-value
        else:
            if _is_contract_tuple(self.contract):
                for index, cntr in enumerate(self.contract):
                    if getattr(cntr, "_consolidated_cashflow") is None:
                        raise CashflowException(
                            f"Cashflow dataclass only receive contracts that has been run. Contract in index {index} has not been run, please run the contract before parse it to the Cashflow dataclass"
                        )
            else:
                raise CashflowException(
                    "Cashflow dataclass only receive psc contract/s."
                )

    def get_single_project_start_end(self) -> None:
        """
        Function to parse the self.start_year and self.end_year
        """
        contract = self.contract
        if not isinstance(contract, (BaseProject, Transition)):
            raise CashflowException("Cashflow dataclass only receive psc contract/s.")

        start_date = cast(date, getattr(contract, "start_date"))
        self.start_year = start_date.year

        end_date = cast(date, getattr(contract, "end_date"))
        self.end_year = end_date.year

    def get_multiple_project_start_end(self) -> None:
        """
        Function to parse the self.start_year and self.end_year from the given contracts.
        """
        contracts = self.contract
        if not isinstance(contracts, tuple):
            raise CashflowException("Cashflow dataclass only receive psc contract/s.")

        self.start_year = min(
            cast(date, getattr(cntr, "start_date")).year for cntr in contracts
        )
        self.end_year = max(
            cast(date, getattr(cntr, "end_date")).year for cntr in contracts
        )

    def get_single_project_years(self) -> None:
        """
        Function to parse the self.start_year and self.end_year from the given contract.
        """
        contract = self.contract
        if not isinstance(contract, (BaseProject, Transition)):
            raise CashflowException("Cashflow dataclass only receive psc contract/s.")

        self.project_years = np.asarray(
            getattr(contract, "project_years"), dtype=np.int64
        )

    def get_multiple_project_years(self) -> None:
        """
        Function to parse the self.project_years from the given contracts.
        """
        contracts = self.contract
        if not isinstance(contracts, tuple):
            raise CashflowException("Cashflow dataclass only receive psc contract/s.")

        start_year_arr = min(
            cast(date, getattr(cntr, "start_date")).year for cntr in contracts
        )
        end_year_arr = max(
            cast(date, getattr(cntr, "end_date")).year for cntr in contracts
        )
        self.project_years = np.arange(
            start=start_year_arr, stop=end_year_arr, step=1, dtype=np.int64
        )

    def check_reference_year(self) -> None:
        """
        Function to check the input of reference year.
        """
        if self.reference_year is not None:
            if self.reference_year < self.start_year:
                raise CashflowException(
                    f"The reference year: {self.reference_year}, is less than start_year {self.start_year}. The reference year should be in the range of the start_year and end_year"
                )
            elif self.reference_year > self.end_year:
                raise CashflowException(
                    f"The reference year: {self.reference_year}, is greater than end_year {self.end_year}. The reference year should be in the range of the start_year and end_year"
                )

    def get_cashflow_all(self) -> None:
        """
        Function to parse self.cashflow from the input of contract/s.
        """
        if isinstance(self.contract, (BaseProject, Transition)):
            self.cashflow = np.asarray(
                getattr(self.contract, "_consolidated_cashflow"), dtype=np.float64
            )

        else:
            if _is_contract_tuple(self.contract):
                self.parse_multiple_cashflow()
            else:
                raise CashflowException(
                    f"The contract type: {type(self.contract)} , is not recognized."
                )

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
        self.cashflow = np.zeros_like(corresponding_year, dtype=np.float64)

        # Iterate through each contract
        for cntr in contracts:
            # Iterate through each year and cashflow in the contract
            contract_years = np.asarray(getattr(cntr, "project_years"), dtype=np.int64)
            contract_cashflow = np.asarray(
                getattr(cntr, "_consolidated_cashflow"), dtype=np.float64
            )
            years = cast(Iterable[np.int64], contract_years)
            cashflows = cast(Iterable[np.float64], contract_cashflow)
            for year, cashflow in zip(years, cashflows):
                if year in corresponding_year:
                    matching_year = np.asarray(
                        corresponding_year == year, dtype=np.bool_
                    )
                    index = cast(np.intp, np.flatnonzero(matching_year)[0])
                    self.cashflow[index] += cashflow

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
            raise CashflowException(
                f"The npv_mode: {self.npv_mode} , is not recognized.The npv_mode input should be in one of NPVSelection."
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
