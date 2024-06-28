from dataclasses import dataclass, field
from typing import Union, Tuple
import numpy as np

from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts.transition import Transition

from pyscnomics.econ.selection import NPVSelection, DiscountingMode

from pyscnomics.econ.indicator import (irr,
                                       npv_nominal_terms,
                                       npv_real_terms,
                                       npv_skk_nominal_terms,
                                       npv_skk_real_terms,
                                       npv_point_forward,
                                       pot_psc)


class CashflowException(Exception):
    """Exception to raise for a misuse of Cashflow Exception"""

    pass


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
    contract: (CostRecovery | GrossSplit | Transition |
               Tuple[Union[CostRecovery, GrossSplit, Transition], ...])

    reference_year: int = field(default=None)
    inflation_rate: float = field(default=0.0)
    discount_rate: float = field(default=0.1)
    npv_mode: NPVSelection = field(default=NPVSelection.NPV_SKK_REAL_TERMS)
    discounting_mode: DiscountingMode = field(default=DiscountingMode.END_YEAR)

    # Attributes associated with the date of the cashflow
    start_year: int = field(default=None, init=False, repr=False)
    end_year: int = field(default=None, init=False, repr=False)

    # Attributes associated with the years which will be defined later
    project_years: np.ndarray = field(default=None, init=False, repr=False)

    # Attributes associated with the economic indicator which will be defined later
    cashflow: np.ndarray = field(default=None, init=False, repr=False)
    npv: float = field(default=None, init=False, repr=False)
    irr: float = field(default=None, init=False, repr=False)
    pot: float = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # Check if the contract/s has been run
        self.check_run_status()

        # Filling the start_year, end_year, and project_years
        if isinstance(self.contract, (BaseProject, CostRecovery, GrossSplit, Transition)):
            self.get_single_project_start_end()
            self.get_single_project_years()

        elif isinstance(self.contract, tuple):
            self.get_multiple_project_start_end()
            self.get_multiple_project_years()

        else:
            raise CashflowException(
                f"The contract type: {type(self.contract)} , is not recognized."
            )

        # Check the reference year
        if self.reference_year is None:
            # Retrieving the first year for the contract's project year as the reference year
            self.reference_year = min(self.project_years)
        else:
            self.check_reference_year()

    def check_run_status(self):
        """
        Function to check whether the contract has been run or not.
        """
        # Condition when the contract is a single-value
        if isinstance(self.contract, (BaseProject, CostRecovery, GrossSplit, Transition)):
            if self.contract._consolidated_cashflow is None:
                raise CashflowException(
                    f"Cashflow dataclass only receive contract that has been run. "
                    f"Please run the contract before parse it to the Cashflow dataclass."
                )
            else:
                pass

        # Condition when the contract is a multi-value
        elif isinstance(self.contract, tuple):
            for index, cntr in enumerate(self.contract):
                if cntr._consolidated_cashflow is None:
                    raise CashflowException(
                        f"Cashflow dataclass only receive contracts that has been run. "
                        f"Contract in index {index} has not been run, please run the contract before parse it to the "
                        f"Cashflow dataclass"
                    )
                else:
                    pass
        else:
            raise CashflowException(
                f"Cashflow dataclass only receive psc contract/s."
            )

    def get_single_project_start_end(self):
        """
        Function to parse the self.start_year and self.end_year
        """
        if self.start_year is None:
            self.start_year = self.contract.start_date.year
        else:
            pass

        if self.end_year is None:
            self.end_year = self.contract.end_date.year
        else:
            pass

    def get_multiple_project_start_end(self):
        """
        Function to parse the self.start_year and self.end_year from the given contracts.
        """
        self.start_year = min(np.array([cntr.start_date.year for cntr in self.contract], dtype=int))
        self.end_year = max(np.array([cntr.end_date.year for cntr in self.contract], dtype=int))

    def get_single_project_years(self):
        """
        Function to parse the self.start_year and self.end_year from the given contract.
        """
        self.project_years = self.contract.project_years

    def get_multiple_project_years(self):
        """
        Function to parse the self.project_years from the given contracts.
        """
        start_year_arr = min(np.array([cntr.start_date.year for cntr in self.contract], dtype=int))
        end_year_arr = max(np.array([cntr.end_date.year for cntr in self.contract], dtype=int))
        self.project_years = np.arange(start=start_year_arr, stop=end_year_arr, step=1)

    def check_reference_year(self):
        """
        Function to check the input of reference year.
        """
        if self.start_year is None:
            pass
        else:
            if self.reference_year < self.start_year:
                raise CashflowException(
                    f"The reference year: {self.reference_year}, is less than start_year {self.start_year}. "
                    f"The reference year should be in the range of the start_year and end_year"
                )
            elif self.reference_year > self.end_year:
                raise CashflowException(
                    f"The reference year: {self.reference_year}, is greater than end_year {self.end_year}. "
                    f"The reference year should be in the range of the start_year and end_year"
                )
            else:
                pass

    def get_cashflow_all(self):
        """
        Function to parse self.cashflow from the input of contract/s.
        """
        if isinstance(self.contract, (BaseProject, CostRecovery, GrossSplit, Transition)):
            self.cashflow = self.contract._consolidated_cashflow

        elif isinstance(self.contract, tuple):
            self.parse_multiple_cashflow()

        else:
            raise CashflowException(
                f"The contract type: {type(self.contract)} , is not recognized."
            )

    def parse_multiple_cashflow(self):
        """
        Function to retrieve the cashflow of the given contracts.
        """
        # Initiating the contract_years
        corresponding_year = self.project_years

        # Initiating the cashflow array
        self.cashflow = np.zeros_like(corresponding_year, dtype=float)

        # Iterate through each contract
        for cntr in self.contract:
            # Iterate through each year and cashflow in the contract
            for year, cashflow in zip(cntr.project_years, cntr._consolidated_cashflow):
                if year in corresponding_year:
                    # Find the index of the year in var_year
                    index = np.where(corresponding_year == year)[0][0]
                    # Add the cashflow to the total cashflows
                    self.cashflow[index] += cashflow

    def get_contract_npv(self):
        """
        Function to get the Net Present Value (NPV) of the obtained cashflow.
        """
        # NPV Calculation for SKK Real Terms
        if self.npv_mode == NPVSelection.NPV_SKK_REAL_TERMS:
            ctr_npv = npv_skk_real_terms(cashflow=self.cashflow,
                                         cashflow_years=self.project_years,
                                         discount_rate=self.discount_rate,
                                         reference_year=self.reference_year,
                                         discounting_mode=self.discounting_mode)

        # NPV Calculation for SKK Nominal Terms
        elif self.npv_mode == NPVSelection.NPV_SKK_NOMINAL_TERMS:
            ctr_npv = npv_skk_nominal_terms(cashflow=self.cashflow,
                                            cashflow_years=self.project_years,
                                            discount_rate=self.discount_rate,
                                            discounting_mode=self.discounting_mode)

        # NPV Calculation for Nominal Terms
        elif self.npv_mode == NPVSelection.NPV_NOMINAL_TERMS:
            ctr_npv = npv_nominal_terms(cashflow=self.cashflow,
                                        cashflow_years=self.project_years,
                                        discount_rate=self.discount_rate,
                                        reference_year=self.reference_year,
                                        discounting_mode=self.discounting_mode)

        # NPV Calculation for Real Terms
        elif self.npv_mode == NPVSelection.NPV_REAL_TERMS:
            ctr_npv = npv_real_terms(cashflow=self.cashflow,
                                     cashflow_years=self.project_years,
                                     discount_rate=self.discount_rate,
                                     reference_year=self.reference_year,
                                     inflation_rate=self.inflation_rate,
                                     discounting_mode=self.discounting_mode)

        # NPV Calculation for Point Forwards
        elif self.npv_mode == NPVSelection.NPV_POINT_FORWARD:
            ctr_npv = npv_point_forward(cashflow=self.cashflow,
                                        cashflow_years=self.project_years,
                                        discount_rate=self.discount_rate,
                                        reference_year=self.reference_year,
                                        discounting_mode=self.discounting_mode)

        # Condition when the npv_mode is not recognized
        else:
            raise CashflowException(
                f"The npv_mode: {self.npv_mode} , is not recognized."
                f"The npv_mode input should be in one of NPVSelection."
            )

        return ctr_npv

    def get_contract_irr(self):
        """
        Function to get the Internal Rate of Return (IRR) of the obtained cashflow.
        """
        return irr(cashflow=self.cashflow)

    def get_contract_pot(self):
        """
        Function to get the Pay-Out Time (POT) of the obtained cashflow.
        """
        return pot_psc(cashflow=self.cashflow,
                       cashflow_years=self.project_years,
                       reference_year=self.reference_year)

    def run(self):
        """
        Function to get the cashflow and economic indicator of the given contract/s.
        """
        # Retrieve the cashflow of the contract/s
        self.get_cashflow_all()

        # Retrieve the economic indicator
        self.npv = self.get_contract_npv()
        self.irr = self.get_contract_irr()
        self.pot = self.get_contract_pot()



