"""
Configure a CashFlow class to handle calculations of cashflow indicators.
"""

from dataclasses import dataclass, field
from datetime import date
import numpy as np

from pyscnomics.econ.selection import FluidType
from pyscnomics.econ import indicator


class CashFlowException(Exception):
    """Exception to raise for a misuse of CashFlow class"""

    pass


@dataclass
class CashFlow:
    """
    A class representing a cash flow with various financial indicators.

    Parameters
    ----------
    start_date : date
        The start date of the cash flow.
    end_date : date
        The end date of the cash flow.
    cash : np.ndarray
        An array representing cash flows over time.
    cashed_year : np.ndarray
        An array representing the cashed year of a cashflow.
    cash_allocation : FluidType, optional
        The cash allocation type for cash flows.
        Default is FluidType.ALL.
    """

    start_date: date
    end_date: date
    cash: np.ndarray
    cashed_year: np.ndarray
    cash_allocation: FluidType = field(default=FluidType.ALL)

    # Attributes to be defined later
    project_duration: int = field(default=None, init=False, repr=False)
    project_years: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # Specify project duration and project years, raise error for inappropriate start date
        if self.start_date <= self.end_date:
            self.project_duration = self.end_date.year - self.start_date.year + 1
            self.project_years = np.arange(
                self.start_date.year, self.end_date.year + 1, 1
            )

        else:
            raise CashFlowException(
                f"start date {self.start_date} "
                f"is after the end date: {self.end_date}"
            )

        # Check input data for unequal length of arrays
        if len(self.cash) != len(self.cashed_year):
            raise CashFlowException(
                f"Unequal length of array: "
                f"cash: {len(self.cash)}, "
                f"cash_year: {len(self.cashed_year)}, "
            )

        # Raise an error message: cashed_year is after the end year of the project
        if np.max(self.cashed_year) > self.end_date.year:
            raise CashFlowException(
                f"Cashed year ({np.max(self.cashed_year)}) "
                f"is after the end year of the project ({self.end_date.year})"
            )

        # Raise an error message: cashed_year is before the start year of the project
        if np.min(self.cashed_year) < self.start_date.year:
            raise CashFlowException(
                f"Cashed year ({np.min(self.cashed_year)}) "
                f"is before the start year of the project ({self.start_date.year})"
            )

    def __len__(self):
        return self.project_duration

    def __eq__(self, other):
        # Between two instances of CashFlow
        if isinstance(other, CashFlow):
            return all(
                (
                    self.start_date == other.start_date,
                    self.end_date == other.end_date,
                    self.cash_allocation == other.cash_allocation,
                    np.allclose(self.cash, other.cash),
                    np.allclose(self.cashed_year, other.cashed_year),
                )
            )

        # Between an instance of CashFlow and an integer/float
        elif isinstance(other, (int, float)):
            return np.allclose(sum(self.cash), other)

        else:
            return False

    def __lt__(self, other):
        # Between two instances of CashFlow
        if isinstance(other, CashFlow):
            return np.sum(self.cash) < np.sum(other.cash)

        # Between an instance of CashFlow and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cash) < other

        else:
            raise CashFlowException(
                f"Must compare an instance of CashFlow with another instance of "
                f"CashFlow, an integer, or a float. "
            )

    def __le__(self, other):
        # Between two instances of CashFlow
        if isinstance(other, CashFlow):
            return np.sum(self.cash) <= np.sum(other.cash)

        # Between an instance of CashFlow and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cash) <= other

        else:
            raise CashFlowException(
                f"Must compare an instance of CashFlow with another instance of "
                f"CashFlow, an integer, or a float. "
            )

    def __gt__(self, other):
        # Between two instances of CashFlow
        if isinstance(other, CashFlow):
            return np.sum(self.cash) > np.sum(other.cash)

        # Between an instance of CashFlow and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cash) > other

        else:
            raise CashFlowException(
                f"Must compare an instance of CashFlow with another instance of "
                f"CashFlow, an integer, or a float."
            )

    def __ge__(self, other):
        # Between two instances of CashFlow
        if isinstance(other, CashFlow):
            return np.sum(self.cash) >= np.sum(other.cash)

        # Between an instance of CashFlow and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cash) >= other

        else:
            raise CashFlowException(
                f"Must compare an instance of CashFlow with another instance of "
                f"CashFlow, an integer, or a float."
            )

    def __add__(self, other):
        # Between an instance of CashFlow and another instance of CashFlow
        if isinstance(other, CashFlow):
            start_year = min(self.start_date.year, other.start_date.year)
            start_month = min(self.start_date.month, other.start_date.month)
            start_day = min(self.start_date.day, other.start_date.day)

            end_year = max(self.end_date.year, other.end_date.year)
            end_month = max(self.end_date.month, other.end_date.month)
            end_day = max(self.end_date.day, other.end_date.day)

            return CashFlow(
                start_date=date(year=start_year, month=start_month, day=start_day),
                end_date=date(year=end_year, month=end_month, day=end_day),
                cash=np.concatenate((self.cash, other.cash)),
                cashed_year=np.concatenate((self.cashed_year, other.cashed_year)),
            )

        # Between an instance of CashFlow and an integer/float
        elif isinstance(other, (int, float)):
            return self.allocate() + other

        else:
            raise CashFlowException(
                f"Must add between an instance of CashFlow "
                f"with another instance of CashFlow or an integer/float. "
                f"{other}({other.__class__.__qualname__}) is not "
                f"an instance of Cashflow nor an integer/float."
            )

    def __sub__(self, other):
        # Between an instance of CashFlow and another instance of CashFlow
        if isinstance(other, CashFlow):
            start_year = min(self.start_date.year, other.start_date.year)
            start_month = min(self.start_date.month, other.start_date.month)
            start_day = min(self.start_date.day, other.start_date.day)

            end_year = max(self.end_date.year, other.end_date.year)
            end_month = max(self.end_date.month, other.end_date.month)
            end_day = max(self.end_date.day, other.end_date.day)

            return CashFlow(
                start_date=date(year=start_year, month=start_month, day=start_day),
                end_date=date(year=end_year, month=end_month, day=end_day),
                cash=np.concatenate((self.cash, -other.cash)),
                cashed_year=np.concatenate((self.cashed_year, other.cashed_year)),
            )

        # Between an instance of CashFlow and an integer/float
        elif isinstance(other, (int, float)):
            return self.allocate() - other

        else:
            raise CashFlowException(
                f"Must subtract between an instance of CashFlow "
                f"with another instance of CashFlow or an integer/float. "
                f"{other}({other.__class__.__qualname__}) is not "
                f"an instance of Cashflow nor an integer/float."
            )

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        # Multiplication is allowed only with an integer/float
        if isinstance(other, (int, float)):
            return CashFlow(
                start_date=self.start_date,
                end_date=self.end_date,
                cash=self.cash * other,
                cashed_year=self.cashed_year,
                cash_allocation=self.cash_allocation,
            )

        else:
            raise CashFlowException(
                f"Must multiply with an integer or a float; "
                f"{other}({other.__class__.__qualname__}) is not an integer nor a float."
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        # Between an instance of CashFlow and another instance of CashFlow
        if isinstance(other, CashFlow):
            return np.sum(self.cash) / np.sum(other.cash)

        # Between an instance of CashFlow and an integer/float
        elif isinstance(other, (int, float)):
            # Cannot divide by zero
            if other == 0:
                raise CashFlowException(f"Cannot divide by zero")

            else:
                return CashFlow(
                    start_date=self.start_date,
                    end_date=self.end_date,
                    cash=self.cash / other,
                    cashed_year=self.cashed_year,
                    cash_allocation=self.cash_allocation,
                )

        else:
            raise CashFlowException(
                f"Must divide with an instance of CashFlow, integer or a float; "
                f"{other}({other.__class__.__qualname__}) is not an instance "
                f"of CashFlow nor an integer/float."
            )

    def allocate(self) -> np.ndarray:
        """
        Allocates cash flows over the project duration.

        This method calculates the allocation of cash flows over the project duration
        based on the cash, cashed_year, and project start_year data.

        Returns
        -------
        cash_allocation: np.ndarray
            The cashflow in which each of the associated elements is aligned with
            its corresponding cashed year.

        Notes
        -----
        The calculation procedures are as follows:
        (1) Generate an array depicting the difference between the cashed_year and the start_year
            of the project,
        (2) Calculate the number of occurrence of each of the elements induced from (1)
            using method numpy.bincount(),
        (3) Multiply the results of (2) with the corresponding cash data to obtain an array
            of cash that is aligned with the corresponding year. To do so, we use the argument
            'weights' of numpy.bincount() method.
        (4) If the length of cost_allocation does not equal to the project duration, modify
            cash_allocation array by adding zeros to the last elements as many as the difference
            between the initial length of cash_allocation and the project duration.
        """
        cash_allocation = np.bincount(
            self.cashed_year - self.start_date.year, weights=self.cash
        )
        zeros = np.zeros(self.project_duration - len(cash_allocation))

        return np.concatenate((cash_allocation, zeros))

    def npv(self, disc_rate: float = 0.1):
        """
        Calculate the net present value (NPV) of the cash flows.

        Parameters
        ----------
        disc_rate : float, optional
            The discount rate used in financial calculations. Default is 0.1 (10%).

        Returns
        -------
        float
            The calculated NPV value.
        """
        return indicator.npv(cashflow=self.cash, disc_rate=disc_rate)

    def xnpv(self, disc_rate: float = 0.1):
        """
        Calculate the NPV of cash flows at irregular intervals.

        Parameters
        ----------
        disc_rate : float, optional
            The discount rate used in financial calculations. Default is 0.1 (10%).

        Returns
        -------
        float
            The calculated NPV value at irregular intervals.
        """
        return indicator.xnpv(
            cashflow=self.cash,
            start_date=self.start_date,
            end_date=self.end_date,
            disc_rate=disc_rate,
        )

    def irr(self):
        """
        Calculate the internal rate of return (IRR) of the cash flows.

        Returns
        -------
        float
            The calculated IRR value.
        """
        return indicator.irr(cashflow=self.cash)

    def xirr(self):
        """
        Calculate the IRR of cash flows at irregular intervals.

        Returns
        -------
        float
            The calculated IRR value at irregular intervals.
        """
        return indicator.xirr(
            cashflow=self.cash,
            start_date=self.start_date,
            end_date=self.end_date,
        )

    def pot(self):
        """
        Calculate the potential investment return of the cash flows.
        (pay out time)

        Returns
        -------
        float
            The calculated potential investment return value.
        """
        return indicator.pot(cashflow=self.cash)












