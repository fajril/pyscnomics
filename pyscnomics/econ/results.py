from dataclasses import dataclass, field
from datetime import date

import numpy as np

from pyscnomics.econ.selection import FluidType
from pyscnomics.econ import indicator


@dataclass
class CashFlow:
    """A class representing a cash flow with various financial indicators.

    Parameters
    ----------
    start_date : date
        The start date of the cash flow.
    end_date : date
        The end date of the cash flow.
    cash : np.ndarray
        An array representing the cash flows over time.
    disc_rate : float, optional
        The discount rate used in financial calculations.
        Default is 0.1 (10%).
    cash_allocation : FluidType, optional
        The cash allocation type for cash flows.
        Default is FluidType.ALL.
    """

    start_date: date
    end_date: date
    cash: np.ndarray
    disc_rate: float = field(default=0.1)
    cash_allocation: FluidType = field(default=FluidType.ALL)

    def __post_init__(self):
        if self.start_date < self.end_date:
            self.year_duration = self.end_date.year() - self.start_date.year() + 1
        else:
            raise ValueError(
                f"start date {self.start_date} "
                f"is after the end date: {self.end_date}"
            )

    def __len__(self):
        return self.year_duration

    def __eq__(self, other):
        if isinstance(other, CashFlow):
            return all(
                (
                    self.start_date == other.start_date,
                    self.end_date == other.end_date,
                    self.cash_allocation == other.cash_allocation,
                    np.allclose(self.cash, other.cash),
                )
            )
        if isinstance(other, (int, float)):
            return np.allclose(sum(self.cash), other)
        return False

    def __lt__(self, other):
        if isinstance(other, CashFlow):
            return np.sum(self.cash) < np.sum(other.cash)
        if isinstance(other, (int, float)):
            return np.sum(self.cash) < other
        return False

    def __le__(self, other):
        if isinstance(other, CashFlow):
            return np.sum(self.cash) <= np.sum(other.cash)
        if isinstance(other, (int, float)):
            return np.sum(self.cash) <= other
        return False

    def __gt__(self, other):
        if isinstance(other, CashFlow):
            return np.sum(self.cash) > np.sum(other.cash)
        if isinstance(other, (int, float)):
            return np.sum(self.cash) > other
        return False

    def __ge__(self, other):
        if isinstance(other, CashFlow):
            return np.sum(self.cash) >= np.sum(other.cash)
        if isinstance(other, (int, float)):
            return np.sum(self.cash) >= other
        return False

    def __add__(self, other):
        # TODO: implementation similar to revenue addition
        raise NotImplementedError()

    def __sub__(self, other):
        # TODO: implementation similar to revenue substaction
        raise NotImplementedError()

    def npv(self):
        """Calculate the net present value (NPV) of the cash flows.

        Returns
        -------
        float
            The calculated NPV value.
        """
        return indicator.npv(cashflow=self.cash, disc_rate=self.disc_rate)

    def xnpv(self):
        """Calculate the NPV of cash flows at irregular intervals.

        Returns
        -------
        float
            The calculated NPV value at irregular intervals.
        """
        return indicator.xnpv(
            cashflow=self.cash,
            start_date=self.start_date,
            end_date=self.end_date,
            disc_rate=self.disc_rate,
        )

    def irr(self):
        """Calculate the internal rate of return (IRR) of the cash flows.

        Returns
        -------
        float
            The calculated IRR value.
        """
        return indicator.irr(cashflow=self.cash)

    def xirr(self):
        """Calculate the IRR of cash flows at irregular intervals.

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
        """Calculate the potential investment return of the cash flows.
        (pay out time)

        Returns
        -------
        float
            The calculated potential investment return value.
        """
        return indicator.pot(cashflow=self.cash)
