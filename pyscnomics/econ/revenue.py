import numpy as np
from dataclasses import dataclass, field
from pyscnomics.econ.selection import FluidType
from pyscnomics.econ.costs import Tangible, Intangible, OPEX


@dataclass
class Lifting:

    """
    Create an object that represents lifting.

    Parameters
    ----------

    """

    start_year: int
    end_year: int
    lifting_rate: np.ndarray
    price: np.ndarray
    fluid_type: FluidType = field(default=FluidType.OIL)
    ghv: np.ndarray = field(default=None, repr=False)
    prod_rate: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):

        # Condition when user does not insert production rate data
        if self.prod_rate is None:
            self.prod_rate = self.lifting_rate.copy()

        # Condition when user does not insert GHV data;
        # The default value of GHV is set to unity
        if self.ghv is None:
            self.ghv = np.ones(len(self.prod_rate))

        # Initial check for inappropriate input data;
        # Raise a "ValueError" for any inappropriate data
        arr_length = self.lifting_rate.shape[0]

        if not all(
                len(arr) == arr_length
                for arr in [
                    self.price,
                    self.ghv,
                    self.prod_rate
                ]
        ):
            raise ValueError(
                f'Inequal length of array: lifting_rate: {len(self.lifting_rate)},'
                f' ghv: {len(self.ghv)},'
                f' production: {len(self.prod_rate)}'
            )

        # Define an attribute depicting project duration;
        # Raise a "ValueError" if start_year is after then end_year
        if self.end_year > self.start_year:
            self.project_duration = self.end_year - self.start_year + 1

        else:
            raise ValueError(
                f"start year {self.start_year} is after the end year: {self.end_year}"
            )

        # Specify an error condition when project duration is less than the length of production data
        if self.project_duration < len(self.prod_rate):
            raise ValueError(
                f'Length of project duration: ({self.project_duration})'
                f' is less than the length of production data: ({len(self.prod_rate)})'
            )

    def revenue(self) -> np.ndarray:

        """
        Calculate the revenue of a particular fluid type.

        Returns
        -------
        rev: np.ndarray
            The revenue of a particular fluid type.
        """

        # Calculate revenue = lifting rate * price * ghv
        rev = self.lifting_rate * self.price * self.ghv

        # When project duration is longer than the length of production data,
        # assign the revenue of the suplementary years with zeros
        if self.project_duration > len(self.prod_rate):
            add_zeros = np.zeros(int(self.project_duration - len(self.prod_rate)))
            rev = np.concatenate((rev, add_zeros))

        return rev

    def __eq__(self, other):
        return all((
            self.fluid_type == other.fluid_type,
            self.start_year == other.start_year,
            self.end_year == other.end_year,
            np.allclose(self.lifting_rate, other.lifting_rate),
            np.allclose(self.price, other.price),
            np.allclose(self.ghv, other.ghv),
            np.allclose(self.prod_rate, other.prod_rate)
        ))

    def __lt__(self, other):
        return np.sum(self.revenue()) < np.sum(other.revenue())

    def __le__(self, other):
        return np.sum(self.revenue()) <= np.sum(other.revenue())

    def __gt__(self, other):
        return np.sum(self.revenue()) > np.sum(other.revenue())

    def __ge__(self, other):
        return np.sum(self.revenue()) >= np.sum(other.revenue())

    def __add__(self, other):

        if isinstance(other, Lifting):

            start_year = min(self.start_year, other.start_year)
            end_year = max(self.end_year, other.end_year)

            self_revenue = self.revenue().copy()
            other_revenue = other.revenue().copy()

            if len(self_revenue) < end_year - start_year + 1:
                self_revenue.resize(end_year - start_year + 1, refcheck=False)
                if self.start_year > other.start_year:
                    self_revenue = np.roll(self_revenue, (self.start_year - start_year))

            if len(other_revenue) < end_year - start_year + 1:
                other_revenue.resize(end_year - start_year + 1, refcheck=False)
                if other.start_year > self.start_year:
                    other_revenue = np.roll(other_revenue, (other.start_year - start_year))

            return self_revenue + other_revenue

        elif isinstance(other, (int, float, np.ndarray)):
            return self.revenue() + other

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return self.revenue() * other

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return other * self.revenue()

    def __truediv__(self, other):
        if isinstance(other, Lifting):
            return np.sum(self.revenue()) / np.sum(other.revenue())

        elif isinstance(other, (int, float)):
            return self.revenue() / other

    def __sub__(self, other):
        if isinstance(other, Lifting):

            start_year = min(self.start_year, other.start_year)
            end_year = max(self.end_year, other.end_year)

            self_revenue = self.revenue().copy()
            other_revenue = other.revenue().copy()

            if len(self_revenue) < end_year - start_year + 1:
                self_revenue.resize(end_year - start_year + 1, refcheck=False)
                if self.start_year > other.start_year:
                    self_revenue = np.roll(self_revenue, (self.start_year - start_year))

            if len(other_revenue) < end_year - start_year + 1:
                other_revenue.resize(end_year - start_year + 1, refcheck=False)
                if other.start_year > self.start_year:
                    other_revenue = np.roll(other_revenue, (other.start_year - start_year))

            return self_revenue - other_revenue

        elif isinstance(other, (int, float, np.ndarray)):
            return self.revenue() - other

        elif isinstance(other, Tangible):
            return self.revenue() - other.total_depreciation_rate()

        elif isinstance(other, Intangible):
            raise NotImplementedError

        elif isinstance(other, OPEX):
            raise NotImplementedError






