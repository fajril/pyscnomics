"""
Prepare and classify cost data based on its components. The associated cost components are:
(1) Tangible,
(2) Intangible,
(3) OPEX,
(4) ASR
"""

import numpy as np
from dataclasses import dataclass, field

import pyscnomics.econ.depreciation as depr
from pyscnomics.tools.helper import (
    apply_inflation,
    apply_vat_pdri,
    apply_lbt,
    apply_pdrd,
    apply_cost_modification,
)
from pyscnomics.econ.selection import (
    FluidType,
    DeprMethod,
    YearReference,
)


class TangibleException(Exception):
    """Exception to be raised if class Tangible is misused"""

    pass


class IntangibleException(Exception):
    """Exception to be raised if class Intangible is misused"""

    pass


class OPEXException(Exception):
    """Exception to be raised if class OPEX is misused"""

    pass


class ASRException(Exception):
    """Exception to be raised if class ASR is misused"""

    pass


@dataclass
class Tangible:
    """
    Manages a tangible asset.

    Parameters
    ----------
    start_year : int
        The start year of the project.
    end_year : int
        The end year of the project.
    cost : numpy.ndarray
        An array representing the cost of a tangible asset.
    expense_year : numpy.ndarray
        An array representing the expense year of a tangible asset.
    pis_year : numpy.ndarray, optional
        An array representing the PIS year of a tangible asset.
    cost_allocation : FluidType
        A list representing the cost allocation of a tangible asset.
    salvage_value : numpy.ndarray, optional
        An array representing the salvage value of a tangible asset.
    useful_life : numpy.ndarray, optional
        An array representing the useful life of a tangible asset.
    depreciation_factor: np.ndarray, optional
        The value of depreciation factor to be used in PSC_DB depreciation method.
        Default value is 0.5 for the entire project duration.
    is_ic_applied: bool, optional
        Whether investment credit is applied or not. Default value is False.
    vat_portion: float | int, optional
        A fraction of tangible cost susceptible for VAT/PPN.
    pdri_portion: float | int, optional
        A fraction of tangible cost susceptible for PDRI.
    description: list[str]
        A list of string description regarding the associated tangible cost.
    """

    start_year: int
    end_year: int
    cost: np.ndarray
    expense_year: np.ndarray
    cost_allocation: FluidType = field(default=FluidType.OIL)
    pis_year: np.ndarray = field(default=None, repr=False)
    salvage_value: np.ndarray = field(default=None, repr=False)
    useful_life: np.ndarray = field(default=None, repr=False)
    depreciation_factor: np.ndarray = field(default=None, repr=False)
    is_ic_applied: bool = field(default=False)
    vat_portion: float | int = field(default=1.0, repr=False)
    pdri_portion: float | int = field(default=1.0, repr=False)
    description: list[str] = field(default=None)

    # Attribute to be defined later on
    project_duration: int = field(default=None, init=False, repr=False)
    project_years: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # Check for inappropriate start and end year project
        if self.end_year >= self.start_year:
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise TangibleException(
                f"start year {self.start_year} "
                f"is after the end year: {self.end_year}"
            )

        # Configure attribute "description"
        if self.description is None:
            self.description = [" " for _ in range(len(self.cost))]

        if self.description is not None:
            if len(self.description) != len(self.cost):
                raise TangibleException(
                    f"Unequal length of array: "
                    f"description: {len(self.description)}, "
                    f"cost: {len(self.cost)}"
                )

        # User does not provide pis_year data
        if self.pis_year is None:
            self.pis_year = self.expense_year.copy()

        # User does not provide salvage_value data
        if self.salvage_value is None:
            self.salvage_value = np.zeros(len(self.cost))

        # User does not provide useful_life data
        if self.useful_life is None:
            self.useful_life = np.repeat(5, len(self.cost))

        # User does not provide depreciation_factor data
        if self.depreciation_factor is None:
            self.depreciation_factor = np.repeat(0.5, len(self.cost))

        # Check input data for unequal length
        arr_length = len(self.cost)

        if not all(
            len(arr) == arr_length
            for arr in [
                self.expense_year,
                self.pis_year,
                self.salvage_value,
                self.useful_life,
                self.depreciation_factor,
            ]
        ):
            raise TangibleException(
                f"Unequal length of array: "
                f"cost: {len(self.cost)}, "
                f"expense_year: {len(self.expense_year)}, "
                f"pis_year: {len(self.pis_year)}, "
                f"salvage_value: {len(self.salvage_value)}, "
                f"useful_life: {len(self.useful_life)}, "
                f"depreciation_factor: {len(self.depreciation_factor)}"
            )

        # Raise an error message: expense year is after the end year of the project
        if np.max(self.expense_year) > self.end_year:
            raise TangibleException(
                f"Expense year ({np.max(self.expense_year)}) "
                f"is after the end year of the project ({self.end_year})"
            )

        # Raise an error message: expense year is before the start year of the project
        if np.min(self.expense_year) < self.start_year:
            raise TangibleException(
                f"Expense year ({np.min(self.expense_year)}) "
                f"is before the start year of the project ({self.start_year})"
            )

    def expenditures(
        self,
        year_ref: YearReference = YearReference.EXPENSE_YEAR,
        inflation_rate: float | int = 0.0,
        vat_rate: np.ndarray | float | int = 0.0,
        vat_discount: np.ndarray | float | int = 0.0,
        pdri_rate: np.ndarray | float | int = 0.0,
        pdri_discount: np.ndarray | float | int = 0.0,
        lbt_discount: np.ndarray | float | int = 0.0,
        pdrd_discount: np.ndarray | float | int = 0.0,
    ):
        """
        Calculate tangible expenditures per year.

        This method calculates the tangible expenditures per year
        based on the expense year and cost data provided.

        Parameters
        ----------
        year_ref : YearReference, optional
            Reference year for expenses (default is YearReference.EXPENSE_YEAR).
        inflation_rate : float or int, optional
            A constant depicting the escalation/inflation rate (default is 0).
        vat_rate : np.ndarray, float, or int, optional
            Value Added Tax (VAT) rate(s) as a multiplier (default is 0).
        vat_discount : np.ndarray, float, or int, optional
            Discount(s) applied to VAT (default is 0).
        pdri_rate : np.ndarray, float, or int, optional
            The PDRI rate(s) as a multiplier (default is 0).
        pdri_discount : np.ndarray, float, or int, optional
            Discount applied to PDRI (default is 0).
        lbt_discount : np.ndarray, float, or int, optional
            Land and Building Tax (LBT) discount(s) as a multiplier (default is 0).
        pdrd_discount : np.ndarray, float, or int, optional
            PDRD discount as a multiplier (default is 0).

        Returns
        -------
        expenses: np.ndarray
            An array depicting the tangible expenditures each year, taking into
            account the inflation, VAT/PPN, PDRI, LBT/PBB, and PDRD.

        Notes
        -----
        This method calculates tangible expenditures while considering various economic factors
        such as inflation, VAT, PDRI, LBT, and PDRD. It uses decorators to apply these factors
        to the core calculation. In the core calculations:
        (1) Function np.bincount() is used to align the cost elements according
            to its corresponding expense year,
        (2) If len(expenses) < project_duration, then add the remaining elements
            with zeros.
        """

        @apply_pdrd(pdrd_discount=pdrd_discount)
        @apply_lbt(lbt_discount=lbt_discount)
        @apply_vat_pdri(
            vat_portion=self.vat_portion,
            vat_rate=vat_rate,
            vat_discount=vat_discount,
            pdri_portion=self.pdri_portion,
            pdri_rate=pdri_rate,
            pdri_discount=pdri_discount,
        )
        @apply_inflation(inflation_rate=inflation_rate)
        def _expenditures() -> np.ndarray:
            if year_ref == YearReference.EXPENSE_YEAR:
                expenses = np.bincount(
                    self.expense_year - self.start_year, weights=self.cost
                )

            else:
                expenses = np.bincount(
                    self.pis_year - self.start_year, weights=self.cost
                )

            zeros = np.zeros(self.project_duration - len(expenses))
            expenses = np.concatenate((expenses, zeros))
            return expenses

        return _expenditures()

    def total_depreciation_rate(
        self,
        depr_method: DeprMethod = DeprMethod.PSC_DB,
        decline_factor: float | int = 2,
        inflation_rate: float | int = 0.0,
        vat_rate: np.ndarray | float | int = 0.0,
        vat_discount: np.ndarray | float | int = 0.0,
        pdri_rate: np.ndarray | float | int = 0.0,
        pdri_discount: np.ndarray | float | int = 0.0,
        lbt_discount: np.ndarray | float | int = 0.0,
        pdrd_discount: np.ndarray | float | int = 0.0,
    ) -> tuple:
        """
        Calculate total depreciation charge and undepreciated asset value based on various parameters.

        Parameters
        -----------
        depr_method : DeprMethod, optional
            The depreciation method to use (default is DeprMethod.PSC_DB).
        decline_factor : float | int, optional
            The decline factor used for declining balance depreciation (default is 2).
        inflation_rate : float | int, optional
            The annual inflation rate as a decimal or integer (default is 0.0).
        vat_rate : np.ndarray | float | int, optional
            An array or single value representing the VAT rate(s)
            as a decimal or integer (default is 0.0).
        vat_discount : np.ndarray | float | int, optional
            An array or single value representing the VAT discount(s)
            as a decimal or integer (default is 0.0).
        pdri_rate : np.ndarray | float | int, optional
            An array or single value representing the PDRI rate(s)
            as a decimal or integer (default is 0.0).
        pdri_discount : np.ndarray | float | int, optional
            An array or single value representing the PDRI discount(s)
            as a decimal or integer (default is 0.0).
        lbt_discount : np.ndarray | float | int, optional
            An array or single value representing the Land and Building Tax (LBT) discount(s)
            as a decimal or integer (default is 0.0).
        pdrd_discount : np.ndarray | float | int, optional
            An array or single value representing PDRD discount(s)
            as a decimal or integer (default is 0.0).

        Returns
        --------
        tuple
            A tuple containing two numpy arrays:
            (1) Total depreciation charges for each period.
            (2) The undepreciated asset value at the end of the analysis.

        Notes
        ------
        (1) This method calculates depreciation charges based on the specified
            depreciation method.
        (2) Prior to the core calculations, attribute 'cost' is modified by
            taking into account various economic factors such as inflation,
            VAT/PPN, PDRI, LBT discount, and PDRD discount.
        (3) The depreciation charges are aligned with the corresponding periods
            based on the pis_year (or expense_year).
        """

        # Configure the modified cost
        cost_modified = apply_cost_modification(
            start_year=self.start_year,
            cost=self.cost,
            expense_year=self.expense_year,
            inflation_rate=inflation_rate,
            vat_portion=self.vat_portion,
            vat_rate=vat_rate,
            vat_discount=vat_discount,
            pdri_portion=self.pdri_portion,
            pdri_rate=pdri_rate,
            pdri_discount=pdri_discount,
            lbt_discount=lbt_discount,
            pdrd_discount=pdrd_discount,
        )

        # Straight line
        if depr_method == DeprMethod.SL:
            depreciation_charge = np.asarray(
                [
                    depr.straight_line_depreciation_rate(
                        cost=c,
                        salvage_value=sv,
                        useful_life=ul,
                        depreciation_len=self.project_duration,
                    )
                    for c, sv, ul in zip(
                        cost_modified,
                        self.salvage_value,
                        self.useful_life,
                    )
                ]
            )

        # Declining balance/double declining balance
        if depr_method == DeprMethod.DB:
            depreciation_charge = np.asarray(
                [
                    depr.declining_balance_depreciation_rate(
                        cost=c,
                        salvage_value=sv,
                        useful_life=ul,
                        decline_factor=decline_factor,
                        depreciation_len=self.project_duration,
                    )
                    for c, sv, ul in zip(
                        cost_modified,
                        self.salvage_value,
                        self.useful_life,
                    )
                ]
            )

        # PSC_DB
        if depr_method == DeprMethod.PSC_DB:
            depreciation_charge = np.asarray(
                [
                    depr.psc_declining_balance_depreciation_rate(
                        cost=c,
                        depreciation_factor=dr,
                        useful_life=ul,
                        depreciation_len=self.project_duration,
                    )
                    for c, dr, ul in zip(
                        cost_modified,
                        self.depreciation_factor,
                        self.useful_life,
                    )
                ]
            )

        # The relative difference of pis_year and start_year
        shift_indices = self.pis_year - self.start_year

        # Modify depreciation_charge so that expenditures are aligned with
        # the corresponding pis_year (or expense_year)
        depreciation_charge = np.asarray(
            [
                np.concatenate((np.zeros(i), row[:-i])) if i > 0 else row
                for row, i in zip(depreciation_charge, shift_indices)
            ]
        )

        total_depreciation_charge = depreciation_charge.sum(axis=0)
        undepreciated_asset = np.sum(cost_modified) - np.sum(total_depreciation_charge)

        return total_depreciation_charge, undepreciated_asset

    def total_depreciation_book_value(
        self,
        depr_method: DeprMethod = DeprMethod.PSC_DB,
        decline_factor: float = 2,
        inflation_rate: float | int = 0.0,
        vat_rate: np.ndarray | float | int = 0.0,
        vat_discount: np.ndarray | float | int = 0.0,
        pdri_rate: np.ndarray | float | int = 0.0,
        pdri_discount: np.ndarray | float | int = 0.0,
        lbt_discount: np.ndarray | float | int = 0.0,
        pdrd_discount: np.ndarray | float | int = 0.0,
    ) -> np.ndarray:
        """
        Calculate the total book value of depreciation for the asset.

        Parameters
        ----------
        depr_method : DeprMethod, optional
            The depreciation method to use (default is DeprMethod.PSC_DB).
        decline_factor : float, optional
            The decline factor used for declining balance depreciation (default is 2).
        inflation_rate : float | int, optional
            The annual inflation rate as a decimal or integer (default is 0.0).
        vat_rate : np.ndarray | float | int, optional
            An array or single value representing the VAT rate(s)
            as a decimal or integer (default is 0.0).
        vat_discount : np.ndarray | float | int, optional
            An array or single value representing the VAT discount(s)
            as a decimal or integer (default is 0.0).
        pdri_rate : np.ndarray | float | int, optional
            An array or single value representing the PDRI rate(s)
            as a decimal or integer (default is 0.0).
        pdri_discount : np.ndarray | float | int, optional
            An array or single value representing the PDRI discount(s)
            as a decimal or integer (default is 0.0).
        lbt_discount : np.ndarray | float | int, optional
            An array or single value representing the Land and Building Tax (LBT/PBB)
            discount(s) as a decimal or integer (default is 0.0).
        pdrd_discount : np.ndarray | float | int, optional
            An array or single value representing the PDRD discount(s)
            as a decimal or integer (default is 0.0).

        Returns
        -------
        np.ndarray
            An array containing the cumulative book value of depreciation for each period.

        Notes
        -----
        (1) This method calculates the cumulative book value of depreciation for the asset
            based on the specified depreciation method and other parameters.
        (2) The cumulative book value is obtained by subtracting the cumulative
            depreciation charges from the cumulative expenditures.
        """

        # Calculate total depreciation charge from method total_depreciation_rate
        depreciation_charge = self.total_depreciation_rate(
            depr_method=depr_method,
            decline_factor=decline_factor,
            inflation_rate=inflation_rate,
            vat_rate=vat_rate,
            vat_discount=vat_discount,
            pdri_rate=pdri_rate,
            pdri_discount=pdri_discount,
            lbt_discount=lbt_discount,
            pdrd_discount=pdrd_discount,
        )[0]

        return np.cumsum(self.expenditures()) - np.cumsum(depreciation_charge)

    def __len__(self):
        return self.project_duration

    def __eq__(self, other):
        # Between two instances of Tangible
        if isinstance(other, Tangible):
            return all(
                (
                    np.allclose(self.cost, other.cost),
                    np.allclose(self.expense_year, other.expense_year),
                    np.allclose(self.pis_year, other.pis_year),
                    np.allclose(self.salvage_value, other.salvage_value),
                    np.allclose(self.useful_life, other.useful_life),
                    np.allclose(self.depreciation_factor, other.depreciation_factor),
                    self.cost_allocation == other.cost_allocation,
                    self.is_ic_applied == other.is_ic_applied,
                    self.vat_portion == other.vat_portion,
                    self.pdri_portion == other.pdri_portion,
                )
            )

        # Between an instance of Tangible and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) == other

        else:
            return False

    def __lt__(self, other):
        # Between an instance of Tangible with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) < np.sum(other.cost)

        # Between an instance of Tangible and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) < other

        else:
            raise TangibleException(
                f"Must compare an instance of Tangible with another instance of "
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float"
            )

    def __le__(self, other):
        # Between an instance of Tangible with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) <= np.sum(other.cost)

        # Between an instance of Tangible and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) <= other

        else:
            raise TangibleException(
                f"Must compare an instance of Tangible with another instance of "
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float"
            )

    def __gt__(self, other):
        # Between an instance of Tangible with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) > np.sum(other.cost)

        # Between an instance of Tangible and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) > other

        else:
            raise TangibleException(
                f"Must compare an instance of Tangible with another instance of "
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float"
            )

    def __ge__(self, other):
        # Between an instance of Tangible with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) >= np.sum(other.cost)

        # Between an instance of Tangible and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) >= other

        else:
            raise TangibleException(
                f"Must compare an instance of Tangible with another instance of "
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float"
            )

    def __add__(self, other):
        # Only allows addition between an instance of Tangible and another instance of Tangible
        if isinstance(other, Tangible):
            if (
                self.cost_allocation != other.cost_allocation
                or self.is_ic_applied != other.is_ic_applied
                or self.vat_portion != other.vat_portion
                or self.pdri_portion != other.pdri_portion
            ):
                raise TangibleException(
                    f"Cost allocation/VAT portion/PDRI portion is not equal. "
                    f"Cost allocation: {self.cost_allocation} vs. {other.cost_allocation}, "
                    f"Is IC applied: {self.is_ic_applied} vs. {other.is_ic_applied}, "
                    f"VAT portion: {self.vat_portion} vs. {other.vat_portion}, "
                    f"PDRI portion: {self.pdri_portion} vs. {other.pdri_portion}."
                )

            else:
                combined_start_year = min(self.start_year, other.start_year)
                combined_end_year = max(self.end_year, other.end_year)
                combined_description = self.description + other.description

                return Tangible(
                    start_year=combined_start_year,
                    end_year=combined_end_year,
                    cost=np.concatenate((self.cost, other.cost)),
                    expense_year=np.concatenate(
                        (self.expense_year, other.expense_year)
                    ),
                    pis_year=np.concatenate((self.pis_year, other.pis_year)),
                    salvage_value=np.concatenate(
                        (self.salvage_value, other.salvage_value)
                    ),
                    useful_life=np.concatenate((self.useful_life, other.useful_life)),
                    depreciation_factor=np.concatenate(
                        (self.depreciation_factor, other.depreciation_factor)
                    ),
                    cost_allocation=self.cost_allocation,
                    is_ic_applied=self.is_ic_applied,
                    vat_portion=self.vat_portion,
                    pdri_portion=self.pdri_portion,
                    description=combined_description,
                )

        else:
            raise TangibleException(
                f"Must add an instance of Tangible with "
                f"another instance of Tangible. "
                f"{other}({other.__class__.__qualname__}) is not "
                f"an instance of Tangible."
            )

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        # Only allows subtraction between an instance of Tangible and another instance of Tangible
        if isinstance(other, Tangible):
            if (
                self.cost_allocation != other.cost_allocation
                or self.is_ic_applied != other.is_ic_applied
                or self.vat_portion != other.vat_portion
                or self.pdri_portion != other.pdri_portion
            ):
                raise TangibleException(
                    f"Cost allocation/VAT portion/PDRI portion is not equal. "
                    f"Cost allocation: {self.cost_allocation} vs. {other.cost_allocation}, "
                    f"Is IC applied: {self.is_ic_applied} vs. {other.is_ic_applied}, "
                    f"VAT portion: {self.vat_portion} vs. {other.vat_portion}, "
                    f"PDRI portion: {self.pdri_portion} vs. {other.pdri_portion}."
                )

            else:
                combined_start_year = min(self.start_year, other.start_year)
                combined_end_year = max(self.end_year, other.end_year)
                combined_description = self.description + other.description

                return Tangible(
                    start_year=combined_start_year,
                    end_year=combined_end_year,
                    cost=np.concatenate((self.cost, -other.cost)),
                    expense_year=np.concatenate(
                        (self.expense_year, other.expense_year)
                    ),
                    pis_year=np.concatenate((self.pis_year, other.pis_year)),
                    salvage_value=np.concatenate(
                        (self.salvage_value, other.salvage_value)
                    ),
                    useful_life=np.concatenate((self.useful_life, other.useful_life)),
                    depreciation_factor=np.concatenate(
                        (self.depreciation_factor, other.depreciation_factor)
                    ),
                    cost_allocation=self.cost_allocation,
                    is_ic_applied=self.is_ic_applied,
                    vat_portion=self.vat_portion,
                    pdri_portion=self.pdri_portion,
                    description=combined_description,
                )

        else:
            raise TangibleException(
                f"Must subtract between an instance of Tangible "
                f"with another instance of Tangible. "
                f"{other}({other.__class__.__qualname__}) is not "
                f"an instance of Tangible."
            )

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        # Multiplication is allowed only with an integer/a float
        if isinstance(other, (int, float)):
            return Tangible(
                start_year=self.start_year,
                end_year=self.end_year,
                cost=self.cost * other,
                expense_year=self.expense_year.copy(),
                pis_year=self.pis_year.copy(),
                salvage_value=self.salvage_value.copy(),
                useful_life=self.useful_life.copy(),
                depreciation_factor=self.depreciation_factor.copy(),
                cost_allocation=self.cost_allocation,
                is_ic_applied=self.is_ic_applied,
                vat_portion=self.vat_portion,
                pdri_portion=self.pdri_portion,
                description=self.description,
            )

        else:
            raise TangibleException(
                f"Must multiply with an integer or a float; "
                f"{other} is of {other.__class__.__qualname__} datatype."
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        # Between an instance of Tangible with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) / np.sum(other.cost)

        # Between an instance of Tangible and an integer/float
        elif isinstance(other, (int, float)):
            # Cannot divide with zero
            if other == 0:
                raise TangibleException(f"Cannot divide with zero")

            else:
                return Tangible(
                    start_year=self.start_year,
                    end_year=self.end_year,
                    cost=self.cost / other,
                    expense_year=self.expense_year.copy(),
                    pis_year=self.pis_year.copy(),
                    salvage_value=self.salvage_value.copy(),
                    useful_life=self.useful_life.copy(),
                    cost_allocation=self.cost_allocation,
                    vat_portion=self.vat_portion,
                    pdri_portion=self.pdri_portion,
                    description=self.description,
                )

        else:
            raise TangibleException(
                f"Must divide with an instance of Tangible/Intangible/OPEX/ASR, integer or a float; "
                f"{other}({other.__class__.__qualname__}) is not an instance "
                f"of Tangible/Intangible/OPEX/ASR nor an integer nor a float."
            )


@dataclass
class Intangible:
    """
    Manages an intangible asset.

    Parameters
    ----------
    start_year : int
        The start year of the project.
    end_year : int
        The end year of the project.
    cost : numpy.ndarray
        An array representing the cost of an intangible asset.
    expense_year : numpy.ndarray
        An array representing the expense year of an intangible asset.
    cost_allocation : FluidType
        A string depicting the cost allocation of an intangible asset.
    vat_portion: float | int
        A fraction of intangible cost susceptible for VAT/PPN.
    pdri_portion: float | int
        A fraction of intangible cost susceptible for PDRI.
    description: list[str]
        A list of string description regarding the associated intangible cost.
    """

    start_year: int
    end_year: int
    cost: np.ndarray
    expense_year: np.ndarray
    cost_allocation: FluidType = field(default=FluidType.OIL)
    vat_portion: float | int = field(default=1.0, repr=False)
    pdri_portion: float | int = field(default=1.0, repr=False)
    description: list[str] = field(default=None)

    # Attribute to be defined later on
    project_duration: int = field(default=None, init=False, repr=False)
    project_years: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # Check for inappropriate start and end year project
        if self.end_year >= self.start_year:
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise IntangibleException(
                f"start year {self.start_year} "
                f"is after the end year {self.end_year}"
            )

        # Configure attribute "description"
        if self.description is None:
            self.description = [" " for _ in range(len(self.cost))]

        if self.description is not None:
            if len(self.description) != len(self.cost):
                raise IntangibleException(
                    f"Unequal length of array: "
                    f"description: {len(self.description)}, "
                    f"cost: {len(self.cost)}"
                )

        # Check input data for unequal length
        if len(self.expense_year) != len(self.cost):
            raise IntangibleException(
                f"Unequal length of array: "
                f"cost: {len(self.cost)}, "
                f"expense_year: {len(self.expense_year)}"
            )

        # Raise an error message: expense year is after the end year of the project
        if np.max(self.expense_year) > self.end_year:
            raise IntangibleException(
                f"Expense year ({np.max(self.expense_year)}) "
                f"is after the end year of the project ({self.end_year})"
            )

        # Raise an error message: expense year is before the start year of the project
        if np.min(self.expense_year) < self.start_year:
            raise IntangibleException(
                f"Expense year ({np.min(self.expense_year)}) "
                f"is before the start year of the project ({self.start_year})"
            )

    def expenditures(
        self,
        inflation_rate: float | int = 0.0,
        vat_rate: np.ndarray | float | int = 0.0,
        vat_discount: np.ndarray | float | int = 0.0,
        pdri_rate: np.ndarray | float | int = 0.0,
        pdri_discount: np.ndarray | float | int = 0.0,
        lbt_discount: np.ndarray | float | int = 0.0,
        pdrd_discount: np.ndarray | float | int = 0.0,
    ) -> np.ndarray:
        """
        Calculate intangible expenditures per year.

        This method calculates the intangible expenditures per year
        based on the expense year and cost data provided.

        Parameters:
        -----------
        inflation_rate : float or int, optional
            The annual inflation rate as a decimal (default is 0).
        vat_rate : numpy.ndarray or float or int, optional
            The VAT (Value Added Tax) rate(s) as a multiplier (default is 0).
        vat_discount : numpy.ndarray or float or int, optional
            The VAT discount(s) as a multiplier (default is 0).
        pdri_rate : numpy.ndarray or float or int, optional
            The PDRI rate(s) as a multiplier (default is 0).
        pdri_discount : numpy.ndarray or float or int, optional
            PDRI discount(s) as a multiplier (default is 0).
        lbt_discount : numpy.ndarray or float or int, optional
            Land and Building Tax (LBT) discount(s) as a multiplier (default is 0).
        pdrd_discount : numpy.ndarray or float or int, optional
            PDRD discount as a multiplier (default is 0).

        Returns
        -------
        expenses: np.ndarray
            An array depicting the intangible expenditures each year, taking into
            account the inflation, VAT/PPN, PDRI, LBT/PBB, and PDRD.

        Notes
        -----
        This method calculates intangible expenditures while considering various economic factors
        such as inflation, VAT, PDRI, LBT, and PDRD. It uses decorators to apply these factors
        to the core calculation. In the core calculations:
        (1) Function np.bincount() is used to align the cost elements according
            to its corresponding expense year,
        (2) If len(expenses) < project_duration, then add the remaining elements
            with zeros.
        """

        @apply_pdrd(pdrd_discount=pdrd_discount)
        @apply_lbt(lbt_discount=lbt_discount)
        @apply_vat_pdri(
            vat_portion=self.vat_portion,
            vat_rate=vat_rate,
            vat_discount=vat_discount,
            pdri_portion=self.pdri_portion,
            pdri_rate=pdri_rate,
            pdri_discount=pdri_discount,
        )
        @apply_inflation(inflation_rate=inflation_rate)
        def _expenditures() -> np.ndarray:
            expenses = np.bincount(
                self.expense_year - self.start_year, weights=self.cost
            )
            zeros = np.zeros(self.project_duration - len(expenses))
            expenses = np.concatenate((expenses, zeros))
            return expenses

        return _expenditures()

    def __len__(self):
        return self.project_duration

    def __eq__(self, other):
        # Between two instances of Intangible
        if isinstance(other, Intangible):
            return all(
                (
                    np.allclose(self.cost, other.cost),
                    np.allclose(self.expense_year, other.expense_year),
                    self.cost_allocation == other.cost_allocation,
                    self.vat_portion == other.vat_portion,
                    self.pdri_portion == other.pdri_portion,
                )
            )

        # Between an instance of Intangible and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) == other

        else:
            return False

    def __lt__(self, other):
        # Between an instance of Intangible with another instance of Intangible/Tangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) < np.sum(other.cost)

        # Between an instance of Intangible and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) < other

        else:
            raise IntangibleException(
                f"Must compare an instance of Intangible with another instance of "
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float."
            )

    def __le__(self, other):
        # Between an instance of Intangible with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) <= np.sum(other.cost)

        # Between an instance of Intangible and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) <= other

        else:
            raise IntangibleException(
                f"Must compare an instance of Intangible with another instance of "
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float."
            )

    def __gt__(self, other):
        # Between an instance of Intangible with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) > np.sum(other.cost)

        # Between an instance of Intangible and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) > other

        else:
            raise IntangibleException(
                f"Must compare an instance of Intangible with another instance of "
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float."
            )

    def __ge__(self, other):
        # Between an instance of Intangible with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) >= np.sum(other.cost)

        # Between an instance of Intangible and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) >= other

        else:
            raise IntangibleException(
                f"Must compare an instance of Intangible with another instance of "
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float."
            )

    def __add__(self, other):
        # Only allows addition between an instance of Intangible and another instance of Intangible
        if isinstance(other, Intangible):
            if (
                self.cost_allocation != other.cost_allocation
                or self.vat_portion != other.vat_portion
                or self.pdri_portion != other.pdri_portion
            ):
                raise IntangibleException(
                    f"Cost allocation/VAT portion/PDRI portion is not equal. "
                    f"Cost allocation: {self.cost_allocation} vs. {other.cost_allocation}, "
                    f"VAT portion: {self.vat_portion} vs. {other.vat_portion}, "
                    f"PDRI portion: {self.pdri_portion} vs. {other.pdri_portion}."
                )

            else:
                combined_start_year = min(self.start_year, other.start_year)
                combined_end_year = max(self.end_year, other.end_year)
                combined_description = self.description + other.description

                return Intangible(
                    start_year=combined_start_year,
                    end_year=combined_end_year,
                    cost=np.concatenate((self.cost, other.cost)),
                    expense_year=np.concatenate(
                        (self.expense_year, other.expense_year)
                    ),
                    cost_allocation=self.cost_allocation,
                    vat_portion=self.vat_portion,
                    pdri_portion=self.pdri_portion,
                    description=combined_description,
                )

        else:
            raise IntangibleException(
                f"Must add between an instance of Intangible "
                f"with another instance of Intangible. "
                f"{other}({other.__class__.__qualname__}) is not "
                f"an instance of Intangible."
            )

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        # Only allows subtraction between an instance of Intangible and another instance of Intangible
        if isinstance(other, Intangible):
            if (
                self.cost_allocation != other.cost_allocation
                or self.vat_portion != other.vat_portion
                or self.pdri_portion != other.pdri_portion
            ):
                raise IntangibleException(
                    f"Cost allocation/VAT portion/PDRI portion is not equal. "
                    f"Cost allocation: {self.cost_allocation} vs. {other.cost_allocation}, "
                    f"VAT portion: {self.vat_portion} vs. {other.vat_portion}, "
                    f"PDRI portion: {self.pdri_portion} vs. {other.pdri_portion}."
                )

            else:
                combined_start_year = min(self.start_year, other.start_year)
                combined_end_year = max(self.end_year, other.end_year)
                combined_description = self.description + other.description

                return Intangible(
                    start_year=combined_start_year,
                    end_year=combined_end_year,
                    cost=np.concatenate((self.cost, -other.cost)),
                    expense_year=np.concatenate(
                        (self.expense_year, other.expense_year)
                    ),
                    cost_allocation=self.cost_allocation,
                    vat_portion=self.vat_portion,
                    pdri_portion=self.pdri_portion,
                    description=combined_description,
                )

        else:
            raise IntangibleException(
                f"Must subtract between an instance of Intangible "
                f"with another instance of Intangible. "
                f"{other}({other.__class__.__qualname__}) is not "
                f"an instance of Intangible."
            )

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        # Multiplication is allowed only with an integer/a float
        if isinstance(other, (int, float)):
            return Intangible(
                start_year=self.start_year,
                end_year=self.end_year,
                cost=self.cost * other,
                expense_year=self.expense_year.copy(),
                cost_allocation=self.cost_allocation,
                vat_portion=self.vat_portion,
                pdri_portion=self.pdri_portion,
                description=self.description,
            )

        else:
            raise IntangibleException(
                f"Must multiply with an integer or a float; "
                f"{other}({other.__class__.__qualname__}) is not an integer nor a float."
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        # Between an instance of Intangible with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) / np.sum(other.cost)

        # Between an instance of Intangible and an integer/float
        elif isinstance(other, (int, float)):
            # Cannot divide with zero
            if other == 0:
                raise IntangibleException(f"Cannot divide with zero")

            else:
                return Intangible(
                    start_year=self.start_year,
                    end_year=self.end_year,
                    cost=self.cost / other,
                    expense_year=self.expense_year.copy(),
                    cost_allocation=self.cost_allocation,
                    vat_portion=self.vat_portion,
                    pdri_portion=self.pdri_portion,
                    description=self.description,
                )

        else:
            raise IntangibleException(
                f"Must divide with an instance of Tangible/Intangible/OPEX/ASR, integer or a float; "
                f"{other}({other.__class__.__qualname__}) is not an instance "
                f"of Tangible/Intangible/OPEX/ASR nor an integer nor a float."
            )


@dataclass
class OPEX:
    """
    Manages an OPEX asset.

    Parameters
    ----------
    start_year : int
        The start year of the project.
    end_year : int
        The end year of the project.
    fixed_cost : np.ndarray
        An array representing the fixed cost of an OPEX asset.
    expense_year : np.ndarray
        An array representing the expense year of an OPEX asset.
    cost_allocation : FluidType
        A string depicting the cost allocation of an OPEX asset.
    prod_rate: np.ndarray
        The production rate of a particular fluid type.
    cost_per_volume: np.ndarray
        Cost associated with production of a particular fluid type.
    vat_portion: float | int
        A fraction of OPEX cost susceptible for VAT/PPN.
    pdri_portion: float | int
        A fraction of OPEX cost susceptible for PDRI.
    description: list[str]
        A list of string description regarding the associated OPEX cost.
    """

    start_year: int
    end_year: int
    fixed_cost: np.ndarray
    expense_year: np.ndarray
    cost_allocation: FluidType = field(default=FluidType.OIL)
    prod_rate: np.ndarray = field(default=None, repr=False)
    cost_per_volume: np.ndarray = field(default=None, repr=False)
    vat_portion: float | int = field(default=1.0, repr=False)
    pdri_portion: float | int = field(default=1.0, repr=False)
    description: list[str] = field(default=None)

    # Attribute to be defined later on
    variable_cost: np.ndarray = field(default=None, init=False, repr=False)
    cost: np.ndarray = field(default=None, init=False, repr=False)
    project_duration: int = field(default=None, init=False, repr=False)
    project_years: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # Check for inappropriate start and end year project
        if self.end_year >= self.start_year:
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise OPEXException(
                f"start year {self.start_year} "
                f"is after the end year {self.end_year}"
            )

        # Configure attribute "description"
        if self.description is None:
            self.description = [" " for _ in range(len(self.fixed_cost))]

        if self.description is not None:
            if len(self.description) != len(self.fixed_cost):
                raise OPEXException(
                    f"Unequal length of array: "
                    f"description: {len(self.description)}, "
                    f"fixed cost: {len(self.fixed_cost)}"
                )

        # Raise an error message: expense year is after the end year of the project
        if np.max(self.expense_year) > self.end_year:
            raise OPEXException(
                f"Expense year ({np.max(self.expense_year)}) "
                f"is after the end year of the project ({self.end_year})"
            )

        # Raise an error message: expense year is before the start year of the project
        if np.min(self.expense_year) < self.start_year:
            raise OPEXException(
                f"Expense year ({np.min(self.expense_year)}) "
                f"is before the start year of the project ({self.start_year})"
            )

        # User provides both prod_rate and cost_per_volume data
        if self.prod_rate is not None and self.cost_per_volume is not None:
            # Check input data for unequal length
            if not all(
                len(i) == len(self.fixed_cost)
                for i in [self.expense_year, self.prod_rate, self.cost_per_volume]
            ):
                raise OPEXException(
                    f"Unequal length of array: "
                    f"fixed_cost: {len(self.fixed_cost)}, "
                    f"expense_year: {len(self.expense_year)}, "
                    f"prod_rate: {len(self.prod_rate)}, "
                    f"cost_per_volume: {len(self.cost_per_volume)}"
                )

            # Specify attribute variable_cost
            self.variable_cost = self.prod_rate * self.cost_per_volume

        # User only provides prod_rate data
        elif self.prod_rate is not None and self.cost_per_volume is None:
            raise OPEXException(f"cost_per_volume data is missing")

        # User only provides cost_per_volume data
        elif self.prod_rate is None and self.cost_per_volume is not None:
            raise OPEXException(f"prod_rate data is missing")

        # User does not provide both prod_rate and cost_per_volume data
        elif self.prod_rate is None and self.cost_per_volume is None:
            self.prod_rate = np.zeros_like(self.fixed_cost)
            self.cost_per_volume = np.zeros_like(self.fixed_cost)
            self.variable_cost = np.zeros_like(self.fixed_cost)

        # Define cost
        self.cost = self.fixed_cost + self.variable_cost

    def expenditures(
        self,
        inflation_rate: float | int = 0.0,
        vat_rate: np.ndarray | float | int = 0.0,
        vat_discount: np.ndarray | float | int = 0.0,
        pdri_rate: np.ndarray | float | int = 0.0,
        pdri_discount: np.ndarray | float | int = 0.0,
        lbt_discount: np.ndarray | float | int = 0.0,
        pdrd_discount: np.ndarray | float | int = 0.0,
    ) -> np.ndarray:
        """
        Calculate OPEX expenditures per year.
        Allocate OPEX expenditures following the associated expense year.

        Parameters:
        -----------
        inflation_rate : float or int, optional
            The annual inflation rate as a decimal (default is 0).
        vat_rate : numpy.ndarray or float or int, optional
            The VAT (Value Added Tax) rate(s) as a multiplier (default is 0).
        vat_discount : numpy.ndarray or float or int, optional
            The VAT discount(s) as a multiplier (default is 0).
        pdri_rate : numpy.ndarray or float or int, optional
            The PDRI rate(s) as a multiplier (default is 0).
        pdri_discount : numpy.ndarray or float or int, optional
            PDRI discount(s) as a multiplier (default is 0).
        lbt_discount : numpy.ndarray or float or int, optional
            Land and Building Tax (LBT) discount(s) as a multiplier (default is 0).
        pdrd_discount : numpy.ndarray or float or int, optional
            PDRD discount as a multiplier (default is 0).

        Returns
        -------
        expenses: np.ndarray
            An array depicting the OPEX expenditures each year, taking into
            account the inflation, VAT/PPN, PDRI, LBT/PBB, and PDRD.

        Notes
        -----
        This method calculates OPEX expenditures while considering various economic factors
        such as inflation, VAT, PDRI, LBT, and PDRD. It uses decorators to apply these factors
        to the core calculation. In the core calculations:
        (1) Function np.bincount() is used to align the cost elements according
            to its corresponding expense year,
        (2) If len(expenses) < project_duration, then add the remaining elements
            with zeros.
        """

        @apply_pdrd(pdrd_discount=pdrd_discount)
        @apply_lbt(lbt_discount=lbt_discount)
        @apply_vat_pdri(
            vat_portion=self.vat_portion,
            vat_rate=vat_rate,
            vat_discount=vat_discount,
            pdri_portion=self.pdri_portion,
            pdri_rate=pdri_rate,
            pdri_discount=pdri_discount,
        )
        @apply_inflation(inflation_rate=inflation_rate)
        def _expenditures() -> np.ndarray:
            expenses = np.bincount(
                self.expense_year - self.start_year, weights=self.cost
            )
            zeros = np.zeros(self.project_duration - len(expenses))
            expenses = np.concatenate((expenses, zeros))
            return expenses

        return _expenditures()

    def __len__(self):
        return self.project_duration

    def __eq__(self, other):
        # Between two instances of OPEX
        if isinstance(other, OPEX):
            return all(
                (
                    np.allclose(self.fixed_cost, other.fixed_cost),
                    np.allclose(self.variable_cost, other.variable_cost),
                    np.allclose(self.expense_year, other.expense_year),
                    self.cost_allocation == other.cost_allocation,
                    self.vat_portion == other.vat_portion,
                    self.pdri_portion == other.pdri_portion,
                )
            )

        # Between an instance of OPEX and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.fixed_cost, self.variable_cost) == other

        else:
            return False

    def __lt__(self, other):
        # Between an instance of OPEX with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) < np.sum(other.cost)

        # Between an instance of OPEX and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) < other

        else:
            raise OPEXException(
                f"Must compare an instance of OPEX with another instance of "
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float"
            )

    def __le__(self, other):
        # Between an instance of OPEX with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) <= np.sum(other.cost)

        # Between an instance of OPEX and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) <= other

        else:
            raise OPEXException(
                f"Must compare an instance of OPEX with another instance of "
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float"
            )

    def __gt__(self, other):
        # Between an instance of OPEX with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) > np.sum(other.cost)

        # Between an instance of OPEX and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) > other

        else:
            raise OPEXException(
                f"Must compare an instance of OPEX with another instance of "
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float"
            )

    def __ge__(self, other):
        # Between an instance of OPEX with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) >= np.sum(other.cost)

        # Between an instance of OPEX and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) >= other

        else:
            raise OPEXException(
                f"Must compare an instance of OPEX with another instance of "
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float"
            )

    def __add__(self, other):
        # Only allows addition between an instance of OPEX and another instance of OPEX
        if isinstance(other, OPEX):
            if (
                self.cost_allocation != other.cost_allocation
                or self.vat_portion != other.vat_portion
                or self.pdri_portion != other.pdri_portion
            ):
                raise OPEXException(
                    f"Cost allocation/VAT portion/PDRI portion is not equal. "
                    f"Cost allocation: {self.cost_allocation} vs. {other.cost_allocation}, "
                    f"VAT portion: {self.vat_portion} vs. {other.vat_portion}, "
                    f"PDRI portion: {self.pdri_portion} vs. {other.pdri_portion}."
                )

            else:
                combined_start_year = min(self.start_year, other.start_year)
                combined_end_year = max(self.end_year, other.end_year)
                combined_description = self.description + other.description

                return OPEX(
                    start_year=combined_start_year,
                    end_year=combined_end_year,
                    fixed_cost=np.concatenate((self.fixed_cost, other.fixed_cost)),
                    expense_year=np.concatenate(
                        (self.expense_year, other.expense_year)
                    ),
                    cost_allocation=self.cost_allocation,
                    prod_rate=np.concatenate((self.prod_rate, other.prod_rate)),
                    cost_per_volume=np.concatenate(
                        (self.cost_per_volume, other.cost_per_volume)
                    ),
                    vat_portion=self.vat_portion,
                    pdri_portion=self.pdri_portion,
                    description=combined_description,
                )

        else:
            raise OPEXException(
                f"Must add an instance of OPEX with another "
                f"instance of OPEX. "
                f"{other}({other.__class__.__qualname__}) is not "
                f"an instance of OPEX."
            )

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        # Only allows subtraction between an instance of OPEX and another instance of OPEX
        if isinstance(other, OPEX):
            if (
                self.cost_allocation != other.cost_allocation
                or self.vat_portion != other.vat_portion
                or self.pdri_portion != other.pdri_portion
            ):
                raise OPEXException(
                    f"Cost allocation/VAT portion/PDRI portion is not equal. "
                    f"Cost allocation: {self.cost_allocation} vs. {other.cost_allocation}, "
                    f"VAT portion: {self.vat_portion} vs. {other.vat_portion}, "
                    f"PDRI portion: {self.pdri_portion} vs. {other.pdri_portion}."
                )

            else:
                combined_start_year = min(self.start_year, other.start_year)
                combined_end_year = max(self.end_year, other.end_year)
                combined_description = self.description + other.description

                return OPEX(
                    start_year=combined_start_year,
                    end_year=combined_end_year,
                    fixed_cost=np.concatenate((self.fixed_cost, -other.fixed_cost)),
                    expense_year=np.concatenate(
                        (self.expense_year, other.expense_year)
                    ),
                    cost_allocation=self.cost_allocation,
                    prod_rate=np.concatenate((self.prod_rate, -other.prod_rate)),
                    cost_per_volume=np.concatenate(
                        (self.cost_per_volume, other.cost_per_volume)
                    ),
                    vat_portion=self.vat_portion,
                    pdri_portion=self.pdri_portion,
                    description=combined_description,
                )

        else:
            raise OPEXException(
                f"Must subtract between an instance of OPEX "
                f"with another instance of OPEX. "
                f"{other}({other.__class__.__qualname__}) is "
                f"not an instance of OPEX"
            )

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        # Multiplication is allowed only with an integer/a float
        if isinstance(other, (int, float)):
            return OPEX(
                start_year=self.start_year,
                end_year=self.end_year,
                fixed_cost=self.fixed_cost * other,
                expense_year=self.expense_year,
                cost_allocation=self.cost_allocation,
                prod_rate=self.prod_rate * other,
                cost_per_volume=self.cost_per_volume,
                vat_portion=self.vat_portion,
                pdri_portion=self.pdri_portion,
                description=self.description,
            )

        else:
            raise OPEXException(
                f"Must multiply with an integer or a float; "
                f"{other}({other.__class__.__qualname__}) is not an integer nor a float."
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        # Between an instance of OPEX with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) / np.sum(other.cost)

        # Between an instance of OPEX and an integer/float
        elif isinstance(other, (int, float)):
            # Cannot divide with zero
            if other == 0:
                raise OPEXException(f"Cannot divide with zero")

            else:
                return OPEX(
                    start_year=self.start_year,
                    end_year=self.end_year,
                    fixed_cost=self.fixed_cost / other,
                    expense_year=self.expense_year,
                    cost_allocation=self.cost_allocation,
                    prod_rate=self.prod_rate / other,
                    cost_per_volume=self.cost_per_volume,
                    vat_portion=self.vat_portion,
                    pdri_portion=self.pdri_portion,
                    description=self.description,
                )

        else:
            raise OPEXException(
                f"Must divide with an instance of Tangible/Intangible/OPEX/ASR, integer or a float; "
                f"{other}({other.__class__.__qualname__}) is not an instance "
                f"of Tangible/Intangible/OPEX/ASR nor an integer nor a float."
            )


@dataclass
class ASR:
    """
    Manages an ASR asset.

    Parameters
    ----------
    start_year : int
        The start year of the project.
    end_year : int
        The end year of the project.
    cost : numpy.ndarray
        An array representing the cost of an ASR asset.
    expense_year : numpy.ndarray
        An array representing the expense year of an ASR asset.
    cost_allocation : FluidType
        A string depicting the cost allocation of an ASR asset.
    vat_portion: float | int
        A fraction of ASR cost susceptible for VAT/PPN.
    pdri_portion: float | int
        A fraction of ASR cost susceptible for PDRI.
    description: list[str]
        A list of string description regarding the associated ASR cost.
    """

    start_year: int
    end_year: int
    cost: np.ndarray
    expense_year: np.ndarray
    cost_allocation: FluidType = field(default=FluidType.OIL)
    vat_portion: float | int = field(default=1.0, repr=False)
    pdri_portion: float | int = field(default=1.0, repr=False)
    description: list[str] = field(default=None)

    # Attribute to de defined later on
    project_duration: int = field(default=None, init=False, repr=False)
    project_years: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # Check for inappropriate start and end year project
        if self.end_year >= self.start_year:
            self.project_duration = self.end_year - self.start_year + 1
            self.project_years = np.arange(self.start_year, self.end_year + 1, 1)

        else:
            raise ASRException(
                f"start year {self.start_year} "
                f"is after the end year {self.end_year}"
            )

        # Configure attribute "description"
        if self.description is None:
            self.description = [" " for _ in range(len(self.cost))]

        if self.description is not None:
            if len(self.description) != len(self.cost):
                raise ASRException(
                    f"Unequal length of array: "
                    f"description: {len(self.description)}, "
                    f"cost: {len(self.cost)}"
                )

        # Check input data for unequal length
        if len(self.expense_year) != len(self.cost):
            raise ASRException(
                f"Unequal length of array: "
                f"cost: {len(self.cost)}, "
                f"expense_year: {len(self.expense_year)}"
            )

        # Raise an error message: expense year is after the end year of the project
        if np.max(self.expense_year) > self.end_year:
            raise ASRException(
                f"Expense year ({np.max(self.expense_year)}) "
                f"is after the end year of the project ({self.end_year})"
            )

        # Raise an error message: expense year is before the start year of the project
        if np.min(self.expense_year) < self.start_year:
            raise ASRException(
                f"Expense year ({np.min(self.expense_year)}) "
                f"is before the start year of the project ({self.start_year})"
            )

    def future_cost(
        self,
        future_rate: float = 0.02,
        inflation_rate: float | int = 0.0,
        vat_rate: np.ndarray | float | int = 0.0,
        vat_discount: np.ndarray | float | int = 0.0,
        pdri_rate: np.ndarray | float | int = 0.0,
        pdri_discount: np.ndarray | float | int = 0.0,
        lbt_discount: np.ndarray | float | int = 0.0,
        pdrd_discount: np.ndarray | float | int = 0.0,
    ) -> np.ndarray:
        """
        Calculate the future cost of an asset.

        Parameters:
        -----------
        future_rate : float, optional
            The future rate of cost as a decimal (default is 0.02).
        inflation_rate : float or int, optional
            The annual inflation rate as a decimal (default is 0.0).
        vat_rate : np.ndarray, float, or int, optional
            The Value Added Tax (VAT) rate(s) to be applied as a multiplier.
            Can be a single value or an array (default is 0.0).
        vat_discount : np.ndarray, float, or int, optional
            The VAT discount(s) to be applied as a multiplier.
            Can be a single value or an array (default is 0.0).
        pdri_rate : np.ndarray, float, or int, optional
            The PDRI rate(s) to be applied as a multiplier.
            Can be a single value or an array (default is 0.0).
        pdri_discount : np.ndarray, float, or int, optional
            The PDRI discount(s) to be applied as a multiplier.
            Can be a single value or an array (default is 0.0).
        lbt_discount : np.ndarray, float, or int, optional
            The Land and Building Tax (LBT) discount rate as a decimal
            or an array of discount rates (default is 0.0).
        pdrd_discount : np.ndarray, float, or int, optional
            The PDRD discount(s) to be applied as a multiplier.
            Can be a single value or an array (default is 0.0).

        Returns:
        --------
        np.ndarray
            An array containing the future cost of the asset.

        Notes:
        ------
        This function calculates the future cost of an asset, taking into account
        inflation and various taxes and discounts schemes.
        """
        # Configure the modified cost
        cost_modified = apply_cost_modification(
            start_year=self.start_year,
            cost=self.cost,
            expense_year=self.expense_year,
            inflation_rate=inflation_rate,
            vat_portion=self.vat_portion,
            vat_rate=vat_rate,
            vat_discount=vat_discount,
            pdri_portion=self.pdri_portion,
            pdri_rate=pdri_rate,
            pdri_discount=pdri_discount,
            lbt_discount=lbt_discount,
            pdrd_discount=pdrd_discount,
        )

        return cost_modified * np.power(
            (1 + future_rate), self.end_year - self.expense_year + 1
        )

    def expenditures(
        self,
        future_rate: float = 0.02,
        inflation_rate: float | int = 0.0,
        vat_rate: np.ndarray | float | int = 0.0,
        vat_discount: np.ndarray | float | int = 0.0,
        pdri_rate: np.ndarray | float | int = 0.0,
        pdri_discount: np.ndarray | float | int = 0.0,
        lbt_discount: np.ndarray | float | int = 0.0,
        pdrd_discount: np.ndarray | float | int = 0.0,
    ) -> np.ndarray:
        """
        Calculate ASR expenditures per year.

        This method calculates the ASR expenditures per year
        based on the expense year and cost data provided.

        Parameters
        ----------
        future_rate : float, optional
            The future rate used in cost calculation (default is 0.02).
        inflation_rate : float or int, optional
            The inflation rate to apply (default is 0).
        vat_rate : np.ndarray, float, or int, optional
            The VAT/PPN rate(s) to apply. Can be a single value or an array (default is 0).
        vat_discount : np.ndarray, float, or int, optional
            The VAT discount(s) to apply. Can be a single value or an array (default is 0).
        pdri_rate : np.ndarray, float, or int, optional
            The PDRI rate(s) to apply. Can be a single value or an array (default is 0).
        pdri_discount : np.ndarray, float, or int, optional
            The PDRI discount(s) to apply. Can be a single value or an array (default is 0).
        lbt_discount : np.ndarray, float, or int, optional
            The LBT discount(s) to apply. Can be a single value or an array (default is 0).
        pdrd_discount : np.ndarray, float, or int, optional
            The PDRD discount(s) to apply. Can be a single value or an array (default is 0).

        Returns
        -------
        expenses: np.ndarray
            An array depicting the ASR expenditures each year, taking into
            account the inflation, VAT/PPN, PDRI, LBT/PBB, and PDRD.

        Notes
        -----
        This method calculates ASR expenditures while considering various economic factors
        such as inflation, VAT, PDRI, LBT, and PDRD.
        """
        # Distance of expense year from the end year of the project
        cost_duration = self.end_year - self.expense_year + 1

        # Cost allocation: cost distribution per year
        cost_alloc = (
            self.future_cost(
                future_rate=future_rate,
                inflation_rate=inflation_rate,
                vat_rate=vat_rate,
                vat_discount=vat_discount,
                pdri_rate=pdri_rate,
                pdri_discount=pdri_discount,
                lbt_discount=lbt_discount,
                pdrd_discount=pdrd_discount,
            )
            / cost_duration
        )

        # Distance of expense year from the start year of the project
        shift_indices = self.expense_year - self.start_year

        # ASR allocation per element per distributed year
        asr_alloc = np.asarray(
            [
                np.concatenate((np.zeros(i), np.repeat(ca, cd)))
                for i, ca, cd in zip(shift_indices, cost_alloc, cost_duration)
            ]
        )

        return asr_alloc.sum(axis=0)

    def __len__(self):
        return self.project_duration

    def __eq__(self, other):
        # Between two instances of ASR
        if isinstance(other, ASR):
            return all(
                (
                    np.allclose(self.cost, other.cost),
                    np.allclose(self.expense_year, other.expense_year),
                    self.cost_allocation == other.cost_allocation,
                    self.vat_portion == other.vat_portion,
                    self.pdri_portion == other.pdri_portion,
                )
            )

        # Between an instance of ASR and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) == other

        else:
            return False

    def __lt__(self, other):
        # Between an instance of ASR with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) < np.sum(other.cost)

        # Between an instance of ASR with an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) < other

        else:
            raise ASRException(
                f"Must compare an instance of ASR with another instance of "
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float."
            )

    def __le__(self, other):
        # Between an instance of ASR with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) <= np.sum(other.cost)

        # Between an instance of ASR and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) <= other

        else:
            raise ASRException(
                f"Must compare an instance of ASR with another instance of "
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float."
            )

    def __gt__(self, other):
        # Between an instance of ASR with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) > np.sum(other.cost)

        # Between an instance of ASR with an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) > other

        else:
            raise ASRException(
                f"Must compare an instance of ASR with another instance of "
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float."
            )

    def __ge__(self, other):
        # Between an instance of ASR with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) >= np.sum(other.cost)

        # Between an instance of ASR and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) >= other

        else:
            raise ASRException(
                f"Must compare an instance of ASR with another instance of "
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float."
            )

    def __add__(self, other):
        # Only allows addition between an instance of ASR and another instance of ASR
        if isinstance(other, ASR):
            if (
                self.cost_allocation != other.cost_allocation
                or self.vat_portion != other.vat_portion
                or self.pdri_portion != other.pdri_portion
            ):
                raise ASRException(
                    f"Cost allocation/VAT portion/PDRI portion is not equal. "
                    f"Cost allocation: {self.cost_allocation} vs. {other.cost_allocation}, "
                    f"VAT portion: {self.vat_portion} vs. {other.vat_portion}, "
                    f"PDRI portion: {self.pdri_portion} vs. {other.pdri_portion}."
                )

            else:
                combined_start_year = min(self.start_year, other.start_year)
                combined_end_year = max(self.end_year, other.end_year)
                combined_description = self.description + other.description

                return ASR(
                    start_year=combined_start_year,
                    end_year=combined_end_year,
                    cost=np.concatenate((self.cost, other.cost)),
                    expense_year=np.concatenate(
                        (self.expense_year, other.expense_year)
                    ),
                    cost_allocation=self.cost_allocation,
                    vat_portion=self.vat_portion,
                    pdri_portion=self.pdri_portion,
                    description=combined_description,
                )

        else:
            raise ASRException(
                f"Must add between an instance of ASR "
                f"with another instance of ASR. "
                f"{other}({other.__class__.__qualname__}) is "
                f"not an instance of ASR."
            )

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        # Only allows subtraction between an instance of ASR and another instance of ASR
        if isinstance(other, ASR):
            if (
                self.cost_allocation != other.cost_allocation
                or self.vat_portion != other.vat_portion
                or self.pdri_portion != other.pdri_portion
            ):
                raise ASRException(
                    f"Cost allocation/VAT portion/PDRI portion is not equal. "
                    f"Cost allocation: {self.cost_allocation} vs. {other.cost_allocation}, "
                    f"VAT portion: {self.vat_portion} vs. {other.vat_portion}, "
                    f"PDRI portion: {self.pdri_portion} vs. {other.pdri_portion}."
                )

            else:
                combined_start_year = min(self.start_year, other.start_year)
                combined_end_year = max(self.end_year, other.end_year)
                combined_description = self.description + other.description

                return ASR(
                    start_year=combined_start_year,
                    end_year=combined_end_year,
                    cost=np.concatenate((self.cost, -other.cost)),
                    expense_year=np.concatenate(
                        (self.expense_year, other.expense_year)
                    ),
                    cost_allocation=self.cost_allocation,
                    vat_portion=self.vat_portion,
                    pdri_portion=self.pdri_portion,
                    description=combined_description,
                )

        else:
            raise ASRException(
                f"Must subtract between an instance of ASR "
                f"with another instance of ASR. "
                f"{other}({other.__class__.__qualname__}) is not "
                f"an instance of ASR."
            )

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        # Multiplication is allowed only with an integer/a float
        if isinstance(other, (int, float)):
            return ASR(
                start_year=self.start_year,
                end_year=self.end_year,
                cost=self.cost * other,
                expense_year=self.expense_year,
                cost_allocation=self.cost_allocation,
                vat_portion=self.vat_portion,
                pdri_portion=self.pdri_portion,
                description=self.description,
            )

        else:
            raise ASRException(
                f"Must multiply with an integer or a float; "
                f"{other}({other.__class__.__qualname__}) is not an integer nor a float."
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        # Between an instance of ASR with another instance of Tangible/Intangible/OPEX/ASR
        if isinstance(other, (Tangible, Intangible, OPEX, ASR)):
            return np.sum(self.cost) / np.sum(other.cost)

        # Between an instance of ASR and an integer/float
        elif isinstance(other, (int, float)):
            # Cannot divide with zero
            if other == 0:
                raise ASRException(f"Cannot divide with zero")

            else:
                return ASR(
                    start_year=self.start_year,
                    end_year=self.end_year,
                    cost=self.cost / other,
                    expense_year=self.expense_year,
                    cost_allocation=self.cost_allocation,
                    vat_portion=self.vat_portion,
                    pdri_portion=self.pdri_portion,
                    description=self.description,
                )

        else:
            raise ASRException(
                f"Must divide with an instance of Tangible/Intangible/OPEX/ASR, integer or a float; "
                f"{other}({other.__class__.__qualname__}) is not an instance "
                f"of Tangible/Intangible/OPEX/ASR nor an integer nor a float."
            )
