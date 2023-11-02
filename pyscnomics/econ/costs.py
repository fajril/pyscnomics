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
from pyscnomics.tools.helper import apply_cost_adjustment
from pyscnomics.econ.selection import FluidType, DeprMethod, TaxType


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
class GeneralCost:
    """
    Manages a GeneralCost asset.

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
    cost_allocation : list[FluidType]
        A list representing the cost allocation of a tangible asset.
    description: list[str]
        A list of string description regarding the associated tangible cost.
    vat_portion: np.ndarray
        The portion of 'cost' that is subject to VAT.
        Must be an array of length equals to the length of 'cost' array.
    vat_discount: float
        The VAT discount to apply.
    lbt_portion: np.ndarray
        The portion of 'cost' that is subject to LBT.
        Must be an array of length equals to the length of 'cost' array.
    lbt_discount: float
        The LBT discount to apply.
    """

    start_year: int
    end_year: int
    cost: np.ndarray
    expense_year: np.ndarray
    cost_allocation: list[FluidType] = field(default=None)
    description: list[str] = field(default=None)
    vat_portion: np.ndarray = field(default=None)
    vat_discount: float | np.ndarray = field(default=0.0)
    lbt_portion: np.ndarray = field(default=None)
    lbt_discount: float | np.ndarray = field(default=0.0)

    # Attribute to be defined later on
    project_duration: int = field(default=None, init=False, repr=False)
    project_years: np.ndarray = field(default=None, init=False, repr=False)

    def expenditures(
        self,
        year_ref: int = None,
        tax_type: TaxType = TaxType.VAT,
        vat_rate: np.ndarray | float = 0.0,
        lbt_rate: np.ndarray | float = 0.0,
        inflation_rate: np.ndarray | float = 0.0,
    ) -> np.ndarray:
        """
        Calculate expenditures per year.

        This method calculates the expenditures per year based on the expense year
        and cost data provided.

        Parameters
        ----------
        year_ref : int
            The reference year for inflation calculation.
        tax_type: TaxType
            The type of tax applied to the corresponding asset.
            Available options: TaxType.VAT or TaxType.LBT
        vat_rate: np.ndarray | float
            The VAT rate to apply. Can be a single value or an array (default is 0.0).
        lbt_rate: np.ndarray | float
            The LBT rate to apply. Can be a single value or an array (default is 0.0).
        inflation_rate: np.ndarray | float
            The inflation rate to apply. Can be a single value or an array (default is 0.0).

        Returns
        -------
        np.ndarray
            An array depicting the expenditures each year, adjusted by tax
            and inflation schemes (if any).

        Notes
        -----
        This method calculates expenditures while considering tax and inflation schemes.
        The core calculations are as follows:
        (1) Apply adjustment to cost due to tax and inflation (if any), by calling
            'apply_cost_adjustment()' function,
        (2) Function np.bincount() is used to align the 'cost_adjusted' elements
            according to its corresponding expense year,
        (3) If len(expenses) < project_duration, then add the remaining elements
            with zeros.
        """
        if year_ref is None:
            year_ref = self.start_year

        cost_adjusted = apply_cost_adjustment(
            start_year=self.start_year,
            end_year=self.end_year,
            cost=self.cost,
            expense_year=self.expense_year,
            project_years=self.project_years,
            year_ref=year_ref,
            tax_type=tax_type,
            vat_portion=self.vat_portion,
            vat_rate=vat_rate,
            vat_discount=self.vat_discount,
            lbt_portion=self.lbt_portion,
            lbt_rate=lbt_rate,
            lbt_discount=self.lbt_discount,
            inflation_rate=inflation_rate,
        )

        expenses = np.bincount(
            self.expense_year - self.start_year, weights=cost_adjusted
        )
        zeros = np.zeros(self.project_duration - len(expenses))

        return np.concatenate((expenses, zeros))

    def __len__(self):
        return self.project_duration


@dataclass
class Tangible(GeneralCost):
    """
    Manages a tangible asset.

    Parameters
    ----------
    pis_year : numpy.ndarray, optional
        An array representing the PIS year of a tangible asset.
    salvage_value : numpy.ndarray, optional
        An array representing the salvage value of a tangible asset.
    useful_life : numpy.ndarray, optional
        An array representing the useful life of a tangible asset.
    depreciation_factor: np.ndarray, optional
        The value of depreciation factor to be used in PSC_DB depreciation method.
        Default value is 0.5 for the entire project duration.
    """

    pis_year: np.ndarray = field(default=None)
    salvage_value: np.ndarray = field(default=None)
    useful_life: np.ndarray = field(default=None)
    depreciation_factor: np.ndarray = field(default=None)

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

        # Configure VAT portion
        if self.vat_portion is None:
            self.vat_portion = np.ones_like(self.cost)

        if self.vat_portion is not None:
            if not isinstance(self.vat_portion, np.ndarray):
                raise TangibleException(
                    f"Attribute VAT portion must be a numpy ndarray; "
                    f"VAT portion ({self.vat_portion}) is of datatype "
                    f"{self.vat_portion.__class__.__qualname__}."
                )

        # Configure LBT portion
        if self.lbt_portion is None:
            self.lbt_portion = np.ones_like(self.cost)

        if self.lbt_portion is not None:
            if not isinstance(self.lbt_portion, np.ndarray):
                raise TangibleException(
                    f"Attribute LBT portion must be a numpy ndarray; "
                    f"LBT portion ({self.lbt_portion}) is of datatype "
                    f"{self.lbt_portion.__class__.__qualname__}."
                )

        # Configure VAT discount
        if isinstance(self.vat_discount, float):
            self.vat_discount = np.repeat(self.vat_discount, len(self.cost))

        if isinstance(self.vat_discount, np.ndarray):
            if len(self.vat_discount) != len(self.cost):
                raise TangibleException(
                    f"Unequal length of array: "
                    f"VAT discount: ({len(self.vat_discount)}), "
                    f"cost: ({len(self.cost)})."
                )

        # Configure LBT discount
        if isinstance(self.lbt_discount, float):
            self.lbt_discount = np.repeat(self.lbt_discount, len(self.cost))

        if isinstance(self.lbt_discount, np.ndarray):
            if len(self.lbt_discount) != len(self.cost):
                raise TangibleException(
                    f"Unequal length of array: "
                    f"LBT discount: ({len(self.lbt_discount)}), "
                    f"cost: ({len(self.cost)})."
                )

        # Configure description data
        if self.description is None:
            self.description = [" " for _ in range(len(self.cost))]

        if self.description is not None:
            if not isinstance(self.description, list):
                raise TangibleException(
                    f"Attribute description must be a list; "
                    f"description (datatype: {self.description.__class__.__qualname__}) "
                    f"is not a list."
                )

        # Configure pis_year data
        if self.pis_year is None:
            self.pis_year = self.expense_year

        if self.pis_year is not None:
            if not isinstance(self.pis_year, np.ndarray):
                raise TangibleException(
                    f"Attribute pis_year must be a numpy.ndarray; "
                    f"pis_year (datatype: {self.description.__class__.__qualname__}) "
                    f"is not a numpy.ndarray."
                )

        # Configure salvage_value data
        if self.salvage_value is None:
            self.salvage_value = np.zeros(len(self.cost))

        if self.salvage_value is not None:
            if not isinstance(self.salvage_value, np.ndarray):
                raise TangibleException(
                    f"Attribute salvage_value must be a numpy.ndarray; "
                    f"salvage_value (datatype: {self.description.__class__.__qualname__}) "
                    f"is not a numpy.ndarray."
                )

        # Configure useful_life data
        if self.useful_life is None:
            self.useful_life = np.repeat(5.0, len(self.cost))

        if self.useful_life is not None:
            if not isinstance(self.useful_life, np.ndarray):
                raise TangibleException(
                    f"Attribute useful_life must be a numpy.ndarray; "
                    f"useful_life (datatype: {self.description.__class__.__qualname__}) "
                    f"is not a numpy.ndarray."
                )

        # Configure depreciation_factor data
        if self.depreciation_factor is None:
            self.depreciation_factor = np.repeat(0.5, len(self.cost))

        if self.depreciation_factor is not None:
            if not isinstance(self.depreciation_factor, np.ndarray):
                raise TangibleException(
                    f"Attribute depreciation_factor must be a numpy.ndarray; "
                    f"depreciation_factor (datatype: {self.description.__class__.__qualname__}) "
                    f"is not a numpy.ndarray."
                )

        # Configure cost_allocation data
        if self.cost_allocation is None:
            self.cost_allocation = [FluidType.OIL for _ in range(len(self.cost))]

        if self.cost_allocation is not None:
            if not isinstance(self.cost_allocation, list):
                raise TangibleException(
                    f"Attribute cost_allocation must be a list. "
                    f"cost_allocation (datatype: {self.cost_allocation.__class__.__qualname__}) "
                    f"is not a list."
                )

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
                self.cost_allocation,
                self.description,
                self.vat_portion,
                self.lbt_portion,
            ]
        ):
            raise TangibleException(
                f"Unequal length of array: "
                f"cost: {len(self.cost)}, "
                f"expense_year: {len(self.expense_year)}, "
                f"pis_year: {len(self.pis_year)}, "
                f"salvage_value: {len(self.salvage_value)}, "
                f"useful_life: {len(self.useful_life)}, "
                f"depreciation_factor: {len(self.depreciation_factor)}, "
                f"cost_allocation: {len(self.cost_allocation)}, "
                f"description: {len(self.description)}, "
                f"vat_portion: {len(self.vat_portion)}, "
                f"lbt_portion: {len(self.lbt_portion)}."
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

    def total_depreciation_rate(
        self,
        depr_method: DeprMethod = DeprMethod.PSC_DB,
        decline_factor: float | int = 2,
        year_ref: int = None,
        tax_type: TaxType = TaxType.VAT,
        vat_rate: np.ndarray | float = 0.0,
        lbt_rate: np.ndarray | float = 0.0,
        inflation_rate: np.ndarray | float = 0.0,
    ) -> tuple:
        """
        Calculate total depreciation charge and undepreciated asset value based on various parameters.

        Parameters
        ----------
        depr_method : DeprMethod, optional
            The depreciation method to use (default is DeprMethod.PSC_DB).
        decline_factor : float | int, optional
            The decline factor used for declining balance depreciation (default is 2).
        year_ref : int
            The reference year for inflation calculation.
        tax_type: TaxType
            The type of tax applied to the corresponding asset.
            Available options: TaxType.VAT or TaxType.LBT (default is TaxType.VAT).
        vat_rate: np.ndarray | float
            The VAT rate to apply. Can be a single value or an array (default is 0.0).
        lbt_rate: np.ndarray | float
            The LBT rate to apply. Can be a single value or an array (default is 0.0).
        inflation_rate: np.ndarray | float
            The inflation rate to apply. Can be a single value or an array (default is 0.0).

        Returns
        -------
        tuple
            A tuple containing:
            (1) Total depreciation charges for each period.
            (2) The undepreciated asset value at the end of the analysis.

        Notes
        ------
        (1) This method calculates depreciation charges based on the specified
            depreciation method.
        (2) Prior to the core calculations, attribute 'cost' is adjusted by tax
            and inflation schemes (if any),
        (3) The depreciation charges are aligned with the corresponding periods
            based on pis_year.
        """
        if year_ref is None:
            year_ref = self.start_year

        cost_adjusted = apply_cost_adjustment(
            start_year=self.start_year,
            end_year=self.end_year,
            cost=self.cost,
            expense_year=self.expense_year,
            project_years=self.project_years,
            year_ref=year_ref,
            tax_type=tax_type,
            vat_portion=self.vat_portion,
            vat_rate=vat_rate,
            vat_discount=self.vat_discount,
            lbt_portion=self.lbt_portion,
            lbt_rate=lbt_rate,
            lbt_discount=self.lbt_discount,
            inflation_rate=inflation_rate,
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
                        cost_adjusted,
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
                        cost_adjusted,
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
                        cost_adjusted,
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
        undepreciated_asset = np.sum(cost_adjusted) - np.sum(total_depreciation_charge)

        return total_depreciation_charge, undepreciated_asset

    def total_depreciation_book_value(
        self,
        depr_method: DeprMethod = DeprMethod.PSC_DB,
        decline_factor: float | int = 2,
        year_ref: int = None,
        tax_type: TaxType = TaxType.VAT,
        vat_rate: np.ndarray | float = 0.0,
        lbt_rate: np.ndarray | float = 0.0,
        inflation_rate: np.ndarray | float = 0.0,
    ) -> np.ndarray:
        """
        Calculate the total book value of depreciation for the asset.

        Parameters
        ----------
        depr_method : DeprMethod, optional
            The depreciation method to use (default is DeprMethod.PSC_DB).
        decline_factor : float, optional
            The decline factor used for declining balance depreciation (default is 2).
        year_ref : int
            The reference year for inflation calculation.
        tax_type: TaxType
            The type of tax applied to the corresponding asset.
            Available options: TaxType.VAT or TaxType.LBT (default is TaxType.VAT).
        vat_rate: np.ndarray | float
            The VAT rate to apply. Can be a single value or an array (default is 0.0).
        lbt_rate: np.ndarray | float
            The LBT rate to apply. Can be a single value or an array (default is 0.0).
        inflation_rate: np.ndarray | float
            The inflation rate to apply. Can be a single value or an array (default is 0.0).

        Returns
        -------
        np.ndarray
            An array containing the cumulative book value of depreciation for each period,
            taking into account tax and inflation schemes.

        Notes
        -----
        (1) This method calculates the cumulative book value of depreciation for the asset
            based on the specified depreciation method and other parameters.
        (2) The cumulative book value is obtained by subtracting the cumulative
            depreciation charges from the cumulative expenditures.
        """
        if year_ref is None:
            year_ref = self.start_year

        # Calculate total depreciation charge from method total_depreciation_rate
        total_depreciation_charge = self.total_depreciation_rate(
            depr_method=depr_method,
            decline_factor=decline_factor,
            year_ref=year_ref,
            tax_type=tax_type,
            vat_rate=vat_rate,
            lbt_rate=lbt_rate,
            inflation_rate=inflation_rate,
        )[0]

        return np.cumsum(
            self.expenditures(
                year_ref=year_ref,
                tax_type=tax_type,
                vat_rate=vat_rate,
                lbt_rate=lbt_rate,
                inflation_rate=inflation_rate,
            )
        ) - np.cumsum(total_depreciation_charge)

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
                    np.allclose(self.vat_portion, other.vat_portion),
                    np.allclose(self.vat_discount, other.vat_discount),
                    np.allclose(self.lbt_portion, other.lbt_portion),
                    np.allclose(self.lbt_discount, other.lbt_discount),
                    self.cost_allocation == other.cost_allocation,
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
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float."
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
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float."
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
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float."
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
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float."
            )

    def __add__(self, other):
        # Only allows addition between an instance of Tangible and another instance of Tangible
        if isinstance(other, Tangible):
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            cost_combined = np.concatenate((self.cost, other.cost))
            expense_year_combined = np.concatenate(
                (self.expense_year, other.expense_year)
            )
            cost_allocation_combined = self.cost_allocation + other.cost_allocation
            description_combined = self.description + other.description
            vat_portion_combined = np.concatenate((self.vat_portion, other.vat_portion))
            vat_discount_combined = np.concatenate(
                (self.vat_discount, other.vat_discount)
            )
            lbt_portion_combined = np.concatenate((self.lbt_portion, other.lbt_portion))
            lbt_discount_combined = np.concatenate(
                (self.lbt_discount, other.lbt_discount)
            )
            pis_year_combined = np.concatenate((self.pis_year, other.pis_year))
            salvage_value_combined = np.concatenate(
                (self.salvage_value, other.salvage_value)
            )
            useful_life_combined = np.concatenate((self.useful_life, other.useful_life))
            depreciation_factor_combined = np.concatenate(
                (self.depreciation_factor, other.depreciation_factor)
            )

            return Tangible(
                start_year=start_year_combined,
                end_year=end_year_combined,
                cost=cost_combined,
                expense_year=expense_year_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
                vat_portion=vat_portion_combined,
                vat_discount=vat_discount_combined,
                lbt_portion=lbt_portion_combined,
                lbt_discount=lbt_discount_combined,
                pis_year=pis_year_combined,
                salvage_value=salvage_value_combined,
                useful_life=useful_life_combined,
                depreciation_factor=depreciation_factor_combined,
            )

        else:
            raise TangibleException(
                f"Must add an instance of Tangible with another instance of Tangible. "
                f"{other}({other.__class__.__qualname__}) is not an instance of Tangible."
            )

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        # Only allows subtraction between an instance of Tangible and another instance of Tangible
        if isinstance(other, Tangible):
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            cost_combined = np.concatenate((self.cost, -other.cost))
            expense_year_combined = np.concatenate(
                (self.expense_year, other.expense_year)
            )
            cost_allocation_combined = self.cost_allocation + other.cost_allocation
            description_combined = self.description + other.description
            vat_portion_combined = np.concatenate((self.vat_portion, other.vat_portion))
            vat_discount_combined = np.concatenate(
                (self.vat_discount, other.vat_discount)
            )
            lbt_portion_combined = np.concatenate((self.lbt_portion, other.lbt_portion))
            lbt_discount_combined = np.concatenate(
                (self.lbt_discount, other.lbt_discount)
            )
            pis_year_combined = np.concatenate((self.pis_year, other.pis_year))
            salvage_value_combined = np.concatenate(
                (self.salvage_value, other.salvage_value)
            )
            useful_life_combined = np.concatenate((self.useful_life, other.useful_life))
            depreciation_factor_combined = np.concatenate(
                (self.depreciation_factor, other.depreciation_factor)
            )

            return Tangible(
                start_year=start_year_combined,
                end_year=end_year_combined,
                cost=cost_combined,
                expense_year=expense_year_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
                vat_portion=vat_portion_combined,
                vat_discount=vat_discount_combined,
                lbt_portion=lbt_portion_combined,
                lbt_discount=lbt_discount_combined,
                pis_year=pis_year_combined,
                salvage_value=salvage_value_combined,
                useful_life=useful_life_combined,
                depreciation_factor=depreciation_factor_combined,
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
                expense_year=self.expense_year,
                cost_allocation=self.cost_allocation,
                description=self.description,
                vat_portion=self.vat_portion,
                vat_discount=self.vat_discount,
                lbt_portion=self.lbt_portion,
                lbt_discount=self.lbt_discount,
                pis_year=self.pis_year,
                salvage_value=self.salvage_value,
                useful_life=self.useful_life,
                depreciation_factor=self.depreciation_factor,
            )

        else:
            raise TangibleException(
                f"Must multiply with an integer or a float; "
                f"{other}({other.__class__.__qualname__}) "
                f"is not an integer nor a float."
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
                    expense_year=self.expense_year,
                    cost_allocation=self.cost_allocation,
                    description=self.description,
                    vat_portion=self.vat_portion,
                    vat_discount=self.vat_discount,
                    lbt_portion=self.lbt_portion,
                    lbt_discount=self.lbt_discount,
                    pis_year=self.pis_year,
                    salvage_value=self.salvage_value,
                    useful_life=self.useful_life,
                    depreciation_factor=self.depreciation_factor,
                )

        else:
            raise TangibleException(
                f"Must divide with an instance of Tangible/Intangible/OPEX/ASR, "
                f"integer or a float; "
                f"{other}({other.__class__.__qualname__}) is not an instance "
                f"of Tangible/Intangible/OPEX/ASR nor an integer nor a float."
            )


@dataclass
class Intangible(GeneralCost):
    """
    Manages an intangible asset.
    """
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

        # Configure VAT portion
        if self.vat_portion is None:
            self.vat_portion = np.ones_like(self.cost)

        if self.vat_portion is not None:
            if not isinstance(self.vat_portion, np.ndarray):
                raise IntangibleException(
                    f"Attribute VAT portion must be a numpy ndarray; "
                    f"VAT portion ({self.vat_portion}) is of datatype "
                    f"{self.vat_portion.__class__.__qualname__}."
                )

        # Configure LBT portion
        if self.lbt_portion is None:
            self.lbt_portion = np.ones_like(self.cost)

        if self.lbt_portion is not None:
            if not isinstance(self.lbt_portion, np.ndarray):
                raise IntangibleException(
                    f"Attribute LBT portion must be a numpy ndarray; "
                    f"LBT portion ({self.lbt_portion}) is of datatype "
                    f"{self.lbt_portion.__class__.__qualname__}."
                )

        # Configure VAT discount
        if isinstance(self.vat_discount, float):
            self.vat_discount = np.repeat(self.vat_discount, len(self.cost))

        if isinstance(self.vat_discount, np.ndarray):
            if len(self.vat_discount) != len(self.cost):
                raise IntangibleException(
                    f"Unequal length of array: "
                    f"VAT discount: ({len(self.vat_discount)}), "
                    f"cost: ({len(self.cost)})."
                )

        # Configure LBT discount
        if isinstance(self.lbt_discount, float):
            self.lbt_discount = np.repeat(self.lbt_discount, len(self.cost))

        if isinstance(self.lbt_discount, np.ndarray):
            if len(self.lbt_discount) != len(self.cost):
                raise IntangibleException(
                    f"Unequal length of array: "
                    f"LBT discount: ({len(self.lbt_discount)}), "
                    f"cost: ({len(self.cost)})."
                )

        # Configure description data
        if self.description is None:
            self.description = [" " for _ in range(len(self.cost))]

        if self.description is not None:
            if not isinstance(self.description, list):
                raise IntangibleException(
                    f"Attribute description must be a list; "
                    f"description (datatype: {self.description.__class__.__qualname__}) "
                    f"is not a list."
                )

        # Configure cost_allocation data
        if self.cost_allocation is None:
            self.cost_allocation = [FluidType.OIL for _ in range(len(self.cost))]

        if self.cost_allocation is not None:
            if not isinstance(self.cost_allocation, list):
                raise IntangibleException(
                    f"Attribute cost_allocation must be a list. "
                    f"cost_allocation ({self.cost_allocation.__class__.__qualname__}) "
                    f"is not a list."
                )

        # Check input data for unequal length
        arr_length = len(self.cost)

        if not all(
            len(arr) == arr_length
            for arr in [
                self.expense_year,
                self.cost_allocation,
                self.description,
                self.vat_portion,
                self.lbt_portion,
            ]
        ):
            raise IntangibleException(
                f"Unequal length of array: "
                f"cost: {len(self.cost)}, "
                f"expense_year: {len(self.expense_year)}, "
                f"cost_allocation: {len(self.cost_allocation)}, "
                f"description: {len(self.description)}, "
                f"vat_portion: {len(self.vat_portion)}, "
                f"lbt_portion: {len(self.lbt_portion)}."
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

    def __eq__(self, other):
        # Between two instances of Intangible
        if isinstance(other, Intangible):
            return all(
                (
                    np.allclose(self.cost, other.cost),
                    np.allclose(self.expense_year, other.expense_year),
                    np.allclose(self.vat_portion, other.vat_portion),
                    np.allclose(self.vat_discount, other.vat_discount),
                    np.allclose(self.lbt_portion, other.lbt_portion),
                    np.allclose(self.lbt_discount, other.lbt_discount),
                    self.cost_allocation == other.cost_allocation,
                )
            )

        # Between an instance of Intangible and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) == other

        else:
            return False

    def __lt__(self, other):
        # Between an instance of Intangible with another instance of Tangible/Intangible/OPEX/ASR
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
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            cost_combined = np.concatenate((self.cost, other.cost))
            expense_year_combined = np.concatenate((self.expense_year, other.expense_year))
            cost_allocation_combined = self.cost_allocation + other.cost_allocation
            description_combined = self.description + other.description
            vat_portion_combined = np.concatenate((self.vat_portion, other.vat_portion))
            vat_discount_combined = np.concatenate((self.vat_discount, other.vat_discount))
            lbt_portion_combined = np.concatenate((self.lbt_portion, other.lbt_portion))
            lbt_discount_combined = np.concatenate((self.lbt_discount, other.lbt_discount))

            return Intangible(
                start_year=start_year_combined,
                end_year=end_year_combined,
                cost=cost_combined,
                expense_year=expense_year_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
                vat_portion=vat_portion_combined,
                vat_discount=vat_discount_combined,
                lbt_portion=lbt_portion_combined,
                lbt_discount=lbt_discount_combined,
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
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            cost_combined = np.concatenate((self.cost, -other.cost))
            expense_year_combined = np.concatenate((self.expense_year, other.expense_year))
            cost_allocation_combined = self.cost_allocation + other.cost_allocation
            description_combined = self.description + other.description
            vat_portion_combined = np.concatenate((self.vat_portion, other.vat_portion))
            vat_discount_combined = np.concatenate((self.vat_discount, other.vat_discount))
            lbt_portion_combined = np.concatenate((self.lbt_portion, other.lbt_portion))
            lbt_discount_combined = np.concatenate((self.lbt_discount, other.lbt_discount))

            return Intangible(
                start_year=start_year_combined,
                end_year=end_year_combined,
                cost=cost_combined,
                expense_year=expense_year_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
                vat_portion=vat_portion_combined,
                vat_discount=vat_discount_combined,
                lbt_portion=lbt_portion_combined,
                lbt_discount=lbt_discount_combined,
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
                expense_year=self.expense_year,
                cost_allocation=self.cost_allocation,
                description=self.description,
                vat_portion=self.vat_portion,
                vat_discount=self.vat_discount,
                lbt_portion=self.lbt_portion,
                lbt_discount=self.lbt_discount,
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
                    expense_year=self.expense_year,
                    cost_allocation=self.cost_allocation,
                    description=self.description,
                    vat_portion=self.vat_portion,
                    vat_discount=self.vat_discount,
                    lbt_portion=self.lbt_portion,
                    lbt_discount=self.lbt_discount,
                )

        else:
            raise IntangibleException(
                f"Must divide with an instance of Tangible/Intangible/OPEX/ASR, "
                f"integer or a float; "
                f"{other}({other.__class__.__qualname__}) is not an instance "
                f"of Tangible/Intangible/OPEX/ASR nor an integer nor a float."
            )


@dataclass
class OPEX(GeneralCost):
    """
    Manages an OPEX asset.

    Parameters
    ----------
    fixed_cost : np.ndarray
        An array representing the fixed cost of an OPEX asset.
    prod_rate: np.ndarray
        The production rate of a particular fluid type.
    cost_per_volume: np.ndarray
        Cost associated with production of a particular fluid type.
    """

    fixed_cost: np.ndarray = field(default=None)
    prod_rate: np.ndarray = field(default=None)
    cost_per_volume: np.ndarray = field(default=None)

    # Attribute to be defined later on
    variable_cost: np.ndarray = field(default=None, init=False)
    cost: np.ndarray = field(default=None, init=False, repr=False)

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

        # Configure VAT portion
        if self.vat_portion is None:
            self.vat_portion = np.ones_like(self.fixed_cost)

        if self.vat_portion is not None:
            if not isinstance(self.vat_portion, np.ndarray):
                raise OPEXException(
                    f"Attribute VAT portion must be a numpy ndarray; "
                    f"VAT portion ({self.vat_portion}) is of datatype "
                    f"{self.vat_portion.__class__.__qualname__}."
                )

        # Configure LBT portion
        if self.lbt_portion is None:
            self.lbt_portion = np.ones_like(self.fixed_cost)

        if self.lbt_portion is not None:
            if not isinstance(self.lbt_portion, np.ndarray):
                raise OPEXException(
                    f"Attribute LBT portion must be a numpy ndarray; "
                    f"LBT portion ({self.lbt_portion}) is of datatype "
                    f"{self.lbt_portion.__class__.__qualname__}."
                )

        # Configure VAT discount
        if isinstance(self.vat_discount, float):
            self.vat_discount = np.repeat(self.vat_discount, len(self.fixed_cost))

        if isinstance(self.vat_discount, np.ndarray):
            if len(self.vat_discount) != len(self.fixed_cost):
                raise OPEXException(
                    f"Unequal length of array: "
                    f"VAT discount: ({len(self.vat_discount)}), "
                    f"Fixed cost: ({len(self.fixed_cost)})."
                )

        # Configure LBT discount
        if isinstance(self.lbt_discount, float):
            self.lbt_discount = np.repeat(self.lbt_discount, len(self.fixed_cost))

        if isinstance(self.lbt_discount, np.ndarray):
            if len(self.lbt_discount) != len(self.fixed_cost):
                raise OPEXException(
                    f"Unequal length of array: "
                    f"LBT discount: ({len(self.lbt_discount)}), "
                    f"Fixed cost: ({len(self.fixed_cost)})."
                )

        # Configure description data
        if self.description is None:
            self.description = [" " for _ in range(len(self.fixed_cost))]

        if self.description is not None:
            if not isinstance(self.description, list):
                raise OPEXException(
                    f"Attribute description must be a list; "
                    f"description ({self.description.__class__.__qualname__}) "
                    f"is not a list."
                )

        # Configure cost_allocation data
        if self.cost_allocation is None:
            self.cost_allocation = [FluidType.OIL for _ in range(len(self.fixed_cost))]

        if self.cost_allocation is not None:
            if not isinstance(self.cost_allocation, list):
                raise OPEXException(
                    f"Attribute cost_allocation must be a list. "
                    f"cost_allocation ({self.cost_allocation.__class__.__qualname__}) "
                    f"is not a list."
                )

        # User provides both prod_rate and cost_per_volume data
        if self.prod_rate is not None and self.cost_per_volume is not None:

            # Check input data for unequal length
            arr_length = len(self.fixed_cost)

            if not all(
                len(i) == arr_length
                for i in [
                    self.expense_year,
                    self.prod_rate,
                    self.cost_per_volume,
                    self.cost_allocation,
                    self.description,
                    self.vat_portion,
                    self.lbt_portion,
                ]
            ):
                raise OPEXException(
                    f"Unequal length of array: "
                    f"fixed_cost: {len(self.fixed_cost)}, "
                    f"expense_year: {len(self.expense_year)}, "
                    f"prod_rate: {len(self.prod_rate)}, "
                    f"cost_per_volume: {len(self.cost_per_volume)}, "
                    f"cost_allocation: {len(self.cost_allocation)}, "
                    f"description: {len(self.description)}, "
                    f"vat_portion: {len(self.vat_portion)}, "
                    f"lbt_portion: {len(self.lbt_portion)}."
                )

            # Specify attribute variable_cost
            self.variable_cost = self.prod_rate * self.cost_per_volume

        # User only provides prod_rate data
        if self.prod_rate is not None and self.cost_per_volume is None:
            raise OPEXException(f"cost_per_volume data is missing")

        # User only provides cost_per_volume data
        if self.prod_rate is None and self.cost_per_volume is not None:
            raise OPEXException(f"prod_rate data is missing")

        # User does not provide both prod_rate and cost_per_volume data
        if self.prod_rate is None and self.cost_per_volume is None:
            self.prod_rate = np.zeros_like(self.fixed_cost)
            self.cost_per_volume = np.zeros_like(self.fixed_cost)
            self.variable_cost = np.zeros_like(self.fixed_cost)

        # Define cost
        self.cost = self.fixed_cost + self.variable_cost

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

    def __eq__(self, other):
        # Between two instances of OPEX
        if isinstance(other, OPEX):
            return all(
                (
                    np.allclose(self.fixed_cost, other.fixed_cost),
                    np.allclose(self.variable_cost, other.variable_cost),
                    np.allclose(self.expense_year, other.expense_year),
                    np.allclose(self.vat_portion, other.vat_portion),
                    np.allclose(self.vat_discount, other.vat_discount),
                    np.allclose(self.lbt_portion, other.lbt_portion),
                    np.allclose(self.lbt_discount, other.lbt_discount),
                    self.cost_allocation == other.cost_allocation,
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
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float."
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
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float."
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
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float."
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
                f"Tangible/Intangible/OPEX/ASR, an integer, or a float."
            )

    def __add__(self, other):
        # Only allows addition between an instance of OPEX and another instance of OPEX
        if isinstance(other, OPEX):
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            expense_year_combined = np.concatenate((self.expense_year, other.expense_year))
            cost_allocation_combined = self.cost_allocation + other.cost_allocation
            description_combined = self.description + other.description
            vat_portion_combined = np.concatenate((self.vat_portion, other.vat_portion))
            vat_discount_combined = np.concatenate((self.vat_discount, other.vat_discount))
            lbt_portion_combined = np.concatenate((self.lbt_portion, other.lbt_portion))
            lbt_discount_combined = np.concatenate((self.lbt_discount, other.lbt_discount))
            fixed_cost_combined = np.concatenate((self.fixed_cost, other.fixed_cost))
            prod_rate_combined = np.concatenate((self.prod_rate, other.prod_rate))
            cost_per_volume_combined = np.concatenate((self.cost_per_volume, other.cost_per_volume))

            return OPEX(
                start_year=start_year_combined,
                end_year=end_year_combined,
                expense_year=expense_year_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
                vat_portion=vat_portion_combined,
                vat_discount=vat_discount_combined,
                lbt_portion=lbt_portion_combined,
                lbt_discount=lbt_discount_combined,
                fixed_cost=fixed_cost_combined,
                prod_rate=prod_rate_combined,
                cost_per_volume=cost_per_volume_combined,
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
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            expense_year_combined = np.concatenate((self.expense_year, other.expense_year))
            cost_allocation_combined = self.cost_allocation + other.cost_allocation
            description_combined = self.description + other.description
            vat_portion_combined = np.concatenate((self.vat_portion, other.vat_portion))
            vat_discount_combined = np.concatenate((self.vat_discount, other.vat_discount))
            lbt_portion_combined = np.concatenate((self.lbt_portion, other.lbt_portion))
            lbt_discount_combined = np.concatenate((self.lbt_discount, other.lbt_discount))
            fixed_cost_combined = np.concatenate((self.fixed_cost, -other.fixed_cost))
            prod_rate_combined = np.concatenate((self.prod_rate, -other.prod_rate))
            cost_per_volume_combined = np.concatenate((self.cost_per_volume, other.cost_per_volume))

            return OPEX(
                start_year=start_year_combined,
                end_year=end_year_combined,
                expense_year=expense_year_combined,
                cost_allocation=cost_allocation_combined,
                description=description_combined,
                vat_portion=vat_portion_combined,
                vat_discount=vat_discount_combined,
                lbt_portion=lbt_portion_combined,
                lbt_discount=lbt_discount_combined,
                fixed_cost=fixed_cost_combined,
                prod_rate=prod_rate_combined,
                cost_per_volume=cost_per_volume_combined,
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
                expense_year=self.expense_year,
                cost_allocation=self.cost_allocation,
                description=self.description,
                vat_portion=self.vat_portion,
                vat_discount=self.vat_discount,
                lbt_portion=self.lbt_portion,
                lbt_discount=self.lbt_discount,
                fixed_cost=self.fixed_cost * other,
                prod_rate=self.prod_rate * other,
                cost_per_volume=self.cost_per_volume,
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
                    expense_year=self.expense_year,
                    cost_allocation=self.cost_allocation,
                    description=self.description,
                    vat_portion=self.vat_portion,
                    vat_discount=self.vat_discount,
                    lbt_portion=self.lbt_portion,
                    lbt_discount=self.lbt_discount,
                    fixed_cost=self.fixed_cost / other,
                    prod_rate=self.prod_rate / other,
                    cost_per_volume=self.cost_per_volume,
                )

        else:
            raise OPEXException(
                f"Must divide with an instance of Tangible/Intangible/OPEX/ASR, integer or a float; "
                f"{other}({other.__class__.__qualname__}) is not an instance "
                f"of Tangible/Intangible/OPEX/ASR nor an integer nor a float."
            )


@dataclass
class ASR(GeneralCost):
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
    cost_allocation : list[FluidType]
        A list representing the cost allocation of an ASR asset.
    description: list[str]
        A list of string description regarding the associated ASR cost.
    """

    start_year: int
    end_year: int
    cost: np.ndarray
    expense_year: np.ndarray
    cost_allocation: list[FluidType] = field(default=None)
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

        # Configure description data
        if self.description is None:
            self.description = [" " for _ in range(len(self.cost))]

        if self.description is not None:
            if not isinstance(self.description, list):
                raise ASRException(
                    f"Attribute description must be a list; "
                    f"description ({self.description.__class__.__qualname__}) "
                    f"is not a list."
                )

        # Configure cost_allocation data
        if self.cost_allocation is None:
            self.cost_allocation = [FluidType.OIL for _ in range(len(self.cost))]

        if self.cost_allocation is not None:
            if not isinstance(self.cost_allocation, list):
                raise ASRException(
                    f"Attribute cost_allocation must be a list. "
                    f"cost_allocation ({self.cost_allocation.__class__.__qualname__}) "
                    f"is not a list."
                )

        # Check input data for unequal length
        arr_length = len(self.cost)

        if not all(
            len(arr) == arr_length
            for arr in [
                self.expense_year,
                self.cost_allocation,
                self.description,
            ]
        ):
            raise ASRException(
                f"Unequal length of array: "
                f"cost: {len(self.cost)}, "
                f"expense_year: {len(self.expense_year)}, "
                f"cost_allocation: {len(self.cost_allocation)}, "
                f"description: {len(self.description)}."
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
        inflation_rate: np.ndarray | float | int = 0.0,
        year_now: int = None,
    ) -> np.ndarray:
        """
        Calculate the future cost of an asset.

        Parameters
        ----------
        future_rate : float, optional
            The future rate used in cost calculation (default is 0.02).
        inflation_rate : np.ndarray or float or int, optional
            The inflation rate to apply. Can be a single value or an array (default is 0).
        year_now : int
            The reference year for inflation calculation.

        Returns
        -------
        np.ndarray
            An array containing the future cost of the asset.

        Notes
        -----
        This function calculates the future cost of an asset, taking into
        account inflation scheme.
        """
        if year_now is None:
            year_now = self.start_year

        cost_adjusted = apply_cost_adjustment(
            start_year=self.start_year,
            end_year=self.end_year,
            cost=self.cost,
            expense_year=self.expense_year,
            project_duration=self.project_duration,
            project_years=self.project_years,
            year_now=year_now,
            inflation_rate=inflation_rate,
        )

        return cost_adjusted * np.power(
            (1 + future_rate), self.end_year - self.expense_year + 1
        )

    def expenditures(
        self,
        future_rate: float = 0.02,
        inflation_rate: np.ndarray | float | int = 0.0,
        year_now: int = None,
    ) -> np.ndarray:
        """
        Calculate ASR expenditures per year.

        This method calculates the ASR expenditures per year
        based on the expense year and cost data provided.

        Parameters
        ----------
        future_rate : float, optional
            The future rate used in cost calculation (default is 0.02).
        inflation_rate : np.ndarray or float or int, optional
            The inflation rate to apply. Can be a single value or an array (default is 0).
        year_now : int
            The reference year for inflation calculation.

        Returns
        -------
        expenses: np.ndarray
            An array depicting ASR expenditures each year, adjusted by
            inflation (if any).

        Notes
        -----
        This method calculates ASR expenditures while considering inflation scheme.
        """
        if year_now is None:
            year_now = self.start_year

        # Distance of expense year from the end year of the project
        cost_duration = self.end_year - self.expense_year + 1

        # Cost allocation: cost distribution per year
        cost_alloc = (
            self.future_cost(
                future_rate=future_rate,
                inflation_rate=inflation_rate,
                year_now=year_now,
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

    def __eq__(self, other):
        # Between two instances of ASR
        if isinstance(other, ASR):
            return all(
                (
                    np.allclose(self.cost, other.cost),
                    np.allclose(self.expense_year, other.expense_year),
                    self.cost_allocation == other.cost_allocation,
                )
            )

        # Between an instance of ASR and an integer/float
        elif isinstance(other, (int, float)):
            return np.sum(self.cost) == other

        else:
            return False

    def __add__(self, other):
        # Only allows addition between an instance of ASR and another instance of ASR
        if isinstance(other, ASR):
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            description_combined = self.description + other.description

            return ASR(
                start_year=start_year_combined,
                end_year=end_year_combined,
                cost=np.concatenate((self.cost, other.cost)),
                expense_year=np.concatenate((self.expense_year, other.expense_year)),
                cost_allocation=self.cost_allocation + other.cost_allocation,
                description=description_combined,
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
            start_year_combined = min(self.start_year, other.start_year)
            end_year_combined = max(self.end_year, other.end_year)
            description_combined = self.description + other.description

            return ASR(
                start_year=start_year_combined,
                end_year=end_year_combined,
                cost=np.concatenate((self.cost, -other.cost)),
                expense_year=np.concatenate((self.expense_year, other.expense_year)),
                cost_allocation=self.cost_allocation + other.cost_allocation,
                description=description_combined,
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
                    description=self.description,
                )

        else:
            raise ASRException(
                f"Must divide with an instance of Tangible/Intangible/OPEX/ASR, integer or a float; "
                f"{other}({other.__class__.__qualname__}) is not an instance "
                f"of Tangible/Intangible/OPEX/ASR nor an integer nor a float."
            )
