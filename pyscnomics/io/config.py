"""
Prepare data input loaded from a target Excel file.
"""

from dataclasses import dataclass, field
from datetime import date
import numpy as np
import pandas as pd


class GeneralConfigDataException(Exception):
    """ Exception to be raised for inappropriate use of GeneralConfigData class """

    pass


class FiscalConfigDataException(Exception):
    """ Exception to be raised for inappropriate use of FiscalConfigData class """

    pass


class OilLiftingDataException(Exception):
    """ Exception to be raised for an inappropriate use of OilLiftingData class """

    pass


class GasLiftingDataException(Exception):
    """ Exception to be raised for an inappropriate use of GasLiftingData class """

    pass


class LPGPropaneLiftingDataException(Exception):
    """ Exception to be raised for an inappropriate use of LPGPropaneLiftingData class """

    pass


class LPGButaneLiftingDataException(Exception):
    """ Exception to be raised for an inappropriate use of LPGButaneLiftingData class """

    pass


class SulfurLiftingDataException(Exception):
    """ Exception to be raised for an inappropriate use of SulfurLiftingData class """

    pass


class ElectricityLiftingDataException(Exception):
    """ Exception to be raised for an inappropriate use of ElectricityLiftingData class """

    pass


class CO2LiftingDataException(Exception):
    """ Exception to be raised for an inappropriate use of CO2LiftingData class """

    pass


class TangibleCostDataException(Exception):
    """ Exception to be raised for an inappropriate use of TangibleCostData class """

    pass


class IntangibleCostDataException(Exception):
    """ Exception to be raised for an inappropriate use of IntangibleCostData class """

    pass


class OPEXDataException(Exception):
    """ Exception to be raised for an inappropriate use of OPEXData class """

    pass


class ASRCostDataException(Exception):
    """ Exception to be raised for an inappropriate use of ASRCostData class """
    pass


class MonteCarloDataException(Exception):
    """ Exception to be raised for an inappropriate use of MonteCarloData class """

    pass


class OptimizationDataException(Exception):
    """ Exception to be raised for an inappropriate use of OptimizationData class """

    pass


@dataclass
class GeneralConfigData:
    """
    A dataclass representing general configuration information for an oil and gas economic project.

    Attributes
    ----------
    start_date_project: date
        The start date of the project.
    end_date_project: date
        The end date of the project.
    type_of_contract: str
        The type of contract associated with the project.
    oil_onstream_date: date, optional
        The onstream date for oil production (defaults to None).
    gas_onstream_date: date, optional
        The onstream date for gas production (defaults to None).
    discount_rate_start_year: int, optional
        The start year for applying discount rate (defaults to None).
    discount_rate: float
        The discount rate applied to project cash flows (defaults to 0.1).
    inflation_rate_applied_to: str, optional
        The attribute to which inflation rate is applied (defaults to None).
    vat_discount: float
        The value-added tax discount (defaults to 0.0).
    lbt_discount: float
        The land and building tax discount (defaults to 0.0).
    """

    start_date_project: date
    end_date_project: date
    type_of_contract: str
    oil_onstream_date: date = field(default=None)
    gas_onstream_date: date = field(default=None)
    discount_rate_start_year: int = field(default=None)
    discount_rate: float = field(default=0.1)
    inflation_rate_applied_to: str = field(default=None)
    vat_discount: float = field(default=0.0)
    lbt_discount: float = field(default=0.0)
    gsa_number: int = field(default=1)

    # Attributes associated with duration of the project
    project_duration: int = field(default=None, init=False)
    project_years: np.ndarray = field(default=None, init=False)

    def __post_init__(self):
        # Check for inappropriate start and end year project
        if self.end_date_project.year >= self.start_date_project.year:
            self.project_duration = self.end_date_project.year - self.start_date_project.year + 1
            self.project_years = np.arange(
                self.start_date_project.year, self.end_date_project.year + 1, 1
            )

        else:
            raise GeneralConfigDataException(
                f"start year {self.start_date_project.year} "
                f"is after the end year: {self.end_date_project.year}"
            )


@dataclass
class FiscalConfigData:
    """
    A dataclass representing fiscal configuration information for an oil and gas economic project.

    Attributes
    ----------
    tax_mode: str
        The tax mode for fiscal configuration.
    tax_payment_method: str
        The method of tax payment.
    tax_psc_cost_recovery: str
        The basis of tax configuration for PSC cost recovery.
    npv_mode: str
        The Net Present Value (NPV) calculation mode.
    future_rate_asr: float
        The future rate used in ASR cost calculation.
    depreciation_method: str
        The depreciation method to use.
    multi_tax: dict
        Attribute that stores information of year and tax_rate for multi value case.
    project_years: np.ndarray
        The array of project years
    """
    tax_mode: str
    tax_rate_input: float = field(repr=False)
    tax_payment_method: str
    tax_psc_cost_recovery: str
    npv_mode: str
    discounting_mode: str
    future_rate_asr: float
    depreciation_method: str
    inflation_rate_mode: str
    inflation_rate_input: float = field(repr=False)
    multi_tax: dict = field(repr=False)
    multi_inflation: dict = field(repr=False)
    project_years: np.ndarray

    # Attributes to be defined later
    tax_rate: None | float | np.ndarray = field(default=None, init=False)
    inflation_rate: float | np.ndarray = field(default=None, init=False)

    def __post_init__(self):
        # Configure attribute tax_rate
        if self.tax_mode == "User Input - Single Value":
            if pd.isna(self.tax_rate_input):
                self.tax_rate = 0.44
            else:
                self.tax_rate = self.tax_rate_input

        elif self.tax_mode == "Nailed Down" or self.tax_mode == "Prevailing":
            self.tax_rate = None

        elif self.tax_mode == "User Input - Multi Value":
            # Filter dict 'self.multi_tax' for NaN values
            multi_tax_adj = {
                key: np.array(list(filter(lambda i: i is not np.nan, self.multi_tax[key])))
                for key in self.multi_tax.keys()
            }

            # Raise error for unequal length of 'year' and 'rate' in 'multi_tax_adj'
            if len(multi_tax_adj["year"]) != len(multi_tax_adj["rate"]):
                raise FiscalConfigDataException(
                    f"Unequal number of arrays: "
                    f"year: {len(multi_tax_adj['year'])}, "
                    f"tax_rate: {len(multi_tax_adj['rate'])}."
                )

            # Specify the minimum and maximum years
            min_year_tax = min(self.project_years)
            max_year_tax = max(self.project_years)

            if min(multi_tax_adj["year"]) < min(self.project_years):
                min_year_tax = min(multi_tax_adj["year"])

            if max(multi_tax_adj["year"]) > max(self.project_years):
                max_year_tax = max(multi_tax_adj["year"])

            # Create new arrays of 'year' and 'rate'
            multi_tax_new = {
                "year": np.arange(min_year_tax, max_year_tax + 1, 1),
                "rate": np.bincount(
                    multi_tax_adj["year"] - min_year_tax,
                    weights=multi_tax_adj["rate"]
                )
            }

            # Specify the index location of multi_val_adj["year"] in array multi_val_new["year"]
            id_tax = np.array(
                [
                    np.argwhere(multi_tax_new["year"] == val).ravel()
                    for val in multi_tax_adj["year"]
                ]
            ).ravel()

            # Modify the value of multi_tax_new["rate"]
            for i, val in enumerate(multi_tax_adj["rate"]):
                if i == (len(multi_tax_adj["rate"]) - 1):
                    break
                multi_tax_new["rate"][id_tax[i]:id_tax[i + 1]] = multi_tax_adj["rate"][i]

            # Add values to the right side of multi_tax_new
            if len(multi_tax_new["year"]) > len(multi_tax_new["rate"]):
                fill_num = len(multi_tax_new["year"]) - len(multi_tax_new["rate"])
                fill_right = np.repeat(multi_tax_adj["rate"][-1], fill_num)
                multi_tax_new["rate"] = np.concatenate((multi_tax_new["rate"], fill_right))

            # Add values to the left side of multi_tax_new
            if id_tax[0] > 0:
                fill_left = np.repeat(multi_tax_adj["rate"][0], id_tax[0])
                multi_tax_new["rate"][0:id_tax[0]] = fill_left

            # Capture 'year' and 'rate' in accordance with project_years
            id_tax_new = np.array(
                [
                    np.argwhere(multi_tax_new["year"] == i).ravel()
                    for i in [min(self.project_years), max(self.project_years)]
                ]
            ).ravel()

            multi_tax_new["year_new"] = (
                multi_tax_new["year"][id_tax_new[0]:int(id_tax_new[1] + 1)]
            )

            multi_tax_new["rate_new"] = (
                multi_tax_new["rate"][id_tax_new[0]:int(id_tax_new[1] + 1)]
            )

            self.tax_rate = multi_tax_new["rate_new"]

        # Configure attribute inflation_rate
        if self.inflation_rate_mode == "User Input - Single Value":
            if pd.isna(self.inflation_rate_input):
                self.inflation_rate = 0.02
            else:
                self.inflation_rate = self.inflation_rate_input

        elif self.inflation_rate_mode == "User Input - Multi Value":
            # Filter dict 'self.multi_inflation' for NaN values
            multi_inflation_adj = {
                key: np.array(list(filter(lambda i: i is not np.nan, self.multi_inflation[key])))
                for key in self.multi_inflation.keys()
            }

            # Raise error for unequal length of 'year' and 'rate' in 'multi_inflation_adj'
            if len(multi_inflation_adj["year"]) != len(multi_inflation_adj["rate"]):
                raise FiscalConfigDataException(
                    f"Unequal number of arrays: "
                    f"year: {len(multi_inflation_adj['year'])}, "
                    f"tax_rate: {len(multi_inflation_adj['rate'])}."
                )

            # Specify the minimum and maximum years
            min_year_inflation = min(self.project_years)
            max_year_inflation = max(self.project_years)

            if min(multi_inflation_adj["year"]) < min(self.project_years):
                min_year_inflation = min(multi_inflation_adj["year"])

            if max(multi_inflation_adj["year"]) > max(self.project_years):
                max_year_inflation = max(multi_inflation_adj["year"])

            # Create new arrays of 'year' and 'rate'
            multi_inflation_new = {
                "year": np.arange(min_year_inflation, max_year_inflation + 1, 1),
                "rate": np.bincount(
                    multi_inflation_adj["year"] - min_year_inflation,
                    weights=multi_inflation_adj["rate"]
                )
            }

            # Specify the index location of multi_inflation_adj["year"] in
            # multi_inflation_new["year"]
            id_inflation = np.array(
                [
                    np.argwhere(multi_inflation_new["year"] == val).ravel() for val in
                    multi_inflation_adj["year"]
                ]
            ).ravel()

            # Modify the value of multi_inflation_new["rate"]
            for i, val in enumerate(multi_inflation_adj["rate"]):
                if i == (len(multi_inflation_adj["rate"]) - 1):
                    break
                (
                    multi_inflation_new["rate"]
                    [id_inflation[i]:id_inflation[i + 1]]
                ) = multi_inflation_adj["rate"][i]

            # Add values to the right side of multi_inflation_new["rate"]
            if len(multi_inflation_new["year"]) > len(multi_inflation_new["rate"]):
                fill_num_infl = len(multi_inflation_new["year"]) - len(multi_inflation_new["rate"])
                fill_right_infl = np.repeat(multi_inflation_adj["rate"][-1], fill_num_infl)
                multi_inflation_new["rate"] = np.concatenate(
                    (multi_inflation_new["rate"], fill_right_infl)
                )

            # Add values to the left side of multi_inflation_new["rate"]
            if id_inflation[0] > 0:
                fill_left_infl = np.repeat(multi_inflation_adj["rate"][0], id_inflation[0])
                multi_inflation_new["rate"][0:id_inflation[0]] = fill_left_infl

            # Capture "year" and "rate" in accordance with project_years
            id_inflation_new = np.array(
                [
                    np.argwhere(multi_inflation_new["year"] == i).ravel()
                    for i in [min(self.project_years), max(self.project_years)]
                ]
            ).ravel()

            multi_inflation_new["year_new"] = (
                multi_inflation_new["year"][id_inflation_new[0]:int(id_inflation_new[1] + 1)]
            )

            multi_inflation_new["rate_new"] = (
                multi_inflation_new["rate"][id_inflation_new[0]:int(id_inflation_new[1] + 1)]
            )

            self.inflation_rate = multi_inflation_new["rate_new"]


@dataclass
class OilLiftingData:
    """
    A dataclass representing oil lifting information for an oil and gas economic project.

    Attributes
    ----------
    project_duration: int
        The duration of the project.
    project_years: numpy.ndarray
        An array representing the project years.
    prod_year: dict
        Dictionary containing production years data.
    oil_lifting_rate : dict
        Dictionary containing oil lifting rate data.
    oil_price : dict
        Dictionary containing oil price data.
    condensate_lifting_rate : dict
        Dictionary containing condensate lifting rate data.
    condensate_price : dict
        Dictionary containing condensate price data.

    Notes
    -----
    This dataclass is used to store and organize information related to oil lifting.
    """

    prod_year: dict
    oil_lifting_rate: dict
    oil_price: dict
    condensate_lifting_rate: dict
    condensate_price: dict

    # Attributes associated with project duration
    project_duration: int
    project_years: np.ndarray

    def __post_init__(self):
        # Prepare attribute prod_year
        if not isinstance(self.prod_year, dict):
            raise OilLiftingDataException(
                f"Attribute prod_year must be provided in the form of dictionary. "
                f"The current datatype of prod_year is "
                f"{self.prod_year.__class__.__qualname__}"
            )

        for i in self.prod_year.keys():
            if self.prod_year[i] is None:
                self.prod_year[i] = self.project_years

        # Prepare attribute oil_lifting_rate
        if not isinstance(self.oil_lifting_rate, dict):
            raise OilLiftingDataException(
                f"Attribute oil_lifting_rate must be provided in the form of dictionary. "
                f"The current datatype of oil_lifting_rate is "
                f"{self.oil_lifting_rate.__class__.__qualname__}"
            )

        for i in self.oil_lifting_rate.keys():
            if self.oil_lifting_rate[i] is None:
                self.oil_lifting_rate[i] = np.zeros_like(self.project_years)

        # Prepare attribute oil_price
        if not isinstance(self.oil_price, dict):
            raise OilLiftingDataException(
                f"Attribute oil_price must be provided in the form of dictionary. "
                f"The current datatype of oil_price is "
                f"{self.oil_price.__class__.__qualname__}"
            )

        for i in self.oil_price.keys():
            if self.oil_price[i] is None:
                self.oil_price[i] = np.zeros_like(self.project_years)

        # Prepare attribute condensate_lifting_rate
        if not isinstance(self.condensate_lifting_rate, dict):
            raise OilLiftingDataException(
                f"Attribute condensate_lifting_rate must be provided in the form of dictionary. "
                f"The current datatype of condensate_lifting_rate is "
                f"{self.condensate_lifting_rate.__class__.__qualname__}"
            )

        for i in self.condensate_lifting_rate.keys():
            if self.condensate_lifting_rate[i] is None:
                self.condensate_lifting_rate[i] = np.zeros_like(self.project_years)

        # Prepare attribute condensate_price
        if not isinstance(self.condensate_price, dict):
            raise OilLiftingDataException(
                f"Attribute condensate_price must be provided in the form of dictionary. "
                f"The current datatype of condensate_price is "
                f"{self.condensate_price.__class__.__qualname__}"
            )

        for i in self.condensate_price.keys():
            if self.condensate_price[i] is None:
                self.condensate_price[i] = np.zeros_like(self.project_years)


@dataclass
class GasLiftingData:
    """
    A dataclass representing gas lifting information for an oil and gas economic project.

    Attributes
    ----------
    gas_gsa_number: int
        The number of GSA.
    prod_year: dict
        Dictionary containing production years data.
    gas_prod_rate: dict
        Dictionary containing gas lifting rate data.
    gas_gsa_lifting_rate: dict
        Dictionary containing gas GSA lifting rate data.
    gas_gsa_ghv: dict
        Dictionary containing gas GSA ghv data.
    gas_gsa_price: dict
        Dictionary containing gas GSA price data.
    project_duration: int
        The duration of the project.
    project_years: numpy.ndarray
        An array representing the project years.
    """

    gas_gsa_number: int
    prod_year: dict
    gas_prod_rate: dict
    gas_gsa_lifting_rate: dict
    gas_gsa_ghv: dict
    gas_gsa_price: dict

    # Attributes associated with project duration
    project_duration: int
    project_years: np.ndarray

    def __post_init__(self):
        # Prepare attribute prod_year
        if not isinstance(self.prod_year, dict):
            raise GasLiftingDataException(
                f"Attribute prod_year must be provided in the form of dictionary. "
                f"The current datatype of prod_year is "
                f"{self.prod_year.__class__.__qualname__}"
            )

        for i in self.prod_year.keys():
            if self.prod_year[i] is None:
                self.prod_year[i] = self.project_years

        # Prepare attribute gas_prod_rate
        if not isinstance(self.gas_prod_rate, dict):
            raise GasLiftingDataException(
                f"Attribute gas_prod_rate must be provided in the form of dictionary. "
                f"The current datatype of gas_prod_rate is "
                f"{self.gas_prod_rate.__class__.__qualname__}"
            )

        for i in self.gas_prod_rate.keys():
            if self.gas_prod_rate[i] is None:
                self.gas_prod_rate[i] = np.zeros_like(self.project_years)

        # Prepare attribute gas_gsa_lifting_rate
        if not isinstance(self.gas_gsa_lifting_rate, dict):
            raise GasLiftingDataException(
                f"Attribute gas_gsa_lifting_rate must be provided in the form of dictionary. "
                f"The current datatype of gas_gsa_lifting_rate is "
                f"{self.gas_gsa_lifting_rate.__class__.__qualname__}"
            )

        for i in self.gas_gsa_lifting_rate.keys():
            for j in self.gas_gsa_lifting_rate[i].keys():
                if self.gas_gsa_lifting_rate[i][j] is None:
                    self.gas_gsa_lifting_rate[i][j] = np.zeros_like(self.project_years)

        # Prepare attribute gas_gsa_ghv
        if not isinstance(self.gas_gsa_ghv, dict):
            raise GasLiftingDataException(
                f"Attribute gas_gsa_ghv must be provided in the form of dictionary. "
                f"The current datatype of gas_gsa_ghv is "
                f"{self.gas_gsa_ghv.__class__.__qualname__}"
            )

        for i in self.gas_gsa_ghv.keys():
            for j in self.gas_gsa_ghv[i].keys():
                if self.gas_gsa_ghv[i][j] is None:
                    self.gas_gsa_ghv[i][j] = np.zeros_like(self.project_years)

        # Prepare attribute gas_gsa_price
        if not isinstance(self.gas_gsa_price, dict):
            raise GasLiftingDataException(
                f"Attribute gas_gsa_price must be provided in the form of dictionary. "
                f"The current datatype of gas_gsa_price is "
                f"{self.gas_gsa_price.__class__.__qualname__}"
            )

        for i in self.gas_gsa_price.keys():
            for j in self.gas_gsa_price[i].keys():
                if self.gas_gsa_price[i][j] is None:
                    self.gas_gsa_price[i][j] = np.zeros_like(self.project_years)


@dataclass
class LPGPropaneLiftingData:
    """
    A dataclass representing LPG Propane lifting information for an oil and gas economic project.

    Attributes
    ----------
    prod_year: dict
        Dictionary containing production years data.
    lifting_rate: dict
        Dictionary containing LPG Propane lifting rate data.
    price: dict
        Dictionary containing LPG Propane price data.
    project_duration: int
        The duration of the project.
    project_years: numpy.ndarray
        An array representing the project years.
    """
    prod_year: dict
    lifting_rate: dict
    price: dict

    # Attributes associated with project duration
    project_duration: int
    project_years: np.ndarray

    def __post_init__(self):
        # Prepare attribute prod_year
        if not isinstance(self.prod_year, dict):
            raise LPGPropaneLiftingDataException(
                f"Attribute prod_year must be provided in the form of dictionary. "
                f"The current datatype of prod_year is "
                f"{self.prod_year.__class__.__qualname__}"
            )

        for i in self.prod_year.keys():
            if self.prod_year[i] is None:
                self.prod_year[i] = self.project_years

        # Prepare attribute lifting_rate
        if not isinstance(self.lifting_rate, dict):
            raise LPGPropaneLiftingDataException(
                f"Attribute lifting_rate must be provided in the form of dictionary. "
                f"The current datatype of lifting_rate is "
                f"{self.lifting_rate.__class__.__qualname__}"
            )

        for i in self.lifting_rate.keys():
            if self.lifting_rate[i] is None:
                self.lifting_rate[i] = np.zeros_like(self.project_years)

        # Prepare attribute price
        if not isinstance(self.price, dict):
            raise LPGPropaneLiftingDataException(
                f"Attribute price must be provided in the form of dictionary. "
                f"The current datatype of price is "
                f"{self.price.__class__.__qualname__}"
            )

        for i in self.price.keys():
            if self.price[i] is None:
                self.price[i] = np.zeros_like(self.project_years)


@dataclass
class LPGButaneLiftingData:
    """
    A dataclass representing LPG Butane lifting information for an oil and gas economic project.

    Attributes
    ----------
    prod_year: dict
        Dictionary containing production years data.
    lifting_rate: dict
        Dictionary containing LPG Butane lifting rate data.
    price: dict
        Dictionary containing LPG Butane price data.
    project_duration: int
        The duration of the project.
    project_years: numpy.ndarray
        An array representing the project years.
    """
    prod_year: dict
    lifting_rate: dict
    price: dict

    # Attributes associated with project duration
    project_duration: int
    project_years: np.ndarray

    def __post_init__(self):
        # Prepare attribute prod_year
        if not isinstance(self.prod_year, dict):
            raise LPGButaneLiftingDataException(
                f"Attribute prod_year must be provided in the form of dictionary. "
                f"The current datatype of prod_year is "
                f"{self.prod_year.__class__.__qualname__}"
            )

        for i in self.prod_year.keys():
            if self.prod_year[i] is None:
                self.prod_year[i] = self.project_years

        # Prepare attribute lifting_rate
        if not isinstance(self.lifting_rate, dict):
            raise LPGButaneLiftingDataException(
                f"Attribute lifting_rate must be provided in the form of dictionary. "
                f"The current datatype of lifting_rate is "
                f"{self.lifting_rate.__class__.__qualname__}"
            )

        for i in self.lifting_rate.keys():
            if self.lifting_rate[i] is None:
                self.lifting_rate[i] = np.zeros_like(self.project_years)

        # Prepare attribute price
        if not isinstance(self.price, dict):
            raise LPGButaneLiftingDataException(
                f"Attribute price must be provided in the form of dictionary. "
                f"The current datatype of price is "
                f"{self.price.__class__.__qualname__}"
            )

        for i in self.price.keys():
            if self.price[i] is None:
                self.price[i] = np.zeros_like(self.project_years)


@dataclass
class SulfurLiftingData:
    """
    A dataclass representing sulfur lifting information for an oil and gas economic project.

    Attributes
    ----------
    prod_year: dict
        Dictionary containing production years data.
    lifting_rate: dict
        Dictionary containing sulfur lifting rate data.
    price: dict
        Dictionary containing sulfur price data.
    project_duration: int
        The duration of the project.
    project_years: numpy.ndarray
        An array representing the project years.
    """
    prod_year: dict
    lifting_rate: dict
    price: dict

    # Attributes associated with project duration
    project_duration: int
    project_years: np.ndarray

    def __post_init__(self):
        # Prepare attribute prod_year
        if not isinstance(self.prod_year, dict):
            raise SulfurLiftingDataException(
                f"Attribute prod_year must be provided in the form of dictionary. "
                f"The current datatype of prod_year is "
                f"{self.prod_year.__class__.__qualname__}"
            )

        for i in self.prod_year.keys():
            if self.prod_year[i] is None:
                self.prod_year[i] = self.project_years

        # Prepare attribute lifting_rate
        if not isinstance(self.lifting_rate, dict):
            raise SulfurLiftingDataException(
                f"Attribute lifting_rate must be provided in the form of dictionary. "
                f"The current datatype of lifting_rate is "
                f"{self.lifting_rate.__class__.__qualname__}"
            )

        for i in self.lifting_rate.keys():
            if self.lifting_rate[i] is None:
                self.lifting_rate[i] = np.zeros_like(self.project_years)

        # Prepare attribute price
        if not isinstance(self.price, dict):
            raise SulfurLiftingDataException(
                f"Attribute price must be provided in the form of dictionary. "
                f"The current datatype of price is "
                f"{self.price.__class__.__qualname__}"
            )

        for i in self.price.keys():
            if self.price[i] is None:
                self.price[i] = np.zeros_like(self.project_years)


@dataclass
class ElectricityLiftingData:
    """
    A dataclass representing electricity lifting information for an oil and gas economic project.

    Attributes
    ----------
    prod_year: dict
        Dictionary containing production years data.
    lifting_rate: dict
        Dictionary containing electricity lifting rate data.
    price: dict
        Dictionary containing electricity price data.
    project_duration: int
        The duration of the project.
    project_years: numpy.ndarray
        An array representing the project years.
    """
    prod_year: dict
    lifting_rate: dict
    price: dict

    # Attributes associated with project duration
    project_duration: int
    project_years: np.ndarray

    def __post_init__(self):
        # Prepare attribute prod_year
        if not isinstance(self.prod_year, dict):
            raise ElectricityLiftingDataException(
                f"Attribute prod_year must be provided in the form of dictionary. "
                f"The current datatype of prod_year is "
                f"{self.prod_year.__class__.__qualname__}"
            )

        for i in self.prod_year.keys():
            if self.prod_year[i] is None:
                self.prod_year[i] = self.project_years

        # Prepare attribute lifting_rate
        if not isinstance(self.lifting_rate, dict):
            raise ElectricityLiftingDataException(
                f"Attribute lifting_rate must be provided in the form of dictionary. "
                f"The current datatype of lifting_rate is "
                f"{self.lifting_rate.__class__.__qualname__}"
            )

        for i in self.lifting_rate.keys():
            if self.lifting_rate[i] is None:
                self.lifting_rate[i] = np.zeros_like(self.project_years)

        # Prepare attribute price
        if not isinstance(self.price, dict):
            raise ElectricityLiftingDataException(
                f"Attribute price must be provided in the form of dictionary. "
                f"The current datatype of price is "
                f"{self.price.__class__.__qualname__}"
            )

        for i in self.price.keys():
            if self.price[i] is None:
                self.price[i] = np.zeros_like(self.project_years)


@dataclass
class CO2LiftingData:
    """
    A dataclass representing CO2 lifting information for an oil and gas economic project.

    Attributes
    ----------
    prod_year: dict
        Dictionary containing production years data.
    lifting_rate: dict
        Dictionary containing CO2 lifting rate data.
    price: dict
        Dictionary containing CO2 price data.
    project_duration: int
        The duration of the project.
    project_years: numpy.ndarray
        An array representing the project years.
    """
    prod_year: dict
    lifting_rate: dict
    price: dict

    # Attributes associated with project duration
    project_duration: int
    project_years: np.ndarray

    def __post_init__(self):
        # Prepare attribute prod_year
        if not isinstance(self.prod_year, dict):
            raise CO2LiftingDataException(
                f"Attribute prod_year must be provided in the form of dictionary. "
                f"The current datatype of prod_year is "
                f"{self.prod_year.__class__.__qualname__}"
            )

        for i in self.prod_year.keys():
            if self.prod_year[i] is None:
                self.prod_year[i] = self.project_years

        # Prepare attribute lifting_rate
        if not isinstance(self.lifting_rate, dict):
            raise CO2LiftingDataException(
                f"Attribute lifting_rate must be provided in the form of dictionary. "
                f"The current datatype of lifting_rate is "
                f"{self.lifting_rate.__class__.__qualname__}"
            )

        for i in self.lifting_rate.keys():
            if self.lifting_rate[i] is None:
                self.lifting_rate[i] = np.zeros_like(self.project_years)

        # Prepare attribute price
        if not isinstance(self.price, dict):
            raise CO2LiftingDataException(
                f"Attribute price must be provided in the form of dictionary. "
                f"The current datatype of price is "
                f"{self.price.__class__.__qualname__}"
            )

        for i in self.price.keys():
            if self.price[i] is None:
                self.price[i] = np.zeros_like(self.project_years)


@dataclass
class TangibleCostData:
    """
    A dataclass representing tangible cost information for an oil and gas economic project.

    Attributes
    ----------
    expense_year: np.ndarray
        Array of expense years.
    cost: np.ndarray
        Array of costs.
    cost_allocation: list
        List of cost allocations.
    pis_year: np.ndarray
        Array of PIS years.
    useful_life: np.ndarray
        Array of useful life.
    depreciation_factor: np.ndarray
        Array of depreciation factors.
    salvage_value: np.ndarray
        Array of salvage values.
    is_ic_applied: list
        List indicating whether IC (Investment Credit) is applied.
    vat_portion: np.ndarray
        Array of VAT (Value Added Tax) portions.
    lbt_portion: np.ndarray
        Array of LBT (Land and Building Tax) portions.
    description: list
        List of descriptions.
    data_length: int
        Length of the captured data.
    project_years: np.ndarray
        Array of project years.
    """
    expense_year: np.ndarray
    cost: np.ndarray
    cost_allocation: list
    pis_year: np.ndarray
    useful_life: np.ndarray
    depreciation_factor: np.ndarray
    salvage_value: np.ndarray
    is_ic_applied: list
    vat_portion: np.ndarray
    lbt_portion: np.ndarray
    description: list

    # Attribute associated with length of the captured data
    data_length: int

    # Attribute associated with project duration
    project_years: np.ndarray

    def __post_init__(self):
        # Prepare attribute expense_year
        if not isinstance(self.expense_year, np.ndarray):
            raise TangibleCostDataException(
                f"Expense year data must be given in the form of numpy.ndarray. "
                f"The current expense_year data is given in the form of: "
                f"({self.expense_year.__class__.__qualname__})."
            )

        expense_year_nan = np.argwhere(np.isnan(self.expense_year)).ravel()

        if len(expense_year_nan) > 0:
            raise TangibleCostDataException(
                f"Expense year data is incomplete. Please re-check the expense_year data. "
                f"The number of expense_year data must be ({self.data_length}), "
                f"not ({self.data_length - len(expense_year_nan)})."
            )

        else:
            self.expense_year = self.expense_year

        # Prepare attribute cost_allocation
        if not isinstance(self.cost_allocation, list):
            raise TangibleCostDataException(
                f"Cost allocation data must be given in the form of list. "
                f"The current cost_allocation data is given in the form of: "
                f"({self.cost_allocation.__class__.__qualname__})."
            )

        cost_allocation_nan = np.argwhere(pd.isna(self.cost_allocation)).ravel()

        if len(cost_allocation_nan) > 0:
            raise TangibleCostDataException(
                f"Cost allocation data is incomplete. "
                f"Please re-check cost_allocation data. "
                f"The number of cost_allocation data must be ({self.data_length}), "
                f"not ({self.data_length - len(cost_allocation_nan)})."
            )

        else:
            self.cost_allocation = self.cost_allocation

        # Prepare attribute cost
        if not isinstance(self.cost, np.ndarray):
            raise TangibleCostDataException(
                f"Cost data must be given in the form of numpy.ndarray. "
                f"The current cost data is given in the form of: "
                f"({self.cost.__class__.__qualname__})."
            )

        cost_nan = np.argwhere(np.isnan(self.cost)).ravel()

        if len(cost_nan) > 0:
            self.cost[cost_nan] = np.zeros(len(cost_nan))

        else:
            self.cost = self.cost

        # Prepare attribute pis_year
        if not isinstance(self.pis_year, np.ndarray):
            raise TangibleCostDataException(
                f"PIS year data must be given in the form of numpy.ndarray. "
                f"The current pis_year data is given in the form of: "
                f"({self.pis_year.__class__.__qualname__})."
            )

        pis_year_nan = np.argwhere(np.isnan(self.pis_year)).ravel()

        if len(pis_year_nan) > 0:
            self.pis_year[pis_year_nan] = self.expense_year[pis_year_nan]

        else:
            self.pis_year = self.pis_year

        # Prepare attribute useful_life
        if not isinstance(self.useful_life, np.ndarray):
            raise TangibleCostDataException(
                f"Useful life data must be given in the form of numpy.ndarray. "
                f"The current useful_life data is given in the form of: "
                f"({self.useful_life.__class__.__qualname__})."
            )

        useful_life_nan = np.argwhere(np.isnan(self.useful_life)).ravel()

        if len(useful_life_nan) > 0:
            self.useful_life[useful_life_nan] = np.repeat(5.0, len(useful_life_nan))

        else:
            self.useful_life = self.useful_life

        # Prepare attribute depreciation_factor
        if not isinstance(self.depreciation_factor, np.ndarray):
            raise TangibleCostDataException(
                f"Depreciation factor data must be given in the form of numpy.ndarray. "
                f"The current depreciation_factor data is given in the form of: "
                f"({self.depreciation_factor.__class__.__qualname__})."
            )

        depreciation_factor_nan = np.argwhere(np.isnan(self.depreciation_factor)).ravel()

        if len(depreciation_factor_nan) > 0:
            self.depreciation_factor[depreciation_factor_nan] = (
                np.repeat(0.5, len(depreciation_factor_nan))
            )

        else:
            self.depreciation_factor = self.depreciation_factor

        # Prepare attribute salvage_value
        if not isinstance(self.salvage_value, np.ndarray):
            raise TangibleCostDataException(
                f"Salvage value data must be given in the form of numpy.ndarray. "
                f"The current salvage_value data is given in the form of: "
                f"({self.salvage_value.__class__.__qualname__})."
            )

        salvage_value_nan = np.argwhere(np.isnan(self.salvage_value)).ravel()

        if len(salvage_value_nan) > 0:
            self.salvage_value[salvage_value_nan] = np.zeros(len(salvage_value_nan))

        else:
            self.salvage_value = self.salvage_value

        # Prepare attribute is_ic_applied
        if not isinstance(self.is_ic_applied, list):
            raise TangibleCostDataException(
                f"Is IC applied data must be given in the form of list. "
                f"The current is_ic_applied data is given in the form of: "
                f"({self.is_ic_applied.__class__.__qualname__})."
            )

        is_ic_applied_nan = np.argwhere(pd.isna(self.is_ic_applied)).ravel()

        if len(is_ic_applied_nan) > 0:
            for i in is_ic_applied_nan:
                self.is_ic_applied[i] = "No"

        else:
            self.is_ic_applied = self.is_ic_applied

        # Prepare attribute vat_portion
        if not isinstance(self.vat_portion, np.ndarray):
            raise TangibleCostDataException(
                f"VAT portion data must be given in the form of numpy.ndarray. "
                f"The current vat_portion data is given in the form of: "
                f"({self.vat_portion.__class__.__qualname__})."
            )

        vat_portion_nan = np.argwhere(np.isnan(self.vat_portion)).ravel()

        if len(vat_portion_nan) > 0:
            self.vat_portion[vat_portion_nan] = np.ones(len(vat_portion_nan))

        else:
            self.vat_portion = self.vat_portion

        # Prepare attribute lbt_portion
        if not isinstance(self.lbt_portion, np.ndarray):
            raise TangibleCostDataException(
                f"LBT portion data must be given in the form of numpy.ndarray. "
                f"The current lbt_portion data is given in the form of: "
                f"({self.lbt_portion.__class__.__qualname__})."
            )

        lbt_portion_nan = np.argwhere(np.isnan(self.lbt_portion)).ravel()

        if len(lbt_portion_nan) > 0:
            self.lbt_portion[lbt_portion_nan] = np.ones(len(lbt_portion_nan))

        else:
            self.lbt_portion = self.lbt_portion

        # Prepare attribute description
        if not isinstance(self.description, list):
            raise TangibleCostDataException(
                f"Description data must be given in the form of list. "
                f"The current description data is given in the form of: "
                f"({self.description.__class__.__qualname__})."
            )

        description_nan = np.argwhere(pd.isna(self.description)).ravel()

        if len(description_nan) > 0:
            for i in description_nan:
                self.description[i] = " "

        else:
            self.description = self.description


@dataclass
class IntangibleCostData:
    """
    A dataclass representing intangible cost information for an oil and gas economic project.

    Attributes
    ----------
    expense_year: np.ndarray
        Array of expense years.
    cost_allocation: list
        List of cost allocations.
    cost: np.ndarray
        Array of costs.
    vat_portion: np.ndarray
        Array of VAT (Value Added Tax) portions.
    lbt_portion: np.ndarray
        Array of LBT (Land and Building Tax) portions.
    description: list
        List of descriptions.
    data_length: int
        Length of the captured data.
    project_years: np.ndarray
        Array of project years.
    """
    expense_year: np.ndarray
    cost_allocation: list
    cost: np.ndarray
    vat_portion: np.ndarray
    lbt_portion: np.ndarray
    description: list

    # Attribute associated with length of the captured data
    data_length: int

    # Attribute associated with project duration
    project_years: np.ndarray

    def __post_init__(self):
        # Prepare attribute expense_year
        if not isinstance(self.expense_year, np.ndarray):
            raise IntangibleCostDataException(
                f"Expense year data must be given in the form of numpy.ndarray. "
                f"The current expense_year data is given in the form of: "
                f"({self.expense_year.__class__.__qualname__})."
            )

        expense_year_nan = np.argwhere(np.isnan(self.expense_year)).ravel()

        if len(expense_year_nan) > 0:
            raise IntangibleCostDataException(
                f"Expense year data is incomplete. Please re-check the expense_year data. "
                f"The number of expense_year data must be ({self.data_length}), "
                f"not ({self.data_length - len(expense_year_nan)})."
            )

        else:
            self.expense_year = self.expense_year

        # Prepare attribute cost_allocation
        if not isinstance(self.cost_allocation, list):
            raise IntangibleCostDataException(
                f"Cost allocation data must be given in the form of list. "
                f"The current cost_allocation data is given in the form of: "
                f"({self.cost_allocation.__class__.__qualname__})."
            )

        cost_allocation_nan = np.argwhere(pd.isna(self.cost_allocation)).ravel()

        if len(cost_allocation_nan) > 0:
            raise IntangibleCostDataException(
                f"Cost allocation data is incomplete. "
                f"Please re-check cost_allocation data. "
                f"The number of cost_allocation data must be ({self.data_length}), "
                f"not ({self.data_length - len(cost_allocation_nan)})."
            )

        else:
            self.cost_allocation = self.cost_allocation

        # Prepare attribute cost
        if not isinstance(self.cost, np.ndarray):
            raise IntangibleCostDataException(
                f"Cost data must be given in the form of numpy.ndarray. "
                f"The current cost data is given in the form of: "
                f"({self.cost.__class__.__qualname__})."
            )

        cost_nan = np.argwhere(np.isnan(self.cost)).ravel()

        if len(cost_nan) > 0:
            self.cost[cost_nan] = np.zeros(len(cost_nan))

        else:
            self.cost = self.cost

        # Prepare attribute vat_portion
        if not isinstance(self.vat_portion, np.ndarray):
            raise IntangibleCostDataException(
                f"VAT portion data must be given in the form of numpy.ndarray. "
                f"The current vat_portion data is given in the form of: "
                f"({self.vat_portion.__class__.__qualname__})."
            )

        vat_portion_nan = np.argwhere(np.isnan(self.vat_portion)).ravel()

        if len(vat_portion_nan) > 0:
            self.vat_portion[vat_portion_nan] = np.ones(len(vat_portion_nan))

        else:
            self.vat_portion = self.vat_portion

        # Prepare attribute lbt_portion
        if not isinstance(self.lbt_portion, np.ndarray):
            raise IntangibleCostDataException(
                f"LBT portion data must be given in the form of numpy.ndarray. "
                f"The current lbt_portion data is given in the form of: "
                f"({self.lbt_portion.__class__.__qualname__})."
            )

        lbt_portion_nan = np.argwhere(np.isnan(self.lbt_portion)).ravel()

        if len(lbt_portion_nan) > 0:
            self.lbt_portion[lbt_portion_nan] = np.ones(len(lbt_portion_nan))

        else:
            self.lbt_portion = self.lbt_portion

        # Prepare attribute description
        if not isinstance(self.description, list):
            raise IntangibleCostDataException(
                f"Description data must be given in the form of list. "
                f"The current description data is given in the form of: "
                f"({self.description.__class__.__qualname__})."
            )

        description_nan = np.argwhere(pd.isna(self.description)).ravel()

        if len(description_nan) > 0:
            for i in description_nan:
                self.description[i] = " "

        else:
            self.description = self.description


@dataclass
class OPEXData:
    """
    A dataclass representing opex information for an oil and gas economic project.

    Attributes
    ----------
    expense_year: np.ndarray
        Array of expense years.
    cost_allocation: list
        List of cost allocations.
    fixed_cost: np.ndarray
        Array of fixed costs.
    prod_rate: np.ndarray
        Array of production rates.
    cost_per_volume: np.ndarray
        Array of costs per volume.
    vat_portion: np.ndarray
        Array of VAT (Value Added Tax) portions.
    lbt_portion: np.ndarray
        Array of LBT (Land and Building Tax) portions.
    description: list
        List of descriptions.
    data_length: int
        Length of the captured data.
    project_years: np.ndarray
        Array of project years.
    """
    expense_year: np.ndarray
    cost_allocation: list
    fixed_cost: np.ndarray
    prod_rate: np.ndarray
    cost_per_volume: np.ndarray
    vat_portion: np.ndarray
    lbt_portion: np.ndarray
    description: list

    # Attribute associated with length of the captured data
    data_length: int

    # Attribute associated with project duration
    project_years: np.ndarray

    def __post_init__(self):
        # Prepare attribute expense year
        if not isinstance(self.expense_year, np.ndarray):
            raise OPEXDataException(
                f"Expense year data must be given in the form of numpy.ndarray. "
                f"The current expense_year data is given in the form of: "
                f"({self.expense_year.__class__.__qualname__})."
            )

        expense_year_nan = np.argwhere(np.isnan(self.expense_year)).ravel()

        if len(expense_year_nan) > 0:
            raise OPEXDataException(
                f"Expense year data is incomplete. Please re-check the expense_year data. "
                f"The number of expense_year data must be ({self.data_length}), "
                f"not ({self.data_length - len(expense_year_nan)})."
            )

        else:
            self.expense_year = self.expense_year

        # Prepare attribute cost_allocation
        if not isinstance(self.cost_allocation, list):
            raise OPEXDataException(
                f"Cost allocation data must be given in the form of list. "
                f"The current cost_allocation data is given in the form of: "
                f"({self.cost_allocation.__class__.__qualname__})."
            )

        cost_allocation_nan = np.argwhere(pd.isna(self.cost_allocation)).ravel()

        if len(cost_allocation_nan) > 0:
            raise OPEXDataException(
                f"Cost allocation data is incomplete. "
                f"Please re-check cost_allocation data. "
                f"The number of cost_allocation data must be ({self.data_length}), "
                f"not ({self.data_length - len(cost_allocation_nan)})."
            )

        else:
            self.cost_allocation = self.cost_allocation

        # Prepare attribute fixed_cost
        if not isinstance(self.fixed_cost, np.ndarray):
            raise OPEXDataException(
                f"Fixed cost data must be given in the form of numpy.ndarray. "
                f"The current fixed_cost data is given in the form of: "
                f"({self.fixed_cost.__class__.__qualname__})."
            )

        fixed_cost_nan = np.argwhere(np.isnan(self.fixed_cost)).ravel()

        if len(fixed_cost_nan) > 0:
            self.fixed_cost[fixed_cost_nan] = np.zeros(len(fixed_cost_nan))

        else:
            self.fixed_cost = self.fixed_cost

        # Prepare attribute prod_rate
        if not isinstance(self.prod_rate, np.ndarray):
            raise OPEXDataException(
                f"Production rate data must be given in the form of numpy.ndarray. "
                f"The current prod_rate data is given in the form of: "
                f"({self.prod_rate.__class__.__qualname__})."
            )

        prod_rate_nan = np.argwhere(np.isnan(self.prod_rate)).ravel()

        if len(prod_rate_nan) > 0:
            self.prod_rate[prod_rate_nan] = np.zeros(len(prod_rate_nan))

        else:
            self.prod_rate = self.prod_rate

        # Prepare attribute cost_per_volume
        if not isinstance(self.cost_per_volume, np.ndarray):
            raise OPEXDataException(
                f"Cost per volume data must be given in the form of numpy.ndarray. "
                f"The current cost_per_volume data is given in the form of: "
                f"({self.cost_per_volume.__class__.__qualname__})."
            )

        cost_per_volume_nan = np.argwhere(np.isnan(self.cost_per_volume)).ravel()

        if len(cost_per_volume_nan) > 0:
            self.cost_per_volume[cost_per_volume_nan] = np.zeros(len(cost_per_volume_nan))

        else:
            self.cost_per_volume = self.cost_per_volume

        # Prepare attribute vat_portion
        if not isinstance(self.vat_portion, np.ndarray):
            raise OPEXDataException(
                f"VAT portion data must be given in the form of numpy.ndarray. "
                f"The current vat_portion data is given in the form of: "
                f"({self.vat_portion.__class__.__qualname__})."
            )

        vat_portion_nan = np.argwhere(np.isnan(self.vat_portion)).ravel()

        if len(vat_portion_nan) > 0:
            self.vat_portion[vat_portion_nan] = np.ones(len(vat_portion_nan))

        else:
            self.vat_portion = self.vat_portion

        # Prepare attribute lbt_portion
        if not isinstance(self.lbt_portion, np.ndarray):
            raise OPEXDataException(
                f"LBT portion data must be given in the form of numpy.ndarray. "
                f"The current lbt_portion data is given in the form of: "
                f"({self.lbt_portion.__class__.__qualname__})."
            )

        lbt_portion_nan = np.argwhere(np.isnan(self.lbt_portion)).ravel()

        if len(lbt_portion_nan) > 0:
            self.lbt_portion[lbt_portion_nan] = np.ones(len(lbt_portion_nan))

        else:
            self.lbt_portion = self.lbt_portion

        # Prepare attribute description
        if not isinstance(self.description, list):
            raise OPEXDataException(
                f"Description data must be given in the form of list. "
                f"The current description data is given in the form of: "
                f"({self.description.__class__.__qualname__})."
            )

        description_nan = np.argwhere(pd.isna(self.description)).ravel()

        if len(description_nan) > 0:
            for i in description_nan:
                self.description[i] = " "

        else:
            self.description = self.description


@dataclass
class ASRCostData:
    """
    A dataclass representing ASR cost information for an oil and gas economic project.

    Attributes
    ----------
    expense_year: np.ndarray
        Array of expense years.
    cost_allocation: list
        List of cost allocations.
    cost: np.ndarray
        Array of costs.
    vat_portion: np.ndarray
        Array of VAT (Value Added Tax) portions.
    lbt_portion: np.ndarray
        Array of LBT (Land and Building Tax) portions.
    description: list
        List of descriptions.
    data_length: int
        Length of the captured data.
    project_years: np.ndarray
        Array of project years.
    """
    expense_year: np.ndarray
    cost_allocation: list
    cost: np.ndarray
    vat_portion: np.ndarray
    lbt_portion: np.ndarray
    description: list

    # Attribute associated with length of the captured data
    data_length: int

    # Attribute associated with project duration
    project_years: np.ndarray

    def __post_init__(self):
        # Prepare attribute expense_year
        if not isinstance(self.expense_year, np.ndarray):
            raise ASRCostDataException(
                f"Expense year data must be given in the form of numpy.ndarray. "
                f"The current expense_year data is given in the form of: "
                f"({self.expense_year.__class__.__qualname__})."
            )

        expense_year_nan = np.argwhere(np.isnan(self.expense_year)).ravel()

        if len(expense_year_nan) > 0:
            raise ASRCostDataException(
                f"Expense year data is incomplete. Please re-check the expense_year data. "
                f"The number of expense_year data must be ({self.data_length}), "
                f"not ({self.data_length - len(expense_year_nan)})."
            )

        else:
            self.expense_year = self.expense_year

        # Prepare attribute cost_allocation
        if not isinstance(self.cost_allocation, list):
            raise ASRCostDataException(
                f"Cost allocation data must be given in the form of list. "
                f"The current cost_allocation data is given in the form of: "
                f"({self.cost_allocation.__class__.__qualname__})."
            )

        cost_allocation_nan = np.argwhere(pd.isna(self.cost_allocation)).ravel()

        if len(cost_allocation_nan) > 0:
            raise ASRCostDataException(
                f"Cost allocation data is incomplete. "
                f"Please re-check cost_allocation data. "
                f"The number of cost_allocation data must be ({self.data_length}), "
                f"not ({self.data_length - len(cost_allocation_nan)})."
            )

        else:
            self.cost_allocation = self.cost_allocation

        # Prepare attribute cost
        if not isinstance(self.cost, np.ndarray):
            raise ASRCostDataException(
                f"Cost data must be given in the form of numpy.ndarray. "
                f"The current cost data is given in the form of: "
                f"({self.cost.__class__.__qualname__})."
            )

        cost_nan = np.argwhere(np.isnan(self.cost)).ravel()

        if len(cost_nan) > 0:
            self.cost[cost_nan] = np.zeros(len(cost_nan))

        else:
            self.cost = self.cost

        # Prepare attribute vat_portion
        if not isinstance(self.vat_portion, np.ndarray):
            raise ASRCostDataException(
                f"VAT portion data must be given in the form of numpy.ndarray. "
                f"The current vat_portion data is given in the form of: "
                f"({self.vat_portion.__class__.__qualname__})."
            )

        vat_portion_nan = np.argwhere(np.isnan(self.vat_portion)).ravel()

        if len(vat_portion_nan) > 0:
            self.vat_portion[vat_portion_nan] = np.ones(len(vat_portion_nan))

        else:
            self.vat_portion = self.vat_portion

        # Prepare attribute lbt_portion
        if not isinstance(self.lbt_portion, np.ndarray):
            raise ASRCostDataException(
                f"LBT portion data must be given in the form of numpy.ndarray. "
                f"The current lbt_portion data is given in the form of: "
                f"({self.lbt_portion.__class__.__qualname__})."
            )

        lbt_portion_nan = np.argwhere(np.isnan(self.lbt_portion)).ravel()

        if len(lbt_portion_nan) > 0:
            self.lbt_portion[lbt_portion_nan] = np.ones(len(lbt_portion_nan))

        else:
            self.lbt_portion = self.lbt_portion

        # Prepare attribute description
        if not isinstance(self.description, list):
            raise ASRCostDataException(
                f"Description data must be given in the form of list. "
                f"The current description data is given in the form of: "
                f"({self.description.__class__.__qualname__})."
            )

        description_nan = np.argwhere(pd.isna(self.description)).ravel()

        if len(description_nan) > 0:
            for i in description_nan:
                self.description[i] = " "

        else:
            self.description = self.description


@dataclass
class PSCCostRecoveryData:
    """
    A dataclass representing attributes associated with Production Sharing Contract
    (PSC) cost recovery.

    Attributes
    ----------
    ftp_availability: str
        Availability status of the First Tranche Petroleum (FTP).
    ftp_is_shared: str
        Shared status of FTP.
    ftp_portion: float
        Portion of FTP.
    split_type: str
        Type of pretax split configuration.
    oil_ctr_pretax: float
        Pretax split configuration for oil.
    gas_ctr_pretax: float
        Pretax split configuration for gas.
    ic_availability: str
        Availability status of the Investment Credit (IC).
    ic_oil: float
        Investment Credit (IC) for oil.
    ic_gas: float
        Investment Credit (IC) for gas.
    oil_cr_cap_rate: float
        Cost Recovery (CR) cap rate for oil.
    gas_cr_cap_rate: float
        Cost Recovery (CR) cap rate for gas.
    dmo_is_weighted: str
        Weighted status of the general Domestic Market Obligation (DMO).
    oil_dmo_holiday: str
        Holiday status for DMO related to oil.
    oil_dmo_period: int | float
        Period for DMO related to oil.
    oil_dmo_start_production: date
        Start production date for DMO related to oil.
    oil_dmo_volume: float
        Volume for DMO related to oil.
    oil_dmo_fee: float
        Fee for DMO related to oil.
    gas_dmo_holiday: str
        Holiday status for DMO related to gas.
    gas_dmo_period: int | float
        Period for DMO related to gas.
    gas_dmo_start_production: date
        Start production date for DMO related to gas.
    gas_dmo_volume: float
        Volume for DMO related to gas.
    gas_dmo_fee: float
        Fee for DMO related to gas.
    rc_split_data: np.ndarray
        Array of rc split data.
    icp_sliding_scale_data : np.ndarray
        Array of icp sliding scale data.

    Notes
    -----
    -   This class is designed to store data related to a Production Sharing Contract
        (PSC) cost recovery scenario.
    -   Some attributes, such as rc_split and icp_sliding_scale, are meant to be
        defined later during class execution.
    """
    # Attributes associated with FTP
    ftp_availability: str
    ftp_is_shared: str
    ftp_portion: float

    # Attributes associated with pretax split configuration
    split_type: str
    oil_ctr_pretax: float
    gas_ctr_pretax: float

    # Attributes associated with investment credit
    ic_availability: str
    ic_oil: float
    ic_gas: float

    # Attributes associated with cap
    oil_cr_cap_rate: float
    gas_cr_cap_rate: float

    # Attribute associated with general DMO
    dmo_is_weighted: str

    # Attributes associated with DMO oil
    oil_dmo_holiday: str
    oil_dmo_period: int | float
    oil_dmo_start_production: date
    oil_dmo_volume: float
    oil_dmo_fee: float

    # Attributes associated with DMO gas
    gas_dmo_holiday: str
    gas_dmo_period: int | float
    gas_dmo_start_production: date
    gas_dmo_volume: float
    gas_dmo_fee: float

    # Attributes associated with rc_split and icp_sliding_scale
    rc_split_data: np.ndarray = field(repr=False)
    icp_sliding_scale_data: np.ndarray = field(repr=False)

    # Attributes to be defined later on
    rc_split: dict = field(default=None, init=False)
    icp_sliding_scale: dict = field(default=None, init=False)

    def __post_init__(self):
        # Prepare attribute rc_split
        if self.split_type == "RC Split":
            rc_split_attrs = [
                "RC Bottom Limit",
                "RC Top Limit",
                "Pre Tax CTR Oil",
                "Pre Tax CTR Gas",
            ]
            rc_split = {key: self.rc_split_data[:, i] for i, key in enumerate(rc_split_attrs)}
            self.rc_split = {
                key: np.array(list(filter(lambda i: i is not None, rc_split[key])))
                for key in rc_split_attrs
            }

        # Prepare attribute icp_sliding_scale
        if self.split_type == "ICP Sliding Scale":
            icp_attrs = [
                "ICP Bottom Limit",
                "ICP Top Limit",
                "Pre Tax CTR Oil",
                "Pre Tax CTR Gas",
            ]
            self.icp_sliding_scale = {
                key: self.icp_sliding_scale_data[:, i] for i, key in enumerate(icp_attrs)
            }


@dataclass
class PSCGrossSplitData:
    """
    A dataclass representing attributes associated with Production Sharing Contract
    (PSC) gross split.

    Attributes
    ----------
    field_status: str
        The status of the field.
    field_location: str
        The location of the field.
    reservoir_depth: str
        The depth of the reservoir.
    infrastructure_availability: str
        The availability of infrastructure.
    reservoir_type: str
        The type of reservoir.
    co2_content: str
        The CO2 content in the production.
    h2s_content: str
        The H2S content in the production.
    oil_api: str
        The API gravity of the oil.
    domestic_content_use: str
        The use of domestic content.
    production_stage: str
        The stage of production.
    ministry_discretion_split: float
        The split determined by ministry discretion.
    dmo_is_weighted: str
        Weighting information for general DMO.
    oil_dmo_holiday: str
        Holiday information for DMO oil.
    oil_dmo_period: int | float
        The period of DMO oil production.
    oil_dmo_start_production: date
        The start date of DMO oil production.
    oil_dmo_volume: float
        The volume of DMO oil production.
    oil_dmo_fee: float
        The fee associated with DMO oil production.
    gas_dmo_holiday: str
        Holiday information for DMO gas.
    gas_dmo_period: int | float
        The period of DMO gas production.
    gas_dmo_start_production: date
        The start date of DMO gas production.
    gas_dmo_volume: float
        The volume of DMO gas production.
    gas_dmo_fee: float
        The fee associated with DMO gas production.

    Notes
    -----
    This class is designed to store data related to a Production Sharing Contract
    (PSC) gross split scenario.
    """
    # Attributes associated with split configuration
    field_status: str
    field_location: str
    reservoir_depth: str
    infrastructure_availability: str
    reservoir_type: str
    co2_content: str
    h2s_content: str
    oil_api: str
    domestic_content_use: str
    production_stage: str
    ministry_discretion_split: float

    # Attribute associated with general DMO
    dmo_is_weighted: str

    # Attributes associated with DMO oil
    oil_dmo_holiday: str
    oil_dmo_period: int | float
    oil_dmo_start_production: date
    oil_dmo_volume: float
    oil_dmo_fee: float

    # Attributes associated with DMO gas
    gas_dmo_holiday: str
    gas_dmo_period: int | float
    gas_dmo_start_production: date
    gas_dmo_volume: float
    gas_dmo_fee: float


@dataclass
class SensitivityData:
    """
    A dataclass representing attributes associated with sensitivity analysis.

    Attributes
    ----------
    parameter: list
        A list of parameters or values for which sensitivity analysis is conducted.
    percentage_min: float
        The minimum percentage value for sensitivity analysis.
    percentage_max: float
        The maximum percentage value for sensitivity analysis.
    step: int
        The step size for the sensitivity analysis.
    """
    parameter: list
    percentage_min: float
    percentage_max: float
    step: int


@dataclass
class MonteCarloData:
    """
    A dataclass representing attributes associated with montecarlo analysis.

    Attributes
    ----------
    parameter: list
        A list of parameters or variables for the simulation.
    distribution: list
        A list of probability distributions corresponding to each parameter.
    min_values: np.ndarray
        An array of minimum values for each parameter.
    max_values: np.ndarray
        An array of maximum values for each parameter.
    std_dev: np.ndarray
        An array of standard deviations for each parameter.
    """
    parameter: list
    distribution: list
    min_values: np.ndarray
    max_values: np.ndarray
    std_dev: np.ndarray

    # Attribute associated with length of the captured data
    data_length: int

    def __post_init__(self):
        # Prepare attribute min_values
        if not isinstance(self.min_values, np.ndarray):
            raise MonteCarloDataException(
                f"Minimum values data must be given in the form of numpy.ndarray. "
                f"The current min_values data is given in the form of: "
                f"({self.min_values.__class__.__qualname__})."
            )

        min_values_nan = list(filter(lambda i: i is np.nan, self.min_values))

        if len(min_values_nan) > 0:
            raise MonteCarloDataException(
                f"Minimum values data is incomplete. Please re-check the min_values data. "
                f"The number of min_values data must be ({self.data_length}), "
                f"not ({self.data_length - len(min_values_nan)})."
            )

        else:
            self.min_values = self.min_values

        # Prepare attribute max_values
        if not isinstance(self.max_values, np.ndarray):
            raise MonteCarloDataException(
                f"Maximum values data must be given in the form of numpy.ndarray. "
                f"The current max_values data is given in the form of: "
                f"({self.max_values.__class__.__qualname__})."
            )

        max_values_nan = list(filter(lambda i: i is np.nan, self.max_values))

        if len(max_values_nan) > 0:
            raise MonteCarloDataException(
                f"Maximum values data is incomplete. Please re-check the max_values data. "
                f"The number of max_values data must be ({self.data_length}), "
                f"not ({self.data_length - len(max_values_nan)})."
            )

        else:
            self.max_values = self.max_values

        # Prepare attribute std_dev
        if not isinstance(self.std_dev, np.ndarray):
            raise MonteCarloDataException(
                f"Standard deviation data must be given in the form of numpy.ndarray. "
                f"The current std_dev data is given in the form of: "
                f"({self.std_dev.__class__.__qualname__})."
            )

        std_dev_nan = list(filter(lambda i: i is np.nan, self.std_dev))

        if len(std_dev_nan) > 0:
            raise MonteCarloDataException(
                f"Standard deviation data is incomplete. Please re-check the std_dev data. "
                f"The number of std_dev data must be ({self.data_length}), "
                f"not ({self.data_length - len(std_dev_nan)})."
            )

        else:
            self.std_dev = self.std_dev

        # Conditions to raise exception
        for i, val in enumerate(self.min_values):
            if self.min_values[i] == self.max_values[i]:
                raise MonteCarloDataException(
                    f"Maximum and minimum value(s) are the same."
                )

            if self.min_values[i] > self.max_values[i]:
                raise MonteCarloDataException(
                    f"Maximum values of parameter(s) are less "
                    f"than their corresponding minimum values."
                )


@dataclass
class OptimizationData:
    """
    A dataclass representing attributes associated with optimization study.

    Attributes
    ----------
    target: dict
        The target data.
    data_cr_init: dict
        The initial cost recovery data.
    data_gs_init: dict
        The initial gross split data.
    """
    target: dict
    data_cr_init: dict = field(repr=False)
    data_gs_init: dict = field(repr=False)

    # Attributes to be defined later
    data_cr: dict = field(default=None, init=False)
    data_gs: dict = field(default=None, init=False)

    def __post_init__(self):
        # Prepare attribute data_cr
        # Step #1: raise exception for inappropriate data input
        check_data_cr = [i <= j for i, j in zip(self.data_cr_init["max"], self.data_cr_init["min"])]

        if True in check_data_cr:
            raise OptimizationDataException(
                f"Error in cost recovery data input. "
                f"The maximum value(s) must be larger than the minimum value(s). "
                f"Max: ({self.data_cr_init['max']}), "
                f"Min: ({self.data_cr_init['min']})."
            )

        # Step #2: filter out 'None' values
        id_cr_unsorted = np.array(
            [i for i, val in enumerate(self.data_cr_init["priority"]) if val is not None]
        )
        data_cr_unsorted = {key: self.data_cr_init[key][id_cr_unsorted] for key in self.data_cr_init}

        # Step #3: sorted the data based on the values of "priority"
        id_cr_sorted = np.argsort(data_cr_unsorted["priority"])
        self.data_cr = {key: data_cr_unsorted[key][id_cr_sorted] for key in data_cr_unsorted}

        # Prepare attribute data_gs
        # Step #1: raise exception for inappropriate data input
        check_data_gs = [i <= j for i, j in zip(self.data_gs_init["max"], self.data_gs_init["min"])]

        if True in check_data_gs:
            raise OptimizationDataException(
                f"Error in cost recovery data input. "
                f"The maximum value(s) must be larger than the minimum value(s). "
                f"Max: ({self.data_gs_init['max']}), "
                f"Min: ({self.data_gs_init['min']})."
            )

        # Step #2: filter out 'None' values
        id_gs_unsorted = np.array(
            [i for i, val in enumerate(self.data_gs_init["priority"]) if val is not None]
        )
        data_gs_unsorted = {key: self.data_gs_init[key][id_gs_unsorted] for key in self.data_gs_init}

        # Step #3: sorted the data based on the values of "priority"
        id_gs_sorted = np.argsort(data_gs_unsorted["priority"])
        self.data_gs = {key: data_gs_unsorted[key][id_gs_sorted] for key in data_gs_unsorted}
