"""
Prepare data input loaded from a target Excel file.
"""

from dataclasses import dataclass, field
from datetime import date
import numpy as np


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
    """

    tax_mode: str
    tax_payment_method: str
    tax_psc_cost_recovery: str
    npv_mode: str
    future_rate_asr: float
    depreciation_method: str

    # Attributes to be defined later
    tax_rate: float = field(default=None)
    year_arr: np.ndarray = field(default=None, repr=False)
    tax_rate_arr: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        # Configure attribute tax_rate
        if self.tax_mode == "User Input - Single Value":
            if self.tax_rate is None:
                self.tax_rate = 0.44
            else:
                self.tax_rate = self.tax_rate

        if self.tax_mode == "Nailed Down" or self.tax_mode == "Prevailing":
            if self.tax_rate is None:
                self.tax_rate = 0.44
            else:
                self.tax_rate = self.tax_rate

        if self.tax_mode == "User Input - Multi Value":
            if not isinstance(self.year_arr, np.ndarray):
                raise FiscalConfigDataException(
                    f"Year data must be inserted as a numpy ndarray. "
                    f"{self.year_arr} is of datatype "
                    f"({self.year_arr.__class__.__qualname__})."
                )

            if not isinstance(self.tax_rate_arr, np.ndarray):
                raise FiscalConfigDataException(
                    f"Tax rate data must be inserted as a numpy ndarray. "
                    f"{self.tax_rate_arr} is of datatype "
                    f"({self.tax_rate_arr.__class__.__qualname__})."
                )

            if len(self.year_arr) != len(self.tax_rate_arr):
                raise FiscalConfigDataException(
                    f"Unequal length of array: "
                    f"year_arr: {len(self.year_arr)}, "
                    f"tax_rate_arr: {len(self.tax_rate_arr)}"
                )

            self.tax_rate = {
                "Year": self.year_arr,
                "Tax Rate": self.tax_rate_arr,
            }


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
    project_years: np.ndarray

    def __post_init__(self):
        # Prepare attribute expense_year
        if not isinstance(self.expense_year, np.ndarray):
            raise TangibleCostDataException(
                f"Incompatible datatype: "
                f"expense_year: ({self.expense_year.__class__.__qualname__}). "
                f"Expense year must be provided as a numpy array."
            )

        expense_year_none = list(filter(lambda i: i is None, self.expense_year))

        if None in expense_year_none:
            if len(expense_year_none) == self.data_length:
                self.expense_year = np.repeat(self.project_years[0], self.data_length)

            else:
                raise TangibleCostDataException(
                    f"Expense year data is incomplete. "
                    f"The number of expense_year data should be ({self.data_length}), "
                    f"not ({self.data_length - len(expense_year_none)})."
                )

        # Prepare attribute cost_allocation
        if not isinstance(self.cost_allocation, list):
            raise TangibleCostDataException(
                f"Incompatible datatype: "
                f"cost_allocation: ({self.cost_allocation.__class__.__qualname__}). "
                f"Cost allocation must be provided as a numpy array."
            )

        cost_allocation_none = list(filter(lambda i: i is None, self.cost_allocation))

        if None in cost_allocation_none:
            if len(cost_allocation_none) == self.data_length:
                self.cost_allocation = None
            else:
                raise TangibleCostDataException(
                    f"Cost allocation data is incomplete. "
                    f"The number of cost_allocation data should be ({self.data_length}), "
                    f"not ({self.data_length - len(cost_allocation_none)})."
                )

        # Prepare attribute cost
        if not isinstance(self.cost, np.ndarray):
            raise TangibleCostDataException(
                f"Incompatible datatype: "
                f"cost: ({self.cost.__class__.__qualname__}). "
                f"Cost must be provided as a numpy array."
            )

        cost_none = list(filter(lambda i: i is None, self.cost))

        if None in cost_none:
            if len(cost_none) == self.data_length:
                self.cost = np.zeros(len(self.data_length))
            else:
                raise TangibleCostDataException(

                )

        print('\t')
        print(f'Filetype: {type(cost_none)}')
        print('cost_none = \n', cost_none)

        # if len(expense_year_not_none) == 0:
        #     self.expense_year = np.repeat(self.project_years[0], self.project_duration)
        #
        # if 0 < len(expense_year_not_none) < self.project_duration:
        #     raise TangibleCostDataException(
        #         f"Expense year data is incomplete. "
        #         f"The number of expense_year data should be ({self.project_duration}), "
        #         f"not ({len(expense_year_not_none)})."
        #     )

    #     if len(expense_year_not_none) > self.project_duration:
    #         raise TangibleCostDataException(
    #             f"The length of expense year data ({len(expense_year_not_none)}) "
    #             f"is longer than the project duration ({self.project_duration})."
    #         )
    #
    #     # Prepare attribute cost_allocation
    #     if not isinstance(self.cost_allocation, list):
    #         raise TangibleCostDataException(
    #             f"Incompatible datatype: "
    #             f"cost_allocation: ({self.cost_allocation.__class__.__qualname__}). "
    #             f"Cost allocation must be provided as a list."
    #         )
    #
    #     if None in self.cost_allocation:
    #         cost_allocation_none = list(filter(lambda i: i is None, self.cost_allocation))
    #         if len(cost_allocation_none) == :
    #             self.cost_allocation = None
    #         else:
    #             raise TangibleCostDataException(
    #                 f"Cost allocation data is incomplete. "
    #                 f"The number of cost_allocation data should be ({self.project_duration}), "
    #                 f"not ({self.project_duration - len(cost_allocation_none)})."
    #             )
    #
    #     elif None not in self.cost_allocation:
    #         if len(self.cost_allocation) > self.project_duration:
    #             raise TangibleCostDataException
    #
    #     print('\t')
    #     print(f'Filetype: {type(self.cost_allocation)}')
    #     print('cost_allocation = ', self.cost_allocation)
