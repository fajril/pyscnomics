"""
Specify constants for several processes using Enum class.
"""

from enum import Enum


class DeprMethod(Enum):

    """
    Enumeration of depreciation methods.

    Attributes
    ----------
    SL : str
        Represents the straight-line depreciation method
    DB : str
        Represents the declining balance depreciation method
    PSC_DB : str
        Represents the PSC double declining balance method
    UOP : str
        Represents the units of production depreciation method
    """

    SL = "sl"
    DB = "db"
    PSC_DB = "psc db"
    UOP = "uop"


class FluidType(Enum):

    """
    Enumeration of fluid types for depreciation calculation.

    Attributes
    ----------
    ALL : str
        Represents all fluid types
    OIL : str
        Represents oil as the fluid type
    GAS : str
        Represents gas as the fluid type
    SULFUR: str
        Represents sulfur as the fluid type
    ELECTRICITY: str
        Represent electricity as the fluid type
    CO2: str
        Represents CO2 as the fluid type
    """

    ALL = "all"
    OIL = "oil"
    GAS = "gas"
    SULFUR = "sulfur"
    ELECTRICITY = "electricity"
    CO2 = "co2"


class TaxType(Enum):
    """
    Enumeration of tax types for incorporation of tax adjustment.

    Attributes
    ----------
    VAT : str
        Represents Value Added Tax.
    LBT : str
        Represents Land and Building Tax.
    """

    VAT = "vat"
    LBT = "lbt"


class TaxSplitTypeCR(Enum):
    """
    Enumeration of split type for cost recovery.

    Attributes
    ----------
    CONVENTIONAL : str
        Represents conventional split type
    SLIDING_SCALE : str
        Represents sliding scale split type
    R2C : str
        Represents revenue to cost split type
    """

    CONVENTIONAL = "conventional"
    SLIDING_SCALE = "ICP sliding scale"
    R2C = "R/C"


class YearReference(Enum):
    """
    Enumeration of different year references used for depreciation calculations.

    Attributes:
    -----------
    PIS_YEAR : str
        Symbolic name for the Placed in Service (PIS) year reference

    EXPENSE_YEAR : str
        Symbolic name for the expense year reference
    """

    PIS_YEAR = "pis_year"
    EXPENSE_YEAR = "expense_year"


class TaxRegime(Enum):
    """
    Enumeration class representing different tax regimes for oil and gas economic analysis.

    Attributes:
    ----------
    UU_36_2008 (str): Tax regime under "UU No.36 Tahun 2008"
    UU_02_2020 (str): Tax regime under "UU No.02 Tahun 2020"
    UU_07_2021 (str): Tax regime under "UU No.07 Tahun 2021"
    NAILED_DOWN (str): Special tax regime designated as "nailed down"
    PREVAILING (str): General tax regime referred to as "prevailing"
    """

    NAILED_DOWN = "nailed down"
    PREVAILING = "prevailing"
    UU_36_2008 = "UU No.36 Tahun 2008"
    UU_02_2020 = "UU No.02 Tahun 2020"
    UU_07_2021 = "UU No.07 Tahun 2021"


class TaxPaymentMode(Enum):
    TAX_DUE_MODE = "Calculation method where it treated like the unrecoverable cost, could be weighted on the next year"
    TAX_DIRECT_MODE = "Calculation method where the tax calculated directly from taxable income"


class FTPTaxRegime(Enum):
    """
    An enumeration depicting the FTP Tax Regime associated with oil and gas
    Production Sharing Contract in Indonesia.

    Attributes
    ----------
    PDJP_20_2017: str
        Manifests Peraturan DIRJEN PAJAK No. PER-20/PJ/2017
    PRE_PDJP_20_2017: str
        Manifests Pre Peraturan DIRJEN PAJAK No. PER-20/PJ/2017
    """
    PDJP_20_2017 = "Peraturan DIRJEN PAJAK No. PER-20/PJ/2017"
    PRE_PDJP_20_2017 = "Pre Peraturan DIRJEN PAJAK No. PER-20/PJ/2017"
    DIRECT_MODE = "The tax will be paid directly"


class GrossSplitRegime(Enum):
    """
    An enumeration representing different Gross Split Regimes in the context of
    oil and gas economic analysis, specifically related to Production Sharing Contracts
    in Indonesia.

    Attributes
    ----------
    PERMEN_ESDM_8_2017: str
        Represents the Gross Split Regime according to Peraturan Menteri ESDM No. 8 Tahun 2017.
    PERMEN_ESDM_52_2017: str
        Represents the Gross Split Regime according to Peraturan Menteri ESDM No. 52 Tahun 2017.
    PERMEN_ESDM_20_2019: str
        Represents the Gross Split Regime according to Peraturan Menteri ESDM No. 20 Tahun 2019.
    PERMEN_ESDM_12_2020: str
        Represents the Gross Split Regime according to Peraturan Menteri ESDM No. 12 Tahun 2020.
    """
    PERMEN_ESDM_8_2017 = 'Peraturan Menteri ESDM No. 8 Tahun 2017'
    PERMEN_ESDM_52_2017 = 'Peraturan Menteri ESDM No. 52 Tahun 2017'
    PERMEN_ESDM_20_2019 = 'Peraturan Menteri ESDM No. 20 Tahun 2019'
    PERMEN_ESDM_12_2020 = 'Peraturan Menteri ESDM No. 12 Tahun 2020'


class DiscountingMode(Enum):
    END_YEAR = "End Year Discounting"
    MID_YEAR = "Mid Year Discounting"


class ContractType(Enum):
    COST_RECOVERY = 'Cost Recovery (CR)'
    GROSS_SPLIT = 'Gross Split (GS)'
    TRANSITION_CR_CR = 'Transition CR - CR'
    TRANSITION_CR_GS = 'Transition CR - GS'
    TRANSITION_GS_GS = 'Transition GS - GS'
    TRANSITION_GS_CR = 'Transition GS - CR'


class NPVSelection(Enum):
    NPV_REAL_TERMS = 'NPV Calculation using real terms method'
    NPV_NOMINAL_TERMS = 'NPV Calculation using nominal terms method'
    NPV_SKK_NOMINAL_TERMS = 'NPV Calculation using SKK Nominal terms method'
    NPV_SKK_REAL_TERMS = 'NPV Calculation using SKK Real terms method'
    NPV_POINT_FORWARD = 'NPV Calculation using Point Forward method'


class OtherRevenue(Enum):
    ADDITION_TO_OIL_REVENUE = "The revenue will be treated as the Oil Revenue Addition"
    ADDITION_TO_GAS_REVENUE = "The revenue will be treated as the Gas Revenue Addition"
    REDUCTION_TO_OIL_OPEX = "The revenue will be treated as the Oil OPEX Reduction"
    REDUCTION_TO_GAS_OPEX = "The revenue will be treated as the GAS OPEX Reduction"


class InflationAppliedTo(Enum):
    CAPEX = "Inlflation rate will be applied to Tangible and Intangible Cost"
    OPEX = "Inflation rate will be applied to Operating Cost"
    CAPEX_AND_OPEX = "Inflation rate will be applied to Tangible, Intangible, Operating Cost"

