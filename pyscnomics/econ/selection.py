"""
Specify constants for several processes using Enum class.
"""

from enum import Enum


class ExpendituresType(Enum):
    """
    Enumeration of expenditures type.

    Attributes
    ----------
    PRE_TAX: str
        Represents pre-tax expenditures.
    POST_TAX: str
        Represents post-tax expenditures.
    """

    PRE_TAX = "Pre-Tax Expenditures"
    POST_TAX = "Post-Tax Expenditures"


class DeprMethod(Enum):
    """
    Enumeration of depreciation methods.

    Attributes
    ----------
    SL: str
        Represents the straight-line depreciation method
    DB: str
        Represents the declining balance depreciation method
    PSC_DB: str
        Represents the PSC declining balance method
    """

    SL = "sl"
    DB = "db"
    PSC_DB = "psc db"


class FluidType(Enum):
    """
    Enumeration of fluid types.

    Attributes
    ----------
    ALL: str
        Represents all fluid types
    OIL: str
        Represents oil as the fluid type
    GAS: str
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
    CONVENTIONAL: str
        Represents conventional split type
    SLIDING_SCALE: str
        Represents sliding scale split type
    R2C: str
        Represents revenue to cost split type
    """

    CONVENTIONAL = "conventional"
    SLIDING_SCALE = "ICP sliding scale"
    R2C = "R/C"


class YearReference(Enum):
    """
    Enumeration of different year references used for depreciation calculations.

    Attributes
    ----------
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

    Attributes
    ----------
    NAILED_DOWN: str
        Special tax regime designated as "nailed down"
    PREVAILING: str
        General tax regime referred to as "prevailing"
    UU_36_2008: str
        Tax regime under "UU No.36 Tahun 2008"
    UU_02_2020: str
        Tax regime under "UU No.02 Tahun 2020"
    UU_07_2021: str
        Tax regime under "UU No.07 Tahun 2021"
    """

    NAILED_DOWN = "nailed down"
    PREVAILING = "prevailing"
    UU_36_2008 = "UU No. 36 Tahun 2008"
    UU_02_2020 = "UU No. 02 Tahun 2020"
    UU_07_2021 = "UU No. 07 Tahun 2021"


class TaxPaymentMode(Enum):
    """
    Enumeration class representing different tax payment modes.

    Attributes
    ----------
    TAX_DUE_MODE: str
        Calculation method where it is treated like an unrecoverable cost,
        and could be weighted on the next year.
    TAX_DIRECT_MODE: str
        Calculation method where the tax is calculated directly from taxable income.
    """

    TAX_DUE_MODE = (
        "Calculation method where it treated like the unrecoverable cost, "
        "could be weighted on the next year"
    )
    TAX_DIRECT_MODE = (
        "Calculation method where the tax calculated directly from taxable income"
    )


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

    PERMEN_ESDM_8_2017 = None
    PERMEN_ESDM_52_2017 = None
    PERMEN_ESDM_20_2019 = None
    PERMEN_ESDM_12_2020 = None


class DiscountingMode(Enum):
    """
    Enumeration class representing different discounting modes for financial calculations.

    Attributes
    ----------
    END_YEAR: str
        Discounting method where the cash flows are discounted to the end of each year.
    MID_YEAR: str
        Discounting method where the cash flows are discounted to the middle of each year.
    """

    END_YEAR = "End Year Discounting"
    MID_YEAR = "Mid Year Discounting"


class ContractType(Enum):
    """
    Enumeration class representing different types of production sharing contracts
    (PSCs) in Indonesia.

    Attributes
    ----------
    COST_RECOVERY: str
        Cost Recovery (CR) contract type.
    GROSS_SPLIT: str
        Gross Split (GS) contract type.
    """

    COST_RECOVERY = "Cost Recovery (CR)"
    GROSS_SPLIT = "Gross Split (GS)"


class NPVSelection(Enum):
    """
    Enumeration class representing different methods for calculating
    Net Present Value (NPV) in economic analysis.

    Attributes
    ----------
    NPV_REAL_TERMS: str
        NPV Calculation using the real terms method.
    NPV_NOMINAL_TERMS: str
        NPV Calculation using the nominal terms method.
    NPV_SKK_NOMINAL_TERMS: str
        NPV Calculation using the SKK nominal terms method.
    NPV_SKK_REAL_TERMS: str
        NPV Calculation using the SKK real terms method.
    NPV_POINT_FORWARD: str
        NPV Calculation using the Point Forward method.
    """
    NPV_REAL_TERMS = "NPV Calculation using real terms method"
    NPV_NOMINAL_TERMS = "NPV Calculation using nominal terms method"
    NPV_SKK_NOMINAL_TERMS = "NPV Calculation using SKK Nominal terms method"
    NPV_SKK_REAL_TERMS = "NPV Calculation using SKK Real terms method"
    NPV_POINT_FORWARD = "NPV Calculation using Point Forward method"


class OtherRevenue(Enum):
    """
    Enumeration class representing different types of other revenue
    in oil and gas economic analysis.

    Attributes
    ----------
    ADDITION_TO_OIL_REVENUE: str
        Revenue treated as addition to oil revenue.
    ADDITION_TO_GAS_REVENUE: str
        Revenue treated as addition to gas revenue.
    REDUCTION_TO_OIL_OPEX: str
        Revenue treated as reduction to oil operational expenditure (OPEX).
    REDUCTION_TO_GAS_OPEX: str
        Revenue treated as reduction to gas operational expenditure (OPEX).
    """
    ADDITION_TO_OIL_REVENUE = "The revenue will be treated as the Oil Revenue Addition"
    ADDITION_TO_GAS_REVENUE = "The revenue will be treated as the Gas Revenue Addition"
    REDUCTION_TO_OIL_OPEX = "The revenue will be treated as the Oil OPEX Reduction"
    REDUCTION_TO_GAS_OPEX = "The revenue will be treated as the GAS OPEX Reduction"


class InflationAppliedTo(Enum):
    """
    Enumeration class representing different aspects to which inflation rate
    is applied in economic analysis.

    Attributes
    ----------
    CAPEX: str
        Inflation rate applied to Tangible and Intangible Costs (CAPEX).
    OPEX: str
        Inflation rate applied to Operating Costs (OPEX).
    CAPEX_AND_OPEX: str
        Inflation rate applied to Tangible, Intangible, and Operating Costs (CAPEX and OPEX).
    """
    CAPEX = "Inflation rate will be applied to Tangible and Intangible Cost"
    OPEX = "Inflation rate will be applied to Operating Cost"
    CAPEX_AND_OPEX = "Inflation rate will be applied to Tangible, Intangible, Operating Cost"


class VariableSplit522017:
    """
    Variable split corresponds to PERMEN No. 52 Year 2017.
    """
    class FieldStatus(Enum):
        """
        Field status variable split component
        """
        POD_I = "Plan of Development I"
        POD_II = "Plan of Development II"
        NO_POD = "No Plan of Development"

    class FieldLocation(Enum):
        """
        Field location variable split component
        """
        ONSHORE = "On Shore"
        OFFSHORE_0_UNTIL_LESSEQUAL_20 = "Offshore 0 < h <= 20"
        OFFSHORE_20_UNTIL_LESSEQUAL_50 = "Offshore 20 < h <= 50"
        OFFSHORE_50_UNTIL_LESSEQUAL_150 = "Offshore 50 < h <= 150"
        OFFSHORE_150_UNTIL_LESSEQUAL_1000 = "Offshore 150 < h <= 1000"
        OFFSHORE_GREATERTHAN_1000 = "Offshore h > 1000"

    class ReservoirDepth(Enum):
        """
        Reservoir depth variable split component
        """
        LESSEQUAL_2500 = "Reservoir Depth <=2500"
        GREATERTHAN_2500 = "Reservoir Depth >2500"

    class InfrastructureAvailability(Enum):
        """
        Infrastucture availability variable split component
        """
        WELL_DEVELOP = "Well Developed"
        NEW_FRONTIER_OFFSHORE = "New Frontier Offshore"
        NEW_FRONTIER_ONSHORE = "New Frontier Onshore"

    class ReservoirType(Enum):
        """
        Reservoir type variable split component
        """
        CONVENTIONAL = "Conventional"
        NON_CONVENTIONAL = "Non Conventional"

    class CO2Content(Enum):
        """
        CO2 content variable split component
        """
        LESSTHAN5 = "< 5"
        EQUAL_5_UNTIL_LESSTHAN_10 = "5 <= x < 10"
        EQUAL_10_UNTIL_LESSTHAN_20 = "10 <= x < 20"
        EQUAL_20_UNTIL_LESSTHAN_40 = "20 <= x < 40"
        EQUAL_40_UNTIL_LESSTHAN_60 = "40 <= x < 60"
        EQUALGREATERTHAN_60 = "60 <= x"

    class H2SContent(Enum):
        """
        H2S content variable split component
        """
        LESSTHAN_100 = "<100"
        EQUAL_100_UNTIL_LESSTHAN_1000 = "100 <= x < 1000"
        EQUAL_1000_UNTIL_LESSTHAN_2000 = "1000 <= x < 2000"
        EQUAL_2000_UNTIL_LESSTHAN_3000 = "2000 <= x < 3000"
        EQUAL_3000_UNTIL_LESSTHAN_4000 = "3000 <= x < 4000"
        EQUALGREATERTHAN_4000 = "4000 <= x"

    class APIOil(Enum):
        """
        API density variable split component
        """
        LESSTHAN_25 = "< 25"
        EQUALGREATERTHAN_25 = "25 <= x"

    class DomesticUse(Enum):
        """
        Domestic use variable split component
        """
        EQUAL_30_UNTIL_LESSTHAN_50 = "30 <= x < 50"
        EQUAL_50_UNTIL_LESSTHAN_70 = "50 <= x < 70"
        EQUAL_70_UNTIL_LESSTHAN_100 = "70 <= x < 100"

    class ProductionStage(Enum):
        """
        Production stage variable split component
        """
        PRIMARY = "Primary"
        SECONDARY = "Secondary"
        TERTIARY = "Tertiary"


class VariableSplit082017:
    """
    Variable split corresponds to PERMEN No. 08 Year 2017.
    """
    class FieldStatus(Enum):
        """
        Field status variable split component
        """
        POD_I = "Plan of Development I"
        POD_II = "Plan of Development II"
        POFD = "Plan of Further Development"
        NO_POD = "No Plan of Development"

    class FieldLocation(Enum):
        """
        Field location variable split component
        """
        ONSHORE = "On Shore"
        OFFSHORE_0_UNTIL_LESSEQUAL_20 = "Offshore 0 < h <= 20"
        OFFSHORE_20_UNTIL_LESSEQUAL_50 = "Offshore 20 < h <= 50"
        OFFSHORE_50_UNTIL_LESSEQUAL_150 = "Offshore 50 < h <= 150"
        OFFSHORE_150_UNTIL_LESSEQUAL_1000 = "Offshore 150 < h <= 1000"
        OFFSHORE_GREATERTHAN_1000 = "Offshore h > 1000"

    class ReservoirDepth(Enum):
        """
        Reservoir depth variable split component
        """
        LESSEQUAL_2500 = "Reservoir Depth <=2500"
        GREATERTHAN_2500 = "Reservoir Depth >2500"

    class InfrastructureAvailability(Enum):
        """
        Infrastucture availability variable split component
        """
        WELL_DEVELOPED = "Well Developed"
        NEW_FRONTIER = "New Frontier"

    class ReservoirType(Enum):
        """
        Reservoir type variable split component
        """
        CONVENTIONAL = "Conventional"
        NON_CONVENTIONAL = "Non Conventional"

    class CO2Content(Enum):
        """
        CO2 content variable split component
        """
        LESSTHAN5 = "< 5"
        EQUAL_5_UNTIL_LESSTHAN_10 = "5 <= x < 10"
        EQUAL_10_UNTIL_LESSTHAN_20 = "10 <= x < 20"
        EQUAL_20_UNTIL_LESSTHAN_40 = "20 <= x < 40"
        EQUAL_40_UNTIL_LESSTHAN_60 = "40 <= x < 60"
        EQUALGREATERTHAN_60 = "60 <= x"

    class H2SContent(Enum):
        """
        H2S content variable split component
        """
        LESSTHAN_100 = "<100"
        EQUAL_100_UNTIL_LESSTHAN_300 = "100 <= x < 300"
        EQUAL_300_UNTIL_LESSTHAN_500 = "1000 <= x < 500"
        EQUALGREATERTHAN_500 = "500 <= x"

    class APIOil(Enum):
        """
        API density variable split component
        """
        LESSTHAN_25 = "< 25"
        EQUALGREATERTHAN_25 = "25 <= x"

    class DomesticUse(Enum):
        """
        Domestic use variable split component
        """
        LESSTHAN_30 = "< 30"
        EQUAL_30_UNTIL_LESSTHAN_50 = "30 <= x < 50"
        EQUAL_50_UNTIL_LESSTHAN_70 = "50 <= x < 70"
        EQUAL_70_UNTIL_LESSTHAN_100 = "70 <= x < 100"

    class ProductionStage(Enum):
        """
        Production stage variable split component
        """
        PRIMARY = "Primary"
        SECONDARY = "Secondary"
        TERTIARY = "Tertiary"


class ContractSample(Enum):
    """
    Selection for generating contract sample.
    """
    CASE_1 = "Cost Recovery Contract Sample"
    CASE_2 = "Cost Recovery Consolidated Contract Sample"
    CASE_3 = "Gross Split Contract Sample"
