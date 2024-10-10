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

    Attributes
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
    PERMEN_ESDM_8_2017 = 'Peraturan Menteri ESDM No. 8 Tahun 2017'
    PERMEN_ESDM_52_2017 = 'Peraturan Menteri ESDM No. 52 Tahun 2017'
    PERMEN_ESDM_20_2019 = 'Peraturan Menteri ESDM No. 20 Tahun 2019'
    PERMEN_ESDM_12_2020 = 'Peraturan Menteri ESDM No. 12 Tahun 2020'


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
    TRANSITION_CR_CR: str
        Transition contract type from Cost Recovery to Cost Recovery.
    TRANSITION_CR_GS: str
        Transition contract type from Cost Recovery to Gross Split.
    TRANSITION_GS_GS: str
        Transition contract type from Gross Split to Gross Split.
    TRANSITION_GS_CR: str
        Transition contract type from Gross Split to Cost Recovery.
    """
    COST_RECOVERY = 'Cost Recovery (CR)'
    GROSS_SPLIT = 'Gross Split (GS)'
    TRANSITION_CR_CR = 'Transition CR - CR'
    TRANSITION_CR_GS = 'Transition CR - GS'
    TRANSITION_GS_GS = 'Transition GS - GS'
    TRANSITION_GS_CR = 'Transition GS - CR'


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
    NPV_REAL_TERMS = 'NPV Calculation using real terms method'
    NPV_NOMINAL_TERMS = 'NPV Calculation using nominal terms method'
    NPV_SKK_NOMINAL_TERMS = 'NPV Calculation using SKK Nominal terms method'
    NPV_SKK_REAL_TERMS = 'NPV Calculation using SKK Real terms method'
    NPV_POINT_FORWARD = 'NPV Calculation using Point Forward method'


# Refactor: OtherRevenue into OtherRevenueTreatment
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


class OptimizationParameter(Enum):
    """
    Enumeration class representing different optimization parameters in
    oil and gas economic analysis.

    Attributes
    ----------
    OIL_CTR_PRETAX: str
        Optimization parameter for Oil Contractor Pre Tax.
    GAS_CTR_PRETAX: str
        Optimization parameter for Gas Contractor Pre Tax.
    OIL_FTP_PORTION: str
        Optimization parameter for Oil FTP Portion.
    GAS_FTP_PORTION: str
        Optimization parameter for Gas FTP Portion.
    OIL_IC: str
        Optimization parameter for Oil Investment Credit (IC).
    GAS_IC: str
        Optimization parameter for Gas Investment Credit (IC).
    OIL_DMO_FEE: str
        Optimization parameter for Oil Domestic Market Obligation (DMO) Fee.
    GAS_DMO_FEE: str
        Optimization parameter for Gas Domestic Market Obligation (DMO) Fee.
    VAT_RATE: str
        Optimization parameter for Value-Added Tax (VAT) Rate.
    EFFECTIVE_TAX_RATE: str
        Optimization parameter for Effective Tax Rate.
    MINISTERIAL_DISCRETION: str
        Optimization parameter for Ministerial Discretion.
    VAT_DISCOUNT: str
        Optimization parameter for VAT Discount.
    LBT_DISCOUNT: str
        Optimization parameter for LBT Discount.


    Notes
    -----
    The impact of the optimization of each parameter into the **contractor** economic indicator are as the following:

    OIL_CTR_PRETAX
        A higher value of this optimization parameter enhances the contractor's economic indicator while
        diminishing the government's economic indicator.

    GAS_CTR_PRETAX
        A higher value of this parameter enhances the contractor's economic indicator while
        diminishing the government's economic indicator.

    OIL_FTP_PORTION
        A lower value of this parameter enhances the contractor's economic indicator while
        diminishing the government's economic indicator.

    GAS_FTP_PORTION
        A lower value of this parameter enhances the contractor's economic indicator while
        diminishing the government's economic indicator.

    OIL_IC
        A higher value of this parameter typically enhances the contractor's economic indicator while
        diminishing the government's. However, under certain conditions, the opposite effect may occur.

    GAS_IC
        A higher value of this parameter typically enhances the contractor's economic indicator while
        diminishing the government's. However, under certain conditions, the opposite effect may occur.

    OIL_DMO_FEE
        A higher value of this parameter enhances the contractor's economic indicator while
        diminishing the government's economic indicator.

    GAS_DMO_FEE
        A higher value of this parameter enhances the contractor's economic indicator while
        diminishing the government's economic indicator.

    VAT_RATE
        A lower value of this parameter enhances the contractor's economic indicator while
        diminishing the government's economic indicator.

    EFFECTIVE_TAX_RATE
        A lower value of this parameter enhances the contractor's economic indicator while
        diminishing the government's economic indicator.

    MINISTERIAL_DISCRETION
        A higher value of this parameter enhances the contractor's economic indicator while
        diminishing the government's economic indicator.

    VAT_DISCOUNT
        A higher value of this parameter enhances the contractor's economic indicator while
        diminishing the government's economic indicator.

    LBT_DISCOUNT
        A higher value of this parameter enhances the contractor's economic indicator while
        diminishing the government's economic indicator.

    DEPRECIATION_ACCELERATION
        A higher value of this parameter enhances the contractor's economic indicator while
        diminishing the government's economic indicator.

    """
    OIL_CTR_PRETAX = 'Oil Contractor Pre Tax'
    GAS_CTR_PRETAX = 'Gas Contractor Pre Tax'
    OIL_FTP_PORTION = 'Oil FTP Portion'
    GAS_FTP_PORTION = 'Gas FTP Portion'
    OIL_IC = 'Oil IC'
    GAS_IC = 'Gas IC'
    OIL_DMO_FEE = 'Oil DMO Fee'
    GAS_DMO_FEE = 'Gas Dmo Fee'
    VAT_RATE = 'VAT Rate'
    EFFECTIVE_TAX_RATE = 'Effective Tax Rate'
    MINISTERIAL_DISCRETION = 'Ministerial Discretion'
    VAT_DISCOUNT = 'VAT Discount'
    LBT_DISCOUNT = 'LBT Discount'
    DEPRECIATION_ACCELERATION = 'Depreciation Acceleration'


class OptimizationTarget(Enum):
    """
    Enumeration class representing different optimization targets in economic analysis.

    Attributes
    ----------
    IRR: str
        Optimization target for Internal Rate of Return (IRR).
    NPV: str
        Optimization target for Net Present Value (NPV).
    PI: str
        Optimization target for Profitability Index (PI).
    """
    IRR = 'IRR'
    NPV = 'NPV'
    PI = 'PI'


class VariableSplit522017:
    """
    Variable Spilt Component for Gross Split No.20 Year 2019 Regime.
    """
    class FieldStatus(Enum):
        """
        Field Status variable split component.
        """
        POD_I = 'Plan of Development I'
        POD_II = 'Plan of Development II'
        NO_POD = 'No Plan of Development'

    class FieldLocation(Enum):
        """
        Field Location variable split component.
        """
        ONSHORE = 'On Shore'
        OFFSHORE_0_UNTIL_LESSEQUAL_20 = 'Offshore 0 < h <= 20'
        OFFSHORE_20_UNTIL_LESSEQUAL_50 = 'Offshore 20 < h <= 50'
        OFFSHORE_50_UNTIL_LESSEQUAL_150 = 'Offshore 50 < h <= 150'
        OFFSHORE_150_UNTIL_LESSEQUAL_1000 = 'Offshore 150 < h <= 1000'
        OFFSHORE_GREATERTHAN_1000 = 'Offshore h > 1000'

    class ReservoirDepth(Enum):
        """
        Reservoir Depth variable split component.
        """
        LESSEQUAL_2500 = 'Reservoir Depth <=2500'
        GREATERTHAN_2500 = 'Reservoir Depth >2500'

    class InfrastructureAvailability(Enum):
        """
        Infrastructure Availability variable split component.
        """
        WELL_DEVELOPED = 'Well Developed'
        NEW_FRONTIER_OFFSHORE = 'New Frontier Offshore'
        NEW_FRONTIER_ONSHORE = 'New Frontier Onshore'

    class ReservoirType(Enum):
        """
        Reservoir Type variable split component.
        """
        CONVENTIONAL = 'Conventional'
        NON_CONVENTIONAL = 'Non Conventional'

    class CO2Content(Enum):
        """
        CO2 Content variable split component.
        """
        LESSTHAN_5 = '< 5'
        EQUAL_5_UNTIL_LESSTHAN_10 = '5 <= x < 10'
        EQUAL_10_UNTIL_LESSTHAN_20 = '10 <= x < 20'
        EQUAL_20_UNTIL_LESSTHAN_40 = '20 <= x < 40'
        EQUAL_40_UNTIL_LESSTHAN_60 = '40 <= x < 60'
        EQUALGREATERTHAN_60 = '60 <= x'

    class H2SContent(Enum):
        """
        H2S Content variable split component.
        """
        LESSTHAN_100 = '<100'
        EQUAL_100_UNTIL_LESSTHAN_1000 = '100 <= x < 1000'
        EQUAL_1000_UNTIL_LESSTHAN_2000 = '1000 <= x < 2000'
        EQUAL_2000_UNTIL_LESSTHAN_3000 = '2000 <= x < 3000'
        EQUAL_3000_UNTIL_LESSTHAN_4000 = '3000 <= x < 4000'
        EQUALGREATERTHAN_4000 = '4000 <= x'

    class APIOil(Enum):
        """
        API Density variable split component.
        """
        LESSTHAN_25 = '< 25'
        EQUALGREATERTHAN_25 = '25 <= x'

    class DomesticUse(Enum):
        """
        Domestic Use variable split component.
        """
        EQUAL_30_UNTIL_LESSTHAN_50 = '30 <= x < 50'
        EQUAL_50_UNTIL_LESSTHAN_70 = '50 <= x < 70'
        EQUAL_70_UNTIL_LESSTHAN_100 = '70 <= x < 100'

    class ProductionStage(Enum):
        """
        Production Stage variable split component.
        """
        PRIMARY = 'Primary'
        SECONDARY = 'Secondary'
        TERTIARY = 'Tertiary'


class VariableSplit082017:
    """
    Variable Spilt Component for Gross Split No.08 Year 2017 Regime.
    """
    class FieldStatus(Enum):
        """
        Field Status variable split component.
        """
        POD_I = 'Plan of Development I'
        POD_II = 'Plan of Development II'
        POFD = 'Plan of Further Development'
        NO_POD = 'No Plan of Development'

    class FieldLocation(Enum):
        """
        Field Location variable split component.
        """
        ONSHORE = 'On Shore'
        OFFSHORE_0_UNTIL_LESSEQUAL_20 = 'Offshore 0 < h <= 20'
        OFFSHORE_20_UNTIL_LESSEQUAL_50 = 'Offshore 20 < h <= 50'
        OFFSHORE_50_UNTIL_LESSEQUAL_150 = 'Offshore 50 < h <= 150'
        OFFSHORE_150_UNTIL_LESSEQUAL_1000 = 'Offshore 150 < h <= 1000'
        OFFSHORE_GREATERTHAN_1000 = 'Offshore h > 1000'

    class ReservoirDepth(Enum):
        """
        Reservoir Depth variable split component.
        """
        LESSEQUAL_2500 = 'Reservoir Depth <=2500'
        GREATERTHAN_2500 = 'Reservoir Depth >2500'

    class InfrastructureAvailability(Enum):
        """
        Infrastructure Availability variable split component.
        """
        WELL_DEVELOPED = 'Well Developed'
        NEW_FRONTIER = 'New Frontier'

    class ReservoirType(Enum):
        """
        Reservoir Type variable split component.
        """
        CONVENTIONAL = 'Conventional'
        NON_CONVENTIONAL = 'Non Conventional'

    class CO2Content(Enum):
        """
        CO2 Content variable split component.
        """
        LESSTHAN_5 = '< 5'
        EQUAL_5_UNTIL_LESSTHAN_10 = '5 <= x < 10'
        EQUAL_10_UNTIL_LESSTHAN_20 = '10 <= x < 20'
        EQUAL_20_UNTIL_LESSTHAN_40 = '20 <= x < 40'
        EQUAL_40_UNTIL_LESSTHAN_60 = '40 <= x < 60'
        EQUALGREATERTHAN_60 = '60 <= x'

    class H2SContent(Enum):
        """
        H2S Content variable split component.
        """
        LESSTHAN_100 = '<100'
        EQUAL_100_UNTIL_LESSTHAN_300 = '100 <= x < 300'
        EQUAL_300_UNTIL_LESSTHAN_500 = '1000 <= x < 500'
        EQUALGREATERTHAN_500 = '500 <= x'

    class APIOil(Enum):
        """
        API Density variable split component.
        """
        LESSTHAN_25 = '< 25'
        EQUALGREATERTHAN_25 = '25 <= x'

    class DomesticUse(Enum):
        """
        Domestic Use variable split component.
        """
        LESSTHAN_30 = '< 30'
        EQUAL_30_UNTIL_LESSTHAN_50 = '30 <= x < 50'
        EQUAL_50_UNTIL_LESSTHAN_70 = '50 <= x < 70'
        EQUAL_70_UNTIL_LESSTHAN_100 = '70 <= x < 100'

    class ProductionStage(Enum):
        """
        Production Stage variable split component.
        """
        PRIMARY = 'Primary'
        SECONDARY = 'Secondary'
        TERTIARY = 'Tertiary'



class ContractSample(Enum):
    """
    Selection for generating contract sample.
    """
    CASE_1 = 'Cost Recovery Contract Sample'
    CASE_2 = 'Cost Recovery Consolidated Contract Sample'
    CASE_3 = 'Gross Split Contract Sample'