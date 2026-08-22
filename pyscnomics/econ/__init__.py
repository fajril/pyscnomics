"""
Specify callable classes and methods from package 'econ'
"""

from . import costs as costs
from . import costs_tools as costs_tools
from . import depreciation as depreciation
from . import indicator as indicator
from . import revenue as revenue
from . import selection as selection
from .revenue import Lifting as Lifting
from .costs import (
    ASR as ASR,
    CapitalCost as CapitalCost,
    CostOfSales as CostOfSales,
    GeneralCost as GeneralCost,
    Intangible as Intangible,
    LBT as LBT,
    OPEX as OPEX,
)
from .depreciation import (
    declining_balance_book_value as declining_balance_book_value,
    declining_balance_depreciation_rate as declining_balance_depreciation_rate,
    psc_declining_balance_book_value as psc_declining_balance_book_value,
    psc_declining_balance_depreciation_rate as psc_declining_balance_depreciation_rate,
    straight_line_book_value as straight_line_book_value,
    straight_line_depreciation_rate as straight_line_depreciation_rate,
    unit_of_production_book_value as unit_of_production_book_value,
    unit_of_production_rate as unit_of_production_rate,
)
from .selection import (
    ContractType as ContractType,
    DeprMethod as DeprMethod,
    DiscountingMode as DiscountingMode,
    ExpendituresType as ExpendituresType,
    FTPTaxRegime as FTPTaxRegime,
    FluidType as FluidType,
    GrossSplitRegime as GrossSplitRegime,
    InflationAppliedTo as InflationAppliedTo,
    NPVSelection as NPVSelection,
    OptimizationParameter as OptimizationParameter,
    OptimizationTarget as OptimizationTarget,
    OtherRevenue as OtherRevenue,
    TaxPaymentMode as TaxPaymentMode,
    TaxRegime as TaxRegime,
    TaxSplitTypeCR as TaxSplitTypeCR,
    TaxType as TaxType,
    VariableSplit082017 as VariableSplit082017,
    VariableSplit522017 as VariableSplit522017,
    YearReference as YearReference,
)



from .indicator import (
    irr as irr,
    npv as npv,
    npv_nominal_terms as npv_nominal_terms,
    npv_point_forward as npv_point_forward,
    npv_real_terms as npv_real_terms,
    npv_skk_nominal_terms as npv_skk_nominal_terms,
    npv_skk_real_terms as npv_skk_real_terms,
    pot as pot,
    pot_psc as pot_psc,
    xirr as xirr,
    xnpv as xnpv,
)

from . import limit as limit
from .limit import (
    LimitMethod as LimitMethod,
    econ_limit as econ_limit,
    np as np,
)
