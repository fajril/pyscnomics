"""
Configuration to undertake sensitivity analysis.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from numbers import Real
from typing import (
    Literal,
    TypeAlias,
    TypeVar,
    TypedDict,
)

import numpy as np
import numpy.typing as npt

from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.costrecovery import RCICPInput
from pyscnomics.contracts.gross_split import GrossSplit
from pyscnomics.contracts.project import BaseProject, require_float_array
from pyscnomics.contracts.transition import Transition
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.costs import ASR, CapitalCost, Intangible, OPEX
from pyscnomics.econ.selection import (
    DeprMethod,
    DiscountingMode,
    FTPTaxRegime,
    InflationAppliedTo,
    NPVSelection,
    OtherRevenue,
    TaxRegime,
    TaxSplitTypeCR,
)


PSCKey: TypeAlias = Literal["PSC 1", "PSC 2"]
T = TypeVar("T")
NumericArray: TypeAlias = npt.NDArray[np.float64]
SingleAggregate: TypeAlias = tuple[T, ...]
TransitionAggregate: TypeAlias = dict[PSCKey, SingleAggregate[T]]
Aggregate: TypeAlias = SingleAggregate[T] | TransitionAggregate[T]

LiftingAggregate: TypeAlias = SingleAggregate[Lifting]
LiftingTransitionAggregate: TypeAlias = TransitionAggregate[Lifting]
LiftingAggregateInput: TypeAlias = LiftingAggregate | LiftingTransitionAggregate
LiftingAggregateOutput: TypeAlias = LiftingAggregate | LiftingTransitionAggregate

OpexAggregate: TypeAlias = SingleAggregate[OPEX]
OpexTransitionAggregate: TypeAlias = TransitionAggregate[OPEX]
OpexAggregateInput: TypeAlias = OpexAggregate | OpexTransitionAggregate
OpexAggregateOutput: TypeAlias = OpexAggregate | OpexTransitionAggregate

CapitalAggregate: TypeAlias = SingleAggregate[CapitalCost]
CapitalTransitionAggregate: TypeAlias = TransitionAggregate[CapitalCost]
CapitalAggregateInput: TypeAlias = CapitalAggregate | CapitalTransitionAggregate
CapitalAggregateOutput: TypeAlias = CapitalAggregate | CapitalTransitionAggregate

IntangibleAggregate: TypeAlias = SingleAggregate[Intangible]
IntangibleTransitionAggregate: TypeAlias = TransitionAggregate[Intangible]
IntangibleAggregateInput: TypeAlias = (
    IntangibleAggregate | IntangibleTransitionAggregate
)
IntangibleAggregateOutput: TypeAlias = (
    IntangibleAggregate | IntangibleTransitionAggregate
)

ASRAggregate: TypeAlias = SingleAggregate[ASR]
ASRTransitionAggregate: TypeAlias = TransitionAggregate[ASR]
ASRAggregateInput: TypeAlias = ASRAggregate | ASRTransitionAggregate

ScalarOrPSC: TypeAlias = T | dict[PSCKey, T]
NumericValue: TypeAlias = float | int | NumericArray
ConfigValue: TypeAlias = (
    NumericValue | None | TaxRegime | str | FTPTaxRegime | DeprMethod | OtherRevenue
)
ConfigT = TypeVar("ConfigT", bound=ConfigValue)

SingleContract: TypeAlias = BaseProject | CostRecovery | GrossSplit
TransitionContracts: TypeAlias = dict[PSCKey, CostRecovery | GrossSplit]
PSCState: TypeAlias = SingleContract | TransitionContracts
ContractArguments: TypeAlias = dict[str, object]
TransitionArguments: TypeAlias = dict[PSCKey, ContractArguments]
ArgumentsState: TypeAlias = ContractArguments | TransitionArguments
SummaryArguments: TypeAlias = dict[str, object]
ActivationResult: TypeAlias = tuple[
    SingleContract | Transition, ArgumentsState | None, SummaryArguments
]


@dataclass
class GeneralConfigData:
    discount_rate_start_year: int
    discount_rate: float
    type_of_contract: str
    start_date_project: date
    end_date_project: date
    oil_onstream_date: date | None
    gas_onstream_date: date | None
    start_date_project_second: date
    end_date_project_second: date
    oil_onstream_date_second: date | None
    gas_onstream_date_second: date | None
    inflation_rate_applied_to: InflationAppliedTo | None


@dataclass
class FiscalConfigData:
    npv_mode: NPVSelection
    discounting_mode: DiscountingMode
    inflation_rate: ScalarOrPSC[NumericValue | None]
    tax_mode: ScalarOrPSC[TaxRegime | str]
    tax_rate: ScalarOrPSC[NumericValue | None]
    tax_payment_config: ScalarOrPSC[FTPTaxRegime | str]
    asr_future_rate: ScalarOrPSC[float | int]
    depreciation_method: ScalarOrPSC[DeprMethod | str]
    decline_factor: ScalarOrPSC[float | int]
    vat_rate: ScalarOrPSC[NumericValue]
    lbt_rate: ScalarOrPSC[NumericValue]
    sulfur_revenue_config: ScalarOrPSC[OtherRevenue | str]
    electricity_revenue_config: ScalarOrPSC[OtherRevenue | str]
    co2_revenue_config: ScalarOrPSC[OtherRevenue | str]
    transferred_unrec_cost: float
    sunk_cost_reference_year: int


@dataclass
class DMOConfig:
    dmo_is_weighted: bool
    oil_dmo_volume: float | NumericArray
    oil_dmo_fee: float | NumericArray
    oil_dmo_period: int
    gas_dmo_volume: float | NumericArray
    gas_dmo_fee: float | NumericArray
    gas_dmo_period: int


@dataclass
class CostRecoveryConfig(DMOConfig):
    oil_ftp_availability: bool
    oil_ftp_is_shared: bool
    oil_ftp_portion: float | NumericArray
    gas_ftp_availability: bool
    gas_ftp_is_shared: bool
    gas_ftp_portion: float | NumericArray
    split_type: TaxSplitTypeCR
    icp_sliding_scale: RCICPInput
    indicator_rc_split_sliding_scale: NumericArray | None
    oil_ctr_pretax: float | NumericArray
    gas_ctr_pretax: float | NumericArray
    ic_oil: float | NumericArray
    ic_gas: float | NumericArray
    ic_availability: bool
    oil_cr_cap_rate: float
    gas_cr_cap_rate: float


@dataclass
class GrossSplitConfig(DMOConfig):
    field_status: str
    field_location: str
    reservoir_depth: str
    infrastructure_availability: str
    reservoir_type: str
    oil_api: str
    domestic_content_use: str
    production_stage: str
    co2_content: str
    h2s_content: str
    ministry_discretion_split: float


@dataclass
class AdjustDataInput:
    """Concrete attribute contract for the legacy adjustment data boundary."""

    general_config_data: GeneralConfigData
    fiscal_config_data: FiscalConfigData
    oil_lifting_aggregate_total: LiftingAggregateInput
    gas_lifting_aggregate_total: LiftingAggregateInput
    sulfur_lifting_aggregate: LiftingAggregateInput
    electricity_lifting_aggregate: LiftingAggregateInput
    co2_lifting_aggregate: LiftingAggregateInput
    capital_cost_aggregate: CapitalAggregateInput
    intangible_cost_aggregate: IntangibleAggregateInput
    opex_aggregate: OpexAggregateInput
    asr_cost_aggregate: ASRAggregateInput
    psc_cr_data: CostRecoveryConfig | None
    psc_gs_data: GrossSplitConfig | None
    psc_transition_cr_to_cr: Mapping[str, CostRecoveryConfig] | None
    psc_transition_cr_to_gs: Mapping[str, CostRecoveryConfig | GrossSplitConfig] | None
    psc_transition_gs_to_gs: Mapping[str, GrossSplitConfig] | None
    psc_transition_gs_to_cr: Mapping[str, GrossSplitConfig | CostRecoveryConfig] | None


class SensitivityData(TypedDict):
    oil_lifting_aggregate_total: LiftingAggregateOutput
    gas_lifting_aggregate_total: LiftingAggregateOutput
    sulfur_lifting_aggregate: LiftingAggregateOutput
    electricity_lifting_aggregate: LiftingAggregateOutput
    co2_lifting_aggregate: LiftingAggregateOutput
    opex_aggregate: OpexAggregateOutput
    tangible_cost_aggregate: CapitalAggregateOutput
    intangible_cost_aggregate: IntangibleAggregateOutput


def _single_lifting_total(data: SensitivityData) -> LiftingAggregate:
    return (
        _single_aggregate(data["oil_lifting_aggregate_total"])
        + _single_aggregate(data["gas_lifting_aggregate_total"])
        + _single_aggregate(data["sulfur_lifting_aggregate"])
        + _single_aggregate(data["electricity_lifting_aggregate"])
        + _single_aggregate(data["co2_lifting_aggregate"])
    )


def _transition_lifting_total(data: SensitivityData) -> LiftingTransitionAggregate:
    oil = _transition_aggregate(data["oil_lifting_aggregate_total"])
    gas = _transition_aggregate(data["gas_lifting_aggregate_total"])
    sulfur = _transition_aggregate(data["sulfur_lifting_aggregate"])
    electricity = _transition_aggregate(data["electricity_lifting_aggregate"])
    co2 = _transition_aggregate(data["co2_lifting_aggregate"])
    return {
        psc: oil[psc] + gas[psc] + sulfur[psc] + electricity[psc] + co2[psc]
        for psc in PSC_KEYS
    }


def _single_aggregate(value: Aggregate[T]) -> SingleAggregate[T]:
    if not isinstance(value, tuple):
        raise AdjustDataException("A single-contract aggregate must be a tuple")
    return value


def _transition_aggregate(value: Aggregate[T]) -> TransitionAggregate[T]:
    if not isinstance(value, dict):
        raise AdjustDataException("A transition aggregate must be a PSC mapping")
    return value


def _single_config(value: ConfigT | Mapping[PSCKey, ConfigT]) -> ConfigT:
    if isinstance(value, Mapping):
        raise AdjustDataException("A single-contract config must be scalar")
    return value


def _transition_config(value: ConfigT | dict[PSCKey, ConfigT], psc: PSCKey) -> ConfigT:
    if not isinstance(value, dict):
        raise AdjustDataException("A transition config must be a PSC mapping")
    if psc not in value:
        raise AdjustDataException(f"Transition config is missing {psc}")
    return value[psc]


def _multiplier(values: NumericArray, index: int) -> float:
    value = values.flat[index]
    if not isinstance(value, Real):
        raise AdjustDataException("Sensitivity multipliers must be numeric")
    return float(value)


class AdjustDataException(Exception):
    """Exception to be raised for a misuse of AdjustData class"""

    pass


PSC_KEYS: tuple[PSCKey, PSCKey] = ("PSC 1", "PSC 2")


def _adjust_lifting(
    lifting: Lifting,
    price_multiplier: float,
    rate_multiplier: float,
) -> Lifting:
    return Lifting(
        start_year=lifting.start_year,
        end_year=lifting.end_year,
        lifting_rate=lifting.lifting_rate * rate_multiplier,
        price=lifting.price * price_multiplier,
        prod_year=lifting.prod_year,
        fluid_type=lifting.fluid_type,
        ghv=lifting.ghv,
        prod_rate=lifting.prod_rate,
        prod_rate_baseline=lifting.prod_rate_baseline,
    )


def _adjust_opex(opex: OPEX, multiplier: float) -> OPEX:
    return OPEX(
        start_year=opex.start_year,
        end_year=opex.end_year,
        expense_year=opex.expense_year,
        cost_allocation=opex.cost_allocation,
        cost_type=opex.cost_type,
        description=opex.description,
        tax_portion=opex.tax_portion,
        tax_discount=opex.tax_discount,
        fixed_cost=require_float_array(opex.fixed_cost, "fixed_cost") * multiplier,
        prod_rate=opex.prod_rate,
        cost_per_volume=require_float_array(opex.cost_per_volume, "cost_per_volume")
        * multiplier,
    )


def _adjust_capital_cost(capital: CapitalCost, multiplier: float) -> CapitalCost:
    return CapitalCost(
        start_year=capital.start_year,
        end_year=capital.end_year,
        expense_year=capital.expense_year,
        cost=require_float_array(capital.cost, "cost") * multiplier,
        cost_allocation=capital.cost_allocation,
        cost_type=capital.cost_type,
        description=capital.description,
        tax_portion=capital.tax_portion,
        tax_discount=capital.tax_discount,
        pis_year=capital.pis_year,
        salvage_value=capital.salvage_value,
        useful_life=capital.useful_life,
        depreciation_factor=capital.depreciation_factor,
        is_ic_applied=capital.is_ic_applied,
    )


def _adjust_intangible(intangible: Intangible, multiplier: float) -> Intangible:
    return Intangible(
        start_year=intangible.start_year,
        end_year=intangible.end_year,
        expense_year=intangible.expense_year,
        cost=require_float_array(intangible.cost, "cost") * multiplier,
        cost_allocation=intangible.cost_allocation,
        cost_type=intangible.cost_type,
        description=intangible.description,
        tax_portion=intangible.tax_portion,
        tax_discount=intangible.tax_discount,
    )


def _cost_recovery_config(
    config: CostRecoveryConfig | GrossSplitConfig | None,
) -> CostRecoveryConfig:
    if not isinstance(config, CostRecoveryConfig):
        raise AdjustDataException("Expected a Cost Recovery configuration")
    return config


def _gross_split_config(
    config: CostRecoveryConfig | GrossSplitConfig | None,
) -> GrossSplitConfig:
    if not isinstance(config, GrossSplitConfig):
        raise AdjustDataException("Expected a Gross Split configuration")
    return config


def _project_dates(
    general: GeneralConfigData, psc: PSCKey
) -> tuple[date, date, date | None, date | None, int]:
    if psc == "PSC 1":
        start_date = general.start_date_project
        end_date = general.end_date_project
        oil_onstream_date = general.oil_onstream_date
        gas_onstream_date = general.gas_onstream_date
    else:
        start_date = general.start_date_project_second
        end_date = general.end_date_project_second
        oil_onstream_date = general.oil_onstream_date_second
        gas_onstream_date = general.gas_onstream_date_second
    return start_date, end_date, oil_onstream_date, gas_onstream_date, start_date.year


def _build_cost_recovery(
    general: GeneralConfigData,
    config: CostRecoveryConfig,
    lifting: LiftingAggregate,
    capital_cost: CapitalAggregate,
    intangible_cost: IntangibleAggregate,
    opex: OpexAggregate,
    asr_cost: ASRAggregate,
    psc: PSCKey,
) -> CostRecovery:
    start_date, end_date, oil_onstream_date, gas_onstream_date, approval_year = (
        _project_dates(general, psc)
    )
    return CostRecovery(
        start_date=start_date,
        end_date=end_date,
        approval_year=approval_year,
        oil_onstream_date=oil_onstream_date,
        gas_onstream_date=gas_onstream_date,
        lifting=lifting,
        capital_cost=capital_cost,
        intangible_cost=intangible_cost,
        opex=opex,
        asr_cost=asr_cost,
        oil_ftp_is_available=config.oil_ftp_availability,
        oil_ftp_is_shared=config.oil_ftp_is_shared,
        oil_ftp_portion=config.oil_ftp_portion,
        gas_ftp_is_available=config.gas_ftp_availability,
        gas_ftp_is_shared=config.gas_ftp_is_shared,
        gas_ftp_portion=config.gas_ftp_portion,
        tax_split_type=config.split_type,
        condition_dict=config.icp_sliding_scale,
        indicator_rc_icp_sliding=config.indicator_rc_split_sliding_scale,
        oil_ctr_pretax_share=config.oil_ctr_pretax,
        gas_ctr_pretax_share=config.gas_ctr_pretax,
        oil_ic_rate=config.ic_oil,
        gas_ic_rate=config.ic_gas,
        ic_is_available=config.ic_availability,
        oil_cr_cap_rate=config.oil_cr_cap_rate,
        gas_cr_cap_rate=config.gas_cr_cap_rate,
        oil_dmo_volume_portion=config.oil_dmo_volume,
        oil_dmo_fee_portion=config.oil_dmo_fee,
        oil_dmo_holiday_duration=config.oil_dmo_period,
        gas_dmo_volume_portion=config.gas_dmo_volume,
        gas_dmo_fee_portion=config.gas_dmo_fee,
        gas_dmo_holiday_duration=config.gas_dmo_period,
    )


def _build_gross_split(
    general: GeneralConfigData,
    config: GrossSplitConfig,
    lifting: LiftingAggregate,
    capital_cost: CapitalAggregate,
    intangible_cost: IntangibleAggregate,
    opex: OpexAggregate,
    asr_cost: ASRAggregate,
    psc: PSCKey,
) -> GrossSplit:
    start_date, end_date, oil_onstream_date, gas_onstream_date, approval_year = (
        _project_dates(general, psc)
    )
    return GrossSplit(
        start_date=start_date,
        end_date=end_date,
        approval_year=approval_year,
        oil_onstream_date=oil_onstream_date,
        gas_onstream_date=gas_onstream_date,
        lifting=lifting,
        capital_cost=capital_cost,
        intangible_cost=intangible_cost,
        opex=opex,
        asr_cost=asr_cost,
        field_status=config.field_status,
        field_loc=config.field_location,
        res_depth=config.reservoir_depth,
        infra_avail=config.infrastructure_availability,
        res_type=config.reservoir_type,
        api_oil=config.oil_api,
        domestic_use=config.domestic_content_use,
        prod_stage=config.production_stage,
        co2_content=config.co2_content,
        h2s_content=config.h2s_content,
        split_ministry_disc=config.ministry_discretion_split,
        oil_dmo_volume_portion=config.oil_dmo_volume,
        oil_dmo_fee_portion=config.oil_dmo_fee,
        oil_dmo_holiday_duration=config.oil_dmo_period,
        gas_dmo_volume_portion=config.gas_dmo_volume,
        gas_dmo_fee_portion=config.gas_dmo_fee,
        gas_dmo_holiday_duration=config.gas_dmo_period,
    )


U = TypeVar("U", bound=ConfigValue)


def _contract_value(value: U | dict[PSCKey, U], psc: PSCKey | None) -> U:
    if psc is None:
        return _single_config(value)
    return _transition_config(value, psc)


def _build_contract_arguments(
    general: GeneralConfigData,
    fiscal: FiscalConfigData,
    config: DMOConfig,
    psc: PSCKey | None,
    is_cost_recovery: bool,
) -> ContractArguments:
    arguments: ContractArguments = {
        "sulfur_revenue": _contract_value(fiscal.sulfur_revenue_config, psc),
        "electricity_revenue": _contract_value(fiscal.electricity_revenue_config, psc),
        "co2_revenue": _contract_value(fiscal.co2_revenue_config, psc),
        "is_dmo_end_weighted": config.dmo_is_weighted,
        "tax_regime": _contract_value(fiscal.tax_mode, psc),
        "effective_tax_rate": _contract_value(fiscal.tax_rate, psc),
        "depr_method": _contract_value(fiscal.depreciation_method, psc),
        "vat_rate": _contract_value(fiscal.vat_rate, psc),
        "inflation_rate": _contract_value(fiscal.inflation_rate, psc),
        "inflation_rate_applied_to": general.inflation_rate_applied_to,
    }
    if is_cost_recovery:
        arguments["ftp_tax_regime"] = _contract_value(fiscal.tax_payment_config, psc)
        arguments["decline_factor"] = _contract_value(fiscal.decline_factor, psc)
    else:
        arguments["decline_factor"] = _contract_value(fiscal.decline_factor, psc)
    return arguments


def get_price_and_rate_adjustment(
    contract_type: str,
    lifting_aggregate: LiftingAggregateInput,
    price_multiplier: float,
    rate_multiplier: float,
) -> LiftingAggregateOutput:
    """
    Adjust price and lifting rate for sensitivity analysis.

    Parameters
    ----------
    contract_type: str
        The type of contract.
    lifting_aggregate: tuple or dict
        The aggregate of lifting data for a particular fluid type.
    price_multiplier: float
        Multiplier to adjust prices.
    rate_multiplier: float
        Multiplier to adjust lifting rates.

    Returns
    -------
    tuple or dict
        A new instance of lifting aggregate with updated price and lifting rate.
    """
    if "Transition" not in contract_type:
        aggregate = _single_aggregate(lifting_aggregate)
        return tuple(
            _adjust_lifting(item, price_multiplier, rate_multiplier)
            for item in aggregate
        )

    aggregate = _transition_aggregate(lifting_aggregate)
    return {
        psc: tuple(
            _adjust_lifting(item, price_multiplier, rate_multiplier)
            for item in aggregate[psc]
        )
        for psc in PSC_KEYS
    }


def get_rate_adjustment(
    contract_type: str,
    lifting_aggregate: LiftingAggregateInput,
    rate_multiplier: float,
) -> LiftingAggregateOutput:
    """
    Adjust lifting rate for sensitivity analysis.

    Parameters
    ----------
    contract_type: str
        The type of contract.
    lifting_aggregate: tuple or dict
        The aggregate of lifting for a particular fluid type.
    rate_multiplier: float
        A scalar multiplier to adjust lifting rate.

    Returns
    -------
    tuple or dict
        A new instance of lifting aggregate with updated lifting rate.
    """
    if "Transition" not in contract_type:
        aggregate = _single_aggregate(lifting_aggregate)
        return tuple(_adjust_lifting(item, 1.0, rate_multiplier) for item in aggregate)

    aggregate = _transition_aggregate(lifting_aggregate)
    return {
        psc: tuple(
            _adjust_lifting(item, 1.0, rate_multiplier) for item in aggregate[psc]
        )
        for psc in PSC_KEYS
    }


def get_opex_adjustment(
    contract_type: str,
    opex_aggregate: OpexAggregateInput,
    opex_multiplier: float,
) -> OpexAggregateOutput:
    """
    Adjust opex for sensitivity analysis.

    Parameters
    ----------
    contract_type: str
        The type of contract.
    opex_aggregate: tuple or dict
        The aggregate of opex.
    opex_multiplier: float
        A scalar multiplier to adjust opex fixed_cost and cost_per_volume.

    Returns
    -------
    tuple or dict
        A new instance of opex aggregate with updated fixed_cost and cost_per_volume.
    """
    if "Transition" not in contract_type:
        aggregate = _single_aggregate(opex_aggregate)
        return tuple(_adjust_opex(item, opex_multiplier) for item in aggregate)

    aggregate = _transition_aggregate(opex_aggregate)
    return {
        psc: tuple(_adjust_opex(item, opex_multiplier) for item in aggregate[psc])
        for psc in PSC_KEYS
    }


def get_tangible_adjustment(
    contract_type: str,
    tangible_aggregate: CapitalAggregateInput,
    tangible_multiplier: float,
) -> CapitalAggregateOutput:
    """
    Adjust tangible cost for sensitivity analysis.

    Parameters
    ----------
    contract_type: str
        The type of contract.
    tangible_aggregate: tuple | dict
        The aggregate of tangible.
    tangible_multiplier: float
        A scalar multiplier to adjust tangible cost.

    Returns
    -------
    tuple or dict
        A new instance of tangible cost aggregate with updated cost.
    """
    if "Transition" not in contract_type:
        aggregate = _single_aggregate(tangible_aggregate)
        return tuple(
            _adjust_capital_cost(item, tangible_multiplier) for item in aggregate
        )

    aggregate = _transition_aggregate(tangible_aggregate)
    return {
        psc: tuple(
            _adjust_capital_cost(item, tangible_multiplier) for item in aggregate[psc]
        )
        for psc in PSC_KEYS
    }


def get_intangible_adjustment(
    contract_type: str,
    intangible_aggregate: IntangibleAggregateInput,
    intangible_multiplier: float,
) -> IntangibleAggregateOutput:
    """
    Adjust intangible cost for sensitivity analysis.

    The aggregate keeps its historical tuple shape for single contracts and PSC
    mapping shape for transitions while only the cost field is scaled.
    """
    if "Transition" not in contract_type:
        aggregate = _single_aggregate(intangible_aggregate)
        return tuple(
            _adjust_intangible(item, intangible_multiplier) for item in aggregate
        )

    aggregate = _transition_aggregate(intangible_aggregate)
    return {
        psc: tuple(
            _adjust_intangible(item, intangible_multiplier) for item in aggregate[psc]
        )
        for psc in PSC_KEYS
    }


@dataclass
class AdjustData:
    """Prepare sensitivity-adjusted inputs and activate a contract."""

    multipliers: NumericArray
    workbook_path: str | None = field(default=None)
    data: AdjustDataInput | None = field(default=None, repr=False)

    contract_type: str | None = field(default=None, init=False, repr=False)
    summary_arguments: SummaryArguments | None = field(
        default=None, init=False, repr=False
    )
    psc: PSCState | None = field(default=None, init=False, repr=False)
    psc_arguments: ArgumentsState | None = field(default=None, init=False, repr=False)
    sensitivity_data: SensitivityData | None = field(default=None, init=False)
    transition: Transition | None = field(default=None, init=False, repr=False)
    transition_arguments: ContractArguments | None = field(
        default=None, init=False, repr=False
    )
    _data: AdjustDataInput = field(init=False, repr=False)

    def _require_data(self) -> AdjustDataInput:
        if self.data is None:
            raise AdjustDataException("AdjustData requires an input data object")
        return self.data

    def _require_summary_arguments(self) -> SummaryArguments:
        if self.summary_arguments is None:
            raise AdjustDataException("AdjustData has not been initialized")
        return self.summary_arguments

    def _require_sensitivity_data(self) -> SensitivityData:
        if self.sensitivity_data is None:
            raise AdjustDataException("AdjustData has not been initialized")
        return self.sensitivity_data

    def __post_init__(self) -> None:
        data = self._require_data()
        self._data = data
        self.contract_type = data.general_config_data.type_of_contract
        summary: SummaryArguments = {
            "reference_year": data.general_config_data.discount_rate_start_year,
            "inflation_rate": data.fiscal_config_data.inflation_rate,
            "discount_rate": data.general_config_data.discount_rate,
            "npv_mode": data.fiscal_config_data.npv_mode,
            "discounting_mode": data.fiscal_config_data.discounting_mode,
        }
        self.summary_arguments = summary
        sensitivity: SensitivityData = {
            "oil_lifting_aggregate_total": get_price_and_rate_adjustment(
                contract_type=self.contract_type,
                lifting_aggregate=data.oil_lifting_aggregate_total,
                price_multiplier=_multiplier(self.multipliers, 0),
                rate_multiplier=_multiplier(self.multipliers, 4),
            ),
            "gas_lifting_aggregate_total": get_price_and_rate_adjustment(
                contract_type=self.contract_type,
                lifting_aggregate=data.gas_lifting_aggregate_total,
                price_multiplier=_multiplier(self.multipliers, 1),
                rate_multiplier=_multiplier(self.multipliers, 4),
            ),
            "sulfur_lifting_aggregate": get_rate_adjustment(
                contract_type=self.contract_type,
                lifting_aggregate=data.sulfur_lifting_aggregate,
                rate_multiplier=_multiplier(self.multipliers, 4),
            ),
            "electricity_lifting_aggregate": get_rate_adjustment(
                contract_type=self.contract_type,
                lifting_aggregate=data.electricity_lifting_aggregate,
                rate_multiplier=_multiplier(self.multipliers, 4),
            ),
            "co2_lifting_aggregate": get_rate_adjustment(
                contract_type=self.contract_type,
                lifting_aggregate=data.co2_lifting_aggregate,
                rate_multiplier=_multiplier(self.multipliers, 4),
            ),
            "opex_aggregate": get_opex_adjustment(
                contract_type=self.contract_type,
                opex_aggregate=data.opex_aggregate,
                opex_multiplier=_multiplier(self.multipliers, 2),
            ),
            "tangible_cost_aggregate": get_tangible_adjustment(
                contract_type=self.contract_type,
                tangible_aggregate=data.capital_cost_aggregate,
                tangible_multiplier=_multiplier(self.multipliers, 3),
            ),
            "intangible_cost_aggregate": get_intangible_adjustment(
                contract_type=self.contract_type,
                intangible_aggregate=data.intangible_cost_aggregate,
                intangible_multiplier=_multiplier(self.multipliers, 3),
            ),
        }
        self.sensitivity_data = sensitivity

    def _get_single_contract_project(self) -> ActivationResult:
        data = self._data
        general = data.general_config_data
        fiscal = data.fiscal_config_data
        sensitivity = self._require_sensitivity_data()
        summary = self._require_summary_arguments()
        psc = BaseProject(
            start_date=general.start_date_project,
            end_date=general.end_date_project,
            approval_year=general.start_date_project.year,
            oil_onstream_date=general.oil_onstream_date,
            gas_onstream_date=general.gas_onstream_date,
            lifting=_single_lifting_total(sensitivity),
            capital_cost=_single_aggregate(sensitivity["tangible_cost_aggregate"]),
            intangible_cost=_single_aggregate(sensitivity["intangible_cost_aggregate"]),
            opex=_single_aggregate(sensitivity["opex_aggregate"]),
            asr_cost=_single_aggregate(data.asr_cost_aggregate),
        )
        project_arguments: ContractArguments = {
            "sulfur_revenue": _single_config(fiscal.sulfur_revenue_config),
            "electricity_revenue": _single_config(fiscal.electricity_revenue_config),
            "co2_revenue": _single_config(fiscal.co2_revenue_config),
            "vat_rate": _single_config(fiscal.vat_rate),
            "inflation_rate": _single_config(fiscal.inflation_rate),
            "inflation_rate_applied_to": general.inflation_rate_applied_to,
        }
        self.psc = psc
        self.psc_arguments = project_arguments
        summary["contract"] = psc
        return psc, None, summary

    def _get_single_contract_cr(self) -> ActivationResult:
        data = self._data
        general = data.general_config_data
        fiscal = data.fiscal_config_data
        sensitivity = self._require_sensitivity_data()
        summary = self._require_summary_arguments()
        config = _cost_recovery_config(data.psc_cr_data)
        psc = _build_cost_recovery(
            general=general,
            config=config,
            lifting=_single_lifting_total(sensitivity),
            capital_cost=_single_aggregate(sensitivity["tangible_cost_aggregate"]),
            intangible_cost=_single_aggregate(sensitivity["intangible_cost_aggregate"]),
            opex=_single_aggregate(sensitivity["opex_aggregate"]),
            asr_cost=_single_aggregate(data.asr_cost_aggregate),
            psc="PSC 1",
        )
        arguments = _build_contract_arguments(
            general=general,
            fiscal=fiscal,
            config=config,
            psc=None,
            is_cost_recovery=True,
        )
        self.psc = psc
        self.psc_arguments = arguments
        summary["contract"] = psc
        return psc, arguments, summary

    def _get_single_contract_gs(self) -> ActivationResult:
        data = self._data
        general = data.general_config_data
        fiscal = data.fiscal_config_data
        sensitivity = self._require_sensitivity_data()
        summary = self._require_summary_arguments()
        config = _gross_split_config(data.psc_gs_data)
        psc = _build_gross_split(
            general=general,
            config=config,
            lifting=_single_lifting_total(sensitivity),
            capital_cost=_single_aggregate(sensitivity["tangible_cost_aggregate"]),
            intangible_cost=_single_aggregate(sensitivity["intangible_cost_aggregate"]),
            opex=_single_aggregate(sensitivity["opex_aggregate"]),
            asr_cost=_single_aggregate(data.asr_cost_aggregate),
            psc="PSC 1",
        )
        arguments = _build_contract_arguments(
            general=general,
            fiscal=fiscal,
            config=config,
            psc=None,
            is_cost_recovery=False,
        )
        self.psc = psc
        self.psc_arguments = arguments
        summary["contract"] = psc
        return psc, arguments, summary

    def _get_transition_contract(
        self,
        config1: CostRecoveryConfig | GrossSplitConfig,
        config2: CostRecoveryConfig | GrossSplitConfig,
        is_cost_recovery1: bool,
        is_cost_recovery2: bool,
    ) -> ActivationResult:
        data = self._data
        general = data.general_config_data
        fiscal = data.fiscal_config_data
        sensitivity = self._require_sensitivity_data()
        summary = self._require_summary_arguments()
        lifting_total = _transition_lifting_total(sensitivity)
        capital_cost = _transition_aggregate(sensitivity["tangible_cost_aggregate"])
        intangible_cost = _transition_aggregate(
            sensitivity["intangible_cost_aggregate"]
        )
        opex = _transition_aggregate(sensitivity["opex_aggregate"])
        asr_cost = _transition_aggregate(data.asr_cost_aggregate)

        if is_cost_recovery1:
            config_cr1 = _cost_recovery_config(config1)
            contract1: CostRecovery | GrossSplit = _build_cost_recovery(
                general=general,
                config=config_cr1,
                lifting=lifting_total["PSC 1"],
                capital_cost=capital_cost["PSC 1"],
                intangible_cost=intangible_cost["PSC 1"],
                opex=opex["PSC 1"],
                asr_cost=asr_cost["PSC 1"],
                psc="PSC 1",
            )
        else:
            config_gs1 = _gross_split_config(config1)
            contract1 = _build_gross_split(
                general=general,
                config=config_gs1,
                lifting=lifting_total["PSC 1"],
                capital_cost=capital_cost["PSC 1"],
                intangible_cost=intangible_cost["PSC 1"],
                opex=opex["PSC 1"],
                asr_cost=asr_cost["PSC 1"],
                psc="PSC 1",
            )

        if is_cost_recovery2:
            config_cr2 = _cost_recovery_config(config2)
            contract2: CostRecovery | GrossSplit = _build_cost_recovery(
                general=general,
                config=config_cr2,
                lifting=lifting_total["PSC 2"],
                capital_cost=capital_cost["PSC 2"],
                intangible_cost=intangible_cost["PSC 2"],
                opex=opex["PSC 2"],
                asr_cost=asr_cost["PSC 2"],
                psc="PSC 2",
            )
        else:
            config_gs2 = _gross_split_config(config2)
            contract2 = _build_gross_split(
                general=general,
                config=config_gs2,
                lifting=lifting_total["PSC 2"],
                capital_cost=capital_cost["PSC 2"],
                intangible_cost=intangible_cost["PSC 2"],
                opex=opex["PSC 2"],
                asr_cost=asr_cost["PSC 2"],
                psc="PSC 2",
            )

        contracts: TransitionContracts = {"PSC 1": contract1, "PSC 2": contract2}
        arguments: TransitionArguments = {
            "PSC 1": _build_contract_arguments(
                general=general,
                fiscal=fiscal,
                config=config1,
                psc="PSC 1",
                is_cost_recovery=is_cost_recovery1,
            ),
            "PSC 2": _build_contract_arguments(
                general=general,
                fiscal=fiscal,
                config=config2,
                psc="PSC 2",
                is_cost_recovery=is_cost_recovery2,
            ),
        }
        transition = Transition(
            contract1=contracts["PSC 1"],
            contract2=contracts["PSC 2"],
            argument_contract1=arguments["PSC 1"],
            argument_contract2=arguments["PSC 2"],
        )
        transition_arguments: ContractArguments = {
            "unrec_portion": fiscal.transferred_unrec_cost
        }
        self.psc = contracts
        self.psc_arguments = arguments
        self.transition = transition
        self.transition_arguments = transition_arguments
        summary["contract"] = transition
        return transition, transition_arguments, summary

    def _get_transition_contract_cr_to_cr(self) -> ActivationResult:
        data = self._data
        configs = data.psc_transition_cr_to_cr
        if configs is None:
            raise AdjustDataException("Missing CR-to-CR transition configuration")
        return self._get_transition_contract(
            config1=configs["Cost Recovery Config"],
            config2=configs["Cost Recovery Config (2)"],
            is_cost_recovery1=True,
            is_cost_recovery2=True,
        )

    def _get_transition_contract_cr_to_gs(self) -> ActivationResult:
        data = self._data
        configs = data.psc_transition_cr_to_gs
        if configs is None:
            raise AdjustDataException("Missing CR-to-GS transition configuration")
        return self._get_transition_contract(
            config1=configs["Cost Recovery Config"],
            config2=configs["Gross Split Config"],
            is_cost_recovery1=True,
            is_cost_recovery2=False,
        )

    def _get_transition_contract_gs_to_gs(self) -> ActivationResult:
        data = self._data
        configs = data.psc_transition_gs_to_gs
        if configs is None:
            raise AdjustDataException("Missing GS-to-GS transition configuration")
        return self._get_transition_contract(
            config1=configs["Gross Split Config"],
            config2=configs["Gross Split Config (2)"],
            is_cost_recovery1=False,
            is_cost_recovery2=False,
        )

    def _get_transition_contract_gs_to_cr(self) -> ActivationResult:
        data = self._data
        configs = data.psc_transition_gs_to_cr
        if configs is None:
            raise AdjustDataException("Missing GS-to-CR transition configuration")
        return self._get_transition_contract(
            config1=configs["Gross Split Config"],
            config2=configs["Cost Recovery Config"],
            is_cost_recovery1=False,
            is_cost_recovery2=True,
        )

    def activate(self) -> ActivationResult:
        """
        Activate the Production Sharing Contract based on its type.

        Returns
        -------
        contract: BaseProject | CostRecovery | GrossSplit | Transition
            Object representing the activated contract.

        Notes
        -----
        This method activates the Production Sharing Contract (PSC) based on
        the specified contract type. It selects the appropriate method to
        retrieve details for a single contract or a transition between contracts.

        (1) If the contract type is "Project", the method returns objects to run
            a single BaseProject contract case.
        (2) If the contract type is "PSC Cost Recovery (CR)," the method returns
            objects to run a single Cost Recovery contract case.
        (3) If the contract type is "PSC Gross Split (GS)," the method returns
            objects to run a single Gross Split contract case.
        (4) If the contract type is "Transition CR - CR," the method returns
            objects to run a transition from Cost Recovery to Cost Recovery case.
        (5) If the contract type is "Transition CR - GS," the method returns
            objects to run a transition from Cost Recovery to Gross Split case.
        (6) If the contract type is "Transition GS - CR," the method returns
            objects to run a transition from Gross Split to Cost Recovery case.
        (7) If the contract type is "Transition GS - GS," the method returns
            objects to run a transition from Gross Split to Gross Split case.
        """
        if self.contract_type == "Project":
            return self._get_single_contract_project()

        elif self.contract_type == "PSC Cost Recovery (CR)":
            return self._get_single_contract_cr()

        elif self.contract_type == "PSC Gross Split (GS)":
            return self._get_single_contract_gs()

        elif self.contract_type == "Transition CR - CR":
            return self._get_transition_contract_cr_to_cr()

        elif self.contract_type == "Transition CR - GS":
            return self._get_transition_contract_cr_to_gs()

        elif self.contract_type == "Transition GS - GS":
            return self._get_transition_contract_gs_to_gs()

        elif self.contract_type == "Transition GS - CR":
            return self._get_transition_contract_gs_to_cr()

        else:
            raise AdjustDataException(
                f"The filled contract type ({self.contract_type}) is not available. "
                + "The available contract types are: "
                + "(1) Project, "
                + "(2) PSC Cost Recovery (CR), "
                + "(3) PSC Gross Split (GS), "
                + "(4) Transition CR - CR, "
                + "(5) Transition CR - GS, "
                + "(6) Transition GS - GS, "
                + "(7) Transition GS - CR."
            )
