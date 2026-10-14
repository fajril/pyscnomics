from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt

from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.summary_profile import PostRunSummaryProfile
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.gross_split import GrossSplit
from pyscnomics.contracts.transition import Transition
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.selection import NPVSelection, DiscountingMode
from pyscnomics.econ.indicator import (
    irr,
    npv_nominal_terms,
    npv_real_terms,
    npv_skk_nominal_terms,
    npv_skk_real_terms,
    npv_point_forward,
    pot_psc,
)
from pyscnomics.input_validation import InputValidationError, engine_input_issue
from pyscnomics.validation_rules import reference_year_violation


class SummaryException(Exception):
    """Exception to raise for a misuse of Summary Method"""

    pass


class SummaryInputError(  # pyright: ignore[reportUnsafeMultipleInheritance]
    SummaryException, InputValidationError
):
    """Verified user-input summary error retaining ``SummaryException``."""

    _input_code: str = "SUMMARY_INPUT_INVALID"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        InputValidationError.__init__(
            self,
            (engine_input_issue(code=code or self._input_code, message=message),),
        )


FloatArray = npt.NDArray[np.float64]
YearArray = npt.NDArray[np.int64]
SummaryValue = float | np.float64
SummaryResult = dict[str, SummaryValue]


def _sum_float(value: FloatArray | None) -> np.float64:
    """Perform a numeric profile reduction."""
    if value is None:
        raise RuntimeError("numeric profile is unavailable before the contract is run")
    return np.sum(value)


def _as_float(value: float | np.float64) -> float:
    """Normalize NumPy scalar output to a Python float."""
    return float(value)


def _last_float(values: FloatArray | None) -> np.float64:
    """Return the final value of a completed numeric profile."""
    if values is None:
        raise SummaryException(
            "numeric profile is unavailable before the contract is run"
        )
    return _sum_float(values[-1:])


def _divide_float(
    numerator: float | np.float64,
    denominator: float | np.float64,
    where: bool = True,
) -> float | np.float64:
    """Return the zero-safe scalar ratio used by legacy summary output."""
    return numerator / denominator if where else 0.0


def _is_nonzero(value: float | np.float64) -> bool:
    """Convert NumPy scalar comparison to the bool expected by ``np.divide``."""
    return bool(value != 0)


def _require_array(value: FloatArray | None, name: str) -> FloatArray:
    if value is None:
        raise SummaryException(f"{name} is unavailable before the contract is run")
    return value


def _require_array_map(
    value: dict[str, FloatArray] | None, name: str
) -> dict[str, FloatArray]:
    if value is None:
        raise SummaryException(f"{name} is unavailable before the contract is run")
    return value


def _require_lifting(value: Lifting | None, name: str) -> Lifting:
    if value is None:
        raise SummaryException(f"{name} is unavailable before the contract is run")
    return value


def _summary_year_bounds(
    contract: BaseProject | CostRecovery | GrossSplit | Transition,
) -> tuple[int, int]:
    if isinstance(contract, Transition):
        return (
            contract.contract1.start_date.year,
            contract.contract2.end_date.year,
        )
    return contract.start_date.year, contract.end_date.year


def get_summary(
    contract: BaseProject | CostRecovery | GrossSplit | Transition,
    reference_year: int | None = None,
    inflation_rate: float = 0,
    discount_rate: float = 0.1,
    npv_mode: NPVSelection = NPVSelection.NPV_SKK_REAL_TERMS,
    discounting_mode: DiscountingMode = DiscountingMode.END_YEAR,
    profitability_discounted: bool = False,
) -> SummaryResult:

    # Condition when the reference year is less than project start date year
    contract_start_object, contract_end_object = _summary_year_bounds(contract)

    # Condition when the reference year is None
    if reference_year is None:
        reference_year = contract_start_object

    direction = reference_year_violation(
        reference_year, contract_start_object, contract_end_object
    )
    if direction == "before":
        raise SummaryInputError(
            f"The Discounting Reference Year {reference_year} "
            + f"is before the project years: {contract_start_object}",
            code="SUMMARY_REFERENCE_YEAR_OUTSIDE_PROJECT",
        )

    # Condition when the reference year is after than project end date year
    if direction == "after":
        raise SummaryInputError(
            f"The Discounting Reference Year {reference_year} "
            + f"is after the project years: {contract_end_object}",
            code="SUMMARY_REFERENCE_YEAR_OUTSIDE_PROJECT",
        )

    if not isinstance(contract, Transition):
        return contract.get_summary(
            discount_rate=discount_rate,
            npv_mode=npv_mode,
            discounting_mode=discounting_mode,
            discount_rate_start_year=reference_year,
            inflation_rate=inflation_rate,
            profitability_discounted=profitability_discounted,
        )

    contract_view = contract.get_post_run_summary_profile()
    oil_lifting = _require_lifting(contract_view.oil_lifting, "_oil_lifting")
    gas_lifting = _require_lifting(contract_view.gas_lifting, "_gas_lifting")

    # Defining the same summary parameters for any contract
    # Lifting
    lifting_oil = _sum_float(oil_lifting.get_lifting_rate_ghv_arr())
    if _sum_float(oil_lifting.get_lifting_rate_ghv_arr()) == 0:
        oil_wap = 0.0
    else:
        oil_wap = _divide_float(
            _sum_float(_require_array(contract_view.oil_revenue, "_oil_revenue")),
            _sum_float(oil_lifting.get_lifting_rate_ghv_arr()),
            where=_is_nonzero(_sum_float(oil_lifting.get_lifting_rate_ghv_arr())),
        )

    lifting_gas = _sum_float(gas_lifting.get_lifting_rate_ghv_arr())
    if _sum_float(gas_lifting.get_lifting_rate_ghv_arr()) == 0:
        gas_wap = 0.0
    else:
        gas_wap = _divide_float(
            _sum_float(
                _require_array(contract_view.gas_wap_price, "_gas_wap_price")
                * gas_lifting.get_lifting_rate_ghv_arr()
                * gas_lifting.get_ghv_arr()
            ),
            _sum_float(
                gas_lifting.get_lifting_rate_ghv_arr() * gas_lifting.get_ghv_arr()
            ),
            where=_is_nonzero(
                _sum_float(
                    gas_lifting.get_lifting_rate_ghv_arr() * gas_lifting.get_ghv_arr()
                )
            ),
        )

    # Gross Revenue
    gross_revenue_oil = _sum_float(
        _require_array(contract_view.oil_revenue, "_oil_revenue")
    )
    gross_revenue_gas = _sum_float(
        _require_array(contract_view.gas_revenue, "_gas_revenue")
    )
    gross_revenue = gross_revenue_oil + gross_revenue_gas

    # Sunk Cost
    oil_sunk_cost = (
        _require_array_map(contract_view.oil_sunk_cost_bulk, "_oil_sunk_cost_bulk")[
            "Tangible"
        ]
        + _require_array_map(contract_view.oil_sunk_cost_bulk, "_oil_sunk_cost_bulk")[
            "Intangible"
        ]
        + _require_array_map(
            contract_view.oil_preonstream_cost_bulk, "_oil_preonstream_cost_bulk"
        )["Tangible"]
        + _require_array_map(
            contract_view.oil_preonstream_cost_bulk, "_oil_preonstream_cost_bulk"
        )["Intangible"]
    )

    gas_sunk_cost = (
        _require_array_map(contract_view.gas_sunk_cost_bulk, "_gas_sunk_cost_bulk")[
            "Tangible"
        ]
        + _require_array_map(contract_view.gas_sunk_cost_bulk, "_gas_sunk_cost_bulk")[
            "Intangible"
        ]
        + _require_array_map(
            contract_view.gas_preonstream_cost_bulk, "_gas_preonstream_cost_bulk"
        )["Tangible"]
        + _require_array_map(
            contract_view.gas_preonstream_cost_bulk, "_gas_preonstream_cost_bulk"
        )["Intangible"]
    )

    sunk_cost = _sum_float(oil_sunk_cost + gas_sunk_cost)

    # Investment (Capital Cost)
    tangible = _sum_float(
        _require_array(
            contract_view.oil_capital_expenditures_post_tax,
            "_oil_capital_expenditures_post_tax",
        )
        + _require_array(
            contract_view.gas_capital_expenditures_post_tax,
            "_gas_capital_expenditures_post_tax",
        )
        + _require_array_map(contract_view.oil_sunk_cost_array, "_oil_sunk_cost_array")[
            "Tangible"
        ]
        + _require_array_map(contract_view.gas_sunk_cost_array, "_gas_sunk_cost_array")[
            "Tangible"
        ]
        + _require_array_map(
            contract_view.oil_preonstream_cost_array,
            "_oil_preonstream_cost_array",
        )["Tangible"]
        + _require_array_map(
            contract_view.gas_preonstream_cost_array,
            "_gas_preonstream_cost_array",
        )["Tangible"],
    )

    intangible = _sum_float(
        _require_array(
            contract_view.oil_intangible_expenditures_post_tax,
            "_oil_intangible_expenditures_post_tax",
        )
        + _require_array(
            contract_view.gas_intangible_expenditures_post_tax,
            "_gas_intangible_expenditures_post_tax",
        )
        + _require_array_map(contract_view.oil_sunk_cost_array, "_oil_sunk_cost_array")[
            "Intangible"
        ]
        + _require_array_map(contract_view.gas_sunk_cost_array, "_gas_sunk_cost_array")[
            "Intangible"
        ]
        + _require_array_map(
            contract_view.oil_preonstream_cost_array,
            "_oil_preonstream_cost_array",
        )["Intangible"]
        + _require_array_map(
            contract_view.gas_preonstream_cost_array,
            "_gas_preonstream_cost_array",
        )["Intangible"],
    )

    investment = tangible + intangible

    # Indirect Taxes
    oil_indirect_taxes = _sum_float(
        _require_array(contract_view.oil_total_indirect_tax, "_oil_total_indirect_tax")
    )
    gas_indirect_taxes = _sum_float(
        _require_array(contract_view.gas_total_indirect_tax, "_gas_total_indirect_tax")
    )
    total_indirect_taxes = oil_indirect_taxes + gas_indirect_taxes

    # Carry Forward Depreciation
    oil_carry_forward_depreciation = _sum_float(
        _require_array(
            contract_view.oil_carry_forward_depreciation,
            "_oil_carry_forward_depreciation",
        )
    )
    gas_carry_forward_depreciation = _sum_float(
        _require_array(
            contract_view.gas_carry_forward_depreciation,
            "_gas_carry_forward_depreciation",
        )
    )
    total_carry_forward_depreciation = (
        oil_carry_forward_depreciation + gas_carry_forward_depreciation
    )

    # Undepreciated Asset
    undepreciated_asset_oil = _sum_float(
        _require_array(
            _require_array(
                contract_view.oil_undepreciated_asset, "_oil_undepreciated_asset"
            ),
            "_oil_undepreciated_asset",
        )
    )
    undepreciated_asset_gas = _sum_float(
        _require_array(
            _require_array(
                contract_view.gas_undepreciated_asset, "_gas_undepreciated_asset"
            ),
            "_gas_undepreciated_asset",
        )
    )
    undepreciated_asset_total = undepreciated_asset_oil + undepreciated_asset_gas

    # Capex
    oil_capex = _sum_float(
        _require_array(
            contract_view.oil_capital_expenditures_post_tax,
            "_oil_capital_expenditures_post_tax",
        )
    )
    gas_capex = _sum_float(
        _require_array(
            contract_view.gas_capital_expenditures_post_tax,
            "_gas_capital_expenditures_post_tax",
        )
    )

    # OPEX and ASR (Non-Capital Cost)
    opex = _sum_float(
        _require_array(
            contract_view.oil_opex_expenditures_post_tax,
            "_oil_opex_expenditures_post_tax",
        )
        + _require_array(
            contract_view.gas_opex_expenditures_post_tax,
            "_gas_opex_expenditures_post_tax",
        ),
    )
    asr = _sum_float(
        _require_array(
            contract_view.oil_asr_expenditures_post_tax,
            "_oil_asr_expenditures_post_tax",
        )
        + _require_array(
            contract_view.gas_asr_expenditures_post_tax,
            "_gas_asr_expenditures_post_tax",
        ),
    )
    lbt = _sum_float(
        _require_array(
            contract_view.oil_lbt_expenditures_post_tax,
            "_oil_lbt_expenditures_post_tax",
        )
        + _require_array(
            contract_view.gas_lbt_expenditures_post_tax,
            "_gas_lbt_expenditures_post_tax",
        ),
    )

    # Cashflow sunk cost
    if sunk_cost == 0:
        cashflow_sunk_cost_pooled = _require_array(
            _require_array(
                contract_view.consolidated_cashflow, "_consolidated_cashflow"
            ),
            "_consolidated_cashflow",
        )
    else:
        cashflow_sunk_cost_pooled = np.concatenate(
            (
                np.array([-sunk_cost]),
                _require_array(
                    _require_array(
                        contract_view.consolidated_cashflow, "_consolidated_cashflow"
                    ),
                    "_consolidated_cashflow",
                )[
                    len(
                        _require_array(
                            contract_view.consolidated_sunk_cost,
                            "_consolidated_sunk_cost",
                        )
                    ) :
                ],
            )
        )

    # Years sunk cost pooled
    years_sunk_cost_pooled = contract_view.project_years[
        (len(contract_view.project_years) - len(cashflow_sunk_cost_pooled)) :
    ]

    # Government DDMO
    gov_ddmo = _sum_float(
        _require_array(contract_view.consolidated_ddmo, "_consolidated_ddmo")
    )

    # Government Take Income
    gov_tax_income = _sum_float(
        _require_array(
            _require_array(
                contract_view.consolidated_tax_payment, "_consolidated_tax_payment"
            ),
            "_consolidated_tax_payment",
        )
    )

    # Government Take
    gov_take = _sum_float(
        _require_array(
            _require_array(
                contract_view.consolidated_government_take,
                "_consolidated_government_take",
            ),
            "_consolidated_government_take",
        )
    )

    # Government Share
    gov_take_over_gross_rev = _divide_float(
        gov_take, gross_revenue, where=_is_nonzero(gross_revenue)
    )

    # Contractor IRR
    ctr_irr = irr(
        cashflow=_require_array(
            _require_array(
                contract_view.consolidated_cashflow, "_consolidated_cashflow"
            ),
            "_consolidated_cashflow",
        )
    )
    ctr_irr_sunk_cost_pooled = irr(cashflow=cashflow_sunk_cost_pooled)

    # Calculation related to NPV Calculation which are depends on the NPV Mode
    # NPV Calculation for SKK Real Terms
    if npv_mode == NPVSelection.NPV_SKK_REAL_TERMS:
        # Contractor NPV
        ctr_npv = npv_skk_real_terms(
            cashflow=_require_array(
                _require_array(
                    contract_view.consolidated_cashflow, "_consolidated_cashflow"
                ),
                "_consolidated_cashflow",
            ),
            cashflow_years=contract_view.project_years,
            discount_rate=discount_rate,
            reference_year=reference_year,
            discounting_mode=discounting_mode,
        )

        # Contractor NPV when the sunk cost is being pooled on the first year
        ctr_npv_sunk_cost_pooled = npv_skk_real_terms(
            cashflow=cashflow_sunk_cost_pooled,
            cashflow_years=years_sunk_cost_pooled,
            discount_rate=discount_rate,
            reference_year=reference_year,
            discounting_mode=discounting_mode,
        )

        # Contractor Investment NPV
        investment_npv = npv_skk_real_terms(
            cashflow=(
                _require_array(
                    contract_view.oil_capital_expenditures_post_tax,
                    "_oil_capital_expenditures_post_tax",
                )
                + _require_array(
                    contract_view.gas_capital_expenditures_post_tax,
                    "_gas_capital_expenditures_post_tax",
                )
                + _require_array(
                    contract_view.oil_intangible_expenditures_post_tax,
                    "_oil_intangible_expenditures_post_tax",
                )
                + _require_array(
                    contract_view.gas_intangible_expenditures_post_tax,
                    "_gas_intangible_expenditures_post_tax",
                )
                + _require_array_map(
                    contract_view.oil_sunk_cost_array, "_oil_sunk_cost_array"
                )["Tangible"]
                + _require_array_map(
                    contract_view.gas_sunk_cost_array, "_gas_sunk_cost_array"
                )["Tangible"]
                + _require_array_map(
                    contract_view.oil_preonstream_cost_array,
                    "_oil_preonstream_cost_array",
                )["Tangible"]
                + _require_array_map(
                    contract_view.gas_preonstream_cost_array,
                    "_gas_preonstream_cost_array",
                )["Tangible"]
                + _require_array_map(
                    contract_view.oil_sunk_cost_array, "_oil_sunk_cost_array"
                )["Intangible"]
                + _require_array_map(
                    contract_view.gas_sunk_cost_array, "_gas_sunk_cost_array"
                )["Intangible"]
                + _require_array_map(
                    contract_view.oil_preonstream_cost_array,
                    "_oil_preonstream_cost_array",
                )["Intangible"]
                + _require_array_map(
                    contract_view.gas_preonstream_cost_array,
                    "_gas_preonstream_cost_array",
                )["Intangible"]
            ),
            cashflow_years=contract_view.project_years,
            discount_rate=discount_rate,
            reference_year=reference_year,
            discounting_mode=discounting_mode,
        )

        # Government Take Net Present Value
        gov_take_npv = npv_skk_real_terms(
            cashflow=_require_array(
                contract_view.consolidated_government_take,
                "_consolidated_government_take",
            ),
            cashflow_years=contract_view.project_years,
            discount_rate=discount_rate,
            reference_year=reference_year,
            discounting_mode=discounting_mode,
        )

    # NPV Calculation for SKK Nominal Terms
    elif npv_mode == NPVSelection.NPV_SKK_NOMINAL_TERMS:
        # Contractor NPV
        ctr_npv = npv_skk_nominal_terms(
            cashflow=_require_array(
                _require_array(
                    contract_view.consolidated_cashflow, "_consolidated_cashflow"
                ),
                "_consolidated_cashflow",
            ),
            cashflow_years=contract_view.project_years,
            discount_rate=discount_rate,
            discounting_mode=discounting_mode,
        )

        # Contractor NPV when the sunk cost is being pooled on the first year
        ctr_npv_sunk_cost_pooled = npv_skk_nominal_terms(
            cashflow=cashflow_sunk_cost_pooled,
            cashflow_years=years_sunk_cost_pooled,
            discount_rate=discount_rate,
            discounting_mode=discounting_mode,
        )

        # Contractor Investment NPV
        investment_npv = npv_skk_nominal_terms(
            cashflow=(
                _require_array(
                    contract_view.oil_capital_expenditures_post_tax,
                    "_oil_capital_expenditures_post_tax",
                )
                + _require_array(
                    contract_view.gas_capital_expenditures_post_tax,
                    "_gas_capital_expenditures_post_tax",
                )
                + _require_array(
                    contract_view.oil_intangible_expenditures_post_tax,
                    "_oil_intangible_expenditures_post_tax",
                )
                + _require_array(
                    contract_view.gas_intangible_expenditures_post_tax,
                    "_gas_intangible_expenditures_post_tax",
                )
                + _require_array_map(
                    contract_view.oil_sunk_cost_array, "_oil_sunk_cost_array"
                )["Tangible"]
                + _require_array_map(
                    contract_view.gas_sunk_cost_array, "_gas_sunk_cost_array"
                )["Tangible"]
                + _require_array_map(
                    contract_view.oil_preonstream_cost_array,
                    "_oil_preonstream_cost_array",
                )["Tangible"]
                + _require_array_map(
                    contract_view.gas_preonstream_cost_array,
                    "_gas_preonstream_cost_array",
                )["Tangible"]
                + _require_array_map(
                    contract_view.oil_sunk_cost_array, "_oil_sunk_cost_array"
                )["Intangible"]
                + _require_array_map(
                    contract_view.gas_sunk_cost_array, "_gas_sunk_cost_array"
                )["Intangible"]
                + _require_array_map(
                    contract_view.oil_preonstream_cost_array,
                    "_oil_preonstream_cost_array",
                )["Intangible"]
                + _require_array_map(
                    contract_view.gas_preonstream_cost_array,
                    "_gas_preonstream_cost_array",
                )["Intangible"]
            ),
            cashflow_years=contract_view.project_years,
            discount_rate=discount_rate,
            discounting_mode=discounting_mode,
        )

        # Government Take Net Present Value
        gov_take_npv = npv_skk_nominal_terms(
            cashflow=_require_array(
                contract_view.consolidated_government_take,
                "_consolidated_government_take",
            ),
            cashflow_years=contract_view.project_years,
            discount_rate=discount_rate,
            discounting_mode=discounting_mode,
        )

    # NPV Calculation for Nominal Terms
    elif npv_mode == NPVSelection.NPV_NOMINAL_TERMS:
        # Contractor NPV
        ctr_npv = npv_nominal_terms(
            cashflow=_require_array(
                _require_array(
                    contract_view.consolidated_cashflow, "_consolidated_cashflow"
                ),
                "_consolidated_cashflow",
            ),
            cashflow_years=contract_view.project_years,
            discount_rate=discount_rate,
            reference_year=reference_year,
            discounting_mode=discounting_mode,
        )

        # Contractor NPV when the sunk cost is being pooled on the first year
        ctr_npv_sunk_cost_pooled = npv_nominal_terms(
            cashflow=cashflow_sunk_cost_pooled,
            cashflow_years=years_sunk_cost_pooled,
            discount_rate=discount_rate,
            reference_year=reference_year,
            discounting_mode=discounting_mode,
        )

        # Contractor Investment NPV
        investment_npv = npv_nominal_terms(
            cashflow=(
                _require_array(
                    contract_view.oil_capital_expenditures_post_tax,
                    "_oil_capital_expenditures_post_tax",
                )
                + _require_array(
                    contract_view.gas_capital_expenditures_post_tax,
                    "_gas_capital_expenditures_post_tax",
                )
                + _require_array(
                    contract_view.oil_intangible_expenditures_post_tax,
                    "_oil_intangible_expenditures_post_tax",
                )
                + _require_array(
                    contract_view.gas_intangible_expenditures_post_tax,
                    "_gas_intangible_expenditures_post_tax",
                )
                + _require_array_map(
                    contract_view.oil_sunk_cost_array, "_oil_sunk_cost_array"
                )["Tangible"]
                + _require_array_map(
                    contract_view.gas_sunk_cost_array, "_gas_sunk_cost_array"
                )["Tangible"]
                + _require_array_map(
                    contract_view.oil_preonstream_cost_array,
                    "_oil_preonstream_cost_array",
                )["Tangible"]
                + _require_array_map(
                    contract_view.gas_preonstream_cost_array,
                    "_gas_preonstream_cost_array",
                )["Tangible"]
                + _require_array_map(
                    contract_view.oil_sunk_cost_array, "_oil_sunk_cost_array"
                )["Intangible"]
                + _require_array_map(
                    contract_view.gas_sunk_cost_array, "_gas_sunk_cost_array"
                )["Intangible"]
                + _require_array_map(
                    contract_view.oil_preonstream_cost_array,
                    "_oil_preonstream_cost_array",
                )["Intangible"]
                + _require_array_map(
                    contract_view.gas_preonstream_cost_array,
                    "_gas_preonstream_cost_array",
                )["Intangible"]
            ),
            cashflow_years=contract_view.project_years,
            discount_rate=discount_rate,
            reference_year=reference_year,
            discounting_mode=discounting_mode,
        )

        # Government Take Net Present Value
        gov_take_npv = npv_nominal_terms(
            cashflow=_require_array(
                contract_view.consolidated_government_take,
                "_consolidated_government_take",
            ),
            cashflow_years=contract_view.project_years,
            discount_rate=discount_rate,
            reference_year=reference_year,
            discounting_mode=discounting_mode,
        )

    # NPV Calculation for Real Terms
    elif npv_mode == NPVSelection.NPV_REAL_TERMS:
        # Contractor NPV
        ctr_npv = npv_real_terms(
            cashflow=_require_array(
                _require_array(
                    contract_view.consolidated_cashflow, "_consolidated_cashflow"
                ),
                "_consolidated_cashflow",
            ),
            cashflow_years=contract_view.project_years,
            discount_rate=discount_rate,
            reference_year=reference_year,
            inflation_rate=inflation_rate,
            discounting_mode=discounting_mode,
        )

        # Contractor NPV when the sunk cost is being pooled on the first year
        ctr_npv_sunk_cost_pooled = npv_real_terms(
            cashflow=cashflow_sunk_cost_pooled,
            cashflow_years=years_sunk_cost_pooled,
            discount_rate=discount_rate,
            reference_year=reference_year,
            inflation_rate=inflation_rate,
            discounting_mode=discounting_mode,
        )

        # Contractor Investment NPV
        investment_npv = npv_real_terms(
            cashflow=(
                _require_array(
                    contract_view.oil_capital_expenditures_post_tax,
                    "_oil_capital_expenditures_post_tax",
                )
                + _require_array(
                    contract_view.gas_capital_expenditures_post_tax,
                    "_gas_capital_expenditures_post_tax",
                )
                + _require_array(
                    contract_view.oil_intangible_expenditures_post_tax,
                    "_oil_intangible_expenditures_post_tax",
                )
                + _require_array(
                    contract_view.gas_intangible_expenditures_post_tax,
                    "_gas_intangible_expenditures_post_tax",
                )
                + _require_array_map(
                    contract_view.oil_sunk_cost_array, "_oil_sunk_cost_array"
                )["Tangible"]
                + _require_array_map(
                    contract_view.gas_sunk_cost_array, "_gas_sunk_cost_array"
                )["Tangible"]
                + _require_array_map(
                    contract_view.oil_preonstream_cost_array,
                    "_oil_preonstream_cost_array",
                )["Tangible"]
                + _require_array_map(
                    contract_view.gas_preonstream_cost_array,
                    "_gas_preonstream_cost_array",
                )["Tangible"]
                + _require_array_map(
                    contract_view.oil_sunk_cost_array, "_oil_sunk_cost_array"
                )["Intangible"]
                + _require_array_map(
                    contract_view.gas_sunk_cost_array, "_gas_sunk_cost_array"
                )["Intangible"]
                + _require_array_map(
                    contract_view.oil_preonstream_cost_array,
                    "_oil_preonstream_cost_array",
                )["Intangible"]
                + _require_array_map(
                    contract_view.gas_preonstream_cost_array,
                    "_gas_preonstream_cost_array",
                )["Intangible"]
            ),
            cashflow_years=contract_view.project_years,
            discount_rate=discount_rate,
            reference_year=reference_year,
            inflation_rate=inflation_rate,
            discounting_mode=discounting_mode,
        )

        # Government Take Net Present Value
        gov_take_npv = npv_real_terms(
            cashflow=_require_array(
                contract_view.consolidated_government_take,
                "_consolidated_government_take",
            ),
            cashflow_years=contract_view.project_years,
            discount_rate=discount_rate,
            reference_year=reference_year,
            inflation_rate=inflation_rate,
            discounting_mode=discounting_mode,
        )

    # NPV Calculation for Point Forwards
    else:
        # Contractor NPV
        ctr_npv = npv_point_forward(
            cashflow=_require_array(
                _require_array(
                    contract_view.consolidated_cashflow, "_consolidated_cashflow"
                ),
                "_consolidated_cashflow",
            ),
            cashflow_years=contract_view.project_years,
            discount_rate=discount_rate,
            reference_year=reference_year,
            discounting_mode=discounting_mode,
        )

        # Contractor NPV when the sunk cost is being pooled on the first year
        ctr_npv_sunk_cost_pooled = npv_point_forward(
            cashflow=cashflow_sunk_cost_pooled,
            cashflow_years=years_sunk_cost_pooled,
            discount_rate=discount_rate,
            reference_year=reference_year,
            discounting_mode=discounting_mode,
        )

        # Contractor Investment NPV
        investment_npv = npv_point_forward(
            cashflow=(
                _require_array(
                    contract_view.oil_capital_expenditures_post_tax,
                    "_oil_capital_expenditures_post_tax",
                )
                + _require_array(
                    contract_view.gas_capital_expenditures_post_tax,
                    "_gas_capital_expenditures_post_tax",
                )
                + _require_array(
                    contract_view.oil_intangible_expenditures_post_tax,
                    "_oil_intangible_expenditures_post_tax",
                )
                + _require_array(
                    contract_view.gas_intangible_expenditures_post_tax,
                    "_gas_intangible_expenditures_post_tax",
                )
                + _require_array_map(
                    contract_view.oil_sunk_cost_array, "_oil_sunk_cost_array"
                )["Tangible"]
                + _require_array_map(
                    contract_view.gas_sunk_cost_array, "_gas_sunk_cost_array"
                )["Tangible"]
                + _require_array_map(
                    contract_view.oil_preonstream_cost_array,
                    "_oil_preonstream_cost_array",
                )["Tangible"]
                + _require_array_map(
                    contract_view.gas_preonstream_cost_array,
                    "_gas_preonstream_cost_array",
                )["Tangible"]
                + _require_array_map(
                    contract_view.oil_sunk_cost_array, "_oil_sunk_cost_array"
                )["Intangible"]
                + _require_array_map(
                    contract_view.gas_sunk_cost_array, "_gas_sunk_cost_array"
                )["Intangible"]
                + _require_array_map(
                    contract_view.oil_preonstream_cost_array,
                    "_oil_preonstream_cost_array",
                )["Intangible"]
                + _require_array_map(
                    contract_view.gas_preonstream_cost_array,
                    "_gas_preonstream_cost_array",
                )["Intangible"]
            ),
            cashflow_years=contract_view.project_years,
            discount_rate=discount_rate,
            reference_year=reference_year,
            discounting_mode=discounting_mode,
        )

        # Government Take Net Present Value
        gov_take_npv = npv_point_forward(
            cashflow=_require_array(
                contract_view.consolidated_government_take,
                "_consolidated_government_take",
            ),
            cashflow_years=contract_view.project_years,
            discount_rate=discount_rate,
            reference_year=reference_year,
            discounting_mode=discounting_mode,
        )

        #  Modifying the Contractor Net Cashflow since the cashflow before the reference year is neglected.
        ref_year_arr = np.full_like(
            _require_array(
                _require_array(
                    contract_view.consolidated_cashflow, "_consolidated_cashflow"
                ),
                "_consolidated_cashflow",
            ),
            fill_value=reference_year,
        )
        cashflow_point_forward = np.where(
            contract_view.project_years >= ref_year_arr,
            _require_array(
                _require_array(
                    contract_view.consolidated_cashflow, "_consolidated_cashflow"
                ),
                "_consolidated_cashflow",
            ),
            0,
        )
        gross_revenue_point_forward = np.where(
            contract_view.project_years >= ref_year_arr,
            _require_array(contract_view.consolidated_revenue, "_consolidated_revenue"),
            0,
        )
        ctr_net_cashflow = _sum_float(cashflow_point_forward)
        gross_revenue_point_forward = _sum_float(gross_revenue_point_forward)
        ctr_net_cashflow_over_gross_rev = ctr_net_cashflow / gross_revenue_point_forward

    # Contractor Present Value ratio to the investment npv
    # Condition when the Profitability Index is calculated using discounted investment
    investment_pi: float | np.float64 = 0.0
    if profitability_discounted:
        ctr_pv_ratio = _divide_float(
            ctr_npv, investment_npv, where=_is_nonzero(investment_npv)
        )

    # Condition when the Profitability Index is calculated using un-discounted investment
    else:
        investment_pi = _sum_float(
            _require_array(
                contract_view.oil_capital_expenditures_post_tax,
                "_oil_capital_expenditures_post_tax",
            )
            + _require_array(
                contract_view.gas_capital_expenditures_post_tax,
                "_gas_capital_expenditures_post_tax",
            )
            + _require_array(
                contract_view.oil_intangible_expenditures_post_tax,
                "_oil_intangible_expenditures_post_tax",
            )
            + _require_array(
                contract_view.gas_intangible_expenditures_post_tax,
                "_gas_intangible_expenditures_post_tax",
            )
            + _require_array_map(
                contract_view.oil_sunk_cost_array, "_oil_sunk_cost_array"
            )["Tangible"]
            + _require_array_map(
                contract_view.gas_sunk_cost_array, "_gas_sunk_cost_array"
            )["Tangible"]
            + _require_array_map(
                contract_view.oil_preonstream_cost_array,
                "_oil_preonstream_cost_array",
            )["Tangible"]
            + _require_array_map(
                contract_view.gas_preonstream_cost_array,
                "_gas_preonstream_cost_array",
            )["Tangible"]
            + _require_array_map(
                contract_view.oil_sunk_cost_array, "_oil_sunk_cost_array"
            )["Intangible"]
            + _require_array_map(
                contract_view.gas_sunk_cost_array, "_gas_sunk_cost_array"
            )["Intangible"]
            + _require_array_map(
                contract_view.oil_preonstream_cost_array,
                "_oil_preonstream_cost_array",
            )["Intangible"]
            + _require_array_map(
                contract_view.gas_preonstream_cost_array,
                "_gas_preonstream_cost_array",
            )["Intangible"]
        )
        ctr_pv_ratio = _divide_float(
            ctr_npv, investment_pi, where=_is_nonzero(investment_pi)
        )

    ctr_pi = 1 + ctr_pv_ratio

    # Condition where the contract is Cost Recovery
    if isinstance(contract, CostRecovery):
        # Cost Recovery
        cost_recovery = _sum_float(
            _require_array(
                contract_view.consolidated_cost_recovery_after_tf,
                "_consolidated_cost_recovery_after_tf",
            )
        )
        cost_recovery_over_gross_rev = _divide_float(
            cost_recovery, gross_revenue, where=_is_nonzero(gross_revenue)
        )

        # Unrecoverable Cost
        unrec_cost = _last_float(
            _require_array(
                contract_view.consolidated_unrecovered_after_transfer,
                "_consolidated_unrecovered_after_transfer",
            )
        )
        unrec_over_costrec = _divide_float(
            unrec_cost, cost_recovery, where=_is_nonzero(cost_recovery)
        )
        unrec_over_gross_rev = _divide_float(
            unrec_cost, gross_revenue, where=_is_nonzero(gross_revenue)
        )

        # Gross Share of Contractor and Government
        ctr_gross_share = _sum_float(
            _require_array(
                contract_view.consolidated_contractor_share,
                "_consolidated_contractor_share",
            )
        )
        gov_gross_share = _sum_float(
            _require_array(
                contract_view.consolidated_government_share,
                "_consolidated_government_share",
            )
        )

        # Contractor Net Share
        ctr_net_share = _sum_float(
            _require_array(
                _require_array(
                    contract_view.consolidated_ctr_net_share,
                    "_consolidated_ctr_net_share",
                ),
                "_consolidated_ctr_net_share",
            )
        )
        ctr_net_share_over_gross_share = _divide_float(
            ctr_net_share, gross_revenue, where=_is_nonzero(gross_revenue)
        )

        # Contractor Net Cashflow
        ctr_net_cashflow = _sum_float(
            _require_array(
                _require_array(
                    contract_view.consolidated_cashflow, "_consolidated_cashflow"
                ),
                "_consolidated_cashflow",
            )
        )
        ctr_net_cashflow_over_gross_rev = _divide_float(
            ctr_net_cashflow, gross_revenue, where=_is_nonzero(gross_revenue)
        )

        # Government FTP Share
        gov_ftp_share = _sum_float(
            _require_array(contract_view.consolidated_ftp_gov, "_consolidated_ftp_gov")
        )

        # Government Equity Share
        gov_equity_share = _sum_float(
            _require_array(
                contract_view.consolidated_government_share,
                "_consolidated_government_share",
            )
        )

        # Contractor POT
        ctr_pot = pot_psc(
            cashflow=_require_array(
                _require_array(
                    contract_view.consolidated_cashflow, "_consolidated_cashflow"
                ),
                "_consolidated_cashflow",
            ),
            cashflow_years=contract_view.project_years,
            reference_year=reference_year,
        )

        # Parsing the obtained variable into a dictionary, Summary
        return {
            "lifting_oil": lifting_oil,
            "oil_wap": oil_wap,
            "lifting_gas": lifting_gas,
            "gas_wap": gas_wap,
            "gross_revenue": gross_revenue,
            "gross_revenue_oil": gross_revenue_oil,
            "gross_revenue_gas": gross_revenue_gas,
            "ctr_gross_share": ctr_gross_share,
            "gov_gross_share": gov_gross_share,
            "investment": investment,
            "oil_capex": oil_capex,
            "gas_capex": gas_capex,
            "sunk_cost": sunk_cost,
            "tangible": tangible,
            "intangible": intangible,
            "opex_asr_lbt": opex + asr + lbt,
            "opex": opex,
            "asr": asr,
            "lbt": lbt,
            "cost_recovery / deductible_cost": cost_recovery,
            "cost_recovery_over_gross_rev": cost_recovery_over_gross_rev,
            "unrec_cost": unrec_cost,
            "unrec_over_costrec": unrec_over_costrec,
            "unrec_over_gross_rev": unrec_over_gross_rev,
            "ctr_net_share": ctr_net_share,
            "ctr_net_share_over_gross_share": ctr_net_share_over_gross_share,
            "ctr_net_cashflow": ctr_net_cashflow,
            "ctr_net_cashflow_over_gross_rev": ctr_net_cashflow_over_gross_rev,
            "ctr_npv": ctr_npv,
            "ctr_npv_sunk_cost_pooled": ctr_npv_sunk_cost_pooled,
            "ctr_irr": ctr_irr,
            "ctr_irr_sunk_cost_pooled": ctr_irr_sunk_cost_pooled,
            "ctr_pot": ctr_pot,
            "ctr_pv_ratio": ctr_pv_ratio,
            "ctr_pi": ctr_pi,
            "gov_ftp_share": gov_ftp_share,
            "gov_equity_share": gov_equity_share,
            "gov_ddmo": gov_ddmo,
            "gov_tax_income": gov_tax_income,
            "gov_take": gov_take,
            "gov_take_over_gross_rev": gov_take_over_gross_rev,
            "gov_take_npv": gov_take_npv,
            "undepreciated_asset_oil": undepreciated_asset_oil,
            "undepreciated_asset_gas": undepreciated_asset_gas,
            "undepreciated_asset_total": undepreciated_asset_total,
            "total_indirect_taxes": total_indirect_taxes,
            "oil_indirect_taxes": oil_indirect_taxes,
            "gas_indirect_taxes": gas_indirect_taxes,
            "total_carry_forward_depreciation": total_carry_forward_depreciation,
            "oil_carry_forward_depreciation": oil_carry_forward_depreciation,
            "gas_carry_forward_depreciation": gas_carry_forward_depreciation,
        }

    # Condition where the contract is Gross Split
    if isinstance(contract, GrossSplit):
        #  Deductible Cost
        deductible_cost = _sum_float(
            _require_array(
                contract_view.consolidated_deductible_cost,
                "_consolidated_deductible_cost",
            )
        )
        deductible_cost_over_gross_rev = _divide_float(
            deductible_cost, gross_revenue, where=_is_nonzero(gross_revenue)
        )

        # Carry Forward Cost
        carry_forward_deductible_cost = _last_float(
            contract_view.consolidated_carward_deduct_cost
        )
        carry_forcost_over_gross_share = _divide_float(
            carry_forward_deductible_cost,
            gross_revenue,
            where=_is_nonzero(gross_revenue),
        )
        carry_forcost_over_deductible_cost = _divide_float(
            carry_forward_deductible_cost,
            deductible_cost,
            where=_is_nonzero(deductible_cost),
        )

        # CTR Gross Share
        ctr_gross_share = _sum_float(
            _require_array(
                contract_view.consolidated_ctr_share_before_tf,
                "_consolidated_ctr_share_before_tf",
            )
        )

        # GOV GOV Gross Share
        gov_gross_share = _sum_float(
            _require_array(
                contract_view.consolidated_gov_share_before_tf,
                "_consolidated_gov_share_before_tf",
            )
        )

        # CTR Net Share
        ctr_net_share = _sum_float(
            _require_array(
                _require_array(
                    contract_view.consolidated_ctr_net_share,
                    "_consolidated_ctr_net_share",
                ),
                "_consolidated_ctr_net_share",
            )
        )
        ctr_net_share_over_gross_share = _divide_float(
            ctr_net_share, gross_revenue, where=_is_nonzero(gross_revenue)
        )

        # CTR Net Cashflow
        ctr_net_cashflow = _sum_float(
            _require_array(
                _require_array(
                    contract_view.consolidated_cashflow, "_consolidated_cashflow"
                ),
                "_consolidated_cashflow",
            )
        )
        ctr_net_cashflow_over_gross_rev = _divide_float(
            ctr_net_cashflow, gross_revenue, where=_is_nonzero(gross_revenue)
        )

        # Contractor IRR
        ctr_irr = irr(
            cashflow=_require_array(
                _require_array(
                    contract_view.consolidated_cashflow, "_consolidated_cashflow"
                ),
                "_consolidated_cashflow",
            )
        )

        # Contractor POT
        ctr_pot = pot_psc(
            cashflow=_require_array(
                _require_array(
                    contract_view.consolidated_cashflow, "_consolidated_cashflow"
                ),
                "_consolidated_cashflow",
            ),
            cashflow_years=contract_view.project_years,
            reference_year=reference_year,
        )

        # Parsing the obtained variable into a dictionary, Summary
        return {
            "lifting_oil": lifting_oil,
            "oil_wap": oil_wap,
            "lifting_gas": lifting_gas,
            "gas_wap": gas_wap,
            "gross_revenue": gross_revenue,
            "gross_revenue_oil": gross_revenue_oil,
            "gross_revenue_gas": gross_revenue_gas,
            "ctr_gross_share": ctr_gross_share,
            "gov_gross_share": gov_gross_share,
            "investment": investment,
            "oil_capex": oil_capex,
            "gas_capex": gas_capex,
            "sunk_cost": sunk_cost,
            "tangible": tangible,
            "intangible": intangible,
            "opex_asr_lbt": opex + asr + lbt,
            "opex": opex,
            "asr": asr,
            "lbt": lbt,
            "cost_recovery / deductible_cost": deductible_cost,
            "cost_recovery_over_gross_rev": deductible_cost_over_gross_rev,
            "unrec_cost": carry_forward_deductible_cost,
            "unrec_over_costrec": carry_forcost_over_deductible_cost,
            "unrec_over_gross_rev": carry_forcost_over_gross_share,
            "ctr_net_share": ctr_net_share,
            "ctr_net_share_over_gross_share": ctr_net_share_over_gross_share,
            "ctr_net_cashflow": ctr_net_cashflow,
            "ctr_net_cashflow_over_gross_rev": ctr_net_cashflow_over_gross_rev,
            "ctr_npv": ctr_npv,
            "ctr_npv_sunk_cost_pooled": ctr_npv_sunk_cost_pooled,
            "ctr_irr": ctr_irr,
            "ctr_irr_sunk_cost_pooled": ctr_irr_sunk_cost_pooled,
            "ctr_pot": ctr_pot,
            "ctr_pv_ratio": ctr_pv_ratio,
            "ctr_pi": ctr_pi,
            "gov_ddmo": gov_ddmo,
            "gov_tax_income": gov_tax_income,
            "gov_take": gov_take,
            "gov_take_over_gross_rev": gov_take_over_gross_rev,
            "gov_take_npv": gov_take_npv,
            "gov_ftp_share": 0,
            "undepreciated_asset_oil": undepreciated_asset_oil,
            "undepreciated_asset_gas": undepreciated_asset_gas,
            "undepreciated_asset_total": undepreciated_asset_total,
            "total_indirect_taxes": total_indirect_taxes,
            "oil_indirect_taxes": oil_indirect_taxes,
            "gas_indirect_taxes": gas_indirect_taxes,
            "total_carry_forward_depreciation": total_carry_forward_depreciation,
            "oil_carry_forward_depreciation": oil_carry_forward_depreciation,
            "gas_carry_forward_depreciation": gas_carry_forward_depreciation,
        }

    transition = contract
    contract1_view = transition.contract1.get_post_run_summary_profile()
    contract2_view = transition.contract2.get_post_run_summary_profile()
    ctr_ftp = _sum_float(_require_array(contract_view.ctr_ftp, "_ctr_ftp"))
    gov_ftp = _sum_float(_require_array(contract_view.gov_ftp, "_gov_ftp"))
    ftp = ctr_ftp + gov_ftp

    deductible_cost = _sum_float(
        _require_array(contract_view.oil_deductible_cost, "_oil_deductible_cost")
        + _require_array(contract_view.gas_deductible_cost, "_gas_deductible_cost")
    )
    deductible_cost_over_grossrev = deductible_cost / gross_revenue

    carry_forward_cost = _last_float(
        _require_array(contract_view.oil_unrec_cost, "_oil_unrec_cost")
        + _require_array(contract_view.gas_unrec_cost, "_gas_unrec_cost")
    )
    carry_forward_cost_over_grossrev = carry_forward_cost / gross_revenue

    ctr_equity_share = _sum_float(
        _require_array(contract_view.oil_ctr_ets, "_oil_ctr_ets")
        + _require_array(contract_view.gas_ctr_ets, "_gas_ctr_ets")
    )
    gov_equity_share = _sum_float(
        _require_array(contract_view.oil_gov_ets, "_oil_gov_ets")
        + _require_array(contract_view.gas_gov_ets, "_gas_gov_ets")
    )
    equity_to_be_split = ctr_equity_share + gov_equity_share

    ctr_net_operating_profit = _sum_float(
        _require_array(contract_view.net_operating_profit, "_net_operating_profit")
    )
    ctr_net_operating_profit_over_grossrev = ctr_net_operating_profit / gross_revenue

    ctr_net_cash_flow = _sum_float(
        _require_array(contract_view.ctr_net_cashflow, "_ctr_net_cashflow")
    )
    ctr_net_cash_flow_over_grossrev = ctr_net_cash_flow / gross_revenue

    # Contractor POT
    ctr_pot = pot_psc(
        cashflow=_require_array(
            _require_array(
                contract_view.consolidated_cashflow, "_consolidated_cashflow"
            ),
            "_consolidated_cashflow",
        ),
        cashflow_years=contract_view.project_years,
        reference_year=reference_year,
    )

    # Contractor Gross Share
    ctr_gross_share = _sum_float(
        (
            _require_array(
                contract1_view.consolidated_ctr_share_before_tf,
                "_consolidated_ctr_share_before_tf",
            )
            if isinstance(transition.contract1, GrossSplit)
            else _require_array(
                contract1_view.consolidated_contractor_share,
                "_consolidated_contractor_share",
            )
        ),
    ) + _sum_float(
        (
            _require_array(
                contract2_view.consolidated_ctr_share_before_tf,
                "_consolidated_ctr_share_before_tf",
            )
            if isinstance(transition.contract2, GrossSplit)
            else _require_array(
                contract2_view.consolidated_contractor_share,
                "_consolidated_contractor_share",
            )
        ),
    )

    # GOV GOV Gross Share
    gov_gross_share = _sum_float(
        (
            _require_array(
                contract1_view.consolidated_gov_share_before_tf,
                "_consolidated_gov_share_before_tf",
            )
            if isinstance(transition.contract1, GrossSplit)
            else _require_array(
                contract1_view.consolidated_government_share,
                "_consolidated_government_share",
            )
        ),
    ) + _sum_float(
        (
            _require_array(
                contract2_view.consolidated_gov_share_before_tf,
                "_consolidated_gov_share_before_tf",
            )
            if isinstance(transition.contract2, GrossSplit)
            else _require_array(
                contract2_view.consolidated_government_share,
                "_consolidated_government_share",
            )
        ),
    )

    # return {'lifting_oil': lifting_oil,
    #         'oil_wap': oil_wap,
    #         'lifting_gas': lifting_gas,
    #         'gas_wap': gas_wap,
    #         'gross_revenue': gross_revenue,
    #         'gross_revenue_oil': gross_revenue_oil,
    #         'gross_revenue_gas': gross_revenue_gas,
    #         'ftp': ftp,
    #         'gov_ftp': gov_ftp,
    #         'ctr_ftp': ctr_ftp,
    #         'sunk_cost': sunk_cost,
    #         'tangible': tangible,
    #         'intangible': intangible,
    #         'opex': opex,
    #         'asr': asr,
    #         'deductible_cost': deductible_cost,
    #         'deductible_cost_over_grossrev': deductible_cost_over_grossrev,
    #         'carry_forward_cost': carry_forward_cost,
    #         'carry_forward_cost_over_grossrev': carry_forward_cost_over_grossrev,
    #         'equity_to_be_split': equity_to_be_split,
    #         'ctr_equity_share': ctr_equity_share,
    #         'gov_equity_share': gov_equity_share,
    #         'ctr_net_operating_profit': ctr_net_operating_profit,
    #         'ctr_net_operating_profit_over_grossrev': ctr_net_operating_profit_over_grossrev,
    #         'ctr_net_cash_flow': ctr_net_cash_flow,
    #         'ctr_net_cash_flow_over_grossrev': ctr_net_cash_flow_over_grossrev,
    #         'ctr_npv': ctr_npv,
    #         'ctr_npv_sunk_cost_pooled': ctr_npv_sunk_cost_pooled,
    #         'ctr_irr': ctr_irr,
    #         'ctr_irr_sunk_cost_pooled': ctr_irr_sunk_cost_pooled,
    #         'ctr_pot': ctr_pot,
    #         'ctr_pv_ratio': ctr_pv_ratio,
    #         'gov_ddmo': gov_ddmo,
    #         'gov_tax_income': gov_tax_income,
    #         'gov_take': gov_take,
    #         'gov_take_over_gross_rev': gov_take_over_gross_rev,
    #         'gov_take_npv': gov_take_npv
    #         }

    # return {'lifting_oil': lifting_oil,
    #         'oil_wap': oil_wap,
    #         'lifting_gas': lifting_gas,
    #         'gas_wap': gas_wap,
    #         'gross_revenue': gross_revenue,
    #         'gross_revenue_oil': gross_revenue_oil,
    #         'gross_revenue_gas': gross_revenue_gas,
    #         'ftp': ftp,
    #         'gov_ftp': gov_ftp,
    #         'ctr_ftp': ctr_ftp,
    #         'sunk_cost': sunk_cost,
    #         'tangible': tangible,
    #         'intangible': intangible,
    #         'opex': opex,
    #         'asr': asr,
    #         'deductible_cost': deductible_cost,
    #         'deductible_cost_over_grossrev': deductible_cost_over_grossrev,
    #         'carry_forward_cost': carry_forward_cost,
    #         'carry_forward_cost_over_grossrev': carry_forward_cost_over_grossrev,
    #         'equity_to_be_split': equity_to_be_split,
    #         'ctr_equity_share': ctr_equity_share,
    #         'gov_equity_share': gov_equity_share,
    #         'ctr_net_operating_profit': ctr_net_operating_profit,
    #         'ctr_net_operating_profit_over_grossrev': ctr_net_operating_profit_over_grossrev,
    #         'ctr_net_cash_flow': ctr_net_cash_flow,
    #         'ctr_net_cash_flow_over_grossrev': ctr_net_cash_flow_over_grossrev,
    #         'ctr_npv': ctr_npv,
    #         'ctr_npv_sunk_cost_pooled': ctr_npv_sunk_cost_pooled,
    #         'ctr_irr': ctr_irr,
    #         'ctr_irr_sunk_cost_pooled': ctr_irr_sunk_cost_pooled,
    #         'ctr_pot': ctr_pot,
    #         'ctr_pv_ratio': ctr_pv_ratio,
    #         'gov_ddmo': gov_ddmo,
    #         'gov_tax_income': gov_tax_income,
    #         'gov_take': gov_take,
    #         'gov_take_over_gross_rev': gov_take_over_gross_rev,
    #         'gov_take_npv': gov_take_npv
    #         }

    return {
        "lifting_oil": lifting_oil,
        "oil_wap": oil_wap,
        "lifting_gas": lifting_gas,
        "gas_wap": gas_wap,
        "gross_revenue": gross_revenue,
        "gross_revenue_oil": gross_revenue_oil,
        "gross_revenue_gas": gross_revenue_gas,
        "ctr_gross_share": ctr_gross_share,
        "gov_gross_share": gov_gross_share,
        "ftp": ftp,
        "gov_ftp": gov_ftp,
        "ctr_ftp": ctr_ftp,
        "sunk_cost": sunk_cost,
        "investment": investment,
        "tangible": tangible,
        "intangible": intangible,
        "opex_asr_lbt": opex + asr + lbt,
        "opex": opex,
        "asr": asr,
        "lbt": lbt,
        "cost_recovery / deductible_cost": deductible_cost,
        "cost_recovery_over_gross_rev": deductible_cost_over_grossrev,
        "unrec_cost": carry_forward_cost,
        "unrec_over_gross_rev": carry_forward_cost_over_grossrev,
        "equity_to_be_split": equity_to_be_split,
        "ctr_net_share": ctr_net_operating_profit,
        "ctr_net_share_over_gross_share": ctr_net_operating_profit_over_grossrev,
        "ctr_equity_share": ctr_equity_share,
        "gov_equity_share": gov_equity_share,
        "ctr_net_operating_profit": ctr_net_operating_profit,
        "ctr_net_operating_profit_over_grossrev": ctr_net_operating_profit_over_grossrev,
        "ctr_net_cashflow": ctr_net_cash_flow,
        "ctr_net_cashflow_over_gross_rev": ctr_net_cash_flow_over_grossrev,
        "ctr_npv": ctr_npv,
        "ctr_npv_sunk_cost_pooled": ctr_npv_sunk_cost_pooled,
        "ctr_irr": ctr_irr,
        "ctr_irr_sunk_cost_pooled": ctr_irr_sunk_cost_pooled,
        "ctr_pot": ctr_pot,
        "ctr_pv_ratio": ctr_pv_ratio,
        "ctr_pi": ctr_pi,
        "gov_ddmo": gov_ddmo,
        "gov_tax_income": gov_tax_income,
        "gov_take": gov_take,
        "gov_take_over_gross_rev": gov_take_over_gross_rev,
        "gov_take_npv": gov_take_npv,
        "gov_ftp_share": gov_ftp,
        "undepreciated_asset_oil": undepreciated_asset_oil,
        "undepreciated_asset_gas": undepreciated_asset_gas,
        "undepreciated_asset_total": undepreciated_asset_total,
        "total_indirect_taxes": total_indirect_taxes,
        "oil_indirect_taxes": oil_indirect_taxes,
        "gas_indirect_taxes": gas_indirect_taxes,
    }


@dataclass
class Summary:
    """
    Dataclass to represent the economic indicator of a contract.

    Parameters
    ----------
    contract: BaseProject | CostRecovery | GrossSplit | Transition
        The contract which the summary of the economic indicator will be retrieved.
    reference_year: int
        The reference year for calculation of the economic indicator of the contract.
    inflation_rate: float
        The inflation rate used to calculate the economic indicator of the contract.
    discount_rate: float
        The discount rate used to calculate the Net Present Value (NPV).
    npv_mode: NPVSelection
        The calculation method to calculate the NPV related indicator.
    discounting_mode: DiscountingMode
        The discounting mode used in NPV calculation.
    profitability_discounted: bool
        The method used to calculate the Profitability Index.

    """

    contract: BaseProject | CostRecovery | GrossSplit | Transition
    _profile: PostRunSummaryProfile = field(init=False, repr=False)
    reference_year: int | None = field(default=None)
    inflation_rate: float = field(default=0.0)
    discount_rate: float = field(default=0.1)
    npv_mode: NPVSelection = field(default=NPVSelection.NPV_SKK_REAL_TERMS)
    discounting_mode: DiscountingMode = field(default=DiscountingMode.END_YEAR)
    profitability_discounted: bool = True

    # Attributes that will be defined later
    lifting_oil: float = field(default=0.0, init=False, repr=False)
    oil_wap: float | np.float64 = field(default=0.0, init=False, repr=False)
    lifting_gas: float = field(default=0.0, init=False, repr=False)
    gas_wap: float | np.float64 = field(default=0.0, init=False, repr=False)
    gross_revenue: float = field(default=0.0, init=False, repr=False)
    gross_revenue_oil: float = field(default=0.0, init=False, repr=False)
    gross_revenue_gas: float = field(default=0.0, init=False, repr=False)
    ctr_gross_share: float = field(default=0.0, init=False, repr=False)
    gov_gross_share: float = field(default=0.0, init=False, repr=False)
    investment: float = field(default=0.0, init=False, repr=False)
    oil_capex: float = field(default=0.0, init=False, repr=False)
    gas_capex: float = field(default=0.0, init=False, repr=False)
    sunk_cost: float = field(default=0.0, init=False, repr=False)
    capital_cost: float = field(default=0.0, init=False, repr=False)
    intangible: float = field(default=0.0, init=False, repr=False)
    opex_and_asr: float = field(default=0.0, init=False, repr=False)
    opex: float = field(default=0.0, init=False, repr=False)
    asr: float | np.float64 = field(default=0.0, init=False, repr=False)
    cost_recovery_deductible_cost: float = field(default=0.0, init=False, repr=False)
    cost_recovery_over_gross_rev: float = field(default=0.0, init=False, repr=False)
    unrec_cost: float | np.float64 = field(default=0.0, init=False, repr=False)
    unrec_over_costrec: float = field(default=0.0, init=False, repr=False)
    unrec_over_gross_rev: float = field(default=0.0, init=False, repr=False)
    ctr_net_share: float = field(default=0.0, init=False, repr=False)
    ctr_net_share_over_gross_share: float = field(default=0.0, init=False, repr=False)
    ctr_net_cashflow: float | np.float64 = field(default=0.0, init=False, repr=False)
    ctr_net_cashflow_over_gross_rev: float = field(default=0.0, init=False, repr=False)
    ctr_npv: float = field(default=0.0, init=False, repr=False)
    ctr_npv_sunk_cost_pooled: float = field(default=0.0, init=False, repr=False)
    ctr_irr: float = field(default=0.0, init=False, repr=False)
    ctr_irr_sunk_cost_pooled: float = field(default=0.0, init=False, repr=False)
    ctr_pot: float = field(default=0.0, init=False, repr=False)
    ctr_pv_ratio: float | np.float64 = field(default=0.0, init=False, repr=False)
    ctr_pi: float | np.float64 = field(default=0.0, init=False, repr=False)
    gov_ftp_share: float = field(default=0.0, init=False, repr=False)
    ctr_equity_share: float = field(default=0.0, init=False, repr=False)
    gov_equity_share: float = field(default=0.0, init=False, repr=False)
    gov_ddmo: float = field(default=0.0, init=False, repr=False)
    gov_tax_income: float = field(default=0.0, init=False, repr=False)
    gov_take: float = field(default=0.0, init=False, repr=False)
    gov_take_over_gross_rev: float = field(default=0.0, init=False, repr=False)
    gov_take_npv: float = field(default=0.0, init=False, repr=False)

    # Transition reporting residuals read from the post-run profile
    lbt: float | np.float64 = field(default=0.0, init=False, repr=False)
    oil_indirect_taxes: float | np.float64 = field(default=0.0, init=False, repr=False)
    gas_indirect_taxes: float | np.float64 = field(default=0.0, init=False, repr=False)
    total_indirect_taxes: float | np.float64 = field(default=0.0, init=False, repr=False)
    undepreciated_asset_oil: float = field(default=0.0, init=False, repr=False)
    undepreciated_asset_gas: float = field(default=0.0, init=False, repr=False)
    undepreciated_asset_total: float = field(default=0.0, init=False, repr=False)
    oil_carry_forward_depreciation: float = field(default=0.0, init=False, repr=False)
    gas_carry_forward_depreciation: float = field(default=0.0, init=False, repr=False)
    total_carry_forward_depreciation: float = field(
        default=0.0, init=False, repr=False
    )

    investment_npv: float | np.float64 = field(default=0.0, init=False, repr=False)
    investment_pi: float | np.float64 = field(default=0.0, init=False, repr=False)
    cost_recovery: float | np.float64 = field(default=0.0, init=False, repr=False)
    deductible_cost: float | np.float64 = field(default=0.0, init=False, repr=False)
    deductible_cost_over_gross_rev: float | np.float64 = field(
        default=0.0, init=False, repr=False
    )
    deductible_cost_over_grossrev: float | np.float64 = field(
        default=0.0, init=False, repr=False
    )
    carry_forward_deductible_cost: float | np.float64 = field(
        default=0.0, init=False, repr=False
    )
    carry_forcost_over_gross_share: float | np.float64 = field(
        default=0.0, init=False, repr=False
    )
    carry_forcost_over_deductible_cost: float | np.float64 = field(
        default=0.0, init=False, repr=False
    )
    carry_forward_cost: float | np.float64 = field(default=0.0, init=False, repr=False)
    carry_forward_cost_over_grossrev: float | np.float64 = field(
        default=0.0, init=False, repr=False
    )
    ftp: float | np.float64 = field(default=0.0, init=False, repr=False)
    ctr_ftp: float | np.float64 = field(default=0.0, init=False, repr=False)
    gov_ftp: float | np.float64 = field(default=0.0, init=False, repr=False)
    ctr_net_operating_profit: float | np.float64 = field(
        default=0.0, init=False, repr=False
    )
    ctr_net_operating_profit_over_grossrev: float | np.float64 = field(
        default=0.0, init=False, repr=False
    )
    ctr_net_cash_flow: float | np.float64 = field(default=0.0, init=False, repr=False)
    ctr_net_cash_flow_over_grossrev: float | np.float64 = field(
        default=0.0, init=False, repr=False
    )
    equity_to_be_split: float | np.float64 = field(default=0.0, init=False, repr=False)

    sunk_cost_extended: FloatArray | None = field(default=None, init=False, repr=False)
    cashflow_sunk_cost_pooled: FloatArray | None = field(
        default=None, init=False, repr=False
    )
    years_sunk_cost_pooled: YearArray | None = field(
        default=None, init=False, repr=False
    )

    def _require_sunk_cost_extended(self) -> FloatArray:
        if self.sunk_cost_extended is None:
            raise SummaryException("Summary has not been run")
        return self.sunk_cost_extended

    def _require_cashflow_sunk_cost_pooled(self) -> FloatArray:
        if self.cashflow_sunk_cost_pooled is None:
            raise SummaryException("Summary has not been run")
        return self.cashflow_sunk_cost_pooled

    def _require_years_sunk_cost_pooled(self) -> YearArray:
        if self.years_sunk_cost_pooled is None:
            raise SummaryException("Summary has not been run")
        return self.years_sunk_cost_pooled

    def _require_reference_year(self) -> int:
        if self.reference_year is None:
            raise SummaryException("Summary reference year is not initialized")
        return self.reference_year

    def __post_init__(self) -> None:
        # Condition when the reference year is less than project start date year
        self._profile = self.contract.get_post_run_summary_profile()
        contract_start_object, contract_end_object = _summary_year_bounds(self.contract)
        if self.reference_year is None:
            self.reference_year = contract_start_object

        direction = reference_year_violation(
            self.reference_year, contract_start_object, contract_end_object
        )
        if direction == "before":
            raise SummaryInputError(
                f"The Discounting Reference Year {self.reference_year} "
                + f"is before the project years: {contract_start_object}",
                code="SUMMARY_REFERENCE_YEAR_OUTSIDE_PROJECT",
            )

        # Condition when the reference year is after than project end date year
        if direction == "after":
            raise SummaryInputError(
                f"The Discounting Reference Year {self.reference_year} "
                + f"is after the project years: {contract_end_object}",
                code="SUMMARY_REFERENCE_YEAR_OUTSIDE_PROJECT",
            )

    def _get_sum_lifting(self):
        """
        Function to retrieve the lifting indicator into the corresponding class attributes.
        Returns
        -------
        Parsing the lifting data to the lifting attributes
        """
        # Lifting Oil
        oil_lifting = _require_lifting(self._profile.oil_lifting, "_oil_lifting")
        gas_lifting = _require_lifting(self._profile.gas_lifting, "_gas_lifting")
        self.lifting_oil = float(_sum_float(oil_lifting.get_lifting_rate_ghv_arr()))
        if _sum_float(oil_lifting.get_lifting_rate_ghv_arr()) == 0:
            self.oil_wap = 0.0
        else:
            self.oil_wap = _divide_float(
                _sum_float(_require_array(self._profile.oil_revenue, "_oil_revenue")),
                _sum_float(oil_lifting.get_lifting_rate_ghv_arr()),
                where=_is_nonzero(_sum_float(oil_lifting.get_lifting_rate_ghv_arr())),
            )

        self.lifting_gas = float(_sum_float(gas_lifting.get_lifting_rate_ghv_arr()))

        # Lifting Gas
        if _sum_float(gas_lifting.get_lifting_rate_ghv_arr()) == 0:
            self.gas_wap = 0.0
        else:
            self.gas_wap = _divide_float(
                _sum_float(
                    _require_array(self._profile.gas_wap_price, "_gas_wap_price")
                    * gas_lifting.get_lifting_rate_ghv_arr()
                ),
                _sum_float(gas_lifting.get_lifting_rate_ghv_arr()),
                where=_is_nonzero(_sum_float(gas_lifting.get_lifting_rate_ghv_arr())),
            )

    def _get_gross_revenue(self):
        """
        Function to retrieve the gross revenue indicator into the corresponding class attributes.

        Returns
        -------
        Parsing the gross revenue data to the corresponding attributes
        """
        # Gross Revenue
        self.gross_revenue_oil = float(
            _sum_float(_require_array(self._profile.oil_revenue, "_oil_revenue"))
        )
        self.gross_revenue_gas = float(
            _sum_float(_require_array(self._profile.gas_revenue, "_gas_revenue"))
        )
        self.gross_revenue = self.gross_revenue_oil + self.gross_revenue_gas

    def _get_sunk_cost(self):
        """
        Function to retrieve the sunk cost indicator into the corresponding class attributes.

        Returns
        -------
        Parsing the sunk cost data to the lifting attributes

        """
        # Sunk Cost
        self.sunk_cost = float(
            _sum_float(
                _require_array(self._profile.oil_sunk_cost, "_oil_sunk_cost")
                + _require_array(self._profile.gas_sunk_cost, "_gas_sunk_cost")
            )
        )
        self.sunk_cost_extended = np.concatenate(
            (
                _require_array(
                    _require_array(
                        self._profile.consolidated_sunk_cost, "_consolidated_sunk_cost"
                    ),
                    "_consolidated_sunk_cost",
                ),
                np.zeros(
                    len(self._profile.project_years)
                    - len(
                        _require_array(
                            self._profile.consolidated_sunk_cost,
                            "_consolidated_sunk_cost",
                        )
                    )
                ),
            )
        )

    def _get_capital_cost(self):
        """
        Function to retrieve the capital cost indicator into the corresponding class attributes.

        Returns
        -------
        Parsing the lifting data to the corresponding attributes
        """
        # Investment Capital Cost
        self.capital_cost = float(
            _sum_float(
                _require_array(
                    self._profile.oil_capital_expenditures_post_tax,
                    "_oil_capital_expenditures_post_tax",
                )
                + _require_array(
                    self._profile.gas_capital_expenditures_post_tax,
                    "_gas_capital_expenditures_post_tax",
                )
            )
        )

    def _get_intangible_cost(self):
        """
        Function to retrieve the intangible cost indicator into the corresponding class attributes.

        Returns
        -------
        Parsing the capital cost data to the lifting attributes
        """
        # Investment Intangible Cost
        self.intangible = float(
            _sum_float(
                _require_array(
                    self._profile.oil_intangible_expenditures_post_tax,
                    "_oil_intangible_expenditures_post_tax",
                )
                + _require_array(
                    self._profile.gas_intangible_expenditures_post_tax,
                    "_gas_intangible_expenditures_post_tax",
                )
                - self._require_sunk_cost_extended(),
            )
        )

    def _get_investment(self):
        """
        Function to retrieve the investment indicator into the corresponding class attributes.

        Returns
        -------
        Parsing the investment data to the corresponding attributes

        """
        # Investment total
        self.investment = self.capital_cost + self.intangible

    def _get_capex(self):
        """
        Function to retrieve the capital expenditures indicator into the corresponding class attributes.

        Returns
        -------
        Parsing the capital expenditures data to the corresponding attributes

        """
        # Capex
        self.oil_capex = float(
            _sum_float(
                _require_array(
                    self._profile.oil_capital_expenditures_post_tax,
                    "_oil_capital_expenditures_post_tax",
                )
            )
        )
        self.gas_capex = float(
            _sum_float(
                _require_array(
                    self._profile.gas_capital_expenditures_post_tax,
                    "_gas_capital_expenditures_post_tax",
                )
            )
        )

    def _get_opex(self):
        """
        Function to retrieve the OPEX indicator into the corresponding class attributes.

        Returns
        -------
        Parsing the opex data to the corresponding attributes

        """
        # OPEX
        self.opex = float(
            _sum_float(
                _require_array(
                    self._profile.oil_opex_expenditures_post_tax,
                    "_oil_opex_expenditures_post_tax",
                )
                + _require_array(
                    self._profile.gas_opex_expenditures_post_tax,
                    "_gas_opex_expenditures_post_tax",
                ),
            )
        )

    def _get_asr(self):
        """
        Function to retrieve the ASR indicator into the corresponding class attributes.

        Returns
        -------
        Parsing the ASR data to the corresponding attributes

        """
        self.asr = float(
            _sum_float(
                _require_array(
                    self._profile.oil_asr_expenditures_post_tax,
                    "_oil_asr_expenditures_post_tax",
                )
                + _require_array(
                    self._profile.gas_asr_expenditures_post_tax,
                    "_gas_asr_expenditures_post_tax",
                ),
            )
        )

    def _get_cashflow_sunkcost_pooled(self):
        """
        Function to retrieve the Cash Flow if the sunk cost is pooled indicator into the corresponding class attributes.

        Returns
        -------
        Parsing the Cash Flow if the sunk cost is pooled data to the corresponding attributes

        """
        # Cashflow sunk cost
        if self.sunk_cost == 0:
            self.cashflow_sunk_cost_pooled = _require_array(
                _require_array(
                    self._profile.consolidated_cashflow, "_consolidated_cashflow"
                ),
                "_consolidated_cashflow",
            )
        else:
            self.cashflow_sunk_cost_pooled = np.concatenate(
                (
                    np.array([-self.sunk_cost]),
                    _require_array(
                        _require_array(
                            self._profile.consolidated_cashflow,
                            "_consolidated_cashflow",
                        ),
                        "_consolidated_cashflow",
                    )[
                        len(
                            _require_array(
                                self._profile.consolidated_sunk_cost,
                                "_consolidated_sunk_cost",
                            )
                        ) :
                    ],
                )
            )

        # Years sunk cost pooled
        self.years_sunk_cost_pooled = self._profile.project_years[
            (
                len(self._profile.project_years)
                - len(self._require_cashflow_sunk_cost_pooled())
            ) :
        ]

    def _get_ddmo(self):
        """
        Function to retrieve the DDMO indicator into the corresponding class attributes.

        Returns
        -------
        Parsing the DDMO data to the corresponding attributes

        """
        # Government DDMO
        if (
            isinstance(self.contract, CostRecovery)
            or isinstance(self.contract, GrossSplit)
            or isinstance(self.contract, Transition)
        ):
            self.gov_ddmo = float(
                _sum_float(
                    _require_array(
                        _require_array(
                            self._profile.consolidated_ddmo, "_consolidated_ddmo"
                        ),
                        "_consolidated_ddmo",
                    )
                )
            )
        else:
            self.gov_ddmo = 0.0

    def _get_gov_tax_income(self):
        """
        Function to retrieve the Government Tax indicator into the corresponding class attributes.

        Returns
        -------
        Parsing the Government Tax data to the corresponding attributes

        """
        # Government Tax Income
        if (
            isinstance(self.contract, CostRecovery)
            or isinstance(self.contract, GrossSplit)
            or isinstance(self.contract, Transition)
        ):
            self.gov_tax_income = float(
                _sum_float(
                    _require_array(
                        self._profile.consolidated_tax_payment,
                        "_consolidated_tax_payment",
                    )
                )
            )
        else:
            self.gov_tax_income = 0.0

    def _get_gov_take(self):
        """
        Function to retrieve the Government Take indicator into the corresponding class attributes.

        Returns
        -------
        Parsing the Government Take data to the corresponding attributes.

        """
        # Government Take and Government Take over Gross Share
        if (
            isinstance(self.contract, CostRecovery)
            or isinstance(self.contract, GrossSplit)
            or isinstance(self.contract, Transition)
        ):
            self.gov_take = float(
                _sum_float(
                    _require_array(
                        self._profile.consolidated_government_take,
                        "_consolidated_government_take",
                    )
                )
            )
            self.gov_take_over_gross_rev = self.gov_take / self.gross_revenue
        else:
            self.gov_take = 0.0
            self.gov_take_over_gross_rev = 0.0

    def _get_contractor_irr(self):
        """
        Function to retrieve the Contractor Take indicator into the corresponding class attributes.

        Returns
        -------
        Parsing the Government Take data to the corresponding attributes

        """
        self.ctr_irr = irr(
            cashflow=_require_array(
                _require_array(
                    self._profile.consolidated_cashflow, "_consolidated_cashflow"
                ),
                "_consolidated_cashflow",
            )
        )
        self.ctr_irr_sunk_cost_pooled = irr(
            cashflow=self._require_cashflow_sunk_cost_pooled()
        )

    def _npv_skk_real_terms(self):
        """
        Function to retrieve the NPV related indicator into the corresponding class attributes.
        The method used within this module is SKK Real Terms.

        Returns
        -------
        Parsing the NPV Related data to the corresponding attributes.

        """
        # Contractor NPV
        self.ctr_npv = npv_skk_real_terms(
            cashflow=_require_array(
                _require_array(
                    self._profile.consolidated_cashflow, "_consolidated_cashflow"
                ),
                "_consolidated_cashflow",
            ),
            cashflow_years=self._profile.project_years,
            discount_rate=self.discount_rate,
            reference_year=self._require_reference_year(),
            discounting_mode=self.discounting_mode,
        )

        # Contractor NPV when the sunk cost is being pooled on the first year
        self.ctr_npv_sunk_cost_pooled = npv_skk_real_terms(
            cashflow=self._require_cashflow_sunk_cost_pooled(),
            cashflow_years=self._require_years_sunk_cost_pooled(),
            discount_rate=self.discount_rate,
            reference_year=self._require_reference_year(),
            discounting_mode=self.discounting_mode,
        )

        # Contractor Investment NPV
        self.investment_npv = npv_skk_real_terms(
            cashflow=(
                _require_array(
                    self._profile.oil_capital_expenditures_post_tax,
                    "_oil_capital_expenditures_post_tax",
                )
                + _require_array(
                    self._profile.gas_capital_expenditures_post_tax,
                    "_gas_capital_expenditures_post_tax",
                )
                + _require_array(
                    self._profile.oil_intangible_expenditures_post_tax,
                    "_oil_intangible_expenditures_post_tax",
                )
                + _require_array(
                    self._profile.gas_intangible_expenditures_post_tax,
                    "_gas_intangible_expenditures_post_tax",
                )
                - self._require_sunk_cost_extended()
            ),
            cashflow_years=self._profile.project_years,
            discount_rate=self.discount_rate,
            reference_year=self._require_reference_year(),
            discounting_mode=self.discounting_mode,
        )

        # Government Take Net Present Value
        self.gov_take_npv = npv_skk_real_terms(
            cashflow=_require_array(
                self._profile.consolidated_government_take,
                "_consolidated_government_take",
            ),
            cashflow_years=self._profile.project_years,
            discount_rate=self.discount_rate,
            reference_year=self._require_reference_year(),
            discounting_mode=self.discounting_mode,
        )

    def _npv_skk_nominal_terms(self):
        """
        Function to retrieve the NPV related indicator into the corresponding class attributes.
        The method used within this module is SKK Nominal Terms.

        Returns
        -------
        Parsing the NPV Related data to the corresponding attributes.

        """
        # Contractor NPV
        self.ctr_npv = npv_skk_nominal_terms(
            cashflow=_require_array(
                _require_array(
                    self._profile.consolidated_cashflow, "_consolidated_cashflow"
                ),
                "_consolidated_cashflow",
            ),
            cashflow_years=self._profile.project_years,
            discount_rate=self.discount_rate,
            discounting_mode=self.discounting_mode,
        )

        # Contractor NPV when the sunk cost is being pooled on the first year
        self.ctr_npv_sunk_cost_pooled = npv_skk_nominal_terms(
            cashflow=self._require_cashflow_sunk_cost_pooled(),
            cashflow_years=self._require_years_sunk_cost_pooled(),
            discount_rate=self.discount_rate,
            discounting_mode=self.discounting_mode,
        )

        # Contractor Investment NPV
        self.investment_npv = npv_skk_nominal_terms(
            cashflow=(
                _require_array(
                    self._profile.oil_capital_expenditures_post_tax,
                    "_oil_capital_expenditures_post_tax",
                )
                + _require_array(
                    self._profile.gas_capital_expenditures_post_tax,
                    "_gas_capital_expenditures_post_tax",
                )
                + _require_array(
                    self._profile.oil_intangible_expenditures_post_tax,
                    "_oil_intangible_expenditures_post_tax",
                )
                + _require_array(
                    self._profile.gas_intangible_expenditures_post_tax,
                    "_gas_intangible_expenditures_post_tax",
                )
                - self._require_sunk_cost_extended()
            ),
            cashflow_years=self._profile.project_years,
            discount_rate=self.discount_rate,
            discounting_mode=self.discounting_mode,
        )

        # Government Take Net Present Value
        self.gov_take_npv = npv_skk_nominal_terms(
            cashflow=_require_array(
                self._profile.consolidated_government_take,
                "_consolidated_government_take",
            ),
            cashflow_years=self._profile.project_years,
            discount_rate=self.discount_rate,
            discounting_mode=self.discounting_mode,
        )

    def _npv_nominal_terms(self):
        """
        Function to retrieve the NPV related indicator into the corresponding class attributes.
        The method used within this module is Nominal Terms.

        Returns
        -------
        Parsing the NPV Related data to the corresponding attributes.

        """
        # Contractor NPV
        self.ctr_npv = npv_nominal_terms(
            cashflow=_require_array(
                _require_array(
                    self._profile.consolidated_cashflow, "_consolidated_cashflow"
                ),
                "_consolidated_cashflow",
            ),
            cashflow_years=self._profile.project_years,
            discount_rate=self.discount_rate,
            reference_year=self._require_reference_year(),
            discounting_mode=self.discounting_mode,
        )

        # Contractor NPV when the sunk cost is being pooled on the first year
        self.ctr_npv_sunk_cost_pooled = npv_nominal_terms(
            cashflow=self._require_cashflow_sunk_cost_pooled(),
            cashflow_years=self._require_years_sunk_cost_pooled(),
            discount_rate=self.discount_rate,
            reference_year=self._require_reference_year(),
            discounting_mode=self.discounting_mode,
        )

        # Contractor Investment NPV
        self.investment_npv = npv_nominal_terms(
            cashflow=(
                _require_array(
                    self._profile.oil_capital_expenditures_post_tax,
                    "_oil_capital_expenditures_post_tax",
                )
                + _require_array(
                    self._profile.gas_capital_expenditures_post_tax,
                    "_gas_capital_expenditures_post_tax",
                )
                + _require_array(
                    self._profile.oil_intangible_expenditures_post_tax,
                    "_oil_intangible_expenditures_post_tax",
                )
                + _require_array(
                    self._profile.gas_intangible_expenditures_post_tax,
                    "_gas_intangible_expenditures_post_tax",
                )
                - self._require_sunk_cost_extended()
            ),
            cashflow_years=self._profile.project_years,
            discount_rate=self.discount_rate,
            reference_year=self._require_reference_year(),
            discounting_mode=self.discounting_mode,
        )

        # Government Take Net Present Value
        self.gov_take_npv = npv_nominal_terms(
            cashflow=_require_array(
                self._profile.consolidated_government_take,
                "_consolidated_government_take",
            ),
            cashflow_years=self._profile.project_years,
            discount_rate=self.discount_rate,
            reference_year=self._require_reference_year(),
            discounting_mode=self.discounting_mode,
        )

    def _npv_real_terms(self):
        """
        Function to retrieve the NPV related indicator into the corresponding class attributes.
        The method used within this module is Real Terms.

        Returns
        -------
        Parsing the NPV Related data to the corresponding attributes.

        """
        # Contractor NPV
        self.ctr_npv = npv_real_terms(
            cashflow=_require_array(
                _require_array(
                    self._profile.consolidated_cashflow, "_consolidated_cashflow"
                ),
                "_consolidated_cashflow",
            ),
            cashflow_years=self._profile.project_years,
            discount_rate=self.discount_rate,
            reference_year=self._require_reference_year(),
            inflation_rate=self.inflation_rate,
            discounting_mode=self.discounting_mode,
        )

        # Contractor NPV when the sunk cost is being pooled on the first year
        self.ctr_npv_sunk_cost_pooled = npv_real_terms(
            cashflow=self._require_cashflow_sunk_cost_pooled(),
            cashflow_years=self._require_years_sunk_cost_pooled(),
            discount_rate=self.discount_rate,
            reference_year=self._require_reference_year(),
            inflation_rate=self.inflation_rate,
            discounting_mode=self.discounting_mode,
        )

        # Contractor Investment NPV
        self.investment_npv = npv_real_terms(
            cashflow=(
                _require_array(
                    self._profile.oil_capital_expenditures_post_tax,
                    "_oil_capital_expenditures_post_tax",
                )
                + _require_array(
                    self._profile.gas_capital_expenditures_post_tax,
                    "_gas_capital_expenditures_post_tax",
                )
                + _require_array(
                    self._profile.oil_intangible_expenditures_post_tax,
                    "_oil_intangible_expenditures_post_tax",
                )
                + _require_array(
                    self._profile.gas_intangible_expenditures_post_tax,
                    "_gas_intangible_expenditures_post_tax",
                )
                - self._require_sunk_cost_extended()
            ),
            cashflow_years=self._profile.project_years,
            discount_rate=self.discount_rate,
            reference_year=self._require_reference_year(),
            inflation_rate=self.inflation_rate,
            discounting_mode=self.discounting_mode,
        )

        # Government Take Net Present Value
        self.gov_take_npv = npv_real_terms(
            cashflow=_require_array(
                self._profile.consolidated_government_take,
                "_consolidated_government_take",
            ),
            cashflow_years=self._profile.project_years,
            discount_rate=self.discount_rate,
            reference_year=self._require_reference_year(),
            inflation_rate=self.inflation_rate,
            discounting_mode=self.discounting_mode,
        )

    def _npv_pointforward_terms(self):
        """
        Function to retrieve the NPV related indicator into the corresponding class attributes.
        The method used within this module is Point Forward.

        Returns
        -------
        Parsing the NPV Related data to the corresponding attributes.

        """
        # Contractor NPV
        self.ctr_npv = npv_point_forward(
            cashflow=_require_array(
                _require_array(
                    self._profile.consolidated_cashflow, "_consolidated_cashflow"
                ),
                "_consolidated_cashflow",
            ),
            cashflow_years=self._profile.project_years,
            discount_rate=self.discount_rate,
            reference_year=self._require_reference_year(),
            discounting_mode=self.discounting_mode,
        )

        # Contractor NPV when the sunk cost is being pooled on the first year
        self.ctr_npv_sunk_cost_pooled = npv_point_forward(
            cashflow=self._require_cashflow_sunk_cost_pooled(),
            cashflow_years=self._require_years_sunk_cost_pooled(),
            discount_rate=self.discount_rate,
            reference_year=self._require_reference_year(),
            discounting_mode=self.discounting_mode,
        )

        # Contractor Investment NPV
        self.investment_npv = npv_point_forward(
            cashflow=(
                _require_array(
                    self._profile.oil_capital_expenditures_post_tax,
                    "_oil_capital_expenditures_post_tax",
                )
                + _require_array(
                    self._profile.gas_capital_expenditures_post_tax,
                    "_gas_capital_expenditures_post_tax",
                )
                + _require_array(
                    self._profile.oil_intangible_expenditures_post_tax,
                    "_oil_intangible_expenditures_post_tax",
                )
                + _require_array(
                    self._profile.gas_intangible_expenditures_post_tax,
                    "_gas_intangible_expenditures_post_tax",
                )
                - self._require_sunk_cost_extended()
            ),
            cashflow_years=self._profile.project_years,
            discount_rate=self.discount_rate,
            reference_year=self._require_reference_year(),
            discounting_mode=self.discounting_mode,
        )

        # Government Take Net Present Value
        self.gov_take_npv = npv_point_forward(
            cashflow=_require_array(
                self._profile.consolidated_government_take,
                "_consolidated_government_take",
            ),
            cashflow_years=self._profile.project_years,
            discount_rate=self.discount_rate,
            reference_year=self._require_reference_year(),
            discounting_mode=self.discounting_mode,
        )

        #  Modifying the Contractor Net Cashflow since the cashflow before the reference year is neglected.
        ref_year_arr = np.full_like(
            _require_array(
                _require_array(
                    self._profile.consolidated_cashflow, "_consolidated_cashflow"
                ),
                "_consolidated_cashflow",
            ),
            fill_value=self.reference_year,
        )

        cashflow_point_forward = np.where(
            self._profile.project_years >= ref_year_arr,
            _require_array(
                _require_array(
                    self._profile.consolidated_cashflow, "_consolidated_cashflow"
                ),
                "_consolidated_cashflow",
            ),
            0,
        )

        gross_revenue_point_forward = np.where(
            self._profile.project_years >= ref_year_arr,
            _require_array(self._profile.consolidated_revenue, "_consolidated_revenue"),
            0,
        )

        self.ctr_net_cashflow = float(_sum_float(cashflow_point_forward))
        gross_revenue_point_forward = _sum_float(gross_revenue_point_forward)
        self.ctr_net_cashflow_over_gross_rev = _as_float(
            self.ctr_net_cashflow / gross_revenue_point_forward
        )

    def _get_npv_indicator(self):
        """
        Function to utilized as wrapper of NPV related indicator.

        Returns
        -------
        Wrapper of the NPV Related data to the corresponding attributes.

        """
        if self.npv_mode == NPVSelection.NPV_SKK_REAL_TERMS:
            self._npv_skk_real_terms()
        elif self.npv_mode == NPVSelection.NPV_SKK_NOMINAL_TERMS:
            self._npv_skk_real_terms()
        elif self.npv_mode == NPVSelection.NPV_NOMINAL_TERMS:
            self._npv_nominal_terms()
        elif self.npv_mode == NPVSelection.NPV_REAL_TERMS:
            self._npv_real_terms()
        elif self.npv_mode == NPVSelection.NPV_POINT_FORWARD:
            self._npv_pointforward_terms()
        else:
            raise SummaryInputError(
                f"The NPV Method {self.npv_mode} , is not recognized",
                code="SUMMARY_NPV_MODE_UNRECOGNIZED",
            )

    def _get_profitability_index(self):
        """
        Function to retrieve the Profitability Index into the corresponding class attributes.

        Returns
        -------
        Parsing the Profitability data to the corresponding attributes.

        """
        if self.profitability_discounted:
            self.ctr_pv_ratio = _divide_float(
                self.ctr_npv, self.investment_npv, where=self.investment_npv != 0
            )

        # Condition when the Profitability Index is calculated using un-discounted investment
        else:
            self.investment_pi = _sum_float(
                _require_array(
                    self._profile.oil_capital_expenditures_post_tax,
                    "_oil_capital_expenditures_post_tax",
                )
                + _require_array(
                    self._profile.gas_capital_expenditures_post_tax,
                    "_gas_capital_expenditures_post_tax",
                )
                + _require_array(
                    self._profile.oil_intangible_expenditures_post_tax,
                    "_oil_intangible_expenditures_post_tax",
                )
                + _require_array(
                    self._profile.gas_intangible_expenditures_post_tax,
                    "_gas_intangible_expenditures_post_tax",
                )
                - self._require_sunk_cost_extended()
            )
            self.ctr_pv_ratio = _divide_float(
                self.ctr_npv,
                self.investment_pi,
                where=_is_nonzero(self.investment_pi),
            )

        self.ctr_pi = 1 + self.ctr_pv_ratio

    def _get_costrec_deductible_cost(self):
        """
        Function to retrieve the Cost Recovery or Deductible Cost indicator into the corresponding class attributes.

        Returns
        -------
        Parsing the Cost Recovery or Deductible Cost data to the corresponding attributes.

        Notes
        -----
        [1] The Cost Recovery indicator will be filled if the contract type is Cost Recovery.
        [2] The Deductible Cost indicator will be filled if the contract type is Gross Split.

        """
        if isinstance(self.contract, CostRecovery):
            # Cost Recovery
            self.cost_recovery = _sum_float(
                _require_array(
                    self._profile.consolidated_cost_recovery_after_tf,
                    "_consolidated_cost_recovery_after_tf",
                )
            )
            self.cost_recovery_over_gross_rev = _as_float(
                self.cost_recovery / self.gross_revenue
            )

        elif isinstance(self.contract, GrossSplit):
            #  Deductible Cost
            self.deductible_cost = _sum_float(
                _require_array(
                    self._profile.consolidated_deductible_cost,
                    "_consolidated_deductible_cost",
                )
            )
            self.deductible_cost_over_gross_rev = (
                self.deductible_cost / self.gross_revenue
            )
        else:
            pass

    def _get_unrecoverable_carryforward_cost(self):
        """
        Function to retrieve the Unrecoverable Cost or Carry Forward Cost indicator into
        the corresponding class attributes.

        Returns
        -------
        Parsing the Unrecoverable Cost or Carry Forward Cost data to the corresponding attributes.

        Notes
        -----
        [1] The Unrecoverable cost indicator will be filled if the contract type is Cost Recovery.
        [2] The Carry Forward indicator will be filled if the contract type is Gross Split.

        """

        if isinstance(self.contract, CostRecovery):
            # Unrecoverable Cost
            self.unrec_cost = _last_float(
                _require_array(
                    self._profile.consolidated_unrecovered_after_transfer,
                    "_consolidated_unrecovered_after_transfer",
                )
            )
            self.unrec_over_costrec = _as_float(self.unrec_cost / self.cost_recovery)
            self.unrec_over_gross_rev = _as_float(self.unrec_cost / self.gross_revenue)

        elif isinstance(self.contract, GrossSplit):
            # Carry Forward Cost
            self.carry_forward_deductible_cost = _last_float(
                self._profile.consolidated_carward_deduct_cost
            )
            self.carry_forcost_over_gross_share = (
                self.carry_forward_deductible_cost / self.gross_revenue
            )
            self.carry_forcost_over_deductible_cost = (
                self.carry_forward_deductible_cost / self.deductible_cost
            )

        else:
            pass

    def _get_gross_share(self):
        """
        Function to retrieve the Contractor and Government Gross Share into the corresponding class attributes.

        Returns
        -------
        Parsing the Gross Share data to the corresponding attributes.

        """

        if isinstance(self.contract, CostRecovery):
            # Cost Recovery Gross Share
            self.ctr_gross_share = float(
                _sum_float(
                    _require_array(
                        self._profile.consolidated_contractor_share,
                        "_consolidated_contractor_share",
                    )
                )
            )
            self.gov_gross_share = float(
                _sum_float(
                    _require_array(
                        self._profile.consolidated_government_share,
                        "_consolidated_government_share",
                    )
                )
            )

        elif isinstance(self.contract, GrossSplit):
            # Gross Split Gross Share
            self.ctr_gross_share = float(
                _sum_float(
                    _require_array(
                        self._profile.consolidated_ctr_share_before_tf,
                        "_consolidated_ctr_share_before_tf",
                    )
                )
            )
            self.gov_gross_share = float(
                _sum_float(
                    _require_array(
                        self._profile.consolidated_gov_share_before_tf,
                        "_consolidated_gov_share_before_tf",
                    )
                )
            )

        else:
            pass

    def _get_net_share(self):
        """
        Function to retrieve the Net Share into the corresponding class attributes.

        Returns
        -------
        Parsing the Net Share data to the corresponding attributes.

        """
        # Contractor Net Share
        self.ctr_net_share = float(
            _sum_float(
                _require_array(
                    self._profile.consolidated_ctr_net_share,
                    "_consolidated_ctr_net_share",
                )
            )
        )
        self.ctr_net_share_over_gross_share = self.ctr_net_share / self.gross_revenue

    def _get_netcashfflow(self):
        """
        Function to retrieve the Net Cashflow of contractor into the corresponding class attributes.

        Returns
        -------
        Parsing the Profitability data to the corresponding attributes.

        """
        # CTR Net Cashflow
        self.ctr_net_cashflow = float(
            _sum_float(
                _require_array(
                    _require_array(
                        self._profile.consolidated_cashflow, "_consolidated_cashflow"
                    ),
                    "_consolidated_cashflow",
                )
            )
        )
        self.ctr_net_cashflow_over_gross_rev = (
            self.ctr_net_cashflow / self.gross_revenue
        )

    def _get_ftp_share(self):
        """
        Function to retrieve the First Tranche Petroleum (FTP) into the corresponding class attributes.

        Returns
        -------
        Parsing the First Tranche Petroleum (FTP) data to the corresponding attributes.

        """
        # Government FTP Share
        self.gov_ftp_share = float(
            _sum_float(
                _require_array(
                    _require_array(
                        self._profile.consolidated_ftp_gov, "_consolidated_ftp_gov"
                    ),
                    "_consolidated_ftp_gov",
                )
            )
        )

    def _get_equity_share(self):
        """
        Function to retrieve the Government Equity Share into the corresponding class attributes.

        Returns
        -------
        Parsing the Government Equity Share data to the corresponding attributes.

        """
        self.gov_equity_share = float(
            _sum_float(
                _require_array(
                    self._profile.consolidated_government_share,
                    "_consolidated_government_share",
                )
            )
        )

    def _get_irr(self):
        """
        Function to retrieve the Government Equity Share into the corresponding class attributes.

        Returns
        -------
        Parsing the Government Equity Share data to the corresponding attributes.

        """
        # Contractor IRR
        self.ctr_irr = irr(
            cashflow=_require_array(
                _require_array(
                    self._profile.consolidated_cashflow, "_consolidated_cashflow"
                ),
                "_consolidated_cashflow",
            )
        )

    def _get_pot(self):
        """
        Function to retrieve the Pay Out Time (POT) into the corresponding class attributes.

        Returns
        -------
        Parsing the POT data to the corresponding attributes.

        """
        self.ctr_pot = pot_psc(
            cashflow=_require_array(
                _require_array(
                    self._profile.consolidated_cashflow, "_consolidated_cashflow"
                ),
                "_consolidated_cashflow",
            ),
            cashflow_years=self._profile.project_years,
            reference_year=self._require_reference_year(),
        )

    def _get_psc_indicator(self):
        """
        Function utilized as wrapper to retrieve the indicator of Production Sharing Contract.

        Returns
        -------
        Parsing the PSC data to the corresponding attributes.

        """
        # Retrieving Cost Recovery or Deductible Cost
        self._get_costrec_deductible_cost()

        # Retrieving Unrecoverable Cost or Carry Forward Cost
        self._get_unrecoverable_carryforward_cost()

        # Retrieving Gross Share
        self._get_gross_share()

        # Retrieving Net Share
        self._get_net_share()

        # Retrieving Net Cash Flow
        self._get_netcashfflow()

        # Retrieving FTP Share (Cost Recovery)
        self._get_ftp_share()

        # Retrieving Government Equity Share
        self._get_equity_share()

        # Retrieving POT
        self._get_pot()

    def _get_baseproject_indicator(self):
        # Retrieving POT
        self._get_pot()

        # Retrieving Net Share
        self._get_net_share()

        # Retrieving Net Cash Flow
        self._get_netcashfflow()

    def _get_transition_indicator(self):
        """
        Function to retrieve the economic indicator of the contract if the contract type is Transition.

        Returns
        -------
        Parsing the economic indicator of Transition contract.

        """
        if not isinstance(self.contract, Transition):
            raise SummaryException(
                "Transition indicators require a Transition contract"
            )
        transition = self.contract
        # The transitioned children carry the post-transfer run that the table
        # and the consolidated profile are built from; the pre-transition
        # originals do not expose the reporting profile for every scheme.
        first_contract, second_contract = transition.get_transitioned_contracts()

        self.ctr_ftp = _sum_float(_require_array(self._profile.ctr_ftp, "_ctr_ftp"))
        self.gov_ftp = _sum_float(_require_array(self._profile.gov_ftp, "_gov_ftp"))
        self.ftp = self.ctr_ftp + self.gov_ftp

        self.deductible_cost = _sum_float(
            _require_array(self._profile.oil_deductible_cost, "_oil_deductible_cost")
            + _require_array(self._profile.gas_deductible_cost, "_gas_deductible_cost"),
        )
        self.deductible_cost_over_grossrev = self.deductible_cost / self.gross_revenue

        self.carry_forward_cost = (
            _require_array(self._profile.oil_unrec_cost, "_oil_unrec_cost")
            + _require_array(self._profile.gas_unrec_cost, "_gas_unrec_cost")
        )[-1]
        self.carry_forward_cost_over_grossrev = (
            self.carry_forward_cost / self.gross_revenue
        )

        self.ctr_equity_share = float(
            _sum_float(
                _require_array(self._profile.oil_ctr_ets, "_oil_ctr_ets")
                + _require_array(self._profile.gas_ctr_ets, "_gas_ctr_ets")
            )
        )
        self.gov_equity_share = float(
            _sum_float(
                _require_array(self._profile.oil_gov_ets, "_oil_gov_ets")
                + _require_array(self._profile.gas_gov_ets, "_gas_gov_ets")
            )
        )
        self.equity_to_be_split = self.ctr_equity_share + self.gov_equity_share

        self.ctr_net_operating_profit = _sum_float(
            _require_array(self._profile.net_operating_profit, "_net_operating_profit")
        )
        self.ctr_net_operating_profit_over_grossrev = (
            self.ctr_net_operating_profit / self.gross_revenue
        )

        self.ctr_net_cash_flow = _sum_float(
            _require_array(self._profile.ctr_net_cashflow, "_ctr_net_cashflow")
        )
        self.ctr_net_cash_flow_over_grossrev = (
            self.ctr_net_cash_flow / self.gross_revenue
        )

        # Contractor POT
        self.ctr_pot = pot_psc(
            cashflow=_require_array(
                _require_array(
                    self._profile.consolidated_cashflow, "_consolidated_cashflow"
                ),
                "_consolidated_cashflow",
            ),
            cashflow_years=self._profile.project_years,
            reference_year=self._require_reference_year(),
        )

        # Land and Building Tax
        self.lbt = _sum_float(
            _require_array(
                self._profile.oil_lbt_expenditures_post_tax,
                "_oil_lbt_expenditures_post_tax",
            )
            + _require_array(
                self._profile.gas_lbt_expenditures_post_tax,
                "_gas_lbt_expenditures_post_tax",
            )
        )

        # Indirect Taxes
        self.oil_indirect_taxes = _sum_float(
            _require_array(self._profile.oil_total_indirect_tax, "_oil_total_indirect_tax")
        )
        self.gas_indirect_taxes = _sum_float(
            _require_array(self._profile.gas_total_indirect_tax, "_gas_total_indirect_tax")
        )
        self.total_indirect_taxes = self.oil_indirect_taxes + self.gas_indirect_taxes

        # Undepreciated Asset (combined scalars produced by the Transition run)
        fiscal_profiles = transition.get_fiscal_profiles()
        self.undepreciated_asset_oil = float(fiscal_profiles.oil_undepreciated_asset)
        self.undepreciated_asset_gas = float(fiscal_profiles.gas_undepreciated_asset)
        self.undepreciated_asset_total = float(
            fiscal_profiles.consolidated_undepreciated_asset
        )

        # Contractor / Government Gross Share. The transitioned children own the
        # authoritative consolidated share, and the sum reproduces the table.
        first_summary = first_contract.get_summary(
            discount_rate=self.discount_rate,
            npv_mode=self.npv_mode,
            discounting_mode=self.discounting_mode,
            discount_rate_start_year=self._require_reference_year(),
            inflation_rate=self.inflation_rate,
            profitability_discounted=self.profitability_discounted,
        )
        second_summary = second_contract.get_summary(
            discount_rate=self.discount_rate,
            npv_mode=self.npv_mode,
            discounting_mode=self.discounting_mode,
            discount_rate_start_year=self._require_reference_year(),
            inflation_rate=self.inflation_rate,
            profitability_discounted=self.profitability_discounted,
        )
        self.ctr_gross_share = _as_float(
            first_summary["ctr_gross_share"] + second_summary["ctr_gross_share"]
        )
        self.gov_gross_share = _as_float(
            first_summary["gov_gross_share"] + second_summary["gov_gross_share"]
        )

        # Carry Forward Depreciation is a scheme-level input reported by each
        # transitioned child; the Transition total is their sum.
        self.oil_carry_forward_depreciation = float(
            first_summary["oil_carry_forward_depreciation"]
            + second_summary["oil_carry_forward_depreciation"]
        )
        self.gas_carry_forward_depreciation = float(
            first_summary["gas_carry_forward_depreciation"]
            + second_summary["gas_carry_forward_depreciation"]
        )
        self.total_carry_forward_depreciation = (
            self.oil_carry_forward_depreciation + self.gas_carry_forward_depreciation
        )

    def run(self):
        """
        Function utilized as the function wrapper which will retrieve the economic
        indicators of the contract.

        Returns
        -------
        Parsing the contract economic indicator to each corresponding attributes.

        """
        # Retrieving Lifting Indicator
        self._get_sum_lifting()

        # Retrieving Gross Revenue
        self._get_gross_revenue()

        # Retrieving Sunk Cost
        self._get_sunk_cost()

        # Retrieving Capital Cost Investment
        self._get_capital_cost()

        # Retrieving Intangible Investment
        self._get_intangible_cost()

        # Retrieving Investment Total
        self._get_investment()

        # Retrieving Capital Expenditure
        self._get_capex()

        # Retrieving OPEX
        self._get_opex()

        # Retrieving ASR
        self._get_asr()

        # Retrieving Cashflow Sunk Cost Pooled
        self._get_cashflow_sunkcost_pooled()

        # Retrieving DDMO
        self._get_ddmo()

        # Retrieving Government Tax Income
        self._get_gov_tax_income()

        # Retrieving Government Take
        self._get_gov_take()

        # Retrieving IRR
        self._get_contractor_irr()

        # Retrieving NPV Related Indicator
        self._get_npv_indicator()

        # Retrieving Profitability Index
        self._get_profitability_index()

        # Condition if the contract type is Cost Recovery or Gross Split
        if isinstance(self.contract, CostRecovery) or isinstance(
            self.contract, GrossSplit
        ):
            self._get_psc_indicator()
        elif isinstance(self.contract, Transition):
            self._get_transition_indicator()
        else:
            self._get_baseproject_indicator()

    def get_summary_dict(self) -> SummaryResult:
        """
        Function to returning the contract economic indicator into a form of a dictionary.

        Returns
        -------
        out: dict
            The dictionary containing the economic indicator summary of the contract.


        Notes
        -----
        This function is made in this form maintain the chain of the previous version.

        """
        # Condition if the contract type is Cost Recovery or Gross Split
        if isinstance(self.contract, CostRecovery):
            return {
                "lifting_oil": self.lifting_oil,
                "oil_wap": self.oil_wap,
                "lifting_gas": self.lifting_gas,
                "gas_wap": self.gas_wap,
                "gross_revenue": self.gross_revenue,
                "gross_revenue_oil": self.gross_revenue_oil,
                "gross_revenue_gas": self.gross_revenue_gas,
                "ctr_gross_share": self.ctr_gross_share,
                "gov_gross_share": self.gov_gross_share,
                "investment": self.investment,
                "oil_capex": self.oil_capex,
                "gas_capex": self.gas_capex,
                "sunk_cost": self.sunk_cost,
                "tangible": self.capital_cost,
                "intangible": self.intangible,
                "opex_and_asr": self.opex + self.asr,
                "opex": self.opex,
                "asr": self.asr,
                "cost_recovery / deductible_cost": self.cost_recovery,
                "cost_recovery_over_gross_rev": self.cost_recovery_over_gross_rev,
                "unrec_cost": self.unrec_cost,
                "unrec_over_costrec": self.unrec_over_costrec,
                "unrec_over_gross_rev": self.unrec_over_gross_rev,
                "ctr_net_share": self.ctr_net_share,
                "ctr_net_share_over_gross_share": self.ctr_net_share_over_gross_share,
                "ctr_net_cashflow": self.ctr_net_cashflow,
                "ctr_net_cashflow_over_gross_rev": self.ctr_net_cashflow_over_gross_rev,
                "ctr_npv": self.ctr_npv,
                "ctr_npv_sunk_cost_pooled": self.ctr_npv_sunk_cost_pooled,
                "ctr_irr": self.ctr_irr,
                "ctr_irr_sunk_cost_pooled": self.ctr_irr_sunk_cost_pooled,
                "ctr_pot": self.ctr_pot,
                "ctr_pv_ratio": self.ctr_pv_ratio,
                "ctr_pi": self.ctr_pi,
                "gov_ftp_share": self.gov_ftp_share,
                "gov_equity_share": self.gov_equity_share,
                "gov_ddmo": self.gov_ddmo,
                "gov_tax_income": self.gov_tax_income,
                "gov_take": self.gov_take,
                "gov_take_over_gross_rev": self.gov_take_over_gross_rev,
                "gov_take_npv": self.gov_take_npv,
            }

        elif isinstance(self.contract, GrossSplit):
            return {
                "lifting_oil": self.lifting_oil,
                "oil_wap": self.oil_wap,
                "lifting_gas": self.lifting_gas,
                "gas_wap": self.gas_wap,
                "gross_revenue": self.gross_revenue,
                "gross_revenue_oil": self.gross_revenue_oil,
                "gross_revenue_gas": self.gross_revenue_gas,
                "ctr_gross_share": self.ctr_gross_share,
                "gov_gross_share": self.gov_gross_share,
                "investment": self.investment,
                "oil_capex": self.oil_capex,
                "gas_capex": self.gas_capex,
                "sunk_cost": self.sunk_cost,
                "tangible": self.capital_cost,
                "intangible": self.intangible,
                "opex_and_asr": self.opex + self.asr,
                "opex": self.opex,
                "asr": self.asr,
                "cost_recovery / deductible_cost": self.deductible_cost,
                "cost_recovery_over_gross_rev": self.deductible_cost_over_gross_rev,
                "unrec_cost": self.carry_forward_deductible_cost,
                "unrec_over_costrec": self.carry_forcost_over_deductible_cost,
                "unrec_over_gross_rev": self.carry_forcost_over_gross_share,
                "ctr_net_share": self.ctr_net_share,
                "ctr_net_share_over_gross_share": self.ctr_net_share_over_gross_share,
                "ctr_net_cashflow": self.ctr_net_cashflow,
                "ctr_net_cashflow_over_gross_rev": self.ctr_net_cashflow_over_gross_rev,
                "ctr_npv": self.ctr_npv,
                "ctr_npv_sunk_cost_pooled": self.ctr_npv_sunk_cost_pooled,
                "ctr_irr": self.ctr_irr,
                "ctr_irr_sunk_cost_pooled": self.ctr_irr_sunk_cost_pooled,
                "ctr_pot": self.ctr_pot,
                "ctr_pv_ratio": self.ctr_pv_ratio,
                "ctr_pi": self.ctr_pi,
                "gov_ddmo": self.gov_ddmo,
                "gov_tax_income": self.gov_tax_income,
                "gov_take": self.gov_take,
                "gov_take_over_gross_rev": self.gov_take_over_gross_rev,
                "gov_take_npv": self.gov_take_npv,
                "gov_ftp_share": 0,
            }

        elif isinstance(self.contract, Transition):
            # Mirrors the Cost Recovery / Gross Split executive-summary shape so
            # ``convert_summary_to_dict`` can consume the Transition result.
            return {
                "lifting_oil": self.lifting_oil,
                "oil_wap": self.oil_wap,
                "lifting_gas": self.lifting_gas,
                "gas_wap": self.gas_wap,
                "gross_revenue": self.gross_revenue,
                "ctr_gross_share": self.ctr_gross_share,
                "sunk_cost": self.sunk_cost,
                "investment": self.investment,
                "tangible": self.capital_cost,
                "intangible": self.intangible,
                "opex_asr_lbt": self.opex + self.asr + self.lbt,
                "opex": self.opex,
                "asr": self.asr,
                "lbt": self.lbt,
                "cost_recovery/deductible_cost": self.deductible_cost,
                "cost_recovery_over_gross_rev": self.deductible_cost_over_grossrev,
                "unrec_cost": self.carry_forward_cost,
                "unrec_over_gross_rev": self.carry_forward_cost_over_grossrev,
                "ctr_net_share": self.ctr_net_operating_profit,
                "ctr_net_share_over_gross_share": self.ctr_net_operating_profit_over_grossrev,
                "ctr_net_cashflow": self.ctr_net_cash_flow,
                "ctr_net_cashflow_over_gross_rev": self.ctr_net_cash_flow_over_grossrev,
                "ctr_npv": self.ctr_npv,
                "ctr_irr": self.ctr_irr,
                "ctr_pot": self.ctr_pot,
                "ctr_pv_ratio": self.ctr_pv_ratio,
                "ctr_pi": self.ctr_pi,
                "gov_gross_share": self.gov_gross_share,
                "gov_ftp_share": self.gov_ftp,
                "gov_ddmo": self.gov_ddmo,
                "gov_tax_income": self.gov_tax_income,
                "gov_take": self.gov_take,
                "gov_take_over_gross_rev": self.gov_take_over_gross_rev,
                "gov_take_npv": self.gov_take_npv,
                "undepreciated_asset_oil": self.undepreciated_asset_oil,
                "undepreciated_asset_gas": self.undepreciated_asset_gas,
                "undepreciated_asset_total": self.undepreciated_asset_total,
                "total_indirect_taxes": self.total_indirect_taxes,
                "oil_indirect_taxes": self.oil_indirect_taxes,
                "gas_indirect_taxes": self.gas_indirect_taxes,
                "total_carry_forward_depreciation": self.total_carry_forward_depreciation,
                "oil_carry_forward_depreciation": self.oil_carry_forward_depreciation,
                "gas_carry_forward_depreciation": self.gas_carry_forward_depreciation,
            }
        else:
            return {
                "lifting_oil": self.lifting_oil,
                "oil_wap": self.oil_wap,
                "lifting_gas": self.lifting_gas,
                "gas_wap": self.gas_wap,
                "gross_revenue": self.gross_revenue,
                "gross_revenue_oil": self.gross_revenue_oil,
                "gross_revenue_gas": self.gross_revenue_gas,
                "investment": self.investment,
                "oil_capex": self.oil_capex,
                "gas_capex": self.gas_capex,
                "sunk_cost": self.sunk_cost,
                "tangible": self.capital_cost,
                "intangible": self.intangible,
                "opex_and_asr": self.opex + self.asr,
                "opex": self.opex,
                "asr": self.asr,
                "ctr_npv": self.ctr_npv,
                "ctr_irr": self.ctr_irr,
                "ctr_pot": self.ctr_pot,
                "ctr_pv_ratio": self.ctr_pv_ratio,
                "ctr_pi": self.ctr_pi,
                "ctr_gross_share": self.gross_revenue,
                "ctr_net_share": self.ctr_net_share,
                "ctr_net_share_over_gross_share": self.ctr_net_share_over_gross_share,
                "ctr_net_cashflow": self.ctr_net_cashflow,
                "ctr_net_cashflow_over_gross_rev": self.ctr_net_cashflow_over_gross_rev,
                # Zero value for the psc terms
                "gov_gross_share": 0,
                "cost_recovery / deductible_cost": 0,
                "cost_recovery_over_gross_rev": 0,
                "unrec_cost": 0,
                "unrec_over_gross_rev": 0,
                "gov_ftp_share": 0,
                "gov_ddmo": 0,
                "gov_tax_income": 0,
                "gov_take": 0,
                "gov_take_over_gross_rev": 0,
                "gov_take_npv": 0,
            }
