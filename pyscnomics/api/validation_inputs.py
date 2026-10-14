"""API-only preparation of the engine's effective onstream inputs.

FastAPI and the converter/adapter stay out of the shared core rules. This module
prepares the exact revenue and lifting production years the engine's
``BaseProject`` uses to resolve a nullable onstream date, so preflight and
execution agree. It performs no contract ``run()`` and never mutates the request.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, cast

from pyscnomics.api.converter import convert_dict_to_lifting
from pyscnomics.econ.revenue import aggregate_lifting_for_fluid
from pyscnomics.econ.selection import FluidType
from pyscnomics.operation_validation import normalize_expenditure_input
from pyscnomics.validation_rules import OnstreamInputs, ProjectBounds

_OnstreamInputsMap = Mapping[str, OnstreamInputs]


def prepare_onstream_inputs(
    payload: Mapping[str, object],
    bounds: ProjectBounds,
) -> _OnstreamInputsMap:
    """Prepare the effective onstream inputs for both fluids.

    Returns an empty mapping when the lifting input cannot be converted (the
    caller then keeps the conservative skipped-check behavior).
    """
    try:
        lifting = convert_dict_to_lifting(payload) or ()
    except Exception:  # noqa: BLE001 - a malformed lifting keeps the skip behavior
        return {}

    result: dict[str, OnstreamInputs] = {}
    for fluid, fluid_type in (("oil", FluidType.OIL), ("gas", FluidType.GAS)):
        aggregated = aggregate_lifting_for_fluid(
            lifting,
            fluid_type=fluid_type,
            start_year=bounds.start_year,
            end_year=bounds.end_year,
        )
        revenue = tuple(
            float(value)
            for value in cast("list[float]", aggregated.revenue().tolist())
        )
        prod_years = tuple(
            int(year) for year in cast("list[int]", aggregated.prod_year.tolist())
        )
        result[fluid] = OnstreamInputs(
            revenue=revenue,
            lifting_prod_years=prod_years,
        )
    return cast("_OnstreamInputsMap", result)


def normalize_expenditure_request(
    data: Mapping[str, object],
    operation: Literal["asr_expenditures", "lbt_expenditures"],
) -> dict[str, object]:
    """Compatibility delegate to the shared core expenditure normalizer.

    The implementation moved to :func:`pyscnomics.operation_validation.
    normalize_expenditure_input` so the standalone guard can build its read-only
    validation view without importing the API layer. The public name and
    behaviour are preserved for existing callers and tests.
    """
    return normalize_expenditure_input(data, operation)
