from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.gross_split import GrossSplit
from pyscnomics.dataset.object_sample import generate_contract_sample
from pyscnomics.dataset.sample import load_testing
from pyscnomics.econ.selection import ContractSample


@pytest.mark.parametrize(
    ("case", "expected_type"),
    [
        (ContractSample.CASE_1, CostRecovery),
        (ContractSample.CASE_2, CostRecovery),
        (ContractSample.CASE_3, GrossSplit),
    ],
)
def test_generate_contract_sample_uses_supported_run_arguments(
    case: ContractSample,
    expected_type: type[CostRecovery] | type[GrossSplit],
) -> None:
    assert isinstance(generate_contract_sample(case), expected_type)


def test_case3_sample_uses_expected_progressive_oil_split() -> None:
    contract = generate_contract_sample(ContractSample.CASE_3)
    expected = np.asarray(
        cast(
            list[float],
            cast(
                object,
                load_testing(dataset_type="case3", key="oil_prog_split"),
            ),
        ),
        dtype=float,
    )

    assert isinstance(contract, GrossSplit)
    np.testing.assert_allclose(
        cast(NDArray[np.float64], contract._oil_prog_split),
        expected,
    )
