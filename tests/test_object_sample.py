import pytest

from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.gross_split import GrossSplit
from pyscnomics.dataset.object_sample import generate_contract_sample
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
