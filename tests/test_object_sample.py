import pandas as pd
import pytest

from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.gross_split import GrossSplit
from pyscnomics.dataset.object_sample import generate_contract_sample
from pyscnomics.dataset.sample import load_testing
from pyscnomics.econ.selection import ContractSample
from pyscnomics.tools.table import get_table_grosssplit_oil


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
    assert isinstance(contract, GrossSplit)
    table = get_table_grosssplit_oil(contract)
    expected_values = load_testing("case3", "oil_prog_split")
    assert isinstance(expected_values, list)
    expected = pd.DataFrame({"progressive_split": expected_values})
    actual = table.loc[:, ["progressive_split"]]
    pd.testing.assert_frame_equal(actual, expected)
