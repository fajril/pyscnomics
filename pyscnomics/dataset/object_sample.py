import numpy as np

from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.gross_split import GrossSplit
from pyscnomics.dataset.sample import load_data
from pyscnomics.econ.selection import ContractSample, FTPTaxRegime, GrossSplitRegime


class ObjectSampleException(Exception):
    """Exception to raise for a misuse of contract sample object generation"""

    pass


def generate_contract_sample(case: ContractSample) -> CostRecovery | GrossSplit:
    """Generate a supported sample contract."""
    return _generate_contract_sample(case)


def _generate_contract_sample(case: object) -> CostRecovery | GrossSplit:
    """
    Function to generate sample of contract object.

    Parameters
    ----------
    case: ContractSample
        The selection of the contract sample

    Returns
    -------
    psc: CostRecovery | GrossSplit
        The contract object

    """
    if case == ContractSample.CASE_1:
        # Defining the contract object
        psc = load_data(dataset_type="case1", contract_type="cost_recovery")

        psc.run(
            effective_tax_rate=0.405,
            ftp_tax_regime=FTPTaxRegime.DIRECT_MODE,
            post_uu_22_year2001=True,
        )
        return psc

    elif case == ContractSample.CASE_2:
        # Defining the contract object
        psc = load_data(dataset_type="case2", contract_type="cost_recovery")

        psc.run(
            effective_tax_rate=0.405,
            ftp_tax_regime=FTPTaxRegime.DIRECT_MODE,
            post_uu_22_year2001=True,
        )
        return psc

    elif case == ContractSample.CASE_3:
        # Defining the contract object
        psc = load_data(dataset_type="case3", contract_type="gross_split")

        # Defining the contract arguments
        tax_rate = 0.22
        vat_rate = np.array(
            [
                0.11,
                0.11,
                0.11,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
                0.12,
            ]
        )
        inflation_rate = 0.0
        inflation_rate_applied_to = None

        psc.run(
            effective_tax_rate=tax_rate,
            vat_rate=vat_rate,
            inflation_rate=inflation_rate,
            inflation_rate_applied_to=inflation_rate_applied_to,
            regime=GrossSplitRegime.PERMEN_ESDM_12_2020,
        )
        return psc
    raise ObjectSampleException(
        f"THE case argument {case} , is should be one of: {[member.name for member in ContractSample]}"
    )
