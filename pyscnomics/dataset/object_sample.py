import numpy as np

from pyscnomics.dataset.sample import load_data
from pyscnomics.econ.selection import FTPTaxRegime, ContractSample
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit


class ObjectSampleException(Exception):
    """Exception to raise for a misuse of contract sample object generation"""

    pass


def generate_contract_sample(case: ContractSample) -> CostRecovery | GrossSplit:
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
        psc = load_data(dataset_type='case1', contract_type='cost_recovery')

        # Defining the contract arguments
        tax_rate = 0.405
        ftp_tax_regime = FTPTaxRegime.DIRECT_MODE
        sunk_cost_reference_year = 2023
        future_rate = 0.0
        post_uu_22_year2001 = True

        contract_arguments = {
            'tax_rate': tax_rate,
            'ftp_tax_regime': ftp_tax_regime,
            'sunk_cost_reference_year': sunk_cost_reference_year,
            'future_rate': future_rate,
            'post_uu_22_year2001': post_uu_22_year2001,
        }

        # Running the contract with contract arguments
        psc.run(**contract_arguments)

        return psc

    elif case == ContractSample.CASE_2:
        # Defining the contract object
        psc = load_data(dataset_type='case2', contract_type='cost_recovery')

        # Defining the contract arguments
        tax_rate = 0.405
        ftp_tax_regime = FTPTaxRegime.DIRECT_MODE
        sunk_cost_reference_year = 2023
        future_rate = 0.0
        post_uu_22_year2001 = True

        contract_arguments = {
            'tax_rate': tax_rate,
            'ftp_tax_regime': ftp_tax_regime,
            'sunk_cost_reference_year': sunk_cost_reference_year,
            'future_rate': future_rate,
            'post_uu_22_year2001': post_uu_22_year2001,
        }

        # Running the contract with contract arguments
        psc.run(**contract_arguments)

        return psc

    elif case == ContractSample.CASE_3:
        # Defining the contract object
        psc = load_data(dataset_type='case3', contract_type='gross_split')

        # Defining the contract arguments
        tax_rate = 0.22
        sunk_cost_reference_year = 2022
        vat_rate = np.array(
            [0.11, 0.11, 0.11, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12,
             0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12,
             0.12, 0.12])
        inflation_rate = 0.0
        future_rate = 0.0
        inflation_rate_applied_to = None

        # Defining the contract arguments
        contract_arguments = {
            "tax_rate": tax_rate,
            "sunk_cost_reference_year": sunk_cost_reference_year,
            "vat_rate": vat_rate,
            "inflation_rate": inflation_rate,
            "future_rate": future_rate,
            "inflation_rate_applied_to": inflation_rate_applied_to,
        }

        # Running the contract with contract arguments
        psc.run(**contract_arguments)

        return psc
    else:
        raise ObjectSampleException(
            f"THE case argument {case} , "
            f"is should be one of: {[member.name for member in ContractSample]}"
        )

