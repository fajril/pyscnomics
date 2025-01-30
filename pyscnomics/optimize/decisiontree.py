import numpy as np
from itertools import product
from pyscnomics.econ.selection import DecisionTreeParameters
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts.transition import Transition, adjust_rows


def make_contract_list(
        oil_ftp_min: float = None,
        oil_ftp_max: float = None,
        oil_split_min: float = None,
        oil_split_max: float = None,
        oil_dmoportion_min: float = None,
        oil_dmoportion_max: float = None,
        oil_dmofee_min:float = None,
        oil_dmofee_max:float = None,
        gas_ftp_min: float = None,
        gas_ftp_max: float = None,
        gas_split_min: float = None,
        gas_split_max: float = None,
        gas_dmoportion_min: float = None,
        gas_dmoportion_max: float = None,
        gas_dmofee_min:float = None,
        gas_dmofee_max:float = None,
        num_variation: int = 3,
        verbose: bool = False,
):

    # Initiating the list
    oil_ftp_list = np.linspace(oil_ftp_min, oil_ftp_max, num_variation)
    oil_split_list = np.linspace(oil_split_min, oil_split_max, num_variation)
    oil_dmoportion_list = np.linspace(oil_dmoportion_min, oil_dmoportion_max, num_variation)
    oil_dmofee_list = np.linspace(oil_dmofee_min, oil_dmofee_max, num_variation)

    gas_ftp_list = np.linspace(gas_ftp_min, gas_ftp_max, num_variation)
    gas_split_list = np.linspace(gas_split_min, gas_split_max, num_variation)
    gas_dmoportion_list = np.linspace(gas_dmoportion_min, gas_dmoportion_max, num_variation)
    gas_dmofee_list = np.linspace(gas_dmofee_min, gas_dmofee_max, num_variation)

    # Get all combinations
    combinations = list(
        product(
            oil_ftp_list,
            oil_split_list,
            oil_dmoportion_list,
            oil_dmofee_list,
            gas_ftp_list,
            gas_split_list,
            gas_dmoportion_list,
            gas_dmofee_list,
        )
    )

    # Print the combinations
    if verbose is True:
        for combination in combinations:
            print(combination)

    return combinations

def adjust_contract(contract: CostRecovery,
                    contract_arguments: dict,
                    parameter_selection: DecisionTreeParameters,
                    value: float):
    if parameter_selection is DecisionTreeParameters.OIL_FTP:
        contract.oil_ftp_portion = value
    elif parameter_selection is DecisionTreeParameters.OIL_SPLIT:
        contract.oil_ctr_pretax_share = value
    elif parameter_selection is DecisionTreeParameters.OIL_DMO_PORTION:
        contract.oil_dmo_volume_portion = value
    elif parameter_selection is DecisionTreeParameters.OIL_DMO_FEE:
        contract.oil_dmo_fee_portion = value

    if parameter_selection is DecisionTreeParameters.GAS_FTP:
        contract.gas_ftp_portion = value
    elif parameter_selection is DecisionTreeParameters.GAS_SPLIT:
        contract.gas_ctr_pretax_share = value
    elif parameter_selection is DecisionTreeParameters.GAS_DMO_PORTION:
        contract.gas_dmo_volume_portion = value
    elif parameter_selection is DecisionTreeParameters.GAS_DMO_FEE:
        contract.gas_dmo_fee_portion = value

    return contract.run(**contract_arguments)

def decision_tree_psc(
        contract: CostRecovery = None,
        contract_arguments: dict = None,
        oil_ftp_min: float = None,
        oil_ftp_max: float = None,
        oil_split_min: float = None,
        oil_split_max: float = None,
        oil_dmoportion_min: float = None,
        oil_dmoportion_max: float = None,
        oil_dmofee_min:float = None,
        oil_dmofee_max:float = None,
        gas_ftp_min: float = None,
        gas_ftp_max: float = None,
        gas_split_min: float = None,
        gas_split_max: float = None,
        gas_dmoportion_min: float = None,
        gas_dmoportion_max: float = None,
        gas_dmofee_min:float = None,
        gas_dmofee_max:float = None,
        num_variation: int = 3,
        verbose: bool = False,
):
    # Get the combination of parameters
    combinations = make_contract_list(
        oil_ftp_min=oil_ftp_min,
        oil_ftp_max=oil_ftp_max,
        oil_split_min=oil_split_min,
        oil_split_max=oil_split_max,
        oil_dmoportion_min=oil_dmoportion_min,
        oil_dmoportion_max=oil_dmoportion_max,
        oil_dmofee_min=oil_dmofee_min,
        oil_dmofee_max=oil_dmofee_max,
        gas_ftp_min=gas_ftp_min,
        gas_ftp_max=gas_ftp_max,
        gas_split_min=gas_split_min,
        gas_split_max=gas_split_max,
        gas_dmoportion_min=gas_dmoportion_min,
        gas_dmoportion_max=gas_dmoportion_max,
        gas_dmofee_min=gas_dmofee_min,
        gas_dmofee_max=gas_dmofee_max,
        num_variation=num_variation,
        verbose=verbose,
    )



test = decision_tree_psc(
    ftp_min=0.10,
    ftp_max=0.25,
    split_min=0.30,
    split_max=0.60,
    dmoportion_min=0.1,
    dmoportion_max=1,
    dmofee_min=0.5,
    dmofee_max=0.25,
)


