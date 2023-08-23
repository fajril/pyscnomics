import json
import numpy as np
from datetime import datetime

from pyscnomics.contracts.project import BaseProject
from pyscnomics.econ.revenue import Lifting, FluidType
from pyscnomics.econ.costs import Tangible, Intangible, OPEX, ASR


def read_json(filename: str) -> any:
    """
    Function to read json file.
    Parameters
    ----------
    filename: str
        The json file name.

    Returns
    -------
    file_content: Any
        The json file that has been read.

    """
    with open(filename) as user_file:
        file_contents = json.load(user_file)
    return file_contents


def read_fluid_type(fluid: str) -> FluidType:
    """
    Function to return the str input to FluidType dataclass.
    Parameters
    ----------
    fluid: str
        The fluid type.

    Returns
    -------
    FluidType
        The enum class for FluidType.

    """
    if fluid == 'Oil':
        return FluidType.OIL
    elif fluid == 'Gas':
        return FluidType.GAS
    elif fluid == 'Sulfur':
        return FluidType.SULFUR
    elif fluid == 'Electricity':
        return FluidType.ELECTRICITY
    elif fluid == 'CO2':
        return FluidType.CO2


def assign_lifting(lifting_data: dict) -> list:
    """
    lifting_data: dict
    Parameters
    ----------
    lifting_data: dict
        The dictionary containing lifting data.

    Returns
    -------
    lifting_list: list
        The List containing Lifting dataclasses.

    """
    # Defining container list for Lifting
    lifting_list = []

    # Iterating lifting data to assign them based on their fluid type
    for key in dict(lifting_data):
        # Since the Lifting data for gas has different arguments input, conditional formatting is applied
        if key == 'gas':
            lifting = Lifting(start_year=lifting_data[key]["start_year"],
                              end_year=lifting_data[key]["end_year"],
                              lifting_rate=np.array(lifting_data[key]["lifting_rate"]),
                              price=np.array(lifting_data[key]["price"]),
                              fluid_type=read_fluid_type(lifting_data[key]["fluid_type"]),
                              ghv=np.array(lifting_data[key]["ghv"]),
                              prod_rate=np.array(lifting_data[key]["prod_rate"])
                              )
            lifting_list.append(lifting)

        else:
            lifting = Lifting(start_year=lifting_data[key]["start_year"],
                              end_year=lifting_data[key]["end_year"],
                              lifting_rate=np.array(lifting_data[key]["lifting_rate"]),
                              price=np.array(lifting_data[key]["price"]),
                              fluid_type=read_fluid_type(lifting_data[key]["fluid_type"]),
                              )
            lifting_list.append(lifting)
    return lifting_list


def assign_cost(tangible_data, intangible_data, opex_data, asr_data) -> tuple:
    # Defining the tangible container and assigning the data to the corresponding dataclass
    tangible_list = []
    for key in dict(tangible_data):
        tangible = Tangible(start_year=tangible_data[key]['start_year'],
                            end_year=tangible_data[key]['end_year'],
                            cost=np.array(tangible_data[key]['cost']),
                            expense_year=np.array(tangible_data[key]['expense_year']),
                            pis_year=np.array(tangible_data[key]['pis_year']),
                            cost_allocation=read_fluid_type(tangible_data[key]['cost_allocation']), )
        tangible_list.append(tangible)

    # Defining the intangible container and assigning the data to the corresponding dataclass
    intangible_list = []
    for key in dict(intangible_data):
        intangible = Intangible(start_year=intangible_data[key]['start_year'],
                                end_year=intangible_data[key]['end_year'],
                                cost=np.array(intangible_data[key]['cost']),
                                expense_year=np.array(intangible_data[key]['expense_year']),
                                cost_allocation=read_fluid_type(intangible_data[key]['cost_allocation']))
        intangible_list.append(intangible)

    # Defining the opex container and assigning the data to the corresponding dataclass
    opex_list = []
    for key in dict(opex_data):
        opex = OPEX(start_year=opex_data[key]['start_year'],
                    end_year=opex_data[key]['end_year'],
                    fixed_cost=np.array(opex_data[key]['fixed_cost']),
                    cost_allocation=read_fluid_type(opex_data[key]['cost_allocation']))
        opex_list.append(opex)

    # Defining the asr container and assigning the data to the corresponding dataclass
    asr_list = []
    for key in dict(asr_data):
        asr = ASR(start_year=asr_data[key]['start_year'],
                  end_year=asr_data[key]['end_year'],
                  cost=np.array(asr_data[key]['cost']),
                  expense_year=np.array(asr_data[key]['expense_year']),
                  cost_allocation=read_fluid_type(asr_data[key]['cost_allocation']))
        asr_list.append(asr)

    return tangible_list, intangible_list, opex_list, asr_list


def get_data(data: dict) -> tuple:
    # Reading the start_date, end_date, lifting and cost data
    data_source = list(data.values())
    start_date = datetime.strptime(data_source[0], '%d/%m/%Y').date()
    end_date = datetime.strptime(data_source[1], '%d/%m/%Y').date()
    lifting_data = data_source[2]
    tangible_data = data_source[3]
    intangible_data = data_source[4]
    opex_data = data_source[5]
    asr_data = data_source[6]

    # Assigning lifting data and cost data to each corresponding dataclass
    lifting_list = assign_lifting(lifting_data=lifting_data)
    tangible_list, intangible_list, opex_list, asr_list = assign_cost(tangible_data=tangible_data,
                                                                      intangible_data=intangible_data,
                                                                      opex_data=opex_data,
                                                                      asr_data=asr_data)

    return (start_date, end_date, tuple(lifting_list), tuple(tangible_list), tuple(intangible_list),
            tuple(opex_list), tuple(asr_list))


def load_data(dataset: str, contract: str = 'project') -> BaseProject:
    """
    Function to load the provided dataset.
    Parameters
    ----------
    dataset: str
        The category of the dataset.
        The available dataset are ['small_oil', 'medium_oil', 'large_oil', 'small_gas', 'medium_gas', 'large_gas'].
    contract: str
        The type of the contract.

    Returns
    -------
    BaseProject
    """
    # Checking the input data, is it exist in the provided dataset
    dataset_list = ['small_oil', 'medium_oil', 'large_oil', 'small_oil', 'medium_oil', 'large_oil',
                    'small_gas', 'medium_gas', 'large_gas', 'small_gas', 'medium_gas', 'large_gas',
                    'small_oil_gas']
    if dataset not in dataset_list:
        raise ValueError('Unknown dataset: "{0}", please check the Dataset Type that available.'.format(dataset))

    # Read the jason file
    data = read_json(filename=dataset + '.json')
    start_date, end_date, lifting, tangible, intangible, opex, asr = get_data(data)

    # Returning BaseProject dataclass for contract type: project
    if contract == 'project':
        return BaseProject(start_date=start_date,
                           end_date=end_date,
                           lifting=lifting,
                           tangible_cost=tangible,
                           intangible_cost=intangible,
                           opex=opex,
                           asr_cost=asr)

    # Returning CostRecovery dataclass for contract type: cr
    elif contract == 'cr':
        return NotImplemented

    # Returning GrossSplit dataclass for contract type: cr
    elif contract == 'gs':
        return NotImplemented

    else:
        raise ValueError('Unknown contract: "{0}", please check the Contract type that available.'.format(contract))

    # Todo: Add the output dataclass for Cost Recovery and Gross Split after both of these dataclasses are finished.


def load_testing(dataset: str, class_type: str) -> dict | ValueError:
    """
    Function to load the testing data for each dataclass that project has.
    Parameters
    ----------
    dataset: str
        The category of the dataset.
        The available dataset are ['small_oil', 'medium_oil', 'large_oil', 'small_gas', 'medium_gas', 'large_gas'].
    class_type: str
        The class_type type dataset.
        The available class_type category are ['lifting', 'tangible', 'intangible', 'opex', 'asr']

    Returns
    -------
    data_testing
        The testing data based on the chosen dataset and data_type.

    """
    dataset_list = ['small_oil', 'medium_oil', 'large_oil', 'small_oil', 'medium_oil', 'large_oil',
                    'small_gas', 'medium_gas', 'large_gas', 'small_gas', 'medium_gas', 'large_gas',
                    'small_oil_gas']
    if dataset not in dataset_list:
        raise ValueError(
            'Unknown dataset: "{0}", please check the dataset or data_type that available.'
            .format(dataset))

    # Reading the json file
    data = read_json(filename=dataset + '.json')
    data_dict = dict(data)

    # Condition where the testing data is not available within the chosen dataset
    if data_dict.get("testing") is None:
        return ValueError("The testing data for this dataset is still in development.")
    else:
        return dict(data["testing"][class_type])


if __name__ == "__main__":
    # Choosing the Dataset and contract type
    data_type = 'small_gas'
    project_type = 'project'

    # Testing the load_data function
    psc = load_data(dataset=data_type, contract=project_type)
    print('# Output of the load_data function:')
    print(psc, '\n')
    print('# PSC attribute:')
    print('Start Date: ', psc.start_date)
    print('End Date: ', psc.end_date, '\n')
    print('Lifting Data: \n', psc.lifting, '\n')
    print('Tangible Cost Data: \n', psc.tangible_cost, '\n')
    print('Intangible Cost Data: \n', psc.intangible_cost, '\n')
    print('Opex Data: ', psc.opex, '\n')
    print('Abandon and Site Restoration (ASR) Data: \n', psc.asr_cost, '\n')

    # Testing the load_test function
    test = load_testing(dataset=data_type, class_type='lifting')
    print('# Output of the load_test function:')
    print(test)