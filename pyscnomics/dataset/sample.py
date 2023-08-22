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


def assign_to_dataclass(data: any) -> tuple:
    """
    Function to assigning the data to each corresponding dataclass.
    Parameters
    ----------
    data: any
        The data that read from json file.

    Returns
    -------
    out: tuple
        start_date: datetime
            The start date of the project.
        end_date:
            The end date of the project.
        data_lifting:
            Lifting dataclass.
        data_tangible:
            Tangible dataclass.
        data_intangible:
            Intangible dataclass.
        data_opex:
            OPEX dataclass.
        data_asr:
            ASR dataclass.
    """
    start_date = datetime.strptime(data["start_date"], '%d/%m/%Y').date()
    end_date = datetime.strptime(data["end_date"], '%d/%m/%Y').date()

    data_lifting = tuple([Lifting(start_year=data["lifting"]["start_year"],
                                  end_year=data["lifting"]["end_year"],
                                  lifting_rate=np.array(data["lifting"]["lifting_rate"]),
                                  price=np.array(data["lifting"]["price"]),
                                  fluid_type=read_fluid_type(data["lifting"]["fluid_type"]))])

    data_tangible = tuple([Tangible(start_year=data["tangible"]["start_year"],
                                    end_year=data["tangible"]["end_year"],
                                    cost=np.array(data["tangible"]["cost"]),
                                    expense_year=np.array(data["tangible"]["expense_year"]),
                                    pis_year=np.array(data["tangible"]["pis_year"]),
                                    cost_allocation=read_fluid_type(data["tangible"]["cost_allocation"]))])

    data_intangible = tuple([Intangible(start_year=data["intangible"]["start_year"],
                                        end_year=data["intangible"]["end_year"],
                                        cost=np.array(data["intangible"]["cost"]),
                                        expense_year=np.array(data["intangible"]["expense_year"]),
                                        cost_allocation=read_fluid_type(data["intangible"]["cost_allocation"]))])

    data_opex = tuple([OPEX(start_year=data["opex"]["start_year"],
                            end_year=data["opex"]["end_year"],
                            fixed_cost=np.array(data["opex"]["fixed_cost"]),
                            cost_allocation=read_fluid_type(data["opex"]["cost_allocation"]))])

    data_asr = tuple([ASR(start_year=data["asr"]["start_year"],
                          end_year=data["asr"]["end_year"],
                          cost=np.array(data["asr"]["cost"]),
                          expense_year=np.array(data["asr"]["expense_year"]),
                          cost_allocation=read_fluid_type(data["asr"]["cost_allocation"]))])

    return start_date, end_date, data_lifting, data_tangible, data_intangible, data_opex, data_asr


def load_data(dataset: str, contract: str = 'project') -> BaseProject:
    """
    Function to load the provided dataset.
    Parameters
    ----------
    dataset: str
        The category of the dataset.
        The available dataset is ['small_oil', 'medium_oil', 'large_oil', 'small_gas', 'medium_gas', 'large_gas'].
    contract: str
        The type of the contract.

    Returns
    -------
    BaseProject
    """
    # Checking the input data, is it exist in the provided dataset
    dataset_list = ['small_oil', 'medium_oil', 'large_oil', 'small_oil', 'medium_oil', 'large_oil']
    if dataset not in dataset_list:
        raise ValueError('Unknown dataset: "{0}", please check the Dataset Type that available.'.format(dataset))

    # Read the jason file
    data = read_json(filename=dataset + '.json')
    start_date, end_date, lifting, tangible, intangible, opex, asr = assign_to_dataclass(data)

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


def load_testing(dataset: str, data_type: str) -> dict:
    """
    Function to load the testing data for each dataclass that project has.
    Parameters
    ----------
    dataset: str
        The category of the dataset.
        The available dataset is ['small_oil', 'medium_oil', 'large_oil', 'small_gas', 'medium_gas', 'large_gas'].
    data_type: str
        The dataclass type.
        The available datatype is ['lifting', 'tangible', 'intangible', 'opex', 'asr']

    Returns
    -------
    data_testing
        The testing data based on the chosen dataset and data_type.

    """
    dataset_list = ['small_oil', 'medium_oil', 'large_oil', 'small_oil', 'medium_oil', 'large_oil']
    dataclass_list = ['lifting', 'tangible', 'intangible', 'opex', 'asr']
    if dataset not in dataset_list or data_type not in dataclass_list:
        raise ValueError(
            'Unknown dataset: "{0}" or "{1}", please check the dataset or data_type that available.'
            .format(dataset, data_type))

    data = read_json(filename=dataset + '.json')
    data_testing = dict(data["testing"][data_type])
    return data_testing


if __name__ == "__main__":
    psc = load_data(dataset='medium_oil', contract='project')
    print('# Output of the load_data function:')
    print(psc, '\n')

    print('# PSC attribute:')
    print('Start Date: ', psc.start_date, '\n')
    print('End Date: ', psc.end_date, '\n')
    print('Lifting Data: \n', psc.lifting, '\n')
    print('Tangible Cost Data: \n', psc.tangible_cost, '\n')
    print('Intangible Cost Data: \n', psc.intangible_cost, '\n')
    print('Opex Data: ', psc.opex, '\n')
    print('Abandon and Site Restoration (ASR) Data: \n', psc.asr_cost, '\n')

    data_test = load_testing(dataset='medium_oil', data_type='lifting')
    print('# Output of the load_testing function:')
    print(data_test)
