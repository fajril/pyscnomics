import json
import os
from datetime import datetime

import numpy as np

from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.econ.costs import Tangible, Intangible, OPEX, ASR
from pyscnomics.econ.revenue import Lifting, FluidType


def get_json_file_names() -> list:
    """
    A function to get a list of json file in the chosen directory.

    Returns
    -------
    list_json_file
        The of available dataset.
    """

    list_json_file = []
    if 'pyscnomics\\pyscnomics' in os.path.dirname(os.getcwd()):
        directory = (os.path.dirname(os.getcwd())) + '/dataset'

    else:
        directory = (os.path.dirname(os.getcwd())) + '/pyscnomics/dataset'

    for filenames in os.listdir(os.path.abspath(directory)):
        if filenames.endswith(".json"):
            full_file_name = filenames
            file_name = os.path.splitext(full_file_name)[0]
            list_json_file.append(file_name)

    return list_json_file


def read_json_file(file_name: str) -> dict:
    """
    A function to read json file.

    Parameters
    ----------
    file_name: str
        The file name of the json file.

    Returns
    -------
    json.loads(file_contents)
        The dictionary of the json file.

    """
    if 'pyscnomics\\pyscnomics' in os.path.dirname(os.getcwd()):
        directory = (os.path.dirname(os.getcwd())) + '/dataset/' + file_name + '.json'

    else:
        directory = (os.path.dirname(os.getcwd())) + '/pyscnomics/dataset/' + file_name + '.json'

    final_dir = os.path.abspath(directory)
    with open(final_dir) as user_file:
        file_contents = user_file.read()

    return json.loads(file_contents)


def read_fluid_type(fluid: str) -> FluidType:
    """
    A function to converting the str input into FluidType Enum class.

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


def assign_onstream_date(date_data: None | str) -> datetime | None:
    """
    A funtion to assign onstream date into datetime format.
    Parameters
    ----------
    date_data: None | str
        date data read from the json file or Ms. Excel file.

    Returns
    -------
    result_date: datetime
        The date in the datetime format

    """
    if date_data is not None:
        result_date = datetime.strptime(date_data, '%d/%m/%Y').date()
    else:
        result_date = None

    return result_date


def assign_lifting(data_raw: dict) -> tuple:
    """
    A function to assign the lifting data to the list of Lifting dataclass.

    Parameters
    ----------
    data_raw: dict
        The dictionary containing the lifting data.

    Returns
    -------
    lifting_list: tuple
        The list containing the lifting dataclass.

    """
    # Defining the data source and the list container for Lifting. Then, assign them based on their fluid type
    lifting_data = data_raw['lifting']
    lifting_list = []
    for key in lifting_data.keys():
        # Since the Lifting data for gas has different arguments input, conditional formatting is applied
        if key == 'Gas':
            lifting = Lifting(start_year=lifting_data[key]["start_year"],
                              end_year=lifting_data[key]["end_year"],
                              lifting_rate=np.array(lifting_data[key]["lifting_rate"]),
                              price=np.array(lifting_data[key]["price"]),
                              fluid_type=read_fluid_type(lifting_data[key]["fluid_type"]),
                              ghv=np.array(lifting_data[key]["ghv"]),
                              prod_rate=np.array(lifting_data[key]["prod_rate"]),
                              prod_year=np.array(lifting_data[key]["prod_year"])
                              )
            lifting_list.append(lifting)

        else:
            lifting = Lifting(start_year=lifting_data[key]["start_year"],
                              end_year=lifting_data[key]["end_year"],
                              lifting_rate=np.array(lifting_data[key]["lifting_rate"]),
                              price=np.array(lifting_data[key]["price"]),
                              fluid_type=read_fluid_type(lifting_data[key]["fluid_type"]),
                              prod_year=np.array(lifting_data[key]["prod_year"])
                              )
            lifting_list.append(lifting)

    return tuple(lifting_list)


def assign_cost(data_raw: dict) -> tuple:
    """
    A function to assign the cost data to the list of Tangible, Intangible, OPEX and ASR dataclass.

    Parameters
    ----------
    data_raw: dict

    Returns
    -------
    tangible_list : tuple
        A tuple containing the Tangible dataclass.
    intangible_list : tuple
        A tuple containing the Intangible dataclass.
    opex_list
        A tuple containing the OPEX dataclass.
    asr_list
        A tuple containing the ASR dataclass.

    """
    # Defining the data source and the list container for Tangible. Then, assign them based on their fluid type
    tangible_data = data_raw['tangible']
    tangible_list = []
    for key in tangible_data.keys():
        tangible = Tangible(start_year=tangible_data[key]['start_year'],
                            end_year=tangible_data[key]['end_year'],
                            cost=np.array(tangible_data[key]['cost']),
                            expense_year=np.array(tangible_data[key]['expense_year']),
                            pis_year=np.array(tangible_data[key]['pis_year']),
                            cost_allocation=read_fluid_type(tangible_data[key]['cost_allocation']),
                            useful_life=np.array(tangible_data[key]['useful_life']),
                            depreciation_factor=np.array(tangible_data[key]['depreciation_factor']))
        tangible_list.append(tangible)

    # Defining the data source and the list container for Intangible. Then, assign them based on their fluid type
    intangible_data = data_raw['intangible']
    intangible_list = []
    for key in intangible_data.keys():
        intangible = Intangible(start_year=intangible_data[key]['start_year'],
                                end_year=intangible_data[key]['end_year'],
                                cost=np.array(intangible_data[key]['cost']),
                                expense_year=np.array(intangible_data[key]['expense_year']),
                                cost_allocation=read_fluid_type(intangible_data[key]['cost_allocation']),
                                )
        intangible_list.append(intangible)

    # Defining the data source and the list container for Intangible. Then, them based on their fluid type
    opex_data = data_raw['opex']
    opex_list = []
    for key in opex_data.keys():
        opex = OPEX(start_year=opex_data[key]['start_year'],
                    end_year=opex_data[key]['end_year'],
                    fixed_cost=np.array(opex_data[key]['fixed_cost']),
                    expense_year=np.array(opex_data[key]['expense_year']),
                    cost_allocation=read_fluid_type(opex_data[key]['cost_allocation']))
        opex_list.append(opex)

    # Defining the data source and the list container for ASR. Then, them based on their fluid type
    asr_data = data_raw['asr']
    asr_list = []
    for key in asr_data.keys():
        asr = ASR(start_year=asr_data[key]['start_year'],
                  end_year=asr_data[key]['end_year'],
                  cost=np.array(asr_data[key]['cost']),
                  expense_year=np.array(asr_data[key]['expense_year']),
                  cost_allocation=read_fluid_type(asr_data[key]['cost_allocation']))
        asr_list.append(asr)

    return tangible_list, intangible_list, opex_list, asr_list


def load_testing(dataset_type: str, key: str) -> dict | ValueError:
    """
    A function to load the available test dataset.

    Parameters
    ----------
    dataset_type
    key

    Returns
    -------

    """
    data = read_json_file(file_name=dataset_type)
    # Condition where the testing data is not available within the chosen dataset
    if data.get("testing") is None:
        return ValueError("The testing data for this dataset is still in development.")

    else:
        data_test = dict(data)['testing'][key]
        return data_test


def load_data(dataset_type: str, contract_type: str = 'project') -> BaseProject | CostRecovery:
    """
    A function to load the available dataset.

    Parameters
    ----------
    dataset_type: str
        The type of the dataset.
    contract_type
        The type of the contract.
    Returns
    -------
    BaseProject | CostRecovery

    """
    # Checking the availability of the dataset, if not available raise a Value Error
    dataset_list = get_json_file_names()
    contract_list = ['project', 'cost_recovery', 'gross_split']
    if dataset_type not in dataset_list or contract_type not in contract_list:
        raise ValueError('Unknown dataset: "{0}", please check the Dataset and Contract Type that available.'
                         .format(dataset_type))

    # Reading the project json file
    data_raw = read_json_file(file_name=dataset_type)

    # Assigning the general config and converting them to datetime format.
    project_start_date = datetime.strptime(data_raw['start_date'], '%d/%m/%Y').date()
    project_end_date = datetime.strptime(data_raw['end_date'], '%d/%m/%Y').date()

    oil_onstream_date = assign_onstream_date(data_raw['oil_onstream_date'])
    gas_onstream_date = assign_onstream_date(data_raw['gas_onstream_date'])

    # Assigning the lifting and cost data.
    lifting_list = assign_lifting(data_raw)
    tangible_list, intangible_list, opex_list, asr_list = assign_cost(data_raw)

    if contract_type == 'project':
        return BaseProject(start_date=project_start_date,
                           end_date=project_end_date,
                           oil_onstream_date=oil_onstream_date,
                           gas_onstream_date=gas_onstream_date,
                           lifting=lifting_list,
                           tangible_cost=tangible_list,
                           intangible_cost=intangible_list,
                           opex=opex_list,
                           asr_cost=asr_list)

    elif contract_type == 'cost_recovery':
        config = read_json_file(file_name=contract_type)
        return CostRecovery(start_date=project_start_date,
                            end_date=project_end_date,
                            oil_onstream_date=oil_onstream_date,
                            gas_onstream_date=gas_onstream_date,
                            lifting=lifting_list,
                            tangible_cost=tangible_list,
                            intangible_cost=intangible_list,
                            opex=opex_list,
                            asr_cost=asr_list,
                            oil_ftp_is_available=config['ftp_is_available'],
                            oil_ftp_is_shared=config['ftp_is_shared'],
                            oil_ftp_portion=config['ftp_portion'],
                            oil_ctr_pretax_share=config['oil_ctr_pretax_share'],
                            gas_ctr_pretax_share=config['gas_ctr_pretax_share'],
                            oil_ic_rate=config['oil_ic_rate'],
                            gas_ic_rate=config['gas_ic_rate'],
                            ic_is_available=config['ic_is_available'],
                            oil_cr_cap_rate=config['oil_cr_cap_rate'],
                            gas_cr_cap_rate=config['gas_cr_cap_rate'],
                            oil_dmo_volume_portion=config['oil_dmo_volume_portion'],
                            oil_dmo_fee_portion=config['oil_dmo_fee_portion'],
                            oil_dmo_holiday_duration=config['oil_dmo_holiday_duration'],
                            gas_dmo_volume_portion=config['gas_dmo_volume_portion'],
                            gas_dmo_fee_portion=config['gas_dmo_fee_portion'],
                            gas_dmo_holiday_duration=config['gas_dmo_holiday_duration'])


if __name__ == "__main__":
    # Choosing the Dataset and contract type
    import timeit
    start_time = timeit.default_timer()
    dataset = 'small_gas'
    contract = 'project'

    # Returning the load_data function
    psc = load_data(dataset_type=dataset, contract_type=contract)
    print('Output of the load_data function \n', psc, '\n')
    end_time = timeit.default_timer()
    print("The Execution Time:", end_time - start_time)