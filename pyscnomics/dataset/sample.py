import json
from pathlib import Path

import numpy as np
from datetime import datetime
import timeit

import pandas as pd

from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.econ.revenue import Lifting, FluidType
from pyscnomics.econ.costs import Tangible, Intangible, OPEX, ASR


def read_json(filename: str) -> dict:
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
    Function to assign lifting data to the corresponding dataclass.

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
    return lifting_list


def assign_cost(tangible_data, intangible_data, opex_data, asr_data) -> tuple:
    """
    Assigning the cost data to each corresponding dataclasses.

    Parameters
    ----------
    tangible_data
    intangible_data
    opex_data
    asr_data

    Returns
    -------

    """
    # Defining the tangible container and assigning the data to the corresponding dataclass
    tangible_list = []
    for key in dict(tangible_data):
        tangible = Tangible(start_year=tangible_data[key]['start_year'],
                            end_year=tangible_data[key]['end_year'],
                            cost=np.array(tangible_data[key]['cost']),
                            expense_year=np.array(tangible_data[key]['expense_year']),
                            pis_year=np.array(tangible_data[key]['pis_year']),
                            cost_allocation=read_fluid_type(tangible_data[key]['cost_allocation']),
                            useful_life=np.array(tangible_data[key]['useful_life']),
                            depreciation_factor=np.array(tangible_data[key]['depreciation_factor']))
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
                    cost_allocation=read_fluid_type(opex_data[key]['cost_allocation']),
                    expense_year=np.array(opex_data[key]['expense_year']))
        opex_list.append(opex)

    # Defining the asr container and assigning the data to the corresponding dataclass
    asr_list = []
    for key in dict(asr_data):
        asr = ASR(start_year=asr_data[key]['start_year'],
                  end_year=asr_data[key]['end_year'],
                  cost=np.array(asr_data[key]['cost']),
                  expense_year=np.array(asr_data[key]['expense_year']),
                  cost_allocation=read_fluid_type(asr_data[key]['cost_allocation']),
                  rate=asr_data[key]['rate'])
        asr_list.append(asr)

    return tangible_list, intangible_list, opex_list, asr_list


def get_data(data: dict) -> tuple:
    """
    Function to retrieve data from lifting and cost json file.

    Parameters
    ----------
    data: dict
        The dictionary containing the information from json file.

    Returns
    -------
    out: tuple
        start_date: tuple
            The starting date of the project.
        end_date: tuple
            The ending date of the project.
        lifting_list: tuple
            The list containing Lifting data.
        tangible_list: tuple
            The list containing Tangible data.
        intangible_list: tuple
            The list containing Intangible data.
        opex_list: tuple
            The list containing Opex data.
        asr_list: tuple
            The list containing ASR data.

    """
    # Reading the start_date, end_date, lifting and cost data
    data_source = list(data.values())
    start_date = datetime.strptime(data_source[0], '%d/%m/%Y').date()
    end_date = datetime.strptime(data_source[1], '%d/%m/%Y').date()
    onstream_date = datetime.strptime(data_source[2], '%d/%m/%Y').date()
    lifting_data = data_source[3]
    tangible_data = data_source[4]
    intangible_data = data_source[5]
    opex_data = data_source[6]
    asr_data = data_source[7]

    # Assigning lifting data and cost data to each corresponding dataclass
    lifting_list = assign_lifting(lifting_data=lifting_data)
    tangible_list, intangible_list, opex_list, asr_list = assign_cost(tangible_data=tangible_data,
                                                                      intangible_data=intangible_data,
                                                                      opex_data=opex_data,
                                                                      asr_data=asr_data)

    return (start_date, end_date, onstream_date, tuple(lifting_list), tuple(tangible_list), tuple(intangible_list),
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
    BaseProject | GrossSplit
    """
    # Checking the input data, is it exist in the provided dataset
    dataset_list = ['small_oil', 'medium_oil', 'large_oil', 'small_oil', 'medium_oil', 'large_oil',
                    'small_gas', 'medium_gas', 'large_gas', 'small_gas', 'medium_gas', 'large_gas',
                    'small_oil_gas', 'gs_real', 'gas_real']
    if dataset not in dataset_list:
        raise ValueError('Unknown dataset: "{0}", please check the Dataset Type that available.'.format(dataset))

    # Read the jason file
    file_path = Path(dataset + '.json')
    data = read_json(filename=str(file_path.resolve()))
    start_date, end_date, onstream_date, lifting, tangible, intangible, opex, asr = get_data(data)

    contract_path = Path(contract + '.json')
    data_config = read_json(filename=str(contract_path.resolve()))

    # Returning BaseProject dataclass for contract type: project
    if contract == 'project':
        result = BaseProject(start_date=start_date,
                             end_date=end_date,
                             lifting=lifting,
                             tangible_cost=tangible,
                             intangible_cost=intangible,
                             opex=opex,
                             asr_cost=asr,
                             onstream_date=onstream_date)

    # Returning CostRecovery dataclass for contract type: cost_recovery
    elif contract == 'cost_recovery':
        result = CostRecovery(start_date=start_date,
                              end_date=end_date,
                              onstream_date=onstream_date,
                              lifting=lifting,
                              tangible_cost=tangible,
                              intangible_cost=intangible,
                              opex=opex,
                              asr_cost=asr,
                              ftp_is_available=data_config['ftp_is_available'],
                              ftp_is_shared=data_config['ftp_is_shared'],
                              ftp_portion=data_config['ftp_portion'],
                              oil_ctr_pretax_share=data_config['oil_ctr_pretax_share'],
                              gas_ctr_pretax_share=data_config['gas_ctr_pretax_share'],
                              oil_ic_rate=data_config['oil_ic_rate'],
                              gas_ic_rate=data_config['gas_ic_rate'],
                              ic_is_available=data_config['ic_is_available'],
                              oil_cr_cap_rate=data_config['oil_cr_cap_rate'],
                              gas_cr_cap_rate=data_config['gas_cr_cap_rate'],
                              oil_dmo_volume_portion=data_config['oil_dmo_volume_portion'],
                              oil_dmo_fee_portion=data_config['oil_dmo_fee_portion'],
                              oil_dmo_holiday_duration=data_config['oil_dmo_holiday_duration'],
                              gas_dmo_volume_portion=data_config['gas_dmo_volume_portion'],
                              gas_dmo_fee_portion=data_config['gas_dmo_fee_portion'],
                              gas_dmo_holiday_duration=data_config['gas_dmo_holiday_duration'], )

    else:
        result = ValueError('Unknown contract: "{0}", please check the Contract type that available.'.format(contract))

    return result


def load_testing(dataset: str, key: str) -> dict | ValueError:
    """
    Function to load the testing data for each dataclass that the project has.

    Parameters
    ----------
    dataset: str
        The category of the dataset.
        The available dataset are ['small_oil', 'medium_oil', 'large_oil', 'small_gas', 'medium_gas', 'large_gas'].
    key: str
        The key type dataset.

    Returns
    -------
    data_testing
        The testing data based on the chosen dataset and data_type.

    """
    # Reading the json file
    data = read_json(filename=dataset + '.json')

    # Condition where the testing data is not available within the chosen dataset
    if data.get("testing") is None:
        return ValueError("The testing data for this dataset is still in development.")

    else:
        data_test = dict(data)['testing'][key]
        return data_test


def load_cost(filename: str,
              start_year: int,
              end_year: int,
              cost_allocation: FluidType = FluidType.OIL,
              template: str = "pyscnomics") -> tuple[Tangible, Intangible, OPEX, ASR] | ValueError:
    """
    Function to load the cost data from Excel file.

    Parameters
    ----------
    filename: str
        The name of the Excel file.
    start_year: int
        The start year of the cost data.
    end_year: int
        The end year of the cost data
    cost_allocation: FluidType
        The fluid type of that the cost will be allocated to.
    template: str
        The type of Excel source that will be read. The available types are: ['pyscnomics', 'questor']

    Returns
    -------
    out: tuple
        Tangible
            The Tangible dataclass.
        Intangible
            The Intangible dataclass
        OPEX
            The OPEX dataclass
        ASR
            The ASR dataclass
    """

    # Defining the available template list and making the condition if not satisfied
    template_list = ['pyscnomics', 'questor']
    if template not in template_list:
        raise ValueError('Unknown Template: "{0}", please check the Template Type in Docstring.'.format(template))

    # Reading the Questor Excels file from column B to W and replacing the value of NaN with 0
    df = pd.read_excel(filename, skiprows=18, header=None, na_values=0).fillna(value=0)
    years_arr = np.arange(start_year, start_year + df.shape[0], 1)
    df.set_index(years_arr, inplace=True)

    # Assigning the Tangible data
    tangible_arr = np.array(df[[5, 7, 8, 9, 10, 11, 12]].sum(axis=1).to_numpy(dtype=float))
    tangible = Tangible(start_year=start_year,
                        end_year=end_year,
                        cost=tangible_arr,
                        expense_year=years_arr,
                        cost_allocation=cost_allocation)

    # Assigning the Intangible data
    intangible_arr = df[6].to_numpy(dtype=float)
    intangible = Intangible(start_year=start_year,
                            end_year=end_year,
                            cost=intangible_arr,
                            expense_year=years_arr,
                            cost_allocation=cost_allocation)

    # Assigning the ASR data
    asr_arr = df[18].to_numpy(dtype=float)
    asr = ASR(start_year=start_year,
              end_year=end_year,
              cost=asr_arr,
              expense_year=years_arr,
              cost_allocation=cost_allocation)

    # Assigning the OPEX data
    fixed_cost_arr = df[[13, 14, 15, 16, 17]].sum(axis=1).to_numpy(dtype=float)
    opex = OPEX(start_year=start_year,
                end_year=end_year,
                fixed_cost=fixed_cost_arr,
                expense_year=years_arr,
                cost_allocation=cost_allocation)

    # Assigning the OPEX data
    # fixed_cost_arr = df[[13, 16, 17]].sum(axis=1).to_numpy(dtype=float)
    # variable_cost_arr = df[14].to_numpy(dtype=float)
    # cost_per_volume_arr = df[15].to_numpy(dtype=float)

    # Reading the produced fluid for determining the prod_rate attributes of the OPEX dataclass
    # if cost_allocation == FluidType.OIL:
    #     prod_arr = df[20].to_numpy(dtype=float)
    #
    # elif cost_allocation == FluidType.GAS:
    #     prod_arr = df[22].to_numpy(dtype=float)
    #
    # else:
    #     prod_arr = df[21].to_numpy(dtype=float)
    #
    # opex = OPEX(start_year=start_year,
    #             end_year=end_year,
    #             fixed_cost=fixed_cost_arr,
    #             expense_year=years_arr,
    #             cost_allocation=cost_allocation,
    #             prod_rate=prod_arr,
    #             cost_per_volume=cost_per_volume_arr,
    #             )

    return tangible, intangible, opex, asr


if __name__ == "__main__":
    # Choosing the Dataset and contract type
    data_type = 'gas_real'
    project_type = 'cost_recovery'

    # Testing the load_data function
    psc = load_data(dataset=data_type, contract=project_type)
    print('Output of the load_data')
    print(psc, '\n')

    print('Time of reading the dataset and running the PSC :')
    print(timeit.timeit(psc.run, number=1))
    print('\n')

    # Reading Questor Template
    questor_result = load_cost(filename='questor_template.xls',
                               cost_allocation=FluidType.OIL,
                               template='questor',
                               start_year=2023,
                               end_year=2042)
    print('Output of the load_cost')
    for cost in questor_result:
        print(cost, '\n')
