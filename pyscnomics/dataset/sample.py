import json
import os
import importlib.resources as pkg_resources
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from pyscnomics.contracts.project import BaseProject
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.econ.costs import CapitalCost, Intangible, OPEX, ASR, LBT, CostOfSales
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
    # if 'pyscnomics\\pyscnomics' in os.path.dirname(os.getcwd()):
    #     directory = (os.path.dirname(os.getcwd())) + '/dataset/' + file_name + '.json'
    #
    # else:
    #     directory = (os.path.dirname(os.getcwd())) + '/pyscnomics/dataset/' + file_name + '.json'
    #
    # final_dir = os.path.abspath(directory)
    # with open(final_dir) as user_file:
    #     file_contents = user_file.read()

    # return json.loads(file_contents)

    # Use importlib.resources.path to locate the JSON file within the package
    with pkg_resources.path('pyscnomics.dataset', file_name + '.json') as file_path:
        final_dir = Path(file_path)

    # Open and read the JSON file, returning its contents as a dictionary
    with open(final_dir) as user_file:
        file_contents = user_file.read()

    return json.loads(file_contents)


def read_fluid_type(fluid: list | str) -> list[FluidType] | FluidType:
    """
    A function to converting the str input into FluidType Enum class.

    Parameters
    ----------
    fluid: str | list
        The fluid type.

    Returns
    -------
    list[FluidType] | FluidType
        The enum class for FluidType.

    """
    if isinstance(fluid, str):
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

    else:

        fluid_mapping = {'Oil': FluidType.OIL,
                         'Gas': FluidType.GAS,
                         'Sulfur': FluidType.SULFUR,
                         'Electricity': FluidType.ELECTRICITY,
                         'CO2': FluidType.CO2}

        # Replace elements in the list using the mapping
        result = [fluid_mapping[i] for i in fluid if i in fluid_mapping]

        return result


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


def assign_lifting(data_raw: dict) -> tuple | None:
    """
    A function to assign the lifting data to the list of Lifting dataclass.

    Parameters
    ----------
    data_raw: dict
        The dictionary containing the lifting data.

    Returns
    -------
    lifting_list: tuple | None
        The list containing the lifting dataclass.

    """
    # Defining the data source and the list container for Lifting. Then, assign them based on their fluid type
    lifting_data = data_raw['lifting']
    if lifting_data is None:
        return None
    else:
        lifting_list = []
        for key in lifting_data.keys():
            # Since the Lifting data for gas has different arguments input, conditional formatting is applied
            if 'Gas' in key or 'GSA' in key:
                lifting = Lifting(start_year=lifting_data[key]["start_year"],
                                  end_year=lifting_data[key]["end_year"],
                                  lifting_rate=np.array(lifting_data[key]["lifting_rate"]),
                                  price=np.array(lifting_data[key]["price"]),
                                  prod_year=np.array(lifting_data[key]["prod_year"]),
                                  fluid_type=read_fluid_type(lifting_data[key]["fluid_type"]),
                                  ghv=None if lifting_data[key]["ghv"] is None else
                                  np.array(lifting_data[key]["ghv"]),
                                  prod_rate=None if lifting_data[key]["prod_rate"] is None or "prod_rate" not in lifting_data[key]
                                  else np.array(lifting_data[key]["prod_rate"]),
                                  prod_rate_baseline=None if lifting_data[key]["prod_rate_baseline"] is None or "prod_rate_baseline" not in lifting_data[key]
                                  else np.array(lifting_data[key]["prod_rate_baseline"]),
                                  )

                lifting_list.append(lifting)

            else:
                lifting = Lifting(start_year=lifting_data[key]["start_year"],
                                  end_year=lifting_data[key]["end_year"],
                                  lifting_rate=np.array(lifting_data[key]["lifting_rate"]),
                                  price=np.array(lifting_data[key]["price"]),
                                  prod_year=np.array(lifting_data[key]["prod_year"]),
                                  fluid_type=read_fluid_type(lifting_data[key]["fluid_type"]),
                                  prod_rate=None if lifting_data[key]["prod_rate"] is None or "prod_rate" not in lifting_data[key]
                                  else np.array(lifting_data[key]["prod_rate"]),
                                  prod_rate_baseline=None if lifting_data[key]["prod_rate_baseline"] is None or "prod_rate_baseline" not in lifting_data[key]
                                  else np.array(lifting_data[key]["prod_rate_baseline"]),
                                  )
                lifting_list.append(lifting)

        return tuple(lifting_list)


def assign_cost(data_raw: dict) -> tuple:
    """
    A function to assign the cost data to the list of Tangible, Intangible, OPEX, ASR, LBT and CostOfSales dataclass.

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
    lbt_list
        A tuple containing the LBT dataclass.
    cos_list
        A tuple containing the CostOfSales dataclass.

    """
    # Defining the data source and the list container for Tangible. Then, assign them based on their fluid type
    capital_cost_data = data_raw['capital_cost']
    capital_cost_list = ([
        CapitalCost(
            start_year=capital_cost_data[key]['start_year'],
            end_year=capital_cost_data[key]['end_year'],
            cost=np.array(capital_cost_data[key]['cost']),
            expense_year=np.array(capital_cost_data[key]['expense_year']),
            cost_allocation=read_fluid_type(fluid=capital_cost_data[key]['cost_allocation']),
            description=capital_cost_data[key]['description'],
            tax_portion=np.array(capital_cost_data[key]['tax_portion']),
            tax_discount=np.array(capital_cost_data[key]['tax_discount']),
            pis_year=np.array(capital_cost_data[key]['pis_year']),
            salvage_value=np.array(capital_cost_data[key]['salvage_value']),
            useful_life=np.array(capital_cost_data[key]['useful_life']),
            depreciation_factor=np.array(capital_cost_data[key]['depreciation_factor']),
            is_ic_applied=capital_cost_data[key]['is_ic_applied'],
        ) for key in capital_cost_data.keys()
    ]) if capital_cost_data is not None else None

    # Defining the data source and the list container for Intangible. Then, assign them based on their fluid type
    intangible_data = data_raw['intangible']
    intangible_list = ([
        Intangible(
            start_year=intangible_data[key]['start_year'],
            end_year=intangible_data[key]['end_year'],
            cost=np.array(intangible_data[key]['cost']),
            expense_year=np.array(intangible_data[key]['expense_year']),
            cost_allocation=read_fluid_type(fluid=intangible_data[key]['cost_allocation']),
            description=intangible_data[key]['description'],
            tax_portion=np.array(intangible_data[key]['tax_portion']),
            tax_discount=np.array(intangible_data[key]['tax_discount']),
        ) for key in intangible_data.keys()
    ]) if intangible_data is not None else None

    # Defining the data source and the list container for Intangible. Then, them based on their fluid type
    opex_data = data_raw['opex']
    opex_list = ([
        OPEX(
            start_year=opex_data[key]['start_year'],
            end_year=opex_data[key]['end_year'],
            expense_year=np.array(opex_data[key]['expense_year']),
            cost_allocation=read_fluid_type(fluid=opex_data[key]['cost_allocation']),
            description=opex_data[key]['description'],
            tax_portion=np.array(opex_data[key]['tax_portion']),
            tax_discount=np.array(opex_data[key]['tax_discount']),
            fixed_cost=np.array(opex_data[key]['fixed_cost'])
        ) for key in opex_data.keys()
    ]) if opex_data is not None else None

    # Defining the data source and the list container for ASR. Then, them based on their fluid type
    asr_data = data_raw['asr']
    asr_list = ([
        ASR(
            start_year=asr_data[key]['start_year'],
            end_year=asr_data[key]['end_year'],
            cost=np.array(asr_data[key]['cost']),
            expense_year=np.array(asr_data[key]['expense_year']),
            cost_allocation=read_fluid_type(fluid=asr_data[key]['cost_allocation']),
            description=asr_data[key]['description'],
            final_year=np.array(asr_data[key]['final_year']),
            future_rate=np.array(asr_data[key]['future_rate']),
        ) for key in asr_data.keys()
    ]) if asr_data is not None else None

    # Defining the data source and the list container for ASR. Then, them based on their fluid type
    lbt_data = data_raw['lbt']
    lbt_list = ([
        LBT(
            start_year=asr_data[key]['start_year'],
            end_year=asr_data[key]['end_year'],
            cost=np.array(asr_data[key]['cost']),
            expense_year=np.array(asr_data[key]['expense_year']),
            cost_allocation=read_fluid_type(fluid=asr_data[key]['cost_allocation']),
            description=asr_data[key]['description'],
            tax_portion=np.array(asr_data[key]['tax_portion']),
            tax_discount=np.array(asr_data[key]['tax_discount']),
            final_year=np.array(asr_data[key]['final_year']),
            utilized_land_area=np.array(asr_data[key]['utilized_land_area']),
            utilized_building_area=np.array(asr_data[key]['utilized_building_area']),
            njop_land=np.array(asr_data[key]['njop_land']),
            njop_building=np.array(asr_data[key]['njop_building']),
            gross_revenue=np.array(asr_data[key]['gross_revenue']),
        ) for key in lbt_data.keys()
    ]) if lbt_data is not None else None

    # Defining the data source and the list container for Cost of Sales. Then, them based on their fluid type
    cos_data = data_raw['cost_of_sales']
    cos_list = ([
        CostOfSales(
            start_year=asr_data[key]['start_year'],
            end_year=asr_data[key]['end_year'],
            cost=np.array(asr_data[key]['cost']),
            expense_year=np.array(asr_data[key]['expense_year']),
            cost_allocation=read_fluid_type(fluid=asr_data[key]['cost_allocation']),
            description=asr_data[key]['description'],
            tax_portion=np.array(asr_data[key]['tax_portion']),
            tax_discount=np.array(asr_data[key]['tax_discount']),
        ) for key in cos_data.keys()
    ]) if cos_data is not None else None

    return capital_cost_list, intangible_list, opex_list, asr_list, lbt_list, cos_list


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


def load_data(dataset_type: str, contract_type: str = 'project') -> BaseProject | CostRecovery | GrossSplit:
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
    BaseProject | CostRecovery | GrossSplit

    """
    # Checking the availability of the dataset, if not available raise a Value Error
    # dataset_list = get_json_file_names()
    # contract_list = ['project', 'cost_recovery', 'gross_split']
    # if dataset_type not in dataset_list or contract_type not in contract_list:
    #     raise ValueError('Unknown dataset: "{0}", please check the Dataset and Contract Type that available.'
    #                      .format(dataset_type))

    # Reading the project json file
    data_raw = read_json_file(file_name=dataset_type)

    # Assigning the general config and converting them to datetime format.
    project_start_date = datetime.strptime(data_raw['start_date'], '%d/%m/%Y').date()
    project_end_date = datetime.strptime(data_raw['end_date'], '%d/%m/%Y').date()

    oil_onstream_date = assign_onstream_date(data_raw['oil_onstream_date'])
    gas_onstream_date = assign_onstream_date(data_raw['gas_onstream_date'])

    # Assigning the lifting and cost data.
    lifting_tuple = assign_lifting(data_raw)
    capital_tuple, intangible_tuple, opex_tuple, asr_tuple, lbt_tuple, cos_tuple = assign_cost(data_raw)

    if contract_type == 'project':
        return BaseProject(
            start_date=project_start_date,
            end_date=project_end_date,
            oil_onstream_date=oil_onstream_date,
            gas_onstream_date=gas_onstream_date,
            lifting=lifting_tuple,
            capital_cost=capital_tuple,
            intangible_cost=intangible_tuple,
            opex=opex_tuple,
            asr_cost=asr_tuple,
            lbt_cost=lbt_tuple,
            cost_of_sales=cos_tuple,
        )

    elif contract_type == 'cost_recovery':
        config = read_json_file(file_name=contract_type)
        return CostRecovery(
            start_date=project_start_date,
            end_date=project_end_date,
            oil_onstream_date=oil_onstream_date,
            gas_onstream_date=gas_onstream_date,
            lifting=lifting_tuple,
            capital_cost=capital_tuple,
            intangible_cost=intangible_tuple,
            opex=opex_tuple,
            asr_cost=asr_tuple,
            lbt_cost=lbt_tuple,
            cost_of_sales=cos_tuple,
            oil_ftp_is_available=config['oil_ftp_is_available'],
            oil_ftp_is_shared=config['oil_ftp_is_shared'],
            oil_ftp_portion=config['oil_ftp_portion'],
            gas_ftp_is_available=config['gas_ftp_is_available'],
            gas_ftp_is_shared=config['gas_ftp_is_shared'],
            gas_ftp_portion=config['gas_ftp_portion'],
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
            gas_dmo_holiday_duration=config['gas_dmo_holiday_duration']
        )

    elif contract_type == 'gross_split':
        config = read_json_file(file_name=contract_type)
        return GrossSplit(
            start_date=project_start_date,
            end_date=project_end_date,
            oil_onstream_date=oil_onstream_date,
            gas_onstream_date=gas_onstream_date,
            lifting=lifting_tuple,
            capital_cost=capital_tuple,
            intangible_cost=intangible_tuple,
            opex=opex_tuple,
            asr_cost=asr_tuple,
            lbt_cost=lbt_tuple,
            cost_of_sales=cos_tuple,
            field_status=config['field_status'],
            field_loc=config['field_loc'],
            res_depth=config['res_depth'],
            infra_avail=config['infra_avail'],
            res_type=config['res_type'],
            api_oil=config['api_oil'],
            domestic_use=config['domestic_use'],
            prod_stage=config['prod_stage'],
            co2_content=config['co2_content'],
            h2s_content=config['h2s_content'],
            base_split_ctr_oil=config['base_split_ctr_oil'],
            base_split_ctr_gas=config['base_split_ctr_gas'],
            split_ministry_disc=config['split_ministry_disc'],
            oil_dmo_volume_portion=config['oil_dmo_volume_portion'],
            oil_dmo_fee_portion=config['oil_dmo_fee_portion'],
            oil_dmo_holiday_duration=config['oil_dmo_holiday_duration'],
            gas_dmo_volume_portion=config['gas_dmo_volume_portion'],
            gas_dmo_fee_portion=config['gas_dmo_fee_portion'],
            gas_dmo_holiday_duration=config['gas_dmo_holiday_duration']
        )


def load_cost(
        filename: str,
        start_year: int = 2023,
        end_year: int = 2043,
        cost_allocation: FluidType = FluidType.OIL,
        template: str = "pyscnomics"
) -> tuple[CapitalCost, Intangible, OPEX, ASR] | ValueError:
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
        Capital
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
    capital_arr = np.array(df[[5, 7, 8, 9, 10, 11, 12]].sum(axis=1).to_numpy(dtype=float))
    capital = CapitalCost(
        start_year=start_year,
        end_year=end_year,
        cost=capital_arr,
        expense_year=years_arr,
        cost_allocation=[cost_allocation] * len(capital_arr)
    )

    # Assigning the Intangible data
    intangible_arr = df[6].to_numpy(dtype=float)
    intangible = Intangible(
        start_year=start_year,
        end_year=end_year,
        cost=intangible_arr,
        expense_year=years_arr,
        cost_allocation=[cost_allocation] * len(capital_arr)
    )

    # Assigning the ASR data
    asr_arr = df[18].to_numpy(dtype=float)
    asr = ASR(
        start_year=start_year,
        end_year=end_year,
        cost=asr_arr,
        expense_year=years_arr,
        cost_allocation=[cost_allocation] * len(capital_arr)
    )

    # Assigning the OPEX data
    fixed_cost_arr = df[[13, 14, 15, 16, 17]].sum(axis=1).to_numpy(dtype=float)
    opex = OPEX(
        start_year=start_year,
        end_year=end_year,
        fixed_cost=fixed_cost_arr,
        expense_year=years_arr,
        cost_allocation=[cost_allocation] * len(capital_arr)
    )

    return capital, intangible, opex, asr