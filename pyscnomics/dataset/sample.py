import json
import numpy as np
from datetime import datetime

from pyscnomics.contracts.project import BaseProject
# from pyscnomics.contracts.grosssplit import GrossSplit
# from pyscnomics.contracts.grosssplit import VariableSplit, SplitConfig, Tax, DMO, InvestmentCredit, Incentive

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


# def assign_gs(data_contract):
#     config = list(data_contract.values())
#     data_variable_split = config[0]
#     data_split_config = config[1]
#     data_tax = config[2]
#     data_dmo_oil = config[3]['oil']
#     data_dmo_gas = config[3]['gas']
#     data_investment_credit = config[4]
#     data_incentive = config[5]
#
#     variable_split = VariableSplit(field_status=data_variable_split['field_status'],
#                                    field_loc=data_variable_split['field_loc'],
#                                    res_depth=data_variable_split['res_depth'],
#                                    infra_avail=data_variable_split['infra_avail'],
#                                    res_type=data_variable_split['res_type'],
#                                    api_oil=data_variable_split['api_oil'],
#                                    domestic_use=data_variable_split['domestic_use'],
#                                    prod_stage=data_variable_split['prod_stage'],
#                                    co2_content=data_variable_split['co2_content'],
#                                    h2s_content=data_variable_split['h2s_content'],
#                                    )
#
#     split_config = SplitConfig(ctr_oil=data_split_config['ctr_oil'],
#                                ctr_gas=data_split_config['ctr_gas'],
#                                ministry_discr=data_split_config['ministry_discr'])
#
#     tax = Tax(corporate_income=data_tax['corporate_income'],
#               branch_profit=data_tax['branch_profit'], )
#
#     dmo_oil = DMO(holiday=data_dmo_oil['holiday'],
#                   period=data_dmo_oil['period'],
#                   start_production=data_dmo_oil['start_production'],
#                   volume=data_dmo_oil['volume'],
#                   fee=data_dmo_oil['fee'])
#
#     dmo_gas = DMO(holiday=data_dmo_gas['holiday'],
#                   period=data_dmo_gas['period'],
#                   start_production=data_dmo_gas['start_production'],
#                   volume=data_dmo_gas['volume'],
#                   fee=data_dmo_gas['fee'])
#
#     investment_credit = InvestmentCredit(oil_is_available=data_investment_credit['oil_is_available'],
#                                          oil_ic=data_investment_credit['oil_ic'],
#                                          gas_is_available=data_investment_credit['gas_is_available'],
#                                          gas_ic=data_investment_credit['gas_ic'], )
#
#     incentive = Incentive(vat_category=data_incentive['vat_categ'],
#                           vat_percent=data_incentive['vat_percent'],
#                           vat_discount=data_incentive['vat_discount'],
#                           lbt_discount=data_incentive['lbt_discount'],
#                           pdrd=data_incentive['pdrd'],
#                           pdrd_discount=data_incentive['pdrd_discount'],
#                           pdri_discount=data_incentive['pdri_discount'],
#                           import_duty=data_incentive['import_duty'],
#                           vat_import=data_incentive['vat_import'],
#                           pph_import=data_incentive['pph_import'],
#                           depre_accel=data_incentive['depre_accel'],
#                           depre_method=data_incentive['depre_method'])
#
#     return variable_split, split_config, tax, dmo_oil, dmo_gas, investment_credit, incentive


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
    data = read_json(filename=dataset + '.json')
    start_date, end_date, lifting, tangible, intangible, opex, asr = get_data(data)

    # Returning BaseProject dataclass for contract type: project
    if contract == 'project':
        result = BaseProject(start_date=start_date,
                             end_date=end_date,
                             lifting=lifting,
                             tangible_cost=tangible,
                             intangible_cost=intangible,
                             opex=opex,
                             asr_cost=asr)

    # Returning GrossSplit dataclass for contract type: gross_split
    # elif contract == 'gross_split':
    #     # Read the jason file for the project config
    #     data_config = read_json(filename=contract + '.json')
    #     variable_split, split_config, tax, dmo_oil, dmo_gas, investment_credit, incentive = assign_gs(data_config)
    #     result = GrossSplit(
    #         start_date=start_date,
    #         end_date=end_date,
    #         lifting=lifting,
    #         tangible_cost=tangible,
    #         intangible_cost=intangible,
    #         opex=opex,
    #         asr_cost=asr,
    #         variable_split=variable_split,
    #         split_config=split_config,
    #         tax=tax,
    #         dmo_oil=dmo_oil,
    #         dmo_gas=dmo_gas,
    #         incentive=incentive)

    # Returning CostRecovery dataclass for contract type: cost_recovery
    elif contract == 'cost_recovery':
        result = NotImplemented

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


if __name__ == "__main__":
    # Choosing the Dataset and contract type
    data_type = 'gas_real'
    project_type = 'project'

    # Testing the load_data function
    psc = load_data(dataset=data_type, contract=project_type)
    print('Output of the load_data module:')
    print(psc, '\n')

    psc_test = load_testing(dataset=data_type, key='FTP Gas')
    print('Output of the test:')
    print(psc_test, '\n')
