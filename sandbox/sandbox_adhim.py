import numpy as np
import pandas as pd
import time

from pyscnomics.dataset.sample import load_data
from pyscnomics.Optimize import optimization
from pyscnomics.econ.selection import (NPVSelection, DiscountingMode, OptimizationTarget, FTPTaxRegime,
                                       OptimizationParameter)

# pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# Initiating the psc object
psc = load_data(dataset_type='case1', contract_type='cost_recovery')

# Editing the CostRecovery attribute as the corresponding case 1
# FTP
psc.oil_ftp_is_available = True
psc.oil_ftp_is_shared = True
psc.oil_ftp_portion = 0.20
psc.gas_ftp_is_available = True
psc.gas_ftp_is_shared = True
psc.gas_ftp_portion = 0.20

# Split Pre Tax
psc.oil_ctr_pretax_share = 0.34722220
psc.gas_ctr_pretax_share = 0.5208330

# DMO
psc.oil_dmo_volume_portion = 0.25
psc.oil_dmo_fee_portion = 0.25
psc.oil_dmo_holiday_duration = 60
psc.gas_dmo_volume_portion = 0.25
psc.gas_dmo_fee_portion = 1
psc.gas_dmo_holiday_duration = 60

# PSC Arguments
tax_rate = 0.424
discount_rate_year = 2021
ftp_tax_regime = FTPTaxRegime.PRE_PDJP_20_2017
sunk_cost_reference_year = 2021
argument_psc = {'tax_rate': tax_rate,
                'ftp_tax_regime': ftp_tax_regime,
                'sunk_cost_reference_year': sunk_cost_reference_year
                }

# Summary Arguments
reference_year = 2022
inflation_rate = 0.1
discount_rate = 0.1
npv_mode = NPVSelection.NPV_SKK_NOMINAL_TERMS
discounting_mode = DiscountingMode.END_YEAR

dict_summary = {'reference_year': reference_year,
                'inflation_rate': inflation_rate,
                'discount_rate': discount_rate,
                'npv_mode': npv_mode,
                'discounting_mode': discounting_mode}

# # Optimization 1
# optimized_params = [OptimizationParameter.OIL_CTR_PRETAX,
#                     OptimizationParameter.GAS_CTR_PRETAX,
#                     OptimizationParameter.OIL_FTP_PORTION,
#                     OptimizationParameter.GAS_FTP_PORTION,
#                     OptimizationParameter.OIL_IC,
#                     OptimizationParameter.GAS_IC,
#                     OptimizationParameter.OIL_DMO_FEE,
#                     OptimizationParameter.GAS_DMO_FEE,
#                     OptimizationParameter.VAT_RATE,
#                     OptimizationParameter.EFFECTIVE_TAX_RATE]
#
# params_priority = list(range(1, 11))
#
# base_parameter = np.array([0.34722220, 0.5208330, 0.20, 0.20, 0, 0, 0.25, 1, 0, 0.424])
# min_parameter = base_parameter * 0.7
# max_parameter = np.where(base_parameter == 1,
#                          1,
#                          base_parameter * 1.4)
#
# dict_of_optimization = {'parameters': optimized_params,
#                         'priority:': params_priority,
#                         'min': min_parameter,
#                         'max': max_parameter}
#
# target = 0.10

# # Optimization 2
# optimized_params = [OptimizationParameter.OIL_CTR_PRETAX,
#                     OptimizationParameter.GAS_CTR_PRETAX,
#                     OptimizationParameter.OIL_FTP_PORTION,
#                     OptimizationParameter.GAS_FTP_PORTION, ]
#
# params_priority = list(range(1, 5))
#
# base_parameter = np.array([0.34722220, 0.5208330, 0.20, 0.20])
# min_parameter = base_parameter * 0.7
# max_parameter = np.where(base_parameter == 1,
#                          1,
#                          base_parameter * 1.3)
#
#
# dict_of_optimization = {'parameters': optimized_params,
#                         'priority:': params_priority,
#                         'min': min_parameter,
#                         'max': max_parameter}
#
# target = 0.09

# # Optimization 3
# optimized_params = [OptimizationParameter.EFFECTIVE_TAX_RATE, OptimizationParameter.EFFECTIVE_TAX_RATE,]
#
# params_priority = [1]
#
# base_parameter = np.array([0.424, 0.34722220])
# min_parameter = np.array([0.40, 0.2])
# max_parameter = np.array([0.44, 0.4])
#
#
# dict_of_optimization = {'parameters': optimized_params,
#                         'priority:': params_priority,
#                         'min': min_parameter,
#                         'max': max_parameter}
#
# target = 0.08090382832197131

# Optimization 4
optimized_params = [
    OptimizationParameter.VAT_RATE,
    OptimizationParameter.EFFECTIVE_TAX_RATE,
    OptimizationParameter.OIL_FTP_PORTION,
    OptimizationParameter.GAS_FTP_PORTION,
    OptimizationParameter.OIL_CTR_PRETAX,
    OptimizationParameter.GAS_CTR_PRETAX,
    OptimizationParameter.OIL_IC,
    OptimizationParameter.GAS_IC,
    OptimizationParameter.OIL_DMO_FEE,
    OptimizationParameter.GAS_DMO_FEE,
]

params_priority = list(range(1, 11))

base_parameter = np.array([0, 0.424, 0.20, 0.20, 0.34722220, 0.5208330, 0, 0, 0.25, 1])
min_parameter = base_parameter * 0.7
max_parameter = np.where(base_parameter == 1,
                         1,
                         base_parameter * 1.4)

dict_of_optimization = {'parameters': optimized_params,
                        'priority:': params_priority,
                        'min': min_parameter,
                        'max': max_parameter}

target = 0.0


start_time = time.time()
optim_result = optimization.optimize_psc(dict_optimization=dict_of_optimization,
                                         contract=psc,
                                         contract_arguments=argument_psc,
                                         target_optimization_value=target,
                                         summary_argument=dict_summary,
                                         target_parameter=OptimizationTarget.NPV)
end_time = time.time()
print('Execution Time:', end_time - start_time, '\n')

df = pd.DataFrame()
df['Min Parameters'] = min_parameter
df['Base Parameters'] = base_parameter
df['Max Parameters'] = max_parameter
df['Optimized Parametes'] = optim_result[0]
df['Parameter Value'] = optim_result[1]

print(df, '\n')
print('Target Value:', target)
print('Optimized Target Value:', optim_result[2])
print('Parameter Value Value:', optim_result[1])
