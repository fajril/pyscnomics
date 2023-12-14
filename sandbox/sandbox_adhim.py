import numpy as np
import pandas as pd
import time

from pyscnomics.dataset.sample import load_data
from pyscnomics.Optimize import optimization
from pyscnomics.econ.selection import OptimizationTarget
from pyscnomics.econ.selection import NPVSelection, DiscountingMode, OptimizationTarget, FTPTaxRegime

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
                'discount_rate_year': discount_rate_year,
                'ftp_tax_regime': ftp_tax_regime,
                'sunk_cost_reference_year': sunk_cost_reference_year
                }

dict_of_optimization = {'parameters': ['OIL_CTR_PRETAX', 'GAS_CTR_PRETAX', 'OIL_FTP_PORTION'],
                        'priority:': [1, 2, 3],
                        'min': [0.34722220, 0.5208330, 0.20],
                        'max': [0.4, 0.7, 0.25]}

target = 0.086

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

# Run Optimization
optimization.optimize_psc(dict_optimization=dict_of_optimization,
                          contract=psc,
                          contract_arguments=argument_psc,
                          target_optimization_value=target,
                          summary_argument=dict_summary,
                          target_parameter=OptimizationTarget.IRR)
