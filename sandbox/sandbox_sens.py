import numpy as np
import pandas as pd
import time
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.Optimize.sensitivity import sensitivity_psc
from pyscnomics.dataset.sample import load_data
from pyscnomics.econ.selection import (NPVSelection, DiscountingMode, OptimizationTarget, FTPTaxRegime,
                                       OptimizationParameter)
from pyscnomics.tools.summary import get_summary

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

argument_summary = {'reference_year': reference_year,
                    'inflation_rate': inflation_rate,
                    'discount_rate': discount_rate,
                    'npv_mode': npv_mode,
                    'discounting_mode': discounting_mode}

# psc.run(tax_rate=tax_rate, ftp_tax_regime=FTPTaxRegime.PRE_PDJP_20_2017,
#         sunk_cost_reference_year=2021)
#
# psc_summary = get_summary(contract=psc,
#                           reference_year=2022,
#                           inflation_rate=0.1,
#                           discount_rate=0.1,
#                           npv_mode=NPVSelection.NPV_SKK_NOMINAL_TERMS,
#                           discounting_mode=DiscountingMode.END_YEAR)
# print(psc_summary['ctr_npv'])
# input()

# Trialing the sengs
start_time = time.time()
result = sensitivity_psc(steps=2,
                         diff=0.5,
                         contract=psc,
                         contract_arguments=argument_psc,
                         summary_arguments=argument_summary)
end_time = time.time()
print('Execution Time:', end_time - start_time)
