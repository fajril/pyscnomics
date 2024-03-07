import numpy as np
from datetime import datetime

from pyscnomics.dataset.sample import load_data
from pyscnomics.econ.selection import FTPTaxRegime, FluidType, InflationAppliedTo
from pyscnomics.econ.revenue import Lifting
from pyscnomics.econ.costs import Tangible, Intangible, OPEX, ASR
from pyscnomics.contracts.costrecovery import CostRecovery
from pyscnomics.contracts.grossplit import GrossSplit
from pyscnomics.contracts.transition import Transition


def generate_contract_sample(case: str) -> CostRecovery | GrossSplit | Transition:
    if case == 'CR_1':
        # Initiating the contract
        psc = load_data(dataset_type='case1', contract_type='cost_recovery')
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

        # Generating contract arguments
        contract_arguments = {'tax_rate': 0.424,
                              'ftp_tax_regime': FTPTaxRegime.PRE_PDJP_20_2017,
                              'sunk_cost_reference_year': 2021}

        psc.run(**contract_arguments)
        return psc

    elif case == 'CR_2':
        # Initiating the psc object
        psc = load_data(dataset_type='case2', contract_type='cost_recovery')

        # Editing the CostRecovery attribute as the corresponding case 1
        # FTP
        psc.oil_ftp_is_available = True
        psc.oil_ftp_is_shared = True
        psc.oil_ftp_portion = 0.2
        psc.gas_ftp_is_available = True
        psc.gas_ftp_is_shared = True
        psc.gas_ftp_portion = 0.05

        # Split Pre Tax
        psc.oil_ctr_pretax_share = 0.34722220
        psc.gas_ctr_pretax_share = 0.72083300

        # DMO
        psc.oil_dmo_volume_portion = 0.25
        psc.oil_dmo_fee_portion = 0.25
        psc.oil_dmo_holiday_duration = 60
        psc.gas_dmo_volume_portion = 0.25
        psc.gas_dmo_fee_portion = 1.0
        psc.gas_dmo_holiday_duration = 60

        # Generating contract arguments
        contract_arguments = {'tax_rate': 0.424,
                              'ftp_tax_regime': FTPTaxRegime.PRE_PDJP_20_2017,
                              'sunk_cost_reference_year': 2021}

        psc.run(**contract_arguments)
        return psc

    elif case == 'CR_3':
        # Initiating the psc object
        psc = load_data(dataset_type='case3', contract_type='cost_recovery')

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

        # Generating contract arguments
        contract_arguments = {'tax_rate': 0.424,
                              'ftp_tax_regime': FTPTaxRegime.PRE_PDJP_20_2017,
                              'sunk_cost_reference_year': 2021}

        psc.run(**contract_arguments)
        return psc

    elif case == 'GS_1':
        # Initiating the psc object
        psc = load_data(dataset_type='small_oil', contract_type='gross_split')
        tax_rate = np.array(
            [0.11, 0.11, 0.11, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12,
             0.12, 0.12])
        inflation_rate = np.array(
            [0, 0, 0.0, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
             0.02])

        contract_arguments = {'tax_rate': 0.22,
                              'vat_rate': tax_rate,
                              'inflation_rate': inflation_rate,
                              'inflation_rate_applied_to': InflationAppliedTo.CAPEX}

        psc.run(**contract_arguments)
        return psc

    elif case == 'Transition_1':
        # Defining Start Date and End Date
        psc_1_start_date = datetime.strptime("01/01/2018", '%d/%m/%Y').date()
        psc_1_end_date = datetime.strptime("22/4/2020", '%d/%m/%Y').date()

        # Defining the Gas lifting data
        psc1_gas_lifting = Lifting(
            start_year=2018,
            end_year=2020,
            lifting_rate=np.array([1.58446786291200, 0.98961346458056]),
            price=np.array([6.260, 6.260]),
            prod_year=np.array([2019, 2020]),
            ghv=np.array([1047.0, 1047.0]),
            fluid_type=FluidType.GAS)

        # Defining the Gas Tangible Data - Drilling Tangible
        psc1_gas_tangible = Tangible(
            start_year=2018,
            end_year=2020,
            cost=np.array([3363.67743703704000, 802.73224043715800]),
            expense_year=np.array([2019, 2020]),
            cost_allocation=[FluidType.GAS] * 2,
            pis_year=np.array([2019, 2020]),
            useful_life=np.array([5, 5]),
            depreciation_factor=np.array([0.25, 0.25]),
        )

        # Defining the Gas Intangible Data
        psc1_gas_intang = Intangible(
            start_year=2018,
            end_year=2020,
            cost=np.array([9532.633600000]),
            expense_year=np.array([2019]),
            cost_allocation=[FluidType.GAS])

        # Defining the Gas OPEX Data
        psc1_gas_opex_cost = OPEX(
            start_year=2018,
            end_year=2020,
            fixed_cost=np.array([2076.908222642980, 1297.582047244550]),
            expense_year=np.array([2019, 2020]),
            cost_allocation=[FluidType.GAS] * 2)

        # Defining the Gas ASR Data
        psc1_gas_asr_cost_opx = OPEX(
            start_year=2018,
            end_year=2020,
            fixed_cost=np.array([35.515809523809900, 10.965263596148900]),
            expense_year=np.array([2019, 2020]),
            cost_allocation=[FluidType.GAS] * 2)

        psc1_gas_asr_cost = ASR(
            start_year=2018,
            end_year=2020,
            cost=np.array([0]),
            expense_year=np.array([2019]),
            cost_allocation=[FluidType.GAS])

        # Parsing the fiscal terms into Cost Recovery
        psc1 = CostRecovery(
            start_date=psc_1_start_date,
            end_date=psc_1_end_date,
            lifting=tuple([psc1_gas_lifting]),
            tangible_cost=tuple([psc1_gas_tangible]),
            intangible_cost=tuple([psc1_gas_intang]),
            opex=tuple([psc1_gas_opex_cost, psc1_gas_asr_cost_opx]),
            asr_cost=tuple([psc1_gas_asr_cost]),
            oil_ftp_is_available=True,
            oil_ftp_is_shared=True,
            oil_ftp_portion=0.2,
            gas_ftp_is_available=True,
            gas_ftp_is_shared=True,
            gas_ftp_portion=0.2,
            oil_ctr_pretax_share=0.3361,
            gas_ctr_pretax_share=0.57692307692307700,
            oil_dmo_volume_portion=0.25,
            oil_dmo_fee_portion=0.15,
            oil_dmo_holiday_duration=0,
            gas_dmo_volume_portion=0.25,
            gas_dmo_fee_portion=0.15,
            gas_dmo_holiday_duration=0)

        """
        Gross Split
        """
        # Defining Start Date and End Date
        psc_2_start_date = datetime.strptime("23/4/2020", '%d/%m/%Y').date()
        psc_2_end_date = datetime.strptime("22/4/2030", '%d/%m/%Y').date()

        # Defining the Gas lifting data
        psc2_gas_lifting = Lifting(
            start_year=2020,
            end_year=2030,
            lifting_rate=np.array(
                [2.21568324370692000, 3.20769606628721000, 3.29284116326400000, 3.37370744832000000,
                 3.44718555313867000, 3.40400062841705000, 3.32543155814400000, 2.00043667046400000]),
            price=np.array([6.2600, 6.2600, 6.2600, 6.2600, 6.2600, 6.2600, 6.2600, 6.2600]),
            prod_year=np.array([2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027]),
            ghv=np.array([1047.0, 1047.0, 1047.0, 1047.0, 1047.0, 1047.0, 1047.0, 1047.0, ]),
            fluid_type=FluidType.GAS)

        # Defining the Gas Tangible Data - Drilling Tangible
        psc2_gas_tangible = Tangible(
            start_year=2020,
            end_year=2030,
            cost=np.array([1959.561038251370, 2834.780000000000]),
            expense_year=np.array([2020, 2025]),
            pis_year=np.array([2020, 2025]),
            useful_life=np.array([5, 5]),
            depreciation_factor=np.array([0.25, 0.25]),
            cost_allocation=[FluidType.GAS] * 2)

        # Defining the Gas Intangible Data
        psc2_gas_intang = Intangible(
            start_year=2020,
            end_year=2030,
            cost=np.array([0]),
            expense_year=np.array([2020]),
            cost_allocation=[FluidType.GAS])

        # Defining the Gas OPEX Data
        psc2_gas_opex_cost = OPEX(
            start_year=2020,
            end_year=2030,
            fixed_cost=np.array(
                [3137.4561521272200, 3834.8277754882300, 3943.8595100171400, 4040.8953543109500, 4114.6574450911200,
                 4272.4109557000400, 5057.6546904997800, 4462.7942473087100, ]),
            expense_year=np.array([2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027]),
            cost_allocation=[FluidType.GAS] * 8)

        psc2_gas_lbt_cost = OPEX(
            start_year=2020,
            end_year=2030,
            fixed_cost=np.array(
                [636.40338233533300, 1062.86476325927000, 1091.05051487026000, 1117.33210409363000, 1140.29666892703000,
                 1296.08012023605000, 1124.78850166335000, 786.84662704513200]),
            expense_year=np.array([2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027]),
            cost_allocation=[FluidType.GAS] * 8)

        # Defining VAT and Import Duty
        psc2_gas_vat = OPEX(
            start_year=2020,
            end_year=2030,
            fixed_cost=np.array(
                [396.5062479576550, 286.7125283653920, 294.7836073052500, 301.9666899432990, 307.4269319547460,
                 553.8846425099840, 377.2323444857470, 333.1977740376320, ]),
            expense_year=np.array([2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027]),
            cost_allocation=[FluidType.GAS] * 8)

        psc2_gas_import_duty = OPEX(
            start_year=2020,
            end_year=2030,
            fixed_cost=np.array(
                [159.965353, 55.922218, 57.496452, 58.897486, 59.962486, 227.566547, 73.577773, 64.988993]),
            expense_year=np.array([2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027]),
            cost_allocation=[FluidType.GAS] * 8)

        # Defining the Gas ASR Data
        psc2_gas_asr_cost_opx = OPEX(
            start_year=2020,
            end_year=2030,
            fixed_cost=np.array(
                [26.513186, 38.355043, 38.355043, 38.355043, 38.355043, 38.355043, 38.355043, 38.355043, ]),
            expense_year=np.array([2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027]),
            cost_allocation=[FluidType.GAS] * 8)

        psc2_gas_asr_cost = ASR(
            start_year=2020,
            end_year=2030,
            cost=np.array([0]),
            expense_year=np.array([2020]),
            cost_allocation=[FluidType.GAS])

        # Parsing the fiscal terms into Gross Split
        psc2 = GrossSplit(
            start_date=psc_2_start_date,
            end_date=psc_2_end_date,
            lifting=tuple([psc2_gas_lifting]),
            tangible_cost=tuple([psc2_gas_tangible]),
            intangible_cost=tuple([psc2_gas_intang]),
            opex=tuple([psc2_gas_opex_cost, psc2_gas_asr_cost_opx, psc2_gas_lbt_cost]),
            asr_cost=tuple([psc2_gas_asr_cost]),
            field_status='No POD',
            field_loc='Onshore',
            res_depth='<=2500',
            infra_avail='Well Developed',
            res_type='Conventional',
            api_oil='>=25',
            domestic_use='70<=x<100',
            prod_stage='Primer',
            co2_content='<5',
            h2s_content='<100',
            base_split_ctr_oil=0.43,
            base_split_ctr_gas=0.48,
            split_ministry_disc=0.0,
            oil_dmo_volume_portion=0.25,
            oil_dmo_fee_portion=1.0,
            oil_dmo_holiday_duration=0,
            gas_dmo_volume_portion=0.25,
            gas_dmo_fee_portion=1.0,
            gas_dmo_holiday_duration=0)

        ftp_tax_regime = FTPTaxRegime.DIRECT_MODE
        eff_tax_rate = 0.48
        # tax_payment_method = TaxPaymentMode.TAX_DIRECT_MODE
        argument_contract1 = {'ftp_tax_regime': ftp_tax_regime,
                              'tax_rate': eff_tax_rate,
                              # 'tax_payment_method': tax_payment_method
                              }

        argument_contract2 = {'tax_rate': 0.40}

        psc_trans = Transition(contract1=psc1,
                               contract2=psc2,
                               argument_contract1=argument_contract1,
                               argument_contract2=argument_contract2)
        psc_trans.run(unrec_portion=0)
        return psc_trans
