"""
This file is utilized to be the adapter of the router into the core codes.
"""

from pydantic.dataclasses import dataclass
from dataclasses import field

from pyscnomics.contracts.costrecovery import CostRecovery


@dataclass
class CostRecoveryBM:
    start_date: str
    end_date: str
    oil_onstream_date: str
    gas_onstream_date: str
    lifting: dict
    tangible: dict
    intangible: dict
    opex: dict
    asr: dict
    oil_ftp_is_available: bool = field(default=True)
    oil_ftp_is_shared: bool = field(default=True)
    oil_ftp_portion: float = field(default=0.2)
    gas_ftp_is_available: bool = field(default=True)
    gas_ftp_is_shared: bool = field(default=True)
    gas_ftp_portion: float = field(default=0.2)
    tax_split_type: str = field(default='conventional')
    condition_dict: dict = field(default_factory=dict)
    indicator_rc_icp_sliding: list | None = field(default=None)
    oil_ctr_pretax_share: float | list = field(default=0.25)
    gas_ctr_pretax_share: float | list = field(default=0.5)
    oil_ic_rate: float = field(default=0)
    gas_ic_rate: float = field(default=0)
    ic_is_available: bool = field(default=False)
    oil_cr_cap_rate: float = field(default=1.0)
    gas_cr_cap_rate: float = field(default=1.0)
    oil_dmo_volume_portion: float = field(default=0.25)
    oil_dmo_fee_portion: float = field(default=0.25)
    oil_dmo_holiday_duration: int = field(default=60)
    gas_dmo_volume_portion: float = field(default=1.0)
    gas_dmo_fee_portion: float = field(default=1.0)
    gas_dmo_holiday_duration: int = field(default=60)

    def generate_contract(self):
        contract = CostRecovery()
        oil_ftp_is_available=,
        oil_ftp_is_shared=,
        oil_ftp_portion=,
        gas_ftp_is_available=,
        gas_ftp_is_shared=,
        gas_ftp_portion=,
        tax_split_type=,
        condition_dict=,
        indicator_rc_icp_sliding=,
        oil_ctr_pretax_share=,
        gas_ctr_pretax_share=,
        oil_ic_rate=,
        gas_ic_rate=,
        ic_is_available=,
        oil_cr_cap_rate=,
        gas_cr_cap_rate=,
        oil_dmo_volume_portion=,
        oil_dmo_fee_portion=,
        oil_dmo_holiday_duration=,
        gas_dmo_volume_portion=,
        gas_dmo_fee_portion=,
        gas_dmo_holiday_duration=,




