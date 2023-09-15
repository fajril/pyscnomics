"""
Handles calculations associated with PSC Gross Split.
"""

from dataclasses import dataclass, field
import numpy as np

from pyscnomics.contracts.project import BaseProject
from pyscnomics.econ.selection import FluidType
from pyscnomics.econ.costs import Tangible, Intangible, OPEX, ASR
from pyscnomics.econ.revenue import Lifting


@dataclass
class GrossSplit(BaseProject):
    field_status: str = field(default='No POD')
    field_loc: str = field(default='Onshore')
    res_depth: str = field(default='<=2500')
    infra_avail: str = field(default='Well Developed')
    res_type: str = field(default='Conventional')
    api_oil: str = field(default='<25')
    domestic_use: str = field(default='50<=x<70')
    prod_stage: str = field(default='Secondary')
    co2_content: str = field(default='<5')
    h2s_content: str = field(default='<100')

    base_split_ctr_oil: float = field(default=0.43)
    base_split_ctr_gas: float = field(default=0.48)
    split_ministry_disc: float = field(default=0.08)

    oil_dmo_volume_portion: float = field(default=0.25)
    oil_dmo_fee_portion: float = field(default=1.0)
    oil_dmo_holiday_duration: int = field(default=60)

    gas_dmo_volume_portion: float = field(default=1.0)
    gas_dmo_fee_portion: float = field(default=1.0)
    gas_dmo_holiday_duration: int = field(default=60)

    conversion_bboe2bscf: float = field(default=5.6)

    # Fields/attributes to be defined later
    _oil_lifting: Lifting = field(default=None, init=False, repr=False)
    _gas_lifting: Lifting = field(default=None, init=False, repr=False)
    _oil_tangible: Tangible = field(default=None, init=False, repr=False)
    _gas_tangible: Tangible = field(default=None, init=False, repr=False)
    _oil_intangible: Intangible = field(default=None, init=False, repr=False)
    _gas_intangible: Intangible = field(default=None, init=False, repr=False)
    _oil_opex: OPEX = field(default=None, init=False, repr=False)
    _gas_opex: OPEX = field(default=None, init=False, repr=False)
    _oil_asr: ASR = field(default=None, init=False, repr=False)
    _gas_asr: ASR = field(default=None, init=False, repr=False)

    _oil_revenue: np.ndarray = field(default=None, init=False, repr=False)
    _gas_revenue: np.ndarray = field(default=None, init=False, repr=False)

    _oil_depreciation: np.ndarray = field(default=None, init=False, repr=False)
    _gas_depreciation: np.ndarray = field(default=None, init=False, repr=False)
    _oil_undepreciated_asset: np.ndarray = field(default=None, init=False, repr=False)
    _gas_undepreciated_asset: np.ndarray = field(default=None, init=False, repr=False)
    _oil_total_expenses: np.ndarray = field(default=None, init=False, repr=False)
    _gas_total_expenses: np.ndarray = field(default=None, init=False, repr=False)

    _cumulative_prod: np.ndarray = field(default=None, init=False, repr=False)

    _oil_base_split: np.ndarray = field(default=None, init=False, repr=False)
    _gas_base_split: np.ndarray = field(default=None, init=False, repr=False)
    _variable_split: float = field(default=0, init=False, repr=False)
    _var_split_array: np.ndarray = field(default=None, init=False, repr=False)
    _oil_prog_split: np.ndarray = field(default=None, init=False, repr=False)
    _gas_prog_split: np.ndarray = field(default=None, init=False, repr=False)
    _oil_ctr_split: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ctr_split: np.ndarray = field(default=None, init=False, repr=False)
    _oil_ctr_share: np.ndarray = field(default=None, init=False, repr=False)
    _gas_ctr_share: np.ndarray = field(default=None, init=False, repr=False)
    _oil_gov_share: np.ndarray = field(default=None, init=False, repr=False)
    _gas_gov_share: np.ndarray = field(default=None, init=False, repr=False)

    def _get_aggregate(self):
        self._oil_lifting = self._get_oil_lifting()
        self._gas_lifting = self._get_gas_lifting()
        self._oil_tangible = self._get_oil_tangible()
        self._gas_tangible = self._get_gas_tangible()
        self._oil_intangible = self._get_oil_intangible()
        self._gas_intangible = self._get_gas_intangible()
        self._oil_opex = self._get_oil_opex()
        self._gas_opex = self._get_gas_opex()
        self._oil_asr = self._get_oil_asr()
        self._gas_asr = self._get_gas_asr()

    def _get_revenue(self):
        self._oil_revenue = self._oil_lifting.revenue()
        self._gas_revenue = self._gas_lifting.revenue()

    @staticmethod
    def _get_base_split(fluid: str) -> float:
        """
        Use to calculate base split of the given fluid type.
        Parameters
        ----------
        fluid: str
            The fluid type.

        Returns
        -------
        Value or the base split.
        """
        if fluid == 'oil':
            return 0.43
        elif fluid == 'gas':
            return 0.48
        else:
            raise ValueError('Unknown fluid type')

    def _get_var_split(self):
        statuses = {
            'POD I': 0.05,
            'POD II': 0.03,
            'No POD': 0,
        }
        locations = {
            'Onshore': 0,
            'Offshore (0<h<=20)': 0.08,
            'Offshore (20<h<=50)': 0.1,
            'Offshore (50<h<=150)': 0.12,
            'Offshore (150<h<=1000)': 0.14,
            'Offshore (h>1000)': 0.16,
        }
        depths = {
            '<=2500': 0,
            '>2500': 0.01,
        }
        infrastructures = {
            'Well Developed': 0,
            'New Frontier Offshore': 0.02,
            'New Frontier Onshore': 0.04,
        }
        reservoir_types = {
            'Conventional': 0,
            'Non Conventional': 0.16,
        }
        apis = {
            '<25': 0.01,
            '>=25': 0
        }
        tkdns = {
            '30<=x<50': 0.02,
            '50<=x<70': 0.03,
            '70<=x<100': 0.04,
        }
        stages = {
            'Primer': 0,
            'Secondary': 0.06,
            'Tertiary': 0.1,
        }
        co2s = {
            '<5': 0,
            '5<=x<10': 0.005,
            '10<=x<20': 0.01,
            '20<=x<40': 0.015,
            '40<=x<60': 0.02,
            'x>=60': 0.04,
        }
        h2ses = {
            '<100': 0,
            '100<=x<1000': 0.01,
            '1000<=x<2000': 0.02,
            '2000<=x<3000': 0.04,
            '3000<=x<4000': 0.04,
            'x>=4000': 0.05,
        }

        if self.field_status not in statuses:
            raise ValueError('unknown status for calculating var split')
        if self.field_loc not in locations:
            raise ValueError('unknown location for calculating var split')
        if self.res_depth not in depths:
            raise ValueError('unknown depth for calculating var split')
        if self.infra_avail not in infrastructures:
            raise ValueError('unknown infrastructure for calculating var split')
        if self.res_type not in reservoir_types:
            raise ValueError('unknown reservoir_type for calculating var split')
        if self.api_oil not in apis:
            raise ValueError('unknown api for calculating var split')
        if self.domestic_use not in tkdns:
            raise ValueError('unknown TKDN for calculating var split')
        if self.prod_stage not in stages:
            raise ValueError('unknown stage for calculating var split')
        if self.co2_content not in co2s:
            raise ValueError('unknown co2 for calculating var split')
        if self.h2s_content not in h2ses:
            raise ValueError('unknown h2s for calculating var split')

        # POD
        self._variable_split += statuses[self.field_status]
        # Location
        self._variable_split += locations[self.field_loc]
        # Depth
        self._variable_split += depths[self.res_depth]
        # Infrastructure
        self._variable_split += infrastructures[self.infra_avail]
        # Reservoir
        self._variable_split += reservoir_types[self.res_type]
        # Density
        self._variable_split += apis[self.api_oil]
        # TKDN
        self._variable_split += tkdns[self.domestic_use]
        # Stage
        self._variable_split += stages[self.prod_stage]
        # CO2
        self._variable_split += co2s[self.co2_content]
        # H2S
        self._variable_split += h2ses[self.h2s_content]

    @staticmethod
    def _calc_prog_split_08_2017(fluid: FluidType, price: float, cum: float):
        # Indonesia's Ministry Regulations No.08 The Year of 2017. At Appendix B Progressive Component
        if fluid == FluidType.OIL:
            if price < 40:
                ps = 0.075
            elif 40 <= price < 55:
                ps = 0.05
            elif 55 <= price < 70:
                ps = 0.025
            elif 70 <= price < 85:
                ps = 0.0
            elif 85 <= price < 100:
                ps = -0.025
            elif 100 <= price < 115:
                ps = -0.05
            elif 115 <= price:
                ps = -0.075
            else:
                ps = 0
        else:
            ps = 0

        # Cumulative Progressive Split
        if 0 < cum < 1:
            px = 0.05
        elif 1 <= cum < 10:
            px = 0.04
        elif 10 <= cum < 20:
            px = 0.03
        elif 20 <= cum < 50:
            px = 0.02
        elif 50 <= cum < 150:
            px = 0.01
        elif 150 <= cum:
            px = 0
        else:
            raise ValueError('No Regulation exist regarding the cumulative value')

        ps = ps + px
        return ps

    @staticmethod
    def _calc_prog_split_52_2017(fluid: FluidType, price: float, cum: float):
        # Indonesia's Ministry Regulations No.52 The Year of 2017. At Appendix B Progressive Component
        if fluid == FluidType.OIL:
            if price < 85:
                ps = (85 - price) * 0.25 / 100
            else:
                ps = 0

        elif fluid == FluidType.GAS:
            if price < 7:
                ps = (7 - price) * 2.5 / 100
            elif price > 10:
                ps = (10 - price) * 2.5 / 100
            else:
                ps = 0
        else:
            raise ValueError('Unknown fluid type')
        #
        # # Cumulative Progressive Split
        # if np.less(cum, 30):
        #     px = 0.1
        # elif np.logical_and(np.less_equal(30, cum), cum < np.less(cum, 60)):
        #     px = 0.09
        # elif np.logical_and(np.less_equal(60, cum), cum < np.less(cum, 90)):
        #     px = 0.08
        # elif np.logical_and(np.less_equal(90, cum), cum < np.less(cum, 125)):
        #     px = 0.06
        # elif np.logical_and(np.less_equal(125, cum), cum < np.less(cum, 175)):
        #     px = 0.04
        # elif np.greater(cum, 175):
        #     px = 0
        # else:
        #     raise ValueError('No Regulation exist regarding the cumulative value')
        #
        # ps = ps + px
        return ps

    def run(self):
        # Compiling the aggregate and revenue
        self._get_aggregate()
        self._get_revenue()

        # Depreciation (tangible cost)
        self._oil_depreciation, self._oil_undepreciated_asset = self._oil_tangible.psc_depreciation_rate()
        self._gas_depreciation, self._gas_undepreciated_asset = self._gas_tangible.psc_depreciation_rate()

        # Variable Split. -> Will set the value of _variable_split
        self._get_var_split()
        # Iterating through Oil Lifting and Gas Lifting

        # Base Split
        self._oil_base_split = np.full_like(self.project_years, fill_value=self.base_split_ctr_oil, dtype=float)
        self._gas_base_split = np.full_like(self.project_years, fill_value=self.base_split_ctr_oil, dtype=float)

        # Variable Split
        self._var_split_array = np.full_like(self.project_years, fill_value=self._variable_split, dtype=float)

        # Cumulative Production
        self._cumulative_prod = np.cumsum(self._oil_lifting.lifting_rate +
                                          (self._gas_lifting.lifting_rate / self.conversion_bboe2bscf))

        # Progressive Split
        vectorized_calc_prog_split = np.vectorize(self._calc_prog_split_52_2017)

        self._oil_prog_split = vectorized_calc_prog_split(
            fluid=self._oil_lifting.fluid_type,
            price=self._oil_lifting.price,
            cum=self._cumulative_prod
        )

        self._gas_prog_split = vectorized_calc_prog_split(
            fluid=self._gas_lifting.fluid_type,
            price=self._gas_lifting.price,
            cum=self._cumulative_prod
        )

        # Ministerial Discretion
        minis_disc_array = np.full_like(self.project_years, fill_value=self.split_ministry_disc, dtype=float)

        # Total Contractor Split
        self._oil_ctr_split = (self._oil_base_split + self._var_split_array + self._oil_prog_split +
                               minis_disc_array)
        self._gas_ctr_split = (self._gas_base_split + self._var_split_array + self._gas_prog_split +
                               minis_disc_array)

        # Contractor Share
        self._oil_ctr_share = self._oil_revenue * self._oil_ctr_split
        self._gas_ctr_share = self._gas_revenue * self._gas_ctr_split

        # Government Share
        self._oil_gov_share = self._oil_revenue - self._oil_ctr_share
        self._gas_gov_share = self._gas_revenue - self._gas_ctr_share

        # Total Investment
        self._oil_total_expenses = (self._oil_tangible.expenditures() + self._oil_intangible.expenditures() +
                                    self._oil_opex.expenditures() + self._oil_asr.expenditures())
        self._gas_total_expenses = (self._gas_tangible.expenditures() + self._gas_intangible.expenditures() +
                                    self._gas_opex.expenditures() + self._gas_asr.expenditures())

        # 4. Calculate Transfer

        # 5. Calculate DMO Volume

        # 6. Calculate DMO Fee

        # 7. Calculate Net DMO

        # 8. Parsing the DMO Holiday

        # 9. Calculate Taxable Income

        # 10. Calculate Net Entitlement

        # 11. Calculate Contractor Cashflow

        # 12. Calculate Government Take

        # 13. Calculate Discounted Cash Flow

        return NotImplementedError
