# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 08:16:58 2023
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from datetime import datetime

# from app.contract.general import GeneralContract
# from app.contract.cost import Cost
# from model.config import General

from pyscnomics.contracts.project import BaseProject
from pyscnomics.econ.revenue import Lifting

class CostRecovery(BaseProject):
    ftp_is_available: bool = field(default=True, repr=False)
    ftp_is_shared: bool = field(default=True, repr=False)
    ftp_portion: float = field(default=None, repr=False)
    pretax: tuple[float, float] = field(default=(None, None), repr=False)

    # def __init__(self,
    #              start_discount_year: int = None,
    #              tax_ef: float = None,
    #              ppn_mode: str = None,
    #              ppn_rate: float = None,
    #              ppn_discount: float = None,
    #              pdri: float = None,
    #              pdri_discount: float = None,
    #              pdrd: float = None,
    #              pdrd_discount: float = None,
    #              pbb_discount: float = None,

    #              dmoh_month: int = None,
    #              dmo_start: float = None,
    #              dmo_portion: float = None,
    #              dmo_fee_rate: float = None,

    #              ic_rate: float = None,
    #              is_depreciation_accelerated: bool = False,
    #              depreciation_method: str = None,
    #              pretax: float = None,

    #              ftp_is_available: bool = False,  # 'Yes' or 'No' , Yes if FTP available
    #              ftp_is_shared: bool = False,  # Yes' or 'No', Yes if 'Shared'
    #              ftp_portion: float = None,

    #              cr_cap_rate: float = None,

    #              # Revenue Data
    #              revenue: np.ndarray = None,

    #              # Depreciable Data
    #              depreciable: np.ndarray = None,

    #              # Cost Data (OC, ASR and LBT)
    #              cost_obj: Cost = None,

    #              # Phase
    #              phase: str = None):
    #     """
    #     """
    #     super().__init__(
    #                      start_discount_year=start_discount_year,
    #                      tax_ef=tax_ef,
    #                      ppn_mode=ppn_mode,
    #                      ppn_rate=ppn_rate,
    #                      ppn_discount=ppn_discount,
    #                      pdri=pdri,
    #                      pdri_discount=pdri_discount,
    #                      pdrd=pdrd,
    #                      pdrd_discount=pdrd_discount,
    #                      pbb_discount=pbb_discount,

    #                      # DMO Data
    #                      dmoh_month = dmoh_month,
    #                      dmo_start = dmo_start,
    #                      dmo_portion=dmo_portion,

    #                      # DMO Fee Data (Depend on phase)
    #                      dmo_fee_rate = dmo_fee_rate,

    #                      # IC Data (Depend on phase)
    #                      ic_rate = ic_rate,

    #                      # Depreciation Method
    #                      is_depreciation_accelerated=is_depreciation_accelerated,
    #                      depreciation_method = depreciation_method,

    #                      # Revenue Data (Depend on phase)
    #                      revenue = revenue,

    #                      # Depreciable Data (Depend on phase)
    #                      depreciable = depreciable,

    #                      # Operating Cost Data (Depend on phase)
    #                      cost_obj = cost_obj)

    #     # Phase: "Oil" or "Gas"
    #     self.phase = phase

    #     # Pre-Tax Split (Depend on phase)
    #     self.pretax = pretax

    #     # FTP Data
    #     self.ftp_is_available = ftp_is_available
    #     self.ftp_is_shared = ftp_is_shared
    #     self.ftp_portion = ftp_portion

    #     # CR Capped Rate Data
    #     self.cr_cap_rate = cr_cap_rate


    def _get_FTP(self, revenue):
        r"""
        Use to calculate First Trench Petroleum (FTP).

        * :math:`FTP_{available} = False`

            .. math::

                FTP_{con \, (t)} &= 0 \\
                FTP_{gov \, (t)} &= 0 \\
                FTP_{total \, (t)} &= 0

        * :math:`FTP_{available} = True`

            .. math::

                FTP_{total \, (t)} = FTP_{percent} \times Revenue_{(t)}

            * :math:`FTP_{shared} = False`

                .. math::

                    FTP_{con \, (t)} &= 0 \\
                    FTP_{gov \, (t)} &= FTP_{total \, (t)}

            * :math:`FTP_{shared} = True`

                .. math::

                    FTP_{con \, (t)} &= pretax \times FTP_{total \, (t)} \\
                    FTP_{gov \, (t)} &= (1 - pretax) \times FTP_{total \, (t)}

        Parameters
        ----------
        self :
            * phase : str
                fluid type (exp. "Gas" or "Oil")

            * ftp_is_available : bool
                FTP is available (exp. True or False)

            * ftp_portion : float
                FTP portion

        Returns
        -------
        out : tup
            A tuple of Contractor FTP, Government FTP, Total FTP, and
            Cumulative FTP values
        """
        if not self.ftp_is_available:
            return np.squeeze(np.hsplit(np.zeros(revenue.size, 4), 4))
        else:
            ftp_arr = self.ftp_portion * revenue

            if self.ftp_is_shared:
                ftp_con_arr = self.pretax * ftp_arr
                ftp_gov_arr = (1 - self.pretax) * ftp_arr
            else:
                ftp_con_arr = np.zeros_like(revenue)
                ftp_gov_arr = np.copy(ftp_arr)

            return ftp_con_arr, ftp_gov_arr, ftp_arr


#     def _get_REV_after_FTP(self, FTP):
#         r"""
#         Use to calculate Revenue after FTP.

#         .. math::

#             \text{REV after FTP}_{(t)} = REV_{(t)} - FTP_{(t)}

#         Parameters
#         ----------
#         FTP : np.ndarray
#             first trench petroleum
#         self :
#             * revenue : np.ndarray
#                 revenue

#         Return
#         ------
#         rev_after_ftp : np.ndarray
#             Revenue after FTP values
#         """
#         return self.revenue - FTP


#     def _get_IC(self):
#         r"""
#         Use to calculate Investment Credit (IC).

#         .. math::

#             IC_{(t)} = IC_{rate} \times Investment_{depr \, (t)}

#         Parameters
#         ----------
#         self :
#             * ic_rate : float
#                 investment credit rate
#             * depreciable : np.ndarray
#                 depreciable values

#         Return
#         ------
#         ic : np.ndarray
#             investment credit values
#         """
#         return self.ic_rate * self.depreciable


#     def _get_ICpaid_ICunrec(self, ic, rev_after_ftp):
#         r"""
#         Use to calculate Investment Credit Paid (IC Paid) and Unrecovered (IC Unrec).

#         * IC Paid

#         .. math::

#             IC_{paid \, (t)} = \left\{\begin{matrix}
#                             \min \left( \text{REV After FTP}_{(t)}, IC_{paid \, (t)}\right), & t = 0 \\
#                             \min \left( \text{REV After FTP}_{(t)}, IC_{paid \, (t)} + IC_{unrec \, (t-1)}\right), & t > 0
#                             \end{matrix}\right.

#         * IC Unrec

#         .. math::
#             IC_{unrec \, (t)} =\left\{\begin{matrix}
#                 IC_{(t)} - IC_{paid \, (t)}, & t = 0 \\
#                 IC_{(t)} - IC_{paid \, (t)} + IC_{unrec \, (t - 1)}, & t>0
#                 \end{matrix}\right.

#         Parameters
#         ----------
#         ic : np.ndarray
#             investment credit
#         rev_after_ftp : np.ndarray
#             revenue after first trench petroleum

#         Returns
#         -------
#         out: tup
#             * ic_paid : np.ndarray
#                 paid investment credit
#             * ic_unrec : np.ndarray
#                 unrecovered investment credit
#         """
#         # Prepare storage for IC Paid and IC UnRec
#         ic_paid = np.zeros_like(ic)
#         ic_unrec = np.zeros_like(ic)

#         # Calculate the IC Paid and IC UnRec values.
#         ic_paid[0] = min(rev_after_ftp[0], ic[0])
#         ic_unrec[0] = ic[0] - ic_paid[0]

#         for idx in range(1, len(ic)):
#             ic_paid[idx] = min(rev_after_ftp[idx], ic[idx] + ic_unrec[idx - 1])
#             ic_unrec[idx] = ic[idx] - ic_paid[idx] + ic_unrec[idx - 1]

#         return ic_paid, ic_unrec


#     def _get_REV_after_IC(self, rev_after_ftp, ic_paid):
#         r"""
#         Use to calculate revenue after investment credit (rev_after_IC).

#         .. math::

#             \text{REV after IC}_{(t)} = \text{REV after FTP}_{(t)} - IC_{paid \, (t)}

#         Parameters
#         ----------
#         rev_after_ftp : np.ndarray
#             revenue after ftp
#         ic_paid : np.ndarray
#             paid investment credit

#         Return
#         ------
#         rev_after_ic : np.ndarray
#             revenue after ftp and ic
#         """
#         return rev_after_ftp - ic_paid


#     def _get_CR_allocation(self, rev_after_ic):
#         r"""
#         Use to calculate cost recovery allocation (CR allocation).

#         .. math::

#             CR_{allocation \, (t)} = CR_{cap \, rate} \times \text{REV after IC}_{(t)}

#         Parameters
#         ----------
#         rev_after_ic : np.ndarray
#             revenue after ftp and ic

#         Returns
#         -------
#         cr_allocation: np.ndarray
#             cost recovery allocation

#         Notes
#         -----
#         * :math:`CR_{cap \, rate} = 1 \rightarrow` Uncapped
#         * :math:`CR_{cap \, rate} < 1 \rightarrow` Capped
#         """
#         return self.cr_cap_rate * rev_after_ic


#     def _get_CR(self, depr_capex, undepr_capex, opex, intangible, sunk_cost):
#         r"""
#         Use to calculate cost recovery (CR).

#         .. math::

#             CR_{(t)} = \text{Depr. Capex}_{(t)} + \text{Undepr. Capex}_{(t)}
#                 + OPEX_{(t)} + Intangible_{(t)} + \text{Sunk Cost}_{(t)}

#         Parameters
#         ----------
#         depr_capex : np.ndarray
#             depreciation in capital expenditure
#         undepr_capex : np.ndarray
#             undepreciation in capital expenditure
#         opex : np.ndarray
#             operating expenditure
#         intangible : np.ndarray
#             intangible cost
#         sunk_cost : np.ndarray
#             sunk cost

#         Returns
#         -------
#         CR : np.ndarray
#             cost recovery
#         """
#         return depr_capex + undepr_capex + opex + intangible + sunk_cost


#     def _get_CTR_UR(self, CR, CR_allocation):
#         r"""
#         Use to calculate cost to be recovered (CTR) and unrecovered (UR).

#         * Cost to be Recovered (CTR)

#             * :math:`\text{REV after IC}_{(t)} \leq CR_{(t)}`

#             * :math:`\text{REV after IC}_{(t)} > CR_{(t)}`


#         * Unrecovered (UR)

#             .. math::

#                 UR_{(t)} = UR_{(t - 1)} + CR_{(t)} - CTR_{(t)}

#         Parameters
#         ----------
#         CR : np.ndarray
#             cost recovery
#         CR_allocation: np.ndarray
#             cost recovery allocation

#         Returns
#         -------
#         out : tup
#             * ctr_arr : np.ndarray
#                 cost to be recovered (CTR)
#             * ur_arr : np.ndarray
#                 unrecovered (UR)
#         """
#         ctr_arr = np.zeros_like(CR)
#         ur_arr = np.zeros_like(CR)

#         for idx, val in enumerate(CR):
#             if idx > 0:
#                 if CR_allocation[idx] <= CR[idx]:
#                     ctr_arr[idx] = min(CR_allocation[idx], CR[idx])
#                 else:
#                     if ur_arr[idx - 1] == 0:
#                         ctr_arr[idx] = CR[idx]
#                     else:
#                         ctr_arr[idx] = CR[idx] \
#                             + min(CR_allocation[idx] - CR[idx], ur_arr[idx - 1])

#                 ur_arr[idx] = ur_arr[idx - 1] + CR[idx] - ctr_arr[idx]

#         return ctr_arr, ur_arr


#     def _get_ETS(self, REV_after_IC, CR):
#         r"""
#         Use to calculate equity to be share (ETS) before Transfer.

#         .. math::

#             \text{ETS PreTransfer}_{(t)} = \text{REV after IC}_{(t)} - CR_{(t)}

#         Parameters
#         ----------
#         REV_after_IC : np.ndarray
#             revenue after ftp and ic
#         CR : np.ndarray
#             cost recovery

#         Return
#         ------
#         ETS_PreTransfer : np.ndarray
#             ETS PreTransfer
#         """
#         return REV_after_IC - CR


#     def _get_Transfer(self, UR_gas, UR_oil, ETS_PreTransfer_gas,
#                       ETS_PreTransfer_oil):
#         r"""
#         Use to calculate Transfer between oil to gas or vice versa.

#         Parameters
#         ----------
#         UR_gas : np.ndarray
#             unrecovered gas
#         UR_oil : np.ndarray
#             unrecovered oil
#         ETS_PreTransfer_gas : np.ndarray
#             equity to be split pre-transfer gas
#         ETS_PreTransfer_oil : np.ndarray
#             equity to be split pre-transfer oil

#         Returns
#         -------
#         Trf2oil : np.ndarray
#             transfer to oil from gas
#         Trf2gas : np.ndarray
#             transfer to gas from oil
#         """
#         Trf2oil = np.zeros_like(UR_oil)
#         Trf2gas = np.zeros_like(UR_gas)

#         # Transfer to oil
#         combined_condition_oil = np.greater(UR_oil, 0) & np.equal(UR_gas, 0)
#         indices_oil = np.argwhere(combined_condition_oil)

#         if np.size(indices_oil) > 0:
#             Trf2oil[indices_oil] = \
#                 np.minimum(ETS_PreTransfer_gas[indices_oil],
#                            UR_oil[indices_oil])

#         # Transfer to gas
#         combined_condition_gas = np.equal(UR_oil, 0) & np.greater(UR_gas, 0)
#         indices_gas = np.argwhere(combined_condition_gas)

#         if np.size(indices_gas) > 0:
#             Trf2gas[indices_gas] = \
#                 np.minimum(ETS_PreTransfer_oil[indices_gas],
#                            UR_gas[indices_gas])

#         return Trf2oil, Trf2gas


#     def _get_UR_After_Transfer(self, UR_PreTransfer, Trfto):
#         r"""
#         Use to calculate Unrecovered after transfer.

#         .. math::

#             \text{UR After Transfer}_{(t)} = \text{UR PreTransfer}_{(t)} -
#                 \text{Transfer To}_{(t)}

#         Parameters
#         ----------
#         UR_PreTransfer : np.ndarray
#             unrecovered before transfer
#         Trfto : np.ndarray
#             transfer to value

#         Return
#         ------
#         UR_After_Transfer : np.ndarray
#             unrecovered after transfer
#         """
#         return UR_PreTransfer - Trfto


#     def _get_ETS_After_Transfer(self, ETS_PreTransfer, Trfto,
#                                 UR_After_Transfer):
#         r"""
#         Use to calculate Equity to be Split after transfer.

#         .. math::

#             \text{ETS After Transfer}_{(t)} = \text{ETS PreTransfer}_{(t)} +
#                 \text{Transfer To}_{(t)}

#         Parameters
#         ----------
#         ETS_PreTransfer : np.ndarray
#             equity to be split before transfer
#         Trfto : np.ndarray
#             transfer to value
#         UR_After_Transfer : np.ndarray
#             unrecovered after transfer

#         Return
#         ------
#         ETS_After_Transfer : np.ndarray
#             equity to be split after transfer
#         """
#         ETS_After_Transfer = np.zeros_like(ETS_PreTransfer)

#         indices = np.equal(UR_After_Transfer, 0)

#         if np.size(indices) > 0:
#             ETS_After_Transfer[indices] = ETS_PreTransfer[indices] \
#                                           + Trfto[indices]

#         return ETS_After_Transfer


#     def _get_ES(self, ETS):
#         r"""
#         Use to calculate equity share (ES).

#         .. math::

#             ES_{con \, (t)} &= pretax \times ETS_{(t)} \\
#             ES_{gov \, (t)} &= (1 - pretax) \times ETS_{(t)}

#         Parameters
#         ----------
#         fluid : str
#             fluid type (exp. "Gas" or "Oil")
#         ETS : np.ndarray
#             equity to be shared

#         Raise
#         -----
#         Exception
#             if the fluid type is not valid.

#         Returns
#         -------
#         out: tup
#             * ES_con : np.ndarray
#                 Equity share contractor
#             * ES_gov : np.ndarray
#                 Equity share goverment
#         """
#         ES_con = self.pretax * ETS
#         ES_gov = (1 - self.pretax) * ETS

#         return ES_con, ES_gov


#     def _get_CS(self, ftp_con, ets_con, ic_paid):
#         r"""
#         Use to calculate Contractor Share (CS).

#         .. math::

#             CS_{(t)} = FTP_{con \, (t)} + ETS_{con \, (t)} + IC_{paid \, (t)}

#         Parameters
#         ----------
#         ftp_con : np.ndarray
#             contractor FTP
#         ets_con : np.ndarray
#             contractor ETS
#         ic_paid : np.ndarray
#             paid investment credit

#         Return
#         ------
#         CS : np.ndarray
#             contractor share
#         """
#         return ftp_con + ets_con + ic_paid


#     def _get_GS(self, ftp_gov, ets_gov):
#         r"""
#         Use to calculate Goverment Share (GS).

#         .. math::

#             GS_{(t)} = FTP_{gov \, (t)} + ETS_{gov \, (t)}

#         Parameters
#         ----------
#         ftp_gov : np.ndarray
#             goverment FTP
#         ets_gov : np.ndarray
#             goverment ETS

#         Return
#         ------
#         GS : np.ndarray
#             goverment share
#         """
#         return ftp_gov + ets_gov

#     #TODO: Perlu di test lebih lanjut untuk DMO
#     def _get_DMO(self, ES_con, year_arr):
#         r"""
#         Use to calculate domestic market obligation (DMO)

#         Parameters
#         ----------
#         fluid : str
#             fluid type (exp. "Gas" or "Oil")
#         ES_con : np.ndarray
#             equity to split contractor
#         prod_date : str
#             production date (DD-MM-YYYY)
#         year_arr : np.ndarray
#             year value

#         Raise
#         -----
#         Exception
#             if the fluid type is not valid.

#         Return
#         ------
#         DMO : np.ndarray
#             domestic market obligation

#         Reference
#         ---------
#         [1] PTK 59 Rev 1/2021 p3.5.2
#         """
#         if self.phase == "Gas":
#             return np.zeros_like(self.revenue)
#         elif self.phase == "Oil":
#             DMO_arr = np.zeros_like(self.revenue)

#             # Convert the date string to a datetime object
#             start_prod_year = datetime.strptime(self.dmo_start, \
#                                                 '%Y-%m-%d').year

#             # Get after holiday periode
#             start_dmo_year = start_prod_year + (self.dmoh_month / 12)

#             # Calculate DMO
#             indices = np.argwhere(year_arr == start_dmo_year)

#             if np.size(indices) > 0:
#                 DMO_arr = min(self.dmo_portion * self.pretax \
#                               * self.revenue[indices], ES_con[indices])

#             return DMO_arr
#         else:
#             raise Exception('Error in get_DMO_Holiday (Unknown fluid type)')


#     def _get_DMO_Fee(self, DMO):
#         r"""
#         Use to calculate domestic market obligation fee (DMO Fee).

#         .. math::

#             DMO_{fee \, (t)} = DMO_{fee \, rate} \times DMO_{(t)}

#         Parameters
#         ----------
#         DMO : np.ndarray
#             domestic obligation market
#         self :
#             * dmo_fee_rate : float
#                 DMO fee rate

#         Return
#         ------
#         DMO_Fee : np.ndarray
#             domestic market obligation fee
#         """
#         return self.dmo_fee_rate * DMO


#     def _get_DDMO(self, DMO, DMO_Fee):
#         r"""
#         Use to calculate differential domestic market obligation (DDMO).

#         .. math::

#             DDMO_{(t)} = DMO_{(t)} - DMO_{fee \, (t)}

#         Parameters
#         ----------
#         DMO : np.ndarray
#             domestic market obligation
#         DMO_Fee : np.ndarray
#             domestic market obligation fee

#         Return
#         ------
#         DDMO : np.ndarray
#             differential domestic market obligation
#         """
#         return DMO - DMO_Fee


#     def _get_TI(self, CS, DDMO):
#         r"""
#         Use to calculate taxable income (TI).

#         .. math::

#             TI_{(t)} = CS_{(t)} - DDMO_{(t)}

#         Parameters
#         ----------
#         CS : np.ndarray
#             contractor share
#         DDMO : np.ndarray
#             differential domestic market obligation

#         Return
#         ------
#         TI : np.ndarray
#             taxable income
#         """
#         return CS - DDMO


#     def _get_tax1(self, TI, FTP_con, UR):
#         r"""
#         Use to calculate tax (tax).

#         .. math::

#             Tax_{(t)} = \left\{\begin{matrix}
#                 0, & \sum_{i=0}^{t} FTP_{con} > UR_{(t)} \\
#                 \text{tax rate} \times TI_{(t)}, & \sum_{i=0}^{t} FTP_{con} \leq UR_{(t)}
#                 \end{matrix}\right.

#         Parameters
#         ----------
#         TI: np.ndarray
#             taxable income
#         FTP_con : np.ndarray
#             contractor FTP
#         UR : np.ndarray
#             unrecovered

#         Return
#         ------
#         tax: float
#             contractor tax
#         """
#         tax = np.zeros_like(TI)

#         # Calculate cumulative FTP Contractor
#         cum_FTP_con = np.cumsum(FTP_con)

#         # Calculate tax where Cum FTP_con <= UR
#         indices = np.argwhere(cum_FTP_con <= UR)
#         tax[indices] = self.tax_ef * TI[indices]

#         return tax


#     def _get_tax2(self, TI, ES_con):
#         r"""
#         Use to calculate tax (tax).

#         .. math::



#         Parameters
#         ----------
#         TI: np.ndarray
#             taxable income
#         ES_con : np.ndarray
#             equity to be share contractor FTP

#         Returns
#         -------
#         out: tup
#             * tax_paid: float
#                 paid contractor tax
#             * tax_carryforward: float
#                 carryforward contractor tax
#         """
#         TI_cumsum = np.cumsum(TI)

#         calc_tax = np.zeros_like(ES_con)
#         tax_paid = np.zeros_like(ES_con)
#         tax_carryforward = np.zeros_like(ES_con)

#         for idx, ES_val in enumerate(ES_con):
#             if idx > 0:
#                 if ES_con[idx-1] == 0 and ES_val > 0:
#                     calc_tax[idx] = self.tax_ef * TI_cumsum[idx]
#                 elif ES_con[idx-1] > 0 and ES_val > 0:
#                     calc_tax[idx] = self.tax_ef * TI[idx]

#                 # Calculate tax total
#                 tax_total = calc_tax[idx] + tax_carryforward[idx - 1]

#                 # Calculate tax paid
#                 tax_paid[idx] = min(tax_total, ES_con[idx])

#                 # Calculate tax payable
#                 tax_carryforward[idx] = tax_total - tax_paid[idx]

#         return tax_paid, tax_carryforward


#     def _get_NCS(self, TI, Tax):
#         r"""
#         Use to calculate net contractor share (NCS).

#         .. math::

#             NCS_{(t)} = TI_{(t)} - Tax_{(t)}

#         Parameters
#         ----------
#         TI : np.ndarray
#             taxable income
#         Tax : np.ndarray
#             tax

#         Return
#         ------
#         NCS : np.ndarray
#             net contractor share
#         """
#         return TI - Tax


#     def _get_CT(self, NCS, CR):
#         r"""
#         Use to calculate contractor take (CT).

#         .. math::

#             CT_{(t)} = NCS_{(t)} + CR_{(t)}

#         Parameters
#         ----------
#         NCS : np.ndarray
#             net contractor share
#         CR : np.ndarray
#             cost recovery

#         Return
#         ------
#         CT : np.ndarray
#             contractor take
#         """
#         return NCS + CR


#     def _get_Expenditure(self, capex_tangible, capex_intangible, opex):
#         r"""
#         Use to calculate Expenditure or Cash Out.

#         .. math::
#             Expenditure_{(t)} = \text{CAPEX}_{tangible \, (t)} +
#                                 \text{CAPEX}_{intangible \, (t)} +
#                                 \text{OPEX}_{(t)}

#         Parameters
#         ----------
#         capex_tangible : np.ndarray
#             capex tangible
#         capex_intangible : np.ndarray
#             capex intangible
#         opex : np.ndarray
#             operating expenditure

#         Returns
#         -------
#         expenditure : np.ndarray
#             expenditure
#         """
#         return capex_tangible + capex_intangible + opex


#     def _get_CF(self, CT, Expenditure):
#         r"""
#         Use to calculate Cashflow (CF).

#         .. math::

#             CF_{(t)} = CT_{(t)} - Expenditure_{(t)}

#         Parameters
#         ----------
#         CT : np.ndarray
#             Contractor Take
#         Expenditure : np.ndarray
#             expenditure or cash out

#         Return
#         ------
#         CF : np.ndarray
#             cashflow
#         """
#         return CT - Expenditure


#     def _get_GOI_Take(self, GS, DDMO, Tax):
#         r"""
#         Use to calculate Goverment of Indonesia Take (GOI Take).

#         .. math::

#             GOI_{take \, (t)} = GS_{(t)} + DDMO_{(t)} + Tax_{(t)}

#         Parameters
#         ----------
#         GS : np.ndarray
#             goverment share
#         DDMO : np.ndarray
#             differential domestic market obligation
#         Tax : np.ndarray
#             tax

#         Return
#         ------
#         GOI_take: np.ndarray
#             goverment of indonesia Take
#         """
#         return GS + DDMO + Tax


    # def run_BeforeTransfer(self, depr_capex, undepr_capex, intangible, sunk_cost):
    def run_BeforeTransfer(self):
        """

        Parameters
        ----------
        depr_capex : TYPE
            DESCRIPTION.
        undepr_capex : TYPE
            DESCRIPTION.
        intangible : TYPE
            DESCRIPTION.
        sunk_cost : TYPE
            DESCRIPTION.
        capex_tangible : TYPE
            DESCRIPTION.
        capex_intangible : TYPE
            DESCRIPTION.
        year_arr : TYPE
            DESCRIPTION.

        Raise
        -----
        Exception
            if the fluid type is not valid.

        Returns
        -------
        result : TYPE
            DESCRIPTION.

        """
        # Calculate Revenue
        revenue = self.lifting.revenue()

        # Calculate FTP (Depend on Phase)
        self.FTP_con, self.FTP_gov, FTP = self._get_FTP(revenue)

#         # Calculate Revenue after FTP (Depend on Phase)
#         REV_after_FTP = self._get_REV_after_FTP(FTP)

#         # Calculate IC (Depend on Phase)
#         IC = self._get_IC()

#         # Calculate IC Paid and IC UnRec
#         self.IC_paid, IC_unrec = self._get_ICpaid_ICunrec(IC, REV_after_FTP)

#         # Calculate Revenue after FTP & IC
#         REV_after_IC = self._get_REV_after_IC(REV_after_FTP, self.IC_paid)

#         # Calculate Cost Recovery Allocation (Depend on Phase)
#         CR_allocation = self._get_CR_allocation(REV_after_IC)

#         # Calculate OPEX (Depend on Phase)
#         self.OPEX = self.cost_obj.get_opex()

#         # Calculate CR
#         CR = self._get_CR(depr_capex, undepr_capex, self.OPEX, intangible,
#                           sunk_cost)

#         # Calculate CTR
#         self.CTR, self.UR_BeforeTransfer = self._get_CTR_UR(CR, CR_allocation)

#         # Calculate ETS PreTransfer
#         self.ETS_BeforeTransfer = self._get_ETS_PreTransfer(REV_after_IC,
#                                                             self.CTR)

#         # Generate dataframe
#         col_lst = ["FTP Con", "FTP Gov", "FTP", "REV After FTP", "IC",
#                    "IC paid", "IC Unrec", "REV After IC", "CR Allocation",
#                    "OPEX", "CR", "CTR", "UR BeforeTransfer",
#                    "ETS BeforeTransfer"]

#         res_arr = np.column_stack((self.FTP_con, self.FTP_gov, FTP,
#                                    REV_after_FTP, IC, self.IC_paid, IC_unrec,
#                                    REV_after_IC, CR_allocation, self.OPEX, CR,
#                                    self.CTR, self.UR_BeforeTransfer,
#                                    self.ETS_BeforeTransfer))

#         res_df = pd.DataFrame(data=res_arr, columns=col_lst)

#         return res_df


#     def run_Transfer(self, gas_result_df, oil_result_df):
#         # Calculate Transfer
#         UR_oil = oil_result_df["UR BeforeTransfer"].to_numpy()
#         UR_gas = gas_result_df["UR BeforeTransfer"].to_numpy()
#         ETS_oil = oil_result_df["ETS BeforeTransfer"].to_numpy()
#         ETS_gas = gas_result_df["ETS BeforeTransfer"].to_numpy()

#         oil_result_df["Transfer To"], gas_result_df["Transfer To"] = \
#             self._get_Transfer(UR_gas, UR_oil, ETS_gas, ETS_oil)

#         return gas_result_df, oil_result_df


#     def run_AfterTransfer(self, capex_tangible, capex_intangible, year_arr,
#                           result_df):

#         TransferTo = result_df["Transfer To"]

#         # Calculate UR After Transfer
#         UR_AfterTransfer = \
#             self._get_UR_After_Transfer(self.UR_BeforeTransfer, TransferTo)

#         # Calculate ETS After Transfer
#         ETS_AfterTransfer = \
#             self._get_ETS_After_Transfer(self.ETS_BeforeTransfer, TransferTo,
#                                          UR_AfterTransfer)

#         # Calculate ES
#         ES_con, ES_gov = self._get_ES(ETS_AfterTransfer)

#         # Calculate CS
#         CS = self._get_CS(self.FTP_con, ES_con, self.IC_paid)

#         # Calculate GS
#         GS = self._get_GS(self.FTP_gov, ES_gov)

#         # Calculate DMO
#         DMO = self._get_DMO(ES_con, year_arr)

#         # Calculate DMO Fee
#         DMO_Fee = self._get_DMO_Fee(DMO)

#         # Calculate DDMO
#         DDMO = self._get_DDMO(DMO, DMO_Fee)

#         # Calculate TI
#         TI = self._get_TI(CS, DDMO)

#         # Calculate Tax
#         Tax_paid, Tax_carryforward = self._get_tax2(TI, ES_con)

#         # Calculate NCS
#         NCS = self._get_NCS(TI, Tax_paid)

#         # Calculate CT
#         CT = self._get_CT(NCS, self.CTR)

#         # Calculate Expenditure
#         Expenditure = \
#             self._get_Expenditure(capex_tangible, capex_intangible, self.OPEX)

#         # Calculate CF
#         CF = self._get_CF(CT, Expenditure)

#         # Calculate GOI Take
#         GOI_take = self._get_GOI_Take(GS, DDMO, Tax_paid)

#         # Generate dataframe
#         col_lst = ["UR AfterTransfer", "ETS AfterTransfer", "ES Con",
#                    "ES Gov", "CS", "GS", "DMO", "DMO Fee", "DDMO", "TI",
#                    "Tax Paid", "Tax Carryforward", "NCS", "CT",
#                    "Expenditure", "CF", "GOI Take"]

#         res_arr = np.column_stack((UR_AfterTransfer, ETS_AfterTransfer,
#                                    ES_con, ES_gov, CS, GS, DMO, DMO_Fee,
#                                    DDMO, TI, Tax_paid, Tax_carryforward,
#                                    NCS, CT, Expenditure, CF, GOI_take))

#         res_df = pd.DataFrame(data=res_arr, columns=col_lst)

#         result_df = result_df.join(res_df)

#         return result_df


# class CostRecoveryCombine(object):
#     def __init__(self, cr_gas, cr_oil):
#         self.cr_gas = cr_gas
#         self.cr_oil = cr_oil

#     def run(self):
#         cr_phase_lst = [self.cr_gas, self.cr_oil]


#         result = {}

#         # Calculate CR Before Transfer
#         for cr_obj in cr_phase_lst:
#             result[cr_obj.phase] = \
#                 cr_obj.run_BeforeTransfer(depr_capex, undepr_capex, intangible,
#                                           sunk_cost)

#         # Calculate CR Transfer
#         result[self.cr_gas.phase], result[self.cr_oil.phase] = \
#             self.cr_gas.run_Transfer(result[self.cr_gas.phase],
#                                      result[self.cr_oil.phase])

#         # Calculate CR After Transfer
#         for cr_obj in cr_phase_lst:
#             result[cr_obj.phase] = \
#                 cr_obj.run_AfterTransfer(capex_tangible, capex_intangible,
#                                          year_arr, result[cr_obj.phase])

#         # Calculate CR Combined
#         combined_df = pd.DataFrame()

#         for col_name in result[self.cr_gas.phase].columns:
#                 combined_df[col_name] = result[self.cr_gas.phase][col_name] \
#                                         + result[self.cr_oil.phase][col_name]

#         result["Combined"] = combined_df

#         return result


if __name__ == "__main__":
    from pyscnomics.datasets import load_data

    data_CR = load_data("test")