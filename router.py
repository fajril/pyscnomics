import base64
import copy
import json
import logging
import traceback
import math
import struct
from datetime import datetime

from fastapi import APIRouter, Body, Request, Depends, Form, HTTPException, status, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from pyscnomics.api.adapter import (get_baseproject,
                                    get_costrecovery,
                                    get_contract_table,
                                    get_contract_optimization,
                                    get_grosssplit,
                                    get_transition,
                                    get_detailed_summary,
                                    get_ltp_dict,
                                    get_rpd_dict,
                                    get_grosssplit_split,
                                    get_transition_split,
                                    get_economic_limit,
                                    get_asr_expenditures,
                                    get_lbt_expenditures,
                                    get_sensitivity,
                                    get_uncertainty)
from pyscnomics.api.converter import Data, EconLimit, ASRExpendituresBM, LBTExpendituresBM
from pyscnomics.api.converter import DataTransition
from pyscnomics.api.converter import LtpBM, RpdBM
from pyscnomics.tools.summarizer import Summary
import numpy as np

from summaries import Summaries
from pscExtractor import pscExtractor
from jdata import pscPacks
from io import BytesIO
import os
from psc_logger import root_print, user_id_ctx

webRouter = APIRouter(prefix='/web-api')

log = logging.getLogger()

def getLastTracebackLine(exc: Exception) -> str:
  try:
    base_path = os.getcwd()
    tb = traceback.extract_tb(exc.__traceback__)
    last_frame = tb[-1]  # ambil frame terakhir
    trace_str = f"Error at {last_frame.filename}:{last_frame.lineno} → {last_frame.line}<br>{type(exc).__name__}: {exc}"
    return trace_str.replace(base_path, '')
  except Exception as e:
    return f"{exc}"  

def print_traceback(exc: Exception, base_path: str = os.getcwd(), use_root_print: bool = False):
  tb_str = traceback.format_exc()
  if base_path:
    tb_str = tb_str.replace(base_path, '')
  tb_str = tb_str.replace('.venv/lib/', '')
  tb_str = tb_str.replace('/usr/local/lib/', '')

  if use_root_print:
    root_print(datetime.utcnow().strftime("\n[%Y-%m-%d %H:%M:%S Z+0000]"))
    root_print(tb_str)
    root_print(f"\n[{user_id_ctx.get()}]")
    root_print(exc)
  else:    
    print(datetime.utcnow().strftime("\n[%Y-%m-%d %H:%M:%S Z+0000]"))
    print(tb_str)
    print(f"\n[{user_id_ctx.get()}]")
    print(f"{type(exc).__name__}: {exc}")

def makeHttpException(exc: Exception, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR, base_path: str = os.getcwd(), use_root_print: bool = False):
  print_traceback(exc, base_path=base_path, use_root_print=use_root_print)
  raise HTTPException(
      status_code=status_code,
      detail=f"PyscErr: {getLastTracebackLine(exc)}"
  )

@webRouter.put('/get_summary') 
async def get_summary(request: Request, data: dict = Body(...)):
    try:
        ctr = data['ctr']
        # print(ctr, flush=True)
        jData: dict = json.loads(base64.b64decode(data['data']).decode("utf-8"))
        sumCalc = Summaries(ctr, jData)
        splitInfo = None
        try:
            splitInfo = (
                get_grosssplit_split(data=jData)
                if ctr == 2
                else get_transition_split(data=jData)
                if ctr >= 3
                else None
            )
        except Exception as err:
          print_traceback(err)
          pass

        return {
            "state": 1,
            "summary": sumCalc.summary,
            "splitInfo": splitInfo
        }
    except Exception as err:
      makeHttpException(err)

@webRouter.put('/get_quick_info')
async def get_quick_info(request: Request, data: dict = Body(...)):
    try:
        ctr = data['ctr']
        # print(ctr, flush=True)
        jData: dict = json.loads(base64.b64decode(data['data']).decode("utf-8"))
        sumCalc = Summaries(ctr, jData)

        return {
            "GOI2GR": sumCalc.summary["gov_take_over_gross_rev"] * 100,
            "IRR": sumCalc.summary["ctr_irr"] * 100,
            "NPV": sumCalc.summary["ctr_npv"],
            "PI": sumCalc.summary["ctr_pi"],
        }
    except Exception as err:
      makeHttpException(err)

@webRouter.put('/get_summary_cf') 
async def get_summary_cf(request: Request, data: dict = Body(...)):
    try:
        ctr = data['ctr']
        jData: dict = json.loads(base64.b64decode(data['data']).decode("utf-8"))
        ctrTable = get_contract_table(data=jData, contract_type = 'Cost Recovery' if ctr == 1 else 'Gross Split' if ctr == 2 else 'Transition' if ctr >=3 else 'Project')
        
        sumCalc = Summaries(ctr, jData)
        return {
            "state": 1,
            "table": ctrTable['consolidated'] if ctr < 3 else { 'contract_1': ctrTable['contract_1']['consolidated'], 'contract_2':ctrTable['contract_2']['consolidated'] }  
        }
    except Exception as err:
      makeHttpException(err)


@webRouter.put('/get_summary_card') 
async def get_summary_card(request: Request, data: dict = Body(...)):
    try:
        if 'ctr' not in data or data['ctr'] is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="PyscErr: Missing 'ctr' in request data"
            )
        ctr = int(data['ctr'])
        jData: dict = json.loads(base64.b64decode(data['data']).decode("utf-8"))
        sumCalc = Summaries(ctr, jData)
        return {
            "state": 1,
            "card": {
                "year": sumCalc.Year,
                "Oil": sumCalc.getOil(),
                "Gas": sumCalc.getGas(),
                "Revenue": sumCalc.getRevenue(),
                "Expenses": sumCalc.getExpenses(),
                "CashFlow": sumCalc.getCashFlow(),
                "GoI": sumCalc.getGoI(),
                "pie": sumCalc.getPie(),
                "NPV": None
            }
        }
    except Exception as err:
      makeHttpException(err)
    
@webRouter.put('/get_summary_card_sens') 
async def get_summary_card_sens(request: Request, data: dict = Body(...)):
    try:
        if 'ctr' not in data or data['ctr'] is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="PyscErr: Missing 'ctr' in request data"
            )
        ctr = data['ctr']
        jData: dict = json.loads(base64.b64decode(data['data']).decode("utf-8"))
        result = get_sensitivity(data=jData, contract_type = 'Cost Recovery' if ctr == 1 else 'Gross Split' if ctr == 2 else 'Transition' if ctr >=3 else 'Project') 
        return {
            "state": 1,
            "result": result
        }
    except Exception as err:
      makeHttpException(err)


@webRouter.put('/calc_sens') 
async def calc_sens(request: Request, data: dict = Body(...)):
    try:
        ctr = data['ctr']
        jData: dict = json.loads(base64.b64decode(data['data']).decode("utf-8"))
        result = get_sensitivity(data=jData, contract_type = 'Cost Recovery' if ctr == 1 else 'Gross Split' if ctr == 2 else 'Transition' if ctr >=3 else 'Project') 
        return {
            "state": 1,
            "result": result
        }
    except Exception as err:
      makeHttpException(err)

@webRouter.put('/calc_optim') 
async def calc_optim(request: Request, data: dict = Body(...)):
    try:
        ctr = data['ctr']
        jData: dict = json.loads(base64.b64decode(data['data']).decode("utf-8"))
        resOptim = get_contract_optimization(data=jData, contract_type = 'Cost Recovery' if ctr == 1 else 'Gross Split' if ctr == 2 else 'Transition' if ctr >=3 else 'Project') 
        # print(optimSummary, flush=True)
        # optimSummary = None
        overrideParams = []
        if resOptim is not None:
            contract = jData if ctr < 3 else jData["contract_2"]
            for i, key in enumerate(resOptim["list_params_value"].keys()):
                if (
                    key == "Oil Contractor Pre Tax"
                    and resOptim["list_params_value"][key] != "Base Value"
                ):
                    contract["costrecovery"]["oil_ctr_pretax_share"] = resOptim[
                        "list_params_value"
                    ][key]
                    overrideParams.append({
                        "param": "oil_ctr_pretax_share",
                        "value": resOptim["list_params_value"][key]
                    })
                elif (
                    key == "Gas Contractor Pre Tax"
                    and resOptim["list_params_value"][key] != "Base Value"
                ):
                    contract["costrecovery"]["gas_ctr_pretax_share"] = resOptim[
                        "list_params_value"
                    ][key]
                    overrideParams.append({
                        "param": "gas_ctr_pretax_share",
                        "value": resOptim["list_params_value"][key]
                    })
                elif (
                    key == "Oil FTP Portion"
                    and resOptim["list_params_value"][key] != "Base Value"
                ):
                    contract["costrecovery"]["oil_ftp_portion"] = resOptim[
                        "list_params_value"
                    ][key]
                    overrideParams.append({
                        "param": "oil_ftp_portion",
                        "value": resOptim["list_params_value"][key]
                    })
                elif (
                    key == "Gas FTP Portion"
                    and resOptim["list_params_value"][key] != "Base Value"
                ):
                    contract["costrecovery"]["gas_ftp_portion"] = resOptim[
                        "list_params_value"
                    ][key]
                    overrideParams.append({
                        "param": "gas_ftp_portion",
                        "value": resOptim["list_params_value"][key]
                    })
                elif (
                    key == "Oil IC"
                    and resOptim["list_params_value"][key] != "Base Value"
                ):
                    contract["costrecovery"]["oil_ic_rate"] = resOptim[
                        "list_params_value"
                    ][key]
                    overrideParams.append({
                        "param": "oil_ic_rate",
                        "value": resOptim["list_params_value"][key]
                    })
                elif (
                    key == "Gas IC"
                    and resOptim["list_params_value"][key] != "Base Value"
                ):
                    contract["costrecovery"]["gas_ic_rate"] = resOptim[
                        "list_params_value"
                    ][key]
                    overrideParams.append({
                        "param": "gas_ic_rate",
                        "value": resOptim["list_params_value"][key]
                    })
                elif (
                    key == "Oil DMO Fee"
                    and resOptim["list_params_value"][key] != "Base Value"
                ):
                    if ctr in [1, 3, 6]:
                        contract["costrecovery"]["oil_dmo_fee_portion"] = resOptim[
                            "list_params_value"
                        ][key]
                    else:
                        contract["grosssplit"]["oil_dmo_fee_portion"] = resOptim[
                            "list_params_value"
                        ][key]
                    overrideParams.append({
                        "param": "oil_dmo_fee_portion",
                        "value": resOptim["list_params_value"][key]
                    })
                elif (
                    key == "Gas DMO Fee"
                    and resOptim["list_params_value"][key] != "Base Value"
                ):
                    if ctr == 1:
                        contract["costrecovery"]["gas_dmo_fee_portion"] = resOptim[
                            "list_params_value"
                        ][key]
                    else:
                        contract["grosssplit"]["gas_dmo_fee_portion"] = resOptim[
                            "list_params_value"
                        ][key]
                    overrideParams.append({
                        "param": "gas_dmo_fee_portion",
                        "value": resOptim["list_params_value"][key]
                    })
                elif (
                    key == "VAT Rate"
                    and resOptim["list_params_value"][key] != "Base Value"
                ):
                    contract["contract_arguments"]["vat_rate"] = resOptim[
                        "list_params_value"
                    ][key]
                    overrideParams.append({
                        "param": "vat_rate",
                        "value": resOptim["list_params_value"][key]
                    })
                elif (
                    key == "Effective Tax Rate"
                    and resOptim["list_params_value"][key] != "Base Value"
                ):
                    contract["contract_arguments"]["effective_tax_rate"] = resOptim[
                        "list_params_value"
                    ][key]
                    overrideParams.append({
                        "param": "effective_tax_rate",
                        "value": resOptim["list_params_value"][key]
                    })
                elif (
                    key == "Ministerial Discretion"
                    and resOptim["list_params_value"][key] != "Base Value"
                ):
                    contract["grosssplit"]["split_ministry_disc"] = resOptim[
                        "list_params_value"
                    ][key]
                    overrideParams.append({
                        "param": "split_ministry_disc",
                        "value": resOptim["list_params_value"][key]
                    })
                elif (
                    key == "Depreciation Acceleration"
                    and resOptim["list_params_value"][key]["depreciation acceleration"]
                    != "Base Value"
                ):
                    for k in contract["capital"].keys():
                        contract["capital"][k]["useful_life"] = resOptim[
                            "list_params_value"
                        ][key]["optimized_useful_life"]["useful_life_optimized"]
                    overrideParams.append({
                        "param": "useful_life",
                        "value": {
                            "year":resOptim["list_params_value"][key]["optimized_useful_life"]["year"],
                            "useful_life": resOptim["list_params_value"][key]["optimized_useful_life"]["useful_life_optimized"]
                        }
                    })
                elif (
                    key == "VAT Discount"
                    and resOptim["list_params_value"][key]
                    != "Base Value"
                ):
                    for k in contract["capital"].keys():
                        for i, v in enumerate(contract["capital"][k]["tax_discount"]):
                            contract["capital"][k]["tax_discount"][i] = resOptim[
                            "list_params_value"][key]
                    for k in contract["intangible"].keys():
                        for i, v in enumerate(contract["intangible"][k]["tax_discount"]):
                            contract["intangible"][k]["tax_discount"][i] = resOptim[
                            "list_params_value"][key]
                    for k in contract["opex"].keys():
                        for i, v in enumerate(contract["opex"][k]["tax_discount"]):
                            contract["opex"][k]["tax_discount"][i] = resOptim[
                            "list_params_value"][key]
                    for k in contract["asr"].keys():
                        for i, v in enumerate(contract["asr"][k]["tax_discount"]):
                            contract["asr"][k]["tax_discount"][i] = resOptim[
                            "list_params_value"][key]
                    overrideParams.append({
                        "param": "vat_discount",
                        "value": resOptim["list_params_value"][key]
                    })
            # print(jData, flush=True)
            optimSummary = Summaries(ctr, jData).summary
        return {
            "state": 1,
            "result": resOptim,
            "summary": optimSummary,
            "override_params": overrideParams
        }
    except Exception as err:
      makeHttpException(err)

@webRouter.put('/calc_monte') 
async def calc_monte(request: Request, data: dict = Body(...)):
    try:
        ctr = data['ctr']
        jData: dict = json.loads(base64.b64decode(data['data']).decode("utf-8"))
        resMonte = get_uncertainty(data=jData, contract_type = 'Cost Recovery' if ctr == 1 else 'Gross Split' if ctr == 2 else 'Transition' if ctr >=3 else 'Project') 
        # print(resMonte, flush=True)
        return {
            "state": 1,
            "result": resMonte,
        }
    except Exception as err:
      makeHttpException(err)

@webRouter.put('/get_compare') 
async def get_compare(request: Request, data: dict = Body(...)):
    try:
        jData: dict = json.loads(base64.b64decode(data['data']).decode("utf-8"))
        resCompare = []
        for i, contract_ in enumerate(jData):
            summary = None
            ctrTable = None
            try:
                ctr = contract_["ctr"]
                summary = Summaries(ctr, contract_["data"]).summary
                ctrTable = get_contract_table(data=contract_["data"], contract_type = 'Cost Recovery' if ctr == 1 else 'Gross Split' if ctr == 2 else 'Transition' if ctr >=3 else 'Project')
            except Exception:
                pass
            resCompare.append({
                "summary": summary,
                "table": ctrTable
            })
        return {
            "state": 1,
            "result": resCompare
        }
    except Exception as err:
      makeHttpException(err)

@webRouter.put('/get_combine', response_class=JSONResponse) 
async def get_combine(request: Request, data: dict = Body(...)):
    try:
        jData: dict = json.loads(base64.b64decode(data['data']).decode("utf-8"))
        # print(jData, flush=True)
        reference_year = jData["reference_year"]
        inflation_rate = jData["inflation_rate"]
        discount_rate = jData["discount_rate"]
        npv_mode = jData["npv_mode"]
        discounting_mode = jData["discounting_mode"]
        profitability_discounted = jData["profitability_discounted"]
        contracts = []
        for i, contract_ in enumerate(jData["contracts"]):
            ctr = contract_["ctr"]
            if (ctr == 0):
                contracts.append(get_baseproject(data=contract_["data"], summary_result=False)[1])
            elif (ctr == 1):
                contracts.append(get_costrecovery(data=contract_["data"], summary_result=False)[1])
            elif (ctr == 2):
                contracts.append(get_grosssplit(data=contract_["data"], summary_result=False)[1])
            else: 
                contracts.append(get_transition(data=contract_["data"])[1])
        summary_ = Summary(
            contract=contracts,
            reference_year=reference_year,
            inflation_rate=inflation_rate,
            discount_rate=discount_rate,
            npv_mode=npv_mode,
            discounting_mode=discounting_mode,
            profitability_discounted=profitability_discounted
        )
        tbl_summ = summary_.case_combine()
        exc_summ = summary_.get_executive_summary()
        # summary_._to_dataframe()
        # print(tbl_summ, flush=True)
        return {
            "state": 1,
            "combine": {
                "summary":{
                    key: None if exc_summ[key] is not None and (math.isnan(exc_summ[key]) or math.isinf(exc_summ[key])) else float(exc_summ[key]) if exc_summ[key] is not None else None  
                    for key in exc_summ
                },
                "table": {
                    key: tbl_summ[key].tolist() if isinstance(tbl_summ[key], np.ndarray) else tbl_summ[key]  
                    for key in tbl_summ
                }  
            },
            "items": None 
        }
    except Exception as err:
      makeHttpException(err)

@webRouter.put('/get_increment') 
async def get_increment(request: Request, data: dict = Body(...)):
    try:
        jData: dict = json.loads(base64.b64decode(data['data']).decode("utf-8"))
        reference_year = jData["reference_year"]
        inflation_rate = jData["inflation_rate"]
        discount_rate = jData["discount_rate"]
        npv_mode = jData["npv_mode"]
        discounting_mode = jData["discounting_mode"]
        profitability_discounted = jData["profitability_discounted"]
        contracts = []
        for i, contract_ in enumerate(jData["contracts"]):
            ctr = contract_["ctr"] if contract_ is not None else -1
            if (ctr == 0):
                contracts.append(get_baseproject(data=contract_["data"], summary_result=False)[1])
            elif (ctr == 1):
                contracts.append(get_costrecovery(data=contract_["data"], summary_result=False)[1])
            elif (ctr == 2):
                contracts.append(get_grosssplit(data=contract_["data"], summary_result=False)[1])
            elif (ctr > 2): 
                contracts.append(get_transition(data=contract_["data"])[1])
        # print(contracts, flush=True)        
        summary_ = Summary(
            contract=contracts,
            reference_year=reference_year,
            inflation_rate=inflation_rate,
            discount_rate=discount_rate,
            npv_mode=npv_mode,
            discounting_mode=discounting_mode,
            profitability_discounted=profitability_discounted
        )
        tbl_summ = summary_.case_incremental()
        exc_summ = summary_.get_executive_summary()
        # print(exc_summ, flush=True)     
        # print(tbl_summ, flush=True)     
        return {
            "state": 1,
            "summary": {
                key: None if exc_summ[key] is not None and (math.isnan(exc_summ[key]) or math.isinf(exc_summ[key])) else float(exc_summ[key]) if exc_summ[key] is not None else None  
                for key in exc_summ
            },
                "table": {
                    key: tbl_summ[key].tolist() if isinstance(tbl_summ[key], np.ndarray) else tbl_summ[key]  
                    for key in tbl_summ
                }  
        }
    except Exception as err:
      makeHttpException(err)

@webRouter.put('/calc_rpd') 
async def calc_rpd(request: Request, data: dict = Body(...)):
    try:
        jData: dict = json.loads(base64.b64decode(data['data']).decode("utf-8"))
        rpd_ = get_rpd_dict(jData)
        return {
            "state": 1,
            "rpd": rpd_,
        }
    except Exception as err:
      makeHttpException(err)


@webRouter.put('/get_cf') 
async def get_cf(request: Request, data: dict = Body(...)):
    def calcEcoLimit(mth: str, dataTable_: dict):
        keys_ = dataTable_.keys()
        if 'CashFlow' in keys_:
            key_ = 'CashFlow'
        elif 'Cashflow' in keys_:
            key_ = 'Cashflow'
        elif 'CTR_Cash_Flow' in keys_:
            key_ = 'CTR_Cash_Flow'
        elif 'C_CashFlow' in keys_:
            key_ = 'C_CashFlow'
        elif 'C_Cashflow' in keys_:
            key_ = 'C_Cashflow'
        else:     
            key_ = None
        return get_economic_limit({
                "years": [int(y) for y in dataTable_[key_].keys()],
                "cash_flow": [float(v) for v in dataTable_[key_].values()],
                "method": mth
            }) if key_ is not None else None

    try:
        ctr = data['ctr']
        ecolimit = data['ecolimit']
        hasGas = data['hasGas']
        jData: dict = json.loads(base64.b64decode(data['data']).decode("utf-8"))
        ctrTable = get_contract_table(data=jData, contract_type = 'Cost Recovery' if ctr == 1 else 'Gross Split' if ctr == 2 else 'Transition' if ctr >=3 else 'Project')
        try:
            if ctr < 3:
                ecoLimitResult = {
                    "oil":  int(calcEcoLimit(ecolimit, ctrTable["oil"])),
                    "gas":  int(calcEcoLimit(ecolimit, ctrTable["gas"])) if hasGas else None,
                    "consolidated":  int(calcEcoLimit(ecolimit, ctrTable["consolidated"]))
                }
            else:
                ecoLimitResult = {
                    "contract_1": {
                        "oil":  int(calcEcoLimit(ecolimit, ctrTable["contract_1"]["oil"])),
                        "gas":  int(calcEcoLimit(ecolimit, ctrTable["contract_1"]["gas"])) if hasGas else None,
                        "consolidated":  int(calcEcoLimit(ecolimit, ctrTable["contract_1"]["consolidated"]))
                    },
                    "contract_2": {
                        "oil":  int(calcEcoLimit(ecolimit, ctrTable["contract_2"]["oil"])),
                        "gas":  int(calcEcoLimit(ecolimit, ctrTable["contract_2"]["gas"])) if hasGas else None,
                        "consolidated":  int(calcEcoLimit(ecolimit, ctrTable["contract_2"]["consolidated"]))
                    }
                }
        except Exception as err:
            pass

        return {
            "state": 1,
            "table": ctrTable,
            "ecolimit": ecoLimitResult
        }
    except Exception as err:
      makeHttpException(err)

@webRouter.put('/get_ecolimit') 
async def get_ecolimit(request: Request, data: dict = Body(...)):
    try:
        jData: dict = json.loads(base64.b64decode(data['data']).decode("utf-8"))
        return {
            "state": 1,
            "ecolimit": int(get_economic_limit(jData))
        }
    except Exception as err:
      makeHttpException(err)


@webRouter.put('/pick_case_psc')
async def pick_case_psc(request: Request, case_ids: str, file: UploadFile = File(...)):
  try:
    # split case_ids by comma then convert to list of int
    split_ids = [int(id) for id in case_ids.split(',')]

    file_bytes = await file.read()

    return {
        "state": 1,
        "data": pscPacks().getCaseData(split_ids, file_bytes)
    }
  except Exception as err:
    makeHttpException(err)

@webRouter.put('/get_psc_case')
async def get_psc_case(request: Request, file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()

        return {
            "state": 1,
            "data": pscPacks().getCase(file_bytes)
        }
    except Exception as err:
      makeHttpException(err)

@webRouter.put('/uppack_psc')
async def uppack_psc(request: Request, file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()

        return {
            "state": 1,
            "data": pscPacks().unpackDataFile(file_bytes)
        }
    except Exception as err:
      makeHttpException(err)

@webRouter.put('/unpack_allpsc')
async def unpack_allpsc(request: Request, file: UploadFile = File(...)):
  try:
    file_bytes = await file.read()
    # check if is valid psc pack file
    hfl, vfl = struct.unpack("@7si", file_bytes[:struct.calcsize("@7si")])
    print({"hfl":hfl, "vfl":vfl})
    validFile = (hfl == b"pSCWapp") or (hfl == b"pySCapp" and vfl >= 4 and vfl <= 20)
    if not validFile:
      return {
        "state": 0,
        "data": {
          "header": {"magic": hfl.decode(errors='ignore'), "version": vfl},
          "payload": None,
        },
      }

    if hfl == b"pSCWapp":
      return {
        "state": 1,
        "data": pscPacks().unpackDataFile(file_bytes)
      }
    elif hfl == b"pySCapp":
      header_size = struct.calcsize("@7si")
      extractor = pscExtractor(hfl= hfl.decode(errors='ignore'), vfl=vfl, file=file_bytes[header_size:])
      cases = await extractor.extractProject()

      return {
        'state': 1,
        'data': {
          "header": {"magic": hfl.decode(errors='ignore'), "version": vfl},
          "cases": cases
        },
      }


  except Exception as err:
    makeHttpException(err)

@webRouter.put('/pack_psc')
async def pack_psc(request: Request, data = Body(...)):
    try:
        jData = base64.b64decode(data["datafile"]).decode("utf-8")
        binary = pscPacks().packDataFile(jData)

        stream = BytesIO(binary)
        return StreamingResponse(
            stream,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": "attachment; filename=packed.psc"
            }
        )
        # return {
        #     "state": 1,
        #     "filename": "packed.psc",
        #     "file": encoded,
        #     "size": len(binary)
        # }
    except Exception as err:
      makeHttpException(err)


@webRouter.post('/extract_psc') 
async def extract_psc(request: Request, file: UploadFile = File(...)):
    __hflb: str = "pySCapp"
    __hfl: str = "pSCWapp"
    __vfl: int = 20
    try:
        # test file
        hfl, vfl = struct.unpack("@7si", await file.read(struct.calcsize("@7si")))
        print({"hfl":hfl, "vfl":vfl}, flush=True)
        validFile = (hfl == __hfl.encode() or hfl == __hflb.encode()) and (vfl >= 4 and vfl <= __vfl)
        if not validFile:
            return {
                "state": 0,
                "output": "Invalid file type"
            }

        flExt = pscExtractor(hfl= __hflb if hfl == __hflb.encode() else __hfl, vfl=vfl, file=file)

        cases = await flExt.extractProject()

        return {
            'state': 1,
            'cases': cases
        }
    except Exception as err:
      makeHttpException(err)

@webRouter.get('/get_logger', response_class=JSONResponse) 
async def get_logger(request: Request):
    try:
      user_id = request.headers.get("X-User-ID", "anonymous")  
      # read the log file into array, only last 500 lines
      log_file = f"logs/user_{user_id}.log"
      # log_file = 'logs/gunicorn-error.log'
      with open(log_file, 'r') as f:
      # read the last 100 lines
        logs = f.readlines()[-500:]
      # remove text contains [INFO] and [ERROR]
      logs = [line for line in logs if not (line.__contains__('[INFO]') or line.__contains__('[ERROR]'))]
      # remove text contains [DEBUG]
      logs = [line for line in logs if not (line.__contains__('[DEBUG]'))]
      # remove text contains [WARNING]
      logs = [line for line in logs if not (line.__contains__('[WARNING]'))]
      # remove text contains [CRITICAL]
      logs = [line for line in logs if not (line.__contains__('[CRITICAL]'))]
      # remove text contains [gunicorn.error]
      logs = [line for line in logs if not (line.__contains__('[gunicorn.error]'))]
      # remove text contains [gunicorn.access]
      logs = [line for line in logs if not (line.__contains__('[gunicorn.access]'))]
      # remove text contains [gunicorn]
      logs = [line for line in logs if not (line.__contains__('[gunicorn]'))]
      # remove text contains path and get only filename
      logs = ['/'.join(line.split('/')[-2:]) if '/' in line else line for line in logs]
      # # remove text contains /home/pscnomics/pscnomics-web/
      # logs = [line.replace('/home/pscnomics/pscnomics-web/', '') for line in logs]
      # # # remove text contains /home/pscnomics/pscnomics-web/venv/lib/python3.10/site-packages/
      # logs = [line.replace('/home/pscnomics/pscnomics-web/.venv/lib/', '') for line in logs]
      # # remove text contains .venv/lib/
      # logs = [line.replace('.venv/lib/', '') for line in logs]
      # # remove text contains /usr/local/lib/
      # logs = [line.replace('/usr/local/lib/', '') for line in logs]

      # then join to text
      logs = ''.join(logs)
      return {
        'state': 1,
        'logs': logs
      }
    except Exception as err:
      makeHttpException(err, use_root_print=True)

@webRouter.put('/get_multiver_psc_case')
async def get_multi_psc_case(request: Request, file: UploadFile = File(...)):
  try:
    # tes if is old psc file, read header
    file_bytes = await file.read()
    hfl, vfl = struct.unpack("@7si", file_bytes[:struct.calcsize("@7si")])
    print({"hfl":hfl, "vfl":vfl}, flush=True)
    if hfl == b"pSCWapp" and vfl >= 1:
      # its new psc file
      result_ = pscPacks().getCase(file_bytes)
    elif hfl == b"pySCapp" and vfl >= 4 and vfl <= 20:  
      # its old psc file
      result_ = await pscExtractor.getCases(file_bytes)
    else:
      # its pscdb file, test using static function getCasefromPSCdb
      result_ = None # await pscPacks.getCasefromPSCdb(file_bytes)
    return {
        "state": 1,
        "header": {"magic": hfl.decode(), "version": vfl},
        "data": result_
    }
  except Exception as err:
    makeHttpException(err)

@webRouter.put('/get_multiver_psc_datacase')
async def get_multiver_psc_datacase(request: Request, case_ids: str, file: UploadFile = File(...)):
  try:
    # split case_ids by comma then convert to list of int
    split_ids = [int(id) for id in case_ids.split(',')]

    # tes if is old psc file, read header
    file_bytes = await file.read()
    hfl, vfl = struct.unpack("@7si", file_bytes[:struct.calcsize("@7si")])
    print({"hfl":hfl, "vfl":vfl}, flush=True)
    if hfl == b"pSCWapp" and vfl >= 1:
      # its new psc file
      result_ = pscPacks().getCaseData(split_ids, file_bytes)
    elif hfl == b"pySCapp" and vfl >= 4 and vfl <= 20:  
      # its old psc file
      result_ = await pscExtractor.getCasesData(split_ids, file_bytes)
    else:
      result_ = None 
    return {
        "state": 1,
        "header": {"magic": hfl.decode(), "version": vfl},
        "data": result_
    }
  except Exception as err:
    makeHttpException(err)
