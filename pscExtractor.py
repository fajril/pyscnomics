import logging
import math
import os
import pickle
import random
import shutil
import string
import struct
from io import BufferedReader, BufferedWriter, BytesIO
from pathlib import Path
from sys import exception
from typing import Any, List

import numpy as np
from fastapi import File, UploadFile
import traceback

class pscPacker:
  __hfl: str = "pSCWapp"
  __vfl: int = 20
    # ver.20: 
    #  (+) add created_at field to case
    #  (+) add field_reserves to gsConfig
    #  (+) add cId field to fiscal_base
    #  (+) add ctr field to fiscal_base
    
  data: dict
  _buffer: BytesIO
  
  def __init__(self, data: dict):
    self.data = data
    self._buffer = BytesIO()

  def pack(self):
    self._buffer.write(struct.pack("@7si", self.__hfl.encode(), self.__vfl))

    self.packCases()

    # for icase in self.data:
    #   self.packGenConfig(icase['GenCfg'])
    #   for idx, ifiscal in enumerate(icase['fiscal']):
    #       self.packFiscalBase(idx, ifiscal)
    #   self.packProducer()
    #   self.packcontrats(icase['GenCfg']["type_of_contract"], icase['contrats'])
    #   # tangible
    #   self.writeCosts('tangible')
    #   # intanginble
    #   self.writeCosts('intangible')
    #   # opex
    #   self.writeCosts('opex')
    #   # ars
    #   self.writeCosts('ars')
    #   # cos
    #   self.writeCosts('cos')
    #   # LBT
    #   self.writeCosts('lbt')

    #   # write sens
    #   for icase in self.data:
    #       await self.writeSens(icase[""])

    #   # write monte
    #   for idx, icase in enumerate(cases):
    #       monteCfg = self.loadmonte(wsPath, icase["id"])
    #       monteRes = self.loadmonteRes(wsPath, icase["id"])
    #       self.writeMonte(Path(), monteCfg, fs)
    #       self.writeMonteRes(Path(), monteRes, fs)

    #   # write optim
    #   for idx, icase in enumerate(cases):
    #       optimCfg = self.loadoptim(wsPath, icase["id"])
    #       self.writeOptim(Path(), optimCfg, fs)
    #   # write compare
    #   comp_ = self.loadCompare(wsPath)
    #   self.writeCompareConf(None, comp_, fs)
    #   # write combine
    #   comb_ = self.loadCombine(wsPath)
    #   self.writeCombineConf(None, comb_, fs)
    #   # write incremental
    #   incr_ = self.loadIncr(wsPath)
    #   self.writeIncrConf(None, incr_, fs)


  def getFormatIndex(self, fmtType: List | str, index: int) -> str:
      if isinstance(fmtType, List):
          return fmtType[index] if index < len(fmtType) else fmtType[-1]
      else:
          return fmtType

  def writeTable(self, arr: dict | List, fmtType: List):
    self.writePack(len(arr), "i")
    if len(arr) > 0:
      if isinstance(arr[0], List):
        self.writePack(len(arr[0]), "i")
      for i, item in enumerate(arr):
        if isinstance(item, dict):
          for idx, key in enumerate(item.keys()):
            self.writePack(item[key], self.getFormatIndex(fmtType, idx))
        else:
          for ii, val in enumerate(item):
            self.writePack(val, self.getFormatIndex(fmtType, ii))

  def writePack(self, val, fmt: str, toFrac: bool = False):
    if fmt == "s":
        lenTxt = len(val) if val is not None else 0
        self._buffer.write(struct.pack("@i", lenTxt))
        if lenTxt > 0:
            self._buffer.write(struct.pack(f"@{lenTxt}s", str(val).encode()))
    elif fmt == "?":
        self._buffer.write(struct.pack("@?", True if val else False))
    else:
        if val is not None and isinstance(val, str):
            if fmt in ["f", "d"]:
                val = float(val) if len(val.strip()) else None
            else:
                val = int(val) if len(val.strip()) else None
        if toFrac and val is not None and fmt in ["f", "d"]:
            val = val / 100
        if fmt in ["f", "d"]:
            self._buffer.write(struct.pack(f"@{fmt}", val if val is not None else np.nan))
        else:
            # integer
            self._buffer.write(struct.pack("@?", True if val is not None else False))
            if val is not None:
                self._buffer.write(struct.pack(f"@{fmt}", val))

  # pack case from data
  def packCases(self):
    def packCase(icase: dict):
      self.writePack(icase["id"], "i")
      self.writePack(icase["name"], "s")
      self.writePack(icase["description"], "s")
      self.writePack(icase["ctrType"], "h")
      self.writePack(icase["updated_at"], "q")
      lidcase: int = 0 if icase["ctrType"] != -1 else len(icase["multicase"])
      self.writePack(lidcase, "i")
      if icase["ctrType"] == -1 and lidcase > 0:
        for i in icase["multicase"]:
          self.writePack(i, "i")
      self.writePack(icase.get("evaluator", ""), "s")
      self.writePack(icase.get("evaluator_date", icase["updated_at"]), "q")
      # ver 20:
      self.writePack(icase["created_at"], "q")

    # pack each case
    self.writePack(len(self.data), "i")
    for icase in self.data:
        packCase(icase)

  # write Combo Value and MultiValue
  def packCombo(self, combo: dict):
    self.writePack(combo["mode"], "h")
    self.writePack(combo["rate_init"], "d", True)
    # write multi
    self.writePack(len(combo["multi"]), "i")
    for item in combo["multi"]:
      self.writePack(item["year"], "i")
      self.writePack(item["rate"], "d", True)
      
  # write Depreciation
  def packDep(self, dep: dict):
    self.writePack(dep["depreciation_method"], "h")
    self.writePack(dep["decline_factor"], "d")

  # write GenConfig config
  def packGenConfig(self, gen_config: dict):
    self.writePack(gen_config["type_of_contract"], "i")
    self.writePack(gen_config["discount_rate_start_year"], "i")
    self.writePack(gen_config["discount_rate"], "d", True)
    self.writePack(gen_config["inflation_rate_applied_to"], "h")
    self.writePack(gen_config["start_date_project"], "q")
    self.writePack(gen_config["end_date_project"], "q")
    self.writePack(gen_config["start_date_project_second"], "q")
    self.writePack(gen_config["end_date_project_second"], "q")
    # start ver 12
    self.writePack(gen_config["delayAccMode"], "h")
    self.writePack(gen_config["delayAccYear"], "h")
    # start ver 15
    self.writePack(gen_config["useCOS"], "?")
    # start ver 19
    self.writePack(gen_config["lbtUseCalc"], "?")

  # write Fiscal Base
  async def packFiscalBase(self, fiscal_base: dict):
      self.writePack(fiscal_base["transferred_unrec_cost"], "d")
      self.packCombo(fiscal_base["Tax"])
      self.writePack(fiscal_base["tax_payment_config"], "h")
      self.writePack(fiscal_base["asr_future_rate"], "d", True)
      self.packDep(fiscal_base["Depreciation"])
      self.packCombo(fiscal_base["Inflation"])
      self.packCombo(fiscal_base["VAT"])
      self.packCombo(fiscal_base["LBT"])
      self.writePack(fiscal_base["vat_discount"], "d", True)
      self.writePack(fiscal_base["lbt_discount"], "d", True)
      self.writePack(fiscal_base["npv_mode"], "h")
      self.writePack(fiscal_base["discounting_mode"], "h")
      self.writePack(fiscal_base["sulfur_revenue_config"], "h")
      self.writePack(fiscal_base["electricity_revenue_config"], "h")
      self.writePack(fiscal_base["co2_revenue_config"], "h")
      self.writePack(fiscal_base["sunk_cost_reference_year"], "i")
      # versi 13
      self.writePack(fiscal_base["profitability_discounted"], "?")
      self.writePack(fiscal_base["regime"], "h")
      # versi 18
      self.writePack(fiscal_base["sum_undepreciated_cost"], "?")
      # versi 20
      # self.writePack(fiscal_base["amortization"], "q")

  # # write producer
  # async def packProducer(self, producer: List[dict]):
  #   # pack each producer
  #   self.writePack(len(producer), "i")
  #   for iprod in producer:
  #     self.writePack(iprod["Tipe"], "h")
  #     self.writePack(iprod["onstream_date"], "q")
  #     self.writePack(len(iprod["prod_price"]), "h")
  #     self.writePack(iprod["GSANumber"], "h")
  #     # if self.__vfl >= 20:
  #     #   # pack name
  #     #   self.writePack(iprod["name"], "s")

  #     # pack prod_price
  #     lenProdPrice = len(iprod["prod_price"])
  #     self.writePack(lenProdPrice, "i")
  #     for iTable in iprod["prod_price"]:
  #       self.writePack(len(iTable), "i")
  #       if iprod["Tipe"] == 1 and len(iTable) > 0:
  #         # Gas Producer
  #         for item in iTable:
  #           self.writePack(item["year"], "i")
  #           self.writePack(item["production"], "d")
  #           gsa = item["gsa"]
  #           self.writePack(len(gsa) * 3, "i")
  #           for iGSA in gsa:
  #             self.writePack(iGSA['vol'], "d")
  #             self.writePack(iGSA['ghv'], "d")
  #             self.writePack(iGSA['price'], "d")

  #           # versi 13
  #           self.writePack(item["base"], "d")
  #       elif iprod["Tipe"] == 0 and len(iTable) > 0:
  #         # Oil Producer
  #         self.writePack(len(iTable), "i")
  #         for item in iTable:
  #           self.writePack(item["year"], "i")
  #           self.writePack(item["sales"], "d")
  #           self.writePack(item["price"], "d")
  #           self.writePack(item["condensate_sales"], "d")
  #           self.writePack(item["condensate_price"], "d")
  #           # versi 13
  #           self.writePack(item["base"], "d")
  #       elif iprod["Tipe"] >= 2 and len(iTable) > 0:
  #         # Other Producer
  #         self.writePack(len(iTable), "i")
  #         for item in iTable:
  #           self.writePack(item["year"], "i")
  #           self.writePack(item["sales"], "d")
  #           self.writePack(item["price"], "d")
  #           # versi 13
  #           self.writePack(item["base"], "d")

  
  # # write contrats
  # async def packcontrats(self, ctrType: int, contrats: dict):
  #   def packDMO(dmo: dict):
  #     self.writePack(bool(dmo["holiday"]), "?")
  #     self.writePack(dmo["period"], "i")
  #     self.writePack(dmo["start_production"], "q")
  #     self.writePack(dmo["volume"], "d")
  #     self.writePack(dmo["fee"], "d")

  #   def packCostRec(costrec: dict):
  #     def packFTP(ftp: dict):
  #       self.writePack(ftp["ftp_availability"], "?")
  #       self.writePack(ftp["ftp_is_shared"], "?")
  #       self.writePack(ftp["ftp_portion"], "d")

  #     def packTaxSplit(taxSplit: dict):
  #       self.writePack(taxSplit["split_type"], "h")
  #       self.writePack(taxSplit["pre_tax_ctr_oil"], "d")
  #       self.writePack(taxSplit["pre_tax_ctr_gas"], "d")

  #     def packIC(ic: dict):
  #       self.writePack(ic["ic_availability"], "?")
  #       self.writePack(ic["ic_oil"], "d")
  #       self.writePack(ic["ic_gas"], "d")

  #     def packCR(cr: dict):
  #       self.writePack(cr["oil_cr_cap_rate"], "d")
  #       self.writePack(cr["gas_cr_cap_rate"], "d")
      
  #     # pack costrec
  #     packFTP(costrec["oil_ftp"])
  #     packFTP(costrec["gas_ftp"])
  #     packTaxSplit(costrec["TaxSplit"])
  #     packIC(costrec["IC"])
  #     packCR(costrec["CR"])
  #     # RCSlidingScale
  #     self.writeTable(costrec["RCSlidingScale"], ["d", "d", "d", "d"])
  #     # ICPSlidingScale
  #     self.writeTable(costrec["ICPSlidingScale"], ["d", "d", "d", "d"])
  #     # Indicator
  #     self.writeTable(costrec["Indicator"], ["i", "d"])
  #     # dmo_is_weighted
  #     self.writePack(costrec["dmo_is_weighted"], "?")
  #     # OilDMO
  #     packDMO(costrec["OilDMO"])
  #     # GasDMO
  #     packDMO(costrec["GasDMO"])

  #     # vfl >= 7
  #     self.writePack(costrec["post_uu_22_year2001"], "?")
  #     # vfl >= 14
  #     self.writePack(costrec["oil_cost_of_sales_applied"], "?")
  #     self.writePack(costrec["gas_cost_of_sales_applied"], "?")
      
  #   def packGS(gs: dict):
  #     for key in [
  #         "field_status",
  #         "field_location",
  #         "reservoir_depth",
  #         "infrastructure_availability",
  #         "reservoir_type",
  #         "co2_content",
  #         "h2s_content",
  #         "oil_api",
  #         "domestic_content_use",
  #         "production_stage",
  #     ]:
  #       self.writePack(gs[key], "h")
  #     self.writePack(gs["ministry_discretion_split"], "d")
  #     self.writePack(gs["oil_base_split"], "d")
  #     self.writePack(gs["gas_base_split"], "d")

  #     self.writePack(gs["dmo_is_weighted"], "?")
  #     packDMO(gs["OilDMO"])
  #     packDMO(gs["GasDMO"])

  #     # vfl >= 7
  #     self.writePack(gs["cum_production_split_offset"]["mode"], "h")
  #     self.writePack(gs["cum_production_split_offset"]["offset"], "d")
  #     self.writeTable(gs["cum_production_split_offset"]["split"], ["i", "d"])

  #     # vfl >= 11
  #     self.writePack(gs["amortization"], "?")

  #     # vfl >= 20
  #     # self.writePack(gs["field_reserves"], "h")

  #   # pack each contract
  #   packCostRec(contrats["contracts"]["cr"][0])
  #   packGS(contrats["contracts"]["gs"][0])
  #   if ctrType >= 3:
  #     (
  #       packGS(contrats["contracts"]["gs"][1]) 
  #       if ctrType in [3, 6] else packCostRec(contrats["contracts"]["cr"][1])
  #     )
  #   # ver 20
  #   # (
  #   #   packCostRec(contrats["contracts"]["cr"][1])
  #   #   if ctrType in [3, 6]
  #   #   else packGS(contrats["contracts"]["gs"][1])
  #   # )


class pscExtractor:
    hfl: str = "pySCapp" # header file pySCapp or pSCWapp (new header file ver.web)
    vfl: int = 20
    fs: UploadFile | BytesIO = File(...)

    def __init__(self, hfl: str, vfl: int, file: UploadFile | bytes = File(...)):
      if isinstance(file, bytes):
        self.fs = BytesIO(file)
        self.source = "bytes"
      else:
        self.source = "file"
        self.fs = file
      self.vfl = vfl
      self.hfl = hfl

    async def readPack(self, fmt: str, defval: Any = None):
      if self.source == "file":
        if fmt.find("s") != -1:
            # string
            if len(fmt) != 1:
                (val,) = struct.unpack(f"@{fmt}", await self.fs.read(struct.calcsize(f"@{fmt}")))
                return val.decode("utf-8") if isinstance(val, bytes) else defval
            else:
                (lentxt,) = struct.unpack("@i", await self.fs.read(struct.calcsize("@i")))
                (val,) = struct.unpack(
                    f"@{lentxt}s", await self.fs.read(struct.calcsize(f"@{lentxt}s"))
                )
                return val.decode("utf-8") if isinstance(val, bytes) else defval
        elif fmt in ["f", "d"]:
            # float
            (val,) = struct.unpack(f"@{fmt}", await self.fs.read(struct.calcsize(f"@{fmt}")))
            return val if not math.isnan(val) else defval
        else:
            # int
            (isval,) = struct.unpack("@?", await self.fs.read(struct.calcsize("@?")))
            if fmt == "?":
                return True if isval else defval
            elif isval:
                (val,) = struct.unpack(f"@{fmt}", await self.fs.read(struct.calcsize(f"@{fmt}")))
                return val
            else:
                return defval
      else:
        if fmt.find("s") != -1:
            # string
            if len(fmt) != 1:
                (val,) = struct.unpack(f"@{fmt}", self.fs.read(struct.calcsize(f"@{fmt}")))
                return val.decode("utf-8") if isinstance(val, bytes) else defval
            else:
                (lentxt,) = struct.unpack("@i", self.fs.read(struct.calcsize("@i")))
                (val,) = struct.unpack(
                    f"@{lentxt}s", self.fs.read(struct.calcsize(f"@{lentxt}s"))
                )
                return val.decode("utf-8") if isinstance(val, bytes) else defval
        elif fmt in ["f", "d"]:
            # float
            (val,) = struct.unpack(f"@{fmt}", self.fs.read(struct.calcsize(f"@{fmt}")))
            return val if not math.isnan(val) else defval
        else:
            # int
            (isval,) = struct.unpack("@?", self.fs.read(struct.calcsize("@?")))
            if fmt == "?":
                return True if isval else defval
            elif isval:
                (val,) = struct.unpack(f"@{fmt}", self.fs.read(struct.calcsize(f"@{fmt}")))
                return val
            else:
                return defval


    async def readCase(self):
        lencase = int(await self.readPack("i", 0))

        async def getiCase():
            id = await self.readPack("i", 0)
            name = await self.readPack("s")
            description = await self.readPack("s")
            type = await self.readPack("h")
            updated_at = await self.readPack("q")
            lidcase = int(await self.readPack("i"))
            multicase = (
                []
                if type != -1 or lidcase == 0
                else [await self.readPack("i") for ii in range(lidcase)]
            )
            evaluator = await self.readPack("s") if self.vfl >= 10 else ""
            evaluator_date = await self.readPack("q") if self.vfl >= 10 else updated_at
            return {
                "id": id,
                "name": name,
                "description": description,
                "type": type,
                "updated_at": updated_at,
                "multicase": multicase,
                "evaluator": evaluator,
                "evaluator_date": evaluator_date,
            }

        return [await getiCase() for i in range(lencase)]

    async def readTable(self, fmtType: dict | List):
        def getfmt(index):
            return fmtType[index] if index < fmtType else fmtType[-1]

        lenTable = int(await self.readPack("i", 0))
        if lenTable > 0:
            if isinstance(fmtType, dict):
                return [
                    {
                        f"{key}": await self.readPack(fmtType[key])
                        for idx, key in enumerate(fmtType.keys())
                    }
                    for i in range(lenTable)
                ]
            else:
                cols = int(await self.readPack("i", 0))
                return [
                    [await self.readPack(getfmt(ii)) for ii in range(cols)]
                    for i in range(lenTable)
                ]
        else:
            if isinstance(fmtType, dict):
                return [{f"{key}": None for idx, key in enumerate(fmtType.keys())}]
            else:
                return [[None] * len(fmtType)]

    async def readTax(self):
        return {
            "mode": await self.readPack("h", 0),
            "rate_init": await self.readPack("d", 0.0),
            "multi": await self.readTable({"year": "i", "rate": "d"}),
        }
    
    async def readDep(self):
        return {
            "depreciation_method": await self.readPack("h", 0),
            "decline_factor": await self.readPack("d", 0.0),
        }

    async def readInflation(self):
        return {
            "mode": await self.readPack("h", 0),
            "rate_init": await self.readPack("d", 0.0),
            "multi": await self.readTable({"year": "i", "rate": "d"}),
        }

    async def readVAT(self):
        return {
            "mode": await self.readPack("h", 0),
            "rate_init": await self.readPack("d", 0.0),
            "multi": await self.readTable({"year": "i", "rate": "d"}),
        }

    async def readLBT(self):
        return {
            "mode": await self.readPack("h", 0),
            "rate_init": await self.readPack("d", 0.0),
            "multi": await self.readTable({"year": "i", "rate": "d"}),
        }


    async def readFiscalBase(self):
        return {
            "transferred_unrec_cost": await self.readPack("d", 0.0),
            "Tax": await self.readTax(),
            "tax_payment_config": await self.readPack("h", 0),
            "asr_future_rate": await self.readPack("d", 0.0),
            "Depreciation": await self.readDep(),
            "Inflation": await self.readInflation(),
            "VAT": await self.readVAT(),
            "LBT": await self.readLBT(),
            "vat_discount": await self.readPack("d", 0.0),
            "lbt_discount": await self.readPack("d", 0.0),
            "npv_mode": await self.readPack("h", 0),
            "discounting_mode": await self.readPack("h", 0),
            "sulfur_revenue_config": await self.readPack("h", 0),
            "electricity_revenue_config": await self.readPack("h", 0),
            "co2_revenue_config": await self.readPack("h", 0),
            "sunk_cost_reference_year": await self.readPack("i", 0),
            # versi 13
            "profitability_discounted": (
                await self.readPack("?", False) if self.vfl >= 13 else False
            ),
            "regime": await self.readPack("h", 3) if self.vfl >= 13 else 3,
            # versi 18
            "sum_undepreciated_cost": (
                await self.readPack("?", True) if self.vfl >= 18 else True
            ),
        }

    async def readProducer(self):
        async def readGSA():
            # read len keys
            lenKey = int(await self.readPack("i", 0))
            gsa = {}
            for i in range(lenKey // 3):
                gsa.update(
                    {
                        f"vol{i+1}": await self.readPack("d"),
                        f"ghv{i+1}": await self.readPack("d"),
                        f"price{i+1}": await self.readPack("d"),
                    }
                )
            return gsa

        async def readTable_(tipeProd: int):
            if tipeProd == 0:  # Oil Producer
                return await self.readTable(
                    (
                        {
                            "year": "i",
                            "sales": "d",
                            "price": "d",
                            "condensate_sales": "d",
                            "condensate_price": "d",
                            "base": "d",
                        }
                        if self.vfl >= 13
                        else {
                            "year": "i",
                            "sales": "d",
                            "price": "d",
                            "condensate_sales": "d",
                            "condensate_price": "d",
                        }
                    )
                )
            elif tipeProd == 1:  # Gas Producer
                len_iTable = int(await self.readPack("i", 0))
                return [
                    (
                        {
                            "year": await self.readPack("i"),
                            "production": await self.readPack("d"),
                            "gsa": await readGSA(),
                            "base": await self.readPack("d"),
                        }
                        if self.vfl >= 13
                        else {
                            "year": await self.readPack("i"),
                            "production": await self.readPack("d"),
                            "gsa": await readGSA(),
                        }
                    )
                    for i in range(len_iTable)
                ]
            else:
                return await self.readTable(
                    (
                        {"year": "i", "sales": "d", "price": "d", "base": "d"}
                        if self.vfl >= 13 else {"year": "i", "sales": "d", "price": "d"}
                    )
                )

        async def readProdPrice(tipeProd: int):
            TableProdPrice = []
            # num Producer
            lenProd = int(await self.readPack("i", 0))
            for iProd in range(lenProd):
                iTable = await readTable_(tipeProd)
                if not iTable:
                    # def value
                    if tipeProd == 0:  # Oil Producer
                        TableProdPrice.append(
                            [
                                {
                                    "year": None,
                                    "sales": None,
                                    "price": None,
                                    "condensate_sales": None,
                                    "condensate_price": None,
                                    "base": None,
                                }
                            ]
                        )
                    elif tipeProd == 1:  # Gas Producer
                        TableProdPrice.append(
                            [
                                {
                                    "year": None,
                                    "production": None,
                                    "gsa": {"vol1": None, "ghv1": None, "price1": None},
                                    "base": None,
                                }
                            ]
                        )
                    else:
                        TableProdPrice.append(
                            [{"year": None, "sales": None, "price": None, "base": None}]
                        )
                else:
                    TableProdPrice.append(iTable)
            return TableProdPrice

        async def readProd():
            tipeProd = int(await self.readPack("h", 0))
            return {
                "Tipe": tipeProd,
                "onstream_date": await self.readPack("q", 0),
                "ProdNumber": await self.readPack("h", 1),
                "GSANumber": await self.readPack("h", 1),
                "prod_price": await readProdPrice(tipeProd),
            }

        lenProd = int(await self.readPack("i", 0))
        if lenProd > 0:
            return [await readProd() for i in range(lenProd)]
        else:
            return []


    async def readDMO(self):
        return {
            "holiday": await self.readPack("?", False),
            "period": await self.readPack("i", 0),
            "start_production": await self.readPack("q", 0),
            "volume": await self.readPack("d", 0.0),
            "fee": await self.readPack("d", 0.0),
        }

    async def readcostRecConfig(self):
        async def readFTP():
            return {
                "ftp_availability": await self.readPack("?", False),
                "ftp_is_shared": await self.readPack("?", False),
                "ftp_portion": await self.readPack("d", 0.0),
            }

        async def readTaxSplit():
            return {
                "split_type": await self.readPack("h", 0),
                "pre_tax_ctr_oil": await self.readPack("d", 0.0),
                "pre_tax_ctr_gas": await self.readPack("d", 0.0),
            }

        async def readIC():
            return {
                "ic_availability": await self.readPack("?", False),
                "ic_oil": await self.readPack("d", 0.0),
                "ic_gas": await self.readPack("d", 0.0),
            }

        async def readCR():
            return {
                "oil_cr_cap_rate": await self.readPack("d", 0.0),
                "gas_cr_cap_rate": await self.readPack("d", 0.0),
            }

        costrec_ = {
            "oil_ftp": await readFTP(),
            "gas_ftp": await readFTP(),
            "TaxSplit": await readTaxSplit(),
            "IC": await readIC(),
            "CR": await readCR(),
            "RCSlidingScale": await self.readTable(
                {
                    "bottom_limit": "d",
                    "top_limit": "d",
                    "pre_tax_ctr_oil": "d",
                    "pre_tax_ctr_gas": "d",
                }
            ),
            "ICPSlidingScale": await self.readTable(
                {
                    "bottom_limit": "d",
                    "top_limit": "d",
                    "pre_tax_ctr_oil": "d",
                    "pre_tax_ctr_gas": "d",
                }
            ),
            "Indicator": await self.readTable({"year": "i", "indicator": "d"}),
            "dmo_is_weighted": await self.readPack("?", False),
            "OilDMO": await self.readDMO(),
            "GasDMO": await self.readDMO(),
        }
        if self.vfl >= 7:
            costrec_.update({"post_uu_22_year2001": await self.readPack("?", True)})
        else:
            costrec_.update({"post_uu_22_year2001": True})
        if self.vfl >= 14:
            costrec_.update(
                {"oil_cost_of_sales_applied": await self.readPack("?", False)}
            )
            costrec_.update(
                {"gas_cost_of_sales_applied": await self.readPack("?", False)}
            )
        else:
            costrec_.update({"oil_cost_of_sales_applied": False})
            costrec_.update({"gas_cost_of_sales_applied": False})
        return costrec_

    async def readgsConfig(self):
        gs_ = {
            "field_status": await self.readPack("h", 0),
            "field_location": await self.readPack("h", 0),
            "reservoir_depth": await self.readPack("h", 0),
            "infrastructure_availability": await self.readPack("h", 0),
            "reservoir_type": await self.readPack("h",  0),
            "co2_content": await self.readPack("h",  0),
            "h2s_content": await self.readPack("h",  0),
            "oil_api": await self.readPack("h",  0),
            "domestic_content_use": await self.readPack("h", 0),
            "production_stage": await self.readPack("h", 0),
            "ministry_discretion_split": await self.readPack("d", 0.0),
            "oil_base_split": await self.readPack("d", 0.0),
            "gas_base_split": await self.readPack("d", 0.0),
            "dmo_is_weighted": await self.readPack("?", False),
            "OilDMO": await self.readDMO(),
            "GasDMO": await self.readDMO(),
            # vfl>=7
            "cum_production_split_offset": {"mode": 0, "offset": 0, "split": []},
            # vfl>=11
            "amortization": False,
            # vfl>=20
            "field_reserves": 0
        }
        if self.vfl >= 7:
            gs_["cum_production_split_offset"]["mode"] = await self.readPack("h", 0)
            gs_["cum_production_split_offset"]["offset"] = await self.readPack("d", 0)
            splitTable = await self.readTable({"year": "i", "split": "d"})
            if isinstance(splitTable, bool):
                splitTable = []
            gs_["cum_production_split_offset"]["split"] = splitTable
        if self.vfl >= 11:
            gs_["amortization"] = await self.readPack("?", False)
        # if self.vfl >= 20:
        #     gs_["field_reserves"] = await self.readPack("h", 0.0)

        return gs_

    async def readcontrats(self, tipe: int):
        result = {
            "cr": await self.readcostRecConfig(),
            "gs": await self.readgsConfig(),
            "second": None,
        }
        if tipe >= 3:
            result.update(
                {"second": await self.readcostRecConfig()}
                if tipe in [3, 6]
                else {"second": await self.readgsConfig()}
            )
        return result

    async def readCosts(self, mode: int):
        async def readTableList(fmt: List, vt: List) -> List[Any]:
            async def readField(index: int | str, ivt: str):
                val = await self.readPack(ivt)
                if (index == 1 or index == "cost_allocation") and val is not None:
                    return "Gas" if val == 1 else "Oil"
                elif (
                    mode == 0 and (index == 6 or index == "is_ic_applied")
                ) and val is not None:
                    return "Yes" if val == 1 else "No"
                return val

            lenTable = int(await self.readPack("i", 0))
            if lenTable > 0:
                if self.vfl >= 19:
                    return [
                        {
                            f"{k}": await readField(k, fmt["fmt"][k])
                            for ir, k in enumerate(fmt["fmt"].keys())
                        }
                        for i in range(lenTable)
                    ]
                else:
                    result_ = []
                    cols = await self.readPack("i", 0)
                    cvt = vt
                    if cols > 0 and cols > len(vt):
                        cvt = vt + ["s"] * (cols - len(vt))
                    for i in range(lenTable):
                        rows = {f"{k}": None for ik, k in enumerate(fmt["fmt"].keys())}
                        if cols > 0:
                            vrow = [await readField(ii, ivt) for ii, ivt in enumerate(cvt)]
                            for ik, k in enumerate(fmt["fmt"].keys()):
                                if fmt["colMap"][ik] is not None and fmt["colMap"][
                                    ik
                                ] < len(vrow):
                                    rows[k] = vrow[fmt["colMap"][ik]]
                        result_.append(rows)
                    return result_
            else:
                return [{f"{k}": None for ik, k in enumerate(fmt["fmt"].keys())}]

        # fmt = ["d"]
        if mode == 0:  # tangible
            fmt = {
                "fmt": {
                    "expense_year": "i",
                    "cost_allocation": "h",
                    "cost": "d",
                    "pis_year": "i",
                    "useful_life": "i",
                    "depreciation_factor": "d",
                    "is_ic_applied": "h",
                    "tax_portion": "d",
                    "description": "s",
                },
                "colMap": [0, 1, 2, 3, 4, 5, 6, 7, 8],
            }
            return await readTableList(fmt, ["i", "h", "d", "i", "i", "d", "h", "d", "s"])
        elif mode == 1:  # intangible
            fmt = {
                "fmt": {
                    "expense_year": "i",
                    "cost_allocation": "h",
                    "cost": "d",
                    "tax_portion": "d",
                    "description": "s",
                },
                "colMap": [0, 1, 2, 3, 4],
            }
            return await readTableList(fmt, ["i", "h", "d", "d", "s"])
        elif mode == 2:  # opex
            fmt = {
                "fmt": {
                    "expense_year": "i",
                    "cost_allocation": "h",
                    "fixed_cost": "d",
                    "prod_rate": "d",
                    "cost_per_volume": "d",
                    "tax_portion": "d",
                    "description": "s",
                },
                "colMap": [0, 1, 2, 3, 4, 5, 7],
            }
            return await readTableList(fmt, ["i", "h", "d", "d", "d", "d", "d", "s"])
        elif mode == 3:  # asr
            fmt = {
                "fmt": {
                    "expense_year": "i",
                    "cost_allocation": "h",
                    "cost": "d",
                    "final_year": "i",
                    "tax_portion": "d",
                    "description": "s",
                },
                "colMap": [0, 1, 2, None, None, 3],
            }
            return await readTableList(fmt, ["i", "h", "d", "s"])
        elif mode == 4:  # COS
            fmt = {
                "fmt": {
                    "expense_year": "i",
                    "cost_allocation": "h",
                    "cost": "d",
                    "tax_portion": "d",
                    "description": "s",
                },
                "colMap": [0, 1, 2, None, None],
            }
            return await readTableList(fmt, ["i", "h", "d"])
        elif mode == 5:  # LBT
            fmt = {
                "fmt": {
                    "expense_year": "i",
                    "cost_allocation": "h",
                    "cost": "d",
                    "tax_portion": "d",
                    "description": "s",
                    "final_year": "i",
                    "utilized_land_area": "d",
                    "utilized_building_area": "d",
                    "njop_land": "d",
                    "njop_building": "d",
                    "gross_revenue": "d",
                },
                "colMap": [0, 1, 2, 3, 4, None, None, None, None, None, None],
            }
            return await readTableList(fmt, ["i", "h", "d", "d", "s"])
        return []

    async def readMonteRes(self):
        hasResult = await self.readPack("?", False)
        async def readResult():
            results = {"params": [], "results": [], "P10": [], "P50": [], "P90": []}
            # read params
            lenParam = int(await self.readPack("i", 0))
            if lenParam:
                results["params"] = (
                    [await self.readPack("s") for i in range(lenParam)]
                    if lenParam
                    else []
                )
            # read results
            lenRows = int(await self.readPack("i", 0))
            if lenRows:
                lenCols = int(await self.readPack("i", 0))
                results["results"] = [
                    [await self.readPack("d") for c in range(lenCols)]
                    for r in range(lenRows)
                ]
            # read percentile
            for p in [10, 50, 90]:
                lenP = int(await self.readPack("i", 0))
                if lenP:
                    results[f"P{p}"] = [await self.readPack("d") for c in range(lenP)]

            return results

        if hasResult:
            hashid = await self.readPack("s")
            result = await readResult() if hashid is not None else None
            return {"hash": hashid, "result": result}

        return {"hash": None, "result": None}

    async def readMonte(self):
        numsim = int(await self.readPack("i", 1000))
        lenParams = int(await self.readPack("i", 0))
        if lenParams:
            montecfgs_ = {
                "numsim": numsim,
                "params": [
                    {
                        "id": await self.readPack("i"),
                        "dist": await self.readPack("h"),
                        "min": await self.readPack("d"),
                        "max": await self.readPack("d"),
                        "base": await self.readPack("d"),
                        "stddev": await self.readPack("d"),
                    }
                    for i in range(lenParams)
                ],
            }
            return montecfgs_
        else:
            return {
                "numsim": 1000,
                "params": [
                    {
                        "id": 0,
                        "dist": 2,
                        "min": None,
                        "max": None,
                        "base": None,
                        "stddev": 1.25,
                    },
                    {
                        "id": 1,
                        "dist": 2,
                        "min": None,
                        "max": None,
                        "base": None,
                        "stddev": 1.25,
                    },
                    {
                        "id": 2,
                        "dist": 2,
                        "min": None,
                        "max": None,
                        "base": None,
                        "stddev": 1.25,
                    },
                    {
                        "id": 3,
                        "dist": 2,
                        "min": None,
                        "max": None,
                        "base": None,
                        "stddev": 1.25,
                    },
                    {
                        "id": 4,
                        "dist": 2,
                        "min": None,
                        "max": None,
                        "base": None,
                        "stddev": 1.25,
                    },
                ]
            }

    def defOptimCfg(self):
        return {
            "target_parameter": 0,
            "target_optimization": 0.0,
            "optimization": [
                {
                    "parameter": i,
                    "min": 0.3 if i in [0, 1] else 0.4 if i == 9 else 0.2,
                    "max": (
                        0.6
                        if i in [0, 1]
                        else 0.44
                        if i == 9
                        else 1.0
                        if i in [6, 7]
                        else 0.4
                    ),
                    "pos": i,
                    "checked": False,
                }
                for i in range(11)
            ],
        }

    async def readOptim(self):
        target_parameter = int(await self.readPack("h", 0))
        target_optimization = await self.readPack("d")
        lenOpti = int(await self.readPack("i", 0))
        if lenOpti:
            return {
                "target_parameter": target_parameter,
                "target_optimization": target_optimization,
                "optimization": [
                    {
                        "parameter": await self.readPack("h"),
                        "min": await self.readPack("d"),
                        "max": await self.readPack("d"),
                        "pos": await self.readPack("h"),
                        "checked": await self.readPack("?"),
                    }
                    for i in range(lenOpti)
                ],
            }
        else:
            return self.defOptimCfg()

    async def extractProject(self):
        cases = await self.readCase()
        for idx, icase in enumerate(cases):
            id = icase["id"]
            type_of_contract = int(await self.readPack("i", 1))
            icase['genConf'] = {
                "type_of_contract": type_of_contract,
                "discount_rate_start_year": await self.readPack("i", 0),
                "discount_rate": await self.readPack("d", 0.0),
                "inflation_rate_applied_to": await self.readPack("h", 0),
                "start_date_project": await self.readPack("q", 0),
                "end_date_project": await self.readPack("q", 0),
                "start_date_project_second": await self.readPack("q", 0),
                "end_date_project_second": await self.readPack("q", 0),
                # start ver. 12
                "delayAccMode": await self.readPack("h", 0) if self.vfl >= 12 else 0,
                "delayAccYear": await self.readPack("h", 0) if self.vfl >= 12 else 0,
                # start ver. 15
                "useCOS": await self.readPack("?", 0) if self.vfl >= 15 else False,
                # start ver. 19
                "lbtUseCalc": await self.readPack("?", 0) if self.vfl >= 19 else False,
            }
            icase['fiscal'] = {
                "Fiskal": await self.readFiscalBase(),
                "Fiskal2": await self.readFiscalBase(),
            }
            icase['producer'] = await self.readProducer()
            icase['contracts'] = await self.readcontrats(type_of_contract)
            icase['tangible'] = await self.readCosts(0)
            icase['intangible'] = await self.readCosts(1)
            icase['opex'] = await self.readCosts(2)
            icase['asr'] = await self.readCosts(3)

            # error! miss write in ver.14, fix on ver.15
            if self.vfl >= 15:
                icase['cos'] = await self.readCosts(4)
            if self.vfl >= 16:
                icase['lbt'] = await self.readCosts(5)

        if self.vfl >= 5:
            # extract sens
            for idx, icase in enumerate(cases):
                icase['sens'] = [await self.readPack("d", 80.0) for i in range(2)]

            # extract monte
            for idx, icase in enumerate(cases):
                icase['monte'] = {
                    'cfg': await self.readMonte(),
                    # read monte result
                    'result': await self.readMonteRes()
                }
            if self.vfl >= 6:
                # extract optim
                for idx, icase in enumerate(cases):
                    icase['optim'] = await self.readOptim()

        return cases

    @staticmethod
    async def getHeaderCase(file: UploadFile = File(...)):
      __hfl = "pySCapp"
      __vfl = 19

      try:
        # read header
        magic, version = struct.unpack("@7si", await file.read(struct.calcsize("@7si")))
        return {
          "magic": magic,
          "version": version
        }
      except Exception as e:
        print(e, flush=True)
        return { "magic": None, "version": None }

    @staticmethod
    async def getCases(file_bytes: bytes):
      __hfl = "pySCapp"
      __vfl = 19

      try:
        # read header
        magic, version = struct.unpack("@7si", file_bytes[:struct.calcsize("@7si")])
        __vfl = version
        # read cases
        header_size = struct.calcsize("@7si")
        extractor = pscExtractor(__hfl, __vfl, file_bytes[header_size:])
        
        cases = await extractor.readCase()
        # chg key type in cases to ctrType
        for case in cases:
          case['ctrType'] = case.pop('type', None)
        return {
          "header": {
            "magic": magic,
            "version": version
          },
          "payload": {
            "tbCases": cases
          }
        }
      except Exception as e:
        print(traceback.format_exc(), flush=True)
        print(e, flush=True)
        return { 
          "header": { "magic": None, "version": None },
          "cases": []
        }

    @staticmethod
    async def getCasesData(split_ids, file_bytes = bytes):
      __hfl = "pySCapp"
      __vfl = 19
      # read header
      magic, version = struct.unpack("@7si", file_bytes[:struct.calcsize("@7si")])
      __vfl = version
      # read cases
      header_size = struct.calcsize("@7si")
      extractor = pscExtractor(__hfl, __vfl, file_bytes[header_size:])
      
      cases = await extractor.extractProject()

      # filter by split_ids
      if split_ids and len(split_ids) > 0:
        cases = [case for case in cases if case['id'] in split_ids]
      return cases



