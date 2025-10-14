import json
import os
import struct
from collections import defaultdict
from pprint import pprint
from fastapi import File, UploadFile
import traceback

import jsonref
from construct import (
    Array,
    Byte,
    Const,
    CString,
    Float64l,
    IfThenElse,
    Int8ul,
    Int32ul,
    Pass,
    PrefixedArray,
    Struct,
)

from jschon import JSON, JSONSchema, create_catalog

from Crypto.Protocol.KDF import PBKDF2
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
from Crypto.Hash import SHA1

def print_struct_fields(struct, indent=0):
    pad = "  " * indent
    if hasattr(struct, "subcons"):
        for sub in struct.subcons:
            # print(f"{pad}- {sub.name}: {type(sub.subcon).__name__}")
            print_struct_fields(sub.subcon, indent + 1)

class BytesEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, bytes):
      return obj.decode("utf-8", errors="replace")  # or use .hex() for binary
    return super().default(obj)
    
class pscPacks:
  __hfl: str = "pSCWapp"
  __latest_schema: str = "project-schema-v2.json"
  __vfl: int = 2
  # version 1 gunakan traping, switch schema manual (ada kesalahan pada schema v1, di gsschema)
  # version 2 gunakan schema v2 sebagai initial

  __type_map = {
    "int32": "i",
    "int64": "q",
    "float64": "d",
    "bool": "?",
    "string": None  # handled separately
  }

  def __init__(self):
    pass

  def load_schema_version(self, version: int =  None) -> dict:
    if version == 1:
      shemaname = 'project-schema.json'
    elif version is not None:
      shemaname = f'project-schema-v{version}.json'
    else:
      shemaname = self.__latest_schema  

    return self.load_schema_with_refs(f"psc-schema/{shemaname}")

  def pack_header(self, version: int) -> bytes:
    """
    Mengemas informasi header ke dalam objek bytes.

    Args:
            version (int): Nomor versi yang akan dimasukkan ke header.

    Returns:
            bytes: Header yang sudah dikemas, terdiri dari string magic 7-byte dan 1-byte versi.
    """
    magic = self.__hfl.encode()  # 7 byte
    return magic + struct.pack("<B", version)  # 1 byte versi

  def parse_header(self, binary: bytes) -> dict:
    """
    Mengurai header dari file biner dan mengekstrak magic bytes serta versi.

    Args:
      binary (bytes): Data biner yang berisi header untuk diurai.

    Returns:
      dict: Dictionary dengan kunci berikut:
        - "magic" (bytes): 7 byte pertama dari header, biasanya sebagai signature file.
        - "version" (int): Nomor versi yang diambil dari byte ke-8 header.
    """
    magic = binary[:7]
    version = struct.unpack("<B", binary[7:8])[0]
    return { "magic": magic, "version": version }

  def pack_scalar(self, value, layout_type):
      """
      Mengemas nilai skalar ke dalam buffer biner sesuai tipe layout yang diberikan.
      Jika input value berupa dictionary, maka diharapkan memiliki kunci "present" (apakah nilai ada)
      dan "value" (nilai yang akan dikemas). Jika bukan dictionary, kehadiran nilai ditentukan dari None.
      Args:
        value (Any atau dict): Nilai skalar yang akan dikemas, atau dictionary dengan kunci "present" dan "value".
        layout_type (str): Tipe nilai, digunakan untuk menentukan format struct packing.
      Returns:
        bytes: Buffer biner hasil kemas, diawali byte presence lalu nilai jika ada.
      Raises:
        KeyError: Jika layout_type tidak ditemukan di type map internal.
        struct.error: Jika nilai tidak bisa dikemas dengan format yang ditentukan.
      """
      if isinstance(value, dict):
        present = value.get("present", 0)
        value = value.get("value", None)
      # check if value is string
      elif value is not None and isinstance(value, str):
        present = 0 if value.strip() == "" else 1
        if present:
          #convert str to int or float or bool
          if self.__type_map[layout_type] == "i":
            value = int(value)
          elif self.__type_map[layout_type] == "q":
            value = int(value)
          elif self.__type_map[layout_type] == "d":
            value = float(value)
          elif self.__type_map[layout_type] == "?":
            value = value.lower() in ['true', '1', 'yes']
      else:
        present = 0 if value is None else 1

      fmt = "<" + self.__type_map[layout_type]
      buf = struct.pack("<B", present)
      if present:
          # print(f"Mengemas skalar: {layout_type} dengan nilai {value} menggunakan format {fmt}")
          buf += struct.pack(fmt, value)
      return buf
 
  def parse_scalar(self, binary: bytes, layout_type: str, offset: int, nullable=False):
    """
    Mengurai nilai skalar dari buffer biner pada offset tertentu, dengan dukungan nullability opsional.

    Args:
      binary (bytes): Buffer biner yang akan diurai.
      layout_type (str): Tipe nilai skalar yang akan diurai, digunakan untuk menentukan format struct.
      offset (int): Offset dalam buffer biner untuk mulai mengurai.
      nullable (bool, optional): Apakah nilai bisa null. Default False.

    Returns:
      tuple: Tuple berisi nilai hasil parsing (atau None jika tidak ada) dan offset terbaru.
    """
    present = struct.unpack_from("<B", binary, offset)[0]
    offset += 1
    if present:
        fmt = "<" + self.__type_map[layout_type]
        value = struct.unpack_from(fmt, binary, offset)[0]
        offset += struct.calcsize(fmt)
        return value, offset
    else:
        return None, offset

  def pack_nullable(self, value, layout_type):
      """
      Mengemas nilai nullable ke dalam format biner dengan flag kehadiran.

      Args:
        value: Nilai yang akan dikemas. Jika None, hanya flag kehadiran yang dikemas.
        layout_type: Tipe data yang digunakan untuk mengemas nilai jika ada.

      Returns:
        bytes: Objek bytes berisi flag kehadiran (1 byte) diikuti nilai yang dikemas jika ada.

      Flag kehadiran bernilai 0 jika value None, dan 1 jika ada nilainya. Jika ada, nilai dikemas
      menggunakan metode `pack_scalar` sesuai layout_type yang diberikan.
      """
      present = 0 if value is None else 1
      buf = struct.pack("<B", present)
      if present:
          buf += self.pack_scalar(value, layout_type)
      return buf

  def pack_string(self, value, nullable=False):
      """
      Mengemas nilai string ke dalam format biner dengan flag kehadiran opsional.

      Jika input berupa dictionary, diharapkan memiliki kunci "present" (int, 0 atau 1) dan "value" (str).
      Jika bukan dictionary, kehadiran ditentukan dari apakah value None atau string kosong/whitespace.

      Format hasil kemasan:
        - 1 byte: flag kehadiran (0 atau 1)
        - Jika hadir:
          - 4 byte: panjang string UTF-8 (unsigned int, little-endian)
          - N byte: string UTF-8

      Args:
        value (str atau dict): String yang akan dikemas, atau dict dengan kunci "present" dan "value".
        nullable (bool, opsional): Jika True, mengizinkan value None atau kosong. Default False.

      Returns:
        bytes: Representasi biner string yang sudah dikemas dengan flag kehadiran.
      """
      if isinstance(value, dict):
        present = value.get("present", 0)
        value = value.get("value", "")
      else:      
        present = 0 if value is None or (type(value) is str and value.strip() == "") else 1
      buf = struct.pack("<B", present)
      if present:
          encoded = value.encode("utf-8")
          buf += struct.pack("<I", len(encoded)) + encoded
      return buf

  def parse_string(self, binary: bytes, offset: int, nullable=False):
    """
    Mengurai string UTF-8 (nullable) dari buffer biner mulai dari offset tertentu.

    Format penyimpanan string:
      - 1 byte: flag kehadiran (0 = None, 1 = ada)
      - 4 byte: panjang string dalam byte (unsigned int, little-endian)
      - N byte: data string UTF-8 (jika ada)

    Args:
      binary (bytes): Buffer biner yang akan diurai.
      offset (int): Offset awal dalam buffer.
      nullable (bool, opsional): Apakah string bisa None. Default False.

    Returns:
      tuple[str | None, int]: Tuple berisi string hasil decode (atau None jika tidak ada)
                  dan offset baru setelah parsing.

    Raises:
      ValueError: Jika string gagal didekode sebagai UTF-8.
    """
    # print(f"Mengurai string pada offset {offset}")
    present = struct.unpack_from("<B", binary, offset)[0]
    offset += 1
    if not present:
        return None, offset
    length = struct.unpack_from("<I", binary, offset)[0]
    offset += 4
    raw = binary[offset:offset+length]
    # print(f"→ panjang: {length}, raw bytes: {raw.hex()}")
    try:
        value = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        print(f"[decode error] offset={offset}, length={length}, raw={raw[:20].hex()}...")
        raise ValueError(f"Gagal mendekode string pada offset {offset}: {e}")
    offset += length
    return value, offset
  
  def pack_nullable_string(self, value):
      present = 0 if value is None or value.strip() == "" else 1
      buf = struct.pack("<B", present)
      if present:
          buf += self.pack_string(value)
      return buf

  def pack_object(self, data, fields):
    """
    Mengemas objek Python ke format biner berdasarkan definisi field yang diberikan.
    Args:
      data (dict atau None): Data objek yang akan dikemas. Jika None, hanya 1 byte penanda ketidakhadiran yang dikemas.
      fields (list of dict): Daftar definisi field, setiap field berupa dictionary berisi:
        - "name" (str): Nama field.
        - "type" (str): Tipe field ("array", "object", "string", atau tipe skalar).
        - "default" (opsional): Nilai default untuk field.
        - "field" (list, opsional): Definisi field bertingkat untuk tipe "object".
    Returns:
      bytes: Representasi biner hasil kemasan objek.
    Raises:
      ValueError: Jika ditemukan tipe field yang tidak didukung.
    """
    buf = b""

    # jika data string contains { and } then substring to get object
    if isinstance(data, str):
      if "{" in data and "}" in data:
        data = data[data.index("{")+1:data.rindex("}")]
        try:
          data = json.loads("{" + data + "}")
        except json.JSONDecodeError:
          data = None
      else:
        data = None

    if data is None:
      return struct.pack("<B", 0)
    else:
      buf += struct.pack("<B", 1)

    for ifield in fields: 
      iname = ifield["name"]
      itype = ifield["type"] or None
      hasDefault = "default" in ifield    
      defValue = ifield.get("default") or None
      if defValue is None and not hasDefault and itype == "array":
          defValue = []
      idata = data[iname] if iname in data and data is not None else data if data is not None else defValue if hasDefault else None
      
      # print(f"Mengemas field '{iname}' tipe '{itype}' dengan nilai: {idata} default: {defValue}")

      if itype == "array":
        buf += self.pack_array(idata, ifield)
      elif itype == "object":
        nested = ifield["field"]
        buf += self.pack_object(idata, nested)
      elif itype == "string":
          buf += self.pack_string(idata)
      elif itype in self.__type_map:
          if itype != "int64" and iname in ['created_at', 'updated_at', 'onstream_date', 'evaluator_date', 'start_date_project', 'end_date_project', 'start_date_project_second', 'end_date_project_second', 'start_production']:
            itype = "int64"
          buf += self.pack_scalar(idata, itype)
      else:
          raise ValueError(f"Tipe field tidak didukung: {itype}")

    return buf

  def parse_object(self, binary: bytes, fields: list, offset: int):
    """
    Mengurai objek terstruktur dari buffer biner sesuai definisi field yang diberikan.
    Args:
      binary (bytes): Buffer data biner yang akan diurai.
      fields (list): Daftar dictionary yang mendeskripsikan field, masing-masing berisi kunci seperti "name", "type", "nullable", dan opsional "field" untuk objek bertingkat.
      offset (int): Offset awal dalam buffer biner.
    Returns:
      tuple: Tuple berisi:
        - dict atau None: Hasil parsing objek sebagai dictionary (nama field ke nilai), atau None jika objek tidak hadir.
        - int: Offset terbaru setelah parsing objek.
    Raises:
      ValueError: Jika ditemukan tipe field yang tidak didukung.
    Catatan:
      - Mendukung tipe field: "string", tipe skalar (sesuai self.__type_map), "object" (nested), dan "array".
      - Menangani field nullable dan flag kehadiran untuk objek.
    """
    # baca flag kehadiran objek
    present = struct.unpack_from("<B", binary, offset)[0] == 1
    offset += 1
    if not present:
      return None, offset

    result = {}

    for field in fields:
        name = field["name"]
        ftype = field["type"]
        # print(f"Mengurai field: {name} tipe {ftype} pada offset {offset}")
        nullable = field.get("nullable", False)

        if ftype == "string":
            value, offset = self.parse_string(binary, offset, nullable)
        elif ftype in self.__type_map:
            value, offset = self.parse_scalar(binary, ftype, offset, nullable)
        elif ftype == "object":
            present = struct.unpack_from("<B", binary, offset)[0]
            offset += 1
            if present:
                nested_fields = field.get("field")
                value, offset = self.parse_object(binary, nested_fields, offset)
            else:
                value = None
        elif ftype == "array":
            value, offset = self.parse_array(binary, field, offset)
        else:
            raise ValueError(f"Tipe field tidak didukung: {ftype}")

        result[name] = value
    return result, offset
  
  def extractName(self, name):
      if "." in name:
          return self.extractName(name.split(".")[-1])
      elif ":" in name:
          return self.extractName(name.split(":")[-1])
      else:
          return name

  def pack_array(self, data, field):
    """
    Mengemas field array ke dalam format biner menggunakan struct packing.
    Args:
      data (dict atau list atau None): Data input yang berisi array yang akan dikemas.
        Jika dict, array diambil menggunakan nama field. Jika list, langsung digunakan.
      field (dict): Definisi skema untuk field array, termasuk nama, tipe, default, dan definisi item.
    Returns:
      bytes: Representasi biner hasil kemasan array, termasuk jumlah elemen dan seluruh item.
    Raises:
      ValueError: Jika ditemukan tipe field yang tidak didukung saat proses pengemasan.
    Catatan:
      - Mendukung array dan objek bertingkat secara rekursif.
      - Mendukung nilai default untuk field yang tidak ada.
      - Penanganan khusus untuk field tanggal agar dikemas sebagai int64.
    """
    # if data is str contains [ and ] then substring to get array
    if isinstance(data, str): 
      if "[" in data and "]" in data:
        data = data[data.index("[")+1:data.rindex("]")]
        try:
          data = json.loads("[" + data + "]")
        except json.JSONDecodeError:
          data = []
      else:
        data = []

    if data is None:
      return struct.pack("<I", 0)

    if isinstance(data, list):
      # Jika sudah list, gunakan langsung
      dataItems = data
    else:
      # Jika dict, ambil array dari nama field
      dataItems = data.get(field["name"], field.get("default") or [])

    buf = struct.pack("<I", len(dataItems))  # jumlah elemen

    for dItem in dataItems:
      for ifield in field.get("items", []):
        iname = ifield["name"] or None
        itype = ifield["type"] or None
        hasDefault = "default" in field
        defValue = ifield.get("default") or None
        if defValue is None and not hasDefault and itype == "array":
            defValue = []
        if isinstance(dItem, dict):
            idata = dItem[iname] if iname in dItem and dItem is not None else dItem if dItem is not None else defValue if hasDefault else None
        else:
            idata = dItem if dItem is not None else defValue if hasDefault else None
        # print(f"Mengemas field '{iname}' tipe '{itype}' dengan nilai: {idata} default: {defValue}")

        if itype == "array":
            buf += self.pack_array(idata, ifield)
        elif itype == "object":
          nested = ifield["field"]
          buf += self.pack_object(idata, nested)
        elif itype == "string":
            buf += self.pack_string(idata)
        elif itype in self.__type_map:
            if itype != "int64" and iname in ['created_at', 'updated_at', 'onstream_date', 'evaluator_date', 'start_date_project', 'end_date_project', 'start_date_project_second', 'end_date_project_second', 'start_production']:
              itype = "int64"
            # print(f"Mengemas field '{iname}' tipe '{itype}' dengan nilai: {idata} default: {defValue}")
            buf += self.pack_scalar(idata, itype)
        else:
            raise ValueError(f"Tipe field tidak didukung: {itype}")

    return buf

  def parse_array(self, binary: bytes, field: dict, offset: int):
    """
    Mengurai struktur array dari buffer biner sesuai definisi field yang diberikan.
    Args:
      binary (bytes): Data biner yang akan diurai.
      field (dict): Dictionary yang mendeskripsikan field array, termasuk struktur item-nya.
      offset (int): Offset saat ini dalam data biner untuk mulai parsing.
    Returns:
      tuple: Tuple berisi:
        - items (list): Daftar item hasil parsing, masing-masing berupa dictionary atau nilai tergantung definisi item.
        - offset (int): Offset terbaru setelah parsing array.
    Raises:
      ValueError: Jika ditemukan tipe field item yang tidak didukung.
    """
    count = struct.unpack_from("<I", binary, offset)[0]
    offset += 4

    items = []
    for i in range(count):
      payload = {}
      for ifield in field.get("items", []): 
        iname = ifield["name"] or None
        itype = ifield["type"] or None
                    
        if itype == "array":
          val, offset = self.parse_array(binary, ifield, offset)
        elif itype == "object":
          nested = ifield["field"]
          val, offset = self.parse_object(binary, nested, offset)
        elif itype == "string":
          val, offset = self.parse_string(binary, offset)
        elif itype in self.__type_map:
            if itype != "int64" and iname in ['created_at', 'updated_at', 'onstream_date', 'evaluator_date', 'start_date_project', 'end_date_project', 'start_date_project_second', 'end_date_project_second', 'start_production']:
              itype = "int64"
            val, offset = self.parse_scalar(binary, itype, offset)
        else:
            raise ValueError(f"Tipe field tidak didukung: {itype}")
        if iname is not None:
          payload[iname] = val
        else:
          payload = val

      items.append(payload)

    return items, offset

  def extract_layout(self, schema: dict, name_prefix="") -> list:
      """
      Mengekstrak definisi layout dari dictionary skema JSON.
      Metode ini memproses skema JSON yang diberikan dan mengembalikan daftar definisi field,
      masing-masing direpresentasikan sebagai dictionary yang berisi metadata seperti nama, tipe, required,
      nullability, default, dan deskripsi. Mendukung objek dan array bertingkat secara rekursif.
      Args:
        schema (dict): Skema JSON yang akan diekstrak layout-nya.
        name_prefix (str, opsional): Prefix untuk nama field pada struktur bertingkat. Default "".
      Returns:
        list: Daftar dictionary, masing-masing mewakili field pada layout skema.
      """
      def merge_allOf(prop: dict) -> dict:
          if "allOf" not in prop:
              return prop
          merged = {}
          for sub in prop["allOf"]:
              merged.update(sub)
          return merged

      layout = []
      props = schema.get("properties", {})
      required = set(schema.get("required", []))

      for name, prop in props.items():
          prop = merge_allOf(prop)  
          # full_name = f"{name_prefix}{name}"
          full_name = name
          types = prop.get("type")

          # Normalisasi tipe dan nullable
          if isinstance(types, list):
              nullable = "null" in types
              types = [t for t in types if t != "null"]
              base_type = types[0] if types else "string"
          else:
              nullable = False
              base_type = types or "string"

          # Pemetaan ke tipe layout
          type_map = {
              "integer": "int32",
              "number": "float64",
              "boolean": "bool",
              "string": "string"
          }
          layout_type = type_map.get(base_type, base_type)
          # cek jika name termasuk field tanggal
          if name in ['created_at', 'updated_at', 'onstream_date', 'evaluator_date', 'start_date_project', 'end_date_project', 'start_date_project_second', 'end_date_project_second', 'start_production']:
              layout_type = "int64"

          field = {
              "name": full_name,
              "type": layout_type,
              "required": name in required
          }

          if nullable:
              field["nullable"] = True
          if "default" in prop:
              field["default"] = prop["default"]
          if "description" in prop:
              field["description"] = prop["description"]

          # Penanganan array
          if base_type == "array" and "items" in prop:
              item_schema = prop["items"]
              if item_schema.get("type") == "object":
                  field["items"] = self.extract_layout(item_schema, name_prefix=f"{full_name}[]:")
              else:
                  item_type = item_schema.get("type", "string")
                  if isinstance(item_type, list):
                      nullable = "null" in item_type
                      item_type = [t for t in item_type if t != "null"]
                      item_type = item_type[0] if item_type else "string"
                  else:
                      nullable = False
                      item_type = item_type or "string"

                  field["items"] = [{
                      "name": full_name,
                      "type": type_map.get(item_type, item_type)
                  }]

          # Penanganan object
          elif base_type == "object":
              nested_fields = self.extract_layout(prop, name_prefix="")
              field = {
                  **field,
                  "field": nested_fields
              }

          layout.append(field)

      return layout


  def load_schema_with_refs(self, path: str) -> dict:
      """
      Memuat skema JSON dari path file yang diberikan dan menyelesaikan semua referensi $ref lokal.
      Args:
        path (str): Path file ke skema JSON utama.
      Returns:
        dict: Skema JSON yang sudah sepenuhnya di-resolve sebagai dictionary, dengan semua $ref lokal dimuat.
      Efek Samping:
        Mencetak direktori dasar dan path absolut file skema untuk debugging.
      Raises:
        FileNotFoundError: Jika file skema utama atau file referensi tidak ditemukan.
        json.JSONDecodeError: Jika ada file yang berisi JSON tidak valid.
      """
      # Pastikan path absolut dan folder dasar
      abs_path = os.path.abspath(path)
      base_dir = os.path.dirname(abs_path)
      # print(f"Direktori dasar: {base_dir}")
      # print(f"Path absolut: {abs_path}")

      # Baca file utama
      with open(abs_path, "r", encoding="utf-8") as f:
          raw = f.read()

      # Loader untuk $ref lokal
      def loader(uri):
          ref_path = os.path.join(base_dir, uri)
          with open(ref_path, "r", encoding="utf-8") as rf:
              return json.load(rf)

      # Resolve semua $ref
      return jsonref.loads(raw, loader=loader, jsonschema=True)

  def encode_nullable(self, val, encoder):
      if val is None:
          return struct.pack("?", False)
      else:
          return struct.pack("?", True) + encoder(val)

  def build_scalar(self, ftype):
      return {
          "int32": Int32ul,
          "float64": Float64l,
          "bool": Byte,
          "string": CString("utf8")
      }.get(ftype)
      
  def collect_nested_fields(self, parent_name):
    prefix = parent_name + "."
    return [
        {**f, "name": f["name"].split(".")[-1]}
        for f in self.current_layout
        if f["name"].startswith(prefix)
    ]      
  
  def layout_to_struct(self, layout):
      """
      Mengonversi definisi layout menjadi objek Struct (dari library construct).

      Args:
        layout (list): Daftar definisi field, di mana setiap field adalah dictionary
          yang minimal memiliki kunci "name".

      Returns:
        Struct: Objek Struct yang terdiri dari field-field hasil build dari layout.
          Hanya field yang "name"-nya tidak mengandung titik (".") yang disertakan.

      Efek Samping:
        Menetapkan self.current_layout ke layout yang diberikan.
      """
      self.current_layout = layout
      fields = [self.build_field(f) for f in layout if "." not in f["name"]]
      return Struct(*fields)

  def resolve_type(self, ftype):
    if isinstance(ftype, list):
        if "integer" in ftype:
            return "int32"
        elif "float" in ftype:
            return "float64"
        elif "string" in ftype:
            return "string"
        elif "boolean" in ftype:
            return "bool"
        elif "object" in ftype:
            return "object"
        elif "array" in ftype:
            return "array"
        else:
            return ftype[0]
    return ftype


  def collect_nested_layout(self, parent_name, layout):
    prefix = parent_name + "."
    result = [
        {**f, "name": f["name"].split(".")[-1]}
        for f in layout
        if f["name"].startswith(prefix)
    ]
    # print(f"Collected nested layout for '{parent_name}' → {len(result)} fields")
    return result

  def print_struct_layout(self, struct, indent=0):
    """
    Mencetak layout dari objek Struct (construct) secara rekursif, menampilkan nama dan tipe subconstruct-nya.
    Args:
      struct: Objek Struct atau subconstruct dari library construct yang ingin dicetak layout-nya.
      indent (int, opsional): Level indentasi untuk tampilan nested. Default 0.
    Catatan:
      - Mendukung struktur bertingkat dengan rekursi pada subconstruct.
      - Menampilkan "(anonymous)" jika subconstruct tidak bernama.
      - Dirancang untuk digunakan dengan Struct dari library 'construct'.
    """
    pad = "  " * indent
    if hasattr(struct, "subcons"):
        for sub in struct.subcons:
            name = sub.name or "(anonymous)"
            typename = type(sub.subcon).__name__

            # print(f"{pad}- {name}: {typename}")

            # Rekursif jika nested Struct, Array, atau wrapper
            if hasattr(sub.subcon, "subcons"):
                self.print_struct_layout(sub.subcon, indent + 1)
            elif hasattr(sub.subcon, "subcon"):
                self.print_struct_layout(sub.subcon, indent + 1)

  def packData(self, data: dict) -> bytes:
    """
    Mengubah dictionary data menjadi objek bytes sesuai layout skema.
    Langkah-langkah:
      1. Memuat skema JSON (termasuk semua $ref) yang mendefinisikan struktur data.
      2. Mengekstrak layout dari skema, berisi field dan tipe datanya.
      3. Mengemas header menggunakan label versi/format internal.
      4. Untuk setiap field di layout:
         - Menentukan tipe field dan mengambil nilai dari data (atau default).
         - Untuk array: gunakan `pack_array`.
         - Untuk object: gunakan `pack_object`.
         - Untuk string: gunakan `pack_string`.
         - Untuk tipe skalar: gunakan `pack_scalar`, dengan perlakuan khusus untuk field tanggal.
         - Jika tipe tidak didukung, raise ValueError.
    Args:
      data (dict): Data yang akan diserialisasi, key sesuai nama field di skema.
    Returns:
      bytes: Representasi biner dari data input.
    Raises:
      ValueError: Jika ditemukan tipe field yang tidak didukung saat serialisasi.
    """
    # print(data)

    # Memuat skema beserta seluruh $ref
    # gunakan latest version
    schema = self.load_schema_with_refs(f"psc-schema/{self.__latest_schema}")

    # Ekstrak layout dari skema
    layout = self.extract_layout(schema)
    print(f"Layout terdeteksi: {len(layout)} field")

    # Encode header
    buf = self.pack_header(self.__vfl)

    # Encode payload sesuai layout
    for ifield in layout:
        iname = self.extractName(ifield["name"])
        itype = ifield["type"]
        defValue = ifield.get("default") or None
        if defValue is None and itype == "array":
            defValue = []
        idata = data[iname] if iname in data else defValue

        # print(f"Mengemas field '{iname}' tipe '{itype}' dengan nilai: {idata} default: {ifield.get('default') or None}")
        print(f"Mengemas field '{iname}' tipe {itype}")
        if itype == "array":
            buf += self.pack_array(idata, ifield)
        elif itype == "object":
            buf += self.pack_object(idata, ifield)
        elif itype == "string":
            buf += self.pack_string(idata)
        elif itype in self.__type_map:
            if itype != "int64" and iname in ['created_at', 'updated_at', 'onstream_date', 'evaluator_date', 'start_date_project', 'end_date_project', 'start_date_project_second', 'end_date_project_second', 'start_production']:
                itype = "int64"
            buf += self.pack_scalar(idata, itype)
        # else:
            # raise ValueError(f"Tipe field tidak didukung: {itype}")

    return buf

  def parseData(self, binary: bytes, usever: int = 2) -> dict:
    """
    Mengurai data biner sesuai skema dan mengekstrak informasi terstruktur.
    Args:
      binary (bytes): Data biner yang akan diurai.
    Returns:
      dict: Dictionary berisi header hasil parsing dan payload. Header memuat metadata
        seperti magic bytes dan versi. Payload adalah dictionary nama field ke nilai hasil parsing,
        sesuai layout skema.
    Raises:
      ValueError: Jika ditemukan tipe field yang tidak didukung saat parsing.
    Catatan:
      - Layout skema dimuat berdasarkan versi yang ditemukan di header.
      - Mendukung parsing array, object, string, dan tipe skalar sesuai skema.
      - Menangani field nullable dan flag kehadiran untuk object.
    """
    offset = 0

    # Parse header
    magic = binary[offset:offset+7]
    offset += 7
    version = struct.unpack_from("<B", binary, offset)[0]
    offset += 1

    header = {
        "magic": magic,
        "version": version
    }
    print(f"Header: {header}")
    if magic != self.__hfl.encode():
        raise ValueError("file header bytes tidak cocok")
    elif version > self.__vfl or version <= 0:
        raise ValueError("file version tidak cocok")
    # TODO: add konverisi data


    # Load layout by version
    if version == 1:
      shemaname = 'project-schema.json' if usever == 1 else 'project-schema-v2.json'
    else:
      shemaname = f'project-schema-v{version}.json'
    schema = self.load_schema_with_refs(f"psc-schema/{shemaname}")
    layout = self.extract_layout(schema)

    # Parse payload
    payload = {}
    for field in layout:
        ftype = field["type"]
        name = field["name"]

        if ftype == "array":
            value, offset = self.parse_array(binary, field, offset)
        elif ftype == "object":
            present = struct.unpack_from("<B", binary, offset)[0]
            offset += 1
            if present:
                nested_fields = field.get("field") or field.get("items")
                value, offset = self.parse_object(binary, nested_fields, offset)
            else:
                value = None
        elif ftype == "string":
            value, offset = self.parse_string(binary, offset, field.get("nullable", False))
        elif ftype in self.__type_map:
            value, offset = self.parse_scalar(binary, ftype, offset, field.get("nullable", False))
        else:
            raise ValueError(f"Tipe field tidak didukung: {ftype}")

        payload[name] = value

    return {
        "header": header,
        "payload": payload
    }

  def parseHeader(self, binary: bytes) -> dict:
    offset = 0

    # Parse header
    magic = binary[offset:offset+7]
    offset += 7
    version = struct.unpack_from("<B", binary, offset)[0]
    offset += 1

    header = {
        "magic": magic,
        "version": version
    }
    print(f"Header: {header}")
    return {
        "header": header,
    }


  def parseCaseData(self, binary: bytes) -> dict:
    offset = 0

    # Parse header
    magic = binary[offset:offset+7]
    offset += 7
    version = struct.unpack_from("<B", binary, offset)[0]
    offset += 1

    header = {
        "magic": magic,
        "version": version
    }
    print(f"Header: {header}")
    if magic != self.__hfl.encode():
        raise ValueError("file header bytes tidak cocok")
    elif version > self.__vfl or version <= 0:
        raise ValueError("file version tidak cocok")

    # Load layout by version
    if version == 1:
      shemaname = 'project-schema.json'
    else:
      shemaname = f'project-schema-v{version}.json'  
    schema = self.load_schema_with_refs(f"psc-schema/{shemaname}")
    layout = self.extract_layout(schema)

    # Parse payload
    payload = {}
    for field in layout:
      if field.get("name") == "tbCases":
        ftype = field["type"]
        name = field["name"]

        if ftype == "array":
            value, offset = self.parse_array(binary, field, offset)
        elif ftype == "object":
            present = struct.unpack_from("<B", binary, offset)[0]
            offset += 1
            if present:
                nested_fields = field.get("field") or field.get("items")
                value, offset = self.parse_object(binary, nested_fields, offset)
            else:
                value = None
        elif ftype == "string":
            value, offset = self.parse_string(binary, offset, field.get("nullable", False))
        elif ftype in self.__type_map:
            value, offset = self.parse_scalar(binary, ftype, offset, field.get("nullable", False))
        else:
            raise ValueError(f"Tipe field tidak didukung: {ftype}")

        payload[name] = value
        break

    return {
        "header": header,
        "payload": payload
    }


  def unpackDataFile(self, binary = bytes):
    try:
      #try use version 2
      data = self.parseData(binary, usever=2)
    except Exception as e:
      print("Gagal parse versi 2, coba versi 1", flush=True)
      #try use version 1
      data = self.parseData(binary, usever=1)  
    # data = self.parseData(binary)
    # convert data to json
    json_data = json.loads(json.dumps(data, indent=2, cls=BytesEncoder))
    return json_data

  def packDataFile(self, jdata: any) -> bytes:
    # convert data from fastapi request json data to dict
    data = json.loads(jdata)
    binary = self.packData(data)

    return binary

  def getCase(self, binary = bytes):
    data = self.parseCaseData(binary)

    # convert data to json
    json_data = json.loads(json.dumps(data, indent=2, cls=BytesEncoder))
    return json_data

  def getCaseData(self, split_ids, file_bytes = bytes):
    try:
      #try use version 2
      data = self.parseData(file_bytes, usever=2)
    except Exception as e:
      print("Gagal parse versi 2, coba versi 1", flush=True)
      #try use version 1
      data = self.parseData(file_bytes, usever=1)
    schemaData = data['payload']
    result = {}
    for data in schemaData:
      if data == 'tbCases':
        # filter tbCases by split_ids
        cases = [case for case in schemaData['tbCases'] if case.get('id') in split_ids]
        result.update({"tbCases": cases})
      elif data in ['tbGenCfg', 'tbCostRec', 'tbGs', 'tbFiscal', 'tbCapCost', 'tbIntangCost', 'tbOpexCost', 'tbAsrCost', 'tbCosCost', 'tbLbtCost']:
        # filter cId by split_ids
        cdata = [cfg for cfg in schemaData[data] if cfg.get('cId') in split_ids]
        result.update({data: cdata})
      elif data == 'tbLiftingH':
        cdata = [cfg for cfg in schemaData[data] if cfg.get('cId') in split_ids]
        result.update({data: cdata})
      elif data == 'tbLiftingD':
        # filter by parent pId in tbLiftingH
        liftingH_ids = [lh.get('id') for lh in schemaData.get('tbLiftingH', []) if lh.get('cId') in split_ids]
        cdata = [cfg for cfg in schemaData[data] if cfg.get('pId') in liftingH_ids]
        result.update({data: cdata})
      elif data == 'tbSens':
        # filter cId by split_ids
        cdata = [cfg for cfg in schemaData[data] if cfg.get('cId') in split_ids]
        result.update({data: cdata})
      elif data == 'tbOptH':
        # filter cId by split_ids
        cdata = [cfg for cfg in schemaData[data] if cfg.get('cId') in split_ids]
        result.update({data: cdata})
      elif data == 'tbOptD':
        # filter cId by split_ids
        cdata = [cfg for cfg in schemaData[data] if cfg.get('cId') in split_ids]
        result.update({data: cdata})
      elif data == 'tbMonteH':
        # filter cId by split_ids
        cdata = [cfg for cfg in schemaData[data] if cfg.get('cId') in split_ids]
        result.update({data: cdata})
      elif data == 'tbMonteD':
        # filter cId by split_ids
        cdata = [cfg for cfg in schemaData[data] if cfg.get('cId') in split_ids]
        result.update({data: cdata})
      elif data == 'tbMonteSummD':
        # filter cId by split_ids
        cdata = [cfg for cfg in schemaData[data] if cfg.get('cId') in split_ids]
        result.update({data: cdata})
      elif data == 'tbMonteOutD':
        # filter by parent tId in tbMonteSummD
        monteSummD_ids = [ms.get('tId') for ms in schemaData.get('tbMonteSummD', []) if ms.get('cId') in split_ids]
        cdata = [cfg for cfg in schemaData[data] if cfg.get('tId') in monteSummD_ids]
        result.update({data: cdata})
    return result

  @staticmethod
  async def getHeaderCase(file: UploadFile = File(...)):
    try:
      binary = await file.read()
      data = pscExtractor(__hfl, __vfl, binary).parse_header(binary)
      return data
    except Exception as e:
      print(e, flush=True)
      return { "magic": None, "version": None }       

  # @staticmethod
  # async def getCasefromPSCdb(ciphertext_bytes: bytes):
  #   try:
  #     hex_str = ciphertext_bytes.decode('utf-8').strip()
  #     print("HEX preview:", hex_str[:100], flush=True)
  #     ciphertext_bytes = bytes.fromhex(hex_str)

  #     # Setup
  #     key = PBKDF2('key', b'salt', dkLen=32, count=100, hmac_hash_module=SHA1)

  #     # Decrypt
  #     iv = 'a2xhcgAAAAAAAAAA'.encode('utf-8')  # sama persis dengan Utf8.parse
      
  #     # harus: 61327868636741414141414141414141
  #     print("IV hex harus = 61327868636741414141414141414141:", iv.hex())

  #     cipher = AES.new(key, AES.MODE_CBC, iv=iv)
  #     raw = cipher.decrypt(ciphertext_bytes)
  #     payload = raw.decode('utf-8', errors='replace')
  #     print("Payload preview:", payload[:100])

  #     print(f"raw before decode: {raw[:100]}...", flush=True)

  #     pad_len = raw[-1]
  #     if raw[-pad_len:] == bytes([pad_len]) * pad_len:
  #         unpadded = raw[:-pad_len]
  #         payload = unpadded.decode('utf-8')
  #     else:
  #         print("⚠️ Padding invalid — skipping unpad")
  #         payload = raw.decode('utf-8', errors='replace')

  #     # unpadded = unpad(raw, AES.block_size)  # optional, tergantung JS

  #     # payload = unpadded.decode('utf-8', errors='replace')  # untuk audit
  #     # payload = raw.decode('utf-8')  # hasilnya: data:text/json;base64,...

  #     print(f"Payload (first 100 bytes): {payload[:100]}...", flush=True)

  #     # Extract base64
  #     base64_part = payload.split('base64,')[1]
  #     json_txt = base64.b64decode(base64_part).decode('utf-8')

  #     # decrypted = unpad(raw, AES.block_size)
  #     # print(f"Decrypted data (first 100 bytes): {decrypted.decode('utf-8')[:100]}...", flush=True)
  #     return {
  #       "header": {
  #         "magic": 'PSCdb',
  #         "version": 1
  #       },
  #       "payload": decrypted.decode("utf-8")
  #     }
  #   except Exception as e:
  #     print(traceback.format_exc(), flush=True)
  #     print(e, flush=True)
  #     return { "header": None, "payload": None }
