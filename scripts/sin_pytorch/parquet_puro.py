"""Lector mínimo de Parquet en Python puro (sin pyarrow): thrift compact, snappy, PLAIN/RLE_DICTIONARY."""
from __future__ import annotations
import struct, zlib
import numpy as np

# ---------------- thrift compact protocol ----------------
class TReader:
    def __init__(self, buf, pos=0):
        self.b = buf; self.p = pos
    def byte(self):
        v = self.b[self.p]; self.p += 1; return v
    def varint(self):
        r = 0; s = 0
        while True:
            x = self.byte(); r |= (x & 0x7F) << s; s += 7
            if not x & 0x80: return r
    def zz(self):
        v = self.varint(); return (v >> 1) ^ -(v & 1)
    def binary(self):
        n = self.varint(); v = self.b[self.p:self.p+n]; self.p += n; return bytes(v)
    def value(self, t):
        if t == 1: return True
        if t == 2: return False
        if t == 3: v = struct.unpack_from('<b', self.b, self.p)[0]; self.p += 1; return v
        if t in (4, 5, 6): return self.zz()
        if t == 7: v = struct.unpack_from('<d', self.b, self.p)[0]; self.p += 8; return v
        if t == 8: return self.binary()
        if t in (9, 10):
            h = self.byte(); n = h >> 4; et = h & 0x0F
            if n == 15: n = self.varint()
            if et in (1, 2):
                return [self.byte() == 1 for _ in range(n)]
            return [self.value(et) for _ in range(n)]
        if t == 11:
            n = self.varint()
            if n == 0: return {}
            kv = self.byte(); kt, vt = kv >> 4, kv & 0x0F
            return {self.value(kt): self.value(vt) for _ in range(n)}
        if t == 12: return self.struct()
        raise ValueError(f"tipo thrift {t}")
    def struct(self):
        out = {}; last = 0
        while True:
            h = self.byte()
            if h == 0: return out
            d = h >> 4; t = h & 0x0F
            fid = last + d if d else self.zz()
            last = fid
            out[fid] = self.value(t)

# ---------------- snappy ----------------
def snappy_decompress(src: bytes) -> bytes:
    p = 0; n = 0; s = 0
    while True:
        x = src[p]; p += 1; n |= (x & 0x7F) << s; s += 7
        if not x & 0x80: break
    out = bytearray()
    L = len(src)
    while p < L:
        tag = src[p]; p += 1; t = tag & 3
        if t == 0:
            ln = tag >> 2
            if ln >= 60:
                nb = ln - 59; ln = int.from_bytes(src[p:p+nb], 'little'); p += nb
            ln += 1
            out += src[p:p+ln]; p += ln
            continue
        if t == 1:
            ln = ((tag >> 2) & 7) + 4; off = ((tag >> 5) << 8) | src[p]; p += 1
        elif t == 2:
            ln = (tag >> 2) + 1; off = int.from_bytes(src[p:p+2], 'little'); p += 2
        else:
            ln = (tag >> 2) + 1; off = int.from_bytes(src[p:p+4], 'little'); p += 4
        start = len(out) - off
        if off >= ln:
            out += out[start:start+ln]
        else:
            for i in range(ln): out.append(out[start + i])
    assert len(out) == n, (len(out), n)
    return bytes(out)

def descomprimir(codec, data, n):
    if codec == 0: return data
    if codec == 1: return snappy_decompress(data)
    if codec == 2: return zlib.decompress(data, 31)
    raise NotImplementedError(f"codec {codec}")

# ---------------- RLE / bit-packed hybrid ----------------
def rle_hybrid(buf, bit_width, n):
    out = np.empty(n, dtype=np.int64); i = 0; r = TReader(buf)
    bw_bytes = (bit_width + 7) // 8
    while i < n and r.p < len(buf):
        h = r.varint()
        if h & 1:  # bit-packed: (h>>1) grupos de 8
            ng = h >> 1; nbytes = ng * bit_width
            chunk = np.frombuffer(buf[r.p:r.p+nbytes], dtype=np.uint8); r.p += nbytes
            if bit_width == 0:
                vals = np.zeros(ng * 8, dtype=np.int64)
            else:
                bits = np.unpackbits(chunk, bitorder='little').reshape(-1, bit_width)
                vals = (bits.astype(np.int64) << np.arange(bit_width)).sum(axis=1)
            k = min(len(vals), n - i); out[i:i+k] = vals[:k]; i += k
        else:
            cnt = h >> 1
            v = int.from_bytes(buf[r.p:r.p+bw_bytes], 'little'); r.p += bw_bytes
            k = min(cnt, n - i); out[i:i+k] = v; i += k
    return out

# ---------------- lectura ----------------
TIPOS = {0: 'bool', 1: 'int32', 2: 'int64', 3: 'int96', 4: 'float', 5: 'double', 6: 'bytes', 7: 'fixed'}

def plain(tipo, buf, n):
    if tipo == 1: return np.frombuffer(buf, '<i4', n)
    if tipo == 2: return np.frombuffer(buf, '<i8', n)
    if tipo == 4: return np.frombuffer(buf, '<f4', n)
    if tipo == 5: return np.frombuffer(buf, '<f8', n)
    if tipo == 0: return np.unpackbits(np.frombuffer(buf, np.uint8), bitorder='little')[:n].astype(bool)
    if tipo == 6:
        out = []; p = 0
        for _ in range(n):
            ln = struct.unpack_from('<i', buf, p)[0]; p += 4
            out.append(buf[p:p+ln].decode('utf-8', 'replace')); p += ln
        return np.array(out, dtype=object)
    raise NotImplementedError(TIPOS.get(tipo))

def metadata(path):
    data = open(path, 'rb').read()
    assert data[:4] == b'PAR1' and data[-4:] == b'PAR1'
    mlen = struct.unpack('<i', data[-8:-4])[0]
    meta = TReader(data, len(data) - 8 - mlen).struct()
    return data, meta

def leer_columna(data, cmeta, tipo, max_def):
    codec = cmeta[4]; total = cmeta[5]
    pos = cmeta.get(11, cmeta[9]) if cmeta.get(11) else cmeta[9]
    fin = pos + cmeta[7]
    dic = None; partes = []
    while pos < fin:
        r = TReader(data, pos); ph = r.struct(); pos = r.p
        csize = ph[3]; usize = ph[2]; raw = data[pos:pos+csize]; pos += csize
        ptype = ph[1]
        if ptype == 2:
            dh = ph[7]; buf = descomprimir(codec, raw, usize)
            dic = plain(tipo, buf, dh[1]); continue
        if ptype == 0:
            dh = ph[5]; nv = dh[1]; enc = dh[2]
            buf = descomprimir(codec, raw, usize); p = 0
            if max_def > 0:
                ln = struct.unpack_from('<i', buf, 0)[0]
                defs = rle_hybrid(buf[4:4+ln], max_def.bit_length(), nv); p = 4 + ln
            else:
                defs = np.ones(nv, dtype=np.int64)
            valores_buf = buf[p:]
        elif ptype == 3:
            dh = ph[8]; nv = dh[1]; enc = dh[4]; dl = dh[5]; rl = dh[6]
            levels = raw[:rl+dl]; resto = raw[rl+dl:]
            comprimido = dh.get(7, True)
            valores_buf = descomprimir(codec, resto, usize - rl - dl) if comprimido else resto
            defs = rle_hybrid(levels[rl:rl+dl], max_def.bit_length(), nv) if max_def > 0 else np.ones(nv, dtype=np.int64)
        else:
            continue
        presentes = int((defs == max_def).sum())
        if enc in (2, 8):  # PLAIN_DICTIONARY / RLE_DICTIONARY
            bw = valores_buf[0]
            idx = rle_hybrid(valores_buf[1:], bw, presentes)
            vals = dic[idx]
        elif enc == 0:
            vals = plain(tipo, valores_buf, presentes)
        else:
            raise NotImplementedError(f"encoding {enc}")
        if presentes == nv:
            partes.append(np.asarray(vals))
        else:
            if tipo in (6,):
                full = np.full(nv, None, dtype=object)
            else:
                full = np.full(nv, np.nan, dtype=np.float64)
            full[defs == max_def] = vals
            partes.append(full)
    return np.concatenate(partes) if partes else np.array([])

def leer_parquet(path):
    import pandas as pd
    data, meta = metadata(path)
    schema = meta[2]
    hojas = [s for s in schema[1:] if not s.get(5)]
    cols = {}
    for rg in meta[4]:
        for ch, el in zip(rg[1], hojas):
            cm = ch[3]; nombre = el[4].decode()
            max_def = 1 if el.get(3, 0) == 1 else 0
            v = leer_columna(data, cm, el[1], max_def)
            cols.setdefault(nombre, []).append(v)
    return pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})

if __name__ == '__main__':
    import sys
    data, meta = metadata(sys.argv[1])
    print('filas', meta[3], 'row groups', len(meta[4]), 'creado por', meta.get(6))
    hojas = [s for s in meta[2][1:] if not s.get(5)]
    print('columnas', len(hojas), 'tipos', {TIPOS.get(h.get(1)) for h in hojas})
    cm = meta[4][0][1][0][3]
    print('codec', cm[4], 'encodings', cm[2])
