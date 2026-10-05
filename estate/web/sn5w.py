"""SN5W version 1: the walk grid of one building, one layer per storey (stdlib only; estate/leaks.py reads it).

Little-endian throughout. Written raw by the web stage (model/<ID>/<ID>_walk.bin); a viewer may gzip it.

    offset  type      field
    0       char[4]   "SN5W"
    4       u16       version = 1
    6       u16       flags: bit 0 = some layer is stored as a delta, bit 1 = coarse fallback (cell 0.2 m, a cell is
                      walkable only where all four 0.1 m cells under it are); other bits 0
    8       f32       cell (m), 0.10 (0.20 with bit 1)
    12      f32       radius (m) of the agent the grid was made for, 0.20
    16      f32       step (m), 0.40: the largest floor-height difference between neighbouring walkable cells
    20      f32       origin_x (m, block-local): the corner of cell (0, 0), its low-x, low-y side
    24      f32       origin_y
    28      u16       nx
    30      u16       ny
    32      u16       n_layers (storeys, bottom up)
    34      u16       ref_layer: the layer delta layers are stored against; always mode raw
    36..63            reserved, 0
    64      n_layers x 32-byte entries:
              +0  char[4]  tag: the storey name, ASCII, NUL-padded, unpadded number ("L1\\0\\0", "L12\\0", "RF\\0\\0")
              +4  f32      ffl: the storey's finished floor level (m, block-local Z)
              +8  u8       mode: 0 raw, 1 delta, 2 same (the reference's raster)
              +9  u8       0
              +10 u16      0
              +12 u32      raster_off   (0 for mode 2)
              +16 u32      raster_len   (2 * nx * ny; 0 for mode 2)
              +20 u32      overflow_count
              +24 u32      overflow_off (0 when the count is 0)
              +28 u32      0
    then each layer's raster and overflow records, in layer order, with no gaps: the sections tile the file.

raster: int16[nx * ny], index iy * nx + ix. The floor height of the main walkable floor of cell (ix, iy) in mm
above the layer's ffl, or 0x7FFF where the cell has none. Mode 1 stores (value - reference value) mod 2^16 as
int16 (wrapping), mode 2 nothing. A layer owns the floors from ffl - 0.25 m up to the next layer's ffl - 0.25 m
(the top layer: everything above); the main floor of a cell is the one closest to ffl, and every other floor of
that cell in the band is an overflow record.
overflow: count x {u16 ix, u16 iy, i16 mm, u16 0}, sorted by (iy, ix, mm).

Nothing in a file is text but the layer tags, which is all the leak scan reads (``layer_tags``). Anything that does
not parse exactly as above (a stray byte, a non-zero pad, a section out of place) raises ValueError, and the scan
then reports the file as not scanned.
"""
from __future__ import annotations

import struct
import sys
from array import array

MAGIC = b"SN5W"
VERSION = 1
HEADER = 64
ENTRY = 32
BLOCKED = 0x7FFF
RAW, DELTA, SAME = 0, 1, 2
MODE_NAMES = {RAW: "raw", DELTA: "delta", SAME: "same"}
MODE_IDS = {v: k for k, v in MODE_NAMES.items()}
FLAG_DELTA, FLAG_COARSE = 1, 2
_HEAD = struct.Struct("<4sHHfffffHHHH28s")
_ENTRY = struct.Struct("<4sfBBHIIIII")
_OVERFLOW = struct.Struct("<HHhH")


def _int16(values) -> array:
    """values as array('h') (little-endian on disk): an array, bytes of int16 LE, a numpy array or any ints."""
    if isinstance(values, array) and values.typecode == "h":
        return array("h", values)
    if hasattr(values, "astype") and hasattr(values, "tobytes"):          # numpy, without importing it
        values = values.astype("<i2").tobytes()
    if isinstance(values, (bytes, bytearray, memoryview)):
        a = array("h")
        a.frombytes(bytes(values))
        if sys.byteorder == "big":
            a.byteswap()
        return a
    return array("h", (int(v) for v in values))


def _le(a: array) -> bytes:
    if sys.byteorder == "big":
        a = array("h", a)
        a.byteswap()
    return a.tobytes()


def _wrap(v: int) -> int:
    return (v + 32768) % 65536 - 32768


def tag_bytes(tag: str) -> bytes:
    b = tag.encode("utf-8")
    if not 1 <= len(b) <= 4 or not all(32 < c < 127 for c in b):
        raise ValueError(f"layer tag {tag!r}: 1-4 printable ASCII characters")
    return b.ljust(4, b"\0")


def choose_mode(raster: array, ref: array) -> int:
    """same when the raster equals the reference's; else delta when fewer cells differ from it than the layer has
    walkable cells (long runs of zeros), else raw (long runs of 0x7FFF)."""
    if raster == ref:
        return SAME
    differ = sum(1 for v, r in zip(raster, ref) if v != r)
    walkable = len(raster) - raster.count(BLOCKED)
    return DELTA if differ < walkable else RAW


def encode(nx: int, ny: int, cell: float, radius: float, step: float, origin, layers: list, ref: int,
           modes=None, coarse: bool = False) -> bytes:
    """The SN5W bytes of a grid. ``layers``: [{tag, ffl, raster (absolute int16 heights, BLOCKED where none, any
    int16 sequence of nx * ny), overflow [(ix, iy, mm)]}] bottom up; ``ref`` the reference layer; ``modes`` an
    explicit mode per layer (names or ids), else choose_mode; the reference is always raw."""
    if not (0 < nx < 65536 and 0 < ny < 65536 and 0 < len(layers) < 65536 and 0 <= ref < len(layers)):
        raise ValueError("grid or layer count out of range")
    rasters = [_int16(lay["raster"]) for lay in layers]
    if any(len(r) != nx * ny for r in rasters):
        raise ValueError("a raster is not nx * ny")
    base = rasters[ref]
    if modes is None:
        modes = [RAW if i == ref else choose_mode(r, base) for i, r in enumerate(rasters)]
    modes = [MODE_IDS[m] if isinstance(m, str) else int(m) for m in modes]
    if modes[ref] != RAW or len(modes) != len(layers):
        raise ValueError("the reference layer is stored raw, and every layer has a mode")
    for m, r in zip(modes, rasters):
        if m == SAME and r != base:
            raise ValueError("a 'same' layer differs from the reference")
    flags = (FLAG_DELTA if DELTA in modes else 0) | (FLAG_COARSE if coarse else 0)
    head = _HEAD.pack(MAGIC, VERSION, flags, cell, radius, step, float(origin[0]), float(origin[1]), nx, ny,
                      len(layers), ref, b"\0" * 28)
    off = HEADER + ENTRY * len(layers)
    table, body = [], []
    for lay, m, r in zip(layers, modes, rasters):
        recs = sorted((int(iy), int(ix), int(mm)) for ix, iy, mm in lay.get("overflow", ()))
        if any(not (0 <= ix < nx and 0 <= iy < ny and -32768 <= mm < BLOCKED) for iy, ix, mm in recs):
            raise ValueError(f"layer {lay['tag']}: an overflow record is out of range")
        if m == SAME:
            r_off, r_len, data = 0, 0, b""
        else:
            data = _le(r if m == RAW else array("h", (_wrap(v - b) for v, b in zip(r, base))))
            r_off, r_len = off, len(data)
        off += len(data)
        ov = b"".join(_OVERFLOW.pack(ix, iy, mm, 0) for iy, ix, mm in recs)
        o_off = off if recs else 0
        off += len(ov)
        table.append(_ENTRY.pack(tag_bytes(lay["tag"]), float(lay["ffl"]), m, 0, 0, r_off, r_len, len(recs), o_off, 0))
        body += [data, ov]
    return head + b"".join(table) + b"".join(body)


def _check_tag(raw: bytes) -> str:
    """A stored tag: 1-4 printable ASCII characters, then NULs only."""
    name = raw.rstrip(b"\0")
    if not name or b"\0" in name or not all(32 < c < 127 for c in name):
        raise ValueError(f"SN5W layer tag {raw!r}")
    return name.decode("ascii")


def read_table(data: bytes) -> dict:
    """Header and layer table of an SN5W file, every section checked; raises ValueError on anything else."""
    data = bytes(data)
    if len(data) < HEADER or data[:4] != MAGIC:
        raise ValueError("not an SN5W file")
    magic, ver, flags, cell, radius, step, ox, oy, nx, ny, n, ref, reserved = _HEAD.unpack_from(data, 0)
    if ver != VERSION or flags & ~(FLAG_DELTA | FLAG_COARSE) or reserved != b"\0" * 28:
        raise ValueError(f"SN5W version {ver}, flags {flags}: not version 1 or reserved bytes set")
    if not (nx and ny and n and ref < n) or len(data) < HEADER + ENTRY * n:
        raise ValueError("SN5W header out of range")
    layers, sections = [], []
    for i in range(n):
        tag, ffl, mode, p0, p1, r_off, r_len, o_cnt, o_off, p2 = _ENTRY.unpack_from(data, HEADER + ENTRY * i)
        if (p0, p1, p2) != (0, 0, 0) or mode not in MODE_NAMES:
            raise ValueError(f"SN5W layer {i}: bad mode or pad")
        if mode == SAME:
            if (r_off, r_len) != (0, 0):
                raise ValueError(f"SN5W layer {i}: a 'same' layer has a raster")
        else:
            if r_len != 2 * nx * ny:
                raise ValueError(f"SN5W layer {i}: raster length {r_len}")
            sections.append((r_off, r_len))
        if o_cnt:
            sections.append((o_off, 8 * o_cnt))
        elif o_off:
            raise ValueError(f"SN5W layer {i}: overflow offset without records")
        layers.append(dict(tag=_check_tag(tag), ffl=ffl, mode=MODE_NAMES[mode], raster_off=r_off, raster_len=r_len,
                           overflow_count=o_cnt, overflow_off=o_off))
    if layers[ref]["mode"] != "raw":
        raise ValueError("SN5W reference layer is not raw")
    pos = HEADER + ENTRY * n
    for off, ln in sections:                     # written in layer order, raster then overflow: no gaps, no overlap
        if off != pos:
            raise ValueError(f"SN5W section at {off}, expected {pos}")
        pos += ln
    if pos != len(data):
        raise ValueError(f"SN5W: {len(data) - pos} bytes after the last section")
    for lay in layers:
        for ix, iy, mm, pad in _OVERFLOW.iter_unpack(data[lay["overflow_off"]:lay["overflow_off"]
                                                          + 8 * lay["overflow_count"]]):
            if pad or ix >= nx or iy >= ny:
                raise ValueError(f"SN5W layer {lay['tag']}: bad overflow record")
    return dict(version=ver, flags=flags, delta=bool(flags & FLAG_DELTA), coarse=bool(flags & FLAG_COARSE),
                cell=cell, radius=radius, step=step, origin=(ox, oy), nx=nx, ny=ny, ref=ref, layers=layers)


def layer_tags(data: bytes) -> list[str]:
    """The text of an SN5W file: its layer tags (the file is checked whole first; ValueError if it does not parse)."""
    return [lay["tag"] for lay in read_table(data)["layers"]]


def decode(data: bytes) -> dict:
    """read_table plus every layer's absolute raster (array('h'), delta and same layers resolved against the
    reference) and its overflow records [(ix, iy, mm)]."""
    t = read_table(data)
    data = bytes(data)
    raw = {}
    for i, lay in enumerate(t["layers"]):
        if lay["mode"] != "same":
            raw[i] = _int16(data[lay["raster_off"]:lay["raster_off"] + lay["raster_len"]])
    base = raw[t["ref"]]
    for i, lay in enumerate(t["layers"]):
        if lay["mode"] == "raw":
            lay["raster"] = raw[i]
        elif lay["mode"] == "same":
            lay["raster"] = array("h", base)
        else:
            lay["raster"] = array("h", (_wrap(d + b) for d, b in zip(raw[i], base)))
        o = lay["overflow_off"]
        lay["overflow"] = [(ix, iy, mm) for ix, iy, mm, _ in
                           _OVERFLOW.iter_unpack(data[o:o + 8 * lay["overflow_count"]])]
    return t
