# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).
# Required Notice: Copyright Arif Gökdaş (https://github.com/Krateian/Sutura)
"""STL intake and output.

Derived from Sutura 0.9.8 ``sutura/sutura_engine/pipeline/mesh_io.py``
(``read_stl``, ``_read_stl_ascii``, ``scan_bad_coordinates``, ``check_units``)
with stricter rejection rules and full float64 / per-facet provenance:

* binary and ASCII STL; binary is recognised by the exact ``84 + 50 n`` size
* truncated files, count/size mismatches, NaN/Inf coordinates, oversize files
  and absurd coordinate magnitudes are rejected with :class:`StlError`
* coordinates are never rescaled or rounded: binary keeps its float32 values
  (exactly representable in float64), ASCII keeps the decimal value parsed to
  float64; ``-0.0`` is folded to ``+0.0`` (same coordinate value)
* indexing is normalised by merging *bit-identical* coordinates only
* each facet keeps its input index (``src``), its 2-byte attribute word and,
  for ASCII, the index of the ``solid`` block it came from
* units are detected as metadata only (header keywords + size heuristic)
"""
from __future__ import annotations

import hashlib
import os
import re
import struct
from dataclasses import dataclass, field

import numpy as np

MAX_BYTES_DEFAULT = 2 * 1024 ** 3          # 2 GiB
MAX_TRIANGLES_DEFAULT = 20_000_000
MAX_ABS_COORD_DEFAULT = 1.0e12

_REC = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])


class StlError(ValueError):
    """Malformed or unsafe STL input."""


@dataclass
class Soup:
    """Raw facets exactly as read: ``tri[k]`` is facet k's three corners."""
    tri: np.ndarray                 # (m,3,3) float64
    attr: np.ndarray                # (m,) uint16 attribute word (0 for ASCII)
    solid: np.ndarray               # (m,) int32 solid-block index (0 for binary)
    solid_names: list
    fmt: str                        # 'binary' | 'ascii'
    header: bytes                   # 80-byte binary header or b'solid <name>'
    sha256: str
    nbytes: int
    warnings: list = field(default_factory=list)


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _looks_ascii(data: bytes) -> bool:
    head = data[:512].lstrip()
    if not head[:5].lower() == b"solid":
        return False
    try:
        data[:4096].decode("ascii")
    except UnicodeDecodeError:
        return False
    return b"facet" in data[:65536].lower() or b"endsolid" in data[:65536].lower()


def read_stl(path, *, max_bytes=MAX_BYTES_DEFAULT, max_triangles=MAX_TRIANGLES_DEFAULT,
             max_abs_coord=MAX_ABS_COORD_DEFAULT) -> Soup:
    size = os.path.getsize(path)
    if size > max_bytes:
        raise StlError("file is %d bytes, above the %d-byte safety limit" % (size, max_bytes))
    with open(path, "rb") as f:
        data = f.read()
    sha = hashlib.sha256(data).hexdigest()
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if len(data) == 84 + 50 * n:
            soup = _read_binary(data, n, max_triangles)
        elif _looks_ascii(data):
            soup = _read_ascii(data, max_triangles)
        elif len(data) < 84 + 50 * n:
            raise StlError("truncated binary STL: header declares %d facets (%d bytes), file has %d bytes"
                           % (n, 84 + 50 * n, len(data)))
        else:
            raise StlError("invalid binary STL: header declares %d facets but file has %d trailing bytes"
                           % (n, len(data) - 84 - 50 * n))
    elif _looks_ascii(data):
        soup = _read_ascii(data, max_triangles)
    else:
        raise StlError("file too short for a binary STL header and not ASCII STL")
    soup.sha256 = sha
    soup.nbytes = len(data)
    t = soup.tri
    if not np.isfinite(t).all():
        bad = int((~np.isfinite(t)).any(axis=(1, 2)).sum())
        raise StlError("%d facets have NaN or infinite coordinates" % bad)
    if t.size and float(np.abs(t).max()) > max_abs_coord:
        raise StlError("coordinate magnitude %.3g exceeds the safety limit %.3g"
                       % (float(np.abs(t).max()), max_abs_coord))
    soup.tri = t + 0.0  # folds -0.0 to +0.0, keeps every value
    return soup


def _read_binary(data: bytes, n: int, max_triangles: int) -> Soup:
    if n > max_triangles:
        raise StlError("%d facets exceed the %d-facet safety limit" % (n, max_triangles))
    rec = np.frombuffer(data, dtype=_REC, count=n, offset=84)
    warn = []
    if n and not np.isfinite(rec["n"]).all():
        warn.append("non-finite stored normals ignored")
    return Soup(tri=rec["v"].astype(np.float64), attr=rec["attr"].astype(np.uint16),
                solid=np.zeros(n, np.int32), solid_names=[], fmt="binary",
                header=bytes(data[:80]), sha256="", nbytes=0, warnings=warn)


def _read_ascii(data: bytes, max_triangles: int) -> Soup:
    """Strict line grammar: solid / facet normal / outer loop / 3 x vertex /
    endloop / endfacet / endsolid (keywords case-insensitive)."""
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise StlError("ASCII STL contains non-ASCII bytes") from exc
    verts, solid_ids, names = [], [], []
    cur, state, tri = -1, "top", []
    for ln, line in enumerate(text.splitlines(), 1):
        p = line.split()
        if not p:
            continue
        w = p[0].lower()
        if state == "top":
            if w == "solid":
                names.append(line.strip()[5:].strip())
                cur += 1
                state = "solid"
                continue
            raise StlError("malformed ASCII STL line %d: expected 'solid'" % ln)
        if state == "solid":
            if w == "facet" and len(p) == 5 and p[1].lower() == "normal":
                state, tri = "facet", []
                continue
            if w == "endsolid":
                state = "top"
                continue
            raise StlError("malformed ASCII STL line %d: expected 'facet normal' or 'endsolid'" % ln)
        if state == "facet":
            if w == "outer" and len(p) == 2 and p[1].lower() == "loop":
                state = "loop"
                continue
            raise StlError("malformed ASCII STL line %d: expected 'outer loop'" % ln)
        if state == "loop":
            if w == "vertex":
                if len(p) != 4 or len(tri) == 3:
                    raise StlError("malformed ASCII STL line %d: bad vertex record" % ln)
                try:
                    tri.append((float(p[1]), float(p[2]), float(p[3])))
                except ValueError as exc:
                    raise StlError("malformed ASCII STL line %d: non-numeric vertex" % ln) from exc
                continue
            if w == "endloop":
                if len(tri) != 3:
                    raise StlError("malformed ASCII STL line %d: facet has %d vertices, not 3" % (ln, len(tri)))
                state = "endloop"
                continue
            raise StlError("malformed ASCII STL line %d: expected 'vertex' or 'endloop'" % ln)
        if state == "endloop":
            if w == "endfacet":
                verts.append(tri)
                solid_ids.append(cur)
                if len(verts) > max_triangles:
                    raise StlError("facet count exceeds the %d-facet safety limit" % max_triangles)
                state = "solid"
                continue
            raise StlError("malformed ASCII STL line %d: expected 'endfacet'" % ln)
    if state != "top":
        raise StlError("truncated ASCII STL: file ends inside a %s block" % state)
    if cur < 0:
        raise StlError("ASCII STL has no solid block")
    tri = np.asarray(verts, dtype=np.float64).reshape(-1, 3, 3)
    header = ("solid " + names[0]).encode() if names else b""
    return Soup(tri=tri, attr=np.zeros(len(tri), np.uint16),
                solid=np.asarray(solid_ids, np.int32).reshape(-1), solid_names=names, fmt="ascii",
                header=header, sha256="", nbytes=0)


# --------------------------------------------------------------------- units
_UNIT_RE = re.compile(rb"(?i)\b(?:units?\s*[=:]\s*|in\s+)(mm|millimet(?:er|re)s?|cm|centimet(?:er|re)s?|m|met(?:er|re)s?|in|inch(?:es)?|ft|feet|um|micron)s?\b")
_UNIT_NAME = {"mm": "mm", "millimeter": "mm", "millimetre": "mm", "cm": "cm", "centimeter": "cm",
              "centimetre": "cm", "m": "m", "meter": "m", "metre": "m", "in": "inch", "inch": "inch",
              "ft": "ft", "feet": "ft", "um": "um", "micron": "um"}


def detect_units(header: bytes, tri: np.ndarray) -> dict:
    """Unit metadata only; nothing is ever rescaled."""
    out = {"declared": None, "declared_source": None, "heuristic": None, "rescaled": False}
    m = _UNIT_RE.search(header or b"")
    if m:
        key = m.group(1).decode().lower().rstrip("s")
        out["declared"] = _UNIT_NAME.get(key, key)
        out["declared_source"] = "header"
    if tri.size:
        ext = tri.reshape(-1, 3).max(0) - tri.reshape(-1, 3).min(0)
        L = float(ext.max())
        out["longest_axis"] = L
        # Sutura check_units heuristic (25-400 mm printable band) as a hint only
        if 25.0 <= 25.4 * L <= 400.0 and not 25.0 <= L <= 400.0:
            out["heuristic"] = "longest axis %.4g: plausible as inches (%.1f mm)" % (L, 25.4 * L)
        elif 25.0 <= 10.0 * L <= 400.0 and not 25.0 <= L <= 400.0:
            out["heuristic"] = "longest axis %.4g: plausible as centimetres (%.1f mm)" % (L, 10 * L)
        elif 25.0 <= L <= 400.0:
            out["heuristic"] = "consistent with millimetres"
        elif 0.025 <= L <= 0.4:
            out["heuristic"] = "longest axis %.4g: plausible as metres" % L
    return out


# --------------------------------------------------------------------- write
def write_stl(path, V, F, *, fmt="binary", attr=None, header=b"", solid_of_face=None, solid_names=None):
    """Write triangles; binary keeps per-facet attribute words, ASCII writes
    one solid block per input solid with round-trip-exact ``%.17g`` values."""
    V = np.asarray(V, np.float64)
    F = np.asarray(F, np.int64)
    P = V[F] if len(F) else np.zeros((0, 3, 3))
    if fmt == "binary":
        if len(P) and not np.array_equal(P.astype(np.float32).astype(np.float64), P):
            raise StlError("coordinates are not exactly representable in float32; write ASCII instead")
        rec = np.zeros(len(F), dtype=_REC)
        rec["v"] = P.astype(np.float32)
        nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]) if len(P) else np.zeros((0, 3))
        ln = np.linalg.norm(nrm, axis=1, keepdims=True) if len(P) else np.zeros((0, 1))
        rec["n"] = np.divide(nrm, ln, out=np.zeros_like(nrm), where=ln > 0).astype(np.float32)
        rec["attr"] = np.zeros(len(F), np.uint16) if attr is None else np.asarray(attr, np.uint16)
        hdr = (header or b"stl_repair_core").ljust(80, b" ")[:80]
        with open(path, "wb") as f:
            f.write(hdr)
            f.write(struct.pack("<I", len(F)))
            f.write(rec.tobytes())
        return
    sid = np.zeros(len(F), np.int64) if solid_of_face is None else np.asarray(solid_of_face, np.int64)
    names = list(solid_names or [])
    groups = sorted(set(sid.tolist())) or [0]
    lines = []
    for g in groups:
        name = names[g] if 0 <= g < len(names) else ("stl_repair_core" if g < 0 else "solid%d" % g)
        lines.append("solid %s" % name)
        for k in np.flatnonzero(sid == g):
            p = P[k]
            n = np.cross(p[1] - p[0], p[2] - p[0])
            ln = np.linalg.norm(n)
            n = n / ln if ln > 0 else n
            lines.append("  facet normal %.9g %.9g %.9g" % tuple(n))
            lines.append("    outer loop")
            for c in p:
                lines.append("      vertex %r %r %r" % (float(c[0]), float(c[1]), float(c[2])))
            lines.append("    endloop")
            lines.append("  endfacet")
        lines.append("endsolid %s" % name)
    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(lines) + "\n")
