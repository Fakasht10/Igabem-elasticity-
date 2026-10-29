"""STL read and write with the CAD face id kept per triangle.

Reading follows materializr (src/io/StlIO.cpp, GPLv3, commit c5f9aa5), which documents three ways STL readers fail
silently, all handled here:
  - format: a binary file may start with the bytes "solid", so the test is not the prefix.  Binary when the size is
    exactly 84 + 50 n for the header's n; otherwise ASCII when the first 512 bytes are printable text after "solid"
    (any case); otherwise binary;
  - truncation: a binary body shorter than the header promises, or an ASCII file whose vertex count is not 3 x its
    facet count, raises StlError (strict) instead of returning a partial mesh;
  - layout: ASCII is parsed by tokens, so blank lines between facets (assimp) and upper-case keywords are fine.
Face ids: binary STL keeps them in each facet's 2-byte attribute word when the header carries FACEID_MARK (written by
write_stl); other files' attribute words (often colours) are ignored unless faceid='attr'.  ASCII files keep one solid
per face ("solid face_<id>"); a multi-solid ASCII file from elsewhere gets one id per solid, starting at 1.
The mesh comes back welded (STL repeats every vertex per facet) and, with repair=True, oriented outward."""
import re

import numpy as np

from .mesh import TriMesh, repair as _repair, weld as _weld

FACEID_MARK = b"IGABEM geomio: attribute word = CAD face id"
_REC = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
_SOLID = re.compile(rb"^\s*solid", re.I)


class StlError(ValueError):
    pass


def _is_binary(data):
    if len(data) >= 84:
        n = int(np.frombuffer(data, "<u4", 1, 80)[0])
        if 84 + 50 * n == len(data):
            return True
    head = data[:512]
    if _SOLID.match(head) and all(32 <= c < 127 or c in (9, 10, 13) for c in head):
        return False
    return True


def _read_binary(data, strict):
    if len(data) < 84:
        raise StlError("binary STL shorter than its 84-byte header")
    n_head = int(np.frombuffer(data, "<u4", 1, 80)[0])
    n_body = (len(data) - 84) // 50
    if n_body < n_head and strict:
        raise StlError("truncated binary STL: header promises %d facets, body holds %d" % (n_head, n_body))
    rec = np.frombuffer(data, _REC, min(n_head, n_body), 84)
    return rec["v"].reshape(-1, 3).astype(float), rec["attr"].astype(np.int64), data[:80]


def _read_ascii(data, strict):
    tok = data.lower()
    nfacet = len(re.findall(rb"\bfacet\b", tok))          # \b: "endfacet" does not count
    num = rb"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
    V, sid = [], []
    solids = [m.start() for m in re.finditer(rb"\bsolid\b", tok)]   # not inside "endsolid"
    bounds = solids[1:] + [len(tok)]
    ids = []
    for s0, s1 in zip(solids or [0], bounds):
        line_end = tok.find(b"\n", s0)
        name = data[s0 + 5:line_end if line_end > 0 else s0 + 5].strip().decode("ascii", "replace")
        m = re.fullmatch(r"face_(\d+)", name)
        ids.append(int(m.group(1)) if m else None)
        chunk = tok[s0:s1]
        v = re.findall(rb"\bvertex\s+" + num + rb"\s+" + num + rb"\s+" + num, chunk)
        V += v
        sid += [len(ids) - 1] * (len(v) // 3)
        if len(v) % 3:
            raise StlError("ASCII STL solid %r has %d vertices, not a multiple of 3" % (name, len(v)))
    if strict and len(V) != 3 * nfacet:
        raise StlError("ASCII STL has %d facets but %d vertices (truncated or malformed)" % (nfacet, len(V)))
    V = np.array(V, float).reshape(-1, 3)
    if all(i is not None for i in ids):
        fid = np.array([ids[s] for s in sid], np.int64)
    else:
        fid = np.array(sid, np.int64) + (1 if len(ids) > 1 else 0)
    return V, fid


def read_stl(path, repair=True, weld_tol=None, faceid="auto", strict=True):
    """-> (TriMesh, info).  faceid: 'auto' (attribute words only from files we wrote), 'attr', or 'none'."""
    with open(path, "rb") as f:
        data = f.read()
    if _is_binary(data):
        V, attr, header = _read_binary(data, strict)
        use = faceid == "attr" or (faceid == "auto" and header.startswith(FACEID_MARK))
        fid = attr if use else np.zeros(len(attr), np.int64)
        fmt = "binary"
    else:
        V, fid = _read_ascii(data, strict)
        if faceid == "none":
            fid[:] = 0
        fmt = "ascii"
    nf = len(V) // 3
    if nf == 0:
        raise StlError("STL holds no facets")
    m = TriMesh(V, np.arange(3 * nf).reshape(-1, 3), fid)
    info = dict(format=fmt, facets=nf)
    if repair:
        m, r = _repair(m, weld_tol)
        info.update(r)
    else:
        m = _weld(m, weld_tol)
    info["triangles"] = len(m.T)
    return m, info


def write_stl(path, m, binary=True):
    """binary: face id in the attribute word (0..65535), header FACEID_MARK; ASCII: one solid per face id.
    Coordinates are float32 (binary) or %.9g (ASCII, float32-exact): STL itself cannot hold more."""
    N = m.normals()
    P = m.V[m.T]
    if binary:
        if len(m.fid) and (m.fid.min() < 0 or m.fid.max() > 0xFFFF):
            raise StlError("face ids must fit the 16-bit attribute word")
        rec = np.zeros(len(m.T), _REC)
        rec["n"], rec["v"], rec["attr"] = N, P, m.fid
        with open(path, "wb") as f:
            f.write(FACEID_MARK.ljust(80, b" "))
            f.write(np.uint32(len(m.T)).tobytes())
            f.write(rec.tobytes())
        return
    with open(path, "w") as f:
        for face in np.unique(m.fid):
            f.write("solid face_%d\n" % face)
            for t in np.flatnonzero(m.fid == face):
                f.write(" facet normal %.9g %.9g %.9g\n  outer loop\n" % tuple(N[t]))
                for p in P[t]:
                    f.write("   vertex %.9g %.9g %.9g\n" % tuple(p))
                f.write("  endloop\n endfacet\n")
            f.write("endsolid face_%d\n" % face)
