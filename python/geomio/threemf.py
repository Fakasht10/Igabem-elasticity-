"""3MF write and read (3MF core spec 1.3, one mesh object), with the CAD face id kept per triangle.

Layout follows materializr's src/io/ThreeMfExport.cpp (GPLv3, commit c5f9aa5): a stored zip holding
[Content_Types].xml, _rels/.rels and 3D/3dmodel.model, vertices de-duplicated, unit millimeter by default.
Face ids go into a core <basematerials> group, one material per face ("face_<id>", a distinct display colour) and
the triangle's pid/p1 pointing to it, so slicers show the faces in colour and read_3mf gives the ids back."""
import zipfile
import xml.etree.ElementTree as ET

import numpy as np

from .mesh import TriMesh

NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
_CT = ('<?xml version="1.0" encoding="UTF-8"?>\n'
       '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
       '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
       '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
_RELS = ('<?xml version="1.0" encoding="UTF-8"?>\n'
         '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
         '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
         'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')


def _colour(k):
    h = (k * 0.618033988749895) % 1.0          # golden-ratio hues: neighbours in id get distant colours
    i, f = int(h * 6), h * 6 - int(h * 6)
    v, s = 0.9, 0.6
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r, g, b = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i % 6]
    return "#%02X%02X%02XFF" % (int(255 * r), int(255 * g), int(255 * b))


def write_3mf(path, m, unit="millimeter", name="part"):
    faces = np.unique(m.fid)
    slot = np.searchsorted(faces, m.fid)
    out = ['<?xml version="1.0" encoding="UTF-8"?>\n<model unit="%s" xml:lang="en-US" xmlns="%s">'
           '<resources><basematerials id="1">' % (unit, NS)]
    out += ['<base name="face_%d" displaycolor="%s"/>' % (f, _colour(int(f)))
            for f in faces]
    out.append('</basematerials><object id="2" type="model" name="%s" pid="1" pindex="0"><mesh><vertices>' % name)
    out += ['<vertex x="%.9g" y="%.9g" z="%.9g"/>' % tuple(v) for v in m.V]
    out.append("</vertices><triangles>")
    out += ['<triangle v1="%d" v2="%d" v3="%d" pid="1" p1="%d"/>' % (a, b, c, s) for (a, b, c), s in zip(m.T, slot)]
    out.append('</triangles></mesh></object></resources><build><item objectid="2"/></build></model>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CT)
        z.writestr("_rels/.rels", _RELS)
        z.writestr("3D/3dmodel.model", "".join(out))


def read_3mf(path):
    """-> TriMesh of all mesh objects in the build (no transforms applied beyond identity); fid from the
    basematerials name face_<id> when present, otherwise the material index + 1, otherwise 0."""
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("3D/3dmodel.model"))
    q = "{%s}" % NS
    mats = {}
    for g in root.iter(q + "basematerials"):
        ids = []
        for k, b in enumerate(g.findall(q + "base")):
            nm = b.get("name", "")
            ids.append(int(nm[5:]) if nm.startswith("face_") and nm[5:].isdigit() else k + 1)
        mats[g.get("id")] = ids
    Vs, Ts, Fs, off = [], [], [], 0
    for obj in root.iter(q + "object"):
        mesh = obj.find(q + "mesh")
        if mesh is None:
            continue
        V = np.array([[float(v.get(c)) for c in "xyz"] for v in mesh.iter(q + "vertex")])
        tri = list(mesh.iter(q + "triangle"))
        T = np.array([[int(t.get(c)) for c in ("v1", "v2", "v3")] for t in tri], np.int64)
        dp, di = obj.get("pid"), obj.get("pindex", "0")
        F = []
        for t in tri:
            pid, p1 = t.get("pid", dp), t.get("p1", di)
            F.append(mats[pid][int(p1)] if pid in mats and p1 is not None else 0)
        Vs.append(V); Ts.append(T + off); Fs.append(F); off += len(V)
    return TriMesh(np.vstack(Vs), np.vstack(Ts), np.concatenate(Fs).astype(np.int64))
