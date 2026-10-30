"""Case files (src/case_io.hpp) <-> the model tree.

`from_case` reads a case dict into a tree, `to_case` writes the tree back as
the same dict; `read_case` / `write_case` do the same with files, keeping the
comment block at the top of the file (the case's notes). Comments inside the
JSON body are not kept. Every key a case file may hold is modelled; any other
key is refused, never dropped, because bc_case would ignore it silently.
"""
from __future__ import annotations

import json
import pathlib

from . import nodes  # noqa: F401
from .model import REGION_KEYS, Model, ModelError, node_type

TOP = ("geometry", "faces", "cache", "material", "solver", "grade", "supports", "cases", "loads", "output")
SECTIONS = ("material", "solver", "grade", "output")


def strip_comments(text: str) -> tuple[str, str]:
    """(JSON text, notes): // comments removed outside strings, as json.hpp does;
    the comment lines before the opening brace become the notes."""
    out, notes, i, in_str, head = [], [], 0, False, True
    while i < len(text):
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\":
                out.append(text[i + 1])
                i += 1
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str, head = True, False
            out.append(c)
        elif text.startswith("//", i):
            j = text.find("\n", i)
            j = len(text) if j < 0 else j
            if head:
                notes.append(text[i + 2:j].removeprefix(" "))
            i = j
            continue
        else:
            if c == "{":
                head = False
            out.append(c)
        i += 1
    return "".join(out), "\n".join(notes)


def _split(kind: str, entry: dict, what: str) -> dict:
    """An entry's props and region, every key accounted for."""
    if not isinstance(entry, dict):
        raise ModelError(f"{what} is not an object: {entry!r}")
    known = {p.key for p in node_type(kind).props}
    unknown = set(entry) - known - set(REGION_KEYS)
    if unknown:
        raise ModelError(f"{what} has keys the model does not hold: {sorted(unknown)}")
    region = {k: entry[k] for k in REGION_KEYS if k in entry}
    return {"props": {k: entry[k] for k in entry if k in known}, "region": region or None}


def from_case(case: dict, notes: str = "") -> Model:
    """A case dict into a new tree. Checks everything before returning."""
    if not isinstance(case, dict):
        raise ModelError("a case file is a JSON object")
    unknown = set(case) - set(TOP)
    if unknown:
        raise ModelError(f"case file has keys the model does not hold: {sorted(unknown)}")
    m = Model()
    root = m.root.tag
    props = {k: case[k] for k in ("geometry", "faces", "cache") if k in case}
    if notes:
        props["notes"] = notes
    for k, v in props.items():
        m.apply({"op": "set", "tag": root, "key": k, "value": v})
    for sec in SECTIONS:
        if sec in case:
            m.apply({"op": "add", "parent": root, "kind": sec, **_split(sec, case[sec], sec)})
    for i, s in enumerate(case.get("supports", [])):
        m.apply({"op": "add", "parent": root, "kind": "support", **_split("support", s, f"support {i}")})
    cases = [(c, False) for c in case.get("cases", [])] + ([({"loads": case["loads"]}, True)] if "loads" in case else [])
    for i, (c, top) in enumerate(cases):
        if not isinstance(c, dict) or set(c) - {"name", "loads"}:
            raise ModelError(f"load case {i} is {{name, loads}}, got {c!r}")
        lc = m.apply({"op": "add", "parent": root, "kind": "loadcase",
                      "props": {**({"name": c["name"]} if "name" in c else {}), **({"top": True} if top else {})}})
        for j, ld in enumerate(c.get("loads", [])):
            m.apply({"op": "add", "parent": lc, "kind": "load", **_split("load", ld, f"load {j} of case {i}")})
    to_case(m)            # the same completeness checks as on write
    return m


def _entry(n, what: str) -> dict:
    t = n.type
    if t.region and n.region is None and not _regionless(n):
        raise ModelError(f"{what} ({n.tag}) has no region")
    out = {}
    if "label" in n.props:
        out["label"] = n.props["label"]
    out.update(n.region or {})
    for p in t.props:
        if p.key in n.props and p.key != "label":
            out[p.key] = n.props[p.key]
    return out


def _regionless(n) -> bool:
    return (n.kind == "support" and "symmetry" in n.props) or (n.kind == "load" and "rbe2" in n.props)


def to_case(model: Model) -> dict:
    """The tree as a case dict, refusing what bc_case would refuse."""
    r = model.root
    out = {k: r.props[k] for k in ("geometry", "faces", "cache") if k in r.props}
    if "geometry" not in out:
        raise ModelError("the case has no geometry")
    kids = r.children
    for sec in ("material", "solver", "grade"):
        for n in kids:
            if n.kind == sec:
                out[sec] = dict(n.props)
    sups = [n for n in kids if n.kind == "support"]
    for i, n in enumerate(sups):
        if not any(k in n.props for k in nodes.SUPPORT_KINDS):
            raise ModelError(f"support {i} ({n.tag}) has no fix, dofs, spring, rbe2 or symmetry")
    out["supports"] = [_entry(n, f"support {i}") for i, n in enumerate(sups)]
    if not out["supports"]:
        raise ModelError("the case has no supports")
    masters = {n.props.get("label") for n in sups if "rbe2" in n.props}
    cases, top = [], None
    for i, lc in enumerate(n for n in kids if n.kind == "loadcase"):
        loads = []
        for j, ld in enumerate(lc.children):
            p = ld.props
            if not any(k in p for k in nodes.LOAD_KINDS + ("force_N", "moment")):
                raise ModelError(f"load {j} of case {i} ({ld.tag}) has no force_N, moment, pressure, traction, "
                                 "bearing or rbe2")
            if "rbe2" in p and p["rbe2"] not in masters:
                raise ModelError(f"load {j} of case {i}: no rbe2 support labelled {p['rbe2']!r}")
            if "direction" in p and "force_N" not in p:
                raise ModelError(f"load {j} of case {i}: direction without force_N")
            loads.append(_entry(ld, f"load {j} of case {i}"))
        if lc.props.get("top"):
            if top is not None:
                raise ModelError("only one load case can be the top-level loads")
            top = loads
        else:
            cases.append({**({"name": lc.props["name"]} if "name" in lc.props else {}), "loads": loads})
    if cases:
        out["cases"] = cases
    if top is not None:
        out["loads"] = top
    if not cases and top is None:
        raise ModelError("the case has no load case")
    for n in kids:
        if n.kind == "output":
            out["output"] = dict(n.props)
    return out


def dumps(case: dict, notes: str = "") -> str:
    """A case dict as the hand-written files lay it out: one support, load case or
    section per line."""
    one = lambda v: json.dumps(v, ensure_ascii=False)   # noqa: E731
    lines = []
    for k, v in case.items():
        if k in ("supports", "cases", "loads"):
            lines.append(f'  "{k}": [\n' + ",\n".join(f"    {one(x)}" for x in v) + "\n  ]")
        else:
            lines.append(f'  "{k}": {one(v)}')
    head = "".join(f"// {ln}".rstrip() + "\n" for ln in notes.split("\n")) if notes else ""
    return head + "{\n" + ",\n".join(lines) + "\n}\n"


def read_case(path) -> Model:
    text, notes = strip_comments(pathlib.Path(path).read_text(encoding="utf-8"))
    return from_case(json.loads(text), notes)


def write_case(model: Model, path) -> pathlib.Path:
    p = pathlib.Path(path)
    p.write_text(dumps(to_case(model), model.root.props.get("notes", "")), encoding="utf-8")
    return p
