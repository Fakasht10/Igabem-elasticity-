"""The model tree: typed nodes with schema-checked props, per-node regions,
plain-dict operations, and snapshot undo. Standard library only.

A model is a tree of `Node`s. What a node may hold is declared once, in its
`NodeType`: its props (key, kind, unit, choices), whether it acts on a region,
which child kinds it allows, and a rule across its props. Everything else is
derived from that declaration: validation here, the settings form and the
agent's tool arguments later (`NodeType.schema()`).

Every change is an operation, a plain dict:

    {"op": "add",    "parent": tag, "kind": k, "props": {...}, "region": {...}}
    {"op": "set",    "tag": tag, "key": k, "value": v}
    {"op": "select", "tag": tag, "region": {...} | None}
    {"op": "remove", "tag": tag}

so a run can be journaled and replayed as data. Each operation is checked in
full before anything changes: a refused one leaves the model as it was. Undo
restores a snapshot of the whole tree (a few kB of JSON), so no operation
needs a hand-written inverse.
"""
from __future__ import annotations

import copy
import json
import math
import pathlib
from dataclasses import dataclass, field
from typing import Callable

FORMAT = "igabem-model"
VERSION = 1
DOFS = ("ux", "uy", "uz", "rx", "ry", "rz")
SELECTORS = {   # case_io.hpp Selector: key -> required fields (optional ones after "|")
    "box": None, "plane": ("point", "normal", "|tol"), "half": ("point", "normal"),
    "cylinder": ("point", "axis", "radius", "|tol"), "ball": ("center", "radius"),
    "normal": ("direction", "|angle_deg"), "faces": None, "patches": None,
}


class ModelError(ValueError):
    """A refused operation, with a reason the caller can act on."""


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _vec3(x) -> bool:
    return isinstance(x, (list, tuple)) and len(x) == 3 and all(_num(v) for v in x)


# ----------------------------------------------------------------- schema
@dataclass(frozen=True)
class Prop:
    """One setting of a node. None means "not set": it is left out of every
    export, so absent and default stay distinguishable."""
    key: str
    kind: str          # int | float | str | bool | choice | names | ints | vec3 | dofs | object
    unit: str = ""
    choices: tuple = ()  # choice / names: the allowed words (names: none = any); object: the allowed keys
    doc: str = ""

    def check(self, v):
        """`v` as stored, or ModelError. Numbers are checked finite; nothing is rounded."""
        if v is None:
            return None
        k, ok = self.kind, False
        if k == "int":
            ok = _int(v)
        elif k == "float":
            ok = _num(v)
        elif k == "str":
            ok = isinstance(v, str)
        elif k == "bool":
            ok = isinstance(v, bool)
        elif k == "choice":
            ok = isinstance(v, str) and v in self.choices
        elif k == "names":
            ok = isinstance(v, list) and all(isinstance(x, str) and (not self.choices or x in self.choices) for x in v)
        elif k == "ints":
            ok = isinstance(v, list) and all(_int(x) for x in v)
        elif k == "vec3":
            ok = _vec3(v)
        elif k == "dofs":
            ok = isinstance(v, dict) and set(v) <= set(DOFS) and all(isinstance(x, bool) for x in v.values())
        elif k == "object":
            ok = isinstance(v, dict) and (not self.choices or set(v) <= set(self.choices))
            if ok:
                json.dumps(v)          # must survive a save
        if not ok:
            want = {"choice": f"one of {list(self.choices)}", "names": f"a list from {list(self.choices)}",
                    "object": f"an object with keys from {list(self.choices)}" if self.choices else "an object"}
            raise ModelError(f"{self.key}: expected {want.get(k, k)}, got {v!r}")
        return copy.deepcopy(list(v) if k == "vec3" else v)

    def schema(self) -> dict:
        s = {"int": {"type": "integer"}, "float": {"type": "number"}, "str": {"type": "string"},
             "bool": {"type": "boolean"}, "choice": {"enum": list(self.choices)},
             "names": {"type": "array", "items": {"enum": list(self.choices)} if self.choices else {"type": "string"}},
             "ints": {"type": "array", "items": {"type": "integer"}},
             "vec3": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
             "dofs": {"type": "object", "properties": {d: {"type": "boolean"} for d in DOFS},
                      "additionalProperties": False},
             "object": {"type": "object", **({"propertyNames": {"enum": list(self.choices)}} if self.choices else {})},
             }[self.kind]
        doc = " ".join(x for x in (self.doc, f"[{self.unit}]" if self.unit else "") if x)
        return {**s, "description": doc} if doc else s


@dataclass(frozen=True)
class NodeType:
    kind: str
    title: str
    prefix: str                       # tag prefix: "sup" gives sup1, sup2, ...
    props: tuple = ()
    region: bool = False              # acts on a region of the surface
    children: tuple = ()              # kinds allowed directly under it
    singleton: bool = False           # at most one per parent
    rule: Callable | None = None      # rule(props, region) -> None, or raise ModelError
    doc: str = ""

    def prop(self, key) -> Prop:
        for p in self.props:
            if p.key == key:
                return p
        raise ModelError(f"{self.kind} has no setting {key!r} (has {[p.key for p in self.props]})")

    def schema(self) -> dict:
        """JSON schema of this node's props: the settings form and the agent's tool arguments."""
        return {"title": self.title, "description": self.doc, "type": "object",
                "properties": {p.key: p.schema() for p in self.props}, "additionalProperties": False}


REGISTRY: dict[str, NodeType] = {}


def register(nt: NodeType) -> NodeType:
    if nt.kind in REGISTRY:
        raise ValueError(f"node kind {nt.kind!r} registered twice")
    REGISTRY[nt.kind] = nt
    return nt


def node_type(kind: str) -> NodeType:
    try:
        return REGISTRY[kind]
    except KeyError:
        raise ModelError(f"unknown node kind {kind!r}") from None


# ----------------------------------------------------------------- region
def check_selector(s) -> None:
    """One selector object (keys ANDed) or a list of them (ORed), as case_io.hpp reads them."""
    if isinstance(s, list):
        if not s:
            raise ModelError("an empty selector list selects nothing")
        for x in s:
            check_selector(x)
        return
    if not isinstance(s, dict) or not s:
        raise ModelError(f"a selector is a non-empty object, got {s!r}")
    for k, v in s.items():
        if k not in SELECTORS:
            raise ModelError(f"unknown selector {k!r} (have {sorted(SELECTORS)})")
        if k == "box":
            if not (isinstance(v, list) and len(v) == 2 and all(_vec3(c) for c in v)):
                raise ModelError(f"box is [[x0, y0, z0], [x1, y1, z1]], got {v!r}")
        elif k in ("faces", "patches"):
            if not (isinstance(v, list) and v and all(_int(x) for x in v)):
                raise ModelError(f"{k} is a non-empty list of integers, got {v!r}")
        else:
            fields = SELECTORS[k]
            need = {f for f in fields if not f.startswith("|")}
            allowed = need | {f[1:] for f in fields if f.startswith("|")}
            if not isinstance(v, dict) or not need <= set(v) or not set(v) <= allowed:
                raise ModelError(f"{k} needs {sorted(need)} (optional {sorted(allowed - need)}), got {v!r}")
            for f, x in v.items():
                if not (_vec3(x) if f in ("point", "normal", "axis", "center", "direction") else _num(x)):
                    raise ModelError(f"{k}.{f}: bad value {x!r}")


REGION_KEYS = ("face_id", "face_ids", "patches", "select", "name")


def check_region(r: dict) -> dict:
    """Where a support or load acts, in case-file keys: face_id | face_ids | patches,
    optionally ANDed with select; or select alone; or the name of a named surface
    alone. CAD face ids survive a remesh; a geometric select also survives a
    re-import that renumbers faces."""
    if not isinstance(r, dict) or not r:
        raise ModelError(f"a region is an object with {list(REGION_KEYS)}, got {r!r}")
    unknown = set(r) - set(REGION_KEYS)
    if unknown:
        raise ModelError(f"region has unknown keys {sorted(unknown)}")
    if "name" in r:
        if not isinstance(r["name"], str) or not r["name"]:
            raise ModelError(f"name is a non-empty string, got {r['name']!r}")
        if len(r) > 1:
            raise ModelError("a named surface stands alone; put the selection in the name")
    if "face_id" in r and not _int(r["face_id"]):
        raise ModelError(f"face_id is an integer, got {r['face_id']!r}")
    for k in ("face_ids", "patches"):
        if k in r and not (isinstance(r[k], list) and r[k] and all(_int(x) for x in r[k])):
            raise ModelError(f"{k} is a non-empty list of integers, got {r[k]!r}")
    if "select" in r:
        check_selector(r["select"])
        if isinstance(r["select"], list) and len(r) > 1:
            raise ModelError("a selector list cannot be combined with face_id, face_ids or patches")
    return copy.deepcopy(r)


# ----------------------------------------------------------------- tree
@dataclass
class Node:
    kind: str
    tag: str
    props: dict = field(default_factory=dict)
    region: dict | None = None
    children: list = field(default_factory=list)

    @property
    def type(self) -> NodeType:
        return node_type(self.kind)

    def get(self, key):
        self.type.prop(key)
        return self.props.get(key)

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "tag": self.tag}
        if self.props:
            d["props"] = copy.deepcopy(self.props)
        if self.region is not None:
            d["region"] = copy.deepcopy(self.region)
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        nt = node_type(d["kind"])
        props = _checked_props(nt, d.get("props", {}))
        region = d.get("region")
        _check_node(nt, props, region)
        n = cls(nt.kind, d["tag"], props, check_region(region) if region is not None else None)
        for c in d.get("children", []):
            child = cls.from_dict(c)
            _check_child(n, child.kind)
            n.children.append(child)
        return n


def _checked_props(nt: NodeType, props: dict) -> dict:
    out = {}
    for k, v in (props or {}).items():
        v = nt.prop(k).check(v)
        if v is not None:
            out[k] = v
    return out


def _check_node(nt: NodeType, props: dict, region):
    if region is not None:
        if not nt.region:
            raise ModelError(f"a {nt.kind} takes no region")
        check_region(region)
    if nt.rule:
        nt.rule(props, region)


def _check_child(parent: Node, kind: str):
    if kind not in parent.type.children:
        raise ModelError(f"a {kind} cannot go under {parent.kind} (allowed: {list(parent.type.children)})")
    if node_type(kind).singleton and any(c.kind == kind for c in parent.children):
        raise ModelError(f"{parent.kind} {parent.tag} already has a {kind}")


class Model:
    """The tree, its operations, and its file."""

    def __init__(self, root: Node | None = None):
        self.root = root or Node("case", "root")

    # -- lookup
    def nodes(self):
        stack = [self.root]
        while stack:
            n = stack.pop()
            yield n
            stack.extend(reversed(n.children))

    def find(self, tag: str) -> Node:
        for n in self.nodes():
            if n.tag == tag:
                return n
        raise ModelError(f"no node {tag!r}")

    def parent_of(self, tag: str) -> Node:
        for n in self.nodes():
            if any(c.tag == tag for c in n.children):
                return n
        raise ModelError(f"{tag!r} has no parent")

    def of_kind(self, kind: str) -> list:
        return [n for n in self.nodes() if n.kind == kind]

    def _new_tag(self, prefix: str) -> str:
        used = {n.tag for n in self.nodes()}
        i = 1
        while f"{prefix}{i}" in used:
            i += 1
        return f"{prefix}{i}"

    # -- operations
    def apply(self, op: dict):
        """Apply one operation. Returns the new node's tag for "add", else None."""
        if not isinstance(op, dict) or "op" not in op:
            raise ModelError(f"an operation is a dict with 'op', got {op!r}")
        kind = op["op"]
        if kind == "add":
            parent = self.find(op.get("parent", self.root.tag))
            nt = node_type(op.get("kind"))
            _check_child(parent, nt.kind)
            props = _checked_props(nt, op.get("props"))
            region = op.get("region")
            _check_node(nt, props, region)
            tag = op.get("tag") or self._new_tag(nt.prefix)
            if any(n.tag == tag for n in self.nodes()):
                raise ModelError(f"tag {tag!r} is taken")
            parent.children.append(Node(nt.kind, tag, props, check_region(region) if region is not None else None))
            return tag
        if kind == "set":
            n = self.find(op["tag"])
            v = n.type.prop(op["key"]).check(op.get("value"))
            props = {k: x for k, x in n.props.items() if k != op["key"]}
            if v is not None:
                props[op["key"]] = v
            _check_node(n.type, props, n.region)
            n.props = props
            return None
        if kind == "select":
            n = self.find(op["tag"])
            region = op.get("region")
            _check_node(n.type, n.props, region)
            n.region = check_region(region) if region is not None else None
            return None
        if kind == "batch":
            ops = op.get("ops")
            if not isinstance(ops, list) or not ops:
                raise ModelError("batch takes a non-empty list of operations")
            before = self.snapshot()
            try:
                return [self.apply(o) for o in ops]
            except Exception:
                self.restore(before)       # all or nothing
                raise
        if kind == "remove":
            if op["tag"] == self.root.tag:
                raise ModelError("the root cannot be removed")
            self.find(op["tag"])
            parent = self.parent_of(op["tag"])
            parent.children = [c for c in parent.children if c.tag != op["tag"]]
            return None
        raise ModelError(f"unknown operation {kind!r} (add, set, select, remove, batch)")

    def replay(self, ops) -> list:
        return [self.apply(op) for op in ops]

    # -- snapshot and file
    def snapshot(self) -> str:
        return json.dumps(self.root.to_dict(), sort_keys=True, separators=(",", ":"))

    def restore(self, snap: str):
        self.root = Node.from_dict(json.loads(snap))

    def to_dict(self) -> dict:
        return {"format": FORMAT, "version": VERSION, "root": self.root.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "Model":
        if d.get("format") != FORMAT:
            raise ModelError(f"not an {FORMAT} file (format {d.get('format')!r})")
        if d.get("version") != VERSION:
            raise ModelError(f"model version {d.get('version')!r}; this build reads {VERSION}")
        return cls(Node.from_dict(d["root"]))

    def save(self, path):
        pathlib.Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Model":
        return cls.from_dict(json.loads(pathlib.Path(path).read_text(encoding="utf-8")))


class History:
    """Operations with undo and redo by snapshot. `log` holds the operations
    that took effect, in order, for the journal; an undone one is popped."""

    def __init__(self, model: Model, depth: int = 200):
        self.model, self.depth = model, depth
        self._undo: list = []
        self._redo: list = []
        self.log: list = []

    def do(self, op: dict):
        before = self.model.snapshot()
        try:
            out = self.model.apply(op)
        except Exception:
            self.model.restore(before)     # apply checks first, but a refusal must never leave a half change
            raise
        self._undo.append((before, op))
        del self._undo[:-self.depth]
        self._redo.clear()
        self.log.append(op)
        return out

    def undo(self) -> bool:
        if not self._undo:
            return False
        before, op = self._undo.pop()
        self._redo.append((self.model.snapshot(), op))
        self.model.restore(before)
        self.log.pop()
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        after, op = self._redo.pop()
        self._undo.append((self.model.snapshot(), op))
        self.model.restore(after)
        self.log.append(op)
        return True
