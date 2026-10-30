"""The IgBEM model core: a typed tree that reads and writes the case files
app/bc_case runs (src/case_io.hpp). Standard library only: nothing here
imports numpy, gmsh, the CAD kernel or the solver."""
from . import nodes  # noqa: F401  (registers the node types)
from .casefile import from_case, read_case, to_case, write_case  # noqa: F401
from .model import History, Model, ModelError, Node, NodeType, Prop, node_type  # noqa: F401
