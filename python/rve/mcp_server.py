"""MCP server for printed-material cards (rve.api): an agent picks a material, layup, raster angle and bead
geometry and gets the orthotropic stiffness, porosity, bond strengths and an orientation card back.

Tools
  list_materials        registry materials with E, nu, sigma_uts
  printed_material      rve.api.printed_card (milliseconds, from the precomputed grid)
  grid_range            the geometry box the grid covers (queries outside are refused)

Run (stdio):  cd python && python3 -m rve.mcp_server
Claude Code:  claude mcp add igabem-rve -- python3 -m rve.mcp_server   (from python/)
Needs the MCP Python SDK 2.x (pip install "mcp>=2").
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mcp.server.mcpserver import MCPServer  # noqa: E402

from rve import api  # noqa: E402

server = MCPServer(name="igabem-rve", version="1.0",
                   instructions="Homogenized stiffness and bond strength of FDM-printed polymers from IgBEM bead "
                                "cells. Porosity is an output; give the bond widths. Read the flags of every answer.")


@server.tool(description="Materials in the project registry (id, E_MPa, nu, sigma_uts_MPa).")
def list_materials() -> list:
    with open(api.REGISTRY) as f:
        reg = json.load(f)
    return [dict(id=r["id"], name=r.get("name"), E_MPa=r.get("E_MPa"), nu=r.get("nu"),
                 sigma_uts_MPa=(r.get("tcd") or {}).get("sigma_uts_MPa")) for r in reg if r.get("E_MPa")]


@server.tool(description="Geometry box of the precomputed grid: h/w, b_int/w, b_lat/h, and nu.")
def grid_range() -> dict:
    g = api._grid()[0]
    return dict(h_over_w=g["axes"]["h"], b_int_over_w=g["axes"]["b_int"], b_lat_over_h=g["axes"]["beta"],
                nu=sorted(float(k) for k in g["c00"]), notch_deg=g["alpha"])


@server.tool(description=(
    "Orthotropic stiffness (6x6, part frame, Voigt 11 22 33 23 13 12), engineering constants, in-plane E(theta), "
    "porosity, bond strengths and an orientation card for a printed polymer. material: registry id. layup: "
    "'aligned' or 'cross' (raster/raster-90). Geometry in mm: w filament pitch, h layer height, b_int interlayer "
    "weld width, b_lat in-layer weld height; or porosity with bond='from_porosity' (flagged). Optional "
    "weld_strength_MPa, weld_efficiency, Gc_N_per_mm."))
def printed_material(material: str, w: float, h: float, b_lat: float, layup: str = "cross",
                     raster_deg: float = 0.0, b_int: float | None = None, porosity: float | None = None,
                     bond: str = "given", weld_strength_MPa: float | None = None,
                     weld_efficiency: float | None = None, Gc_N_per_mm: float | None = None) -> dict:
    return api.printed_card(material, layup=layup, raster_deg=raster_deg, w=w, h=h, b_int=b_int, b_lat=b_lat,
                            porosity=porosity, bond=bond, weld_strength_MPa=weld_strength_MPa,
                            weld_efficiency=weld_efficiency, Gc_N_per_mm=Gc_N_per_mm)


if __name__ == "__main__":
    server.run("stdio")
