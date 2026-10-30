# Vendored from nurb-fea-tcd (github.com/Fakasht10/nurb-fea-tcd, MIT, Aleksey Kashtelyan), feakit/materials.py at
# commit 553dd2a. See python/failure/LICENSE-nurb-fea-tcd. Local changes are listed in python/failure/README.md.
"""Production material registry with explicit numerical/provenance boundaries."""
from __future__ import annotations
import copy
import hashlib
import json
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_INVENTORY = Path(__file__).with_name("data") / "material_inventory.json"
WEIBULL_ARTIFACT_SCHEMA_VERSION = 2
WEIBULL_GATE_VERSION = "weibull-3d-thickness-aware-v2"
_SOURCE = {
    "failure_law": {"citation": "Fidan & Unal (2026), stress weakest-link volume law", "doi": "10.1007/s41779-026-01364-7"},
    "pla_calibration_dataset": {"citation": "Ahmed & Susmel (2018), PLA experimental data", "doi": "10.1111/ffe.12746", "note": "Archived frozen 3-D calibration derived for the listed raster cards."},
    "geometry_convention": {"description": "3-D octant, full-domain tensile quadrature, 1 N reference load"},
}


_GENERIC_REGISTRY = Path(__file__).with_name("data") / "generic_material_registry.json"
_GENERIC_WEIBULL_FORMULA = "S0=UTS/Gamma(1+1/m), inferred at V0=1 mm3; rounded as supplied"
_GENERIC_LIMITATIONS = ("Provisional engineering estimates only; validate against grade-, process-, "
                        "orientation-, environment-, and geometry-specific test data before design use.")


def _generic_cards():
    """Load generic cards from the visible database table's single JSON source.

    Fracture display inputs remain MPa sqrt(m) in the table; solver inputs are
    converted here to MPa sqrt(mm), preventing an easy 1000x unit mistake.
    """
    rows = json.loads(_GENERIC_REGISTRY.read_text(encoding="utf-8"))
    cards = []
    for raw in rows:
        row = copy.deepcopy(raw)
        if not row.get("solve_enabled", True):
            row.update(status="evidence_only", solve_enabled=False)
            cards.append(row)
            continue
        display_tcd = row["tcd"]
        display_k = float(display_tcd["K_Ic_MPa_sqrt_m"])
        tcd = dict(display_tcd,
                   K_Ic_MPa_sqrt_mm=display_k * (1000.0 ** .5),
                   status="provisional-engineering-estimate",
                   screening_status="provisional-engineering-estimate",
                   provisional=True,
                   source="Card-authoritative provisional engineering screening input; no literature citation is claimed.",
                   rationale="Finite screening default retained while material-specific fracture qualification is pending.",
                   limitations=_GENERIC_LIMITATIONS)
        weibull = dict(row["weibull"], status="provisional-engineering-estimate", provisional=True,
                       formula=_GENERIC_WEIBULL_FORMULA,
                       source="Card-authoritative provisional engineering screening input; no calibration artifact was checked.",
                       rationale="Mean-UTS-derived provisional weakest-link screening default.",
                       limitations=_GENERIC_LIMITATIONS)
        row.update(status="generic_approximation", solve_enabled=True, tcd=tcd, weibull=weibull,
                   E_status="provisional-engineering-approximation",
                   nu_status="provisional-engineering-approximation",
                   provenance={"label": "textbook/approximate generic engineering card; visibly provisional and not a calibrated fracture card",
                               "restrictions": _GENERIC_LIMITATIONS})
        cards.append(row)
    return cards
_WEIBULL_CAMPAIGN = _ROOT / "data" / "calibration" / "weibull_campaign.json"
_ABS_TET10_MATERIAL_FIT = _ROOT / "data" / "calibration" / "abs_tet10_material_fit.json"

_TCD_LITERATURE_BASELINES = {
    "pla": {
        "K_Ic_MPa_sqrt_mm": 117.00427342623004,
        "sigma_uts_MPa": 42.9,
        "status": "literature-baseline",
        "provisional": True,
        "angle_fitted": False,
        "source": {
            "path": "incoming/fracture_screening_work_handoff_2026-09-09/fracture_screening_work_handoff_2026-09-09/fracture_handoff_validation/baseline/pla_dataset/source.pdf",
            "citation": "Ahmed & Susmel (2018), PLA source; K_Ic=3.7 MPa sqrt(m), sigma_UTS=42.9 MPa",
            "conversion": "K_Ic multiplied by sqrt(1000) to express MPa sqrt(mm)",
        },
    },
    "abs-tet10": {
        "K_Ic_MPa_sqrt_mm": 82.21921916437786,
        "sigma_uts_MPa": 23.0,
        "status": "literature-baseline",
        "provisional": True,
        "angle_fitted": False,
        "source": {
            "path": "incoming/fracture_handoff_validation/baseline/hsv_percent/README.md",
            "citation": "Ng--Susmel process baseline; K_Ic=2.6 MPa sqrt(m), sigma_c=23 MPa",
            "conversion": "K_Ic multiplied by sqrt(1000) to express MPa sqrt(mm)",
        },
    },
}


def _with_tcd_literature_baseline(cards, family):
    """Attach a disclosed shared literature baseline, never an angle-specific fit."""
    baseline = _TCD_LITERATURE_BASELINES[family]
    return [dict(card, tcd=copy.deepcopy(baseline)) for card in cards]


def canonical_weibull_artifact_sha256(artifact: dict) -> str:
    """Hash all evidence except recursive artifact/card self-reference fields."""
    payload = copy.deepcopy(artifact)
    payload.pop("artifact_sha256", None)
    for card in payload.get("cards", []):
        card.get("weibull", {}).pop("calibration_sha256", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _valid_calibrated_artifact(artifact: dict) -> bool:
    if artifact.get("schema_version") != WEIBULL_ARTIFACT_SCHEMA_VERSION or artifact.get("gate_version") != WEIBULL_GATE_VERSION:
        return False
    if artifact.get("campaign_status") != "PASS" or artifact.get("numerical_status") != "PASS":
        return False
    if not artifact.get("calibration_consistency", {}).get("pass", False):
        return False
    digest = artifact.get("artifact_sha256")
    if not isinstance(digest, str) or digest != canonical_weibull_artifact_sha256(artifact):
        return False
    cards = artifact.get("cards")
    if not isinstance(cards, list) or not cards:
        return False
    for card in cards:
        weibull = card.get("weibull", {})
        if not card.get("solve_enabled") or weibull.get("calibration_sha256") != digest:
            return False
        if not all(isinstance(weibull.get(key), (int, float)) for key in ("m", "S0_MPa", "V0_mm3")):
            return False
    return True


def _calibrated():
    """Load only self-consistent, canonical-hash-gated generated Weibull cards."""
    if not _WEIBULL_CAMPAIGN.exists():
        return []
    try:
        artifact = json.loads(_WEIBULL_CAMPAIGN.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not _valid_calibrated_artifact(artifact):
        return []
    cards = []
    for raw in artifact["cards"]:
        card = dict(raw)
        card["weibull"] = dict(raw["weibull"], source=_SOURCE)
        cards.append(card)
    return cards


def _abs_tet10_final():
    """Load final ABS cards only from the hash- and all-gates-qualified artifact."""
    if not _ABS_TET10_MATERIAL_FIT.exists():
        return []
    try:
        artifact = json.loads(_ABS_TET10_MATERIAL_FIT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    gates = artifact.get("gates", {})
    if (artifact.get("scope") != "final ABS production TET10 medium radial-only material fit"
            or artifact.get("campaign_status") != "PASS" or artifact.get("numerical_status") != "PASS"
            or not gates or not all(gate.get("pass") is True for gate in gates.values())
            or artifact.get("artifact_sha256") != canonical_weibull_artifact_sha256(artifact)):
        return []
    digest = artifact["artifact_sha256"]
    cards = artifact.get("cards", [])
    if len(cards) != 3:
        return []
    expected = {"abs-tet10-raster-0-weibull", "abs-tet10-raster-30-weibull", "abs-tet10-raster-45-weibull"}
    if {card.get("id") for card in cards} != expected:
        return []
    for card in cards:
        weibull = card.get("weibull", {})
        if (not card.get("solve_enabled") or card.get("E_MPa") != 1590.0 or card.get("nu") != .36
                or weibull.get("calibration_sha256") != digest
                or not all(isinstance(weibull.get(key), (int, float)) for key in ("m", "S0_MPa", "V0_mm3"))):
            return []
    return cards


_ASA_FRADDCO = Path(__file__).with_name("data") / "asa_fraddco_cards.json"


def _asa_fraddco():
    """ASA screening-fit cards (FRADDCO SENB data, Zenodo 10.5281/zenodo.14065523).

    Refused as a whole when the file does not hash to its recorded digest, so a
    hand edit to one constant cannot ship under the fit's provenance.
    """
    try:
        doc = json.loads(_ASA_FRADDCO.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    cards = doc.get("cards", [])
    # sealed with its contact (pads / strips) too: that field says what the constants were fitted on
    digest = hashlib.sha256(json.dumps({"contact": doc.get("contact"), "cards": cards}, sort_keys=True,
                                       separators=(",", ":")).encode()).hexdigest()
    if doc.get("schema_version") != 1 or digest != doc.get("cards_sha256"):
        return []
    return cards


def _evidence_cards():
    rows = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    return [{"id": "evidence-" + re.sub(r"[^a-z0-9]+", "-", r["material"].lower()).strip("-") + "-" + hashlib.sha256(r["material"].encode("utf-8")).hexdigest()[:10], "name": r["material"], "status": "evidence_only", "solve_enabled": False, "source_id": r["source"], "notes": r["screening_notes"], "provenance": {"checked": r["checked"], "priority": r["priority"]}} for r in rows]


# What each card's constants were fitted on or checked against, so a user or an agent can tell which card fits a
# part (agent test 2026-10-23, finding C: an agent on a 5 mm SENB ligament kept the tension-plate ABS card, L 4.1 mm,
# and never switched to ABS-M30-iso, which is the one checked on that specimen). Kept outside the hashed artifacts:
# adding it does not touch their digests. kind: "fitted" (constants fitted to these tests), "checked" (set by hand and
# compared with these tests), "none" (generic estimate). family: the material, for suggesting a card of the same one.
_CALIBRATED_ON = {
    "ABS-M30-iso": {"kind": "checked", "family": "ABS",
                    "geometry": "SENB three-point bend, 5 mm ligament, notch root radius 0.25-2 mm",
                    "source": "Project values set 2026-10-14 (K_Ic 75 MPa sqrt(mm), sigma_0 63 MPa, S0 100 MPa, m 5); "
                              "compared with Cicero's measured SENB loads (Table 2): -8 to +6 % over the four radii"},
    "pla-raster-": {"kind": "fitted", "family": "PLA",
                    "geometry": "PLA notched specimens of Ahmed & Susmel (2018), the named raster",
                    "source": "Weibull m, S0 fitted (archived 3-D campaign); TCD constants are that paper's literature "
                              "baseline (K_Ic 3.7 MPa sqrt(m), sigma 42.9 MPa), not fitted here"},
    "abs-tet10-raster-": {"kind": "fitted", "family": "ABS",
                          "geometry": "Ng-Susmel ABS notched tension plates (UB/UI/US/OB/OI/OS), the named raster, "
                                      "TET10 h 0.15 mm",
                          "source": "Weibull m, S0 fitted on UB and US, held out on UI/OB/OI/OS; TCD constants are the "
                                    "Ng-Susmel literature baseline (K_Ic 2.6 MPa sqrt(m), sigma 23 MPa, L 4.1 mm), which "
                                    "suits plates with long ligaments, not short bend specimens"},
    "asa-fraddco-": {"kind": "fitted", "family": "ASA",
                     "geometry": "SENB three-point bend, notch radius 0-2 mm, the named raster (FRADDCO, 71 tests)",
                     "source": "Arrieta, Cicero et al., FRADDCO ASA notch data (Zenodo 10.5281/zenodo.14065523); "
                               "screening fit, see the card's use_note"},
}
_GENERIC_FAMILY = {"PLA": "PLA", "PETG": "PETG", "ASA": "ASA", "PC": "PC", "PA12": "PA12", "PA6": "PA6",
                   "Onyx-chopped-fiber-nylon": "PA (chopped fibre)", "ULTEM-9085-PEI": "PEI"}


def calibrated_on(card):
    """{kind, family, geometry, source} for a card: what its constants were fitted on or checked against."""
    cid = str(card.get("id") or "")
    if card.get("calibrated_on"):
        return card["calibrated_on"]
    for key, info in _CALIBRATED_ON.items():
        if cid == key or (key.endswith("-") and cid.startswith(key)):
            return dict(info)
    return {"kind": "none", "family": _GENERIC_FAMILY.get(cid), "geometry": None,
            "source": "generic estimate from typical published values; not fitted to notched tests"}


def calibrated_on_text(card):
    """One line for a card list: 'fitted on ...', 'checked against ...' or 'not calibrated (generic estimate)'."""
    c = calibrated_on(card)
    if c["kind"] == "none" or not c.get("geometry"):
        return "not calibrated (generic estimate)"
    return f"{'fitted on' if c['kind'] == 'fitted' else 'checked against'} {c['geometry']}"


def all_cards():
    # Evidence-only database rows are deliberately table-only.  Keeping them
    # out of all_cards avoids duplicating the separately imported inventory.
    generic = [card for card in _generic_cards() if card["solve_enabled"]]
    cards = [*generic,
             *_with_tcd_literature_baseline(_calibrated(), "pla"),
             *_with_tcd_literature_baseline(_abs_tet10_final(), "abs-tet10"),
             *_asa_fraddco()]
    return [dict(c, calibrated_on=calibrated_on(c)) for c in cards] + _evidence_cards()


def solve_material_ids():
    return [c["id"] for c in all_cards() if c["solve_enabled"]]


def material_table():
    """Generic database rows shown by the viewer, including evidence-only rows."""
    return _generic_cards()


def weibull_cards():
    """Return calibrated Weibull cards only; provisional solve cards stay in all_cards()."""
    return [c for c in all_cards() if (c.get("weibull") or {}).get("calibration_sha256") and c["solve_enabled"]]


def card(material_id):
    for item in all_cards():
        if item["id"] == material_id:
            return item
    raise KeyError(material_id)


def inventory_path():
    return _INVENTORY
