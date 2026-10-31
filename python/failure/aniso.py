"""Print-orientation failure of FDM parts: ply frames, three tension-only modes, competing-risk weakest link.

The stress comes from the isotropic solve; anisotropy enters through the strength only (scope doc, 2026-10-30).
A print is a build direction d and a layup of plies (angle about d from an in-plane reference axis, volume share).
Each ply has the frame e1 = filament, e2 = in-plane transverse, e3 = d, and three Hashin-form modes:

    F  filament           sigma_F = sqrt(<s11>^2 + (X/S12)^2 (s12^2 + s13^2))   (Hashin; shear_ratio 0 by default)
    T  filament bond      sigma_T = sqrt(<s22>^2 + (Y/S12)^2 s12^2)
    L  layer bond         sigma_L = sqrt(<s33>^2 + (Z/S13)^2 (s13^2 + s23^2))
    P  isotropic          sigma_P = max(lambda_max(sigma), 0)   (the vendored weibull.py law; regression mode)

Every mode stress is homogeneous of degree 1 in sigma, so the hazard of a mode scales as load^m and the hazards add
(independent flaw populations). With one exponent m for all modes the characteristic load is closed form:

    H(F) = (F / F_ref)^m sum_k J_k / S0_k^m,   J_k = (1/V0) sum_p w_p int (sigma_k,p / n_k,p)^m dV

n is the optional gradient support of Maede, Schmitz, Gottschalk & Beck (arXiv 1709.00320), n = 1 + sqrt(l chi),
chi = |grad sigma_k| / sigma_k, with l a TCD-like length per mode; n = 1 without it.

Units: MPa, mm, mm^3, N. Cards are plain dicts:
    {"m": 4.6, "V0_mm3": 1.0, "modes": {"F": {"S0_MPa": 350}, "T": {"S0_MPa": 300, "shear_ratio": 1.2},
                                        "L": {"S0_MPa": 180, "shear_ratio": 1.0, "support_mm": 0.3}}}
A mode missing from "modes" is off (never guessed).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MODES = ("P", "F", "T", "L")


# ------------------------------------------------------------------------------------------- print frames
@dataclass(frozen=True)
class Print:
    """Build direction, layup angles (deg, about d from the in-plane reference) and their volume shares."""
    build_dir: tuple = (0.0, 0.0, 1.0)
    angles_deg: tuple = (45.0, -45.0)
    shares: tuple | None = None          # None: equal shares
    ref_dir: tuple = (1.0, 0.0, 0.0)     # projected onto the layer plane; the axis least aligned with d if parallel

    def plies(self):
        s = np.full(len(self.angles_deg), 1.0 / len(self.angles_deg)) if self.shares is None \
            else np.asarray(self.shares, float)
        if not np.isclose(s.sum(), 1.0) or np.any(s <= 0):
            raise ValueError("ply shares must be positive and sum to one")
        return np.asarray(self.angles_deg, float), s


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def ply_frames(build_dirs, angles_deg, ref_dir=(1.0, 0.0, 0.0)):
    """Frames R (nd, np, 3, 3) with columns e1, e2, e3 for build directions (nd, 3) and ply angles (np,)."""
    d = _unit(np.atleast_2d(build_dirs))
    ref = np.broadcast_to(np.asarray(ref_dir, float), d.shape).copy()
    par = np.abs(np.einsum("di,di->d", _unit(ref), d)) > 1.0 - 1e-9
    ref[par] = np.eye(3)[np.argmin(np.abs(d[par]), axis=1)]             # the global axis least aligned with d
    a = _unit(ref - np.einsum("di,di->d", ref, d)[:, None] * d)          # in-plane reference axis
    b = np.cross(d, a)
    t = np.radians(np.asarray(angles_deg, float))
    e1 = np.cos(t)[None, :, None] * a[:, None] + np.sin(t)[None, :, None] * b[:, None]
    e2 = np.cross(d[:, None], e1)
    e3 = np.broadcast_to(d[:, None], e1.shape)
    return np.stack([e1, e2, e3], axis=-1)


def fibonacci_hemisphere(n):
    """n near-uniform unit vectors with z >= 0 (d and -d give the same print)."""
    i = np.arange(n) + 0.5
    z = 1.0 - i / n
    r = np.sqrt(1.0 - z * z)
    phi = i * math.pi * (3.0 - math.sqrt(5.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


# ------------------------------------------------------------------------------------------- mode stresses
_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))       # s11 s22 s33 s12 s13 s23


def components(sig, R, grad=None):
    """Ply components c (6, ..., n) of sig (n, 3, 3) in frames R (..., 3, 3), and with grad (n, 3, 3, 3) = d sigma_ij
    / d x_l their gradients g (6, ..., n, 3). Precompute once per print; every mode and parameter reads them."""
    E = np.moveaxis(R, -1, 0)
    c = np.stack([np.einsum("...i,...j,nij->...n", E[a], E[b], sig) for a, b in _PAIRS])
    g = None if grad is None else np.stack([np.einsum("...i,...j,nijl->...nl", E[a], E[b], grad) for a, b in _PAIRS])
    return c, g


def mode_from_components(mode, c, shear_ratio=None):
    if shear_ratio is None:
        shear_ratio = 0.0 if mode == "F" else 1.0
    if mode == "F":
        f = np.maximum(c[0], 0.0)
        return f if not shear_ratio else np.sqrt(f ** 2 + shear_ratio ** 2 * (c[3] ** 2 + c[4] ** 2))
    if mode == "T":
        return np.sqrt(np.maximum(c[1], 0.0) ** 2 + (shear_ratio * c[3]) ** 2)
    if mode == "L":
        return np.sqrt(np.maximum(c[2], 0.0) ** 2 + shear_ratio ** 2 * (c[4] ** 2 + c[5] ** 2))
    raise ValueError(f"unknown mode {mode!r}; one of {MODES}")


def mode_gradient_from_components(mode, c, g, shear_ratio=None):
    """|grad sigma_k| by the chain rule on the component gradients."""
    if shear_ratio is None:
        shear_ratio = 0.0 if mode == "F" else 1.0
    if mode == "F" and not shear_ratio:
        d = g[0] * (c[0] > 0)[..., None]
    else:
        n, shears = {"F": (0, (3, 4)), "T": (1, (3,)), "L": (2, (4, 5))}[mode]
        d = np.maximum(c[n], 0.0)[..., None] * g[n]
        for k in shears:
            d = d + shear_ratio ** 2 * c[k][..., None] * g[k]
        s = mode_from_components(mode, c, shear_ratio)
        d = np.divide(d, s[..., None], out=np.zeros_like(d), where=s[..., None] > 0)
    return np.linalg.norm(d, axis=-1)


def mode_stress(mode, sig, R=None, shear_ratio=None):
    """Equivalent stress (..., n) of one mode; R (..., 3, 3) is the ply frame (unused by P)."""
    if mode == "P":
        return np.maximum(np.linalg.eigvalsh(0.5 * (sig + np.swapaxes(sig, -1, -2)))[..., -1], 0.0)
    r = (0.0 if mode == "F" else 1.0) if shear_ratio is None else shear_ratio
    return mode_from_components(mode, components(sig, R)[0], r)


def mode_gradient(mode, sig, grad, R, shear_ratio=None):
    """|grad sigma_k| (..., n) from the stress gradient grad (n, 3, 3, 3) = d sigma_ij / d x_l (chain rule)."""
    if mode == "P":
        w, v = np.linalg.eigh(sig)
        e = v[..., -1]
        g = np.einsum("ni,nj,nijl->nl", e, e, grad) * (w[..., -1] > 0)[:, None]
        return np.linalg.norm(g, axis=-1)
    r = (0.0 if mode == "F" else 1.0) if shear_ratio is None else shear_ratio
    return mode_gradient_from_components(mode, *components(sig, R, grad), r)


def support_factor(s, grad_norm, length_mm):
    """n = 1 + sqrt(l chi), chi = |grad s| / s (Maede et al. 2017); 1 where s = 0."""
    chi = np.divide(grad_norm, s, out=np.zeros_like(s), where=s > 0)
    return 1.0 + np.sqrt(length_mm * chi)


# ------------------------------------------------------------------------------------------- weakest link
def log_power_integral(seq, w, m, V0=1.0):
    """log( sum w seq^m / V0 ) over seq > 0, as weibull.log_stress_power_integral (-inf for an empty set)."""
    seq, w = np.asarray(seq, float).reshape(-1), np.broadcast_to(np.asarray(w, float), np.shape(seq)).reshape(-1)
    pos = seq > 0.0
    if not np.any(pos):
        return -math.inf
    logs = np.log(w[pos]) + m * np.log(seq[pos])
    a = float(np.max(logs))
    return a + math.log(float(np.exp(logs - a).sum())) - math.log(V0)


def _card_modes(card):
    modes = card.get("modes") or {}
    bad = set(modes) - set(MODES)
    if bad:
        raise ValueError(f"unknown modes {sorted(bad)}")
    if not modes:
        raise ValueError("the card has no failure mode")
    return modes


def mode_hazards(sig, w, card, pr=None, grad=None, R=None, comps=None, s_p=None):
    """{mode: log(J_k / S0_k^m)} at the reference load of sig (n, 3, 3) with weights w (n,) mm^3.
    grad (n, 3, 3, 3) is needed only for modes with "support_mm". R overrides the frames of `pr` (np, 3, 3);
    comps = components(sig, R, grad) precomputed for those frames skips the rotation, and s_p = mode_stress("P", sig)
    the eigenvalues (fits over many cards)."""
    m, V0 = float(card["m"]), float(card.get("V0_mm3", 1.0))
    modes = _card_modes(card)
    shares = pr.plies()[1] if pr is not None else np.ones(1)
    if comps is None and set(modes) - {"P"}:
        if R is None:
            if pr is None:
                raise ValueError("ply modes need a print (build direction and layup)")
            R = ply_frames(pr.build_dir, pr.plies()[0], pr.ref_dir)[0]
        comps = components(sig, R, grad)
    out = {}
    for k, p in modes.items():
        ratio, l = float(p.get("shear_ratio", 0.0 if k == "F" else 1.0)), p.get("support_mm")
        if k == "P":
            s, ws = (mode_stress("P", sig) if s_p is None else s_p)[None], [1.0]
            if l:
                s = s / support_factor(s, mode_gradient("P", sig, grad, None)[None], float(l))
        else:
            c, g = comps
            s, ws = mode_from_components(k, c, ratio), shares                 # (np, n)
            if l:
                s = s / support_factor(s, mode_gradient_from_components(k, c, g, ratio), float(l))
        lj = [log_power_integral(si, w * wi, m, V0) for si, wi in zip(s, ws)]
        out[k] = _logsumexp(lj) - m * math.log(float(p["S0_MPa"]))
    return out


def _logsumexp(v):
    v = np.asarray(v, float)
    a = np.max(v)
    return float(a) if not np.isfinite(a) else float(a + np.log(np.exp(v - a).sum()))


def characteristic_load(log_h, m, ref_load_N=1.0):
    """63.2 % load of the competing modes: F_ref (sum_k H_k)^(-1/m)."""
    total = _logsumexp(list(log_h.values()))
    if not np.isfinite(total):
        raise ValueError("no mode carries tension: the hazard is zero")
    return ref_load_N * math.exp(-total / float(m))


def mode_shares(log_h):
    """Fraction of the hazard per mode (which mode governs)."""
    total = _logsumexp(list(log_h.values()))
    return {k: math.exp(v - total) if np.isfinite(v) else 0.0 for k, v in log_h.items()}


def failure_probability(load_N, log_h, m, ref_load_N=1.0):
    """Weakest-link P_f at a load (with the card's m; for scatter bands use failure.prob)."""
    return -math.expm1(-(load_N / characteristic_load(log_h, m, ref_load_N)) ** float(m))


def sweep(sig, w, card, pr, dirs, grad=None, chunk=16):
    """Characteristic load factor (per unit reference load) and governing mode for each build direction (nd, 3),
    with the layup of `pr`. The isotropic stress does not depend on the print, so one solve serves every direction."""
    angles, _ = pr.plies()
    dirs = _unit(np.atleast_2d(dirs))
    eta, gov = np.empty(len(dirs)), []
    for i in range(0, len(dirs), chunk):
        Rs = ply_frames(dirs[i:i + chunk], angles, pr.ref_dir)
        for j, R in enumerate(Rs):
            lh = mode_hazards(sig, w, card, pr, grad, R)
            eta[i + j] = characteristic_load(lh, card["m"])
            gov.append(max(lh, key=lh.get))
    return eta, gov


def tcd_point(sig_line, distances, card, pr=None, R=None):
    """Point method per mode along an extraction line: sig_line (nd, 3, 3) at distances (nd,) mm from the root.
    card["modes"][k] carries K (MPa sqrt(mm)) and sigma_0 (MPa), optionally shear_ratio; L_k = (K/sigma_0)^2 / pi.
    Returns {mode: {"L_mm", "probed", "ratio"}} (probed: the worst ply's mode stress at L_k/2) and the governing
    mode; the failure load is applied / max ratio (linear elasticity)."""
    modes = _card_modes(card)
    if R is None and pr is not None:
        R = ply_frames(pr.build_dir, pr.plies()[0], pr.ref_dir)[0]
    out = {}
    for k, p in modes.items():
        L = (float(p["K"]) / float(p["sigma_0"])) ** 2 / math.pi
        s = mode_stress(k, sig_line)[None] if k == "P" else \
            mode_stress(k, sig_line, R, float(p.get("shear_ratio", 0.0 if k == "F" else 1.0)))
        probed = max(float(np.interp(0.5 * L, distances, si)) for si in s)
        out[k] = {"L_mm": L, "probed": probed, "ratio": probed / float(p["sigma_0"])}
    gov = max(out, key=lambda k: out[k]["ratio"])
    return out, gov

# ------------------------------------------------------------------------------------------- closed forms
def off_axis_strength(theta_deg, card, mode=None):
    """Uniaxial strength (63.2 %, unit volume) of a bar loaded at theta to the layer plane, unidirectional ply along
    the projected load. sigma_F = s cos^2, sigma_L from s33 = s sin^2, s13 = s sin cos. G2's analytic reference."""
    t = np.radians(np.asarray(theta_deg, float))
    c, s = np.cos(t), np.sin(t)
    m, V0 = float(card["m"]), float(card.get("V0_mm3", 1.0))
    h = np.zeros_like(t)
    for k, p in _card_modes(card).items():
        if mode and k != mode:
            continue
        r = float(p.get("shear_ratio", 0.0 if k == "F" else 1.0))
        u = {"F": np.sqrt((c * c) ** 2 + (r * s * c) ** 2), "T": np.zeros_like(t),
             "L": np.sqrt((s * s) ** 2 + (r * s * c) ** 2), "P": np.ones_like(t)}[k]
        h = h + (u / float(p["S0_MPa"])) ** m / V0
    with np.errstate(divide="ignore"):
        return h ** (-1.0 / m)


# ------------------------------------------------------------------------------------------- cards
def load_card(card_id, path=None):
    """An orientation card from data/orientation_cards.json; "base" + "weibull_modes_add" extend another card."""
    import copy
    import json
    import os
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "orientation_cards.json")
    cards = json.load(open(path))
    if card_id not in cards or card_id.startswith("_"):
        raise KeyError(f"no orientation card {card_id!r}; have {sorted(k for k in cards if not k.startswith('_'))}")
    c = copy.deepcopy(cards[card_id])
    if "base" in c:
        b = load_card(c["base"], path)
        b["weibull"]["modes"].update(c.get("weibull_modes_add", {}))
        b.update({k: v for k, v in c.items() if k not in ("base", "weibull_modes_add")})
        c = b
    return c


def parse_layup(text):
    """"45/-45" or "0,90" -> (45.0, -45.0); one angle theta -> the cross-ply (theta, theta - 90)."""
    a = tuple(float(x) for x in str(text).replace(",", "/").split("/") if x.strip())
    return a if len(a) > 1 else (a[0], a[0] - 90.0)
