"""Figures and table for the XIGABEM reproduction (python/xigabem_harness.py output, results/xigabem/<case>/).

penny: (a) K_I along the front / exact, (b) K_I error vs dof next to the paper's Table 1, (c) relative L2 error of
the crack opening vs dof, XIGABEM vs plain IgBEM, (d) opening profile near the front. Writes penny.png/.pdf and
penny_table.md next to case.json.

usage: plot_xigabem.py [case ...]
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(ROOT, "results", "xigabem")
C = {"ogrid": "#2a78d6", "disk": "#eb6834", "paper": "#1a1a19", "plain": "#8a8a85"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e4e0", "grid.linewidth": 0.6, "lines.linewidth": 1.6, "lines.markersize": 5,
                     "legend.frameon": False, "savefig.dpi": 200})


def rd(path):
    with open(path) as f:
        return {k: np.array([float(r[k]) for r in rows]) for rows in [list(csv.DictReader(f))] for k in rows[0]}


def base(out, lay=None):
    """Runs of the shift-0.1, ungraded baseline (baseline_shift0.1.json), sorted by level."""
    path = os.path.join(out, "baseline_shift0.1.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        runs = json.load(f)["runs"]
    return sorted((r for r in runs if lay is None or r["layout"] == lay), key=lambda r: r["n"])


BASE = dict(color="#b5b5b0", ls=":", marker=".", lw=1.2)


def dofaxis(a):
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
    a.xaxis.set_major_locator(FixedLocator([400, 1000, 2000, 4000]))
    a.xaxis.set_major_formatter(ScalarFormatter())
    a.xaxis.set_minor_locator(NullLocator())
    a.set_xlim(330, 4500)


def penny(d, out):
    runs = d["runs"]
    sel = lambda f, lay: sorted((r for r in runs if r["formulation"] == f and r["layout"] == lay), key=lambda r: r["n"])
    fig, ax = plt.subplots(2, 2, figsize=(9.0, 5.6))
    # (a) K_I along the front
    a = ax[0, 0]
    xs = sel("xigabem", "ogrid")
    shades = ["#86b6ef", "#3987e5", "#1c5cab"]
    for r, c in zip([xs[0], xs[len(xs) // 2], xs[-1]], shades):
        s = rd(os.path.join(out, r["tag"] + "_sif.csv"))
        th = np.degrees(np.mod(s["theta_c"], 2 * np.pi))
        o = np.argsort(th)
        a.plot(th[o], s["KI"][o] / s["KI_exact"][o], color=c, label="n = %d (%d dof)" % (r["n"], r["unknowns"]))
    a.axhline(1.0, color=C["paper"], lw=0.8, ls="--")
    a.set_xlabel(r"front angle $\theta$ [deg]")
    a.set_ylabel(r"$K_I / K_I^{exact}$")
    a.set_xticks([0, 90, 180, 270, 360])
    a.set_title("(a) $K_I$ along the front, O-grid", loc="left")
    a.legend(fontsize=7)
    # (b) K_I error vs dof
    b = ax[0, 1]
    for lay in ("ogrid", "disk"):
        xs = sel("xigabem", lay)
        b.loglog([r["unknowns"] for r in xs], [r["KI_avg_err"] for r in xs], "o-", color=C[lay],
                 label="this code, %s layout" % lay)
    xs = sel("xigabem", "disk_graded")
    if xs:
        b.loglog([r["unknowns"] for r in xs], [r["KI_avg_err"] for r in xs], "D--", color=C["disk"], mfc="white",
                 ms=4, lw=1.0, label="disk, front graded")
    bs = base(out, "disk")
    if bs:
        b.loglog([r["unknowns"] for r in bs], [r["KI_avg_err"] for r in bs], label="disk, end shift 0.1 (before)", **BASE)
    # independent K: interaction integral on the interior field (enriched, and plain IgBEM with no K unknowns)
    for f, mk, lab in (("xigabem", "v", "M-integral, XIGABEM field"), ("igabem", "^", "M-integral, plain IgBEM")):
        xs = [r for r in sel(f, "disk") if r.get("KM_I_avg_err", -1) >= 0]
        if xs:
            b.loglog([r["unknowns"] for r in xs], [r["KM_I_avg_err"] for r in xs], ":", marker=mk, color=C["disk"],
                     mfc="white", ms=4, lw=1.0, label=lab)
    p = d["paper"]["table1_KI_signed_avg_err"]
    b.loglog([int(k) for k in p], [abs(v) for v in p.values()], "s--", color=C["paper"], mfc="white",
             label="paper, Table 1")
    dofaxis(b)
    b.set_xlabel("degrees of freedom")
    b.set_ylabel(r"mean $|K_I - K_I^{exact}| / K_I^{exact}$")
    b.set_title("(b) $K_I$ error (XIGABEM)", loc="left")
    b.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    # (c) COD L2 vs dof
    c = ax[1, 0]
    for lay in ("ogrid", "disk"):
        for f, ls, mk in (("xigabem", "-", "o"), ("igabem", ":", "^")):
            xs = sel(f, lay)
            c.loglog([r["unknowns"] for r in xs], [r["cod_L2"] for r in xs], ls, marker=mk, color=C[lay],
                     label="%s, %s" % ("XIGABEM" if f == "xigabem" else "IgBEM", lay))
    dofaxis(c)
    c.set_ylim(1e-4, 2e-1)
    c.set_xlabel("degrees of freedom")
    c.set_ylabel("relative $L_2$ error of the opening")
    c.set_title("(c) crack opening: enriched vs plain", loc="left")
    c.legend(fontsize=7, ncol=2)
    # (d) opening profile near the front, coarse O-grid
    e = ax[1, 1]
    n0 = sel("xigabem", "ogrid")[1]["n"]
    for f, col, lab in (("xigabem", C["ogrid"], "XIGABEM"), ("igabem", C["plain"], "IgBEM")):
        s = rd(os.path.join(out, "%s_ogrid_n%d_cod.csv" % (f, n0)))
        e.plot(s["rho"], s["cod_n"], color=col, label="%s, n = %d" % (lab, n0))
    e.plot(s["rho"], s["cod_exact"], "--", color=C["paper"], lw=1.0, label="exact")
    e.set_xlim(0.8, 1.0)
    e.set_ylim(0, None)
    e.set_xlabel(r"radius $\rho / R$")
    e.set_ylabel(r"opening $\Delta u_n$ [$\sigma R / E$]")
    e.set_title("(d) opening near the front", loc="left")
    e.legend(fontsize=7)
    fig.suptitle("Penny-shaped crack under remote tension (Rocha et al. 2024, Sec. 4.1), $\\nu = 0$", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, "penny." + ext))
    # table
    rows = ["| layout | n | dof (XIG) | K_I mean err | K_I max err | K_I signed | paper signed (dof) | K_M err XIG | K_M err IgBEM | COD L2 XIG | COD L2 IgBEM | XIG time [s] |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    km = lambda r: "%.2e" % r["KM_I_avg_err"] if r and r.get("KM_I_avg_err", -1) >= 0 else "-"
    pk = {int(k): v for k, v in p.items()}
    for lay in ("ogrid", "disk", "disk_graded"):
        pl = {r["n"]: r for r in sel("igabem", lay)}
        for r in sel("xigabem", lay):
            near = min(pk, key=lambda k: abs(k - r["unknowns"]))
            ps = "%.2e (%d)" % (pk[near], near) if abs(near - r["unknowns"]) < 15 else "-"
            rows.append("| %s | %d | %d | %.2e | %.2e | %+.2e | %s | %s | %s | %.2e | %s | %.1f |" % (
                lay, r["n"], r["unknowns"], r["KI_avg_err"], r["KI_max_err"], r["KI_signed_err"], ps, km(r),
                km(pl.get(r["n"])), r["cod_L2"], "%.2e" % pl[r["n"]]["cod_L2"] if r["n"] in pl else "-",
                r["t_assemble"] + r["t_solve"]))
    for r in base(out, "disk"):
        rows.append("| disk, shift 0.1 (before) | %d | %d | %.2e | %.2e | %+.2e | | | | %.2e | | %.1f |" % (
            r["n"], r["unknowns"], r["KI_avg_err"], r["KI_max_err"], r["KI_signed_err"], r["cod_L2"],
            r["t_assemble"] + r["t_solve"]))
    with open(os.path.join(out, "penny_table.md"), "w") as f:
        f.write("\n".join(rows) + "\n")
    print("\n".join(rows))


def penny_inclined(d, out):
    runs = sorted((r for r in d["runs"] if r["formulation"] == "xigabem"), key=lambda r: r["n"])
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    a = ax[0]
    fin = runs[-1]
    s = rd(os.path.join(out, fin["tag"] + "_sif.csv"))
    th = np.degrees(np.mod(s["theta_c"], 2 * np.pi))
    o = np.argsort(th)
    norm = s["KI_exact"].max()
    for k, col in (("KI", "#2a78d6"), ("KII", "#eb6834"), ("KIII", "#1baf7a")):
        a.plot(th[o], s[k + "_exact"][o] / norm, "-", color=col, lw=1.0, alpha=0.6)
        a.plot(th[o][::4], s[k][o][::4] / norm, "o", color=col, ms=3.5, mfc="white",
               label=r"$K_{%s}$" % k[1:])
    a.plot([], [], "-", color="#8a8a85", lw=1.0, label="exact")
    a.set_xticks([0, 90, 180, 270, 360])
    a.set_xlabel(r"front angle $\theta$ [deg]")
    a.set_ylabel(r"$K / K_I^{exact}$")
    a.set_ylim(-1.5, 1.15)
    a.set_title("(a) SIFs along the front, n = %d (%d dof)" % (fin["n"], fin["unknowns"]), loc="left")
    a.legend(fontsize=7, ncol=4, loc="lower center")
    b = ax[1]
    dof = [r["unknowns"] for r in runs]
    for k, col, lab in (("KI_avg_err", "#2a78d6", r"$K_I$"), ("KII_avg_err", "#eb6834", r"$K_{II}$"),
                        ("KIII_avg_err", "#1baf7a", r"$K_{III}$")):
        b.loglog(dof, [r[k] for r in runs], "o-" if k != "KIII_avg_err" else "^:", color=col, label=lab + " mean")
    b.loglog(dof, [r["KI_max_err"] for r in runs], "--", color="#8a8a85", lw=1.0, label=r"$K_I$ max")
    bs = base(out)
    if bs:
        b.loglog([r["unknowns"] for r in bs], [r["KI_avg_err"] for r in bs], label=r"$K_I$ mean, shift 0.1 (before)",
                 **BASE)
    b.loglog([d["paper"]["finest_dof"]], [d["paper"]["max_rel_err_finest"]], "s", color="#1a1a19", mfc="white",
             label="paper, max (finest)")
    dofaxis(b)
    b.set_xlabel("degrees of freedom")
    b.set_ylabel("relative error")
    b.set_title("(b) SIF error vs dof", loc="left")
    b.legend(fontsize=7)
    fig.suptitle(r"Penny crack inclined by $\pi/4$ (Rocha et al. 2024, Fig. 12), $\nu = 0$", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, "penny_inclined." + ext))
    rows = ["| n | dof | K_I mean | K_II mean | K_III mean | K_I max | time [s] |", "|---|---|---|---|---|---|---|"]
    for r in runs:
        rows.append("| %d | %d | %.2e | %.2e | %.2e | %.2e | %.1f |" % (r["n"], r["unknowns"], r["KI_avg_err"],
                    r["KII_avg_err"], r["KIII_avg_err"], r["KI_max_err"], r["t_assemble"] + r["t_solve"]))
    rows.append("| paper | 1626 | | | | 2.03e-03 (all SIFs) | |")
    with open(os.path.join(out, "penny_inclined_table.md"), "w") as f:
        f.write("\n".join(rows) + "\n")
    print("\n".join(rows))


def prism(d, out):
    runs = sorted(d["runs"], key=lambda r: r["n"])
    modes = ("I", "II", "III")
    col = {"I": "#2a78d6", "II": "#eb6834", "III": "#1baf7a"}
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7))
    a = ax[0]
    for r, ls in ((runs[1], ":"), (runs[-1], "-")):
        s = rd(os.path.join(out, r["tag"] + "_sif.csv"))
        for i, m in enumerate(modes):
            sel = s["mode"] == i + 1
            k = s[["KI", "KII", "KIII"][i]][sel]
            a.plot(s["z"][sel], (k - 1.0) * 1e4, ls, color=col[m], label=r"$K_{%s}$" % m if r is runs[-1] else None)
    a.plot([], [], ":", color="#8a8a85", label="%d dof" % runs[1]["unknowns"])
    a.plot([], [], "-", color="#8a8a85", label="%d dof" % runs[-1]["unknowns"])
    a.set_xlabel("$z$ along the front")
    a.set_ylabel(r"$(K_M - 1) \times 10^4$")
    a.set_title("(a) pure-mode SIF along the front", loc="left", fontsize=8.5)
    a.legend(fontsize=6, ncol=2, loc="lower center")
    b = ax[1]
    dof = [r["unknowns"] for r in runs]
    for i, m in enumerate(modes):
        b.loglog(dof, [r["K%s_err" % m] for r in runs], "o-", color=col[m], label=r"$K_{%s}$" % m)
        pt = d["paper"]["table3"][m]
        b.loglog([int(k) for k in pt], [abs(v[i] - 1.0) for v in pt.values()], "s--", color=col[m], mfc="white",
                 lw=1.0, label=None)
    b.plot([], [], "s--", color="#1a1a19", mfc="white", lw=1.0, label="paper, Table 3")
    b.loglog(dof, [r["cross_max"] for r in runs], "^:", color="#8a8a85", label="largest spurious $K$")
    bs = base(out)
    if bs:
        b.loglog([r["unknowns"] for r in bs], [r["KI_err"] for r in bs], label=r"$K_I$, shift 0.1 (before)", **BASE)
    dofaxis(b)
    b.set_xlim(330, 5000)
    b.set_xlabel("degrees of freedom")
    b.set_ylabel(r"mean $|K_M - 1|$")
    b.set_title("(b) SIF error", loc="left", fontsize=8.5)
    b.legend(fontsize=5.5, loc="lower left")
    c = ax[2]
    for m in modes:
        c.loglog(dof, [r["mode" + m]["u_L2"] for r in runs], "o-", color=col[m], label="mode " + m)
    dofaxis(c)
    c.set_xlim(330, 5000)
    c.set_xlabel("degrees of freedom")
    c.set_ylabel("relative $L_2$ error of $u$")
    c.set_title("(c) displacement error", loc="left", fontsize=8.5)
    c.legend(fontsize=6)
    fig.suptitle("Prism with edge crack, pure modes (Rocha et al. 2024, Sec. 4.3), $\\nu = 0$", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, "prism." + ext))
    rows = ["| k | dof | mean abs(K_I - 1), mode I | abs(K_II - 1), mode II | abs(K_III - 1), mode III | largest spurious K | u L2 (I / II / III) | time [s] |",
            "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        rows.append("| %d | %d | %.2e | %.2e | %.2e | %.2e | %.1e / %.1e / %.1e | %.1f |" % (
            r["n"], r["unknowns"], r["KI_err"], r["KII_err"], r["KIII_err"], r["cross_max"], r["modeI"]["u_L2"],
            r["modeII"]["u_L2"], r["modeIII"]["u_L2"], r["t_assemble"] + r["t_solve"]))
    for dofp in (639, 1275, 2175):
        t = d["paper"]["table3"]
        e = [abs(t[m][str(dofp)][i] - 1.0) for i, m in enumerate(modes)]
        cr = max(t[m][str(dofp)][j] for i, m in enumerate(modes) for j in range(3) if j != i)
        rows.append("| paper | %d | %.2e | %.2e | %.2e | %.2e | | |" % (dofp, e[0], e[1], e[2], cr))
    with open(os.path.join(out, "prism_table.md"), "w") as f:
        f.write("\n".join(rows) + "\n")
    print("\n".join(rows))


def ellipse(d, out):
    """Both ellipse cases on one figure (reads ellipse_inclined/case.json too)."""
    with open(os.path.join(OUT, "ellipse_inclined", "case.json")) as f:
        di = json.load(f)
    oi = os.path.join(OUT, "ellipse_inclined")
    xs = sorted((r for r in d["runs"] if r["formulation"] == "xigabem" and r["layout"] == "disk"), key=lambda r: r["n"])
    xg = sorted((r for r in d["runs"] if r["layout"] == "disk_graded"), key=lambda r: r["n"])
    ps = sorted((r for r in d["runs"] if r["formulation"] == "igabem"), key=lambda r: r["n"])
    ri = sorted(di["runs"], key=lambda r: r["n"])
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.6))
    def front(path):
        s = rd(path)
        th = np.degrees(np.mod(s["theta_c"], 2 * np.pi))
        o = np.argsort(th)
        return th[o], {k: s[k][o] for k in s}
    a = ax[0, 0]
    for r, c in ((xs[0], "#86b6ef"), (xs[-1], "#1c5cab")):
        th, s = front(os.path.join(out, r["tag"] + "_sif.csv"))
        a.plot(th[::3], s["KI"][::3], "o", ms=3, mfc="white", color=c, label="n = %d (%d dof)" % (r["n"], r["unknowns"]))
    a.plot(th, s["KI_exact"], "-", color="#1a1a19", lw=0.9, label="Irwin")
    a.set_xticks([0, 90, 180, 270, 360])
    a.set_xlabel(r"parametric angle $\theta_c$ [deg]")
    a.set_ylabel(r"$K_I / (\sigma\sqrt{a})$")
    a.set_title("(a) in plane: $K_I$ along the front", loc="left")
    a.legend(fontsize=7)
    b = ax[0, 1]
    th, s = front(os.path.join(oi, ri[-1]["tag"] + "_sif.csv"))
    for k, c in (("KI", "#2a78d6"), ("KII", "#eb6834"), ("KIII", "#1baf7a")):
        b.plot(th, s[k + "_exact"], "-", color=c, lw=0.9, alpha=0.7)
        b.plot(th[::3], s[k][::3], "o", ms=3, mfc="white", color=c, label=r"$K_{%s}$" % k[1:])
    b.plot([], [], "-", color="#8a8a85", lw=0.9, label="Kassir & Sih")
    b.set_xticks([0, 90, 180, 270, 360])
    b.set_xlabel(r"parametric angle $\theta_c$ [deg]")
    b.set_title(r"(b) inclined $\pi/6$, n = %d (%d dof)" % (ri[-1]["n"], ri[-1]["unknowns"]), loc="left")
    b.set_ylim(-0.5, 1.15)
    b.legend(fontsize=6.5, ncol=4, loc="upper center")
    c = ax[1, 0]
    dof = [r["unknowns"] for r in xs]
    c.loglog(dof, [r["KI_avg_err"] for r in xs], "o-", color="#2a78d6", label=r"in plane, mean $K_I$")
    if xg:
        c.loglog([r["unknowns"] for r in xg], [r["KI_avg_err"] for r in xg], "D--", color="#2a78d6", mfc="white",
                 ms=4, lw=1.0, label="in plane, front graded")
    bs = base(out)
    if bs:
        c.loglog([r["unknowns"] for r in bs], [r["KI_avg_err"] for r in bs], label="in plane, shift 0.1 (before)",
                 **BASE)
    lab = {"t2_KI_0": (r"$K_I$, $\theta_c = 0$", "#2a78d6"), "t2_KII_0": (r"$K_{II}$, $\theta_c = 0$", "#eb6834"),
           "t2_KI_90": (r"$K_I$, $\theta_c = \pi/2$", "#6f47d6"), "t2_KIII_90": (r"$K_{III}$, $\theta_c = \pi/2$", "#1baf7a")}
    for k, (l, col) in lab.items():
        c.loglog([r["unknowns"] for r in ri], [r[k] for r in ri], "^:", color=col, label="inclined, " + l)
        c.axhline(di["paper"]["table2"][k][0], color=col, lw=0.8, ls="--")
    c.plot([], [], "--", color="#8a8a85", lw=0.8, label="paper Table 2 (dof not given)")
    dofaxis(c)
    c.set_xlabel("degrees of freedom")
    c.set_ylabel("relative SIF error")
    c.set_title("(c) SIF error", loc="left")
    c.set_ylim(3e-4, 3.0)
    c.legend(fontsize=5.5, loc="upper right", ncol=2)
    e = ax[1, 1]
    e.loglog(dof, [r["cod_L2"] for r in xs], "o-", color="#2a78d6", label="XIGABEM")
    e.loglog([r["unknowns"] for r in ps], [r["cod_L2"] for r in ps], "^:", color="#8a8a85", label="IgBEM")
    dofaxis(e)
    e.set_xlabel("degrees of freedom")
    e.set_ylabel("relative $L_2$ error of the opening")
    e.set_title("(d) in plane: crack opening", loc="left")
    e.legend(fontsize=7)
    fig.suptitle(r"Elliptical crack $a = 2b$, $\nu = 0.3$ (Rocha et al. 2024, Sec. 4.2)", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, "ellipse." + ext))
    rows = ["| n | dof | in plane: mean K_I err | shift 0.1 (before) | graded (dof) | opening L2 XIG / IgBEM | inclined: K_I(0) | K_II(0) | K_I(pi/2) | K_III(pi/2) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    pl = {r["n"]: r for r in ps}
    ii = {r["n"]: r for r in ri}
    bb = {r["n"]: r for r in bs}
    gg = {r["n"]: r for r in xg}
    for r in xs:
        q = ii.get(r["n"])
        rows.append("| %d | %d | %.2e | %s | %s | %.2e / %.2e | %s |" % (
            r["n"], r["unknowns"], r["KI_avg_err"],
            "%.2e" % bb[r["n"]]["KI_avg_err"] if r["n"] in bb else "-",
            "%.2e (%d)" % (gg[r["n"]]["KI_avg_err"], gg[r["n"]]["unknowns"]) if r["n"] in gg else "-",
            r["cod_L2"], pl[r["n"]]["cod_L2"], " | ".join("%.2e" % q[k] for k in lab) if q else "| | |"))
    t = di["paper"]["table2"]
    for i, name in enumerate(["paper XIGABEM", "paper VCCI [72]", "paper M-integral [72]"]):
        rows.append("| %s | | | | | | %s |" % (name, " | ".join("%.2e" % t[k][i] for k in lab)))
    with open(os.path.join(out, "ellipse_table.md"), "w") as f:
        f.write("\n".join(rows) + "\n")
    print("\n".join(rows))


def prism_tension(d, out):
    runs = d["runs"]
    lays = (("split", "#2a78d6", "lateral faces split at the crack, enriched"),
            ("aligned", "#eb6834", "lateral faces aligned with crack and front"),
            ("enriched", "#1baf7a", "one enriched patch per lateral face"),
            ("split_graded", "#6f47d6", "split, front graded"))
    F = d["paper"]["F_2d"]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
    a = ax[0]
    for lay, col, lab in lays:
        rs = sorted((r for r in runs if r["layout"] == lay), key=lambda r: r["n"])
        r = rs[-1]
        s = rd(os.path.join(out, r["tag"] + "_sif.csv"))
        a.plot(s["z"], s["KI"], "-", color=col, label="%s, %d dof" % (lay, r["unknowns"]))
    a.axhline(F, color="#1a1a19", lw=0.9, ls="--", label="2-D plane strain (Eq. 30)")
    a.set_xlabel("$z$ along the front")
    a.set_ylabel(r"$K_I / (\sigma_0 \sqrt{\pi a})$")
    a.set_ylim(2.0, 3.2)
    a.set_xticks([0, 1.5, 3, 4.5, 6, 7.5])
    a.set_title("(a) $K_I$ along the front (cf. paper Fig. 25)", loc="left", fontsize=8.5)
    a.legend(fontsize=6, loc="lower center")
    b = ax[1]
    for lay, col, lab in lays:
        rs = sorted((r for r in runs if r["layout"] == lay), key=lambda r: r["n"])
        b.semilogx([r["unknowns"] for r in rs], [r["KI_mid"] for r in rs], "o-", color=col, label=lay)
    bs = base(out, "split")
    if bs:
        b.semilogx([r["unknowns"] for r in bs], [r["KI_mid"] for r in bs], label="split, shift 0.1 (before)", **BASE)
    b.axhline(F, color="#1a1a19", lw=0.9, ls="--", label="2-D plane strain")
    b.set_xlabel("degrees of freedom")
    b.set_ylabel(r"$K_I / (\sigma_0 \sqrt{\pi a})$ at $z = 3.75$")
    b.set_title("(b) mid-front $K_I$ under refinement", loc="left", fontsize=8.5)
    b.legend(fontsize=6)
    fig.suptitle(r"Prism with edge crack, uniform tension, $a/b = 0.5$, $\nu = 0.3$ (Rocha et al. 2024, Sec. 4.4)",
                 fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, "prism_tension." + ext))
    rows = ["| lateral layout | crack refinement | dof | K_I mid / sigma sqrt(pi a) | diff vs 2-D | time [s] |",
            "|---|---|---|---|---|---|"]
    for lay, _, _ in lays:
        for r in sorted((r for r in runs if r["layout"] == lay), key=lambda r: r["n"]):
            rows.append("| %s | %d | %d | %.4f | %+.2f%% | %.1f |" % (lay, r["n"], r["unknowns"], r["KI_mid"],
                        100 * (r["KI_mid"] / F - 1), r["t_assemble"] + r["t_solve"]))
    for r in base(out, "split"):
        rows.append("| split, shift 0.1 (before) | %d | %d | %.4f | %+.2f%% | %.1f |" % (
            r["n"], r["unknowns"], r["KI_mid"], 100 * (r["KI_mid"] / F - 1), r["t_assemble"] + r["t_solve"]))
    for k, v in d["paper"]["fig25_mid_diff"].items():
        rows.append("| paper %s | | | | %.2f%% (sign not stated) | |" % (k, 100 * v))
    with open(os.path.join(out, "prism_tension_table.md"), "w") as f:
        f.write("\n".join(rows) + "\n")
    print("\n".join(rows))


def prism_a3(d, out):
    """Mode-I prism with the r^3/2 Williams term: how the K unknown absorbs the remainder, and the front grading."""
    runs = d["runs"]
    lays = (("split", "#eb6834", "uniform"), ("split_graded", "#2a78d6", "front graded"))
    sel = lambda lay: sorted((r for r in runs if r["layout"] == lay), key=lambda r: r["n"])
    bs = base(out)
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7))
    a = ax[0]
    for lay, col, lab in lays:
        r = sel(lay)[-1]
        s = rd(os.path.join(out, r["tag"] + "_sif.csv"))
        m = s["mode"] == 1
        a.plot(s["z"][m], (s["KI"][m] - 1.0) * 100, "-", color=col, label="%s, %d dof" % (lab, r["unknowns"]))
    a.axhline(0.0, color="#1a1a19", lw=0.8, ls="--")
    a.set_xlabel("$z$ along the front")
    a.set_ylabel(r"$K_I - 1$ [%]")
    a.set_title("(a) $K_I$ along the front", loc="left", fontsize=8.5)
    a.legend(fontsize=6)
    b = ax[1]
    for lay, col, lab in lays:
        rs = sel(lay)
        b.loglog([r["unknowns"] for r in rs], [r["KI_err"] for r in rs], "o-", color=col, label=lab + ", shift 0.02")
    if bs:
        b.loglog([r["unknowns"] for r in bs], [r["KI_err"] for r in bs], label="uniform, shift 0.1 (before)", **BASE)
    for lay, col, lab in lays:  # interaction integral: its own error, and the gap it reports for K_dof
        rs = [r for r in sel(lay) if "KM_err" in r and r["KM_err"][0] >= 0]
        if rs:
            b.loglog([r["unknowns"] for r in rs], [r["KM_err"][0] for r in rs], "v:", color=col, mfc="white", ms=4,
                     lw=1.0, label=lab + ", M-integral")
            b.loglog([r["unknowns"] for r in rs], [r["KM_gap"][0] for r in rs], "x", color=col, ms=5,
                     label=lab + r", $|K_{dof} - K_M|$")
    dofaxis(b)
    b.set_xlim(330, 5000)
    b.set_xlabel("degrees of freedom")
    b.set_ylabel(r"mean $|K_I - 1|$")
    b.set_title("(b) $K_I$ error", loc="left", fontsize=8.5)
    b.legend(fontsize=5, loc="lower left", framealpha=0.9)
    c = ax[2]
    for lay, col, lab in lays:
        rs = sel(lay)
        c.loglog([r["unknowns"] for r in rs], [r["modeI"]["u_L2"] for r in rs], "o-", color=col, label=lab)
    dofaxis(c)
    c.set_xlim(330, 5000)
    c.set_xlabel("degrees of freedom")
    c.set_ylabel("relative $L_2$ error of $u$")
    c.set_title("(c) displacement error", loc="left", fontsize=8.5)
    c.legend(fontsize=6)
    fig.suptitle(r"Edge-cracked prism, mode I + $a_3 r^{3/2}$ term ($a_3 = 1$, exact $K_I = 1$), $\nu = 0$",
                 fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, "prism_a3." + ext))
    rows = ["| mesh | k | dof | mean abs(K_I - 1) | K_M err | abs(K_dof - K_M) at stations | true K_dof err at stations | track ratio | u L2 | time [s] |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for lay, _, lab in lays:
        for r in sel(lay):
            m = ("%.2e | %.2e | %.2e | %.2f" % (r["KM_err"][0], r["KM_gap"][0], r["Kdof_err_at_stations"][0],
                                               r["KM_track_I"])) if r.get("KM_track_I", -1) > 0 else "- | - | - | -"
            rows.append("| %s, shift 0.02 | %d | %d | %.2e | %s | %.2e | %.1f |" % (lab, r["n"], r["unknowns"],
                        r["KI_err"], m, r["modeI"]["u_L2"], r["t_assemble"] + r["t_solve"]))
    for r in bs:
        rows.append("| uniform, shift 0.1 (before) | %d | %d | %.2e | - | - | - | - | %.2e | %.1f |" % (
            r["n"], r["unknowns"], r["KI_err"], r["modeI"]["u_L2"], r["t_assemble"] + r["t_solve"]))
    with open(os.path.join(out, "prism_a3_table.md"), "w") as f:
        f.write("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    for name in sys.argv[1:] or ["penny"]:
        with open(os.path.join(OUT, name, "case.json")) as f:
            globals()[name](json.load(f), os.path.join(OUT, name))
