"""Stage times of the old and the fast dense path (results/dense_fast.csv from
results/run_dense_fast.sh) -> results/fig_dense_fast.{pdf,png} and a Markdown table on stdout."""
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "dense_fast.csv")
OUT = os.path.join(ROOT, "results", "fig_dense_fast")

rows = list(csv.DictReader(open(CSV)))
f = lambda r, k: float(r[k])


def stages(r):
    parts = ["t_mesh", "t_g", "t_h", "t_rigid", "t_lu", "t_solve", "t_post"]
    alloc = f(r, "t_total") - sum(f(r, k) for k in parts)
    return [
        ("Mesh + matrix allocation", f(r, "t_mesh") + alloc),
        ("Assembly (H, G or G·t)", f(r, "t_g") + f(r, "t_h") + f(r, "t_rigid")),
        ("Factorization (LU / block preconditioner)", f(r, "t_lu")),
        ("Solve (triangular / GMRES)", f(r, "t_solve")),
        ("Error evaluation", f(r, "t_post")),
    ]


COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # validated categorical order
NAMES = {
    "plate_hole_p4_t0.2_r3": "t = 0.2",
    "plate_hole_p4_t0.2_r4": "t = 0.2",
    "plate_hole_L3_p4_t0.2_r4": "L = 3, t = 0.2",
    "plate_hole_p4_t0.02_r4": "t = 0.02 (thin)",
}
meshes = []
for r in rows:
    m = r["label"].rsplit("_", 1)[0]
    if m not in meshes:
        meshes.append(m)
by = {(r["label"].rsplit("_", 1)[0], r["label"].rsplit("_", 1)[1]): r for r in rows}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7.6, 1.6 + 0.75 * len(meshes)))
ylabels, y = [], 0.0
yt = []
for m in meshes:
    for kind in ("old", "fast"):
        r = by.get((m, kind))
        if r is None:
            continue
        left = 0.0
        for (name, t), c in zip(stages(r), COLORS):
            ax.barh(y, t, left=left, height=0.62, color=c, edgecolor="#fcfcfb", linewidth=1.0,
                    label=name if y == 0 else None)
            left += t
        txt = "%.1f s" % f(r, "t_total")
        if kind == "fast" and (m, "old") in by:
            txt += "  (%.1f× faster)" % (f(by[(m, "old")], "t_total") / f(r, "t_total"))
        ax.text(left + 1.0, y, txt, va="center", fontsize=8, color="#0b0b0b")
        yt.append(y)
        ylabels.append("%s, %s unknowns — %s" % (NAMES.get(m, m), "{:,}".format(int(r["unknowns"])), kind))
        y += 0.8
    y += 0.5
ax.set_yticks(yt)
ax.set_yticklabels(ylabels, fontsize=8, color="#52514e")
ax.invert_yaxis()
ax.set_xlabel("wall time [s], 4 cores")
ax.grid(axis="x", color="#e4e3df", linewidth=0.6)
ax.set_axisbelow(True)
ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.16), ncol=3, fontsize=7.5, frameon=False)
ax.set_title("Dense plate-with-hole solve (p = 4, Kirsch field): old vs fast path", fontsize=10, loc="left")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT + "." + ext, dpi=200)

print("| Mesh | Unknowns | Path | Assembly [s] | Factor [s] | Solve [s] (iters) | Total [s] | Peak RSS [GB] | Kirsch error |")
print("|---|---|---|---|---|---|---|---|---|")
for m in meshes:
    for kind in ("old", "fast"):
        r = by.get((m, kind))
        if r is None:
            continue
        s = dict(stages(r))
        it = int(r["refinements"])
        print("| %s | %s | %s | %.1f | %.1f | %.1f%s | %.1f | %.1f | %.2e |" % (
            NAMES.get(m, m), "{:,}".format(int(r["unknowns"])), kind, s["Assembly (H, G or G·t)"],
            s["Factorization (LU / block preconditioner)"], s["Solve (triangular / GMRES)"],
            " (%d)" % it if kind == "fast" else "", f(r, "t_total"), f(r, "rss_MB") / 1024, f(r, "err_kirsch")))
