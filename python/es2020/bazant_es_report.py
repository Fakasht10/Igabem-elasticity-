"""Ezeh-Susmel report for the Bazant chain: S-N bands in the paper's variables, predicted vs observed life scatter,
score table. Reads the stored chains (data/bazant_es_<tag>.npz); no fitting. Usage:
    python -m es2020.bazant_es_report <outdir> <chain tag> <single-link tag> [more tags for the table ...]"""
import sys, json, math, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import sde, bazant as B

HERE = os.path.dirname(os.path.abspath(__file__))
C_CHAIN, C_ONE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
LAB = {"plain": "plain (own lab)", "sharp": "V (sharp)", "intermediate": "U r1", "blunt": "U r3"}


def load(tag):
    js = json.load(open(os.path.join(HERE, "data", f"bazant_es_{tag}.json")))
    ch = np.load(os.path.join(HERE, "data", f"bazant_es_{tag}.npz"))["chain"]
    return js, ch


def setup(js):
    rows = sde.dataset()
    match = json.load(open(os.path.join(HERE, "review", "fig1_source_match.json")))
    plain = [r for r in rows if r["notch"] == "plain"]
    own = B.es_plain_own(rows, match)
    allrows = [r for r, o in zip(plain, own) if o] + [r for r in rows if r["notch"] != "plain"]
    bins, W, prof = B.es_inputs(js["gauge"])
    kw = dict(l0=js["l0"], gamma_fixed=1.0 if js["variant"] == "gamma1" else None, nu_one=js["variant"] == "nu1")
    return allrows, B.Chain(allrows, bins, W, prof, **kw)


def series(rows):
    ser = collections.defaultdict(list)
    for r in rows:
        ser[(r["notch"], r["R"])].append(r)
    return ser


def sd_lnN_obs(rs):
    v = np.array([(math.log(r["s_max"]), math.log(r["N"])) for r in rs if not r["runout"]])
    A = np.c_[np.ones(len(v)), v[:, 0]]
    c, *_ = np.linalg.lstsq(A, v[:, 1], rcond=None)
    return float((v[:, 1] - A @ c).std(ddof=2)), float(np.exp(np.median(v[:, 0])))


def sd_lnN_pred(model, draws, kind, R, s):
    g = np.linspace(0, 40, 801)
    sub = B._subset(model, kind, R, np.full(len(g), s))
    F = B.mix_cdf(sub, draws, lnN=g)
    p = np.diff(F); x = 0.5 * (g[1:] + g[:-1]); p = p / p.sum()
    mu = (p * x).sum()
    return float(math.sqrt((p * (x - mu) ** 2).sum()))


def main(out, tag_chain, tag_one, extra):
    rng = np.random.default_rng(1)
    jc, chc = load(tag_chain); jo, cho = load(tag_one)
    rows, mc = setup(jc); _, mo = setup(jo)
    dc = chc[rng.choice(len(chc), 40, replace=False)]; do = cho[rng.choice(len(cho), 40, replace=False)]
    ser = series(rows)
    # ---------------------------------------------------------------- Fig 1: S-N bands, paper variables
    kinds = ("plain", "sharp", "intermediate", "blunt")
    fig, ax = plt.subplots(2, 4, figsize=(15, 7.2), sharex=True, sharey=True)
    for i, R in enumerate((-1.0, 0.0)):
        for j, k in enumerate(kinds):
            a = ax[i, j]; rs = ser[(k, R)]
            held = k in ("intermediate", "blunt")
            sv = np.geomspace(min(r["s_max"] for r in rs) * 0.8, max(r["s_max"] for r in rs) * 1.2, 24)
            for m, d, col, nm in ((mc, dc, C_CHAIN, "finite chain"), (mo, do, C_ONE, "single link")):
                q = B.band(m, d, k, R, sv)
                a.fill_betweenx(sv, q[0], q[2], color=col, alpha=0.15, lw=0)
                a.plot(q[1], sv, color=col, lw=2, label=f"{nm}: P50, band P10–P90")
            f = [r for r in rs if not r["runout"]]; o = [r for r in rs if r["runout"]]
            mk = dict(s=26, edgecolors=INK, linewidths=0.9)
            a.scatter([r["N"] for r in f], [r["s_max"] for r in f], facecolors="none" if held else INK,
                      label=("specimens, HELD OUT" if held else "specimens, fitted"), **mk)
            if o:
                a.scatter([r["N"] for r in o], [r["s_max"] for r in o], marker=">", facecolors="none", label="run-out", **mk)
            a.set_xscale("log"); a.set_yscale("log"); a.grid(True, which="major", color="#e4e3df", lw=0.6)
            a.set_title(f"{LAB[k]}, R = {R:g}" + ("  (held out)" if held else ""), fontsize=10, color=INK)
            if i == 1: a.set_xlabel("cycles to failure N_f")
            if j == 0: a.set_ylabel("σ_max [MPa] (net for notches)")
            if i == 0 and j == 3: a.legend(fontsize=7.5, loc="upper right", frameon=False)
    a.set_xlim(1e2, 1e7)
    fig.suptitle(f"Ezeh–Susmel 2020: Bažant chain ({tag_chain}) vs single link ({tag_one}); fit = own-lab plain + V, held out = U r1, U r3",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_bazant_es_sn.png"), dpi=200); fig.savefig(os.path.join(out, "fig_bazant_es_sn.pdf"))
    # ---------------------------------------------------------------- Fig 2: life scatter, predicted vs observed
    fig, a = plt.subplots(figsize=(6.2, 5.6))
    res = []
    for (k, R), rs in sorted(ser.items()):
        if sum(not r["runout"] for r in rs) < 6: continue
        so, s50 = sd_lnN_obs(rs)
        pc = sd_lnN_pred(mc, dc, k, R, s50); po = sd_lnN_pred(mo, do, k, R, s50)
        res.append(dict(series=f"{LAB[k]} R {R:g}", obs=so, chain=pc, single=po))
        a.scatter(so, pc, color=C_CHAIN, s=40, edgecolors="white", linewidths=1.5, zorder=3)
        a.scatter(so, po, color=C_ONE, s=40, marker="s", edgecolors="white", linewidths=1.5, zorder=3)
        a.annotate(f"{LAB[k]} R{R:g}", (so, pc), fontsize=7, color=MUTED, xytext=(4, 3 if R != 0 else -9), textcoords="offset points")
    lim = [0, 2.6]; a.plot(lim, lim, color=MUTED, lw=1, ls="--", label="1:1")
    a.scatter([], [], color=C_CHAIN, label="finite chain (predicted)"); a.scatter([], [], color=C_ONE, marker="s", label="single link (predicted)")
    a.set_xlim(lim); a.set_ylim(lim); a.set_xlabel("observed sd of ln N about the series Basquin line")
    a.set_ylabel("predicted sd of ln N at the series median stress"); a.legend(frameon=False, fontsize=8)
    a.set_title("Life scatter per series (every series with ≥ 6 failures)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_bazant_es_scatter.png"), dpi=200); fig.savefig(os.path.join(out, "fig_bazant_es_scatter.pdf"))
    # ---------------------------------------------------------------- table
    lines = ["| model | p | MAP nll (fit) | n | γ | m (tail) | m̄ = m/(n−2) | FIT rms / bias / tilt per dec / cover | HELD rms / bias / tilt per dec / cover / <P10 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for t in [tag_chain, tag_one] + list(extra):
        js, _ = load(t); p = js["post"]; f = js["scores"]["FIT"]; h = js["scores"]["HELD"]
        lines.append(f"| {t} | {len(js['map'])} | {js['map_nll']:.1f} | {p['n'][1]:.2f} | {p['gamma'][1]:.2f} | {p['m_tail'][1]:.1f} | {p['mbar_life'][1]:.2f} | "
                     f"{f['rms']:.3f} / {f['bias']:+.3f} / {f['slope']:+.3f} / {f['coverage']*100:.0f}% | "
                     f"{h['rms']:.3f} / {h['bias']:+.3f} / {h['slope']:+.3f} / {h['coverage']*100:.0f}% / {h['below_P10']*100:.0f}% |")
    lines += ["", "| series | observed sd ln N | chain | single link |", "| --- | --- | --- | --- |"]
    lines += [f"| {r['series']} | {r['obs']:.2f} | {r['chain']:.2f} | {r['single']:.2f} |" for r in res]
    open(os.path.join(out, "bazant_es_scores.md"), "w").write("\n".join(lines) + "\n")
    json.dump(res, open(os.path.join(out, "bazant_es_scatter.json"), "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
