"""Yaren-Susmel driver: fit plain + V at all in-fills, hold out U r1 / U r3; a* = a_s + kappa_v dv.
Usage: python -m es2020.bazant_ys_run [base|gamma1|nu1] [l0]   (3D IgBEM surface fields, bazant.ys_inputs3d)."""
import sys, json, math, time, os
import numpy as np, emcee
from scipy.optimize import minimize
from scipy.special import expit
from . import sde, bazant as B

var = sys.argv[1] if len(sys.argv) > 1 else "base"; l0 = float(sys.argv[2]) if len(sys.argv) > 2 else B.L0
tag = f"{var}_l{l0:g}"
HERE = os.path.dirname(os.path.abspath(__file__))
rows = B.ys_rows()
isfit = np.array([r["notch"] in ("plain", "V_r0.15") for r in rows])
fitrows = [r for r, f in zip(rows, isfit) if f]
bins, W, prof = B.ys_inputs3d()
kw = dict(l0=l0, gamma_fixed=1.0 if var == "gamma1" else None, nu_one=(var == "nu1"))
fit = B.Chain(fitrows, bins, W, prof, dv=[r["dv"] for r in fitrows], **kw)
lo, hi = fit._box(); lp = fit.loglike
best = None; t0 = time.time()
for a in (0.01, 0.3):
    for kv in (0.3, 2.0):
        for n0 in (3.5, 6.0):
            for lpg in (-6.0, -3.0):
                x0 = np.array([math.log(30), n0, 0.5, math.log(a), math.log(kv), math.log(0.5), lpg])
                if var == "gamma1": x0 = np.delete(x0, 2)
                sds = np.linspace(math.log(2), math.log(5e4), 60); vals = []
                for v in sds:
                    x0[0] = v; vals.append(lp(x0))
                x0[0] = sds[int(np.argmax(vals))]
                o = minimize(lambda z: -lp(z), x0, method="Nelder-Mead", options=dict(maxiter=10000, maxfev=10000, xatol=1e-6, fatol=1e-8))
                if best is None or o.fun < best.fun: best = o
print(tag, "MAP nll", round(best.fun, 2), np.round(best.x, 3), "t", round(time.time() - t0), flush=True)
nd = len(best.x); nw = 4 * nd; rng = np.random.default_rng(0)
p0 = np.clip(best.x + 1e-3 * (hi - lo) * rng.standard_normal((nw, nd)), lo + 1e-9, hi - 1e-9)
smp = emcee.EnsembleSampler(nw, nd, lp, moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)])
smp.run_mcmc(p0, 3000, progress=False)
ch = smp.get_chain()[1500:].reshape(-1, nd)
np.savez(os.path.join(HERE, "data", f"bazant_ys_{tag}.npz"), chain=ch, map=best.x)
full = np.array([fit.unpack(c) for c in ch])
names = ("sigma_D", "n", "gamma", "a_s", "kappa_v", "s_lnr", "P_gr"); tr = [math.exp, float, float, math.exp, math.exp, math.exp, expit]
q = np.percentile(full, [5, 50, 95], axis=0)
post = {nm: [tr[i](v) for v in q[:, i]] for i, nm in enumerate(names)}
mtail = [B.tail_modulus(math.exp(c[5]), expit(c[6])) for c in full]
post["m_tail"] = list(np.percentile(mtail, [5, 50, 95])); post["mbar_life"] = list(np.percentile([mt / (c[1] - 2) for mt, c in zip(mtail, full)], [5, 50, 95]))
print(tag, "post", {k: [round(x, 4) for x in v] for k, v in post.items()}, flush=True)
draws = ch[rng.choice(len(ch), 100, replace=False)]
m = B.Chain(rows, bins, W, prof, dv=[r["dv"] for r in rows], **kw)
r = B.predictive_residual(m, draws)
logN = m.lnN / math.log(10); fail = ~m.ro; inf = np.array([x["infill"] for x in rows])
out = dict(tag=tag, variant=var, l0=l0, map=list(best.x), map_nll=best.fun, post=post, n_fit=int(isfit.sum()), scores={},
           fields="3D IgBEM surfaces (igabem_fields_3d.json, ramlnd a67e6c3)")
groups = [(g, m.notch == g) for g in B.YS_KINDS] + [("FIT", isfit), ("HELD", ~isfit)] + [(f"HELD {i}%", (~isfit) & (inf == i)) for i in (100, 80, 60, 40, 20)] \
         + [(f"FIT {i}%", isfit & (inf == i)) for i in (100, 80, 60, 40, 20)]
for name, mask in groups:
    sc = sde.score(r, mask & fail, logN); sc["below_P10"] = float(np.mean(r["F"][mask & fail] < 0.1)); out["scores"][name] = sc
    print(f"  {tag} {name:9s} n {sc['n']:3d} rms {sc['rms']:.3f} bias {sc['bias']:+.3f} slope/dec {sc['slope']:+.3f}{' TILTED' if sc['tilted'] else ''} cover {sc['coverage']*100:.0f}% <P10 {sc['below_P10']*100:.0f}%", flush=True)
np.savez(os.path.join(HERE, "data", f"bazant_ys_{tag}_resid.npz"), dlogS=r["dlogS"], F=r["F"], notch=m.notch, R=m.R, s=m.s, lnN=m.lnN, ro=m.ro, infill=inf)
json.dump(out, open(os.path.join(HERE, "data", f"bazant_ys_{tag}.json"), "w"), indent=1)
print(tag, "done", round(time.time() - t0), "s")
