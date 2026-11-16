"""Ezeh-Susmel driver: fit own-lab plain + sharp V, hold out U r1 / U r3. Usage:
    python -m es2020.bazant_es_run <gauge_mm> [gamma1|nu1] [l0]
Writes es2020/data/bazant_es_<tag>.json (constants, scores) and the chain .npz."""
import sys, json, math, time, os
import numpy as np, emcee
from scipy.optimize import minimize
from scipy.special import expit
from . import sde, bazant as B

gauge = float(sys.argv[1]); var = sys.argv[2] if len(sys.argv) > 2 else "base"; l0 = float(sys.argv[3]) if len(sys.argv) > 3 else B.L0
tag = f"g{gauge:g}_{var}_l{l0:g}"
HERE = os.path.dirname(os.path.abspath(__file__))
rows = sde.dataset()
match = json.load(open(os.path.join(HERE, "review", "fig1_source_match.json")))
plain = [r for r in rows if r["notch"] == "plain"]
own = B.es_plain_own(rows, match)
plain_own = [r for r, o in zip(plain, own) if o]
fitrows = plain_own + [r for r in rows if r["notch"] == "sharp"]
allrows = plain_own + [r for r in rows if r["notch"] != "plain"]
bins, W, prof = B.es_inputs(gauge)
kw = dict(l0=l0, gamma_fixed=1.0 if var == "gamma1" else None, nu_one=(var == "nu1"))
fit = B.Chain(fitrows, bins, W, prof, **kw)
lo, hi = fit._box()
lp = fit.loglike
best = None; t0 = time.time()
for a in (0.05, 0.3, 1.0):
    for n0 in (3.5, 6.0):
        for lpg in (-6.0, -3.0):
            x0 = np.array([math.log(60), n0, 0.45, math.log(a), math.log(0.4), lpg])
            if var == "gamma1": x0 = np.delete(x0, 2)
            # scan sigma_D for a finite start
            sds = np.linspace(math.log(5), math.log(5e4), 60); vals = []
            for v in sds:
                x0[0] = v; vals.append(lp(x0))
            x0[0] = sds[int(np.argmax(vals))]
            o = minimize(lambda z: -lp(z), x0, method="Nelder-Mead", options=dict(maxiter=8000, maxfev=8000, xatol=1e-6, fatol=1e-8))
            if best is None or o.fun < best.fun: best = o
print(tag, "MAP nll", round(best.fun, 2), np.round(best.x, 3), "t", round(time.time() - t0), flush=True)
nd = len(best.x); nw = 4 * nd; rng = np.random.default_rng(0)
p0 = np.clip(best.x + 1e-3 * (hi - lo) * rng.standard_normal((nw, nd)), lo + 1e-9, hi - 1e-9)
smp = emcee.EnsembleSampler(nw, nd, lp, moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)])
smp.run_mcmc(p0, 3000, progress=False)
ch = smp.get_chain()[1500:].reshape(-1, nd)
np.savez(os.path.join(HERE, "data", f"bazant_es_{tag}.npz"), chain=ch, map=best.x)
full = np.array([fit.unpack(c) for c in ch])
q = np.percentile(full, [5, 50, 95], axis=0)
tr = [math.exp, float, float, math.exp, math.exp, expit]
post = {nm: [tr[i](v) for v in q[:, i]] for i, nm in enumerate(B.NAMES)}
mtail = [B.tail_modulus(math.exp(c[4]), expit(c[5])) for c in full]
mbar = [mt / (c[1] - 2) for mt, c in zip(mtail, full)]
post["m_tail"] = list(np.percentile(mtail, [5, 50, 95])); post["mbar_life"] = list(np.percentile(mbar, [5, 50, 95]))
print(tag, "post", {k: [round(x, 4) for x in v] for k, v in post.items()}, flush=True)
draws = ch[rng.choice(len(ch), 100, replace=False)]
m = B.Chain(allrows, bins, W, prof, **kw)
r = B.predictive_residual(m, draws)
logN = m.lnN / math.log(10); fail = ~m.ro
isfit = np.array([x["notch"] in ("plain", "sharp") for x in allrows])
out = dict(tag=tag, gauge=gauge, variant=var, l0=l0, map=list(best.x), map_nll=best.fun, post=post, n_fit=int(isfit.sum()),
           runtime_s=time.time() - t0, scores={})
for name, mask in [("plain", m.notch == "plain"), ("V", m.notch == "sharp"), ("U r1", m.notch == "intermediate"),
                   ("U r3", m.notch == "blunt"), ("FIT", isfit), ("HELD", ~isfit)]:
    sc = sde.score(r, mask & fail, logN); sc["below_P10"] = float(np.mean(r["F"][mask & fail] < 0.1))
    out["scores"][name] = sc
    print(f"  {tag} {name:6s} n {sc['n']:3d} rms {sc['rms']:.3f} bias {sc['bias']:+.3f} slope/dec {sc['slope']:+.3f}{' TILTED' if sc['tilted'] else ''} cover {sc['coverage']*100:.0f}% <P10 {sc['below_P10']*100:.0f}%", flush=True)
for Rr in sorted(set(m.R)):
    mk = (~isfit) & fail & (m.R == Rr)
    if mk.any(): out["scores"][f"HELD R{Rr:+g}"] = dict(n=int(mk.sum()), bias=float(np.mean(r["dlogS"][mk])))
np.savez(os.path.join(HERE, "data", f"bazant_es_{tag}_resid.npz"), dlogS=r["dlogS"], F=r["F"], notch=m.notch, R=m.R, s=m.s, lnN=m.lnN, ro=m.ro)
json.dump(out, open(os.path.join(HERE, "data", f"bazant_es_{tag}.json"), "w"), indent=1)
print(tag, "done", round(time.time() - t0), "s")
