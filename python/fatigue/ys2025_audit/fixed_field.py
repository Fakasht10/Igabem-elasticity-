"""Re-score after the shared-field fix in review/defects/rank.py (2026-11-16 audit):
  fields clipped at the ligament half-width b (no periodic mirror), 128-point Gauss weight-function mean, and in the crack rule
  a + L/2 < b required (net-section collapse otherwise); the finite-ligament factor Y_fw only with --fw (off by default).
YS: rank's crack rule on the 2D field; the 3D baseline (3D mid-plane field + plain hourglass field); the same + triaxiality R -1 term.
ES: rank's crack rule; reconcile.py's finite-width crack (the review's 0.075 row) on own-lab 125 and pooled 195 plain.
Scores, failures, fitted (plain + V) and held-out (U) separately: log-stress rms/bias (rank's units: log10 of S*/S), log-N rms/bias,
tilt per decade vs log N_exp AND vs log N_pred, coverage of the model P10-P90 band in log N (nominal 80 %).
One likelihood evaluation ~1 ms (profiled); run time is the Nelder-Mead iteration count."""
import sys, os, json, math, time, numpy as np
for p_ in ('/home/user/wt_uxm/python', '/mnt/project-files/review', '/mnt/project-files/review/defects'):
    sys.path.insert(0, p_)
import rank as RK
C = '/home/user/igabem-elasticity-/cache/ys2025/'; NM = {'plain': 'plain', 'V_r0.15': 'V', 'U_r1': 'U1', 'U_r3': 'U3'}
def stats(d, p, F):
    se = d.S * F * ((1 - d.R) / 2) ** p['gam']; ep = p['k'] * (p['lnsD'] - np.log(se)) / math.log(10)
    eN = d.lnN / math.log(10) - ep; eS = -eN / p['k']          # log10(S*/S), rank's sign; sZ = math.exp(p['lnsZ']) / math.log(10); m = ~d.ro; out = {}
    for nm_, msk in (('fit', d.fit & m), ('held', ~d.fit & m)):
        x, r = d.lnN[msk] / math.log(10), eN[msk]
        out[nm_] = dict(n=int(msk.sum()), rms_S=float(np.sqrt(np.mean(eS[msk] ** 2))), bias_S=float(eS[msk].mean()), rms_N=float(np.sqrt(np.mean(r ** 2))),
                        bias_N=float(r.mean()), tilt_exp=float(np.polyfit(x, r, 1)[0]), tilt_pred=float(np.polyfit(ep[msk], r, 1)[0]),
                        cov=float(100 * np.mean(np.abs(r) <= 1.2816 * sZ)))
    if d.name == 'YS':
        u1, u3 = (d.g == 'U1') & m, (d.g == 'U3') & m; out['U1_minus_U3_N'] = float(eN[u1].mean() - eN[u3].mean())
        inf = 100 - 20 * d.lev; out['U1_minus_U3_by_infill'] = {int(l): float(eN[u1 & (inf == l)].mean() - eN[u3 & (inf == l)].mean()) for l in (100, 80, 60, 40, 20)}
    return out
def ys3d(d, rm):
    P3 = {NM[k]: dict(np.load(C + f'profile_{k}_h0.npz')) for k in NM}; TX = np.load(C + 'triax_bisector.npz')
    fld = {g: (p['d'], p['s1'][-1]) for g, p in P3.items()}
    D = np.array([{NM[k]: max(TX[k + '_h'][0][0] - 1 / 3, 0) for k in NM}[g] for g in d.g]); comp = np.maximum(-d.R, 0); bmax = 1 / D.max()
    d.K = lambda g, x: np.interp(np.clip(np.asarray(x, float), 0, d.b), *fld[g])
    def build(p):
        def f(g, dv, ph, lv):
            a = p['a0'] + p['kap'] * dv; r = RK.L0 / 2
            if a + r >= d.b: return math.inf
            Wf = RK.W(a, r) * (RK.Y_fw(a, d.b) if RK.FINITE_WIDTH else 1.0)
            return ph ** -p['n'] * (float(d.K(g, a + r)) + (Wf - 1) * float(RK.wf_mean(d, g, a)[0]))
        return RK.per_group(d, f) * (1 + p.get('b', 0) * D * comp)
    base = RK.models(d)['crack superposed on local field (my version)']
    ex, st, bd = list(base[0]), [list(s) for s in base[1]], dict(base[2])
    if rm: ex += ['b']; st = [s + [0.5 * bmax] for s in st]; bd['b'] = (0, bmax)
    return (ex, st, bd, build)
JOBS = []
JOBS.append(('YS', 'rank crack rule, 2D field (review)', lambda d: RK.models(d)['crack superposed on local field (my version)']))
JOBS.append(('YS', '3D mid-plane + plain field (baseline)', lambda d: ys3d(d, False)))
JOBS.append(('YS', '3D baseline + triaxiality R-1 term', lambda d: ys3d(d, True)))
JOBS.append(('ES', 'rank crack rule (ES)', lambda d: RK.models(d)['crack superposed on local field (my version)']))
def recon(d, Wp):
    from es2020 import sde
    def build(p):
        a = p['a0']
        return RK.per_group(d, lambda g, dv, ph, lv: float(sde.f_sent(a / Wp)) / 1.12 if g == 'plain' else float(RK.wf_mean(d, g, a)[0]) * float(sde.f_sent(a / 6.0)) / 1.12)
    return (['a0'], [[0.3], [1.0], [1.8], [2.4]], dict(a0=(1e-3, 2.9)), build)
JOBS.append(('ES', 'reconcile finite-width crack, own-lab 125 (review 0.075 row)', lambda d: recon(d, 3.0)))
JOBS.append(('ESpool', 'reconcile finite-width crack, pooled 195', lambda d: recon(d, 3.0)))
OUT = {}
RK.FINITE_WIDTH = "--fw" in sys.argv
for ds, name, mk in JOBS:
    if ds == 'ESpool':
        import reconcile_pool as rp; d = rp.pooled()
    else:
        d = RK.load_ys() if ds == 'YS' else RK.load_es()
    spec = mk(d); t = time.time(); o = RK.run(d, name, spec, verbose=False); p = o['par']; s = stats(d, p, spec[3](p))
    a_max = (p['a0'] + p.get('kap', 0) * d.dv.max()) if 'a0' in p else None
    OUT[f'{ds} | {name}'] = dict(aic=o['aic'], par=p, a_max=a_max, **s, seconds=time.time() - t)
    f_, h_ = s['fit'], s['held']
    print(f"{ds:6s} {name:58s} AIC {o['aic']:.1f} a_max {a_max if a_max is None else round(a_max, 2)} | FIT S {f_['rms_S']:.3f}/{f_['bias_S']:+.3f} N {f_['rms_N']:.2f}/{f_['bias_N']:+.2f} "
          f"tilt {f_['tilt_exp']:+.2f}/{f_['tilt_pred']:+.2f} cov {f_['cov']:.0f}% | HELD S {h_['rms_S']:.3f}/{h_['bias_S']:+.3f} N {h_['rms_N']:.2f}/{h_['bias_N']:+.2f} "
          f"tilt {h_['tilt_exp']:+.2f}/{h_['tilt_pred']:+.2f} cov {h_['cov']:.0f}%" + (f" | U1-U3 {s['U1_minus_U3_N']:+.2f}" if 'U1_minus_U3_N' in s else '') + f" | {time.time()-t:.0f} s", flush=True)
json.dump(OUT, open('/mnt/project-files/yaren-susmel-2025/fixed_field' + ('_fw' if RK.FINITE_WIDTH else '') + '.json', 'w'), indent=1)
