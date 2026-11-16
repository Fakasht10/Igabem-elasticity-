"""Re-run the 7 PLA fatigue specimen solves (ES 3 notches, YS 3 notches + plain hourglass) whose solution caches were
lost with their cloud sessions. Logs unknowns and timing lines from bc_case; results compared against stored extracts
by check_pla_solves.py. Run from python/: IGABEM_CACHE=<dir> python rerun_pla_solves.py"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fatigue.specimens import Specimen
from fatigue import fields, ys2025
from failure import igabem_backend as _ib
_init = _ib.Solve.__init__
def _init2(self, *a, **k):
    _init(self, *a, **k)
    open(os.path.join(self.dir, "report.txt"), "w").write("\n".join(self.report) + f"\nwall_to_ready_s {self.seconds:.3f}\n")
_ib.Solve.__init__ = _init2

ES = {"sharp": Specimen("edge", T=3.0, rho=0.15, alpha_deg=35.0, WG=25.0, WN=6.0, L=86.0),
      "intermediate": Specimen("edge", T=3.0, rho=1.0, alpha_deg=0.0, WG=25.0, WN=6.0, L=86.0),
      "blunt": Specimen("edge", T=3.0, rho=3.0, alpha_deg=0.0, WG=25.0, WN=6.0, L=86.0)}
out = {}
log = lambda s: print(s, flush=True)
which = sys.argv[1:] or (["ES:" + k for k in ES] + ["YS:" + k for k in ("plain", "V_r0.15", "U_r1", "U_r3")])
for w in which:
    src, name = w.split(":")
    t0 = time.perf_counter()
    if src == "ES":
        f = fields.field(ES[name], h=1.5, work=os.path.join(fields.ROOT, "cache", "es2020"), volume=False, log=log)
        out[w] = dict(Kt_mid=float(f["Kt_mid"]), unknowns=int(f.get("unknowns", -1)), wall=time.perf_counter() - t0)
    else:
        f = ys2025.profiles(name, log=log)
        out[w] = dict(Kt_mid=float(f["s1"][-1, 0]), unknowns=int(f["unknowns"]), solver_s=float(f["seconds"]),
                      wall=time.perf_counter() - t0)
    log(f"DONE {w}: {out[w]}")
json.dump(out, open("rerun_pla_solves.json", "w"), indent=1)
