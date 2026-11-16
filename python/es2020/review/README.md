# Independent review of the notch-fatigue collapse and SDE work (Ezeh–Susmel 2020, Yaren–Susmel 2025)

Post-processing only: every script reads stored fields (`es2020/data/*.json`, `ys/profiles.json`) and stored specimen data. The one new field quantity is the Yaren–Susmel volume quadrature (`ys_wl_fields.py`, existing 2D FE `ys/fe2d.py`, seconds).

## Summary of findings
- **Fig. 1 plain pool.** Of the 195 plain points in Ezeh & Susmel 2020 Fig. 1, 61 are other-lab (Letcher R −1, Afrose R 0), 9 ambiguous, 125 own-lab (`fig1_source_match.json`). The plain factor of about 2 (σ₁ at L/2 = 0.2 mm against plain) stays 1.96 on the own-lab points; other-lab alone 2.77. It is not a batch effect: in net stress own-lab plain and the sharp V lie on one curve.
- **TCD variants of the SDE** (`sde_tcd*.py`): best is L = 0.4 mm fixed, Walker damage, free plain factor (held-out 0.074); tied plain fails (0.20–0.25). Lumped self-heating does not explain the gap.
- **Defect / size-effect models** (`rank.py`, `combo.py`, `wl_gauge.py`, `reconcile.py`). Held-out = rms log10 stress on U r1 + U r3.
  - Ezeh–Susmel, one basis: finite-width crack with automatic path (SDE thread rule, W 3 mm plain thickness, 6 mm ligament) 0.075; weakest link over volume with m tied to scatter and a fixed plain gauge 0.085–0.098 (10–33 mm; 33 mm = ASTM D638 Type IV, an ASSUMPTION), 4 constants, no plain factor; free plain factor 0.073 and net stress 0.095 are references, not geometry-free.
  - Yaren–Susmel (fit plain + V at 5 in-fills, hold out U): crack on the local field a = 0.41 + 2.6 d_v AIC 575 / 0.097; line method with L growing with void 610 / 0.087; weakest link 626 / 0.107; combined 566–568 / 0.118; uniform knock-downs (paper's homogenised crack, Murakami, porosity) 780–802 / 0.21–0.25.
  - On Ezeh–Susmel every one-constant model fits plain + V identically, so AIC cannot rank there.
- **Hidden inputs to flag:** weakest link on Ezeh–Susmel needs the plain gauge length (unreported; free in `rank.py`, fixed in `wl_gauge.py`); the crack path rule needs the plain thickness.

## Reproduce
Requirements: `numpy scipy matplotlib pymupdf scikit-fem triangle`. Run from `python/`:

    export PYTHONPATH=$PWD:$PWD/es2020/review
    R=es2020/review

| Output | Command | Runtime |
|---|---|---|
| `sde_tcd_results.json` (SDE variants 0–9) | `python $R/sde_tcd.py` | ~5 min |
| `sde_tcd_more_results.json` (L sweep, split γ) | `python $R/sde_tcd_more.py` | ~5 min |
| `fig_sde_tcd_heldout.png/pdf` | `python $R/sde_tcd_fig.py` | s |
| `sde_tcd_heat2_results.json` (lumped self-heating) | `python $R/sde_tcd_heat2.py` | ~3 min |
| `fig_review_bead_depth.png/pdf` (fixed-depth rule) | `python $R/bead_depth_fig.py` | s |
| `rank_fig8.txt`, `rank_figs57.txt` (ranking bootstrap) | `python $R/rank_bootstrap.py fig8 > $R/rank_fig8.txt`; same with `figs57`. **Needs `es2020/collapse.py` from branch `claude/project-thread-m5ejoo` (commit e438184, `field_rank`), which is not on this branch**: `git checkout origin/claude/project-thread-m5ejoo -- python/es2020/collapse.py` first, then restore it. | ~2 min each |
| `fig1_source_match.json` | `python $R/match_fig1_sources.py <mdp2.45.pdf>` (PDF not redistributed; the JSON is committed) | s |
| `plain_split_results.json` (split by R) | `python $R/plain_split.py` | ~5 min |
| `plain_split2_results.json` (split by source) | `python $R/plain_split2.py` | ~3 min |
| `ys_wl_fields.npz` | `python $R/ys_wl_fields.py` | 10 s |
| `rank_ES.json`, `rank_YS.json` | `python $R/rank.py ES`; `python $R/rank.py YS` | 1 + 3 min |
| `combo_tables.npz` | `python $R/combo_tables.py` | 1 min |
| `combo_ES.json`, `combo_YS.json` | `python $R/combo.py ES`; `python $R/combo.py YS` | 1 + 5 min |
| `wl_gauge.json` | `python $R/wl_gauge.py` | 20 s |
| `reconcile.json` | `python $R/reconcile.py` | 1 min |

Each script prints one line per model: `p` constants, AIC on the fitted set, FIT and HELD blocks as `bias/rms/slope` of log10(S*/S) per decade of life (`T` = tilted), implied K_f, constants.

## Data labels
- `ys/specimens_ijf2025.csv`: REAL specimens, vector-extracted from Yaren & Susmel IJF 193 (2025) 108750 Figs 4–7 (thread "Extract fatigue data from homogenised-material paper"). `ys/table*.csv`: REAL, transcribed tables. `ys/profiles.json`: 2D plane-stress FE bisector profiles (`ys/fe2d.py`).
- `fig1_source_match.json`: DERIVED attribution (image overlay), not stated by the authors point by point.
- Ezeh–Susmel specimens and IgBEM fields: `es2020/data/`.

## Clean-checkout check (2026-11-16)
Fresh clone of this branch, every command above run in order: all 12 result JSON files reproduce the committed ones exactly (no difference above 1e-3 in any number); both figures re-render. The only script that does not run on this branch as is is `rank_bootstrap.py` (see its row). Total runtime about 15 min.
