#!/bin/sh
# Canned probes and 3D fields (results/fields/*.csv, *.vtu; figures results/fig_probes_*,
# results/fig_fields_*). Run from the repository root after building; about 1 min on
# 4 cores with the operators in cache/ (run_plate.sh fills it). The .vtu files open
# in ParaView and are not committed (git-ignored, ~10 MB each).
set -e
B=./build/plate_test
O=results/fields
mkdir -p $O
# Kirsch field, traction everywhere, p = 4, 1,024 elements: everything against the exact field.
$B geo/plate_hole_p4_t0.2_r3.dat fields kirsch 0.2 --fields $O/p4_r3
# Thick plate (t/a = 2), Kirsch load on the sides, free faces: the 3D thickness effect.
$B geo/plate_hole_p4_t0.4_r3.dat fields kirsch_free 0.4 --fields $O/p4_t0.4_r3
# Long strip (L = 3, d/W = 0.4), clamped at x = -3/2, pulled at x = +3/2.
$B geo/plate_hole_L3_p4_t0.2_r3.dat fields tension 0.2 --length 3 --bc clamp --fields $O/clamp_L3_r3
python3 python/plot_probes.py plate $O/p4_r3_kirsch --t 0.2 --out results/fig_probes_kirsch.png
python3 python/render_vtu.py $O/p4_r3_kirsch --out results/fig_fields_kirsch.png
python3 python/plot_probes.py plate $O/p4_t0.4_r3_kirsch_free --t 0.4 --out results/fig_probes_thick.png
python3 python/render_vtu.py $O/p4_t0.4_r3_kirsch_free --out results/fig_fields_thick.png
python3 python/plot_probes.py plate $O/clamp_L3_r3_tension --t 0.2 --out results/fig_probes_strip_clamped.png
python3 python/render_vtu.py $O/clamp_L3_r3_tension --out results/fig_fields_strip_clamped.png
