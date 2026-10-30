#!/bin/sh
# Hierarchical operator (src/hmat: element-column H-matrix, sampled CUR on [H | gamma G], near LU +
# affine coarse correction, GMRES to 1e-8) on the degree-4 plate with a hole, 3D Kirsch field, from
# 17k to 88k unknowns at eps 1e-5, and 88k at eps 1e-6 -> results/hmat_gate.csv.
# Run from the repository root after building. About 10 min on 4 cores; peak memory about 8 GB.
set -e
B=./build/hmat_bench
C=results/hmat_gate.csv
G=build/geo
mkdir -p $G
rm -f $C
python3 python/make_geometries.py 5 --plate 4 0.2 $G > /dev/null
(cd python && python3 -c "
import make_geometries as g
for r in (4, 5):
    pl = g.plate_hole(0.2, L=3.0)
    for _ in range(2): pl = [g.elevate_bezier(g.elevate_bezier(P, 0), 1) for P in pl]
    for P in pl:
        for _ in range(r): g.refine_uniform(P)
    g.write_dat('../$G/plate_hole_L3_p4_t0.2_r%d.dat' % r, pl, 'plate_hole_L3_p4_t0.2_r%d' % r)")
export OPENBLAS_NUM_THREADS=1
for m in plate_hole_p4_t0.2_r4 plate_hole_L3_p4_t0.2_r4 plate_hole_p4_t0.2_r5 plate_hole_L3_p4_t0.2_r5; do
  $B $G/$m.dat --eps 1e-5 --csv $C ${m}_eps1e-5 | tail -1
done
$B $G/plate_hole_L3_p4_t0.2_r5.dat --eps 1e-6 --csv $C plate_hole_L3_p4_t0.2_r5_eps1e-6 | tail -1
