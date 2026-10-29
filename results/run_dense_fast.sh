#!/bin/sh
# Old dense path (SbieEntries row by row, G formed, LAPACK dgetrf; entry rule tol 1e-7, minimum
# far order 4) against the fast path (DenseSbie, G c on the fly, block-preconditioned GMRES to 1e-10;
# degree-based far order) on the degree-4 plate with a hole, 3D Kirsch field -> results/dense_fast.csv.
# Run from the repository root after building. About 8 min on 4 cores; peak memory about 11 GB (26k, old).
set -e
B=./build/dense_bench
C=results/dense_fast.csv
G=build/geo
rm -f $C
python3 python/make_geometries.py 4 --plate 4 0.2 $G > /dev/null
python3 python/make_geometries.py 4 --plate 4 0.02 $G > /dev/null
(cd python && python3 -c "
import make_geometries as g
pl = g.plate_hole(0.2, L=3.0)
for _ in range(2): pl = [g.elevate_bezier(g.elevate_bezier(P, 0), 1) for P in pl]
for P in pl:
    for _ in range(4): g.refine_uniform(P)
g.write_dat('../$G/plate_hole_L3_p4_t0.2_r4.dat', pl, 'plate_hole_L3_p4_t0.2_r4')")
for m in plate_hole_p4_t0.2_r3 plate_hole_p4_t0.2_r4 plate_hole_L3_p4_t0.2_r4 plate_hole_p4_t0.02_r4; do
  $B $G/$m.dat --csv $C ${m}_fast | tail -1
  $B $G/$m.dat --assembly legacy --lu d --order 4 --csv $C ${m}_old | tail -1
done
