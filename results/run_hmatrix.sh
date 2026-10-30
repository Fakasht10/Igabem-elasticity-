#!/bin/sh
# Hierarchical compression of H and G on the degree-4 plate with a hole -> results/hmatrix_tests.csv
# (run from the repository root after building). Frozen settings: ACA tolerance 1e-6, eta 3,
# leaf 192. The 5.8k meshes are also compared with the dense operators (errors, Kirsch solution);
# the larger ones are compression only (dense H and G would need 5 GB at 17k and 45 GB at 57k).
# About 1 h on 4 cores; peak memory about 12 GB (57k).
set -e
export OPENBLAS_NUM_THREADS=1   # compression is parallel over blocks
B=./build/hmatrix_bench
C=results/hmatrix_tests.csv
G=build/geo
rm -f $C
python3 python/make_geometries.py 5 --plate 4 0.2 $G
python3 python/make_geometries.py 5 --plate 4 0.02 $G
O="--eps 1e-6 --eta 3 --leaf 192 --csv $C"
for t in 0.2 0.02; do
  for r in 1 2 3; do $B $G/plate_hole_p4_t${t}_r$r.dat p4_t${t}_r$r $t $O --dense; done
  $B $G/plate_hole_p4_t${t}_r4.dat p4_t${t}_r4 $t $O --cache none
done
$B $G/plate_hole_p4_t0.2_r5.dat p4_t0.2_r5 0.2 $O --cache none
# ACA tolerance sweep against the dense solution (5.8k).
$B geo/plate_hole_p4_t0.2_r3.dat p4_t0.2_r3_sweep 0.2 --eps 1e-4,1e-8 --eta 3 --leaf 192 --dense --csv $C
python3 python/plot_hmatrix.py
