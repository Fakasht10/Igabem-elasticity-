#!/bin/sh
# Reproduces results/plate_tests.csv and results/plate_profile_*.csv
# (run from the repository root after building; about 3 min on 4 cores from scratch, about 10 s
# once the operators are in cache/).
set -e
B=./build/plate_test
C=results/plate_tests.csv
rm -f $C results/plate_profile_*.csv
ALL=patch,kirsch,kirsch_free,tension
# Error against unknowns and time: degree 2 and 4, t/a = 1, all load cases per mesh.
for p in 2 4; do
  for r in 0 1 2 3; do
    $B geo/plate_hole_p${p}_t0.2_r$r.dat plate_p${p}_t0.2_r$r $ALL 0.2 --csv $C \
       --profile results/plate_profile_p${p}_t0.2_r$r
  done
done
# Effect of interpolating the traction data: same mesh with exact data.
$B geo/plate_hole_p2_t0.2_r3.dat plate_p2_t0.2_r3 $ALL 0.2 --exact-data --csv $C
# Thickness effect, t/a = 0.5 and 2 (t/a = 1 above), degree 4.
for t in 0.1 0.4; do
  for r in 2 3; do
    $B geo/plate_hole_p4_t${t}_r$r.dat plate_p4_t${t}_r$r $ALL $t --csv $C \
       --profile results/plate_profile_p4_t${t}_r$r
  done
done
# Finite-width strip, L = 3 (loaded ends 7.5 a from the hole), d/W = 0.4, degree 4.
for r in 1 2 3; do
  $B geo/plate_hole_L3_p4_t0.2_r$r.dat plate_L3_p4_t0.2_r$r patch,tension 0.2 --length 3 --csv $C \
     --profile results/plate_profile_L3_p4_t0.2_r$r
done
