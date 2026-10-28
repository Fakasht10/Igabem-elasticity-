#!/bin/sh
# Reproduces results/cavity_tests.csv and results/cavity_profile_*.csv
# (run from the repository root after building; about 35 min on 4 cores).
set -e
B=./build/cavity_test
C=results/cavity_tests.csv
rm -f $C results/cavity_profile_*.csv
# Patch test u = alpha x: quadrature convergence.
$B geo/cube_cavity_r0.dat cube_cavity_r0 patch --nmin 2 --nmax 16 --csv $C
# Quadrature convergence at one refinement for the three load cases.
for k in pressure goodier tension; do
  $B geo/cube_cavity_r1.dat cube_cavity_r1 $k --nmin 2 --nmax 12 --csv $C
done
# h-refinement at n = 12, with stress profiles along the cavity meridian.
for k in pressure goodier tension; do
  for r in 0 1 2 3; do
    $B geo/cube_cavity_r$r.dat cube_cavity_r$r $k --nmin 12 --nmax 12 --csv $C \
       --profile results/cavity_profile_${k}_r$r.csv
  done
done
