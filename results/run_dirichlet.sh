#!/bin/sh
# Reproduces results/dirichlet_tests.csv (run from the repository root after building).
set -e
B=./build/dirichlet_test
C=results/dirichlet_tests.csv
rm -f $C
# Linear patch test (u = alpha x): exact for every n by construction (see README)
$B geo/cube_p2.dat cube_p2 --nmin 1 --nmax 20 --csv $C
$B geo/sphere_r0.dat sphere_r0 --nmin 1 --nmax 12 --csv $C
$B geo/spheroid_k2_r0.dat spheroid_k2_r0 --nmin 1 --nmax 12 --csv $C
# Kelvin point load outside the sphere: quadrature (n) and h-convergence
for r in 0 1 2; do
  $B geo/sphere_r$r.dat src_sphere_r$r --nmin 1 --nmax 12 --source 1.5 1.7 2.0 --csv $C
done
$B geo/sphere_r3.dat src_sphere_r3 --nmin 2 --nmax 10 --source 1.5 1.7 2.0 --csv $C
