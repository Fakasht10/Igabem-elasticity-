#!/bin/sh
# Reproduces results/patch_tests.csv (run from the repository root after building).
set -e
B=./build/patch_test
C=results/patch_tests.csv
rm -f $C
$B geo/cube_p2.dat cube_p2 --nmin 1 --nmax 24 --csv $C
$B geo/cube_p2_imperfect.dat cube_p2_imperfect --nmin 1 --nmax 24 --csv $C
for g in sphere_r0 spheroid_k2_r0 sphere_perturbed_r0 sphere_r1; do
  $B geo/$g.dat $g --nmin 1 --nmax 16 --csv $C
done
for g in sphere_r0 spheroid_k2_r0; do
  $B geo/$g.dat $g --nmin 1 --nmax 24 --no-partition --csv $C
done
