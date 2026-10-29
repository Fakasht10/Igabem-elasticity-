#!/bin/sh
# Reproduces results/regions/*.csv (two-region bodies; run from the repository root after
# building). About 1 min on 4 cores from scratch, about 25 s with the operators in cache/.
set -e
B=./build/region_validate
D=results/regions
G=geo/regions
mkdir -p $D
rm -f $D/spheres.csv $D/probe_*.csv $D/coupon_*.csv
# Exact fields on the concentric spheres (shell 0.5 < r < 1, core r < 0.5), refinements 0..3.
for r in 0 1 2 3; do
  for c in patch lame eshelby; do
    $B $c $G/sph2_a0.5_r${r}_host.dat $G/sph2_a0.5_r${r}_core.dat --csv $D/spheres.csv \
       --probe $D/probe_${c}_r$r.csv
  done
done
# Stiffness contrast E_core / E_shell (infill 0.01 ... stiff inclusion 10), Eshelby, r2.
for k in 0.01 1 10; do
  $B eshelby $G/sph2_a0.5_r2_host.dat $G/sph2_a0.5_r2_core.dat --contrast $k --csv $D/spheres.csv
done
# Printed coupon: effective modulus against infill density, three meshes.
for n in 2 4 6; do
  $B coupon $G/coupon_w0.15_n${n}_host.dat $G/coupon_w0.15_n${n}_core.dat --csv $D/coupon_n$n.csv
done
