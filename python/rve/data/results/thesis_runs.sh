#!/bin/bash
# Biswas 2019 Table 7 geometries (identified from the porosities), 0/90 cells, l1g2 mesh: ~1 min each
cd "$(dirname "$0")/../../../.."; OUT=${OUT:-/tmp/thesis_runs}; mkdir -p $OUT
A=14.2248
run() { python3 python/rve/beads.py 090 $OUT/$1.dat --w $2 --h $3 --bint $4 --blat $5 --alpha $A --refine 1 --grade 2 > $OUT/$1.gen; ./build/rve_cell $OUT/$1.dat --bc kubc,pbc --ng 12 --json $OUT/$1.json > $OUT/$1.log 2>&1; }
run base 1.0 0.3625 0.6537 0.2240
run LH1 1.0 0.54375 0.6537 0.2240
run FW1 1.2 0.3625 0.6537 0.2240
run BW1 1.0 0.3625 0.78444 0.2240
run BW2 1.0 0.3625 0.52296 0.2240
echo done
