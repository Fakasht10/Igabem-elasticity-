Sutura commit `158d9ab`; hole limit for `ours` = 5 % of bbox diagonal (`ours25`: 25 %). Wall time 1149 s.

Every column was measured with the same yardstick (stl_repair_core diagnosis + manifold3d), never the tool's own claim.
Distances are vertex-sampled maxima as a fraction of the bbox diagonal (out→in: invented geometry, in→out: lost geometry).

| mesh | faces | input defects (loops/NMv/SI) | ours | ours wt | ours25 | Sutura S1 wt | S1 healthy comps | S1 SI | S1 out→in | S1 in→out | full wt | full SI | full out→in | full in→out | full s | ours s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| artec_bovine-heart | 89999 | 4/2/20 | FAIL | no | FAIL | yes | 2/0 | 20 | 0.0e+00 | 0.0e+00 | yes | 4 | 3.0e-04 | 1.4e-03 | 10 | 9.1 |
| artec_copper-key | 90000 | 4/0/2 | FAIL | no | FAIL | yes | 1/1 | 2 | 0.0e+00 | 1.0e-03 | yes | 2 | 0.0e+00 | 1.0e-03 | 9 | 12.8 |
| artec_crocodile-statue | 90000 | 12/0/412 | FAIL | no | FAIL | yes | 6/8 | 412 | 0.0e+00 | 1.5e-03 | yes | 324 | 0.0e+00 | 1.5e-03 | 54 | 28.9 |
| artec_lobster-hd | 90000 | 0/0/0 | PASS | yes | PASS | yes | 1/1 | 0 | 0.0e+00 | 0.0e+00 | yes | 0 | 0.0e+00 | 0.0e+00 | 5 | 4.2 |
| artec_metal-nut | 89999 | 8/2/688 | FAIL | no | FAIL | no | 1/0 | 683 | 0.0e+00 | 1.8e-03 | timeout | – | – | – | – | 29.2 |
| artec_pipe-bend | 90000 | 0/0/41 | FAIL | yes | FAIL | yes | 1/1 | 41 | 0.0e+00 | 0.0e+00 | yes | 10 | 1.3e-08 | 2.0e-03 | 40 | 4.2 |
| artec_plastic-bolt | 90000 | 0/0/0 | PASS | yes | PASS | yes | 1/1 | 0 | 0.0e+00 | 0.0e+00 | yes | 0 | 0.0e+00 | 0.0e+00 | 5 | 4.1 |
| artec_spanner | 90000 | 0/0/6 | FAIL | yes | FAIL | yes | 1/1 | 6 | 0.0e+00 | 0.0e+00 | yes | 0 | 0.0e+00 | 2.0e-04 | 9 | 3.4 |
| thingi10k_100045 | 420 | 0/0/64 | FAIL | yes | FAIL | yes | 3/3 | 64 | 0.0e+00 | 0.0e+00 | yes | 0 | 2.7e-15 | 0.0e+00 | 1 | 0.2 |
| thingi10k_100077 | 528 | 0/0/0 | PASS | yes | PASS | yes | 1/1 | 0 | 0.0e+00 | 0.0e+00 | yes | 0 | 0.0e+00 | 0.0e+00 | 0 | 0.2 |
| thingi10k_100173 | 250 | 1/0/0 | PASS | no | PASS | yes | 1/0 | 0 | 0.0e+00 | 0.0e+00 | yes | 0 | 0.0e+00 | 0.0e+00 | 0 | 0.0 |
| thingi10k_100281 | 90000 | 0/43/3341 | FAIL | no | FAIL | yes | 1/0 | 3341 | 0.0e+00 | 0.0e+00 | yes | 3497 | 4.8e-08 | 0.0e+00 | 102 | 28.2 |
| thingi10k_100827 | 71 | 0/10/0 | FAIL | no | FAIL | no | 0/0 | 28 | 0.0e+00 | 0.0e+00 | no | 28 | 0.0e+00 | 0.0e+00 | 9 | 0.0 |
| thingi10k_1038439 | 4578 | 0/9/413 | FAIL | no | FAIL | yes | 7/4 | 1073 | 0.0e+00 | 0.0e+00 | yes | 0 | 1.1e-10 | 1.2e-02 | 5 | 17.1 |
| thingi10k_1038441 | 10418 | 5/18/4582 | FAIL | no | FAIL | yes | 56/21 | 4623 | 0.0e+00 | 0.0e+00 | yes | 280 | 8.0e-08 | 4.3e-02 | 16 | 39.0 |
| thingi10k_1356633 | 14328 | 0/0/3 | FAIL | yes | FAIL | yes | 1/1 | 3 | 0.0e+00 | 0.0e+00 | yes | 0 | 0.0e+00 | 0.0e+00 | 1 | 2.2 |
| thingi10k_136634 | 17180 | 0/0/0 | PASS | yes | PASS | yes | 1/1 | 0 | 0.0e+00 | 0.0e+00 | yes | 0 | 0.0e+00 | 0.0e+00 | 2 | 2.8 |
| thingi10k_145065 | 35614 | 0/18/210 | FAIL | no | FAIL | no | 0/0 | 210 | 0.0e+00 | 0.0e+00 | yes | 252 | 1.4e-07 | 0.0e+00 | 65 | 16.8 |
| thingi10k_224108 | 11180 | 0/0/0 | FAIL | no | FAIL | no | 0/0 | 5 | 0.0e+00 | 0.0e+00 | yes | 0 | 0.0e+00 | 9.0e-03 | 47 | 2.1 |
| thingi10k_228302 | 12400 | 0/0/0 | PASS | yes | PASS | yes | 8/8 | 0 | 0.0e+00 | 0.0e+00 | yes | 0 | 0.0e+00 | 0.0e+00 | 1 | 2.7 |
| thingi10k_235725 | 1980 | 0/0/0 | PASS | yes | PASS | yes | 1/1 | 0 | 0.0e+00 | 0.0e+00 | yes | 0 | 0.0e+00 | 0.0e+00 | 1 | 0.6 |
| thingi10k_248395 | 13264 | 0/0/0 | FAIL | yes | FAIL | yes | 0/0 | 0 | 9.5e-09 | 8.9e-09 | yes | 5 | 9.5e-09 | 8.9e-09 | 8 | 52.6 |
| thingi10k_260537 | 65266 | 0/0/0 | PASS | yes | PASS | yes | 1/1 | 0 | 0.0e+00 | 0.0e+00 | yes | 0 | 0.0e+00 | 0.0e+00 | 4 | 6.0 |
| thingi10k_313444 | 90000 | 0/0/0 | PASS | yes | PASS | yes | 2/5 | 0 | 0.0e+00 | 3.6e-03 | yes | 0 | 0.0e+00 | 6.5e-03 | 6 | 5.2 |
| thingi10k_331105 | 90000 | 0/0/8 | FAIL | yes | FAIL | yes | 1/1 | 8 | 0.0e+00 | 0.0e+00 | yes | 2 | 0.0e+00 | 4.9e-04 | 5 | 4.1 |
| thingi10k_39507 | 90000 | 0/0/339 | FAIL | yes | FAIL | yes | 3/3 | 339 | 0.0e+00 | 0.0e+00 | yes | 20 | 1.4e-08 | 4.4e-02 | 26 | 4.7 |
| thingi10k_40886 | 90000 | 0/0/207 | FAIL | no | FAIL | yes | 2/0 | 230 | 0.0e+00 | 0.0e+00 | yes | 2 | 0.0e+00 | 4.1e-03 | 15 | 10.8 |
| thingi10k_42844 | 748 | 0/0/0 | PASS | yes | PASS | yes | 1/1 | 0 | 1.1e-08 | 1.1e-08 | yes | 0 | 1.1e-08 | 1.1e-08 | 0 | 0.7 |
| thingi10k_46012 | 90000 | 8/0/929 | FAIL | no | FAIL | no | 0/0 | 207 | 0.0e+00 | 0.0e+00 | yes | 69 | 2.0e-04 | 4.8e-04 | 142 | 58.5 |
| thingi10k_475828 | 278 | 0/0/0 | FAIL | yes | FAIL | yes | 1/1 | 0 | 0.0e+00 | 0.0e+00 | yes | 0 | 0.0e+00 | 0.0e+00 | 0 | 0.0 |
| thingi10k_502009 | 8816 | 0/0/0 | FAIL | yes | FAIL | yes | 1/1 | 0 | 1.2e-08 | 1.3e-08 | yes | 0 | 1.2e-08 | 1.3e-08 | 3 | 20.0 |
| thingi10k_55280 | 90000 | 0/0/32 | FAIL | yes | FAIL | yes | 1/1 | 32 | 0.0e+00 | 0.0e+00 | yes | 2 | 0.0e+00 | 4.7e-03 | 6 | 5.2 |
| thingi10k_55772 | 4418 | 0/0/39 | FAIL | no | FAIL | yes | 14/14 | 31 | 2.8e-08 | 2.9e-08 | yes | 0 | 2.8e-08 | 4.4e-02 | 6 | 23.2 |
| thingi10k_57854 | 132 | 0/0/0 | PASS | yes | PASS | yes | 1/1 | 0 | 1.3e-08 | 1.3e-08 | yes | 0 | 1.3e-08 | 1.3e-08 | 1 | 0.2 |
| thingi10k_59226 | 11530 | 0/0/0 | PASS | yes | PASS | yes | 1/1 | 0 | 1.6e-08 | 1.6e-08 | yes | 0 | 1.6e-08 | 1.6e-08 | 1 | 5.1 |
| thingi10k_60916 | 12430 | 0/0/0 | PASS | yes | PASS | yes | 6/6 | 0 | 2.1e-08 | 2.3e-08 | yes | 0 | 2.1e-08 | 2.3e-08 | 2 | 10.6 |
| thingi10k_63785 | 90000 | 0/0/118 | FAIL | yes | FAIL | yes | 1/1 | 118 | 0.0e+00 | 0.0e+00 | yes | 118 | 0.0e+00 | 0.0e+00 | 36 | 7.2 |
| thingi10k_70561 | 1052 | 0/0/0 | PASS | yes | PASS | yes | 1/1 | 0 | 1.9e-08 | 1.5e-08 | yes | 0 | 1.9e-08 | 1.5e-08 | 1 | 0.8 |
| thingi10k_71691 | 11005 | 0/0/2 | FAIL | no | FAIL | yes | 1/0 | 5 | 0.0e+00 | 4.8e-05 | yes | 0 | 0.0e+00 | 6.1e-05 | 1 | 1.3 |
| thingi10k_81221 | 6948 | 0/0/0 | PASS | yes | PASS | yes | 1/1 | 0 | 0.0e+00 | 0.0e+00 | yes | 0 | 0.0e+00 | 0.0e+00 | 1 | 3.2 |

**Totals (n=40):** ours PASS 15 (ours25 PASS 15); Sutura Stage 1 watertight 35; Sutura full watertight 38 (timeouts 1).
Proper self-intersections left in output: Stage 1 22 meshes, full 15 meshes. Component count reduced by Stage 1: 5 meshes; changed by full: 10 meshes.
Vertex-sampled deviation > 1e-3 x diagonal: Stage 1 4 meshes, full 12 meshes; ours > 0 on 0 meshes.
