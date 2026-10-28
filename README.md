# igabem-elasticity

A 3D collocation isogeometric boundary element method (IgBEM) for linear elastostatics, following
Taus, Rodin, Hughes & Scott, *Isogeometric boundary element methods and patch tests for linear elastic
problems*, CMAME 357 (2019) 112591. This is the core that the planned extensions build on, starting
with a thickness-robust solver for thin curved plates.

**Status.**

- **Milestone 1 (SBIE, Neumann):** the interior Neumann patch test passes to machine precision on the
  watertight quadratic cube and on exact rational spheres, spheroids and a perturbed sphere. The
  imperfect cube stalls at a constant error, as it does in the paper.
- **Milestone 2 (HSBIE, Dirichlet):** the traction equation is regularized by Taylor subtraction and
  solved for a discontinuous traction space. The linear Dirichlet patch test is exact; a Kelvin
  point-load solution checks quadrature and h-convergence. See [Results](#results).
- **Milestone 3 (cube with a spherical cavity, traction-only):** a two-surface body (unit cube, exact
  rational sphere of radius 0.2 with inward-facing patches). The patch test converges to 1.6e-13; exact
  pressurized-cavity and Goodier fields converge under refinement; the stress concentration factor
  matches Goodier's 93/46 = 2.0217 to 1.1e-4. The finite cube in tension gives SCF 2.268.
- **Milestone 4 (plate with a circular through-hole, traction-only), on a reusable library:** exact
  rational geometry (volume exact to 1e-13). The 3D Kirsch field converges at O(h³) for p = 2 and
  O(h⁵) for p = 4; SCF error 1.2e-5 at 5,808 unknowns in 10 s from scratch. Thickness effect and a
  finite-width strip agree with the literature. Operators are assembled once, cached on disk and
  reused for every load case.

## Build and run

Requires a C++17 compiler, CMake ≥ 3.16, Eigen ≥ 3.3, optionally OpenMP; Python 3 with NumPy and
Matplotlib for geometry generation and plots.

```sh
python3 python/make_geometries.py 3      # writes geo/*.dat (cube, imperfect cube, sphere family, cube with cavity, plate with hole; r0..r3)
cmake -S . -B build && cmake --build build -j
./build/patch_test geo/cube_p2.dat cube --nmin 1 --nmax 24
./build/dirichlet_test geo/sphere_r0.dat sphere --nmin 1 --nmax 10 [--source 1.5 1.7 2.0]
./build/cavity_test geo/cube_cavity_r2.dat cav goodier --nmin 12 --nmax 12   # patch|pressure|goodier|tension
./build/plate_test geo/plate_hole_p4_t0.2_r3.dat p4r3 patch,kirsch,kirsch_free,tension 0.2   # all cases, one assembly
ctest --test-dir build                   # Kelvin kernels (finite differences) + regression of every validated case (~25 s)
./results/run_all.sh                     # Neumann runs -> results/patch_tests.csv (~10 min on 4 cores)
./results/run_dirichlet.sh               # Dirichlet runs -> results/dirichlet_tests.csv (~20 min)
./results/run_cavity.sh                  # cavity runs -> results/cavity_tests.csv, cavity_profile_*.csv (~35 min)
./results/run_plate.sh                   # plate runs -> results/plate_tests.csv, plate_profile_*.csv (~100 s; ~30 s with cache/)
python3 python/plot_patch_tests.py       # -> results/fig_patch_tests.{pdf,png}
python3 python/plot_dirichlet_tests.py   # -> results/fig_dirichlet_tests.{pdf,png}
python3 python/plot_cavity_tests.py      # -> results/fig_cavity_tests.{pdf,png}
python3 python/plot_plate_tests.py       # -> results/fig_plate_tests.{pdf,png}
python3 python/gen_cavity_exact.py       # (needs SymPy) re-derives src/cavity_exact_gen.hpp
python3 python/gen_kirsch_exact.py       # (needs SymPy) re-derives src/kirsch_exact_gen.hpp
```

## Library layout

Header-only C++17 in `src/`; each new case is a problem header plus a thin driver plus regression
values, not new numerics.

| Layer | Files | Content |
| --- | --- | --- |
| Geometry | `nurbs.hpp`, `mesh.hpp` | Multipatch rational B-splines, conforming dofs, Greville collocation |
| Quadrature | `quadrature.hpp`, `integrator.hpp` | Cell partition, Duffy singular rule, near-singular quadtree, adaptive far field with cached element rules |
| Kernels | `kelvin.hpp` | Kelvin U, T and traction kernels (finite-difference tested) |
| Operators | `sbie.hpp`, `hsbie.hpp`, `traction_space.hpp` | SBIE matrices H, G (H u = G t), HSBIE assembly, discontinuous traction space |
| Model | `neumann_model.hpp`, `cache.hpp` | `NeumannModel`: H and G assembled once and cached on disk (key: geometry bytes, material, quadrature options), bordered LU factored once, any number of load cases per solve |
| Post-processing | `recovery.hpp` | Boundary stress from u_h and t, point location on the surface |
| Problems | `problems/cube_cavity.hpp`, `problems/plate_hole.hpp`, `*_exact_gen.hpp` | Load cases, exact fields, quantities of interest, shared by drivers and tests |
| Tests | `tests/test_kernels.cpp`, `tests/test_regression.cpp` | Kernels; every validated result reproduced on a small mesh, including cache hit = fresh |

Not yet in the library: geometry refinement (still `python/make_geometries.py`), and mixed boundary
conditions (the next step reuses the same H and G columns). `patch_test` and `dirichlet_test` still
reassemble per n on purpose, because they are quadrature-convergence studies.

## Plan check

**Theory.** Displacement boundary integral equation (SBIE) for an interior domain with outward normal,
Kelvin fundamental solution in the Brebbia–Dominguez convention (`src/kelvin.hpp`). The free term is
never formed: rigid translations give σ(x) = −PV∫T dΓ, so the equation is used in the regularized form

```
∫_Γ T(x,y) [u(y) − u(x)] dΓ_y = ∫_Γ U(x,y) t(y) dΓ_y
```

Both integrands are O(1/r), so the same scheme handles smooth points, C⁰ patch edges and cube corners.
This is the paper's σ = −K·1 identity (App. C).

The traction equation (HSBIE) follows from applying the traction operator at x to the interior
representation, t_i(x) = ∫ D_im t_m − ∫ S_im u_m with D = T_x U (O(1/r²)) and S = T_x (T_y U)ᵀ (O(1/r³))
(`KelvinTraction` in `src/kelvin.hpp`; verified against finite differences by `tests/test_kernels.cpp`).
It is regularized by subtracting the linear field u_L(y) = u(x) + ∇u(x)(y − x), whose traction σ(x)n
satisfies the representation exactly. That removes the free term and leaves two O(1/r) integrands:

```
∫_Γ D(x,y) [t(y) − σ(x) n(y)] dΓ_y = ∫_Γ S(x,y) [u(y) − u(x) − ∇u(x)(y − x)] dΓ_y
```

∇u(x) combines the tangential derivatives of the Dirichlet data with the normal derivative implied by
t(x) through Hooke's law, so the unknown t(x) appears on both sides (`HsbieAssembler::gradient`). The
data part of the right-hand side is subtracted point by point, which keeps round-off at 1e-14 instead
of growing like n⁴. The subtraction needs u ∈ C¹ and t continuous at x, so traction collocation points
are kept strictly inside patches.

The cube with a cavity is a doubly bounded solid. The regularized SBIE holds unchanged, with Γ the
union of both surfaces and n outward from the solid, so into the cavity on the sphere. The rigid
translation identity behind the regularization holds over the full boundary, and the rigid modes are
still six because the solid is connected. Boundary stresses come from the solution without domain
evaluation: tangential gradient from u_h, normal derivative from the prescribed t via Hooke's law
(`src/recovery.hpp`, shared with the HSBIE).

**Physics.** λ = μ = 1 (ν = 1/4). Exact field u = αx, α = diag(−¼, −¼, 1), so σ = diag(0, 0, 5/2) and
t = σn. Pure Neumann data leaves the six rigid-body modes undetermined; they are removed (see Numerics).
For the Dirichlet runs a second exact field is the Kelvin solution of a unit point load at
(1.5, 1.7, 2.0), outside the unit sphere, in direction (1, 2, 3)/√14. It is smooth in the body but not
polynomial, so it tests quadrature and approximation together.
For the cavity (radius a = 0.2, centre c = (½, ½, ½)) three more exact fields are used, each with its
traction σn prescribed on both surfaces:

- the Lamé field u = p a³ (x − c) / (4μ|x − c|³), a pressure p = 1 in a cavity of an infinite medium;
- the Goodier (1933) field for remote uniaxial tension S = 1 along z around a traction-free spherical
  cavity. `python/gen_cavity_exact.py` derives it from three Papkovich–Neuber terms, checks Navier's
  equation and the free cavity symbolically, and generates the C++. Its stress concentration factor is
  σ_zz(equator)/S = (27 − 15ν)/(2(7 − 5ν)) = 93/46 ≈ 2.0217 at ν = ¼.
- the engineering case, without exact solution: the finite cube pulled by t = S e_z on z = 0, 1, with
  free lateral faces and a free cavity.

For the plate with a hole (square plate W = 1, thickness t, hole radius a = 0.2 through the thickness,
remote stress S = 1 along x) the plane-strain Kirsch field is an exact 3D solution when σ_zz = ν(σ_xx +
σ_yy) is prescribed on the faces z = ±t/2 (`python/gen_kirsch_exact.py` checks equilibrium and the free
hole symbolically). Its SCF is 3 at every z. Two cases without exact solution measure 3D effects: the
Kirsch traction on the four sides with free faces (thickness effect), and a strip of length L = 3
pulled at its ends (finite width, d/W = 0.4).

**Numerics.**

| Item | Choice | Where |
| --- | --- | --- |
| Geometry | Multipatch NURBS in Bembel `.dat` format, any degree, rational weights | `src/nurbs.hpp` |
| Displacement space | Isoparametric: the geometry's own rational basis. Control points that coincide across patches are merged into one conforming dof, so a gap (imperfect cube) leaves dofs unmerged | `Mesh::build` |
| Collocation | Greville abscissae of each dof's first owning patch | `Mesh::build` |
| Singular quadrature | Duffy split of the integration cell into up to 4 triangles at the collocation point, n×n Gauss–Legendre per triangle | `duffyRect` |
| Curvature / aspect partitioning | App. B: split cells until the normal-angle change across a cell is ≤ 1/8 rad (κh ≤ 1/8); aspect ratio ≤ 2 (App. B: 3) and opposite-edge ratio ≤ 1.5 (taper, not in App. B) | `partitionElement` |
| Far-field quadrature | Fixed n (paper) or, with `far_tol` > 0, Gauss order m per cell from the Bernstein-ellipse bound ρ^(−2m) < far_tol, ρ from distance / bounding radius; element-wide rules with geometry, basis and data evaluated once | `ElementIntegrator` |
| Near-singular quadrature | App. B: quadtree refinement towards the closest point while d ≤ 3h | `nearRect` |
| Solver | Dense bordered system [A R; Rᵀ 0] with the 6 rigid modes R, partial-pivot LU | `app/patch_test.cpp` |
| Error measure | Relative L²(Γ) displacement error after removing the L²-best rigid motion, 10×10 Gauss per cell | `relativeL2Error` |
| Rank / conditioning | Full SVD of A every run: σ₆/σ_max (should vanish with the quadrature error, rank deficiency 6) and σ₇/σ_max (smallest non-rigid singular value) | `patch_test` output |
| Kernel check | ‖A·R_rot‖/‖A‖: rotations carry zero traction, so this must vanish. It catches sign or index errors in T | `patch_test` output |
| Traction space (HSBIE) | The geometry's rational basis on each patch, discontinuous across patches (no merging), as the paper prescribes at edges and corners | `TractionSpace` |
| Traction collocation | Greville points with the first and last moved halfway inwards, so every point is interior to its patch | `TractionSpace::shifted` |
| Dirichlet data | Exact coefficients for u = αx; Greville interpolation for the point-load field | `interpolateDisplacement` |
| Solver (HSBIE) | Dense partial-pivot LU; 2-norm condition number from a full SVD every run | `app/dirichlet_test.cpp` |
| Shared quadrature | One driver for both equations: Duffy at the source point, App. B near-singular quadtree, Gauss elsewhere | `src/integrator.hpp` |
| Cavity geometry | Quadratic cube (6 patches) plus Bembel's sphere scaled to a = 0.2 with u and v swapped so its normals point into the cavity; both refined together (12 → 768 elements). Orientation check: the computed volume is 1 − 4πa³/3 to 1e-13 | `make_geometries.py` |
| Cavity solver | Same bordered SBIE system; general traction data t(y, n); SVD rank check every run | `app/cavity_test.cpp` |
| Stress recovery | σ on Γ from tangential gradient of u_h plus t, at the equator point c + a e_x and along the meridian in the x–z plane | `src/recovery.hpp` |
| Plate geometry | 16 biquadratic rational patches (each face quadrant blends the hole arc to the square edge; straight radial lines), exact degree elevation to p = 4, then k-refinement (16 → 1,024 elements); optional end blocks for a longer strip. Geometry error is zero: volume (L − πa²)t to 1e-13 | `make_geometries.py` |
| Plate data | Traction data interpolated in the traction space (t ≈ Σ N_j t_j), so the right-hand side is G t for any load; integrating the data exactly is kept as a check (`--exact-data`) | `NeumannModel::solve` |

**Visualization.** Semilog plots of error against integration points per direction, matching the
paper's Figs. 2, 3, 6 and 7, plus a CSV of every run (`results/*_tests.csv`). The plate figure adds
log-log error against unknowns and against wall time with h^(p+1) reference slopes, SCF error, and
the through-thickness stress profile.

## Assumptions and deviations (please read)

1. **The base quadrature of Taus, Rodin & Hughes (MMMAS 2016) was not available and is reconstructed.**
   This code uses Duffy/polar splitting at the collocation point on the regularized integrands. That is a
   standard scheme with the same property the paper relies on (exponential convergence in n), but it is
   not guaranteed to be their rule.
2. **The HSBIE regularization is also a reconstruction.** It is the classical constant-strain (Taylor)
   subtraction, not necessarily the "same technique as Laplace" the 2019 paper cites. One consequence:
   for u = αx the regularized residual of the exact solution vanishes point by point, so the linear
   Dirichlet patch test is passed for every n and does not test the quadrature. The paper's Fig. 5 shows
   exponential convergence in n, so its regularization must differ. The point-load test fills that gap.
3. **Cross-check with Keshava & Schanz (2025, [arXiv 2505.00713](https://arxiv.org/abs/2505.00713)).**
   Followed: the Kelvin tensor (their eq. 5, identical to ours after substituting ν), the form of the
   HBIE (their eq. 7: K′ = T_x U and D = −T_x(T_y U)ᵀ, our D and S up to that sign convention), and Duffy
   + Gauss for weakly singular collocation integrals. Not followed: their regularization of the
   hypersingular operator (eq. 23, after Han and Nédélec). It moves surface-curl derivatives onto the
   Galerkin test function v(x), which does not exist at a collocation point; their own collocation
   examples use only the SBIE. Their line integrals vanish on our closed surfaces, and they matter again
   only for open surfaces or a split into Dirichlet and Neumann parts.
4. The κh ≤ 1/8 criterion is estimated from the normal-angle span of a cell (centre-to-corner angle × 2),
   not from principal curvatures.
5. Traction data are evaluated exactly at quadrature points (t = σn with the discrete geometry's normal)
   instead of being projected onto a traction basis, except in the plate runs, which interpolate it so
   the operators can be reused for every load (item 8).
6. Rigid-body removal: the paper says only that rigid motion is "nullified"; here it is a bordered system
   plus an L²-best rigid fit in the error.
7. The sphere is Bembel's exact six-patch rational quartic (`geo/bembel_sphere.dat`, from
   [temf/bembel](https://github.com/temf/bembel), GPL-3.0). The spheroid is its exact affine image. The
   perturbed sphere is a smooth radial perturbation of the control points, not the paper's
   parametrization. Bembel itself is not linked yet; its H² compression is the planned route to the fast
   solver.
8. **Plate runs use three numerics changes not in the paper.** (a) Partition aspect ratio 2 instead of
   3 and a taper split: without them the plate's tapered cells converged slowly in n. The cube, sphere
   and cavity results are unchanged (regression test). (b) Adaptive far-field Gauss order
   (`far_tol` = 1e-10) instead of a fixed n: it reproduces the fixed n = 12 results to 8 digits at
   1/25 of the cost. (c) Interpolated traction data: at p = 2, r3 the Kirsch error is 4.6e-4 with
   interpolated data and 8.3e-4 with exact data, so interpolation costs nothing in accuracy here.

## Results

### Interior Neumann (SBIE)

| Geometry | Dofs (unknowns) | Integration cells | n for error < 1e-12 | Best error (n) | Error at n = 10, no partitioning |
| --- | --- | --- | --- | --- | --- |
| Cube, 6 quadratic patches (watertight) | 26 (78) | 6 | 15 | 1.4e-14 (23) | 3.2e-9 (flat faces: same as partitioned) |
| Cube, one control point moved by 1e-3 | 27 (81) | 6 | never: stalls at 1.5e-3 | 1.5e-3 | 1.5e-3 |
| Exact sphere, 6 rational quartic elements | 98 (294) | 6,144 | 9 | 4.1e-14 (10) | 1.3e-5 |
| Exact spheroid κ = 2 | 98 (294) | 3,816 | 13 | 8.1e-14 (15) | 4.5e-4 |
| Perturbed sphere | 98 (294) | 5,052 | 9 | 3.7e-14 (12) | not run |
| Exact sphere, refined once (24 elements) | 152 (456) | 6,144 | 9 | 3.6e-14 (10) | not run |

Diagnostics on every watertight case: the rotation residual ‖A·R_rot‖/‖A‖ reaches 1e-14, the six rigid-mode
singular values fall to the quadrature error, and σ₇/σ_max stays fixed (0.082 cube, 0.0048 sphere), so
A has numerical rank 3M − 6 and the bordered system is well conditioned. On the imperfect cube the rotation
residual stays at 4.7e-5: the gap breaks the rigid-body identity, which is exactly what this patch test is
meant to expose. The paper reports a stall near 1e-4 for its imperfect cube (Fig. 3); our 1.5e-3 uses a
different normalization (relative L²(Γ) after rigid-motion removal) and is not directly comparable.

Without partitioning the error still decays exponentially but much more slowly (sphere 9e-12 and spheroid
6e-7 at n = 24), reproducing the qualitative message of the paper's Figs. 6–7.

![Patch tests](results/fig_patch_tests.png)

### Interior Dirichlet (HSBIE)

| Case | Traction unknowns | Result | Reference |
| --- | --- | --- | --- |
| Linear patch test, quadratic cube | 162 | 2.5e-15 – 9e-14 for every n | exact |
| Linear patch test, exact sphere | 450 | 4e-14 (n = 1) to 1.5e-12 (n = 10) | exact |
| Linear patch test, spheroid κ = 2 | 450 | stalls at 3.2e-2 for every n | t = σn is not in the traction space (n is not rational on a spheroid) |
| Point load, sphere, 6 elements | 450 | 2.7e-2 | interpolation 1.3e-2 |
| Point load, sphere, 24 elements | 648 | 7.6e-3 | interpolation 3.7e-3 |
| Point load, sphere, 96 elements | 1,152 | 8.1e-4 | interpolation 8.5e-4 |
| Point load, sphere, 384 elements | 2,592 | 2.0e-5 (n ≥ 5) | interpolation 1.7e-5 |

All results are relative L²(Γ) traction errors.

- **Linear patch test.** Exact to round-off on the cube and the sphere, but for every n, including n = 1.
  With Taylor subtraction the residual of the exact solution vanishes point by point (see assumption 2),
  so this test checks consistency of the formulation, not the quadrature. On the curved sphere round-off
  grows slowly with n (1.5e-12 at n = 10), from cancellation inside the t(x)-dependent matrix terms. The
  spheroid stalls, as in the paper's Fig. 5, because its normal is not exactly representable.
- **Point-load solution.** The quadrature converges by n ≈ 4–6 at every refinement level. The error then
  stays within a factor of about 2 of the traction interpolation error at every level, so the method is
  quasi-optimal. Between 96 and 384 elements the error drops by a factor of 40, a rate of 5.3, matching
  p + 1 = 5 for the quartic sphere. The coarser levels are still pre-asymptotic: the source sits 1.9 radii
  from the centre.
- **Conditioning.** 2-norm condition numbers are 146 (cube) and 1,241 → 139 (sphere, 6 → 384 elements).
  They decrease under refinement, consistent with the paper's claim that the regularized system stays well
  conditioned.

![Dirichlet tests](results/fig_dirichlet_tests.png)

### Cube with a spherical cavity (SBIE, traction-only)

| Case | Elements (dofs) | Displacement error | σ_zz at equator | Reference |
| --- | --- | --- | --- | --- |
| Patch test u = αx | 12 (124) | 1.6e-13 at n = 16 | 2.5000000 | 5/2 |
| Pressurized cavity | 12 → 768 (124 → 1,216) | 7.1e-2 → 6.0e-4 | 0.50257 → 0.50004 | p/2 = 0.5 |
| Goodier field | 12 → 768 | 1.4e-2 → 1.6e-4 | 2.0331 → 2.02152 | 93/46 = 2.02174 |
| Finite cube in tension | 12 → 768 | — | 2.286 → 2.268 | none (Goodier 2.022 for an infinite body) |

Displacement errors are relative L²(Γ) after rigid-motion removal; n = 12 unless noted.

- **Patch test.** Exponential convergence in n, but slower than on the lone sphere (1e-12 at n = 14
  rather than n = 8). The slowest integrals are the near-singular cube-face integrals seen from the
  cavity, 0.3 away from faces of size 1. The quadrature is not the bottleneck for the load cases: σ_zz at
  the equator changes by less than 1e-5 relative for n ≥ 7 and converges exponentially (panel a).
- **h-refinement.** From 192 to 768 elements the Goodier displacement error drops by 13 and the SCF error
  by 15 (rates 3.7 and 3.9), and the pressure case by 12 (rate 3.6). The coarser steps are pre-asymptotic.
  The quadratic cube faces cap the asymptotic displacement rate at p + 1 = 3 there, below the quartic
  sphere; degree-elevating the cube would lift it. The SCF matches Goodier to 1.1e-4 relative at 768
  elements.
- **Finite cube in tension.** SCF 2.268 at 768 elements, changing by 1e-2 from 192 elements, so about
  2.27 to within ~1e-3 by analogy with the Goodier sequence. It is 12% above the infinite-medium value,
  the finite-width effect of a cavity spanning 40% of the cube. The compressive pole stress also grows,
  from −0.587 S to about −0.74 S. This number has no independent reference here; a FEM cross-check is
  the obvious next validation.
- **Rank.** σ₆/σ_max falls to 4e-13 with n (six rigid modes, exact null space); σ₇/σ_max rises from
  1.4e-3 to 7.1e-3 under refinement, so the reduced system stays well conditioned.

![Cavity tests](results/fig_cavity_tests.png)

### Plate with a circular through-hole (SBIE, traction-only)

All runs: `NeumannModel` with adaptive far field (`far_tol` = 1e-10, n = 12 near field), interpolated
traction data, 4 cores. Time = assembly of H and G + LU + solve of all four load cases; a rerun with the
operators in `cache/` skips assembly.

| Mesh | Unknowns | Kirsch u error | SCF mid-plane (exact 3) | Assembly | LU | 4 solves |
| --- | --- | --- | --- | --- | --- | --- |
| p = 2, 64 el. | 432 | 5.6e-2 | 3.1682 | 0.12 s | 0.005 s | 0.002 s |
| p = 2, 256 el. | 1,200 | 1.0e-2 | 3.0071 | 0.32 s | 0.05 s | 0.01 s |
| p = 2, 1,024 el. | 3,888 | 4.6e-4 | 2.99941 | 1.95 s | 0.72 s | 0.07 s |
| p = 4, 64 el. | 1,200 | 6.6e-3 | 2.99676 | 0.66 s | 0.04 s | 0.01 s |
| p = 4, 256 el. | 2,352 | 5.0e-3 | 3.0123 | 1.30 s | 0.19 s | 0.03 s |
| p = 4, 1,024 el. | 5,808 | 3.8e-5 | 2.999963 | 7.57 s | 2.06 s | 0.17 s |

- **Order.** The last step gives rate 4.5 for p = 2 and 7.0 for p = 4 in h (expected p + 1 = 3 and 5; both still above the asymptote);
  the middle p = 4 step is pre-asymptotic (the interpolation error of the space shows the same plateau,
  and the BEM error stays within 2.4× of it). An SCF accurate to 1e-3 needs about 3,000 unknowns at
  p = 2 and about 1,200 at p = 4; 1e-5 needs p = 4 and about 6,000 (panel c).
- **Geometry error is zero:** the rational patches reproduce the hole and the faces exactly.
- **Patch test** u = αx on every mesh: 1e-9 to 3e-6, the floor set by `far_tol`, 10× below the finest
  Kirsch error.
- **Thickness effect** (Kirsch traction on the sides, free faces, p = 4, 1,024 el.):

  | t/a | SCF mid-plane | SCF at the face |
  | --- | --- | --- |
  | 0.5 | 3.026 | 2.957 |
  | 1 | 3.062 | 2.879 |
  | 2 | 3.089 | 2.752 |

  Uncertainty ≈ 0.01 (change from the 256-element mesh). The mid-plane rises above 3 and the face drops
  below it as the plate thickens, as in Folias & Wang, *Comput. Mech.* 6 (1990) 379–391, who give
  curves but no tables, so the comparison is qualitative.
- **Finite-width strip** (L = 3, d/W = 0.4, t/a = 1, p = 4): SCF on the gross stress 3.794, 3.810, 3.832
  at 1,800, 3,528, 8,712 unknowns (face 3.557, 3.560, 3.578); the r3 run takes 30 s. Plane stress
  (Howland 1930 via the Peterson curve fit Kt,net = 2 + 0.284(1−d/W) − 0.600(1−d/W)² + 1.32(1−d/W)³,
  quoted from memory and not checked against the book) gives Kt,net = 2.240, gross 3.733; times the
  mid-plane thickness factor 3.062/3 from the table above, 3.81. The BEM agrees to 0.6%, the size of its
  own last mesh change. Heywood's simpler 2 + (1−d/W)³ gives 3.69.
- **Square plate in tension** (L = W = 1): SCF 4.89 mid-plane. The loaded ends are only 1.5 a from the
  hole edge, so this is a geometry study, not a Kirsch comparison, and it has no reference.

![Plate tests](results/fig_plate_tests.png)

## Next steps

1. Mixed Dirichlet–Neumann problems on `NeumannModel`'s H and G (e.g. the plate gripped at one end, the
   cavity cube clamped on z = 0); exterior Neumann (Eshelby) patch test; FEM cross-check of the
   finite-cube SCF. Move geometry refinement into C++.
2. Thin curved plate (cylindrical panel built as exact NURBS): Neumann patch test for t/R = 10⁻¹…10⁻³,
   then Scordelis–Lo as a 3D solid.
3. Iterative solver, preconditioner and H² compression.

## License

GPL-3.0 (see `LICENSE`), matching Bembel, whose sphere geometry is included and whose library is the
intended base.
