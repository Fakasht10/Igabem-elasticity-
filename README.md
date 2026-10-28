# igabem-elasticity

A 3D collocation isogeometric boundary element method (IgBEM) for linear elastostatics, following
Taus, Rodin, Hughes & Scott, *Isogeometric boundary element methods and patch tests for linear elastic
problems*, CMAME 357 (2019) 112591. This is the core that the planned extensions build on, starting
with a thickness-robust solver for thin curved plates.

**Status (milestone 1):** the interior Neumann patch test passes to machine precision on the watertight
quadratic cube and on exact rational spheres, spheroids and a perturbed sphere. The imperfect cube
stalls at a constant error, as it does in the paper. See [Results](#results).

## Build and run

Requires a C++17 compiler, CMake ≥ 3.16, Eigen ≥ 3.3, optionally OpenMP; Python 3 with NumPy and
Matplotlib for geometry generation and plots.

```sh
python3 python/make_geometries.py 1      # writes geo/*.dat (cube, imperfect cube, sphere family)
cmake -S . -B build && cmake --build build -j
./build/patch_test geo/cube_p2.dat cube --nmin 1 --nmax 24
./results/run_all.sh                     # all runs -> results/patch_tests.csv (~10 min on 4 cores)
python3 python/plot_patch_tests.py       # -> results/fig_patch_tests.{pdf,png}
```

## Plan check

**Theory.** Displacement boundary integral equation (SBIE) for an interior domain with outward normal,
Kelvin fundamental solution in the Brebbia–Dominguez convention (`src/kelvin.hpp`). The free term is
never formed: rigid translations give σ(x) = −PV∫T dΓ, so the equation is used in the regularized form

```
∫_Γ T(x,y) [u(y) − u(x)] dΓ_y = ∫_Γ U(x,y) t(y) dΓ_y
```

Both integrands are O(1/r), so the same scheme handles smooth points, C⁰ patch edges and cube corners.
This is the paper's σ = −K·1 identity (App. C). The hypersingular equation (HSBIE) is not implemented
yet.

**Physics.** λ = μ = 1 (ν = 1/4). Exact field u = αx, α = diag(−¼, −¼, 1), so σ = diag(0, 0, 5/2) and
t = σn. Pure Neumann data leaves the six rigid-body modes undetermined; they are removed (see Numerics).

**Numerics.**

| Item | Choice | Where |
| --- | --- | --- |
| Geometry | Multipatch NURBS in Bembel `.dat` format, any degree, rational weights | `src/nurbs.hpp` |
| Displacement space | Isoparametric: the geometry's own rational basis. Control points that coincide across patches are merged into one conforming dof, so a gap (imperfect cube) leaves dofs unmerged | `Mesh::build` |
| Collocation | Greville abscissae of each dof's first owning patch | `Mesh::build` |
| Singular quadrature | Duffy split of the integration cell into up to 4 triangles at the collocation point, n×n Gauss–Legendre per triangle | `duffyRect` |
| Curvature / aspect partitioning | App. B: split cells until the normal-angle change across a cell is ≤ 1/8 rad (κh ≤ 1/8) and the aspect ratio is ≤ 3 | `partitionElement` |
| Near-singular quadrature | App. B: quadtree refinement towards the closest point while d ≤ 3h | `nearRect` |
| Solver | Dense bordered system [A R; Rᵀ 0] with the 6 rigid modes R, partial-pivot LU | `app/patch_test.cpp` |
| Error measure | Relative L²(Γ) displacement error after removing the L²-best rigid motion, 10×10 Gauss per cell | `relativeL2Error` |
| Rank / conditioning | Full SVD of A every run: σ₆/σ_max (should vanish with the quadrature error, rank deficiency 6) and σ₇/σ_max (smallest non-rigid singular value) | `patch_test` output |
| Kernel check | ‖A·R_rot‖/‖A‖: rotations carry zero traction, so this must vanish. It catches sign or index errors in T | `patch_test` output |

**Visualization.** Semilog plots of error against integration points per direction, matching the
paper's Figs. 2, 3, 6 and 7, plus a CSV of every run (`results/patch_tests.csv`).

## Assumptions and deviations (please read)

1. **The base quadrature of Taus, Rodin & Hughes (MMMAS 2016) was not available and is reconstructed.**
   This code uses Duffy/polar splitting at the collocation point on the regularized integrands. That is a
   standard scheme with the same property the paper relies on (exponential convergence in n), but it is
   not guaranteed to be their rule. The hypersingular regularization from that paper is not attempted yet.
2. The κh ≤ 1/8 criterion is estimated from the normal-angle span of a cell (centre-to-corner angle × 2),
   not from principal curvatures.
3. Traction data are evaluated exactly at quadrature points (t = σn with the discrete geometry's normal)
   instead of being projected onto a traction basis. For this test that removes data error rather than
   adding it.
4. Rigid-body removal: the paper says only that rigid motion is "nullified"; here it is a bordered system
   plus an L²-best rigid fit in the error.
5. The sphere is Bembel's exact six-patch rational quartic (`geo/bembel_sphere.dat`, from
   [temf/bembel](https://github.com/temf/bembel), GPL-3.0). The spheroid is its exact affine image. The
   perturbed sphere is a smooth radial perturbation of the control points, not the paper's
   parametrization. Bembel itself is not linked yet; its H² compression is the planned route to the fast
   solver.

## Results

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

## Next steps

1. Interior Dirichlet patch test (HSBIE) and exterior Neumann (Eshelby) patch test.
2. Thin curved plate (cylindrical panel built as exact NURBS): Neumann patch test for t/R = 10⁻¹…10⁻³,
   then Scordelis–Lo as a 3D solid.
3. Iterative solver, preconditioner and H² compression.

## License

GPL-3.0 (see `LICENSE`), matching Bembel, whose sphere geometry is included and whose library is the
intended base.
