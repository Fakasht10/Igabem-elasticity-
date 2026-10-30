# Vendored from nurb-fea-tcd (github.com/Fakasht10/nurb-fea-tcd, MIT, Aleksey Kashtelyan), feakit/fracture.py at
# commit 553dd2a. See python/failure/LICENSE-nurb-fea-tcd. Local changes are listed in python/failure/README.md.
"""Energy TCD: failure prediction by reducing strain energy density over a
material-scale domain near a stress concentration.

The criterion family unifies the Theory of Critical Distances (TCD) and the
Averaged Strain Energy Density (ASED) criterion, following

    Seibert P., Taylor D., Berto F., Razavi N. (2022), "Energy TCD - robust and
    simple failure prediction unifying the TCD and ASED criterion",
    Engineering Fracture Mechanics 271, 108652.  CC BY.

Every length scale below is closed form in (K_Ic, sigma_UTS, nu). Nothing here
needs a mesh, a solve, or the paywalled H(2a, R0/rho) table that the classical
blunt-notch NSIF route depends on -- which is exactly why this is the path we
build first. `sed_per_tet` is the one piece that touches a solution.

Two things the paper settles that are easy to get wrong:

* There is no Energy Line Method. W is quadratic in stress, so the crack-tip
  1/sqrt(r) stress singularity becomes a 1/r energy singularity, whose line
  integral diverges. `R0_energy_lm` does not exist; that is deliberate, not an
  omission (paper Eq. 7).
* The domain radii are all different, and each has its own formula. Reusing the
  TCD `L` as an ASED or Energy AM radius is a real and commonly-made error --
  at nu=0.2 the Energy AM radius is ~1.5x L. Use `transfer_length_scale` when
  moving a calibrated radius between domains.
"""

import math

import numpy as np
from scipy.spatial import cKDTree



def _fea():
    """nurb-fea-tcd's JAX-FEM element routines, needed only by the FEM-side helpers (sed_per_tet,
    principal_stresses_per_tet, quad_point_samples, recovered_nodal_fields, assess_point_method); imported
    lazily so the boundary-element path runs without JAX."""
    from feakit import fea_gmsh
    return fea_gmsh

# Plane-state selectors. Every length scale here is state-dependent -- the
# choice moves R0 by ~20% and W by more, so it is a required argument
# everywhere rather than a defaulted one.
PLANE_STRAIN = "plane_strain"
PLANE_STRESS = "plane_stress"
_STATES = (PLANE_STRAIN, PLANE_STRESS)


def _check_state(state):
    if state not in _STATES:
        raise ValueError(f"state must be one of {_STATES}, got {state!r}")


def _k_over_sigma_sq(K_Ic, sigma_uts):
    """(K_Ic/sigma_UTS)^2, the common factor of every length scale.

    Units follow the inputs: K_Ic in MPa*sqrt(mm) with sigma in MPa gives mm.
    A K_Ic quoted in the usual MPa*sqrt(m) must be scaled by sqrt(1000) first.
    """
    if K_Ic <= 0 or sigma_uts <= 0:
        raise ValueError("K_Ic and sigma_uts must be positive")
    return (K_Ic / sigma_uts) ** 2


def critical_sed(sigma_uts, E):
    """W_c = sigma_UTS^2 / (2E), the critical strain energy density (paper Sec. 2.1).

    This is the failure threshold for every method in this module: a smooth
    specimen fails when its (uniform) SED reaches W_c, and that is what pins
    all the length scales below.
    """
    if sigma_uts <= 0 or E <= 0:
        raise ValueError("sigma_uts and E must be positive")
    return sigma_uts**2 / (2.0 * E)


def L_tcd(K_Ic, sigma_uts):
    """Classical TCD length scale, L = (1/pi)(K_Ic/sigma_UTS)^2 (paper Eq. 8).

    The stress-based methods sample at fixed multiples of L: the Point Method
    at L/2, the Line Method averaging over 0..2L, the Area Method over a
    semicircle of radius L. Independent of nu -- unlike every energy-based
    radius here, because principal stress alone carries no multiaxiality.
    """
    return _k_over_sigma_sq(K_Ic, sigma_uts) / np.pi


def R0_ased(K_Ic, sigma_uts, nu, state=PLANE_STRAIN):
    """Classical ASED control radius for the crescent/sector domain (paper Eq. 2).

        plane strain:  (1+nu)(5-8nu)/(4pi) * (K_Ic/sigma_UTS)^2
        plane stress:  (5-3nu)/(4pi)       * (K_Ic/sigma_UTS)^2
    """
    _check_state(state)
    if state == PLANE_STRAIN:
        c = (1.0 + nu) * (5.0 - 8.0 * nu) / (4.0 * np.pi)
    else:
        c = (5.0 - 3.0 * nu) / (4.0 * np.pi)
    return c * _k_over_sigma_sq(K_Ic, sigma_uts)


def R0_energy_pm(K_Ic, sigma_uts, nu, state=PLANE_STRAIN):
    """Energy Point Method radius: W is probed at this distance along the notch
    bisector (paper Eq. 6).

        plane strain:  (1-nu-2nu^2)/pi * (K_Ic/sigma_UTS)^2
        plane stress:  (1-nu)/pi       * (K_Ic/sigma_UTS)^2

    This is the domain to calibrate with -- a single line plot of SED along the
    bisector -- but not the one to assess coarse meshes with: point probing
    degrades with mesh size about as fast as the stress-based TCD PM, whereas
    the averaged domains stay robust (paper Sec. 4.3).
    """
    _check_state(state)
    if state == PLANE_STRAIN:
        c = (1.0 - nu - 2.0 * nu**2) / np.pi
    else:
        c = (1.0 - nu) / np.pi
    return c * _k_over_sigma_sq(K_Ic, sigma_uts)


def R0_energy_am(K_Ic, sigma_uts, nu, state=PLANE_STRAIN):
    """Energy Area Method radius: W averaged over a semicircle of this radius
    centred at the notch root (paper Eq. 9).

        plane strain:  [(1-nu-2nu^2)/pi (1+2/pi) + (1+nu)/(4pi)] * (K_Ic/sigma)^2
        plane stress:  [(1-nu)/pi (1+2/pi)       + (1+nu)/(4pi)] * (K_Ic/sigma)^2

    Mesh-robust like ASED. Note the paper's caveat: on a blunt notch the
    semicircle is tangential to the notch surface, so partitioning a mesh to
    align with it produces slivers at the root (their Fig. 4) -- the ASED
    crescent avoids that. With Gauss-point inclusion instead of partitioning,
    the tangency is harmless.
    """
    _check_state(state)
    base = (1.0 - nu - 2.0 * nu**2) if state == PLANE_STRAIN else (1.0 - nu)
    c = base / np.pi * (1.0 + 2.0 / np.pi) + (1.0 + nu) / (4.0 * np.pi)
    return c * _k_over_sigma_sq(K_Ic, sigma_uts)


def transfer_length_scale(R_pm_measured, nu, target, state=PLANE_STRAIN):
    """Carry an experimentally calibrated Energy PM radius onto another domain
    (paper Eq. 10).

    The Energy PM is the only energy domain that can be calibrated from a line
    plot, which is what makes point-based calibration available to ASED at all.
    The LEFM formulas then act as pure ratios:

        R_target = R_target(nu)/R_pm(nu) * R_pm_measured

    `target` is "ased" or "energy_am". The (K_Ic/sigma)^2 factor cancels, so the
    ratio depends only on nu and the plane state -- dummy values are fine here.
    """
    if target not in ("ased", "energy_am"):
        raise ValueError("target must be 'ased' or 'energy_am'")
    fn = R0_ased if target == "ased" else R0_energy_am
    ratio = fn(1.0, 1.0, nu, state) / R0_energy_pm(1.0, 1.0, nu, state)
    return ratio * R_pm_measured


def sed_from_principal(s1, s2, s3, E, nu):
    """Strain energy density from principal stresses (paper Eq. 4, first line):

        W = 1/(2E) [s1^2 + s2^2 + s3^2 - 2nu(s1 s2 + s2 s3 + s3 s1)]

    Accepts scalars or arrays.
    """
    s1, s2, s3 = np.asarray(s1), np.asarray(s2), np.asarray(s3)
    return (
        s1**2 + s2**2 + s3**2 - 2.0 * nu * (s1 * s2 + s2 * s3 + s3 * s1)
    ) / (2.0 * E)


def sed_per_tet(fe, u, E, nu):
    """Per-element strain energy density (MPa) and element volumes.

    Returns (w, volumes). Built on `element_stresses`, i.e. on JAX-FEM's own
    precomputed shape gradients and quadrature weights -- no element loop and no
    re-derived geometry.

    W is 1/2 sigma_ij eps_ij, which for a linear tet is exactly
    1/2 d^T K_e d / V_e: the element strain energy density, computed from nodal
    displacements and never from a recovered or extrapolated stress field. That
    distinction is why the energy methods tolerate coarse meshes -- displacements
    are the primary variational unknowns and are compatible across element
    boundaries, while stresses are not equilibrated across them.

    Deliberately NOT routed through `nodal_average` -- that is for rendering.
    Averaging nodal SED reintroduces the mesh sensitivity this avoids.
    """
    stress, volumes = _fea().element_stresses(fe, u, E, nu)
    C_inv_strain = np.linalg.solve(_fea()._elastic_C(E, nu), stress.T).T
    return 0.5 * np.einsum("ni,ni->n", C_inv_strain, stress), volumes


def principal_stresses_per_tet(fe, u, E, nu):
    """Per-element principal stresses, sorted descending (s1 >= s2 >= s3).

    Returns (n_cells, 3). s1 is what the stress-based TCD probes; all three feed
    `sed_from_principal`. `np.linalg.eigvalsh` is batched over the whole stack.
    """
    s, _vol = _fea().element_stresses(fe, u, E, nu)
    n = len(s)
    tensor = np.empty((n, 3, 3))
    tensor[:, 0, 0] = s[:, 0]; tensor[:, 1, 1] = s[:, 1]; tensor[:, 2, 2] = s[:, 2]
    tensor[:, 0, 1] = tensor[:, 1, 0] = s[:, 3]
    tensor[:, 1, 2] = tensor[:, 2, 1] = s[:, 4]
    tensor[:, 0, 2] = tensor[:, 2, 0] = s[:, 5]
    return np.linalg.eigvalsh(tensor)[:, ::-1]


def nodal_from_tets(mesh, values, volumes):
    """Volume-weighted element -> node reduction, for fields about to be sampled
    at a point.

    Volume weighting (not the plain count average used for rendering) keeps a
    large far-field element from outvoting the small elements that resolve a
    notch root. `values` may be (n_cells,) or (n_cells, k); the scatter is a
    single bincount over the flattened connectivity.

    For POINT PROBING only. Volume averaging over a control domain must go
    through `averaged_sed` on the element field directly -- reducing to nodes
    first and averaging those is what reintroduces mesh sensitivity.
    """
    values = np.asarray(values, float)
    volumes = np.asarray(volumes, float)
    squeeze = values.ndim == 1
    if squeeze:
        values = values[:, None]
    nodes_per_cell = mesh.cells.shape[1]
    flat_nodes = mesh.cells.reshape(-1)
    weights = np.repeat(volumes, nodes_per_cell)
    den = np.bincount(flat_nodes, weights=weights, minlength=mesh.n_nodes)
    # one bincount per component, vectorised across all cells; k is 1 or 3 here,
    # never a per-element loop
    num = np.stack([
        np.bincount(
            flat_nodes,
            weights=np.repeat(values[:, k], nodes_per_cell) * weights,
            minlength=mesh.n_nodes,
        )
        for k in range(values.shape[1])
    ], axis=1)
    den = np.where(den == 0, 1.0, den)
    out = num / den[:, None]
    return out[:, 0] if squeeze else out


def _cell_locator(mesh):
    """A KD-tree over cell centroids plus the barycentric maps, cached on the mesh.

    Point location was the other O(n_cells)-per-sample cost: testing every
    element for every probe point. The tree turns that into a k-nearest query,
    so a probe costs the same whether the mesh has ten thousand elements or a
    million.
    """
    cached = getattr(mesh, "_locator", None)
    if cached is not None:
        return cached
    pts = mesh.points[mesh.corners]  # (n,4,3) -- corners, so TET10 works unchanged
    M = np.stack(
        [pts[:, 0] - pts[:, 3], pts[:, 1] - pts[:, 3], pts[:, 2] - pts[:, 3]], axis=-1
    )
    cached = (cKDTree(pts.mean(axis=1)), np.linalg.inv(M), pts[:, 3])
    try:
        mesh._locator = cached
    except AttributeError:
        pass
    return cached


def sample_field_at_points(mesh, nodal_values, points, tol=1e-9, k=48):
    """Interpolate a nodal field at arbitrary points inside the mesh.

    Returns an array of sampled values; points outside the mesh come back NaN
    rather than silently clamped -- a probe that leaves the part is a real
    modelling error (R0 larger than the ligament, a bisector aimed the wrong
    way) and must not be papered over.

    Linear barycentric interpolation inside the containing element, exact for
    any field that is linear over that element. Candidate elements come from a
    k-nearest-centroid query, and the barycentric test runs batched over all
    (point, candidate) pairs at once.
    """
    tree, Minv, origin = _cell_locator(mesh)
    nodal_values = np.asarray(nodal_values, float)
    points = np.atleast_2d(np.asarray(points, float))
    k = int(min(k, len(mesh.cells)))

    _d, cand = tree.query(points, k=k)          # (n_pts, k) candidate cell ids
    cand = np.atleast_2d(cand)
    rel = points[:, None, :] - origin[cand]      # (n_pts, k, 3)
    lam3 = np.einsum("pkij,pkj->pki", Minv[cand], rel)
    lam = np.concatenate([lam3, 1.0 - lam3.sum(axis=2, keepdims=True)], axis=2)
    inside = np.all(lam >= -tol, axis=2)         # (n_pts, k)

    out = np.full(len(points), np.nan)
    hit_any = inside.any(axis=1)
    if not hit_any.any():
        return out
    # a point on a shared face lands in several tets; the field is continuous
    # there, so the first hit is as good as any
    first = np.argmax(inside, axis=1)
    rows = np.flatnonzero(hit_any)
    cells = cand[rows, first[rows]]
    weights = lam[rows, first[rows]]             # (m, 4)
    if mesh.cells.shape[1] >= 10:
        # TET10: interpolate with the QUADRATIC shape functions over all ten
        # nodes. Using the four corners with linear weights -- which is what this
        # did -- throws away the six mid-side values entirely, so a TET10 probe
        # returned exactly what the same mesh's corners would have given and the
        # extra nodes bought nothing.
        #
        # Abaqus C3D10 order (what build_mesh stores): 0-3 corners, 4=(0,1)
        # 5=(1,2) 6=(0,2) 7=(0,3) 8=(1,3) 9=(2,3).
        #
        # The barycentric coordinates come from the STRAIGHT-edged tet, so on a
        # curved element this is a first-order-accurate parametrisation of a
        # second-order field rather than an exact isoparametric inverse (which
        # would need a Newton solve per point). The field interpolation itself is
        # exact for any quadratic.
        L = weights                                    # (m, 4), sums to 1
        N = np.empty((len(L), 10))
        N[:, :4] = L * (2.0 * L - 1.0)
        for col, (i, j) in enumerate(
            ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))
        ):
            N[:, 4 + col] = 4.0 * L[:, i] * L[:, j]
        out[rows] = np.einsum("mn,mn->m", N, nodal_values[mesh.cells[cells]])
    else:
        out[rows] = np.einsum("mn,mn->m", weights, nodal_values[mesh.corners[cells]])
    return out


def bisector_profile(mesh, nodal_values, root, direction, distances):
    """Sample a nodal field along a ray from a notch root -- the line plot both
    Point Methods read, and the plot the point-based calibration intersects.

    `root` is the notch root point, `direction` points into the material along
    the bisector (normalised here). Returns values at each distance, NaN where
    the ray has left the part.
    """
    root = np.asarray(root, float)
    direction = np.asarray(direction, float)
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        raise ValueError("direction must be a non-zero vector")
    direction = direction / norm
    distances = np.atleast_1d(np.asarray(distances, float))
    if np.any(distances < 0):
        raise ValueError("distances must be non-negative")
    return sample_field_at_points(mesh, nodal_values, root + np.outer(distances, direction))


def surface_node_ids(mesh):
    """Indices of nodes that lie on the part surface.

    The notch root is on a surface, so a search for the stress concentration
    must be restricted to surface nodes. Searching the whole volume lets the
    root drift inward, which both misplaces the path origin and starts it inside
    material where the field is already decaying.
    """
    return np.unique(np.asarray(mesh.surface_triangles).reshape(-1))


def local_element_size(mesh, node_id):
    """Element size at a node: the median, over the cells that contain it, of
    the mean corner-to-corner edge length (mid-side nodes of a TET10 are not
    corners, so the size means the same thing for both element orders)."""
    cells = np.asarray(mesh.cells)
    corners = cells[np.any(cells == node_id, axis=1)][:, :4]
    if not len(corners):
        raise ValueError(f"node {node_id} belongs to no cell")
    P = mesh.points[corners]                                   # (n, 4, 3)
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    edges = np.stack([np.linalg.norm(P[:, a] - P[:, b], axis=1) for a, b in pairs], axis=1)
    return float(np.median(edges.mean(axis=1)))


# A point method reads the field at a fixed distance from the root. With fewer
# elements than this between the root and that station the value is an
# interpolation inside the first element or two, whatever the fits say.
MIN_ELEMENTS_TO_STATION = 2.0          # the partition recommendation aims for 3; below 2 the read is inside the first element or two
# A blunt root (rho/8 coarser than the nearer station / 3) varies on the scale of rho, not of the stations: the field
# near it is a function of r/rho. recommend_partition and Suggest never go below rho/8 there (PR #48), and assess
# accepts an element at the root up to BLUNT_ROOT_TOLERANCE x that (gmsh's median edge there runs above the requested
# size: 1.10, 1.28 and 1.17 x on SENB rho 1, 2, 0.5 mm, 2026-10-23), so the two rules agree. Evidence that rho/8 is
# enough: SENB rho 0.5-2 mm, within about 1 % of 6x finer meshes (PR #48 run, and the agent test of 2026-10-23).
BLUNT_ROOT_ELEMENTS = 8.0
BLUNT_ROOT_TOLERANCE = 1.4


def blunt_root_floor(rho, near_station_mm):
    """rho / 8 when that is the binding resolution rule at this root (coarser than the nearer station / 3), else None."""
    if not rho or not near_station_mm or rho <= 0:
        return None
    floor = float(rho) / BLUNT_ROOT_ELEMENTS
    return floor if floor > float(near_station_mm) / 3 else None


def snap_to_node(mesh, point, surface_only=True):
    """Move a picked point onto the nearest actual mesh node.

    A click in the viewer raycasts onto a surface TRIANGLE, so the point it
    returns is generally between nodes. The solution is only defined at nodes,
    so the path origin should be one: otherwise the first samples along the ray
    are interpolated from a location that is not a solution point, and the
    reported root is a place the solver never evaluated.

    Returns (node_id, coordinates, distance_moved).
    """
    point = np.asarray(point, float)
    if point.shape != (3,):
        raise ValueError("point must be an (x, y, z) triple")

    candidates = surface_node_ids(mesh) if surface_only else np.arange(mesh.n_nodes)
    offsets = np.linalg.norm(mesh.points[candidates] - point, axis=1)
    best = int(candidates[int(np.argmin(offsets))])
    return best, mesh.points[best].copy(), float(offsets.min())


def snap_to_peak(mesh, field, point, radius, surface_only=True):
    """Move the root to the largest field value near the picked point.

    Measured motivation: on one notch, re-picking by eye moved the predicted
    failure load from about 900 N to 1.5 kN -- a 60% swing from nothing but
    where the user clicked. A path that starts beside the concentration rather
    than on it samples the field RISING before it decays, and a decay law fitted
    through that returns an exponent past its physical limit.

    `radius` bounds the search so the root cannot jump to a different feature.
    Restricted to surface nodes for the reason in `surface_node_ids`.

    Returns (node_id, coordinates, distance_moved, n_candidates).
    """
    point = np.asarray(point, float)
    field = np.asarray(field, float)

    candidates = surface_node_ids(mesh) if surface_only else np.arange(mesh.n_nodes)
    offsets = np.linalg.norm(mesh.points[candidates] - point, axis=1)
    within = candidates[offsets <= float(radius)]

    if within.size == 0:
        # Nothing in range: fall back to the nearest node rather than silently
        # leaving the root off-mesh.
        node_id, coords, moved = snap_to_node(mesh, point, surface_only)
        return node_id, coords, moved, 0

    best = int(within[int(np.argmax(field[within]))])
    moved = float(np.linalg.norm(mesh.points[best] - point))
    return best, mesh.points[best].copy(), moved, int(within.size)


def inward_normal_at(mesh, point):
    """The inward direction into the material at a point on the part surface.

    Taken from gmsh's own outward-wound surface triangles -- the nearest one to
    `point` -- and negated. Used when the caller supplies a feature location but
    not a direction, which is the common case: the notch root is on a surface,
    and "into the material" is not a choice the user should have to express as a
    vector.
    """
    point = np.asarray(point, float)
    tris = mesh.surface_triangles
    a, b, c = (mesh.points[tris[:, i]] for i in range(3))
    centroids = (a + b + c) / 3.0
    near = int(np.argmin(np.linalg.norm(centroids - point, axis=1)))
    normal = np.cross(b[near] - a[near], c[near] - a[near])
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        raise ValueError("degenerate surface triangle at the requested point")
    return -normal / norm


# Below this many integration points in the fit window the regression is not
# worth reading -- measured, not guessed: see path_regression.
MIN_TRUSTED_POINTS = 30
MIN_TRUSTED_R2 = 0.90

# Hard physical ceilings on the fitted decay exponents. A mode-I sharp CRACK
# is the most severe stress concentration that exists: its stress decays as
# r^-0.5, and every V-notch is milder (see williams_exponent). SED goes as
# stress squared, so its exponent is exactly twice and cannot exceed 1.0.
#
# A fit returning more than these is not describing a notch field. In
# practice it means the path did not start at the concentration: it walks in
# from beside the root, crosses the peak, and the fit through a curve that
# RISES before decaying reports an impossible exponent. This check has caught
# cases where R2 was 0.99 -- statistics cannot see it, because a smooth wrong
# field fits a power law beautifully.
MAX_STRESS_EXPONENT = 0.5
MAX_SED_EXPONENT = 1.0

# How far past the limit counts as untrustworthy rather than merely worth
# noting. 1.25 keeps the genuinely wrong paths flagged -- the ones measured
# at 0.577 and 1.382 against limits of 0.5 and 1.0, i.e. 1.15x and 1.38x --
# while not voiding a result whose exponent is mildly inflated by a blunt
# root or a structural gradient.
EXPONENT_EXCESS_UNTRUSTED = 1.25


def quad_point_samples(fe, u, E, nu):
    """Flattened integration-point cloud: coordinates, SED and first principal.

    Returns (points (n, 3), sed (n,), s1 (n,)). One row per integration point,
    NOT per element -- a TET10 with a 4-point rule contributes four independent
    samples, which is the resolution a regression along the extraction path
    needs.
    """
    points, stress = _fea().element_stresses_at_quads(fe, u, E, nu)
    pts = points.reshape(-1, 3)
    sig = stress.reshape(-1, 6)

    C_inv_strain = np.linalg.solve(_fea()._elastic_C(E, nu), sig.T).T
    sed = 0.5 * np.einsum("ni,ni->n", C_inv_strain, sig)

    sxx, syy, szz, sxy, syz, szx = sig.T
    mat = np.empty((len(sig), 3, 3))
    mat[:, 0, 0], mat[:, 1, 1], mat[:, 2, 2] = sxx, syy, szz
    mat[:, 0, 1] = mat[:, 1, 0] = sxy
    mat[:, 1, 2] = mat[:, 2, 1] = syz
    mat[:, 0, 2] = mat[:, 2, 0] = szx
    s1 = np.linalg.eigvalsh(mat)[:, -1]
    return pts, sed, s1


def recovered_nodal_fields(mesh, fe, u, E, nu):
    """Nodal SED and first principal stress by patch recovery, not averaging.

    Returns (sed_nodal, s1_nodal). Falls back per node to the volume-weighted
    average wherever a patch cannot support the fit, so this is never worse than
    the path it replaces.

    Measured against that path on a plate with a hole, probing s1 at the crown
    across five systematic refinements: averaging spread 0.92% and never
    settled; this settles to ~2.557 with a 0.45% spread. The difference matters
    more than the width -- a quantity that wanders cannot yield a convergence
    order, which is why the earlier mesh study came out noise-limited.
    """
    from feakit import recovery

    points, stress = _fea().element_stresses_at_quads(fe, u, E, nu)
    n_cells, n_quads = stress.shape[0], stress.shape[1]
    sig = stress.reshape(-1, 6)

    C_inv_strain = np.linalg.solve(_fea()._elastic_C(E, nu), sig.T).T
    sed_q = (0.5 * np.einsum("ni,ni->n", C_inv_strain, sig)).reshape(n_cells, n_quads)

    sxx, syy, szz, sxy, syz, szx = sig.T
    mat = np.empty((len(sig), 3, 3))
    mat[:, 0, 0], mat[:, 1, 1], mat[:, 2, 2] = sxx, syy, szz
    mat[:, 0, 1] = mat[:, 1, 0] = sxy
    mat[:, 1, 2] = mat[:, 2, 1] = syz
    mat[:, 0, 2] = mat[:, 2, 0] = szx
    s1_q = np.linalg.eigvalsh(mat)[:, -1].reshape(n_cells, n_quads)

    # the averaged path, used as the per-node fallback
    sed_elem, volumes = sed_per_tet(fe, u, E, nu)
    s1_elem = principal_stresses_per_tet(fe, u, E, nu)[:, 0]
    sed_avg = nodal_from_tets(mesh, sed_elem, volumes)
    s1_avg = nodal_from_tets(mesh, s1_elem, volumes)

    sed_nodal, _ = recovery.spr_nodal(mesh, points, sed_q, fallback=sed_avg)
    s1_nodal, _ = recovery.spr_nodal(mesh, points, s1_q, fallback=s1_avg)
    return sed_nodal, s1_nodal


def failure_load_from_probe(probed, applied_N, sigma_uts=None, W_c=None,
                            energy=False):
    """Load at which a probed field value meets the criterion.

    Linear elasticity: stress scales with load, so s1/sigma_UTS is linear in it.
    An energy density scales with load SQUARED, so sqrt(W/W_c) is linear in it
    too. Either way the criterion is met at applied / ratio.

    This is the number the assessment is FOR. Comparing two estimates of the
    probed field overstates their disagreement for the energy method, because
    the square root halves it: a SED estimate 8% low predicts a load only 4%
    high. Estimates should be compared where the decision is made.
    """
    probed = float(probed)
    if not np.isfinite(probed) or probed <= 0 or not applied_N:
        return None, None
    if energy:
        if not W_c or W_c <= 0:
            return None, None
        ratio = float(np.sqrt(probed / W_c))
    else:
        if not sigma_uts or sigma_uts <= 0:
            return None, None
        ratio = probed / float(sigma_uts)
    if ratio <= 0:
        return None, None
    return float(applied_N) / ratio, ratio


def williams_exponent(opening_angle_deg=0.0):
    """Mode-I stress singularity exponent for a sharp V-notch.

    Solves the Williams eigenvalue equation for a wedge of half-angle
    g = pi - a, where 2a is the notch OPENING angle:

        sin(2*lam*g) + lam*sin(2*g) = 0

    and returns the decay exponent 1 - lam, which is what a power-law fit to
    sigma(r) along the bisector measures. A crack (opening 0) gives exactly 0.5;
    a 60-degree V-notch gives 0.4878; 90 degrees gives 0.4555.

    This is the reference a fitted exponent should be compared against on a
    SHARP notch, where Creager-Paris does not apply -- there is no root radius
    to fit, so a fitted rho is reporting mesh resolution, not geometry.
    """
    from scipy.optimize import brentq

    a = np.radians(float(opening_angle_deg)) / 2.0
    if a <= 0:
        return 0.5
    g = np.pi - a
    lam = brentq(lambda L: np.sin(2 * L * g) + L * np.sin(2 * g), 0.35, 0.999)
    return float(1.0 - lam)


def singularity_consistency(sed_regression, stress_regression):
    """Cross-check two regressions that describe the SAME singular field.

    Requires no knowledge of the geometry, which is what makes it useful: an
    energy density goes as stress squared, so whatever the notch angle, the
    fitted SED exponent must be TWICE the fitted stress exponent. A departure
    is a statement about the mesh, not about the material.

    Likewise a fitted Creager-Paris rho is a geometric root radius: both fields
    must return the same one. Two different answers mean rho is not being
    measured -- on a sharp notch there is no radius to measure and the fit
    returns the smallest length the mesh resolves.

    Returns None when either regression is missing.
    """
    if not sed_regression or not stress_regression:
        return None
    out = {}
    ps, pt = sed_regression.get("power"), stress_regression.get("power")
    if ps and pt and pt.get("exponent"):
        ratio = float(ps["exponent"]) / float(pt["exponent"])
        out["lambda_ratio"] = ratio
        out["lambda_ratio_expected"] = 2.0
        out["lambda_ratio_pct"] = 100.0 * (ratio - 2.0) / 2.0
    cs, ct = sed_regression.get("creager"), stress_regression.get("creager")
    if cs and ct and ct.get("rho"):
        rr = float(cs["rho"]) / float(ct["rho"])
        out["rho_ratio"] = rr
        out["rho_sed"] = float(cs["rho"])
        out["rho_stress"] = float(ct["rho"])
        # a genuine root radius is one number; a factor of 1.5 apart is not noise
        out["rho_disagrees"] = bool(rr > 1.5 or rr < 1 / 1.5)
    return out or None


def creager_paris(r, K_I, rho, squared=False):
    """Creager-Paris blunt-notch field along the bisector.

    The crack-tip solution with its origin moved rho/2 BEHIND the notch root, so
    with r measured from the root and s = r + rho/2:

        sigma_yy(s) = K_I / sqrt(2*pi*s) * (1 + rho/(2s))

    At the root (r = 0, s = rho/2) this gives the known 2*K_I/sqrt(pi*rho), and
    as rho -> 0 it collapses to the sharp-crack 1/sqrt(r). `squared=True` returns
    the shape for an energy density, which goes as stress squared.

    This is an ASYMPTOTIC notch field: it describes a deep notch whose near-root
    field is dominated by the singular term. It is not the right field for a
    circular hole in a wide plate -- that is Kirsch, which has r^-2 and r^-4
    terms and no singularity at all.
    """
    r = np.asarray(r, float)
    s = r + 0.5 * rho
    f = (1.0 / np.sqrt(2.0 * np.pi * s)) * (1.0 + rho / (2.0 * s))
    return K_I * f * f if squared else K_I * f


def creager_paris_fit(r, v, rho=None, squared=False, rho_bounds=None, n_rho=400):
    """Fit the Creager-Paris field to samples along the bisector.

    Returns {K_I, rho, r2, value_at} or None. The model is LINEAR in the
    amplitude for any fixed rho, so this is a 1-D search over rho with a closed
    form least squares inside it -- no optimiser, no starting guess, no failure
    to converge.

    Pass `rho` to hold the notch root radius at its CAD value, which is the
    honest thing to do when the geometry is known: then the only fitted quantity
    is K_I. Leave it None to fit rho as well, in which case comparing the fitted
    rho against the CAD radius is itself a check on whether the notch field
    model applies at all.
    """
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    if len(r) < 4:
        return None

    def amplitude_and_r2(rho_try):
        s = r + 0.5 * rho_try
        f = (1.0 / np.sqrt(2.0 * np.pi * s)) * (1.0 + rho_try / (2.0 * s))
        basis = f * f if squared else f
        denom = float(basis @ basis)
        if denom <= 0:
            return None
        amp = float(basis @ v) / denom
        resid = v - amp * basis
        ss = float(((v - v.mean()) ** 2).sum())
        r2 = 1.0 - float(resid @ resid) / ss if ss > 0 else float("nan")
        return amp, r2

    if rho is not None:
        got = amplitude_and_r2(float(rho))
        if got is None:
            return None
        amp, r2 = got
        best = (float(rho), amp, r2)
    else:
        lo, hi = rho_bounds or (0.02 * float(r.max()), 20.0 * float(r.max()))
        best = None
        for rho_try in np.geomspace(lo, hi, int(n_rho)):
            got = amplitude_and_r2(float(rho_try))
            if got is None:
                continue
            amp, r2 = got
            if amp <= 0:          # a negative amplitude is not a physical field
                continue
            if best is None or r2 > best[2]:
                best = (float(rho_try), amp, r2)
        if best is None:
            return None
    return {"rho": best[0], "K_I": best[1], "r2": best[2]}


def _collect_along_ray(points, values, root, direction, r_max, tube):
    """Integration points lying within `tube` of the ray, with their distance.

    Returns (r, v) sorted by r, keeping only points ahead of the root. Selecting
    a TUBE rather than sampling a line is what makes this use real data: no
    integration point lies exactly on an arbitrary ray, so any line probe must
    interpolate. Here the samples are used where they actually are, and their
    off-axis offset is bounded by `tube`.
    """
    rel = points - root
    t = rel @ direction                              # distance along the ray
    perp = np.linalg.norm(rel - t[:, None] * direction, axis=1)
    keep = (t > 0) & (t <= r_max) & (perp <= tube)
    order = np.argsort(t[keep])
    return t[keep][order], np.asarray(values)[keep][order]


def path_regression(points, values, root, direction, r_eval, r_max=None,
                    tube=None, window=(0.4, 2.5), min_points=6,
                    rho=None, squared=False, series_terms=4, series_rho=None, near_max=None,
                    r_valid=None, williams=None):
    """Regress an integration-point field along the extraction path.

    Fits BOTH forms over the samples bracketing `r_eval` and returns both,
    because where they disagree is exactly where the sampling is not yet
    mesh-converged:

    * `power`  -- log(v) against log(r). Near a notch root the stress goes as
      r^-lambda and the SED as r^-2*lambda, so a power law is straight in
      log-log while a straight line in r is not. The fitted exponent is a free
      diagnostic: it should sit near the expected singularity order.
    * `linear` -- ordinary least squares of v against r over the same window.
      Assumes nothing about the singular form, but is only defensible over a
      short span, so it is reported with its own R^2 rather than trusted.

    `window` scales `r_eval` to set the fit range; the default brackets the
    probe distance rather than starting at the root, which is singular and must
    never be sampled.

    Returns None when too few integration points fall in the window -- a real
    answer, meaning the mesh cannot support the regression there, not a number
    to be trusted anyway.
    """
    direction = np.asarray(direction, float)
    direction = direction / np.linalg.norm(direction)
    r_eval = float(r_eval)
    lo, hi = float(window[0]) * r_eval, float(window[1]) * r_eval
    if r_max is None:
        r_max = hi
    if tube is None:
        tube = 0.5 * r_eval

    r, v = _collect_along_ray(points, values, np.asarray(root, float),
                              direction, r_max, tube)
    band = (r >= lo) & (r <= hi)
    r, v = r[band], v[band]
    # Samples closer to the root than `r_valid` are kept for the plot but not
    # fitted: there the mesh does not resolve the field (see the caller), and
    # a smooth series cannot follow what it does instead.
    all_r, all_v = r, v
    keep = r >= float(r_valid) if r_valid else np.ones(len(r), bool)
    r, v = r[keep], v[keep]
    if len(r) < min_points:
        return None

    # the samples themselves, so a caller can PLOT what was fitted rather than
    # take the fit on trust -- a fit whose scatter is not shown is unfalsifiable
    out = {"n_points": int(len(r)), "r_min": float(r.min()), "r_max": float(r.max()),
           "window": [lo, hi], "tube": float(tube), "samples_r": all_r, "samples_v": all_v,
           "samples_fitted": keep, "r_valid": float(r_valid) if r_valid else None}

    # Creager-Paris: the analytic blunt-notch field, not a curve of convenience.
    # A straight line in r used to sit here as the second fit; it was dropped
    # because nothing about a notch field is linear, so its disagreement with the
    # power law carried no information about the mesh -- only about the fact that
    # a line cannot describe a decay.
    # The single-term Creager-Paris and the power law are ASYMPTOTIC near-root
    # forms: over a path that runs several process zones out into a bending
    # ligament they cannot follow the structural gradient, and fitting them
    # there biases the near field they are meant to describe. They are fitted
    # over the near field r <= near_max (the process zone d = 2L by default);
    # the series, which has the terms for the far field, over the whole path.
    near = float(near_max) if near_max is not None else hi
    nm = r <= near
    rn, vn = r[nm], v[nm]
    out["near_max"] = near
    out["n_points_near"] = int(len(rn))
    cp = creager_paris_fit(rn, vn, rho=rho, squared=squared) if len(rn) >= min_points else None
    if cp is not None:
        out["creager"] = {
            "value": float(creager_paris(r_eval, cp["K_I"], cp["rho"], squared=squared)),
            "K_I": cp["K_I"],
            "rho": cp["rho"],
            "rho_fitted": rho is None,
            "r2": cp["r2"],
            "n_points": int(len(rn)),
        }
    else:
        out["creager"] = None

    # The Creager-Paris SERIES over the same samples: leading blunt-notch term
    # plus T-stress and higher Williams terms, origin shift from the CAD rho or
    # the single-term fit, conditioning enforced (see creager_paris_series_fit).
    # `series_rho` (the measured root radius) holds the series origin shift:
    # left free, rho trades off against the higher-order terms (measured on
    # the SENB: free rho 0.39 mm and K_I 39% above E399; at the CAD 0.25 mm,
    # K_I within 3%).
    out["series"] = creager_paris_series_fit(r, v, series_rho if series_rho is not None else rho,
                                             n_terms=series_terms, squared=squared,
                                             williams=williams)    # the stress exponent: SED is fitted as its square
    if out["series"]:
        out["series"]["value"] = float(creager_paris_series(np.array([r_eval]), out["series"], squared=squared)[0])

    # power law -- only over strictly positive samples, since log is undefined
    # otherwise and a compressive s1 is a legitimate value, not an error
    pos = vn > 0
    if pos.sum() >= min_points:
        lr, lv = np.log(rn[pos]), np.log(vn[pos])
        Ap = np.column_stack([np.ones_like(lr), lr])
        cp, *_ = np.linalg.lstsq(Ap, lv, rcond=None)
        predp = Ap @ cp
        ssp = float(((lv - lv.mean()) ** 2).sum())
        out["power"] = {
            "value": float(np.exp(cp[0]) * r_eval ** cp[1]),
            "exponent": float(-cp[1]),        # reported as lambda in r^-lambda
            "r2": float(1.0 - ((lv - predp) ** 2).sum() / ssp) if ssp > 0 else float("nan"),
            "n_points": int(pos.sum()),
        }
    else:
        out["power"] = None

    # Whether the fit is worth reading at all. Measured on the Kirsch case
    # (TET10, analytic truth): 14 points in the window gave -37%, 38 points gave
    # +3.3%, and 636 and 4536 points both gave +3.3%. So the answer converges
    # from a few tens of points, and below that the numbers are noise wearing a
    # decimal point. Reporting them unqualified reads as a disagreement between
    # methods when it is really an under-resolved mesh.
    # Does the field actually DECAY from the root, or rise first?
    #
    # This is the direct test for the failure mode the exponent check was
    # reaching at indirectly. A path that starts beside the stress concentration
    # rather than on it walks in, crosses the peak, and then decays -- and a
    # decay law fitted through that returns a confident, meaningless exponent at
    # excellent R^2. Measured: one such path scored R^2 = 0.99.
    #
    # Binned medians rather than raw samples, because integration-point scatter
    # at a single radius is large and a single high sample near the root would
    # otherwise read as a rise.
    rising = False
    if len(r) >= 20:
        edges = np.linspace(r.min(), r.max(), 6)
        medians = []
        for lo_e, hi_e in zip(edges[:-1], edges[1:]):
            sel = (r >= lo_e) & (r <= hi_e)
            if sel.sum() >= 3:
                medians.append(float(np.median(v[sel])))
        if len(medians) >= 3:
            # the field should be largest closest to the root
            rising = medians[0] < max(medians) * 0.97

    reasons = []
    if rising:
        reasons.append(
            "The stress rises before it falls, so the path starts beside the "
            "notch root rather than on it. Move the feature point onto the peak.")
        out["profile_rises"] = True
    if out["n_points"] < MIN_TRUSTED_POINTS:
        reasons.append(
            f"Only {out['n_points']} samples along the path, and "
            f"{MIN_TRUSTED_POINTS} are needed for a stable fit. Refine the mesh "
            f"near the feature."
        )
    best_r2 = max([f["r2"] for f in (out.get("power"), out.get("creager"))
                   if f and np.isfinite(f["r2"])], default=float("nan"))
    if np.isfinite(best_r2) and best_r2 < MIN_TRUSTED_R2:
        reasons.append(
            f"The curve fit is poor (R2 {best_r2:.2f}, {MIN_TRUSTED_R2} or "
            f"better expected). Usually too few samples, or elements too coarse "
            f"near the root.")

    # The physics gate. Deliberately checked separately from R2: the two
    # disagree, and when they do the physics is right. Observed repeatedly on
    # one notch -- the path with the best R2 (0.99) was the one violating the
    # exponent limit, while the physically consistent path was the one flagged
    # for low R2.
    #
    # The limit applies to the asymptotic notch field IN ISOLATION. Two things
    # legitimately push a fitted exponent past it, and neither means the path is
    # wrong:
    #
    #  * a blunt root, where the field is finite at the notch and the fit window
    #    reaches into the transition zone rather than the singular one;
    #  * a structural stress gradient superimposed on the notch field. In a deep
    #    notch with a small remaining ligament the bending gradient is not small
    #    over the fit window, and the two decays add.
    #
    # Observed on a SENB specimen with a 5 mm ligament and a window running to
    # 0.58 mm -- over 10% of the ligament -- where four independent fits agreed
    # within 4% at R^2 > 0.98 while the exponent read 0.618. Calling that "not a
    # notch field" was wrong, so the excess is reported as something to
    # interpret, and only a LARGE excess is treated as untrustworthy.
    limit = MAX_SED_EXPONENT if squared else MAX_STRESS_EXPONENT
    fitted = (out.get("power") or {}).get("exponent")
    if fitted is not None and np.isfinite(fitted) and fitted > limit:
        what = "SED" if squared else "Stress"
        out["exponent_unphysical"] = True
        excess = fitted / limit
        message = (
            f"{what} falls off as r^-{fitted:.2f}, steeper than the {limit} a "
            f"notch gives on its own")
        # Informational, never on its own a reason to distrust the result:
        # whether the path started at the concentration is tested directly by
        # the rising-profile check above, which is what this was reaching at.
        # Just the measurement. The shared explanation is added once by the
        # caller, which sees both methods -- repeating it per method printed the
        # same paragraph twice with only the field name changed.
        out.setdefault("notes", []).append(message + ".")
    out["trusted"] = not reasons
    out["warnings"] = reasons
    return out


def probe_path_from_point(mesh, point, direction=None, snap=None):
    """Build the extraction path from the user-supplied feature location.

    The location is an INPUT -- the notch the user identified -- not something
    discovered from the solved field. The same point defines the mesh partition
    (see fea_gmsh.Refinement), so it is stated once and everything is built from
    it.

    Returns (root, direction). If `direction` is omitted it is the inward
    surface normal at the point; pass one explicitly for a notch bisector that
    is not simply normal to the surface.
    """
    root = np.asarray(point, float)
    if root.shape != (3,):
        raise ValueError("point must be an (x, y, z) triple")

    if snap is not None:
        root = np.asarray(snap, float)

    if direction is None:
        return root, inward_normal_at(mesh, root)
    direction = np.asarray(direction, float)
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        raise ValueError("direction must be a non-zero vector")
    return root, direction / norm


def point_method(mesh, nodal_values, root, direction, distance):
    """The Point Method probe: one field value at one distance along the bisector.

    Stress-based TCD PM passes the nodal first principal stress and distance
    L/2; the Energy PM passes nodal SED and distance R0^PM. Same probe, and that
    is the point -- they differ only in the field and the length scale.
    """
    value = bisector_profile(mesh, nodal_values, root, direction, [distance])[0]
    if np.isnan(value):
        raise ValueError(
            f"probe at distance {distance} left the mesh; check the bisector "
            "direction points into the material and the length scale fits the ligament"
        )
    return float(value)


def averaged_sed(w, volumes, mask):
    """Volume-weighted average of element SED over a control domain:

        W_bar = sum_(e in D) U_e / sum_(e in D) V_e ,   U_e = w_e V_e

    `mask` is a boolean array over elements selecting the domain. Numerator and
    denominator necessarily share it -- using different inclusion rules for the
    two biases W_bar directly, which is the most common way to get this wrong.
    """
    w, volumes, mask = np.asarray(w), np.asarray(volumes), np.asarray(mask, bool)
    if not mask.any():
        raise ValueError("control domain selected no elements")
    v = volumes[mask]
    return float((w[mask] * v).sum() / v.sum())


def failure_ratio(W_bar, W_c):
    """sqrt(W_bar/W_c): the load-scale factor at which failure is predicted.

    The square root makes this directly comparable to a stress ratio (W is
    quadratic in stress), and it is what the paper plots as its accuracy
    measure. >= 1 predicts failure.
    """
    if W_c <= 0:
        raise ValueError("W_c must be positive")
    return float(np.sqrt(W_bar / W_c))


def assess_point_method(
    mesh, fe, u, point, E, nu, K_Ic, sigma_uts, state=PLANE_STRAIN, energy=True,
    direction=None, profile_at=None, nodal_field=None, snap="peak",
    snap_radius=None,
):
    """Full Point Method assessment at a user-identified feature location.

    `fe` is the JAX-FEM FiniteElement returned by the solve.
    `point` is the notch the user pointed at -- the same location that defined
    the mesh partition. `direction` defaults to the inward surface normal there.

    `energy=True` runs the Energy PM: SED probed at R0^PM, reported as
    sqrt(W/W_c). `energy=False` runs the classical stress-based TCD PM: first
    principal stress probed at L/2, reported as s1/sigma_UTS. Both return a
    dimensionless ratio on the same scale, where >= 1 predicts failure.

    The two agree closely by construction: at phi=0 on the bisector, sqrt(W) and
    s1 differ only by a material constant, absorbed by the difference between
    sqrt(W_c) and sigma_UTS. They diverge where the stress state is not uniaxial
    -- which is the point of the energy method, since it carries the
    multiaxiality that s1 alone cannot.

    Returns a dict with the verdict and everything needed to plot or audit it:
    ratio, distance, probed value, root, direction, and (if `profile_at` gives
    distances) the field sampled along the path -- the line plot that the
    point-based calibration intersects.
    """
    # `nodal_field` lets a caller supply the field it already has -- the solve
    # runs in a separate process, so the parent holds finished nodal fields but
    # not the FiniteElement needed to recompute them.
    if energy:
        distance = R0_energy_pm(K_Ic, sigma_uts, nu, state)
        if nodal_field is None:
            w, volumes = sed_per_tet(fe, u, E, nu)
            nodal_field = nodal_from_tets(mesh, w, volumes)
    else:
        distance = 0.5 * L_tcd(K_Ic, sigma_uts)
        if nodal_field is None:
            _s, volumes = _fea().element_stresses(fe, u, E, nu)
            nodal_field = nodal_from_tets(
                mesh, principal_stresses_per_tet(fe, u, E, nu)[:, 0], volumes)
    field = np.asarray(nodal_field, float)

    # Snap the path origin onto a real node, and preferably onto the stress
    # concentration itself. A viewer click lands on a surface TRIANGLE, so the
    # raw point is generally between nodes and often beside the hot spot
    # rather than on it. Measured consequence of not doing this: re-picking
    # the same notch by eye moved the predicted failure load from about 900 N
    # to 1.5 kN, and produced decay exponents past their physical limit
    # because the path crossed the peak instead of starting at it.
    snap_info = None
    snapped = None
    if snap:
        node_id, coords, moved = snap_to_node(mesh, point)
        snap_info = {"mode": "node", "node_id": node_id,
                     "moved_mm": moved, "from": list(map(float, np.asarray(point, float)))}
        if snap == "peak":
            # Search within the probe distance: far enough to find the true
            # root when the click was close, tight enough that it cannot
            # wander to a different feature.
            radius = float(snap_radius) if snap_radius else float(distance)
            node_id, coords, _from_node, n_near = snap_to_peak(
                mesh, field, coords, radius)
            # Report the distance from the point the USER picked, not from the
            # intermediate nearest-node. snap_to_peak measures from whatever it
            # was handed, and labelling that as 'moved from the picked point'
            # under-reports the move -- it read 0.000 mm on a root that had
            # actually shifted 0.012 mm.
            moved = float(np.linalg.norm(coords - np.asarray(point, float)))
            snap_info.update({"mode": "peak", "node_id": node_id,
                              "moved_mm": moved, "search_radius_mm": radius,
                              "candidates": n_near})
        snapped = coords
        snap_info["to"] = list(map(float, coords))
        snap_info["field_at_root"] = float(field[node_id])

    root, direction = probe_path_from_point(mesh, point, direction, snap=snapped)
    probed = point_method(mesh, field, root, direction, distance)
    ratio = failure_ratio(probed, critical_sed(sigma_uts, E)) if energy else probed / sigma_uts
    out = {
        "ratio": ratio,
        "distance": distance,
        "probed": probed,
        "root": root,
        "direction": direction,
        "method": "energy_pm" if energy else "tcd_pm",
        "snap": snap_info,
    }
    if profile_at is not None:
        out["profile_distances"] = np.asarray(profile_at, float)
        out["profile"] = bisector_profile(mesh, field, root, direction, profile_at)
    return out

# ---------------------------------------------------------------------------
# Extraction path: where it starts and which way it goes.
#
# The direction is the inward surface normal at the origin (the FKM relative
# stress gradient is defined along the surface normal at the hotspot, and for
# a symmetric notch that is the bisector). What changed here:
#
#   * the normal comes from the CURVED TRI6 surface, not the nearest flat
#     facet. On a 0.25 mm root the flat-facet normal at the peak node was
#     measured 6.7 deg off the bisector; the P2 surface map has the curvature.
#   * the solved field's steepest-descent direction is computed too and the
#     angle between the two is reported. They agree on a symmetric notch;
#     when they do not, either the origin is on an edge or the loading is
#     mixed, and the assessment says so instead of silently picking one.
#   * the origin can be LOCATED from the field: surface sigma_1 peaks with the
#     BC and load faces excluded, because contact and load edges are singular
#     and always win a naive argmax (measured: the coarse-mesh von Mises peak
#     on senb_rho025 sat on a support-pad corner, not the notch).
#   * a station gate: a probe that reads a few percent of the root value is
#     outside the notch-affected zone -- the material's L is too long for the
#     ligament -- and the point method is then not measuring the notch.
# ---------------------------------------------------------------------------

def _tri6_dshape(u, v):
    """d/du and d/dv of the six P2 weights in tri6_shape_weights' order."""
    l0, l1, l2 = 1.0 - u - v, u, v
    du = np.array([-(4.0 * l0 - 1.0), 4.0 * l1 - 1.0, 0.0, 4.0 * (l0 - l1), 4.0 * l2, -4.0 * l2])
    dv = np.array([-(4.0 * l0 - 1.0), 0.0, 4.0 * l2 - 1.0, -4.0 * l1, 4.0 * l1, 4.0 * (l0 - l2)])
    return du, dv


_TRI6_NODE_UV = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5))


def surface_normal_at_node(mesh, node_id):
    """Inward unit normal at a surface node from the curved TRI6 faces.

    Every TRI6 that contains the node contributes the normal of its P2 map at
    the node's own parametric location, oriented like gmsh's outward flat
    winding; contributions are area-weighted. Falls back to the flat-facet
    average when the mesh has no TRI6 surface (TET4 meshes).
    """
    pts = mesh.points
    tri6 = getattr(mesh, "surface_tri6", None)
    acc = np.zeros(3)
    if tri6 is not None and len(tri6):
        tri6 = np.asarray(tri6)
        for conn in tri6[np.any(tri6 == node_id, axis=1)]:
            k = int(np.where(conn == node_id)[0][0])
            u, v = _TRI6_NODE_UV[k]
            du, dv = _tri6_dshape(u, v)
            P = pts[conn]
            n = np.cross(du @ P, dv @ P)
            flat = np.cross(P[1] - P[0], P[2] - P[0])
            if np.dot(n, flat) < 0:
                n = -n
            acc += n            # |n| ~ local area: area weighting for free
    else:
        tris = np.asarray(mesh.surface_triangles)
        for t in tris[np.any(tris == node_id, axis=1)]:
            a, b, c = pts[t]
            acc += np.cross(b - a, c - a)
    norm = np.linalg.norm(acc)
    if norm < 1e-14:
        raise ValueError(f"node {node_id} is not on the surface")
    return -acc / norm


def field_gradient_at_node(mesh, field, node_id, rings=2):
    """Least-squares gradient of a nodal field at a node, over its element patch.

    Fits f = f0 + g . (x - x0) to the nodes within `rings` element rings of
    the node. Returns (g, unit steepest-descent direction). Two rings on a
    TET10 mesh gives a few dozen samples, enough for a stable plane through a
    field that is singular one node away.
    """
    cells = np.asarray(mesh.cells)
    field = np.asarray(field, float)
    patch = {int(node_id)}
    for _ in range(rings):
        rows = np.any(np.isin(cells, list(patch)), axis=1)
        patch |= set(map(int, np.unique(cells[rows])))
    ids = np.array(sorted(patch))
    X = mesh.points[ids] - mesh.points[node_id]
    A = np.hstack([np.ones((len(ids), 1)), X])
    coef, *_ = np.linalg.lstsq(A, field[ids], rcond=None)
    g = coef[1:]
    norm = np.linalg.norm(g)
    if norm < 1e-14:
        return g, None
    return g, -g / norm


def path_directions(mesh, field, node_id):
    """Normal and gradient directions at a node, and the angle between them."""
    normal = surface_normal_at_node(mesh, node_id)
    _g, descent = field_gradient_at_node(mesh, field, node_id)
    out = {"normal": [float(v) for v in normal], "gradient": None, "angle_deg": None,
           "gradient_points_outward": None}
    if descent is not None:
        cosang = float(np.clip(np.dot(normal, descent), -1.0, 1.0))
        out["gradient"] = [float(v) for v in descent]
        out["angle_deg"] = float(np.degrees(np.arccos(cosang)))
        out["gradient_points_outward"] = bool(cosang < 0)
    return out


def locate_hotspots(mesh, field, exclude_face_ids=(), band_mm=0.0, n=5, cluster_mm=None, exclude_zones=()):
    """Surface nodes ranked by a nodal field, with contact and load faces excluded.

    `exclude_face_ids` are CAD face ids carrying a BC or a load; every node on
    them, plus any surface node within `band_mm` of them, is dropped before
    ranking. `exclude_zones` are (points, radius) round edge and point targets. Candidates are greedily clustered so one feature yields one
    entry. Each entry carries the node, its coordinates, the field value, the
    CAD face id it lies on, and the path directions at it.
    """
    field = np.asarray(field, float)
    surf = surface_node_ids(mesh)
    excluded = set()
    for fid in exclude_face_ids:
        excluded |= set(map(int, np.asarray(mesh.face_nodes.get(int(fid), []))))
    keep = np.array([int(i) for i in surf if int(i) not in excluded])
    if band_mm > 0 and excluded and keep.size:
        tree = cKDTree(mesh.points[sorted(excluded)])
        d, _ = tree.query(mesh.points[keep])
        keep = keep[d > band_mm]
    # edge and point supports/loads: the surface within their disturbed zone (points, radius)
    for pts, radius in exclude_zones:
        if keep.size:
            d, _ = cKDTree(np.asarray(pts, float)).query(mesh.points[keep])
            keep = keep[d > float(radius)]
    if keep.size == 0:
        return []
    order = keep[np.argsort(field[keep])[::-1]]
    if cluster_mm is None:
        bbox = mesh.points.max(axis=0) - mesh.points.min(axis=0)
        cluster_mm = 0.05 * float(np.linalg.norm(bbox))
    face_of = {}
    sf = getattr(mesh, "surface_face_ids", None)
    if sf is not None:
        for tri, fid in zip(np.asarray(mesh.surface_triangles), np.asarray(sf)):
            for nid in tri:
                face_of.setdefault(int(nid), int(fid))
    picked, out = [], []
    for nid in order:
        p = mesh.points[nid]
        if any(np.linalg.norm(p - q) < cluster_mm for q in picked):
            continue
        picked.append(p)
        entry = {"node_id": int(nid), "point": [float(v) for v in p], "value": float(field[nid]),
                 "face_id": face_of.get(int(nid))}
        try:
            entry["directions"] = path_directions(mesh, field, int(nid))
        except ValueError as exc:
            entry["directions"] = {"error": str(exc)}
        out.append(entry)
        if len(out) >= n:
            break
    return out


# A probe reading below this fraction of the root value is outside the zone
# the notch controls: the material length scale is too long for the ligament
# and the point method is measuring the structure, not the feature.
STATION_MIN_FRACTION_OF_ROOT = 0.10


def station_check(root_value, probed, distance, label):
    """One line of judgement on where a station sits, or None when it is fine."""
    if root_value is None or not np.isfinite(root_value) or root_value <= 0:
        return None
    frac = float(probed) / float(root_value)
    if frac < STATION_MIN_FRACTION_OF_ROOT:
        return (f"{label} station at {distance:.3f} mm reads {100 * frac:.1f}% of the root value: "
                f"outside the notch-affected zone, so the method is probing the structure, not the feature. "
                f"The material's length scale is too long for this ligament.")
    return None


# ---------------------------------------------------------------------------
# MATS: the extraction path from the solved field (Matvienko & Semenova 2015,
# DOI 10.3221/IGF-ESIS.34.27, extended to a 3-D TET10 hotspot).
#
# Along candidate rays from the origin, the stress tensor is averaged over a
# length d (the paper: the process zone 2L; the app passes L, see mats_path)
# and the crack plane containing the ray with the
# largest averaged normal stress is found (lambda_max of the averaged tensor
# projected onto the ray's perpendicular plane). The ray with the largest such
# value is the extraction path, its eigenvector the crack-plane normal.
#
# Two things the 2-D paper does not have to say, both measured here first:
#   * rays are cast in the plane NORMAL TO THE CRACK FRONT. A hemisphere on a
#     notch root picks the ray along the root line (every point on it is at
#     peak stress: 1.10 vs 0.51 MPa/N, loads 44 N against 92 N) -- that is
#     the front, not a path. The front tangent is the direction along which
#     the surface normal does not change, estimated from the normals on the
#     feature's own CAD face.
#   * origins are the located peak node and its surface neighbours within
#     d/2, because the peak node's own normal can lean 12 deg on a coarse
#     curved root and the sweep from a neighbour recovers the bisector to 2.
# Verified on senb_rho025..rho2 (bisector within 3 deg, loads within 1%) and
# on in-plane shear (deflects toward the shear); out-of-plane shear near a
# free surface is within 8 deg and 2%, limited by the front estimate.
# ---------------------------------------------------------------------------

def front_tangent(mesh, node_id, radius):
    """Local crack-front direction at a surface node, with an anisotropy score.

    Surface nodes within `radius` ON THE SAME CAD FACES as the node give a
    scatter of curved-surface normals; the eigenvector of their covariance
    with the smallest eigenvalue is the direction along which the normal does
    not change -- the front. anisotropy = 1 - lambda_min/lambda_mid: near 1 a
    clear front (a root line, a fillet axis); near 0 a point-like feature.
    Returns (tau, anisotropy, n_normals).
    """
    surf = surface_node_ids(mesh)
    p0 = mesh.points[node_id]
    near = surf[np.linalg.norm(mesh.points[surf] - p0, axis=1) <= radius]
    sf = getattr(mesh, "surface_face_ids", None)
    if sf is not None:
        tris = np.asarray(mesh.surface_triangles)
        sf = np.asarray(sf)
        own = {int(f) for f in sf[np.any(tris == node_id, axis=1)]}
        on_face = set(map(int, np.unique(tris[np.isin(sf, list(own))])))
        near = np.array([n for n in near if int(n) in on_face])
    if len(near) < 6:
        raise ValueError(f"only {len(near)} surface normals near node {node_id}; cannot estimate a front")
    N = np.array([surface_normal_at_node(mesh, int(n)) for n in near])
    # Only the TANGENTIAL change of the normal measures curvature (it is the
    # shape operator acting on the offsets); its component along the normal
    # itself is second order, 1 - cos(theta). The full 3x3 covariance used to
    # be decomposed here, and on a root curved both ways -- a groove round a
    # bar, a shaft fillet -- that second-order variance along the normal was
    # as small as the variance along the front, so the "front" came out tilted
    # towards the normal (6.5 deg on a r 1 groove in a r 10 bar) or ON it (a
    # shoulder fillet; MATS then searched a plane containing no front at all).
    # Decompose in the tangent plane instead: the front is the principal
    # direction of least curvature.
    n0 = surface_normal_at_node(mesh, node_id)
    e1 = np.cross(n0, [1.0, 0, 0] if abs(n0[0]) < 0.9 else [0, 1.0, 0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n0, e1)
    T = np.c_[N @ e1, N @ e2]
    w, v = np.linalg.eigh(np.cov(T.T))
    tau = v[0, 0] * e1 + v[1, 0] * e2
    tau = tau / np.linalg.norm(tau)
    aniso = float(1.0 - (w[0] / w[1] if w[1] > 1e-18 else 1.0))
    return tau, aniso, int(len(near))


def root_radius_at_node(mesh, node_id, tau, radius, min_angle_deg=1.0):
    """Radius of curvature of the surface at a node, across the front `tau`,
    measured from the mesh: on an arc of radius rho the surface normal turns
    by theta over an arc length rho*theta, so rho = |x| / theta for the
    neighbouring surface nodes on the same CAD face, with x the offset
    projected into the plane normal to tau. Least squares through the origin.

    Returns (rho, r2, n_used) or None on a flat surface (no normal turns by
    `min_angle_deg`). This is what the Creager-Paris series holds its origin
    shift at when the part does not state its root radius: geometry, not a
    quantity traded off against the higher-order terms of a fit.
    """
    tau = np.asarray(tau, float); tau = tau / np.linalg.norm(tau)
    p0 = mesh.points[node_id]
    n0 = surface_normal_at_node(mesh, node_id)
    n0p = n0 - (n0 @ tau) * tau; n0p /= np.linalg.norm(n0p)
    xs, ths = [], []
    for nid in _own_face_nodes_near(mesh, node_id, radius):
        if nid == node_id:
            continue
        x = mesh.points[nid] - p0; x = x - (x @ tau) * tau
        n = surface_normal_at_node(mesh, int(nid)); n = n - (n @ tau) * tau
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        th = float(np.arccos(np.clip(n @ n0p, -1.0, 1.0)))
        if np.degrees(th) >= min_angle_deg:
            xs.append(float(np.linalg.norm(x))); ths.append(th)
    if len(ths) < 3:
        return None
    xs, ths = np.asarray(xs), np.asarray(ths)
    rho = float((xs * ths).sum() / (ths * ths).sum())
    resid = xs - rho * ths
    ss = float(((xs - xs.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss if ss > 0 else float("nan")
    return rho, r2, int(len(ths))


def _own_face_nodes_near(mesh, node_id, radius):
    """Surface nodes within `radius` that lie on a CAD face containing the node."""
    surf = surface_node_ids(mesh)
    p0 = mesh.points[node_id]
    near = surf[np.linalg.norm(mesh.points[surf] - p0, axis=1) <= radius]
    sf = getattr(mesh, "surface_face_ids", None)
    if sf is not None:
        tris = np.asarray(mesh.surface_triangles)
        sf = np.asarray(sf)
        own = {int(f) for f in sf[np.any(tris == node_id, axis=1)]}
        on_face = set(map(int, np.unique(tris[np.isin(sf, list(own))])))
        near = np.array([n for n in near if int(n) in on_face])
    return near


def rays_normal_to_front(normal, tau, n=121, span_deg=80.0):
    """Unit rays in the plane normal to `tau`, theta measured from the inward
    normal projected into that plane, + towards b = tau x n.
    Returns (rays (n,3), theta_deg (n,), n_projected, b)."""
    nrm = normal - (normal @ tau) * tau
    nrm = nrm / np.linalg.norm(nrm)
    b = np.cross(tau, nrm)
    b = b / np.linalg.norm(b)
    th = np.radians(np.linspace(-span_deg, span_deg, n))
    rays = np.cos(th)[:, None] * nrm[None, :] + np.sin(th)[:, None] * b[None, :]
    return rays, np.degrees(th), nrm, b


def _perp_basis(t):
    a = np.array([1.0, 0, 0]) if abs(t[0]) < 0.9 else np.array([0, 1.0, 0])
    q1 = np.cross(t, a)
    q1 /= np.linalg.norm(q1)
    return np.c_[q1, np.cross(t, q1)]                      # (3, 2)


def process_zone_radii(d, n_r):
    """Sample radii and weights for the line average over [0, d]: midpoints in
    s with r = d s^2 and weight 2 s. The substitution integrates the r^-1/2
    crack-tip term exactly; uniform radii d/n .. d under-weighted it by 13 %
    at n = 30, which turned the MATS angle up to 2.7 deg off the closed form
    under T-stress (bench/mats_vs_closed_form.py). On a notch, whose field is
    finite at the root, both converge; this one samples nearer the root."""
    s = (np.arange(n_r) + 0.5) / n_r
    return d * s ** 2, 2.0 * s


def _ray_fan(mesh, sig6_nodal, p, nrm, tau, r, n_rays, span_deg, w=None):
    """Every ray of the fan from `p`: the stress tensor averaged over the
    sample radii `r` (with quadrature weights `w`, uniform when None), and its
    largest and smallest eigenvalues in the plane normal to the ray with the
    crack-plane normal. Rays that leave the mesh get -inf and valid False."""
    rays, th, _nrm, _b = rays_normal_to_front(nrm, tau, n_rays, span_deg)
    n_r = len(r)
    P = (p[None, None, :] + r[None, :, None] * rays[:, None, :]).reshape(-1, 3)
    comps = np.stack([sample_field_at_points(mesh, sig6_nodal[:, c], P) for c in range(6)], axis=1)
    comps = comps.reshape(len(rays), n_r, 6)
    valid = ~np.isnan(comps).any(axis=(1, 2))
    lam = np.full(len(rays), -np.inf)
    lam2 = np.full(len(rays), -np.inf)
    mvec = np.zeros((len(rays), 3))
    if not valid.any():
        return rays, th, lam, lam2, mvec, valid
    sb = comps.mean(axis=1) if w is None else np.einsum("r,nrc->nc", np.asarray(w, float) / np.sum(w), comps)
    S = np.zeros((len(rays), 3, 3))
    S[:, 0, 0], S[:, 1, 1], S[:, 2, 2] = sb[:, 0], sb[:, 1], sb[:, 2]
    S[:, 0, 1] = S[:, 1, 0] = sb[:, 3]
    S[:, 1, 2] = S[:, 2, 1] = sb[:, 4]
    S[:, 0, 2] = S[:, 2, 0] = sb[:, 5]
    for k in np.flatnonzero(valid):
        Q = _perp_basis(rays[k])
        w, v = np.linalg.eigh(Q.T @ S[k] @ Q)
        lam[k], lam2[k] = w[-1], w[0]
        mvec[k] = Q @ v[:, -1]
    return rays, th, lam, lam2, mvec, valid


def mats_fan_from(mesh, sig6_nodal, origin, normal, tau, d, n_rays=121, n_r=30, span_deg=80.0, band_tol=0.02):
    """The MATS fan cast from any point, e.g. the arc apex, with theta measured
    from `normal`. A cross-check of the path, not a search: on a mode-I notch
    the best ray from the apex should lie on its normal and the band of rays
    within `band_tol` of it should be centred there. Returns the fan, the best
    angle, the band and its centre, and how much the best beats theta = 0."""
    origin, normal, tau = (np.asarray(v, float) for v in (origin, normal, tau))
    r, w = process_zone_radii(d, n_r)
    rays, th, lam, _l2, _m, valid = _ray_fan(mesh, np.asarray(sig6_nodal, float), origin, normal, tau, r, n_rays, span_deg, w)
    if not valid.any():
        return None
    k, k0 = int(np.argmax(lam)), int(np.argmin(np.abs(th)))
    if not lam[k] > 0:
        # no ray carries tension (a point under a load pad): there is no opening direction to pick, and the band
        # below came out empty (break test 25, a 500)
        return None
    band = th[lam >= (1.0 - band_tol) * lam[k]]
    # the peak of a parabola through the best ray and its two neighbours: the
    # fan is 1.33 deg apart, which is coarser than the angles being compared
    th_peak = float(th[k])
    if 0 < k < len(th) - 1 and np.all(np.isfinite(lam[k - 1:k + 2])):
        a, b, _c = np.polyfit(th[k - 1:k + 2], lam[k - 1:k + 2], 2)
        if a < 0:
            th_peak = float(-b / (2 * a))
    return {"theta_deg": [round(float(t), 3) for t in th],
            "lam_rel": [round(float(x / lam[k]), 5) if np.isfinite(x) and lam[k] > 0 else None for x in lam],
            "best_deg": th_peak, "band_deg": [float(band.min()), float(band.max())],
            "band_centre_deg": float(0.5 * (band.min() + band.max())),
            "margin_over_normal": (float(lam[k] / lam[k0] - 1.0) if np.isfinite(lam[k0]) and lam[k0] > 0 else None),
            "d_mm": float(d)}


#: MATS must beat the stress averaged along the surface normal by this much
#: before the path leaves the normal. On a symmetric mode-I notch the two are
#: equal in principle, and the maximum over 121 rays and 5 nearby origins wins
#: by ~1 % on discretisation noise alone (measured 0.4-1.1 % on the ASA SENB,
#: tilting the path 7-11 deg). A real mixed-mode feature beats it by far more.
MATS_NORMAL_MARGIN = 0.05


#: faces meeting at less than this angle between their normals are a smooth join, not an edge
CREASE_MIN_DEG = 10.0


def reentrant_edge_at(mesh, point, tol):
    """The sharp re-entrant edge (V-notch tip, internal corner) within `tol` of a point.

    A rounded root has a peak round its arc (notch_apex). A sharp V has none:
    the field is singular on the edge itself, the hottest node is a flank node
    beside it, and a path cast along that flank's normal reads a false mixed
    mode (ct_specimen: origin 0.16 mm off the tip, path 43 deg off the
    bisector, MATS +39 % "mixed mode" on a pin-loaded, symmetric, mode-I CT).
    So the origin goes ON the edge, and the path along the bisector of the
    two faces' inward normals, which is what Williams' solution is symmetric
    about.

    Edges are taken from the mesh: surface nodes shared by triangles of two
    CAD faces. An edge counts when the faces meet re-entrantly (each lies on
    the outer side of the other's plane), i.e. the notch opening is under 180
    degrees. Returns {origin, bisector, tangent, opening_deg, faces,
    distance_mm} for the nearest such edge, or None.
    """
    sf = getattr(mesh, "surface_face_ids", None)
    if sf is None:
        return None
    point = np.asarray(point, float)
    pts = np.asarray(mesh.points)
    tris = np.asarray(mesh.surface_triangles)[:, :3]
    sf = np.asarray(sf)
    reach = 4.0 * tol
    near_tri = np.any(np.linalg.norm(pts[tris] - point, axis=2) <= reach, axis=1)
    tris, sf = tris[near_tri], sf[near_tri]
    if not len(tris):
        return None
    faces_of = {}
    for t, f in zip(tris, sf):
        for n in t:
            faces_of.setdefault(int(n), set()).add(int(f))
    # inward normal and centroid of each face's triangles near the point
    # (gmsh winds surface triangles outward)
    info = {}
    for f in np.unique(sf):
        T = tris[sf == f]
        a, b, c = pts[T[:, 0]], pts[T[:, 1]], pts[T[:, 2]]
        nrm = -np.cross(b - a, c - a).sum(axis=0)
        if np.linalg.norm(nrm) < 1e-14:
            continue
        info[int(f)] = (nrm / np.linalg.norm(nrm), pts[np.unique(T)].mean(axis=0))
    best = None
    pairs = {tuple(sorted(fs)[i:i + 2]) for fs in faces_of.values() if len(fs) >= 2 for i in range(len(fs) - 1)}
    for A, B in pairs:
        if A not in info or B not in info:
            continue
        (nA, cA), (nB, cB) = info[A], info[B]
        edge = np.array([n for n, fs in faces_of.items() if A in fs and B in fs])
        if len(edge) < 2:
            continue
        e0 = pts[edge].mean(axis=0)
        # the faces' normals AT the edge, from the triangles touching it: a
        # tangent-continuous join (a flank running into a round root) has no
        # crease there, however much the root turns further on
        eset = set(map(int, edge))
        def _edge_normal(f):
            T = tris[(sf == f) & np.array([bool(eset.intersection(map(int, t))) for t in tris])]
            a, b, c = pts[T[:, 0]], pts[T[:, 1]], pts[T[:, 2]]
            v = -np.cross(b - a, c - a).sum(axis=0)
            return v / np.linalg.norm(v)
        nA, nB = _edge_normal(A), _edge_normal(B)
        if np.degrees(np.arccos(np.clip(nA @ nB, -1.0, 1.0))) < CREASE_MIN_DEG:
            continue
        t = np.cross(nA, nB)
        t /= np.linalg.norm(t)
        # re-entrant: each face lies on the OUTER side of the other's plane
        if not ((cB - e0) @ nA < -1e-3 * tol and (cA - e0) @ nB < -1e-3 * tol):
            continue
        origin = e0 + ((point - e0) @ t) * t
        dist = float(np.linalg.norm(point - origin))
        if dist > tol or (best is not None and dist >= best["distance_mm"]):
            continue
        bis = nA + nB
        bis -= (bis @ t) * t
        bis /= np.linalg.norm(bis)
        opening = 180.0 - float(np.degrees(np.arccos(np.clip(nA @ nB, -1.0, 1.0))))
        r6 = lambda v: [round(float(x), 6) + 0.0 for x in v]
        best = {"origin": r6(origin), "bisector": r6(bis), "tangent": r6(t), "opening_deg": round(opening, 2),
                "faces": [A, B], "distance_mm": round(dist, 6), "edge_nodes": int(len(edge)),
                "inward_normals": [r6(nA), r6(nB)]}
    return best


def edge_polyline(mesh, faces, step):
    """Points every `step` mm along the whole edge where two CAD faces meet.

    reentrant_edge_at looks only near the picked point; the Weibull exclusion
    round a sharp edge needs all of it. The edge nodes (surface nodes on
    triangles of both faces) are ordered along the edge's main direction and
    joined by straight segments, so a straight edge is exact and a gently
    curved one is followed node to node. Returns (k, 3), or None."""
    sf = getattr(mesh, "surface_face_ids", None)
    if sf is None:
        return None
    A, B = (int(f) for f in faces)
    tris, sf = np.asarray(mesh.surface_triangles)[:, :3], np.asarray(sf)
    edge = np.intersect1d(np.unique(tris[sf == A]), np.unique(tris[sf == B]))
    if len(edge) < 2:
        return None
    P = np.asarray(mesh.points, float)[edge]
    c = P.mean(axis=0)
    t = np.linalg.svd(P - c, full_matrices=False)[2][0]
    P = P[np.argsort((P - c) @ t)]
    out = [P[:1]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out.append(a + (b - a) * (np.arange(1, n + 1)[:, None] / n))
    return np.vstack(out)


def mats_path(mesh, sig6_nodal, node_id, d, n_origins=5, n_rays=121, n_r=30, span_deg=80.0,
              band_tol=0.02, plane_tol=0.05, normal_margin=MATS_NORMAL_MARGIN):
    """The MATS extraction path at a located hotspot.

    `sig6_nodal` is the nodal stress tensor (n_nodes, 6: xx yy zz xy yz zx),
    sampled along each ray with the element shape functions; rays whose
    samples leave the mesh are rejected. Returns a dict with the chosen origin,
    ray, crack-plane normal, the front and its anisotropy, the angle from the
    origin's surface normal, the band of rays within `band_tol` of the
    maximum, and `plane_undetermined` when the two in-plane eigenvalues are
    within `plane_tol` of each other.

    `d` is the length each ray is averaged over. Keep it to the region the
    method actually reads (about L for the point methods): averaged over 2L on
    a three-point-bend bar with a 4.6 mm ligament, every ray ran through the
    neutral axis into compression, all averages collapsed to ~4 MPa, and a ray
    53 deg off the bisector won by 0.4 MPa (bench/repro_mats.py).

    The path stays on the surface normal at `node_id` unless the best ray beats
    the normal's average by more than `normal_margin`; `margin_over_normal`
    reports by how much it did, and `normal_preferred` whether the normal was
    kept for lack of it.
    """
    sig6_nodal = np.asarray(sig6_nodal, float)
    p0 = mesh.points[node_id]
    tau, aniso, n_norm = front_tangent(mesh, node_id, d)
    surf = surface_node_ids(mesh)
    dist = np.linalg.norm(mesh.points[surf] - p0, axis=1)
    near = [int(n) for n in surf[np.argsort(dist)] if np.linalg.norm(mesh.points[n] - p0) <= 0.5 * d][:n_origins]
    if node_id not in near:
        near = [int(node_id)] + near[: n_origins - 1]
    r, w = process_zone_radii(d, n_r)
    best, at_node = None, None
    for nid in near:
        p = mesh.points[nid]
        nrm = surface_normal_at_node(mesh, nid)
        rays, th, lam, lam2, mvec, valid = _ray_fan(mesh, sig6_nodal, p, nrm, tau, r, n_rays, span_deg, w)
        if not valid.any():
            continue
        k = int(np.argmax(lam))
        k0 = int(np.argmin(np.abs(th)))
        # the whole fan, for the bisector close-up: angle and averaged
        # crack-plane stress of every ray, relative to this origin's best
        fan = {"theta_deg": [round(float(t), 3) for t in th],
               "lam_rel": [round(float(x / lam[k]), 5) if np.isfinite(x) and lam[k] > 0 else None for x in lam]}
        if int(nid) == int(node_id) and np.isfinite(lam[k0]):
            # the normal at the snapped peak itself: what MATS has to beat
            at_node = {"node_id": int(nid), "origin": [float(x) for x in p], "ray": [float(x) for x in rays[k0]],
                       "crack_plane_normal": [float(x) for x in mvec[k0]], "lam": float(lam[k0]),
                       "surface_normal": [float(x) for x in nrm], "fan": fan}
        if best is None or lam[k] > best["lam_max"]:
            band = th[lam >= (1.0 - band_tol) * lam[k]]
            best = {"node_id": int(nid), "origin": [float(x) for x in p], "ray": [float(x) for x in rays[k]],
                    "crack_plane_normal": [float(x) for x in mvec[k]], "lam_max": float(lam[k]),
                    "lam_on_normal": float(lam[k0]) if np.isfinite(lam[k0]) else None,
                    "theta_deg": float(th[k]), "band_deg": [float(band.min()), float(band.max())],
                    "plane_undetermined": bool(lam[k] > 0 and (lam[k] - lam2[k]) / lam[k] < plane_tol),
                    "rays_in_material": int(valid.sum()), "rays_cast": int(len(rays)),
                    "surface_normal": [float(x) for x in nrm], "fan": fan}
    if best is None:
        raise ValueError("no MATS ray stays inside the material over the process zone")
    margin = None
    if at_node is not None and at_node["lam"] > 0:
        margin = best["lam_max"] / at_node["lam"] - 1.0
        if margin <= normal_margin:
            best.update({k: at_node[k] for k in ("node_id", "origin", "ray", "crack_plane_normal", "surface_normal", "fan")},
                        theta_deg=0.0)
    best["margin_over_normal"] = margin
    best["normal_preferred"] = bool(margin is not None and margin <= normal_margin)
    ray = np.asarray(best["ray"])
    nrm = np.asarray(best["surface_normal"])
    best.update({"front_tangent": [float(x) for x in tau], "front_anisotropy": aniso, "front_normals": n_norm,
                 "angle_from_normal_deg": float(np.degrees(np.arccos(np.clip(ray @ nrm, -1.0, 1.0)))),
                 "d_mm": float(d), "origins_tried": len(near)})
    return best


def notch_apex(mesh, field, node_id, tau, rho, inward, slab=None):
    """Where the path should start on a curved root: the field peak round the
    ARC, between nodes, and the arc's own normal there.

    The nodal argmax lands on whichever node happens to be hottest, and on a
    flat-topped peak that is decided by noise in the recovered field (+-5 % node
    to node at the root of the ASA SENB). On the 0.25 mm SENB root it picked a
    node 0.026 mm round the arc, whose surface normal is turned by s/rho =
    5.9 deg: a "bisector" visibly off the symmetry axis. fracture.mats_path
    keeps that normal, correctly, because no ray beats it. Here the root is
    located from the whole arc instead:

      1. the surface nodes of the node's own CAD faces within 2.2 rho, in a
         slab two elements thick across the front `tau`, projected into the
         plane normal to the front;
      2. a circle through them (Kasa least squares, one pass dropping nodes
         more than 5 % of R off it, i.e. the flanks): centre c, radius R;
      3. the field against the angle phi round c, fitted with its first
         harmonic c0 + a cos(phi) + b sin(phi); the peak is at atan2(b, a).
         Over the whole arc that reads the centre of the bell, which a local
         parabola through the noisy top cannot;
      4. origin c + R(cos phi*, sin phi*); direction along the radius, signed
         to point into the material (the same side as `inward`).

    Every normal of a circle passes through its centre, so step 4 is the
    surface normal at phi*, not an approximation of it. Returns None (keep the
    node) when there are too few nodes, the surface is not an arc, or the fitted
    peak is not inside the arc.
    """
    tau = np.asarray(tau, float); tau = tau / np.linalg.norm(tau)
    inward = np.asarray(inward, float)
    p0 = mesh.points[node_id]
    n0 = inward - (inward @ tau) * tau; n0 /= np.linalg.norm(n0)
    b = np.cross(tau, n0)
    reach = 2.2 * float(rho)
    if slab is None:
        slab = 2.0 * local_element_size(mesh, node_id)
    ids = np.asarray([int(n) for n in _own_face_nodes_near(mesh, node_id, reach + slab)])
    if not len(ids):
        return None
    X = mesh.points[ids] - p0
    keep = np.abs(X @ tau) <= slab
    ids, X = ids[keep], X[keep]
    if len(ids) < 8:
        return None
    u, v = X @ b, X @ n0

    def circle(m):
        A = np.c_[2 * u[m], 2 * v[m], np.ones(int(m.sum()))]
        (cu, cv, k), *_ = np.linalg.lstsq(A, u[m] ** 2 + v[m] ** 2, rcond=None)
        return cu, cv, float(np.sqrt(max(k + cu * cu + cv * cv, 0.0)))

    m = np.ones(len(ids), bool)
    cu, cv, R = circle(m)
    m = np.abs(np.hypot(u - cu, v - cv) - R) <= 0.05 * R
    if m.sum() < 8:
        return None
    cu, cv, R = circle(m)
    if not (0.2 * rho < R < 5.0 * rho):
        return None
    resid = np.hypot(u[m] - cu, v[m] - cv) - R
    # angles round c, 0 at the node, growing towards +b
    phi0 = np.arctan2(-cu, -cv)
    phi = (np.arctan2(u[m] - cu, v[m] - cv) - phi0 + np.pi) % (2 * np.pi) - np.pi
    f_all = np.asarray(field, float)[ids[m]]
    # The window is kept SYMMETRIC about the current estimate, and tapered: a first harmonic
    # fitted to a symmetric peak over a lopsided window is pulled towards the
    # longer side (7.6 deg on a shoulder fillet whose peak is 17 deg from the
    # end of the arc). So: half-width = the distance to the nearer end of the
    # arc, at most 90 deg, re-centred a few times. On a near-full circle (a
    # hole) the 90 deg cap matters for another reason: round the whole circle
    # the field is dominated by its SECOND harmonic (Kirsch, 1 - 2 cos 2 theta,
    # two equal peaks), which has no first-harmonic projection there; over a
    # half-circle round one peak it does.
    full = (phi.max() - phi.min()) > 1.5 * np.pi
    phis = 0.0
    for _ in range(4):
        half = np.pi / 2 if full else min(np.pi / 2, phis - phi.min(), phi.max() - phis)
        if half < np.radians(5.0):
            return None                    # the peak is at the end of the arc: keep the node
        rel = (phi - phis + np.pi) % (2 * np.pi) - np.pi
        w = np.abs(rel) <= half + 1e-9
        if w.sum() < 8:
            return None
        # tapered (cos^2) so the rings at the window's edge, which come and go
        # as it re-centres, carry almost no weight
        A = np.c_[np.ones(int(w.sum())), np.cos(rel[w]), np.sin(rel[w])]
        sw = np.cos(0.5 * np.pi * rel[w] / half)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], f_all[w] * sw, rcond=None)
        if coef[1] <= 0:
            return None
        step = float(np.arctan2(coef[2], coef[1]))
        if not (rel[w].min() < step < rel[w].max()):
            return None
        phis = (phis + step + np.pi) % (2 * np.pi) - np.pi
        if abs(step) < 1e-4:
            break
    f, pred = f_all[w], A @ coef
    ss = float(((f - f.mean()) ** 2).sum())
    r2 = 1.0 - float(((f - pred) ** 2).sum()) / ss if ss > 0 else float("nan")
    centre = p0 + cu * b + cv * n0
    radial = np.sin(phi0 + phis) * b + np.cos(phi0 + phis) * n0
    sign = 1.0 if radial @ n0 > 0 else -1.0       # concave root: the centre is in the air
    origin = centre + R * radial
    ray = sign * radial
    # the linear point locator sees straight-sided tets; nudge in until the
    # origin is inside one, so the profile has a value at r = 0
    inset = 0.0
    if np.isnan(sample_field_at_points(mesh, field, origin[None])[0]):
        for eps in R * np.logspace(-5, -2, 7):
            if not np.isnan(sample_field_at_points(mesh, field, (origin + eps * ray)[None])[0]):
                inset = float(eps); break
        else:
            return None
        origin = origin + inset * ray
    span = np.degrees([phi.min(), phi.max()])          # the arc the circle was fitted to
    return {"origin": [float(x) for x in origin], "ray": [float(x) for x in ray],
            "centre": [float(x) for x in centre], "R_fit_mm": R,
            "circle_rms_mm": float(np.sqrt(np.mean(resid ** 2))),
            "phi_deg": float(np.degrees(phis)), "arc_deg": [float(span[0]), float(span[1])],
            "peak_fit_r2": r2, "n_nodes": int(m.sum()),
            "from_node": int(node_id), "moved_mm": float(np.linalg.norm(origin - p0)),
            "inset_mm": inset}


def notch_section(mesh, origin, ray, tau, half_width):
    """The part's outline in the plane through `origin` normal to the front:
    the surface triangles cut by that plane, as 2D segments in (x along
    tau x ray, y along ray) mm, within `half_width` of the origin. What the
    bisector close-up draws."""
    origin, ray, tau = (np.asarray(v, float) for v in (origin, ray, tau))
    b = np.cross(tau, ray); b /= np.linalg.norm(b)
    tris = np.asarray(mesh.surface_triangles)
    P = mesh.points[tris]                                    # (n,3,3)
    near = np.linalg.norm(P.mean(axis=1) - origin, axis=1) <= 1.5 * half_width
    P = P[near]
    d = (P - origin) @ tau                                   # signed distance to the plane
    segs = []
    for tri, dd in zip(P, d):
        pts = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            if (dd[i] > 0) != (dd[j] > 0):
                t = dd[i] / (dd[i] - dd[j])
                pts.append(tri[i] + t * (tri[j] - tri[i]))
        if len(pts) == 2:
            q = [((p - origin) @ b, (p - origin) @ ray) for p in pts]
            if all(abs(x) <= half_width and abs(y) <= half_width for x, y in q):
                segs.append([round(float(c), 5) for xy in q for c in xy])
    return segs


def local_root_gradient(r, values, h):
    """Model-free root value and slope from dense, noise-free samples (exact BEM
    stress along the path): a cubic through the samples on [0, h], and again on
    [0, h/2]. Returns value, slope, relative gradient |dv/dr| / v (FKM chi' /
    Kwon for sigma1) for the full window, the relative gradient on the half
    window and their change in percent -- the truncation check. Needs at least
    5 samples in the half window."""
    r = np.asarray(r, float)
    values = np.asarray(values, float)

    def fit(hh):
        keep = r <= hh * (1 + 1e-9)
        if keep.sum() < 5:
            raise ValueError(f"only {int(keep.sum())} samples within {hh:g} mm of the root")
        c = np.polyfit(r[keep], values[keep], 3)
        return float(c[-1]), float(c[-2]), int(keep.sum())
    v0, dv, n = fit(h)
    v0b, dvb, _ = fit(0.5 * h)
    rel = abs(dv) / v0 if v0 > 0 else None
    relb = abs(dvb) / v0b if v0b > 0 else None
    return {"value_root": v0, "dvalue_dr": dv, "relative_gradient_per_mm": rel,
            "relative_gradient_half_window": relb, "n_points": n,
            "window_change_pct": (round(100.0 * (relb - rel) / rel, 3) if rel and relb else None)}


def creager_paris_root_gradient(K_I, rho):
    """Stress at the root and its gradient along the bisector from the fitted
    Creager-Paris field, sigma(s) = K/sqrt(2 pi s) (1 + rho/(2s)), s = r + rho/2.

    Returns (sigma_root, dsigma_dr_at_root, relative_gradient_per_mm) with
    relative gradient = |dsigma/dr| / sigma at the root -- the FKM chi' / Kwon
    quantity, taken analytically from the fit rather than from two mesh
    samples. dsigma/dr is negative (the stress decays); the sign is kept.
    """
    K_I, rho = float(K_I), float(rho)
    if K_I <= 0 or rho <= 0:
        raise ValueError("K_I and rho must be positive")
    s0 = 0.5 * rho
    sigma_root = 2.0 * K_I / math.sqrt(math.pi * rho)
    # d/ds [ K (2 pi s)^-1/2 (1 + rho/(2 s)) ]
    a = K_I / math.sqrt(2.0 * math.pi)
    dsig = a * (-0.5 * s0 ** -1.5 * (1.0 + rho / (2.0 * s0)) + s0 ** -0.5 * (-rho / (2.0 * s0 ** 2)))
    return sigma_root, dsig, abs(dsig) / sigma_root


# ---------------------------------------------------------------------------
# Creager-Paris SERIES: the blunt-notch leading term plus higher-order Williams
# terms, fitted linearly with a conditioning guard. The single-term fit above
# describes the singular zone; over the whole extraction path the T-stress and
# the structural gradient are not small, and they are what the extra terms
# absorb. The fit is LINEAR in every coefficient (the origin shift rho/2 is
# taken from the CAD or the single-term fit), so there is no optimiser and
# the conditioning can be measured and enforced rather than hoped for.
# ---------------------------------------------------------------------------

def _cp_series_columns(r, rho, n_terms, enriched, s_ref, exponent=None):
    """Basis columns g_k(s) and their derivatives dg_k/dr, s = r + rho/2 with
    the Williams terms in s_hat = s/s_ref so every column is O(1).

    k = 0 : the Creager-Paris term (2 pi s)^-1/2 (1 + rho/2s)  [enriched], or
            the sharp-crack (2 pi s)^-1/2                       [plain], or,
            with `exponent` e, the sharp V-notch Williams term (2 pi r)^-e
            (rho = 0, e from williams_exponent of the opening angle);
            its coefficient is K_I in MPa sqrt(mm) (MPa mm^e for a V-notch).
    k >= 1: s_hat^((k-1)/2): the T-stress (k=1), then s^1/2, s^1, ...
    """
    r = np.asarray(r, float)
    s = r + 0.5 * rho
    sh = s / s_ref
    cols, dcols = [], []
    f0 = 1.0 / np.sqrt(2.0 * np.pi * s)
    if exponent is not None:
        with np.errstate(divide="ignore"):
            g0 = (2.0 * np.pi * s) ** -float(exponent)
        cols.append(g0)
        dcols.append(-float(exponent) * g0 / s)
    elif enriched:
        cols.append(f0 * (1.0 + rho / (2.0 * s)))
        dcols.append(f0 * (-0.5 / s * (1.0 + rho / (2.0 * s)) - rho / (2.0 * s * s)))
    else:
        cols.append(f0)
        dcols.append(f0 * (-0.5 / s))
    for k in range(1, n_terms):
        p = 0.5 * (k - 1)
        cols.append(sh ** p)
        dcols.append((p * sh ** (p - 1.0) / s_ref) if p != 0 else np.zeros_like(sh))
    return np.column_stack(cols), np.column_stack(dcols)


def creager_paris_series(r, fit, squared=False):
    """Evaluate a fit from creager_paris_series_fit at distances r from the root."""
    A, _ = _cp_series_columns(r, fit["rho"], fit["n_terms"], fit["enriched"], fit["s_ref"], fit.get("williams_exponent"))
    y = A @ np.asarray(fit["coefficients"], float)
    return y * y if squared else y


def creager_paris_series_slope(r, fit, squared=False):
    """d/dr of a creager_paris_series_fit at distances r: a smooth derivative
    along the whole path for gradient criteria (chi' at r = 0 is fit["root"])."""
    A, dA = _cp_series_columns(r, fit["rho"], fit["n_terms"], fit["enriched"], fit["s_ref"], fit.get("williams_exponent"))
    c = np.asarray(fit["coefficients"], float)
    return 2.0 * (A @ c) * (dA @ c) if squared else dA @ c


def creager_paris_series_fit(r, v, rho=None, n_terms=4, enriched=True, squared=False,
                             max_cond=1e3, min_points_per_term=4, rho_bounds=None, n_rho=200,
                             weighting="relative", williams=None):
    """Fit sigma(r) = K_I g_0(s) + sum_k a_k s_hat^((k-1)/2) along the path.

    `n_terms` is the number REQUESTED; the fit backs off one term at a time
    until the column-normalised design matrix has a condition number below
    `max_cond` and there are at least `min_points_per_term` samples per
    coefficient, and reports what it used. An energy density is fitted as the
    square of the series (the series is fitted to sqrt(v), v >= 0).

    Pass `rho` to hold the root radius at its CAD value. Leave it None and it
    is found by a 1-D search (the model is linear in everything else, so each
    trial is one least-squares solve): the rho that best explains the samples
    WITH the higher-order terms present, which the single-term fit cannot do
    when a T-stress or structural gradient is superimposed.

    Returns None when even the one-term fit is not supported, else a dict with
    K_I, the coefficients, n_terms, enriched, cond, r2 (against the samples as
    given, i.e. after squaring for an energy density), the sample span, and
    `root`: the analytic value, slope and relative gradient at r = 0 from the
    fitted series -- the FKM chi' / Kwon quantity taken from the fit rather
    than from two mesh samples.

    Four terms by default: CP, T-stress, s^1/2 and s^1. Measured on the SENB
    (5 mm ligament, path 1.16 mm): three terms leave the tail 14% (stress) and
    31% (SED) off the SPR profile; the linear term, which is the bending
    gradient across the ligament, brings both within 2% at condition 200.

    `williams=e` fits a SHARP V-notch instead: leading term (2 pi r)^-e with
    e = williams_exponent(opening angle) and no root radius (rho = 0), plus
    the same higher terms. A Creager-Paris rho fitted there only reports the
    mesh (ct_specimen's 60-degree V read rho 0.022 mm from a 0.058 mm mesh).
    The field is singular at r = 0, so `root` carries no value or gradient.

    `weighting="relative"` divides each row by |y| so the residuals are
    fractional: a field that decays a decade along the path would otherwise
    be fitted at the root and ignored in the tail, which on a log axis reads
    as divergence. "none" is the plain least squares.
    """
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    if len(r) < min_points_per_term:
        return None
    y = np.sqrt(np.maximum(v, 0.0)) if squared else v
    ss = float(((v - v.mean()) ** 2).sum())
    if weighting == "relative":
        wt = 1.0 / np.maximum(np.abs(y), 1e-3 * float(np.abs(y).max()) if np.abs(y).max() > 0 else 1.0)
    else:
        wt = np.ones_like(y)

    def solve_at(rho_try):
        s_ref = float(r.max() + 0.5 * rho_try)
        for n in range(int(n_terms), 0, -1):
            if len(r) < min_points_per_term * n:
                continue
            A, dA = _cp_series_columns(r, rho_try, n, enriched, s_ref, williams)
            Aw = A * wt[:, None]
            scale = np.linalg.norm(Aw, axis=0)
            if not np.all(scale > 0):
                continue
            As = Aw / scale
            cond = float(np.linalg.cond(As))
            if cond > max_cond and n > 1:
                continue
            c_s, *_ = np.linalg.lstsq(As, y * wt, rcond=None)
            coef = c_s / scale
            if coef[0] <= 0:                   # a negative K_I is not a notch field
                if n > 1:
                    continue
                return None
            pred = A @ coef
            pred_v = pred * pred if squared else pred
            r2 = 1.0 - float(((v - pred_v) ** 2).sum()) / ss if ss > 0 else float("nan")
            return {"n": n, "coef": coef, "cond": cond, "r2": r2, "s_ref": s_ref, "rho": float(rho_try)}
        return None

    if williams is not None:
        if not np.all(r > 0):                  # the tip itself is singular: fit off it
            return creager_paris_series_fit(r[r > 0], v[r > 0], n_terms=n_terms, squared=squared, max_cond=max_cond,
                                            min_points_per_term=min_points_per_term, weighting=weighting,
                                            williams=williams)
        best = solve_at(0.0)
    elif rho is not None:
        if float(rho) <= 0:
            return None
        best = solve_at(float(rho))
    else:
        lo, hi = rho_bounds or (0.02 * float(r.max()), 20.0 * float(r.max()))
        best = None
        for rho_try in np.geomspace(lo, hi, int(n_rho)):
            got = solve_at(float(rho_try))
            if got is not None and (best is None or got["r2"] > best["r2"]):
                best = got
    if best is None:
        return None
    n, coef, cond, rho_used, s_ref = best["n"], best["coef"], best["cond"], best["rho"], best["s_ref"]
    if williams is not None:                   # singular at the tip: no root value, no root gradient
        return {"K_I": float(coef[0]), "coefficients": [float(c) for c in coef], "rho": 0.0, "rho_fitted": False,
                "williams_exponent": float(williams), "n_terms": int(n), "n_terms_requested": int(n_terms),
                "enriched": False, "s_ref": s_ref, "cond": cond, "r2": best["r2"], "n_points": int(len(r)),
                "weighting": weighting, "r_min": float(r.min()), "r_max": float(r.max()),
                "root": {"sigma_root_MPa": None, "dsigma_dr_MPa_per_mm": None, "relative_gradient_per_mm": None}}
    A0, dA0 = _cp_series_columns(np.zeros(1), rho_used, n, enriched, s_ref)
    sig0 = float(A0[0] @ coef)
    dsig0 = float(dA0[0] @ coef)
    root = {"sigma_root_MPa": sig0, "dsigma_dr_MPa_per_mm": dsig0,
            "relative_gradient_per_mm": (abs(dsig0) / sig0) if sig0 > 0 else None}
    if squared:                                # v = sigma^2: dv/dr = 2 sigma dsigma/dr, relative = 2 chi'
        root = {"value_root": sig0 * sig0, "dvalue_dr": 2.0 * sig0 * dsig0,
                "relative_gradient_per_mm": (2.0 * abs(dsig0) / sig0) if sig0 > 0 else None,
                "sigma_root_MPa": sig0, "dsigma_dr_MPa_per_mm": dsig0}
    return {"K_I": float(coef[0]), "coefficients": [float(c) for c in coef], "rho": rho_used,
            "rho_fitted": rho is None,
            "n_terms": int(n), "n_terms_requested": int(n_terms), "enriched": bool(enriched),
            "s_ref": s_ref, "cond": cond, "r2": best["r2"], "n_points": int(len(r)), "weighting": weighting,
            "r_min": float(r.min()), "r_max": float(r.max()), "root": root}
