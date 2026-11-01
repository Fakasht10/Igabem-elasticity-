// stl_repair_core native kernel: exact triangle-pair classification.
//
// Licensed under the PolyForm Noncommercial License 1.0.0 (see ../LICENSE).
// Required Notice: Copyright Arif Gökdaş (https://github.com/Krateian/Sutura)
//
// Re-implementation of the classification contract of Sutura 0.9.8
// `rust/sutura-geom/src/triangle_intersection.rs` (six orient3d signs, coplanar
// overlap kept separate from proper crossings, boundary contacts classified as
// touches).  Signs come from Shewchuk's adaptive-exact predicates (`robust`);
// the rare non-coplanar interval comparison is done in exact rationals, so the
// classification is exact for every finite f64 input and independent of
// evaluation order or thread count.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Signed, Zero};
use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArrayLike2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use robust::{orient2d, orient3d, Coord, Coord3D};

type P = [f64; 3];

pub const DISJOINT: u8 = 0;
pub const ADJ_OK: u8 = 1; // adjacent pair meeting only in the shared vertex/edge
pub const TOUCH: u8 = 2; // boundary contact without crossing or area overlap
pub const COPLANAR: u8 = 3; // coplanar pair with positive-area overlap
pub const PROPER: u8 = 4; // interiors cross transversally
pub const DUPLICATE: u8 = 5; // same three vertex indices
pub const DEGENERATE: u8 = 6; // one triangle has zero area (not classified)

#[inline]
fn sgn(x: f64) -> i8 {
    if x > 0.0 {
        1
    } else if x < 0.0 {
        -1
    } else {
        0
    }
}

#[inline]
fn c3(p: &P) -> Coord3D<f64> {
    Coord3D { x: p[0], y: p[1], z: p[2] }
}

#[inline]
fn o3(a: &P, b: &P, c: &P, d: &P) -> i8 {
    sgn(orient3d(c3(a), c3(b), c3(c), c3(d)))
}

#[inline]
fn proj(p: &P, ax: usize) -> Coord<f64> {
    // drop coordinate `ax`, keep cyclic order so orientation is consistent
    match ax {
        0 => Coord { x: p[1], y: p[2] },
        1 => Coord { x: p[2], y: p[0] },
        _ => Coord { x: p[0], y: p[1] },
    }
}

#[inline]
fn o2(a: &P, b: &P, c: &P, ax: usize) -> i8 {
    sgn(orient2d(proj(a, ax), proj(b, ax), proj(c, ax)))
}

/// Exact zero-area test: collinear iff all three axis projections are.
pub fn degenerate(t: &[P; 3]) -> bool {
    (0..3).all(|ax| o2(&t[0], &t[1], &t[2], ax) == 0)
}

fn rat(x: f64) -> BigRational {
    BigRational::from_float(x).expect("finite coordinate")
}

type R3 = [BigRational; 3];

fn rp(p: &P) -> R3 {
    [rat(p[0]), rat(p[1]), rat(p[2])]
}
fn rsub(a: &R3, b: &R3) -> R3 {
    [&a[0] - &b[0], &a[1] - &b[1], &a[2] - &b[2]]
}
fn rcross(a: &R3, b: &R3) -> R3 {
    [
        &a[1] * &b[2] - &a[2] * &b[1],
        &a[2] * &b[0] - &a[0] * &b[2],
        &a[0] * &b[1] - &a[1] * &b[0],
    ]
}
fn rdot(a: &R3, b: &R3) -> BigRational {
    &a[0] * &b[0] + &a[1] * &b[1] + &a[2] * &b[2]
}

/// Interval of T ∩ plane(n, q0) projected on direction `d`; plus whether T
/// strictly crosses the plane (vertices strictly on both sides).
fn interval(t: &[R3; 3], n: &R3, q0: &R3, d: &R3) -> (Option<(BigRational, BigRational)>, bool) {
    let sd: Vec<BigRational> = t.iter().map(|p| rdot(n, &rsub(p, q0))).collect();
    let pos = sd.iter().any(|x| x.is_positive());
    let neg = sd.iter().any(|x| x.is_negative());
    let mut ts: Vec<BigRational> = Vec::with_capacity(3);
    for i in 0..3 {
        if sd[i].is_zero() {
            ts.push(rdot(d, &t[i]));
        }
        let j = (i + 1) % 3;
        if (sd[i].is_positive() && sd[j].is_negative()) || (sd[i].is_negative() && sd[j].is_positive()) {
            let lam = &sd[i] / (&sd[i] - &sd[j]);
            let di = rdot(d, &t[i]);
            let dj = rdot(d, &t[j]);
            ts.push(&di + (&dj - &di) * lam);
        }
    }
    if ts.is_empty() {
        return (None, pos && neg);
    }
    let mut lo = ts[0].clone();
    let mut hi = ts[0].clone();
    for x in ts.iter().skip(1) {
        if *x < lo {
            lo = x.clone();
        }
        if *x > hi {
            hi = x.clone();
        }
    }
    (Some((lo, hi)), pos && neg)
}

/// Same-direction test for two rays a->b and a->d known to be collinear.
fn same_dir(a: &P, b: &P, d: &P) -> bool {
    for k in 0..3 {
        if b[k] != a[k] {
            return (b[k] > a[k]) == (d[k] > a[k]) && d[k] != a[k];
        }
    }
    false
}

/// Classify one triangle pair.  `sh` lists shared vertices as (i in T1, j in T2).
pub fn classify(p: &[P; 3], q: &[P; 3], sh: &[(usize, usize)]) -> u8 {
    if sh.len() >= 3 {
        return DUPLICATE;
    }
    if degenerate(p) || degenerate(q) {
        return DEGENERATE;
    }
    let s: [i8; 3] = [o3(&q[0], &q[1], &q[2], &p[0]), o3(&q[0], &q[1], &q[2], &p[1]), o3(&q[0], &q[1], &q[2], &p[2])];
    let t: [i8; 3] = [o3(&p[0], &p[1], &p[2], &q[0]), o3(&p[0], &p[1], &p[2], &q[1]), o3(&p[0], &p[1], &p[2], &q[2])];
    let same = |v: &[i8; 3]| v[0] != 0 && v[0] == v[1] && v[1] == v[2];
    if same(&s) || same(&t) {
        return DISJOINT;
    }
    let coplanar = s.iter().all(|&x| x == 0);
    if coplanar {
        return classify_coplanar(p, q, sh);
    }
    if sh.len() == 2 {
        // non-coplanar triangles sharing an edge meet exactly in that edge
        return ADJ_OK;
    }
    if sh.len() == 1 {
        let (i, j) = sh[0];
        // non-shared vertices of T2 strictly on one side of plane(T1) -> only the vertex
        let tn: Vec<i8> = (0..3).filter(|&k| k != j).map(|k| t[k]).collect();
        let sn: Vec<i8> = (0..3).filter(|&k| k != i).map(|k| s[k]).collect();
        if (tn[0] != 0 && tn[0] == tn[1]) || (sn[0] != 0 && sn[0] == sn[1]) {
            return ADJ_OK;
        }
        if tn[0] * tn[1] < 0 && sn[0] * sn[1] < 0 {
            // T2 meets plane(T1) in [a, X2], X2 on edge de.  X2 is on the ray
            // a->X1 iff it lies strictly inside T1's angle at a; the side of
            // X2 w.r.t. line ab (in plane T1) equals orient3d(a,b,d,e).
            let a = &p[i];
            let (b, c) = (&p[(i + 1) % 3], &p[(i + 2) % 3]);
            let (d, e) = (&q[(j + 1) % 3], &q[(j + 2) % 3]);
            let s1 = o3(a, b, d, e) * o3(a, b, d, c);
            let s2 = o3(a, c, d, e) * o3(a, c, d, b);
            if s1 < 0 || s2 < 0 {
                return ADJ_OK;
            }
            if s1 > 0 && s2 > 0 {
                return PROPER;
            }
        }
    }
    // exact interval overlap on the line plane(T1) ∩ plane(T2)
    let pr = [rp(&p[0]), rp(&p[1]), rp(&p[2])];
    let qr = [rp(&q[0]), rp(&q[1]), rp(&q[2])];
    let n1 = rcross(&rsub(&pr[1], &pr[0]), &rsub(&pr[2], &pr[0]));
    let n2 = rcross(&rsub(&qr[1], &qr[0]), &rsub(&qr[2], &qr[0]));
    let d = rcross(&n1, &n2);
    let (i1, x1) = interval(&pr, &n2, &qr[0], &d);
    let (i2, x2) = interval(&qr, &n1, &pr[0], &d);
    let (Some((lo1, hi1)), Some((lo2, hi2))) = (i1, i2) else {
        return DISJOINT;
    };
    let lo = if lo1 > lo2 { lo1 } else { lo2 };
    let hi = if hi1 < hi2 { hi1 } else { hi2 };
    if lo > hi {
        return DISJOINT;
    }
    if lo == hi {
        // a single common point
        return if sh.len() == 1 { ADJ_OK } else { TOUCH };
    }
    if x1 && x2 {
        PROPER
    } else {
        TOUCH
    }
}

fn classify_coplanar(p: &[P; 3], q: &[P; 3], sh: &[(usize, usize)]) -> u8 {
    // projection axis: any axis where T1 projects non-degenerately (exact)
    let ax = {
        let n = [
            (p[1][1] - p[0][1]) * (p[2][2] - p[0][2]) - (p[1][2] - p[0][2]) * (p[2][1] - p[0][1]),
            (p[1][2] - p[0][2]) * (p[2][0] - p[0][0]) - (p[1][0] - p[0][0]) * (p[2][2] - p[0][2]),
            (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]),
        ];
        let mut order = [0usize, 1, 2];
        order.sort_by(|&a, &b| n[b].abs().partial_cmp(&n[a].abs()).unwrap());
        *order.iter().find(|&&a| o2(&p[0], &p[1], &p[2], a) != 0).unwrap_or(&2)
    };
    let op = o2(&p[0], &p[1], &p[2], ax);
    let oq = o2(&q[0], &q[1], &q[2], ax);
    if op == 0 || oq == 0 {
        return DEGENERATE;
    }
    // side of point x w.r.t. edge (a,b) of a triangle, inside-positive
    let mut area = true;
    let mut touch = true;
    for (tri, other, o) in [(p, q, op), (q, p, oq)] {
        for k in 0..3 {
            let (a, b) = (&tri[k], &tri[(k + 1) % 3]);
            let sides = [
                o2(a, b, &other[0], ax) * o,
                o2(a, b, &other[1], ax) * o,
                o2(a, b, &other[2], ax) * o,
            ];
            if sides.iter().all(|&v| v <= 0) {
                area = false;
            }
            if sides.iter().all(|&v| v < 0) {
                touch = false;
            }
        }
    }
    if area {
        return COPLANAR;
    }
    if !touch {
        return DISJOINT;
    }
    match sh.len() {
        2 => ADJ_OK, // opposite sides of the shared edge (same side would overlap)
        1 => {
            let (i, j) = sh[0];
            let a = &p[i];
            let pe = [&p[(i + 1) % 3], &p[(i + 2) % 3]];
            let qe = [&q[(j + 1) % 3], &q[(j + 2) % 3]];
            for b in pe.iter() {
                for d in qe.iter() {
                    if o2(a, b, d, ax) == 0 && same_dir(a, b, d) {
                        return TOUCH; // collinear edges folding onto each other
                    }
                }
            }
            ADJ_OK
        }
        _ => TOUCH,
    }
}

fn read_v(v: &numpy::ndarray::ArrayView2<f64>, i: i64) -> P {
    let i = i as usize;
    [v[[i, 0]], v[[i, 1]], v[[i, 2]]]
}

/// Classify candidate pairs.  verts (n,3) f64, tris (m,3) i64, pairs (k,2) i64.
#[pyfunction]
#[pyo3(signature = (verts, tris, pairs))]
fn classify_pairs<'py>(
    py: Python<'py>,
    verts: PyArrayLike2<'py, f64, AllowTypeChange>,
    tris: PyArrayLike2<'py, i64, AllowTypeChange>,
    pairs: PyArrayLike2<'py, i64, AllowTypeChange>,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let v = verts.as_array();
    let t = tris.as_array();
    let pr = pairs.as_array();
    if v.ncols() != 3 || t.ncols() != 3 || (pr.nrows() > 0 && pr.ncols() != 2) {
        return Err(PyValueError::new_err("shapes must be (n,3), (m,3), (k,2)"));
    }
    if v.iter().any(|x| !x.is_finite()) {
        return Err(PyValueError::new_err("non-finite coordinate"));
    }
    let nv = v.nrows() as i64;
    let m = t.nrows() as i64;
    if t.iter().any(|&x| x < 0 || x >= nv) || pr.iter().any(|&x| x < 0 || x >= m) {
        return Err(PyValueError::new_err("index out of range"));
    }
    let mut out = Vec::with_capacity(pr.nrows());
    for r in 0..pr.nrows() {
        let (f, g) = (pr[[r, 0]] as usize, pr[[r, 1]] as usize);
        let fi = [t[[f, 0]], t[[f, 1]], t[[f, 2]]];
        let gi = [t[[g, 0]], t[[g, 1]], t[[g, 2]]];
        let mut sh: Vec<(usize, usize)> = Vec::new();
        for i in 0..3 {
            for j in 0..3 {
                if fi[i] == gi[j] {
                    sh.push((i, j));
                }
            }
        }
        let p = [read_v(&v, fi[0]), read_v(&v, fi[1]), read_v(&v, fi[2])];
        let q = [read_v(&v, gi[0]), read_v(&v, gi[1]), read_v(&v, gi[2])];
        out.push(classify(&p, &q, &sh));
    }
    Ok(PyArray1::from_vec(py, out))
}

/// Exact zero-area mask for every triangle.
#[pyfunction]
#[pyo3(signature = (verts, tris))]
fn degenerate_mask<'py>(
    py: Python<'py>,
    verts: PyArrayLike2<'py, f64, AllowTypeChange>,
    tris: PyArrayLike2<'py, i64, AllowTypeChange>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let v = verts.as_array();
    let t = tris.as_array();
    let nv = v.nrows() as i64;
    if t.iter().any(|&x| x < 0 || x >= nv) {
        return Err(PyValueError::new_err("index out of range"));
    }
    let out: Vec<bool> = (0..t.nrows())
        .map(|r| {
            let p = [read_v(&v, t[[r, 0]]), read_v(&v, t[[r, 1]]), read_v(&v, t[[r, 2]])];
            degenerate(&p)
        })
        .collect();
    Ok(PyArray1::from_vec(py, out))
}

/// Exact orient3d signs for k quadruples of vertex indices (k,4).
#[pyfunction]
#[pyo3(signature = (verts, quads))]
fn orient3d_signs<'py>(
    py: Python<'py>,
    verts: PyArrayLike2<'py, f64, AllowTypeChange>,
    quads: PyArrayLike2<'py, i64, AllowTypeChange>,
) -> PyResult<Bound<'py, PyArray1<i8>>> {
    let v = verts.as_array();
    let q = quads.as_array();
    let nv = v.nrows() as i64;
    if q.iter().any(|&x| x < 0 || x >= nv) {
        return Err(PyValueError::new_err("index out of range"));
    }
    let out: Vec<i8> = (0..q.nrows())
        .map(|r| {
            o3(&read_v(&v, q[[r, 0]]), &read_v(&v, q[[r, 1]]), &read_v(&v, q[[r, 2]]), &read_v(&v, q[[r, 3]]))
        })
        .collect();
    Ok(PyArray1::from_vec(py, out))
}


/// All pairs (i<j) of closed axis-aligned boxes that overlap (sort and sweep
/// along the axis of largest total span).  Deterministic output order.
#[pyfunction]
#[pyo3(signature = (lo, hi))]
fn box_pairs<'py>(
    py: Python<'py>,
    lo: PyArrayLike2<'py, f64, AllowTypeChange>,
    hi: PyArrayLike2<'py, f64, AllowTypeChange>,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let lo = lo.as_array();
    let hi = hi.as_array();
    let m = lo.nrows();
    if lo.ncols() != 3 || hi.ncols() != 3 || hi.nrows() != m {
        return Err(PyValueError::new_err("lo/hi must be (m,3)"));
    }
    let mut ax = 0usize;
    let mut best = -1.0f64;
    for k in 0..3 {
        let mut a = f64::INFINITY;
        let mut b = f64::NEG_INFINITY;
        for i in 0..m {
            a = a.min(lo[[i, k]]);
            b = b.max(hi[[i, k]]);
        }
        if b - a > best {
            best = b - a;
            ax = k;
        }
    }
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&x, &y| lo[[x, ax]].partial_cmp(&lo[[y, ax]]).unwrap().then(x.cmp(&y)));
    let (o1, o2) = ((ax + 1) % 3, (ax + 2) % 3);
    let mut out: Vec<i64> = Vec::new();
    for s in 0..m {
        let i = order[s];
        let hx = hi[[i, ax]];
        for &j in order[s + 1..].iter() {
            if lo[[j, ax]] > hx {
                break;
            }
            if lo[[j, o1]] <= hi[[i, o1]] && hi[[j, o1]] >= lo[[i, o1]]
                && lo[[j, o2]] <= hi[[i, o2]] && hi[[j, o2]] >= lo[[i, o2]]
            {
                let (a, b) = if i < j { (i, j) } else { (j, i) };
                out.push(a as i64);
                out.push(b as i64);
            }
        }
    }
    let k = out.len() / 2;
    let arr = numpy::ndarray::Array2::from_shape_vec((k, 2), out).unwrap();
    Ok(PyArray2::from_owned_array(py, arr))
}

#[allow(dead_code)]
fn _bigint_marker(_: BigInt) {}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(classify_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(degenerate_mask, m)?)?;
    m.add_function(wrap_pyfunction!(orient3d_signs, m)?)?;
    m.add_function(wrap_pyfunction!(box_pairs, m)?)?;
    m.add("DISJOINT", DISJOINT)?;
    m.add("ADJ_OK", ADJ_OK)?;
    m.add("TOUCH", TOUCH)?;
    m.add("COPLANAR", COPLANAR)?;
    m.add("PROPER", PROPER)?;
    m.add("DUPLICATE", DUPLICATE)?;
    m.add("DEGENERATE", DEGENERATE)?;
    Ok(())
}
