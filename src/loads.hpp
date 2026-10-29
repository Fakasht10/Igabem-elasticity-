// Load distributions and spring laws on a region (RegionMesh). Every
// distribution is normalised with the region's own quadrature, so its resultant
// is exactly the requested one on the discrete region:
//
//   traction   t = q                                (q per unit area)
//   pressure   t = -p n                             (p > 0 pushes into the solid)
//   force      t = F / A                            (uniform, bemkit's "force_N")
//   wrench     t = F / A + w x (x - c),  w = J^-1 (M_c),  J = int (|r|^2 I - r r^T)
//              (force F and moment M about a reference point, linear like an
//              RBE3 coupling: pure bending or torsion of a section is exact)
//   bearing    t = -p0 max(0, -n.d) n,  p0 = |F| / int max(0, -n.d)^2
//              (cosine pressure of a pin on a bore, resultant |F| along d)
//
//   springs    k = k_n n n^T + k_t (I - n n^T), or diag(kx, ky, kz), per unit area
#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <string>

#include "bc_model.hpp"
#include "regions.hpp"

namespace igabem {

using RegionPtr = std::shared_ptr<const RegionMesh>;

inline RegionLoad tractionLoad(const std::string& label, RegionPtr R, const Vec3& q) {
  return {label, R, [q](const RPoint&) { return q; }};
}

inline RegionLoad pressureLoad(const std::string& label, RegionPtr R, double p) {
  return {label, R, [p](const RPoint& r) { return Vec3(-p * r.n); }};
}

inline RegionLoad forceLoad(const std::string& label, RegionPtr R, const Vec3& F) {
  if (R->area <= 0.0) throw std::runtime_error(label + ": empty region");
  const Vec3 t = F / R->area;
  return {label, R, [t](const RPoint&) { return t; }};
}

inline RegionLoad wrenchLoad(const std::string& label, RegionPtr R, const Vec3& F, const Vec3& M,
                             const Vec3& ref) {
  if (R->area <= 0.0) throw std::runtime_error(label + ": empty region");
  const Vec3 c = R->centroid, t0 = F / R->area;
  const Vec3 Mc = M - (c - ref).cross(F);
  const Vec3 w = R->inertia.ldlt().solve(Mc);
  return {label, R, [t0, w, c](const RPoint& r) { return Vec3(t0 + w.cross(r.x - c)); }};
}

inline RegionLoad bearingLoad(const std::string& label, RegionPtr R, const Vec3& F) {
  const Vec3 d = F.normalized();
  double I = 0.0;
  R->forCloud([&](const RPoint& r) {
    const double c = std::max(0.0, -r.n.dot(d));
    I += c * c * r.w;
  });
  if (I <= 0.0) throw std::runtime_error(label + ": no part of the region faces the bearing direction");
  const double p0 = F.norm() / I;
  return {label, R, [p0, d](const RPoint& r) { return Vec3(-p0 * std::max(0.0, -r.n.dot(d)) * r.n); }};
}

inline std::function<Eigen::Matrix3d(const Vec3&, const Vec3&)> springNT(double kn, double kt) {
  return [kn, kt](const Vec3&, const Vec3& n) {
    return Eigen::Matrix3d(kn * n * n.transpose() + kt * (Eigen::Matrix3d::Identity() - n * n.transpose()));
  };
}

inline std::function<Eigen::Matrix3d(const Vec3&, const Vec3&)> springXYZ(const Vec3& k) {
  return [k](const Vec3&, const Vec3&) { return Eigen::Matrix3d(k.asDiagonal()); };
}

}  // namespace igabem
