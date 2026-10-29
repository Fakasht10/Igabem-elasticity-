// Boundary solution of one load case: displacement coefficients in the
// conforming space and traction coefficients in the traction space, whatever
// was prescribed and whatever was solved for. Everything downstream (stress
// recovery, interior evaluation, probes, VTK output) works from this.
// Tractions on sub-regions (local loads, springs; bc_model.hpp) are kept as
// what they are, discontinuous region densities on top of the traction-space
// part, and enter traction(), the stress recovery and the interior evaluation.
#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <memory>
#include <vector>

#include "kelvin.hpp"
#include "mesh.hpp"
#include "recovery.hpp"
#include "regions.hpp"
#include "traction_space.hpp"

namespace igabem {

struct BoundarySolution {
  const Mesh* mesh = nullptr;
  const TractionSpace* ts = nullptr;
  Material mat;
  Eigen::VectorXd u;  // 3 per displacement dof
  Eigen::VectorXd t;  // 3 per traction dof
  std::vector<std::shared_ptr<const RegionDensity>> region;  // column 0 of each, added to t

  Vec3 displacement(int patch, const PatchEval& E) const {
    Vec3 v = Vec3::Zero();
    for (int l = 0; l < E.n_loc; ++l) v += E.R[l] * u.segment<3>(3 * mesh->l2g[patch][E.idx[l]]);
    return v;
  }
  Vec3 traction(int patch, const PatchEval& E) const {
    Vec3 v = Vec3::Zero();
    const int off = ts->offset[patch];
    for (int l = 0; l < E.n_loc; ++l) v += E.R[l] * t.segment<3>(3 * (off + E.idx[l]));
    if (!region.empty()) {
      const Vec3 n = (E.normal() * mesh->normal_sign).normalized();
      for (const auto& d : region)
        if (d->R->contains(patch, E.x, n)) v += d->t(RPoint{patch, E.x, n, 1.0, E.n_loc, E.idx, E.R}).col(0);
    }
    return v;
  }
  Vec3 displacement(const SurfacePoint& s) const { return displacement(s.patch, s.E); }
  Vec3 traction(const SurfacePoint& s) const { return traction(s.patch, s.E); }
  // Full displacement gradient on the boundary: tangential part from u_h, normal
  // part from t_h through Hooke's law (recovery.hpp).
  Eigen::Matrix3d gradient(const SurfacePoint& s) const {
    return surfaceGradient(mat, tangentialGradient(*mesh, s.patch, s.E, u), traction(s), s.N);
  }
  Eigen::Matrix3d stress(const SurfacePoint& s) const { return mat.stress(gradient(s)); }
};

// Exact field, if any: fills u and sigma at x.
using ExactField = std::function<void(const Vec3& x, Vec3& u, Eigen::Matrix3d& sigma)>;

inline Eigen::Matrix3d strainOf(const Eigen::Matrix3d& grad) { return 0.5 * (grad + grad.transpose()); }

inline double vonMises(const Eigen::Matrix3d& s) {
  const Eigen::Matrix3d d = s - s.trace() / 3.0 * Eigen::Matrix3d::Identity();
  return std::sqrt(1.5 * d.squaredNorm());
}

}  // namespace igabem
