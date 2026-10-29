// Plate [-L/2, L/2] x [-1/2, 1/2] x [-t/2, t/2] (L = 1 by default) with a circular
// through-hole of radius a = 0.2 on the z axis (python/make_geometries.py), remote
// stress S along x.
//
// Load cases:
//   Patch       u = alpha x, t = sigma n on every face
//   Kirsch      exact plane-strain Kirsch field, its traction on every face
//               (sigma_zz = nu (sigma_xx + sigma_yy) on z = +-t/2): SCF 3 at every z
//   KirschFree  Kirsch traction on the four sides only; free faces and hole, so a
//               departure from 3 is the thickness (3D) effect
//   Tension     t = S e_x on x = +-L/2, all else free (finite-width plate)
#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <string>
#include <vector>

#include "../kelvin.hpp"
#include "../kirsch_exact_gen.hpp"
#include "../mesh.hpp"
#include "../recovery.hpp"
#include "../solution.hpp"

namespace igabem {

struct PlateHole {
  enum Case { Patch, Kirsch, KirschFree, Tension };
  static constexpr double a = 0.2, S = 1.0, half = 0.5, tol = 1e-9;
  Material mat;
  double halfL = 0.5;  // half length along x
  Eigen::Matrix3d alpha = Eigen::Vector3d(-0.25, -0.25, 1.0).asDiagonal();

  static bool parse(const std::string& s, Case& c) {
    if (s == "patch") c = Patch;
    else if (s == "kirsch") c = Kirsch;
    else if (s == "kirsch_free") c = KirschFree;
    else if (s == "tension") c = Tension;
    else return false;
    return true;
  }
  static bool exact(Case c) { return c == Patch || c == Kirsch; }

  Eigen::Matrix3d kirschStress(const Vec3& x) const {
    Vec3 u;
    Eigen::Matrix3d G;
    kirschField(x(0), x(1), x(2), a, S, mat.mu, mat.nu(), u, G);
    return mat.stress(G);
  }
  Vec3 displacement(Case c, const Vec3& x) const {
    if (c == Patch) return alpha * x;
    Vec3 u;
    Eigen::Matrix3d G;
    kirschField(x(0), x(1), x(2), a, S, mat.mu, mat.nu(), u, G);
    return u;
  }
  Eigen::Matrix3d stress(Case c, const Vec3& x) const {
    return c == Patch ? Eigen::Matrix3d(mat.stress(alpha)) : kirschStress(x);
  }
  Vec3 traction(Case c, const Vec3& y, const Vec3& n) const {
    const bool side = std::abs(y(0)) > halfL - tol || std::abs(y(1)) > half - tol;
    if (exact(c)) return stress(c, y) * n;
    if (c == KirschFree) return side ? Vec3(kirschStress(y) * n) : Vec3::Zero();
    return std::abs(y(0)) > halfL - tol ? Vec3(S * n(0) * Vec3::UnitX()) : Vec3::Zero();
  }

  // Patches of the hole wall (centre on the cylinder r = a).
  static std::vector<bool> holePatches(const Mesh& m) {
    std::vector<bool> hole(m.patches.size(), false);
    for (size_t p = 0; p < m.patches.size(); ++p) {
      const igabem::Patch& P = m.patches[p];
      PatchEval E;
      P.eval(0.5 * (P.U.front() + P.U.back()), 0.5 * (P.V.front() + P.V.back()), E);
      // loose enough for geometry fitted to CAD (python/step_auto.py, error ~1e-5)
      hole[p] = std::abs(std::hypot(E.x(0), E.x(1)) - a) < 1e-3 * a;
    }
    return hole;
  }

  // Exact field for the probes and VTK output (empty when there is none).
  ExactField exactField(Case c) const {
    if (!exact(c)) return nullptr;
    return [this, c](const Vec3& x, Vec3& u, Eigen::Matrix3d& s) { u = displacement(c, x); s = stress(c, x); };
  }

  // Dirichlet patches of the clamped plate: the end face x = -L/2.
  bool clampedEnd(const Vec3& centre) const { return centre(0) < -halfL + tol; }

  // Point on the outer boundary of the mid-plane section in direction theta.
  Vec3 outer(double theta) const {
    const double c = std::cos(theta), s = std::sin(theta);
    const double r = std::min(std::abs(c) > 1e-14 ? halfL / std::abs(c) : 1e300,
                              std::abs(s) > 1e-14 ? half / std::abs(s) : 1e300);
    return Vec3(r * c, r * s, 0.0);
  }

  // sigma_xx at the hole edge (0, a, z) from the solution coefficients c.
  double sxxAtHole(const Mesh& m, const std::vector<bool>& hole, Case cs, const Eigen::VectorXd& c,
                   double z) const {
    const SurfacePoint sp = locate(m, Vec3(0.0, a, z), [&](int p) { return bool(hole[p]); });
    return surfaceStress(m, mat, sp, c, traction(cs, sp.E.x, sp.N))(0, 0);
  }
};

}  // namespace igabem
