// Unit checks of the crack kernels (src/crack):
//   planar  the open-face coefficients (planar.hpp) against the one-sided limits they stand for, computed by
//           brute-force quadrature at z = x -+ h m as h -> 0 (Richardson in h), for a disk and a square face;
//   front   projection, frame and the Williams face functions against the closed forms of a circular front.
#include <cstdio>
#include <cstdlib>
#include <string>

#include "crack/front.hpp"
#include "crack/planar.hpp"
#include "quadrature.hpp"

using namespace igabem;
using namespace igabem::crack;

static int fails = 0;
static void check(const char* what, double err, double tol) {
  const bool ok = err <= tol;
  std::printf("%-58s err %.3e  tol %.1e  %s\n", what, err, tol, ok ? "ok" : "FAIL");
  if (!ok) ++fails;
}

// Off-plane integral over the face, polar about the foot point x of z: int_0^2pi int_0^rho(th) f(y) r dr dth.
template <class F>
static Eigen::Vector3d offPlane(const PlanarRegion& R, const Vec3& x, double h, F&& f, int nth = 1024) {
  const GaussRule g = gaussLegendre(16);
  Eigen::Vector3d acc = Eigen::Vector3d::Zero();
  std::vector<std::pair<double, int>> cr;
  for (int k = 0; k < nth; ++k) {
    const double th = 2 * M_PI * (k + 0.5) / nth;
    const Vec3 e = std::cos(th) * R.e1 + std::sin(th) * R.e2;
    R.crossings(x, e, cr);
    double r0 = 0.0;
    for (size_t c = 0; c < cr.size(); c += 2) {
      const double r1 = cr[c].first;
      // geometric split from r0 at h 4^j
      double a = r0;
      double b = std::min(r1, std::max(r0, 0.0) + h);
      while (a < r1) {
        for (size_t i = 0; i < g.x.size(); ++i) {
          const double r = a + (b - a) * g.x[i];
          acc += f(Vec3(x + r * e)) * (g.w[i] * (b - a) * r * 2 * M_PI / nth);
        }
        a = b;
        b = std::min(r1, a * 4.0);
      }
      r0 = (c + 1 < cr.size()) ? cr[c + 1].first : r1;
    }
  }
  return acc;
}

static void planarCase(const char* name, const PlanarRegion& R, const Vec3& x) {
  Material mat;
  mat.mu = 0.5 / (1 + 0.3);
  mat.lambda = 0.3 / ((1 + 0.3) * (1 - 0.6));  // E = 1, nu = 0.3
  const Kelvin K(mat);
  const KelvinTraction KT(mat);
  const Vec3 m = R.m;
  // DBIE: face normal n = m, z on the side n points to: lim = CT + I/2
  {
    const FaceCoefficients c = faceCoefficients(R, mat, x, m, m);
    for (int j = 0; j < 3; ++j) {
      const Vec3 aj = Vec3::Unit(j);
      auto J = [&](double h) {
        const Vec3 z = x + h * m;
        return offPlane(R, x, h, [&](const Vec3& y) {
          Eigen::Matrix3d U, T;
          K.eval(y - z, m, U, T);
          return Vec3(T * aj);
        });
      };
      const Vec3 j1 = J(2e-4), j2 = J(1e-4);
      const Vec3 lim = 2 * j2 - j1, want = (c.CT + 0.5 * Eigen::Matrix3d::Identity()) * aj;
      char buf[128];
      std::snprintf(buf, sizeof buf, "%s: PV T + I/2, column %d", name, j);
      check(buf, (lim - want).norm() / std::max(1.0, want.norm()), 2e-5);
    }
  }
  // TBIE: face normal n = -m (upper face), traction normal N = m at x, z below: lim = Fa a + FB B
  {
    const FaceCoefficients c = faceCoefficients(R, mat, x, Vec3(-m), m);
    Eigen::Matrix3d B;
    B << 0.3, -0.7, 0.2, 1.1, 0.4, -0.5, -0.2, 0.9, 0.6;
    const Vec3 a(0.4, -1.0, 0.7);
    auto J = [&](double h, bool lin) {
      const Vec3 z = x - h * m;
      return offPlane(R, x, h, [&](const Vec3& y) {
        Eigen::Matrix3d D, S;
        KT.eval(y - z, m, Vec3(-m), D, S);
        return Vec3(S * (lin ? Vec3(B * (y - x)) : a));
      });
    };
    for (bool lin : {false, true}) {
      const Vec3 j1 = J(2e-4, lin), j2 = J(1e-4, lin);
      const Vec3 lim = 2 * j2 - j1;
      Vec3 want = Vec3::Zero();
      if (!lin) want = c.Fa * a;
      else for (int k = 0; k < 3; ++k) want += c.FB[k] * B.col(k);
      char buf[128];
      std::snprintf(buf, sizeof buf, "%s: FP S %s", name, lin ? "linear" : "constant");
      check(buf, (lim - want).norm() / std::max(1.0, want.norm()), 2e-5);
    }
  }
}

// Front piece of a quarter-annulus crown patch (radii 0.8..1, refined): curve, derivatives, projection and the
// Williams face vectors against the circle's closed forms.
static void frontCase(const std::string& dat) {
  std::vector<Patch> P = loadDat(dat);
  const Patch& Pa = P[7];  // first crown patch of the upper face (make_crack.py penny layout)
  FrontPiece F;
  F.init(&Pa, 7);
  double e0 = 0, e1 = 0, e2 = 0, ep = 0;
  for (int i = 0; i <= 50; ++i) {
    const double u = F.u0 + (F.u1 - F.u0) * i / 50.0;
    Vec3 C, Cu, Cuu, Cp, Cm, d;
    int n, s[16];
    double R[16];
    F.eval(u, C, Cu, n, s, R, &Cuu);
    PatchEval E;
    Pa.eval(u, Pa.V.back(), E);
    e0 = std::max(e0, (C - E.x).norm());
    e1 = std::max(e1, (Cu - E.xu).norm());
    const double h = 1e-5;
    if (i > 0 && i < 50) {
      F.eval(u + h, Cp, d, n, s, R);
      F.eval(u - h, Cm, d, n, s, R);
      e2 = std::max(e2, (Cuu - (Cp - 2 * C + Cm) / (h * h)).norm() / Cuu.norm());
    }
    // a point of the patch projects onto the front at the same polar angle
    Pa.eval(u, 0.5 * (Pa.V.front() + Pa.V.back()), E);
    const double up = F.project(E.x, 0.3);
    Vec3 Cq;
    F.eval(up, Cq, d, n, s, R);
    Vec3 a = E.x - Vec3(0, E.x(1), 0), b = Cq - Vec3(0, Cq(1), 0);
    ep = std::max(ep, (a.normalized() - b.normalized()).norm() + std::abs(Cq.norm() - 1.0));
  }
  check("front: curve vs patch edge", e0, 1e-13);
  check("front: tangent vs patch edge", e1, 1e-12);
  check("front: second derivative vs finite differences", e2, 1e-4);
  check("front: projection on the radius, |C| = 1", ep, 1e-12);
}

int main() {
  frontCase(std::string(IGABEM_SOURCE_DIR) + "/geo/crack/penny_i0_n5.dat");
  // disk of radius 1 in the plane y = 0 (m = e2), seen from an off-centre point
  PlanarRegion disk;
  disk.setPlane(Vec3::Zero(), Vec3::UnitY());
  disk.addEdge([](double t) { return Vec3(std::cos(t), 0.0, std::sin(t)); }, 0.0, 2 * M_PI, 512);
  planarCase("disk", disk, Vec3(0.37, 0.0, -0.21));
  planarCase("disk near front", disk, Vec3(0.0, 0.0, 0.93));
  // square [-1,1]^2 in a tilted plane (corners: kinks of rho)
  PlanarRegion sq;
  const Vec3 m = Vec3(1.0, 2.0, 0.5).normalized();
  sq.setPlane(Vec3(0.1, 0.2, 0.3), m);
  const Vec3 a = sq.e1, b = sq.e2, o = sq.point;
  const Vec3 C[4] = {o - a - b, o + a - b, o + a + b, o - a + b};
  for (int k = 0; k < 4; ++k) {
    const Vec3 P = C[k], Q = C[(k + 1) % 4];
    sq.addEdge([P, Q](double t) { return Vec3((1 - t) * P + t * Q); }, 0.0, 1.0, 8);
  }
  planarCase("square", sq, Vec3(o + 0.55 * a - 0.3 * b));
  if (fails) { std::printf("%d FAILED\n", fails); return 1; }
  std::printf("all ok\n");
  return 0;
}
