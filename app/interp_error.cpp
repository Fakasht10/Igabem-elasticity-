// Interpolation errors of the Kelvin point-load field (source (1.5, 1.7, 2.0),
// direction (1, 2, 3)/sqrt(14)) in the displacement and traction spaces, as a
// best-approximation reference for the Dirichlet tests.
// Prints: geometry, elements, relative L2 error of u, relative L2 error of t.
#include <cmath>
#include <cstdio>
#include "hsbie.hpp"
using namespace igabem;
int main(int argc, char** argv) {
  Mesh mesh; PartitionOptions popt; popt.enabled = false;
  if (argc < 2) { std::fprintf(stderr, "usage: interp_error <geometry.dat>\n"); return 1; }
  mesh.build(loadDat(argv[1]), popt);
  TractionSpace ts; ts.build(mesh);
  Material mat; Kelvin K(mat);
  Eigen::Vector3d xs(1.5, 1.7, 2.0), dir = Eigen::Vector3d(1,2,3).normalized();
  auto uex = [&](const Vec3& y) -> Vec3 { Eigen::Matrix3d U,T; K.eval(y-xs, Vec3::UnitZ(), U, T); return U.transpose()*dir; };
  auto tex = [&](const Vec3& y, const Vec3& n) -> Vec3 { Eigen::Matrix3d U,T; K.eval(y-xs, n, U, T); return T.transpose()*dir; };
  Eigen::VectorXd uc = interpolateDisplacement(mesh, uex);
  // traction interpolation at traction collocation points (per patch)
  Eigen::VectorXd tc(3*ts.ndof);
  for (size_t ip=0; ip<mesh.patches.size(); ++ip) {
    const Patch& P = mesh.patches[ip]; int nl = P.nu*P.nv, off = ts.offset[ip];
    Eigen::MatrixXd Nm = Eigen::MatrixXd::Zero(nl,nl), rhs(nl,3);
    for (int a=0;a<nl;++a){ const auto& c = ts.points[off+a];
      for (int l=0;l<c.E.n_loc;++l) Nm(a,c.E.idx[l]) += c.E.R[l];
      rhs.row(a) = tex(c.x, c.N).transpose(); }
    Eigen::MatrixXd s = Nm.partialPivLu().solve(rhs);
    for (int a=0;a<nl;++a) tc.segment<3>(3*(off+a)) = s.row(a).transpose();
  }
  // u error
  GaussRule g = gaussLegendre(10); PatchEval E; double nu=0,du=0;
  for (auto& el: mesh.elements) for (int i=0;i<10;++i) for(int j=0;j<10;++j){
    const Rect& R = el.rect; mesh.patches[el.patch].evalSpan(el.su,el.sv,R.u0+R.du()*g.x[i],R.v0+R.dv()*g.x[j],E);
    double w=g.w[i]*g.w[j]*R.du()*R.dv()*E.normal().norm(); Vec3 uh=Vec3::Zero();
    for(int l=0;l<E.n_loc;++l) uh+=E.R[l]*uc.segment<3>(3*mesh.l2g[el.patch][E.idx[l]]);
    nu+=w*(uh-uex(E.x)).squaredNorm(); du+=w*uex(E.x).squaredNorm(); }
  std::printf("%s,%zu,%.6e,%.6e\n", argv[1], mesh.elements.size(), std::sqrt(nu/du), tractionL2Error(mesh, ts, tc, tex));
}
