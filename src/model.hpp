// One entry point for drivers: a traction-only (NeumannModel) or a mixed
// (MixedModel) problem chosen at run time, with the same interface. Both read
// the same cached SBIE rows, so switching boundary conditions never reassembles.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "mixed_model.hpp"
#include "neumann_model.hpp"

namespace igabem {

class Model {
 public:
  // use_dirichlet(patch, centre, normal); an empty function gives the Neumann problem.
  using Selector = std::function<bool(int, const Vec3&, const Vec3&)>;

  void setup(const std::string& geometry_file, const Material& m, const SolverOptions& o,
             const Selector& use_dirichlet = nullptr, const std::string& cache_dir = "cache") {
    if (use_dirichlet) {
      mixed_ = std::make_unique<MixedModel>();
      mixed_->setup(geometry_file, m, o, use_dirichlet, cache_dir);
    } else {
      neumann_ = std::make_unique<NeumannModel>();
      neumann_->setup(geometry_file, m, o, cache_dir);
    }
  }

  bool mixed() const { return bool(mixed_); }
  const Mesh& mesh() const { return mixed_ ? mixed_->mesh : neumann_->mesh; }
  const TractionSpace& ts() const { return mixed_ ? mixed_->ts : neumann_->ts; }
  int unknowns() const { return mixed_ ? mixed_->unknowns() : neumann_->unknowns(); }
  bool fromCache() const { return mixed_ ? mixed_->from_cache : neumann_->from_cache; }
  double tAssemble() const { return mixed_ ? mixed_->t_assemble : neumann_->t_assemble; }
  double tLoad() const { return mixed_ ? mixed_->t_load : neumann_->t_load; }
  double tFactor() const { return mixed_ ? mixed_->t_factor : neumann_->t_factor; }
  const NeumannModel* neumann() const { return neumann_.get(); }
  const MixedModel* mixedModel() const { return mixed_.get(); }

  // traction(y, n) and displacement(x) return 3 x K TractionBlocks; displacement
  // is used only on Dirichlet patches.
  template <class Ft, class Fu>
  std::vector<BoundarySolution> solutions(Ft&& traction, Fu&& displacement, int K) const {
    return mixed_ ? mixed_->solutions(traction, displacement, K) : neumann_->solutions(traction, K);
  }

 private:
  std::unique_ptr<NeumannModel> neumann_;
  std::unique_ptr<MixedModel> mixed_;
};

}  // namespace igabem
