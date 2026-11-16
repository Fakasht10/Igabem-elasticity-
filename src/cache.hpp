// On-disk cache of assembled operators. The key is a hash of everything the
// operators depend on: the geometry file's bytes, the material, the quadrature
// and partition options, and a format version. Any change to one of them gives
// a new key, so a stale operator is never reused.
#pragma once

#include <Eigen/Dense>
#include <Eigen/LU>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <unistd.h>
#include <sstream>
#include <string>
#include <vector>

#include "kelvin.hpp"
#include "mesh.hpp"
#include "quadrature.hpp"

namespace igabem {

inline std::uint64_t fnv1a(const std::string& s, std::uint64_t h = 1469598103934665603ull) {
  for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
  return h;
}

inline std::string readFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Bump when the assembly itself changes (quadrature rules, kernels), so that
// operators assembled by older code are not reused.
constexpr int kOperatorVersion = 2;  // 2: singular cells split around the source point

inline std::string operatorKey(const std::string& geometry_file, const std::string& kind,
                               const Material& mat, const SolverOptions& so,
                               const PartitionOptions& po) {
  std::ostringstream ss;
  ss.precision(17);
  ss << kind << '|' << kOperatorVersion << '|' << mat.lambda << ',' << mat.mu << '|' << so.n_gauss << ',' << so.near_refine << ','
     << so.near_max_depth << ',' << so.far_tol << '|' << po.enabled << ',' << po.kappa_h << ','
     << po.aspect << ',' << po.taper << ',' << po.max_depth;
  char hex[17];
  std::snprintf(hex, sizeof hex, "%016llx",
                static_cast<unsigned long long>(fnv1a(ss.str(), fnv1a(readFile(geometry_file)))));
  return kind + "_" + hex;
}

// Written to a temporary name and renamed: a reader sharing the cache directory (another process, a shared
// operator library) never sees a partial file.
inline void saveMatrices(const std::string& path, const std::vector<const Eigen::MatrixXd*>& M) {
  const std::string tmp = path + ".tmp" + std::to_string(::getpid());
  {
  std::ofstream f(tmp, std::ios::binary);
  const char magic[8] = {'I', 'G', 'B', 'E', 'M', 'O', 'P', '1'};
  f.write(magic, 8);
  const std::int64_t n = std::int64_t(M.size());
  f.write(reinterpret_cast<const char*>(&n), sizeof n);
  for (const auto* m : M) {
    const std::int64_t r = m->rows(), c = m->cols();
    f.write(reinterpret_cast<const char*>(&r), sizeof r);
    f.write(reinterpret_cast<const char*>(&c), sizeof c);
    f.write(reinterpret_cast<const char*>(m->data()), std::streamsize(sizeof(double) * r * c));
  }
  if (!f) { f.close(); std::remove(tmp.c_str()); return; }
  }
  std::rename(tmp.c_str(), path.c_str());
}

// False if the file is missing or malformed.
inline bool loadMatrices(const std::string& path, const std::vector<Eigen::MatrixXd*>& M) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  char magic[8];
  f.read(magic, 8);
  if (!f || std::string(magic, 8) != "IGBEMOP1") return false;
  std::int64_t n;
  f.read(reinterpret_cast<char*>(&n), sizeof n);
  if (!f || n != std::int64_t(M.size())) return false;
  for (auto* m : M) {
    std::int64_t r, c;
    f.read(reinterpret_cast<char*>(&r), sizeof r);
    f.read(reinterpret_cast<char*>(&c), sizeof c);
    if (!f || r < 0 || c < 0) return false;
    m->resize(r, c);
    f.read(reinterpret_cast<char*>(m->data()), std::streamsize(sizeof(double) * r * c));
    if (!f) return false;
  }
  return true;
}

// Hash of the bytes of a vector or matrix, chained onto h.
template <class V>
std::uint64_t fnvData(const V& v, std::uint64_t h) {
  return fnv1a(std::string(reinterpret_cast<const char*>(v.data()), sizeof(*v.data()) * v.size()), h);
}

// Partial-pivoting LU that can be written to and read from the operator cache: a new load case on a stored
// operator then costs two triangular solves, not a factorisation.
struct StoredLU : Eigen::PartialPivLU<Eigen::MatrixXd> {
  using Eigen::PartialPivLU<Eigen::MatrixXd>::PartialPivLU;
  void save(const std::string& path) const {
    Eigen::MatrixXd P(m_p.size(), 1), meta(2, 1);
    for (Eigen::Index i = 0; i < m_p.size(); ++i) P(i) = m_p.indices()(i);
    meta << m_l1_norm, m_det_p;
    saveMatrices(path, {&m_lu, &P, &meta});
  }
  bool load(const std::string& path, Eigen::Index n) {
    Eigen::MatrixXd P, meta;
    if (!loadMatrices(path, {&m_lu, &P, &meta}) || m_lu.rows() != n || m_lu.cols() != n || P.size() != n) return false;
    m_p.resize(n);
    for (Eigen::Index i = 0; i < n; ++i) m_p.indices()(i) = int(P(i));
    m_rowsTranspositions.resize(0);
    m_l1_norm = meta(0);
    m_det_p = static_cast<signed char>(meta(1));
    m_isInitialized = true;
    return true;
  }
};

}  // namespace igabem
