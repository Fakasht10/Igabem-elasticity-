// Zeroed storage for large dense operators: an anonymous mapping populated up
// front (MAP_POPULATE), so the kernel zeroes it in one pass instead of faulting
// it in page by page from the assembly. Measured on the 4-core benchmark VM for
// 2.4 GB: Eigen::MatrixXd::Zero 7 s, parallel first touch 3.9 s, transparent
// huge pages 3-13 s, MAP_POPULATE 0.85 s.
#pragma once

#include <sys/mman.h>

#include <cstddef>
#include <new>

namespace igabem {

class HugeBuffer {
 public:
  explicit HugeBuffer(size_t n_doubles) : n_(n_doubles), bytes_(n_doubles * sizeof(double)) {
    void* p = mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    if (p == MAP_FAILED) throw std::bad_alloc();
    p_ = static_cast<double*>(p);
  }
  ~HugeBuffer() { munmap(p_, bytes_); }
  HugeBuffer(const HugeBuffer&) = delete;
  HugeBuffer& operator=(const HugeBuffer&) = delete;
  double* data() { return p_; }
  const double* data() const { return p_; }
  size_t size() const { return n_; }

 private:
  double* p_ = nullptr;
  size_t n_ = 0, bytes_ = 0;
};

}  // namespace igabem
