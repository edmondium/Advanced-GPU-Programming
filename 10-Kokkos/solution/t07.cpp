#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>
#include <cstdio>
#include <cstdlib>

using view_type = Kokkos::View<double*, Kokkos::CudaUVMSpace>;
using idx_type  = Kokkos::View<int**, Kokkos::CudaUVMSpace>;

template <class Device>
struct localsum {
  idx_type::const_type idx;
  view_type dest;

  using dev_view_type = 
    Kokkos::View<view_type::const_data_type, view_type::array_layout,
               view_type::device_type>;
  
  
   dev_view_type src;

  localsum(idx_type idx_, view_type dest_, view_type src_)
      : idx(idx_), dest(dest_), src(src_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(int i) const {
    double tmp = 0.0;
    for (int j = 0; j < int(idx.extent(1)); j++) {
      const double val = src(idx(i, j));
      tmp += val * val + 0.5 * (idx.extent(0) * val - idx.extent(1) * val);
    }
    dest(i) += tmp;
  }
};

int main(int narg, char* arg[]) {
  Kokkos::initialize(narg, arg);

  {
    int size = 1000000;

    // Create Views
    idx_type idx("Idx", size, 64);
    view_type dest("Dest", size);
    view_type src("Src", size);

    srand(134231);

    Kokkos::fence();

    // When using UVM Cuda views can be accessed on the Host directly
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < int(idx.extent(1)); j++)
        idx(i, j) = (size + i + (rand() % 500 - 250)) % size;
    }

    Kokkos::fence();
    Kokkos::Timer timer;
    Kokkos::parallel_for(size,
                         localsum<view_type::execution_space>(idx, dest, src));
    Kokkos::fence();
    double sec1_dev = timer.seconds();

    // No data transfer will happen now, since nothing is accessed on the host
    timer.reset();
    Kokkos::parallel_for(size,
                         localsum<view_type::execution_space>(idx, dest, src));
    Kokkos::fence();
    double sec2_dev = timer.seconds();

    timer.reset();
    Kokkos::parallel_for(
        size, localsum<Kokkos::HostSpace::execution_space>(idx, dest, src));
    Kokkos::fence();
    double sec1_host = timer.seconds();

    // No data transfers will happen now
    timer.reset();
    Kokkos::parallel_for(
        size, localsum<Kokkos::HostSpace::execution_space>(idx, dest, src));
    Kokkos::fence();
    double sec2_host = timer.seconds();

    printf("Device Time with Sync: %e without Sync: %e \n", sec1_dev, sec2_dev);
    printf("Host   Time with Sync: %e without Sync: %e \n", sec1_host,
           sec2_host);
  }

  Kokkos::finalize();
}