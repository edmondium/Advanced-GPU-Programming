
#include <Kokkos_Core.hpp>
#include <cstdio>

using view_type = Kokkos::View<double * [3]>;
using host_view_type = view_type::HostMirror;

struct ReduceFunctor {
  view_type a;
  ReduceFunctor(view_type a_) : a(a_) {}
  using value_type = int;  // Specify type for reduction value, lsum

  KOKKOS_INLINE_FUNCTION
  void operator()(int i, int &lsum) const {
    lsum += a(i, 0) - a(i, 1) + a(i, 2);
  }
};

int main() {
  Kokkos::initialize();

  {
    view_type a("A", 10);
    
    // TODO: Initialize a on the host and copy it to the device
    

    int sum = 0;
    Kokkos::parallel_reduce(10, ReduceFunctor(a), sum);
    printf("Result is %i\n", sum);
  }

  Kokkos::finalize();
}