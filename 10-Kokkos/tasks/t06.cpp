
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
    host_view_type h_a = Kokkos::create_mirror_view(a);

    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < 3; j++) {
        h_a(i, j) = i * 10 + j;
      }
    }
    Kokkos::deep_copy(a, h_a);  // Copy from host to device.

    int sum = 0;
    Kokkos::parallel_reduce(10, ReduceFunctor(a), sum);
    printf("Result is %i\n", sum);
  }

  Kokkos::finalize();
}