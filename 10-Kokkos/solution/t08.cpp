#include <Kokkos_Core.hpp>
#include <cstdio>


using view_type = Kokkos::View<double*[3]>;


struct InitView {
  view_type a;

  InitView(view_type a_) : a(a_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    a(i, 0) = 1.0 * i;
    a(i, 1) = 1.0 * i * i;
    a(i, 2) = 1.0 * i * i * i;
  }
};

struct ReduceFunctor {
  view_type a;

  ReduceFunctor(view_type a_) : a(a_) {}

  using value_type = double;

  KOKKOS_INLINE_FUNCTION
  void operator()(int i, double& lsum) const {
    lsum += a(i, 0) * a(i, 1) / (a(i, 2) + 0.1);
  }
};

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    const int N = 10;
    view_type a("A", N);

    auto i0 = Kokkos::pair(0,2);
    

    Kokkos::parallel_for(N, InitView(a));
    double sum = 0;

    // TODO : Declare a Subview and use it in the parallel_reduce
    
    
    printf("Result: %f\n", sum);
    
  }  
  Kokkos::finalize();
}