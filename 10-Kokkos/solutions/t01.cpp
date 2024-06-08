#include <Kokkos_Core.hpp>
#include <cstdio>
#include <typeinfo>

// Exercise 01: We show how to use a functor with Kokkos::parallel_for
struct hello_world {
  __host__ __device__ 
  void operator()(const int i) const {
    Kokkos::printf("Hello from i = %i\n", i);
  }
};

int main(int argc, char* argv[]) {  
  Kokkos::initialize(argc, argv);

  printf("Hello World on Kokkos execution space %s\n", 
  typeid(Kokkos::DefaultExecutionSpace).name());

  auto functor = hello_world();

  Kokkos::parallel_for("HelloWorld", 15, functor);

  // You must call finalize() after you are done using Kokkos.
  Kokkos::finalize();
}