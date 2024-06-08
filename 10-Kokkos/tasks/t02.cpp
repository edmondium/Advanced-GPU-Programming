#include <Kokkos_Core.hpp>
#include <cstdio>
#include <typeinfo>

// Exercise 02: Replace the functor in ex01.cpp with a lambda function

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  printf("Hello World on Kokkos execution space %s\n",
         typeid(Kokkos::DefaultExecutionSpace).name());

  Kokkos::parallel_for("HelloWorld", 15, [=] __host__ __device__ (const int i) 
  {
    Kokkos::printf("Hello from i = %i\n", i);
  }
  );

  // You must call finalize() after you are done using Kokkos.
  Kokkos::finalize();
}