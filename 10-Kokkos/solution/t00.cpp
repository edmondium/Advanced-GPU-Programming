#include <Kokkos_Core.hpp>
#include <cstdio>
#include <typeinfo>

int main(int argc, char* argv[]) {  
  Kokkos::initialize(argc, argv);

  printf("Hello World on Kokkos execution space %s\n", 
  typeid(Kokkos::DefaultExecutionSpace).name());

  // You must call finalize() after you are done using Kokkos.
  Kokkos::finalize();
}