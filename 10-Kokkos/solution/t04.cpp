#include <Kokkos_Core.hpp>
#include <cstdio>


// TODO : Implement a reduction functor

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  const int n = 10;

  int sum = 0;
  // TODO : Call the reduction functor
  printf(
      "Sum of squares of integers from 0 to %i, "
      "computed in parallel, is %i\n",
      n - 1, sum);

  // Compare to a sequential loop.
  int seqSum = 0;
  for (int i = 0; i < n; ++i) {
    seqSum += i * i;
  }
  printf(
      "Sum of squares of integers from 0 to %i, "
      "computed sequentially, is %i\n",
      n - 1, seqSum);
  Kokkos::finalize();
  return (sum == seqSum) ? 0 : -1;
}