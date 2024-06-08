#include <Kokkos_Core.hpp>
#include <cstdio>


struct squaresum {
  using value_type = int;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, int& lsum) const {
    lsum += i * i;  // compute the sum of squares
  }
};

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  const int n = 10;

  int sum = 0;
  Kokkos::parallel_reduce(n, squaresum(), sum);
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