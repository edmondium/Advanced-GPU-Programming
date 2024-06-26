//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <Kokkos_Core.hpp>
#include <cstdio>

template <class ViewType>
struct MDFunctor3D {
  using value_type = long;

  ViewType v;
  size_t size;

  MDFunctor3D(const ViewType& v_, const size_t size_) : v(v_), size(size_) {}

  // 3D case - used by parallel_for
  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j, const int k) const {
    v(i, j, k) = i * j * k;  // compute the product of indices
  }
};

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  // Bound(s) for MDRangePolicy
  const int n = 100;

  // ViewType aliases for Rank<2>, Rank<3> for example usage
  using ScalarType  = int;
  using ViewType = Kokkos::View<ScalarType***>;

  {
    // Rank<3> Case: Rank, inner iterate pattern, outer iterate pattern provided
    using MDPolicyType_3D = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

    // Construct 3D MDRangePolicy: lower, upper bounds, tile dims provided
    MDPolicyType_3D mdpolicy_3d({{0, 0, 0}}, {{n, n, n}});

    // Construct a 3D view to store result of product of indices
    ViewType a("v3", n, n, n);

    // Execute parallel_for with rank 3 MDRangePolicy
    Kokkos::parallel_for("md3d", mdpolicy_3d, MDFunctor3D<ViewType>(a, n));

    using host_view_type = ViewType::HostMirror;
    host_view_type h_a = Kokkos::create_mirror_view(a);

    Kokkos::deep_copy(h_a, a);

    for (std::size_t i=0; i< n ; ++i) {
        for (std::size_t j=0; j< n ; ++j) {
            for (std::size_t k=0; k< n ; ++k) {
                if( h_a(i,j,k) != i * j * k ){
                    printf("Error: h_a(%d,%d,%d) = %d != %d\n", i, j, k, h_a(i,j,k), i * j * k);
                }
            }
        }
    }

    printf("All values were verified correctly!\n");
  }

  Kokkos::finalize();

  return EXIT_SUCCESS;
}