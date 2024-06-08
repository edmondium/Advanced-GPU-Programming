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
struct MultiplyFunctor {
  using value_type = long;

  ViewType v;
  size_t size;

  MultiplyFunctor(const ViewType& v_, const size_t size_) : v(v_), size(size_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j, const int k) const {
    v(i, j, k) = i * j * k;  
  }
};

template <class ViewType>
struct AddFunctor {
  using value_type = long;

  ViewType v;
  size_t size;

  AddFunctor(const ViewType& v_, const size_t size_) : v(v_), size(size_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j, const int k) const {
    v(i, j, k) = i + j + k;  // compute the product of indices
  }
};


int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  // Bound(s) for MDRangePolicy
  const int n = 100;

  using ScalarType  = int;
  
  // TODO: Write all code inside this scope
  {
  }

  Kokkos::finalize();

  return EXIT_SUCCESS;
}