!
!  Copyright 2015 NVIDIA Corporation
!
!  Licensed under the Apache License, Version 2.0 (the "License");
!  you may not use this file except in compliance with the License.
!  You may obtain a copy of the License at
!
!      http://www.apache.org/licenses/LICENSE-2.0
!
!  Unless required by applicable law or agreed to in writing, software
!  distributed under the License is distributed on an "AS IS" BASIS,
!  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
!  See the License for the specific language governing permissions and
!  limitations under the License.
!

program main
! use omp_lib
implicit none

double precision, external :: WTime ! from common.f90
integer, external :: poisson2d_reference
logical, external :: check_results

integer, parameter :: NN = 4096, NM = 4096

real(8), allocatable :: A(:,:), Aref(:,:), Anew(:,:)

real(8) :: t1, t2, t1_ref, t2_ref, runtime, runtime_ref, pi, y0, tol

integer :: i, j, num_openmp_threads, iter, iter_max, ist

  iter_max = 500
  pi    = 2.d0 * asin(1.d0)
  tol   = 1.d-5

  allocate(A(NN,NM), Aref(NN,NM), Anew(NN,NM), stat=ist)

  if (ist /= 0) stop "failed to allocate arrays!"


! set boundary conditions
  do j = 1, NM
    y0 = sin( 2.0 * pi * j / (NM-1))

    A(1,j)    = y0
    A(NN,j)   = y0

    Aref(1,j) = y0
    Aref(NN,j)= y0

    Anew(1,j) = y0
    Anew(NN,j)= y0

  end do ! j

  write(*,fmt="('Jacobi relaxation Calculation: ',i0,' x ',i0,' mesh')") NN,NM

  write(*,fmt="('Calculate reference solution and time CPU execution.')")
  write(*,fmt="('')")
  write(*,fmt="('   Iter')")
  write(*,fmt="('--------')")
  t1_ref = Wtime()
  num_openmp_threads = poisson2d_reference(iter_max, tol, Anew, Aref, NM, NN)
  t2_ref = Wtime()
  runtime_ref = t2_ref-t1_ref
  write(*,fmt="('GPU execution.')")




  t1 = Wtime()

  iter = 0
 



do while (iter < iter_max) 

!! TODO: Use do concurrent to parallelise this loop.  

  
    do concurrent(i=2 : NN-1, j=2 : Nm-1)

      Anew(i,j) = 0.25 * ( A(i+1,j) + A(i-1,j) + A(i,j-1) + A(i,j+1) )

    end do 

! Swap input/output
!! TODO:  swap input / output with do concurrent  
      do concurrent(i=2 : NN-1, j=2 : NM-1)
        A(i,j) = Anew(i,j)
      enddo
! Periodic boundary conditions
!! TODO: apply periodic boundary conditions using do concurrent. 
       do concurrent(i=2 : NN-1)
         A(i,1)  = A(i,NM-1)
       enddo
       do concurrent(i=2 : NN-1)
         A(i,NM) = A(i,2)
       enddo

  if(mod(iter,100) == 0) write(*,fmt="(2x,i4)") iter

  iter = iter + 1
end do ! while

t2 = Wtime()
runtime = t2-t1
if (check_results(Aref, A, NM, NN, tol)) write(*,fmt="(i0,' x ',i0,': ','1 GPU: ',f0.4,'s, ',i0,' CPU cores: ',f0.4,'s',', speedup: ',f0.2)") NN, NM, runtime, num_openmp_threads,runtime_ref,runtime_ref/runtime


end program ! main
