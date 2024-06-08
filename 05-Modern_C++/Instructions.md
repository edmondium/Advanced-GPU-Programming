# Advanced CUDA Course 2024

* Date: 03 -- 07 June 2024
* Location: _online_
* Institute: Jülich Supercomputing Centre

## Session 5: CUDA++

### Task 1: Writing a templated gemm routine

#### Setup

Go to `~/CUDA-Course-Adv/05-Modern_C++/exercises/tasks/gemm`.

In this task you will generalize a gemm routine to arbitrary numerical types.

Please follow the instruction in the source code.

### Task 2: Taking advantage of cuda::std::span

Go to `~/CUDA-Course-Adv/05-Modern_C++/exercises/tasks/axpy`.

In this task you will generalize your kernel so it can be called with different types of containers.

Please follow the instruction in the source code.


### Task 2: A better Mandelbrot

#### Setup

Go to `~/CUDA-Course-Adv/05-Modern_C++/exercises/tasks/Mandelbrot`.

In this taks you'll use C++ to write a better program to calculate the Mandelbrot set.

Step 1: Follow the instructions in the source code to take advantage of cuda::std::complex.

Step 2: Include mandelbrot.cuh and use a variable of type Mandelbrot to pass width, height, and data to the mandelbrot function instead of a raw pointer.


