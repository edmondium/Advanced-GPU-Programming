# Advanced GPU Programming

* Date: 3 -- 7 June 2024
* Location: _online_
* Institute: Jülich Supercomputing Centre

## Session 9: pSTL

We'll be using the Nvidia HPC SDK for these exercise and some Python for plotting. Please load `NVHPC` and `SciPy-Stack` by typing 

```bash
ml NVHPC
ml SciPy-Stack
```

### Task 1: Writing saxpy with std::transform

#### Setup

Go to `~/CUDA-Course-Adv/09-pSTL/exercises/tasks/transform`.

In this task you will write a axpy routine for arbitrary numerical types using std::transform.

Please follow the instruction in the source code.

### Task 2: Writing saxpy using std::for_each

#### Setup

Go to `~/CUDA-Course-Adv/09-CUDA++/exercises/tasks/for_each`.

In this task you will write a axpy routine for arbitrary numerical types using std::for_each and an index array or a counting iterator.

Please follow the instruction in the source code.

### Task 3: Writing saxpy using thrust::for_each

#### Setup

Go to `~/CUDA-Course-Adv/09-CUDA++/exercises/tasks/thrust_for_each`.

In this task you will write a axpy routine for arbitrary numerical types using thrust::for_each and an index array or a counting iterator.

Please follow the instruction in the source code.

### Task 4: Writing a Jacobi solver

#### Setup

Go to `~/CUDA-Course-Adv/09-pSTL/exercises/tasks/jacobi`.

In this task you will learn how to implement the Jacobi solver using algorithms from the standard library.

Please follow the instruction in the source code.

Visualize the final output by running the following cell:


```code
%matplotlib inline
import matplotlib.pyplot as plt
import numpy
data = numpy.loadtxt("exercises/tasks/jacobi/final.dat")
plt.imshow(data)
plt.colorbar()
```


### Task 5: Calculate the Mandelbrot set with std::transform

Start with the escape_time function from 05-Modern_C++ and use std::transform to implement the calculation of the Mandelbrot set.

Create a directory `~/CUDA-Course-Adv/09-pSTL/exercises/tasks/mandelbrot` and start coding.

