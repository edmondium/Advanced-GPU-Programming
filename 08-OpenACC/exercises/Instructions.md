# JSC Advanced CUDA Course 2024

-   Date: 7 June 2024
-   Location: *online*
-   Institute: Jülich Supercomputing Centre

## Session 8: OpenACC Introduction

Please open up a Terminal window via `File` → `New` → `Terminal`.

In the Terminal, make your way to
`~/CUDA-Course-Adv/08-OpenACC/exercises`, which should already be in
your account after bootstrapping the environment.

Load the NVHPC module with `module load NVHPC`, or just
`source setup.sh`.

Choose either C or Fortran as the programming language; the course is
best experienced in C, but Fortran will technically work as well.

You find the tasks to work on in the `Task/` directory. There is also a
directory called `Solutions/` which contains solutions to each of the
tasks you work on. You decide for yourself if and when you look into the
solutions. Don\'t give up too quickly!

This document contains descriptions for each of the six tasks. Please
only work on the individual task as specified in the lecture.

### Task 1: A Simple Profile

_Note: For the Advanced CUDA Course, this task is skipped_

We use `gcc` and `gprof` to generate a simple profile of the
application. Instead of calling invididual commands, we prepared
everything neatly for you with targets in the `Makefile` -- not only for
this task, but for all following.

Go to the `Task1` subdirectory and compile the application with

``` bash
make task1
```

(Usually you only compile with `make`, but this task is special: We use
GCC instead of NVHPC.)

After compilation, you can submit a run to the batch system with

``` bash
make task1_profile
```

(Also this deviates a bit from the following commands; usually you
submit with `make run`, but... this task is special!)

Study the output!

### Task 2: Add Parallel Region

Your first proper programming example!

Make your way to the `Task2` directory in the Jupyter Terminal. In there
you find `poisson2d.c` which is the file we are going to change in this
session. Either open it directly in the Terminal with `vim`, or -- which
we recommend -- make your way to the same file also in Jupyter\'s file
browser drawer and open it in Jupyter\'s file editor by double-clicking
it.

Have a look at the `TODO`! Please add OpenACC parallelism for the
double-`for` loop!

Again, compile and run with the following two commands:

``` bash
make
make run
```

### Task 3: More Parallel Regions

Move to the `Task3` directory and again look at `poisson2d.c`.

You\'ll find new `TODO`s indiciating the region in which you add
individual `parallel loop` or a `kernels` routine(s). Only parallelize
the indiviual `for` loops inside the `while` as indicated in the source
code comments.

As before,

``` bash
make
make run
```

### Task 4: Data Copies

To be more portable and better understand data movement behaviours, we
removed managed memory transfers by not specifying `-gpu=manged` during
compilation.

We now need to add `copy` clauses to the parallel OpenACC reegions. Have
a look at `poission2d.c` in the `Task4` directory, you\'ll find new
`TODO`s.

To compile and run, use

``` bash
make
make run
```

### Task 5: Data Regions

Instead of using individual data copies (as we did in Task 4 with the
`copy` clauses), we rather want to keep the data on the GPU for the
whole runtime of the program.

Please implement the according directive as outlined in the `TODO` of
`Task5`\'s `poisson2d.c`.

Compile and run with

``` bash
make
make run
```

### Task 6: Refactoring

The final task of this session.

To show OpenACC\'s capabilities to work in larger programs, we want to
extract the core double-loop into a dedicated function called
`inner_loop()`. Please add a level of parallelism to this inner loop
(which makes it necessary to declare it a `acc routine`).

As usual, there are `TODO` hints inside `poisson2d.c` of this `Task6`.

Compile and run with

``` bash
make
make run
```

You are done with this task!

**Congratulations for accelerating your first OpenACC program!**
