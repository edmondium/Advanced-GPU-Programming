# Kokkos Tutorial

## How to Compile

0. Setup the Environment

```
module purge 
module load GCC CUDA hwloc CMake git
```

1. Configure the Code

```
# Run in Source Directory
cmake --preset system -B build -S .
cmake --build build --target ex01
srun <options> build/ex01
```