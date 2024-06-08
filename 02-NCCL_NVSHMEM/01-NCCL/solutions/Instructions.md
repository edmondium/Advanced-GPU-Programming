# Hands-On 1: Using NCCL for Inter-GPU Communication

## Description

The purpose of this task is to use NCCL instead of MPI to implement a multi-GPU Jacobi solver. The starting point of this task is the MPI variant of the Jacobi solver. You need to work on `TODO`s in `jacobi.cpp`:

- Initialize NCCL:
  - Include NCCL headers
  - Create a NCCL unique ID, and initialize it
  - Create a NCCL communicator and initilize it
- Replace the `MPI_Sendrecv` calls with `ncclRecv` and `ncclSend` calls for the warmup stage
  - Replace MPI for the periodic boundary conditions with NCCL 
  - Fix output message to indicate NCCL rather than MPI
  - Destroy NCCL comunicator

Compile with

``` {.bash}
make
```

Submit your compiled application to the batch system with

``` {.bash}
make run
```

Study the performance by inspecting the profile generated with
`make profile`. 

For `make run` and `make profile`, the environment variable `NP` can be set to change the number of processes.

