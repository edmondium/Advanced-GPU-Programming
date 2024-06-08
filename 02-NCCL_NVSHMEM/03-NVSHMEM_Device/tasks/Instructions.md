# Hands-On 3 (optional): Device-initiated Communication with NVSHMEM

## Task: Using the NVSHMEM device API

### Description

In our last hands-on, we use the NVSHMEM device API instead of MPI to implement a multi-GPU Jacobi solver. The starting point of this task is the MPI variant of the Jacobi solver. You need to work on `TODOs` in `jacobi.cu`:

- Initialize NVSHMEM (this is same as in previous hands-on 2):
  - Include NVSHMEM headers.
  - Initialize and shutdown NVSHMEM using `MPI_COMM_WORLD`.
  - Allocate work arrays `a` and `a_new` from the NVSHMEM symmetric heap. Again, ensure the size you pass in is the same value on all participating ranks!
  - Calculate halo/boundary row index of top and bottom neighbors.
  - Add necessary inter PE synchronization.
- Modify `jacobi_kernel` (this is the device-initiated part):
  - Pass in halo/boundary row index of top and bottom neighbors.
  - Use `nvshmem_float_p` to directly push values needed by top and bottom neighbors from the kernel.
  - Remove no longer needed MPI communication.

Compile with

``` {.bash}
make
```

Submit your compiled application to the batch system with

``` {.bash}
make run
```

Study the performance by inspecting the profile generated with
`make profile`. For `make run` and `make profile`, the environment variable `NP` can be set to change the number of processes.

#### Note


The Slurm installation on the JSC systems sets `CUDA_VISIBLE_DEVICES` automatically so that each spawned process only sees the GPU it should use 
(see e.g. [GPU Devices](https://apps.fz-juelich.de/jsc/hps/juwels/booster-overview.html#gpu-devices) in the JUWELS Booster Overview documentation).

This is not supported for NVSHMEM. 
The automatic setting of `CUDA_VISIBLE_DEVICES` can be disabled by setting `CUDA_VISIBLE_DEVICES=0,1,2,3` in the shell that executes `srun`. 

With `CUDA_VISIBLE_DEVICES` set explicitly, all spawned processes can see all GPUs listed. This is automatically done for the `sanitize`, `run` and `profile` make targets.

## Advanced Task: Use `nvshmemx_float_put_nbi_block`

### Description

This is an optional task to use `nvshmemx_float_put_nbi_block` instead of `nvshmem_float_p` for more efficient multi node execution. There are no `TODO`s prepared. Use the solution of the previous task as a starting point. Some tips:

- You only need to change `jacobi_kernel`.
- Switching to a 1-dimensional CUDA block can simplify the task.
- The difficult part is calculating the right offsets and size for calling into `nvshmemx_float_put_nbi_block`.
- If a CUDA blocks needs to communicate data with `nvshmemx_float_put_nbi_block`, all threads in that block need to call into `nvshmemx_float_put_nbi_block`.
- The [`nvshmem_opt`](https://github.com/NVIDIA/multi-gpu-programming-models/blob/master/nvshmem_opt/jacobi.cu#L154) variant in the [Multi GPU Programming Models Github repository](https://github.com/NVIDIA/multi-gpu-programming-models) implements the same strategy.
