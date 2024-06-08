## Module 1 Advanced Multi-GPU Programming with MPI

### Task 1 Handling GPU affinity

* run with default settings (see example output below) and report performance
* run with `CUDA_VISIBLE_DEVICES=0,1,2,3` set and report performance
* profile with `CUDA_VISIBLE_DEVICES=0,1,2,3` and inspect with Nsight Systems:

```console
[kraus1@jrc0437 task]$ $JSC_SUBMIT_CMD --interactive --pty /bin/bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 nsys profile --trace=mpi,nvtx,cuda -o jacobi.cvd srun -n 4 ./jacobi -niter 10"
Warning: LBR backtrace method is not supported on this platform. DWARF backtrace method will be used.
Single GPU jacobi relaxation: 10 iterations on 8192 x 8192 mesh
    0, 22.626005
Jacobi relaxation: 10 iterations on 8192 x 8192 mesh
    0, 22.626055
Num GPUs: 4.
8192x8192: 1 GPU:   0.2500 s, 4 GPUs:   0.0794 s, speedup:     3.15, efficiency:    78.74
Generating '/tmp/nsys-report-5128.qdstrm'
[1/1] [========================100%] jacobi.cvd.nsys-rep
Generated:
    /p/home/jusers/kraus1/jureca/cuda-advanced/01-MPI/01-GPU_affinity/exercises/task/jacobi.cvd.nsys-rep
```

* Handle GPU affinity with `MPI_COMM_TYPE_SHARED`
  * Follow `TODO`s in `01-MPI/01-GPU_affinity/exercises/task/jacobi.cpp`
* run without and with `CUDA_VISIBLE_DEVICES=0,1,2,3` set and report performance

#### Make Targets

* `run`: run `jacobi` with `$NP` procs.
* `jacobi`: build `jacobi` bin (default)
* `sanitize`: run with [`compute-sanitizer`](https://docs.nvidia.com/cuda/sanitizer-docs/ComputeSanitizer/index.html)
* `profile`: profile with [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-profiling)

#### Example Output

```console
Ⓒ [kraus1@jwlogin24 task1]$ make run
nvcc -lineinfo -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80 -Xptxas --optimize-float-atomics -std=c++11 jacobi_kernels.cu -c
mpicxx -DUSE_NVTX -I/p/software/jurecadc/stages/2023/software/CUDA/11.7/include -std=c++11 jacobi.cpp jacobi_kernels.o -L/p/software/jurecadc/stages/2023/software/CUDA/11.7/lib64 -lcudart -lnvToolsExt -o jacobi
srun -A exalab -p dc-gpu-devel  -N 1 --ntasks-per-node=4 -n 4 ./jacobi
Single GPU jacobi relaxation: 1000 iterations on 8192 x 8192 mesh
    0, 22.626003
  100, 0.634895
  200, 0.378724
  300, 0.279734
  400, 0.225568
  500, 0.190864
  600, 0.166512
  700, 0.148344
  800, 0.134219
  900, 0.122885
Jacobi relaxation: 1000 iterations on 8192 x 8192 mesh
    0, 22.626051
  100, 0.634932
  200, 0.378759
  300, 0.279747
  400, 0.225576
  500, 0.190874
  600, 0.166513
  700, 0.148353
  800, 0.134230
  900, 0.122891
Num GPUs: 4.
8192x8192: 1 GPU:   5.4542 s, 4 GPUs:   1.4229 s, speedup:     3.83, efficiency:    95.83
```
