## Module 1 Advanced Multi-GPU Programming with MPI

### Task 2 Experiment with GPUDirect

* run and profile with default settings, report performance and inspect with Nsight Systems
* run and profile without GPUDirect P2P (exclude UCX `cuda_ipc` TL) `UCX_TLS=rc_x,self,sm,cuda_copy`, report performance and inspect with Nsight Systems
* run and profile without GPUDirect P2P `UCX_TLS=rc_x,self,sm,cuda_copy` and GPUDirect RDMA disabled `UCX_IB_GPU_DIRECT_RDMA=no`, report performance and inspect with Nsight Systems

#### Make Targets

* `run`: run `jacobi` with `$NP` procs.
* `jacobi`: build `jacobi` bin (default)
* `sanitize`: run with [`compute-sanitizer`](https://docs.nvidia.com/cuda/sanitizer-docs/ComputeSanitizer/index.html)
* `profile`: profile with [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-profiling)

#### Example Output

```console
[kraus1@jrlogin01 code]$ UCX_TLS=rc_x,self,sm,cuda_copy make profile
nvcc -lineinfo -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80 -Xptxas --optimize-float-atomics -std=c++11 jacobi_kernels.cu -c
mpicxx -DUSE_NVTX -I/p/software/jurecadc/stages/2023/software/CUDA/11.7/include -std=c++11 jacobi.cpp jacobi_kernels.o -L/p/software/jurecadc/stages/2023/software/CUDA/11.7/lib64 -lcudart -lnvToolsExt -o jacobi
srun -A exalab -p dc-gpu-devel  -N 1 --ntasks-per-node=4 -n 4 nsys profile --trace=mpi,cuda,nvtx -o jacobi.%q{PMIX_RANK} ./jacobi -niter 10
Warning: LBR backtrace method is not supported on this platform. DWARF backtrace method will be used.
Warning: LBR backtrace method is not supported on this platform. DWARF backtrace method will be used.
Warning: LBR backtrace method is not supported on this platform. DWARF backtrace method will be used.
Warning: LBR backtrace method is not supported on this platform. DWARF backtrace method will be used.
Single GPU jacobi relaxation: 10 iterations on 8192 x 8192 mesh
    0, 22.626007
Jacobi relaxation: 10 iterations on 8192 x 8192 mesh
    0, 22.626051
Num GPUs: 4.
8192x8192: 1 GPU:   0.0700 s, 4 GPUs:   0.0306 s, speedup:     2.29, efficiency:    57.21
Generating '/tmp/nsys-report-4a2b.qdstrm'
Generating '/tmp/nsys-report-0b03.qdstrm'
Generating '/tmp/nsys-report-2f4f.qdstrm'
Generating '/tmp/nsys-report-25cb.qdstrm'
[1/1] [========================100%] jacobi.2.nsys-rep
Generated:
    /p/home/jusers/kraus1/jureca/cuda-advanced/01-MPI/02-GPUDirect/code/jacobi.2.nsys-rep
[1/1] [========================100%] jacobi.1.nsys-rep
[1/1] [========================100%] jacobi.0.nsys-rep
Generated:
    /p/home/jusers/kraus1/jureca/cuda-advanced/01-MPI/02-GPUDirect/code/jacobi.0.nsys-rep
Generated:
    /p/home/jusers/kraus1/jureca/cuda-advanced/01-MPI/02-GPUDirect/code/jacobi.1.nsys-rep
[1/1] [========================100%] jacobi.3.nsys-rep
Generated:
    /p/home/jusers/kraus1/jureca/cuda-advanced/01-MPI/02-GPUDirect/code/jacobi.3.nsys-rep
```
