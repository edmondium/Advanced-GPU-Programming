## Module 1 Advanced Multi-GPU Programming with MPI

### Task 3 UCC

Run the `osu_allreduce` benchmarks from the OSU Microbenchmarks (OMB) with UCC accelerated collectives:
* Set `OMPI_MCA_coll_ucc_enable=0` to disable UCC collective
* Set `UCC_TL_CUDA_TUNE=allreduce:cuda:inf` prioritize CUDA Team Layer (TL) for allreduce on CUDA device memory buffers (use `0` instead of `inf` to disable)
* Set `UCC_TL_NCCL_TUNE=allreduce:cuda:inf` prioritize NCCL Team Layer (TL) for allreduce on CUDA device memory buffers (use `0` instead of `inf` to disable)
* Set `UCC_TL_UCP_TUNE=allreduce:cuda:inf` prioritize UCP Team Layer (TL) for allreduce on CUDA device memory buffers (use `0` instead of `inf` to disable)

See [`user_guide.md`](https://github.com/openucx/ucc/blob/master/docs/user_guide.md) for more information.

#### Building OMB

The OSU Micro-Benchmarks (OMB) 7.0.1 are installed in `$PROJECT_training2418/OMB/osu-micro-benchmarks-7.0.1-install/` and have been build with

```console
    curl --proto '=https' -fSsL https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.0.1.tar.gz | tar xz
    cd osu-micro-benchmarks-7.0.1/
    ./configure CC=mpicc CXX=mpicxx --enable-cuda --with-cuda-include=$CUDA_HOME/include --with-cuda-libpath=$CUDA_HOME/lib64 --prefix=${PWD}/../osu-micro-benchmarks-7.0.1-install/
    make 
    make install
```

#### Example Output

Running `osu_allreduce` without UCC and a 16 MiB message

```console
[kraus1@jwlogin22 03-Collectives]$ OMPI_MCA_coll_ucc_enable=0 $JSC_SUBMIT_CMD -n 4 osu_allreduce -M 16777216 -d cuda --full -r gpu -x 200 -i 100 -m 16777216:16777216
srun: job 9906524 queued and waiting for resources
srun: job 9906524 has been allocated resources

# OSU MPI-CUDA Allreduce Latency Test v7.0
# Size       Avg Latency(us)   Min Latency(us)   Max Latency(us)  Iterations
16777216             8630.58           8619.50           8642.46         10
```

Running `osu_allreduce` with a 64x larger 1 GiB message and UCC, forcing CUDA TL still runs with lower latency:

```console
[kraus1@jwlogin22 03-Collectives]$ UCC_TL_CUDA_TUNE=allreduce:cuda:inf $JSC_SUBMIT_CMD -n 4 osu_allreduce -M 1073741824 -d cuda --full -r gpu -x 200 -i 100 -m 1073741824:1073741824
srun: job 9906533 queued and waiting for resources
srun: job 9906533 has been allocated resources

# OSU MPI-CUDA Allreduce Latency Test v7.0
# Size       Avg Latency(us)   Min Latency(us)   Max Latency(us)  Iterations
1073741824           7327.25           7324.88           7330.34         100
```
