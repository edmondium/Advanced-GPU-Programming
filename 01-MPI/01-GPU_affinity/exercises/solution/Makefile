# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
JSC_SUBMIT_CMD ?= srun
NP ?= 4
NVCC=nvcc
MPICXX=mpicxx
GENCODE_SM30	:= -gencode arch=compute_30,code=sm_30
GENCODE_SM35	:= -gencode arch=compute_35,code=sm_35
GENCODE_SM37	:= -gencode arch=compute_37,code=sm_37
GENCODE_SM50	:= -gencode arch=compute_50,code=sm_50
GENCODE_SM52	:= -gencode arch=compute_52,code=sm_52
GENCODE_SM60	:= -gencode arch=compute_60,code=sm_60
GENCODE_SM70	:= -gencode arch=compute_70,code=sm_70
GENCODE_SM80	:= -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
GENCODE_FLAGS	:= $(GENCODE_SM70) $(GENCODE_SM80)
NVCC_FLAGS = -lineinfo $(GENCODE_FLAGS) -Xptxas --optimize-float-atomics -std=c++11
MPICXX_FLAGS = -DUSE_NVTX -I$(CUDA_HOME)/include -std=c++11
LD_FLAGS = -L$(CUDA_HOME)/lib64 -lcudart -lnvToolsExt
jacobi: Makefile jacobi.cpp jacobi_kernels.o
	$(MPICXX) $(MPICXX_FLAGS) jacobi.cpp jacobi_kernels.o $(LD_FLAGS) -o jacobi

jacobi_kernels.o: Makefile jacobi_kernels.cu
	$(NVCC) $(NVCC_FLAGS) jacobi_kernels.cu -c

.PHONY.: clean
clean:
	rm -f jacobi jacobi_kernels.o *.compute-sanitizer.log *.nsys-rep

sanitize: jacobi
	$(JSC_SUBMIT_CMD) -n $(NP) compute-sanitizer --log-file jacobi.%q{PMIX_RANK}.compute-sanitizer.log ./jacobi -niter 10

run: jacobi
	$(JSC_SUBMIT_CMD) -n $(NP) ./jacobi

profile: jacobi
	$(JSC_SUBMIT_CMD) -n $(NP) nsys profile --trace=mpi,cuda,nvtx -o jacobi.%q{PMIX_RANK} ./jacobi -niter 10
