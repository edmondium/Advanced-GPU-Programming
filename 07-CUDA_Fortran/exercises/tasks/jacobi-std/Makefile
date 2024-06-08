# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.

FC = nvfortran
NVFLAGS = -Mcuda=cc80 -fast -Minfo -stdpar


JSC_SUBMIT_CMD ?= srun --partition=booster --gres=gpu:1 

all: poisson2d

poisson2d: poisson2d.f90 poisson2d_reference.f90 Makefile
	$(FC) $(NVFLAGS) poisson2d.f90 poisson2d_reference.f90 -o $@

.PHONY: clean run profile
clean:
	rm -f poisson2d poisson2d*.o *.mod

run: poisson2d
	$(JSC_SUBMIT_CMD) ./poisson2d



