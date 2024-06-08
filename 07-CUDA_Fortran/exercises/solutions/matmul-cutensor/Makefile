FC = nvfortran
GPUFLAGS = -gpu cc80 -fast -i8 -Minfo -cudalib=cutensor -L${NVHPC_ROOT}/math_libs/lib64
CPUFLAGS = -fast -i8 -Minfo

JSC_SUBMIT_CMD ?= srun --partition=gpus --gres=gpu:1 

all: mmul-GPU

mmul-GPU: mmul-GPU.cuf Makefile
	$(FC) $(GPUFLAGS) mmul-GPU.cuf -o $@

.PHONY: clean run profile
clean:
	rm -f mmul-GPU mmul*.o *.mod

run: mmul-GPU
	$(JSC_SUBMIT_CMD) ./mmul-GPU 8192


