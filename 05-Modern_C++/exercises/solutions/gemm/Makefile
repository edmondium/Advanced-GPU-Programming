NVCC=nvcc
NVCC_FLAGS=-std=c++17 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80
NVCC_FLAGS+=-lineinfo

.PHONY: all

all: gemm

gemm: gemm.cu Makefile
	$(NVCC) $(CFLAGS) $(NVCC_FLAGS) $< $(LIBS) -o $@


.PHONY: clean

clean:
	rm -f gemm

.PHONY: run 

run: all
	$(JSC_SUBMIT_CMD) ./gemm
