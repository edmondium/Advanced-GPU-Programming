NVCC=nvcc
NVCC_FLAGS=-std=c++20 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80
NVCC_FLAGS+=-lineinfo

.PHONY: all

all: axpy

axpy: axpy.cu Makefile
	$(NVCC) $(CFLAGS) $(NVCC_FLAGS) $< $(LIBS) -o $@


.PHONY: clean

clean:
	rm -f axpy

.PHONY: run 

run: all
	$(JSC_SUBMIT_CMD) ./axpy
