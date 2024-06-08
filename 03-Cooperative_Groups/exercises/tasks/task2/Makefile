NVCC=nvcc
NVCC_FLAGS= -lineinfo -g -gencode arch=compute_80,code=sm_80

.PHONY: all

all: cg-reduce

cg-reduce: cg-reduce.cu Makefile
ifndef CUDA_PATH
 $(error Please load CUDA before building)
endif
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

.PHONY: clean

clean:
	rm cg-reduce

.PHONY: run

run: all
	$(JSC_SUBMIT_CMD) ./cg-reduce
