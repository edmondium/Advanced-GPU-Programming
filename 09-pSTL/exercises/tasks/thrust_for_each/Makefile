NVCC=nvcc
NVCC_FLAGS=-std=c++17 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80
NVCC_FLAGS+=-lineinfo 
NVCC_FLAGS+=--expt-extended-lambda 

.PHONY: run

all: run

for_each: for_each.cu Makefile
	$(NVCC) $(CFLAGS) $(NVCC_FLAGS) $< $(LIBS) -o $@


.PHONY: clean

clean:
	rm -f for_each

.PHONY: run profile memcheck

run: for_each
	$(JSC_SUBMIT_CMD) ./for_each

profile: for_each
	$(JSC_SUBMIT_CMD) nvprof --unified-memory-profiling per-process-device ./for_each

memcheck: for_each
	$(JSC_SUBMIT_CMD) cuda-memcheck ./for_each
	

