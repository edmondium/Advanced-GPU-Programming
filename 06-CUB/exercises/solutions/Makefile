cub: cub_sort.cu
	nvcc -std=c++17 -arch sm_80 -o cub_sort -lcurand cub_sort.cu

.PHONY: clean

clean:
	rm -f gemm

.PHONY: run

run: all
	$(JSC_SUBMIT_CMD) ./gemm

