NVCC=nvc++
NVCC_FLAGS=-std=c++17 -stdpar -gpu cc80

.PHONY: all

all: transform

transform: axpy.cpp Makefile
	$(NVCC) $(CFLAGS) $(NVCC_FLAGS) $< $(LIBS) -o $@


.PHONY: clean

clean:
	rm -f transform

.PHONY: run

run: all
	$(JSC_SUBMIT_CMD) ./transform
