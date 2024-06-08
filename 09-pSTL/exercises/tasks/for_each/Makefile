NVCC=nvc++
NVCC_FLAGS=-std=c++17 -stdpar -gpu cc80

.PHONY: all

all: for_each

for_each: for_each.cpp Makefile
	$(NVCC) $(CFLAGS) $(NVCC_FLAGS) $< $(LIBS) -o $@


.PHONY: clean

clean:
	rm -f for_each

.PHONY: run

run: all
	$(JSC_SUBMIT_CMD) ./for_each
