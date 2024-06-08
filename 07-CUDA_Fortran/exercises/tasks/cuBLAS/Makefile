CC = nvcc
FC = nvfortran
FFLAGS = -cudalib=curand,cublas  -g -O3
.PHONY: all

all: dgemm
	
dgemm: dgemm.cuf 
	$(FC) -o $@ $<  $(FFLAGS)
	
.PHONY: clean

clean:
	rm -f *.o dgemm

.PHONY: run 

run: all 
	$(JSC_SUBMIT_CMD) ./dgemm 100
