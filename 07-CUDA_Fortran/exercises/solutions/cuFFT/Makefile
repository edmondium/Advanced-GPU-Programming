CC = nvcc
FC = nvfortran
FFLAGS = -cudalib=cufft -g -O3
.PHONY: all

all: fft
	
fft: fft.cuf 
	$(FC) -o $@ $<  $(FFLAGS)
	
.PHONY: clean

clean:
	rm -f *.o fft *.dat


run: all 
	$(JSC_SUBMIT_CMD) ./fft
