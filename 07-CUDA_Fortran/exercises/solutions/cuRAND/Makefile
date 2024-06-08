FC = nvfortran
CC = nvcc
FFLAGS  = -Minfo=accel -cudalib=curand

.PHONY: all

all:  MCint

MCint:MCint.cuf Makefile
	$(FC) $(FFLAGS) -o MCint MCint.cuf

.PHONY: clean

clean:
	rm -f MCint

.PHONY: run

run: MCint
	$(JSC_SUBMIT_CMD) ./MCint
	