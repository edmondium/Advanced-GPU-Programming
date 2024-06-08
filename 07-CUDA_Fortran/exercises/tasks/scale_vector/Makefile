FC = nvfortran

NVFLAGS = -Mcuda=cc80 -Minfo

JSC_SUBMIT_CMD ?= srun --partition=booster --gres=gpu:1 

.PHONY: all

all: scale_vector_um scale_vector
	
scale_vector_um: scale_vector_um.cuf Makefile
	$(FC) -o $@ $(NVFLAGS) $<	

scale_vector: scale_vector.cuf Makefile
	$(FC) -o $@ $(NVFLAGS) $<
	
.PHONY: clean

clean:
	rm -f scale_vector scale_vector_um

.PHONY: run run-um 

run: scale_vector
	$(JSC_SUBMIT_CMD) ./scale_vector
run-um:scale_vector_um
	$(JSC_SUBMIT_CMD) ./scale_vector_um
