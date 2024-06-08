FC = nvfortran
CMDL_PROFILER = nsys profile
NVFLAGS = -gpu=cc80 -Minfo

JSC_SUBMIT_CMD ?= srun --partition=booster --gres=gpu:1 

.PHONY: all

all: task_a task_b task_c
	

task_a: task_a.cuf Makefile
	$(FC) -o $@ $(NVFLAGS) $<
task_b: task_b.cuf Makefile
	$(FC) -o $@ $(NVFLAGS) $<
task_c: task_c.cuf Makefile
	$(FC) -o $@ $(NVFLAGS) $<
	
.PHONY: clean

clean:
	rm -f task_a task_b task_c *.nsys-rep

.PHONY: run_task_a run_task_b run_task_c profile_task_a profile_task_b profile_task_c

run_task_a: task_a
	$(JSC_SUBMIT_CMD) ./task_a
run_task_b:task_b
	$(JSC_SUBMIT_CMD) ./task_b
run_task_c:task_c
	$(JSC_SUBMIT_CMD) ./task_c

profile_task_a: task_a
	$(JSC_SUBMIT_CMD) $(CMDL_PROFILER) -o nsys_task_a ./task_a
profile_task_b:task_b
	$(JSC_SUBMIT_CMD) $(CMDL_PROFILER) -o nsys_task_b ./task_b
profile_task_c:task_c
	$(JSC_SUBMIT_CMD) $(CMDL_PROFILER) -o nsys_task_c ./task_c


