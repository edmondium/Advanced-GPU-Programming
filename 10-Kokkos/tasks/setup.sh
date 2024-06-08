module purge
module load GCC CUDA hwloc CMake git
export OMP_PROC_BIND=spread 
export OMP_PLACES=threads