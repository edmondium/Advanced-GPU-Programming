#include <iostream>
#include <cub/cub.cuh>

template <int BLOCK_THREADS, int ITEMS_PER_THREAD, BlockLoadAlgorithm LOAD_ALGO, 
        BlockReduceAlgorithm REDUCE_ALGO, typename T>
__global__ void ExampleKernel(const T* in, T* out){
    
    // Specialize the template for double precision and BLOCK_THREADS threads w/ ITEMS_PER_THREAD 
items per thread   
    typedef cub::BlockLoad<const T*, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_ALGO> BlockLoadT;
    // Specialize the template for double precision and BLOCK_THREADS threads    
    typedef cub::BlockReduce<T, BLOCK_THREADS> BlockReduceT;
    // Declare shared storage
    
    __shared__ union {
        typename BlockLoadT::TempStorage load;
        typename BlockReduceT::TempStorage reduce;
    } temp_storage;
    
    T items[ITEMS_PER_THREAD];
    
    BlockLoadT(temp_storage.load).Load(in, items);
    __syncthreads();
    
    // Instantiate an instance of BlockReduceT
    T result = BlockReduceT(temp_storage.reduce).Sum(items);
    
    if (threadIdx.x == 0){
        *out = result;
    }
}


int main(){
    
    double* d = new double[4096];
    double* d_gpu = NULL;
    double result = 0.0;
    double* result_gpu = NULL; 
    
    for (int i = 0; i < 4096; ++i) {
        d[i] = 1.0/4096;
    }
    
    // Allocate memory on the GPU
    cudaMalloc(&d_gpu, 4096 * sizeof(double));
    cudaMalloc(&result_gpu, sizeof(double));
    cudaMemcpy(d_gpu, d, 4096 * sizeof(double), cudaMemcpyHostToDevice);
    // TODO: Add timing routine
    // Call the kernel
    ExampleKernel<1024, 4, BLOCK_LOAD_TRANSPOSE, BLOCK_REDUCE_RAKING><<<1, 1024>>>(d_gpu, 
                                                                                   result_gpu);
    ExampleKernel<256, 16, BLOCK_LOAD_TRANSPOSE, BLOCK_REDUCE_RAKING><<<1, 256>>>(d_gpu, 
                                                                                   result_gpu);
    
    cudaMemcpy(&result, result_gpu, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "The result is " << result << std::endl;
    
}
