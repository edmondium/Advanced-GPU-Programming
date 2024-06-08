#include <iostream>
#include <cuda/std/complex>
#define CU_CHK(ERRORCODE) \ {cudaError_t error = ERRORCODE; \  if (error != 0) \  { std::cerr << cudaGetErrorName(error) << ": " << cudaGetErrorString(error) << \    " at " << __FILE__ << ":" << __LINE__ << "\n";}}

__host__ __device__
auto escape_time(cuda::std::complex<double> c, int maxiter) -> int {
	cuda::std::complex<double> z = 0;
	for (int i = 0; i < maxiter; ++i){
		z = z * z + c;
		if(abs(z) > 2) {
			return i; 
		}
	}
	return maxiter;
}

__global__
void mandelbrot(double rmin, double rmax, double imin, double imax, int width, int height, int* M, int maxtime=100) {
	auto i = blockIdx.x * blockDim.x + threadIdx.x;
	auto j = blockIdx.y * blockDim.y + threadIdx.y;
	auto dr = (rmax - rmin) / width;
	auto di = (imax - imin) / height;
	cuda::std::complex<double> c(rmin + i * dr, imin + j * di);
	if ((i < width) && (j < height)){
		M[j * width + i] = escape_time(c, maxtime);
	}
}


auto main() -> int {
	int width{64};
	int height{48};
	int* M = nullptr;
	cudaMallocManaged(&M, width * height * sizeof(int));
	dim3 block(16, 16);
	dim3 grid(width % block.x == 0 ? width / block.x : width / block.x + 1, height % block.y == 0 ? height / block.y : height / block.y + 1);

        mandelbrot<<<grid, block>>>(-2.0, 1.0, -1.0, 1.0, width, height, M);
	cudaError_t error = cudaGetLastError();
	cudaDeviceSynchronize();
	std::cerr << "Error: " << cudaGetErrorName(error) << '\n';
	for(int j = 0; j < height; ++j){
		for(int i = 0; i < width; ++i){
			std::cout << M[j * width + i] << ' ';
		}
		std::cout << "\n";
	}
	cudaFree(M);
}
