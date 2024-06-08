#include <iostream>
#include <cuda/std/complex>
#include "mandelbrot.cuh"

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
void mandelbrot(double rmin, double rmax, double imin, double imax, Mandelbrot M, int maxtime=100) {
	auto i = blockIdx.x * blockDim.x + threadIdx.x;
	auto j = blockIdx.y * blockDim.y + threadIdx.y;
	auto dr = (rmax - rmin) / M.width;
	auto di = (imax - imin) / M.height;
	cuda::std::complex<double> c(rmin + i * dr, imin + j * di);
	if ((i < M.width) && (j < M.height)){
		M.data[j * M.width + i] = escape_time(c, maxtime);
	}
}


auto main() -> int {
	int width{64};
	int height{48};
	dim3 block(16, 16);
	dim3 grid(width % block.x == 0 ? width / block.x : width / block.x + 1, height % block.y == 0 ? height / block.y : height / block.y + 1);

	Mandelbrot myMandelbrot(height, width);

	mandelbrot<<<grid, block>>>(-2.0, 1.0, -1.0, 1.0, myMandelbrot);
	cudaError_t error = cudaGetLastError();
	cudaDeviceSynchronize();
	std::cerr << "Error: " << cudaGetErrorName(error) << '\n';
	for(int j = 0; j < myMandelbrot.height; ++j){
		for(int i = 0; i < myMandelbrot.width; ++i){
			std::cout << myMandelbrot.data[j * width + i] << ' ';
		}
		std::cout << "\n";
	}
}
