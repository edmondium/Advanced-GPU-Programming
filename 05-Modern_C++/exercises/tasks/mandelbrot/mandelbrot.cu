#include <iostream>
#include <cuda/std/complex>
#define CU_CHK(ERRORCODE) \ {cudaError_t error = ERRORCODE; \  if (error != 0) \  { std::cerr << cudaGetErrorName(error) << ": " << cudaGetErrorString(error) << \    " at " << __FILE__ << ":" << __LINE__ << "\n";}}

/** TODO: Use cuda::std::complex to simplify the escape_time function */
__host__ __device__
auto escape_time(double real, double imag, int maxiter) -> int {
	double z_real = 0;
	double z_imag = 0;
	for (int i = 0; i < maxiter; ++i){
		// z = z * z + c;
		auto tz_r = z_real * z_real - z_imag * z_imag + real;
		z_imag = 2 * z_real * z_imag + imag;
		z_real = tz_r;
		// if(abs(z) > 2) {
		if(sqrt(z_real * z_real + z_imag * z_imag) > 2.0) {
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
	/** TODO: Use cuda::std::complex instead of real and imag */
	auto real = (rmin + i * dr); 
	auto imag = imin + j * di;
	if ((i < width) && (j < height)){
		M[j * width + i] = escape_time(real, imag, maxtime);
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
