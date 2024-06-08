#include <algorithm>
#include <iostream>
#include <numeric>
#include <numbers>


template <class T>
// TODO: Replace the pointers with cuda::std::span. Don't forget to include the header.
__global__ void axpy(T alpha, T* x, T* y, int n)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: Use the size of the span to check if the index is valid.
    if (i < n)
    {
        y[i] = alpha * x[i] + y[i];
    }
}

auto main() -> int {
    auto N{1000000ul};
    auto alpha{std::numbers::pi};
    double* x{nullptr};
    double* y{nullptr};
    cudaMallocManaged(&x, N * sizeof(double));
    cudaMallocManaged(&y, N * sizeof(double));
    std::fill(x, x + N, 0.5 / (N * alpha);
    std::fill(y, y + N, 0.5 / N);
    auto block_size{256};
    auto grid_size{(N + block_size - 1) / block_size};
    // TODO: A span needs to know its size. You can pass two arguments to the constructor using {,},
    //   e.g., {x, N}.
    axpy<<<grid_size, block_size>>>(alpha, x, y, N);
    cudaDeviceSynchronize();
    std::cout << "The sum over y is " << std::reduce(y, y + N, 0.0) << "\n.";
    cudaFree(x);
    cudaFree(y);
}