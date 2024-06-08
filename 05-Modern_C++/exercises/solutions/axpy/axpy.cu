#include <algorithm>
#include <iostream>
#include <numeric>
#include <numbers>

#include <cuda/std/span>

template <class T>
__global__ void axpy(T alpha, cuda::std::span<T> x, cuda::std::span<T> y)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < x.size())
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
    axpy<<<grid_size, block_size>>>(alpha, {x, N}, {y, N});
    cudaDeviceSynchronize();
    std::cout << "The sum over y is " << std::reduce(y, y + N, 0.0) << "\n.";
    cudaFree(x);
    cudaFree(y);
}