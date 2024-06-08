#include <algorithm>
#include <chrono>
#include <execution>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <thrust/iterator/counting_iterator.h>

template <class T>
auto saxpy(std::vector<T> &x, std::vector<T> &y, T a){

    std::vector idx(x.size(), 0);
    std::iota(idx.begin(), idx.end(), 0);
    /* NOTE: This is a workaround. We use the pointer, which can be captured by value
     *       instead of the vectors directly. This should become unnecessary in future
     *       releases.
     */
    auto ptr_x = x.data();
    auto ptr_y = y.data();
    // TODO: Add an execution policy as first argument. Fill the body of the function
    //       using ptr_x and ptr_y to implement y = a*x + y.
    std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [=](auto i){
        ptr_y[i] += a * ptr_x[i];
    });

}

template <class T>
auto saxpy2(std::vector<T> &x, std::vector<T> &y, T a){
    auto r = thrust::counting_iterator<int>(0);
    /* NOTE: This is a workaround. We use the pointer, which can be captured by value
     *       instead of the vectors directly. This should become unnecessary in future
     *       releases.
     */
    auto ptr_x = x.data();
    auto ptr_y = y.data();
    std::for_each(std::execution::par_unseq, r, r + x.size(), [=](auto i){
        ptr_y[i] += a * ptr_x[i];
    });
}


auto main(int argc, char** argv) -> int {
    size_t N = (argc > 1) ? std::stoi(argv[1]) : 1000;
    std::vector x(N, 0.0f);
    std::vector y(N, 0.0f);
    for(int i = 0; i < N; ++i){
        x[i] = i;
        y[i] = 1;
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    saxpy(x, y, 3.1415f);
    auto end_time = std::chrono::high_resolution_clock::now();
    int errorCount = 0;
    for(int i = 0; i < N; ++i){
        if (std::abs(y[i] - (3.1415 * i + 1)) > (i * 1.0e-5)){
            std::cout << "There's an error in element " << i << "! ";
            std::cout << "y[" << i << "] = " << y[i] << " not " << (3.1415 * i + 1) << ".\n";
            errorCount++;
        }
    }
    if (errorCount == 0){
        std::cout << "saxpy with " << N << " elements took "
        << std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() *
            1e-6 << " s" << std::endl;
    }
}
