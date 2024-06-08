#include <algorithm>
#include <chrono>
#include <execution>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

auto main() -> int {
    int rows = 128;
    int cols = 256;
    int number_of_iterations = 1000000;
    double max_difference = 1.0e-6;
    double diff = 1000000 * max_difference;

    std::vector x(cols, 0);
    std::vector y(rows, 0);
    std::iota(x.begin(), x.end(), 0);
    std::iota(y.begin(), y.end(), 0);
    std::vector index(rows * cols, std::make_pair(0, 0) );
    std::for_each(y.begin(), y.end(), [=, &index](auto idy){
        std::for_each(x.begin(), x.end(), [=, &index](auto idx){
            index[idy * cols + idx] = std::make_pair(idy, idx);
        });
    });
    std::vector grid(rows * cols, 0.0);
    std::vector grid2(rows * cols, 0.0);
    std::vector dgrid(rows * cols, -1.0);
    std::generate(grid.begin(), grid.end(),
                  [n = 0] () mutable { return n++; });
    auto t0 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < number_of_iterations; ++i){
        auto ptr_grid = grid.data();
        // TODO: Add an execution policy
        std::transform(index.begin(), index.end(), grid2.begin(),
            [=](auto &id){
                auto res = 0.0;
                if (id.first > 0)
                    res += ptr_grid[(id.first - 1) * cols + id.second];
                else
                    res += ptr_grid[(rows - 1) * cols + id.second];
                if (id.first < (rows - 1))
                    res += ptr_grid[(id.first + 1) * cols + id.second];
                else
                    res += ptr_grid[id.second];
                if (id.second > 0)
                    res += ptr_grid[id.first * cols + id.second - 1];
                else
                    return 0.0;
                if (id.second < (cols - 1))
                    res += ptr_grid[id.first * cols + id.second + 1];
                else
                    return 1.0;

                return 0.25 * res;
        });
        // TODO: Calculate the maximum difference in the grid between the previous and the current
        //       iteration using std::transform_reduce.
        /* The first operations is the reduction, the second the transformation */
        diff = std::transform_reduce(std::execution::par, grid.begin(), grid.end(), grid2.begin(),
                                     );
        std::swap(grid, grid2);
        if (diff < max_difference){
            number_of_iterations = i + 1;
            break;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    for(auto j : y){
        for(auto i : x){
            std::cout << grid[j * cols + i] << " ";
        }
        std::cout << "\n";
    }
    std::chrono::duration<double> dt = t1 - t0;
    std::cout << "# Finished " << number_of_iterations << " iterations in " << dt.count() << " s.\n";
    std::cout << "# This corresponds to " << dt.count() / number_of_iterations * 1000
              << " ms per iteration.\n";
    std::cout << "# Maximum difference was " << diff << ".\n";

}
