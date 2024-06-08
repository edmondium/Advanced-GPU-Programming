#define SOLUTION 1
#include <algorithm>
//TODO: Add header execution
#include <execution>
#include <iostream>
#include <numeric>
#include <vector>

/** Calculate y = ax + y for each element of x and y
 * @param a scale factor for x
 * @param x vector
 * @param y target vector
 */
template <class T>
auto axpy(T&& x, T&& y, double a){
    // TODO: Add an execution policy
    std::transform(std::execution::par, x.begin(), x.end(), y.begin(), y.begin(),
                   [=](auto x, auto y){
        return a * x + y;
    });
}

auto main() -> int{
    size_t N = 10000;
    std::vector x(N, 1.0 / N);
    std::vector y(N, 0.0);
    double a = 2;
    axpy(x, y, a);
    std::cout << "The sum of the scaled elements of x is " << std::reduce(std::execution::par_unseq, y.begin(), y.end()) << ".\n";
}

