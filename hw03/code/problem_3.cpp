#include <iomanip>
#include "lu_factorization_and_solve.h"


template <typename T>
auto generate_matrix_a(std::size_t n) {
    auto a = zeros<T>(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        a[i][i] = 1;
        a[i][n - 1] = 1;
        for (std::size_t j = 0; j < i; ++j) {
            a[i][j] = -1;
        }
    }
    return a;
}


void apply_lu_factorization_on_single_matrix(std::size_t n) {
    auto a = generate_matrix_a<double>(n);
    auto p = std::vector<std::size_t>();
    lu_factorization_with_partial_pivoting_in_place(a, p);

    enable_vector_output_max_size = false;
    std::cout << "n = " << n << std::endl;
    std::cout << "L, U = " << std::endl << a << std::endl;
    std::cout << "P = " << p << std::endl;
    std::cout << std::endl;
}


int main() {
    apply_lu_factorization_on_single_matrix(8);
    apply_lu_factorization_on_single_matrix(100);
    return 0;
}
