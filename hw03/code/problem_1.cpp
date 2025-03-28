#include "lu_factorization_and_solve.h"


int main() {
    std::size_t n = 2000;
    auto a = generate_random_matrix<double>(n, n, true);
    auto a_copy = a;

    auto p = std::vector<std::size_t>();
    lu_factorization_with_partial_pivoting_in_place(a_copy, p);

    auto pa = std::vector<std::vector<double>>(n);
    for (std::size_t i = 0; i < n; ++i) {
        pa[i] = a[p[i]];
    }

    auto u = upper_triangle(a_copy);
    auto l = lower_triangle(a_copy);
    for (std::size_t i = 0; i < n; ++i) {
        l[i][i] = 1;
    }

    std::cout << "n = " << n << std::endl;
    std::cout << "PA - LU Frobenius norm: " << matrix_norm<NormType::FROBENIUS>(pa - l * u) << std::endl;
}
