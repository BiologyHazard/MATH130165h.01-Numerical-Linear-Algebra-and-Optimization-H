#ifndef CHOLESKY_FACTORIZATION_H
#define CHOLESKY_FACTORIZATION_H

#include "matrix_vector_operations.h"


template <typename T>
auto cholesky_factorization_in_place(std::vector<std::vector<T>> &a) {
    auto m = a.size();
    for (std::size_t k = 0; k < m; ++k) {
        for (std::size_t j = k + 1; j < m; ++j) {
            for (std::size_t l = j; l < m; ++l) {
                a[j][l] -= a[k][l] * a[k][j] / a[k][k];
            }
        }
        auto sqrt_a_k_k = std::sqrt(a[k][k]);
        for (std::size_t j = k; j < m; ++j) {
            a[k][j] /= sqrt_a_k_k;
        }
    }
}

#endif  // CHOLESKY_FACTORIZATION_H
