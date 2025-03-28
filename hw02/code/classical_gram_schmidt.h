#ifndef CLASSICAL_GRAM_SCHMIDT_H
#define CLASSICAL_GRAM_SCHMIDT_H

#include <iostream>
#include "matrix_vector_operations.h"


// CGS 简单实现
// 也许列主序会更好，但是这里为了便于理解，使用行主序
template <typename T>
auto classical_gram_schmidt(const std::vector<std::vector<T>> &A) {
    auto m = A.size(), n = A[0].size();
    auto q = zeros<T>(m, n);
    auto r = zeros<T>(n, n);

    for (std::size_t j = 0; j < n; ++j) {
        auto v = get_column(A, j);
        for (std::size_t i = 0; i < j; ++i) {
            r[i][j] = inner_product(get_column(q, i), get_column(A, j));
            v -= r[i][j] * get_column(q, i);
        }
        r[j][j] = vector_norm(v, 2);
        set_column(q, j, v / r[j][j]);
    }

    return std::make_pair(q, r);
}

#endif // CLASSICAL_GRAM_SCHMIDT_H
