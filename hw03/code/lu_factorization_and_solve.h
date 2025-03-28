#ifndef LU_FACTORIZATION_AND_SOLVE_H
#define LU_FACTORIZATION_AND_SOLVE_H

#include "matrix_vector_operations.h"


// 列选主元的原地 LU 分解
// 输入的矩阵 a 会被修改，上三角部分为 U，严格下三角部分为 L 的严格下三角部分
template <typename T>
auto lu_factorization_with_partial_pivoting_in_place(std::vector<std::vector<T>> &a, std::vector<std::size_t> &p) {
    auto n = a.size();
    p.resize(n);

    for (std::size_t k = 0; k < n; ++k) {
        p[k] = k;
    }

    for (std::size_t k = 0; k < n - 1; ++k) {
        auto max_index = k;
        auto max_value = a[k][k];
        for (std::size_t i = k + 1; i < n; ++i) {
            if (std::abs(a[i][k]) > max_value) {
                max_index = i;
                max_value = std::abs(a[i][k]);
            }
        }

        std::swap(p[k], p[max_index]);
        std::swap(a[k], a[max_index]);

        for (std::size_t j = k + 1; j < n; ++j) {
            a[j][k] /= a[k][k];
            for (std::size_t l = k + 1; l < n; ++l) {
                a[j][l] -= a[j][k] * a[k][l];
            }
        }
    }
}


// 使用 LU 分解解线性方程组
// 输入的矩阵 a 会被修改
template <typename T>
auto solve_linear_system_in_place(std::vector<std::vector<T>> &a, const std::vector<T> &b) {
    auto n = a.size();
    auto p = std::vector<std::size_t>();

    lu_factorization_with_partial_pivoting_in_place(a, p);

    // 解 Ly = Pb
    auto y = std::vector<T>(n);
    for (std::size_t i = 0; i < n; ++i) {
        y[i] = b[p[i]];
        for (std::size_t j = 0; j < i; ++j) {
            y[i] -= a[i][j] * y[j];
        }
    }

    // 解 Ux = y
    auto x = std::vector<T>(n);
    for (std::size_t i = n - 1; i < n; --i) {
        x[i] = y[i];
        for (std::size_t j = i + 1; j < n; ++j) {
            x[i] -= a[i][j] * x[j];
        }
        x[i] /= a[i][i];
    }

    return x;
}

#endif  // LU_FACTORIZATION_AND_SOLVE_H
