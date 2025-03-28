#ifndef HOUSEHOLDER_QR_H
#define HOUSEHOLDER_QR_H

#include "matrix_vector_operations.h"

template <typename T>
inline auto sign(T x) {
    return x >= 0 ? 1 : -1;
}

/*```
for k = 1 to n
    x = A[k:m, k]
    v[k] = sign(x[1]) * ||x||_2 * e_1 + x
    v[k] = v[k] / ||v[k]||_2
    A[k:m, k:n] = A[k:m, k:n] - 2 * v[k] * (v[k]^* * A[k:m, k:n])
```*/
template <typename T>
auto householder_qr(const std::vector<std::vector<T>> &A) {
    auto m = A.size(), n = A[0].size();
    auto v_s = std::vector<std::vector<T>>();
    auto R = A;

    for (std::size_t k = 0; k < n; ++k) {
        auto x = flatten(get_slice(R, k, m, k, k + 1));
        auto v = x;
        v[0] += sign(x[0]) * vector_norm(x, 2);
        v /= vector_norm(v, 2);
        v_s.emplace_back(v);
        set_slice(
            R, k, m, k, n,
            get_slice(R, k, m, k, n)
            - 2 * to_column_vector(v) * (to_row_vector(v) * get_slice(R, k, m, k, n))
        );
    }
    return std::make_pair(R, v_s);
}

template <typename T>
auto apply_q_star_on_b(const std::vector<std::vector<T>> &v_s, const std::vector<T> &b) {
    auto m = v_s[0].size(), n = v_s.size();
    auto bb = b;
    for (std::size_t k = 0; k < n; ++k) {
        auto &v = v_s[k];
        auto b_k_m = get_slice(bb, k, m);
        set_slice(bb, k, m, b_k_m - 2 * v * inner_product(v, b_k_m));
    }
    return bb;
}

template <typename T>
auto apply_q_on_x(const std::vector<std::vector<T>> &v_s, const std::vector<T> &x) {
    auto m = v_s[0].size(), n = v_s.size();
    auto xx = x;
    for (std::size_t k = n - 1; k < n; --k) {
        auto &v = v_s[k];
        auto x_k_m = get_slice(xx, k, m);
        set_slice(xx, k, m, x_k_m - 2 * v * inner_product(v, x_k_m));
    }
    return xx;
}

template <typename T>
auto construct_q(const std::vector<std::vector<T>> &v_s) {
    auto m = v_s[0].size(), n = v_s.size();
    auto q = zeros<T>(m, n);
    for (std::size_t i = 0; i < n; ++i) {
        auto e_i = std::vector<T>(m, 0);
        e_i[i] = 1;
        auto q_i = apply_q_on_x(v_s, e_i);
        set_slice(q, 0, m, i, i + 1, to_column_vector(q_i));
    }
    return q;
}

#endif // HOUSEHOLDER_QR_H
