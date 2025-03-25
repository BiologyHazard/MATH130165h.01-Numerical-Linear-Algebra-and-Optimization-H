#include <iostream>
#include "matrix_vector_operations.h"

template <typename T>
inline auto sign(T x) {
    return x == 0 ? 0 : x > 0 ? 1 : -1;
}

/*```
for k = 1 to n
    x = A[k:m, k]
    v[k] = sign(x[1]) * ||x||_2 * e_1 + x
    v[k] = v[k] / ||v[k]||_2
    A[k:m, k:n] = A[k:m, k:n] - 2 * v[k] * (v[k]^* * A[k:m, k:n])
```*/
template <typename T>
auto householder_qr(const std::vector<std::vector<T>> &a) {
    auto m = a.size(), n = a[0].size();
    auto v_r = std::vector<std::vector<T>>(m + 1, std::vector<T>(n, 0));
    set_slice(v_r, 0, m, 0, n, a);
    for (std::size_t k = 0; k < n; ++k) {
        auto x = flatten(get_slice(a, k, m, k, k + 1));
        auto v = x;
        v[0] += sign(x[0]) * vector_norm(x, 2);
        v /= vector_norm(v, 2);
        set_slice(v_r, k + 1, m + 1, k, k + 1, to_column_vector(v));
        set_slice(
            v_r, k, m, k, n,
            get_slice(v_r, k, m, k, n)
            - 2 * to_column_vector(v) * (to_row_vector(v) * get_slice(v_r, k, m, k, n))
        );
    }
    return v_r;
}

template <typename T>
auto apply_q_star_on_b(const std::vector<std::vector<T>> &v_r, const std::vector<T> &b) {
    auto m = v_r.size() - 1, n = v_r[0].size();
    auto bb = b;
    for (std::size_t k = 0; k < n; ++k) {
        auto v = flatten(get_slice(v_r, k + 1, m + 1, k, k + 1));
        auto b_k_m = get_slice(bb, k, m);
        set_slice(bb, k, m, b_k_m - 2 * v * inner_product(v, b_k_m));
    }
    return bb;
}

template <typename T>
auto apply_q_on_x(const std::vector<std::vector<T>> &v_r, const std::vector<T> &x) {
    auto m = v_r.size() - 1, n = v_r[0].size();
    auto xx = x;
    for (std::size_t k = n - 1; k < n; --k) {
        auto v = flatten(get_slice(v_r, k + 1, m + 1, k, k + 1));
        auto x_k_m = get_slice(xx, k, m);
        set_slice(xx, k, m, x_k_m - 2 * v * inner_product(v, x_k_m));
    }
    return xx;
}

template <typename T>
auto construct_r(const std::vector<std::vector<T>> &v_r) {
    auto n = v_r[0].size();
    auto r = std::vector<std::vector<T>>(n, std::vector<T>(n, 0));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i; j < n; ++j) {
            r[i][j] = v_r[i][j];
        }
    }
    return r;
}

template <typename T>
auto construct_q(const std::vector<std::vector<T>> &v_r) {
    auto m = v_r.size() - 1, n = v_r[0].size();
    auto q = std::vector<std::vector<T>>(m, std::vector<T>(n, 0));
    for (std::size_t i = 0; i < n; ++i) {
        auto e_i = std::vector<T>(m, 0);
        e_i[i] = 1;
        auto q_i = apply_q_on_x(v_r, e_i);
        set_slice(q, 0, m, i, i + 1, transpose(std::vector<std::vector<T>>{ q_i }));
    }
    return q;
}

// 测试示例
int main() {
    std::size_t m = 2000, n = 2000;
    auto A = generate_random_matrix<double>(m, n);
    auto [time, v_r] = timeit(householder_qr<double>, A);

    std::cout << "QR factorization succeeded in " << time << " seconds" << std::endl;

    return 0;
}
