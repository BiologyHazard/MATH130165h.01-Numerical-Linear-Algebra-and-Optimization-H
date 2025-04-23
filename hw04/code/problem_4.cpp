#include "matrix_vector_operations.h"


template <typename T>
inline auto sign(T x) {
    return x >= 0 ? 1 : -1;
}


template <typename T>
auto to_bidiagonal(const std::vector<std::vector<T>> &a) {
    auto m = a.size();
    auto v_s = std::vector<std::vector<T>>();
    auto w_s = std::vector<std::vector<T>>();
    auto r = a;
    for (std::size_t k = 0; k < m; ++k) {
        auto x = flatten(get_slice(r, k, m, k, k + 1));
        auto v = x;
        v[0] += sign(x[0]) * vector_norm(x, 2);
        v /= vector_norm(v, 2);
        v_s.emplace_back(v);
        auto r_slice = get_slice(r, k, m, k, m);
        set_slice(
            r, k, m, k, m,
            r_slice - 2 * to_column_vector(v) * (to_row_vector(v) * r_slice)
        );

        if (k == m - 1) {
            break;
        }

        x = flatten(get_slice(r, k, k + 1, k + 1, m));
        auto w = x;
        w[0] += sign(x[0]) * vector_norm(x, 2);
        w /= vector_norm(w, 2);
        w_s.emplace_back(w);
        r_slice = get_slice(r, k, m, k + 1, m);
        set_slice(
            r, k, m, k + 1, m,
            r_slice - (r_slice * to_column_vector(w)) * (2 * to_row_vector(w))
        );
    }
    return std::make_tuple(r, v_s, w_s);
}


template <typename T>
auto apply_q_on_x(const std::vector<std::vector<T>> &v_s, const std::vector<T> &x) {
    auto n = v_s.size();
    auto xx = x;
    for (std::size_t k = n - 1; k < n; --k) {
        auto &v = v_s[k];
        auto x_k_n = get_slice(xx, k, n);
        set_slice(xx, k, n, x_k_n - 2 * v * inner_product(v, x_k_n));
    }
    return xx;
}


template <typename T>
auto apply_p_on_x(const std::vector<std::vector<T>> &w_s, const std::vector<T> &x) {
    auto n = w_s.size() + 1;
    auto xx = x;
    for (std::size_t k = n - 1; k >= 1; --k) {
        auto &v = w_s[k - 1];
        auto x_k_n = get_slice(xx, k, n);
        set_slice(xx, k, n, x_k_n - 2 * v * inner_product(v, x_k_n));
    }
    return xx;
}


template <typename T>
auto construct_q(const std::vector<std::vector<T>> &v_s) {
    auto n = v_s.size();
    auto q = zeros<T>(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        auto e_i = e_i_vector<T>(n, i);
        auto q_i = apply_q_on_x(v_s, e_i);
        set_slice(q, 0, n, i, i + 1, to_column_vector(q_i));
    }
    return q;
}


template <typename T>
auto construct_p(const std::vector<std::vector<T>> &w_s) {
    auto n = w_s.size() + 1;
    auto p = zeros<T>(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        auto e_i = e_i_vector<T>(n, i);
        auto p_i = apply_p_on_x(w_s, e_i);
        set_slice(p, 0, n, i, i + 1, to_column_vector(p_i));
    }
    return p;
}


int main() {
    const std::size_t n = 10;
    auto a = generate_random_matrix(n, n, true);

    auto [bidiagonal_matrix, v_s, w_s] = to_bidiagonal(a);

    std::cout << "A = " << std::endl << a << std::endl;
    std::cout << "Bi-diagonal matrix = " << std::endl << bidiagonal_matrix << std::endl;
    // std::cout << "v_s = " << std::endl << v_s << std::endl;
    // std::cout << "w_s = " << std::endl << w_s << std::endl;

    auto q = construct_q(v_s);
    auto p = construct_p(w_s);
    std::cout << "Q = " << std::endl << q << std::endl;
    std::cout << "P = " << std::endl << p << std::endl;

    auto QBPT = q * bidiagonal_matrix * transpose(p);

    std::cout << "n = " << n << std::endl;
    std::cout << "(A - QBPT) Frobenius norm: " << matrix_norm<NormType::FROBENIUS>(a - QBPT) << std::endl;

    return 0;
}
