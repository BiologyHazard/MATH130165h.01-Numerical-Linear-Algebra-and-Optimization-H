#include "matrix_vector_operations.h"


template <typename T>
inline auto sign(T x) {
    return x >= 0 ? 1 : -1;
}


template <typename T>
auto to_hessenberg(const std::vector<std::vector<T>> &a) {
    auto m = a.size();
    auto v_s = std::vector<std::vector<T>>();
    auto r = a;
    for (std::size_t k = 0; k < m - 2; ++k) {
        auto x = flatten(get_slice(r, k + 1, m, k, k + 1));
        auto v = x;
        v[0] += sign(x[0]) * vector_norm(x, 2);
        v /= vector_norm(v, 2);
        v_s.emplace_back(v);
        set_slice(
            r, k + 1, m, k, m,
            get_slice(r, k + 1, m, k, m)
            - 2 * to_column_vector(v) * (to_row_vector(v) * get_slice(r, k + 1, m, k, m))
        );
        // take advantage of the Hermitian property
        set_slice(
            r, k, m, k + 1, m,
            get_slice(r, k, m, k + 1, m)
            - (get_slice(r, k, m, k + 1, m) * to_column_vector(v)) * (2 * to_row_vector(v))
        );
    }
    return std::make_pair(r, v_s);
}


template <typename T>
auto apply_q_on_x(const std::vector<std::vector<T>> &v_s, const std::vector<T> &x) {
    auto n = v_s.size() + 2;
    auto xx = x;
    for (std::size_t k = n - 2; k >= 1; --k) {
        auto &v = v_s[k - 1];
        auto x_k_n = get_slice(xx, k, n);
        set_slice(xx, k, n, x_k_n - 2 * v * inner_product(v, x_k_n));
    }
    return xx;
}


template <typename T>
auto construct_q(const std::vector<std::vector<T>> &v_s) {
    auto n = v_s.size() + 2;
    auto q = zeros<T>(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        auto e_i = e_i_vector<T>(n, i);
        auto q_i = apply_q_on_x(v_s, e_i);
        set_slice(q, 0, n, i, i + 1, to_column_vector(q_i));
    }
    return q;
}


int main() {
    const std::size_t n = 10;
    auto a = generate_random_matrix(n, n, true);
    a = (a + transpose(a)) / 2; // Make it Hermitian

    auto [hessenberg_matrix, v_s] = to_hessenberg(a);

    auto q = construct_q(v_s);
    auto QHQT = q * hessenberg_matrix * transpose(q);

    std::cout << "n = " << n << std::endl;
    std::cout << "A = " << std::endl << a << std::endl;
    std::cout << "Hessenberg matrix = " << std::endl << hessenberg_matrix << std::endl;
    std::cout << "Q = " << std::endl << q << std::endl;
    std::cout << "(A - QHQT) Frobenius norm: " << matrix_norm<NormType::FROBENIUS>(a - QHQT) << std::endl;

    return 0;
}
