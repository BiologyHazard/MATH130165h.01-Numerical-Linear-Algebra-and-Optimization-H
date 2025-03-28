#include "qr_factorization.h"
#include "cholesky_factorization.h"


auto decay_pattern(std::size_t k) {
    return 1 / std::pow(k + 1, 4);
}


// 通过对随机矩阵进行 QR 分解得到随机正交矩阵
auto generate_random_orthogonal_matrix(std::size_t n, bool fixed_seed = false, unsigned long seed = 0) {
    auto a = generate_random_matrix<double>(n, n, fixed_seed, seed);
    auto [flat_a, tau] = qr_using_dgeqrf(a);
    auto q = construct_q(n, n, flat_a, tau);
    return q;
}


auto generate_matrix(std::size_t n) {
    auto q = generate_random_orthogonal_matrix(n, true);
    auto eigenvalues = std::vector<double>();
    for (std::size_t i = 0; i < n; ++i) {
        eigenvalues.emplace_back(decay_pattern(i));
    }
    auto d = to_diag_matrix(eigenvalues);
    return q * d * transpose(q);
}


auto apply_cholesky_factorization_on_single_matrix(std::size_t n) {
    auto a = generate_matrix(n);
    auto a_copy = a;
    cholesky_factorization_in_place(a_copy);
    auto r = upper_triangle(a_copy);
    std::cout << "n = " << n << std::endl;
    // std::cout << "A = " << a << std::endl;
    // std::cout << "R = " << r << std::endl;
    std::cout << "Residual Forbenius norm: " << matrix_norm<NormType::FROBENIUS>(a - transpose(r) * r) << std::endl;
    std::cout << std::endl;
}


int main() {
    apply_cholesky_factorization_on_single_matrix(10);
    apply_cholesky_factorization_on_single_matrix(1000);
    return 0;
}
