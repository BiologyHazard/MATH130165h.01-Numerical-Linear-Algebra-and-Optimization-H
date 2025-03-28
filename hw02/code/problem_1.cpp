#include "classical_gram_schmidt.h"

int main() {
    std::size_t m = 2000, n = 2000;
    auto A = generate_random_matrix(m, n, true);
    auto [time, Q_R] = timeit(classical_gram_schmidt<double>, A);

    std::cout << "Classical Gram Schmidt succeeded in " << time << " seconds" << std::endl;

    auto [Q, R] = Q_R;
    auto QR = Q * R;

    auto QTQ_I_frobenuis_norm = matrix_norm(transpose(Q) * Q - identity_matrix<double>(m));
    std::cout << "Q^T*Q - I Frobenius norm: " << QTQ_I_frobenuis_norm << std::endl;

    auto residual_frobenius_norm = matrix_norm(A - QR);
    std::cout << "Residual Frobenius norm: " << residual_frobenius_norm << std::endl;

    return 0;
}
