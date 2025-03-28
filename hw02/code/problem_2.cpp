#include "householder_qr.h"

int main() {
    std::size_t m = 2000, n = 2000;
    auto A = generate_random_matrix<double>(m, n, true);
    auto [time, R_v_s] = timeit(householder_qr<double>, A);

    std::cout << "Householder QR factorization succeeded in " << time << " seconds" << std::endl;

    auto [R, v_s] = R_v_s;
    auto Q = construct_q(v_s);
    auto QR = Q * R;

    auto QTQ_I_frobenuis_norm = matrix_norm(transpose(Q) * Q - identity_matrix<double>(m));
    std::cout << "Q^T*Q - I Frobenius norm: " << QTQ_I_frobenuis_norm << std::endl;

    auto residual_frobenius_norm = matrix_norm(A - QR);
    std::cout << "Residual Frobenius norm: " << residual_frobenius_norm << std::endl;

    return 0;
}
