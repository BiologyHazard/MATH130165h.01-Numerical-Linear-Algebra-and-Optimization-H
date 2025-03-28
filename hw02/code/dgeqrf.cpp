#include "matrix_vector_operations.h"

extern "C" {
    void dgeqrf_(const int *m, const int *n, double *a, const int *lda, double *tau, double *work, const int *lwork, int *info);
    void dorgqr_(const int *m, const int *n, const int *k, double *a, const int *lda, const double *tau, double *work, const int *lwork, int *info);
}

auto qr_using_dgeqrf(const std::vector<std::vector<double>> &A) {
    auto m = static_cast<int>(A.size()), n = static_cast<int>(A[0].size());

    auto tau = std::vector<double>(std::min(m, n));
    auto flat_A = flatten(transpose(A));
    int lwork = -1;
    auto work = std::vector<double>(1);
    int info = 0;
    dgeqrf_(&m, &n, flat_A.data(), &m, tau.data(), work.data(), &lwork, &info);
    lwork = static_cast<int>(work[0]);
    work.resize(lwork);
    dgeqrf_(&m, &n, flat_A.data(), &m, tau.data(), work.data(), &lwork, &info);

    if (info != 0) {
        throw std::runtime_error("QR factorization failed with error code " + std::to_string(info));
    }

    return std::make_pair(flat_A, tau);
}

auto construct_q(int m, int n, const std::vector<double> &flat_A, const std::vector<double> &tau) {
    auto k = std::min(m, n);
    auto flat_Q = flat_A;
    auto tau_copy = tau;
    int lwork = -1;
    auto work = std::vector<double>(1);
    int info = 0;
    dorgqr_(&m, &k, &k, flat_Q.data(), &m, tau_copy.data(), work.data(), &lwork, &info);
    lwork = static_cast<int>(work[0]);
    work.resize(lwork);
    dorgqr_(&m, &k, &k, flat_Q.data(), &m, tau_copy.data(), work.data(), &lwork, &info);

    if (info != 0) {
        throw std::runtime_error("Q matrix construction failed with error code " + std::to_string(info));
    }

    auto Q = transpose(unflatten(flat_Q, k, m));

    return Q;
}

auto construct_r(int m, int n, const std::vector<double> &flat_A) {
    auto k = std::min(m, n);
    auto R = zeros<double>(k, n);
    for (int i = 0; i < k; ++i) {
        for (int j = i; j < n; ++j) {
            R[i][j] = flat_A[i + j * m];
        }
    }
    return R;
}

int main() {
    const int m = 2000, n = 2000;
    auto A = generate_random_matrix<double>(m, n, true);
    auto [time, flat_A_tau] = timeit(qr_using_dgeqrf, A);

    std::cout << "QR factorization succeeded in " << time << " seconds" << std::endl;

    auto [flat_A, tau] = flat_A_tau;
    auto Q = construct_q(m, n, flat_A, tau);
    auto R = construct_r(m, n, flat_A);
    auto QR = Q * R;

    auto QTQ_I_frobenuis_norm = matrix_norm(transpose(Q) * Q - identity_matrix<double>(m));
    std::cout << "Q^T*Q - I Frobenius norm: " << QTQ_I_frobenuis_norm << std::endl;

    auto residual_frobenius_norm = matrix_norm(A - QR);
    std::cout << "Residual Frobenius norm: " << residual_frobenius_norm << std::endl;

    return 0;
}
