#ifndef QR_FACTORIZATION_H
#define QR_FACTORIZATION_H

#include "matrix_vector_operations.h"


extern "C" {
    void dgeqrf_(const int *m, const int *n, double *a, const int *lda, double *tau, double *work, const int *lwork, int *info);
    void dorgqr_(const int *m, const int *n, const int *k, double *a, const int *lda, const double *tau, double *work, const int *lwork, int *info);
}


auto qr_using_dgeqrf(const std::vector<std::vector<double>> &a) {
    auto m = static_cast<int>(a.size()), n = static_cast<int>(a[0].size());

    auto tau = std::vector<double>(std::min(m, n));
    auto flat_a = flatten(transpose(a));
    int lwork = -1;
    auto work = std::vector<double>(1);
    int info = 0;
    dgeqrf_(&m, &n, flat_a.data(), &m, tau.data(), work.data(), &lwork, &info);
    lwork = static_cast<int>(work[0]);
    work.resize(lwork);
    dgeqrf_(&m, &n, flat_a.data(), &m, tau.data(), work.data(), &lwork, &info);

    if (info != 0) {
        throw std::runtime_error("QR factorization failed with error code " + std::to_string(info));
    }

    return std::make_pair(flat_a, tau);
}


auto construct_q(int m, int n, const std::vector<double> &flat_a, const std::vector<double> &tau) {
    auto k = std::min(m, n);
    auto flat_q = flat_a;
    auto tau_copy = tau;
    int lwork = -1;
    auto work = std::vector<double>(1);
    int info = 0;
    dorgqr_(&m, &k, &k, flat_q.data(), &m, tau_copy.data(), work.data(), &lwork, &info);
    lwork = static_cast<int>(work[0]);
    work.resize(lwork);
    dorgqr_(&m, &k, &k, flat_q.data(), &m, tau_copy.data(), work.data(), &lwork, &info);

    if (info != 0) {
        throw std::runtime_error("Q matrix construction failed with error code " + std::to_string(info));
    }

    auto q = transpose(unflatten(flat_q, k, m));

    return q;
}


auto construct_r(int m, int n, const std::vector<double> &flat_a) {
    auto k = std::min(m, n);
    auto r = zeros<double>(k, n);
    for (int i = 0; i < k; ++i) {
        for (int j = i; j < n; ++j) {
            r[i][j] = flat_a[i + j * m];
        }
    }
    return r;
}

#endif  // QR_FACTORIZATION_H
