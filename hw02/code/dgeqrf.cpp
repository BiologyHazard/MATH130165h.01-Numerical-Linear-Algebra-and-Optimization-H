#include <iostream>
#include <vector>
#include <random>
#include "matrix_vector_operations.h"

extern "C" {
    void dgeqrf_(const int *m, const int *n, double *a, const int *lda, double *tau, double *work, const int *lwork, int *info);
}

int qr_using_dgeqrf(const std::vector<std::vector<double>> &A) {
    auto m = static_cast<int>(A.size()), n = static_cast<int>(A[0].size());
    auto tau = std::vector<double>(std::min(m, n));
    auto flat_A = flatten(transpose(A));
    int lda = m;
    int lwork = -1;
    double wkopt;
    auto info = 0;
    dgeqrf_(&m, &n, flat_A.data(), &lda, tau.data(), &wkopt, &lwork, &info);
    lwork = static_cast<int>(wkopt);
    auto work = std::vector<double>(lwork);
    dgeqrf_(&m, &n, flat_A.data(), &lda, tau.data(), work.data(), &lwork, &info);
    return info;
}

int main() {
    const int m = 2000, n = 2000;
    std::vector<std::vector<double>> A = generate_random_matrix<double>(m, n);
    auto [time, info] = timeit(qr_using_dgeqrf, A);

    if (info == 0) {
        std::cout << "QR factorization succeeded in " << time << " seconds" << std::endl;
    }
    else {
        std::cerr << "QR factorization failed with error code " << info << std::endl;
    }

    return 0;
}
