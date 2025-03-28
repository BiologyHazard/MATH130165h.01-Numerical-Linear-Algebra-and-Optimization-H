#include "classical_gram_schmidt.h"
#include "householder_qr.h"

int main() {
    std::size_t m, n;
    std::cin >> m >> n;

    auto A = zeros<double>(m, n);
    std::cin >> A;

    std::cout.precision(std::numeric_limits<double>::max_digits10);

    auto [cgs_Q, cgs_R] = classical_gram_schmidt(A);
    for (std::size_t i = 0; i < n; ++i) {
        std::cout << cgs_R[i][i] << " ";
    }
    std::cout << std::endl;

    auto householder_R = householder_qr(A).second;
    for (std::size_t i = 0; i < n; ++i) {
        std::cout << householder_R[i][i] << " ";
    }
    std::cout << std::endl;
}
