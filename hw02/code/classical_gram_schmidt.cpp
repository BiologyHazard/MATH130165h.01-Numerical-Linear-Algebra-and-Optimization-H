#include <iostream>
#include "matrix_vector_operations.h"

// CGS 简单实现
// 也许列主序会更好，但是这里为了便于理解，使用行主序
template <typename T>
auto classical_gram_schmidt(const std::vector<std::vector<T>> &a) {
    auto m = a.size(), n = a[0].size();
    auto q = std::vector<std::vector<T>>(m, std::vector<T>(n, 0));
    auto r = std::vector<std::vector<T>>(n, std::vector<T>(n, 0));

    for (std::size_t j = 0; j < n; ++j) {
        auto v = get_column(a, j);
        for (std::size_t i = 0; i < j; ++i) {
            r[i][j] = inner_product(get_column(q, i), get_column(a, j));
            v -= r[i][j] * get_column(q, i);
        }
        r[j][j] = vector_norm(v, 2);
        set_column(q, j, v / r[j][j]);
    }

    return std::make_pair(q, r);
}

// 测试示例
int main() {
    std::size_t m = 2000, n = 2000;
    auto A = generate_random_matrix<double>(m, n);
    auto [time, qr] = timeit(classical_gram_schmidt<double>, A);
    auto [Q, R] = qr;
    std::cout << "QR factorization succeeded in " << time << " seconds" << std::endl;
    std::cout << "Q matrix:" << std::endl << Q << std::endl;
    std::cout << "R matrix:" << std::endl << R << std::endl;
    std::cout << "QR matrix:" << std::endl << Q * R << std::endl;

    return 0;
}
