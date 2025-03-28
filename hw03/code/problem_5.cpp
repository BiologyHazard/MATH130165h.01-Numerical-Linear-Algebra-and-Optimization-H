#include "cholesky_factorization.h"
#include "matrix_vector_operations.h"


int main() {
    std::size_t n = 10;
    auto c = generate_random_matrix<double>(n, n, true);
    auto a = c * transpose(c);
    auto a_copy = a;

    cholesky_factorization_in_place(a_copy);
    auto r = upper_triangle(a_copy);

    std::cout << "n = " << n << std::endl;
    // std::cout << "A = " << a << std::endl;
    // std::cout << "R = " << r << std::endl;
    std::cout << "Residual Forbenius norm: " << matrix_norm<NormType::FROBENIUS>(a - transpose(r) * r) << std::endl;
}
