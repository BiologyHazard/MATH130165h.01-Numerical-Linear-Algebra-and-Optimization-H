#include "lu_factorization_and_solve.h"


int main() {
    std::size_t n = 2000;
    auto a = generate_random_matrix<double>(n, n, true);
    auto b = generate_random_vector<double>(n, true);
    auto a_copy = a;
    auto b_copy = b;
    auto x = solve_linear_system_in_place(a_copy, b_copy);

    std::cout << "n = " << n << std::endl;
    // std::cout << "A = " << a << std::endl;
    // std::cout << "b = " << b << std::endl;
    // std::cout << "L, U = " << std::endl << a_copy << std::endl;
    // std::cout << "x = " << x << std::endl;
    std::cout << "b - Ax 2-norm: " << vector_norm(b - a * x, 2) << std::endl;
}
