#include <chrono>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// 声明 dgemm_ 函数
extern "C" {
    void dgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k,
        const double *alpha, const double *a, const int *lda, const double *b, const int *ldb,
        const double *beta, double *c, const int *ldc);
}

// 将矩阵的一行转换为字符串
template <typename T>
std::string row_to_string(const std::vector<T> &row, const std::size_t max_size = 10) {
    std::size_t n = row.size();
    std::string s = "[";
    for (std::size_t i = 0; i < n; ++i) {
        s += std::to_string(row[i]);
        if (i < n - 1) {
            s += ", ";
        }
        if (i == max_size - 1) {
            s += "...";
            break;
        }
    }
    s += "]";
    return s;
}

// 将矩阵转换为字符串
template <typename T>
std::string matrix_to_string(const std::vector<std::vector<T>> &matrix, const std::size_t max_size = 10) {
    std::size_t rows = matrix.size();
    std::string s = "[";
    for (std::size_t i = 0; i < rows; ++i) {
        s += row_to_string(matrix[i], max_size);
        if (i < rows - 1) {
            s += ",\n ";
        }
        if (i == max_size - 1) {
            s += "...\n";
            break;
        }
    }
    s += "]";
    return s;
}

// 生成随机矩阵
std::vector<std::vector<double>> generate_random_matrix(std::size_t rows, std::size_t cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0, 1);

    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            matrix[i][j] = dis(gen);
        }
    }

    return matrix;
}

// 手动实现矩阵乘法
template <typename T>
std::vector<std::vector<T>> matrix_multiply_by_hand(const std::vector<std::vector<T>> &A, const std::vector<std::vector<T>> &B) {
    std::size_t m = A.size();
    std::size_t n = A[0].size();
    std::size_t k = B[0].size();

    if (n != B.size()) {
        throw std::invalid_argument("Unable to multiply matrices of incompatible sizes. The cols of A must be equal to the rows of B.");
    }

    std::vector<std::vector<T>> C(m, std::vector<T>(k, static_cast<T>(0)));

    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < k; ++j) {
            for (std::size_t l = 0; l < n; ++l) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }
    return C;
}

// 将二维矩阵展平为一维数组
template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>> &matrix) {
    std::size_t rows = matrix.size();
    std::size_t cols = matrix[0].size();
    std::vector<T> flat(rows * cols);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            flat[i + j * rows] = matrix[i][j];  // column-major storage
        }
    }
    return flat;
}

// 将一维数组恢复为二维矩阵
template <typename T>
std::vector<std::vector<T>> unflatten(const std::vector<T> &flat, std::size_t rows, std::size_t cols) {
    std::vector<std::vector<T>> matrix(rows, std::vector<T>(cols));
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            matrix[i][j] = flat[i + j * rows];  // column-major storage
        }
    }
    return matrix;
}

// 使用 dgemm_ 函数进行矩阵乘法
template <typename T>
std::vector<std::vector<T>> matrix_multiply_by_dgemm(const std::vector<std::vector<T>> &A, const std::vector<std::vector<T>> &B) {
    std::size_t m = A.size();
    std::size_t n = A[0].size();
    std::size_t k = B[0].size();

    if (n != B.size()) {
        throw std::invalid_argument("Unable to multiply matrices of incompatible sizes. The cols of A must be equal to the rows of B.");
    }

    std::vector<T> flat_a = flatten(A);
    std::vector<T> flat_b = flatten(B);
    std::vector<T> flat_c(m * k);

    const double alpha = 1.0;
    const double beta = 0.0;

    int int_m = static_cast<int>(m);
    int int_n = static_cast<int>(n);
    int int_k = static_cast<int>(k);

    dgemm_("N", "N", &int_m, &int_k, &int_n, &alpha, flat_a.data(), &int_m, flat_b.data(), &int_n, &beta, flat_c.data(), &int_m);

    return unflatten(flat_c, m, k);
}

// 计时函数
template<typename Func, typename... Args>
auto timeit(Func func, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();
    auto result = func(std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    return std::make_pair(duration_seconds, result);
}

int main() {
    using T = double;

    std::size_t rows_A = 500;
    std::size_t cols_A = 500;
    std::size_t rows_B = 500;
    std::size_t cols_B = 500;

    std::vector<std::vector<T>> A = generate_random_matrix(rows_A, cols_A);
    std::vector<std::vector<T>> B = generate_random_matrix(rows_B, cols_B);

    std::cout << "A =" << std::endl << matrix_to_string(A) << std::endl;
    std::cout << "B =" << std::endl << matrix_to_string(B) << std::endl;

    try {
        // 手动矩阵乘法计时
        auto [duration_seconds_by_hand, C_by_hand] = timeit(matrix_multiply_by_hand<T>, A, B);
        std::cout << "Multiplication by hand took " << duration_seconds_by_hand << " seconds" << std::endl;

        // 使用 dgemm_ 函数矩阵乘法计时
        auto [duration_seconds_by_dgemm, C_by_dgemm] = timeit(matrix_multiply_by_dgemm<T>, A, B);
        std::cout << "Multiplication by dgemm took " << duration_seconds_by_dgemm << " seconds" << std::endl;
    }
    catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
