#include <chrono>
#include <cmath>
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

// 定义 Matrix 类型
template <typename T>
using Matrix = std::vector<std::vector<T>>;

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
std::string matrix_to_string(const Matrix<T> &matrix, const std::size_t max_size = 10) {
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
template <typename T>
Matrix<T> generate_random_matrix(std::size_t rows, std::size_t cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(0, 1);

    Matrix<T> matrix(rows, std::vector<T>(cols));
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            matrix[i][j] = dis(gen);
        }
    }

    return matrix;
}

// 手动实现矩阵乘法
template <typename T>
Matrix<T> matrix_multiply_by_hand(const Matrix<T> &A, const Matrix<T> &B) {
    std::size_t m = A.size();
    std::size_t n = A[0].size();
    std::size_t k = B[0].size();

    if (n != B.size()) {
        throw std::invalid_argument("Unable to multiply matrices of incompatible sizes. The cols of A must be equal to the rows of B.");
    }

    Matrix<T> C(m, std::vector<T>(k, static_cast<T>(0)));

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
std::vector<T> flatten(const Matrix<T> &matrix) {
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
Matrix<T> unflatten(const std::vector<T> &flat, std::size_t rows, std::size_t cols) {
    Matrix<T> matrix(rows, std::vector<T>(cols));
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            matrix[i][j] = flat[i + j * rows];  // column-major storage
        }
    }
    return matrix;
}

// 使用 dgemm_ 函数进行矩阵乘法
template <typename T>
Matrix<T> matrix_multiply_by_dgemm(const Matrix<T> &A, const Matrix<T> &B) {
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

// 比较两个矩阵是否近似相等
template <typename T>
bool all_close(const Matrix<T> &A, const Matrix<T> &B, double relative_tolerance = 1.0e-5, double absolute_tolerance = 1.0e-8) {
    if (A.size() != B.size() || A[0].size() != B[0].size()) {
        return false;
    }

    std::size_t rows = A.size();
    std::size_t cols = A[0].size();

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            T diff = std::abs(A[i][j] - B[i][j]);
            if (diff > (absolute_tolerance + relative_tolerance * std::abs(B[i][j]))) {
                return false;
            }
        }
    }
    return true;
}

// 计时函数
template<typename Func, typename... Args>
std::pair<double, typename std::result_of<Func(Args...)>::type> timeit(Func func, Args&&... args) {
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

    Matrix<T> A = generate_random_matrix<T>(rows_A, cols_A);
    Matrix<T> B = generate_random_matrix<T>(rows_B, cols_B);

    std::cout << "A =" << std::endl << matrix_to_string(A) << std::endl;
    std::cout << "B =" << std::endl << matrix_to_string(B) << std::endl;

    try {
        // 手动矩阵乘法计时
        auto result_by_hand = timeit(matrix_multiply_by_hand<T>, A, B);
        double duration_seconds_by_hand = result_by_hand.first;
        Matrix<T> C_by_hand = result_by_hand.second;
        std::cout << "C by hand =" << std::endl << matrix_to_string(C_by_hand) << std::endl;
        std::cout << "Multiplication by hand took " << duration_seconds_by_hand << " seconds" << std::endl;

        // 使用 dgemm_ 函数矩阵乘法计时
        auto result_by_dgemm = timeit(matrix_multiply_by_dgemm<T>, A, B);
        double duration_seconds_by_dgemm = result_by_dgemm.first;
        Matrix<T> C_by_dgemm = result_by_dgemm.second;
        std::cout << "C by dgemm =" << std::endl << matrix_to_string(C_by_dgemm) << std::endl;
        std::cout << "Multiplication by dgemm took " << duration_seconds_by_dgemm << " seconds" << std::endl;

        // 比较两个矩阵是否近似相等
        bool is_close = all_close(C_by_hand, C_by_dgemm);
        std::cout << "The two matrices are " << (is_close ? "" : "not ") << "close." << std::endl;
    }
    catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
