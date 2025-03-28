// 所有函数都未考虑异常情况，如矩阵维度不匹配等
// 请确保矩阵维度匹配后再调用这些函数

#ifndef VECTOR_OPERATIONS_H
#define VECTOR_OPERATIONS_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <vector>


template <typename T>
inline auto zeros(std::size_t n) {
    return std::vector<T>(n, 0);
}

template <typename T>
inline auto ones(std::size_t m, std::size_t n) {
    return std::vector<std::vector<T>>(m, std::vector<T>(n, 1));
}

template <typename T>
inline auto e_i_vector(std::size_t n, std::size_t i) {
    std::vector<T> e(n, 0);
    e[i] = 1;
    return e;
}

template <typename T>
inline auto zeros(std::size_t m, std::size_t n) {
    return std::vector<std::vector<T>>(m, std::vector<T>(n, 0));
}

template <typename T>
inline auto ones(std::size_t n) {
    return std::vector<T>(n, 1);
}

template <typename T>
inline auto to_diag_matrix(const std::vector<T> &v) {
    auto n = v.size();
    auto result = zeros<T>(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        result[i][i] = v[i];
    }
    return result;
}

template <typename T>
inline auto identity_matrix(std::size_t n) {
    auto diag = ones<T>(n);
    return to_diag_matrix(diag);
}


// 将向量转换为字符串
template <typename T>
auto vector_to_string(const std::vector<T> &v, std::size_t max_size = 10) {
    auto n = v.size();
    auto show_ellipsis = n > max_size;
    auto show_element_count = std::min(n, max_size);

    std::string s = "{";
    for (std::size_t i = 0; i < show_element_count; ++i) {
        s += std::to_string(v[i]);
        if (i < show_element_count - 1) {
            s += ", ";
        }
    }
    if (show_ellipsis) {
        s += ", ...";
    }
    s += "}";
    return s;
}

// 将矩阵转换为字符串
template <typename T>
auto matrix_to_string(const std::vector<std::vector<T>> &matrix, std::size_t max_size = 10) {
    auto rows = matrix.size();
    auto show_ellipsis = rows > max_size;
    auto show_row_count = std::min(rows, max_size);

    std::string s = "{";
    for (std::size_t i = 0; i < show_row_count; ++i) {
        s += vector_to_string(matrix[i]);
        if (i < show_row_count - 1) {
            s += ",\n ";
        }
    }
    if (show_ellipsis) {
        s += ",\n ...\n";
    }
    s += "}";
    return s;
}

// 重载输出运算符
template <typename T>
inline auto &operator<<(std::ostream &os, const std::vector<T> &v) {
    os << vector_to_string(v);
    return os;
}

template <typename T>
inline auto &operator<<(std::ostream &os, const std::vector<std::vector<T>> &matrix) {
    os << matrix_to_string(matrix);
    return os;
}

// 重载输入运算符
template <typename T>
inline auto &operator>>(std::istream &is, std::vector<T> &v) {
    auto n = v.size();
    for (std::size_t i = 0; i < n; ++i) {
        is >> v[i];
    }
    return is;
}

template <typename T>
inline auto &operator>>(std::istream &is, std::vector<std::vector<T>> &matrix) {
    for (auto &row : matrix) {
        is >> row;
    }
    return is;
}


// 向量内积
template <typename T>
inline auto inner_product(const std::vector<T> &v1, const std::vector<T> &v2) {
    T result = 0;
    for (std::size_t i = 0; i < v1.size(); ++i) {
        result += v1[i] * v2[i];
    }
    return result;
}

// 向量 p-范数
// 当 v 为整数向量且 p = 0 / 1 / inf 时，理论上结果也是整数
// 这里为了简便，无论如何都返回 double 类型
template <typename T>
inline double vector_norm(const std::vector<T> &v, double p = 2) {
    // 无穷范数
    if (p == std::numeric_limits<double>::infinity()) {
        T result = 0;
        for (const auto &elem : v) {
            result = std::max(result, std::abs(elem));
        }
        return result;
    }

    // 0 范数
    if (p == 0) {
        return std::count_if(v.begin(), v.end(), [](T elem) { return elem != 0; });
    }

    // p 范数
    double result = 0;
    for (const auto &elem : v) {
        result += std::pow(std::abs(elem), p);
    }
    return std::pow(result, 1.0 / p);
}

enum class NormType {
    ONE,
    INF,
    FROBENIUS,
};

// 矩阵 F-范数
template <typename T, NormType norm_type = NormType::FROBENIUS>
auto matrix_norm(const std::vector<std::vector<T>> &matrix) {
    if constexpr (norm_type == NormType::ONE) {
        T result = 0;
        for (std::size_t j = 0; j < matrix[0].size(); ++j) {
            T column_sum = 0;
            for (const auto &row : matrix) {
                column_sum += std::abs(row[j]);
            }
            result = std::max(result, column_sum);
        }
        return result;
    }
    else if constexpr (norm_type == NormType::INF) {
        T result = 0;
        for (const auto &row : matrix) {
            T row_sum = 0;
            for (const auto &elem : row) {
                row_sum += std::abs(elem);
            }
            result = std::max(result, row_sum);
        }
        return result;
    }
    else if constexpr (norm_type == NormType::FROBENIUS) {
        double result = 0;
        for (const auto &row : matrix) {
            for (const auto &elem : row) {
                result += elem * elem;
            }
        }
        return std::sqrt(result);
    }
    else {
        throw std::invalid_argument("Invalid norm type");
    }
}

// 取矩阵的一列
template <typename T>
inline auto get_column(const std::vector<std::vector<T>> &matrix, std::size_t col) {
    std::vector<T> column{};
    for (const auto &row : matrix) {
        column.emplace_back(row[col]);
    }
    return column;
}

// 设置矩阵的一列
template <typename T>
inline auto set_column(std::vector<std::vector<T>> &matrix, std::size_t col, const std::vector<T> &new_col) {
    for (std::size_t i = 0; i < matrix.size(); ++i) {
        matrix[i][col] = new_col[i];
    }
}

// 取向量的切片
template <typename T>
inline auto get_slice(const std::vector<T> &v, std::size_t start, std::size_t end) {
    return std::vector<T>(v.begin() + start, v.begin() + end);
}

// 设置向量的切片
template <typename T>
inline auto set_slice(std::vector<T> &v, std::size_t start, std::size_t end, const std::vector<T> &new_slice) {
    std::copy(new_slice.begin(), new_slice.end(), v.begin() + start);
}

// 取矩阵的切片
template <typename T>
inline auto get_slice(
    const std::vector<std::vector<T>> &matrix,
    std::size_t row_start,
    std::size_t row_end,
    std::size_t col_start,
    std::size_t col_end
) {
    auto slice = std::vector<std::vector<T>>();
    for (std::size_t i = row_start; i < row_end; ++i) {
        slice.emplace_back(std::vector<T>(matrix[i].begin() + col_start, matrix[i].begin() + col_end));
    }
    return slice;
}

// 设置矩阵的切片
template <typename T>
inline auto set_slice(
    std::vector<std::vector<T>> &matrix,
    std::size_t row_start,
    std::size_t row_end,
    std::size_t col_start,
    std::size_t col_end,
    const std::vector<std::vector<T>> &new_slice
) {
    for (std::size_t i = row_start; i < row_end; ++i) {
        std::copy(new_slice[i - row_start].begin(), new_slice[i - row_start].end(), matrix[i].begin() + col_start);
    }
}

// 操作符重载实现
// 向量加法
template <typename T>
inline auto operator+(const std::vector<T> &v1, const std::vector<T> &v2) {
    std::vector<T> result(v1.size(), 0);
    for (std::size_t i = 0; i < v1.size(); ++i) {
        result[i] = v1[i] + v2[i];
    }
    return result;
}

template <typename T>
inline auto &operator+=(std::vector<T> &v1, const std::vector<T> &v2) {
    for (std::size_t i = 0; i < v1.size(); ++i) {
        v1[i] += v2[i];
    }
    return v1;
}

// 向量减法
template <typename T>
inline auto operator-(const std::vector<T> &v1, const std::vector<T> &v2) {
    std::vector<T> result(v1.size(), 0);
    for (std::size_t i = 0; i < v1.size(); ++i) {
        result[i] = v1[i] - v2[i];
    }
    return result;
}

template <typename T>
inline auto &operator-=(std::vector<T> &v1, const std::vector<T> &v2) {
    for (std::size_t i = 0; i < v1.size(); ++i) {
        v1[i] -= v2[i];
    }
    return v1;
}

// 向量数乘
template <typename VectorType, typename ScalarType>
inline auto operator*(ScalarType scalar, const std::vector<VectorType> &v) {
    using T = decltype(scalar *v[0]);
    std::vector<T> result(v.size(), 0);
    for (std::size_t i = 0; i < v.size(); ++i) {
        result[i] = scalar * v[i];
    }
    return result;
}

template <typename VectorType, typename ScalarType>
inline auto operator*(const std::vector<VectorType> &v, ScalarType scalar) {
    return scalar * v;
}

template <typename VectorType, typename ScalarType>
inline auto &operator*= (std::vector<VectorType> &v, ScalarType scalar) {
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] *= scalar;
    }
    return v;
}

template <typename VectorType, typename ScalarType>
inline auto operator/(const std::vector<VectorType> &v, ScalarType scalar) {
    return (1.0 / scalar) * v;
}

template <typename VectorType, typename ScalarType>
inline auto &operator/=(std::vector<VectorType> &v, ScalarType scalar) {
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] /= scalar;
    }
    return v;
}

// 矩阵加法
template <typename T>
inline auto operator+(const std::vector<std::vector<T>> &m1, const std::vector<std::vector<T>> &m2) {
    auto m = m1.size(), n = m1[0].size();
    auto result = zeros(m, n);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            result[i][j] = m1[i][j] + m2[i][j];
        }
    }
    return result;
}

template <typename T>
inline auto &operator+=(std::vector<std::vector<T>> &m1, const std::vector<std::vector<T>> &m2) {
    auto m = m1.size(), n = m1[0].size();
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            m1[i][j] += m2[i][j];
        }
    }
    return m1;
}

// 矩阵减法
template <typename T>
inline auto operator-(const std::vector<std::vector<T>> &m1, const std::vector<std::vector<T>> &m2) {
    auto m = m1.size(), n = m1[0].size();
    auto result = zeros<T>(m, n);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            result[i][j] = m1[i][j] - m2[i][j];
        }
    }
    return result;
}

template <typename T>
inline auto &operator-=(std::vector<std::vector<T>> &m1, const std::vector<std::vector<T>> &m2) {
    auto m = m1.size(), n = m1[0].size();
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            m1[i][j] -= m2[i][j];
        }
    }
    return m1;
}

// 矩阵数乘
template <typename MatrixType, typename ScalarType>
inline auto operator*(ScalarType scalar, const std::vector<std::vector<MatrixType>> &matrix) {
    using T = decltype(scalar *matrix[0][0]);
    auto m = matrix.size(), n = matrix[0].size();
    auto result = zeros<T>(m, n);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            result[i][j] = scalar * matrix[i][j];
        }
    }
    return result;
}

template <typename MatrixType, typename ScalarType>
inline auto operator*(const std::vector<std::vector<MatrixType>> &matrix, ScalarType scalar) {
    return scalar * matrix;
}

template <typename MatrixType, typename ScalarType>
inline auto &operator*=(std::vector<std::vector<MatrixType>> &matrix, ScalarType scalar) {
    for (auto &row : matrix) {
        for (auto &elem : row) {
            elem *= scalar;
        }
    }
    return matrix;
}

template <typename MatrixType, typename ScalarType>
inline auto operator/(const std::vector<std::vector<MatrixType>> &matrix, ScalarType scalar) {
    return (1.0 / scalar) * matrix;
}

template <typename MatrixType, typename ScalarType>
inline auto &operator/=(std::vector<std::vector<MatrixType>> &matrix, ScalarType scalar) {
    for (auto &row : matrix) {
        for (auto &elem : row) {
            elem /= scalar;
        }
    }
    return matrix;
}

// 矩阵转置
template <typename T>
inline auto transpose(const std::vector<std::vector<T>> &matrix) {
    auto m = matrix.size(), n = matrix[0].size();
    auto result = zeros<T>(n, m);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}


// 矩阵-向量乘法
template <typename T>
inline auto operator*(const std::vector<std::vector<T>> &matrix, const std::vector<T> &v) {
    auto m = matrix.size(), n = matrix[0].size();
    std::vector<T> result(m, 0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            result[i] += matrix[i][j] * v[j];
        }
    }
    return result;
}

// 矩阵-矩阵乘法
template <typename T>
inline auto operator*(const std::vector<std::vector<T>> &m1, const std::vector<std::vector<T>> &m2) {
    auto m = m1.size(), n = m1[0].size(), p = m2[0].size();
    auto result = zeros<T>(m, p);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            for (std::size_t k = 0; k < n; ++k) {
                result[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }
    return result;
}


// 生成随机矩阵
template <typename T = double>
auto generate_random_matrix(std::size_t rows, std::size_t cols, bool fixed_seed = false, unsigned int seed = 0) {
    std::mt19937 gen;
    if (fixed_seed) {
        gen = std::mt19937(seed);
    }
    else {
        std::random_device rd;
        gen = std::mt19937(rd());
    }
    std::uniform_real_distribution<T> dis(0, 1);

    auto matrix = zeros<T>(rows, cols);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            matrix[i][j] = dis(gen);
        }
    }

    return matrix;
}

// 将二维矩阵展平为一维数组
template <typename T>
inline auto flatten(const std::vector<std::vector<T>> &matrix) {
    std::vector<T> result{};
    for (const auto &row : matrix) {
        result.insert(result.end(), row.begin(), row.end());
    }
    return result;
}

// 将一维数组恢复为二维矩阵
template <typename T>
inline auto unflatten(const std::vector<T> &flat, std::size_t rows, std::size_t cols) {
    auto matrix = zeros<T>(rows, cols);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            matrix[i][j] = flat[i * cols + j];
        }
    }
    return matrix;
}

// 将 1d 向量转换为 2d 行向量
template <typename T>
inline auto to_row_vector(const std::vector<T> &v) {
    return std::vector<std::vector<T>>{v};
}

// 将 1d 向量转换为 2d 列向量
template <typename T>
inline auto to_column_vector(const std::vector<T> &v) {
    auto result = std::vector<std::vector<T>>{};
    for (const auto &elem : v) {
        result.emplace_back(std::vector<T>{elem});
    }
    return result;
}

// 比较两个矩阵是否近似相等
template <typename T>
auto all_close(
    const std::vector<std::vector<T>> &A,
    const std::vector<std::vector<T>> &B,
    double relative_tolerance = 1.0e-5,
    double absolute_tolerance = 1.0e-8
) {
    if (A.size() != B.size() || A[0].size() != B[0].size()) {
        return false;
    }

    auto rows = A.size(), cols = A[0].size();
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
auto timeit(Func func, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();
    if constexpr (std::is_void_v<std::invoke_result_t<Func, Args...>>) {
        func(std::forward<Args>(args)...);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        return duration_seconds;
    }
    else {
        auto result = func(std::forward<Args>(args)...);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        return std::make_pair(duration_seconds, result);
    }
}

#endif // VECTOR_OPERATIONS_H
