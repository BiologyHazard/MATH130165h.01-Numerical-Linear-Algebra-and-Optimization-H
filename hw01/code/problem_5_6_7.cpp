#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace std;


extern "C" {
    void dgemm_(const char *transa, const char *transb, const size_t *m, const size_t *n, const size_t *k,
        const double *alpha, const double *a, const size_t *lda, const double *b, const size_t *ldb,
        const double *beta, double *c, const size_t *ldc);
}


template <typename T>
string row_to_string(const vector<T> &row, const size_t max_size = 10) {
    size_t n = row.size();
    string s = "[";
    for (size_t i = 0; i < n; ++i) {
        s += to_string(row[i]);
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


template <typename T>
string matrix_to_string(const vector<vector<T>> &matrix, const size_t max_size = 10) {
    size_t rows = matrix.size();
    string s = "[";
    for (size_t i = 0; i < rows; ++i) {
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


vector<vector<double>> generate_random_matrix(size_t rows, size_t cols) {
    // unsigned int seed = 0;
    // mt19937 gen(seed);

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(0, 1);

    vector<vector<double>> matrix(rows, vector<double>(cols));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            matrix[i][j] = dis(gen);
        }
    }

    return matrix;
}


template <typename T>
vector<vector<T>> matrix_multiply_by_hand(const vector<vector<T>> &A, const vector<vector<T>> &B) {
    size_t m = A.size();
    size_t n = A[0].size();
    size_t k = B[0].size();

    if (n != B.size()) {
        throw invalid_argument("Unable to multiply matrices of incompatible sizes. The cols of A must be equal to the rows of B.");
    }

    vector<vector<T>> C(m, vector<T>(k, static_cast<T>(0)));

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < k; ++j) {
            for (size_t k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}


template <typename T>
vector<T> flatten(const vector<vector<T>> &matrix) {
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    vector<T> flat(rows * cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            flat[i + j * rows] = matrix[i][j];  // column-major storage
        }
    }
    return flat;
}


template <typename T>
vector<vector<T>> unflatten(const vector<T> &flat, size_t rows, size_t cols) {
    vector<vector<T>> matrix(rows, vector<T>(cols));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            matrix[i][j] = flat[i + j * rows];  // column-major storage
        }
    }
    return matrix;
}


template <typename T>
vector<vector<T>> matrix_multiply_by_dgemm(const vector<vector<T>> &A, const vector<vector<T>> &B) {
    size_t m = A.size();
    size_t n = A[0].size();
    size_t k = B[0].size();

    if (n != B.size()) {
        throw invalid_argument("Unable to multiply matrices of incompatible sizes. The cols of A must be equal to the rows of B.");
    }

    vector<T> flat_a = flatten(A);
    vector<T> flat_b = flatten(B);
    vector<T> flat_c(m * k);

    const double alpha = 1.0;
    const double beta = 0.0;

    dgemm_("N", "N", &m, &k, &n, &alpha, flat_a.data(), &m, flat_b.data(), &n, &beta, flat_c.data(), &m);

    return unflatten(flat_c, m, k);
}


template<typename Func, typename... Args>
auto timeit(Func func, Args&&... args) {
    auto start = chrono::high_resolution_clock::now();
    auto result = func(forward<Args>(args)...);
    auto end = chrono::high_resolution_clock::now();
    auto duration_seconds = chrono::duration_cast<chrono::duration<double>>(end - start).count();
    return make_pair(duration_seconds, result);
}


int main() {
    using T = double;

    size_t rows_A = 500;
    size_t cols_A = 500;
    size_t rows_B = 500;
    size_t cols_B = 500;

    vector<vector<T>> A = generate_random_matrix(rows_A, cols_A);
    vector<vector<T>> B = generate_random_matrix(rows_B, cols_B);

    cout << "A =" << endl << matrix_to_string(A) << endl;
    cout << "B =" << endl << matrix_to_string(B) << endl;

    // Multiply by hand
    auto [duration_seconds_by_hand, C_by_hand] = timeit(matrix_multiply_by_hand<T>, A, B);
    cout << "Multiplication by hand took " << duration_seconds_by_hand << " seconds" << endl;


    // Multiply by dgemm
    auto [duration_seconds_by_dgemm, C_by_dgemm] = timeit(matrix_multiply_by_dgemm<T>, A, B);
    cout << "Multiplication by dgemm took " << duration_seconds_by_dgemm << " seconds" << endl;

    return 0;
}
