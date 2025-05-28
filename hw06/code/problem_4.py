import numpy as np
from numpy.linalg import norm


def generate_symmetric_matrix():
    eigenvalues = np.concatenate((np.arange(1.00, 9.01, 0.01), np.array([10, 12, 16, 24])))
    Q = np.random.randn(805, 805)
    Q, _ = np.linalg.qr(Q)
    A = Q @ np.diag(eigenvalues) @ Q.T
    return A


def minres_restart(A, b, x0=None, max_iter=65536, tol=1e-6, restart=20):
    n = len(b)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.copy(x0)
    residuals = []
    r = b - A @ x
    beta = norm(r)
    for outer_iter in range(max_iter // restart + 1):
        if beta < tol:
            break
        T = np.zeros((restart+1, restart))
        Q_givens = np.zeros((restart+1, restart))
        g = np.zeros(restart+1)
        g[0] = beta
        V = np.zeros((n, restart+1))
        V[:, 0] = r / beta
        for j in range(min(restart, max_iter - outer_iter * restart)):
            w = A @ V[:, j]
            alpha = np.dot(w, V[:, j])
            T[j, j] = alpha
            w = w - alpha * V[:, j]
            if j > 0:
                w = w - T[j, j-1] * V[:, j-1]
            for i in range(j):
                w = w - np.dot(w, V[:, i]) * V[:, i]
            beta = norm(w)
            if j < restart - 1:
                T[j+1, j] = beta
                V[:, j+1] = w / beta if beta != 0 else np.zeros(n)
            for i in range(j):
                temp = T[i, j]
                T[i, j] = Q_givens[i, i] * temp + Q_givens[i+1, i] * T[i+1, j]
                T[i+1, j] = -Q_givens[i+1, i] * temp + Q_givens[i, i] * T[i+1, j]
            if j < restart - 1:
                a = T[j, j]
                b_val = T[j+1, j]
                r_val = np.sqrt(a**2 + b_val**2)
                c = a / r_val
                s = b_val / r_val
                if j < restart - 1:
                    Q_givens[j, j] = c
                    Q_givens[j+1, j] = s
                T[j, j] = r_val
                T[j+1, j] = 0.0
                temp = g[j]
                g[j] = c * temp
                g[j+1] = -s * temp
            residuals.append(abs(g[j+1]))
            if residuals[-1] < tol:
                break
        y = np.linalg.lstsq(T[:j+2, :j+1], g[:j+2], rcond=None)[0]
        x = x + V[:, :j+1] @ y
        r = b - A @ x
        beta = norm(r)
        if beta < tol:
            break
    return x, residuals


if __name__ == "__main__":
    A = generate_symmetric_matrix()
    b = np.random.randn(805)
    x, residuals = minres_restart(A, b, max_iter=100000, tol=1e-6, restart=1000)
    iter_times = len(residuals)
    Ax_b = A @ x - b
    print(f"迭代次数: {iter_times}")
    print(f"Ax - b 的范数: {np.linalg.norm(Ax_b)}")
