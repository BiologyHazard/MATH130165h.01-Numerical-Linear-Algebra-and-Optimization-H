import matplotlib.pyplot as plt
import numpy as np


def lbfgs_with_backtracking(f, grad_f, x0, m=10, alpha_init=1.0, c=1e-4, rho=0.5, tol=1e-6, max_iter=1000):
    n = len(x0)
    x_k = x0.copy()
    g_k = grad_f(x_k)
    S = []
    Y = []
    R = []

    history = {'x': [x_k.copy()], 'f': [f(x_k)], 'grad_norm': [np.linalg.norm(g_k)]}

    k = 0
    while k < max_iter and np.linalg.norm(g_k) > tol:
        q = g_k.copy()
        num_corr = len(S)
        alpha_list = np.zeros(num_corr)

        for i in range(num_corr-1, -1, -1):
            alpha_i = R[i] * np.dot(S[i], q)
            alpha_list[i] = alpha_i
            q = q - alpha_i * Y[i]

        if num_corr > 0:
            s_last = S[-1]
            y_last = Y[-1]
            gamma_k = np.dot(s_last, y_last) / np.dot(y_last, y_last)
            r = gamma_k * q
        else:
            r = 1.0 * q

        for i in range(0, num_corr):
            beta_i = R[i] * np.dot(Y[i], r)
            r = r + S[i] * (alpha_list[i] - beta_i)
        p_k = -r

        alpha = alpha_init
        while True:
            x_trial = x_k + alpha * p_k
            f_trial = f(x_trial)
            f_current = f(x_k)
            g_dot_p = np.dot(g_k, p_k)
            armijo = f_current + c * alpha * g_dot_p
            if f_trial <= armijo:
                break
            alpha *= rho

        x_next = x_k + alpha * p_k
        g_next = grad_f(x_next)

        s_k = x_next - x_k
        y_k = g_next - g_k
        rho_k = 1.0 / np.dot(y_k, s_k)

        S.append(s_k)
        Y.append(y_k)
        R.append(rho_k)

        if len(S) > m:
            S.pop(0)
            Y.pop(0)
            R.pop(0)

        x_k = x_next
        g_k = g_next
        k += 1

        history['x'].append(x_k.copy())
        history['f'].append(f(x_k))
        history['grad_norm'].append(np.linalg.norm(g_k))

    return x_k, history


n = 100
np.random.seed(42)

A = np.random.randn(n, n)
A = A.T @ A + 1e-3 * np.eye(n)

b = np.random.randn(n)


def f_quad(x):
    return 0.5 * x.T @ A @ x - b.T @ x


def grad_f_quad(x):
    return A @ x - b


x_min = np.linalg.solve(A, b)
f_min = f_quad(x_min)

m_list = [1, 3, 5, 10, 20, 50]
max_iter = 500
x0 = np.zeros(n)

convergence_data = {}

for m_val in m_list:
    print(f"Running L-BFGS with m = {m_val}...")
    x_opt, history = lbfgs_with_backtracking(
        f_quad,
        grad_f_quad,
        x0,
        m=m_val,
        tol=1e-10,
        max_iter=max_iter
    )
    f_residual = [f_val - f_min for f_val in history['f']]
    convergence_data[m_val] = f_residual

plt.figure(figsize=(10, 6))

for m_val, residuals in convergence_data.items():
    iterations = range(len(residuals))
    plt.semilogy(iterations, residuals, label=f'm = {m_val}')

plt.xlabel('Iteration (k)')
plt.ylabel('log10(f(x_k) - f*)')
plt.title('L-BFGS Convergence on Quadratic Function (n={})'.format(n))
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
