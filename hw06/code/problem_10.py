import numpy as np


def line_search(f, grad, x, p):
    return 0.01


def lbfgs(f, grad, x0, max_iter=1000, tol=1e-6, m=10):
    x = np.copy(x0)
    n = len(x)
    s_list = []
    y_list = []

    for k in range(max_iter):
        g = grad(x)

        if np.linalg.norm(g) < tol:
            break

        q = np.copy(g)
        alpha_list = []

        for i in reversed(range(min(k, m))):
            s = s_list[i]
            y = y_list[i]
            rho = 1.0 / np.dot(y, s)
            alpha = rho * np.dot(s, q)
            alpha_list.append(alpha)
            q = q - alpha * y

        if k > 0:
            s_prev = s_list[-1]
            y_prev = y_list[-1]
            gamma = np.dot(s_prev, y_prev) / np.dot(y_prev, y_prev)
        else:
            gamma = 1.0

        r = gamma * q

        for i in range(min(k, m)):
            s = s_list[i]
            y = y_list[i]
            rho = 1.0 / np.dot(y, s)
            beta = rho * np.dot(y, r)
            alpha = alpha_list.pop()
            r = r + s * (alpha - beta)

        p = -r

        alpha = line_search(f, grad, x, p)

        s = alpha * p
        x_new = x + s

        g_new = grad(x_new)
        y = g_new - g

        s_list.append(s)
        y_list.append(y)

        if len(s_list) > m:
            s_list.pop(0)
            y_list.pop(0)

        x = x_new

    return x


def f(x):
    return np.sum(x**2) + np.sum(np.sin(x))


def grad_f(x):
    return 2 * x + np.cos(x)


if __name__ == "__main__":
    x0 = np.random.randn(10)
    x_opt = lbfgs(f, grad_f, x0)
    print(f"最优解: {x_opt}")
    print(f"目标函数值: {f(x_opt)}")
