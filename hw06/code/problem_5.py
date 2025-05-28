import numpy as np
from problem_4 import generate_symmetric_matrix, minres_restart

if __name__ == "__main__":
    A = generate_symmetric_matrix()
    b = np.random.randn(805)
    x, residuals = minres_restart(A, b, max_iter=100000, tol=1e-6, restart=20)
    iter_times = len(residuals)
    Ax_b = A @ x - b
    print(f"迭代次数: {iter_times}")
    print(f"Ax - b 的范数: {np.linalg.norm(Ax_b)}")
