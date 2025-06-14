import unittest
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix

from src.constrained_min import InteriorPointSolver
from src.function import Linear
from src.image_denoising import load_image, show_image
from src.cones import SOC


class TestImageDenoising(unittest.TestCase):
    def setUp(self):
        # Reduce mu for more gradual progress
        self.Solver = InteriorPointSolver(mu=5, epsilon=1e-8)

    def test_image_denoising_SOCP_auto_init(self):
        """Test with automatic initial point finding"""
        n = 8  # Start with small image
        offset = 100
        Y = load_image('noisy_img.jpg')[offset:offset + n, offset:offset + n]
        show_image(Y, "noisy")

        m, n = Y.shape
        num_pixels = m * n

        # Build constraints
        print("Building constraints...")
        ineq = []

        # TV constraints
        num_tv_vars = (m - 1) * n + m * (n - 1)
        tv_idx = num_pixels

        # Vertical TV constraints
        for i in range(m - 1):
            for j in range(n):
                curr = i * n + j
                next = (i + 1) * n + j

                # ||[x_next - x_curr]||_2 <= t
                total_vars = num_pixels + num_tv_vars + 1
                A = sparse.lil_matrix((1, total_vars))
                A[0, curr] = -1.0
                A[0, next] = 1.0

                c = np.zeros(total_vars)
                c[tv_idx] = 1.0

                ineq.append(SOC(A.tocsr(), c=c))
                tv_idx += 1

        # Horizontal TV constraints
        for i in range(m):
            for j in range(n - 1):
                curr = i * n + j
                next = i * n + (j + 1)

                total_vars = num_pixels + num_tv_vars + 1
                A = sparse.lil_matrix((1, total_vars))
                A[0, curr] = -1.0
                A[0, next] = 1.0

                c = np.zeros(total_vars)
                c[tv_idx] = 1.0

                ineq.append(SOC(A.tocsr(), c=c))
                tv_idx += 1

        # Data fidelity constraint: ||x - y||_2 <= t_data
        t_data_idx = num_pixels + num_tv_vars
        total_vars = t_data_idx + 1

        A_data = sparse.eye(num_pixels, total_vars, format='csr')
        b_data = -Y.ravel()
        c_data = np.zeros(total_vars)
        c_data[t_data_idx] = 1.0

        ineq.append(SOC(A_data, b=b_data, c=c_data))

        # Objective: minimize sum of TV terms + lambda * data_fidelity
        lam = 0.5  # Regularization parameter
        objective_coeffs = np.zeros(total_vars)
        objective_coeffs[num_pixels:t_data_idx] = 1.0  # TV terms
        objective_coeffs[t_data_idx] = lam  # Data fidelity

        f = Linear(objective_coeffs)

        print(f"Problem size: {total_vars} variables, {len(ineq)} constraints")

        # Let the solver find initial point
        print("Solving with automatic initial point finding...")
        result = self.Solver.solve(func=f, ineq_constraints=ineq, x0=total_vars)

        # Extract and display result
        X = result['x'][:num_pixels].reshape((m, n))
        X = np.clip(X, 0, 1)
        show_image(X, "clean_auto_init")

        print(f"Optimization completed in {result['iterations']} iterations")
        print(f"Final objective value: {result['y']:.6f}")


if __name__ == '__main__':
    unittest.main()