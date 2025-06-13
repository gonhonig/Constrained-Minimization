import unittest
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, lil_matrix

from src.constrained_min import InteriorPointSolver
from src.function import SumSquares, TotalVariation, Linear
from src.image_denoising import load_image, show_image
from src.cones import *


class TestImageDenoising(unittest.TestCase):
    def setUp(self):
        self.Solver = InteriorPointSolver()

    def test_image_denoising_SOCP(self):
        n = 10
        offset = 100
        Y = load_image('noisy_img.jpg')[offset:offset + n, offset:offset + n]
        show_image(Y, "noisy")

        m, n = Y.shape
        num_pixels = m * n

        # Optimized constraint construction
        print("Building TV constraints...")
        ineq = self._build_tv_constraints_optimized(m, n, num_pixels)

        print("Building data fidelity constraint...")
        # Data fidelity constraint: ||x - y||_2 <= t_data
        t_data_idx = num_pixels + len(ineq)  # Last variable
        len_x = t_data_idx + 1

        A_data = sparse.eye(num_pixels, len_x, format='csr')
        b_data = -Y.ravel()
        c_data = np.zeros(len_x)
        c_data[t_data_idx] = 1

        ineq.append(SOC(A_data, b=b_data, c=c_data))

        # Objective: minimize sum of TV variables + lambda * data_fidelity_variable
        objective_coeffs = np.zeros(len_x)
        objective_coeffs[num_pixels:t_data_idx] = 1  # TV variables
        objective_coeffs[t_data_idx] = 0.3  # Data fidelity variable

        f = Linear(objective_coeffs)

        print(f"Total variables: {len_x}")
        print(f"Total constraints: {len(ineq)}")
        print("Solving...")

        result = self.Solver.solve(func=f, ineq_constraints=ineq, x0=len_x)

        X = result['x'][:num_pixels].reshape((m, n))
        show_image(X, "clean")

    def _build_tv_constraints_optimized(self, m, n, num_pixels):
        """Optimized TV constraint construction using vectorized operations"""
        ineq = []
        constraint_idx = 0

        # Pre-allocate arrays for constraint indices
        vertical_constraints = []
        horizontal_constraints = []

        # Collect all vertical difference constraints
        for i in range(m - 1):
            for j in range(n):
                curr_pixel = i * n + j
                next_pixel = (i + 1) * n + j
                t_idx = num_pixels + constraint_idx

                vertical_constraints.append((curr_pixel, next_pixel, t_idx))
                constraint_idx += 1

        # Collect all horizontal difference constraints
        for i in range(m):
            for j in range(n - 1):
                curr_pixel = i * n + j
                next_pixel = i * n + (j + 1)
                t_idx = num_pixels + constraint_idx

                horizontal_constraints.append((curr_pixel, next_pixel, t_idx))
                constraint_idx += 1

        total_vars = num_pixels + constraint_idx + 1  # +1 for data fidelity variable

        # Build vertical constraints in batch
        print(f"Building {len(vertical_constraints)} vertical constraints...")
        for curr_pixel, next_pixel, t_idx in vertical_constraints:
            # Use direct sparse matrix construction
            row = np.array([0, 0])
            col = np.array([curr_pixel, next_pixel])
            data = np.array([-1.0, 1.0])

            A = csr_matrix((data, (row, col)), shape=(1, total_vars))
            c = np.zeros(total_vars)
            c[t_idx] = 1.0

            ineq.append(SOC(A, c=c))

        # Build horizontal constraints in batch
        print(f"Building {len(horizontal_constraints)} horizontal constraints...")
        for curr_pixel, next_pixel, t_idx in horizontal_constraints:
            # Use direct sparse matrix construction
            row = np.array([0, 0])
            col = np.array([curr_pixel, next_pixel])
            data = np.array([-1.0, 1.0])

            A = csr_matrix((data, (row, col)), shape=(1, total_vars))
            c = np.zeros(total_vars)
            c[t_idx] = 1.0

            ineq.append(SOC(A, c=c))

        return ineq

    @unittest.skip("Temporarily disabled")
    def test_image_denoising_TV(self):
        n = 30
        offset = 100
        noisy_image = load_image('noisy_img.jpg')[offset:offset + n, offset:offset + n]
        show_image(noisy_image, "noisy")
        f = TotalVariation(noisy_image.shape) + 0.3 * SumSquares(A=noisy_image)
        result = self.Solver.solve(func=f, x0=noisy_image.size)
        show_image(result['x'].reshape(noisy_image.shape), "clean")


if __name__ == '__main__':
    unittest.main()