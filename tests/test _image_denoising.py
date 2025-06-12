import unittest

import numpy as np

from src.common import Linear, Quadratic, SumSquares, TotalVariation, Variable, Stack
from src.cones import SOC, NormSOC
from src.constrained_min import InteriorPointSolver
from src.image_denoising import load_image, show_image
from src.phase_one import get_initial_point
from skimage.transform import resize


class TestImageDenoising(unittest.TestCase):
    def setUp(self):
        self.Solver = InteriorPointSolver()

    def test_image_denoising_SOCP(self):
        return
        n = 30
        offset = 100
        Y = load_image('noisy_img.jpg')[offset:offset + n, offset:offset + n]
        show_image(Y, "noisy")

        x0 = get_initial_point(Y, method="median")
        ineq = []

        m, n = Y.shape
        X = Variable((m,n))
        T = Variable((m-1,n-1))

        for i in range(n-1):
            for j in range(n-1):
                t = T[i, j]
                dx = X[i, j + 1] - X[i, j]
                dy = X[i + 1, j] - X[i, j]
                ineq.append(NormSOC(t, Stack((dx, dy))))


        self.Solver.solve(func=f,
                          ineq_constraints=ineq,
                          eq_constraints_mat=A,
                          eq_constraints_rhs=b,
                          x0=x0)

    def test_image_denoising_TV(self):
        n = 30
        offset = 100
        noisy_image = load_image('noisy_img.jpg')[offset:offset + n, offset:offset + n]
        show_image(noisy_image, "noisy")
        x0 = get_initial_point(noisy_image, method="median")

        f = TotalVariation(noisy_image.shape) + 0.3 * SumSquares(A=noisy_image)
        result = self.Solver.solve(func=f, x0=x0)
        show_image(result['x'], "clean")


if __name__ == '__main__':
    unittest.main()