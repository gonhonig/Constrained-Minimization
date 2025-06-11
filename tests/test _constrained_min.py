import unittest

import numpy as np

from src.common import Linear, Quadratic, SumSquares, TotalVariation
from src.cones import SOC
from src.constrained_min import InteriorPointSolver
from src.image_denoising import load_image, show_image
from src.phase_one import get_initial_point
from skimage.transform import resize


class TestUnconstrained(unittest.TestCase):
    def setUp(self):
        self.Solver = InteriorPointSolver()

    def test_quadratic(self):
        return
        ineq = [Linear(row) for row in -np.eye(3)]
        f = Quadratic(np.eye(3)) + Linear([0, 0, 2]) + 1
        x0 = np.array([0.1, 0.2, 0.7])
        A = [1, 1, 1]
        b = 1
        self.Solver.solve(func=f,
                          ineq_constraints=ineq,
                          eq_constraints_mat=A,
                          eq_constraints_rhs=b,
                          x0=x0)

    def test_SOCP(self):
        return
        f = SumSquares()
        ineq = [
            -Linear([1,0]),
            -Linear([0,1]),
            SOC(A=np.eye(2), d=2)
        ]
        A = [1,1]
        b = 1
        x0 = np.array([0.1, 0.9])
        self.Solver.solve(func=f,
                          ineq_constraints=ineq,
                          eq_constraints_mat=A,
                          eq_constraints_rhs=b,
                          x0=x0)

    def test_image_denoising_TV(self):
        noisy_image = load_image('noisy_img.jpg')
        noisy_image = resize(noisy_image, (50, 50), anti_aliasing=True)
        show_image(noisy_image, "noisy")
        f = TotalVariation(noisy_image.shape) + SumSquares(A=noisy_image)
        x0 = get_initial_point(noisy_image, method="median")
        result = self.Solver.solve(func=f, x0=x0)
        show_image(result['x'], "clean")


if __name__ == '__main__':
    unittest.main()