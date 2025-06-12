import unittest

import numpy as np

from src.common import Linear, Quadratic, SumSquares, TotalVariation, Variable
from src.cones import SOC, NormSOC
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


if __name__ == '__main__':
    unittest.main()