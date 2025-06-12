import unittest

import numpy as np

from src.function import Linear, Quadratic, SumSquares, TotalVariation
from src.cones import GenericSOC, SOC
from src.constrained_min import InteriorPointSolver
from src.image_denoising import load_image, show_image
from src.phase_one import get_initial_point
from skimage.transform import resize
from src.variable import Variable


class TestUnconstrained(unittest.TestCase):
    def setUp(self):
        self.Solver = InteriorPointSolver()

    @unittest.skip("Temporarily disabled")
    def test_quadratic(self):
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

    def test_SOCP_generic(self):
        f = Quadratic(np.eye(2))
        ineq = [
            Linear([-1,0]),
            Linear([0,-1]),
            GenericSOC(A=np.eye(2), d=2)
        ]
        A = [1,1]
        b = 1
        x0 = np.array([0.1, 0.9])
        self.Solver.solve(func=f,
                          ineq_constraints=ineq,
                          eq_constraints_mat=A,
                          eq_constraints_rhs=b,
                          x0=x0)

    def test_SOCP(self):
        I = np.eye(2)
        I[-1,-1] = 0
        f = Quadratic(I)
        x = Variable(2)
        t = Variable(1)
        ineq = [
            Linear([-1,0,0]),
            Linear([0,-1,0]),
            SOC(x=x, t=t)
        ]
        A = [1,1,0]
        b = 1
        x0 = np.array([0.1, 0.9])
        self.Solver.solve(func=f,
                          ineq_constraints=ineq,
                          eq_constraints_mat=A,
                          eq_constraints_rhs=b,
                          x0=None,
                          variables=(x,t))


if __name__ == '__main__':
    unittest.main()