import unittest

import numpy as np

from src.common import Linear, Quadratic
from src.cones import SOC
from src.constrained_min import InteriorPointSolver


class TestUnconstrained(unittest.TestCase):
    def setUp(self):
        self.Solver = InteriorPointSolver()

    # def test_quadratic(self):
    #     ineq = [Linear(row) for row in -np.eye(3)]
    #     f = Quadratic(np.eye(3)) + Linear([0, 0, 2]) + 1
    #     x0 = np.array([0.1, 0.2, 0.7])
    #     A = [1, 1, 1]
    #     b = 1
    #     self.Solver.solve(func=f,
    #                       ineq_constraints=ineq,
    #                       eq_constraints_mat=A,
    #                       eq_constraints_rhs=b,
    #                       x0=x0)

    def test_SOCP(self):
        f = Quadratic(np.eye(2))
        ineq = [
            Linear([-1,0]),
            Linear([0,-1]),
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

if __name__ == '__main__':
    unittest.main()