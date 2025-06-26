import unittest

import numpy as np

from src.constrained_min import InteriorPointSolver
from src.function import Linear, Quadratic


class TestUnconstrained(unittest.TestCase):
    def setUp(self):
        self.Solver = InteriorPointSolver()

    # @unittest.skip("Temporarily disabled")
    def test_quadratic(self):
        ineq = [Linear(row) for row in -np.eye(3)]
        f = Quadratic(np.eye(3)) + Linear([0, 0, 2]) + 1
        x0 = np.array([0.1, 0.2, 0.7])
        A = [1, 1, 1]
        b = 1
        self.Solver.solve(func=f,
                          x0=x0,
                          ineq_constraints=ineq,
                          eq_constraints_mat=A,
                          eq_constraints_rhs=b)

if __name__ == '__main__':
    unittest.main()