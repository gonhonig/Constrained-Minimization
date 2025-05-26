import unittest

import numpy as np

from src.common import Function
from src.unconstrained_min import GD, Newton
from src.utils import plot_function_and_paths, plot_objective_vs_iterations
from tests import examples


x0 = np.array([1, 1])
obj_tol = 1e-8
param_tol = 1e-12
max_iter = 100
wolfe_const = 0.01
backtracking_const = 0.5

class TestMinimizers(unittest.TestCase):
    def setUp(self):
        self.GD = GD(obj_tol, param_tol, max_iter, wolfe_const, backtracking_const)
        self.Newton = Newton(obj_tol, param_tol, max_iter, wolfe_const, backtracking_const)

    def solve_and_plot(self):
        self.GD.solve(self.f, self.x0)
        self.Newton.solve(self.f, self.x0)
        plot_function_and_paths([self.GD, self.Newton], self.f)
        plot_objective_vs_iterations([self.GD, self.Newton], self.f)

        return True

    def test_circle(self):
        self.f = examples.circle
        self.x0 = x0
        self.assertTrue(self.solve_and_plot())

    def test_ellipses(self):
        self.f = examples.ellipses
        self.x0 = x0
        self.assertTrue(self.solve_and_plot())

    def test_rotated_ellipses(self):
        self.f = examples.rotated_ellipses
        self.x0 = x0
        self.assertTrue(self.solve_and_plot())


if __name__ == '__main__':
    unittest.main()