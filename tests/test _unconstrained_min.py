import unittest

import numpy as np

from src.constrained_min import Newton
from src.unconstrained_min import GD
from src.utils import plot_function_and_paths, plot_objective_vs_iterations, print_table
from tests import examples

x0 = np.array([1, 1])
obj_tol = 1e-8
param_tol = 1e-12
max_iter = 100
wolfe_const = 0.01
backtracking_const = 0.5

class TestUnconstrained(unittest.TestCase):
    def setUp(self):
        self.GD = GD(obj_tol, param_tol, wolfe_const, backtracking_const)
        self.Newton = Newton(obj_tol, param_tol, wolfe_const, backtracking_const)

    def solve_and_plot(self):
        gd = self.GD.solve(self.f, self.x0, self.max_iter)
        nt = self.Newton.solve(self.f, self.x0, self.max_iter)
        plot_function_and_paths([self.GD, self.Newton], self.f)
        plot_objective_vs_iterations([self.GD, self.Newton], self.f)
        print_table([gd, nt])

        return True

    def test_circle(self):
        self.f = examples.circle
        self.x0 = x0
        self.max_iter = max_iter
        self.assertTrue(self.solve_and_plot())

    def test_ellipses(self):
        self.f = examples.ellipses
        self.x0 = x0
        self.max_iter = max_iter
        self.assertTrue(self.solve_and_plot())

    def test_rotated_ellipses(self):
        self.f = examples.rotated_ellipses
        self.x0 = x0
        self.max_iter = max_iter
        self.assertTrue(self.solve_and_plot())

    def test_rosenbrock(self):
        self.f = examples.Rosenbrock()
        self.x0 = np.array([-1, 2])
        self.max_iter = 10000
        self.assertTrue(self.solve_and_plot())

    def test_sum_of_exponents(self):
        self.f = examples.SumOfExponents()
        self.x0 = x0
        self.max_iter = max_iter
        self.assertTrue(self.solve_and_plot())

    def test_linear(self):
        self.f = examples.linear
        self.x0 = x0
        self.max_iter = max_iter
        self.assertTrue(self.solve_and_plot())


if __name__ == '__main__':
    unittest.main()