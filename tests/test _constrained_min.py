import unittest

import numpy as np

from src.constrained_min import InteriorPointSolver
from src.utils import plot_function_and_path, plot_objective_vs_iterations, print_table
from tests.examples import get_qp_params, get_lp_params


class TestUnconstrained(unittest.TestCase):
    def setUp(self):
        self.Solver = InteriorPointSolver()
        np.set_printoptions(precision=5, suppress=True)

    def test_qp(self):
        params = get_qp_params()
        results = self.Solver.solve(**params)
        print(f"x: {results['x']}\n")
        self.create_plots(results=results,
                          func=params['func'],
                          ineq_constraints=params['ineq_constraints'],
                          A=params['eq_constraints_mat'],
                          b=params['eq_constraints_rhs'],
                          history=results['history'])

    def test_lp(self):
        params = get_lp_params()
        results = self.Solver.solve(**params)
        print(f"x: {results['x']}\n")
        self.create_plots(results=results,
                          func=params['func'],
                          ineq_constraints=params['ineq_constraints'],
                          history=results['history'],
                          limits=[[0,2],[0,1]])

    def create_plots(self, results, func, ineq_constraints=None, A=None, b=None, history = None, limits=None):
        plot_function_and_path(func, ineq_constraints, A, b, history, limits)
        plot_objective_vs_iterations(func, history)
        print_table(results, ineq_constraints, A, b)

if __name__ == '__main__':
    unittest.main()