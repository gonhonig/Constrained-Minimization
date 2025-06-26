import unittest

from src.constrained_min import InteriorPointSolver
from tests.examples import get_qp_params, get_lp_params


class TestUnconstrained(unittest.TestCase):
    def setUp(self):
        self.Solver = InteriorPointSolver()

    def test_qp(self):
        results = self.Solver.solve(**get_qp_params())
        print(f"x: {results['x']}\n")

    def test_lp(self):
        results = self.Solver.solve(**get_lp_params())
        print(f"x: {results['x']}\n")

if __name__ == '__main__':
    unittest.main()