import numpy as np

from src.function import *


class Rosenbrock(Function):
    def __init__(self):
        super().__init__(name=Rosenbrock.__name__, dim=2)

    def eval_impl(self, x):
        x = np.asarray(x)
        y = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
        g = np.array([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]),
                         200 * (x[1] - x[0] ** 2)])
        h = np.array([[1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]],
                         [-400 * x[0], 200]])
        return y, g, h


class SumOfExponents(Function):
    def __init__(self):
        super().__init__(name="Sum Of Exponents", dim=2)

    def get_exponents(self, x):
        x1, x2 = x
        A = np.exp(x1 + 3 * x2 - 0.1)
        B = np.exp(x1 - 3 * x2 - 0.1)
        C = np.exp(-x1 - 0.1)

        return A, B, C

    def eval_impl(self, x):
        A, B, C = self.get_exponents(x)
        df_dx1 = A + B - C
        df_dx2 = 3 * A - 3 * B
        d2f_dx1dx1 = A + B + C
        d2f_dx1dx2 = 3 * A - 3 * B
        d2f_dx2dx2 = 9 * A + 9 * B

        y = A + B + C
        g = np.array([df_dx1, df_dx2])
        h = np.array([
            [d2f_dx1dx1, d2f_dx1dx2],
            [d2f_dx1dx2, d2f_dx2dx2]
        ])

        return y, g, h


A = np.array([[np.sqrt(3) / 2, -0.5], [0.5, np.sqrt(3) / 2]])
B = np.array([[100, 0], [0, 1]])

circle = Quadratic([[1, 0],[0, 1]], name="Circle")
ellipses = Quadratic([[1, 0],[0, 100]], name="Ellipses")
rotated_ellipses = Quadratic(A.T @ B @ A, name="Rotated Ellipses")
linear = Linear([-1,2], name="Linear")