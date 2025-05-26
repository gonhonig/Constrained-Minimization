import numpy as np

from src.common import Function


class Quadratic(Function):
    def __init__(self, Q, name):
        super().__init__(name)
        self.Q = np.asarray(Q)

    def y(self, x):
        return x.T @ self.Q @ x

    def g(self, x):
        return 2 * self.Q @ x

    def h(self, x):
        return 2 * self.Q


circle = Quadratic([[1, 0],[0, 1]], "Circle")
ellipses = Quadratic([[1, 0],[0, 100]], "Ellipses")

A = np.array([[np.sqrt(3) / 2, -0.5], [0.5, np.sqrt(3) / 2]])
B = np.array([[100, 0], [0, 1]])

rotated_ellipses = Quadratic(A.T @ B @ A, "Rotated Ellipses")