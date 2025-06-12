import numpy as np

from src.common import Function, affine_vars


class SOC(Function):
    def __init__(self, A, b = None, c = None, d = None):
        super().__init__(dim=2)
        self.A, self.b, self.c, self.d = affine_vars(A, b, c, d)
        self.m, self.n = A.shape


    def eval(self, x):
        x = np.asarray(x)
        quad_part = self.A @ x + self.b
        linear_part = self.c @ x + self.d
        norm_quad = np.linalg.norm(quad_part, 2)
        grad_quad = self.A.T @ (quad_part / norm_quad) if norm_quad < 1e-12 else np.zeros(self.n)
        I = np.eye(self.m)
        outer_prod = np.outer(quad_part, quad_part)
        hess_matrix = (I / norm_quad - outer_prod / (norm_quad ** 3)) if norm_quad < 1e-12 else np.zeros((self.n, self.n))

        y = norm_quad - linear_part
        g = grad_quad - self.c
        h = self.A.T @ hess_matrix @ self.A

        return y, g, h