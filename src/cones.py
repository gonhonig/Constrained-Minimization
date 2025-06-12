import numpy as np

from src.function import Function
from src.utils import parse_affine_vars


class GenericSOC(Function):
    def __init__(self, A, b = None, c = None, d = None):
        super().__init__(dim=2)
        self.A, self.b, self.c, self.d = parse_affine_vars(A, b, c, d)
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


class SOC(Function):
    def __init__(self, t, x):
        super().__init__(dim=2)
        self.t = t
        self.x = x
        n = np.prod(x.shape) + 1
        A = np.eye(n)
        A[-1,-1] = 0
        c = np.zeros(n)
        c[-1] = 1
        self.generic_soc = GenericSOC(A=A, c=c)

    def eval(self, x):
        t = self.t(x)
        x = self.x(x).ravel()
        y, g, h = self.generic_soc.eval(np.hstack((x, t)))

        return y, g, h


