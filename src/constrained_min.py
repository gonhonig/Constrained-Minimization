import numpy as np

from src.common import Function
from src.unconstrained_min import Solver


class Newton(Solver):
    def __init__(self, obj_tol = 1e-8, param_tol = 1e-12, wolfe_const = 0.01, backtracking_const = 0.5):
        super().__init__(obj_tol, param_tol, wolfe_const, backtracking_const)
        self.A = None
        self.b = None

    def solve(self, f: Function, x0, max_iter = 100, A = None, b = None):
        if A is not None:
            self.A = np.asarray(A)
            if self.A.ndim == 1:
                self.A = self.A.reshape(1, -1)

            self.b = np.asarray(b) if b is not None else (None if A is None else np.zeros(self.A.shape[0]))

        return super().solve(f, x0, max_iter)

    def next_direction(self, x, y, g, h):
        if np.all(h == 0):
            return None

        if self.A is None:
            return np.linalg.solve(h, -g)

        n, m = self.A.shape
        lhs = np.block([[h, self.A.T],
                        [self.A, np.zeros((n, n))]])
        rhs = np.concatenate((-g, np.zeros(n)))

        return np.linalg.solve(lhs, rhs)[:m]

    def should_terminate(self, x, x_next, y, g, h, p):
        return 0.5 * p.T @ h @ p < self.obj_tol



class LogBarrierFunction(Function):
    def __init__(self, f: Function, ineq_constraints: list[Function]):
        super().__init__(LogBarrierFunction.__name__, f.dim)
        self.f = f
        self.ineq_constraints = ineq_constraints
        self.t = 1
        self.x = None

    def eval(self, x):
        y_ineq = np.array([ineq.y(x) for ineq in self.ineq_constraints])
        g_ineq = np.array([ineq.g(x) for ineq in self.ineq_constraints]) / -y_ineq
        h_ineq = np.array([ineq.h(x) for ineq in self.ineq_constraints])
        h_ineq = (g_ineq @ g_ineq.T) / (y_ineq ** 2) + np.sum(h_ineq / -y_ineq)
        f_y, f_g, f_h = self.f.eval(x)
        y = self.t * f_y + np.sum(y_ineq)
        g = self.t * f_g + np.sum(g_ineq)
        h = self.t * f_h + np.sum(h_ineq)

        return y, g, h

    def y(self, x):
        return eval(x)[0]

    def g(self, x):
        return eval(x)[1]

    def h(self, x):
        return eval(x)[2]

    def set_t(self, t):
        self.t = t



class InteriorPointSolver:
    def __init__(self, mu = 10, epsilon = 1e-10):
        self.mu = mu
        self.epsilon = epsilon

    def solve(self, func: Function, ineq_constraints: list[Function], eq_constraints_mat, eq_constraints_rhs, x0):
        t = 1
        m = len(ineq_constraints)
        f = LogBarrierFunction(func, ineq_constraints)
        newton = Newton()
        x = x0
        A = eq_constraints_mat
        b = eq_constraints_rhs
        while m / t >= self.epsilon:
            f.set_t(t)
            _, _, next_x, _, success, is_valid = newton.solve(f=f, x0=x, A=A, b=b)
            t *= self.mu
            x = next_x

        return x

