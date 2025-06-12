import numpy as np

from src.common import Function, affine_vars
from src.unconstrained_min import Solver


class Newton(Solver):
    def __init__(self, obj_tol = 1e-8, param_tol = 1e-12, wolfe_const = 0.01, backtracking_const = 0.5):
        super().__init__(obj_tol, param_tol, wolfe_const, backtracking_const)
        self.A = None
        self.b = None

    def solve(self, f: Function, x0, max_iter = 100, A = None, b = None, verbose = True):
        self.A, self.b, _, _ = affine_vars(A, b)
        return super().solve(f, x0, max_iter, verbose)

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
        y, g, h = self.f.eval(x)

        if self.ineq_constraints:
            eval_ineq = [ineq.eval(x) for ineq in self.ineq_constraints]
            y_ineq = np.array([eval[0] for eval in eval_ineq])
            g_ineq = np.array([eval[1] for eval in eval_ineq])
            h_ineq = np.array([eval[2] for eval in eval_ineq])
            h_ineq = np.array([np.outer(g_i, g_i) for g_i in g_ineq]) / (y_ineq ** 2)[:,None,None] + (h_ineq / -y_ineq[:,None,None])
            y = self.t * y - np.sum(np.log(-y_ineq))
            g = self.t * g + np.sum(g_ineq / -y_ineq[:,None], axis=0)
            h = self.t * h + np.sum(h_ineq, axis=0)

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

    def solve(self, func: Function, x0, ineq_constraints: list[Function] = None, eq_constraints_mat = None, eq_constraints_rhs = None, verbose = True):
        t = 1
        m = len(ineq_constraints) if ineq_constraints else 0
        f = LogBarrierFunction(func, ineq_constraints)
        newton = Newton()
        x = x0
        shape = x.shape
        x = x.ravel()
        A = eq_constraints_mat
        b = eq_constraints_rhs
        i = 1
        history = []
        if verbose:
            print("Solving using interior point method")

        while i == 1 or m / t >= self.epsilon:
            f.set_t(t)
            y, _, _ = func.eval(x)
            history.append(np.append(x, y))
            if verbose:
                print(f"[{i}] x: {x}, y: {y}")
            x_new = newton.solve(f=f, x0=x, A=A, b=b, verbose=False)['x']
            if np.linalg.norm(x_new - x) < 1e-12:
                break
            x = x_new
            t *= self.mu
            i += 1

        y, _, _ = func.eval(x)
        history.append(np.append(x, y))
        if verbose:
            print()

        x = x.reshape(shape)

        return {
            'x': x,
            'f': y,
            'iterations': i,
            'history': history
        }

