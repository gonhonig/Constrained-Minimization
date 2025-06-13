from typing import Sequence, Callable

import numpy as np

from src.function import Function, Linear
from src.unconstrained_min import Solver
from src.utils import parse_affine_vars


class Newton(Solver):
    def __init__(self, obj_tol = 1e-8, param_tol = 1e-12, wolfe_const = 0.01, backtracking_const = 0.5):
        super().__init__(obj_tol, param_tol, wolfe_const, backtracking_const)
        self.A = None
        self.b = None

    def solve(self, f: Function, x0, max_iter = 100, A = None, b = None, verbose = True):
        self.A, self.b, _, _ = parse_affine_vars(A, b)
        return super().solve(f, x0, max_iter, verbose)

    def next_direction(self, x, y, g, h):
        if np.all(h == 0):
            return None

        if self.A is None:
            return np.linalg.lstsq(h, -g, rcond=None)[0]

        n, m = self.A.shape
        lhs = np.block([[h, self.A.T],
                        [self.A, np.zeros((n, n))]])
        rhs = np.concatenate((-g, np.zeros(n)))
        x = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

        return x[:m]

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

    def pad(self, pad_width, constant_values=0):
        return self


class InteriorPointSolver:
    def __init__(self, mu = 10, epsilon = 1e-10):
        self.mu = mu
        self.epsilon = epsilon

    def solve(self, func: Function, x0: int|Sequence|np.ndarray, ineq_constraints: list[Function] = None, eq_constraints_mat = None, eq_constraints_rhs = None, custom_break: Callable = None):
        print("Interior point solver started")
        print("+++++++++++++++++++++++++++++")
        t = 1
        m = len(ineq_constraints) if ineq_constraints else 0
        f = LogBarrierFunction(func, ineq_constraints)
        newton = Newton()
        A = eq_constraints_mat
        b = eq_constraints_rhs

        if isinstance(x0, int):
            x = self.find_x0(x0, ineq_constraints, A, b)
        else:
            x = np.asarray(x0)
            x = x.ravel()

        i = 1
        history = []

        while i == 1 or m / t >= self.epsilon:
            newton.name = f"Inner {i:>2}"
            f.set_t(t)
            y, _, _ = func.eval(x)
            history.append(np.append(x, y))
            print(f"[Outer {i:>2}] y: {y:.4f}")
            x_new = newton.solve(f=f, x0=x, A=A, b=b, verbose=False)['x']
            print()
            if np.linalg.norm(x_new - x) < 1e-12 or (custom_break is not None and custom_break(x)):
                break
            x = x_new
            t *= self.mu
            i += 1

        y, _, _ = func.eval(x)
        history.append(np.append(x, y))
        print("Done!")
        print("+++++++++++++++++++++++++++++")

        return {
            'x': x,
            'y': y,
            'iterations': i,
            'history': history
        }

    def find_x0(self, length, ineq_constraints, A, b):
        print("\nFinding initial point")
        A, b, _, _ = parse_affine_vars(A, b)

        if A is None:
            x0 = np.random.rand(length)
        else:
            x0 = np.linalg.lstsq(A, b, rcond=None)[0]
            A = np.pad(A, (0,1), 'constant')
            b = np.pad(b, (0,1), 'constant')

        if ineq_constraints:
            all_satisfied = all(ineq.eval(x0)[0] < 0 for ineq in ineq_constraints)
            if all_satisfied:
                return x0

        max_violation = np.max([ineq.eval(x0)[0] for ineq in ineq_constraints])
        x0 = np.append(x0, max_violation + 1)
        s = np.zeros_like(x0)
        s[-1] = 1
        f = Linear(s)
        ineq_constraints_with_s = [constraint.pad((0,1)) - f for constraint in ineq_constraints]
        custom_break = lambda x: x[-1] < 0
        result = self.solve(func=f, x0=x0, ineq_constraints=ineq_constraints_with_s, eq_constraints_mat=A, eq_constraints_rhs=b, verbose=False, custom_break=custom_break)
        s = result['x'][-1]

        if s > 0:
            raise ValueError("The problem is infeasible")

        return result['x'][:-1]
