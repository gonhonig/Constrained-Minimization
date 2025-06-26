from typing import Callable

import numpy as np
from tqdm import tqdm

from src.function import Function, LogBarrierFunction
from src.utils import parse_affine_vars


class Newton:
    def __init__(self, obj_tol = 1e-8, param_tol = 1e-12, wolfe_const = 0.01, backtracking_const = 0.5, ineq_constraints: list[Function]=None, A=None, b=None):
        self.f = None
        self.obj_tol = obj_tol
        self.param_tol = param_tol
        self.wolfe_const = wolfe_const
        self.backtracking_const = backtracking_const
        self.history = None
        self.success = False
        self.is_valid = True
        self.ineq_constraints = ineq_constraints
        self.A, self.b, _, _ = parse_affine_vars(A, b)
        self.lhs = None
        self.rhs = None

        if self.A is not None and self.b is not None:
            m, n = self.A.shape
            self.lhs = np.block([[np.zeros((n, n)), self.A.T],
                                 [self.A, np.zeros((m, m))]])
            self.rhs = np.zeros(m+n)

    def backtrack_step_size(self, f: Function, x, p):
        alpha = 1
        y, g, _ = f.eval(x)
        max_iter = 50
        min_step = 1e-12

        while alpha >= min_step and max_iter > 0:
            next_x = x + alpha * p
            y_next, _, _ = f.eval(next_x)
            if y_next <= y + self.wolfe_const * alpha * g.T @ p and check_feasibility(next_x, self.ineq_constraints, self.A, self.b):
                break
            alpha *= self.backtracking_const
            max_iter -= 1

        return alpha

    def next_direction(self, g, h):
        if np.allclose(h, 0):
            return None

        if self.lhs is None:
            try:
                return np.linalg.solve(h, -g)
            except np.linalg.LinAlgError:
                return np.linalg.lstsq(h, -g, rcond=None)[0]


        m, n = self.A.shape
        self.lhs[:n, :n] = h
        self.rhs[:n] = -g
        sol = solve_linear_system(self.lhs, self.rhs)

        return sol[:n]


    def solve(self, f, x0, outer_iter, verbose):
        """Newton solver for inner iterations"""
        x = x0.copy()
        max_inner_iter = 100

        for i in tqdm(range(max_inner_iter), desc=f"[Inner {outer_iter:>2}]", disable=(not verbose)):
            y, g, h = f.eval(x)

            if np.linalg.norm(g) < 1e-8:
                break

            p = self.next_direction(g, h)
            if p is None:
                print(f"  [Inner {i}] Failed to find next newton step, stopping")
                break

            alpha = self.backtrack_step_size(f, x, p)
            if alpha < 1e-16:
                print(f"  [Inner {i}] Line search failed, stopping")
                break

            x_new = x + alpha * p

            if np.linalg.norm(x_new - x) < 1e-12 or self.should_terminate(h, p):
                x = x_new
                break

            x = x_new

        return {'x': x}

    def should_terminate(self, h, p):
        return 0.5 * p.T @ h @ p < self.obj_tol


class InteriorPointSolver:
    def __init__(self, mu=10, epsilon=1e-10):
        self.mu = mu
        self.epsilon = epsilon

    def solve(self, func: Function, x0: np.ndarray, ineq_constraints: list[Function] = None,
              eq_constraints_mat=None, eq_constraints_rhs=None, mode="min", verbose=True):
        if verbose:
            print("Interior point solver started")
            print("+++++++++++++++++++++++++++++")

        t = 1
        m = len(ineq_constraints) if ineq_constraints else 0
        func = -func if mode == "max" else func
        f = LogBarrierFunction(func, ineq_constraints)
        A, b, _, _ = parse_affine_vars(eq_constraints_mat, eq_constraints_rhs)
        newton = Newton(ineq_constraints=ineq_constraints, A=A, b=b)
        x = np.asarray(x0)

        i = 1
        history = []
        if verbose:
            print()

        while i == 1 or m / t >= self.epsilon:
            f.set_t(t)
            y, _, _ = func.eval(x)
            y = -y if mode == "max" else y
            history.append(np.append(x, y))
            if verbose:
                print(f"[Outer {i:>2}]: y: {y:.4f}, t: {t:d}")

            if not check_feasibility(x, ineq_constraints, A, b):
                print(f"Error: feasibility check failed.")
                break

            result = newton.solve(f, x, i, verbose=verbose)
            x_new = result['x']

            if np.linalg.norm(x_new - x) < 1e-12:
                break

            x = x_new
            t *= self.mu
            i += 1

        y, _, _ = func.eval(x)
        y = -y if mode == "max" else y
        history.append(np.append(x, y))
        if verbose:
            print(f"Done! y: {y:.4f}")
            print("+++++++++++++++++++++++++++++")

        return {
            'x': x,
            'y': y,
            'iterations': i,
            'history': history
        }


def check_feasibility(x, ineq_constraints, A, b):
    if A is not None and not np.isclose(A @ x, b):
        return False

    if ineq_constraints:
        constraint_values = np.array([ineq.eval(x)[0] for ineq in ineq_constraints])
        return all(constraint_values < 0)

    return True


def solve_linear_system(lhs, rhs):
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]