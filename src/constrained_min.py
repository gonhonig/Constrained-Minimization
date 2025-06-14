from typing import Sequence, Callable

import numpy as np
from tqdm import tqdm

from src.function import Function, Linear
from src.unconstrained_min import Solver
from src.utils import parse_affine_vars


class Newton:
    def __init__(self, obj_tol = 1e-8, param_tol = 1e-12, wolfe_const = 0.01, backtracking_const = 0.5, A = None, b = None):
        self.f = None
        self.obj_tol = obj_tol
        self.param_tol = param_tol
        self.wolfe_const = wolfe_const
        self.backtracking_const = backtracking_const
        self.history = None
        self.success = False
        self.is_valid = True
        self.A, self.b, _, _ = parse_affine_vars(A, b)

    def backtrack_step_size(self, f: Function, x, p, extra_condition: Callable = None):
        alpha = 1
        y, g, _ = f.eval(x)
        max_iter = 50
        min_step = 1e-12
        extra_condition = extra_condition if extra_condition is not None else lambda c: True

        while alpha >= min_step and max_iter > 0:
            next_x = x + alpha * p
            y_next, _, _ = f.eval(next_x)
            if y_next <= y + self.wolfe_const * alpha * g.T @ p and extra_condition(next_x):
                break
            alpha *= self.backtracking_const
            max_iter -= 1

        return alpha

    def next_direction(self, x, y, g, h):
        if np.allclose(h, 0):
            return None

        if self.A is None:
            try:
                return np.linalg.solve(h, -g)
            except np.linalg.LinAlgError:
                return np.linalg.lstsq(h, -g, rcond=None)[0]

        n, m = self.A.shape
        lhs = np.block([[h, self.A.T],
                        [self.A, np.zeros((n, n))]])
        rhs = np.concatenate((-g, np.zeros(n)))

        try:
            sol = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

        return sol[:m]

    def should_terminate(self, x, x_next, y, g, h, p):
        return 0.5 * p.T @ h @ p < self.obj_tol


class LogBarrierFunction(Function):
    def __init__(self, f: Function, ineq_constraints: list[Function]):
        super().__init__(LogBarrierFunction.__name__, f.dim)
        self.f = f
        self.ineq_constraints = ineq_constraints
        self.t = 1
        self.x = None

        # Cache for avoiding repeated evaluations
        self._ineq_cache = [None] * len(ineq_constraints) if ineq_constraints else []

    def eval(self, x):
        y, g, h = self.f.eval(x)

        if self.ineq_constraints:
            # Evaluate all constraints and cache results
            ineq_evals = []
            for i, ineq in enumerate(self.ineq_constraints):
                eval_result = ineq.eval(x)
                ineq_evals.append(eval_result)
                self._ineq_cache[i] = eval_result

            y_ineq = np.array([eval_result[0] for eval_result in ineq_evals])
            g_ineq = np.array([eval_result[1] for eval_result in ineq_evals])
            h_ineq = np.array([eval_result[2] for eval_result in ineq_evals])

            # Check for constraint violations
            if np.any(y_ineq >= 0):
                # Return large but finite values instead of inf
                large_val = 1e10
                return large_val, np.full_like(g, large_val), np.full_like(h, large_val)

            # Compute barrier terms more efficiently
            log_terms = np.log(-y_ineq)
            inv_y_ineq = 1.0 / y_ineq
            inv_y_ineq_sq = inv_y_ineq ** 2

            # Vectorized barrier function computation
            barrier_val = -np.sum(log_terms)

            # Gradient: sum of g_i / (-y_i)
            barrier_grad = np.sum(g_ineq * inv_y_ineq.reshape(-1, 1), axis=0)

            # Hessian: sum of (g_i * g_i^T) / y_i^2 + h_i / (-y_i)
            barrier_hess = np.zeros_like(h)
            for i in range(len(ineq_evals)):
                barrier_hess += (np.outer(g_ineq[i], g_ineq[i]) * inv_y_ineq_sq[i] +
                                 h_ineq[i] * (-inv_y_ineq[i]))

            # Combine original function with barrier
            y = self.t * y + barrier_val
            g = self.t * g + barrier_grad
            h = self.t * h + barrier_hess

        return y, g, h

    def set_t(self, t):
        self.t = t

    def pad(self, pad_width, constant_values=0):
        return self


class InteriorPointSolver:
    def __init__(self, mu=10, epsilon=1e-10):
        self.mu = mu
        self.epsilon = epsilon

    def solve(self, func: Function, x0: int | Sequence | np.ndarray, ineq_constraints: list[Function] = None,
              eq_constraints_mat=None, eq_constraints_rhs=None, custom_break: Callable = None, verbose=True):
        if verbose:
            print("Interior point solver started")
            print("+++++++++++++++++++++++++++++")

        t = 1
        m = len(ineq_constraints) if ineq_constraints else 0
        f = LogBarrierFunction(func, ineq_constraints)
        A, b, _, _ = parse_affine_vars(eq_constraints_mat, eq_constraints_rhs)
        newton = Newton(A=A, b=b)

        if isinstance(x0, int):
            x = self.find_initial_point(x0, ineq_constraints, A, b)
        else:
            x = np.asarray(x0).ravel()

        i = 1
        history = []
        if verbose:
            print()

        while i == 1 or m / t >= self.epsilon:
            f.set_t(t)
            y, _, _ = func.eval(x)
            history.append(np.append(x, y))
            if verbose:
                print(f"[Outer {i:>2}]: y: {y:.4f}, t: {t:d}")

            if not self.feasibility_check(x, ineq_constraints, A, b):
                print(f"Error: feasibility check failed.")
                break

            result = self.solve_inner(f, x, ineq_constraints, newton, i, verbose=verbose)
            x_new = result['x']

            if verbose:
                print()
            if np.linalg.norm(x_new - x) < 1e-12 or (custom_break is not None and custom_break(x)):
                break

            x = x_new
            t *= self.mu
            i += 1

        y, _, _ = func.eval(x)
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

    def solve_inner(self, f, x0, ineq_constraints, newton, outer_iter, verbose):
        """Newton solver for inner iterations"""
        x = x0.copy()
        max_inner_iter = 100

        for i in tqdm(range(max_inner_iter), desc=f"[Inner {outer_iter:>2}]", disable=not verbose):
            y, g, h = f.eval(x)

            if np.linalg.norm(g) < 1e-8:
                break

            p = newton.next_direction(x, y, g, h)
            if p is None:
                print(f"  [Inner {i}] Failed to find next newton step, stopping")
                break

            extra_condition = lambda c: self.feasibility_check(c, ineq_constraints, newton.A, newton.b)
            alpha = newton.backtrack_step_size(f, x, p, extra_condition)
            if alpha < 1e-16:
                print(f"  [Inner {i}] Line search failed, stopping")
                break

            x_new = x + alpha * p

            if np.linalg.norm(x_new - x) < 1e-12 or newton.should_terminate(x, x_new, y, g, h, p):
                x = x_new
                break

            x = x_new

        return {'x': x}

    def find_initial_point(self, length, ineq_constraints, A, b):
        print("\nFinding initial point")

        if A is None:
            x0 = np.ones(length) * 0.5
        else:
            x0 = np.linalg.lstsq(A, b, rcond=None)[0]
            if x0.size < length:
                pad_value = 0.1
                x0 = np.pad(x0, (0, length - x0.size), constant_values=pad_value)
            x0 = np.array([2, -1])

        if not ineq_constraints:
            return x0

        if self.feasibility_check(x0, ineq_constraints, A, b):
            print("Initial point satisfies all constraints")
            return x0

        print("Finding feasible initial point using phase-1 method...")

        constraint_values = np.array([ineq.eval(x0)[0] for ineq in ineq_constraints])
        max_violation = max(0, max(constraint_values))
        x0_extended = np.append(x0, max_violation + 1.0)
        s = np.zeros(length + 1)
        s[-1] = 1
        f = Linear(s)
        ineq_constraints_with_s = [constraint.pad((0, 1)) - f for constraint in ineq_constraints]
        A_extended = np.pad(A, ((0, 0), (0, 1)), 'constant') if A is not None else None
        custom_break = lambda x: x[-1] < 1e-6

        result = self.solve(func=f, x0=x0_extended,
                            ineq_constraints=ineq_constraints_with_s,
                            eq_constraints_mat=A_extended,
                            eq_constraints_rhs=b,
                            custom_break=custom_break,
                            verbose=False)

        s_val = result['x'][-1]

        if s_val > 1e-6:
            raise ValueError(f"The problem is infeasible (slack = {s_val})")
        print("Initial point found")

        return result['x'][:-1]

    @staticmethod
    def feasibility_check(x, ineq_constraints, A, b):
        if A is not None and not np.isclose(A @ x, b):
            return False

        if ineq_constraints:
            constraint_values = np.array([ineq.eval(x)[0] for ineq in ineq_constraints])
            return all(constraint_values < 0)

        return True
