from typing import Sequence, Callable

import numpy as np

from src.function import Function, Linear
from src.unconstrained_min import Solver
from src.utils import parse_affine_vars


class Newton(Solver):
    def __init__(self, obj_tol=1e-8, param_tol=1e-12, wolfe_const=0.01, backtracking_const=0.5):
        super().__init__(obj_tol, param_tol, wolfe_const, backtracking_const)
        self.A = None
        self.b = None

    def solve(self, f: Function, x0, max_iter=100, A=None, b=None, verbose=True):
        self.A, self.b, _, _ = parse_affine_vars(A, b)
        return super().solve(f, x0, max_iter, verbose)

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
                return np.inf, np.full_like(g, np.inf), np.full_like(h, np.inf)

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
              eq_constraints_mat=None, eq_constraints_rhs=None, custom_break: Callable = None):
        print("Interior point solver started")
        print("+++++++++++++++++++++++++++++")

        t = 1
        m = len(ineq_constraints) if ineq_constraints else 0
        f = LogBarrierFunction(func, ineq_constraints)
        newton = Newton()
        A = eq_constraints_mat
        b = eq_constraints_rhs

        if isinstance(x0, int):
            x = self._find_x0_optimized(x0, ineq_constraints, A, b)
        else:
            x = x0.ravel()

        i = 1
        history = []

        while i == 1 or m / t >= self.epsilon:
            newton.name = f"Inner {i:>2}"
            f.set_t(t)
            y, _, _ = func.eval(x)
            history.append(np.append(x, y))
            print(f"[Outer {i:>2}] y: {y:.4f}, t: {t:.2e}")

            # Solve inner problem
            result = newton.solve(f=f, x0=x, A=A, b=b, verbose=False)
            x_new = result['x']

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

    def _find_x0_optimized(self, length, ineq_constraints, A, b):
        print("\nFinding initial point")
        A, b, _, _ = parse_affine_vars(A, b)

        if A is None:
            x0 = np.random.rand(length)
        else:
            x0 = np.linalg.lstsq(A, b, rcond=None)[0]
            if x0.size < length:
                x0 = np.pad(x0, (0, length - x0.size))

        if ineq_constraints:
            # Check if initial point satisfies constraints
            constraint_values = [ineq.eval(x0)[0] for ineq in ineq_constraints]
            all_satisfied = all(val < 0 for val in constraint_values)

            if all_satisfied:
                print("Initial point satisfies all constraints")
                return x0

            print("Finding feasible initial point...")
            max_violation = max(constraint_values)

            # Extend x0 with slack variable
            x0_extended = np.append(x0, max_violation + 1)

            # Create extended constraints
            s = np.zeros(length + 1)
            s[-1] = 1
            f = Linear(s)

            ineq_constraints_with_s = []
            for constraint in ineq_constraints:
                padded_constraint = constraint.pad((0, 1))
                ineq_constraints_with_s.append(padded_constraint - f)

            # Extend A matrix if needed
            if A is not None:
                A_extended = np.pad(A, ((0, 0), (0, 1)), 'constant')
            else:
                A_extended = None

            custom_break = lambda x: x[-1] < 0

            result = self.solve(func=f, x0=x0_extended,
                                ineq_constraints=ineq_constraints_with_s,
                                eq_constraints_mat=A_extended,
                                eq_constraints_rhs=b,
                                custom_break=custom_break)

            s_val = result['x'][-1]

            if s_val > 0:
                raise ValueError("The problem is infeasible")

            return result['x'][:-1]

        return x0