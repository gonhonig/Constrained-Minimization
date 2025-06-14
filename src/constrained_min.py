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

    def next_step_size(self, x, p):
        alpha = 1
        y, g, _ = self.f.eval(x)
        max_iter = 50
        min_step = 1e-12

        while alpha >= min_step and max_iter > 0:
            y_next, _, _ = self.f.eval(x + alpha * p)
            if y_next <= y + self.wolfe_const * alpha * g.T @ p:
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
              eq_constraints_mat=None, eq_constraints_rhs=None, custom_break: Callable = None):
        print("Interior point solver started")
        print("+++++++++++++++++++++++++++++")

        t = 1
        m = len(ineq_constraints) if ineq_constraints else 0
        f = LogBarrierFunction(func, ineq_constraints)
        A, b, _, _ = parse_affine_vars(eq_constraints_mat, eq_constraints_rhs)
        newton = Newton(A=A, b=b)

        if isinstance(x0, int):
            x = self._find_x0_optimized(x0, ineq_constraints, A, b)
        else:
            x = x0.ravel()

        i = 1
        history = []

        while i == 1 or m / t >= self.epsilon:
            f.set_t(t)
            y, _, _ = func.eval(x)
            history.append(np.append(x, y))
            print(f"[Outer {i:>2}]: y: {y:.4f}, t: {t:d}")

            # Check feasibility before solving
            if ineq_constraints:
                constraint_values = []
                for ineq in ineq_constraints:
                    val = ineq.eval(x)[0]
                    # Handle both scalar and array outputs
                    if isinstance(val, np.ndarray):
                        val = val.item() if val.size == 1 else np.max(val)
                    constraint_values.append(val)
                max_violation = max(constraint_values)
                if max_violation >= 0:
                    print(f"Warning: Starting point violates constraints by {max_violation}")

            # Solve inner problem with modified Newton solver
            result = self._solve_inner_with_feasibility(f, x, A, b, ineq_constraints, newton, i)
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

    def _solve_inner_with_feasibility(self, f, x0, A, b, ineq_constraints, newton, i):
        """Modified Newton solver that maintains feasibility"""
        x = x0.copy()
        max_inner_iter = 100

        for iter in tqdm(range(max_inner_iter), desc=f"[Inner {i:>2}]"):
            y, g, h = f.eval(x)

            # Check if we've converged
            if np.linalg.norm(g) < 1e-8:
                break

            # Get Newton direction
            p = newton.next_direction(x, y, g, h)
            if p is None:
                break

            # Line search with feasibility check
            alpha = self._feasible_line_search(x, p, f, ineq_constraints, g)

            if alpha < 1e-16:
                print(f"  [Inner {iter}] Line search failed, stopping")
                break

            x_new = x + alpha * p

            # Check termination
            if np.linalg.norm(x_new - x) < 1e-12 or newton.should_terminate(x, x_new, y, g, h, p):
                x = x_new
                break

            x = x_new

        return {'x': x}

    def _feasible_line_search(self, x, p, f, ineq_constraints, g):
        """Line search that maintains strict feasibility"""
        alpha = 1.0
        beta = 0.5
        c1 = 0.01  # Armijo condition constant

        # First, find maximum feasible step
        if ineq_constraints:
            alpha_max = self._max_feasible_step(x, p, ineq_constraints)
            alpha = min(alpha, 0.99 * alpha_max)  # Stay strictly inside

        # Standard backtracking line search with Armijo condition
        y, _, _ = f.eval(x)

        while alpha > 1e-16:
            x_new = x + alpha * p

            # Check feasibility
            if ineq_constraints:
                feasible = True
                for ineq in ineq_constraints:
                    val = ineq.eval(x_new)[0]
                    if isinstance(val, np.ndarray):
                        val = np.max(val)
                    if val >= 0:
                        feasible = False
                        break

                if not feasible:
                    alpha *= beta
                    continue

            # Check Armijo condition
            y_new, _, _ = f.eval(x_new)
            if y_new <= y + c1 * alpha * g.T @ p:
                return alpha

            alpha *= beta

        return alpha

    def _max_feasible_step(self, x, p, ineq_constraints):
        """Find maximum step size that maintains feasibility"""
        alpha_max = 1.0

        for ineq in ineq_constraints:
            y_ineq, g_ineq, _ = ineq.eval(x)

            # We need: ineq(x + alpha*p) < 0
            # Linear approximation: y_ineq + alpha * g_ineq^T p < 0
            gp = np.dot(g_ineq, p)

            if gp >= 0:  # Moving away from constraint
                continue

            # Maximum alpha such that y_ineq + alpha * gp < 0
            alpha_i = -y_ineq / gp
            alpha_max = min(alpha_max, alpha_i)

        return alpha_max

    def _find_x0_optimized(self, length, ineq_constraints, A, b):
        print("\nFinding initial point")

        # Start with a reasonable guess
        if A is None:
            x0 = np.ones(length) * 0.5  # Start in middle of [0,1] range
        else:
            x0 = np.linalg.lstsq(A, b, rcond=None)[0]
            if x0.size < length:
                # Pad with reasonable values
                pad_value = 0.1  # Small positive value for auxiliary variables
                x0 = np.pad(x0, (0, length - x0.size), constant_values=pad_value)

        if ineq_constraints:
            # Check if initial point satisfies constraints
            constraint_values = []
            for ineq in ineq_constraints:
                val = ineq.eval(x0)[0]
                # Handle both scalar and array outputs
                if isinstance(val, np.ndarray):
                    val = val.item() if val.size == 1 else np.max(val)
                constraint_values.append(val)
            all_satisfied = all(val < 0 for val in constraint_values)

            if all_satisfied:
                print("Initial point satisfies all constraints")
                return x0

            print("Finding feasible initial point using phase-1 method...")

            # For SOCP problems, we need a smarter initialization
            # Try to find a point that satisfies most constraints
            x0 = self._smart_initialization(length, ineq_constraints)

            # Check again
            constraint_values = []
            for ineq in ineq_constraints:
                val = ineq.eval(x0)[0]
                if isinstance(val, np.ndarray):
                    val = val.item() if val.size == 1 else np.max(val)
                constraint_values.append(val)

            if all(val < 0 for val in constraint_values):
                print("Smart initialization found feasible point")
                return x0

            # If still not feasible, use phase-1 method
            max_violation = max(0, max(constraint_values))

            # Extend x0 with slack variable
            # Initialize slack to be larger than max violation
            x0_extended = np.append(x0, max_violation + 1.0)

            # Create extended constraints: original_constraint(x) - s <= 0
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

            custom_break = lambda x: x[-1] < 1e-6

            result = self.solve(func=f, x0=x0_extended,
                                ineq_constraints=ineq_constraints_with_s,
                                eq_constraints_mat=A_extended,
                                eq_constraints_rhs=b,
                                custom_break=custom_break)

            s_val = result['x'][-1]

            if s_val > 1e-6:
                raise ValueError(f"The problem is infeasible (slack = {s_val})")

            return result['x'][:-1]

        return x0

    def _smart_initialization(self, length, ineq_constraints):
        """Try to find a good initial point for SOCP constraints"""
        # For image denoising, we know the structure:
        # - First num_pixels variables are pixel values (should be in [0,1])
        # - Next variables are TV auxiliary variables (should be small positive)
        # - Last variable is data fidelity (should be moderate)

        x0 = np.zeros(length)

        # Guess the number of pixels (assuming square image)
        # This is a heuristic based on the problem structure
        num_constraints = len(ineq_constraints)

        # For a m×n image:
        # - (m-1)×n vertical TV constraints
        # - m×(n-1) horizontal TV constraints
        # - 1 data fidelity constraint
        # Total TV constraints ≈ 2mn - m - n

        # Rough estimate: if we have k constraints, pixels ≈ k/2
        estimated_pixels = int(np.sqrt(num_constraints / 2))
        num_pixels = estimated_pixels * estimated_pixels

        if num_pixels > length / 2:
            # Fallback to simpler estimate
            num_pixels = int(length / 3)

        # Initialize pixel values to middle range
        x0[:num_pixels] = 0.5

        # Initialize TV auxiliary variables
        num_tv_vars = num_constraints - 1  # Subtract data fidelity constraint
        if num_pixels + num_tv_vars < length:
            x0[num_pixels:num_pixels + num_tv_vars] = 0.1

        # Initialize data fidelity variable
        if length > num_pixels + num_tv_vars:
            x0[-1] = 0.5

        return x0