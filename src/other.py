import numpy as np
from scipy.linalg import solve, LinAlgError
from typing import Callable, Optional, Tuple, List
import warnings


class InteriorPointSolver:
    """
    Interior Point Solver for constrained optimization problems using
    Newton's method and log barrier method.

    Solves problems of the form:
        minimize f(x)
        subject to g_i(x) <= 0 for i = 1, ..., m (inequality constraints)
                   h_j(x) = 0 for j = 1, ..., p (equality constraints)
    """

    def __init__(self,
                 objective: Callable,
                 grad_objective: Callable,
                 hess_objective: Callable,
                 inequalities: Optional[List[Callable]] = None,
                 grad_inequalities: Optional[List[Callable]] = None,
                 hess_inequalities: Optional[List[Callable]] = None,
                 equalities: Optional[List[Callable]] = None,
                 grad_equalities: Optional[List[Callable]] = None,
                 hess_equalities: Optional[List[Callable]] = None):
        """
        Initialize the interior point solver.

        Args:
            objective: Objective function f(x)
            grad_objective: Gradient of objective function
            hess_objective: Hessian of objective function
            inequalities: List of inequality constraint functions g_i(x) <= 0
            grad_inequalities: List of gradients of inequality constraints
            hess_inequalities: List of Hessians of inequality constraints
            equalities: List of equality constraint functions h_j(x) = 0
            grad_equalities: List of gradients of equality constraints
            hess_equalities: List of Hessians of equality constraints
        """
        self.f = objective
        self.grad_f = grad_objective
        self.hess_f = hess_objective

        self.g = inequalities or []
        self.grad_g = grad_inequalities or []
        self.hess_g = hess_inequalities or []

        self.h = equalities or []
        self.grad_h = grad_equalities or []
        self.hess_h = hess_equalities or []

        self.m = len(self.g)  # number of inequality constraints
        self.p = len(self.h)  # number of equality constraints

    def barrier_function(self, x: np.ndarray, mu: float) -> float:
        """
        Compute the log barrier function.

        Args:
            x: Current point
            mu: Barrier parameter

        Returns:
            Barrier function value
        """
        barrier_term = 0.0
        for g_i in self.g:
            g_val = g_i(x)
            if g_val >= 0:
                return np.inf  # Infeasible point
            barrier_term -= np.log(-g_val)

        return self.f(x) + mu * barrier_term

    def barrier_gradient(self, x: np.ndarray, mu: float) -> np.ndarray:
        """
        Compute the gradient of the barrier function.

        Args:
            x: Current point
            mu: Barrier parameter

        Returns:
            Gradient of barrier function
        """
        grad = self.grad_f(x).copy()

        for i, (g_i, grad_g_i) in enumerate(zip(self.g, self.grad_g)):
            g_val = g_i(x)
            if g_val >= 0:
                raise ValueError(f"Infeasible point: constraint {i} violated")
            grad -= mu / g_val * grad_g_i(x)

        return grad

    def barrier_hessian(self, x: np.ndarray, mu: float) -> np.ndarray:
        """
        Compute the Hessian of the barrier function.

        Args:
            x: Current point
            mu: Barrier parameter

        Returns:
            Hessian of barrier function
        """
        hess = self.hess_f(x).copy()

        for i, (g_i, grad_g_i, hess_g_i) in enumerate(zip(self.g, self.grad_g, self.hess_g)):
            g_val = g_i(x)
            if g_val >= 0:
                raise ValueError(f"Infeasible point: constraint {i} violated")

            grad_g_val = grad_g_i(x)
            try:
                hess -= mu / g_val * hess_g_i(x)
            except Exception as e:
                print(e)
            hess += mu / (g_val ** 2) * np.outer(grad_g_val, grad_g_val)

        return hess

    def kkt_system(self, x: np.ndarray, lam: np.ndarray, mu: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the KKT system for the barrier subproblem.

        Args:
            x: Current point
            lam: Lagrange multipliers for equality constraints
            mu: Barrier parameter

        Returns:
            KKT matrix and right-hand side vector
        """
        n = len(x)

        # Build the KKT matrix
        if self.p > 0:
            # With equality constraints
            kkt_matrix = np.zeros((n + self.p, n + self.p))

            # Top-left: Hessian of Lagrangian
            hess_lag = self.barrier_hessian(x, mu)
            for j, (h_j, hess_h_j) in enumerate(zip(self.h, self.hess_h)):
                hess_lag += lam[j] * hess_h_j(x)
            kkt_matrix[:n, :n] = hess_lag

            # Top-right and bottom-left: Jacobian of equality constraints
            for j, grad_h_j in enumerate(self.grad_h):
                grad_h_val = grad_h_j(x)
                kkt_matrix[:n, n + j] = grad_h_val
                kkt_matrix[n + j, :n] = grad_h_val

            # Right-hand side
            rhs = np.zeros(n + self.p)
            rhs[:n] = -self.barrier_gradient(x, mu)
            for j, grad_h_j in enumerate(self.grad_h):
                rhs[:n] -= lam[j] * grad_h_j(x)

            for j, h_j in enumerate(self.h):
                rhs[n + j] = -h_j(x)

        else:
            # Without equality constraints
            kkt_matrix = self.barrier_hessian(x, mu)
            rhs = -self.barrier_gradient(x, mu)

        return kkt_matrix, rhs

    def newton_step(self, x: np.ndarray, lam: np.ndarray, mu: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Newton step for the barrier subproblem.

        Args:
            x: Current point
            lam: Current Lagrange multipliers
            mu: Barrier parameter

        Returns:
            Newton step for x and lambda
        """
        kkt_matrix, rhs = self.kkt_system(x, lam, mu)

        try:
            if self.p > 0:
                step = solve(kkt_matrix, rhs)
                dx = step[:len(x)]
                dlam = step[len(x):]
            else:
                dx = solve(kkt_matrix, rhs)
                dlam = np.array([])

            return dx, dlam

        except LinAlgError:
            # Fallback to regularized system
            reg = 1e-8
            if self.p > 0:
                kkt_matrix[:len(x), :len(x)] += reg * np.eye(len(x))
            else:
                kkt_matrix += reg * np.eye(len(x))

            step = solve(kkt_matrix, rhs)
            if self.p > 0:
                dx = step[:len(x)]
                dlam = step[len(x):]
            else:
                dx = step
                dlam = np.array([])

            return dx, dlam

    def line_search(self, x: np.ndarray, dx: np.ndarray, mu: float,
                    alpha_max: float = 1.0, beta: float = 0.5,
                    sigma: float = 1e-4) -> float:
        """
        Backtracking line search for barrier method.

        Args:
            x: Current point
            dx: Search direction
            mu: Barrier parameter
            alpha_max: Maximum step size
            beta: Backtracking parameter
            sigma: Armijo parameter

        Returns:
            Step size
        """
        alpha = alpha_max

        # Ensure we stay in the feasible region
        for g_i in self.g:
            if len(dx) > 0:
                # Find maximum step that keeps constraints satisfied
                g_val = g_i(x)
                if g_val >= 0:
                    return 0.0

                # Compute directional derivative
                try:
                    grad_g_val = self.grad_g[self.g.index(g_i)](x)
                    directional_deriv = np.dot(grad_g_val, dx)
                    if directional_deriv > 0:
                        max_step = -0.99 * g_val / directional_deriv
                        alpha = min(alpha, max_step)
                except:
                    alpha = min(alpha, 0.01)

        # Armijo condition
        f0 = self.barrier_function(x, mu)
        grad0 = self.barrier_gradient(x, mu)
        slope = np.dot(grad0, dx)

        while alpha > 1e-12:
            x_new = x + alpha * dx
            try:
                f_new = self.barrier_function(x_new, mu)
                if f_new <= f0 + sigma * alpha * slope:
                    return alpha
            except:
                pass
            alpha *= beta

        return alpha

    def solve_barrier_subproblem(self, x0: np.ndarray, mu: float,
                                 tol: float = 1e-8, max_iter: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the barrier subproblem for a fixed mu.

        Args:
            x0: Initial point
            mu: Barrier parameter
            tol: Convergence tolerance
            max_iter: Maximum iterations

        Returns:
            Optimal point and Lagrange multipliers
        """
        x = x0.copy()
        lam = np.zeros(self.p)

        for k in range(max_iter):
            try:
                dx, dlam = self.newton_step(x, lam, mu)

                # Check convergence
                if np.linalg.norm(dx) < tol:
                    break

                # Line search
                alpha = self.line_search(x, dx, mu)

                # Update
                x += alpha * dx
                if self.p > 0:
                    lam += alpha * dlam

            except Exception as e:
                warnings.warn(f"Newton step failed at iteration {k}: {e}")
                break

        return x, lam

    def solve(self, x0: np.ndarray, mu0: float = 1.0, mu_factor: float = 0.1,
              tol: float = 1e-6, max_outer_iter: int = 50,
              max_inner_iter: int = 50) -> dict:
        """
        Solve the constrained optimization problem.

        Args:
            x0: Initial point (must be strictly feasible)
            mu0: Initial barrier parameter
            mu_factor: Factor to reduce barrier parameter
            tol: Convergence tolerance
            max_outer_iter: Maximum outer iterations
            max_inner_iter: Maximum inner iterations per subproblem

        Returns:
            Dictionary with solution information
        """
        # Check initial feasibility
        for i, g_i in enumerate(self.g):
            if g_i(x0) >= 0:
                raise ValueError(f"Initial point violates inequality constraint {i}")

        for j, h_j in enumerate(self.h):
            if abs(h_j(x0)) > 1e-6:
                raise ValueError(f"Initial point violates equality constraint {j}")

        x = x0.copy()
        mu = mu0
        lam = np.zeros(self.p)

        history = {
            'x': [x.copy()],
            'f': [self.f(x)],
            'mu': [mu],
            'iterations': []
        }

        for outer_iter in range(max_outer_iter):
            # Solve barrier subproblem
            x_new, lam_new = self.solve_barrier_subproblem(x, mu, max_iter=max_inner_iter)

            # Check convergence
            if self.m * mu < tol:
                x = x_new
                lam = lam_new
                break

            # Update
            x = x_new
            lam = lam_new
            mu *= mu_factor

            # Store history
            history['x'].append(x.copy())
            history['f'].append(self.f(x))
            history['mu'].append(mu)
            history['iterations'].append(outer_iter + 1)

        # Check constraint violations
        ineq_violations = [max(0, g_i(x)) for g_i in self.g]
        eq_violations = [abs(h_j(x)) for h_j in self.h]

        return {
            'x': x,
            'f': self.f(x),
            'lambda': lam,
            'success': self.m * mu < tol,
            'iterations': outer_iter + 1,
            'ineq_violations': ineq_violations,
            'eq_violations': eq_violations,
            'history': history
        }


# Example usage and test functions
def example_3d_problem():
    """
    Example: 3D constrained optimization problem
    minimize x^2 + y^2 + (z + 1)^2
    subject to x + y + z = 1     (equality constraint)
             x >= 0, y >= 0, z >= 0  (inequality constraints: -x <= 0, -y <= 0, -z <= 0)
    """

    # Objective function: f(x,y,z) = x^2 + y^2 + (z + 1)^2
    def f(vars):
        x, y, z = vars[0], vars[1], vars[2]
        return x ** 2 + y ** 2 + (z + 1) ** 2

    def grad_f(vars):
        x, y, z = vars[0], vars[1], vars[2]
        return np.array([2 * x, 2 * y, 2 * (z + 1)], dtype=float)

    def hess_f(vars):
        return np.array([[2, 0, 0],
                         [0, 2, 0],
                         [0, 0, 2]], dtype=float)

    # Equality constraint: h(x,y,z) = x + y + z - 1 = 0
    def h1(vars):
        x, y, z = vars[0], vars[1], vars[2]
        return x + y + z - 1

    def grad_h1(vars):
        return np.array([1, 1, 1], dtype=float)

    def hess_h1(vars):
        return np.zeros((3, 3), dtype=float)

    # Inequality constraints: g(x,y,z) <= 0
    def g1(vars):  # -x <= 0 (i.e., x >= 0)
        return -vars[0]

    def g2(vars):  # -y <= 0 (i.e., y >= 0)
        return -vars[1]

    def g3(vars):  # -z <= 0 (i.e., z >= 0)
        return -vars[2]

    def grad_g1(vars):
        return np.array([-1, 0, 0], dtype=float)

    def grad_g2(vars):
        return np.array([0, -1, 0], dtype=float)

    def grad_g3(vars):
        return np.array([0, 0, -1], dtype=float)

    def hess_g1(vars):
        return np.zeros((3, 3), dtype=float)

    def hess_g2(vars):
        return np.zeros((3, 3), dtype=float)

    def hess_g3(vars):
        return np.zeros((3, 3), dtype=float)

    # Set up solver
    solver = InteriorPointSolver(
        objective=f,
        grad_objective=grad_f,
        hess_objective=hess_f,
        inequalities=[g1, g2, g3],
        grad_inequalities=[grad_g1, grad_g2, grad_g3],
        hess_inequalities=[hess_g1, hess_g2, hess_g3],
        equalities=[h1],
        grad_equalities=[grad_h1],
        hess_equalities=[hess_h1]
    )

    # Solve (start from feasible point that satisfies all constraints)
    # Need: x + y + z = 1 and x,y,z > 0
    x0 = np.array([0.1, 0.2, 0.7])  # Satisfies x + y + z = 1 and all >= 0
    result = solver.solve(x0)

    return result


if __name__ == "__main__":
    # Run example
    result = example_3d_problem()
    print("Interior Point Solver Results:")
    print(f"Optimal point (x, y, z): {result['x']}")
    print(f"Optimal value: {result['f']:.6f}")
    print(f"Success: {result['success']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Equality constraint violation (x+y+z-1): {result['eq_violations']}")
    print(f"Inequality violations [x>=0, y>=0, z>=0]: {result['ineq_violations']}")

    # Verify the solution
    x_opt, y_opt, z_opt = result['x']
    print(f"\nVerification:")
    print(f"x + y + z = {x_opt + y_opt + z_opt:.6f} (should be 1.0)")
    print(f"x = {x_opt:.6f}, y = {y_opt:.6f}, z = {z_opt:.6f} (all should be >= 0)")

    # Analytical solution for comparison:
    # Using Lagrange multipliers: ∇f = λ∇h at optimum
    # 2x = λ, 2y = λ, 2(z+1) = λ, subject to x + y + z = 1
    # This gives x = y = (z+1)/2, and substituting into constraint:
    # x + y + z = 1 → (z+1)/2 + (z+1)/2 + z = 1 → 2z + 1 = 1 → z = 0
    # Therefore: x = y = 1/2, z = 0, with optimal value = (1/2)² + (1/2)² + 1² = 1.5
    print(f"\nAnalytical solution: x = 0.5, y = 0.5, z = 0.0, f = 1.5")