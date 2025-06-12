import numpy as np

from src.function import Function, Linear
from src.unconstrained_min import Solver
from src.utils import parse_affine_vars
from src.variable import Variable


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
        super().__init__(None, LogBarrierFunction.__name__, f.dim)
        self.f = f
        self.ineq_constraints = ineq_constraints
        self.t = 1
        self.x = None

    def eval_impl(self, x):
        y, g, h = self.f.eval(x)

        if self.ineq_constraints:
            eval_ineq = [ineq.eval(x) for ineq in self.ineq_constraints]
            y_ineq = np.array([eval[0] for eval in eval_ineq])
            try:
                g_ineq = np.array([eval[1] for eval in eval_ineq])
            except Exception as e:
                pass
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

    def solve(self, func: Function, x0 = None, ineq_constraints: list[Function] = None, eq_constraints_mat = None, eq_constraints_rhs = None, verbose = True, variables:list[Variable]=None):
        t = 1
        m = len(ineq_constraints) if ineq_constraints else 0
        f = LogBarrierFunction(func, ineq_constraints)
        newton = Newton()
        A = eq_constraints_mat
        b = eq_constraints_rhs
        x = None
        variables = variables if variables is not None else list(set(v for f in [func] + ineq_constraints for v in f.get_vars() if isinstance(v, Variable)))

        if x0 is not None:
            x = x0
            x0_len = self.set_variables_positions(variables)
            # if len(x0) != x0_len:
            #     raise ValueError("x0 must match the shape of the sum of all variables")
        elif variables is not None:
            x = self.find_x0(variables, ineq_constraints, A, b)

        x = x.ravel()
        i = 1
        history = []
        if verbose:
            print("Solving using interior point method")

        while i == 1 or m / t >= self.epsilon:
            f.set_t(t)
            y, _, _ = func.eval(x)
            history.append(np.append(x, y))
            if verbose:
                print(f"[{i}] y: {y}")
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

        return {
            'x': x,
            'f': y,
            'iterations': i,
            'history': history
        }

    def set_variables_positions(self, variables):
        if variables is None:
            return 0

        length = 0
        for variable in variables:
            if not isinstance(variable, Variable):
                continue
            variable.pos = length
            length += len(variable)
        return length

    def find_x0(self, variables, ineq_constraints, A, b):
        length = self.set_variables_positions(variables)
        A, b, _, _ = parse_affine_vars(A, b)

        if A is None:
            x0 = np.random.rand(length)
        else:
            x0, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

        max_violation = np.max([ineq.eval(x0)[0] for ineq in ineq_constraints])
        x0 = np.append(x0, max_violation + 1)

        s = Variable(1)
        f = Linear([1], s)

        result = self.solve(func=f, x0=x0, ineq_constraints=ineq_constraints, eq_constraints_mat=A, eq_constraints_rhs=b, verbose=False, variables=variables+[s])

        return result['x'][:-1]



