from abc import abstractmethod
import numpy as np

from src.common import Function, Minimizer
from src.utils import plot_function_and_paths, plot_objective_vs_iterations


class MinimizerWrapper():
    def __init__(self, obj_tol, param_tol, max_iter, wolfe_const, backtracking_const):
        self.f = None
        self.GD = GD(obj_tol, param_tol, max_iter, wolfe_const, backtracking_const)
        self.Newton = Newton(obj_tol, param_tol, max_iter, wolfe_const, backtracking_const)

    def solve(self, f: Function, x0):
        self.f = f
        gd = self.GD.solve(f, x0)
        newton = self.Newton.solve(f, x0)
        print(f"[GD]: {'success' if gd[2] else 'failure'} | x: {gd[0]} | y: {gd[1]}")
        print(f"[NT]: {'success' if newton[2] else 'failure'} | x: {newton[0]} | y: {newton[1]}")

    def plot(self):
        plot_function_and_paths([self.GD, self.Newton], self.f)
        plot_objective_vs_iterations([self.GD, self.Newton], self.f)


class MinimizerBase(Minimizer):
    def __init__(self, obj_tol, param_tol, max_iter, wolfe_const, backtracking_const):
        self.f = None
        self.obj_tol = obj_tol
        self.param_tol = param_tol
        self.max_iter = max_iter
        self.wolfe_const = wolfe_const
        self.backtracking_const = backtracking_const
        self.history = None
        self.success = False

    def solve(self, f: Function, x0):
        print(f"Solving {f.name} using {self.__class__.__name__} minimizer...")
        self.history = []
        self.f = f
        i = 0
        x = x0
        y = None
        self.success = False

        while i < self.max_iter:
            y, g, h = self.f.eval(x)
            self.history.append(np.append(x, y))
            print(f"[{i}] x: {x}, y: {y}")

            if self.success:
                break

            p = self.next_direction(x, y, g, h)
            alpha = self.next_step_size(x, p)
            x_next = x + alpha * p

            if np.linalg.norm(x_next - x) < self.param_tol or self.should_terminate(x, x_next, y, g, h, p):
                self.success = True

            x = x_next
            i += 1

        self.history = np.asarray(self.history)
        print()

        return x, y, self.success

    def next_step_size(self, x, p):
        alpha = 1

        while self.f.y(x + alpha * p) > self.f.y(x) + self.wolfe_const * alpha * self.f.g(x).T @ p:
            alpha *= self.backtracking_const

        return alpha

    @abstractmethod
    def next_direction(self, x, y, g, h):
        pass

    @abstractmethod
    def should_terminate(self, x, x_next, y, g, h, p):
        pass


class GD(MinimizerBase):
    def next_direction(self, x, y, g, h):
        return -g

    def should_terminate(self, x, x_next, y, g, h, p):
        return np.linalg.norm(y - self.f.y(x_next)) < self.obj_tol


class Newton(MinimizerBase):
    def next_direction(self, x, y, g, h):
        return np.linalg.solve(h, -g)

    def should_terminate(self, x, x_next, y, g, h, p):
        return 0.5 * p.T @ h @ p < self.obj_tol
