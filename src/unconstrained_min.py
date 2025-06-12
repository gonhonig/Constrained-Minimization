from abc import abstractmethod, ABC
import numpy as np
from src.function import Function


class Solver(ABC):
    def __init__(self, obj_tol = 1e-8, param_tol = 1e-12, wolfe_const = 0.01, backtracking_const = 0.5):
        self.f = None
        self.obj_tol = obj_tol
        self.param_tol = param_tol
        self.wolfe_const = wolfe_const
        self.backtracking_const = backtracking_const
        self.history = None
        self.success = False
        self.is_valid = True

    def solve(self, f: Function, x0, max_iter = 100, verbose = True):
        name = self.__class__.__name__
        self.history = []
        self.f = f
        i = 0
        x = x0
        y = None
        self.success = False
        self.is_valid = True

        if verbose:
            print(f"\nSolving {f.name} using {name} solver...")

        while i <= max_iter:
            y, g, h = self.f.eval(x)

            self.history.append(np.append(x, y))
            if verbose:
                print(f"[{name}:{i}] x: {x}, y: {y}")

            if self.success:
                break

            try:
                p = self.next_direction(x, y, g, h)
            except Exception as e:
                print(e)

            if p is None:
                self.success = False
                self.is_valid = False
                break

            alpha = self.next_step_size(x, p)
            x_next = x + alpha * p

            if np.linalg.norm(x_next - x) < self.param_tol or self.should_terminate(x, x_next, y, g, h, p):
                self.success = True

            x = x_next
            i += 1

        self.history = np.asarray(self.history)
        i -= 0 if self.success else 1

        return {
            'name': name,
            'iterations': i,
            'x': x,
            'y': y,
            'success': self.success,
            'is_valid': self.is_valid
        }

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

    @abstractmethod
    def next_direction(self, x, y, g, h):
        pass

    @abstractmethod
    def should_terminate(self, x, x_next, y, g, h, p):
        pass


class GD(Solver):
    def next_direction(self, x, y, g, h):
        return None if g is None else -g

    def should_terminate(self, x, x_next, y, g, h, p):
        y_next, _ , _ = self.f.eval(x_next)
        return np.linalg.norm(y - y_next) < self.obj_tol

