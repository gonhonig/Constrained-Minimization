from abc import ABC, abstractmethod


class Function(ABC):
    def __init__(self, name):
        self.name = name

    def eval(self, x):
        return self.y(x), self.g(x), self.h(x)

    @abstractmethod
    def y(self, x):
        pass

    @abstractmethod
    def g(self, x):
        pass

    @abstractmethod
    def h(self, x):
        pass


class Minimizer(ABC):
    @abstractmethod
    def solve(self, f: Function, x0):
        pass

    @abstractmethod
    def next_step_size(self, x, p):
        pass

    @abstractmethod
    def next_direction(self, x, y, g, h):
        pass

    @abstractmethod
    def should_terminate(self, x, x_next, y, g, h, p):
        pass