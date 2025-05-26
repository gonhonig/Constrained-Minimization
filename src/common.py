from abc import ABC, abstractmethod


class Function(ABC):
    def __init__(self, name, is_quadratic):
        self.name = name
        self.is_quadratic = is_quadratic

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