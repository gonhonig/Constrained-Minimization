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