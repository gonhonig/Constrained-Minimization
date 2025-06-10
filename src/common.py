from abc import ABC, abstractmethod

import numpy as np


class Function(ABC):
    def __init__(self, name = None, dim = 1):
        self.name = name if name else self.__class__.__name__
        self.dim = dim

    @abstractmethod
    def eval(self, x):
        pass

    def __add__(self, o):
        other = o if isinstance(o, Function) else Const(o)
        return Add(self, other)

    def __sub__(self, o):
        other = o if isinstance(o, Function) else Const(o)
        return Add(self, Neg(other))


class Const(Function):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def eval(self, x):
        return self.value, 0, 0


class Neg(Function):
    def __init__(self, base: Function):
        super().__init__()
        self.base = base

    def eval(self, x):
        y,g,h = self.base.eval(x)
        return -y, -g, -h


class Add(Function):
    def __init__(self, a: Function, b: Function):
        super().__init__(f"{a.name} + {b.name}", max(a.dim, b.dim))
        self.a = a
        self.b = b

    def eval(self, x):
        ay, ag, ah = self.a.eval(x)
        by, bg, bh = self.b.eval(x)
        return ay + by, ag + bg, ah + bh


class Quadratic(Function):
    def __init__(self, Q, name = None):
        Q = np.asarray(Q)
        super().__init__(name, Q.shape[0])
        self.Q = np.asarray(Q)

    def eval(self, x):
        x = np.asarray(x)
        y =  x.T @ self.Q @ x
        g = 2 * self.Q @ x
        h = 2 * self.Q
        return y, g, h


class Linear(Function):
    def __init__(self, a, name = None):
        a = np.asarray(a)
        super().__init__(name, a.shape[0])
        self.a = np.asarray(a)

    def eval(self, x):
        x = np.asarray(x)
        y =  self.a @ x
        g = self.a
        h = np.zeros((self.dim, self.dim))
        return y, g, h
