from abc import ABC, abstractmethod

import numpy as np


class Function(ABC):
    def __init__(self, name = None, dim = 1):
        self.name = name if name else self.__class__.__name__
        self.dim = dim

    def eval(self, x):
        x = np.asarray(x)
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

    def y(self, x):
        return self.value

    def g(self, x):
        return 1

    def h(self, x):
        return 0


class Neg(Function):
    def __init__(self, base: Function):
        super().__init__()
        self.base = base

    def eval(self, x):
        y,g,h = self.base.eval(x)
        return -y, -g, -h

    def y(self, x):
        return -self.base.y(x)

    def g(self, x):
        return -self.base.g(x)

    def h(self, x):
        return -self.base.h(x)



class Add(Function):
    def __init__(self, a: Function, b: Function):
        super().__init__(f"{a.name} + {b.name}", max(a.dim, b.dim))
        self.a = a
        self.b = b

    def eval(self, x):
        ay, ag, ah = self.a.eval(x)
        by, bg, bh = self.b.eval(x)
        return ay + by, ag + by, ah + bg

    def y(self, x):
        ay = self.a.y(x)
        by = self.b.y(x)
        return ay + by

    def g(self, x):
        ag = self.a.g(x)
        bg = self.b.g(x)
        return ag + bg

    def h(self, x):
        ah = self.a.h(x)
        bh = self.b.h(x)
        return ah + bh


class Quadratic(Function):
    def __init__(self, Q, name = None):
        Q = np.asarray(Q)
        super().__init__(name, Q.shape[0])
        self.Q = np.asarray(Q)

    def y(self, x):
        return x.T @ self.Q @ x

    def g(self, x):
        return 2 * self.Q @ x

    def h(self, x):
        return 2 * self.Q


class Linear(Function):
    def __init__(self, a, name = None):
        a = np.asarray(a)
        super().__init__(name, a.shape[0])
        self.a = np.asarray(a)

    def y(self, x):
        return self.a @ x

    def g(self, x):
        return self.a

    def h(self, x):
        n = self.a.shape[0]
        return np.zeros((n, n))
