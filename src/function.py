from abc import abstractmethod, ABC

import numpy as np


class Function(ABC):
    def __init__(self, dim, name = None):
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

    def __neg__(self):
        return Neg(self)

    def __mul__(self, o):
        if not np.isscalar(o):
            raise TypeError("Function can only be multiplied by scalar")

        return Mul(self, o)

    def __rmul__(self, other):
        if not np.isscalar(other):
            raise TypeError("Function can only be multiplied by scalar")

        return Mul(self, other)


class Const(Function):
    def __init__(self, value):
        if not np.isscalar(value):
            raise TypeError("Const can only be scalar")
        super().__init__(1)
        self.value = value

    def eval(self, x):
        return self.value, 0, 0


class Neg(Function):
    def __init__(self, base: Function):
        super().__init__(base.dim)
        self.base = base

    def eval(self, x):
        y,g,h = self.base.eval(x)
        return -y, -g, -h


class Mul(Function):
    def __init__(self, base: Function, scalar):
        super().__init__(base.dim)
        self.base = base
        self.scalar = scalar

    def eval(self, x):
        y,g,h = self.base.eval(x)
        return self.scalar * y, self.scalar * g, self.scalar * h


class Add(Function):
    def __init__(self, a: Function, b: Function):
        super().__init__(max(a.dim, b.dim), f"{a.name} + {b.name}")
        self.a = a
        self.b = b

    def eval(self, x):
        ay, ag, ah = self.a.eval(x)
        by, bg, bh = self.b.eval(x)
        return ay + by, ag + bg, ah + bh


class Quadratic(Function):
    def __init__(self, Q):
        self.Q = np.asarray(Q)
        super().__init__(self.Q.shape[0])

    def eval(self, x):
        y =  x.T @ self.Q @ x
        g = 2 * self.Q @ x
        h = 2 * self.Q
        return y, g, h


class Linear(Function):
    def __init__(self, a):
        self.a = np.asarray(a)
        super().__init__(self.a.shape[0])

    def eval(self, x):
        y =  self.a @ x
        g = self.a
        h = np.zeros((self.dim, self.dim))
        return y, g, h


class LogBarrierFunction(Function):
    def __init__(self, f: Function, ineq_constraints: list[Function]):
        super().__init__(f.dim)
        self.f = f
        self.ineq_constraints = ineq_constraints
        self.t = 1
        self.x = None

    def eval(self, x):
        y, g, h = self.f.eval(x)

        if self.ineq_constraints:
            eval_ineq = [ineq.eval(x) for ineq in self.ineq_constraints]
            y_ineq = np.array([eval[0] for eval in eval_ineq])
            g_ineq = np.array([eval[1] for eval in eval_ineq])
            h_ineq = np.array([eval[2] for eval in eval_ineq])

            if np.any(y_ineq >= 0):
                large_val = 1e10
                return large_val, np.full_like(g, large_val), np.full_like(h, large_val)

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

