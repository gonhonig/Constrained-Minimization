import numpy as np


class VariableBase:
    def __add__(self, other):
        if isinstance(other, VariableBase):
            other = ConstVariable(other)
        return SumVariable(self, other)

    def __sub__(self, other):
        if isinstance(other, VariableBase):
            other = ConstVariable(other)
        return SubtractVariable(self, other)

    def __getitem__(self, key):
        return IndexVariable(self, key)


class Variable(VariableBase):
    def __init__(self, shape):
        self.pos = 0
        self.shape = shape
        self.len = np.prod(shape)

    def __call__(self, x):
        return x[self.pos:self.pos+self.len].reshape(self.shape)

    def __len__(self):
        return self.len


class IndexVariable(VariableBase):
    def __init__(self, base, key):
        self.base = base
        self.key = key

    def __call__(self, x):
        return self.base(x)[self.key]


class ConstVariable(VariableBase):
    def __init__(self, value):
        self.value = value

    def __call__(self, x):
        return self.value


class SumVariable(VariableBase):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, x):
        return self.a(x) + self.b(x)


class SubtractVariable(VariableBase):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, x):
        return self.a(x) - self.b(x)


def hstack(arrays):
    return lambda x: np.hstack([(v(x) if callable(v) else v) for v in arrays])


def vstack(arrays):
    return lambda x: np.vstack([(v(x) if callable(v) else v) for v in arrays])