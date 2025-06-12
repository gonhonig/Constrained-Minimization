from abc import ABC, abstractmethod

import numpy as np


class Function(ABC):
    def __init__(self, Variable, name = None, dim = 1):
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


class Mul(Function):
    def __init__(self, base: Function, scalar):
        super().__init__()
        self.base = base
        self.scalar = scalar

    def eval(self, x):
        y,g,h = self.base.eval(x)
        return self.scalar * y, self.scalar * g, self.scalar * h


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


class SumSquares(Function):
    def __init__(self, A = None):
        super().__init__(dim=2)
        self.A = np.asarray(A).ravel() if A is not None else None

    def eval(self, x):
        x = np.asarray(x)
        shape = x.shape
        x = x.ravel()
        x = x - (self.A if self.A is not None else 0)
        y = x.T @ x
        g = 2 * x
        g = g.reshape(shape)
        h = 2 * np.eye(x.shape[0])

        return y, g, h


class TotalVariation(Function):
    def __init__(self, shape):
        super().__init__(dim=2)
        self.epsilon = 1e-8
        self.shape = shape

    def eval(self, x):
        X = np.asarray(x).reshape(self.shape)
        m, n = X.shape

        # Forward differences
        Dx = X[1:, :] - X[:-1, :]  # shape: (m-1, n)
        Dy = X[:, 1:] - X[:, :-1]  # shape: (m, n-1)

        # Pad to make same size for norm computation
        Dx_padded = np.zeros((m, n))
        Dx_padded[:-1, :] = Dx

        Dy_padded = np.zeros((m, n))
        Dy_padded[:, :-1] = Dy

        # Stack and compute norms
        Dxy = np.stack((Dx_padded, Dy_padded))  # shape: (2, m, n)

        epsilon = 1e-8
        norm = np.linalg.norm(Dxy, axis=0) + epsilon  # shape: (m, n)
        y = np.sum(norm)  # TV value

        # === GRADIENT COMPUTATION ===
        # d/dx_ij of ||∇f||_2 = (∇f) / ||∇f||_2
        # Each pixel contributes to up to 4 norm terms

        grad = np.zeros((m, n))

        # Contribution from being the "center" pixel in forward differences
        # For norm at (i,j): contributes Dx_padded[i,j]/norm[i,j] and Dy_padded[i,j]/norm[i,j]
        grad += Dx_padded / norm  # contribution as x difference
        grad += Dy_padded / norm  # contribution as y difference

        # Contribution from being the "previous" pixel in x direction
        # For norm at (i+1,j): contributes -Dx_padded[i+1,j]/norm[i+1,j]
        grad[:-1, :] -= Dx_padded[1:, :] / norm[1:, :]

        # Contribution from being the "previous" pixel in y direction
        # For norm at (i,j+1): contributes -Dy_padded[i,j+1]/norm[i,j+1]
        grad[:, :-1] -= Dy_padded[:, 1:] / norm[:, 1:]
        grad = grad.ravel()

        # === HESSIAN COMPUTATION ===
        # This is more complex due to the 1/||∇f|| terms
        # H_ij,kl = ∂²TV/∂x_ij∂x_kl

        # For efficiency, we'll compute the Hessian sparsely
        # The Hessian has a specific sparsity pattern due to the local nature of TV

        total_pixels = m * n
        H = np.zeros((total_pixels, total_pixels))

        def idx(i, j):
            """Convert 2D indices to 1D index"""
            return i * n + j

        for i in range(m):
            for j in range(n):
                curr_idx = idx(i, j)
                norm_val = norm[i, j]

                # Self-interaction terms
                # From d²/dx² of norms involving this pixel

                # Diagonal term from all norms this pixel participates in
                diag_val = 0.0

                # From norm at (i,j) - pixel appears in both Dx and Dy
                if i < m - 1 or j < n - 1:  # if this pixel contributes to any norm
                    # Second derivative of sqrt(Dx² + Dy²) w.r.t. pixel value
                    Dx_val = Dx_padded[i, j]
                    Dy_val = Dy_padded[i, j]

                    # d²/dx² of sqrt(Dx² + Dy²) = Dy²/(Dx² + Dy²)^(3/2)
                    diag_val += (Dy_val ** 2) / (norm_val ** 3)
                    diag_val += (Dx_val ** 2) / (norm_val ** 3)

                # From norm at (i-1,j) if exists
                if i > 0:
                    prev_norm = norm[i - 1, j]
                    Dx_prev = Dx_padded[i - 1, j]
                    Dy_prev = Dy_padded[i - 1, j]
                    diag_val += (Dy_prev ** 2) / (prev_norm ** 3)

                    # Cross term with (i-1,j)
                    if Dx_prev != 0:
                        cross_val = -(Dx_prev * Dy_prev) / (prev_norm ** 3)
                        H[curr_idx, idx(i - 1, j)] += cross_val
                        H[idx(i - 1, j), curr_idx] += cross_val

                # From norm at (i,j-1) if exists
                if j > 0:
                    prev_norm = norm[i, j - 1]
                    Dx_prev = Dx_padded[i, j - 1]
                    Dy_prev = Dy_padded[i, j - 1]
                    diag_val += (Dx_prev ** 2) / (prev_norm ** 3)

                    # Cross term with (i,j-1)
                    if Dy_prev != 0:
                        cross_val = -(Dx_prev * Dy_prev) / (prev_norm ** 3)
                        H[curr_idx, idx(i, j - 1)] += cross_val
                        H[idx(i, j - 1), curr_idx] += cross_val

                H[curr_idx, curr_idx] = diag_val

                # Off-diagonal terms for neighboring pixels
                # Interaction with right neighbor
                if j < n - 1:
                    neighbor_idx = idx(i, j + 1)
                    # Both pixels contribute to norm at (i,j)
                    cross_val = -(Dx_padded[i, j] * Dy_padded[i, j]) / (norm_val ** 3)
                    H[curr_idx, neighbor_idx] += cross_val
                    H[neighbor_idx, curr_idx] += cross_val

                # Interaction with bottom neighbor
                if i < m - 1:
                    neighbor_idx = idx(i + 1, j)
                    # Both pixels contribute to norm at (i,j)
                    cross_val = -(Dx_padded[i, j] * Dy_padded[i, j]) / (norm_val ** 3)
                    H[curr_idx, neighbor_idx] += cross_val
                    H[neighbor_idx, curr_idx] += cross_val

        return y, grad, H

    def calc_grad(self, X):
        n, m = X.shape
        grad = np.zeros_like(X)

        Dx = np.zeros_like(X)
        Dy = np.zeros_like(X)

        Dx[:-1, :] = X[1:, :] - X[:-1, :]
        Dy[:, :-1] = X[:, 1:] - X[:, :-1]

        # Compute magnitude
        mag = np.sqrt(Dx ** 2 + Dy ** 2 + self.epsilon)

        # Gradients w.r.t. Dx
        Dx_grad = Dx / mag
        Dy_grad = Dy / mag

        # Backprop Dx
        grad[:-1, :] -= Dx_grad[:-1, :]
        grad[1:, :] += Dx_grad[:-1, :]

        # Backprop Dy
        grad[:, :-1] -= Dy_grad[:, :-1]
        grad[:, 1:] += Dy_grad[:, :-1]

        return grad