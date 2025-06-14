import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, lil_matrix

from src.function import Function
from src.utils import parse_affine_vars


class SOC(Function):
    def __init__(self, A, b=None, c=None, d=None):
        super().__init__(dim=2)
        self.A, self.b, self.c, self.d = parse_affine_vars(A, b, c, d)

        # Convert to CSR format for efficient operations
        if not sparse.issparse(self.A):
            self.A = csr_matrix(self.A)
        elif not isinstance(self.A, csr_matrix):
            self.A = self.A.tocsr()

        self.m, self.n = self.A.shape

        # Pre-compute A.T for efficiency
        self.A_T = self.A.T.tocsr()

        # Numerical stability parameters
        self.eps = 1e-10
        self.min_norm = 1e-8

    def eval(self, x):
        x = np.asarray(x).ravel()

        # Compute Ax + b efficiently
        quad_part = self.A.dot(x) + self.b

        # Compute c^T x + d more efficiently
        linear_part = np.dot(self.c, x) + self.d

        # Compute 2-norm with numerical stability
        norm_quad = np.linalg.norm(quad_part)

        # SOC constraint: ||Ax + b||_2 <= c^T x + d
        # We return f(x) = ||Ax + b||_2 - c^T x - d
        y = norm_quad - linear_part

        # Handle near-zero norm case with better numerical stability
        if norm_quad < self.min_norm:
            # Use regularized gradient
            reg_norm = np.sqrt(norm_quad ** 2 + self.eps ** 2)
            normalized_quad = quad_part / reg_norm
            grad_quad = self.A_T.dot(normalized_quad)

            # Regularized Hessian
            if self.n < 1000:
                # For small problems, compute dense hessian
                AtA = self.A_T.dot(self.A).toarray() / reg_norm

                # Regularized outer product correction
                Atu = self.A_T.dot(quad_part)
                outer_correction = np.outer(Atu, Atu) / (reg_norm ** 3)

                h = AtA - outer_correction

                # Add small regularization to ensure positive definiteness
                h += self.eps * np.eye(self.n)
            else:
                # For large problems, use sparse approximation
                h = (self.A_T.dot(self.A) / reg_norm).toarray()
                h += self.eps * np.eye(self.n)
        else:
            # Standard case: norm is not near zero
            inv_norm = 1.0 / norm_quad
            normalized_quad = quad_part * inv_norm
            grad_quad = self.A_T.dot(normalized_quad)

            # Hessian computation - optimized for sparse matrices
            if self.n < 1000:
                # For small problems, compute exact dense hessian
                AtA = self.A_T.dot(self.A).toarray() * inv_norm

                # Outer product correction
                Atu = self.A_T.dot(quad_part)
                outer_correction = np.outer(Atu, Atu) / (norm_quad ** 3)

                h = AtA - outer_correction
            else:
                # For large problems, use sparse representation
                # This is an approximation but maintains sparsity
                AtA_sparse = self.A_T.dot(self.A) * inv_norm

                # For very large problems, we might want to keep it sparse
                # and use iterative solvers later
                if self.n > 5000:
                    h = AtA_sparse  # Keep as sparse matrix
                else:
                    h = AtA_sparse.toarray()

        # Final gradient
        g = grad_quad - self.c

        return y, g, h

    def pad(self, pad_width, constant_values=0):
        if isinstance(pad_width, tuple) and len(pad_width) == 2:
            pad_left, pad_right = pad_width
        else:
            raise ValueError(f"Expected tuple of length 2 for pad_width, got {pad_width}")

        # Efficient sparse matrix padding
        if pad_right > 0 or pad_left > 0:
            # Create new sparse matrix with proper dimensions
            new_n = self.n + pad_left + pad_right

            # Use lil_matrix for efficient construction
            A_new = lil_matrix((self.m, new_n))

            # Copy old data to the correct position
            A_new[:, pad_left:pad_left + self.n] = self.A

            # Convert back to CSR
            A_new = A_new.tocsr()
        else:
            A_new = self.A.copy()

        # Vector operations
        b_new = self.b.copy() if self.b is not None else None

        if self.c is not None:
            c_new = np.pad(self.c, pad_width=(pad_left, pad_right),
                           constant_values=constant_values, mode='constant')
        else:
            c_new = None

        d_new = self.d

        return SOC(A_new, b_new, c_new, d_new)

    def is_feasible(self, x, tol=1e-6):
        """Check if point x satisfies the SOC constraint"""
        x = np.asarray(x).ravel()
        quad_part = self.A.dot(x) + self.b
        norm_quad = np.linalg.norm(quad_part)
        linear_part = np.dot(self.c, x) + self.d
        return norm_quad <= linear_part + tol