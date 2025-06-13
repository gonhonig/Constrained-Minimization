import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix

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

    def eval(self, x):
        x = np.asarray(x).ravel()

        # Compute Ax + b efficiently
        quad_part = self.A.dot(x) + self.b

        # Compute c^T x + d more efficiently
        linear_part = np.dot(self.c, x) + self.d

        # Compute 2-norm efficiently
        norm_quad = np.linalg.norm(quad_part)

        # Handle near-zero norm case
        if norm_quad < 1e-12:
            grad_quad = np.zeros(self.n)
            # Return sparse zero matrix for efficiency
            h = sparse.csr_matrix((self.n, self.n))
        else:
            # Gradient: A^T * (Ax + b) / ||Ax + b||
            inv_norm = 1.0 / norm_quad
            normalized_quad = quad_part * inv_norm
            grad_quad = self.A_T.dot(normalized_quad)

            # Hessian computation - optimized for sparse matrices
            # H = A^T * ((I - uu^T/||u||^2) / ||u||) * A

            # For small problems, compute dense hessian
            if self.n < 1000:
                # First term: A^T * A / ||u||
                AtA = self.A_T.dot(self.A).toarray() * inv_norm

                # Second term: outer product correction
                Atu = grad_quad * norm_quad  # A^T * u (before normalization)
                outer_correction = np.outer(Atu, Atu) / (norm_quad ** 3)

                h = AtA - outer_correction
            else:
                # For large problems, keep it sparse and approximate
                AtA_sparse = self.A_T.dot(self.A) * inv_norm
                h = AtA_sparse.toarray()  # Convert to dense for now

        # Final computations
        y = norm_quad - linear_part
        g = grad_quad - self.c

        return y, g, h

    def pad(self, pad_width, constant_values=0):
        if isinstance(pad_width, tuple) and len(pad_width) == 2:
            pad_left, pad_right = pad_width
        else:
            raise ValueError(f"Expected tuple of length 2 for pad_width, got {pad_width}")

        # Efficient sparse matrix padding
        if pad_right > 0:
            # Add zero columns to the right
            zero_cols = sparse.csr_matrix((self.m, pad_right))
            A_new = sparse.hstack([self.A, zero_cols], format='csr')
        else:
            A_new = self.A.copy()

        if pad_left > 0:
            # Add zero columns to the left
            zero_cols = sparse.csr_matrix((self.m, pad_left))
            A_new = sparse.hstack([zero_cols, A_new], format='csr')

        # Vector operations
        b_new = self.b.copy() if self.b is not None else None

        if self.c is not None:
            c_new = np.pad(self.c, pad_width=(pad_left, pad_right),
                           constant_values=constant_values, mode='constant')
        else:
            c_new = None

        d_new = self.d

        return SOC(A_new, b_new, c_new, d_new)