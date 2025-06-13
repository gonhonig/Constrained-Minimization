import unittest

from src.constrained_min import InteriorPointSolver
from src.function import SumSquares, TotalVariation, Linear
from src.image_denoising import load_image, show_image
from src.cones import *


class TestImageDenoising(unittest.TestCase):
    def setUp(self):
        self.Solver = InteriorPointSolver()

    def test_image_denoising_SOCP(self):
        n = 10
        offset = 100
        Y = load_image('noisy_img.jpg')[offset:offset + n, offset:offset + n]
        show_image(Y, "noisy")
        ineq = []

        m, n = Y.shape
        len_x = m*n + (m-1)*(n-1) + 1

        for i in range(m-1):
            for j in range(n-1):
                A = np.zeros((len_x, len_x))
                A[0, i*n + j] = 1
                A[0, i*n + j + 1] = -1
                A[1, i*n + j] = 1
                A[1, (i+1)*n + j] = -1
                c = np.zeros(len_x)
                c[m*n + i*(n-1) + j] = 1
                ineq.append(SOC(A=A, c=c))

        A = np.zeros(len_x)
        b = np.zeros(len_x)
        c = np.zeros(len_x)
        A[:m*n] = 1
        A = np.diag(A)
        c[-1] = 1
        b[:m*n] = Y.ravel()
        ineq.append(SOC(A=A, c=c))

        tv = np.zeros(len_x)
        tv[m*n:(m-1)*(n-1)] = 1
        f = Linear(c) + 0.3 * Linear(tv)

        result = self.Solver.solve(func=f, ineq_constraints=ineq, x0=len_x)
        X = result['x'][:m*n].reshape((m,n))
        show_image(X, "clean")


if __name__ == '__main__':
    unittest.main()