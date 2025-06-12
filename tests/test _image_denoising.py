import unittest
from src.function import *
from src.cones import *
from src.constrained_min import InteriorPointSolver
from src.image_denoising import load_image, show_image
from src.phase_one import get_initial_point
from skimage.transform import resize


class TestImageDenoising(unittest.TestCase):
    def setUp(self):
        self.Solver = InteriorPointSolver()

    def test_image_denoising_SOCP(self):
        return
        n = 30
        offset = 100
        Y = load_image('noisy_img.jpg')[offset:offset + n, offset:offset + n]
        show_image(Y, "noisy")

        x0 = get_initial_point(Y, method="median")
        ineq = []

        m, n = Y.shape
        X = Variable((m,n))
        T = Variable((m-1,n-1))
        s = Variable(1)

        for i in range(n-1):
            for j in range(n-1):
                t = T[i, j]
                dx = X[i, j + 1] - X[i, j]
                dy = X[i + 1, j] - X[i, j]
                ineq.append(SOC(t, hstack((dx, dy))))

        ineq += [SOC(s, X - Y)]

        self.Solver.solve(func=f,
                          ineq_constraints=ineq,
                          x0=x0)

    @unittest.skip("Temporarily disabled")
    def test_image_denoising_TV(self):
        n = 30
        offset = 100
        noisy_image = load_image('noisy_img.jpg')[offset:offset + n, offset:offset + n]
        show_image(noisy_image, "noisy")
        x0 = get_initial_point(noisy_image, method="median")

        f = TotalVariation(noisy_image.shape) + 0.3 * SumSquares(A=noisy_image)
        result = self.Solver.solve(func=f, x0=x0)
        show_image(result['x'], "clean")


if __name__ == '__main__':
    unittest.main()