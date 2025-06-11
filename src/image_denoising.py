import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cvxpy as cp


def load_image(img_path, format='L'):
    return np.asarray(Image.open(img_path).convert(format)).astype('float32') / 255.0


def show_image(np_img, name = "img", grayscale=True):
    fig, ax = plt.subplots()
    im_path = f'{name}.png'
    if not grayscale:
        ax.imshow(np_img, aspect='equal')
        plt.imsave(im_path, np_img)
    else:
        cmap = plt.get_cmap('gray')
        ax.imshow(np_img, cmap=cmap)
        plt.imsave(im_path, np_img, cmap=cmap)
    ax.axis("off")
    plt.show()


def denoise_image(Y, l = 0.1):
    n, m = Y.shape
    X = cp.Variable((n, m))
    T = cp.Variable((n - 1, m - 1))
    x = cp.vec(X, order='C')
    y = Y.flatten(order='C')
    s = cp.Variable()
    constraints = []

    for i in range(n - 1):
        for j in range(m - 1):
            t = T[i, j]
            dx = X[i, j + 1] - X[i, j]
            dy = X[i + 1, j] - X[i, j]
            constraints.append(cp.SOC(t, cp.hstack([dx, dy])))

    constraints += [cp.SOC(s, x - y)]
    constraints += [X >= 0, X <= 1]
    objective = cp.Minimize(cp.square(s) + l * cp.sum(T))

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=True)

    return X.value


def denoise_image_vectorized(Y, l = 0.1):
    n, m = Y.shape
    X = cp.Variable((n, m))

    objective = cp.Minimize(cp.sum_squares(X - Y) + l * cp.tv(X))
    constraints = [X >= 0, X <= 1]

    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=True)

    return X.value