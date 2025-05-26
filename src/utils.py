import os
from math import floor

import matplotlib.pyplot as plt
import numpy as np
from pandas.core.computation.expressions import where


def plot_function_and_paths(minimizers, f, limits = None):
    os.makedirs('./plots', exist_ok=True)
    N = 100
    left, top, right, bottom = (-2,2,2,-2) if limits is None else limits

    if minimizers is not None:
        total_history = [m.history for m in minimizers]
        total_history = np.concatenate(total_history)
        left = np.min(total_history[:,0])
        right = np.max(total_history[:,0])
        bottom = np.min(total_history[:,1])
        top = np.max(total_history[:,1])
        x_buffer = 0.2 * (right - left)
        y_buffer = 0.2  * (top - bottom)
        left -= x_buffer
        right += x_buffer
        bottom -= y_buffer
        top += y_buffer

    x = np.linspace(left, right, N)
    y = np.linspace(bottom, top, N)
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X, Y], axis=-1)
    Z = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            Z[i, j] = f.y(XY[i, j])

    if f.is_quadratic:
        levels = np.geomspace(np.min(Z) + 1e-6, np.quantile(Z, 0.60), 15)
    else:
        levels = np.linspace(np.min(Z), np.max(Z), 15)

    plt.contour(X, Y, Z, levels=levels, cmap='viridis')
    plt.xlabel('x1')
    plt.ylabel('x2')
    title = f'{f.name} - Function and Paths'
    plt.title(title)

    if minimizers is not None:
        i = 0
        for minimizer in minimizers:
            if minimizer.history.shape[0] < 2:
                continue

            xs = minimizer.history[:,0]
            ys = minimizer.history[:,1]
            plt.plot(xs, ys, 'o--', color=f'C{i % 10}', label=minimizer.__class__.__name__)
            plt.plot(xs[-1], ys[-1], '*', color=f'C{2 if minimizer.success else 3}')
            i += 1

    plt.legend()
    plt.savefig(f'./plots/{title}.png')
    plt.close()



def plot_objective_vs_iterations(minimizers, f):
    os.makedirs('./plots', exist_ok=True)
    i = 0
    for minimizer in minimizers:
        if minimizer.history.shape[0] < 2:
            continue

        iterations = minimizer.history.shape[0]
        xs = np.linspace(0, iterations, iterations)
        ys = minimizer.history[:, -1]
        plt.plot(xs, ys, '-', color=f'C{i % 10}', label=minimizer.__class__.__name__)
        x_end = xs[-1].item()
        y_end = ys[-1].item()
        color = f'C{2 if minimizer.success else 3}'
        plt.plot(x_end, y_end, '*', color=color)
        plt.text(x_end, y_end - 4.5, f"{len(xs) - 1}", fontsize=9, color=color, ha='center')
        i += 1

    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    title = f'{f.name} - Objective Value Vs. Iterations'
    plt.title(title)
    plt.legend()
    plt.savefig(f'./plots/{title}.png')
    plt.close()
