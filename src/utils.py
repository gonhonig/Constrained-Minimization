from math import floor

import matplotlib.pyplot as plt
import numpy as np

def plot_function_and_paths(minimizers, f):
    N = 100
    total_history = [m.history for m in minimizers]
    total_history = np.concatenate(total_history)
    left = np.floor(np.min(total_history[:,0]))
    right = np.ceil(np.max(total_history[:,0]))
    bottom = np.floor(np.min(total_history[:,1]))
    top = np.ceil(np.max(total_history[:,1]))
    x_buffer = 0.1 * (right - left)
    y_buffer = 0.1 * (top - bottom)
    x = np.linspace(left - x_buffer, right + x_buffer, N)
    y = np.linspace(bottom - y_buffer, top + y_buffer, N)
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X, Y], axis=-1)
    Z = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            Z[i, j] = f.y(XY[i, j])

    plt.contour(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(label='f(x)')
    plt.xlabel('x1')
    plt.ylabel('x2')
    title = f'{f.name} - Function and Paths'
    plt.title(title)

    i = 0
    for minimizer in minimizers:
        xs = minimizer.history[:,0]
        ys = minimizer.history[:,1]
        plt.plot(xs, ys, 'o--', color=f'C{i % 10}', label=minimizer.__class__.__name__)
        plt.plot(xs[-1], ys[-1], '*', color=f'C{2 if minimizer.success else 3}')
        i += 1

    plt.legend()
    plt.savefig(f'plots/{title}.png')
    plt.close()



def plot_objective_vs_iterations(minimizers, f):
    i = 0
    for minimizer in minimizers:
        iterations = minimizer.history.shape[0]
        xs = np.linspace(0, iterations, iterations)
        ys = minimizer.history[:, -1]
        plt.plot(xs, ys, '-', color=f'C{i % 10}', label=minimizer.__class__.__name__)
        plt.plot(xs[-1], ys[-1], '*', color=f'C{2 if minimizer.success else 3}')
        i += 1

    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    title = f'{f.name} - Objective Value Vs. Iterations'
    plt.title(title)
    plt.legend()
    plt.savefig(f'plots/{title}.png')
    plt.close()
