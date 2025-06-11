import os
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable


def plot_function_and_paths(solvers, f):
    os.makedirs('./plots', exist_ok=True)
    total_history = np.concatenate([s.history for s in solvers if s.is_valid])
    left = np.min(total_history[:, 0])
    right = np.max(total_history[:, 0])
    bottom = np.min(total_history[:, 1])
    top = np.max(total_history[:, 1])
    x_buffer = 0.2 * (right - left)
    y_buffer = 0.2 * (top - bottom)
    left -= x_buffer
    right += x_buffer
    bottom -= y_buffer
    top += y_buffer
    plot_function(f, (left, top, right, bottom))
    title = f'{f.name} - Function and Paths'
    plt.title(title)

    if solvers is not None:
        i = 0
        for solver in solvers:
            if not solver.is_valid:
                continue

            xs = solver.history[:, 0]
            ys = solver.history[:, 1]
            plt.plot(xs, ys, 'o--', color=f'C{i % 10}', label=f"{solver.__class__.__name__} {'✔' if solver.success else '✖'}")
            plt.plot(xs[-1], ys[-1], '*', color=f'C{2 if solver.success else 3}')
            i += 1

    plt.legend()
    plt.savefig(f'./plots/{title}.png')
    plt.close()


def plot_objective_vs_iterations(solvers, f):
    os.makedirs('./plots', exist_ok=True)
    i = 0
    for solver in solvers:
        if not solver.is_valid:
            continue

        iterations = solver.history.shape[0]
        xs = np.linspace(0, iterations, iterations)
        ys = solver.history[:, -1]
        plt.plot(xs, ys, '-', color=f'C{i % 10}', label=f"{solver.__class__.__name__} {'✔' if solver.success else '✖'}")
        x_end = xs[-1].item()
        y_end = ys[-1].item()
        color = f'C{2 if solver.success else 3}'
        plt.plot(x_end, y_end, '*', color=color)
        ax = plt.gca()  # Get current axis
        ymin, ymax = ax.get_ylim()
        offset = (ymax - ymin) * 0.01
        plt.text(x_end, y_end - offset, f"{len(xs) - 1}", fontsize=9, color=color, ha='center', va='top')
        i += 1

    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    title = f'{f.name} - Objective Value Vs. Iterations'
    plt.title(title)
    plt.legend()
    plt.savefig(f'./plots/{title}.png')
    plt.close()


def plot_function(f, limits=None):
    dim = f.dim
    left, top, right, bottom = (-2, 2, 2, -2) if limits is None else limits
    N = 100
    x = np.linspace(left, right, N)
    y = np.linspace(bottom, top, N)
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X, Y], axis=-1)
    get_y = lambda x: f.eval(x)[0]
    Z = np.apply_along_axis(get_y, -1, XY)
    quantile_levels = np.linspace(0.05, 0.95, 15)
    levels = np.quantile(Z.flatten(), quantile_levels)
    plt.contour(X, Y, Z, levels=levels, cmap='viridis')
    plt.xlabel('x1')
    plt.ylabel('x2')


def show_function(f, limits=None):
    plot_function(f, limits)
    plt.title(f.name)
    plt.legend()
    plt.show()



def print_table(solvers):
    headers = ["Iterations", "x", "y", "Status"]
    rows = [[r['iterations'], r['x'], r['y'], "Success" if r['success'] else f"Failure: {"max_iter" if r['is_valid'] else f"can't solve"}"] for r in solvers]
    transposed = list(zip(*([headers] + rows)))
    table = PrettyTable()
    table.field_names = ["Solver"] + [r['name'] for r in solvers]
    for row in transposed:
        table.add_row(row)
    print(f"\n{table}\n")