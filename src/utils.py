import numbers
import os
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable


def plot_function_and_path(func, ineq_constraints=None, A=None, b=None, history = None, limits=None):
    os.makedirs('./plots', exist_ok=True)
    plot_function(func, ineq_constraints, A, b, history, limits)
    title = f'{func.name} - Function and Path'
    plt.title(title)
    plt.savefig(f'./plots/{title}.png')
    plt.close()


def plot_objective_vs_iterations(func, history = None):
    os.makedirs('./plots', exist_ok=True)
    history = np.array(history)
    iterations = history.shape[0]
    xs = np.linspace(0, iterations, iterations)
    ys = history[:, -1]
    plt.plot(xs, ys, '-', color='C2')
    x_end = xs[-1].item()
    y_end = ys[-1].item()
    color = f'C2'
    plt.plot(x_end, y_end, '*', color=color)
    ax = plt.gca()  # Get current axis
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * 0.01
    plt.text(x_end, y_end - offset, f"{len(xs) - 1}", fontsize=9, color=color, ha='center', va='top')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    title = f'{func.name} - Objective Value Vs. Iterations'
    plt.title(title)
    plt.savefig(f'./plots/{title}.png')
    plt.close()


def plot_function(func, ineq_constraints=None, A=None, b=None, history = None, limits=None):
    dim = func.dim
    if dim != 2 and dim != 3:
        raise ValueError('Cannot plot function with dimension {}'.format(dim))

    limits = np.asarray(limits) if limits is not None else np.asarray([[0,1], [0,1], [0,1]])
    A, b, _, _ = parse_affine_vars(A, b)

    N = 50 if dim == 3 else 100
    x = np.linspace(limits[0,0], limits[0,1], N)
    y = np.linspace(limits[1,0], limits[1,1], N)

    if dim == 2:
        X, Y = np.meshgrid(x, y)
        points = np.stack([X, Y], axis=-1)
    else:
        z = np.linspace(limits[2,0], limits[2,1], N)
        X, Y, Z = np.meshgrid(x, y, z)
        points = np.stack([X, Y, Z], axis=-1)

    if A is not None:
        mask = np.isclose(points @ A.T, b)
    else:
        mask = np.expand_dims(np.ones_like(X, dtype=bool), axis=-1)

    if ineq_constraints:
        for constraint in ineq_constraints:
            get_val = lambda x: constraint.eval(x)[0]
            f_val = np.apply_along_axis(get_val, -1, points).reshape(mask.shape)
            mask = np.logical_and(mask, f_val <= 0)

    get_val = lambda x: func.eval(x)[0]
    f_val = np.apply_along_axis(get_val, -1, points)
    f_val = np.where(mask.squeeze(), f_val, np.nan)

    feasible_points = np.where(mask, points, np.nan)
    X = feasible_points[..., 0].flatten()
    Y = feasible_points[..., 1].flatten()
    f_val = f_val.flatten()

    if dim == 3:
        Z = feasible_points[..., 2].flatten()
        mask = ~(np.isnan(X) | np.isnan(Y) | np.isnan(Z))
        X = X[mask]
        Y = Y[mask]
        Z = Z[mask]
        f_val = f_val[mask]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X, Y, Z, c=f_val, cmap='viridis', alpha=0.6, s=5, zorder=1)
        plt.colorbar(scatter, ax=ax, shrink=0.8)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('x3')
        ax.view_init(elev=30, azim=30)
    else:
        plt.scatter(X, Y, c=f_val, cmap='viridis', alpha=0.6, s=3, zorder=1)
        plt.colorbar(shrink=0.8)
        plt.xlabel('x1')
        plt.ylabel('x2')

    if history is not None:
        history = np.asarray(history)
        xs = history[:, 0]
        ys = history[:, 1]
        if dim == 3:
            zs = history[:, 2]
            ax.plot(xs, ys, zs, 'o--', color='C2', zorder=10)
            ax.plot(xs[-1], ys[-1], zs[-1], '*', color='C3', zorder=10)
        else:
            plt.plot(xs, ys, 'o--', color='C2')
            plt.plot(xs[-1], ys[-1], '*', color='C3')


def show_function(f, ineq_constraints, A=None, b=None, path = None, limits=None):
    plot_function(f, ineq_constraints, A, b, path, limits)
    plt.show()


def print_table(results, ineq_constraints = None, A = None, b = None):
    headers = ["Iterations", "x", "y"]
    rows = [len(results['history']), results['x'], f"{results['y']:.3f}"]
    if ineq_constraints is not None:
        headers += [f"{constraint} ≤ 0" for constraint in ineq_constraints]
        rows += [f"{constraint.eval(results['x'])[0]:.3f}" for constraint in ineq_constraints]
    if A is not None:
        A, b, _, _ = parse_affine_vars(A, b)
        headers += [f"{A[i]}x - {b[i]} = 0" for i in range(len(A))]
        rows += [f"{A[i] @ results['x'] - b[i]:.3f}" for i in range(len(A))]
    transposed = list(zip(*([headers] + [rows])))
    table = PrettyTable()
    for row in transposed:
        table.add_row(row)
    print(f"\n{table.get_string(header=False)}\n")


def parse_affine_vars(A, b = None, c = None, d = None):
    if A is None:
        return None, None, None, None

    if isinstance(A, list):
        A = np.asarray(A)
    if A.ndim == 1:
        A = A.reshape(1, -1)

    if b is None:
        b = np.zeros(A.shape[0])
    elif isinstance(b, (list, numbers.Number)):
        b = np.asarray(b)
    if b.ndim == 0:
        b = np.expand_dims(b, 0)

    if c is None:
        c = np.zeros(A.shape[1])
    elif isinstance(c, list):
        c = np.asarray(c)
    if c.ndim == 0:
        c = np.expand_dims(c, 0)

    d = d or 0

    return A, b, c, d