import numpy as np

from src.function import Linear, Quadratic


def get_qp_params():
    ineq = [Linear(row) for row in -np.eye(3, dtype=int)]
    f = Quadratic(np.eye(3)) + Linear([0, 0, 2]) + 1
    f.name = 'Quadratic'
    x0 = np.array([0.1, 0.2, 0.7])
    A = [1, 1, 1]
    b = 1

    return {
        'func': f,
        'x0': x0,
        'ineq_constraints': ineq,
        'eq_constraints_mat': A,
        'eq_constraints_rhs': b
    }

def get_lp_params():
    ineq = [Linear([-1,-1]) + 1,
            Linear([0,1]) - 1,
            Linear([1,0]) - 2,
            Linear([0,-1])]
    f = Linear([1,1])
    x0 = np.array([0.5, 0.75])

    return {
        'func': f,
        'x0': x0,
        'ineq_constraints': ineq,
        'mode': 'max'
    }