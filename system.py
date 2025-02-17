import numpy as np


def main_system(t, state):
    x, y, z = state
    dxdt = y - 9 * x + y * z
    dydt = 30 * x - 3 * y - x * z
    dzdt = -3.5 * z + x * y + x ** 2
    return [dxdt, dydt, dzdt]


def system_built(koefs):
    a, b, c, k = koefs

    def system(state, t):
        x, y, z = state
        dxdt = y - a * x + y * z
        dydt = b * x - k * y - x * z
        dzdt = -c * z + x * y + x ** 2
        return np.array([dxdt, dydt, dzdt])

    return system


def main_jacobian(state, t):
    x, y, z = state
    J = np.array([
        [-9, 1 + z, y],
        [30 - z, -3, -x],
        [y + 2 * x, x, -3.5]
    ])
    return J


def jacobian_built(koefs):
    a, b, c, k = koefs

    # Определение Якобиана
    def jacobian(state, t):
        x, y, z = state
        J = np.array([
            [-a, 1 + z, y],
            [b - z, -k, -x],
            [y + 2 * x, x, -c]
        ])
        return J

    return jacobian


