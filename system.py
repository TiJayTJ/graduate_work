import hashlib

import numpy as np


def main_system(t, state):
    # x, y, z = state
    # dxdt = y - 9 * x + y * z
    # dydt = 30 * x - 3 * y - x * z
    # dzdt = -3.5 * z + x * y + x ** 2
    # return [dxdt, dydt, dzdt]
    x, y, z = state
    dxdt = 10*(y-x)
    dydt = x*(28-z) - y
    dzdt = x*y-8/3*z
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


def sha512hash(text):
    hashed_text = hashlib.sha512(text.encode('utf-8')).hexdigest()
    return hashed_text


def get_initial_state(image):
    # Приведение изображение к тектовому виду
    image_str = np.array2string(image)
    image_hash = sha512hash(image_str)[:128]
    h32 = [0]*32
    for i in range(0, 128, 4):
        h32[i // 4] = int(image_hash[i:i+4], 16)
    keys = [0]*6
    for i in range(6):
        h_sum5 = 0
        for j in range(5):
            h_sum5 += h32[i*5 + j]
        h_sum5 = int(h_sum5 / 5) % 256
        keys[i] = h_sum5
    return keys