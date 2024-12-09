import lyapynov
import numpy as np

from lyapynov import LCE
from matplotlib import pyplot as plt

from differential_equations import solve_sde
from graphs_output import print_phase_space

N = 20000
t = np.linspace(0, 100, N)
t_step = t[1] - t[0]
x0_start, y0_start, z0_start = 1, 1, 1
koef = [9, 30, 3, 3]

# DE solution
x_t_valuesdop, y_t_valuesdop, z_t_valuesdop = solve_sde(t, x0_start, y0_start, z0_start, koef)

# show phase space
# print_phase_space(x_t_valuesdop, y_t_valuesdop, z_t_valuesdop)

x0, y0, z0 = x_t_valuesdop[-1], y_t_valuesdop[-1], z_t_valuesdop[-1]


# Определение системы уравнений
a, b, c, k = 9, 30, 3.5, 3


def system_built(a, b, c, k):
    def system(state, t):
        x, y, z = state
        dxdt = y - a * x + y * z
        dydt = b * x - k * y - x * z
        dzdt = -c * z + x * y + x ** 2
        return np.array([dxdt, dydt, dzdt])

    return system


def jacobian_built(a, b, c, k):
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


T_step = 0.0005
a = 0
LCE_results = []
while a <= 50:
    system = system_built(a, b, c, k)
    jacobian = jacobian_built(a, b, c, k)
    continuous_system = lyapynov.ContinuousDS(np.array([1, 2, 10]), float(t[0]), system, jacobian, T_step)
    continuous_system.forward(len(t), False)

    # Computation of LCE
    LCE_output, history = LCE(continuous_system, 3, 0, len(t), True)
    LCE_results.append(LCE_output)
    print(a)
    a += 0.1

# Plot of LCE
plt.figure(figsize=(10, 6))
plt.plot(LCE_results)
plt.xlabel("A params")
plt.ylabel("LCE")
plt.title("Evolution of the LCE for the first 5000 time steps")
plt.show()