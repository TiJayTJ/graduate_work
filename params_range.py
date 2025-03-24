import lyapynov
import numpy as np

from lyapynov import LCE
from matplotlib import pyplot as plt

from system import system_built, jacobian_built
from differential_equations import solve_sde
from printers import print_phase_space

N = 20000
t = np.linspace(0, 100, N)
t_step = t[1] - t[0]
x0_start, y0_start, z0_start = 1, 1, 1
koefs = [9, 30, 3.5, 3]

# DE solution
x_t_valuesdop, y_t_valuesdop, z_t_valuesdop = solve_sde(t, x0_start, y0_start, z0_start, koefs)

# show phase space
# print_phase_space(x_t_valuesdop, y_t_valuesdop, z_t_valuesdop)

x0, y0, z0 = x_t_valuesdop[-1], y_t_valuesdop[-1], z_t_valuesdop[-1]

# T_step = 0.0005
T_step = 0.5
koefs_titles = ['a', 'b', 'c', 'k']
for i in range(len(koefs)):
    print('koef[', i, ']', sep='')
    koefs_buffer = koefs.copy()
    koef = 0
    LCE_results = []
    while koef <= 50:
        koefs_buffer[i] = koef
        system = system_built(koefs_buffer)
        jacobian = jacobian_built(koefs_buffer)
        continuous_system = lyapynov.ContinuousDS(np.array([1, 2, 10]), float(t[0]), system, jacobian, T_step)
        continuous_system.forward(len(t), False)

        # Computation of LCE
        LCE_output, history = LCE(continuous_system, 3, 0, len(t), True)
        LCE_results.append(LCE_output)
        print(koef)
        koef += 0.1

    # Plot of LCE
    plt.figure(figsize=(10, 6))
    plt.plot(LCE_results)
    plt.xlabel("Parameter " + koefs_titles[i])
    plt.ylabel("LCE")
    plt.title("Evolution of the LCE for the first 5000 time steps")
    plt.show()


