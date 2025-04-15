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
x0y0z0 = np.array([1, 2, 10])
koefs = [9, 30, 3.5, 3]
T_step = 0.0005
koef_step = 0.1
koefs_titles = ['a', 'b', 'c', 'k']

for i in range(len(koefs)):
    print('koef[', i, ']', sep='')
    koefs_buffer = koefs.copy()
    koef = 0
    LCE_results = []
    while koef <= 50.1:
        koefs_buffer[i] = koef
        system = system_built(koefs_buffer)
        jacobian = jacobian_built(koefs_buffer)
        continuous_system = lyapynov.ContinuousDS(x0y0z0, float(t[0]), system, jacobian, T_step)
        continuous_system.forward(len(t), False)

        # Computation of LCE
        LCE_output, history = LCE(continuous_system, 3, 0, len(t), True)
        LCE_results.append(LCE_output)
        print(koef)
        koef += koef_step

    # Plot of LCE
    plt.figure(figsize=(10, 6))
    plt.plot(LCE_results)
    plt.xlabel("Параметр " + koefs_titles[i])
    plt.ylabel("Старшие показатели Ляпунова")
    # Настройка шкалы X — только целые значения
    x_vals = np.arange(len(LCE_results))
    x_labels = x_vals / 10
    mask = (x_labels % 10 == 0)
    plt.xticks(x_vals[mask], x_labels[mask].astype(int))

    plt.show()


