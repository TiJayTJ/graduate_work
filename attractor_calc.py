import numpy as np
from scipy.integrate import solve_ivp

# from differential_equations import solve_sde
from graphs_output import print_phase_space
from system import main_system

N = 20000
t = np.linspace(0, 100, N)
x0_y0_z0 = [1.0, 1.0, 1.0]

# x0_start, y0_start, z0_start = 1, 1, 1
# koefs = [9, 30, 3.5, 3]
# x_t_valuesdop, y_t_valuesdop, z_t_valuesdop = solve_sde(t, x0_start, y0_start, z0_start, koefs)
# print_phase_space(x_t_valuesdop, y_t_valuesdop, z_t_valuesdop)

# DE solution
solution = solve_ivp(main_system, [0, 100], x0_y0_z0, args=(), t_eval=t)
x_values, y_values, z_values = solution.y

# show phase space
print_phase_space(x_values, y_values, z_values)