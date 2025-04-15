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

system = system_built(koefs)
jacobian = jacobian_built(koefs)
continuous_system = lyapynov.ContinuousDS(x0y0z0, float(t[0]), system, jacobian, T_step)
continuous_system.forward(len(t), False)

# Computation of LCE
LCE_output, history = LCE(continuous_system, 3, 0, len(t), True)
print(LCE_output)

d_l = 2 + (LCE_output[0] + LCE_output[1]) / abs(LCE_output[2])
print(d_l)