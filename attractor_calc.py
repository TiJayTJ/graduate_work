import numpy as np
import matplotlib.pyplot as plt
import math

def dx_dt(x, y, z, koef):
    return y - koef[0]*x + y*z


def dy_dt(x, y, z, koef):
    return koef[1]*x-koef[3]*y-x*z


def dz_dt(x, y, z, koef):
    return -koef[2]*z + x*y + x**2

N = 20000
t = np.linspace(0, 100, N)
x0_start, y0_start, z0_start = 1, 1, 1


def runge_kutta(x, y, z, dt, koef):
    k1x = dt * dx_dt(x, y, z, koef)
    k1y = dt * dy_dt(x, y, z, koef)
    k1z = dt * dy_dt(x, y, z, koef)
    k2x = dt * dx_dt(x + 0.5 * k1x, y + 0.5 * k1y, z + 0.5 * k1z, koef)
    k2y = dt * dy_dt(x + 0.5 * k1x, y + 0.5 * k1y, z + 0.5 * k1z, koef)
    k2z = dt * dz_dt(x + 0.5 * k1x, y + 0.5 * k1y, z + 0.5 * k1z, koef)
    k3x = dt * dx_dt(x + 0.5 * k2x, y + 0.5 * k2y, z + 0.5 * k2z, koef)
    k3y = dt * dy_dt(x + 0.5 * k2x, y + 0.5 * k2y, z + 0.5 * k2z, koef)
    k3z = dt * dz_dt(x + 0.5 * k2x, y + 0.5 * k2y, z + 0.5 * k2z, koef)
    k4x = dt * dx_dt(x + k3x, y + k3y, z + k3z, koef)
    k4y = dt * dy_dt(x + k3x, y + k3y, z + k3z, koef)
    k4z = dt * dz_dt(x + k3x, y + k3y, z + k3z, koef)

    x_new = x + (k1x + 2 * k2x + 2 * k3x + k4x) / 6
    y_new = y + (k1y + 2 * k2y + 2 * k3y + k4y) / 6
    z_new = z + (k1z + 2 * k2z + 2 * k3z + k4z) / 6

    return x_new, y_new, z_new


def solve_SDE(t, x0, y0, z0, koef):
    x_values = []
    y_values = []
    z_values = []
    x = x0
    y = y0
    z = z0

    for i in range(len(t)):
        x_values.append(x)
        y_values.append(y)
        z_values.append(z)
        x, y, z = runge_kutta(x, y, z, t[1] - t[0], koef)

    return x_values, y_values, z_values


koef = [9, 30, 3, 3.5]
x_t_valuesdop, y_t_valuesdop, z_t_valuesdop = solve_SDE(t, x0_start, y0_start, z0_start, koef)

fig = plt.figure(figsize=(10, 8))

# xyz plot
xyz_plot = fig.add_subplot(2, 2, 1, projection='3d')
xyz_plot.plot(
    x_t_valuesdop, y_t_valuesdop, z_t_valuesdop, color="blue", linewidth=0.5, alpha=0.6
)

xyz_plot.set_xlabel("x")
xyz_plot.set_ylabel("y")
xyz_plot.set_zlabel('z')

# xy plot
xy_plot = fig.add_subplot(2, 2, 2)
xy_plot.plot(
    x_t_valuesdop, y_t_valuesdop, color="blue", linewidth=0.5, alpha=0.6
)

xy_plot.set_xlabel("x")
xy_plot.set_ylabel("y")
plt.grid(True)

# xz plot
xz_plot = fig.add_subplot(2, 2, 3)
xz_plot.plot(
    x_t_valuesdop, z_t_valuesdop, color="blue", linewidth=0.5, alpha=0.6
)

xz_plot.set_xlabel("x")
xz_plot.set_ylabel("z")
plt.grid(True)

# yz plot
yz_plot = fig.add_subplot(2, 2, 4)
yz_plot.plot(
    y_t_valuesdop, z_t_valuesdop, color="blue", linewidth=0.5, alpha=0.6
)

yz_plot.set_xlabel("y")
yz_plot.set_ylabel("z")

plt.grid(True)
plt.savefig("attractor.pdf", format='pdf')
plt.show()