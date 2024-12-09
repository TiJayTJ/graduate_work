def dx_dt(x, y, z, koef):
    return y - koef[0]*x + y*z
    # return koef[1]*(y-x)


def dy_dt(x, y, z, koef):
    return koef[1]*x-koef[3]*y-x*z
    # return koef[0]*x - y - x*z


def dz_dt(x, y, z, koef):
    return -koef[2]*z + x*y + x**2
    # return -koef[2]*z + x*y


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


def solve_sde(t, x0, y0, z0, koef):
    x_values = []
    y_values = []
    z_values = []
    x = x0
    y = y0
    z = z0
    t_step = t[1] - t[0]

    for i in range(len(t)):
        x_values.append(x)
        y_values.append(y)
        z_values.append(z)
        x, y, z = runge_kutta(x, y, z, t_step, koef)

    return x_values, y_values, z_values