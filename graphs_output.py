from matplotlib import pyplot as plt


def print_phase_space(x_t_valuesdop, y_t_valuesdop, z_t_valuesdop):
    fig = plt.figure(figsize=(10, 8))

    # xyz plot
    xyz_plot = fig.add_subplot(2, 2, 1, projection='3d')
    xyz_plot.plot(
        x_t_valuesdop, y_t_valuesdop, z_t_valuesdop, color="blue", linewidth=0.5, alpha=0.6
    )
    xyz_plot.scatter(x_t_valuesdop[-1], y_t_valuesdop[-1], z_t_valuesdop[-1], color="red", s=10)

    xyz_plot.set_xlabel("x")
    xyz_plot.set_ylabel("y")
    xyz_plot.set_zlabel('z')

    # xy plot
    xy_plot = fig.add_subplot(2, 2, 2)
    xy_plot.plot(
        x_t_valuesdop, y_t_valuesdop, color="blue", linewidth=0.5, alpha=0.6
    )
    xy_plot.scatter(x_t_valuesdop[-1], y_t_valuesdop[-1], color="red", s=10)

    xy_plot.set_xlabel("x")
    xy_plot.set_ylabel("y")

    # xz plot
    xz_plot = fig.add_subplot(2, 2, 3)
    xz_plot.plot(
        x_t_valuesdop, z_t_valuesdop, color="blue", linewidth=0.5, alpha=0.6
    )
    xz_plot.scatter(x_t_valuesdop[-1], z_t_valuesdop[-1], color="red", s=10)

    xz_plot.set_xlabel("x")
    xz_plot.set_ylabel("z")

    # yz plot
    yz_plot = fig.add_subplot(2, 2, 4)
    yz_plot.plot(
        y_t_valuesdop, z_t_valuesdop, color="blue", linewidth=0.5, alpha=0.6
    )
    yz_plot.scatter(y_t_valuesdop[-1], z_t_valuesdop[-1], color="red", s=10)

    yz_plot.set_xlabel("y")
    yz_plot.set_ylabel("z")

    plt.grid(True)
    plt.show()