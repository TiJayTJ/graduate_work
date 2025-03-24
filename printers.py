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


def print_three_img(image, encoded_image, unmixed_matrix):
    # Создание фигуры и осей
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title("Оригинальное изображение")

    axes[1].imshow(encoded_image)
    axes[1].axis("off")
    axes[1].set_title("Зашифрованное изображение")

    axes[2].imshow(unmixed_matrix)
    axes[2].axis("off")
    axes[2].set_title("Восстановленное изображение из ДНК")
    plt.show()

def print_image(image, title):
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)
    plt.show()