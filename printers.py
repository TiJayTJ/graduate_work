from matplotlib import pyplot as plt


def print_phase_space(x_t_valuesdop, y_t_valuesdop, z_t_valuesdop):
    fig = plt.figure(figsize=(10, 8))

    # xyz plot
    xyz_plot = fig.add_subplot(2, 2, 1, projection='3d')
    xyz_plot.plot(
        x_t_valuesdop, y_t_valuesdop, z_t_valuesdop, color="blue", linewidth=0.5, alpha=0.6
    )

    xyz_plot.set_xlabel("x")
    xyz_plot.set_ylabel("y")
    xyz_plot.set_zlabel('z')

    xyz_plot.grid(True)
    xyz_plot.xaxis._axinfo["grid"]['linewidth'] = 0.1
    xyz_plot.yaxis._axinfo["grid"]['linewidth'] = 0.1
    xyz_plot.zaxis._axinfo["grid"]['linewidth'] = 0.1

    # xy plot
    xy_plot = fig.add_subplot(2, 2, 2)
    xy_plot.plot(
        x_t_valuesdop, y_t_valuesdop, color="blue", linewidth=0.5, alpha=0.6
    )

    xy_plot.set_xlabel("x")
    xy_plot.set_ylabel("y", rotation=0)
    xy_plot.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)

    # xz plot
    xz_plot = fig.add_subplot(2, 2, 3)
    xz_plot.plot(
        x_t_valuesdop, z_t_valuesdop, color="blue", linewidth=0.5, alpha=0.6
    )

    xz_plot.set_xlabel("x")
    xz_plot.set_ylabel("z", rotation=0)
    xz_plot.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)

    # yz plot
    yz_plot = fig.add_subplot(2, 2, 4)
    yz_plot.plot(
        y_t_valuesdop, z_t_valuesdop, color="blue", linewidth=0.5, alpha=0.6
    )

    yz_plot.set_xlabel("y")
    yz_plot.set_ylabel("z", rotation=0)
    yz_plot.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)

    plt.show()


def print_encryption_result(image, encoded_image, unmixed_matrix):
    # Создание фигуры и осей
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title("Исходное изображение")

    axes[1].imshow(encoded_image)
    axes[1].axis("off")
    axes[1].set_title("Зашифрованное изображение")

    axes[2].imshow(unmixed_matrix)
    axes[2].axis("off")
    axes[2].set_title("Восстановленное изображение")
    plt.show()


def print_six_img(images, titles):
    # Вывод
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0][0].imshow(images[0])
    axes[0][0].axis("off")
    axes[0][0].set_title(titles[0])

    axes[0][1].imshow(images[1])
    axes[0][1].axis("off")
    axes[0][1].set_title(titles[1])

    axes[0][2].imshow(images[2])
    axes[0][2].axis("off")
    axes[0][2].set_title(titles[2])

    axes[1][0].imshow(images[3])
    axes[1][0].axis("off")
    axes[1][0].set_title(titles[3])

    axes[1][1].imshow(images[4])
    axes[1][1].axis("off")
    axes[1][1].set_title(titles[4])

    axes[1][2].imshow(images[5])
    axes[1][2].axis("off")
    axes[1][2].set_title(titles[5])

    plt.show()

def print_image(image, title):
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)
    plt.show()