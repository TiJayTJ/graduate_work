import math

import numpy as np
from matplotlib import pyplot as plt

from encrypt_decrypt import encrypt_image, decrypt_image
from matrix import array_to_matrix
from printers import print_image


# ---------------------------------------------------------------------------------------------
def histogram_analise(image, encoded_image):
    # Строим гистограммы
    plt.figure(figsize=(15, 3))

    # Показываем исходное изображение
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.axis('off')

    # Гистограмма исходного изображения
    plt.subplot(1, 4, 2)
    for i, color in enumerate(['red', 'green', 'blue']):
        hist_values, _ = np.histogram(image[:, :, i], bins=256, range=(0, 256))
        plt.bar(range(256), hist_values, color=color, alpha=0.7, width=1.0, label=color)
    plt.xlabel('Количество серого')
    plt.ylabel('Количество пикселей')
    plt.legend()

    # Показываем зашифрованное изображение
    plt.subplot(1, 4, 3)
    plt.imshow(encoded_image)
    plt.axis('off')

    # Гистограмма зашифрованного изображения
    plt.subplot(1, 4, 4)
    for i, color in enumerate(['red', 'green', 'blue']):
        hist_values, _ = np.histogram(encoded_image[:, :, i], bins=256, range=(0, 256))
        plt.bar(range(256), hist_values, color=color, alpha=0.7, width=1.0, label=color)
    plt.xlabel('Количество серого')
    plt.ylabel('Количество пикселей')
    plt.legend()

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------------------------
def analyse_key_sensitivity(image_array, initial_state12):
    img_col, img_row, _ = image_array.shape
    encrypted_image = encrypt_image(image_array, initial_state12)
    decrypted_images = np.empty((3, img_row, img_col, 3), dtype=np.uint8)
    for i in range(3):
        new_initial_state = [0.0]*6
        for j in range(6):
            new_initial_state[j] = initial_state12[j]
        new_initial_state[i] += 10**(-14)
        decrypted_images[i] = decrypt_image(encrypted_image, new_initial_state)
    decrypted_image = decrypt_image(encrypted_image, initial_state12)

    # Вывод
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0][0].imshow(image_array)
    axes[0][0].axis("off")
    axes[0][0].set_title("Исходное изображение")

    axes[0][1].imshow(encrypted_image)
    axes[0][1].axis("off")
    axes[0][1].set_title("Зашифрованное изображение, исхожный ключ")

    axes[0][2].imshow(decrypted_images[0])
    axes[0][2].axis("off")
    axes[0][2].set_title("Расшифрованное изображение, ключ 1")

    axes[1][0].imshow(decrypted_images[1])
    axes[1][0].axis("off")
    axes[1][0].set_title("Расшифрованное изображение, ключ 2")

    axes[1][1].imshow(decrypted_images[2])
    axes[1][1].axis("off")
    axes[1][1].set_title("Расшифрованное изображение, ключ 3")

    axes[1][2].imshow(decrypted_image)
    axes[1][2].axis("off")
    axes[1][2].set_title("Расшифрованное изображение, исходный ключ")

    plt.show()


# ---------------------------------------------------------------------------------------------
def g_func(x):
    n = len(x)
    return sum(x) / n


def f_func(x, g_func_x):
    n = len(x)
    return sum((xi - g_func_x)**2 for xi in x) / n


def cov_func(x, y, g_func_x, g_func_y):
    n = len(x)
    return sum((xi - g_func_x) * (yi - g_func_y) for xi, yi in zip(x, y)) / n


def correlation_coef(x, y):
    g_func_x = g_func(x)
    g_func_y = g_func(y)
    return cov_func(x, y, g_func_x, g_func_y) / (f_func(x, g_func_x) * f_func(y, g_func_y)) ** 0.5


def analyse_correlation(image_array):
    # Разделяем каналы
    r_channel, g_channel, b_channel = (image_array[:, :, 0],
                                       image_array[:, :, 1],
                                       image_array[:, :, 2])
    channels = [r_channel, g_channel, b_channel]
    colors = ['r', 'g', 'b']

    # Строим гистограммы
    fig = plt.figure(figsize=(20, 5))

    # Показываем исходное изображение
    plt.subplot(1, 4, 1)
    plt.imshow(image_array)
    plt.axis('off')

    for idx, (channel, color) in enumerate(zip(channels, colors)):
        # Создаём пары из пикселя и его соседей (верхний, правый, верхне-правый)
        h_adj = channel[:, :-1].flatten(), channel[:, 1:].flatten()
        v_adj = channel[:-1, :].flatten(), channel[1:, :].flatten()
        d_adj = channel[:-1, :-1].flatten(), channel[1:, 1:].flatten()

        # Вычисляем коэффициенты корреляции
        h_correlation = correlation_coef(h_adj[0], h_adj[1])
        v_correlation = correlation_coef(v_adj[0], v_adj[1])
        d_correlation = correlation_coef(d_adj[0], d_adj[1])
        print(f"Цвет {color}:")
        # print('(Посчитанные вручную)')
        # print(f"h:\t", h_correlation)
        # print(f"v:\t", v_correlation)
        # print(f"d:\t", d_correlation)
        # print('(Точные)')
        print(f"h:\t", np.corrcoef(h_adj[0], h_adj[1])[0, 1])
        print(f"h:\t", np.corrcoef(v_adj[0], v_adj[1])[0, 1])
        print(f"h:\t", np.corrcoef(d_adj[0], d_adj[1])[0, 1])

        pixel_values = np.concatenate([h_adj[0], v_adj[0], d_adj[0]])
        adjacent_values = np.concatenate([h_adj[1], v_adj[1], d_adj[1]])
        directions = np.concatenate([np.zeros_like(h_adj[0]), np.ones_like(v_adj[0]), 2 * np.ones_like(d_adj[0])])

        # Вывод
        ax = fig.add_subplot(1, 4, idx + 2, projection='3d')
        ax.scatter(directions, pixel_values, adjacent_values, c=color, s=0.2)
        ax.set_xlabel('Направление')
        ax.set_ylabel('Значение пикселя')
        ax.set_zlabel('Значение соседнего пикселя')
        ax.set_title(f'Корреляция {color.upper()}-канала')

    plt.show()


# ---------------------------------------------------------------------------------------------
def information_entropy(x):
    image_array = np.array(x)

    # Строим гистограмму
    hist = np.bincount(image_array.flatten(), minlength=256)
    probs = hist / np.sum(hist)  # Нормируем, чтобы получить вероятности

    # Вычисляем энтропию (игнорируем нулевые вероятности)
    entropy = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))
    return entropy


def analysis_information_entropy(image_array):
    # Разделяем каналы
    r_channel, g_channel, b_channel = (image_array[:, :, 0],
                                       image_array[:, :, 1],
                                       image_array[:, :, 2])
    information_entropies = [information_entropy(r_channel[:, :].flatten()),
                             information_entropy(g_channel[:, :].flatten()),
                             information_entropy(b_channel[:, :].flatten())]
    color = ['red', 'green', 'blue']
    for i in range(3):
        print(f'Цвет: {color[i]}, Значение энтропии: {information_entropies[i]}')


# ---------------------------------------------------------------------------------------------
def analyse_noise_attacks(image_array, intensity):
    row, col, high = image_array.shape
    noise = array_to_matrix(np.random.normal(0, 255 * intensity, row * col), row, col)
    noise_image = np.empty((row, col, high), dtype=np.uint8)
    for i in range(high):
        noise_image[:, :, i] = image_array[:, :, i] + noise

    return noise_image


def analyse_cropping_attacks(image_array, intensity):
    row, col, high = image_array.shape

    for i in range(round(row * intensity * 2 / 100)):
        for j in range(round(col * intensity * 2 / 100)):
            for k in range(high):
                image_array[i, j, k] = 0

    return image_array