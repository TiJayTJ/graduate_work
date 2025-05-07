import math

import numpy as np
from matplotlib import pyplot as plt

from encrypt_decrypt import encrypt_image, decrypt_image
from matrix import array_to_matrix
from printers import print_image, print_six_img


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

    print_six_img([image_array, encrypted_image, decrypted_images[0], decrypted_images[1], decrypted_images[2],
                   decrypted_image],
                  ["Исходное изображение", "Зашифрованное изображение, key",
                   "Расшифрованное изображение, key1", "Расшифрованное изображение, key2",
                   "Расшифрованное изображение, key3", "Расшифрованное изображение, key"])

# ---------------------------------------------------------------------------------------------


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
        # Создаём пары из пикселя и его соседей (правый, верхний, верхне-правый)
        h_adj = channel[:, :-1].flatten(), channel[:, 1:].flatten()
        v_adj = channel[:-1, :].flatten(), channel[1:, :].flatten()
        d_adj = channel[:-1, :-1].flatten(), channel[1:, 1:].flatten()

        # Вычисляем коэффициенты корреляции
        print(f"Цвет {color}:")
        print(f"h:\t", np.corrcoef(h_adj[0], h_adj[1])[0, 1])
        print(f"v:\t", np.corrcoef(v_adj[0], v_adj[1])[0, 1])
        print(f"d:\t", np.corrcoef(d_adj[0], d_adj[1])[0, 1])

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

    # plt.show()


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
def noise_attack(image_array, intensity):
    row, col, high = image_array.shape
    noise_layer = array_to_matrix(np.random.normal(0, 255 * intensity, row * col), row, col)
    noise = np.empty((row, col, high), dtype=np.uint8)
    noise_image = np.empty((row, col, high), dtype=np.uint8)
    for i in range(high):
        noise_image[:, :, i] = image_array[:, :, i] + noise_layer
        noise[:, :, i] = noise_layer

    return noise, noise_image


def analyse_noise_attacks(encrypted_image, initial_state12):
    noise001, noised_image001 = noise_attack(encrypted_image, 0.01)
    noised_decryted_image001 = decrypt_image(noised_image001, initial_state12)

    noise005, noised_image005 = noise_attack(encrypted_image, 0.05)
    noised_decryted_image005 = decrypt_image(noised_image005, initial_state12)

    noise02, noised_image02 = noise_attack(encrypted_image, 0.2)
    noised_decryted_image02 = decrypt_image(noised_image02, initial_state12)

    print_six_img([noise001, noise005, noise02, noised_decryted_image001, noised_decryted_image005,
                   noised_decryted_image02],
                  ["Шум, интенсивность 0.01", "Шум, интенсивность 0.05", "Шум, интенсивность 0.2",
                   "Расшифрование, интенсивность 0.01", "Расшифрование, интенсивность 0.05",
                   "Расшифрование, интенсивность 0.2"])


def cropping_attacks(image_array, intensity):
    row, col, high = image_array.shape

    cropped_image = image_array.copy()
    for i in range(round(row * intensity)):
        for j in range(round(col * intensity)):
            for k in range(high):
                cropped_image[i, j, k] = 0

    return cropped_image


def analyse_cropping_attacks(encrypted_image, initial_state12):
    cropped_image0125 = cropping_attacks(encrypted_image, 0.125)
    cropped_decryted_image0125 = decrypt_image(cropped_image0125, initial_state12)
    cropped_image025 = cropping_attacks(encrypted_image, 0.25)
    cropped_decryted_image025 = decrypt_image(cropped_image025, initial_state12)
    cropped_image05 = cropping_attacks(encrypted_image, 0.5)
    cropped_decryted_image05 = decrypt_image(cropped_image05, initial_state12)

    print_six_img([cropped_image0125, cropped_image025, cropped_image05, cropped_decryted_image0125,
                   cropped_decryted_image025, cropped_decryted_image05],
                  ["Обрезано 1.56%", "Обрезано 6.25%", "Обрезано 25%",
                   "Расшифрованное изображение", "Расшифрованное изображение",
                   "Расшифрованное изображение"])

# ---------------------------------------------------------------------------------------------