import numpy as np
from PIL import Image
from scipy.integrate import solve_ivp
from skimage import data
from matplotlib import pyplot as plt
from nolds import corr_dim

from dna import dna_diffusion, dna_diffusion_reverse, \
    uint8_matrix_to_dna, dna_matrix_to_uint8, dna_xor_diffusion
from graphs_output import print_image, print_phase_space
from system import main_system, get_initial_state
from matrix import mix_matrix, unmix_matrix


def array_to_matrix(array, n, m):
    x_val, y_val, z_val = array

    # Подгоняем размер под изображение
    x_val_converted = x_val[:n*m]
    y_val_converted = y_val[:n*m]
    z_val_converted = z_val[:n*m]

    # Преобразования решения в матрицу
    for i in range(n*m):
        x_val_converted[i] = int(x_val_converted[i] * 10**5) % 256
        y_val_converted[i] = int(y_val_converted[i] * 10**5) % 256
        z_val_converted[i] = int(z_val_converted[i] * 10**5) % 256

    matrix = np.empty((n, m, 3), dtype=np.uint8)
    for i in range(n*m):
        matrix[i // m][i % m][0] = x_val_converted[i]
        matrix[i // m][i % m][1] = y_val_converted[i]
        matrix[i // m][i % m][2] = z_val_converted[i]

    return matrix


def histogram_analise(image_array, encoded_image):
    # Строим гистограммы
    plt.figure(figsize=(15, 3))

    # Показываем исходное изображение
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.axis('off')

    # Гистограмма исходного изображения
    plt.subplot(1, 4, 2)
    for i, color in enumerate(['red', 'green', 'blue']):
        hist_values, _ = np.histogram(image_array[:, :, i], bins=256, range=(0, 256))
        plt.bar(range(256), hist_values, color=color, alpha=0.7, width=1.0, label=color)
    plt.xlabel('Gray value')
    plt.ylabel('The number of pixels')
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
    plt.xlabel('Gray value')
    plt.ylabel('The number of pixels')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Загружаем изображение
# image = data.astronaut()
image = Image.open("lena.png")
image_array = np.array(image)
img_col_size, img_row_size, _ = image_array.shape

# --------------------------------- Шифрование ---------------------------------

# Решение системы
initial_state12 = get_initial_state(image_array)    # Начальные условия
initial_state1 = initial_state12[:3]                #
initial_state2 = initial_state12[3:]                #

solution1 = solve_ivp(
        main_system, [0, 1200], initial_state1, t_eval=np.linspace(1000, 1200, img_col_size*img_row_size)
    )
solution2 = solve_ivp(
        main_system, [0, 1200], initial_state2, t_eval=np.linspace(1000, 1200, img_col_size*img_row_size)
    )

# print_phase_space(solution1.y[0], solution1.y[1], solution1.y[2])
# print_phase_space(solution2.y[0], solution2.y[1], solution2.y[2])

# Вычисление корреляционной размерности
# dimension = corr_dim(np.array(solution1.y[0]), emb_dim=3)
# print(f"Корреляционная размерность: {dimension}")

# Получаем матрицы для шифрования изображения
matrix_a = array_to_matrix(solution1.y, img_row_size, img_col_size)
matrix_b = array_to_matrix(solution2.y, img_row_size, img_col_size)

# Применяем метод ротационного арифметичесского извлечения
mixed_matrix = mix_matrix(image_array)
# print_image(mixed_matrix, "Перемешанное изображение")

# Применение ДНК кодирование для изображения и матрицы
dna_image = uint8_matrix_to_dna(mixed_matrix)
dna_matrix_a = uint8_matrix_to_dna(matrix_a)
dna_matrix_b = uint8_matrix_to_dna(matrix_b)

# Применяем диффузию между изображением и матрицей
dna_encoded_image = dna_diffusion(dna_image, dna_matrix_a)
dna_encoded_image = dna_xor_diffusion(dna_encoded_image, dna_matrix_b)

# Преобразуем ДНК код обратно в байты
encoded_image = dna_matrix_to_uint8(dna_encoded_image)

# --------------------------------- Численныйт анализ ---------------------------------

histogram_analise(image_array, encoded_image)

# --------------------------------- Дешифровка ---------------------------------

# Применение ДНК кодирование для изображения и матрицы
dna_image = uint8_matrix_to_dna(encoded_image)

# Применяем диффузию между изображением и матрицей
dna_encoded_image = dna_xor_diffusion(dna_image, dna_matrix_b)
dna_encoded_image = dna_diffusion_reverse(dna_encoded_image, dna_matrix_a)

# Преобразуем ДНК код обратно в байты
binary_image = dna_matrix_to_uint8(dna_encoded_image)

# Применяем обратный метод ротационного арифметичесского извлечения
unmixed_matrix = unmix_matrix(binary_image)

# --------------------------------- Вывод изображений ---------------------------------

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