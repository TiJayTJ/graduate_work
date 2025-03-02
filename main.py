import numpy as np
from scipy.integrate import solve_ivp
from skimage import data
from matplotlib import pyplot as plt

from dna import dna_diffusion, dna_diffusion_reverse, \
    uint8_matrix_to_dna, dna_matrix_to_uint8, dna_xor_diffusion
from graphs_output import print_image, print_phase_space
from system import main_system, get_initial_state
from testing import mix_matrix, unmix_matrix


def array_to_matrix(array, n, m):
    x_val, y_val, z_val = array

    # Подгоняем размер под изображение
    x_val_converted = x_val[:n*m]
    y_val_converted = y_val[:n*m]
    z_val_converted = z_val[:n*m]

    # Преобразования решения в матрицу
    for i in range(n*m):
        x_val_converted[i] = int(x_val_converted[i]) % 256
        y_val_converted[i] = int(y_val_converted[i]) % 256
        z_val_converted[i] = int(z_val_converted[i]) % 256

    matrix = np.empty((n, m, 3), dtype=np.uint8)
    for i in range(n*m):
        matrix[i // m][i % m][0] = x_val_converted[i]
        matrix[i // m][i % m][1] = y_val_converted[i]
        matrix[i // m][i % m][2] = z_val_converted[i]

    return matrix


# Загружаем изображение
image = data.astronaut()
img_col_size, img_row_size, _ = image.shape
image_array = np.array(image)

# Показываем изображение
print_image(image, "Оригинальное изображение")

# ------------------------- Шифрование -------------------------

# Решение системы
initial_state12 = get_initial_state(image_array)  # Начальные условия
initial_state1 = initial_state12[:3]
initial_state2 = initial_state12[3:]
solution1 = solve_ivp(
        main_system, [0, 1200], initial_state1, t_eval=np.linspace(1000, 1200, img_col_size*img_row_size)
    )
solution2 = solve_ivp(
        main_system, [0, 1200], initial_state2, t_eval=np.linspace(1000, 1200, img_col_size*img_row_size)
    )
print(len(solution1.y[0]))
print_phase_space(solution1.y[0], solution1.y[1], solution1.y[2])
# Получаем матрицы для шифрования изображения
matrix_a = array_to_matrix(solution1.y, img_row_size, img_col_size)
matrix_b = array_to_matrix(solution2.y, img_row_size, img_col_size)
# Применяем метод ротационного арифметичесского извлечения
mixed_matrix = mix_matrix(image_array)
print_image(mixed_matrix, "Перемешанное изображение")
# Применение ДНК кодирование для изображения и матрицы
dna_image = uint8_matrix_to_dna(mixed_matrix)
dna_matrix_a = uint8_matrix_to_dna(matrix_a)
dna_matrix_b = uint8_matrix_to_dna(matrix_b)
# Применяем диффузию между изображением и матрицей
dna_encoded_image = dna_diffusion(dna_image, dna_matrix_a)
dna_encoded_image = dna_xor_diffusion(dna_encoded_image, dna_matrix_b)
# Преобразуем ДНК код обратно в байты
binary_image = dna_matrix_to_uint8(dna_encoded_image)
# Отображаем зашифрованное изображение
print_image(binary_image, "Зашифрованное изображение")

# ------------------------- Дешифровка -------------------------

# Применение ДНК кодирование для изображения и матрицы
dna_image = uint8_matrix_to_dna(binary_image)
# Применяем диффузию между изображением и матрицей
dna_encoded_image = dna_xor_diffusion(dna_image, dna_matrix_b)
dna_encoded_image = dna_diffusion_reverse(dna_encoded_image, dna_matrix_a)
# Преобразуем ДНК код обратно в байты
binary_image = dna_matrix_to_uint8(dna_encoded_image)
# Применяем обратный метод ротационного арифметичесского извлечения
unmixed_matrix = unmix_matrix(binary_image)
# Отображаем восстановленное изображение
print_image(unmixed_matrix, "Восстановленное изображение из ДНК")