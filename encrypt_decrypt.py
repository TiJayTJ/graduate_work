import numpy as np
from scipy.integrate import solve_ivp

from dna import uint8_matrix_to_dna, dna_diffusion, dna_xor_diffusion, dna_matrix_to_uint8, dna_diffusion_reverse
from matrix import mix_matrix, unmix_matrix, mix_image, unmix_image
from system import main_system


def convert_for_encrypt(array, n, m):
    x_val, y_val, z_val = array

    # Подгоняем размер под изображение
    x_val_converted = x_val[:n*m]
    y_val_converted = y_val[:n*m]
    z_val_converted = z_val[:n*m]

    # Преобразования решения в матрицу
    p = 10
    for i in range(n*m):
        x_val_converted[i] = int(x_val_converted[i] * 10**p) % 256
        y_val_converted[i] = int(y_val_converted[i] * 10**p) % 256
        z_val_converted[i] = int(z_val_converted[i] * 10**p) % 256

    matrix = np.empty((n, m, 3), dtype=np.uint8)
    for i in range(n*m):
        matrix[i // m][i % m][0] = x_val_converted[i]
        matrix[i // m][i % m][1] = y_val_converted[i]
        matrix[i // m][i % m][2] = z_val_converted[i]

    return matrix


def encrypt_image(image_array, initial_state12):
    img_row, img_col, _ = image_array.shape

    initial_state1 = initial_state12[:3]    # Начальнные данные
    initial_state2 = initial_state12[3:]    #

    solution1 = solve_ivp(
        main_system, [0, 200], initial_state1, t_eval=np.linspace(100, 200, img_col * img_row)
    )
    solution2 = solve_ivp(
        main_system, [0, 200], initial_state2, t_eval=np.linspace(100, 200, img_col * img_row)
    )

    # Вычисление корреляционной размерности
    # dimension = corr_dim(np.array(solution1.y[0]), emb_dim=3)
    # print(f"Корреляционная размерность: {dimension}")

    # Получаем матрицы для шифрования изображения
    matrix_a = convert_for_encrypt(solution1.y, img_row, img_col)
    matrix_b = convert_for_encrypt(solution2.y, img_row, img_col)

    # Применяем метод ротационного арифметичесского извлечения
    mixed_matrix = mix_image(image_array)
    # print_image(mixed_matrix, "Перемешанное изображение")

    # Применение ДНК кодирование для изображения и матрицы
    dna_image = uint8_matrix_to_dna(mixed_matrix)
    dna_matrix_a = uint8_matrix_to_dna(matrix_a)
    dna_matrix_b = uint8_matrix_to_dna(matrix_b)

    # Применяем диффузию между изображением и матрицей
    dna_encoded_image = dna_diffusion(dna_image, dna_matrix_a)
    dna_encoded_image = dna_xor_diffusion(dna_encoded_image, dna_matrix_b)

    # Преобразуем ДНК код обратно в байты
    return dna_matrix_to_uint8(dna_encoded_image)


def decrypt_image(encoded_image, initial_state12):
    img_row, img_col, _ = encoded_image.shape

    initial_state1 = initial_state12[:3]  # Начальнные данные
    initial_state2 = initial_state12[3:]  #

    solution1 = solve_ivp(
        main_system, [0, 200], initial_state1, t_eval=np.linspace(100, 200, img_col * img_row)
    )
    solution2 = solve_ivp(
        main_system, [0, 200], initial_state2, t_eval=np.linspace(100, 200, img_col * img_row)
    )

    # Получаем матрицы для шифрования изображения
    matrix_a = convert_for_encrypt(solution1.y, img_row, img_col)
    matrix_b = convert_for_encrypt(solution2.y, img_row, img_col)

    # Применение ДНК кодирование для изображения и матрицы
    dna_image = uint8_matrix_to_dna(encoded_image)
    dna_matrix_a = uint8_matrix_to_dna(matrix_a)
    dna_matrix_b = uint8_matrix_to_dna(matrix_b)

    # Применяем диффузию между изображением и матрицей
    dna_encoded_image = dna_xor_diffusion(dna_image, dna_matrix_b)
    dna_encoded_image = dna_diffusion_reverse(dna_encoded_image, dna_matrix_a)

    # Преобразуем ДНК код обратно в байты
    binary_image = dna_matrix_to_uint8(dna_encoded_image)

    # Применяем обратный метод ротационного арифметичесского извлечения
    unmixed_matrix = unmix_image(binary_image)
    return unmixed_matrix