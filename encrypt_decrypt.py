import time

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
    p = 5
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
    time1 = time.time()
    img_row, img_col, _ = image_array.shape

    initial_state1 = initial_state12[:3]    # Начальнные данные
    initial_state2 = initial_state12[3:]    #

    solution1 = solve_ivp(
        main_system, [0, 200], initial_state1, t_eval=np.linspace(100, 200, img_col * img_row)
    )
    solution2 = solve_ivp(
        main_system, [0, 200], initial_state2, t_eval=np.linspace(100, 200, img_col * img_row)
    )
    time2 = time.time()
    # Вычисление корреляционной размерности
    # dimension = corr_dim(np.array(solution1.y[0]), emb_dim=3)
    # print(f"Корреляционная размерность: {dimension}")

    # Получаем матрицы для шифрования изображения
    matrix_a = convert_for_encrypt(solution1.y, img_row, img_col)
    matrix_b = convert_for_encrypt(solution2.y, img_row, img_col)
    time3 = time.time()
    # Применяем метод ротационного арифметичесского извлечения
    mixed_matrix = mix_matrix(image_array)
    # print_image(mixed_matrix, "Перемешанное изображение")
    time4 = time.time()
    # Применение ДНК кодирование для изображения и матрицы

    dna_image, dna_matrix_a, dna_matrix_b = uint8_matrix_to_dna(mixed_matrix, matrix_a, matrix_b)
    time5 = time.time()
    # Применяем диффузию между изображением и матрицей
    dna_encoded_image = dna_diffusion(dna_image, dna_matrix_a)
    time6 = time.time()
    dna_encoded_image = dna_xor_diffusion(dna_encoded_image, dna_matrix_b)
    time7 = time.time()
    # Преобразуем ДНК код обратно в байты
    result = dna_matrix_to_uint8(dna_encoded_image)
    time8 = time.time()

    print(f'Решение ДС: {time2 - time1:.4f} секунд')
    print(f'Получаем матрицы для шифрования изображения: {time3 - time2:.4f} секунд')
    print(f'Перемешиваем матрицы: {time4 - time3:.4f} секунд')
    print(f'ДНК кодирование: {time5 - time4:.4f} секунд')
    print(f'Диффузия1: {time6 - time5:.4f} секунд')
    print(f'Диффузия2: {time7 - time6:.4f} секунд')
    print(f'Обратное ДНК кодирование: {time8 - time7:.4f} секунд')
    return result


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
    dna_image, dna_matrix_a, dna_matrix_b = uint8_matrix_to_dna(encoded_image, matrix_a, matrix_b)

    # Применяем диффузию между изображением и матрицей
    dna_encoded_image = dna_xor_diffusion(dna_image, dna_matrix_b)
    dna_encoded_image = dna_diffusion_reverse(dna_encoded_image, dna_matrix_a)

    # Преобразуем ДНК код обратно в байты
    binary_image = dna_matrix_to_uint8(dna_encoded_image)

    # Применяем обратный метод ротационного арифметичесского извлечения
    unmixed_matrix = unmix_matrix(binary_image)
    return unmixed_matrix