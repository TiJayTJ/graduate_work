import hashlib
import time

import numpy as np
from PIL import Image
from scipy.integrate import solve_ivp
from skimage import data
from numba import njit

from encrypt_decrypt import decrypt_image
from printers import print_encryption_result

# Загружаем изображение
astronaut = data.astronaut()
lena = Image.open("images/lena.png")
baboon = Image.open("images/baboon.png")
image_array = np.array(lena)
img_row, img_col, _ = image_array.shape


def sha512hash(text):
    hashed_text = hashlib.sha512(text.encode('utf-8')).hexdigest()
    return hashed_text


def get_initial_state(image):
    # Приведение изображение к тектовому виду
    image_str = np.array2string(image)
    image_hash = sha512hash(image_str)[:128]
    h64 = [0]*64
    for i in range(0, 128, 2):
        h64[i // 2] = int(image_hash[i:i+2], 16)
    keys = [0.0]*6
    for i in range(6):
        h_sum = 0
        for j in range(10):
            h_sum += h64[i*10 + j]
        h_sum = round(int(h_sum / 10) / 256, 2)
        keys[i] = h_sum
    return keys


def main_system(t, state):
    x, y, z = state
    dxdt = y - 9 * x + y * z
    dydt = 30 * x - 3 * y - x * z
    dzdt = -3.5 * z + x * y + x ** 2
    return [dxdt, dydt, dzdt]


def convert_for_encrypt(array, n, m):
    x_val, y_val, z_val = array

    # Преобразования решения в матрицу
    p = 10
    scale_mod = lambda arr: (arr[:n * m] * 10 ** p).astype(int) % 256
    x_val_converted = scale_mod(x_val)
    y_val_converted = scale_mod(y_val)
    z_val_converted = scale_mod(z_val)

    matrix = np.stack([
        x_val_converted.reshape((n, m)),
        y_val_converted.reshape((n, m)),
        z_val_converted.reshape((n, m))
    ], axis=-1).astype(np.uint8)

    return matrix


def mix_matrix(matrix):
    n = len(matrix)     # кол-во столбцов матрицы
    m = len(matrix[0])      # кол-во строк матрицы
    h = len(matrix[0][0])  # глубина матрицы
    row_half = n // 2
    mixed_matrix = np.empty((n, m, h), dtype=np.uint8)

    for k in range(h):
        counter = 0
        for i in range(row_half):
            for j in range(i, m - i - 1):
                counter_2_div_n = ((counter * 2) // n) * 2
                mixed_matrix[counter_2_div_n][(counter * 2) % n][k] = matrix[i][j][k]
                mixed_matrix[counter_2_div_n][(counter * 2) % n + 1][k] = matrix[j][n - i - 1][(k + 1) % 3]
                mixed_matrix[counter_2_div_n + 1][(counter * 2) % n + 1][k] = matrix[n - i - 1][n - j - 1][(k + 2) % 3]
                mixed_matrix[counter_2_div_n + 1][(counter * 2) % n][k] = matrix[n - j - 1][i][k]
                counter += 1
    return mixed_matrix


BITS_TO_DNA = {
    0: "A",  # 00
    1: "G",  # 01
    2: "C",  # 10
    3: "T"   # 11
}

DNA_TO_BITS = {
    "A": 0b00,
    "G": 0b01,
    "C": 0b10,
    "T": 0b11
}


def byte_to_dna(byte):
    byte = int(byte)  # Приводим np.uint8 к обычному int
    return [BITS_TO_DNA[int((byte >> (6 - 2 * i)) & 0b11)] for i in range(4)]


def dna_to_byte(dna):
    return sum(DNA_TO_BITS[base] << (6 - 2 * i) for i, base in enumerate(dna))


BITS_TO_DNA2 = np.array(["A", "G", "C", "T"])  # Индексация: 00 → A, 01 → G, 10 → C, 11 → T


def uint8_matrix_to_dna(matrix1, matrix2, matrix3):
    def matrix_to_dna(matrix):
        flat = matrix.reshape(-1)
        # Получаем 4 пары битов для каждого байта
        dna_bases = np.empty((flat.shape[0], 4), dtype=object)
        for i in range(4):
            shift = 6 - 2 * i
            indices = (flat >> shift) & 0b11
            dna_bases[:, i] = BITS_TO_DNA2[indices]
        return dna_bases.reshape((*matrix.shape[:2], 3, 4))  # (H, W, 3, 4)

    # Преобразуем каждую матрицу в ДНК представление
    dna_matrix1 = matrix_to_dna(matrix1)
    dna_matrix2 = matrix_to_dna(matrix2)
    dna_matrix3 = matrix_to_dna(matrix3)

    # Возвращаем как три отдельных матрицы
    return dna_matrix1, dna_matrix2, dna_matrix3


def dna_matrix_to_uint8(matrix):
    # Предполагаем, что matrix имеет форму (H, W, 3, 4), где 4 — количество ДНК-баз
    dna_bytes = np.vectorize(lambda b: DNA_TO_BITS[b])(matrix)  # Преобразуем каждую букву в 2 бита
    shifts = np.array([6, 4, 2, 0], dtype=np.uint8)

    # Массив битов сдвигается и складывается по каждой строке (т.е. по 4 базам)
    byte_values = np.sum(dna_bytes << shifts.reshape((1, 1, 1, 4)), axis=-1).astype(np.uint8)

    return byte_values



DNA_ADD_RULES = {
    ("A", "A"): "A", ("A", "G"): "G", ("A", "C"): "C", ("A", "T"): "T",
    ("G", "G"): "C", ("G", "C"): "T", ("G", "T"): "A",
    ("C", "C"): "A", ("C", "T"): "G",
    ("T", "T"): "C"
}

DNA_XOR_RULES = {
    ("A", "A"): "A", ("A", "G"): "G", ("A", "C"): "C", ("A", "T"): "T",
    ("G", "G"): "A", ("G", "C"): "T", ("G", "T"): "C",
    ("C", "C"): "A", ("C", "T"): "G",
    ("T", "T"): "A"
}

DNA_SUB_RULES = {
    ("A", "A"): "A", ("A", "G"): "T", ("A", "C"): "C", ("A", "T"): "G",
    ("G", "A"): "G", ("G", "G"): "A", ("G", "C"): "T", ("G", "T"): "C",
    ("C", "A"): "C", ("C", "G"): "G", ("C", "C"): "A", ("C", "T"): "T",
    ("T", "A"): "T", ("T", "G"): "C", ("T", "C"): "G", ("T", "T"): "A"
}


def apply_dna_rule(rule_dict, base1, base2):
    return [rule_dict.get((b1, b2)) or rule_dict.get((b2, b1)) for b1, b2 in zip(base1, base2)]


def dna_add(base1, base2):
    return apply_dna_rule(DNA_ADD_RULES, base1, base2)


def dna_xor(base1, base2):
    return apply_dna_rule(DNA_XOR_RULES, base1, base2)


def dna_sub(base1, base2):
    return [DNA_SUB_RULES[(b1, b2)] for b1, b2 in zip(base1, base2)]


def dna_diffusion(dna_matrix1, dna_matrix2):
    # Предполагается, что размерность: (H, W, 3, 4) — 3 канала, 4 символа на байт
    res = np.empty((dna_matrix1.shape[0], dna_matrix1.shape[1], 3, 4), dtype=object)

    # Векторизованная реализация операций через np.char.add и правила
    def vectorized_dna_op(mat1, mat2, rule_dict):
        # Предполагается shape (H, W, 4)
        h, w, _ = mat1.shape
        result = np.empty((h, w, 4), dtype=object)
        for i in range(4):
            b1 = mat1[:, :, i].flatten()
            b2 = mat2[:, :, i].flatten()
            out = [rule_dict.get((x, y)) or rule_dict.get((y, x)) for x, y in zip(b1, b2)]
            result[:, :, i] = np.array(out, dtype=object).reshape(h, w)
        return result

    res[:, :, 0] = vectorized_dna_op(dna_matrix1[:, :, 0], dna_matrix2[:, :, 0], DNA_ADD_RULES)
    res[:, :, 1] = vectorized_dna_op(dna_matrix1[:, :, 1], dna_matrix2[:, :, 1], DNA_SUB_RULES)
    res[:, :, 2] = vectorized_dna_op(dna_matrix1[:, :, 2], dna_matrix2[:, :, 2], DNA_XOR_RULES)

    return res


def dna_xor_diffusion(dna_matrix1, dna_matrix2):
    def vectorized_xor(mat1, mat2):
        flat_shape = (-1, 4)
        m1 = mat1.reshape(flat_shape)
        m2 = mat2.reshape(flat_shape)
        result = np.empty_like(m1)
        for i in range(4):
            b1 = m1[:, i]
            b2 = m2[:, i]
            result[:, i] = [DNA_XOR_RULES.get((x, y)) or DNA_XOR_RULES.get((y, x)) for x, y in zip(b1, b2)]
        return result.reshape(mat1.shape)

    dna_matrix1[:, :, 0] = vectorized_xor(dna_matrix1[:, :, 0], dna_matrix2[:, :, 0])
    dna_matrix1[:, :, 1] = vectorized_xor(dna_matrix1[:, :, 1], dna_matrix2[:, :, 1])
    dna_matrix1[:, :, 2] = vectorized_xor(dna_matrix1[:, :, 2], dna_matrix2[:, :, 2])
    return dna_matrix1


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


# Решение системы
initial_state12 = get_initial_state(image_array)    # Начальные условия

start_time = time.time()
encrypted_image = encrypt_image(image_array, initial_state12)
print(f"\nОбщее время выполнения кодирования: {time.time() - start_time:.4f} секунд")

# --------------------------------- Дешифровка ---------------------------------

start_time = time.time()
decrypted_image = decrypt_image(encrypted_image, initial_state12)
print(f"\nОбщее время выполнения декодирования: {time.time() - start_time:.4f} секунд")

# --------------------------------- Вывод изображений ---------------------------------

print_encryption_result(image_array, encrypted_image, decrypted_image)