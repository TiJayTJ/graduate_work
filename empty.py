import hashlib

import numpy as np
from PIL import Image
from scipy.integrate import solve_ivp
from scipy.stats import pearsonr
from matplotlib import pyplot as plt

from encrypt_decrypt import decrypt_image
from encryption_analysis import histogram_analise, analyse_correlation, analysis_information_entropy, \
    analyse_noise_attacks, analyse_cropping_attacks
from printers import print_encryption_result, print_image


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
                mixed_matrix[((counter * 2) // n) * 2][(counter * 2) % n][k] = matrix[i][j][k]
                mixed_matrix[((counter * 2) // n) * 2][(counter * 2) % n + 1][k] = matrix[j][n - i - 1][(k + 1) % 3]
                mixed_matrix[((counter * 2) // n) * 2 + 1][(counter * 2) % n + 1][k] = matrix[n - i - 1][n - j - 1][(k + 2) % 3]
                mixed_matrix[((counter * 2) // n) * 2 + 1][(counter * 2) % n][k] = matrix[n - j - 1][i][k]
                counter += 1
    return mixed_matrix


# Функция преобразования 2-битных пар в ДНК основания
import numpy as np


def bits_to_dna(bits):
    mapping = {"00": "A", "01": "G", "10": "C", "11": "T"}
    return mapping[bits]


# Функция преобразования ДНК основания в 2-битных пар
def dna_to_bits(dna):
    mapping = {"A": "00", "G": "01", "C": "10", "T": "11"}
    return mapping[dna]


# Функция разбиения байта (8 бит) на 4 пары битов и их преобразование в ДНК
def byte_to_dna(byte):
    binary_str = format(byte, "08b")  # Преобразуем байт в 8-битную строку
    return [bits_to_dna(binary_str[i:i+2]) for i in range(0, 8, 2)]


def dna_to_byte(dna):
    binary_str = "".join([dna_to_bits(base) for base in dna])  # Соединяем 4 пары в 8-битное число
    return int(binary_str, 2)  # Переводим в десятичное значение


def uint8_matrix_to_dna(matrix):
    matrix_row, matrix_col, _ = matrix.shape
    dna_matrix = np.empty((matrix_row, matrix_col, 3), dtype=object)

    for i in range(matrix_row):
        for j in range(matrix_col):
            for k in range(3):
                dna_matrix[i, j, k] = byte_to_dna(matrix[i, j, k])

    return dna_matrix


def dna_matrix_to_uint8(matrix):
    matrix_row, matrix_col, _ = matrix.shape
    dna_matrix = np.zeros((matrix_row, matrix_col, 3), dtype=np.uint8)

    for i in range(matrix_row):
        for j in range(matrix_col):
            for k in range(3):  # R, G, B
                dna_matrix[i, j, k] = dna_to_byte(matrix[i, j, k])

    return dna_matrix


# Функция сложения оснований ДНК
def dna_add(base1, base2):
    addition_rules = {
        ("A", "A"): "A", ("A", "G"): "G", ("A", "C"): "C", ("A", "T"): "T",
        ("G", "G"): "C", ("G", "C"): "T", ("G", "T"): "A",
        ("C", "C"): "A", ("C", "T"): "G",
        ("T", "T"): "C"
    }
    return [addition_rules.get((base1[i], base2[i])) or addition_rules.get((base2[i], base1[i])) for i in range(4)]


# Функция xor оснований ДНК
def dna_xor(base1, base2):
    addition_rules = {
        ("A", "A"): "A", ("A", "G"): "G", ("A", "C"): "C", ("A", "T"): "T",
        ("G", "G"): "A", ("G", "C"): "T", ("G", "T"): "C",
        ("C", "C"): "A", ("C", "T"): "G",
        ("T", "T"): "A"
    }
    return [addition_rules.get((base1[i], base2[i])) or addition_rules.get((base2[i], base1[i])) for i in range(4)]


# Функция вычитания оснований ДНК
def dna_sub(base1, base2):
    addition_rules = {
        ("A", "A"): "A", ("A", "G"): "T", ("A", "C"): "C", ("A", "T"): "G",
        ("G", "A"): "G", ("G", "G"): "A", ("G", "C"): "T", ("G", "T"): "C",
        ("C", "A"): "C", ("C", "G"): "G", ("C", "C"): "A", ("C", "T"): "T",
        ("T", "A"): "T", ("T", "G"): "C", ("T", "C"): "G", ("T", "T"): "A"
    }
    return [addition_rules.get((base1[i], base2[i])) for i in range(4)]


def dna_diffusion(dna_matrix1, dna_matrix2):
    matrix_row, matrix_col, _ = dna_matrix1.shape
    # Складываем ДНК основания по правилам
    dna_encoded_image = np.empty((matrix_row, matrix_col, 3), dtype=object)
    for i in range(matrix_row):
        for j in range(matrix_col):
            dna_encoded_image[i, j, 0] = dna_add(dna_matrix1[i, j, 0], dna_matrix2[i, j, 0])
            dna_encoded_image[i, j, 1] = dna_sub(dna_matrix1[i, j, 1], dna_matrix2[i, j, 1])
            dna_encoded_image[i, j, 2] = dna_xor(dna_matrix1[i, j, 2], dna_matrix2[i, j, 2])

    return dna_encoded_image


def dna_diffusion_reverse(dna_matrix1, dna_matrix2):
    matrix_row, matrix_col, _ = dna_matrix1.shape
    # Складываем ДНК основания по правилам
    dna_encoded_matrix = np.empty((matrix_row, matrix_col, 3), dtype=object)
    for i in range(matrix_row):
        for j in range(matrix_col):
            dna_encoded_matrix[i, j, 0] = dna_sub(dna_matrix1[i, j, 0], dna_matrix2[i, j, 0])
            dna_encoded_matrix[i, j, 1] = dna_add(dna_matrix1[i, j, 1], dna_matrix2[i, j, 1])
            dna_encoded_matrix[i, j, 2] = dna_xor(dna_matrix1[i, j, 2], dna_matrix2[i, j, 2])
    return dna_encoded_matrix


def dna_xor_diffusion(dna_matrix1, dna_matrix2):
    matrix_row, matrix_col, _ = dna_matrix1.shape
    # Складываем ДНК основания по правилам
    dna_encoded_image = np.empty((matrix_row, matrix_col, 3), dtype=object)
    for i in range(matrix_row):
        for j in range(matrix_col):
            dna_encoded_image[i, j, 0] = dna_xor(dna_matrix1[i, j, 0], dna_matrix2[i, j, 0])
            dna_encoded_image[i, j, 1] = dna_xor(dna_matrix1[i, j, 1], dna_matrix2[i, j, 1])
            dna_encoded_image[i, j, 2] = dna_xor(dna_matrix1[i, j, 2], dna_matrix2[i, j, 2])

    return dna_encoded_image


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

    # Получаем матрицы для шифрования изображения
    matrix_a = convert_for_encrypt(solution1.y, img_row, img_col)
    matrix_b = convert_for_encrypt(solution2.y, img_row, img_col)

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
    return dna_matrix_to_uint8(dna_encoded_image)


# Загружаем изображение
lena = Image.open("images/lena.png")
image_array = np.array(lena)
img_row, img_col, _ = image_array.shape

# --------------------------------- Шифрование ---------------------------------

# Решение системы
initial_state12 = get_initial_state(image_array)    # Начальные условия

encrypted_image = encrypt_image(image_array, initial_state12)

# --------------------------------- Численныйт анализ ---------------------------------

# Шумовые атаки

noised_image = analyse_noise_attacks(encrypted_image, 0.2)
noised_decryted_image = decrypt_image(noised_image, initial_state12)
print_encryption_result(image_array, noised_image, noised_decryted_image)