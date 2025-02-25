import numpy as np
import hashlib
from PIL import Image
from scipy.integrate import solve_ivp
from skimage import data
from matplotlib import pyplot as plt

from dna import byte_to_dna, dna_add, dna_sub, dna_xor, dna_to_byte
from graphs_output import print_phase_space
from system import main_system


def sha512hash(Password):
    HashedPassword = hashlib.sha512(Password.encode('utf-8')).hexdigest()
    return HashedPassword

def get_initial_state(image):
    # Приведение изображение к тектовому виду
    image_str = np.array2string(image)
    image_hash = sha512hash(image_str)[:128]
    h32 = [0]*32
    for i in range(0, 128, 4):
        h32[i // 4] = int(image_hash[i:i+4], 16)
    keys = [0]*6
    for i in range(6):
        h_sum5 = 0
        for j in range(5):
            h_sum5 += h32[i*5 + j]
        h_sum5 = int(h_sum5 / 5) % 256
        keys[i] = h_sum5
    return keys


def array_to_matrix(array):
    x_val, y_val, z_val = array

    # Подгоняем размер под изображение
    x_val_converted = x_val[:512*512]
    y_val_converted = y_val[:512*512]
    z_val_converted = z_val[:512*512]

    # Преобразования решения в матрицу
    for i in range(512*512):
        x_val_converted[i] = int(x_val_converted[i]) % 256
        y_val_converted[i] = int(y_val_converted[i]) % 256
        z_val_converted[i] = int(z_val_converted[i]) % 256

    matrix_a = np.empty((512, 512, 3), dtype=np.uint8)
    for i in range(512*512):
        matrix_a[i // 512][i % 512][0] = x_val_converted[i]
        matrix_a[i // 512][i % 512][1] = y_val_converted[i]
        matrix_a[i // 512][i % 512][2] = z_val_converted[i]

    return matrix_a


def uint8_matrix_to_dna(matrix):
    img_row_size = len(matrix)
    img_col_size = len(matrix[0])
    dna_matrix = np.empty((img_row_size, img_col_size, 3), dtype=object)

    for i in range(img_row_size):
        for j in range(img_col_size):
            for k in range(3):
                dna_matrix[i, j, k] = byte_to_dna(matrix[i, j, k])

    return dna_matrix


# Загружаем изображение
image = data.astronaut()
img_col_size, img_row_size, _ = image.shape
image_array = np.array(image)

# Показываем изображение
plt.imshow(image)
plt.axis('off')
# plt.show()

# Решение системы
initial_state = get_initial_state(image_array)[:3]  # Начальные условия
solution = solve_ivp(
        main_system, [0, 1200], initial_state, t_eval=np.linspace(1000, 1200, 512*512)
    )

# Получаем матрицу для шифрования изображения
matrix_a = array_to_matrix(solution.y)

# Применение ДНК кодирования
dna_image = uint8_matrix_to_dna(image_array)
dna_matrix_a = uint8_matrix_to_dna(matrix_a)


def dna_diffusion(dna_matrix1, dna_matrix2):
    # Складываем ДНК основания по правилам
    dna_encoded_image = np.empty((512, 512, 3), dtype=object)
    for i in range(img_row_size):
        for j in range(img_col_size):
            dna_encoded_image[i, j, 0] = dna_add(dna_matrix1[i, j, 0], dna_matrix2[i, j, 0])
            dna_encoded_image[i, j, 1] = dna_sub(dna_matrix1[i, j, 1], dna_matrix2[i, j, 1])
            dna_encoded_image[i, j, 2] = dna_xor(dna_matrix1[i, j, 2], dna_matrix2[i, j, 2])

    return dna_encoded_image


# Складываем ДНК основания по правилам
dna_encoded_image = dna_diffusion(dna_image, dna_matrix_a)

def dna_matrix_to_uint8(matrix):
    img_row_size = len(matrix)
    img_col_size = len(matrix[0])
    dna_matrix = np.zeros((img_row_size, img_col_size, 3), dtype=np.uint8)

    for i in range(img_row_size):
        for j in range(img_col_size):
            for k in range(3):  # R, G, B
                dna_matrix[i, j, k] = dna_to_byte(matrix[i, j, k])

    return dna_matrix


# Заполняем матрицу, преобразуя ДНК обратно в байты
binary_image = dna_matrix_to_uint8(dna_encoded_image)

# --- Отображаем восстановленное изображение ---
plt.imshow(binary_image)
plt.axis("off")
plt.title("Зашифрованное изображение")
# plt.show()

# ------------------------- Дешифровка -------------------------
dna_image = uint8_matrix_to_dna(binary_image)

def dna_diffusion_reverse(dna_matrix1, dna_matrix2):
    # Складываем ДНК основания по правилам
    dna_encoded_matrix = np.empty((512, 512, 3), dtype=object)
    for i in range(img_row_size):
        for j in range(img_col_size):
            dna_encoded_matrix[i, j, 0] = dna_sub(dna_matrix1[i, j, 0], dna_matrix2[i, j, 0])
            dna_encoded_matrix[i, j, 1] = dna_add(dna_matrix1[i, j, 1], dna_matrix2[i, j, 1])
            dna_encoded_matrix[i, j, 2] = dna_xor(dna_matrix1[i, j, 2], dna_matrix2[i, j, 2])
    return dna_encoded_matrix


# Складываем ДНК основания по правилам
dna_encoded_image = dna_diffusion_reverse(dna_image, dna_matrix_a)
binary_image = dna_matrix_to_uint8(dna_encoded_image)

# --- Отображаем восстановленное изображение ---
plt.imshow(binary_image)
plt.axis("off")
plt.title("Восстановленное изображение из ДНК")
plt.show()

