import numpy as np
from PIL import Image
from scipy.integrate import solve_ivp
from skimage import data
from matplotlib import pyplot as plt

from graphs_output import print_phase_space
from system import main_system


# Функция преобразования 2-битных пар в ДНК основания
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

# Загружаем изображение
image = data.astronaut()
img_col_size, img_row_size, _ = image.shape
image_array = np.array(image)

# Показываем изображение
plt.imshow(image)
plt.axis('off')
plt.show()

# Решение системы
initial_state = [1.0, 1.0, 1.0]  # Начальные условия
solution = solve_ivp(
        main_system, [0, 1200], initial_state, t_eval=np.linspace(1000, 1200, 512*512)
    )
x_val, y_val, z_val = solution.y
# print_phase_space(x_values, y_values, z_values)

x_val_converted = x_val[:512*512]
y_val_converted = y_val[:512*512]
z_val_converted = z_val[:512*512]

# Преобразования решения
for i in range(512*512):
    x_val_converted[i] = int(x_val_converted[i]) % 256
    y_val_converted[i] = int(y_val_converted[i]) % 256
    z_val_converted[i] = int(z_val_converted[i]) % 256

matrix_A = np.empty((512, 512, 3), dtype=np.uint8)
for i in range(512*512):
    matrix_A[i // 512][i % 512][0] = x_val_converted[i]
    matrix_A[i // 512][i % 512][1] = y_val_converted[i]
    matrix_A[i // 512][i % 512][2] = z_val_converted[i]

print(matrix_A)

# Применение ДНК кодирования
dna_image = np.empty((512, 512, 3), dtype=object)
dna_matrix_A = np.empty((512, 512, 3), dtype=object)
for i in range(img_col_size):
    for j in range(img_row_size):
        for k in range(3):
            dna_image[i, j, k] = byte_to_dna(image_array[i, j, k])
            dna_matrix_A[i, j, k] = byte_to_dna(matrix_A[i, j, k])

# Складываем ДНК основания по правилам
dna_encoded_image = np.empty((512, 512, 3), dtype=object)
for i in range(img_col_size):
    for j in range(img_row_size):
        dna_encoded_image[i, j, 0] = dna_add(dna_image[i, j, 0], dna_matrix_A[i, j, 0])
        dna_encoded_image[i, j, 1] = dna_sub(dna_image[i, j, 1], dna_matrix_A[i, j, 1])
        dna_encoded_image[i, j, 2] = dna_xor(dna_image[i, j, 2], dna_matrix_A[i, j, 2])

# Заполняем матрицу, преобразуя ДНК обратно в байты
binary_image = np.zeros((512, 512, 3), dtype=np.uint8)
for i in range(min(512, img_col_size)):
    for j in range(min(512, img_row_size)):
        for k in range(3):  # R, G, B
            binary_image[i, j, k] = dna_to_byte(dna_encoded_image[i, j, k])

# --- Отображаем восстановленное изображение ---
plt.imshow(binary_image)
plt.axis("off")
plt.title("Восстановленное изображение из ДНК")
plt.show()