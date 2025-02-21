import numpy as np
from PIL import Image
from skimage import data
from matplotlib import pyplot as plt


# Функция преобразования 2-битных пар в ДНК основания
def bits_to_dna(bits):
    mapping = {"00": "A", "01": "G", "10": "C", "11": "T"}
    return mapping[bits]


# Функция разбиения байта (8 бит) на 4 пары битов и их преобразование в ДНК
def byte_to_dna(byte):
    binary_str = format(byte, "08b")  # Преобразуем байт в 8-битную строку
    return [bits_to_dna(binary_str[i:i+2]) for i in range(0, 8, 2)]


# Функция сложения оснований ДНК
def dna_add(base1, base2):
    addition_rules = {
        ("A", "A"): "A", ("A", "G"): "G", ("A", "C"): "C", ("A", "T"): "T",
        ("G", "G"): "C", ("G", "C"): "T", ("G", "T"): "A",
        ("C", "C"): "A", ("C", "T"): "G",
        ("T", "T"): "C"
    }
    return [addition_rules.get((base1[i], base2[i])) or addition_rules.get((base2[i], base1[i])) for i in range(4)]


# Загружаем изображение
image = data.astronaut()
img_col_size, img_row_size, _ = image.shape
image_array = np.array(image)

# Показываем изображение
plt.imshow(image)
plt.axis('off')
# plt.show()

# Преобразование в двоичную запись
dna_image = np.empty((512, 512, 3), dtype=object)
id_dna_image = np.empty((512, 512, 3), dtype=object)
for i in range(img_col_size):
    for j in range(img_row_size):
        for k in range(3):
            dna_image[i, j, k] = byte_to_dna(image[i, j, k])
            id_dna_image[i, j, k] = byte_to_dna(50)

# Складываем ДНК основания по правилам
dna_sum = np.empty((512, 512, 3), dtype=object)
for i in range(img_col_size):
    for j in range(img_row_size):
        for k in range(3):
            dna_sum[i, j, k] = dna_add(dna_image[i, j, k], id_dna_image[i, j, k])

# Таблица обратного преобразования ДНК → 2-битные пары
dna_to_bits = {"A": "00", "G": "01", "C": "10", "T": "11"}
# Создаем пустую матрицу для двоичных данных
binary_image = np.zeros((512, 512, 3), dtype=np.uint8)

# Заполняем матрицу, преобразуя ДНК обратно в байты
for i in range(min(512, img_col_size)):
    for j in range(min(512, img_row_size)):
        for k in range(3):  # R, G, B
            binary_str = "".join([dna_to_bits[base] for base in dna_sum[i, j, k]])  # Соединяем 4 пары в 8-битное число
            binary_image[i, j, k] = int(binary_str, 2)  # Переводим в десятичное значение

# --- Отображаем восстановленное изображение ---
plt.imshow(binary_image)
plt.axis("off")
plt.title("Восстановленное изображение из ДНК")
plt.show()

# Вывод изображения
# plt.imshow(decimal_image)
# plt.axis("off")
# plt.show()