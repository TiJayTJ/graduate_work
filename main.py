import numpy as np
from scipy.integrate import solve_ivp
from skimage import data
from matplotlib import pyplot as plt

from encrypt_decrypt import encrypt_image, decrypt_image
from system import get_initial_state

# Загружаем изображение
image = data.astronaut()
# image = Image.open("lena.png")
image_array = np.array(image)
img_col, img_row, _ = image_array.shape

# --------------------------------- Шифрование ---------------------------------

# Решение системы
initial_state12 = get_initial_state(image)    # Начальные условия

encrypted_image = encrypt_image(image_array, initial_state12)

# --------------------------------- Численныйт анализ ---------------------------------

# histogram_analise(image_array, encrypted_image)

# analise_key_sensitivity(image_array, initial_state12)

# --------------------------------- Дешифровка ---------------------------------

# decrypted_image = decrypt_image(encrypted_image, initial_state12)

# --------------------------------- Вывод изображений ---------------------------------

# print_three_img(image, encrypted_image, decrypted_image)