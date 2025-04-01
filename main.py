import numpy as np
from PIL import Image
from scipy.integrate import solve_ivp
from scipy.stats import pearsonr
from skimage import data
from matplotlib import pyplot as plt

from encrypt_decrypt import encrypt_image, decrypt_image
from encryption_analysis import histogram_analise, analyse_correlation, analysis_information_entropy, \
    analyse_noise_attacks, analyse_cropping_attacks
from printers import print_encryption_result, print_image
from system import get_initial_state

# Загружаем изображение
astronaut = data.astronaut()
lena = Image.open("images/lena.png")
baboon = Image.open("images/baboon.png")
image_array = np.array(lena)
img_row, img_col, _ = image_array.shape

# --------------------------------- Шифрование ---------------------------------

# Решение системы
initial_state12 = get_initial_state(image_array)    # Начальные условия

encrypted_image = encrypt_image(image_array, initial_state12)

# --------------------------------- Численныйт анализ ---------------------------------

# histogram_analise(image_array, encrypted_image)

# analise_key_sensitivity(image_array, initial_state12)

# print('Коэффициенты корреляции исходного изображения')
# analyse_correlation(image_array)
# print('\n-------------------------------------------------------------------------------\n')
# print('Коэффициенты корреляции зашифрованного изображения')
# analyse_correlation(encrypted_image)

# print('Информационная энтропия исходного изображения')
# analysis_information_entropy(image_array)
# print('-------------------------------------------------------------------------------')
# print('Информационная энтропия зашифрованного изображения:')
# analysis_information_entropy(encrypted_image)
# print()

noised_image = analyse_noise_attacks(encrypted_image, 0.15)
noised_decryted_image = decrypt_image(noised_image, initial_state12)
print_encryption_result(image_array, noised_image, noised_decryted_image)

# cropped_image = analyse_cropping_attacks(encrypted_image, 25)
# cropped_decryted_image = decrypt_image(cropped_image, initial_state12)
# print_encryption_result(image_array, cropped_image, cropped_decryted_image)

# --------------------------------- Дешифровка ---------------------------------

# decrypted_image = decrypt_image(encrypted_image, initial_state12)

# --------------------------------- Вывод изображений ---------------------------------

# print_encryption_result(image_array, encrypted_image, decrypted_image)