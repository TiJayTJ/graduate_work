import numpy as np
from matplotlib import pyplot as plt

from encrypt_decrypt import encrypt_image, decrypt_image


def histogram_analise(image, encoded_image):
    # Строим гистограммы
    plt.figure(figsize=(15, 3))

    # Показываем исходное изображение
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.axis('off')

    # Гистограмма исходного изображения
    plt.subplot(1, 4, 2)
    for i, color in enumerate(['red', 'green', 'blue']):
        hist_values, _ = np.histogram(image[:, :, i], bins=256, range=(0, 256))
        plt.bar(range(256), hist_values, color=color, alpha=0.7, width=1.0, label=color)
    plt.xlabel('Количество серого')
    plt.ylabel('Количество пикселей')
    plt.legend()

    # Показываем зашифрованное изображение
    plt.subplot(1, 4, 3)
    plt.imshow(encoded_image)
    plt.axis('off')

    # Гистограмма зашифрованного изображения
    plt.subplot(1, 4, 4)
    for i, color in enumerate(['red', 'green', 'blue']):
        hist_values, _ = np.histogram(encoded_image[:, :, i], bins=256, range=(0, 256))
        plt.bar(range(256), hist_values, color=color, alpha=0.7, width=1.0, label=color)
    plt.xlabel('Количество серого')
    plt.ylabel('Количество пикселей')
    plt.legend()

    plt.tight_layout()
    plt.show()


def analise_key_sensitivity(image_array, initial_state12):
    img_col, img_row, _ = image_array.shape
    encrypted_image = encrypt_image(image_array, initial_state12)
    decrypted_images = np.empty((3, img_row, img_col, 3), dtype=np.uint8)
    for i in range(3):
        new_initial_state = [0.0]*6
        for j in range(6):
            new_initial_state[j] = initial_state12[j]
        new_initial_state[i] += 10**(-14)
        decrypted_images[i] = decrypt_image(encrypted_image, new_initial_state)
    decrypted_image = decrypt_image(encrypted_image, initial_state12)

    # Создание фигуры и осей
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0][0].imshow(image_array)
    axes[0][0].axis("off")
    axes[0][0].set_title("Исходное изображение")

    axes[0][1].imshow(encrypted_image)
    axes[0][1].axis("off")
    axes[0][1].set_title("Зашифрованное изображение, исхожный ключ")

    axes[0][2].imshow(decrypted_images[0])
    axes[0][2].axis("off")
    axes[0][2].set_title("Расшифрованное изображение, ключ 1")

    axes[1][0].imshow(decrypted_images[1])
    axes[1][0].axis("off")
    axes[1][0].set_title("Расшифрованное изображение, ключ 2")

    axes[1][1].imshow(decrypted_images[2])
    axes[1][1].axis("off")
    axes[1][1].set_title("Расшифрованное изображение, ключ 3")

    axes[1][2].imshow(decrypted_image)
    axes[1][2].axis("off")
    axes[1][2].set_title("Расшифрованное изображение, исходный ключ")

    plt.show()
