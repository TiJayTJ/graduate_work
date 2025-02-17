from skimage import data
from matplotlib import pyplot as plt

# Загружаем изображение
image = data.baboon()

# Показываем изображение
plt.imshow(image, cmap='viridis')
plt.show()