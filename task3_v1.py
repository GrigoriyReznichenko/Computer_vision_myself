import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import unsupervised_wiener
from scipy.signal import convolve2d as conv2

path = r'/home/grigoriy/Data science/Tasks/Task3/img3.jpg'
src = cv2.imread(path, cv2.IMREAD_GRAYSCALE)


class LenError(SystemError):
    pass


class Filters:

    def __init__(self, n, m):
        Filters.__validate_len(self, n)
        Filters.__validate_len(self, m)
        self.n = n
        self.m = m

    def get_box_filter(self):
        return np.ones(self.n, self.n) / (self.n * self.m)

    def get_gauss_kernel(self, sigma):
        n, m = self.n // 2, self.m // 2
        x, y = np.arange(-n, n + 1), np.arange(-m, m + 1)
        x, y = np.meshgrid(x, y)
        return 1 / (2 * np.pi * np.square(sigma)) * np.exp(
            -0.5 * (np.square(x) + np.square(y)) / np.square(sigma)) * (1 / self.n * self.m)

    def __validate_len(self, x):
        if x % 2 == 0:
            raise ValueError


# Конвертируем ко float64 и сужаем диапазон
f_xy = src.astype(np.float64) / 255

# Ядро блюр-фильтра
obj = Filters(23, 23)
h_xy = obj.get_gauss_kernel(100)

# Моделируем искажение
g_xy = conv2(f_xy, h_xy)# Свертка

# Деконволюция Винера-Ханта c автоматическим подбором параметров
f_hat_xy, _ = unsupervised_wiener(g_xy, h_xy)


plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(g_xy[10:-11, 10:-11], 'gray')
plt.subplot(1, 2, 2)
plt.imshow(f_hat_xy[10:-11, 10:-11], 'gray')
plt.show()
