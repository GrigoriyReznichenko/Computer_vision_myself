"""
Нерезкое маскирование
1) Сглаживаем исходное изображение
2) Инвертируем
3) Добавляем взвешенную версию к исходному (попиксельно)
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2

path = r'/home/grigoriy/Data science/Tasks/Task3/WienerImageDeblurringExample_02.png'
src = cv2.imread(path)


def nothing(arg):
    pass

cv2.namedWindow('pic')
cv2.createTrackbar('Длина ядра', 'pic', 1, 50, nothing)
cv2.createTrackbar('Ширина ядра', 'pic', 1, 50, nothing)
cv2.createTrackbar('СКО по длине', 'pic', 1, 100, nothing)
cv2.createTrackbar('СКО по ширине', 'pic', 1, 10000, nothing)
cv2.createTrackbar('Вес', 'pic', 1, 10000, nothing)
cv2.createTrackbar('Сдвиг', 'pic', 1, 100, nothing)

while True:

    img = src.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    l = 2 * cv2.getTrackbarPos('Длина ядра', 'pic') + 1
    w = 2 * cv2.getTrackbarPos('Ширина ядра', 'pic') + 1
    sx = cv2.getTrackbarPos('СКО по длине', 'pic')
    sy = cv2.getTrackbarPos('СКО по ширине', 'pic')
    alpha = cv2.getTrackbarPos('Вес', 'pic') / 10000
    gamma = cv2.getTrackbarPos('Сдвиг', 'pic')

    blurred = cv2.GaussianBlur(img, (l, w), sx, sy)
    inv_blurred = 255 - blurred
    dst = cv2.addWeighted(img, 1 - alpha, inv_blurred, alpha, gamma) # f2(x, y) = (1-a) * f0(x, y) + a * f1(x, y)

    cv2.imshow('pic', dst)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
## ВЫВОД: Для слабого деблюра