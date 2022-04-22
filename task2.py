import cv2
import numpy as np
import matplotlib.pyplot as plt


def nothing(agrs):
    pass


""""---------------------------------Создаем трекбары---------------------------------"""
# trackbarName - Имя трекбара
# windowName - Имя окна с трекбаром
# value - Минимальное значение
# count - Максимальное значение
# onChange - При движении трекбара вызвать функцию
cv2.namedWindow('pic')
cv2.createTrackbar('min hue', 'pic', 160, 180, nothing)
cv2.createTrackbar('max hue', 'pic', 180, 180, nothing)
cv2.createTrackbar('min saturation', 'pic', 0, 255, nothing)
cv2.createTrackbar('max saturation', 'pic', 255, 255, nothing)
cv2.createTrackbar('min val', 'pic', 0, 255, nothing)
cv2.createTrackbar('max val', 'pic', 255, 255, nothing)
cv2.createTrackbar('new hue', 'pic', 120, 180, nothing)
cv2.createTrackbar('new saturation', 'pic', 220, 255, nothing)
cv2.createTrackbar('new val', 'pic', 170, 255, nothing)

# Захватываем видео
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)  # Конвертируем в HSV здесь(чтобы не было записи)

    # Считываем данные трекбаров
    h1 = cv2.getTrackbarPos('min hue', 'pic')
    h2 = cv2.getTrackbarPos('max hue', 'pic')
    s1 = cv2.getTrackbarPos('min saturation', 'pic')
    s2 = cv2.getTrackbarPos('max saturation', 'pic')
    v1 = cv2.getTrackbarPos('min val', 'pic')
    v2 = cv2.getTrackbarPos('max val', 'pic')
    new_hue = cv2.getTrackbarPos('new hue', 'pic')
    new_sat = cv2.getTrackbarPos('new saturation', 'pic')
    new_val = cv2.getTrackbarPos('new val', 'pic')

    MIN_VAL = (h1, s1, v1)
    MAX_VAL = (h2, s2, v2)
    VAL = (new_hue, new_sat, new_val)

    mask = cv2.inRange(hsv, MIN_VAL, MAX_VAL)  # Формирует маску внутри диапазонов
    # Маска равна 255 если ограниченное HUE подпространство лежит внутри MIN_VAL и MAX_VAL
    hsv[mask == 255] = VAL

    res = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    cv2.imshow('pic', res)
    if cv2.waitKey(1) == 27:  # Выход по esc
        break
cv2.destroyAllWindows()
