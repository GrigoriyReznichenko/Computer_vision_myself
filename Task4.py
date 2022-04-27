from generator import super_generaton
import numpy as np
import cv2
from cv2 import HoughLinesP

""""
АЛГОРИТМ
________
1) Изображение неточечное, поэтому получается толстые(точнее множество рядом) прямых Хаффа
2) Из полученного изображения извлекаем контуры
3) Ищем по контуру минимальные прямоугольники, которые огибают строчечные оксиды

"""

# Генерируем изображение из соседнего файла .py
src = super_generaton()


# Создаем трекбары для преобразования Хаффа
def nothing(arg):
    pass


cv2.namedWindow('img')
cv2.createTrackbar('rho', 'img', 1, 5, nothing)
cv2.createTrackbar('theta', 'img', 180, 180, nothing)
cv2.createTrackbar('threshold', 'img', 200, 500, nothing)
cv2.createTrackbar('minLineLength', 'img', 50, 500, nothing)
cv2.createTrackbar('maxLineGap', 'img', 100, 500, nothing)
cv2.createTrackbar('w', 'img', 0, 100, nothing)
cv2.createTrackbar('h', 'img', 0, 100, nothing)


# Функция строящая прямые по алгоритму Хаффа
def get_lines(linesP):
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(empty, (l[0], l[1]), (l[2], l[3]), 255, 3, cv2.LINE_AA)


# Функция строит прямоугольники огибающие контур с минимальной площадью
def get_min_area_rects(contours, add_w, add_h):
    for contour in contours:
        point, (w, h), angle = cv2.minAreaRect(contour)
        rect = (point, (w + add_w, h + add_h), angle)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, (255, 0, 0), 3)


while True:
    empty = np.zeros(src.shape, dtype='uint8')  # Пустое изображение хранит прямые Хаффа
    # Извлекаем из трекбаров данные в переменные
    rho = cv2.getTrackbarPos('rho', 'img')
    if rho < 1:
        rho = 1
    theta_num = cv2.getTrackbarPos('theta', 'img')
    threshold = cv2.getTrackbarPos('threshold', 'img')
    min_line_length = cv2.getTrackbarPos('minLineLength', 'img')
    max_line_gap = cv2.getTrackbarPos('maxLineGap', 'img')
    add_w = cv2.getTrackbarPos('w', 'img')
    add_h = cv2.getTrackbarPos('h', 'img')

    # Загружаем изображение в новую область памяти
    img = src.copy()

    # Преобразование Хаффа для детектирования линий
    linesP = HoughLinesP(img,  # Переменная, хранящее исходное изображение
                         rho=rho,  # Разрешение в пикселях по длине
                         theta=np.pi / theta_num,  # Разрешение в углах
                         threshold=threshold,  # Порог определяющий максимумы в аккумуляторной матрицы Хаффа
                         minLineLength=min_line_length,  # Минимальная Евклидова длина прямой
                         maxLineGap=max_line_gap)  # Максимальное расстояние между точками одной прямой

    # Отметить прямые на картинке и прорисовать
    get_lines(linesP)

    # Отобразить
    contours, hierarhy = cv2.findContours(empty, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Чтобы построить цветые прямоугольники
    get_min_area_rects(contours, add_w, add_h)  # Строим эти прямоугольники
    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)  # Подгоняем размер изображения
    cv2.imshow('img', img)  # строим

    if cv2.waitKey(1) == 27:
        break
