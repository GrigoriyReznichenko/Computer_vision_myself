import cv2
import numpy as np

model_file = 'opencv_face_detector.pbtxt'  # Алгоритм извлечения ответов
cfg_file = 'opencv_face_detector_uint8.pb'  # Конфигурация нейросети
net = cv2.dnn.readNetFromTensorflow(cfg_file, model_file)  # Создаем объект net для инференса

cap = cv2.VideoCapture(0)


# Создает трекбар для управления порогом, в терминах вероятности.
# Если значение выше conf_thresold то принимается решения об отрисовке boundbox
def nothing(arg):
    pass


cv2.namedWindow('webcam')
cv2.createTrackbar('conf_threshold % from 1', 'webcam', 80, 100, nothing)

while True:
    _, frame = cap.read()
    # Ивзлекаем длину и ширину окна для декартовой системы
    #       ----------> x
    #       |
    #       |
    #       |
    #       \/
    #       y
    frame_length, frame_width = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).shape

    # Нормирование изображения
    blob = cv2.dnn.blobFromImage(frame,  # Что нормируем
                                 1.0,  # Взвешивающий пописксельный фактор
                                 (300, 300),  # Размерность изображения с которыми работает нейросеть(шкалирование)
                                 [104, 117, 123],  # Нормирование по среднему для RGB
                                 False,  # Не конвертируем RGB -> BGR
                                 False)  # Не образаем после resize

    # Извлекаем трешолд от ручки
    conf_threshold = cv2.getTrackbarPos('conf_threshold % from 1', 'webcam') / 100

    net.setInput(blob)  # Вход нейросети
    detections = net.forward()  # Выход нейросети
    print(detections.shape[2])
    bboxes = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]  # В этой координате храниться значение вероятности
        if conf > conf_threshold:  # Если оно больше порога
            # Извлекаем точки диагонали boundbox, нормированное к длине и ширине(точки (0, 1) множества)
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_length)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_length)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

    cv2.imshow('webcam', frame)

    if cv2.waitKey(1) == 113: # Завершить по q
        cv2.destroyAllWindows()
        break
