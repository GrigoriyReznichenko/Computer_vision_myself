import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()  # Создает объект захвата лица
predictor = dlib.shape_predictor(
    'shape_predictor_68_face_landmarks.dat')  # Нейросетевое прогнозирование 68 face landmarks

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Захватывает координаты bbox 2 и более лиц. Поэтому далее цикл
    rects = detector(gray_frame, 1)

    for rect in rects:
        shape = predictor(gray_frame, rect)  # создать объект с landmarks
        landmarks = np.zeros((68, 2), dtype='int')

        for i in range(68):
            landmarks[i] = (shape.part(i).x, shape.part(i).y)  # По индексу обращаемся к x, y координатам

        # Отрисовываем круги в точках с радиусом 1
        for i, (x, y) in enumerate(landmarks):
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 113:
        cv2.destroyAllWindows()
        break
