import cv2
import dlib
import argparse

detector = dlib.get_frontal_face_detector() # Создает объект захвата лица
predictor = dlib.shape_predictor()

cap = cv2.VideoCapture(0)


while True:
    _, frame = cap.read()

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 113:
        cv2.destroyAllWindows()
        break