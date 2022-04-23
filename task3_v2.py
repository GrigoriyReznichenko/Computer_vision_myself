import matplotlib.pyplot as plt
from skimage.restoration import wiener
from scipy import signal
import numpy as np
import cv2


cap = cv2.VideoCapture(0)
ret, frame = cap.read()
n, m, k = frame.shape



def nothing(arg):
    pass


cv2.namedWindow('webcam')
cv2.createTrackbar('kerlen', 'webcam', 1, 100, nothing)
cv2.createTrackbar('std', 'webcam', 1, 1000, nothing)
cv2.createTrackbar('balance', 'webcam', 0, 2000, nothing)

while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    kerlen = 2 * cv2.getTrackbarPos('kerlen', 'webcam') + 1
    std = cv2.getTrackbarPos('std', 'webcam')
    balance = (cv2.getTrackbarPos('balance', 'webcam') - 2500) / 5000

    gauss_1d = signal.gaussian(kerlen, std).reshape(kerlen, 1)
    gauss_2d = np.outer(gauss_1d, gauss_1d)



    img = wiener(frame, gauss_2d, balance)
    print(img)
    cv2.imshow('webcam', img)

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        break
