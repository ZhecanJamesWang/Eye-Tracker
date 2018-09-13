
import numpy as np
import cv2

cam=cv2.VideoCapture(0)

while 1:

    ret, draw=cam.read()
    # draw = cv2.resize(draw, (960, 540))

    cv2.namedWindow('Face Detection',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Face Detection', 1920, 1080)
    cv2.imshow('Face Detection',draw)
    cv2.waitKey(1)
