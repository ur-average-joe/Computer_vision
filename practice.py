import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True: #open webcam
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break