import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Image Size: %d x %d" % (width, height))

bs = cv2.createBackgroundSubtractorKNN(detectShadows = True)

while True:
    key = cv2.waitKey(1) & 0xFF
    if  key == ord('q' ):
        cv2.destroyAllWindows()
        break
    else:
        frame_1 = cap.read()[1]
        frame_2 = cap.read()[1]

        detect_frame = cv2.absdiff(frame_1, frame_2)
        detect_frame = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)
        ret, detect_frame = cv2.threshold(detect_frame, 30, 255, cv2.THRESH_BINARY)

        fgmask = bs.apply(frame_2)
        cv2.imshow('detect_frame' ,detect_frame)
        cv2.imshow('KNN' ,fgmask)
