import cv2
import sys
import numpy as np

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

source = cv2.VideoCapture(s)

win_name = 'Camera Preview'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

while cv2.waitKey(1) != 27: # Escape
    has_frame, frame = source.read()
    if not has_frame:
        break
    matrix = np.ones(frame.shape, dtype="uint8") * 255
    frame = cv2.flip(frame, 1)

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # img_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
    # retval, img_thresh = retval, img_thresh = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY)

    img_NZ_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    negative = cv2.subtract(matrix, img_NZ_rgb)
    cv2.imshow(win_name, negative)

source.release()
cv2.destroyWindow(win_name)