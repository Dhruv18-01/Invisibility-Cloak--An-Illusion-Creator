import cv2
import time

cap = cv2.VideoCapture(0)
time.sleep(3)  # wait for camera to adjust

for i in range(30):
    ret, background = cap.read()
background = cv2.flip(background, 1)
cv2.imwrite('background.jpg', background)

cap.release()
cv2.destroyAllWindows()
