import cv2
import numpy as np
import time

# Step 1: Load background image
background = cv2.imread('background.jpg')

# Step 2: Start video capture
cap = cv2.VideoCapture(0)
time.sleep(2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for mirror effect
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Step 3: Create mask for red color
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    # Step 4: Refine the mask (remove noise)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    red_mask = cv2.dilate(red_mask, np.ones((3, 3), np.uint8), iterations=1)

    # Step 5: Create inverse mask
    inverse_mask = cv2.bitwise_not(red_mask)

    # Step 6: Segment the parts
    cloak_area = cv2.bitwise_and(background, background, mask=red_mask)
    non_cloak_area = cv2.bitwise_and(frame, frame, mask=inverse_mask)

    # Step 7: Combine both to get the final output
    final_output = cv2.addWeighted(cloak_area, 1, non_cloak_area, 1, 0)

    cv2.imshow("Invisibility Cloak", final_output)

    if cv2.waitKey(1) == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
