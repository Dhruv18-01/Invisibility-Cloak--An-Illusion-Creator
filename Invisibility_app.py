import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image

# Set page config
st.set_page_config(page_title="Invisibility Cloak", layout="centered")

# Title
st.title("🧥 Invisibility Cloak Illusion App")
st.markdown("Hold a cloth with the color you wish to turn invisible on camera!")

# Cloak color selector
cloak_color = st.selectbox(
    "Choose your cloak color:",
    ("Red", "Green", "Blue", "Yellow", "Black")
)

# Get HSV range based on selected color
def get_hsv_range(color):
    if color == "Red":
        lower1 = np.array([0, 120, 70])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 120, 70])
        upper2 = np.array([180, 255, 255])
    elif color == "Green":
        lower1 = np.array([36, 50, 70])
        upper1 = np.array([89, 255, 255])
        lower2 = None
        upper2 = None
    elif color == "Blue":
        lower1 = np.array([94, 80, 2])
        upper1 = np.array([126, 255, 255])
        lower2 = None
        upper2 = None
    elif color == "Yellow":
        lower1 = np.array([22, 93, 0])
        upper1 = np.array([45, 255, 255])
        lower2 = None
        upper2 = None
    elif color == "Black":
        lower1 = np.array([0, 0, 0])
        upper1 = np.array([180, 255, 30])
        lower2 = None
        upper2 = None
    return lower1, upper1, lower2, upper2

# Load background image once
def capture_background():
    cap = cv2.VideoCapture(0)
    st.info("Capturing background. Please move out of the frame.")
    time.sleep(3)  # Wait for camera to adjust

    for i in range(30):  # Capture multiple frames for clarity
        ret, bg_frame = cap.read()
        if ret:
            background = cv2.flip(bg_frame, 1)
    cap.release()
    cv2.imwrite("background.jpg", background)
    st.success("Background captured!")

# Capture background on button click
if st.button("Capture Background"):
    capture_background()

@st.cache_data
def load_background():
    return cv2.imread("background.jpg")

background = load_background()

# Button to start webcam invisibility
if st.button("Start Invisibility Effect"):
    cap = cv2.VideoCapture(0)
    time.sleep(2)

    FRAME_WINDOW = st.image([])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to read from webcam.")
            break

        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red color detection
        lower1, upper1, lower2, upper2 = get_hsv_range(cloak_color)

        mask2 = cv2.inRange(hsv, lower2, upper2) if lower2 is not None else 0
        mask1 = cv2.inRange(hsv, lower1, upper1)
        cloak_mask = mask1 + mask2 if mask2 is not None else mask1
        

        # Clean the mask
        cloak_mask = cv2.morphologyEx(cloak_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
        cloak_mask = cv2.dilate(cloak_mask, np.ones((3, 3), np.uint8), iterations=1)
        inverse_mask = cv2.bitwise_not(cloak_mask)

        # Replace red with background
        cloak_area = cv2.bitwise_and(background, background, mask=cloak_mask)
        non_cloak_area = cv2.bitwise_and(frame, frame, mask=inverse_mask)
        final_output = cv2.addWeighted(cloak_area, 1, non_cloak_area, 1, 0)

        # Convert to RGB for Streamlit
        final_output_rgb = cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(final_output_rgb)

    cap.release()
