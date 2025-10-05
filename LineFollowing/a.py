# Author: Endri Dibra 
# Bachelor Thesis: Smart Unmanned Ground Vehicle

# Application: Tape Width Measurement via Contour Bounding Box

import cv2
import numpy as np

# Open camera with resolution 320x240
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

if not camera.isOpened():
    print("[Error] Camera not accessible")
    exit(1)

while True:
    ret, frame = camera.read()
    if not ret:
        print("[Error] Failed to read frame")
        break

    # Convert frame to HSV for black color detection
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Black color HSV range
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])

    # Create mask
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        # Draw bounding box on frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Print tape width in pixels
        print(f"Tape width in pixels: {w}")

    else:
        print("No tape detected")

    cv2.imshow("Tape Width Measurement", frame)

    if cv2.waitKey(500) & 0xFF == ord('q'):  # Update twice per second
        break

camera.release()
cv2.destroyAllWindows()
