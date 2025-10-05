# Author: Endri Dibra 
# Bachelor Thesis: Smart Unmanned Ground Vehicle

# Importing the required libraries 
import os
import cv2
import time
import serial
import numpy as np
from ultralytics import YOLO
from datetime import datetime


# Setting Arduino serial port and baud rate
arduinoPort = 'COM15'
baudRate = 9600

# Initial speed values and flags for speed control
currentSpeed = 210
minSpeed = 0
speedStep = 15
speedReduced = False

# Serial connection with Arduino Mega
try:
    
    serialConnection = serial.Serial(arduinoPort, baudRate, timeout=1)
    time.sleep(2)
    print(f"Connected to Arduino on {arduinoPort}")

except serial.SerialException:
    
    print(f"Failed to connect to Arduino on {arduinoPort}")
    serialConnection = None

# Loading YOLOv11 segmentation model for people detection
model = YOLO("yolo11n-seg.pt")

# Setting overcrowd detection parameters
overcrowdThreshold = 7
overcrowdLogged = False
overcrowdImageCounter = 0

# Setting up log file and image directory for overcrowd events
logFilePath = "logs/overCrowdAlert.txt"
imageFolder = "images/OverCrowdImages"

os.makedirs(imageFolder, exist_ok=True)

# Opening the default camera for video capture
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Checking if camera opened successfully
if not camera.isOpened():
    
    print("Error! Camera did not open.")
    
    exit()

# Looping through camera frames
while camera.isOpened():

    # Reading a frame from the camera
    success, frame = camera.read()

    # If frame read failed, exit loop
    if not success:
        
        print("Error! Failed to capture frame.")
        
        break

    # Running the YOLO model inference on each frame
    results = model(frame, save=False)[0]

    # Creating a copy of the frame to draw overlays
    overlay = frame.copy()

    # Initializing person count to zero
    totalPeople = 0

    # Checking if any detection boxes are present
    if results.boxes is not None:

        # Looping through all detected boxes
        for i in range(len(results.boxes.cls)):

            # Getting class ID of detected object
            clsID = int(results.boxes.cls[i])

            # Only process person class, ID = 0 
            if clsID != 0:
                
                continue

            # Incrementing counter
            totalPeople += 1

            # If segmentation masks exist, apply mask overlay
            if results.masks is not None:
                
                mask = results.masks.data[i].cpu().numpy()
                maskResized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                
                coloredMask = np.zeros_like(frame, dtype=np.uint8)
                coloredMask[:, :, 1] = (maskResized * 255).astype(np.uint8)
                
                overlay = cv2.addWeighted(overlay, 1.0, coloredMask, 0.4, 0)

            # Getting bounding box coordinates for the detected person
            xyxy = results.boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

            # Getting confidence score for the detection
            conf = results.boxes.conf[i].item()

            # Preparing label text with confidence
            label = f"Person {conf:.2f}"

            # Drawing bounding box rectangle on overlay image
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Putting label text above the bounding box
            cv2.putText(overlay, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Displaying the total people detected count on the overlay
    cv2.putText(overlay, f"People Detected: {totalPeople}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Checking if total people exceed overcrowd threshold
    if totalPeople > overcrowdThreshold:

        # Perform actions only once per overcrowd event
        if not overcrowdLogged:

            # Getting current timestamp string
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Logging overcrowd event with timestamp and people count
            with open(logFilePath, "a") as log:
                log.write(f"{timestamp} - Overcrowd detected: {totalPeople} people\n")

            # Informing console about overcrowd logging
            print(f"Overcrowd logged at {timestamp} with {totalPeople} people")

            # Incrementing overcrowd image counter for unique filenames
            overcrowdImageCounter += 1

            # Creating filename and full path for saving image
            imageFilename = f"images/overCrowdImages/overCrowd{overcrowdImageCounter}.jpg"
            imagePath = os.path.join(imageFolder, imageFilename)

            # Saving current frame image to disk
            cv2.imwrite(imagePath, frame)

            # Informing console about saved image
            print(f"Saved overcrowd frame: {imagePath}")

            # Reducing speed once by defined step if not already reduced
            if not speedReduced:
                
                currentSpeed = max(currentSpeed - speedStep, minSpeed)
                
                if serialConnection:
                
                    command = f"speed{currentSpeed}\n"
                    serialConnection.write(command.encode())
                    print(f"Speed reduced to {currentSpeed}")
                
                speedReduced = True

            # Marking overcrowd event as logged
            overcrowdLogged = True

    else:
        
        # If below threshold and speed were reduced
        # resetting speed to max again
        if speedReduced:
        
            currentSpeed = 255
        
            if serialConnection:
        
                command = f"speed{currentSpeed}\n"
                serialConnection.write(command.encode())
        
                print(f"People below {overcrowdThreshold}. Speed reset to {currentSpeed}")
        
            speedReduced = False

        # Resetting overcrowd event flag
        overcrowdLogged = False

    # Showing the final overlay image with bounding boxes and counts
    cv2.imshow("People Detection", overlay)

    # Breaking the loop and closing when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
       
        break

# Releasing camera resources and closing windows
camera.release()
cv2.destroyAllWindows()

# Closing serial connection if open
if serialConnection:
    
    serialConnection.close()