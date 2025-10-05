# Author: Endri Dibra 
# Bachelor Thesis: Smart Unmanned Ground Vehicle

# Importing the required libraries
import os
import cv2
import time
import serial
import numpy as np
import mediapipe as mp
from datetime import datetime

# Arduino serial setup 
arduinoPort = 'COM15'
baudRate = 9600

# Initial robot speed setup
currentSpeed = 210
minSpeed = 0
speedStep = 15
speedReduced = False

# Opening serial connection to Arduino Mega
try:
    
    serialConnection = serial.Serial(arduinoPort, baudRate, timeout=1)
    
    time.sleep(2)
    
    print(f"Connected to Arduino on {arduinoPort}")

except serial.SerialException:

    print(f"Failed to connect to Arduino on {arduinoPort}")

    serialConnection = None

# MediaPipe FaceMesh setup 
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=30)
mpDrawing = mp.solutions.drawing_utils

# Overcrowd logging setup 
logFilePath = "logs/overcrowdAlert.txt"
imageFolder = "images/OvercrowdImages"
os.makedirs(imageFolder, exist_ok=True)

# Overcrowd condition setup
overcrowdThreshold = 5
overcrowdLogged = False
overcrowdImageCounter = 0

# Opening default camera 
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Checking if camera is opened successfully
if not camera.isOpened():
   
    print("Error! Camera did not open.")
   
    exit()

# Looping through camera frames  
while camera.isOpened():

    # Reading a frame from the camera
    success, frame = camera.read()

    # Checking if the frame was successfully captured
    if not success:
        
        print("Error! Failed to capture frame.")
        
        break

    # Converting frame to RGB for MediaPipe processing
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processing the RGB frame for face mesh detection
    results = faceMesh.process(rgbFrame)

    # Initializing total face counter
    totalFaces = 0

    # Creating a blank mask for the face regions
    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    # Copy of the original frame for drawing landmarks
    landmarkFrame = frame.copy()

    # Checking if any faces are detected
    if results.multi_face_landmarks:

        # Counting total number of detected faces
        totalFaces = len(results.multi_face_landmarks)

        # Applying a single Gaussian blur to the entire frame for performance
        blurredFrame = cv2.GaussianBlur(frame, (101, 101), 30)

        # Iterating through all detected faces
        for faceLandmarks in results.multi_face_landmarks:

            # Getting the frame dimensions
            h, w, _ = frame.shape

            # Extracting all landmark (x, y) coordinates
            points = np.array([
                
                [int(landmark.x * w), int(landmark.y * h)]
                for landmark in faceLandmarks.landmark
            ], dtype=np.int32)

            # Creating a convex hull mask around the face
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(mask, hull, 255)

            # Drawing the landmarks on the landmark frame
            mpDrawing.draw_landmarks(
                
                landmarkFrame,
                faceLandmarks,
                mpFaceMesh.FACEMESH_TESSELATION,
                mpDrawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mpDrawing.DrawingSpec(color=(255, 0, 0), thickness=1)
            )

        # Blending the blurred faces onto the original frame
        frame = np.where(mask[:, :, None] == 255, blurredFrame, frame)

    # Overcrowd detection and logging
    # If total faces are >= 5 
    if totalFaces >= overcrowdThreshold:

        # Only acting once per overcrowd event
        if not overcrowdLogged:

            # Getting the timestamp for the event
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Logging the event into the text file
            with open(logFilePath, "a") as log:
                
                log.write(f"{timestamp} - Overcrowd detected: {totalFaces} faces\n")

            print(f"Overcrowd logged at {timestamp} with {totalFaces} faces")

            # Saving the current frame image
            overcrowdImageCounter += 1

            imageFilename = f"overCrowd{overcrowdImageCounter}.jpg"
            
            # Saving the detected image 
            imagePath = os.path.join(imageFolder, imageFilename)
            
            cv2.imwrite(imagePath, frame)
            
            print(f"Saved overcrowd frame: {imagePath}")

            # Reducing speed once by 15 points
            if not speedReduced:
                
                currentSpeed = max(currentSpeed - speedStep, minSpeed)
                
                # If there is a connection with Arduino
                if serialConnection:
                
                    command = f"speed{currentSpeed}\n"

                    # Sending reduced speed
                    serialConnection.write(command.encode())
                
                    print(f"Speed reduced to {currentSpeed}")
                
                speedReduced = True

            # Marking that the event has been logged
            overcrowdLogged = True

    else:
        
        # Resetting logic when face count drops below threshold, thus < 5
        if speedReduced:
        
            currentSpeed = 255
        
            if serialConnection:
        
                command = f"speed{currentSpeed}\n"
        
                serialConnection.write(command.encode())
        
                print(f"Faces below {overcrowdThreshold}. Speed reset to {currentSpeed}")
        
            speedReduced = False

        overcrowdLogged = False

    # Displaying information
    label = f"Faces Detected: {totalFaces}"
    
    cv2.putText(frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    cv2.putText(landmarkFrame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Showing the processed frames
    cv2.imshow("Face Detection with Polygon Blur", frame)
    cv2.imshow("Face Detection with Landmarks", landmarkFrame)

    # Breaking the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        
        break

# Cleanup after loop ends
camera.release()
cv2.destroyAllWindows()
faceMesh.close()

if serialConnection:
    
    serialConnection.close()