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


# Arduino serial connection parameters
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

# Loading YOLOv11 pose model for pose estimation
model = YOLO("yolo11n-pose.pt")

# Setting up overcrowd (fall) logging parameters
logFilePath = "logs/fallDetectionLog.txt"
imageFolder = "images/fallImages"

os.makedirs(imageFolder, exist_ok=True)

fallLogged = False
fallImageCounter = 0

# Opening camera using DirectShow backend
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Checking if webcam opened successfully, exiting if not
if not camera.isOpened():
    
    print("Error! Camera did not open.")
    
    exit()

# Initializing fall frame counters for temporal fall tracking per detected person
# key: person index, value: consecutive fall frames
fallFrameCounter = {}  

# Setting threshold for confirming fall after sustained fall pose (frames)
# About 4 to 5 seconds, so after that period the fall will be confirmed
fallFrameThreshold = 40  


# This function is used to detect a fall from a person
def detectFall(keypoints):

    # Detecting fall based on keypoint positions and torso angle
    # Returns True if fall posture detected, False otherwise
    try:
        
        # Extracting (x, y) coordinates of
        # keypoints for relevant joints
        # Nose keypoints
        nose = keypoints[0][:2]
        
        # Left shoulder keypoints
        leftShoulder = keypoints[5][:2]
        
        # Right shoulder keypoints
        rightShoulder = keypoints[6][:2]
        
        # Left hip keypoints
        leftHip = keypoints[11][:2]
        
        # Right hip keypoints
        rightHip = keypoints[12][:2]
        
        # Left ankle keypoints
        leftAnkle = keypoints[15][:2]
        
        # Right ankle keypoints
        rightAnkle = keypoints[16][:2]

        # Calculating midpoints of shoulders, hips
        # and ankles to approximate torso alignment
        midShoulder = (leftShoulder + rightShoulder) / 2
        
        midHip = (leftHip + rightHip) / 2
        
        midAnkle = (leftAnkle + rightAnkle) / 2

        # Calculating torso angle between
        # shoulders and hips in degrees
        xDegrees = midShoulder[0] - midHip[0]
        
        yDegrees = midShoulder[1] - midHip[1]
        
        # vertical about 90° and horizontal about 0°
        angle = np.degrees(np.arctan2(yDegrees, xDegrees)) 

        # Checking if head [nose] is lower than hips and
        # close to ankles, that would probably mean a possible fall
        headLow = nose[1] > midHip[1] and nose[1] > midAnkle[1] - 30

        # Returning True if torso is horizontal (angle < 45°)
        # or head is low [fall posture]
        if abs(angle) < 45 or headLow:
            
            return True

    except:
        
        # Handling any exceptions from missing/invalid
        # keypoints by returning False
        return False

    return False


# Looping through each camera frame
while camera.isOpened():
    
    # Reading frame from camera
    success, frame = camera.read()

    # Breaking loop if failure to read
    if not success:
        
        print("Error! Camera stopped.")
        
        break

    # Running pose detection model on current frame
    results = model(frame)
    
    # Plotting pose keypoints and skeleton on frame
    finalFrame = results[0].plot()

    # Resetting fall frame counters if no
    # detected keypoints in current frame
    if results[0].keypoints is None:
        
        fallFrameCounter.clear()
    
    else:
    
        # Variable to track if any fall confirmed this frame
        fallConfirmedThisFrame = False

        # Looping through each detected person’s keypoints
        for idx, person in enumerate(results[0].keypoints.data):
    
            personKeyPoints = person.cpu().numpy()

            # Detecting fall posture based on keypoints
            isFalling = detectFall(personKeyPoints)

            # Initializing fall counter for new detected persons
            if idx not in fallFrameCounter:
    
                fallFrameCounter[idx] = 0

            # Incrementing or resetting fall counter
            # based on current detection
            if isFalling:
    
                fallFrameCounter[idx] += 1
    
            else:
    
                fallFrameCounter[idx] = 0

            # If fall confirmed for this person, set flag to True
            if fallFrameCounter[idx] >= fallFrameThreshold:
                
                fallConfirmedThisFrame = True

            # Displaying fall status text on frame with color coding
            if fallFrameCounter[idx] >= fallFrameThreshold:
    
                cv2.putText(finalFrame, "Fall Detection Confirmed!", (50, 50 + 30 * idx),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            elif isFalling:
                
                cv2.putText(finalFrame, "Possible Fall Detected!", (50, 50 + 30 * idx),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # If fall confirmed this frame and was not already logged, do logging and speed reduction
    if fallConfirmedThisFrame and not fallLogged:

        # Getting timestamp string for logging
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Logging fall event with timestamp to text file
        with open(logFilePath, "a") as log:
            
            log.write(f"{timestamp} - Fall detected\n")
        
        imageFilename = f"images/fallImages/fall{fallImageCounter}.jpg"
        
        imagePath = os.path.join(imageFolder, imageFilename)
        
        cv2.imwrite(imagePath, frame)

        # Increment image counter and save frame image
        fallImageCounter += 1
        
        print(f"Fall event logged at {timestamp}, image saved as {imageFilename}")

        # Reducing speed once by 15 points if not already reduced
        if not speedReduced:
           
            currentSpeed = max(currentSpeed - speedStep, minSpeed)
           
            if serialConnection:
           
                command = f"speed{currentSpeed}\n"
           
                serialConnection.write(command.encode())
           
                print(f"Speed reduced to {currentSpeed}")
           
            speedReduced = True

        # Defining fall as logged to
        # avoid redundant saves/logs
        fallLogged = True

    # If no fall confirmed and speed was reduced
    # the resetting speed and flag
    if not fallConfirmedThisFrame and speedReduced:
        
        currentSpeed = 255
        
        if serialConnection:
        
            command = f"speed{currentSpeed}\n"
        
            serialConnection.write(command.encode())
        
            print(f"Fall cleared. Speed reset to {currentSpeed}")
        
        speedReduced = False
        
        fallLogged = False

    # Showing the frame with annotations
    cv2.imshow("Pose Fall Detection", finalFrame)

    # Breaking loop and closing if 'q':quit key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        
        break

# Releasing webcam and destroying
# all OpenCV windows after exiting loop
camera.release()
cv2.destroyAllWindows()