# Author: Endri Dibra 
# Bachelor Thesis: Smart Unmanned Ground Vehicle

# Importing the required libraries
import os
import cv2
import time
import serial
import numpy as np
from ultralytics import YOLO


# Setting up Arduino serial communication parameters: port and baud rate
arduinoPort = 'COM15'  
baudRate = 9600

# Attempting to open serial connection with Arduino device
try:

    # Opening serial port with specified parameters and timeout
    serialConnection = serial.Serial(arduinoPort, baudRate, timeout=1)

    # Waiting for serial connection to initialize properly
    time.sleep(2)

    print(f"✅ Connected to Arduino on {arduinoPort}")

# Handling failure to open serial port
except serial.SerialException:

    print(f"❌ Failed to connect to Arduino on {arduinoPort}")

    # Setting serialConnection to None when failure
    serialConnection = None

# Defining the set of target classes for detection
targetClasses = {

    'bicycle', 'car', 'motorbike', 'bus', 'cat', 'dog',
    'knife', 'chair', 'sofa', 'pottedplant', 'bed',
    'diningtable', 'vase', 'person'
}

# Defining the subset of classes triggering speed reduction
speedReduceClasses = {

    'bicycle', 'car', 'motorbike', 'bus', 'chair', 'sofa',
    'pottedplant', 'bed', 'diningtable', 'vase', 'cat', 'dog'
}

# Loading the YOLOv11 segmentation model 
model = YOLO("yolo11n-seg.pt")

# Retrieving the dictionary of class names from the loaded model
classNames = model.names

# Creating a list of class IDs corresponding to targetClasses for filtering
targetClassIDs = [i for i, name in classNames.items() if name.lower() in targetClasses]

# Initializing and opening the camera
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Checking if camera was opened successfully, exiting if not
if not camera.isOpened():

    print("❌ Error! Camera did not open.")

    # Exiting the script due to no camera
    exit()

# Defining folder to save detected knife events and creating folder if missing
knifeFolder = "images/KnifeDetection"

os.makedirs(knifeFolder, exist_ok=True)

# Defining the log file path for saving detection records
logFilePath = "logs/knifeDetectionLogs.txt"

# Initializing current speed of the robot (255 = full speed)
currentSpeed = 210

# Setting flag for tracking if speed was reduced to avoid redundant commands
speedReduced = False

# Setting flag to ensure knife + person detection is logged once per event
knifePersonLogged = False

# Initializing counter for unique filenames of saved knife + person images
knifePersonLogCounter = 0 

# Starting main loop for capturing and processing camera frames
while camera.isOpened():

    # Capturing a frame from the camera
    success, frame = camera.read()

    # Checking if frame was captured successfully, breaking loop if not
    if not success:

        print("❌ Error! Camera stopped.")

        # Breaking the main loop due to failure
        break

    # Running inference on the captured frame using the YOLO model
    results = model(frame, save=False)[0]

    # Copying the frame to prepare for overlay drawing
    overlay = frame.copy()

    # Initializing detection flags for current frame
    knifeDetected = False
    personDetected = False
    anySpeedObjectDetected = False

    # Checking if detection boxes are present in the results
    if results.boxes is not None:

        # Looping through all detected boxes
        for i in range(len(results.boxes.cls)):

            # Getting class ID of current detection
            clsID = int(results.boxes.cls[i])

            # Skipping detection if class ID is not in target list
            if clsID not in targetClassIDs:

                continue

            # Getting class name in lowercase
            className = classNames[clsID].lower()

            # Extracting bounding box coordinates as integers
            xyxy = results.boxes.xyxy[i].cpu().numpy().astype(int)

            x1, y1, x2, y2 = xyxy

            # Setting detection flags based on class name
            if className == "knife":

                knifeDetected = True

            elif className == "person":

                personDetected = True

            if className in speedReduceClasses:

                anySpeedObjectDetected = True

            # Drawing segmentation mask if available
            if results.masks is not None:

                # Extracting mask for current detection
                mask = results.masks.data[i].cpu().numpy()
                
                # Resizing mask to frame dimensions
                maskResized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

                # Creating blank image for mask coloring
                coloredMask = np.zeros_like(frame, dtype=np.uint8)
                
                # Coloring green channel with mask data
                coloredMask[:, :, 1] = (maskResized * 255).astype(np.uint8)

                # Blending mask overlay onto frame copy
                overlay = cv2.addWeighted(overlay, 1.0, coloredMask, 0.4, 0)

            # Drawing bounding box and label on overlay
            conf = results.boxes.conf[i].item()

            label = f"{classNames[clsID]} {conf:.2f}"

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cv2.putText(overlay, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Handling speed reduction logic
    if anySpeedObjectDetected and not speedReduced:

        # Reducing current speed by 15 with minimum zero
        currentSpeed = max(currentSpeed - 15, 0)

        # Sending speed reduction command if serial connected
        if serialConnection:

            cmd = f"speed{currentSpeed}\n"

            serialConnection.write(cmd.encode())

        print(f"⚠️ Speed reduced to {currentSpeed}")

        # Marking speed as reduced
        speedReduced = True

    elif not anySpeedObjectDetected and speedReduced:

        # Resetting speed to full when no speed objects detected
        currentSpeed = 255

        # Sending speed reset command if serial connected
        if serialConnection:

            cmd = f"speed{currentSpeed}\n"

            serialConnection.write(cmd.encode())

        # Printing speed reset status
        print(f"✅ No obstacle. Speed reset to {currentSpeed}")

        # Marking speed as not reduced
        speedReduced = False

    # Logging knife + person detection only once per event
    if knifeDetected and personDetected:

        # Checking if event has not already been logged
        if not knifePersonLogged:

            # Incrementing log counter to create unique filenames
            knifePersonLogCounter += 1

            # Creating filename for saving the detection image
            filename = f"images/KnifeDetection/knifePersonEvent_{knifePersonLogCounter}.jpg"

            # Creating full path for saving the image
            logImagePath = os.path.join(knifeFolder, filename)

            # Saving current frame image to disk
            cv2.imwrite(logImagePath, frame)

            print(f"Knife and person detected. Saved: {logImagePath}")

            # Appending detection event info to log file with timestamp
            with open(logFilePath, "a") as log:

                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                log.write(f"{timestamp} - Knife and person detected. Image saved as {filename}\n")

            # Marking event as logged to prevent duplicate logs
            knifePersonLogged = True

    else:

        # Resetting log flag when knife + person not detected together
        knifePersonLogged = False

    # Showing the processed frame with overlays on screen
    cv2.imshow("Object Recognition", overlay)

    # Exiting loop on pressing 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break

# Releasing camera resource after loop ends
camera.release()

# Closing all OpenCV windows
cv2.destroyAllWindows()

# Closing serial connection if open
if serialConnection:

    serialConnection.close() 