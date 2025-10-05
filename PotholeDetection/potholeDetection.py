# Author: Endri Dibra 
# Bachelor Thesis: Smart Unmanned Ground Vehicle

# Application: Pothole Detection using YOLOv11n

# Importing the required libraries 
import os
import cv2
import time
import serial
from ultralytics import YOLO
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

# Loading trained YOLOv11n model for pothole detection
model = YOLO("runs/detect/train/weights/best.pt")

# Overcrowd logging setup 
logFilePath = "logs/potholeLog.txt"
imageFolder = "images/PotholeImages"

os.makedirs(imageFolder, exist_ok=True)

# Pothole detection logging flags
potholeLogged = False
potholeImageCounter = 0

# Opening camera 
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Checking if camera opened successfully
if not camera.isOpened():
    
    print("Error! Could not open camera.")
    
    exit()

# Starting video loop
while camera.isOpened():
    
    # Reading camera frames
    success, frame = camera.read()
    
    # Case frame reading is unsuccessfull
    if not success:
    
        print("Error! Failed to read frame.")
    
        break

    # Running detection on the current frame
    results = model(frame)[0]

    # Flag to check if pothole detected
    detected = False

    # Looping through results
    for box in results.boxes:
        
        # Frame coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Percentage of confidence from prediction
        confidence = float(box.conf[0])

        # Predicted class
        classID = int(box.cls[0])
    
        label = results.names[classID]

        # Displaying pothole if > 50 percent sure
        if confidence > 0.5:
    
            detected = True
            
            # Blue color for pothole box
            color = (255, 0, 0)
    
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
            labelText = f"{label}: {confidence:.2f}"
    
            cv2.putText(frame, labelText, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Logging, saving image and speed control logic
    if detected and not potholeLogged:

        # Logging pothole detection with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(logFilePath, "a") as log:
           
            log.write(f"{timestamp} - Pothole detected\n")
        
        print(f"Pothole detected logged at {timestamp}")

        
        imageFilename = f"images/PotholeImages/pothole{potholeImageCounter}.jpg"
        
        imagePath = os.path.join(imageFolder, imageFilename)
        
        cv2.imwrite(imagePath, frame)

        # Saving the frame image
        potholeImageCounter += 1
        
        print(f"Pothole image saved: {imageFilename}")

        # Reducing speed once by speedStep
        if not speedReduced:
            
            currentSpeed = max(currentSpeed - speedStep, minSpeed)
            
            if serialConnection:
                
                command = f"speed{currentSpeed}\n"
                
                serialConnection.write(command.encode())
                
                print(f"Speed reduced to {currentSpeed}")
            
            speedReduced = True

        potholeLogged = True

    elif not detected and speedReduced:

        # Resetting speed if no pothole detected
        currentSpeed = 255
        
        if serialConnection:
            
            command = f"speed{currentSpeed}\n"
            
            serialConnection.write(command.encode())
            
            print(f"No pothole detected. Speed reset to {currentSpeed}")
        
        speedReduced = False
        potholeLogged = False

    # Showing "Normal" if no potholes detected
    if not detected:
    
        cv2.putText(frame, "Normal", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Showing the frame
    cv2.imshow("Pothole Detection", frame)

    # Terminating process if 'q':quit is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
    
        break

# Closing camera and all OpenCV windows
camera.release()
cv2.destroyAllWindows()