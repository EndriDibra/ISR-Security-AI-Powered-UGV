# Author: Endri Dibra 
# Bachelor Thesis: Smart Unmanned Ground Vehicle

# Live Face Mask Detection with MediaPipe and MobileNetV2 model

# Importing the required libraries
import os
import csv
import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# Loading the trained MobileNetV2 model
model = load_model('best_mobilenetv2_model.keras')

# Setting up MediaPipe face detection
mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Defining class labels and colors based on the training label mapping
labels = ['Mask', 'No Mask']

# Green color for Mask and Red for No Mask
colors = [(0, 255, 0), (0, 0, 255)]

# Defining model input size
imgSize = 224

# Counter of no mask images
counter = 0

# Creating logs and output folder if they do not exist
os.makedirs("images/noMaskImages", exist_ok=True)

logFile = "logs/noMaskLog.csv"

# Creating CSV file with headers if not present
if not os.path.isfile(logFile):
    
    with open(logFile, mode='w', newline='') as file:
    
        writer = csv.writer(file)
    
        writer.writerow(['Timestamp', 'Prediction Score', 'Image Filename'])

# Initializing last no mask detection state
noMaskPreviouslyDetected = False

# Opening camera
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Looping through each camera frame
while camera.isOpened():

    # Reading camera frame
    success, frame = camera.read()

    # If camera reading fails, program breaks
    if not success:
        
        break

    # Converting BGR to RGB for MediaPipe processing
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Performing face detection
    results = faceDetection.process(rgbFrame)

    # If face detected
    if results.detections:

        for detection in results.detections:

            # Bounding box for face detected
            bboxC = detection.location_data.relative_bounding_box

            # Face shape: Height, Width
            h, w, _ = frame.shape

            # Calculating bounding box coordinates with margin
            margin = 10
            x1 = max(int(bboxC.xmin * w) - margin, 0)
            y1 = max(int(bboxC.ymin * h) - margin, 0)
            x2 = min(int((bboxC.xmin + bboxC.width) * w) + margin, w)
            y2 = min(int((bboxC.ymin + bboxC.height) * h) + margin, h)

            # Cropping face image
            faceImg = frame[y1:y2, x1:x2]

            if faceImg.size == 0:
                
                continue

            # Resizing and preprocessing face image
            faceResized = cv2.resize(faceImg, (imgSize, imgSize))
            faceRGB = cv2.cvtColor(faceResized, cv2.COLOR_BGR2RGB)
            facePreprocessed = preprocess_input(faceRGB)
            faceExpanded = np.expand_dims(facePreprocessed, axis=0)

            # Predicting mask/no mask
            pred = model.predict(faceExpanded)[0][0]
            
            classID = 0 if pred < 0.5 else 1

            # Getting label and color
            label = labels[classID]
            color = colors[classID]

            # Drawing bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            cv2.putText(frame, f"{label} ({pred:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # If No Mask detected and not previously detected
            if classID == 1 and not noMaskPreviouslyDetected:

                # Generating timestamp and filename
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                
                filename = f"images/noMaskImages/NoMask{counter}.jpg"

                # Saving the image
                cv2.imwrite(filename, frame)

                # Increasing counter
                counter += 1

                # Logging event to CSV
                with open(logFile, mode='a', newline='') as file:
                   
                    writer = csv.writer(file)
                   
                    writer.writerow([timestamp, f"{pred:.2f}", os.path.basename(filename)])

                # Setting detection flag
                noMaskPreviouslyDetected = True

            # If mask is detected again, reset flag
            elif classID == 0:
                
                noMaskPreviouslyDetected = False

    # Showing the frame
    cv2.imshow("Face Mask Detection", frame)

    # Terminating process when pressing 'q':quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
       
        break

# Closing camera and all OpenCV windows
camera.release()
cv2.destroyAllWindows()