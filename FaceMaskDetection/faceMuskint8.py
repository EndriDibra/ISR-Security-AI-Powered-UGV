# Author: Endri Dibra 
# Bachelor Thesis: Smart Unmanned Ground Vehicle

# Live Face Mask Detection with MediaPipe and TFLite INT8 model

# Importing the required libraries
import os
import csv
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from datetime import datetime
# The preprocess_input function is not used for this INT8 model, as it expects input in the [0, 1] range.
# We will perform manual normalization instead.


# --- TFLITE MODEL SETUP ---
# Load the TFLite model and allocate tensors.
# IMPORTANT: Make sure your INT8 TFLite model file ('best_model_int8.tflite') is
# in the same directory as this script.
tflite_model_path = 'best_model_int8.tflite'
try:
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
except Exception as e:
    print(f"Error loading TFLite model: {e}")
    print("Please ensure 'best_model_int8.tflite' is in the same directory.")
    exit()

# Get input and output tensor details from the interpreter
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("--- TFLite Model Details ---")
print(f"Input details: {input_details}")
print(f"Output details: {output_details}")
print("----------------------------")

# Defining class labels and colors based on the training label mapping
labels = ['Mask', 'No Mask']

# Green color for Mask and Red for No Mask
colors = [(0, 255, 0), (0, 0, 255)]

# Defining model input size (from the TFLite model's input details)
# The TFLite input shape is typically [1, height, width, 3]
_, imgSize, _, _ = input_details[0]['shape']

# Counter of no mask images
counter = 0

# Creating logs and output folder if they do not exist
os.makedirs("images/noMaskImages", exist_ok=True)
os.makedirs("logs", exist_ok=True) # Ensure the logs directory exists
logFile = "logs/noMaskLog.csv"

# Creating CSV file with headers if not present
if not os.path.isfile(logFile):
    with open(logFile, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Prediction Score', 'Image Filename'])

# Initializing last no mask detection state
noMaskPreviouslyDetected = False

# Setting up MediaPipe face detection
mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

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

            # Resizing and normalizing face image for TFLite model
            faceResized = cv2.resize(faceImg, (imgSize, imgSize))
            faceRGB = cv2.cvtColor(faceResized, cv2.COLOR_BGR2RGB)
            
            # Manually normalize to the [0, 1] range as the model expects
            faceNormalized = faceRGB.astype(np.float32) / 255.0
            faceExpanded = np.expand_dims(faceNormalized, axis=0)
            
            # --- TFLITE INFERENCE WITH INT8 QUANTIZATION ---
            # Extract quantization parameters from the input details
            input_scale, input_zero_point = input_details[0]['quantization']
            
            # Quantize the input data
            face_quantized = faceExpanded / input_scale + input_zero_point
            
            # Set the input tensor with the correct data type (usually int8)
            interpreter.set_tensor(input_details[0]['index'], face_quantized.astype(input_details[0]['dtype']))
            
            # Run inference
            interpreter.invoke()
            
            # Get the output tensor (which is also quantized)
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # Extract quantization parameters from the output details
            output_scale, output_zero_point = output_details[0]['quantization']
            
            # Dequantize the single output score
            pred = (output_data[0][0] - output_zero_point) * output_scale
            
            # Clip the prediction to ensure it is within the valid range [0, 1]
            pred = np.clip(pred, 0, 1)
            
            # Determining the class ID based on the single confidence score
            # A score below 0.5 indicates 'Mask' (class 0)
            classID = 0 if pred <= 0.5 else 1
            
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
