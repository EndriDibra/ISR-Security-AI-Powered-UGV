# Author: Endri Dibra 
# Bsc Thesis: Smart Security UGV

# Importing necessary libraries 
import os
import cv2
import time
import csv
import psutil
import platform
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# Specifying the type of model to run
# Use "keras", "tflite_float32", "tflite_float16", or "tflite_int8"
model_type_to_run = "tflite_int8"

# Setting the corresponding path to the model file
# Note: Ensuring the path points to a .tflite file for the TFLite interpreter
model_path_to_run = "globalModelLite/fedModelRound_7_int8.tflite"

# Defining the class-to-index and index-to-class mapping
class_to_index = {"bezos": 0, "unknown": 1, "zuckerberg": 2}
index_to_class = {v: k for k, v in class_to_index.items()}

# Setting the confidence threshold for labeling a prediction as 'unknown'
confidence_threshold = 0.5

# Defining the input image size expected by your MobileNetV2 model
img_size = 96


# Function for Model Loading and Inference 
def load_keras_model(path):

    # Loading a Keras model from a given path
    # Checking if the model file exists
    if not os.path.exists(path):

        print(f"ERROR: Keras model file '{path}' not found.")

        return None

    try:

        # Loading the Keras model
        model = keras.models.load_model(path)

        print(f"Loaded Keras model from '{path}'.")

        # Returning the loaded model
        return model

    except Exception as e:

        # Printing an error message if the loading fails
        print(f"Failed to load Keras model from '{path}': {e}")

        return None


def load_tflite_interpreter(path):

    #Loading a TFLite model and returning an interpreter
    # Checking if the TFLite file exists
    if not os.path.exists(path):

        print(f"ERROR: TFLite model file '{path}' not found.")

        return None

    try:

        # Initializing the TFLite interpreter with the model file
        interpreter = tf.lite.Interpreter(model_path=path, num_threads=2)
        
        # Allocating the tensors for the interpreter
        interpreter.allocate_tensors()

        print(f"Loaded TFLite model from '{path}'.")

        # Returning the initialized interpreter
        return interpreter

    except Exception as e:

        # Printing an error message if the loading fails
        print(f"Failed to load TFLite model from '{path}': {e}")

        return None


def get_system_metrics():

    # Getting CPU usage as a percentage
    cpu_percent = psutil.cpu_percent(interval=None)
   
    # Getting virtual memory usage as a percentage
    mem_percent = psutil.virtual_memory().percent
    
    # On Windows, disk usage requires a specific drive letter, e.g., 'C:\\'.
    # On Linux/macOS, it's typically '/'. We'll use 'C:\\' for this Windows-specific script.
    # Getting disk usage as a percentage
    disk_percent = psutil.disk_usage('C:\\').percent
    
    # Returning the collected metrics
    return cpu_percent, mem_percent, disk_percent


# Loading the single model based on the configuration
model = None

if "keras" in model_type_to_run:

    # Loading the Keras model
    model = load_keras_model(model_path_to_run)

elif "tflite" in model_type_to_run:

    # Loading the TFLite interpreter
    model = load_tflite_interpreter(model_path_to_run)

# Checking if the model was loaded successfully
if not model:

    print("Model failed to load. Exiting.")

    exit()

# Getting the model size in megabytes
model_size_mb = os.path.getsize(model_path_to_run) / (1024 * 1024)

# MediaPipe Setup 
# Initializing MediaPipe Face Detection
mp_face = mp.solutions.face_detection

# Creating a face detection object with specified confidence and model
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Main Video Loop
# Using CAP_DSHOW for better compatibility on Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Checking if the camera was opened successfully
if not cap.isOpened():

    print("Error: Could not open camera.")

    exit()

print("Camera is active. Press 'q' to quit.")
print("-" * 50)
print(f"Current model: {model_type_to_run}")

# Initializing lists to store per-frame data for CSV output
data_rows = []
frame_id = 0
prev_bbox_center = None
start_time_fps = time.perf_counter()

# Initializing frame dimensions for tracking stability normalization
h, w = 0, 0

ret, frame = cap.read()

if ret:

    h, w, _ = frame.shape

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Calculating the diagonal distance of the frame for normalization
diagonal_distance = np.sqrt(w**2 + h**2)

# Starting the main video processing loop
while cap.isOpened():

    # Reading a frame from the camera
    success, frame = cap.read()

    if not success:

        print("Error: Failed to read frame from camera.")

        break

    # Incrementing the frame counter
    frame_id += 1
    
    # Getting the current timestamp
    current_time = datetime.now()
    
    # Calculating FPS
    elapsed_time_fps = time.perf_counter() - start_time_fps
    fps = 1.0 / elapsed_time_fps if elapsed_time_fps > 0 else 0
    start_time_fps = time.perf_counter()

    # Flipping the frame horizontally for a more natural feel
    frame = cv2.flip(frame, 1)
    
    # Converting the frame from BGR to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Processing the frame to detect faces
    results = face_detector.process(rgb)
    
    # Initializing variables for the current frame
    latency_ms = 0
    label = "no_face"
    conf = 0
    fd_conf = 0
    tracking_stability = 0
    
    # Checking if any faces were detected
    if results.detections:

        # Iterating through each detected face
        for det in results.detections:

            # Getting the bounding box data
            bbox = det.location_data.relative_bounding_box

            h, w, _ = frame.shape
            
            # Getting the face detection confidence score
            fd_conf = det.score[0]
            
            # Defining a margin for the bounding box
            margin = 20

            # Calculating the coordinates of the bounding box with a margin
            x1 = max(int(bbox.xmin * w) - margin, 0)
            y1 = max(int(bbox.ymin * h) - margin, 0)

            x2 = min(int((bbox.xmin + bbox.width) * w) + margin, w)
            y2 = min(int((bbox.ymin + bbox.height) * h) + margin, h)

            # Cropping the face from the frame
            face = frame[y1:y2, x1:x2]

            if face.size == 0:

                continue

            # Calculating tracking stability and converting to a percentage (0-100)
            current_bbox_center = (x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2)

            if prev_bbox_center and diagonal_distance > 0:

                raw_stability = np.sqrt(

                    (current_bbox_center[0] - prev_bbox_center[0])**2 +
                    (current_bbox_center[1] - prev_bbox_center[1])**2
                )

                tracking_stability = (raw_stability / diagonal_distance) * 100

            prev_bbox_center = current_bbox_center

            # Resizing and preprocessing the face image for model input
            face_resized = cv2.resize(face, (img_size, img_size))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_processed = preprocess_input(face_rgb)
           
            # Expanding dimensions to create a batch of size 1
            face_input = np.expand_dims(face_processed, axis=0)

            # Live Inference and Latency Measurement
            start_time_inf = time.perf_counter()
            
            # Keras model inference
            if "keras" in model_type_to_run:

                pred = model.predict(face_input, verbose=0)[0]

            # TFLite interpreter inference
            elif "tflite" in model_type_to_run:

                # Getting input and output tensor details
                input_details = model.get_input_details()
                output_details = model.get_output_details()
                
                if 'int8' in model_type_to_run:

                    # Applying quantization for int8 models
                    input_scale, input_zero_point = input_details[0]['quantization']
                    face_input_quantized = face_input / input_scale + input_zero_point
                    face_input_quantized = face_input_quantized.astype(input_details[0]['dtype'])

                    # Setting the tensor for inference
                    model.set_tensor(input_details[0]['index'], face_input_quantized)

                else:

                    # Setting the tensor for non-int8 models
                    model.set_tensor(input_details[0]['index'], face_input.astype(np.float32))

                # Running the inference
                model.invoke()

                # Getting the prediction results from the output tensor
                pred = model.get_tensor(output_details[0]['index'])[0]

                if 'int8' in model_type_to_run:

                    # Dequantizing the output for int8 models
                    output_scale, output_zero_point = output_details[0]['quantization']
                    pred = pred.astype(np.float32)
                    pred = (pred - output_zero_point) * output_scale

            # Calculating the inference latency in milliseconds
            end_time_inf = time.perf_counter()
            latency_ms = (end_time_inf - start_time_inf) * 1000
            
            # Getting the prediction results
            conf = np.max(pred)
            idx = np.argmax(pred)
            predicted_label = index_to_class.get(idx, "unknown")
            
            # Determining the final label based on the confidence threshold
            label = predicted_label if conf >= confidence_threshold else "unknown"

            # Setting the color for the bounding box based on the label
            color = (0, 255, 0) if label != "unknown" else (0, 0, 255)

            # Drawing the bounding box around the detected face
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Putting the label and confidence text on the frame
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Getting system metrics (CPU, memory, disk usage)
    cpu, mem, disk = get_system_metrics()
    
    # Preparing a dictionary with all the data for the current frame
    data_row = {

        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "frame_id": frame_id,
        "frame_path": "live_camera",
        "label": label,
        "conf": conf,
        "fd_conf": fd_conf,
        "fps": fps,
        "tracking_stability": tracking_stability,
        "cpu": cpu,
        "mem": mem,
        "latency_ms": latency_ms,
        "disk": disk,
    }

    # Appending the data row to the list
    data_rows.append(data_row)
    
    # Displaying summary metrics on the frame
    cv2.putText(frame, f"Latency: {latency_ms:.2f}ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Displaying the live video feed
    cv2.imshow("Live Model Benchmark", frame)
    
    # Breaking the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        
        break


# Cleanup 
# Releasing the camera resource
cap.release()

# Closing all OpenCV windows
cv2.destroyAllWindows()

# Saving Data to CSV 
if data_rows:
    
    # Ensuring the output directory exists
    os.makedirs("modelComparisonResults", exist_ok=True)

    # Creating the filename for the raw data CSV
    csv_filename_data = f"modelComparisonResults/results_{model_type_to_run}_{platform.node()}.csv"
    
    with open(csv_filename_data, 'w', newline='') as f:
    
        writer = csv.writer(f)
        
        # Writing the header row
        header = list(data_rows[0].keys())
        writer.writerow(header + ["model_size_mb"])
        
        # Writing each data row
        for row in data_rows:
    
            writer.writerow(list(row.values()) + [model_size_mb])
            
    print(f"\nPer-frame performance data saved to '{csv_filename_data}'.")

    # Saving Averages to a Separate CSV 
    csv_filename_avg = f"modelComparisonResults/averages_{model_type_to_run}_{platform.node()}.csv"
    
    with open(csv_filename_avg, 'w', newline='') as f:
    
        writer = csv.writer(f)
        
        # Defining the keys for which to calculate averages
        avg_keys = ["conf", "fd_conf", "fps", "cpu", "latency_ms"]
        
        avg_row = {}
    
        for key in avg_keys:
    
            try:
    
                # Averaging confidence scores only for frames with detected faces
                if key in ["conf", "fd_conf"]:
    
                    values = [float(row[key]) for row in data_rows if row["label"] != "no_face"]
    
                else:
    
                    values = [float(row[key]) for row in data_rows if isinstance(row.get(key), (int, float))]
                
                # Calculating the mean and handling the case of an empty list
                avg_row[key] = np.mean(values) if values else 0
    
            except (KeyError, TypeError):
    
                # Handling missing keys or type errors
                avg_row[key] = "NA"
        
        # Writing the header row for the averages
        writer.writerow(list(avg_row.keys()) + ["model_size_mb"])
        
        # Writing the average values
        writer.writerow(list(avg_row.values()) + [model_size_mb])
            
    print(f"Average performance data saved to '{csv_filename_avg}'.")

else:
    
    print("\nNo performance data was collected. No CSV files were saved.")

print("Thank you for using the live demo!")