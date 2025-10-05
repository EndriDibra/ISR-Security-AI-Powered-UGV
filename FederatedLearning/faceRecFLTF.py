# Author: Endri Dibra 
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# Configuration
# Path to your final federated model file.
# Make sure to update this path to the correct round number.
model_path = "globalModelLite/fedModelRound_7.tflite" 

# Define class-to-index and index-to-class mapping.
# This MUST match the configuration used during training.
class_to_index = {"bezos": 0, "unknown": 1, "zuckerberg": 2}
index_to_class = {v: k for k, v in class_to_index.items()}

# Confidence threshold for labeling a prediction as 'unknown'.
# A prediction is only accepted if its confidence is above this value.
confidence_threshold = 0.50

# The input image size expected by your MobileNetV2 model.
img_size = 96

# Model Loading
try:

    # TFLite SPECIFIC: LOAD THE INTERPRETER
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get input tensor size for validation
    input_shape = input_details[0]['shape']

    print("Model loaded successfully.")
    print(f"Model will predict {len(class_to_index)} classes.")
    print(f"Model expects input shape: {input_shape}")

except Exception as e:
  
    print(f"Error loading TFLite model from {model_path}: {e}")
  
    exit()

# MediaPipe Setup
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Main Video Loop 
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    
    print("Error: Could not open camera.")
    
    exit()

print("Camera is active. Press 'q' to quit.")

while cap.isOpened():
    
    success, frame = cap.read()
    
    if not success:
        
        print("Error: Failed to read frame from camera.")
        
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)

    if results.detections:
        
        for det in results.detections:
            
            bbox = det.location_data.relative_bounding_box
            
            h, w, _ = frame.shape
            
            margin = 20
            
            x1 = max(int(bbox.xmin * w) - margin, 0)
            y1 = max(int(bbox.ymin * h) - margin, 0)
            
            x2 = min(int((bbox.xmin + bbox.width) * w) + margin, w)
            y2 = min(int((bbox.ymin + bbox.height) * h) + margin, h)

            face = frame[y1:y2, x1:x2]
            
            if face.size == 0:
              
                continue

            face_resized = cv2.resize(face, (img_size, img_size))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # Prediction 
            # Preprocess the image
            face_processed = preprocess_input(face_rgb)
            
            # Expand dimensions to create a batch of 1
            # Note: TFLite input expects float32
            face_input = np.expand_dims(face_processed, axis=0).astype(np.float32)
            
            # TFLite SPECIFIC: PERFORM PREDICTION
            # Set the input tensor
            interpreter.set_tensor(input_details[0]['index'], face_input)
            
            # Invoke the interpreter
            interpreter.invoke()
            
            # Get the output tensor
            pred = interpreter.get_tensor(output_details[0]['index'])[0]
            
            # Get the confidence and predicted class index.
            confidence = np.max(pred)
            idx = np.argmax(pred)

            # Determine the final label based on the prediction and confidence threshold.
            predicted_label = index_to_class.get(idx, "unknown")
            
            if confidence < confidence_threshold:
            
                label = "unknown"
                color = (0, 0, 255) # Red for low confidence
            
            else:
                
                label = predicted_label
                color = (0, 255, 0) # Green for known

            # Display Results
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
  
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display the final frame with annotations.
    cv2.imshow("Live Face Recognition", frame)
    
    # Break the loop if the 'q' key is pressed.
    if cv2.waitKey(1) & 0xFF == ord("q"):
  
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()