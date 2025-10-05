# Author: Endri Dibra 
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp 
import sys
import os
from collections import deque, Counter


# Load the trained global model
model = tf.keras.models.load_model("globalModel/fedModelRound_7.keras")

# Define class-to-index and index-to-class (must match training config)
class_to_index = {"bezos": 0, "unknown": 1, "zuckerberg": 2}
index_to_class = {v: k for k, v in class_to_index.items()}

threshold = 0.50  # Confidence threshold for unknowns

# MediaPipe face detection
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Input size
img_size = 96

# The number of previous frames to consider for smoothing
history_size = 5  

prediction_history = deque(maxlen=history_size)

# Handle webcam or video file input
if len(sys.argv) > 1:

    video_source = sys.argv[1]

    if not os.path.exists(video_source):

        print(f"Error: Video file not found at {video_source}. Please check the path.")
        sys.exit(1)

    print(f"Reading from video file: {video_source}")

    cap = cv2.VideoCapture(video_source)

else:

    print("Reading from webcam...")
    cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Error: Could not open video source.")
    sys.exit(1)


while cap.isOpened():

    success, frame = cap.read()

    if not success:

        print("End of video stream or camera error. Exiting.")

        break

    if len(sys.argv) <= 1:

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

            face_input = np.expand_dims(face_rgb, axis=0)
            
            # Predict on the current frame. The model will handle the scaling.
            pred = model.predict(face_input, verbose=0)[0]
            confidence = np.max(pred)
            idx = np.argmax(pred)

            predicted_label = index_to_class.get(idx, "unknown")
            prediction_history.append(predicted_label)
            
            # Use the most common prediction from the history
            most_common_label, _ = Counter(prediction_history).most_common(1)[0]
            
            # The final label is the smoothed label
            label = most_common_label
            
            # Confidence is still based on the current frame's prediction
            if confidence < threshold or label == "unknown":

                color = (0, 0, 255) # Red for low confidence or unknown

            else:

                color = (0, 255, 0) # Green for high confidence
            
            # Draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Live Face Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
    
        break

# Release resources
cap.release()
cv2.destroyAllWindows() 