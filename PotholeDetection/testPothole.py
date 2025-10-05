# Author: Endri Dibra 
# Bachelor Thesis: Smart Unmanned Ground Vehicle

# Importing therequired libraries
import cv2
from ultralytics import YOLO

# Path to the trained weights 
weightsPath = 'runs/detect/train/weights/best.pt'

# Loading the trained YOLOv11n model for Potholes Detection
model = YOLO(weightsPath)

# Opening the test video
videoPath = 'sampleVideo.mp4'

# Opening the camera for video reading
camera = cv2.VideoCapture(videoPath)

# Checking if camera for video is working
if not camera.isOpened():
    
    print(f"Error! Could not open video file {videoPath}")
    
    exit()

# Looping through video frames
while camera.isOpened():
    
    # Reading video frames
    success, frame = camera.read()
    
    # In case of failed reading or video stream ended
    if not success:
    
        print("End of video or cannot read the frame.")
    
        break

    # Running inference on the frame
    results = model(frame)[0]

    # Drawign bounding boxes and labels on the frame
    annotatedFrame = results.plot()

    # Showing the frame with detection results
    cv2.imshow('Pothole Detection', annotatedFrame)

    # Terminating process if 'q':quit is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
    
        break

# Release and closing camera and OpenCV windows
camera.release()
cv2.destroyAllWindows()