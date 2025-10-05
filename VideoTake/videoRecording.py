# Author: Endri Dibra 
# Bachelor Thesis: Smart Unmanned Ground Vehicle

# Importing the required libraries
import os
import cv2
import datetime


# Creating a folder to store the recorded videos
videoFolder = "Videos"

os.makedirs(videoFolder, exist_ok=True)

# Opening the camera
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Checking if camera opened successfully
if not camera.isOpened():

    print("Error! Camera did not open.")

    exit()

# Getting frame dimensions
frameWidth = int(camera.get(3))
frameHeight = int(camera.get(4))

# Frames per second
fps = 20.0

 # .mp4 format
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

# Flags
recording = False
videoWriter = None

print("Press 'SPACEBAR' to start/stop recording, ESC to exit.")

# Looping through each video frame
while camera.isOpened():
    
    success, frame = camera.read()
    
    if not success:
    
        print("Frame capture failure.")
    
        break

    # Showing frame
    displayFrame = frame.copy()
    
    # If SPACEBAR is pressed
    if recording:
    
        cv2.putText(displayFrame, "● REC", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Camera", displayFrame)

    key = cv2.waitKey(1)

    # Using 'SPACEBAR' to toggle recording
    if key == 32:
        
        # True condition
        recording = not recording

        if recording:
    
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
            videoName = os.path.join(videoFolder, f"video_{timestamp}.mp4")
    
            videoWriter = cv2.VideoWriter(videoName, fourcc, fps, (frameWidth, frameHeight))
    
            print(f"✅ Started recording: {videoName}")

        # When recording is stopped
        else:
    
            videoWriter.release()
    
            print("🛑 Stopped recording.")

    # Writing frame if recording
    if recording and videoWriter is not None:
    
        videoWriter.write(frame)

    # Using ESC to exit
    elif key == 27:
    
        if recording and videoWriter:
    
            videoWriter.release()
    
        print("Exiting...")
    
        break

# Releasing and destroying the opened camera windows
camera.release()
cv2.destroyAllWindows()