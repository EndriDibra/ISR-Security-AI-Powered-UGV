# Author: Endri Dibra 
# Bachelor Thesis: Smart Unmanned Ground Vehicle

# Importing the required libraries
import os
import cv2
import datetime


# Creating a folder named "Photos" to store the captured images
photoFolder = "images/Photos"

os.makedirs(photoFolder, exist_ok=True)

# Opening the camera 
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Checking if camera opened successfully
if not camera.isOpened():
   
    print("Error! Camera did not open.")
   
    exit()

print("Press SPACE to capture image or ESC to exit.")

# Looping through each video frame
while camera.isOpened():

    success, frame = camera.read()
    
    if not success:
    
        print("Frame capture failure.")
    
        break

    # Showing the camera frames
    cv2.imshow("Camera", frame)

    # Waitting for key press for 1 ms
    key = cv2.waitKey(1)

    # Using SPACEBAR to capture an image
    if key == 32:
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Storing the photos into the folder
        photoName = os.path.join(photoFolder, f"photo_{timestamp}.png")
        
        # Capturing an image
        cv2.imwrite(photoName, frame)
        
        print(f"✅ Photo saved as {photoName}")
    
    # Using ESC to quit
    elif key == 27:
        
        print("Exiting...")
        
        break

# Releasing and destroying the opened camera windows
camera.release()
cv2.destroyAllWindows()