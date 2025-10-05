# Author: Endri Dibra 
# Bachelor Thesis: Smart Unmanned Ground Vehicle

# Importing the required libraries 
import cv2
import easyocr
import numpy as np
from ultralytics import YOLO


# Enhancing plate image function 
def EnhancePlateImage(plateImg):

    # Resizing image to improve OCR accuracy (2x enlargement)
    plateImg = cv2.resize(plateImg, (plateImg.shape[1]*2, plateImg.shape[0]*2))

    # Converting to grayscale
    gray = cv2.cvtColor(plateImg, cv2.COLOR_BGR2GRAY)

    # Applying Gaussian Blur to reduce noise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Applying Otsu's thresholding to binarize the image
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Sharpening image using a kernel to highlight text
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    
    sharpened = cv2.filter2D(thresh, -1, kernel)

    # Applying morphological closing to fill small holes in characters
    kernelMorph = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    
    closed = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernelMorph)

    return closed


# Loading YOLOv11n license plate detection model
model = YOLO("runs/detect/train/weights/best.pt")

# Initializing EasyOCR reader
reader = easyocr.Reader(['en'])

# Opening camera 
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Looping through each camera frame
while camera.isOpened():

    # Reading frame from the camera
    success, frame = camera.read()

    # If frame not read properly, skip this iteration
    if not success:
        
        continue

    # Running YOLOv11n detection
    results = model(frame)[0]

    # Looping through detected bounding boxes
    for box in results.boxes:

        # Extracting license plate coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Cropping the detected plate region
        plateCrop = frame[y1:y2, x1:x2]

        # Enhancing the cropped plate image before OCR
        enhanced = EnhancePlateImage(plateCrop)

        # Performing OCR using EasyOCR
        result = reader.readtext(enhanced)

        # If text is recognized
        if result:

            # Extracting recognized text
            text = result[0][1]

            # Drawing bounding box and text label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Printing recognized license plate
            print("License Plate:", text)

    # Displaying the processed camera feed
    cv2.imshow("Licence Plate Recognition", frame)

    # Terminating process if 'q':quit key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        
        break

# Releasing the camera and closing windows
camera.release()
cv2.destroyAllWindows()