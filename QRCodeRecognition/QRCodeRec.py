# Author: Endri Dibra 
# Bachelor Thesis: Smart Unmanned Ground Vehicle

# Importing the required libraries
import cv2
from pyzbar import pyzbar
from datetime import datetime


# Defining file paths for authorized dataset and unauthorized logging
authorizedFile = "logs/authorizedPeople.txt"
logFile = "logs/unAuthorizedPeople.txt"


# Loading authorized QR codes from a text file
def loadAuthorizedCodes(filename):

    try:
        
        with open(filename, "r") as file:
            
            # Reading all non-empty lines and stripping whitespace
            codes = {line.strip() for line in file if line.strip()}
            
            print(f"Loaded {len(codes)} authorized QR codes.")
            
            return codes

    except FileNotFoundError:
        
        # Handling missing authorized file by starting with empty set
        print(f"Authorized file '{filename}' not found. Starting with empty authorized set.")
        
        return set()


# Loading previously logged unauthorized QR codes to avoid duplicate logging
def loadLoggedUnauthorized(filename):

    logged = set()
    
    try:
        
        with open(filename, "r") as file:
            
            # Parsing each log line for unauthorized QR code entries
            for line in file:
                
                if "Unauthorized QR code detected:" in line:
                    
                    # Extracting QR code data from log line
                    qrData = line.strip().split(":")[-1].strip()
                    
                    # Adding to set of logged unauthorized codes
                    logged.add(qrData)

    except FileNotFoundError:
        
        # Handling missing log file by returning empty set
        pass

    return logged


# Loading authorized and logged unauthorized codes at program start
authorizedQRCodes = loadAuthorizedCodes(authorizedFile)
loggedUnauthorized = loadLoggedUnauthorized(logFile)


# Logging unauthorized QR code detections with timestamp if not already logged
def logUnauthorized(qrData):

    if qrData in loggedUnauthorized:
        
        # Skipping logging if QR code already logged
        return

    # Getting current timestamp string
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Appending unauthorized detection event to log file
    with open(logFile, "a") as file:
        
        file.write(f"{timestamp} - Unauthorized QR code detected: {qrData}\n")

    # Adding QR code to logged set to prevent repeated logging
    loggedUnauthorized.add(qrData)

    # Printing alert message to console
    print(f"Unauthorized QR code detected and logged: {qrData}")


# Detecting QR codes in frame and checking authorization
def checkQRCodes(frame):

    # Decoding all barcodes and QR codes in the frame
    decodedObjects = pyzbar.decode(frame)

    for obj in decodedObjects:

        # Decoding QR code data from bytes to string
        qrData = obj.data.decode("utf-8")
        
        # Getting polygon points of detected code bounding box
        points = obj.polygon
        
        # Creating list of (x, y) tuples for polygon vertices
        pts = [(point.x, point.y) for point in points]
        
        # Closing the polygon by appending first point to end of list
        pts.append(pts[0])

        # Drawing bounding polygon around detected code on the frame
        for i in range(len(pts) - 1):

            cv2.line(frame, pts[i], pts[i + 1], (0, 255, 0), 2)

        if qrData in authorizedQRCodes:

            # Preparing authorized label text
            labelText = f"Authorized: {qrData}"

            # Setting color to green for authorized codes
            color = (0, 255, 0)

        else:

            # Preparing unauthorized label text
            labelText = f"Unauthorized: {qrData}"

            # Setting color to red for unauthorized codes
            color = (0, 0, 255)

            # Logging unauthorized QR code detection
            logUnauthorized(qrData)

        # Putting label text slightly above bounding polygon on frame
        cv2.putText(frame, labelText, (pts[0][0], pts[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Returning the annotated frame
    return frame


# Capturing video from camera and running QR code detection loop
def main():

    # Opening camera using DirectShow backend
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Checking if camera opened successfully
    if not camera.isOpened():

        print("Error! Could not open camera.")

        exit()

    # Starting the video frame reading loop
    while camera.isOpened():

        # Reading a frame from the camera
        success, frame = camera.read()

        # Handling failed frame read
        if not success:

            print("Error! Failed to read frame.")

            break

        # Detecting and annotating QR codes in the frame
        frame = checkQRCodes(frame)

        # Displaying the frame with annotations
        cv2.imshow("QR Code Authorization Check", frame)

        # Waiting for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):

            break

    # Releasing camera and destroying all OpenCV windows
    camera.release()
    cv2.destroyAllWindows()


# Running main function when script is executed directly
if __name__ == "__main__":
    
    main()