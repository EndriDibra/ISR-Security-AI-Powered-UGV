# Author: Endri Dibra
# Bachelor Thesis: Smart Unmanned Ground Vehicle
# Application: Calibrated Black Tape Line Following

# Importing OpenCV library for image processing functions
import cv2

# Importing time module for delays and timing control
import time

# Importing serial module for serial communication with Arduino
import serial

# Importing NumPy library for numerical operations and array handling
import numpy as np

# Attempting to establish serial connection with Arduino on COM15 at 9600 baud rate
try:
    # Opening serial port COM15 with baud rate 9600 and timeout of 1 second
    arduino = serial.Serial('COM15', 9600, timeout=1)
    # Waiting for 2 seconds to allow Arduino to initialize properly
    time.sleep(2)
except serial.SerialException as e:
    # Printing error message if serial connection fails
    print(f"[Error] Could not connect to Arduino: {e}")
    # Exiting program due to failed serial connection
    exit(1)

# Initializing camera capture on device index 0 using DirectShow backend
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Checking if the camera was opened successfully
if not camera.isOpened():
    # Printing error message if camera is not accessible
    print("[Error] Camera not accessible")
    # Exiting program due to inaccessible camera
    exit(1)

# Setting the width of the camera frame to 640 pixels
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

# Setting the height of the camera frame to 360 pixels
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# Defining constant command character for moving the robot forward
FORWARD = 'f'

# Defining constant command character for moving the robot backward
BACKWARD = 'b'

# Defining constant command character for turning the robot left
LEFT = 'l'

# Defining constant command character for turning the robot right
RIGHT = 'r'

# Defining constant command character for stopping the robot
STOP = 's'

# Declaring variable to store the last command sent to Arduino to avoid repetition
lastCommand = None

# Defining function for sending commands to Arduino only if different from last sent command
def sendingCommand(command):
    # Accessing the global lastCommand variable to track previous command
    global lastCommand
    # Checking if the new command is different from the last command sent
    if command != lastCommand:
        try:
            # Sending the command string encoded as bytes over serial to Arduino
            arduino.write((command + '\n').encode())
            # Printing confirmation of command sent
            print(f"Command sent to robot: {command}")
            # Updating lastCommand to current command
            lastCommand = command
        except serial.SerialException as e:
            # Printing serial communication error message
            print(f"[Serial Error] {e}")
            # If last command was not STOP, attempt to send STOP command to Arduino
            if lastCommand != STOP:
                try:
                    arduino.write((STOP + '\n').encode())
                    lastCommand = STOP  # Updating lastCommand to STOP
                except:
                    # Silently passing if sending STOP command also fails
                    pass

# Defining function for calibrating camera offset by averaging detected black line centroids
def calibratingOffset(cam, numFrames=50):
    # Printing calibration start message to user
    print("Calibration started: Align robot on line and keep still...")

    # Creating empty list to store centroid x-coordinates detected during calibration frames
    cxValues = []

    # Looping through number of frames to collect calibration data
    for _ in range(numFrames):
        # Reading a frame from the camera
        ret, frame = cam.read()
        # Continuing loop if frame reading fails
        if not ret:
            continue

        # Applying Gaussian blur to reduce noise in the frame
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        # Converting blurred frame from BGR color space to HSV color space
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Defining lower bound of black color in HSV space
        lowerBlack = np.array([0, 0, 0])
        # Defining upper bound of black color in HSV space
        upperBlack = np.array([180, 255, 50])

        # Creating a binary mask where black pixels fall within the defined HSV range
        mask = cv2.inRange(hsv, lowerBlack, upperBlack)
        # Finding contours from the binary mask image
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Checking if any contours were found
        if contours:
            # Selecting the largest contour by area assuming it is the black line
            largest = max(contours, key=cv2.contourArea)
            # Calculating spatial moments of the largest contour to find centroid
            M = cv2.moments(largest)
            # Checking that contour area is not zero to avoid division errors
            if M['m00'] != 0:
                # Calculating centroid x-coordinate of the largest contour
                cx = int(M['m10'] / M['m00'])
                # Appending the centroid x value to the list for averaging later
                cxValues.append(cx)

        # Displaying current frame in a window named "Calibration" for visual feedback
        cv2.imshow("Calibration", frame)
        # Waiting briefly for user to press 'q' key to exit calibration early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Closing the calibration window after completing calibration or early exit
    cv2.destroyWindow("Calibration")

    # Checking if no centroids were detected during calibration frames
    if not cxValues:
        # Printing failure message if no line was detected
        print("Calibration failed: No line detected")
        # Returning zero offset if calibration failed
        return 0

    # Calculating average centroid x position from collected values
    avgCx = sum(cxValues) / len(cxValues)
    # Getting width of the last captured frame
    frameWidth = frame.shape[1]
    # Calculating calibration offset as difference between frame center and average centroid x
    calibrationOffset = frameWidth // 2 - int(avgCx)

    # Printing the calculated calibration offset value
    print(f"Calibration done! Calibration offset: {calibrationOffset} pixels")
    # Returning the calibration offset for use in frame processing
    return calibrationOffset

# Defining function for processing each camera frame and controlling robot based on line position
def processingFrame(frame, calibrationOffset, margin=25):
    # Applying Gaussian blur to reduce noise in the frame
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    # Converting blurred frame to HSV color space for color segmentation
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Defining HSV lower bound for black color
    lowerBlack = np.array([0, 0, 0])
    # Defining HSV upper bound for black color
    upperBlack = np.array([180, 255, 50])
    # Creating binary mask isolating black pixels within HSV range
    mask = cv2.inRange(hsv, lowerBlack, upperBlack)

    # Finding contours from the mask image
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Checking if any contours are found
    if contours:
        # Selecting the largest contour assuming it is the black line to follow
        largest = max(contours, key=cv2.contourArea)
        # Calculating moments to find centroid of the largest contour
        M = cv2.moments(largest)
        # Ensuring contour area is non-zero before calculating centroid
        if M['m00'] != 0:
            # Calculating centroid x-coordinate
            cx = int(M['m10'] / M['m00'])
            # Calculating centroid y-coordinate
            cy = int(M['m01'] / M['m00'])

            # Correcting centroid x position by applying calibration offset
            correctedCx = cx + calibrationOffset

            # Drawing the largest contour on the frame in blue color
            cv2.drawContours(frame, [largest], -1, (255, 0, 0), 2)
            # Drawing a yellow circle at the centroid position
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

            # Getting width of the current frame
            width = frame.shape[1]
            # Calculating center x position of the frame
            centerX = width // 2

            # Drawing green vertical lines representing left margin boundary
            cv2.line(frame, (centerX - margin, 0), (centerX - margin, frame.shape[0]), (0, 255, 0), 1)
            # Drawing green vertical lines representing right margin boundary
            cv2.line(frame, (centerX + margin, 0), (centerX + margin, frame.shape[0]), (0, 255, 0), 1)
            # Drawing red vertical line representing exact center of the frame
            cv2.line(frame, (centerX, 0), (centerX, frame.shape[0]), (0, 0, 255), 1)

            # Deciding motion command based on corrected centroid position relative to center and margin
            if correctedCx < centerX - margin:
                # Sending turn left command if centroid is left of left margin
                sendingCommand(LEFT)
            elif correctedCx > centerX + margin:
                # Sending turn right command if centroid is right of right margin
                sendingCommand(RIGHT)
            else:
                # Sending move forward command if centroid is within margin of center
                sendingCommand(FORWARD)
        else:
            # Sending stop command if contour area is zero (invalid)
            sendingCommand(STOP)
    else:
        # Sending stop command if no contours detected (no line)
        sendingCommand(STOP)

    # Returning the annotated frame for display
    return frame

# Defining main function to run calibration and line following loop
def main():
    # Calibrating camera offset before starting control loop
    calibrationOffset = calibratingOffset(camera)
    # Printing calibration offset value to user
    print(f"Using calibration offset: {calibrationOffset}")

    try:
        # Starting infinite loop to continuously capture and process frames
        while True:
            # Reading a frame from the camera
            ret, frame = camera.read()
            # Checking if frame was successfully captured
            if not ret:
                print("Error reading frame")
                break  # Exiting loop if frame capture fails

            # Processing the captured frame and sending motion commands
            processedFrame = processingFrame(frame, calibrationOffset)
            # Displaying the processed frame with annotations
            cv2.imshow('Black Line Following', processedFrame)

            # Checking if user pressed 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Exiting loop on quit key press
    except Exception as e:
        # Catching any unexpected runtime errors and printing them
        print(f"[Runtime Error] {e}")
    finally:
        # Sending stop command to robot before exiting program
        sendingCommand(STOP)
        # Releasing camera resources
        camera.release()
        # Closing all OpenCV windows
        cv2.destroyAllWindows()

# Running main function when script is executed
if __name__ == "__main__":
    main()
