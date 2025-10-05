# Author: Endri Dibra 
# Bachelor Thesis: Smart Unmanned Ground Vehicle

# Importing the required libraries 
import cv2
import time
import serial
import mediapipe as mp


# Establishing Bluetooth communication with the robot 
try:

    # Using 'COM15' for my specific Bluetooth port
    bluetooth = serial.Serial('COM15', 9600)

    # Waiting 2 seconds for the Bluetooth connection to initialize
    time.sleep(2)

# In case of no connection, failure
except serial.SerialException as e:

    print(f"Bluetooth connection failed: {e}")
    
    exit(1)


# Defining a class for detecting and analyzing hand gestures
class Hand_Detector:

    # Initializing hand tracking settings
    def __init__(self, mode=False, max_hands=1, detection_con=0.5, track_con=0.5):

        # Setting static image mode or video stream mode
        self.mode = mode  

        # Defining the maximum number of hands to detect
        self.max_hands = max_hands

        # Setting minimum detection confidence
        self.detection_con = detection_con

        # Setting minimum tracking confidence
        self.track_con = track_con 

        # Loading Mediapipe's hand tracking application solution
        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(

            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )

        # Setting up utilities for drawing landmarks on the hand
        self.mp_draw = mp.solutions.drawing_utils


    # Detecting hands in a video frame and drawing landmarks
    def findHands(self, frame, draw=True):
        
        # Converting BGR to RGB
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Processing the frame for hand landmarks
        self.results = self.hands.process(frameRGB)

        # Drawing landmarks if hands are detected
        if self.results.multi_hand_landmarks:

            for handLandmarks in self.results.multi_hand_landmarks:

                if draw:

                    self.mp_draw.draw_landmarks(frame, handLandmarks, self.mp_hands.HAND_CONNECTIONS)
        
        # Returning the annotated frame
        return frame


    # Finding the positions of hand landmarks
    def findPosition(self, frame, hand_no=0, draw=True):
        
        # Initializing the list to store landmark coordinates
        lmList = []  

        if self.results.multi_hand_landmarks:

            # Selecting the hand to analyze
            hand = self.results.multi_hand_landmarks[hand_no]  

            # Looping through each landmark and converting to pixel coordinates
            for id, lm in enumerate(hand.landmark):

                height, width, _ = frame.shape

                cx, cy = int(lm.x * width), int(lm.y * height)

                # Appending landmark info to the list
                lmList.append([id, cx, cy]) 

                # Drawing a circle at each landmark position
                if draw:
                    
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Returning list of landmark positions
        return lmList  


    # Counting the number of fingers being held up
    def countFingers(self, lmList):
        
        # Defining landmark IDs for fingertips
        fingerTipIds = [4, 8, 12, 16, 20]

        # Initializing finger count
        fingers = 0

        # Checking thumb position (works best for right hand)
        if lmList[4][1] < lmList[3][1]:

            fingers += 1

        # Checking each of the other four fingers
        for id in range(1, 5):

            if lmList[fingerTipIds[id]][2] < lmList[fingerTipIds[id] - 2][2]:

                fingers += 1

        # Returning the total number of fingers up
        return fingers


# Sending a command to the robot via Bluetooth
def sendCommand(command):

    try:

        # Sending the command
        bluetooth.write((command + "\n").encode())

        # Printing the command for confirmation
        print(f"Command sent: {command}")  

     # Handling communication errors
    except Exception as e:
        
        print(f"Failed to send command: {e}")


# Defining the main function to run the gesture control loop
def main():
    
    # Opening the default camera in MS OS
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # Creating an instance of the hand detector
    detector = Hand_Detector()
    
    # Storing the last command sent, to avoid any duplicates
    lastCommand = None

    # Starting the video capture loop
    while camera.isOpened():
        
        # Reading a frame from the camera
        success, frame = camera.read()

        # Handling camera access issues
        if not success:
    
            print("Error! Camera did not open.")
            break
        
        # Flipping the frame horizontally for mirror view
        frame = cv2.flip(frame, 1)

        # Detecting hands and drawing them
        frame = detector.findHands(frame)
    
        # Getting landmark positions
        lmList = detector.findPosition(frame)

        # Resetting command for each frame
        command = None 

        if len(lmList) != 0:
          
            # Counting fingers
            fingers = detector.countFingers(lmList)

            # Displaying the result
            print("Number of fingers:", fingers)  

            # Mapping number of fingers to movement commands
            if fingers == 0:
                
                # Sending Stop command
                command = 's'  
          
            elif fingers == 1:
                
                # Sending Forward command
                command = 'f'  
          
            elif fingers == 2:
                
                # Sending Backward command
                command = 'b' 
          
            elif fingers == 3:
                
                # Sending Left command
                command = 'l'  
          
            elif fingers == 4:
                
                # Sending Right command 
                command = 'r'  

            # Sending command only if it has changed
            if command and command != lastCommand:
          
                sendCommand(command)
          
                lastCommand = command

            # Displaying number of fingers on the video frame
            cv2.putText(frame, f"Fingers: {fingers}", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        # Showing the final frame in a window
        cv2.imshow("Gesture Control", frame)

        # Breaking the loop when 'q':quit key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
           
            break

    # Releasing the camera 
    camera.release()

    # And closing all windows
    cv2.destroyAllWindows()


# Running the main function, when the Python program starts
if __name__ == "__main__":
    
    main()