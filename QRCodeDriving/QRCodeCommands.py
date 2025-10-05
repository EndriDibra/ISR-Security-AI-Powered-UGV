# Author: Endri Dibra 
# Bachelor Thesis: Smart Unmanned Ground Vehicle

# Application: QR Code Command Scanner for Robot Control

# Importing the required libraries 
import cv2
import serial
import time
from pyzbar.pyzbar import decode


# Bluetooth serial port and baud rate
bluetoothSerialPort = 'COM15'  
baudRate = 9600

# Establishing Bluetooth Connection 
serialCon = serial.Serial(bluetoothSerialPort, baudRate)

# Waiting for connection to stabilize
time.sleep(2)  


# Function to Send Command Over Bluetooth 
def sendCommand(cmd):
    
    serialCon.write((cmd + '\n').encode())
    
    print(f"Sent command: {cmd}")


# Initializing and openning Camera 
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Preventing command repetition 
lastCommand = None

# Two seconds
cooldownTime = 2  
lastSentTime = 0

# Starting video loop
try:

    while camera.isOpened():
        
        # Reading camera frames
        success, frame = camera.read()
        
        # Case on unsuccessfull frame reading
        if not success:
            
            continue

        # Decoding QR codes in the frame
        decodedObjects = decode(frame)

        for obj in decodedObjects:
            
            # Extracting and cleaning data from QR
            data = obj.data.decode('utf-8').lower().strip()
            
            print(f"QR Detected: {data}")

            # Mapping QR text to robot commands
            QRcommands = {
                
                'go_forward': 'f',
                'go_backward': 'b',
                'turn_left': 'l',
                'turn_right': 'r',
                'stop': 's',
                'autonomous_mode': 'a',
                'manual_mode': 'm'
            }

            # If command is recognized and not recently sent
            if data in QRcommands:
                
                currentTime = time.time()
                
                if lastCommand != data or (currentTime - lastSentTime) > cooldownTime:
                    
                    sendCommand(QRcommands[data])
                    
                    lastCommand = data
                    lastSentTime = currentTime

        # Displaying the camera feed
        cv2.imshow("QR Autonomous Driving", frame)

        # Terminating process after pressing 'q':quit 
        if cv2.waitKey(1) & 0xFF == ord('q'):
           
            break

# Closing camera, Bluetooth connection and all OpenCV windows 
finally:
    
    camera.release()
    serialCon.close()
    cv2.destroyAllWindows()