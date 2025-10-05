# Author: Endri Dibra
# Bachelor Thesis: Smart Unmanned Ground Vehicle
# Application: Person Following using YOLOv11n

import cv2
import time
import serial
import numpy as np
from ultralytics import YOLO

# Setup serial connection parameters
BLUETOOTH_PORT = 'COM15'  # Change to your Bluetooth port
BAUD_RATE = 9600

# Movement commands
FORWARD = 'f'
BACKWARD = 'b'
LEFT = 'l'
RIGHT = 'r'
STOP = 's'

# Command sending control
COMMAND_DELAY = 0.25  # seconds between commands to avoid flooding
PERSON_LOST_TIMEOUT = 2.0  # seconds to wait before sending stop after losing person

# Globals for command management
lastCommand = None
lastCommandTime = 0
lastPersonTime = time.time()

# Serial connection object (will be set in connect_serial)
arduino = None

def connect_serial(port, baud):
    """Try to connect to serial port with retries until successful."""
    global arduino
    while True:
        try:
            arduino = serial.Serial(port, baud, timeout=1)
            time.sleep(2)  # wait for Arduino reset/init
            print(f"[INFO] Connected to Arduino on {port}")
            break
        except serial.SerialException as e:
            print(f"[WARN] Could not connect to Arduino: {e}. Retrying in 3 seconds...")
            time.sleep(3)

def sendingCommand(command):
    """Send command to Arduino if different from last and respecting command delay."""
    global lastCommand, lastCommandTime
    now = time.time()
    if command != lastCommand and (now - lastCommandTime) > COMMAND_DELAY:
        if arduino and arduino.is_open:
            try:
                arduino.write((command + '\n').encode())
                print(f"Command sent to robot: {command}")
                lastCommand = command
                lastCommandTime = now
            except serial.SerialException as e:
                print(f"[Serial Error] {e}")
                # On serial error, try to send STOP once
                if lastCommand != STOP:
                    try:
                        arduino.write((STOP + '\n').encode())
                        lastCommand = STOP
                    except:
                        pass
        else:
            print("[WARN] Serial port not open. Cannot send command.")

def decide_command(x_center, frame_width, margin=50, area=0):
    """
    Decide movement command based on person position and size.

    Args:
        x_center: horizontal center of detected person bbox
        frame_width: width of camera frame
        margin: dead zone margin in pixels
        area: bbox area (approximate person size)

    Returns:
        command char for Arduino
    """

    center_x = frame_width // 2

    # If person too small, stop (adjust threshold as needed)
    if area < 2000:
        return STOP

    # Person left of center - margin: turn left
    if x_center < center_x - margin:
        return RIGHT
    # Person right of center + margin: turn right
    elif x_center > center_x + margin:
        return LEFT
    else:
        # Centered: move forward
        return FORWARD

def main():
    global lastPersonTime

    # Connect to Arduino with retry loop
    connect_serial(BLUETOOTH_PORT, BAUD_RATE)

    # Setup camera
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        print("[Error] Camera not accessible")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    # Load YOLOv11n model
    model = YOLO('yolo11n.pt')  # Adjust path if needed

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("[Error] Could not read frame")
                break

            results = model(frame)[0]  # Run inference

            # Filter detections for 'person' class (class id 0)
            person_boxes = []
            for det in results.boxes.data.cpu().numpy():
                cls = int(det[5])
                if cls == 0:  # person
                    x1, y1, x2, y2, score = det[:5]
                    person_boxes.append((x1, y1, x2, y2, score))

            command = STOP  # default command

            if person_boxes:
                # Reset person lost timer
                lastPersonTime = time.time()

                # Choose largest bbox (closest person)
                largest_box = max(person_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

                x1, y1, x2, y2, score = largest_box
                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)
                area = (x2 - x1) * (y2 - y1)

                # Draw bbox and center point
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (x_center, y_center), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"Person {score:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                command = decide_command(x_center, frame.shape[1], margin=50, area=area)

            else:
                # Check if person lost timeout reached -> send stop
                if time.time() - lastPersonTime > PERSON_LOST_TIMEOUT:
                    command = STOP

            sendingCommand(command)

            # Show frame with status
            cv2.putText(frame, f"Command: {command}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Person Following", frame)

            # Quit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user")
    except Exception as e:
        print(f"[Runtime Error] {e}")
    finally:
        sendingCommand(STOP)
        if camera.isOpened():
            camera.release()
        cv2.destroyAllWindows()
        if arduino and arduino.is_open:
            arduino.close()


if __name__ == "__main__":
    main()
