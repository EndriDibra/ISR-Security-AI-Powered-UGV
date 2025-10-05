# Author: Endri Dibra 
# Bachelor Thesis: Smart Unmanned Ground Vehicle

# Importing the required libraries 
import os 
import csv
import json
import time
import serial
import pygame
import requests
import pandas as pd
from joblib import load
from datetime import datetime
from threading import Thread, Lock
from flask import render_template  
from flask import Flask, request, Response
from serial.tools.list_ports import comports
from requests.exceptions import ConnectionError


# Creating the Flask backend server
app = Flask(__name__)


# This function is used for port number choice to connect with the Bluetooth module
def SerialPort():
    
    # Defining all available ports
    ports = list(comports())
    
    # Checking if there are available ports to connect
    if not ports:
    
        print("❌ No serial ports found.")
    
        return None
    
    else:

        print("\n🔌 Available Serial Ports:")
    
    # Printing the info about the available ports
    for i, port in enumerate(ports):
    
        print(f"{i}: {port.device} - {port.description}")

    # Choosing the correct ComPort 
    while True:
        
        # Ensuring integer as an input value
        try:
    
            portNumber = int(input("Select the serial port number to connect: "))

            # Ensuring the right range of ports
            if portNumber >= 0 and portNumber < len(ports):
    
                return ports[portNumber].device

            # In case input is not integer
            else:
    
                print("⚠️ Invalid port number. Please try again.")
    
        except ValueError:
    
            print("⚠️ Invalid input. Enter an Integer number.")


# Clearing the file contents
open("logs/BlackBoxUGV.txt", "w", encoding="utf-8").close()

# URL of the Flask server
url = 'http://127.0.0.1:5000/receiveData'

# Defining the path to the CSV file
csvFile = 'Models/sensorData.csv'

# Loading pre-trained model and scaler
model = load('Models/randomForest_model.joblib')
scaler = load('Models/randomForest_scaler.joblib')

# Thread-safe protection, for data reading and fetching sync
dataLock = Lock()

# Variable to store the latest temperature, humidity and Gas values
latestData = {"Temperature": None, "Humidity": None, "Gas":None}


# Waiting for server to be ready after 10 chances
def waitForServer(url, retries=10, delay=1):
    
    for _ in range(retries):
    
        try:
    
            request = requests.get(url)

            # Acceptable for readiness
            if request.status_code in [200, 404]:
    
                return True
    
        except ConnectionError:
    
            time.sleep(delay)
    
    return False


# PyGame UI to send commands via Bluetooth to Arduino Mega
def keyboardControl():
    
    # Initializing PyGame GUI
    pygame.init()

    # Creating a small GUI screen 
    screen = pygame.display.set_mode((300, 300))

    pygame.display.set_caption("Robot Control")

    print("[ARROWS] Move | [SPACE] Stop | [m] Mode | [+] Speed Up | [-] Speed Down | [ESC] Exit")

    running = True
    
    # Default motor speed
    currentSpeed = 255

    # Servo control state
    servoUp = False
    servoDown = False
    servo2Up = False
    servo2Down = False

    
    # Infinite loop until event stop
    while running:
        
        # Looping through each given event
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
        
                running = False

            elif event.type == pygame.KEYDOWN:
        
                command = None

                #  Manual mode
                if event.key == pygame.K_m:

                    command = "m"

                    print("🕹️ Switched to MANUAL mode")

                # Autonomous mode
                elif event.key == pygame.K_a:

                    command = "a"  

                    print("🧠 Switched to AUTONOMOUS mode")

                # Forward
                elif event.key == pygame.K_UP:
                    
                    command = "f"  
                
                # Backward
                elif event.key == pygame.K_DOWN:

                    command = "b"  
                
                # Left
                elif event.key == pygame.K_LEFT:
                    
                        command = "l"  
                
                # Right
                elif event.key == pygame.K_RIGHT:
                    
                        command = "r"  
                
                # Stop
                elif event.key == pygame.K_SPACE:
                
                        command = "s"  

                # Override
                elif event.key == pygame.K_o:
                
                        command = "o"  

                # Quit/Reset override 
                elif event.key == pygame.K_q:
                
                        command = "q"                   
                
                # Increasing the robot's speed until maximum:255, with step 5
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                
                    if currentSpeed <= 250:
                
                        currentSpeed += 5
                
                        command = f"speed{currentSpeed}"
                
                    else:
                
                        print("⚠️ Maximum speed reached!")
                
                # Decreasing the robot's speed until minimum:0, with step 5
                elif event.key == pygame.K_MINUS:
                
                    if currentSpeed >= 5:
                
                        currentSpeed -= 5
                
                        command = f"speed{currentSpeed}"
                
                    else:
                
                        print("⚠️ Minimum speed reached!")
                
                # Servo right motion
                elif event.key == pygame.K_RIGHTBRACKET:
                
                    if not servoUp:
                
                        serialConnection.write("servo_up_start\n".encode())
                
                        servoUp = True
                
                # Servo left motion
                elif event.key == pygame.K_LEFTBRACKET:
                
                    if not servoDown:
                
                        serialConnection.write("servo_down_start\n".encode())
                
                        servoDown = True

                # Vertical servo up (camera tilt up)
                elif event.key == pygame.K_k:
                    
                    if not servo2Up:
                        
                        serialConnection.write("servo2_up_start\n".encode())
                        
                        servo2Up = True

                # Vertical servo down (camera tilt down)
                elif event.key == pygame.K_j:
                    
                    if not servo2Down:
                    
                        serialConnection.write("servo2_down_start\n".encode())
                        
                        servo2Down = True

                # Exiting PyGame UI window
                elif event.key == pygame.K_ESCAPE:
                
                    running = False
                
                    break
                
                # Sending the command via Bluetooth 
                if command:
                
                    serialConnection.write((command + "\n").encode())
                
                    print(f"Sent command: {command}")

            elif event.type == pygame.KEYUP:
                
                # Stopping movement immediately when arrow keys released
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                
                    serialConnection.write("s\n".encode())
                
                    print("Sent command: s (stop)")

                # Stopping servo movements on key release
                if event.key == pygame.K_RIGHTBRACKET and servoUp:
                
                    serialConnection.write("servo_up_stop\n".encode())
                
                    servoUp = False
                
                if event.key == pygame.K_LEFTBRACKET and servoDown:
                
                    serialConnection.write("servo_down_stop\n".encode())
                
                    servoDown = False
                
                # Stopping vertical servo up
                if event.key == pygame.K_k and servo2Up:
                    
                    serialConnection.write("servo2_up_stop\n".encode())
                    
                    servo2Up = False

                # Stopping vertical servo down
                if event.key == pygame.K_j and servo2Down:
                   
                    serialConnection.write("servo2_down_stop\n".encode())
                   
                    servo2Down = False


    # Exiting PyGame UI
    pygame.quit()

    print("Keyboard control exited.")


# Function to send data to Flask server
def sendDataToServer(data):
    
    try:
        
        response = requests.post(url, json=data)
    
        print("Server Response:", response.text)
    
    except Exception as e:
    
        print("Error sending data to server:", e)


# Function to read data from Arduino and save it to a CSV file
def readAndSaveToCsv():

    # Waiting for Flask server to start before sending data
    if not waitForServer(url):
        
        print("❌ Flask server not responding in time. Exiting data logger.")
        
        return
    
    print("✅ Flask server is up. Starting data read loop.")

    # Reading data from Arduino Mega until termination or error
    while True:
        
        # Checking if there is any data sent
        if serialConnection.in_waiting > 0:

            # Reading the data
            dataStr = serialConnection.readline().decode('utf-8').rstrip()

            # Checking if the line looks like JSON
            if '{' in dataStr and '}' in dataStr:
                
                try:
                
                    dataDict = json.loads(dataStr)

                    print("Received data from Arduino:", dataDict)
                
                except json.JSONDecodeError:
                
                    print("❌ Invalid JSON structure:", dataStr)
                
                    continue
            else:
                
                if dataStr.startswith("[BLACKBOX]"):
                    
                    # Saving data to the black box log file
                    try:
                    
                        with open("logs/BlackBoxUGV.txt", "a", encoding="utf-8") as logFile:
                    
                            timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
                    
                            logFile.write(timestamp + dataStr + "\n")
                    
                    except Exception as e:
                    
                        print("⚠️ Error writing to BlackBoxUGV.txt:", e)

                    # skipping further processing for this line
                    continue

                else:

                    # This is a plain text status/debug message from Arduino
                    print("Arduino says:", dataStr)

                    continue
            
            # Checking if the received data dictionary contains all required keys
            if all(key in dataDict for key in ["Temperature", "Humidity", "Gas"]):
                
                temperature = float(dataDict["Temperature"])

                humidity = float(dataDict["Humidity"])

                gas = dataDict["Gas"]  

                if "value" in gas and "status" in gas:
                    
                    gasValue = int(gas["value"])
                
                    gasStatus = gas["status"]
                
                else:
                
                    gasValue = -1
                
                    gasStatus = "Unknown"

                # Converting all values to a Dataframe
                inputDF = pd.DataFrame([[temperature, humidity, gasValue]], columns=["Temperature", "Humidity", "Gas"])

                # Real-time anomaly detection
                inputScaled = scaler.transform(inputDF)

                prediction = model.predict(inputScaled)[0]
                
                anomalyStatus = 1 if prediction == -1 else 0

                if anomalyStatus == 1:
                
                    print("⚠️ Anomaly Detected (Smoke/Fire Possible)")
                
                else:
                
                    print("✅ Normal Reading")
                
                # Updating the latest data
                with dataLock:
                    
                    latestData.update({
                        
                        "Temperature": temperature,
                        
                        "Humidity": humidity,
                        
                        "Gas": gasValue,
                        
                        "Gas Status": gasStatus,
                        
                        "Anomaly": anomalyStatus
                    })

                # Saving the data to the CSV file
                saveToCsv(temperature, humidity, gasValue, anomalyStatus)
                
                # Sending the data to the Flask server
                sendDataToServer(dataDict)
            
            else:
            
                print("Incomplete data received from Arduino:", dataDict)
                
                time.sleep(0.01)


# Function to save data to CSV file
def saveToCsv(temperature, humidity, gas, anomaly):
    
    fileExists = os.path.exists(csvFile)
    
    # Reading the last row, excluding timestamp
    if fileExists:
    
        try:
    
            with open(csvFile, 'r') as file:
    
                rows = list(csv.reader(file))
    
                if len(rows) > 1:
    
                    lastRow = rows[-1]
    
                    if lastRow[1:] == [str(temperature), str(humidity), str(gas), str(anomaly)]:
    
                        print("Duplicate entry detected — skipping data save.")
    
                        return
    
        except Exception as e:
    
            print(f"Error checking for duplicates: {e}")

    # Preparing timestamp and writing data
    timeStamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(csvFile, mode='a', newline='') as file:
    
        writer = csv.writer(file)
    
        if not fileExists:
    
            writer.writerow(["Timestamp", "Temperature", "Humidity", "Gas", "Anomaly"])
    
        writer.writerow([timeStamp, temperature, humidity, gas, anomaly])

# Sending data to the url
@app.route('/receiveData', methods=['POST'])
def receiveData():

    if request.method == 'POST':

        if request.headers['Content-Type'] == 'application/json':

            data = request.json

            print("Received data from Arduino:", data)

            # Processing the data as needed
            return "Data received by server!"

        else:

            return "Unsupported Media Type", 415

    else:

        return "Method Not Allowed", 405


# Root server
@app.route("/")
def index():
    
    return "IoT UGV Flask Server Running"


# Getting data from the url
@app.route('/receiveData', methods=['GET'])
def getData():

    with dataLock:

        # Returning the most recent data from the Arduino (Temperature and Humidity)
        if latestData["Temperature"] is not None and latestData["Humidity"] is not None and latestData["Gas"] is not None:
        
            return latestData
        
        else:
        
            return {"message": "No data received from Arduino yet"}, 404


# Flask route to serve the CSV file
@app.route('/csvData')
def serveCsv():

    # Checking if CSV file exists
    if os.path.exists(csvFile):
        
        with open(csvFile, 'r') as file:
        
            csvContent = file.read()
        
        return Response(csvContent, mimetype='text/csv')
    
    else:
    
        return "CSV file not found", 404
    

# A dashboard to store and visualise the
# environmental data and the anomaly status
@app.route("/dashboard")
def dashboard():
    
    return render_template("dashboard.html")


# Running the main program
def main():
    
    # Selecting COM port
    selectedPort = SerialPort()

    if selectedPort is None:
    
        print("Exiting due to no port.")
    
        return

    print("Port selected: ", selectedPort)

    try:
    
        global serialConnection
    
        serialConnection = serial.Serial(selectedPort, 9600, timeout=1)
    
        time.sleep(2)
    
    except serial.SerialException as e:
    
        print(f"❌ Failed to connect to serial port: {e}")
    
        return

    # Running the Flask server thread
    serverThread = Thread(target=app.run, kwargs={'host':'0.0.0.0', 'port':5000, 'threaded': True})
    serverThread.start()

    # Running the read data from arduino thread
    readDataThread = Thread(target=readAndSaveToCsv)
    readDataThread.start()

    # Starting PyGame only after safe connection
    keyboardControl()


# Running main function
if __name__ == "__main__":
    
    main()