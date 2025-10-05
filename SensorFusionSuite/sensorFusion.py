# Author: Endri Dibra 
# Bachelor Thesis: Smart Unmanned Ground Vehicle

# Importing the required libraries
import os
import csv
import time
import shutil
from datetime import datetime


# Defining relative paths for sensor data, fire logs, and image source folder
sensorCsv = os.path.join("..", "UI", "Models", "sensorData.csv")
fireCsv = os.path.join("..", "FireDetection", "logs", "fireLog.csv")
imageSrcDir = os.path.join("..", "FireDetection", "images", "fireImages")

# Getting base folder for this fusion script
fusionDir = os.path.dirname(__file__)

# Defining paths for output CSV log, fusion event text log, fusion images, and debug log file
detectionCsv = os.path.join(fusionDir, "logs", "fireDetection.csv")
fusionLog = os.path.join(fusionDir, "logs", "sensorFusionFire.txt")
fusionImgDir = os.path.join(fusionDir, "images", "sensorFusionImages")
debugLog = os.path.join(fusionDir, "logs", "sensorFusion.txt")

# Creating logs folder and images folder if they do not exist
os.makedirs(os.path.dirname(fusionLog), exist_ok=True)
os.makedirs(fusionImgDir, exist_ok=True)

# Initializing the fireDetection.csv file with headers if it does not exist yet
if not os.path.exists(detectionCsv):

    with open(detectionCsv, 'w', newline='') as f:

        writer = csv.writer(f)
        
        writer.writerow(['Timestamp', 'FusionLabel'])

# Initializing a counter for tracking consecutive seconds of fire detection
consecutiveFireCount = 0

# Number of consecutive seconds required for confirming fire
requiredDuration = 5  


# Defining a function for reading the last row from a CSV file
def readLastRow(filePath):

    try:

        with open(filePath, 'r') as f:

            rows = list(csv.DictReader(f))

            # Returning the last row as a dictionary or None if file is empty
            return rows[-1] if rows else None

    except FileNotFoundError:

        # Returning None if file does not exist yet
        return None


# Defining a function for logging a confirmed fire detection event
def logFireDetection(imageFilename):

    # Getting current timestamp for logging and file naming
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Appending event info to fusion text log file
    with open(fusionLog, 'a') as logFile:

        logFile.write(f'{timestamp} - Fire Detection (Fused)\n')

    # Appending event info to fusion CSV log file
    with open(detectionCsv, 'a', newline='') as csvFile:

        writer = csv.writer(csvFile)
       
        writer.writerow([timestamp, 'Fire'])

    # Copying the fire detection image from source folder to fusion images folder
    src = os.path.join(imageSrcDir, imageFilename)

    dstFilename = f"{timestamp.replace(':', '-')}_{imageFilename}"

    dst = os.path.join(fusionImgDir, dstFilename)

    if os.path.exists(src):

        shutil.copy(src, dst)

    else:

        print(f"Image not found: {src}")

# Starting the main monitoring loop 
try:

    print("Starting sensor fusion monitoring.\n")

    while True:

        # Reading the latest row from sensor data CSV
        sensorRow = readLastRow(sensorCsv)

        # Reading the latest row from fire detection CSV
        fireRow = readLastRow(fireCsv)

        # Checking if either file is empty or missing, waiting and retrying
        if not sensorRow or not fireRow:

            time.sleep(1)

            continue

        # Extracting anomaly and label values from the latest rows
        anomaly = sensorRow.get("Anomaly", "").strip()

        label = fireRow.get("Label", "").strip().lower()

        imageFile = fireRow.get("Image_File", "").strip()

        # Logging current anomaly and label readings with timestamp
        with open(debugLog, 'a') as dbg:

            dbg.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Anomaly: {anomaly}, Label: {label}\n')

        # Checking if both anomaly and label indicate fire detection
        if anomaly == "1" and label == "fire":

            consecutiveFireCount += 1

            print(f"Fusion matching {consecutiveFireCount}/{requiredDuration}")

        else:

            # Resetting the counter if condition breaks
            if consecutiveFireCount > 0:

                print("Fusion broken - resetting counter.")

            consecutiveFireCount = 0

        # Confirming fire detection after required consecutive matches
        if consecutiveFireCount >= requiredDuration:

            print("Fire confirmed via fusion!")

            logFireDetection(imageFile)

            consecutiveFireCount = 0

        # Pausing for 1 second before next check
        time.sleep(1)

# Handling graceful exit on user interrupt (Ctrl+C)
except KeyboardInterrupt:

    print("\nStopped by user.")