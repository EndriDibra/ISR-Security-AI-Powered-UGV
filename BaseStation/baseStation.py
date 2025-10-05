# Author: Endri Dibra 
# Bachelor Thesis: Smart Unmanned Ground Vehicle

# Importing the required libraries 
import os
import sys
import time
import shutil
import signal
from datetime import datetime


# Setting up base folders relative to
# this program: baseStation.py location

# Script is inside BaseStation folder
baseFolder = '.'  

# Folder to store copied txt files
baseFilesFolder = os.path.join(baseFolder, 'baseFiles')  

# Folder to store copied image folders
baseImagesFolder = os.path.join(baseFolder, 'baseImages')   

# List of main folders to monitor
# and sync from relative path to this script
mainFolders = [
    
    os.path.join('..', 'FaceCounting'),
    os.path.join('..', 'FaceRecognition'),
    os.path.join('..', 'FaceMaskDetection'),
    os.path.join('..', 'FireDetection'),
    os.path.join('..', 'SensorFusionSuite'),
    os.path.join('..', 'ObjectRecognition'),
    os.path.join('..', 'PeopleCounting'),
    os.path.join('..', 'PhotoTake'),
    os.path.join('..', 'VideoTake'),
    os.path.join('..', 'PoseEstimation'),
    os.path.join('..', 'PotholeDetection'),
    os.path.join('..', 'QRCodeRecognition'),
    os.path.join('..', 'UI')
]

# Defining log file path for saving the synchronization logs
logFile = os.path.join(baseFolder, 'baseLogFile.txt')

# Flag for infinite loop
infinite = True

# Function to log messages with timestamps both to console and to a file
def logMessages(msg):
    
    # Taking the current time 
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Creating full log message
    fullMsg = f'[{timestamp}] {msg}'
    
    # Printing to console
    print(fullMsg)
    
    # Appending the message to the log file 
    with open(logFile, 'a', encoding='utf-8') as f:
        
        f.write(fullMsg + '\n')


# Function to check if source file is newer than
# destination or if destination doesn't exist
def isFileNewer(src, dst):
    
    # If destination file doesn't exist, source is considered newer
    if not os.path.exists(dst):
        
        return True
    
    # Otherwise comparing modification times
    return os.path.getmtime(src) > os.path.getmtime(dst)


# Function to copy all new or updated .txt files from
# 'logs' subfolder of a main folder
def copyLogs(folder):
    
    # Define the logs path inside the current main folder
    logsPath = os.path.join(folder, 'logs')
    
    # Checking if logs folder exists
    if os.path.exists(logsPath):
        
        # Looping through logs folder recursively
        for root, dirs, files in os.walk(logsPath):
            
            for file in files:
                
                # Processing .txt and csv files 
                if file.lower().endswith('.txt') or file.lower().endswith('.csv'):
                    
                    srcFile = os.path.join(root, file)
                    destFile = os.path.join(baseFilesFolder, file)
                    
                    # Copying file if new or updated
                    if isFileNewer(srcFile, destFile):
                    
                        shutil.copy2(srcFile, destFile)
                        
                        logMessages(f"Copied/Updated txt file: {srcFile} to {destFile}")
    else:
        
        logMessages(f"No logs folder found in {folder}")


# Function to copy new image folders inside
# 'images' subfolder of a main folder
def copyImages(folder):
    
    # Defining the images path inside the current main folder
    imagesPath = os.path.join(folder, 'images')
    
    # Checking if images folder exists
    if os.path.exists(imagesPath):
        
        # Looping through all items in images folder
        for item in os.listdir(imagesPath):
            
            srcSubdir = os.path.join(imagesPath, item)
            
            # Processing only directories of image folders
            if os.path.isdir(srcSubdir):
            
                destSubdir = os.path.join(baseImagesFolder, item)
                
                # Copying the folder if it does not already exist in destination
                if not os.path.exists(destSubdir):
            
                    shutil.copytree(srcSubdir, destSubdir)
                    
                    logMessages(f"Copied new image folder: {srcSubdir} to {destSubdir}")
            
                else:
            
                    logMessages(f"Image folder already exists, skipping: {destSubdir}")
    else:
        
        logMessages(f"No images folder found in {folder}")


# Function to exit on receiving
# termination signals like Ctrl+C
def programExit(signum, frame):
    
    # Log exit signal reception
    logMessages("Received exit signal. Shutting down gracefully...")
    
    # Exit the program
    sys.exit(0)

# Registering signal handlers for
# shutdown on SIGINT and SIGTERM
signal.signal(signal.SIGINT, programExit)
signal.signal(signal.SIGTERM, programExit)


# Main program execution
if __name__ == "__main__":
    
    # Infinite loop to keep checking
    # for updates every 2 minutes 
    while infinite:
        
        # Ensuring required destination folders exist
        os.makedirs(baseFolder, exist_ok=True)
        os.makedirs(baseImagesFolder, exist_ok=True)
        os.makedirs(baseFilesFolder, exist_ok=True)
        
        # Looping on each main folder 
        for folder in mainFolders:
            
            # Log folder check start
            logMessages(f"Checking folder: {folder}")
            
            # Checking if folder exists and list
            # contents or log missing folder
            if os.path.exists(folder):

                logMessages(f"Contents: {os.listdir(folder)}")

            else:

                logMessages("Folder missing")
            
            # Invoking functions to copy logs
            # and images if available
            copyLogs(folder)
            copyImages(folder)
        
        # Log completion of current sync cycle
        # waiting until next check
        logMessages("Process of copy check complete. Waiting 2 minutes...")
        time.sleep(120) 