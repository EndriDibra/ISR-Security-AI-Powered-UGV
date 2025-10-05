# Author: Endri Dibra 
# Bachelor Thesis: Smart Unmanned Ground Vehicle

# Importing the required libraries 
import time
import serial


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


print("Robot Control - Type the following commands:")

print("[f] - Move the robot forward")
print("[b] - Move the robot backward")
print("[l] - Turn the robot left")
print("[r] - Turn the robot right")
print("[s] - Stop the robot")
print("[e] - Exit the program")


# Tracking last command to avoid repetition
lastCommand = None

# Running the process
while True:

    # Getting user input 
    command = input("Enter a command: ").strip().lower()

    # Checking if the command is valid

    # Forward motion
    if command == "f":
    
        command = "f"
    
    # Backward motion
    elif command == "b":
    
        command = "b"
    
    # Left turn motion
    elif command == "l":
    
        command = "l"
    
    # Right turn motion
    elif command == "r":
    
        command = "r"
    
    # Stoping the robot
    elif command == "s":
    
        command = "s"
    
    # Terminating process
    elif command == "e":
    
        print("Exiting program...")
        
        break
    
    # Case of invalid input
    else:
    
        print("Invalid command. Please try again.")
        
        continue

    # Checking if input is valid and different from the last command
    if command and command != lastCommand:
        
        # Sending the command to the robot via Bluetooth 
        bluetooth.write((command + "\n").encode())
        
        print(f"Command sent: {command}")

        # Updating last command
        lastCommand = command

# Closing the Bluetooth connection when exiting
bluetooth.close()

print("Program exited.")