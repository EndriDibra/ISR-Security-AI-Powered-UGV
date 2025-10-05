# Author: Endri Dibra 
# Bachelor Thesis: Smart Unmanned Ground Vehicle

# Importing the required libraries 
import time
import serial
import speech_recognition as sr


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


# Initializing recognizer for capturing voice input
recognizer = sr.Recognizer()


# Function for capturing voice input
def captureVoiceInput():

    with sr.Microphone() as source:

        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)

        print("Listening for command...")
        audio = recognizer.listen(source)

    return audio


# Function for converting audio to text
def convertVoicetoText(audio):

    try:

        text = recognizer.recognize_google(audio)
        
        print("You said:", text)

    except sr.UnknownValueError:

        text = ""
        
        print("Sorry, I didn't understand that.")

    except sr.RequestError as e:

        text = ""
        
        print("Speech recognition error:", e)

    return text.lower()


# Function to send command via Bluetooth
def sendCommand(cmd):

    try:

        bluetooth.write((cmd + "\n").encode())
        
        print(f"Command sent: {cmd}")

    except Exception as e:

        print(f"Failed to send command: {e}")


# Function for interpreting and processing voice commands
def processVoiceCommand(text):

    if "forward" in text:

        sendCommand("f")

    elif "backward" in text:

        sendCommand("b")

    elif "left" in text:

        sendCommand("l")

    elif "right" in text:

        sendCommand("r")

    elif "stop" in text:

        sendCommand("s")

    elif "goodbye" in text or "exit" in text:

        print("Exiting program. Goodbye!")

        return True

    else:

        print("Command not recognized.")

    return False


# Main loop
def main():

    endProgram = False

    # Running program until termination
    while not endProgram:

        # Listening to voice:speech
        audio = captureVoiceInput()

        # Converting to text
        text = convertVoicetoText(audio)

        # If there is a text 
        if text:

            endProgram = processVoiceCommand(text)

    # Closing Bluetooth connection
    bluetooth.close()


# Running the main function, when the Python program starts
if __name__ == "__main__":

    main()