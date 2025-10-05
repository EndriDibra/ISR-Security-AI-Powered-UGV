# Author: Endri Dibra 
# Bachelor Thesis: Smart Unmanned Ground Vehicle

# Application: QR Code Generator for Robot Commands

# Importing the required library
import qrcode


# Defining the list of robot commands to encode
commands = [
    
    "go_forward", 
    "go_backward", 
    "turn_left", 
    "turn_right", 
    "stop",
    "autonomous_mode", 
    "manual_mode"
]

# Looping through each command to generate its QR code
for cmd in commands:
    
    # Creating QR image from command
    img = qrcode.make(cmd)
    
    # Saving QR code as PNG image
    img.save(f"{cmd}.png")
    
    # Confirming save operation
    print(f"Saved QR code for: {cmd}") 