# Author: Endri Dibra 
# Bachelor Thesis: Smart Unmanned Ground Vehicle

# Importing all the required libraries
import os
from PIL import Image


# Function to convert to RGB format
def convertDatasetToRgb(inputDir, outputDir):

    # Walking through the input directory
    for root, dirs, files in os.walk(inputDir):

        # Maintaining folder structure
        relPath = os.path.relpath(root, inputDir)

        saveDir = os.path.join(outputDir, relPath)

        os.makedirs(saveDir, exist_ok=True)
        
        # Looping through each subfolder
        for filename in files:

            # Processing only image files
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):

                inputPath = os.path.join(root, filename)

                outputPath = os.path.join(saveDir, filename)
                
                # Opening image file
                with Image.open(inputPath) as img:

                    # Converting palette images with transparency to RGBA first
                    if img.mode == 'P':

                        img = img.convert('RGBA')
                    
                    # Converting all non-RGB images to RGB
                    if img.mode != 'RGB':
                  
                        img = img.convert('RGB')
                    
                    # Saving converted image
                    img.save(outputPath)


# Running main program
if __name__ == "__main__":
    
    # Defining original dataset folder
    inputDatasetDir = "dataset/data"
    
    # Defining output folder for RGB images
    outputDatasetDir = "dataset/dataRgb"
    
    # Starting conversion process
    convertDatasetToRgb(inputDatasetDir, outputDatasetDir)
    
    print("Conversion completed!")