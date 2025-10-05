# Author: Endri Dibra 
# Bachelor Thesis: Smart Unmanned Ground Vehicle

# Application: Dataset Cleaning for Pothole Detection YOLOv11n Training

# Importing the required libraries
import os


# Path to the root dataset folder containing images and labels subfolders
datasetPath = 'C:/Users/User/Documents/AI_Robotics Projects/Industrial_UGV/PotholeDetection/dataset'

# List of valid image extensions to check for corresponding images
imageExtensions = ['.jpg', '.jpeg', '.png']

# Counter for deleted label files without matching images
deletedLabelsCount = 0

# Looping through all subfolders in dataset recursively
for root, dirs, files in os.walk(datasetPath):
    
    # Filtering label files in current directory
    labelFiles = [f for f in files if f.endswith('.txt')]
    
    # For each label file, checking if corresponding image exists
    for labelFile in labelFiles:
        
        labelPath = os.path.join(root, labelFile)
        baseName = os.path.splitext(labelFile)[0]
        
        # Checking current folder for images first
        imageExists = False
        
        for ext in imageExtensions:
        
            imagePathCurrent = os.path.join(root, baseName + ext)
        
            if os.path.exists(imagePathCurrent):
        
                imageExists = True
        
                break
        
        # If not found in current folder, check for 'images' folder parallel to current folder
        if not imageExists:
            
            parentDir = os.path.dirname(root)
            
            imagesFolder = os.path.join(parentDir, 'images')
            
            for ext in imageExtensions:
            
                imagePathParallel = os.path.join(imagesFolder, baseName + ext)
            
                if os.path.exists(imagePathParallel):
            
                    imageExists = True
            
                    break
        
        # If still not found, delete the label file
        if not imageExists:
            
            print(f"Deleting label file without matching image: {labelPath}")
            
            os.remove(labelPath)
            
            deletedLabelsCount += 1

print(f"Deleted {deletedLabelsCount} label files without matching images.") 