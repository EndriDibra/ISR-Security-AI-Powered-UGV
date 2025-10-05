# Author: Endri Dibra 
# Bachelor Thesis: Smart Unmanned Ground Vehicle

# Importing the required libraries 
import os
from glob import glob
import xml.etree.ElementTree as ET


# Required paths 

# XML annotations
xmlFolder = "Dataset/xml"

# PNG images
imageFolder = "Dataset/images"

# YOLO accepted format labels
outputLabelFolder = "Dataset/labels"

# Creating the output label directory
os.makedirs(outputLabelFolder, exist_ok=True)

# Class Mapping 

# Accepted class names from annotations to be considered
acceptedClasses = ['licence', 'license_plate', 'plate', 'tag']

# Mapping all variations of plate labels to a single class ID (0)
classToIndex = {

    'licence': 0,
    'license_plate': 0,
    'plate': 0,
    'tag': 0
}

# Conversion Loop 

# Iterating through each XML file in the annotation folder
for xmlFile in glob(os.path.join(xmlFolder, "*.xml")):
    
    # Parsing the XML annotation file
    tree = ET.parse(xmlFile)
    root = tree.getroot()

    # Extracting the corresponding image filename
    imgFilename = root.find("filename").text
    
    # Extracting image dimensions
    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    # List to hold YOLO formatted labels
    yoloLines = []

    # Looping through each annotated object in the XML
    for obj in root.findall("object"):
        
        # Getting the class name and converting it to lowercase
        clsName = obj.find("name").text.strip().lower()

        # Skipping labels not in accepted class list
        if clsName not in acceptedClasses:

            print(f"Skipping unknown class: {clsName}")

            continue

        # Getting class ID based on the dictionary mapping
        classID = classToIndex[clsName]

        # Extracting bounding box coordinates
        bbox = obj.find("bndbox")

        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))

        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))

        # YOLO format: center x center y width height (all normalized to [0,1])
        xCenter = ((xmin + xmax) / 2) / width
        yCenter = ((ymin + ymax) / 2) / height

        boxWidth = (xmax - xmin) / width
        boxHeight = (ymax - ymin) / height

        # Formatting the label line in YOLO format
        yoloLines.append(f"{classID} {xCenter:.6f} {yCenter:.6f} {boxWidth:.6f} {boxHeight:.6f}")

    # Defining the output .txt filename matching the image
    txtFilename = os.path.splitext(imgFilename)[0] + ".txt"

    # Writing YOLO labels to the output file
    with open(os.path.join(outputLabelFolder, txtFilename), "w") as f:
        
        f.write("\n".join(yoloLines))

print("✅ Conversion completed! The new output subfolder is in 'Dataset/labels' folder.")