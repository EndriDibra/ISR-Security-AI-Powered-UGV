# Author: Endri Dibra 
# Bachelor Thesis: Smart Unmanned Ground Vehicle

# Importing the required libraries 
import os
import cv2
from PIL import Image, UnidentifiedImageError
from pathlib import Path


# Definition and configuration
datasetPath = r"C:\Users\User\Documents\AI_Robotics Projects\Industrial_UGV\PotholeDetection\dataset"

imageFormats = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
labelExt = '.txt'

imageDirNames = ['train/images', 'validation/images']
labelDirNames = ['train/labels', 'validation/labels']


# Function to check if image is valid, thus not corrupted
def CheckingImageValidity(imagePath):
    
    try:
    
        with Image.open(imagePath) as img:
    
            img.verify()
    
        return cv2.imread(str(imagePath)) is not None
    
    except (UnidentifiedImageError, OSError, IOError):
    
        return False


# Function to convert .gif images to .jpg
def ConvertingGifToJpg(gifPath):
    
    try:
    
        with Image.open(gifPath) as gif:
    
            rgbImg = gif.convert('RGB')
    
            newPath = gifPath.with_suffix('.jpg')
    
            rgbImg.save(newPath)
    
            print(f"Converted {gifPath.name} to {newPath.name}")
    
            return newPath
    
    except Exception as e:
    
        print(f"Failed to convert {gifPath.name}: {e}")
    
        return None


# Function to clean a single image-label folder pair
def CleaningFolder(imageFolder, labelFolder):
    
    for imagePath in Path(imageFolder).glob("*"):
    
        suffix = imagePath.suffix.lower()

        # Skipping unsupported formats
        if suffix not in imageFormats and suffix != '.gif':
    
            print(f"Unsupported format skipped: {imagePath.name}")
            continue

        # Converting GIFs
        if suffix == '.gif':
    
            converted = ConvertingGifToJpg(imagePath)
    
            if converted:
    
                imagePath.unlink()
    
                imagePath = converted
    
            else:
    
                continue

        # Removing corrupted images
        if not CheckingImageValidity(imagePath):
    
            print(f"Corrupted image removed: {imagePath.name}")
    
            imagePath.unlink()
    
            continue

        # Removing images with missing labels
        labelPath = Path(labelFolder) / imagePath.with_suffix(labelExt).name
    
        if not labelPath.exists():
    
            print(f"Missing label removed: {imagePath.name}")
    
            imagePath.unlink()


# Main loop for cleaning all train/validation folders
for imgDirName, lblDirName in zip(imageDirNames, labelDirNames):
    
    imgFolder = os.path.join(datasetPath, imgDirName)
    lblFolder = os.path.join(datasetPath, lblDirName)
    
    print(f"\nCleaning {imgFolder}")
    
    CleaningFolder(imgFolder, lblFolder)

print("\nDataset cleaning complete.")