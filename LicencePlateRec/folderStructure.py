# Author: Endri Dibra 
# Bachelor Thesis: Smart Unmanned Ground Vehicle

# Importing the required libraries 
import os 
import shutil
import random
from glob import glob


# Setting a fixed seed for reproducibility in shuffling
random.seed(42)

# Path to the folder containing PNG images
imageFolder = "Dataset/images"     

# Path to the folder containing YOLO-formatted labels
labelFolder = "Dataset/labels"     

# Creating output directories for training and validation images and labels
os.makedirs("images/train", exist_ok=True)
os.makedirs("images/val", exist_ok=True)
os.makedirs("labels/train", exist_ok=True)
os.makedirs("labels/val", exist_ok=True)

# Collecting all PNG images from the dataset
allImages = glob(os.path.join(imageFolder, "*.png"))

# Randomizing the image list to ensure unbiased splitting
random.shuffle(allImages)

# Defining the index to split into 80% training and 20% validation
splitIDX = int(len(allImages) * 0.8)

# Splitting the image paths accordingly
trainImages = allImages[:splitIDX]
validationImages = allImages[splitIDX:]


# This function is used to move images and corresponding label files to destination folders
def moveFiles(images, img_out_dir, label_out_dir):
    
    # Looping through each path in image folder
    for imgPath in images:
    
        # Extracting the image filename and base name
        filename = os.path.basename(imgPath)
        name, _ = os.path.splitext(filename)

        # Corresponding label path
        labelPath = os.path.join(labelFolder, name + ".txt")

        # Copying the image and label file to the respective folders
        shutil.copy(imgPath, os.path.join(img_out_dir, filename))
        shutil.copy(labelPath, os.path.join(label_out_dir, name + ".txt"))


# Moving training images and labels
moveFiles(trainImages, "images/train", "labels/train")

# Moving validation images and labels
moveFiles(validationImages, "images/val", "labels/val")

print("✅ Done! Your data is split into images/ and labels/")