# Author: Endri Dibra 
# Bachelor Thesis: Smart Unmanned Ground Vehicle

# Importing the required libraries 
import os 
from ultralytics import YOLO


# Ensuring the correct path
os.chdir("C:/Users/User/Documents/AI_Robotics Projects/Industrial_UGV/PotholeDetection")

# Loading base model to train it on the datasets
# and be able to recognise licence plates
model = YOLO("yolo11n.pt")

# Starting training process
model.train(data="pothole.yaml", epochs=40, imgsz=640, seed=42)