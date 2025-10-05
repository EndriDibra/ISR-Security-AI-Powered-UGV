# Author: Endri Dibra 
# Project: Fire Detection using TensorFlow Lite and OpenCV

# Importing required libraries
import os
import csv
import cv2
import numpy as np
from datetime import datetime
import tensorflow as tf

# Resizing image with letterbox padding while preserving aspect ratio
def Letterbox(image, newShape=(640, 640), color=(114, 114, 114)):
    Shape = image.shape[:2]  # Getting current height and width
    Ratio = min(newShape[0] / Shape[0], newShape[1] / Shape[1])  # Calculating scale ratio
    NewUnpad = (int(Shape[1] * Ratio), int(Shape[0] * Ratio))  # Computing new width and height
    Dw = newShape[1] - NewUnpad[0]  # Calculating width padding
    Dh = newShape[0] - NewUnpad[1]  # Calculating height padding
    Dw /= 2  # Dividing padding on left and right
    Dh /= 2

    # Resizing image to new unpadded size
    Resized = cv2.resize(image, NewUnpad, interpolation=cv2.INTER_LINEAR)
    # Calculating padding values for top, bottom, left and right
    Top, Bottom = int(round(Dh - 0.1)), int(round(Dh + 0.1))
    Left, Right = int(round(Dw - 0.1)), int(round(Dw + 0.1))
    # Adding padding to resized image to get final shape
    Padded = cv2.copyMakeBorder(Resized, Top, Bottom, Left, Right, cv2.BORDER_CONSTANT, value=color)

    return Padded, Ratio, Dw, Dh

def NonMaxSuppression(Detections, IoUThreshold=0.38):
    # Checking if there are any detections to process
    if not Detections:
        # Returning empty list if no detections are present
        return []

    # Extracting bounding boxes coordinates from detections
    Boxes = np.array([det['bbox'] for det in Detections])
    # Extracting confidence scores from detections
    Scores = np.array([det['conf'] for det in Detections])

    # Splitting bounding boxes into individual corner coordinates
    X1 = Boxes[:, 0]  # Top-left x coordinate
    Y1 = Boxes[:, 1]  # Top-left y coordinate
    X2 = Boxes[:, 2]  # Bottom-right x coordinate
    Y2 = Boxes[:, 3]  # Bottom-right y coordinate

    # Calculating the area of each bounding box
    Areas = (X2 - X1 + 1) * (Y2 - Y1 + 1)
    # Sorting indices of boxes by descending confidence scores
    Order = Scores.argsort()[::-1]

    Keep = []  # List for keeping indices of selected boxes

    # Looping while there are boxes to process
    while Order.size > 0:
        # Selecting the box with the highest current confidence score
        I = Order[0]
        # Adding the index of this box to the keep list
        Keep.append(I)

        # Calculating the coordinates of the intersection rectangle 
        # between the box I and all other remaining boxes
        XX1 = np.maximum(X1[I], X1[Order[1:]])
        YY1 = np.maximum(Y1[I], Y1[Order[1:]])
        XX2 = np.minimum(X2[I], X2[Order[1:]])
        YY2 = np.minimum(Y2[I], Y2[Order[1:]])

        # Computing the width and height of the intersection rectangle,
        # ensuring no negative values by taking max with 0
        W = np.maximum(0.0, XX2 - XX1 + 1)
        H = np.maximum(0.0, YY2 - YY1 + 1)

        # Calculating the area of the intersection rectangle
        Inter = W * H

        # Computing the Intersection over Union (IoU) between box I
        # and the other boxes
        IoU = Inter / (Areas[I] + Areas[Order[1:]] - Inter)

        # Finding indices of boxes with IoU less or equal to threshold,
        # i.e. boxes that do not overlap significantly with box I
        Inds = np.where(IoU <= IoUThreshold)[0]
        # Updating the order list to only keep boxes that passed the threshold,
        # adjusting indices by adding 1 because IoU excludes box I itself
        Order = Order[Inds + 1]

    # Returning detections corresponding to the kept indices,
    # effectively filtering overlapping boxes
    return [Detections[i] for i in Keep]


# Loading TensorFlow Lite model and allocating tensors for inference
Interpreter = tf.lite.Interpreter(model_path="saved_model/best_float32.tflite")
Interpreter.allocate_tensors()

InputDetails = Interpreter.get_input_details()
OutputDetails = Interpreter.get_output_details()

InputShape = InputDetails[0]['shape']  # Example: [1, 640, 640, 3]
InputDtype = InputDetails[0]['dtype']

# Creating directories for saving fire images and logs if they don't exist
os.makedirs("images/fireImages", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Initializing CSV log file and writing header if it does not exist
LogFile = "logs/fireLog.csv"
FileExists = os.path.isfile(LogFile)
with open(LogFile, mode='a', newline='') as f:
    Writer = csv.writer(f)
    if not FileExists:
        Writer.writerow(["Timestamp", "Label", "Confidence", "Image_File"])

# Setting variables to track fire detection state and saved image count
FireDetectedLast = False
FireCounter = 0

# Opening default camera using DirectShow backend
Camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not Camera.isOpened():
    print("Error! Could not open camera.")
    exit()

# Preprocessing frame for model input: letterboxing, normalizing, converting color, expanding dims
def PreprocessFrame(Frame):
    Img, Ratio, Dw, Dh = Letterbox(Frame, newShape=(InputShape[1], InputShape[2]))
    Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
    Img = Img.astype(np.float32) / 255.0
    Img = np.expand_dims(Img, axis=0)  # Adding batch dimension
    Img = Img.astype(InputDtype)
    return Img, Ratio, Dw, Dh

# Postprocessing raw model output to extract bounding boxes in original frame coordinates
def PostprocessOutput(OutputData, OriginalFrameShape, Ratio, Dw, Dh, ConfThreshold=0.5):
    Detections = []
    OrigH, OrigW = OriginalFrameShape[:2]

    Output = np.squeeze(OutputData)  # Shape: (5, N)
    NumDetections = Output.shape[1]

    for i in range(NumDetections):
        Cx = Output[0, i]
        Cy = Output[1, i]
        W = Output[2, i]
        H = Output[3, i]
        Conf = Output[4, i]

        if Conf < ConfThreshold:
            continue

        # Converting box coordinates from model scale to original image scale
        X1 = (Cx - W / 2 - Dw) / Ratio
        Y1 = (Cy - H / 2 - Dh) / Ratio
        X2 = (Cx + W / 2 - Dw) / Ratio
        Y2 = (Cy + H / 2 - Dh) / Ratio

        # Clamping coordinates to image boundaries
        X1 = max(0, min(int(X1), OrigW - 1))
        Y1 = max(0, min(int(Y1), OrigH - 1))
        X2 = max(0, min(int(X2), OrigW - 1))
        Y2 = max(0, min(int(Y2), OrigH - 1))

        Detections.append({
            "bbox": (X1, Y1, X2, Y2),
            "conf": Conf,
            "classId": 0  # Single class 'fire'
        })

    return Detections

# Defining label map for class IDs
LabelMap = {0: "fire"}

# Running main loop for reading camera frames and detecting fire
while Camera.isOpened():
    Success, Frame = Camera.read()
    if not Success:
        print("Error! Failed to read frame.")
        break

    InputData, Ratio, Dw, Dh = PreprocessFrame(Frame)

    Interpreter.set_tensor(InputDetails[0]['index'], InputData)
    Interpreter.invoke()
    OutputData = Interpreter.get_tensor(OutputDetails[0]['index'])

    Detections = PostprocessOutput(OutputData, Frame.shape, Ratio, Dw, Dh)
    Detections = NonMaxSuppression(Detections, IoUThreshold=0.4)  # Applying NMS to reduce overlapping boxes

    Detected = False

    # Drawing bounding boxes and labels for detected fires
    for Det in Detections:
        X1, Y1, X2, Y2 = Det["bbox"]
        Confidence = Det["conf"]
        Label = LabelMap[Det["classId"]]

        if Confidence > 0.5:
            Detected = True
            # Drawing rectangle in red color for detected fire
            cv2.rectangle(Frame, (X1, Y1), (X2, Y2), (0, 0, 255), 2)
            LabelText = f"{Label}: {Confidence:.2f}"
            # Putting label text in green above the rectangle
            cv2.putText(Frame, LabelText, (X1, Y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Saving image and logging details only when fire is newly detected
            if not FireDetectedLast:
                Timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                ImageFilename = f"images/fireImages/fire{FireCounter}.jpg"
                FireCounter += 1
                cv2.imwrite(ImageFilename, Frame)
                with open(LogFile, mode='a', newline='') as f:
                    Writer = csv.writer(f)
                    Writer.writerow([Timestamp, Label, f"{Confidence:.2f}", ImageFilename])

    # Showing normal condition message when no fire detected
    if not Detected:
        cv2.putText(Frame, "Normal Condition", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    FireDetectedLast = Detected

    # Displaying the resulting frame
    cv2.imshow("Fire Detection", Frame)

    # Breaking loop when 'q' key pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing camera and closing all windows on exit
Camera.release()
cv2.destroyAllWindows()
