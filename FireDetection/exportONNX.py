from ultralytics import YOLO 

model = YOLO("runs/detect/train/weights/best.pt")  # your trained fire detection model

# Export to ONNX
model.export(format="onnx", opset=12)