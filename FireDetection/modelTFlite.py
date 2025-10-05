import onnx
import tensorflow as tf
import subprocess
import os

# Paths
onnx_model_path = "runs/detect/train/weights/best.onnx"
saved_model_dir = "saved_model/best_float32"
tflite_model_path = "saved_model/best_float32.tflite"

# Step 1: Convert ONNX to TensorFlow SavedModel using onnx2tf CLI
print("Converting ONNX model to TensorFlow SavedModel (float32)...")
command = [
    "onnx2tf",
    "-i", onnx_model_path,
    "--output_folder", saved_model_dir
]
subprocess.run(command, check=True)
print("✅ ONNX to TensorFlow SavedModel conversion completed.")

# Step 2: Convert SavedModel to TFLite (float32, no quantization)
print("Converting TensorFlow SavedModel to TFLite (float32, no quantization)...")
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
# No quantization applied here - full float32 precision
tflite_model = converter.convert()

# Make sure the output folder exists
os.makedirs(os.path.dirname(tflite_model_path), exist_ok=True)

# Save the TFLite model
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"✅ TFLite model saved at: {tflite_model_path}")
