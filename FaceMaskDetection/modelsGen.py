import tensorflow as tf
import numpy as np

# Load your best Keras model
model = tf.keras.models.load_model('best_model.keras')

# ---- 1. Convert to a TFLite model with no quantization (float32) ----
# Create a new converter for this specific task
converter_float32 = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model_float32 = converter_float32.convert()
with open('best_model_float32.tflite', 'wb') as f:
    f.write(tflite_model_float32)

# ---- 2. Convert to TFLite with float16 quantization ----
# Create a new converter for this specific task
converter_float16 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_float16.optimizations = [tf.lite.Optimize.DEFAULT]
converter_float16.target_spec.supported_types = [tf.float16]
tflite_model_float16 = converter_float16.convert()
with open('best_model_float16.tflite', 'wb') as f:
    f.write(tflite_model_float16)

# ---- 3. Convert to TFLite with full integer quantization (requires a representative dataset) ----
# Create a new converter for this specific task
converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)

# Create a generator for your representative dataset
# This should be a function that yields a small batch of your original training data.
def representative_data_gen():
    for _ in range(100):
        # Replace this with a function to load and preprocess a small batch of your actual data
        image_batch = np.random.rand(1, 224, 224, 3).astype(np.float32)
        yield [image_batch]

converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_data_gen
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type = tf.int8
converter_int8.inference_output_type = tf.int8

tflite_model_int8 = converter_int8.convert()
with open('best_model_int8.tflite', 'wb') as f:
    f.write(tflite_model_int8)