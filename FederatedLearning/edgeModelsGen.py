# Author: Endri Dibra 
# Bsc Thesis: Smart Security UGV

# Importing the required libraries
import numpy as np
import tensorflow as tf


# Loading the best-performing Keras model from a specified path
model = tf.keras.models.load_model('globalModel/fedModelRound_7.keras')

# Converting to a TFLite model with no quantization (float32)
# Creating a new converter instance for the float32 conversion
converter_float32 = tf.lite.TFLiteConverter.from_keras_model(model)

# Converting the Keras model to a TFLite model with float32 data types
tflite_model_float32 = converter_float32.convert()

# Writing the converted TFLite model to a file
with open('fedModelRound_7_float32.tflite', 'wb') as f:

    f.write(tflite_model_float32)

# Converting to TFLite with float16 quantization 
# Creating a new converter instance for the float16 conversion
converter_float16 = tf.lite.TFLiteConverter.from_keras_model(model)

# Applying default optimizations to the converter
converter_float16.optimizations = [tf.lite.Optimize.DEFAULT]

# Specifying that the target supported types should be float16
converter_float16.target_spec.supported_types = [tf.float16]

# Converting the model to TFLite with float16 quantization
tflite_model_float16 = converter_float16.convert()

# Writing the converted TFLite model with float16 quantization to a file
with open('fedModelRound_7_float16.tflite', 'wb') as f:

    f.write(tflite_model_float16)

# Converting to TFLite with full integer quantization 
# Creating a new converter instance for the integer quantization
converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)


# Creating a generator function to provide a representative dataset for calibration
# This function is crucial for determining the dynamic range for quantization
def representative_data_gen():

    # Looping to yield a small number of data batches
    for _ in range(100):

        # Generating a random batch of data as a placeholder; this should be replaced with real data
        image_batch = np.random.rand(1, 96, 96, 3).astype(np.float32)

    # Yielding the batch of data
    yield [image_batch]


# Applying default optimizations to the converter, which includes quantization
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]

# Setting the representative dataset for the converter
converter_int8.representative_dataset = representative_data_gen

# Specifying that the target supported operations should be built-in integer operations
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Setting the input type for inference to 8-bit integers
converter_int8.inference_input_type = tf.int8

# Setting the output type for inference to 8-bit integers
converter_int8.inference_output_type = tf.int8

# Converting the model to TFLite with full integer quantization
tflite_model_int8 = converter_int8.convert()

# Writing the converted TFLite model with integer quantization to a file
with open('fedModelRound_7_int8.tflite', 'wb') as f:

    f.write(tflite_model_int8)