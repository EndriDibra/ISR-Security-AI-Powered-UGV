import os 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# CONFIGURATION
# List of model paths to evaluate. The script will handle .keras and .tflite models.
MODEL_PATHS = [

    "globalModel/fedModelRound_7.keras",
    "globalModelLite/fedModelRound_7_float32.tflite",
    "globalModelLite/fedModelRound_7_float16.tflite",
    "globalModelLite/fedModelRound_7_int8.tflite"
]

TEST_DATASET_DIR = "globalTestDataset"
IMG_SIZE = (96, 96)
BATCH_SIZE = 32
CLASS_NAMES = ["bezos", "unknown", "zuckerberg"]


# Function to preprocess images for the TFLite model 
def preprocess_image(image, label):

    """Applies MobileNetV2-specific preprocessing to image data."""
    # The preprocess_input function handles the normalization to [-1, 1]
    image = preprocess_input(image)

    return image, label


# TFLite Model Evaluation Function
def evaluate_tflite_model(model_path, dataset_batches):

    """
    Evaluates a TFLite model on a given dataset.
    Returns: loss, accuracy, true labels, and predicted labels.
    """
    print(f"📦 Loading TFLite model from {model_path}...")

    interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=2)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()

    output_details = interpreter.get_output_details()

    y_true = []

    y_pred_probs = []

    print("🔍 Making predictions on test dataset...")

    # Check the expected input type of the model
    input_dtype = input_details[0]['dtype']

    is_int8_model = input_dtype == np.int8
    
    if is_int8_model:

        # Get quantization parameters for int8 scaling

        input_scale = input_details[0]['quantization'][0]

        input_zero_point = input_details[0]['quantization'][1]

        print(f"  > Model is quantized (int8). Using scale={input_scale}, zero_point={input_zero_point}")

    else:

        print(f"  > Model is non-quantized (float). Input dtype: {input_dtype}")

    # Iterate through the pre-loaded dataset to avoid OutOfRange errors
    for images, labels in dataset_batches:

        for i in range(images.shape[0]):

            image = images[i]

            true_label = labels[i].numpy()
            
            input_tensor = tf.expand_dims(image, 0)
            
            if is_int8_model:

                # The MobileNetV2 preprocessing scales to [-1, 1].
                # We need to scale this to [0, 1] before quantizing to int8,
                # as the model was likely quantized with this range.
                scaled_tensor = (input_tensor + 1) / 2
                
                # Quantize the float32 image to int8 using the model's parameters
                int8_tensor = np.round(scaled_tensor / input_scale + input_zero_point).astype(np.int8)
                interpreter.set_tensor(input_details[0]['index'], int8_tensor)

            else:

                # For float models, simply cast the tensor
                float_tensor = tf.cast(input_tensor, input_details[0]['dtype'])
                interpreter.set_tensor(input_details[0]['index'], float_tensor)
            
            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]['index'])[0]
            
            y_true.append(true_label)
            y_pred_probs.append(output_data)

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    accuracy = np.mean(y_true == y_pred)
    loss = None  # Loss is not provided by the TFLite interpreter

    return loss, accuracy, y_true, y_pred


# Keras Model Evaluation Function
def evaluate_keras_model(model_path, dataset_batches):

    """
    Evaluates a standard Keras model on a given dataset.
    Returns: loss, accuracy, true labels, and predicted labels.
    """
    print(f"📦 Loading Keras model from {model_path}...")

    model = tf.keras.models.load_model(model_path)

    # Convert dataset batches to a single tensor for evaluation
    images = np.concatenate([x for x, y in dataset_batches], axis=0)
    labels = np.concatenate([y for x, y in dataset_batches], axis=0)

    print("🔍 Evaluating on test dataset...")
    loss, accuracy = model.evaluate(images, labels, verbose=0)
    
    # Generate predictions to get the confusion matrix and classification report
    y_pred_probs = model.predict(images, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    return loss, accuracy, labels, y_pred


# Plotting functions
def plot_classification_metrics(report, class_names, title):

    """Plots precision, recall, and f1-score from a classification report."""
    report_df = pd.DataFrame(report).transpose()

    report_df = report_df.drop(["accuracy", "macro avg", "weighted avg"])
    
    metrics_df = report_df.iloc[:, 0:3].stack().reset_index()
    metrics_df.columns = ['class', 'metric', 'value']

    plt.figure(figsize=(10, 6))
    sns.barplot(x='class', y='value', hue='metric', data=metrics_df, palette="viridis")
    plt.title(f'Classification Report Metrics - {title}', fontsize=16)
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Class', fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend(title='Metric')
    plt.show()


def plot_confusion_matrix_heatmap(y_true, y_pred, class_names, title):

    """Plots a confusion matrix as a heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {title}', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.show()


def plot_model_comparison(results_df):

    """Plots a line chart comparing accuracy and loss for all models."""
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    sns.set_style("whitegrid")
    
    # Plot accuracy on primary y-axis as a line plot
    ax1.set_title('Model Performance Comparison: Accuracy and Loss', fontsize=16)
    ax1.set_ylabel('Accuracy', color='tab:blue', fontsize=12)
    ax1.set_ylim(0, 1.0)
    
    accuracy_plot = sns.lineplot(x='Model', y='Accuracy', data=results_df, marker='o', color='tab:blue', ax=ax1, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    # Plot loss on secondary y-axis as a line plot
    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', color='tab:red', fontsize=12)
    
    # Filter out models with no loss value
    loss_data = results_df.dropna(subset=['Loss'])
    
    loss_plot = sns.lineplot(x='Model', y='Loss', data=loss_data, marker='o', color='tab:red', ax=ax2, label='Loss')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Add labels to the points for accuracy
    for index, row in results_df.iterrows():

        ax1.annotate( f'{row["Accuracy"]:.2f}',
                        (row["Model"], row["Accuracy"]),
                        ha='center', va='bottom',
                        xytext=(0, 5), textcoords='offset points',
                        fontsize=10, color='tab:blue' )
    
    # Add labels to the points for loss
    for index, row in loss_data.iterrows():
       
        ax2.annotate(f'{row["Loss"]:.2f}', (row["Model"], row["Loss"]), ha='center', va='bottom',
                     xytext=(0, 5), textcoords='offset points',
                     fontsize=10, color='tab:red')

    fig.tight_layout()
    
    # Create combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    
    ax1.get_legend().remove() # Remove the first legend
    plt.show()


if __name__ == "__main__":

    # Load the test dataset once
    print(f"📁 Loading test data from {TEST_DATASET_DIR}...")

    test_ds = tf.keras.utils.image_dataset_from_directory(

        TEST_DATASET_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    # Preprocess the dataset
    test_ds = test_ds.map(preprocess_image)
    
    # Collect all batches into a list to avoid iterator issues
    print("Collecting dataset batches into memory...")
    test_ds_batches = list(test_ds)

    # List to store results for comparison plot
    all_results = []

    # Loop through each model path and evaluate
    for model_path in MODEL_PATHS:

        # Get the model name from the path for plot titles
        model_name = os.path.basename(model_path)
        
        if model_path.endswith(".tflite"):

            loss, accuracy, y_true, y_pred = evaluate_tflite_model(model_path, test_ds_batches)

        elif model_path.endswith(".keras"):

            loss, accuracy, y_true, y_pred = evaluate_keras_model(model_path, test_ds_batches)

        else:

            print(f"Skipping unknown model type: {model_path}")
            continue

        print("\n" + "="*50)
        print(f"Results for Model: {model_name}")
        print("="*50)
        print(f"✅ Global Model Accuracy: {accuracy * 100:.2f}%")

        if loss is None:

            print(f"📉 Loss: N/A (TFLite interpreter does not provide loss)")
            all_results.append({'Model': model_name, 'Accuracy': accuracy, 'Loss': np.nan})

        else:

            print(f"📉 Loss: {loss:.4f}")
            all_results.append({'Model': model_name, 'Accuracy': accuracy, 'Loss': loss})

        try:

            report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
            print("\n📊 Classification Report:")
            print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
            
            # Plot the results
            plot_classification_metrics(report, CLASS_NAMES, model_name)
            plot_confusion_matrix_heatmap(y_true, y_pred, CLASS_NAMES, model_name)

        except ValueError as e:

            print(f"Could not generate classification report or plots: {e}")
            print("This can happen if a model predicts none of the classes, resulting in an empty confusion matrix.")

    # Convert results to a DataFrame and plot the comparison
    if all_results:

        results_df = pd.DataFrame(all_results)
        
        print("\nGenerating model comparison plot...")
        plot_model_comparison(results_df) 