# Author: Endri Dibra 
# Description: This script reads average performance metrics from multiple
# CSV files, combines them, and generates two separate figures with bar plots
# to visually compare the performance of different models.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define the paths to your four CSV files.
# Make sure the files exist in the specified locations.
file_path1 = "modelComparisonResults/averages_keras.csv"
file_path2 = "modelComparisonResults/averages_tflite_float32.csv"
file_path3 = "modelComparisonResults/averages_tflite_float16.csv"
file_path4 = "modelComparisonResults/averages_tflite_int8.csv"


# Set a color palette for the plots
sns.set_palette("viridis")

try:
    # Read the data from each CSV file into a pandas DataFrame.
    # Add a 'source' column to each DataFrame to identify the original file.
    df1 = pd.read_csv(file_path1)
    df1['source'] = os.path.basename(file_path1)

    df2 = pd.read_csv(file_path2)
    df2['source'] = os.path.basename(file_path2)

    df3 = pd.read_csv(file_path3)
    df3['source'] = os.path.basename(file_path3)

    df4 = pd.read_csv(file_path4)
    df4['source'] = os.path.basename(file_path4)

    # Combine all DataFrames into a single one for plotting
    df = pd.concat([df1, df2, df3, df4], ignore_index=True)

    # Clean up the 'source' names for better plot readability
    df['source'] = df['source'].str.replace('averages_', '').str.replace('.csv', '', regex=False)

    # Map the cleaned source names to more descriptive, shorter labels
    name_map = {
        'keras': 'Keras',
        'tflite_float32': 'TFLite (FP32)',
        'tflite_float16': 'TFLite (FP16)',
        'tflite_int8': 'TFLite (INT8)'
    }
    df['source'] = df['source'].replace(name_map)

    # --- FIGURE 1: Core Performance and Size ---
    # Create a figure with a set of three subplots for visualization
    # Adjusted figsize to make plots smaller and added more horizontal space
    fig1, axes1 = plt.subplots(1, 3, figsize=(16, 5))
    fig1.suptitle('Model Core Performance and Size Comparison on Cloud Server', fontsize=20, fontweight='bold')

    # Plot 1: Frames Per Second (FPS)
    sns.barplot(x='source', y='fps', data=df, ax=axes1[0])
    axes1[0].set_title('Frames Per Second (FPS)', fontsize=16)
    axes1[0].set_ylabel('FPS', fontsize=12)
    axes1[0].set_xlabel('Model Type', fontsize=12)
    axes1[0].grid(axis='y', linestyle='--', alpha=0.7)
    axes1[0].tick_params(axis='x', rotation=0, labelsize=8) # Reduced labelsize

    # Plot 2: Latency (ms)
    sns.barplot(x='source', y='latency_ms', data=df, ax=axes1[1])
    axes1[1].set_title('Latency (ms)', fontsize=16)
    axes1[1].set_ylabel('Latency (ms)', fontsize=12)
    axes1[1].set_xlabel('Model Type', fontsize=12)
    axes1[1].grid(axis='y', linestyle='--', alpha=0.7)
    axes1[1].tick_params(axis='x', rotation=0, labelsize=8) # Reduced labelsize

    # Plot 3: Model Size (MB)
    sns.barplot(x='source', y='model_size_mb', data=df, ax=axes1[2])
    axes1[2].set_title('Model Size (MB)', fontsize=16)
    axes1[2].set_ylabel('Model Size (MB)', fontsize=12)
    axes1[2].set_xlabel('Model Type', fontsize=12)
    axes1[2].grid(axis='y', linestyle='--', alpha=0.7)
    axes1[2].tick_params(axis='x', rotation=0, labelsize=8) # Reduced labelsize
    
    # Adjust layout to prevent titles from overlapping and add space between plots
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig1.subplots_adjust(wspace=0.3)
    plt.show()


    # --- FIGURE 2: Resource Usage and Prediction Quality ---
    # Create a second figure with two subplots
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle('Model Resource Usage and Prediction Confidence on Cloud Server', fontsize=24, fontweight='bold')

    # Plot 4: CPU Usage (%)
    sns.barplot(x='source', y='cpu', data=df, ax=axes2[0])
    axes2[0].set_title('CPU Usage (%)', fontsize=18)
    axes2[0].set_ylabel('CPU Usage (%)', fontsize=14)
    axes2[0].set_xlabel('Model Type', fontsize=14)
    axes2[0].grid(axis='y', linestyle='--', alpha=0.7)
    axes2[0].tick_params(axis='x', rotation=0, labelsize=12)

    # Plot 5: Prediction Confidence
    sns.barplot(x='source', y='conf', data=df, ax=axes2[1])
    axes2[1].set_title('Prediction Confidence', fontsize=18)
    axes2[1].set_ylabel('Average Confidence', fontsize=14)
    axes2[1].set_xlabel('Model Type', fontsize=14)
    axes2[1].set_ylim(0, 1) # Set Y-axis from 0 to 1 for confidence scores
    axes2[1].grid(axis='y', linestyle='--', alpha=0.7)
    axes2[1].tick_params(axis='x', rotation=0, labelsize=12)
    
    # Adjust layout and display the second figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

except FileNotFoundError:
    print(f"Error: One or more files were not found. Please ensure the paths are correct.")
    print("Example: file_path1 = 'modelComparisonResults/averages_keras.csv'")

