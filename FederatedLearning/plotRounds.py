import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_fed_learning_results_from_csv():

    """
    Reads federated learning training data from CSV files and plots the
    results for accuracy and loss.
    
    This script assumes the following file structure on your local machine:
    /Rounds/
    - csv_data_1.csv
    - csv_data_2.csv
    - csv_data_3.csv
    """
    
    # Define your file paths
    # These paths are relative to the location of this Python script.
    csv1 = "Rounds/csv_data_1.csv"
    csv2 = "Rounds/csv_data_2.csv"
    csv3 = "Rounds/csv_data_3.csv"
    
    # Read each CSV file into a DataFrame
    try:

        df_7 = pd.read_csv(csv1)
        df_11 = pd.read_csv(csv2)
        df_12 = pd.read_csv(csv3)

    except FileNotFoundError as e:

        print(f"Error: {e}. Please ensure the CSV files exist in the specified path.")

        return

    # Add a 'Model' column to each DataFrame to distinguish the data
    df_7['Model'] = '7_Rounds_0.03_Contrast'
    df_11['Model'] = '11_Rounds_0.03_Contrast'
    df_12['Model'] = '12_Rounds_0.04_Contrast'
    
    # Combine all DataFrames into a single DataFrame
    df = pd.concat([df_7, df_11, df_12], ignore_index=True)
    
    # Set the style for the plots
    sns.set_theme(style="whitegrid")
    
    # Create the figure and axes for the plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Federated Learning Model Performance Over Rounds', fontsize=16)
    
    # Plot Accuracy
    sns.lineplot(

        ax=axes[0],
        data=df,
        x='Round',
        y='accuracy',
        hue='Model',
        marker='o'
    )

    axes[0].set_title('Accuracy vs. Rounds')
    axes[0].set_xlabel('Round Number')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0.5, 1.0)

    # Plot Loss
    sns.lineplot(

        ax=axes[1],
        data=df,
        x='Round',
        y='loss',
        hue='Model',
        marker='o'
    )
    axes[1].set_title('Loss vs. Rounds')
    axes[1].set_xlabel('Round Number')
    axes[1].set_ylabel('Loss')
    axes[1].set_ylim(0, 1.0)

    # Adjust layout and display the plots
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# Run the plotting function
if __name__ == '__main__':

    plot_fed_learning_results_from_csv()