# Author: Endri Dibra 
# Bsc Thesis: Smart Security UGV

# Importing the required libraries
import os
from PIL import Image


# Defining the function to resize all images in a given dataset directory
def resize_images_in_dataset(dataset_path, new_size=(96, 96)):

    # Checking if the provided dataset path exists
    if not os.path.exists(dataset_path):

        print(f"Error: Dataset path '{dataset_path}' not found.")

        return

    # Iterating through all subdirectories and files in the dataset path
    for subdir, dirs, files in os.walk(dataset_path):

        print(f"Processing folder: {subdir}")

        # Iterating through each file in the current subdirectory
        for filename in files:

            # Checking for common image file extensions
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):

                # Constructing the full file path
                filepath = os.path.join(subdir, filename)

                try:

                    # Opening the image file using a 'with' statement for proper resource management
                    with Image.open(filepath) as img:

                        # Resizing the image using the LANCZOS filter for high quality
                        resized_img = img.resize(new_size, Image.LANCZOS)

                        # Saving the resized image back to the same path, overwriting the original
                        resized_img.save(filepath)

                        print(f"Resized image: {filepath}")

                except Exception as e:

                    # Catching and printing any errors that occur during processing a specific image
                    print(f"Could not process image {filepath}. Error: {e}")


# Checking if the script is being executed as the main program
if __name__ == "__main__":

    # The name of the dataset directory to be processed
    dataset_name = "globalTestDataset"

    # Calling the function to resize images in the specified dataset
    resize_images_in_dataset(dataset_name)

    print("Image resizing process complete.")