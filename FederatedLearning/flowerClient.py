# Author: Endri Dibra 
# Bsc thesis: Smart Security UGV

# Importing the required libraries
import os
import sys
import json
import flwr as fl 
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout


# Defining the Flower client class, inheriting from fl.client.NumPyClient
class FlowerClient(fl.client.NumPyClient):

    # Initializing the client with dataset directory and other parameters
    def __init__(self, dataset_dir, client_id="1", img_size=(96,96), batch_size=32):

        # Assigning instance variables
        self.client_id = client_id
        self.dataset_dir = dataset_dir
        self.img_size = img_size
        self.batch_size = batch_size
        
        # Loading and preprocessing the datasets
        self.train_ds, self.val_ds = self.load_data()
        
        # Building the model after the data is prepared
        self.model = self.build_model()
        
        # Printing dataset sizes for verification
        print(f"Client ID: {self.client_id}")
        print(f"Train batches: {tf.data.experimental.cardinality(self.train_ds).numpy()}")
        print(f"Validation batches: {tf.data.experimental.cardinality(self.val_ds).numpy()}")


    # Defining the method to build the MobileNetV2-based model
    def build_model(self):

        # Initializing the MobileNetV2 base model with pre-trained ImageNet weights
        baseModel = MobileNetV2(input_shape=(*self.img_size, 3), include_top=False, weights="imagenet")
        
        # Freezing the first layers of the base model
        for layer in baseModel.layers[:-47]:

            layer.trainable = False
        
        # Unfreezing the last layers of the base model for fine-tuning
        for layer in baseModel.layers[-47:]:

            layer.trainable = True

        # Adding a global average pooling layer
        x = baseModel.output
        x = GlobalAveragePooling2D()(x)
        
        # Adding a dropout layer for regularization
        x = Dropout(0.4)(x)
        
        # Adding the final dense classification layer with softmax activation
        # DENSE LAYER SET TO 3 CLASSES
        outputs = Dense(3, activation="softmax")(x)

        # Creating the final model
        model = Model(inputs=baseModel.input, outputs=outputs)

        # Compiling the model with the Adam optimizer and specified loss and metrics
        model.compile(
            
            optimizer=Adam(learning_rate=1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Returning the compiled model
        return model


    # Defining the method to load and preprocess the data
    def load_data(self):

        # Constructing the paths for the training and validation data
        data_path = os.path.join(self.dataset_dir, "train")
        val_path = os.path.join(self.dataset_dir, "validation")
        
        # Defining the data augmentation and preprocessing layers
        data_augmentation = tf.keras.Sequential([
         
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(factor=0.11),
            tf.keras.layers.RandomZoom(height_factor=0.07),

            tf.keras.layers.RandomBrightness(factor=0.03),
            tf.keras.layers.RandomContrast(factor=0.03)
        ])


        # Defining the data pipeline for training data, including augmentation and preprocessing
        def data_pipeline(image, label):

            image = data_augmentation(image, training=True)

            image = preprocess_input(image)

            return image, label


        # Defining the data pipeline for validation data, including only preprocessing
        def val_pipeline(image, label):

            image = preprocess_input(image)

            return image, label


        # Loading the training dataset from the directory
        train_ds = tf.keras.utils.image_dataset_from_directory(

            data_path,
            labels="inferred",
            label_mode="int",
            image_size=self.img_size,
            batch_size=self.batch_size,
            shuffle=True,
            class_names=["bezos", "unknown", "zuckerberg"]
        )

        # Loading the validation dataset from the directory
        val_ds = tf.keras.utils.image_dataset_from_directory(

            val_path,
            labels="inferred",
            label_mode="int",
            image_size=self.img_size,
            batch_size=self.batch_size,
            shuffle=False,
            class_names=["bezos", "unknown", "zuckerberg"]
        )
        
        # Applying the preprocessing pipelines and prefetching the data for performance
        train_ds = train_ds.map(data_pipeline, num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = val_ds.map(val_pipeline, num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)

        # Returning the preprocessed training and validation datasets
        return train_ds, val_ds


    # Defining the method to get the model's parameters (weights)
    def get_parameters(self, config):

        return self.model.get_weights()


    # Defining the method to fit the model on the local data
    def fit(self, parameters, config):

        # Setting the model's weights to the ones received from the server
        self.model.set_weights(parameters)
        
        # Getting the number of local epochs and the round number from the configuration
        epochs = config.get("local_epochs", 1)
        roundNum = config.get("round_number", -1)

        # Initializing EarlyStopping and ReduceLROnPlateau callbacks
        early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7, verbose=1)

        # Fitting the model with the local training data and callbacks
        history = self.model.fit(

            self.train_ds,
            epochs=epochs,
            validation_data=self.val_ds,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        # Evaluating the model's final performance
        final_loss, final_accuracy = self.model.evaluate(self.val_ds, verbose=0)

        # Creating the directory to save the training results
        os.makedirs("trainResults3", exist_ok=True)
        
        # Defining the filename for the results
        filename = f"trainResults3/client{self.client_id}_round{roundNum}.json"
        
        # Creating a dictionary with the training results to be saved
        save_dict = {

            "client_id": self.client_id,
            "round": roundNum,
            "epochs": epochs,
            "history": history.history,
            "final_eval": {
                "loss": final_loss,
                "accuracy": final_accuracy
            }
        }

        # Saving the results to a JSON file
        with open(filename, "w") as f:

            json.dump(save_dict, f, indent=4)

        # Returning the model weights, number of examples, and metrics
        return self.model.get_weights(), len(self.train_ds)*self.batch_size, {}


    # Defining the method to evaluate the model on the local data
    def evaluate(self, parameters, config):

        # Setting the model's weights to the ones received from the server
        self.model.set_weights(parameters)

        # Evaluating the model on the validation dataset
        loss, accuracy = self.model.evaluate(self.val_ds, verbose=0)

        # Returning the loss, number of examples, and accuracy
        return loss, len(self.val_ds)*self.batch_size, {"accuracy": accuracy}


# Checking if the script is being run directly
if __name__ == "__main__":

    # Checking for command-line arguments
    if len(sys.argv) < 2:

        # Printing usage instructions if arguments are missing and exiting
        print("Usage: python flowerClient.py <dataset_path> [client_id]")

        sys.exit(1)

    # Getting the dataset path from the first command-line argument
    dataset_path = sys.argv[1]

    # Getting the client ID from the second command-line argument or defaulting to "1"
    client_id = sys.argv[2] if len(sys.argv) > 2 else "1"

    # Creating an instance of the FlowerClient
    client = FlowerClient(dataset_dir=dataset_path, client_id=client_id)

    # Starting the Flower client
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)