# Author: Endri Dibra  
# Bsc thesis: Smart Security UGV

# Importing the required libraries
import os
import datetime
import flwr as fl 
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from flwr.common import parameters_to_ndarrays
from typing import List, Tuple, Dict, Optional
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout


# Building MobileNetV2-based model
def build_model(num_classes: int) -> tf.keras.Model:
   
    # Initializing the MobileNetV2 base model with pre-trained ImageNet weights, excluding the top classification layer, and specifying the input shape
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(96, 96, 3))
    
    # Iterating through the first layers of the base model to set their trainability to False
    for layer in base_model.layers[:-47]:
   
        layer.trainable = False
        
    # Iterating through the last layers of the base model to set their trainability to True
    for layer in base_model.layers[-47:]:
   
        layer.trainable = True

    # Applying a global average pooling to the output of the base model
    x = GlobalAveragePooling2D()(base_model.output)
    
    # Adding a dropout layer for regularization, preventing overfitting
    x = Dropout(0.4)(x)
    
    # Adding the final dense layer for classification, with the number of classes and a softmax activation function
    # DENSE LAYER SET TO 3 CLASSES
    output = Dense(num_classes, activation="softmax")(x)

    # Creating the final Keras model with the specified input and output layers
    model = Model(inputs=base_model.input, outputs=output)

    # Compiling the model with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy metrics
    model.compile(
        
        optimizer=Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Returning the compiled model
    return model


# Converting Flower parameters to NumPy arrays
def parameters_to_weights(parameters) -> list:

    return parameters_to_ndarrays(parameters)


# Defining a custom strategy for saving the model
class SaveModelStrategy(fl.server.strategy.FedAvg):

    # Initializing the class with the model and total number of rounds
    def __init__(self, model: tf.keras.Model, total_rounds: int, *args, **kwargs):

        # Calling the constructor of the parent class
        super().__init__(*args, **kwargs)

        # Assigning the model and total rounds to instance variables
        self.model = model
        self.total_rounds = total_rounds

    # Aggregating the model weights from all clients after a fitting round
    def aggregate_fit(

        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[Tuple[fl.common.Parameters, Dict]]:

        # Calling the parent class's aggregation method
        aggregated = super().aggregate_fit(rnd, results, failures)

        # Checking if aggregation was successful
        if aggregated is not None:

            # Unpacking the parameters and metrics from the aggregated results
            parameters, _metrics = aggregated

            # Converting the parameters to a list of NumPy arrays
            weights = parameters_to_weights(parameters)

            # Setting the aggregated weights to the local model
            self.model.set_weights(weights)
            
            # Creating directories for saving the models if they don't already exist
            os.makedirs("globalModel", exist_ok=True)
            os.makedirs("globalModelLite", exist_ok=True)

            # Checking if the current round is the last round
            if rnd == self.total_rounds:

                # Defining the path for the Keras model
                model_path = f"globalModel/fedModelRound_{rnd}.keras"

                # Saving the Keras model
                self.model.save(model_path)

                # Printing a confirmation message
                print(f"[Server] ✅ Keras model saved to {model_path}")

                # Initializing the TFLite converter from the Keras model
                converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

                # Applying default optimizations to the converter
                converter.optimizations = [tf.lite.Optimize.DEFAULT]

                # Converting the model to TFLite format
                tflite_model = converter.convert()

                # Defining the path for the TFLite model
                tflite_path = f"globalModelLite/fedModelRound_{rnd}.tflite"

                # Opening the file in write-binary mode
                with open(tflite_path, "wb") as f:

                    # Writing the TFLite model to the file
                    f.write(tflite_model)

                # Printing a confirmation message
                print(f"[Server] 📱 TensorFlow Lite model saved to {tflite_path}")

        # Returning the aggregated results
        return aggregated


    # Aggregating the evaluation results from all clients
    def aggregate_evaluate(

        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[Tuple[float, Dict]]:

        # Checking if there are any results to aggregate
        if not results:

            # Returning None if there are no results
            return None

        # Initializing variables for aggregating evaluation metrics
        num_examples_total = 0
        loss_sum = 0.0
        accuracy_sum = 0.0

        # Iterating through the evaluation results from each client
        for _, eval_res in results:

            # Getting the number of examples and metrics for the current client
            n = eval_res.num_examples

            metrics = eval_res.metrics or {}

            # Accumulating the total number of examples, loss, and accuracy
            num_examples_total += n

            loss_sum += eval_res.loss * n

            accuracy_sum += metrics.get("accuracy", 0.0) * n

        # Calculating the average loss and accuracy
        loss_avg = loss_sum / num_examples_total

        accuracy_avg = accuracy_sum / num_examples_total

        # Creating a dictionary of aggregated metrics
        metrics_aggregated = {

            "loss": loss_avg,
            "accuracy": accuracy_avg,
        }

        # Printing the aggregated metrics to the console
        print(f"[Server] Round {rnd} evaluation metrics:")

        for k, v in metrics_aggregated.items():

            print(f"  {k}: {v:.4f}")

        # Getting the current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Creating the logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        # Appending the evaluation metrics to a log file
        with open("logs/federatedLearningMetrics.txt", "a") as f:

            f.write(f"Round {rnd} - {timestamp}\n")

            for k, v in metrics_aggregated.items():

                f.write(f"{k}: {v:.4f}\n")

            f.write("\n")

        # Returning the average loss and the aggregated metrics
        return loss_avg, metrics_aggregated


# Configuring the fitting process for each round
def on_fit_config(rnd: int):

    # Returning a dictionary with configuration parameters for the client's fit function
    return {

        "local_epochs": 4,
        "batch_size": 32,
        "cid": str(rnd),
        "round_number": rnd
    }


# Configuring the evaluation process for each round
def on_evaluate_config(rnd: int):

    # Returning a dictionary with configuration parameters for the client's evaluate function
    return {

        "val_steps": 10,
        "cid": str(rnd),
        "round_number": rnd
    }


# Checking if the script is being run directly
if __name__ == "__main__":

    # Creating the logs directory if it doesn't already exist
    os.makedirs("logs", exist_ok=True) 
    
    # Defining the number of classes and total number of rounds
    NUM_CLASSES = 3
    TOTAL_ROUNDS = 11

    # Building the model with the specified number of classes
    model = build_model(NUM_CLASSES)

    # Initializing the custom strategy for the federated learning process
    strategy = SaveModelStrategy(
        
        # Input: The actual model 
        model=model,

        # Total federated learning rounds of training
        total_rounds=TOTAL_ROUNDS,
        
        # Setting the fraction of clients to be included in each round for fitting
        fraction_fit=1.0,

        # Setting the fraction of clients to be included in each round for evaluation
        fraction_evaluate=1.0,
        
        # Setting the minimum number of clients required for fitting
        min_fit_clients=2,

        # Setting the minimum number of clients required for evaluation
        min_evaluate_clients=2,

        # Setting the minimum number of available clients
        min_available_clients=2,
        
        # Specifying the function to configure the fitting process
        on_fit_config_fn=on_fit_config,

        # Specifying the function to configure the evaluation process
        on_evaluate_config_fn=on_evaluate_config
    )

    # Printing a message indicating the server is starting
    print("[Server] 🚀 Starting Flower server...")

    # Starting the Flower federated learning server
    fl.server.start_server(
        
        # Defining the server's address
        server_address="0.0.0.0:8080",

        # Configuring the server with the total number of rounds
        config=fl.server.ServerConfig(num_rounds=TOTAL_ROUNDS),

        # Using the custom-defined strategy
        strategy=strategy,
    )