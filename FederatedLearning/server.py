# Author: Endri Dibra         
# Bachelor Thesis: Smart Security UGV

# Importing the required libraries
import os
import cv2
import time
import socket
import datetime
import numpy as np
import tensorflow as tf
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes, hmac
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


# Cryptographic Configuration 
# Generating a new RSA key pair for the server, once

# Private key generation
server_private_key = rsa.generate_private_key(
    
    public_exponent=65537,
    key_size=2048,
    backend=default_backend()
)

# Public key generation
server_public_key = server_private_key.public_key()

# Serializing the public key to send to the client
serialized_public_key = server_public_key.public_bytes(

    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)

# Model setup
# TFLite model
MODEL_PATH = "globalModelLite/fedModelRound_7_float16.tflite"   

# Class labels for prediction
CLASS_NAMES = ['bezos', 'unknown', 'zuckerberg']

# Threshold for accepting a prediction as confident
CONFIDENCE_THRESHOLD = 0.50

# TCP server setup
HOST = '0.0.0.0'   

# Port number for incoming connections
PORT = 5050   

# Size of each buffer read from socket
BUFFER_SIZE = 4096   

# Defining a txt file that will store client's requests
LOG_FILE = "receivedRequests.txt"

# Loading TFLite model and allocating tensors
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads=2)

# Allocating memory and preparing the model
interpreter.allocate_tensors()

# Getting input tensor details
input_details = interpreter.get_input_details()

# Getting output tensor details
output_details = interpreter.get_output_details()

# Determining input shape (batch_size, height, width, channels)
input_shape = input_details[0]['shape']


# Preprocessing face image: Resizing it,
# converting to RGB, Normalizing, adding batch dimention
def preprocess_image(img):

    # Resizing image to match model input shape
    #img = cv2.resize(img, (input_shape[2], input_shape[1]))

    # Converting image from BGR to RGB color space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalizing image values to range [0, 1]
    img = img.astype(np.float32) / 255.0

    # Adding batch dimension to the image
    img = np.expand_dims(img, axis=0)   

    # Returning preprocessed image
    return img


# Predicting using TFLite model
def predict(img):

    # Getting input data/image/frame
    input_data = preprocess_image(img)

    # Setting input tensor for inference
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Running inference on the model
    interpreter.invoke()

    # Getting output tensor from model
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # Calculating probability of most likely prediction
    prob = float(np.max(output_data))

    # Getting index of class with highest probability
    idx = int(np.argmax(output_data))

    # The result will be unknown, if probability is less than CONFIDENCE_THRESHOLD 
    label = 'Unknown' if prob < CONFIDENCE_THRESHOLD else CLASS_NAMES[idx]

    # Returning prediction label and probability
    return label, prob


# Logging request information into the text file
def log_request(timestamp, filename, label, prob, tps):

    # Writing log entry to file
    with open(LOG_FILE, "a") as f:

        # Formatting log line
        f.write(f"{timestamp} | File: {filename} | Label: {label} | Confidence: {prob:.2f} | TPS: {tps:.6f}\n")

print("Defined all functions.")


# Handling individual client connections
def handle_client(conn, addr):

    # Notifying that a client has connected
    print(f"[+] Connected: {addr}")

    try:
        
        # Cryptography Handshake process
        # Sending the server's public key to the client
        print("[+] Sending RSA public key to client...")
        
        conn.sendall(serialized_public_key)

        # Receiving the encrypted AES key and HMAC key from the client
        # Assuming a fixed size of 256 bytes per key
        encrypted_aes_key = conn.recv(256)
        encrypted_hmac_key = conn.recv(256)
        
        # Decrypting the AES and HMAC keys using the server's private key
        aes_key = server_private_key.decrypt(
            encrypted_aes_key,
        
            padding.OAEP(
        
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        hmac_key = server_private_key.decrypt(
            encrypted_hmac_key,
        
            padding.OAEP(
      
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
       
        print("[+] AES and HMAC keys received and decrypted.")

        # The beginning of a secure communication process
        # All subsequent communication will use these keys
        # The `READY` message is now an encrypted response to the key exchange
        encrypted_ready_message = b'READY'
      
        # Creating a unique IV for this message
        iv_ready = os.urandom(16)
        cipher_ready = Cipher(algorithms.AES(aes_key), modes.CBC(iv_ready), backend=default_backend())
        encryptor_ready = cipher_ready.encryptor()
      
        # Padding the ready message
        padded_ready = encrypted_ready_message + b'\x00' * (16 - len(encrypted_ready_message) % 16)
        encrypted_ready = encryptor_ready.update(padded_ready) + encryptor_ready.finalize()
      
        # Computing HMAC for integrity
        h = hmac.HMAC(hmac_key, hashes.SHA256(), backend=default_backend())
        h.update(encrypted_ready)
        integrity_ready = h.finalize()
      
        # Sending IV, encrypted data, and HMAC
        conn.sendall(iv_ready + encrypted_ready + integrity_ready)
        
        # Receiving encrypted header size and data, verifying its integrity
        # Receive the size of the header packet first (4 bytes)
        header_size_bytes = conn.recv(4)
      
        if not header_size_bytes:
      
            print("[!] Failed to receive header size.")
      
            return
      
        header_packet_size = int.from_bytes(header_size_bytes, 'big')

        # Now receive the full header packet
        header_packet = b''
      
        while len(header_packet) < header_packet_size:
      
            packet = conn.recv(header_packet_size - len(header_packet))
      
            if not packet:
      
                break
      
            header_packet += packet
        
        # The message format is now IV(16) + Encrypted_Data + HMAC(32)
        iv_from_client = header_packet[:16]
        hmac_from_client = header_packet[-32:]
        encrypted_header = header_packet[16:-32]
        
        # Verifying the HMAC for message integrity
        h_client = hmac.HMAC(hmac_key, hashes.SHA256(), backend=default_backend())
        h_client.update(encrypted_header)
        h_client.verify(hmac_from_client)
      
        print("[+] Header integrity verified.")
        
        # Decrypting the header now that its integrity is verified
        # Note: The cipher is created with the received IV, fixing the previous bug
        decryptor = Cipher(algorithms.AES(aes_key), modes.CBC(iv_from_client), 
            backend=default_backend()).decryptor()
      
        header = decryptor.update(encrypted_header) + decryptor.finalize()

        # Receiving initial header data from client
        header = header.strip(b'\x00').decode()

        # If nothing received, terminating 
        if not header:
      
            return

        # Splitting header into format, filename and size
        format_, filename, size_str = header.split('|')        

        # Parsing file size
        file_size = int(size_str)
        
        # The client will know to send the image after receiving
        # the encrypted `READY` message that we sent in the handshake.

        # Initializing byte buffer
        received_data = b''
        bytes_to_receive = file_size

        # Receiving image data until full file is received
        while bytes_to_receive > 0:
            
            # Receive the size of the image data packet first (4 bytes)
            image_packet_size_bytes = conn.recv(4)
            
            if not image_packet_size_bytes:
            
                break
            
            image_packet_size = int.from_bytes(image_packet_size_bytes, 'big')

            # Receive the full image packet
            packet = b''
            
            while len(packet) < image_packet_size:
            
                buffer = conn.recv(image_packet_size - len(packet))
            
                if not buffer:
            
                    break
            
                packet += buffer

            if not packet:
            
                break

            # The message format is now IV(16) + Encrypted_Data + HMAC(32)
            iv_packet = packet[:16]
            hmac_packet = packet[-32:]
            encrypted_data_packet = packet[16:-32]
            
            # Verifying the HMAC for message integrity
            h_packet = hmac.HMAC(hmac_key, hashes.SHA256(), backend=default_backend())
            h_packet.update(encrypted_data_packet)
            h_packet.verify(hmac_packet)

            # Decrypting the packet
            decryptor_packet = Cipher(algorithms.AES(aes_key), modes.CBC(iv_packet), 
                backend=default_backend()).decryptor()
            
            decrypted_packet = decryptor_packet.update(encrypted_data_packet) + decryptor_packet.finalize()
            
            received_data += decrypted_packet
            bytes_to_receive -= len(decrypted_packet)

        # Padding removal for the final packet
        received_data = received_data.strip(b'\x00')

        # Converting buffer to NumPy array
        img_array = np.frombuffer(received_data, dtype=np.uint8)

        # Decoding image from array
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # If decoding fails, return error
        if img is None:
            
            # Acknowledging failure with encrypted message
            response = b'ERROR'
            
            iv_error = os.urandom(16)
            
            cipher_error = Cipher(algorithms.AES(aes_key), modes.CBC(iv_error), 
                backend=default_backend()).encryptor()
            
            padded_error = response + b'\x00' * (16 - len(response) % 16)
            
            encrypted_error = cipher_error.update(padded_error) + cipher_error.finalize()
            
            h_error = hmac.HMAC(hmac_key, hashes.SHA256(), backend=default_backend())
            h_error.update(encrypted_error)
            
            integrity_error = h_error.finalize()
            
            conn.sendall(iv_error + encrypted_error + integrity_error)
            
            return

        # Recording start time for TPS calculation
        tps_start = time.perf_counter()

        # Performing prediction
        label, prob = predict(img)

        # Calculating total processing speed
        tps = time.perf_counter() - tps_start

        # Generating timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Logging request
        log_request(timestamp, filename, label, prob, tps)

        # Preparing response to client
        response = f"{label}|{prob:.2f}|{tps:.6f}"

        # Encrypting the response before sending to client
        # The IV should be unique for each message
        iv_response = os.urandom(16)
        cipher_response = Cipher(algorithms.AES(aes_key), modes.CBC(iv_response), backend=default_backend())
        encryptor = cipher_response.encryptor()
        
        # Padding the response to a multiple of 16 (AES block size)
        response_padded = response.encode() + b'\x00' * (16 - len(response.encode()) % 16)
        
        encrypted_response = encryptor.update(response_padded) + encryptor.finalize()
        
        # Computing HMAC for integrity
        h_response = hmac.HMAC(hmac_key, hashes.SHA256(), backend=default_backend())
        h_response.update(encrypted_response)
        integrity_response = h_response.finalize()
        
        # Sending the IV with the encrypted data and HMAC to client
        conn.sendall(iv_response + encrypted_response + integrity_response)

    # Catching and reporting any errors, including HMAC verification failure
    except InvalidSignature as e:
        
        print(f"[!] HMAC verification failed. Connection terminated.")
        
    except Exception as e:

        # Printing error message
        print(f"[!] Error: {e}")

    finally:

        # Closing connection
        conn.close()

        # Notifying that client has disconnected
        print(f"[-] Disconnected: {addr}")


# Starting the TCP server
def start_server():

    # Printing server startup message
    print(f"🖥️ Server running on {HOST}:{PORT}")

    # Creating socket context
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

        # Binding socket to host and port
        s.bind((HOST, PORT))

        # Starting to listen for connections
        s.listen(5)

        # Infinite loop to accept clients
        while True:

            # Accepting client connection
            conn, addr = s.accept()

            # Handling connected client
            handle_client(conn, addr)


# Entry point of the program
if __name__ == "__main__":

    # Starting the server
    start_server()