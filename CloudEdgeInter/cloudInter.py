# Author: Endri Dibra
# Bachelor Thesis: Smart Unmanned Ground Vehicle
# Application: Multi-Threaded File Server for Pi Communication

# Importing required libraries
import socket
import threading
import os
from tqdm import tqdm  # For visual progress bars

# Define server IP and port to listen on
SERVER_IP = '0.0.0.0'         # Listen on all interfaces
SERVER_PORT = 9999            # Port for incoming connections
BUFFER_SIZE = 4096            # Buffer size for data chunks

# Set folder to save received files
RECEIVED_FOLDER = 'server_received'
os.makedirs(RECEIVED_FOLDER, exist_ok=True)

# Set folder for files to send to Pi
TO_SEND_FOLDER = 'server_to_send'
os.makedirs(TO_SEND_FOLDER, exist_ok=True)

# Function to handle each client connection in a separate thread
def handle_client(client_socket, address):
    print(f"📡 Connection established from {address}")
    
    try:
        # Receive initial command/header from client
        header = client_socket.recv(BUFFER_SIZE).decode()

        # === Pi wants to send file ===
        if header.startswith("SEND"):
            # Parse header
            _, filename, filesize = header.split("|")
            filename = os.path.basename(filename)
            filesize = int(filesize)

            # Define where to save the incoming file
            save_path = os.path.join(RECEIVED_FOLDER, filename)

            # Open file and start receiving data
            with open(save_path, "wb") as f, tqdm(total=filesize, unit="B", unit_scale=True, desc=f"Receiving {filename}") as pbar:
                while filesize > 0:
                    bytes_read = client_socket.recv(min(BUFFER_SIZE, filesize))
                    if not bytes_read:
                        break
                    f.write(bytes_read)
                    f.flush()
                    os.fsync(f.fileno())
                    filesize -= len(bytes_read)
                    pbar.update(len(bytes_read))
            print("✅ File received.")

        # === Pi requests to receive file ===
        elif header.startswith("RECEIVE"):
            # Check if file exists to send
            files = os.listdir(TO_SEND_FOLDER)
            if not files:
                client_socket.send(b"NOFILE")
                print("❌ No file available to send.")
            else:
                # Just send the first file in folder
                filename = files[0]
                filepath = os.path.join(TO_SEND_FOLDER, filename)
                filesize = os.path.getsize(filepath)

                # Send metadata first
                client_socket.send(f"FILE|{filename}|{filesize}".encode())

                # Send file data
                with open(filepath, "rb") as f, tqdm(total=filesize, unit="B", unit_scale=True, desc=f"Sending {filename}") as pbar:
                    while True:
                        bytes_read = f.read(BUFFER_SIZE)
                        if not bytes_read:
                            break
                        client_socket.sendall(bytes_read)
                        pbar.update(len(bytes_read))
                print("✅ File sent.")

        # === Invalid request ===
        else:
            print("❌ Unknown request received.")

    except Exception as e:
        print(f"💥 Error handling client {address}: {e}")

    finally:
        client_socket.close()
        print(f"🔌 Connection closed from {address}")

# Main function to run the threaded server
def start_server():
    # Create a TCP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((SERVER_IP, SERVER_PORT))
    server_socket.listen(5)
    print(f"🚀 Server listening on {SERVER_IP}:{SERVER_PORT}...")

    while True:
        try:
            # Accept new client
            client_socket, address = server_socket.accept()
            # Start new thread to handle this client
            thread = threading.Thread(target=handle_client, args=(client_socket, address))
            thread.start()
        except KeyboardInterrupt:
            print("🛑 Server shutting down...")
            break
        except Exception as e:
            print(f"⚠️ Server error: {e}")

    server_socket.close()

# Run the server
if __name__ == "__main__":
    start_server()
