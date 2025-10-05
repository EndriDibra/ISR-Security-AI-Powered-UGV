import socket 
import os
import sys  
import time  


# 🦾 Change these to your Pi 5 IP and server port
pi_ip = '192.168.1.172'     # Pi 5 IP address here

pi_port = 5050              # Port where Pi server listens

buffer_size = 4096

# Path to your .tflite model file to send
model_dir = os.path.join(os.getcwd(), "")
model_filename = "deployContract.py"  # Change the round number accordingly
model_path = os.path.join(model_dir, model_filename)


def print_progress_bar(sent_bytes, total_bytes, elapsed_time, bar_length=40):

    progress = sent_bytes / total_bytes

    block = int(bar_length * progress)

    bar = "#" * block + "-" * (bar_length - block)

    percent = progress * 100

    sys.stdout.write(

        f"\rSending: [{bar}] {percent:6.2f}% "
        f"({sent_bytes}/{total_bytes} bytes) "
        f"Time elapsed: {elapsed_time:.2f}s"
    )

    sys.stdout.flush()


def send_file(file_path):

    filesize = os.path.getsize(file_path)
    file_type = file_path.split('.')[-1]   
    relative_path = model_filename         # Just filename, no folder structure

    # Prepare header as: "file_type|relative_path|filesize"
    header = f"{file_type}|{relative_path}|{filesize}"
    
    with socket.socket() as s:

        print(f"[+] Connecting to Pi 5 server at {pi_ip}:{pi_port}...")

        s.connect((pi_ip, pi_port))

        print("[+] Connected.")

        # Send header first
        s.send(header.encode())

        # Wait for READY response from server
        response = s.recv(buffer_size).decode()

        if response != "READY":

            print(f"[!] Unexpected response from server: {response}")

            return

        print("[+] Sending model file...")

        sent_bytes = 0
        start_time = time.time()

        # Send file data in chunks
        with open(file_path, "rb") as f:

            while True:

                bytes_read = f.read(buffer_size)

                if not bytes_read:

                    break

                s.sendall(bytes_read)

                sent_bytes += len(bytes_read)

                elapsed_time = time.time() - start_time

                print_progress_bar(sent_bytes, filesize, elapsed_time)

        print()  # Newline after progress bar completes
        print(f"[+] File '{relative_path}' sent successfully!")


if __name__ == "__main__":

    if not os.path.exists(model_path):

        print(f"[!] Model file does not exist: {model_path}")

    else:

        send_file(model_path)