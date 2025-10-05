# 📡 Importing the required libraries 
import socket
import os
import threading

# 🔧 Setting server configuration
serverIp = '0.0.0.0'  # Listening on all interfaces
serverPort = 5050
bufferSize = 4096

# 📁 Defining base directories for storing received files
baseDir = os.path.join(os.getcwd(), 'ServerData')
filesDir = os.path.join(baseDir, 'files')    # For .txt and .csv files
imagesDir = os.path.join(baseDir, 'images')  # For images and zip files

# 🏗️ Creating directories if they don't exist
os.makedirs(filesDir, exist_ok=True)
os.makedirs(imagesDir, exist_ok=True)

# 🤝 Handling incoming client connection and receiving file
def handleClient(conn):
    try:
        # 🔍 Receiving header with metadata: file_type|relative_path|filesize
        header = conn.recv(bufferSize).decode()
        fileType, relativePath, fileSize = header.split('|')
        fileSize = int(fileSize)

        # 📂 Choosing save directory based on file type
        if fileType in ['txt', 'csv']:
            saveDir = filesDir
        elif fileType in ['zip', 'jpg', 'jpeg', 'png', 'bmp', 'gif']:
            saveDir = imagesDir
        else:
            print(f"[!] Unknown file type: {fileType}, saving to 'files'")
            saveDir = filesDir

        # 📁 Constructing full file path and ensuring folder structure exists
        fullPath = os.path.join(saveDir, relativePath)
        os.makedirs(os.path.dirname(fullPath), exist_ok=True)

        # ✅ Sending READY acknowledgment to client to start sending data
        conn.send(b'READY')

        # 💾 Writing received file data to disk in chunks
        receivedBytes = 0
        with open(fullPath, 'wb') as f:
            while receivedBytes < fileSize:
                data = conn.recv(bufferSize)
                if not data:
                    break
                f.write(data)
                receivedBytes += len(data)

        print(f"[✅] Received: {relativePath} ({fileType}, {fileSize} bytes)")

    except Exception as e:
        # ⚠️ Handling any exceptions during file transfer
        print(f"[!] Error: {e}")

    finally:
        # 🔌 Closing connection with client
        conn.close()

# 🚀 Starting the TCP server and listening for incoming connections
def startServer():
    server = socket.socket()
    server.bind((serverIp, serverPort))
    server.listen(5)
    print(f"[🔌] Server listening on {serverIp}:{serverPort}...")

    # 🔁 Accepting clients in a loop and spawning new thread for each
    while True:
        conn, addr = server.accept()
        print(f"[+] Connected from {addr}")
        threading.Thread(target=handleClient, args=(conn,)).start()

# ▶️ Running the server on script execution
if __name__ == "__main__":
    startServer()
