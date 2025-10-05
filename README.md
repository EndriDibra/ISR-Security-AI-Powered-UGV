# ISR-Security-AI-Powered-UGV
This is my BSc Thesis Project, conducted during the summer period (from the beginning of July to the end of August) of 2025 (academic year 2024–2025) by Endri Dibra (myself) under the supervision of Dr. Athanasios Papaioannou, from the University of Athens (NKUA), Department of Digital Industry Technologies.

The project consists of a 3D-printed Ground Robot Vehicle, whose purpose is Intelligence, Surveillance, and Reconnaissance (ISR), as well as ensuring the safety of critical areas such as industrial complexes, warehouses, research labs, intelligence zones, and other similar high-importance locations.

The robot accomplishes its tasks by leveraging state-of-the-art technologies and techniques, including AI, Machine Learning models for data processing, Natural Language Processing (NLP) tasks such as an AI chatbot, Text-to-Speech (TTS) and Speech-to-Text (STT) systems, Deep Learning Computer Vision using CNN models and object detectors (TensorFlow, Keras, YOLOv11, Transfer Learning, MobileNetV2, MobileNetV3-Light), a Blockchain (Ganache) environment for logging critical events to ensure undeniability, and cryptographic protocols for data security.

A Web UI (Flask and HTML5, CSS3, JS) displays weather and sensor data readings for anomaly detection (e.g., smoke or fire occurrence). The project also implements Federated Learning for tasks such as face recognition, ensuring data isolation and integrity, and enabling the robot to recognize previously unseen faces by sharing only the model’s weights. Additionally, AI model quantization is used for optimization and improved speed (>FPS).

The robot uses an MPU6050 Accelerometer sensor module to strengthen its situational awareness, by knowing if its wheels are touching the ground or if it is tilted left, right, front, back or upside down. Also has a dynamic speed adjustment based on the terrain slope and roughness, ensuring a proper navigation without major risks.

Ultimately, its Task Offloading core intelligence is performed between the Robot UGV (Arduino MEGA and Raspberry Pi 5) and a Server (Dell Laptop), distributing computational tasks based on load to ensure robust and continuous operation without risking overflow or high latency.

This project demonstrates the fascinating outcomes achievable through the integration of multiple modern technologies into a cohesive and intelligent robotic system.
