# ISR-Security-AI-Powered-UGV

This repository contains my **BSc Thesis Project**, developed during the summer of 2025 (academic year 2024–2025) at the Department of **Digital Industry Technologies**, University of Athens, under the supervision of professor **Athanasios PapaIoannou**.

The project focuses on the design and implementation of an **AI-powered Unmanned Ground Vehicle (UGV)** for **Intelligence, Surveillance, and Reconnaissance (ISR)** missions, with emphasis on protecting critical environments such as industrial facilities, warehouses, laboratories, restricted zones, and other high-value infrastructure.

The platform combines robotics, embedded systems, artificial intelligence, cybersecurity, and edge/cloud computing into a unified autonomous security system.

---

# Project Overview

The robot is a custom **3D-printed smart ground vehicle** equipped with onboard sensing, computer vision, secure communications, and autonomous navigation capabilities.

Its mission objectives include:

- Autonomous patrol of sensitive areas  
- Threat and anomaly detection  
- Real-time environmental monitoring  
- Human interaction through voice AI  
- Secure logging of critical events  
- Edge/server cooperative intelligence  
- Adaptive movement across uneven terrain  

The system is built around a hybrid architecture using:

- **Arduino Mega** for low-level hardware control  
- **Raspberry Pi 5** for onboard AI inference  
- **Remote server** (Dell workstation/laptop) for heavy computation and task offloading  

---

# Core Technologies Used

## Artificial Intelligence & Machine Learning

The UGV integrates multiple AI pipelines:

- Deep Learning Computer Vision  
- Real-time object detection 
- CNN image classification 
- Transfer Learning  
- Face recognition systems  
- People counting and crowd awareness  
- Fire / smoke detection  
- License plate recognition  
- Pose estimation  
- Road pothole detection  
- QR code recognition and navigation  
- Line following and person following  

## Natural Language Processing

Human-robot interaction features include:

- AI chatbot assistant  
- Speech-to-Text (STT)  
- Text-to-Speech (TTS)  
- Voice command execution  

## Security & Integrity

The platform also includes:

- Blockchain event logging   
- Cryptographic communication protocols  
- Secure telemetry exchange  
- Evidence-grade incident recording  

## Federated Learning

Privacy-preserving learning was explored for face recognition tasks:

- Only model weights are shared  
- Raw data remains local  
- Supports recognition improvement across distributed devices  

## Optimization

To improve real-time deployment:

- AI model quantization  
- Faster inference speeds  
- Reduced compute load  
- Improved FPS performance on embedded hardware  

---

# Situational Awareness Features

The robot uses an MPU6050 sensor for motion awareness.

This enables detection of:

- Wheel lift / loss of contact  
- Left/right tilt  
- Front/back tilt  
- Upside-down state  
- Rough terrain conditions  
- Slope estimation  

Using this data, the UGV dynamically adjusts movement speed for safer navigation.

---

# Task Offloading Architecture

A key contribution of the project is **distributed intelligence** between:

### Edge Device (Robot)

- Low-latency control  
- Sensor fusion  
- Local AI inference  
- Immediate navigation decisions  

### Remote Server

- Heavy neural network workloads  
- Large-scale analytics  
- Model updates  
- Long-term storage  

This hybrid design increases robustness while preventing onboard overload or excessive latency.

---

# Repository Structure

## Main Modules

- `Arduino/` → Embedded firmware for motors, sensors, robotics control  
- `BaseStation/` → Command center / remote operations logic  
- `UI/` → Web dashboard (Flask + HTML/CSS/JS)  
- `CloudEdgeInter/` → Edge-cloud communication systems  
- `Task_OffLoading_PI5/` → Distributed computing logic  
- `SensorFusionSuite/` → Multi-sensor integration  

## AI Vision Modules

- `ObjectRecognition/`  
- `FireDetection/`  
- `FaceRecognition/`  
- `FaceMaskDetection/`  
- `PeopleCounting/`  
- `LicencePlateRec/`  
- `PoseEstimation/`  
- `QRCodeRecognition/`  
- `QRCodeDriving/`  
- `PotholeDetection/`  

## Robotics Behaviors

- `LineFollowing/`  
- `PersonFollowing/`  
- `PhotoTake/`  
- `VideoTake/`  

## Intelligence Systems

- `Chatbot/`  
- `FederatedLearning/`  

## Mechanical Design

- `Design_STL_Files/` → 3D printable chassis/components  

---

# Web Dashboard

A Flask-based dashboard provides:

- Live sensor telemetry  
- Weather data integration  
- Threat alerts  
- Fire/smoke anomaly notifications  
- Robot status monitoring  
- Remote command interface  

---

# Educational Value

This thesis demonstrates how multiple modern technologies can be integrated into one cohesive autonomous system, including:

- Robotics  
- AI  
- IoT  
- Embedded Systems  
- Cybersecurity  
- Edge Computing  
- Blockchain  
- Human-Robot Interaction  

It serves as a practical example of next-generation smart security robotics.

---

# Future Improvements

Potential next steps:

- Full SLAM navigation  
- Multi-robot swarm coordination  
- Thermal vision integration  
- Night surveillance mode  
- LTE / 5G remote deployment  
- Autonomous charging dock  
- Advanced threat prediction models  

---

# Author

**Endri Dibra**  
BSc Thesis Project – Summer 2025

---
