# YOLOv8 + Face‑Recognition Motion Alert System

---

## 1. Introduction & Motivation

Modern video‑based security solutions often suffer from false alarms—pets, shadows or minor scene changes can trigger alerts intended for genuine intruders. This project replaces hardware radar sensors with a purely vision‑based pipeline that:

1. Detects motion via frame differencing  
2. Runs a lightweight YOLOv8 model to localize people  
3. Extracts and visualizes deep convolutional features  
4. Recognizes “admin” faces from a library of selfies  
5. Plays an audible buzzer when an **unknown** person is detected (with cooldown)

By combining classical motion detection with AI‑driven object‑ and face‑recognition, we drastically reduce false alarms and provide richer insights (heatmaps, embeddings) for each event.

---

## 2. Project Objectives

- **Reliable Motion Filtering**  
  Only process frames with significant pixel changes, ignoring minor flicker or lighting shifts.  
- **Real‑Time Human Detection**  
  Use YOLOv8‑n for sub‑100 ms inference on a consumer GPU or CPU.  
- **Feature‑Map Visualization**  
  Hook into the last convolutional layer to generate a heatmap overlay, aiding model interpretability.  
- **Admin Face Recognition**  
  Enroll any number of “admin” selfies; label known users in green, unknown in red.  
- **Audible Alert with Cooldown**  
  Play a buzzer sound on unknown detections, but suppress repeated alarms within a configurable cooldown window.

---

## 3. Design Specifications

| Component               | Description                                                                                  |
|-------------------------|----------------------------------------------------------------------------------------------|
| **Motion Detection**    | Frame‑differencing (grayscale abs‑diff + threshold + pixel‑count). Configurable via `.env`. |
| **Object Detection**    | YOLOv8‑n model (via `ultralytics.YOLO`). Bounding boxes drawn on live video.                |
| **Feature Hooking**     | Forward‑hook on the penultimate `Conv2d` layer. Compute channel‑mean activation heatmap.     |
| **Face Recognition**    | `face_recognition` library. Encodings built from all images in `admin_images/`.              |
| **Alert System**        | `pygame` plays `buzzer.wav` on unknown face. Cooldown enforced in Python.                    |
| **Configuration**       | All parameters (`paths`, `thresholds`, `cooldown`, etc.) live in `.env`.                     |

---
## 4. Future Insights

1. **Embedded Deployment**  
   - Cross‑compile dependencies (PyTorch, OpenCV) for ARM on Raspberry Pi/Orange Pi.  
   - Use the lightweight `yolov8n` or convert to TFLite/ONNX for faster edge inference.  
   - Leverage the Pi Camera or USB webcam; optimize with GPU acceleration (Coral TPU, NVIDIA Jetson Nano).

2. **IoT Integration**  
   - Containerize the application (Docker) or build a minimal Python service.  
   - On alarm, send notifications via MQTT, HTTP webhook, or push services (Pushover, Twilio SMS).  
   - Expose a simple web dashboard (Flask/React) for live view, logs, and admin management.

3. **Power & Enclosure**  
   - Run headless on battery + UPS HAT for Raspberry Pi for uninterrupted operation.  
   - Design a 3D‑printed case with mounting points, ventilation, and cable management.

4. **Scalability & Cloud**  
   - Aggregate multiple devices’ alerts in the cloud (AWS IoT Core, Azure IoT Hub).  
   - Perform periodic model updates and remote configuration via OTA firmware.

---

## 5. Setup & Usage Guide

#### After cloning the repository on your machine:

### 5.1 Environment

1. Copy and edit the example .env file:
   ```bash
   cp .env.example .env
    ```
   
2. Open `.env` and adjust any setting to your preference:
    ```dotenv
   YOLO_MODEL_PATH=yolov8n.pt
    ADMIN_IMAGE_DIR=admin_images
    BUZZER_AUDIO=buzzer.wav
    MOTION_THRESHOLD=25
    MOTION_PIXELS=5000
    COOLDOWN_SECONDS=3
    VIDEO_SOURCE=0
    ```

### 5.2 Install Dependencies

-
    ```bash
    pip install -r requirements.txt
    ```
> **Note**: This project requires **Linux**. The `face_recognition` library does not support Windows.

### 5.3 Add Admin Selfies
Create an `admin_images/` directory in your project and
place one or more face images (`.jpg`, `.png`) into it. The script will automatically encode every image found.
You can also set a custom directory for your admin images in the `.env` file.

### 5.4 Run

-
    ```bash
    python main.py
    ```
* Two windows titled "Live" and "Feature Heatmap Overlay" will show detection boxes and heatmap overlays.
* Press **q** to quit.
---
***&copy; 2025 Elyar KordKatool & Bahar Naderlou***
