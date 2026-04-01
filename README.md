# 🛩️ Barcode Detection & Tracking: CPU vs. GPU Performance Analysis

This project was developed as part of the **Master's Degree in Autonomous Systems** at **ISEP**. It focuses on real-time barcode detection and multiple object tracking (MOT), benchmarking performance across different hardware configurations and deep learning architectures.

---

## 💻 Hardware Specifications
*The following machines were used to benchmark inference times, FPS stability, and power efficiency:*

| Device | CPU | GPU | VRAM |
| :--- | :--- | :--- | :--- |
| **PC 1** | Intel Core i7-13650HX | NVIDIA RTX 4060 | 8GB |
| **PC 2** | Intel Core i7-13700K | NVIDIA RTX 3070 | 8GB |
| **PC 3** | Intel Core i5-8300H | NVIDIA GTX 1050 | 4GB |

---

## 🧠 Model Selection and Benchmarking
Before implementing the final tracking pipeline, several architectures were tested within a **Webots & ROS2** simulation environment to find the optimal balance between precision and speed.

* **Models Tested:** YOLOv8, Faster R-CNN, SSD (Single Shot Detector), and RetinaNet.
* **Key Metrics:** mAP (Mean Average Precision), IoU (Intersection over Union), Inference Time (ms), and Energy Consumption.
* **Final Choice:** **YOLOv8** was selected as the primary detector due to its superior feature extraction and real-time performance on edge-case hardware.

---

## 🚀 Key Features

* **Real-Time Detection:** Powered by a custom-trained YOLOv8 model optimized for high-speed barcode symbology recognition.
* **Enhanced Multi-Object Tracking (MOT):** Implementation of the **ByteTrack** algorithm, fine-tuned to maintain object IDs across multiple items simultaneously (e.g., Pasta and Yogurt packaging).
* **Robustness to Occlusion:** Configured to handle brief overlaps, light reflections on plastic surfaces, and motion blur using a 60-frame `track_buffer`.
* **Performance Monitoring:** Comparative analysis of FPS and latency between CPU and GPU (CUDA) execution.

---

## 🛠️ Tracker Optimization (ByteTrack)
To ensure bounding box stability in a logistics/retail context, the following hyperparameters were applied:
- `track_high_thresh`: 0.5 (High-confidence detection threshold)
- `track_low_thresh`: 0.1 (Recovery of low-confidence detections during blur)
- `match_thresh`: 0.9 (Strict spatial association)
- `fuse_score`: Enabled (Better motion-detection integration)

---

## 📁 Project Structure
* **`Code/`**: Python scripts for real-time inference, tracking, and benchmarking.
* **`Weights/`**: Custom-trained YOLOv8 model ('.pt' file) and ByteTrack config.
* **`Project Documentation/`**: Full technical paper with detailed methodology and experimental results.
  
---
## 📸 In Action
![System Demo](Media/Barcode_Tracking_GIF.gif)
