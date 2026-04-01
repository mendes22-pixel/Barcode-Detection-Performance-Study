# 🛩️ Barcode Detection & Tracking: CPU vs. GPU Performance Analysis
---
This project was developed during our **Master's Degree in Autonomous Systems** at ISEP. It focuses on real-time barcode detection and tracking using state-of-the-art Deep Learning models, benchmarking the performance across different hardware configurations.
---

## 💻 Hardware Specifications

The following machines were used to benchmark the inference times and FPS stability:

| Device | CPU | GPU | VRAM |
| :--- | :--- | :--- | :--- |
| **PC 1** | Intel Core i7-13650HX | NVIDIA RTX 4060 | 8GB |
| **PC 2** | Intel Core i7-13700K | NVIDIA RTX 3070 | 8GB |
| **PC 3** | Intel Core i5-8300H | NVIDIA GTX 1050 | 4GB |
---

##  Key Features
- **Simulation Environment:** Webots & ROS2.
- **Models Tested:** YOLOv8, Faster R-CNN, Single Shot Detector (SSD) and RetinaNet.
- **Metrics:** mAP, IoU, Inference Time, and Energy Consumption.

* **Real-Time Detection:** Powered by a custom-trained YOLOv8 model optimized for barcode symbologies.
* **Enhanced Multi-Object Tracking (MOT):** Implementation of the **ByteTrack** algorithm, fine-tuned for stability in logistics environments.
* **Robustness to Occlusion:** Designed to maintain tracking during brief overlaps, reflections, or motion blur (tested with retail packaging like pasta and yogurt).
* **Hardware Benchmarking:** Live monitoring of FPS and inference latency across different compute architectures.

---

## 🛠️ Tracker Configuration


## 📊 Performance Summary

As detailed in the technical paper included in this repository, the benchmarks show:

* **GPU (NVIDIA RTX):** Achieved smooth, high-frequency playback (~[COLOCA_AQUI_TEU_FPS] FPS), ideal for robotics and drone-based scanning.
* **CPU (Intel/AMD):** Performance peaked at ~[COLOCA_AQUI_TEU_FPS] FPS, highlighting the necessity of hardware acceleration for industrial-grade real-time systems.

---







