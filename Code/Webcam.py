from ultralytics import YOLO
import cv2
import time
import torch

# Choose device (CPU or GPU --> CUDA)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device used: {device}")

# Load the fine tuned detection model
model = YOLO("Yolov8_best.pt")

# Class name (only barcode)
class_names = ["Barcode"]

# Webcam dimensions
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Tracking model (bytetrack.yaml | botsort.yaml)
    results = model.track(
        frame, 
        persist=True, 
        conf=0.25, 
        imgsz=1280,
        device=device, 
        tracker="tracker_cfg.yaml",
        verbose=False
        )

    for r in results:
        boxes = r.boxes
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().tolist()

            for box, track_id in zip(boxes, ids):
                # Bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = f"{class_names[cls_id]} {conf:.2f}" #ID {track_id}

                # Draw 
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    cv2.putText(frame, f"{device.upper()} FPS: {fps:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show webcam window
    cv2.imshow("Real-Time Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
