from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os 
import torch
import time
from PIL import Image
import numpy as np
import matplotlib.patches as mpatches
from codecarbon import EmissionsTracker

# ---------------- FUNCTIONS ---------------- #
# Convert YOLO format (txt) to (x_min, y_min, x_max, y_max)
def yolo_to_xyxy(x_center, y_center, width, height, img_w, img_h):
    x_min = int((x_center - width / 2) * img_w)
    y_min = int((y_center - height / 2) * img_h)
    x_max = int((x_center + width / 2) * img_w)
    y_max = int((y_center + height / 2) * img_h)
    return [x_min, y_min, x_max, y_max]

# Calculate Intersection of Union (IoU)
def box_iou(box1, box2):
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])

    inter_width = max(0, x_max - x_min)
    inter_height = max(0, y_max - y_min)
    inter_area = inter_width * inter_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

# ---------------- MAIN CODE ---------------- #

device = 'cuda'  # or 'cpu'
model = YOLO('Yolov8_best.pt')

# Folders
test_folder = 'Dataset/YOLOv8/test/images'
labels_folder = 'Dataset/YOLOv8/test/labels'

inference_times = []
detections = []
iou_list = []

# Empty lists for the Average Precision (AP) values
tp_list, fp_list, conf_list, total_gt = [], [], [], 0

# Initialize the tracker
tracker = EmissionsTracker(project_name=f"Yolov8_{device}")
tracker.start()

# Loop through the images
for filename in os.listdir(test_folder)[:100]:
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(test_folder, filename)
        label_path = os.path.join(labels_folder, filename.replace('.jpg', '.txt').replace('.png', '.txt'))

        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        # YOLO Detections 
        start_time = time.time()
        results = model(image_path, device=device)[0]
        if device == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        inference_times.append((end_time - start_time) * 1000)

        # Predictions
        pred_bboxes = []
        pred_conf = []
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf <= 0.75:   
                continue    
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            pred_bboxes.append([int(x1), int(y1), int(x2), int(y2)])
            pred_conf.append(conf)

        # Ground Truth 
        gt_bboxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    cls, x_c, y_c, bw, bh = map(float, line.split())
                    gt_bboxes.append(yolo_to_xyxy(x_c, y_c, bw, bh, w, h))
        total_gt += len(gt_bboxes)

        # IoU calculation (Ground Truth vs Prediction)
        img_ious = []
        used_gt = set()
        for pi, pred in enumerate(pred_bboxes):
            best_iou, best_gt = 0, -1
            for gi, gt in enumerate(gt_bboxes):
                iou = box_iou(gt, pred)
                if iou > best_iou:
                    best_iou, best_gt = iou, gi

            if best_iou >= 0.75 and best_gt not in used_gt:  
                tp_list.append(1)
                fp_list.append(0)
                used_gt.add(best_gt)
            else:
                tp_list.append(0)
                fp_list.append(1)

            conf_list.append(pred_conf[pi])
            img_ious.append(best_iou)

        avg_iou_img = np.mean(img_ious) if img_ious else 0
        iou_list.append(avg_iou_img)

        detections.append({
            'image': image,
            'results': results,
            'filename': filename,
            'gt': gt_bboxes
        })
    
emissions: float = tracker.stop()

# ---------------- CALCULATE AP ---------------- #

# Ordenar por confiança
sorted_idx = np.argsort(-np.array(conf_list))
tp = np.array(tp_list)[sorted_idx]
fp = np.array(fp_list)[sorted_idx]

tp_cum = np.cumsum(tp)
fp_cum = np.cumsum(fp)

recall = tp_cum / total_gt if total_gt > 0 else np.zeros_like(tp_cum)
precision = tp_cum / (tp_cum + fp_cum + 1e-6)
AP = np.trapz(precision, recall)

# ---------------- PLOTS ---------------- #

for det in detections[:5]:  
    image = det['image']
    results = det['results']
    gt_bboxes = det['gt']

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # YOLO Predictions (RED)
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    # Ground Truth (GREEN)
    for gt in gt_bboxes:
        gx1, gy1, gx2, gy2 = gt
        rect = patches.Rectangle((gx1, gy1), gx2-gx1, gy2-gy1,
                                 linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)

    # Subtitles/Legend
    gt_patch = mpatches.Patch(color='green', label='Ground Truth')
    pred_patch = mpatches.Patch(color='red', label='Prediction')
    ax.legend(handles=[gt_patch, pred_patch], loc='upper right')

    plt.show()

# ---------------- RESULTS ---------------- #

# Average IoU & Average AP
mean_iou = np.mean(iou_list) * 100
print(f"Yolov8 --> IoU médio: {mean_iou:.2f}%")
print(f"Yolov8 --> AP médio: {AP:.4f}")

# Average inference time
media_inferencia = sum(inference_times) / len(inference_times)
print(f'Yolov8 --> Tempo médio de inferência ({device}): {media_inferencia:.2f}ms')
