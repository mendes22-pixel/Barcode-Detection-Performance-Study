from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    model.train(
        data='Dataset/YOLOv8/data.yaml',
        epochs=20,
        imgsz=640,
        batch=4,
        device='cuda'  
    )


