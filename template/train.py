from ultralytics import YOLO

model = YOLO("../yolov11models/yolo11n-seg.pt")
reults = model.train(data="D:/Projects/AIProjects/DermaIQ/woundSegmentation/data/wound-seg.yaml", epochs=100, imgsz=640, workers=0)