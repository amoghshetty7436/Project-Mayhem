from ultralytics import YOLO
import cv2
import numpy as np
import torch

# Load the YOLOv8 model (choose model based on your preference, e.g., yolov8S.pt)
model = YOLO('yolov8n.pt')

model.train(data = '/home/eclipse/pt-gpu/Trays/data.yaml', epochs = 50, imgsz = 640)

