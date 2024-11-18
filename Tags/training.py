import os
import cv2
from ultralytics import YOLO
import torch

model_tag = YOLO('runs/detect/train9/weights/best.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_tag.to(device)

class_name_tag = "Tag"

model_tag.train(data='data.yaml', epochs=50, imgsz=640)
