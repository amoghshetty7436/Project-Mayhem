from ultralytics import YOLO
import cv2
import numpy as np
import torch
import os


# Load the YOLOv8 model for detecting suitcases and backpacks (COCO pretrained model)
model_coco = YOLO('yolov8n.pt')  # Use the yolov8n.pt for COCO detection (pretrained)

# Load the YOLOv8 model for detecting tags (your custom model)
model_tag = YOLO('runs/detect/train10/weights/best.pt')

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Move both models to the correct device
model_coco.to(device)
model_tag.to(device)

# Class names for COCO model
coco_class_names = model_coco.names  # Get class names from COCO model

# Class name for tag detection
class_name_tag = "Tag"

# Function to process image
def process_image(image_path):
    # Read the image
    frame = cv2.imread(image_path)

    # Perform inference on the image using the COCO model (suitcases, backpacks detection)
    results_coco = model_coco.predict(source=frame, imgsz=640, conf=0.5, verbose=False)
    detections_coco = results_coco[0].boxes.data.cpu().numpy()  # Get detections

    # Process each detection for suitcases or backpacks
    for detection in detections_coco:
        x1, y1, x2, y2, confidence, cls_id = detection
        class_name = coco_class_names[int(cls_id)]  # Get class name from class ID

        # If detected class is "suitcase" or "backpack", crop the image and run tag detection
        if class_name in ['suitcase', 'backpack']:
            cropped_frame = frame[int(y1):int(y2), int(x1):int(x2)]  # Crop the detected object
            
            # Perform tag detection on the cropped image
            results_tag = model_tag.predict(source=cropped_frame, imgsz=640, conf=0.5, verbose=False)
            detections_tag = results_tag[0].boxes.data.cpu().numpy()

            # Check if a tag is detected
            tag_detected = False
            for detection_tag in z:
                _, _, _, _, confidence, cls_id = detection_tag
                if confidence >= 0.5:
                    tag_detected = True
                    label = f"{class_name_tag} {confidence:.2f}"

                    # Draw bounding box on the cropped image
                    color = (0, 255, 0)  # Green for "Tag"
                    cv2.rectangle(cropped_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(cropped_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    break
            
            # If no tag is detected, print the image or return it
            if not tag_detected:
                print(f"No tag detected in {class_name} region.")
                cv2.imwrite(f'no_tag_detected_{image_path}', cropped_frame)
            else:
                # Optionally, save the image where tag is detected
                cv2.imwrite(f'tag_detected_{image_path}', cropped_frame)

    # Show the processed image with bounding boxes (optional)
    cv2.imshow('Processed Image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path):
    # Ensure the directory to save cropped images exists
    if not os.path.exists('cropped'):
        os.makedirs('cropped')

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # VideoWriter to save the output video (optional)
    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference on the current frame using COCO model (detect suitcases and backpacks)
        results_coco = model_coco.predict(source=frame, imgsz=640, conf=0.5, verbose=False)
        detections_coco = results_coco[0].boxes.data.cpu().numpy()

        for detection in detections_coco:
            x1, y1, x2, y2, confidence, cls_id = detection
            class_name = coco_class_names[int(cls_id)]

            # If the class is "suitcase" or "backpack", crop and process the region
            if class_name in ['suitcase', 'backpack']:
                cropped_frame = frame[int(y1):int(y2), int(x1):int(x2)]  # Crop the region
                crop_path = f'cropped/frame{frame_count}_{class_name}.jpg'

                # Save the cropped frame
                cv2.imwrite(crop_path, cropped_frame)

                # Perform tag detection
                results_tag = model_tag.predict(source=cropped_frame, imgsz=640, conf=0.5, verbose=False)
                detections_tag = results_tag[0].boxes.data.cpu().numpy()

                tag_detected = False
                for detection_tag in detections_tag:
                    _, _, _, _, tag_confidence, tag_cls_id = detection_tag
                    if tag_confidence >= 0.5:
                        tag_detected = True
                        cv2.putText(frame, "Tag Detected", (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        break

                if tag_detected:
                    print(f"Tag detected in {class_name} region at frame {frame_count}")
                else:
                    print(f"No tag detected in {class_name} region at frame {frame_count}")

        # Write the frame with annotations to the output video
        out.write(frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def identify_tag(image_path):
    frame = cv2.imread(image_path)
    results_tag = model_tag.predict(source=frame, imgsz=640, conf=0.5, verbose=False)
    detections_tag = results_tag[0].boxes.data.cpu().numpy()

    tag_detected = False
    for detection_tag in detections_tag:
        x1, y1, x2, y2, confidence, cls_id = detection_tag
        if confidence >= 0.5:
            tag_detected = True
            label = f"{class_name_tag} {confidence:.2f}"
            color = (0, 255, 0)  # Green for "Tag"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if tag_detected:
        print("Tag detected in the image.")
    else:
        print("No tag detected in the image.")

    # Save or display the result
    output_path = f'tag_result_{os.path.basename(image_path)}'
    cv2.imwrite(output_path, frame)

# Process an image
# process_image('input_image.jpg')  # Provide the path to your image

# identify_tag("input_image.jpg")

# Process a video
process_video('img/testing2.mp4')  # Uncomment to process a video