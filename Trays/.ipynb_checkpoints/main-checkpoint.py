from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLOv8 model (choose model based on your preference, e.g., yolov8S.pt)
model = YOLO('yolov8n.pt')

# Define classes (make sure they match your data.yaml file)
classes = {0: "Bag", 1: "Bagless"}  # Update with your actual class mapping

# Function to process image
def process_image(image_path):
    # Read the image
    frame = cv2.imread(image_path)

    # Perform inference on the image
    results = model.predict(source=frame, imgsz=640, conf=0.2, verbose=False)
    detections = results[0].boxes.data.cpu().numpy()  # Get detections

    # Draw bounding boxes and labels
    for detection in detections:
        x1, y1, x2, y2, confidence, cls_id = detection
        class_id = int(cls_id)
        label = f"{classes.get(class_id, 'Unknown')} {confidence:.2f}"

        # Draw bounding box and label
        color = (0, 255, 0) if class_id == 0 else (0, 0, 255)  # Green for Bag, Red for Bagless
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    cv2.waitKey(0)  # Wait for a key press to close the image window
    cv2.destroyAllWindows()

    # Optionally, save the processed image
    cv2.imwrite('output_image.jpg', frame)

# Function to process video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # VideoWriter to save the output video (optional)
    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Process video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference on the current frame
        results = model.predict(source=frame, imgsz=640, conf=0.5, verbose=False)
        detections = results[0].boxes.data.cpu().numpy()  # Get detections

        # Draw bounding boxes on the frame
        for detection in detections:
            x1, y1, x2, y2, confidence, cls_id = detection
            class_id = int(cls_id)
            label = f"{classes.get(class_id, 'Unknown')} {confidence:.2f}"

            # Draw bounding box and label
            color = (0, 255, 0) if class_id == 0 else (0, 0, 255)  # Green for Bag, Red for Bagless
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


        # Write the frame to the output video
        out.write(frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Process an image
# process_image('input_image.jpg')  # Provide the path to your image

# Process a video
#process_video('input_video.mp4')  # Uncomment to process a video
