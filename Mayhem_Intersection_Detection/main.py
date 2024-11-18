from ultralytics import YOLO
import cv2
import supervision as sv
import numpy as np

model = YOLO('yolov8n.pt')

# IoU Calculation Function
def calculate_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box1[1])
    iou = interArea / float(box1Area + box2Area - interArea)
    return iou

# Settings
LINE_START = sv.Point(0 + 50, 1500)
LINE_END = sv.Point(3840 - 50, 1500)
TARGET_VIDEO_PATH = "output_video.mp4"
SELECTED_CLASS_IDS = [24, 28]
SOURCE_VIDEO_PATH = "input_video.mp4"

# Create BYTETracker instance
byte_tracker = sv.ByteTrack(
    track_activation_threshold=0.4,
    lost_track_buffer=50,
    minimum_matching_threshold=0.8,
    frame_rate=30,
    minimum_consecutive_frames=3
)
byte_tracker.reset()

video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

box_annotator = sv.BoxAnnotator(thickness=4)
label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1.5, text_color=sv.Color.BLACK)
trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)
line_zone_annotator = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=2)

def callback(frame: np.ndarray, index: int) -> np.ndarray:
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
    detections = byte_tracker.update_with_detections(detections)

    # Check for overlaps between bags
    overlapping_bags = []
    for i, box1 in enumerate(detections.xyxy):
        for j, box2 in enumerate(detections.xyxy):
            if i >= j:  # Avoid duplicate comparisons (i.e., comparing same pair in reverse)
                continue
            # Calculate IoU between boxes
            iou = calculate_iou(box1, box2)
            if iou > 0.05:  # Adjust the threshold based on your needs
                overlapping_bags.append((detections.tracker_id[i], detections.tracker_id[j], iou))
                # Annotate the frame with red boxes and "OVERLAP" text
                x1, y1, x2, y2 = box1
                print(f"Detected overlap: IoU={iou} between tracker {detections.tracker_id[i]} and {detections.tracker_id[j]}")
                # Draw red box and overlap text for the first box
                cv2.putText(frame, "OVERLAP", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                x1, y1, x2, y2 = box2
                # Draw red box and overlap text for the second box
                cv2.putText(frame, "OVERLAP", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    if overlapping_bags:
        print(f"Frame {index} - Overlapping bags detected: {overlapping_bags}")

    # Perform the default supervision annotation
    labels = [
        f"#{tracker_id} {confidence:0.2f}"
        for confidence, tracker_id in zip(detections.confidence, detections.tracker_id)
    ]

    # Copy the frame for applying annotations
    annotated_frame = frame.copy()

    # Apply Supervision annotations
    annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    line_zone.trigger(detections)

    # Apply line zone annotation at the end
    annotated_frame = line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

    # Ensure OpenCV annotations (red boxes and overlap text) are kept by adding them on top of Supervision annotations
    return annotated_frame

sv.process_video(
    source_path=SOURCE_VIDEO_PATH,
    target_path=TARGET_VIDEO_PATH,
    callback=callback
)
