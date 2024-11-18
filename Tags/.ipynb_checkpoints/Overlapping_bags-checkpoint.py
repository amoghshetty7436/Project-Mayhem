from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2

model = YOLO('yolov8m.pt')

# Settings
LINE_START = sv.Point(0 + 50, 1500)
LINE_END = sv.Point(3840 - 50, 1500)
TARGET_VIDEO_PATH = "img/result_bytetracker.mp4"
SELECTED_CLASS_IDS = [24, 28]  # Select class IDs for suitcase and backpack
SOURCE_VIDEO_PATH = "img/carousel.mp4"  # Replace with your source video path

# Create BYTETracker instance
byte_tracker = sv.ByteTrack(
    track_activation_threshold=0.4,
    lost_track_buffer=50,
    minimum_matching_threshold=0.8,
    frame_rate=30,
    minimum_consecutive_frames=3
)

byte_tracker.reset()

# Create VideoInfo instance
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

# Create frame generator
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

# Create LineZone instance
line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

# Create instance of BoxAnnotator, LabelAnnotator, and TraceAnnotator
box_annotator = sv.BoxAnnotator(thickness=4)
label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1.5, text_color=sv.Color.BLACK)
trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)

# Create LineZoneAnnotator instance
line_zone_annotator = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=2)

# Dictionary to keep track of overlaps
overlap_counter = {}
# Function to calculate Intersection over Union (IoU)
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1.xmin, box1.ymin, box1.xmax, box1.ymax
    x1_prime, y1_prime, x2_prime, y2_prime = box2.xmin, box2.ymin, box2.xmax, box2.ymax
    inter_x1 = max(x1, x1_prime)
    inter_y1 = max(y1, y1_prime)
    inter_x2 = min(x2, x2_prime)
    inter_y2 = min(y2, y2_prime)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_prime - x1_prime) * (y2_prime - y1_prime)
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

# Updated callback function
def callback(frame: np.ndarray, index: int) -> np.ndarray:
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
    detections = byte_tracker.update_with_detections(detections)
    labels = [
        f"#{tracker_id} {confidence:0.2f}"
        for confidence, tracker_id in zip(detections.confidence, detections.tracker_id)
    ]
    
    annotated_frame = frame.copy()
    annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    line_zone.trigger(detections)
    
    # Check for overlapping bounding boxes
    current_overlaps = []
    for i, det1 in enumerate(detections):
        for j, det2 in enumerate(detections):
            if i >= j:
                continue
            
            iou = compute_iou(det1, det2)
            if iou > 0.05:
                pair = tuple(sorted([det1.tracker_id, det2.tracker_id]))
                current_overlaps.append(pair)
                if pair in overlap_counter:
                    overlap_counter[pair] += 1
                else:
                    overlap_counter[pair] = 1
                if overlap_counter[pair] >= 3:
                    print(f"Box {pair[0]} is overlapping with Box {pair[1]}")
            else:
                if pair in overlap_counter:
                    overlap_counter[pair] = 0

    for pair in list(overlap_counter):
        if pair not in current_overlaps:
            del overlap_counter[pair]

    return line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)
