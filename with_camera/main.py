import cv2
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace with your desired model (e.g., yolov8s.pt)

# Background subtractor for motion detection
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=25)

def detect_motion(frame):
    results = model.predict(source=frame, conf=0.5, stream=True)  # Stream results for real-time performance
    is_moving = False

    for result in results:
        for box in result.boxes:
            # YOLOv8 box attributes
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class ID

            # Check if detected object is a person (class 0 for 'person')
            if cls == 0:
                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Define ROI (upper half of the bounding box)
                roi = frame[y1:y1 + (y2 - y1) // 2, x1:x2]

                # Detect motion within the ROI
                mask = bg_subtractor.apply(roi)
                motion = cv2.countNonZero(mask)

                # Threshold for waving motion
                if motion > 3000:  # Adjust threshold based on environment
                    is_moving = True
                    cv2.putText(frame, "Motion Detected", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame, is_moving


# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Enable RGB stream
pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert RealSense frame to numpy array
        frame = np.asanyarray(color_frame.get_data())

        # Process frame for waving detection
        frame, is_waving = detect_motion(frame)
        cv2.imshow("Motion Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
