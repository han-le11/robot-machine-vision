import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Use the Nano model for speed; maybe switch to yolov8s.pt or yolov8m.pt for more accuracy

# Background subtractor for motion detection
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=25)

def detect_and_wave(frame):
    """
    Detect a human and detect if they are waving.
    :param frame:
    :return:
    """
    results = model.predict(source=frame, conf=0.5, stream=True)  # Stream results for real-time performance
    is_waving = False

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
                    is_waving = True
                    cv2.putText(frame, "Waving Detected", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame, is_waving


# Video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame, is_waving = detect_and_wave(frame)
    cv2.imshow("Waving Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
