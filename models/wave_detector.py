import cv2
import time
from ultralytics import YOLO
# ADAPTED FROM AKSU'S CODE

# Load YOLOv8n-pose model to track joints and keypoints
model = YOLO('yolov8n-pose.pt')  # Use the Nano model for speed and testing; S provides way more accuracy with roughly double the latency

wrist_positions = {"left": None, "right": None}  # Store wrist positions to track waving motion
WAVE_THRESHOLD = 20  # Threshold to consider moving hand as waving
waving_timestamp = 0  # Store the last time the user waved
WAVE_PERSISTENCE_TIME = 1  # Time in seconds to show "Waving: YES" after the user stops waving
waving_active = False  # Flag to track if the "Waving: YES" status is still active

def detect_wave(frame):
    global waving_timestamp, waving_active

    results = model.predict(source=frame, conf=0.5, stream=True)
    is_waving = False
    current_time = time.time()

    for result in results:
        for keypoints in result.keypoints.xy:
            for x, y in keypoints:  # Extract each keypoint's coordinates
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # Draw keypoints

            if len(keypoints) > 10: # Ensure wrists are detected
                left_wrist = keypoints[9]
                right_wrist = keypoints[10]

                # Convert to integer coordinates
                left_x, left_y = map(int, left_wrist)
                right_x, right_y = map(int, right_wrist)

                cv2.circle(frame, (left_x, left_y,), 5, (0, 0, 255), -1) # Draw left wrist with red color
                cv2.circle(frame, (right_x, right_y), 5, (255, 0, 0), -1) # Draw right wrist with blue color

                # Check for waving motion
                if wrist_positions["left"] and wrist_positions["right"]:
                    left_movement = abs(left_x - wrist_positions["left"][0])
                    right_movement = abs(right_x - wrist_positions["right"][0])

                    if left_movement > WAVE_THRESHOLD or right_movement > WAVE_THRESHOLD:
                        is_waving = True
                        waving_timestamp = current_time

                # Update stored wrist positions
                wrist_positions["left"] = (left_x, left_y)
                wrist_positions["right"] = (right_x, right_y)

    # Check if waving status must be still displayed
    if current_time - waving_timestamp < WAVE_PERSISTENCE_TIME:
        is_waving = True

    # Display waving status
    status_text = "Waving: YES" if is_waving else "Waving: NO"
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0) if is_waving else (0, 0, 255), 2)


    return frame, is_waving


# Video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame, is_waving = detect_wave(frame)
    cv2.imshow("Waving Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()