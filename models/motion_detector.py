import cv2
import numpy as np
import os
from ultralytics import YOLO

class MotionDetector:
    def __init__(self, model_path='yolov8n.pt', motion_threshold=500):
        """
        Initialize the WavingDetector class.

        Args:
            model_path (str): Path to the YOLO model.
            motion_threshold (int): Threshold for detecting motion.
        """
        self.model = YOLO(model_path)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=25)
        self.motion_threshold = motion_threshold

    def detect_motions(self, frame: np.ndarray):
        """
        Detect a motion in the given frame.

        Args:
            frame (numpy.ndarray): A single video frame.

        Returns:
            tuple: Processed frame and a boolean indicating if motion/movement was detected.
        """
        results = self.model.predict(source=frame, conf=0.5, stream=True)
        is_moving = False

        for result in results:
            for box in result.boxes:
                # Box attributes
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
                    mask = self.bg_subtractor.apply(roi)
                    motion = cv2.countNonZero(mask)

                    # Threshold for motion detection
                    if motion > self.motion_threshold:
                        is_moving = True
                        cv2.putText(frame, "Motion Detected", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame, is_moving

    def process_videos(self, folder_path: str):
        """
        Process all video files in the specified folder.

        Args:
            folder_path (str): Path to the folder containing test video files.
        """
        for video_file in os.listdir(folder_path):
            if video_file.endswith((".mp4", ".avi", ".mov")):
                video_path = os.path.join(folder_path, video_file)
                print(f"Processing video: {video_file}")

                # Open video file
                cap = cv2.VideoCapture(video_path)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break  # Break when the video ends

                    # Process the frame for moving detection
                    frame, is_waving = self.detect_motions(frame)
                    cv2.imshow("Motion Detection", frame)

                    # Press 'q' to skip to the next video
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                cap.release()
                print(f"Finished processing: {video_file}")

        cv2.destroyAllWindows()

