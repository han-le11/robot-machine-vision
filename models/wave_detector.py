import cv2
import time
from ultralytics import YOLO


class WaveDetector:
    def __init__(self, model_path='yolov8n-pose.pt', wave_threshold=20, wave_persistence_time=1):
        """
        Initialize the WaveDetector class.

        Args:
            model_path (str): Path to the YOLO model.
            wave_threshold (int): Threshold to consider hand movement as waving.
            wave_persistence_time (int): Time in seconds to continue showing "Waving: YES" after waving.
        """
        self.model = YOLO(model_path)
        self.wrist_positions = {"left": None, "right": None}  # Store wrist positions
        self.WAVE_THRESHOLD = wave_threshold
        self.WAVE_PERSISTENCE_TIME = wave_persistence_time
        self.waving_timestamp = 0  # Last detected waving timestamp

    def detect_wave(self, frame):
        """
        Detect waving motion in the given frame.

        Args:
            frame (numpy.ndarray): The current video frame.

        Returns:
            tuple: The processed frame and a boolean indicating if waving was detected.
        """
        results = self.model.predict(source=frame, conf=0.5, stream=True)
        is_waving = False
        current_time = time.time()

        for result in results:
            # Iterate over detected keypoints for each person in the frame
            for keypoints in result.keypoints.xy:
                # Draw all keypoints
                for x, y in keypoints:
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

                # Ensure there are enough keypoints to detect wrists
                if len(keypoints) > 10:
                    left_wrist = keypoints[9]
                    right_wrist = keypoints[10]
                    left_x, left_y = map(int, left_wrist)
                    right_x, right_y = map(int, right_wrist)

                    # Draw the wrists in different colors
                    cv2.circle(frame, (left_x, left_y), 5, (0, 0, 255), -1)  # Red for left wrist
                    cv2.circle(frame, (right_x, right_y), 5, (255, 0, 0), -1)  # Blue for right wrist

                    # If we have previous positions, compute movement
                    if self.wrist_positions["left"] and self.wrist_positions["right"]:
                        left_movement = abs(left_x - self.wrist_positions["left"][0])
                        right_movement = abs(right_x - self.wrist_positions["right"][0])
                        if left_movement > self.WAVE_THRESHOLD or right_movement > self.WAVE_THRESHOLD:
                            is_waving = True
                            self.waving_timestamp = current_time

                    # Update stored wrist positions
                    self.wrist_positions["left"] = (left_x, left_y)
                    self.wrist_positions["right"] = (right_x, right_y)

        # Maintain waving status if within persistence time
        if current_time - self.waving_timestamp < self.WAVE_PERSISTENCE_TIME:
            is_waving = True

        # Display the waving status on the frame
        status_text = "Waving: YES" if is_waving else "Waving: NO"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8,
                    (0, 255, 0) if is_waving else (0, 0, 255), 2)

        return frame, is_waving

    def process_camera(self, camera_index=0):
        """
        Process video input from a webcam.

        Args:
            camera_index (int): Index of the camera (default is 0 for the built-in webcam).
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame, is_waving = self.detect_wave(frame)
            cv2.imshow("Waving Detection", frame)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
