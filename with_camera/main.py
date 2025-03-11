import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

class WavingDetector:
    def __init__(self, model_path='yolov8n.pt', motion_threshold=500):
        """
        Initialize the WavingDetector class.

        Args:
            model_path (str): Path to the YOLO model.
            motion_threshold (int): Threshold for detecting waving motion.
        """
        self.model = YOLO(model_path)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=25)
        self.motion_threshold = motion_threshold

    def detect_and_wave(self, frame):
        """
        Detect a waving motion in the given frame.

        Args:
            frame (numpy.ndarray): A single video frame.

        Returns:
            tuple: Processed frame and a boolean indicating if waving was detected.
        """
        results = self.model.predict(source=frame, conf=0.5, stream=True)
        is_waving = False

        for result in results:
            for box in result.boxes:
                # YOLOv8 box attributes
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                # Check if detected object is a person (class 0 for 'person')
                cls = int(box.cls[0].item())  # Class ID
                if cls == 0:
                    # Draw the bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Define ROI (upper half of the bounding box)
                    roi = frame[y1:y1 + (y2 - y1) // 2, x1:x2]

                    # Detect motion within the ROI
                    mask = self.bg_subtractor.apply(roi)
                    motion = cv2.countNonZero(mask)

                    # Threshold for waving motion
                    if motion > self.motion_threshold:
                        is_waving = True
                        cv2.putText(frame, "Human Detected", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame, is_waving

    def process_webcam(self):
        """
        Process input from the MacBook's built-in webcam.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame for waving detection
            frame, is_waving = self.detect_and_wave(frame)
            cv2.imshow("Waving Detection", frame)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def process_camera(self):
        """
        Process input from an Intel RealSense camera.
        """
        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        # Start streaming
        profile = pipeline.start(config)

        try:
            while True:
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                # Convert images to numpy arrays
                frame = np.asanyarray(color_frame.get_data())

                # Process the frame for waving detection
                frame, is_waving = self.detect_and_wave(frame)
                cv2.imshow("Waving Detection", frame)

                # Press 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            # Stop streaming
            pipeline.stop()
            cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    detector = WavingDetector()
    detector.process_webcam()
