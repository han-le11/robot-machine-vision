import time
import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

class LandmarkerDetector:
    def __init__(self, model_path="mediapipe_models/landmarker_model.task"):
        self.mp = mp
        self.gesture_result = None
        self.landmark_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self._landmark_result_callback
        )
        self.landmarker = HandLandmarker.create_from_options(self.landmark_options)

    def _landmark_result_callback(self, result, output_image, timestamp_ms):
        """Callback function to process detection results."""
        # if not result.hand_landmarks:  # Check if any hands are detected
        #     print('No hands detected')
        #     return

        for hand_landmarks in result.hand_landmarks:
            # Check if the middle finger is extended while others are bent
            if self.is_middle_finger_extended_only(hand_landmarks):
                print("Middle finger is detected")
            else:
                print("Not detected")

    @staticmethod
    def is_middle_finger_extended_only(landmarks):
        """Check if the middle finger is extended and other fingers are bent."""
        # Indexes for the middle finger landmarks
        MCP = 9  # Middle finger's base
        PIP = 10  # Middle finger's proximal joint
        DIP = 11  # Middle finger's distal joint)
        TIP = 12  # Middle finger tip

        # Check if the landmarks are above their preceding joints
        return (
                landmarks[TIP].y < landmarks[DIP].y and
                landmarks[DIP].y < landmarks[PIP].y and
                landmarks[PIP].y < landmarks[MCP].y
        )

    def run(self):
        """Start video stream and process frames."""
        cap = cv2.VideoCapture(1)

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break

                # Convert frame to MediaPipe Image format
                mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame)

                # Use a monotonically increasing timestamp
                current_timestamp = int(time.time() * 1000)  # Current time in milliseconds

                # Process frame
                self.landmarker.detect_async(mp_image, current_timestamp)

                # Show the video feed
                cv2.imshow('Hand Tracking', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

# Usage
if __name__ == "__main__":
    detector = LandmarkerDetector("mediapipe_models/landmarker_model.task")
    detector.run()
