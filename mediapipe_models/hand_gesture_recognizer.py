import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
# from mediapipe.tasks.BaseOptions import BaseOptions

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

class GestureDetector:
    def __init__(self, model_path="mediapipe_models/gesture_recognizer.task"):
        """Initialize the GestureDetector with the given model."""
        self.model_path = model_path
        self.gesture_result = None  # Store detected gesture

        # Define callback function
        def result_callback(result, output_image, timestamp_ms):
            """Callback function to handle gesture recognition results."""
            if result.gestures:
                self.gesture_result = result.gestures[0][0].category_name  # Most confident gesture
            else:
                self.gesture_result = None  # No gesture detected

        # Set up gesture recognizer options
        self.options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,  # Required for real-time mode
            result_callback=result_callback  # Callback function for async processing
        )

        # Initialize the gesture recognizer
        self.recognizer = GestureRecognizer.create_from_options(self.options)
        self.cap = cv2.VideoCapture(1)  # Initialize the camera

    def recognize_gesture(self, frame):
        """Recognize hand gestures from a given video frame."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        recognition_result = self.recognizer.recognize(mp_image)

        # Check if gestures are detected
        if recognition_result.gestures:
            gesture_name = recognition_result.gestures[0][0].category_name  # Most confident gesture
            return gesture_name
        return None

    def run(self):
        """Start real-time gesture recognition from the camera."""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Recognize gesture
            gesture = self.recognize_gesture(frame)

            # Display the recognized gesture on the frame
            if gesture:
                cv2.putText(frame, f'Gesture: {gesture}', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Show the video feed
            cv2.imshow("Gesture Recognition", frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

# Run the gesture detector
if __name__ == "__main__":
    detector = GestureDetector("mediapipe_models/gesture_recognizer.task")
    detector.run()
