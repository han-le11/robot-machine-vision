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
        self.timestamp = 0  # Monotonic timestamp counter

        # Initialize MediaPipe Hands for skeleton tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

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

    def draw_hand_skeleton(self, frame, hand_landmarks):
        """Draw the hand skeleton using MediaPipe landmarks."""
        for landmarks in hand_landmarks:
            for connection in self.mp_hands.HAND_CONNECTIONS:
                start = landmarks.landmark[connection[0]]
                end = landmarks.landmark[connection[1]]

                # Convert to pixel coordinates
                h, w, _ = frame.shape
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))

                # Draw connections between hand joints in white
                cv2.line(frame, start_point, end_point, (255, 255, 255), 2)

            # Draw hand keypoints in red
            for landmark in landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 0, 128), -1)

    def run(self):
        """Start real-time gesture recognition with hand skeleton tracking."""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Convert frame to RGB format (MediaPipe expects RGB images)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect hand landmarks
            hand_results = self.hands.process(frame_rgb)

            # Draw hand skeleton if detected
            if hand_results.multi_hand_landmarks:
                self.draw_hand_skeleton(frame, hand_results.multi_hand_landmarks)

            # Convert to MediaPipe Image format for gesture recognition
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Send frame for recognition (async mode) with a manual timestamp
            self.timestamp += 30  # Increment timestamp (approx. 30ms per frame)
            self.recognizer.recognize_async(mp_image, self.timestamp)

            # Display detected gesture on the frame
            if self.gesture_result:
                cv2.putText(frame, f'Gesture: {self.gesture_result}', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)

            # Show the video feed
            cv2.imshow("Gesture Recognition with Hand Tracking", frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

# Run the gesture detector
if __name__ == "__main__":
    detector = GestureDetector("mediapipe_models/gesture_recognizer.task")
    detector.run()
