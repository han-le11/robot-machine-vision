import cv2
import mediapipe as mp
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from socketMessage import RobotSocketClient

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

class GestureDetector:
    def __init__(self, model_path="mediapipe_models/gesture_recognizer.task"):
        """Initialize the GestureDetector with the given model path."""
        self.model_path = model_path
        self.gesture_result = None  # Store detected gesture
        self.last_gesture = None
        self.timestamp = 0  # Monotonic timestamp counter

        self.robot_client = RobotSocketClient(host="192.168.125.1", port=5000)

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
            base_options=BaseOptions(model_asset_buffer=open(model_path, "rb").read()),
            running_mode=VisionRunningMode.LIVE_STREAM,  # Required for real-time video mode
            result_callback=result_callback  # Callback function for async processing
        )

        # Initialize the gesture recognizer
        self.recognizer = GestureRecognizer.create_from_options(self.options)
        # The camera index 0 is often the built-in webcam. You may need to change it if you have multiple cameras.
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cv2.namedWindow("Gesture Recognition with Hand Tracking", cv2.WINDOW_NORMAL)

    def detect_middle_finger(self, hand_landmarks) -> bool:
        """
        Detect if the middle finger is extended while other fingers are bent.

        Args:
            hand_landmarks: Hand landmarks detected by MediaPipe's Hands.

        Returns:
            bool: True if the middle finger gesture is detected, False otherwise.
        """
        # Finger landmark indices as defined by MediaPipe
        THUMB_TIP, THUMB_BASE = 4, 2
        INDEX_TIP, INDEX_BASE = 8, 6
        MIDDLE_TIP, MIDDLE_BASE = 12, 10
        RING_TIP, RING_BASE = 16, 14
        PINKY_TIP, PINKY_BASE = 20, 18

        # Helper function with logic to detect middle finger:
        # Middle finger's tip must be above its base, and other fingers' tips should be below their bases.
        def is_extended(tip, base):
            return tip.y < base.y

        middle_finger_extended = is_extended(hand_landmarks.landmark[MIDDLE_TIP],
                                             hand_landmarks.landmark[MIDDLE_BASE])
        other_fingers_bent = all(
            not is_extended(hand_landmarks.landmark[tip], hand_landmarks.landmark[base])
            for tip, base in [
                (THUMB_TIP, THUMB_BASE),
                (INDEX_TIP, INDEX_BASE),
                (RING_TIP, RING_BASE),
                (PINKY_TIP, PINKY_BASE),
            ]
        )
        return middle_finger_extended and other_fingers_bent

    def draw_hand_skeleton(self, frame, hand_landmarks) -> None:
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

            # Detect if the middle finger gesture is present
            if self.detect_middle_finger(landmarks):
                cv2.putText(frame, 'Detected: Middle finger', (0, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                # TODO: comment out the line below if not connected to server
                self.send_gesture_message("Middle_Finger")

    def send_gesture_message(self, gesture):
        """Send socket message only if gesture has changed."""
        if gesture and gesture.strip().lower() != "none":
            self.robot_client.send_message(gesture)
        else:
            print("Skipping invalid gesture:", gesture)

    def run(self):
        """Start real-time gesture recognition with hand tracking."""
        last_message_time = 0  # Last message timestamp
        if not self.cap.isOpened():
            print("Could not open camera. Check the camera connection.")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            def display_text(text):
                cv2.putText(frame, f'Detected: {self.gesture_result}', (0, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

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

            current_time = time.time()
            # Display detected gesture on the screen
            if self.gesture_result:
                display_text(f'Gesture: {self.gesture_result}')
                if current_time - last_message_time >= 1.5:
                    print(self.gesture_result + " lasts at least 1.5s. Sending message to robot.")
                    # TODO: comment out the line below if not connected to server
                    self.send_gesture_message(self.gesture_result)
                    last_message_time = current_time
            # Show the video feed
            cv2.imshow("Gesture Recognition with Hand Tracking", frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self.robot_client.close()

# Run the gesture detector
if __name__ == "__main__":
    detector = GestureDetector("mediapipe_models/gesture_recognizer.task")
    detector.run()
